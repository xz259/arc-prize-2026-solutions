"""stage2_v1 'deadlinefix' part: an append that reaches the engine core after its persistent session was aborted is
answered as aborted instead of killing the engine (the engine_patch deadline race deferred in GAP_PLAN, 2026-09-30).

At a contract deadline the client aborts every session. Aborts also take the engine core's fast path (aborts_queue,
processed right after a step), so an append already queued for the same session can be handled after the session is
gone. engine_patch then sees a persistent request with no existing session and an empty prompt (an empty
continuation) and raises PersistentStateError("A new persistent session must have a nonempty initial prompt") inside
the engine core's input loop, which is fatal (g4prof2 09:24:44, g4d 18:22:04: the traceback runs from
_process_input_queue through EngineCore.add_request into engine_patch's Scheduler.add_request).
A genuine new session always has a nonempty prompt, so exactly this case (that message, the request id not in the
scheduler) is answered with FinishReason.ABORT to its client, the outcome the abort already gave the session. Nothing
was mutated before the raise (engine_patch validates before the stock add). Every other error still propagates.

2026-10-06 (run 381, 02:13:13): an append that fails engine_eviction.validate_turn_capacity ("Turn exceeds the QSA
descriptor page-table capacity" / "... the original model position limit") raised the same way inside the input loop
and killed the engine, so every game's session failed. The session service clamps a turn to the resident budget, but
the engine's check also counts the session's group padding (tokens evicted from 4-token QSA groups that keep their
descriptor because a formatting token in the group is never evicted); g046 reached 618 against the 500 tokens the
21-page table has above the 64,684 context. Now only that session ends: its paused request is finished through the stock
abort path (scheduler.finish_requests, which frees its KV) and its client is answered with FinishReason.ABORT; the
harness restarts that game's level on a new session. The check raises before the turn is appended (engine_eviction's
append validates before old_append), so no other session or the engine's state is touched.
Fable's final review (E1, 2026-10-06 ~04:20): every other per-session error raised in add_request was still fatal
("Missing or unhealthy eviction session state", "Logical and resident output ledgers diverged" from state.paused,
engine_patch's other PersistentStateErrors). Any EngineEvictionError or PersistentStateError from add_request now ends
only that request the same way (its id aborted if the scheduler holds it, ABORT sent to its client); other exception
classes still propagate.
v9b (2026-10-27, Claude main session 099768dc; the user: no fatal condition in the Kaggle run, even one that should not
trigger; IE's inventory CAPGUARD_REVIEW_ie_2026-10-27.md (b) 3): the per-session classes also include LookaheadError
(one_lookahead.validate_admission / validate_append, before the stock add or the append's mutation), ResidencyError (the
ledger's checks), SchedFastError (VERIFY-gated) and the plain RuntimeError that image_admission.require raises (matched by
the raising module, not the class). A raise after a partial mutation ends only that session the same way (the stock abort
frees it); before, the engine died."""
import logging

PINS = {'vllm.v1.engine.core': 'f4b1e07b6d91fac1549bc74a5c804e5a88793d663b5dc045c9e455db6550f5ad'}
STATS = dict(rejected=0, capacity_aborted=0, session_aborted=0, last_session_error=None)
MESSAGE = 'A new persistent session must have a nonempty initial prompt'
CAPACITY = ('Turn exceeds the QSA descriptor page-table capacity', 'Turn exceeds the original model position limit')
PER_SESSION = ('EngineEvictionError', 'PersistentStateError', 'LookaheadError', 'ResidencyError', 'SchedFastError')   # v9b: + 3
PER_SESSION_ORIGINS = ('image_admission.py',)   # v9b: plain RuntimeErrors raised inside these modules (image_admission.require)
logger = logging.getLogger('stage2_v1.deadlinefix')


def _per_session(exc):
    if type(exc).__name__ in PER_SESSION:
        return True
    if type(exc) is not RuntimeError:
        return False
    tb = exc.__traceback__
    while tb is not None and tb.tb_next is not None:
        tb = tb.tb_next
    return tb is not None and tb.tb_frame.f_code.co_filename.endswith(PER_SESSION_ORIGINS)


def patch_core(mod):
    cls = mod.EngineCore
    original = cls.add_request

    def add_request(self, request, request_wave=0):
        try:
            return original(self, request, request_wave)
        except Exception as exc:
            if (type(exc).__name__ == 'PersistentStateError' and str(exc) == MESSAGE
                    and request.request_id not in self.scheduler.requests
                    and hasattr(self, '_send_abort_outputs_to_client')):
                STATS['rejected'] += 1
                logger.warning('stage2 deadlinefix: append %s for a persistent session that no longer exists '
                               '(aborted at a deadline) answered as aborted', request.request_id)
                self._send_abort_outputs_to_client([request.request_id], request.client_index)
                return None
            if _per_session(exc) and hasattr(self, '_send_abort_outputs_to_client'):
                if type(exc).__name__ == 'EngineEvictionError' and str(exc) in CAPACITY:
                    STATS['capacity_aborted'] += 1
                    logger.warning('stage2 deadlinefix: append %s failed the turn-capacity check (%s); that session is '
                                   'aborted and the engine continues (capacity aborts so far: %d)', request.request_id,
                                   exc, STATS['capacity_aborted'])
                else:
                    STATS['session_aborted'] += 1
                    STATS['last_session_error'] = ('%s: %s' % (type(exc).__name__, exc))[:300]
                    logger.warning('stage2 deadlinefix: request %s raised %s (%s) in add_request; that session is '
                                   'aborted and the engine continues (other session aborts so far: %d)',
                                   request.request_id, type(exc).__name__, exc, STATS['session_aborted'])
                if request.request_id in self.scheduler.requests:
                    self.abort_requests([request.request_id])
                self._send_abort_outputs_to_client([request.request_id], request.client_index)
                return None
            raise

    add_request.__wrapped__ = original
    cls.add_request = add_request


PATCHERS = {'vllm.v1.engine.core': patch_core}
