"""stage2_v1 'schedfast' part: the engine core's KV-allocation demand wrapper at O(1) per call (GAP_PLAN item 4,
2026-09-30).

Production's engine_eviction wraps SingleTypeKVCacheManager.get_num_blocks_to_allocate and allocate_new_blocks for the
QSA compressed-cache managers (_demand_wrapper). Every call binds the full signature (inspect.Signature.bind) and reads
state.group_padding, which re-sums every evicted interval twice (LogicalState.deleted_groups and, through
deleted_tokens, ResidentLedger.resident_tokens). With about 24 requests, the compressed managers and two methods per
step, and evicted intervals accumulating over a run, Scheduler.schedule grew from 0.3 ms (decode waves) to 3-5 ms per
step in gameplay (g4fix steptimer), past the drafter's GPU tail that hides it in the single-batch engine.

- residency.ResidentLedger.resident_tokens and engine_eviction.LogicalState.deleted_groups are memoised per
  evicted-ranges tuple. normalize_ranges returns a fresh immutable tuple whenever the ranges change (construction,
  commit); logical_tokens stays live. The cached values are the originals' results by construction.
- engine_eviction._demand_wrapper keeps the semantics without Signature.bind: request_id, num_tokens and
  num_tokens_main_model are located by parameter position once; the padding is added to num_tokens and
  num_tokens_main_model where passed; the poisoned-state and capacity checks are unchanged.
ARC3_S2_SCHEDFAST_VERIFY=N: one memoised read or demand call in N is also computed the original way and compared (1 = all,
for validation; a large N keeps a live check in gameplay at negligible cost); a difference raises.
capguard (2026-10-27, Claude main session 099768dc; the user: no fatal conditions in the Kaggle run, even ones that should not
trigger; IE's suggestion): the demand wrapper's two raises (a poisoned eviction state; a QSA virtual group demand above the worker
page-table capacity) are fatal in the engine core's schedule(). A wrapper on Scheduler._reserve_prefill_lookahead, the last call
before allocate_slots on the running and the waiting path, computes the same demand first (min(computed + new + lookahead,
max_model_len) + group padding, and the full sequence for a waiting request under scheduler_reserve_full_isl) for every compressed
manager; a request that would raise is not scheduled (0 tokens: no allocation, no preemption) and is finished with FINISHED_ERROR by
vLLM's own per-request error path (grammar_compile_error_reqs in update_from_output). The engine lives; the client's session error
restarts that game's level (agent.py PROTOCOL_RESTARTS). An error inside the guard is counted and leaves the call as it was.
ARC3_S2_CAPGUARD=0 disables it; ARC3_S2_CAPGUARD_TEST_MARGIN=M (tests only) lowers the capacity by M for the first refusal."""
import functools
import inspect
import logging
import os
import time

PINS = {
    'engine_eviction': '1c49513acc9543b6f2392cdcf1f8c6eb172f84f343d853ff2464314f558ae2b5',
    'residency': '307187f8247069d1f7c7a89b9b276c6e0aa14b211b1f47471bb3a4fa8c6adbaf',
    'vllm.v1.core.sched.scheduler': 'c710f49e41e974e5b7b8f1cd2f5fb9722523f4da0165252e16351199ebd03124',   # stallguard's pin
}
VERIFY = int(os.environ.get('ARC3_S2_SCHEDFAST_VERIFY', '0') or 0)
CAPGUARD = os.environ.get('ARC3_S2_CAPGUARD', '1') != '0'
CAPGUARD_TEST_MARGIN = int(os.environ.get('ARC3_S2_CAPGUARD_TEST_MARGIN', '0') or 0)   # tests only: the first refusal
STATS = dict(demand_calls=0, demand_padded=0, ledger_recomputes=0, groups_recomputes=0, verified=0, capguard=CAPGUARD,
             capguard_refused=0, capguard_requests=0, capguard_poisoned=0, capguard_capacity=0, capguard_test=0,
             capguard_errors=0, capguard_last=None, capguard_last_error=None)   # refused: schedule attempts; requests: distinct ids
_SEEN = set()                  # refused request ids (IE should-fix 1: a refused id is refused again until its finish is processed)
_TICK = [0]


def _check():
    if not VERIFY:
        return False
    _TICK[0] += 1
    return _TICK[0] % VERIFY == 0


class SchedFastError(RuntimeError):
    pass


def patch_residency(mod):
    cls = mod.ResidentLedger
    original = cls.resident_tokens.fget

    def resident_tokens(self):
        ev = self.evicted
        c = self.__dict__.get('_s2_evicted_len')
        if c is None or c[0] is not ev:
            c = (ev, sum(end - start for start, end in ev))
            self.__dict__['_s2_evicted_len'] = c
            STATS['ledger_recomputes'] += 1
        value = self.logical_tokens - c[1]
        if _check():
            if value != original(self):
                raise SchedFastError('memoised resident_tokens differs')
            STATS['verified'] += 1
        return value

    cls.resident_tokens = property(resident_tokens)


def patch_engine_eviction(mod):
    cls = mod.LogicalState
    original_groups = cls.deleted_groups.fget

    def deleted_groups(self):
        ev = self.ledger.evicted
        c = self.__dict__.get('_s2_deleted_groups')
        if c is None or c[0] is not ev:
            c = (ev, original_groups(self))
            self.__dict__['_s2_deleted_groups'] = c
            STATS['groups_recomputes'] += 1
        if _check():
            if c[1] != original_groups(self):
                raise SchedFastError('memoised deleted_groups differs')
            STATS['verified'] += 1
        return c[1]

    cls.deleted_groups = property(deleted_groups)
    original_factory = mod._demand_wrapper
    error = mod.EngineEvictionError

    def _demand_wrapper(original):
        sig = inspect.signature(original)
        names = list(sig.parameters)                       # names[0] is self
        pos = {n: names.index(n) - 1 for n in ('request_id', 'num_tokens', 'num_tokens_main_model') if n in names}

        @functools.wraps(original)
        def wrapped(self, *args, **kwargs):
            if not getattr(self, '_arc3_has_compressed', False):
                return original(self, *args, **kwargs)
            STATS['demand_calls'] += 1
            rid = kwargs['request_id'] if 'request_id' in kwargs else args[pos['request_id']]
            state = self._arc3_eviction_states.get(rid)
            if state is None:
                return original(self, *args, **kwargs)
            if state.poisoned:
                raise error("Cannot schedule poisoned eviction state")
            padding = state.group_padding
            check = _check()
            if check:                                       # the original algorithm on the unadjusted arguments
                ref = sig.bind(self, *args, **kwargs)
                for key in ('num_tokens', 'num_tokens_main_model'):
                    if key in ref.arguments:
                        ref.arguments[key] += padding
            args, kwargs = list(args), dict(kwargs)
            for key in ('num_tokens', 'num_tokens_main_model'):
                if key in kwargs:
                    kwargs[key] += padding
                elif key in pos and pos[key] < len(args):
                    args[pos[key]] += padding
            num_tokens = kwargs['num_tokens'] if 'num_tokens' in kwargs else args[pos['num_tokens']]
            if check:
                mine = sig.bind(self, *args, **kwargs)
                if dict(mine.arguments) != dict(ref.arguments):
                    raise SchedFastError('adjusted demand arguments differ')
                STATS['verified'] += 1
            if num_tokens > self._arc3_table_capacity:
                raise error("QSA virtual group demand exceeds worker page-table capacity")
            STATS['demand_padded'] += 1
            return original(self, *args, **kwargs)

        return wrapped

    mod._demand_wrapper = _demand_wrapper
    mod._s2_original_demand_wrapper = original_factory


def _capguard_reason(sched, request, num_computed_tokens, num_new_tokens):
    """The demand wrapper's raise conditions, evaluated before allocate_slots: ('poisoned' | 'capacity', demand, capacity) or None."""
    from vllm.v1.request import RequestStatus
    mml = sched.max_model_len
    full = (getattr(sched, 'scheduler_reserve_full_isl', False)
            and request.status in (RequestStatus.WAITING, RequestStatus.PREEMPTED))
    rid = request.request_id
    for m in sched.kv_cache_manager.coordinator.single_type_managers:
        if not getattr(m, '_arc3_has_compressed', False):
            continue
        state = getattr(m, '_arc3_eviction_states', {}).get(rid)
        if state is None:
            continue
        if state.poisoned:
            return 'poisoned', 0, 0
        padding = state.group_padding
        need = min(min(num_computed_tokens, mml) + num_new_tokens + sched.num_lookahead_tokens, mml) + padding
        if full:
            need = max(need, min(request.num_tokens, mml) + padding)
        cap = m._arc3_table_capacity
        if CAPGUARD_TEST_MARGIN and not STATS['capguard_test']:
            cap -= CAPGUARD_TEST_MARGIN
        if need > cap:
            return 'capacity', need, cap
    return None


def patch_scheduler(sm):
    if not CAPGUARD:
        return
    cls = sm.Scheduler
    original = cls._reserve_prefill_lookahead

    @functools.wraps(original)
    def _reserve_prefill_lookahead(self, request, num_computed_tokens, num_new_tokens):
        n = original(self, request, num_computed_tokens, num_new_tokens)
        if n <= 0:
            return n
        try:
            why = _capguard_reason(self, request, num_computed_tokens, n)
        except Exception as exc:  # noqa: BLE001 - the guard must never raise
            STATS['capguard_errors'] += 1
            STATS['capguard_last_error'] = repr(exc)[:300]
            return n
        if why is None:
            return n
        STATS['capguard_refused'] += 1
        rid = request.request_id
        if rid not in _SEEN:
            if len(_SEEN) > 10000:
                _SEEN.clear()
            _SEEN.add(rid)
            STATS['capguard_requests'] += 1
            STATS['capguard_' + why[0]] += 1
        if CAPGUARD_TEST_MARGIN and why[0] == 'capacity' and not STATS['capguard_test']:
            STATS['capguard_test'] = 1
        STATS['capguard_last'] = dict(request=str(request.request_id)[:80], reason=why[0], demand=why[1], capacity=why[2],
                                      status=str(request.status), time=round(time.time(), 1))
        self.grammar_compile_error_reqs.add(request.request_id)
        try:
            logging.getLogger('arc3.capguard').warning(
                'capguard: request %s not scheduled (%s, demand %s, capacity %s); finished with FINISHED_ERROR',
                request.request_id, *why)
        except Exception:  # noqa: BLE001
            pass
        return 0

    cls._reserve_prefill_lookahead = _reserve_prefill_lookahead


PATCHERS = {'residency': patch_residency, 'engine_eviction': patch_engine_eviction, 'vllm.v1.core.sched.scheduler': patch_scheduler}
