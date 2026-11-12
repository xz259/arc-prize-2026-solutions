"""stage2_v1 'replayfx' part (2026-10-10, DFlash plan Phase 1, exact replay "A"; design in
dflash_20261006/EXPORTER_DESIGN_2026-10-10.md). Replay mode for the persistent API server: a request body may carry
`arc3_replay` = {output_token_ids, finish_reason, evict_ranges, evict_digest, appended_sha256} taken from the corpus
token log (the operation's pause and evict_commit). Never part of a production launch (ARC3_S2_REPLAY=1 required).

- VllmBackend.prepare: the record is bound to the operation's task (contextvar); after rendering, the new input ids must
  hash to the recorded appended_sha256 (sha256 of the JSON id list), else the operation is refused before mutation.
- history_eviction.plan_history_eviction: with a record, the decision is ledger.plan(recorded ranges); its digest must
  equal the recorded evict_commit digest (the ledger's digest covers logical tokens, prior evictions, ranges, survivors),
  else refused. No record ranges = the no-op plan. Production's planner (QSA frequency, agecap, hollow groups) is not run.
- VllmBackend.turn: the engine receives the rendered delta + output_token_ids[:-1] as input and samples exactly
  output_token_ids[-1] (max_tokens 1, allowed_token_ids: production's hard-close mechanism). The manager then receives
  the full recorded output and finish reason, so session.token_ids, segments, marks, the prefix digest (checked by the
  next append's expected_prefix_sha256) and the response match the original run.
Production's gameplay server profile (gameplay_profile 1) replaces VllmBackend.turn at startup with a copy of the
original source (its collector), which bypasses any turn wrapper installed at import; its install_targets is wrapped
here so the replay turn wrapper is re-applied outermost after it (a nested wrapper passes a forced turn through).
Receipt: STATS."""
import contextvars
import copy
import hashlib
import json
import os

PINS = {'persistent_endpoint': 'ceead0163e99ffcdf3f376affee74a597a343ef06781a737e0b09bf9a56b0001',
        'history_eviction': '88a5d9aef289c87b485104a1cf8d08701a96600198a70196ccd43cd7803c8fc0',
        'gameplay_server_profile': 'fb4d8f7048d4bab3d629850afd0e029f19cedf63c910709e9031382e28a126f6'}
ENABLED = os.environ.get('ARC3_S2_REPLAY', '0') == '1'
STATS = dict(enabled=ENABLED, operations=0, records=0, forced_tokens=0, forced_plans=0, evicted_tokens=0,
             append_mismatch=0, digest_mismatch=0, force_mismatch=0, last_error=None)
_REC = contextvars.ContextVar('arc3_replay_record', default=None)


def ids_sha256(ids):
    return hashlib.sha256(json.dumps([int(i) for i in ids], separators=(',', ':')).encode()).hexdigest()


def patch_persistent_endpoint(mod):
    cls = mod.VllmBackend
    orig_prepare = cls.prepare
    SessionError = mod.SessionError

    async def prepare(self, session, payload, *, initial):
        rec = payload.get('arc3_replay') if ENABLED else None
        if rec is not None and (not isinstance(rec, dict) or not rec.get('output_token_ids')):
            raise SessionError('arc3_replay requires output_token_ids', 400)
        _REC.set(rec)
        STATS['operations'] += 1
        turn = await orig_prepare(self, session, payload, initial=initial)
        if rec is not None:
            STATS['records'] += 1
            want = rec.get('appended_sha256')
            if want is not None and ids_sha256(turn.token_ids) != want:
                STATS['append_mismatch'] += 1
                STATS['last_error'] = 'appended ids differ (%s)' % session.session_id
                raise SessionError('replay: rendered input differs from the recorded appended_token_ids', 409)
        return turn

    cls.prepare = prepare
    wrap_turn(cls)


def wrap_turn(cls):
    """Install the replay turn wrapper over cls.turn as it is now (at import, and again after the gameplay profile)."""
    orig_turn = cls.turn

    async def turn(self, session, turn, *, initial):
        rec = _REC.get()
        if rec is None or getattr(turn, '_arc3_replay_forced', False):
            return await orig_turn(self, session, turn, initial=initial)
        out = [int(t) for t in rec['output_token_ids']]
        forced = copy.copy(turn)
        forced.prompt = dict(turn.prompt, prompt_token_ids=list(turn.prompt['prompt_token_ids']) + out[:-1])
        forced.sampling = dict(turn.sampling, max_tokens=1, allowed_token_ids=[out[-1]])
        forced._arc3_replay_forced = True
        result = await orig_turn(self, session, forced, initial=initial)
        if [int(t) for t in result['token_ids']] != [out[-1]]:
            STATS['force_mismatch'] += 1
            STATS['last_error'] = 'forced token not emitted (%s)' % session.session_id
            raise RuntimeError('replay: the engine did not emit the forced last token')
        STATS['forced_tokens'] += len(out)
        finish = rec.get('finish_reason') or result['finish_reason']
        return dict(result, token_ids=out, logprobs=[], finish_reason=finish,
                    stop_reason=result.get('stop_reason') if finish == 'stop' else None)

    turn.__wrapped__ = orig_turn
    cls.turn = turn
    STATS['turn_wraps'] = STATS.get('turn_wraps', 0) + 1


def patch_gameplay_profile(mod):
    original = mod.install_targets

    def install_targets(endpoint, service, protocol, *args, **kwargs):
        out = original(endpoint, service, protocol, *args, **kwargs)
        wrap_turn(endpoint.VllmBackend)              # the profile replaced turn: re-apply the replay wrapper outermost
        STATS['after_profile'] = True
        return out

    mod.install_targets = install_targets


def patch_history_eviction(mod):
    original = mod.plan_history_eviction

    def plan_history_eviction(policy, registry, ledger, **kw):
        rec = _REC.get()
        if rec is None:
            return original(policy, registry, ledger, **kw)
        ranges = tuple((int(a), int(b)) for a, b in (rec.get('evict_ranges') or ()))
        plan = ledger.plan(ranges)
        want = rec.get('evict_digest')
        if want and plan.digest != want:
            STATS['digest_mismatch'] += 1
            STATS['last_error'] = 'plan digest differs'
            raise mod.HistoryValidationError('replay: the recorded eviction plan does not reproduce on this ledger')
        if ranges:
            STATS['forced_plans'] += 1
            STATS['evicted_tokens'] += plan.removed_tokens
        receipt = dict(policy='replay', replay=True, total_removed_tokens=plan.removed_tokens,
                       history_removed_tokens=plan.removed_tokens, resident_tokens_after=plan.resident_after,
                       resident_tokens_after_input=plan.resident_after + int(kw.get('incoming_tokens') or 0))
        return mod.HistoryDecision(ranges, ranges, (), plan, receipt)

    mod.plan_history_eviction = plan_history_eviction


PATCHERS = {'persistent_endpoint': patch_persistent_endpoint, 'history_eviction': patch_history_eviction,
            'gameplay_server_profile': patch_gameplay_profile}
