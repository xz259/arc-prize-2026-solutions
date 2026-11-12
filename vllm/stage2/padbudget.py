"""stage2_v1 'padbudget' part (2026-10-06; run 381's engine death at 02:13:13, the user's decision ~03:50 and Fable's
review): the session service counts the session's QSA group padding as occupied context.

The engine's turn check (engine_eviction.validate_turn_capacity) requires resident + max_tokens + group_padding <= the
QSA descriptor page table (21 pages x 3,104 = 65,184 at the 64,684 context; every tier sits 500 tokens under a page
multiple since the 47,056 crash of 09-19), while the session service clamps max_tokens to budget - (resident after
eviction + input) without the padding. Padding is the evicted tokens of 4-token groups that keep their descriptor
(mostly beside formatting tokens, which frequency selection never evicts); g046 reached 618 and the engine died.
Two layers here (the engine-side abort of only that session, deadlinefix.py, is the last resort):
  1. SessionManager._run_turn: on entry, the session's (frozen) policy is replaced by a copy whose resident_token_budget
     is the session's original budget minus its current padding P, computed from session.residency.evicted with the
     engine's formula (deleted - 4 x whole groups). Every later read of the budget in the turn (the frequency-snapshot
     condition, the planner's trigger and capacity need, the 413 check, the max_tokens clamp, the rebase re-admission)
     sees it, so evictions come P tokens earlier and the clamp leaves room for the padding. The eviction quantum
     (fraction x target_basis_tokens) is unchanged.
  2. VllmBackend.turn: after this turn's eviction is committed and its input appended, max_tokens is re-clamped to
     base - P_after - resident (at least 1), so padding added by this turn's own eviction cannot reach the check
     either; a turn cut here ends as a length pause, and the next append's reduced budget triggers a real eviction.
The receipts' resident_token_budget shows the reduced budget. Errors leave production's behaviour (counted)."""
import dataclasses

PINS = {'session_service': '7ce0eaef072e45354cfae8476719313a401387ea379b3fe9c3ca9c1ab40193e8',
        'persistent_endpoint': 'ceead0163e99ffcdf3f376affee74a597a343ef06781a737e0b09bf9a56b0001'}
STATS = dict(turns=0, turns_with_padding=0, max_padding=0, reclamped=0, reclamp_tokens=0, floor_one=0, errors=0,
             last_error=None)


_MEMO = {}          # id(evicted) -> (evicted, padding); 2026-10-15 (final profile: per-turn cost grew with session age)
_MEMO_MAX = 64      # >= sessions; a full cache is cleared (old ledgers' tuples are dropped with it)


def group_padding(evicted):
    """engine_eviction.LogicalState.group_padding over a ledger's evicted [a, b) ranges: evicted tokens in 4-token QSA
    groups that keep their descriptor. Memoized on the evicted object: ResidentLedger replaces its evicted tuple at every
    commit (normalize_ranges) and nothing mutates it in place, so one count per eviction serves every turn and poolpress
    call until the next commit (the cached reference keeps the id from being reused)."""
    hit = _MEMO.get(id(evicted))
    if hit is not None and hit[0] is evicted:
        return hit[1]
    deleted = whole = 0
    for a, b in evicted:
        a, b = int(a), int(b)
        deleted += b - a
        whole += max(0, b // 4 - (a + 3) // 4)
    pad = deleted - 4 * whole
    if isinstance(evicted, tuple):          # immutable only: a list could change under the same id
        if len(_MEMO) >= _MEMO_MAX:
            _MEMO.clear()
        _MEMO[id(evicted)] = (evicted, pad)
    return pad


def _base(session):
    base = getattr(session, '_arc3_padbudget_base', None)
    if base is None:
        base = session._arc3_padbudget_base = int(session.history_policy.resident_token_budget)
    return base


def patch_session_service(mod):
    cls = mod.SessionManager
    original = cls._run_turn

    async def _run_turn(self, session, payload, *, initial):
        try:
            policy = getattr(session, 'history_policy', None)
            residency = getattr(session, 'residency', None)
            if policy is not None and residency is not None:
                base = _base(session)
                pad = group_padding(residency.evicted)
                want = max(1, base - pad)
                if policy.resident_token_budget != want:
                    session.history_policy = dataclasses.replace(policy, resident_token_budget=want)
                STATS['turns'] += 1
                STATS['turns_with_padding'] += pad > 0
                STATS['max_padding'] = max(STATS['max_padding'], pad)
        except Exception as exc:  # noqa: BLE001  production's budget stands
            STATS['errors'] += 1
            STATS['last_error'] = repr(exc)[:300]
        return await original(self, session, payload, initial=initial)

    _run_turn.__wrapped__ = original
    cls._run_turn = _run_turn


def patch_persistent_endpoint(mod):
    cls = mod.VllmBackend
    original = cls.turn

    async def turn(self, session, turn_input, *, initial):
        try:
            base = getattr(session, '_arc3_padbudget_base', None)
            sampling = getattr(turn_input, 'sampling', None)
            if base is not None and isinstance(sampling, dict) and type(sampling.get('max_tokens')) is int:
                residency = session.residency
                limit = base - group_padding(residency.evicted) - residency.resident_tokens
                if sampling['max_tokens'] > limit:
                    new = max(1, limit)
                    STATS['reclamped'] += 1
                    STATS['reclamp_tokens'] += sampling['max_tokens'] - new
                    STATS['floor_one'] += limit < 1
                    sampling['max_tokens'] = new
        except Exception as exc:  # noqa: BLE001  the session service's clamp stands
            STATS['errors'] += 1
            STATS['last_error'] = repr(exc)[:300]
        return await original(self, session, turn_input, initial=initial)

    turn.__wrapped__ = original
    cls.turn = turn


PATCHERS = {'session_service': patch_session_service, 'persistent_endpoint': patch_persistent_endpoint}
