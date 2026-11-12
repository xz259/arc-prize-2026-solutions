"""stage2_v1 'rebaseobs' part (2026-10-03 22:25; user ~21:50, 21:59, 22:12): the position rebase keeps only the latest
full interaction(s) reaching back to the protected tail, and each session keeps processed images only for what such a
rebase can restore. The rebased engine request therefore prefills a much shorter context.

Production (session_service 7ce0eaef):
  _rebase_window  the earliest observation (else assistant) start w <= the pair-guarded protected-tail start with
                  P + (end - w) <= target, the policy basis (30,748 at the 36K tier): the rebuilt frame is the prefix plus
                  about a basis of verbatim recent history, its images re-sent from the stored rendered parts;
  _record_part    stores every image-bearing append's rendered input (processed pixel tensors, ~3 MB per 512x512 board)
                  and prunes those ending before len - 2 x context (NLL probe off, policy not oldest_recompute).
Here:
  _rebase_window  w = the latest observation start at or before hend = max(P, end - protect_recent_tokens). The rebuilt
                  frame is [prefix][that full observation][everything after it]: the protected tail, extended back to
                  the start of the interaction it begins in (a cut inside an assistant response keeps that response's
                  observation; a cut inside an observation restores the whole observation). Every v6 opener, resumptions
                  included, registers an observation, so this adds at most one turn to the tail. The window stays
                  contiguous, so production's rebase bookkeeping (one shift, kind 'interaction') is unchanged, and the
                  live managed suffix, which belongs to the latest observation, is always inside it. Production's choice
                  is used instead when w is the prefix itself or the window would exceed the target.
  _record_part    production's pruning condition with the horizon at that same observation start for the current
                  length. The horizon only moves forward within a frame (production resets parts at a rebase), so every
                  part a later rebase needs is kept, and the fallback window always starts at or after it.
Half-context mode (2026-10-03 23:02, the user's fallback ~22:37: "keep approximately half of the context length equivalent
recent full interactions and rebase using it"): ARC3_S2_REBASE_KEEP=<tokens> (0 = the tail rule above). The window then
starts at the first observation at or after end - KEEP, so it holds at most KEEP tokens of whole interactions; when no
observation starts between end - KEEP and the protected tail, the tail rule applies.
Long-interaction edge case (2026-10-04 00:33, user ~00:15 "maybe we can append back the observation turn?"): when the
window from the chosen observation exceeds the target (that one interaction alone is too long), the rebuilt frame is
[prefix][that observation][its most recent whole assistant units that fit, each with its tool turn][everything after],
i.e. two kept pieces after the prefix; the unit boundary is the earliest assistant start u <= the pair-guarded tail
start with P + len(observation) + (end - u) within the limit: P + KEEP in half-context mode when that interaction alone
exceeds KEEP, else the rebase target (the basis, then the current resident count, as production's fallback order).
When nothing fits, the ordinary path keeps the whole interaction if it fits the target.
The bookkeeping is production's _rebase with a piecewise map (_rebase_two_piece); production's own choice (an
assistant boundary without the observation) remains the last resort.
Receipts: STATS (windows, fallbacks by reason, two-piece rebases, the last window, parts pruned / kept at most)."""
import os
import time

KEEP = int(os.environ.get('ARC3_S2_REBASE_KEEP', '0'))
PINS = {'session_service': '7ce0eaef072e45354cfae8476719313a401387ea379b3fe9c3ca9c1ab40193e8'}
STATS = dict(windows=0, fallbacks=0, fallback_reasons={}, two_piece=0, last=None, parts_recorded=0, parts_pruned=0,
             parts_kept_max=0)


def window_start(session, end):
    """(w, hend): the latest registered observation start at or before hend = max(P, end - protect); (0, 0) before any
    observation is registered."""
    obs = session.history_registry.observations
    if not obs:
        return 0, 0
    P = obs[0].start
    protect = int(getattr(session.history_policy, 'protect_recent_tokens', 0) or 0)
    hend = max(P, end - protect)
    w = P
    for o in obs:                                   # registry order is strictly increasing (production checks it)
        if o.start > hend:
            break
        w = o.start
    if KEEP > protect:                              # half-context mode: whole interactions back to about end - KEEP
        lo = max(P, end - KEEP)
        first = next((o.start for o in obs if o.start >= lo), None)
        if first is not None and first <= hend:
            w = first
    return w, hend


def _pair_guarded_tail(session, end, P):
    """Production's protected-tail start (_rebase_window): hend moved back to the assistant turn owning it."""
    protect = int(getattr(session.history_policy, 'protect_recent_tokens', 0) or 0)
    hend = max(P, end - protect)
    owner = prev = None
    for kind, a, b in session.segments:
        if kind == 'assistant':
            prev = a
        if a < hend < b or (a == hend and kind != 'assistant'):
            owner = prev
            break
        if a > hend:
            break
    return max(P, owner) if owner is not None and owner < hend else hend


def edge_plan(session):
    """(o_s, o_e, u, target, tail) when the window from the chosen observation exceeds the target and a two-piece frame
    fits; None otherwise (the ordinary path)."""
    end = len(session.token_ids)
    obs = session.history_registry.observations
    if not obs:
        return None
    P = obs[0].start
    resident_now = session.residency.resident_tokens
    basis = int(getattr(session.history_policy, 'target_basis_tokens', 0) or 0)
    protect = int(getattr(session.history_policy, 'protect_recent_tokens', 0) or 0)
    w, _ = window_start(session, end)
    if w <= P:
        return None
    limits = []
    if KEEP > protect and end - w > KEEP:         # half-context mode: the chosen interaction alone exceeds KEEP
        limits.append(P + KEEP)
    if P + (end - w) > (basis or resident_now):   # it exceeds the rebase target
        limits += [basis, resident_now] if basis else [resident_now]
    if not limits:
        return None
    o_e = next((int(x[2]) for x in session.segments if x[0] == 'observation' and int(x[1]) == w), None)
    if o_e is None:
        return None
    nxt = next((o.start for o in obs if o.start > w), end)
    tail = _pair_guarded_tail(session, end, P)
    units = [int(x[1]) for x in session.segments if x[0] == 'assistant' and o_e <= int(x[1]) < nxt]
    for limit in limits:
        fit = [u for u in units if u <= tail and P + (o_e - w) + (end - u) <= limit]
        if fit:
            return w, o_e, min(fit), limit, tail
    return None


async def _rebase_two_piece(self, mod, session, turn, local_history_tokens, plan):
    """Production's _rebase (session_service 7ce0eaef) with the kept pieces [(0, P), (o_s, o_e), (u, end)] and a piecewise
    position map; otherwise line by line the same bookkeeping, engine restart and receipt."""
    started = time.monotonic()
    if session.open_image is not None:
        raise RuntimeError("rebase with an open image span")
    open_call = getattr(session, "open_tool_call", None)
    old_len = len(session.token_ids); resident_now = session.residency.resident_tokens
    obs = session.history_registry.observations; P = obs[0].start
    o_s, o_e, u, target, tail = plan
    keep = [(0, P), (o_s, o_e), (u, old_len)]
    prompt, resident, n_images = mod.build_probe_prompt(session.token_ids, keep, old_len,
                                                        [tuple(x) for x in session.images], list(session.parts), ())
    obs_len = o_e - o_s
    new_len = P + obs_len + old_len - u
    if resident != new_len or len(prompt["prompt_token_ids"]) != new_len:
        raise RuntimeError(f"two-piece rebase prompt has {resident} tokens, pieces expect {new_len}")

    def m(q):                                   # old position -> new position (None when dropped)
        if q < P:
            return q
        if o_s <= q < o_e:
            return P + q - o_s
        return P + obs_len + q - u if q >= u else None

    def mspan(a, b):                            # a span wholly inside one kept piece, else None
        if b <= P:
            return (a, b)
        if o_s <= a and b <= o_e:
            return (P + a - o_s, P + b - o_s)
        return (P + obs_len + a - u, P + obs_len + b - u) if a >= u else None

    ev = session.residency.evicted
    restored = sum(max(0, min(b, o_e) - max(a, o_s)) + max(0, min(b, old_len) - max(a, u)) for a, b in ev)
    segments = []
    for sk, a, b in session.segments:
        if a < P:
            segments.append([sk, a, min(b, P)])
        elif sk == 'observation' and a == o_s:
            segments.append([sk, P, P + min(b, o_e) - o_s])
        elif a >= u:
            segments.append([sk, P + obs_len + a - u, P + obs_len + b - u])
    images = sorted([list(mspan(a, b)) for a, b in session.images if mspan(a, b) is not None])
    image_kinds = {mspan(a, b)[0]: session.image_kinds.get(a, "other") for a, b in session.images if mspan(a, b) is not None}
    structural = [m(q) for q in session.structural if m(q) is not None]
    session.open_tool_call = m(open_call) if open_call is not None else None
    owner_id = next(o.observation_id for o in obs if o.start == o_s)
    kept_obs = [o for o in obs if o.start >= u]
    dropped = [o.observation_id for o in obs if o.start < o_s]
    new_obs = [mod.ObservationStart(owner_id, P)] + [mod.ObservationStart(o.observation_id, m(o.start)) for o in kept_obs]
    if new_obs[0].start != P or any(x.start >= y.start for x, y in zip(new_obs, new_obs[1:])):
        raise RuntimeError(f"two-piece rebase observation registry would not start at the prefix or not increase: {new_obs[:3]}")
    reg = session.history_registry
    reg.observations = tuple(new_obs); reg.registered_through = new_len; reg.revision += 1
    sreg = session.suffix_registry
    for oid, (a, b) in list(sreg.live.items()):
        sp = mspan(a, b)
        if sp is None:
            raise RuntimeError(f"live managed suffix {oid} [{a},{b}) lies outside the rebase pieces")
        sreg.live[oid] = sp
    for oid, rec in sreg.records.items():
        span = rec.get("logical_token_range")
        if span is not None:
            rec["logical_token_range"] = mspan(span[0], span[1])
    session.token_ids = list(prompt["prompt_token_ids"])
    session.residency = mod.ResidentLedger(new_len)
    session.segments = segments; session.images = images; session.structural = structural; session.image_kinds = image_kinds
    session.parts = [(0, new_len, prompt)] if prompt.get("type") == "multimodal" else []
    session.frequency_epoch_start = new_len; session.frequency_epoch_reset_pending = False
    session.eviction_revision = 0
    session.rebases += 1
    turn.prompt = prompt; turn.recompute = True
    await self.backend.restart_request(session)
    info = {"rebase": session.rebases, "from_logical_tokens": old_len, "to_logical_tokens": new_len,
            "prefix_tokens": P, "window_start": u, "boundary": "observation+units", "observation_piece": [o_s, o_e],
            "dropped_unit_tokens": u - o_e, "protected_tail_start": tail,
            "resident_before": resident_now, "target_tokens": target, "target_fallback": False,
            "restored_tokens": restored, "net_dropped_tokens": resident_now + local_history_tokens - new_len,
            "local_history_tokens": local_history_tokens, "stale_tool_call": open_call,
            "dropped_observation_ids": dropped, "partial_observation_id": None, "images": n_images,
            "restart_seconds": time.monotonic() - started, "engine_request_id": session.engine_request_id}
    session.rebase_info = info
    self.audit("rebase", session, operation_id=session.current_operation, **info)
    STATS['two_piece'] += 1
    STATS['last'] = dict(two_piece=[o_s, o_e, u], end=old_len, new_len=new_len)
    _save()
    return info


def _save():
    try:
        from . import install
        install.save()
    except Exception:                               # receipts are diagnostics; never fail a turn for them
        pass


def patch_session_service(mod):
    cls = mod.SessionManager
    production = cls.__dict__['_rebase_window']
    production = production.__func__ if isinstance(production, staticmethod) else production

    def _rebase_window(session, target):
        end = len(session.token_ids)
        P = session.history_registry.observations[0].start
        w, hend = window_start(session, end)
        reason = None
        if w <= P:
            reason = 'no observation after the prefix'
        elif P + (end - w) > target:
            reason = 'window exceeds the target'
        if reason is None:
            STATS['windows'] += 1
            STATS['last'] = dict(end=end, prefix=P, window_start=w, protected_from=hend, window_tokens=end - w,
                                 target=target)
            _save()
            return w, 'interaction', hend
        STATS['fallbacks'] += 1
        STATS['fallback_reasons'][reason] = STATS['fallback_reasons'].get(reason, 0) + 1
        _save()
        return production(session, target)

    def _record_part(self, session, start, length, prompt):
        session.parts.append((start, length, prompt))
        STATS['parts_recorded'] += 1
        if not mod.nll_probe_rate() and session.history_policy.policy != 'oldest_recompute':
            horizon, _ = window_start(session, len(session.token_ids))
            n = len(session.parts)
            session.parts = [q for q in session.parts if q[0] + q[1] > horizon]
            STATS['parts_pruned'] += n - len(session.parts)
            STATS['parts_kept_max'] = max(STATS['parts_kept_max'], len(session.parts))

    production_rebase = cls._rebase

    async def _rebase(self, session, turn, local_history_tokens=0):
        plan = edge_plan(session)
        if plan is None:
            return await production_rebase(self, session, turn, local_history_tokens)
        return await _rebase_two_piece(self, mod, session, turn, local_history_tokens, plan)

    cls._rebase_window = staticmethod(_rebase_window)
    cls._record_part = _record_part
    cls._rebase = _rebase


PATCHERS = {'session_service': patch_session_service}
