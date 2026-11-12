"""stage2_v1 'rebase74' part (2026-10-23, Claude main session 099768dc; the user's rebase arm of 10-22: REBASE_PROPOSAL_2026-10-22.md,
Decisions 1-3 and the stale-mechanism audit, V67_DESIGN 263-267; built after "yes, start the build"). The server side of the rebase arm
(the 9-h 110-game A/B at 80K: refresh vs rebase). Without a client request (no arc3_rebase / arc3_suspend field) and with
ARC3_S2_REBASE_CAP unset it leaves production's behaviour as rebaseobs gives it; the admission count and the pool estimate differ only
while a session is suspended.

Forced rebase. An append with "arc3_rebase": true rebases on that append (production's _rebase through rebaseobs's window rules) at
any logical length: the client's 0.5M decision (the in-place rebase) and the resume of a parked game. Production's threshold
(ARC3_REBASE_AT) stays as the backstop. On every rebase turn the frequency snapshot is skipped: the turn's eviction plan is
acknowledged locally and the rebase discards it (production fetches the counters anyway; on a resumed session the request is gone).
The audit row frequency_snapshot_failed then carries 'rebase74: skipped on a rebase turn'.

Suspend / resume. An append whose body is {"arc3_suspend": true[, "expected_revision": n]} suspends a paused session: the engine
request is aborted (backend.close frees its KV pages; the collector and output future are dropped), the session object stays (token
ids, registries, segments, images; the rendered parts pruned to what the next rebase's window can need), state 'suspended'. The
next append (the resume) is admitted against the active-session limit, the session becomes 'paused' again and the append rebases: a
fresh engine request prefills the rebuilt frame. A continuation, an injection or a tail eviction is refused on a suspended session;
a resumed turn refused before submission (a 400, a 413) leaves the session suspended. close() releases a suspended session as any
other session (the harness closes every session at the run's end).

Admission and pool. A suspended session holds no engine request: start()'s admission limit counts only unsuspended sessions
(max_sessions reads the configured limit plus the suspended count), a resume is refused (429) while the unsuspended sessions fill the
configured limit, and poolpress's pool estimate skips suspended sessions.

The P-aware cap (IE Q2; the user: "the up to latest 60k including the system prompt"). ARC3_S2_REBASE_CAP=<tokens> (0 = off:
rebaseobs's ARC3_S2_REBASE_KEEP as configured): rebaseobs's half-context KEEP = CAP - P for the session's prefix P at every window
and parts decision, so the rebuilt frame, prefix included, holds at most CAP tokens of whole interactions from an observation start
(rebaseobs's two-piece frame and production's fallback only when the last interaction alone does not fit).

The C5 pin (sumpin; the stale-mechanism audit, item 1). While the session's summary pin [P, Q) is held, a rebase keeps the whole
first opener and drops only its generation prompt: the frame is [0, E) + [w, end), E = the generation prompt's <|im_start|> after the
opener (so the opener's <|im_end|> and newline stay and the window's observation follows as the next user message), w = rebaseobs's
window start for KEEP = CAP - E, E + (end - w) <= CAP (boundary 'pin_keep_first'); [P, Q) keeps its positions and its hold. With the
cap set, the initial append's rendered part is kept for the opener's images (session._arc3_pin_part; dropped when the pin is released
and at close). When that frame cannot be built the pin
is released at the rebase (in the new frame its positions would name other tokens) and the ordinary window applies.
v7.7 (2026-10-24; sumpin's board pin, apply_sumpin77.py): the same while either pin is held (the text pin [P, Q_k) or the board pin
[Q_k, Q_n), both inside the first opener): the keep-first frame keeps both at their positions, each with its own hold; when that
frame cannot be built, every held pin is released at the rebase (sumpin's counters per pin: pin_released_rebase,
board_pin_released_rebase).

The mode (the user 10-23: "a toggle between refresh and rebase modes"): ARC3_REBASE_MODE=refresh | rebase (default refresh), the
same variable the harness reads. refresh: the part is inert (the cap 0, production's threshold) and refuses arc3_rebase / arc3_suspend
(400), so a harness in the other mode fails loudly. rebase: ARC3_S2_REBASE_CAP defaults to 60,000 and production's threshold
ARC3_REBASE_AT, when unset, becomes the 600,000 backstop.

The finish guard (2026-10-23, V67_DESIGN 282 / 286; both modes): a turn ends only on a stop or a length finish. The persistent
collector takes any finish_reason as the turn boundary, so vLLM's retryable ERROR output (no tokens, finish_reason 'error': the engine's
answer to a multimodal processor-cache miss, its frontend shadow P0 having sent an image as a hash the engine cache P1 had evicted)
became a 0-token turn and the opaque 'Invalid emitted token count at pause'. VllmBackend.turn now raises an explicit error for any
other finish (counted in STATS error_finishes, the last in last_error_finish); the session fails as before and the harness falls back.

Receipts: STATS (forced appends / rebases, rebases, keep-first rebases, pin releases by reason, suspends, resumes, refusals, the
suspended count now / at most, parts pruned at suspend, snapshots skipped, error finishes, the last rebase); audit rows 'suspend' and
'resume'."""
import asyncio
import os
import time

PINS = {'session_service': '7ce0eaef072e45354cfae8476719313a401387ea379b3fe9c3ca9c1ab40193e8',
        'persistent_endpoint': 'ceead0163e99ffcdf3f376affee74a597a343ef06781a737e0b09bf9a56b0001'}
MODE = os.environ.get('ARC3_REBASE_MODE', 'refresh')
if MODE not in ('refresh', 'rebase'):
    raise RuntimeError('ARC3_REBASE_MODE must be refresh or rebase')
CAP = int(os.environ.get('ARC3_S2_REBASE_CAP', '60000' if MODE == 'rebase' else '0') or 0)
if CAP and MODE != 'rebase':
    raise RuntimeError('ARC3_S2_REBASE_CAP needs ARC3_REBASE_MODE=rebase')
BACKSTOP = '600000'                                      # the rebase mode's ARC3_REBASE_AT when unset (the user 10-23)
STATS = dict(mode=MODE, cap=CAP, backstop=None, refused_refresh_mode=0, forced_appends=0, forced_rebases=0, forced_not_due=0, rebases=0, rebase_tokens_max=0, keep_first=0,
             pin_released_rebase=0, keep_first_text_released=0, keep_first_refused={}, suspends=0, suspend_noop=0, suspend_refused=0, resumes=0,
             resume_refused=0, resume_restored=0, suspended_now=0, suspended_max=0, suspend_parts_pruned=0,
             suspend_parts_kept_max=0, suspend_seconds_max=0.0, snapshots_skipped=0, error_finishes=0, last_error_finish=None,
             errors=0, last_error=None, last=None)
FINISH_TEXT = ('rebase74 finish guard: the engine ended the turn with finish reason %r and %d token(s); only stop or length ends a turn '
               '(a retryable engine error, e.g. a multimodal processor-cache miss, is not a turn boundary)')
SKIP_TEXT = 'rebase74: skipped on a rebase turn (the plan is acknowledged locally; the rebase discards it)'
# v7.5 B3 (IE; the user 10-23, review Part 2): in refresh mode ARC3_S2_REBASE74_NORECORD=1 records no rendered image parts. Refresh mode
# never rebases (production's threshold max_logical - 24,576 = 2,072,576 under rope2m; the client refreshes at 0.5M logical), so
# rebaseobs's part bookkeeping (q74's refresh arm: 4,374 parts recorded, a window computation each) is never used there.
NORECORD = MODE == 'refresh' and os.environ.get('ARC3_S2_REBASE74_NORECORD', '0') == '1'
if NORECORD:
    STATS.update(norecord=True, parts_not_recorded=0, norecord_rebase_due=0)


def _save():
    try:
        from . import install
        install.save()
    except Exception:                                   # noqa: BLE001  receipts are diagnostics; never fail a turn for them
        pass


def _err(exc):
    STATS['errors'] += 1
    STATS['last_error'] = repr(exc)[:300]


def _n_suspended(sessions):
    return sum(1 for s in sessions if getattr(s, 'state', '') == 'suspended')


def _set_keep(rebaseobs, base):
    """rebaseobs's KEEP for a frame whose kept head ends at `base` (the prefix P, or the first opener's end E)."""
    if CAP:
        rebaseobs.KEEP = max(0, CAP - int(base))


def _held_pins(session):
    """The session's held pins: sumpin's text pin and (v7.7) its board pin."""
    reg = session.history_registry
    return [p for p in (getattr(reg, '_s2_pin', None), getattr(reg, '_s2_pin_b', None)) if isinstance(p, dict) and p.get('held')]


def _held_pin(session):
    """A held pin of the session (the text pin first), else None."""
    pins = _held_pins(session)
    return pins[0] if pins else None


def _horizon(session, rebaseobs):
    """The earliest position a later rebase of this session can keep after the prefix (rebaseobs's window start now: the window
    only moves forward as the session grows)."""
    obs = session.history_registry.observations
    if not obs:
        return 0
    _set_keep(rebaseobs, obs[0].start)
    w, _ = rebaseobs.window_start(session, len(session.token_ids))
    return w


# ---------------------------------------------------------------------------------------------------------- pin keep-first
def _keep_first_plan(self, session, rebaseobs, pins):
    """((E, w, E0), None) for a rebase with the pins `pins` held (v7.7: a list; one pin dict as before), else (None, reason)."""
    pins = [pins] if isinstance(pins, dict) else list(pins)
    obs = session.history_registry.observations
    if len(obs) < 2:
        return None, 'one observation'
    P = obs[0].start
    seg = session.segments[0] if session.segments else None
    if seg is None or seg[0] != 'observation' or int(seg[1]) != 0 or int(seg[2]) <= P:
        return None, 'no initial observation segment'
    E0 = int(seg[2])
    rev = {tok: ident for ident, tok in self._control_ids().items()}
    im_start, im_end = rev.get('<|im_start|>'), rev.get('<|im_end|>')
    if im_start is None or im_end is None:
        return None, 'no control token ids'
    toks = session.token_ids
    E = next((q for q in range(E0 - 1, P, -1) if toks[q] == im_start), None)
    if E is None and session.__dict__.get('_arc3_keep_first_E') == E0:
        E = E0                                          # a frame built by a keep-first rebase: the opener ends the segment
    if E is None or im_end in toks[E:E0] or im_end not in toks[P:E]:
        return None, 'no generation prompt after the opener'
    if not all(P <= int(pin['lo']) < int(pin['hi']) <= E for pin in pins):
        return None, 'pin outside the opener'
    if any(int(a) < E < int(b) for a, b in session.images):
        return None, 'image across the cut'
    head_images = [(int(a), int(b)) for a, b in session.images if int(b) <= E]
    parts = list(session.parts) + ([session._arc3_pin_part] if session.__dict__.get('_arc3_pin_part') else [])
    if head_images and not all(any(p[0] <= a < p[0] + p[1] for p in parts) for a, _ in head_images):
        return None, 'no rendered part for the opener images'
    limit = CAP or int(getattr(session.history_policy, 'target_basis_tokens', 0) or 0) or session.residency.resident_tokens
    end = len(toks)
    if E >= limit:
        return None, 'opener exceeds the cap'
    _set_keep(rebaseobs, E)
    w, _ = rebaseobs.window_start(session, end)
    if w < E0 or w <= P:
        return None, 'no window after the first interaction'
    if E + (end - w) > limit:
        return None, 'window exceeds the cap'
    if any(int(a) < w for a, _ in session.suffix_registry.live.values()):
        return None, 'live managed suffix before the window'
    return (E, w, E0), None


def _release_pin(session, pins, reason):
    """The pins' tokens leave with the rebase: released, not placed again in this session (sumpin's rule). `pins`: the held pins
    (v7.7: a list; one pin dict as before). STATS pin_released_rebase counts the rebases that released pins; sumpin counts each pin."""
    pins = [pins] if isinstance(pins, dict) else list(pins)
    for pin in pins:
        try:
            boards = pin.get('name') == 'boards'
            pin['held'] = False
            setattr(session.history_registry, '_s2_released_b' if boards else '_s2_released', True)   # sumpin._flag
            pin['released'] = dict(eviction=pin.get('index'), seconds=round(time.time() - float(pin.get('t0') or time.time()), 1),
                                   reason='rebase', pin='boards' if boards else 'text')
            from . import sumpin
            for key in ('pin_released', 'pin_released_rebase'):
                key = ('board_' + key) if boards else key
                sumpin.STATS[key] = sumpin.STATS.get(key, 0) + 1
            sumpin.STATS['releases'] = (sumpin.STATS['releases'] + [pin['released']])[-64:]
        except Exception as exc:                        # noqa: BLE001  the release itself is done above
            _err(exc)
    session.__dict__.pop('_arc3_pin_part', None)
    STATS['pin_released_rebase'] += 1
    STATS['keep_first_refused'][reason] = STATS['keep_first_refused'].get(reason, 0) + 1


async def _rebase_keep_first(self, mod, session, turn, local_history_tokens, plan):
    """Production's _rebase (session_service 7ce0eaef) with the kept pieces [(0, E), (w, end)]: the prefix and the first opener
    whole (its generation prompt [E, E0) dropped), then the window; otherwise line by line the same bookkeeping, engine restart and
    receipt."""
    started = time.monotonic()
    if session.open_image is not None:
        raise RuntimeError("rebase with an open image span")
    E, w, E0 = plan
    open_call = getattr(session, "open_tool_call", None)
    old_len = len(session.token_ids); resident_now = session.residency.resident_tokens
    obs = session.history_registry.observations; P = obs[0].start
    keep = [(0, E), (w, old_len)]
    parts = list(session.parts) + ([session._arc3_pin_part] if session.__dict__.get('_arc3_pin_part') else [])
    prompt, resident, n_images = mod.build_probe_prompt(session.token_ids, keep, old_len,
                                                        [tuple(x) for x in session.images], parts, ())
    new_len = E + old_len - w
    if resident != new_len or len(prompt["prompt_token_ids"]) != new_len:
        raise RuntimeError(f"keep-first rebase prompt has {resident} tokens, pieces expect {new_len}")
    shift = w - E

    def m(q):                                           # old position -> new position (None when dropped)
        return q if q < E else (q - shift if q >= w else None)

    def mspan(a, b):                                    # a span wholly inside one kept piece, else None
        if b <= E:
            return (a, b)
        return (a - shift, b - shift) if a >= w else None

    ev = session.residency.evicted
    restored = sum(max(0, min(b, E) - a) + max(0, min(b, old_len) - max(a, w)) for a, b in ev)
    segments = []
    for sk, a, b in session.segments:
        if b <= E:
            segments.append([sk, a, b])
        elif a < E:
            segments.append([sk, a, E])                 # the first opener's segment without its generation prompt
        elif a >= w:
            segments.append([sk, a - shift, b - shift])
    images = sorted([list(mspan(a, b)) for a, b in session.images if mspan(a, b) is not None])
    image_kinds = {mspan(a, b)[0]: session.image_kinds.get(a, "other") for a, b in session.images if mspan(a, b) is not None}
    structural = [m(q) for q in session.structural if m(q) is not None]
    session.open_tool_call = m(open_call) if open_call is not None else None
    kept_obs = [o for o in obs[1:] if o.start >= w]
    dropped = [o.observation_id for o in obs[1:] if o.start < w]
    new_obs = [obs[0]] + [mod.ObservationStart(o.observation_id, o.start - shift) for o in kept_obs]
    if any(x.start >= y.start for x, y in zip(new_obs, new_obs[1:])):
        raise RuntimeError(f"keep-first rebase observation registry would not increase: {new_obs[:3]}")
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
    session._arc3_keep_first_E = E
    turn.prompt = prompt; turn.recompute = True
    await self.backend.restart_request(session)
    info = {"rebase": session.rebases, "from_logical_tokens": old_len, "to_logical_tokens": new_len,
            "prefix_tokens": P, "window_start": w, "boundary": "pin_keep_first", "first_opener": [P, E],
            "dropped_generation_prompt": E0 - E, "protected_tail_start": None,
            "resident_before": resident_now, "target_tokens": CAP or None, "target_fallback": False,
            "restored_tokens": restored, "net_dropped_tokens": resident_now + local_history_tokens - new_len,
            "local_history_tokens": local_history_tokens, "stale_tool_call": open_call,
            "dropped_observation_ids": dropped, "partial_observation_id": None, "images": n_images,
            "restart_seconds": time.monotonic() - started, "engine_request_id": session.engine_request_id}
    session.rebase_info = info
    self.audit("rebase", session, operation_id=session.current_operation, **info)
    return info


# ---------------------------------------------------------------------------------------------------------- suspend
async def _suspend(self, mod, rebaseobs, sid, payload):
    session = self.get(sid)
    t0 = time.monotonic()
    async with session.lock:
        if session.state == 'suspended':
            STATS['suspend_noop'] += 1
            return dict(session_id=session.session_id, state='suspended', revision=session.revision, already=True)
        if session.state != 'paused' or session.current_operation or session.__dict__.get('_arc3_resuming'):
            STATS['suspend_refused'] += 1
            raise mod.SessionError('Suspend requires a paused session with no turn in flight (state %s)' % session.state, 409)
        if session.finish_reason == 'length':           # its next request would be a continuation, which cannot rebase
            STATS['suspend_refused'] += 1
            raise mod.SessionError('Suspend requires a completed assistant turn (not a length pause)', 409)
        expected = payload.get('expected_revision')
        if expected is not None and expected != session.revision:
            STATS['suspend_refused'] += 1
            raise mod.SessionError('Stale session revision', 409)
        if not session.history_registry.observations:
            STATS['suspend_refused'] += 1
            raise mod.SessionError('Suspend requires a registered observation (the resume rebases from it)', 409)
        session.state = 'suspended'                     # before the await: admission and the pool estimate see it at once
        try:
            await self.backend.close(session)           # aborts the engine request: its KV pages are freed
        except BaseException as exc:
            session.state = 'error'
            session.error = f'{type(exc).__name__}: {exc}'
            self.audit('failed', session, error='suspend: ' + session.error)
            raise
        for name in ('collector', 'output_future'):     # as sessmem at a close: no traceback keeps the last request output
            fut = getattr(session, name, None)
            if fut is not None and fut.done():
                if fut.cancelled():
                    try:
                        fut.exception()
                    except asyncio.CancelledError:
                        pass
                setattr(session, name, None)
        n = len(session.parts)
        try:
            horizon = _horizon(session, rebaseobs)
            session.parts = [q for q in session.parts if q[0] + q[1] > horizon]
        except Exception as exc:                        # noqa: BLE001  keep every part: the resume still works
            _err(exc)
            horizon = None
        session._arc3_suspended_at = time.time()
    seconds = time.monotonic() - t0
    STATS['suspends'] += 1
    STATS['suspend_parts_pruned'] += n - len(session.parts)
    STATS['suspend_parts_kept_max'] = max(STATS['suspend_parts_kept_max'], len(session.parts))
    STATS['suspend_seconds_max'] = max(STATS['suspend_seconds_max'], round(seconds, 3))
    STATS['suspended_now'] = _n_suspended(self.sessions.values())
    STATS['suspended_max'] = max(STATS['suspended_max'], STATS['suspended_now'])
    receipt = dict(session_id=session.session_id, state='suspended', revision=session.revision,
                   logical_tokens=len(session.token_ids), resident_tokens=session.residency.resident_tokens,
                   parts_kept=len(session.parts), parts_pruned=n - len(session.parts), parts_horizon=horizon,
                   suspended_sessions=STATS['suspended_now'], seconds=round(seconds, 4))
    self.audit('suspend', session, **{k: v for k, v in receipt.items() if k != 'session_id'})
    _save()
    return receipt


# ---------------------------------------------------------------------------------------------------------- patches
def _backstop():
    """The rebase mode's backstop: production's ARC3_REBASE_AT (read at every check) set to BACKSTOP when unset."""
    if MODE == 'rebase' and not os.environ.get('ARC3_REBASE_AT'):
        os.environ['ARC3_REBASE_AT'] = BACKSTOP
    STATS['backstop'] = os.environ.get('ARC3_REBASE_AT')


def patch_session_service(mod):
    from . import rebaseobs
    if os.environ.get('ARC3_REBASE', '1') != '1':
        raise RuntimeError('stage2_v1 rebase74: needs ARC3_REBASE=1 (the rendered parts a rebase re-sends are recorded only then)')
    _backstop()
    cls = mod.SessionManager
    SessionError = mod.SessionError

    def _get_max(self):                                 # the configured limit plus the suspended sessions (no engine request)
        d = self.__dict__
        base = d.get('_arc3_max_sessions_base', d.get('max_sessions', 0))
        return base + _n_suspended(getattr(self, 'sessions', {}).values())

    def _set_max(self, value):
        self.__dict__['_arc3_max_sessions_base'] = value

    cls.max_sessions = property(_get_max, _set_max)

    orig_append = cls.append

    async def append(self, sid, payload):
        if MODE != 'rebase' and isinstance(payload, dict) and (payload.get('arc3_suspend') is not None
                                                              or payload.get('arc3_rebase') is not None):
            STATS['refused_refresh_mode'] += 1
            raise SessionError('This server runs ARC3_REBASE_MODE=refresh: no rebase or suspend requests', 400)
        if isinstance(payload, dict) and payload.get('arc3_suspend') is not None:
            if payload.get('arc3_suspend') is not True:
                raise SessionError('arc3_suspend must be true', 400)
            return await _suspend(self, mod, rebaseobs, sid, payload)
        session = self.get(sid)
        resuming = False
        if session.state == 'suspended' and not session.__dict__.get('_arc3_resuming'):
            if (not isinstance(payload, dict) or payload.get('continuation') or payload.get('assistant_injection') is not None
                    or payload.get('tail_eviction') is not None):
                STATS['resume_refused'] += 1
                raise SessionError('A suspended session resumes with an external append (no continuation, injection or tail '
                                   'eviction)', 400)
            base = self.__dict__.get('_arc3_max_sessions_base')
            if base is not None:
                live = sum(1 for s in self.sessions.values()
                           if s.state not in ('closed', 'error', 'suspended') or s.__dict__.get('_arc3_resuming'))
                if live >= base:
                    STATS['resume_refused'] += 1
                    raise SessionError('Resident session admission limit reached (resume of a suspended session)', 429)
            session._arc3_resuming = True
            resuming = True
        try:
            return await orig_append(self, sid, payload)
        finally:
            if resuming:
                session.__dict__.pop('_arc3_resuming', None)

    append.__wrapped__ = orig_append
    cls.append = append

    orig_run_turn = cls._run_turn

    async def _run_turn(self, session, payload, *, initial):
        resumed = session.state == 'suspended'
        forced = bool(resumed or (not initial and isinstance(payload, dict) and payload.get('arc3_rebase') is True))
        if forced:
            STATS['forced_appends'] += 1
        if resumed:                                     # the resume: the turn rebases (a fresh engine request)
            session.state = 'paused'
            STATS['resumes'] += 1
            STATS['suspended_now'] = _n_suspended(self.sessions.values())
            at = session.__dict__.pop('_arc3_suspended_at', None)
            self.audit('resume', session, operation_id=session.current_operation, logical_tokens=len(session.token_ids),
                       suspended_seconds=round(time.time() - at, 1) if at else None,
                       suspended_sessions=STATS['suspended_now'])
        n0, rev0 = len(session.token_ids), session.revision
        session._arc3_rebase_forced = forced
        session._arc3_rebase_due_now = False
        try:
            return await orig_run_turn(self, session, payload, initial=initial)
        except BaseException:
            if resumed and session.state == 'paused' and len(session.token_ids) == n0 and session.revision == rev0:
                session.state = 'suspended'             # refused before submission: still suspended (no engine request)
                session._arc3_suspended_at = time.time()
                STATS['resume_restored'] += 1
                STATS['suspended_now'] = _n_suspended(self.sessions.values())
            raise
        finally:
            session._arc3_rebase_forced = False
            session._arc3_rebase_due_now = False

    _run_turn.__wrapped__ = orig_run_turn
    cls._run_turn = _run_turn

    orig_due = cls._rebase_due

    def _rebase_due(self, session, new_ids, *, initial, continuation):
        due = orig_due(self, session, new_ids, initial=initial, continuation=continuation)
        if NORECORD and due:                            # v7.5 B3: unreachable in refresh mode; counted if it ever happens
            STATS['norecord_rebase_due'] += 1
        if getattr(session, '_arc3_rebase_forced', False):
            if not due:
                due = bool(not initial and not continuation and new_ids
                           and session.history_policy.policy != 'oldest_recompute' and session.history_registry.observations)
            STATS['forced_rebases' if due else 'forced_not_due'] += 1
        session._arc3_rebase_due_now = bool(due)
        return due

    _rebase_due.__wrapped__ = orig_due
    cls._rebase_due = _rebase_due

    inner_record = cls._record_part                     # rebaseobs's: its parts horizon reads KEEP

    def _record_part(self, session, start, length, prompt):
        if NORECORD:                                    # v7.5 B3: refresh mode keeps no rendered parts
            STATS['parts_not_recorded'] += 1
            return None
        obs = session.history_registry.observations
        if obs:
            _set_keep(rebaseobs, obs[0].start)
        held = _held_pin(session) is not None
        if (CAP and start == 0 and held and isinstance(prompt, dict)
                and prompt.get('type') == 'multimodal'):
            session._arc3_pin_part = (start, length, prompt)   # the first opener's images for a pin-held rebase (the cap set)
        elif not held:
            session.__dict__.pop('_arc3_pin_part', None)        # released by sumpin's rule: no keep-first rebase any more
        return inner_record(self, session, start, length, prompt)

    _record_part.__wrapped__ = inner_record
    cls._record_part = _record_part

    inner_rebase = cls._rebase                          # rebaseobs's: the two-piece edge plan, else production's

    async def _rebase(self, session, turn, local_history_tokens=0):
        STATS['rebases'] += 1
        pins = _held_pins(session)                      # v7.7: the text pin and the board pin
        info = None
        if pins:
            plan, why = _keep_first_plan(self, session, rebaseobs, pins)
            if plan is not None:
                info = await _rebase_keep_first(self, mod, session, turn, local_history_tokens, plan)
                STATS['keep_first'] += 1
                tp = getattr(session.history_registry, '_s2_pin', None)
                if isinstance(tp, dict) and not tp.get('held'):  # v7.7 (IE 10-24): the text pin released, the board pin held:
                    STATS['keep_first_text_released'] += 1       # the frame re-prefills the released text, unpinned
            else:
                _release_pin(session, pins, why)
        if info is None:
            obs = session.history_registry.observations
            if obs:
                _set_keep(rebaseobs, obs[0].start)
            info = await inner_rebase(self, session, turn, local_history_tokens)
        STATS['rebase_tokens_max'] = max(STATS['rebase_tokens_max'], int(info.get('to_logical_tokens') or 0))
        STATS['last'] = dict(boundary=info.get('boundary'), from_logical=info.get('from_logical_tokens'),
                             to_logical=info.get('to_logical_tokens'), prefix=info.get('prefix_tokens'),
                             window_start=info.get('window_start'), forced=bool(getattr(session, '_arc3_rebase_forced', False)),
                             t=round(time.time(), 1))
        _save()
        return info

    _rebase.__wrapped__ = inner_rebase
    cls._rebase = _rebase

    inner_close = cls._close_session                    # (sessmem's when on: its frees run first)

    async def _close_session(self, session):
        try:
            return await inner_close(self, session)
        finally:
            session.__dict__.pop('_arc3_pin_part', None)  # a closed session keeps no rendered images

    _close_session.__wrapped__ = inner_close
    cls._close_session = _close_session

    if 'poolpress' in _parts():                         # its pool estimate counts unsuspended sessions only
        from . import poolpress
        if not getattr(poolpress.pool_pages, '_arc3_rebase74', False):
            inner_pages = poolpress.pool_pages

            def pool_pages(sessions, growth=None):
                return inner_pages([s for s in sessions if getattr(s, 'state', '') != 'suspended'], growth)

            pool_pages._arc3_rebase74 = True
            pool_pages.__wrapped__ = inner_pages
            poolpress.pool_pages = pool_pages


def _parts():
    try:
        from . import install
        return install.PARTS
    except Exception:                                   # noqa: BLE001  (the CPU test imports the part alone)
        return frozenset(p for p in os.environ.get('ARC3_STAGE2', '').split(',') if p)


def patch_persistent_endpoint(mod):
    cls = mod.VllmBackend
    orig = cls.frequency_snapshot

    async def frequency_snapshot(self, session):
        if getattr(session, '_arc3_rebase_due_now', False):
            STATS['snapshots_skipped'] += 1
            from qsa_frequency import QSAFrequencyError
            raise QSAFrequencyError(SKIP_TEXT)
        return await orig(self, session)

    frequency_snapshot.__wrapped__ = orig
    cls.frequency_snapshot = frequency_snapshot

    orig_turn = cls.turn

    async def turn(self, session, turn, *, initial):     # the finish guard (module docstring)
        result = await orig_turn(self, session, turn, initial=initial)
        reason = result.get('finish_reason') if isinstance(result, dict) else None
        if reason not in ('stop', 'length'):
            n = len(result.get('token_ids') or []) if isinstance(result, dict) else 0
            STATS['error_finishes'] += 1
            STATS['last_error_finish'] = dict(t=round(time.time(), 1), reason=str(reason), tokens=n, initial=bool(initial),
                                              session=str(getattr(session, 'session_id', ''))[-12:])
            _save()
            raise RuntimeError(FINISH_TEXT % (reason, n))
        return result

    turn.__wrapped__ = orig_turn
    cls.turn = turn


PATCHERS = {'session_service': patch_session_service, 'persistent_endpoint': patch_persistent_endpoint}
