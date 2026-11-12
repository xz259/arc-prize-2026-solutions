"""stage2_v1 'sumpin' part (2026-10-17, Claude 099768dc; harness v6.5 C5, user via IE; design IE-approved 10-17 ~04:0x): the
restart opener's leading section (the newest attempt notes and the previous level's summary, <= the client's pin cap) is
pinned: protected from eviction as an extension of the system prompt's protected prefix, scored but never selected, and
released whole when the release rule fires. Inert without the client's request (arm B: production's plans exactly).

Request. /start carries arc3_summary_pin = {message_index, parts_end} (harness v6.5 transport, arm A only): the opener message
and the number of its leading text parts that form the section. render_initial's result gets the section's span: the
rendering of [messages before the opener, the opener with its first parts_end parts] without the generation prompt, its
trailing "<|im_end|>\\n" removed, must be a token prefix of the initial input; then [P, Q) = [the opener's observation start,
that prefix's length], carried to the history registry in the observation record's proof (the registry accepts a proof
dict). Otherwise no pin (pin_boundary_mismatch).

Hold. While held, [P, Q) is removed from the survivors _plan_split sees (QSA groups, the oldest split and the hollow sweep
never select it; its groups stay in the frequency snapshot), and from any decision whose ranges reach it (agecap's whole old
interactions): the decision is re-planned without those tokens under production's protection plus the pin.

Release. At every triggered plan while held: the plan's index since the pin is counted; from index GRACE + 1 on, an eviction
whose effective policy is qsa_frequency and whose receipt has a cut_score is evaluated: p = the nearest-rank Q-quantile
(default p75) of the pinned tokens' group scores (the layer nanmean, as production ranks groups; unexposed groups excluded); a
failure is p < cut_score; STREAK (default 6) consecutive failures release the pin at that plan: [P, Q)'s resident tokens are
added to the decision whole, and the pin is not placed again in that session. An eviction that is not evaluated (another
policy, no cut, no finite pinned score) leaves the streak as it is (pin_unevaluated). A capacity failure with the pin held is
retried without the hold and releases the pin whole (pin_capacity_releases).

Receipts: STATS (pins_placed, pinned_tokens, pin_boundary_mismatch, pin_released, pin_capacity_releases, pin_fail_streak_max,
pin_evals, pin_unevaluated, pin_agecap_trimmed tokens, releases [eviction index, seconds held, reason]) and per triggered plan
receipt['summary_pin'] (range, held, eviction index, evaluated, p, cut, margin p - cut, streak, released, seconds held).
Invariants (IE 10-17 05:3x; on receipt_check3's investigate list, counted inside the planner, never changing a decision):
pin_plan_touched = a plan returned with the pin held whose ranges intersect [P, Q); pin_plan_after_release = a plan evaluated as
held on a registry whose pin was released earlier (no re-pin in a session); last_violation records the latest one.
Receipt save (IE 10-17 06:1x; sumpin had none, so its STATS reached the receipt only with another part's save): install.save()
at every release, violation and error, from the planner every ARC3_S2_SUMPIN_SAVE_S (300) s, and at a normal interpreter exit.
ARC3_S2_SUMPIN_GRACE / _STREAK / _Q (gate overrides).

v7.7 board pin (apply_sumpin77.py, 2026-10-24, Claude main session 099768dc; the user: "pin the text part and images separately",
V67_DESIGN 316-318). A request with parts_split = k (1 <= k < parts_end; parts_end = len(content); the opener the last message;
its first k parts text) places two pins: the text pin [P, Q_k), Q_k from the probe of the parts [0, k) as above, and the board pin
[Q_k, Q_n), Q_n = the opener's end read from the initial input's tail (the input ends with the opener's <|im_end|>\\n and the
generation prompt), so the images are not rendered a second time. Each pin is held (both spans leave _plan_split's survivors and
agecap's trim), scored, evaluated and released on its own by the same GRACE / STREAK / QUANT rule; a capacity failure is retried
with the board pin released first, then with neither. Without parts_split nothing changes. Receipts: STATS pins_split and board_*
(the text pin's counters, prefixed), receipt['board_pin'] beside receipt['summary_pin'], the releases' pin name."""
import atexit
import contextvars
import copy
import math
import os
import time
import warnings

PINS = {'history_eviction': '88a5d9aef289c87b485104a1cf8d08701a96600198a70196ccd43cd7803c8fc0',     # as agecap's
        'protocol': '7a11929c16dc682ef92a907ba8aaaeda0c4a9061f04b31eb799bebbf18660a23',
        'session_service': '7ce0eaef072e45354cfae8476719313a401387ea379b3fe9c3ca9c1ab40193e8'}      # as sessmem's
GRACE = int(os.environ.get('ARC3_S2_SUMPIN_GRACE', '2'))
STREAK = int(os.environ.get('ARC3_S2_SUMPIN_STREAK', '6'))
QUANT = float(os.environ.get('ARC3_S2_SUMPIN_Q', '0.75'))
RATIO = 4
STATS = dict(requests=0, pins_placed=0, pinned_tokens=0, pin_boundary_mismatch=0, pin_released=0, pin_released_rule=0,
             pin_released_capacity=0, pin_capacity_releases=0,
             pin_fail_streak_max=0, pin_evals=0, pin_unevaluated=0, pin_agecap_trimmed=0, plans_held=0, errors=0,
             pin_plan_touched=0, pin_plan_after_release=0, last_violation=None,
             last_error=None, releases=[], config=dict(grace=GRACE, streak=STREAK, q=QUANT))
STATS.update({'board_' + k: 0 for k in ('pins_placed', 'pinned_tokens', 'pin_released', 'pin_released_rule', 'pin_released_capacity',
                                       'pin_capacity_releases', 'pin_released_rebase', 'pin_fail_streak_max', 'pin_evals',
                                       'pin_unevaluated', 'pin_agecap_trimmed', 'pin_plan_touched', 'pin_plan_after_release')},
             pins_split=0)                                       # v7.7: the board pin's counters
_REQ = contextvars.ContextVar('arc3_sumpin_request', default=None)
_CUR = contextvars.ContextVar('arc3_sumpin_hold', default=None)
SAVE_EVERY = float(os.environ.get('ARC3_S2_SUMPIN_SAVE_S', '300'))
_LAST_SAVE = [0.0]
_ATEXIT = [False]


def _save(force=False):
    """The process receipt: install.save() refreshes REPORT['sumpin'] from STATS, as for every part (never raises)."""
    if not force and time.time() - _LAST_SAVE[0] < SAVE_EVERY:
        return
    _LAST_SAVE[0] = time.time()
    try:
        from . import install
        install.save()
    except Exception:                                           # noqa: BLE001  receipts are diagnostics
        pass


def _err(exc):
    STATS['errors'] += 1
    STATS['last_error'] = repr(exc)[:300]
    _save(force=True)


def _name(pin):
    """v7.7: 'boards' for the board pin, 'text' for the text pin (a pin without a name is the text pin, as before)."""
    return 'boards' if pin.get('name') == 'boards' else 'text'


def _key(pin, key):
    """v7.7: the STATS key of a pin's counter: the text pin's own, the board pin's prefixed board_."""
    return 'board_' + key if _name(pin) == 'boards' else key


def _inc(pin, key, n=1):
    k = _key(pin, key)
    STATS[k] = STATS.get(k, 0) + n


def _flag(pin):
    """v7.7: the registry flag a pin's release sets (the invariant pin_plan_after_release reads it): _s2_released for the text
    pin (as before), _s2_released_b for the board pin."""
    return '_s2_released_b' if _name(pin) == 'boards' else '_s2_released'


def _rkey(pin):
    """v7.7: the plan receipt's key for a pin: summary_pin (the text pin, as before), board_pin."""
    return 'board_pin' if _name(pin) == 'boards' else 'summary_pin'


def _held(registry):
    """v7.7: the registry's held pins, the text pin first."""
    return [p for p in (getattr(registry, '_s2_pin', None), getattr(registry, '_s2_pin_b', None))
            if isinstance(p, dict) and p.get('held')]


def _spans(hold):
    """_CUR's value: one span (lo, hi) (the single pin) or a list of spans (v7.7)."""
    hold = list(hold)
    return [tuple(hold)] if len(hold) == 2 and not isinstance(hold[0], (tuple, list)) else [tuple(s) for s in hold]


def _held_ok(pin, decision):
    """Invariant: a plan returned with the pin held leaves [P, Q) alone (counts pin_plan_touched otherwise; never raises)."""
    try:
        lo, hi = pin['lo'], pin['hi']
        if pin['held'] and any(int(a) < hi and lo < int(b) for a, b in decision.ranges):
            _inc(pin, 'pin_plan_touched')
            STATS['last_violation'] = dict(kind='touched', pin=_name(pin), lo=lo, hi=hi, eviction=pin['index'],
                                           t=round(time.time(), 1))
            _save(force=True)
    except Exception as exc:                                    # noqa: BLE001  a receipt check never breaks a plan
        _err(exc)
    return decision


# ------------------------------------------------------------------------------------------------------------ request
def _valid_request(req):
    """{message_index, parts_end} (the section [0, parts_end)), or v7.7's {message_index, parts_end, parts_split} (the text pin
    [0, parts_split) and the board pin [parts_split, parts_end))."""
    if (isinstance(req, dict) and set(req) in ({'message_index', 'parts_end'}, {'message_index', 'parts_end', 'parts_split'})
            and type(req['message_index']) is int and type(req['parts_end']) is int and req['message_index'] >= 0
            and req['parts_end'] >= 1
            and ('parts_split' not in req or (type(req['parts_split']) is int and 1 <= req['parts_split'] < req['parts_end']))):
        return dict(req)
    return None


def patch_session_service(mod):
    """SessionManager.start: the request's pin field rides in a context variable to render_initial (same task)."""
    cls = mod.SessionManager
    orig = cls.start

    async def start(self, payload):
        req = payload.get('arc3_summary_pin') if isinstance(payload, dict) else None
        token = _REQ.set(_valid_request(req) if req is not None else None)
        if req is not None and _REQ.get() is None:
            STATS['pin_boundary_mismatch'] += 1                  # malformed request: no pin
        try:
            return await orig(self, payload)
        finally:
            _REQ.reset(token)

    start.__wrapped__ = orig
    cls.start = start


def patch_protocol(mod):
    """render_initial: the section's span [P, Q) in the observation start's proof (see the module docstring)."""
    orig = mod.render_initial

    async def render_initial(serving_chat, payload, *, managed_suffixes=None, request_factory=None):
        rendered = await orig(serving_chat, payload, managed_suffixes=managed_suffixes, request_factory=request_factory)
        req = _REQ.get()
        if req is not None:
            STATS['requests'] += 1
            try:
                span = await _span(mod, serving_chat, payload, rendered, req, request_factory)
            except Exception as exc:                            # noqa: BLE001  never break a start: no pin
                _err(exc)
                span = None
            if span is None:
                STATS['pin_boundary_mismatch'] += 1
            else:
                rec = rendered.observation_starts[0]
                proof = dict(rec.get('proof') or {})
                proof['arc3_pin'] = list(span[:2])
                if len(span) == 3:                              # v7.7: the board pin [Q_k, Q_n)
                    proof['arc3_pin_boards'] = list(span[1:])
                rec['proof'] = proof
        return rendered

    render_initial.__wrapped__ = orig
    mod.render_initial = render_initial


async def _span(mod, serving_chat, payload, rendered, req, request_factory):
    """(P, Q) for a section request; v7.7: (P, Q_k, Q_n) for a split request (module docstring); None when the request does not
    resolve to a token boundary of the initial input."""
    starts = rendered.observation_starts
    msgs = payload.get('messages') or []
    i, n = req['message_index'], req['parts_end']
    k = req.get('parts_split')
    if len(starts) != 1 or not 0 <= i < len(msgs):
        return None
    msg = msgs[i]
    content = msg.get('content')
    if k is not None:                                           # v7.7: the whole opener, its first k parts text
        if (msg.get('role') != 'user' or not isinstance(content, list) or n != len(content) or i != len(msgs) - 1
                or not 1 <= k < n or any(not isinstance(p, dict) for p in content)
                or any(p.get('type') != 'text' for p in content[:k])):
            return None
    elif (msg.get('role') != 'user' or not isinstance(content, list) or n >= len(content)
            or any(not isinstance(p, dict) or p.get('type') != 'text' for p in content[:n])):
        return None
    m = n if k is None else k                                   # the probe renders text parts only
    probe = dict(payload, messages=copy.deepcopy(msgs[:i]) + [dict(copy.deepcopy(msg), content=copy.deepcopy(content[:m]))],
                 add_generation_prompt=False, continue_final_message=False)
    probe.pop('max_tokens', None)
    probe.pop('max_completion_tokens', None)
    _, out = await mod._render(serving_chat, probe, request_factory or mod._request_factory)
    ids = list(out['prompt_token_ids'])
    tokenizer = serving_chat.renderer.tokenizer
    end = mod._ids(tokenizer, '<|im_end|>\n')
    if not end or ids[-len(end):] != end:
        return None
    pref = ids[:-len(end)]
    full = rendered.engine_input['prompt_token_ids']
    p = int(starts[0]['local_start_token'])
    if list(full[:len(pref)]) != pref or not p < len(pref):
        return None
    if k is None:
        return p, len(pref)
    gen = mod._ids(tokenizer, '<|im_start|>assistant\n<think>\n')   # v7.7: the opener's end Q_n from the input's tail: the input
    qn = len(full) - len(gen) - len(end)                             # ends with its <|im_end|>\n and the generation prompt
    if (not gen or list(full[len(full) - len(gen):]) != gen or list(full[qn:qn + len(end)]) != end
            or not len(pref) < qn):
        return None
    STATS['pins_split'] += 1
    return p, len(pref), qn


def _pin_span(span):
    return isinstance(span, list) and len(span) == 2 and all(type(x) is int for x in span) and 0 <= span[0] < span[1]


def patch_history_registry(cls):
    """The pin rides from the initial observation's proof into the session's registry when the registration commits (v7.7: the
    board pin too, when its span follows the text pin's)."""
    orig_plan, orig_commit = cls.plan_registration, cls.commit

    def plan_registration(self, observation_starts, *, logical_offset, delta_tokens):
        transition = orig_plan(self, observation_starts, logical_offset=logical_offset, delta_tokens=delta_tokens)
        for rec in observation_starts if isinstance(observation_starts, list) else []:
            proof = (rec.get('proof') or {}) if isinstance(rec, dict) else {}
            span = proof.get('arc3_pin')
            if _pin_span(span) and getattr(self, '_s2_pin', None) is None:
                boards = proof.get('arc3_pin_boards')
                boards = ((logical_offset + boards[0], logical_offset + boards[1])
                          if _pin_span(boards) and boards[0] == span[1] else None)
                self._s2_pin_pending = (transition, (logical_offset + span[0], logical_offset + span[1]), boards)
        return transition

    def commit(self, transition):
        pending = getattr(self, '_s2_pin_pending', None)
        orig_commit(self, transition)
        if pending is not None and pending[0] is transition:
            lo, hi = pending[1]
            self._s2_pin = dict(lo=lo, hi=hi, held=True, index=0, streak=0, evals=0, unevaluated=0, t0=time.time(),
                                released=None)
            STATS['pins_placed'] += 1
            STATS['pinned_tokens'] += hi - lo
            if pending[2] is not None:                          # v7.7: the board pin, held and released on its own
                blo, bhi = pending[2]
                self._s2_pin_b = dict(name='boards', lo=blo, hi=bhi, held=True, index=0, streak=0, evals=0, unevaluated=0,
                                      t0=time.time(), released=None)
                STATS['board_pins_placed'] += 1
                STATS['board_pinned_tokens'] += bhi - blo
        self._s2_pin_pending = None

    plan_registration.__wrapped__, commit.__wrapped__ = orig_plan, orig_commit
    cls.plan_registration, cls.commit = plan_registration, commit


# ------------------------------------------------------------------------------------------------------------ planner
def _merge(ranges):
    out = []
    for a, b in sorted((int(a), int(b)) for a, b in ranges if b > a):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _protected(mod, registry, ledger, rec, kw):
    """Production's protection for this plan: the prefix, the tail and the pinned latest observation (as agecap rebuilds it)."""
    logical = ledger.logical_tokens
    obs = registry.observations
    prefix = obs[0].start if obs else 0
    tail4 = max(0, logical - 4)
    protected = [(0, prefix)] if prefix else []
    if logical:
        protected = [(0, logical)] if (protected and tail4 <= prefix) else protected + [(tail4, logical)]
    pin = (rec.get('observation_pin') or {}).get('range')
    if pin and not (rec.get('observation_pin') or {}).get('relaxed'):
        protected = mod._union(protected, tuple(pin))
    return protected


def _replan(mod, decision, ledger, protected, combined, history, kw):
    plan = ledger.plan(combined, protected=protected)
    rec = decision.receipt
    budget = rec.get('resident_token_budget')
    inc = kw.get('incoming_tokens', 0)
    if budget is not None and plan.resident_after + inc >= budget:
        raise mod.HistoryCapacityError('summary pin: the re-planned decision leaves no output capacity', receipt=rec)
    rec.update(total_removed_tokens=plan.removed_tokens, resident_tokens_after=plan.resident_after,
               resident_tokens_after_input=plan.resident_after + inc,
               minimum_output_capacity=(budget - plan.resident_after - inc) if budget is not None else None,
               combined_ranges=[list(p) for p in combined], history_ranges=[list(p) for p in history])
    return mod.HistoryDecision(tuple(combined), tuple(history), decision.suffix_ranges, plan, rec)


def _quantile(frequency, lo, hi):
    """The nearest-rank QUANT quantile of the pinned tokens' group scores (layer nanmean); None without a finite score."""
    import numpy as np
    if not frequency or frequency.get('matrix') is None:
        return None
    M = frequency['matrix']
    lay = np.asarray(frequency['group_ids_array'], dtype=np.int64)
    gids = np.arange(lo // RATIO, (hi - 1) // RATIO + 1, dtype=np.int64)
    if not len(lay):
        return None
    idx = np.searchsorted(lay, gids)
    ok = idx < len(lay)
    ok[ok] = lay[idx[ok]] == gids[ok]
    if not ok.any():
        return None
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        m = np.nanmean(M[:, idx[ok]], axis=0)
    g = gids[ok]
    w = np.minimum(hi, (g + 1) * RATIO) - np.maximum(lo, g * RATIO)
    fin = np.isfinite(m) & (w > 0)
    if not fin.any():
        return None
    vals = np.sort(np.repeat(m[fin], w[fin]))
    k = max(0, min(len(vals) - 1, math.ceil(QUANT * len(vals)) - 1))
    return float(vals[k])


class _Capacity(Exception):
    """The held plan failed for capacity: `decision` is the held plan when agecap's re-plan failed (the release takes it, as
    before), None when the planner itself failed (the release re-plans without the hold)."""

    def __init__(self, decision):
        super().__init__('summary pin: capacity')
        self.decision = decision


def _protect(mod, registry, ledger, rec, kw, spans):
    """Production's protection plus the given spans (the pins still held)."""
    out = _protected(mod, registry, ledger, rec, kw)
    for s in spans:
        out = mod._union(out, tuple(s))
    return out


def _held_decision(mod, inner, policy, registry, ledger, kw, held):
    """The plan with the pins `held` held: their spans leave _plan_split's survivors; agecap's whole old interactions that
    reach them are trimmed and the decision re-planned under production's protection plus the spans. (decision, {id(pin):
    trimmed tokens}); raises _Capacity."""
    spans = [(p['lo'], p['hi']) for p in held]
    token = _CUR.set(spans)
    try:
        decision = inner(policy, registry, ledger, **kw)
    except mod.HistoryCapacityError:
        decision = None
    finally:
        _CUR.reset(token)
    if decision is None:
        raise _Capacity(None)
    trimmed = {}
    if any(a < hi and lo < b for a, b in decision.ranges for lo, hi in spans):   # agecap's whole old interactions reach a pin
        rec = decision.receipt
        try:
            combined = _merge(mod._subtract([tuple(r) for r in decision.ranges], spans))
            history = _merge(mod._subtract([tuple(r) for r in decision.history_ranges], spans))
            for p in held:
                trimmed[id(p)] = sum(min(b, p['hi']) - max(a, p['lo']) for a, b in decision.ranges if a < p['hi'] and p['lo'] < b)
            replanned = _replan(mod, decision, ledger, _protect(mod, registry, ledger, rec, kw, spans), combined, history, kw)
        except mod.HistoryCapacityError:
            raise _Capacity(decision)
        decision = replanned
        for p in held:
            _inc(p, 'pin_agecap_trimmed', trimmed[id(p)])
    return decision, trimmed


def patch_history_eviction(mod):
    orig_split = mod._plan_split

    def _plan_split(policy, survivors, *args, **kwargs):
        hold = _CUR.get()
        if hold is not None:
            survivors = mod._subtract(list(survivors), _spans(hold))
        return orig_split(policy, survivors, *args, **kwargs)

    _plan_split.__wrapped__ = orig_split
    mod._plan_split = _plan_split
    inner = mod.plan_history_eviction

    def plan_history_eviction(policy, registry, ledger, **kw):
        _save()                                                  # periodic (SAVE_EVERY)
        pins = _held(registry)
        if not pins:
            return inner(policy, registry, ledger, **kw)
        STATS['plans_held'] += 1
        for pin in pins:                                         # invariant: nothing is held after that pin's release
            if getattr(registry, _flag(pin), False):
                _inc(pin, 'pin_plan_after_release')
                STATS['last_violation'] = dict(kind='after_release', pin=_name(pin), lo=pin['lo'], hi=pin['hi'],
                                               eviction=pin['index'], t=round(time.time(), 1))
                _save(force=True)
        held, dropped, decision, trimmed = list(pins), [], None, {}
        while held:                                              # the hold cost the capacity: release the board pin first,
            try:                                                 # then the rest (each whole)
                decision, trimmed = _held_decision(mod, inner, policy, registry, ledger, kw, held)
                break
            except _Capacity as cap:
                drop = [p for p in held if _name(p) == 'boards'] if len(held) > 1 else list(held)
                held = [p for p in held if all(p is not d for d in drop)]
                dropped += drop
                if not held:
                    decision = cap.decision if cap.decision is not None else inner(policy, registry, ledger, **kw)
        if dropped:
            decision = _release(mod, [(p, 'capacity') for p in dropped], registry, ledger, decision, kw)
            if not held:
                return decision
        rec = decision.receipt
        if not rec.get('triggered'):
            for pin in held:
                _held_ok(pin, decision)
            return decision
        fire = []
        for pin in held:
            lo, hi = pin['lo'], pin['hi']
            pin['index'] += 1
            info = dict(range=[lo, hi], tokens=hi - lo, held=True, eviction=pin['index'], evaluated=False,
                        agecap_trimmed=trimmed.get(id(pin), 0), held_seconds=round(time.time() - pin['t0'], 1))
            if pin['index'] > GRACE:
                cut = rec.get('cut_score')
                q = (_quantile(kw.get('frequency'), lo, hi) if (rec.get('effective_policy') == 'qsa_frequency' and cut is not None)
                     else None)
                if q is None:
                    pin['unevaluated'] += 1
                    _inc(pin, 'pin_unevaluated')
                else:
                    pin['evals'] += 1
                    _inc(pin, 'pin_evals')
                    pin['streak'] = pin['streak'] + 1 if q < cut else 0
                    k = _key(pin, 'pin_fail_streak_max')
                    STATS[k] = max(STATS.get(k, 0), pin['streak'])
                    info.update(evaluated=True, p=round(q, 6), cut=round(float(cut), 6), margin=round(q - float(cut), 6))
            info['streak'] = pin['streak']
            rec[_rkey(pin)] = info
            if pin['streak'] >= STREAK:
                fire.append(pin)
        if fire:
            decision = _release(mod, [(p, 'rule') for p in fire], registry, ledger, decision, kw)
        for pin in held:
            if pin['held']:
                _held_ok(pin, decision)
        return decision

    plan_history_eviction.__wrapped__ = inner
    mod.plan_history_eviction = plan_history_eviction
    patch_history_registry(mod.HistoryRegistry)
    if not _ATEXIT[0]:                                           # the planner's process: a last save at a normal exit
        atexit.register(_save, True)
        _ATEXIT[0] = True


def _release(mod, released, registry, ledger, decision, kw):
    """Each (pin, reason) of `released`: the pin's resident tokens join the decision whole; the pin is not placed again in this
    session. One re-plan under production's protection plus the pins still held (v7.7)."""
    rec = decision.receipt
    logical = ledger.logical_tokens
    gone = _merge(list(ledger.evicted) + [tuple(r) for r in decision.ranges])
    adds = []
    for pin, reason in released:
        lo, hi = pin['lo'], pin['hi']
        add = mod._subtract([(lo, min(hi, logical))], gone) if lo < logical else []
        adds += add
        pin['held'] = False
        setattr(registry, _flag(pin), True)                      # the invariant pin_plan_after_release reads it
        pin['released'] = dict(eviction=pin['index'], seconds=round(time.time() - pin['t0'], 1), reason=reason, pin=_name(pin))
        _inc(pin, 'pin_released')
        if reason == 'capacity':                                 # IE: pressure releases apart from the rule's
            _inc(pin, 'pin_capacity_releases')
            _inc(pin, 'pin_released_capacity')
            hend = max(lo, logical - int(rec.get('protect_recent_tokens') or 0))
            prefix = registry.observations[0].start if registry.observations else 0
            res = mod._subtract([(prefix, hend)], _merge(list(ledger.evicted)))
            pool = sum(b - a for a, b in res) - sum(max(0, min(b, hi) - max(a, lo)) for a, b in res)
            pin['released'].update(target=rec.get('required_tokens'), pool=pool, resident=rec.get('resident_tokens_after_suffix'),
                                   budget=rec.get('resident_token_budget'))
        else:
            _inc(pin, 'pin_released_rule')
        STATS['releases'] = (STATS['releases'] + [pin['released']])[-64:]
        info = dict(rec.get(_rkey(pin)) or dict(range=[lo, hi], tokens=hi - lo, eviction=pin['index']))
        info.update(held=False, released=True, reason=reason, released_tokens=sum(b - a for a, b in add),
                    **{k: pin['released'][k] for k in ('target', 'pool', 'resident', 'budget') if k in pin['released']})
        rec[_rkey(pin)] = info
    _save(force=True)
    if not adds:
        return decision
    combined = _merge([tuple(r) for r in decision.ranges] + adds)
    history = _merge([tuple(r) for r in decision.history_ranges] + adds)
    try:
        return _replan(mod, decision, ledger, _protect(mod, registry, ledger, rec, kw, [(p['lo'], p['hi']) for p in _held(registry)]),
                       combined, history, kw)
    except Exception as exc:                                     # noqa: BLE001  production's decision stands
        _err(exc)
        return decision


PATCHERS = {'session_service': patch_session_service, 'protocol': patch_protocol, 'history_eviction': patch_history_eviction}
