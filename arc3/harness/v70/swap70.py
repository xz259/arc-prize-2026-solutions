"""v70/swap70.py (2026-10-19, Claude Opus 5.5 builder subagent of session 099768dc; V67_DESIGN decisions 145-152): the
expectation-based swap-out's pure parts, stdlib only, so the harness (v62/agent.py, v62/solver.py) and the allocation
simulator (simalloc69) can compute the same numbers:
  load(path)      the index file (SCHEMA below: simalloc69's key layout, 18:41 UTC), validated, flattened; ValueError names
                  the first bad key
  index(C, s)     the index of one game's state s (expected score gain per token), or (index, components) with detail=True
  swap_rule(...)  the park decision from the compared indexes and the gates (pure; the agent supplies the numbers)
  REGISTRY        the running games' latest indexes (one lock), for the 'worst running game' gate and the read
  CLAIMS          pending index / cap parks (one claim per waiting game)
  ADMITS          INDEX_ADMIT's admission receipts
  TLS             the game session of the current thread (set by agent.py's play wrapper; the gate's acquire reads it)

The index (simalloc69 x69.py's chain-Gittins index; closed form instead of x69's 5K grid):
  idx = max over D in horizons (D <= tokens left) and over k = 0.. further levels of
        [P_D V_L(a + rate_a E_D) + sum_{i<=k} reach_i v_i] / [c + E_D + sum_{i<=k} reach_{i-1} E_i]
  h(t, L, pace) = exp(alpha[bin(t)] + beta_logL log L + beta_pace pace) per token, x tail_scale from tail_from; S = the
  survival from the level's current tokens t0; P_D = 1 - S(t0 + D) / S(t0); E_D = the integral of S over the window (exact
  per constant-hazard segment). V_L = L / W x min(score_cap, 100 (b_hat / a_total)^2) ('official'; 'lw': L / W x s_fresh),
  b_hat = the median actions of the game's cleared levels (else b0); the chain of fresh levels j = L+1..n over chain_horizon
  tokens each, v_j = j / W x s_fresh, cut where its tokens exceed the tokens left. W = n (n + 1) / 2. Tokens left = time left x
  lane rate (lane.tps0 x max(lane.floor, 1 - lane.decline_per_h x hours run)), capped by the cap left (hard caps only), minus
  c; 0 when that is <= min_tokens_left. c = switch.reorient_tokens for a parked game (0 otherwise); a parked game's hazard is
  multiplied by switch.theta (the context-loss hazard factor). rate_a = (a + rate_prior x rate_prior_tokens) / (t0 +
  rate_prior_tokens) (x69 uses the world trajectory's own rate, which the harness does not know); pace = log(median tokens on
  level at the game's clears / med_clock), 0 before a clear.

The swap rule (INDEX_SWAP): park the running game R for the best waiting game W iff
  I_W > (1 + margin_rel) I_R + margin_abs + cost_weight (I_R - I_R,parked)
  and R ran >= hysteresis_tokens since its last admission and executed an action since, and time left >= min_left_s +
  expected_wait_s (+ reorient_tokens / lane rate with add_reorient_time), and R is the worst running candidate, and a claim is
  free, and check_every_tokens have passed since R's last check.
  margin_rel alone (margin_abs 0, cost_weight 0) is x69's rule; cost_weight 1 charges the switch cost in index units (R's index
  drop from the parked flag: its re-orientation tokens and the context-loss factor)."""
from __future__ import annotations

import json
import math
import statistics
import threading

SCHEMA_ID = 'swap70-index-v1'
PROVISIONAL_ID = 'simalloc69.swap_policy/0.1'    # simalloc69's interim file (18:37 UTC), accepted as status 'provisional'
# The file layout is simalloc69's (its key names adopted, 18:41 UTC): section -> {key: (kind, default or REQ, meaning)}.
# load() returns the flat constants dict C that index() reads (C's keys in brackets).
REQ = object()
SCHEMA = {
    'hazard': {
        'bin_edges_tokens': ('edges', REQ, 'n + 1 increasing edges, the first 0; the last bin is open past its start [edges]'),
        'log_hazard_per_token': ('list', REQ, 'n log hazards per token (<= 0), one per bin [alpha]'),
        'beta_log_level': ('num', REQ, 'coefficient of ln(level) [beta_logL]'),
        'beta_pace': ('num', REQ, 'coefficient of the pace [beta_pace]'),
        'med_clock_tokens': ('pos', REQ, 'pace = ln(median tokens on level at the clears / this) [med_clock]'),
        'tail_scale': ('pos', 1.0, 'hazard multiplier from tail_from_tokens on [tail_scale]'),
        'tail_from_tokens': ('nonneg', 300000.0, 'tokens [tail_from]')},
    'value': {
        'mode': ('mode', None, "'completion' (V = w_L s_level) | 'official' (s = min(score_cap, 100 (b_hat / actions)^2)); "
                               "default decision.value_mode, else 'completion' [kind]"),
        's_level': ('pos', REQ, 'completion value of a level, also the chain levels\' value [s_fresh]'),
        'b_hat_prior_actions': ('pos', REQ, 'b_hat before the game clears a level (official mode) [b0]'),
        'actions_per_token_prior': ('pos', REQ, 'the action rate before the level has data [rate_prior]'),
        'chain_horizon_tokens': ('pos', REQ, 'tokens per further level in the chain [chain_horizon]'),
        'score_cap': ('pos', 115.0, 'per-level score cap, official mode [score_cap]'),
        'rate_prior_tokens': ('nonneg', 20000.0, 'prior weight of the rate in tokens [rate_prior_tokens]')},
    'index': {
        'horizons_tokens': ('incr', REQ, 'window lengths D [horizons]'),
        'min_tokens_left': ('nonneg', 5000.0, 'index 0 at or below this many tokens left [min_tokens_left]'),
        'default_n_levels': ('int', 7, 'levels assumed when a game reports none [default_n_levels]'),
        'integration_step_tokens': ('nonneg', 0.0, "simalloc69's grid (information: this module integrates exactly)")},
    'switch': {
        'switch_cost_tokens.mean': ('nonneg', REQ, "a parked game's return cost c [reorient_tokens]"),
        'context_loss_hazard_factor.policy': ('pos', REQ, "a parked game's hazard factor [theta]"),
        'notes_turn_tokens.q50': ('nonneg', 0.0, 'the notes turn (information) [notes_tokens]')},
    'lane': {
        'tps0': ('pos', 2050.0 / 36.0, "tokens / s per lane at the start (simalloc69 Sim: R36 / 36) [lane_tps0]"),
        'decline_per_h': ('nonneg', 0.027, 'fraction per hour [lane_decline]'),
        'floor': ('frac', 0.5, 'never below floor x tps0 [lane_floor]')},
    'decision': {
        'swap_out_if_enabled.margin': ('nonneg', REQ, 'I_W > (1 + margin) I_R + ... [margin_rel]'),
        'swap_out_if_enabled.hysteresis_tokens': ('nonneg', REQ, 'tokens since admission before a park [hysteresis_tokens]'),
        'swap_out_if_enabled.min_time_left_s': ('nonneg', REQ, 'no park with less time left [min_left_s]'),
        'swap_out_if_enabled.margin_abs': ('nonneg', 0.0, 'index units added to the threshold [margin_abs]'),
        'swap_out_if_enabled.cost_weight': ('nonneg', 0.0, 'weight of I_R - I_R,parked in the threshold [cost_weight]'),
        'swap_out_if_enabled.expected_wait_s': ('nonneg', 0.0, 'added to min_time_left_s [expected_wait_s]'),
        'swap_out_if_enabled.park_new_only': ('bool', False, 'compare with never-started games only [park_new_only]'),
        'swap_out_if_enabled.add_reorient_time': ('bool', False, "add the switch cost / lane rate to the time floor (the user's "
                                                                  "'expected wait plus re-orientation'; x69: false) [add_reorient_time]"),
        'check_every_tokens': ('nonneg', 0.0, "a game's swap check at most once per this many tokens [check_every_tokens]")},
}
FLAT = {('hazard', 'bin_edges_tokens'): 'edges', ('hazard', 'log_hazard_per_token'): 'alpha',
        ('hazard', 'beta_log_level'): 'beta_logL', ('hazard', 'beta_pace'): 'beta_pace', ('hazard', 'med_clock_tokens'): 'med_clock',
        ('hazard', 'tail_scale'): 'tail_scale', ('hazard', 'tail_from_tokens'): 'tail_from', ('value', 'mode'): 'kind',
        ('value', 's_level'): 's_fresh', ('value', 'b_hat_prior_actions'): 'b0', ('value', 'actions_per_token_prior'): 'rate_prior',
        ('value', 'chain_horizon_tokens'): 'chain_horizon', ('value', 'score_cap'): 'score_cap',
        ('value', 'rate_prior_tokens'): 'rate_prior_tokens', ('index', 'horizons_tokens'): 'horizons',
        ('index', 'min_tokens_left'): 'min_tokens_left', ('index', 'default_n_levels'): 'default_n_levels',
        ('index', 'integration_step_tokens'): 'integration_step', ('switch', 'switch_cost_tokens.mean'): 'reorient_tokens',
        ('switch', 'context_loss_hazard_factor.policy'): 'theta', ('switch', 'notes_turn_tokens.q50'): 'notes_tokens',
        ('lane', 'tps0'): 'lane_tps0', ('lane', 'decline_per_h'): 'lane_decline', ('lane', 'floor'): 'lane_floor',
        ('decision', 'swap_out_if_enabled.margin'): 'margin_rel', ('decision', 'swap_out_if_enabled.hysteresis_tokens'): 'hysteresis_tokens',
        ('decision', 'swap_out_if_enabled.min_time_left_s'): 'min_left_s', ('decision', 'swap_out_if_enabled.margin_abs'): 'margin_abs',
        ('decision', 'swap_out_if_enabled.cost_weight'): 'cost_weight',
        ('decision', 'swap_out_if_enabled.expected_wait_s'): 'expected_wait_s',
        ('decision', 'swap_out_if_enabled.park_new_only'): 'park_new_only', ('decision', 'check_every_tokens'): 'check_every_tokens',
        ('decision', 'swap_out_if_enabled.add_reorient_time'): 'add_reorient_time'}


def _bad(path, why):
    raise ValueError('index file: %s %s' % (path, why))


def _num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def _check(path, v, kind):
    if kind in ('list', 'edges', 'incr'):
        if not isinstance(v, list) or not v or not all(_num(x) for x in v):
            _bad(path, 'must be a non-empty list of finite numbers')
        if kind == 'edges' and (len(v) < 2 or v[0] != 0 or any(b <= a for a, b in zip(v, v[1:]))):
            _bad(path, 'must start at 0 and increase (n + 1 edges for n bins)')
        if kind == 'incr' and (v[0] <= 0 or any(b <= a for a, b in zip(v, v[1:]))):
            _bad(path, 'must be positive and increasing')
        return
    if kind == 'mode':
        if v not in ('completion', 'official'):
            _bad(path, "must be 'completion' or 'official'")
        return
    if kind == 'bool':
        if not isinstance(v, bool):
            _bad(path, 'must be true or false')
        return
    if not _num(v):
        _bad(path, 'must be a finite number')
    if kind == 'int' and (int(v) != v or v < 1):
        _bad(path, 'must be a positive integer')
    if kind in ('pos', 'frac') and v <= 0:
        _bad(path, 'must be > 0')
    if kind == 'frac' and v > 1:
        _bad(path, 'must be <= 1')
    if kind == 'nonneg' and v < 0:
        _bad(path, 'must be >= 0')


def _get(d, dotted):
    for part in dotted.split('.'):
        if not isinstance(d, dict) or part not in d:
            return REQ
        d = d[part]
    return d


def validate(F, path_name='(file)'):
    """the file's JSON object -> the flat constants dict C (ValueError names the first bad key)"""
    if not isinstance(F, dict):
        _bad('(top)', 'must be a JSON object')
    sid = str(F.get('schema') or '')
    if sid == SCHEMA_ID:
        status = F.get('status')
        if status not in ('placeholder', 'provisional', 'fitted'):
            _bad('status', "must be 'placeholder', 'provisional' or 'fitted'")
    elif sid.startswith(PROVISIONAL_ID):
        status = 'provisional'
    else:
        _bad('schema', 'must be %r (or simalloc69\'s %r)' % (SCHEMA_ID, PROVISIONAL_ID))
    C = dict(schema=sid, status=status, name=str(F.get('name') or path_name), written_utc=F.get('written_utc'))
    for sec, spec in SCHEMA.items():
        for key, (kind, default, _) in spec.items():
            v = _get(F.get(sec) if isinstance(F.get(sec), dict) else {}, key)
            if v is REQ:
                if default is REQ:
                    _bad('%s.%s' % (sec, key), 'is missing')
                v = default
            if v is None and kind == 'mode':
                v = _get(F.get('decision') or {}, 'value_mode')
                v = 'completion' if v is REQ or v is None else v
            _check('%s.%s' % (sec, key), v, kind)
            C[FLAT[(sec, key)]] = v
    e, a = C['edges'], C['alpha']
    if len(a) != len(e) - 1:
        _bad('hazard.log_hazard_per_token', 'needs one value per bin (%d edges -> %d bins)' % (len(e), len(e) - 1))
    if any(x > 0 for x in a):
        _bad('hazard.log_hazard_per_token', 'is a log hazard per token (each <= 0)')
    C['edges'] = list(e[:-1])                          # bin starts; the last bin is open
    C['default_n_levels'] = int(C['default_n_levels'])
    return C


def load(path):
    """the index file at path -> the flat constants dict (ValueError when invalid)"""
    import os
    with open(path) as f:
        try:
            F = json.load(f)
        except json.JSONDecodeError as exc:
            raise ValueError('index file: not JSON (%s)' % exc) from None
    return validate(F, os.path.basename(path))


def describe():
    """the schema as text lines (for the hand-back and CHANGES)"""
    out = []
    for sec, spec in SCHEMA.items():
        for key, (kind, default, meaning) in spec.items():
            out.append('%s.%s  %s  %s  %s' % (sec, key, kind, 'required' if default is REQ else 'default %r' % (default,),
                                              meaning))
    return out


# ------------------------------------------------------------------------------------------------------------- the index
def lane_tps(C, elapsed_s):
    return C['lane_tps0'] * max(C['lane_floor'], 1.0 - C['lane_decline'] * max(0.0, float(elapsed_s or 0.0)) / 3600.0)


def _breaks(C):
    pts = sorted(set(list(C['edges']) + [C['tail_from']]))
    rates = []
    for p in pts:
        b = max(i for i, x in enumerate(C['edges']) if x <= p)
        rates.append(math.exp(C['alpha'][b]) * (C['tail_scale'] if p >= C['tail_from'] else 1.0))
    return pts, rates


def window(C, t0, D, scale, _br=None):
    """(P, E): the probability of a clear in the next D tokens from t0 on the level, and the expected tokens spent in that
    window (the integral of the conditional survival), for the hazard x scale; exact for the piecewise-constant hazard"""
    pts, rates = _br or _breaks(C)
    t, end, logS, E = float(t0), float(t0) + max(0.0, float(D)), 0.0, 0.0
    while t < end - 1e-9:
        i = max(k for k, p in enumerate(pts) if p <= t)
        nxt = pts[i + 1] if i + 1 < len(pts) else math.inf
        b = min(end, nxt)
        r = rates[i] * scale
        d = b - t
        S = math.exp(logS)
        E += S * d if r <= 0 else S * (1.0 - math.exp(-r * d)) / r
        logS -= r * d
        t = b
    return 1.0 - math.exp(logS), E


def level_score(C, base, actions):
    return 0.0 if actions <= 0 else min(C['score_cap'], 100.0 * (base / actions) ** 2)


def state_scale(C, level, pace):
    return math.exp(C['beta_logL'] * math.log(max(int(level), 1)) + C['beta_pace'] * pace)


def pace_of(C, cleared):
    clocks = [float(c[0]) for c in (cleared or ()) if c and c[0] is not None]
    if not clocks:
        return 0.0
    return math.log(max(statistics.median(clocks), 1.0) / C['med_clock'])


def b_hat_of(C, cleared):
    acts = [float(c[1]) for c in (cleared or ()) if c and c[1] is not None]
    return float(statistics.median(acts)) if acts else float(C['b0'])


def index(C, s, detail=False):
    """the expected score gain per token of state s (a dict):
      level, n_levels (None: default_n_levels), tokens_on_level (over all the level's conversations), actions_on_level,
      cleared [[tokens on level at the clear, actions on the level at the clear], ...] for the game's cleared levels,
      time_left_s (None: no deadline), elapsed_s, cap_left (None: no cap or a soft cap), never_started, parked.
    detail=True returns (index, components)."""
    L = max(1, int(s.get('level') or 1))
    n = int(s.get('n_levels') or C['default_n_levels'])
    n = max(n, L)
    never = bool(s.get('never_started'))
    parked = bool(s.get('parked')) and not never
    cost = float(C['reorient_tokens']) if parked else 0.0
    elapsed = float(s.get('elapsed_s') or 0.0)
    tl = s.get('time_left_s')
    tok_left = math.inf if tl is None else max(0.0, float(tl)) * lane_tps(C, elapsed)
    if s.get('cap_left') is not None:
        tok_left = min(tok_left, max(0.0, float(s['cap_left'])))
    tok_left -= cost
    comp = dict(level=L, n_levels=n, cost=cost, tok_left=None if math.isinf(tok_left) else round(tok_left, 1),
                never_started=never, parked=parked)
    if tok_left <= C['min_tokens_left']:
        comp.update(index=0.0, why='no_tokens_left')
        return (0.0, comp) if detail else 0.0
    cleared = [] if never else list(s.get('cleared') or [])
    t0 = 0.0 if never else max(0.0, float(s.get('tokens_on_level') or 0))
    a_now = 0.0 if never else max(0.0, float(s.get('actions_on_level') or 0))
    k = float(C['rate_prior_tokens'])
    rate_a = (C['rate_prior'] if (never or t0 + k <= 0) else (a_now + C['rate_prior'] * k) / (t0 + k))
    pace = pace_of(C, cleared)
    b_hat = b_hat_of(C, cleared)
    c = state_scale(C, L, pace) * (C['theta'] if parked else 1.0)
    W = n * (n + 1) / 2.0
    br = _breaks(C)
    chain = []
    for j in range(L + 1, n + 1):
        pj, ej = window(C, 0.0, C['chain_horizon'], state_scale(C, j, pace), br)
        chain.append((pj, ej, j / W * C['s_fresh']))
    best, arg = 0.0, None
    for D0 in C['horizons']:
        D = min(D0, tok_left)
        P, E = window(C, t0, D, c, br)
        a_tot = a_now + rate_a * E
        vl = L / W * (level_score(C, b_hat, max(a_tot, 1.0)) if C['kind'] == 'official' else C['s_fresh'])
        num, den = P * vl, cost + E
        r = num / den if den > 0 else 0.0
        if r > best:
            best, arg = r, dict(D=D, k=0, P=P, E=E, a_total=a_tot, v_level=vl)
        reach, tok = P, E
        for i, (pj, ej, vj) in enumerate(chain, 1):
            if tok + ej > tok_left:
                break
            den += reach * ej
            tok += ej
            reach *= pj
            num += reach * vj
            r = num / den
            if r > best:
                best, arg = r, dict(D=D, k=i, P=P, E=E, a_total=a_tot, v_level=vl)
        if D >= tok_left:
            break
    comp.update(index=best, t0=t0, a_now=a_now, rate_a=rate_a, pace=pace, b_hat=b_hat, scale=c, W=W, chain=len(chain),
                **({k_: (round(x, 6) if isinstance(x, float) else x) for k_, x in arg.items()} if arg else {}))
    return (best, comp) if detail else best


def swap_rule(C, I_R, I_R_parked, I_W):
    """(park, threshold): the index comparison alone (the gates are the agent's)"""
    thr = (1.0 + C['margin_rel']) * I_R + C['margin_abs'] + C['cost_weight'] * max(0.0, I_R - I_R_parked)
    return I_W > thr + 1e-15, thr


def min_left_s(C, elapsed_s):
    return C['min_left_s'] + C['expected_wait_s'] + (C['reorient_tokens'] / max(1e-9, lane_tps(C, elapsed_s))
                                                      if C['add_reorient_time'] else 0.0)


# ------------------------------------------------------------------------------------------------------- shared state
class _Registry:
    """gid -> the game's latest published entry (index, running, eligible, ...), one lock"""

    def __init__(self):
        self.lock = threading.Lock()
        self.d = {}

    def publish(self, gid, **entry):
        with self.lock:
            self.d[gid] = dict(self.d.get(gid) or {}, **entry)

    def get(self, gid):
        with self.lock:
            return dict(self.d.get(gid) or {})

    def running(self):
        with self.lock:
            return {g: dict(e) for g, e in self.d.items() if e.get('running')}

    def clear(self):
        with self.lock:
            self.d.clear()


class _Claims:
    """pending index / cap parks: decided (notes turn running), lane not yet given up"""

    def __init__(self):
        self.lock = threading.Lock()
        self.pending = set()

    def take(self, gid, limit):
        with self.lock:
            if gid in self.pending:
                return True
            if len(self.pending) >= limit:
                return False
            self.pending.add(gid)
            return True

    def drop(self, gid):
        with self.lock:
            self.pending.discard(gid)

    def count(self):
        with self.lock:
            return len(self.pending)


class _Admits:
    """INDEX_ADMIT receipts: every admission from the queue (game, kind, index, N, time left, the competing entries)"""

    def __init__(self, cap=20000):
        self.lock = threading.Lock()
        self.rows, self.cap, self.n = [], cap, 0

    def note(self, row):
        with self.lock:
            self.n += 1
            if len(self.rows) < self.cap:
                self.rows.append(row)

    def records(self):
        with self.lock:
            return list(self.rows)

    def count(self):
        with self.lock:
            return self.n


REGISTRY = _Registry()
CLAIMS = _Claims()
ADMITS = _Admits()
TLS = threading.local()
