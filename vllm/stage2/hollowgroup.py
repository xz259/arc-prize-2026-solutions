"""stage2_v1 'hollowgroup' part (2026-10-04 02:55; user 00:15 "always group the tool turn with the assistant turn that
emitted it", agreed for the hollow rule; ~02:45 "It is a small change just add it back").

Production's hollow rule (history_eviction._hollow_blocks, ARC3_HOLLOW_FRACTION, floor 8) judges every turn segment on its
own: an assistant segment and the tool segment holding its tool's output are separate blocks. Here a tool segment that
starts where the preceding assistant segment ends joins it, so the rule judges the unit (assistant + its tool turn)
together: the unit goes whole once its remaining content (formatting tokens excluded, as before) falls to
max(8, (1 - fraction) x content). Observation segments and a tool segment with no adjacent assistant segment before it
are unchanged. Only the hollow rule sees the grouped list; frequency selection, protection and receipts' segment lists
are untouched. receipt['hollow_grouping'] = 'assistant+tool'. An error leaves production's per-segment rule in place
(counted).

Sweep (2026-10-06, after run 381's engine death; user ~04:10, Fable's review): production's rule only re-tests the
units this selection touched, so a unit that fell under the threshold another way (its remaining content groups never
exposed again, an evicted image, age-cap ranges merged after planning, an oldest-split fallback that skips the hollow
step) was never retired. Its formatting tokens stayed resident beside evicted neighbours, and those partial 4-token QSA
groups are the group padding that tripped the engine's turn check (g046: 553 of ~713 tokens of padding sat in units
with <= 25% left). The sweep applies the same test, unchanged, to every unit inside [lo, hi) that the selection did not
touch; a hollow one goes whole (its remaining pieces, formatting included). Protected ranges named in the receipt are
subtracted as a guard. receipt['hollow_sweep_units'] / ['hollow_sweep_tokens']. ARC3_S2_HOLLOW_SWEEP=0 turns it off;
an error leaves the unswept result (counted)."""
import os
from bisect import bisect_left, bisect_right

PINS = {'history_eviction': '88a5d9aef289c87b485104a1cf8d08701a96600198a70196ccd43cd7803c8fc0'}
SWEEP = os.environ.get('ARC3_S2_HOLLOW_SWEEP', '1') != '0'
# 2026-10-15 (final profile, plan time grew with the session): the receipt-only hollow_sweep_diag walked every 4-token group
# of every resident unit and counted every unit's formatting tokens one by one. It is opt-in now (ARC3_S2_HOLLOW_DIAG=1);
# without it the sweep applies production's own early test (remaining content cannot fall under the limit -> skip the
# per-token count), so its decisions are unchanged and its cost no longer grows with the session's formatting tokens.
DIAG = os.environ.get('ARC3_S2_HOLLOW_DIAG', '0') == '1'
STATS = dict(calls=0, units_merged=0, errors=0, last_error=None, sweeps=0, sweep_units=0, sweep_tokens=0,
             sweep_errors=0, sweep_last_error=None)
PROTECTED_KEYS = ('protected_prefix_range', 'protected_tail_range', 'protected_active_range', 'pair_guard')


def group_units(segments):
    """[kind, start, end] lists with every tool segment that starts where an assistant segment ends merged into it."""
    out = []
    for seg in segments:
        k, a, b = seg[0], int(seg[1]), int(seg[2])
        if k == 'tool' and out and out[-1][0] == 'assistant' and out[-1][2] == a:
            out[-1] = ['assistant', out[-1][1], b]
        else:
            out.append([k, a, b])
    return out


def _protected(receipt):
    out = []
    for key in PROTECTED_KEYS:
        r = receipt.get(key)
        if isinstance(r, (list, tuple)) and len(r) == 2 and r[0] is not None and r[1] is not None and r[1] > r[0]:
            out.append((int(r[0]), int(r[1])))
    pin = receipt.get('observation_pin')
    r = pin.get('range') if isinstance(pin, dict) else None
    if isinstance(r, (list, tuple)) and len(r) == 2 and r[1] > r[0]:
        out.append((int(r[0]), int(r[1])))
    return out


def sweep(mod, units, hpieces, history, structural, lo, hi, receipt):
    """(pieces, units): the remaining pieces of every unit in [lo, hi) not touched by `history` that production's hollow
    test retires, with the same fraction, floor and formatting rule (history_eviction._hollow_blocks)."""
    if not hpieces:
        return [], 0
    frac = mod._hollow_fraction()
    H = mod._Cover(sorted(hpieces))
    Y = mod._Cover(sorted(history)) if history else None
    starts = [int(u[1]) for u in units]
    touched = set()
    for a, b in history or ():
        i = max(0, bisect_right(starts, a) - 1)
        while i < len(units) and starts[i] < b:
            if int(units[i][2]) > a:
                touched.add(starts[i])
            i += 1
    extra, n = [], 0
    # receipt-only diagnostic (Fable's final review, Q6): [units, padding] for the units in [lo, hi) still holding tokens
    # after this selection: those it touched, and the untouched ones by remaining-content share
    diag = {k: [0, 0] for k in ('touched', 'le25', 'le50', 'le75', 'gt75')}
    for u in units:
        s0, s1 = int(u[1]), int(u[2])
        if s0 < lo or s1 > hi or s1 <= s0:
            continue
        res_after = H.count(s0, s1) - (Y.count(s0, s1) if Y else 0)
        if res_after <= 0:
            continue
        pad = _unit_padding(H, Y, s0, s1) if DIAG else 0
        if s0 in touched:
            diag['touched'][0] += 1
            diag['touched'][1] += pad
            continue
        j0, j1 = bisect_left(structural, s0), bisect_left(structural, s1)
        n_struct = j1 - j0
        content_total = (s1 - s0) - n_struct
        if res_after >= s1 - s0 or content_total <= 0:
            diag['gt75'][0] += 1
            diag['gt75'][1] += pad
            continue
        limit = max(mod.HOLLOW_FLOOR, int((1.0 - frac) * content_total))
        if not DIAG and res_after - n_struct > limit:   # left >= res_after - n_struct: cannot be hollow (production's test)
            continue
        struct_after = sum(1 for q in structural[j0:j1] if H.has(q) and not (Y is not None and Y.has(q)))
        left = res_after - struct_after
        share = left / content_total
        key = 'le25' if share <= .25 else 'le50' if share <= .5 else 'le75' if share <= .75 else 'gt75'
        diag[key][0] += 1
        diag[key][1] += pad
        if left <= limit:
            extra.extend(mod._subtract(H.pieces(s0, s1), Y.pieces(s0, s1) if Y is not None else []))
            n += 1
    if DIAG:
        receipt['hollow_sweep_diag'] = diag
    guard = _protected(receipt)
    if guard and extra:
        extra = mod._subtract(extra, guard)
    return extra, n


def _unit_padding(H, Y, s0, s1):
    """Group padding after this selection in the 4-token groups that start inside [s0, s1): the evicted members of each
    group that still holds a resident member (the engine's deleted - 4 x whole groups, per unit)."""
    pad = 0
    for g in range((s0 + 3) // 4, (s1 + 3) // 4):
        a = 4 * g
        r = H.count(a, a + 4) - (Y.count(a, a + 4) if Y is not None else 0)
        if 0 < r < 4:
            pad += 4 - r
    return pad


def patch_history_eviction(mod):
    original = mod._hollow_blocks

    def _hollow_blocks(segments, hpieces, history, structural, lo, hi, receipt):
        STATS['calls'] += 1
        try:
            grouped = group_units(segments)
            STATS['units_merged'] += len(segments) - len(grouped)
        except Exception as exc:                        # noqa: BLE001  production's per-segment rule stands
            STATS['errors'] += 1
            STATS['last_error'] = repr(exc)[:300]
            return original(segments, hpieces, history, structural, lo, hi, receipt)
        out = original(grouped, hpieces, history, structural, lo, hi, receipt)
        receipt['hollow_grouping'] = 'assistant+tool'
        if SWEEP:
            try:
                more, n = sweep(mod, grouped, hpieces, history, structural, lo, hi, receipt)
                if out and more:
                    more = mod._subtract(more, out)
                k = mod._count(more)
                STATS['sweeps'] += 1
                STATS['sweep_units'] += n
                STATS['sweep_tokens'] += k
                receipt['hollow_sweep_units'] = n
                receipt['hollow_sweep_tokens'] = k
                if more:
                    out = list(out) + list(more)
            except Exception as exc:                    # noqa: BLE001  the unswept result stands
                STATS['sweep_errors'] += 1
                STATS['sweep_last_error'] = repr(exc)[:300]
        return out

    mod._hollow_blocks = _hollow_blocks


PATCHERS = {'history_eviction': patch_history_eviction}
