"""stage2_v1 'agecap' part (2026-10-04 00:30; rebase design option 3, user ~23:45-00:20: per-tier age gap, full
interactions, edge rules 00:15). At every triggered eviction plan, rows older than the age gap A
(ARC3_S2_AGECAP tokens; 0 = off) are removed on top of production's selection, so no resident row is older than about
A + one eviction interval and the system prompt can sit just behind the oldest row.
Units (user 00:15): an interaction = an observation block up to the next observation; an assistant unit = an
assistant segment with the tool segment(s) that follow it. With B = logical end - A:
  - every interaction starting before B, except the latest, is removed whole (its resident parts);
  - in the latest interaction, when it starts before B, the whole assistant units starting before B are removed,
    never its last unit (in progress) and never its observation (the pinned latest observation);
  - never the system prompt [0, prefix), the planner's protected tail (receipt protected_tail_range, pair-guarded) or
    the pinned observation.
The added ranges are merged into the decision (one ledger plan with production's protection), and receipt['age_cap']
records the gap, B, the interactions / units and tokens added. Untriggered plans are unchanged (the cap rides on
production's evictions)."""
import os

PINS = {'history_eviction': '88a5d9aef289c87b485104a1cf8d08701a96600198a70196ccd43cd7803c8fc0'}
AGECAP = int(os.environ.get('ARC3_S2_AGECAP', '0'))
STATS = dict(plans=0, capped=0, interactions=0, units=0, tokens=0, errors=0, last=None, last_error=None)


def merge(ranges):
    """Sorted union of [a, b) ranges (overlaps and adjacency merged); empty ranges dropped."""
    out = []
    for a, b in sorted((int(a), int(b)) for a, b in ranges if b > a):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def age_ranges(observations, segments, logical, prefix, cap):
    """(ranges, interactions, units): the logical ranges the age gap removes (before resident / protection filtering)."""
    b = logical - cap
    if cap <= 0 or b <= prefix or not observations:
        return [], 0, 0
    out, n_int, n_units = [], 0, 0
    starts = [o.start for o in observations]
    for i, s in enumerate(starts[:-1]):
        if s < b:
            out.append((s, starts[i + 1])); n_int += 1
    last = starts[-1]
    if last < b:                                  # the latest interaction straddles B: whole units before B
        units = [int(x[1]) for x in segments if x[0] == 'assistant' and int(x[1]) >= last]
        for j, a in enumerate(units[:-1]):        # never the last unit (in progress)
            if a < b:
                out.append((a, units[j + 1])); n_units += 1
    return out, n_int, n_units


def patch_history_eviction(mod):
    original = mod.plan_history_eviction

    def plan_history_eviction(policy, registry, ledger, **kw):
        decision = original(policy, registry, ledger, **kw)
        if AGECAP <= 0:
            return decision
        STATS['plans'] += 1
        rec = decision.receipt
        if not rec.get('triggered'):
            return decision
        try:
            logical = ledger.logical_tokens
            obs = registry.observations
            prefix = obs[0].start if obs else 0
            segments = kw.get('segments') or ()
            ranges, n_int, n_units = age_ranges(obs, segments, logical, prefix, AGECAP)
            if not ranges:
                rec['age_cap'] = dict(cap=AGECAP, boundary=logical - AGECAP, tokens=0, interactions=0, units=0)
                return decision
            pin = None
            if mod._pin_observation_enabled() and not kw.get('incoming_observation'):
                pin = mod._pinned_observation(obs, segments, logical)
            tail = rec.get('protected_tail_range') or [max(0, logical - 4), logical]
            keep_out = [(0, prefix), (int(tail[0]), logical)] + ([tuple(pin)] if pin else [])
            resident = mod._subtract([(0, logical)], merge(list(ledger.evicted) + [tuple(r) for r in decision.ranges]))
            add = []
            for a, b in merge(ranges):
                add.extend(mod._intersections(a, b, resident))
            add = merge(mod._subtract(merge(add), merge(keep_out))) if add else []
            if not add:
                rec['age_cap'] = dict(cap=AGECAP, boundary=logical - AGECAP, tokens=0, interactions=n_int, units=n_units)
                return decision
            combined = merge((*decision.ranges, *add))
            history = merge((*decision.history_ranges, *add))
            protected = [(0, prefix)] if prefix else []
            tail4 = max(0, logical - 4)
            protected = [(0, logical)] if (protected and tail4 <= prefix) else protected + [(tail4, logical)]
            plan = ledger.plan(combined, protected=mod._union(protected, pin))
            tokens = sum(b - a for a, b in add)
            budget = policy.resident_token_budget
            inc = kw.get('incoming_tokens', 0)
            rec.update(total_removed_tokens=plan.removed_tokens, resident_tokens_after=plan.resident_after,
                       resident_tokens_after_input=plan.resident_after + inc,
                       minimum_output_capacity=budget - plan.resident_after - inc,
                       combined_ranges=[list(p) for p in combined], history_ranges=[list(p) for p in history],
                       age_cap=dict(cap=AGECAP, boundary=logical - AGECAP, tokens=tokens, interactions=n_int,
                                    units=n_units, ranges=[list(p) for p in add][:64]))
            STATS['capped'] += 1
            STATS['interactions'] += n_int
            STATS['units'] += n_units
            STATS['tokens'] += tokens
            STATS['last'] = rec['age_cap']
            return mod.HistoryDecision(combined, history, decision.suffix_ranges, plan, rec)
        except Exception as exc:                        # noqa: BLE001  never break planning: production's decision stands
            STATS['errors'] += 1
            STATS['last_error'] = repr(exc)[:300]
            rec['age_cap_error'] = repr(exc)[:200]
            return decision

    mod.plan_history_eviction = plan_history_eviction


PATCHERS = {'history_eviction': patch_history_eviction}
