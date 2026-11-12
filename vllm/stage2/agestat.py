"""stage2_v1 'agestat' part (2026-10-03 23:05; rebase design option 3, user ~22:37: how much QSA selection an age cap
would remove). Diagnostic only: at every triggered eviction plan, the selection mass and the resident groups by age
(logical end - group start) from the frequency snapshot the planner already holds (the layer-mean selection frequency
per resident 4-token group, as _plan_split_qsa ranks them). Adds receipt['age_stats']:
  end, prefix_tokens, edges (age bucket edges in tokens), mass_share (share of the snapshot's selection mass per age
  bucket, system-prompt groups excluded), groups (resident groups per bucket), prefix_mass_share, max_age.
Planning is unchanged; an error is counted in STATS and leaves production's receipt as it was."""

PINS = {'history_eviction': '88a5d9aef289c87b485104a1cf8d08701a96600198a70196ccd43cd7803c8fc0'}
import os

EDGES = tuple(int(x) for x in os.environ.get('ARC3_S2_AGESTAT_EDGES', '25000,50000,100000,150000,200000').split(','))
STATS = dict(calls=0, stats=0, errors=0, last=None, last_error=None)


def age_stats(frequency, end, prefix, ratio):
    """The age profile of one frequency snapshot; None without a matrix snapshot."""
    import numpy as np
    m, ids = frequency.get('matrix'), frequency.get('group_ids_array')
    if m is None or ids is None or len(ids) == 0:
        return None
    ids = np.asarray(ids, dtype=np.int64)
    with np.errstate(invalid='ignore'):
        f = np.nanmean(np.asarray(m, dtype=np.float64), axis=0)
    f = np.where(np.isnan(f), 0.0, f)
    pos = ids * ratio
    pre = pos < prefix
    age = end - pos
    total = float(f.sum())
    bucket = np.searchsorted(np.asarray(EDGES), age, side='right')
    share = lambda sel: round(float(f[sel].sum()) / total, 6) if total > 0 else None
    return dict(end=int(end), prefix_tokens=int(prefix), edges=list(EDGES),
                mass_share=[share((bucket == b) & ~pre) for b in range(len(EDGES) + 1)],
                groups=[int(((bucket == b) & ~pre).sum()) for b in range(len(EDGES) + 1)],
                prefix_mass_share=share(pre), max_age=int(age[~pre].max()) if (~pre).any() else 0)


def patch_history_eviction(mod):
    original = mod.plan_history_eviction

    def plan_history_eviction(policy, registry, ledger, **kw):
        decision = original(policy, registry, ledger, **kw)
        STATS['calls'] += 1
        try:
            freq = kw.get('frequency')
            if freq is not None and decision.receipt.get('triggered'):
                from qsa_frequency import RATIO
                obs = registry.observations
                st = age_stats(freq, ledger.logical_tokens, obs[0].start if obs else 0, RATIO)
                if st is not None:
                    decision.receipt['age_stats'] = st
                    STATS['stats'] += 1
                    STATS['last'] = st
        except Exception as exc:                        # noqa: BLE001  a diagnostic never changes planning
            STATS['errors'] += 1
            STATS['last_error'] = repr(exc)[:300]
        return decision

    mod.plan_history_eviction = plan_history_eviction


PATCHERS = {'history_eviction': patch_history_eviction}
