"""paired_live_tau.py (2026-10-22, Claude main session 099768dc; the step-conditioned round's gate, user 10-22 V67_DESIGN 262, IE's
review): a PAIRED comparison of two heads scored on the same windows (mtp_eval_val50.py --save-anchors files, e.g. next55's 55
windows): per-window live renewal tau (live_tau.renewal: the window's sum v L / sum v), the per-window relative difference
d = tau_cand / tau_base - 1, its mean, SE and 90% t interval; the pooled delta (live_tau.py's headline) beside it; the per-slot
conditional acceptance (uniform over anchors, the trainer's alpha_k) deltas in points with their paired SE (the mechanism).

Gate (IE, the user's adoption, at --gate-mode / --gate-K, default prob_srv_T06 K 4):
  PASS       paired mean >= +0.5% and the 90% interval excludes 0 (lower bound > 0);
  NOTHING    the interval includes 0, |mean| < 0.25% and slots 2..4 each moved < 0.2 points (form B is next only if the trainer's
             val_stepemb_rms also plateaued: checked separately);
  WORSE      the whole interval below 0;
  UNCLEAR    anything else (e.g. significant but below the bar, or not significant with a larger mean / slot movement).
usage: python paired_live_tau.py BASE_ANCHORS.pt CAND_ANCHORS.pt [--modes prob_srv_T06 prob_srv_T10] [--K 3 4 5]
       [--gate-mode prob_srv_T06] [--gate-K 4] [--json OUT.json]"""
import argparse
import json
import math
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from live_tau import renewal  # noqa: E402


def t90(df):
    """Two-sided 90% t quantile (Cornish-Fisher; 1.6736 at df 54, 1.6449 as df -> inf)."""
    z = 1.6448536269514722
    return z + (z ** 3 + z) / (4 * df) + (5 * z ** 5 + 16 * z ** 3 + 3 * z) / (96 * df * df)


def mean_se(x):
    n = len(x)
    m = sum(x) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in x) / (n - 1)) if n > 1 else float('nan')
    return m, sd / math.sqrt(n)


def per_window(data, mode, K):
    """sid -> (tau_live, sum v, per-slot mean conditional acceptance [K] over the window's anchors)."""
    out = {}
    for sid, d in data.items():
        c = d[mode] * d['slot_mask'].to(d[mode].dtype)
        svl, sv, _, _ = renewal(d['anchors'], c, K)
        out[sid] = (svl / sv, sv, [float(c[:, k].double().mean()) for k in range(K)], svl)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('base'); ap.add_argument('cand')
    ap.add_argument('--modes', nargs='+', default=['prob_srv_T06', 'prob_srv_T10'])
    ap.add_argument('--K', type=int, nargs='+', default=[3, 4, 5])
    ap.add_argument('--gate-mode', default='prob_srv_T06'); ap.add_argument('--gate-K', type=int, default=4)
    ap.add_argument('--json', default='')
    a = ap.parse_args()
    B, C = torch.load(a.base), torch.load(a.cand)
    common = sorted(set(B) & set(C))
    if len(common) != len(B) or len(common) != len(C):
        print('WARNING: windows differ: base %d, cand %d, common %d' % (len(B), len(C), len(common)))
    for sid in common:
        assert torch.equal(B[sid]['anchors'], C[sid]['anchors']), 'anchors differ in %s: not the same windows' % sid
    B = {s: B[s] for s in common}; C = {s: C[s] for s in common}
    res = dict(base=a.base, cand=a.cand, windows=len(common), rows={})
    print('paired: %d windows; base %s; cand %s' % (len(common), a.base, a.cand))
    for mode in a.modes:
        for K in a.K:
            pb, pc = per_window(B, mode, K), per_window(C, mode, K)
            d = [pc[s][0] / pb[s][0] - 1 for s in common]
            m, se = mean_se(d)
            tq = t90(len(d) - 1)
            pooled_b = sum(pb[s][3] for s in common) / sum(pb[s][1] for s in common)
            pooled_c = sum(pc[s][3] for s in common) / sum(pc[s][1] for s in common)
            slots = []
            for k in range(K):
                dk = [pc[s][2][k] - pb[s][2][k] for s in common]
                mk, sek = mean_se(dk)
                slots.append(dict(slot=k + 1, base=sum(pb[s][2][k] for s in common) / len(common), delta_pts=100 * mk, se_pts=100 * sek))
            row = dict(mode=mode, K=K, pooled_base=pooled_b, pooled_cand=pooled_c, pooled_delta_pct=100 * (pooled_c / pooled_b - 1),
                       paired_mean_pct=100 * m, paired_se_pct=100 * se, ci90_pct=[100 * (m - tq * se), 100 * (m + tq * se)], slots=slots)
            res['rows']['%s_K%d' % (mode, K)] = row
            print('%-13s K %d: pooled tau %.4f -> %.4f (%+.2f%%) | paired mean %+.2f%% (SE %.2f%%, 90%% CI [%+.2f, %+.2f])' % (
                mode, K, pooled_b, pooled_c, row['pooled_delta_pct'], row['paired_mean_pct'], row['paired_se_pct'], *row['ci90_pct']))
            print('                slot acceptance (base, delta pts +- SE): ' + '; '.join(
                '%d %.4f %+.2f+-%.2f' % (s['slot'], s['base'], s['delta_pts'], s['se_pts']) for s in slots))
    g = res['rows'].get('%s_K%d' % (a.gate_mode, a.gate_K))
    if g is None:
        verdict = 'NONE (gate row not computed)'
    else:
        lo, hi = g['ci90_pct']
        mov = [abs(s['delta_pts']) for s in g['slots'][1:4]]
        if g['paired_mean_pct'] >= 0.5 and lo > 0:
            verdict = 'PASS'
        elif hi < 0:
            verdict = 'WORSE'
        elif lo <= 0 <= hi and abs(g['paired_mean_pct']) < 0.25 and all(x < 0.2 for x in mov):
            verdict = 'NOTHING (form B only if val_stepemb_rms plateaued; else extend)'
        else:
            verdict = 'UNCLEAR'
        verdict += ': %s K %d paired mean %+.2f%%, 90%% CI [%+.2f, %+.2f], slots 2-4 moved %s pts' % (
            a.gate_mode, a.gate_K, g['paired_mean_pct'], lo, hi, ', '.join('%+.2f' % s['delta_pts'] for s in g['slots'][1:4]))
    res['verdict'] = verdict
    print('GATE ' + verdict)
    if a.json:
        json.dump(res, open(a.json, 'w'), indent=1)


if __name__ == '__main__':
    main()
