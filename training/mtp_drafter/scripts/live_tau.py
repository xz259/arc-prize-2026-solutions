"""live_tau.py (2026-10-12, Claude fork 5d20b23b): the live-process (renewal) acceptance length from per-anchor acceptance
probabilities, against the uniform-anchor mean that the validation metrics report.

Live decoding drafts only at round starts: a round at anchor a accepts j slots with probability
p_j = prod_{i<=j} c_i * (1 - c_{j+1}) (c_i = slot i's acceptance given its predecessors, times its assistant mask;
c_{K+1} = 0), emits j + 1 tokens and the next round starts at a + j + 1. Rounds therefore start less often inside easy
stretches (crossed in long accepts) than uniform anchors assume. The visiting probabilities v(a) follow by forward
dynamic programming over the anchors of each window (mass landing on a non-anchor row, e.g. past the end of an assistant
segment, moves to the next anchor: the next segment's first round); the live expected length is
tau_live = sum_a v(a) L(a) / sum_a v(a) with L(a) = 1 + sum_k prod_{j<=k} c_j, against the uniform mean of L(a).
Input: mtp_eval_val50.py --save-anchors output (per session: anchors, slot_mask, per-mode acceptance [A, K]).
usage: python live_tau.py ANCHORS.pt [--modes prob_srv_T10 ...] [--K 3 4 5]"""
import argparse

import torch


def renewal(anchors, c, K):
    """anchors [A] sorted row indices; c [A, K] masked conditional slot acceptance. -> (sum v L, sum v, sum L, A)."""
    A = anchors.shape[0]
    c = c[:, :K].double()
    pref = torch.cumprod(c, dim=1)                                             # P(slots 1..k all accepted)
    L = 1.0 + pref.sum(1)
    pj = torch.cat([torch.ones(A, 1, dtype=torch.float64), pref], dim=1)       # P(at least j accepted), j = 0..K
    pj = pj - torch.cat([pref, torch.zeros(A, 1, dtype=torch.float64)], dim=1)  # P(exactly j accepted)
    idx = {int(a): i for i, a in enumerate(anchors.tolist())}
    pos = anchors.tolist()
    v = [0.0] * A
    v[0] = 1.0
    pending = {}
    carry = 0.0
    for i in range(A):
        a = pos[i]
        # mass that landed on rows before a that are not anchors moves here (the next round start)
        for p in [p for p in pending if p < a]:
            carry += pending.pop(p)
        v[i] += pending.pop(a, 0.0) + carry
        carry = 0.0
        vi = v[i]
        if vi == 0.0:
            continue
        for j in range(K + 1):
            nxt = a + j + 1
            m = vi * float(pj[i, j])
            if m == 0.0:
                continue
            if nxt in idx:
                v[idx[nxt]] += m
            else:
                pending[nxt] = pending.get(nxt, 0.0) + m
    vt = torch.tensor(v, dtype=torch.float64)
    return float((vt * L).sum()), float(vt.sum()), float(L.sum()), A


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('anchors')
    ap.add_argument('--modes', nargs='+', default=['prob_srv_T10', 'prob_srv_T06', 'greedy_srv_T10', 'ceil_srv_T10'])
    ap.add_argument('--K', type=int, nargs='+', default=[3, 4, 5])
    a = ap.parse_args()
    data = torch.load(a.anchors)
    for mode in a.modes:
        for K in a.K:
            tot = [0.0, 0.0, 0.0, 0]
            for sid, d in data.items():
                c = d[mode] * d['slot_mask'].to(d[mode].dtype)
                r = renewal(d['anchors'], c, K)
                for i in range(4):
                    tot[i] += r[i]
            print('%-14s K %d: uniform-anchor L %.4f | live renewal tau %.4f (ratio %.4f; anchors %d)' % (
                mode, K, tot[2] / tot[3], tot[0] / tot[1], (tot[0] / tot[1]) / (tot[2] / tot[3]), tot[3]), flush=True)


if __name__ == '__main__':
    main()
