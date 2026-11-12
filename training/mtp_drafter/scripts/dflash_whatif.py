"""dflash_whatif.py (2026-10-13, Claude fork 5d20b23b): how much better DFlash's slots would have to be to reach MTP3 in
live-process throughput. Takes DFlash 15K's per-anchor acceptance at every val50 anchor (dflash_dense_anchors.py) and cuts
each slot's rejection probability by a factor r at every anchor, c' = 1 - (1 - c)(1 - r), then computes the renewal
(live) tau (live_tau.py) and throughput = tau / step (DFlash step 0.89 + 0.035 K, the MTP3 step = 1). Feature probes put
the MTP-input gain at slot 1 at r ~ 0.09-0.10 (MTP streams added to the taps) to 0.14-0.16 (streams instead of the taps).
usage: python dflash_whatif.py ANCHORS.pt [--modes sel_srv_T10 sel_srv_T06] [--K 4 5 6 8] [--r 0 0.1 ...]"""
import argparse

import torch

from live_tau import renewal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('anchors')
    ap.add_argument('--modes', nargs='+', default=['sel_srv_T10', 'sel_srv_T06'])
    ap.add_argument('--K', type=int, nargs='+', default=[4, 5, 6, 8])
    ap.add_argument('--r', type=float, nargs='+', default=[0.0, 0.1, 0.15, 0.25, 0.35, 0.5])
    a = ap.parse_args()
    data = torch.load(a.anchors)
    for mode in a.modes:
        for r in a.r:
            sess = []
            s1 = n1 = 0.0
            for sid, d in data.items():
                m = d['slot_mask'].double()
                c = m * (1.0 - (1.0 - d[mode].double() * m) * (1.0 - r))
                sess.append((d['anchors'], c))
                s1 += float(c[:, 0].sum()); n1 += float(m[:, 0].sum())
            row = []
            for K in a.K:
                tot = [0.0, 0.0, 0.0, 0]
                for anchors, c in sess:
                    res = renewal(anchors, c, K)
                    for i in range(4):
                        tot[i] += res[i]
                tau = tot[0] / tot[1]
                row.append('K %d: uniform L %.3f, live tau %.3f, throughput %.3f' % (K, tot[2] / tot[3], tau, tau / (0.89 + 0.035 * K)))
            print('%s r %.2f (slot 1 %.3f) | %s' % (mode, r, s1 / n1, ' | '.join(row)), flush=True)


if __name__ == '__main__':
    main()
