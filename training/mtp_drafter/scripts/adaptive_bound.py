"""adaptive_bound.py (2026-10-12, Claude fork 5d20b23b): live-process throughput (tokens per unit of step time, the MTP3
step = 1) for fixed K and an ORACLE adaptive K per round (an upper bound for confidence-scheduled verification: the
round's K is chosen from the true per-slot acceptance probabilities, which a real policy only estimates).
Live process as live_tau.py (renewal over the anchors of each window). Step cost models (relative to MTP3):
  dflash: 0.89 + 0.035 K   (one verify row ~3.4%; K 6 = 1.10, K 8 = 1.17)
  mtp:    1.00 + 0.09 (K - 3)   (each extra MTP draft: a sequential MoE pass + a verify row; measured at K 4)
Oracle: per round maximise E[tokens | K] - lambda * step(K), lambda iterated to the throughput (Dinkelbach).
Ragged verify (IE 10-13, K <= 4): the drafter still runs 4 passes nearly every step (the deepest of 32 lanes), so a round
of depth K saves only its verify rows: cost(K) = 1.09 - (4 - K) c_row (--ragged-row c_row, e.g. 0.034 / 0.05; --kmax 4); the
fixed-K table keeps the uniform model (a fixed K < 4 also drops drafter passes).
usage: python adaptive_bound.py ANCHORS.pt --kind dflash|mtp --mode sel_srv_T10 [--kmax 15] [--ragged-row 0.034]"""
import argparse

import torch


def step_cost(kind, K):
    return 0.89 + 0.035 * K if kind == 'dflash' else 1.0 + 0.09 * (K - 3)


def renewal_var(anchors, c, Ks):
    """anchors [A]; c [A, Kmax] masked conditional acceptance; Ks [A] per-anchor K. -> (sum v * tokens, sum v * 1, v)."""
    A, Kmax = c.shape
    pref = torch.cumprod(c.double(), dim=1)
    ar = torch.arange(Kmax)
    keep = ar[None, :] < Ks[:, None]
    pk = pref * keep                                                         # P(slots 1..k accepted), k <= K
    tokens = 1.0 + pk.sum(1)
    atleast = torch.cat([torch.ones(A, 1, dtype=torch.float64), pk], dim=1)  # P(>= j accepted), j = 0..Kmax
    exact = atleast - torch.cat([pk, torch.zeros(A, 1, dtype=torch.float64)], dim=1)
    idx = {int(x): i for i, x in enumerate(anchors.tolist())}
    pos = anchors.tolist()
    v = [0.0] * A
    v[0] = 1.0
    pending, carry = {}, 0.0
    ex = exact.tolist()
    for i in range(A):
        a = pos[i]
        for p in [p for p in pending if p < a]:
            carry += pending.pop(p)
        v[i] += pending.pop(a, 0.0) + carry
        carry = 0.0
        vi = v[i]
        if vi == 0.0:
            continue
        for j in range(int(Ks[i]) + 1):
            m = vi * ex[i][j]
            if m == 0.0:
                continue
            nxt = a + j + 1
            if nxt in idx:
                v[idx[nxt]] += m
            else:
                pending[nxt] = pending.get(nxt, 0.0) + m
    return torch.tensor(v, dtype=torch.float64), tokens


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('anchors')
    ap.add_argument('--kind', choices=('dflash', 'mtp'), required=True)
    ap.add_argument('--mode', required=True)
    ap.add_argument('--kmax', type=int, default=0)
    ap.add_argument('--ragged-row', type=float, default=0.0, help='mtp, K <= 4: the oracle pays only verify rows below K 4')
    a = ap.parse_args()
    data = torch.load(a.anchors)
    sess = []
    for sid, d in data.items():
        c = (d[a.mode] * d['slot_mask'].to(d[a.mode].dtype)).double()
        if a.kmax:
            c = c[:, :a.kmax]
        sess.append((d['anchors'], c))
    kmax = sess[0][1].shape[1]
    costs = torch.tensor([step_cost(a.kind, K) for K in range(0, kmax + 1)], dtype=torch.float64)
    fixed_costs = costs.clone()
    if a.ragged_row:
        assert a.kind == 'mtp' and kmax == 4, 'ragged verify: --kind mtp --kmax 4'
        costs = torch.tensor([1.09 - (4 - K) * a.ragged_row for K in range(0, kmax + 1)], dtype=torch.float64)

    def throughput(choose, costs=costs):
        num = den = rounds = 0.0
        ksum = 0.0
        for anchors, c in sess:
            Ks = choose(c)
            v, tokens = renewal_var(anchors, c, Ks)
            num += float((v * tokens).sum()); den += float((v * costs[Ks]).sum())
            rounds += float(v.sum()); ksum += float((v * Ks.double()).sum())
        return num / den, num / rounds, ksum / rounds

    print('fixed K (live-process tau, tokens per unit step time):')
    for K in range(1, kmax + 1):
        thr, tau, _ = throughput(lambda c, K=K: torch.full((c.shape[0],), K, dtype=torch.long), fixed_costs)
        print('  K %2d: tau %.4f, step %.3f, throughput %.4f' % (K, tau, step_cost(a.kind, K), thr), flush=True)
    lam = 2.5
    for it in range(8):
        def choose(c, lam=lam):
            pref = torch.cumprod(c, dim=1)
            gain = torch.cat([torch.ones(c.shape[0], 1, dtype=torch.float64), 1.0 + torch.cumsum(pref, dim=1)], dim=1)  # E[tokens | K], K = 0..kmax
            obj = gain - lam * costs[None, :]
            obj[:, 0] = -1e9                                              # K >= 1 (a round always drafts)
            return obj.argmax(1)
        thr, tau, kbar = throughput(choose)
        print('oracle adaptive K, iteration %d: lambda %.4f -> throughput %.4f (tau %.4f, mean K %.2f)' % (it, lam, thr, tau, kbar), flush=True)
        if abs(thr - lam) < 1e-4:
            break
        lam = thr


if __name__ == '__main__':
    main()
