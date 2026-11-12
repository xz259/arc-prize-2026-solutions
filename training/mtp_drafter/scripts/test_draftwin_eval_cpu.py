"""CPU test of mtp_eval_val50.py --draftwin (2026-10-13, Claude fork 5d20b23b): Context.token_mask on synthetic group keys.
(1) draftwin 0 and draftwin 2^40 select exactly what the unwindowed selection does; (2) with W = 4,096 every selected
row is causal and lies at or after the first position of group floor((q - W + 1) / 4) (or in the query's incomplete
group); (3) where the window holds >= 512 complete visible groups, exactly 512 groups (2,048 rows) plus the tail are
selected; (4) queries whose window reaches back past the first row select exactly what the unwindowed selection does.
usage: python test_draftwin_eval_cpu.py   (from the scripts directory; CPU only)"""
import os
import sys
from types import SimpleNamespace

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mtp_eval_val50 as E


def context(N, p0, G_dim=128, seed=0):
    g = torch.Generator().manual_seed(seed)
    c = object.__new__(E.Context)
    c.N, c.p0 = N, p0
    c.pos = torch.arange(p0, p0 + N, dtype=torch.int64)
    c.g0 = -(-p0 // E.GROUP)
    c.G = max(0, (p0 + N) // E.GROUP - c.g0)
    c.kg = torch.randn(c.G, G_dim, generator=g)
    c.args = SimpleNamespace(attention='qsa', draftwin=0)
    return c


def main():
    torch.manual_seed(1)
    N, p0, W = 12000, 1001, 4096
    c = context(N, p0)
    n = 96
    qrow = torch.cat([torch.randint(0, 3000, (n // 3,)), torch.randint(3000, N, (n - n // 3,))])
    qpos = c.pos[qrow] + torch.randint(0, 4, (n,))          # later draft steps query beyond the anchor row
    iq = torch.randn(n, 4, 128)
    m_ref = c.token_mask(iq, qpos, qrow)
    c.args.draftwin = 1 << 40
    assert torch.equal(c.token_mask(iq, qpos, qrow), m_ref), 'draftwin 2^40 differs from off'
    c.args.draftwin = W
    m = c.token_mask(iq, qpos, qrow)
    rows = torch.arange(N)
    q = torch.minimum(qpos, c.pos[qrow])
    lo_pos = torch.div(q - W + 1, E.GROUP, rounding_mode='floor') * E.GROUP     # first position of the first visible group
    tail_start = ((c.pos[qrow] + 1) // E.GROUP) * E.GROUP
    full, same = 0, 0
    for i in range(n):
        sel = rows[m[i]]
        assert (sel <= qrow[i]).all(), 'non-causal row selected'
        assert (c.pos[sel] >= lo_pos[i]).all(), 'row before the window selected (query %d)' % i
        in_groups = (c.pos[sel] < tail_start[i]).sum().item()
        window_groups = (tail_start[i] - torch.clamp(lo_pos[i], min=c.g0 * E.GROUP)).item() // E.GROUP
        if window_groups >= E.BLOCK_TOPK:
            assert in_groups == E.BLOCK_TOPK * E.GROUP, 'query %d: %d group rows, expected 2,048' % (i, in_groups)
            full += 1
        if lo_pos[i] <= c.g0 * E.GROUP:
            assert torch.equal(m[i], m_ref[i]), 'query %d: window covers every row but the selection differs' % i
            same += 1
    assert full > 0 and same > 0, (full, same)
    print('draftwin eval mask ok: %d queries, %d with a full window (2,048 rows), %d untouched by the window' % (n, full, same))


if __name__ == '__main__':
    main()
