"""test_scorer_options_cpu.py (2026-10-13, Claude fork 5d20b23b): mtp_eval_val50.py --trunc-sweep on random logits: the
sweep mode at the production truncation (20, 0.95) equals prob_srv exactly, other truncations differ, and every mode is in
[0, 1]. (--index-share is checked on the GPU by a one-session smoke run: the mask reuse and the dropped chain change the
deep slots only.) usage: python test_scorer_options_cpu.py"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mtp_eval_val50 as E  # noqa: E402

g = torch.Generator().manual_seed(0)
n, S, V = 64, 300, 1000
sub = torch.randperm(V, generator=g)[:S].sort().values
y = torch.randn(n, S, generator=g) * 3
pl = torch.randn(n, V, generator=g) * 3
pl[:, sub] += 1.5                                         # most target mass on the subset, as in the data
trunc = [(20, 0.95), (10, 0.90), (40, 0.98)]
acc = {m: torch.zeros(n, 1) for m in E.mode_names(trunc)}
E.score(acc, 0, 0, y, pl, sub, trunc)
for t in E.TEMPS:
    tag = '%02d' % round(t * 10)
    base = acc['prob_srv_T' + tag]
    assert torch.equal(acc['prob_srv_T%s_k20p95' % tag], base), tag
    assert not torch.equal(acc['prob_srv_T%s_k10p90' % tag], base), tag
    for m in [k for k in acc if k.startswith('prob_srv_T' + tag)]:
        assert float(acc[m].min()) >= 0 and float(acc[m].max()) <= 1 + 1e-6, m
    print('  T %s: prob_srv %.4f, k10p90 %.4f, k40p98 %.4f' % (tag, float(base.mean()), float(acc['prob_srv_T%s_k10p90' % tag].mean()),
          float(acc['prob_srv_T%s_k40p98' % tag].mean())))
assert E.mode_names() == E.mode_names(())
print('scorer options test ok')
