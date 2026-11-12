"""test_flex_compiled.py (2026-10-13, Claude fork 5d20b23b): train_mtp_sp.flex_causal_attention_lse_compiled (torch.compile,
end padding to a multiple of 1,024) against the kit's eager flex_causal_attention_lse at the MTP layer's shapes (16 query
heads, 2 KV heads, head dim 256, bf16): output, log-sum-exp and q / k / v gradients of a fixed random projection; then
forward + backward time and peak memory per length (eager may run out of memory at long lengths: reported, not fatal).
usage: python test_flex_compiled.py [--device cuda:0] [--lengths 4092 8188 16380]"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_mtp_sp as T  # noqa: E402

M = T.M


def run(fn, q, k, v, proj):
    q, k, v = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    out, lse = fn(q, k, v, 256 ** -0.5)
    loss = (out.float() * proj).sum() + lse.float().sum() * 1e-3
    loss.backward()
    return out.detach(), lse.detach(), q.grad, k.grad, v.grad


def timed(fn, q, k, v, proj, dev, reps=3):
    torch.cuda.synchronize(dev); torch.cuda.reset_peak_memory_stats(dev)
    t0 = time.time()
    for _ in range(reps):
        run(fn, q, k, v, proj)
    torch.cuda.synchronize(dev)
    return (time.time() - t0) / reps, torch.cuda.max_memory_allocated(dev) / 2**30


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--lengths', type=int, nargs='+', default=[4092, 8188, 16380])
    a = ap.parse_args()
    dev = torch.device(a.device)
    eager, comp = M.flex_causal_attention_lse, T.flex_causal_attention_lse_compiled
    for n in a.lengths:
        g = torch.Generator(device=dev).manual_seed(n)
        q = torch.randn(n, 16, 256, device=dev, dtype=torch.bfloat16, generator=g)
        k = torch.randn(n, 2, 256, device=dev, dtype=torch.bfloat16, generator=g)
        v = torch.randn(n, 2, 256, device=dev, dtype=torch.bfloat16, generator=g)
        proj = torch.randn(n, 16, 256, device=dev, generator=g)
        rc = run(comp, q, k, v, proj)                                       # compiles on first use
        tc, mc = timed(comp, q, k, v, proj, dev)
        try:
            re = run(eager, q, k, v, proj)
            te, me = timed(eager, q, k, v, proj, dev)
            rel = lambda x, y: float((x.float() - y.float()).norm() / y.float().norm().clamp_min(1e-12))
            print('T %5d: compiled %.3f s %.1f GiB | eager %.3f s %.1f GiB | rel diff out %.2e lse %.2e dq %.2e dk %.2e dv %.2e' % (
                n, tc, mc, te, me, rel(rc[0], re[0]), rel(rc[1], re[1]), rel(rc[2], re[2]), rel(rc[3], re[3]), rel(rc[4], re[4])), flush=True)
            del re
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            print('T %5d: compiled %.3f s %.1f GiB | eager out of memory' % (n, tc, mc), flush=True)
        del rc
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
