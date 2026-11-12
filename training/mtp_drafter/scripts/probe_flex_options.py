"""probe_flex_options.py (2026-10-13, Claude fork 5d20b23b): which flex_attention kernel_options compile for the MTP layer's
shapes on sm_120 (16 query heads, 2 KV heads, head dim 256; the default backward template needs 114,688 B of shared memory,
the RTX PRO 6000 has 101,376 B). Tries option sets in order: forward + backward at 4,096 rows, then agreement with the eager
version and time; prints the first ones that work.
usage: python probe_flex_options.py [--device cuda:0] [--T 4096]"""
import argparse
import time

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

CANDIDATES = [
    dict(BLOCK_M=64, BLOCK_N=32, num_stages=1),
    dict(BLOCK_M=32, BLOCK_N=32, num_stages=1),
    dict(BLOCK_M1=32, BLOCK_N1=64, BLOCK_M2=64, BLOCK_N2=32, num_stages=1),
    dict(BLOCK_M1=16, BLOCK_N1=32, BLOCK_M2=32, BLOCK_N2=16, num_stages=1),
    dict(fwd_BLOCK_M=64, fwd_BLOCK_N=32, fwd_num_stages=1, bwd_BLOCK_M1=32, bwd_BLOCK_N1=32, bwd_BLOCK_M2=32,
         bwd_BLOCK_N2=32, bwd_num_stages=1),
    dict(fwd_BLOCK_M=32, fwd_BLOCK_N=32, fwd_num_stages=1, bwd_BLOCK_M1=16, bwd_BLOCK_N1=32, bwd_BLOCK_M2=32,
         bwd_BLOCK_N2=16, bwd_num_stages=1),
    dict(BLOCK_M=16, BLOCK_N=16, num_stages=1),
]


def run(fn, q, k, v, bm, opts, proj):
    q, k, v = (t.detach().clone().requires_grad_(True) for t in (q, k, v))
    kw = dict(block_mask=bm, scale=256 ** -0.5, enable_gqa=True, return_lse=True)
    if opts is not None:
        kw['kernel_options'] = opts
    out, lse = fn(q, k, v, **kw)
    ((out.float() * proj).sum() + lse.float().sum() * 1e-3).backward()
    return out.detach(), q.grad, k.grad, v.grad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--T', type=int, default=4096)
    a = ap.parse_args()
    dev = torch.device(a.device)
    T = a.T
    g = torch.Generator(device=dev).manual_seed(0)
    q = torch.randn(1, 16, T, 256, device=dev, dtype=torch.bfloat16, generator=g)
    k = torch.randn(1, 2, T, 256, device=dev, dtype=torch.bfloat16, generator=g)
    v = torch.randn(1, 2, T, 256, device=dev, dtype=torch.bfloat16, generator=g)
    proj = torch.randn(1, 16, T, 256, device=dev, generator=g)
    bm = create_block_mask(lambda b, h, qi, ki: qi >= ki, 1, None, T, T, device=dev)
    ref = run(flex_attention, q, k, v, bm, None, proj)
    rel = lambda x, y: float((x.float() - y.float()).norm() / y.float().norm().clamp_min(1e-12))
    for opts in CANDIDATES:
        torch._dynamo.reset()
        fn = torch.compile(flex_attention, dynamic=False)
        try:
            r = run(fn, q, k, v, bm, opts, proj)
            torch.cuda.synchronize(dev)
            t0 = time.time()
            for _ in range(3):
                run(fn, q, k, v, bm, opts, proj)
            torch.cuda.synchronize(dev)
            print('OK   %s: %.3f s fwd+bwd; rel diff vs eager out %.2e dq %.2e dk %.2e dv %.2e' % (
                opts, (time.time() - t0) / 3, rel(r[0], ref[0]), rel(r[1], ref[1]), rel(r[2], ref[2]), rel(r[3], ref[3])), flush=True)
        except Exception as e:                                          # noqa: BLE001
            print('FAIL %s: %s' % (opts, str(e).strip().splitlines()[-1][:160]), flush=True)
    torch.cuda.synchronize(dev)
    t0 = time.time()
    for _ in range(3):
        run(flex_attention, q, k, v, bm, None, proj)
    torch.cuda.synchronize(dev)
    print('eager: %.3f s fwd+bwd' % ((time.time() - t0) / 3), flush=True)


if __name__ == '__main__':
    main()
