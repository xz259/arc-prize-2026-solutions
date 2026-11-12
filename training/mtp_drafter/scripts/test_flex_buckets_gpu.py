"""GPU test of train_mtp_sp.flex_causal_attention_lse_compiled after the 10-13 recompile-limit fix: 14 window lengths between
37 and 16,384 rows (more than 8 distinct shapes before the fix) run forward + backward at the MTP layer's shapes (24 query
heads, 2 KV heads, head dim 256, BF16) without falling back to eager (peak memory per call stays far below the dense
backward's ~7.6 GiB score matrix at 16K), and the outputs match the dense reference (qwen38_mtp's eager attention math)
at a short length. usage: python test_flex_buckets_gpu.py [--device cuda:0]"""
import argparse
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_mtp_sp as T  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    dev = a.device
    torch.manual_seed(0)
    lens = [16384, 37, 9216, 15000, 5000, 12000, 2047, 8193, 16000, 11000, 7000, 3000, 14000, 13100]
    worst = 0.0
    for L in lens:
        q = torch.randn(L, 24, 256, device=dev, dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(L, 2, 256, device=dev, dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(L, 2, 256, device=dev, dtype=torch.bfloat16, requires_grad=True)
        torch.cuda.synchronize(dev); torch.cuda.reset_peak_memory_stats(dev)
        base = torch.cuda.memory_allocated(dev)
        t = time.time()
        out, lse = T.flex_causal_attention_lse_compiled(q, k, v, 256 ** -0.5)
        (out.float().square().mean() + lse.mean()).backward()
        torch.cuda.synchronize(dev)
        peak = (torch.cuda.max_memory_allocated(dev) - base) / 2**30
        worst = max(worst, peak)
        print('T %5d (padded %5d): %.2f s, peak +%.2f GiB' % (L, max(1024, -(-L // 2048) * 2048), time.time() - t, peak), flush=True)
        assert peak < 3.0, 'peak %.2f GiB: eager dense fallback?' % peak
        del q, k, v, out, lse
    # correctness at a short length against plain causal softmax attention (fp32)
    L = 300
    q = torch.randn(L, 24, 256, device=dev, dtype=torch.bfloat16)
    k = torch.randn(L, 2, 256, device=dev, dtype=torch.bfloat16)
    v = torch.randn(L, 2, 256, device=dev, dtype=torch.bfloat16)
    out, lse = T.flex_causal_attention_lse_compiled(q, k, v, 256 ** -0.5)
    kk, vv = k.float().repeat_interleave(12, dim=1), v.float().repeat_interleave(12, dim=1)
    s = torch.einsum('thd,shd->hts', q.float(), kk) * 256 ** -0.5
    s = s.masked_fill(torch.triu(torch.ones(L, L, dtype=torch.bool, device=dev), 1), float('-inf'))
    ref = torch.einsum('hts,shd->thd', s.softmax(-1), vv)
    err = float((out.float() - ref).abs().max())
    print('max |out - reference| at T 300: %.3e; worst peak +%.2f GiB over %d lengths' % (err, worst, len(lens)))
    assert err < 3e-2, err
    print('flex bucket test ok')


if __name__ == '__main__':
    main()
