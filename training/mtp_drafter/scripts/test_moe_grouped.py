"""test_moe_grouped.py (2026-10-13, Claude fork 5d20b23b): the grouped-GEMM MoE against the per-expert loop (unbind) in
train_mtp_sp.py on one real 4,096-row window: loss, per-step metrics, gradient norms and the relative gradient difference
per parameter group, plus the step time of each (forward + losses + backward).
usage: python test_moe_grouped.py [--device cuda:0]"""
import argparse
import glob
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_mtp_sp as T  # noqa: E402

M = T.M
TARGET = '/workspace/models/Qwen3.8-Flash-Next-W4A16-intel'


def run(model, win, item, dev, impl):
    M.MoE.forward = T.moe_forward_grouped if impl == 'grouped' else T.moe_forward_unbind
    model.zero_grad(set_to_none=True)
    h, tok, pos, mask = win.load(item, dev)
    torch.cuda.synchronize(dev); t0 = time.time()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
        s = model.forward_steps(h, tok, pos, num_steps=3, attention='flex')
        loss, met = model.step_losses(s, h, tok, mask, step_weights=[1, 0.8, 0.64], kl_weight=1.0)
    loss.backward()
    torch.cuda.synchronize(dev)
    grads = {n: p.grad.detach().float().cpu() for n, p in model.named_parameters() if p.grad is not None}
    return float(loss), met, grads, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    dev = a.device
    cfg = M.MTPConfig.from_text_config(json.load(open('/workspace/models/Qwen3.8-Flash-Next-mtp-draft-nvfp4a16-noisc/config.json'))['text_config'])
    model = M.Qwen38MTP(cfg)
    M.load_from_checkpoint(model, TARGET + '/model_extra_tensors.safetensors', TARGET, strict=True)
    model.to(dev)
    files = sorted(glob.glob('/mnt/shared/arc3-inference/sidepod_20261012/features/sp_mtptrain20m/*.safetensors'))[:6]
    win = T.Windows(files, '/workspace/arc3-inference/dflash_streams_20261011', 4096, 3, hc=4, pos_key='position_ids')
    item = None
    for it in win.items:                                   # a window with plenty of assistant targets
        if int(win.load(it, 'cpu')[3].sum()) > 1500:
            item = it
            break
    print('window', item, flush=True)
    runs = {}
    for k, impl in enumerate(('loop', 'grouped', 'loop', 'loop', 'grouped', 'grouped')):
        res = run(model, win, item, dev, impl)
        print('%-8s loss %.6f time %.2f s' % (impl, res[0], res[3]), flush=True)
        if k >= 2:
            runs.setdefault(impl, []).append(res)

    def cmp(a, b, label):
        worst = []
        for n, g in a[2].items():
            rel = float((b[2][n] - g).norm() / g.norm().clamp_min(1e-12))
            worst.append((rel, n, float(g.norm())))
        worst.sort(reverse=True)
        print('%s: loss diff %.3e; worst grad rel diffs:' % (label, b[0] - a[0]), flush=True)
        for rel, n, gn in worst[:5]:
            print('  %.3e  |g| %.3e  %s' % (rel, gn, n), flush=True)

    cmp(runs['loop'][0], runs['loop'][1], 'loop vs loop')
    cmp(runs['grouped'][0], runs['grouped'][1], 'grouped vs grouped')
    cmp(runs['loop'][0], runs['grouped'][0], 'loop vs grouped')

if __name__ == '__main__':
    main()
