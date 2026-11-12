"""test_moe_grouped_fp32.py (2026-10-13, Claude fork 5d20b23b): is the grouped-GEMM MoE (train_mtp_sp.moe_forward_grouped,
fresh BF16 weight casts per call) as accurate as the per-expert loop? On one real 2,048-row window, K 3, served-KL loss:
gradients of (a) the loop in FP32 without autocast (the reference), (b) the loop under BF16 autocast (the trainer today),
(c) grouped under BF16 autocast (all three with the reference attention, fp32 inside); relative error of (b) and (c) against (a) per parameter group (experts, router, shared
expert, attention, rest). Then the time of one forward + backward at 4,096 rows, K 4, for loop vs grouped, and the time to
load a window from the volume (prefetch candidate).
usage: python test_moe_grouped_fp32.py [--device cuda:0]"""
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
SP = '/mnt/shared/arc3-inference/sidepod_20261012'


def group(n):
    if n.startswith('layers.0.mlp.w_'):
        return 'experts'
    if n == 'layers.0.mlp.gate.weight':
        return 'router'
    if 'shared_expert' in n:
        return 'shared'
    if 'self_attn' in n:
        return 'attention'
    return 'rest'


def grads(model, win, item, dev, impl, K, autocast, wsub, t2d, attention='flex'):
    T._moe_impl = T.moe_forward_grouped if impl == 'grouped' else T.moe_forward_unbind
    model.zero_grad(set_to_none=True)
    h, tok, pos, mask = win.load(item, dev)
    if not autocast:                     # the FP32 reference: the exported features are BF16 (10-13 06:22 dtype error)
        h = h.float()
    torch.cuda.synchronize(dev); t0 = time.time()
    with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=autocast):
        s = model.forward_steps(h, tok, pos, num_steps=K, attention=attention)
        loss, _ = T.step_losses_served(model, s, h, tok, mask, [1.0, 0.6, 0.26, 0.12][:K], 'served_kl', wsub, t2d)
    loss.backward()
    torch.cuda.synchronize(dev)
    dt = time.time() - t0
    g = {}
    for n, p in model.named_parameters():
        if p.grad is not None:
            g.setdefault(group(n), []).append(p.grad.detach().float().flatten().cpu())
    return float(loss), {k: torch.cat(v) for k, v in g.items()}, dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    a = ap.parse_args()
    dev = a.device
    cfg = M.MTPConfig.from_text_config(json.load(open('/workspace/models/Qwen3.8-Flash-Next-mtp-draft-nvfp4a16-noisc/config.json'))['text_config'])
    model = M.Qwen38MTP(cfg)
    M.load_from_checkpoint(model, TARGET + '/model_extra_tensors.safetensors', TARGET, strict=True)
    model.to(dev)
    M.flex_causal_attention_lse = T.flex_causal_attention_lse_compiled
    sub, t2d = T.load_subset(T.DRAFT_VOCAB, model.lm_head.weight.shape[0], dev)
    wsub = model.lm_head.weight.index_select(0, sub).contiguous()
    files = sorted(glob.glob(SP + '/features/sp_mtptrain20m/*.safetensors'))[:6]
    win2 = T.Windows(files, '/workspace/arc3-inference/dflash_streams_20261011', 2048, 3, hc=4, pos_key='position_ids')
    item = next(it for it in win2.items if int(win2.load(it, 'cpu')[3].sum()) > 800)
    ref = grads(model, win2, item, dev, 'loop', 3, False, wsub, t2d, 'reference')   # same (fp32-internal) attention in all three
    for impl in ('loop', 'grouped'):
        r = grads(model, win2, item, dev, impl, 3, True, wsub, t2d, 'reference')
        errs = {k: float((r[1][k] - ref[1][k]).norm() / ref[1][k].norm().clamp_min(1e-30)) for k in ref[1]}
        print('%-7s bf16 vs fp32 loop: loss %.6f vs %.6f; rel grad error %s' % (
            impl, r[0], ref[0], ' '.join('%s %.2e' % kv for kv in sorted(errs.items()))), flush=True)
    del ref
    win4 = T.Windows(files, '/workspace/arc3-inference/dflash_streams_20261011', 4096, 4, hc=4, pos_key='position_ids')
    it4 = win4.items[1]
    for impl in ('loop', 'grouped', 'loop', 'grouped'):
        r = grads(model, win4, it4, dev, impl, 4, True, wsub, t2d)
        print('%-7s 4,096 rows K 4: %.2f s forward + backward' % (impl, r[2]), flush=True)
    t0 = time.time()
    for it in win4.items[10:18]:
        win4.load(it, 'cpu')
    print('window load from the volume: %.3f s per 4,096-row window (8 windows)' % ((time.time() - t0) / 8), flush=True)


if __name__ == '__main__':
    main()
