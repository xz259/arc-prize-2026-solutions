"""profile_mtp_step.py (2026-10-13, Claude fork 5d20b23b): where a train_mtp_sp.py step spends its 37 s (window 4,096,
K 3, CE + KL): data load, forward_steps (per step: fuse, attention, MoE), losses, backward, optimizer, timed with CUDA
synchronisation on one real window; plus torch.profiler's top operators for one step.
usage: python profile_mtp_step.py [--device cuda:0] [--window 4096]"""
import argparse
import glob
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_mtp_sp as T  # noqa: E402  (its step_losses patch, Windows)

M = T.M
TARGET = '/workspace/models/Qwen3.8-Flash-Next-W4A16-intel'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--window', type=int, default=4096)
    a = ap.parse_args()
    dev = a.device
    cfg = M.MTPConfig.from_text_config(json.load(open('/workspace/models/Qwen3.8-Flash-Next-mtp-draft-nvfp4a16-noisc/config.json'))['text_config'])
    model = M.Qwen38MTP(cfg)
    M.load_from_checkpoint(model, TARGET + '/model_extra_tensors.safetensors', TARGET, strict=True)
    model.to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=1e-6, betas=(0.9, 0.95))
    files = sorted(glob.glob('/mnt/shared/arc3-inference/sidepod_20261012/features/sp_mtptrain20m/*.safetensors'))[:4]
    win = T.Windows(files, '/workspace/arc3-inference/dflash_streams_20261011', a.window, 3, hc=4, pos_key='position_ids')
    sync = lambda: torch.cuda.synchronize(dev)
    timers = {}

    def tick(name, t0):
        sync(); timers[name] = timers.get(name, 0.0) + time.time() - t0; return time.time()

    orig_moe = M.MoE.forward

    def moe_timed(self, x):
        t0 = time.time(); out = orig_moe(self, x); tick('moe_fwd', t0); return out

    M.MoE.forward = moe_timed
    for it in range(3):
        timers.clear()
        t = time.time()
        h, tok, pos, mask = win.load(win.items[it], dev)
        t = tick('load', t)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            s = model.forward_steps(h, tok, pos, num_steps=3, attention='flex')
            t = tick('forward_steps', t)
            loss, met = model.step_losses(s, h, tok, mask, step_weights=[1, 0.8, 0.64], kl_weight=1.0)
            t = tick('losses', t)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        t = tick('backward', t)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        t = tick('optimizer', t)
        print('iteration %d: %s' % (it, {k: round(v, 2) for k, v in timers.items()}), flush=True)
    M.MoE.forward = orig_moe
    from torch.profiler import ProfilerActivity, profile
    h, tok, pos, mask = win.load(win.items[3], dev)
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            s = model.forward_steps(h, tok, pos, num_steps=3, attention='flex')
            loss, met = model.step_losses(s, h, tok, mask, step_weights=[1, 0.8, 0.64], kl_weight=1.0)
        loss.backward()
        sync()
    print(prof.key_averages().table(sort_by='cuda_time_total', row_limit=18), flush=True)
    print(prof.key_averages().table(sort_by='cpu_time_total', row_limit=12), flush=True)


if __name__ == '__main__':
    main()
