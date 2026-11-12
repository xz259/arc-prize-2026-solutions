"""train_mtp.py (2026-10-11; MTP fine-tune backup plan): fine-tune the Qwen3.8-Flash-Next MTP head (qwen38_mtp.py) on replay
exports that carry the target's pre-final-mixer multi-stream state per row (`mtp_hidden_states` [N, hc*H], written by the
dflashtaps MTP option, see README.md), with the corpus training mask (assistant targets) from the Phase 6 streams.

Data: safetensors files (one per session or segment) with token_ids [N], position_ids [N] (stream index),
served_position_ids [N] (the engine's 1-D positions, used for RoPE as at serving) and the MTP input: hidden_states
[N, slots, H] whose last hc slots are the multi-stream state (dflashtaps with ARC3_S2_TAPS_MTP=1, apply_dflashtaps_mtp.py),
or a mtp_hidden_states [N, hc*H] tensor;
--streams STREAMS_DIR supplies each session's role array (<run>/<session>.npz: 1 = trained assistant target), matched by
the session id in the file name (`arc3-persistent-<sid>-<suffix>[_sNNNN]`), positions by position_ids. Files are cut into
windows of --window rows (overlapping by num_steps + 1 so every label is used once); --val-files go to validation.
Training: AdamW on the head (--train all | nonexpert: experts and router frozen), cosine or WSD schedule with warmup,
loss = sum_k w_k [CE_k + kl_weight * KL_k] (qwen38_mtp.step_losses), BF16 autocast, gradient clipping, checkpoints with
save_mtp_checkpoint (the checkpoint's own names) for export_mtp_drafter.py.
Validation: per-step CE / top-1 and the expected acceptance with probabilistic drafting from the target's own
distribution (alpha_k = sum_x min(p_target, q_mtp); tau = 1 + a1 + a1 a2 + a1 a2 a3), the production metric's form.
usage: python train_mtp.py --data FILES_OR_DIRS --streams DIR --mtp-file BF16_MTP --target-dir TARGET --out OUTDIR
       [--val-files N] [--steps N] [--lr 2e-5] [--warmup 100] [--schedule wsd|cosine] [--num-steps 3]
       [--step-weights 1,0.8,0.64] [--kl-weight 0] [--window 8192] [--train all|nonexpert] [--eval-every 200]
       [--save-every 1000] [--attention flex|reference] [--device cuda] [--seed 0]"""
import argparse
import glob
import json
import math
import os
import random
import re
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen38_mtp as M  # noqa: E402

SID = re.compile(r'(arc3-[0-9a-f]{32})')


def list_files(specs):
    out = []
    for s in specs:
        out += sorted(glob.glob(os.path.join(s, '*.safetensors'))) if os.path.isdir(s) else sorted(glob.glob(s))
    return out


class Windows:
    """(file, start, end) windows over the export files; loads lazily."""

    def __init__(self, files, streams, window, num_steps, hc=4):
        self.items = []
        self.hc = hc
        self.streams = streams
        self.roles = {}
        if streams:
            for line in open(os.path.join(streams, 'sessions.jsonl')):
                d = json.loads(line)
                self.roles[d['session_id']] = os.path.join(streams, d['run'], d['session_id'] + '.npz')
        from safetensors import safe_open
        for f in files:
            with safe_open(f, 'pt') as fh:
                n = fh.get_slice('token_ids').get_shape()[0]
                keys = set(fh.keys())
                assert 'mtp_hidden_states' in keys or ('hidden_states' in keys and fh.get_slice('hidden_states').get_shape()[1] > hc), \
                    'not an MTP export: %s' % f
            step = window - num_steps - 1
            for a in range(0, max(1, n - num_steps - 1), step):
                b = min(n, a + window)
                if b - a > num_steps + 2:
                    self.items.append((f, a, b))

    def load(self, item, device):
        from safetensors import safe_open
        f, a, b = item
        with safe_open(f, 'pt') as fh:
            tok = fh.get_slice('token_ids')[a:b]
            pos = fh.get_slice('served_position_ids')[a:b]
            sidx = fh.get_slice('position_ids')[a:b]
            if 'mtp_hidden_states' in fh.keys():
                h = fh.get_slice('mtp_hidden_states')[a:b]
            else:                                                   # the last hc slots: the multi-stream state
                hs = fh.get_slice('hidden_states')
                S = hs.get_shape()[1]
                h = hs[a:b, S - self.hc:S, :].reshape(b - a, -1)
        mask = torch.ones_like(tok)
        m = SID.search(os.path.basename(f))
        if self.streams and m and m.group(1) in self.roles:
            role = np.load(self.roles[m.group(1)])['role']
            idx = sidx.numpy()
            ok = idx < len(role)
            mask = torch.from_numpy(np.where(ok, role[np.minimum(idx, len(role) - 1)] == 1, False).astype(np.int64))
        return (h.to(device), tok.to(device), pos.to(device), mask.to(device))


def lr_at(step, total, base, warmup, schedule, decay_ratio=0.2):
    if step < warmup:
        return base * step / max(1, warmup)
    if schedule == 'cosine':
        return base * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))
    start = max(warmup, total - int(total * decay_ratio))
    if step < start:
        return base
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - start) / max(1, total - start))))


@torch.no_grad()
def evaluate(model, windows, items, device, num_steps, attention):
    model.eval()
    agg = {}
    acc_terms = [[] for _ in range(num_steps)]
    for it in items:
        h, tok, pos, mask = windows.load(it, device)
        with torch.autocast(device_type=device.split(':')[0], dtype=torch.bfloat16, enabled=device != 'cpu'):
            s = model.forward_steps(h, tok, pos, num_steps=num_steps, attention=attention)
            _, met = model.step_losses(s, h, tok, mask)
            L = s[0].shape[0]
            for k in range(1, num_steps + 1):
                m = mask[k + 1:L + k + 1].bool()
                if not m.any():
                    continue
                q = model.lm_head(s[k - 1][m]).float().softmax(-1)
                p = model.lm_head(model.teacher_hidden(h[k:L + k][m])).float().softmax(-1)
                acc_terms[k - 1].append(torch.minimum(p, q).sum(-1).cpu())
        for k, v in met.items():
            agg.setdefault(k, []).append(v)
    out = {k: float(np.mean(v)) for k, v in agg.items()}
    alphas = [float(torch.cat(t).mean()) if t else float('nan') for t in acc_terms]
    tau, prod = 1.0, 1.0
    for a in alphas:
        prod *= a
        tau += prod
    out.update({'alpha_step%d' % (k + 1): a for k, a in enumerate(alphas)}, tau_est=tau)
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', nargs='+', required=True); ap.add_argument('--streams', default='')
    ap.add_argument('--mtp-file', required=True); ap.add_argument('--target-dir', required=True)
    ap.add_argument('--draft-config', default='/workspace/models/Qwen3.8-Flash-Next-mtp-draft-nvfp4a16-noisc/config.json')
    ap.add_argument('--out', required=True); ap.add_argument('--val-files', type=int, default=8)
    ap.add_argument('--steps', type=int, default=2000); ap.add_argument('--lr', type=float, default=2e-5)
    ap.add_argument('--warmup', type=int, default=100); ap.add_argument('--schedule', default='wsd')
    ap.add_argument('--num-steps', type=int, default=3); ap.add_argument('--step-weights', default='1,0.8,0.64')
    ap.add_argument('--kl-weight', type=float, default=0.0); ap.add_argument('--window', type=int, default=8192)
    ap.add_argument('--train', default='all', choices=('all', 'nonexpert')); ap.add_argument('--eval-every', type=int, default=200)
    ap.add_argument('--save-every', type=int, default=1000); ap.add_argument('--attention', default='flex')
    ap.add_argument('--device', default='cuda'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--weight-decay', type=float, default=0.0); ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--tiny', action='store_true', help='tests: a tiny random model instead of the checkpoint')
    a = ap.parse_args()
    random.seed(a.seed); torch.manual_seed(a.seed)
    os.makedirs(a.out, exist_ok=False)
    json.dump(vars(a), open(os.path.join(a.out, 'args.json'), 'w'), indent=1)
    files = list_files(a.data)
    random.shuffle(files)
    val_files, train_files = files[:a.val_files], files[a.val_files:]
    if a.tiny:
        import test_qwen38_mtp_cpu as T
        model = T.tiny_model()
    else:
        cfg = M.MTPConfig.from_text_config(json.load(open(a.draft_config))['text_config'])
        model = M.Qwen38MTP(cfg)
        M.load_from_checkpoint(model, a.mtp_file, a.target_dir, strict=True)
    model.to(a.device)
    if a.train == 'nonexpert':
        for n, p in model.named_parameters():
            if n.startswith('layers.0.mlp.w_') or n == 'layers.0.mlp.gate.weight':
                p.requires_grad_(False)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, betas=(0.9, 0.95), weight_decay=a.weight_decay)
    tr = Windows(train_files, a.streams, a.window, a.num_steps, hc=model.cfg.hc_count)
    va = Windows(val_files, a.streams, a.window, a.num_steps, hc=model.cfg.hc_count)
    print('train windows %d (%d files), val windows %d (%d files); trainable %.3fB' % (
        len(tr.items), len(train_files), len(va.items), len(val_files), sum(p.numel() for p in params) / 1e9), flush=True)
    weights = [float(x) for x in a.step_weights.split(',')][:a.num_steps]
    log = open(os.path.join(a.out, 'log.jsonl'), 'a')
    order = []
    t0 = time.time()
    if a.eval_every:
        ev = evaluate(model, va, va.items, a.device, a.num_steps, a.attention)
        log.write(json.dumps(dict(step=0, **{'val_' + k: v for k, v in ev.items()})) + '\n'); log.flush()
        print('step 0 val', ev, flush=True)
    for step in range(1, a.steps + 1):
        if not order:
            order = list(range(len(tr.items)))
            random.shuffle(order)
        h, tok, pos, mask = tr.load(tr.items[order.pop()], a.device)
        for g in opt.param_groups:
            g['lr'] = lr_at(step, a.steps, a.lr, a.warmup, a.schedule)
        with torch.autocast(device_type=a.device.split(':')[0], dtype=torch.bfloat16, enabled=a.device != 'cpu'):
            s = model.forward_steps(h, tok, pos, num_steps=a.num_steps, attention=a.attention)
            loss, met = model.step_losses(s, h, tok, mask, step_weights=weights, kl_weight=a.kl_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, a.clip)
        opt.step()
        rec = dict(step=step, loss=float(loss), grad_norm=float(gn), lr=opt.param_groups[0]['lr'], sec=round(time.time() - t0, 1), **met)
        log.write(json.dumps(rec) + '\n')
        if step % 10 == 0:
            log.flush()
            print(json.dumps(rec), flush=True)
        if a.eval_every and step % a.eval_every == 0:
            ev = evaluate(model, va, va.items, a.device, a.num_steps, a.attention)
            log.write(json.dumps(dict(step=step, **{'val_' + k: v for k, v in ev.items()})) + '\n'); log.flush()
            print('step %d val' % step, ev, flush=True)
        if step % a.save_every == 0 or step == a.steps:
            n = M.save_mtp_checkpoint(model, os.path.join(a.out, 'head_step%d.safetensors' % step))
            print('saved head_step%d (%d tensors)' % (step, n), flush=True)


if __name__ == '__main__':
    main()
