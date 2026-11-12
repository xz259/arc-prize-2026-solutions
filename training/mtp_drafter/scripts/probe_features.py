"""probe_features.py (2026-10-12, Claude fork 5d20b23b): SIDE_POD_PLAN section 5, the feature probes that gate the
DFlash input levers (Inference Engineering 10-12: ask whether the MTP's input adds beyond the five taps).

Task (mirrors DFlash's slot 1): at an anchor a (loss_mask[a] = 1, slot valid where loss_mask[a + 1] = 1), from the
features of row t = a - 1 (the last row with target features) plus the embedding of the anchor token x_a, predict the
target's distribution at row a (for x_{a+1}); loss KL(p_target || q_probe) over the full vocabulary, q = lm_head(probe(.))
with the target's frozen lm_head. Inputs (each 2,560 slot Gemma-free RMS-normalised, then concatenated with the normalised
anchor embedding):
  taps5      taps 2, 13, 24, 35, 46 (DFlash's inputs)
  taps5+hc4  + the 4 pre-final-mixer streams (the MTP's raw input)
  taps5+mix  + the MTP layer's own input mix: attn_hc.mix(fuse(h_t, x_a)) (one slot, the vector the MTP reads)
  hc4        the 4 streams alone
  mix        the MTP mix alone
Probes: linear (in -> 2,560) and MLP (in -> 4,096 -> SiLU -> 2,560), then lm_head. Split by session (seed 0): 40 train /
10 test. Metrics on the test sessions: KL, top-1 agreement with the target's argmax, slot-1 acceptance sum_v min(p, q) at
T 1 (untruncated), each with a bootstrap 95% interval over test sessions.
Data: the side pod's val50 export (sp_val50mtp: slots taps 0-4, final 5, streams 6-9) and the val50 pack (masks).
usage: python probe_features.py --features DIR --out OUT.json [--device cuda:1] [--epochs 3] [--variants ...]"""
import argparse
import json
import math
import os
import random
import sys
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open

PACK = '/workspace/arc3-inference/dflash_packed_20261011/cache_val50'
TARGET = '/workspace/models/Qwen3.8-Flash-Next-W4A16-intel'
KIT = '/workspace/arc3-inference/mtp_ft_20261011'
VARIANTS = {'taps5': ('taps',), 'taps5+hc4': ('taps', 'hc'), 'taps5+mix': ('taps', 'mix'), 'hc4': ('hc',), 'mix': ('mix',)}


def rms(x):
    x = x.float()
    return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6)


def load_rows(args, dev, mtp):
    """Per session: features of rows t = a - 1 for the valid anchors, the anchor ids and the target final state of row a."""
    segs = [json.loads(l) for l in open(os.path.join(args.pack, 'segments.jsonl'))]
    recs = [json.loads(l) for l in open(os.path.join(args.pack, 'rows.jsonl'))]
    files = {f.split('-', 2)[2].rsplit('-', 1)[0]: os.path.join(args.features, f) for f in os.listdir(args.features)
             if f.startswith('arc3-persistent-') and f.endswith('.safetensors')}
    data = []
    for seg, rec in zip(segs, recs):
        sid = seg['session_id']
        f = safe_open(files[sid], 'pt')
        hs = f.get_tensor('hidden_states'); ids = f.get_tensor('token_ids'); pos = f.get_tensor('position_ids')
        N = hs.shape[0]; w = N - len(rec['input_ids'])
        assert torch.equal(ids[w:], torch.tensor(rec['input_ids'])), sid
        m = torch.zeros(N, dtype=torch.bool); m[w:] = torch.tensor(rec['loss_mask'], dtype=torch.bool)
        a = torch.nonzero(m[:-1] & m[1:], as_tuple=False).squeeze(1)        # anchor a and slot 1 (a + 1) assistant
        a = a[a >= w + 1]
        t = a - 1
        d = dict(sid=sid, run=seg['run'], taps=hs[t, 0:5].clone(), hc=hs[t, 6:10].clone(), fin=hs[a, 5].clone(), x=ids[a].clone())
        with torch.no_grad():
            x0 = mtp.fuse(d['hc'].reshape(len(t), -1).to(dev), d['x'].to(dev))
            d['mix'] = mtp.layers[0].attn_hyper_connection.mix(x0)[0].unsqueeze(1).cpu()
        data.append(d)
        print('%s %-12s anchors %d' % (sid, seg['run'], len(t)), flush=True)
    return data


class Probe(torch.nn.Module):
    def __init__(self, n_in, kind):
        super().__init__()
        d = (n_in + 1) * 2560
        if kind == 'linear':
            self.net = torch.nn.Linear(d, 2560)
        else:
            self.net = torch.nn.Sequential(torch.nn.Linear(d, 4096), torch.nn.SiLU(), torch.nn.Linear(4096, 2560))

    def forward(self, feats, emb):
        return self.net(torch.cat([rms(feats).flatten(1), rms(emb)], dim=1))


def batch_inputs(d, idx, parts, emb_w, dev):
    feats = torch.cat([d[p][idx] for p in parts], dim=1).to(dev)            # [n, slots, 2560]
    return feats, emb_w[d['x'][idx].to(dev)].float()


def run_variant(name, parts, kind, train, test, W, emb_w, args, dev):
    torch.manual_seed(0)
    n_in = sum(5 if p == 'taps' else 4 if p == 'hc' else 1 for p in parts)
    probe = Probe(n_in, kind).to(dev)
    opt = torch.optim.AdamW(probe.parameters(), lr=args.lr_linear if kind == 'linear' else args.lr_mlp, weight_decay=0.01)
    items = [(si, j) for si, d in enumerate(train) for j in range(0, len(d['x']), args.batch)]
    steps = args.epochs * len(items)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / steps))))
    rng = random.Random(0)
    t0 = time.time()
    for ep in range(args.epochs):
        rng.shuffle(items)
        for si, j in items:
            d = train[si]
            idx = torch.arange(j, min(len(d['x']), j + args.batch))
            feats, emb = batch_inputs(d, idx, parts, emb_w, dev)
            with torch.no_grad():
                lp = (d['fin'][idx].to(dev) @ W.t()).float().log_softmax(-1)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                h = probe(feats, emb)
            lq = (h.to(torch.bfloat16) @ W.t()).float().log_softmax(-1)
            loss = (lp.exp() * (lp - lq)).sum(-1).mean()
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); sched.step()
    per = []
    with torch.no_grad():
        for d in test:
            s = dict(n=0.0, kl=0.0, top1=0.0, acc=0.0)
            for j in range(0, len(d['x']), args.batch):
                idx = torch.arange(j, min(len(d['x']), j + args.batch))
                feats, emb = batch_inputs(d, idx, parts, emb_w, dev)
                lp = (d['fin'][idx].to(dev) @ W.t()).float().log_softmax(-1)
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    h = probe(feats, emb)
                lq = (h.to(torch.bfloat16) @ W.t()).float().log_softmax(-1)
                s['n'] += len(idx); s['kl'] += float((lp.exp() * (lp - lq)).sum())
                s['top1'] += float((lp.argmax(-1) == lq.argmax(-1)).sum()); s['acc'] += float(torch.minimum(lp.exp(), lq.exp()).sum())
            per.append(s)
    out = {}
    for k in ('kl', 'top1', 'acc'):
        val = sum(p[k] for p in per) / sum(p['n'] for p in per)
        boots = []
        r = random.Random(1)
        for _ in range(1000):
            smp = [per[r.randrange(len(per))] for _ in per]
            boots.append(sum(p[k] for p in smp) / sum(p['n'] for p in smp))
        boots.sort()
        out[k] = [round(val, 4), round(boots[25], 4), round(boots[974], 4)]
    out['seconds'] = round(time.time() - t0)
    print('%-10s %-6s KL %s top1 %s acc %s (%ds)' % (name, kind, out['kl'], out['top1'], out['acc'], out['seconds']), flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--pack', default=PACK)
    ap.add_argument('--out', required=True)
    ap.add_argument('--device', default='cuda:1')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch', type=int, default=1024)
    ap.add_argument('--lr-linear', type=float, default=1e-3)
    ap.add_argument('--lr-mlp', type=float, default=5e-4)
    ap.add_argument('--variants', nargs='*', default=list(VARIANTS))
    ap.add_argument('--kinds', nargs='*', default=['linear', 'mlp'])
    args = ap.parse_args()
    dev = torch.device(args.device)
    sys.path.insert(0, KIT)
    import qwen38_mtp as Q
    tc = json.load(open(os.path.join(TARGET, 'config.json'))); tc = tc.get('text_config', tc)
    mtp = Q.Qwen38MTP(Q.MTPConfig.from_text_config(tc), with_teacher=False).to(torch.bfloat16)
    Q.load_from_checkpoint(mtp, os.path.join(TARGET, 'model_extra_tensors.safetensors'), TARGET)
    mtp = mtp.to(dev).eval()
    W, emb_w = mtp.lm_head.weight, mtp.embed_tokens.weight
    data = load_rows(args, dev, mtp)
    order = list(range(len(data))); random.Random(0).shuffle(order)
    test = [data[i] for i in order[:10]]; train = [data[i] for i in order[10:]]
    print('train anchors %d (%d sessions), test anchors %d (%d sessions)' % (sum(len(d['x']) for d in train), len(train),
          sum(len(d['x']) for d in test), len(test)), flush=True)
    res = {}
    for kind in args.kinds:
        for name in args.variants:
            res['%s/%s' % (name, kind)] = run_variant(name, VARIANTS[name], kind, train, test, W, emb_w, args, dev)
            json.dump(dict(args=vars(args), results=res), open(args.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
