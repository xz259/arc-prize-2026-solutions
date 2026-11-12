"""fuse_rms.py (2026-10-22, Claude main session 099768dc; the user's go for the step-conditioned round's CPU prep, V67_DESIGN 262):
the scale of the MTP fuse output x0 = fc_hidden(norm(h)) + fc_embedding(norm(embed(tok))) on real windows, for sizing the
step-embedding LR (IE: lr x effective steps should reach ~20-30% of the per-coordinate RMS). s_k is one H vector added to
all hc streams of x0 (step_emb.py), so the relevant scale of coordinate j is the RMS of x0[:, :, j] over rows and streams.

Step-1 inputs only (the target's multi-stream state h at row t, token t+1): the hidden branch is RMS-normalised over all
hc*H before fc_hidden, so its scale does not depend on whether h is the target's state (step 1) or the drafter's own
(steps 2+); the per-coordinate directions may differ. CPU only, reads a row slice of each feature file.

usage: fuse_rms.py --head HEAD.safetensors --target-dir TARGET --draft-config CFG --features DIR [--files 8] [--rows 4096]
       [--threads 8] [--warmup 94 --decay-start 1500 --steps 1875] [--out OUT.json]"""
import argparse
import glob
import json
import os
import sys

import torch
from safetensors import safe_open

sys.path.insert(0, os.environ.get('MTP_KIT', '/workspace/arc3-inference/mtp_ft_20261011'))
import qwen38_mtp as M  # noqa: E402

EMB = 'model.language_model.embed_tokens.weight'


def q(x, p):
    return float(torch.quantile(x, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--head', required=True); ap.add_argument('--target-dir', required=True)
    ap.add_argument('--draft-config', required=True); ap.add_argument('--features', required=True)
    ap.add_argument('--files', type=int, default=8); ap.add_argument('--rows', type=int, default=4096)
    ap.add_argument('--threads', type=int, default=8)
    ap.add_argument('--warmup', type=int, default=94); ap.add_argument('--decay-start', type=int, default=1500)
    ap.add_argument('--steps', type=int, default=1875); ap.add_argument('--out', default='')
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    cfg = M.MTPConfig.from_text_config(json.load(open(a.draft_config))['text_config'])
    H, hc, eps = cfg.hidden_size, cfg.hc_count, cfg.rms_eps

    with safe_open(a.head, 'pt') as f:
        w_ne = f.get_tensor('mtp.pre_fc_norm_embedding.weight').float()
        w_fe = f.get_tensor('mtp.fc_embedding.weight').float()
        w_nh = f.get_tensor('mtp.pre_fc_norm_hidden.weight').float()
        w_fh = f.get_tensor('mtp.fc_hidden.weight').float()
    idx = json.load(open(os.path.join(a.target_dir, 'model.safetensors.index.json')))['weight_map']
    with safe_open(os.path.join(a.target_dir, idx[EMB]), 'pt') as f:
        emb = f.get_tensor(EMB)                                       # [V, H] BF16, rows gathered per window

    files = sorted(glob.glob(os.path.join(a.features, '*.safetensors')))[:a.files]
    s1 = torch.zeros(H, dtype=torch.float64); s2 = torch.zeros(H, dtype=torch.float64); n = 0
    tot = dict(x0=0.0, e=0.0, h=0.0); per_stream = torch.zeros(hc, dtype=torch.float64)
    for path in files:
        with safe_open(path, 'pt') as fh:
            keys = set(fh.keys())
            tok = fh.get_tensor('token_ids')
            N = tok.shape[0]
            R = min(a.rows, N - 1)
            s = max(0, (N - R - 1) // 2)
            if 'mtp_hidden_states' in keys:
                h = fh.get_slice('mtp_hidden_states')[s:s + R].float()
            else:
                h = fh.get_slice('hidden_states')[s:s + R]
                h = h[:, -hc:, :].reshape(R, hc * H).float()
        t = tok[s + 1:s + 1 + R].long()
        with torch.no_grad():
            e = torch.nn.functional.linear(M.gemma_rmsnorm(emb[t].float(), w_ne, eps), w_fe)              # [R, H]
            hn = M.gemma_rmsnorm(h, w_nh, eps).unflatten(-1, (hc, H))
            hb = torch.nn.functional.linear(hn, w_fh)                                                      # [R, hc, H]
            x0 = hb + e.unsqueeze(-2)
        x = x0.double()
        s1 += x.sum((0, 1)); s2 += x.square().sum((0, 1)); n += R * hc
        tot['x0'] += float(x.square().sum()); tot['e'] += float(e.double().square().sum()) * hc
        tot['h'] += float(hb.double().square().sum()); per_stream += x.square().sum((0, 2))
        print('%s rows %d (from %d of %d): x0 rms %.4f' % (os.path.basename(path)[:60], R, s, N,
                                                          float(x.square().mean().sqrt())), flush=True)
    mean = s1 / n
    rms_j = (s2 / n).sqrt()                                           # per-coordinate RMS over rows and streams
    sd_j = (s2 / n - mean.square()).clamp_min(0).sqrt()
    out = dict(files=len(files), rows_x_streams=n,
               x0_rms=(tot['x0'] / (n * H)) ** 0.5, e_rms=(tot['e'] / (n * H)) ** 0.5, hidden_branch_rms=(tot['h'] / (n * H)) ** 0.5,
               per_stream_rms=[float(v) for v in (per_stream / (n / hc * H)).sqrt()],
               coord_rms=dict(p10=q(rms_j.float(), .1), p50=q(rms_j.float(), .5), p90=q(rms_j.float(), .9), max=float(rms_j.max())),
               coord_sd=dict(p10=q(sd_j.float(), .1), p50=q(sd_j.float(), .5), p90=q(sd_j.float(), .9)),
               coord_abs_mean=dict(p50=q(mean.abs().float(), .5), p90=q(mean.abs().float(), .9), max=float(mean.abs().max())))
    W, D, S = a.warmup, a.decay_start, a.steps
    eff = W / 2 + (D - W) + (S - D) / 2                               # sum of lr(t) / peak over the WSD schedule (cosine to 0)
    out['schedule'] = dict(warmup=W, decay_start=D, steps=S, effective_full_lr_steps=eff)
    out['lr_for_reach'] = {str(f): f * out['coord_rms']['p50'] / eff for f in (0.2, 0.25, 0.3)}
    print(json.dumps(out, indent=1))
    if a.out:
        json.dump(out, open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
