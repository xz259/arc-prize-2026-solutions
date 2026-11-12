"""train_mtp_sp.py: the side pod's copy of mtp_ft_20261011/train_mtp.py (2026-10-12, Claude fork 5d20b23b) with
(1) the kit imported from MTP_KIT (default /workspace/arc3-inference/mtp_ft_20261011), (2) --pos-key: the RoPE positions
(default position_ids = the raw stream index, the drafter's RoPE position in production: draftqsa adds each lane's
deleted count to the served position; the original used served_position_ids, equal up to a constant inside unbroken
spans). Side-pod additions (10-13): checkpointed vocabulary tail, unbound / grouped MoE (speed); after the IE recipe
review: --loss served_kl | tv | lk on the served pair (p': T, top-k 20, top-p 0.95; q on the 32K drafthead subset),
--cooldown-loss, --accept-weights (accepted-path importance weights), --accum (windows per step), --train noexperts
(routed experts frozen, router at --router-lr-scale), --decay-ratio, --compile-flex, --fused-adam, the routed-expert load
entropy in the log, and the served alpha / tau / out-of-subset mass in validation; for the full run (10-13): full
training states (stop / resume / branch), --expert-weights served (frozen routed experts at their production NVFP4 values),
--unfreeze-experts (the escape hatch), --min-window (drops the 20-row segment tails) and --token-weighted (every trained
position of an optimizer step weighs the same). Defaults reproduce the kit.

train_mtp.py (2026-10-11; MTP fine-tune backup plan): fine-tune the Qwen3.8-Flash-Next MTP head (qwen38_mtp.py) on replay
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
import shutil
import sys
import threading
import time

import zlib

import numpy as np
import torch

sys.path.insert(0, os.environ.get('MTP_KIT', '/workspace/arc3-inference/mtp_ft_20261011'))
import qwen38_mtp as M  # noqa: E402
import step_emb as SE  # noqa: E402
import torch.nn.functional as F  # noqa: E402
import torch.utils.checkpoint as tuc  # noqa: E402


def step_losses_ckpt(self, samples, h0, tokens, loss_mask, step_weights=None, kl_weight=0.0, chunk=2048):
    """qwen38_mtp.step_losses with each row chunk's vocabulary tail under a non-reentrant activation checkpoint (side pod
    10-12: the original keeps every chunk's [rows, 248K] logits for backward and ran out of memory at 8,192-row windows);
    same losses and metrics, the teacher and drafter logits recomputed in backward."""
    K = len(samples)
    L = samples[0].shape[0]
    step_weights = step_weights or [1.0] * K
    total = samples[0].new_zeros((), dtype=torch.float32)
    metrics = {}
    for k, s in enumerate(samples, start=1):
        labels = tokens[k + 1:L + k + 1]
        mask = loss_mask[k + 1:L + k + 1].float()
        ce_sum = kl_sum = hits = 0.0

        def f(sc, lab, m, hk):
            logits = self.lm_head(sc).float()
            ce = (F.cross_entropy(logits, lab, reduction='none') * m).sum()
            kl = logits.new_zeros(())
            if kl_weight:
                with torch.no_grad():
                    tl = self.lm_head(self.teacher_hidden(hk)).float().log_softmax(-1)
                kl = ((tl.exp() * (tl - logits.log_softmax(-1))).sum(-1) * m).sum()
            hit = ((logits.argmax(-1) == lab).float() * m).sum().detach()
            return ce, kl, hit

        for a in range(0, L, chunk):
            b = min(L, a + chunk)
            ce, kl, hit = tuc.checkpoint(f, s[a:b], labels[a:b], mask[a:b], h0[a + k:b + k], use_reentrant=False)
            ce_sum = ce_sum + ce
            kl_sum = kl_sum + kl
            hits = hits + hit
        n = mask.sum().clamp(min=1)
        total = total + step_weights[k - 1] * (ce_sum + kl_weight * kl_sum) / n
        metrics['ce_step%d' % k] = float(ce_sum / n)
        metrics['top1_step%d' % k] = float(hits / n)
        if kl_weight:
            metrics['kl_step%d' % k] = float(kl_sum / n)
    return total, metrics


M.Qwen38MTP.step_losses = step_losses_ckpt


def moe_forward_unbind(self, x):
    """qwen38_mtp.MoE.forward with the stacked expert weights unbound once per call (side pod 10-13, profile: indexing
    the stacked [512, I, H] parameters per expert made every select's backward zero-fill and add a full-size gradient,
    4,608 times per step, 28 of a 38-s backward; unbind's backward stacks all expert gradients in one op). Same math."""
    T_, H = x.shape
    w, idx = self.route(x)
    out = torch.zeros(T_, H, dtype=torch.float32, device=x.device)
    flat = idx.reshape(-1)
    order = torch.argsort(flat, stable=True)
    counts = torch.bincount(flat, minlength=self.cfg.num_experts).tolist()
    tok = order // self.cfg.top_k
    wt = w.reshape(-1)[order]
    wg, wu, wd = self.w_gate.unbind(0), self.w_up.unbind(0), self.w_down.unbind(0)
    start = 0
    for e, n in enumerate(counts):
        if n == 0:
            continue
        t_e = tok[start:start + n]
        xe = x[t_e]
        hh = F.silu(xe @ wg[e].t()) * (xe @ wu[e].t())
        out.index_add_(0, t_e, (hh @ wd[e].t()).float() * wt[start:start + n].unsqueeze(-1))
        start += n
    se = self.shared_expert
    shared = se['down_proj'](F.silu(se['gate_proj'](x)) * se['up_proj'](x))
    out = out + shared.float() * torch.sigmoid(self.shared_expert_gate(x).float())
    return out.to(x.dtype)


_CAST = {}


def _bf16_weights(self):
    """The experts' fp32 master weights as BF16 [E, H, I] / [E, I, H] views for grouped GEMM, cast once per parameter
    version (once per optimizer step: the three draft steps of a forward share them; a new version recasts)."""
    if torch.is_grad_enabled():   # training: a fresh cast per call, so each draft step's weight gradient returns to
        # fp32 through its own cast and the three steps sum in fp32 as in the loop (a shared BF16 cast summed them in
        # BF16: up to 3.6% gradient difference against the loop on a real window)
        return (self.w_gate.to(torch.bfloat16).transpose(1, 2), self.w_up.to(torch.bfloat16).transpose(1, 2),
                self.w_down.to(torch.bfloat16).transpose(1, 2))
    key = id(self)
    ver = (self.w_gate._version, self.w_up._version, self.w_down._version, torch.is_grad_enabled())
    c = _CAST.get(key)
    if c is None or c[0] != ver:
        c = (ver, self.w_gate.to(torch.bfloat16).transpose(1, 2), self.w_up.to(torch.bfloat16).transpose(1, 2),
             self.w_down.to(torch.bfloat16).transpose(1, 2))
        _CAST[key] = c
    return c[1:]


def moe_forward_grouped(self, x):
    """qwen38_mtp.MoE.forward with the routed experts as three torch._grouped_mm calls over the expert-sorted rows
    (side pod 10-13; bitwise equal to the per-expert loop in a unit test on sm_120): no Python loop over 512 experts."""
    T_, H = x.shape
    w, idx = self.route(x)
    flat = idx.reshape(-1)
    order = torch.argsort(flat, stable=True)
    offs = torch.cumsum(torch.bincount(flat, minlength=self.cfg.num_experts), 0).to(torch.int32)
    tok = order // self.cfg.top_k
    xs = x[tok].to(torch.bfloat16)
    wg, wu, wd = _bf16_weights(self)
    hh = F.silu(torch._grouped_mm(xs, wg, offs=offs)) * torch._grouped_mm(xs, wu, offs=offs)
    y = torch._grouped_mm(hh, wd, offs=offs).float() * w.reshape(-1)[order].unsqueeze(-1)
    out = torch.zeros(T_, H, dtype=torch.float32, device=x.device).index_add_(0, tok, y)
    se = self.shared_expert
    shared = se['down_proj'](F.silu(se['gate_proj'](x)) * se['up_proj'](x))
    out = out + shared.float() * torch.sigmoid(self.shared_expert_gate(x).float())
    return out.to(x.dtype)


M.MoE.forward = moe_forward_grouped if os.environ.get('MTP_MOE', 'loop') == 'grouped' else moe_forward_unbind   # grouped: gradients differ from the loop by up to 3.6% (forward bitwise equal); default loop until an fp32-reference check


TRACK = [True]          # off during backward, so activation-checkpoint recomputation does not count the load twice


def _track_load(moe, flat):
    """Accumulate the routed-expert load (top-k assignments per expert) while training, for the load-entropy log."""
    if moe.training and TRACK[0] and getattr(moe, '_load', None) is not None:
        moe._load += torch.bincount(flat, minlength=moe.cfg.num_experts)


_moe_impl = M.MoE.forward


def moe_forward_tracked(self, x):
    if self.training and TRACK[0] and getattr(self, '_load', None) is not None:
        with torch.no_grad():
            _track_load(self, self.route(x)[1].reshape(-1))
    return _moe_impl(self, x)


M.MoE.forward = moe_forward_tracked


def forward_steps_ckpt(self, h0, tokens, positions, num_steps=3, attention='reference'):
    """qwen38_mtp.forward_steps with each draft step's layer (fuse, attention, MoE, mixers) under a non-reentrant
    activation checkpoint (side pod 10-13: K 5 at 8-16K rows exceeds 95 GB): kept per step are its input h, the step-1
    keys / values and the chain keys / values; the rest is recomputed in backward. Same math and outputs."""
    T_ = tokens.shape[0]
    L = T_ - num_steps - 1
    assert L > 0, (T_, num_steps)
    scale = self.cfg.head_dim ** -0.5

    def step(k, h, tok, pos, k1, v1, *chain_kv):
        attend = M.flex_causal_attention_lse if attention == 'flex' else M.causal_attention_lse
        x0 = SE.fuse_k(self, h, tok, k)                  # = self.fuse(h, tok) unless the model has step embeddings
        q, kk, vv, gate, inj = self.layer_pre(x0, pos)
        if k1 is None:
            out, _ = attend(q, kk, vv, scale)
        else:
            chain = list(zip(chain_kv[0::2], chain_kv[1::2])) + [(kk, vv)]
            o1, lse1 = attend(q, k1, v1, scale)
            out = M.merge_chain(q, o1, lse1, chain, scale)
        sample, h_next = self.layer_post(x0, out, gate, inj)
        return sample, h_next, kk, vv

    samples, chain_kv = [], []
    h = h0[:L]
    k1 = v1 = None
    for k in range(1, num_steps + 1):
        sample, h, kk, vv = tuc.checkpoint(step, k, h, tokens[k:L + k], positions[:L] + (k - 1), k1, v1, *chain_kv,
                                           use_reentrant=False)
        if k == 1:
            k1, v1 = kk, vv
        else:
            chain_kv += [kk, vv]
        samples.append(sample)
    return samples

# ---- served-pair losses (side pod 10-13, IE recipe review): the verifier compares the served target p' (T, top-k 20,
# top-p 0.95) with the draft q on the 32K drafthead subset; the optimal q on the subset is p' renormalised over it, so
# KL(p'_in || q_in) is the loss, with p''s mass outside the subset (m_out) a fixed rejection that is only logged.
DRAFT_VOCAB = '/workspace/arc3-inference/runs/stage0-reference-20260919-v1/stage12/harness_revision_v529/spec_decode_stage2/draft_vocab_32k.json'
TOPK, TOPP = 20, 0.95             # served target: T, top-k 20, top-p 0.95 (mtp_eval_val50.served_target)
DRAFT_K, DRAFT_P = 20, 0.95       # served draft: top-20 of the raw subset logits, top-p 0.95 prefix, then T (DRAFTTRUNC)


def load_subset(path, V, device):
    """(sub [S] vocab ids of the drafthead subset, t2d [V] subset position or -1); path 'tiny': the first half of V."""
    if path == 'tiny':
        ids = list(range(V // 2))
    else:
        v = json.load(open(path))
        ids = sorted({i for a, b in v['token_ranges'] for i in range(int(a), int(b))} | set(v.get('extra_ids', [])))
    sub = torch.tensor(ids, dtype=torch.long, device=device)
    t2d = torch.full((V,), -1, dtype=torch.long, device=device)
    t2d[sub] = torch.arange(len(ids), device=device)
    return sub, t2d


def served_target(logits_t):
    """[n, V] T-scaled target logits -> (p' [n, 20], vocab ids [n, 20]): top-k 20, then the top-p 0.95 prefix."""
    v, i = logits_t.topk(TOPK, dim=-1)
    p = torch.softmax(v, dim=-1)
    p = p * ((p.cumsum(-1) - p) < TOPP)
    return p / p.sum(-1, keepdim=True), i


def served_draft(y, t):
    """[n, S] raw subset logits -> (q' [n, 20], subset positions [n, 20]) as drafthead serves them at temperature t."""
    v, i = torch.topk(y.float(), DRAFT_K, dim=1)
    q1 = torch.softmax(v, dim=1)
    v = torch.where(torch.cumsum(q1, dim=1) - q1 < DRAFT_P, v, float('-inf'))
    return torch.softmax(v / t, dim=1), i


def step_losses_served(self, samples, h0, tokens, loss_mask, step_weights, kind, wsub, t2d, temp=1.0,
                       accept_weights=False, chunk=2048, norm=None):
    """Per step k: rows' loss on the served pair (kind served_kl: KL(p'_in || q_in); tv: 1 - alpha; lk: LK-hybrid
    lambda KL + (1 - lambda) TV, lambda = exp(-3 alpha) detached), alpha = sum_v min(p'(v), q(v)) over the subset with q
    the untruncated subset softmax at T. accept_weights: step k's rows weighted by w_k = prod_{j<k} min(1, q'_j(x)/p'_j(x))
    at the recorded tokens (served q', p'; detached), the importance weight of the accepted-path input distribution
    min(p', q')/alpha relative to the recorded p' samples. Step loss = step_weight * sum(mask w loss) / sum(mask).
    The served target is computed once per window row before the steps (10-13: it does not depend on the head; row r
    is step k's target for t = r - k), not per step inside the checkpoint, which recomputed it in backward.
    norm (--token-weighted, 10-13): step loss = step_weight * sum(mask w loss) / norm instead, and metrics['_n'] = the
    window's trained positions (mean over the steps), for one normalisation over all windows of the optimizer step."""
    K = len(samples)
    L = samples[0].shape[0]
    dev = samples[0].device
    total = samples[0].new_zeros((), dtype=torch.float32)
    metrics = {}
    w = torch.ones(L, device=dev)
    with torch.no_grad():                                                        # served target per row, once
        pps, pis = [], []
        for a in range(0, h0.shape[0], chunk):
            pp, pi = served_target(self.lm_head(self.teacher_hidden(h0[a:a + chunk])).float() / temp)
            pps.append(pp); pis.append(pi)
        PP, PI = torch.cat(pps), torch.cat(pis)
        del pps, pis

    def f(sc, pp, pi, lab, m, wr):
        y = (sc @ wsub.t()).float()                                              # [n, S] subset logits
        with torch.no_grad():
            pos = t2d[pi]
            ins = pos >= 0
            ppi = pp * ins
            mout = 1.0 - ppi.sum(1)
            pin = ppi / ppi.sum(1, keepdim=True).clamp_min(1e-12)
            lab_pos = t2d[lab]
            p_lab = (pp * (pi == lab[:, None])).sum(1)
        logq = torch.log_softmax(y / temp, dim=-1)
        lq = logq.gather(1, pos.clamp_min(0))
        kl = torch.where(pin > 0, pin * (torch.log(pin.clamp_min(1e-30)) - lq), torch.zeros_like(lq)).sum(1)
        alpha = torch.minimum(ppi, lq.exp() * ins).sum(1)
        if kind == 'served_kl':
            row = kl
        elif kind == 'tv':
            row = 1.0 - alpha
        else:
            lam = torch.exp(-3.0 * alpha.detach())
            row = lam * kl + (1.0 - lam) * (1.0 - alpha)
        loss = (row * wr).sum()
        with torch.no_grad():
            qd, qs = served_draft(y, temp)
            q_lab = (qd * (qs == lab_pos[:, None])).sum(1) * (lab_pos >= 0)
            r = torch.where(p_lab > 0, (q_lab / p_lab.clamp_min(1e-12)).clamp(max=1.0), torch.zeros_like(p_lab))
            lam_d = torch.exp(-3.0 * alpha.detach())                      # the LK-hybrid weight (logged for every loss)
            stats = torch.stack([(kl.detach() * m).sum(), (alpha.detach() * m).sum(), (mout * m).sum(),
                                 ((y.argmax(-1) == lab_pos).float() * m).sum(), (wr).sum(), (lam_d * m).sum(),
                                 ((lam_d > 0.5).float() * m).sum()])
        return loss, stats, r

    for k, s in enumerate(samples, start=1):
        labels = tokens[k + 1:L + k + 1]
        mask = loss_mask[k + 1:L + k + 1].float()
        wr_all = mask * w if accept_weights else mask
        loss_sum = 0.0
        st = torch.zeros(7, device=dev)
        rs = []
        for a in range(0, L, chunk):
            b = min(L, a + chunk)
            loss, stats, r = tuc.checkpoint(f, s[a:b], PP[a + k:b + k], PI[a + k:b + k], labels[a:b], mask[a:b], wr_all[a:b], use_reentrant=False)
            loss_sum = loss_sum + loss
            st += stats
            rs.append(r)
        n = mask.sum().clamp(min=1)
        total = total + step_weights[k - 1] * loss_sum / (n if norm is None else norm)
        if norm is not None:
            metrics['_n'] = metrics.get('_n', 0.0) + float(mask.sum()) / K
        w = w * torch.cat(rs)
        for i, name in enumerate(('kl', 'alpha', 'mout', 'top1', 'w', 'lam', 'lamhi')):    # lam: mean LK lambda; lamhi: share > 0.5
            metrics['%s_step%d' % (name, k)] = float(st[i] / n)
    return total, metrics


def token_scale_grads(params, ntok, norm, world):
    """--token-weighted (10-13, user; the standard gradient-accumulation normalisation): every trained position of the
    optimizer step weighs the same, whichever window or rank it came from. Each window's backward used sum / norm; the
    gradients are summed over the data-parallel ranks and scaled by norm / (the step's trained positions over all ranks).
    Per-step position counts differ only at window edges (<= num_steps rows), so one count per window serves every step.
    Returns the global count."""
    n = torch.tensor([float(ntok)], dtype=torch.float64, device=params[0].device)
    if world > 1:
        import torch.distributed as dist
        allreduce_grads(params, 1.0)
        dist.all_reduce(n)
    scale = norm / max(float(n.item()), 1.0)
    for p in params:
        if p.grad is not None:
            p.grad.mul_(scale)
    return float(n.item())


def allreduce_grads(params, divisor):
    """Sum the gradients over the data-parallel ranks and divide by `divisor` (the ranks that had data this step): large
    tensors (the stacked experts) in place, the rest as one flattened bucket; a parameter without a gradient contributes 0."""
    import torch.distributed as dist
    from torch._utils import _flatten_dense_tensors, _unflatten_dense_tensors
    small = []
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
        if p.grad.numel() >= (1 << 24):
            dist.all_reduce(p.grad)
            p.grad.div_(divisor)
        else:
            small.append(p.grad)
    if small:
        flat = _flatten_dense_tensors(small)
        dist.all_reduce(flat)
        flat.div_(divisor)
        for g, x in zip(small, _unflatten_dense_tensors(flat, small)):
            g.copy_(x)


STATE_META = {}      # what a resume must match: the frozen experts' source and the unfreeze step (set in main)


def save_state(path, model, opts, step, order, pending, extra=None):
    """Full training state for stop / resume / branching (10-13, user: resume the full run on a 3-4 GPU pod): trained
    parameters (FP32 masters), every optimizer state, the step, RNG states and the data order (prefetched-but-unused
    windows put back first). Written to a temporary file and renamed, so a crash never leaves a half-written state."""
    st = dict(step=step, model={n: p.detach() for n, p in model.named_parameters() if p.requires_grad},
              opts=[o.state_dict() for o in opts], order=list(order) + list(reversed(list(pending))),
              rng=dict(python=random.getstate(), torch=torch.get_rng_state(),
                       cuda=dict(device=torch.cuda.current_device(), state=torch.cuda.get_rng_state())   # this device only
                       if torch.cuda.is_available() else None), extra=extra or {},
              meta=dict(STATE_META))
    tmp = path + '.tmp'
    torch.save(st, tmp)
    os.replace(tmp, path)


def _retry_io(fn, tries=12, wait=10, what='I/O'):
    """Network-volume robustness (10-13, Phase 6 lesson: a MooseFS EIO while appending a log ended attempt v7r2c): retry
    an OSError (not a missing file) or a safetensors read error every `wait` s, `tries` times in all."""
    for i in range(tries):
        try:
            return fn()
        except FileNotFoundError:
            raise
        except Exception as e:                      # OSError, or safetensors' I/O-wrapped errors
            if not isinstance(e, OSError) and 'safetensor' not in type(e).__name__.lower():
                raise
            if i == tries - 1:
                raise
            print('%s error (%s: %s); retry %d/%d in %d s' % (what, type(e).__name__, e, i + 1, tries - 1, wait), flush=True)
            time.sleep(wait)


class RetryLog:
    """log.jsonl on the volume: lines are buffered and appended by open-write-close on flush(), retrying every 5 s for up to
    10 min (Phase 6's EIO lesson); nothing is dropped while it retries."""

    def __init__(self, path):
        self.path, self.buf = path, []

    def write(self, s):
        self.buf.append(s)

    def flush(self):
        if not self.buf:
            return
        data = ''.join(self.buf)

        def app():
            with open(self.path, 'a') as fh:
                fh.write(data)
        _retry_io(app, tries=121, wait=5, what='log append')
        self.buf = []


def write_json_retry(path, obj):
    """A small JSON file (sidecar, status), written to a temporary name and renamed, with retries."""
    def w():
        tmp = path + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(obj, fh)
        os.replace(tmp, path)
    _retry_io(w, tries=30, wait=10, what='json write')


class StateSaver:
    """Full states (10-13, Phase 6 lessons). With a local directory (--state-local, on the container disk), a state is
    written there first (fast, no network-volume errors) into one scratch slot (atomic rename), then copied to its name in
    the run directory on the volume by a background thread: temporary name, size check, rename, and a sidecar
    <name>.json (step, bytes, time, the stream state) that the resume tools read without loading the 31 GB file. A new
    save first waits for the previous copy; sync=True (stop, end) waits for its own copy. Without a local directory the
    state goes to the volume directly (atomic, with retries). Copy failures are reported (errors) and printed."""

    def __init__(self, out, local=''):
        self.out, self.local, self.th, self.errors = out, local, None, []
        if local:
            os.makedirs(local, exist_ok=True)

    def save(self, name, model, opts, step, order, pending, extra, sync=False):
        dst = os.path.join(self.out, name)
        side = dict(step=step, name=name, time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), **(extra or {}))
        if not self.local:
            _retry_io(lambda: save_state(dst, model, opts, step, order, pending, extra), tries=10, wait=30, what='state save')
            write_json_retry(dst + '.json', dict(side, bytes=os.path.getsize(dst)))
            return dst
        self.wait()                                   # the previous copy may still read the scratch slot
        src = os.path.join(self.local, 'state_scratch.pt')
        save_state(src, model, opts, step, order, pending, extra)
        side['bytes'] = os.path.getsize(src)
        write_json_retry(src + '.json', side)
        self.th = threading.Thread(target=self._copy, args=(src, dst, side), name='state-copy')
        self.th.start()
        if sync:
            self.wait()
        return dst

    def _copy(self, src, dst, side):
        try:
            def cp():
                tmp = dst + '.tmp'
                shutil.copyfile(src, tmp)
                if os.path.getsize(tmp) != side['bytes']:
                    raise OSError('size mismatch copying %s (%d vs %d)' % (src, os.path.getsize(tmp), side['bytes']))
                os.replace(tmp, dst)
            t = time.time()
            _retry_io(cp, tries=10, wait=30, what='state copy')
            write_json_retry(dst + '.json', side)
            print('state %s on the volume (step %d, %.1f GB, %.0f s)' % (os.path.basename(dst), side['step'],
                                                                        side['bytes'] / 1e9, time.time() - t), flush=True)
        except Exception as e:
            self.errors.append('%s: %s' % (dst, e))
            print('STATE COPY FAILED %s: %s' % (dst, e), flush=True)

    def wait(self):
        if self.th is not None:
            self.th.join()
            self.th = None


GROUP_KEYS = ('role', 'lr_scale', 'warm_from', 'warm_len')     # per-group settings that the current run decides


def load_state(st, model, opts, fresh_last=False):
    """Restore a full state (a path or a loaded state). The current run's group settings (GROUP_KEYS) win over the saved
    ones: load_state_dict restores saved hyperparameters, so before 10-13 a resume with a new --router-lr-scale kept the
    saved scale (the IE router guard halves it at a resume). fresh_last: the state predates --unfreeze-experts, whose
    AdamW (the last optimizer) starts fresh."""
    if isinstance(st, str):
        st = torch.load(st, map_location='cpu', weights_only=False)
    if len(st['opts']) != len(opts) - (1 if fresh_last else 0):
        raise SystemExit('the state has %d optimizers, this run %d' % (len(st['opts']), len(opts)))
    named = dict(model.named_parameters())
    with torch.no_grad():
        for n, t in st['model'].items():
            named[n].copy_(t.to(named[n].device))
    want = [[{k: g[k] for k in GROUP_KEYS if k in g} for g in o.param_groups] for o in opts]
    for o, sd in zip(opts, st['opts']):
        o.load_state_dict(sd)
    for o, w in zip(opts, want):
        for g, kv in zip(o.param_groups, w):
            g.update(kv)
    random.setstate(st['rng']['python']); torch.set_rng_state(st['rng']['torch'])
    c = st['rng']['cuda']
    if c is not None and torch.cuda.is_available():  # this device only (10-14: *_all touched GPU 0, the replay's full GPU)
        if isinstance(c, dict):
            torch.cuda.set_rng_state(c['state'])
        else:                                        # states before 10-14: one per visible GPU
            i = torch.cuda.current_device()
            torch.cuda.set_rng_state(c[i] if i < len(c) else c[0])
    return st


def apply_train_mode(model, mode, tiny=False):
    """--train: all (the kit's trainable set), nonexpert (routed experts and router frozen), noexperts (routed experts frozen;
    router and shared expert train; frozen experts held in BF16: the source checkpoint's values, or the production NVFP4
    values with --expert-weights served, load_served_experts), experts (10-13 ablation, user:
    run if frozen experts is competitive: only the MoE block trains, i.e. routed experts, router, shared expert and its gate;
    attention, the input projections and norms, and the hyper-connections are frozen)."""
    for n, p in model.named_parameters():
        if mode in ('nonexpert', 'noexperts'):
            if n.startswith('layers.0.mlp.w_') or (mode == 'nonexpert' and n == 'layers.0.mlp.gate.weight'):
                p.requires_grad_(False)
                if n.startswith('layers.0.mlp.w_') and not tiny:
                    p.data = p.data.to(torch.bfloat16)
        elif mode == 'experts' and not n.startswith('layers.0.mlp.'):
            p.requires_grad_(False)


def init_weights(model, path):
    """--init-from (post-run rounds, IE 10-14): the starting weights of a fresh run (new optimizer). A full-run state (.pt):
    its trained tensors (FP32 masters) copied by name; a step_emb the state lacks stays zero. A head (.safetensors): overlaid
    on the loaded checkpoint as mtp_eval_val50.py --head does. Returns the number of tensors set."""
    named = dict(model.named_parameters())
    if path.endswith('.pt'):
        st = torch.load(path, map_location='cpu', weights_only=False)
        unknown = [n for n in st['model'] if n not in named]
        if unknown:
            raise SystemExit('--init-from: the state has tensors this model lacks (step embeddings: --step-emb): %s' % unknown[:4])
        with torch.no_grad():
            for n, t in st['model'].items():
                named[n].copy_(t.to(named[n].device, named[n].dtype))
        return len(st['model'])
    if SE.head_rows(path) and SE.NAME not in named:
        raise SystemExit('--init-from: the head has step embeddings: --step-emb')
    return len(M.load_from_checkpoint(model, path, None, strict=False))


def load_served_experts(model, draft_dir):
    """--expert-weights served (user 10-13: the frozen routed experts run forward at their test-time weights): the
    production NVFP4 experts of the served drafter (model-mtp-draft.safetensors; only the 512 routed experts are NVFP4
    there, the router, shared expert and its gate are BF16 as here) dequantized and rounded to BF16 exactly as
    mtp_eval_val50.py --experts served scores them, into the frozen expert tensors. Returns the number of matrices."""
    from safetensors import safe_open
    from export_mtp_drafter import nvfp4_dequantize
    mlp, n = model.layers[0].mlp, 0
    with torch.no_grad(), safe_open(os.path.join(draft_dir, 'model-mtp-draft.safetensors'), 'pt') as f:
        for e in range(mlp.cfg.num_experts):
            for proj, attr in (('gate_proj', 'w_gate'), ('up_proj', 'w_up'), ('down_proj', 'w_down')):
                b = 'mtp.layers.0.mlp.experts.%d.%s.' % (e, proj)
                pk, sc = f.get_tensor(b + 'weight'), f.get_tensor(b + 'weight_scale')
                w = nvfp4_dequantize(pk, sc, f.get_tensor(b + 'weight_scale_2'), block=pk.shape[1] * 2 // sc.shape[1])
                dst = getattr(mlp, attr)
                assert not dst.requires_grad, attr
                dst.data[e].copy_(w.to(torch.bfloat16))
                n += 1
    return n


def unfreeze_experts(model):
    """--unfreeze-experts (the escape hatch, IE 10-13): the routed experts train again from the values the frozen phase ran
    with (served or source), as FP32 masters in their own AdamW (fresh moments; LR warmup from the unfreeze step)."""
    ps = []
    for n, p in model.named_parameters():
        if n.startswith('layers.0.mlp.w_'):
            p.data = p.data.float()
            p.requires_grad_(True)
            ps.append(p)
    return ps


def save_head(model, path, trainable_only=False):
    """save_mtp_checkpoint, or (trainable_only, frozen experts) only the trained tensors under the checkpoint's names:
    mtp_eval_val50.py overlays a head on the base non-strictly, so a frozen-experts head is ~0.2 GB instead of 5.2 GB."""
    trained = {n for n, p in model.named_parameters() if p.requires_grad}
    if not trainable_only or any(n.startswith('layers.0.mlp.w_') for n in trained):
        return M.save_mtp_checkpoint(model, path)
    from safetensors.torch import save_file
    out = {'mtp.' + n: p.detach().to(torch.bfloat16).cpu().contiguous() for n, p in model.named_parameters()
           if n in trained and not n.startswith(('embed_tokens', 'lm_head', 'target_mixer'))}
    save_file(out, path, metadata={'format': 'pt', 'source': 'train_mtp_sp.py, trained tensors only'})
    return len(out)


_FLEX = {}


def flex_causal_attention_lse_compiled(q, k, v, scale):
    """qwen38_mtp.flex_causal_attention_lse with flex_attention under torch.compile (fused kernels; the eager version
    materialises the full score matrix) and one shape: rows are zero-padded at the end to a multiple of 1,024 (causal,
    so real queries never see padded keys) and the block mask is cached per padded length."""
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    T = q.shape[0]
    # padded lengths: multiples of 2,048 (at most 9 compiled shapes for 16K windows) and a recompile limit of 64. With
    # 1,024-row padding, 16K windows (shorter at session ends) gave > 8 shapes, torch.compile hit its default recompile
    # limit (8) and fell back to eager flex attention, whose dense backward ran out of memory (10-13 10:36, r3_k5_w16k).
    Tp = max(1024, -(-T // 2048) * 2048)
    if 'fn' not in _FLEX:
        import torch._dynamo
        for lim in ('recompile_limit', 'cache_size_limit'):          # (not `k`: that is the key tensor)
            if hasattr(torch._dynamo.config, lim):
                setattr(torch._dynamo.config, lim, max(64, getattr(torch._dynamo.config, lim)))
        _FLEX['fn'] = torch.compile(flex_attention, dynamic=False)
    bm = _FLEX.get(Tp)
    if bm is None:
        bm = _FLEX[Tp] = create_block_mask(lambda b, h, qi, ki: qi >= ki, 1, None, Tp, Tp, device=q.device)
    pad = lambda t: F.pad(t, (0, 0, 0, 0, 0, Tp - T)) if Tp > T else t
    qq, kk, vv = (pad(t).transpose(0, 1).unsqueeze(0) for t in (q, k, v))
    out, lse = _FLEX['fn'](qq, kk, vv, block_mask=bm, scale=scale, enable_gqa=True, return_lse=True, kernel_options=FLEX_OPTS)
    return out[0, :, :T].transpose(0, 1), lse[0, :, :T].transpose(0, 1).float()


# sm_120 (RTX PRO 6000) has 101,376 B of shared memory per block; the default backward template for head dim 256 needs
# 114,688 B and fails to compile. These tiles compile and match eager to 3e-3 (probe_flex_options.py, 10-13 02:13 UTC).
FLEX_OPTS = dict(fwd_BLOCK_M=64, fwd_BLOCK_N=32, fwd_num_stages=1, bwd_BLOCK_M1=32, bwd_BLOCK_N1=32, bwd_BLOCK_M2=32,
                 bwd_BLOCK_N2=32, bwd_num_stages=1)

SID = re.compile(r'(arc3-[0-9a-f]{32})')


def list_files(specs):
    out = []
    for s in specs:
        out += sorted(glob.glob(os.path.join(s, '*.safetensors'))) if os.path.isdir(s) else sorted(glob.glob(s))
    return out


def window_bounds(n, window, num_steps, min_window=0):
    """(start, end) windows over n rows: stride window - num_steps - 1, so every label is used once; a window needs more
    than num_steps + 2 rows, and with min_window (10-13, user: the full run drops them) at least min_window rows: every
    65,536-row segment otherwise ends in a 20-row window (18% of the full run's windows, 0.03% of its rows), which got a
    full window's weight in a step with ~15 positions and ~20 tokens of context."""
    step = window - num_steps - 1
    least = max(num_steps + 3, min_window)
    return [(a, min(n, a + window)) for a in range(0, max(1, n - num_steps - 1), step) if min(n, a + window) - a >= least]


class Windows:
    """(file, start, end) windows over the export files; loads lazily."""

    def __init__(self, files, streams, window, num_steps, hc=4, pos_key='position_ids', min_window=0):
        self.items = []
        self.pos_key = pos_key
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
            self.items += [(f, a, b) for a, b in window_bounds(n, window, num_steps, min_window)]

    def load(self, item, device):
        """_load with retries on volume I/O errors (12 x 10 s; a missing file is not retried)."""
        return _retry_io(lambda: self._load(item, device), what='window read')

    def _load(self, item, device):
        from safetensors import safe_open
        f, a, b = item
        with safe_open(f, 'pt') as fh:
            tok = fh.get_slice('token_ids')[a:b]
            pos = fh.get_slice(self.pos_key)[a:b]
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


class StreamWindows:
    """Full-run data source (10-13): the replay's tap writer renames each finished segment file into DIR; segments are
    taken as they appear, cut into windows as Windows does, and served at random from a buffer spanning at least `pool`
    segments (concurrent sessions and runs mix). A segment file is deleted once every one of its windows has been trained
    on (the user's standing approval of 10-13: the pipeline's own consumed chunks inside its own RAM directory, nothing
    else). The stream ends when DONE exists and nothing is left. State for resume: the consumed segment names and the
    trained windows of partly used segments; a consumed segment that reappears (replay restart) is deleted unseen."""

    def __init__(self, sdir, streams, window, num_steps, hc, pos_key, pool=8, seed=0, done_file='', rank=0, world=1, min_window=0):
        self.dir, self.window, self.num_steps, self.pool, self.done_file = sdir, window, num_steps, pool, done_file
        self.min_window = min_window                   # a segment without a window this long is consumed unseen
        self.rank, self.world = rank, world             # data parallel: this rank owns the segments with crc32(name) % world == rank
        self.loader = Windows([], streams, window, num_steps, hc=hc, pos_key=pos_key)
        self.load = self.loader.load
        self.rng = random.Random(seed)
        self.consumed, self.known, self.trained_w = set(), set(), {}     # consumed / trained_w by seg_key; known: names
        self.buffer, self.left, self.active = [], {}, {}                  # active: seg_key -> the file being trained
        self.wait_s = 0.0
        self.items = []                         # interface parity with Windows (len() is not meaningful for a stream)

    def _windows(self, f):
        from safetensors import safe_open
        with safe_open(f, 'pt') as fh:
            n = fh.get_slice('token_ids').get_shape()[0]
        return [(f, a, b) for a, b in window_bounds(n, self.window, self.num_steps, self.min_window)]

    def _scan(self):
        """Take new segment files. Segments are identified by session and segment number (seg_key; the request-id suffix
        in the name changes when a resume replays a session again): a consumed segment that reappears is deleted unseen,
        a second copy of a segment still being trained waits until that one is consumed (then it is deleted), and a
        partly trained segment resumes with its untrained windows (10-13: resume and pod-move support)."""
        new = 0
        names = [n for n in os.listdir(self.dir) if n.endswith('.safetensors')
                 and (n not in self.known or seg_key(n) in self.consumed)
                 and (self.world == 1 or zlib.crc32(seg_key(n).encode()) % self.world == self.rank)]

        def mtime(n):
            try:
                return os.stat(os.path.join(self.dir, n)).st_mtime
            except FileNotFoundError:
                return 0
        for name in sorted(names, key=mtime):
            f = os.path.join(self.dir, name)
            key = seg_key(name)
            if key in self.consumed:                     # re-written after a replay restart: already trained on
                self.known.add(name)
                try:
                    os.remove(f)
                except FileNotFoundError:
                    pass
                continue
            if key in self.active:                       # another copy is being trained: decide once it is consumed
                continue
            self.known.add(name)
            done = self.trained_w.get(key, set())
            ws = [w for w in self._windows(f) if w[1] not in done]
            if not ws:
                self._finish(name, f)
                continue
            self.active[key] = f
            self.buffer += ws
            self.left[f] = len(ws)
            new += 1
        return new

    def _finish(self, name, f):
        try:
            os.remove(f)
        except FileNotFoundError:
            pass
        key = seg_key(name)
        self.consumed.add(key)
        self.trained_w.pop(key, None)
        self.left.pop(f, None)
        self.active.pop(key, None)

    def next(self):
        """A window (file, start, end), or None once the stream has ended."""
        while True:
            nfiles = len({w[0] for w in self.buffer})
            if nfiles < self.pool:
                self._scan()
                nfiles = len({w[0] for w in self.buffer})
            if self.buffer and (nfiles >= self.pool or os.path.exists(self.done_file) or self.wait_s > 0):
                i = self.rng.randrange(len(self.buffer))
                self.buffer[i], self.buffer[-1] = self.buffer[-1], self.buffer[i]
                self.wait_s = 0.0
                return self.buffer.pop()
            if not self.buffer and os.path.exists(self.done_file) and not self._scan() and not self.buffer:
                return None
            time.sleep(2)
            self.wait_s += 2

    def trained(self, item):
        f, a, _ = item
        name = os.path.basename(f)
        self.trained_w.setdefault(seg_key(name), set()).add(a)
        self.left[f] -= 1
        if self.left[f] == 0:
            self._finish(name, f)

    def cleanup(self):
        """At the stream's end: delete leftover copies of consumed segments (a resume replay re-wrote them while the first
        copy was still being trained). Only this rank's segments; returns the number deleted."""
        n = 0
        for name in os.listdir(self.dir):
            key = seg_key(name)
            if name.endswith('.safetensors') and key in self.consumed and \
                    (self.world == 1 or zlib.crc32(key.encode()) % self.world == self.rank):
                try:
                    os.remove(os.path.join(self.dir, name))
                    n += 1
                except FileNotFoundError:
                    pass
        return n

    def state(self):
        return dict(consumed=sorted(self.consumed), trained_w={k: sorted(v) for k, v in self.trained_w.items()})

    def load_state(self, st):                           # states written before 10-13 06:3x hold file names: map to keys
        self.consumed = {seg_key(n) for n in st['consumed']}
        self.trained_w = {}
        for k, v in st['trained_w'].items():
            self.trained_w.setdefault(seg_key(k), set()).update(v)


_SEG = re.compile(r'(arc3-[0-9a-f]{32})-[^_/]*_s(\d+)\.safetensors$')


def seg_key(name):
    """A segment's identity across replay attempts: session id + segment number ('arc3-<32 hex>_s0003'); the
    request-id suffix in the file name (arc3-persistent-<session>-<suffix>_s0003.safetensors) differs between attempts.
    Names that do not match (and keys themselves) map to themselves."""
    m = _SEG.search(name)
    if m:
        return '%s_s%s' % (m.group(1), m.group(2))
    return name[:-len('.safetensors')] if name.endswith('.safetensors') else name


def lr_at(step, total, base, warmup, schedule, decay_ratio=0.2, decay_start=-1):
    if step < warmup:
        return base * step / max(1, warmup)
    if schedule == 'cosine':
        return base * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - warmup) / max(1, total - warmup))))
    start = decay_start if decay_start >= 0 else max(warmup, total - int(total * decay_ratio))
    if step < start:
        return base
    return base * 0.5 * (1 + math.cos(math.pi * min(1.0, (step - start) / max(1, total - start))))


@torch.no_grad()
def evaluate(model, windows, items, device, num_steps, attention, wsub=None, sub=None, t2d=None, chunk=2048):
    """Per-step CE / top-1 (step_losses), alpha_k = sum min(p, q) untruncated over the full vocabulary (tau_est), and with
    the subset given the served pair at T 1 (alpha_srv_k: served p' vs the truncated subset draft q', tau_srv; mout_k:
    the served target's mass outside the subset)."""
    model.eval()
    agg = {}
    acc_terms = [[] for _ in range(num_steps)]
    srv_terms = [[] for _ in range(num_steps)]
    srv6_terms = [[] for _ in range(num_steps)]                      # the served pair at T 0.6 (IE 10-13: log T 1 and 0.6)
    out_terms = [[] for _ in range(num_steps)]
    for it in items:
        h, tok, pos, mask = windows.load(it, device)
        with torch.autocast(device_type=device.split(':')[0], dtype=torch.bfloat16, enabled=device != 'cpu'):
            s = model.forward_steps(h, tok, pos, num_steps=num_steps, attention=attention)
            _, met = model.step_losses(s, h, tok, mask)
            L = s[0].shape[0]
            for k in range(1, num_steps + 1):
                rows = torch.nonzero(mask[k + 1:L + k + 1].bool()).squeeze(1)
                for c in range(0, len(rows), chunk):
                    rr = rows[c:c + chunk]
                    pl = model.lm_head(model.teacher_hidden(h[k:L + k][rr])).float()
                    yq = model.lm_head(s[k - 1][rr]).float()
                    acc_terms[k - 1].append(torch.minimum(pl.softmax(-1), yq.softmax(-1)).sum(-1).cpu())
                    if wsub is not None:
                        ys = yq.index_select(1, sub)
                        pp, pi = served_target(pl)
                        qd, qs = served_draft(ys, 1.0)
                        q_on_p = ((pi[:, :, None] == sub[qs][:, None, :]) * qd[:, None, :]).sum(-1)
                        srv_terms[k - 1].append(torch.minimum(pp, q_on_p).sum(-1).cpu())
                        out_terms[k - 1].append((pp * (t2d[pi] < 0)).sum(-1).cpu())
                        pp6, pi6 = served_target(pl / 0.6)
                        qd6, qs6 = served_draft(ys, 0.6)
                        q6 = ((pi6[:, :, None] == sub[qs6][:, None, :]) * qd6[:, None, :]).sum(-1)
                        srv6_terms[k - 1].append(torch.minimum(pp6, q6).sum(-1).cpu())
                    del pl, yq
        for k, v in met.items():
            agg.setdefault(k, []).append(v)
    out = {k: float(np.mean(v)) for k, v in agg.items()}

    def chain(terms, name, tau_name):
        alphas = [float(torch.cat(t).mean()) if t else float('nan') for t in terms]
        tau, prod = 1.0, 1.0
        for a in alphas:
            prod *= a
            tau += prod
        out.update({'%s_step%d' % (name, k + 1): a for k, a in enumerate(alphas)})
        out[tau_name] = tau

    chain(acc_terms, 'alpha', 'tau_est')
    if wsub is not None:
        chain(srv_terms, 'alpha_srv', 'tau_srv')
        chain(srv6_terms, 'alpha_srv06', 'tau_srv06')
        out.update({'mout_step%d' % (k + 1): float(torch.cat(t).mean()) for k, t in enumerate(out_terms) if t})
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
    ap.add_argument('--train', default='all', choices=('all', 'nonexpert', 'noexperts', 'experts'),
                    help='nonexpert: routed experts and router frozen; noexperts: routed experts frozen (router, shared expert '
                         'train); experts: only the MoE block trains (routed experts, router, shared expert)')
    ap.add_argument('--eval-every', type=int, default=200)
    ap.add_argument('--save-every', type=int, default=1000); ap.add_argument('--attention', default='flex')
    ap.add_argument('--device', default='cuda'); ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--pos-key', default='position_ids', choices=('position_ids', 'served_position_ids'))
    ap.add_argument('--weight-decay', type=float, default=0.0); ap.add_argument('--clip', type=float, default=1.0)
    ap.add_argument('--loss', default='cekl', choices=('cekl', 'served_kl', 'tv', 'lk'),
                    help='cekl: CE + kl_weight * full-vocabulary KL (the kit); served_kl / tv / lk: the served pair on the subset')
    ap.add_argument('--cooldown-loss', default='', choices=('', 'served_kl', 'tv', 'lk'), help='loss during the WSD decay')
    ap.add_argument('--loss-temp', type=float, default=1.0); ap.add_argument('--vocab', default=DRAFT_VOCAB)
    ap.add_argument('--accept-weights', action='store_true', help='served losses: accepted-path importance weights w_k')
    ap.add_argument('--accum', type=int, default=1, help='windows per optimizer step')
    ap.add_argument('--decay-ratio', type=float, default=0.2); ap.add_argument('--router-lr-scale', type=float, default=1.0)
    ap.add_argument('--compile-flex', action='store_true'); ap.add_argument('--fused-adam', action='store_true')
    ap.add_argument('--save-at', default='', help='comma-separated steps to save (overrides --save-every)')
    ap.add_argument('--ckpt-steps', action='store_true', help='activation checkpoint per draft step (memory for K 5 / long windows)')
    ap.add_argument('--prefetch', type=int, default=0, help='windows loaded ahead on CPU threads while the GPU computes')
    ap.add_argument('--optimizer', default='adamw', choices=('adamw', 'muon'),
                    help='muon: torch.optim.Muon on the 2D non-expert matrices (match-RMS LR, momentum 0.95, weight decay\n'
                         '--muon-wd), AdamW on the rest (stacked experts, norms, vectors), as Speculators splits them')
    ap.add_argument('--muon-wd', type=float, default=0.0)
    ap.add_argument('--ckpt-every', type=int, default=0, help='full training state every N steps (two rolling slots)')
    ap.add_argument('--resume', default='', help='a full training state to continue from (or branch: with --decay-start)')
    ap.add_argument('--keep-every', type=int, default=0, help='also keep (never overwrite) a full state every N steps: branch points')
    ap.add_argument('--state-local', default='', help='write full states to this local directory first (container disk), '
                    'then copy them to --out in the background (one scratch slot locally)')
    ap.add_argument('--stop-file', default='', help='graceful stop: when this file exists (default <out>/STOP), save a full '
                    'state at the end of the step, rename the file to <file>.done_step<N> and exit 0')
    ap.add_argument('--decay-start', type=int, default=-1, help='absolute step where the WSD decay starts (default: from --decay-ratio)')
    ap.add_argument('--stream-dir', default='', help='full run: train on segment files streamed into this RAM directory (deleted once used)')
    ap.add_argument('--stream-done', default='', help='file the replay side creates when it has finished')
    ap.add_argument('--stream-pool', type=int, default=8, help='segments mixed in the shuffle buffer')
    ap.add_argument('--val-sessions', type=int, default=0, help='hold out whole sessions for validation (overrides --val-files)')
    ap.add_argument('--val-windows', type=int, default=0, help='cap on validation windows (a fixed random subset)')
    ap.add_argument('--save-trainable-only', action='store_true', help='heads hold the trained tensors only (frozen experts omitted)')
    ap.add_argument('--devices', default='', help='data-parallel (torchrun): CUDA device per local rank, e.g. 1,2,3')
    ap.add_argument('--expert-weights', default='source', choices=('source', 'served'),
                    help='frozen routed experts (--train noexperts | nonexpert): source = the BF16 checkpoint (the rounds); '
                         'served = the production NVFP4 experts as scored (the full run, user 10-13), from --draft-config\'s dir')
    ap.add_argument('--unfreeze-experts', action='store_true', help='escape hatch, at a resume of a frozen-experts run: the '
                    'routed experts train from the resumed step on, in their own AdamW (IE 10-13); later resumes repeat it')
    ap.add_argument('--expert-lr-scale', type=float, default=0.5, help='--unfreeze-experts: expert LR / main LR after the warmup')
    ap.add_argument('--expert-warmup', type=int, default=200, help='--unfreeze-experts: linear LR warmup steps from 0')
    ap.add_argument('--min-window', type=int, default=0, help='drop windows shorter than this (the full run: 2048; 10-13)')
    ap.add_argument('--token-weighted', action='store_true', help='served losses: every trained position of an optimizer '
                    'step weighs the same across its windows and ranks (10-13); default: each window\'s own mean, windows equal')
    ap.add_argument('--step-emb', action='store_true', help='post-run round, form A (IE 10-14): a learned per-step embedding '
                    'added to the fuse output of draft steps >= 2 (step_emb.py; zero init, own AdamW group, no weight decay)')
    ap.add_argument('--step-emb-lr-scale', type=float, default=10.0, help='--step-emb: its LR / the base LR')
    ap.add_argument('--trunk-lr-scale', type=float, default=1.0, help='LR scale of every other trained tensor (0 = the '
                    'trunk held: still saved in the head, never updated)')
    ap.add_argument('--init-from', default='', help='start (fresh optimizer) from a full-run state (.pt: its trained FP32 '
                    'tensors) or a head (.safetensors, overlaid on --mtp-file), e.g. the finished full run')
    ap.add_argument('--fresh-stream', action='store_true', help='with --resume in stream mode: restore the model, optimizers, '
                    'step and RNG but not the stream state, so a new replay of sessions the state already trained is read '
                    '(the fixed-teacher refit from the 10k state, user 10-14)')
    ap.add_argument('--stop-at', type=int, default=0, help='graceful stop right after this step, as the stop file does (full '
                    'state, status stopped); in-process, so no external timing (10-15: an external helper missed step 10,000)')
    ap.add_argument('--tiny', action='store_true', help='tests: a tiny random model instead of the checkpoint')
    a = ap.parse_args()
    if a.fresh_stream and not (a.resume and a.stream_dir):
        raise SystemExit('--fresh-stream: with --resume and --stream-dir only')
    if a.token_weighted and 'cekl' in (a.loss, a.cooldown_loss or a.loss):
        raise SystemExit('--token-weighted: served losses only (served_kl | tv | lk)')
    world = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    if world > 1:                        # data parallel: one process per GPU, gradients averaged before every step
        import torch.distributed as dist
        lrank = int(os.environ.get('LOCAL_RANK', '0'))
        if a.device != 'cpu':
            a.device = 'cuda:%s' % (a.devices.split(',')[lrank] if a.devices else lrank)
            torch.cuda.set_device(a.device)
        dist.init_process_group('gloo' if a.device == 'cpu' else 'nccl')
    elif a.device.startswith('cuda:'):    # 10-14: else the current device stays cuda:0 (fused Adam's stream check, RNG),
        torch.cuda.set_device(a.device)    # where the replay server leaves < 1 GB: a milestone branch died of OOM there
    is_main = rank == 0
    if not is_main:                      # only rank 0 prints and logs
        import builtins
        builtins.print = lambda *x, **k: None
    random.seed(a.seed); torch.manual_seed(a.seed)
    if is_main:
        os.makedirs(a.out, exist_ok=bool(a.resume))              # resuming may continue in the run's own directory
        json.dump(dict(vars(a), world=world), open(os.path.join(a.out, 'args_resume.json' if a.resume else 'args.json'), 'w'), indent=1)
    if world > 1:
        dist.barrier()
    files = list_files(a.data)
    random.shuffle(files)
    if a.val_sessions:              # session-disjoint validation: every segment of a.val_sessions held-out sessions
        sids = sorted({SID.search(os.path.basename(f)).group(1) for f in files})
        held = set(random.Random(a.seed + 1).sample(sids, a.val_sessions))
        val_files = [f for f in files if SID.search(os.path.basename(f)).group(1) in held]
        train_files = [f for f in files if SID.search(os.path.basename(f)).group(1) not in held]
    else:
        val_files, train_files = files[:a.val_files], files[a.val_files:]
    if a.tiny:
        import test_qwen38_mtp_cpu as T
        model = T.tiny_model()
    else:
        cfg = M.MTPConfig.from_text_config(json.load(open(a.draft_config))['text_config'])
        model = M.Qwen38MTP(cfg)
        M.load_from_checkpoint(model, a.mtp_file, a.target_dir, strict=True)
    model.to(a.device)
    if a.compile_flex:
        M.flex_causal_attention_lse = flex_causal_attention_lse_compiled
    if a.ckpt_steps:
        M.Qwen38MTP.forward_steps = forward_steps_ckpt
    apply_train_mode(model, a.train, a.tiny)
    frozen = a.train in ('nonexpert', 'noexperts')
    if a.expert_weights == 'served':
        if not frozen:
            raise SystemExit('--expert-weights served: frozen routed experts only (--train noexperts | nonexpert)')
        n = load_served_experts(model, os.path.dirname(a.draft_config))
        print('routed experts: the served NVFP4 weights (%d matrices, dequantized to BF16 as scored)' % n, flush=True)
    if a.step_emb:
        SE.install(model, a.num_steps - 1, device=a.device)
        M.Qwen38MTP.forward_steps = forward_steps_ckpt          # the step-aware unroll (the reference one has no step index)
        print('step embeddings: steps 2-%d, zero init, %.2fx LR' % (a.num_steps, a.step_emb_lr_scale), flush=True)
    if a.init_from:
        if a.resume:
            raise SystemExit('--init-from: a fresh start, not with --resume')
        print('initial weights: %s (%d tensors)' % (a.init_from, init_weights(model, a.init_from)), flush=True)
    st0 = torch.load(a.resume, map_location='cpu', weights_only=False) if a.resume else None
    meta0 = (st0 or {}).get('meta', {})
    if st0 is not None and frozen and meta0.get('expert_weights', 'source') != a.expert_weights:
        raise SystemExit('the state ran with --expert-weights %s' % meta0.get('expert_weights', 'source'))
    unfreeze_at = 0
    if a.unfreeze_experts:
        if not (frozen and st0 is not None):
            raise SystemExit('--unfreeze-experts: at a resume of a frozen-experts run (--train noexperts | nonexpert)')
        unfreeze_at = meta0.get('unfreeze_at') or st0['step'] + 1
    elif meta0.get('unfreeze_at'):
        raise SystemExit('the state trained its experts from step %d: resume with --unfreeze-experts, or from an earlier '
                         'state to revert' % meta0['unfreeze_at'])
    STATE_META.update(train=a.train, expert_weights=a.expert_weights, unfreeze_at=unfreeze_at)
    if a.step_emb:
        STATE_META.update(step_emb=a.num_steps - 1)
    if st0 is not None and meta0.get('step_emb', 0) != STATE_META.get('step_emb', 0):
        raise SystemExit('the state has step embeddings for %d steps, this run %d' % (meta0.get('step_emb', 0), STATE_META.get('step_emb', 0)))
    experts = unfreeze_experts(model) if unfreeze_at else []          # unfrozen experts: their own optimizer, not the groups
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad and not any(p is q for q in experts)]
    params = [p for _, p in named] + experts
    groups = [dict(params=[p for n, p in named if n not in ('layers.0.mlp.gate.weight', SE.NAME)], lr_scale=a.trunk_lr_scale, role='main')]
    router = [p for n, p in named if n == 'layers.0.mlp.gate.weight']
    if router:
        groups.append(dict(params=router, lr_scale=a.router_lr_scale * a.trunk_lr_scale, role='router'))
    stepemb = [p for n, p in named if n == SE.NAME]
    if stepemb:
        groups.append(dict(params=stepemb, lr_scale=a.step_emb_lr_scale, role='stepemb', weight_decay=0.0))
    opts = []
    if a.optimizer == 'muon':                 # Muon: 2D matrices with both dims > 1 (not the stacked 3D experts)
        # the router (hidden -> expert logits, embedding-like) stays on AdamW at --router-lr-scale, as Muon practice keeps
        # such matrices and as the frozen-experts round arm measured it (IE 10-13, the full run: frozen routed experts)
        is_muon = lambda n, p: p.ndim == 2 and min(p.shape) > 1 and n not in ('layers.0.mlp.gate.weight', SE.NAME)
        mu = [p for n, p in named if is_muon(n, p)]
        groups = [dict(g, params=[p for p in g['params'] if not any(p is q for q in mu)]) for g in groups]
        groups = [g for g in groups if g['params']]
        opts.append(torch.optim.Muon([dict(params=mu, lr_scale=a.trunk_lr_scale, role='muon')], lr=a.lr, weight_decay=a.muon_wd, momentum=0.95,
                                     ns_steps=5, adjust_lr_fn='match_rms_adamw'))
        print('Muon: %d matrices (%.1fM params); AdamW: %d tensors (router %s at %.2fx)' % (
              len(mu), sum(p.numel() for p in mu) / 1e6, sum(len(g['params']) for g in groups),
              'trained' if router else 'frozen', a.router_lr_scale), flush=True)
    opts.append(torch.optim.AdamW(groups, lr=a.lr, betas=(0.9, 0.95), weight_decay=a.weight_decay, **({'fused': True} if a.fused_adam else {})))
    opt = opts[-1]
    if experts:                               # always the last optimizer (a state from before the unfreeze has one fewer)
        opts.append(torch.optim.AdamW([dict(params=experts, lr_scale=a.expert_lr_scale, role='experts', warm_from=unfreeze_at,
                                            warm_len=max(1, a.expert_warmup))], lr=a.lr, betas=(0.9, 0.95),
                                      weight_decay=a.weight_decay, **({'fused': True} if a.fused_adam else {})))
        print('routed experts unfrozen from step %d: own AdamW, %.2fx LR after a %d-step warmup (%.2fB params, FP32 masters)' % (
              unfreeze_at, a.expert_lr_scale, a.expert_warmup, sum(p.numel() for p in experts) / 1e9), flush=True)
    mlp = model.layers[0].mlp
    mlp._load = torch.zeros(mlp.cfg.num_experts, dtype=torch.long, device=a.device)
    sub = t2d = wsub = None
    if a.loss != 'cekl' or a.cooldown_loss or a.eval_every:
        sub, t2d = load_subset('tiny' if a.tiny else a.vocab, model.lm_head.weight.shape[0], a.device)
        wsub = model.lm_head.weight.index_select(0, sub).contiguous()
    if a.stream_dir:
        tr = StreamWindows(a.stream_dir, a.streams, a.window, a.num_steps, model.cfg.hc_count, a.pos_key, a.stream_pool,
                           a.seed + rank, a.stream_done, rank, world, a.min_window)
        train_files = []
    else:
        tr = Windows(train_files, a.streams, a.window, a.num_steps, hc=model.cfg.hc_count, pos_key=a.pos_key,
                     min_window=a.min_window)
    va = Windows(val_files, a.streams, a.window, a.num_steps, hc=model.cfg.hc_count, pos_key=a.pos_key, min_window=a.min_window)
    if a.val_windows and len(va.items) > a.val_windows:
        va.items = sorted(random.Random(a.seed + 2).sample(va.items, a.val_windows))
    print('train windows %d (%d files), val windows %d (%d files); trainable %.3fB; loss %s (cooldown %s), accum %d, train %s' % (
        len(tr.items), len(train_files), len(va.items), len(val_files), sum(p.numel() for p in params) / 1e9, a.loss,
        a.cooldown_loss or a.loss, a.accum, a.train), flush=True)
    weights = [float(x) for x in a.step_weights.split(',')][:a.num_steps]
    decay_start = (a.decay_start if a.decay_start >= 0 else max(a.warmup, a.steps - int(a.steps * a.decay_ratio))) \
        if a.schedule == 'wsd' else a.steps + 1
    log = RetryLog(os.path.join(a.out, 'log.jsonl')) if is_main else open(os.devnull, 'w')
    saver = StateSaver(a.out, a.state_local) if is_main else None
    stop_file = a.stop_file or os.path.join(a.out, 'STOP')
    status_file = os.path.join(a.out, 'trainer_status.json')
    order = []
    t0 = time.time()

    def run_eval(step):
        ev = evaluate(model, va, va.items, a.device, a.num_steps, a.attention, wsub, sub, t2d)
        log.write(json.dumps(dict(step=step, **{'val_' + k: v for k, v in ev.items()})) + '\n'); log.flush()
        print('step %d val' % step, ev, flush=True)

    order_rng = [random]                   # files mode: the global RNG (restored on resume); see the resume below

    def next_item():                       # an index into tr.items, or (stream) a window item / None at the end
        if a.stream_dir:
            return tr.next()
        if not order:
            order.extend(range(len(tr.items)))
            order_rng[0].shuffle(order)
        return order.pop()

    item_of = (lambda i: i) if a.stream_dir else (lambda i: tr.items[i])

    from collections import deque
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(max_workers=max(1, a.prefetch)) if a.prefetch else None
    ahead, ahead_idx = deque(), deque()
    start = 1
    if a.resume:
        st = load_state(st0, model, opts, fresh_last=bool(experts) and not meta0.get('unfreeze_at'))
        start = st['step'] + 1
        if a.stream_dir:                   # the stream re-derives untrained windows from its state (trained_w)
            order[:] = []
        elif all(isinstance(i, int) for i in st['order']):
            order[:] = st['order']
        else:                              # files mode from a stream-mode state (milestone branch cooldowns, 10-14): its
            order[:] = []                  # order holds stream windows; start a fresh order from a fixed seed, so every
            order_rng[0] = random.Random(a.seed)       # milestone's branch sees the same data in the same order
        if a.stream_dir and not a.fresh_stream:
            tr.load_state(st['extra']['stream'])
        elif a.stream_dir:                 # --fresh-stream: a new replay of sessions this state already trained
            print('fresh stream: the state\'s consumed segments and trained windows are not restored', flush=True)
        print('resumed from %s at step %d (decay from %d, total %d)' % (a.resume, st['step'], decay_start, a.steps), flush=True)
        st = st0 = None                                  # the loaded tensors are in the model and optimizers now
    elif a.eval_every and is_main:
        run_eval(0)
    slot, ended, last, stopped_at, stream_over = 0, [False], start - 1, None, False
    if is_main:
        if os.path.exists(stop_file):                    # a stale stop request would end this attempt after one step
            stale = stop_file + '.stale_%d' % int(time.time())
            os.replace(stop_file, stale)
            print('stale stop file moved to %s' % stale, flush=True)
        write_json_retry(status_file, dict(status='running', from_step=start - 1, world=world,
                                           time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    def extra():                         # the stream state; with data parallel the union over ranks (a collective)
        if not a.stream_dir:
            return None
        st = tr.state()
        if world > 1:
            allst = [None] * world
            dist.all_gather_object(allst, st)
            st = dict(consumed=sorted(set().union(*[x['consumed'] for x in allst])),
                      trained_w={k: v for x in allst for k, v in x['trained_w'].items()})
        return dict(stream=st)
    for step in range(start, a.steps + 1):
        t_step = time.time()
        lr = lr_at(step, a.steps, a.lr, a.warmup, a.schedule, a.decay_ratio, decay_start)
        for o in opts:
            for g in o.param_groups:
                g['lr'] = lr * g['lr_scale'] * (min(1.0, (step - g['warm_from'] + 1) / g['warm_len']) if 'warm_from' in g else 1.0)
        kind = a.cooldown_loss if (a.cooldown_loss and step >= decay_start) else a.loss
        for o in opts:
            o.zero_grad(set_to_none=True)
        tot, mets, ntok = 0.0, [], 0.0
        used = []
        for _ in range(a.accum):
            if a.prefetch:
                while len(ahead) <= a.prefetch and not ended[0]:
                    i = next_item()
                    if i is None:
                        ended[0] = True
                        break
                    ahead_idx.append(i)
                    ahead.append(pool.submit(tr.load, item_of(i), 'cpu'))
                if not ahead:
                    break
                i = ahead_idx.popleft()
                h, tok, pos, mask = (t.to(a.device, non_blocking=True) for t in ahead.popleft().result())
            else:
                i = None if ended[0] else next_item()
                if i is None:
                    ended[0] = True
                    break
                h, tok, pos, mask = tr.load(item_of(i), a.device)
            used.append(i)
            with torch.autocast(device_type=a.device.split(':')[0], dtype=torch.bfloat16, enabled=a.device != 'cpu'):
                s = model.forward_steps(h, tok, pos, num_steps=a.num_steps, attention=a.attention)
                if kind == 'cekl':
                    loss, met = model.step_losses(s, h, tok, mask, step_weights=weights, kl_weight=a.kl_weight)
                else:
                    loss, met = step_losses_served(model, s, h, tok, mask, weights, kind, wsub, t2d, a.loss_temp, a.accept_weights,
                                                   norm=float(a.window) if a.token_weighted else None)
            TRACK[0] = False
            (loss if a.token_weighted else loss / a.accum).backward()
            TRACK[0] = True
            if a.token_weighted:
                ntok += met['_n']
                tot += float(loss.detach()) * a.window         # the sum; divided by this rank's positions below
            else:
                tot += float(loss) / a.accum
            mets.append(met)
            del s, loss
        if world > 1:            # train while any rank has data; a rank without data adds zero gradients, and the sum is
            have = torch.tensor([1.0 if used else 0.0], device=a.device)        # divided by the ranks that had data
            dist.all_reduce(have)
            if have.item() == 0:
                ended[0] = stream_over = True
                break
            if a.token_weighted:
                token_scale_grads(params, ntok, float(a.window), world)
            else:
                allreduce_grads(params, have.item())
        elif not used:                                 # the stream has ended
            stream_over = True
            break
        elif a.token_weighted:
            token_scale_grads(params, ntok, float(a.window), 1)
        if a.token_weighted:
            tot = tot / max(ntok, 1.0)                 # this rank's token-weighted loss (rank 0 logs it)
        gns = {}                                       # IE 10-13: gradient norms per optimizer group (before clipping)
        for o in opts:
            for g in o.param_groups:
                gs = [p.grad for p in g['params'] if p.grad is not None]
                if gs:
                    gns['gn_' + g.get('role', 'main')] = float(torch.linalg.vector_norm(torch.stack(
                        [torch.linalg.vector_norm(x.float()) for x in gs])))
        gn = torch.nn.utils.clip_grad_norm_(params, a.clip)
        for o in opts:
            o.step()
        if a.stream_dir:
            for i in used:
                tr.trained(i)                          # deletes a segment once all its windows are trained
        if a.token_weighted and mets:                  # the metrics weighted like the loss
            wts = [m.pop('_n') for m in mets]
            met = {k: float(np.average([m[k] for m in mets], weights=wts)) if sum(wts) > 0 else 0.0 for k in mets[0]}
            met['ntok'] = float(sum(wts))
        else:
            met = {k: float(np.mean([m[k] for m in mets])) for k in mets[0]} if mets else {}
        load = mlp._load.double()
        pl = load / load.sum().clamp_min(1)
        met['load_entropy'] = float(-(pl * torch.log(pl.clamp_min(1e-30))).sum() / math.log(len(pl)))
        met['load_max'] = float(pl.max() * len(pl))
        met['load_dead'] = int((load == 0).sum())                 # experts with no tokens in this step (router guard)
        mlp._load.zero_()
        rec = dict(step=step, loss=tot, grad_norm=float(gn), lr=lr, kind=kind, sec=round(time.time() - t0, 1),
                   mem_gb=round(torch.cuda.max_memory_allocated(a.device) / 2**30, 1) if a.device.startswith('cuda') else 0.0, **met)
        rec.update({'lr_' + g['role']: g['lr'] for o in opts for g in o.param_groups if g.get('role') in ('router', 'experts', 'stepemb')})
        rec.update(gns, step_s=round(time.time() - t_step, 2))
        last = step
        if a.stream_dir:
            rec.update(stream_wait_s=tr.wait_s, stream_segments=len(tr.left), stream_consumed=len(tr.consumed))
        log.write(json.dumps(rec) + '\n')
        if step % 10 == 0:
            log.flush()
            print(json.dumps(rec), flush=True)
        if a.eval_every and step % a.eval_every == 0 and is_main:
            run_eval(step)
        save_at = {int(x) for x in a.save_at.split(',') if x} if a.save_at else None
        if is_main and ((step in save_at) if save_at is not None else (a.save_every > 0 and (step % a.save_every == 0 or step == a.steps))):
            n = save_head(model, os.path.join(a.out, 'head_step%d.safetensors' % step), a.save_trainable_only)
            print('saved head_step%d (%d tensors)' % (step, n), flush=True)
        do_ck = a.ckpt_every and (step % a.ckpt_every == 0 or step == a.steps)    # two rolling slots, the run's own states
        do_keep = a.keep_every and step % a.keep_every == 0                       # kept: milestone branch points
        do_stable = a.ckpt_every and a.schedule == 'wsd' and step == decay_start - 1   # kept: the last stable-phase state
        saved_rolling = None
        if do_ck or do_keep or do_stable:
            ex = extra()                                                        # a collective in data parallel
            if is_main and do_ck:
                saved_rolling = saver.save('state_%s.pt' % 'ab'[slot], model, opts, step, order, ahead_idx, ex)
                print('saved state %s at step %d' % (os.path.basename(saved_rolling), step), flush=True)
            slot = 1 - slot if do_ck else slot
            if is_main and do_keep:
                saver.save('state_keep_step%d.pt' % step, model, opts, step, order, ahead_idx, ex)
                print('saved state_keep_step%d.pt' % step, flush=True)
            if is_main and do_stable:
                saver.save('state_stable_step%d.pt' % step, model, opts, step, order, ahead_idx, ex)
                print('saved state_stable_step%d.pt' % step, flush=True)
        # graceful stop (10-13, Phase 6 lesson: a stop just before a save lost ~790 steps): rank 0 sees the stop file,
        # every rank learns it in the same step, the full state goes to a rolling slot (copied to the volume before exit)
        stop = is_main and (os.path.exists(stop_file) or (a.stop_at > 0 and step >= a.stop_at))   # --stop-at: in-process
        if world > 1:
            flag = torch.tensor([1.0 if stop else 0.0], device=a.device)
            dist.broadcast(flag, 0)
            stop = flag.item() > 0
        if stop:
            ex = extra() if not do_ck else None                                # same branch on every rank
            if is_main:
                if not do_ck:
                    saved_rolling = saver.save('state_%s.pt' % 'ab'[slot], model, opts, step, order, ahead_idx, ex, sync=True)
                saver.wait()                                                    # the state is on the volume before exit
                log.flush()
                try:
                    if os.path.exists(stop_file):                   # absent when --stop-at stopped the run
                        os.replace(stop_file, stop_file + '.done_step%d' % step)
                except OSError as e:
                    print('could not rename %s: %s' % (stop_file, e), flush=True)
                write_json_retry(status_file, dict(status='stopped', step=step, state=saved_rolling,
                                                   copy_errors=saver.errors, time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
                print('stopped by %s at step %d; state %s' % ('--stop-at' if a.stop_at and step >= a.stop_at else stop_file,
                                                              step, saved_rolling), flush=True)
            stopped_at = step
            break

    # the stream ended for every rank before --steps: keep the head and the state. (ended[0] is per rank: one rank's own
    # stream can end while others still train, so it must not choose a branch with collectives: 10-13 two-rank deadlock.)
    if stream_over:
        if a.stream_dir:
            n = tr.cleanup()
            if n:
                print('rank %d: deleted %d leftover copies of consumed segments' % (rank, n), flush=True)
        ex = extra() if a.ckpt_every else None
        if is_main:
            print('stream ended after step %d' % last, flush=True)
            n = save_head(model, os.path.join(a.out, 'head_step%d.safetensors' % last), a.save_trainable_only)
            print('saved head_step%d (%d tensors)' % (last, n), flush=True)
            if a.ckpt_every:
                saver.save('state_end_step%d.pt' % last, model, opts, last, order, ahead_idx, ex, sync=True)
    if is_main:
        saver.wait()
        log.flush()
        if stopped_at is None:
            write_json_retry(status_file, dict(status='stream_ended' if stream_over else 'finished', step=last,
                                               copy_errors=saver.errors, time=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())))
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()
    if is_main and saver.errors:
        print('state copies failed: %s' % saver.errors, flush=True)
        sys.exit(3)


if __name__ == '__main__':
    main()
