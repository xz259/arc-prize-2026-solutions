"""mtp_eval_val50.py (2026-10-12, Claude fork 5d20b23b): SIDE_POD_PLAN section 3, the production MTP3 drafter on the
Phase 6 validation windows (val50), scored like DFlash's validation (dflash_20261006/eval_accept_modes.py: teacher-forced
per-slot acceptance, per-anchor products, the assistant mask).

Data: the side pod's val50 export with the MTP input (sp_export_val50mtp.sh): per session the stream rows
[start - 16,384, end), hidden_states [N, 10, 2560] = taps 2,13,24,35,46, the final state, the 4 pre-final-mixer streams;
token_ids, position_ids (raw stream index = the drafter's RoPE position), served_position_ids. Windows and loss masks:
dflash_packed_20261011/cache_val50 (segments.jsonl, rows.jsonl: the 16,384 window rows, the first 1,024 mask-free).

Drafter (production as served; MTP_BASELINE_SCOPING_2026-10-12.md section 2):
  - weights: the BF16 head (model_extra_tensors.safetensors, mtp.*) with the routed experts replaced by the served NVFP4
    experts dequantized (--experts served; --experts bf16 = the BF16 reference); a fine-tuned head (--head, the
    save_mtp_checkpoint names) with --experts bf16, or requant (its experts NVFP4-quantized as export_mtp_drafter.py
    would serve them, then dequantized), or served (10-13: a head trained with frozen experts, scored with the
    production NVFP4 experts unchanged);
  - attention (--attention qsa): the QSA indexer emulated: q = index_qk_proj -> 4 heads x 128, Gemma norm, RoPE at the
    query position; group keys = mean of the 4 raw keys of each complete 4-position group, Gemma norm, RoPE at the
    group's first position; score = sum_h relu(q.k) / sqrt(128); the top 512 groups (all when fewer are visible) plus the
    query's incomplete group; every draft step selects again (index_share_for_mtp_iteration is unset in production);
    steps >= 2 also attend to their own chain (positions pos + 1 .. pos + k - 1; always kept). --attention dense: every
    row <= the query (the dense reference). Context = the exported rows only (16K before the window; job 145: a 16K MTP
    window is within 0.1% of production). --index-share (10-13, IE): vLLM's index_share_for_mtp_iteration emulated: steps
    >= 2 reuse step 1's token selection (indexer_qsa.py returns the step-0 buffer under skip_topk; mtp.py
    compact_topk_indices keeps each request's step-1 row) and, since the reader attends exactly those indices
    (qsa.py forward_qsa), the in-draft chain pos + 1 .. pos + k - 1 is not attended. --draftwin W (10-13): stage2_v1/draftwin.py's serving window, only groups
    inside the query's last W positions are selectable (group g masked when g < floor((q - W + 1) / 4)); 0 = off. --kv lutc (default): the drafter's K/V as production stores them
    (ARC3_S2_KVROT_DRAFT=lutc: centred by the drafter layer's means in kvlut_means_10020543.pt, Hadamard-rotated,
    4-bit codebook with block-MSE E4M3 scales; stage2_v1/kvfq.fake_quant 'h_lutpm+c', the kernels' reference); --kv bf16
    unquantized. Not emulated: eviction-shifted group boundaries (groups and RoPE use position_ids).
  - draft distribution: the BF16 lm_head rows of the 32K subset (draft_vocab_32k.json); top-20, then the top-p 0.95
    prefix of the raw logits (ARC3_S2_DRAFTTRUNC=20,0.95), then T (probabilistic); greedy = argmax over the subset.
Target: p = lm_head(final state of row a + k - 1) / T, served as T -> top-k 20 -> top-p 0.95 (as eval_accept_modes).
Anchors: window rows a with loss_mask[a] = 1 and a + K inside the window (DFlash: select_anchors over valid rows); slot k
(predicting x_{a+k}) counts where loss_mask[a + k] = 1, a 0 ends the block (aggregate()). Step 1 of anchor a runs on row
a - 1: (h_{a-1}, x_a) at position pos_{a-1} (the bonus token x_a is known, its state is not).
--trunc-sweep K:P,... (10-13): extra modes prob_srv_T<t>_k<K>p<P*100> with the draft truncated at top-K / top-p P instead
of DRAFTTRUNC 20 / 0.95 (the target's served truncation unchanged), from the same logits (no extra pass).
Modes per T: prob_srv sum_v min(p'_T, q'_T) (production's probabilistic drafts); greedy_srv p'_T(argmax q); ceil_srv
max p'_T; prob_full sum_v min(p_T, q_T) (both untruncated; q on the subset); and greedy_T0 [argmax q == argmax p].
Output: json with, per mode, slots 1..K mean acceptance, L_1..L_K (1 + sum_k prod_{j<=k} a_j) and the conditional
ratios r_1 = L_1 - 1, r_k = (L_k - L_{k-1}) / (L_{k-1} - L_{k-2}); per-session sums; per-anchor tensors (--save-anchors).
usage: python mtp_eval_val50.py --features DIR --out OUT.json [--experts served|bf16] [--attention qsa|dense] [--K 5]
       [--sessions N] [--device cuda:1]"""
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import step_emb as SE  # noqa: E402  (heads with step embeddings, IE 10-14 form A)

TEMPS = (1.0, 0.6)
TOPK, TOPP = 20, 0.95             # served target: T, top-k 20, top-p 0.95
DRAFT_K, DRAFT_P = 20, 0.95       # ARC3_S2_DRAFTTRUNC=20,0.95 on the raw subset logits
GROUP, BLOCK_TOPK = 4, 512        # QSA: 4-token groups, 2,048 tokens = 512 groups
SLOT_STREAMS, SLOT_FINAL = slice(6, 10), 5


def served_target(logits_t):
    """eval_accept_modes._served: top-k then top-p over T-scaled logits [n, V] -> (probs [n, 20], ids [n, 20])."""
    v, i = logits_t.topk(TOPK, dim=-1)
    p = torch.softmax(v, dim=-1)
    keep = (p.cumsum(-1) - p) < TOPP
    p = p * keep
    return p / p.sum(-1, keepdim=True), i


def served_draft(y, t, dk=None, dp=None):
    """drafthead.truncate at T = 1 (top-20 of the subset logits, smallest prefix reaching 0.95), then T: y [n, S] subset
    logits -> (probs [n, 20], subset positions [n, 20])."""
    v, i = torch.topk(y.float(), DRAFT_K if dk is None else dk, dim=1)
    q1 = torch.softmax(v, dim=1)
    v = torch.where(torch.cumsum(q1, dim=1) - q1 < (DRAFT_P if dp is None else dp), v, float('-inf'))
    return torch.softmax(v / t, dim=1), i


def load_model(args, dev):
    sys.path.insert(0, args.kit)
    import qwen38_mtp as Q
    tc = json.load(open(os.path.join(args.target, 'config.json')))
    tc = tc.get('text_config', tc)
    cfg = Q.MTPConfig.from_text_config(tc)
    model = Q.Qwen38MTP(cfg, with_teacher=False).to(torch.bfloat16)
    Q.load_from_checkpoint(model, os.path.join(args.target, 'model_extra_tensors.safetensors'), args.target)
    if args.head:                                       # a fine-tuned head (save_mtp_checkpoint names) over the base
        rows = SE.head_rows(args.head)                  # step embeddings must exist before the non-strict load fills them
        if rows:
            SE.install(model, rows, dtype=torch.bfloat16)
            print('head with step embeddings for steps 2-%d' % (rows + 1), flush=True)
        filled = Q.load_from_checkpoint(model, args.head, None, strict=False)
        if rows and SE.NAME not in filled:
            raise SystemExit('step embeddings not loaded from %s' % args.head)
    if args.experts == 'requant':                       # the head's experts as they would be served (export_mtp_drafter)
        from export_mtp_drafter import expert_scale_2, nvfp4_dequantize, nvfp4_quantize
        mlp = model.layers[0].mlp
        with torch.no_grad():
            for e in range(cfg.num_experts):
                s2 = expert_scale_2(mlp.w_gate.data[e], mlp.w_up.data[e])
                for attr, shared in (('w_gate', s2), ('w_up', s2), ('w_down', None)):
                    w = getattr(mlp, attr).data[e].float()
                    pk, sc, t2 = nvfp4_quantize(w, scale_2=shared)
                    getattr(mlp, attr).data[e].copy_(nvfp4_dequantize(pk, sc, t2).to(torch.bfloat16))
    if args.experts == 'served':
        # with --head (10-13): a head trained with frozen experts, served with the production NVFP4 experts unchanged
        from export_mtp_drafter import nvfp4_dequantize
        mlp = model.layers[0].mlp
        with safe_open(os.path.join(args.draft, 'model-mtp-draft.safetensors'), 'pt') as f:
            for e in range(cfg.num_experts):
                for proj, attr in (('gate_proj', 'w_gate'), ('up_proj', 'w_up'), ('down_proj', 'w_down')):
                    b = 'mtp.layers.0.mlp.experts.%d.%s.' % (e, proj)
                    w = nvfp4_dequantize(f.get_tensor(b + 'weight'), f.get_tensor(b + 'weight_scale'), f.get_tensor(b + 'weight_scale_2'))
                    getattr(mlp, attr).data[e].copy_(w.to(torch.bfloat16))
    model = model.to(dev).eval()
    v = json.load(open(args.vocab))
    ids = sorted({i for a, b in v['token_ranges'] for i in range(int(a), int(b))} | set(v.get('extra_ids', [])))
    sub = torch.tensor(ids, dtype=torch.long, device=dev)
    return Q, model, cfg, sub, model.lm_head.weight.index_select(0, sub).contiguous()


S2 = '/workspace/arc3-inference/runs/stage0-reference-20260919-v1/stage12/harness_revision_v529/spec_decode_stage2'
_KV = {}


def kv_quant(args, k, v):
    """The drafter's stored K / V ([T, 2, 256] post-RoPE) as attention reads them (--kv lutc) or unchanged."""
    if args.kv == 'bf16':
        return k, v
    if not _KV:
        sys.path.insert(0, os.path.join(S2, 'stage2_v1'))
        import kvfq
        m = torch.load(os.path.join(S2, 'kvlut_means_10020543.pt'))['mtp.layers.48.self_attn.attn']
        _KV.update(fq=kvfq.fake_quant, mk=m['k'].float().to(k.device), mv=m['v'].float().to(k.device))
    return (_KV['fq'](k, 'h_lutpm+c', _KV['mk']), _KV['fq'](v, 'h_lutpm+c', _KV['mv']))


def index_qk(Q, model, cfg, block_in, pos):
    """The indexer's query (normed, RoPE) [T, 4, 128] and raw keys [T, 128] from the attention input."""
    ix = model.layers[0].self_attn.indexer
    qk = ix.index_qk_proj(block_in)
    q, k = qk.split((cfg.indexer_heads * cfg.indexer_head_dim, cfg.indexer_kv_heads * cfg.indexer_head_dim), dim=-1)
    q = ix.q_layernorm(q.reshape(-1, cfg.indexer_heads, cfg.indexer_head_dim))
    return Q.rope_neox(q, pos, cfg.rotary_dim, cfg.rope_theta), k


class Context:
    """Step-1 KV entries of every exported row and the QSA group keys."""

    def __init__(self, Q, model, cfg, k1, v1, ik, pos, args):
        self.k1, self.v1, self.pos, self.args = k1, v1, pos, args
        N = k1.shape[0]
        self.N = N
        p0 = int(pos[0])
        # groups by position: group g = positions [4g, 4g + 4); rows are consecutive positions (checked by the caller)
        g0 = -(-p0 // GROUP)                                   # first complete group inside the rows
        g1 = (p0 + N) // GROUP                                 # groups up to the last row
        self.g0, self.G = g0, max(0, g1 - g0)
        if self.G and args.attention == 'qsa':
            r0 = g0 * GROUP - p0
            raw = ik[r0:r0 + self.G * GROUP].float().view(self.G, GROUP, -1).mean(1).to(ik.dtype)   # pooled raw keys
            ix = model.layers[0].self_attn.indexer
            kg = ix.k_layernorm(raw).unsqueeze(1)                                                  # [G, 1, 128]
            first = pos[r0:r0 + self.G * GROUP:GROUP]
            self.kg = Q.rope_neox(kg, first, cfg.rotary_dim, cfg.rope_theta)[:, 0].float()         # [G, 128]
        self.p0 = p0

    def token_mask(self, iq, qpos, qrow):
        """Selected step-1 rows per query: iq [n, 4, 128] (query's indexer q), qpos [n] its position, qrow [n] the last
        visible row (step 1: the query row; later steps: the anchor row). -> bool [n, N]."""
        n, N = iq.shape[0], self.N
        rows = torch.arange(N, device=iq.device)
        causal = rows[None, :] <= qrow[:, None]
        if self.args.attention == 'dense' or self.G == 0:
            return causal
        vis = torch.clamp((torch.minimum(qpos, self.pos[qrow]) + 1) // GROUP - self.g0, min=0)   # complete groups <= query
        vis = torch.clamp(vis, max=self.G)
        s = torch.einsum('nhd,gd->nhg', iq.float(), self.kg).clamp_min(0).sum(1) / math.sqrt(iq.shape[-1])   # [n, G]
        s = s.masked_fill(torch.arange(self.G, device=iq.device)[None, :] >= vis[:, None], float('-inf'))
        W = getattr(self.args, 'draftwin', 0)
        if W:                                                  # draftwin: groups ending before the last W positions
            lo = torch.div(torch.minimum(qpos, self.pos[qrow]) - W + 1, GROUP, rounding_mode='floor') - self.g0
            s = s.masked_fill(torch.arange(self.G, device=iq.device)[None, :] < lo[:, None], float('-inf'))
        k = min(BLOCK_TOPK, self.G)
        top = s.topk(k, dim=1)
        gsel = torch.where(torch.isfinite(top.values), top.indices, -1)                          # [n, k]
        gm = torch.zeros(n, self.G + 1, dtype=torch.bool, device=iq.device)
        gm.scatter_(1, torch.where(gsel >= 0, gsel, self.G), True)
        gm = gm[:, :self.G]                                                                      # [n, G] selected groups
        r0 = self.g0 * GROUP - self.p0
        tok = torch.zeros(n, N, dtype=torch.bool, device=iq.device)
        tok[:, r0:r0 + self.G * GROUP] = gm.repeat_interleave(GROUP, dim=1)
        # the incomplete group of the last visible row (always kept)
        lastpos = self.pos[qrow]
        tail_start = ((lastpos + 1) // GROUP) * GROUP
        tok |= (self.pos[None, :] >= tail_start[:, None]) & (rows[None, :] <= qrow[:, None])
        return tok & causal


def attend(q, ctx, mask, chain_k, chain_v, scale):
    """q [n, nh, D]; step-1 keys ctx.k1 / v1 [N, nkv, D] under mask [n, N]; plus each query's own chain keys
    chain_k / v [n, c, nkv, D] (c >= 0). SDPA over the concatenation with a per-query mask."""
    n = q.shape[0]
    c = 0 if chain_k is None else chain_k.shape[1]
    K = ctx.k1 if c == 0 else torch.cat([ctx.k1, chain_k.reshape(n * c, *chain_k.shape[2:])])
    V = ctx.v1 if c == 0 else torch.cat([ctx.v1, chain_v.reshape(n * c, *chain_v.shape[2:])])
    if c:
        own = torch.zeros(n, n * c, dtype=torch.bool, device=q.device)
        own[torch.arange(n, device=q.device).repeat_interleave(c), torch.arange(n * c, device=q.device)] = True
        mask = torch.cat([mask, own], dim=1)
    out = F.scaled_dot_product_attention(q.transpose(0, 1).unsqueeze(0), K.transpose(0, 1).unsqueeze(0),
                                         V.transpose(0, 1).unsqueeze(0), attn_mask=mask[None, None], scale=scale,
                                         enable_gqa=True)
    return out[0].transpose(0, 1)


def run_session(Q, model, cfg, sub, wsub, feat, seg, rec, args, dev):
    K = args.K
    with safe_open(feat, 'pt') as f:
        hs = f.get_tensor('hidden_states')
        tok = f.get_tensor('token_ids').to(dev)
        pos = f.get_tensor(args.pos_key).to(dev)
    N = hs.shape[0]
    s, e = int(seg['start']), int(seg['end'])
    s0 = e - N
    w0 = s - s0
    assert s0 >= 0 and w0 >= 0 and e - s == len(rec['input_ids']), (feat, N, s, e)
    assert torch.equal(tok[w0:].cpu(), torch.tensor(rec['input_ids'])), 'window tokens differ: ' + feat
    assert torch.equal(pos[1:] - pos[:-1], torch.ones(N - 1, dtype=pos.dtype, device=dev)), 'positions not consecutive'
    mask = torch.zeros(N, dtype=torch.bool, device=dev)
    mask[w0:] = torch.tensor(rec['loss_mask'], dtype=torch.bool, device=dev)
    anchors = torch.nonzero(mask, as_tuple=False).squeeze(1)
    anchors = anchors[(anchors >= w0 + 1) & (anchors + K <= N - 1)]
    att = model.layers[0].self_attn
    scale = cfg.head_dim ** -0.5
    H4 = hs[:, SLOT_STREAMS].reshape(N, -1)
    fin = hs[:, SLOT_FINAL]
    with torch.no_grad():
        # step-1 KV of every row j (pairs (h_j, x_{j+1})), chunked
        k1 = torch.empty(N - 1, cfg.num_kv_heads, cfg.head_dim, dtype=torch.bfloat16, device=dev)
        v1 = torch.empty_like(k1)
        ik = torch.empty(N - 1, cfg.indexer_head_dim, dtype=torch.bfloat16, device=dev)
        for a in range(0, N - 1, args.chunk):
            b = min(N - 1, a + args.chunk)
            x0 = model.fuse(H4[a:b].to(dev), tok[a + 1:b + 1])
            block_in, _ = model.layers[0].attn_hyper_connection.mix(x0)
            _, kk, vv, _ = att.project(block_in, pos[a:b])
            k1[a:b], v1[a:b] = kv_quant(args, kk, vv)
            ik[a:b] = index_qk(Q, model, cfg, block_in, pos[a:b])[1]
        ctx = Context(Q, model, cfg, k1, v1, ik, pos[:N - 1], args)
        # anchors in chunks: step 1 on row j = a - 1, then steps 2..K
        acc = {m: torch.zeros(len(anchors), K, device=dev) for m in mode_names(args.trunc)}
        slot_mask = torch.stack([mask[anchors + k] for k in range(1, K + 1)], dim=1)               # [A, K]
        for c0 in range(0, len(anchors), args.qchunk):
            an = anchors[c0:c0 + args.qchunk]
            j = an - 1
            n = len(j)
            h = H4[j.cpu()].to(dev)
            ck = cv = None
            m1 = None
            for k in range(1, K + 1):
                x0 = SE.fuse_k(model, h, tok[j + k], k)       # = model.fuse(...) unless the head has step embeddings
                qp = pos[j] + (k - 1)
                block_in, inj = model.layers[0].attn_hyper_connection.mix(x0)
                q, kk, vv, gate = att.project(block_in, qp)
                kk, vv = kv_quant(args, kk, vv)
                iq = index_qk(Q, model, cfg, block_in, qp)[0]
                if args.index_share and k >= 2:           # the step-1 selection, no chain (see the module docstring)
                    m = m1
                else:
                    m = ctx.token_mask(iq, qp, j)
                    m1 = m
                if k >= 2 and not args.index_share:
                    ck = kk.unsqueeze(1) if ck is None else torch.cat([ck, kk.unsqueeze(1)], dim=1)
                    cv = vv.unsqueeze(1) if cv is None else torch.cat([cv, vv.unsqueeze(1)], dim=1)
                out = attend(q, ctx, m, ck, cv, scale)
                sample, h = model.layer_post(x0, out, gate, inj)
                y = (sample @ wsub.t()).float()                                                    # [n, S] subset logits
                pl = (fin[(j + k).cpu()].to(dev) @ model.lm_head.weight.t()).float()              # target row a + k - 1
                score(acc, c0, k - 1, y, pl, sub, args.trunc)
                del pl, y
        out = {}
        valid = slot_mask.any(1).float()                    # aggregate(): blocks with no valid slot are not counted
        for name, a in acc.items():
            ab = a * slot_mask
            pref = ab.cumprod(1)
            out[name] = dict(slot_sum=ab.sum(0).tolist(), slot_n=slot_mask.sum(0).tolist(),
                             L_sum=[float(((1 + pref[:, :k].sum(1)) * valid).sum()) for k in range(1, K + 1)],
                             n_anchors=int(valid.sum()))
    return out, (acc, slot_mask, anchors)


def mode_names(trunc=()):
    names = ['greedy_T0']
    for t in TEMPS:
        tag = '%02d' % round(t * 10)
        names += ['prob_srv_T' + tag, 'greedy_srv_T' + tag, 'ceil_srv_T' + tag, 'prob_full_T' + tag]
        names += ['prob_srv_T%s_k%dp%02d' % (tag, dk, round(dp * 100)) for dk, dp in trunc]
    return names


def score(acc, c0, k, y, pl, sub, trunc=()):
    n = y.shape[0]
    d_sub = y.argmax(-1)
    d = sub[d_sub]                                                                              # greedy draft (vocab id)
    acc['greedy_T0'][c0:c0 + n, k] = (d == pl.argmax(-1)).float()
    for t in TEMPS:
        tag = '%02d' % round(t * 10)
        pp, pi = served_target(pl / t)
        qp, qs = served_draft(y, t)
        qi = sub[qs]
        q_on_p = ((pi[:, :, None] == qi[:, None, :]) * qp[:, None, :]).sum(-1)
        acc['prob_srv_T' + tag][c0:c0 + n, k] = torch.minimum(pp, q_on_p).sum(-1)
        for dk, dp in trunc:                                  # --trunc-sweep: the draft truncated differently
            qp2, qs2 = served_draft(y, t, dk, dp)
            q2 = ((pi[:, :, None] == sub[qs2][:, None, :]) * qp2[:, None, :]).sum(-1)
            acc['prob_srv_T%s_k%dp%02d' % (tag, dk, round(dp * 100))][c0:c0 + n, k] = torch.minimum(pp, q2).sum(-1)
        acc['greedy_srv_T' + tag][c0:c0 + n, k] = (pp * (pi == d[:, None])).sum(-1)
        acc['ceil_srv_T' + tag][c0:c0 + n, k] = pp.max(-1).values
        pt = torch.softmax(pl / t, dim=-1).index_select(1, sub)
        qt = torch.softmax(y / t, dim=-1)
        acc['prob_full_T' + tag][c0:c0 + n, k] = torch.minimum(pt, qt).sum(-1)
        del pt, qt


def summarize(tot, K):
    out = {}
    for name, d in tot.items():
        slots = [d['slot_sum'][k] / max(1.0, d['slot_n'][k]) for k in range(K)]
        L = [d['L_sum'][k] / max(1, d['n_anchors']) for k in range(K)]
        r = [L[0] - 1.0] + [(L[k] - L[k - 1]) / max(1e-9, (L[k - 1] - (L[k - 2] if k >= 2 else 1.0))) for k in range(1, K)]
        out[name] = dict(slots=[round(x, 4) for x in slots], L=[round(x, 4) for x in L], cond=[round(x, 4) for x in r],
                         anchors=d['n_anchors'])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', required=True)
    ap.add_argument('--pack', default='/workspace/arc3-inference/dflash_packed_20261011/cache_val50')
    ap.add_argument('--target', default='/workspace/models/Qwen3.8-Flash-Next-W4A16-intel')
    ap.add_argument('--draft', default='/workspace/models/Qwen3.8-Flash-Next-mtp-draft-nvfp4a16-noisc')
    ap.add_argument('--vocab', default='/workspace/arc3-inference/runs/stage0-reference-20260919-v1/stage12/harness_revision_v529/spec_decode_stage2/draft_vocab_32k.json')
    ap.add_argument('--kit', default='/workspace/arc3-inference/mtp_ft_20261011')
    ap.add_argument('--experts', choices=('served', 'bf16', 'requant'), default='served')
    ap.add_argument('--head', default='', help='fine-tuned head (mtp.* safetensors) loaded over the BF16 base')
    ap.add_argument('--attention', choices=('qsa', 'dense'), default='qsa')
    ap.add_argument('--draftwin', type=int, default=0, help='serving window W in positions for the QSA selection (0 = off)')
    ap.add_argument('--index-share', action='store_true', help="emulate vLLM's index_share_for_mtp_iteration (steps >= 2 reuse "
                    "step 1's selection; the in-draft chain is not attended)")
    ap.add_argument('--trunc-sweep', default='', help='extra draft truncations K:P,... scored from the same logits (modes '
                    'prob_srv_T<t>_k<K>p<P*100>)')
    ap.add_argument('--kv', choices=('lutc', 'bf16'), default='lutc')
    ap.add_argument('--pos-key', default='position_ids')
    ap.add_argument('--K', type=int, default=5)
    ap.add_argument('--sessions', type=int, default=0)
    ap.add_argument('--chunk', type=int, default=4096)
    ap.add_argument('--qchunk', type=int, default=256)
    ap.add_argument('--device', default='cuda:1')
    ap.add_argument('--out', required=True)
    ap.add_argument('--save-anchors', default='')
    args = ap.parse_args()
    args.trunc = [(int(x.split(':')[0]), float(x.split(':')[1])) for x in args.trunc_sweep.split(',') if x]
    dev = torch.device(args.device)
    t0 = time.time()
    Q, model, cfg, sub, wsub = load_model(args, dev)
    print('model loaded in %.0f s (experts %s, attention %s, kv %s, subset %d)' % (time.time() - t0, args.experts, args.attention, args.kv, len(sub)), flush=True)
    segs = [json.loads(l) for l in open(os.path.join(args.pack, 'segments.jsonl'))]
    recs = [json.loads(l) for l in open(os.path.join(args.pack, 'rows.jsonl'))]
    files = {f.split('-', 2)[2].rsplit('-', 1)[0]: os.path.join(args.features, f) for f in os.listdir(args.features)
             if f.startswith('arc3-persistent-') and f.endswith('.safetensors')}
    order = sorted(range(len(segs)), key=lambda i: segs[i]['end'] - max(0, segs[i]['start'] - 16384))
    if args.sessions:
        order = order[:args.sessions]
    tot, per, saved = {}, {}, {}
    for i in order:
        seg, rec = segs[i], recs[i]
        sid = seg['session_id']
        if sid not in files:
            print('missing features for', sid, flush=True)
            continue
        t1 = time.time()
        res, (acc, sm, an) = run_session(Q, model, cfg, sub, wsub, files[sid], seg, rec, args, dev)
        per[sid] = res
        for name, d in res.items():
            T = tot.setdefault(name, dict(slot_sum=[0.0] * args.K, slot_n=[0.0] * args.K, L_sum=[0.0] * args.K, n_anchors=0))
            for k in range(args.K):
                T['slot_sum'][k] += d['slot_sum'][k]; T['slot_n'][k] += d['slot_n'][k]; T['L_sum'][k] += d['L_sum'][k]
            T['n_anchors'] += d['n_anchors']
        if args.save_anchors:
            saved[sid] = dict(anchors=an.cpu(), slot_mask=sm.cpu(), **{k: v.cpu() for k, v in acc.items()})
        s1 = summarize({k: v for k, v in res.items() if k in ('prob_srv_T10', 'ceil_srv_T10')}, args.K)
        print('%s %s: %d anchors, %.0f s; prob_srv_T10 L %s; ceil L %s' % (sid, seg['run'], res['greedy_T0']['n_anchors'], time.time() - t1,
              s1['prob_srv_T10']['L'], s1['ceil_srv_T10']['L']), flush=True)
    summ = summarize(tot, args.K)
    json.dump(dict(args=vars(args), summary=summ, per_session=per, seconds=time.time() - t0), open(args.out, 'w'), indent=1)
    if args.save_anchors:
        torch.save(saved, args.save_anchors)
    for name in ('prob_srv_T10', 'prob_srv_T06', 'greedy_srv_T10', 'greedy_srv_T06', 'ceil_srv_T10', 'ceil_srv_T06', 'greedy_T0'):
        print(name, summ[name], flush=True)
    print('done in %.0f s ->' % (time.time() - t0), args.out, flush=True)


if __name__ == '__main__':
    main()
