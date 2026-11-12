"""qwen38_mtp.py (2026-10-11; backup plan: fine-tune the Qwen3.8-Flash-Next MTP drafter on our corpus if DFlash 2
underperforms). A trainable PyTorch port of the serving MTP head (vLLM `models/qwen3_8_flash_next/nvidia/mtp.py`,
`model.py` Qwen3_8FlashNextDecoderLayer, `common/hyperconnection.py` GatedResidual, `qwen3_next.py` attention / MoE),
with the checkpoint's own parameter names (prefix `mtp.` stripped), so the BF16 weights in the target's
`model_extra_tensors.safetensors` load and save one-to-one.

Serving semantics reproduced (see README.md for the source references):
  - inputs per row t: the target's pre-final-mixer multi-stream state h_t [hc*H] (the `_mtp_hidden_buffer` the target
    fills for the drafter) and the next token x_{t+1}; RoPE position = the target position of row t (+ k - 1 at step k);
  - fusion: e = fc_embedding(GemmaRMSNorm_H(embed(x))); h = fc_hidden(GemmaRMSNorm_{hc*H}(h) viewed [hc, H]) (shared
    across streams); x0 = h + e (broadcast to every stream);
  - one HC decoder layer: attn_hc.mix -> gated full attention (q and a sigmoid output gate per head, Gemma q/k norms,
    neox RoPE on the first 64 of 256 dims, GQA 24 / 2) -> attn_hc.combine -> mlp_hc.mix -> MoE (512 experts, top-10,
    softmax + renormalize, sigmoid-gated shared expert) -> mlp_hc.combine = the multi-stream output (next step's input);
  - final mixer (mix only) -> the single stream that lm_head (shared with the target, no norm) consumes;
  - HC ops: grouped Gemma RMSNorm per stream; lora = silu(down(xn) / hc); block_input = mean_s(sigmoid(up(lora)) * xn);
    combine = h + out * 2 sigmoid(inject(xn) / hc) (ops/hc.py kernels).
Attention: dense causal. Serving uses QSA (top-2,048 tokens by the indexer, in 4-token groups); with <= 2,048 context
tokens every token is selected, so the two agree there (the GPU parity test uses such contexts); longer windows train
with dense attention (an approximation; the indexer weights load and stay frozen).
Multi-step (K draft steps, one shared layer as in serving): step k's query at row t attends to the step-1 keys of rows
<= t (the context's KV cache) plus the step-2..k keys of row t itself (the draft chain written at positions
pos_t + 1 .. pos_t + k - 1), merged exactly through the log-sum-exp; `incremental_reference` checks this against a
cache-by-cache simulation of serving.
Loss: per step, cross-entropy on the corpus tokens (sampled from the target at T = 1, so its expectation is the forward
KL up to a constant), optionally the forward KL to the target's own distribution (teacher logits = lm_head(target final
mixer(h))), both computed in row chunks against the 248K vocabulary."""
import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class MTPConfig:
    hidden_size: int = 2560
    hc_count: int = 4
    hc_lowrank: int = 320
    num_heads: int = 24
    num_kv_heads: int = 2
    head_dim: int = 256
    rotary_dim: int = 64
    rope_theta: float = 1e7
    num_experts: int = 512
    top_k: int = 10
    moe_intermediate: int = 640
    shared_intermediate: int = 640
    vocab_size: int = 248320
    rms_eps: float = 1e-6
    norm_topk_prob: bool = True
    indexer_heads: int = 4
    indexer_kv_heads: int = 1
    indexer_head_dim: int = 128

    @classmethod
    def from_text_config(cls, tc: dict) -> 'MTPConfig':
        rp = tc.get('rope_parameters') or {}
        return cls(hidden_size=tc['hidden_size'], hc_count=tc['hc_count'], hc_lowrank=tc['hc_lowrank'],
                   num_heads=tc['num_attention_heads'], num_kv_heads=tc['num_key_value_heads'], head_dim=tc['head_dim'],
                   rotary_dim=int(tc['head_dim'] * rp.get('partial_rotary_factor', tc.get('partial_rotary_factor', 0.25))),
                   rope_theta=float((tc.get('mtp') or {}).get('rope_theta') or rp.get('rope_theta', 1e7)),
                   num_experts=tc['num_experts'], top_k=tc['num_experts_per_tok'], moe_intermediate=tc['moe_intermediate_size'],
                   shared_intermediate=tc['shared_expert_intermediate_size'], vocab_size=tc['vocab_size'],
                   rms_eps=tc['rms_norm_eps'], norm_topk_prob=bool(tc.get('norm_topk_prob', True)),
                   indexer_heads=tc.get('indexer_n_heads') or 4, indexer_kv_heads=tc.get('indexer_kv_heads') or 1,
                   indexer_head_dim=tc.get('indexer_head_dim') or 128)


def gemma_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float, group: int | None = None) -> torch.Tensor:
    """Gemma RMSNorm in fp32 with the (1 + w) affine; `group` normalizes each group of that many trailing elements."""
    xf = x.float()
    if group is not None and group != xf.shape[-1]:
        g = xf.unflatten(-1, (xf.shape[-1] // group, group))
        xf = (g * torch.rsqrt(g.square().mean(-1, keepdim=True) + eps)).flatten(-2)
    else:
        xf = xf * torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    return (xf * (1.0 + weight.float())).to(x.dtype)


class GemmaRMSNorm(nn.Module):
    def __init__(self, size: int, eps: float, group: int | None = None):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(size))
        self.eps, self.group = eps, group

    def forward(self, x):
        return gemma_rmsnorm(x, self.weight, self.eps, self.group)


class GatedResidual(nn.Module):
    """HyperConnection gated residual (checkpoint names: hc_norm, input_mix_weight_down / _up, block_inject_weight)."""

    def __init__(self, cfg: MTPConfig, use_combine: bool = True):
        super().__init__()
        hh = cfg.hc_count * cfg.hidden_size
        self.hc, self.h, self.use_combine = cfg.hc_count, cfg.hidden_size, use_combine
        self.hc_norm = GemmaRMSNorm(hh, cfg.rms_eps, group=cfg.hidden_size)
        self.input_mix_weight_down = nn.Linear(hh, cfg.hc_lowrank, bias=False)
        self.input_mix_weight_up = nn.Linear(cfg.hc_lowrank, hh, bias=False)
        if use_combine:
            self.block_inject_weight = nn.Linear(hh, cfg.hc_count, bias=False)

    def mix(self, x: torch.Tensor):
        """x [T, hc*H] -> (block_input [T, H], injection logits [T, hc] or None)."""
        xn = self.hc_norm(x)
        lora = F.silu(self.input_mix_weight_down(xn).float() / self.hc).to(x.dtype)
        gate = torch.sigmoid(self.input_mix_weight_up(lora).float()).unflatten(-1, (self.hc, self.h))
        block_input = (gate * xn.float().unflatten(-1, (self.hc, self.h))).mean(-2).to(x.dtype)
        inj = self.block_inject_weight(xn) if self.use_combine else None
        return block_input, inj

    def combine(self, x: torch.Tensor, block_output: torch.Tensor, inj: torch.Tensor) -> torch.Tensor:
        w = 2.0 * torch.sigmoid(inj.float() / self.hc)                                   # [T, hc]
        out = x.float().unflatten(-1, (self.hc, self.h)) + block_output.float().unsqueeze(-2) * w.unsqueeze(-1)
        return out.flatten(-2).to(x.dtype)


def rope_neox(x: torch.Tensor, positions: torch.Tensor, rotary_dim: int, theta: float) -> torch.Tensor:
    """Neox-style RoPE on the first rotary_dim dims of the last axis; x [T, heads, D], positions [T] (text MRoPE with
    identical t/h/w positions reduces to this)."""
    half = rotary_dim // 2
    inv = 1.0 / (theta ** (torch.arange(0, rotary_dim, 2, device=x.device, dtype=torch.float64) / rotary_dim))
    ang = positions.to(torch.float64).unsqueeze(-1) * inv                               # [T, half]
    cos, sin = ang.cos().float().unsqueeze(-2), ang.sin().float().unsqueeze(-2)          # [T, 1, half]
    xf = x.float()
    x1, x2, rest = xf[..., :half], xf[..., half:rotary_dim], xf[..., rotary_dim:]
    return torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin, rest), dim=-1).to(x.dtype)


class Indexer(nn.Module):
    """QSA indexer weights (loaded, frozen; dense attention does not use them)."""

    def __init__(self, cfg: MTPConfig):
        super().__init__()
        self.index_qk_proj = nn.Linear(cfg.hidden_size, (cfg.indexer_heads + cfg.indexer_kv_heads) * cfg.indexer_head_dim, bias=False)
        self.q_layernorm = GemmaRMSNorm(cfg.indexer_head_dim, cfg.rms_eps)
        self.k_layernorm = GemmaRMSNorm(cfg.indexer_head_dim, cfg.rms_eps)


class Attention(nn.Module):
    def __init__(self, cfg: MTPConfig):
        super().__init__()
        self.cfg = cfg
        nh, nkv, hd = cfg.num_heads, cfg.num_kv_heads, cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, nh * hd * 2, bias=False)        # per head: [q (hd), gate (hd)]
        self.k_proj = nn.Linear(cfg.hidden_size, nkv * hd, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, nkv * hd, bias=False)
        self.o_proj = nn.Linear(nh * hd, cfg.hidden_size, bias=False)
        self.q_norm = GemmaRMSNorm(hd, cfg.rms_eps)
        self.k_norm = GemmaRMSNorm(hd, cfg.rms_eps)
        self.indexer = Indexer(cfg)

    def project(self, x: torch.Tensor, positions: torch.Tensor):
        """x [T, H] -> q [T, nh, hd], k, v [T, nkv, hd] (post norm and RoPE), gate [T, nh*hd] (pre-sigmoid)."""
        c = self.cfg
        T = x.shape[0]
        q, gate = self.q_proj(x).view(T, c.num_heads, 2 * c.head_dim).chunk(2, dim=-1)
        k = self.k_proj(x).view(T, c.num_kv_heads, c.head_dim)
        v = self.v_proj(x).view(T, c.num_kv_heads, c.head_dim)
        q = rope_neox(self.q_norm(q), positions, c.rotary_dim, c.rope_theta)
        k = rope_neox(self.k_norm(k), positions, c.rotary_dim, c.rope_theta)
        return q, k, v, gate.reshape(T, c.num_heads * c.head_dim)

    def output(self, attn: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        T = attn.shape[0]
        return self.o_proj((attn.reshape(T, -1).float() * torch.sigmoid(gate.float())).to(attn.dtype))


def causal_attention_lse(q, k, v, scale, chunk: int = 1024):
    """Dense causal GQA attention with its log-sum-exp; q [T, nh, D], k / v [T, nkv, D] -> out [T, nh, D], lse [T, nh]
    (fp32 math, query-chunked; the reference path, also used on CPU)."""
    T, nh, D = q.shape
    g = nh // k.shape[1]
    kf = k.float().repeat_interleave(g, dim=1)                                           # [T, nh, D]
    vf = v.float().repeat_interleave(g, dim=1)
    outs, lses = [], []
    for a in range(0, T, chunk):
        b = min(T, a + chunk)
        s = torch.einsum('qhd,khd->hqk', q[a:b].float(), kf[:b]) * scale                # [nh, b-a, b]
        mask = torch.arange(b, device=q.device)[None, :] > torch.arange(a, b, device=q.device)[:, None]
        s = s.masked_fill(mask[None], float('-inf'))
        lse = torch.logsumexp(s, dim=-1)                                                 # [nh, b-a]
        outs.append(torch.einsum('hqk,khd->qhd', torch.exp(s - lse.unsqueeze(-1)), vf[:b]))
        lses.append(lse.transpose(0, 1))
    return torch.cat(outs).to(q.dtype), torch.cat(lses)


def flex_causal_attention_lse(q, k, v, scale):
    """GPU path: torch flex_attention with the causal block mask and its log-sum-exp (same contract)."""
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention
    T = q.shape[0]
    bm = create_block_mask(lambda b, h, qi, ki: qi >= ki, 1, None, T, T, device=q.device)
    qq, kk, vv = (t.transpose(0, 1).unsqueeze(0) for t in (q, k, v))                     # [1, heads, T, D]
    try:
        from torch.nn.attention.flex_attention import AuxRequest
        out, aux = flex_attention(qq, kk, vv, block_mask=bm, scale=scale, enable_gqa=True, return_aux=AuxRequest(lse=True))
        lse = aux.lse
    except ImportError:
        out, lse = flex_attention(qq, kk, vv, block_mask=bm, scale=scale, enable_gqa=True, return_lse=True)
    return out[0].transpose(0, 1), lse[0].transpose(0, 1).float()


def merge_chain(q, out1, lse1, chain, scale):
    """Add the per-row chain keys (step 2..k of the same row) to a causal result exactly: chain = [(k_j, v_j)] with
    k_j / v_j [T, nkv, D]; returns the attention output over (causal keys) U (chain keys of the row)."""
    if not chain:
        return out1
    nh = q.shape[1]
    g = nh // chain[0][0].shape[1]
    ls = [lse1]
    vs = []
    for kj, vj in chain:
        ls.append((q.float() * kj.float().repeat_interleave(g, dim=1)).sum(-1) * scale)   # [T, nh]
        vs.append(vj.float().repeat_interleave(g, dim=1))
    L = torch.stack(ls)                                                                 # [1 + n, T, nh]
    m = L.max(0).values
    w = torch.exp(L - m)                                                                # [1 + n, T, nh]
    num = w[0].unsqueeze(-1) * out1.float() + sum(w[i + 1].unsqueeze(-1) * vs[i] for i in range(len(vs)))
    return (num / w.sum(0).unsqueeze(-1)).to(q.dtype)


class MoE(nn.Module):
    """Qwen3-Next sparse MoE: softmax router (fp32) -> top-k -> renormalize; SiLU-gated experts; sigmoid-gated shared
    expert. Experts are stacked [E, I, H] / [E, H, I]; checkpoint names experts.<e>.{gate,up,down}_proj.weight map in
    `load_state_dict_from_checkpoint`."""

    def __init__(self, cfg: MTPConfig):
        super().__init__()
        self.cfg = cfg
        E, I, H = cfg.num_experts, cfg.moe_intermediate, cfg.hidden_size
        self.gate = nn.Linear(H, E, bias=False)
        self.w_gate = nn.Parameter(torch.empty(E, I, H))
        self.w_up = nn.Parameter(torch.empty(E, I, H))
        self.w_down = nn.Parameter(torch.empty(E, H, I))
        for w in (self.w_gate, self.w_up, self.w_down):
            nn.init.normal_(w, std=0.02)
        self.shared_expert = nn.ModuleDict(dict(gate_proj=nn.Linear(H, cfg.shared_intermediate, bias=False),
                                                up_proj=nn.Linear(H, cfg.shared_intermediate, bias=False),
                                                down_proj=nn.Linear(cfg.shared_intermediate, H, bias=False)))
        self.shared_expert_gate = nn.Linear(H, 1, bias=False)

    def route(self, x):
        probs = torch.softmax(self.gate(x).float(), dim=-1)
        w, idx = probs.topk(self.cfg.top_k, dim=-1)
        if self.cfg.norm_topk_prob:
            w = w / w.sum(-1, keepdim=True)
        return w, idx

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T, H = x.shape
        w, idx = self.route(x)
        out = torch.zeros(T, H, dtype=torch.float32, device=x.device)
        flat = idx.reshape(-1)
        order = torch.argsort(flat, stable=True)
        counts = torch.bincount(flat, minlength=self.cfg.num_experts).tolist()
        tok = (order // self.cfg.top_k)
        wt = w.reshape(-1)[order]
        start = 0
        for e, n in enumerate(counts):
            if n == 0:
                continue
            t_e = tok[start:start + n]
            xe = x[t_e]
            h = F.silu(xe @ self.w_gate[e].t()) * (xe @ self.w_up[e].t())
            out.index_add_(0, t_e, (h @ self.w_down[e].t()).float() * wt[start:start + n].unsqueeze(-1))
            start += n
        se = self.shared_expert
        shared = se['down_proj'](F.silu(se['gate_proj'](x)) * se['up_proj'](x))
        out = out + shared.float() * torch.sigmoid(self.shared_expert_gate(x).float())
        return out.to(x.dtype)


class MTPLayer(nn.Module):
    def __init__(self, cfg: MTPConfig):
        super().__init__()
        self.attn_hyper_connection = GatedResidual(cfg)
        self.self_attn = Attention(cfg)
        self.mlp_hyper_connection = GatedResidual(cfg)
        self.mlp = MoE(cfg)


class Qwen38MTP(nn.Module):
    """The MTP head (trainable) plus the frozen shared embedding / lm_head and, for distillation, the target's final
    mixer (teacher logits from the same multi-stream state)."""

    def __init__(self, cfg: MTPConfig, with_teacher: bool = True):
        super().__init__()
        self.cfg = cfg
        H, hh = cfg.hidden_size, cfg.hc_count * cfg.hidden_size
        self.pre_fc_norm_embedding = GemmaRMSNorm(H, cfg.rms_eps)
        self.pre_fc_norm_hidden = GemmaRMSNorm(hh, cfg.rms_eps)                            # one RMS over all hc*H
        self.fc_embedding = nn.Linear(H, H, bias=False)
        self.fc_hidden = nn.Linear(H, H, bias=False)                                      # shared across streams
        self.layers = nn.ModuleList([MTPLayer(cfg)])
        self.hyper_connection_mixer = GatedResidual(cfg, use_combine=False)
        self.embed_tokens = nn.Embedding(cfg.vocab_size, H)
        self.lm_head = nn.Linear(H, cfg.vocab_size, bias=False)
        self.target_mixer = GatedResidual(cfg, use_combine=False) if with_teacher else None
        for m in (self.embed_tokens, self.lm_head, self.target_mixer):
            if m is not None:
                m.requires_grad_(False)
        self.layers[0].self_attn.indexer.requires_grad_(False)

    # ---- one draft step over all rows --------------------------------------------------------------------------
    def fuse(self, h_multi: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        c = self.cfg
        e = self.fc_embedding(self.pre_fc_norm_embedding(self.embed_tokens(tokens)))      # [T, H]
        hn = self.pre_fc_norm_hidden(h_multi).unflatten(-1, (c.hc_count, c.hidden_size))
        return (self.fc_hidden(hn) + e.unsqueeze(-2)).flatten(-2)                           # [T, hc*H]

    def layer_pre(self, x0: torch.Tensor, positions: torch.Tensor):
        layer = self.layers[0]
        block_in, inj = layer.attn_hyper_connection.mix(x0)
        q, k, v, gate = layer.self_attn.project(block_in, positions)
        return q, k, v, gate, inj

    def layer_post(self, x0, attn, gate, inj):
        layer = self.layers[0]
        x1 = layer.attn_hyper_connection.combine(x0, layer.self_attn.output(attn, gate), inj)
        block_in, inj2 = layer.mlp_hyper_connection.mix(x1)
        x2 = layer.mlp_hyper_connection.combine(x1, layer.mlp(block_in), inj2)              # multi-stream output
        sample, _ = self.hyper_connection_mixer.mix(x2)                                    # single stream for lm_head
        return sample, x2

    def forward_steps(self, h0: torch.Tensor, tokens: torch.Tensor, positions: torch.Tensor, num_steps: int = 3,
                      attention: str = 'reference'):
        """Teacher-forced K-step draft over one window (no batch dim). Rows t = 0 .. L-1 with L = T - num_steps - 1:
        step k (1-based) takes h (step 1: the target's h_t; then the previous step's output for row t) and x_{t+k} at
        position pos_t + k - 1, and its logits predict x_{t+k+1}. Returns [sample_hidden_k [L, H] for k = 1..K]."""
        T = tokens.shape[0]
        L = T - num_steps - 1
        assert L > 0, (T, num_steps)
        scale = self.cfg.head_dim ** -0.5
        attend = flex_causal_attention_lse if attention == 'flex' else causal_attention_lse
        samples, chain = [], []
        h = h0[:L]
        k1 = v1 = None
        for k in range(1, num_steps + 1):
            x0 = self.fuse(h, tokens[k:L + k])
            q, kk, vv, gate, inj = self.layer_pre(x0, positions[:L] + (k - 1))
            if k == 1:
                k1, v1 = kk, vv
                out, _ = attend(q, kk, vv, scale)
            else:
                chain.append((kk, vv))
                o1, lse1 = attend(q, k1, v1, scale)
                out = merge_chain(q, o1, lse1, chain, scale)
            sample, h = self.layer_post(x0, out, gate, inj)
            samples.append(sample)
        return samples

    def incremental_reference(self, h0, tokens, positions, num_steps: int = 3):
        """Serving simulation for tests: row by row, attention over an explicit KV cache = step-1 entries of rows <= t
        plus this chain's steps 2..k. Must equal forward_steps (validates the batched attention)."""
        T = tokens.shape[0]
        L = T - num_steps - 1
        scale = self.cfg.head_dim ** -0.5
        x0 = self.fuse(h0[:L], tokens[1:L + 1])
        q, k1, v1, gate, inj = self.layer_pre(x0, positions[:L])
        out1, _ = causal_attention_lse(q, k1, v1, scale)
        s1, h1 = self.layer_post(x0, out1, gate, inj)
        res = [[s1[t] for t in range(L)]] + [[None] * L for _ in range(num_steps - 1)]
        g = self.cfg.num_heads // self.cfg.num_kv_heads
        for t in range(L):
            keys, vals = [k1[:t + 1]], [v1[:t + 1]]
            h = h1[t:t + 1]
            for k in range(2, num_steps + 1):
                x0k = self.fuse(h, tokens[t + k:t + k + 1])
                qk, kk, vk, gk, ik = self.layer_pre(x0k, positions[t:t + 1] + (k - 1))
                keys.append(kk); vals.append(vk)
                K = torch.cat(keys).float().repeat_interleave(g, dim=1)
                V = torch.cat(vals).float().repeat_interleave(g, dim=1)
                p = torch.softmax(torch.einsum('qhd,khd->hqk', qk.float(), K) * scale, dim=-1)
                att = torch.einsum('hqk,khd->qhd', p, V).to(qk.dtype)
                sk, h = self.layer_post(x0k, att, gk, ik)
                res[k - 1][t] = sk[0]
        return [torch.stack(r) for r in res]

    # ---- losses --------------------------------------------------------------------------------------------------
    def teacher_hidden(self, h_multi: torch.Tensor) -> torch.Tensor:
        sample, _ = self.target_mixer.mix(h_multi)
        return sample

    def step_losses(self, samples, h0, tokens, loss_mask, step_weights=None, kl_weight: float = 0.0, chunk: int = 2048):
        """CE of step k's logits on x_{t+k+1} (rows whose label position has loss_mask 1); with kl_weight > 0 also
        the forward KL from the target's distribution at row t + k (lm_head(target mixer(h_{t+k})))."""
        K = len(samples)
        L = samples[0].shape[0]
        step_weights = step_weights or [1.0] * K
        total = samples[0].new_zeros((), dtype=torch.float32)
        metrics = {}
        for k, s in enumerate(samples, start=1):
            labels = tokens[k + 1:L + k + 1]
            mask = loss_mask[k + 1:L + k + 1].float()
            ce_sum = kl_sum = 0.0
            hits = 0.0
            for a in range(0, L, chunk):
                b = min(L, a + chunk)
                logits = self.lm_head(s[a:b]).float()
                m = mask[a:b]
                ce = F.cross_entropy(logits, labels[a:b], reduction='none')
                ce_sum = ce_sum + (ce * m).sum()
                hits = hits + ((logits.argmax(-1) == labels[a:b]).float() * m).sum()
                if kl_weight:
                    with torch.no_grad():
                        tl = self.lm_head(self.teacher_hidden(h0[a + k:b + k])).float().log_softmax(-1)
                    kl = (tl.exp() * (tl - logits.log_softmax(-1))).sum(-1)
                    kl_sum = kl_sum + (kl * m).sum()
            n = mask.sum().clamp(min=1)
            loss_k = (ce_sum + kl_weight * kl_sum) / n
            total = total + step_weights[k - 1] * loss_k
            metrics['ce_step%d' % k] = float(ce_sum / n)
            metrics['top1_step%d' % k] = float(hits / n)
            if kl_weight:
                metrics['kl_step%d' % k] = float(kl_sum / n)
        return total, metrics


# ---- checkpoint mapping ----------------------------------------------------------------------------------------
EXPERT_KEYS = {'gate_proj': 'w_gate', 'up_proj': 'w_up', 'down_proj': 'w_down'}


def checkpoint_to_module_name(name: str):
    """'mtp.layers.0.mlp.experts.17.up_proj.weight' -> ('layers.0.mlp.w_up', 17); other 'mtp.X' -> ('X', None);
    target tensors: embed / lm_head / the target's final mixer."""
    if name == 'model.language_model.embed_tokens.weight':
        return 'embed_tokens.weight', None
    if name == 'lm_head.weight':
        return 'lm_head.weight', None
    if name.startswith('model.language_model.hyper_connection_mixer.'):
        return 'target_mixer.' + name.removeprefix('model.language_model.hyper_connection_mixer.'), None
    if not name.startswith('mtp.'):
        return None, None
    n = name.removeprefix('mtp.')
    if '.mlp.experts.' in n:
        pre, rest = n.split('.mlp.experts.')
        e, proj, _ = rest.split('.', 2)
        return '%s.mlp.%s' % (pre, EXPERT_KEYS[proj]), int(e)
    if n.endswith('mlp.shared_expert.gate_proj.weight') or n.endswith('mlp.shared_expert.up_proj.weight') or \
            n.endswith('mlp.shared_expert.down_proj.weight'):
        return n.replace('mlp.shared_expert.', "mlp.shared_expert."), None
    return n, None


def load_from_checkpoint(model: Qwen38MTP, mtp_file: str, target_dir: str | None = None, strict: bool = True):
    """Load the BF16 MTP tensors (model_extra_tensors.safetensors) and, from target_dir's index, the embedding,
    lm_head and the target's final mixer. Returns the set of module names filled."""
    import json
    import os
    from safetensors import safe_open
    sd = dict(model.named_parameters())
    filled = set()

    def put(name, tensor):
        mod, e = checkpoint_to_module_name(name)
        if mod is None or mod not in sd:
            return
        p = sd[mod]
        with torch.no_grad():
            if e is None:
                assert tuple(p.shape) == tuple(tensor.shape), (name, p.shape, tensor.shape)
                p.copy_(tensor.to(p.dtype))
                filled.add(mod)
            else:
                assert tuple(p.shape[1:]) == tuple(tensor.shape), (name, p.shape, tensor.shape)
                p[e].copy_(tensor.to(p.dtype))
                filled.add('%s[%d]' % (mod, e))

    with safe_open(mtp_file, 'pt') as f:
        for k in f.keys():
            put(k, f.get_tensor(k))
    if target_dir:
        wm = json.load(open(os.path.join(target_dir, 'model.safetensors.index.json')))['weight_map']
        want = [k for k in wm if checkpoint_to_module_name(k)[0] and not k.startswith('mtp.')]
        for fname in sorted({wm[k] for k in want}):
            with safe_open(os.path.join(target_dir, fname), 'pt') as f:
                for k in want:
                    if wm[k] == fname:
                        put(k, f.get_tensor(k))
    if strict:
        missing = [n for n in sd if n not in filled and not n.startswith(('layers.0.mlp.w_')) and
                   (target_dir or not n.startswith(('embed_tokens', 'lm_head', 'target_mixer')))]
        experts = {n for n in sd if n.startswith('layers.0.mlp.w_')}
        for n in experts:
            got = sum(1 for x in filled if x.startswith(n + '['))
            if got != model.cfg.num_experts:
                missing.append('%s (%d of %d experts)' % (n, got, model.cfg.num_experts))
        assert not missing, missing
    return filled


def save_mtp_checkpoint(model: Qwen38MTP, path: str, dtype=torch.bfloat16):
    """Write the trainable head back under the checkpoint's names (`mtp.*`, experts unstacked), BF16 safetensors."""
    from safetensors.torch import save_file
    out = {}
    for n, p in model.named_parameters():
        if n.startswith(('embed_tokens', 'lm_head', 'target_mixer')):
            continue
        t = p.detach().to(dtype).cpu()
        if n.startswith('layers.0.mlp.w_'):
            proj = {v: k for k, v in EXPERT_KEYS.items()}[n.rsplit('.', 1)[1]]
            for e in range(t.shape[0]):
                out['mtp.layers.0.mlp.experts.%d.%s.weight' % (e, proj)] = t[e].contiguous()
        else:
            out['mtp.' + n] = t.contiguous()
    save_file(out, path, metadata={'format': 'pt', 'source': 'mtp_ft qwen38_mtp.py'})
    return len(out)
