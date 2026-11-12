"""GDN rollback by replay (speculative decoding phase B, stage 1; design in ../ROLLBACK_DESIGN.md).

Two Triton kernels derived line for line from vLLM's `fused_sigmoid_gating_delta_rule_update_kernel`
(third_party/flash_linear_attention/ops/fused_sigmoid_gating.py, runtime copy in ../rollback_study/), so the
arithmetic, and therefore the rounding, matches the stock speculative kernel:

- `gdn_verify_nowb`: the verify pass. It loads each request's committed state (column 0 of its state indices),
  computes the outputs of all k+1 verify tokens, and stores no state (stock stores one state per position).
- `gdn_commit`: after rejection sampling. It reloads the committed state, replays the first `num_sampled` verify
  tokens from the step's saved inputs (k, v, a, b) and stores the result once. This equals the stock kernel's
  per-position state at index num_sampled - 1 (gate G1).

Only one state slot per request is needed, whatever the draft length.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _gdn_step(b_h, b_k, b_v, b_a, b_b, A_log_v, dt_bias_v, beta, threshold, USE_QK_L2NORM: tl.constexpr):
    """One delta-rule update, in the stock kernel's exact operation order. Returns (new state, update vector)."""
    x = b_a + dt_bias_v
    softplus_x = tl.where(beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x)
    b_g = -tl.exp(A_log_v) * softplus_x
    b_beta = tl.sigmoid(b_b)
    if USE_QK_L2NORM:
        b_k = b_k * (tl.rsqrt(tl.sum(b_k * b_k) + 1e-6))
    b_h *= tl.exp(b_g)
    b_v -= tl.sum(b_h * b_k[None, :], 1)
    b_v *= b_beta
    b_h += b_v[:, None] * b_k[None, :]
    return b_h


@triton.jit(do_not_specialize=["N", "T"])
def _gdn_verify_nowb_kernel(A_log, a, b, dt_bias, beta, threshold, q, k, v, o, h0, cu_seqlens, state_idx, scale,
                            N: tl.int64, T: tl.int64, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr,
                            V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, stride_state: tl.constexpr,
                            stride_idx: tl.constexpr, USE_QK_L2NORM: tl.constexpr):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    T = eos - bos
    if T == 0:
        return
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_a = a + bos * HV + i_hv
    p_b = b + bos * HV + i_hv
    p_o = o + (bos * HV + i_hv) * V + o_v
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    slot = tl.load(state_idx + i_n * stride_idx).to(tl.int64)
    if slot <= 0:        # NULL_BLOCK_ID, as the stock kernel
        return
    p_h0 = h0 + slot * stride_state + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h += tl.load(p_h0, mask=mask_h, other=0).to(tl.float32)
    A_log_v = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_v = tl.load(dt_bias + i_hv).to(tl.float32)
    for i_t in range(0, T):
        b_q = tl.load(p_q, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)
        b_a = tl.load(p_a).to(tl.float32)
        # the stock kernel normalises q and k together; the state update only needs k
        if USE_QK_L2NORM:
            b_q = b_q * (tl.rsqrt(tl.sum(b_q * b_q) + 1e-6))
        b_q = b_q * scale
        b_h = _gdn_step(b_h, b_k, b_v, b_a, b_b, A_log_v, dt_bias_v, beta, threshold, USE_QK_L2NORM)
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        p_b += HV
        p_a += HV


@triton.jit(do_not_specialize=["N"])
def _gdn_commit_kernel(A_log, a, b, dt_bias, beta, threshold, k, v, h0, cu_seqlens, state_idx, num_sampled,
                       N: tl.int64, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                       BK: tl.constexpr, BV: tl.constexpr, stride_state: tl.constexpr, stride_idx: tl.constexpr,
                       USE_QK_L2NORM: tl.constexpr):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    n = tl.minimum(tl.load(num_sampled + i_n).to(tl.int64), eos - bos)
    if n <= 0:
        return
    slot = tl.load(state_idx + i_n * stride_idx).to(tl.int64)
    if slot <= 0:
        return
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_a = a + bos * HV + i_hv
    p_b = b + bos * HV + i_hv
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]
    p_h = h0 + slot * stride_state + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    b_h += tl.load(p_h, mask=mask_h, other=0).to(tl.float32)
    A_log_v = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_v = tl.load(dt_bias + i_hv).to(tl.float32)
    for i_t in range(0, n):
        b_k = tl.load(p_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(p_v, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(p_b).to(tl.float32)
        b_a = tl.load(p_a).to(tl.float32)
        b_h = _gdn_step(b_h, b_k, b_v, b_a, b_b, A_log_v, dt_bias_v, beta, threshold, USE_QK_L2NORM)
        p_k += H * K
        p_v += HV * V
        p_b += HV
        p_a += HV
    tl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=mask_h)


@triton.jit(do_not_specialize=["N", "T"])
def _gdn_lazy_verify_kernel(A_log, a, b, dt_bias, beta, threshold, q, k, v, o, h0, pend, cu_seqlens, state_idx,
                            num_accepted, scale, N: tl.int64, T: tl.int64, H: tl.constexpr, HV: tl.constexpr,
                            K: tl.constexpr, V: tl.constexpr, stride_state: tl.constexpr, stride_idx: tl.constexpr,
                            stride_pend_slot: tl.constexpr, stride_pend_head: tl.constexpr, USE_QK_L2NORM: tl.constexpr):
    """Lazy replay fused into the verify pass: one program per (request, value head) owns that head's whole state
    tile and a private pending area (flag, then per token k[K], v[V], a, b), so no program reads data another
    program writes. Step: replay the previous window's num_accepted tokens (if the flag is set) and store the
    committed state once; run the new window without storing state; save its inputs as the new pending window."""
    i_nh = tl.program_id(0)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    T = eos - bos
    if T == 0:
        return
    slot = tl.load(state_idx + i_n * stride_idx).to(tl.int64)
    if slot <= 0:
        return
    o_k = tl.arange(0, K)
    o_v = tl.arange(0, V)
    p_h = h0 + slot * stride_state + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.load(p_h).to(tl.float32)
    A_log_v = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_v = tl.load(dt_bias + i_hv).to(tl.float32)
    p_pend = pend + slot * stride_pend_slot + i_hv * stride_pend_head
    flag = tl.load(p_pend).to(tl.float32)
    n_rep = tl.where(flag > 0.5, tl.load(num_accepted + i_n).to(tl.int64), 0)
    W: tl.constexpr = K + V + 2
    for i_t in range(0, n_rep):
        base = p_pend + 1 + i_t * W
        b_k = tl.load(base + o_k).to(tl.float32)
        b_v = tl.load(base + K + o_v).to(tl.float32)
        b_a = tl.load(base + K + V).to(tl.float32)
        b_b = tl.load(base + K + V + 1).to(tl.float32)
        b_h = _gdn_step(b_h, b_k, b_v, b_a, b_b, A_log_v, dt_bias_v, beta, threshold, USE_QK_L2NORM)
    if n_rep > 0:
        hb = b_h.to(p_h.dtype.element_ty)
        tl.store(p_h, hb)
        b_h = hb.to(tl.float32)          # continue from the stored (rounded) state, as stock reloads its slot
    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_a = a + bos * HV + i_hv
    p_b = b + bos * HV + i_hv
    p_o = o + (bos * HV + i_hv) * V + o_v
    for i_t in range(0, T):
        r_q = tl.load(p_q)
        r_k = tl.load(p_k)
        r_v = tl.load(p_v)
        r_a = tl.load(p_a)
        r_b = tl.load(p_b)
        b_q = r_q.to(tl.float32)
        if USE_QK_L2NORM:
            b_q = b_q * (tl.rsqrt(tl.sum(b_q * b_q) + 1e-6))
        b_q = b_q * scale
        b_h = _gdn_step(b_h, r_k.to(tl.float32), r_v.to(tl.float32), r_a.to(tl.float32), r_b.to(tl.float32),
                        A_log_v, dt_bias_v, beta, threshold, USE_QK_L2NORM)
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty))
        base = p_pend + 1 + i_t * W
        tl.store(base + o_k, r_k.to(pend.dtype.element_ty))
        tl.store(base + K + o_v, r_v.to(pend.dtype.element_ty))
        tl.store(base + K + V, r_a.to(pend.dtype.element_ty))
        tl.store(base + K + V + 1, r_b.to(pend.dtype.element_ty))
        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        p_b += HV
        p_a += HV
    tl.store(p_pend, (flag * 0.0 + 1.0).to(pend.dtype.element_ty))


@triton.jit(do_not_specialize=["N"])
def _gdn_commit_pending_kernel(A_log, dt_bias, beta, threshold, h0, pend, flags, state_idx, num_accepted,
                               N: tl.int64, H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                               BK: tl.constexpr, BV: tl.constexpr, W: tl.constexpr, PCOL: tl.constexpr,
                               stride_state: tl.constexpr, stride_idx: tl.constexpr, stride_pend: tl.constexpr,
                               USE_QK_L2NORM: tl.constexpr):
    """Two-launch lazy replay, launch 1: commit the previous window of each request from its slot's pending area
    (layout per slot: T_max tokens of [k (H*K) | v (HV*V) | a (HV) | b (HV)], then a flag). Stock tiling (BV=32)
    and the stock update, so the committed state equals stock's per-position state bit for bit."""
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    slot = tl.load(state_idx + i_n * stride_idx).to(tl.int64)
    if slot <= 0:
        return
    pslot = tl.load(state_idx + i_n * stride_idx + PCOL).to(tl.int64)
    p_pend = pend + pslot * stride_pend
    flag = tl.load(flags + slot).to(tl.float32)
    n = tl.where(flag > 0.5, tl.load(num_accepted + i_n).to(tl.int64), 0)
    if n <= 0:
        return
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]
    p_h = h0 + slot * stride_state + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    b_h += tl.load(p_h, mask=mask_h, other=0).to(tl.float32)
    A_log_v = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_v = tl.load(dt_bias + i_hv).to(tl.float32)
    for i_t in range(0, n):
        base = p_pend + i_t * W
        b_k = tl.load(base + i_h * K + o_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(base + H * K + i_hv * V + o_v, mask=mask_v, other=0).to(tl.float32)
        b_a = tl.load(base + H * K + HV * V + i_hv).to(tl.float32)
        b_b = tl.load(base + H * K + HV * V + HV + i_hv).to(tl.float32)
        b_h = _gdn_step(b_h, b_k, b_v, b_a, b_b, A_log_v, dt_bias_v, beta, threshold, USE_QK_L2NORM)
    tl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=mask_h)


@triton.jit
def _gdn_stash_kernel(k, v, a, b, pend, flags, cu_seqlens, state_idx, H: tl.constexpr, HV: tl.constexpr,
                      K: tl.constexpr, V: tl.constexpr, W: tl.constexpr, PCOL: tl.constexpr, stride_idx: tl.constexpr,
                      stride_pend: tl.constexpr, BLOCK: tl.constexpr):
    """Launch 3 (after the verify): copy each request's window inputs into its slot's pending area, set the flag.
    One program per request; the previous window was consumed by launch 1, so nothing here races with a read."""
    i_n = tl.program_id(0)
    slot = tl.load(state_idx + i_n * stride_idx).to(tl.int64)
    if slot <= 0:
        return
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    pslot = tl.load(state_idx + i_n * stride_idx + PCOL).to(tl.int64)
    p_pend = pend + pslot * stride_pend
    offs = tl.arange(0, BLOCK)
    for t in range(0, eos - bos):
        tok = bos + t
        base = p_pend + t * W
        for s in range(0, H * K, BLOCK):
            m = s + offs < H * K
            tl.store(base + s + offs, tl.load(k + tok * H * K + s + offs, mask=m), mask=m)
        for s in range(0, HV * V, BLOCK):
            m = s + offs < HV * V
            tl.store(base + H * K + s + offs, tl.load(v + tok * HV * V + s + offs, mask=m), mask=m)
        m = offs < HV
        tl.store(base + H * K + HV * V + offs, tl.load(a + tok * HV + offs, mask=m), mask=m)
        tl.store(base + H * K + HV * V + HV + offs, tl.load(b + tok * HV + offs, mask=m), mask=m)
    tl.store(flags + slot + tl.arange(0, 1), tl.full([1], 1.0, tl.float32).to(flags.dtype.element_ty))


@triton.jit(do_not_specialize=["N", "T"])
def _gdn_fused_db_kernel(A_log, a, b, dt_bias, beta, threshold, q, k, v, o, h0, pend, flags, cu_seqlens, state_idx,
                         num_accepted, scale, N: tl.int64, T: tl.int64, H: tl.constexpr, HV: tl.constexpr,
                         K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr, BV: tl.constexpr, W: tl.constexpr,
                         BANK: tl.constexpr, PCOL: tl.constexpr, stride_state: tl.constexpr,
                         stride_idx: tl.constexpr, stride_pend: tl.constexpr, USE_QK_L2NORM: tl.constexpr):
    """Replay and verify in one launch with a double-buffered pending area (stock tiling, BV = 32, so bit-identical).
    flags[slot]: 0 = nothing pending, 1 = pending window in bank 0, 2 = in bank 1. This launch replays the previous
    window from bank (flag - 1) into the committed state (stored once, as stock stores its accepted position), runs the
    new window from the rounded state without storing state, and writes the new window's inputs into the other bank.
    Nothing read here is written here; `gdn_flip_flags` (a separate launch) then records the new bank."""
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    G: tl.constexpr = HV // H
    i_h = i_hv // G
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    T = eos - bos
    if T == 0:
        return
    slot = tl.load(state_idx + i_n * stride_idx).to(tl.int64)
    if slot <= 0:
        return
    pslot = tl.load(state_idx + i_n * stride_idx + PCOL).to(tl.int64)
    f = tl.load(flags + slot).to(tl.int32)
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]
    p_h = h0 + slot * stride_state + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    b_h = tl.zeros([BV, BK], dtype=tl.float32)
    b_h += tl.load(p_h, mask=mask_h, other=0).to(tl.float32)
    A_log_v = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_v = tl.load(dt_bias + i_hv).to(tl.float32)
    p_bank = pend + pslot * stride_pend
    bank_out = tl.where(f > 0, 2 - f, 0)
    if f > 0:
        n = tl.load(num_accepted + i_n).to(tl.int64)
        p_in = p_bank + (f - 1) * BANK
        for i_t in range(0, n):
            base = p_in + i_t * W
            b_k = tl.load(base + i_h * K + o_k, mask=mask_k, other=0).to(tl.float32)
            b_v = tl.load(base + H * K + i_hv * V + o_v, mask=mask_v, other=0).to(tl.float32)
            b_a = tl.load(base + H * K + HV * V + i_hv).to(tl.float32)
            b_b = tl.load(base + H * K + HV * V + HV + i_hv).to(tl.float32)
            b_h = _gdn_step(b_h, b_k, b_v, b_a, b_b, A_log_v, dt_bias_v, beta, threshold, USE_QK_L2NORM)
        if n > 0:
            hb = b_h.to(p_h.dtype.element_ty)
            tl.store(p_h, hb, mask=mask_h)
            b_h = hb.to(tl.float32)
    p_out = p_bank + bank_out * BANK
    p_q = q + (bos * H + i_h) * K + o_k
    p_k = k + (bos * H + i_h) * K + o_k
    p_v = v + (bos * HV + i_hv) * V + o_v
    p_a = a + bos * HV + i_hv
    p_b = b + bos * HV + i_hv
    p_o = o + (bos * HV + i_hv) * V + o_v
    write_k = (i_v == 0) & (i_hv % G == 0)
    write_ab = i_v == 0
    for i_t in range(0, T):
        r_q = tl.load(p_q, mask=mask_k, other=0)
        r_k = tl.load(p_k, mask=mask_k, other=0)
        r_v = tl.load(p_v, mask=mask_v, other=0)
        r_b = tl.load(p_b)
        r_a = tl.load(p_a)
        b_q = r_q.to(tl.float32)
        if USE_QK_L2NORM:
            b_q = b_q * (tl.rsqrt(tl.sum(b_q * b_q) + 1e-6))
        b_q = b_q * scale
        b_h = _gdn_step(b_h, r_k.to(tl.float32), r_v.to(tl.float32), r_a.to(tl.float32), r_b.to(tl.float32),
                        A_log_v, dt_bias_v, beta, threshold, USE_QK_L2NORM)
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(p_o, b_o.to(p_o.dtype.element_ty), mask=mask_v)
        base = p_out + i_t * W
        tl.store(base + H * K + i_hv * V + o_v, r_v.to(pend.dtype.element_ty), mask=mask_v)
        if write_k:
            tl.store(base + i_h * K + o_k, r_k.to(pend.dtype.element_ty), mask=mask_k)
        if write_ab:
            tl.store(base + H * K + HV * V + i_hv, r_a.to(pend.dtype.element_ty))
            tl.store(base + H * K + HV * V + HV + i_hv, r_b.to(pend.dtype.element_ty))
        p_q += H * K
        p_k += H * K
        p_o += HV * V
        p_v += HV * V
        p_b += HV
        p_a += HV


@triton.jit
def _gdn_flip_flags_kernel(flags, cu_seqlens, state_idx, stride_idx: tl.constexpr):
    i_n = tl.program_id(0)
    slot = tl.load(state_idx + i_n * stride_idx).to(tl.int64)
    T = tl.load(cu_seqlens + i_n + 1) - tl.load(cu_seqlens + i_n)
    if (slot <= 0) | (T == 0):
        return
    f = tl.load(flags + slot).to(tl.int32)
    new = tl.where(f == 1, 2, 1)
    tl.store(flags + slot + tl.arange(0, 1), tl.full([1], 0, tl.int32) + new)


def gdn_fused_db_step(A_log, a, b, dt_bias, q, k, v, state, pend, flags, cu_seqlens, state_idx, num_accepted, W,
                      max_tokens, pcol=1, scale=None, out=None, beta=1.0, threshold=20.0, use_qk_l2norm=True,
                      bv=None, num_warps=4, num_stages=3):
    """One speculative step for one GDN layer in two launches: fused replay + verify (+ stash into the free bank),
    then the flag flip. flags: int32 per state slot; pend rows hold 2 banks of max_tokens x W values."""
    _, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    N = cu_seqlens.numel() - 1
    BK, BV, NV = _blocks(K, V)
    if bv is not None:
        BV, NV = bv, triton.cdiv(V, bv)
    if scale is None:
        scale = K ** -0.5
    assert state_idx.ndim == 2 and state_idx.shape[1] > pcol and flags.dtype == torch.int32
    assert pend.shape[1] >= 2 * max_tokens * W
    o = out if out is not None else q.new_empty(T, HV, V)
    _gdn_fused_db_kernel[(1, NV, N * HV)](
        A_log, a.contiguous(), b.contiguous(), dt_bias, beta, threshold, q.contiguous(), k.contiguous(),
        v.contiguous(), o, state, pend, flags, cu_seqlens, state_idx, num_accepted, scale, N, T, H=H, HV=HV, K=K,
        V=V, BK=BK, BV=BV, W=W, BANK=max_tokens * W, PCOL=pcol, stride_state=state.stride(0),
        stride_idx=state_idx.stride(0), stride_pend=pend.stride(0), USE_QK_L2NORM=use_qk_l2norm, num_warps=num_warps,
        num_stages=num_stages)
    _gdn_flip_flags_kernel[(N,)](flags, cu_seqlens, state_idx, stride_idx=state_idx.stride(0))
    return o


def pending_width(H, HV, K, V):
    """Values per token in the pending area: [k (H*K) | v (HV*V) | a (HV) | b (HV)]."""
    return H * K + HV * V + 2 * HV


def gdn_commit_pending(A_log, dt_bias, state, pend, flags, state_idx, num_accepted, H, K, W, pcol=1, beta=1.0,
                       threshold=20.0, use_qk_l2norm=True):
    """state_idx: [N, >=2]; column 0 is the committed state slot, column `pcol` the slot whose pend row holds the
    pending window (stage 1: the request's second, otherwise unused speculative slot; fold B: the same slot).
    flags: one value per state slot (column 0), set by the stash and cleared by prefill."""
    HV, V = state.shape[1], state.shape[2]
    N = state_idx.shape[0]
    BK, BV, NV = _blocks(K, V)
    _gdn_commit_pending_kernel[(1, NV, N * HV)](
        A_log, dt_bias, beta, threshold, state, pend, flags, state_idx, num_accepted, N, H=H, HV=HV, K=K, V=V,
        BK=BK, BV=BV, W=W, PCOL=pcol, stride_state=state.stride(0), stride_idx=state_idx.stride(0),
        stride_pend=pend.stride(0), USE_QK_L2NORM=use_qk_l2norm, num_warps=4, num_stages=3)


def gdn_stash(k, v, a, b, pend, flags, cu_seqlens, state_idx, W, pcol=1):
    _, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    N = cu_seqlens.numel() - 1
    _gdn_stash_kernel[(N,)](k.contiguous(), v.contiguous(), a.contiguous(), b.contiguous(), pend, flags, cu_seqlens,
                            state_idx, H=H, HV=HV, K=K, V=V, W=W, PCOL=pcol, stride_idx=state_idx.stride(0),
                            stride_pend=pend.stride(0), BLOCK=1024, num_warps=4)


def gdn_spec_step(A_log, a, b, dt_bias, q, k, v, state, pend, flags, cu_seqlens, state_idx, num_accepted, W, pcol=1,
                  scale=None, out=None):
    """One speculative step for one GDN layer: commit previous window, verify this one, stash its inputs."""
    H, K = k.shape[2], k.shape[3]
    assert state_idx.ndim == 2 and state_idx.shape[1] > pcol
    gdn_commit_pending(A_log, dt_bias, state, pend, flags, state_idx, num_accepted, H, K, W, pcol)
    o = gdn_verify_nowb(A_log, a, b, dt_bias, q, k, v, state, cu_seqlens, state_idx, scale=scale, out=out)
    gdn_stash(k, v, a, b, pend, flags, cu_seqlens, state_idx, W, pcol)
    return o


def pending_shape(num_slots, HV, K, V, max_tokens):
    """Per slot and value head: a flag, then max_tokens x (k[K], v[V], a, b)."""
    return (num_slots, HV, 1 + max_tokens * (K + V + 2))


def gdn_lazy_verify(A_log, a, b, dt_bias, q, k, v, state, pend, cu_seqlens, state_idx, num_accepted, scale=None,
                    beta=1.0, threshold=20.0, use_qk_l2norm=True, out=None, num_warps=8):
    """Replay-then-verify in one launch. pend: pending_shape(...) in the state dtype, zero-initialised (flag 0)."""
    _, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    N = cu_seqlens.numel() - 1
    if scale is None:
        scale = K ** -0.5
    o = out if out is not None else q.new_empty(T, HV, V)
    stride_idx = state_idx.stride(0) if state_idx.ndim == 2 else 1
    _gdn_lazy_verify_kernel[(N * HV,)](
        A_log, a.contiguous(), b.contiguous(), dt_bias, beta, threshold, q.contiguous(), k.contiguous(),
        v.contiguous(), o, state, pend, cu_seqlens, state_idx, num_accepted, scale, N, T, H=H, HV=HV, K=K, V=V,
        stride_state=state.stride(0), stride_idx=stride_idx, stride_pend_slot=pend.stride(0),
        stride_pend_head=pend.stride(1), USE_QK_L2NORM=use_qk_l2norm, num_warps=num_warps, num_stages=1)
    return o


def _blocks(K, V):
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    assert triton.cdiv(K, BK) == 1
    return BK, BV, triton.cdiv(V, BV)


def gdn_verify_nowb(A_log, a, b, dt_bias, q, k, v, state, cu_seqlens, state_idx, scale=None, beta=1.0,
                    threshold=20.0, use_qk_l2norm=True, out=None):
    """q, k: [1, T, H, K]; v: [1, T, HV, V]; a, b: [T, HV]; state: [slots, HV, V, K]; state_idx: [N] or [N, *]
    (column 0 = committed slot). Returns o: [T, HV, V]."""
    _, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    N = cu_seqlens.numel() - 1
    BK, BV, NV = _blocks(K, V)
    if scale is None:
        scale = K ** -0.5
    o = out if out is not None else q.new_empty(T, HV, V)
    stride_idx = state_idx.stride(0) if state_idx.ndim == 2 else 1
    _gdn_verify_nowb_kernel[(1, NV, N * HV)](
        A_log, a.contiguous(), b.contiguous(), dt_bias, beta, threshold, q.contiguous(), k.contiguous(),
        v.contiguous(), o, state, cu_seqlens, state_idx, scale, N, T, H=H, HV=HV, K=K, V=V, BK=BK, BV=BV,
        stride_state=state.stride(0), stride_idx=stride_idx, USE_QK_L2NORM=use_qk_l2norm, num_warps=4,
        num_stages=3)
    return o


def gdn_commit(A_log, a, b, dt_bias, k, v, state, cu_seqlens, state_idx, num_sampled, beta=1.0, threshold=20.0,
               use_qk_l2norm=True):
    """Replay the first num_sampled[i] verify tokens of request i from its committed state; store once."""
    _, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    N = cu_seqlens.numel() - 1
    BK, BV, NV = _blocks(K, V)
    stride_idx = state_idx.stride(0) if state_idx.ndim == 2 else 1
    _gdn_commit_kernel[(1, NV, N * HV)](
        A_log, a.contiguous(), b.contiguous(), dt_bias, beta, threshold, k.contiguous(), v.contiguous(), state,
        cu_seqlens, state_idx, num_sampled, N, H=H, HV=HV, K=K, V=V, BK=BK, BV=BV, stride_state=state.stride(0),
        stride_idx=stride_idx, USE_QK_L2NORM=use_qk_l2norm, num_warps=4, num_stages=3)
