"""Matrix-form (WY) GDN replay + verify for speculative decoding (kernel work block 2, 2026-09-29).

Same contract as `rollback_kernels.gdn_fused_db_step` (double-buffered pending banks, int32 flags, one committed state
slot per request), but the replayed previous window (n accepted tokens) and the new verify window (T tokens) are
processed as one chunk of C = n + T rows in matrix form instead of a sequential per-token loop:

    per value head, with g_r the log decay, beta_r, unit keys k_r and scaled unit queries q_r (rows of the chunk),
    G = cumsum(g), D[r, i] = exp(G_r - G_i) for i <= r,
    (I + A) U = diag(beta) (V - diag(exp G) K S0^T),        A[r, i] = beta_r D[r, i] (k_r . k_i), i < r
    O_r   = exp(G_r) q_r S0^T + sum_{i <= r} D[r, i] (q_r . k_i) U_i          (verify rows)
    S_n   = exp(G_{n-1}) S0 + sum_{i < n} exp(G_{n-1} - G_i) U_i^T k_i        (stored once, BF16)

The state tile enters only through two tensor-core products (K S0^T and Q S0^T, exact BF16 operands since the
normalisation is applied to the products) and one rank-C update. FP32 x FP32 products run as three BF16 products on
a hi/lo split; the triangular solve is a column elimination over the valid rows of the C x BV tile.

Not bit-identical to the recurrent kernel: the verify rows see the unrounded replayed state and the sums run in a
different order. Against an FP64 recurrence (test_gdn_wy.py) the outputs are ~12% closer than the exact recurrent
kernel's and the committed states equally close (both limited by the BF16 state).

scaled16 (2026-10-15, Claude fork 5d20b23b; SCALED constexpr, off unless the caller passes scale_state; v2: one FP32 scale per 32
V rows, [slots, HV, V / 32], s16.py): S0 = s * M enters the two state products as a column scale (Yk, Yq times the rows' block scale,
like rk / rq), and the commit forms S1 in FP32, then stores (M1, s1) by s16.s16_rescale (s1 = s * exp(Gn), fresh / power-of-two
renormalisation per block; a BV = 64 program owns 2 blocks).
With SCALED False the kernel is the pre-scaled16 one.

gdn32 (2026-10-22, Claude session 099768dc; F32 constexpr, on for an FP32 state; the user's FP32 GDN-state A/B, V67_DESIGN 257): the
state loads, carries and stores in FP32. The two state products split S0 into BF16 hi + lo parts (~16 mantissa bits, the precision
of _dot_f32) against the exact BF16 keys / queries: two tensor-core products each, not TF32 (10 bits). The commit forms S1 in FP32
as before and stores it unrounded. With F32 False (BF16 / FP16 states) the kernel is unchanged.

Tuning (cold 36 layers, 37 lanes x 4, 3 accepted): BV = 64 with 4 warps is best (222 registers, 2 programs per SM).
Slower variants, measured: a transposed form with S0 as the A operand (4.4 ms vs 3.3), issuing the state load first
(4.2), inverting (I + A) before the state products (3.7), register caps 168/128 (spills, 3.4/5.0), tf32x3 instead of
the BF16 split (+0.6 ms at BV = 32).
"""
import os

import torch
import triton
import triton.language as tl

from .rb_kernels import _gdn_flip_flags_kernel

_F32_BV = int(os.environ.get('ARC3_S2_WY_F32_BV', '32') or 32)     # gdn32: the F32 path's V tile (register budget; tuned on the GPU)
from .s16 import BLK as S16_BLK, HI as S16_HI, LO as S16_LO, SR as S16_SR, s16_rescale, s16_round


@triton.jit
def _dot_f32(a, b):
    """FP32 x FP32 product as three BF16 products on a hi/lo split (about 16 mantissa bits per operand)."""
    a_hi = a.to(tl.bfloat16)
    a_lo = (a - a_hi.to(tl.float32)).to(tl.bfloat16)
    b_hi = b.to(tl.bfloat16)
    b_lo = (b - b_hi.to(tl.float32)).to(tl.bfloat16)
    acc = tl.dot(a_hi, b_lo)
    acc = tl.dot(a_lo, b_hi, acc)
    return tl.dot(a_hi, b_hi, acc)


@triton.jit
def _dot_f32_bf16(a, b, acc):
    """acc + a @ b for FP32 a and an exact BF16 b (two BF16 products on a's hi/lo split)."""
    a_hi = a.to(tl.bfloat16)
    a_lo = (a - a_hi.to(tl.float32)).to(tl.bfloat16)
    acc = tl.dot(a_lo, b, acc)
    return tl.dot(a_hi, b, acc)


@triton.jit(do_not_specialize=["N", "T"])
def _gdn_wy_db_kernel(A_log, a, b, dt_bias, beta, threshold, q, k, v, o, h0, pend, flags, cu_seqlens, state_idx,
                      num_accepted, scale, sc, N: tl.int64, T: tl.int64, H: tl.constexpr, HV: tl.constexpr,
                      K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr, C: tl.constexpr, W: tl.constexpr,
                      BANK: tl.constexpr, PCOL: tl.constexpr, stride_state: tl.constexpr, stride_idx: tl.constexpr,
                      stride_pend: tl.constexpr, USE_QK_L2NORM: tl.constexpr, SCALED: tl.constexpr = False,
                      stride_sc: tl.constexpr = 0, LO: tl.constexpr = 0.5, HI: tl.constexpr = 2.0, BLK: tl.constexpr = 32,
                      F32: tl.constexpr = False):
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    G: tl.constexpr = HV // H
    i_h = i_hv // G
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    Tn = eos - bos
    if Tn == 0:
        return
    slot = tl.load(state_idx + i_n * stride_idx).to(tl.int64)
    if slot <= 0:
        return
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    p_h = h0 + slot * stride_state + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    pslot = tl.load(state_idx + i_n * stride_idx + PCOL).to(tl.int64)
    f = tl.load(flags + slot).to(tl.int32)
    n = tl.where(f > 0, tl.load(num_accepted + i_n).to(tl.int64), 0)
    nT = n + Tn
    p_bank = pend + pslot * stride_pend
    bank_in = p_bank + tl.maximum(f - 1, 0) * BANK
    bank_out = p_bank + tl.where(f > 0, 2 - f, 0) * BANK

    r = tl.arange(0, C)
    is_rep = r < n
    is_new = (r >= n) & (r < nT)
    valid = r < nT
    rn = tl.where(is_new, r - n, 0)
    tok = bos + rn

    # chunk rows: previous window's accepted tokens (pending bank), then this window's tokens
    kr = tl.load(bank_in + r[:, None] * W + i_h * K + o_k[None, :], mask=is_rep[:, None], other=0.0)
    kn = tl.load(k + tok[:, None] * (H * K) + i_h * K + o_k[None, :], mask=is_new[:, None], other=0.0)
    Kc = tl.where(is_rep[:, None], kr, kn)
    vr = tl.load(bank_in + r[:, None] * W + H * K + i_hv * V + o_v[None, :], mask=is_rep[:, None], other=0.0)
    vn = tl.load(v + tok[:, None] * (HV * V) + i_hv * V + o_v[None, :], mask=is_new[:, None], other=0.0)
    Vc = tl.where(is_rep[:, None], vr.to(tl.float32), vn.to(tl.float32))
    ar = tl.load(bank_in + r * W + H * K + HV * V + i_hv, mask=is_rep, other=0.0)
    an = tl.load(a + tok * HV + i_hv, mask=is_new, other=0.0)
    br = tl.load(bank_in + r * W + H * K + HV * V + HV + i_hv, mask=is_rep, other=0.0)
    bn = tl.load(b + tok * HV + i_hv, mask=is_new, other=0.0)
    ac = tl.where(is_rep, ar.to(tl.float32), an.to(tl.float32))
    bc = tl.where(is_rep, br.to(tl.float32), bn.to(tl.float32))
    Qc = tl.load(q + tok[:, None] * (H * K) + i_h * K + o_k[None, :], mask=is_new[:, None], other=0.0)

    # gates, as the recurrent kernel
    A_log_v = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_v = tl.load(dt_bias + i_hv).to(tl.float32)
    x = ac + dt_bias_v
    softplus_x = tl.where(beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x)
    g = tl.where(valid, -tl.exp(A_log_v) * softplus_x, 0.0)
    bt = tl.where(valid, tl.sigmoid(bc), 0.0)
    Gc = tl.cumsum(g, axis=0)
    eG = tl.exp(Gc)

    if USE_QK_L2NORM:
        kf = Kc.to(tl.float32)
        qf = Qc.to(tl.float32)
        rk = tl.rsqrt(tl.sum(kf * kf, axis=1) + 1e-6)
        rq = tl.rsqrt(tl.sum(qf * qf, axis=1) + 1e-6) * scale
    else:
        rk = tl.full([C], 1.0, tl.float32)
        rq = tl.full([C], 1.0, tl.float32) * scale

    KK = tl.dot(Kc, tl.trans(Kc)) * rk[:, None] * rk[None, :]
    QK = tl.dot(Qc, tl.trans(Kc)) * rq[:, None] * rk[None, :]
    D = tl.where(r[:, None] >= r[None, :], tl.exp(tl.minimum(Gc[:, None] - Gc[None, :], 0.0)), 0.0)
    A = tl.where(r[:, None] > r[None, :], bt[:, None] * D * KK, 0.0)

    S0 = tl.load(p_h)
    if F32:                                                  # gdn32: S0 = hi + lo in BF16 against the exact BF16 Kc / Qc
        S0h = S0.to(tl.bfloat16)
        S0l = (S0 - S0h.to(tl.float32)).to(tl.bfloat16)
        Yk = tl.dot(Kc, tl.trans(S0l))
        Yk = tl.dot(Kc, tl.trans(S0h), Yk) * rk[:, None]
        Yq = tl.dot(Qc, tl.trans(S0l))
        Yq = tl.dot(Qc, tl.trans(S0h), Yq) * rq[:, None]
    else:
        S0t = tl.trans(S0)
        Yk = tl.dot(Kc.to(S0.dtype), S0t) * rk[:, None]      # FP16 state (10-14): operands in the state's dtype
        Yq = tl.dot(Qc.to(S0.dtype), S0t) * rq[:, None]      # (exact for BF16 inputs; a no-op for a BF16 state)
    if SCALED:                                               # scaled16: S0 = s0 * M, s0 per V row = per column of Yk / Yq
        s0 = tl.load(sc + slot * stride_sc + i_hv * (V // BLK) + o_v // BLK)           # the rows' block scales (gathered)
        p_sb = sc + slot * stride_sc + i_hv * (V // BLK) + i_v * (BV // BLK) + tl.arange(0, BV // BLK)
        Yk = Yk * s0[None, :]
        Yq = Yq * s0[None, :]

    # (I + A) U = diag(beta) (V - diag(exp G) Yk), by column elimination over the valid rows
    U = bt[:, None] * (Vc - eG[:, None] * Yk)
    for i in range(0, nT - 1):
        u_i = tl.sum(tl.where(r[:, None] == i, U, 0.0), axis=0)
        a_i = tl.sum(tl.where(r[None, :] == i, A, 0.0), axis=1)
        U = U - a_i[:, None] * u_i[None, :]

    O = eG[:, None] * Yq + _dot_f32(D * QK, U)
    tl.store(o + tok[:, None] * (HV * V) + i_hv * V + o_v[None, :], O.to(o.dtype.element_ty), mask=is_new[:, None])

    if n > 0:
        Gn = tl.sum(tl.where(r == n - 1, Gc, 0.0), axis=0)
        wv = tl.where(is_rep, tl.exp(Gn - Gc) * rk, 0.0)
        Uw = tl.trans(U * wv[:, None])
        if SCALED:
            dec = tl.exp(Gn)
            S1 = _dot_f32_bf16(Uw, Kc, dec * (S0.to(tl.float32) * s0[:, None]))
            M1, s1 = s16_rescale(S1, tl.load(p_sb) * dec, dec, BV, BLK, LO, HI)
            tl.store(p_h, M1.to(p_h.dtype.element_ty))
            tl.store(p_sb, s1)
        else:
            S1 = _dot_f32_bf16(Uw, Kc, tl.exp(Gn) * S0.to(tl.float32))
            tl.store(p_h, S1.to(p_h.dtype.element_ty))

    # this window's inputs become the pending window (the other bank)
    tl.store(bank_out + rn[:, None] * W + H * K + i_hv * V + o_v[None, :], vn, mask=is_new[:, None])
    if i_v == 0:
        tl.store(bank_out + rn * W + H * K + HV * V + i_hv, an, mask=is_new)
        tl.store(bank_out + rn * W + H * K + HV * V + HV + i_hv, bn, mask=is_new)
        if i_hv % G == 0:
            tl.store(bank_out + rn[:, None] * W + i_h * K + o_k[None, :], kn, mask=is_new[:, None])


def gdn_wy_db_step(A_log, a, b, dt_bias, q, k, v, state, pend, flags, cu_seqlens, state_idx, num_accepted, W,
                   max_tokens, pcol=1, scale=None, out=None, beta=1.0, threshold=20.0, use_qk_l2norm=True, bv=64,
                   num_warps=4, scale_state=None, bv_f32=None):
    """Drop-in for gdn_fused_db_step (BF16, FP16 or FP32 state; the chunk holds up to 2 * max_tokens rows). scale_state: the scaled16
    block scales [slots, HV, V / 32] FP32 (FP16 state only); None = unscaled. An FP32 state (gdn32) takes the F32 path with BV
    ARC3_S2_WY_F32_BV (default 32) unless the caller passes bv_f32."""
    _, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    N = cu_seqlens.numel() - 1
    if scale is None:
        scale = K ** -0.5
    assert state.dtype in (torch.bfloat16, torch.float16, torch.float32) and pend.dtype == torch.bfloat16
    f32 = state.dtype == torch.float32
    if f32:
        assert scale_state is None, 'gdn32: an FP32 state has no scaled16 scales'
        assert q.dtype == k.dtype == torch.bfloat16, ('gdn32: the F32 path needs BF16 q / k (exact hi / lo products)', q.dtype, k.dtype)
        bv = bv_f32 if bv_f32 else _F32_BV
    assert state_idx.ndim == 2 and state_idx.shape[1] > pcol and flags.dtype == torch.int32
    assert pend.shape[1] >= 2 * max_tokens * W and V % bv == 0
    if scale_state is not None:
        assert state.dtype == torch.float16 and scale_state.dtype == torch.float32 and scale_state.shape[1:] == (HV, V // S16_BLK)
        assert scale_state.stride(1) == V // S16_BLK and scale_state.stride(2) == 1 and bv % S16_BLK == 0
    C = max(16, triton.next_power_of_2(2 * max_tokens))
    o = out if out is not None else q.new_empty(T, HV, V)
    (_gdn_wy_db_kernel_sr if (scale_state is not None and S16_SR) else _gdn_wy_db_kernel)[(V // bv, N * HV)](
        A_log, a.contiguous(), b.contiguous(), dt_bias, beta, threshold, q.contiguous(), k.contiguous(),
        v.contiguous(), o, state, pend, flags, cu_seqlens, state_idx, num_accepted, scale,
        scale_state if scale_state is not None else state, N, T, H=H, HV=HV, K=K,
        V=V, BV=bv, C=C, W=W, BANK=max_tokens * W, PCOL=pcol, stride_state=state.stride(0),
        stride_idx=state_idx.stride(0), stride_pend=pend.stride(0), USE_QK_L2NORM=use_qk_l2norm,
        SCALED=scale_state is not None, stride_sc=scale_state.stride(0) if scale_state is not None else 0,
        LO=S16_LO, HI=S16_HI, BLK=S16_BLK, F32=f32, num_warps=num_warps, num_stages=1)
    _gdn_flip_flags_kernel[(N,)](flags, cu_seqlens, state_idx, stride_idx=state_idx.stride(0))
    return o


# ---- scaled16sr (2026-10-27, Claude fork bdfc5402): appended so every kernel above keeps its line and compile cache (s16.py).

@triton.jit(do_not_specialize=["N", "T"])
def _gdn_wy_db_kernel_sr(A_log, a, b, dt_bias, beta, threshold, q, k, v, o, h0, pend, flags, cu_seqlens, state_idx,
                      num_accepted, scale, sc, N: tl.int64, T: tl.int64, H: tl.constexpr, HV: tl.constexpr,
                      K: tl.constexpr, V: tl.constexpr, BV: tl.constexpr, C: tl.constexpr, W: tl.constexpr,
                      BANK: tl.constexpr, PCOL: tl.constexpr, stride_state: tl.constexpr, stride_idx: tl.constexpr,
                      stride_pend: tl.constexpr, USE_QK_L2NORM: tl.constexpr, SCALED: tl.constexpr = False,
                      stride_sc: tl.constexpr = 0, LO: tl.constexpr = 0.5, HI: tl.constexpr = 2.0, BLK: tl.constexpr = 32,
                      F32: tl.constexpr = False):
    """_gdn_wy_db_kernel with the scaled16sr store (s16_round); otherwise its exact source (kept apart: the original's compile cache stays).
    """
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    G: tl.constexpr = HV // H
    i_h = i_hv // G
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    Tn = eos - bos
    if Tn == 0:
        return
    slot = tl.load(state_idx + i_n * stride_idx).to(tl.int64)
    if slot <= 0:
        return
    o_k = tl.arange(0, K)
    o_v = i_v * BV + tl.arange(0, BV)
    p_h = h0 + slot * stride_state + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    pslot = tl.load(state_idx + i_n * stride_idx + PCOL).to(tl.int64)
    f = tl.load(flags + slot).to(tl.int32)
    n = tl.where(f > 0, tl.load(num_accepted + i_n).to(tl.int64), 0)
    nT = n + Tn
    p_bank = pend + pslot * stride_pend
    bank_in = p_bank + tl.maximum(f - 1, 0) * BANK
    bank_out = p_bank + tl.where(f > 0, 2 - f, 0) * BANK

    r = tl.arange(0, C)
    is_rep = r < n
    is_new = (r >= n) & (r < nT)
    valid = r < nT
    rn = tl.where(is_new, r - n, 0)
    tok = bos + rn

    # chunk rows: previous window's accepted tokens (pending bank), then this window's tokens
    kr = tl.load(bank_in + r[:, None] * W + i_h * K + o_k[None, :], mask=is_rep[:, None], other=0.0)
    kn = tl.load(k + tok[:, None] * (H * K) + i_h * K + o_k[None, :], mask=is_new[:, None], other=0.0)
    Kc = tl.where(is_rep[:, None], kr, kn)
    vr = tl.load(bank_in + r[:, None] * W + H * K + i_hv * V + o_v[None, :], mask=is_rep[:, None], other=0.0)
    vn = tl.load(v + tok[:, None] * (HV * V) + i_hv * V + o_v[None, :], mask=is_new[:, None], other=0.0)
    Vc = tl.where(is_rep[:, None], vr.to(tl.float32), vn.to(tl.float32))
    ar = tl.load(bank_in + r * W + H * K + HV * V + i_hv, mask=is_rep, other=0.0)
    an = tl.load(a + tok * HV + i_hv, mask=is_new, other=0.0)
    br = tl.load(bank_in + r * W + H * K + HV * V + HV + i_hv, mask=is_rep, other=0.0)
    bn = tl.load(b + tok * HV + i_hv, mask=is_new, other=0.0)
    ac = tl.where(is_rep, ar.to(tl.float32), an.to(tl.float32))
    bc = tl.where(is_rep, br.to(tl.float32), bn.to(tl.float32))
    Qc = tl.load(q + tok[:, None] * (H * K) + i_h * K + o_k[None, :], mask=is_new[:, None], other=0.0)

    # gates, as the recurrent kernel
    A_log_v = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_v = tl.load(dt_bias + i_hv).to(tl.float32)
    x = ac + dt_bias_v
    softplus_x = tl.where(beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x)
    g = tl.where(valid, -tl.exp(A_log_v) * softplus_x, 0.0)
    bt = tl.where(valid, tl.sigmoid(bc), 0.0)
    Gc = tl.cumsum(g, axis=0)
    eG = tl.exp(Gc)

    if USE_QK_L2NORM:
        kf = Kc.to(tl.float32)
        qf = Qc.to(tl.float32)
        rk = tl.rsqrt(tl.sum(kf * kf, axis=1) + 1e-6)
        rq = tl.rsqrt(tl.sum(qf * qf, axis=1) + 1e-6) * scale
    else:
        rk = tl.full([C], 1.0, tl.float32)
        rq = tl.full([C], 1.0, tl.float32) * scale

    KK = tl.dot(Kc, tl.trans(Kc)) * rk[:, None] * rk[None, :]
    QK = tl.dot(Qc, tl.trans(Kc)) * rq[:, None] * rk[None, :]
    D = tl.where(r[:, None] >= r[None, :], tl.exp(tl.minimum(Gc[:, None] - Gc[None, :], 0.0)), 0.0)
    A = tl.where(r[:, None] > r[None, :], bt[:, None] * D * KK, 0.0)

    S0 = tl.load(p_h)
    if F32:                                                  # gdn32: S0 = hi + lo in BF16 against the exact BF16 Kc / Qc
        S0h = S0.to(tl.bfloat16)
        S0l = (S0 - S0h.to(tl.float32)).to(tl.bfloat16)
        Yk = tl.dot(Kc, tl.trans(S0l))
        Yk = tl.dot(Kc, tl.trans(S0h), Yk) * rk[:, None]
        Yq = tl.dot(Qc, tl.trans(S0l))
        Yq = tl.dot(Qc, tl.trans(S0h), Yq) * rq[:, None]
    else:
        S0t = tl.trans(S0)
        Yk = tl.dot(Kc.to(S0.dtype), S0t) * rk[:, None]      # FP16 state (10-14): operands in the state's dtype
        Yq = tl.dot(Qc.to(S0.dtype), S0t) * rq[:, None]      # (exact for BF16 inputs; a no-op for a BF16 state)
    if SCALED:                                               # scaled16: S0 = s0 * M, s0 per V row = per column of Yk / Yq
        s0 = tl.load(sc + slot * stride_sc + i_hv * (V // BLK) + o_v // BLK)           # the rows' block scales (gathered)
        p_sb = sc + slot * stride_sc + i_hv * (V // BLK) + i_v * (BV // BLK) + tl.arange(0, BV // BLK)
        Yk = Yk * s0[None, :]
        Yq = Yq * s0[None, :]

    # (I + A) U = diag(beta) (V - diag(exp G) Yk), by column elimination over the valid rows
    U = bt[:, None] * (Vc - eG[:, None] * Yk)
    for i in range(0, nT - 1):
        u_i = tl.sum(tl.where(r[:, None] == i, U, 0.0), axis=0)
        a_i = tl.sum(tl.where(r[None, :] == i, A, 0.0), axis=1)
        U = U - a_i[:, None] * u_i[None, :]

    O = eG[:, None] * Yq + _dot_f32(D * QK, U)
    tl.store(o + tok[:, None] * (HV * V) + i_hv * V + o_v[None, :], O.to(o.dtype.element_ty), mask=is_new[:, None])

    if n > 0:
        Gn = tl.sum(tl.where(r == n - 1, Gc, 0.0), axis=0)
        wv = tl.where(is_rep, tl.exp(Gn - Gc) * rk, 0.0)
        Uw = tl.trans(U * wv[:, None])
        if SCALED:
            dec = tl.exp(Gn)
            S1 = _dot_f32_bf16(Uw, Kc, dec * (S0.to(tl.float32) * s0[:, None]))
            M1, s1 = s16_rescale(S1, tl.load(p_sb) * dec, dec, BV, BLK, LO, HI)
            tl.store(p_h, s16_round(M1, i_hv * V * K + o_v[:, None] * K + o_k[None, :]))
            tl.store(p_sb, s1)
        else:
            S1 = _dot_f32_bf16(Uw, Kc, tl.exp(Gn) * S0.to(tl.float32))
            tl.store(p_h, S1.to(p_h.dtype.element_ty))

    # this window's inputs become the pending window (the other bank)
    tl.store(bank_out + rn[:, None] * W + H * K + i_hv * V + o_v[None, :], vn, mask=is_new[:, None])
    if i_v == 0:
        tl.store(bank_out + rn * W + H * K + HV * V + i_hv, an, mask=is_new)
        tl.store(bank_out + rn * W + H * K + HV * V + HV + i_hv, bn, mask=is_new)
        if i_hv % G == 0:
            tl.store(bank_out + rn[:, None] * W + i_h * K + o_k[None, :], kn, mask=is_new[:, None])
