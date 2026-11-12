"""stage2_v1 'persist' kernels: commit a paused lane's pending GDN window at resume (M2, option C).

The WY / fused double-buffered verify kernels keep the last speculative window of a slot in bank f - 1 of its pending
area (f = flags[slot] in {1, 2}; layout per token [k (H*K) | v (HV*V) | a (HV) | b (HV)], bank = (k + 1) tokens), and the
next speculative step commits num_accepted of those tokens. A persistent lane that stops pauses with that window
uncommitted; at resume the scheduler's frontier F and the window start W give the kept count m = F - W. This kernel
replays m tokens with the stock recurrence (rb_kernels._gdn_step: stock tiling, FP32, one BF16 rounding), i.e. the
committed state equals stock's per-position state bit for bit, as rb_kernels._gdn_commit_pending_kernel.
scaled16 (2026-10-15, Claude fork 5d20b23b; SCALED constexpr, off unless the caller passes scale_state): loads S = s * M (s16.py),
replays the same FP32 recurrence, tracks the decay as the product of the tokens' exp(g) (the factors applied to the state), and
stores (M, s) by s16.s16_rescale."""
import torch
import triton
import triton.language as tl

from .rb_kernels import _blocks, _gdn_step
from .s16 import BLK as S16_BLK, HI as S16_HI, LO as S16_LO, SR as S16_SR, s16_rescale, s16_round


@triton.jit(do_not_specialize=["N"])
def _gdn_resume_commit_kernel(A_log, dt_bias, beta, threshold, h0, pend, flags, slots, num_commit, sc, N: tl.int64,
                              H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr,
                              BV: tl.constexpr, W: tl.constexpr, BANK: tl.constexpr, stride_state: tl.constexpr,
                              stride_pend: tl.constexpr, USE_QK_L2NORM: tl.constexpr, SCALED: tl.constexpr = False,
                              stride_sc: tl.constexpr = 0, LO: tl.constexpr = 0.5, HI: tl.constexpr = 2.0, BLK: tl.constexpr = 32):
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    slot = tl.load(slots + i_n).to(tl.int64)
    if slot <= 0:
        return
    f = tl.load(flags + slot).to(tl.int32)
    n = tl.where(f > 0, tl.load(num_commit + i_n).to(tl.int64), 0)
    if n <= 0:
        return
    bank = pend + slot * stride_pend + (f - 1).to(tl.int64) * BANK
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
    if SCALED:
        s0 = tl.load(sc + slot * stride_sc + i_hv * (V // BLK) + o_v // BLK)            # the rows' block scales (gathered)
        p_sb = sc + slot * stride_sc + i_hv * (V // BLK) + i_v * (BV // BLK) + tl.arange(0, BV // BLK)
        b_h = b_h * s0[:, None]
        dec = A_log_v * 0.0 + 1.0
    for i_t in range(0, n):
        base = bank + i_t * W
        b_k = tl.load(base + i_h * K + o_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(base + H * K + i_hv * V + o_v, mask=mask_v, other=0).to(tl.float32)
        b_a = tl.load(base + H * K + HV * V + i_hv).to(tl.float32)
        b_b = tl.load(base + H * K + HV * V + HV + i_hv).to(tl.float32)
        b_h = _gdn_step(b_h, b_k, b_v, b_a, b_b, A_log_v, dt_bias_v, beta, threshold, USE_QK_L2NORM)
        if SCALED:                                   # the factor _gdn_step applied: exp(g), g = -exp(A_log) softplus(a + dt_bias)
            x = b_a + dt_bias_v
            softplus_x = tl.where(beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x)
            dec = dec * tl.exp(-tl.exp(A_log_v) * softplus_x)
    if SCALED:
        M1, s1 = s16_rescale(b_h, tl.load(p_sb) * dec, dec, BV, BLK, LO, HI)
        tl.store(p_h, M1.to(p_h.dtype.element_ty), mask=mask_h)
        tl.store(p_sb, s1)
    else:
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=mask_h)


def gdn_resume_commit(A_log, dt_bias, state, pend, flags, slots, num_commit, H, K, W, T1, beta=1.0, threshold=20.0,
                      use_qk_l2norm=True, scale_state=None):
    """state: [slots, HV, V, K] (committed recurrent states); pend: [slots, 2 * T1 * W] BF16 pending banks; flags:
    int32 per slot; slots, num_commit: [N] (int32/int64) resumed lanes and their kept counts. Leaves the flags set
    (the caller clears them after the conv realignment). scale_state: scaled16 block scales [slots, HV, V / 32] FP32, or None."""
    HV, V = state.shape[1], state.shape[2]
    N = slots.shape[0]
    BK, BV, NV = _blocks(K, V)
    if scale_state is not None:
        assert state.dtype == torch.float16 and scale_state.dtype == torch.float32 and scale_state.shape[1:] == (HV, V // S16_BLK)
        assert scale_state.stride(1) == V // S16_BLK and scale_state.stride(2) == 1 and BV % S16_BLK == 0
    (_gdn_resume_commit_kernel_sr if (scale_state is not None and S16_SR) else _gdn_resume_commit_kernel)[(1, NV, N * HV)](
        A_log, dt_bias, beta, threshold, state, pend, flags, slots, num_commit,
        scale_state if scale_state is not None else state, N, H=H, HV=HV, K=K, V=V, BK=BK,
        BV=BV, W=W, BANK=T1 * W, stride_state=state.stride(0), stride_pend=pend.stride(0),
        USE_QK_L2NORM=use_qk_l2norm, SCALED=scale_state is not None,
        stride_sc=scale_state.stride(0) if scale_state is not None else 0, LO=S16_LO, HI=S16_HI, BLK=S16_BLK, num_warps=4,
        num_stages=3)


def realign_conv(conv, slots, offsets, width):
    """conv: [slots, dim, L] logical conv-state view (token axis last); for each resumed lane, move the `width` columns
    starting at `offsets` (= m - 1, the speculative rolling offset of its last window) to columns 0..width-1, where the
    prefill kernels read the initial state. Pure copies."""
    import torch
    if slots.numel() == 0:
        return
    rows = conv.index_select(0, slots)                                  # [N, dim, L]
    idx = offsets.view(-1, 1, 1) + torch.arange(width, device=conv.device).view(1, 1, width)
    moved = rows.gather(2, idx.expand(-1, rows.shape[1], -1))           # [N, dim, width]
    rows[..., :width] = moved
    conv.index_copy_(0, slots, rows)


# ---- scaled16sr (2026-10-27, Claude fork bdfc5402): appended so every kernel above keeps its line and compile cache (s16.py).

@triton.jit(do_not_specialize=["N"])
def _gdn_resume_commit_kernel_sr(A_log, dt_bias, beta, threshold, h0, pend, flags, slots, num_commit, sc, N: tl.int64,
                              H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr,
                              BV: tl.constexpr, W: tl.constexpr, BANK: tl.constexpr, stride_state: tl.constexpr,
                              stride_pend: tl.constexpr, USE_QK_L2NORM: tl.constexpr, SCALED: tl.constexpr = False,
                              stride_sc: tl.constexpr = 0, LO: tl.constexpr = 0.5, HI: tl.constexpr = 2.0, BLK: tl.constexpr = 32):
    """_gdn_resume_commit_kernel with the scaled16sr store (s16_round); otherwise its exact source (kept apart: the original's compile cache stays).
    """
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    slot = tl.load(slots + i_n).to(tl.int64)
    if slot <= 0:
        return
    f = tl.load(flags + slot).to(tl.int32)
    n = tl.where(f > 0, tl.load(num_commit + i_n).to(tl.int64), 0)
    if n <= 0:
        return
    bank = pend + slot * stride_pend + (f - 1).to(tl.int64) * BANK
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
    if SCALED:
        s0 = tl.load(sc + slot * stride_sc + i_hv * (V // BLK) + o_v // BLK)            # the rows' block scales (gathered)
        p_sb = sc + slot * stride_sc + i_hv * (V // BLK) + i_v * (BV // BLK) + tl.arange(0, BV // BLK)
        b_h = b_h * s0[:, None]
        dec = A_log_v * 0.0 + 1.0
    for i_t in range(0, n):
        base = bank + i_t * W
        b_k = tl.load(base + i_h * K + o_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(base + H * K + i_hv * V + o_v, mask=mask_v, other=0).to(tl.float32)
        b_a = tl.load(base + H * K + HV * V + i_hv).to(tl.float32)
        b_b = tl.load(base + H * K + HV * V + HV + i_hv).to(tl.float32)
        b_h = _gdn_step(b_h, b_k, b_v, b_a, b_b, A_log_v, dt_bias_v, beta, threshold, USE_QK_L2NORM)
        if SCALED:                                   # the factor _gdn_step applied: exp(g), g = -exp(A_log) softplus(a + dt_bias)
            x = b_a + dt_bias_v
            softplus_x = tl.where(beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x)
            dec = dec * tl.exp(-tl.exp(A_log_v) * softplus_x)
    if SCALED:
        M1, s1 = s16_rescale(b_h, tl.load(p_sb) * dec, dec, BV, BLK, LO, HI)
        tl.store(p_h, s16_round(M1, i_hv * V * K + o_v[:, None] * K + o_k[None, :]), mask=mask_h)
        tl.store(p_sb, s1)
    else:
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), mask=mask_h)
