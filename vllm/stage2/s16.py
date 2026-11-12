"""stage2_v1 's16' module (2026-10-15, Claude fork 5d20b23b; scaled16 GDN state, dev build for the inference-stack session to integrate):
the one accessor for the scaled16 recurrent state, used by every state site.

State: per GDN layer and slot, the FP16 matrix M [HV, V, K] (the existing ssm cache, dtype float16) and an FP32 scale s [HV, V / BLK],
one per (value head, block of BLK = 32 V rows): S = s[h, r // BLK] * M[h, r, :]. The scale lives in the GDN page after rollback's
pending area (HV * V / BLK * 4 = 768 bytes, inside the page's existing padding: no block-size or KV change; v1's per-row scale,
24,576 bytes, cost 0.93% of the KV tokens at N 32 / 64,684, lane check 10-15 02:20), so a page carries its scale. Every kernel owns
whole 32-row blocks (WY BV 64 = 2 blocks, persist / decode BV 32 = 1 block), so the renormalisation stays program-local; the decay is
per head, identical for all blocks. Within a block FP16's exponent covers the rows' range (elements down to 2^-14 of the block max
stay normal); the measured emulation used one scale per head (128 rows).

Store rule (the measured emulation, harnesspod gdn_store_emul.scaled_store with FP16, lo 0.5, hi 2; per row instead of per head):
  s_new = s_old * D, D = exp(sum of the log gates since the last store) (carries the decay exactly: no stagnation of M);
  fresh (s_new = amax, or 1 if amax = 0) for new requests (no initial state), D < 2^-20, s_new < 1e-30, or s_new non-finite / <= 0;
  ratio = amax / s_new with amax = max |S| over the block; outside [LO, HI] (and amax > 0): s_new *= 2^e, e = floor(log2(ratio) + 0.5),
  clamped to [-120, 120] (a power of two: exact); M = S / s_new, rounded to FP16 by the store.
  (The emulation rounds e with torch.round (ties to even); ties need log2(ratio) = k + 1/2 exactly, which an FP32 ratio never has.)
Loads: S = M * s (FP32). The WY kernel applies s to its products (K S^T, Q S^T) as a column scale, like its rk / rq normalisation.

Switch: ARC3_S2_GDN_STATE=scaled16 (launch-time; needs ARC3_S2_SSM_DTYPE=float16 and the rollback + wy parts). Unset: nothing here
runs and the kernels' SCALED constexpr is False (the compiled kernels are the pre-scaled16 ones).

Prefill implementation (2026-10-19, Claude session 099768dc; ARC3_S2_S16_PREFILL=torch (default, unset) | triton): 'torch' is the
eager torch read / store above (~57 small ops per GDN layer per prefill call, and seq_log_decay's FP64 cumsum over the packed gates,
a sequential scan that costs ~0.78 ms per layer at 4K prefill tokens). 'triton' (v2, V67_DESIGN 147) runs the read and the store
rule as two Triton kernels, operation for operation (FP64 where store_rule uses FP64, IEEE FP32 division, the FP16 rounding of the
stock store line done in the kernel), except the per-sequence log decay: v2 sums each sequence's own gates in FP64 inside the store
kernel (blocks of TRI_BT tokens, a fixed order: deterministic), instead of the difference of two global FP64 prefix sums. That is more
precise (no cancellation against the running prefix) and not bit-identical to 'torch': about 1e-4 of the scales move by one FP32 ulp.
With the decay injected (tri_store ld=..., tests only) the kernel is bit-identical to 'torch'. The store kernel returns M as FP16, so
the stock `.to(ssm_state.dtype)` is a no-op and the stock index_put writes it. STATS['prefill_impl'] / ['prefill_decay'] record it."""
import os

import torch
import triton
import triton.language as tl
from triton.language.extra import libdevice

MODE = os.environ.get('ARC3_S2_GDN_STATE', ''); ENABLED = MODE in ('scaled16', 'scaled16sr'); SR = MODE == 'scaled16sr'   # SR: see the end
PREFILL = os.environ.get('ARC3_S2_S16_PREFILL', '') or 'torch'
assert PREFILL in ('torch', 'triton'), ('ARC3_S2_S16_PREFILL must be torch or triton', PREFILL)
LO, HI = 0.5, 2.0
BLK = 32           # V rows per scale
D_MIN = 2.0 ** -20
S_MIN = 1e-30
E_MAX = 120
TRI_BV = 32         # V rows per program in the Triton prefill kernels (one scale block)
TRI_BT = 256        # gate tokens per step of the v2 decay sum
DBL_MAX = 1.7976931348623157e308
STATS = dict(enabled=ENABLED, sr=SR, prefill_impl=PREFILL, prefill_decay='direct' if PREFILL == 'triton' else 'cumsum', prefill_stores=0,
             prefill_rows=0, decode_calls=0, layers_wrapped=0)


def scale_bytes(HV, V):
    return HV * (V // BLK) * 4


@triton.jit
def s16_rescale(S1, s1, dec, BV: tl.constexpr, BLK: tl.constexpr, LO: tl.constexpr, HI: tl.constexpr):
    """S1: [BV, K] FP32 new states (BV V rows of one head = BV / BLK scale blocks); s1: [BV / BLK] carried block scales (s_old * dec);
    dec: FP32 scalar decay since the last store. Returns (M [BV, K] FP32, block max |M| in [LO, HI] or 0; s [BV / BLK] FP32)."""
    NB: tl.constexpr = BV // BLK
    amax = tl.max(tl.reshape(tl.max(tl.abs(S1), axis=1), [NB, BLK]), axis=1)
    fresh = (dec < 9.5367431640625e-07) | (s1 < 1e-30) | (s1 != s1) | (s1 > 3.0e38)
    s = tl.where(fresh, tl.where(amax > 0, amax, 1.0), s1)
    ratio = amax / s
    out = (ratio > HI) | ((ratio < LO) & (amax > 0))
    e = tl.where(out, tl.floor(tl.log2(tl.where(out, ratio, 1.0)) + 0.5), 0.0)
    e = tl.minimum(tl.maximum(e, -120.0), 120.0)
    p2 = ((e.to(tl.int32) + 127) << 23).to(tl.float32, bitcast=True)      # exactly 2^e
    s = s * p2
    s_rows = tl.reshape(tl.broadcast_to(s[:, None], [NB, BLK]), [BV])
    return S1 / s_rows[:, None], s


def expand(s):
    """Block scales [..., V / BLK] -> per-row scales [..., V]."""
    return s.repeat_interleave(BLK, dim=-1)


def store_rule(S, s_old, log_decay, has_init):
    """Torch form of the store rule (prefill site and tests). S: [n, HV, V, K] FP32 new states; s_old: [n, HV, V / BLK] FP32 old block
    scales (any content where has_init is False); log_decay: [n, HV] summed log gates since the last store (FP64 or FP32); has_init:
    [n] bool. Returns (M [n, HV, V, K] FP32 (the caller's cast rounds it to FP16), s [n, HV, V / BLK] FP32)."""
    n, HV, V, K = S.shape
    dec = torch.exp(log_decay.double())                                    # [n, HV]
    amax = S.abs().reshape(n, HV, V // BLK, BLK * K).amax(-1).double()     # [n, HV, V / BLK]
    s = s_old.double() * dec[..., None]
    fresh = ((~has_init.bool())[:, None, None] | (dec < D_MIN)[..., None] | ~torch.isfinite(s) | (s <= 0) | (s < S_MIN))
    s = torch.where(fresh, torch.where(amax > 0, amax, torch.ones_like(amax)), s)
    ratio = amax / s
    out = (ratio > HI) | ((ratio < LO) & (amax > 0))
    e = torch.where(out, torch.floor(torch.log2(torch.where(out, ratio, torch.ones_like(ratio))) + 0.5),
                    torch.zeros_like(ratio)).clamp(-E_MAX, E_MAX)
    s = (s * torch.exp2(e)).float()
    return S / expand(s)[..., None], s


def load(M, s):
    """S = M * s in FP32. M: [..., V, K] (FP16 cache rows); s: [..., V / BLK] block scales."""
    return M.float() * expand(s)[..., None]


def seq_log_decay(g, cu, n):
    """g: [1, T, HV] (or [T, HV]) FP32 log gates of the packed prefill tokens; cu: [>= n + 1] offsets. Returns [n, HV] FP64 sums."""
    HV = g.shape[-1]
    g2 = g.reshape(-1, HV).double()
    cs = torch.cat([g2.new_zeros(1, HV), torch.cumsum(g2, 0)], 0)
    c = cu[:n + 1].long()
    return cs[c[1:]] - cs[c[:-1]]


if os.environ.get('TRITON_INTERPRET', '') == '1':        # the CPU interpreter has no libdevice (logic tests only)
    @triton.jit
    def _exp64(x):
        return tl.exp(x)

    @triton.jit
    def _log2_64(x):
        return tl.log2(x)
else:
    @triton.jit
    def _exp64(x):
        return libdevice.exp(x)

    @triton.jit
    def _log2_64(x):
        return libdevice.log2(x)


@triton.jit
def _s16_prefill_read_kernel(M, out, scale, idx, has_init, stride_sc, HV: tl.constexpr, V: tl.constexpr, K: tl.constexpr,
                             BV: tl.constexpr, BLK: tl.constexpr):
    """prefill_read in one pass: out = M * s (FP32), exact zero where has_init is False. One program per (V block, sequence * HV)."""
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    o_v = i_v * BV + tl.arange(0, BV)
    o_k = tl.arange(0, K)
    off = i_nh.to(tl.int64) * (V * K) + o_v[:, None] * K + o_k[None, :]
    slot = tl.load(idx + i_n).to(tl.int64)
    hi = tl.load(has_init + i_n).to(tl.int32)
    s = tl.load(scale + slot * stride_sc + i_hv * (V // BLK) + o_v // BLK)
    m = tl.load(M + off).to(tl.float32)
    tl.store(out + off, tl.where(hi != 0, m * s[:, None], 0.0))


@triton.jit
def _s16_prefill_store_kernel(S, Mout, G, cu, LD, LD_OUT, scale, idx, has_init, stride_sc, HV: tl.constexpr, V: tl.constexpr,
                              K: tl.constexpr, BV: tl.constexpr, BLK: tl.constexpr, LO: tl.constexpr, HI: tl.constexpr,
                              D_MIN: tl.constexpr, S_MIN: tl.constexpr, E_MAX: tl.constexpr, DBL_MAX: tl.constexpr,
                              BT: tl.constexpr, USE_LD: tl.constexpr, WRITE_LD: tl.constexpr):
    """store_rule, the scale write and the stock FP16 rounding in one pass, operation for operation: Mout = (S / s).to(FP16),
    scale[slot] = s. The summed log gate (v2): the FP64 sum of the sequence's own gates G[bos:eos, head] (FP32 [T, HV]), BT tokens per
    step into a [BT] FP64 accumulator, then one tree sum: a fixed order. USE_LD (tests): take it from LD [n, HV] (FP64) instead, which
    makes the kernel bit-identical to the torch path; WRITE_LD (tests): write it to LD_OUT. One program per (V block, sequence * HV)."""
    NB: tl.constexpr = BV // BLK
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    if USE_LD:
        ld = tl.load(LD + i_n * HV + i_hv)
    else:
        bos = tl.load(cu + i_n).to(tl.int64)
        eos = tl.load(cu + i_n + 1).to(tl.int64)
        o_t = tl.arange(0, BT)
        acc = tl.zeros([BT], dtype=tl.float64)
        for t0 in range(bos, eos, BT):
            acc += tl.load(G + (t0 + o_t) * HV + i_hv, mask=(t0 + o_t) < eos, other=0.0).to(tl.float64)
        ld = tl.sum(acc, axis=0)
    if WRITE_LD:
        tl.store(LD_OUT + i_n * HV + i_hv, ld, mask=i_v == 0)
    dec = _exp64(ld)                                                              # FP64, as torch.exp(log_decay.double())
    o_v = i_v * BV + tl.arange(0, BV)
    o_k = tl.arange(0, K)
    off = i_nh.to(tl.int64) * (V * K) + o_v[:, None] * K + o_k[None, :]
    S1 = tl.load(S + off)                                                         # [BV, K] FP32
    amax = tl.max(tl.reshape(tl.max(tl.abs(S1), axis=1), [NB, BLK]), axis=1).to(tl.float64)
    nan = tl.max(tl.reshape(tl.max((S1 != S1).to(tl.int32), axis=1), [NB, BLK]), axis=1)
    amax = tl.where(nan > 0, tl.full([NB], 0x7FF8000000000000, tl.int64).to(tl.float64, bitcast=True), amax)   # torch's amax: NaN
    slot = tl.load(idx + i_n).to(tl.int64)
    p_s = scale + slot * stride_sc + i_hv * (V // BLK) + i_v * NB + tl.arange(0, NB)
    s = tl.load(p_s).to(tl.float64) * dec
    hi = tl.load(has_init + i_n).to(tl.int32)
    fresh = ((hi == 0) | (dec < tl.full([], D_MIN, tl.float64)) | (s != s) | (s > tl.full([], DBL_MAX, tl.float64)) | (s <= 0)
             | (s < tl.full([], S_MIN, tl.float64)))
    s = tl.where(fresh, tl.where(amax > 0, amax, 1.0), s)
    ratio = amax / s                                                              # FP64 (div.rn.f64)
    out = (ratio > HI) | ((ratio < LO) & (amax > 0))
    e = tl.where(out, tl.floor(_log2_64(tl.where(out, ratio, 1.0)) + 0.5), 0.0)
    e = tl.minimum(tl.maximum(e, -E_MAX), E_MAX)
    s32 = (s * ((e.to(tl.int64) + 1023) << 52).to(tl.float64, bitcast=True)).to(tl.float32)     # exactly 2^e (torch.exp2)
    s_rows = tl.reshape(tl.broadcast_to(s32[:, None], [NB, BLK]), [BV])
    M = tl.math.div_rn(S1, tl.broadcast_to(s_rows[:, None], [BV, K]))            # IEEE FP32 division, as torch
    tl.store(Mout + off, M.to(Mout.dtype.element_ty))                             # the stock .to(FP16), round to nearest even
    tl.store(p_s, s32)


def _tri_check(x, scale, V, K):
    assert V % TRI_BV == 0 and TRI_BV % BLK == 0 and K == triton.next_power_of_2(K), ('s16 triton prefill: shapes', V, K)
    assert scale.dtype == torch.float32 and scale.stride(1) == V // BLK and scale.stride(2) == 1, ('s16 triton prefill: scale', scale.stride())
    assert x.is_contiguous(), 's16 triton prefill: state not contiguous'


def tri_read(initial_state, scale, idx, has_init):
    """prefill_read's Triton form (one launch)."""
    n, HV, V, K = initial_state.shape
    _tri_check(initial_state, scale, V, K)
    out = torch.empty(initial_state.shape, dtype=torch.float32, device=initial_state.device)
    if n:
        _s16_prefill_read_kernel[(V // TRI_BV, n * HV)](initial_state, out, scale, idx, has_init, scale.stride(0), HV=HV, V=V, K=K,
                                                        BV=TRI_BV, BLK=BLK, num_warps=4)
    return out


def tri_store(final_state, scale, idx, has_init, g, cu, ld=None, ld_out=None):
    """prefill_store's Triton form (v2): writes scale[idx] and returns M as FP16; one launch (the decay is summed in the kernel).
    Tests only: ld (FP64 [n, HV], e.g. seq_log_decay's) replaces the in-kernel decay; ld_out (FP64 [n, HV]) receives it."""
    n, HV, V, K = final_state.shape
    _tri_check(final_state, scale, V, K)
    G = g.reshape(-1, HV)
    if G.dtype != torch.float32 or not G.is_contiguous():
        G = G.float().contiguous()
    M = torch.empty(final_state.shape, dtype=torch.float16, device=final_state.device)
    if n:
        (_s16_prefill_store_kernel_sr if SR else _s16_prefill_store_kernel)[(V // TRI_BV, n * HV)](final_state, M, G, cu, G if ld is None else ld, G if ld_out is None else ld_out,
                                                         scale, idx, has_init, scale.stride(0), HV=HV, V=V, K=K, BV=TRI_BV, BLK=BLK,
                                                         LO=LO, HI=HI, D_MIN=D_MIN, S_MIN=S_MIN, E_MAX=E_MAX, DBL_MAX=DBL_MAX,
                                                         BT=TRI_BT, USE_LD=ld is not None, WRITE_LD=ld_out is not None,
                                                         num_warps=4)
    return M


def prefill_read(initial_state, scale, idx, has_init):
    """The prefill read site: initial_state = the gathered FP16 M rows (rows without initial state already zeroed by the stock code);
    returns the FP32 initial states M * s (exact zero where has_init is False, whatever the stale scale holds)."""
    if PREFILL == 'triton':
        return tri_read(initial_state, scale, idx, has_init)
    s = scale.index_select(0, idx)
    return torch.where(has_init.bool()[:, None, None, None], load(initial_state, s), torch.zeros((), device=s.device))


def prefill_store(final_state, scale, idx, has_init, g, cu):
    """The prefill store site: final_state [n, HV, V, K] FP32 from the chunk kernel; writes the new scales into scale[idx] and returns M
    (FP32, in band; FP16 already rounded in the 'triton' mode), which the stock line `ssm_state[idx] = last_recurrent_state.to(
    ssm_state.dtype)` rounds to FP16 (a no-op for 'triton') and stores."""
    if PREFILL == 'triton':
        M = tri_store(final_state, scale, idx, has_init, g, cu)
    else:
        n = final_state.shape[0]
        M, s = store_rule(final_state, scale.index_select(0, idx), seq_log_decay(g, cu, n), has_init)
        scale.index_copy_(0, idx, s)
    STATS['prefill_stores'] += 1
    return M


def wrap_chunk_op(layer, scale, get_forward_context):
    """Wrap one GDN layer's chunk op instance (CustomOp._forward_method) so the prefill reads M * s and stores (M, s). Needs the layer's
    prefill metadata (prefill_state_indices, prefill_has_initial_state, prefill_query_start_loc), read from the forward context."""
    op = layer.chunk_gated_delta_rule
    if getattr(op, '_s16_wrapped', False):
        return
    orig = op._forward_method
    prefix = layer.prefix

    def forward(*args, **kw):
        assert not args, 'scaled16: the chunk op is called with keyword arguments in qwen_gdn_linear_attn'
        md = get_forward_context().attn_metadata
        md = md[prefix] if isinstance(md, dict) else md
        idx = md.prefill_state_indices if PREFILL == 'triton' else md.prefill_state_indices.long()
        has_init = md.prefill_has_initial_state
        kw['initial_state'] = prefill_read(kw['initial_state'], scale, idx, has_init)
        o, fin = orig(**kw)
        return o, prefill_store(fin, scale, idx, has_init, kw['g'], kw['cu_seqlens'])

    op._forward_method = forward
    op._s16_wrapped = True
    STATS['layers_wrapped'] += 1


@triton.jit(do_not_specialize=["N"])
def _gdn_s16_decode_kernel(A_log, a, b, dt_bias, beta, threshold, q, k, v, o, h0, sc, cu_seqlens, state_idx, scale, N: tl.int64,
                           H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr,
                           BV: tl.constexpr, stride_state: tl.constexpr, stride_sc: tl.constexpr, LO: tl.constexpr,
                           HI: tl.constexpr, USE_QK_L2NORM: tl.constexpr, BLK: tl.constexpr):
    """Non-speculative rows (0-draft decode) on a scaled16 state: the stock fused_sigmoid_gating_delta_rule_update recurrence (same
    operation order, FP32), loading S = M * s once and storing (M, s) once after the row's tokens."""
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    Tn = eos - bos
    slot = tl.load(state_idx + i_n).to(tl.int64)
    if (slot <= 0) | (Tn <= 0):
        return
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]
    p_h = h0 + slot * stride_state + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    s0 = tl.load(sc + slot * stride_sc + i_hv * (V // BLK) + o_v // BLK)                # per row (block scale gathered)
    p_sb = sc + slot * stride_sc + i_hv * (V // BLK) + i_v * (BV // BLK) + tl.arange(0, BV // BLK)
    s0b = tl.load(p_sb)
    b_h = tl.load(p_h, mask=mask_h, other=0).to(tl.float32) * s0[:, None]
    A_log_v = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_v = tl.load(dt_bias + i_hv).to(tl.float32)
    dec = A_log_v * 0.0 + 1.0
    for i_t in range(0, Tn):
        t = bos + i_t
        b_q = tl.load(q + (t * H + i_h) * K + o_k, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(k + (t * H + i_h) * K + o_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(v + (t * HV + i_hv) * V + o_v, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(b + t * HV + i_hv).to(tl.float32)
        x = tl.load(a + t * HV + i_hv).to(tl.float32) + dt_bias_v
        softplus_x = tl.where(beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x)
        b_g = -tl.exp(A_log_v) * softplus_x
        b_beta = tl.sigmoid(b_b)
        if USE_QK_L2NORM:
            b_q = b_q * (tl.rsqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k * (tl.rsqrt(tl.sum(b_k * b_k) + 1e-6))
        b_q = b_q * scale
        b_h *= tl.exp(b_g)
        dec = dec * tl.exp(b_g)
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_v *= b_beta
        b_h += b_v[:, None] * b_k[None, :]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(o + (t * HV + i_hv) * V + o_v, b_o.to(o.dtype.element_ty), mask=mask_v)
    M1, s1 = s16_rescale(b_h, s0b * dec, dec, BV, BLK, LO, HI)
    tl.store(p_h, M1.to(p_h.dtype.element_ty), mask=mask_h)
    tl.store(p_sb, s1)


def gdn_s16_decode(A_log, a, b, dt_bias, q, k, v, initial_state, scale_state, cu_seqlens, ssm_state_indices, scale=None, beta=1.0,
                   threshold=20.0, use_qk_l2norm_in_kernel=True, inplace_final_state=True, **unused):
    """Drop-in for the non-speculative fused_sigmoid_gating_delta_rule_update calls of qwen_gdn_linear_attn (1-D state indices, state
    updated in place) on a scaled16 state. Returns (o [1, T, HV, V], initial_state)."""
    assert inplace_final_state and ssm_state_indices is not None and ssm_state_indices.ndim == 1
    assert initial_state.dtype == torch.float16 and scale_state.dtype == torch.float32
    B, T, H, K = k.shape
    HV, V = v.shape[2], v.shape[3]
    assert scale_state.shape[1:] == (HV, V // BLK) and scale_state.stride(1) == V // BLK and scale_state.stride(2) == 1
    N = cu_seqlens.numel() - 1
    if scale is None:
        scale = K ** -0.5
    BK, BV = triton.next_power_of_2(K), min(triton.next_power_of_2(V), 32)
    assert BK == K and V % BV == 0 and BV % BLK == 0
    o = q.new_empty(1, T, HV, V)
    (_gdn_s16_decode_kernel_sr if SR else _gdn_s16_decode_kernel)[(1, V // BV, N * HV)](
        A_log, a.contiguous(), b.contiguous(), dt_bias, beta, threshold, q.contiguous(), k.contiguous(), v.contiguous(), o,
        initial_state, scale_state, cu_seqlens, ssm_state_indices, scale, N, H=H, HV=HV, K=K, V=V, BK=BK, BV=BV,
        stride_state=initial_state.stride(0), stride_sc=scale_state.stride(0), LO=LO, HI=HI, USE_QK_L2NORM=use_qk_l2norm_in_kernel,
        BLK=BLK, num_warps=4, num_stages=3)
    STATS['decode_calls'] += 1
    return o, initial_state


# ---- scaled16sr (2026-10-27, Claude fork bdfc5402; the user: "Let's choose scaled16 + stochastic rounding" after IE's note and the
# emulation gdn_emul_i16.py's PS16S arm). Appended at the end of the file on purpose: Triton 3.7's kernel cache key includes each JIT
# function's starting line number, so every kernel above keeps its line, source and compile cache (the scaled16 tiers stay warm).

@triton.jit
def s16_round(M, idx):
    """The FP16 store of M = S / s with stochastic rounding onto the FP16 grid (scaled16sr). The random bits are a hash of the element's
    FP32 bits and its index in the slot's state (idx = (head * V + row) * K + col; murmur3's finalizer), so a store is deterministic (no
    position, seed or RNG state) and different elements, or one element stored with different values, draw independent bits. Adding
    r < 2^13 to the magnitude bits and truncating (round toward zero) rounds an FP16-normal result (|M| >= 2^-14) up with probability
    (the 13 dropped bits) / 2^13, exactly; a carry into the next binade is that binade's correct value. Below 2^-14 (FP16 subnormals:
    under 2^-15 of the block maximum after the renormalisation) the store rounds to nearest, as scaled16 does: there the FP16 step is
    2^14..2^23 FP32 steps, which 13 bits do not cover (truncation would ratchet those values toward zero: the emulation's layer 44,
    10-27), and an exact law there needs per-element exponent math and a carry clamp (+13.8 vs +3.7 us per layer at 28 lanes, bench_sr).
    NaN stays NaN; |M| <= 2, so no overflow. The bits repeat when M's pre-round FP32 value repeats exactly, e.g. a frozen state fed the
    same token every step (test_s16_sr's fixed-input case); served decoding feeds a new token each step (the emulation's PS16H arm on
    the recorded sessions matches ideal draws)."""
    xb = M.to(tl.int32, bitcast=True)
    h = (xb ^ (idx * -1640531535)).to(tl.uint32, bitcast=True)                  # 0x9E3779B1
    h = h ^ (h >> 16)
    h = h * tl.full([], 2246822507, tl.uint32)                                   # 0x85EBCA6B
    h = h ^ (h >> 13)
    h = h * tl.full([], 3266489909, tl.uint32)                                   # 0xC2B2AE35
    h = h ^ (h >> 16)
    xm = xb & 0x7FFFFFFF
    y = ((xb & -2147483648) | (xm + (h & 0x1FFF).to(tl.int32, bitcast=True))).to(tl.float32, bitcast=True)
    return tl.where(xm < 0x38800000, M.to(tl.float16), y.to(tl.float16, fp_downcast_rounding='rtz'))   # below 2^-14: to nearest


@triton.jit
def _s16_prefill_store_kernel_sr(S, Mout, G, cu, LD, LD_OUT, scale, idx, has_init, stride_sc, HV: tl.constexpr, V: tl.constexpr,
                              K: tl.constexpr, BV: tl.constexpr, BLK: tl.constexpr, LO: tl.constexpr, HI: tl.constexpr,
                              D_MIN: tl.constexpr, S_MIN: tl.constexpr, E_MAX: tl.constexpr, DBL_MAX: tl.constexpr,
                              BT: tl.constexpr, USE_LD: tl.constexpr, WRITE_LD: tl.constexpr):
    """_s16_prefill_store_kernel with the scaled16sr store (s16_round); otherwise its exact source (kept apart: the original's compile cache stays).
    """
    """store_rule, the scale write and the stock FP16 rounding in one pass, operation for operation: Mout = (S / s).to(FP16),
    scale[slot] = s. The summed log gate (v2): the FP64 sum of the sequence's own gates G[bos:eos, head] (FP32 [T, HV]), BT tokens per
    step into a [BT] FP64 accumulator, then one tree sum: a fixed order. USE_LD (tests): take it from LD [n, HV] (FP64) instead, which
    makes the kernel bit-identical to the torch path; WRITE_LD (tests): write it to LD_OUT. One program per (V block, sequence * HV)."""
    NB: tl.constexpr = BV // BLK
    i_v, i_nh = tl.program_id(0), tl.program_id(1)
    i_n, i_hv = i_nh // HV, i_nh % HV
    if USE_LD:
        ld = tl.load(LD + i_n * HV + i_hv)
    else:
        bos = tl.load(cu + i_n).to(tl.int64)
        eos = tl.load(cu + i_n + 1).to(tl.int64)
        o_t = tl.arange(0, BT)
        acc = tl.zeros([BT], dtype=tl.float64)
        for t0 in range(bos, eos, BT):
            acc += tl.load(G + (t0 + o_t) * HV + i_hv, mask=(t0 + o_t) < eos, other=0.0).to(tl.float64)
        ld = tl.sum(acc, axis=0)
    if WRITE_LD:
        tl.store(LD_OUT + i_n * HV + i_hv, ld, mask=i_v == 0)
    dec = _exp64(ld)                                                              # FP64, as torch.exp(log_decay.double())
    o_v = i_v * BV + tl.arange(0, BV)
    o_k = tl.arange(0, K)
    off = i_nh.to(tl.int64) * (V * K) + o_v[:, None] * K + o_k[None, :]
    S1 = tl.load(S + off)                                                         # [BV, K] FP32
    amax = tl.max(tl.reshape(tl.max(tl.abs(S1), axis=1), [NB, BLK]), axis=1).to(tl.float64)
    nan = tl.max(tl.reshape(tl.max((S1 != S1).to(tl.int32), axis=1), [NB, BLK]), axis=1)
    amax = tl.where(nan > 0, tl.full([NB], 0x7FF8000000000000, tl.int64).to(tl.float64, bitcast=True), amax)   # torch's amax: NaN
    slot = tl.load(idx + i_n).to(tl.int64)
    p_s = scale + slot * stride_sc + i_hv * (V // BLK) + i_v * NB + tl.arange(0, NB)
    s = tl.load(p_s).to(tl.float64) * dec
    hi = tl.load(has_init + i_n).to(tl.int32)
    fresh = ((hi == 0) | (dec < tl.full([], D_MIN, tl.float64)) | (s != s) | (s > tl.full([], DBL_MAX, tl.float64)) | (s <= 0)
             | (s < tl.full([], S_MIN, tl.float64)))
    s = tl.where(fresh, tl.where(amax > 0, amax, 1.0), s)
    ratio = amax / s                                                              # FP64 (div.rn.f64)
    out = (ratio > HI) | ((ratio < LO) & (amax > 0))
    e = tl.where(out, tl.floor(_log2_64(tl.where(out, ratio, 1.0)) + 0.5), 0.0)
    e = tl.minimum(tl.maximum(e, -E_MAX), E_MAX)
    s32 = (s * ((e.to(tl.int64) + 1023) << 52).to(tl.float64, bitcast=True)).to(tl.float32)     # exactly 2^e (torch.exp2)
    s_rows = tl.reshape(tl.broadcast_to(s32[:, None], [NB, BLK]), [BV])
    M = tl.math.div_rn(S1, tl.broadcast_to(s_rows[:, None], [BV, K]))            # IEEE FP32 division, as torch
    tl.store(Mout + off, s16_round(M, i_hv * V * K + o_v[:, None] * K + o_k[None, :]))                          # scaled16sr: stochastic rounding
    tl.store(p_s, s32)


@triton.jit(do_not_specialize=["N"])
def _gdn_s16_decode_kernel_sr(A_log, a, b, dt_bias, beta, threshold, q, k, v, o, h0, sc, cu_seqlens, state_idx, scale, N: tl.int64,
                           H: tl.constexpr, HV: tl.constexpr, K: tl.constexpr, V: tl.constexpr, BK: tl.constexpr,
                           BV: tl.constexpr, stride_state: tl.constexpr, stride_sc: tl.constexpr, LO: tl.constexpr,
                           HI: tl.constexpr, USE_QK_L2NORM: tl.constexpr, BLK: tl.constexpr):
    """_gdn_s16_decode_kernel with the scaled16sr store (s16_round); otherwise its exact source (kept apart: the original's compile cache stays).
    """
    """Non-speculative rows (0-draft decode) on a scaled16 state: the stock fused_sigmoid_gating_delta_rule_update recurrence (same
    operation order, FP32), loading S = M * s once and storing (M, s) once after the row's tokens."""
    i_k, i_v, i_nh = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    i_n, i_hv = i_nh // HV, i_nh % HV
    i_h = i_hv // (HV // H)
    bos, eos = tl.load(cu_seqlens + i_n).to(tl.int64), tl.load(cu_seqlens + i_n + 1).to(tl.int64)
    Tn = eos - bos
    slot = tl.load(state_idx + i_n).to(tl.int64)
    if (slot <= 0) | (Tn <= 0):
        return
    o_k = i_k * BK + tl.arange(0, BK)
    o_v = i_v * BV + tl.arange(0, BV)
    mask_k = o_k < K
    mask_v = o_v < V
    mask_h = mask_v[:, None] & mask_k[None, :]
    p_h = h0 + slot * stride_state + i_hv * V * K + o_v[:, None] * K + o_k[None, :]
    s0 = tl.load(sc + slot * stride_sc + i_hv * (V // BLK) + o_v // BLK)                # per row (block scale gathered)
    p_sb = sc + slot * stride_sc + i_hv * (V // BLK) + i_v * (BV // BLK) + tl.arange(0, BV // BLK)
    s0b = tl.load(p_sb)
    b_h = tl.load(p_h, mask=mask_h, other=0).to(tl.float32) * s0[:, None]
    A_log_v = tl.load(A_log + i_hv).to(tl.float32)
    dt_bias_v = tl.load(dt_bias + i_hv).to(tl.float32)
    dec = A_log_v * 0.0 + 1.0
    for i_t in range(0, Tn):
        t = bos + i_t
        b_q = tl.load(q + (t * H + i_h) * K + o_k, mask=mask_k, other=0).to(tl.float32)
        b_k = tl.load(k + (t * H + i_h) * K + o_k, mask=mask_k, other=0).to(tl.float32)
        b_v = tl.load(v + (t * HV + i_hv) * V + o_v, mask=mask_v, other=0).to(tl.float32)
        b_b = tl.load(b + t * HV + i_hv).to(tl.float32)
        x = tl.load(a + t * HV + i_hv).to(tl.float32) + dt_bias_v
        softplus_x = tl.where(beta * x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta * x)), x)
        b_g = -tl.exp(A_log_v) * softplus_x
        b_beta = tl.sigmoid(b_b)
        if USE_QK_L2NORM:
            b_q = b_q * (tl.rsqrt(tl.sum(b_q * b_q) + 1e-6))
            b_k = b_k * (tl.rsqrt(tl.sum(b_k * b_k) + 1e-6))
        b_q = b_q * scale
        b_h *= tl.exp(b_g)
        dec = dec * tl.exp(b_g)
        b_v -= tl.sum(b_h * b_k[None, :], 1)
        b_v *= b_beta
        b_h += b_v[:, None] * b_k[None, :]
        b_o = tl.sum(b_h * b_q[None, :], 1)
        tl.store(o + (t * HV + i_hv) * V + o_v, b_o.to(o.dtype.element_ty), mask=mask_v)
    M1, s1 = s16_rescale(b_h, s0b * dec, dec, BV, BLK, LO, HI)
    tl.store(p_h, s16_round(M1, i_hv * V * K + o_v[:, None] * K + o_k[None, :]), mask=mask_h)
    tl.store(p_sb, s1)


assert not SR or PREFILL == 'triton', 'scaled16sr needs ARC3_S2_S16_PREFILL=triton (the torch prefill store rounds to nearest)'
