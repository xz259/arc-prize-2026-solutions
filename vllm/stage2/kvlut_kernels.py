"""Kernels of the codebook 4-bit main-QSA KV format (milestone 2, 2026-10-02; kv_screen2's winner h_lutm+c, user 03:58 /
04:40: implement a better 4-bit format if the screen finds one, efficient from the start).

Layout: unchanged NVFP4 bytes per 256-element K or V head side (128 packed 4-bit codes, 16 E4M3 block scales, one FP32
head scale; 148 bytes; 296-byte rows), so eviction, the pool size and the allocator are unchanged. Only the code's
meaning changes: code c (0-15) stands for the level L(s), s = c - 7.5, of an odd quintic fitted to the 16-level
Lloyd-Max codebook of blocks of 16 Gaussian values normalized by the block maximum (kvfq.LUT16; the fitted levels LUTP
differ from it by at most 5.7e-4, simulated MSE x1.0002). The polynomial replaces a table: with s' = s / 2 built as the
float 8 + c / 2 minus 11.75 (one shift-or with the nibble extraction, one add), L = s' (A + s'^2 (B + C s'^2)) is two FMAs
and two multiplies, cheaper than the E2M1 bit decode. (A select-tree table decode cost the reader 6-9% at equal
registers; test_kvlut_diag / kvlut_exp 2026-10-02 07:00.)

store   per head side: x - mu (CENTER: the per-layer, per-KV-head mean; K means are zero on the 64 rotary dims), then the
        normalized 256-point Hadamard (ROTATE), head scale max / 448 (exact division), per block of 16 the E4M3 scale
        code among c-1 .. c+2 around the round-to-nearest code of max|block| / head scale (block maxima normalized to 1,
        as kvfq's _blocks(x, 1.0, ...)) by the block's squared error (strictly smaller wins, in that order), codes by
        nearest level (ties to the lower magnitude, as
        torch.bucketize); UNBIAS rescales the head scale by |x|^2 / <x^, x> (kvfq '+u'). This is kvfq's h_lutp(+c)(+u)
        with exact divisions and RNE E4M3.
attend  the production sparse split-K reader (arc3_qsa_nvfp4_attention) with the polynomial decode and VMEAN: the
        rotated V mean R mu is added to every normalized partial (the merge of split partials is a convex combination,
        so it reaches the result; rows without valid keys stay 0). The query and output rotations stay outside, as in
        kvrot (kvrot.rotate_rows; R is an involution, so rotating the output back turns R mu into mu). Fusing them into
        this reader raised its registers from 168 to 255 and cost 24-33%; capped at 168 with tf32x3 dots, 6-13%.
E2M1 mode (lut=False, no means) is the production reader exactly (test_kvlut bitwise check of the copy).

scale16 (2026-10-02 19:05; job 75 ablation, bench_qsadiag sweep): the reader loads each token side's 16 E4M3 block scales
once ([16, BLOCK_N]), multiplies them by the head scale and broadcasts the product to [256, BLOCK_N] in registers,
instead of gathering [256, BLOCK_N] scale bytes (each loaded 16 times). Same FP32 products per element: bitwise equal to
the gather at the same launch configuration. Launch rule (attention): warps 4 at <= 64 base programs (rows x KV
heads), else 2; splits 16 at <= 256, 8 at <= 512, 4 at <= 2048, else 1 (production's profile: warps 4; splits 16 / 4 / 1
at <= 256 / 512 / more). 1.26x at 228 decode rows, 1.29-1.40x at 128-512, 1.31x on 4,096-row prefill.
profile='production' (or ARC3_S2_KVLUT_PROFILE=production) restores production's launch, for identity tests."""
import os

import torch
import triton
import triton.language as tl

_PROFILE = os.environ.get('ARC3_S2_KVLUT_PROFILE', 'scale16')
LUT16 = (0.05286, 0.15959, 0.26949, 0.38465, 0.50827, 0.64419, 0.79859, 0.9814)
POLY = (0.21203680336475372, 0.0022320810239762068, 9.21660684980452e-05)       # A, B, C in s' = (c - 7.5) / 2
LUTP = (0.053044166415929794, 0.15999113023281097, 0.26968681812286377, 0.3845396935939789, 0.5078223943710327,
        0.6440170407295227, 0.7991613745689392, 0.9811936616897583)              # L for c = 8 .. 15 (FP32)
BOUNDS = (0.10651765018701553, 0.21483898162841797, 0.3271132707595825, 0.446181058883667, 0.5759197473526001,
          0.721589207649231, 0.8901774883270264)                                 # midpoints of LUTP (FP32)
_PA = tl.constexpr(POLY[0])
_PB = tl.constexpr(POLY[1])
_PC = tl.constexpr(POLY[2])


@triton.jit
def _h16():
    i = tl.arange(0, 16)[:, None]
    j = tl.arange(0, 16)[None, :]
    b = i & j
    parity = (b ^ (b >> 1) ^ (b >> 2) ^ (b >> 3)) & 1
    return tl.where(parity == 0, 1.0, -1.0)


@triton.jit
def _poly_from_bits(bits):
    """bits: int32 0x41000000 | c << 19 (the float 8 + c / 2); returns L(c - 7.5)."""
    sp = bits.to(tl.float32, bitcast=True) - 11.75
    s2 = sp * sp
    return sp * tl.fma(s2, tl.fma(s2, _PC, _PB), _PA)


@triton.jit
def _poly_level(c):
    return _poly_from_bits((c.to(tl.int32) << 19) | 0x41000000)


@triton.jit
def _decode_e2m1(nibble):
    # As arc3_qsa_nvfp4._decode_e2m1 (copied, so the E2M1 mode of the reader needs no production import here).
    bits = nibble.to(tl.uint32)
    exponent = (bits >> 1) & 3
    mantissa = bits & 1
    normal = ((exponent + 126) << 23) | (mantissa << 22)
    subnormal = tl.where(mantissa != 0, 126 << 23, 0).to(tl.uint32)
    value = tl.where(exponent == 0, subnormal, normal) | ((bits & 8) << 28)
    return value.to(tl.float32, bitcast=True)


@triton.jit
def _poly_encode(n):
    """Code of the nearest level (ties to the lower magnitude); n FP32 normalized (block max near LUTP[7])."""
    a = tl.abs(n)
    idx = ((a > 0.10651765018701553).to(tl.int32) + (a > 0.21483898162841797).to(tl.int32)
           + (a > 0.3271132707595825).to(tl.int32) + (a > 0.446181058883667).to(tl.int32)
           + (a > 0.5759197473526001).to(tl.int32) + (a > 0.721589207649231).to(tl.int32)
           + (a > 0.8901774883270264).to(tl.int32))
    sign = (n.to(tl.uint32, bitcast=True) >> 31).to(tl.int32)
    return (8 + idx) ^ (sign * 15)


@triton.jit
def _e4m3_code(x):
    zero = tl.zeros_like(x)
    return tl.inline_asm_elementwise(
        asm='cvt.rn.satfinite.e4m3x2.f32 $0, $2, $1;',
        constraints='=h,f,f', args=[x, zero], dtype=tl.uint16, is_pure=True, pack=1).to(tl.uint8)


@triton.jit
def _store_lut_kernel(key, value, cache, slots, kmean, vmean,
                      stride_kr, stride_kh, stride_vr, stride_vh, stride_cb, stride_ch, stride_ct,
                      num_blocks, PAGE: tl.constexpr, ROTATE: tl.constexpr, NCAND: tl.constexpr,
                      CENTER: tl.constexpr, UNBIAS: tl.constexpr):
    row = tl.program_id(0)
    head = tl.program_id(1)
    side = tl.program_id(2)
    slot = tl.load(slots + row)
    if slot >= 0 and slot < num_blocks * PAGE:
        idx = tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]
        if side == 0:
            Y = tl.load(key + row * stride_kr + head * stride_kh + idx).to(tl.float32)
            if CENTER:
                Y = Y - tl.load(kmean + head * 256 + idx)
        else:
            Y = tl.load(value + row * stride_vr + head * stride_vh + idx).to(tl.float32)
            if CENTER:
                Y = Y - tl.load(vmean + head * 256 + idx)
        if ROTATE:
            H = _h16()
            T = tl.dot(H, Y, input_precision="ieee")
            Y = tl.dot(T, H, input_precision="ieee") * 0.0625
        y0, y1 = tl.split(tl.reshape(Y, [16, 8, 2]))
        maxima = tl.max(tl.maximum(tl.abs(y0), tl.abs(y1)), axis=1)
        maximum = tl.max(maxima, axis=0)
        head_scale = tl.where(maximum == 0., 1., tl.maximum(tl.div_rn(maximum, 448.), 1e-30))
        code = _e4m3_code(tl.minimum(tl.div_rn(maxima, head_scale), 448.))
        best_code = code
        best_err = tl.full([16], float('inf'), tl.float32)
        best_c0 = tl.zeros([16, 8], dtype=tl.int32)
        best_c1 = tl.zeros([16, 8], dtype=tl.int32)
        best_r0 = tl.zeros([16, 8], dtype=tl.float32)
        best_r1 = tl.zeros([16, 8], dtype=tl.float32)
        for i in tl.static_range(NCAND):
            if NCAND > 1:
                c = tl.minimum(tl.maximum(code.to(tl.int32) + (i - 1), 0), 126).to(tl.uint8)
            else:
                c = code
            dq = head_scale * c.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            safe = tl.where(dq > 0., dq, 1.)
            n0 = tl.where(dq[:, None] > 0., tl.div_rn(y0, safe[:, None]), 0.)
            n1 = tl.where(dq[:, None] > 0., tl.div_rn(y1, safe[:, None]), 0.)
            c0 = _poly_encode(n0)
            c1 = _poly_encode(n1)
            r0 = _poly_level(c0) * dq[:, None]
            r1 = _poly_level(c1) * dq[:, None]
            err = tl.sum((r0 - y0) * (r0 - y0) + (r1 - y1) * (r1 - y1), axis=1)
            take = err < best_err
            best_err = tl.where(take, err, best_err)
            best_code = tl.where(take, c, best_code)
            best_c0 = tl.where(take[:, None], c0, best_c0)
            best_c1 = tl.where(take[:, None], c1, best_c1)
            best_r0 = tl.where(take[:, None], r0, best_r0)
            best_r1 = tl.where(take[:, None], r1, best_r1)
        if UNBIAS:
            dot = tl.sum(tl.sum(best_r0 * y0 + best_r1 * y1, axis=1), axis=0)
            energy = tl.sum(tl.sum(y0 * y0 + y1 * y1, axis=1), axis=0)
            head_scale = tl.where(dot > 0., head_scale * tl.div_rn(energy, tl.where(dot > 0., dot, 1.)), head_scale)
        address = cache + (slot // PAGE).to(tl.int64) * stride_cb + head * stride_ch + (slot % PAGE) * stride_ct + side * 148
        pair = tl.arange(0, 16)[:, None] * 8 + tl.arange(0, 8)[None, :]
        tl.store(address + pair, (best_c0 | (best_c1 << 4)).to(tl.uint8))
        tl.store(address + 128 + tl.arange(0, 16), best_code)
        tl.store((address + 144).to(tl.pointer_type(tl.float32)), head_scale)


_CHECKED = {}


def _check_cache(cache):
    key = (cache.data_ptr(), tuple(cache.shape), tuple(cache.stride()))
    if key not in _CHECKED:
        from arc3_qsa_nvfp4 import split_cache
        split_cache(cache)                           # dtype, 296-byte rows, four-byte alignment of the head scales
        _CHECKED[key] = True


def store(key, value, cache, slots, kmean=None, vmean=None, rotate=True, mse=True, unbias=False):
    """Drop-in for arc3_qsa_nvfp4.store_nvfp4 with the codebook format. kmean/vmean: FP32 [H, 256] contiguous or None."""
    if key.dtype != torch.bfloat16 or value.dtype != key.dtype or key.ndim != 3 or key.shape != value.shape or key.shape[-1] != 256:
        raise ValueError('kvlut: QSA writes require 256-dimensional BF16 K/V')
    if (slots.ndim != 1 or slots.dtype not in (torch.int32, torch.int64) or slots.stride(0) != 1
            or len(slots) > len(key) or key.stride(-1) != 1 or value.stride(-1) != 1):
        raise ValueError('kvlut: unsupported QSA input or slot layout')
    _check_cache(cache)
    if cache.shape[1] != key.shape[1]:
        raise ValueError('kvlut: QSA cache head count mismatch')
    if not len(slots):
        return
    center = kmean is not None
    if center and (vmean is None or kmean.shape != (key.shape[1], 256) or vmean.shape != kmean.shape
                   or kmean.dtype != torch.float32 or not kmean.is_contiguous() or not vmean.is_contiguous()):
        raise ValueError('kvlut: means must be contiguous FP32 [heads, 256] for both sides')
    _store_lut_kernel[(len(slots), key.shape[1], 2)](
        key, value, cache, slots, kmean if center else key, vmean if center else key,
        key.stride(0), key.stride(1), value.stride(0), value.stride(1),
        cache.stride(0), cache.stride(1), cache.stride(2), cache.shape[0],
        PAGE=cache.shape[2], ROTATE=rotate, NCAND=4 if mse else 1, CENTER=center, UNBIAS=unbias, num_warps=4)


@triton.jit
def _lut_sparse_splitk_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr, k_scale_ptr, v_scale_ptr, k_block_scale_ptr, v_block_scale_ptr,
    indices_ptr, block_table_ptr, token_to_req_ptr, partial_output_ptr, partial_lse_ptr, output_ptr, vmean_ptr,
    stride_q_row, stride_q_head, stride_k_block, stride_k_token, stride_k_head,
    stride_v_block, stride_v_token, stride_v_head, stride_indices_row, stride_table_req,
    stride_output_row, stride_output_head, num_rows, num_cache_blocks, num_requests,
    TOPK: tl.constexpr, PAGE_SIZE: tl.constexpr, PAGE_TABLE_WIDTH: tl.constexpr, GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr, NUM_QUERY_HEADS: tl.constexpr, NUM_SPLITS: tl.constexpr, NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, LUT: tl.constexpr, VMEAN: tl.constexpr,
) -> None:
    # The production reader (arc3_qsa_nvfp4_attention._nvfp4_sparse_splitk_kernel) with the polynomial decode and the
    # rotated V mean (see the module docstring).
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    block_offsets = tl.arange(0, HEAD_DIM // 16)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    query = tl.load(
        q_ptr + row * stride_q_row + (first_head + head_offsets[:, None]) * stride_q_head + dim_offsets[None, :],
        mask=head_offsets[:, None] < GROUP_SIZE, other=0.0)

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634

    split_tile_start = split_id * NUM_TILES // NUM_SPLITS
    split_tile_end = (split_id + 1) * NUM_TILES // NUM_SPLITS
    for tile in range(split_tile_start, split_tile_end):
        columns = tile * BLOCK_N + column_offsets
        logical_token = tl.load(indices_ptr + row * stride_indices_row + columns, mask=columns < TOPK, other=-1)
        safe_token = tl.maximum(logical_token, 0)
        logical_page = safe_token // PAGE_SIZE
        page_offset = safe_token % PAGE_SIZE
        valid = (request >= 0) & (request < num_requests) & (logical_token >= 0) & (logical_page < PAGE_TABLE_WIDTH)
        physical_page = tl.load(
            block_table_ptr + safe_request * stride_table_req + tl.minimum(logical_page, PAGE_TABLE_WIDTH - 1),
            mask=valid, other=-1)
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)
        keys = tl.load(
            k_cache_ptr + safe_page[None, :] * stride_k_block + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head + (dim_offsets[:, None] // 2), mask=valid[None, :], other=0.0)
        values = tl.load(
            v_cache_ptr + safe_page[:, None] * stride_v_block + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head + (dim_offsets[None, :] // 2), mask=valid[:, None], other=0.0)
        k_scale = tl.load(
            k_scale_ptr + safe_page * (stride_k_block // 4) + page_offset * (stride_k_token // 4)
            + kv_head * (stride_k_head // 4), mask=valid, other=1.0)
        v_scale = tl.load(
            v_scale_ptr + safe_page * (stride_v_block // 4) + page_offset * (stride_v_token // 4)
            + kv_head * (stride_v_head // 4), mask=valid, other=1.0)
        # scale16: each block scale loaded once, times the head scale, broadcast over its 16 dims (bitwise equal to
        # gathering [HEAD_DIM, BLOCK_N] scale bytes)
        k_blocks = tl.load(
            k_block_scale_ptr + safe_page[None, :] * stride_k_block + page_offset[None, :] * stride_k_token
            + kv_head * stride_k_head + block_offsets[:, None], mask=valid[None, :], other=0.0).to(tl.float32)
        v_blocks = tl.load(
            v_block_scale_ptr + safe_page[:, None] * stride_v_block + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head + block_offsets[None, :], mask=valid[:, None], other=0.0).to(tl.float32)
        k_mult = tl.reshape(tl.broadcast_to((k_blocks * k_scale[None, :])[:, None, :], (HEAD_DIM // 16, 16, BLOCK_N)),
                            (HEAD_DIM, BLOCK_N))
        v_mult = tl.reshape(tl.broadcast_to((v_blocks * v_scale[:, None])[:, :, None], (BLOCK_N, HEAD_DIM // 16, 16)),
                            (BLOCK_N, HEAD_DIM))
        if LUT:
            # the code's nibble straight into the mantissa of 8 + c / 2 (even dims: low nibble, odd: high nibble)
            k_bits = ((keys.to(tl.int32) << (19 - (dim_offsets[:, None] % 2) * 4)) & 0x780000) | 0x41000000
            v_bits = ((values.to(tl.int32) << (19 - (dim_offsets[None, :] % 2) * 4)) & 0x780000) | 0x41000000
            keys = (_poly_from_bits(k_bits) * k_mult).to(tl.bfloat16)
            values = (_poly_from_bits(v_bits) * v_mult).to(tl.bfloat16)
        else:
            k_nibbles = (keys >> ((dim_offsets[:, None] % 2) * 4)) & 15
            v_nibbles = (values >> ((dim_offsets[None, :] % 2) * 4)) & 15
            keys = (_decode_e2m1(k_nibbles) * k_mult).to(tl.bfloat16)
            values = (_decode_e2m1(v_nibbles) * v_mult).to(tl.bfloat16)
        scores = tl.dot(query, keys)
        scores *= softmax_scale_log2
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0)
        accumulator = tl.dot(probabilities.to(values.dtype), values, acc=accumulator * alpha[:, None])
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    normalized_output = tl.where(has_values[:, None], accumulator / tl.maximum(normalizer[:, None], 1.0e-20), 0.0)
    if VMEAN:
        mean = tl.load(vmean_ptr + kv_head * HEAD_DIM + dim_offsets)
        normalized_output = tl.where(has_values[:, None], normalized_output + mean[None, :], 0.0)
    output_mask = head_offsets[:, None] < GROUP_SIZE
    if NUM_SPLITS == 1:
        tl.store(output_ptr + row * stride_output_row + (first_head + head_offsets[:, None]) * stride_output_head
                 + dim_offsets[None, :], normalized_output, mask=output_mask)
    else:
        partial_lse = tl.where(has_values, max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)), -float("inf"))
        tl.store(partial_output_ptr + ((split_id * num_rows + row) * NUM_QUERY_HEADS + first_head
                                       + head_offsets[:, None]) * HEAD_DIM + dim_offsets[None, :],
                 normalized_output, mask=output_mask)
        tl.store(partial_lse_ptr + (split_id * num_rows + row) * NUM_QUERY_HEADS + first_head + head_offsets,
                 partial_lse, mask=head_offsets < GROUP_SIZE)


_VIEWS = {}


def _views(cache):
    key = (cache.data_ptr(), tuple(cache.shape), tuple(cache.stride()))
    v = _VIEWS.get(key)
    if v is None:
        from arc3_qsa_nvfp4 import split_cache
        v = _VIEWS[key] = split_cache(cache)
    return v


def launch_profile(base_programs, block_m, profile=None):
    """(block_n, target splits, warps, stages): production's quantized_sparse_profile, or the scale16 rule."""
    from arc3_qsa_profiles import quantized_sparse_profile
    block_n, target_splits, partial_warps, compiler_stages = quantized_sparse_profile(base_programs, block_m, 'nvfp4')
    if (profile or _PROFILE) == 'production':
        return block_n, target_splits, partial_warps, compiler_stages
    warps = 4 if base_programs <= 64 else 2
    splits = 16 if base_programs <= 256 else 8 if base_programs <= 512 else 4 if base_programs <= 2048 else 1
    return block_n, splits, warps, compiler_stages


def attention(query, cache, indices, block_table, token_to_req, out, vmean=None, lut=True, profile=None):
    """As arc3_qsa_nvfp4_attention.attention_nvfp4, for the codebook format: query already rotated (kvrot.rotate_rows),
    output in the rotated domain (the caller rotates it back); vmean: the rotated V mean, FP32 [kv heads, 256].
    profile: None (ARC3_S2_KVLUT_PROFILE, default the scale16 launch rule) or 'production'."""
    (k_cache, k_blocks, k_scale), (v_cache, v_blocks, v_scale) = _views(cache)
    q = query
    if not q.shape[0]:
        return out
    group_size = q.shape[1] // k_cache.shape[2]
    block_m = triton.next_power_of_2(group_size)
    base_programs = q.shape[0] * k_cache.shape[2]
    block_n, target_splits, partial_warps, compiler_stages = launch_profile(base_programs, block_m, profile)
    num_tiles = triton.cdiv(indices.shape[1], block_n)
    num_splits = min(1 << (num_tiles.bit_length() - 1), target_splits)
    if num_splits == 1:
        partial_output = partial_lse = out
    else:
        partial_output = torch.empty((num_splits, *q.shape), dtype=torch.float32, device=q.device)
        partial_lse = torch.empty((num_splits, q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device)
    _lut_sparse_splitk_kernel[(q.shape[0], k_cache.shape[2], num_splits)](
        q, k_cache, v_cache, k_scale, v_scale, k_blocks, v_blocks, indices, block_table, token_to_req,
        partial_output, partial_lse, out, vmean if vmean is not None else q,
        q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), indices.stride(0), block_table.stride(0),
        out.stride(0), out.stride(1), q.shape[0], k_cache.shape[0], block_table.shape[0],
        TOPK=indices.shape[1], PAGE_SIZE=k_cache.shape[1], PAGE_TABLE_WIDTH=block_table.shape[1],
        GROUP_SIZE=group_size, HEAD_DIM=q.shape[2], NUM_QUERY_HEADS=q.shape[1], NUM_SPLITS=num_splits,
        NUM_TILES=num_tiles, BLOCK_M=block_m, BLOCK_N=block_n, LUT=lut, VMEAN=vmean is not None,
        num_warps=partial_warps, num_stages=compiler_stages)
    if num_splits == 1:
        return out
    from vllm.models.qwen3_8_flash_next.nvidia.ops.qsa import _qsa_merge_splitk_kernel
    _qsa_merge_splitk_kernel[(q.shape[0], q.shape[1])](
        partial_output, partial_lse, out, out.stride(0), out.stride(1), q.shape[0],
        HEAD_DIM=q.shape[2], NUM_QUERY_HEADS=q.shape[1], NUM_SPLITS=num_splits,
        BLOCK_SPLITS=triton.next_power_of_2(num_splits), num_warps=2, num_stages=1)
    return out


@triton.jit
def _levels_probe(dst):
    c = tl.arange(0, 16)
    tl.store(dst + c, _poly_level(c))


def levels(device='cuda'):
    """The 16 decoded levels L(c - 7.5), c = 0..15, as the kernels compute them (tests)."""
    out = torch.empty(16, dtype=torch.float32, device=device)
    _levels_probe[(1,)](out)
    return out
