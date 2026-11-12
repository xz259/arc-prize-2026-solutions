"""kvring_fast.py (2026-10-27, Claude fork bdfc5402; the user: "Yes, build the two-kernel reader"): kvring's two-source reader with the
lutc pass sized to the rows it reads (ARC3_S2_KVRING_FAST=1, kvring.py); the same inputs and result as kvring_kernels.attention up to
the summation order.

kvring_kernels.attention (the three-pass reader) runs kvlut's reader over each query row's full index list with the window entries set
to -1: their loads are masked, but every tile is still dequantized and multiplied, so the lutc pass costs as much as without a ring;
then a ring pass with one program per (query row, KV head) loops over the compacted window rows in 64-row tiles without pipelining
(bench_parts.py, 10-27: at 1,632 rows and ~540 window rows per query row, 300 + 80 us per layer against 323 us for kvlut alone).
Here:
  pack   _fast_pack_kernel: the three-pass reader's compaction (window test, tag check, fallback counters, compact list) plus each query
         row's packed lutc list: its entries >= 0 not read from the ring, in list order, then -1 (lut_count of them);
  lutc   _lut_packed_kernel: kvlut's reader (kvring_kernels._lut_lse_kernel, the same loop body and epilogue) over the packed list's
         ceil(lut_count / BLOCK_N) tiles, split as production splits its tile range; production's split-K merge;
  ring   kvring_kernels._ring_merge_kernel itself (attention over the compacted window rows from the BF16 ring, merged in FP32 with the
         lutc part) in 32-row tiles with two pipeline stages (bench_parts.py's best of 64 / 32 rows x 1 / 2 stages).
A row whose list holds no window entry and no -1 entry reads kvlut's tiles in kvlut's order and is not touched by the ring pass: bitwise
kvlut's result and the three-pass reader's (tested). The one-pass prototype (kvring_fused.py, 10-27) was slower at small window shares
and in prefill (its lutc loop ran 15-20% slower beside the ring loop) and is not used.
"""
import os

import torch
import triton
import triton.language as tl

from .kvlut_kernels import _poly_from_bits, _decode_e2m1, launch_profile, _views
from .kvring_kernels import _ring_merge_kernel

RING_BLOCK_N, RING_STAGES, RING_WARPS = (int(x) for x in os.environ.get('ARC3_S2_KVRING_FAST_RING', '32,2,4').split(','))


@triton.jit
def _fast_pack_kernel(indices_ptr, lut_ptr, lut_count_ptr, compact_ptr, count_ptr, token_to_req_ptr, seq_lens_ptr, ring_meta_ptr,
                      slot_rows_ptr, stats_ptr, stride_indices_row, num_requests, ring_rows,
                      TOPK: tl.constexpr, BLOCK: tl.constexpr, CAP: tl.constexpr, N: tl.constexpr, R: tl.constexpr,
                      TAGS: tl.constexpr, COUNT: tl.constexpr):
    # kvring_kernels._ring_compact_kernel's window test, tag check, counters and compaction (its alias-event debug records left
    # out), then the packed lutc list: the row's entries >= 0 not read from the ring, in list order, then -1 (lut_count of them)
    row = tl.program_id(0)
    request = tl.load(token_to_req_ptr + row)
    request_ok = (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    seq_len = tl.load(seq_lens_ptr + safe_request)
    ring_id = tl.load(ring_meta_ptr + safe_request * 3)
    off = tl.load(ring_meta_ptr + safe_request * 3 + 1)
    boundary = tl.maximum(seq_len - 1 - N, tl.load(ring_meta_ptr + safe_request * 3 + 2))
    columns = tl.arange(0, BLOCK)
    token = tl.load(indices_ptr + row * stride_indices_row + columns, mask=columns < TOPK, other=-1)
    candidate = request_ok & (ring_id >= 0) & (token > boundary) & (token >= 0) & (token < seq_len)
    if TAGS:
        slot = ring_id * R + (tl.maximum(token, 0) + off) % R
        tag = tl.load(slot_rows_ptr + slot, mask=candidate & (slot >= 0) & (slot < ring_rows), other=-2)
        valid = candidate & (tag == token + off)
        if COUNT:
            cleared = tl.sum((candidate & (tag == -1)).to(tl.int64), axis=0)
            aliased = tl.sum((candidate & (tag != -1) & (tag != token + off)).to(tl.int64), axis=0)
            if cleared > 0:
                tl.atomic_add(stats_ptr + 1, cleared)
            if aliased > 0:
                tl.atomic_add(stats_ptr + 2, aliased)
    else:
        valid = candidate
    flags = valid.to(tl.int32)
    position = tl.cumsum(flags, axis=0) - 1
    tl.store(compact_ptr + row * CAP + position, token, mask=valid & (position < CAP))
    tl.store(count_ptr + row, tl.minimum(tl.sum(flags, axis=0), CAP))
    keep = (columns < TOPK) & (token >= 0) & (flags == 0)
    kept = keep.to(tl.int32)
    kpos = tl.cumsum(kept, axis=0) - 1
    nkept = tl.sum(kept, axis=0)
    tl.store(lut_ptr + row * TOPK + kpos, token, mask=keep)
    tl.store(lut_ptr + row * TOPK + columns, tl.full([BLOCK], -1, tl.int32), mask=(columns >= nkept) & (columns < TOPK))
    tl.store(lut_count_ptr + row, nkept)


@triton.jit
def _lut_packed_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr, k_scale_ptr, v_scale_ptr, k_block_scale_ptr, v_block_scale_ptr,
    lut_ptr, lut_count_ptr, block_table_ptr, token_to_req_ptr, partial_output_ptr, partial_lse_ptr, output_ptr, vmean_ptr,
    stride_q_row, stride_q_head, stride_k_block, stride_k_token, stride_k_head,
    stride_v_block, stride_v_token, stride_v_head, stride_table_req,
    stride_output_row, stride_output_head, num_rows, num_cache_blocks, num_requests,
    TOPK: tl.constexpr, PAGE_SIZE: tl.constexpr, PAGE_TABLE_WIDTH: tl.constexpr, GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr, NUM_QUERY_HEADS: tl.constexpr, NUM_SPLITS: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, LUT: tl.constexpr, VMEAN: tl.constexpr,
) -> None:
    # kvring_kernels._lut_lse_kernel (kvlut's reader + the one-split lse store) over the packed list: ceil(lut_count / BLOCK_N) tiles
    # instead of NUM_TILES, split the same way
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

    lut_tiles = tl.cdiv(tl.load(lut_count_ptr + row), BLOCK_N)
    split_tile_start = split_id * lut_tiles // NUM_SPLITS
    split_tile_end = (split_id + 1) * lut_tiles // NUM_SPLITS
    for tile in range(split_tile_start, split_tile_end):
        columns = tile * BLOCK_N + column_offsets
        logical_token = tl.load(lut_ptr + row * TOPK + columns, mask=columns < TOPK, other=-1)
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
        # kvring: the lse of the direct result too (split 0), for the ring merge
        tl.store(partial_lse_ptr + row * NUM_QUERY_HEADS + first_head + head_offsets,
                 tl.where(has_values, max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)), -float("inf")),
                 mask=head_offsets < GROUP_SIZE)
    else:
        partial_lse = tl.where(has_values, max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)), -float("inf"))
        tl.store(partial_output_ptr + ((split_id * num_rows + row) * NUM_QUERY_HEADS + first_head
                                       + head_offsets[:, None]) * HEAD_DIM + dim_offsets[None, :],
                 normalized_output, mask=output_mask)
        tl.store(partial_lse_ptr + (split_id * num_rows + row) * NUM_QUERY_HEADS + first_head + head_offsets,
                 partial_lse, mask=head_offsets < GROUP_SIZE)


def attention(query, cache, indices, block_table, token_to_req, seq_lens, ring_k, ring_v, ring_meta, out, vmean=None, n=256,
              profile=None, slot_rows=None, stats=None, r=None):
    """kvring_kernels.attention's signature and result (no debug shadow): query already rotated, output in the rotated domain;
    slot_rows: the ring's tags; stats: int64 [4] counters ([1] / [2] window rows read from lutc because their slot's tag was -1 /
    another row's)."""
    (k_cache, k_blocks, k_scale), (v_cache, v_blocks, v_scale) = _views(cache)
    q = query
    if not q.shape[0]:
        return out
    group_size = q.shape[1] // k_cache.shape[2]
    block_m = triton.next_power_of_2(group_size)
    base_programs = q.shape[0] * k_cache.shape[2]
    block_n, target_splits, partial_warps, compiler_stages = launch_profile(base_programs, block_m, profile)
    topk = indices.shape[1]
    num_tiles = triton.cdiv(topk, block_n)
    lut_splits = min(1 << (num_tiles.bit_length() - 1), target_splits)            # production's own choice
    if lut_splits == 1:
        partial_output = out                                                        # unused by the one-split epilogue
        partial_lse = torch.empty((1, q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device)
    else:
        partial_output = torch.empty((lut_splits, *q.shape), dtype=torch.float32, device=q.device)
        partial_lse = torch.empty((lut_splits, q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device)
    vm = vmean if vmean is not None else q
    lut = torch.empty((q.shape[0], topk), dtype=torch.int32, device=q.device)
    lut_count = torch.empty((q.shape[0],), dtype=torch.int32, device=q.device)
    compact = torch.empty((q.shape[0], n), dtype=torch.int32, device=q.device)
    count = torch.empty((q.shape[0],), dtype=torch.int32, device=q.device)
    tags = slot_rows is not None
    rr = r if r is not None else n
    _fast_pack_kernel[(q.shape[0],)](
        indices, lut, lut_count, compact, count, token_to_req, seq_lens, ring_meta,
        slot_rows if tags else token_to_req, stats if stats is not None else count, indices.stride(0), seq_lens.shape[0],
        ring_k.shape[0], TOPK=topk, BLOCK=triton.next_power_of_2(topk), CAP=n, N=n, R=rr, TAGS=tags,
        COUNT=tags and stats is not None, num_warps=8)
    _lut_packed_kernel[(q.shape[0], k_cache.shape[2], lut_splits)](
        q, k_cache, v_cache, k_scale, v_scale, k_blocks, v_blocks, lut, lut_count, block_table, token_to_req,
        partial_output, partial_lse, out, vm,
        q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), block_table.stride(0),
        out.stride(0), out.stride(1), q.shape[0], k_cache.shape[0], block_table.shape[0],
        TOPK=topk, PAGE_SIZE=k_cache.shape[1], PAGE_TABLE_WIDTH=block_table.shape[1],
        GROUP_SIZE=group_size, HEAD_DIM=q.shape[2], NUM_QUERY_HEADS=q.shape[1], NUM_SPLITS=lut_splits,
        BLOCK_M=block_m, BLOCK_N=block_n, LUT=True, VMEAN=vmean is not None,
        num_warps=partial_warps, num_stages=compiler_stages)
    if lut_splits > 1:
        from vllm.models.qwen3_8_flash_next.nvidia.ops.qsa import _qsa_merge_splitk_kernel
        _qsa_merge_splitk_kernel[(q.shape[0], q.shape[1])](
            partial_output, partial_lse, out, out.stride(0), out.stride(1), q.shape[0],
            HEAD_DIM=q.shape[2], NUM_QUERY_HEADS=q.shape[1], NUM_SPLITS=lut_splits,
            BLOCK_SPLITS=triton.next_power_of_2(lut_splits), num_warps=2, num_stages=1)
    _ring_merge_kernel[(q.shape[0], k_cache.shape[2])](
        q, ring_k, ring_v, compact, count, token_to_req, ring_meta, partial_output, partial_lse, out, vm,
        q.stride(0), q.stride(1), ring_k.stride(0), ring_k.stride(1), out.stride(0), out.stride(1),
        q.shape[0], seq_lens.shape[0], ring_k.shape[0],
        CAP=n, GROUP_SIZE=group_size, HEAD_DIM=q.shape[2], NUM_QUERY_HEADS=q.shape[1], LUT_SPLITS=lut_splits,
        BLOCK_SPLITS=triton.next_power_of_2(max(lut_splits, 2)), BLOCK_M=block_m, BLOCK_N=RING_BLOCK_N,
        VMEAN=vmean is not None, N=n, R=rr, num_warps=RING_WARPS, num_stages=RING_STAGES)
    return out
