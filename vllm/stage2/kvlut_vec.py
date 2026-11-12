"""Load-vectorized codebook QSA reader (milestone 2, 2026-10-02 18:30; job 75's ablation: the per-row reader
kvlut_kernels._lut_sparse_splitk_kernel spends about half its time on load instructions, not bytes: block scales are
gathered as [256, 16] (each E4M3 byte 16 times) and code bytes as [256, 16] (each byte twice), all as scattered byte
loads; removing the scale loads alone saved 23%).

Same computation, token-major loads: the 296-byte cache row is read as 74 int32 words (K side words 0-36, V side 37-73:
32 code words, 4 block-scale words, 1 FP32 head scale per side). Per tile of BLOCK_N tokens: codes [BLOCK_N, 32] words,
block scales [BLOCK_N, 4] words, head scales [BLOCK_N], each loaded once. Nibble k of word w is dimension 8 w + k (little
endian: byte 4 w + k // 2, low nibble even); scale byte j is block j. Codes and per-block multipliers are expanded in
registers to [BLOCK_N, 256] and K is transposed for the QK dot. Every element gets the same FP32 operations as the
per-row reader (level from the mantissa trick, times block scale x head scale, BF16), so the output should be bitwise
equal to kvlut_kernels.attention (tested by bench_kvlutvec.py)."""
import torch
import triton
import triton.language as tl

from .kvlut_kernels import _decode_e2m1, _poly_from_bits


@triton.jit
def _lut_vec_splitk_kernel(
    q_ptr, words_ptr, indices_ptr, block_table_ptr, token_to_req_ptr, partial_output_ptr, partial_lse_ptr, output_ptr,
    vmean_ptr, stride_q_row, stride_q_head, stride_w_block, stride_w_head, stride_w_token, stride_indices_row,
    stride_table_req, stride_output_row, stride_output_head, num_rows, num_cache_blocks, num_requests,
    TOPK: tl.constexpr, PAGE_SIZE: tl.constexpr, PAGE_TABLE_WIDTH: tl.constexpr, GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr, NUM_QUERY_HEADS: tl.constexpr, NUM_SPLITS: tl.constexpr, NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, LUT: tl.constexpr, VMEAN: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    request = tl.load(token_to_req_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    head_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    code_words = tl.arange(0, HEAD_DIM // 8)                       # 32 words of 8 nibbles
    scale_words = tl.arange(0, HEAD_DIM // 64)                     # 4 words of 4 E4M3 block scales
    nibble_shift = tl.arange(0, 8) * 4
    byte_shift = tl.arange(0, 4) * 8
    SIDE: tl.constexpr = HEAD_DIM // 8 + HEAD_DIM // 64 + 1        # 37 words per side
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
        base = words_ptr + safe_page * stride_w_block + kv_head * stride_w_head + page_offset * stride_w_token
        k_codes = tl.load(base[:, None] + code_words[None, :], mask=valid[:, None], other=0)
        v_codes = tl.load(base[:, None] + SIDE + code_words[None, :], mask=valid[:, None], other=0)
        k_sw = tl.load(base[:, None] + HEAD_DIM // 8 + scale_words[None, :], mask=valid[:, None], other=0)
        v_sw = tl.load(base[:, None] + SIDE + HEAD_DIM // 8 + scale_words[None, :], mask=valid[:, None], other=0)
        k_scale = tl.load(base + HEAD_DIM // 8 + HEAD_DIM // 64, mask=valid, other=0).to(tl.float32, bitcast=True)
        v_scale = tl.load(base + SIDE + HEAD_DIM // 8 + HEAD_DIM // 64, mask=valid,
                          other=0).to(tl.float32, bitcast=True)
        k_scale = tl.where(valid, k_scale, 1.0)
        v_scale = tl.where(valid, v_scale, 1.0)
        # per-block multipliers (block scale x head scale, as the per-row reader), expanded to [BLOCK_N, HEAD_DIM]
        k_blk = ((k_sw[:, :, None] >> byte_shift[None, None, :]) & 255).to(tl.uint8)
        v_blk = ((v_sw[:, :, None] >> byte_shift[None, None, :]) & 255).to(tl.uint8)
        k_blk = tl.reshape(k_blk, (BLOCK_N, HEAD_DIM // 16)).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        v_blk = tl.reshape(v_blk, (BLOCK_N, HEAD_DIM // 16)).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        k_mult = tl.reshape(tl.broadcast_to((k_blk * k_scale[:, None])[:, :, None], (BLOCK_N, HEAD_DIM // 16, 16)),
                            (BLOCK_N, HEAD_DIM))
        v_mult = tl.reshape(tl.broadcast_to((v_blk * v_scale[:, None])[:, :, None], (BLOCK_N, HEAD_DIM // 16, 16)),
                            (BLOCK_N, HEAD_DIM))
        k_nib = tl.reshape((k_codes[:, :, None] >> nibble_shift[None, None, :]) & 15, (BLOCK_N, HEAD_DIM))
        v_nib = tl.reshape((v_codes[:, :, None] >> nibble_shift[None, None, :]) & 15, (BLOCK_N, HEAD_DIM))
        if LUT:
            keys_t = (_poly_from_bits((k_nib << 19) | 0x41000000) * k_mult).to(tl.bfloat16)
            values = (_poly_from_bits((v_nib << 19) | 0x41000000) * v_mult).to(tl.bfloat16)
        else:
            keys_t = (_decode_e2m1(k_nib) * k_mult).to(tl.bfloat16)
            values = (_decode_e2m1(v_nib) * v_mult).to(tl.bfloat16)
        scores = tl.dot(query, tl.trans(keys_t))
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


LAST = {}


def attention(query, cache, indices, block_table, token_to_req, out, vmean=None, lut=True, splits=None, warps=None,
              stages=None, block_n=None):
    """As kvlut_kernels.attention (same split profile unless overridden), with token-major word loads."""
    q = query
    if not q.shape[0]:
        return out
    if (cache.dtype != torch.uint8 or cache.ndim != 4 or cache.shape[-1] != 296 or cache.stride(-1) != 1
            or any(s % 4 for s in cache.stride()[:3]) or cache.data_ptr() % 4):
        raise ValueError('kvlut_vec: expects a uint8 [blocks, heads, page, 296] QSA cache with word-aligned rows')
    words = cache.view(torch.int32)                                   # [blocks, heads, page, 74]
    kv_heads = cache.shape[1]
    group_size = q.shape[1] // kv_heads
    block_m = triton.next_power_of_2(group_size)
    base_programs = q.shape[0] * kv_heads
    from arc3_qsa_profiles import quantized_sparse_profile
    bn, target_splits, partial_warps, compiler_stages = quantized_sparse_profile(base_programs, block_m, 'nvfp4')
    bn = block_n or bn
    num_tiles = triton.cdiv(indices.shape[1], bn)
    num_splits = splits if splits is not None else min(1 << (num_tiles.bit_length() - 1), target_splits)
    if num_splits == 1:
        partial_output = partial_lse = out
    else:
        partial_output = torch.empty((num_splits, *q.shape), dtype=torch.float32, device=q.device)
        partial_lse = torch.empty((num_splits, q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device)
    LAST['reader'] = _lut_vec_splitk_kernel[(q.shape[0], kv_heads, num_splits)](
        q, words, indices, block_table, token_to_req, partial_output, partial_lse, out,
        vmean if vmean is not None else q,
        q.stride(0), q.stride(1), words.stride(0), words.stride(1), words.stride(2), indices.stride(0),
        block_table.stride(0), out.stride(0), out.stride(1), q.shape[0], cache.shape[0], block_table.shape[0],
        TOPK=indices.shape[1], PAGE_SIZE=cache.shape[2], PAGE_TABLE_WIDTH=block_table.shape[1], GROUP_SIZE=group_size,
        HEAD_DIM=q.shape[2], NUM_QUERY_HEADS=q.shape[1], NUM_SPLITS=num_splits, NUM_TILES=num_tiles, BLOCK_M=block_m,
        BLOCK_N=bn, LUT=lut, VMEAN=vmean is not None, num_warps=warps or partial_warps,
        num_stages=stages or compiler_stages)
    if num_splits == 1:
        return out
    from vllm.models.qwen3_8_flash_next.nvidia.ops.qsa import _qsa_merge_splitk_kernel
    _qsa_merge_splitk_kernel[(q.shape[0], q.shape[1])](
        partial_output, partial_lse, out, out.stride(0), out.stride(1), q.shape[0],
        HEAD_DIM=q.shape[2], NUM_QUERY_HEADS=q.shape[1], NUM_SPLITS=num_splits,
        BLOCK_SPLITS=triton.next_power_of_2(num_splits), num_warps=2, num_stages=1)
    return out
