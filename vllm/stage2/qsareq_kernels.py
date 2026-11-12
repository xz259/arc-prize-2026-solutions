"""Request-level QSA reader for decode batches (milestone 2, 2026-10-02 17:20; user 17:05: "Do the QSA reader first").
Diagnostic basis (job 36, seloverlap): the top-2,048 selections of one request's 4 verify rows overlap 0.77 pairwise and
their union is 1.51x one selection, so a reader that loads each selected token once per request moves about 2.7x fewer
K/V bytes than the per-row reader (kvlut_kernels._lut_sparse_splitk_kernel, 561 us per layer at 228 rows, 12.2% of the
decode step; B2 14:10).

Index format (production expand_qsa_block_indices_cuda): row r's list holds its K_r selected complete compressed blocks
expanded to 4 tokens each (columns 0 .. 4 K_r - 1, in top-k order), then the causal tail of its open group (at most 3
tokens, columns 4 K_r .. 4 K_r + 2), then -1. A 4-column group k is a complete block iff its 4th entry is valid; the
tail group is the first group whose 4th entry is -1 (k = 512: columns 2048-2050).

Batch layout: request g owns rows query_start_loc[g] .. query_start_loc[g + 1] - 1 (block_table row g), at most R of
them. vLLM's v2 runner pads FULL decode graphs with 0-row requests (query_start_loc repeats num_tokens) and, with
adaptive verification, replays one varlen decode graph for any mix of 1..4 rows per request, so the grouping is read
from query_start_loc on the device, never assumed from the row count. The caller guarantees max_query_len <= R.

union_build   one program per request: keys block * 4 + r of its rows' complete blocks, tl.sort, dedupe; per union
              block a row mask (bit r: row r selected it), sorted by block id; per row its tail tokens (R x 4 slots,
              slot 3 always -1).
attention     one program per (request, KV head, split) over the union tiles (BLOCK_N / 4 blocks per tile; 16 or
              32 tokens) plus one tail tile; the query tile is the request's rows x 16 padded heads (BLOCK_M 64 for R 4); each row's scores
              are masked to its own selection (row-mask bit; that row's tail slots), so every row attends exactly its
              production set. Token decode, scales, VMEAN and the split-K partial layout are kvlut's; the production
              merge kernel combines the splits (partial LSEs start at -inf, so rows no request owns merge to 0). Only
              the summation order differs from the per-row reader."""
import torch
import triton
import triton.language as tl

from .kvlut_kernels import _decode_e2m1, _poly_from_bits


@triton.jit
def _union_build_kernel(indices_ptr, qsl_ptr, ublk_ptr, umask_ptr, ucount_ptr, utail_ptr, ws_ptr, stride_ir,
                        WIDTH: tl.constexpr, R: tl.constexpr, NB: tl.constexpr, UW: tl.constexpr):
    group = tl.program_id(0)
    start = tl.load(qsl_ptr + group)
    count = tl.minimum(tl.load(qsl_ptr + group + 1) - start, R)
    rr = tl.arange(0, R)[:, None]
    kk = tl.arange(0, NB)[None, :]
    live = rr < count
    rows = indices_ptr + (start + rr).to(tl.int64) * stride_ir
    first = tl.load(rows + kk * 4, mask=live, other=-1)
    last = tl.load(rows + kk * 4 + 3, mask=live, other=-1)
    key = tl.where((first >= 0) & (last >= 0), (first >> 2) * 4 + rr, 0x7FFFFFFF)
    keys = tl.sort(tl.reshape(key, [R * NB]))
    # neighbours of the sorted keys through a per-request scratch row: [-4 x4 | keys | sentinel x4]
    e = tl.arange(0, R * NB)
    ws = ws_ptr + group.to(tl.int64) * (R * NB + 8)
    four = tl.arange(0, 4)
    tl.store(ws + four, tl.full([4], -4, tl.int32))
    tl.store(ws + 4 + e, keys)
    tl.store(ws + 4 + R * NB + four, tl.full([4], 0x7FFFFFFF, tl.int32))
    tl.debug_barrier()
    prev = tl.load(ws + 3 + e, cache_modifier=".cg")
    n1 = tl.load(ws + 5 + e, cache_modifier=".cg")
    n2 = tl.load(ws + 6 + e, cache_modifier=".cg")
    n3 = tl.load(ws + 7 + e, cache_modifier=".cg")
    gid = keys >> 2
    new = (keys != 0x7FFFFFFF) & ((prev >> 2) != gid)
    # a block appears at most once per row, so its run is at most R <= 4 long
    mask = ((1 << (keys & 3)) | tl.where((n1 >> 2) == gid, 1 << (n1 & 3), 0)
            | tl.where((n2 >> 2) == gid, 1 << (n2 & 3), 0) | tl.where((n3 >> 2) == gid, 1 << (n3 & 3), 0))
    newi = new.to(tl.int32)
    pos = tl.cumsum(newi, axis=0) - 1
    tl.store(ublk_ptr + group.to(tl.int64) * UW + pos, gid, mask=new)
    tl.store(umask_ptr + group.to(tl.int64) * UW + pos, mask, mask=new)
    tl.store(ucount_ptr + group, tl.sum(newi, axis=0))
    tail_group = tl.min(tl.where(last < 0, kk, NB), axis=1)
    oo = tl.arange(0, 4)[None, :]
    cols = tail_group[:, None] * 4 + oo
    tail = tl.load(rows + cols, mask=live & (oo < 3) & (cols < WIDTH), other=-1)
    tl.store(utail_ptr + group * (R * 4) + rr * 4 + oo, tail)


@triton.jit
def _req_lut_splitk_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr, k_scale_ptr, v_scale_ptr, k_block_scale_ptr, v_block_scale_ptr,
    ublk_ptr, umask_ptr, ucount_ptr, utail_ptr, block_table_ptr, qsl_ptr,
    partial_output_ptr, partial_lse_ptr, output_ptr, vmean_ptr,
    stride_q_row, stride_q_head, stride_k_block, stride_k_token, stride_k_head,
    stride_v_block, stride_v_token, stride_v_head, stride_table_req,
    stride_output_row, stride_output_head, num_rows, num_cache_blocks, num_requests,
    UW: tl.constexpr, R: tl.constexpr, PAGE_SIZE: tl.constexpr, PAGE_TABLE_WIDTH: tl.constexpr,
    GROUP_SIZE: tl.constexpr, HEADS_P: tl.constexpr, HEAD_DIM: tl.constexpr, NUM_QUERY_HEADS: tl.constexpr,
    NUM_SPLITS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, LUT: tl.constexpr, VMEAN: tl.constexpr,
) -> None:
    group = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_id = tl.program_id(2)
    start = tl.load(qsl_ptr + group)
    row_count = tl.minimum(tl.load(qsl_ptr + group + 1) - start, R)
    request = group
    safe_request = tl.minimum(request, num_requests - 1)

    m_offsets = tl.arange(0, BLOCK_M)
    m_row = m_offsets // HEADS_P
    m_head = m_offsets % HEADS_P
    row_ok = (m_head < GROUP_SIZE) & (m_row < row_count)
    out_row = start + m_row
    row_ok &= out_row < num_rows
    dim_offsets = tl.arange(0, HEAD_DIM)
    column_offsets = tl.arange(0, BLOCK_N)
    first_head = kv_head * GROUP_SIZE
    query = tl.load(
        q_ptr + out_row[:, None] * stride_q_row + (first_head + m_head[:, None]) * stride_q_head
        + dim_offsets[None, :], mask=row_ok[:, None], other=0.0)

    max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
    softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634

    count = tl.load(ucount_ptr + group)
    union_tiles = (count + BLOCK_N // 4 - 1) // (BLOCK_N // 4)
    num_tiles = tl.where(row_count > 0, union_tiles + 1, 0)
    split_tile_start = split_id * num_tiles // NUM_SPLITS
    split_tile_end = (split_id + 1) * num_tiles // NUM_SPLITS
    entry_in_tile = column_offsets // 4
    for tile in range(split_tile_start, split_tile_end):
        entry = tile * (BLOCK_N // 4) + entry_in_tile
        in_union = entry < count
        block = tl.load(ublk_ptr + group * UW + entry, mask=in_union, other=-1)
        row_mask = tl.load(umask_ptr + group * UW + entry, mask=in_union, other=0)
        is_tail = tile == union_tiles
        tail_token = tl.load(utail_ptr + group * (R * 4) + column_offsets,
                             mask=is_tail & (column_offsets < R * 4), other=-1)
        logical_token = tl.where(is_tail, tail_token, tl.where(block >= 0, block * 4 + column_offsets % 4, -1))
        row_mask = tl.where(is_tail, 1 << entry_in_tile, row_mask)
        safe_token = tl.maximum(logical_token, 0)
        logical_page = safe_token // PAGE_SIZE
        page_offset = safe_token % PAGE_SIZE
        valid = (request < num_requests) & (logical_token >= 0) & (logical_page < PAGE_TABLE_WIDTH)
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
            + kv_head * stride_k_head + dim_offsets[:, None] // 16, mask=valid[None, :], other=0.0).to(tl.float32)
        v_blocks = tl.load(
            v_block_scale_ptr + safe_page[:, None] * stride_v_block + page_offset[:, None] * stride_v_token
            + kv_head * stride_v_head + dim_offsets[None, :] // 16, mask=valid[:, None], other=0.0).to(tl.float32)
        if LUT:
            k_bits = ((keys.to(tl.int32) << (19 - (dim_offsets[:, None] % 2) * 4)) & 0x780000) | 0x41000000
            v_bits = ((values.to(tl.int32) << (19 - (dim_offsets[None, :] % 2) * 4)) & 0x780000) | 0x41000000
            keys = (_poly_from_bits(k_bits) * (k_blocks * k_scale[None, :])).to(tl.bfloat16)
            values = (_poly_from_bits(v_bits) * (v_blocks * v_scale[:, None])).to(tl.bfloat16)
        else:
            k_nibbles = (keys >> ((dim_offsets[:, None] % 2) * 4)) & 15
            v_nibbles = (values >> ((dim_offsets[None, :] % 2) * 4)) & 15
            keys = (_decode_e2m1(k_nibbles) * (k_blocks * k_scale[None, :])).to(tl.bfloat16)
            values = (_decode_e2m1(v_nibbles) * (v_blocks * v_scale[:, None])).to(tl.bfloat16)
        allowed = valid[None, :] & (((row_mask[None, :] >> m_row[:, None]) & 1) != 0)
        scores = tl.dot(query, keys)
        scores *= softmax_scale_log2
        scores = tl.where(allowed, scores, -1.0e20)
        next_max = tl.maximum(max_value, tl.max(scores, axis=1))
        alpha = tl.math.exp2(max_value - next_max)
        probabilities = tl.where(allowed, tl.math.exp2(scores - next_max[:, None]), 0.0)
        accumulator = tl.dot(probabilities.to(values.dtype), values, acc=accumulator * alpha[:, None])
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        max_value = next_max

    has_values = normalizer > 0
    normalized_output = tl.where(has_values[:, None], accumulator / tl.maximum(normalizer[:, None], 1.0e-20), 0.0)
    if VMEAN:
        mean = tl.load(vmean_ptr + kv_head * HEAD_DIM + dim_offsets)
        normalized_output = tl.where(has_values[:, None], normalized_output + mean[None, :], 0.0)
    if NUM_SPLITS == 1:
        tl.store(output_ptr + out_row[:, None] * stride_output_row
                 + (first_head + m_head[:, None]) * stride_output_head + dim_offsets[None, :],
                 normalized_output, mask=row_ok[:, None])
    else:
        partial_lse = tl.where(has_values, max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)), -float("inf"))
        tl.store(partial_output_ptr + ((split_id * num_rows + out_row[:, None]) * NUM_QUERY_HEADS + first_head
                                       + m_head[:, None]) * HEAD_DIM + dim_offsets[None, :],
                 normalized_output, mask=row_ok[:, None])
        tl.store(partial_lse_ptr + (split_id * num_rows + out_row) * NUM_QUERY_HEADS + first_head + m_head,
                 partial_lse, mask=row_ok)


MAX_GROUPS = 512
_BUF = {}
LAST = {}                                            # last compiled-kernel handles (register diagnostics)


def buffers(R=4, NB=512, device='cuda'):
    """One fixed set per (R, NB, device), sized for MAX_GROUPS requests. Allocate it before graph capture (the caller's
    first eager call does), so captured graphs never see a reallocation. Shared by every layer: stream-ordered."""
    key = (R, NB, str(device))
    b = _BUF.get(key)
    if b is None:
        UW = R * NB
        b = _BUF[key] = dict(
            ublk=torch.empty((MAX_GROUPS, UW), dtype=torch.int32, device=device),
            umask=torch.empty((MAX_GROUPS, UW), dtype=torch.int32, device=device),
            ucount=torch.empty((MAX_GROUPS,), dtype=torch.int32, device=device),
            utail=torch.empty((MAX_GROUPS, R * 4), dtype=torch.int32, device=device),
            ws=torch.empty((MAX_GROUPS, UW + 8), dtype=torch.int32, device=device))
    return b


def union_build(indices, query_start_loc, R=4, compress=4):
    """indices: [rows, TOPK + compress - 1] int32 (production expansion); query_start_loc: int32 [requests + 1].
    Returns (ublk, umask, ucount, utail) views for the requests."""
    width = indices.shape[1]
    groups = query_start_loc.shape[0] - 1
    if compress != 4 or indices.dtype != torch.int32 or indices.stride(1) != 1 or query_start_loc.dtype != torch.int32:
        raise ValueError('qsareq: unsupported index layout')
    if groups > MAX_GROUPS:
        raise ValueError('qsareq: too many requests')
    NB = (width - compress + 1) // compress
    b = buffers(R, NB, indices.device)
    if groups > 0:
        LAST['union'] = _union_build_kernel[(groups,)](
            indices, query_start_loc, b['ublk'], b['umask'], b['ucount'], b['utail'], b['ws'], indices.stride(0),
            WIDTH=width, R=R, NB=NB, UW=R * NB, num_warps=4)
    return b['ublk'][:groups], b['umask'][:groups], b['ucount'][:groups], b['utail'][:groups]


def attention(query, cache, indices, block_table, query_start_loc, out, vmean=None, lut=True, R=4, splits=4,
              warps=8, stages=1, block_n=16, union=None):
    """As kvlut_kernels.attention for a decode batch (at most R rows per request, rows of request g from
    query_start_loc). Rows no request owns are left as they are (splits 1) or set to 0 (splits > 1)."""
    from .kvlut_kernels import _views
    (k_cache, k_blocks, k_scale), (v_cache, v_blocks, v_scale) = _views(cache)
    q = query
    rows = q.shape[0]
    groups = query_start_loc.shape[0] - 1
    if not rows or groups <= 0:
        return out
    ublk, umask, ucount, utail = union if union is not None else union_build(indices, query_start_loc, R)
    group_size = q.shape[1] // k_cache.shape[2]
    heads_p = triton.next_power_of_2(group_size)
    if splits == 1:
        partial_output = partial_lse = out
    else:
        partial_output = torch.empty((splits, *q.shape), dtype=torch.float32, device=q.device)
        partial_lse = torch.full((splits, rows, q.shape[1]), -float('inf'), dtype=torch.float32, device=q.device)
    LAST['reader'] = _req_lut_splitk_kernel[(groups, k_cache.shape[2], splits)](
        q, k_cache, v_cache, k_scale, v_scale, k_blocks, v_blocks, ublk, umask, ucount, utail, block_table,
        query_start_loc, partial_output, partial_lse, out, vmean if vmean is not None else q,
        q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), block_table.stride(0),
        out.stride(0), out.stride(1), rows, k_cache.shape[0], block_table.shape[0],
        UW=ublk.shape[1], R=R, PAGE_SIZE=k_cache.shape[1], PAGE_TABLE_WIDTH=block_table.shape[1],
        GROUP_SIZE=group_size, HEADS_P=heads_p, HEAD_DIM=q.shape[2], NUM_QUERY_HEADS=q.shape[1], NUM_SPLITS=splits,
        BLOCK_M=R * heads_p, BLOCK_N=block_n, LUT=lut, VMEAN=vmean is not None, num_warps=warps, num_stages=stages)
    if splits == 1:
        return out
    from vllm.models.qwen3_8_flash_next.nvidia.ops.qsa import _qsa_merge_splitk_kernel
    _qsa_merge_splitk_kernel[(rows, q.shape[1])](
        partial_output, partial_lse, out, out.stride(0), out.stride(1), rows,
        HEAD_DIM=q.shape[2], NUM_QUERY_HEADS=q.shape[1], NUM_SPLITS=splits,
        BLOCK_SPLITS=triton.next_power_of_2(splits), num_warps=2, num_stages=1)
    return out


def union_reference(indices, query_start_loc, R=4):
    """Host view of the union representation: per real row, the set of logical tokens it gives that row (tests)."""
    ublk, umask, ucount, utail = (t.cpu() for t in union_build(indices, query_start_loc, R))
    qsl = query_start_loc.cpu().tolist()
    sets = []
    for g in range(len(qsl) - 1):
        n = int(ucount[g])
        blk, msk = ublk[g, :n].tolist(), umask[g, :n].tolist()
        for r in range(min(qsl[g + 1] - qsl[g], R)):
            s = {b * 4 + o for b, m in zip(blk, msk) if m >> r & 1 for o in range(4)}
            s |= {t for t in utail[g, r * 4:(r + 1) * 4].tolist() if t >= 0}
            sets.append(s)
    return sets, ublk, ucount
