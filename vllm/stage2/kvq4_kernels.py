"""Kernels of g4, the quad-shared codebook 4-bit main-QSA KV format (2026-10-26, Claude fork bdfc5402; the user: "g4 looks good.
I think we will try to implement it"; design and IE's review: review74/engine_prefill/fork_20261026/G4_DESIGN_2026-10-26.md).

Layout: 266-byte token rows. Four consecutive rows (a quad, at a page offset divisible by 4; a quad never crosses a page) hold
  0     K codes of t0..t3 (128 bytes each, nibble pairs as kvlut_kernels)
  512   V codes of t0..t3
  1024  K block scales (16 E4M3)    1040  K head scale (FP32)
  1044  V block scales              1060  V head scale
so a quad is 4 x 266 = 1,064 bytes and every FP32 is 4-byte aligned (PAGE even). One scale set per quad and side.

store   per quad and side, over the rows present so far (this update's rows; a continued quad's earlier rows from the side
        buffer): y = H(x - mu) exactly as kvlut_kernels._store_lut_kernel, block maxima over the present rows, the head scale
        max / 448, per block the E4M3 code among c-1 .. c+2 by the squared error summed over the present rows in row order
        (((e0 + e1) + e2) + e3; strictly smaller wins, in candidate order), codes by nearest LUTP level. A quad with one present row
        is kvlut's per-token store bit for bit; a quad stored in pieces equals the quad stored at once.
buffer  BF16 originals (the K / V rows the store receives) of each request's newest 8 rows per update, at id * 8 + (slot + off) % 8,
        tagged row + off (id: kvring's ring id; off: the request's deleted tokens, kvring's ring_meta; PAGE % 8 == 0 makes
        slot = row mod 8, and row + off is the row's original position, so a compaction keeps the key). A verify step writes K + 1
        = 5 rows and rolls back at most 4, so the earlier rows of a continued quad ([s - 3, s)) are never overwritten (mod 8 is the
        minimum, IE). A leader whose earlier row is not in the buffer (no ring id, row not found, tag mismatch) re-encodes it from
        its decoded stored values: lossy, counted in stats[0] (expected 0; the smoke asserts it).
attend  kvlut_kernels' sparse split-K reader (and kvring's copy, LSE_DIRECT) with quad addressing; arithmetic and launch rules
        unchanged (bitwise equal to the 296-byte readers on a cache whose four rows per quad carry equal scales)."""
import torch
import triton
import triton.language as tl

from .kvlut_kernels import _h16, _poly_encode, _poly_level, _poly_from_bits, _e4m3_code, launch_profile
from .kvring_kernels import _row_from_slot, _ring_compact_kernel, _ring_merge_kernel

ROW = 266
QUAD = 4 * ROW


@triton.jit
def _load_y(k_src, v_src, r, head, stride_kr, stride_kh, stride_vr, stride_vh, kmean, vmean, side, CENTER: tl.constexpr):
    """y = H(x - mu) of row r, head, side (FP32 [16, 16]) as kvlut_kernels._store_lut_kernel computes it."""
    idx = tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]
    if side == 0:
        Y = tl.load(k_src + r * stride_kr + head * stride_kh + idx).to(tl.float32)
        if CENTER:
            Y = Y - tl.load(kmean + head * 256 + idx)
    else:
        Y = tl.load(v_src + r * stride_vr + head * stride_vh + idx).to(tl.float32)
        if CENTER:
            Y = Y - tl.load(vmean + head * 256 + idx)
    H = _h16()
    T = tl.dot(H, Y, input_precision="ieee")
    return tl.dot(T, H, input_precision="ieee") * 0.0625


@triton.jit
def _decoded_pairs(quad, p, side):
    """The stored row p of a quad side, decoded (level x block scale x head scale): the (even, odd) dims [16, 8] each."""
    pair = tl.arange(0, 16)[:, None] * 8 + tl.arange(0, 8)[None, :]
    b = tl.load(quad + side * 512 + p * 128 + pair).to(tl.int32)
    blocks = tl.load(quad + 1024 + side * 20 + tl.arange(0, 16)).to(tl.float8e4nv, bitcast=True).to(tl.float32)
    hs = tl.load((quad + 1040 + side * 20).to(tl.pointer_type(tl.float32)))
    dq = hs * blocks
    return _poly_level(b & 15) * dq[:, None], _poly_level((b >> 4) & 15) * dq[:, None]


@triton.jit
def _quad_row(p, i0, t, qslot, n_rows, slots, key, value, buf_k, buf_v, buf_tag, quad, buf_ok, ring_id, off, row_t, head, side,
              stride_kr, stride_kh, stride_vr, stride_vh, stride_br, stride_bh, kmean, vmean, chain, CENTER: tl.constexpr):
    """Row p of the leader's quad: (y0, y1, present, chain, fallback). Earlier rows (p < i0) come from the side buffer (or, without
    a matching entry, from their decoded stored values); rows p >= i0 are this update's consecutive slots."""
    zero = tl.zeros([16, 8], dtype=tl.float32)
    y0 = zero
    y1 = zero
    present = 0
    fallback = 0
    if p < i0:
        present = 1
        bidx = ring_id * 8 + (qslot + p + off) % 8
        tag = tl.load(buf_tag + tl.maximum(bidx, 0), mask=buf_ok, other=-1)
        hit = buf_ok & (tag == row_t - i0 + p + off)
        if hit:
            Y = _load_y(buf_k, buf_v, bidx, head, stride_br, stride_bh, stride_br, stride_bh, kmean, vmean, side, CENTER)
            y0, y1 = tl.split(tl.reshape(Y, [16, 8, 2]))
        else:
            y0, y1 = _decoded_pairs(quad, p, side)
            fallback = 1
    else:
        bi = t + p - i0
        nxt = tl.load(slots + tl.minimum(bi, n_rows - 1)).to(tl.int64)
        chain = chain & (bi < n_rows) & (nxt == qslot + p)
        if chain:
            present = 1
            Y = _load_y(key, value, bi, head, stride_kr, stride_kh, stride_vr, stride_vh, kmean, vmean, side, CENTER)
            y0, y1 = tl.split(tl.reshape(Y, [16, 8, 2]))
    return y0, y1, present, chain, fallback


@triton.jit
def _cand(y0, y1, dq, safe):
    n0 = tl.where(dq[:, None] > 0., tl.div_rn(y0, safe[:, None]), 0.)
    n1 = tl.where(dq[:, None] > 0., tl.div_rn(y1, safe[:, None]), 0.)
    c0 = _poly_encode(n0)
    c1 = _poly_encode(n1)
    r0 = _poly_level(c0) * dq[:, None]
    r1 = _poly_level(c1) * dq[:, None]
    return c0, c1, tl.sum((r0 - y0) * (r0 - y0) + (r1 - y1) * (r1 - y1), axis=1)


@triton.jit
def _block_max(y0, y1, present):
    return tl.where(present != 0, tl.max(tl.maximum(tl.abs(y0), tl.abs(y1)), axis=1), 0.)


@triton.jit
def _store_q4_kernel(key, value, cache, slots, token_to_req, ring_meta, block_table, seq_lens, kmean, vmean,
                     buf_k, buf_v, buf_tag, stats,
                     stride_kr, stride_kh, stride_vr, stride_vh, stride_cb, stride_ch, stride_br, stride_bh, stride_bt,
                     n_rows, num_blocks, num_requests, num_tables,
                     PAGE: tl.constexpr, NCAND: tl.constexpr, CENTER: tl.constexpr, W: tl.constexpr, BLOCK_W: tl.constexpr,
                     STATS: tl.constexpr):
    t = tl.program_id(0)
    head = tl.program_id(1)
    side = tl.program_id(2)
    slot = tl.load(slots + t).to(tl.int64)
    i0 = slot % 4
    prev = tl.load(slots + tl.maximum(t - 1, 0)).to(tl.int64)
    follows = (t > 0) & (i0 > 0) & (prev == slot - 1)          # the quad's previous slot is this update's previous row
    if (slot >= 0) & (slot < num_blocks * PAGE) & (follows == 0):
        qslot = slot - i0
        quad = cache + (qslot // PAGE) * stride_cb + head * stride_ch + (qslot % PAGE) * 266
        request = tl.load(token_to_req + t)
        req_ok = (request >= 0) & (request < num_requests)
        safe_req = tl.minimum(tl.maximum(request, 0), num_requests - 1)
        ring_id = tl.load(ring_meta + safe_req * 3)
        off = tl.load(ring_meta + safe_req * 3 + 1)
        seq_len = tl.load(seq_lens + safe_req)
        row_t, found, _s = _row_from_slot(slots, block_table, t, safe_req, stride_bt, num_tables, seq_len, W, BLOCK_W, PAGE)
        buf_ok = req_ok & (ring_id >= 0) & found
        chain = (t >= 0)
        y0a, y1a, pa, chain, fa = _quad_row(0, i0, t, qslot, n_rows, slots, key, value, buf_k, buf_v, buf_tag, quad, buf_ok,
                                            ring_id, off, row_t, head, side, stride_kr, stride_kh, stride_vr, stride_vh,
                                            stride_br, stride_bh, kmean, vmean, chain, CENTER)
        y0b, y1b, pb, chain, fb = _quad_row(1, i0, t, qslot, n_rows, slots, key, value, buf_k, buf_v, buf_tag, quad, buf_ok,
                                            ring_id, off, row_t, head, side, stride_kr, stride_kh, stride_vr, stride_vh,
                                            stride_br, stride_bh, kmean, vmean, chain, CENTER)
        y0c, y1c, pc, chain, fc = _quad_row(2, i0, t, qslot, n_rows, slots, key, value, buf_k, buf_v, buf_tag, quad, buf_ok,
                                            ring_id, off, row_t, head, side, stride_kr, stride_kh, stride_vr, stride_vh,
                                            stride_br, stride_bh, kmean, vmean, chain, CENTER)
        y0d, y1d, pd, chain, fd = _quad_row(3, i0, t, qslot, n_rows, slots, key, value, buf_k, buf_v, buf_tag, quad, buf_ok,
                                            ring_id, off, row_t, head, side, stride_kr, stride_kh, stride_vr, stride_vh,
                                            stride_br, stride_bh, kmean, vmean, chain, CENTER)
        if STATS:
            if (head == 0) & (side == 0):
                nf = fa + fb + fc + fd
                if nf > 0:
                    tl.atomic_add(stats + 0, nf.to(tl.int64))
                    if buf_ok == 0:
                        tl.atomic_add(stats + 1, 1)
                    # debug (2026-10-26): the first 64 fallback leaders, 10 fields each at stats[8 + 10 e]
                    e = tl.atomic_add(stats + 4, (ring_id >= 0).to(tl.int64))     # gameplay events only (warmup dummies: ring_id -1)
                    if (e < 64) & (ring_id >= 0):
                        pb_idx = ring_id * 8 + (slot - 1 + off) % 8
                        tprev = tl.load(buf_tag + tl.maximum(pb_idx, 0), mask=(ring_id >= 0), other=-7)
                        base = stats + 8 + e * 10
                        tl.store(base + 0, request.to(tl.int64))
                        tl.store(base + 1, ring_id.to(tl.int64))
                        tl.store(base + 2, row_t.to(tl.int64))
                        tl.store(base + 3, i0.to(tl.int64))
                        tl.store(base + 4, slot.to(tl.int64))
                        tl.store(base + 5, found.to(tl.int64))
                        tl.store(base + 6, tprev.to(tl.int64))
                        tl.store(base + 7, (row_t - 1 + off).to(tl.int64))
                        tl.store(base + 8, off.to(tl.int64))
                        tl.store(base + 9, seq_len.to(tl.int64))
        maxima = tl.maximum(tl.maximum(tl.maximum(_block_max(y0a, y1a, pa), _block_max(y0b, y1b, pb)),
                                       _block_max(y0c, y1c, pc)), _block_max(y0d, y1d, pd))
        maximum = tl.max(maxima, axis=0)
        head_scale = tl.where(maximum == 0., 1., tl.maximum(tl.div_rn(maximum, 448.), 1e-30))
        code = _e4m3_code(tl.minimum(tl.div_rn(maxima, head_scale), 448.))
        best_code = code
        best_err = tl.full([16], float('inf'), tl.float32)
        izero = tl.zeros([16, 8], dtype=tl.int32)
        ba0, ba1, bb0, bb1, bc0, bc1, bd0, bd1 = izero, izero, izero, izero, izero, izero, izero, izero
        for i in tl.static_range(NCAND):
            if NCAND > 1:
                c = tl.minimum(tl.maximum(code.to(tl.int32) + (i - 1), 0), 126).to(tl.uint8)
            else:
                c = code
            dq = head_scale * c.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            safe = tl.where(dq > 0., dq, 1.)
            a0, a1, ea = _cand(y0a, y1a, dq, safe)
            b0, b1, eb = _cand(y0b, y1b, dq, safe)
            c0, c1, ec = _cand(y0c, y1c, dq, safe)
            d0, d1, ed = _cand(y0d, y1d, dq, safe)
            err = tl.zeros([16], dtype=tl.float32)          # 0 + e0 == e0: one present row is kvlut's error bit for bit
            err = err + tl.where(pa != 0, ea, 0.)
            err = err + tl.where(pb != 0, eb, 0.)
            err = err + tl.where(pc != 0, ec, 0.)
            err = err + tl.where(pd != 0, ed, 0.)
            take = err < best_err
            best_err = tl.where(take, err, best_err)
            best_code = tl.where(take, c, best_code)
            ba0 = tl.where(take[:, None], a0, ba0)
            ba1 = tl.where(take[:, None], a1, ba1)
            bb0 = tl.where(take[:, None], b0, bb0)
            bb1 = tl.where(take[:, None], b1, bb1)
            bc0 = tl.where(take[:, None], c0, bc0)
            bc1 = tl.where(take[:, None], c1, bc1)
            bd0 = tl.where(take[:, None], d0, bd0)
            bd1 = tl.where(take[:, None], d1, bd1)
        pair = tl.arange(0, 16)[:, None] * 8 + tl.arange(0, 8)[None, :]
        codes = quad + side * 512 + pair
        if pa != 0:
            tl.store(codes, (ba0 | (ba1 << 4)).to(tl.uint8))
        if pb != 0:
            tl.store(codes + 128, (bb0 | (bb1 << 4)).to(tl.uint8))
        if pc != 0:
            tl.store(codes + 256, (bc0 | (bc1 << 4)).to(tl.uint8))
        if pd != 0:
            tl.store(codes + 384, (bd0 | (bd1 << 4)).to(tl.uint8))
        tl.store(quad + 1024 + side * 20 + tl.arange(0, 16), best_code)
        tl.store((quad + 1040 + side * 20).to(tl.pointer_type(tl.float32)), head_scale)


@triton.jit
def _buffer_q4_kernel(key, value, slots, token_to_req, ring_meta, block_table, seq_lens, buf_k, buf_v, buf_tag, stats,
                      stride_kr, stride_kh, stride_vr, stride_vh, stride_br, stride_bh, stride_bt,
                      n_rows, num_blocks, num_requests, num_tables,
                      PAGE: tl.constexpr, W: tl.constexpr, BLOCK_W: tl.constexpr, STATS: tl.constexpr):
    # After the store (stream order): each request's newest min(8, n) rows of this update into the side buffer, with their tags.
    t = tl.program_id(0)
    head = tl.program_id(1)
    side = tl.program_id(2)
    slot = tl.load(slots + t).to(tl.int64)
    request = tl.load(token_to_req + t)
    if (slot >= 0) & (slot < num_blocks * PAGE) & (request >= 0) & (request < num_requests):
        later = tl.load(token_to_req + tl.minimum(t + 8, n_rows - 1))
        tail = (t + 8 >= n_rows) | (later != request)
        ring_id = tl.load(ring_meta + request * 3)
        if tail & (ring_id >= 0):
            off = tl.load(ring_meta + request * 3 + 1)
            seq_len = tl.load(seq_lens + request)
            row, found, _s = _row_from_slot(slots, block_table, t, request, stride_bt, num_tables, seq_len, W, BLOCK_W, PAGE)
            if found:
                bidx = ring_id * 8 + (slot + off) % 8
                idx = tl.arange(0, 256)
                if side == 0:
                    tl.store(buf_k + bidx * stride_br + head * stride_bh + idx, tl.load(key + t * stride_kr + head * stride_kh + idx))
                else:
                    tl.store(buf_v + bidx * stride_br + head * stride_bh + idx, tl.load(value + t * stride_vr + head * stride_vh + idx))
                if (head == 0) & (side == 0):
                    tl.store(buf_tag + bidx, row + off)
            elif STATS:
                if (head == 0) & (side == 0):
                    tl.atomic_add(stats + 2, 1)


def check_cache(cache):
    if (cache.dtype != torch.uint8 or cache.ndim != 4 or cache.shape[-1] != ROW or cache.stride(-1) != 1 or cache.stride(2) != ROW
            or cache.shape[2] % 8 or cache.stride(1) % 4 or cache.stride(0) % 4 or cache.data_ptr() % 4):
        raise ValueError('kvq4: expects a uint8 [blocks, heads, PAGE (a multiple of 8), 266] cache with 4-byte aligned heads, got %s %s'
                         % (tuple(cache.shape), tuple(cache.stride())))


def store(key, value, cache, slots, token_to_req, ring_meta, block_table, seq_lens, buf_k, buf_v, buf_tag,
          kmean=None, vmean=None, mse=True, stats=None, n=None):
    """The g4 store of this update's rows. key / value BF16 [tokens, kv heads, 256]; cache uint8 [blocks, kv heads, PAGE, 266];
    slots the update's slot mapping; token_to_req int32 [tokens] (-1 = padding); ring_meta int32 [requests, 3] (kvring: ring_id,
    off, floor); block_table / seq_lens: this layer's attention metadata (each token's row, as kvring_kernels.ring_store);
    buf_k / buf_v BF16 [ids * 8, kv heads, 256] and buf_tag int32 [ids * 8]: the side buffer; kmean / vmean FP32 [kv heads, 256];
    stats int64 [4]: [0] earlier rows re-encoded from decoded values, [1] leaders without a usable buffer, [2] newest rows not
    buffered (row not found)."""
    t = len(slots) if n is None else n
    if key.dtype != torch.bfloat16 or value.dtype != key.dtype or key.ndim != 3 or key.shape != value.shape or key.shape[-1] != 256:
        raise ValueError('kvq4: QSA writes require 256-dimensional BF16 K/V')
    if (slots.ndim != 1 or slots.dtype not in (torch.int32, torch.int64) or slots.stride(0) != 1 or t > len(key) or t > len(slots)
            or key.stride(-1) != 1 or value.stride(-1) != 1 or token_to_req.shape[0] < t):
        raise ValueError('kvq4: unsupported QSA input or slot layout')
    check_cache(cache)
    if cache.shape[1] != key.shape[1] or buf_k.shape[1:] != key.shape[1:] or buf_v.stride() != buf_k.stride() or buf_k.stride(-1) != 1:
        raise ValueError('kvq4: head count or side-buffer layout mismatch')
    if block_table.stride(-1) != 1:
        raise ValueError('kvq4: the page table must be row-contiguous')
    if not t:
        return
    center = kmean is not None
    if center and (vmean is None or kmean.shape != (key.shape[1], 256) or vmean.shape != kmean.shape
                   or kmean.dtype != torch.float32 or not kmean.is_contiguous() or not vmean.is_contiguous()):
        raise ValueError('kvq4: means must be contiguous FP32 [heads, 256] for both sides')
    w = block_table.shape[1]
    common = (key.stride(0), key.stride(1), value.stride(0), value.stride(1))
    st = stats if stats is not None else seq_lens
    _store_q4_kernel[(t, key.shape[1], 2)](
        key, value, cache, slots, token_to_req, ring_meta, block_table, seq_lens,
        kmean if center else key, vmean if center else key, buf_k, buf_v, buf_tag, st,
        *common, cache.stride(0), cache.stride(1), buf_k.stride(0), buf_k.stride(1), block_table.stride(0),
        t, cache.shape[0], seq_lens.shape[0], block_table.shape[0],
        PAGE=cache.shape[2], NCAND=4 if mse else 1, CENTER=center, W=w, BLOCK_W=max(16, triton.next_power_of_2(w)),
        STATS=stats is not None, num_warps=4)
    _buffer_q4_kernel[(t, key.shape[1], 2)](
        key, value, slots, token_to_req, ring_meta, block_table, seq_lens, buf_k, buf_v, buf_tag, st,
        *common, buf_k.stride(0), buf_k.stride(1), block_table.stride(0),
        t, cache.shape[0], seq_lens.shape[0], block_table.shape[0],
        PAGE=cache.shape[2], W=w, BLOCK_W=max(16, triton.next_power_of_2(w)), STATS=stats is not None, num_warps=4)


@triton.jit
def _q4_sparse_splitk_kernel(
    q_ptr, cache_ptr, cache_f32_ptr, indices_ptr, block_table_ptr, token_to_req_ptr, partial_output_ptr, partial_lse_ptr,
    output_ptr, vmean_ptr,
    stride_q_row, stride_q_head, stride_c_block, stride_c_head, stride_indices_row, stride_table_req,
    stride_output_row, stride_output_head, num_rows, num_cache_blocks, num_requests,
    TOPK: tl.constexpr, PAGE_SIZE: tl.constexpr, PAGE_TABLE_WIDTH: tl.constexpr, GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr, NUM_QUERY_HEADS: tl.constexpr, NUM_SPLITS: tl.constexpr, NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, VMEAN: tl.constexpr, LSE_DIRECT: tl.constexpr,
) -> None:
    # kvlut_kernels._lut_sparse_splitk_kernel (LUT mode) with quad addressing; LSE_DIRECT: kvring's copy (the one-split epilogue
    # also stores the lse for the ring merge).
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
        # quad addressing: the token's quad (1,064 bytes), its codes at (offset % 4) x 128 per side, the quad's scales
        quad = safe_page * stride_c_block + kv_head * stride_c_head + (page_offset // 4).to(tl.int64) * 1064
        sub = (page_offset % 4) * 128
        keys = tl.load(cache_ptr + quad[None, :] + sub[None, :] + (dim_offsets[:, None] // 2), mask=valid[None, :], other=0)
        values = tl.load(cache_ptr + quad[:, None] + 512 + sub[:, None] + (dim_offsets[None, :] // 2), mask=valid[:, None],
                         other=0)
        k_scale = tl.load(cache_f32_ptr + ((quad + 1040) >> 2), mask=valid, other=1.0)
        v_scale = tl.load(cache_f32_ptr + ((quad + 1060) >> 2), mask=valid, other=1.0)
        k_blocks = tl.load(cache_ptr + quad[None, :] + 1024 + block_offsets[:, None], mask=valid[None, :],
                           other=0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        v_blocks = tl.load(cache_ptr + quad[:, None] + 1044 + block_offsets[None, :], mask=valid[:, None],
                           other=0).to(tl.float8e4nv, bitcast=True).to(tl.float32)
        k_mult = tl.reshape(tl.broadcast_to((k_blocks * k_scale[None, :])[:, None, :], (HEAD_DIM // 16, 16, BLOCK_N)),
                            (HEAD_DIM, BLOCK_N))
        v_mult = tl.reshape(tl.broadcast_to((v_blocks * v_scale[:, None])[:, :, None], (BLOCK_N, HEAD_DIM // 16, 16)),
                            (BLOCK_N, HEAD_DIM))
        k_bits = ((keys.to(tl.int32) << (19 - (dim_offsets[:, None] % 2) * 4)) & 0x780000) | 0x41000000
        v_bits = ((values.to(tl.int32) << (19 - (dim_offsets[None, :] % 2) * 4)) & 0x780000) | 0x41000000
        keys = (_poly_from_bits(k_bits) * k_mult).to(tl.bfloat16)
        values = (_poly_from_bits(v_bits) * v_mult).to(tl.bfloat16)
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
        if LSE_DIRECT:
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


_F32 = {}


def _f32(cache):
    key = (cache.data_ptr(), tuple(cache.shape), tuple(cache.stride()))
    v = _F32.get(key)
    if v is None:
        check_cache(cache)
        if not cache.is_contiguous():
            raise ValueError('kvq4: the reader needs a contiguous cache (flat FP32 view of the head scales)')
        v = _F32[key] = cache.reshape(-1).view(torch.float32)
    return v


def _launch(q, cache, indices, block_table, token_to_req, out, partial_output, partial_lse, vmean, num_splits, num_tiles,
            block_n, block_m, warps, stages, lse_direct):
    kvh = cache.shape[1]
    _q4_sparse_splitk_kernel[(q.shape[0], kvh, num_splits)](
        q, cache, _f32(cache), indices, block_table, token_to_req, partial_output, partial_lse, out,
        vmean if vmean is not None else q,
        q.stride(0), q.stride(1), cache.stride(0), cache.stride(1), indices.stride(0), block_table.stride(0),
        out.stride(0), out.stride(1), q.shape[0], cache.shape[0], block_table.shape[0],
        TOPK=indices.shape[1], PAGE_SIZE=cache.shape[2], PAGE_TABLE_WIDTH=block_table.shape[1],
        GROUP_SIZE=q.shape[1] // kvh, HEAD_DIM=q.shape[2], NUM_QUERY_HEADS=q.shape[1], NUM_SPLITS=num_splits,
        NUM_TILES=num_tiles, BLOCK_M=block_m, BLOCK_N=block_n, VMEAN=vmean is not None, LSE_DIRECT=lse_direct,
        num_warps=warps, num_stages=stages)


def attention(query, cache, indices, block_table, token_to_req, out, vmean=None, lut=True, profile=None):
    """kvlut_kernels.attention for the quad layout (same signature; lut must be True: the E2M1 mode has no quad layout)."""
    if not lut:
        raise ValueError('kvq4: the quad layout has no E2M1 mode')
    q = query
    if not q.shape[0]:
        return out
    kvh = cache.shape[1]
    group_size = q.shape[1] // kvh
    block_m = triton.next_power_of_2(group_size)
    base_programs = q.shape[0] * kvh
    block_n, target_splits, partial_warps, compiler_stages = launch_profile(base_programs, block_m, profile)
    num_tiles = triton.cdiv(indices.shape[1], block_n)
    num_splits = min(1 << (num_tiles.bit_length() - 1), target_splits)
    if num_splits == 1:
        partial_output = partial_lse = out
    else:
        partial_output = torch.empty((num_splits, *q.shape), dtype=torch.float32, device=q.device)
        partial_lse = torch.empty((num_splits, q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device)
    _launch(q, cache, indices, block_table, token_to_req, out, partial_output, partial_lse, vmean, num_splits, num_tiles,
            block_n, block_m, partial_warps, compiler_stages, False)
    if num_splits == 1:
        return out
    from vllm.models.qwen3_8_flash_next.nvidia.ops.qsa import _qsa_merge_splitk_kernel
    _qsa_merge_splitk_kernel[(q.shape[0], q.shape[1])](
        partial_output, partial_lse, out, out.stride(0), out.stride(1), q.shape[0],
        HEAD_DIM=q.shape[2], NUM_QUERY_HEADS=q.shape[1], NUM_SPLITS=num_splits,
        BLOCK_SPLITS=triton.next_power_of_2(num_splits), num_warps=2, num_stages=1)
    return out


def ring_attention(query, cache, indices, block_table, token_to_req, seq_lens, ring_k, ring_v, ring_meta, out, vmean=None, n=256,
                   profile=None, ring_block_n=64, debug=None, slot_rows=None, stats=None, r=None):
    """kvring_kernels.attention for the quad layout (same signature): kvring's compaction and merge kernels unchanged, the lutc
    part read by the quad reader with the lse epilogue. The G2a shadow debug path is not supported."""
    if debug is not None:
        raise ValueError('kvq4: kvring debug shadows are not supported with the quad layout')
    q = query
    if not q.shape[0]:
        return out
    kvh = cache.shape[1]
    group_size = q.shape[1] // kvh
    block_m = triton.next_power_of_2(group_size)
    base_programs = q.shape[0] * kvh
    block_n, target_splits, partial_warps, compiler_stages = launch_profile(base_programs, block_m, profile)
    num_tiles = triton.cdiv(indices.shape[1], block_n)
    lut_splits = min(1 << (num_tiles.bit_length() - 1), target_splits)
    if lut_splits == 1:
        partial_output = out
        partial_lse = torch.empty((1, q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device)
    else:
        partial_output = torch.empty((lut_splits, *q.shape), dtype=torch.float32, device=q.device)
        partial_lse = torch.empty((lut_splits, q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device)
    vm = vmean if vmean is not None else q
    compact = torch.empty((q.shape[0], n), dtype=torch.int32, device=q.device)
    count = torch.empty((q.shape[0],), dtype=torch.int32, device=q.device)
    masked = torch.empty_like(indices)
    tags = slot_rows is not None
    rr = r if r is not None else n
    _ring_compact_kernel[(q.shape[0],)](
        indices, masked, token_to_req, seq_lens, ring_meta, compact, count,
        slot_rows if tags else token_to_req, stats if stats is not None else count, token_to_req, count, seq_lens,
        indices.stride(0), masked.stride(0), seq_lens.shape[0], ring_k.shape[0], TOPK=indices.shape[1],
        BLOCK=triton.next_power_of_2(indices.shape[1]), CAP=n, N=n, R=rr, TAGS=tags, COUNT=tags and stats is not None,
        ALIAS_EVENTS=0, num_warps=8)
    _launch(q, cache, masked, block_table, token_to_req, out, partial_output, partial_lse, vmean, lut_splits, num_tiles,
            block_n, block_m, partial_warps, compiler_stages, True)
    if lut_splits > 1:
        from vllm.models.qwen3_8_flash_next.nvidia.ops.qsa import _qsa_merge_splitk_kernel
        _qsa_merge_splitk_kernel[(q.shape[0], q.shape[1])](
            partial_output, partial_lse, out, out.stride(0), out.stride(1), q.shape[0],
            HEAD_DIM=q.shape[2], NUM_QUERY_HEADS=q.shape[1], NUM_SPLITS=lut_splits,
            BLOCK_SPLITS=triton.next_power_of_2(lut_splits), num_warps=2, num_stages=1)
    _ring_merge_kernel[(q.shape[0], kvh)](
        q, ring_k, ring_v, compact, count, token_to_req, ring_meta, partial_output, partial_lse, out, vm,
        q.stride(0), q.stride(1), ring_k.stride(0), ring_k.stride(1), out.stride(0), out.stride(1),
        q.shape[0], seq_lens.shape[0], ring_k.shape[0],
        CAP=n, GROUP_SIZE=group_size, HEAD_DIM=q.shape[2], NUM_QUERY_HEADS=q.shape[1], LUT_SPLITS=lut_splits,
        BLOCK_SPLITS=triton.next_power_of_2(max(lut_splits, 2)), BLOCK_M=block_m, BLOCK_N=ring_block_n,
        VMEAN=vmean is not None, N=n, R=rr, num_warps=4, num_stages=1)
    return out
