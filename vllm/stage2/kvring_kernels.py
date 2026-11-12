"""Kernels of the BF16 recent window over the lutc main-QSA KV (item 4a, 2026-10-15; design: harnesspod_20261013/kvring/
KVRING_DESIGN_2026-10-15.md, reviewed by IE; the user picked BF16-256 at 13:0x UTC).

Dual storage. The lutc pages are written exactly as today (kvlut_kernels.store). In addition, the newest N resident rows of every
request live in a per-request BF16 ring holding the rotated, centred vectors y = H(x - mu), the domain of the lutc codes, so the
query rotation, the V mean (VMEAN) and the output rotation are unchanged.

  rows      a request's resident rows are 0 .. seq_len - 1 after the step (seq_lens of the main QSA metadata). A token's row is the
            slot the lutc store writes it to (the metadata's slot_mapping) inverted through the request's page table, the reader's own
            map (row -> block_table[request, row // PAGE] * PAGE + row % PAGE); padding (slot -1) has no row. (Until 10-16 the row was
            seq_len - q_len + j from query_start_loc: wrong for a padded request at the context cap at K 4, g2k4a.)
  tags      slot_rows[slot] = row + off of the token last written to the slot (off = the request's deleted tokens, not reduced mod
            N: row + off is invariant under compactions below the window). The reader takes a window row from the ring only if its
            slot's tag equals row + off; any other window row reads lutc and is counted (fallbacks). A stale, aliased or skipped
            slot can therefore never be read. Tags are cleared when a ring id is assigned and when a compaction deletes rows inside
            a request's window (kvring.py).
  boundary  seq_len - 1 - N: rows above it read the ring, rows at or below it read lutc. One boundary per request per step (verify
            rows and prefill chunks share it: the window is the request's last N rows after the step).
  ring slot ring_id * R + (row + off) % R, R = N + slack slots per request (10-16, the stale-read fix: a step the scheduler makes
            shorter than its predecessor leaves up to K rows written beyond the new frontier; with R = N they would hold the slots of
            the newly exposed window-bottom rows (aliased fallbacks); with R >= N + K a window row and a row up to R - N beyond the
            frontier never share a slot); ring_meta int32 [max requests, 3] = (ring_id, off, floor) per request: off = rows evicted
            so far (full count) (a compaction that renumbers the rows below the window keeps every window row in its slot); floor (default -1) lifts
            the read boundary to max(seq_len - 1 - N, floor) after an event that rewrote window rows outside the stores (promptmove
            overlap); ring_id -1 = no ring: every row reads lutc.
  write     a token's row is written to the ring iff it has a row, row > boundary (IE's mask: a chunk longer than N would otherwise
            race several rows onto one slot) and row < seq_len, from the same FP32 y the lutc store quantizes (same centring and rotation
            arithmetic).
  read      a per-row pass splits each query row's selected indices: the window rows compacted (<= N, list order) and the list
            with them set to -1. kvlut's reader (a verbatim copy whose one-split epilogue also stores the lse) reads the masked
            list with production's split choice and merge, so a row without window rows ends bitwise as today. For rows with
            window rows a last kernel attends over them from the BF16 ring (no decode; VMEAN like every lutc partial) and merges
            in FP32 with the lutc part (from the FP32 split partials, or from the direct result and its lse when the reader ran
            one split: then the lutc part was already rounded to the output dtype, as for big prefill batches).
"""
import torch
import triton
import triton.language as tl

from .kvlut_kernels import _h16, _poly_from_bits, _decode_e2m1, launch_profile, _views


@triton.jit
def _row_from_slot(slot_mapping, block_table, token, request, stride_bt, num_tables, seq_len, W: tl.constexpr,
                   BLOCK_W: tl.constexpr, PAGE: tl.constexpr):
    # The token's resident row: its lutc slot (slot_mapping) inverted through the step's batch-ordered page table (the attention
    # metadata's, which the slot mapping was computed from), the reader's own map row -> block_table[request, row // PAGE] * PAGE
    # + row % PAGE. Only the request's valid prefix ceil(seq_len / PAGE) is searched (IE: stale ids past it may name a page freed
    # and reallocated). found = 0 for padding (slot < 0), for a request without a page-table row and for a slot outside the prefix.
    s = tl.load(slot_mapping + token).to(tl.int64)
    s = tl.where(request < num_tables, s, -1)
    page = tl.where(s >= 0, s // PAGE, -2)
    within = tl.where(s >= 0, s % PAGE, 0)
    cols = tl.arange(0, BLOCK_W)
    npages = (seq_len + PAGE - 1) // PAGE
    entries = tl.load(block_table + request.to(tl.int64) * stride_bt + cols, mask=(cols < W) & (cols < npages) & (s >= 0),
                      other=-1)
    b = tl.max(tl.where(entries.to(tl.int64) == page, cols, -1), axis=0)
    row = (b.to(tl.int64) * PAGE + within).to(tl.int32)
    return row, (b >= 0) & (s >= 0), s


@triton.jit
def _ring_store_kernel(key, value, ring_k, ring_v, token_to_req, query_start_loc, seq_lens, ring_meta, kmean, vmean, slot_rows,
                       slot_mapping, block_table, diag, events, stats,
                       stride_kr, stride_kh, stride_vr, stride_vh, stride_rs, stride_rh, stride_bt, num_requests, num_tables,
                       ring_rows, N: tl.constexpr, R: tl.constexpr, CENTER: tl.constexpr, SLOT_ROWS: tl.constexpr, W: tl.constexpr, BLOCK_W: tl.constexpr,
                       PAGE: tl.constexpr, DIAG: tl.constexpr, EVENTS: tl.constexpr, STATS: tl.constexpr):
    token = tl.program_id(0)
    head = tl.program_id(1)
    side = tl.program_id(2)
    request = tl.load(token_to_req + token)
    if (request >= 0) & (request < num_requests):
        q0 = tl.load(query_start_loc + request)
        q1 = tl.load(query_start_loc + request + 1)
        seq_len = tl.load(seq_lens + request)
        row, found, s = _row_from_slot(slot_mapping, block_table, token, request, stride_bt, num_tables, seq_len, W, BLOCK_W,
                                       PAGE)
        ring_id = tl.load(ring_meta + request * 3)
        off = tl.load(ring_meta + request * 3 + 1)
        slot = ring_id * R + (tl.maximum(row, 0) + off) % R
        if STATS:                                                     # production receipt counters (one layer; IE: skim list)
            if (head == 0) & (side == 0):
                if (s >= 0) & (found == 0):
                    tl.atomic_add(stats + 0, 1)
                if found & (row >= seq_len):
                    tl.atomic_add(stats + 3, 1)
        write = found & (row > seq_len - 1 - N) & (row < seq_len) & (ring_id >= 0) & (slot >= 0) & (slot < ring_rows)
        if write:
            idx = tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]
            if side == 0:
                Y = tl.load(key + token * stride_kr + head * stride_kh + idx).to(tl.float32)
                if CENTER:
                    Y = Y - tl.load(kmean + head * 256 + idx)
            else:
                Y = tl.load(value + token * stride_vr + head * stride_vh + idx).to(tl.float32)
                if CENTER:
                    Y = Y - tl.load(vmean + head * 256 + idx)
            H = _h16()
            T = tl.dot(H, Y, input_precision="ieee")
            Y = tl.dot(T, H, input_precision="ieee") * 0.0625
            offset = slot.to(tl.int64) * stride_rs + head * stride_rh + idx
            if side == 0:
                tl.store(ring_k + offset, Y.to(tl.bfloat16))
                if SLOT_ROWS and head == 0:                           # the slot's tag: row + off of the row it now holds
                    tl.store(slot_rows + slot, row + off)
            else:
                tl.store(ring_v + offset, Y.to(tl.bfloat16))
        if DIAG:                                                      # debug servers: the old arithmetic vs the slot's row
            if (head == 0) & (side == 0):
                old_row = seq_len - (q1 - q0) + (token - q0)
                pad = s < 0
                unmapped = (s >= 0) & (found == 0)
                disagree = found & (old_row != row)
                beyond = found & (row >= seq_len)
                tl.atomic_add(diag + 0, disagree.to(tl.int64))
                tl.atomic_add(diag + 1, pad.to(tl.int64))
                tl.atomic_add(diag + 2, unmapped.to(tl.int64))
                tl.atomic_add(diag + 3, beyond.to(tl.int64))
                tl.atomic_add(diag + 4, write.to(tl.int64))
                tl.atomic_add(diag + 5, 1)
                if disagree | unmapped | beyond:
                    e = tl.atomic_add(diag + 7, 1)
                    if e < EVENTS:
                        base = events + e * 8
                        tl.store(base + 0, disagree.to(tl.int32) + 2 * unmapped.to(tl.int32) + 4 * beyond.to(tl.int32))
                        tl.store(base + 1, request)
                        tl.store(base + 2, token - q0)
                        tl.store(base + 3, old_row)
                        tl.store(base + 4, row)
                        tl.store(base + 5, seq_len)
                        tl.store(base + 6, q1 - q0)
                        tl.store(base + 7, s.to(tl.int32))


def ring_store(key, value, ring_k, ring_v, token_to_req, query_start_loc, seq_lens, ring_meta, kmean=None, vmean=None,
               n=256, num_tokens=None, slot_rows=None, slot_mapping=None, block_table=None, page=None, diag=None, events=None,
               stats=None, r=None):
    """Write the window rows of this step's tokens into the BF16 ring. key / value: BF16 [tokens, kv heads, 256];
    ring_k / ring_v: BF16 [ring slots, kv heads, 256]; token_to_req int32 [tokens] (-1 = padding); query_start_loc /
    seq_lens: the main QSA metadata (int32); ring_meta int32 [requests, 3] = (ring_id, off, floor); kmean / vmean FP32 [kv heads, 256];
    slot_mapping [tokens] (the lutc store's slots), block_table int32 [requests, W] and page (rows per page) give each token's row;
    slot_rows int32 [ring slots]: tags (row + off); diag int64 [8] / events int32 [E, 8]: debug diagnostics; stats int64 [4]: the
    production counters ([0] real tokens without a row, [3] rows at or beyond seq_len, given on one layer; [1] / [2] are the
    reader's, summed over every layer) (kvring.py)."""
    t = key.shape[0] if num_tokens is None else num_tokens
    if not t:
        return
    if key.dtype != torch.bfloat16 or key.shape[-1] != 256 or ring_k.dtype != torch.bfloat16 or ring_k.shape[1:] != key.shape[1:]:
        raise ValueError('kvring: BF16 [tokens, heads, 256] K/V and a matching ring are required')
    if ring_k.stride() != ring_v.stride() or ring_k.stride(-1) != 1 or key.stride(-1) != 1 or value.stride(-1) != 1:
        raise ValueError('kvring: unsupported layout')
    if slot_mapping is None or block_table is None or page is None:
        raise ValueError('kvring: rows come from the slot mapping and the page table (10-16); both are required')
    if slot_mapping.shape[0] < t or block_table.stride(-1) != 1:
        raise ValueError('kvring: the slot mapping does not cover the tokens, or the page table is not row-contiguous')
    center = kmean is not None
    w = block_table.shape[1]
    _ring_store_kernel[(t, key.shape[1], 2)](
        key, value, ring_k, ring_v, token_to_req, query_start_loc, seq_lens, ring_meta,
        kmean if center else key, vmean if center else key, slot_rows if slot_rows is not None else token_to_req,
        slot_mapping, block_table, diag if diag is not None else seq_lens, events if events is not None else token_to_req,
        stats if stats is not None else seq_lens,
        key.stride(0), key.stride(1), value.stride(0), value.stride(1), ring_k.stride(0), ring_k.stride(1), block_table.stride(0),
        seq_lens.shape[0], block_table.shape[0], ring_k.shape[0], N=n, R=r if r is not None else n, CENTER=center, SLOT_ROWS=slot_rows is not None, W=w,
        BLOCK_W=max(16, triton.next_power_of_2(w)), PAGE=int(page), DIAG=diag is not None,
        EVENTS=events.shape[0] if events is not None else 0, STATS=stats is not None, num_warps=4)


@triton.jit
def _shadow_store_kernel(key, value, shadow_k, shadow_v, token_to_req, seq_lens, ring_meta, kmean, vmean, slot_mapping, block_table,
                         stride_kr, stride_kh, stride_vr, stride_vh, stride_rs, stride_rh, stride_bt, num_requests, num_tables,
                         SH: tl.constexpr, CENTER: tl.constexpr, W: tl.constexpr, BLOCK_W: tl.constexpr, PAGE: tl.constexpr):
    # Debug only (G2a): the same y as the ring, for EVERY row with a slot, addressed by row (ring_id * SH + row); no window mask, no
    # offset. The row is the slot's row, as in the ring store.
    token = tl.program_id(0)
    head = tl.program_id(1)
    side = tl.program_id(2)
    request = tl.load(token_to_req + token)
    if (request >= 0) & (request < num_requests):
        seq_len = tl.load(seq_lens + request)
        row, found, s = _row_from_slot(slot_mapping, block_table, token, request, stride_bt, num_tables, seq_len, W, BLOCK_W,
                                       PAGE)
        ring_id = tl.load(ring_meta + request * 3)
        if found & (row < SH) & (ring_id >= 0):
            idx = tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]
            if side == 0:
                Y = tl.load(key + token * stride_kr + head * stride_kh + idx).to(tl.float32)
                if CENTER:
                    Y = Y - tl.load(kmean + head * 256 + idx)
            else:
                Y = tl.load(value + token * stride_vr + head * stride_vh + idx).to(tl.float32)
                if CENTER:
                    Y = Y - tl.load(vmean + head * 256 + idx)
            H = _h16()
            T = tl.dot(H, Y, input_precision="ieee")
            Y = tl.dot(T, H, input_precision="ieee") * 0.0625
            offset = (ring_id * SH + row).to(tl.int64) * stride_rs + head * stride_rh + idx
            if side == 0:
                tl.store(shadow_k + offset, Y.to(tl.bfloat16))
            else:
                tl.store(shadow_v + offset, Y.to(tl.bfloat16))


def shadow_store(key, value, shadow_k, shadow_v, token_to_req, seq_lens, ring_meta, kmean, vmean, sh, num_tokens, slot_mapping,
                 block_table, page):
    if not num_tokens:
        return
    center = kmean is not None
    w = block_table.shape[1]
    _shadow_store_kernel[(num_tokens, key.shape[1], 2)](
        key, value, shadow_k, shadow_v, token_to_req, seq_lens, ring_meta,
        kmean if center else key, vmean if center else key, slot_mapping, block_table,
        key.stride(0), key.stride(1), value.stride(0), value.stride(1), shadow_k.stride(0), shadow_k.stride(1),
        block_table.stride(0), seq_lens.shape[0], block_table.shape[0], SH=sh, CENTER=center, W=w, BLOCK_W=max(16, triton.next_power_of_2(w)),
        PAGE=int(page), num_warps=4)


@triton.jit
def _debug_check_kernel(compact_ptr, count_ptr, token_to_req_ptr, ring_meta_ptr, seq_lens_ptr, expected_ptr, slot_rows_ptr,
                        ring_k_ptr, ring_v_ptr, shadow_k_ptr, shadow_v_ptr, counters_ptr, first_ptr,
                        stride_r_slot, stride_r_head, num_requests,
                        CAP: tl.constexpr, KV_HEADS: tl.constexpr, HEAD_DIM: tl.constexpr, SH: tl.constexpr, N: tl.constexpr, R: tl.constexpr):
    # Debug only (G2a, IE): (1) content: every compacted window row's ring slot equals its shadow row bitwise; (2) the kernels'
    # seq_len never exceeds the scheduler's own count (expected = num_computed + num_scheduled, staged in prepare_attn; an upper
    # bound under async spec decode: rejected drafts make it lower). Counters: [0] mismatching (row, token) pairs, [1] rows with
    # seq_len > expected (must stay 0), [2] window rows checked, [4] rows with seq_len < expected (info); first [req, token, slot, off].
    row = tl.program_id(0)
    request = tl.load(token_to_req_ptr + row)
    if (request >= 0) & (request < num_requests):
        seq_len = tl.load(seq_lens_ptr + request)
        expected = tl.load(expected_ptr + request)
        if expected >= 0:                                 # the scheduler's CPU count is an upper bound under async spec decode
            tl.atomic_add(counters_ptr + 1, (seq_len > expected).to(tl.int64))
            tl.atomic_add(counters_ptr + 4, (seq_len < expected).to(tl.int64))
        ring_id = tl.load(ring_meta_ptr + request * 3)
        off = tl.load(ring_meta_ptr + request * 3 + 1)
        count = tl.load(count_ptr + row)
        dims = tl.arange(0, HEAD_DIM)
        for j in range(0, count):
            token = tl.load(compact_ptr + row * CAP + j)
            slot = (ring_id * R + (token + off) % R).to(tl.int64)
            shadow = (ring_id * SH + token).to(tl.int64)
            tl.atomic_add(counters_ptr + 5, (tl.load(slot_rows_ptr + slot) != token + off).to(tl.int64))   # slot holds another row
            diff = 0
            for h in tl.static_range(KV_HEADS):
                rk = tl.load(ring_k_ptr + slot * stride_r_slot + h * stride_r_head + dims).to(tl.int16, bitcast=True)
                sk = tl.load(shadow_k_ptr + shadow * stride_r_slot + h * stride_r_head + dims).to(tl.int16, bitcast=True)
                rv = tl.load(ring_v_ptr + slot * stride_r_slot + h * stride_r_head + dims).to(tl.int16, bitcast=True)
                sv = tl.load(shadow_v_ptr + shadow * stride_r_slot + h * stride_r_head + dims).to(tl.int16, bitcast=True)
                diff += tl.sum((rk != sk).to(tl.int32) + (rv != sv).to(tl.int32), axis=0)
            if diff > 0:
                tl.atomic_add(counters_ptr + 0, 1)
                if tl.atomic_cas(first_ptr + 4, 0, 1) == 0:
                    tl.store(first_ptr + 0, request)
                    tl.store(first_ptr + 1, token)
                    tl.store(first_ptr + 2, slot.to(tl.int32))
                    tl.store(first_ptr + 3, off)
        tl.atomic_add(counters_ptr + 2, count.to(tl.int64))


@triton.jit
def _lut_lse_kernel(
    q_ptr, k_cache_ptr, v_cache_ptr, k_scale_ptr, v_scale_ptr, k_block_scale_ptr, v_block_scale_ptr,
    indices_ptr, block_table_ptr, token_to_req_ptr, partial_output_ptr, partial_lse_ptr, output_ptr, vmean_ptr,
    stride_q_row, stride_q_head, stride_k_block, stride_k_token, stride_k_head,
    stride_v_block, stride_v_token, stride_v_head, stride_indices_row, stride_table_req,
    stride_output_row, stride_output_head, num_rows, num_cache_blocks, num_requests,
    TOPK: tl.constexpr, PAGE_SIZE: tl.constexpr, PAGE_TABLE_WIDTH: tl.constexpr, GROUP_SIZE: tl.constexpr,
    HEAD_DIM: tl.constexpr, NUM_QUERY_HEADS: tl.constexpr, NUM_SPLITS: tl.constexpr, NUM_TILES: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, LUT: tl.constexpr, VMEAN: tl.constexpr,
) -> None:
    # kvlut_kernels._lut_sparse_splitk_kernel verbatim (same loop, same arithmetic) except that the one-split epilogue also
    # stores the lse (kvring's merge needs it); see kvlut_kernels for the reader itself.
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


@triton.jit
def _ring_compact_kernel(indices_ptr, masked_ptr, token_to_req_ptr, seq_lens_ptr, ring_meta_ptr, compact_ptr, count_ptr,
                         slot_rows_ptr, stats_ptr, alias_ptr, alias_count_ptr, sched_ptr,
                         stride_indices_row, stride_masked_row, num_requests, ring_rows,
                         TOPK: tl.constexpr, BLOCK: tl.constexpr, CAP: tl.constexpr, N: tl.constexpr, R: tl.constexpr, TAGS: tl.constexpr,
                         COUNT: tl.constexpr, ALIAS_EVENTS: tl.constexpr):
    # One program per query row: the row's selected indices above its window boundary whose ring slot holds them (tag = row + off),
    # compacted in list order (<= N of them), and the row's index list with those entries set to -1 for the unchanged lutc reader.
    # A window row whose slot holds anything else stays in the lutc list, counted (COUNT, one layer) in stats_ptr[1] (tag -1: never
    # written or cleared) or stats_ptr[2] (another row's tag: aliased).
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
                if ALIAS_EVENTS > 0:                                  # debug (10-16): one record per aliased candidate
                    hit = candidate & (tag != -1) & (tag != token + off)
                    e = tl.atomic_add(alias_count_ptr + columns * 0, hit.to(tl.int64), mask=hit)
                    ok = hit & (e < ALIAS_EVENTS)
                    base = alias_ptr + e * 8
                    sched = tl.load(sched_ptr + safe_request)
                    tl.store(base + 0, tl.zeros_like(token) + ring_id, mask=ok)
                    tl.store(base + 1, token, mask=ok)
                    tl.store(base + 2, token + off, mask=ok)
                    tl.store(base + 3, tag, mask=ok)
                    tl.store(base + 4, tl.zeros_like(token) + seq_len, mask=ok)
                    tl.store(base + 5, tl.zeros_like(token) + boundary, mask=ok)
                    tl.store(base + 6, tl.zeros_like(token) + off, mask=ok)
                    tl.store(base + 7, tl.zeros_like(token) + sched, mask=ok)
    else:
        valid = candidate
    flags = valid.to(tl.int32)
    position = tl.cumsum(flags, axis=0) - 1
    tl.store(compact_ptr + row * CAP + position, token, mask=valid & (position < CAP))
    tl.store(count_ptr + row, tl.minimum(tl.sum(flags, axis=0), CAP))
    tl.store(masked_ptr + row * stride_masked_row + columns, tl.where(valid, -1, token), mask=columns < TOPK)


@triton.jit
def _ring_merge_kernel(
    q_ptr, ring_k_ptr, ring_v_ptr, compact_ptr, count_ptr, token_to_req_ptr, ring_meta_ptr,
    partial_output_ptr, partial_lse_ptr, output_ptr, vmean_ptr,
    stride_q_row, stride_q_head, stride_r_slot, stride_r_head, stride_output_row, stride_output_head,
    num_rows, num_requests, ring_rows,
    CAP: tl.constexpr, GROUP_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr, NUM_QUERY_HEADS: tl.constexpr,
    LUT_SPLITS: tl.constexpr, BLOCK_SPLITS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    VMEAN: tl.constexpr, N: tl.constexpr, R: tl.constexpr, SHADOW_SH: tl.constexpr = 0,
) -> None:
    # Rows with compacted window rows only: attention over them from the BF16 ring (online softmax, VMEAN like every lutc
    # partial), merged with the lutc result in FP32 (from the FP32 split partials when the reader split, else from the direct
    # result and its lse) and stored over the output row. Rows without window rows are not touched (bitwise the reader's result).
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    count = tl.load(count_ptr + row)
    if count > 0:
        request = tl.load(token_to_req_ptr + row)
        safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
        ring_id = tl.load(ring_meta_ptr + safe_request * 3)
        off = tl.load(ring_meta_ptr + safe_request * 3 + 1)
        head_offsets = tl.arange(0, BLOCK_M)
        dim_offsets = tl.arange(0, HEAD_DIM)
        column_offsets = tl.arange(0, BLOCK_N)
        first_head = kv_head * GROUP_SIZE
        head_mask = head_offsets < GROUP_SIZE
        query = tl.load(
            q_ptr + row * stride_q_row + (first_head + head_offsets[:, None]) * stride_q_head + dim_offsets[None, :],
            mask=head_mask[:, None], other=0.0)
        max_value = tl.full((BLOCK_M,), -1.0e20, dtype=tl.float32)
        normalizer = tl.zeros((BLOCK_M,), dtype=tl.float32)
        accumulator = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
        softmax_scale_log2: tl.constexpr = (HEAD_DIM**-0.5) * 1.4426950408889634
        for start in range(0, count, BLOCK_N):
            columns = start + column_offsets
            valid = columns < count
            token = tl.load(compact_ptr + row * CAP + columns, mask=valid, other=0)
            if SHADOW_SH > 0:                                                  # debug (G2a): row-addressed shadow
                slot = (ring_id * SHADOW_SH + token).to(tl.int64)
            else:
                slot = (ring_id * R + (token + off) % R).to(tl.int64)
            valid &= (slot >= 0) & (slot < ring_rows)
            keys = tl.load(ring_k_ptr + slot[None, :] * stride_r_slot + kv_head * stride_r_head + dim_offsets[:, None],
                           mask=valid[None, :], other=0.0)
            values = tl.load(ring_v_ptr + slot[:, None] * stride_r_slot + kv_head * stride_r_head + dim_offsets[None, :],
                             mask=valid[:, None], other=0.0)
            scores = tl.dot(query, keys)
            scores *= softmax_scale_log2
            scores = tl.where(valid[None, :], scores, -1.0e20)
            next_max = tl.maximum(max_value, tl.max(scores, axis=1))
            alpha = tl.math.exp2(max_value - next_max)
            probabilities = tl.where(valid[None, :], tl.math.exp2(scores - next_max[:, None]), 0.0)
            accumulator = tl.dot(probabilities.to(values.dtype), values, acc=accumulator * alpha[:, None])
            normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
            max_value = next_max
        ring_has = normalizer > 0
        ring_out = tl.where(ring_has[:, None], accumulator / tl.maximum(normalizer[:, None], 1.0e-20), 0.0)
        if VMEAN:
            mean = tl.load(vmean_ptr + kv_head * HEAD_DIM + dim_offsets)
            ring_out = tl.where(ring_has[:, None], ring_out + mean[None, :], 0.0)
        ring_lse = tl.where(ring_has, max_value + tl.math.log2(tl.maximum(normalizer, 1.0e-20)), -float("inf"))
        # the lutc part (FP32): merged from the split partials, or the direct result with its lse
        if LUT_SPLITS == 1:
            lut_lse = tl.load(partial_lse_ptr + row * NUM_QUERY_HEADS + first_head + head_offsets, mask=head_mask,
                              other=-float("inf"))
            lut_out = tl.load(output_ptr + row * stride_output_row + (first_head + head_offsets[:, None]) * stride_output_head
                              + dim_offsets[None, :], mask=head_mask[:, None], other=0.0).to(tl.float32)
        else:
            split_offsets = tl.arange(0, BLOCK_SPLITS)
            split_mask = split_offsets < LUT_SPLITS
            lses = tl.load(partial_lse_ptr + (split_offsets[None, :] * num_rows + row) * NUM_QUERY_HEADS + first_head
                           + head_offsets[:, None], mask=split_mask[None, :] & head_mask[:, None], other=-float("inf"))
            lse_max = tl.max(lses, axis=1)
            lut_has = lse_max > -float("inf")
            weights = tl.where(lut_has[:, None], tl.math.exp2(lses - tl.where(lut_has, lse_max, 0.0)[:, None]), 0.0)
            denominator = tl.sum(weights, axis=1)
            lut_out = tl.zeros((BLOCK_M, HEAD_DIM), dtype=tl.float32)
            for s in tl.static_range(LUT_SPLITS):
                part = tl.load(partial_output_ptr + ((s * num_rows + row) * NUM_QUERY_HEADS + first_head
                                                     + head_offsets[:, None]) * HEAD_DIM + dim_offsets[None, :],
                               mask=head_mask[:, None], other=0.0)
                w = tl.sum(tl.where(split_offsets[None, :] == s, weights, 0.0), axis=1)
                lut_out += w[:, None] * part
            lut_out = tl.where(lut_has[:, None], lut_out / tl.maximum(denominator, 1.0e-30)[:, None], 0.0)
            lut_lse = tl.where(lut_has, lse_max + tl.math.log2(tl.maximum(denominator, 1.0e-30)), -float("inf"))
        top = tl.maximum(lut_lse, ring_lse)
        any_values = top > -float("inf")
        safe_top = tl.where(any_values, top, 0.0)
        w_lut = tl.where(lut_lse > -float("inf"), tl.math.exp2(lut_lse - safe_top), 0.0)
        w_ring = tl.where(ring_lse > -float("inf"), tl.math.exp2(ring_lse - safe_top), 0.0)
        merged = (w_lut[:, None] * lut_out + w_ring[:, None] * ring_out) / tl.maximum(w_lut + w_ring, 1.0e-30)[:, None]
        merged = tl.where(any_values[:, None], merged, 0.0)
        tl.store(output_ptr + row * stride_output_row + (first_head + head_offsets[:, None]) * stride_output_head
                 + dim_offsets[None, :], merged, mask=head_mask[:, None])


def attention(query, cache, indices, block_table, token_to_req, seq_lens, ring_k, ring_v, ring_meta, out, vmean=None, n=256,
              profile=None, ring_block_n=64, debug=None, slot_rows=None, stats=None, r=None):
    # debug may carry alias_events (int32 [E, 8]) / alias_count (int64 [1]): aliased-fallback records (2026-10-16)
    """Two-source reader: kvlut_kernels.attention's lutc reader on the rows at or below each request's boundary plus the BF16 ring
    for the rows above it, merged. query already rotated (kvrot.rotate_rows), output in the rotated domain (the caller rotates it
    back); vmean: the rotated V mean, FP32 [kv heads, 256]; slot_rows: the ring's tags (a window row is read from the ring only if its
    slot's tag is row + off); stats: int64 [4] production counters (given on every layer): [1] / [2] window rows read from lutc because
    their slot's tag was -1 / another row's."""
    (k_cache, k_blocks, k_scale), (v_cache, v_blocks, v_scale) = _views(cache)
    q = query
    if not q.shape[0]:
        return out
    group_size = q.shape[1] // k_cache.shape[2]
    block_m = triton.next_power_of_2(group_size)
    base_programs = q.shape[0] * k_cache.shape[2]
    block_n, target_splits, partial_warps, compiler_stages = launch_profile(base_programs, block_m, profile)
    num_tiles = triton.cdiv(indices.shape[1], block_n)
    lut_splits = min(1 << (num_tiles.bit_length() - 1), target_splits)            # production's own choice
    if lut_splits == 1:
        partial_output = out                                                        # unused by the one-split epilogue
        partial_lse = torch.empty((1, q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device)
    else:
        partial_output = torch.empty((lut_splits, *q.shape), dtype=torch.float32, device=q.device)
        partial_lse = torch.empty((lut_splits, q.shape[0], q.shape[1]), dtype=torch.float32, device=q.device)
    vm = vmean if vmean is not None else q
    compact = torch.empty((q.shape[0], n), dtype=torch.int32, device=q.device)
    count = torch.empty((q.shape[0],), dtype=torch.int32, device=q.device)
    masked = torch.empty_like(indices)
    tags = slot_rows is not None
    rr = r if r is not None else n                    # ring slots per request (R >= n; n = the window)
    alias = debug.get('alias_events') if debug is not None else None
    _ring_compact_kernel[(q.shape[0],)](
        indices, masked, token_to_req, seq_lens, ring_meta, compact, count,
        slot_rows if tags else token_to_req, stats if stats is not None else count,
        alias if alias is not None else token_to_req, debug['alias_count'] if alias is not None else count,
        debug['expected'] if alias is not None else seq_lens, indices.stride(0), masked.stride(0),
        seq_lens.shape[0], ring_k.shape[0], TOPK=indices.shape[1], BLOCK=triton.next_power_of_2(indices.shape[1]), CAP=n, N=n, R=rr,
        TAGS=tags, COUNT=tags and stats is not None, ALIAS_EVENTS=alias.shape[0] if (alias is not None and tags and stats is not None) else 0,
        num_warps=8)
    # rows at or below the boundary: kvlut's reader (verbatim copy; its one-split epilogue also stores the lse) on the masked list
    _lut_lse_kernel[(q.shape[0], k_cache.shape[2], lut_splits)](
        q, k_cache, v_cache, k_scale, v_scale, k_blocks, v_blocks, masked, block_table, token_to_req,
        partial_output, partial_lse, out, vm,
        q.stride(0), q.stride(1), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        v_cache.stride(0), v_cache.stride(1), v_cache.stride(2), masked.stride(0), block_table.stride(0),
        out.stride(0), out.stride(1), q.shape[0], k_cache.shape[0], block_table.shape[0],
        TOPK=indices.shape[1], PAGE_SIZE=k_cache.shape[1], PAGE_TABLE_WIDTH=block_table.shape[1],
        GROUP_SIZE=group_size, HEAD_DIM=q.shape[2], NUM_QUERY_HEADS=q.shape[1], NUM_SPLITS=lut_splits,
        NUM_TILES=num_tiles, BLOCK_M=block_m, BLOCK_N=block_n, LUT=True, VMEAN=vmean is not None,
        num_warps=partial_warps, num_stages=compiler_stages)
    if lut_splits > 1:
        from vllm.models.qwen3_8_flash_next.nvidia.ops.qsa import _qsa_merge_splitk_kernel
        _qsa_merge_splitk_kernel[(q.shape[0], q.shape[1])](
            partial_output, partial_lse, out, out.stride(0), out.stride(1), q.shape[0],
            HEAD_DIM=q.shape[2], NUM_QUERY_HEADS=q.shape[1], NUM_SPLITS=lut_splits,
            BLOCK_SPLITS=triton.next_power_of_2(lut_splits), num_warps=2, num_stages=1)
    if debug is not None:                                   # G2a: the same merge, addressed through the row-indexed shadow
        reference = out.clone()
        sk, sv, sh = debug['shadow_k'], debug['shadow_v'], debug['sh']
        _ring_merge_kernel[(q.shape[0], k_cache.shape[2])](
            q, sk, sv, compact, count, token_to_req, ring_meta, partial_output, partial_lse, reference, vm,
            q.stride(0), q.stride(1), sk.stride(0), sk.stride(1), reference.stride(0), reference.stride(1),
            q.shape[0], seq_lens.shape[0], sk.shape[0],
            CAP=n, GROUP_SIZE=group_size, HEAD_DIM=q.shape[2], NUM_QUERY_HEADS=q.shape[1], LUT_SPLITS=lut_splits,
            BLOCK_SPLITS=triton.next_power_of_2(max(lut_splits, 2)), BLOCK_M=block_m, BLOCK_N=ring_block_n,
            VMEAN=vmean is not None, N=n, R=rr, SHADOW_SH=sh, num_warps=4, num_stages=1)
        _debug_check_kernel[(q.shape[0],)](
            compact, count, token_to_req, ring_meta, seq_lens, debug['expected'], debug['slot_rows'], ring_k, ring_v, sk, sv,
            debug['counters'], debug['first'], ring_k.stride(0), ring_k.stride(1), seq_lens.shape[0],
            CAP=n, KV_HEADS=k_cache.shape[2], HEAD_DIM=q.shape[2], SH=sh, N=n, R=rr, num_warps=4)
    _ring_merge_kernel[(q.shape[0], k_cache.shape[2])](
        q, ring_k, ring_v, compact, count, token_to_req, ring_meta, partial_output, partial_lse, out, vm,
        q.stride(0), q.stride(1), ring_k.stride(0), ring_k.stride(1), out.stride(0), out.stride(1),
        q.shape[0], seq_lens.shape[0], ring_k.shape[0],
        CAP=n, GROUP_SIZE=group_size, HEAD_DIM=q.shape[2], NUM_QUERY_HEADS=q.shape[1], LUT_SPLITS=lut_splits,
        BLOCK_SPLITS=triton.next_power_of_2(max(lut_splits, 2)), BLOCK_M=block_m, BLOCK_N=ring_block_n,
        VMEAN=vmean is not None, N=n, R=rr, num_warps=4, num_stages=1)
    if debug is not None:                                   # [3] output elements differing between ring and shadow addressing
        debug['counters'][3:4] += (out.view(torch.int16) != reference.view(torch.int16)).sum().to(torch.int64)   # bitwise (NaN-safe)
    return out
