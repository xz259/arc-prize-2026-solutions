"""kvring_exact.py (2026-10-28, Claude main session 099768dc; the user: "For idea 2 ... implement it and do 4.5h run A/B"; IE's idea 2
lever, the frequency-keyed exact set; design reviewed by IE 10-28 (review74/PRECISION_LEVERS_ie_2026-10-28.md, idea 2 step 2: safe;
its three effectiveness changes are built in: rate admission, one count per request per step, pinned sinks): ARC3_S2_KVRING_EXACT = M > 0
(kvring.py, FAST reader only) adds to every request's ring of every target QSA layer M BF16 slots for rows outside the recency window.
M = 0 leaves kvring exactly as it was (none of this module runs).

Keyed by the current row, not by row + off (a change after IE's review, found while building: a compaction renumbers every row
below its deleted ranges, so row + off of a row below the window is not its original position - a key kept from the promotion would
name the row just below it after one deletion above it). Each exact entry is found through a per-request row -> slot map and is
re-keyed from the compaction plan's keep rows at every commit (rekey); a deleted row's entry is dropped.

Layout (per target QSA layer; RS = R + M data slots per request; P = ARC3_S2_KVRING_EXACT_PIN, default 64, P < M):
  data      ring_k / ring_v [max requests x RS, kv heads, 256] BF16 (kvring allocates RS instead of R): slots [0, R) of a request are the
            recency ring (ring_id * RS + (row + off) % R, kvring's slot with RS-strided requests), slots R + j (j < M) the exact set.
  pinned    j = row for rows below P, written by the store whenever such a row is stored and never displaced by a promotion (the
            attention sinks and the head of the system prompt, which a 4,096-token prefill chunk never passes through the ring).
  direct    j = P + (t % (M - P)) for a promoted row of original position t (placement only: lookups go through the map).
  ltags     int32 [max requests x R]: this layer's recency tags (row + off; kvring's shared slot_rows is not used: a promotion reads the
            old tag of the slot it is about to overwrite, which an earlier layer's store would already have replaced).
  etags     int32 [max requests x M]: the current row held by each exact slot, -1 = empty; opos: its original position (for rates).
  xmap      int16 [max requests x max rows]: row -> exact slot j, -1 = none; a hit needs xmap[row] = j and etags[j] = row.
  counts    int32 [max requests x RS]: selections per data slot, counted once per request per step (its first query row of the step;
            IE: draft rows select nearly alike, and prefill and decode then count alike); a recency slot's count restarts at 0 when a
            new row is stored into it; a promoted row keeps its window count.
Per step and layer (kvring.forward_qsa), on one stream:
  decide    per token whose row is stored into the recency ring: the slot's old row t (old tag, another row, below the new one, not
            exact already) is a candidate; its selection rate c / (now - t) bids (atomic max: rate bits, then t) for its direct slot
            iff >= RATE (ARC3_S2_KVRING_EXACT_RATE, default 0.02) and above the occupant's cx / (now - ox) (empty: 0) (IE: rate, not
            raw count, so incumbents decay); now = the new row's position.
  store     kvring's store with RS-strided data slots and this layer's ltags; a winning bid first copies its old row's K and V (every
            head) into its exact slot; rows below P are also written to their pinned slot.
  commit    winners take their slot (the occupant's map entry dropped, the new one set, the bid cleared); a row stored this step
            loses any exact entry (a rollback re-stores rows); pinned rows get their entry.
  read      kvring_fast's pack pass plus exact hits for selected rows at or below the window boundary and above the floor: compact
            entries -(R + j) - 2; the merge kernel reads entries >= 0 from the recency ring and entries <= -2 from data slot
            ring_id * RS + (-e - 2). Compact capacity N + M.
kvring clears this layer's ltags with slot_rows (ring assignment, window deletions), clears the exact state at ring assignment, and
re-keys it at every compaction commit (async eviction only: asserted). A promptmove (the system prompt's K rows rotated forward by a
position delta in the lutc pages) rotates the K copies of the exact entries in the moved block by the same increment (rerope;
2026-10-28, the user: "build the promptmove fix for the pinned rows"; before, the move cleared the whole set and the pinned rows never
came back): promptmove's own steps on the BF16 copies (kvrot's rotation undone, the K mean added, the 64 rotary dims (NeoX pairs i,
i + 32) rotated by promptmove's angles, then the mean and the rotation again); V and the rows after the block are not touched.
Consistency check (IE 10-28, after the re-key review): after every re-key, at the request's next eager step (kvring), each layer's
exact entries are compared with the software dequant of the lutc row their tag claims (the reader's own dequant code): relative RMS error
above 0.3 (a correct key reads about 0.07, another row about 1.4) = a mis-key: the entry is dropped and counted.
Counters (xstats int64 [10], summed over layers): promotions, exact hits, rows selected below the boundary and above the floor (the hit
share's denominator), refusals (a candidate at or above RATE not above the occupant), pinned writes, candidates below RATE, promotions at
bulk exits (the step's query length for the request > 16: prefill chunks), entries checked, mis-keys dropped, the checks' max relative
error x 1e6.
"""
import numpy as np
import torch
import triton
import triton.language as tl

from .kvlut_kernels import _h16, _poly_from_bits, _views, launch_profile
from .kvring_kernels import _row_from_slot
from .kvring_fast import _lut_packed_kernel, RING_BLOCK_N, RING_STAGES, RING_WARPS


@triton.jit
def _exact_decide_kernel(token_to_req, seq_lens, ring_meta, ltags, etags, opos, counts, xmap, bids, dec_slot, dec_key, dec_count,
                         xstats, slot_mapping, block_table, stride_bt, num_requests, num_tables, maxrows, rate,
                         N: tl.constexpr, R: tl.constexpr, M: tl.constexpr, P: tl.constexpr, RS: tl.constexpr,
                         W: tl.constexpr, BLOCK_W: tl.constexpr, PAGE: tl.constexpr):
    token = tl.program_id(0)
    request = tl.load(token_to_req + token)
    req_ok = (request >= 0) & (request < num_requests)
    sreq = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    seq_len = tl.load(seq_lens + sreq)
    row, found, s = _row_from_slot(slot_mapping, block_table, token, sreq, stride_bt, num_tables, seq_len, W, BLOCK_W, PAGE)
    ring_id = tl.load(ring_meta + sreq * 3)
    off = tl.load(ring_meta + sreq * 3 + 1)
    rid = tl.maximum(ring_id, 0)
    write = req_ok & found & (row > seq_len - 1 - N) & (row < seq_len) & (ring_id >= 0)
    now = tl.maximum(row, 0) + off
    srel = now % R
    t = tl.load(ltags + rid * R + srel, mask=write, other=-1)
    trow = t - off
    cand = write & (t >= 0) & (t < now) & (trow >= 0) & (trow < maxrows)
    jin = tl.load(xmap + rid.to(tl.int64) * maxrows + tl.maximum(trow, 0), mask=cand, other=-1).to(tl.int32)
    ein = tl.load(etags + rid * M + tl.maximum(jin, 0), mask=cand & (jin >= 0), other=-1)
    cand = cand & ((jin < 0) | (ein != trow))                       # not exact already (pinned rows)
    c = tl.load(counts + rid * RS + srel, mask=cand, other=0)
    rc = c.to(tl.float32) / tl.maximum(now - t, 1).to(tl.float32)
    j = P + tl.maximum(t, 0) % (M - P)
    xi = rid * M + j
    ex = tl.load(etags + xi, mask=cand, other=-1)
    ox = tl.load(opos + xi, mask=cand, other=0)
    cx = tl.load(counts + rid * RS + R + j, mask=cand, other=0)
    rx = tl.where(ex >= 0, cx.to(tl.float32) / tl.maximum(now - ox, 1).to(tl.float32), 0.0)
    bid = cand & (rc >= rate) & (rc > rx)
    key = (rc.to(tl.int32, bitcast=True).to(tl.int64) << 32) | tl.maximum(t, 0).to(tl.int64)
    if bid:
        tl.atomic_max(bids + xi, key)
    if cand & (rc < rate):
        tl.atomic_add(xstats + 5, 1)
    if cand & (rc >= rate) & (rc <= rx):
        tl.atomic_add(xstats + 3, 1)
    tl.store(dec_slot + token, tl.where(bid, xi, -1))
    tl.store(dec_key + token, tl.where(bid, key, 0))
    tl.store(dec_count + token, c)


@triton.jit
def _exact_store_kernel(key, value, ring_k, ring_v, token_to_req, seq_lens, ring_meta, kmean, vmean, ltags, counts, bids, dec_slot,
                        dec_key, slot_mapping, block_table, stats,
                        stride_kr, stride_kh, stride_vr, stride_vh, stride_rs, stride_rh, stride_bt, num_requests, num_tables,
                        ring_rows, N: tl.constexpr, R: tl.constexpr, M: tl.constexpr, P: tl.constexpr, RS: tl.constexpr,
                        CENTER: tl.constexpr, W: tl.constexpr, BLOCK_W: tl.constexpr, PAGE: tl.constexpr, STATS: tl.constexpr):
    # kvring_kernels._ring_store_kernel (production counters kept; debug diagnostics left out) with RS-strided data slots and this
    # layer's tags, after the copy of a winning bid's old row (this program's head and side) into its exact slot, plus the pinned
    # write of rows below P
    token = tl.program_id(0)
    head = tl.program_id(1)
    side = tl.program_id(2)
    request = tl.load(token_to_req + token)
    if (request >= 0) & (request < num_requests):
        seq_len = tl.load(seq_lens + request)
        row, found, s = _row_from_slot(slot_mapping, block_table, token, request, stride_bt, num_tables, seq_len, W, BLOCK_W,
                                       PAGE)
        ring_id = tl.load(ring_meta + request * 3)
        off = tl.load(ring_meta + request * 3 + 1)
        p = tl.maximum(row, 0) + off
        srel = p % R
        slot = ring_id * RS + srel
        if STATS:
            if (head == 0) & (side == 0):
                if (s >= 0) & (found == 0):
                    tl.atomic_add(stats + 0, 1)
                if found & (row >= seq_len):
                    tl.atomic_add(stats + 3, 1)
        ok = found & (row < seq_len) & (ring_id >= 0)
        write = ok & (row > seq_len - 1 - N) & (slot >= 0) & (slot < ring_rows)
        pslot = ring_id * RS + R + tl.maximum(row, 0)
        pin = ok & (row < P) & (pslot < ring_rows)
        idx = tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]
        xi = tl.load(dec_slot + token)
        if write & (xi >= 0):
            if tl.load(bids + xi) == tl.load(dec_key + token):
                dst = ring_id * RS + R + (xi - ring_id * M)
                src_offset = slot.to(tl.int64) * stride_rs + head * stride_rh + idx
                dst_offset = dst.to(tl.int64) * stride_rs + head * stride_rh + idx
                if side == 0:
                    tl.store(ring_k + dst_offset, tl.load(ring_k + src_offset))
                else:
                    tl.store(ring_v + dst_offset, tl.load(ring_v + src_offset))
        tl.debug_barrier()                                            # every thread's copy load before any store of the new row
        if write | pin:
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
            if write:
                offset = slot.to(tl.int64) * stride_rs + head * stride_rh + idx
                if side == 0:
                    tl.store(ring_k + offset, Y.to(tl.bfloat16))
                    if head == 0:
                        tl.store(ltags + ring_id * R + srel, p)
                        tl.store(counts + slot, 0)
                else:
                    tl.store(ring_v + offset, Y.to(tl.bfloat16))
            if pin:
                poffset = pslot.to(tl.int64) * stride_rs + head * stride_rh + idx
                if side == 0:
                    tl.store(ring_k + poffset, Y.to(tl.bfloat16))
                else:
                    tl.store(ring_v + poffset, Y.to(tl.bfloat16))


@triton.jit
def _exact_commit_kernel(token_to_req, query_start_loc, seq_lens, ring_meta, etags, opos, counts, xmap, bids, dec_slot, dec_key,
                         dec_count, xstats, slot_mapping, block_table, stride_bt, num_requests, num_tables, maxrows,
                         N: tl.constexpr, R: tl.constexpr, M: tl.constexpr, P: tl.constexpr, RS: tl.constexpr,
                         W: tl.constexpr, BLOCK_W: tl.constexpr, PAGE: tl.constexpr):
    token = tl.program_id(0)
    request = tl.load(token_to_req + token)
    if (request >= 0) & (request < num_requests):
        seq_len = tl.load(seq_lens + request)
        row, found, s = _row_from_slot(slot_mapping, block_table, token, request, stride_bt, num_tables, seq_len, W, BLOCK_W,
                                       PAGE)
        ring_id = tl.load(ring_meta + request * 3)
        off = tl.load(ring_meta + request * 3 + 1)
        if ring_id >= 0:
            base = ring_id.to(tl.int64) * maxrows
            ok = found & (row < seq_len) & (row < maxrows)
            if ok & (row >= P):                                        # a stored row is not an exact row (rollbacks re-store rows)
                jo = tl.load(xmap + base + row).to(tl.int32)
                if jo >= 0:
                    if tl.load(etags + ring_id * M + jo) == row:
                        tl.store(etags + ring_id * M + jo, -1)
                    tl.store(xmap + base + row, -1)
            if ok & (row < P):                                         # pinned: written by the store this step
                tl.store(etags + ring_id * M + row, row)
                tl.store(opos + ring_id * M + row, row + off)
                tl.store(counts + ring_id * RS + R + row, 0)
                tl.store(xmap + base + row, row)
                tl.atomic_add(xstats + 4, 1)
            xi = tl.load(dec_slot + token)
            if xi >= 0:
                k = tl.load(dec_key + token)
                if tl.load(bids + xi) == k:                            # won (only the winner clears the bid)
                    t = (k & 0xFFFFFFFF).to(tl.int32)
                    trow = t - off
                    ex = tl.load(etags + xi)
                    if (ex >= 0) & (ex < maxrows):
                        if tl.load(xmap + base + ex).to(tl.int32) == xi - ring_id * M:
                            tl.store(xmap + base + ex, -1)
                    tl.store(xmap + base + trow, xi - ring_id * M)
                    tl.store(etags + xi, trow)
                    tl.store(opos + xi, t)
                    tl.store(counts + ring_id * RS + R + (xi - ring_id * M), tl.load(dec_count + token))
                    tl.store(bids + xi, 0)
                    tl.atomic_add(xstats + 0, 1)
                    if tl.load(query_start_loc + request + 1) - tl.load(query_start_loc + request) > 16:
                        tl.atomic_add(xstats + 6, 1)                   # a bulk exit (prefill chunk)


@triton.jit
def _exact_check_kernel(ring_k, ring_v, k_cache_ptr, v_cache_ptr, k_scale_ptr, v_scale_ptr, k_block_scale_ptr, v_block_scale_ptr,
                        block_table_ptr, seq_lens, etags, xmap, rings, brows, xstats,
                        stride_rs, stride_rh, stride_k_block, stride_k_token, stride_k_head, stride_v_block, stride_v_token,
                        stride_v_head, stride_table_req, num_cache_blocks, maxrows, thresh,
                        R: tl.constexpr, M: tl.constexpr, RS: tl.constexpr, KVH: tl.constexpr, HEAD_DIM: tl.constexpr,
                        PAGE_SIZE: tl.constexpr, PAGE_TABLE_WIDTH: tl.constexpr):
    # one program per (checked ring, exact slot): the slot's BF16 K / V (every KV head) against the software dequant of the lutc row
    # its tag claims (kvring_fast._lut_packed_kernel's addressing and LUT dequant); relative RMS error above thresh = a mis-key
    pid = tl.program_id(0)
    c = pid // M
    j = pid % M
    ring_id = tl.load(rings + c)
    brow = tl.load(brows + c)
    r = tl.load(etags + ring_id * M + j)
    seq_len = tl.load(seq_lens + brow)
    if (r >= 0) & (r < seq_len):
        lpage = r // PAGE_SIZE
        poff = r % PAGE_SIZE
        page = tl.load(block_table_ptr + brow * stride_table_req + tl.minimum(lpage, PAGE_TABLE_WIDTH - 1))
        if (lpage < PAGE_TABLE_WIDTH) & (page >= 0) & (page < num_cache_blocks):
            d = tl.arange(0, HEAD_DIM)
            pg = page.to(tl.int64)
            slot = (ring_id * RS + R + j).to(tl.int64)
            worst = 0.0
            for h in tl.static_range(KVH):
                ks = tl.load(k_cache_ptr + pg * stride_k_block + poff * stride_k_token + h * stride_k_head + d // 2)
                vs = tl.load(v_cache_ptr + pg * stride_v_block + poff * stride_v_token + h * stride_v_head + d // 2)
                ksc = tl.load(k_scale_ptr + pg * (stride_k_block // 4) + poff * (stride_k_token // 4) + h * (stride_k_head // 4))
                vsc = tl.load(v_scale_ptr + pg * (stride_v_block // 4) + poff * (stride_v_token // 4) + h * (stride_v_head // 4))
                kb = tl.load(k_block_scale_ptr + pg * stride_k_block + poff * stride_k_token + h * stride_k_head + d // 16).to(tl.float32)
                vb = tl.load(v_block_scale_ptr + pg * stride_v_block + poff * stride_v_token + h * stride_v_head + d // 16).to(tl.float32)
                kq = _poly_from_bits(((ks.to(tl.int32) << (19 - (d % 2) * 4)) & 0x780000) | 0x41000000) * (kb * ksc)
                vq = _poly_from_bits(((vs.to(tl.int32) << (19 - (d % 2) * 4)) & 0x780000) | 0x41000000) * (vb * vsc)
                kr = tl.load(ring_k + slot * stride_rs + h * stride_rh + d).to(tl.float32)
                vr = tl.load(ring_v + slot * stride_rs + h * stride_rh + d).to(tl.float32)
                ek = tl.sqrt(tl.sum((kr - kq) * (kr - kq), axis=0) / tl.maximum(tl.sum(kr * kr, axis=0), 1e-20))
                ev = tl.sqrt(tl.sum((vr - vq) * (vr - vq), axis=0) / tl.maximum(tl.sum(vr * vr, axis=0), 1e-20))
                worst = tl.maximum(worst, tl.maximum(ek, ev))
            tl.atomic_add(xstats + 7, 1)
            tl.atomic_max(xstats + 9, (worst * 1000000.0).to(tl.int64))
            if worst > thresh:
                tl.store(etags + ring_id * M + j, -1)
                if r < maxrows:
                    if tl.load(xmap + ring_id.to(tl.int64) * maxrows + r).to(tl.int32) == j:
                        tl.store(xmap + ring_id.to(tl.int64) * maxrows + r, -1)
                tl.atomic_add(xstats + 8, 1)


@triton.jit
def _exact_pack_kernel(indices_ptr, lut_ptr, lut_count_ptr, compact_ptr, count_ptr, token_to_req_ptr, query_start_loc_ptr,
                       seq_lens_ptr, ring_meta_ptr, ltags_ptr, etags_ptr, counts_ptr, xmap_ptr, stats_ptr, xstats_ptr,
                       stride_indices_row, num_requests, maxrows,
                       TOPK: tl.constexpr, BLOCK: tl.constexpr, CAP: tl.constexpr, N: tl.constexpr, R: tl.constexpr,
                       M: tl.constexpr, RS: tl.constexpr, COUNT: tl.constexpr):
    # kvring_fast._fast_pack_kernel (window test with this layer's tags, the cleared / aliased counters, compaction, the packed lutc
    # list) plus the exact hits and the selection counts (the request's first query row of the step only)
    row = tl.program_id(0)
    request = tl.load(token_to_req_ptr + row)
    request_ok = (request >= 0) & (request < num_requests)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)
    seq_len = tl.load(seq_lens_ptr + safe_request)
    ring_id = tl.load(ring_meta_ptr + safe_request * 3)
    off = tl.load(ring_meta_ptr + safe_request * 3 + 1)
    floor = tl.load(ring_meta_ptr + safe_request * 3 + 2)
    rid = tl.maximum(ring_id, 0)
    first = row == tl.load(query_start_loc_ptr + safe_request)
    boundary = tl.maximum(seq_len - 1 - N, floor)
    columns = tl.arange(0, BLOCK)
    token = tl.load(indices_ptr + row * stride_indices_row + columns, mask=columns < TOPK, other=-1)
    inrange = request_ok & (ring_id >= 0) & (token >= 0) & (token < seq_len)
    p = tl.maximum(token, 0) + off
    srel = p % R
    cand_w = inrange & (token > boundary)
    tag = tl.load(ltags_ptr + rid * R + srel, mask=cand_w, other=-2)
    valid_w = cand_w & (tag == p)
    if COUNT:
        cleared = tl.sum((cand_w & (tag == -1)).to(tl.int64), axis=0)
        aliased = tl.sum((cand_w & (tag != -1) & (tag != p)).to(tl.int64), axis=0)
        if cleared > 0:
            tl.atomic_add(stats_ptr + 1, cleared)
        if aliased > 0:
            tl.atomic_add(stats_ptr + 2, aliased)
    cand_x = inrange & (token <= boundary) & (token > floor) & (token < maxrows)
    j = tl.load(xmap_ptr + rid.to(tl.int64) * maxrows + tl.maximum(token, 0), mask=cand_x, other=-1).to(tl.int32)
    etag = tl.load(etags_ptr + rid * M + tl.maximum(j, 0), mask=cand_x & (j >= 0), other=-2)
    valid_x = cand_x & (j >= 0) & (etag == token)
    if first:
        ones = tl.full([BLOCK], 1, tl.int32)
        tl.atomic_add(counts_ptr + rid * RS + srel, ones, mask=valid_w)
        tl.atomic_add(counts_ptr + rid * RS + R + tl.maximum(j, 0), ones, mask=valid_x)
    fw = valid_w.to(tl.int32)
    fx = valid_x.to(tl.int32)
    nw = tl.sum(fw, axis=0)
    nx = tl.sum(fx, axis=0)
    pos_w = tl.cumsum(fw, axis=0) - 1
    tl.store(compact_ptr + row * CAP + pos_w, token, mask=valid_w & (pos_w < CAP))
    pos_x = nw + tl.cumsum(fx, axis=0) - 1
    tl.store(compact_ptr + row * CAP + pos_x, -(R + j) - 2, mask=valid_x & (pos_x < CAP))
    tl.store(count_ptr + row, tl.minimum(nw + nx, CAP))
    if COUNT:
        ncx = tl.sum(cand_x.to(tl.int64), axis=0)
        if ncx > 0:
            tl.atomic_add(xstats_ptr + 2, ncx)
        if nx > 0:
            tl.atomic_add(xstats_ptr + 1, nx.to(tl.int64))
    keep = (columns < TOPK) & (token >= 0) & (fw == 0) & (fx == 0)
    kept = keep.to(tl.int32)
    kpos = tl.cumsum(kept, axis=0) - 1
    nkept = tl.sum(kept, axis=0)
    tl.store(lut_ptr + row * TOPK + kpos, token, mask=keep)
    tl.store(lut_ptr + row * TOPK + columns, tl.full([BLOCK], -1, tl.int32), mask=(columns >= nkept) & (columns < TOPK))
    tl.store(lut_count_ptr + row, nkept)


@triton.jit
def _exact_merge_kernel(
    q_ptr, ring_k_ptr, ring_v_ptr, compact_ptr, count_ptr, token_to_req_ptr, ring_meta_ptr,
    partial_output_ptr, partial_lse_ptr, output_ptr, vmean_ptr,
    stride_q_row, stride_q_head, stride_r_slot, stride_r_head, stride_output_row, stride_output_head,
    num_rows, num_requests, ring_rows,
    CAP: tl.constexpr, GROUP_SIZE: tl.constexpr, HEAD_DIM: tl.constexpr, NUM_QUERY_HEADS: tl.constexpr,
    LUT_SPLITS: tl.constexpr, BLOCK_SPLITS: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
    VMEAN: tl.constexpr, R: tl.constexpr, RS: tl.constexpr,
) -> None:
    # kvring_kernels._ring_merge_kernel (no debug shadow) over compacted entries from both regions: e >= 0 a window row (recency
    # slot), e <= -2 an exact slot R + j = -e - 2
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
            entry = tl.load(compact_ptr + row * CAP + columns, mask=valid, other=0)
            slot = tl.where(entry >= 0, ring_id * RS + (tl.maximum(entry, 0) + off) % R, ring_id * RS + (-entry - 2)).to(tl.int64)
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


class ExactState:
    """One target QSA layer's exact-set state (kvring allocates one per ring, before any CUDA-graph capture) and the shared step
    buffers (bids, the per-token decisions, xstats)."""

    def __init__(self, max_reqs, maxrows, n, r, m, p, rate, device, shared=None):
        assert 0 < p < m < 32768 and n > 0 and r >= n
        self.max_reqs, self.maxrows, self.n, self.r, self.m, self.p, self.rate = max_reqs, maxrows, n, r, m, p, float(rate)
        self.rs = r + m
        self.ltags = torch.full((max_reqs * r,), -1, dtype=torch.int32, device=device)
        self.etags = torch.full((max_reqs * m,), -1, dtype=torch.int32, device=device)
        self.opos = torch.zeros((max_reqs * m,), dtype=torch.int32, device=device)
        self.counts = torch.zeros((max_reqs * self.rs,), dtype=torch.int32, device=device)
        self.xmap = torch.full((max_reqs * maxrows,), -1, dtype=torch.int16, device=device)
        self.pending = set()                     # ring ids re-keyed / re-roped since their last consistency check (kvring)
        self.kmean = None                        # the layer's K mean FP32 [heads, 256] (kvring sets it at the first forward)
        if shared is None:
            tokens = 16384
            shared = dict(bids=torch.zeros((max_reqs * m,), dtype=torch.int64, device=device),
                          dec_slot=torch.full((tokens,), -1, dtype=torch.int32, device=device),
                          dec_key=torch.zeros((tokens,), dtype=torch.int64, device=device),
                          dec_count=torch.zeros((tokens,), dtype=torch.int32, device=device), tokens=tokens,
                          xstats=torch.zeros((10,), dtype=torch.int64, device=device))      # used when the caller gives none
        self.shared = shared

    def nbytes(self):
        return sum(t.numel() * t.element_size() for t in (self.ltags, self.etags, self.opos, self.counts, self.xmap))

    def clear_tags(self, rid):
        self.ltags[rid * self.r:(rid + 1) * self.r].fill_(-1)

    def clear_exact(self, rid):
        self.pending.discard(rid)
        m = self.m
        self.etags[rid * m:(rid + 1) * m].fill_(-1)
        self.opos[rid * m:(rid + 1) * m].fill_(0)
        self.counts[rid * self.rs:(rid + 1) * self.rs].fill_(0)
        self.xmap[rid * self.maxrows:(rid + 1) * self.maxrows].fill_(-1)
        self.shared['bids'][rid * m:(rid + 1) * m].fill_(0)

    def rerope(self, rid, k, cos, sin, ring_k):
        """A promptmove of request ring rid's leading block (rows [0, k)): rotate the K copies of the exact entries with row < k by the
        angles cos / sin FP32 [32] (the move's increment, promptmove._angles), as promptmove._move rotates the lutc rows. Returns the
        number of entries rotated."""
        from .kvrot import rotate_rows
        m = self.m
        e = self.etags[rid * m:(rid + 1) * m]
        sel = torch.nonzero((e >= 0) & (e < k)).flatten()
        if not sel.numel():
            return 0
        slots = rid * self.rs + self.r + sel
        x = rotate_rows(ring_k[slots].float().contiguous())          # the stored domain -> K - mean, [n, heads, 256] FP32
        if self.kmean is not None:
            x = x + self.kmean[None]
        a = x[..., :32].clone()
        b = x[..., 32:64].clone()
        x[..., :32] = a * cos - b * sin
        x[..., 32:64] = a * sin + b * cos
        if self.kmean is not None:
            x = x - self.kmean[None]
        ring_k[slots] = rotate_rows(x.contiguous()).to(torch.bfloat16)
        return int(sel.numel())

    def rekey(self, rid, keep, old_rows):
        """A compaction of request ring rid: new row i <- old row keep[i] (keep: sorted int64 device tensor of the kept old rows;
        rows from old_rows on, appended after the plan, move down by the deleted count). Entries of deleted rows are dropped; the
        row -> slot map is rebuilt; data, opos and counts stay in their slots."""
        m, mr = self.m, self.maxrows
        e = self.etags[rid * m:(rid + 1) * m].to(torch.int64)
        nk = keep.numel()
        deleted = old_rows - nk
        valid = e >= 0
        below = valid & (e < old_rows)
        if nk:
            pos = torch.searchsorted(keep, e.clamp(min=0))
            hit = below & (pos < nk) & (keep[pos.clamp(max=nk - 1)] == e)
        else:
            pos = torch.zeros_like(e)
            hit = torch.zeros_like(valid)
        beyond = valid & (e >= old_rows)
        new = torch.where(hit, pos, torch.where(beyond, e - deleted, torch.full_like(e, -1)))
        new = torch.where((new >= 0) & (new < mr), new, torch.full_like(new, -1))
        self.etags[rid * m:(rid + 1) * m] = new.to(torch.int32)
        xm = self.xmap[rid * mr:(rid + 1) * mr]
        xm.fill_(-1)
        ok = new >= 0
        xm[new[ok]] = torch.arange(m, device=e.device, dtype=torch.int64)[ok].to(torch.int16)

    def store(self, key, value, ring_k, ring_v, token_to_req, seq_lens, ring_meta, kmean, vmean, slot_mapping, block_table, page,
              query_start_loc, stats=None, xstats=None):
        """kvring_kernels.ring_store's inputs (stats: the production counters, given on one layer) plus xstats int64 [10]: the
        exact-set decisions, the store with the promotion copies and pinned writes, the commit."""
        t = key.shape[0]
        if not t:
            return
        sh = self.shared
        if t > sh['tokens']:
            raise ValueError('kvring_exact: %d tokens exceed the decision buffers (%d)' % (t, sh['tokens']))
        if key.dtype != torch.bfloat16 or key.shape[-1] != 256 or ring_k.dtype != torch.bfloat16 or ring_k.shape[1:] != key.shape[1:]:
            raise ValueError('kvring_exact: BF16 [tokens, heads, 256] K/V and a matching ring are required')
        if ring_k.stride() != ring_v.stride() or ring_k.stride(-1) != 1 or key.stride(-1) != 1 or value.stride(-1) != 1:
            raise ValueError('kvring_exact: unsupported layout')
        if slot_mapping.shape[0] < t or block_table.stride(-1) != 1 or ring_k.shape[0] != self.max_reqs * self.rs:
            raise ValueError('kvring_exact: slot mapping / page table / ring size mismatch')
        w = block_table.shape[1]
        bw = max(16, triton.next_power_of_2(w))
        xs = xstats if xstats is not None else sh['xstats']
        common = dict(N=self.n, R=self.r, M=self.m, P=self.p, RS=self.rs, W=w, BLOCK_W=bw, PAGE=int(page))
        _exact_decide_kernel[(t,)](
            token_to_req, seq_lens, ring_meta, self.ltags, self.etags, self.opos, self.counts, self.xmap, sh['bids'], sh['dec_slot'],
            sh['dec_key'], sh['dec_count'], xs, slot_mapping, block_table, block_table.stride(0), seq_lens.shape[0],
            block_table.shape[0], self.maxrows, self.rate, num_warps=4, **common)
        center = kmean is not None
        _exact_store_kernel[(t, key.shape[1], 2)](
            key, value, ring_k, ring_v, token_to_req, seq_lens, ring_meta, kmean if center else key, vmean if center else key,
            self.ltags, self.counts, sh['bids'], sh['dec_slot'], sh['dec_key'], slot_mapping, block_table,
            stats if stats is not None else seq_lens,
            key.stride(0), key.stride(1), value.stride(0), value.stride(1), ring_k.stride(0), ring_k.stride(1), block_table.stride(0),
            seq_lens.shape[0], block_table.shape[0], ring_k.shape[0], CENTER=center, STATS=stats is not None, num_warps=4, **common)
        _exact_commit_kernel[(t,)](
            token_to_req, query_start_loc, seq_lens, ring_meta, self.etags, self.opos, self.counts, self.xmap, sh['bids'], sh['dec_slot'],
            sh['dec_key'], sh['dec_count'], xs, slot_mapping, block_table, block_table.stride(0), seq_lens.shape[0],
            block_table.shape[0], self.maxrows, num_warps=1, **common)

    def check(self, cache, block_table, seq_lens, brows, rids, ring_k, ring_v, xstats, thresh=0.3):
        """The consistency check of rings rids (batch rows brows of this step's block table and seq_lens): every exact entry's BF16 rows
        against the lutc rows its tag names; mis-keys (relative RMS error above thresh) dropped and counted in xstats [7] / [8] / [9]."""
        if not rids:
            return
        (k_cache, k_blocks, k_scale), (v_cache, v_blocks, v_scale) = _views(cache)
        rings = torch.tensor(list(rids), dtype=torch.int32).to(ring_k.device)
        br = torch.tensor(list(brows), dtype=torch.int32).to(ring_k.device)
        _exact_check_kernel[(len(rids) * self.m,)](
            ring_k, ring_v, k_cache, v_cache, k_scale, v_scale, k_blocks, v_blocks, block_table, seq_lens, self.etags, self.xmap, rings,
            br, xstats, ring_k.stride(0), ring_k.stride(1), k_cache.stride(0), k_cache.stride(1), k_cache.stride(2), v_cache.stride(0),
            v_cache.stride(1), v_cache.stride(2), block_table.stride(0), k_cache.shape[0], self.maxrows, float(thresh),
            R=self.r, M=self.m, RS=self.rs, KVH=ring_k.shape[1], HEAD_DIM=ring_k.shape[2], PAGE_SIZE=k_cache.shape[1],
            PAGE_TABLE_WIDTH=block_table.shape[1], num_warps=4)

    def attention(self, query, cache, indices, block_table, token_to_req, query_start_loc, seq_lens, ring_k, ring_v, ring_meta, out,
                  vmean=None, profile=None, stats=None, xstats=None):
        """kvring_fast.attention with the exact hits: query already rotated, output in the rotated domain."""
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
        cap = self.n + self.m
        lut = torch.empty((q.shape[0], topk), dtype=torch.int32, device=q.device)
        lut_count = torch.empty((q.shape[0],), dtype=torch.int32, device=q.device)
        compact = torch.empty((q.shape[0], cap), dtype=torch.int32, device=q.device)
        count = torch.empty((q.shape[0],), dtype=torch.int32, device=q.device)
        count_on = stats is not None and xstats is not None
        _exact_pack_kernel[(q.shape[0],)](
            indices, lut, lut_count, compact, count, token_to_req, query_start_loc, seq_lens, ring_meta, self.ltags, self.etags,
            self.counts, self.xmap, stats if count_on else count, xstats if count_on else count, indices.stride(0),
            seq_lens.shape[0], self.maxrows, TOPK=topk, BLOCK=triton.next_power_of_2(topk), CAP=cap, N=self.n, R=self.r,
            M=self.m, RS=self.rs, COUNT=count_on, num_warps=8)
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
        _exact_merge_kernel[(q.shape[0], k_cache.shape[2])](
            q, ring_k, ring_v, compact, count, token_to_req, ring_meta, partial_output, partial_lse, out, vm,
            q.stride(0), q.stride(1), ring_k.stride(0), ring_k.stride(1), out.stride(0), out.stride(1),
            q.shape[0], seq_lens.shape[0], ring_k.shape[0],
            CAP=cap, GROUP_SIZE=group_size, HEAD_DIM=q.shape[2], NUM_QUERY_HEADS=q.shape[1], LUT_SPLITS=lut_splits,
            BLOCK_SPLITS=triton.next_power_of_2(max(lut_splits, 2)), BLOCK_M=block_m, BLOCK_N=RING_BLOCK_N,
            VMEAN=vmean is not None, R=self.r, RS=self.rs, num_warps=RING_WARPS, num_stages=RING_STAGES)
        return out


def keep_tensor(keep_rows, device):
    """The compaction plan's kept old rows as a sorted int64 device tensor."""
    keep = np.asarray(keep_rows, dtype=np.int64)
    if keep.size > 1 and not (np.diff(keep) > 0).all():
        raise ValueError('kvring_exact: the keep plan is not strictly increasing')
    return torch.from_numpy(keep).to(device)
