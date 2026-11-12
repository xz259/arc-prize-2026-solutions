"""Triton kernels of l4port_marlindet_ep.py (2026-11-01, ARC2 - main), loaded on first use. segment_starts is stage2_v1/marlindet.py's
_segment_starts verbatim; place_mapped is its _place with each entry's expert taken through expert_map (global -> local id, -1 for
an expert this rank does not hold), so entries of non-local experts match no program."""
import triton
import triton.language as tl


@triton.jit
def segment_starts(expert_ids_ptr, ntpp_ptr, seg_start_ptr, num_blocks, block_size, BLOCK: tl.constexpr):
    b = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ntpp = tl.load(ntpp_ptr)
    valid = (b < num_blocks) & (b * block_size < ntpp)
    e = tl.load(expert_ids_ptr + b, mask=valid, other=-1)
    prev = tl.load(expert_ids_ptr + b - 1, mask=valid & (b > 0), other=-1)
    tl.store(seg_start_ptr + e, (b * block_size).to(tl.int32), mask=valid & (e != prev))


@triton.jit
def place_mapped(topk_ids_ptr, expert_map_ptr, seg_start_ptr, sorted_ids_ptr, n, num_experts, BLOCK: tl.constexpr):
    e = tl.program_id(0)
    start = tl.load(seg_start_ptr + e)
    running = tl.zeros((), dtype=tl.int32)
    for off in range(0, n, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        g = tl.load(topk_ids_ptr + idx, mask=idx < n, other=-1)
        ok = (idx < n) & (g >= 0) & (g < num_experts)
        loc = tl.load(expert_map_ptr + g, mask=ok, other=-1)
        m = ((loc == e) & ok).to(tl.int32)
        pos = start + running + tl.cumsum(m, axis=0) - 1
        tl.store(sorted_ids_ptr + pos, idx.to(tl.int32), mask=m > 0)
        running += tl.sum(m, axis=0)
