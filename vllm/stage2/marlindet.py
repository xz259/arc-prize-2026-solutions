"""stage2_v1 'marlindet' part (milestone 2, 2026-10-01 07:05): deterministic Marlin MoE.

W1 (Intel W4A16 experts through INC -> MarlinExperts) was not run-to-run repeatable: the same temperature-0 request gave
different prompt logprobs on every token and different greedy generations (M2_DECISIONS 06:22-07:00). Cause, measured
at kernel level (test_marlin_det.py, test_marlin_canon.py): vLLM's CUDA moe_align_block_size places each (token, slot)
entry inside its expert's segment by atomics, so the order differs on almost every call (same multiset), and Marlin's
result for a row can depend on its position in the segment (one bf16 ulp; split-K stripes). Discrete steps downstream
(the NVFP4 KV cache's per-block FP4 rounding, QSA's top-k block selection) amplify that into visible logprob changes;
our FP8 dense and lm_head tiers are weight-only (W8A16) and do not quantize activations. Production's NVFP4 path (FlashInfer CUTLASS, unfused finalize) is
deterministic by design.

This part wraps moe_align_block_size as Marlin MoE sees it and rewrites the valid entries of each expert's segment in
ascending flat (token, slot) index: the canonical order. Same blocks, per-expert multisets, expert_ids and padding (the
CUDA kernel's padding sentinels stay in place), one fixed order. No host sync (CUDA-graph safe). Applies to every
MarlinExperts layer (the Intel target's INT4 experts and the NVFP4-W4A16 drafter). Expert-parallel calls (expert_map
given) pass through unchanged.

Two implementations of the same order:
  canonical_sort    sorts the valid region by (block expert, flat index). Reference; +40-66 us per call inside CUDA
                    graphs (bench_marlindet.py, 07:20), about 5% of a 50-lane decode step at 48 layers.
  canonical         (default, 07:30) two Triton kernels: segment starts from expert_ids (first block of each expert's
                    run), then one program per expert writing its entries in flat-index order (block-wise prefix sums).
test_marlindet.py checks that both give identical arrays."""
import torch
import triton
import triton.language as tl

PINS = {'vllm.model_executor.layers.fused_moe.experts.marlin_moe':
        'f00460310b033bcb84644677e16af83773464f026c0dad15bcfe25bd28b39745'}
STATS = dict(wrapped=0, calls=0, passthrough=0)
_ORIGINAL = [None]


@triton.jit
def _segment_starts(expert_ids_ptr, ntpp_ptr, seg_start_ptr, num_blocks, block_size, BLOCK: tl.constexpr):
    b = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ntpp = tl.load(ntpp_ptr)
    valid = (b < num_blocks) & (b * block_size < ntpp)
    e = tl.load(expert_ids_ptr + b, mask=valid, other=-1)
    prev = tl.load(expert_ids_ptr + b - 1, mask=valid & (b > 0), other=-1)
    tl.store(seg_start_ptr + e, (b * block_size).to(tl.int32), mask=valid & (e != prev))


@triton.jit
def _place(topk_ids_ptr, seg_start_ptr, sorted_ids_ptr, n, BLOCK: tl.constexpr):
    e = tl.program_id(0)
    start = tl.load(seg_start_ptr + e)
    running = tl.zeros((), dtype=tl.int32)
    for off in range(0, n, BLOCK):
        idx = off + tl.arange(0, BLOCK)
        m = (tl.load(topk_ids_ptr + idx, mask=idx < n, other=-1) == e).to(tl.int32)
        pos = start + running + tl.cumsum(m, axis=0) - 1
        tl.store(sorted_ids_ptr + pos, idx.to(tl.int32), mask=m > 0)
        running += tl.sum(m, axis=0)


def canonical_sort(topk_ids, block_size, num_experts, expert_map=None, pad_sorted_ids=False, ignore_invalid_experts=False):
    s, e, n = _ORIGINAL[0](topk_ids, block_size, num_experts, expert_map, pad_sorted_ids, ignore_invalid_experts)
    if expert_map is not None:
        return s, e, n
    L = s.numel()
    S = topk_ids.numel() + 1                             # > the padding sentinel (numel)
    assert num_experts * S + L < 2 ** 31, (num_experts, S, L)
    pos = torch.arange(L, device=s.device, dtype=torch.int32)
    blk = e[(pos // block_size).long()]
    key = torch.where(pos < n, blk * S + s, num_experts * S + pos)
    return s[torch.sort(key).indices], e, n


def canonical(topk_ids, block_size, num_experts, expert_map=None, pad_sorted_ids=False, ignore_invalid_experts=False):
    s, e, n = _ORIGINAL[0](topk_ids, block_size, num_experts, expert_map, pad_sorted_ids, ignore_invalid_experts)
    STATS['calls'] += 1                                  # Python-level: eager calls and graph captures
    if expert_map is not None:
        STATS['passthrough'] += 1
        return s, e, n
    flat = topk_ids.reshape(-1)
    assert flat.dtype == torch.int32 and flat.is_contiguous() and s.dtype == torch.int32 and e.dtype == torch.int32
    seg = torch.empty(num_experts, dtype=torch.int32, device=s.device)
    nb = e.numel()
    _segment_starts[(triton.cdiv(nb, 256),)](e, n, seg, nb, block_size, BLOCK=256)
    numel = flat.numel()
    _place[(num_experts,)](flat, seg, s, numel, BLOCK=1024 if numel <= 8192 else 4096)
    return s, e, n


def patch_marlin_moe(mod):
    assert callable(mod.moe_align_block_size) and _ORIGINAL[0] is None
    _ORIGINAL[0] = mod.moe_align_block_size
    canonical.__wrapped__ = _ORIGINAL[0]
    mod.moe_align_block_size = canonical
    STATS['wrapped'] += 1


PATCHERS = {'vllm.model_executor.layers.fused_moe.experts.marlin_moe': patch_marlin_moe}
