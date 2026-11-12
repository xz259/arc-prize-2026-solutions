"""stage2_v1 'idxfq' part (milestone 2, 2026-10-02; plan Revision 7 B5, diagnostic, record only): the QSA indexer's
compressed keys fake-quantized to FP8 E4M3 with a per-row FP32 scale (amax / 448, round to nearest even), in place,
right after the fused pre-indexer writes them and before the step's selection scores them. Selection therefore sees
what an FP8 compressed-key cache would hold (the FP8 indexer-key design of upstream vLLM #54890).

The indexer keeps two side caches per QSA layer (vllm/models/qwen3_8_flash_next/common/qsa_cache.py): a raw-key ring
of compress_ratio + speculative rows per request (8 rows; BF16 here, it stays BF16), and the compressed-key cache, one
normalized 128-dim row per complete group of 4 tokens (64 B per token and layer in BF16; FP8 + scale 33 B). Only the
latter carries memory, so only it is quantized. Every QSA indexer is affected, the drafter's included. The pass is pure
GPU work on the call's own slot mapping, so CUDA graph capture records it (static per launch: ARC3_S2_IDXFQ=e4m3).

Wraps indexer_qsa.qsa_pre_indexer (imported there at module level): positional arguments 13 and 14 are the compressed
cache [pages, page_size, 1, 128] and its slot mapping (-1 = no row written for that token)."""
import os

import torch
import triton
import triton.language as tl

PINS = {'vllm.models.qwen3_8_flash_next.nvidia.indexer_qsa': 'e2f398a2fe29466c9681627651ccb5b8eb2b5980445c9e44b270ff45bcb61066'}
MODE = os.environ.get('ARC3_S2_IDXFQ', 'e4m3')
STATS = dict(mode=MODE, calls=0, errors=0)


@triton.jit
def _fq_rows_kernel(cache, slots, stride_p, stride_o, stride_w, page_size, num_slots, WIDTH: tl.constexpr):
    r = tl.program_id(0)
    slot = tl.load(slots + r).to(tl.int64)
    if slot >= 0 and slot < num_slots:
        base = cache + (slot // page_size) * stride_p + (slot % page_size) * stride_o
        offs = tl.arange(0, WIDTH) * stride_w
        x = tl.load(base + offs).to(tl.float32)
        amax = tl.max(tl.abs(x), axis=0)
        s = tl.where(amax > 0, tl.div_rn(amax, 448.0), 1.0)
        q = tl.div_rn(x, s).to(tl.float8e4nv).to(tl.float32) * s
        tl.store(base + offs, q.to(tl.bfloat16))


def fake_quant_rows(cache, slots):
    """In place: the rows of a [pages, page_size, 1, width] BF16 cache named by slots (int, -1 skipped)."""
    if cache.ndim != 4 or cache.shape[2] != 1 or cache.dtype != torch.bfloat16 or slots.ndim != 1:
        raise ValueError('idxfq: unexpected compressed-cache or slot layout')
    n = slots.numel()
    if n:
        _fq_rows_kernel[(n,)](cache, slots, cache.stride(0), cache.stride(1), cache.stride(3), cache.shape[1],
                              cache.shape[0] * cache.shape[1], WIDTH=cache.shape[3], num_warps=1)


def patch_indexer(mod):
    if MODE != 'e4m3':
        raise RuntimeError('idxfq: ARC3_S2_IDXFQ must be e4m3, got %r' % MODE)
    original = mod.qsa_pre_indexer

    def qsa_pre_indexer(*args, **kwargs):
        result = original(*args, **kwargs)
        fake_quant_rows(args[13], args[14])
        STATS['calls'] += 1
        return result
    mod.qsa_pre_indexer = qsa_pre_indexer


PATCHERS = {'vllm.models.qwen3_8_flash_next.nvidia.indexer_qsa': patch_indexer}
