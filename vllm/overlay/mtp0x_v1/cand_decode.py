"""Candidate-based decode sampler for MRv2 (MTP0 test overlay; kernels from sampling/cand_sampler.py).

Replaces FlashInfer's top-k/top-p sampling after apply_sampling_params (penalties, bias, temperature already applied
to the FP32 logits): per 2,048-token block the 32 largest (value, index) keys; per row the top-k by threshold (ties
kept) and the reference top-p rule on those candidates; then Gumbel-max over the kept candidates with the stock MRv2
noise rand(randint(seed, pos), token id), i.e. exactly the distribution of vLLM's non-FlashInfer (gumbel) path under
the reference top-k/top-p rule. Rows whose ties at the k-th value exceed the 32 slots keep the 32 largest keys by
(value, token id) (flagged; counted in the receipt).
"""
import torch
import triton
import triton.language as tl

from vllm.triton_utils import tldevice
from vllm.v1.worker.gpu.sample.gumbel import tl_rand32

KMAX = 32
BLOCK = 2048


@triton.jit
def _sortable(v):
    b = v.to(tl.int32, bitcast=True)
    return b ^ ((b >> 31) & 0x7FFFFFFF)


@triton.jit
def _unsortable(k):
    b = k ^ ((k >> 31) & 0x7FFFFFFF)
    return b.to(tl.float32, bitcast=True)


@triton.jit
def _block_topk_kernel(logits_ptr, logits_stride, ckey_ptr, V, NB, BLOCK: tl.constexpr, KC: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    blk = tl.program_id(1)
    offs = blk * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(logits_ptr + row * logits_stride + offs, mask=offs < V, other=float('-inf')).to(tl.float32)
    thr = tl.min(tl.topk(x, KC), axis=0)
    gt = x > thr
    eq = x == thr
    n_gt = tl.sum(gt.to(tl.int32), axis=0)
    rank = tl.where(gt, tl.cumsum(gt.to(tl.int32), axis=0) - 1, n_gt + tl.cumsum(eq.to(tl.int32), axis=0) - 1)
    key = (_sortable(x).to(tl.int64) << 32) | offs.to(tl.int64)
    tl.store(ckey_ptr + (row * NB + blk) * KC + rank, key, mask=(gt | eq) & (rank < KC))


@triton.jit
def _select_sample_kernel(ckey_ptr, topk_ptr, topp_ptr, exp_idx_ptr, seed_ptr, pos_ptr, out_ptr, flag_ptr, NB,
                          HAS_P: tl.constexpr, NBK: tl.constexpr, KC: tl.constexpr, KMAX: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    o = tl.arange(0, NBK)
    keys = tl.load(ckey_ptr + row * NB * KC + o, mask=o < NB * KC, other=-9223372036854775807 - 1)
    top = tl.sort(tl.topk(keys, KMAX), descending=True)
    vals = _unsortable((top >> 32).to(tl.int32))
    ids = (top & 0xFFFFFFFF).to(tl.int32)
    j = tl.arange(0, KMAX)
    k = tl.load(topk_ptr + row)
    vk = tl.sum(tl.where(j == k - 1, vals, 0.0), axis=0)
    keep = vals >= vk
    overflow = tl.max(tl.where(j == KMAX - 1, vals, float('-inf')), axis=0) >= vk
    if HAS_P:
        p = tl.load(topp_ptr + row)
        vmax = tl.max(vals, axis=0)
        e = tl.where(keep, tl.exp(vals - vmax), 0.0)
        total = tl.sum(e, axis=0)
        suffix = total - tl.cumsum(e, axis=0) + e
        keep = keep & ((suffix / total > 1.0 - p) | (j == 0))
    req = tl.load(exp_idx_ptr + row).to(tl.int64)
    seed = tl.load(seed_ptr + req)
    pos = tl.load(pos_ptr + row)
    gseed = tl.randint(seed, pos)
    u = tl_rand32(gseed, ids, includes_zero=False)
    score = tl.where(keep, vals - tl.log(-tldevice.log1p(-u)), float('-inf'))
    best = tl.argmax(score, axis=0)
    tl.store(out_ptr + row, tl.sum(tl.where(j == best, ids, 0), axis=0).to(tl.int64))
    tl.store(flag_ptr + row, overflow.to(tl.int32))


class DecodeSampler:
    def __init__(self, max_rows, vocab, device):
        self.vocab = vocab
        self.nb = triton.cdiv(vocab, BLOCK)
        self.nbk = triton.next_power_of_2(self.nb * KMAX)
        self.ckey = torch.empty(max_rows * self.nb * KMAX, dtype=torch.int64, device=device)
        self.flag = torch.zeros(max_rows, dtype=torch.int32, device=device)
        self.ones = torch.ones(max_rows, dtype=torch.float32, device=device)

    def __call__(self, logits, top_k, top_p, expanded_idx_mapping, seeds, pos):
        """logits [R, V] FP32 (processed); top_k [R] int (every row <= KMAX - 1); top_p [R] or None;
        seeds [max_reqs] int64; pos [R] int64. Returns sampled [R] int64."""
        R, V = logits.shape
        assert V == self.vocab and logits.stride(-1) == 1 and R * self.nb * KMAX <= self.ckey.numel()
        out = torch.empty(R, dtype=torch.int64, device=logits.device)
        _block_topk_kernel[(R, self.nb)](logits, logits.stride(0), self.ckey, V, self.nb, BLOCK=BLOCK, KC=KMAX,
                                         num_warps=4)
        _select_sample_kernel[(R,)](self.ckey, top_k, top_p if top_p is not None else self.ones,
                                    expanded_idx_mapping, seeds, pos, out, self.flag, self.nb,
                                    HAS_P=top_p is not None, NBK=self.nbk, KC=KMAX, KMAX=KMAX, num_warps=4)
        return out
