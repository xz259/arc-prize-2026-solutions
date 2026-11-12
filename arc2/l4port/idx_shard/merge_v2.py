"""Lever 4 (the QSA indexer dedupe at TP 2; ARC2 - main, 2026-10-31): the selection's merge, kept free of vLLM so it can be tested on
the CPU against arc3_determinism.exact_topk. Layout (INDEXER_DEDUPE_DESIGN_2026-10-31.md): rank r stores the compressed blocks g with
g % world == r at local column l = g // world, so its visible local columns are the prefix [0, local_length(V, r)) of the global
prefix [0, V). Every function here is shape-static and free of host synchronization, so it can run under CUDA-graph capture."""
import torch

INVALID = 1 << 30   # the sort key of an empty candidate slot: after every real block index


def local_length(visible, rank, world):
    """Local visible prefix of rank `rank` for global visible prefix lengths `visible` (any int tensor)."""
    return torch.clamp((visible - rank + world - 1) // world, min=0)


def local_slots(slots, rank, world):
    """Compressed-cache slots (physical_page * S + row, -1 = none) -> this rank's slots in its [pages, S / world] view, -1 elsewhere.
    Needs S % world == 0, so slot % world == row % world and slot // world = page * (S / world) + row // world."""
    keep = (slots >= 0) & (slots % world == rank)
    return torch.where(keep, torch.div(slots, world, rounding_mode="floor"), torch.full_like(slots, -1))


def local_candidates(logits, idx, rank, world):
    """A local top-k (local column indices, -1 = none) -> (scores [rows, k] float32, global indices [rows, k] int64 with INVALID)."""
    if logits.shape[1] == 0:          # no local columns (static shape): no candidates from this rank
        return (torch.full(idx.shape, -float("inf"), dtype=torch.float32, device=idx.device),
                torch.full(idx.shape, INVALID, dtype=torch.long, device=idx.device))
    valid = idx >= 0
    safe = torch.where(valid, idx, torch.zeros_like(idx)).long()
    scores = logits.gather(1, safe).float()
    scores = torch.where(valid, scores, torch.full_like(scores, -float("inf")))
    gidx = torch.where(valid, safe * world + rank, torch.full_like(safe, INVALID))
    return scores, gidx


def merge(scores, gidx, k):
    """Candidates from every rank ([rows, C] each, concatenated) -> the global top-k as exact_topk writes it: block indices in
    ascending order, -1 padding last. Order: score descending, ties by the lower block index (exact_topk's stable argsort);
    empty slots (INVALID) never beat a real block, whatever their score."""
    order = torch.argsort(gidx, dim=1, stable=True)                      # index ascending ...
    s = scores.gather(1, order)
    g = gidx.gather(1, order)
    real = g < INVALID
    key = torch.where(real, s, torch.full_like(s, -float("inf")))
    order2 = torch.argsort(key, dim=1, descending=True, stable=True)    # ... then score descending: ties keep the lower index
    # among equal keys a real -inf block (lower index) stays ahead of an empty slot (INVALID sorts last in the first pass)
    top = g.gather(1, order2[:, :k])
    top = torch.sort(top, dim=1).values                                  # ascending; INVALID last
    return torch.where(top < INVALID, top, torch.full_like(top, -1)).to(torch.int32)


# ---- composite keys (v2): one int64 per candidate, (order-preserving score key, LOW_MAX - global block); larger = better ----
KEY_SHIFT = 1 << 32
LOW_MAX = (1 << 31) - 1
MASKED = -((1 << 31) - 1) * KEY_SHIFT   # below every candidate key (the -inf key is -2139095041)


def _key32(scores):
    """float32 -> int32 with the same order (non-NaN); -0.0 is first made +0.0, so it ties with +0.0 as in exact_topk."""
    s = torch.where(scores == 0, torch.zeros_like(scores), scores).contiguous()
    bits = s.view(torch.int32)
    return bits ^ ((bits >> 31) & 0x7FFFFFFF)


def local_keys(logits, idx, rank, world):
    """A local top-k (local column indices, -1 = none) -> composite keys [rows, k] int64. Empty slots: the -inf key and INVALID,
    so a real block (even at -inf) outranks them; equal scores rank the lower global block first (exact_topk's stable order)."""
    scores, gidx = local_candidates(logits, idx, rank, world)
    return _key32(scores).to(torch.int64) * KEY_SHIFT + (LOW_MAX - gidx)


def merge_keys(keys, k):
    """Composite keys of every rank ([rows, C] int64, C >= k) -> the global top-k as exact_topk writes it (ascending, -1 last)."""
    top = torch.topk(keys, k, dim=1, largest=True, sorted=False).values
    g = torch.sort(LOW_MAX - torch.remainder(top, KEY_SHIFT), dim=1).values
    return torch.where(g < INVALID, g, torch.full_like(g, -1)).to(torch.int32)


# ---- Triton versions of local_keys and merge_keys (CUDA; same results bit for bit; one program per row) ----
_TRITON = None


def _triton_kernels():
    global _TRITON
    if _TRITON is None:
        import triton
        import triton.language as tl

        @triton.jit
        def local_keys_kernel(logits, l_stride, idx, i_stride, out, rank, world, K: tl.constexpr, KP: tl.constexpr):
            row = tl.program_id(0).to(tl.int64)
            j = tl.arange(0, KP)
            m = j < K
            ix = tl.load(idx + row * i_stride + j, mask=m, other=-1).to(tl.int64)
            valid = ix >= 0
            safe = tl.where(valid, ix, 0)
            s = tl.load(logits + row * l_stride + safe, mask=m & valid, other=float("-inf")).to(tl.float32)
            s = tl.where(s == 0.0, 0.0, s)
            bits = s.to(tl.int32, bitcast=True)
            key = bits ^ ((bits >> 31) & 0x7FFFFFFF)
            g = tl.where(valid, safe * world + rank, 1073741824)
            comp = key.to(tl.int64) * 4294967296 + (2147483647 - g)
            tl.store(out + row * K + j, comp, mask=m)

        @triton.jit
        def merge_kernel(keys, rows, out, o_stride, K: tl.constexpr, KP: tl.constexpr, WORLD: tl.constexpr):
            row = tl.program_id(0)
            offs = tl.arange(0, WORLD * KP)
            r = offs // KP
            j = offs % KP
            valid = j < K
            x = tl.load(keys + (r * rows + row) * K + j, mask=valid, other=-9223372032559808512)
            xs = tl.sort(x, descending=True)
            t = tl.sum(tl.where(offs == K - 1, xs, 0), axis=0)           # the k-th largest key
            g = 2147483647 - (x - ((x >> 32) << 32))                       # low 32 bits (arithmetic shift: floor)
            g = tl.where(valid & (x >= t), g, 1073741824)                 # ties at t are only empty slots (INVALID)
            gs = tl.sort(g)
            tl.store(out + row * o_stride + offs, tl.where(gs < 1073741824, gs, -1).to(tl.int32), mask=offs < K)

        _TRITON = (triton, local_keys_kernel, merge_kernel)
    return _TRITON


def _pow2(n):
    return 1 << max(0, (int(n) - 1).bit_length())


def local_keys_fast(logits, idx, rank, world):
    """local_keys on CUDA with one Triton program per row (CPU: the torch version)."""
    if not logits.is_cuda or logits.shape[1] == 0:
        return local_keys(logits, idx, rank, world)
    _, kern, _ = _triton_kernels()
    rows, k = idx.shape
    out = torch.empty((rows, k), dtype=torch.int64, device=idx.device)
    if rows:
        kern[(rows,)](logits, logits.stride(0), idx, idx.stride(0), out, rank, world, K=k, KP=_pow2(k), num_warps=4)
    return out


def merge_gathered_fast(gathered, rows, k, world):
    """gathered [world * rows, k] (an all-gather along dim 0 of every rank's local_keys) -> merge_keys' result, on CUDA in one
    Triton program per row reading the rank blocks in place (CPU: the torch version)."""
    if not gathered.is_cuda:
        return merge_keys(gathered.reshape(world, rows, k).permute(1, 0, 2).reshape(rows, world * k), k)
    _, _, kern = _triton_kernels()
    out = torch.empty((rows, k), dtype=torch.int32, device=gathered.device)
    if rows:
        kern[(rows,)](gathered, rows, out, k, K=k, KP=_pow2(k), WORLD=world, num_warps=4)
    return out
