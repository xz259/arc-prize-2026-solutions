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
