"""CPU test of idx_shard/merge.py against exact_topk (copied verbatim from /mnt/shared/arc3-inference/arc3_determinism.py, the
selection's correctness reference): for random logits with heavy ties, -inf columns, short and empty visible prefixes, odd and even
column counts and k above and below the visible length, splitting the columns by parity over world ranks, taking each rank's local
exact_topk on its local prefix and merging must reproduce exact_topk on the full logits exactly. Run: python3 t_merge.py"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import merge  # noqa: E402


def exact_topk(logits, lengths, output, k, columns):   # verbatim from arc3_determinism.py
    output.fill_(-1)
    width = min(k, columns)
    if not width:
        return
    indices = torch.arange(columns, device=logits.device)
    for start in range(0, logits.shape[0], 128):
        end = min(start + 128, logits.shape[0])
        valid = indices[None, :] < lengths[start:end, None]
        scores = logits[start:end, :columns].masked_fill(~valid, -float('inf'))
        selected = torch.argsort(scores, dim=-1, descending=True, stable=True)[:, :width]
        selected = selected.masked_fill(selected >= lengths[start:end, None], columns)
        selected = selected.sort(dim=-1).values
        selected = selected.masked_fill(selected == columns, -1)
        output[start:end, :width].copy_(selected)


def run_case(gen, rows, columns, k, world, levels, p_ninf):
    if levels:
        logits = torch.randint(0, levels, (rows, columns), generator=gen).float() * 0.25
    else:
        logits = torch.randn(rows, columns, generator=gen)
    if p_ninf:
        logits = logits.masked_fill(torch.rand(rows, columns, generator=gen) < p_ninf, -float("inf"))
    lengths = torch.randint(0, columns + 1, (rows,), generator=gen)
    lengths[0] = 0
    lengths[-1] = columns
    ref = torch.empty(rows, k, dtype=torch.long)
    exact_topk(logits, lengths, ref, k, columns)
    scores, gidx = [], []
    for r in range(world):
        loc = logits[:, r::world].contiguous()
        cols = loc.shape[1]
        ll = merge.local_length(lengths, r, world)
        assert int(ll.max()) <= cols
        idx = torch.empty(rows, k, dtype=torch.long)
        exact_topk(loc, ll, idx, k, cols)
        s, g = merge.local_candidates(loc, idx, r, world)
        scores.append(s)
        gidx.append(g)
    out = merge.merge(torch.cat(scores, 1), torch.cat(gidx, 1), k)
    bad = (out.long() != ref).any(dim=1)
    return int(bad.sum()), ref, out, lengths


def main():
    gen = torch.Generator().manual_seed(1031)
    cases = 0
    for world in (2, 4):
        for columns in (1, 2, 7, 64, 513, 4096):
            for k in (1, 3, 64, 512):
                for levels, p_ninf in ((0, 0.0), (3, 0.0), (5, 0.1), (2, 0.5), (0, 0.9)):
                    n_bad, ref, out, lengths = run_case(gen, 37, columns, k, world, levels, p_ninf)
                    cases += 1
                    if n_bad:
                        i = int(((out.long() != ref).any(dim=1)).nonzero()[0])
                        print("FAIL world %d columns %d k %d levels %d p_ninf %.1f: %d rows differ; row %d length %d\n ref %s\n out %s" % (
                            world, columns, k, levels, p_ninf, n_bad, i, int(lengths[i]), ref[i][:12].tolist(), out[i][:12].tolist()))
                        sys.exit(1)
    # local_slots: the parity view addresses the same bytes
    S, pages, world = 8, 5, 2
    full = torch.arange(pages * S * 128, dtype=torch.float32).view(pages, S, 1, 128)
    slots = torch.tensor([-1, 0, 1, 2, 3, 9, 15, 16, 39])
    for r in range(world):
        view = full[:, r::world]                     # rank r's rows of the full layout
        ls = merge.local_slots(slots, r, world)
        for s, l in zip(slots.tolist(), ls.tolist()):
            if s < 0 or s % world != r:
                assert l == -1, (s, l, r)
            else:
                page, row = divmod(s, S)
                lp, lr = divmod(l, S // world)
                assert (page, row // world) == (lp, lr), (s, l, r)
                assert torch.equal(view[lp, lr], full[page, row])
    print("ALL PASS: %d merge cases (worlds 2 and 4, ties, -inf, prefixes 0..columns, k above and below), local_slots OK" % cases)


if __name__ == "__main__":
    main()
