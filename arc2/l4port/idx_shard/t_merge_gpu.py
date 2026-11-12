"""GPU test of idx_shard/merge_v2.py's Triton kernels (lever 4, 2026-10-31, ARC2 - main): local_keys_fast and merge_gathered_fast
must equal their torch twins (local_keys, merge_keys) bit for bit and the merged result must equal exact_topk on the full logits
(verbatim copy, the selection's reference). TP ranks are simulated on one GPU (column parity split; the all-gather is a concat
along dim 0). Then timing per call (CUDA events, eager and inside one CUDA graph) of v1's sort merge path and v2's key path, at the
decode-like and prefill-like row counts, without the collective. Run: runtime_python.sh t_merge_gpu.py [--quick]."""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import merge_v2 as mg  # noqa: E402


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


def make(gen, rows, columns, levels, p_ninf, dev):
    if levels:
        logits = torch.randint(0, levels, (rows, columns), generator=gen).float() * 0.25
        logits = torch.where((logits == 0) & (torch.rand(rows, columns, generator=gen) < 0.5), torch.full_like(logits, -0.0), logits)
    else:
        logits = torch.randn(rows, columns, generator=gen)
    if p_ninf:
        logits = logits.masked_fill(torch.rand(rows, columns, generator=gen) < p_ninf, -float("inf"))
    lengths = torch.randint(0, columns + 1, (rows,), generator=gen)
    lengths[0] = 0
    lengths[-1] = columns
    return logits.to(dev), lengths.to(dev)


def split(logits, lengths, k, world):
    """Per rank: local logits (padded to a common width, as the real per-rank column count is page-aligned), local top-k idx (int32)."""
    out = []
    cols = (logits.shape[1] + world - 1) // world
    for r in range(world):
        loc = logits[:, r::world]
        pad = torch.full((logits.shape[0], cols), -float("inf"), device=logits.device)
        pad[:, : loc.shape[1]] = loc
        ll = mg.local_length(lengths, r, world)
        idx = torch.empty(logits.shape[0], k, dtype=torch.long, device=logits.device)
        exact_topk(pad, ll, idx, k, cols)
        out.append((pad, idx.to(torch.int32)))
    return out


def check(gen, rows, columns, k, world, levels, p_ninf, dev):
    logits, lengths = make(gen, rows, columns, levels, p_ninf, dev)
    ref = torch.empty(rows, k, dtype=torch.long, device=dev)
    exact_topk(logits, lengths, ref, k, columns)
    parts = split(logits, lengths, k, world)
    keys_t = [mg.local_keys(p, i, r, world) for r, (p, i) in enumerate(parts)]
    keys_f = [mg.local_keys_fast(p, i, r, world) for r, (p, i) in enumerate(parts)]
    for r in range(world):
        if not torch.equal(keys_t[r], keys_f[r]):
            return "local keys differ on rank %d (%d entries)" % (r, int((keys_t[r] != keys_f[r]).sum()))
    gathered = torch.cat(keys_f, 0)
    fast = mg.merge_gathered_fast(gathered, rows, k, world)
    slow = mg.merge_keys(gathered.reshape(world, rows, k).permute(1, 0, 2).reshape(rows, world * k), k)
    if not torch.equal(fast, slow):
        return "merge differs from merge_keys in %d rows" % int((fast != slow).any(1).sum())
    if not torch.equal(fast.long(), ref):
        return "merge differs from exact_topk in %d rows" % int((fast.long() != ref).any(1).sum())
    return None


def timed(fn, reps=50):
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    start.record()
    for _ in range(reps):
        fn()
    end.record()
    torch.cuda.synchronize()
    eager = start.elapsed_time(end) * 1000 / reps
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        for _ in range(3):
            fn()
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    for _ in range(3):
        g.replay()
    torch.cuda.synchronize()
    start.record()
    for _ in range(reps):
        g.replay()
    end.record()
    torch.cuda.synchronize()
    return eager, start.elapsed_time(end) * 1000 / reps


def bench(rows, columns, k, world, dev):
    gen = torch.Generator().manual_seed(7)
    logits, lengths = make(gen, rows, columns, 0, 0.0, dev)
    parts = split(logits, lengths, k, world)
    p0, i0 = parts[0]

    def v1():
        s, g = mg.local_candidates(p0, i0, 0, world)
        packed = torch.cat([s, g.to(torch.float32)], dim=1)
        gathered = torch.cat([packed] * world, dim=1).reshape(rows, world, 2, k)    # stands in for the dim-1 all-gather's copy
        a = gathered[:, :, 0, :].reshape(rows, world * k)
        b = gathered[:, :, 1, :].reshape(rows, world * k).to(torch.long)
        return mg.merge(a, b, k)

    def v2():
        keys = mg.local_keys_fast(p0, i0, 0, world)
        return mg.merge_gathered_fast(torch.cat([keys] * world, 0), rows, k, world)

    e1, g1 = timed(v1)
    e2, g2 = timed(v2)
    print("timing rows %5d local columns %5d k %d: v1 sort merge eager %7.1f us graph %7.1f us | v2 keys eager %7.1f us graph %7.1f us"
          % (rows, p0.shape[1], k, e1, g1, e2, g2), flush=True)
    return g1, g2


def main():
    dev = torch.device("cuda")
    print("device %s, torch %s" % (torch.cuda.get_device_name(0), torch.__version__), flush=True)
    gen = torch.Generator().manual_seed(1031)
    n = 0
    quick = "--quick" in sys.argv
    for world in (2,) if quick else (2, 4):
        for rows in (1, 7, 40, 129):
            for columns in (1, 9, 513, 1024, 4097):
                for k in (64, 512):
                    for levels, p_ninf in ((0, 0.0), (3, 0.0), (5, 0.1), (2, 0.5), (0, 0.9)):
                        err = check(gen, rows, columns, k, world, levels, p_ninf, dev)
                        n += 1
                        if err:
                            print("FAIL world %d rows %d columns %d k %d levels %d p_ninf %.1f: %s" % (world, rows, columns, k, levels, p_ninf, err))
                            sys.exit(1)
    print("ALL PASS: %d GPU cases: Triton local keys == torch, Triton merge == merge_keys == exact_topk" % n, flush=True)
    for rows, columns in ((40, 15000), (80, 15000), (2048, 15000)):
        bench(rows, columns, 512, 2, dev)


if __name__ == "__main__":
    main()
