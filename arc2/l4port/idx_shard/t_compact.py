"""CPU test of l4prod/l4port_idxshard_v2.py's sharded compaction (lever 4, 2026-10-31, ARC2 - main) against production's own
eviction_storage.compact_cache_rows_ and eviction_validation.stored_rows_digest on the full (unsharded) cache.

TP ranks are threads; vllm.distributed is a stub whose all-gather is a barrier exchange (concat along dim, rank order, as vLLM's).
Each case: random BF16 bit patterns (NaN payloads and signed zeros included) in a full cache [pages, S, 1, 128]; rank r's sharded
cache [pages, S, 1, 128 / w] holds rows i % w == r of every page at local row i // w; random request pages (unique, non-null, any
order) and kept rows (all, prefix, suffix, sparse, single, none moved). Pass: every rank's whole cache equals the parity slice of
production's compacted full cache, byte for byte (pages outside the plan and stale rows included); the sharded digests equal
production's on the full cache before and after; the receipts' row counts agree. Also: an unregistered cache of sharded shape
refuses. Run: python3 -I t_compact.py /path/to/runs/dense_fp8_v1 (the directory holding eviction_storage.py, eviction_validation.py)."""
import os
import sys
import threading
import types

import numpy as np
import torch

torch.set_num_threads(1)
HERE = os.path.dirname(os.path.abspath(__file__))
PROD = os.path.abspath(sys.argv[1])
sys.path.insert(0, PROD)
sys.path.insert(0, os.path.join(HERE, "..", "l4prod"))
os.environ["ARC3_L4PORT_IDX_SHARD"] = "1"

TLS = threading.local()


class Exchange:
    def __init__(self, world):
        self.world, self.barrier, self.slots = world, threading.Barrier(world), [None] * world

    def all_gather(self, rank, tensor, dim):
        self.slots[rank] = tensor.clone()
        self.barrier.wait()
        out = torch.cat(self.slots, dim=dim)
        self.barrier.wait()
        return out


class _Group:
    device_communicator = None

    @property
    def world_size(self):
        return TLS.world


dist = types.ModuleType("vllm.distributed")
dist.get_tensor_model_parallel_rank = lambda: TLS.rank
dist.get_tensor_model_parallel_world_size = lambda: TLS.world
dist.tensor_model_parallel_all_gather = lambda t, dim=-1: TLS.ex.all_gather(TLS.rank, t, dim)
dist.get_tp_group = lambda: _Group()
vllm = types.ModuleType("vllm")
vllm.distributed = dist
sys.modules["vllm"] = vllm
sys.modules["vllm.distributed"] = dist

import eviction_storage as es  # noqa: E402
import eviction_validation as ev  # noqa: E402
import l4port_idxshard_v2 as sh  # noqa: E402

assert sh.ON
ORIG_COMPACT, ORIG_DIGEST = es.compact_cache_rows_, ev.stored_rows_digest
sh._patch_storage(es)
sh._patch_validation(ev)


def bits(gen, shape):
    raw = torch.randint(-32768, 32768, shape, generator=gen, dtype=torch.int32).to(torch.int16)
    raw.view(-1)[::97] = 0x7FC1           # a NaN payload
    raw.view(-1)[5::89] = -32768          # -0.0
    return raw.view(torch.bfloat16)


def keep_sets(gen, old_rows):
    full = list(range(old_rows))
    sets = [full, full[: max(1, old_rows // 2)], full[old_rows // 3:], [old_rows - 1]]
    for p in (0.1, 0.5, 0.9):
        mask = torch.rand(old_rows, generator=gen) < p
        sets.append([i for i in full if mask[i]] or [old_rows - 1])
    gap = full[:3] + full[3 + old_rows // 4:]
    sets.append(gap)
    return sets


def run_ranks(world, fn):
    ex, out, errs = Exchange(world), [None] * world, []

    def body(r):
        TLS.rank, TLS.world, TLS.ex = r, world, ex
        try:
            out[r] = fn(r)
        except BaseException as e:      # noqa: BLE001
            errs.append((r, repr(e)))
            ex.barrier.abort()

    threads = [threading.Thread(target=body, args=(r,)) for r in range(world)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    if errs:
        raise RuntimeError("rank failure: %s" % errs)
    return out


def case(gen, world, S, pages_total, n_pages, old_rows, keep):
    full = bits(gen, (pages_total, S, 1, 128))
    blocks = [int(b) for b in (torch.randperm(pages_total - 1, generator=gen)[:n_pages] + 1)]
    plan = es.page_compaction(blocks, keep, old_rows=old_rows, storage_block_size=S)
    ref = full.clone()
    rec_ref = ORIG_COMPACT(ref, plan, token_axis=1)
    d_before_ref = ORIG_DIGEST(full, plan, token_axis=1)
    d_after_ref = ORIG_DIGEST(ref, plan, token_axis=1, after=True)
    caches = []
    for r in range(world):
        c = torch.empty((pages_total, S, 1, 128 // world), dtype=torch.bfloat16)
        c.view(pages_total, S // world, 1, 128)[:] = full[:, r::world]
        sh._SHARDED.add(c.data_ptr())
        caches.append(c)

    def per_rank(r):
        before = ev.stored_rows_digest(caches[r], plan, token_axis=1)
        rec = es.compact_cache_rows_(caches[r], plan, token_axis=1, index_cache={})
        after = ev.stored_rows_digest(caches[r], plan, token_axis=1, after=True)
        return before, rec, after

    results = run_ranks(world, per_rank)
    for c in caches:                     # freed CPU storage can be reused by a later tensor: keep the registry exact
        sh._SHARDED.discard(c.data_ptr())
    for r, (before, rec, after) in enumerate(results):
        got = caches[r].view(pages_total, S // world, 1, 128).view(torch.int16)
        want = ref[:, r::world].contiguous().view(torch.int16)
        if not torch.equal(got, want):
            bad = (got != want).reshape(pages_total, S // world, -1).any(-1).nonzero()[:4].tolist()
            return "rank %d bytes differ at (page, local row) %s" % (r, bad)
        if before["sha256"] != d_before_ref["sha256"] or after["sha256"] != d_after_ref["sha256"]:
            return "rank %d digest differs from production's on the full cache" % r
        if before != after:
            return "rank %d before/after digests differ" % r
        for key in ("copied_rows", "retained_rows", "retained_blocks", "releasable_blocks", "row_bytes"):
            if rec[key] != rec_ref[key]:
                return "rank %d receipt %s %s vs production %s" % (r, key, rec[key], rec_ref[key])
    return None


def main():
    gen = torch.Generator().manual_seed(1031)
    n = 0
    for world in (2, 4):
        for S in (8, 16, 64):
            for n_pages in (1, 3, 9):
                for frac in (1.0, 0.6):
                    old_rows = max(1, int(n_pages * S * frac))
                    for keep in keep_sets(gen, old_rows):
                        err = case(gen, world, S, 23, n_pages, old_rows, keep)
                        n += 1
                        if err:
                            print("FAIL world %d S %d pages %d old_rows %d kept %d: %s" % (world, S, n_pages, old_rows, len(keep), err))
                            sys.exit(1)
    # an unregistered cache of sharded shape must refuse (registry and shape disagree)
    assert not sh._SHARDED, len(sh._SHARDED)
    stray = torch.zeros((4, 8, 1, 64), dtype=torch.bfloat16)
    plan = es.page_compaction([1, 2], [0, 5, 9], old_rows=12, storage_block_size=8)

    def stray_rank(r):
        try:
            es.compact_cache_rows_(stray if r == 0 else stray.clone(), plan, token_axis=1)
        except RuntimeError as e:
            return str(e)
        return None

    msgs = run_ranks(2, stray_rank)
    assert all(m and "never used" in m for m in msgs), msgs
    # the main KV layout (uint8, token axis 2) and the unsharded BF16 [.., 128] layout take production's path
    TLS.rank, TLS.world = 0, 2
    main_kv = torch.randint(0, 256, (5, 2, 8, 296), generator=gen, dtype=torch.int32).to(torch.uint8)
    ref = main_kv.clone()
    plan = es.page_compaction([3, 1], [0, 2, 3, 9, 15], old_rows=16, storage_block_size=8)
    ORIG_COMPACT(ref, plan, token_axis=2)
    es.compact_cache_rows_(main_kv, plan, token_axis=2)
    assert torch.equal(main_kv, ref)
    draft = bits(gen, (5, 8, 1, 128))
    ref = draft.clone()
    ORIG_COMPACT(ref, plan, token_axis=1)
    es.compact_cache_rows_(draft, plan, token_axis=1)
    assert torch.equal(draft.view(torch.int16), ref.view(torch.int16))
    print("ALL PASS: %d sharded compaction cases (worlds 2 and 4, S 8/16/64, 1-9 pages, kept sets incl. none moved), bytes and "
          "digests equal production's on the full cache; stray cache refused; main KV and unsharded BF16 use production's path" % n)


if __name__ == "__main__":
    main()
