"""GPU test of l4prod/l4port_idxshard_v2.py's sharded compaction (lever 4, 2026-10-31, ARC2 - main): two processes on two GPUs, one
NCCL group (torch.distributed, standing in for vLLM's TP group), each rank's sharded cache on its own GPU. The compaction runs on a
side CUDA stream (as production's async eviction does) and must equal production's eviction_storage.compact_cache_rows_ on the full
cache (on GPU), byte for byte on every page; the gathered digests must equal production's stored_rows_digest on the full cache.
Run: runtime_python.sh t_compact_gpu.py PROD_DIR (the directory holding eviction_storage.py, eviction_validation.py)."""
import os
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

HERE = os.path.dirname(os.path.abspath(__file__))


def worker(rank, world, prod, port, q):
    try:
        torch.cuda.set_device(rank)
        dist.init_process_group("nccl", init_method="tcp://127.0.0.1:%d" % port, rank=rank, world_size=world)
        sys.path.insert(0, prod)
        sys.path.insert(0, os.path.join(HERE, "..", "l4prod"))
        os.environ["ARC3_L4PORT_IDX_SHARD"] = "1"
        import eviction_storage as es
        import eviction_validation as ev
        import l4port_idxshard_v2 as sh
        orig_compact, orig_digest = es.compact_cache_rows_, ev.stored_rows_digest
        sh._patch_storage(es)
        sh._patch_validation(ev)
        sh._tp = lambda: (rank, world)

        def all_gather0(buf):
            out = torch.empty((world * buf.shape[0],) + tuple(buf.shape[1:]), dtype=buf.dtype, device=buf.device)
            dist.all_gather_into_tensor(out, buf.contiguous())
            return out

        sh._all_gather0 = all_gather0
        dev = torch.device("cuda", rank)
        side = torch.cuda.Stream(device=dev)
        n_cases = 0
        for case in range(60):
            g = torch.Generator().manual_seed(1000 + case)                  # the same full cache and plan on every rank
            S = (8, 16, 64, 796)[case % 4] if case % 4 != 3 else 796
            S = S - (S % world)
            pages = 9
            n_pages = 1 + case % 5
            raw = torch.randint(-32768, 32768, (pages, S, 1, 128), generator=g, dtype=torch.int32).to(torch.int16)
            raw.view(-1)[::97] = 0x7FC1
            raw.view(-1)[5::89] = -32768
            full = raw.view(torch.bfloat16).to(dev)
            blocks = [int(b) for b in (torch.randperm(pages - 1, generator=g)[:n_pages] + 1)]
            old_rows = max(1, int(n_pages * S * (1.0 if case % 2 else 0.7)))
            p = (0.05, 0.3, 0.6, 0.95, 1.0)[case % 5]
            keep = [i for i in range(old_rows) if float(torch.rand(1, generator=g)) < p] or [old_rows - 1]
            plan = es.page_compaction(blocks, keep, old_rows=old_rows, storage_block_size=S)
            ref = full.clone()
            orig_compact(ref, plan, token_axis=1)
            torch.cuda.synchronize(dev)
            d_before_ref = orig_digest(full, plan, token_axis=1)
            d_after_ref = orig_digest(ref, plan, token_axis=1, after=True)
            mine = torch.empty((pages, S, 1, 128 // world), dtype=torch.bfloat16, device=dev)
            mine.view(pages, S // world, 1, 128)[:] = full[:, rank::world]
            sh._SHARDED.add(mine.data_ptr())
            side.wait_stream(torch.cuda.current_stream(dev))
            with torch.cuda.stream(side):                                   # production's async path: the eviction stream
                before = ev.stored_rows_digest(mine, plan, token_axis=1)
                rec = es.compact_cache_rows_(mine, plan, token_axis=1, index_cache={})
                after = ev.stored_rows_digest(mine, plan, token_axis=1, after=True)
            torch.cuda.synchronize(dev)
            sh._SHARDED.discard(mine.data_ptr())
            got = mine.view(pages, S // world, 1, 128).view(torch.int16)
            want = ref[:, rank::world].contiguous().view(torch.int16)
            if not torch.equal(got, want):
                q.put((rank, "case %d (S %d, %d pages, %d of %d kept): bytes differ" % (case, S, n_pages, len(keep), old_rows)))
                return
            if before["sha256"] != d_before_ref["sha256"] or after["sha256"] != d_after_ref["sha256"] or before["sha256"] != after["sha256"]:
                q.put((rank, "case %d: digests differ" % case))
                return
            if rec["copied_rows"] != len(keep) - es.row_copy_schedule(plan, row_bytes=256).first_moved:
                q.put((rank, "case %d: receipt copied_rows %d" % (case, rec["copied_rows"])))
                return
            n_cases += 1
        q.put((rank, "ok %d cases" % n_cases))
    except BaseException as e:      # noqa: BLE001
        q.put((rank, "error %r" % (e,)))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def main():
    prod = os.path.abspath(sys.argv[1])
    world = 2
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    port = 29500 + os.getpid() % 1000
    procs = [ctx.Process(target=worker, args=(r, world, prod, port, q)) for r in range(world)]
    for p in procs:
        p.start()
    res = [q.get(timeout=900) for _ in procs]
    for p in procs:
        p.join(timeout=60)
    print("device %s x %d, torch %s" % (torch.cuda.get_device_name(0), world, torch.__version__))
    for r, msg in sorted(res):
        print("rank %d: %s" % (r, msg))
    ok = all(m.startswith("ok") for _, m in res)
    print("ALL PASS: sharded compaction on GPU (NCCL, side stream) == production's compaction of the full cache, bytes and digests"
          if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
