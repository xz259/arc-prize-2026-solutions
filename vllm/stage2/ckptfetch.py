"""stage2_v1 'ckptfetch' part (milestone 2 B9 startup, 2026-10-02 19:15): bounded parallel prefetch of the safetensors
shards into the page cache while the loader works through them (the design of dfranzen's patch 5).

Measured on the pod's MooseFS/FUSE volume (18:58, shard reads with the file's page cache dropped): one reader 0.55 GiB/s,
4 / 8 / 16 readers in disjoint ranges 2.05 / 2.85 / 3.45 GiB/s. The loader reads each shard as one stream: the target's
68.5 GiB took 218.8 s of a 249.6 s start (A3's launch, 09:20). vLLM's own prefetch (load strategy 'prefetch', or
automatic only on NFS/Lustre) reads every shard at once, unbounded, which here competes with the PLE worker's 57 GB for
about 108 GiB of container memory.

Wraps weight_utils.safetensors_weights_iterator for the default strategy (None): shards in the loader's order (its
natural sort); before the loader opens shard i, shard i's needed bytes have been read by THREADS threads (pread of
disjoint pieces, 16 MiB blocks), and shard i + 1 is started in the background; never more than one shard ahead, so at
most two shards are added to the page cache at a time. Needed bytes: the data ranges of the tensors the loader will read
(the safetensors header's offsets, minus tensors should_skip_weight drops, which includes loadskip's skips), merged.
The loader is production's iterator, called once per shard with the same strategy, expert ids and options: the same
safe_open reads of the same files and tensors, so the weights are identical. Any other strategy, or a single file, is
passed through unchanged. ARC3_S2_CKPTFETCH_THREADS (default 8).

2026-10-28 (arc3-code v12; the user: "add the parallel PLE read to the next build"): ARC3_S2_CKPTFETCH_FILES = F > 1 reads up to
F shards at once, one sequential stream (16 MiB preads) per shard, in the loader's order and at most F shards ahead of it, while
the started-but-unread shards stay within ARC3_S2_CKPTFETCH_AHEAD_GB (default 48; the shard the loader waits for always starts).
Kaggle (10-28, the v11b smokes): the PLE worker's 20 shards came in one at a time at about 53 s per 5.4 GB shard (about
0.1 GB/s, nearly all of a 19-min start), while vLLM's NFS auto-prefetch, one stream per file across many files, moved the
target's 15 shards at about 0.45-0.6 GB/s; the 8-readers-on-one-shard pattern above did not speed one shard up there. The
loader and the bytes it reads are unchanged (the same per-shard production iterator), so the weights are identical. Unset or 1:
the one-shard-ahead mode above, unchanged."""
import concurrent.futures
import json
import os
import struct
import time

PINS = {'vllm.model_executor.model_loader.weight_utils': '25bb3393105f9a642b7fa58bc277865db9976748be320787f8db2e03b16fd667'}
THREADS = int(os.environ.get('ARC3_S2_CKPTFETCH_THREADS', '8'))
FILES = int(os.environ.get('ARC3_S2_CKPTFETCH_FILES', '1') or 1)               # v12: shards read concurrently (1: one ahead)
AHEAD = int(float(os.environ.get('ARC3_S2_CKPTFETCH_AHEAD_GB', '48') or 48) * (1 << 30))
BLOCK = 16 << 20
STATS = dict(threads=THREADS, iterators=0, passthrough=0, shards=0, bytes=0, prefetch_s=0.0, loader_wait_s=0.0, errors=0,
             per_pid={}, files=FILES, ahead_gb=AHEAD / (1 << 30), max_in_flight=0)


def needed_ranges(path, skip):
    """Merged (start, end) byte ranges of the tensors in a safetensors file that skip(name) keeps."""
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        header = json.loads(f.read(n))
    base = 8 + n
    spans = sorted((base + v['data_offsets'][0], base + v['data_offsets'][1])
                   for k, v in header.items() if k != '__metadata__' and not skip(k))
    merged = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], e)
        else:
            merged.append([s, e])
    return merged


def _read(path, pieces):
    fd = os.open(path, os.O_RDONLY)
    try:
        for s, e in pieces:
            while s < e:
                got = len(os.pread(fd, min(BLOCK, e - s), s))
                if not got:
                    break
                s += got
    finally:
        os.close(fd)


def prefetch(path, ranges, pool, threads):
    """Read the ranges with `threads` workers of `pool` (contiguous, nearly equal shares); returns bytes read."""
    total = sum(e - s for s, e in ranges)
    if not total:
        return 0
    share = -(-total // threads)
    jobs, cur, left = [], [], share
    for s, e in ranges:
        while s < e:
            take = min(e - s, left)
            cur.append((s, s + take))
            s += take
            left -= take
            if not left:
                jobs.append(cur)
                cur, left = [], share
    if cur:
        jobs.append(cur)
    for fut in [pool.submit(_read, path, j) for j in jobs]:
        fut.result()
    return total


def patch_weight_utils(mod):
    original = mod.safetensors_weights_iterator

    def safetensors_weights_iterator(hf_weights_files, use_tqdm_on_load, safetensors_load_strategy=None,
                                     local_expert_ids=None, **kwargs):
        files = sorted(hf_weights_files, key=mod._natural_sort_key)
        if safetensors_load_strategy is not None or len(files) < 2 or THREADS < 1:
            STATS['passthrough'] += 1
            yield from original(hf_weights_files, use_tqdm_on_load, safetensors_load_strategy, local_expert_ids,
                                **kwargs)
            return
        STATS['iterators'] += 1
        pid = str(os.getpid())
        mine = STATS['per_pid'].setdefault(pid, dict(shards=0, bytes=0, prefetch_s=0.0, loader_wait_s=0.0))
        if FILES > 1:
            yield from _multi(files, mine, original, mod, use_tqdm_on_load, safetensors_load_strategy, local_expert_ids, kwargs)
            return
        readers = concurrent.futures.ThreadPoolExecutor(THREADS, thread_name_prefix='arc3-ckptfetch')
        control = concurrent.futures.ThreadPoolExecutor(1, thread_name_prefix='arc3-ckptfetch-ctl')

        def fetch(path):
            t0 = time.perf_counter()
            try:
                got = prefetch(path, needed_ranges(path, lambda k: mod.should_skip_weight(k, local_expert_ids)),
                               readers, THREADS)
            except Exception:                                  # noqa: BLE001 (prefetch is advisory: the loader reads anyway)
                STATS['errors'] += 1
                got = 0
            dt = time.perf_counter() - t0
            for d in (STATS, mine):
                d['bytes'] += got
                d['prefetch_s'] += dt
            return got

        try:
            ahead = control.submit(fetch, files[0])
            for i, path in enumerate(files):
                t0 = time.perf_counter()
                ahead.result()
                waited = time.perf_counter() - t0
                ahead = control.submit(fetch, files[i + 1]) if i + 1 < len(files) else None
                for d in (STATS, mine):
                    d['loader_wait_s'] += waited
                    d['shards'] += 1
                yield from original([path], use_tqdm_on_load, safetensors_load_strategy, local_expert_ids, **kwargs)
        finally:
            control.shutdown(wait=False, cancel_futures=True)
            readers.shutdown(wait=False, cancel_futures=True)
            from . import install
            install.save()

    safetensors_weights_iterator.__wrapped__ = original
    mod.safetensors_weights_iterator = safetensors_weights_iterator


def _multi(files, mine, original, mod, use_tqdm_on_load, safetensors_load_strategy, local_expert_ids, kwargs):
    """v12: up to FILES shards in flight, one stream each, in the loader's order, at most FILES shards from the one the loader
    reads and (beyond that one) within AHEAD bytes of started-but-unread shards; the loader still gets one shard per call."""
    pool = concurrent.futures.ThreadPoolExecutor(FILES, thread_name_prefix='arc3-ckptfetch')
    sizes = []
    for p in files:
        try:
            sizes.append(os.path.getsize(p))
        except OSError:
            sizes.append(0)

    def fetch(path):
        t0 = time.perf_counter()
        try:
            ranges = needed_ranges(path, lambda k: mod.should_skip_weight(k, local_expert_ids))
            _read(path, ranges)
            got = sum(e - s for s, e in ranges)
        except Exception:                                      # noqa: BLE001 (prefetch is advisory: the loader reads anyway)
            STATS['errors'] += 1
            got = 0
        dt = time.perf_counter() - t0
        for d in (STATS, mine):
            d['bytes'] += got
            d['prefetch_s'] += dt
        return got

    futs, state = {}, dict(next=0, ahead=0)

    def top_up(i):
        while state['next'] < len(files) and state['next'] - i < FILES and (
                state['next'] <= i or state['ahead'] + sizes[state['next']] <= AHEAD):
            j = state['next']
            futs[j] = pool.submit(fetch, files[j])
            state['ahead'] += sizes[j]
            state['next'] += 1
        STATS['max_in_flight'] = max(STATS['max_in_flight'], state['next'] - i)

    try:
        for i, path in enumerate(files):
            top_up(i)
            t0 = time.perf_counter()
            futs.pop(i).result()
            waited = time.perf_counter() - t0
            for d in (STATS, mine):
                d['loader_wait_s'] += waited
                d['shards'] += 1
            yield from original([path], use_tqdm_on_load, safetensors_load_strategy, local_expert_ids, **kwargs)
            state['ahead'] -= sizes[i]
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
        from . import install
        install.save()


PATCHERS = {'vllm.model_executor.model_loader.weight_utils': patch_weight_utils}
