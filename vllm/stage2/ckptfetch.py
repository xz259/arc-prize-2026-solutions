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
the one-shard-ahead mode above, unchanged.

2026-10-30 (arc3-code v13; the user: "Add the threaded PLE copy to a v13 build"): ARC3_S2_PLECOPY_THREADS = T > 1 splits each PLE
row copy of the PLE offload worker over T threads, so the page faults of the mmap'd shard overlap (a competitor's report from
Kaggle: PLE shards 48.4 -> 7.0 s each, ready at 9-10 min instead of 23). The vendor copy (copy_ple_embedding_shard_, one
target.copy_ per checkpoint tensor; OMP_NUM_THREADS=1 here) runs on T contiguous row ranges of the tensor, each a call of the
vendor function itself, so the rows and their places are the vendor's (exact: the same bytes). It is chained on ple_layer under
the pin pleconv and b1inc already use; tensors that are not CPU-to-CPU, same-dtype and >= 65,536 rows take the vendor call.
In that worker the loader no longer waits for a shard's stream: the copy threads fault in what the streams have not read
(FILES >= 1 streams, one per shard, the same read-ahead bound), and a consumed shard's stream stops at its next block.
Receipts per process: shard_s (the loader's seconds per shard), iter_s, ple_worker, done; plecopy_* totals. Unset or 1:
v12's code paths, unchanged (no ple_layer hook).

2026-10-30 (arc3-code v14; the user: "Let's do it", after v13's Kaggle smoke: the PLE worker's 20 shards in 399 s, median
19.1 s each, but the server process's 15 target shards on v12's path took 734 s, median 36.3 s, and set the start at 1,025 s):
ARC3_S2_FAULTIN_THREADS = N > 1 gives the server process's loads (the target and the one-file drafter) the PLE worker's kind of
reading: while the loader reads a shard, N threads read its needed bytes in contiguous pieces through one mmap of the file (torch
copies into small buffers, the GIL released, so the page faults overlap; preads if the file cannot be mapped), and the loader no
longer waits for the shard's stream (the FILES streams keep reading ahead). The threads only read: the loader's bytes and the
weights are unchanged. The PLE worker keeps v13's path. Receipts: faultin_* totals; one per_pid entry per load (pid, then
pid.N). Unset or 1: v13's paths."""
import concurrent.futures
import json
import os
import struct
import threading
import time

PINS = {'vllm.model_executor.model_loader.weight_utils': '25bb3393105f9a642b7fa58bc277865db9976748be320787f8db2e03b16fd667'}
THREADS = int(os.environ.get('ARC3_S2_CKPTFETCH_THREADS', '8'))
FILES = int(os.environ.get('ARC3_S2_CKPTFETCH_FILES', '1') or 1)               # v12: shards read concurrently (1: one ahead)
AHEAD = int(float(os.environ.get('ARC3_S2_CKPTFETCH_AHEAD_GB', '48') or 48) * (1 << 30))
PLECOPY = int(os.environ.get('ARC3_S2_PLECOPY_THREADS', '1') or 1)            # v13: threads per PLE row copy (1: v12)
PLECOPY_MIN_ROWS = 65536
PLE_LAYER = 'vllm.models.qwen3_8_flash_next.nvidia.ple_layer'
PLE_LAYER_PIN = '160d553ec81bae0990934cfe7c14f98d2d65b6ee3d5c5a85ff8421b7ddb17952'  # pleconv's and b1inc's pin (production)
BLOCK = 16 << 20
STATS = dict(threads=THREADS, iterators=0, passthrough=0, shards=0, bytes=0, prefetch_s=0.0, loader_wait_s=0.0, errors=0,
             per_pid={}, files=FILES, ahead_gb=AHEAD / (1 << 30), max_in_flight=0)
FAULTIN = int(os.environ.get('ARC3_S2_FAULTIN_THREADS', '1') or 1)            # v14: threads faulting in a server shard (1: v13)
FAULT_CHUNK = 8 << 20
if PLECOPY > 1:
    STATS.update(plecopy=PLECOPY, plecopy_installed=False, plecopy_calls=0, plecopy_fallback=0, plecopy_rows=0, plecopy_bytes=0,
                 plecopy_s=0.0, plecopy_errors=[], ungated=0)
if FAULTIN > 1:
    STATS.update(faultin=FAULTIN, faultin_shards=0, faultin_bytes=0, faultin_s=0.0, faultin_mode=None, faultin_errors=[], ungated=0)


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


def _read(path, pieces, stop=None):
    fd = os.open(path, os.O_RDONLY)
    try:
        for s, e in pieces:
            while s < e:
                if stop is not None and stop.is_set():      # v13: the shard is consumed (ungated PLE worker)
                    return
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
        ple = _ple_worker(mod) if (PLECOPY > 1 or FAULTIN > 1) else False
        fault = FAULTIN > 1 and not ple             # v14: the server process's loads, one file (the drafter) included
        if safetensors_load_strategy is not None or THREADS < 1 or not files or (len(files) < 2 and not fault):
            STATS['passthrough'] += 1
            yield from original(hf_weights_files, use_tqdm_on_load, safetensors_load_strategy, local_expert_ids,
                                **kwargs)
            return
        STATS['iterators'] += 1
        pid = str(os.getpid())
        if (PLECOPY > 1 or FAULTIN > 1) and pid in STATS['per_pid']:
            pid = '%s.%d' % (pid, STATS['iterators'])    # v14: one receipt entry per load (the target, then the drafter)
        mine = STATS['per_pid'].setdefault(pid, dict(shards=0, bytes=0, prefetch_s=0.0, loader_wait_s=0.0))
        if PLECOPY > 1 and ple:                     # v13: the copy threads read what the streams have not; no wait per shard
            mine.update(ple_worker=True, ungated=0, shard_s=[], done=False)
            yield from _multi(files, mine, original, mod, use_tqdm_on_load, safetensors_load_strategy, local_expert_ids, kwargs,
                              in_flight=max(FILES, 1), gate=False)
            return
        if fault:                                   # v14: FAULTIN threads fault the shard in while the loader reads it; no wait
            mine.update(ungated=0, shard_s=[], done=False, faultin_bytes=0)
            yield from _multi(files, mine, original, mod, use_tqdm_on_load, safetensors_load_strategy, local_expert_ids, kwargs,
                              in_flight=max(FILES, 1), gate=False, faultin=FAULTIN)
            return
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


def _multi(files, mine, original, mod, use_tqdm_on_load, safetensors_load_strategy, local_expert_ids, kwargs, in_flight=None,
           gate=True, faultin=1):
    """v12: up to FILES shards in flight, one stream each, in the loader's order, at most FILES shards from the one the loader
    reads and (beyond that one) within AHEAD bytes of started-but-unread shards; the loader still gets one shard per call.
    v13 (gate False: the PLE worker with ARC3_S2_PLECOPY_THREADS > 1): in_flight streams; the loader gets each shard without
    waiting for its stream, and a consumed shard's stream stops at its next block (a queued one is cancelled).
    v14 (faultin > 1: the server process with ARC3_S2_FAULTIN_THREADS > 1): while the loader reads a shard, faultin threads read
    its needed bytes in contiguous pieces through one mmap of the file (page faults, as the PLE copy's), stopped when the
    loader is done with it."""
    F = in_flight or FILES
    pool = concurrent.futures.ThreadPoolExecutor(F, thread_name_prefix='arc3-ckptfetch')
    fpool = concurrent.futures.ThreadPoolExecutor(faultin, thread_name_prefix='arc3-faultin') if faultin > 1 else None
    fstop = None
    sizes = []
    for p in files:
        try:
            sizes.append(os.path.getsize(p))
        except OSError:
            sizes.append(0)

    def fetch(path, stop=None):
        t0 = time.perf_counter()
        try:
            ranges = needed_ranges(path, lambda k: mod.should_skip_weight(k, local_expert_ids))
            _read(path, ranges, stop)
            got = sum(e - s for s, e in ranges)
        except Exception:                                      # noqa: BLE001 (prefetch is advisory: the loader reads anyway)
            STATS['errors'] += 1
            got = 0
        dt = time.perf_counter() - t0
        for d in (STATS, mine):
            d['bytes'] += got
            d['prefetch_s'] += dt
        return got

    futs, stops, state = {}, {}, dict(next=0, ahead=0)

    def top_up(i):
        while state['next'] < len(files) and state['next'] - i < F and (
                state['next'] <= i or state['ahead'] + sizes[state['next']] <= AHEAD):
            j = state['next']
            stops[j] = None if gate else threading.Event()
            futs[j] = pool.submit(fetch, files[j], stops[j])
            state['ahead'] += sizes[j]
            state['next'] += 1
        STATS['max_in_flight'] = max(STATS['max_in_flight'], state['next'] - i)

    t_iter = time.perf_counter()
    mine.setdefault('shard_s', [])
    try:
        for i, path in enumerate(files):
            top_up(i)
            t0 = time.perf_counter()
            fut = futs.pop(i)
            if gate:
                fut.result()
            else:
                for d in (STATS, mine):
                    d['ungated'] += 1
            waited = time.perf_counter() - t0
            for d in (STATS, mine):
                d['loader_wait_s'] += waited
                d['shards'] += 1
            if fpool is not None:
                fstop = threading.Event()
                _faultin_start(path, mod, local_expert_ids, fpool, faultin, fstop, mine)
            t1 = time.perf_counter()
            yield from original([path], use_tqdm_on_load, safetensors_load_strategy, local_expert_ids, **kwargs)
            mine['shard_s'].append(round(time.perf_counter() - t1, 2))
            if fstop is not None:
                fstop.set()
            if not gate:
                stops.pop(i).set()
                fut.cancel()
            state['ahead'] -= sizes[i]
        mine['done'] = True
    finally:
        for ev in list(stops.values()) + [fstop]:
            if ev is not None:
                ev.set()
        pool.shutdown(wait=False, cancel_futures=True)
        if fpool is not None:
            fpool.shutdown(wait=False, cancel_futures=True)
        if not gate:
            _plecopy_shutdown()
        mine['iter_s'] = round(time.perf_counter() - t_iter, 1)
        from . import install
        install.save()


def _ple_worker(mod):
    """True in the PLE offload worker (weight_utils imports vLLM's is_offload_process)."""
    try:
        return bool(mod.is_offload_process())
    except Exception:                                          # noqa: BLE001 (no PLE offload here: the main paths)
        return False


_FAULT_LOCK = threading.Lock()


def _split(ranges, n):
    """The merged ranges cut into n contiguous jobs of nearly equal bytes (prefetch()'s shares)."""
    total = sum(e - s for s, e in ranges)
    share = max(1, -(-total // n))
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
    return jobs


def _faultin_job(path, m, job, stop):
    """Read one job's ranges: chunk copies out of the file's mmap (torch, the GIL released), else preads. Advisory: never raises."""
    t0, n = time.perf_counter(), 0
    try:
        if m is not None:
            import torch
            buf = torch.empty(min(FAULT_CHUNK, max(e - s for s, e in job)), dtype=torch.uint8)
            for s, e in job:
                while s < e and not stop.is_set():
                    k = min(FAULT_CHUNK, e - s)
                    buf[:k].copy_(m[s:s + k])
                    s, n = s + k, n + k
        else:
            fd = os.open(path, os.O_RDONLY)
            try:
                for s, e in job:
                    while s < e and not stop.is_set():
                        got = len(os.pread(fd, min(FAULT_CHUNK, e - s), s))
                        if not got:
                            break
                        s, n = s + got, n + got
            finally:
                os.close(fd)
    except Exception as exc:                                   # noqa: BLE001 (the loader reads anyway)
        with _FAULT_LOCK:
            if len(STATS['faultin_errors']) < 20:
                STATS['faultin_errors'].append(repr(exc)[:160])
    with _FAULT_LOCK:
        STATS['faultin_bytes'] += n
        STATS['faultin_s'] += time.perf_counter() - t0
    return n


def _faultin_start(path, mod, local_expert_ids, fpool, n, stop, mine):
    """v14: the shard's needed bytes in n contiguous jobs on fpool, read through one torch mmap of the file (pread if it cannot
    be mapped); the loader does not wait for them."""
    try:
        ranges = needed_ranges(path, lambda k: mod.should_skip_weight(k, local_expert_ids))
        m = None
        try:
            import torch
            m = torch.from_file(path, shared=False, size=os.path.getsize(path), dtype=torch.uint8)
            STATS['faultin_mode'] = STATS.get('faultin_mode') or 'mmap'
        except Exception as exc:                               # noqa: BLE001
            STATS['faultin_mode'] = 'pread (%s)' % type(exc).__name__
        for job in _split(ranges, n):
            fpool.submit(_faultin_job, path, m, job, stop)
        STATS['faultin_shards'] += 1
        mine['faultin_bytes'] += sum(e - s for s, e in ranges)
    except Exception as exc:                                   # noqa: BLE001
        with _FAULT_LOCK:
            if len(STATS['faultin_errors']) < 20:
                STATS['faultin_errors'].append('start: %r' % (exc,))


_PLECOPY_POOL = []
_PLECOPY_LOCK = threading.Lock()


def _plecopy_pool():
    with _PLECOPY_LOCK:
        if not _PLECOPY_POOL:
            _PLECOPY_POOL.append(concurrent.futures.ThreadPoolExecutor(PLECOPY, thread_name_prefix='arc3-plecopy'))
        return _PLECOPY_POOL[0]


def _plecopy_shutdown():
    with _PLECOPY_LOCK:
        while _PLECOPY_POOL:
            _PLECOPY_POOL.pop().shutdown(wait=False)


def patch_ple_layer(mod):
    """v13: ple_layer's copy_ple_embedding_shard_ (the PLE worker's row copy into the CPU table) on PLECOPY row ranges in
    parallel. Each range is one call of the vendor function on a row slice of the checkpoint tensor with its checkpoint_start
    moved by the slice's offset: the vendor computes the overlap with the TP range and copies, so the union of the slices' rows
    is the full call's. Anything unexpected keeps the vendor path (this is a startup speed-up; it never blocks the server)."""
    try:
        original = mod.copy_ple_embedding_shard_
    except Exception as exc:                                   # noqa: BLE001
        STATS['plecopy_errors'].append('install: %r' % (exc,))
        return

    def copy_ple_embedding_shard_(destination, loaded_weight, *, checkpoint_start, tp_start, tp_end):
        def vendor():
            return original(destination, loaded_weight, checkpoint_start=checkpoint_start, tp_start=tp_start, tp_end=tp_end)
        try:
            rows = int(loaded_weight.shape[0]) if loaded_weight.dim() >= 1 else 0
            ok = (rows >= PLECOPY_MIN_ROWS and destination.device.type == 'cpu' and loaded_weight.device.type == 'cpu'
                  and destination.dtype == loaded_weight.dtype)
        except Exception:                                      # noqa: BLE001
            ok = False
        if not ok:
            STATS['plecopy_fallback'] += 1
            return vendor()
        t0 = time.perf_counter()
        try:
            step = -(-rows // PLECOPY)
            pool = _plecopy_pool()
            futs = [pool.submit(original, destination, loaded_weight.narrow(0, a, min(step, rows - a)),
                                checkpoint_start=checkpoint_start + a, tp_start=tp_start, tp_end=tp_end)
                    for a in range(0, rows, step)]
            concurrent.futures.wait(futs)
            copied = sum(f.result() for f in futs)
        except Exception as exc:                               # noqa: BLE001 (the vendor call redoes every row, or raises its own)
            STATS['plecopy_errors'].append(repr(exc)[:200])
            return vendor()
        STATS['plecopy_calls'] += 1
        STATS['plecopy_rows'] += copied
        STATS['plecopy_bytes'] += copied * loaded_weight.element_size() * (loaded_weight.numel() // rows)
        STATS['plecopy_s'] += time.perf_counter() - t0
        return copied

    copy_ple_embedding_shard_.__wrapped__ = original
    mod.copy_ple_embedding_shard_ = copy_ple_embedding_shard_
    STATS['plecopy_installed'] = True


PATCHERS = {'vllm.model_executor.model_loader.weight_utils': patch_weight_utils}
if PLECOPY > 1:                                                 # v13: chained after pleconv and b1inc (the same pin)
    PINS[PLE_LAYER] = PLE_LAYER_PIN
    PATCHERS[PLE_LAYER] = patch_ple_layer
