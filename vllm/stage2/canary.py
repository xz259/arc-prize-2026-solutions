"""stage2_v1 canary (diagnostics, not a part; 2026-10-15, user-approved 4-h aging run): does per-object work get slower with process
age, and is it the allocator or the garbage collector?

fq1 (10-15): fqfast's numpy frequency step grew 2.42 -> 5.49 ms over 50 min of API-server age with constant inputs (production's
path 21 -> 37 ms), roughly in proportion to each path's Python-object allocations; the engine core and worker host times stayed flat.
age1 (10-15/16, 4 h): API server canary x3.4 (flat after ~1.5 h), engine core x2.9 and still rising at a flat RSS; GC cost per
collection flat in the API server, x2.5 in the engine core; engine-core per-call costs grew by about the canary's factor.

Every stage-2 process with ARC3_S2_CANARY_DIR set runs a daemon thread that, every ARC3_S2_CANARY_PERIOD seconds (60):
- times a fixed allocation-heavy workload shaped like the planner's (a 53K-int list from numpy, its tuple, 18.5K tuples of 4, an
  18.5K-entry dict, a list copy): median of 3 repetitions, wall and thread CPU (GIL waits excluded from CPU);
- v2 (2026-10-16, IE's discriminators): the five steps timed separately (median of the 3 repetitions each), and where the workload's
  objects land: distinct 16-KiB pymalloc pools (id >> 14) and 2-MiB regions (id >> 21) of the 53K ints and of the 18.5K 4-tuples
  (contiguous allocation needs ~104 pools for the ints; thousands means scattered holes);
- reports the garbage collector's time per generation since the last record, from gc.callbacks (start / stop), with counts and the
  longest collection, plus gc.get_count(), gc.get_stats(), RSS and /proc/self/smaps_rollup's AnonHugePages / Anonymous;
- v2: at ages ARC3_S2_CANARY_MALLOCSTATS (minutes, comma list; default 5,60,120,180,240) writes sys._debugmallocstats() (pymalloc
  arenas / pools / per-size-class used vs available blocks) and a NUMA page summary of /proc/self/numa_maps (N0 / N1 pages) to
  $ARC3_S2_CANARY_DIR/mallocstats-<pid>-<minute>.txt. _debugmallocstats prints to fd 2: fd 2 is redirected to the file for the call
  (a few ms; other threads' stderr in that window lands in the file too).
Lines go to $ARC3_S2_CANARY_DIR/canary-<pid>.jsonl with the process's comm and argv head. Nothing runs on the serving threads except
the gc callback (two perf_counter reads per collection)."""
import gc
import json
import os
import sys
import threading
import time

DIR = os.environ.get('ARC3_S2_CANARY_DIR')
PERIOD = float(os.environ.get('ARC3_S2_CANARY_PERIOD', '60') or 60)
MALLOCSTATS = [float(x) for x in os.environ.get('ARC3_S2_CANARY_MALLOCSTATS', '5,60,120,180,240').split(',') if x.strip()]
_GC = dict(start=None, gen=None, n=[0, 0, 0], ms=[0.0, 0.0, 0.0], max_ms=[0.0, 0.0, 0.0])
_LOCK = threading.Lock()
_STARTED = []
STEPS = ('tolist', 'tuple', 'tuples4', 'dict', 'copy')


def _gc_callback(phase, info):
    if phase == 'start':
        _GC['start'] = time.perf_counter()
        _GC['gen'] = info.get('generation', 0)
    elif phase == 'stop' and _GC['start'] is not None:
        g = min(2, max(0, int(_GC['gen'] or 0)))
        ms = (time.perf_counter() - _GC['start']) * 1e3
        with _LOCK:
            _GC['n'][g] += 1
            _GC['ms'][g] += ms
            _GC['max_ms'][g] = max(_GC['max_ms'][g], ms)
        _GC['start'] = None


def _workload():
    import numpy as np
    a = np.arange(74000, dtype=np.int64) + 300                 # ids > 256: every int is a heap object (as token ids)
    pc = time.perf_counter
    t0, c0 = pc(), time.thread_time()
    lst = a[:53000].tolist()
    t1 = pc()
    tup = tuple(lst)
    t2 = pc()
    members = tuple(tuple(r) for r in a.reshape(-1, 4).tolist())
    t3 = pc()
    d = dict(zip(lst[:18500], lst[18500:37000]))
    t4 = pc()
    ids = list(lst[:18500])
    t5 = pc()
    w, c = (t5 - t0) * 1e3, (time.thread_time() - c0) * 1e3
    steps = [(t1 - t0) * 1e3, (t2 - t1) * 1e3, (t3 - t2) * 1e3, (t4 - t3) * 1e3, (t5 - t4) * 1e3]
    ints_pools = len({id(x) >> 14 for x in lst})
    ints_regions = len({id(x) >> 21 for x in lst})
    tup_pools = len({id(x) >> 14 for x in members})
    tup_regions = len({id(x) >> 21 for x in members})
    del lst, tup, members, d, ids
    return w, c, steps, (ints_pools, ints_regions, tup_pools, tup_regions)


def _rss_mb():
    try:
        with open('/proc/self/statm') as f:
            return int(f.read().split()[1]) * os.sysconf('SC_PAGE_SIZE') / 2 ** 20
    except Exception:                                   # noqa: BLE001
        return None


def _smaps():
    out = {}
    try:
        with open('/proc/self/smaps_rollup') as f:
            for line in f:
                k, _, v = line.partition(':')
                if k in ('Anonymous', 'AnonHugePages', 'Rss'):
                    out[k] = int(v.split()[0]) // 1024        # MiB
    except Exception:                                   # noqa: BLE001
        pass
    return out


def _numa():
    pages = {}
    try:
        with open('/proc/self/numa_maps') as f:
            for line in f:
                for tok in line.split():
                    if tok.startswith('N') and '=' in tok:
                        node, _, n = tok.partition('=')
                        if node[1:].isdigit():
                            pages[node] = pages.get(node, 0) + int(n)
    except Exception:                                   # noqa: BLE001
        pass
    return pages


def _mallocstats(minute):
    path = os.path.join(DIR, f'mallocstats-{os.getpid()}-{int(minute)}.txt')
    try:
        sys.stderr.flush()
        saved = os.dup(2)
        with open(path, 'w') as f:
            f.write(f'# pid {os.getpid()} age {minute:.0f} min numa_pages {json.dumps(_numa())} smaps {json.dumps(_smaps())}\n')
            f.flush()
            os.dup2(f.fileno(), 2)
            try:
                sys._debugmallocstats()
            finally:
                os.dup2(saved, 2)
                os.close(saved)
    except Exception:                                   # noqa: BLE001  diagnostics only
        pass


def _run():
    path = os.path.join(DIR, f'canary-{os.getpid()}.jsonl')
    try:
        comm = open('/proc/self/comm').read().strip()
    except Exception:                                   # noqa: BLE001
        comm = '?'
    head = ' '.join(sys.argv[:3])[:160]
    t_start = time.monotonic()
    pending_stats = sorted(MALLOCSTATS)
    while True:
        time.sleep(PERIOD)
        try:
            runs = [_workload() for _ in range(3)]
            order = sorted(range(3), key=lambda i: runs[i][0])
            w, c, _, spread = runs[order[1]]
            steps = [sorted(r[2][j] for r in runs)[1] for j in range(len(STEPS))]
            with _LOCK:
                n, ms, mx = list(_GC['n']), list(_GC['ms']), list(_GC['max_ms'])
                _GC['n'], _GC['ms'], _GC['max_ms'] = [0, 0, 0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]
            age = time.monotonic() - t_start
            row = dict(t=time.monotonic(), age_s=round(age, 1), comm=comm, argv=head,
                       wall_ms=round(w, 3), cpu_ms=round(c, 3), steps_ms=[round(x, 3) for x in steps],
                       ints_pools=spread[0], ints_regions=spread[1], tup_pools=spread[2], tup_regions=spread[3],
                       gc_n=n, gc_ms=[round(x, 2) for x in ms],
                       gc_max_ms=[round(x, 2) for x in mx], gc_count=list(gc.get_count()),
                       gc_stats=[dict(s) for s in gc.get_stats()], gc_threshold=list(gc.get_threshold()), rss_mb=_rss_mb(),
                       smaps_mb=_smaps())
            with open(path, 'a') as f:
                f.write(json.dumps(row, separators=(',', ':')) + '\n')
            while pending_stats and age >= pending_stats[0] * 60:
                _mallocstats(pending_stats.pop(0))
        except Exception:                               # noqa: BLE001  diagnostics only
            pass


def start():
    if not DIR or _STARTED:
        return
    os.makedirs(DIR, exist_ok=True)
    gc.callbacks.append(_gc_callback)
    threading.Thread(target=_run, name='arc3-s2-canary', daemon=True).start()
    _STARTED.append(True)
