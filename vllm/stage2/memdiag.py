"""stage2_v1 memdiag: memory attribution for the stage-2 memory audit (diagnostics only; no numeric effect).

Enabled by ARC3_S2_MEMDIAG=<max trace entries> (unset or 0: off). It is not an ARC3_STAGE2 part, so the audited server
shares the compile cache and cache tag of the configuration it audits.
  - CUDA-init tracer (every process; the bootstrap installs it before the production chain): the first torch CUDA
    initialisation of a process writes pid, process name and Python stack to $ARC3_S2_RECEIPTS/cuda-init-<pid>.txt.
  - Allocator history: torch memory history (allocation stacks, Python frames, a ring of <max> events without
    free_requested), started right after CUDA initialisation, so every allocation of the worker is attributed.
  - OOM observer: an allocation failure writes the same summary of the failing moment to receipts/oom-<pid>.json.
  - Worker RPC (dev-mode /collective_rpc) arc3_s2_mem_report(clear=False, top=40, dump=None): per-pool segment
    summary, live-block attribution, and the allocated-bytes peak inside the trace window with the blocks live at the
    peak (persistent or transient). clear=True starts a new trace window after the report.
  - Retry observer (2026-10-16, item 7 lane sizing; user: analyse the cause of allocator retries): ARC3_S2_MEMDIAG_RETRY=1
    starts a daemon thread in every CUDA process that polls the caching allocator's counters every 0.25 s. A new retry or OOM
    appends one line to receipts/retries-<pid>.jsonl (UTC time, counters, reserved / allocated / inactive split, device free)
    and, for the first 5, writes receipts/retry-<pid>-<n>.json = the summary above of that moment (the trace window before
    it: its peak's transient sites with stacks, the largest allocations). ARC3_S2_MEMDIAG_STOP_S=<s> stops the stack-
    recording history <s> s after CUDA init (start-up burst covered, steady-state overhead removed); counters keep polling."""
import importlib.abc
import os
import sys
import threading
import time
import traceback
from pathlib import Path

PINS = {'arc3_host_diagnostics': '1dc228caf3591fe4135df1fcc83ee5e806404b2eb17ef6592b35cbfa0eb7fcba'}
STATE = dict(max_entries=0, cuda_init=None, history_started=None, windows=0)
_SKIP = ('/torch/', 'contextlib.py', '<frozen', 'functools.py', 'threading.py', 'runpy.py', '/multiprocessing/',
         '/concurrent/', 'stage2_v1/memdiag.py')


def _enabled():
    return os.environ.get('ARC3_S2_MEMDIAG', '0') not in ('', '0')


# ---------------------------------------------------------------- CUDA-init tracer and history start
def _receipt(stack):
    root = os.environ.get('ARC3_S2_RECEIPTS')
    if not root:
        return
    try:
        comm = Path('/proc/self/comm').read_text().strip()
        cmd = Path('/proc/self/cmdline').read_bytes().replace(b'\0', b' ')[:300].decode(errors='replace')
        p = Path(root); p.mkdir(parents=True, exist_ok=True)
        (p / f'cuda-init-{os.getpid()}.txt').write_text(
            f'pid {os.getpid()} ppid {os.getppid()} comm {comm} time {time.time():.3f}\ncmd {cmd}\n\n{stack}')
    except Exception:
        pass


def _oom_observer(device, alloc, device_alloc, device_free):
    """torch OOM observer: write the memdiag summary of the failing moment to $ARC3_S2_RECEIPTS/oom-<pid>.json."""
    try:
        import json
        import torch
        out = summarize(torch.cuda.memory._snapshot(), top=40, device=device)
        out.update(pid=os.getpid(), time=time.time(), request=alloc, device_allocated=device_alloc, device_free=device_free)
        root = os.environ.get('ARC3_S2_RECEIPTS')
        if root:
            Path(root).mkdir(parents=True, exist_ok=True)
            (Path(root) / f'oom-{os.getpid()}.json').write_text(json.dumps(out, default=str))
    except Exception as exc:
        _receipt(f'oom observer failed: {exc!r}')


def start_history(clear=False):
    import torch
    torch.cuda.memory._record_memory_history(enabled='all', context='alloc', stacks='python',
                                             max_entries=STATE['max_entries'], clear_history=clear,
                                             skip_actions=['free_requested'])
    STATE['history_started'] = time.time(); STATE['windows'] += 1
    if not STATE.get('oom_observer'):
        torch._C._cuda_attach_out_of_memory_observer(_oom_observer)
        STATE['oom_observer'] = True


def _wrap(torch):
    C = torch._C
    orig = getattr(C, '_cuda_init', None)
    if orig is None or getattr(orig, '_s2_memdiag', False):
        return

    def _cuda_init(*args, **kwargs):
        stack = ''.join(traceback.format_stack(limit=80)[:-1])
        out = orig(*args, **kwargs)
        STATE['cuda_init'] = time.time()
        _receipt(stack)
        if STATE['max_entries']:
            try:
                start_history()
            except Exception as exc:
                _receipt(f'history start failed: {exc!r}\n\n{stack}')
        try:
            start_retry_watch()
        except Exception as exc:
            _receipt(f'retry watch start failed: {exc!r}')
        return out

    _cuda_init._s2_memdiag = True
    C._cuda_init = _cuda_init


class _TorchLoader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        m = getattr(self.original, 'create_module', None)
        return m(spec) if m else None

    def exec_module(self, module):
        self.original.exec_module(module)
        _wrap(module)

    def __getattr__(self, name):
        return getattr(self.original, name)


class _TorchFinder(importlib.abc.MetaPathFinder):
    def __init__(self):
        self._local = threading.local()

    def find_spec(self, fullname, path=None, target=None):
        if fullname != 'torch' or getattr(self._local, 'active', False):
            return None
        self._local.active = True
        try:
            started = False
            for finder in list(sys.meta_path):
                if finder is self:
                    started = True
                    continue
                if not started or not hasattr(finder, 'find_spec'):
                    continue
                spec = finder.find_spec(fullname, path, target)
                if spec is not None:
                    spec.loader = _TorchLoader(spec.loader)
                    return spec
            return None
        finally:
            self._local.active = False


def install_early():
    STATE['max_entries'] = int(os.environ.get('ARC3_S2_MEMDIAG', '0') or 0)
    if 'torch' in sys.modules:
        _wrap(sys.modules['torch'])
    elif not any(isinstance(f, _TorchFinder) for f in sys.meta_path):
        sys.meta_path.insert(0, _TorchFinder())


# ---------------------------------------------------------------- retry observer (2026-10-16)
def _retry_watch():
    import json
    import torch
    root = os.environ.get('ARC3_S2_RECEIPTS')
    if not root:
        return
    out_dir = Path(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    stop_s = float(os.environ.get('ARC3_S2_MEMDIAG_STOP_S', '0') or 0)
    last, n = (0, 0), 0                                  # the counters start at 0: a retry before the first poll counts too
    while True:
        time.sleep(0.25)
        try:
            if (stop_s and STATE.get('history_started') and not STATE.get('history_stopped')
                    and time.time() - (STATE.get('cuda_init') or time.time()) > stop_s):
                torch.cuda.memory._record_memory_history(enabled=None)
                STATE['history_stopped'] = time.time()
            dev = torch.cuda.current_device()
            st = torch.cuda.memory_stats(dev)
            cur = (int(st.get('num_alloc_retries', 0)), int(st.get('num_ooms', 0)))
            if cur == last:
                continue
            n += 1
            free, total = torch.cuda.mem_get_info(dev)
            line = dict(time=round(time.time(), 3), utc=time.strftime('%H:%M:%S', time.gmtime()), pid=os.getpid(),
                        retries=cur[0], ooms=cur[1], new_retries=cur[0] - last[0], new_ooms=cur[1] - last[1],
                        reserved=st.get('reserved_bytes.all.current'), allocated=st.get('allocated_bytes.all.current'),
                        active=st.get('active_bytes.all.current'), inactive_split=st.get('inactive_split_bytes.all.current'),
                        peak_reserved=st.get('reserved_bytes.all.peak'), segments=st.get('segment.all.current'),
                        device_free=free, device_total=total,
                        history_on=bool(STATE.get('history_started')) and not STATE.get('history_stopped'))
            with open(out_dir / f'retries-{os.getpid()}.jsonl', 'a') as fh:
                fh.write(json.dumps(line) + '\n')
            if n <= 5:
                summary = summarize(torch.cuda.memory._snapshot(), top=30, device=dev)
                summary.update(line)
                (out_dir / f'retry-{os.getpid()}-{n}.json').write_text(json.dumps(summary, default=str))
            last = cur
        except Exception as exc:
            STATE['retry_watch_error'] = repr(exc)[:300]
            time.sleep(5)


def start_retry_watch():
    if os.environ.get('ARC3_S2_MEMDIAG_RETRY', '0') in ('', '0') or STATE.get('retry_watch'):
        return
    STATE['retry_watch'] = time.time()
    threading.Thread(target=_retry_watch, name='arc3-memdiag-retry', daemon=True).start()


# ---------------------------------------------------------------- analysis
def _key(frames, depth=3):
    out = []
    for f in frames or ():
        fn = f.get('filename', '?')
        if any(s in fn for s in _SKIP):
            continue
        out.append(f"{'/'.join(fn.split('/')[-2:])}:{f.get('line')}:{f.get('name')}")
        if len(out) == depth:
            break
    return ' <- '.join(out) or '(no python frames)'


def _groups(items, top):
    g = {}
    for size, frames in items:
        k = _key(frames)
        e = g.setdefault(k, [0, 0, 0])
        e[0] += size; e[1] += 1; e[2] = max(e[2], size)
    rows = sorted(g.items(), key=lambda kv: -kv[1][0])
    return [dict(site=k, bytes=v[0], count=v[1], largest=v[2]) for k, v in rows[:top]], len(rows)


def summarize(snap, top=40, device=0):
    pools, live = {}, {}
    for s in snap['segments']:
        if s.get('device', device) != device:
            continue
        pid = str(tuple(s.get('segment_pool_id') or (0, 0)))
        p = pools.setdefault(pid, dict(segments=0, total=0, allocated=0, inactive=0, largest_inactive=0,
                                       small_total=0, inactive_blocks=0))
        p['segments'] += 1; p['total'] += s['total_size']
        if s.get('segment_type') == 'small':
            p['small_total'] += s['total_size']
        for b in s['blocks']:
            if b['state'] in ('active_allocated', 'active_pending_free'):
                p['allocated'] += b['size']
                live[b['address']] = (b['size'], b.get('frames') or [], pid)
            else:
                p['inactive'] += b['size']; p['inactive_blocks'] += 1
                p['largest_inactive'] = max(p['largest_inactive'], b['size'])
    out = dict(pools=pools, live_bytes=sum(v[0] for v in live.values()), live_blocks=len(live))
    by_pool = {}
    for size, frames, pid in live.values():
        by_pool.setdefault('default' if pid == '(0, 0)' else 'graph', []).append((size, frames))
    out['live_sites'] = {}
    for name, items in by_pool.items():
        rows, n = _groups(items, top)
        out['live_sites'][name] = dict(sites=n, top=rows)

    traces = snap.get('device_traces') or []
    trace = traces[device] if device < len(traces) else []
    out['trace'] = dict(events=len(trace))
    if not trace:
        return out
    tot = out['live_bytes']
    best, best_i = -1, len(trace) - 1
    for i in range(len(trace) - 1, -1, -1):      # tot = allocated bytes after event i
        if tot > best:
            best, best_i = tot, i
        e = trace[i]
        if e['action'] == 'alloc':
            tot -= e['size']
        elif e['action'] == 'free_completed':
            tot += e['size']
    start_bytes = tot
    peak = {a: (v[0], v[1], True) for a, v in live.items()}
    for i in range(len(trace) - 1, best_i, -1):
        e = trace[i]
        if e['action'] == 'alloc':
            peak.pop(e['addr'], None)
        elif e['action'] == 'free_completed':
            peak[e['addr']] = (e['size'], None, False)
    last_alloc = {}
    for i in range(best_i + 1):
        e = trace[i]
        if e['action'] == 'alloc':
            last_alloc[e['addr']] = e
    persistent, transient, big = [], [], []
    for a, (size, frames, at_end) in peak.items():
        if frames is None:
            e = last_alloc.get(a)
            frames = e.get('frames') if e else [{'filename': '(allocated before the trace window)', 'line': 0, 'name': ''}]
        (persistent if at_end else transient).append((size, frames))
        if not at_end:
            big.append((size, frames))
    t0, t1 = trace[0].get('time_us', 0), trace[-1].get('time_us', 0)
    out['trace'].update(window_s=(t1 - t0) / 1e6, start_bytes=start_bytes, end_bytes=out['live_bytes'],
                        peak_bytes=best, peak_index=best_i, peak_at_s=(trace[best_i].get('time_us', 0) - t0) / 1e6,
                        peak_transient_bytes=sum(s for s, _ in transient),
                        peak_persistent_bytes=sum(s for s, _ in persistent))
    out['trace']['peak_transient_sites'] = _groups(transient, top)[0]
    out['trace']['peak_largest_transient'] = [dict(bytes=s, site=_key(f, 6)) for s, f in sorted(big, key=lambda x: -x[0])[:15]]
    sizes = sorted((e['size'], _key(e.get('frames'), 4)) for e in trace if e['action'] == 'alloc')[-10:]
    out['trace']['largest_allocs'] = [dict(bytes=s, site=k) for s, k in reversed(sizes)]
    return out


# ---------------------------------------------------------------- worker RPC
def patch_diagnostics(mod):
    def arc3_s2_mem_report(self, clear=False, top=40, dump=None):
        import torch
        torch.cuda.synchronize()
        dev = torch.cuda.current_device()
        snap = torch.cuda.memory._snapshot()
        out = summarize(snap, top=int(top), device=dev)
        if dump:
            import pickle
            with open(dump, 'wb') as f:
                pickle.dump(snap, f)
        del snap
        free, total = torch.cuda.mem_get_info()
        stats = torch.cuda.memory_stats()
        out.update(pid=os.getpid(), free=free, total=total, allocated=torch.cuda.memory_allocated(),
                   reserved=torch.cuda.memory_reserved(), peak_allocated=stats.get('allocated_bytes.all.peak'),
                   peak_reserved=stats.get('reserved_bytes.all.peak'), retries=stats.get('num_alloc_retries'),
                   inactive_split=stats.get('inactive_split_bytes.all.current'),
                   state={k: v for k, v in STATE.items()})
        if clear:
            start_history(clear=True)
            torch.cuda.reset_peak_memory_stats()
        return out

    mod.WorkerDiagnostics.arc3_s2_mem_report = arc3_s2_mem_report
