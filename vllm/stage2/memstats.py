"""stage2_v1 memstats: trigger-file host-memory accounting in every stage-2 process (2026-10-03, user + Fable's review:
the API server grew to ~31 GB over 7.5 h of screens; live data or allocator retention?).

Off unless ARC3_S2_MEMSTATS_DIR is set. Then each process (API server, engine core, GPU worker, PLE helper) runs a daemon
thread that polls <dir>/trigger every 2 s. A line '<id> <light|full>' runs one reading per new id and writes
<dir>/memstats-<id>-<pid>.json (+ .xml, glibc's malloc_info):
  light  /proc/self status and smaps_rollup; glibc mallinfo2 (in-use, free, mmapped bytes); malloc_info per arena
         (system bytes against free chunks: system - free = in use); Python's allocated block count. Cheap; no pause.
  full   light, plus a gc walk: object counts and sys.getsizeof sums by type (top 40), str / bytes / list / dict totals,
         live session_service.Session objects (all of them, including any no longer in a manager), each manager's
         len(sessions), and per-session lengths (token_ids, parts, images, structural, segments) with the summed string
         and bytes sizes reachable from parts (bounded walk). The walk holds the GIL for seconds on a large heap: run it
         only when the server is idle (after a run's arms finish).
         21:22 (code read: SessionManager never removes a closed session, and session.operations keeps every turn's task
         and result): per session also the deep bytes of parts (CPU tensor storages of the processed images, strings,
         int lists), of the completed operations' results and of the other per-session fields, with totals by state.
  owners (2026-10-07, user: profile the API server's growth; 392b's end reading found 1.16 GB in 158,617 lists reachable
         from no Session): who holds the large lists. Of the lists with >= 1,000 elements, the 300 longest and a random
         700 more are traced one or two holders up (an instance attribute, a dict key pattern with digits and hex runs as
         '#', a list / tuple / deque element, a slot object's field); aggregated by holder path with list counts and
         element totals; plus a histogram of all lists by length. Holds the GIL for seconds: profiling runs only.
Every reading also records the collector's state (enabled, thresholds, counts, per-generation stats, frozen objects) and,
in a process with the sessmem part loaded (the API server), sessmem's counters (2026-10-08).
Collector monitor (2026-10-07): while memstats is on, gc.callbacks time every collection; every 60 s the poll thread
appends to <dir>/gcprof-<pid>.jsonl the collections, seconds and the longest pause per generation since the last line,
the pauses of >= 50 ms (time, generation, seconds, collected), the allocated block count, RSS and the busiest threads'
CPU since the last line.
No server requests, no allocator changes, no CUDA access; errors go to <dir>/memstats-<id>-<pid>.error."""
import ctypes
import dataclasses
import gc
import json
import os
import sys
import threading
import time
from collections import Counter, deque
from collections.abc import Mapping

DIR = os.environ.get('ARC3_S2_MEMSTATS_DIR')
_STARTED = []


class _MallInfo2(ctypes.Structure):
    _fields_ = [(n, ctypes.c_size_t) for n in ('arena', 'ordblks', 'smblks', 'hblks', 'hblkhd', 'usmblks', 'fsmblks',
                                                'uordblks', 'fordblks', 'keepcost')]


def _proc():
    out = {}
    for name, keys in (('status', ('VmRSS', 'VmHWM', 'RssAnon', 'RssFile', 'RssShmem', 'Threads')),
                       ('smaps_rollup', ('Rss', 'Anonymous', 'Private_Dirty', 'Swap'))):
        try:
            for line in open('/proc/self/' + name):
                k = line.split(':')[0]
                if k in keys:
                    v = line.split()[1]
                    out[k + ('' if k == 'Threads' else '_kb')] = int(v)
        except OSError:
            pass
    return out


def _glibc(xml_path):
    libc = ctypes.CDLL(None)
    out = {}
    try:
        f = libc.mallinfo2
        f.restype = _MallInfo2
        m = f()
        out['mallinfo2'] = {k: int(getattr(m, k)) for k, _ in _MallInfo2._fields_}
        out['mallinfo2']['note'] = 'uordblks = in-use (arenas), fordblks = free in arenas, hblkhd = mmapped blocks'
    except AttributeError:
        out['mallinfo2'] = None
    try:
        libc.fopen.restype = ctypes.c_void_p
        libc.fopen.argtypes = [ctypes.c_char_p, ctypes.c_char_p]
        libc.malloc_info.argtypes = [ctypes.c_int, ctypes.c_void_p]
        libc.fclose.argtypes = [ctypes.c_void_p]
        fp = libc.fopen(xml_path.encode(), b'w')
        if fp:
            libc.malloc_info(0, fp)
            libc.fclose(fp)
            out['arenas'] = _parse_malloc_info(xml_path)
    except Exception as exc:                            # noqa: BLE001
        out['arenas_error'] = repr(exc)
    return out


def _parse_malloc_info(path):
    import xml.etree.ElementTree as ET
    root = ET.parse(path).getroot()
    heaps = []
    for h in root.findall('heap'):
        free = sum(int(t.get('size', 0)) for t in h.findall('total') if t.get('type') in ('fast', 'rest'))
        system = next((int(s.get('size', 0)) for s in h.findall('system') if s.get('type') == 'current'), 0)
        heaps.append(dict(nr=int(h.get('nr', -1)), system=system, free=free, in_use=system - free))
    totals = {t.get('type'): dict(count=int(t.get('count', 0)), size=int(t.get('size', 0))) for t in root.findall('total')}
    systems = {s.get('type'): int(s.get('size', 0)) for s in root.findall('system')}
    return dict(heaps=len(heaps), system=sum(h['system'] for h in heaps), free=sum(h['free'] for h in heaps),
                in_use=sum(h['in_use'] for h in heaps), largest=sorted(heaps, key=lambda h: -h['system'])[:8],
                totals=totals, systems=systems)


def _deep_strings(obj, budget, seen):
    """Summed sizes of str / bytes / bytearray reachable from obj through dict / list / tuple (bounded)."""
    total = 0
    stack = [obj]
    while stack and budget[0] > 0:
        o = stack.pop()
        if id(o) in seen:
            continue
        seen.add(id(o))
        budget[0] -= 1
        if isinstance(o, (str, bytes, bytearray)):
            total += sys.getsizeof(o)
        elif isinstance(o, dict):
            stack.extend(o.values())
        elif isinstance(o, (list, tuple)):
            stack.extend(o)
    return total


_ENTER = ('vllm.multimodal', 'session_service', 'stage2_v1')     # objects whose __dict__ the deep walk enters
_KINDS = ('str', 'cpu_tensor', 'ndarray', 'ints', 'containers', 'other')   # host bytes; device_tensor is reported apart
_OTHER_FIELDS = ('images', 'segments', 'structural', 'image_kinds', 'initial', 'residency', 'suffix_registry',
                 'history_registry', 'history_policy', 'rebase_info', 'prepare_profile', 'eviction_endpoint_profile')


def _deep_bytes(roots, budget, seen, acc):
    """Adds to acc (a Counter) the bytes reachable from the roots, by kind: str (str / bytes / bytearray), cpu_tensor /
    device_tensor (torch storages, each once), ndarray (numpy buffers, each once), ints (32 B per int element of a list or
    tuple, estimated without visiting the ints), containers (getsizeof of each list / tuple / mapping / set / object
    entered), other (getsizeof of an object not entered). Enters lists, tuples, Mappings (UserDict too), sets, deques,
    dataclasses and objects of vllm.multimodal / session_service / stage2_v1 modules, nothing else, so it does not wander
    into the engine client or the event loop. seen is shared across calls (shared objects count once); budget[0] bounds
    the visits (acc['truncated'] counts a stop); acc['errors'] counts objects skipped on an exception."""
    torch, np = sys.modules.get('torch'), sys.modules.get('numpy')
    stack = list(roots)                                 # the roots container is not visited (it may be a temporary)
    while stack:
        o = stack.pop()
        t = type(o)
        if o is None or t in (bool, int, float):
            continue
        if id(o) in seen:
            continue
        if budget[0] <= 0:
            acc['truncated'] += 1
            return
        budget[0] -= 1
        seen.add(id(o))
        try:
            if t in (str, bytes, bytearray):
                acc['str'] += sys.getsizeof(o)
            elif torch is not None and isinstance(o, torch.Tensor):
                try:
                    s = o.untyped_storage()
                    key, n = ('storage', o.device.type, s.data_ptr()), s.nbytes()
                except Exception:                       # noqa: BLE001
                    key, n = ('tensor', id(o)), o.element_size() * o.nelement()
                if key not in seen:
                    seen.add(key)
                    acc['cpu_tensor' if o.device.type == 'cpu' else 'device_tensor'] += n
            elif np is not None and isinstance(o, np.ndarray):
                key = ('ndarray', o.__array_interface__['data'][0], o.nbytes)
                if key not in seen:
                    seen.add(key)
                    acc['ndarray'] += o.nbytes
            elif t in (list, tuple):
                acc['containers'] += sys.getsizeof(o)
                ints = sum(1 for x in o if type(x) is int)
                acc['ints'] += 32 * ints
                if ints < len(o):
                    stack.extend(x for x in o if type(x) is not int)
            elif isinstance(o, Mapping):
                acc['containers'] += sys.getsizeof(o)
                stack.extend(list(o.values()))
            elif isinstance(o, (set, frozenset, deque)):
                acc['containers'] += sys.getsizeof(o)
                stack.extend(list(o))
            elif dataclasses.is_dataclass(o) and not isinstance(o, type):
                acc['containers'] += sys.getsizeof(o)
                stack.extend(getattr(o, f.name, None) for f in dataclasses.fields(o))
            elif _tmod(t).startswith(_ENTER) and hasattr(o, '__dict__') and not isinstance(o, type):
                acc['containers'] += sys.getsizeof(o)
                stack.extend(list(vars(o).values()))
            else:
                acc['other'] += sys.getsizeof(o)
        except Exception:                               # noqa: BLE001  a racing mutation or an odd type: skip it
            acc['errors'] += 1


def _tmod(t):
    """t.__module__ as a string: a heap type that defines __module__ for its instances returns a descriptor instead."""
    m = getattr(t, '__module__', None)
    return m if isinstance(m, str) else '?'


def _tqual(t):
    q = getattr(t, '__qualname__', None)
    return q if isinstance(q, str) else str(getattr(t, '__name__', '?'))


def _op_results(session):
    """Results of the session's completed operations (each operation keeps {'digest', 'task'} for the session's life)."""
    out = []
    for op in list((getattr(session, 'operations', None) or {}).values()):
        task = op.get('task') if isinstance(op, dict) else None
        try:
            if task is not None and task.done() and not task.cancelled() and task.exception() is None:
                out.append(task.result())
        except Exception:                               # noqa: BLE001
            pass
    return out


def _host_bytes(acc):
    return sum(acc.get(k, 0) for k in _KINDS)


def _full():
    gc.collect()
    objs = gc.get_objects()
    count, size = Counter(), Counter()
    sessions, managers = [], []
    for o in objs:
        t = type(o)
        mod, qual = _tmod(t), _tqual(t)
        name = mod + '.' + qual
        count[name] += 1
        try:
            size[name] += sys.getsizeof(o)
        except Exception:                               # noqa: BLE001
            pass
        if qual == 'Session' and mod.endswith('session_service'):
            sessions.append(o)
        elif qual == 'SessionManager' and mod.endswith('session_service'):
            managers.append(o)
    out = dict(objects=len(objs), top_by_size=size.most_common(40),
               builtins={k: dict(count=count[k], bytes=size[k]) for k in ('builtins.str', 'builtins.bytes',
                                                                          'builtins.list', 'builtins.dict', 'builtins.tuple')})
    live = set()
    out['managers'] = []
    for m in managers:
        sid = getattr(m, 'sessions', {}) or {}
        live.update(id(s) for s in sid.values())
        out['managers'].append(dict(sessions=len(sid), max_sessions=getattr(m, 'max_sessions', None)))
    per = []
    deep_seen = set()                                   # shared: an object reachable from two sessions counts once
    for s in sessions:
        budget, seen = [200000], set()
        parts_acc, ops_acc, other_acc = Counter(), Counter(), Counter()
        _deep_bytes([getattr(s, 'parts', None)], [400000], deep_seen, parts_acc)
        results = _op_results(s)
        _deep_bytes(results, [400000], deep_seen, ops_acc)
        _deep_bytes([getattr(s, f, None) for f in _OTHER_FIELDS], [200000], deep_seen, other_acc)
        n_tok = len(getattr(s, 'token_ids', []) or [])
        per.append(dict(state=getattr(s, 'state', None), in_manager=id(s) in live,
                        token_ids=n_tok, parts=len(getattr(s, 'parts', []) or []),
                        images=len(getattr(s, 'images', []) or []), structural=len(getattr(s, 'structural', []) or []),
                        segments=len(getattr(s, 'segments', []) or []), rebases=getattr(s, 'rebases', None),
                        parts_string_bytes=_deep_strings(getattr(s, 'parts', []), budget, seen),
                        token_ids_est_bytes=40 * n_tok,  # 8-B list slot + a 32-B int block per token (estimate)
                        operations=len(getattr(s, 'operations', None) or {}), op_results=len(results),
                        parts_deep=dict(parts_acc), ops_deep=dict(ops_acc), other_deep=dict(other_acc),
                        retained_bytes=40 * n_tok + _host_bytes(parts_acc) + _host_bytes(ops_acc)
                        + _host_bytes(other_acc)))
    by_state = {}
    for p in per:
        b = by_state.setdefault(str(p['state']), Counter())
        b['sessions'] += 1
        b['in_manager'] += int(p['in_manager'])
        b['operations'] += p['operations']
        b['token_ids_est'] += p['token_ids_est_bytes']
        b['retained'] += p['retained_bytes']
        for name in ('parts_deep', 'ops_deep', 'other_deep'):
            for k, v in p[name].items():
                b[name[:-5] + '.' + k] += v
    out['sessions'] = dict(objects=len(per), in_manager=sum(1 for p in per if p['in_manager']),
                           states=Counter(str(p['state']) for p in per).most_common(),
                           token_ids=sum(p['token_ids'] for p in per), parts=sum(p['parts'] for p in per),
                           parts_string_bytes=sum(p['parts_string_bytes'] for p in per),
                           retained_bytes=sum(p['retained_bytes'] for p in per),
                           by_state={k: dict(v) for k, v in by_state.items()},
                           largest=sorted(per, key=lambda p: -p['retained_bytes'])[:5])
    del objs
    return out


def _gc_state():
    try:
        return dict(enabled=gc.isenabled(), threshold=gc.get_threshold(), count=gc.get_count(), stats=gc.get_stats(),
                    frozen=gc.get_freeze_count())
    except Exception as exc:                            # noqa: BLE001
        return dict(error=repr(exc))


_GC = dict(t0=None, n=[0, 0, 0], secs=[0.0, 0.0, 0.0], max=[0.0, 0.0, 0.0], collected=[0, 0, 0], long=[])


def _gc_cb(phase, info):
    """gc.callbacks hook: times each collection (the collecting thread holds the GIL throughout)."""
    try:
        if phase == 'start':
            _GC['t0'] = time.perf_counter()
            return
        t0, _GC['t0'] = _GC['t0'], None
        if t0 is None:
            return
        d = time.perf_counter() - t0
        g = min(2, int(info.get('generation', 0)))
        _GC['n'][g] += 1
        _GC['secs'][g] += d
        _GC['max'][g] = max(_GC['max'][g], d)
        _GC['collected'][g] += int(info.get('collected', 0))
        if d >= 0.05 and len(_GC['long']) < 500:
            _GC['long'].append((round(time.time(), 3), g, round(d, 4), int(info.get('collected', 0))))
    except Exception:                                   # noqa: BLE001  never disturb the collection
        pass


def _threads():
    out = {}
    try:
        tasks = os.listdir('/proc/self/task')
    except OSError:
        return out
    for t in tasks:
        try:
            stat = open('/proc/self/task/%s/stat' % t).read()
            name = stat[stat.index('(') + 1:stat.rindex(')')]
            f = stat[stat.rindex(')') + 2:].split()
            out[t] = (name, int(f[11]) + int(f[12]))
        except (OSError, ValueError, IndexError):
            pass
    return out


_MON = dict(last=None, threads=None)


def _gc_line():
    now = time.time()
    th = _threads()
    tick = os.sysconf('SC_CLK_TCK')
    busy = []
    if _MON['threads'] is not None:
        for t, (name, ticks) in th.items():
            d = ticks - _MON['threads'].get(t, (name, ticks))[1]
            if d > 0:
                busy.append((round(d / tick, 2), name, t))
    busy.sort(reverse=True)
    snap = dict(n=list(_GC['n']), secs=[round(x, 3) for x in _GC['secs']], max=[round(x, 4) for x in _GC['max']],
                collected=list(_GC['collected']), long=list(_GC['long']))
    for k in ('n', 'secs', 'collected'):
        _GC[k] = [0, 0, 0] if k != 'secs' else [0.0, 0.0, 0.0]
    _GC['max'], _GC['long'] = [0.0, 0.0, 0.0], []
    line = dict(utc=time.strftime('%H:%M:%S', time.gmtime(now)), interval_s=round(now - _MON['last'], 1) if _MON['last'] else None,
                gc=snap, gc_count=gc.get_count(), blocks=sys.getallocatedblocks(),
                rss_kb=_proc().get('VmRSS_kb'), busiest_threads_cpu_s=busy[:6])
    _MON['last'], _MON['threads'] = now, th
    with open(os.path.join(DIR, 'gcprof-%d.jsonl' % os.getpid()), 'a') as f:
        f.write(json.dumps(line) + '\n')


def _keyname(k):
    if isinstance(k, str):
        import re
        return re.sub(r'[0-9a-f]{6,}|\d+', '#', k)[:40]
    return type(k).__name__


def _owners(min_len=1000, top=300, sample=700):
    """Holders of the large lists (see the module docstring)."""
    import random
    gc.collect()
    objs = gc.get_objects()
    lists = [o for o in objs if type(o) is list]
    hist = Counter()
    elems = Counter()
    for o in lists:
        n = len(o)
        b = '0-99' if n < 100 else '100-999' if n < 1000 else '1K-9K' if n < 10000 else '10K-99K' if n < 100000 else '100K+'
        hist[b] += 1
        elems[b] += n
    big = sorted((o for o in lists if len(o) >= min_len), key=len, reverse=True)
    rest = big[top:]
    random.Random(0).shuffle(rest)
    target = big[:top] + rest[:sample]
    tid = {id(o) for o in target}
    owner_of_dict = {}
    for o in objs:
        if isinstance(o, type):
            continue
        try:
            d = object.__getattribute__(o, '__dict__')
        except Exception:                               # noqa: BLE001
            continue
        if type(d) is dict:
            owner_of_dict[id(d)] = _tmod(type(o)) + '.' + _tqual(type(o))
    holders = {}                                        # id(target or holder) -> [(kind, holder id, label)]

    def scan(want):
        for o in objs:
            t = type(o)
            try:
                if t is dict:
                    for k, v in o.items():
                        if id(v) in want:
                            holders.setdefault(id(v), []).append(('dict', id(o), _keyname(k)))
                elif t in (list, tuple, deque):
                    if id(o) in tid or (len(o) > 100000 and type(o[0]) is int):
                        continue
                    for v in o:
                        if id(v) in want:
                            holders.setdefault(id(v), []).append((t.__name__, id(o), None))
                elif not isinstance(o, type) and hasattr(o, '__dict__'):     # instance attributes (3.11+: often inline)
                    d = object.__getattribute__(o, '__dict__')
                    if type(d) is dict:
                        for k, v in d.items():
                            if id(v) in want:
                                holders.setdefault(id(v), []).append(
                                    ('attr', id(o), '%s.%s.%s' % (_tmod(t), _tqual(t), str(k)[:40])))
                else:
                    for v in gc.get_referents(o):
                        if id(v) in want:
                            holders.setdefault(id(v), []).append(('slots', id(o), _tmod(t) + '.' + _tqual(t)))
            except Exception:                           # noqa: BLE001
                continue
    frontier, seen_ids = set(tid), set()
    for _ in range(3):                                  # the list, its holder, the holder's holder (three levels up)
        scan(frontier)
        seen_ids |= frontier
        nxt = set()
        for x in frontier:
            for kind, hid, label in holders.get(x, []):
                if hid not in seen_ids and not (kind == 'dict' and hid in owner_of_dict):
                    nxt.add(hid)
        if not nxt:
            break
        frontier = nxt

    def describe(kind, hid, label, depth):
        if kind in ('slots', 'attr'):
            here = label
        elif kind == 'dict' and hid in owner_of_dict:
            return '%s.%s' % (owner_of_dict[hid], label)
        else:
            here = ('dict[%s]' % label) if kind == 'dict' else kind
        h = holders.get(hid)
        if depth == 0 or not h:
            return here
        return here + ' <- ' + describe(*h[0], depth - 1)

    by = {}
    for o in target:
        h = holders.get(id(o))
        path = describe(*h[0], 2) if h else '(no tracked holder)'
        b = by.setdefault(path, dict(lists=0, elements=0, examples=[]))
        b['lists'] += 1
        b['elements'] += len(o)
        if len(b['examples']) < 5:
            b['examples'].append(dict(len=len(o), first=type(o[0]).__name__ if o else None, holders=len(h or [])))
    out = dict(objects=len(objs), lists=len(lists), list_elements=sum(elems.values()), by_length=dict(hist),
               elements_by_length=dict(elems), traced=len(target), traced_elements=sum(len(o) for o in target),
               of_big=len(big), big_elements=sum(len(o) for o in big),
               holders=sorted(by.items(), key=lambda kv: -kv[1]['elements'])[:40])
    del objs, lists, big, target
    return out


def _reading(rid, mode):
    import multiprocessing
    roles = [m for m in ('vllm.v1.engine.core', 'vllm.v1.worker.gpu_worker', 'vllm.entrypoints.openai.api_server',
                         'session_service', 'eviction_runner') if m in sys.modules]
    base = os.path.join(DIR, f'memstats-{rid}-{os.getpid()}')
    t0 = time.monotonic()
    out = dict(id=rid, mode=mode, pid=os.getpid(), process=multiprocessing.current_process().name,
               argv=' '.join(sys.argv)[:200], roles=roles, utc=time.strftime('%H:%M:%S', time.gmtime()),
               proc=_proc(), python_allocated_blocks=sys.getallocatedblocks())
    out.update(_glibc(base + '.xml'))
    out['gc'] = _gc_state()
    sm = sys.modules.get('stage2_v1.sessmem')      # 2026-10-08: the API server's sessmem counters (its receipt is saved
    if sm is not None:                             # only at startup)
        out['sessmem'] = dict(getattr(sm, 'STATS', None) or {})
    if mode == 'owners':
        out['owners'] = _owners()
    if mode == 'full':
        out['full'] = _full()
        out['proc_after_gc'] = _proc()
    out['seconds'] = round(time.monotonic() - t0, 3)
    with open(base + '.json.tmp', 'w') as f:
        json.dump(out, f, default=str)
    os.replace(base + '.json.tmp', base + '.json')


def _poll():
    done = set()
    trig = os.path.join(DIR, 'trigger')
    try:                                                # a reading triggered before this process started is not run
        with open(trig) as f:
            done.add(f.read().split()[0])
    except (OSError, IndexError):
        pass
    next_line = time.time() + 60
    while True:
        time.sleep(2.0)
        if time.time() >= next_line:
            next_line += 60
            try:
                _gc_line()
            except Exception:                           # noqa: BLE001
                pass
        try:
            with open(trig) as f:
                words = f.read().split()
            rid, mode = words[:2]
        except (OSError, ValueError):
            continue
        if rid in done:
            continue
        done.add(rid)
        # 2026-10-08 (profiling run): an optional third word "only=<pid>" limits the reading to that process (the owners
        # trace holds the GIL for seconds; the engine core must not stall)
        only = next((w[5:] for w in words[2:] if w.startswith('only=')), None)
        if only is not None and only != str(os.getpid()):
            continue
        try:
            _reading(rid, mode if mode in ('full', 'owners') else 'light')
        except Exception as exc:                        # noqa: BLE001  never disturb the serving process
            try:
                with open(os.path.join(DIR, f'memstats-{rid}-{os.getpid()}.error'), 'w') as f:
                    f.write(repr(exc))
            except OSError:
                pass


def start():
    if not DIR or _STARTED:
        return
    os.makedirs(DIR, exist_ok=True)
    try:
        gc.callbacks.append(_gc_cb)
        _MON['last'], _MON['threads'] = time.time(), _threads()
    except Exception:                                   # noqa: BLE001  the monitor never blocks the start
        pass
    t = threading.Thread(target=_poll, name='arc3-s2-memstats', daemon=True)
    t.start()
    _STARTED.append(t)
