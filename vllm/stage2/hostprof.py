"""stage2_v1 hostprof: trigger-file sampling profiler of the Python threads in every stage-2 process (overnight plan
step 4, 2026-09-30: where the host spends the GPU-idle time of gameplay steps; the torch profiler only sees the worker).

Off unless ARC3_S2_HOSTPROF_DIR is set. Then each process (API server, engine core, GPU worker, PLE helper) runs a
daemon thread that polls <dir>/trigger once a second. A line '<window id> <seconds>' starts one window per new id:
sys._current_frames() sampled every 5 ms for that many seconds, then <dir>/hostprof-<id>-<pid>.json with the process
name and role modules, and per thread: samples, the share whose innermost Python frame is a known wait (select, poll,
recv, wait, sleep, get, acquire, ...), the top innermost ("self") and cumulative functions, and the top collapsed
stacks. No CUDA access, no synchronisation; each sample holds the GIL for a frame walk (tens of microseconds)."""
import json
import os
import sys
import threading
import time
from collections import Counter

DIR = os.environ.get('ARC3_S2_HOSTPROF_DIR')
PERIOD = 0.005
WAITS = ('select', 'poll', 'epoll', 'recv', 'recv_multipart', 'wait', 'sleep', 'get', 'acquire', 'read', 'readinto',
         'accept', '_recv_msg', 'result', 'join', 'dequeue', 'wait_for', '_wait_for_tstate_lock')
_STARTED = []


def _label(code):
    return f'{os.path.basename(code.co_filename)}:{code.co_name}'


def _window(wid, seconds):
    me = threading.get_ident()
    names = {t.ident: t.name for t in threading.enumerate()}
    per = {}
    t_end = time.monotonic() + seconds
    n = 0
    while time.monotonic() < t_end:
        for tid, frame in sys._current_frames().items():
            if tid == me:
                continue
            stack = []
            f = frame
            while f is not None and len(stack) < 64:
                stack.append(_label(f.f_code))
                f = f.f_back
            d = per.get(tid)
            if d is None:
                d = per[tid] = dict(samples=0, waiting=0, self=Counter(), cum=Counter(), stacks=Counter())
            d['samples'] += 1
            inner = stack[0] if stack else '?'
            if inner.rsplit(':', 1)[-1] in WAITS:
                d['waiting'] += 1
            d['self'][inner] += 1
            for fn in set(stack):
                d['cum'][fn] += 1
            d['stacks'][' <- '.join(stack[:24])] += 1
        n += 1
        time.sleep(PERIOD)
    names.update({t.ident: t.name for t in threading.enumerate()})
    import multiprocessing
    roles = [m for m in ('vllm.v1.engine.core', 'vllm.v1.worker.gpu_worker', 'vllm.entrypoints.openai.api_server',
                         'eviction_runner', 'engine_patch') if m in sys.modules]
    out = dict(window=wid, seconds=seconds, pid=os.getpid(), process=multiprocessing.current_process().name,
               argv=' '.join(sys.argv)[:200], roles=roles, rounds=n, threads={})
    for tid, d in sorted(per.items(), key=lambda x: -x[1]['samples']):
        busy = d['samples'] - d['waiting']
        out['threads'][f'{names.get(tid, "?")}:{tid}'] = dict(
            samples=d['samples'], waiting_share=round(d['waiting'] / max(d['samples'], 1), 4), busy_samples=busy,
            self=d['self'].most_common(150), cum=d['cum'].most_common(300), stacks=d['stacks'].most_common(400))
    path = os.path.join(DIR, f'hostprof-{wid}-{os.getpid()}.json')
    with open(path + '.tmp', 'w') as f:
        json.dump(out, f)
    os.replace(path + '.tmp', path)


def _poll():
    done = set()
    trig = os.path.join(DIR, 'trigger')
    try:                                                # a window triggered before this process started is not run
        with open(trig) as f:
            done.add(f.read().split()[0])
    except (OSError, IndexError):
        pass
    while True:
        time.sleep(1.0)
        try:
            with open(trig) as f:
                wid, secs = f.read().split()[:2]
        except (OSError, ValueError):
            continue
        if wid in done:
            continue
        done.add(wid)
        try:
            _window(wid, min(float(secs), 120.0))
        except Exception as exc:                        # never disturb the serving process
            try:
                with open(os.path.join(DIR, f'hostprof-{wid}-{os.getpid()}.error'), 'w') as f:
                    f.write(repr(exc))
            except OSError:
                pass


def start():
    if not DIR or _STARTED:
        return
    os.makedirs(DIR, exist_ok=True)
    t = threading.Thread(target=_poll, name='arc3-s2-hostprof', daemon=True)
    t.start()
    _STARTED.append(t)
