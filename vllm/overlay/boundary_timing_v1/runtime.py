"""Wall-clock journal of every executed step and every boundary RPC, written from a background thread.

Worker process: GPUModelRunner.execute_model (period between calls, call duration, batch composition),
one_lookahead.worker_settle (settlement fence RPC), engine_eviction.worker_compact (eviction transaction RPC).
Engine-core process: one_lookahead.settle_core (drain + RPC + audit, core side).
Records are JSON lines: {"k": kind, "t": monotonic, "w": wall, "d": duration, ...}."""
import json
import os
import queue
import threading
import time

ENABLED = os.environ.get('ARC3_BOUNDARY_TIMING', '0') == '1'
ROOT = os.environ.get('ARC3_BOUNDARY_TIMING_RECEIPTS', '')
_Q = queue.SimpleQueue()
_THREAD = None
_PREV_STEP = [None]
STATS = dict(version='arc3-boundary-timing-v1', pid=os.getpid(), installed={}, records=0, dropped=0)


def _writer(path):
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    try:
        while True:
            item = _Q.get()
            if item is None:
                break
            try:
                os.write(fd, (json.dumps(item, separators=(',', ':')) + '\n').encode())
                STATS['records'] += 1
            except Exception:
                STATS['dropped'] += 1
    finally:
        os.close(fd)


def _ensure_writer(role):
    global _THREAD
    if _THREAD is None and ROOT:
        os.makedirs(ROOT, exist_ok=True)
        path = os.path.join(ROOT, f'{role}-{os.getpid()}.jsonl')
        _THREAD = threading.Thread(target=_writer, args=(path,), name='arc3-boundary-timing', daemon=True); _THREAD.start()
        with open(os.path.join(ROOT, f'{role}-{os.getpid()}.installed.json'), 'w') as f:
            json.dump(STATS, f, indent=1)


def emit(record):
    if _THREAD is not None:
        _Q.put(record)


def _timed_rpc(module, name, kind, role, extra=None):
    original = getattr(module, name)

    def wrapper(*args, **kwargs):
        _ensure_writer(role)
        t0 = time.monotonic(); w0 = time.time()
        try:
            return original(*args, **kwargs)
        finally:
            rec = dict(k=kind, t=t0, w=w0, d=time.monotonic() - t0)
            if extra is not None:
                try:
                    rec.update(extra(*args, **kwargs))
                except Exception:
                    pass
            emit(rec)

    wrapper.__wrapped__ = original; wrapper.__name__ = name
    setattr(module, name, wrapper)
    STATS['installed'][f'{module.__name__}.{name}'] = kind


def install_model_runner(module):
    """Worker: one record per execute_model call."""
    R = module.GPUModelRunner
    original = R.execute_model

    def execute_model(self, scheduler_output, *args, **kwargs):
        _ensure_writer('worker')
        t0 = time.monotonic(); w0 = time.time()
        try:
            return original(self, scheduler_output, *args, **kwargs)
        finally:
            d = time.monotonic() - t0
            try:
                n = len(scheduler_output.num_scheduled_tokens); tok = scheduler_output.total_num_scheduled_tokens
                new = len(scheduler_output.scheduled_new_reqs); fin = len(scheduler_output.finished_req_ids)
            except Exception:
                n = tok = new = fin = -1
            prev = _PREV_STEP[0]; _PREV_STEP[0] = t0
            emit(dict(k='step', t=t0, w=w0, d=d, p=(t0 - prev) if prev is not None else None, n=n, tok=tok, new=new, fin=fin))

    R.execute_model = execute_model
    STATS['installed']['GPUModelRunner.execute_model'] = 'step'


def install_one_lookahead(module):
    _timed_rpc(module, 'worker_settle', 'worker_settle', 'worker', extra=lambda worker, wires: dict(n=len(wires)))
    _timed_rpc(module, 'settle_core', 'settle_core', 'core', extra=lambda core, outputs, *a, **k: dict(n=len(module.pending(core.scheduler))))


def install_engine_eviction(module):
    if hasattr(module, 'worker_compact'):
        _timed_rpc(module, 'worker_compact', 'worker_compact', 'worker')
    if hasattr(module, 'evict_resident'):
        _timed_rpc(module, 'evict_resident', 'evict_resident', 'core')
