"""stage2_v1 'stepev' part (2026-10-15, item 6 baseline; diagnostics only): a profiler-free GPU timeline of every step from
CUDA events on the worker's step stream, to size the decode step's GPU idle without the torch profiler's host inflation.

ps1t's trace (10-15): a plain decode step is eager prep -> target graph (43.6 ms) -> eager sampler -> 3 drafter graphs with
eager glue -> eager tail. The seams between the target graph, the sampler and the drafter graphs cost ~2-4 us each (the host
runs ahead). The idle (~2.2 ms per step under the profiler) sits at the step boundary (~0.5 ms before the step starts, ~0.77
ms of gaps between ~100 small eager prep launches) and inside the target graph (~0.5 ms). The profiler slows host launches,
so the boundary is measured here without it.

Events (timing-enabled, recorded on the current stream; never during graph capture or dummy runs):
  A  execute_model entry (before any launch of the step)    B / C  around the target's FULL-graph replay
  D  sample_tokens entry                                     E  sample_tokens exit (after the drafter launches)
GPU intervals per step (ms): boundary = E(previous step) -> A (the GPU waited for the host to begin this step; ~0 when the
host enqueued the step before the GPU drained), prep = A -> B (eager prep: kernel time + launch gaps), target = B -> C,
cd = C -> D, sample = D -> E (sampler + drafter passes). Host intervals between the same points (perf_counter) alongside.
A daemon thread writes completed steps (all events done; no synchronisation on the step path) to
$ARC3_S2_STEPEV_DIR/stepev-<pid>.jsonl every 2 s. Steps without a target replay (eager / mixed) carry no B / C."""
import collections
import json
import os
import threading
import time

PINS = {'vllm.v1.worker.gpu.model_runner': 'f08ba0f5ad41c8b9d5145fb1fba115ffd97f35f1e59017602e235ec0823abce8',
        'vllm.v1.worker.gpu.cudagraph_utils': 'ee1f6eb37bc2f456a3e7e9142d5a53455afed2250a57f52780546cf1aef27e2c'}
DIR = os.environ.get('ARC3_S2_STEPEV_DIR')
STATS = dict(steps=0, written=0, skipped_capture=0, errors=0, last_error=None)
_LOCAL = threading.local()
_LAST = {}                      # the step awaiting its sample_tokens
_DONE = collections.deque()     # steps with all events recorded, awaiting completion on the GPU
_STARTED = []


def _event():
    import torch
    e = torch.cuda.Event(enable_timing=True)
    e.record()
    return e


def _writer():
    prev_e = None
    path = os.path.join(DIR, f'stepev-{os.getpid()}.jsonl')
    while True:
        time.sleep(2.0)
        out = []
        try:
            while _DONE:
                rec = _DONE[0]
                ev = rec['ev']
                if not ev['E'].query():
                    break
                _DONE.popleft()
                h = rec['host']
                row = dict(t=h['A'], new=rec['new'], tok=rec['tok'])
                row['prep'] = ev['A'].elapsed_time(ev['B']) if 'B' in ev else None
                row['target'] = ev['B'].elapsed_time(ev['C']) if 'C' in ev else None
                row['cd'] = ev['C'].elapsed_time(ev['D']) if 'C' in ev else ev['A'].elapsed_time(ev['D'])
                row['sample'] = ev['D'].elapsed_time(ev['E'])
                row['boundary'] = prev_e.elapsed_time(ev['A']) if prev_e is not None else None
                row['h_prep'] = (h['B'] - h['A']) * 1e3 if 'B' in h else None
                row['h_exec'] = (h['X'] - h['A']) * 1e3
                row['h_x_to_d'] = (h['D'] - h['X']) * 1e3
                row['h_sample'] = (h['E'] - h['D']) * 1e3
                row['h_from_prev'] = (h['A'] - rec['prev_host_e']) * 1e3 if rec.get('prev_host_e') else None
                prev_e = ev['E']
                out.append(row)
        except Exception as exc:                        # noqa: BLE001  diagnostics only
            STATS['errors'] += 1
            STATS['last_error'] = repr(exc)[:300]
        if out:
            try:
                with open(path, 'a') as f:
                    for row in out:
                        f.write(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()},
                                           separators=(',', ':')) + '\n')
                STATS['written'] += len(out)
            except Exception as exc:                    # noqa: BLE001
                STATS['errors'] += 1
                STATS['last_error'] = repr(exc)[:300]


def _start():
    if _STARTED or not DIR:
        return
    os.makedirs(DIR, exist_ok=True)
    threading.Thread(target=_writer, name='arc3-s2-stepev', daemon=True).start()
    _STARTED.append(True)


def patch_model_runner(mod):
    import torch
    cls = mod.GPUModelRunner
    orig_exec, orig_sample = cls.execute_model, cls.sample_tokens
    last_host_e = [None]

    def execute_model(self, scheduler_output, *args, **kwargs):
        if (not DIR or kwargs.get('dummy_run') or (len(args) > 1 and args[1])
                or torch.cuda.is_current_stream_capturing()):
            STATS['skipped_capture'] += 1
            return orig_exec(self, scheduler_output, *args, **kwargs)
        _start()
        rec = dict(new=len(scheduler_output.scheduled_new_reqs or ()), tok=scheduler_output.total_num_scheduled_tokens,
                   ev={}, host={}, prev_host_e=last_host_e[0])
        rec['host']['A'] = time.perf_counter()
        rec['ev']['A'] = _event()
        _LOCAL.rec = rec
        try:
            return orig_exec(self, scheduler_output, *args, **kwargs)
        finally:
            _LOCAL.rec = None
            rec['host']['X'] = time.perf_counter()
            _LAST['rec'] = rec

    def sample_tokens(self, *args, **kwargs):
        rec = _LAST.pop('rec', None)
        if rec is None or torch.cuda.is_current_stream_capturing():
            return orig_sample(self, *args, **kwargs)
        rec['host']['D'] = time.perf_counter()
        rec['ev']['D'] = _event()
        try:
            return orig_sample(self, *args, **kwargs)
        finally:
            rec['ev']['E'] = _event()
            rec['host']['E'] = last_host_e[0] = time.perf_counter()
            STATS['steps'] += 1
            _DONE.append(rec)

    cls.execute_model = execute_model
    cls.sample_tokens = sample_tokens


def patch_cudagraph_utils(mod):
    cls = mod.CudaGraphManager
    orig = cls.run_fullgraph

    def run_fullgraph(self, desc):
        rec = getattr(_LOCAL, 'rec', None)
        if rec is None or 'B' in rec['ev']:             # drafter replays (inside sample_tokens) and later replays
            return orig(self, desc)
        rec['host']['B'] = time.perf_counter()
        rec['ev']['B'] = _event()
        try:
            return orig(self, desc)
        finally:
            rec['ev']['C'] = _event()
            rec['host']['C'] = time.perf_counter()

    cls.run_fullgraph = run_fullgraph


PATCHERS = {'vllm.v1.worker.gpu.model_runner': patch_model_runner, 'vllm.v1.worker.gpu.cudagraph_utils': patch_cudagraph_utils}
