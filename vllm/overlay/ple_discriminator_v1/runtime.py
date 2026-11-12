"""PLE critical-path discriminator — stage 5 of THROUGHPUT_MEMORY_PLAN_2026-09-19. THROUGHPUT-ONLY; never for quality.

Modes (ARC3_PLE_DISCRIMINATOR):
  off      nothing installed (byte-identical serving).
  lookup   offload process only: PleOffloadRunner._handle_requests skips the CPU hash/gather and the 75 KiB H2D copy and
           signals each output semaphore right after its wait_reset. Keeps the runner's D2H staging, the ZMQ request,
           the worker turnaround and the cross-process stream signal. Isolates the CPU lookup + DMA component.
  handoff  GPU worker only: PleOffloadConnector._launch never submits a request and PleOffloadLayer.forward never
           enqueues cuStreamWaitValue32; the PLE rows stay at the zeros written by the capture-time dummy signal.
           Removes the entire cross-process handoff from the critical path = upper bound for any PLE-critical-path
           implementation (minus that implementation's own GPU work).
  nowait   GPU worker only: PleOffloadLayer.forward never enqueues cuStreamWaitValue32 but the connector still stages,
           acknowledges and sends every request and the offload process still looks up and copies. Isolates the
           in-graph wait op (and any GPU-side stall it causes) from the host-side transport.
  timing   nothing skipped (serving byte-identical): PleOffloadConnector._launch (model thread: event record + enqueue +
           acknowledgement wait), PleOffloadConnector._process_request (request thread: D2H staging incl. its
           synchronize, ZMQ send) and PleOffloadRunner._handle_requests (offload process: lookup + H2D enqueue) are
           timed with perf_counter; count/mean/max/p50/p90 per process go to the receipts every 2,000 calls.
  nogpu    both sides: the GPU worker skips the in-graph wait (as nowait) and still stages/sends every request; the
           offload process performs the CPU hash/gather into its pinned buffer but issues NO GPU work (no wait_reset,
           no H2D, no signal). Isolates the cost of the offload process's per-step activity in its own CUDA context
           (context switching on the shared GPU) from everything else. Throughput-only.
Receipts: one JSON per process under ARC3_PLE_DISCRIMINATOR_RECEIPTS.
"""
import json
import os
import sys
import time

MODES = ('off', 'lookup', 'handoff', 'nowait', 'timing', 'nogpu')
MODE = os.environ.get('ARC3_PLE_DISCRIMINATOR', 'off')
assert MODE in MODES, MODE
PINS = {   # runtime/base vLLM sources, 2026-09-19 (pod e1e22a0755d6)
    'vllm.v1.ple_offload.worker': 'f93a0a8c40ee3c536e60cb027f5d6e76cde0a51e62b540c82e2c15b2a72b7f5f',
    'vllm.v1.ple_offload.connector': 'b911d00cf4a39556d946e86ac34ea70e6230683b22adf5ef14ac299e9ef36206',
    'vllm.model_executor.layers.ple_offload_layer': '9c0ab215116d0e1b2dbc4c337a51339946b186a41d35f8480d762e2d123acfc1',
}
REPORT = dict(version='arc3-ple-discriminator-v1', mode=MODE, pid=os.getpid(), argv0=sys.argv[0][-60:],
              installed={}, handled_requests=0, skipped_launches=0, errors=[], written=None)


def write_receipt():
    d = os.environ.get('ARC3_PLE_DISCRIMINATOR_RECEIPTS')
    if not d:
        return
    try:
        os.makedirs(d, exist_ok=True)
        REPORT['written'] = time.time()
        tmp = os.path.join(d, f'proc-{os.getpid()}.json.tmp')
        with open(tmp, 'w') as f:
            json.dump(REPORT, f, indent=1)
        os.replace(tmp, os.path.join(d, f'proc-{os.getpid()}.json'))
    except Exception as e:   # receipts must never break serving
        REPORT['errors'].append(f'receipt: {e!r}')


def install_worker(module):
    """mode=lookup: the offload process signals immediately after wait_reset; no hash/gather, no H2D."""
    W = module.PleOffloadRunner   # the in-process runner (busy_loop/_handle_requests); PleOffloadWorker only spawns/manages it
    assert not getattr(W, '_arc3_ple_discriminator', False)
    original = W._handle_requests

    def _handle_requests(self, requests):
        requests_by_dp = {}
        for request in requests:
            if request.dp_rank not in self._worker_targets or request.dp_rank in requests_by_dp:
                continue   # original logs and skips; the diagnostic just skips
            requests_by_dp[request.dp_rank] = request
        for layer_name in self._layers:
            for dp_rank in requests_by_dp:
                for target in self._worker_targets[dp_rank][layer_name]:
                    # Same ordering as the original: never signal before the GPU released the previous output.
                    target.copy_stream.synchronize()
                    target.sem.wait_reset(target.copy_stream)
                    target.sem.signal(target.copy_stream)
        REPORT['handled_requests'] += len(requests_by_dp)
        if REPORT['handled_requests'] % 2000 < len(requests_by_dp):
            write_receipt()

    W._handle_requests = _handle_requests
    W._arc3_ple_discriminator = 'lookup'
    W._arc3_original_handle_requests = original
    REPORT['installed']['PleOffloadRunner._handle_requests'] = 'lookup-skip'
    write_receipt()


def install_layer(module):
    """mode=handoff: the GPU placeholder returns its buffer without cuStreamWaitValue32."""
    L = module.PleOffloadLayer
    assert not getattr(L, '_arc3_ple_discriminator', False)
    original = L.forward

    def forward(self, hidden_states, input_ids, *args, **kwargs):
        if self._is_cpu_offloaded:
            return self._gpu_output_buffer[: input_ids.shape[0]]
        return self.forward_impl(hidden_states, input_ids, *args, **kwargs)

    L.forward = forward
    L._arc3_ple_discriminator = MODE   # 'handoff', 'nowait' or 'nogpu'
    L._arc3_original_forward = original
    REPORT['installed']['PleOffloadLayer.forward'] = 'no-wait'
    write_receipt()


def install_connector(module):
    """mode=handoff: never stage inputs or send a PLE request (the wait is gone, so nothing consumes it)."""
    C = module.PleOffloadConnector
    assert not getattr(C, '_arc3_ple_discriminator', False)
    original = C._launch

    def _launch(self, num_reqs, num_tokens):
        REPORT['skipped_launches'] += 1
        if REPORT['skipped_launches'] % 2000 == 1:
            write_receipt()

    C._launch = _launch
    C._arc3_ple_discriminator = 'handoff'
    C._arc3_original_launch = original
    REPORT['installed']['PleOffloadConnector._launch'] = 'no-request'
    write_receipt()


class _Timer:
    """Reservoir-sampled wall-clock statistics for one hot method (microseconds)."""
    def __init__(self, name, keep=4096):
        self.name = name; self.count = 0; self.total = 0.0; self.max = 0.0; self.keep = keep; self.samples = []
    def add(self, seconds):
        us = seconds * 1e6; self.count += 1; self.total += us; self.max = max(self.max, us)
        if len(self.samples) < self.keep:
            self.samples.append(us)
        else:
            import random
            j = random.randrange(self.count)
            if j < self.keep: self.samples[j] = us
    def summary(self):
        s = sorted(self.samples); n = len(s)
        pick = lambda q: (s[min(n - 1, int(q * n))] if n else None)
        return dict(count=self.count, mean_us=(self.total / self.count if self.count else None), max_us=self.max,
                    p50_us=pick(0.5), p90_us=pick(0.9), p99_us=pick(0.99), sampled=n)


TIMERS = {}


def _timed(cls, method, label):
    original = getattr(cls, method); timer = TIMERS.setdefault(label, _Timer(label))

    def wrapper(self, *args, **kwargs):
        t0 = time.perf_counter()
        try:
            return original(self, *args, **kwargs)
        finally:
            timer.add(time.perf_counter() - t0)
            if timer.count % 2000 == 0:
                REPORT['timing'] = {k: v.summary() for k, v in TIMERS.items()}
                write_receipt()

    setattr(cls, method, wrapper)
    REPORT['installed'][f'{cls.__name__}.{method}'] = f'timed:{label}'


def install_timing_connector(module):
    C = module.PleOffloadConnector
    assert not getattr(C, '_arc3_ple_discriminator', False)
    _timed(C, '_launch', 'model_thread_launch_incl_ack')
    _timed(C, '_process_request', 'request_thread_stage_and_send')
    _timed(C, '_copy_cuda_inputs', 'request_thread_d2h_stage_sync')
    C._arc3_ple_discriminator = 'timing'
    write_receipt()


def install_timing_worker(module):
    W = module.PleOffloadRunner
    assert not getattr(W, '_arc3_ple_discriminator', False)
    _timed(W, '_handle_requests', 'offload_process_handle_lookup_h2d')
    W._arc3_ple_discriminator = 'timing'
    write_receipt()


def install_worker_nogpu(module):
    """mode=nogpu: offload runner does the CPU lookup into its pinned buffer and touches no CUDA stream."""
    W = module.PleOffloadRunner
    assert not getattr(W, '_arc3_ple_discriminator', False)
    original = W._handle_requests

    def _handle_requests(self, requests):
        requests_by_dp = {}
        for request in requests:
            if request.dp_rank not in self._worker_targets or request.dp_rank in requests_by_dp:
                continue
            requests_by_dp[request.dp_rank] = request
        if self._clamp_input_ids:
            for dp_rank, request in requests_by_dp.items():
                self._input_bufs[dp_rank].input_ids_buf[: request.num_tokens].clamp_min_(0)
        for layer_name, layer in self._layers.items():
            for dp_rank, request in requests_by_dp.items():
                input_bufs = self._input_bufs[dp_rank]
                ngram_context = (input_bufs.ngram_context_buf[: request.num_reqs]
                                 if input_bufs.ngram_context_buf is not None else None)
                layer.forward_impl(input_bufs.input_ids_buf[: request.num_tokens], input_bufs.input_ids_buf[: request.num_tokens],
                                   input_bufs.query_start_loc_buf[: request.num_reqs + 1], ngram_context,
                                   output_buffer=self._pinned_bufs[dp_rank][layer_name])
        REPORT['handled_requests'] += len(requests_by_dp)
        if REPORT['handled_requests'] % 2000 < len(requests_by_dp):
            write_receipt()

    W._handle_requests = _handle_requests
    W._arc3_ple_discriminator = 'nogpu'
    W._arc3_original_handle_requests = original
    REPORT['installed']['PleOffloadRunner._handle_requests'] = 'cpu-lookup-no-gpu'
    write_receipt()


def install_for(module_name, module):
    """Dispatch by module and mode; called by the import hook after the module executed."""
    if MODE == 'lookup' and module_name == 'vllm.v1.ple_offload.worker':
        install_worker(module)
    elif MODE == 'handoff' and module_name == 'vllm.model_executor.layers.ple_offload_layer':
        install_layer(module)
    elif MODE == 'handoff' and module_name == 'vllm.v1.ple_offload.connector':
        install_connector(module)
    elif MODE == 'nowait' and module_name == 'vllm.model_executor.layers.ple_offload_layer':
        install_layer(module)
    elif MODE == 'nogpu' and module_name == 'vllm.model_executor.layers.ple_offload_layer':
        install_layer(module)
    elif MODE == 'nogpu' and module_name == 'vllm.v1.ple_offload.worker':
        install_worker_nogpu(module)
    elif MODE == 'timing' and module_name == 'vllm.v1.ple_offload.connector':
        install_timing_connector(module)
    elif MODE == 'timing' and module_name == 'vllm.v1.ple_offload.worker':
        install_timing_worker(module)
