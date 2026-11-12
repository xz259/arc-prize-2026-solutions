"""Offload-process side of the host-poll handoff: accept the hostpoll registration, poll the request flag, run the
CPU lookup straight into the shared result buffer, publish the per-layer result flag (plain store). No CUDA work.
Applied to the executed `vllm.v1.ple_offload.worker` module by the import hook.

2026-10-26 (the engine plan, Claude fork bdfc5402): after the PLE weights load, every PLE layer gets fasthash.py's exact per-token form
of its lookup (the model's forward_impl hashes the padded [requests x longest request] rectangle: ~29x the work on a mixed step), once
fasthash.selfcheck has compared it with the original on 300 random batches against the real table bit for bit (decode, mixed, empty
requests, graph padding, EOS-heavy); any mismatch or error keeps the original lookup. ARC3_PLE_FASTHASH=0 keeps the original
everywhere. The receipt records fasthash (on / fallback / off), the layers and the self-check."""
import json
import os
import pickle
import time

RECEIPT = dict(version='arc3-ple-hostpoll-v1', pid=os.getpid(), role='offload', handled=0, batches=0, skipped_seq=0,
               torn_reads=0, idle_polls=0, poll_sleep_us=None, lookup_us_mean=None, written=None, fasthash=None,
               fasthash_layers=[], fasthash_selfcheck=None)
_LOOKUP_TOTAL = [0.0, 0]


def write_receipt():
    d = os.environ.get('ARC3_PLE_HOSTPOLL_RECEIPTS')
    if not d:
        return
    try:
        os.makedirs(d, exist_ok=True)
        RECEIPT['written'] = time.time()
        RECEIPT['lookup_us_mean'] = (_LOOKUP_TOTAL[0] / _LOOKUP_TOTAL[1] * 1e6) if _LOOKUP_TOTAL[1] else None
        tmp = os.path.join(d, f'offload-{os.getpid()}.json.tmp')
        with open(tmp, 'w') as f:
            json.dump(RECEIPT, f, indent=1)
        os.replace(tmp, os.path.join(d, f'offload-{os.getpid()}.json'))
    except Exception:
        pass


def install(module, registration_cls_name='PleHostpollRegistration'):
    Runner = module.PleOffloadRunner
    original_init = Runner.__init__
    original_accept = Runner.accept_registrations
    original_load = Runner._load_weights

    def _load_weights(self):
        original_load(self)
        if os.environ.get('ARC3_PLE_FASTHASH', '1') == '0':
            RECEIPT['fasthash'] = 'off'
            return
        from . import fasthash
        for name, layer in self._layers.items():
            original = layer.forward_impl
            try:
                t0 = time.perf_counter()
                cases, bad = fasthash.selfcheck(layer, original)
                RECEIPT['fasthash_selfcheck'] = dict(layer=name, cases=cases, mismatches=bad, s=round(time.perf_counter() - t0, 1))
            except Exception as exc:                                  # noqa: BLE001 (the original lookup stays)
                bad = -1
                RECEIPT['fasthash_selfcheck'] = dict(layer=name, error=repr(exc)[:300])
            if bad == 0 and fasthash.install(layer):
                RECEIPT['fasthash_layers'].append(name)
        RECEIPT['fasthash'] = 'on' if RECEIPT['fasthash_layers'] else 'fallback'
        module.logger.info('PleOffload(hostpoll): per-token lookup %s on %s (self-check %s)', RECEIPT['fasthash'],
                           RECEIPT['fasthash_layers'], RECEIPT['fasthash_selfcheck'])

    def __init__(self, vllm_config):
        original_init(self, vllm_config)
        self._hp_request_flags = {}
        self._hp_result_bufs = {}      # dp -> layer -> shared CPU tensor
        self._hp_result_flags = {}     # dp -> layer -> numpy view of shared int32[1]

    def accept_registrations(self, pull_socket, num_workers):
        original_accept(self, pull_socket, num_workers)
        dp_size = self.vllm_config.parallel_config.data_parallel_size
        for _ in range(dp_size):
            item = pickle.loads(pull_socket.recv())
            if type(item).__name__ != registration_cls_name:
                raise RuntimeError(f'Expected {registration_cls_name}, got {type(item).__name__}')
            buf = item.request_flag_buf
            if buf.dtype != module.torch.int32 or buf.numel() != 4 or not buf.is_shared():
                raise RuntimeError('hostpoll request flag must be a shared int32[4] CPU tensor')
            if item.dp_rank in self._hp_request_flags:
                raise RuntimeError(f'duplicate hostpoll registration for dp_rank={item.dp_rank}')
            if set(item.result_bufs) != set(self.layer_names) or set(item.result_flags) != set(self.layer_names):
                raise RuntimeError(f'hostpoll result buffers {sorted(item.result_bufs)} do not match layers {sorted(self.layer_names)}')
            for name in self.layer_names:
                rb, rf = item.result_bufs[name], item.result_flags[name]
                if not rb.is_shared() or not rf.is_shared() or rf.dtype != module.torch.int32 or rf.numel() != 1:
                    raise RuntimeError(f'hostpoll result buffer/flag for {name} must be shared CPU tensors')
            self._hp_request_flags[item.dp_rank] = buf
            self._hp_result_bufs[item.dp_rank] = dict(item.result_bufs)
            self._hp_result_flags[item.dp_rank] = {name: flag.numpy() for name, flag in item.result_flags.items()}
        if set(self._hp_request_flags) != set(range(dp_size)):
            raise RuntimeError(f'hostpoll registrations {sorted(self._hp_request_flags)} != DP ranks {list(range(dp_size))}')
        module.logger.info('PleOffload(hostpoll): registered request flags + result buffers for dp_ranks=%s', sorted(self._hp_request_flags))

    def _handle_hostpoll(self, requests):
        """requests: list of (dp_rank, num_tokens, num_reqs, seq). CPU lookup into the shared result, then flag := seq."""
        if self._clamp_input_ids:
            for dp_rank, num_tokens, _, _ in requests:
                self._input_bufs[dp_rank].input_ids_buf[:num_tokens].clamp_min_(0)
        t0 = time.perf_counter()
        for layer_name, layer in self._layers.items():
            for dp_rank, num_tokens, num_reqs, seq in requests:
                input_bufs = self._input_bufs[dp_rank]
                ngram_context = (input_bufs.ngram_context_buf[:num_reqs] if input_bufs.ngram_context_buf is not None else None)
                layer.forward_impl(input_bufs.input_ids_buf[:num_tokens], input_bufs.input_ids_buf[:num_tokens],
                                   input_bufs.query_start_loc_buf[:num_reqs + 1], ngram_context,
                                   output_buffer=self._hp_result_bufs[dp_rank][layer_name])
                # The gather above is complete (synchronous CPU ops); publishing the flag last is the release.
                self._hp_result_flags[dp_rank][layer_name][0] = seq
        _LOOKUP_TOTAL[0] += time.perf_counter() - t0; _LOOKUP_TOTAL[1] += 1

    def busy_loop(self, pull_socket, shutdown_event):
        poll_sleep = float(os.environ.get('ARC3_PLE_HOSTPOLL_POLL_SLEEP_US', '50')) * 1e-6
        RECEIPT['poll_sleep_us'] = poll_sleep * 1e6
        views = {dp: buf.numpy() for dp, buf in self._hp_request_flags.items()}
        last = {dp: int(view[3]) for dp, view in views.items()}
        module.logger.info('PleOffload(hostpoll): busy-loop polling %d flag(s), sleep %.0f us', len(views), poll_sleep * 1e6)
        write_receipt()
        while not shutdown_event.is_set():
            requests = []
            for dp, view in views.items():
                seq = int(view[3])
                if seq == last[dp]:
                    continue
                num_tokens = int(view[0]); num_reqs = int(view[1]); seq2 = int(view[3])
                if seq2 != seq:
                    RECEIPT['torn_reads'] += 1
                    continue
                if seq != last[dp] + 1:
                    RECEIPT['skipped_seq'] += 1
                last[dp] = seq
                requests.append((dp, num_tokens, num_reqs, seq))
            if requests:
                self._handle_hostpoll(requests)
                RECEIPT['handled'] += len(requests); RECEIPT['batches'] += 1
                if RECEIPT['batches'] % 2000 == 0:
                    write_receipt()
            else:
                RECEIPT['idle_polls'] += 1
                time.sleep(poll_sleep)
        write_receipt()

    Runner._load_weights = _load_weights
    Runner.__init__ = __init__
    Runner.accept_registrations = accept_registrations
    Runner._handle_hostpoll = _handle_hostpoll
    Runner.busy_loop = busy_loop
    Runner._arc3_ple_hostpoll = True
    return Runner
