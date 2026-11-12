"""Offload-process side of ple_hostpoll_tp (L4 port, 2026-10-31, ARC2 - main): production's ple_hostpoll_v1 worker patch,
then two of its methods replaced. Installed by install.py as `ple_hostpoll_v1.worker_patch`, so production's import hook
applies it to the executed `vllm.v1.ple_offload.worker` module (production's files are read, never written).

  accept_registrations: CUDA-free. Production's version calls the base accept first, which opens every TP rank's CUDA IPC
    output buffers and semaphores and makes a copy stream and a pinned staging buffer per layer and DP rank: a CUDA context
    on every GPU of the PLE stage (238 / 206 MiB on the L4's GPUs 0-1), none of it used by the host-poll busy loop. Here the
    first registrations must carry no CUDA tensors (the TP connector sends none); TP rank 0's shared inputs are kept per
    DP rank; then production's host-poll registrations (one per DP rank: the request flag and the shared result buffers +
    flags that every TP rank polls), plus the verify mode's aux buffers when present.
  _handle_hostpoll: production's, plus aux = [seq, checksum of the rows' int16 view] written before the flag (verify mode).
busy_loop, __init__ and _load_weights (fasthash) stay production's."""
import hashlib
import importlib.util
import os
import pickle
import time

PROD = "/workspace/arc3-inference/runs/dense_fp8_v1/ple_hostpoll_v1/worker_patch.py"
PROD_MD5 = "68839d031f7e621e079f54c15748adcd"
_PROD = None


def _prod():
    global _PROD
    if _PROD is None:
        raw = open(PROD, "rb").read()
        assert hashlib.md5(raw).hexdigest() == PROD_MD5, ("production ple_hostpoll_v1 worker_patch changed", PROD)
        # loaded inside production's package so its relative import (`from . import fasthash`) resolves there
        spec = importlib.util.spec_from_file_location("ple_hostpoll_v1._worker_patch_prod", PROD)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _PROD = mod
    return _PROD


def install(module, registration_cls_name='PleHostpollRegistration'):
    prod = _prod()
    Runner = prod.install(module, registration_cls_name)
    torch = module.torch
    prod.RECEIPT['l4port_tp'] = dict(accept='cuda-free', worker_patch_tp=os.path.abspath(__file__), aux=None,
                                     cuda_visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'))
    if os.environ.get('ARC3_L4PORT_PLE_HPTP_NOCUDA', '0') == '1':
        # Test gate (the reviewer, 18:5x): spawn the offload process with no visible GPU, so any CUDA use there fails loudly.
        W = module.PleOffloadWorker
        orig_make_process = W.make_process

        def make_process(*args, **kwargs):
            old = os.environ.get('CUDA_VISIBLE_DEVICES')
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            try:
                return orig_make_process(*args, **kwargs)
            finally:
                if old is None:
                    os.environ.pop('CUDA_VISIBLE_DEVICES', None)
                else:
                    os.environ['CUDA_VISIBLE_DEVICES'] = old

        W.make_process = staticmethod(make_process)
        prod.RECEIPT['l4port_tp']['nocuda_spawn'] = True

    def accept_registrations(self, pull_socket, num_workers):
        parallel = self.vllm_config.parallel_config
        dp_size, tp_size = parallel.data_parallel_size, parallel.tensor_parallel_size
        if num_workers != dp_size * tp_size:
            raise RuntimeError(f'Expected {dp_size * tp_size} registrations for DP={dp_size}, TP={tp_size}, got {num_workers}')
        module.logger.info('PleOffload(hostpoll-tp): waiting for %d GPU worker registration(s) (no CUDA IPC) ...', num_workers)
        registrations = []
        for index in range(num_workers):
            item = pickle.loads(pull_socket.recv())
            if not isinstance(item, module.PleOffloadRegistration):
                raise RuntimeError(f'Expected PleOffloadRegistration during setup, got {type(item).__name__} ({index + 1}/{num_workers})')
            if item.gpu_output_buffers or item.sem_flag_tensors:
                raise RuntimeError('hostpoll-tp: a registration carries CUDA tensors (the worker side is not the TP connector)')
            registrations.append(item)
            module.logger.info('GPU worker %d registered (dp_rank=%d, tp_rank=%d).', item.worker_id, item.dp_rank, item.tp_rank)
        by_dp = {}
        for item in registrations:
            by_dp.setdefault(item.dp_rank, set()).add(item.tp_rank)
        if set(by_dp) != set(range(dp_size)) or any(ranks != set(range(tp_size)) for ranks in by_dp.values()):
            raise RuntimeError(f'hostpoll-tp: registrations by DP rank {by_dp} do not cover DP={dp_size} x TP={tp_size}')
        for item in registrations:
            # All TP ranks of a DP group receive the same input, so TP rank zero's shared buffers are its source.
            if item.tp_rank == 0:
                self._input_bufs[item.dp_rank] = module.PleOffloadInputBuffers(
                    input_ids_buf=item.input_ids_buf, query_start_loc_buf=item.query_start_loc_buf,
                    ngram_context_buf=item.ngram_context_buf)
        self._hp_aux = {}
        for _ in range(dp_size):
            item = pickle.loads(pull_socket.recv())
            if type(item).__name__ != registration_cls_name:
                raise RuntimeError(f'Expected {registration_cls_name}, got {type(item).__name__}')
            buf = item.request_flag_buf
            if buf.dtype != torch.int32 or buf.numel() != 4 or not buf.is_shared():
                raise RuntimeError('hostpoll request flag must be a shared int32[4] CPU tensor')
            if item.dp_rank in self._hp_request_flags:
                raise RuntimeError(f'duplicate hostpoll registration for dp_rank={item.dp_rank}')
            if set(item.result_bufs) != set(self.layer_names) or set(item.result_flags) != set(self.layer_names):
                raise RuntimeError(f'hostpoll result buffers {sorted(item.result_bufs)} do not match layers {sorted(self.layer_names)}')
            for name in self.layer_names:
                rb, rf = item.result_bufs[name], item.result_flags[name]
                if not rb.is_shared() or not rf.is_shared() or rf.dtype != torch.int32 or rf.numel() != 1:
                    raise RuntimeError(f'hostpoll result buffer/flag for {name} must be shared CPU tensors')
            aux = getattr(item, 'aux_bufs', None)
            if aux:
                if set(aux) != set(self.layer_names):
                    raise RuntimeError(f'hostpoll-tp aux buffers {sorted(aux)} do not match layers {sorted(self.layer_names)}')
                for name, a in aux.items():
                    if not a.is_shared() or a.dtype != torch.int64 or a.numel() != 2:
                        raise RuntimeError(f'hostpoll-tp aux buffer for {name} must be a shared int64[2] CPU tensor')
                self._hp_aux[item.dp_rank] = dict(aux)
            self._hp_request_flags[item.dp_rank] = buf
            self._hp_result_bufs[item.dp_rank] = dict(item.result_bufs)
            self._hp_result_flags[item.dp_rank] = {name: flag.numpy() for name, flag in item.result_flags.items()}
        if set(self._hp_request_flags) != set(range(dp_size)):
            raise RuntimeError(f'hostpoll registrations {sorted(self._hp_request_flags)} != DP ranks {list(range(dp_size))}')
        prod.RECEIPT['l4port_tp']['aux'] = sorted(self._hp_aux) or None
        module.logger.info('PleOffload(hostpoll-tp): registered request flags + shared result buffers for dp_ranks=%s '
                           '(TP=%d, aux %s); no CUDA used', sorted(self._hp_request_flags), tp_size, sorted(self._hp_aux))

    def _handle_hostpoll(self, requests):
        """requests: list of (dp_rank, num_tokens, num_reqs, seq). CPU lookup into the shared result, (aux,) then flag := seq."""
        if self._clamp_input_ids:
            for dp_rank, num_tokens, _, _ in requests:
                self._input_bufs[dp_rank].input_ids_buf[:num_tokens].clamp_min_(0)
        t0 = time.perf_counter()
        aux_all = getattr(self, '_hp_aux', None) or {}
        for layer_name, layer in self._layers.items():
            for dp_rank, num_tokens, num_reqs, seq in requests:
                input_bufs = self._input_bufs[dp_rank]
                ngram_context = (input_bufs.ngram_context_buf[:num_reqs] if input_bufs.ngram_context_buf is not None else None)
                out = self._hp_result_bufs[dp_rank][layer_name]
                layer.forward_impl(input_bufs.input_ids_buf[:num_tokens], input_bufs.input_ids_buf[:num_tokens],
                                   input_bufs.query_start_loc_buf[:num_reqs + 1], ngram_context, output_buffer=out)
                aux = aux_all.get(dp_rank, {}).get(layer_name)
                if aux is not None:   # verify mode: the checksum, then its seq, both before the flag
                    aux[1] = int(out[:num_tokens].view(torch.int16).sum(dtype=torch.int64))
                    aux[0] = seq
                # The gather above is complete (synchronous CPU ops); publishing the flag last is the release.
                self._hp_result_flags[dp_rank][layer_name][0] = seq
        prod._LOOKUP_TOTAL[0] += time.perf_counter() - t0; prod._LOOKUP_TOTAL[1] += 1

    if os.environ.get('ARC3_L4PORT_PLE_HPTP_METAINIT', '1') == '1' and hasattr(module, 'initialize_model'):
        # 2026-10-31 (ph1 on the 4090, the no-GPU spawn gate): the offload process's model-structure build (under the meta
        # device) still initializes CUDA, because the GDN layers build their gated RMSNorm with device=current_platform.
        # current_device() (an explicit cuda:0, which overrides the meta default): a CUDA context on GPU 0. While the
        # structure is built, current_device() returns the meta device here (vLLM forwards the name to torch.cuda;
        # an instance attribute shadows that for the call). The PLE weights still load on the CPU afterwards.
        orig_initialize_model = module.initialize_model

        def initialize_model(*args, **kwargs):
            from vllm.platforms import current_platform
            current_platform.current_device = lambda: torch.device('meta')
            try:
                return orig_initialize_model(*args, **kwargs)
            finally:
                try:
                    del current_platform.current_device
                except AttributeError:
                    pass

        module.initialize_model = initialize_model
        prod.RECEIPT['l4port_tp']['metainit'] = True

    Runner.accept_registrations = accept_registrations
    Runner._handle_hostpoll = _handle_hostpoll
    Runner._arc3_ple_hostpoll_tp = True
    return Runner
