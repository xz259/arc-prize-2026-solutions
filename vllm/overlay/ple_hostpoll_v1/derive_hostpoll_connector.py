"""Derive ple_hostpoll_v1/connector.py from the direct connector (connector_direct_base.py in the same directory)."""
from pathlib import Path
import sys
HERE = Path(sys.argv[1]); t = (HERE / 'connector_direct_base.py').read_text()
def rep(old, new, count=1):
    global t
    assert t.count(old) == count, (old[:90], t.count(old)); t = t.replace(old, new)
rep('arc3 ple_direct_handoff_v1 (2026-09-20): direct staging.', 'arc3 ple_hostpoll_v1 (2026-09-20): direct staging + host-poll results. Builds on ple_direct_handoff_v1 (direct staging).')
rep('No request thread, no enqueue acknowledgement, no per-step ZMQ message. Registration still uses ZMQ once.\n"""',
    'No request thread, no enqueue acknowledgement, no per-step ZMQ message. Results: the offload process gathers rows\n'
    'straight into a shared pinned buffer per layer and publishes a host flag (plain store); the GPU placeholder runs one\n'
    'custom op = polling kernel on that flag + H2D copy on the model stream. No semaphores, no second-context GPU work.\n"""')
rep('''@dataclass
class PleDirectFlagRegistration:
    """Second setup message of the direct handoff: the DP rank's shared pinned request flag."""

    dp_rank: int
    request_flag_buf: torch.Tensor   # int32[4] = [num_tokens, num_reqs, reserved, seq]
''', '''@dataclass
class PleHostpollRegistration:
    """Second setup message: the DP rank's shared request flag, and per-layer shared result buffers + result flags."""

    dp_rank: int
    request_flag_buf: torch.Tensor                 # int32[4] = [num_tokens, num_reqs, reserved, seq]
    result_bufs: dict[str, torch.Tensor]           # layer -> shared pinned [max_tokens, ple_embed_dim] (offload dtype)
    result_flags: dict[str, torch.Tensor]          # layer -> shared pinned int32[1] = seq of the rows present
''')
rep('        self._flag_src: torch.Tensor | None = None   # pinned 2-slot ring of [4] int32 written on the CPU\n',
    '        self._flag_src: torch.Tensor | None = None   # pinned 2-slot ring of [4] int32 written on the CPU\n'
    '        self._result_bufs: dict[str, torch.Tensor] = {}\n        self._result_flags: dict[str, torch.Tensor] = {}\n'
    '        self._expected_dev: dict[str, torch.Tensor] = {}\n        self._timed_out_dev: dict[str, torch.Tensor] = {}\n'
    '        self._dummy_sem_flags: dict[str, torch.Tensor] = {}\n        self._timed_out_host: torch.Tensor | None = None\n')
rep('''        config = vllm_config.model_config.hf_text_config
        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        for layer in layers.values():
            # The CPU worker writes results here through CUDA IPC. The GPU
            # placeholder waits on the paired cross-process semaphore.
            output_buffer = torch.empty(
                max_num_tokens,
                int(config.ple_embed_dim),
                dtype=layer.get_offload_output_dtype(vllm_config.model_config.dtype),
                device=self.device,
            )
            layer.setup_cross_process_offload(
                output_buffer,
                CpuGpuSemaphore(self.device),
            )
        return layers
''', '''        config = vllm_config.model_config.hf_text_config
        max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        for name, layer in layers.items():
            dtype = layer.get_offload_output_dtype(vllm_config.model_config.dtype)
            output_buffer = torch.empty(max_num_tokens, int(config.ple_embed_dim), dtype=dtype, device=self.device)
            # Host-poll: the offload process gathers into this shared CPU buffer (page-locked below) and publishes
            # result_flag = seq; the placeholder polls the flag against expected_dev and copies the rows itself.
            result_buf = torch.zeros(max_num_tokens, int(config.ple_embed_dim), dtype=dtype, device="cpu").share_memory_()
            result_flag = torch.zeros(1, dtype=torch.int32, device="cpu").share_memory_()
            expected_dev = torch.zeros(1, dtype=torch.int32, device=self.device)
            timed_out_dev = torch.zeros(1, dtype=torch.int32, device=self.device)
            self._result_bufs[name] = result_buf
            self._result_flags[name] = result_flag
            self._expected_dev[name] = expected_dev
            self._timed_out_dev[name] = timed_out_dev
            # The unchanged first registration message still carries a semaphore tensor per layer; nobody uses it.
            self._dummy_sem_flags[name] = torch.zeros(1, dtype=torch.int32, device=self.device)
            layer.setup_hostpoll_offload(output_buffer, result_buf, result_flag, expected_dev, timed_out_dev)
        self._timed_out_host = torch.zeros(len(layers), dtype=torch.int32, device="cpu", pin_memory=True)
        return layers
''')
rep('''            sem_flag_tensors={
                name: layer._sem.flag_tensor for name, layer in self._layers.items()
            },''', '''            sem_flag_tensors=dict(self._dummy_sem_flags),''')
rep('''        buffers = [self._input_ids_buf, self._query_start_loc_buf, self._request_flag_buf]
        if self._ngram_context_buf is not None:
            buffers.append(self._ngram_context_buf)
        for buffer in buffers:
            if buffer.device.type != "cpu" or not buffer.is_shared():
                raise RuntimeError("PLE input buffers must be shared CPU tensors")
            if not buffer.is_contiguous():
                raise RuntimeError("PLE input buffers must be contiguous")
            _cuda_check(
                cuda_driver.cuMemHostRegister(
                    buffer.data_ptr(),
                    buffer.numel() * buffer.element_size(),
                    cuda_driver.CU_MEMHOSTREGISTER_PORTABLE,
                ),
                "cuMemHostRegister(PLE input buffer)",
            )
''', '''        buffers = [self._input_ids_buf, self._query_start_loc_buf, self._request_flag_buf]
        if self._ngram_context_buf is not None:
            buffers.append(self._ngram_context_buf)
        buffers.extend(self._result_bufs.values())
        mapped = list(self._result_flags.values())   # polled by a kernel through their device pointers
        for buffer in buffers + mapped:
            if buffer.device.type != "cpu" or not buffer.is_shared():
                raise RuntimeError("PLE input buffers must be shared CPU tensors")
            if not buffer.is_contiguous():
                raise RuntimeError("PLE input buffers must be contiguous")
            flags = cuda_driver.CU_MEMHOSTREGISTER_PORTABLE
            if any(buffer is m for m in mapped):
                flags = flags | cuda_driver.CU_MEMHOSTREGISTER_DEVICEMAP
            _cuda_check(
                cuda_driver.cuMemHostRegister(
                    buffer.data_ptr(),
                    buffer.numel() * buffer.element_size(),
                    flags,
                ),
                "cuMemHostRegister(PLE input buffer)",
            )
''')
rep('''        registration = PleDirectFlagRegistration(dp_rank=self.dp_rank, request_flag_buf=self._request_flag_buf)''',
    '''        registration = PleHostpollRegistration(dp_rank=self.dp_rank, request_flag_buf=self._request_flag_buf,
                                               result_bufs=dict(self._result_bufs), result_flags=dict(self._result_flags))''')
rep('        logger.info("PleOffload(direct): registered request flag for dp_rank=%d", self.dp_rank)',
    '        logger.info("PleOffload(hostpoll): registered request flag + %d result buffer(s) for dp_rank=%d", len(self._result_bufs), self.dp_rank)')
rep('    def _register_flag_with_offload_worker(self) -> None:\n        """Second registration message: the shared pinned request-flag buffer for this DP rank."""',
    '    def _register_flag_with_offload_worker(self) -> None:\n        """Second registration message: request flag + per-layer shared result buffers and flags for this DP rank."""')
rep('''        if self._uses_cuda_inputs:
            self._copy_cuda_inputs(request)
        else:
            self._copy_cpu_inputs(request)
''', '''        if self._uses_cuda_inputs:
            self._copy_cuda_inputs(request)
        else:
            self._copy_cpu_inputs(request)
        # The placeholder's polling kernel waits for result_flag == expected; publish this step's seq on the model
        # stream so it precedes the consumer graph.
        for expected in self._expected_dev.values():
            expected.fill_(request.seq)
''')
rep('''        stream = torch.cuda.current_stream(self.device)
        for layer in self._layers.values():
            layer._gpu_output_buffer[:num_tokens].zero_()
            layer._sem.signal(stream)
''', '''        for name, layer in self._layers.items():
            layer._gpu_output_buffer[:num_tokens].zero_()
            # expected := the flag's current value, so the in-graph poll is satisfied without a CPU result.
            self._expected_dev[name].copy_(self._result_flags[name], non_blocking=True)
''')
rep('''        stream = torch.cuda.current_stream(self.device)
        for layer in self._layers.values():
            layer.release_offloaded_output(stream)
''', '''        # No semaphore to reset. Surface a poll timeout of the previous step (its D2H below has landed by now) and
        # queue this step's timed_out mirror; the model thread never synchronizes.
        assert self._timed_out_host is not None
        if int(self._timed_out_host.max()) != 0:
            raise RuntimeError("PLE host-poll timed out waiting for the offload process; see ple-hostpoll receipts")
        for index, timed_out in enumerate(self._timed_out_dev.values()):
            self._timed_out_host[index : index + 1].copy_(timed_out, non_blocking=True)
''')
# Registration before pinning: ForkingPickler (file_system strategy) may re-share the storage of the result buffers
# and flags, so their final addresses exist only after the second message has been sent (same rule as the original
# connector's first message). 2026-09-20 04:10 launch failure: flag not pinned at forward time.
rep("""            if self.tp_rank == 0:
                # ForkingPickler may replace CPU storage while converting its
                # sharing strategy, so register only the final addresses.
                with torch.accelerator.device_index(self.device.index):
                    self._pin_input_buffers()
                    if self._uses_cuda_inputs:
                        self._d2h_stream = torch.cuda.Stream(device=self.device)
                        self._input_ready_event = torch.cuda.Event()
                        self._d2h_done_event = torch.cuda.Event()
                        self._flag_dev = torch.zeros(4, dtype=torch.int32, device=self.device)
                        self._flag_src = torch.zeros(2, 4, dtype=torch.int32, device="cpu", pin_memory=True)
                self._register_flag_with_offload_worker()
""", """            if self.tp_rank == 0:
                # ForkingPickler may replace CPU storage while converting its
                # sharing strategy, so send every shared buffer first and
                # register only the final addresses.
                self._register_flag_with_offload_worker()
                with torch.accelerator.device_index(self.device.index):
                    self._pin_input_buffers()
                    if self._uses_cuda_inputs:
                        self._d2h_stream = torch.cuda.Stream(device=self.device)
                        self._input_ready_event = torch.cuda.Event()
                        self._d2h_done_event = torch.cuda.Event()
                        self._flag_dev = torch.zeros(4, dtype=torch.int32, device=self.device)
                        self._flag_src = torch.zeros(2, 4, dtype=torch.int32, device="cpu", pin_memory=True)
""")
(HERE / 'connector.py').write_text(t)
import py_compile; py_compile.compile(str(HERE / 'connector.py'), doraise=True)
for bad in ('CpuGpuSemaphore(self.device)', 'PleDirectFlagRegistration', 'layer._sem'):
    assert bad not in t, bad
print('hostpoll connector derived,', len(t.splitlines()), 'lines')
