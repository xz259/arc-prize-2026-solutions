# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Exchange PLE data between a GPU worker and the CPU-offload process.

arc3 ple_hostpoll_tp (2026-10-31, L4 port, ARC2 - main): ple_hostpoll_v1 for TP > 1, derived by
l4port/ple_hostpoll_tp/derive_tp_connector.py (see there): no CUDA tensors in the first registration; every TP rank
polls TP rank 0's shared result buffers; an opt-in verify mode (ARC3_L4PORT_PLE_HP_VERIFY=1).

arc3 ple_hostpoll_v1 (2026-09-20): direct staging + host-poll results. Builds on ple_direct_handoff_v1 (direct staging). The model thread records input readiness and enqueues the
three D2H staging copies plus a 16-byte flag DMA on the D2H stream itself; the offload process polls the pinned flag.
No request thread, no enqueue acknowledgement, no per-step ZMQ message. Results: the offload process gathers rows
straight into a shared pinned buffer per layer and publishes a host flag (plain store); the GPU placeholder runs one
custom op = polling kernel on that flag + H2D copy on the model stream. No semaphores, no second-context GPU work.
"""

import os
from dataclasses import dataclass
from multiprocessing.reduction import ForkingPickler
from typing import Any

import msgspec
import torch
import torch.nn as nn
import zmq
from cuda.bindings import driver as cuda_driver

from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_dp_group, get_tp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.ple_offload_layer import (
    CpuGpuSemaphore,
    PleOffloadLayer,
)
from vllm.v1.ple_offload.protocol import (
    PleOffloadRegistration,
    PleOffloadRequest,
)

logger = init_logger(__name__)


@dataclass
class PleHostpollRegistration:
    """Second setup message: the DP rank's shared request flag, and per-layer shared result buffers + result flags."""

    dp_rank: int
    request_flag_buf: torch.Tensor                 # int32[4] = [num_tokens, num_reqs, reserved, seq]
    result_bufs: dict[str, torch.Tensor]           # layer -> shared pinned [max_tokens, ple_embed_dim] (offload dtype)
    result_flags: dict[str, torch.Tensor]          # layer -> shared pinned int32[1] = seq of the rows present
    aux_bufs: dict[str, torch.Tensor] | None = None   # L4 port verify mode: layer -> shared int64[2] = [seq, checksum]


@dataclass
class PleDirectRequest:
    """Per-step request; transported through the flag buffer, never through ZMQ."""

    dp_rank: int
    num_tokens: int
    num_reqs: int
    seq: int


def _cuda_check(result: Any, operation: str) -> Any:
    """Check the ``(CUresult, ...)`` tuple returned by cuda-python calls."""
    error = result[0] if isinstance(result, tuple) else result
    if error.value != 0:
        raise RuntimeError(f"{operation} failed: {error}")
    return result


class PleOffloadConnector:
    """Connect a GPU runner to the shared PLE CPU worker.

    MRV1 and MRV2 share the same CPU-input and CUDA-output IPC protocol.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
        device: torch.device,
        ipc_addr: str,
        *,
        input_ids_source: torch.Tensor,
        query_start_loc_source: torch.Tensor,
        ngram_context_source: torch.Tensor | None,
    ) -> None:
        self.device = device
        self.dp_rank = get_dp_group().rank_in_group
        self.tp_rank = get_tp_group().rank_in_group
        self._result_bufs: dict[str, torch.Tensor] = {}
        self._result_flags: dict[str, torch.Tensor] = {}
        self._expected_dev: dict[str, torch.Tensor] = {}
        self._timed_out_dev: dict[str, torch.Tensor] = {}
        self._dummy_sem_flags: dict[str, torch.Tensor] = {}
        self._timed_out_host: torch.Tensor | None = None
        # L4 port (TP > 1): the verify mode's shared aux buffers and device counters [checked, mismatched, skipped]
        self._hp_verify = os.environ.get("ARC3_L4PORT_PLE_HP_VERIFY", "0") == "1"
        self._aux_bufs: dict[str, torch.Tensor] = {}
        self._aux_dev: dict[str, torch.Tensor] = {}
        self._v_counts: torch.Tensor | None = None
        self._v_host: torch.Tensor | None = None
        self._v_last: tuple[int, int] | None = None   # (num_tokens, seq) of the last real forward
        self._v_steps = 0
        self._layers = self._setup_layers(vllm_config, model)
        if self._hp_verify:
            for name in self._layers:
                if self.tp_rank == 0:
                    self._aux_bufs[name] = torch.zeros(2, dtype=torch.int64, device="cpu").share_memory_()
                self._aux_dev[name] = torch.zeros(2, dtype=torch.int64, device=self.device)
            self._v_counts = torch.zeros(3, dtype=torch.int64, device=self.device)
            self._v_host = torch.zeros(3, dtype=torch.int64, device="cpu", pin_memory=True)

        # Both runner paths stage into the same shared buffers. TP0 registers
        # them with CUDA so MRV2 can use asynchronous D2H copies.
        scheduler_config = vllm_config.scheduler_config
        self._input_ids_buf = torch.empty(
            scheduler_config.max_num_batched_tokens,
            dtype=torch.int32,
            device="cpu",
        ).share_memory_()
        self._query_start_loc_buf = torch.empty(
            scheduler_config.max_num_seqs + 1,
            dtype=torch.int32,
            device="cpu",
        ).share_memory_()
        self._ngram_context_buf = None
        config = vllm_config.model_config.hf_text_config
        ngram_context_len = int(config.ngram_size) - 1
        if ngram_context_len > 0:
            self._ngram_context_buf = torch.empty(
                scheduler_config.max_num_seqs,
                ngram_context_len,
                dtype=torch.int32,
                device="cpu",
            ).share_memory_()

        # Direct handoff: [num_tokens, num_reqs, reserved, seq] lands in this shared pinned buffer by a 16-byte D2H
        # DMA enqueued after the staging copies on the same stream, so seq visibility implies staged inputs.
        self._request_flag_buf = torch.zeros(4, dtype=torch.int32, device="cpu").share_memory_()
        self._request_seq = 0
        self._flag_dev: torch.Tensor | None = None
        self._flag_src: torch.Tensor | None = None   # pinned 2-slot ring of [4] int32 written on the CPU
        # Runner input allocations are address-stable, so bind them once.
        self._input_ids_source = input_ids_source
        self._query_start_loc_source = query_start_loc_source
        self._ngram_context_source = ngram_context_source
        self._uses_cuda_inputs = self._input_ids_source.is_cuda
        self._validate_input_sources()

        self._pinned_input_buffers: list[torch.Tensor] = []
        self._zmq_ctx: zmq.Context | None = None
        self._registration_socket: zmq.Socket | None = None
        self._d2h_stream: torch.cuda.Stream | None = None
        self._input_ready_event: torch.cuda.Event | None = None
        self._d2h_done_event: torch.cuda.Event | None = None

        try:
            self._zmq_ctx = zmq.Context()
            self._registration_socket = self._zmq_ctx.socket(zmq.PUSH)
            self._registration_socket.connect(ipc_addr)
            self._register_with_offload_worker(vllm_config, ipc_addr)

            if self.tp_rank == 0:
                # ForkingPickler may replace CPU storage while converting its
                # sharing strategy, so send every shared buffer first and
                # register only the final addresses.
                payload = self._register_flag_with_offload_worker()
                # L4 port (TP > 1): the other TP ranks of this group poll the same flags and copy the same rows.
                # bytes: ForkingPickler.dumps returns a memoryview (ZMQ takes it; the group's object broadcast pickles it)
                get_tp_group().broadcast_object(bytes(payload), src=0)
                with torch.accelerator.device_index(self.device.index):
                    self._pin_input_buffers()
                    if self._uses_cuda_inputs:
                        self._d2h_stream = torch.cuda.Stream(device=self.device)
                        self._input_ready_event = torch.cuda.Event()
                        self._d2h_done_event = torch.cuda.Event()
                        self._flag_dev = torch.zeros(4, dtype=torch.int32, device=self.device)
                        self._flag_src = torch.zeros(2, 4, dtype=torch.int32, device="cpu", pin_memory=True)
            else:
                self._adopt_shared_results(get_tp_group().broadcast_object(None, src=0))
        except Exception:
            self.close()
            raise

    def _setup_layers(
        self,
        vllm_config: VllmConfig,
        model: nn.Module,
    ) -> dict[str, PleOffloadLayer]:
        """Attach output buffers and semaphores to GPU PLE placeholders."""
        layers = {
            name: module
            for name, module in model.named_modules()
            if isinstance(module, PleOffloadLayer)
        }
        if not layers:
            raise RuntimeError(
                "VLLM_PLE_CPU_OFFLOAD is enabled, but the model has no PleOffloadLayer"
            )

        config = vllm_config.model_config.hf_text_config
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

    def _pin_input_buffers(self) -> None:
        """Page-lock shared input allocations without replacing their storage."""
        buffers = [self._input_ids_buf, self._query_start_loc_buf, self._request_flag_buf]
        if self._ngram_context_buf is not None:
            buffers.append(self._ngram_context_buf)
        buffers.extend(self._result_bufs.values())
        buffers.extend(self._aux_bufs.values())   # L4 port verify mode (async H2D after each real forward)
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
            self._pinned_input_buffers.append(buffer)
            if not buffer.is_pinned():
                raise RuntimeError("CUDA did not page-lock a PLE input buffer")

    def _unpin_input_buffers(self) -> None:
        """Release CUDA registrations (no request thread exists in the direct handoff)."""
        for buffer in reversed(self._pinned_input_buffers):
            try:
                _cuda_check(
                    cuda_driver.cuMemHostUnregister(buffer.data_ptr()),
                    "cuMemHostUnregister(PLE input buffer)",
                )
            except RuntimeError:
                logger.exception("Failed to unregister a PLE input buffer")
        self._pinned_input_buffers.clear()

    def _register_with_offload_worker(
        self, vllm_config: VllmConfig, ipc_addr: str
    ) -> None:
        """Register CUDA IPC outputs and shared CPU inputs with the worker."""
        # Each GPU worker owns distinct output buffers, while TP0's shared
        # inputs become the request source for its DP rank.
        registration = PleOffloadRegistration(
            worker_id=(
                self.dp_rank * vllm_config.parallel_config.world_size
                + vllm_config.parallel_config.rank
            ),
            tp_rank=self.tp_rank,
            dp_rank=self.dp_rank,
            # L4 port: no CUDA tensors (the host-poll offload never writes GPU memory; CUDA IPC handles would give the
            # offload process a context on every GPU of the PLE stage)
            gpu_output_buffers={},
            sem_flag_tensors={},
            input_ids_buf=self._input_ids_buf,
            query_start_loc_buf=self._query_start_loc_buf,
            ngram_context_buf=self._ngram_context_buf,
        )

        # ForkingPickler transmits tensors through shared-memory and CUDA IPC.
        import torch.multiprocessing as torch_mp

        original_strategy = torch_mp.get_sharing_strategy()
        torch_mp.set_sharing_strategy("file_system")
        try:
            payload = ForkingPickler.dumps(registration)
        finally:
            torch_mp.set_sharing_strategy(original_strategy)
        assert self._registration_socket is not None
        self._registration_socket.send(payload)

        logger.info(
            "PleOffload: registered %d PleOffloadLayer(s) "
            "(dp_rank=%d, tp_rank=%d, ipc_addr=%s): %s",
            len(self._layers),
            self.dp_rank,
            self.tp_rank,
            ipc_addr,
            sorted(self._layers),
        )

    def _register_flag_with_offload_worker(self) -> None:
        """Second registration message: request flag + per-layer shared result buffers and flags for this DP rank."""
        import torch.multiprocessing as torch_mp

        registration = PleHostpollRegistration(dp_rank=self.dp_rank, request_flag_buf=self._request_flag_buf,
                                               result_bufs=dict(self._result_bufs), result_flags=dict(self._result_flags),
                                               aux_bufs=dict(self._aux_bufs) if self._hp_verify else None)
        original_strategy = torch_mp.get_sharing_strategy()
        torch_mp.set_sharing_strategy("file_system")
        try:
            payload = ForkingPickler.dumps(registration)
        finally:
            torch_mp.set_sharing_strategy(original_strategy)
        assert self._registration_socket is not None
        self._registration_socket.send(payload)
        logger.info("PleOffload(hostpoll): registered request flag + %d result buffer(s) for dp_rank=%d", len(self._result_bufs), self.dp_rank)
        return payload

    def _adopt_shared_results(self, payload: bytes) -> None:
        """L4 port (TP rank != 0): poll TP rank 0's shared result buffers / flags (and aux) instead of this rank's own."""
        registration = ForkingPickler.loads(payload)
        if type(registration).__name__ != "PleHostpollRegistration" or registration.dp_rank != self.dp_rank:
            raise RuntimeError("PLE host-poll TP: unexpected payload from TP rank 0")
        if set(registration.result_bufs) != set(self._layers) or set(registration.result_flags) != set(self._layers):
            raise RuntimeError("PLE host-poll TP: TP rank 0's layers differ from this rank's")
        for name, layer in self._layers.items():
            result_buf, result_flag = registration.result_bufs[name], registration.result_flags[name]
            mine = self._result_bufs[name]
            if result_buf.shape != mine.shape or result_buf.dtype != mine.dtype or not result_buf.is_shared():
                raise RuntimeError(f"PLE host-poll TP: result buffer {name} differs from TP rank 0's")
            self._result_bufs[name] = result_buf
            self._result_flags[name] = result_flag
            layer.setup_hostpoll_offload(layer._gpu_output_buffer, result_buf, result_flag, self._expected_dev[name],
                                         self._timed_out_dev[name])
        if self._hp_verify:
            if not registration.aux_bufs or set(registration.aux_bufs) != set(self._layers):
                raise RuntimeError("PLE host-poll TP: verify mode on, but TP rank 0 sent no aux buffers")
            self._aux_bufs = dict(registration.aux_bufs)
        # Page-lock the shared rows (async H2D) and map the shared flags (polled through their device pointers) in this
        # process too; close() unregisters them with the rest.
        mapped = list(self._result_flags.values())
        with torch.accelerator.device_index(self.device.index):
            for buffer in list(self._result_bufs.values()) + list(self._aux_bufs.values()) + mapped:
                flags = cuda_driver.CU_MEMHOSTREGISTER_PORTABLE
                if any(buffer is m for m in mapped):
                    flags = flags | cuda_driver.CU_MEMHOSTREGISTER_DEVICEMAP
                _cuda_check(
                    cuda_driver.cuMemHostRegister(buffer.data_ptr(), buffer.numel() * buffer.element_size(), flags),
                    "cuMemHostRegister(PLE shared result)",
                )
                self._pinned_input_buffers.append(buffer)
                if not buffer.is_pinned():
                    raise RuntimeError("CUDA did not page-lock a shared PLE result buffer")
        logger.info("PleOffload(hostpoll-tp): TP rank %d polls TP rank 0's %d shared result buffer(s) (dp_rank=%d)",
                    self.tp_rank, len(self._result_bufs), self.dp_rank)

    def _copy_cpu_inputs(self, request: PleDirectRequest) -> None:
        """Stage MRV1's existing CPU mirrors in the notifier thread."""
        num_tokens = request.num_tokens
        num_reqs = request.num_reqs
        with torch.cuda.nvtx.range("ple_offload.copy_input_ids"):
            self._input_ids_buf[:num_tokens].copy_(self._input_ids_source[:num_tokens])
        with torch.cuda.nvtx.range("ple_offload.copy_query_start_loc"):
            self._query_start_loc_buf[: num_reqs + 1].copy_(
                self._query_start_loc_source[: num_reqs + 1]
            )
        if self._ngram_context_buf is not None:
            assert self._ngram_context_source is not None
            with torch.cuda.nvtx.range("ple_offload.copy_ngram_context"):
                self._ngram_context_buf[:num_reqs].copy_(
                    self._ngram_context_source[:num_reqs]
                )
        # CPU path: the copies above are synchronous, so plain stores publish the request (seq last).
        flag = self._request_flag_buf
        flag[0] = num_tokens
        flag[1] = num_reqs
        flag[3] = request.seq

    def _validate_input_sources(self) -> None:
        """Validate fixed runner sources against shared input buffers."""
        sources = [
            ("input_ids", self._input_ids_source, self._input_ids_buf),
            (
                "query_start_loc",
                self._query_start_loc_source,
                self._query_start_loc_buf,
            ),
        ]
        if (self._ngram_context_source is None) != (self._ngram_context_buf is None):
            raise ValueError("PLE ngram_context source and buffer must match")
        if self._ngram_context_source is not None:
            assert self._ngram_context_buf is not None
            sources.append(
                (
                    "ngram_context",
                    self._ngram_context_source,
                    self._ngram_context_buf,
                )
            )

        expected_device = self.device if self._uses_cuda_inputs else torch.device("cpu")
        for name, source, buffer in sources:
            if (
                source.device != expected_device
                or source.dtype != buffer.dtype
                or source.ndim != buffer.ndim
                or source.shape[0] < buffer.shape[0]
                or source.shape[1:] != buffer.shape[1:]
            ):
                raise ValueError(f"PLE {name} source is incompatible")

    def _copy_cuda_inputs(self, request: PleDirectRequest) -> None:
        """Stage MRV2 inputs on the background D2H stream."""
        if (
            self._d2h_stream is None
            or self._input_ready_event is None
            or self._d2h_done_event is None
        ):
            raise RuntimeError("PLE D2H resources are not initialized")

        with torch.accelerator.device_index(self.device.index):
            with torch.cuda.stream(self._d2h_stream):
                self._d2h_stream.wait_event(self._input_ready_event)
                with torch.cuda.nvtx.range("ple_offload.copy_input_ids"):
                    self._input_ids_buf[: request.num_tokens].copy_(
                        self._input_ids_source[: request.num_tokens],
                        non_blocking=True,
                    )
                with torch.cuda.nvtx.range("ple_offload.copy_query_start_loc"):
                    self._query_start_loc_buf[: request.num_reqs + 1].copy_(
                        self._query_start_loc_source[: request.num_reqs + 1],
                        non_blocking=True,
                    )
                if self._ngram_context_buf is not None:
                    assert self._ngram_context_source is not None
                    with torch.cuda.nvtx.range("ple_offload.copy_ngram_context"):
                        self._ngram_context_buf[: request.num_reqs].copy_(
                            self._ngram_context_source[: request.num_reqs],
                            non_blocking=True,
                        )
                # Publish [num_tokens, num_reqs, 0, seq]: H2D from a pinned slot into a device scratch, then a
                # 16-byte D2H into the shared pinned flag. Same stream as the staging copies, so the offload
                # process sees seq only after the staged inputs have landed.
                assert self._flag_src is not None and self._flag_dev is not None
                slot = self._flag_src[request.seq & 1]
                slot[0] = request.num_tokens
                slot[1] = request.num_reqs
                slot[2] = 0
                slot[3] = request.seq
                self._flag_dev.copy_(slot, non_blocking=True)
                self._request_flag_buf.copy_(self._flag_dev, non_blocking=True)
                self._d2h_done_event.record(self._d2h_stream)

    def _launch(
        self,
        num_reqs: int,
        num_tokens: int,
    ) -> None:
        """Queue one batch while keeping staging off the model thread."""
        # Inputs are replicated across TP ranks. One request per DP rank drives
        # the CPU result fan-out to every registered TP output buffer.
        if self.tp_rank != 0:
            # L4 port (TP > 1): keep this rank's seq in lockstep with TP rank 0's (every TP rank runs prepare_forward for
            # the same forwards), so its placeholders poll the shared flag for this step's rows; rank 0 alone stages.
            self._request_seq += 1
            for expected in self._expected_dev.values():
                expected.fill_(self._request_seq)
            self._v_last = (num_tokens, self._request_seq)
            return

        if self._uses_cuda_inputs:
            assert self._input_ready_event is not None
            # The background copy stream waits for runner input production
            # without making the model stream wait for D2H completion.
            self._input_ready_event.record(torch.cuda.current_stream(self.device))
        self._request_seq += 1
        request = PleDirectRequest(
            dp_rank=self.dp_rank,
            num_tokens=num_tokens,
            num_reqs=num_reqs,
            seq=self._request_seq,
        )
        # Enqueue the staging copies and the flag DMA here, on the model thread, before the consumer graph is
        # launched: the ordering the Sept 16 acknowledgement enforced now holds by construction, without a
        # thread hand-off or a wait.
        if self._uses_cuda_inputs:
            self._copy_cuda_inputs(request)
        else:
            self._copy_cpu_inputs(request)
        # The placeholder's polling kernel waits for result_flag == expected; publish this step's seq on the model
        # stream so it precedes the consumer graph.
        for expected in self._expected_dev.values():
            expected.fill_(request.seq)
        self._v_last = (num_tokens, request.seq)

    def prepare_forward(
        self,
        num_reqs: int,
        num_tokens: int,
        dummy_run: bool,
    ) -> None:
        """Submit real inputs or satisfy the PLE wait for a dummy forward."""
        if dummy_run:
            self.signal_dummy_outputs(num_tokens)
            return
        self._launch(num_reqs, num_tokens)

    def signal_dummy_outputs(self, num_tokens: int) -> None:
        """Locally satisfy PLE waits for dummy and capture forwards."""
        # Dummy and capture forwards do not send CPU requests, but every PLE
        # placeholder still waits for a completed output semaphore.
        self._v_last = None   # L4 port verify mode: nothing to check after a dummy forward
        for name, layer in self._layers.items():
            layer._gpu_output_buffer[:num_tokens].zero_()
            # expected := the flag's current value, so the in-graph poll is satisfied without a CPU result.
            self._expected_dev[name].copy_(self._result_flags[name], non_blocking=True)

    def release_outputs(self) -> None:
        """Mark GPU output buffers reusable after the model consumes them."""
        # Reset only after the consumer forward so the CPU worker cannot
        # overwrite an output that a GPU PLE placeholder may still read.
        # No semaphore to reset. Surface a poll timeout of the previous step (its D2H below has landed by now) and
        # queue this step's timed_out mirror; the model thread never synchronizes.
        assert self._timed_out_host is not None
        if int(self._timed_out_host.max()) != 0:
            raise RuntimeError("PLE host-poll timed out waiting for the offload process; see ple-hostpoll receipts")
        for index, timed_out in enumerate(self._timed_out_dev.values()):
            self._timed_out_host[index : index + 1].copy_(timed_out, non_blocking=True)
        if self._hp_verify:
            self._verify_last()

    def _verify_last(self) -> None:
        """L4 port verify mode: compare each layer's GPU rows of the last real forward with the offload's checksum."""
        assert self._v_counts is not None and self._v_host is not None
        if self._v_last is not None:
            num_tokens, seq = self._v_last
            for name, layer in self._layers.items():
                aux = self._aux_dev[name]
                aux.copy_(self._aux_bufs[name], non_blocking=True)
                checksum = layer._gpu_output_buffer[:num_tokens].view(torch.int16).sum(dtype=torch.int64)
                same_seq = aux[0] == seq
                self._v_counts[0] += same_seq.to(torch.int64)
                self._v_counts[1] += (same_seq & (aux[1] != checksum)).to(torch.int64)
                self._v_counts[2] += (~same_seq).to(torch.int64)
            self._v_last = None
        self._v_steps += 1
        if self._v_steps % 256 == 0:
            # The mirror copied 256 steps ago has landed (the model thread never runs that far ahead of the GPU).
            if int(self._v_host[1]) != 0:
                self._write_verify_receipt()
                raise RuntimeError("PLE host-poll TP verify: %d mismatched forward(s); see ple-hostpoll receipts" % int(self._v_host[1]))
            if self._v_steps % 4096 == 0:
                self._write_verify_receipt()
            self._v_host.copy_(self._v_counts, non_blocking=True)

    def _write_verify_receipt(self) -> None:
        """L4 port verify mode: counts so far into ARC3_PLE_HOSTPOLL_RECEIPTS/verify-PID.json."""
        root = os.environ.get("ARC3_PLE_HOSTPOLL_RECEIPTS")
        if not root or self._v_host is None:
            return
        try:
            import json
            import time
            os.makedirs(root, exist_ok=True)
            record = dict(version="arc3-ple-hostpoll-tp", role="gpu-verify", pid=os.getpid(), tp_rank=self.tp_rank,
                          dp_rank=self.dp_rank, forwards=self._v_steps, checked=int(self._v_host[0]),
                          mismatched=int(self._v_host[1]), skipped=int(self._v_host[2]), epoch=time.time())
            tmp = os.path.join(root, "verify-%d.json.tmp" % os.getpid())
            with open(tmp, "w") as f:
                json.dump(record, f)
            os.replace(tmp, os.path.join(root, "verify-%d.json" % os.getpid()))
        except Exception:
            logger.exception("PLE host-poll TP: verify receipt failed")

    def close(self) -> None:
        """Stop request transport and release host registrations."""
        if self._hp_verify and self._v_host is not None:
            self._write_verify_receipt()
        if self._d2h_stream is not None:
            # Let in-flight staging DMAs finish before unregistering their host targets.
            self._d2h_stream.synchronize()
        if self._pinned_input_buffers:
            with torch.accelerator.device_index(self.device.index):
                self._unpin_input_buffers()
        self._d2h_done_event = None
        self._input_ready_event = None
        self._d2h_stream = None
        if self._registration_socket is not None:
            self._registration_socket.close(linger=0)
            self._registration_socket = None
        if self._zmq_ctx is not None:
            self._zmq_ctx.term()
            self._zmq_ctx = None
