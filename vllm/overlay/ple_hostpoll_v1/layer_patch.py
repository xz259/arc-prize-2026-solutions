"""GPU-worker side of the host-poll handoff: the PLE placeholder's forward becomes one custom op that (1) runs a
one-block polling kernel until the layer's host result flag equals the per-step expected seq and (2) copies the rows
from the shared pinned result buffer into the GPU output buffer on the current (model) stream. Both are ordinary
kernel / memcpy nodes under CUDA-graph capture. Applied to the executed `vllm.model_executor.layers.ple_offload_layer`."""
import os

import torch

MAX_SPINS = int(os.environ.get('ARC3_PLE_HOSTPOLL_MAX_SPINS', str(75_000_000)))   # ~30 s at 400 ns per spin
_KERNEL = None


def _kernel():
    global _KERNEL
    if _KERNEL is None:
        from . import poll_kernel
        _KERNEL = poll_kernel.load()
    return _KERNEL


def install(module):
    """Register the custom op and patch PleOffloadLayer (forward / setup / release)."""
    L = module.PleOffloadLayer
    assert not getattr(L, '_arc3_ple_hostpoll', False)
    direct_register_custom_op = module.direct_register_custom_op

    def _wait_copy_impl(result_flag_host: torch.Tensor, expected_dev: torch.Tensor, timed_out_dev: torch.Tensor,
                        result_host: torch.Tensor, gpu_output_buffer: torch.Tensor, num_tokens: int) -> None:
        _kernel().arc3_poll_host_flag(result_flag_host, expected_dev, timed_out_dev, MAX_SPINS)
        gpu_output_buffer[:num_tokens].copy_(result_host[:num_tokens], non_blocking=True)

    def _wait_copy_fake(result_flag_host: torch.Tensor, expected_dev: torch.Tensor, timed_out_dev: torch.Tensor,
                        result_host: torch.Tensor, gpu_output_buffer: torch.Tensor, num_tokens: int) -> None:
        return None

    direct_register_custom_op(op_name='arc3_ple_hostpoll_wait_copy', op_func=_wait_copy_impl,
                              mutates_args=['gpu_output_buffer', 'timed_out_dev'], fake_impl=_wait_copy_fake)
    op = torch.ops.vllm.arc3_ple_hostpoll_wait_copy

    def setup_hostpoll_offload(self, gpu_output_buffer, result_host, result_flag, expected_dev, timed_out_dev):
        self._is_cpu_offloaded = True
        self._gpu_output_buffer = gpu_output_buffer
        self._hp_result_host = result_host
        self._hp_result_flag = result_flag
        self._hp_expected_dev = expected_dev
        self._hp_timed_out_dev = timed_out_dev

    def forward(self, hidden_states, input_ids, *args, **kwargs):
        if self._is_cpu_offloaded:
            num_tokens = input_ids.shape[0]
            op(self._hp_result_flag, self._hp_expected_dev, self._hp_timed_out_dev, self._hp_result_host,
               self._gpu_output_buffer, num_tokens)
            return self._gpu_output_buffer[:num_tokens]
        return self.forward_impl(hidden_states, input_ids, *args, **kwargs)

    def release_offloaded_output(self, stream=None):
        return None   # no semaphore in the host-poll handoff

    L.setup_hostpoll_offload = setup_hostpoll_offload
    L.forward = forward
    L.release_offloaded_output = release_offloaded_output
    L._arc3_ple_hostpoll = True
    return L
