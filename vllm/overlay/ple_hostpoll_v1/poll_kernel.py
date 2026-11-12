"""One-block polling kernel: waits until a host-mapped int32 flag equals the expected device-side value.

Replaces the in-graph cuStreamWaitValue32 memop with an ordinary kernel node: `flag` is a page-locked host tensor
(cuMemHostRegister'd shared memory, read through its UVA pointer with volatile loads), `expected` a device int32[1]
that the model thread updates per step (seq). The kernel spins with __nanosleep and ends with a system-scope fence
so the rows the CPU wrote before publishing the flag are visible to the following H2D copy / consumers."""
import os
from pathlib import Path

import torch
from torch.utils.cpp_extension import load_inline

CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAStream.h>

__global__ void arc3_poll_host_flag_kernel(const volatile int* host_flag, const int* expected, unsigned long long max_spins,
                                           int* timed_out) {
    if (threadIdx.x != 0) return;
    const int want = *expected;
    unsigned long long spins = 0;
    while (*host_flag != want) {
        __nanosleep(400);
        if (++spins > max_spins) { *timed_out = 1; return; }
    }
    __threadfence_system();
}

void arc3_poll_host_flag(torch::Tensor host_flag, torch::Tensor expected, torch::Tensor timed_out, int64_t max_spins) {
    TORCH_CHECK(host_flag.dtype() == torch::kInt32 && host_flag.is_pinned(), "host_flag must be a pinned int32 tensor");
    TORCH_CHECK(expected.is_cuda() && expected.dtype() == torch::kInt32, "expected must be a CUDA int32 tensor");
    TORCH_CHECK(timed_out.is_cuda() && timed_out.dtype() == torch::kInt32, "timed_out must be a CUDA int32 tensor");
    int* dev_ptr = nullptr;
    cudaError_t err = cudaHostGetDevicePointer((void**)&dev_ptr, host_flag.data_ptr(), 0);
    TORCH_CHECK(err == cudaSuccess, "cudaHostGetDevicePointer failed: ", cudaGetErrorString(err));
    cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();
    arc3_poll_host_flag_kernel<<<1, 32, 0, stream>>>(dev_ptr, expected.data_ptr<int>(), (unsigned long long)max_spins,
                                                     timed_out.data_ptr<int>());
}
"""
CPP_SRC = "void arc3_poll_host_flag(torch::Tensor host_flag, torch::Tensor expected, torch::Tensor timed_out, int64_t max_spins);"


def _cudart_ldflags():
    """torch's cpp_extension adds -L$CUDA_HOME/lib64 -lcudart, but this extracted runtime keeps libcudart under
    targets/x86_64-linux/lib (and torch's own copy under nvidia/cu13/lib); add those search paths."""
    import glob
    flags = []
    cuda_home = os.environ.get('CUDA_HOME', '')
    for d in (os.path.join(cuda_home, 'targets', 'x86_64-linux', 'lib'),
              str(Path(torch.__file__).resolve().parents[1] / 'nvidia' / 'cu13' / 'lib')):
        if glob.glob(os.path.join(d, 'libcudart.so*')):
            flags.append('-L' + d)
    return flags


def load(build_dir=None, verbose=False):
    build_dir = build_dir or os.environ.get('ARC3_PLE_HOSTPOLL_BUILD', str(Path(__file__).with_name('build')))
    Path(build_dir).mkdir(parents=True, exist_ok=True)
    return load_inline(name='arc3_ple_hostpoll_v1', cpp_sources=[CPP_SRC], cuda_sources=[CUDA_SRC],
                       functions=['arc3_poll_host_flag'], extra_cuda_cflags=['-O3'], extra_ldflags=_cudart_ldflags(),
                       build_directory=build_dir, verbose=verbose, with_cuda=True, extra_include_paths=[])
