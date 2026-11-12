// SPDX-License-Identifier: Apache-2.0
// Small isolated launcher for the attributed exact single-CTA radix selector.
#include "exact_topk_radix.cuh"
#include <algorithm>

namespace {
constexpr int K = 512;
constexpr int MAX_DEVICES = 16;
int shared_limit[MAX_DEVICES] = {};

template <int Threads>
__global__ void select_rows(const float* logits, const int* lengths, int* output,
                            int rows, int columns, int64_t stride, int smem_bytes) {
  const int row = blockIdx.x;
  if (row >= rows) return;
  const int n = max(0, min(lengths[row], columns));
  int* out = output + static_cast<int64_t>(row) * K;
  if (n <= K) {
    for (int i = threadIdx.x; i < K; i += Threads) out[i] = i < n ? i : -1;
    return;
  }
  extern __shared__ __align__(16) unsigned char shared[];
  arc3_radix::det_select_row<K, Threads>(
      logits + static_cast<int64_t>(row) * stride, n, out, shared, smem_bytes);
}

template <int Threads>
cudaError_t initialize(int limit) {
  return cudaFuncSetAttribute(select_rows<Threads>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize, limit);
}

template <int Threads>
cudaError_t launch(const float* logits, const int* lengths, int* output,
                   int rows, int columns, int64_t stride, int limit,
                   cudaStream_t stream, bool cache_row) {
  size_t wanted = arc3_radix::det_select_row_fixed_bytes<K, Threads>();
  if (cache_row) wanted += static_cast<size_t>(columns) * sizeof(uint32_t);
  const int smem = static_cast<int>(std::min(wanted, static_cast<size_t>(limit)));
  select_rows<Threads><<<rows, Threads, smem, stream>>>(
      logits, lengths, output, rows, columns, stride, smem);
  return cudaGetLastError();
}
} // namespace

extern "C" int arc3_radix_initialize(int device) {
  if (device < 0 || device >= MAX_DEVICES) return cudaErrorInvalidDevice;
  cudaDeviceProp prop{};
  cudaError_t status = cudaGetDeviceProperties(&prop, device);
  if (status != cudaSuccess) return status;
  // This kernel has no static shared storage. Keep a small margin below opt-in.
  const int limit = std::min(static_cast<int>(prop.sharedMemPerBlockOptin), 96*1024);
  if ((status = initialize<128>(limit)) != cudaSuccess) return status;
  if ((status = initialize<256>(limit)) != cudaSuccess) return status;
  if ((status = initialize<512>(limit)) != cudaSuccess) return status;
  if ((status = initialize<1024>(limit)) != cudaSuccess) return status;
  shared_limit[device] = limit;
  return cudaSuccess;
}

extern "C" int arc3_radix_topk(const float* logits, const int* lengths, int* output,
                              int rows, int columns, int64_t stride, int threads,
                              int device, void* stream_ptr, int cache_row) {
  if (device < 0 || device >= MAX_DEVICES || !shared_limit[device] || rows < 0 ||
      columns < 0 || stride < columns) return cudaErrorInvalidValue;
  if (rows == 0) return cudaSuccess;
  const auto stream = static_cast<cudaStream_t>(stream_ptr);
  const int limit = shared_limit[device];
#define RUN(T) launch<T>(logits, lengths, output, rows, columns, stride, limit, stream, cache_row != 0)
  switch (threads) {
    case 128: return RUN(128);
    case 256: return RUN(256);
    case 512: return RUN(512);
    case 1024: return RUN(1024);
    default: return cudaErrorInvalidValue;
  }
#undef RUN
}

extern "C" const char* arc3_radix_error_string(int code) {
  return cudaGetErrorString(static_cast<cudaError_t>(code));
}
