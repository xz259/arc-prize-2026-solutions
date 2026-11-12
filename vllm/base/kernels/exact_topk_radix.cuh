// SPDX-License-Identifier: Apache-2.0
// Derived verbatim from vLLM PR 55122, commit 7cfd04a395fab32d6ee3364327df297a393f35d2.
// Only the independent single-CTA selector and its helpers are retained.
// https://github.com/vllm-project/vllm/pull/55122
#pragma once
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <cstdint>
namespace arc3_radix {
__device__ __forceinline__ auto convert_to_uint32_v2(float x) -> uint32_t {
  uint32_t bits = __float_as_uint(x);
  // -0.0 and +0.0 are numerically equal, so the documented rule (value
  // descending, ties by index ascending) must treat them as a tie. Their bit
  // patterns differ, which would otherwise order +0.0 above -0.0. Canonicalise
  // to +0 before the order-preserving transform.
  if ((bits & 0x7FFFFFFFu) == 0u) bits = 0u;
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

__device__ __forceinline__ void load_float4(const float* ptr, float& v0,
                                            float& v1, float& v2, float& v3) {
  uint32_t r0, r1, r2, r3;
  asm volatile("ld.global.cg.v4.u32 {%0,%1,%2,%3}, [%4];\n"
               : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
               : "l"(ptr));
  v0 = __uint_as_float(r0);
  v1 = __uint_as_float(r1);
  v2 = __uint_as_float(r2);
  v3 = __uint_as_float(r3);
}

template <int TopK, int N_THREADS>
__host__ __device__ constexpr size_t det_select_row_fixed_bytes() {
  using ScanT = cub::BlockScan<uint32_t, N_THREADS>;
  return 2048 + ((sizeof(typename ScanT::TempStorage) + 127) & ~size_t(127));
}
// Bytes needed to keep a row of n keys cached (host side sizing helper).
template <int TopK, int N_THREADS>
__host__ __device__ constexpr size_t det_select_row_bytes(size_t n) {
  return det_select_row_fixed_bytes<TopK, N_THREADS>() + n * sizeof(uint32_t);
}
template <int TopK, int N_THREADS>
__device__ void det_select_row(const float* __restrict__ row, int n,
                               int32_t* __restrict__ out, void* smem,
                               size_t smem_bytes) {
  static_assert(N_THREADS <= 0xFFFF,
                "the emission packs two per-tile counts into one uint32");
  static_assert(N_THREADS >= 32, "the bin scan runs in one full warp");
  using ScanT = cub::BlockScan<uint32_t, N_THREADS>;
  uint32_t* hist = reinterpret_cast<uint32_t*>(smem);  // [256]
  uint32_t* hist2 = hist + 256;                        // [256]
  auto* scan_tmp = reinterpret_cast<typename ScanT::TempStorage*>(
      reinterpret_cast<char*>(smem) + 2048);
  const size_t fixed = det_select_row_fixed_bytes<TopK, N_THREADS>();
  uint32_t* keys =
      reinterpret_cast<uint32_t*>(reinterpret_cast<char*>(smem) + fixed);
  const bool cached =
      (fixed + static_cast<size_t>(n) * sizeof(uint32_t) <= smem_bytes);
  const int tx = threadIdx.x;
  uint32_t prefix = 0;
  uint32_t remaining = TopK;
  for (int pass = 0; pass < 4; pass++) {
    const int shift = 24 - pass * 8;
    const uint32_t hi_mask = (pass == 0) ? 0u : (0xFFFFFFFFu << (shift + 8));
    for (int i = tx; i < 256; i += N_THREADS) hist[i] = 0;
    __syncthreads();
    if (pass == 0) {
      // first pass: read the row once; keep the ordered keys if they fit
      const int n4 = n & ~3;
      const bool aligned = ((reinterpret_cast<uintptr_t>(row) & 15) == 0);
      for (int i = tx * 4; i < n4; i += N_THREADS * 4) {
        float v0, v1, v2, v3;
        if (aligned)
          load_float4(row + i, v0, v1, v2, v3);
        else {
          v0 = row[i];
          v1 = row[i + 1];
          v2 = row[i + 2];
          v3 = row[i + 3];
        }
        const uint32_t k0 = convert_to_uint32_v2(v0),
                       k1 = convert_to_uint32_v2(v1);
        const uint32_t k2 = convert_to_uint32_v2(v2),
                       k3 = convert_to_uint32_v2(v3);
        if (cached) {
          keys[i] = k0;
          keys[i + 1] = k1;
          keys[i + 2] = k2;
          keys[i + 3] = k3;
        }
        atomicAdd(&hist[k0 >> 24], 1);
        atomicAdd(&hist[k1 >> 24], 1);
        atomicAdd(&hist[k2 >> 24], 1);
        atomicAdd(&hist[k3 >> 24], 1);
      }
      for (int i = n4 + tx; i < n; i += N_THREADS) {
        const uint32_t k = convert_to_uint32_v2(row[i]);
        if (cached) keys[i] = k;
        atomicAdd(&hist[k >> 24], 1);
      }
    } else {
      for (int i = tx; i < n; i += N_THREADS) {
        const uint32_t key = cached ? keys[i] : convert_to_uint32_v2(row[i]);
        if ((key & hi_mask) == prefix)
          atomicAdd(&hist[(key >> shift) & 0xFF], 1);
      }
    }
    __syncthreads();
    // Suffix sum over the 256 bins and the threshold search, in ONE warp:
    // lane l owns bins [8l, 8l+8), sums them serially, then a Hillis-Steele
    // suffix scan over the 32 lane totals gives what lies above the lane.
    // suf[b] = #elements in this prefix group with byte >= b, and the first
    // bin of the next lane has suf = `above`, so the search is lane-local too.
    // The previous 8-step double-buffered version cost 8 __syncthreads() here,
    // 32 over the four passes; this costs none.
    if (tx < 32) {
      uint32_t local[8];
      uint32_t total = 0;
#pragma unroll
      for (int j = 7; j >= 0; j--) {
        total += hist[tx * 8 + j];
        local[j] = total;
      }
      uint32_t s_suf = total;  // inclusive suffix over lane totals
#pragma unroll
      for (int off = 1; off < 32; off <<= 1) {
        const uint32_t v = __shfl_down_sync(0xFFFFFFFFu, s_suf, off);
        if (tx + off < 32) s_suf += v;
      }
      const uint32_t above = s_suf - total;  // strictly higher lanes
#pragma unroll
      for (int j = 0; j < 8; j++) {
        const uint32_t suf_b = local[j] + above;
        const uint32_t suf_b1 = (j < 7) ? (local[j + 1] + above) : above;
        if (suf_b >= remaining && suf_b1 < remaining) {
          hist2[0] = static_cast<uint32_t>(tx * 8 + j);
          hist2[1] = suf_b1;
          hist2[2] = suf_b - suf_b1;  // population of the threshold bin
        }
      }
    }
    __syncthreads();
    const uint32_t thr = hist2[0];
    const uint32_t bin_pop = hist2[2];
    remaining -= hist2[1];
    prefix |= thr << shift;
    // Early exit: the threshold bin holds exactly what is still needed, so all
    // of it is selected and the lower key bytes cannot change the answer. The
    // selection becomes `key >= prefix`, i.e. `key > prefix - 1`, with no ties
    // to rank. (`remaining == 0` can never happen: the bin search guarantees
    // suf_b1 < remaining, so the subtraction above always leaves at least 1.)
    if (bin_pop == remaining && prefix != 0u) {
      prefix -= 1u;
      remaining = 0u;
      __syncthreads();
      break;
    }
    __syncthreads();
  }
  const uint32_t pivot = prefix;
  const uint32_t fin = remaining;
  // Blocked emission, four elements per thread: thread `tx` owns the four
  // consecutive indices [base + 4*tx, base + 4*tx + 4). Blocked (not striped)
  // ownership is what keeps the position formula below valid unchanged -- it
  // needs the scan to run in index order, and blocked layout preserves index
  // order both within a thread and across threads. This runs one BlockScan and
  // one barrier per 4*N_THREADS elements instead of per N_THREADS: at
  // n = 16384 that is 4 of each rather than 16. The selected set and its order
  // are untouched, since the position is a pure function of index, pivot and
  // `fin`.
  constexpr int kItems = 4;
  constexpr int kTile = kItems * N_THREADS;
  static_assert(kTile <= 0xFFFF,
                "the emission packs two per-tile counts into one uint32");
  // `keys` sits at `smem + fixed`, and `fixed` is 2048 plus a multiple of 128,
  // so the 128-bit blocked load below is aligned and conflict-free.
  const bool row_aligned = ((reinterpret_cast<uintptr_t>(row) & 15) == 0);
  uint32_t run_gt = 0, run_eq = 0;
  for (int base = 0; base < n; base += kTile) {
    const int mine = base + tx * kItems;
    uint32_t key[kItems];
    bool valid[kItems];
    if (mine + kItems <= n) {
      if (cached) {
        const uint4 v = *reinterpret_cast<const uint4*>(keys + mine);
        key[0] = v.x;
        key[1] = v.y;
        key[2] = v.z;
        key[3] = v.w;
      } else if (row_aligned) {
        float v0, v1, v2, v3;
        load_float4(row + mine, v0, v1, v2, v3);
        key[0] = convert_to_uint32_v2(v0);
        key[1] = convert_to_uint32_v2(v1);
        key[2] = convert_to_uint32_v2(v2);
        key[3] = convert_to_uint32_v2(v3);
      } else {
#pragma unroll
        for (int j = 0; j < kItems; ++j)
          key[j] = convert_to_uint32_v2(row[mine + j]);
      }
#pragma unroll
      for (int j = 0; j < kItems; ++j) valid[j] = true;
    } else {
#pragma unroll
      for (int j = 0; j < kItems; ++j) {
        const int i = mine + j;
        valid[j] = (i < n);
        key[j] =
            valid[j] ? (cached ? keys[i] : convert_to_uint32_v2(row[i])) : 0u;
      }
    }
    uint32_t fgt[kItems], feq[kItems];
    uint32_t agg = 0;
#pragma unroll
    for (int j = 0; j < kItems; ++j) {
      fgt[j] = (valid[j] && key[j] > pivot) ? 1u : 0u;
      feq[j] = (valid[j] && key[j] == pivot) ? 1u : 0u;
      // Both flags in one scan: they are mutually exclusive and a tile holds at
      // most kTile elements, so each count fits in 16 bits.
      agg += fgt[j] | (feq[j] << 16);
    }
    uint32_t packed_rank, packed_total;
    ScanT(*scan_tmp).ExclusiveSum(agg, packed_rank, packed_total);
    // Final ascending position directly. The number of selected elements at
    // lower indices is (# greater before) + min(# equal before, fin), since
    // exactly the first `fin` equal elements by index are kept. True for a `>`
    // element (the min saturates) and for a kept `==` element (it does not),
    // so one expression serves both and no reordering pass is needed. The
    // block-wide exclusive prefix is continued serially over this thread's own
    // four, in index order.
    uint32_t g = run_gt + (packed_rank & 0xFFFFu);
    uint32_t e = run_eq + (packed_rank >> 16);
#pragma unroll
    for (int j = 0; j < kItems; ++j) {
      if (fgt[j] || (feq[j] && e < fin)) out[g + (e < fin ? e : fin)] = mine + j;
      g += fgt[j];
      e += feq[j];
    }
    run_gt += packed_total & 0xFFFFu;
    run_eq += packed_total >> 16;
    __syncthreads();
  }
}


} // namespace arc3_radix
