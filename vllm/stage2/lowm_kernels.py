"""BF16-weight dense GEMM for small and medium M (milestone 2, 2026-10-02; plan Revision 7 B3: dfranzen's low-M BF16 GEMM
patch, from Gabriel Olympie's SM120 patch 0004, in production's deterministic fp8w design).

y[M, N] = x[M, K] @ W[N, K]^T, BF16 in and out, FP32 accumulation. The kernel is production's _fp8w_kernel
(runs/dense_fp8_v1/mtp0x_v1/fp8w_kernels.py) without the FP8 weight conversion and scale: deterministic split-K (every
split stores its FP32 partial tile; the last-arriving program of a tile, by a per-tile counter, sums the partials in
fixed split order and resets the counter), and programs ordered so the M tiles of one weight tile run together. Unlike
the patch's fp32 atomics, the result does not depend on arrival order. cuBLAS serves these shapes under CUDA graphs with
weak WMMA split-K kernels on SM120 (the patch measured 20-70% of DRAM bandwidth at m <= 8); the table holds the
configuration per (N, K) and M bucket where this kernel measured faster (bench_lowm.py)."""
import torch
import triton
import triton.language as tl


@triton.jit
def _lowm_kernel(x_ptr, w_ptr, y_ptr, p_ptr, cnt_ptr, M, NUM_M, stride_xm, stride_ym,
                 N: tl.constexpr, K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                 BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr):
    pid = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_n = pid // NUM_M
    pid_m = pid % NUM_M
    K_PER: tl.constexpr = K // SPLIT_K
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mm = offs_m[:, None] < M
    nm = offs_n[:, None] < N
    k0 = pid_k * K_PER
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K_PER, BLOCK_K):
        kk = k0 + k + offs_k
        a = tl.load(x_ptr + offs_m[:, None] * stride_xm + kk[None, :], mask=mm, other=0.0)
        wb = tl.load(w_ptr + offs_n[:, None] * K + kk[None, :], mask=nm, other=0.0)
        acc += tl.dot(a, tl.trans(wb))
    omask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if SPLIT_K > 1:
        tile = pid_n * NUM_M + pid_m
        pidx = tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
        pbase = p_ptr + tile * (SPLIT_K * BLOCK_M * BLOCK_N)
        tl.store(pbase + pid_k * (BLOCK_M * BLOCK_N) + pidx, acc)
        tl.debug_barrier()
        prev = tl.atomic_add(cnt_ptr + tile, 1, sem='acq_rel')
        if prev == SPLIT_K - 1:
            tot = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for s in range(SPLIT_K):
                tot += tl.load(pbase + s * (BLOCK_M * BLOCK_N) + pidx, volatile=True)
            tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :], tot.to(tl.bfloat16), mask=omask)
            tl.atomic_xchg(cnt_ptr + tile, 0)
    else:
        tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :], acc.to(tl.bfloat16), mask=omask)


class Workspace:
    """One shared FP32 partial buffer and one counter buffer, grown on demand (as fp8w's). Launches on one stream run in
    order, so consecutive GEMMs can share them; each split-K tile's last program resets its counter."""

    def __init__(self, device):
        self.device = device
        self.p = torch.empty(1, dtype=torch.float32, device=device)
        self.cnt = torch.zeros(1, dtype=torch.int32, device=device)

    def get(self, partial_elems, tiles):
        if self.p.numel() < partial_elems:
            self.p = torch.empty(partial_elems, dtype=torch.float32, device=self.device)
        if self.cnt.numel() < tiles:
            self.cnt = torch.zeros(tiles, dtype=torch.int32, device=self.device)
        return self.p, self.cnt


def lowm_linear(x, w, cfg, ws, out=None):
    """x [M, K] BF16 with unit last stride; w [N, K] BF16 contiguous; cfg (BM, BN, BK, SPLIT_K, warps, stages)."""
    M, K = x.shape
    N = w.shape[0]
    bm, bn, bk, sk, warps, stages = cfg
    if K % (sk * bk):
        raise ValueError('lowm: K %d not a multiple of split %d x block %d' % (K, sk, bk))
    num_m, num_n = triton.cdiv(M, bm), triton.cdiv(N, bn)
    y = out if out is not None else torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    p, cnt = ws.get(num_n * num_m * sk * bm * bn if sk > 1 else 1, num_n * num_m)
    _lowm_kernel[(num_n * num_m, sk)](x, w, y, p, cnt, M, num_m, x.stride(0), y.stride(0), N=N, K=K,
                                      BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=sk, num_warps=warps, num_stages=stages)
    return y
