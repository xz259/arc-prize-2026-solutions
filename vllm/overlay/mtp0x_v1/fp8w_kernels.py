"""FP8 weight-only (W8A16) GEMM for the production dense projections, as a candidate replacement for Marlin
(verify-path kernel work, 2026-09-29).

y[M, N] = x[M, K] (BF16) @ (Wq[N, K] (E4M3) * s[N])^T, FP32 accumulation, BF16 output. Weights stay in the plain
row-major E4M3 layout the dense adapter quantizes at load (before Marlin repacking); the per-output-channel scale is
applied to the accumulator. Split-K uses stage 14's deterministic scheme: every split stores its FP32 partial tile,
and the last-arriving program of a tile (per-tile counter) sums the partials in fixed split order, then resets the
counter. Programs are ordered so the M tiles of one weight tile run together (the weight tile is read from HBM once
and from L2 by the others).
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _fp8w_kernel(x_ptr, w_ptr, s_ptr, y_ptr, p_ptr, cnt_ptr, M, NUM_M, stride_xm, stride_ym,
                 N: tl.constexpr, K: tl.constexpr, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr,
                 BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr, CONV: tl.constexpr = 0):
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
        if CONV == 1:
            # E4M3 -> BF16 by moving the bit fields: sign to bit 15, exponent|mantissa to bits 4..10. The result is
            # the exact value times 2^-120 (normals and subnormals alike); 2^120 is folded into the scale below.
            wi = tl.load(w_ptr + offs_n[:, None] * K + kk[None, :], mask=nm, other=0).to(tl.uint16)
            wb = (((wi & 0x80) << 8) | ((wi & 0x7F) << 4)).to(tl.bfloat16, bitcast=True)
        else:
            wb = tl.load(w_ptr + offs_n[:, None] * K + kk[None, :], mask=nm, other=0.0).to(tl.bfloat16)
        acc += tl.dot(a, tl.trans(wb))
    omask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    scale = tl.load(s_ptr + offs_n, mask=offs_n < N, other=0.0)
    if CONV == 1:
        scale = scale * 1.329227995784916e36      # 2^120
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
            tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :], (tot * scale[None, :]).to(tl.bfloat16),
                     mask=omask)
            tl.atomic_xchg(cnt_ptr + tile, 0)
    else:
        tl.store(y_ptr + offs_m[:, None] * stride_ym + offs_n[None, :], (acc * scale[None, :]).to(tl.bfloat16),
                 mask=omask)


class Workspace:
    """One shared FP32 partial buffer and one counter buffer (grown on demand). Launches on one stream run in order,
    so consecutive GEMMs can share them; each split-K tile's last program resets its counter to zero."""

    def __init__(self, device):
        self.device = device
        self.p = torch.empty(1, dtype=torch.float32, device=device)
        self.cnt = torch.zeros(1, dtype=torch.int32, device=device)

    def get(self, key, partial_elems, tiles):
        if self.p.numel() < partial_elems:
            self.p = torch.empty(partial_elems, dtype=torch.float32, device=self.device)
        if self.cnt.numel() < tiles:
            self.cnt = torch.zeros(tiles, dtype=torch.int32, device=self.device)
        return self.p, self.cnt


def fp8w_linear(x, wq, scale, cfg, ws, out=None):
    """x [M, K] bf16 contiguous rows; wq [N, K] float8_e4m3fn contiguous; scale [N] fp32; cfg (BM, BN, BK, SPLIT_K,
    warps, stages[, CONV]); CONV 1 = bit-field conversion (weights passed as uint8)."""
    M, K = x.shape
    N = wq.shape[0]
    bm, bn, bk, sk, warps, stages = cfg[:6]
    conv = cfg[6] if len(cfg) > 6 else 0
    if conv == 1:
        wq = wq.view(torch.uint8)
    assert K % (sk * bk) == 0, (K, sk, bk)
    num_m, num_n = triton.cdiv(M, bm), triton.cdiv(N, bn)
    y = out if out is not None else torch.empty((M, N), dtype=torch.bfloat16, device=x.device)
    p, cnt = ws.get((N, K, cfg, num_m), num_n * num_m * sk * bm * bn if sk > 1 else 1, num_n * num_m)
    _fp8w_kernel[(num_n * num_m, sk)](x, wq, scale, y, p, cnt, M, num_m, x.stride(0), y.stride(0), N=N, K=K,
                                      BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=sk, CONV=conv,
                                      num_warps=warps, num_stages=stages)
    return y
