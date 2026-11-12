"""Triton small-M GEMMs for the served decode step, registered as opaque custom ops (torch.ops.vllm.arc3_*)."""
import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

M_MAX = 64  # decode-only kernels; larger M (prefill, NLL probe) keeps the vendor path bit-for-bit


@triton.jit
def _skinny_kernel(x_ptr, w_ptr, p_ptr, cnt_ptr, o_ptr, M, N, K, stride_xm, stride_wn, stride_om,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr,
                   SILU_HC: tl.constexpr, launch_pdl: tl.constexpr):
    """o = x @ w.T (bf16 in, fp32 accumulate, bf16 out). SPLIT_K > 1: partials + deterministic fixed-order fixup by
    the last arriving program of each N tile. SILU_HC > 0: A = bf16(silu(x / SILU_HC)) (the vendor hc_silu boundary)."""
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    k_per = tl.cdiv(K, SPLIT_K)
    k0 = pid_k * k_per
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(k0, k0 + k_per, BLOCK_K):
        kk = k + offs_k
        km = kk < tl.minimum(k0 + k_per, K)
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + kk[None, :], mask=(offs_m[:, None] < M) & km[None, :], other=0.0)
        if SILU_HC > 0:
            xf = x.to(tl.float32) / SILU_HC
            x = (xf * tl.sigmoid(xf)).to(tl.bfloat16)
        w = tl.load(w_ptr + offs_n[:, None] * stride_wn + kk[None, :], mask=(offs_n[:, None] < N) & km[None, :], other=0.0)
        acc += tl.dot(x, tl.trans(w))
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if SPLIT_K == 1:
        if launch_pdl:
            tl.extra.cuda.gdc_launch_dependents()
        tl.store(o_ptr + offs_m[:, None] * stride_om + offs_n[None, :], acc.to(tl.bfloat16), mask=mask)
    else:
        pidx = tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
        pbase = p_ptr + pid_n * (SPLIT_K * BLOCK_M * BLOCK_N)
        tl.store(pbase + pid_k * (BLOCK_M * BLOCK_N) + pidx, acc)
        tl.debug_barrier()
        prev = tl.atomic_add(cnt_ptr + pid_n, 1, sem='acq_rel')
        if prev == SPLIT_K - 1:
            tot = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for s in range(SPLIT_K):
                tot += tl.load(pbase + s * (BLOCK_M * BLOCK_N) + pidx, volatile=True)
            if launch_pdl:
                tl.extra.cuda.gdc_launch_dependents()
            tl.store(o_ptr + offs_m[:, None] * stride_om + offs_n[None, :], tot.to(tl.bfloat16), mask=mask)
            tl.atomic_xchg(cnt_ptr + pid_n, 0)
        elif launch_pdl:
            tl.extra.cuda.gdc_launch_dependents()


def _block_m(m: int) -> int:
    return 16 if m <= 16 else (32 if m <= 32 else 64)


# (N, K) -> (BLOCK_N, BLOCK_K, SPLIT_K, m_min, m_max); configs from skinny_bench3 / hc_fused_bench2 (HBM-cold, SM120)
GEMM_TABLE = {
    (96, 2560): (16, 128, 16, 2, 15),         # GDN in_proj_ba (cuBLAS pathology at M = 2..8)
    # v1 also dispatched shared_expert down_proj (2560, 640) for M <= 64: 5.1 vs 5.8 us in situ, but the inductor
    # kernel feeding it was regenerated 1.0 us slower (net zero at N39); dropped in v2 (2026-09-21 05:10 UTC).
}
HC_UP_CFG = (32, 64, 1)                        # input_mix_weight_up [10240, 320], silu fused
_BUFFERS: dict = {}
_DUMMY: dict = {}


def _buffers(key, device, nb, sk, bm, bn):
    b = _BUFFERS.get(key)
    if b is None:
        if sk == 1:
            d = _DUMMY.get(device)
            if d is None:
                d = _DUMMY[device] = (torch.zeros(1, device=device, dtype=torch.float32), torch.zeros(1, device=device, dtype=torch.int32))
            b = d
        else:
            b = (torch.zeros(nb * sk * bm * bn, device=device, dtype=torch.float32), torch.zeros(nb, device=device, dtype=torch.int32))
        _BUFFERS[key] = b
    return b


def preallocate(device) -> None:
    """Allocate every partial/counter buffer up front (never inside a CUDA-graph capture)."""
    for (n, k), (bn, bk, sk, m_min, m_max) in GEMM_TABLE.items():
        for bm in (16, 32, 64):
            if bm // 2 < m_max:
                _buffers((n, k, bm), device, triton.cdiv(n, bn), sk, bm, bn)
    _buffers(('hc_up', 0), device, 1, 1, 16, 16)


def _launch(x, w, out, cfg, silu_hc, key):
    M, K = x.shape
    N = w.shape[0]
    bn, bk, sk = cfg
    bm = _block_m(M)
    nb = triton.cdiv(N, bn)
    P, cnt = _buffers(key + (bm,), x.device, nb, sk, bm, bn)
    _skinny_kernel[(nb, sk)](x, w, P, cnt, out, M, N, K, x.stride(0), w.stride(0), out.stride(0),
                             BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, SPLIT_K=sk, SILU_HC=silu_hc,
                             launch_pdl=current_platform.is_arch_support_pdl(), num_warps=4, num_stages=3)
    return out


def _small_gemm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    """x @ w.T for the tabulated shapes and M ranges; otherwise exactly the vendor default (F.linear)."""
    M = x.shape[0]
    N, K = w.shape
    entry = GEMM_TABLE.get((N, K))
    if entry is None or x.dim() != 2 or x.dtype != torch.bfloat16 or w.dtype != torch.bfloat16 or x.stride(1) != 1 or w.stride(1) != 1:
        return torch.nn.functional.linear(x, w)
    bn, bk, sk, m_min, m_max = entry
    if not (m_min <= M <= m_max):
        return torch.nn.functional.linear(x, w)
    return _launch(x, w, x.new_empty(M, N), (bn, bk, sk), 0, (N, K))


def _small_gemm_fake(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[0], w.shape[0]))


def _hc_up_silu_gemm(lora: torch.Tensor, w: torch.Tensor, hc_count: int) -> torch.Tensor:
    """gate = bf16(silu(lora / hc)) @ w.T with the silu fused into the A load (decode); vendor path otherwise."""
    M, K = lora.shape
    N = w.shape[0]
    if (M > M_MAX or M == 0 or lora.dtype != torch.bfloat16 or w.dtype != torch.bfloat16 or lora.stride(1) != 1 or w.stride(1) != 1):
        from vllm.models.qwen3_8_flash_next.nvidia.ops.hc import hc_silu
        return torch.nn.functional.linear(hc_silu(lora, hc_count), w)
    return _launch(lora, w, lora.new_empty(M, N), HC_UP_CFG, hc_count, ('hc_up', 0))


def _hc_up_silu_gemm_fake(lora: torch.Tensor, w: torch.Tensor, hc_count: int) -> torch.Tensor:
    return lora.new_empty((lora.shape[0], w.shape[0]))


direct_register_custom_op(op_name="arc3_small_gemm", op_func=_small_gemm, fake_impl=_small_gemm_fake)
direct_register_custom_op(op_name="arc3_hc_up_silu_gemm", op_func=_hc_up_silu_gemm, fake_impl=_hc_up_silu_gemm_fake)


def small_gemm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    return torch.ops.vllm.arc3_small_gemm(x, w)


def hc_up_silu_gemm(lora: torch.Tensor, w: torch.Tensor, hc_count: int) -> torch.Tensor:
    return torch.ops.vllm.arc3_hc_up_silu_gemm(lora, w, hc_count)
