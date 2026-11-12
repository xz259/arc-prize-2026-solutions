"""Stage 14 decode kernels (2026-09-27): the hyper-connection mix site and main-stream small GEMMs.

hcf_down(xn, w, lora_rank, hc_count)   one launch for down GEMM + split-K reduction + hc_silu at every HC mix site:
    out = xn @ w.T with a deterministic fixed-order split-K fixup by the last-arriving program of each N tile; the
    fixup writes bf16(silu(bf16(acc) / hc_count)) for columns < lora_rank and bf16(acc) for the injection/pad
    columns (the vendor's two materialized bf16 boundaries). Replaces cuBLAS split-K GEMM + splitKreduce + hc_silu.
up_gemm(a, w)                           the up projection [10240, 320] as 640 programs of 16 rows (cuBLAS runs it at
    ~320 CTAs); K = 320 in static BLOCK_K chunks. Bit-identical to cuBLAS on the benchmark data.
splitk_gemm(x, w)                       single-launch split-K GEMM with the same fixup for the main-stream decode
    GEMMs that cuBLAS runs as GEMM + splitKreduce: MoE router [512, 2560], GDN in_proj_ba [96, 2560] and the QSA
    index_qk_proj [640, 2560] (the last two tuned at BLOCK_M 32 only; vendor above M = 32).
All three run for 17 <= M <= 64 only (decode at the served lane counts); anything else, including prefill and the NLL
panel, takes exactly the vendor path. Partial/counter buffers are preallocated per (shape, BLOCK_M) and are only
safe on one stream: every dispatched site runs on the model's main stream (the shared expert, which runs on the
auxiliary stream, is not dispatched).
Configs: HBM-cold microbenchmarks (experiments/stage14_skinny_gemm_20260927, SM120, 96 distinct weights per shape)."""
import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

M_MIN, M_MAX = 17, 64


@triton.jit
def _splitk_kernel(x_ptr, w_ptr, p_ptr, cnt_ptr, o_ptr, M, stride_xm, stride_wn, stride_om,
                   N: tl.constexpr, K: tl.constexpr, LORA: tl.constexpr, HC: tl.constexpr,
                   BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr,
                   EPI_SILU: tl.constexpr, launch_pdl: tl.constexpr):
    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    K_PER: tl.constexpr = K // SPLIT_K
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    k0 = pid_k * K_PER
    mm = offs_m[:, None] < M
    nm = offs_n[:, None] < N
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K_PER, BLOCK_K):
        kk = k0 + k + offs_k
        x = tl.load(x_ptr + offs_m[:, None] * stride_xm + kk[None, :], mask=mm, other=0.0)
        w = tl.load(w_ptr + offs_n[:, None] * stride_wn + kk[None, :], mask=nm, other=0.0)
        acc += tl.dot(x, tl.trans(w))
    omask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    if SPLIT_K > 1:
        pidx = tl.arange(0, BLOCK_M)[:, None] * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
        pbase = p_ptr + pid_n * (SPLIT_K * BLOCK_M * BLOCK_N)
        tl.store(pbase + pid_k * (BLOCK_M * BLOCK_N) + pidx, acc)
        tl.debug_barrier()
        prev = tl.atomic_add(cnt_ptr + pid_n, 1, sem='acq_rel')
        if prev == SPLIT_K - 1:
            tot = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
            for s in range(SPLIT_K):
                tot += tl.load(pbase + s * (BLOCK_M * BLOCK_N) + pidx, volatile=True)
            y = tot.to(tl.bfloat16)
            if EPI_SILU:
                yf = y.to(tl.float32) / HC
                y = tl.where(offs_n[None, :] < LORA, (yf * tl.sigmoid(yf)).to(tl.bfloat16), y)
            if launch_pdl:
                tl.extra.cuda.gdc_launch_dependents()
            tl.store(o_ptr + offs_m[:, None] * stride_om + offs_n[None, :], y, mask=omask)
            tl.atomic_xchg(cnt_ptr + pid_n, 0)
        elif launch_pdl:
            tl.extra.cuda.gdc_launch_dependents()
    else:
        y = acc.to(tl.bfloat16)
        if EPI_SILU:
            yf = y.to(tl.float32) / HC
            y = tl.where(offs_n[None, :] < LORA, (yf * tl.sigmoid(yf)).to(tl.bfloat16), y)
        if launch_pdl:
            tl.extra.cuda.gdc_launch_dependents()
        tl.store(o_ptr + offs_m[:, None] * stride_om + offs_n[None, :], y, mask=omask)


@triton.jit
def _up_kernel(a_ptr, w_ptr, o_ptr, M, stride_am, stride_wn, stride_om, K: tl.constexpr,
               BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, launch_pdl: tl.constexpr):
    pid = tl.program_id(0)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = pid * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    mm = offs_m[:, None] < M
    if launch_pdl:
        tl.extra.cuda.gdc_wait()
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in tl.static_range(0, K, BLOCK_K):
        a = tl.load(a_ptr + offs_m[:, None] * stride_am + (k + offs_k)[None, :], mask=mm, other=0.0)
        w = tl.load(w_ptr + offs_n[:, None] * stride_wn + (k + offs_k)[None, :])
        acc += tl.dot(a, tl.trans(w))
    if launch_pdl:
        tl.extra.cuda.gdc_launch_dependents()
    tl.store(o_ptr + offs_m[:, None] * stride_om + offs_n[None, :], acc.to(tl.bfloat16), mask=mm)


def _block_m(m: int) -> int:
    return 32 if m <= 32 else 64


# (N, K) -> {BLOCK_M: (BLOCK_N, BLOCK_K, SPLIT_K, num_warps, num_stages)}
DOWN_TABLE = {
    (336, 10240): {32: (16, 128, 8, 4, 4), 64: (16, 128, 8, 8, 4)},   # input_mix_weight_down_block_inject
    (320, 10240): {32: (16, 128, 8, 4, 4), 64: (16, 128, 8, 8, 4)},   # final mixer input_mix_weight_down
}
SPLITK_TABLE = {   # main-stream shapes only (the partial buffers are shared per shape)
    (512, 2560): {32: (16, 128, 5, 4, 3), 64: (16, 64, 4, 8, 4)},     # MoE router gate: 4.81 -> 3.61 us (M=25)
    (96, 2560): {32: (16, 64, 10, 4, 3)},                             # GDN in_proj_ba: 4.33 -> 2.80 us (M=25)
    (640, 2560): {32: (32, 64, 5, 8, 3)},                             # QSA index_qk_proj: 5.17 -> 4.02 us (M=25)
}
# set by runtime when the v1 'gemm' part is active: in_proj_ba below M_MIN keeps its v1 kernel (cuBLAS pathology fix)
SMALL_M_FALLBACK = None
# (N, K) -> {BLOCK_M: (BLOCK_N, BLOCK_K, num_warps, num_stages)}
UP_TABLE = {
    (10240, 320): {32: (16, 64, 4, 1), 64: (16, 64, 4, 1)},           # input_mix_weight_up
}
_BUFFERS: dict = {}


def _buffers(n, k, bm, device):
    key = (n, k, bm)
    b = _BUFFERS.get(key)
    if b is None:
        table = DOWN_TABLE if (n, k) in DOWN_TABLE else SPLITK_TABLE
        bn, _, sk, _, _ = table[(n, k)][bm]
        nb = triton.cdiv(n, bn)
        b = _BUFFERS[key] = (torch.zeros(max(1, nb * sk * bm * bn), device=device, dtype=torch.float32),
                             torch.zeros(nb, device=device, dtype=torch.int32))
    return b


def preallocate(device) -> None:
    """Every partial/counter buffer, allocated before any CUDA-graph capture."""
    for table in (DOWN_TABLE, SPLITK_TABLE):
        for (n, k), cfgs in table.items():
            for bm in cfgs:
                _buffers(n, k, bm, device)


def _eligible(x, w, table):
    M = x.shape[0]
    return (M_MIN <= M <= M_MAX and x.dim() == 2 and w.dim() == 2 and tuple(w.shape) in table
            and _block_m(M) in table[tuple(w.shape)]
            and x.dtype == torch.bfloat16 and w.dtype == torch.bfloat16 and x.stride(1) == 1 and w.stride(1) == 1
            and x.shape[1] == w.shape[1])


def _launch_splitk(x, w, out, table, lora, hc, epi):
    M, K = x.shape
    N = w.shape[0]
    bm = _block_m(M)
    bn, bk, sk, nw, ns = table[(N, K)][bm]
    P, cnt = _buffers(N, K, bm, x.device)
    _splitk_kernel[(triton.cdiv(N, bn), sk)](x, w, P, cnt, out, M, x.stride(0), w.stride(0), out.stride(0),
                                             N=N, K=K, LORA=lora, HC=hc, BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk,
                                             SPLIT_K=sk, EPI_SILU=epi, launch_pdl=True, num_warps=nw, num_stages=ns)
    return out


def _hcf_down(xn: torch.Tensor, w: torch.Tensor, lora_rank: int, hc_count: int) -> torch.Tensor:
    """Columns < lora_rank: bf16(silu(bf16(xn @ w.T) / hc_count)); other columns: bf16(xn @ w.T)."""
    if not _eligible(xn, w, DOWN_TABLE):
        from vllm.models.qwen3_8_flash_next.nvidia.ops.hc import hc_silu
        out = torch.nn.functional.linear(xn, w)
        out[:, :lora_rank] = hc_silu(out[:, :lora_rank], hc_count)
        return out
    return _launch_splitk(xn, w, xn.new_empty(xn.shape[0], w.shape[0]), DOWN_TABLE, lora_rank, hc_count, True)


def _hcf_down_fake(xn, w, lora_rank: int, hc_count: int):
    return xn.new_empty((xn.shape[0], w.shape[0]))


def _up_gemm(a: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    if not _eligible(a, w, UP_TABLE):
        return torch.nn.functional.linear(a, w)
    M, K = a.shape
    N = w.shape[0]
    bm = _block_m(M)
    bn, bk, nw, ns = UP_TABLE[(N, K)][bm]
    out = a.new_empty(M, N)
    _up_kernel[(triton.cdiv(N, bn),)](a, w, out, M, a.stride(0), w.stride(0), out.stride(0), K=K,
                                      BLOCK_M=bm, BLOCK_N=bn, BLOCK_K=bk, launch_pdl=True, num_warps=nw, num_stages=ns)
    return out


def _up_gemm_fake(a, w):
    return a.new_empty((a.shape[0], w.shape[0]))


def _splitk_gemm(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
    if not _eligible(x, w, SPLITK_TABLE):
        if SMALL_M_FALLBACK is not None:
            return SMALL_M_FALLBACK(x, w)   # v1 table shapes in their M range, F.linear otherwise
        return torch.nn.functional.linear(x, w)
    return _launch_splitk(x, w, x.new_empty(x.shape[0], w.shape[0]), SPLITK_TABLE, 0, 1, False)


def _splitk_gemm_fake(x, w):
    return x.new_empty((x.shape[0], w.shape[0]))


direct_register_custom_op(op_name="arc3_hcf_down", op_func=_hcf_down, fake_impl=_hcf_down_fake)
direct_register_custom_op(op_name="arc3_up_gemm", op_func=_up_gemm, fake_impl=_up_gemm_fake)
direct_register_custom_op(op_name="arc3_splitk_gemm", op_func=_splitk_gemm, fake_impl=_splitk_gemm_fake)


def hcf_down(xn, w, lora_rank, hc_count):
    return torch.ops.vllm.arc3_hcf_down(xn, w, lora_rank, hc_count)


def up_gemm(a, w):
    return torch.ops.vllm.arc3_up_gemm(a, w)


def splitk_gemm(x, w):
    return torch.ops.vllm.arc3_splitk_gemm(x, w)
