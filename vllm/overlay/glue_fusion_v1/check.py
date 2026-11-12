"""GPU gate for glue_fusion_v1 (pod, runtime python, server down; cwd = runs/dense_fp8_v1):
pins; hc: fused silu-GEMM vs vendor hc_silu+F.linear for every capture size and the M>64 fallback (bit-equal fraction,
max |diff|, determinism, graph replay); gemm: table shapes vs F.linear (numerics, determinism, fallback outside the M
range) plus a cold timing of F.linear vs the op at M=8 for in_proj_ba; gdn: strided gated norm == vendor copy path."""
import hashlib, importlib.util, statistics, sys, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from glue_fusion_v1.import_hook import PINS
ok = True
for name, pin in PINS.items():
    h = hashlib.sha256(Path(importlib.util.find_spec(name).origin).read_bytes()).hexdigest()
    print(f'pin {name}: {"ok" if h == pin else "MISMATCH " + h}'); ok &= h == pin
dev = torch.device('cuda'); torch.manual_seed(0)
from vllm.models.qwen3_8_flash_next.nvidia.ops.hc import hc_silu
from glue_fusion_v1.kernels import hc_up_silu_gemm, small_gemm, preallocate, GEMM_TABLE
preallocate(dev)
HC, R = 4, 320
w_up = (torch.randn(10240, R, device=dev) * 0.05).to(torch.bfloat16)
print('--- hc: silu-fused up GEMM vs vendor (hc_silu -> F.linear)')
for M in (1, 2, 4, 8, 16, 24, 25, 30, 32, 39, 52, 64, 100, 4096):
    lora = (torch.randn(M, R, device=dev) * 2.0).to(torch.bfloat16)
    ref = torch.nn.functional.linear(hc_silu(lora, HC), w_up)
    f1 = hc_up_silu_gemm(lora, w_up, HC); f2 = hc_up_silu_gemm(lora, w_up, HC)
    eq = torch.eq(ref, f1).float().mean().item(); rel = (ref.float() - f1.float()).abs().max().item() / ref.float().abs().max().item()
    det = torch.equal(f1, f2); ok &= det and rel < 1e-2 and (M <= 64 or eq == 1.0)
    print(f'  M={M:4d}: bit-equal {eq*100:6.2f}%  max rel diff {rel:.1e}  deterministic {det}{"  (vendor fallback path)" if M > 64 else ""}')
print('--- gemm: table shapes vs F.linear')
for (N, K), (bn, bk, sk, m_min, m_max) in GEMM_TABLE.items():
    w = (torch.randn(N, K, device=dev) * 0.02).to(torch.bfloat16)
    for M in (1, 2, 4, 8, 15, 16, 24, 32, 39, 64, 65, 4096):
        x = (torch.randn(M, K, device=dev) * 0.5).to(torch.bfloat16)
        ref = torch.nn.functional.linear(x, w); f1 = small_gemm(x, w); f2 = small_gemm(x, w)
        eq = torch.eq(ref, f1).float().mean().item(); rel = (ref.float() - f1.float()).abs().max().item() / ref.float().abs().max().item()
        det = torch.equal(f1, f2); inrange = m_min <= M <= m_max; ok &= det and rel < 1e-2 and (inrange or eq == 1.0)
        print(f'  [{N},{K}] M={M:4d}: {"triton " if inrange else "vendor "} bit-equal {eq*100:6.2f}%  max rel diff {rel:.1e}  deterministic {det}')
print('--- graph replay equality (served kernels run inside CUDA graphs)')
def graph_eq(fn):
    eager = fn().clone(); out = torch.empty_like(eager)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s): out.copy_(fn())
    torch.cuda.current_stream().wait_stream(s)
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): out.copy_(fn())
    out.zero_(); g.replay(); torch.cuda.synchronize(); return torch.equal(out, eager)
lora39 = (torch.randn(39, R, device=dev) * 2.0).to(torch.bfloat16); x8 = torch.randn(8, 2560, device=dev).to(torch.bfloat16); w_ba = (torch.randn(96, 2560, device=dev) * 0.02).to(torch.bfloat16)
x39 = torch.randn(39, 640, device=dev).to(torch.bfloat16); w_sd = (torch.randn(2560, 640, device=dev) * 0.02).to(torch.bfloat16)
for name, fn in (('hc_up M39', lambda: hc_up_silu_gemm(lora39, w_up, HC)), ('in_proj_ba M8', lambda: small_gemm(x8, w_ba)), ('shared_down M39', lambda: small_gemm(x39, w_sd))):
    r = graph_eq(fn); ok &= r; print(f'  {name}: graph replay == eager: {r}')
print('--- cold timing, in_proj_ba [96,2560] at M=8: F.linear (as served) vs op, 36 rotating weights')
flush = torch.empty(256 * 2**20, device=dev, dtype=torch.uint8); ws = [(torch.randn(96, 2560, device=dev) * 0.02).to(torch.bfloat16) for _ in range(36)]
def tg(fn, n):
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s): fn(); fn()
    torch.cuda.current_stream().wait_stream(s); g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g): fn()
    g.replay(); torch.cuda.synchronize(); t0 = torch.cuda.Event(enable_timing=True); t1 = torch.cuda.Event(enable_timing=True); ts = []
    for _ in range(40): t0.record(); g.replay(); t1.record(); torch.cuda.synchronize(); ts.append(t0.elapsed_time(t1))
    return statistics.median(ts) * 1e3 / n
tf = tg(lambda: flush.zero_(), 1)
def lin():
    flush.zero_()
    for w in ws: torch.nn.functional.linear(x8, w)
def op():
    flush.zero_()
    for w in ws: small_gemm(x8, w)
print(f'  F.linear {tg(lin, 36) - tf/36:.1f} us  |  arc3_small_gemm {tg(op, 36) - tf/36:.1f} us')
print('--- gdn: strided gated RMSNorm == vendor copy path')
from vllm.third_party.flash_linear_attention.ops.layernorm_guard import layer_norm_fwd, layer_norm_fwd_kernel, calc_rows_per_block
from vllm.triton_utils import triton
H, D = 48, 128; weight = (1 + 0.1 * torch.randn(D, device=dev)).to(torch.bfloat16); eps = 1e-6
for M in (1, 8, 39, 52):
    mixed = torch.randn(M, 4 * H * D, device=dev).to(torch.bfloat16); gate = mixed[:, -H * D:].reshape(M, H, D)
    x = torch.randn(M, H, D, device=dev).to(torch.bfloat16); ref = torch.empty_like(x)
    layer_norm_fwd(x.reshape(-1, D), weight, None, eps, z=gate.reshape(-1, D), out=ref.reshape(-1, D), group_size=D, norm_before_gate=True, is_rms_norm=True, activation='silu')
    y = x.clone(); x2 = y.view(M, H * D); z2 = gate.view(M, H * D); rpb = calc_rows_per_block(M * H, dev); rstd = torch.empty((H * M,), dtype=torch.float32, device=dev)
    layer_norm_fwd_kernel[(triton.cdiv(M, rpb), H)](x2, x2, weight.repeat(H), None, z2, None, rstd, x2.stride(0), x2.stride(0), z2.stride(0), M, D, eps,
                                                    BLOCK_N=D, ROWS_PER_BLOCK=rpb, HAS_BIAS=False, HAS_Z=True, NORM_BEFORE_GATE=True, IS_RMS_NORM=True, num_warps=1, ACTIVATION='silu')
    same = torch.equal(ref, y); ok &= same; print(f'  M={M:2d}: bit-identical {same}')
print('CHECK', 'PASS' if ok else 'FAIL')
