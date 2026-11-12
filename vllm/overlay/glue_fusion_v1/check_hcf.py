"""GPU checks for the stage-14 kernels (kernels_hcf). argv[1] = directory holding the glue_fusion_v1 package.
For every M in the capture-size range: hcf_down (merged down+inject [336,10240] and final-mixer down [320,10240]),
up_gemm [10240,320] and splitk_gemm (router [512,2560], in_proj_ba [96,2560], index_qk [640,2560]) against the vendor
ops (F.linear, hc_silu): within bf16-rounding tolerance where dispatched, bit-exact where they fall back (M < 17,
M > 64, shape/block-size not tuned); split-K counters back to zero; eager calls deterministic; CUDA-graph replay with
new inputs equal to eager. in_proj_ba below M = 17 must equal the v1 small_gemm when that fallback is set."""
import sys
import torch
sys.path.insert(0, sys.argv[1])
from glue_fusion_v1 import kernels_hcf as K
from glue_fusion_v1 import kernels as V1
from vllm.models.qwen3_8_flash_next.nvidia.ops.hc import hc_silu

dev = torch.device('cuda'); torch.manual_seed(0)
K.preallocate(dev); V1.preallocate(dev); K.SMALL_M_FALLBACK = V1._small_gemm
ok = True
def check(name, cond, info=''):
    global ok; ok &= bool(cond)
    if not cond or 'PRINT' in name: print(('PASS' if cond else 'FAIL'), name, info, flush=True)
def rel(a, b): return ((a.float() - b.float()).abs().max() / (b.float().abs().max() + 1e-12)).item()
bf = lambda *s, sc=1.0: (torch.randn(*s, device=dev) * sc).to(torch.bfloat16)
Wd = bf(336, 10240, sc=0.01); Wf = bf(320, 10240, sc=0.01); Wu = bf(10240, 320, sc=0.05)
Wo = {'router': bf(512, 2560, sc=0.02), 'in_proj_ba': bf(96, 2560, sc=0.02), 'index_qk': bf(640, 2560, sc=0.02)}
Ms = [1, 2, 4, 8, 15, 16, 17, 24, 25, 30, 32, 33, 40, 47, 48, 64, 65, 128, 512]
n_disp = n_fb = 0
for M in Ms:
    disp = K.M_MIN <= M <= K.M_MAX
    xn = bf(M, 10240)
    for w, nm in ((Wd, 'down_inject'), (Wf, 'final_down')):
        out = K.hcf_down(xn, w, 320, 4); torch.cuda.synchronize()
        ref = torch.nn.functional.linear(xn, w); ref_l = hc_silu(ref[:, :320], 4)
        if disp:
            check(f'{nm} M={M} lora within tolerance', rel(out[:, :320], ref_l) < 2e-2, rel(out[:, :320], ref_l))
            if w.shape[0] > 320:
                check(f'{nm} M={M} injection within tolerance', rel(out[:, 320:324], ref[:, 320:324]) < 2e-2)
            again = K.hcf_down(xn, w, 320, 4); check(f'{nm} M={M} deterministic', torch.equal(out, again))
            n_disp += 1
        else:
            check(f'{nm} M={M} fallback bit-exact', torch.equal(out[:, :320], ref_l) and torch.equal(out[:, 320:], ref[:, 320:]))
            n_fb += 1
    a = hc_silu(torch.nn.functional.linear(xn, Wd)[:, :320], 4)   # strided view, as served
    out = K.up_gemm(a, Wu); ref = torch.nn.functional.linear(a, Wu); torch.cuda.synchronize()
    check(f'up M={M} ' + ('within tolerance' if disp else 'fallback bit-exact'), (rel(out, ref) < 1e-2) if disp else torch.equal(out, ref), rel(out, ref))
    for nm, w in Wo.items():
        x = bf(M, 2560); out = K.splitk_gemm(x, w); torch.cuda.synchronize()
        tuned = disp and K._block_m(M) in K.SPLITK_TABLE[tuple(w.shape)]
        if tuned:
            check(f'{nm} M={M} within tolerance', rel(out, torch.nn.functional.linear(x, w)) < 2e-2, rel(out, torch.nn.functional.linear(x, w)))
        else:
            ref = V1._small_gemm(x, w) if nm == 'in_proj_ba' else torch.nn.functional.linear(x, w)
            check(f'{nm} M={M} fallback equals {"v1 small_gemm" if nm == "in_proj_ba" else "F.linear"}', torch.equal(out, ref))
counters = [int(c.max()) for _, c in K._BUFFERS.values()]
check('split-K counters all zero after the sweep', max(counters) == 0, counters)
# CUDA graph: capture the chain at M = 25 and 47 on static buffers, replay with new inputs, compare with eager
for M in (25, 47):
    xs = bf(M, 10240); xr = bf(M, 2560)
    def chain():
        d = K.hcf_down(xs, Wd, 320, 4); g = K.up_gemm(d[:, :320], Wu); r = K.splitk_gemm(xr, Wo['router'])
        return d, g, r
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        chain()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        gd, gg, gr = chain()
    bad = 0
    for _ in range(5):
        xs.copy_(bf(M, 10240)); xr.copy_(bf(M, 2560)); graph.replay(); torch.cuda.synchronize()
        ed, eg, er = chain(); torch.cuda.synchronize()
        bad += int(not (torch.equal(gd, ed) and torch.equal(gg, eg) and torch.equal(gr, er)))
    check(f'graph replay == eager (M={M}, 5 new inputs)', bad == 0, bad)
print(f'dispatched cases {n_disp}, fallback cases {n_fb}; counters {max(counters)}')
print('ALL PASS' if ok else 'SOME FAILED'); sys.exit(0 if ok else 1)
