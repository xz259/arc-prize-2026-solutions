"""CPU gates (routing, quantization math, wrapper semantics) and GPU gates (Marlin numerics, graph replay) for dense_fp8."""
import argparse, json, os, sys, types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dense_fp8_adapter_v1 import runtime as r


def cpu_checks():
    os.environ['ARC3_DENSE_FP8_TIER'] = 'A'
    L = 'model.language_model.layers.'
    assert r.selected(L + '0.linear_attn.in_proj_qkvz') == 'A'
    assert r.selected(L + '3.self_attn.qkv_proj') == 'A' and r.selected(L + '3.self_attn.o_proj') == 'A'
    assert r.selected(L + '7.linear_attn.out_proj') == 'A'
    assert r.selected(L + '7.attn_hyper_connection.input_mix_weight_down') is None          # tier B2 not in A
    assert r.selected(L + '0.mlp.shared_expert.gate_up_proj') is None                        # tier C not in A
    assert r.selected(L + '0.mlp.gate') is None and r.selected('lm_head') is None
    assert r.selected('mtp.layers.0.linear_attn.out_proj') is None
    assert r.selected('model.mtp.layers.0.self_attn.o_proj') is None
    assert r.selected('model.visual.blocks.0.attn.proj') is None
    os.environ['ARC3_DENSE_FP8_TIER'] = 'AB2C'
    assert r.selected(L + '7.attn_hyper_connection.input_mix_weight_down') == 'B2'
    assert r.selected('model.language_model.hyper_connection_mixer.input_mix_weight_up') == 'B2'
    assert r.selected(L + '0.mlp.shared_expert.down_proj') == 'C'
    os.environ['ARC3_DENSE_FP8_TIER'] = 'inventory'
    assert r.selected(L + '0.linear_attn.in_proj_qkvz') is None
    os.environ['ARC3_DENSE_FP8_TIER'] = 'A_out'
    assert r.selected(L + '3.self_attn.o_proj') == 'A_out' and r.selected(L + '0.linear_attn.in_proj_qkvz') is None
    os.environ['ARC3_DENSE_FP8_TIER'] = 'A_qkvz'
    assert r.selected(L + '0.linear_attn.in_proj_qkvz') == 'A_qkvz' and r.selected(L + '3.self_attn.qkv_proj') is None
    # quantization math against a NumPy reference
    import numpy as np, torch
    torch.manual_seed(1)
    w = (torch.randn(64, 96) * 0.05).to(torch.bfloat16); w[3] *= 40; w[10] = 0
    q, s, recon = r.quantize_per_channel(w)
    wn = w.float().numpy(); amax = np.maximum(np.abs(wn).max(axis=1), 1e-12); sn = amax / 448.0
    assert np.allclose(s.numpy(), sn, rtol=1e-6)
    deq = q.float().numpy() * sn[:, None]
    assert np.abs(deq).max() <= 448.0 * sn.max() + 1e-6
    rel = np.linalg.norm(deq - wn) / np.linalg.norm(wn)
    assert abs(rel - recon) < 1e-6 and rel < 0.06, rel                       # E4M3 per-channel: ~3-5% relative RMS on Gaussian rows
    assert np.all(deq[10] == 0)
    # wrapper semantics with fake classes (no vLLM, no GPU)
    os.environ['ARC3_DENSE_FP8_TIER'] = 'inventory'
    calls = []

    class Method:
        def process_weights_after_loading(self, layer): calls.append(('post', layer.prefix))
        def apply(self, layer, x, bias=None): calls.append(('apply', layer.prefix)); return 'orig'
    r.ORIGINAL.clear(); r.REPORT['installed'] = False
    mod = types.SimpleNamespace(UnquantizedLinearMethod=Method)
    r.install(mod)
    layer = types.SimpleNamespace(prefix=L + '0.linear_attn.in_proj_qkvz', weight=w, bias=None)
    m = Method(); m.process_weights_after_loading(layer); assert m.apply(layer, None) == 'orig'
    assert calls == [('post', layer.prefix), ('apply', layer.prefix)]
    assert layer.prefix in r.REPORT['inventory'] and r.REPORT['inventory'][layer.prefix]['shape'] == [64, 96]
    assert not r.LAYERS
    print(json.dumps(dict(cpu_passed=True, tiers=list(r.TIERS), quant_rel_rms=float(rel))))


def gpu_checks(output):
    import torch, torch.nn.functional as F
    os.environ['ARC3_DENSE_FP8_TIER'] = 'A'
    r.ORIGINAL.clear(); r.LAYERS.clear(); r.REPORT['installed'] = False

    class Method:
        def process_weights_after_loading(self, layer): pass
        def apply(self, layer, x, bias=None): return F.linear(x, layer.weight, bias)
    import vllm.model_executor.utils  # replace_parameter target
    r.install(types.SimpleNamespace(UnquantizedLinearMethod=Method))
    rows = []
    torch.manual_seed(20260919)
    for name, (n, k) in {'in_proj_qkvz': (16384, 2560), 'out_proj': (2560, 6144), 'qkv_proj': (13312, 2560)}.items():
        W = (torch.randn(n, k, device='cuda') * 0.02).to(torch.bfloat16); W[::997] *= 8
        prefix = f'model.language_model.layers.0.{"linear_attn" if name != "qkv_proj" else "self_attn"}.{name}'
        layer = torch.nn.Module(); layer.prefix = prefix; layer.tp_size = 1; layer.bias = None
        layer.weight = torch.nn.Parameter(W.clone(), requires_grad=False)
        m = Method(); m.process_weights_after_loading(layer)
        assert prefix in r.LAYERS and layer.weight.dtype == torch.int32
        for M in (25, 30, 48, 2048):
            x = torch.randn(M, k, device='cuda', dtype=torch.bfloat16)
            ref = (x.float() @ W.float().t())
            y = m.apply(layer, x)
            err = (torch.linalg.vector_norm(y.float() - ref) / torch.linalg.vector_norm(ref)).item()
            bf16_err = (torch.linalg.vector_norm(F.linear(x, W).float() - ref) / torch.linalg.vector_norm(ref)).item()
            # graph replay tracks eager under input mutation and row permutation
            for _ in range(3): m.apply(layer, x)
            torch.cuda.synchronize(); g = torch.cuda.CUDAGraph()
            with torch.cuda.graph(g): yg = m.apply(layer, x)
            x.copy_(torch.randn(M, k, device='cuda', dtype=torch.bfloat16)); direct = m.apply(layer, x).clone(); g.replay(); torch.cuda.synchronize()
            assert torch.equal(direct, yg), (name, M)
            perm = torch.randperm(M, device='cuda'); x.copy_(x[perm]); g.replay(); torch.cuda.synchronize()
            if M <= 64:   # decode batches: exact row-position invariance (same contract as dense_projection_adapter_v1 at N25/N30)
                assert torch.equal(direct[perm], yg), (name, M, 'permutation')
            else:         # prefill sizes: deterministic, but tile position may change accumulation order by a BF16 ulp (as cuBLAS)
                again = m.apply(layer, x).clone(); assert torch.equal(again, yg), (name, M, 'run-to-run')
                pd = ((direct[perm].float() - yg.float()).abs().max() / direct.float().abs().mean()).item(); assert pd < 0.05, (name, M, pd)
            rows.append(dict(layer=name, M=M, w8a16_rel_rms=err, bf16_rel_rms=bf16_err,
                             weight_recon=r.REPORT['layers'][prefix]['weight_recon_rel_rms']))
            assert err < 0.08, rows[-1]
        del W, layer; torch.cuda.empty_cache()
    out = dict(gpu_passed=True, rows=rows, report_layers=r.REPORT['layers'])
    if output: Path(output).write_text(json.dumps(out, indent=1) + '\n')
    print(json.dumps(dict(gpu_passed=True, rows=rows)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--gpu', action='store_true'); p.add_argument('--output')
    a = p.parse_args()
    cpu_checks()
    if a.gpu: gpu_checks(a.output)
