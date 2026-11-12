"""CPU routing and GPU numerical/graph gates for the serving projection adapter."""
import argparse
import json
from pathlib import Path
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dense_projection_adapter_v1 import runtime as r


def main():
    p = argparse.ArgumentParser(); p.add_argument('--gpu', action='store_true'); p.add_argument('--output', type=Path)
    a = p.parse_args()
    r.verify_sources()
    assert r.family('model.language_model.layers.0.linear_attn.in_proj_qkvz') == 'qkvz'
    assert r.family('model.language_model.layers.3.self_attn.o_proj') == 'out'
    assert r.family('model.language_model.layers.0.mlp.experts.out_proj') is None
    assert r.family('model.mtp.layers.0.linear_attn.out_proj') is None
    assert r.family('model.visual.blocks.0.attn.proj') is None
    if not a.gpu:
        print(json.dumps(dict(cpu_passed=True, sources=r.PINS))); return
    import torch
    import torch.nn.functional as F
    torch.manual_seed(20260915); torch.set_num_threads(1)

    class Method:
        def process_weights_after_loading(self, layer): pass
        def apply(self, layer, x, bias=None): return F.linear(x, layer.weight, bias)

    r.install(types.SimpleNamespace(UnquantizedLinearMethod=Method))
    rows = []
    with torch.inference_mode():
        for family in ('qkvz', 'out'):
            shape = r.SHAPES[family]
            suffix = 'in_proj_qkvz' if family == 'qkvz' else 'out_proj'
            prefix = 'model.language_model.layers.0.linear_attn.' + suffix
            layer = types.SimpleNamespace(prefix=prefix, tp_size=1,
                weight=torch.randn(shape, device='cuda', dtype=torch.bfloat16) / shape[1]**0.5)
            method = Method(); method.process_weights_after_loading(layer)
            for n in (25, 30):
                x = torch.randn(n, shape[1], device='cuda', dtype=torch.bfloat16)
                apply = lambda x: method.apply(layer, x)
                expected = apply(x).clone()
                # Capture the actual opaque serving operator, then mutate inputs.
                for _ in range(3): apply(x)
                torch.cuda.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph): y = apply(x)
                original = x.clone()
                for power in (0, -8, -16, -32):
                    x.copy_((original.float()*2.**power).to(x.dtype))
                    direct = apply(x).clone()
                    graph.replay(); torch.cuda.synchronize()
                    assert torch.equal(direct, y)
                    for _ in range(4):
                        graph.replay(); torch.cuda.synchronize(); assert torch.equal(direct, y)
                    perm = torch.randperm(n, device='cuda'); x.copy_(x[perm])
                    graph.replay(); torch.cuda.synchronize()
                    assert torch.equal(direct, y[torch.argsort(perm)])
                    x.copy_((original.float()*2.**power).to(x.dtype))
                    columns = torch.arange(0, shape[0], shape[0]//64, device='cuda')
                    reference = x.double() @ layer.weight[columns].double().T
                    relative = float((direct[:,columns].double()-reference).norm()/reference.norm())
                    assert relative < .005, (family,n,power,relative)
                    rows.append(dict(family=family,n=n,power=power,relative_l2_fp64=relative,
                                     finite=bool(torch.isfinite(direct).all()),graph_permutation_exact=True))
                x.copy_(original); graph.replay(); torch.cuda.synchronize(); assert torch.equal(expected,y)
                del graph,y
            # Non-specialized and noncontiguous geometries must use exact stock.
            for n in (1,24,32,127):
                x = torch.randn(n,shape[1]*2,device='cuda',dtype=torch.bfloat16)[:,::2]
                assert torch.equal(apply(x),F.linear(x,layer.weight))
            # Dynamic Dynamo tracing must retain opaque dispatch for N25/N30.
            compiled = torch.compile(apply, dynamic=True, fullgraph=True)
            for n in (32,25,30):
                x = torch.randn(n,shape[1],device='cuda',dtype=torch.bfloat16)
                assert torch.equal(compiled(x),apply(x))
            del compiled,layer
    result = dict(passed=True,rows=rows,dispatch=r.REPORT)
    assert set(r.MODULES) == set(r.PINS)
    if a.output: a.output.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(passed=True,cases=len(rows),loaded=list(r.MODULES))))


if __name__ == '__main__': main()
