"""Use the four reviewed generated GEMMs behind one opaque Torch operator.

Only N25/N30 BF16 projections on the measured SM120 geometry are specialized.
Other geometries retain the original vLLM method. No global autotuning or
compile_sizes changes. Generated sources are admitted by exact SHA256.
"""
import hashlib
import json
import os
from pathlib import Path
import sys
import types

HERE = Path(__file__).resolve().parent
PINS = {
    'out_n25': '6d203a9f8b2327c9ada6baad545939fcee2ee54f1838459b7ccb236f398ea32a',
    'out_n30': 'b1ea22bcc059c69b325e9ab0eb70b62ce8ddabe9d5a8f96a2060ada5a95c963c',
    'qkvz_n25': 'bdf070f97eadbb82ea084434d2b7d07cb905706ef752a5dbb3b76107ea1934c1',
    'qkvz_n30': '4528d3321066884402fcbab6a094071f69228ed0d7ff340721704bc88e270f50',
}
SHAPES = {'qkvz': (16384, 2560), 'out': (2560, 6144)}
MODULES = {}
LAYERS = {}
SEEN = set()
LINEAR = None
ORIGINAL = None
REPORT = {'schema': 'arc3.dense-projection.v1', 'pid': os.getpid(), 'pins': PINS,
          'loaded': [], 'layers': {}, 'dispatch': [], 'installed': False}


def family(prefix):
    if '.layers.' not in prefix or '.mtp.' in prefix:
        return None
    if prefix.endswith('.linear_attn.in_proj_qkvz'):
        return 'qkvz'
    if prefix.endswith(('.linear_attn.out_proj', '.self_attn.o_proj')):
        return 'out'
    return None


def save():
    root = os.environ.get('ARC3_DENSE_RECEIPTS')
    if root:
        p = Path(root); p.mkdir(parents=True, exist_ok=True)
        dst = p / f'worker-{os.getpid()}.json'
        tmp = dst.with_suffix('.tmp')
        tmp.write_text(json.dumps(REPORT, indent=2) + '\n'); tmp.replace(dst)


def verify_sources():
    for key, digest in PINS.items():
        assert hashlib.sha256((HERE/'generated'/f'{key}.py').read_bytes()).hexdigest() == digest, key
    return dict(PINS)


def load_kernel(key):
    if key not in MODULES:
        import torch
        device = torch.cuda.get_device_properties(0)
        assert (device.major, device.minor, device.multi_processor_count) == (12, 0, 188)
        path = HERE/'generated'/f'{key}.py'
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == PINS[key]
        name = f'arc3_dense_fixed_{key}'
        module = types.ModuleType(name); module.__file__ = str(path)
        sys.modules[name] = module
        exec(compile(raw, str(path), 'exec'), module.__dict__)
        MODULES[key] = module
        REPORT['loaded'].append(key); save()
    return MODULES[key].call


def register():
    global LINEAR
    if LINEAR is not None:
        return LINEAR
    import torch

    @torch.library.custom_op('arc3_dense_v1::linear', mutates_args=())
    def linear(x: torch.Tensor, weight: torch.Tensor, prefix: str) -> torch.Tensor:
        f = family(prefix)
        eligible = (f is not None and x.ndim == 2 and x.shape[0] in (25, 30)
                    and tuple(weight.shape) == SHAPES[f] and x.shape[1] == weight.shape[1]
                    and x.dtype == weight.dtype == torch.bfloat16
                    and x.device == weight.device == torch.device('cuda', 0)
                    and x.is_contiguous() and weight.is_contiguous())
        key = f'{f}_n{x.shape[0]}' if eligible else 'original'
        seen = (prefix, key)
        if seen not in SEEN:
            SEEN.add(seen)
            REPORT['dispatch'].append({'prefix': prefix, 'kernel': key,
                                       'input_shape': list(x.shape), 'stride': list(x.stride())})
            save()
        if eligible:
            return load_kernel(key)([x, weight])[0]
        method, layer = LAYERS[prefix]
        return ORIGINAL(method, layer, x, None)

    @linear.register_fake
    def fake(x, weight, prefix):
        return x.new_empty((*x.shape[:-1], weight.shape[0]))

    LINEAR = linear
    return linear


def install(linear_module):
    global ORIGINAL
    assert ORIGINAL is None, 'Duplicate dense installation'
    verify_sources()
    ORIGINAL = linear_module.UnquantizedLinearMethod.apply
    original_post = linear_module.UnquantizedLinearMethod.process_weights_after_loading
    op = register()

    def post(self, layer):
        result = original_post(self, layer)
        f = family(layer.prefix)
        if f is not None:
            import torch
            assert layer.tp_size == 1 and tuple(layer.weight.shape) == SHAPES[f], layer.prefix
            assert layer.weight.dtype == torch.bfloat16 and layer.weight.is_contiguous()
            assert layer.prefix not in LAYERS
            LAYERS[layer.prefix] = (self, layer)
            REPORT['layers'][layer.prefix] = dict(family=f, shape=list(layer.weight.shape), dtype=str(layer.weight.dtype))
            save()
        return result

    def apply(self, layer, x, bias=None):
        # Shape selection stays INSIDE the opaque op; Dynamo must not erase the
        # N25/N30 path when tracing with a different warmup batch size.
        if bias is None and family(layer.prefix) is not None:
            return op(x, layer.weight, layer.prefix)
        return ORIGINAL(self, layer, x, bias)

    linear_module.UnquantizedLinearMethod.process_weights_after_loading = post
    linear_module.UnquantizedLinearMethod.apply = apply
    REPORT['installed'] = True; save()
