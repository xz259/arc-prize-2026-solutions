"""FP8 weight-only (W8A16, per-output-channel E4M3) for the model's BF16 dense projections.

Installed the same way as dense_projection_adapter_v1: wrap UnquantizedLinearMethod.process_weights_after_loading
and .apply on the pinned serving linear module. Selected layers are quantized once after loading (the BF16 weight is
released), packed for the Marlin FP8 kernel, and served through one opaque Torch operator. Every other layer keeps the
inner (original or dense-adapter) method untouched. Selection is by tier (ARC3_DENSE_FP8_TIER); 'inventory' records
every unquantized linear layer without changing anything.
"""
import hashlib
import json
import os
from pathlib import Path
import time

HERE = Path(__file__).resolve().parent
SCHEMA = 'arc3.dense-fp8.v1'
FP8_MAX = 448.0
TIER_SUFFIXES = {
    'A': ('.linear_attn.in_proj_qkvz', '.linear_attn.out_proj', '.self_attn.qkv_proj', '.self_attn.o_proj'),
    'B2': ('.attn_hyper_connection.input_mix_weight_down', '.attn_hyper_connection.input_mix_weight_up',
           '.attn_hyper_connection.block_inject_weight', '.mlp_hyper_connection.input_mix_weight_down',
           '.mlp_hyper_connection.input_mix_weight_up', '.mlp_hyper_connection.block_inject_weight',
           'hyper_connection_mixer.input_mix_weight_down', 'hyper_connection_mixer.input_mix_weight_up'),
    'C': ('.mlp.shared_expert.gate_up_proj', '.mlp.shared_expert.down_proj'),
}
TIER_SUFFIXES.update({
    'A_qkvz': ('.linear_attn.in_proj_qkvz',),
    'A_out': ('.linear_attn.out_proj', '.self_attn.o_proj'),
    'A_qkv': ('.self_attn.qkv_proj',),
})
TIERS = {'off': (), 'inventory': (), 'A': ('A',), 'AB2': ('A', 'B2'), 'AB2C': ('A', 'B2', 'C'),
         'A_qkvz': ('A_qkvz',), 'A_out': ('A_out',), 'A_qkv': ('A_qkv',)}   # per-family attribution variants
# lm_head (tier B1) goes through UnquantizedEmbeddingMethod, handled by install_lm_head when ARC3_DENSE_FP8_LM_HEAD=1.

LAYERS = {}
SEEN = set()
ORIGINAL = {}
OP = None
REPORT = {'schema': SCHEMA, 'pid': os.getpid(), 'tier': None, 'installed': False, 'inventory': {}, 'layers': {},
          'dispatch': [], 'bytes_bf16_released': 0, 'bytes_fp8_added': 0, 'errors': []}


def tier():
    t = os.environ.get('ARC3_DENSE_FP8_TIER', 'off')
    assert t in TIERS, t
    return t


def selected(prefix):
    if tier() in ('off', 'inventory'):
        return None
    if '.mtp.' in prefix or prefix.startswith('mtp.') or '.visual.' in prefix or prefix.startswith('visual'):
        return None
    for name in TIERS[tier()]:
        for suffix in TIER_SUFFIXES[name]:
            if prefix.endswith(suffix):
                return name
    return None


def save():
    root = os.environ.get('ARC3_DENSE_FP8_RECEIPTS')
    if root:
        p = Path(root); p.mkdir(parents=True, exist_ok=True)
        dst = p / f'worker-{os.getpid()}.json'; tmp = dst.with_suffix('.tmp')
        tmp.write_text(json.dumps(REPORT, indent=2) + '\n'); tmp.replace(dst)


def quantize_per_channel(weight):
    """BF16/FP16 (N, K) -> (E4M3 (N, K), FP32 scale (N,)), scale = rowwise amax / 448."""
    import torch
    assert weight.dim() == 2
    w = weight.float()
    amax = w.abs().amax(dim=1).clamp_min(1e-12)
    scale = amax / FP8_MAX
    q = (w / scale[:, None]).clamp_(-FP8_MAX, FP8_MAX).to(torch.float8_e4m3fn)
    recon_rel_rms = (torch.linalg.vector_norm(q.float() * scale[:, None] - w) / torch.linalg.vector_norm(w)).item()
    return q, scale, recon_rel_rms


class _Shim:
    """Minimal object with the attributes prepare_fp8_layer_for_marlin reads/writes."""


def prepare_marlin(q_nk, scale, orig_dtype):
    import torch
    from vllm.model_executor.layers.quantization.utils import marlin_utils_fp8 as mfp8
    n, k = q_nk.shape
    shim = _Shim()
    shim.output_size_per_partition = n; shim.input_size_per_partition = k
    shim.logical_widths = [n]; shim.orig_dtype = orig_dtype; shim.weight_block_size = None
    shim.weight = torch.nn.Parameter(q_nk.t().contiguous(), requires_grad=False)         # size_k_first (K, N)
    shim.weight_scale = torch.nn.Parameter(scale.to(torch.float32), requires_grad=False)
    mfp8.prepare_fp8_layer_for_marlin(shim, True, input_dtype=orig_dtype)
    return shim.weight.data, shim.weight_scale.data, shim.workspace


def register_op():
    global OP
    if OP is not None:
        return OP
    import torch
    from vllm.model_executor.layers.quantization.utils import marlin_utils_fp8 as mfp8

    @torch.library.custom_op('arc3_dense_fp8_v1::linear', mutates_args=())
    def linear(x: torch.Tensor, prefix: str) -> torch.Tensor:
        st = LAYERS[prefix]
        key = (prefix, tuple(x.shape))
        if key not in SEEN and len(REPORT['dispatch']) < 400:
            SEEN.add(key); REPORT['dispatch'].append({'prefix': prefix, 'input_shape': list(x.shape)}); save()
        return mfp8.apply_fp8_marlin_linear(x, st['qweight'], st['scales'], st['workspace'], st['n'], st['k'], None,
                                            input_dtype=st['dtype'])

    @linear.register_fake
    def fake(x, prefix):
        return x.new_empty((*x.shape[:-1], LAYERS[prefix]['n']))

    OP = linear
    return linear


def quantize_layer(layer, kind):
    import torch
    from vllm.model_executor.utils import replace_parameter
    prefix = layer.prefix
    w = layer.weight
    assert w.dtype in (torch.bfloat16, torch.float16) and w.dim() == 2, (prefix, w.dtype, tuple(w.shape))
    assert getattr(layer, 'tp_size', 1) == 1, prefix
    n, k = w.shape
    fake = os.environ.get('ARC3_DENSE_FP8_FAKE', 'off')
    if fake != 'off':   # stage 13: fake quantization on the E4M3 or E5M2 grid, BF16 GEMM path kept (no Marlin)
        assert fake in ('e4m3', 'e5m2'), fake
        dt, fmax = (torch.float8_e4m3fn, 448.0) if fake == 'e4m3' else (torch.float8_e5m2, 57344.0)
        wf = w.data.float(); amax = wf.abs().amax(dim=1).clamp_min(1e-12); sc = amax / fmax
        deq = (wf / sc[:, None]).clamp_(-fmax, fmax).to(dt).float() * sc[:, None]
        recon = (torch.linalg.vector_norm(deq - wf) / torch.linalg.vector_norm(wf)).item()
        w.data.copy_(deq.to(w.dtype)); del wf, deq
        REPORT['layers'][prefix] = dict(tier=kind, fake=fake, shape=[n, k], recon_rel_rms=recon)
        REPORT['fake'] = fake; save()
        return 'fake'
    q, scale, recon = quantize_per_channel(w.data)
    qweight, scales, workspace = prepare_marlin(q, scale, w.dtype)
    register_op()
    bytes_before = w.numel() * w.element_size()
    bytes_after = qweight.numel() * qweight.element_size() + scales.numel() * scales.element_size()
    LAYERS[prefix] = dict(qweight=qweight, scales=scales, workspace=workspace, n=n, k=k, dtype=w.dtype, tier=kind)
    layer.arc3_fp8_qweight = qweight; layer.arc3_fp8_scales = scales; layer.arc3_fp8_workspace = workspace
    replace_parameter(layer, 'weight', qweight)   # releases the BF16 parameter
    del q, scale, w
    torch.cuda.empty_cache()   # return the freed BF16 segments to the device before vLLM profiles the KV pool
    REPORT['layers'][prefix] = dict(tier=kind, shape=[n, k], bytes_bf16=bytes_before, bytes_fp8=bytes_after,
                                    weight_recon_rel_rms=recon, packed_shape=list(qweight.shape), packed_dtype=str(qweight.dtype))
    REPORT['bytes_bf16_released'] += bytes_before; REPORT['bytes_fp8_added'] += bytes_after
    save()


def install(linear_module):
    """Wrap UnquantizedLinearMethod on the pinned serving linear module (outermost wrapper)."""
    assert 'linear' not in ORIGINAL, 'Duplicate dense-fp8 installation'
    REPORT['tier'] = tier()
    method_cls = linear_module.UnquantizedLinearMethod
    inner_post = method_cls.process_weights_after_loading
    inner_apply = method_cls.apply
    ORIGINAL['linear'] = (inner_post, inner_apply)
    if tier() == 'off':
        REPORT['installed'] = False; save(); return
    # the opaque operator (and its vLLM import) is registered on the first quantized layer

    def post(self, layer):
        prefix = getattr(layer, 'prefix', '')
        try:
            w = layer.weight
            REPORT['inventory'][prefix] = dict(cls=type(layer).__name__, shape=list(w.shape), dtype=str(w.dtype),
                                               bias=getattr(layer, 'bias', None) is not None)
        except Exception as exc:  # noqa: BLE001
            REPORT['inventory'][prefix] = dict(cls=type(layer).__name__, error=repr(exc))
        kind = selected(prefix)
        if kind is None:
            return inner_post(self, layer)
        assert getattr(layer, 'bias', None) is None, ('bias not supported', prefix)
        try:
            mode = quantize_layer(layer, kind)
        except Exception as exc:  # noqa: BLE001
            REPORT['errors'].append(dict(prefix=prefix, error=repr(exc))); save()
            raise
        save()
        if mode == 'fake':
            return inner_post(self, layer)   # inner adapters register the (rounded) BF16 weight normally
        return None

    def apply(self, layer, x, bias=None):
        prefix = getattr(layer, 'prefix', '')
        if prefix in LAYERS:
            assert bias is None
            return OP(x, prefix)
        return inner_apply(self, layer, x, bias)

    method_cls.process_weights_after_loading = post
    method_cls.apply = apply
    REPORT['installed'] = True; save()


def install_lm_head(embedding_module):
    """Tier B1: lm_head through UnquantizedEmbeddingMethod (logits projection only, not the input embedding gather)."""
    if os.environ.get('ARC3_DENSE_FP8_LM_HEAD') != '1' or tier() in ('off', 'inventory'):
        return
    assert 'embedding' not in ORIGINAL
    method_cls = embedding_module.UnquantizedEmbeddingMethod
    inner_post = method_cls.process_weights_after_loading
    inner_apply = method_cls.apply
    ORIGINAL['embedding'] = (inner_post, inner_apply)

    def post(self, layer):
        # VocabParallelEmbedding/ParallelLMHead do not store their prefix; the model has exactly one ParallelLMHead.
        if type(layer).__name__ == 'ParallelLMHead':
            assert 'lm_head' not in LAYERS, 'second ParallelLMHead'
            layer.prefix = 'lm_head'
            REPORT['inventory']['lm_head'] = dict(cls='ParallelLMHead', shape=list(layer.weight.shape), dtype=str(layer.weight.dtype),
                                                  bias=getattr(layer, 'bias', None) is not None)
            assert getattr(layer, 'bias', None) is None
            if quantize_layer(layer, 'B1') == 'fake':
                return inner_post(self, layer)
            return None
        return inner_post(self, layer)

    def apply(self, layer, x, bias=None):
        prefix = getattr(layer, 'prefix', '')
        if prefix in LAYERS:
            assert bias is None
            return OP(x, prefix)
        return inner_apply(self, layer, x, bias)

    method_cls.process_weights_after_loading = post
    method_cls.apply = apply
    REPORT['lm_head_installed'] = True; save()
