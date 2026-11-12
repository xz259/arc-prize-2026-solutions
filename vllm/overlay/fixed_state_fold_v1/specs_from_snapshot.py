"""Rebuild per-layer KVCacheSpecs (as the model reports them, before grouping) from a saved arc3_memory_snapshot."""
import json, re
from types import SimpleNamespace
import torch
from vllm.v1.kv_cache_interface import (CircularBufferSpec, FullAttentionSpec, KVQuantMode, MLAAttentionSpec, MambaSpec)
from vllm.model_executor.layers.mamba.abstract import MambaAttentionBackendEnum  # noqa: F401  (name used by the enum lookup)


def _parse_repr(text):
    """Evaluate a dataclass repr like FullAttentionSpec(block_size=2768, dtype=torch.uint8, kv_quant_mode=<KVQuantMode.NVFP4: 5>, ...)."""
    text = re.sub(r"<KVQuantMode\.([A-Z0-9_]+): \d+>", r"KVQuantMode.\1", text)
    text = re.sub(r"<MambaAttentionBackendEnum\.([A-Z0-9_]+): [^>]+>", r"MambaAttentionBackendEnum.\1", text)
    ns = dict(torch=torch, KVQuantMode=KVQuantMode, MambaAttentionBackendEnum=MambaAttentionBackendEnum,
              FullAttentionSpec=FullAttentionSpec, MLAAttentionSpec=MLAAttentionSpec, CircularBufferSpec=CircularBufferSpec, MambaSpec=MambaSpec)
    return eval(text, ns)


def specs_from_snapshot(path):
    s = json.load(open(path))['results'][0]; ci = s['cache_inventory']; out = {}
    for g in ci['groups']:
        spec = g['spec']
        if 'kv_cache_specs' in spec:
            for name, r in spec['kv_cache_specs'].items():
                sp = _parse_repr(r)
                out[name] = sp.__class__(**{**sp.__dict__, 'page_size_padded': None})   # grouping assigns the padding
        else:
            assert g['spec_type'] == 'MambaSpec'
            mt = spec['mamba_type'].split('.')[-1]
            dtypes = tuple(getattr(torch, d.split('.')[-1]) for d in spec['dtypes'])
            ms = MambaSpec(block_size=spec['block_size'], shapes=tuple(tuple(x) for x in spec['shapes']), dtypes=dtypes,
                           page_size_padded=None, mamba_type=getattr(MambaAttentionBackendEnum, mt), mamba_cache_mode=spec['mamba_cache_mode'],
                           num_speculative_blocks=spec['num_speculative_blocks'], tp_replicated=spec['tp_replicated'])
            for name in g['layer_names']:
                out[name] = ms
    return out, ci['num_blocks'], s['runtime_memory']['available_kv_cache_memory_bytes'], s['runtime_memory']['max_model_len']


def fake_vllm_config(max_model_len, num_kv_heads=2):
    """The fields the CSA+linear grouping and config functions read."""
    model_config = SimpleNamespace(max_model_len=max_model_len, get_num_kv_heads=lambda pc: num_kv_heads, get_total_num_hidden_layers=lambda: 48)
    parallel_config = SimpleNamespace(pipeline_parallel_size=1, tensor_parallel_size=1, data_parallel_size=1, world_size=1, rank=0)
    cache_config = SimpleNamespace(enable_prefix_caching=False, prefix_match_unit=None, num_gpu_blocks_override=None, block_size=2768, mamba_cache_mode='none', mamba_block_size=None)
    return SimpleNamespace(model_config=model_config, parallel_config=parallel_config, cache_config=cache_config,
                           speculative_config=None, scheduler_config=SimpleNamespace(max_num_seqs=37, max_num_batched_tokens=8192))
