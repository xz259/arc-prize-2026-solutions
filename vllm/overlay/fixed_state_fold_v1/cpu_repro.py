"""Stage 3 gate 0: reproduce the live inventory on CPU from the saved snapshot with vLLM's own grouping + layout."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fixed_state_fold_v1.specs_from_snapshot import specs_from_snapshot, fake_vllm_config
from vllm.v1.core import kv_cache_utils as ku
from vllm.v1.kv_cache_interface import CircularBufferSpec, MambaSpec, UniformTypeKVCacheSpecs

snap = sys.argv[1]
specs, live_blocks, avail, max_model_len = specs_from_snapshot(snap)
cfg = fake_vllm_config(max_model_len)
groups = ku._get_kv_cache_groups_csa_linear(cfg, specs)
num_blocks, tensors = ku._get_kv_cache_config_csa_linear(cfg, groups, avail)
layout = ku._get_csa_linear_tensor_layout(groups)
print('groups:', [(len(g.layer_names), type(g.kv_cache_spec).__name__ if not isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs) else 'Uniform[' + type(g.kv_cache_spec.first_spec).__name__ + ']', g.kv_cache_spec.page_size_bytes) for g in groups])
print('num_blocks', num_blocks, 'live', live_blocks, 'bytes_per_block', layout.bytes_per_block, 'tensors', len(tensors))
fixed = sum(1 for g in groups[1:])   # every non-attention group claims one block per request
pages = math.ceil(max_model_len / groups[0].kv_cache_spec.block_size)
lanes = (num_blocks - 1) // (fixed + pages)
print(f'fixed blocks per lane {fixed}, history pages {pages}, lanes at {max_model_len}: {lanes}')
assert num_blocks == live_blocks == 781 and layout.bytes_per_block == 21789696 and len(tensors) == 24 and len(groups) == 6 and fixed == 5 and lanes == 37
shared = {t.shared_by[0]: t.shared_by for t in tensors}
main0 = next(v for k, v in shared.items() if k.endswith('layers.3.self_attn.attn'))
print('main tensor 0 shared_by:', [n.split('.')[-2] + '.' + n.split('.')[-1] for n in main0])
assert any(n.endswith('.ple') for n in main0) and sum('linear_attn' in n for n in main0) == 3
from vllm.v1.kv_cache_interface import KVCacheSpecRegistry
from vllm.v1.core import single_type_kv_cache_manager as m
m.register_all_kvcache_specs(cfg)
print('managers:', [KVCacheSpecRegistry.get_manager_class(g.kv_cache_spec.first_spec if isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs) else g.kv_cache_spec).__name__ for g in groups])
print('CPU REPRO OK')
