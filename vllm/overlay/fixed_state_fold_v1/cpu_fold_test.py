"""Stage 3 gate 1 (CPU): fold option A on the snapshot specs — 5 groups, 4 fixed blocks, identical tensor layout,
39 lanes at 44K / 41 at 40K, scheduler unwrap -> CircularBufferManager, worker projection and Mamba discovery intact."""
import math, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fixed_state_fold_v1.specs_from_snapshot import specs_from_snapshot, fake_vllm_config
from fixed_state_fold_v1 import fold
from vllm.v1.core import kv_cache_utils as ku
from vllm.v1.kv_cache_interface import CircularBufferSpec, MambaSpec, UniformTypeKVCacheSpecs, KVCacheSpecRegistry, KVCacheConfig
from vllm.v1.core import single_type_kv_cache_manager as m

specs, live_blocks, avail, max_model_len = specs_from_snapshot(sys.argv[1]); cfg = fake_vllm_config(max_model_len)
base_groups = ku._get_kv_cache_groups_csa_linear(cfg, specs); base_blocks, base_tensors = ku._get_kv_cache_config_csa_linear(cfg, base_groups, avail)
originals = fold.install(ku, 'A')
groups = ku._get_kv_cache_groups_csa_linear(cfg, specs); num_blocks, tensors = ku._get_kv_cache_config_csa_linear(cfg, groups, avail)
assert num_blocks == base_blocks == 781 and len(tensors) == len(base_tensors) == 24
assert [(t.size, sorted(t.shared_by)) for t in tensors] == [(t.size, sorted(t.shared_by)) for t in base_tensors], 'tensor layout must be unchanged'
kinds = [('Uniform[' + type(g.kv_cache_spec.first_spec).__name__ + ']' if isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs) else type(g.kv_cache_spec).__name__, len(g.layer_names)) for g in groups]
print('folded groups:', kinds); assert len(groups) == 5 and kinds[1] == ('Uniform[CircularBufferSpec]', 13)
merged = groups[1]; assert sum(isinstance(s, MambaSpec) for s in merged.kv_cache_spec.kv_cache_specs.values()) == 1
assert merged.kv_cache_spec.max_num_blocks_per_req(cfg, max_model_len) == 1, 'ring and PLE must agree on a one-entry block table'
fixed = len(groups) - 1
for ctx in (40000, 44000, 44288, 47056):
    pages = math.ceil(ctx / groups[0].kv_cache_spec.block_size); lanes = (num_blocks - 1) // (fixed + pages); print(f'  context {ctx}: {pages} pages, {fixed + pages} blocks/lane, lanes {lanes}')
assert (num_blocks - 1) // (fixed + 16) == 39 and (num_blocks - 1) // (fixed + 15) == 41
# scheduler view: unwrap -> first spec -> manager
m.register_all_kvcache_specs(cfg)
sched = [g.kv_cache_spec.first_spec if isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs) else g.kv_cache_spec for g in groups]
managers = [KVCacheSpecRegistry.get_manager_class(s).__name__ for s in sched]; print('scheduler managers:', managers)
assert managers[1] == 'CircularBufferManager' and managers[2:] == ['MambaManager'] * 3
# worker view: projection keeps membership; Mamba discovery sees the three GDN groups only
proj = ku._project_kv_cache_groups_to_worker(groups, specs); assert [g.layer_names for g in proj] == [g.layer_names for g in groups]
from vllm.v1.worker.mamba_utils import get_mamba_groups
kvc = KVCacheConfig(num_blocks=num_blocks, kv_cache_tensors=tensors, kv_cache_groups=proj)
mg = get_mamba_groups(kvc); print('worker mamba groups:', {type(k).__name__ + ':' + k.mamba_type.name: v for k, v in mg.items()})
assert sorted(i for ids in mg.values() for i in ids) == [2, 3, 4]
# page sizes for the worker's per-layer views are the members' own (padded) sizes
assert merged.kv_cache_spec.kv_cache_specs[merged.layer_names[0]].page_size_bytes == 177152 and merged.kv_cache_spec.kv_cache_specs[merged.layer_names[-1]].page_size_bytes == 1638656
# restore and re-check the original path still works
ku._get_kv_cache_groups_csa_linear, ku._get_csa_linear_tensor_layout, ku._get_kv_cache_config_csa_linear = originals['groups'], originals['layout'], originals['config']
assert ku._get_kv_cache_config_csa_linear(cfg, ku._get_kv_cache_groups_csa_linear(cfg, specs), avail)[0] == 781
print('CPU FOLD TEST OK')
