"""Stage 3 gate (CPU): fold option B on the snapshot specs — 4 groups, 3 fixed blocks per lane, compressed page
184,320 B shared by compressed keys / rings / PLE, PLE in compressed tensor 0, rings in GDN group A, PLE in GDN group B;
block and lane arithmetic for the natural and the pinned pool; scheduler managers; owner/tensor page consistency."""
import math, sys
from dataclasses import replace
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from fixed_state_fold_v1.specs_from_snapshot import specs_from_snapshot, fake_vllm_config
from fixed_state_fold_v1 import fold
from vllm.v1.core import kv_cache_utils as ku
from vllm.v1.kv_cache_interface import (CircularBufferSpec, FullAttentionSpec, KVCacheConfig, KVCacheSpecRegistry, MLAAttentionSpec,
                                        MambaSpec, UniformTypeKVCacheSpecs)
from vllm.v1.core import single_type_kv_cache_manager as m

PINNED = 17018807910                      # --kv-cache-memory-bytes of the accepted composition (15.85 GiB)
specs, live_blocks, avail, max_model_len = specs_from_snapshot(sys.argv[1]); cfg = fake_vllm_config(max_model_len)
base_groups = ku._get_kv_cache_groups_csa_linear(cfg, specs); base_blocks, base_tensors = ku._get_kv_cache_config_csa_linear(cfg, base_groups, avail)
assert base_blocks == 781 and len(base_groups) == 6 and len(base_tensors) == 24
originals = fold.install(ku, 'B')
groups = ku._get_kv_cache_groups_csa_linear(cfg, specs)
kind = lambda g: ('Uniform[' + type(g.kv_cache_spec.first_spec).__name__ + ']' if isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs) else type(g.kv_cache_spec).__name__, len(g.layer_names))
kinds = [kind(g) for g in groups]; print('folded groups:', kinds)
assert kinds == [('Uniform[FullAttentionSpec]', 24), ('Uniform[CircularBufferSpec]', 24), ('Uniform[MambaSpec]', 13), ('MambaSpec', 12)], kinds
attn, a2, b2, c = groups
comp = [s for s in attn.kv_cache_spec.kv_cache_specs.values() if type(s) is MLAAttentionSpec]
full = [s for s in attn.kv_cache_spec.kv_cache_specs.values() if type(s) is FullAttentionSpec]
assert len(comp) == 12 and all(s.page_size_bytes == 184320 and s.unpadded_page_size_bytes == 177152 for s in comp)
assert len(full) == 12 and all(s.page_size_bytes == 1638656 and s.page_size_padded is None for s in full)
rings = [s for s in a2.kv_cache_spec.kv_cache_specs.values() if type(s) is CircularBufferSpec]
gdn_a = [s for s in a2.kv_cache_spec.kv_cache_specs.values() if isinstance(s, MambaSpec) and not s.tp_replicated]
assert len(rings) == 12 and all(s.page_size_bytes == 184320 and s.unpadded_page_size_bytes == 1120 for s in rings)
assert len(gdn_a) == 12 and all(s.page_size_bytes == 1638656 and not s.tp_replicated for s in gdn_a)
ple = [s for s in b2.kv_cache_spec.kv_cache_specs.values() if isinstance(s, MambaSpec) and s.tp_replicated]
assert len(ple) == 1 and ple[0].page_size_bytes == 184320 and replace(ple[0], page_size_padded=None).page_size_bytes == 184320
first = a2.kv_cache_spec.first_spec
assert type(first) is CircularBufferSpec and a2.kv_cache_spec.block_size == first.block_size == 4, 'merged group must be ring-first (kernel block size 4, no slot mapping, CircularBufferManager)'
assert all('raw_key_cache' in n for n in a2.layer_names[:12]) and all('linear_attn' in n for n in a2.layer_names[12:24]) and len(a2.layer_names) == 24
assert isinstance(b2.kv_cache_spec.first_spec, MambaSpec) and not b2.kv_cache_spec.first_spec.tp_replicated and all('linear_attn' in n for n in b2.layer_names[:12]) and b2.layer_names[12].endswith('.ple')
assert b2.kv_cache_spec.max_num_blocks_per_req(cfg, max_model_len) == 1
assert all(s.block_size % first.block_size == 0 for s in rings), 'KVBlockZeroer: every attention member must be divisible by the kernel block size'
assert a2.kv_cache_spec.max_num_blocks_per_req(cfg, max_model_len) == 1, 'every member must agree on a one-entry block table'
assert isinstance(c.kv_cache_spec, MambaSpec) and not c.kv_cache_spec.tp_replicated and len(c.layer_names) == 12
layout = ku._get_csa_linear_tensor_layout(groups)
assert layout.compressed_page_size == 184320 and layout.main_kv_page_size == 1638656 and layout.bytes_per_block == 21875712
assert len(layout.mamba_groups) == 4 and len(layout.compressor_state_names) == 12 and layout.mamba_groups[-1].kv_cache_spec.tp_replicated
for pool, expect in ((avail, avail // 21875712), (PINNED, 777), (780 * 21875712, 780)):
    nb, tensors = ku._get_kv_cache_config_csa_linear(cfg, groups, pool)
    assert nb == expect and len(tensors) == 24, (pool, nb)
    mains, comps = tensors[:12], tensors[12:]
    assert all(t.size == 1638656 * nb for t in mains) and all(t.size == 184320 * nb for t in comps)
    assert not any(n.endswith('.ple') for t in mains for n in t.shared_by) and all(sum('linear_attn' in n for n in t.shared_by) == 3 and len(t.shared_by) == 4 for t in mains)
    assert any(n.endswith('.ple') for n in comps[0].shared_by) and len(comps[0].shared_by) == 3 and all(len(t.shared_by) == 2 for t in comps[1:])
    assert all(any('raw_key_cache' in n for n in t.shared_by) and any('compressed_key_cache' in n for n in t.shared_by) for t in comps)
    fixed = len(groups) - 1; assert fixed == 3
    lanes = {ctx: (nb - 1) // (fixed + math.ceil(ctx / 2768)) for ctx in (40000, 44000)}
    print(f'  pool {pool:,} B -> {nb} blocks; lanes 40K {lanes[40000]}, 44K {lanes[44000]}')
    if pool == PINNED: assert lanes == {40000: 43, 44000: 40}
    if nb == 780: assert lanes[44000] == 41
print(f'  41 lanes at 44K need 780 blocks = {780 * 21875712:,} B = {780 * 21875712 / 1024**3:.4f} GiB (pinned pool now {PINNED / 1024**3:.4f} GiB)')
nb, tensors = ku._get_kv_cache_config_csa_linear(cfg, groups, PINNED)
m.register_all_kvcache_specs(cfg)
sched = [g.kv_cache_spec.first_spec if isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs) else g.kv_cache_spec for g in groups]
managers = [KVCacheSpecRegistry.get_manager_class(s).__name__ for s in sched]; print('scheduler managers:', managers)
assert managers == ['FullAttentionManager', 'CircularBufferManager', 'MambaManager', 'MambaManager']
proj = ku._project_kv_cache_groups_to_worker(groups, specs); assert [g.layer_names for g in proj] == [g.layer_names for g in groups]
per_layer = {}
for g in proj:
    if isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs): per_layer.update(g.kv_cache_spec.kv_cache_specs)
    else: per_layer.update({n: g.kv_cache_spec for n in g.layer_names})
group_of = {n: i for i, g in enumerate(proj) for n in g.layer_names}
for t in tensors:
    pages = {per_layer[n].page_size_bytes for n in t.shared_by}
    assert len(pages) == 1 and next(iter(pages)) * nb == t.size, (t.shared_by, pages, t.size)
    owners = [group_of[n] for n in t.shared_by]
    assert len(owners) == len(set(owners)), ('owners sharing a tensor must hold different block ids (different groups); aliasing:', t.shared_by)
print('no tensor is shared by two owners of one group (no page aliasing)')
from vllm.v1.worker.mamba_utils import get_mamba_groups
kvc = KVCacheConfig(num_blocks=nb, kv_cache_tensors=tensors, kv_cache_groups=proj)
mg = get_mamba_groups(kvc); ids = sorted({i for v in mg.values() for i in v})
assert ids == [2, 3], ids
assert len(mg) == 2 and sorted(len(v) for v in mg.values()) == [1, 2], mg   # GDN-B and GDN-C share one spec (ids 2, 3); the PLE spec maps to group 2
print('worker Mamba discovery sees GDN groups', ids, 'with', len(mg), 'specs (GDN-A inside the ring-first group is invisible to it, as the PLE was under fold A; its callers are align/spec-decode paths, none at MTP0)')
ku._get_kv_cache_groups_csa_linear, ku._get_csa_linear_tensor_layout, ku._get_kv_cache_config_csa_linear = originals['groups'], originals['layout'], originals['config']
assert ku._get_kv_cache_config_csa_linear(cfg, ku._get_kv_cache_groups_csa_linear(cfg, specs), avail)[0] == 781
print('CPU FOLD B TEST OK')
