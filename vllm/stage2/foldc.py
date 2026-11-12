"""stage2_v1 'foldc' part (2026-10-04, overnight item d; Fable's candidate "PLE padding out of the compressed tensors", user
~06:00: adopt only if hazard free). Production's fixed-state fold B (dense_fp8_v1/fixed_state_fold_v1/fold.py, read-only)
puts the PLE state (one TP-replicated Mamba layer, 245,760 B) in compressed tensor 0, so every compressed-tensor owner
(13 compressed-key caches, 13 raw-key rings, the PLE) is padded to the PLE's size: 245,760 B against the compressed keys'
198,656 B, i.e. 13 x 47,104 B = 612,352 B of the 27,083,264 B block (2.26%).
Fold C keeps fold B's groups (rings with GDN-A, ring-first; the PLE with GDN-B, Mamba-first; 3 fixed blocks per lane) and
moves only the PLE's view: from compressed tensor 0 to the first main-KV tensor that no GDN layer shares (the 13th, the MTP
drafter's attention layer; the 36 GDN layers fill indices 0-11 of the three GDN groups), padded to the main-KV page as
fold A did. The compressed-tensor owners keep their own size (max of the compressed keys' and the rings' unpadded pages).
Owners sharing a tensor still sit in different groups (MTP main KV: attention group; PLE: GDN-B group), so no page is
aliased. Patch points: vllm.v1.core.kv_cache_utils._get_kv_cache_groups_csa_linear (fold B's result, re-padded) and
_get_kv_cache_config_csa_linear (fold B's arithmetic with the PLE listed under the main tensor); fold B's layout reader
(unfold + the stock recogniser) is used unchanged. Needs ARC3_FIXED_STATE_FOLD=B installed before this patch (asserted)."""
import os
from dataclasses import replace

KU = 'vllm.v1.core.kv_cache_utils'
PIN = 'f6643bad29b7b3cb044482b6736a042afa91a4f43a11f034317a8e8029d3a8a4'          # as fixed_state_fold_v1.import_hook
STATS = dict(installed=False, groups_calls=0, config_calls=0, compressed_page=None, main_page=None, ple_slot=None,
             bytes_per_block=None, num_blocks=None, errors=0, last_error=None)


def _roles(groups):
    from vllm.v1.kv_cache_interface import CircularBufferSpec, FullAttentionSpec, MambaSpec, UniformTypeKVCacheSpecs
    attn = ring = ple = None
    for g in groups:
        s = g.kv_cache_spec
        if isinstance(s, UniformTypeKVCacheSpecs):
            first = s.first_spec
            if type(first) is FullAttentionSpec:
                attn = g
            elif type(first) is CircularBufferSpec:
                ring = g
            elif isinstance(first, MambaSpec) and any(isinstance(x, MambaSpec) and x.tp_replicated for x in s.kv_cache_specs.values()):
                ple = g
    if attn is None or ring is None or ple is None:
        raise ValueError('foldc: expected fold B groups (attention, ring-first, Mamba-first with the PLE)')
    return attn, ring, ple


def refold(groups):
    """Fold B groups -> fold C groups: compressed keys and rings at their own page, the PLE at the main-KV page."""
    from vllm.v1.kv_cache_interface import (CircularBufferSpec, FullAttentionSpec, KVCacheGroupSpec, MambaSpec,
                                            MLAAttentionSpec, UniformTypeKVCacheSpecs)
    attn, ring, ple = _roles(groups)
    a = attn.kv_cache_spec.kv_cache_specs
    main = {s.page_size_bytes for s in a.values() if type(s) is FullAttentionSpec}
    if len(main) != 1:
        raise ValueError('foldc: main-KV pages differ: %s' % sorted(main))
    main_page = main.pop()
    comp_raw = max(replace(s, page_size_padded=None).page_size_bytes for s in a.values() if type(s) is MLAAttentionSpec)
    rings = {n: s for n, s in ring.kv_cache_spec.kv_cache_specs.items() if type(s) is CircularBufferSpec}
    page_c = max(comp_raw, max(s.unpadded_page_size_bytes for s in rings.values()))
    p = ple.kv_cache_spec.kv_cache_specs
    ple_names = [n for n, s in p.items() if isinstance(s, MambaSpec) and s.tp_replicated]
    if len(ple_names) != 1 or replace(p[ple_names[0]], page_size_padded=None).page_size_bytes > main_page:
        raise ValueError('foldc: one PLE owner no larger than a main-KV page expected')
    attn2 = KVCacheGroupSpec(list(attn.layer_names), UniformTypeKVCacheSpecs(
        block_size=attn.kv_cache_spec.block_size,
        kv_cache_specs={n: (replace(s, page_size_padded=page_c) if type(s) is MLAAttentionSpec else s) for n, s in a.items()}))
    ring2 = KVCacheGroupSpec(list(ring.layer_names), UniformTypeKVCacheSpecs(
        block_size=ring.kv_cache_spec.block_size,
        kv_cache_specs={n: (replace(s, page_size_padded=page_c) if type(s) is CircularBufferSpec else s)
                        for n, s in ring.kv_cache_spec.kv_cache_specs.items()}))
    ple2 = KVCacheGroupSpec(list(ple.layer_names), UniformTypeKVCacheSpecs(
        block_size=ple.kv_cache_spec.block_size,
        kv_cache_specs={n: (replace(s, page_size_padded=main_page) if n in ple_names else s) for n, s in p.items()}))
    for g in (attn2, ring2, ple2):
        assert all(s.max_num_blocks_per_req is not None for s in g.kv_cache_spec.kv_cache_specs.values())
    STATS.update(compressed_page=page_c, main_page=main_page)
    return [attn2 if g is attn else ring2 if g is ring else ple2 if g is ple else g for g in groups]


def config(ku, layout, vllm_config, available_memory):
    """Fold B's tensor arithmetic with the PLE listed under the first main-KV tensor that no GDN layer shares."""
    num_blocks = available_memory // layout.bytes_per_block
    num_blocks = ku.may_override_num_blocks(vllm_config, num_blocks)
    gdn = [g for g in layout.mamba_groups if not g.kv_cache_spec.tp_replicated]
    ple_names = [n for g in layout.mamba_groups if g.kv_cache_spec.tp_replicated for n in g.layer_names]
    if len(ple_names) != 1:
        raise ValueError('foldc: one PLE layer expected, got %s' % ple_names)
    free = [i for i in range(len(layout.main_kv_names)) if all(i >= len(g.layer_names) for g in gdn)]
    if not free:
        raise ValueError('foldc: every main-KV tensor has a GDN owner')
    slot = free[0]
    T = ku.KVCacheTensor
    tensors = [T(size=layout.main_kv_page_size * num_blocks,
                 shared_by=[m] + [g.layer_names[i] for g in gdn if i < len(g.layer_names)] + (ple_names if i == slot else []))
               for i, m in enumerate(layout.main_kv_names)]
    tensors.extend(T(size=layout.compressed_page_size * num_blocks, shared_by=[c, layout.compressor_state_names[i]])
                   for i, c in enumerate(layout.compressed_names))
    STATS.update(ple_slot=slot, bytes_per_block=layout.bytes_per_block, num_blocks=num_blocks)
    return num_blocks, tensors


def patch_kv_cache_utils(ku):
    if os.environ.get('ARC3_FIXED_STATE_FOLD') != 'B' or ku._get_kv_cache_groups_csa_linear.__name__ != 'groups_csa_linear':
        raise RuntimeError('foldc: needs production fold B installed on kv_cache_utils first (ARC3_FIXED_STATE_FOLD=B)')
    groups_b, layout_b = ku._get_kv_cache_groups_csa_linear, ku._get_csa_linear_tensor_layout

    def groups_csa_linear_c(vllm_config, kv_cache_spec):
        groups = groups_b(vllm_config, kv_cache_spec)
        if groups is None:
            return None
        STATS['groups_calls'] += 1
        return refold(groups)

    def config_csa_linear_c(vllm_config, kv_cache_groups, available_memory):
        layout = layout_b(kv_cache_groups)
        if layout is None:
            return None
        STATS['config_calls'] += 1
        return config(ku, layout, vllm_config, available_memory)

    ku._get_kv_cache_groups_csa_linear = groups_csa_linear_c
    ku._get_kv_cache_config_csa_linear = config_csa_linear_c
    STATS['installed'] = True
    try:
        from . import install
        install.REPORT['foldc'] = STATS
    except Exception:                                    # noqa: BLE001  receipts are diagnostics
        pass


PATCHERS = {KU: patch_kv_cache_utils}
PINS = {KU: PIN}
