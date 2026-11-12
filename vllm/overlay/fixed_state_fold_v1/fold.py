"""Fixed-state folds.

Option A fold: the PLE state (one TP-replicated Mamba layer, padded to a main-KV page) joins the raw-key ring
group (12 CircularBufferSpec layers, padded to compressed pages), so both claim ONE block per request instead of two.
Tensor layout is unchanged (rings in the compressed tensors, PLE in main tensor 0); only block ids are shared.

Ring-first ordering keeps the scheduler manager = CircularBufferManager (one block per request, claimed at admission,
freed at the end — the same lifetime MambaManager mode 'none' gives a one-block group) and keeps the PLE invisible to
the worker's Mamba-group discovery, whose callers are all speculative/align paths that do not run at MTP0.

Patch points (module-level functions of vllm.v1.core.kv_cache_utils, looked up by global name at call time):
  _get_kv_cache_groups_csa_linear   -> original, then fold
  _get_csa_linear_tensor_layout     -> recognise the merged group (rings for the compressed slots, PLE as an extra
                                       main-tensor-0 owner)
  _get_kv_cache_config_csa_linear   -> original layout arithmetic + the extra owner on main tensor 0

Option B fold (5 -> 3 fixed blocks per lane): the compressed-key page grows from 177,152 to 184,320 B (the PLE
state's size; +7,168 B per page, +86,016 B per block), the 12 raw-key rings join GDN group A (ring-first: CircularBufferManager, one block per
request for the lifetime, as fold A) and the PLE joins GDN group B (Mamba-first: MambaManager); the PLE view moves
from main tensor 0 to compressed tensor 0. Rings and PLE must sit in different groups because they now share a
tensor: owners sharing a tensor need distinct block ids. Every compressed-tensor owner (compressed keys, ring, PLE) is
padded to the same 184,320 B page, so all three views share one block stride; the compressed-key kernels address
blocks through the tensor strides (qsa_pre_indexer / qsa select pass stride(0)), the ring already used a padded
strided view, and the worker builds the PLE page view from the tensor it is listed under. The ring and PLE groups
disappear (6 -> 4 groups). Known limitation (shared with fold A): the worker's Mamba-group discovery
(`get_mamba_groups`) only looks at Mamba-first groups, so the GDN-A layers inside the ring-first group are invisible
to it; its callers are the speculative/align paths only, which do not run at MTP0 with mamba_cache_mode
'none'.
"""
from dataclasses import dataclass, replace


def _classes(ku):
    from vllm.v1.kv_cache_interface import CircularBufferSpec, KVCacheGroupSpec, MambaSpec, UniformTypeKVCacheSpecs
    return CircularBufferSpec, KVCacheGroupSpec, MambaSpec, UniformTypeKVCacheSpecs


def fold_ring_ple(ku, groups):
    CircularBufferSpec, KVCacheGroupSpec, MambaSpec, UniformTypeKVCacheSpecs = _classes(ku)
    ring = [g for g in groups if isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs) and type(g.kv_cache_spec.first_spec) is CircularBufferSpec]
    ple = [g for g in groups if isinstance(g.kv_cache_spec, MambaSpec) and g.kv_cache_spec.tp_replicated]
    if len(ring) != 1 or len(ple) != 1:
        raise ValueError(f'fold_ring_ple expects one ring group and one replicated mamba group, got {len(ring)}/{len(ple)}')
    ring, ple = ring[0], ple[0]
    if len(ple.layer_names) != 1:
        raise ValueError('fold_ring_ple expects a single PLE layer')
    ring_page = ring.kv_cache_spec.first_spec.page_size_bytes
    specs = dict(ring.kv_cache_spec.kv_cache_specs)
    for name in ple.layer_names:
        specs[name] = ple.kv_cache_spec                      # already padded to the main-KV page by the CSA grouping
    assert all(spec.max_num_blocks_per_req is not None for spec in specs.values())
    merged = KVCacheGroupSpec(list(ring.layer_names) + list(ple.layer_names),
                              UniformTypeKVCacheSpecs(block_size=ring.kv_cache_spec.block_size, kv_cache_specs=specs))
    out = [merged if g is ring else g for g in groups if g is not ple]
    return out


def split_merged(ku, group):
    """(ring specs, extra mamba owners) of a possibly-merged compressor-state group."""
    CircularBufferSpec, _, MambaSpec, UniformTypeKVCacheSpecs = _classes(ku)
    spec = group.kv_cache_spec
    rings = {n: s for n, s in spec.kv_cache_specs.items() if type(s) is CircularBufferSpec}
    extra = {n: s for n, s in spec.kv_cache_specs.items() if isinstance(s, MambaSpec)}
    assert len(rings) + len(extra) == len(spec.kv_cache_specs)
    return rings, extra


def _roles_b(ku, groups):
    CircularBufferSpec, KVCacheGroupSpec, MambaSpec, UniformTypeKVCacheSpecs = _classes(ku)
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MLAAttentionSpec
    attn = [g for g in groups if isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs) and type(g.kv_cache_spec.first_spec) is FullAttentionSpec]
    ring = [g for g in groups if isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs) and type(g.kv_cache_spec.first_spec) is CircularBufferSpec]
    ple = [g for g in groups if isinstance(g.kv_cache_spec, MambaSpec) and g.kv_cache_spec.tp_replicated]
    gdn = [g for g in groups if isinstance(g.kv_cache_spec, MambaSpec) and not g.kv_cache_spec.tp_replicated]
    if len(attn) != 1 or len(ring) != 1 or len(ple) != 1 or len(gdn) < 2 or len(ple[0].layer_names) != 1:
        raise ValueError(f'fold B expects attention/ring/PLE groups and >= 2 GDN groups, got {len(attn)}/{len(ring)}/{len(ple)}/{len(gdn)}')
    if len(groups) != 3 + len(gdn):
        raise ValueError('fold B expects exactly attention + ring + GDN groups + PLE')
    return attn[0], ring[0], gdn, ple[0], MLAAttentionSpec


def fold_option_b(ku, groups):
    CircularBufferSpec, KVCacheGroupSpec, MambaSpec, UniformTypeKVCacheSpecs = _classes(ku)
    attn, ring, gdn, ple, MLAAttentionSpec = _roles_b(ku, groups)
    compressed = {n: s for n, s in attn.kv_cache_spec.kv_cache_specs.items() if type(s) is MLAAttentionSpec}
    if not compressed or any(s.page_size_padded is not None for s in compressed.values()):
        raise ValueError('fold B expects unpadded compressed-key specs')
    ple_unpadded = replace(ple.kv_cache_spec, page_size_padded=None).page_size_bytes
    ring_unpadded = max(s.unpadded_page_size_bytes for s in ring.kv_cache_spec.kv_cache_specs.values())
    old_page = next(iter(compressed.values())).page_size_bytes
    new_page = max(old_page, ple_unpadded, ring_unpadded)
    attn_specs = {n: (replace(s, page_size_padded=new_page) if type(s) is MLAAttentionSpec else s) for n, s in attn.kv_cache_spec.kv_cache_specs.items()}
    attn2 = KVCacheGroupSpec(list(attn.layer_names), UniformTypeKVCacheSpecs(block_size=attn.kv_cache_spec.block_size, kv_cache_specs=attn_specs))
    rings = {n: replace(s, page_size_padded=new_page) for n, s in ring.kv_cache_spec.kv_cache_specs.items()}
    ple_spec = replace(ple.kv_cache_spec, page_size_padded=new_page)
    a, b = gdn[0], gdn[1]
    # Two merged groups, because the rings and the PLE now live in the SAME tensor (compressed tensor 0 holds
    # compressed key 0, ring 0 and the PLE): owners that share a tensor must hold different block ids, i.e. sit in
    # different groups (a single ring+PLE group aliased ring 0 and the PLE on one page: NLL 0.33972 vs 0.34616 and a
    # MoE kernel crash, 2026-09-20 21:26 UTC).
    #  - rings + GDN-A, ring-first: the group's kernel block size and slot-mapping policy follow its first member
    #    (ring: kernel size 4, no slot mapping; the GDN backend accepts any size), scheduler manager
    #    CircularBufferManager (one block per request for the lifetime), the zeroer skips the Mamba members and the
    #    GDN layers take their state index from the group's block table. GDN-first would hand the rings the GDN
    #    'kernel block size' (max_model_len) and trip KVBlockZeroer's divisibility assert (observed 21:02 UTC).
    #  - GDN-B + PLE, Mamba-first: every member is a MambaSpec, so MambaManager, no attention member for the zeroer,
    #    and the worker's Mamba discovery accepts it (two specs, one group id).
    a2 = KVCacheGroupSpec(list(ring.layer_names) + list(a.layer_names),
                          UniformTypeKVCacheSpecs(block_size=ring.kv_cache_spec.block_size,
                                                  kv_cache_specs={**rings, **{n: a.kv_cache_spec for n in a.layer_names}}))
    b2 = KVCacheGroupSpec(list(b.layer_names) + list(ple.layer_names),
                          UniformTypeKVCacheSpecs(block_size=b.kv_cache_spec.block_size,
                                                  kv_cache_specs={**{n: b.kv_cache_spec for n in b.layer_names}, ple.layer_names[0]: ple_spec}))
    for merged in (a2, b2):
        assert all(spec.max_num_blocks_per_req is not None for spec in merged.kv_cache_spec.kv_cache_specs.values())
    out = []
    for g in groups:
        if g is attn: out.append(attn2)
        elif g is a: out.append(a2)
        elif g is b: out.append(b2)
        elif g is ring or g is ple: continue
        else: out.append(g)
    return out


def unfold_b(ku, kv_cache_groups):
    """Rebuild the six-group view the original layout recogniser expects: the ring-first group splits into the ring
    group and its GDN group (in place); the Mamba-first group with the PLE splits into its GDN group (in place) and a
    synthetic one-layer PLE group after the real Mamba groups. Page sizes (already padded) are kept."""
    CircularBufferSpec, KVCacheGroupSpec, MambaSpec, UniformTypeKVCacheSpecs = _classes(ku)
    unfolded = []; ple_items = []
    for g in kv_cache_groups:
        spec = g.kv_cache_spec
        if not isinstance(spec, UniformTypeKVCacheSpecs) or not any(isinstance(s, MambaSpec) for s in spec.kv_cache_specs.values()):
            unfolded.append(g); continue
        rings = {n: s for n, s in spec.kv_cache_specs.items() if type(s) is CircularBufferSpec}
        gdn_names = [n for n in g.layer_names if isinstance(spec.kv_cache_specs[n], MambaSpec) and not spec.kv_cache_specs[n].tp_replicated]
        ple_items += [(n, spec.kv_cache_specs[n]) for n in g.layer_names if isinstance(spec.kv_cache_specs[n], MambaSpec) and spec.kv_cache_specs[n].tp_replicated]
        if rings:
            unfolded.append(KVCacheGroupSpec(list(rings), UniformTypeKVCacheSpecs(block_size=spec.block_size, kv_cache_specs=rings)))
        if gdn_names:
            unfolded.append(KVCacheGroupSpec(gdn_names, spec.kv_cache_specs[gdn_names[0]]))
    for n, sp in ple_items:
        unfolded.append(KVCacheGroupSpec([n], sp))
    return unfolded

def summarize(groups):
    from vllm.v1.kv_cache_interface import UniformTypeKVCacheSpecs
    return [dict(layers=len(g.layer_names), spec=('Uniform[' + type(g.kv_cache_spec.first_spec).__name__ + ']' if isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs) else type(g.kv_cache_spec).__name__),
                 page_size_bytes=g.kv_cache_spec.page_size_bytes) for g in groups]


def install(ku, mode, on_fold=None):
    """mode: 'A' (ring + PLE share one block), 'B' (rings + PLE into the GDN blocks; compressed page 184,320 B) or 'off'.
    Wraps the three module-level functions of kv_cache_utils."""
    if mode == 'off':
        return {}
    assert mode in ('A', 'B'), mode
    CircularBufferSpec, KVCacheGroupSpec, MambaSpec, UniformTypeKVCacheSpecs = _classes(ku)
    orig_groups, orig_layout, orig_config = ku._get_kv_cache_groups_csa_linear, ku._get_csa_linear_tensor_layout, ku._get_kv_cache_config_csa_linear
    Layout = ku._CSALinearTensorLayout
    KVCacheTensor = ku.KVCacheTensor

    def groups_csa_linear(vllm_config, kv_cache_spec):
        groups = orig_groups(vllm_config, kv_cache_spec)
        if groups is None:
            return None
        folded = fold_ring_ple(ku, groups) if mode == 'A' else fold_option_b(ku, groups)
        if on_fold is not None:
            on_fold(dict(before=summarize(groups), after=summarize(folded)))
        return folded

    def layout_csa_linear(kv_cache_groups):
        if mode == 'B':
            return orig_layout(unfold_b(ku, kv_cache_groups))
        # Rebuild an unfolded view for the original recogniser: the merged group becomes the ring group, and the
        # PLE member becomes a synthetic one-layer Mamba group listed after the real mamba groups.
        unfolded = []; extra_all = {}
        for g in kv_cache_groups:
            if isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs) and type(g.kv_cache_spec.first_spec) is CircularBufferSpec:
                rings, extra = split_merged(ku, g)
                unfolded.append(KVCacheGroupSpec([n for n in g.layer_names if n in rings], UniformTypeKVCacheSpecs(block_size=g.kv_cache_spec.block_size, kv_cache_specs=rings)))
                extra_all.update(extra)
            else:
                unfolded.append(g)
        for name, spec in extra_all.items():
            unfolded.append(KVCacheGroupSpec([name], spec))
        return orig_layout(unfolded)

    def config_csa_linear(vllm_config, kv_cache_groups, available_memory):
        layout = layout_csa_linear(kv_cache_groups)
        if layout is None:
            return None
        num_blocks = available_memory // layout.bytes_per_block
        num_blocks = ku.may_override_num_blocks(vllm_config, num_blocks)
        if mode == 'B':
            # The TP-replicated PLE owner lives in compressed tensor 0 (its page now equals the compressed page); every
            # other mamba group keeps one layer per main tensor as before.
            main_groups = [g for g in layout.mamba_groups if not g.kv_cache_spec.tp_replicated]
            ple_names = [n for g in layout.mamba_groups if g.kv_cache_spec.tp_replicated for n in g.layer_names]
            assert len(ple_names) == 1 and layout.compressed_page_size >= replace(layout.mamba_groups[-1].kv_cache_spec, page_size_padded=None).page_size_bytes
        else:
            main_groups, ple_names = layout.mamba_groups, []
        tensors = [KVCacheTensor(size=layout.main_kv_page_size * num_blocks,
                                 shared_by=[main] + [group.layer_names[i] for group in main_groups if i < len(group.layer_names)])
                   for i, main in enumerate(layout.main_kv_names)]
        tensors.extend(KVCacheTensor(size=layout.compressed_page_size * num_blocks, shared_by=[c, layout.compressor_state_names[i]] + (ple_names if i == 0 else []))
                       for i, c in enumerate(layout.compressed_names))
        return num_blocks, tensors

    ku._get_kv_cache_groups_csa_linear = groups_csa_linear
    ku._get_csa_linear_tensor_layout = layout_csa_linear
    ku._get_kv_cache_config_csa_linear = config_csa_linear
    return dict(groups=orig_groups, layout=orig_layout, config=orig_config)
