"""stage2_v1 'draftring' part (2026-10-15, item 4b; design: harnesspod_20261013/draftring/DRAFTRING_DESIGN_2026-10-15.md): the MTP
drafter's KV cache and the PLE state leave the shared KV pool.

Section 1 (this file, core and worker processes): fold D, the KV grouping without the drafter and the PLE.
Today (fold B + foldc) the pool cuts one (main, compressed) tensor pair per CSA layer: 12 target QSA layers + the drafter's
mtp.layers.48. Block = 13 × (1,837,568 + 198,656) = 26,470,912 B. The drafter's pair (slot 12) also holds the PLE state (main
tensor 12 of GDN-B blocks, foldc) and the drafter's raw-key ring (compressed tensor 12 of the ring + GDN-A blocks).

Fold D takes vLLM's stock grouping / layout / config functions from production's fold B wrappers (their closures; fold B's
`_roles_b` requires a PLE group, so it cannot be called without one). It drops the drafter's three owners (`mtp.` names) and the
TP-replicated PLE from the spec map before the stock grouping: attention (12 main + 12 compressed), ring (12), GDN 3 × 12 (the
same round-robin dealing as today: cdiv(36, 12) = cdiv(36, 13) = 3). It then merges the rings into GDN group A ring-first, as
fold B does but without the PLE: rings and GDN-A share one block per request, so a lane keeps 3 fixed blocks. The compressed
page is not grown (no PLE): max(compressed, ring unpadded) = 198,656 B, as foldc. Layout and config are the stock functions on
the unfolded view (rings separate): main tensor i shared by main_kv_i and GDN layer i of the three GDN groups, compressed tensor
i by compressed_i and ring_i. Block = 12 × (1,837,568 + 198,656) = 24,434,688 B.
The removed owners (EXTRA_OWNERS) are placed by the worker into private extra groups (section 2); the scheduler, the eviction
plan and the pool never see them. Requires ARC3_FIXED_STATE_FOLD=B (fold B's wrappers present) and excludes foldc."""
import os
from dataclasses import replace

KU = 'vllm.v1.core.kv_cache_utils'
PIN = 'f6643bad29b7b3cb044482b6736a042afa91a4f43a11f034317a8e8029d3a8a4'          # as fixed_state_fold_v1.import_hook / foldc
PINS = {KU: PIN}
DRAFTER_PREFIX = 'mtp.'
STATS = dict(fold_installed=False, groups_calls=0, config_calls=0, extra_owners=[], groups=None, bytes_per_block=None,
             num_blocks=None, errors=0, last_error=None)
EXTRA_OWNERS = {}               # name -> spec removed from the pool (filled by the grouping; the worker reads it)


def _stock(fn, name):
    """vLLM's own module function captured in fold B's wrapper closure."""
    for cell in (getattr(fn, '__closure__', None) or ()):
        v = cell.cell_contents
        if callable(v) and getattr(v, '__name__', '') == name and getattr(v, '__module__', '') == KU:
            return v
    raise RuntimeError(f'draftring: stock {name} not found in fold B wrapper (is ARC3_FIXED_STATE_FOLD=B installed?)')


def split_specs(kv_cache_spec):
    """(pool specs, extra specs): the drafter's owners and the TP-replicated PLE leave the pool."""
    from vllm.v1.kv_cache_interface import MambaSpec
    pool, extra = {}, {}
    for name, spec in kv_cache_spec.items():
        if name.startswith(DRAFTER_PREFIX) or (isinstance(spec, MambaSpec) and spec.tp_replicated):
            extra[name] = spec
        else:
            pool[name] = spec
    return pool, extra


def merge_rings(groups):
    """Stock groups [attention, ring, GDN-A, GDN-B, GDN-C] -> [attention, rings + GDN-A (ring-first), GDN-B, GDN-C]."""
    from vllm.v1.kv_cache_interface import (CircularBufferSpec, FullAttentionSpec, KVCacheGroupSpec, MambaSpec, MLAAttentionSpec,
                                            UniformTypeKVCacheSpecs)
    attn = [g for g in groups if isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs) and type(g.kv_cache_spec.first_spec) is FullAttentionSpec]
    ring = [g for g in groups if isinstance(g.kv_cache_spec, UniformTypeKVCacheSpecs) and type(g.kv_cache_spec.first_spec) is CircularBufferSpec]
    gdn = [g for g in groups if isinstance(g.kv_cache_spec, MambaSpec) and not g.kv_cache_spec.tp_replicated]
    if len(attn) != 1 or len(ring) != 1 or len(gdn) < 1 or len(groups) != 2 + len(gdn):
        raise ValueError(f'draftring fold D expects attention/ring/GDN groups only, got {[type(g.kv_cache_spec).__name__ for g in groups]}')
    attn, ring = attn[0], ring[0]
    compressed = [s for s in attn.kv_cache_spec.kv_cache_specs.values() if type(s) is MLAAttentionSpec]
    page = compressed[0].page_size_bytes
    if any(s.page_size_bytes != page for s in compressed) or any(s.page_size_bytes != page for s in ring.kv_cache_spec.kv_cache_specs.values()):
        raise ValueError('draftring fold D: rings must already sit at the compressed page (stock CSA padding)')
    a = gdn[0]
    merged = KVCacheGroupSpec(list(ring.layer_names) + list(a.layer_names),
                              UniformTypeKVCacheSpecs(block_size=ring.kv_cache_spec.block_size,
                                                      kv_cache_specs={**ring.kv_cache_spec.kv_cache_specs,
                                                                      **{n: a.kv_cache_spec for n in a.layer_names}}))
    assert all(spec.max_num_blocks_per_req is not None for spec in merged.kv_cache_spec.kv_cache_specs.values())
    return [attn, merged] + gdn[1:]


def unfold(groups):
    """The stock recogniser's view: the ring-first merged group splits into the ring group and its GDN group (in place)."""
    from vllm.v1.kv_cache_interface import CircularBufferSpec, KVCacheGroupSpec, MambaSpec, UniformTypeKVCacheSpecs
    out = []
    for g in groups:
        spec = g.kv_cache_spec
        if isinstance(spec, UniformTypeKVCacheSpecs) and type(spec.first_spec) is CircularBufferSpec:
            rings = {n: s for n, s in spec.kv_cache_specs.items() if type(s) is CircularBufferSpec}
            gdn = [n for n in g.layer_names if isinstance(spec.kv_cache_specs[n], MambaSpec)]
            out.append(KVCacheGroupSpec(list(rings), UniformTypeKVCacheSpecs(block_size=spec.block_size, kv_cache_specs=rings)))
            out.append(KVCacheGroupSpec(gdn, spec.kv_cache_specs[gdn[0]]))
        else:
            out.append(g)
    return out


def install_fold(ku):
    """Wrap fold B's three wrappers with fold D (stock functions underneath)."""
    stock_groups = _stock(ku._get_kv_cache_groups_csa_linear, '_get_kv_cache_groups_csa_linear')
    stock_layout = _stock(ku._get_csa_linear_tensor_layout, '_get_csa_linear_tensor_layout')

    def stock_config(vllm_config, kv_cache_groups, available_memory):
        """kv_cache_utils._get_kv_cache_config_csa_linear of the pinned file (f6643bad, lines 1335-1365), verbatim: fold B
        replaces it with its own arithmetic and keeps no reference to it."""
        layout = stock_layout(kv_cache_groups)
        if layout is None:
            return None
        num_blocks = available_memory // layout.bytes_per_block
        num_blocks = ku.may_override_num_blocks(vllm_config, num_blocks)
        kv_cache_tensors = [
            ku.KVCacheTensor(size=layout.main_kv_page_size * num_blocks,
                             shared_by=[main_kv_name] + [group.layer_names[index] for group in layout.mamba_groups
                                                         if index < len(group.layer_names)])
            for index, main_kv_name in enumerate(layout.main_kv_names)
        ]
        kv_cache_tensors.extend(
            ku.KVCacheTensor(size=layout.compressed_page_size * num_blocks,
                             shared_by=[compressed_name, layout.compressor_state_names[index]])
            for index, compressed_name in enumerate(layout.compressed_names)
        )
        return num_blocks, kv_cache_tensors

    def groups_csa_linear(vllm_config, kv_cache_spec):
        pool, extra = split_specs(kv_cache_spec)
        groups = stock_groups(vllm_config, pool)
        if groups is None:
            return None
        STATS['groups_calls'] += 1
        EXTRA_OWNERS.clear()
        EXTRA_OWNERS.update(extra)
        STATS['extra_owners'] = sorted(extra)
        folded = merge_rings(groups)
        STATS['groups'] = [(type(g.kv_cache_spec).__name__, len(g.layer_names)) for g in folded]
        return folded

    def layout_csa_linear(kv_cache_groups):
        return stock_layout(unfold(kv_cache_groups))

    def config_csa_linear(vllm_config, kv_cache_groups, available_memory):
        unfolded = unfold(kv_cache_groups)
        layout = stock_layout(unfolded)
        if layout is None:
            return None
        STATS['config_calls'] += 1
        result = stock_config(vllm_config, unfolded, available_memory)
        STATS['bytes_per_block'] = layout.bytes_per_block
        STATS['num_blocks'] = result[0] if result else None
        return result

    ku._get_kv_cache_groups_csa_linear = groups_csa_linear
    ku._get_csa_linear_tensor_layout = layout_csa_linear
    ku._get_kv_cache_config_csa_linear = config_csa_linear
    STATS['fold_installed'] = True


def patch_kv_cache_utils(mod):
    if os.environ.get('ARC3_FIXED_STATE_FOLD') != 'B':
        raise RuntimeError('draftring: needs ARC3_FIXED_STATE_FOLD=B (fold B wrappers hold the stock functions)')
    install_fold(mod)


PATCHERS = {KU: patch_kv_cache_utils}


# ------------------------------------------------------------------------------------------------ section 2: worker extras
# (IE review 2026-10-15 ~20:55 applied: the worker reports pool specs only (the core asserts every reported layer is grouped);
# extras come from the worker's own full map; KVBlockZeroer excludes the drafter owners; static rows with session ids >= 1
# (page 0 / id 0 = null); extras appended to the config the runner keeps; the commit hook is installed in the worker only.)
MR = 'vllm.v1.worker.gpu.model_runner'
PINS[MR] = 'f08ba0f5ad41c8b9d5145fb1fba115ffd97f35f1e59017602e235ec0823abce8'
P = int(os.environ.get('ARC3_S2_DRAFTRING_P', '8') or 8)            # ring pages per session (24,832 rows at 3,104)
STATS.update(worker_extra_owners=[], extra_groups=None, extra_tensor_bytes=None, sessions_max=None, sessions_assigned=0,
             sessions_freed=0, sessions_live_max=0, zeroed_pages=0, rows_attn=0, cache_shapes=None, renumbers=0,
             renumber_rows=0, renumber_stale_rows=0, renumber_ms=0.0, renumber_errors=0, last_renumber_error=None,
             dummy_context_guards=0, persist_rows=0)
_W = dict(owners=None, caches={}, groups=None, n_pool=None, by_req={}, free=[], runner=None, width=None)


def extra_roles(extra):
    from vllm.v1.kv_cache_interface import CircularBufferSpec, FullAttentionSpec, MambaSpec, MLAAttentionSpec
    main = [n for n, s in extra.items() if type(s) is FullAttentionSpec]
    comp = [n for n, s in extra.items() if type(s) is MLAAttentionSpec]
    ring = [n for n, s in extra.items() if type(s) is CircularBufferSpec]
    ple = [n for n, s in extra.items() if isinstance(s, MambaSpec) and s.tp_replicated]
    if not (len(main) == len(comp) == len(ring) == len(ple) == 1):
        raise RuntimeError(f'draftring: unexpected extra owners {sorted(extra)}')
    return main[0], comp[0], ring[0], ple[0]


def extend_config(cfg, extra, sessions, pages):
    """A shallow copy of the core's KVCacheConfig with the three extra groups and their four tensors appended."""
    import copy
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec, KVCacheTensor, UniformTypeKVCacheSpecs
    main, comp, ring, ple = extra_roles(extra)
    attn_spec = UniformTypeKVCacheSpecs.from_specs({main: extra[main], comp: extra[comp]})
    if attn_spec is None:
        raise RuntimeError('draftring: drafter main / compressed specs are not uniform-compatible')
    ring_spec = replace(extra[ring], page_size_padded=extra[comp].page_size_bytes)     # padded as the stock CSA pool pads rings
    ple_spec = replace(extra[ple], page_size_padded=None)                              # unpadded (IE): 245,760-B pages
    n_attn = sessions * pages
    out = copy.copy(cfg)
    out.kv_cache_groups = list(cfg.kv_cache_groups) + [KVCacheGroupSpec([main, comp], attn_spec),
                                                       KVCacheGroupSpec([ring], ring_spec), KVCacheGroupSpec([ple], ple_spec)]
    tensors = [KVCacheTensor(size=extra[main].page_size_bytes * n_attn, shared_by=[main]),
               KVCacheTensor(size=extra[comp].page_size_bytes * n_attn, shared_by=[comp]),
               KVCacheTensor(size=ring_spec.page_size_bytes * sessions, shared_by=[ring]),
               KVCacheTensor(size=ple_spec.page_size_bytes * sessions, shared_by=[ple])]
    out.kv_cache_tensors = list(cfg.kv_cache_tensors) + tensors
    return out, (main, comp, ring, ple), sum(t.size for t in tensors)


def static_row(kind, sid, width, pages=None):
    """The block-table row of session sid in an extra group: the ring's logical page j -> sid * P + j mod P; one page otherwise."""
    pages = P if pages is None else pages
    if kind == 'attn':
        return [sid * pages + (j % pages) for j in range(width)]
    return [sid]


def _first_dim_views(obj):
    import torch
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return [t for x in obj for t in _first_dim_views(x)]
    return []


def _zero_session(sid):
    import torch
    main, comp, ring, ple = _W['owners']
    with torch.inference_mode():                               # the caches are inference tensors (dr2, 10-15)
        for name, lo, hi in ((main, sid * P, (sid + 1) * P), (comp, sid * P, (sid + 1) * P), (ring, sid, sid + 1), (ple, sid, sid + 1)):
            for t in _first_dim_views(_W['caches'].get(name)):
                t[lo:hi].zero_()
                STATS['zeroed_pages'] += hi - lo


def session_id(req_id):
    s = _W['by_req'].get(req_id)
    if s is None:
        if not _W['free']:
            raise RuntimeError('draftring: no free session id')
        s = _W['free'].pop(0)
        _W['by_req'][req_id] = s
        _zero_session(s)
        if SHADOW:
            _sh_assign(req_id)
        STATS['sessions_assigned'] += 1
        STATS['sessions_live_max'] = max(STATS['sessions_live_max'], len(_W['by_req']))
    return s


def free_session(req_id):
    s = _W['by_req'].pop(req_id, None)
    if SHADOW:
        _sh_release(req_id)
    if s is not None:
        _W['free'].append(s)
        _W['free'].sort()
        STATS['sessions_freed'] += 1


def _wrap_block_tables(runner):
    bt = runner.block_tables
    original = bt.append_block_ids
    n_pool, groups, width = _W['n_pool'], _W['groups'], _W['width']

    def append_block_ids(req_index, new_block_ids, overwrite):
        ids = tuple(new_block_ids)
        if len(ids) == bt.num_kv_cache_groups:                  # MRv2's warmup_kernels allocates ids for every group of the
            ids = ids[:n_pool]                                  # runner's config (dr1, 10-15): the extra groups take static rows
            STATS['full_tuples'] = STATS.get('full_tuples', 0) + 1
        if len(ids) != n_pool:
            raise RuntimeError(f'draftring: {len(ids)} block-id groups for {n_pool} pool groups')
        if overwrite:
            sid = session_id(runner.req_states.index_to_req_id[req_index])
            rows = (static_row('attn', sid, width), static_row('ring', sid, 1), static_row('ple', sid, 1))
            STATS['rows_attn'] += 1
        else:
            rows = ([], [], [])
        return original(req_index, ids + rows, overwrite)

    bt.append_block_ids = append_block_ids


def patch_model_runner(mod):
    cls = mod.GPUModelRunner
    orig_spec, orig_init, orig_zero = cls.get_kv_cache_spec, cls.initialize_kv_cache, cls._init_kv_zero_meta
    orig_finish = cls.finish_requests
    orig_init_kv = mod.init_kv_cache

    def get_kv_cache_spec(self):
        pool, extra = split_specs(orig_spec(self))
        self._s2_draftring_extra = extra
        STATS['worker_extra_owners'] = sorted(extra)
        return pool

    def init_kv_cache(*args, **kwargs):                        # model_runner's global name: capture the extra owners' views
        caches = orig_init_kv(*args, **kwargs)
        if _W['owners']:
            _W['caches'] = {n: caches[n] for n in _W['owners'] if n in caches}
            STATS['cache_shapes'] = {n: [list(t.shape) for t in _first_dim_views(c)] for n, c in _W['caches'].items()}
        return caches

    def initialize_kv_cache(self, kv_cache_config):
        if getattr(self.cache_config, 'mamba_cache_mode', 'none') == 'align':
            raise RuntimeError('draftring: mamba_cache_mode align reads every Mamba group (the PLE extra group has no pool blocks)')
        extra = getattr(self, '_s2_draftring_extra', None)
        if not extra:
            _, extra = split_specs(orig_spec(self))
        sessions = int(self.max_num_reqs) + 1                  # id 0 = null
        cfg, owners, nbytes = extend_config(kv_cache_config, extra, sessions, P)
        _W.update(owners=owners, n_pool=len(kv_cache_config.kv_cache_groups), runner=self,
                  free=list(range(1, sessions)), by_req={})
        _W['groups'] = list(range(_W['n_pool'], _W['n_pool'] + 3))
        STATS.update(extra_groups=_W['groups'], extra_tensor_bytes=nbytes, sessions_max=sessions - 1)
        result = orig_init(self, cfg)                          # deep-copies cfg into self.kv_cache_config (rollback / persist read it)
        bpk = [int(self.block_tables.blocks_per_kv_block[g]) for g in _W['groups']]
        if bpk != [1, 1, 1]:                                   # static rows and the ring arithmetic are in KV pages
            raise RuntimeError(f'draftring: extra groups need kernel block == KV block, blocks_per_kv_block {bpk}')
        _W['width'] = int(self.block_tables.input_block_tables[_W['groups'][0]].shape[1])
        STATS['width'] = _W['width']
        _wrap_block_tables(self)
        _install_commit_hook()
        return result

    def _init_kv_zero_meta(self):
        from vllm.v1.worker.utils import KVBlockZeroer
        main, comp, ring, _ = _W['owners']
        self.kv_block_zeroer = KVBlockZeroer(
            self.device,
            attn_groups_iter=(g for groups in self.attn_groups for g in groups),
            kernel_block_sizes=self.kernel_block_sizes,
            cache_dtype=self.cache_config.cache_dtype,
            static_forward_context=self.compilation_config.static_forward_context,
            runner_only_attn_layers={main, comp, ring},
        )

    def finish_requests(self, scheduler_output):
        for rid in set(scheduler_output.finished_req_ids or ()) | set(scheduler_output.preempted_req_ids or ()):
            free_session(rid)
        return orig_finish(self, scheduler_output)

    orig_dummy_ctx = mod.set_dummy_context

    def set_dummy_context(input_batch, block_tables, context_len, num_kv_blocks, max_model_len):
        """Stock fills every group's table with pool block ids (mod num_kv_blocks), out of range for the extra tensors:
        the extra groups' rows are reset to the null page afterwards (IE pre-launch item 4)."""
        result = orig_dummy_ctx(input_batch, block_tables, context_len, num_kv_blocks, max_model_len)
        if _W['groups']:
            for g in _W['groups']:
                block_tables.input_block_tables[g][:input_batch.num_reqs].zero_()
            STATS['dummy_context_guards'] += 1
        return result

    get_kv_cache_spec.__wrapped__ = orig_spec
    initialize_kv_cache.__wrapped__ = orig_init
    cls.get_kv_cache_spec = get_kv_cache_spec
    cls.initialize_kv_cache = initialize_kv_cache
    cls._init_kv_zero_meta = _init_kv_zero_meta
    cls.finish_requests = finish_requests
    mod.init_kv_cache = init_kv_cache
    mod.set_dummy_context = set_dummy_context


def group_block_ids(req_id, block_ids):
    """The scheduler's per-group block ids of an existing request, extended by its static extra-group rows (persist's resume
    commit reads the PLE state slot as ids[group][0]; the scheduler sends pool groups only)."""
    if not _W['owners']:
        return block_ids
    ids = tuple(block_ids)[:_W['n_pool']]
    sid = _W['by_req'].get(req_id)
    if sid is None:
        raise RuntimeError(f'draftring: resumed request {req_id} has no session')
    STATS['persist_rows'] += 1
    return ids + (static_row('attn', sid, _W['width']), static_row('ring', sid, 1), static_row('ple', sid, 1))


PATCHERS[MR] = patch_model_runner


# ------------------------------------------------------------------------------------------------ section 3: eviction renumbering
def ring_renumber_plan(keep_rows, rows_per_page, pages=None, old_rows=None):
    """(old rows, new rows) to copy inside one session's ring after a compaction: the last P x rows_per_page new rows, each from
    the old row it came from (keep_rows[new]); and the count of sources that had already left the old ring (stale), from the
    plan's old row count (old_rows; the last kept row + 1 when absent)."""
    import numpy as np
    pages = P if pages is None else pages
    keep = np.asarray(keep_rows, dtype=np.int64)
    n, cap = len(keep), pages * rows_per_page
    lo = max(0, n - cap)
    new = np.arange(lo, n, dtype=np.int64)
    old = keep[lo:n]
    old_len = int(old_rows) if old_rows is not None else (int(keep[-1]) + 1 if n else 0)
    stale = int(np.count_nonzero(old < old_len - cap)) if n else 0
    return old, new, stale


def ring_slots(rows, sid, rows_per_page, pages=None):
    """(page, offset) of rows of session sid in the ring: page sid * P + (row // rows_per_page) mod P."""
    pages = P if pages is None else pages
    return sid * pages + (rows // rows_per_page) % pages, rows % rows_per_page


def renumber_ring(cache, axis, old, new, sid, rows_per_page):
    """Two-phase gather -> scatter (no overlap hazard): rows `old` move to `new` in session sid's ring of `cache`
    ([pages, ...] with the token axis `axis`: 2 for the lut main KV, 1 for the compressed keys)."""
    import torch
    if not len(new):
        return 0
    o = torch.as_tensor(old, device=cache.device)
    n = torch.as_tensor(new, device=cache.device)
    sp, so = ring_slots(o, sid, rows_per_page)
    dp, do = ring_slots(n, sid, rows_per_page)
    if axis == 2:
        buf = cache[sp, :, so].clone()
        cache[dp, :, do] = buf
    elif axis == 1:
        buf = cache[sp, so].clone()
        cache[dp, do] = buf
    else:
        raise ValueError(axis)
    return len(new)


def _renumber_after_commit(request_id, plans):
    """The session's ring after a committed compaction: renumber its window rows by the engine's keep plans (main per token,
    compressed per group); the shadow (gate) takes the same plans over the full context."""
    sid = _W['by_req'].get(request_id)
    if sid is None:
        return
    main, comp, _, _ = _W['owners']
    for kind, name, axis in (('main', main, 2), ('compressed', comp, 1)):
        plan = plans.get(kind)
        if plan is None:
            continue
        old, new, stale = ring_renumber_plan(plan.keep_rows, plan.storage_block_size, old_rows=plan.old_rows)
        if kind == 'main' and len(plan.keep_rows):              # P choice: how far back the oldest window row's source lay
            w0 = plan.keep_rows[max(0, len(plan.keep_rows) - 16384)]
            h = STATS.setdefault('window_src_pages', {})
            b = str((int(plan.old_rows) - 1 - int(w0)) // int(plan.storage_block_size))
            h[b] = h.get(b, 0) + 1
        for t in _first_dim_views(_W['caches'].get(name)):
            if t.shape[axis] != plan.storage_block_size:
                raise RuntimeError(f'draftring: {name} token axis {tuple(t.shape)}[{axis}] != plan page {plan.storage_block_size}')
            STATS['renumber_rows'] += renumber_ring(t, axis, old, new, sid, plan.storage_block_size)
        STATS['renumber_stale_rows'] += stale
        if SHADOW:
            _sh_compact(request_id, kind, plan, len(new) and int(new[0]), stale)
    STATS['renumbers'] += 1


def _install_commit_hook():
    """Wrap engine_eviction_async.worker_commit_async in THIS (worker) process only: the core pickles the function by reference."""
    import engine_eviction_async as eea
    if getattr(eea.worker_commit_async, '_s2_draftring', False):
        return
    original = eea.worker_commit_async

    def worker_commit_async(worker, request_id):
        runner = worker.model_runner
        entry = getattr(runner, '_arc3_async_evictions', {}).get(request_id)
        plans = {}
        if entry is not None:
            for group in entry['wire']['groups']:
                for item in group['copies']:
                    plans.setdefault(item['kind'], item['plan'])
        result = original(worker, request_id)
        if isinstance(result, dict) and result.get('worker_synchronized') and plans:
            import time as _t
            import torch
            t0 = _t.perf_counter()
            try:
                with torch.inference_mode():                    # dr2 (10-15): the RPC runs outside inference mode and the
                    _renumber_after_commit(request_id, plans)   # caches are inference tensors (in-place copies need it)
            except Exception as exc:                            # noqa: BLE001  drafter only: tau, never outputs
                STATS['renumber_errors'] += 1
                STATS['last_renumber_error'] = repr(exc)[:300]
            STATS['renumber_ms'] += (_t.perf_counter() - t0) * 1e3
        return result

    worker_commit_async._s2_draftring = True
    worker_commit_async.__wrapped__ = original
    eea.worker_commit_async = worker_commit_async


# ------------------------------------------------------------------------------------------------ section 4: exactness gate (debug)
# IE's gate (2026-10-15): ARC3_S2_DRAFTRING_SHADOW=<S> (test servers only; 0 = off) keeps a full-context shadow of the drafter's
# main-KV rows and compressed keys for the first S sessions (a session that ends hands its slot to the next new one), indexed by
# resident row, and checks every step (structural) and every ARC3_S2_DRAFTRING_SHADOW_EVERY steps (content):
#   slots     the step's slot mapping of the drafter's attention group equals the static ring address of every token's position;
#   tables    the drafter's three groups' block-table rows equal the static rows of their sessions (attention: sid x P + j mod P);
#   content   every stable row of a shadowed session's window (the last 16,448 resident rows below the copy mark; compressed: the
#             last 4,112) equals its shadow: rows enter the shadow from the ring once stable (64 rows / 16 groups behind the host's
#             computed count, which async scheduling may overstate by a few drafts), and an eviction commit compacts the shadow with
#             the engine's own keep plan over the FULL context (the pool-layout reference), so a renumbering, an alias, a zeroing or
#             another session's write that changes a window row shows as a mismatch. Rows whose source had left the ring before a
#             compaction (counted 'stale'; tau only) are excluded through a floor and reported.
# Counters stay on the GPU; every 200 steps they are read (a host sync) into STATS['shadow'] and ARC3_S2_DRAFTRING_SHADOW_FILE.
SHADOW = int(os.environ.get('ARC3_S2_DRAFTRING_SHADOW', '0') or 0)
SHADOW_EVERY = int(os.environ.get('ARC3_S2_DRAFTRING_SHADOW_EVERY', '8') or 8)
SHADOW_FILE = os.environ.get('ARC3_S2_DRAFTRING_SHADOW_FILE', '')
SH_MARGIN, SH_CMARGIN, SH_WIN, SH_CWIN = 64, 16, 16384 + 64, (16384 + 64) // 4
MS = 'vllm.v1.worker.gpu.model_states.mamba_hybrid'
_SH = dict(slot_of={}, free=list(range(SHADOW)), hw=[0] * SHADOW, chw=[0] * SHADOW, floor=[-1] * SHADOW, cfloor=[-1] * SHADOW,
           sid=[0] * SHADOW, main=None, comp=None, counters=None, per=None, steps=0, bs=None, cbs=None)
_SH_NAMES = ('slot_mismatch', 'slot_tokens', 'table_mismatch', 'table_rows', 'main_mismatch', 'main_rows', 'comp_mismatch',
             'comp_rows')
if SHADOW:
    STATS['shadow'] = dict(sessions=SHADOW, every=SHADOW_EVERY, steps=0, assigned=0, compactions=0, stale_window_main=0,
                           stale_window_comp=0, copies_main=0, copies_comp=0, errors=0, last_error=None,
                           **{k: 0 for k in _SH_NAMES})


def _sh_assign(req_id):
    if _SH['free'] and not str(req_id).startswith('_dummy_req_'):
        k = _SH['free'].pop(0)
        _SH['slot_of'][req_id] = k
        _SH['hw'][k] = _SH['chw'][k] = 0
        _SH['floor'][k] = _SH['cfloor'][k] = -1
        _SH['sid'][k] = _W['by_req'][req_id]
        STATS['shadow']['assigned'] += 1


def _sh_release(req_id):
    k = _SH['slot_of'].pop(req_id, None)
    if k is not None:
        _SH['free'].append(k)
        _SH['free'].sort()


def _sh_alloc(dev):
    import torch
    main, comp, _, _ = _W['owners']
    mc = _first_dim_views(_W['caches'].get(main))[0]          # [pages, heads, rows per page, 296] (uint8)
    cc = _first_dim_views(_W['caches'].get(comp))[0]          # [pages, groups per page, ...]
    _SH['bs'], _SH['cbs'] = int(mc.shape[2]), int(cc.shape[1])
    rows = int(_W['runner'].max_model_len) + 4096
    _SH['main'] = torch.zeros((SHADOW, rows, mc.shape[1], *mc.shape[3:]), dtype=mc.dtype, device=dev)
    _SH['comp'] = torch.zeros((SHADOW, rows // 4 + 64, *cc.shape[2:]), dtype=cc.dtype, device=dev)
    _SH['counters'] = torch.zeros(len(_SH_NAMES), dtype=torch.int64, device=dev)
    _SH['per'] = torch.zeros((SHADOW, 2), dtype=torch.int64, device=dev)
    STATS['shadow'].update(shadow_bytes=_SH['main'].numel() * _SH['main'].element_size() + _SH['comp'].numel() * _SH['comp'].element_size(),
                           shapes=[list(mc.shape), list(cc.shape)])


def _sh_rows(cache, axis, rows, sid, bs):
    import torch
    r = torch.as_tensor(rows, dtype=torch.int64).to(cache.device, non_blocking=True)
    pg, off = ring_slots(r, sid, bs)
    return cache[pg, :, off] if axis == 2 else cache[pg, off]


def _sh_compare(k, kind, lo, hi):
    import numpy as np
    if hi <= lo:
        return
    main, comp, _, _ = _W['owners']
    sid = _SH['sid'][k]
    if kind == 'main':
        ring = _sh_rows(_first_dim_views(_W['caches'][main])[0], 2, np.arange(lo, hi), sid, _SH['bs'])
        ref = _SH['main'][k, lo:hi]
        c, cnt, j = 4, 5, 0
    else:
        ring = _sh_rows(_first_dim_views(_W['caches'][comp])[0], 1, np.arange(lo, hi), sid, _SH['cbs'])
        ref = _SH['comp'][k, lo:hi]
        c, cnt, j = 6, 7, 1
    bad = (ring != ref).reshape(hi - lo, -1).any(dim=1).sum()
    _SH['counters'][c] += bad
    _SH['counters'][cnt] += hi - lo
    _SH['per'][k, j] += bad


def _sh_copy(k, kind, lo, hi):
    import numpy as np
    if hi <= lo:
        return
    main, comp, _, _ = _W['owners']
    sid = _SH['sid'][k]
    if kind == 'main':
        _SH['main'][k, lo:hi] = _sh_rows(_first_dim_views(_W['caches'][main])[0], 2, np.arange(lo, hi), sid, _SH['bs'])
        STATS['shadow']['copies_main'] += hi - lo
    else:
        _SH['comp'][k, lo:hi] = _sh_rows(_first_dim_views(_W['caches'][comp])[0], 1, np.arange(lo, hi), sid, _SH['cbs'])
        STATS['shadow']['copies_comp'] += hi - lo


def _sh_step(state, input_batch, block_tables, slot_mappings):
    import numpy as np
    import torch
    n = int(input_batch.num_reqs)
    if n == 0 or not _W['groups']:
        return
    req_ids = list(input_batch.req_ids[:n])
    sids = [_W['by_req'].get(r, 0) for r in req_ids]
    if not any(sids):
        return
    dev = slot_mappings.device
    if _SH['main'] is None:
        _sh_alloc(dev)
    g0, g1, g2 = _W['groups']
    cnt = _SH['counters']
    sid_t = torch.tensor(sids, dtype=torch.int64).to(dev, non_blocking=True)
    valid = sid_t > 0
    # tables: the attention group's rows and the ring / PLE rows of every scheduled session
    width = _W['width']
    exp = sid_t[:, None] * P + (torch.arange(width, device=dev) % P)[None, :]
    cnt[2] += ((block_tables[g0][:n, :width].long() != exp) & valid[:, None]).sum()
    for g in (g1, g2):
        cnt[2] += ((block_tables[g][:n, 0].long() != sid_t) & valid).sum()
    cnt[3] += int(sum(1 for x in sids if x)) * (width + 2)
    # slots: the drafter's attention group's slot of every token = its position's static ring address
    T = int(input_batch.num_tokens)
    qsl = np.asarray(input_batch.query_start_loc_np[:n + 1], dtype=np.int64)
    b_of_t = torch.from_numpy(np.repeat(np.arange(n, dtype=np.int64), qsl[1:] - qsl[:-1])).to(dev, non_blocking=True)
    pos = input_batch.positions[:T].long()
    st = sid_t[b_of_t]
    bs = _SH['bs']
    want = (st * P + (pos // bs) % P) * bs + pos % bs
    cnt[0] += ((slot_mappings[g0][:T].long() != want) & (st > 0)).sum()
    cnt[1] += int(qsl[-1] - qsl[0])
    # content: copy newly stable rows of shadowed sessions; compare their windows every SHADOW_EVERY steps
    states = getattr(_W['runner'], '_arc3_eviction_states', {}) or {}
    computed = np.asarray(input_batch.num_computed_tokens_np[:n], dtype=np.int64)
    check = _SH['steps'] % SHADOW_EVERY == 0
    for b, rid in enumerate(req_ids):
        k = _SH['slot_of'].get(rid)
        if k is None:
            continue
        st_ = states.get(rid) or {}
        pad = int(st_.get('deleted_tokens', 0)) - 4 * int(st_.get('deleted_groups', 0))
        top, ctop = int(computed[b]) - SH_MARGIN, (int(computed[b]) + pad) // 4 - SH_CMARGIN
        if check:
            hw, chw = _SH['hw'][k], _SH['chw'][k]
            _sh_compare(k, 'main', max(0, hw - SH_WIN, _SH['floor'][k] + 1), hw)
            _sh_compare(k, 'comp', max(0, chw - SH_CWIN, _SH['cfloor'][k] + 1), chw)
        if top > _SH['hw'][k]:
            _sh_copy(k, 'main', _SH['hw'][k], top)
            _SH['hw'][k] = top
        if ctop > _SH['chw'][k]:
            _sh_copy(k, 'comp', _SH['chw'][k], ctop)
            _SH['chw'][k] = ctop


def _sh_compact(req_id, kind, plan, lo, stale):
    """After the ring renumbering: the shadow takes the engine's keep plan over the full context (the pool-layout reference)."""
    import numpy as np
    import torch
    k = _SH['slot_of'].get(req_id)
    if k is None or _SH['main'] is None:
        return
    keep = np.asarray(plan.keep_rows, dtype=np.int64)
    n_keep = len(keep)
    main = kind == 'main'
    hw = _SH['hw'][k] if main else _SH['chw'][k]
    fl = _SH['floor'][k] if main else _SH['cfloor'][k]
    kk = int(np.searchsorted(keep, hw))                     # kept rows the shadow holds (a prefix: keep is increasing)
    buf = _SH['main'] if main else _SH['comp']
    if kk:
        src = torch.as_tensor(keep[:kk]).to(buf.device, non_blocking=True)
        buf[k, :kk] = buf[k].index_select(0, src)
    new_fl = int(np.searchsorted(keep, fl + 1)) - 1 if fl >= 0 else -1
    if stale:
        new_fl = max(new_fl, lo + stale - 1)
        win = SH_WIN if main else SH_CWIN
        STATS['shadow']['stale_window_main' if main else 'stale_window_comp'] += max(0, lo + stale - max(lo, n_keep - win))
    if main:
        _SH['hw'][k], _SH['floor'][k] = min(kk, n_keep), new_fl
    else:
        _SH['chw'][k], _SH['cfloor'][k] = min(kk, n_keep), new_fl
    STATS['shadow']['compactions'] += 1


def shadow_flush():
    if _SH['counters'] is None:
        return
    c = _SH['counters'].tolist()
    STATS['shadow'].update(dict(zip(_SH_NAMES, c)), steps=_SH['steps'], per_slot=_SH['per'].tolist(),
                           slots={str(k): dict(sid=_SH['sid'][k], hw=_SH['hw'][k], chw=_SH['chw'][k], floor=_SH['floor'][k],
                                               cfloor=_SH['cfloor'][k]) for k in _SH['slot_of'].values()})
    if SHADOW_FILE:
        import json
        tmp = SHADOW_FILE + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(STATS, fh, default=str)
        os.replace(tmp, SHADOW_FILE)


def patch_model_state(mod):
    cls = mod.MambaHybridModelState
    orig = cls.prepare_attn

    def prepare_attn(self, input_batch, cudagraph_mode, block_tables, slot_mappings, attn_groups, kv_cache_config,
                     for_capture=False):
        import torch
        if not for_capture and not torch.cuda.is_current_stream_capturing():
            try:
                _sh_step(self, input_batch, block_tables, slot_mappings)
                _SH['steps'] += 1
                if _SH['steps'] % 200 == 0:
                    shadow_flush()
            except Exception as exc:                            # noqa: BLE001  test servers: count, never break the step
                STATS['shadow']['errors'] += 1
                STATS['shadow']['last_error'] = repr(exc)[:300]
        return orig(self, input_batch, cudagraph_mode, block_tables, slot_mappings, attn_groups, kv_cache_config,
                    for_capture=for_capture)

    cls.prepare_attn = prepare_attn


if SHADOW:
    from .rollback import PINS as _RB_PINS                     # the same module as rollback / kvring: chained hooks
    PINS[MS] = _RB_PINS[MS]
    PATCHERS[MS] = patch_model_state
