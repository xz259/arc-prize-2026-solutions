"""l4port_draftring_pp.py (2026-10-31, ARC2 - main; the L4 production-path port, after l4p35): stage2_v1's 'draftring' part under
pipeline parallelism. draftring moves the MTP drafter's KV (main + compressed + raw-key ring) and the TP-replicated PLE state out of
the shared pool into worker-private extra groups, and its worker section assumes all four owners sit on one worker (three extra
groups, '[1, 1, 1]'). Under TP2 x PP2 the PLE lives on stage 0 (language_model.model.layers.1.ple) and the drafter on stage 1
(mtp.layers.48.*), so l4p35 stopped in extra_roles on every worker.

Here each worker appends the extra groups of the owners it has, in draftring's order (drafter attention, drafter ring, PLE):
stage 0 one group (PLE), stage 1 two (drafter attention, ring); block-table rows, the bpk check, the persist resume rows and the
compaction renumbering follow the groups present. With all four owners on one worker every function computes what draftring's
does (same groups, tensors, rows and checks, in the same order). The engine-core half (fold D's grouping) already works on the
merged spec map and is unchanged. At pp_size 1 draftring's own rule stands (all four owners on the worker, else a refusal).
Session ids are allocated per worker (session_id in append_block_ids), so one request has independent sids on the two stages; every
static row is worker-local (including persist's resume rows via group_block_ids), so nothing relies on a shared sid.

Mechanism: the replacement bodies below are written with draftring's own global names and swapped into draftring's function
objects (__code__; none of them has free variables), so every reference to those functions, including stage2_v1's registered
model-runner patcher, runs the new code with draftring's module globals. Applied once, at bootstrap (draftring is imported by
stage2_v1's install; vllm.v1.worker.gpu.model_runner is not yet), and only on the pinned draftring source. The functions here are
never called directly. Receipt line '[l4port] draftring PP: ...'."""
import hashlib
import sys

PIN = '89d0f3fe819ecfe140474cb04ef0519f61cc0a5cb305f52d5ea3138f87c6420a'
NAME = 'stage2_v1.draftring'
SWAPPED = []


# ------------------------------------------------------------------------------------------------ bodies (draftring's globals)
def extra_roles(extra):
    from vllm.v1.kv_cache_interface import CircularBufferSpec, FullAttentionSpec, MambaSpec, MLAAttentionSpec
    main = [n for n, s in extra.items() if type(s) is FullAttentionSpec]
    comp = [n for n, s in extra.items() if type(s) is MLAAttentionSpec]
    ring = [n for n, s in extra.items() if type(s) is CircularBufferSpec]
    ple = [n for n, s in extra.items() if isinstance(s, MambaSpec) and s.tp_replicated]
    drafter = (len(main), len(comp), len(ring))
    try:
        from vllm.distributed.parallel_state import get_pp_group
        pp_size = int(get_pp_group().world_size)
    except Exception:                                           # noqa: BLE001  no PP group (CPU tools): single-stage rules
        pp_size = 1
    if pp_size == 1 and not (drafter == (1, 1, 1) and len(ple) == 1):
        raise RuntimeError(f'draftring: unexpected extra owners {sorted(extra)}')             # draftring's own rule at PP 1
    if drafter not in ((1, 1, 1), (0, 0, 0)) or len(ple) > 1 or (drafter == (0, 0, 0) and not ple):
        raise RuntimeError(f'draftring: unexpected extra owners {sorted(extra)}')
    return (main[0] if main else None, comp[0] if comp else None, ring[0] if ring else None, ple[0] if ple else None)


def extend_config(cfg, extra, sessions, pages):
    import copy
    from vllm.v1.kv_cache_interface import KVCacheGroupSpec, KVCacheTensor, UniformTypeKVCacheSpecs
    main, comp, ring, ple = extra_roles(extra)
    groups, tensors = [], []
    if main is not None:
        attn_spec = UniformTypeKVCacheSpecs.from_specs({main: extra[main], comp: extra[comp]})
        if attn_spec is None:
            raise RuntimeError('draftring: drafter main / compressed specs are not uniform-compatible')
        ring_spec = replace(extra[ring], page_size_padded=extra[comp].page_size_bytes)     # padded as the stock CSA pool pads rings
        n_attn = sessions * pages
        groups += [KVCacheGroupSpec([main, comp], attn_spec), KVCacheGroupSpec([ring], ring_spec)]
        tensors += [KVCacheTensor(size=extra[main].page_size_bytes * n_attn, shared_by=[main]),
                    KVCacheTensor(size=extra[comp].page_size_bytes * n_attn, shared_by=[comp]),
                    KVCacheTensor(size=ring_spec.page_size_bytes * sessions, shared_by=[ring])]
    if ple is not None:
        ple_spec = replace(extra[ple], page_size_padded=None)                              # unpadded (IE): 245,760-B pages
        groups.append(KVCacheGroupSpec([ple], ple_spec))
        tensors.append(KVCacheTensor(size=ple_spec.page_size_bytes * sessions, shared_by=[ple]))
    out = copy.copy(cfg)
    out.kv_cache_groups = list(cfg.kv_cache_groups) + groups
    out.kv_cache_tensors = list(cfg.kv_cache_tensors) + tensors
    return out, (main, comp, ring, ple), sum(t.size for t in tensors)


def _wrap_block_tables(runner):
    bt = runner.block_tables
    original = bt.append_block_ids
    n_pool, groups, width = _W['n_pool'], _W['groups'], _W['width']
    kinds = list(_W['kinds'])

    def append_block_ids(req_index, new_block_ids, overwrite):
        ids = tuple(new_block_ids)
        if len(ids) == bt.num_kv_cache_groups:                  # MRv2's warmup_kernels allocates ids for every group of the
            ids = ids[:n_pool]                                  # runner's config (dr1, 10-15): the extra groups take static rows
            STATS['full_tuples'] = STATS.get('full_tuples', 0) + 1
        if len(ids) != n_pool:
            raise RuntimeError(f'draftring: {len(ids)} block-id groups for {n_pool} pool groups')
        if overwrite:
            sid = session_id(runner.req_states.index_to_req_id[req_index])
            rows = tuple(static_row(kind, sid, width if kind == 'attn' else 1) for kind in kinds)
            STATS['rows_attn'] += 1
        else:
            rows = tuple([] for _ in kinds)
        return original(req_index, ids + rows, overwrite)

    bt.append_block_ids = append_block_ids


def group_block_ids(req_id, block_ids):
    """The scheduler's per-group block ids of an existing request, extended by its static extra-group rows (persist's resume
    commit reads the PLE state slot as ids[group][0]; the scheduler sends pool groups only)."""
    if not _W['owners'] or not any(_W['owners']):
        return block_ids
    ids = tuple(block_ids)[:_W['n_pool']]
    sid = _W['by_req'].get(req_id)
    if sid is None:
        raise RuntimeError(f'draftring: resumed request {req_id} has no session')
    STATS['persist_rows'] += 1
    return ids + tuple(static_row(kind, sid, _W['width'] if kind == 'attn' else 1) for kind in _W['kinds'])


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
            _W['caches'] = {n: caches[n] for n in _W['owners'] if n is not None and n in caches}
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
        n_extra = len(cfg.kv_cache_groups) - len(kv_cache_config.kv_cache_groups)
        kinds = (['attn', 'ring'] if owners[0] is not None else []) + (['ple'] if owners[3] is not None else [])
        if len(kinds) != n_extra:
            raise RuntimeError(f'draftring: {n_extra} extra groups for the roles {kinds}')
        _W.update(owners=owners, n_pool=len(kv_cache_config.kv_cache_groups), runner=self,
                  free=list(range(1, sessions)), by_req={}, kinds=kinds)
        _W['groups'] = list(range(_W['n_pool'], _W['n_pool'] + n_extra))
        STATS.update(extra_groups=_W['groups'], extra_tensor_bytes=nbytes, sessions_max=sessions - 1, extra_kinds=kinds)
        result = orig_init(self, cfg)                          # deep-copies cfg into self.kv_cache_config (rollback / persist read it)
        bpk = [int(self.block_tables.blocks_per_kv_block[g]) for g in _W['groups']]
        if bpk != [1] * n_extra:                               # static rows and the ring arithmetic are in KV pages
            raise RuntimeError(f'draftring: extra groups need kernel block == KV block, blocks_per_kv_block {bpk}')
        _W['width'] = int(self.block_tables.input_block_tables[_W['groups'][0]].shape[1]) if owners[0] is not None else 1
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
            runner_only_attn_layers={n for n in (main, comp, ring) if n is not None},
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


def _renumber_after_commit(request_id, plans):
    """The session's ring after a committed compaction: renumber its window rows by the engine's keep plans (main per token,
    compressed per group); the shadow (gate) takes the same plans over the full context. A worker without the drafter's owners
    (an earlier pipeline stage) has no ring to renumber."""
    sid = _W['by_req'].get(request_id)
    if sid is None:
        return
    main, comp, _, _ = _W['owners']
    if main is None:
        return
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


BODIES = ('extra_roles', 'extend_config', '_wrap_block_tables', 'group_block_ids', 'patch_model_runner', '_renumber_after_commit')


def install():
    """Swap the bodies into the imported draftring module (once; pinned source; before the model runner is imported)."""
    mod = sys.modules.get(NAME)
    if mod is None or SWAPPED:
        return list(SWAPPED)
    digest = hashlib.sha256(open(mod.__file__, 'rb').read()).hexdigest()
    if digest != PIN:
        raise RuntimeError('l4port draftring PP: draftring.py changed (%s); re-review the port' % digest[:16])
    if 'vllm.v1.worker.gpu.model_runner' in sys.modules:
        raise RuntimeError('l4port draftring PP: the model runner was imported before the swap')
    import builtins
    src = {n: globals()[n] for n in BODIES}
    for n in BODIES:
        target = getattr(mod, n)
        if target.__code__.co_freevars or src[n].__code__.co_freevars:
            raise RuntimeError('l4port draftring PP: %s has free variables' % n)
        a, b = target.__code__, src[n].__code__
        if (a.co_argcount, a.co_kwonlyargcount, a.co_posonlyargcount, a.co_varnames[:a.co_argcount]) != \
                (b.co_argcount, b.co_kwonlyargcount, b.co_posonlyargcount, b.co_varnames[:b.co_argcount]):
            raise RuntimeError('l4port draftring PP: %s signature differs' % n)
    used = set()
    for n in BODIES:
        used |= _global_names(src[n].__code__)
    missing = sorted(g for g in used if g not in mod.__dict__ and not hasattr(builtins, g))
    if missing:
        raise RuntimeError('l4port draftring PP: names not in draftring: %s' % missing)
    for n in BODIES:
        getattr(mod, n).__code__ = src[n].__code__
        SWAPPED.append(n)
    print('[l4port] draftring PP: %d bodies swapped into %s (pin %s; %d global names resolved)' % (
        len(SWAPPED), NAME, PIN[:8], len(used)), flush=True)
    return list(SWAPPED)


def _global_names(code):
    """Names a code object (and its nested code objects) loads as globals."""
    import dis
    out = {i.argval for i in dis.get_instructions(code) if i.opname in ('LOAD_GLOBAL', 'STORE_GLOBAL', 'DELETE_GLOBAL', 'LOAD_NAME')}
    for c in code.co_consts:
        if hasattr(c, 'co_code'):
            out |= _global_names(c)
    return out
