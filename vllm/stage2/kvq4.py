"""stage2_v1 'kvq4' part (g4, 2026-10-26, Claude fork bdfc5402; the user: "g4 looks good. I think we will try to implement it,
make the rest of the stack compatible"; design and IE's review: review74/engine_prefill/fork_20261026/G4_DESIGN_2026-10-26.md).
The main-QSA lutc KV with one scale set per quad of four token rows (kvq4_kernels.py): 266-byte rows instead of 296, so the
same KV memory holds ~11% more tokens (the attention block follows the GDN page: 3,184 -> 3,552 tokens). Needs kvrot (lut
formats on every owner) and kvring (its ring ids and offsets key the side buffer); installed after both.

  spec      arc3_qsa_cache.side_bytes returns 133 for nvfp4 (qsa.py imports it at call time: shape, spec, page alignment).
  store     the QSA impl's cache update (kvrot's) runs kvq4_kernels.store with this layer's attention metadata (forward context),
            kvring's ring_meta and the layer's side buffer (allocated with the KV cache, before any capture).
  readers   kvlut_kernels.attention / kvring_kernels.attention are the quad readers; kvlut_kernels.store refuses 266-byte caches.
  tags      a ring id newly assigned to a request clears its side-buffer tags in every layer (a previous owner's rows never match).
  eviction  engine_eviction_async admits 266-byte main rows (source transform, as kvevict's) and checks every main-owner plan
            moves whole quads (check_plan; a violation raises: the planner snaps to groups, so it means a bug);
            the planner (history_eviction.plan_history_eviction, outermost wrapper) snaps every removal to whole QSA groups
            (positions // 4) and keeps every image whole or removes it whole (the engine refuses a cut multimodal feature):
            an image's edge group goes with the image, formatting tokens included, unless it holds a kept-out token or
            another kept image, which keeps the image whole; any other partly planned group is kept whole (the capacity
            retry expands it into evictable text instead); snap_groups (v2, 10-26 08:19: v1 kept image edge groups that
            held formatting, leaving images cut: the engine refused those evictions and the sessions closed).
  promptmove the prompt block's K moves by quads (K codes + K scales of whole quads; the block is a multiple of 4 rows under
            the snap rule, else the move is refused and counted).
STATS: the receipt; stats_dev (int64 [4], every layer; only with ARC3_S2_KVQ4_STATS=1, a diagnostic, off by default since
10-26 16:02 (the user: remove any diagnostic affecting the throughput): the store kernels then compile without the counters): [0] earlier quad rows re-encoded from decoded values (lossy; the smoke
asserts 0), [1] leaders without a usable buffer, [2] newest rows not buffered."""
import os

import numpy as np

from .kvrot import PINS as _KVROT_PINS
from .rollback import PINS as _ROLLBACK_PINS
from .persist import PINS as _PERSIST_PINS

QSA = 'vllm.models.qwen3_8_flash_next.nvidia.qsa'
MR = 'vllm.v1.worker.gpu.model_runner'
MS = 'vllm.v1.worker.gpu.model_states.mamba_hybrid'
PINS = {QSA: _KVROT_PINS[QSA], MR: _PERSIST_PINS[MR], MS: _ROLLBACK_PINS[MS],
        'engine_eviction_async': 'a9bd65dc98300a435be76cb0b40b49d15d021ca3e5550db56c2b8f24b78ef278',
        'history_eviction': '88a5d9aef289c87b485104a1cf8d08701a96600198a70196ccd43cd7803c8fc0'}
SIDE = 133
STATS_ON = os.environ.get('ARC3_S2_KVQ4_STATS', '0') == '1'   # the fallback counters / event log (diagnostic)
MODE = os.environ.get('ARC3_S2_KVQ4_MODE', 'full')   # 'order' (diagnostic, 10-26): only the head-major stride order, lutc unchanged
assert MODE in ('full', 'order'), MODE
STATS = dict(side_bytes_calls=0, store_calls=0, layers=0, buffers=0, buffer_bytes=0, tag_clears=0, plans_checked=0,
             plan_violations=0, last_violation=None, snaps=0, snap_groups=0, snap_shrunk=0, snap_expanded=0, snap_orphans=0,
             snap_struct_removed=0, snap_images_kept=0, snap_rounds_max=0,
             snap_capacity_retries=0, last_snap=None, pmove_refused=0, errors=0, last_error=None, stats_dev=None)
_BUFS = {}                 # layer name -> (buf_k, buf_v, buf_tag)
_DEV = {}                  # 'stats': int64 [4]
_SEEN = {}                 # req_id -> the ring id whose tags were cleared for it


def _err(exc):
    STATS['errors'] += 1
    STATS['last_error'] = repr(exc)[:400]


def _ring_meta():
    from . import kvring
    m = kvring._META.get('dev')
    if m is None:
        raise RuntimeError('kvq4: kvring has no ring_meta (kvring installed? its KV-init hook run?)')
    return m


def export_stats():
    """The device counters into STATS without a sync: the previous non-blocking copy if it has landed, then a new one."""
    import torch
    s = _DEV.get('stats')
    if s is None or torch.cuda.is_current_stream_capturing():
        return
    ev = _DEV.get('event')
    if ev is not None:
        if not ev.query():
            return
        v = [int(x) for x in _DEV['host'].tolist()]
        STATS['stats_dev'] = v[:4]
        n = min(v[4], 64)
        names = ('req', 'ring_id', 'row', 'i0', 'slot', 'found', 'tag_prev', 'expected_prev', 'off', 'seq_len')
        STATS['fallback_events'] = v[4]
        STATS['fallback_first'] = [dict(zip(names, v[8 + 10 * e:18 + 10 * e])) for e in range(n)]
    _DEV['host'].copy_(s, non_blocking=True)
    ev = torch.cuda.Event()
    ev.record()
    _DEV['event'] = ev


# ------------------------------------------------------------------------------------------------ QSA impl
def patch_qsa(mod):
    import torch
    import arc3_qsa_cache as qc
    from . import kvlut_kernels, kvring_kernels, kvq4_kernels, kvrot
    if mod._ARC3_QSA_KV_FORMAT != 'nvfp4':
        raise RuntimeError('kvq4 needs ARC3_QSA_KV_FORMAT=nvfp4 (the lutc owners), got %r' % mod._ARC3_QSA_KV_FORMAT)
    target, draft = kvrot.role_formats()
    if not (target.startswith('lut') and draft.startswith('lut')):
        raise RuntimeError('kvq4 needs codebook formats on every QSA owner (ARC3_S2_KVROT_TARGET/DRAFT lut*), got %r/%r'
                           % (target, draft))

    # Head-major main QSA caches (vLLM's HND order, [blocks, heads, PAGE, 266] contiguous): the default NHD order interleaves the
    # two KV heads per token, so a head's four quad rows would not be one 1,064-byte run. Every reader, writer and mover of these
    # caches addresses them by strides or tensor indexing (eviction_storage.compact_cache_rows_, promptmove, kvring).
    backend = mod.Qwen3_8FlashNextQSAFlashAttentionBackend

    def get_kv_cache_stride_order(include_num_layers_dimension=False):
        return (1, 2, 0, 3, 4) if include_num_layers_dimension else (0, 1, 2, 3)

    backend.get_kv_cache_stride_order = staticmethod(get_kv_cache_stride_order)
    STATS['mode'] = MODE
    if MODE == 'order':
        return

    orig_side = qc.side_bytes

    def side_bytes(head_dim, fmt):
        STATS['side_bytes_calls'] += 1
        if fmt == 'nvfp4' and head_dim == 256:
            return SIDE
        return orig_side(head_dim, fmt)

    qc.side_bytes = side_bytes

    def refuse_store(key, value, cache, slots, *a, **k):
        if cache.ndim == 4 and cache.shape[-1] == kvq4_kernels.ROW:
            raise RuntimeError('kvq4: a 296-byte lutc store reached a quad cache')
        return orig_store(key, value, cache, slots, *a, **k)

    orig_store = kvlut_kernels.store
    kvlut_kernels.store = refuse_store
    kvlut_kernels.attention = kvq4_kernels.attention
    kvring_kernels.attention = kvq4_kernels.ring_attention

    cls = mod.Qwen3_8FlashNextQSAFlashAttentionImpl
    inner = cls.do_kv_cache_update
    if getattr(inner, '__qualname__', '').split('.')[0] != 'patch_qsa':
        raise RuntimeError('kvq4 must be installed after kvrot')

    def role(layer):
        r = getattr(layer, '_arc3_kvrot_role', None)
        if r is None:                       # as kvrot's role(): resolved on the owner's first call
            name = getattr(layer, 'layer_name', '')
            is_draft = bool(kvrot._DRAFT_NAME.search(name))
            fmt = draft if is_draft else target
            r = (fmt, is_draft, kvrot.layer_means(name) if fmt.startswith('lutc') else None)
            layer._arc3_kvrot_role = r
            if is_draft:
                kvrot.STATS['draft_layers'].append(name)
            else:
                kvrot.STATS['target_layers'] += 1
        return r

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        fmt, is_draft, mu = role(layer)
        name = getattr(layer, 'layer_name', '')
        bufs = _BUFS.get(name)
        if bufs is None:
            raise RuntimeError('kvq4: no side buffer for %s (allocated at KV-cache init)' % name)
        from vllm.forward_context import get_forward_context
        md = get_forward_context().attn_metadata
        if isinstance(md, list):
            md = md[0]
        main = md[name]
        side = md[layer.indexer.raw_key_cache.prefix]
        tr = side.token_to_req
        n = min(int(slot_mapping.shape[0]), int(tr.shape[0]), int(key.shape[0]))
        kvrot.STATS['store_calls'] += 1
        kvrot.STATS['store_rows'] += int(slot_mapping.shape[0])
        if is_draft:
            kvrot.STATS['draft_store_calls'] += 1
        STATS['store_calls'] += 1
        mk, mv = (mu[0], mu[1]) if mu is not None else (None, None)
        kvq4_kernels.store(key, value, kv_cache, slot_mapping, tr, _ring_meta(), main.block_table, main.seq_lens, *bufs,
                           mk, mv, stats=_DEV.get('stats'), n=n)

    do_kv_cache_update.__qualname__ = 'patch_qsa.<locals>.do_kv_cache_update'
    cls.do_kv_cache_update = do_kv_cache_update
    try:
        from . import promptmove
        _patch_promptmove(promptmove)
    except Exception as exc:                                         # noqa: BLE001
        _err(exc)


# ------------------------------------------------------------------------------------------------ buffers and tags
def install_model_runner(module):
    cls = module.GPUModelRunner
    orig = cls.initialize_kv_cache

    def initialize_kv_cache(self, kv_cache_config):
        result = orig(self, kv_cache_config)
        if MODE == 'order':
            return result
        import torch
        from vllm.config import get_layers_from_vllm_config
        from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('kvq4: side buffers would be allocated inside a CUDA-graph capture')
        ids = int(self.max_num_reqs)
        if STATS_ON:
            _DEV['stats'] = torch.zeros(8 + 64 * 10, dtype=torch.int64, device=self.device)
            _DEV['host'] = torch.zeros(8 + 64 * 10, dtype=torch.int64).pin_memory()
        for name, layer in get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase).items():
            if type(getattr(layer, 'impl', None)).__name__ != 'Qwen3_8FlashNextQSAFlashAttentionImpl' or name in _BUFS:
                continue
            kvh, d = int(layer.num_kv_heads), int(layer.head_dim)
            bk = torch.zeros((ids * 8, kvh, d), dtype=torch.bfloat16, device=self.device)
            _BUFS[name] = (bk, torch.zeros_like(bk), torch.full((ids * 8,), -1, dtype=torch.int32, device=self.device))
            STATS['buffers'] += 1
            STATS['buffer_bytes'] += 2 * bk.numel() * 2 + ids * 8 * 4
            cache = getattr(layer, 'kv_cache', None)
            cache = cache[0] if isinstance(cache, (list, tuple)) else cache
            if torch.is_tensor(cache) and cache.numel():
                from . import kvq4_kernels
                kvq4_kernels.check_cache(cache)
                STATS['page'] = int(cache.shape[2])
        STATS['layers'] = len(_BUFS)
        return result

    cls.initialize_kv_cache = initialize_kv_cache


def _clear(rid):
    import torch
    with torch.inference_mode():
        for _, _, tag in _BUFS.values():
            tag[rid * 8:(rid + 1) * 8].fill_(-1)
    STATS['tag_clears'] += 1


def install_model_state(module):
    cls = module.MambaHybridModelState
    orig_add, orig_remove = cls.add_request, cls.remove_request

    def add_request(self, req_index, new_req_data):
        result = orig_add(self, req_index, new_req_data)
        try:
            from . import kvring
            req_id = new_req_data.req_id
            rid = kvring._IDS.get(req_id)
            if rid is not None and _SEEN.get(req_id) != rid and _BUFS:
                _clear(rid)
                _SEEN[req_id] = rid
        except Exception as exc:                                     # noqa: BLE001
            _err(exc)
        return result

    def remove_request(self, req_id):
        result = orig_remove(self, req_id)
        try:
            from . import kvring
            if req_id not in kvring._IDS:
                _SEEN.pop(req_id, None)
            export_stats()
        except Exception as exc:                                     # noqa: BLE001
            _err(exc)
        return result

    cls.add_request, cls.remove_request = add_request, remove_request


# ------------------------------------------------------------------------------------------------ engine eviction
OLD = 'if item["kind"] == "main" and (cache.dtype != torch.uint8 or cache.shape[-1] != 296):'
NEW = ('if item["kind"] == "main" and cache.dtype == torch.uint8 and cache.shape[-1] == 266:   # kvq4: whole quads only\n'
       '                __import__("stage2_v1.kvq4", fromlist=["check_plan"]).check_plan(page, item["layer_name"])\n'
       '            if item["kind"] == "main" and (cache.dtype != torch.uint8 or cache.shape[-1] not in (296, 266)):')


def transform_eviction_async(src):
    assert src.count(OLD) == 1, 'kvq4: gate anchor'
    return src.replace(OLD, NEW)


_PLAN_OK = {}


def check_plan(page, layer_name):
    """A main-owner keep plan (new row j <- old row keep[j]) moves whole quads: keep[j] = j (mod 4) and the rows of each new quad
    are consecutive old rows (a deletion inside a quad would shift its rows across scale sets)."""
    key = id(page)
    if _PLAN_OK.get(key) is page:
        return
    keep = np.asarray(page.keep_rows, dtype=np.int64)
    STATS['plans_checked'] += 1
    j = np.arange(len(keep))
    bad = keep % 4 != j % 4
    if len(keep) > 1:
        bad[1:] |= (j[1:] % 4 != 0) & (keep[1:] != keep[:-1] + 1)
    if bad.any():
        first = int(np.argmax(bad))
        STATS['plan_violations'] += 1
        STATS['last_violation'] = dict(layer=layer_name, new_row=first, old_row=int(keep[first]), rows=int(len(keep)),
                                       old_rows=int(getattr(page, 'old_rows', -1)))
        raise RuntimeError('kvq4: eviction plan splits a quad: %r' % STATS['last_violation'])
    if len(_PLAN_OK) > 64:
        _PLAN_OK.clear()
    _PLAN_OK[key] = page


# ------------------------------------------------------------------------------------------------ planner
def _merge(ranges):
    out = []
    for a, b in sorted((int(a), int(b)) for a, b in ranges if int(b) > int(a)):
        if out and a <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], b))
        else:
            out.append((a, b))
    return out


def _in_ranges(starts, ends, x):
    """Membership of positions x (int array) in sorted disjoint ranges."""
    i = np.searchsorted(starts, x, side='right') - 1
    ok = i >= 0
    out = np.zeros(len(x), dtype=bool)
    out[ok] = x[ok] < ends[i[ok]]
    return out


def _runs(mask):
    """Sorted disjoint (start, end) runs of True in a bool array."""
    if not mask.any():
        return []
    d = np.diff(np.concatenate([[0], mask.view(np.int8), [0]]))
    return [(int(a), int(b)) for a, b in zip(np.nonzero(d == 1)[0], np.nonzero(d == -1)[0])]


def snap_groups(ranges, resident, keep_out, structural, images, expand_text=False):
    """Removal ranges snapped to whole QSA groups (logical positions // 4), every image removed whole or kept whole.
    ranges: the planned removals; resident: the resident logical ranges before the plan; keep_out: ranges never removed
    (prefix, protected tail, pin); structural: positions the planner never selects (chat formatting); images: (start, end)
    spans (vision_start .. vision_end, a superset of the engine's feature rows).
    Rules, applied until stable: an image partly planned is kept whole; a group partly planned whose planned part holds
    image tokens is removed whole, formatting included (the image is removed, its edge groups go with it), unless the
    group's other resident tokens hold a kept-out token or another image's tokens: then those planned images are kept
    whole; any other partly planned group is kept whole, or with expand_text (the capacity retry) removed whole when its
    other tokens are evictable text. Returns (ranges, info); info['orphans'] counts images left cut (always 0: else a
    RuntimeError)."""
    plan = _merge(ranges)
    zero = dict(groups=0, shrunk=0, expanded=0, orphans=0, struct_removed=0, images_kept=0, rounds=0)
    if not plan:
        return [], zero
    res = _merge(resident)
    spans = [(int(a), int(b)) for a, b in images if int(b) > int(a)]
    ko = _merge(keep_out)
    top = max([b for _, b in plan] + [b for _, b in res] + [b for _, b in spans] + [b for _, b in ko])
    n = -(-top // 4) * 4
    R = np.zeros(n, bool)
    P = np.zeros(n, bool)
    K = np.zeros(n, bool)
    S = np.zeros(n, bool)
    I = np.full(n, -1, np.int64)
    for a, b in res:
        R[a:b] = True
    for a, b in plan:
        P[a:b] = True
    for a, b in ko:
        K[a:b] = True
    st = np.fromiter((int(x) for x in structural), dtype=np.int64)
    S[st[(st >= 0) & (st < n)]] = True
    for k, (a, b) in enumerate(spans):
        I[a:b] = k
    P &= R & ~K
    P0 = P.copy()
    A = np.fromiter((a for a, _ in spans), dtype=np.int64, count=len(spans))
    B = np.fromiter((b for _, b in spans), dtype=np.int64, count=len(spans))

    def resident_in_images(M):
        c = np.concatenate([[0], np.cumsum(M & R, dtype=np.int64)])
        return c[B] - c[A]

    rc = resident_in_images(R)
    groups = None
    for rounds in range(1, 65):
        changed = False
        pc = resident_in_images(P)
        for k in np.nonzero((pc > 0) & (pc < rc))[0].tolist():     # no image is cut
            P[A[k]:B[k]] = False
            changed = True
        Pg, Rg = P.reshape(-1, 4), R.reshape(-1, 4)
        part = np.nonzero((Pg & Rg).any(1) & (Rg & ~Pg).any(1))[0]
        if groups is None:
            groups = len(part)
        for g in part.tolist():
            lo, hi = 4 * g, 4 * g + 4
            planned = P[lo:hi] & R[lo:hi]
            rest = R[lo:hi] & ~P[lo:hi]
            if not planned.any() or not rest.any():
                continue
            imgs = set(I[lo:hi][planned].tolist()) - {-1}
            blocked = bool(K[lo:hi][rest].any() or (I[lo:hi][rest] >= 0).any())
            if imgs and not blocked:
                P[lo:hi] |= R[lo:hi]                               # the image's edge group goes whole
            elif imgs:
                for k in imgs:                                     # the image stays whole
                    a, b = spans[k]
                    P[a:b] = False
                P[lo:hi] = False
            elif expand_text and not blocked and not S[lo:hi][rest].any():
                P[lo:hi] |= R[lo:hi]
            else:
                P[lo:hi] = False
            changed = True
        if not changed:
            break
    else:
        raise RuntimeError('kvq4 snap did not converge')
    Pg, Rg = P.reshape(-1, 4), R.reshape(-1, 4)
    if ((Pg & Rg).any(1) & (Rg & ~Pg).any(1)).any():
        raise RuntimeError('kvq4 snap left a partial group')
    pc, pc0 = resident_in_images(P), resident_in_images(P0)
    orphans = int(((pc > 0) & (pc < rc)).sum())
    if orphans:
        raise RuntimeError('kvq4 snap left %d cut images' % orphans)
    kept = int(((pc0 > 0) & (pc == 0)).sum())
    return _runs(P), dict(groups=groups or 0, shrunk=int((P0 & ~P).sum()), expanded=int((P & ~P0).sum()), orphans=orphans,
                          struct_removed=int((P & ~P0 & S).sum()), images_kept=kept, rounds=rounds)


def _subtract(pieces, removed):
    out = []
    removed = sorted(removed)
    for a, b in sorted(pieces):
        cur = a
        for c, d in removed:
            if d <= cur or c >= b:
                continue
            if c > cur:
                out.append((cur, c))
            cur = max(cur, d)
            if cur >= b:
                break
        if cur < b:
            out.append((cur, b))
    return out


def patch_history_eviction(mod):
    if MODE == 'order':
        return
    original = mod.plan_history_eviction

    def plan_history_eviction(policy, registry, ledger, **kw):
        decision = original(policy, registry, ledger, **kw)
        if not decision.ranges:
            return decision
        rec = decision.receipt
        logical = ledger.logical_tokens
        obs = registry.observations
        prefix = obs[0].start if obs else 0
        segments = kw.get('segments') or ()
        pin = None
        if mod._pin_observation_enabled() and not kw.get('incoming_observation') and rec.get('triggered'):
            pin = mod._pinned_observation(obs, segments, logical)
        tail = rec.get('protected_tail_range') or [max(0, logical - 4), logical]
        keep_out = [(0, prefix), (int(tail[0]), logical), (max(0, logical - 4), logical)] + ([tuple(pin)] if pin else [])
        resident = mod._subtract([(0, logical)], _merge(ledger.evicted)) if logical else []
        protected = [(0, prefix)] if prefix else []
        tail4 = max(0, logical - 4)
        protected = [(0, logical)] if (protected and tail4 <= prefix) else protected + [(tail4, logical)]
        budget = policy.resident_token_budget
        inc = kw.get('incoming_tokens', 0)
        structural = kw.get('structural') or ()
        images = kw.get('images') or ()
        snapped, info = snap_groups(decision.ranges, resident, keep_out, structural, images)
        if snapped == _merge(decision.ranges):
            return decision
        try:
            plan = ledger.plan(snapped, protected=mod._union(protected, pin))
            if rec.get('triggered') and plan.resident_after + inc >= budget:
                STATS['snap_capacity_retries'] += 1
                snapped, info = snap_groups(decision.ranges, resident, keep_out, structural, images, expand_text=True)
                plan = ledger.plan(snapped, protected=mod._union(protected, pin))
        except mod.ResidencyError as exc:
            raise mod.HistoryValidationError('kvq4 snap: %s' % exc) from exc
        if rec.get('triggered') and plan.resident_after + inc >= budget:
            raise mod.HistoryCapacityError('kvq4: the group-snapped plan leaves no output-token capacity', receipt=rec)
        suffix = _merge(decision.suffix_ranges)
        new_suffix = _merge(r for a, b in suffix for r in mod._intersections(a, b, snapped))
        history = _merge(_subtract(snapped, new_suffix))
        STATS['snaps'] += 1
        STATS['snap_groups'] += info['groups']
        STATS['snap_shrunk'] += info['shrunk']
        STATS['snap_expanded'] += info['expanded']
        STATS['snap_orphans'] += info['orphans']
        STATS['snap_struct_removed'] += info['struct_removed']
        STATS['snap_images_kept'] += info['images_kept']
        STATS['snap_rounds_max'] = max(STATS['snap_rounds_max'], info['rounds'])
        STATS['last_snap'] = dict(info, removed_before=int(decision.eviction_plan.removed_tokens), removed_after=int(plan.removed_tokens))
        rec.update(total_removed_tokens=plan.removed_tokens, resident_tokens_after=plan.resident_after,
                   resident_tokens_after_input=plan.resident_after + inc, minimum_output_capacity=budget - plan.resident_after - inc,
                   combined_ranges=[list(p) for p in snapped], history_ranges=[list(p) for p in history],
                   suffix_ranges=[list(p) for p in new_suffix], quad_snap=STATS['last_snap'])
        return mod.HistoryDecision(tuple(snapped), tuple(history), tuple(new_suffix), plan, rec)

    mod.plan_history_eviction = plan_history_eviction


# ------------------------------------------------------------------------------------------------ promptmove
def _patch_promptmove(pm):
    import time
    import torch
    from . import kvq4_kernels as Q
    ROW = Q.ROW

    def _context(runner):
        ctx = pm._CTX.get(id(runner))
        if ctx is not None:
            return ctx
        if os.environ.get('ARC3_PREFIX_SHARE', '1') != '0':
            raise RuntimeError('promptmove: needs ARC3_PREFIX_SHARE=0 (a shared prefix page would move every lane)')
        sfc = runner.compilation_config.static_forward_context
        gidx = {}
        for i, g in enumerate(runner.kv_cache_config.kv_cache_groups):
            for n in g.layer_names:
                gidx[n] = i
        owners = []
        for name, layer in sfc.items():
            if not name.endswith('.attn'):
                continue
            if pm._drafter_out_of_pool(name):
                pm.STATS['skipped_drafter'] = name
                continue
            cache = pm._tensor(getattr(layer, 'kv_cache', None))
            if cache is None or cache.dtype != torch.uint8 or cache.ndim != 4 or cache.shape[3] != ROW:
                continue
            cname = name[:-len('.attn')] + '.indexer.compressed_key_cache'
            ccache = pm._tensor(getattr(sfc.get(cname), 'kv_cache', None)) if cname in sfc else None
            if ccache is not None and (ccache.dtype != torch.bfloat16 or ccache.ndim != 4 or ccache.shape[1] * 4 != cache.shape[2]):
                raise RuntimeError('promptmove: unexpected compressed key cache for %s: %s' % (name, tuple(ccache.shape)))
            owners.append(dict(name=name, layer=layer, cache=cache, g=gidx[name], page=int(cache.shape[2]), heads=int(cache.shape[1]),
                               cname=cname, ccache=ccache, cg=gidx.get(cname), crows=int(ccache.shape[1]) if ccache is not None else 0))
        if not owners:
            raise RuntimeError('promptmove: no QSA owners found')
        rot = None
        for m in runner.model.modules():
            if getattr(m, 'rotary_dim', None) == 64 and getattr(m, 'head_size', None) == 256 and hasattr(m, 'base'):
                rot = m
                break
        if rot is None or not getattr(rot, 'is_neox_style', True):
            raise RuntimeError('promptmove: attention rotary embedding (rotary_dim 64, NeoX) not found')
        inv = (1.0 / (float(rot.base) ** (torch.arange(0, 64, 2, dtype=torch.float) / 64))).to(torch.float64)
        from . import kvlut_kernels
        lut = torch.tensor([-v for v in reversed(kvlut_kernels.LUTP)] + list(kvlut_kernels.LUTP), dtype=torch.float32)
        dev = owners[0]['cache'].device
        ctx = pm._CTX[id(runner)] = dict(owners=owners, inv=inv, lut=lut.to(dev), dev=dev, scratch=None, ref=None, q4=True)
        pm.STATS['owners'] = len(owners)
        pm.STATS['owners_without_compressed'] = [o['name'] for o in owners if o['ccache'] is None]
        return ctx

    def quad_index(block_ids, quads, page, dev):
        """(block, first row) of the quads (quad q = rows 4q .. 4q + 3)."""
        rows = np.asarray(quads, dtype=np.int64) * 4
        return pm._main_index(block_ids, rows, page, dev)

    def read_k(cache, blk, off):
        """K bytes of quads [n, H, 532]: codes 0..511, then K scales 1024..1043 (blk, off: each quad's page and first row)."""
        rows = torch.stack([cache[blk, :, off + i] for i in range(4)], 2).reshape(len(blk), cache.shape[1], 4 * ROW)
        return torch.cat([rows[..., :512], rows[..., 1024:1044]], -1)

    def write_k(cache, blk, off, rec):
        rows = torch.stack([cache[blk, :, off + i] for i in range(4)], 2).reshape(len(blk), cache.shape[1], 4 * ROW)
        rows[..., :512] = rec[..., :512]
        rows[..., 1024:1044] = rec[..., 512:532]
        rows = rows.reshape(len(blk), cache.shape[1], 4, ROW)
        for i in range(4):
            cache[blk, :, off + i] = rows[:, :, i]

    def decode(ctx, rec):
        """K quad records [n, H, 532] -> stored (centered, Hadamard-domain) rows [4n, H, 256] FP32."""
        n, h = rec.shape[0], rec.shape[1]
        codes = rec[..., :512].reshape(n, h, 4, 128).long()
        lv = torch.stack((ctx['lut'][codes & 15], ctx['lut'][codes >> 4]), dim=-1).reshape(n, h, 4, 16, 16)
        bs = rec[..., 512:528].contiguous().view(torch.float8_e4m3fn).float()            # [n, h, 16]
        hs = rec[..., 528:532].contiguous().view(torch.float32)                          # [n, h, 1]
        vals = lv * (bs * hs)[:, :, None, :, None]
        return vals.reshape(n, h, 4, 256).permute(0, 2, 1, 3).reshape(4 * n, h, 256)

    def encode(ctx, owner, kfull, role):
        """The quad store of BF16 K rows [4n, H, 256] through a scratch quad page; returns the K records [n, H, 532]."""
        n4, page, heads = kfull.shape[0], owner['page'], owner['heads']
        nb = (n4 + page - 1) // page
        s = ctx['scratch']
        if s is None or s.shape[0] < nb or s.shape[1] != heads or s.shape[2] != page:
            s = ctx['scratch'] = torch.empty((nb, heads, page, ROW), dtype=torch.uint8, device=ctx['dev'])
            ctx['scratch_meta'] = dict(
                tr=torch.zeros(page * nb, dtype=torch.int32, device=ctx['dev']),
                meta=torch.tensor([[-1, 0, -1]], dtype=torch.int32, device=ctx['dev']),
                bt=torch.arange(nb, dtype=torch.int32, device=ctx['dev'])[None].contiguous(),
                buf=torch.zeros((8, heads, 256), dtype=torch.bfloat16, device=ctx['dev']),
                tag=torch.full((8,), -1, dtype=torch.int32, device=ctx['dev']))
        sm = ctx['scratch_meta']
        slots = torch.arange(n4, dtype=torch.int64, device=ctx['dev'])
        mu = role[2]
        mk, mv = (mu[0], mu[1]) if mu is not None else (None, None)
        seq = torch.tensor([n4], dtype=torch.int32, device=ctx['dev'])
        Q.store(kfull, kfull, s, slots, sm['tr'][:n4], sm['meta'], sm['bt'], seq, sm['buf'], sm['buf'], sm['tag'], mk, mv)
        quads = torch.arange(n4 // 4, dtype=torch.int64, device=ctx['dev'])
        return read_k(s, quads * 4 // page, (quads * 4) % page)

    def _capture(ctx, block_ids, R):
        t0 = time.perf_counter()
        if R % 4:
            STATS['pmove_refused'] += 1
            raise RuntimeError('kvq4 promptmove: the block (%d rows) is not whole quads' % R)
        out = {}
        G = (R + 3) // 4
        for o in ctx['owners']:
            blk, off = quad_index(block_ids[o['g']], np.arange(R // 4), o['page'], ctx['dev'])
            k = read_k(o['cache'], blk, off).to('cpu').pin_memory()
            c = None
            if o['ccache'] is not None:
                cblk, coff = pm._main_index(block_ids[o['cg']], np.arange(G), o['crows'], ctx['dev'])
                c = o['ccache'][cblk, coff, 0, :].to('cpu').pin_memory()
            out[o['name']] = (k, c)
        pm.STATS['captures'] += 1
        pm.STATS['capture_ms'] += 1000 * (time.perf_counter() - t0)
        return out

    def _move(ctx, block_ids, orig, rows, src_pos, ranks, src_gid, delta):
        rows = np.asarray(rows, dtype=np.int64)
        src_pos = np.asarray(src_pos, dtype=np.int64)
        k = len(rows)
        if k % 4 or (k and (np.any(rows != np.arange(k)) or np.any(src_pos[0::4] % 4) or
                            np.any(src_pos.reshape(-1, 4) != src_pos[0::4, None] + np.arange(4)))):
            STATS['pmove_refused'] += 1
            raise RuntimeError('kvq4 promptmove: the moved rows are not whole quads (k %d)' % k)
        cs = pm._angles(ctx, delta)
        writes = []
        from .kvrot import rotate_rows
        for o in ctx['owners']:
            role = pm._role(o)
            k0, c0 = orig[o['name']]
            rec = k0[torch.from_numpy(src_pos[0::4] // 4)].to(ctx['dev'], non_blocking=True)
            x = rotate_rows(decode(ctx, rec))
            mu = role[2]
            if mu is not None:
                x = x + mu[0][None]
            kb = encode(ctx, o, pm._rope_(x, cs).to(torch.bfloat16), role)
            blk, off = quad_index(block_ids[o['g']], np.arange(k // 4), o['page'], ctx['dev'])
            c = cblk = coff = None
            if c0 is not None:
                c = pm._rope_(c0[torch.from_numpy(np.asarray(src_gid, dtype=np.int64))].to(ctx['dev'], non_blocking=True).float(), cs)
                c = c.to(torch.bfloat16)
                cblk, coff = pm._main_index(block_ids[o['cg']], ranks, o['crows'], ctx['dev'])
            writes.append((o, blk, off, kb.clone(), cblk, coff, c))
        for o, blk, off, kb, cblk, coff, c in writes:
            write_k(o['cache'], blk, off, kb)
            if c is not None:
                o['ccache'][cblk, coff, 0, :] = c
        if pm.VERIFY:
            pm.STATS['verify_moves'] += 1
            for o, blk, off, kb, cblk, coff, c in writes:
                if not torch.equal(read_k(o['cache'], blk, off), kb) or (c is not None and not torch.equal(o['ccache'][cblk, coff, 0, :], c)):
                    pm.STATS['verify_failures'] += 1

    pm._context, pm._capture, pm._move = _context, _capture, _move
    pm.STATS['kvq4'] = True


PATCHERS = {QSA: patch_qsa, MR: install_model_runner, MS: install_model_state,
            'engine_eviction_async': ('source', transform_eviction_async)}
PLANNER_PATCHERS = {'history_eviction': patch_history_eviction}
