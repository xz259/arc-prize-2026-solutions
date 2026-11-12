"""stage2_v1 'resumefast' part: the worker's resume admission without redundant list round trips (GAP_PLAN item 1,
2026-09-30).

steptimer v6 (g4g): a resume's worker add_requests takes 23.3 ms. The reducible part is production's eviction staging
(11.8 ms: context validation 5.6, MRoPE rows 2.2, QSA lookup 1.8) and the MRoPE staged write (3.0 ms: 3 x 34K Python
ints turned into a tensor at apply). Production's checks and values are kept; repeated conversions go:
- validation converts the resident positions to a numpy array once (production did it twice) and reuses it for the
  deleted-group count, the MRoPE gather and the QSA lookup. The distinct-group count is a run count on the sorted
  group ids instead of np.unique; identical, because the positions are first validated strictly increasing;
- the MRoPE rows stay a numpy array (no .tolist()). The rope state's staged-write tensor gets array-aware staging:
  segments concatenated in staging order and uploaded from numpy. Only that instance changes, because the block tables'
  fused writer reads the list form of its own tensors;
- the QSA lookup table is built and uploaded as numpy (for _stage_request; other callers keep production's function).
Exactness: the success path produces the same validated context, rows, delta and table values. Any failing or unusual
input re-runs production's function, which raises production's error.
ARC3_S2_RESUMEFAST_VERIFY=N: every N-th staging also runs production's validation, a full MRoPE construction and
production's lookup, and compares (a difference raises).
ARC3_S2_RESUMEFAST_SWITCH=<file> containing 'off' restores production's functions (A/B; the rope tensor keeps the
array-aware staging, which stages lists as arrays).

2026-10-03 (Fable's review): the per-request MRoPE cache is a bounded LRU (_cache_cap: max_num_reqs + 16 entries);
entries of finished games and of rebased sessions were never dropped (~10 MB each near a rebase)."""
import bisect
import copy
import os
import time

import numpy as np

STATS = dict(stages=0, validate_fast=0, validate_fallbacks=0, verified=0, array_writes=0, rope_installs=0, switch_off=0,
             mrope_cache_evictions=0, mrope_cache_max=0)
VERIFY = int(os.environ.get('ARC3_S2_RESUMEFAST_VERIFY', '0') or 0)
_SWITCH = dict(path=os.environ.get('ARC3_S2_RESUMEFAST_SWITCH'), t=0.0, off=False)
_POS = dict(list=None, arr=None)          # the last validated positions list and its int64 array (identity-keyed)
_TICK = [0]
_REQUIRED = {'logical_token_ids', 'resident_positions', 'logical_computed', 'eviction_revision', 'deleted_tokens',
             'deleted_groups', 'logical_image_features'}


def _off():
    s = _SWITCH
    if s['path'] is None:
        return False
    now = time.monotonic()
    if now - s['t'] > 1.0:
        s['t'] = now
        try:
            with open(s['path']) as f:
                s['off'] = f.read().strip() == 'off'
        except OSError:
            s['off'] = False
    return s['off']


def _pos_array(positions):
    if _POS['list'] is positions:
        return _POS['arr']
    arr = np.fromiter(positions, dtype=np.int64, count=len(positions))
    _POS.update(list=positions, arr=arr)
    return arr


def _distinct_sorted(values):
    """Number of distinct values of a non-decreasing int array (np.unique(values).size)."""
    n = len(values)
    return int(np.count_nonzero(values[1:] != values[:-1])) + 1 if n else 0


def fast_validate(mod, context, dense_ids, dense_computed, max_resident_tokens, max_logical_tokens, ngram_context_len,
                  allow_closed_frontier, trusted_ids):
    """Production's _validate_context success path, or None wherever production raises (the caller re-runs it)."""
    if not isinstance(context, dict) or not _REQUIRED <= set(context):
        return None
    ids = mod._ids_with_trusted_prefix(context['logical_token_ids'], trusted_ids)
    values = context['resident_positions']
    if not isinstance(values, (list, tuple)) or not values or set(map(type, values)) != {int}:
        return None
    try:
        arr = np.fromiter(values, dtype=np.int64, count=len(values))
    except OverflowError:
        return None
    if int(arr.min()) < 0 or (len(arr) > 1 and not bool((arr[1:] > arr[:-1]).all())):
        return None
    positions = list(values)
    if not ids or positions[-1] != len(ids) - 1:
        return None
    if len(ids) > max_logical_tokens or len(positions) > max_resident_tokens:
        return None
    computed = context['logical_computed']
    if type(allow_closed_frontier) is not bool:
        return None
    last_computed = len(ids) if allow_closed_frontier else len(ids) - 1
    if type(computed) is not int or not 0 <= computed <= last_computed:
        return None
    for name in ('eviction_revision', 'deleted_tokens', 'deleted_groups'):
        if type(context[name]) is not int or context[name] < 0:
            return None
    if context['deleted_tokens'] != len(ids) - len(positions):
        return None
    efficient_copy = mod._efficiency_enabled('ARC3_EFFICIENCY_VALIDATED_CONTEXT_COPY')
    if dense_ids is not None:
        try:
            if set(map(type, dense_ids)) != {int}:
                raise TypeError('non-int dense ids')
            if mod._ol_flag('ARC3_OL_FASTPATH', '1'):
                ok = len(dense_ids) == len(positions) and list(dense_ids) == mod._gather(ids, positions)
            else:
                dense = np.fromiter(dense_ids, dtype=np.int64, count=len(dense_ids))
                ok = (dense.shape[0] == len(positions) and bool(np.array_equal(
                    dense, np.fromiter(ids, dtype=np.int64, count=len(ids))[arr])))
        except (TypeError, ValueError, OverflowError):
            ok = list(dense_ids) == [ids[p] for p in positions]
        if not ok:
            return None
    computed_dense = bisect.bisect_left(positions, computed)
    if dense_computed is not None and dense_computed != computed_dense:
        return None
    if positions[computed_dense:] != list(range(computed, len(ids))):
        return None
    predecessors = list(range(max(0, computed - ngram_context_len), computed))
    if predecessors and positions[max(0, computed_dense - len(predecessors)):computed_dense] != predecessors:
        return None
    k = bisect.bisect_left(positions, 4 * (computed // 4))
    if context['deleted_groups'] != computed // 4 - _distinct_sorted(arr[:k] // 4):
        return None
    images = context['logical_image_features']
    if not isinstance(images, list):
        return None
    previous_end = 0
    for image in images:
        if not isinstance(image, dict) or set(image) != {'offset', 'length', 'grid_thw'}:
            return None
        offset, length = image['offset'], image['length']
        grid = mod._ints(image['grid_thw'], 'Image grid dimensions')
        if (type(offset) is not int or type(length) is not int or length <= 0
                or offset < previous_end or offset + length > len(ids)
                or len(grid) != 3 or grid[0] != 1 or min(grid) <= 0):
            return None
        previous_end = offset + length
    if not efficient_copy:
        out = copy.deepcopy(context)
    else:
        memo = {}
        for name, validated in (('logical_token_ids', ids), ('resident_positions', positions)):
            original = context[name]
            if id(original) not in memo:
                if type(original) is list:
                    memo[id(original)] = validated
                elif type(original) is tuple:
                    memo[id(original)] = original
        out = copy.deepcopy(context, memo)
    _POS.update(list=out['resident_positions'], arr=arr)
    return out


def fast_lookup(mod, context):
    """historical_virtual_lookup with a numpy table (same values; production's function returns a list)."""
    if mod._efficiency_enabled('ARC3_EFFICIENCY_IDENTITY_LOOKUP') and context['deleted_tokens'] == 0:
        if context['deleted_groups'] != 0:
            raise mod.EvictionRunnerError('Identity mapping cannot delete QSA groups')
        return np.empty(0, dtype=np.int64), 0, 0
    positions = context['resident_positions']
    pos = _pos_array(positions)
    q = pos // 4
    first = np.ones(len(q), dtype=bool)
    first[1:] = q[1:] != q[:-1]                     # q is non-decreasing: group starts
    groups = q[first]
    full = np.full((len(groups), 4), -1, dtype=np.int64)
    full[np.cumsum(first) - 1, pos % 4] = np.arange(len(pos))    # == searchsorted(groups, q) for sorted groups
    table = full.ravel()
    padding = context['deleted_tokens'] - 4 * context['deleted_groups']
    cutoff = len(positions) + padding
    if cutoff < 0 or cutoff > len(table):
        raise mod.EvictionRunnerError('Invalid QSA virtual frontier')
    return table[:cutoff], cutoff, padding


def install_array_writes(t):
    """Array-aware staging on one StagedWriteTensor instance (the rope's prefill positions)."""
    if t.__dict__.get('_s2_array_writes'):
        return
    import torch
    from vllm.v1.worker.gpu import buffer_utils as bu
    np_dtype = {torch.int32: np.int32, torch.int64: np.int64, torch.float32: np.float32}[t.dtype]
    state = dict(segs=[], total=0)
    if t._staged_write_contents:                    # writes staged before the install keep their place
        state['segs'].append(np.asarray(t._staged_write_contents, dtype=np_dtype))
        state['total'] = len(t._staged_write_contents)
        t._staged_write_contents.clear()

    def stage_write(index, start, x):
        assert index >= 0
        assert start >= 0
        n = len(x)
        if n == 0:
            return
        t._staged_write_indices.append(index)
        t._staged_write_starts.append(start)
        if isinstance(x, np.ndarray):
            state['segs'].append(x)
            STATS['array_writes'] += 1
        else:
            state['segs'].append(np.asarray(x, dtype=np_dtype))
        state['total'] += n
        t._staged_write_cu_lens.append(state['total'])

    def stage_write_elem(index, x):
        assert index >= 0
        t._staged_write_indices.append(index)
        t._staged_write_starts.append(0)
        state['segs'].append(np.asarray([x], dtype=np_dtype))
        state['total'] += 1
        t._staged_write_cu_lens.append(state['total'])

    def clear_staged_writes():
        t._staged_write_indices.clear()
        t._staged_write_starts.clear()
        t._staged_write_contents.clear()
        t._staged_write_cu_lens.clear()
        state['segs'] = []
        state['total'] = 0

    def apply_write():
        n = len(t._staged_write_indices)
        if n == 0:
            return
        indices_uva = t.write_indices.copy_to_uva(t._staged_write_indices)
        starts_uva = t.write_starts.copy_to_uva(t._staged_write_starts)
        cu_lens_uva = t.write_cu_lens.copy_to_uva(t._staged_write_cu_lens)
        contents = np.concatenate(state['segs']).astype(np_dtype, copy=False)
        write_contents = bu.async_tensor_h2d(contents, device=t.device, dtype=t.dtype)
        bu._apply_write_kernel[(n,)](t.gpu, t.gpu.stride(0), indices_uva, starts_uva, write_contents, cu_lens_uva, None,
                                     BLOCK_SIZE=1024, MULTI_GROUP=False)
        clear_staged_writes()

    t.stage_write, t.stage_write_elem = stage_write, stage_write_elem
    t.apply_write, t.clear_staged_writes = apply_write, clear_staged_writes
    t._s2_array_writes = True
    STATS['rope_installs'] += 1


def _cache_cap(runner):
    """Entries kept in the MRoPE cache (2026-10-03, Fable's review): ARC3_S2_RESUMEFAST_CACHE_CAP, default the
    runner's max_num_reqs + 16. Each entry holds a session's logical token ids and a (3, L + 8192) int32 buffer (~10 MB
    near a rebase), and entries of finished games and of rebased sessions (a new request id) were never dropped."""
    cap = int(os.environ.get('ARC3_S2_RESUMEFAST_CACHE_CAP', '0') or 0)
    if cap > 0:
        return cap
    n = getattr(runner, 'max_num_reqs', None) or getattr(getattr(runner, 'req_states', None), 'max_num_reqs', None)
    return int(n or 64) + 16


def _lru_trim(caches, cap):
    """Drop least recently used entries beyond cap. The cache is only a fast path: a dropped live session is rebuilt
    by _mrope_full at its next resume (exact either way)."""
    while len(caches) > cap:
        del caches[next(iter(caches))]
        STATS['mrope_cache_evictions'] += 1
    STATS['mrope_cache_max'] = max(STATS['mrope_cache_max'], len(caches))


def patch_eviction_runner(mod):
    """Called from the guards part's eviction_runner patcher (one patcher per module)."""
    import torch
    orig_validate, orig_mrope, orig_stage = mod._validate_context, mod._mrope_rows, mod._stage_request
    orig_lookup = mod.historical_virtual_lookup

    def _validate_context(context, *, dense_ids=None, dense_computed=None, max_resident_tokens, max_logical_tokens,
                          ngram_context_len=2, allow_closed_frontier=False, trusted_ids=None):
        kw = dict(dense_ids=dense_ids, dense_computed=dense_computed, max_resident_tokens=max_resident_tokens,
                  max_logical_tokens=max_logical_tokens, ngram_context_len=ngram_context_len,
                  allow_closed_frontier=allow_closed_frontier, trusted_ids=trusted_ids)
        if _off():
            STATS['switch_off'] += 1
            return orig_validate(context, **kw)
        try:
            out = fast_validate(mod, context, dense_ids, dense_computed, max_resident_tokens, max_logical_tokens,
                                ngram_context_len, allow_closed_frontier, trusted_ids)
        except Exception:
            out = None
        if out is None:
            STATS['validate_fallbacks'] += 1
            return orig_validate(context, **kw)     # production's own verdict and error
        STATS['validate_fast'] += 1
        return out

    def _mrope_rows(runner, req_id, context):
        if _off() or not mod._ol_flag('ARC3_OL_FASTPATH', '1'):
            return orig_mrope(runner, req_id, context)
        install_array_writes(runner.model_state.rope_state.prefill_positions)
        model = runner.model_state.model
        caches = runner.__dict__.setdefault('_arc3_mrope_cache', {})
        ids = context['logical_token_ids']; images = context['logical_image_features']; L = len(ids)
        entry = caches.pop(req_id, None)                # re-inserted below: dict order = least recently used first
        if entry is not None:
            n, k = len(entry['ids']), len(entry['images'])
            if not (n <= L and len(images) >= k and ids[:n] == entry['ids'] and images[:k] == entry['images']
                    and all(image['offset'] >= n for image in images[k:])):
                entry = None
        if entry is None:
            arr, _ = mod._mrope_full(model, context)
            buf = np.empty((3, max(L + 8192, 2 * L)), dtype=np.int32); buf[:, :L] = arr
            entry = dict(ids=list(ids), images=copy.deepcopy(images), buf=buf, max=int(arr.max()) if L else -1)
        elif L > n:
            suffix = dict(logical_token_ids=ids[n:],
                          logical_image_features=[dict(image, offset=image['offset'] - n) for image in images[k:]])
            arr, _ = mod._mrope_full(model, suffix)
            base = entry['max'] + 1
            buf = entry['buf']
            if L > buf.shape[1]:
                grown = np.empty((3, max(L + 8192, 2 * buf.shape[1])), dtype=np.int32); grown[:, :n] = buf[:, :n]
                buf = entry['buf'] = grown
            buf[:, n:L] = arr + base
            entry['max'] = base + int(arr.max())
            entry['ids'].extend(ids[n:]); entry['images'].extend(copy.deepcopy(images[k:]))
        caches[req_id] = entry
        _lru_trim(caches, _cache_cap(runner))
        rows = entry['buf'][:, _pos_array(context['resident_positions'])]     # (3, resident) int32, a fresh array
        delta = entry['max'] + 1 - L + context['deleted_tokens']
        if mod._ol_flag('ARC3_OL_FASTPATH_VERIFY', '0') and (rows.tolist(), delta) != tuple(mod.original_mrope_rows(model, context)):
            raise mod.EvictionRunnerError('Cached MRoPE rows differ from the full construction')
        return rows, delta

    def _stage_request(runner, req_index, new_req_data, context):
        if _off():
            return orig_stage(runner, req_index, new_req_data, context)
        STATS['stages'] += 1
        profiling = os.environ.get('ARC3_EVICTION_PROFILE', '0') == '1'     # production's staging profile, kept
        started = phase_start = time.perf_counter() if profiling else 0
        phases = {}
        limits = mod._limits(runner)
        existing = runner._arc3_eviction_states.get(new_req_data.req_id)
        checked = mod.validate_context(context, dense_ids=new_req_data.prefill_token_ids,
                                       dense_computed=new_req_data.num_computed_tokens,
                                       trusted_ids=mod.trusted_ids(existing), **limits)
        if existing is not None:
            old = existing['logical_token_ids']
            if checked['logical_token_ids'][:len(old)] != old:
                raise mod.EvictionRunnerError('Worker original token prefix was rewritten')
            if checked['eviction_revision'] != existing['eviction_revision']:
                raise mod.EvictionRunnerError('Worker eviction revision changed without a committed transaction')
        if profiling:
            now = time.perf_counter(); phases['validate_context_seconds'] = now - phase_start; phase_start = now
        rows, delta = mod._mrope_rows(runner, new_req_data.req_id, checked)
        if profiling:
            now = time.perf_counter(); phases['original_mrope_seconds'] = now - phase_start; phase_start = now
        table, cutoff, padding = fast_lookup(mod, checked)
        if profiling:
            now = time.perf_counter(); phases['lookup_build_seconds'] = now - phase_start; phase_start = now
        _TICK[0] += 1
        if VERIFY and _TICK[0] % VERIFY == 0:
            ref = orig_validate(context, dense_ids=new_req_data.prefill_token_ids,
                                dense_computed=new_req_data.num_computed_tokens,
                                trusted_ids=mod.trusted_ids(existing), **limits)
            if ref != checked:
                raise mod.EvictionRunnerError('resumefast: validated context differs from production')
            ref_rows, ref_delta = mod.original_mrope_rows(runner.model_state.model, checked)
            got_rows = rows.tolist() if isinstance(rows, np.ndarray) else rows
            if (got_rows, delta) != (ref_rows, ref_delta):
                raise mod.EvictionRunnerError('resumefast: MRoPE rows differ from the full construction')
            if (table.tolist(), cutoff, padding) != tuple(orig_lookup(checked)):
                raise mod.EvictionRunnerError('resumefast: QSA lookup differs from production')
            STATS['verified'] += 1
        buffers = runner.model_state._arc3_eviction_buffers
        if cutoff > buffers.lookup.shape[1]:
            raise mod.EvictionRunnerError('QSA virtual mapping exceeds worker buffer width')
        with mod.publish_scalars(buffers, req_index, checked['deleted_tokens'], padding, cutoff):
            if cutoff:
                buffers.lookup[req_index, :cutoff].copy_(
                    torch.from_numpy(np.ascontiguousarray(table, dtype=np.int32)).to(buffers.lookup.device))
        runner._arc3_eviction_states[new_req_data.req_id] = dict(checked, awaiting_readd=False, req_index=req_index)
        if profiling:
            now = time.perf_counter(); phases['buffer_staging_seconds'] = now - phase_start
            phases['total_seconds'] = now - started
            totals = getattr(runner.model_state, '_arc3_eviction_staging_profile', None)
            if totals is None:
                totals = {'completed_calls': 0, 'identity_calls': 0, 'lookup_entries_uploaded': 0, 'seconds': {}}
                runner.model_state._arc3_eviction_staging_profile = totals
            totals['completed_calls'] += 1
            totals['identity_calls'] += int(checked['deleted_tokens'] == 0)
            totals['lookup_entries_uploaded'] += cutoff
            for key, value in phases.items():
                totals['seconds'][key] = totals['seconds'].get(key, 0) + value
        return rows, delta

    mod._validate_context = _validate_context
    mod._mrope_rows = _mrope_rows
    mod._stage_request = _stage_request
