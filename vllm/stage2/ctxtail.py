"""stage2_v1 'ctxtail' part (2026-10-04 04:05; user ~03:50: make every per-turn step depend only on resident and new
tokens, before the 1-hour runs; with option 3 there is no rebase before 2M, so a session's logical history keeps
growing and production's per-turn work grew with it).

Production rebuilds a session's whole history at every worker staging (each resume, each re-add after an eviction) and
at every eviction transaction:
  engine core  LogicalState.context: a copy of every logical token id and a deep copy of every image placement, shipped
               to the worker inside the request's sampling params (staging) or the eviction wire;
  worker       validation compares the full id list with its trusted copy and copies it, loops over every image and
               deep-copies them, and resumefast's MRoPE cache compares the full id list and image list again.
At 2M logical tokens that is >= ~45 ms per staging on the engine step's critical path.

Here contexts carry only what the worker may not have yet, and the worker checks only that:
  engine core  LogicalState.context returns logical_token_ids = ids[s:] and logical_image_features = images[i:], where
               s / i are the lengths of the last context the worker consumed through a NewRequestData (recorded where
               the scheduler builds it: every staging context reaches the worker that way), plus arc3_ids_base = s,
               arc3_ids_len = L, arc3_imgs_base = i, arc3_imgs_len and arc3_ids_digest (BLAKE2b-128 over all L ids as
               int32, kept incrementally). The tracking dict lives on the request (eviction proxies are shallow copies
               and share it). A context replaced before scheduling is never lost: the next one starts from the same s.
  worker       a per-request store keeps the full id list, the image placements, the digest state and an MRoPE buffer,
               all append-only. A context may overlap the store (an eviction wire extends it before the engine's s
               moves): the overlap must equal the store, the rest is appended, and the digest of the result must equal
               the engine's. A base beyond the store, a mismatch or a digest difference raises EvictionRunnerError
               (production's failure path). Validation is production's, restricted to what can change: the new ids and
               images, the resident positions, the dense ids at the resident positions, the computed frontier, the PLE
               predecessors and the deleted groups. MRoPE rows: the stored buffer, extended by the model's positions of
               the appended ids (continuing from the running maximum, as resumefast's cache), gathered at the resident
               positions. The QSA lookup is resumefast's.
  Eviction transactions (worker_compact_async) call eviction_runner.validate_context, patched the same way.
Per-staging cost: O(resident + new). Store memory (ids, positions): ~100 MB per session at 2M, the same data resumefast's
cache held. Rebase restarts are new requests (fresh store, base 0). Contexts without the arc3 keys take production's
path unchanged (both halves are installed together by the part).
ARC3_S2_CTXTAIL_VERIFY=N: every N-th staging also rebuilds the full context and checks validation, MRoPE rows and the
lookup against production's functions (a difference raises).

2026-10-15 (item 5 / prestage): the staging's CPU part is compute(), shared by stage_tail and the 'prestage' part (which
runs it on the worker while the previous step executes; stage_tail then takes its result). The resident positions are
converted to an array once (_positions: production's _ints accept / reject set) and reused for the group count, the MRoPE
gather and the lookup (three np.fromiter passes of ~0.8 ms each at 54K positions before)."""
import bisect
import os
import time

import numpy as np

PINS = {'engine_eviction': '1c49513acc9543b6f2392cdcf1f8c6eb172f84f343d853ff2464314f558ae2b5',
        'eviction_runner': '03ecae34a987b0b3203df42cd609f41c051226371c517ae85d69591851253ed8',
        'vllm.v1.core.sched.output': '51643386fab3a56b7747aa784ad657833966ebe823acee1f37222e33c1e5696a'}
CONTEXT_KEY = 'arc3_eviction_context'
VERIFY = int(os.environ.get('ARC3_S2_CTXTAIL_VERIFY', '0') or 0)
STATS = dict(contexts=0, context_ids_sent=0, context_ids_total=0, consumed=0, stagings=0, staging_ms=0.0,
             validations=0, appended_ids=0, overlap_ids=0, freed=0, verified=0, errors=0, last_error=None, prestaged=0)
_TICK = [0]
_REG = {}                                   # id(store id list) -> (store, runner, request id): the eviction path's lookup
_MOD = []                                   # the worker's eviction_runner module (set when it is patched)


def _save(n):
    """Receipts every 200 calls (install.save rate-limits writes to one a minute without force)."""
    if n % 200 == 1:
        try:
            from . import install
            install.save(force=False)
        except Exception:                               # noqa: BLE001  receipts are diagnostics
            pass


def _parts():
    from . import install
    return install.PARTS


def _hasher():
    import hashlib
    return hashlib.blake2b(digest_size=16)


def _id_bytes(ids):
    return np.asarray(ids, dtype=np.int32).tobytes()


# ------------------------------------------------------------------------------------------------ engine core
def patch_engine_eviction(mod):
    import copy

    def context(self, request):
        d = request.__dict__
        ct = d.get('_arc3_ct')
        if ct is None:
            ct = d['_arc3_ct'] = dict(sent_ids=0, sent_imgs=0, hasher=_hasher(), hashed=0)
        ids, imgs = self.logical_ids, self.logical_image_features
        L, IL = len(ids), len(imgs)
        s, i = ct['sent_ids'], ct['sent_imgs']
        if s > L or i > IL or ct['hashed'] > L:
            raise mod.EngineEvictionError('ctxtail: the worker copy is longer than the logical history')
        if ct['hashed'] < L:
            ct['hasher'].update(_id_bytes(ids[ct['hashed']:L]))
            ct['hashed'] = L
        STATS['contexts'] += 1
        _save(STATS['contexts'])
        STATS['context_ids_sent'] += L - s
        STATS['context_ids_total'] += L
        return dict(logical_token_ids=list(ids[s:L]), resident_positions=self.ledger.resident_positions(),
                    logical_computed=request.num_computed_tokens + self.deleted_tokens,
                    logical_image_features=copy.deepcopy(imgs[i:IL]), eviction_revision=self.eviction_revision,
                    deleted_tokens=self.deleted_tokens, deleted_groups=self.deleted_groups,
                    arc3_ids_base=s, arc3_ids_len=L, arc3_ids_digest=ct['hasher'].hexdigest(),
                    arc3_imgs_base=i, arc3_imgs_len=IL)

    mod.LogicalState.context = context


def patch_sched_output(mod):
    cls = mod.NewRequestData
    original = cls.from_request.__func__

    def from_request(c, request, block_ids, prefill_token_ids=None):
        out = original(c, request, block_ids, prefill_token_ids)
        try:
            ctx = (getattr(request.sampling_params, 'extra_args', None) or {}).get(CONTEXT_KEY)
            ct = request.__dict__.get('_arc3_ct')
            if isinstance(ctx, dict) and 'arc3_ids_len' in ctx and ct is not None:
                ct['sent_ids'], ct['sent_imgs'] = ctx['arc3_ids_len'], ctx['arc3_imgs_len']
                STATS['consumed'] += 1
        except Exception as exc:                        # noqa: BLE001  the next context then overlaps (still exact)
            STATS['errors'] += 1
            STATS['last_error'] = repr(exc)[:300]
        return out

    cls.from_request = classmethod(from_request)


# ------------------------------------------------------------------------------------------------ worker
def _store(runner, req_id, create):
    store = runner.__dict__.setdefault('_arc3_ctxtail', {})
    states = runner.__dict__.get('_arc3_eviction_states', {})
    for rid in [r for r in store if r not in states and r != req_id]:
        _REG.pop(id(store[rid]['ids']), None)
        del store[rid]
        STATS['freed'] += 1
    st = store.get(req_id)
    if st is None and create:
        st = store[req_id] = dict(ids=[], imgs=[], hasher=_hasher(), buf=np.empty((3, 0), dtype=np.int32), buf_n=0, max=-1)
        _REG[id(st['ids'])] = (st, runner, req_id)
    return st


def _resolve(mod, runner, req_id, context):
    """Append the context's new ids and images to the request's store (after the overlap and digest checks); returns
    the store. Nothing is modified when a check fails."""
    E = mod.EvictionRunnerError
    states = runner._arc3_eviction_states
    st = _store(runner, req_id, create=False)
    if st is None:
        if req_id in states:
            raise E('ctxtail: worker eviction state without a history store')
        if context['arc3_ids_base'] != 0 or context['arc3_imgs_base'] != 0:
            raise E("ctxtail: a session's first context must start at 0")
        st = _store(runner, req_id, create=True)
    base, L, ib, IL = (context['arc3_ids_base'], context['arc3_ids_len'], context['arc3_imgs_base'], context['arc3_imgs_len'])
    tail, new_imgs = context['logical_token_ids'], context['logical_image_features']
    n, k = len(st['ids']), len(st['imgs'])
    if (type(base) is not int or type(L) is not int or not 0 <= base <= n or L < n or not isinstance(tail, list)
            or len(tail) != L - base or type(ib) is not int or type(IL) is not int or not 0 <= ib <= k or IL < k
            or not isinstance(new_imgs, list) or len(new_imgs) != IL - ib):
        raise E('ctxtail: context base %r/%r (length %r/%r) does not extend the worker copy %d/%d' % (base, ib, L, IL, n, k))
    tail = mod._ints(tail, 'Logical token IDs')
    if tail[:n - base] != st['ids'][base:n] or new_imgs[:k - ib] != st['imgs'][ib:k]:
        raise E('Worker original token prefix was rewritten')
    add, add_imgs = tail[n - base:], new_imgs[k - ib:]
    h = st['hasher'].copy()
    if add:
        h.update(_id_bytes(add))
    if h.hexdigest() != context['arc3_ids_digest']:
        raise E('ctxtail: logical id digest differs from the engine')
    prev_end = st['imgs'][-1]['offset'] + st['imgs'][-1]['length'] if st['imgs'] else 0
    for image in add_imgs:
        if not isinstance(image, dict) or set(image) != {'offset', 'length', 'grid_thw'}:
            raise E('Original image layout supports offset/length/grid_thw only')
        offset, length = image['offset'], image['length']
        grid = mod._ints(image['grid_thw'], 'Image grid dimensions')
        if (type(offset) is not int or type(length) is not int or length <= 0 or offset < prev_end or offset + length > L
                or len(grid) != 3 or grid[0] != 1 or min(grid) <= 0):
            raise E('Invalid or overlapping original image layout')
        prev_end = offset + length
    st['ids'].extend(add)
    st['imgs'].extend(dict(x, grid_thw=list(x['grid_thw'])) for x in add_imgs)
    st['hasher'] = h
    STATS['appended_ids'] += len(add)
    STATS['overlap_ids'] += n - base
    return st


def _positions(mod, values):
    """mod._ints(values, 'Resident positions', ordered=True) and the positions as an int64 array (None if empty): the same
    accept / reject set (list or tuple, exact int type, nonnegative, strictly increasing) and a list copy."""
    if not isinstance(values, (list, tuple)):
        raise mod.EvictionRunnerError('Resident positions must contain nonnegative integers')
    n = len(values)
    if not n:
        return [], None
    if set(map(type, values)) != {int}:
        raise mod.EvictionRunnerError('Resident positions must contain nonnegative integers')
    try:
        arr = np.fromiter(values, dtype=np.int64, count=n)
    except OverflowError:                               # beyond int64: production's own checks and error
        return mod._ints(values, 'Resident positions', ordered=True), None
    if int(arr.min()) < 0:
        raise mod.EvictionRunnerError('Resident positions must contain nonnegative integers')
    if n > 1 and not bool((arr[1:] > arr[:-1]).all()):
        raise mod.EvictionRunnerError('Resident positions must be strictly increasing')
    return list(values), arr


def _check(mod, st, context, *, dense_ids, dense_computed, limits, allow_closed_frontier, out=None):
    """Production's _validate_context checks that depend on more than the store (resident-sized); the checked context
    (store lists referenced, not copied). out (a dict) receives the positions array as 'arr'."""
    E = mod.EvictionRunnerError
    ids = st['ids']
    L = len(ids)
    if context['arc3_ids_len'] != L:
        raise E('ctxtail: store length differs from the context')
    positions, arr = _positions(mod, context['resident_positions'])
    if not positions or positions[-1] != L - 1:
        raise E('Original/resident stream must preserve the accepted tail')
    if L > limits['max_logical_tokens'] or len(positions) > limits['max_resident_tokens']:
        raise E('Original/resident position capacity exceeded')
    computed = context['logical_computed']
    if type(allow_closed_frontier) is not bool:
        raise E('Closed-frontier allowance must be an explicit boolean')
    last_computed = L if allow_closed_frontier else L - 1
    if type(computed) is not int or not 0 <= computed <= last_computed:
        raise E('Original computed frontier is outside the pending stream')
    for name in ('eviction_revision', 'deleted_tokens', 'deleted_groups'):
        if type(context[name]) is not int or context[name] < 0:
            raise E(f'Invalid {name}')
    if context['deleted_tokens'] != L - len(positions):
        raise E('Deleted token count disagrees with retained positions')
    if dense_ids is not None:
        try:
            if set(map(type, dense_ids)) != {int}:
                raise TypeError('non-int dense ids')
            ok = len(dense_ids) == len(positions) and list(dense_ids) == mod._gather(ids, positions)
        except (TypeError, ValueError, OverflowError, IndexError):
            ok = False
        if not ok:
            raise E('Dense prompt IDs differ from the original retained subsequence')
    computed_dense = bisect.bisect_left(positions, computed)
    if dense_computed is not None and dense_computed != computed_dense:
        raise E('Dense/original computed frontiers disagree')
    if positions[computed_dense:] != list(range(computed, L)):
        raise E('A deletion intersects the newly forwarded contiguous tail')
    ngram = limits.get('ngram_context_len', 2)
    predecessors = list(range(max(0, computed - ngram), computed))
    if predecessors and positions[max(0, computed_dense - len(predecessors)):computed_dense] != predecessors:
        raise E('Eviction changes PLE immediate predecessor tokens')
    if arr is None:                                     # positions beyond int64 (never valid here): production's conversion
        arr = np.fromiter(positions, dtype=np.int64, count=len(positions))
    kk = bisect.bisect_left(positions, 4 * (computed // 4))
    q = arr[:kk] // 4
    distinct = int(np.count_nonzero(q[1:] != q[:-1])) + 1 if len(q) else 0
    if context['deleted_groups'] != computed // 4 - distinct:
        raise E('Deleted QSA group count disagrees with original groups')
    STATS['validations'] += 1
    checked = {key: value for key, value in context.items() if not key.startswith('arc3_')}
    checked.update(logical_token_ids=ids, resident_positions=positions, logical_image_features=st['imgs'])
    if out is not None:
        out['arr'] = arr
    return checked


def _full_context(st, context):
    out = {key: value for key, value in context.items() if not key.startswith('arc3_')}
    out.update(logical_token_ids=list(st['ids']), logical_image_features=[dict(x, grid_thw=list(x['grid_thw'])) for x in st['imgs']],
               resident_positions=list(context['resident_positions']))
    return out


def _mrope(mod, runner, st, positions, deleted_tokens, pos_arr=None):
    """(3, resident) int32 rows and the delta, from the store's buffer extended to the store length. pos_arr: positions as
    an int64 array, when the caller has it."""
    model = runner.model_state.model
    ids, imgs, L = st['ids'], st['imgs'], len(st['ids'])
    n = st['buf_n']
    if L > n:
        j = bisect.bisect_left([x['offset'] for x in imgs], n)
        if j > 0 and imgs[j - 1]['offset'] + imgs[j - 1]['length'] > n:
            n, j = 0, 0                                 # an image straddles the buffer end: rebuild (never expected)
            st['max'] = -1
        suffix = dict(logical_token_ids=ids[n:L], logical_image_features=[dict(x, offset=x['offset'] - n) for x in imgs[j:]])
        arr, _ = mod._mrope_full(model, suffix)
        base = st['max'] + 1
        buf = st['buf']
        if L > buf.shape[1]:
            grown = np.empty((3, max(L + 8192, 2 * buf.shape[1])), dtype=np.int32)
            grown[:, :n] = buf[:, :n]
            buf = st['buf'] = grown
        buf[:, n:L] = arr + base
        st['max'] = base + int(arr.max())
        st['buf_n'] = L
    pos = pos_arr if pos_arr is not None else np.fromiter(positions, dtype=np.int64, count=len(positions))
    return st['buf'][:, pos], st['max'] + 1 - L + deleted_tokens


def compute(mod, runner, req_id, dense_ids, dense_computed, context):
    """The staging's CPU part: store append (append-only, digest-checked), validation, MRoPE rows, QSA lookup table. No GPU
    call and no worker state other than the request's append-only store and MRoPE buffer. Returns
    (checked, rows, delta, table, cutoff, padding)."""
    from . import resumefast
    limits = mod._limits(runner)
    st = _resolve(mod, runner, req_id, context)
    out = {}
    checked = _check(mod, st, context, dense_ids=dense_ids, dense_computed=dense_computed, limits=limits,
                     allow_closed_frontier=False, out=out)
    arr = out['arr']
    rows, delta = _mrope(mod, runner, st, checked['resident_positions'], checked['deleted_tokens'], arr)
    resumefast._POS.update(list=checked['resident_positions'], arr=arr)    # fast_lookup's identity-keyed conversion
    table, cutoff, padding = resumefast.fast_lookup(mod, checked)
    return checked, rows, delta, table, cutoff, padding


def prestage_compute(runner, req_id, dense_ids, dense_computed, context):
    """compute() for the 'prestage' part, on the worker while the previous step runs: only a resume of a session the worker
    already stages (its store exists; a fresh session's store is created at its staging), with an incremental context."""
    if not _MOD or not (isinstance(context, dict) and 'arc3_ids_base' in context):
        return None
    if req_id not in getattr(runner, '_arc3_eviction_states', {}) or req_id not in runner.__dict__.get('_arc3_ctxtail', {}):
        return None
    return compute(_MOD[0], runner, req_id, dense_ids, dense_computed, context)


def stage_tail(mod, runner, req_index, new_req_data, context):
    import torch
    from . import resumefast
    t0 = time.perf_counter()
    req_id = new_req_data.req_id
    limits = mod._limits(runner)
    existing = runner._arc3_eviction_states.get(req_id)
    pre = None
    if 'prestage' in _parts():
        from . import prestage
        pre = prestage.take(req_id, new_req_data, context)
    if pre is not None:
        checked, rows, delta, table, cutoff, padding = pre
        STATS['prestaged'] += 1
    else:
        checked, rows, delta, table, cutoff, padding = compute(mod, runner, req_id, new_req_data.prefill_token_ids,
                                                               new_req_data.num_computed_tokens, context)
    if existing is not None and checked['eviction_revision'] != existing['eviction_revision']:
        raise mod.EvictionRunnerError('Worker eviction revision changed without a committed transaction')
    resumefast.install_array_writes(runner.model_state.rope_state.prefill_positions)
    _TICK[0] += 1
    if VERIFY and _TICK[0] % VERIFY == 0:
        st = _store(runner, req_id, create=False)
        full = _full_context(st, context)
        ref = mod._validate_context(full, dense_ids=new_req_data.prefill_token_ids, dense_computed=new_req_data.num_computed_tokens,
                                    allow_closed_frontier=False, trusted_ids=None, **limits)
        got = dict(checked, logical_token_ids=list(checked['logical_token_ids']),
                   logical_image_features=[dict(x) for x in checked['logical_image_features']])
        if ref != got:
            raise mod.EvictionRunnerError('ctxtail: validated context differs from production')
        ref_rows, ref_delta = mod.original_mrope_rows(runner.model_state.model, full)
        if (rows.tolist(), delta) != (ref_rows, ref_delta):
            raise mod.EvictionRunnerError('ctxtail: MRoPE rows differ from the full construction')
        ref_table = mod.historical_virtual_lookup(full)
        if (list(np.asarray(table).tolist()), cutoff, padding) != (list(ref_table[0]), ref_table[1], ref_table[2]):
            raise mod.EvictionRunnerError('ctxtail: QSA lookup differs from production')
        STATS['verified'] += 1
    buffers = runner.model_state._arc3_eviction_buffers
    if cutoff > buffers.lookup.shape[1]:
        raise mod.EvictionRunnerError('QSA virtual mapping exceeds worker buffer width')
    with mod.publish_scalars(buffers, req_index, checked['deleted_tokens'], padding, cutoff):
        if cutoff:
            buffers.lookup[req_index, :cutoff].copy_(torch.from_numpy(np.ascontiguousarray(table, dtype=np.int32)).to(buffers.lookup.device))
    runner._arc3_eviction_states[req_id] = dict(checked, awaiting_readd=False, req_index=req_index)
    STATS['stagings'] += 1
    STATS['staging_ms'] += 1000 * (time.perf_counter() - t0)
    _save(STATS['stagings'])
    return rows, delta


def patch_eviction_runner(mod):
    original_stage, original_validate = mod._stage_request, mod.validate_context
    _MOD[:] = [mod]

    def _stage_request(runner, req_index, new_req_data, context):
        if isinstance(context, dict) and 'arc3_ids_base' in context:
            try:
                return stage_tail(mod, runner, req_index, new_req_data, context)
            except Exception as exc:                    # recorded, then production's failure path (the turn fails)
                STATS['errors'] += 1
                STATS['last_error'] = repr(exc)[:300]
                raise
        return original_stage(runner, req_index, new_req_data, context)

    def validate_context(context, *, dense_ids=None, dense_computed=None, max_resident_tokens, max_logical_tokens,
                         ngram_context_len=2, allow_closed_frontier=False, trusted_ids=None):
        """Incremental contexts outside staging (the eviction transaction's worker side): resolved against the store of
        the request whose trusted ids are passed (worker_compact_async passes them for its request)."""
        if not (isinstance(context, dict) and 'arc3_ids_base' in context):
            return original_validate(context, dense_ids=dense_ids, dense_computed=dense_computed,
                                     max_resident_tokens=max_resident_tokens, max_logical_tokens=max_logical_tokens,
                                     ngram_context_len=ngram_context_len, allow_closed_frontier=allow_closed_frontier,
                                     trusted_ids=trusted_ids)
        entry = _REG.get(id(trusted_ids)) if trusted_ids is not None else None
        if entry is None or entry[0]['ids'] is not trusted_ids:
            raise mod.EvictionRunnerError('ctxtail: no history store for this eviction context')
        st = _resolve(mod, entry[1], entry[2], context)
        limits = dict(max_resident_tokens=max_resident_tokens, max_logical_tokens=max_logical_tokens, ngram_context_len=ngram_context_len)
        return _check(mod, st, context, dense_ids=dense_ids, dense_computed=dense_computed, limits=limits,
                      allow_closed_frontier=allow_closed_frontier)

    mod._stage_request = _stage_request
    mod.validate_context = validate_context


PATCHERS = {'engine_eviction': patch_engine_eviction, 'eviction_runner': patch_eviction_runner,
            'vllm.v1.core.sched.output': patch_sched_output}
