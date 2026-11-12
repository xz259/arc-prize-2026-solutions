"""stage2_v1 'kvring' part (item 4a, 2026-10-15; the user's KV window decision 13:0x UTC: BF16-256; design
harnesspod_20261013/kvring/KVRING_DESIGN_2026-10-15.md, reviewed by IE): the newest N resident rows of every TARGET QSA owner are
read in BF16 from a per-request ring; older rows read today's lutc (dual storage, kvring_kernels). The MTP drafter's owner is
untouched. Needs 'kvrot' with a lut target format (installed before this part; kvrot's per-layer role and means are reused).

Worker state:
  ring ids   one per request (req_id), assigned at its first scheduled step, kept across the eviction runner's remove + re-add
             (its _READDING set), freed at the request's real removal; none free -> ring_id -1 (all lutc), counted.
  off        deleted_tokens from the eviction runner's state (the full count since 10-16; reduced mod R << 20 only for int32): the
             main owners compact per token (engine_eviction's computed_keep_indices), so a compaction below the window renumbers
             every window row by the tokens it deleted; ring slot = (row + off) mod R = the row's original position mod R.
             (Corrected 10-15 after G2a2: the first build used 4 x deleted_groups, the indexer's virtual space.)
  rows, tags (10-16, after g2k4a: 49 window rows at the K 4 context cap read another row's slot; design reviewed by IE) a token's
             row comes from its lutc slot (slot_mapping) inverted through the step's batch-ordered page table (the attention
             metadata's; only the request's valid prefix ceil(seq_len / PAGE)), not from seq_len - q_len; every slot carries the tag
             row + off of the row it holds, and the reader takes a window row from the ring only when the tag matches (else lutc,
             counted by cause over all layers: ring_fallback_cleared (tag -1) / ring_fallback_aliased). Tags are cleared when a ring id is assigned
             and when a compaction deletes rows inside the window (window_deletions; deletions below the window keep row + off of
             every window row). ring_unmapped_tokens / ring_rows_beyond_seq count real tokens without a row / at or beyond seq_len.
  floor      rows at or below it read lutc although inside the window: set at ring assignment when the request already has rows
             (floor = computed - 1: rows that predate the ring; IE) and by promptmove (it re-RoPEs the leading block's rows in the lutc
             pages only; expected never). A compaction does not decrement a floor: conservative (more rows read lutc), never unsafe -
             do not "fix" it by subtracting D.
  per step   the store, the compaction and both readers of a step read one ring_meta (ring_id, off, floor) and one seq_lens, staged
             before the forward from the eviction state as it is at prepare_attn; a commit between steps changes the next step's pair.
  ring_meta  int32 [max requests, 3] = (ring_id, off, floor) per batch row, staged from pinned memory in the model state's
             prepare_attn (before the forward; persistent, so FULL decode graphs replay it); capture and padded rows get -1.
  slots      (10-16, the stale-read fix; user: fix the aliased reads although they are exact) each ring holds R = N + SLACK slots
             for an N-row window: slot = (row + off) mod R. A step the scheduler makes shorter than its predecessor (partial drafts
             behind a prefill chunk under the token budget, a 1-row resumed chunk) lowers seq_len by <= K - 1 and leaves the
             previous step's rows beyond the new frontier; with R = N those rows held the slots of the newly exposed window-bottom
             rows (aliased fallbacks: tk34b 31, tk33 60 in 3 h). With R = N + SLACK a window row and any row up to SLACK beyond the
             frontier have distinct slots; the tag check stays as the backstop.
  exact      (10-28, IE's idea 2 lever; kvring_exact.py) ARC3_S2_KVRING_EXACT = M > 0 (the two-kernel reader only; async eviction
             only): every request's ring gets RS = R + M data slots, the last M an exact set of rows outside the window (pinned rows
             below ARC3_S2_KVRING_EXACT_PIN, others promoted at ring exit by selection rate >= ARC3_S2_KVRING_EXACT_RATE), keyed by
             the current row through a row -> slot map that every compaction commit re-keys from the plan's keep rows; cleared at
             ring assignment; a promptmove rotates the exact copies of the moved block's rows by the move's increment (rerope,
             10-28; before, it cleared the set); after each re-key or rerope, the request's next eager step checks every exact entry
             against the lutc row its tag names (IE: mis-keys dropped and counted). Clears counted by reason (exact_clears_*).
             M = 0: unchanged (R-strided rings, none of kvring_exact runs).
Env: ARC3_S2_KVRING_N (default 256), ARC3_S2_KVRING_SLACK (default 8), ARC3_S2_KVRING_EXACT (default 0) / _PIN (64) / _RATE (0.02) /
_MAXROWS (default ARC3_CONTEXT: the max model length). STATS: the receipt (the ring_* and exact_* counters refreshed every 1,000 steps
without a sync)."""
import os

import torch

N = int(os.environ.get('ARC3_S2_KVRING_N', '256'))
SLACK = int(os.environ.get('ARC3_S2_KVRING_SLACK', '8'))
R = N + SLACK                                                  # ring slots per request (the window stays N rows)
assert N > 0 and SLACK >= 0
DEBUG_SH = int(os.environ.get('ARC3_S2_KVRING_DEBUG', '0'))       # G2a test servers only: shadow rows per request (0 = off)
FAST = os.environ.get('ARC3_S2_KVRING_FAST', '0') == '1'         # 10-27: the two-kernel reader (kvring_fast.py), the 82kr1632 tier
assert not (FAST and DEBUG_SH), 'kvring: the two-kernel reader has no debug shadow (ARC3_S2_KVRING_DEBUG needs ARC3_S2_KVRING_FAST=0)'
DEBUG_FILE = os.environ.get('ARC3_S2_KVRING_DEBUG_FILE', '')
EXACT = int(os.environ.get('ARC3_S2_KVRING_EXACT', '0'))           # 10-28: exact slots per request (kvring_exact.py); 0 = off
EXACT_PIN = int(os.environ.get('ARC3_S2_KVRING_EXACT_PIN', '64'))
EXACT_RATE = float(os.environ.get('ARC3_S2_KVRING_EXACT_RATE', '0.02'))
EXACT_MAXROWS = int(os.environ.get('ARC3_S2_KVRING_EXACT_MAXROWS', os.environ.get('ARC3_CONTEXT', '0')) or 0)
RS = R + EXACT                                                   # data slots per request: the recency ring, then the exact set
if EXACT:
    assert FAST, 'kvring: the exact set needs the two-kernel reader (ARC3_S2_KVRING_FAST=1)'
    assert 0 < EXACT_PIN < EXACT < 32768, 'kvring: ARC3_S2_KVRING_EXACT_PIN must lie in (0, ARC3_S2_KVRING_EXACT)'
    assert EXACT_MAXROWS > 0, 'kvring: the exact set needs ARC3_S2_KVRING_EXACT_MAXROWS or ARC3_CONTEXT (the max model length)'
    assert os.environ.get('ARC3_EVICTION_MODE', 'fused') == 'async', \
        'kvring: the exact set is re-keyed at async compaction commits only (ARC3_EVICTION_MODE=async)'
STATS = dict(n=N, r=R, rs=RS, fast=FAST, exact=EXACT, exact_pin=EXACT_PIN, exact_rate=EXACT_RATE, exact_maxrows=EXACT_MAXROWS,
             exact_promotions=0, exact_hits=0, exact_nonwindow=0, exact_refusals=0, exact_pinned_writes=0, exact_below_rate=0,
             exact_promotions_bulk=0, exact_checked=0, exact_miskeys=0, exact_check_max_rel=0.0, exact_checks=0,
             exact_rekeys=0, exact_clears_assign=0, exact_clears_promptmove=0, exact_clears_unexpected=0, exact_bytes=0,
             exact_reropes=0, exact_rerope_rows=0, ring_layers=0, ring_calls=0, ids_assigned=0, ids_freed=0, ids_kept_readd=0, no_ring_steps=0,
             floor_events=0, max_requests=0, ring_bytes=0, errors=0, last_error=None, promptmove_guard=False,
             ring_unmapped_tokens=0, ring_fallback_cleared=0, ring_fallback_aliased=0, ring_rows_beyond_seq=0,
             tag_clears_assign=0, window_deletions=0, commits_seen=0, commit_hook=False)
# Post-run skim, nonzero means investigate (IE 10-16): ring_unmapped_tokens, ring_fallback_aliased, ring_rows_beyond_seq,
# window_deletions; ring_fallback_cleared is expected only after a window deletion or a ring assigned to a request with rows.
OFF_MOD = R << 20                                                # off stays an int32 (a multiple of R: the slot is unchanged)
_RINGS = {}             # layer name -> (ring_k, ring_v)
_EXACT = {}             # layer name -> kvring_exact.ExactState (EXACT > 0)
_LAYERS = {}            # layer name -> (kv heads, head dim, device): target lut owners seen (rings allocated outside capture)
_IDS = {}               # req_id -> ring id
_FREE = []
_FLOOR = {}             # req_id -> floor row
_META = {}              # 'dev' / 'host': [max requests, 3] int32
_DEBUG = {}             # G2a: shadows per layer, expected seq lens, counters, first mismatch


_MAKE_DUMMY = __import__('re').compile(r'req_\d+_[0-9a-f]+$')   # InputBatch.make_dummy: f'req_{i}_{random_uuid()}' (G2a3)


def _err(exc):
    STATS['errors'] += 1
    STATS['last_error'] = repr(exc)[:400]


def _ensure_meta(max_reqs, device):
    # Host side is pageable on purpose: a pageable H2D copy is staged by the driver before it returns, so the next step may
    # refill the host tensor while the device copy is still queued (async scheduling runs the CPU ahead of the GPU).
    if 'dev' not in _META:
        _META['dev'] = torch.full((max_reqs, 3), -1, dtype=torch.int32, device=device)
        _META['host'] = torch.full((max_reqs, 3), -1, dtype=torch.int32)
        _META['host_np'] = _META['host'].numpy()          # shares memory: per-row writes without torch indexing (~4 us each)
        _META['slot_rows'] = torch.full((max_reqs * R,), -1, dtype=torch.int32, device=device)   # tag (row + off) of each slot
        # unmapped, fallback cleared / aliased, beyond seq; then the exact set's xstats [10] (kvring_exact module docstring)
        _META['stats'] = torch.zeros(14, dtype=torch.int64, device=device)
        _META['xstats'] = _META['stats'][4:]
        _META['stats_host'] = torch.zeros(14, dtype=torch.int64).pin_memory()
        _META['calls'] = 0
        _FREE.extend(range(max_reqs - 1, -1, -1))
        STATS['max_requests'] = max_reqs
    _allocate_rings()
    return _META['dev']


def _allocate_rings():
    # Rings must exist before any CUDA-graph capture (an allocation inside a capture comes from the graph's private pool).
    if 'dev' not in _META:
        return
    for name, (kvh, d, device) in _LAYERS.items():
        if name in _RINGS:
            continue
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError('kvring: ring of %s would be allocated inside a CUDA-graph capture' % name)
        shape = (_META['dev'].shape[0] * RS, kvh, d)
        _RINGS[name] = (torch.zeros(shape, dtype=torch.bfloat16, device=device),
                        torch.zeros(shape, dtype=torch.bfloat16, device=device))
        if EXACT:
            from .kvring_exact import ExactState
            _EXACT[name] = ExactState(_META['dev'].shape[0], EXACT_MAXROWS, N, R, EXACT, EXACT_PIN, EXACT_RATE, device,
                                      shared=_META.get('exact_shared'))
            _META['exact_shared'] = _EXACT[name].shared
            STATS['exact_bytes'] += _EXACT[name].nbytes()
        _META.setdefault('count_layer', name)
        STATS['ring_layers'] += 1
        STATS['ring_bytes'] += 2 * _RINGS[name][0].numel() * 2
        if DEBUG_SH:
            dshape = (_META['dev'].shape[0] * DEBUG_SH, kvh, d)
            _DEBUG.setdefault('shadows', {})[name] = (torch.zeros(dshape, dtype=torch.bfloat16, device=device),
                                                      torch.zeros(dshape, dtype=torch.bfloat16, device=device))
            if 'counters' not in _DEBUG:
                rows = _META['dev'].shape[0]
                _DEBUG['counters'] = torch.zeros(8, dtype=torch.int64, device=device)
                _DEBUG['first'] = torch.zeros(8, dtype=torch.int32, device=device)
                _DEBUG['expected'] = torch.full((rows,), -1, dtype=torch.int32, device=device)
                _DEBUG['expected_host'] = torch.full((rows,), -1, dtype=torch.int32)
                _DEBUG['steps'] = 0
                _DEBUG['diag'] = torch.zeros(8, dtype=torch.int64, device=device)        # the store's row diagnostics (one layer)
                _DEBUG['events'] = torch.full((64, 8), -1, dtype=torch.int32, device=device)
                _DEBUG['diag_layer'] = _META.get('count_layer')
                _DEBUG['alias_events'] = torch.full((256, 8), -1, dtype=torch.int32, device=device)   # aliased fallbacks (10-16)
                _DEBUG['alias_count'] = torch.zeros(1, dtype=torch.int64, device=device)


def _ring(layer):
    return _RINGS.get(getattr(layer, 'layer_name', id(layer)))


def _readding():
    try:
        import eviction_runner
        return eviction_runner._READDING.get()
    except Exception:
        return frozenset()


def _stats_into(v):
    STATS.update(ring_unmapped_tokens=int(v[0]), ring_fallback_cleared=int(v[1]), ring_fallback_aliased=int(v[2]),
                 ring_rows_beyond_seq=int(v[3]))
    if EXACT and len(v) >= 14:
        STATS.update(exact_promotions=int(v[4]), exact_hits=int(v[5]), exact_nonwindow=int(v[6]), exact_refusals=int(v[7]),
                     exact_pinned_writes=int(v[8]), exact_below_rate=int(v[9]), exact_promotions_bulk=int(v[10]),
                     exact_checked=int(v[11]), exact_miskeys=int(v[12]), exact_check_max_rel=int(v[13]) / 1e6)


def _export_fallbacks():
    """Every 1,000 steps: the previous copy of the device counters into STATS (if it has landed), then a new non-blocking copy."""
    _META['calls'] += 1
    if _META['calls'] % 1000 or torch.cuda.is_current_stream_capturing():
        return
    ev = _META.get('stats_event')
    if ev is not None:
        if not ev.query():
            return
        _stats_into(_META['stats_host'].tolist())
    _META['stats_host'].copy_(_META['stats'], non_blocking=True)
    ev = torch.cuda.Event()
    ev.record()
    _META['stats_event'] = ev


def patch_qsa(mod):
    cls = mod.Qwen3_8FlashNextQSAFlashAttentionImpl
    inner = cls.forward_qsa                                  # kvrot's (asserted by install order)
    if getattr(inner, '__qualname__', '').split('.')[0] != 'patch_qsa':
        raise RuntimeError('kvring must be installed after kvrot')

    def forward_qsa(self, layer, query, key, value, kv_cache, attn_metadata, output, token_to_req,
                    output_scale=None, output_block_scale=None):
        role = getattr(layer, '_arc3_kvrot_role', None)
        if role is not None and not role[1] and role[0].startswith('lut'):
            name = getattr(layer, 'layer_name', id(layer))
            if name not in _LAYERS:
                _LAYERS[name] = (key.shape[1], key.shape[2], key.device)
                _allocate_rings()
        ring = _ring(layer) if role is not None and not role[1] and role[0].startswith('lut') else None
        if ring is None or 'dev' not in _META:
            return inner(self, layer, query, key, value, kv_cache, attn_metadata, output, token_to_req,
                         output_scale, output_block_scale)
        from . import kvring_kernels
        from .kvrot import rotate_rows
        fmt, draft, mu = role
        n = attn_metadata.num_actual_tokens
        output.zero_()
        if n:
            meta = _META['dev']
            rk, rv = ring
            kmean, vmean = (mu[0], mu[1]) if mu is not None else (None, None)
            page = kv_cache.shape[2]                             # rows per page (the lutc store's PAGE)
            counted = name == _META.get('count_layer')            # store-side counters on one layer (every layer writes the same rows)
            diag = _DEBUG['diag'] if DEBUG_SH and counted else None
            xs = _EXACT.get(name) if EXACT else None
            if EXACT and xs is None:
                raise RuntimeError('kvring: no exact state for %s' % name)
            if xs is not None and xs.kmean is None:
                xs.kmean = kmean                                  # the layer's K mean, for a promptmove's rerope
            if xs is not None:                                    # 10-28: the store with the exact set's promotions (kvring_exact)
                xs.store(key[:n], value[:n], rk, rv, token_to_req[:n], attn_metadata.seq_lens, meta, kmean, vmean,
                         attn_metadata.slot_mapping, attn_metadata.block_table, page, attn_metadata.query_start_loc,
                         stats=_META['stats'] if counted else None, xstats=_META['xstats'])
                if xs.pending and not torch.cuda.is_current_stream_capturing():   # IE: the consistency check after a re-key
                    hn = _META['host_np']
                    pairs = [(b, int(hn[b, 0])) for b in range(min(hn.shape[0], attn_metadata.seq_lens.shape[0]))
                             if int(hn[b, 0]) in xs.pending]
                    if pairs:
                        xs.check(kv_cache, attn_metadata.block_table, attn_metadata.seq_lens, [b for b, _ in pairs],
                                 [r for _, r in pairs], rk, rv, _META['xstats'])
                        xs.pending.difference_update(r for _, r in pairs)
                        STATS['exact_checks'] += 1
            else:
                kvring_kernels.ring_store(key[:n], value[:n], rk, rv, token_to_req[:n], attn_metadata.query_start_loc,
                                          attn_metadata.seq_lens, meta, kmean, vmean, n=N, r=R, slot_rows=_META['slot_rows'],
                                          slot_mapping=attn_metadata.slot_mapping, block_table=attn_metadata.block_table, page=page,
                                          diag=diag, events=_DEBUG['events'] if diag is not None else None,
                                          stats=_META['stats'] if counted else None)
            debug = None
            if DEBUG_SH:
                sk, sv = _DEBUG['shadows'][getattr(layer, 'layer_name', id(layer))]
                kvring_kernels.shadow_store(key, value, sk, sv, token_to_req, attn_metadata.seq_lens, meta, kmean, vmean, DEBUG_SH, n,
                                            attn_metadata.slot_mapping, attn_metadata.block_table, page)
                debug = dict(shadow_k=sk, shadow_v=sv, sh=DEBUG_SH, expected=_DEBUG['expected'],
                             counters=_DEBUG['counters'], first=_DEBUG['first'], slot_rows=_META['slot_rows'],
                             alias_events=_DEBUG['alias_events'], alias_count=_DEBUG['alias_count'])
            if xs is not None:                                    # the two-kernel reader with the exact hits
                xs.attention(rotate_rows(query[:n]), kv_cache, layer.topk_indices_buffer[:n], attn_metadata.block_table,
                             token_to_req[:n], attn_metadata.query_start_loc, attn_metadata.seq_lens, rk, rv, meta, output[:n],
                             vmean=mu[2] if mu is not None else None, stats=_META['stats'], xstats=_META['xstats'])
            elif FAST:                                            # the same inputs and result up to the summation order
                from . import kvring_fast
                kvring_fast.attention(rotate_rows(query[:n]), kv_cache, layer.topk_indices_buffer[:n], attn_metadata.block_table,
                                      token_to_req[:n], attn_metadata.seq_lens, rk, rv, meta, output[:n],
                                      vmean=mu[2] if mu is not None else None, n=N, r=R, slot_rows=_META['slot_rows'],
                                      stats=_META['stats'])
            else:
                kvring_kernels.attention(rotate_rows(query[:n]), kv_cache, layer.topk_indices_buffer[:n], attn_metadata.block_table,
                                         token_to_req[:n], attn_metadata.seq_lens, rk, rv, meta, output[:n],
                                         vmean=mu[2] if mu is not None else None, n=N, r=R, debug=debug,
                                         slot_rows=_META['slot_rows'],
                                         stats=_META['stats'])   # read-side counters on every layer (IE: each has its own top-k)
            rotate_rows(output[:n], out=output[:n])
            STATS['ring_calls'] += 1
        return output

    cls.forward_qsa = forward_qsa
    try:
        from . import promptmove
        install_promptmove_guard(promptmove)
        STATS['promptmove_guard'] = True
    except Exception as exc:
        _err(exc)


def _debug_step(input_batch, for_capture):
    """G2a: the scheduler's own row count per batch row (num_computed + num_scheduled), staged for the kernels' boundary check;
    every 200 steps the counters are read (a host sync, test servers only) into STATS and ARC3_S2_KVRING_DEBUG_FILE."""
    eh = _DEBUG['expected_host']
    eh.fill_(-1)
    if not for_capture:
        n = int(input_batch.num_reqs)
        eh[:n] = torch.from_numpy((input_batch.num_computed_tokens_np[:n] + input_batch.num_scheduled_tokens[:n]).astype('int32'))
    _DEBUG['expected'].copy_(eh, non_blocking=True)
    _DEBUG['steps'] += 1
    if _DEBUG['steps'] % 200 == 0 and not for_capture and not torch.cuda.is_current_stream_capturing():
        debug_flush()


def debug_flush():
    if 'counters' not in _DEBUG:
        return
    c = _DEBUG['counters'].tolist()
    f = _DEBUG['first'].tolist()
    STATS.update(debug_content_mismatches=c[0], debug_seq_len_above_scheduler=c[1], debug_window_rows_checked=c[2],
                 debug_output_mismatches=c[3], debug_seq_len_below_scheduler=c[4], debug_slot_holds_other_row=c[5],
                 debug_first=dict(req=f[0], row=f[1], slot=f[2], off=f[3]) if f[4] else None, debug_steps=_DEBUG['steps'])
    _stats_into(_META['stats'].tolist())
    if 'diag' in _DEBUG:
        d = _DEBUG['diag'].tolist()
        STATS.update(debug_rows_disagree=d[0], debug_pad_tokens=d[1], debug_unmapped=d[2], debug_row_beyond_seq=d[3],
                     debug_rows_written=d[4], debug_tokens_seen=d[5], debug_events_total=d[7])
        names = ('kind', 'req', 'j', 'old_row', 'row', 'seq_len', 'q_len', 'slot')
        STATS['debug_events'] = [dict(zip(names, e)) for e in _DEBUG['events'][:min(d[7], 64)].tolist()]
    if 'alias_count' in _DEBUG:
        na = int(_DEBUG['alias_count'].item())
        anames = ('ring_id', 'row', 'expected_tag', 'found_tag', 'seq_len', 'boundary', 'off', 'sched_expected')
        STATS['debug_alias_total'] = na
        STATS['debug_alias_events'] = [dict(zip(anames, e)) for e in _DEBUG['alias_events'][:min(na, 256)].tolist()]
        STATS['debug_ring_owner'] = {str(rid): req[-12:] for req, rid in list(_IDS.items())}
    if DEBUG_FILE:
        import json
        tmp = DEBUG_FILE + '.tmp'
        with open(tmp, 'w') as fh:
            json.dump(STATS, fh, default=str)
        os.replace(tmp, DEBUG_FILE)


def _debug_compact(req_id, keep_rows):
    """G2a: apply the eviction runner's own main-KV keep plan (new row i <- old row keep_rows[i]) to this request's shadow rows
    and to its ring's slot tags, so both stay addressed by the current row after a compaction. The checks then test the ring's
    offset against the runner's real row movement, not against the offset formula itself."""
    rid = _IDS.get(req_id)
    if rid is None or not _DEBUG.get('shadows'):
        STATS['debug_compactions_skipped'] = STATS.get('debug_compactions_skipped', 0) + 1
        return
    sh = DEBUG_SH
    device = _META['slot_rows'].device
    keep = torch.as_tensor(list(keep_rows), dtype=torch.int64, device=device)
    if keep.numel() and int(keep.max()) >= sh:
        STATS['debug_compactions_clipped'] = STATS.get('debug_compactions_clipped', 0) + 1
        keep = keep[keep < sh]
    n = keep.numel()
    base = rid * sh
    for sk, sv in _DEBUG['shadows'].values():
        sk[base:base + n] = sk[base + keep]                  # advanced indexing gathers a copy first
        sv[base:base + n] = sv[base + keep]
    STATS['debug_compactions'] = STATS.get('debug_compactions', 0) + 1     # (tags are row + off: no renumbering since 10-16)


def _window_intact(keep_rows, old_rows):
    """A compaction keeps every window row (the last min(N, old_rows) rows) iff the sorted keep plan ends with exactly those rows."""
    if old_rows is None:
        return False
    old_rows = int(old_rows)
    k = min(N, old_rows)
    if k <= 0:
        return True
    if len(keep_rows) < k:
        return False
    return int(keep_rows[-k]) == old_rows - k and int(keep_rows[-1]) == old_rows - 1


def _clear_tags(rid):
    with torch.inference_mode():                             # dr2: in-place writes outside inference mode can raise
        _META['slot_rows'][rid * R:(rid + 1) * R].fill_(-1)
        for xs in _EXACT.values():                            # each layer's own recency tags (kvring_exact)
            xs.clear_tags(rid)


def _clear_exact(rid, reason):
    """The request ring's exact set on every layer: empty; reason (counted, IE): assign (a new request), promptmove, unexpected (a
    commit that cannot be re-keyed, or an error)."""
    with torch.inference_mode():
        for xs in _EXACT.values():
            xs.clear_exact(rid)
    STATS['exact_clears_' + reason] += 1


def _rekey_exact(rid, plan):
    """A compaction commit: re-key the request ring's exact entries on every layer from the plan's keep rows (kvring_exact);
    anything unexpected clears the set instead (a stale key could name another row)."""
    try:
        old_rows = getattr(plan, 'old_rows', None) if plan is not None else None
        if old_rows is None or not _EXACT:
            _clear_exact(rid, 'unexpected')
            return
        from .kvring_exact import keep_tensor
        keep = keep_tensor(plan.keep_rows, _META['slot_rows'].device)
        with torch.inference_mode():
            for xs in _EXACT.values():
                xs.rekey(rid, keep, int(old_rows))
                xs.pending.add(rid)
        STATS['exact_rekeys'] += 1
    except Exception as exc:
        _err(exc)
        _clear_exact(rid, 'unexpected')


def _install_commit_hook():
    """Worker side: wrap the async eviction commit (production's ARC3_EVICTION_MODE=async path; the core sends the function by
    reference, so the worker resolves this module attribute). A compaction that deletes rows inside a request's window clears the
    request's tags (rows below the deleted range would otherwise compute another row's tag); debug servers also mirror the runner's
    keep plan onto the row-addressed shadow."""
    import engine_eviction_async as eea
    orig = eea.worker_commit_async
    if getattr(orig, '_arc3_kvring_hook', False):
        return

    def worker_commit_async(worker, request_id):
        plan = None
        entry = (getattr(worker.model_runner, '_arc3_async_evictions', None) or {}).get(request_id)
        if entry is not None:
            try:
                plan = next((item['plan'] for group in entry['wire']['groups'] for item in group['copies']
                             if item['kind'] == 'main'), None)
            except Exception as exc:
                _err(exc)
        result = orig(worker, request_id)
        try:
            if plan is not None and isinstance(result, dict) and result.get('worker_synchronized'):
                STATS['commits_seen'] += 1
                rid = _IDS.get(request_id)
                if rid is not None and not _window_intact(plan.keep_rows, getattr(plan, 'old_rows', None)):
                    _clear_tags(rid)
                    STATS['window_deletions'] += 1
                if rid is not None and EXACT:
                    _rekey_exact(rid, plan)
                if DEBUG_SH:
                    _debug_compact(request_id, plan.keep_rows)
            elif EXACT and entry is not None and isinstance(result, dict) and result.get('worker_synchronized'):
                rid = _IDS.get(request_id)                       # a commit without a main plan: nothing to re-key from
                if rid is not None:
                    _clear_exact(rid, 'unexpected')
        except Exception as exc:
            _err(exc)
            if EXACT and _IDS.get(request_id) is not None:
                _clear_exact(_IDS[request_id], 'unexpected')
        return result

    worker_commit_async.__name__ = worker_commit_async.__qualname__ = 'worker_commit_async'
    worker_commit_async.__module__ = orig.__module__
    worker_commit_async._arc3_kvring_hook = True
    eea.worker_commit_async = worker_commit_async
    STATS['commit_hook'] = True


def install_model_state(module):
    cls = module.MambaHybridModelState
    orig_prepare, orig_remove = cls.prepare_attn, cls.remove_request

    def prepare_attn(self, input_batch, cudagraph_mode, block_tables, slot_mappings, attn_groups, kv_cache_config,
                     for_capture=False):
        try:
            dev = _ensure_meta(int(self.max_num_reqs), getattr(self, 'device', None) or torch.device('cuda'))
            host = _META['host']
            hn = _META['host_np']
            hn.fill(-1)
            if not for_capture:
                runner = getattr(self, '_arc3_eviction_runner', None)
                states = getattr(runner, '_arc3_eviction_states', {}) if runner is not None else {}
                n = int(input_batch.num_reqs)
                for i, req_id in enumerate(input_batch.req_ids[:n]):
                    rid = _IDS.get(req_id)             # assigned at add_request
                    if rid is None:
                        if not (req_id.startswith('_dummy_req_') or _MAKE_DUMMY.match(req_id)):   # vLLM's dummy batches
                            STATS['no_ring_steps'] += 1
                            seen = STATS.setdefault('no_ring_first', [])
                            if len(seen) < 8 and req_id[:48] not in seen:
                                seen.append(req_id[:48])
                        continue
                    st = states.get(req_id) or {}
                    off = int(st.get('deleted_tokens', 0)) % OFF_MOD   # main rows shift by every deleted token (module docstring)
                    hn[i, 0] = rid
                    hn[i, 1] = off
                    hn[i, 2] = _FLOOR.get(req_id, -1)
            dev.copy_(host, non_blocking=True)        # prepare_attn runs outside any capture (a pageable H2D copy would fail inside one)
            if not for_capture:
                _export_fallbacks()
            if DEBUG_SH and 'expected' in _DEBUG:
                _debug_step(input_batch, for_capture)
        except Exception as exc:                              # never break the step: every row reads lutc
            _err(exc)
            if 'dev' in _META:
                _META['dev'].fill_(-1)
        return orig_prepare(self, input_batch, cudagraph_mode, block_tables, slot_mappings, attn_groups, kv_cache_config,
                            for_capture=for_capture)

    orig_add = cls.add_request

    def add_request(self, req_index, new_req_data):
        try:
            req_id = new_req_data.req_id
            if req_id not in _IDS:                     # every added request (vLLM's warmup requests are finished and freed)
                if 'dev' not in _META:
                    _ensure_meta(int(self.max_num_reqs), getattr(self, 'device', None) or torch.device('cuda'))
                if _FREE:
                    _IDS[req_id] = _FREE.pop()
                    STATS['ids_assigned'] += 1
                    _clear_tags(_IDS[req_id])                  # a previous owner's tags must never match this request's rows
                    STATS['tag_clears_assign'] += 1
                    if EXACT:
                        _clear_exact(_IDS[req_id], 'assign')   # nor its exact entries
                    # IE: rows that predate this ring (a request added with computed rows) read lutc until the window
                    # has left them; 0 for a new request (floor -1)
                    computed = int(getattr(new_req_data, 'num_computed_tokens', 0) or 0)
                    if computed > 0:
                        _FLOOR[req_id] = max(_FLOOR.get(req_id, -1), computed - 1)
                        STATS['floor_at_assignment'] = STATS.get('floor_at_assignment', 0) + 1
                else:
                    STATS['no_free_id'] = STATS.get('no_free_id', 0) + 1
        except Exception as exc:
            _err(exc)
        return orig_add(self, req_index, new_req_data)

    def remove_request(self, req_id):
        result = orig_remove(self, req_id)
        try:
            if req_id in _readding():
                STATS['ids_kept_readd'] += 1
            else:
                rid = _IDS.pop(req_id, None)
                _FLOOR.pop(req_id, None)
                if rid is not None:
                    _FREE.append(rid)
                    STATS['ids_freed'] += 1
        except Exception as exc:
            _err(exc)
        return result

    cls.prepare_attn, cls.remove_request, cls.add_request = prepare_attn, remove_request, add_request


def _rerope_exact(rid, req_id, module, runner):
    """After a promptmove of request req_id: rotate the exact K copies of the moved block's rows (row < the move's rows) on every
    layer by the move's increment (delta - previous), with promptmove's own angles; anything unexpected clears the set instead."""
    try:
        last = module.STATS.get('last') or {}
        if last.get('req') != req_id[-24:]:
            raise RuntimeError('kvring: promptmove last record is another request')
        k = int(last['rows'])
        inc = int(last['delta']) - int(last['previous'])
        cos, sin = module._angles(module._context(runner), inc)
        n = 0
        with torch.inference_mode():
            for name, xs in _EXACT.items():
                n += xs.rerope(rid, k, cos, sin, _RINGS[name][0])
                xs.pending.add(rid)                              # the consistency check at the next eager step
        STATS['exact_reropes'] += 1
        STATS['exact_rerope_rows'] += n
    except Exception as exc:
        _err(exc)
        _clear_exact(rid, 'promptmove')


def install_promptmove_guard(module):
    """After a promptmove of request req_id's leading rows [0, k): if row k - 1 lies in the window of the request's current resident
    rows (len(resident_positions)), lift its read boundary to k - 1."""
    orig = module.maybe_move

    def maybe_move(runner, req_id, block_ids, rows):
        before = dict(module.STATS)
        result = orig(runner, req_id, block_ids, rows)
        try:
            if module.STATS.get('moves', 0) > before.get('moves', 0):
                if EXACT and _IDS.get(req_id) is not None:     # the moved block's exact copies hold the old RoPE: rotate them too
                    _rerope_exact(_IDS[req_id], req_id, module, runner)
                state = runner._arc3_eviction_states.get(req_id) or {}
                resident = len(state.get('resident_positions', ()))
                k = int((module.STATS.get('last') or {}).get('rows', 0))
                if k and k - 1 > resident - 1 - N:
                    _FLOOR[req_id] = max(_FLOOR.get(req_id, -1), k - 1)
                    STATS['floor_events'] += 1
        except Exception as exc:
            _err(exc)
        return result

    module.maybe_move = maybe_move


def install_model_runner(module):
    """Rings allocated right after the KV cache (after profiling, before any CUDA-graph capture; IE): every target QSA owner, as
    kvrot's target format is a lut one (the drafter's owner, 'mtp.', never gets a ring)."""
    cls = module.GPUModelRunner
    orig = cls.initialize_kv_cache

    def initialize_kv_cache(self, kv_cache_config):
        result = orig(self, kv_cache_config)
        try:
            from .kvrot import role_formats, _DRAFT_NAME
            if role_formats()[0].startswith('lut'):
                from vllm.config import get_layers_from_vllm_config
                from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
                _ensure_meta(int(self.max_num_reqs), self.device)
                for name, layer in get_layers_from_vllm_config(self.vllm_config, AttentionLayerBase).items():
                    impl = getattr(layer, 'impl', None)
                    if type(impl).__name__ != 'Qwen3_8FlashNextQSAFlashAttentionImpl' or _DRAFT_NAME.search(name):
                        continue
                    _LAYERS.setdefault(name, (int(layer.num_kv_heads), int(layer.head_dim), self.device))
                _allocate_rings()
                STATS['rings_at_kv_init'] = len(_RINGS)
                _install_commit_hook()
        except Exception as exc:
            _err(exc)
        if EXACT:                                                # outside the guard: never serve an exact set that cannot be re-keyed
            ml = int(getattr(self, 'max_model_len', 0) or 0)
            if not STATS['commit_hook'] or len(_EXACT) != len(_RINGS) or (ml and ml > EXACT_MAXROWS):
                raise RuntimeError('kvring: exact set not safe (commit hook %s, %d exact states for %d rings, max_model_len %d > '
                                   'maxrows %d)' % (STATS['commit_hook'], len(_EXACT), len(_RINGS), ml, EXACT_MAXROWS))
        return result

    cls.initialize_kv_cache = initialize_kv_cache


PATCHERS = {
    'vllm.models.qwen3_8_flash_next.nvidia.qsa': patch_qsa,
    'vllm.v1.worker.gpu.model_states.mamba_hybrid': install_model_state,
    'vllm.v1.worker.gpu.model_runner': install_model_runner,
}
from .kvrot import PINS as _KVROT_PINS          # noqa: E402  (the same modules, so the same pins: chained hooks)
from .rollback import PINS as _ROLLBACK_PINS    # noqa: E402
from .persist import PINS as _PERSIST_PINS      # noqa: E402
PINS = {'vllm.models.qwen3_8_flash_next.nvidia.qsa': _KVROT_PINS['vllm.models.qwen3_8_flash_next.nvidia.qsa'],
        'vllm.v1.worker.gpu.model_states.mamba_hybrid': _ROLLBACK_PINS['vllm.v1.worker.gpu.model_states.mamba_hybrid'],
        'vllm.v1.worker.gpu.model_runner': _PERSIST_PINS['vllm.v1.worker.gpu.model_runner']}
