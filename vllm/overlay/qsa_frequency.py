"""Selection-frequency counters for QSA-guided eviction.

Inert unless ARC3_QSA_FREQUENCY_COLLECT=1. The counters record, per owner slot,
QSA layer and compacted descriptor row, how many informative committed query
rows selected that descriptor. Updates are fused into the existing expanded-
index remap kernel and use the virtual index it already loads, before that
index is overwritten. Exposure is not tracked per token: it is derived at
snapshot time from group birth versus the epoch's committed query positions.

Address spaces: the counter row is the *compacted descriptor row* (virtual
index // 4, non-tail slots only). Original group identity comes from
QSALayout.group_ids at snapshot time; it is never inferred from a resident
token index.
"""
import os
from types import SimpleNamespace

import numpy as np

LAYER_IDS = tuple(range(3, 48, 4))
RATIO = 4
FLAG = 'ARC3_QSA_FREQUENCY_COLLECT'
CACHE_NAME = 'language_model.model.layers.{}.self_attn.indexer.compressed_key_cache'


class QSAFrequencyError(RuntimeError):
    pass


def enabled():
    value = os.environ.get(FLAG, '0')
    if value not in ('0', '1'):
        raise QSAFrequencyError(f'{FLAG} must be 0 or 1')
    return value == '1'


def layer_map(bound_layers):
    """{compressed-key-cache data_ptr: layer index}; built once after weights load."""
    result = {}
    for index, layer_id in enumerate(LAYER_IDS):
        name = CACHE_NAME.format(layer_id)
        layer = bound_layers.get(name)
        cache = getattr(layer, 'kv_cache', None)
        if cache is None:
            raise QSAFrequencyError('Missing compressed key cache owner: ' + name)
        pointer = cache.data_ptr()
        if pointer in result or pointer == 0:
            raise QSAFrequencyError('Compressed key caches must have distinct nonzero addresses')
        result[pointer] = index
    return result


def descriptor_capacity(max_resident_tokens):
    # Descriptor rows: at most resident/4 complete groups plus one partially resident group per
    # unaligned boundary of the resident set. Turn-level history eviction (stage 12) removes many
    # ranges whose ends are not multiples of RATIO and the partial groups persist for the session,
    # so the slack is proportional (resident/16 rows, >= 2,900 at 46.5K) rather than a fixed 16.
    return max_resident_tokens // RATIO + max_resident_tokens // (RATIO * 4) + 16


def allocate(torch, *, slots, capacity, device, bound_layers):
    """Counters are allocated at model load; KV caches do not exist yet then, so the
    layer map is resolved lazily from `bound_layers()` on the first selector call."""
    if not callable(bound_layers):
        raise QSAFrequencyError('bound_layers must be a callable returning the forward context')
    return SimpleNamespace(
        hits=torch.zeros((slots, len(LAYER_IDS), capacity), dtype=torch.int32, device=device),
        informative=torch.zeros((slots, len(LAYER_IDS)), dtype=torch.int32, device=device),
        enabled=torch.zeros((1,), dtype=torch.int32, device=device),
        layer_index={}, bound_layers=bound_layers, layer_ids=LAYER_IDS,
        capacity=capacity, slots=slots)


def layer_for_cache(counters, k_cache):
    """Resolve the QSA layer from the cache tensor address; build the map once, lazily.
    Under graph capture the first call happens during capture, so the returned integer is
    baked into the captured launch exactly as an install-time map would have been."""
    pointer = k_cache.data_ptr()
    layer = counters.layer_index.get(pointer, -1)
    if layer < 0 and not counters.layer_index:
        counters.layer_index = layer_map(counters.bound_layers())
        layer = counters.layer_index.get(pointer, -1)
    if layer < 0:
        raise QSAFrequencyError('Selector called with an unregistered compressed key cache')
    return layer


def reference_update(counters, indices, token_to_req, mapping, *, layer, token_topk):
    """CPU reference with the fused kernel's exact semantics; `indices` is pre-remap (virtual).

    A descriptor is counted once per informative query row and layer, through
    its designated slot: the first column of its quad within the selected
    region. Automatic tail slots (columns beyond token_topk), padding, invalid
    slots/requests and rows whose visible descriptors fit inside the top-k
    budget are excluded. Tokens at or beyond a slot's virtual cutoff are
    identity-shifted residents with ordinary descriptors (an owner without
    deletions has cutoff 0), so the descriptor row is virtual // 4 throughout.
    """
    if int(counters.enabled[0]) != 1:
        return
    topk_groups = token_topk // RATIO
    table = mapping.virtual_to_resident
    slots = (list(range(table.shape[0])) if mapping.request_to_state_slot is None
             else mapping.request_to_state_slot.tolist())
    lengths = mapping.virtual_seq_lens.tolist()
    positions = mapping.virtual_query_positions.tolist()
    rows, width = indices.shape
    for row, request in enumerate(token_to_req.tolist()):
        if not (0 <= request < len(slots)):
            continue
        slot = slots[request]
        if not (0 <= slot < counters.slots):
            continue
        qpos = positions[row]
        if qpos // RATIO <= topk_groups:
            continue                                    # every visible group fits: no ranking information
        counters.informative[slot, layer] += 1
        row_values = indices[row].tolist()
        for column in range(0, min(width, token_topk), RATIO):
            virtual = row_values[column]
            if virtual < 0 or virtual >= lengths[request]:
                continue
            drow = virtual // RATIO
            if drow < counters.capacity:
                counters.hits[slot, layer, drow] += 1


# ---------------------------------------------------------------- lifecycle

def on_fresh_admission(counters, slot):
    """A new owner takes a slot: drop any previous owner's statistics.

    Also arms the device flag. Ordinary owners are admitted only after warmup
    and graph capture, so nothing captured or warmed can have counted.
    """
    if not (0 <= slot < counters.slots):
        raise QSAFrequencyError('Admission slot outside counter allocation')
    counters.hits[slot].zero_()
    counters.informative[slot].zero_()
    counters.enabled.fill_(1)


def on_compaction(counters, slot, keep_rows, *, reset_epoch):
    """After a committed compaction: keep_rows[i] is the old descriptor row now at row i.

    Statistics follow their descriptors. A history eviction additionally
    starts a new epoch (reset_epoch), because the v1 policy counts since the
    previous committed history eviction; other deletions preserve counts.
    """
    if not (0 <= slot < counters.slots):
        raise QSAFrequencyError('Compaction slot outside counter allocation')
    if reset_epoch:
        counters.hits[slot].zero_()
        counters.informative[slot].zero_()
        return
    rows = list(keep_rows)
    if any(type(r) is not int or r < 0 or r >= counters.capacity for r in rows):
        raise QSAFrequencyError('Descriptor keep rows outside counter capacity')
    if len(set(rows)) != len(rows):
        raise QSAFrequencyError('Descriptor keep rows must be unique')
    block = counters.hits[slot]
    if rows:
        import torch
        index = torch.tensor(rows, dtype=torch.int64, device=block.device)
        kept = block.index_select(1, index)
        block[:, :len(rows)] = kept
    block[:, len(rows):] = 0


def descriptor_keep_rows_from_wire(wire):
    """The shared descriptor order for this transaction, from the compressed owners' plans."""
    orders = []
    for group in wire.get('groups', ()):
        for item in group.get('copies', ()):
            if item.get('kind') == 'compressed':
                orders.append(tuple(item['plan'].keep_rows))
    if not orders:
        return None
    if any(o != orders[0] for o in orders):
        raise QSAFrequencyError('Compressed owners disagree on descriptor keep rows')
    return orders[0]


# ---------------------------------------------------------------- snapshot

def worker_snapshot(worker, request_id, resident_positions, logical_computed):
    """Worker RPC: one owner's counters for the paused frontier the core reports.

    Read-only. The owner is paused, so its rows are not being written; other
    owners may keep decoding. The worker's stored eviction context is the
    frontier at the owner's last (re)admission and is stale for a live owner
    (a fresh owner is admitted with logical_computed == 0), so the layout is
    built from the core's paused context and only cross-checked against the
    stored one. Rows are trimmed to that layout to bound the transfer.
    """
    import torch
    from eviction_qsa import qsa_layout
    runner = worker.model_runner
    counters = getattr(runner.model_state, '_arc3_qsa_frequency', None)
    if counters is None:
        raise QSAFrequencyError('Frequency collection is not enabled on this worker')
    slot = runner.req_states.req_id_to_index.get(request_id)
    if type(slot) is not int:
        raise QSAFrequencyError('Worker owner is missing')
    current = getattr(runner, '_arc3_eviction_states', {}).get(request_id)
    if not isinstance(current, dict) or current.get('awaiting_readd') is not False:
        raise QSAFrequencyError('Missing usable worker eviction context')
    if current.get('req_index') != slot:
        raise QSAFrequencyError('Stored eviction owner occupies another slot')
    positions = tuple(int(p) for p in resident_positions)
    computed = int(logical_computed)
    stored = tuple(int(p) for p in current['resident_positions'])
    if positions[:len(stored)] != stored or computed < int(current['logical_computed']):
        raise QSAFrequencyError('Core frontier does not extend the worker admission context')
    try:
        rows = len(qsa_layout(positions, logical_computed=computed).group_ids)
    except Exception as exc:
        raise QSAFrequencyError(f'Paused layout rejected: {exc}') from exc
    if rows > counters.capacity:
        raise QSAFrequencyError('Descriptor layout exceeds counter capacity')
    # 2026-09-23: the paused owner's rows are not being written (it has no queries), so the copy can run on its own
    # stream without a device-wide synchronize; the counters travel as an int32 buffer, not a 130K-element list.
    if counters.hits.is_cuda:
        stream = getattr(counters, 'copy_stream', None)
        if stream is None:
            stream = counters.copy_stream = torch.cuda.Stream(device=runner.device)
        with torch.cuda.stream(stream):
            hits_cpu = counters.hits[slot, :, :rows].to('cpu')
            informative_cpu = counters.informative[slot].to('cpu')
    else:
        hits_cpu = counters.hits[slot, :, :rows].clone(); informative_cpu = counters.informative[slot].clone()
    hits_cpu = hits_cpu.contiguous().to(torch.int32)
    return dict(slot=slot, rows=rows,
                hits=hits_cpu.numpy().tobytes(), hits_layout=dict(dtype='int32', shape=list(hits_cpu.shape)),
                informative=informative_cpu.tolist(),
                context=dict(resident_positions=list(positions), logical_computed=computed,
                             eviction_revision=int(current['eviction_revision']),
                             deleted_tokens=int(current['deleted_tokens'])))


# ------------------------------------------------------- compact snapshot (stage B, 2026-09-24)

def _positions_digest(positions):
    """sha256 of the positions as little-endian int64 (numpy arrays, lists and tuples agree)."""
    import hashlib
    arr = np.ascontiguousarray(np.asarray(positions, dtype='<i8'))
    return hashlib.sha256(arr.tobytes()).hexdigest()


def worker_snapshot_compact(worker, request_id, expected_revision, expected_deleted, rows_bound):
    """Worker RPC: the paused owner's counters up to rows_bound; no layout, no position echo.

    The worker cross-checks its stored admission context (revision and deleted-token count
    must be the core's) and returns a digest of its stored resident positions so the API,
    which rebuilds the paused positions from the core's ledger, can prove the paused layout
    extends the admission layout (the guarantee the legacy prefix comparison gave).
    """
    import torch
    runner = worker.model_runner
    counters = getattr(runner.model_state, '_arc3_qsa_frequency', None)
    if counters is None:
        raise QSAFrequencyError('Frequency collection is not enabled on this worker')
    slot = runner.req_states.req_id_to_index.get(request_id)
    if type(slot) is not int:
        raise QSAFrequencyError('Worker owner is missing')
    current = getattr(runner, '_arc3_eviction_states', {}).get(request_id)
    if not isinstance(current, dict) or current.get('awaiting_readd') is not False:
        raise QSAFrequencyError('Missing usable worker eviction context')
    if current.get('req_index') != slot:
        raise QSAFrequencyError('Stored eviction owner occupies another slot')
    if int(current['eviction_revision']) != int(expected_revision) or int(current['deleted_tokens']) != int(expected_deleted):
        raise QSAFrequencyError('Worker and core eviction contexts differ')
    rows = min(int(rows_bound), int(counters.capacity))   # the API checks the layout fits the returned rows
    if rows < 0:
        raise QSAFrequencyError('Negative descriptor row bound')
    digest = current.get('_resident_positions_sha256')
    if digest is None:
        digest = current['_resident_positions_sha256'] = _positions_digest(current['resident_positions'])
    if counters.hits.is_cuda:
        stream = getattr(counters, 'copy_stream', None)
        if stream is None:
            stream = counters.copy_stream = torch.cuda.Stream(device=runner.device)
        with torch.cuda.stream(stream):
            hits_cpu = counters.hits[slot, :, :rows].to('cpu')
            informative_cpu = counters.informative[slot].to('cpu')
    else:
        hits_cpu = counters.hits[slot, :, :rows].clone(); informative_cpu = counters.informative[slot].clone()
    hits_cpu = hits_cpu.contiguous().to(torch.int32)
    return dict(slot=slot, rows=rows,
                hits=hits_cpu.numpy().tobytes(), hits_layout=dict(dtype='int32', shape=list(hits_cpu.shape)),
                informative=informative_cpu.tolist(),
                stored=dict(length=len(current['resident_positions']), sha256=digest,
                            logical_computed=int(current['logical_computed']),
                            eviction_revision=int(current['eviction_revision']), deleted_tokens=int(current['deleted_tokens'])))


def core_snapshot_compact(core, external):
    """EngineCore utility (stage B): read-only and copy-free; requires a paused healthy owner with no work in flight.

    The paused logical length is prompt_logical_length + the owner's output tokens (what
    LogicalState.paused would record) and the frontier is num_computed_tokens + deleted
    tokens (what LogicalState.context reports); the evicted ranges travel instead of the
    resident positions.  prepare_transaction re-derives and re-validates all of it later.
    """
    import engine_eviction
    scheduler = core.scheduler
    request = engine_eviction._request_for_external(scheduler, external)
    if request.status.name != 'WAITING_FOR_STREAMING_REQ' or request.num_in_flight_tokens:
        raise QSAFrequencyError('Frequency snapshot requires a paused owner with no work in flight')
    state = getattr(request, '_arc3_eviction_state', None)
    if state is None or state.poisoned:
        raise QSAFrequencyError('Owner is not a healthy eviction session')
    before = _state_fingerprint(state)
    ledger = state.ledger
    evicted = [[int(a), int(b)] for a, b in ledger.evicted]
    deleted = int(sum(b - a for a, b in evicted))
    logical_len = int(state.prompt_logical_length) + len(request._output_token_ids)
    computed = int(request.num_computed_tokens) + deleted
    if evicted and evicted[-1][1] > logical_len:
        raise QSAFrequencyError('Evicted ranges exceed the paused logical length')
    resident = logical_len - deleted
    rows_bound = resident // RATIO + 2 * len(evicted) + 2
    replies = core.collective_rpc(worker_snapshot_compact, timeout=60,
                                  args=(request.request_id, int(state.eviction_revision), deleted, rows_bound))
    if _state_fingerprint(state) != before:
        raise QSAFrequencyError('Live logical state changed during snapshot preparation')
    if type(replies) is not list or len(replies) != 1:
        raise QSAFrequencyError('Exactly one worker reply required')
    reply = replies[0]
    stored = reply['stored']
    if stored['eviction_revision'] != state.eviction_revision or stored['deleted_tokens'] != deleted:
        raise QSAFrequencyError('Worker and core eviction contexts differ')
    reply['form'] = 'compact'
    reply['context'] = dict(logical_tokens=logical_len, evicted=evicted, logical_computed=computed,
                            eviction_revision=int(state.eviction_revision), deleted_tokens=deleted, rows_bound=rows_bound)
    return reply


def compact_positions(logical_tokens, evicted):
    """Resident original positions of a ledger (numpy complement of the evicted ranges)."""
    n = int(logical_tokens)
    mask = np.ones(n, dtype=bool)
    last = 0
    for a, b in evicted:
        a = int(a); b = int(b)
        if a < last or b <= a or b > n:
            raise QSAFrequencyError('Evicted ranges must be sorted, disjoint and within the logical length')
        mask[a:b] = False; last = b
    return np.flatnonzero(mask)


def expand_compact_snapshot(snapshot):
    """Compact form -> the legacy snapshot dict (positions rebuilt, stored prefix proven, rows = counter width)."""
    ctx = snapshot['context']
    positions = compact_positions(ctx['logical_tokens'], ctx['evicted'])
    stored = snapshot['stored']
    n = int(stored['length'])
    if n > len(positions) or _positions_digest(positions[:n]) != stored['sha256']:
        raise QSAFrequencyError('Core frontier does not extend the worker admission context')
    if int(ctx['logical_computed']) < int(stored['logical_computed']):
        raise QSAFrequencyError('Core frontier does not extend the worker admission context')
    layout = snapshot.get('hits_layout') or {}
    shape = tuple(int(x) for x in layout.get('shape', ()))
    if len(shape) != 2 or shape[1] != int(snapshot['rows']) or int(snapshot['rows']) > int(ctx['rows_bound']):
        raise QSAFrequencyError('Compact snapshot rows disagree with the row bound')
    return dict(slot=snapshot['slot'], rows=int(snapshot['rows']), hits=snapshot['hits'], hits_layout=snapshot['hits_layout'],
                informative=snapshot['informative'],
                context=dict(resident_positions=positions.tolist(), logical_computed=int(ctx['logical_computed']),
                             eviction_revision=int(ctx['eviction_revision']), deleted_tokens=int(ctx['deleted_tokens'])))


def _state_fingerprint(state):
    ledger = state.ledger
    return (len(state.logical_ids), state.prompt_logical_length, ledger.logical_tokens, ledger.resident_tokens,
            tuple(tuple(r) for r in ledger.evicted), state.eviction_revision, state.poisoned)


def core_snapshot(core, external):
    """EngineCore utility: read-only; requires a paused healthy owner with no work in flight.

    The layout comes from a private deep copy of the core's logical state extended
    to the paused frontier (engine_eviction.prepare_transaction mutates the live
    state, so it is never called here); the live state is fingerprinted before and
    after. The worker receives that context and cross-checks its own.
    """
    import copy
    import engine_eviction
    scheduler = core.scheduler
    request = engine_eviction._request_for_external(scheduler, external)
    if request.status.name != 'WAITING_FOR_STREAMING_REQ' or request.num_in_flight_tokens:
        raise QSAFrequencyError('Frequency snapshot requires a paused owner with no work in flight')
    state = getattr(request, '_arc3_eviction_state', None)
    if state is None or state.poisoned:
        raise QSAFrequencyError('Owner is not a healthy eviction session')
    before = _state_fingerprint(state)
    private = copy.deepcopy(state)
    try:
        private.paused(request)
        context = private.context(request)
    except Exception as exc:
        raise QSAFrequencyError(f'Paused context unavailable: {exc}') from exc
    if _state_fingerprint(state) != before:
        raise QSAFrequencyError('Live logical state changed during snapshot preparation')
    positions = [int(p) for p in context['resident_positions']]
    computed = int(context['logical_computed'])
    replies = core.collective_rpc(worker_snapshot, timeout=60, args=(request.request_id, positions, computed))
    if type(replies) is not list or len(replies) != 1:
        raise QSAFrequencyError('Exactly one worker reply required')
    reply = replies[0]
    ctx = reply['context']
    if (ctx['eviction_revision'] != state.eviction_revision or ctx['deleted_tokens'] != context['deleted_tokens']
            or ctx['logical_computed'] != computed or ctx['resident_positions'] != positions):
        raise QSAFrequencyError('Worker and core eviction contexts differ')
    return reply


# ----------------------------------------------------------------- scoring

class _LazyTables:
    """List-like view of per-layer {group: frequency|None} dicts, materialized on first use (2026-09-23)."""
    def __init__(self, matrix, group_ids):
        self._matrix = matrix; self._ids = group_ids; self._tables = None
    def _build(self):
        if self._tables is None:
            self._tables = [{g: (v if v == v else None) for g, v in zip(self._ids, layer.tolist())} for layer in self._matrix]
        return self._tables
    def __len__(self): return int(self._matrix.shape[0])
    def __iter__(self): return iter(self._build())
    def __getitem__(self, i): return self._build()[i]


def _hits_matrix(snapshot):
    """Snapshot hits as an int64 [layers, rows] array: bytes form (worker) or nested lists (tests/legacy)."""
    hits = snapshot['hits']
    if isinstance(hits, (bytes, bytearray, memoryview)):
        layout = snapshot.get('hits_layout') or {}
        shape = tuple(int(x) for x in layout.get('shape', ()))
        if len(shape) != 2 or layout.get('dtype') != 'int32':
            raise QSAFrequencyError('Snapshot hits buffer has no int32 [layers, rows] layout')
        arr = np.frombuffer(bytes(hits), dtype=np.int32)
        if arr.size != shape[0] * shape[1]:
            raise QSAFrequencyError('Snapshot hits buffer size disagrees with its layout')
        return arr.reshape(shape).astype(np.int64)
    if not hits:
        return np.zeros((0, 0), dtype=np.int64)
    width = min(len(layer) for layer in hits)
    return np.array([list(layer[:width]) for layer in hits], dtype=np.int64)


def group_frequencies(snapshot, *, epoch_start_logical):
    """Per layer and original group: hits / analytic exposure. None marks zero exposure.

    Exposure counts the epoch's query positions that could see the complete
    group: a group g is complete from original position 4g+3, so queries at
    positions >= 4g+4 see it. Every query row at this context length is
    informative, so exposure is derived from positions rather than tracked
    per token. The per-layer informative counts are returned for cross-checks.
    """
    from eviction_qsa import qsa_layout
    if snapshot.get('form') == 'compact':
        snapshot = expand_compact_snapshot(snapshot)
    ctx = snapshot['context']
    computed = int(ctx['logical_computed'])
    layout = qsa_layout(tuple(ctx['resident_positions']), logical_computed=computed)
    rows = len(layout.group_ids)
    hits = _hits_matrix(snapshot)
    if rows > int(snapshot['rows']) or hits.shape[1] < rows:
        raise QSAFrequencyError('Snapshot rows do not cover the descriptor layout')
    start = int(epoch_start_logical)
    gids = np.fromiter(layout.group_ids, dtype=np.int64, count=rows)
    expo = np.maximum(0, computed - np.maximum(start, (gids + 1) * RATIO))
    exposure = dict(zip(layout.group_ids, expo.tolist()))
    with np.errstate(divide='ignore', invalid='ignore'):
        matrix = np.where(expo > 0, hits[:, :rows].astype(np.float64) / np.where(expo > 0, expo, 1), np.nan)
    frequency = _LazyTables(matrix, list(layout.group_ids))   # per-layer dicts only if a legacy consumer asks
    return dict(frequency=frequency, matrix=matrix, group_ids_array=gids, exposure=exposure, layout_rows=rows,
                informative=[int(x) for x in snapshot['informative']],
                epoch_start_logical=start, logical_computed=computed,
                epoch_query_positions=max(0, computed - start))


def score_blocks(eligible, frequencies, *, min_coverage=0.5):
    """Score eligible interactions: mean over layers of sum_g n(I,g)*freq[l,g] / resident(I).

    n(I,g) counts I's surviving tokens in original group g (logical position
    // 4), so a group crossing an interaction boundary is split by membership.
    Groups with zero exposure contribute nothing and lower coverage; a block
    below min_coverage makes the whole ranking unusable (caller falls back).
    """
    layers = frequencies['frequency']
    scores, coverages = [], []
    for _observation, _end, pieces, tokens in eligible:
        counts = {}
        for start, stop in pieces:
            for position in range(start, stop):
                g = position // RATIO
                counts[g] = counts.get(g, 0) + 1
        per_layer = []
        for table in layers:
            total = 0.0
            for g, n in counts.items():
                value = table.get(g)
                if value is not None:
                    total += n * value
            per_layer.append(total / tokens)
        covered = sum(n for g, n in counts.items() if layers[0].get(g) is not None)
        scores.append(sum(per_layer) / len(per_layer))
        coverages.append(covered / tokens)
    return scores, coverages, all(c >= min_coverage for c in coverages)
