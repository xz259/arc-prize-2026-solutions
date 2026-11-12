"""Original-group QSA semantics for token eviction; no model/runtime mutation.

A partial original group retains its already-conditioned index representative.
It expands only to surviving members. Fully removed groups lose their index
rows. This is deliberately not recompression of a retokenized short history.
"""
from __future__ import annotations

from dataclasses import dataclass
from bisect import bisect_left, bisect_right

from eviction_storage import EvictionStorageError, _positions

import numpy as np


@dataclass(frozen=True)
class QSALayout:
    ratio: int
    logical_computed: int
    resident_positions: tuple[int, ...]
    group_ids: tuple[int, ...]
    group_ends: tuple[int, ...]
    members: tuple[tuple[int, ...], ...]
    open_group_id: int
    open_members: tuple[int, ...]

    def visible_groups(self, query_position):
        """Only original groups wholly processed by this causal query exist."""
        return bisect_right(self.group_ends, query_position)


def qsa_layout(resident_positions, *, logical_computed, ratio=4):
    positions = _positions(resident_positions, "Resident original token positions")
    if (type(ratio) is not int or ratio != 4 or type(logical_computed) is not int
            or logical_computed < 0):
        raise EvictionStorageError("This prototype supports original four-token groups")
    if positions and positions[-1] > logical_computed:
        raise EvictionStorageError("Only one accepted uncomputed token may follow the frontier")
    complete = logical_computed//ratio
    # 2026-09-23 numpy path (same result as the dict construction it replaces; see test_evict_fastpath.py):
    # positions are strictly increasing, so those below the frontier are a prefix and their groups are nondecreasing.
    pos = np.fromiter(positions, dtype=np.int64, count=len(positions))
    n = int(np.searchsorted(pos, logical_computed, side="left"))
    groups = pos[:n]//ratio; offsets = pos[:n]%ratio
    m = int(np.searchsorted(groups, complete, side="left"))
    ids_arr = np.unique(groups[:m])
    members_arr = np.full((len(ids_arr), ratio), -1, dtype=np.int64)
    members_arr[np.searchsorted(ids_arr, groups[:m]), offsets[:m]] = np.arange(m)
    open_arr = np.full(ratio, -1, dtype=np.int64)
    sel = groups[m:] == complete
    open_arr[offsets[m:][sel]] = np.arange(m, n)[sel]
    ids = tuple(ids_arr.tolist())
    return QSALayout(ratio, logical_computed, positions, ids,
                     tuple((group+1)*ratio-1 for group in ids), tuple(tuple(r) for r in members_arr.tolist()),
                     complete, tuple(open_arr.tolist()))


def descriptor_keep_rows(before, after):
    """Physical compressed-key rows to gather, without modifying their bytes."""
    if before.ratio != after.ratio or before.logical_computed != after.logical_computed:
        raise EvictionStorageError("Compaction cannot advance or rebase the original frontier")
    def _index_into(sorted_values, wanted, message):
        base = np.fromiter(sorted_values, dtype=np.int64, count=len(sorted_values))
        want = np.fromiter(wanted, dtype=np.int64, count=len(wanted))
        if not len(want):
            return want
        if not len(base):
            raise EvictionStorageError(message)
        idx = np.searchsorted(base, want)
        if int(idx.max()) >= len(base) or not np.array_equal(base[idx], want):
            raise EvictionStorageError(message)
        return idx
    _index_into(before.resident_positions, after.resident_positions, "Compaction introduced new token identities")
    rows = _index_into(before.group_ids, after.group_ids, "Compaction introduced a compressed group")
    return tuple(rows.tolist())


def descriptor_rows_fast(old_positions, kept_positions, *, logical_computed, ratio=4):
    """(len(qsa_layout(old).group_ids), descriptor_keep_rows(qsa_layout(old), qsa_layout(kept))) without building
    either layout (2026-09-24 stage D; differential test in evict_trims_v1/test_evict_trims.py)."""
    old = _positions(old_positions, "Resident original token positions")
    kept = _positions(kept_positions, "Resident original token positions")
    if (type(ratio) is not int or ratio != 4 or type(logical_computed) is not int or logical_computed < 0):
        raise EvictionStorageError("This prototype supports original four-token groups")
    if (old and old[-1] > logical_computed) or (kept and kept[-1] > logical_computed):
        raise EvictionStorageError("Only one accepted uncomputed token may follow the frontier")
    o = np.fromiter(old, dtype=np.int64, count=len(old)); k = np.fromiter(kept, dtype=np.int64, count=len(kept))
    if len(k):
        if not len(o):
            raise EvictionStorageError("Compaction introduced new token identities")
        idx = np.searchsorted(o, k)
        if int(idx.max()) >= len(o) or not np.array_equal(o[idx], k):
            raise EvictionStorageError("Compaction introduced new token identities")
    complete = logical_computed // ratio
    go = np.unique(o[:int(np.searchsorted(o, logical_computed, side="left"))] // ratio); go = go[go < complete]
    gk = np.unique(k[:int(np.searchsorted(k, logical_computed, side="left"))] // ratio); gk = gk[gk < complete]
    if len(gk):
        if not len(go):
            raise EvictionStorageError("Compaction introduced a compressed group")
        rows = np.searchsorted(go, gk)
        if int(rows.max()) >= len(go) or not np.array_equal(go[rows], gk):
            raise EvictionStorageError("Compaction introduced a compressed group")
    else:
        rows = gk
    return int(len(go)), tuple(rows.tolist())


def expand_selected_groups(layout, selected_rows, *, query_position, token_topk=2048):
    """CPU causal reference, retaining fixed-width slots for partial groups.

    The selector supplies descriptor-row indices, not original group IDs.
    For a forward chunk, build the layout with all processed query rows; group
    end positions enforce each query's causal boundary. A current incomplete
    group's members are appended automatically as in stock QSA.
    """
    if type(query_position) is not int or not 0 <= query_position < layout.logical_computed:
        raise EvictionStorageError("Query must be within the processed original stream")
    if type(token_topk) is not int or token_topk < 4 or token_topk%4:
        raise EvictionStorageError("Token selection budget must be a positive multiple of four")
    selected = tuple(selected_rows)
    if len(selected) > token_topk//4 or len(set(selected)) != len(selected):
        raise EvictionStorageError("Invalid/duplicate selected compressed rows")
    result = []
    visible = layout.visible_groups(query_position)
    for row in selected:
        if type(row) is not int or not 0 <= row < visible:
            raise EvictionStorageError("Selected compressed group is absent or future")
        result.extend(layout.members[row])
    # A prefilling query's incomplete group may already be complete by the end
    # of the scheduled chunk; use original identities rather than only the
    # final request's open-ring descriptor.
    lookup = {position:index for index,position in enumerate(layout.resident_positions)}
    tail_start = ((query_position+1)//layout.ratio)*layout.ratio
    result.extend(lookup.get(position,-1) for position in range(tail_start,query_position+1))
    width = token_topk+layout.ratio-1
    return tuple(result + [-1]*(width-len(result)))


def new_group_storage_rows(existing_group_ids, *, old_logical_computed, query_positions, ratio=4):
    """Map future completion positions to compact compressed descriptor slots.

    Called for a *contiguous newly processed* absolute suffix. Partial prior
    groups that lost members still close at their original boundary, using the
    retained raw ring; their expansion map masks those deleted members.
    """
    existing = _positions(existing_group_ids, "Existing original group IDs")
    query = tuple(query_positions)
    if ratio != 4 or type(old_logical_computed) is not int or old_logical_computed < 0:
        raise EvictionStorageError("Invalid original compression frontier")
    if query != tuple(range(old_logical_computed,old_logical_computed+len(query))):
        raise EvictionStorageError("New index keys must process a contiguous original suffix")
    if existing and existing[-1] >= old_logical_computed//ratio:
        raise EvictionStorageError("Existing descriptor lies beyond the old frontier")
    next_row = len(existing)
    output = []
    for position in query:
        if (position+1)%ratio:
            output.append(-1)
        else:
            output.append(next_row)
            next_row += 1
    return tuple(output)


def virtual_query_position(layout, query_position):
    """Position used ONLY for selector visibility/expansion, never for RoPE.

    Dense compressed descriptors preserve the order of original four-token
    groups. Missing members retain holes within their original groups.
    """
    if (type(query_position) is not int or query_position < 0
            or query_position >= layout.logical_computed
            or query_position not in layout.resident_positions):
        raise EvictionStorageError("Selector query must be a processed original position")
    group = query_position // layout.ratio
    return bisect_left(layout.group_ids, group)*layout.ratio + query_position%layout.ratio


@dataclass(frozen=True)
class QSACompactedPrefix:
    logical_computed: int
    resident_computed: int
    deleted_tokens: int
    deleted_groups: int
    group_padding: int
    virtual_cutoff: int
    virtual_to_resident: tuple[int, ...]
    compressed_group_ids: tuple[int, ...]

    def remap(self, virtual_index):
        if type(virtual_index) is not int:
            raise EvictionStorageError("Expected an integer selector index")
        if virtual_index < 0:
            return -1
        if virtual_index < self.virtual_cutoff:
            return self.virtual_to_resident[virtual_index]
        # Every future token lies after all previous deletions. Its packed row
        # is known by a constant shift; no growing lookup table is necessary.
        return virtual_index - self.group_padding


def compacted_prefix(resident_positions, *, logical_computed):
    """Build once at an eviction boundary with the accepted final token alive."""
    layout = qsa_layout(resident_positions, logical_computed=logical_computed)
    if not layout.resident_positions or layout.resident_positions[-1] != logical_computed:
        raise EvictionStorageError("Compacted prefix must retain exactly the accepted uncomputed final token")
    resident_computed = len(layout.resident_positions)-1
    deleted_tokens = logical_computed-resident_computed
    deleted_groups = logical_computed//4-len(layout.group_ids)
    table = tuple(member for group in layout.members for member in group)
    table += layout.open_members[:logical_computed%4]
    cutoff = logical_computed-4*deleted_groups
    padding = deleted_tokens-4*deleted_groups
    if len(table) != cutoff or padding < 0:
        raise EvictionStorageError("Invalid retained QSA grouping")
    return QSACompactedPrefix(logical_computed, resident_computed, deleted_tokens,
                             deleted_groups, padding, cutoff, table, layout.group_ids)


@dataclass(frozen=True)
class QSASelectionMapping:
    """Worker-owned device buffers shared by all QSA layers in one forward.

    The first two buffers are indexed exactly like the normal metadata. The
    historical table may use stable request slots; request_to_state_slot maps
    current batch request indices into those slots. If virtual_cutoffs and
    group_padding are omitted, the table covers the complete virtual stream.
    Otherwise indices beyond each cutoff use the analytic future-tail shift.
    CPU ownership/state mutation is deliberately outside this module.
    """
    virtual_query_positions: object
    virtual_seq_lens: object
    virtual_to_resident: object
    virtual_cutoffs: object | None = None
    group_padding: object | None = None
    request_to_state_slot: object | None = None


def _validate_mapping(mapping, rows, requests):
    if not isinstance(mapping, QSASelectionMapping):
        raise EvictionStorageError("Expected a QSASelectionMapping")
    if (len(mapping.virtual_query_positions.shape) != 1
            or mapping.virtual_query_positions.shape[0] < rows
            or mapping.virtual_seq_lens.shape != (requests,)
            or len(mapping.virtual_to_resident.shape) != 2):
        raise EvictionStorageError("Invalid QSA mapping buffer shapes")
    table_rows = mapping.virtual_to_resident.shape[0]
    if mapping.request_to_state_slot is None:
        if table_rows != requests:
            raise EvictionStorageError("QSA table rows must match requests without an explicit slot map")
    elif mapping.request_to_state_slot.shape != (requests,):
        raise EvictionStorageError("Invalid QSA request-to-state slot mapping")
    if (mapping.virtual_cutoffs is None) != (mapping.group_padding is None):
        raise EvictionStorageError("Analytic tail needs both cutoff and padding buffers")
    if mapping.virtual_cutoffs is not None and (
            mapping.virtual_cutoffs.shape != (table_rows,)
            or mapping.group_padding.shape != (table_rows,)):
        raise EvictionStorageError("Invalid analytic QSA tail buffer shapes")


def remap_selected_indices_(indices, token_to_req, mapping, counters=None, layer=-1, token_topk=None):
    """Preserve optimized selection; replace only its expanded index addresses."""
    import torch
    if (indices.ndim != 2 or token_to_req.shape != (indices.shape[0],)
            or indices.dtype != torch.int32):
        raise EvictionStorageError("Expected fixed-width int32 selection and per-row request IDs")
    _validate_mapping(mapping, indices.shape[0], mapping.virtual_seq_lens.shape[0])
    buffers = [token_to_req, mapping.virtual_query_positions, mapping.virtual_seq_lens,
               mapping.virtual_to_resident, mapping.virtual_cutoffs,
               mapping.group_padding, mapping.request_to_state_slot]
    if any(value is not None and value.device != indices.device for value in buffers):
        raise EvictionStorageError("QSA buffers must share the selection device")
    if any(value is not None and value.dtype not in (torch.int32, torch.int64) for value in buffers):
        raise EvictionStorageError("QSA mapping buffers must contain signed integer indices")
    if any(value is not None and value.stride(0) != 1 for value in (
            mapping.virtual_seq_lens, mapping.virtual_cutoffs, mapping.group_padding,
            mapping.request_to_state_slot)):
        raise EvictionStorageError("Per-request QSA mapping vectors must be contiguous")
    if indices.is_cuda:
        from _eviction_qsa_gpu import remap_indices_cuda_
        return remap_indices_cuda_(indices, token_to_req, mapping, counters, layer, token_topk)
    if counters is not None:
        from qsa_frequency import reference_update
        reference_update(counters, indices, token_to_req, mapping, layer=layer, token_topk=token_topk)
    # CPU correctness reference, not the serving implementation. Keeping it
    # explicit makes padding/sentinel handling straightforward to inspect.
    output = indices.clone()
    table = mapping.virtual_to_resident.tolist()
    request_slots = (list(range(len(table))) if mapping.request_to_state_slot is None
                     else mapping.request_to_state_slot.tolist())
    cutoffs = None if mapping.virtual_cutoffs is None else mapping.virtual_cutoffs.tolist()
    padding = None if mapping.group_padding is None else mapping.group_padding.tolist()
    lengths = mapping.virtual_seq_lens.tolist()
    for row, request in enumerate(token_to_req.tolist()):
        slot = request_slots[request] if 0 <= request < len(request_slots) else -1
        for column, virtual in enumerate(indices[row].tolist()):
            mapped = -1
            if 0 <= slot < len(table) and 0 <= virtual < lengths[request]:
                if cutoffs is not None and virtual >= cutoffs[slot]:
                    mapped = virtual-padding[slot]
                elif virtual < len(table[slot]):
                    mapped = table[slot][virtual]
            output[row, column] = mapped if mapped >= 0 else -1
    indices.copy_(output)
    return indices


def mapping_from_forward_context():
    """The runner attaches one shared mapping to compressed-cache metadata."""
    from vllm.forward_context import get_forward_context
    metadata = get_forward_context().attn_metadata
    if isinstance(metadata, list):
        metadata = metadata[0]
    if not isinstance(metadata, dict):
        return None
    mappings = [getattr(value, "_arc3_qsa_selection_mapping", None)
                for value in metadata.values()]
    mappings = [value for value in mappings if value is not None]
    if not mappings:
        return None
    if any(value is not mappings[0] for value in mappings):
        raise EvictionStorageError("QSA layers received different ownership mappings")
    return mappings[0]


def install_qsa_selector_wrapper(provider=None, *, module=None, counters_provider=None):
    """Explicit opt-in patch; the existing scorer/top-k kernels remain intact."""
    import functools
    if module is None:
        import importlib
        module = importlib.import_module("vllm.models.qwen3_8_flash_next.nvidia.ops.qsa")
    original = module.qsa_select_paged_tokens
    if getattr(original, "_arc3_eviction_qsa_wrapper", False):
        return original
    provider = provider or mapping_from_forward_context

    @functools.wraps(original)
    def select(q, k_cache, page_table, token_to_req, query_positions,
               sequence_lengths, token_topk, compress_ratio, out=None):
        mapping = provider()
        if mapping is None:
            # Exact original call, no mapping allocation or kernel launch.
            return original(q, k_cache, page_table, token_to_req, query_positions,
                            sequence_lengths, token_topk, compress_ratio, out)
        if compress_ratio != 4:
            raise EvictionStorageError("QSA eviction supports original groups of four only")
        _validate_mapping(mapping, q.shape[0], sequence_lengths.shape[0])
        result = original(q, k_cache, page_table, token_to_req,
                          mapping.virtual_query_positions[:q.shape[0]],
                          mapping.virtual_seq_lens, token_topk, compress_ratio, out)
        counters = counters_provider() if counters_provider is not None else None
        if counters is None:
            return remap_selected_indices_(result, token_to_req, mapping)
        from qsa_frequency import layer_for_cache
        return remap_selected_indices_(result, token_to_req, mapping, counters,
                                       layer_for_cache(counters, k_cache), token_topk)

    select._arc3_eviction_qsa_wrapper = True
    module.qsa_select_paged_tokens = select
    return select
