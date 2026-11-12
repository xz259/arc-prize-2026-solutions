"""One address-only remap kernel after the unchanged optimized QSA selector.

With HAS_COUNTERS the same kernel also records selection frequency for
QSA-guided eviction, reading the virtual index it already loaded before that
index is overwritten. The HAS_COUNTERS=False specialization contains no
counter code (constexpr elision), so the serving path is unchanged when the
collector is off.
"""
import triton
import triton.language as tl


@triton.jit
def _remap_indices_kernel(
    indices, requests, lengths, table, cutoffs, padding, slots,
    rows, width, num_requests, num_slots, table_width,
    index_row_stride, index_column_stride, request_stride,
    table_row_stride, table_column_stride,
    hits, informative, enabled, query_positions, position_stride,
    layer, num_layers, capacity, token_topk, hits_slot_stride, hits_layer_stride,
    HAS_TAIL: tl.constexpr, HAS_SLOT_MAP: tl.constexpr, HAS_COUNTERS: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.program_id(1)*BLOCK+tl.arange(0, BLOCK)
    valid_column = columns < width
    request = tl.load(requests+row*request_stride)
    valid_request = (request >= 0) & (request < num_requests)
    slot = request
    if HAS_SLOT_MAP:
        slot = tl.load(slots+request, mask=valid_request, other=-1)
    valid_slot = valid_request & (slot >= 0) & (slot < num_slots)
    seq_len = tl.load(lengths+request, mask=valid_request, other=0)
    virtual = tl.load(indices+row*index_row_stride+columns*index_column_stride,
                      mask=valid_column, other=-1)
    valid = valid_column & valid_slot & (virtual >= 0) & (virtual < seq_len)
    tail = tl.full((BLOCK,), False, tl.int1)
    shift = 0
    cutoff = 0
    if HAS_TAIL:
        cutoff = tl.load(cutoffs+slot, mask=valid_slot, other=0)
        shift = tl.load(padding+slot, mask=valid_slot, other=0)
        tail = valid & (virtual >= cutoff)
    if HAS_COUNTERS:
        # Count before the virtual index is overwritten. One designated slot per
        # selected quad (its first column); the automatic tail slots beyond
        # token_topk carry no selection. Tokens at or beyond the slot's virtual
        # cutoff are identity-shifted residents with ordinary descriptors (an
        # owner without deletions has cutoff 0), so drow = virtual // 4 for every
        # valid selected token. Rows whose visible descriptors fit in the top-k
        # budget carry no ranking information.
        on = tl.load(enabled) == 1
        qpos = tl.load(query_positions+row*position_stride)
        informative_row = on & valid_slot & ((qpos // 4) > (token_topk // 4))
        designated = valid & (columns < token_topk) & ((columns % 4) == 0)
        drow = virtual // 4
        base = hits + slot.to(tl.int64)*hits_slot_stride + layer*hits_layer_stride
        tl.atomic_add(base + drow, 1, mask=designated & informative_row & (drow < capacity))
        first = tl.program_id(1) == 0
        tl.atomic_add(informative + slot.to(tl.int64)*num_layers + layer, 1,
                      mask=first & informative_row)
    table_value = tl.load(table+slot.to(tl.int64)*table_row_stride
                         +virtual.to(tl.int64)*table_column_stride,
                         mask=valid & ~tail & (virtual < table_width), other=-1)
    mapped = tl.where(tail, virtual-shift, table_value)
    tl.store(indices+row*index_row_stride+columns*index_column_stride,
             tl.where(valid & (mapped >= 0), mapped, -1), mask=valid_column)


def remap_indices_cuda_(indices, token_to_req, mapping, counters=None, layer=-1, token_topk=None):
    rows, width = indices.shape
    if not rows or not width:
        return indices
    table = mapping.virtual_to_resident
    has_counters = counters is not None
    if has_counters:
        if layer < 0 or token_topk is None:
            raise ValueError('Counter collection requires a layer index and the selection width')
        positions = mapping.virtual_query_positions
        hits, informative, enabled = counters.hits, counters.informative, counters.enabled
        extra = (hits, informative, enabled, positions, positions.stride(0), layer, hits.shape[1],
                 counters.capacity, token_topk, hits.stride(0), hits.stride(1))
    else:
        extra = (indices, indices, indices, indices, 0, 0, 0, 0, 0, 0, 0)
    _remap_indices_kernel[(rows, triton.cdiv(width, 256))](
        indices, token_to_req, mapping.virtual_seq_lens, table,
        mapping.virtual_cutoffs, mapping.group_padding, mapping.request_to_state_slot,
        rows, width, mapping.virtual_seq_lens.shape[0], table.shape[0], table.shape[1],
        indices.stride(0), indices.stride(1), token_to_req.stride(0),
        table.stride(0), table.stride(1),
        *extra,
        HAS_TAIL=mapping.virtual_cutoffs is not None,
        HAS_SLOT_MAP=mapping.request_to_state_slot is not None,
        HAS_COUNTERS=has_counters,
        BLOCK=256, num_warps=4)
    return indices
