"""Stable packed-row deletion, independent of vLLM request/allocator mutation.

The caller must fence the worker, preserve the accepted final token,
and free pages only after every owning layer has acknowledged its copy. Imports
perform no torch initialization and no runtime patch installation.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import ceil

import numpy as np


class EvictionStorageError(ValueError):
    pass


_VALIDATED = {}   # id(tuple) -> tuple: a tuple proven valid skips re-validation while this memo holds it (2026-09-23)
_VALIDATED_LIMIT = 64


def _positions(values, name):
    if type(values) is tuple and _VALIDATED.get(id(values)) is values:
        return values
    result = tuple(values)
    if (any(type(x) is not int or x < 0 for x in result)
            or any(a >= b for a, b in zip(result, result[1:]))):
        raise EvictionStorageError(f"{name} must be strictly increasing nonnegative integers")
    if len(_VALIDATED) >= _VALIDATED_LIMIT:
        _VALIDATED.pop(next(iter(_VALIDATED)))
    _VALIDATED[id(result)] = result
    return result


@dataclass(frozen=True)
class TokenCompaction:
    old_positions: tuple[int, ...]
    kept_positions: tuple[int, ...]
    keep_indices: tuple[int, ...]
    computed_keep_indices: tuple[int, ...]
    old_computed: int
    new_computed: int

    @property
    def removed_tokens(self):
        return len(self.old_positions)-len(self.kept_positions)


def token_compaction(old_positions, kept_positions, *, old_computed, allow_fully_computed=False):
    old = _positions(old_positions, "Old positions")
    kept = _positions(kept_positions, "Retained positions")
    if type(allow_fully_computed) is not bool:
        raise EvictionStorageError("Closed-frontier allowance must be an explicit boolean")
    allowed = (len(old)-1, len(old)) if allow_fully_computed else (len(old)-1,)
    if type(old_computed) is not int or old_computed not in allowed or not old:
        raise EvictionStorageError("Expected one pending token, or an explicitly admitted fully computed frontier")
    if not kept or kept[-1] != old[-1]:
        raise EvictionStorageError("The accepted final token must survive")
    old_a = np.fromiter(old, dtype=np.int64, count=len(old))
    kept_a = np.fromiter(kept, dtype=np.int64, count=len(kept))
    idx = np.searchsorted(old_a, kept_a)
    if int(idx[-1]) >= len(old) or not np.array_equal(old_a[idx], kept_a):
        raise EvictionStorageError("Retention is not a subset of resident token positions")
    indices = tuple(idx.tolist())
    computed = tuple(idx[idx < old_computed].tolist())
    if not computed:
        raise EvictionStorageError("At least one computed token must survive")
    return TokenCompaction(old, kept, indices, computed, old_computed, len(computed))


@dataclass(frozen=True)
class PageCompaction:
    old_block_ids: tuple[int, ...]
    keep_rows: tuple[int, ...]
    old_rows: int
    storage_block_size: int
    kept_block_ids: tuple[int, ...]
    released_block_ids: tuple[int, ...]

    @property
    def retained_rows(self):
        return len(self.keep_rows)


def page_compaction(block_ids, keep_rows, *, old_rows, storage_block_size):
    blocks = tuple(block_ids)
    rows = _positions(keep_rows, "Retained cache rows")
    if (type(storage_block_size) is not int or storage_block_size < 1
            or type(old_rows) is not int or old_rows < 0):
        raise EvictionStorageError("Invalid row count or storage page size")
    if (any(type(b) is not int or b <= 0 for b in blocks)
            or len(set(blocks)) != len(blocks)):
        raise EvictionStorageError("Only exclusively owned non-null unique pages are supported")
    if old_rows > len(blocks)*storage_block_size or (rows and rows[-1] >= old_rows):
        raise EvictionStorageError("Rows lie outside initialized cache storage")
    needed = ceil(len(rows)/storage_block_size)
    return PageCompaction(blocks, rows, old_rows, storage_block_size,
                          blocks[:needed], blocks[needed:])


DEFAULT_SCRATCH_ROWS = 49152   # 2026-09-24 stage E: one chunk per 46.5K-row layer
DEFAULT_SCRATCH_BYTES = 32 * 1024 * 1024   # 2026-09-24 stage E: 46.5K x 592 B main rows fit one gather


@dataclass(frozen=True)
class RowCopySchedule:
    first_moved: int
    retained_rows: int
    row_bytes: int
    scratch_row_bound: int
    scratch_byte_bound: int

    @property
    def copied_rows(self):
        return self.retained_rows-self.first_moved

    @property
    def chunks(self):
        if not self.copied_rows:
            return ()
        return tuple((begin, min(begin+self.scratch_row_bound, self.retained_rows))
                     for begin in range(self.first_moved, self.retained_rows,
                                        self.scratch_row_bound))


def row_copy_schedule(plan, *, row_bytes, scratch_rows=DEFAULT_SCRATCH_ROWS,
                      scratch_bytes=DEFAULT_SCRATCH_BYTES):
    """Plan stable deletion with a bound on the gathered data allocation.

    Index vectors are separate: four int64 vectors over the moved rows. The
    byte bound applies to the data gather, not the tensor allocator's cached
    blocks or temporary index arithmetic. A row larger than the bound fails
    before any cache mutation, unless the complete operation is a no-op.
    """
    if not isinstance(plan, PageCompaction):
        raise EvictionStorageError("Expected a validated page compaction plan")
    if (type(row_bytes) is not int or row_bytes < 1
            or type(scratch_rows) is not int or scratch_rows < 1
            or type(scratch_bytes) is not int or scratch_bytes < 1):
        raise EvictionStorageError("Scratch bounds and stored row bytes must be positive integers")
    # In a validated stable-deletion plan source[i] >= i, and source[i]-i is
    # nondecreasing. Once a row moves, every subsequent retained row moves.
    # Locate that boundary without scanning a potentially 40K unchanged prefix
    # for each owner when retiring a suffix near the end of the conversation.
    low, high = 0, len(plan.keep_rows)
    while low < high:
        middle = (low+high)//2
        if plan.keep_rows[middle] == middle:
            low = middle+1
        else:
            high = middle
    first_moved = low
    bounded_rows = min(scratch_rows, scratch_bytes//row_bytes)
    if first_moved < len(plan.keep_rows) and not bounded_rows:
        raise EvictionStorageError("A complete cache row exceeds the scratch byte bound")
    return RowCopySchedule(first_moved, len(plan.keep_rows), row_bytes,
                           bounded_rows, scratch_bytes)


def compact_cache_rows_(cache, plan, *, token_axis,
                        scratch_rows=DEFAULT_SCRATCH_ROWS,
                        scratch_bytes=DEFAULT_SCRATCH_BYTES, index_cache=None):
    """Move complete stored bytes in-place using bounded gather/scatter chunks.

    ``cache`` has physical-block dimension zero; ``token_axis`` identifies its
    row within a page. For NVFP4 [pages, heads, tokens, 296], use token_axis=2.
    Scale bytes travel with both packed K/V values. No decode/requantize/rotation
    occurs. CPU tensors are accepted for synthetic validation; actual serving
    transactions must independently synchronize before and after the mutation.

    Advanced tensor indexing already gathers into separate storage; cloning it
    again would duplicate both memory and work. That gather completes before
    its scatter on the same stream. Ascending destination chunks are safe
    because retained source rows never precede their destination. A fused
    overlapping read/write kernel would not have this guarantee.

    The larger chunks replace the original 64-row diagnostic default. The
    explicit byte bound limits gathered data even if another storage format
    has wider rows. Precomputed physical indices avoid repeating GPU integer
    arithmetic for every chunk. No speedup is claimed until GPU measurement.
    """
    import torch
    if not isinstance(plan, PageCompaction):
        raise EvictionStorageError("Expected a validated page compaction plan")
    if (type(token_axis) is not int or not 0 < token_axis < cache.ndim-1
            or cache.stride(-1) != 1
            or cache.shape[token_axis] != plan.storage_block_size):
        raise EvictionStorageError("Unsupported cache layout")
    if plan.old_block_ids and max(plan.old_block_ids) >= cache.shape[0]:
        raise EvictionStorageError("Physical cache page is out of bounds")
    # Reinterpret, do not numerically cast. This also preserves all NaN payloads
    # and signed zeros for BF16 storage and FP32 scale bytes in quantized rows.
    view = cache.view(torch.uint8).movedim(token_axis, 1)
    row_bytes = cache.element_size()
    for axis, size in enumerate(cache.shape):
        if axis not in (0, token_axis):
            row_bytes *= size
    schedule = row_copy_schedule(plan, row_bytes=row_bytes,
                                 scratch_rows=scratch_rows, scratch_bytes=scratch_bytes)
    page_size = plan.storage_block_size
    if schedule.copied_rows:
        # 2026-09-24: layers of one group/kind carry the same plan object; build the index vectors once per plan.
        _key = (page_size, str(cache.device)); _hit = index_cache.get(id(plan)) if index_cache is not None else None
        if _hit is not None and _hit[0] is plan and _hit[1] == _key:
            source_pages, source_offsets, destination_pages, destination_offsets = _hit[2]
        else:
            blocks = torch.tensor(plan.old_block_ids, dtype=torch.int64, device=cache.device)
            source = torch.tensor(plan.keep_rows[schedule.first_moved:], dtype=torch.int64,
                                  device=cache.device)
            destination = torch.arange(schedule.first_moved, schedule.retained_rows,
                                       dtype=torch.int64, device=cache.device)
            source_pages = blocks[source//page_size]
            source_offsets = source%page_size
            destination_pages = blocks[destination//page_size]
            destination_offsets = destination%page_size
            del blocks, source, destination
            if index_cache is not None:
                index_cache[id(plan)] = (plan, _key, (source_pages, source_offsets, destination_pages, destination_offsets))
        for begin, end in schedule.chunks:
            indices = slice(begin-schedule.first_moved, end-schedule.first_moved)
            temporary = view[source_pages[indices], source_offsets[indices]]
            view[destination_pages[indices], destination_offsets[indices]] = temporary
            # Drop the Python reference before allocating the next gather. On
            # this same stream the allocator may reuse its storage only after
            # the preceding scatter, so two data scratch chunks need not live
            # concurrently. Caller fencing remains necessary across requests.
            del temporary
    return {"copied_rows":schedule.copied_rows, "row_bytes":row_bytes,
            "scratch_row_bound":schedule.scratch_row_bound,
            "scratch_byte_bound":scratch_bytes,
            "max_gather_bytes":min(schedule.copied_rows, schedule.scratch_row_bound)*row_bytes,
            "index_vector_bytes":schedule.copied_rows*4*8,
            "copy_chunks":len(schedule.chunks),
            "retained_rows":len(plan.keep_rows),
            "retained_blocks":len(plan.kept_block_ids),
            "releasable_blocks":len(plan.released_block_ids),
            "synchronized":False}
