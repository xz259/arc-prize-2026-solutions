"""Opt-in proof of real block-pool reacquisition, without a model forward.

The pinned APC-off allocator prepends freed blocks and allocates from the front.
Sweeping all free blocks and returning them in the same order preserves its
queue exactly. Call only from the serialized, drained, keep-paused core utility.
No GPU tensor or unrelated live block is modified.
"""
from __future__ import annotations

import hashlib
import json


class AllocatorVerificationError(RuntimeError):
    pass


def _ids(blocks):
    return [block.block_id for block in blocks]


def _digest(ids):
    return hashlib.sha256(json.dumps(ids,separators=(",", ":")).encode()).hexdigest()


def validate_allocator_diagnostic(pool):
    """Preflight before a cache transaction; does not allocate or release."""
    if pool.enable_caching is not False:
        raise AllocatorVerificationError("Allocator verification requires APC disabled")
    if pool.metrics_collector is not None:
        raise AllocatorVerificationError("Allocator verification excludes side-effectful metrics callbacks")
    free=pool.free_block_queue.get_all_free_blocks()
    ids=_ids(free)
    if (len(free)!=pool.get_num_free_blocks() or len(set(ids))!=len(ids)
            or any(block.ref_cnt!=0 or block.is_null or block.block_id<=0 for block in free)):
        raise AllocatorVerificationError("Invalid free-pool ownership before verification")
    return free


def verify_allocator_reacquisition(pool,released_blocks):
    """Reacquire all free pages, including the freed tail, and restore ownership.

This demonstrates actual get_new_blocks/free_blocks operation on the newly freed
page objects. It is deliberately not evidence of attention using them for a
different request; the diagnostic never writes to the physical cache buffers.
"""
    released=tuple(released_blocks)
    if not released:
        return {"kind":"actual_allocator_reacquisition_without_forward", "attempted":False,
                "reason":"no_physical_pages_released"}
    before=validate_allocator_diagnostic(pool)
    before_ids=_ids(before)
    before_objects={id(block) for block in before}
    if (len({id(block) for block in released})!=len(released)
            or any(id(block) not in before_objects for block in released)):
        raise AllocatorVerificationError("A released page was not returned to the actual free pool")
    live_before={block.block_id:block.ref_cnt for block in pool.blocks if block.ref_cnt>0}
    if set(before_ids)&set(live_before):
        raise AllocatorVerificationError("Free queue overlaps a live cache owner")
    borrowed=[]
    acquired_ids=[]
    try:
        borrowed=pool.get_new_blocks(len(before))
        acquired_ids=_ids(borrowed)
        if (acquired_ids!=before_ids or pool.get_num_free_blocks()!=0
                or any(block.ref_cnt!=1 for block in borrowed)
                or any(block.ref_cnt!=1 for block in released)):
            raise AllocatorVerificationError("Real allocator reacquisition violated exclusive ownership")
        if {block.block_id:block.ref_cnt for block in pool.blocks
                if block.block_id in live_before}!=live_before:
            raise AllocatorVerificationError("Allocator diagnostic changed another live owner")
    finally:
        # Return the actual objects received from the allocator, including after
        # an assertion failure in the post-allocation checks. The preflight
        # excludes external metrics callbacks inside get_new_blocks itself.
        if borrowed:
            pool.free_blocks(borrowed)
    after=validate_allocator_diagnostic(pool)
    if (_ids(after)!=before_ids or any(block.ref_cnt!=0 for block in borrowed)
            or {block.block_id:block.ref_cnt for block in pool.blocks if block.ref_cnt>0}!=live_before):
        raise AllocatorVerificationError("Allocator verification did not restore queue/order/live ownership")
    return {"kind":"actual_allocator_reacquisition_without_forward", "attempted":True,
            "released_block_ids":_ids(released),"reacquired_released_block_ids":_ids(released),
            "reacquired_ref_cnt":1,"temporary_blocks":len(acquired_ids),
            "free_count_before":len(before),"free_count_after":len(after),
            "free_order_before_sha256":_digest(before_ids),
            "free_order_after_sha256":_digest(_ids(after)),
            "queue_order_restored":True,"other_live_owners_unchanged":True,
            "all_temporary_owners_released":True,"model_forward_performed":False}
