"""Execute pinned block-pool methods and queue code on CPU metadata objects."""
from __future__ import annotations

import ast
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import sys
from types import MethodType,SimpleNamespace as NS
import unittest
from unittest.mock import patch

import engine_eviction
from eviction_allocator import (AllocatorVerificationError,validate_allocator_diagnostic,
                               verify_allocator_reacquisition)

ROOT=Path(os.environ.get("ARC3_RUNTIME_SOURCE",str(
    Path(__file__).resolve().parent.parent/"persistent_state_20260912"/"source-snapshot"/"vllm")))


@dataclass
class Block:
    block_id: int
    ref_cnt: int=0
    block_hash: object=None
    is_null: bool=False
    prev_free_block: object=None
    next_free_block: object=None


def source_class(relative,name,methods=None):
    data=(ROOT/relative).read_bytes()
    if hashlib.sha256(data).hexdigest()!=engine_eviction.SOURCE_HASHES[relative]:
        raise AssertionError("Allocator test source is not the pinned runtime")
    tree=ast.parse(data)
    cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name==name)
    if methods is not None:
        cls.body=[n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name in methods]
    namespace={"KVCacheBlock":Block}
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),
                 str(ROOT/relative),"exec",flags=__import__("__future__").annotations.compiler_flag),namespace)
    return namespace[name]


def actual_pool(count=20):
    Queue=source_class("v1/core/kv_cache_utils.py","FreeKVCacheBlockQueue")
    Methods=source_class("v1/core/block_pool.py","BlockPool",
                         {"get_new_blocks","free_blocks","get_num_free_blocks"})
    blocks=[Block(i) for i in range(count)]
    queue=Queue(blocks)
    null=queue.popleft();null.is_null=True
    pool=NS(blocks=blocks,free_block_queue=queue,enable_caching=False,metrics_collector=None)
    for name in ("get_new_blocks","free_blocks","get_num_free_blocks"):
        setattr(pool,name,MethodType(getattr(Methods,name),pool))
    return pool


class AllocatorReacquisition(unittest.TestCase):
    def test_real_allocation_returns_freed_objects_and_restores_order(self):
        pool=actual_pool()
        live=pool.get_new_blocks(6)
        released=[live[3],live[5],live[4]]
        pool.free_blocks(released)
        before=[b.block_id for b in validate_allocator_diagnostic(pool)]
        self.assertEqual(before[:3],[4,6,5]) # Actual APC-off prepend semantics.
        refs={b.block_id:b.ref_cnt for b in live[:3]}
        calls=[]
        actual=pool.get_new_blocks
        def tracked(n):
            acquired=actual(n)
            calls.append((n,[b.block_id for b in acquired],tuple(b.ref_cnt for b in released)))
            return acquired
        pool.get_new_blocks=tracked
        receipt=verify_allocator_reacquisition(pool,released)
        self.assertEqual(calls,[(len(before),before,(1,1,1))])
        self.assertEqual(receipt["reacquired_released_block_ids"],[4,6,5])
        self.assertEqual(receipt["free_order_before_sha256"],receipt["free_order_after_sha256"])
        self.assertEqual(receipt["free_count_before"],receipt["free_count_after"])
        self.assertTrue(receipt["all_temporary_owners_released"])
        self.assertFalse(receipt["model_forward_performed"])
        self.assertEqual([b.block_id for b in validate_allocator_diagnostic(pool)],before)
        self.assertEqual({b.block_id:b.ref_cnt for b in live[:3]},refs)

    def test_single_newly_freed_page_is_reacquired_with_empty_remainder(self):
        pool=actual_pool(5)
        live=pool.get_new_blocks(4)
        pool.free_blocks([live[2]])
        receipt=verify_allocator_reacquisition(pool,[live[2]])
        self.assertEqual(receipt["temporary_blocks"],1)
        self.assertEqual([b.ref_cnt for b in live],[1,1,0,1])

    def test_no_freed_page_does_not_claim_reuse(self):
        pool=actual_pool()
        before=pool.free_block_queue.get_all_free_blocks()
        receipt=verify_allocator_reacquisition(pool,[])
        self.assertFalse(receipt["attempted"])
        self.assertEqual(pool.free_block_queue.get_all_free_blocks(),before)

    def test_post_allocation_check_failure_still_returns_every_temporary_owner(self):
        pool=actual_pool()
        live=pool.get_new_blocks(2)
        pool.free_blocks([live[1]])
        before=pool.free_block_queue.get_all_free_blocks()
        actual_count=pool.get_num_free_blocks
        def bad_after_allocate_count():
            result=actual_count()
            return result if result else 1
        pool.get_num_free_blocks=bad_after_allocate_count
        with self.assertRaisesRegex(AllocatorVerificationError,"reacquisition"):
            verify_allocator_reacquisition(pool,[live[1]])
        self.assertEqual(pool.free_block_queue.get_all_free_blocks(),before)
        self.assertTrue(all(block.ref_cnt==0 for block in before))
        self.assertEqual(live[0].ref_cnt,1)

    def test_unsupported_callbacks_and_cached_pool_rejected_without_mutation(self):
        for field,value in (("enable_caching",True),("metrics_collector",object())):
            with self.subTest(field=field):
                pool=actual_pool()
                before=pool.free_block_queue.get_all_free_blocks()
                setattr(pool,field,value)
                with self.assertRaises(AllocatorVerificationError):
                    validate_allocator_diagnostic(pool)
                self.assertEqual(pool.free_block_queue.get_all_free_blocks(),before)

    def test_live_or_fake_released_object_rejected_without_allocation(self):
        for fake in (False,True):
            with self.subTest(fake=fake):
                pool=actual_pool()
                live=pool.get_new_blocks(2)
                released=[Block(7)] if fake else [live[0]]
                before=pool.free_block_queue.get_all_free_blocks()
                with self.assertRaisesRegex(AllocatorVerificationError,"actual free pool"):
                    verify_allocator_reacquisition(pool,released)
                self.assertEqual(pool.free_block_queue.get_all_free_blocks(),before)
                self.assertEqual([b.ref_cnt for b in live],[1,1])

    def test_core_optional_diagnostic_runs_after_release_before_ack(self):
        import engine_patch
        from test_engine_eviction import fixture,fake_specs
        f=fixture()
        pool=actual_pool(12)
        owned=pool.get_new_blocks(6)
        f.managers[0].req_to_blocks[f.request.request_id]=owned[:5]
        f.managers[1].req_to_blocks[f.request.request_id]=owned[5:]
        f.core.scheduler.kv_cache_manager.block_pool=pool
        with patch.dict(sys.modules,fake_specs()),patch.dict(os.environ,{"ARC3_EVICTION_VERIFY_ALLOCATOR":"1"}),\
                patch.object(engine_patch,"_audit",lambda *args,**kwargs:None):
            ack=engine_eviction.evict_resident(f.core,"example",f.payload)
        receipt=ack["allocator_reacquisition"]
        self.assertEqual(receipt["released_block_ids"],[4,5])
        self.assertEqual(receipt["temporary_blocks"],7)
        self.assertEqual([b.block_id for b in pool.free_block_queue.get_all_free_blocks()],[4,5,7,8,9,10,11])
        self.assertEqual([b.ref_cnt for b in owned],[1,1,1,0,0,1])
        self.assertFalse(f.request._arc3_eviction_state.poisoned)


if __name__=="__main__":unittest.main()
