import random
import unittest
from dataclasses import replace
from math import ceil, log2

from eviction_storage import (token_compaction, page_compaction, EvictionStorageError,
                              row_copy_schedule)
from engine_eviction import LogicalState,_demand_wrapper
from residency import ResidentLedger


class StoragePlans(unittest.TestCase):
    def test_original_positions_survive_repeated_deletion(self):
        old=list(range(37)); kept=[x for x in old if not 5<=x<18]
        first=token_compaction(old,kept,old_computed=36)
        self.assertEqual(first.new_computed,23)
        self.assertEqual(first.computed_keep_indices,tuple([*range(5),*range(18,36)]))
        second_kept=[x for x in kept if not 22<=x<28]
        second=token_compaction(kept,second_kept,old_computed=len(kept)-1)
        self.assertEqual([kept[i] for i in second.keep_indices],second_kept)
        self.assertEqual(second.kept_positions[-1],36)

    def test_uncomputed_tail_is_not_copied_or_deleted(self):
        plan=token_compaction(range(16),[0,1,12,13,14,15],old_computed=15)
        self.assertEqual(plan.computed_keep_indices,(0,1,12,13,14))
        for positions in ([0,1,2],[],[15]):
            with self.assertRaises(EvictionStorageError):
                token_compaction(range(16),positions,old_computed=15)

    def test_bad_mappings_rejected(self):
        for old,kept in [([0,0,1],[0,1]),([2,1],[1]),([0,1,2],[0,9,2]),
                         ([0,1,2],[1,0,2]),([0,1,2],[False,2])]:
            with self.assertRaises(EvictionStorageError):
                token_compaction(old,kept,old_computed=len(old)-1)

    def test_exact_tail_page_release(self):
        page=page_compaction([9,2,7,5],list(range(10))+list(range(73,100)),
                             old_rows=100,storage_block_size=32)
        self.assertEqual(page.kept_block_ids,(9,2))
        self.assertEqual(page.released_block_ids,(7,5))
        self.assertEqual(page.retained_rows,37)

    def test_small_deletion_creates_tail_slack_without_false_free(self):
        page=page_compaction([9,2],list(range(8))+list(range(9,60)),
                             old_rows=60,storage_block_size=32)
        self.assertEqual(page.released_block_ids,())
        self.assertEqual(64-page.retained_rows,5)

    def test_null_alias_outside_rows_rejected(self):
        for blocks,rows,n,b in [([0],[],0,4),([1,1],[0],1,4),([1],[4],4,4),
                               ([1],[0],5,4),([1],[],0,0)]:
            with self.assertRaises(EvictionStorageError):
                page_compaction(blocks,rows,old_rows=n,storage_block_size=b)

    def test_random_nonoverlap_move_schedule(self):
        rng=random.Random(157)
        for length in range(3,150):
            keep=sorted(rng.sample(range(length-1),rng.randrange(1,length)))+[length-1]
            original=[bytes([i%256])*7 for i in range(length)]
            for chunk in (1,3,16,64):
                working=list(original)
                for begin in range(0,len(keep),chunk):
                    subset=keep[begin:begin+chunk]
                    temporary=[working[i] for i in subset]
                    working[begin:begin+len(subset)]=temporary
                self.assertEqual(working[:len(keep)],[original[i] for i in keep])

    def test_byte_budget_and_exact_unchanged_prefix(self):
        page=page_compaction([5,3,8], [*range(13),*range(21,95)],
                             old_rows=95,storage_block_size=32)
        schedule=row_copy_schedule(page,row_bytes=592,scratch_rows=8192,
                                   scratch_bytes=7*592+591)
        self.assertEqual(schedule.first_moved,13)
        self.assertEqual(schedule.scratch_row_bound,7)
        self.assertEqual(schedule.copied_rows,74)
        self.assertEqual(schedule.chunks[0],(13,20))
        self.assertEqual(schedule.chunks[-1],(83,87))
        self.assertTrue(all((end-begin)*592<=schedule.scratch_byte_bound
                            for begin,end in schedule.chunks))

    def test_scratch_configuration_rejected_before_movement(self):
        page=page_compaction([1],[1,2],old_rows=4,storage_block_size=4)
        for options in (dict(row_bytes=0),dict(row_bytes=True),
                        dict(row_bytes=8,scratch_rows=0),
                        dict(row_bytes=8,scratch_bytes=7),
                        dict(row_bytes=8,scratch_bytes=False)):
            with self.assertRaises(EvictionStorageError):
                row_copy_schedule(page,**options)

    def test_noop_and_tail_only_deletion_allocate_no_gather(self):
        for keep in ([],list(range(3)),list(range(8))):
            page=page_compaction([2,1],keep,old_rows=8,storage_block_size=4)
            schedule=row_copy_schedule(page,row_bytes=256,scratch_bytes=1)
            self.assertEqual(schedule.chunks,())
            self.assertEqual(schedule.copied_rows,0)

    def test_bounded_schedule_preserves_overlap_and_nonmonotonic_pages(self):
        rng=random.Random(95013)
        for length in (3,32,99,1023,16387):
            pages=list(range(1,(length+31)//32+1));rng.shuffle(pages)
            original={page:[(page,row) for row in range(32)] for page in pages}
            keep=sorted(rng.sample(range(length),max(1,length*9//10)))
            expected=[original[pages[i//32]][i%32] for i in keep]
            plan=page_compaction(pages,keep,old_rows=length,storage_block_size=32)
            for rows,budget in ((1,592),(64,64*592),(8192,16*1024*1024),
                                (8192,1000*592-1)):
                schedule=row_copy_schedule(plan,row_bytes=592,scratch_rows=rows,
                                           scratch_bytes=budget)
                working={page:list(values) for page,values in original.items()}
                for begin,end in schedule.chunks:
                    gathered=[working[pages[i//32]][i%32] for i in keep[begin:end]]
                    for i,value in zip(range(begin,end),gathered):
                        working[pages[i//32]][i%32]=value
                actual=[working[pages[i//32]][i%32] for i in range(len(keep))]
                self.assertEqual(actual,expected)
                self.assertTrue(all(working[p]==original[p] for p in plan.released_block_ids))

    def test_long_cache_moves_in_one_chunk_with_the_raised_defaults(self):
        plan=page_compaction(range(1,41),[*range(17),*range(4017,39999)],
                             old_rows=39999,storage_block_size=1024)
        schedule=row_copy_schedule(plan,row_bytes=296)
        self.assertEqual(schedule.first_moved,17)
        self.assertEqual(len(schedule.chunks),1)   # 2026-09-24 stage E defaults (49152 rows / 32 MiB)
        self.assertLess(schedule.scratch_row_bound*296,16*1024*1024)
        legacy=row_copy_schedule(plan,row_bytes=296,scratch_rows=8192,scratch_bytes=16*1024*1024)
        self.assertEqual(len(legacy.chunks),5)   # the pre-stage-E bounds still chunk five ways

    def test_prefix_boundary_matches_linear_reference_for_general_deletions(self):
        rng=random.Random(95113)
        for length in (1,2,3,15,16,17,63,64,65,999):
            candidates=[[],list(range(length)),list(range(length-1)),
                        list(range(1,length))]
            candidates += [sorted(rng.sample(range(length),rng.randrange(length+1)))
                           for _ in range(20)]
            for keep in candidates:
                plan=page_compaction(range(1,ceil(length/16)+1),keep,
                                     old_rows=length,storage_block_size=16)
                expected=next((i for i,source in enumerate(keep) if source!=i),len(keep))
                schedule=row_copy_schedule(plan,row_bytes=592)
                self.assertEqual(schedule.first_moved,expected)

    def test_near_tail_prefix_search_uses_logarithmic_index_reads(self):
        class IndexedTuple(tuple):
            reads=0
            def __getitem__(self,index):
                self.reads+=1
                return super().__getitem__(index)
            def __iter__(self):
                raise AssertionError("Prefix boundary must not linearly scan retained rows")

        length=65536
        for keep in (list(range(length)),list(range(length-1)),
                     [*range(length-2),length-1],list(range(1,length)),[]):
            validated=page_compaction(range(1,65),keep,old_rows=length,storage_block_size=1024)
            indexed=IndexedTuple(validated.keep_rows)
            schedule=row_copy_schedule(replace(validated,keep_rows=indexed),row_bytes=592)
            expected=next((i for i,source in enumerate(keep) if source!=i),len(keep))
            self.assertEqual(schedule.first_moved,expected)
            self.assertLessEqual(indexed.reads,ceil(log2(len(keep)+1)))

    def test_group_padding_is_not_token_div_four(self):
        state=LogicalState(list(range(50)),ResidentLedger(50,[(1,8),(11,13)]),50,
                           (),[])
        self.assertEqual(state.deleted_tokens,9)
        self.assertEqual(state.deleted_groups,1)
        self.assertEqual(state.group_padding,5)

    def test_virtual_allocator_demand_and_capacity(self):
        class Manager:
            _arc3_has_compressed=True
            _arc3_table_capacity=60
        state=LogicalState(list(range(50)),ResidentLedger(50,[(1,8),(11,13)]),50,(),[])
        manager=Manager(); manager._arc3_eviction_states={"a":state}
        def original(self,request_id,num_tokens,num_tokens_main_model):
            return num_tokens,num_tokens_main_model
        method=_demand_wrapper(original)
        self.assertEqual(method(manager,"a",40,40),(45,45))
        self.assertEqual(method(manager,"baseline",40,40),(40,40))
        with self.assertRaisesRegex(RuntimeError,"capacity"):
            method(manager,"a",56,56)


if __name__=="__main__":unittest.main()
