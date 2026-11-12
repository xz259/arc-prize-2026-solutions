"""Synthetic CPU/CUDA byte-copy checks; no model/server/network access.

CPU cases run wherever torch is installed. CUDA cases additionally check the
asynchronous gather/scatter ordering used by the worker implementation.
"""
import unittest

from eviction_validation import stored_rows_digest
try:
    import torch
except ImportError:
    torch=None
from eviction_storage import page_compaction,compact_cache_rows_,EvictionStorageError


class StoredBytesCases:
    device="cpu"

    def verify(self,shape,token_axis,dtype,keep,old_rows,blocks,chunk,
               *,scratch_bytes=16*1024*1024,noncontiguous=False):
        allocated_shape=(shape[0],shape[2],shape[1],shape[3]) if noncontiguous else shape
        if dtype==torch.uint8:
            cache=torch.randint(0,256,allocated_shape,dtype=torch.uint8,device=self.device)
        else:
            byte_shape=(*allocated_shape[:-1],allocated_shape[-1]*2)
            cache=torch.randint(0,256,byte_shape,dtype=torch.uint8,device=self.device).view(dtype)
        if noncontiguous:
            cache=cache.transpose(1,2)
            self.assertFalse(cache.is_contiguous())
        before=cache.view(torch.uint8).clone().movedim(token_axis,1)
        plan=page_compaction(blocks,keep,old_rows=old_rows,storage_block_size=shape[token_axis])
        source_hash=stored_rows_digest(cache,plan,token_axis=token_axis,chunk_rows=chunk)
        receipt=compact_cache_rows_(cache,plan,token_axis=token_axis,scratch_rows=chunk,
                                    scratch_bytes=scratch_bytes)
        if self.device=="cuda":torch.cuda.synchronize()
        after=cache.view(torch.uint8).movedim(token_axis,1)
        size=shape[token_axis]
        source=torch.tensor(keep,device=self.device);dest=torch.arange(len(keep),device=self.device)
        pages=torch.tensor(blocks,device=self.device)
        self.assertTrue(torch.equal(after[pages[dest//size],dest%size],
                                    before[pages[source//size],source%size]))
        untouched=[i for i in range(shape[0]) if i not in plan.kept_block_ids]
        self.assertTrue(torch.equal(after[untouched],before[untouched]))
        self.assertEqual(receipt["retained_rows"],len(keep))
        self.assertLessEqual(receipt["max_gather_bytes"],scratch_bytes)
        self.assertEqual(receipt["index_vector_bytes"],receipt["copied_rows"]*4*8)
        destination_hash=stored_rows_digest(cache,plan,token_axis=token_axis,after=True,chunk_rows=chunk)
        self.assertEqual(source_hash,destination_hash)
        self.assertEqual(source_hash["bytes"],len(keep)*receipt["row_bytes"])
        # A simulated new owner overwrites every released byte; survivors must
        # remain exactly intact. Real allocator reuse is a separate live test.
        survivor=after[pages[dest//size],dest%size].clone()
        for page in plan.released_block_ids:after[page].fill_(0xD3)
        self.assertTrue(torch.equal(after[pages[dest//size],dest%size],survivor))
        self.assertEqual(stored_rows_digest(cache,plan,token_axis=token_axis,after=True)["sha256"],source_hash["sha256"])
        after[(blocks[0],0)+(0,)*(after.ndim-2)]^=1
        self.assertNotEqual(stored_rows_digest(cache,plan,token_axis=token_axis,after=True)["sha256"],source_hash["sha256"])

    def test_nvfp4_complete_scale_rows_and_overlap(self):
        for chunk in (1,7,64,8192):
            self.verify((17,2,32,296),2,torch.uint8,
                        list(range(7))+list(range(69,100)),100,[5,2,11,1],chunk)

    def test_bf16_descriptor_bit_patterns(self):
        for chunk in (1,19,64,8192):
            self.verify((17,16,1,128),1,torch.bfloat16,
                        [0,2,3,9,14,18,19,21,24,30,37,40],42,[8,3,12],chunk)

    def test_noncontiguous_nvfp4_token_axis_and_bounded_scratch(self):
        self.verify((17,2,32,296),2,torch.uint8,
                    [*range(7),*range(18,100)],100,[5,2,11,1],8192,
                    scratch_bytes=592*11,noncontiguous=True)

    def test_noncontiguous_bf16_descriptor_bytes(self):
        self.verify((17,16,2,128),1,torch.bfloat16,
                    [0,2,3,9,14,18,19,21,24,30,37,40],42,[8,3,12],8192,
                    scratch_bytes=512*5,noncontiguous=True)

    def test_large_chunk_boundary_overlapping_rows(self):
        self.verify((32,1,512,296),2,torch.uint8,
                    [*range(7),*range(8,15000)],15000,list(reversed(range(1,31))),8192)

    def test_byte_bound_failure_leaves_original_cache_untouched(self):
        cache=torch.randint(0,256,(3,2,16,296),dtype=torch.uint8,device=self.device)
        original=cache.clone()
        plan=page_compaction([2,1],[0,2,3],old_rows=17,storage_block_size=16)
        with self.assertRaisesRegex(EvictionStorageError,"exceeds"):
            compact_cache_rows_(cache,plan,token_axis=2,scratch_bytes=591)
        self.assertTrue(torch.equal(cache,original))


@unittest.skipUnless(torch is not None,"PyTorch required")
class CPUStoredBytes(StoredBytesCases,unittest.TestCase):
    pass


@unittest.skipUnless(torch is not None and torch.cuda.is_available(),"CUDA required")
class GPUStoredBytes(StoredBytesCases,unittest.TestCase):
    device="cuda"


if __name__=="__main__":unittest.main()
