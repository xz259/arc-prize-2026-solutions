"""Option B makes the compressed-key view strided (block stride 184,320 B over 177,152 B of rows). The overlay's
eviction compaction and byte digests must give the same result on such a view as on a contiguous cache, and must
leave the padding bytes untouched."""
import sys, torch
sys.path.insert(0, '/workspace/arc3-inference/runs/dense_fp8_v1')
from eviction_storage import page_compaction, compact_cache_rows_
from eviction_validation import stored_rows_digest
torch.manual_seed(0)
nb, rows, width = 9, 692, 128; page_elems = 184320 // 2
raw = torch.randint(-32768, 32767, (nb * page_elems,), dtype=torch.int16).view(torch.bfloat16)
strided = torch.as_strided(raw, size=(nb, rows, 1, width), stride=(page_elems, width, width, 1))
assert not strided.is_contiguous() and strided.stride(0) == page_elems
dense = strided.clone(); assert dense.is_contiguous()
pad_before = raw.view(torch.int16).view(nb, page_elems)[:, rows * width:].clone()   # bit compare (NaN payloads)
blocks = [3, 5, 1, 7]; old_rows = 4 * rows - 17
keep = tuple(i for i in range(old_rows) if i % 7 != 3 and not (900 <= i < 1300))
plan = page_compaction(blocks, keep, old_rows=old_rows, storage_block_size=rows)
d0 = stored_rows_digest(dense, plan, token_axis=1); s0 = stored_rows_digest(strided, plan, token_axis=1)
assert d0 == s0, 'pre-compaction digests differ between dense and strided views'
r1 = compact_cache_rows_(dense, plan, token_axis=1); r2 = compact_cache_rows_(strided, plan, token_axis=1)
assert r1['copied_rows'] == r2['copied_rows'] > 0 and r1['copy_chunks'] == r2['copy_chunks']
assert torch.equal(strided.view(torch.int16), dense.view(torch.int16)), 'compaction result differs on the strided view'   # bit compare: random patterns include NaNs
d1 = stored_rows_digest(dense, plan, token_axis=1, after=True); s1 = stored_rows_digest(strided, plan, token_axis=1, after=True)
assert d1 == s1 == d0
assert torch.equal(raw.view(torch.int16).view(nb, page_elems)[:, rows * width:], pad_before), 'padding bytes were touched'
print(f'STRIDED COMPACTION OK: {r2["copied_rows"]} rows moved in {r2["copy_chunks"]} chunks; digests equal; padding intact')
