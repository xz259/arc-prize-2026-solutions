"""Opt-in synchronous evidence for actual retained cache bytes.

This diagnostic intentionally transfers bounded byte chunks to CPU and is not a
throughput optimization. No model values are written to logs; only SHA256
digests and byte counts are returned. It does not certify untouched recurrent
or raw-ring state; those owners are excluded from the compaction transaction.
"""
from __future__ import annotations

import hashlib


def stored_rows_digest(cache,plan,*,token_axis,after=False,chunk_rows=64):
    """Hash retained rows in final resident order, before or after compaction."""
    import torch
    from eviction_storage import EvictionStorageError,PageCompaction

    if (not isinstance(plan,PageCompaction) or type(chunk_rows) is not int or chunk_rows<1
            or type(token_axis) is not int or not 0<token_axis<cache.ndim-1
            or cache.stride(-1)!=1 or cache.shape[token_axis]!=plan.storage_block_size
            or (plan.old_block_ids and max(plan.old_block_ids)>=cache.shape[0])):
        raise EvictionStorageError("Unsupported byte-verification plan or layout")
    view=cache.view(torch.uint8).movedim(token_axis,1)
    rows=tuple(range(len(plan.keep_rows))) if after else plan.keep_rows
    blocks=torch.tensor(plan.old_block_ids,dtype=torch.int64,device=cache.device)
    checksum=hashlib.sha256()
    byte_count=0
    for begin in range(0,len(rows),chunk_rows):
        ids=torch.tensor(rows[begin:begin+chunk_rows],dtype=torch.int64,device=cache.device)
        chunk=view[blocks[ids//plan.storage_block_size],ids%plan.storage_block_size]
        data=chunk.detach().to(device="cpu").contiguous().numpy().tobytes()
        checksum.update(data)
        byte_count+=len(data)
    return {"sha256":checksum.hexdigest(),"bytes":byte_count,"rows":len(rows),
            "chunk_row_bound":chunk_rows}
