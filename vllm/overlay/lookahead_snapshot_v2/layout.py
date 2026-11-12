"""Strict owner geometry and chronological read plans, without Torch imports."""
from .receipt_schema import expected_owners, require, same, MAX_CHUNK_BYTES


def spec_record(spec):
    fields = ('block_size', 'storage_block_size', 'num_kv_heads', 'head_size',
              'head_size_v', 'state_content_bytes', 'compress_ratio',
              'page_size_padded', 'indexes_kv_by_block_stride',
              'sliding_window', 'attention_chunk_size', 'non_causal',
              'mamba_cache_mode', 'num_speculative_blocks', 'tp_replicated')
    result = {'type': type(spec).__name__}
    for field in fields:
        if hasattr(spec, field):
            result[field] = getattr(spec, field)
    if hasattr(spec, 'dtype'):
        result['dtype'] = str(spec.dtype)
    for field in ('kv_quant_mode', 'mamba_type'):
        if hasattr(spec, field):
            value = getattr(spec, field)
            result[field] = type(value).__name__ + '.' + value.name
    if hasattr(spec, 'shapes'):
        result['shapes'] = [list(x) for x in spec.shapes]
        result['dtypes'] = [str(x) for x in spec.dtypes]
    return result


def validate_specs(groups):
    """Reject normalized or expanded inventory; return real per-owner specs."""
    require(len(groups) == 6, 'Expected exactly six cache groups')
    found, records = {}, []
    for group_id, group in enumerate(groups):
        names = list(group.layer_names)
        require(len(names) == len(set(names)), 'Duplicate owner within group')
        spec = group.kv_cache_spec
        if group_id < 2:
            require(type(spec).__name__ == 'UniformTypeKVCacheSpecs', 'Lost physical owner specifications')
            members = spec.kv_cache_specs
            require(set(members) == set(names), 'Owner map differs from group names')
        else:
            require(type(spec).__name__ == 'MambaSpec', 'Unknown recurrent group')
            members = {name: spec for name in names}
        for name in names:
            require(name not in found, 'Duplicate owner across groups')
            found[name] = members[name]
            records.append((name, group_id))
    require(sorted(records) == sorted((x['owner'], x['group_id']) for x in expected_owners()),
            'Unknown or missing cache owner')
    for owner in expected_owners():
        s = found[owner['owner']]
        categories = owner['categories']
        if categories == ['main_nvfp4']:
            expected = dict(type='FullAttentionSpec', block_size=2768, storage_block_size=2768,
                num_kv_heads=2, head_size=256, head_size_v=256, dtype='torch.uint8',
                state_content_bytes=296, kv_quant_mode='KVQuantMode.NVFP4')
        elif categories == ['qsa_compressed']:
            expected = dict(type='MLAAttentionSpec', block_size=2768, storage_block_size=692,
                num_kv_heads=1, head_size=128, head_size_v=0, dtype='torch.bfloat16', compress_ratio=4,
                kv_quant_mode='KVQuantMode.NONE')
        elif categories == ['qsa_raw_key', 'qsa_raw_rope_position']:
            expected = dict(type='CircularBufferSpec', block_size=4, storage_block_size=4,
                num_kv_heads=1, head_size=140, head_size_v=0, dtype='torch.bfloat16',
                page_size_padded=177152, kv_quant_mode='KVQuantMode.NONE')
        else:
            ple = categories == ['ple_short_conv']
            expected = dict(type='MambaSpec', block_size=40000,
                shapes=[[9, 10240]] if ple else [[3, 10240], [48, 128, 128]],
                dtypes=['torch.bfloat16'] * (1 if ple else 2), mamba_cache_mode='none',
                num_speculative_blocks=0, tp_replicated=ple, page_size_padded=1638656,
                mamba_type='MambaAttentionBackendEnum.' + ('SHORT_CONV' if ple else 'GDN_ATTN'))
        actual = spec_record(s)
        require(all(same(actual.get(k), v) for k, v in expected.items()),
                'Unsupported spec: ' + owner['owner'])
        if owner['group_id'] < 2:
            require(s.indexes_kv_by_block_stride is True, 'Unsupported page addressing')
            for field in ('sliding_window', 'attention_chunk_size'):
                require(getattr(s, field, None) is None, 'Unsupported attention window')
            require(getattr(s, 'non_causal', False) is False, 'Noncausal cache unsupported')
    return found


def validate_block_ids(groups):
    require(type(groups) is list and len(groups) == 6, 'Expected six physical block lists')
    for group in groups:
        require(type(group) is list and len(group) == 1,
                'Bounded C<=2048 scope requires one owned block per group')
        require(all(type(x) is int and x > 0 for x in group), 'Null or malformed physical block')
    # Group block IDs may coincide: distinct groups have distinct physical pools.
    return groups


def allocation_records(config, specs):
    """Match pinned attn_utils packed/unpacked allocation, without fixed pool size."""
    result = {}
    for allocation in config.kv_cache_tensors:
        size, offset, stride = allocation.size, allocation.offset, allocation.block_stride
        require(all(type(x) is int for x in (size, offset, stride))
                and size > 0 and offset >= 0 and stride >= 0, 'Malformed allocation descriptor')
        for name in allocation.shared_by:
            require(name in specs and name not in result, 'Unknown/duplicate allocation owner')
            spec = specs[name]
            page = 177152 if (type(spec).__name__ in ('MLAAttentionSpec', 'CircularBufferSpec')) else 1638656
            physical_stride = stride or page
            require(size % physical_stride == 0 and offset+page <= physical_stride,
                    'Allocation page extent exceeds physical stride')
            require(stride > 0 or offset == 0, 'Unpacked allocation has offset')
            require(size//physical_stride == config.num_blocks, 'Allocation block count differs')
            result[name] = dict(size=size, offset=offset, stride=physical_stride,
                                page=page, num_blocks=size//physical_stride)
    require(set(result) == set(specs), 'Missing physical allocation owner')
    return result


def require_disjoint_ranges(ranges):
    """Different owner pages may share backing/base, never selected byte ranges."""
    ordered = sorted(ranges)
    require(all(type(start) is int and type(end) is int and 0 < start < end
                for start,end,_ in ordered), 'Malformed owned page extent')
    for (start, end, name), following in zip(ordered, ordered[1:]):
        require(end <= following[0], 'Owned cache page overlap: '+name)


def row_chunks(rows, row_bytes, block_size, block_ids, *, ring=False):
    """Yield (physical block, offset, row count) without reading inactive padding."""
    require(type(row_bytes) is int and 0 < row_bytes <= MAX_CHUNK_BYTES, 'Invalid row width')
    require(type(block_size) is int and block_size > 0, 'Invalid storage block size')
    positions = list(rows)
    require(all(type(x) is int and x >= 0 for x in positions), 'Invalid logical row')
    limit = MAX_CHUNK_BYTES // row_bytes
    i = 0
    while i < len(positions):
        logical = positions[i]
        page = 0 if ring else logical // block_size
        require(page < len(block_ids), 'Computed row lacks an owned page')
        offset = logical % block_size
        count = 1
        while (i+count < len(positions) and count < limit
               and positions[i+count] == logical+count and offset+count < block_size):
            count += 1
        yield block_ids[page], offset, count
        i += count


def hash_chunks(chunks, expected_bytes):
    import hashlib
    digest, total = hashlib.sha256(), 0
    for raw in chunks:
        require(type(raw) is bytes and 0 < len(raw) <= MAX_CHUNK_BYTES, 'Unbounded/empty byte copy')
        total += len(raw)
        require(total <= expected_bytes, 'Read exceeded selected-byte budget')
        digest.update(raw)
    require(total == expected_bytes, 'Missing selected bytes')
    return digest.hexdigest()
