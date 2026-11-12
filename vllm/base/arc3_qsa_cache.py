"""Opt-in main-QSA storage dispatch; side/indexer/GDN caches stay unchanged.

The physical byte layout is private to these readers/writers. KVQuantMode is
allocator metadata, not a claim of compatibility with another backend's bytes.
"""

import os

FORMATS = ('fp8_e4m3', 'fp8_e5m2', 'nvfp4')
FP8_READER = os.environ.get('ARC3_QSA_FP8_READER', 'baseline')
if FP8_READER not in {'baseline', 'scale_placement'}:
    raise ValueError('Unsupported FP8 reader')
READER_CALLS = {'baseline': 0, 'scale_placement': 0}


def reader_status():
    return dict(fp8_reader=FP8_READER, calls=dict(READER_CALLS),
                scope='Launch-time choice; counts include warmup/capture, not every graph replay')


def side_bytes(head_dim, fmt):
    if head_dim != 256 or fmt not in FORMATS:
        raise ValueError('Quantized QSA currently requires D256 and a supported format')
    return 148 if fmt == 'nvfp4' else 260


def quant_mode(fmt):
    from vllm.v1.kv_cache_interface import KVQuantMode
    if fmt not in FORMATS:
        raise ValueError('Unsupported QSA storage format')
    return KVQuantMode.NVFP4 if fmt == 'nvfp4' else KVQuantMode.FP8_PER_TOKEN_HEAD


def customize_main_spec(spec, fmt):
    from dataclasses import replace
    import torch
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    # The side backend may be chosen for the synthetic main-cache alignment
    # probe. Its actual MLA and circular-buffer specs must remain unchanged.
    if type(spec) is not FullAttentionSpec:
        return spec
    return replace(spec, dtype=torch.uint8, kv_quant_mode=quant_mode(fmt),
                   state_content_bytes=2*side_bytes(spec.head_size, fmt))


def store(key, value, cache, slots, fmt):
    if fmt == 'nvfp4':
        from arc3_qsa_nvfp4 import store_nvfp4
        return store_nvfp4(key, value, cache, slots)
    if fmt not in FORMATS:
        raise ValueError('Unsupported QSA storage format')
    from arc3_qsa_quant import store_fp8
    return store_fp8(key, value, cache, slots, fmt)


def attention(query, cache, indices, block_table, requests, output, fmt):
    if fmt == 'nvfp4':
        from arc3_qsa_nvfp4_attention import attention_nvfp4
        return attention_nvfp4(query, cache, indices, block_table, requests, output)
    if fmt not in FORMATS:
        raise ValueError('Unsupported QSA storage format')
    if FP8_READER == 'scale_placement':
        from arc3_qsa_fp8_scale_placement import attention_fp8
    else:
        from arc3_qsa_quant import attention_fp8
    READER_CALLS[FP8_READER] += 1
    return attention_fp8(query, cache, indices, block_table, requests, output, fmt)
