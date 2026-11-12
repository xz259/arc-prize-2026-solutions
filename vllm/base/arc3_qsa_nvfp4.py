# SPDX-License-Identifier: Apache-2.0
"""Inactive NVFP4 storage primitives for the later main-QSA cache experiment.

Each 256-element K or V head stores 128 packed E2M1 bytes, 16 E4M3FN block
scales (one per 16 values), and one FP32 head scale: 148 bytes per side.
The head scale is dynamic per token/head, rather than a calibrated layer scale.
No runtime backend references this module until separately integrated/tested.
"""
import torch
import triton
import triton.language as tl


SIDE_BYTES = 148
HEAD_DIM = 256


def split_cache(cache):
    if cache.dtype != torch.uint8 or cache.ndim != 4 or cache.shape[-1] != 2*SIDE_BYTES:
        raise ValueError('Invalid packed QSA NVFP4 cache')
    if cache.storage_offset()%4 or cache.stride(-1)!=1 or any(s%4 for s in cache.stride()[:-1]):
        raise ValueError('QSA NVFP4 head scales require four-byte alignment')
    views=[]
    for side in cache.split(SIDE_BYTES,dim=-1):
        data=side[...,:128].transpose(1,2)
        blocks=side[...,128:144].view(torch.float8_e4m3fn).transpose(1,2)
        scale=side[...,144:148].view(torch.float32).squeeze(-1).transpose(1,2)
        views.append((data,blocks,scale))
    return views


@triton.jit
def _pack_e2m1_pair(first,second):
    # NVIDIA CUDA 13 cuda_fp4.hpp __nv_cvt_float2_to_fp4x2 uses this instruction
    # order: the first value occupies the low nibble, second the high nibble.
    return tl.inline_asm_elementwise(
        asm='''{ .reg .b8 packed;
            cvt.rn.satfinite.e2m1x2.f32 packed, $2, $1;
            mov.b16 $0, {packed, 0}; }''',
        constraints='=h,f,f',args=[first,second],dtype=tl.uint16,
        is_pure=True,pack=1).to(tl.uint8)


@triton.jit
def _decode_e2m1(nibble):
    # Exact FP32 encodings of E2M1 values avoid the scalar exp2 conversion.
    # All 16 encodings, including signed zero, were checked independently.
    bits=nibble.to(tl.uint32)
    exponent=(bits >> 1)&3
    mantissa=bits&1
    normal=((exponent+126) << 23) | (mantissa << 22)
    subnormal=tl.where(mantissa!=0,126 << 23,0).to(tl.uint32)
    value=tl.where(exponent==0,subnormal,normal) | ((bits&8) << 28)
    return value.to(tl.float32,bitcast=True)


@triton.jit
def _store_nvfp4_kernel(key,value,cache,slots,
    stride_kr,stride_kh,stride_vr,stride_vh,stride_cb,stride_ch,stride_ct,
    num_blocks,PAGE:tl.constexpr):
    row=tl.program_id(0);head=tl.program_id(1);side=tl.program_id(2)
    slot=tl.load(slots+row)
    if slot>=0 and slot<num_blocks*PAGE:
        # Reduce blocks without transposing or allocating a temporary matrix.
        pair=tl.arange(0,16)[:,None]*8+tl.arange(0,8)[None,:]
        if side==0:
            p=key+row*stride_kr+head*stride_kh
        else:
            p=value+row*stride_vr+head*stride_vh
        x0=tl.load(p+pair*2).to(tl.float32)
        x1=tl.load(p+pair*2+1).to(tl.float32)
        maxima=tl.max(tl.maximum(tl.abs(x0),tl.abs(x1)),axis=1)
        maximum=tl.max(maxima,axis=0)
        head_scale=tl.where(maximum==0.,1.,tl.maximum(tl.div_rn(maximum,2688.),1e-30))
        block_scale=tl.minimum(tl.div_rn(maxima,6.*head_scale),448.).to(tl.float8e4nv)
        dequant=head_scale*block_scale.to(tl.float32)
        safe_dequant=tl.where(dequant>0.,dequant,1.)
        n0=tl.where(dequant[:,None]>0.,tl.div_rn(x0,safe_dequant[:,None]),0.)
        n1=tl.where(dequant[:,None]>0.,tl.div_rn(x1,safe_dequant[:,None]),0.)
        packed=_pack_e2m1_pair(n0,n1)
        address=cache+(slot//PAGE).to(tl.int64)*stride_cb+head*stride_ch+(slot%PAGE)*stride_ct+side*148
        tl.store(address+pair,packed)
        tl.store(address+128+tl.arange(0,16),block_scale.to(tl.uint8,bitcast=True))
        tl.store((address+144).to(tl.pointer_type(tl.float32)),head_scale)


def store_nvfp4(key,value,cache,slots):
    if key.dtype!=torch.bfloat16 or value.dtype!=key.dtype or key.ndim!=3 or key.shape!=value.shape or key.shape[-1]!=256:
        raise ValueError('NVFP4 QSA writes require 256-dimensional BF16 K/V')
    if (slots.ndim!=1 or slots.dtype not in (torch.int32,torch.int64) or slots.stride(0)!=1
            or len(slots)>len(key) or key.stride(-1)!=1 or value.stride(-1)!=1):
        raise ValueError('Unsupported QSA NVFP4 input or slot layout')
    if not key.is_cuda or key.device!=value.device or key.device!=cache.device or key.device!=slots.device:
        raise ValueError('QSA cache tensors must share a CUDA device')
    split_cache(cache)
    if cache.shape[1]!=key.shape[1]:raise ValueError('QSA cache head count mismatch')
    if not len(slots):return
    _store_nvfp4_kernel[(len(slots),key.shape[1],2)](key,value,cache,slots,
        key.stride(0),key.stride(1),value.stride(0),value.stride(1),
        cache.stride(0),cache.stride(1),cache.stride(2),cache.shape[0],PAGE=cache.shape[2],num_warps=4)


@triton.jit
def _unpack_nvfp4_kernel(cache,output,stride_cb,stride_ch,stride_ct,
    PAGE:tl.constexpr,HEADS:tl.constexpr,SIDE:tl.constexpr):
    row=tl.program_id(0);head=tl.program_id(1)
    dims=tl.arange(0,256)
    address=cache+(row//PAGE).to(tl.int64)*stride_cb+head*stride_ch+(row%PAGE)*stride_ct+SIDE*148
    packed=tl.load(address+dims//2)
    nibble=(packed >> ((dims%2)*4))&15
    blocks=tl.load(address+128+dims//16).to(tl.float8e4nv,bitcast=True).to(tl.float32)
    head_scale=tl.load((address+144).to(tl.pointer_type(tl.float32)))
    decoded=_decode_e2m1(nibble)*(blocks*head_scale)
    tl.store(output+row.to(tl.int64)*HEADS*256+head*256+dims,decoded)


def unpack_nvfp4(cache):
    """Dense BF16 reconstruction for unit tests only; not the serving read path."""
    split_cache(cache)
    blocks,heads,page,_=cache.shape
    outputs=[]
    for side in range(2):
        out=torch.empty((blocks,page,heads,256),device=cache.device,dtype=torch.bfloat16)
        _unpack_nvfp4_kernel[(blocks*page,heads)](cache,out,*cache.stride()[:3],
            PAGE=page,HEADS=heads,SIDE=side,num_warps=4)
        outputs.append(out)
    return outputs
