"""stage2_v1 'kvrot' part: main-QSA NVFP4 KV stored Hadamard-rotated with block-MSE scales, the `h_nvfp4m` format of the
milestone-2 KV study (kv_study 2026-10-02: prefill KL 0.0267 against NVFP4's 0.0302, decode hazard 0.052 against
0.133; user adopted it 01:35).

Format: unchanged NVFP4 layout (arc3_qsa_nvfp4: per 256-element K or V head, 128 packed E2M1 bytes, 16 E4M3 block
scales, one FP32 head scale; 148 bytes per side, 296-byte rows), so the NVFP4 attention readers, the eviction row
size and the KV memory are unchanged. What changes:
  store   y = H x for every K and V head (H: the 256-point Sylvester-Hadamard matrix / 16, orthonormal and symmetric,
          computed as (H16 X H16) / 16 on the 16 x 16 view of x, FP32 IEEE dots, a fixed order per vector, so the
          result does not depend on the batch); head scale max|y| / 2688 (exact division); for each block of 16 the
          E4M3 scale is chosen among the codes c-1, c, c+1, c+2 around the round-to-nearest code c of
          max|block| / (6 * head scale) by the block's squared reconstruction error (strictly smaller wins, in that
          order, as kvfq._blocks); E2M1 codes by cvt.rn.satfinite (as production).
  attend  the query is rotated the same way before the NVFP4 attention (scores q.k are invariant under the joint
          orthonormal rotation), and the attention output is rotated back (H = H^-1): out = H sum p (H v) = sum p v~.
So attention sees exactly kvfq's fake quantization x~ = H q(H x), up to arithmetic order and BF16 rounding of the
rotated query and output.
Two differences from kvfq's replication, both deliberate (kvfq_diag, 2026-10-02): the head scale uses an exact
division (PyTorch divides by a Python scalar through its reciprocal, 1 ulp off in about a quarter of heads), and the
E4M3 block-scale code is rounded by cvt.rn.satfinite.e4m3x2.f32, exact round-to-nearest-even from FP32 (production's
Triton .to(float8e4nv) rounds values just below a tie upward). Both flip a code only at exact ties.
Requires ARC3_QSA_KV_FORMAT=nvfp4 (refuses otherwise). Applies to every QSA owner, the MTP drafter's included.

Per-role formats (2026-10-02 03:10; g7r1 showed speculative acceptance 0.05 below stored NVFP4's, and the KV study never
measured the drafter's layer): ARC3_S2_KVROT_TARGET and ARC3_S2_KVROT_DRAFT, each one of
  rot    h_nvfp4m as above (the default for both: unchanged behaviour);
  mse    nvfp4m: the block-MSE scale choice without the rotation (same store kernel, ROTATE off; no query/output rotation);
  plain  production NVFP4: the original do_kv_cache_update (arc3_qsa_nvfp4.store_nvfp4) and no rotation, i.e. G2.
The drafter's QSA owner is the one whose layer name has an 'mtp.' component (mtp.layers.48.self_attn.attn); the role is
resolved once per owner and recorded in STATS (draft_layers, target_layers).

Codebook formats (2026-10-02 07:20; kv_screen2's winner h_lutm+c, kvlut_kernels.py): lut, lutc, lutu, lutcu. Same
148-byte layout; 4-bit codes stand for the levels of an odd quintic fitted to the 16-level Lloyd-Max codebook (decoded
without a table); rotated, block-MSE E4M3 scales; 'c' centers K and V by the per-layer, per-KV-head means of
ARC3_S2_KVROT_MEANS ({layer name: {'k', 'v'}: FP32 [heads, 256]}; K zero on the rotary dims; the K mean shifts every
logit of a query equally, so it is never added back), 'u' rescales the head scale (unbiased). The reader is
kvlut_kernels.attention between the same query and output rotations as 'rot'; it adds the rotated V mean, which the
output rotation turns into the V mean. The production forward_qsa is bypassed for these owners (its quantized path is
only the reader call)."""
import os
import re

import torch
import triton
import triton.language as tl

PINS = {'vllm.models.qwen3_8_flash_next.nvidia.qsa': '964d81e8e383c9a7e5a785cea8e8b8998bff6059b11c5622993a26c7486daa61'}
STATS = dict(store_calls=0, store_rows=0, query_rotations=0, output_rotations=0, errors=0)
FORMATS = ('rot', 'mse', 'plain', 'lut', 'lutc', 'lutu', 'lutcu')
_MEANS_PATH = os.environ.get('ARC3_S2_KVROT_MEANS', '')
_MEANS = {}
_DRAFT_NAME = re.compile(r'(^|\.)mtp\.')


@triton.jit
def _h16():
    """The 16 x 16 Sylvester-Hadamard matrix (entries +-1): H[i][j] = (-1)^popcount(i & j)."""
    i = tl.arange(0, 16)[:, None]
    j = tl.arange(0, 16)[None, :]
    b = i & j
    parity = (b ^ (b >> 1) ^ (b >> 2) ^ (b >> 3)) & 1
    return tl.where(parity == 0, 1.0, -1.0)


@triton.jit
def _rotate(X):
    """X: FP32 [16, 16] holding a 256-vector row-major (x[16a+b] = X[a, b]). Returns H256 x / 16 in the same layout:
    H256 = H16 (x) H16 (Sylvester), so H256 x = vec(H16 X H16)."""
    H = _h16()
    T = tl.dot(H, X, input_precision="ieee")
    return tl.dot(T, H, input_precision="ieee") * 0.0625


@triton.jit
def _e4m3_code(x):
    """Round-to-nearest-even E4M3FN code of x (FP32, 0 <= x <= 448), exact from FP32 (sm_89+)."""
    zero = tl.zeros_like(x)
    return tl.inline_asm_elementwise(
        asm='cvt.rn.satfinite.e4m3x2.f32 $0, $2, $1;',
        constraints='=h,f,f', args=[x, zero], dtype=tl.uint16, is_pure=True, pack=1).to(tl.uint8)


@triton.jit
def _pack_e2m1_pair(first, second):
    # As arc3_qsa_nvfp4: the first value occupies the low nibble, the second the high nibble.
    return tl.inline_asm_elementwise(
        asm='''{ .reg .b8 packed;
            cvt.rn.satfinite.e2m1x2.f32 packed, $2, $1;
            mov.b16 $0, {packed, 0}; }''',
        constraints='=h,f,f', args=[first, second], dtype=tl.uint16,
        is_pure=True, pack=1).to(tl.uint8)


@triton.jit
def _decode_e2m1(nibble):
    # As arc3_qsa_nvfp4: exact FP32 encodings of the 16 E2M1 codes.
    bits = nibble.to(tl.uint32)
    exponent = (bits >> 1) & 3
    mantissa = bits & 1
    normal = ((exponent + 126) << 23) | (mantissa << 22)
    subnormal = tl.where(mantissa != 0, 126 << 23, 0).to(tl.uint32)
    value = tl.where(exponent == 0, subnormal, normal) | ((bits & 8) << 28)
    return value.to(tl.float32, bitcast=True)


@triton.jit
def _store_kernel(key, value, cache, slots,
                  stride_kr, stride_kh, stride_vr, stride_vh, stride_cb, stride_ch, stride_ct,
                  num_blocks, PAGE: tl.constexpr, ROTATE: tl.constexpr, NCAND: tl.constexpr):
    row = tl.program_id(0)
    head = tl.program_id(1)
    side = tl.program_id(2)
    slot = tl.load(slots + row)
    if slot >= 0 and slot < num_blocks * PAGE:
        idx = tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]
        if side == 0:
            p = key + row * stride_kr + head * stride_kh
        else:
            p = value + row * stride_vr + head * stride_vh
        Y = tl.load(p + idx).to(tl.float32)
        if ROTATE:
            Y = _rotate(Y)
        y0, y1 = tl.split(tl.reshape(Y, [16, 8, 2]))       # even and odd elements of each block of 16, [16, 8]
        maxima = tl.max(tl.maximum(tl.abs(y0), tl.abs(y1)), axis=1)
        maximum = tl.max(maxima, axis=0)
        head_scale = tl.where(maximum == 0., 1., tl.maximum(tl.div_rn(maximum, 2688.), 1e-30))
        code = _e4m3_code(tl.minimum(tl.div_rn(maxima, 6. * head_scale), 448.))
        best_code = code
        best_err = tl.full([16], float('inf'), tl.float32)
        best_packed = tl.zeros([16, 8], dtype=tl.uint8)
        for i in tl.static_range(NCAND):                    # NCAND 4: codes c-1, c, c+1, c+2; NCAND 1: c only
            if NCAND > 1:
                c = tl.minimum(tl.maximum(code.to(tl.int32) + (i - 1), 0), 126).to(tl.uint8)
            else:
                c = code
            dq = head_scale * c.to(tl.float8e4nv, bitcast=True).to(tl.float32)
            safe = tl.where(dq > 0., dq, 1.)
            n0 = tl.where(dq[:, None] > 0., tl.div_rn(y0, safe[:, None]), 0.)
            n1 = tl.where(dq[:, None] > 0., tl.div_rn(y1, safe[:, None]), 0.)
            packed = _pack_e2m1_pair(n0, n1)
            r0 = _decode_e2m1(packed & 15) * dq[:, None]
            r1 = _decode_e2m1(packed >> 4) * dq[:, None]
            err = tl.sum((r0 - y0) * (r0 - y0) + (r1 - y1) * (r1 - y1), axis=1)
            take = err < best_err
            best_err = tl.where(take, err, best_err)
            best_code = tl.where(take, c, best_code)
            best_packed = tl.where(take[:, None], packed, best_packed)
        address = cache + (slot // PAGE).to(tl.int64) * stride_cb + head * stride_ch + (slot % PAGE) * stride_ct + side * 148
        pair = tl.arange(0, 16)[:, None] * 8 + tl.arange(0, 8)[None, :]
        tl.store(address + pair, best_packed)
        tl.store(address + 128 + tl.arange(0, 16), best_code)
        tl.store((address + 144).to(tl.pointer_type(tl.float32)), head_scale)


def store(key, value, cache, slots, rotate=True, mse=True):
    """Drop-in for arc3_qsa_nvfp4.store_nvfp4 (same checks and layout); rotate/mse False only for tests."""
    if key.dtype != torch.bfloat16 or value.dtype != key.dtype or key.ndim != 3 or key.shape != value.shape or key.shape[-1] != 256:
        raise ValueError('kvrot: QSA writes require 256-dimensional BF16 K/V')
    if (slots.ndim != 1 or slots.dtype not in (torch.int32, torch.int64) or slots.stride(0) != 1
            or len(slots) > len(key) or key.stride(-1) != 1 or value.stride(-1) != 1):
        raise ValueError('kvrot: unsupported QSA input or slot layout')
    if not key.is_cuda or key.device != value.device or key.device != cache.device or key.device != slots.device:
        raise ValueError('kvrot: QSA cache tensors must share a CUDA device')
    from arc3_qsa_nvfp4 import split_cache
    split_cache(cache)                              # dtype, 296-byte rows, four-byte alignment of the head scales
    if cache.shape[1] != key.shape[1]:
        raise ValueError('kvrot: QSA cache head count mismatch')
    if not len(slots):
        return
    _store_kernel[(len(slots), key.shape[1], 2)](key, value, cache, slots,
        key.stride(0), key.stride(1), value.stride(0), value.stride(1),
        cache.stride(0), cache.stride(1), cache.stride(2), cache.shape[0],
        PAGE=cache.shape[2], ROTATE=rotate, NCAND=4 if mse else 1, num_warps=4)


@triton.jit
def _rotate_rows_kernel(src, dst, stride_sr, stride_sh, stride_dr, stride_dh):
    row = tl.program_id(0)
    head = tl.program_id(1)
    idx = tl.arange(0, 16)[:, None] * 16 + tl.arange(0, 16)[None, :]
    X = tl.load(src + row * stride_sr + head * stride_sh + idx).to(tl.float32)
    Y = _rotate(X)
    tl.store(dst + row * stride_dr + head * stride_dh + idx, Y.to(dst.dtype.element_ty))


def rotate_rows(x, out=None):
    """H x / 16 for every [row, head] 256-vector of a [N, H, 256] tensor (BF16 or FP32); in place when out is x."""
    if x.ndim != 3 or x.shape[-1] != 256 or x.stride(-1) != 1:
        raise ValueError('kvrot: rotate_rows needs [N, H, 256] with a unit last stride')
    if out is None:
        out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    if out.shape != x.shape or out.stride(-1) != 1:
        raise ValueError('kvrot: rotate_rows output shape or stride')
    if x.shape[0]:
        _rotate_rows_kernel[(x.shape[0], x.shape[1])](x, out, x.stride(0), x.stride(1), out.stride(0), out.stride(1), num_warps=4)
    return out


def role_formats():
    """(target, draft) formats from ARC3_S2_KVROT_TARGET / ARC3_S2_KVROT_DRAFT (default rot, rot)."""
    target = os.environ.get('ARC3_S2_KVROT_TARGET', 'rot')
    draft = os.environ.get('ARC3_S2_KVROT_DRAFT', 'rot')
    if target not in FORMATS or draft not in FORMATS:
        raise RuntimeError('kvrot: ARC3_S2_KVROT_TARGET/DRAFT must be one of %s, got %r/%r' % (FORMATS, target, draft))
    return target, draft


def layer_means(name):
    """(K mean, V mean, rotated V mean) FP32 [heads, 256] on the GPU for a QSA owner, from ARC3_S2_KVROT_MEANS."""
    if not _MEANS:
        if not _MEANS_PATH or not os.path.exists(_MEANS_PATH):
            STATS['errors'] += 1
            raise RuntimeError('kvrot: a centered codebook format needs ARC3_S2_KVROT_MEANS (got %r)' % _MEANS_PATH)
        _MEANS.update(torch.load(_MEANS_PATH))
    m = _MEANS.get(name)
    if m is None:
        STATS['errors'] += 1
        raise RuntimeError('kvrot: no means for %s in %s' % (name, _MEANS_PATH))
    if not isinstance(m, tuple):
        mk, mv = (m[x].to(device='cuda', dtype=torch.float32).contiguous() for x in ('k', 'v'))
        mk, mv = _rank_heads(mk), _rank_heads(mv)               # L4 port (2026-10-30): this rank's KV head(s) at TP > 1
        m = _MEANS[name] = (mk, mv, rotate_rows(mv[None])[0].contiguous())
    return m


def _rank_heads(t):
    """L4 port (2026-10-30): the per-KV-head means of this rank's head(s). TP <= heads: a contiguous slice; TP > heads:
    vLLM replicates each KV head on tp // heads ranks (kv shard = rank // replicas)."""
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
    tp = get_tensor_model_parallel_world_size()
    if tp == 1:
        return t
    heads, r = t.shape[0], get_tensor_model_parallel_rank()
    if heads >= tp:
        local = heads // tp
        return t[r * local:(r + 1) * local].contiguous()
    h = r // (tp // heads)
    return t[h:h + 1].contiguous()


def patch_qsa(mod):
    if mod._ARC3_QSA_KV_FORMAT != 'nvfp4':
        STATS['errors'] += 1
        raise RuntimeError('kvrot needs the NVFP4 main-QSA cache (ARC3_QSA_KV_FORMAT=nvfp4), got %r' % mod._ARC3_QSA_KV_FORMAT)
    target_format, draft_format = role_formats()
    STATS.update(target_format=target_format, draft_format=draft_format, target_layers=0, draft_layers=[],
                 draft_store_calls=0, draft_query_rotations=0, lut_attention_calls=0, means_file=_MEANS_PATH or None)
    cls = mod.Qwen3_8FlashNextQSAFlashAttentionImpl
    original_update = cls.do_kv_cache_update
    original_forward = cls.forward_qsa

    def role(layer):
        """(format, is_draft, means or None) of a QSA owner, resolved on its first call."""
        r = getattr(layer, '_arc3_kvrot_role', None)
        if r is None:
            name = getattr(layer, 'layer_name', '')
            draft = bool(_DRAFT_NAME.search(name))
            fmt = draft_format if draft else target_format
            r = (fmt, draft, layer_means(name) if fmt.startswith('lutc') else None)
            layer._arc3_kvrot_role = r
            if draft:
                STATS['draft_layers'].append(name)
            else:
                STATS['target_layers'] += 1
        return r

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        fmt, draft, mu = role(layer)
        STATS['store_calls'] += 1
        STATS['store_rows'] += int(slot_mapping.shape[0])
        if draft:
            STATS['draft_store_calls'] += 1
        if fmt == 'plain':
            return original_update(self, layer, key, value, kv_cache, slot_mapping)
        if fmt.startswith('lut'):
            from . import kvlut_kernels
            mk, mv = mu[:2] if mu is not None else (None, None)
            return kvlut_kernels.store(key, value, kv_cache, slot_mapping, mk, mv, unbias=fmt.endswith('u'))
        return store(key, value, kv_cache, slot_mapping, rotate=fmt == 'rot', mse=True)

    def forward_qsa(self, layer, query, key, value, kv_cache, attn_metadata, output, token_to_req,
                    output_scale=None, output_block_scale=None):
        fmt, draft, mu = role(layer)
        if fmt.startswith('lut'):
            # The production forward_qsa's quantized path with the codebook reader, between the 'rot' rotations.
            from . import kvlut_kernels
            n = attn_metadata.num_actual_tokens
            output.zero_()
            if n:
                kvlut_kernels.attention(rotate_rows(query[:n]), kv_cache, layer.topk_indices_buffer[:n],
                                        attn_metadata.block_table, token_to_req[:n], output[:n],
                                        vmean=mu[2] if mu is not None else None)
                rotate_rows(output[:n], out=output[:n])
                STATS['lut_attention_calls'] += 1
                STATS['query_rotations'] += 1
                STATS['output_rotations'] += 1
                if draft:
                    STATS['draft_query_rotations'] += 1
            return output
        n = attn_metadata.num_actual_tokens if fmt == 'rot' else 0
        if n:
            query = rotate_rows(query[:n])
            STATS['query_rotations'] += 1
            if draft:
                STATS['draft_query_rotations'] += 1
        result = original_forward(self, layer, query, key, value, kv_cache, attn_metadata, output, token_to_req,
                                  output_scale, output_block_scale)
        if n:
            rotate_rows(output[:n], out=output[:n])
            STATS['output_rotations'] += 1
        return result

    cls.do_kv_cache_update = do_kv_cache_update
    cls.forward_qsa = forward_qsa


PATCHERS = {'vllm.models.qwen3_8_flash_next.nvidia.qsa': patch_qsa}
