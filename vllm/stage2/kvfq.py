"""stage2_v1 'kvfq' part (milestone 2, 2026-10-01; diagnostic only): fake-quantized main-QSA K/V for the 4-bit KV
format study (user 14:10: try INT4 after a Hadamard rotation (QuaRot) and other variants against our NVFP4).

On a BF16-KV launch (ARC3_STAGE_KV_FORMAT=bf16), the QSA impl's do_kv_cache_update quantizes and dequantizes K and V
with the selected candidate before the BF16 cache write. forward_qsa reads K/V only from the cache (it deletes its
key/value arguments), so attention sees exactly what a stored format would give it. The cache stays BF16, so one launch
scores every candidate with kl_panel against the same reference.

Candidate: the first word of the file named by ARC3_S2_KVFQ_SWITCH, re-read at most every 0.5 s ('off' or a missing
file: unchanged). Every QSA layer is affected, the drafter's included (prompt logprobs are the target's). Eager or
piecewise steps only: a FULL CUDA graph replays the candidate active at capture ('off' at startup), so score prompt
logprobs (prefill), not decode. Static mode (16:20): ARC3_S2_KVFQ_STATIC=<candidate> fixes the candidate for the launch,
so graph capture records the fake-quantization ops too (pure GPU tensor ops, no host reads or syncs) and decode sees it:
one launch per candidate, for confirming the prefill screen's best candidates with decode_compare.

Candidates (K and V alike, or 'k=<a>,v=<b>' per side; 'bf16' = unchanged):
  nvfp4     our stored format, replicated (arc3_qsa_nvfp4._store_nvfp4_kernel): FP32 head scale max/2688, E4M3 scale
            per 16 (RNE, <= 448), E2M1 RNE saturating at 6. 148 bytes per side. Validation: KL near qK4's 0.0317
  fp8       our FP8 E4M3 per token and head (arc3_qsa_quant._store_fp8_kernel: max/448, RNE). 260 bytes. Validation:
            KL near qK8's 0.0169
  nvfp4m    NVFP4 with each block scale chosen among E4M3 neighbours (one below the RNE scale to two above) by block MSE
  int4s     INT4 symmetric per 16 (levels -7..7), E4M3 block scale under an FP32 head scale max/(7*448). 148 bytes
  int4sm    int4s with the block-MSE scale choice
  int4a     INT4 asymmetric per 16 (0..15, 4-bit zero point, E4M3 scale of the range). 156 bytes
  int4s128  INT4 symmetric per 128 with an FP32 scale. 136 bytes
  int4a128  INT4 asymmetric per 128 with FP32 scale and minimum (QuaRot's KV setting). 144 bytes
  h_<c>     <c> after a 256-point normalized Sylvester-Hadamard rotation of each head: x -> H q(H x) (H = H^T = H^-1).
            Stored rotated, K and Q rotate alike (scores unchanged) and V's attention output is un-rotated, so the
            fake quantization H q(H x) is exactly what attention would see.
Byte counts are per 256-element head side; NVFP4 KV today is 148.

Additions (2026-10-02 04:00; user 03:58: screen the format ideas discussed with the session Fable):
  lut, lutm  the NVFP4 layout with the E2M1 grid replaced by a 16-entry table: Lloyd-Max levels for blocks of 16
            Gaussian values normalized by the block maximum (+-LUT16, no zero code; a kernel's nibble decode becomes a
            table lookup). E4M3 block scale of the block maximum under an FP32 head scale max/448; lutm adds the
            block-MSE scale choice. 148 bytes. Synthetic Gaussian blocks, relative MSE: E2M1 + MSE scales 0.0074,
            INT4 symmetric + MSE scales 0.0067, lutm 0.0051 (int4a, 156 bytes: 0.0051).
  lutp, lutpm  (07:20) as lut, lutm with the stored codebook format's levels (kvlut_kernels: an odd quintic in the code,
            within 5.7e-4 of LUT16, decoded without a table); the reference fake for that format's tests.
  +c        centering (exact in attention): each side quantizes x - mu and adds mu back, mu the per-layer, per-KV-head
            mean from the 'stats' pass (K: zero on the 64 rotary dims). For K the logit shift q.mu is constant over keys, so a stored format would
            never add mu back; for V a stored format adds mu to the output (weights sum to 1).
  +u        unbiased scale: the reconstruction of each (centered) head vector is multiplied by |x|^2 / <x^, x>, which a
            stored format folds into the FP32 head scale (MSE-chosen scales shrink the reconstruction).
  +a, +a2, +a3  attention-aware linear transform (exact in attention): each side quantizes T x (then the base's own
            rotation) and reconstructs T^-1 (...), T per layer and KV head from ARC3_S2_KVFQ_TRANSFORMS (default
            /tmp/kvfq_transforms.pt, written by a2_transforms.py from the 'stats' dump; variants a, a2, a3 = ridge
            1e-3, 1e-2, 1e-1 on RoPE-phase-symmetrized statistics): for K, T^T T = S^-1 # M
            (S the key covariance, M the query second moment; a stored format multiplies queries by T^-T); for V the
            same with M replaced by the output projection's Gram matrix (the stored format applies T^-1 to the output).
  Modifiers follow the base, e.g. h_lutm+c+u, k=h_lutm+c+a,v=h_int4a. Order: center, transform, rotate, quantize,
  unbias (in the quantized domain), un-rotate, inverse transform, un-center.
Statistics (2026-10-02 04:00): switch word 'stats' accumulates, on eager calls, per QSA owner and KV head the count, sums
and second moments of K and V as stored (post-RoPE) and the second moment of the queries of that KV head's 12 query
heads (post-RoPE; forward_qsa); no quantization meanwhile. Switch word 'dump' saves them once to ARC3_S2_KVFQ_STATS
(default /tmp/kvfq_stats.pt) for the offline attention-aware-rotation analysis, with up to 16,384 random stored K/V
rows per owner (512 per call) for the per-layer error proxy (kv_proxy.py). '+c' uses the means accumulated in this
process, or loads that file when the process has none (static mode)."""
import math
import os
import re
import time

import torch

PINS = {'vllm.models.qwen3_8_flash_next.nvidia.qsa': 'a75b7b0fb72761de2fb246ee28dd685054fe6f85112387c7c91efa3c65db52a6'}
STATS = dict(calls=0, tokens=0, candidates={}, errors=0, stats_calls=0, stats_query_calls=0, dumps=0)
_SWITCH = os.environ.get('ARC3_S2_KVFQ_SWITCH', '')
_STATIC = os.environ.get('ARC3_S2_KVFQ_STATIC', '')   # a fixed candidate for the whole launch: captured into CUDA graphs
_STATS_PATH = os.environ.get('ARC3_S2_KVFQ_STATS', '/tmp/kvfq_stats.pt')
_STATE = dict(t=0.0, cand='off', dumped=False)
_HAD = {}
_ACC = {}                                              # layer name -> accumulators (float64, on the layer's device)
_MEANS = {}                                            # (layer name, device) -> (mu_k [H, 256], mu_v [H, 256]) float32
_TRANSFORMS_PATH = os.environ.get('ARC3_S2_KVFQ_TRANSFORMS', '/tmp/kvfq_transforms.pt')
_TF = {}                                               # (variant, layer name, device) -> (tk, tki, tv, tvi) [H, 256, 256] float32
ROTARY = 64                                            # partial RoPE: dims [0, 64) rotary (neox pairs i, i + 32)
_SAMPLE_ROWS, _SAMPLE_CAP = 512, 16384                 # stats: random K/V rows kept per call and per owner (CPU, BF16)
BASES = ('bf16', 'fp8', 'nvfp4', 'nvfp4m', 'int4s', 'int4sm', 'int4a', 'int4s128', 'int4a128', 'lut', 'lutm', 'lutp',
         'lutpm', 'int8s', 'int8sm', 'int8t')   # 10-15: 8-bit recent-window ring candidates (kvwin)
MODIFIERS = ('c', 'u', 'a', 'a2', 'a3')
# 2026-10-03 22:54 (rebase design option 1, M2_DECISIONS 22:38): K-side modifier '+r<N>' = N re-rotation events. After
# the first fake quantization, each event turns the dequantized key's 64 rotary dims by ROT_DELTA positions (neox pairs,
# theta 1e7; M-RoPE axes shift alike) and quantizes it again in the same format; at the end the key turns back by
# N x ROT_DELTA exactly (no quantization). Scores are those of a key moved N times by ROT_DELTA, with the query moved
# alike, so the panel's KL against a single quantization ('+r0' = the base) sizes the requantization noise per event.
# K side of a 'k=...,v=...' candidate only (V carries no RoPE), with an 'h_' base, not combined with '+a'.
ROT_DELTA = int(os.environ.get('ARC3_S2_KVFQ_ROT_DELTA', '200000'))
ROPE_THETA = 1.0e7
INV_FREQ64 = [ROPE_THETA ** (-(2.0 * i) / 64.0) for i in range(32)]
LUT16 = (0.05286, 0.15959, 0.26949, 0.38465, 0.50827, 0.64419, 0.79859, 0.9814)   # positive levels; the code is +-these
# lutp: the stored codebook format's levels (kvlut_kernels.LUTP, an odd quintic in the code fitted to LUT16, FP32 as the
# kernels compute it) and its decision boundaries (kvlut_kernels.BOUNDS)
LUTP = (0.053044166415929794, 0.15999113023281097, 0.26968681812286377, 0.3845396935939789, 0.5078223943710327,
        0.6440170407295227, 0.7991613745689392, 0.9811936616897583)
LUTP_BOUNDS = (0.10651765018701553, 0.21483898162841797, 0.3271132707595825, 0.446181058883667, 0.5759197473526001,
               0.721589207649231, 0.8901774883270264)


def candidate():
    if _STATIC:
        return _STATIC
    now = time.monotonic()
    if now - _STATE['t'] > 0.5:
        _STATE['t'] = now
        try:
            with open(_SWITCH) as f:
                words = f.read().split()
            _STATE['cand'] = words[0] if words else 'off'
        except OSError:
            _STATE['cand'] = 'off'
    return _STATE['cand']


def parse(c):
    """'off', 'stats', 'dump' -> None; 'x' -> (x, x); 'k=a,v=b' -> (a, b). Each side: a base, optionally prefixed
    'h_', optionally followed by modifiers '+c', '+u'."""
    if c in ('off', 'stats', 'dump'):
        return None
    if c.startswith('k='):
        k, v = c.split(',')
        sides = (k[2:], v.split('=', 1)[1])
    else:
        sides = (c, c)
    for i, s in enumerate(sides):
        base, *mods = s.split('+')
        cyc = [m for m in mods if re.fullmatch(r'r\d+', m)]
        assert (base[2:] if base.startswith('h_') else base) in BASES and all(m in MODIFIERS or m in cyc for m in mods), c
        assert not cyc or (i == 0 and c.startswith('k=') and base.startswith('h_') and len(cyc) == 1
                           and not any(m.startswith('a') for m in mods)), ('+r<N>: K side of k=,v= with an h_ base only', c)
    return sides


def hadamard(device):
    h = _HAD.get(device)
    if h is None:
        h = torch.ones(1, 1, dtype=torch.float64)
        while h.shape[0] < 256:
            h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
        h = (h / 16.0).to(torch.float32).to(device)
        _HAD[device] = h
    return h


def e2m1_rne(n):
    """Round to the E2M1 grid {0, 0.5, 1, 1.5, 2, 3, 4, 6}, ties to even, saturating (cvt.rn.satfinite.e2m1x2.f32)."""
    a = n.abs()
    q = torch.zeros_like(a)
    q = torch.where(a > 0.25, 0.5, q)
    q = torch.where(a >= 0.75, 1.0, q)
    q = torch.where(a > 1.25, 1.5, q)
    q = torch.where(a >= 1.75, 2.0, q)
    q = torch.where(a > 2.5, 3.0, q)
    q = torch.where(a >= 3.5, 4.0, q)
    q = torch.where(a > 5.0, 6.0, q)
    return torch.copysign(q, n)


def _e4m3(x):
    return x.to(torch.float8_e4m3fn).to(torch.float32)


def _neighbours(bs):
    """E4M3 values around bs: one step below, bs, one and two steps above (by the code's bit pattern)."""
    bits = bs.to(torch.float8_e4m3fn).view(torch.uint8).to(torch.int16)
    out = []
    for d in (-1, 0, 1, 2):
        b = torch.clamp(bits + d, 0, 126)          # 0x7E = 448, the largest finite E4M3FN; sign bit stays 0
        out.append(b.to(torch.uint8).view(torch.float8_e4m3fn).to(torch.float32))
    return out


def _blocks(x, levels, grid, mse_search):
    """Symmetric per-16 quantization with an E4M3 block scale under an FP32 head scale (the NVFP4 layout). levels: the
    grid's largest magnitude (6 for E2M1, 7 for INT4); grid: rounding function on the normalized values."""
    T, H, D = x.shape
    xb = x.view(T, H, D // 16, 16)
    maxima = xb.abs().amax(-1)
    maximum = maxima.amax(-1, keepdim=True)
    hs = torch.where(maximum == 0, torch.ones_like(maximum), torch.clamp(maximum / (levels * 448.0), min=1e-30))
    bs = _e4m3(torch.clamp(maxima / (levels * hs), max=448.0))

    def recon(b):
        dq = hs * b
        safe = torch.where(dq > 0, dq, torch.ones_like(dq))
        n = torch.where(dq[..., None] > 0, xb / safe[..., None], torch.zeros_like(xb))
        return grid(n) * dq[..., None]

    if not mse_search:
        return recon(bs).view(T, H, D)
    best, err = None, None
    for b in _neighbours(bs):
        r = recon(b)
        e = (r - xb).square().sum(-1)
        if best is None:
            best, err = r, e
        else:
            take = e < err
            best = torch.where(take[..., None], r, best)
            err = torch.where(take, e, err)
    return best.view(T, H, D)


def _int4_grid(n):
    return torch.clamp(torch.round(n), -7, 7)


def _int8_grid(n):
    return torch.clamp(torch.round(n), -127, 127)


def _int8t(x):
    s = x.abs().amax(-1, keepdim=True) / 127.0
    s = torch.where(s > 0, s, torch.ones_like(s))
    return torch.clamp(torch.round(x / s), -127, 127) * s


_LUTT = {}


def _lut_grid(n):
    """Nearest of the 16 levels +-LUT16 (n normalized so that the block maximum is 1)."""
    t = _LUTT.get(n.device)
    if t is None:
        lv = torch.tensor(LUT16, dtype=torch.float32, device=n.device)
        t = _LUTT[n.device] = (lv, (lv[1:] + lv[:-1]) / 2)
    lv, bounds = t
    return torch.copysign(lv[torch.bucketize(n.abs(), bounds)], n)


def _lutp_grid(n):
    """Nearest of the 16 levels +-LUTP (the stored codebook format, kvlut_kernels), ties to the lower magnitude."""
    t = _LUTT.get(('p', n.device))
    if t is None:
        t = _LUTT[('p', n.device)] = (torch.tensor(LUTP, dtype=torch.float32, device=n.device),
                                      torch.tensor(LUTP_BOUNDS, dtype=torch.float32, device=n.device))
    lv, bounds = t
    return torch.copysign(lv[torch.bucketize(n.abs(), bounds)], n)


def _int4a16(x):
    T, H, D = x.shape
    xb = x.view(T, H, D // 16, 16)
    mn, mx = xb.amin(-1), xb.amax(-1)
    rng = mx - mn
    top = rng.amax(-1, keepdim=True)
    hs = torch.where(top == 0, torch.ones_like(top), torch.clamp(top / (15 * 448.0), min=1e-30))
    dq = hs * _e4m3(torch.clamp(rng / (15 * hs), max=448.0))
    safe = torch.where(dq > 0, dq, torch.ones_like(dq))
    z = torch.clamp(torch.round(-mn / safe), 0, 15)
    q = torch.clamp(torch.round(xb / safe[..., None]) + z[..., None], 0, 15)
    out = (q - z[..., None]) * dq[..., None]
    return torch.where(dq[..., None] > 0, out, mn[..., None].expand_as(xb)).view(T, H, D)


def _int4g128(x, asym):
    T, H, D = x.shape
    xg = x.view(T, H, D // 128, 128)
    if not asym:
        s = xg.abs().amax(-1, keepdim=True) / 7.0
        s = torch.where(s > 0, s, torch.ones_like(s))
        return (torch.clamp(torch.round(xg / s), -7, 7) * s).view(T, H, D)
    mn, mx = xg.amin(-1, keepdim=True), xg.amax(-1, keepdim=True)
    s = (mx - mn) / 15.0
    safe = torch.where(s > 0, s, torch.ones_like(s))
    q = torch.clamp(torch.round((xg - mn) / safe), 0, 15)
    return torch.where(s > 0, q * s + mn, mn.expand_as(xg)).view(T, H, D)


def _fp8(x):
    amax = x.abs().amax(-1, keepdim=True)
    s = torch.where(amax == 0, torch.ones_like(amax), torch.clamp(amax / 448.0, min=1e-30))
    return _e4m3(torch.clamp(x / s, -448.0, 448.0)) * s


def quantize(x, base):
    """x: float32 [T, H, 256] -> float32 dequantized by the base format."""
    if base == 'bf16':
        return x
    if base == 'fp8':
        return _fp8(x)
    if base in ('nvfp4', 'nvfp4m'):
        return _blocks(x, 6.0, e2m1_rne, base == 'nvfp4m')
    if base in ('int4s', 'int4sm'):
        return _blocks(x, 7.0, _int4_grid, base == 'int4sm')
    if base in ('int8s', 'int8sm'):
        return _blocks(x, 127.0, _int8_grid, base == 'int8sm')
    if base == 'int8t':
        return _int8t(x)
    if base in ('lut', 'lutm'):
        return _blocks(x, 1.0, _lut_grid, base == 'lutm')
    if base in ('lutp', 'lutpm'):
        return _blocks(x, 1.0, _lutp_grid, base == 'lutpm')
    if base == 'int4a':
        return _int4a16(x)
    if base in ('int4s128', 'int4a128'):
        return _int4g128(x, base == 'int4a128')
    raise ValueError(base)


def rope_rotate(x, delta):
    """x: float32 [..., 256] post-RoPE rows; the 64 rotary dims (neox pairs i, i + 32) turned by delta positions (the
    composition rule R(p + d) = R(d) R(p)); the angles are reduced mod 2 pi in float64."""
    ang = torch.tensor([(delta * f) % (2.0 * math.pi) for f in INV_FREQ64], dtype=torch.float64)
    c, s = ang.cos().to(x.device, torch.float32), ang.sin().to(x.device, torch.float32)
    out = x.clone()
    a, b = x[..., :32], x[..., 32:64]
    out[..., :32] = a * c - b * s
    out[..., 32:64] = a * s + b * c
    return out


def _unbias(q, z):
    dot = (q * z).sum(-1, keepdim=True)
    return q * torch.where(dot > 0, z.square().sum(-1, keepdim=True) / torch.where(dot > 0, dot, torch.ones_like(dot)),
                           torch.ones_like(dot))


def fake_quant(t, side, mean=None, tf=None):
    """t: BF16 [T, H, 256]; side: a parsed side ('h_' prefix = rotated; '+c', '+u', '+a' modifiers); mean: [H, 256]
    float32 for '+c'; tf: (T, T^-1), each [H, 256, 256] float32, for '+a'. Returns BF16 of the same shape."""
    base, *mods = side.split('+')
    if base == 'bf16' and not mods:
        return t
    x = t.to(torch.float32).contiguous()
    if 'c' in mods:
        if mean is None:
            raise RuntimeError('kvfq: +c needs the means of a stats pass')
        x = x - mean
    z = x
    if any(m.startswith('a') for m in mods):
        if tf is None:
            raise RuntimeError('kvfq: +a needs the transforms file')
        z = torch.einsum('nhd,hed->nhe', x, tf[0])
    if base.startswith('h_'):
        h = hadamard(z.device)
        z = (z.reshape(-1, z.shape[-1]) @ h).view_as(z)
        q = quantize(z, base[2:])
    else:
        q = quantize(z, base)
    if 'u' in mods:
        q = _unbias(q, z)
    if base.startswith('h_'):
        q = (q.reshape(-1, q.shape[-1]) @ h).view_as(q)
    cycles = next((int(m[1:]) for m in mods if re.fullmatch(r'r\d+', m)), 0)
    for _ in range(cycles):                       # '+r<N>': N re-rotation events, each a second quantization
        q = rope_rotate(q, ROT_DELTA)
        z = (q.reshape(-1, q.shape[-1]) @ h).view_as(q)
        q = quantize(z, base[2:])
        if 'u' in mods:
            q = _unbias(q, z)
        q = (q.reshape(-1, q.shape[-1]) @ h).view_as(q)
    if cycles:
        q = rope_rotate(q, -cycles * ROT_DELTA)   # back to the key's own position, exactly
    if any(m.startswith('a') for m in mods):
        q = torch.einsum('nhd,hed->nhe', q, tf[1])
    if 'c' in mods:
        q = q + mean
    # contiguous: the '+a' einsum returns permuted strides, and the BF16 cache writer reads rows assuming contiguous heads
    # (screen 2026-10-02 06:07: every '+a' candidate stored scrambled heads, KL about 1.7)
    return q.to(torch.bfloat16).contiguous()


def transforms(name, device, variant='a'):
    """(tk, tki, tv, tvi) of a transform variant for a QSA owner, from the transforms file; None when absent."""
    key = (variant, name, device)
    if key not in _TF:
        if not os.path.exists(_TRANSFORMS_PATH):
            return None
        d = torch.load(_TRANSFORMS_PATH).get(variant, {}).get(name)
        _TF[key] = None if d is None else tuple(d[k].to(device=device, dtype=torch.float32) for k in ('tk', 'tki', 'tv', 'tvi'))
    return _TF[key]


def _acc(name, heads, device):
    a = _ACC.get(name)
    if a is None:
        z = lambda *s: torch.zeros(*s, dtype=torch.float64, device=device)
        a = _ACC[name] = dict(n=z(()), nq=z(()), sk=z(heads, 256), sv=z(heads, 256), kk=z(heads, 256, 256),
                              vv=z(heads, 256, 256), qq=z(heads, 256, 256))
    return a


def accumulate_kv(name, key, value):
    """key, value: [N, H, 256] rows actually stored."""
    a = _acc(name, key.shape[1], key.device)
    k, v = key.to(torch.float32), value.to(torch.float32)
    a['n'] += k.shape[0]
    a['sk'] += k.sum(0).double()
    a['sv'] += v.sum(0).double()
    a['kk'] += torch.einsum('nhi,nhj->hij', k, k).double()
    a['vv'] += torch.einsum('nhi,nhj->hij', v, v).double()
    have = int(a['ks'].shape[0]) if 'ks' in a else 0
    if have < _SAMPLE_CAP and k.shape[0]:
        take = torch.randperm(k.shape[0], device=k.device)[:min(_SAMPLE_ROWS, _SAMPLE_CAP - have)]
        ks, vs = key[take].to('cpu', torch.bfloat16), value[take].to('cpu', torch.bfloat16)
        a['ks'] = torch.cat([a['ks'], ks]) if have else ks
        a['vs'] = torch.cat([a['vs'], vs]) if have else vs


def accumulate_q(name, query, kv_heads):
    """query: [N, QH, 256]; pooled over each KV head's group of QH / kv_heads query heads (contiguous groups)."""
    a = _acc(name, kv_heads, query.device)
    n, qh, d = query.shape
    q = query.to(torch.float32).reshape(n, kv_heads, qh // kv_heads, d)
    a['nq'] += n * (qh // kv_heads)
    a['qq'] += torch.einsum('nhgi,nhgj->hij', q, q).double()


def dump_stats(path=None):
    path = path or _STATS_PATH
    torch.save({name: {k: v.cpu() for k, v in a.items()} for name, a in _ACC.items()}, path)
    STATS['dumps'] += 1


def means(name, device):
    m = _MEANS.get((name, device))
    if m is None:
        a = _ACC.get(name)
        if a is None:
            if not _ACC and os.path.exists(_STATS_PATH):
                for n, d in torch.load(_STATS_PATH).items():
                    _ACC[n] = {k: (v if k in ('ks', 'vs') else v.to(device)) for k, v in d.items()}
            a = _ACC.get(name)
        if a is None or float(a['n']) == 0:
            return None
        mk = (a['sk'] / a['n']).float()
        mk[..., :ROTARY] = 0                       # post-RoPE means of rotary dims depend on the position range
        m = _MEANS[(name, device)] = (mk, (a['sv'] / a['n']).float())
    return m


def patch_qsa(mod):
    cls = mod.Qwen3_8FlashNextQSAFlashAttentionImpl
    original = cls.do_kv_cache_update
    original_forward = cls.forward_qsa

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        c = candidate()
        if c == 'stats':
            rows = int(slot_mapping.shape[0])
            accumulate_kv(layer.layer_name, key[:rows], value[:rows])
            STATS['stats_calls'] += 1
        elif c == 'dump':
            if not _STATE['dumped']:
                dump_stats()
                _STATE['dumped'] = True
        else:
            _STATE['dumped'] = False
        sides = parse(c)
        if sides is not None:
            if mod._ARC3_QSA_KV_FORMAT != 'bf16':
                STATS['errors'] += 1
                raise RuntimeError('kvfq needs a BF16 main-QSA cache (ARC3_STAGE_KV_FORMAT=bf16)')
            if key.shape[-1] != 256:
                raise RuntimeError('kvfq expects 256-dimensional heads')
            mk = mv = tfk = tfv = None
            mods_k, mods_v = sides[0].split('+')[1:], sides[1].split('+')[1:]
            if 'c' in mods_k or 'c' in mods_v:
                m = means(layer.layer_name, key.device)
                if m is None:
                    STATS['errors'] += 1
                    raise RuntimeError('kvfq: +c candidate %s before a stats pass for %s' % (c, layer.layer_name))
                mk, mv = m
            for mods, which in ((mods_k, 0), (mods_v, 1)):
                var = next((x for x in mods if x.startswith('a')), None)
                if var is not None:
                    tf = transforms(layer.layer_name, key.device, var)
                    if tf is None:
                        STATS['errors'] += 1
                        raise RuntimeError('kvfq: +%s candidate %s without transforms for %s' % (var, c, layer.layer_name))
                    if which == 0:
                        tfk = (tf[0], tf[1])
                    else:
                        tfv = (tf[2], tf[3])
            key, value = fake_quant(key, sides[0], mk, tfk), fake_quant(value, sides[1], mv, tfv)
            STATS['calls'] += 1
            STATS['tokens'] += int(key.shape[0])
            STATS['candidates'][c] = STATS['candidates'].get(c, 0) + 1
        return original(self, layer, key, value, kv_cache, slot_mapping)

    def forward_qsa(self, layer, query, key, value, kv_cache, attn_metadata, output, token_to_req,
                    output_scale=None, output_block_scale=None):
        if candidate() == 'stats':
            n = attn_metadata.num_actual_tokens
            if n:
                accumulate_q(layer.layer_name, query[:n], getattr(self, 'num_kv_heads', None) or layer.num_kv_heads)
                STATS['stats_query_calls'] += 1
        return original_forward(self, layer, query, key, value, kv_cache, attn_metadata, output, token_to_req,
                                output_scale, output_block_scale)

    cls.do_kv_cache_update = do_kv_cache_update
    cls.forward_qsa = forward_qsa


PATCHERS = {'vllm.models.qwen3_8_flash_next.nvidia.qsa': patch_qsa}
