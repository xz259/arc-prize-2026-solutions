"""stage2_v1 'kvwin' part (2026-10-15, precision work; IE's KV recent-window test, design agreed 02:4x UTC): emulates a KIVI-style
residual window on a BF16-KV launch (ARC3_STAGE_KV_FORMAT=bf16), target QSA owners only (the MTP drafter's owner is untouched).

  write   every new token's K / V is stored in the RING format (ARC3_S2_KVWIN_RING: 'bf16' = unchanged, 'fp8' = kvfq's FP8 E4M3 per
          token and head, or any kvfq candidate) - the tokens inside the window;
  age-out when a request's token at position p is written, its row at position p - N (N = ARC3_S2_KVWIN_N) is fake-quantized IN PLACE
          with the AGED format (ARC3_S2_KVWIN_AGED, a kvfq candidate; 'h_lutpm+c' = the stored lutc format: Hadamard rotation,
          the codebook levels, block-MSE E4M3 scales, centring by the kvlut means of ARC3_S2_KVWIN_MEANS) from the values stored in
          the cache, so an FP8 ring is quantized twice (FP8, then lutc), as a production FP8 ring would be.
          lutc's scales are per 16 elements along each token's head vector and its rotation / centring per vector, so quantizing
          a row late equals quantizing it at write time (no cross-row grouping): attention sees each token in the ring format while
          it is among the last N of its request and in the aged format afterwards, which is what a ring plus age-out quantization
          does in production. Each position ages exactly once (the step that writes position p + N).
  N = 0   no window: every token is stored in the AGED format at write time (the emulated-lutc reference arm, or a per-side
          diagnostic such as 'k=fp8,v=h_lutpm+c').
The age-out runs at the start of forward_qsa (after this step's cache writes, before attention) with data-dependent indexing, so it
needs eager attention (piecewise graphs, i.e. prefill / teacher-forced scoring); under CUDA-graph capture it is skipped and counted
(STATS['capture_skips']); arms must check that counter is the warm-up's only. Measurement only."""
import os

import torch

from . import kvfq

N = int(os.environ.get('ARC3_S2_KVWIN_N', '0'))
RING = os.environ.get('ARC3_S2_KVWIN_RING', 'bf16')
AGED = os.environ.get('ARC3_S2_KVWIN_AGED', '')
MEANS_PATH = os.environ.get('ARC3_S2_KVWIN_MEANS', '')
assert AGED, 'kvwin: ARC3_S2_KVWIN_AGED (a kvfq candidate) is required'
assert N >= 0
PINS = dict(kvfq.PINS)
STATS = dict(n=N, ring=RING, aged=AGED, writes=0, aged_rows=0, age_steps=0, capture_skips=0, draft_skips=0, errors=0)
_MEANS = {}


def _sides(c):
    s = kvfq.parse(c)
    assert s is not None, ('kvwin: not a kvfq candidate', c)
    return s


RING_SIDES = _sides(RING) if RING != 'bf16' else None
AGED_SIDES = _sides(AGED)


def _means(name, device):
    m = _MEANS.get((name, device))
    if m is None:
        if not MEANS_PATH:
            raise RuntimeError('kvwin: a +c candidate needs ARC3_S2_KVWIN_MEANS (the kvlut means file)')
        if not _MEANS.get('_loaded'):
            _MEANS['_file'] = torch.load(MEANS_PATH)
            _MEANS['_loaded'] = True
        d = _MEANS['_file'].get(name)
        if d is None:
            raise RuntimeError('kvwin: no means for %s in %s' % (name, MEANS_PATH))
        m = _MEANS[(name, device)] = (d['k'].float().to(device), d['v'].float().to(device))
    return m


def _fq(key, value, sides, name):
    mk = mv = None
    if any('c' in s.split('+')[1:] for s in sides):
        mk, mv = _means(name, key.device)
    return kvfq.fake_quant(key, sides[0], mk), kvfq.fake_quant(value, sides[1], mv)


def is_target(layer):
    return '.mtp.' not in ('.' + layer.layer_name + '.') and not layer.layer_name.startswith('mtp.')


def age_rows(kv_cache, positions_new, req_new, block_table, n_window, sides, name, head=256):
    """In-place age-out: for every new token (position p, request r), fake-quantize the cache row of position p - n_window of request r.
    kv_cache: [num_blocks, kv_heads, block_size, 2 * head] BF16 (FlashAttentionBackend's layout, K | V on the last axis; fixed 04:2x
    10-15: the first build assumed [num_blocks, block_size, kv_heads, 2 * head]). Returns the number of aged rows."""
    aged = positions_new - n_window
    ok = aged >= 0
    if not bool(ok.any()):
        return 0
    a = aged[ok].long(); r = req_new[ok].long()
    B = kv_cache.shape[2]
    col = a // B
    inb = col < block_table.shape[1]                                     # dummy / profiling batches: ignore what is not addressable
    a, r, col = a[inb], r[inb], col[inb]
    blk = block_table[r, col].long()
    keep = (blk >= 0) & (blk < kv_cache.shape[0])
    blk, off = blk[keep], (a % B)[keep]
    if blk.numel() == 0:
        return 0
    rows = kv_cache[blk, :, off]                                         # [n, kv_heads, 2 * head]
    k, v = _fq(rows[..., :head].contiguous(), rows[..., head:].contiguous(), sides, name)
    kv_cache[blk, :, off] = torch.cat([k, v], -1)
    return int(blk.numel())


def patch_qsa(mod):
    cls = mod.Qwen3_8FlashNextQSAFlashAttentionImpl
    original = cls.do_kv_cache_update
    original_forward = cls.forward_qsa

    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):
        if mod._ARC3_QSA_KV_FORMAT != 'bf16':
            STATS['errors'] += 1
            raise RuntimeError('kvwin needs a BF16 main-QSA cache (ARC3_STAGE_KV_FORMAT=bf16)')
        if is_target(layer):
            sides = AGED_SIDES if N == 0 else RING_SIDES
            if sides is not None:
                key, value = _fq(key, value, sides, layer.layer_name)
            STATS['writes'] += 1
        else:
            STATS['draft_skips'] += 1
        return original(self, layer, key, value, kv_cache, slot_mapping)

    def forward_qsa(self, layer, query, key, value, kv_cache, attn_metadata, output, token_to_req,
                    output_scale=None, output_block_scale=None):
        n = attn_metadata.num_actual_tokens
        if N > 0 and n and is_target(layer):
            if torch.cuda.is_current_stream_capturing():
                STATS['capture_skips'] += 1
            else:
                assert kv_cache.dim() == 4 and kv_cache.shape[-1] == 2 * self.head_size and \
                    kv_cache.shape[1] == self.num_kv_heads, ('kvwin: unexpected BF16 QSA cache layout', tuple(kv_cache.shape))
                req = token_to_req[:n].long()
                qsl = attn_metadata.query_start_loc.long(); sl = attn_metadata.seq_lens.long()
                tok = torch.arange(n, device=req.device)
                pos = sl[req] - (qsl[req + 1] - tok)                     # absolute position of each new token in its request
                STATS['aged_rows'] += age_rows(kv_cache, pos, req, attn_metadata.block_table, N, AGED_SIDES, layer.layer_name,
                                               self.head_size)
                STATS['age_steps'] += 1
        return original_forward(self, layer, query, key, value, kv_cache, attn_metadata, output, token_to_req,
                                output_scale, output_block_scale)

    cls.do_kv_cache_update = do_kv_cache_update
    cls.forward_qsa = forward_qsa


PATCHERS = {'vllm.models.qwen3_8_flash_next.nvidia.qsa': patch_qsa}
