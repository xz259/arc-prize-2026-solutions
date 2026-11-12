"""stage2_v1 'seloverlap' part (milestone 2, 2026-10-02; B4 groundwork, diagnostic only): how much the QSA top-2,048
selections of one request's verify rows overlap, sized for a reader that would process a request's rows together.

Wraps the QSA impl's forward_qsa (chained after kvrot's patch on the same module). On every ARC3_S2_SELOVERLAP_EVERY-th
eager call of a target QSA layer (not during graph capture; the drafter's layer is skipped), it reads the layer's
topk_indices_buffer (logical token ids, -1 padded) and token_to_req, and records for requests with 2-4 rows in the step
(verify rows; prefill requests have more):
  pairs      consecutive rows of one request: |sel(i) & sel(i+1)| / |sel(i)|
  quartets   requests with exactly 4 rows: |union of the 4 selections| / |sel(first row)|
Selections are compared as sets of token ids. Only rows with at least MIN_SEL selected tokens count (contexts past the
top-k budget; at short contexts the 'selection' is the whole context and trivially shared). Host syncs only on sampled
calls. Decode steps replay CUDA graphs, so a diagnostic launch keeps every capture size below the decode batch (eager).
(16:10: the first run captured its 4-row decode steps, CAPTURE=4, and sampled only short-context rows: degenerate.)"""
import os

import torch

PINS = {'vllm.models.qwen3_8_flash_next.nvidia.qsa': '964d81e8e383c9a7e5a785cea8e8b8998bff6059b11c5622993a26c7486daa61'}
EVERY = int(os.environ.get('ARC3_S2_SELOVERLAP_EVERY', '5'))
MIN_SEL = int(os.environ.get('ARC3_S2_SELOVERLAP_MIN', '1024'))
STATS = dict(every=EVERY, calls=0, sampled=0, pairs=0, overlap_sum=0.0, overlap_min=1.0, quartets=0, union_sum=0.0,
             union_max=0.0, selected_sum=0, errors=0)


def measure(idx, req):
    """idx [n, W] int32 (-1 padded), req [n] int32 (rows of a request consecutive)."""
    n, W = idx.shape
    reql = req.long()
    counts = torch.bincount(reql.clamp(min=0))
    rows = counts[reql.clamp(min=0)]
    verify = (rows >= 2) & (rows <= 4) & (reql >= 0)
    same = (reql[1:] == reql[:-1]) & verify[1:] & verify[:-1]
    if bool(same.any()):
        A, B = idx[:-1][same], idx[1:][same]
        Bs = B.sort(dim=1).values.contiguous()
        pos = torch.searchsorted(Bs, A.contiguous()).clamp(max=W - 1)
        found = (Bs.gather(1, pos) == A) & (A >= 0)
        valid = (A >= 0).sum(1)
        ov = found.sum(1).float() / valid.clamp(min=1).float()
        ok = (valid >= MIN_SEL) & ((B >= 0).sum(1) >= MIN_SEL)
        if bool(ok.any()):
            ov = ov[ok]
            STATS['pairs'] += int(ov.numel())
            STATS['overlap_sum'] += float(ov.sum())
            STATS['overlap_min'] = min(STATS['overlap_min'], float(ov.min()))
            STATS['selected_sum'] += int(valid[ok].sum())
    starts = torch.nonzero((rows == 4) & torch.cat([torch.ones(1, dtype=torch.bool, device=idx.device),
                                                     reql[1:] != reql[:-1]]), as_tuple=True)[0]
    starts = starts[starts + 3 < n]
    if starts.numel():
        quad = torch.stack([idx[starts + k] for k in range(4)], 1).reshape(starts.numel(), 4 * W)
        s = quad.sort(dim=1).values
        validq = s >= 0
        new = torch.ones_like(validq)
        new[:, 1:] = s[:, 1:] != s[:, :-1]
        union = (new & validq).sum(1).float()
        base = (idx[starts] >= 0).sum(1).clamp(min=1).float()
        keep = base >= MIN_SEL
        union, base = union[keep], base[keep]
        if not union.numel():
            return
        ratio = union / base
        STATS['quartets'] += int(ratio.numel())
        STATS['union_sum'] += float(ratio.sum())
        STATS['union_max'] = max(STATS['union_max'], float(ratio.max()))


def patch_qsa(mod):
    cls = mod.Qwen3_8FlashNextQSAFlashAttentionImpl
    original = cls.forward_qsa

    def forward_qsa(self, layer, query, key, value, kv_cache, attn_metadata, output, token_to_req, *args, **kwargs):
        STATS['calls'] += 1
        name = getattr(layer, 'layer_name', '')
        if (STATS['calls'] % EVERY == 0 and '.mtp.' not in '.' + name and not torch.cuda.is_current_stream_capturing()):
            try:
                n = int(attn_metadata.num_actual_tokens)
                buf = getattr(layer, 'topk_indices_buffer', None)
                if n > 1 and buf is not None:
                    measure(buf[:n], token_to_req[:n])
                    STATS['sampled'] += 1
            except Exception:                          # noqa: BLE001 (diagnostic: never break the step)
                STATS['errors'] += 1
        return original(self, layer, query, key, value, kv_cache, attn_metadata, output, token_to_req, *args, **kwargs)

    cls.forward_qsa = forward_qsa


PATCHERS = {'vllm.models.qwen3_8_flash_next.nvidia.qsa': patch_qsa}
