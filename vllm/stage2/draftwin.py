"""stage2_v1 'draftwin' part (diagnostic; milestone 2 probe, 2026-10-03, user-approved order 02:39): the MTP drafter's QSA
selection limited to its last W tokens, to measure how much acceptance (tau) depends on the drafter seeing distant
context. Motivation: the drafter's QSA layer is the 13th owner of every KV block (1/13 of the pool at full context);
if tau holds at a few thousand tokens, a separate drafter cache holding only a W-token window would shrink every block
by 1/13 (about 55 -> 59 lanes at 36,748, +1-2 at 123,660). This part changes no memory; it only restricts the
selection, so the probe measures tau alone. Draft numerics only: the target verifies every draft (rejection
sampling), so outputs keep the target's distribution.

Mechanism. A QSA indexer scores compressed 4-token groups (ops.qsa.qsa_mqa_paged -> logits [rows, groups]) and keeps
the top token_topk / 4 = 512 groups. While the drafter's indexer runs (QSAIndexer instances with layer_id >= the
target's layer count; the MTP layer is 48), the group scores of groups that end before the last W tokens of each
query row are set to -inf: group g (positions 4g .. 4g + 3) is masked when g < floor((q - W + 1) / 4), q the row's
query position in the selector's coordinates (the same ones its causal visibility uses). With W >= 2,048 at least 512
groups stay visible once the context exceeds W, so the top-512 come from the window; shorter contexts are untouched.
Later draft steps reuse step 0's selection (QSAIndexer.skip_topk). The frequency counters of QSA eviction cover the
target's 12 QSA layers only (qsa_frequency.LAYER_IDS 3 .. 47), so eviction is unaffected.

W lives in a device tensor (graph-safe; 'off' = 2^40, which masks nothing). ARC3_S2_DRAFTWIN=<tokens> sets the initial
W (0 = off). ARC3_S2_DRAFTWIN_SWITCH=<file> (in-run A/B): an integer W, or 'off' / '0', read at most once a second
from the drafter's indexer call (the QSA transaction is a custom op that runs eagerly between the piecewise graphs;
STATS['eager_calls'] confirms it). Receipt: STATS."""
import os
import time

PINS = {'vllm.models.qwen3_8_flash_next.nvidia.indexer_qsa': 'e2f398a2fe29466c9681627651ccb5b8eb2b5980445c9e44b270ff45bcb61066',
        'vllm.models.qwen3_8_flash_next.nvidia.ops.qsa': 'f7815dcce49ffecd50544a58fffaea614c9ed779bb2aa57617451893bda10fef'}
OFF = 1 << 40
TARGET_LAYERS = 48
STATS = dict(window=0, eager_calls=0, masked_calls=0, switches=0, drafter_layers=[])
_S = dict(path=os.environ.get('ARC3_S2_DRAFTWIN_SWITCH'), t=0.0, w=int(os.environ.get('ARC3_S2_DRAFTWIN', '0') or 0),
          tensor=None, active=False)
STATS['window'] = _S['w']


def _parse(text):
    text = text.strip()
    if text in ('', 'off', '0'):
        return 0
    return int(text)


def _w_tensor(device):
    import torch
    if _S['tensor'] is None:
        _S['tensor'] = torch.full((), _S['w'] or OFF, dtype=torch.int64, device=device)
    return _S['tensor']


def _sync(device):
    t = _w_tensor(device)
    if _S['path'] is None:
        return
    now = time.monotonic()
    if now - _S['t'] < 1.0:
        return
    _S['t'] = now
    try:
        with open(_S['path']) as f:
            w = _parse(f.read())
    except (OSError, ValueError):
        return
    if w != _S['w']:
        _S['w'] = w
        t.fill_(w or OFF)
        STATS['window'] = w
        STATS['switches'] += 1


def mask_groups(logits, query_positions, compress_ratio, w_tensor):
    """In place: -inf for the groups before each row's last W tokens (see the module docstring)."""
    import torch
    lo = torch.div(query_positions.to(torch.int64) - w_tensor + 1, compress_ratio, rounding_mode='floor')
    cols = torch.arange(logits.shape[1], device=logits.device)
    logits.masked_fill_(cols[None, :] < lo[:, None], float('-inf'))
    return logits


def patch_ops(mod):
    original = mod.qsa_mqa_paged

    def qsa_mqa_paged(q, k_cache, page_table, token_to_req, query_positions, sequence_lengths, compress_ratio,
                      *args, **kwargs):
        logits, visible = original(q, k_cache, page_table, token_to_req, query_positions, sequence_lengths,
                                   compress_ratio, *args, **kwargs)
        if _S['active'] and _S['tensor'] is not None:
            mask_groups(logits, query_positions, compress_ratio, _S['tensor'])
            STATS['masked_calls'] += 1
        return logits, visible

    qsa_mqa_paged.__wrapped__ = original
    mod.qsa_mqa_paged = qsa_mqa_paged


def patch_indexer(mod):
    cls = mod.QSAIndexer
    original = cls.forward

    def forward(self, hidden_states, positions, out=None):
        if getattr(self, 'layer_id', -1) < TARGET_LAYERS:
            return original(self, hidden_states, positions, out)
        if self.layer_id not in STATS['drafter_layers']:
            STATS['drafter_layers'].append(self.layer_id)
        STATS['eager_calls'] += 1
        _sync(hidden_states.device)
        _S['active'] = True
        try:
            return original(self, hidden_states, positions, out)
        finally:
            _S['active'] = False

    forward.__wrapped__ = original
    cls.forward = forward


PATCHERS = {'vllm.models.qwen3_8_flash_next.nvidia.indexer_qsa': patch_indexer,
            'vllm.models.qwen3_8_flash_next.nvidia.ops.qsa': patch_ops}
