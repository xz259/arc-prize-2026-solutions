"""Opt-in serving adapter for the separately validated sampler search guard.

The shared dispatcher imports this adapter only when its startup flag is one.
There is no production per-row GPU instrumentation and no RNG replacement.
"""
from arc3_sampler_topk_guard_impl import apply_top_k_top_p_triton as _implementation

_FORCE_SKIP_REDUNDANT_TOPK = True
_CALLS = 0
_ROWS = {}


def apply_top_k_top_p_triton(logits, k, p, mask_value=float('-inf')):
    global _CALLS
    _CALLS += 1
    rows = int(logits.shape[0])
    _ROWS[rows] = _ROWS.get(rows, 0)+1
    return _implementation(logits, k, p, mask_value,
                           skip_redundant_topk=_FORCE_SKIP_REDUNDANT_TOPK)
