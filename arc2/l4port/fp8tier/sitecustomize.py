"""ARC-2 FP8 tier hook (2026-10-30, ARC2 - main; the user: FP8 on GDN in_proj_qkvz + QSA qkv_proj only, out projections BF16).

Sits in the bootstrap chain in place of dense_fp8_adapter_v1's bootstrap: runs that bootstrap by path (ARC3_L4PORT_FP8TIER_NEXT),
then, when ARC3_DENSE_FP8_TIER names a tier defined here, adds it to the adapter's TIERS before any layer is selected
(dense_fp8_adapter_v1.runtime imports only the standard library at module level; the adapter's import hook later imports the
same module object). Tiers:
  A_in = A_qkvz + A_qkv (48 layers: 36 linear_attn.in_proj_qkvz + 12 self_attn.qkv_proj; out_proj / o_proj stay BF16).
The lm_head stays on ARC3_DENSE_FP8_LM_HEAD as before. No production file changes; with any other tier this only runs the next
bootstrap. Failures exit the process (93), as the adapter's own bootstrap does (92)."""
import os
import runpy

EXTRA = {'A_in': ('A_qkvz', 'A_qkv')}
try:
    runpy.run_path(os.environ['ARC3_L4PORT_FP8TIER_NEXT'], run_name='arc3_chained_sitecustomize')
    t = os.environ.get('ARC3_DENSE_FP8_TIER', 'off')
    if t in EXTRA:
        from dense_fp8_adapter_v1 import runtime
        for name in EXTRA[t]:
            assert name in runtime.TIER_SUFFIXES, name
        runtime.TIERS.setdefault(t, EXTRA[t])
        assert runtime.TIERS[t] == EXTRA[t], (t, runtime.TIERS[t])
except BaseException:
    import traceback
    traceback.print_exc()
    os._exit(93)
