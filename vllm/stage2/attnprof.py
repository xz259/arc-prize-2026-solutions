"""stage2_v1 'attnprof' part: split counts of production's NVFP4 sparse QSA reader retuned for MTP3 batch shapes
(GAP_PLAN item 5, 2026-09-30).

arc3_qsa_profiles.quantized_sparse_profile ("measured SM120 launch policy") keeps tile 16, four warps and one stage for
NVFP4 (spill-free), and picks split-K by base programs (query rows x 2 KV heads): 16 up to 256, 4 up to 512, 1 above.
The thresholds date from MTP0 shapes. With MTP3 a decode step already has about 192 base programs and a step carrying a
typical resume (about 92 decode rows plus 250 prefill rows) about 680, which falls to one split although four measure
faster (bench_nvfp4_profile.py). Only the split count changes, and only for the ranges in SPLITS; tile, warps and stages
stay production's. Split-K changes the reduction order (not bit-identical; within production's own batch-dependent
variation, since the split count already depends on the batch). Decode steps replay FULL CUDA graphs captured with the
profile of their capture; mixed steps run the attention eagerly (PIECEWISE splits at attention), so
ARC3_S2_ATTNPROF_SWITCH=<file> containing 'default' restores production's profile for them within a run (A/B)."""
import os
import time

PINS = {'arc3_qsa_profiles': '85657cb0b6899f770202a9db9bcaf1bded2d48abaad5faf501b216aee72e2ea1'}
# ((low, high, splits), ...) over base programs, inclusive. bench_nvfp4_profile.py --splits (92 decode rows plus P prefill
# rows): 4 splits against production's 1 at 524 / 684 / 984 / 1,284 / 1,584 / 1,884 base programs: 1.04 / 1.34 / 1.15 /
# 1.04 / 1.06 / 1.06x; from 2,384 production's 1 is best; production's own ranges (<= 512) are kept.
SPLITS = ((513, 2000, 4),)
STATS = dict(calls=0, changed=0)
_SWITCH = dict(path=os.environ.get('ARC3_S2_ATTNPROF_SWITCH'), t=0.0, default=False)


def _default():
    s = _SWITCH
    if s['path'] is None:
        return False
    now = time.monotonic()
    if now - s['t'] > 1.0:
        s['t'] = now
        try:
            with open(s['path']) as f:
                s['default'] = f.read().strip() == 'default'
        except OSError:
            s['default'] = False
    return s['default']


def profile(original, base_programs, block_m, fmt):
    prof = original(base_programs, block_m, fmt)
    if fmt != 'nvfp4':
        return prof
    STATS['calls'] += 1
    if _default():
        return prof
    for low, high, splits in SPLITS:
        if low <= base_programs <= high:
            if splits != prof[1]:
                STATS['changed'] += 1
            return (prof[0], splits, prof[2], prof[3])
    return prof


def patch_profiles(mod):
    original = mod.quantized_sparse_profile

    def quantized_sparse_profile(base_programs, block_m, fmt='fp8'):
        return profile(original, base_programs, block_m, fmt)

    mod.quantized_sparse_profile = quantized_sparse_profile


PATCHERS = {'arc3_qsa_profiles': patch_profiles}
