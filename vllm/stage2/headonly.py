"""stage2_v1 'headonly' part (milestone 2, 2026-10-01): the FP8 lm_head tier (B1) without a dense tier.

The dense-FP8 adapter installs the lm_head tier only with a dense tier on, in two places:
  1. its bootstrap installs the adapter's import hook only when ARC3_DENSE_FP8_TIER != 'off'
     (dense_fp8_adapter_v1/bootstrap/sitecustomize.py), so under 'off' install_lm_head is never called;
  2. install_lm_head itself returns when `ARC3_DENSE_FP8_LM_HEAD != '1' or tier() in ('off', 'inventory')`
     (runtime.py:205).
So BF16 dense with an FP8 head could not be launched: `qH` equalled `qR` exactly (M2_DECISIONS 13:33). User 15:25: BF16
dense is the likely choice; is the FP8 head still nearly free in decode?

apply_now(), called from stage2_v1.install() before stage2_v1's own finder is inserted:
  - Under tier 'off' with ARC3_DENSE_FP8_LM_HEAD=1, it installs the adapter's import hook (its finder at the front of
    sys.meta_path), as the adapter's bootstrap does for a dense tier. stage2_v1's finder is then inserted in front of
    it, which is production's order. On an embedding import, stage2_v1's loader runs the adapter's loader (module, then
    install_lm_head) and then its own embedding patchers (guards' patch_embedding needs the adapter's lm_head
    installed: it keeps the drafter's second ParallelLMHead on the stock path). If stage2_v1's finder is already
    present, the adapter's is moved behind it.
  - It redefines install_lm_head from the function's own source with the gate narrowed to `tier() == 'inventory'`,
    in the module namespace (in memory; the production file is unchanged, its pin checked).
selected() still quantizes no dense layer under 'off'; the head goes through whatever quantize_layer is installed
(production: mtp0x's fp8w).

Revisions: v1 (import-time transform) was refused, because mtp0x imports the adapter early. v2 patched only the
function; under tier 'off' the adapter's hook was never installed, so the head stayed BF16 (qH1 at 16:25 equalled qR)."""
import hashlib
import importlib
import inspect
import os
import sys

MODULE = 'dense_fp8_adapter_v1.runtime'
PINS = {MODULE: 'acd782838a74189a5ec8f6f220a62d0d83b82b9f33d95c50ccb91013fa51bbed'}
OLD = ("    if os.environ.get('ARC3_DENSE_FP8_LM_HEAD') != '1' or tier() in ('off', 'inventory'):\n"
       "        return\n"
       "    assert 'embedding' not in ORIGINAL\n")
NEW = ("    if os.environ.get('ARC3_DENSE_FP8_LM_HEAD') != '1' or tier() == 'inventory':   # headonly: also with tier 'off'\n"
       "        return\n"
       "    assert 'embedding' not in ORIGINAL\n")
LINEAR = 'vllm.model_executor.layers.linear'
EMBEDDING = 'vllm.model_executor.layers.vocab_parallel_embedding'
STATS = dict(patched=False, hook_installed=False, hook_reordered=False, late_install=False, skipped=None)


def transform(src):
    assert src.count(OLD) == 1, 'headonly: install_lm_head gate anchor'
    return src.replace(OLD, NEW)


def _install_adapter_hook(stage2_finder_cls):
    """Under tier 'off', install the adapter's import hook (its bootstrap skips it), behind stage2_v1's finder."""
    if os.environ.get('ARC3_DENSE_FP8_TIER', 'off') != 'off' or os.environ.get('ARC3_DENSE_FP8_LM_HEAD') != '1':
        return
    import dense_fp8_adapter_v1.import_hook as ih
    if not any(isinstance(f, ih.Finder) for f in sys.meta_path):
        if {LINEAR, EMBEDDING}.intersection(sys.modules):      # this process imported vLLM layers early: no model here
            STATS['skipped'] = 'vLLM layer modules already imported; adapter hook not installed'
            return
        ih.install()                                            # inserts the adapter's finder at the front
        STATS['hook_installed'] = True
    if stage2_finder_cls is not None:                           # keep production's order: stage2_v1's finder in front
        mp = sys.meta_path
        ours = [i for i, f in enumerate(mp) if isinstance(f, stage2_finder_cls)]
        theirs = [i for i, f in enumerate(mp) if isinstance(f, ih.Finder)]
        if ours and theirs and theirs[0] < ours[0]:
            f = mp.pop(theirs[0])
            mp.insert(ours[0], f)                               # ours shifted left by one: insert right behind it
            STATS['hook_reordered'] = True


def apply_now(stage2_finder_cls=None):
    try:
        rt = importlib.import_module(MODULE)
    except ImportError as exc:                 # a process without the dense-FP8 adapter on its path builds no head
        STATS['skipped'] = repr(exc)
        return STATS
    digest = hashlib.sha256(open(rt.__file__, 'rb').read()).hexdigest()
    assert digest == PINS[MODULE], f'stage2_v1 headonly: {MODULE} changed ({digest}); re-pin'
    exec(compile(transform(inspect.getsource(rt.install_lm_head)), rt.__file__, 'exec'), rt.__dict__)
    STATS['patched'] = True
    _install_adapter_hook(stage2_finder_cls)
    emb = sys.modules.get(EMBEDDING)
    if emb is not None and 'embedding' not in rt.ORIGINAL:
        rt.install_lm_head(emb)
        STATS['late_install'] = 'embedding' in rt.ORIGINAL
    return STATS
