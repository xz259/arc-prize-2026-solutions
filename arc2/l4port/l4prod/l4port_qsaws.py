"""L4 port lane item (2026-11-01, ARC2 - main; "the drafter QSA buffer"): the QSA selection's logits workspace, env
ARC3_L4PORT_QSA_WS_MIB=<MiB> (unset or 0: unchanged, 128 MiB). ops/qsa.py's qsa_select_paged_tokens, and lever 4's sharded select
(l4port_idxshard_v2._sharded_select, which reads the same constant), score the query rows in chunks of at most
_LOGITS_WORKSPACE_BYTES of float32 logits. At batched tokens 2048 the drafter's select holds 2,048 x 15,124 x 4 B = 118 MiB at once
(pbd6's memdiag: PP 1's largest transient, under the draftwin wrapper's frame), the sharded target select 59 MiB. Every step of the
selection is row-local (the scoring kernel's program per row, the top-k per row, the shard's keys and merge per row), so a smaller
workspace selects the same tokens bit for bit (qsa_ws/t_qsa_ws.py); prefill steps launch more, smaller chunks; decode steps (<= 85
rows) are unchanged above ~6 MiB. Post-import hook on vllm.models.qwen3_8_flash_next.nvidia.ops.qsa: sets the module constant
(both selects read it at call time; the selection is an eager custom op, outside the compiled graphs and the AOT key). One receipt
line per process."""
import importlib.abc
import os
import sys

NAME = "vllm.models.qwen3_8_flash_next.nvidia.ops.qsa"
STOCK_BYTES = 128 * 1024 * 1024
STATE = dict(printed=False)


def _mib():
    return int(os.environ.get("ARC3_L4PORT_QSA_WS_MIB", "0") or 0)


def _patch(module):
    mib = _mib()
    old = module._LOGITS_WORKSPACE_BYTES
    if old != STOCK_BYTES:
        raise RuntimeError("qsaws: ops.qsa._LOGITS_WORKSPACE_BYTES is %d, not the stock %d" % (old, STOCK_BYTES))
    module._LOGITS_WORKSPACE_BYTES = mib * 1024 * 1024
    if not STATE["printed"]:
        STATE["printed"] = True
        print("[l4port] qsaws: QSA logits workspace %d -> %d MiB (pid %d)" % (old >> 20, mib, os.getpid()), flush=True)


_BUSY = set()


class _Finder(importlib.abc.MetaPathFinder):
    # v2 (2026-11-01): a re-entrancy guard. Several L4 hooks put a finder for the same module (ops.qsa: qsaws and qsasort) on
    # sys.meta_path, each delegating to every other finder, so without the guard they recursed into each other (pbl1: RecursionError
    # at startup; the l4port_ppdraftbc / ppbcast precedent on the 4090's c24a)
    def find_spec(self, fullname, path=None, target=None):
        if fullname != NAME or fullname in _BUSY:
            return None
        _BUSY.add(fullname)
        try:
            for finder in list(sys.meta_path):
                if finder is self:
                    continue
                spec = finder.find_spec(fullname, path, target)
                if spec is not None:
                    orig_exec = spec.loader.exec_module

                    def exec_module(module, _orig=orig_exec):
                        _orig(module)
                        _patch(module)

                    spec.loader.exec_module = exec_module
                    return spec
            return None
        finally:
            _BUSY.discard(fullname)


def install():
    mib = _mib()
    if mib <= 0:
        return
    if mib > STOCK_BYTES >> 20:
        raise RuntimeError("ARC3_L4PORT_QSA_WS_MIB=%d is above the stock %d MiB" % (mib, STOCK_BYTES >> 20))
    if NAME in sys.modules:
        _patch(sys.modules[NAME])
    else:
        sys.meta_path.insert(0, _Finder())
