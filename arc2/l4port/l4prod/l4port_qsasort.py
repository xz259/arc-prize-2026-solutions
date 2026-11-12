"""L4 port (2026-11-01, ARC2 - main; the user: fix it): a fixed order for the native QSA top-k's selection, env ARC3_L4PORT_QSA_SORT=1
(unset: unchanged). On sm_89 the L4 launcher runs ARC3_QSA_MODE=native (the radix selector has no sm_89 kernel), and the native
torch.ops._C.persistent_topk writes each row's selected blocks in an order that changes from run to run (qsa_ws/t_qsa_det.py: every
row of a 2,048-row batch with more than 512 visible blocks). The sparse attention reduces over the index list in that order (split-K
tiles, online softmax), so the order reaches its output at the ulp level. Rows with <= 512 visible blocks (<= 2,048 tokens) come back
ascending already. Lever 4's sharded selection writes its merge ascending, so on the shipped config only the drafter's (unsharded)
select is affected; TP 1 or shard-off runs expose every QSA layer.
This hook sorts the first min(visible, k) entries of each row of the stock select's top-k output ascending (the entries the
expansion reads) and writes -1 after them, between the native op and the expansion: it wraps ops.qsa.qsa_select_paged_tokens to
mark its calls and arc3_determinism.select_topk to sort its output inside a marked call (the sharded select's local top-k is left
as is). Among exactly tied scores the native op's choice of set stays as it is. Eager (the QSA transaction runs outside the CUDA
graphs). Post-import hook on vllm.models.qwen3_8_flash_next.nvidia.ops.qsa. Test: qsa_ws/t_qsa_sort.py. Receipt: one line per
process; STATE."""
import importlib.abc
import os
import sys

NAME = "vllm.models.qwen3_8_flash_next.nvidia.ops.qsa"
STATE = dict(depth=0, installed=0, sorted_calls=0, printed=False)


def _sort_rows(output, lengths, k):
    import torch
    big = torch.iinfo(output.dtype).max
    count = torch.clamp(lengths.to(torch.int64), min=0, max=k)
    pos = torch.arange(output.shape[1], device=output.device)
    key = torch.where(pos[None, :] < count[:, None], output, torch.full_like(output, big))
    vals = torch.sort(key, dim=1).values
    output.copy_(torch.where(vals == big, torch.full_like(vals, -1), vals))


def _patch(module):
    import arc3_determinism as det
    orig_select = module.qsa_select_paged_tokens
    orig_topk = det.select_topk

    def select_topk(native_op, logits, lengths, output, workspace, k, columns):
        orig_topk(native_op, logits, lengths, output, workspace, k, columns)
        if STATE["depth"] > 0 and output.numel():
            _sort_rows(output, lengths, k)
            STATE["sorted_calls"] += 1
            if not STATE["printed"]:
                STATE["printed"] = True
                print("[l4port] qsasort: first select sorted (rows %d, k %d, pid %d)" % (output.shape[0], k, os.getpid()), flush=True)

    def qsa_select_paged_tokens(*args, **kwargs):
        STATE["depth"] += 1
        try:
            return orig_select(*args, **kwargs)
        finally:
            STATE["depth"] -= 1

    select_topk.__wrapped__ = orig_topk
    qsa_select_paged_tokens.__wrapped__ = orig_select
    det.select_topk = select_topk
    module.qsa_select_paged_tokens = qsa_select_paged_tokens
    STATE["installed"] += 1
    print("[l4port] qsasort: installed (QSA mode %s, pid %d)" % (det.QSA_MODE, os.getpid()), flush=True)


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
    if os.environ.get("ARC3_L4PORT_QSA_SORT") != "1":
        return
    if NAME in sys.modules:
        _patch(sys.modules[NAME])
    else:
        sys.meta_path.insert(0, _Finder())
