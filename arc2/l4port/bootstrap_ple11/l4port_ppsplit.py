"""L4 port (2026-10-31, ARC2 - main): uneven TP2 x PP2 layer split for the main stages only (env ARC3_L4PORT_PP_SPLIT=25,23).
vLLM's VLLM_PP_LAYER_PARTITION is read by every get_pp_indices call in every process, and the PLE offload worker builds the full
model with pp_size 1, so the 4090's l27b (VLLM_PP_LAYER_PARTITION=25,23) died at startup: 'len(partitions)=2 does not match
pp_size=1'. This wraps vllm.distributed.utils.get_pp_indices (its callers import it inside their function bodies, so the module
attribute is what they get): the split applies when pp_size equals its length and num_hidden_layers its sum (the target model on
the two main stages; KV-group projection and the worker's layer queries ask the same question); every other call (the PLE worker's
pp_size 1, the drafter's own layer count) goes to the stock function, which then sees no partition env. Refuses to run with
VLLM_PP_LAYER_PARTITION also set. One receipt line per (layers, rank, size) applied."""
import importlib
import importlib.abc
import os
import sys

MOD = "vllm.distributed.utils"
SPLIT = [int(x) for x in os.environ.get("ARC3_L4PORT_PP_SPLIT", "").split(",") if x.strip()]
_SEEN = set()
_BUSY = set()


def _patch(module):
    orig = module.get_pp_indices
    if getattr(orig, "_l4_split", False):
        return

    def get_pp_indices(num_hidden_layers, pp_rank, pp_size):
        if len(SPLIT) == pp_size and sum(SPLIT) == num_hidden_layers and pp_size > 1:
            start = sum(SPLIT[:pp_rank])
            out = (start, start + SPLIT[pp_rank])
            key = (num_hidden_layers, pp_rank, pp_size)
            if key not in _SEEN:
                _SEEN.add(key)
                print("[l4port] PP split: %d layers, rank %d of %d -> [%d, %d)" % (num_hidden_layers, pp_rank, pp_size, *out),
                      flush=True)
            return out
        return orig(num_hidden_layers, pp_rank, pp_size)

    get_pp_indices._l4_split = True
    get_pp_indices.__wrapped__ = orig
    module.get_pp_indices = get_pp_indices


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != MOD or fullname in _BUSY:
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
    if not SPLIT:
        return
    if os.environ.get("VLLM_PP_LAYER_PARTITION"):
        raise RuntimeError("ARC3_L4PORT_PP_SPLIT and VLLM_PP_LAYER_PARTITION are both set")
    if MOD in sys.modules:
        _patch(sys.modules[MOD])
    else:
        sys.meta_path.insert(0, _Finder())
