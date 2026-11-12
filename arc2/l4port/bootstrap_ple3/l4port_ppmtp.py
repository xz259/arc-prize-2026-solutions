"""L4 port (2026-10-30, ARC2 - main; the user: "We will probably do 2+2 and MTP 3 or 4"): the MTP drafter under pipeline
parallelism, env ARC3_L4PORT_PP_MTP=1. vLLM builds the drafter only on the last PP rank (v1/worker/gpu/model_runner.py:
init_speculator when is_last_pp_rank) and, under PP, does not share the target's embedding with it (spec_decode/eagle/utils.py;
the fr1cfinal checkpoint carries its own embed_tokens and lm_head). The fork's MTP module (models/qwen3_8_flash_next/nvidia/
mtp.py) still branches on get_pp_group().is_first_rank / is_last_rank, so on stage 1 its forward took the 'non-first rank'
branch and asserted intermediate_tensors (4090 c18a, profile run). The drafter is a single-stage module on the last rank, so the
module-level get_pp_group of mtp.py is replaced by one returning a view that reports first = last = True (everything else
forwarded to the real group); at PP 1 the real group is returned unchanged. Post-import hook, no source edit; one receipt line
per process at the first patched call."""
import importlib.abc
import os
import sys

NAME = "vllm.models.qwen3_8_flash_next.nvidia.mtp"


class _SingleStage:
    is_first_rank = True
    is_last_rank = True

    def __init__(self, group):
        self._group = group

    def __getattr__(self, key):
        if key == "_group":
            raise AttributeError(key)
        return getattr(self._group, key)


def _patch(module):
    orig = module.get_pp_group
    cache = {}

    def get_pp_group():
        # First called eagerly in Qwen3_8FlashNextMTP.__init__ (the lm_head choice), which fills the cache; the forward, traced
        # later by torch.compile, then only does a dict lookup and reads constant class attributes (no prints, no new objects).
        v = cache.get("view")
        if v is not None:
            return v
        g = orig()
        if g.world_size > 1 and g.is_last_rank:
            cache["view"] = _SingleStage(g)
            print("[l4port] PP MTP: the drafter module sees a single-stage PP group (pp world %d, last rank)" % g.world_size,
                  flush=True)
            return cache["view"]
        return g

    get_pp_group.__wrapped__ = orig
    module.get_pp_group = get_pp_group


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != NAME:
            return None
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


def install():
    if os.environ.get("ARC3_L4PORT_PP_MTP") != "1":
        return
    if NAME in sys.modules:
        _patch(sys.modules[NAME])
    else:
        sys.meta_path.insert(0, _Finder())
