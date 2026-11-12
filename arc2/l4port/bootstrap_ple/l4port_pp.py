"""L4 port (2026-10-30): pipeline-parallel support hooks for the fork's model, without touching its sources (model.py is pinned
by eprune, rollback and dflashtaps; model_state.py by promptmove and steptimer). Installed by the bootstrap's sitecustomize
when ARC3_L4PORT_PP_SKIPMIXER=1. Each hook prints one receipt line per process.
  model.py        on a non-last PP stage Qwen3_8FlashNextModel sets hyper_connection_mixer = None, but its load_weights still
                  receives the checkpoint's hyper_connection_mixer.* tensors and AutoWeightsLoader raises (c5dpp): those
                  tensors are dropped on stages without the mixer.
  model_state.py  Qwen3_8FlashNextModelState refuses PP > 1 on every rank because non-first stages lack the raw input_ids
                  the n-gram context needs (c7pp). This model's PLE sits at layer 2, inside stage 0, so: __init__ runs with
                  the PP size presented as 1, and on non-first stages the n-gram context is switched off (no PLE layer there).
                  The first stage keeps it unchanged."""
import importlib.abc
import sys

STATS = dict(dropped=0)


def _patch_model(module):
    cls = module.Qwen3_8FlashNextModel
    orig = cls.load_weights

    def load_weights(self, weights):
        if getattr(self, "hyper_connection_mixer", 1) is None:
            def keep(stream):
                for name, w in stream:
                    if "hyper_connection_mixer." in name:
                        STATS["dropped"] += 1
                        continue
                    yield name, w
            out = orig(self, keep(weights))
            print("[l4port] PP: dropped %d hyper_connection_mixer tensors on a stage without the final mixer" % STATS["dropped"],
                  flush=True)
            return out
        return orig(self, weights)

    cls.load_weights = load_weights


def _patch_state(module):
    cls = module.Qwen3_8FlashNextModelState
    orig = cls.__init__

    def __init__(self, vllm_config, *args, **kwargs):
        pc = vllm_config.parallel_config
        pp = pc.pipeline_parallel_size
        if pp <= 1:
            return orig(self, vllm_config, *args, **kwargs)
        from vllm.distributed import get_pp_group
        first = get_pp_group().is_first_rank
        object.__setattr__(pc, "pipeline_parallel_size", 1)
        try:
            orig(self, vllm_config, *args, **kwargs)
        finally:
            object.__setattr__(pc, "pipeline_parallel_size", pp)
        if not first:
            self.uses_ngram_embedding = False
        print("[l4port] PP: model state on the %s stage, n-gram context %s" % (
            "first" if first else "a later", "kept" if self.uses_ngram_embedding else "off"), flush=True)

    cls.__init__ = __init__


PATCHES = {"vllm.models.qwen3_8_flash_next.nvidia.model": _patch_model,
           "vllm.models.qwen3_8_flash_next.nvidia.model_state": _patch_state}


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        patch = PATCHES.get(fullname)
        if patch is None:
            return None
        for finder in list(sys.meta_path):
            if finder is self:
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                loader = spec.loader
                orig_exec = loader.exec_module

                def exec_module(module, _orig=orig_exec, _patch=patch):
                    _orig(module)
                    _patch(module)

                loader.exec_module = exec_module
                return spec
        return None


def install():
    for name, patch in PATCHES.items():
        mod = sys.modules.get(name)
        if mod is not None:          # already imported (e.g. by an earlier bootstrap): patch the module object directly
            patch(mod)
    if any(name not in sys.modules for name in PATCHES):
        sys.meta_path.insert(0, _Finder())
