"""L4 port (2026-10-30, ARC2 - main): input embedding tables on the host under TP2 x PP2 (env ARC3_L4PORT_PP_EMBHOST=1). Production's
'embedhost' part (stage2_v1/embedhost.py) moves the target's embed_tokens to pinned host memory behind a UVA view and asserts the
MTP drafter shares that module; under PP the target's embedding exists only on the first stage (PPMissingLayer elsewhere) and vLLM
gives the drafter, which lives on the last stage, its own copy (eagle/utils skips sharing under PP). That copy is byte-identical to
the target's table (torch.equal, checked 2026-10-30) and costs ~0.59 GiB per GPU at TP=2 on the stage that binds the shared KV
block count (the 4090's c25a KV dump: stage 1 room 1.745 GiB vs 2.64 on stage 0). After GPUModelRunner.load_model this moves, with
embedhost.host_view (whole-table chunk check through the UVA view before the swap, so gathers stay bit-identical):
  - on the first PP rank: the target's embed_tokens;
  - on the last PP rank: the drafter's own embed_tokens (Qwen3_8FlashNextMTP.model.embed_tokens).
Then empty_cache before the profile run, as embedhost does. PP 1 is left to the stock 'embedhost' part. One receipt line per rank
(what moved, bytes, allocated freed). Post-import hook on vllm.v1.worker.gpu.model_runner with a re-entrancy guard (the draft
broadcast hook also hooks that module)."""
import gc
import importlib
import importlib.abc
import os
import sys

MR = "vllm.v1.worker.gpu.model_runner"
_BUSY = set()


def _move(self):
    import torch
    from vllm.distributed import get_pp_group
    g = get_pp_group()
    if g.world_size == 1:
        return
    eh = importlib.import_module("stage2_v1.embedhost")
    model = self.model
    lm = model.get_language_model() if hasattr(model, "get_language_model") else model
    todo = []
    emb = getattr(getattr(lm, "model", None), "embed_tokens", None)
    if g.is_first_rank and type(emb).__name__ == "VocabParallelEmbedding" and emb.weight.is_cuda:
        todo.append(("target", emb))
    draft = getattr(getattr(getattr(self, "speculator", None), "model", None), "model", None)
    demb = getattr(draft, "embed_tokens", None)
    if g.is_last_rank and type(demb).__name__ == "VocabParallelEmbedding" and demb.weight.is_cuda and demb is not emb:
        todo.append(("drafter", demb))
    if not todo:
        return
    torch.cuda.synchronize()
    alloc0 = torch.cuda.memory_allocated()
    moved = []
    for who, e in todo:
        w = e.weight
        assert w.dtype == torch.bfloat16 and w.dim() == 2 and w.is_contiguous(), (who, w.dtype, tuple(w.shape))
        host, chunks = eh.host_view(w)
        e._arc3_host_weight = host
        moved.append("%s %s (%d chunks)" % (who, tuple(host.shape), chunks))
        del w
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    freed = alloc0 - torch.cuda.memory_allocated()
    print("[l4port] PP embhost: pp rank %d of %d moved %s to pinned host memory; allocated freed %.3f GiB"
          % (g.rank_in_group, g.world_size, ", ".join(moved), freed / 2**30), flush=True)


def _patch(module):
    cls = module.GPUModelRunner
    if getattr(cls, "_l4_ppembhost", False):
        return
    cls._l4_ppembhost = True
    orig = cls.load_model

    def load_model(self, *args, **kwargs):
        out = orig(self, *args, **kwargs)
        if not kwargs.get("load_dummy_weights", args[0] if args else False):
            _move(self)
        return out

    load_model.__wrapped__ = orig
    cls.load_model = load_model


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != MR or fullname in _BUSY:
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
    if os.environ.get("ARC3_L4PORT_PP_EMBHOST") != "1":
        return
    if MR in sys.modules:
        _patch(sys.modules[MR])
    else:
        sys.meta_path.insert(0, _Finder())
