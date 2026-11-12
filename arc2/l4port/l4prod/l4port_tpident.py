"""l4port_tpident.py (2026-10-31, ARC2 - main; the reviewer's identity check after l4p38, before any long run): are the TP ranks of each
pipeline stage bitwise identical where vLLM requires it? The QSA counters of the two ranks differ at ~1e-6 (l4p38), which a selector
tie-break explains, but so would a one-ulp input difference between the ranks, and that would be a desync risk (each rank samples its
own token). Diagnostic only, env-gated (ARC3_L4PORT_TPIDENT=N: the first N model calls of each worker; ARC3_L4PORT_TPIDENT_DIR: the
output directory); installed by bootstrap_l4prod v6; off in every production and throughput launch.

Per worker, for its first N GPUModelRunner.execute_model calls, a 64-bit position-weighted sum of the raw bits of:
  pp_out  the IntermediateTensors a non-last stage returns (every tensor), i.e. the hidden state the next stage receives;
  logits  the target model's compute_logits output (last stage);
  tokens  the sampler's sampled token ids (last stage).
Each record: {call, kind, pp, tp, shape, h}. compare_tpident.py joins the TP ranks of a stage by (call, kind) and counts mismatches.
Expected: 0 for all three while the QSA counters still differ (then the cause is the selector's tie-break; no fix needed)."""
import importlib.abc
import json
import os
import sys

MR = "vllm.v1.worker.gpu.model_runner"
N = int(os.environ.get("ARC3_L4PORT_TPIDENT", "0") or 0)
DIR = os.environ.get("ARC3_L4PORT_TPIDENT_DIR", "/tmp/tpident")
_BUSY = set()
STATE = dict(calls=0, file=None, w={}, rank=None)


def _rank():
    if STATE["rank"] is None:
        from vllm.distributed.parallel_state import get_pp_group, get_tp_group
        STATE["rank"] = (int(get_pp_group().rank_in_group), int(get_tp_group().rank_in_group))
    return STATE["rank"]


def _hash(x):
    import torch
    flat = x.detach().contiguous().view(-1)
    size = flat.element_size()
    bits = flat.view({1: torch.int8, 2: torch.int16, 4: torch.int32, 8: torch.int64}[size]).to(torch.int64)
    n = bits.numel()
    w = STATE["w"].get((n, str(bits.device)))
    if w is None:
        w = STATE["w"][(n, str(bits.device))] = torch.arange(n, device=bits.device, dtype=torch.int64) % 65521 + 1
    return [int(bits.sum()), int((bits * w).sum())]


def _record(kind, x):
    if STATE["calls"] > N or x is None:
        return
    pp, tp = _rank()
    if STATE["file"] is None:
        os.makedirs(DIR, exist_ok=True)
        STATE["file"] = open(os.path.join(DIR, "tpident_pp%d_tp%d_%d.jsonl" % (pp, tp, os.getpid())), "w")
    seqs = STATE.setdefault("seq", {})
    k = (STATE["calls"], kind)
    seqs[k] = seqs.get(k, -1) + 1
    STATE["file"].write(json.dumps(dict(call=STATE["calls"], kind=kind, seq=seqs[k], pp=pp, tp=tp, shape=list(x.shape),
                                        h=_hash(x))) + "\n")
    STATE["file"].flush()


def _patch_mr(module):
    cls = module.GPUModelRunner
    if getattr(cls, "_l4_tpident", False):
        return
    cls._l4_tpident = True
    orig_exec, orig_load = cls.execute_model, cls.load_model

    def execute_model(self, *a, **k):
        STATE["calls"] += 1
        out = orig_exec(self, *a, **k)
        if STATE["calls"] <= N:
            tensors = getattr(out, "tensors", None)
            if isinstance(tensors, dict):
                for name in sorted(tensors):
                    _record("pp_out:" + name, tensors[name])
        return out

    def load_model(self, *a, **k):
        res = orig_load(self, *a, **k)
        model = self.model
        orig_logits = model.compute_logits

        def compute_logits(*la, **lk):
            y = orig_logits(*la, **lk)
            if STATE["calls"] <= N:
                _record("logits", y)
            return y

        model.compute_logits = compute_logits
        sampler = getattr(self, "sampler", None)
        if sampler is not None:
            import torch
            scls = type(sampler)
            meth = "forward" if isinstance(sampler, torch.nn.Module) else "__call__"
            orig_m = getattr(scls, meth)
            if not getattr(orig_m, "_l4_tpident", False):
                def wrapped(this, *sa, **sk):
                    o = orig_m(this, *sa, **sk)
                    if STATE["calls"] <= N:
                        _record("tokens", getattr(o, "sampled_token_ids", None))
                    return o

                wrapped._l4_tpident = True
                wrapped.__wrapped__ = orig_m
                setattr(scls, meth, wrapped)
        print("[l4port] tpident: recording the first %d model calls of this worker into %s" % (N, DIR), flush=True)
        return res

    execute_model.__wrapped__ = orig_exec
    load_model.__wrapped__ = orig_load
    cls.execute_model = execute_model
    cls.load_model = load_model


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != MR or fullname in _BUSY:
            return None
        _BUSY.add(fullname)
        try:
            for finder in list(sys.meta_path):
                if finder is self or getattr(finder, "find_spec", None) is None:
                    continue
                spec = finder.find_spec(fullname, path, target)
                if spec is not None:
                    orig_exec = spec.loader.exec_module

                    def exec_module(module, _orig=orig_exec):
                        _orig(module)
                        _patch_mr(module)

                    spec.loader.exec_module = exec_module
                    return spec
            return None
        finally:
            _BUSY.discard(fullname)


def install():
    if N <= 0:
        return
    if MR in sys.modules:
        _patch_mr(sys.modules[MR])
    else:
        sys.meta_path.insert(0, _Finder())
