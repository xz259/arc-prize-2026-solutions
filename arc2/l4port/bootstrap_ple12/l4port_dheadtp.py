"""L4 port (2026-10-31, ARC2 - main): the MTP drafter's LM head over production's draft vocabulary S at TP > 1 (env
ARC3_L4PORT_DHEAD_TP=1; S = ARC3_S2_DRAFTHEAD_VOCAB, draft_vocab_32k.json's token ranges, as stage2_v1/drafthead.py).
The L4 profile (l8pp, PP K 4) put 4 x 2.59 ms per stage-1 microbatch into the drafter's full head: V = 248,320, so each TP=2
rank reads a ~636 MB shard per draft pass. drafthead (production, TP 1) slices the full head weight by global ids and fails at
TP 2 (a rank holds only its vocab shard). Here each rank keeps the rows of S inside its own shard ([n_r, H] copy, ~82 MB at TP 2),
computes those logits, takes its local top-K (K = ARC3_L4PORT_DHEAD_K, default the vptopk K or 20), and all-gathers values and
global ids ([M, 2K] FP32, ids exact below 2^24); the [M, V] logits are -inf except the tp x K candidates, which hold the head's
values. That is vptopk's drafter rule (today the drafter's logits are already the top-K candidates of the full head) applied to
S: drafts come only from S's per-rank top-K, the verifier checks each draft against that same cached distribution, so the output
stays exact and only acceptance can move. Installed by wrapping GPUModelRunner.load_model: there (eager, before the profile run)
the subset rows are copied and checked against the full head's columns on a random probe, and the class of the loaded speculator
model is patched. Memory (env ARC3_L4PORT_DHEAD_FREE=1): under PP the drafter carries its own full head (stage 1 weighs 15.27 GiB vs
stage 0's 13.92 on the L4); if that head is a separate tensor and torch.equal to the target's head shard on this rank, its weight
is released (the patched compute_logits never reads it) and the CUDA cache emptied before KV sizing, as the host-embedding hook
does. Refuses to run beside the 'drafthead' part. Receipt lines at load and at the first call."""
import importlib
import importlib.abc
import json
import os
import sys

MR = "vllm.v1.worker.gpu.model_runner"
_BUSY = set()
STATE = dict(ready=False, printed=False, calls=0)


def _ids(path):
    d = json.load(open(path))
    ids = sorted({i for a, b in d["token_ranges"] for i in range(int(a), int(b))} | set(d.get("extra_ids", [])))
    assert ids and ids[0] >= 0, path
    return ids


def _prepare(model, target_head):
    import torch
    from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
    lm = model.lm_head
    si = lm.shard_indices
    lp = model.logits_processor
    V = int(getattr(lp, "org_vocab_size", None) or lp.vocab_size)
    ids = _ids(os.environ["ARC3_S2_DRAFTHEAD_VOCAB"])
    assert ids[-1] < V, (ids[-1], V)
    lo, hi = si.org_vocab_start_index, si.org_vocab_end_index
    loc = [i - lo for i in ids if lo <= i < hi]
    k = int(os.environ.get("ARC3_L4PORT_DHEAD_K") or os.environ.get("ARC3_L4PORT_VPTOPK") or 20)
    assert len(loc) >= k, ("too few draft-vocab rows in this shard", lo, hi, len(loc), k)
    w = lm.weight
    rows = torch.tensor(loc, dtype=torch.long, device=w.device)
    sub = w.index_select(0, rows).contiguous()
    g = torch.Generator(device=w.device).manual_seed(0)
    h = torch.randn(8, w.shape[1], device=w.device, dtype=w.dtype, generator=g)
    diff = float((torch.nn.functional.linear(h, w).index_select(1, rows).float() - torch.nn.functional.linear(h, sub).float()).abs().max())
    shared = target_head is not None and (target_head is lm or getattr(target_head, "weight", None) is w)
    equal = (not shared and target_head is not None and getattr(target_head, "weight", None) is not None
             and tuple(target_head.weight.shape) == tuple(w.shape) and bool(torch.equal(target_head.weight, w)))
    STATE.update(ready=True, V=V, n=len(ids), n_local=len(loc), k=k, lo=lo, w=sub, gid=(rows + lo).to(torch.float32),
                 tp=get_tensor_model_parallel_world_size(), rank=get_tensor_model_parallel_rank(), dtype=w.dtype, check_diff=diff,
                 shared=shared, equal=equal, freed=0)
    if os.environ.get("ARC3_L4PORT_DHEAD_FREE") == "1" and equal:
        nbytes = w.numel() * w.element_size()
        lm.weight = torch.nn.Parameter(torch.empty((0, w.shape[1]), dtype=w.dtype, device=w.device), requires_grad=False)
        del w
        import gc
        gc.collect()
        torch.cuda.empty_cache()
        STATE["freed"] = nbytes
    print("[l4port] drafter head TP: rank %d of %d, %d of %d draft-vocab rows in shard [%d, %d) (%.1f MB copy), top-%d per rank, "
          "V %d, probe max |subset - full head| %.3g; drafter head shared with the target %s, equal %s, released %.1f MB" % (
              STATE["rank"], STATE["tp"], len(loc), len(ids), lo, hi, sub.numel() * sub.element_size() / 1e6, k, V, diff,
              shared, equal, STATE["freed"] / 1e6), flush=True)


def _patch_model(cls):
    if getattr(cls, "_l4_dheadtp", False):
        return
    cls._l4_dheadtp = True
    orig = cls.compute_logits

    def compute_logits(self, hidden_states, spec_step_idx=0):
        import torch
        st = STATE
        assert st["ready"], "drafter head TP: not prepared at load"

        st["calls"] += 1
        m = hidden_states.shape[0]
        y = torch.nn.functional.linear(hidden_states, st["w"]).float()          # [M, n_local]
        v, i = torch.topk(y, st["k"], dim=1)
        pair = torch.cat([v, st["gid"][i]], dim=1)                               # [M, 2K]: values, global ids
        if st["tp"] > 1:
            from vllm.distributed import tensor_model_parallel_all_gather
            pair = tensor_model_parallel_all_gather(pair, dim=-1)                # [M, tp x 2K]
        pair = pair.view(m, -1, 2, st["k"])
        vals = pair[:, :, 0, :].reshape(m, -1)
        gids = pair[:, :, 1, :].reshape(m, -1).long()
        out = torch.full((m, st["V"]), float("-inf"), dtype=st["dtype"], device=hidden_states.device)
        out.scatter_(1, gids, vals.to(st["dtype"]))
        if not st["printed"]:
            st["printed"] = True
            print("[l4port] drafter head TP: first call, %d rows, candidates per row %d" % (m, vals.shape[1]), flush=True)
        return out

    compute_logits.__wrapped__ = orig
    cls.compute_logits = compute_logits


def _patch_mr(module):
    cls = module.GPUModelRunner
    if getattr(cls, "_l4_dheadtp_mr", False):
        return
    cls._l4_dheadtp_mr = True
    orig_load = cls.load_model

    def load_model(self, *a, **kw):
        out = orig_load(self, *a, **kw)
        dm = getattr(getattr(self, "speculator", None), "model", None)
        if dm is not None and hasattr(dm, "compute_logits") and hasattr(dm, "lm_head") and hasattr(dm.lm_head, "shard_indices"):
            tm = self.model.get_language_model() if hasattr(self.model, "get_language_model") else self.model
            _prepare(dm, getattr(tm, "lm_head", None))
            _patch_model(type(dm))
            print("[l4port] drafter head TP: patched %s.compute_logits" % type(dm).__name__, flush=True)
        return out

    load_model.__wrapped__ = orig_load
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
                        _patch_mr(module)

                    spec.loader.exec_module = exec_module
                    return spec
            return None
        finally:
            _BUSY.discard(fullname)


def install():
    if os.environ.get("ARC3_L4PORT_DHEAD_TP") != "1":
        return
    if "drafthead" in os.environ.get("ARC3_STAGE2", "").split(","):
        raise RuntimeError("ARC3_L4PORT_DHEAD_TP=1 with the 'drafthead' part")
    assert os.path.isfile(os.environ.get("ARC3_S2_DRAFTHEAD_VOCAB", "")), "ARC3_S2_DRAFTHEAD_VOCAB"
    if MR in sys.modules:
        _patch_mr(sys.modules[MR])
    else:
        sys.meta_path.insert(0, _Finder())
