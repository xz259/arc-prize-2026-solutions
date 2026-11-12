"""L4 port (2026-10-30; throughput): vocab-parallel top-K logits gather at TP > 1, env ARC3_L4PORT_VPTOPK=K (e.g. 20).

The stock LogitsProcessor._get_logits all-gathers every rank's [M, V / tp] logits shard into [M, V] BF16: ~0.5 MB per row at
V = 248,320, so ~50 MB for an MTP verify step at 20 lanes x 5 tokens and ~10 MB per draft step, over Kaggle's host-memory NCCL
(~4.7 GB/s). When K >= every request's top_k, the sampled distribution depends only on the global top-K logits, which lie in
the union of the ranks' local top-K. So each rank all-gathers its local top-K (values and gathered-layout positions, [M, 2K]
FP32), and the [M, V] tensor is rebuilt on every rank: -inf everywhere except the tp x K candidates, which hold their exact
BF16 values. Temperature, top-k <= K, top-p, min-p 0, the rejection sampler's target probabilities and the random draws (shape
[M, V], independent of the values) then see the same numbers as the stock path. The drafter shares the head, so its draft
distribution is restricted to the candidates; the drafts are verified against that same distribution, so the output stays
exact and only acceptance can move.
NOT exact for top_k > K or top_k = -1, for logprobs / prompt_logprobs (non-candidates read -inf) and for penalties that can
raise a non-candidate (repetition_penalty < 1). The harness samples T 0.6, top-k 20, top-p 0.95, no logprobs.
The patch is a post-import hook on vllm.model_executor.layers.logits_processor (no source edit); one receipt line per
process at the first patched call.
Exactness check (ARC3_L4PORT_VPTOPK_CHECK=N, default 0; 2026-10-30, after the cross-server sample probe proved too noisy: separate
launches are not bitwise identical, so sampled texts diverge either way): for the first N patched calls the stock all-gather runs
too, and the stock tensor's top-K positions must hold the same values in the rebuilt tensor and the rebuilt tensor's top-K values
must equal the stock top-K values (sorted). One receipt line per process when the N calls are done (mismatching rows, max |dv|)."""
import importlib.abc
import os
import sys

K = int(os.environ.get("ARC3_L4PORT_VPTOPK", "0") or 0)
CHECK = int(os.environ.get("ARC3_L4PORT_VPTOPK_CHECK", "0") or 0)
STATE = dict(printed=False, checks=0, rows=0, bad_rows=0, max_dv=0.0, reported=False)
NAME = "vllm.model_executor.layers.logits_processor"


def _patch(module):
    cls = module.LogitsProcessor
    orig = cls._get_logits

    def _get_logits(self, hidden_states, lm_head, embedding_bias):
        tp = lm_head.tp_size
        if K <= 0 or tp <= 1:
            return orig(self, hidden_states, lm_head, embedding_bias)
        import torch
        from vllm.distributed import get_tensor_model_parallel_rank, tensor_model_parallel_all_gather
        logits = self._apply_head(lm_head, hidden_states, embedding_bias)      # [M, Vl]: this rank's padded shard
        m, vl = logits.shape[0], logits.shape[-1]
        rank = get_tensor_model_parallel_rank()
        si = lm_head.shard_indices
        n_org = si.org_vocab_end_index - si.org_vocab_start_index              # real entries lead the local shard
        loc = logits.float()
        if n_org < vl:
            loc[..., n_org:] = float("-inf")                                    # padding / added vocab never sampled
        vals, idx = torch.topk(loc, K, dim=-1)
        pair = torch.cat([vals, (idx + rank * vl).float()], dim=-1)             # gathered layout: rank r's shard at r * Vl
        g = tensor_model_parallel_all_gather(pair, dim=-1).view(m, tp, 2, K)    # [M, tp, 2, K]
        cv = g[:, :, 0, :].reshape(m, tp * K)
        cp = g[:, :, 1, :].reshape(m, tp * K).long()
        out = torch.full((m, vl * tp), float("-inf"), dtype=logits.dtype, device=logits.device)
        out.scatter_(1, cp, cv.to(logits.dtype))
        if STATE["checks"] < CHECK and not torch.cuda.is_current_stream_capturing():   # exactness check (eager calls only)
            STATE["checks"] += 1
            full = self._gather_logits(logits)[..., : self.org_vocab_size].float()
            sp = out[..., : self.org_vocab_size].float()
            fv, fi = torch.topk(full, K, dim=-1)
            sv, _ = torch.topk(sp, K, dim=-1)
            at = sp.gather(1, fi)
            bad = ((at != fv).any(dim=-1) | (sv != fv).any(dim=-1))
            STATE["rows"] += m
            STATE["bad_rows"] += int(bad.sum())
            STATE["max_dv"] = max(STATE["max_dv"], float((at - fv).abs().max()), float((sv - fv).abs().max()))
            if STATE["checks"] == CHECK and not STATE["reported"]:
                STATE["reported"] = True
                print("[l4port] vptopk check: rank %d, %d calls, %d rows, top-%d mismatching rows %d, max |dv| %g"
                      % (rank, STATE["checks"], STATE["rows"], K, STATE["bad_rows"], STATE["max_dv"]), flush=True)
        if not STATE["printed"]:
            STATE["printed"] = True
            print("[l4port] vptopk: rank %d tp %d K %d shard %d (org %d), rows %d: candidates gathered instead of %d logits per row"
                  % (rank, tp, K, vl, n_org, m, vl * tp), flush=True)
        return out[..., : self.org_vocab_size]

    _get_logits.__wrapped__ = orig
    cls._get_logits = _get_logits


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
    if K <= 0:
        return
    if NAME in sys.modules:
        _patch(sys.modules[NAME])
    else:
        sys.meta_path.insert(0, _Finder())
