"""L4 port (2026-10-30, ARC2 - main): draft tokens on the non-last PP stages, env ARC3_L4PORT_PP_DRAFTSYNC=1.
MRv2 builds a verify step's input ids on every rank from the GPU tensor req_states.draft_tokens (combine_sampled_and_draft_tokens
in GPUModelRunner.prepare_inputs), but only the last PP rank writes it (the speculator's propose, model_runner.py ~1907); on the
other stages it stays at its zero initialisation, so stage 0 embedded token 0 at every draft position. The target then saw the
right prefix only up to the first draft position: the 4090's c22a (PP + eprune 64 + MTP K 3) accepted 0.262 per draft, tau 1.787
= 1 + the first position's usual acceptance, against 3.2-3.5 at TP=4. The wrapper writes the scheduler's draft ids
(scheduler_output.scheduled_spec_decode_tokens, the values the last rank proposed and the engine scheduled) into
req_states.draft_tokens for the scheduled requests on non-last PP ranks before the stock prepare_inputs (same stream, so the
copy lands before the input ids are combined); negative placeholders are clamped to 0. PP 1 and the last rank are unchanged.
Post-import hook on vllm.v1.worker.gpu.model_runner; one receipt line per process."""
import importlib.abc
import os
import sys

NAME = "vllm.v1.worker.gpu.model_runner"
STATE = dict(printed=False, steps=0, rows=0)


def _patch(module):
    cls = module.GPUModelRunner
    orig = cls.prepare_inputs

    def prepare_inputs(self, scheduler_output, batch_req_state, batch_desc):
        k = getattr(self, "num_speculative_steps", 0)
        drafts = getattr(scheduler_output, "scheduled_spec_decode_tokens", None)
        if k > 0 and drafts and not self.is_last_pp_rank:
            import torch
            idx, rows = [], []
            r2i = self.req_states.req_id_to_index
            for req_id, toks in drafts.items():
                i = r2i.get(req_id)
                if i is None or not toks:
                    continue
                row = [max(0, int(t)) for t in toks[:k]]
                rows.append(row + [0] * (k - len(row)))
                idx.append(i)
            if idx:
                vals = torch.tensor(rows, dtype=torch.int64).pin_memory().to(self.device, non_blocking=True)
                where = torch.tensor(idx, dtype=torch.int64).pin_memory().to(self.device, non_blocking=True)
                self.req_states.draft_tokens.index_copy_(0, where, vals)
                STATE["steps"] += 1
                STATE["rows"] += len(idx)
                if not STATE["printed"]:
                    STATE["printed"] = True
                    print("[l4port] PP drafts: non-last rank writes the scheduled draft ids into req_states.draft_tokens "
                          "(first step: %d requests, K %d)" % (len(idx), k), flush=True)
        return orig(self, scheduler_output, batch_req_state, batch_desc)

    prepare_inputs.__wrapped__ = orig
    cls.prepare_inputs = prepare_inputs


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
    if os.environ.get("ARC3_L4PORT_PP_DRAFTSYNC") != "1":
        return
    if NAME in sys.modules:
        _patch(sys.modules[NAME])
    else:
        sys.meta_path.insert(0, _Finder())
