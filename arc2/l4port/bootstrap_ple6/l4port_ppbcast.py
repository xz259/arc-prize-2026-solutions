"""L4 port (2026-10-30, ARC2 - main): the PP sampled-token broadcast under speculative decoding, env ARC3_L4PORT_PP_BCASTPAD=1.
MRv2's PPHandler (v1/worker/gpu/pp_utils.py) receives on every non-last stage into [num_reqs, num_speculative_steps + 1], but the
last stage broadcasts sampler_output.sampled_token_ids as produced: the rejection sampler's [num_reqs, K + 1] on verify steps, the
plain Sampler's sampled.view(-1, 1) = [num_reqs, 1] on steps without drafts (prefill; sample/sampler.py:172). The NCCL broadcast
counts then differ, the root returns and the receivers wait forever (4090 c19a / c20a: stage 0 spinning in the pp_broadcast
BROADCAST after the warmup prefill while the last stage moved on). The wrapper pads the broadcast tensor to max_sample_len columns
with -1 before the stock broadcast; receivers read only num_sampled tokens per row (post_update), so the padding is never used.
Eager (sample_tokens runs outside CUDA graphs). Post-import hook on vllm.v1.worker.gpu.pp_utils; one receipt line."""
import importlib.abc
import os
import sys

NAME = "vllm.v1.worker.gpu.pp_utils"
STATE = dict(printed=False, padded=0)


def _patch(module):
    cls = module.PPHandler
    orig = cls.broadcast

    def broadcast(self, sampled_token_ids, num_sampled, num_rejected, input_batch):
        import torch
        w = sampled_token_ids.shape[-1] if sampled_token_ids.dim() == 2 else None
        if w is not None and w < self.max_sample_len:
            pad = torch.full((sampled_token_ids.shape[0], self.max_sample_len - w), -1,
                             dtype=sampled_token_ids.dtype, device=sampled_token_ids.device)
            sampled_token_ids = torch.cat([sampled_token_ids, pad], dim=1)
            STATE["padded"] += 1
            if not STATE["printed"]:
                STATE["printed"] = True
                print("[l4port] PP broadcast: sampled tokens padded from %d to %d columns (rows %d)"
                      % (w, self.max_sample_len, sampled_token_ids.shape[0]), flush=True)
        return orig(self, sampled_token_ids, num_sampled, num_rejected, input_batch)

    broadcast.__wrapped__ = orig
    cls.broadcast = broadcast


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
    if os.environ.get("ARC3_L4PORT_PP_BCASTPAD") != "1":
        return
    if NAME in sys.modules:
        _patch(sys.modules[NAME])
    else:
        sys.meta_path.insert(0, _Finder())
