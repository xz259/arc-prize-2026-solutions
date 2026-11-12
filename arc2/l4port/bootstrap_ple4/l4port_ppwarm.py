"""L4 port (2026-10-30, ARC2 - main): MRv2's post-capture warmup under pipeline parallelism with a drafter, env
ARC3_L4PORT_PP_WARMCAP=N (N >= 2; default off). warmup_kernels (v1/worker/gpu/warmup.py) sizes its batch as
min(max_num_seqs, max_num_batched_tokens // prompt_len, (num_blocks - 1) // max_blocks_per_req), and max_blocks_per_req sums the
LOCAL KV cache groups. With the MTP drafter only on the last stage, the stages can derive different request counts, so their
collectives stop matching: on the 4090 (c19a) stage 0 waited 600 s in the pp_broadcast NCCL broadcast of the sampled tokens
(64 elements = 16 requests x 4) while the last stage had already moved on to the next decode step. The wrapper sets
scheduler_config.max_num_seqs to min(max_num_seqs, N) for the duration of the warmup on PP > 1 (restored after), so every stage
takes the same count (the other two terms are far larger at N = 4). Kernels whose shapes the smaller warmup skips compile at
first use. PP 1 is unchanged. Post-import hook on vllm.v1.worker.gpu_worker (the name it imported); one receipt line."""
import importlib.abc
import os
import sys

NAME = "vllm.v1.worker.gpu_worker"
CAP = int(os.environ.get("ARC3_L4PORT_PP_WARMCAP", "0") or 0)


def _patch(module):
    orig = module.warmup_kernels

    def warmup_kernels(model_runner, worker_execute_model, worker_sample_tokens):
        from vllm.distributed import get_pp_group
        g = get_pp_group()
        if g.world_size == 1:
            return orig(model_runner, worker_execute_model, worker_sample_tokens)
        sc = model_runner.scheduler_config
        old = sc.max_num_seqs
        sc.max_num_seqs = max(2, min(old, CAP))
        print("[l4port] PP warmup: max_num_seqs %d -> %d during warmup_kernels (pp rank %d of %d)"
              % (old, sc.max_num_seqs, g.rank_in_group, g.world_size), flush=True)
        try:
            return orig(model_runner, worker_execute_model, worker_sample_tokens)
        finally:
            sc.max_num_seqs = old

    warmup_kernels.__wrapped__ = orig
    module.warmup_kernels = warmup_kernels


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
    if CAP < 2:
        return
    if NAME in sys.modules:
        _patch(sys.modules[NAME])
    else:
        sys.meta_path.insert(0, _Finder())
