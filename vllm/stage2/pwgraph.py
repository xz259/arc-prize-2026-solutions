"""stage2_v1 'pwgraph' part (measurement): a runtime switch for the target model's PIECEWISE CUDA graphs above a token
limit (2026-10-01, concurrency trade-off: Kaggle goalpost, GAP_PLAN).

The PIECEWISE graphs for mixed steps of 160-1,024 tokens (item 2) cost about 0.6 GiB, about 1.8 lanes of KV at 35,484.
To price them at high concurrency within one run, ARC3_S2_PWGRAPH_SWITCH=<file> containing 'eager' makes the target's
graph manager (ModelCudaGraphManager) return the eager descriptor for PIECEWISE dispatches above
ARC3_S2_PWGRAPH_LIMIT tokens (the largest decode size of the list): exactly the behaviour of a capture list without
those graphs (same compiled pieces, eager launches). FULL decode graphs, smaller PIECEWISE graphs and the drafter's
managers are untouched. Any other content, or no file, leaves dispatch unchanged."""
import os
import time

PINS = {'vllm.v1.worker.gpu.cudagraph_utils': 'ee1f6eb37bc2f456a3e7e9142d5a53455afed2250a57f52780546cf1aef27e2c'}
STATS = dict(dispatches=0, piecewise_above=0, eager_forced=0)
_SWITCH = dict(path=os.environ.get('ARC3_S2_PWGRAPH_SWITCH'), t=0.0, eager=False)
LIMIT = int(os.environ.get('ARC3_S2_PWGRAPH_LIMIT', '120'))


def _eager():
    s = _SWITCH
    if s['path'] is None:
        return False
    now = time.monotonic()
    if now - s['t'] > 1.0:
        s['t'] = now
        try:
            with open(s['path']) as f:
                s['eager'] = f.read().strip() == 'eager'
        except OSError:
            s['eager'] = False
    return s['eager']


def patch_cudagraph_utils(mod):
    cls = mod.ModelCudaGraphManager
    from vllm.config.compilation import CUDAGraphMode
    original = mod.CudaGraphManager.dispatch          # inherited by the model manager; the subclass gets its own

    def dispatch(self, num_reqs, num_tokens, uniform_token_count, num_active_loras, max_query_len=None):
        desc = original(self, num_reqs, num_tokens, uniform_token_count, num_active_loras, max_query_len)
        STATS['dispatches'] += 1
        if desc.cg_mode == CUDAGraphMode.PIECEWISE and num_tokens > LIMIT:
            STATS['piecewise_above'] += 1
            if _eager():
                STATS['eager_forced'] += 1
                return mod.BatchExecutionDescriptor(cg_mode=CUDAGraphMode.NONE, num_tokens=num_tokens, num_reqs=num_reqs,
                                                    num_active_loras=self._resolve_effective_loras(num_active_loras))
        return desc

    cls.dispatch = dispatch


PATCHERS = {'vllm.v1.worker.gpu.cudagraph_utils': patch_cudagraph_utils}
