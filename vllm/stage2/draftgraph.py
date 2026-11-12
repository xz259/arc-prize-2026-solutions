"""stage2_v1 'draftgraph' part (measurement): a runtime switch for the MTP drafter's decode CUDA graphs above 16 requests
(GAP_PLAN item 3, in-run A/B, 2026-09-30).

The drafter's decode graph manager (SpeculatorCudaGraphManager with decode_query_len 1) captures one token per request
and keeps sizes up to max_num_reqs. With the production-derived capture list (4, 8, 16, 32, ...) its only sizes are 4,
8 and 16, so 17-30 lanes run both draft decode passes eagerly; capturing 20, 24, 25 and 30 gives them graphs. With
those sizes captured, ARC3_S2_DRAFTGRAPH_SWITCH=<file> containing 'eager' makes that manager's dispatch return the
eager descriptor for more than 16 tokens: exactly the production list's behaviour for the drafter (same kernels,
eager launches), switchable within one run. Any other content, or no file, leaves dispatch unchanged. The target's and
the draft prefill's managers are untouched."""
import os
import time

PINS = {'vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils':
        '13392ef0a9ed59eb9c2b2bad43d7c61beb212a8805849e46b5130c230275d0a5'}
STATS = dict(dispatches=0, eager_forced=0)
_SWITCH = dict(path=os.environ.get('ARC3_S2_DRAFTGRAPH_SWITCH'), t=0.0, eager=False)
LIMIT = 16


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
    cls = mod.SpeculatorCudaGraphManager
    from vllm.config.compilation import CUDAGraphMode
    from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor
    original = cls.dispatch               # inherited CudaGraphManager.dispatch

    def dispatch(self, num_reqs, num_tokens, uniform_token_count, num_active_loras, max_query_len=None):
        desc = original(self, num_reqs, num_tokens, uniform_token_count, num_active_loras, max_query_len)
        if self.decode_query_len == 1:
            STATS['dispatches'] += 1
            if num_tokens > LIMIT and desc.cg_mode != CUDAGraphMode.NONE and _eager():
                STATS['eager_forced'] += 1
                return BatchExecutionDescriptor(cg_mode=CUDAGraphMode.NONE, num_tokens=num_tokens, num_reqs=num_reqs,
                                                num_active_loras=self._resolve_effective_loras(num_active_loras))
        return desc

    cls.dispatch = dispatch


PATCHERS = {'vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils': patch_cudagraph_utils}
