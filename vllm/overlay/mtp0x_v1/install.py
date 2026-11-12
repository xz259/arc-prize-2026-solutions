"""mtp0x_v1 install (ARC3_MTP0X = comma list of parts):

fp8w     Tier-A dense layers and lm_head keep the adapter's per-channel E4M3 quantization but stay in the plain
         row-major layout (no Marlin repack) and run through the fp8w Triton kernel (deterministic split-K) with the
         merged per-shape, per-row-count config table. Installed by replacing dense_fp8_adapter_v1.runtime's
         quantize_layer and its dispatch operator; the adapter's receipts keep being written (readiness gate).
sampler  MRv2 Sampler.sample: after the stock apply_sampling_params, eligible batches (no greedy row, every row with
         top_k <= 31, no processed logprobs or sampling mask) sample through the candidate decode sampler instead of
         FlashInfer / the Triton top-k-top-p + Gumbel path. Anything else runs the stock method.

Pins: the adapter runtime and the MRv2 sampler module (sources of 2026-09-29). Receipts: $ARC3_MTP0X_RECEIPTS."""
import hashlib
import importlib.abc
import json
import os
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PARTS = frozenset(p for p in os.environ.get('ARC3_MTP0X', '0').split(',') if p and p != '0')
ADAPTER_PIN = 'acd782838a74189a5ec8f6f220a62d0d83b82b9f33d95c50ccb91013fa51bbed'
SAMPLER_MODULE = 'vllm.v1.worker.gpu.sample.sampler'
SAMPLER_PIN = '5c06d57bf7df726146e385e678f8350bd2918fa29c7cfc233f12b81ba3e49f34'
REPORT = dict(schema='arc3.mtp0x.v1', pid=os.getpid(), parts=sorted(PARTS), fp8w_layers={}, fp8w_shapes={},
              sampler_installed=False, sampler_calls=0, sampler_fallbacks=0, sampler_fallback_reasons={},
              errors=[], updated=None)
_last_save = [0.0]


def save(force=False):
    root = os.environ.get('ARC3_MTP0X_RECEIPTS')
    if not root or (not force and time.time() - _last_save[0] < 30):
        return
    _last_save[0] = time.time()
    p = Path(root); p.mkdir(parents=True, exist_ok=True)
    REPORT['updated'] = time.time()
    dst = p / f'proc-{os.getpid()}.json'; tmp = dst.with_suffix('.tmp')
    tmp.write_text(json.dumps(REPORT, indent=1) + '\n'); tmp.replace(dst)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ---------------------------------------------------------------- fp8w dense
FP8W = dict(layers={}, op=None, ws=None, table=None)


def _config(n, k, m):
    import bisect
    rows = FP8W['table'][(n, k)]
    i = bisect.bisect_left([r[0] for r in rows], m)
    return tuple(rows[min(i, len(rows) - 1)][1])


def _workspace_need(n, k, max_m=8192):
    import triton
    elems = tiles = 1
    rows = FP8W['table'][(n, k)]
    for i, (M, cfg) in enumerate(rows):
        bm, bn, bk, sk = cfg[:4]
        m = max(M, max_m) if i == len(rows) - 1 else M
        nm, nn = triton.cdiv(m, bm), triton.cdiv(n, bn)
        tiles = max(tiles, nm * nn)
        if sk > 1:
            elems = max(elems, nm * nn * sk * bm * bn)
    return elems, tiles


def _register_op():
    import torch
    from .fp8w_kernels import fp8w_linear

    @torch.library.custom_op('arc3_mtp0x::fp8w_linear', mutates_args=())
    def fp8w_op(x: torch.Tensor, prefix: str) -> torch.Tensor:
        st = FP8W['layers'][prefix]
        lead = x.shape[:-1]
        x2 = x.reshape(-1, x.shape[-1])
        if x2.stride(-1) != 1:                                     # the kernel takes any row stride
            x2 = x2.contiguous()
        m = x2.shape[0]
        if m == 0:
            return x.new_empty((*lead, st['n']))
        y = fp8w_linear(x2, st['q'], st['scale'], _config(st['n'], st['k'], m), FP8W['ws'])
        return y.reshape(*lead, st['n'])

    @fp8w_op.register_fake
    def _fake(x, prefix):
        return x.new_empty((*x.shape[:-1], FP8W['layers'][prefix]['n']))

    return fp8w_op


def install_fp8w():
    import dense_fp8_adapter_v1.runtime as rt
    assert sha(rt.__file__) == ADAPTER_PIN, 'mtp0x: dense_fp8_adapter_v1/runtime.py changed; re-pin'
    table = json.loads((HERE / 'fp8w_table.json').read_text())
    FP8W['table'] = {tuple(int(v) for v in key.split(',')): rows for key, rows in table.items()}

    def quantize_layer(layer, kind):
        import torch
        from vllm.model_executor.utils import replace_parameter
        from .fp8w_kernels import Workspace
        prefix = layer.prefix
        w = layer.weight
        assert w.dtype in (torch.bfloat16, torch.float16) and w.dim() == 2, (prefix, w.dtype, tuple(w.shape))
        assert getattr(layer, 'tp_size', 1) == 1, prefix
        n, k = w.shape
        assert (n, k) in FP8W['table'], ('mtp0x: no fp8w configs for shape', prefix, n, k)
        q, scale, recon = rt.quantize_per_channel(w.data)          # the adapter's own quantization
        q = q.contiguous(); scale = scale.to(torch.float32).contiguous()
        if FP8W['op'] is None:
            FP8W['op'] = _register_op()
            FP8W['ws'] = Workspace(q.device)
        rt.OP = FP8W['op']                                         # the adapter's apply dispatches through rt.OP
        FP8W['layers'][prefix] = dict(q=q, scale=scale, n=n, k=k)
        elems, tiles = _workspace_need(n, k)
        FP8W['ws'].get(None, elems, tiles)                         # sized before profiling and graph capture
        bytes_before = w.numel() * w.element_size()
        bytes_after = q.numel() * q.element_size() + scale.numel() * 4
        rt.LAYERS[prefix] = dict(n=n, k=k, dtype=w.dtype, tier=kind, kernel='fp8w')
        layer.arc3_fp8_qweight = q; layer.arc3_fp8_scales = scale
        replace_parameter(layer, 'weight', q)
        del w
        torch.cuda.empty_cache()
        rt.REPORT['layers'][prefix] = dict(tier=kind, shape=[n, k], bytes_bf16=bytes_before, bytes_fp8=bytes_after,
                                           weight_recon_rel_rms=recon, packed_shape=list(q.shape),
                                           packed_dtype=str(q.dtype), kernel='fp8w')
        rt.REPORT['bytes_bf16_released'] += bytes_before; rt.REPORT['bytes_fp8_added'] += bytes_after
        rt.save()
        REPORT['fp8w_layers'][prefix] = [n, k]
        REPORT['fp8w_shapes'][f'{n},{k}'] = REPORT['fp8w_shapes'].get(f'{n},{k}', 0) + 1
        REPORT['fp8w_workspace'] = dict(partial_floats=FP8W['ws'].p.numel(), counters=FP8W['ws'].cnt.numel())
        save(force=True)

    rt.quantize_layer = quantize_layer
    REPORT['fp8w_installed'] = True
    save(force=True)


# ---------------------------------------------------------------- candidate decode sampler
def patch_sampler(module):
    import numpy as np
    import torch
    from vllm.config.model import PROCESSED_LOGPROBS_MODES
    from .cand_decode import DecodeSampler, KMAX
    cls = module.Sampler
    stock_sample = cls.sample
    state = dict(decoder=None, lock=threading.Lock())

    def fallback(reason):
        REPORT['sampler_fallbacks'] += 1
        REPORT['sampler_fallback_reasons'][reason] = REPORT['sampler_fallback_reasons'].get(reason, 0) + 1
        save()

    def sample(self, logits, expanded_idx_mapping, idx_mapping, idx_mapping_np, pos, input_ids, expanded_local_pos,
               return_logprobs=False):
        st = self.sampling_states
        if return_logprobs and self.logprobs_mode in PROCESSED_LOGPROBS_MODES:
            fallback('processed_logprobs')
        elif self.return_sampling_mask:
            fallback('sampling_mask')
        elif st.any_greedy(idx_mapping_np):
            fallback('greedy')
        elif not bool(np.all(st.top_k.np[idx_mapping_np] <= KMAX - 1)):
            fallback('top_k')
        else:
            processed = self.apply_sampling_params(logits, expanded_idx_mapping, idx_mapping, idx_mapping_np, pos,
                                                   input_ids, expanded_local_pos, skip_top_k_top_p=True)
            top_k, top_p = st.get_top_k_top_p(expanded_idx_mapping, idx_mapping_np)
            if processed.dtype != torch.float32:
                processed = processed.float()
            if state['decoder'] is None or state['decoder'].ckey.numel() < processed.shape[0] * state['decoder'].nb * KMAX:
                with state['lock']:
                    state['decoder'] = DecodeSampler(max(1024, processed.shape[0]), processed.shape[1],
                                                     processed.device)
            sampled = state['decoder'](processed, top_k, top_p, expanded_idx_mapping, st.seeds.gpu, pos)
            REPORT['sampler_calls'] += 1
            save()
            return sampled, processed
        return stock_sample(self, logits, expanded_idx_mapping, idx_mapping, idx_mapping_np, pos, input_ids,
                            expanded_local_pos, return_logprobs=return_logprobs)

    cls.sample = sample
    REPORT['sampler_installed'] = True
    save(force=True)


class _Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        m = getattr(self.original, 'create_module', None)
        return m(spec) if m else None

    def exec_module(self, module):
        assert sha(module.__spec__.origin) == SAMPLER_PIN, 'mtp0x: MRv2 sampler module changed; re-pin'
        self.original.exec_module(module)
        patch_sampler(module)

    def __getattr__(self, name):
        return getattr(self.original, name)


class _Finder(importlib.abc.MetaPathFinder):
    def __init__(self):
        self._local = threading.local()

    def find_spec(self, fullname, path=None, target=None):
        if fullname != SAMPLER_MODULE:
            return None
        active = getattr(self._local, 'active', None)
        if active is None:
            active = self._local.active = set()
        if fullname in active:
            return None
        active.add(fullname)
        try:
            started = False
            for finder in list(sys.meta_path):
                if finder is self:
                    started = True
                    continue
                if not started or not hasattr(finder, 'find_spec'):
                    continue
                spec = finder.find_spec(fullname, path, target)
                if spec is not None:
                    spec.loader = _Loader(spec.loader)
                    return spec
            return None
        finally:
            active.discard(fullname)


def install():
    assert PARTS <= {'fp8w', 'sampler'}, PARTS
    try:
        if 'fp8w' in PARTS:
            install_fp8w()
        if 'sampler' in PARTS:
            assert SAMPLER_MODULE not in sys.modules, 'mtp0x: sampler module imported before the hook'
            if not any(isinstance(f, _Finder) for f in sys.meta_path):
                sys.meta_path.insert(0, _Finder())
    except Exception as exc:
        REPORT['errors'].append(repr(exc)); save(force=True)
        raise
    save(force=True)
