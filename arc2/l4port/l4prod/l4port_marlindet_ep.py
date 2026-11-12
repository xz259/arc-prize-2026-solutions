"""L4 port (2026-11-01, ARC2 - main; the reviewer's diagnosis of the PP path's non-bitwise reruns; the user: fix it): deterministic
Marlin MoE under expert parallelism, env ARC3_L4PORT_MARLINDET_EP=1 (unset: unchanged).
stage2_v1/marlindet.py (production, 2026-10-01) rewrites each expert segment of vLLM's moe_align_block_size in ascending flat
(token, slot) order: the CUDA op places entries by atomics, so their order changes from call to call, and Marlin's result for a row
depends on its position in the segment (one bf16 ulp; split-K stripes), which the NVFP4 KV rounding and the QSA top-k amplify. It
passes expert-parallel calls (expert_map given) through unchanged, and the L4 launcher runs --enable-expert-parallel at TP > 1, so
on TP2 x PP2 every MoE call keeps the atomics order (receipts: marlindet calls = passthrough, 72 / 72 and 56 / 56 on pbq1's workers).
Marlin MoE calls moe_align_block_size(topk_ids, block, global_num_experts, expert_map, ignore_invalid_experts=True)
(experts/marlin_moe.py:340): the CUDA op maps each entry through expert_map, skips the -1 (non-local) entries and lays out one
segment per local expert, valid entries first, then the padding sentinels; expert_ids hold local ids. This hook gives that call
marlindet's canonical order with marlindet's two kernels (segment starts from expert_ids; one program per expert writing its
entries in ascending flat index), each entry's expert taken through expert_map (marlindet_ep_kernels.py). The blocks,
expert_ids, padding and per-segment entry sets are unchanged; no host sync (CUDA-graph safe). Every other call goes to the
previous function (marlindet's canonical, or the stock align). Post-import hook on
vllm.model_executor.layers.fused_moe.experts.marlin_moe; correct whether it is installed before or after marlindet's own patch.
Test: l4prod/tests/t_marlindet_ep.py. Receipts: one install line and one first-call line per process; STATS."""
import importlib.abc
import importlib.util
import os
import sys

NAME = "vllm.model_executor.layers.fused_moe.experts.marlin_moe"
ALIGN = "vllm.model_executor.layers.fused_moe.moe_align_block_size"
HERE = os.path.dirname(os.path.abspath(__file__))
STATS = dict(installed=0, calls=0, canonical_ep=0, delegated=0)
_PREV = [None]
_STOCK = [None]
_K = [None]


def _kernels():
    if _K[0] is None:
        spec = importlib.util.spec_from_file_location("l4port_marlindet_ep_kernels", os.path.join(HERE, "marlindet_ep_kernels.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _K[0] = mod
    return _K[0]


def canonical_ep(topk_ids, block_size, num_experts, expert_map=None, pad_sorted_ids=False, ignore_invalid_experts=False):
    STATS["calls"] += 1                                  # Python-level: eager calls and graph captures
    if expert_map is None or not ignore_invalid_experts:
        STATS["delegated"] += 1
        return _PREV[0](topk_ids, block_size, num_experts, expert_map, pad_sorted_ids, ignore_invalid_experts)
    import torch
    import triton
    s, e, n = _STOCK[0](topk_ids, block_size, num_experts, expert_map, pad_sorted_ids, ignore_invalid_experts)
    flat = topk_ids.reshape(-1)
    assert flat.dtype == torch.int32 and flat.is_contiguous() and s.dtype == torch.int32 and e.dtype == torch.int32
    assert expert_map.dtype == torch.int32 and expert_map.is_contiguous() and expert_map.numel() == num_experts
    k = _kernels()
    seg = torch.empty(num_experts, dtype=torch.int32, device=s.device)
    nb = e.numel()
    k.segment_starts[(triton.cdiv(nb, 256),)](e, n, seg, nb, block_size, BLOCK=256)
    numel = flat.numel()
    k.place_mapped[(num_experts,)](flat, expert_map, seg, s, numel, num_experts, BLOCK=1024 if numel <= 8192 else 4096)
    if not STATS["canonical_ep"]:
        print("[l4port] marlindet-ep: first expert-parallel align canonicalized (experts %d, entries %d, pid %d)"
              % (num_experts, numel, os.getpid()), flush=True)
    STATS["canonical_ep"] += 1
    return s, e, n


def _patch(module):
    import importlib
    _PREV[0] = module.moe_align_block_size
    _STOCK[0] = importlib.import_module(ALIGN).moe_align_block_size
    canonical_ep.__wrapped__ = _PREV[0]
    module.moe_align_block_size = canonical_ep
    STATS["installed"] += 1
    print("[l4port] marlindet-ep: installed over %s.%s (pid %d)" % (getattr(_PREV[0], "__module__", "?"),
                                                                     getattr(_PREV[0], "__name__", "?"), os.getpid()), flush=True)


_BUSY = set()


class _Finder(importlib.abc.MetaPathFinder):
    # v2 (2026-11-01): a re-entrancy guard. Several L4 hooks put a finder for the same module (ops.qsa: qsaws and qsasort) on
    # sys.meta_path, each delegating to every other finder, so without the guard they recursed into each other (pbl1: RecursionError
    # at startup; the l4port_ppdraftbc / ppbcast precedent on the 4090's c24a)
    def find_spec(self, fullname, path=None, target=None):
        if fullname != NAME or fullname in _BUSY:
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
    if os.environ.get("ARC3_L4PORT_MARLINDET_EP") != "1":
        return
    if NAME in sys.modules:
        _patch(sys.modules[NAME])
    else:
        sys.meta_path.insert(0, _Finder())
