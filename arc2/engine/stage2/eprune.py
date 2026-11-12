"""stage2_v1 'eprune' part (2026-10-26, Claude fork bdfc5402; the user: "Since no pruning needs to be done on the shard level, we will
make this a flag if we end up adopting it", "Let's go straight to step 2 and 3"): the routed experts listed in ARC3_S2_EPRUNE_FILE
(xprune's format: {'target:<MoE runner module>': [expert ids]}; a bare file name is read from this directory) are removed from the
target model at load time; unset, the file is eprune_k16.json here (the bottom-16 set per layer of EXPERT_USAGE_2026-10-26.md, xprune's
prune_k16.json), so a KDEF profile selects pruning by its part list alone. The checkpoint is unchanged; the drafter is unchanged.
  build   each listed layer's FusedMoE is built with E - n experts, so only the kept experts' weights are allocated (the bottom-16
          set: 16 x 2.55 MB per layer, 1.83 GiB over the 48 layers). The router gate keeps its E rows.
  load    the weight stream is rewritten before the model's loader sees it: kept expert e becomes local expert rank(e) (the kept ids
          in ascending order), the pruned experts' tensors are dropped, and the gate's rows are permuted to [kept..., pruned...].
  route   the last n gate logits (the pruned experts) are set to -inf in place before the router's top-k, so vLLM's E-wide softmax
          top-k kernel (its 512 specialisation) returns local ids directly. Against xprune's mask (GPU test t_router_perm.py): the
          kept logits are bitwise equal, the ids are equal, the weights agree within 2.4e-7 (the softmax sums in another column
          order); the fill is 1.2 us per layer. Rejected: a 496-row gate (vLLM's top-k has no 496 kernel: 12 vs 5 us per layer at
          decode, 69 vs 10 us at 4,096 tokens) and a fused -inf gate bias (cuBLAS then picks other GEMM algorithms: logits move by
          bf16 ulps and near-tie top-k picks flip).
Checked at load (raises otherwise): every listed layer was built with E - n experts and is found as a module after loading, its gate
rows were permuted once, every kept expert received the same number of tensors (> 0), every pruned expert's tensors were dropped,
and the routed-expert parameters have E - n rows. A load that sees none of the listed layers' tensors (the PLE offload worker reads
only the n-gram tables; 20:31 on 10-26 the first version's check failed there and stopped the engine) is counted, not checked; the GPU
worker's load must have been checked before the routing attaches. Not combinable with xprune (the mask).
IE's review (2026-10-27 03:30:36 UTC, V9_TIERS_REVIEW_ie_2026-10-27.md, should-fix 3 / 4): the attach also refuses a monolithic MoE
kernel (it routes inside the kernel from the full logits, so the -inf fill would be bypassed) and a gate fused with the shared-expert
gate (its logit column sits after the pruned tail and would be filled too); the rewrite refuses any other mlp.gate.* tensor of a
listed layer (only gate.weight is permuted). STATS records this file's and the prune file's sha256 (first 16 hex digits)."""
import hashlib
import json
import os
import re
from collections import Counter
from pathlib import Path

from .persist import PINS as _PERSIST_PINS

MR = 'vllm.v1.worker.gpu.model_runner'
MODEL = 'vllm.models.qwen3_8_flash_next.nvidia.model'
PINS = {MR: _PERSIST_PINS[MR], MODEL: 'd900cd6fcacba18f460e00b3f018fbf36fbe6ecc310692b6adb1451f7f53cc17'}
_RAW = os.environ.get('ARC3_S2_EPRUNE_FILE') or 'eprune_k16.json'
FILE = str(Path(__file__).resolve().parent / _RAW) if _RAW and os.sep not in _RAW else _RAW
ROLE = 'target:'
STATS = dict(file=FILE, layers=0, experts=0, built=0, gates_permuted=0, tensors_renamed=0, tensors_dropped=0, bytes_freed=0,
             loads_without_experts=0, code_sha256=None, file_sha256=None)
_PLAN = {}                      # MoE runner module name -> dict(pruned, n, kept, local)
_SEEN = {}                      # module name -> Counter: 'gate', 'dropped', local expert id -> tensors
_RX = re.compile(r'^model\.language_model\.layers\.(\d+)\.mlp\.(?:experts\.(\d+)\.(.+)|gate\.weight)$')
_GATE = re.compile(r'^model\.language_model\.layers\.(\d+)\.mlp\.gate\.')   # any gate tensor; only gate.weight is permuted


def _plan():
    if not _PLAN:
        STATS['file_sha256'] = hashlib.sha256(Path(FILE).read_bytes()).hexdigest()[:16]
        raw = json.load(open(FILE))
        orders = {k[len('order:'):]: [int(i) for i in v] for k, v in raw.items() if k.startswith('order:')}   # l4port patch 06
        for key, ids in raw.items():
            if key.startswith('order:'):
                continue
            assert key.startswith(ROLE), ('eprune: target layers only', key)
            ids = sorted(set(int(i) for i in ids))
            name = key[len(ROLE):]
            if ids or name in orders:
                _PLAN[name] = dict(pruned=ids)
                if name in orders:
                    _PLAN[name]['order'] = orders[name]
        assert _PLAN, 'eprune: the file lists no experts'
    return _PLAN


def patch_model(module):
    import torch
    import vllm.model_executor.models.qwen3_next as q3n          # imported by the model module: its FusedMoEFactory global

    block = module.Qwen3_8FlashNextSparseMoeBlock
    init = block.__init__

    def __init__(self, vllm_config, prefix=''):
        plan = _plan().get(prefix + '.experts')
        if plan is None:
            return init(self, vllm_config, prefix=prefix)
        factory = q3n.FusedMoEFactory

        def pruned_factory(*a, **k):
            assert not a and k.get('prefix') == prefix + '.experts', ('eprune: unexpected factory call', k.get('prefix'))
            n, ids = int(k['num_experts']), plan['pruned']
            assert 'n' not in plan and (not ids or ids[-1] < n) and n - len(ids) > int(k['top_k']), ('eprune: bad layer plan', prefix, n, len(ids))
            drop = set(ids)
            kept = [e for e in range(n) if e not in drop]
            if 'order' in plan:                                   # l4port patch 06: placement order of the kept experts
                assert sorted(plan['order']) == kept, ('eprune: the order is not a permutation of the kept experts', prefix)
                kept = list(plan['order'])
                STATS['ordered'] = STATS.get('ordered', 0) + 1
            plan.update(n=n, kept=kept, local={e: i for i, e in enumerate(kept)})
            STATS['built'] += 1
            return factory(*a, **dict(k, num_experts=len(kept)))

        q3n.FusedMoEFactory = pruned_factory                      # this block's construction only
        try:
            return init(self, vllm_config, prefix=prefix)
        finally:
            q3n.FusedMoEFactory = factory

    block.__init__ = __init__

    top = module.Qwen3_8FlashNextForConditionalGeneration
    load_weights = top.load_weights

    def rewrite(weights):
        plan = _plan()
        for name, w in weights:
            m = _RX.match(name)
            p = plan.get('language_model.model.layers.%s.mlp.experts' % m.group(1)) if m is not None else None
            if m is None:
                g = _GATE.match(name)
                assert g is None or plan.get('language_model.model.layers.%s.mlp.experts' % g.group(1)) is None, \
                    ('eprune: an unpermuted gate tensor of a listed layer', name)
            if p is not None and 'n' not in p:                   # l4port patch 07: a layer of another PP stage, not built here:
                yield name, w                                      # passed through unchanged, the loader skips it as PP-missing
                continue
            if p is not None:
                assert 'n' in p, ('eprune: a listed layer was not built pruned', name)
                seen = _SEEN.setdefault('language_model.model.layers.%s.mlp.experts' % m.group(1), Counter())
                if m.group(2) is None:                            # the gate: rows [kept..., pruned...]
                    assert w.dim() == 2 and w.shape[0] == p['n'], ('eprune: gate shape', name, tuple(w.shape))
                    w = w.index_select(0, torch.tensor(p['kept'] + p['pruned'], device=w.device))
                    seen['gate'] += 1
                else:
                    j = p['local'].get(int(m.group(2)))
                    if j is None:                                 # a pruned expert: never loaded
                        seen['dropped'] += 1
                        continue
                    name = 'model.language_model.layers.%s.mlp.experts.%d.%s' % (m.group(1), j, m.group(3))
                    seen[j] += 1
            yield name, w

    def load_weights_pruned(self, weights):
        loaded = load_weights(self, rewrite(weights))
        if not any(_SEEN.get(key) for key in _plan()):             # a reader without the routed experts (the PLE offload
            STATS['loads_without_experts'] += 1                    # worker keeps the n-gram tables only); _attach checks that
            return loaded                                          # the GPU worker's load was rewritten
        for key, p in _plan().items():
            if 'n' not in p:                                       # l4port patch 07: not built on this PP stage
                continue
            seen = _SEEN.get(key, Counter())
            per = [seen[j] for j in range(len(p['kept']))]
            assert seen['gate'] == 1 and min(per) == max(per) > 0 and seen['dropped'] == per[0] * len(p['pruned']), \
                ('eprune: incomplete load', key, seen['gate'], min(per), max(per), seen['dropped'])
            STATS['gates_permuted'] += 1
            STATS['tensors_renamed'] += sum(per)
            STATS['tensors_dropped'] += seen['dropped']
        return loaded

    top.load_weights = load_weights_pruned


def _attach(runner):
    plan, found = _plan(), 0
    built = [key for key, p in plan.items() if 'n' in p]                  # l4port patch 07: this rank's layers (all at PP 1)
    assert STATS['gates_permuted'] == len(built), ('eprune: the target load was not rewritten', STATS['gates_permuted'], len(built))
    for name, m in runner.model.named_modules():
        p = plan.get(name)
        if p is None:
            continue
        k = len(p['kept'])
        r, ex = m.router, m.routed_experts
        assert getattr(ex.quant_method, 'is_monolithic', None) is False, \
            ('eprune: a monolithic MoE kernel routes inside the kernel from the full logits', name)
        assert getattr(m, 'shared_expert_gate', None) is None and not getattr(m, '_fse_fuse_gate', False), \
            ('eprune: a gate fused with the shared-expert gate (its column would be filled too)', name)
        from vllm.distributed import get_tensor_model_parallel_world_size       # L4 port (2026-10-30): EP holds k / ep per rank
        local = int(ex.local_num_experts)
        ep = k // local if local else 0
        assert r.global_num_experts == k and ex.global_num_experts == k and local * ep == k and \
            ep in (1, get_tensor_model_parallel_world_size()), \
            ('eprune: expert counts', name, r.global_num_experts, ex.global_num_experts, ex.local_num_experts, ep)
        assert m.gate.weight.shape[0] == p['n'], ('eprune: gate rows', name, tuple(m.gate.weight.shape))
        nbytes = 0
        for pn, prm in ex.named_parameters():
            assert prm.dim() == 0 or prm.shape[0] != p['n'], ('eprune: a full-size expert parameter', name, pn, tuple(prm.shape))
            if prm.dim() and prm.shape[0] == local:
                nbytes += prm.numel() * prm.element_size()
        assert nbytes > 0, ('eprune: no expert parameters', name)
        orig = r._compute_routing

        def routed(hidden_states, router_logits, indices_type, *a, _orig=orig, _k=k, **kw):
            router_logits[:, _k:].fill_(float('-inf'))            # the pruned experts' logits (the permuted gate's tail)
            return _orig(hidden_states, router_logits, indices_type, *a, **kw)

        r._compute_routing = routed
        found += 1
        STATS['experts'] += p['n'] - k
        STATS['bytes_freed'] += nbytes * (p['n'] - k) // k          # per GPU: (nbytes / local) x (n - k) / ep
        STATS['ep'] = ep
    assert found == len(built) == STATS['built'] > 0, ('eprune: layers found / built here / built', found, len(built), STATS['built'])
    STATS['layers'] = found
    STATS['code_sha256'] = hashlib.sha256(Path(globals()['__file__']).read_bytes()).hexdigest()[:16] if '__file__' in globals() else None
    print('[eprune] %d layers, %d experts removed, %.3f GiB of expert weights not allocated; %d tensors renamed, %d dropped (%s)' % (
        found, STATS['experts'], STATS['bytes_freed'] / 2 ** 30, STATS['tensors_renamed'], STATS['tensors_dropped'], FILE), flush=True)
    if STATS.get('ordered'):
        print('[eprune] l4port placement: %d layers ordered by the file' % STATS['ordered'], flush=True)


def install_model_runner(module):
    cls = module.GPUModelRunner
    orig = cls.load_model

    def load_model(self, *a, **k):
        result = orig(self, *a, **k)
        _attach(self)                                              # failures raise: a silent no-op would void the flag
        return result

    cls.load_model = load_model


PATCHERS = {MODEL: patch_model, MR: install_model_runner}
