"""stage2_v1 'precmode' part (2026-10-15, precision audit; launch-time numerics modes for teacher-forced arms and, if adopted, serving).

ARC3_S2_PRECMODE = comma list of modes, fixed for the launch (so CUDA graphs capture them); empty = no change.
  router32  the routed-expert gate of every Qwen3Next sparse MoE block (Qwen3NextSparseMoeBlock.gate: ReplicatedLinear
            hidden 2560 -> 512 experts, BF16 weight, quant_config None) returns FP32 logits: F.linear(x.float(), W.float()), the
            same products with FP32 accumulation, without the final BF16 rounding of the logits (BF16 keeps 8 significant bits:
            a logit near 4 is rounded by up to 2^-6). vLLM's fused top-k (topk_softmax, FP32 internally) then selects the top 10 and
            computes the softmax weights from unrounded logits. The gate runs inside the MoE runner's custom op (eager, captured
            in CUDA graphs), outside the compiled graph. Applies to every block built from the class: the 48 target layers and
            the MTP drafter's layer. The shared-expert gate (one logit, sigmoid) is unchanged.
Measurement and, if adopted, production: the FP32 weight is converted per call (2.6 MB of BF16 read per layer per step), so a
production version would keep an FP32 copy (5.2 MB per layer) or use a BF16-input / FP32-output GEMM.
"""
import os

import torch

MODES = frozenset(m for m in os.environ.get('ARC3_S2_PRECMODE', '').split(',') if m)
KNOWN = frozenset({'router32'})
assert MODES <= KNOWN, f'precmode: unknown modes {sorted(MODES - KNOWN)} (known: {sorted(KNOWN)})'
STATS = dict(modes=sorted(MODES), router32_gates=0)
PINS = {'vllm.model_executor.models.qwen3_next': '28aaeee979d75d8776ceb39b39b1e13a8a3930a1ed26defce5632c599e70f05e'}


def fp32_gate_forward(gate):
    """A forward for a ReplicatedLinear gate (no bias, no quantization) that returns (FP32 logits, None)."""
    assert getattr(gate, 'bias', None) is None, 'precmode router32: the gate has a bias'

    def forward(x, _gate=gate):
        return torch.nn.functional.linear(x.float(), _gate.weight.float()), None
    return forward


def patch_qwen3_next(mod):
    cls = mod.Qwen3NextSparseMoeBlock
    orig_init = cls.__init__

    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        if 'router32' in MODES:
            self.gate.forward = fp32_gate_forward(self.gate)     # instance attribute: nn.Module.__call__ dispatches to it
            STATS['router32_gates'] += 1

    cls.__init__ = __init__


PATCHERS = {'vllm.model_executor.models.qwen3_next': patch_qwen3_next}
