# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""HyperConnection (Gated Residual) utilities — NVIDIA model variant.

Implements the HyperConnection residual scheme proposed in
"HyperConnections" (https://arxiv.org/abs/2409.19606). This NVIDIA variant
delays each HC combine to the following HC mix boundary. HC glue kernels,
including fused combine+RMSNorm, live in ``ops/hc.py``; projections remain
standard vLLM Linear modules.

Hidden states between layers have shape ``[..., HC*HS]`` with HS inner
(HC outer, HS inner — checkpoint-native layout).

Typical usage inside a transformer decoder layer::

    self.attn_hc = GatedResidual(hc_config)

    hidden_states, block_input, injection = self.attn_hc.mix(hidden_states)
    attention_output = attention(block_input)
    hidden_states, block_input, injection = self.mlp_hc.combine_and_mix(
        hidden_states, attention_output, injection
    )
"""

import os

import torch
from torch import nn

from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    ReplicatedLinear,
)
from vllm.model_executor.models.utils import maybe_prefix

from ..common.hyperconnection import (
    GroupedGemmaRMSNorm,
    HyperConnectionConfig,
)
from .ops.hc import (
    grouped_gemma_rmsnorm,
    hc_combine,
    hc_combine_norm,
    hc_gate_mix,
    hc_silu,
)


# ---------------------------------------------------------------------------
# Gated-residual variant
# ---------------------------------------------------------------------------
_HC_SHARD = os.environ.get("ARC3_L4PORT_HC_SHARD", "")
_HC_SHARD_RECEIPT = []


class _ColShardedDown(nn.Module):
    """L4 port (2026-10-30, 02c): an HC down (+inject) projection sharded by output across the TP ranks. This rank keeps
    rows [rank * B, (rank + 1) * B) of the merged [lora | inject | pad] output (B = total / tp), so the all-gather of the
    local outputs is the unsharded column order (no regroup). Checkpoint shards (the stacked down = 0, inject = 1) arrive
    with the shard id as a tensor attribute (WeightsMapper.orig_to_new_stacked); padding rows are never loaded (discarded
    downstream, as in the unsharded layer)."""

    def __init__(self, in_features, out_sizes, dtype):
        super().__init__()
        from vllm.distributed import get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
        self.tp, self.rank = get_tensor_model_parallel_world_size(), get_tensor_model_parallel_rank()
        self.out_sizes = list(out_sizes)
        total = sum(self.out_sizes)
        assert total % self.tp == 0, (out_sizes, self.tp)
        self.block = total // self.tp
        self.weight = nn.Parameter(torch.zeros(self.block, in_features, dtype=dtype), requires_grad=False)
        self.weight.weight_loader = self._load
        if not _HC_SHARD_RECEIPT:
            _HC_SHARD_RECEIPT.append(1)
            print("[l4port] HC down shard col: tp %d rank %d, local rows %d of %d" % (self.tp, self.rank, self.block, total),
                  flush=True)

    def _load(self, param, loaded_weight, shard_id=None):
        if shard_id is None:
            shard_id = getattr(loaded_weight, "shard_id", None)
        i = 0 if shard_id is None else int(shard_id)
        assert tuple(loaded_weight.shape) == (self.out_sizes[i], param.shape[1]), (tuple(loaded_weight.shape), i)
        o = sum(self.out_sizes[:i])                        # this shard's rows in the merged output: [o, o + size)
        lo, hi = self.rank * self.block, (self.rank + 1) * self.block
        a, b = max(o, lo), min(o + self.out_sizes[i], hi)
        if a < b:
            param.data[a - lo:b - lo].copy_(loaded_weight[a - o:b - o])

    def forward(self, x):
        from vllm.distributed import tensor_model_parallel_all_gather
        return tensor_model_parallel_all_gather(torch.nn.functional.linear(x, self.weight), -1)


class GatedResidual(nn.Module):
    """Gated HyperConnection with learnable low-rank mixing and injection.

    ``combine_and_mix()`` runs the pre pipeline (grouped GemmaRMSNorm -> merged
    low-rank down+inject GEMM -> silu -> up GEMM -> sigmoid -> gated mean
    over the HC streams). When passed a pending block output and an injection,
    it fuses their residual combine with the RMSNorm. Final mixers use
    ``use_combine=False`` and do not produce a new injection.

    Weights: the norm owns the grouped GemmaRMSNorm affine; the projections
    are vLLM Linear modules (merged replicated linear for down+inject), so
    GEMM dispatch (e.g. the low-latency skinny GEMM) applies through the
    standard quant_method mechanism.
    """

    def __init__(
        self,
        config: HyperConnectionConfig,
        use_combine: bool = True,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.lora_rank = config.hc_lowrank
        self.hc_count = config.hc_count
        self.hidden_size = config.hidden_size
        self.use_combine = use_combine

        norm_size = (
            self.hyper_hidden_size if config.hc_per_branch_norm else config.hidden_size
        )
        group_size = config.hidden_size if config.hc_per_branch_norm else None
        # Normalize each H-sized HC stream independently while retaining a
        # separate affine weight for every element of the HC*H layout.
        self.hc_norm = GroupedGemmaRMSNorm(
            norm_size,
            eps=config.rms_norm_eps,
            group_size=group_size,
            dtype=config.params_dtype,
        )

        # -- vLLM Linear weights --------------------------------------------
        # The merged skinny-GEMM shape is physically padded to 16 rows to ensure
        # good alignment and performant implementation chosen by CuBLAS heuristics.
        self.pad_size = (-(self.lora_rank + self.hc_count)) % 16 if use_combine else 0
        _tp = 1
        if _HC_SHARD == "col":
            from vllm.distributed import get_tensor_model_parallel_world_size
            _tp = get_tensor_model_parallel_world_size()
        if use_combine and _tp > 1:
            self.input_mix_weight_down_block_inject = _ColShardedDown(
                self.hyper_hidden_size,
                [self.lora_rank, self.hc_count] + ([self.pad_size] if self.pad_size else []),
                config.params_dtype,
            )
        elif _tp > 1:
            self.input_mix_weight_down = _ColShardedDown(self.hyper_hidden_size, [self.lora_rank], config.params_dtype)
        elif use_combine:
            self.input_mix_weight_down_block_inject = MergedColumnParallelLinear(
                self.hyper_hidden_size,
                [self.lora_rank, self.hc_count]
                + ([self.pad_size] if self.pad_size else []),
                bias=False,
                params_dtype=config.params_dtype,
                quant_config=None,
                prefix=maybe_prefix(prefix, "input_mix_weight_down_block_inject"),
                return_bias=False,
                disable_tp=True,
            )
        else:
            self.input_mix_weight_down = ReplicatedLinear(
                self.hyper_hidden_size,
                self.lora_rank,
                bias=False,
                params_dtype=config.params_dtype,
                quant_config=None,
                prefix=maybe_prefix(prefix, "input_mix_weight_down"),
                return_bias=False,
            )
        self.input_mix_weight_up = ReplicatedLinear(
            self.lora_rank,
            self.hyper_hidden_size,
            bias=False,
            params_dtype=config.params_dtype,
            quant_config=None,
            prefix=maybe_prefix(prefix, "input_mix_weight_up"),
            return_bias=False,
        )

    def mix(
        self, hidden_states: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        xn = grouped_gemma_rmsnorm(
            hidden_states,
            self.hc_norm.weight,
            self.config.rms_norm_eps,
            self.hc_count,
        )

        if self.use_combine:
            # produce injection logits for combine
            split_sizes = [self.lora_rank, self.hc_count, self.pad_size]
            down_and_injection = self.input_mix_weight_down_block_inject(xn)
            lora, injection, _ = down_and_injection.split(split_sizes, dim=-1)
        else:
            lora = self.input_mix_weight_down(xn)
            injection = None

        lora = hc_silu(lora, self.hc_count)
        gate = self.input_mix_weight_up(lora)  # [M, D]
        block_input = hc_gate_mix(xn, gate, self.hc_count)

        return hidden_states, block_input, injection

    def combine_and_mix(
        self,
        hidden_states: torch.Tensor,
        prev_block_output: torch.Tensor,
        prev_injection: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        """Consume a pending combine, then prepare the next block input.

        ``hidden_states`` is the multi-stream state from before the pending
        block's mix. Its combine with ``block_output`` is fused with this
        module's input RMSNorm.
        """
        hidden_states, xn = hc_combine_norm(
            hidden_states,
            prev_block_output,
            prev_injection,
            self.hc_norm.weight,
            self.config.rms_norm_eps,
            self.hc_count,
        )

        if self.use_combine:
            # produce injection logits for combine
            split_sizes = [self.lora_rank, self.hc_count, self.pad_size]
            down_and_injection = self.input_mix_weight_down_block_inject(xn)
            lora, injection, _ = down_and_injection.split(split_sizes, dim=-1)
        else:
            lora = self.input_mix_weight_down(xn)
            injection = None

        lora = hc_silu(lora, self.hc_count)
        gate = self.input_mix_weight_up(lora)  # [M, D]
        block_input = hc_gate_mix(xn, gate, self.hc_count)

        return hidden_states, block_input, injection

    def combine(
        self,
        hidden_states: torch.Tensor,
        block_output: torch.Tensor,
        injection: torch.Tensor,
    ) -> torch.Tensor:
        return hc_combine(hidden_states, block_output, injection, self.hc_count)

    @property
    def hyper_hidden_size(self) -> int:
        return self.hc_count * self.hidden_size


__all__ = [
    "GatedResidual",
    "GroupedGemmaRMSNorm",
    "HyperConnectionConfig",
]
