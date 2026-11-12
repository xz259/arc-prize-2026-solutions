"""Install-time patches (called by the pinned import hook right after each target module executed)."""
import os
import torch

_MODE = os.environ.get('ARC3_GLUE_FUSION', '0')
PARTS = frozenset() if _MODE == '0' else (frozenset({'hc', 'gemm', 'gdn'}) if _MODE == 'all' else frozenset(_MODE.split(',')))
assert PARTS <= {'hc', 'gemm', 'gdn', 'hcf', 'splitk'}, _MODE
assert not {'hc', 'hcf'} <= PARTS, 'hc (rejected silu-fused up GEMM) and hcf (stage 14 fused HC site) are exclusive'


def install_hyperconnection(module):
    if 'hcf' in PARTS:
        return install_hyperconnection_hcf(module)
    if 'hc' not in PARTS:
        return
    from .kernels import hc_up_silu_gemm, preallocate
    preallocate(torch.device('cuda'))
    GatedResidual = module.GatedResidual
    grouped_gemma_rmsnorm, hc_combine_norm, hc_gate_mix = module.grouped_gemma_rmsnorm, module.hc_combine_norm, module.hc_gate_mix
    hc_silu = module.hc_silu

    def _up_site(self, xn, lora):
        up = self.input_mix_weight_up
        w = getattr(up, 'weight', None)
        if w is not None and w.dtype == torch.bfloat16 and getattr(up, 'bias', None) is None and lora.dtype == torch.bfloat16:
            gate = hc_up_silu_gemm(lora, w, self.hc_count)
        else:  # quantized (tier B2) or unexpected dtype: vendor pipeline
            gate = up(hc_silu(lora, self.hc_count))
        return hc_gate_mix(xn, gate, self.hc_count)

    def _tail(self, xn):
        if self.use_combine:
            split_sizes = [self.lora_rank, self.hc_count, self.pad_size]
            down_and_injection = self.input_mix_weight_down_block_inject(xn)
            lora, injection, _ = down_and_injection.split(split_sizes, dim=-1)
        else:
            lora = self.input_mix_weight_down(xn)
            injection = None
        return _up_site(self, xn, lora), injection

    def mix(self, hidden_states):
        xn = grouped_gemma_rmsnorm(hidden_states, self.hc_norm.weight, self.config.rms_norm_eps, self.hc_count)
        block_input, injection = _tail(self, xn)
        return hidden_states, block_input, injection

    def combine_and_mix(self, hidden_states, prev_block_output, prev_injection):
        hidden_states, xn = hc_combine_norm(hidden_states, prev_block_output, prev_injection,
                                            self.hc_norm.weight, self.config.rms_norm_eps, self.hc_count)
        block_input, injection = _tail(self, xn)
        return hidden_states, block_input, injection

    GatedResidual.mix = mix
    GatedResidual.combine_and_mix = combine_and_mix
    GatedResidual._arc3_glue_fusion = 'hc'


def install_hyperconnection_hcf(module):
    """Stage 14 (2026-09-27): at every HC mix site, down GEMM + split-K reduction + hc_silu in one launch
    (hcf_down) and the up projection as a 640-program GEMM (up_gemm); hc_gate_mix stays the vendor kernel. Both ops
    fall back to the vendor pipeline outside 17 <= M <= 64 (prefill, small batches) or for unexpected dtypes."""
    from .kernels_hcf import hcf_down, up_gemm, preallocate
    preallocate(torch.device('cuda'))
    GatedResidual = module.GatedResidual
    grouped_gemma_rmsnorm, hc_combine_norm, hc_gate_mix = module.grouped_gemma_rmsnorm, module.hc_combine_norm, module.hc_gate_mix

    def _tail(self, xn):
        if self.use_combine:
            down = hcf_down(xn, self.input_mix_weight_down_block_inject.weight, self.lora_rank, self.hc_count)
            lora, injection, _ = down.split([self.lora_rank, self.hc_count, self.pad_size], dim=-1)
        else:
            lora, injection = hcf_down(xn, self.input_mix_weight_down.weight, self.lora_rank, self.hc_count), None
        gate = up_gemm(lora, self.input_mix_weight_up.weight)
        return hc_gate_mix(xn, gate, self.hc_count), injection

    def mix(self, hidden_states):
        xn = grouped_gemma_rmsnorm(hidden_states, self.hc_norm.weight, self.config.rms_norm_eps, self.hc_count)
        block_input, injection = _tail(self, xn)
        return hidden_states, block_input, injection

    def combine_and_mix(self, hidden_states, prev_block_output, prev_injection):
        hidden_states, xn = hc_combine_norm(hidden_states, prev_block_output, prev_injection,
                                            self.hc_norm.weight, self.config.rms_norm_eps, self.hc_count)
        block_input, injection = _tail(self, xn)
        return hidden_states, block_input, injection

    GatedResidual.mix = mix
    GatedResidual.combine_and_mix = combine_and_mix
    GatedResidual._arc3_glue_fusion = 'hcf'


def install_linear(module):
    if not {'gemm', 'splitk'} & PARTS:
        return
    cls = module.UnquantizedLinearMethod
    vendor_apply = cls.apply
    tables = []
    if 'gemm' in PARTS:
        from .kernels import small_gemm, preallocate, GEMM_TABLE
        preallocate(torch.device('cuda'))
        tables.append((GEMM_TABLE, small_gemm))
    if 'splitk' in PARTS:   # stage 14: single-launch split-K fixup for router / in_proj_ba / index_qk (main stream)
        from . import kernels_hcf
        kernels_hcf.preallocate(torch.device('cuda'))
        if 'gemm' in PARTS:
            from .kernels import _small_gemm
            kernels_hcf.SMALL_M_FALLBACK = _small_gemm
        tables.insert(0, (kernels_hcf.SPLITK_TABLE, kernels_hcf.splitk_gemm))

    def apply(self, layer, x, bias=None):
        w = getattr(layer, 'weight', None)
        if bias is None and w is not None and x.dim() == 2:
            shape = tuple(w.shape)
            for table, op in tables:
                if shape in table:
                    return op(x, w)
        return vendor_apply(self, layer, x, bias)

    cls.apply = apply
    cls._arc3_glue_fusion = ','.join(sorted({'gemm', 'splitk'} & PARTS))


def install_gdn(module):
    if 'gdn' not in PARTS:
        return
    cls = module.QwenGatedDeltaNetAttention
    get_forward_context, GDNAttentionMetadata = module.get_forward_context, module.GDNAttentionMetadata
    from vllm.third_party.flash_linear_attention.ops.layernorm_guard import layer_norm_fwd_kernel, calc_rows_per_block
    from vllm.triton_utils import triton

    def _forward_core_fused_norm(self, mixed_qkv, b, a, output_gate, core_attn_out):
        forward_context = get_forward_context()
        attn_metadata_raw = forward_context.attn_metadata
        if attn_metadata_raw is None:
            self._warmup_prefill_kernels(mixed_qkv, 0)
            return
        assert isinstance(attn_metadata_raw, dict)
        attn_metadata = attn_metadata_raw[self.prefix]
        assert isinstance(attn_metadata, GDNAttentionMetadata)
        if self._can_use_fused_gdn_mtp_decode(attn_metadata) and attn_metadata.num_prefills == 0:
            self._forward_core_decode_spec_fused_norm(mixed_qkv=mixed_qkv, b=b, a=a, output_gate=output_gate,
                                                      core_attn_out=core_attn_out, attn_metadata=attn_metadata)
            return
        packed = (self.enable_packed_recurrent_decode and attn_metadata.spec_sequence_masks is None
                  and attn_metadata.num_prefills == 0 and attn_metadata.num_decodes > 0)
        # The packed non-spec decode kernel takes row strides (only stride(-1) == 1 is required), so the [M, 2H] ->
        # b/a chunk views go in directly; every other path keeps the vendor copies.
        self._forward_core(mixed_qkv=mixed_qkv, b=b if packed else b.contiguous(), a=a if packed else a.contiguous(),
                           core_attn_out=core_attn_out)
        num_actual_tokens = attn_metadata.num_actual_tokens
        self._rms_norm_gated_cuda(core_attn_out[:num_actual_tokens], output_gate[:num_actual_tokens], core_attn_out[:num_actual_tokens])

    vendor_rms_norm_gated = cls._rms_norm_gated_cuda

    def _rms_norm_gated_cuda(self, x, output_gate, out):
        # Strided [M, H, D] output gate (a view into mixed_qkvz): run the same kernel with H groups per row instead of
        # copying the gate to a contiguous [M*H, D] buffer. Same tile (ROWS_PER_BLOCK from the vendor's M*H row count,
        # BLOCK_N = D, same num_warps) and the same per-row reduction: bit-identical (check.py).
        if (x.dim() != 3 or self.norm.bias is not None or self.norm.group_size is not None or output_gate.is_contiguous()
                or output_gate.stride(2) != 1 or output_gate.stride(1) != x.shape[2] or not x.is_contiguous()
                or out.data_ptr() != x.data_ptr() or out.shape != x.shape):
            return vendor_rms_norm_gated(self, x, output_gate, out)
        M, H, D = x.shape
        x_2d = x.view(M, H * D)
        z_2d = output_gate.view(M, H * D)
        w = getattr(self, '_arc3_norm_weight_rep', None)
        if w is None or w.shape[0] != H * D:
            w = self.norm.weight.detach().contiguous().repeat(H)
            self._arc3_norm_weight_rep = w
        rows_per_block = calc_rows_per_block(M * H, x.device)
        rstd = torch.empty((H * M,), dtype=torch.float32, device=x.device)
        BLOCK_N = triton.next_power_of_2(D)
        num_warps = min(max(BLOCK_N // 256, 1), 8)
        layer_norm_fwd_kernel[(triton.cdiv(M, rows_per_block), H)](
            x_2d, x_2d, w, None, z_2d, None, rstd, x_2d.stride(0), x_2d.stride(0), z_2d.stride(0), M, D, self.norm.eps,
            BLOCK_N=BLOCK_N, ROWS_PER_BLOCK=rows_per_block, HAS_BIAS=False, HAS_Z=True,
            NORM_BEFORE_GATE=self.norm.norm_before_gate, IS_RMS_NORM=True, num_warps=num_warps, ACTIVATION=self.norm.activation)

    cls._forward_core_fused_norm = _forward_core_fused_norm
    cls._rms_norm_gated_cuda = _rms_norm_gated_cuda
    cls._arc3_glue_fusion = 'gdn'
