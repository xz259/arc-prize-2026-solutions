"""stage2_v1 'b1inc' part (milestone 2 B1b, 2026-10-01): Intel's AutoRound W4A16 experts (vLLM INC config,
GPTQ-Marlin MoE) in the production composition. Two compatibility patches, both inert unless the target's quant config
is INCConfig:
  1. ple_layer._get_ple_embedding_quant_method: under INC the PLE n-gram embedding takes production's FP8
     global-scale method. The hybrid folder carries NVIDIA's FP8 PLE tensors, as production does (they are Intel's
     BF16 table quantized with one global scale: M2_DECISIONS, PLE identity). The stock function selects FP8 only for
     ModelOpt/Fp8 configs. ARC3_S2_B1INC_PLE=bf16 leaves the stock choice (unquantized BF16 table), for a target folder
     carrying Intel's BF16 PLE shard.
  2. INCConfig.get_quant_method: ParallelLMHead gets None, i.e. vLLM's default UnquantizedEmbeddingMethod, as under
     production's ModelOpt config, so the dense-FP8 adapter's lm_head tier applies unchanged. INC would return
     UnquantizedLinearMethod, bypassing the tier and keeping a BF16 head.
Everything else (dense layers, router, shared experts, vision, embedding) resolves to unquantized under both configs and
takes production's dense-FP8 tier the same way."""

PINS = {'vllm.models.qwen3_8_flash_next.nvidia.ple_layer': '160d553ec81bae0990934cfe7c14f98d2d65b6ee3d5c5a85ff8421b7ddb17952',
        'vllm.model_executor.layers.quantization.inc.inc': '03d4b80cfb52c202079c1a1658f32e6f9cbd7bf0841bc3cc2b19f0744a147aa1'}
STATS = dict(ple_fp8_under_inc=0, ple_bf16_under_inc=0, lm_head_default_under_inc=0)
PLE_MODE = __import__('os').environ.get('ARC3_S2_B1INC_PLE', 'fp8')
assert PLE_MODE in ('fp8', 'bf16'), PLE_MODE


def patch_ple_layer(mod):
    original = mod._get_ple_embedding_quant_method

    def _get_ple_embedding_quant_method(quant_config, prefix):
        if type(quant_config).__name__ == 'INCConfig':
            if PLE_MODE == 'bf16':
                STATS['ple_bf16_under_inc'] += 1
                return original(quant_config, prefix)
            STATS['ple_fp8_under_inc'] += 1
            return mod.Qwen3_8FlashNextPLEFp8EmbeddingMethod()
        return original(quant_config, prefix)

    mod._get_ple_embedding_quant_method = _get_ple_embedding_quant_method


def patch_inc(mod):
    cls = mod.INCConfig
    original = cls.get_quant_method

    def get_quant_method(self, layer, prefix):
        if type(layer).__name__ == 'ParallelLMHead':
            STATS['lm_head_default_under_inc'] += 1
            return None
        return original(self, layer, prefix)

    get_quant_method.__wrapped__ = original
    cls.get_quant_method = get_quant_method


PATCHERS = {'vllm.models.qwen3_8_flash_next.nvidia.ple_layer': patch_ple_layer,
            'vllm.model_executor.layers.quantization.inc.inc': patch_inc}
