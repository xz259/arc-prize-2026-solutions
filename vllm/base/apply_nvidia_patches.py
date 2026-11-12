"""Small, hash-guarded NVIDIA loader fixes for the supplied vLLM runtime.

PLE handling follows the upstream FP8 PLE fix. MTP dispatch/remapping is adapted
from PR #55513 at cc3cf6e9d6ad9046f638bc720994e5f22dc62378. No kernels or weights
are replaced. Original source is retained under patches/original/.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCES = {
    "vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py": "a71144c1d36e06f22a2da1b1ada900076597fe5e824a911e7ada86249a0993e7",
    "vllm/model_executor/layers/quantization/modelopt.py": "3f3ca743fd3c66d72be92b7544591a8632f1aa422b73bb50c83e8a3281196e7d",
    "vllm/models/qwen3_8_flash_next/nvidia/mtp.py": "7735cee47d0d1e4776bebd30d907e4a62160409ce4ef2d65611559f8d58af431",
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def once(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError(f"Expected exactly one patch anchor: {old[:100]!r}")
    return text.replace(old, new, 1)


def patched_source(name, source):
    s = source.decode()
    if name.endswith('/ple_layer.py'):
        s = once(s, 'from vllm.model_executor.layers.quantization.fp8 import Fp8Config\n',
                 'from vllm.model_executor.layers.quantization.fp8 import Fp8Config\n'
                 'from vllm.model_executor.layers.quantization.modelopt import ModelOptMixedPrecisionConfig\n')
        s = once(s, '            scale_dtype=torch.bfloat16,\n', '            scale_dtype=torch.float32,\n')
        s = once(s, '        layer.register_parameter("weight_scale", weight_scale)\n\n    def apply(',
                 '        layer.register_parameter("weight_scale", weight_scale)\n\n'
                 '    def process_weights_after_loading(self, layer: nn.Module) -> None:\n'
                 '        sentinel = torch.finfo(torch.float32).min\n'
                 '        if torch.any(layer.weight_scale == sentinel):\n'
                 '            raise ValueError("FP8 PLE checkpoint is missing its global scale")\n\n'
                 '    def apply(')
        s = once(s, '    if not isinstance(quant_config, Fp8Config):\n',
                 '    if isinstance(quant_config, ModelOptMixedPrecisionConfig):\n'
                 '        if quant_config._resolve_quant_algo(prefix) == "FP8":\n'
                 '            return Qwen3_8FlashNextPLEFp8EmbeddingMethod()\n'
                 '        return None\n\n'
                 '    if not isinstance(quant_config, Fp8Config):\n')
    elif name.endswith('/modelopt.py'):
        s = once(s, 'from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod\n',
                 'from vllm.model_executor.layers.quantization.fp8 import Fp8Config, Fp8MoEMethod\n'
                 'from vllm.model_executor.layers.quantization.kv_cache import BaseKVCacheMethod\n')
        s = once(s, 'logger = init_logger(__name__)\n',
                 'logger = init_logger(__name__)\n\n'
                 '_BLOCK_FP8_MOE_ALGOS = ("FP8_PB_WO", "FP8_BLOCK_SCALES")\n')
        s = once(s, '        self.mxfp8_config = mxfp8_config\n',
                 '        self.mxfp8_config = mxfp8_config\n'
                 '        block_sizes = {\n'
                 '            int(info.get("group_size", 128))\n'
                 '            for info in quantized_layers.values()\n'
                 '            if info.get("quant_algo", "").upper() in _BLOCK_FP8_MOE_ALGOS\n'
                 '        }\n'
                 '        if len(block_sizes) > 1:\n'
                 '            raise ValueError(f"Mixed block-FP8 group sizes: {sorted(block_sizes)}")\n'
                 '        block_size = next(iter(block_sizes), 128)\n'
                 '        self.fp8_block_config = Fp8Config(\n'
                 '            is_checkpoint_fp8_serialized=True, activation_scheme="dynamic",\n'
                 '            weight_block_size=[block_size, block_size],\n'
                 '        )\n')
        s = once(s, '        if isinstance(layer, RoutedExperts):\n            if quant_algo == "FP8":\n',
                 '        if isinstance(layer, RoutedExperts):\n'
                 '            if quant_algo in _BLOCK_FP8_MOE_ALGOS:\n'
                 '                return Fp8MoEMethod(self.fp8_block_config, layer)\n'
                 '            if quant_algo == "FP8":\n')
    elif name.endswith('/mtp.py'):
        s = once(s, '\ndef _remap_mtp_weight_name(name: str) -> str | None:\n',
                 '\ndef _remap_quantized_layers(quantized_layers: dict[str, dict], mtp_start_layer_idx: int) -> dict[str, dict]:\n'
                 '    """Map checkpoint MTP indices into the standalone draft model."""\n'
                 '    return {\n'
                 '        _remap_ignored_layers([name], mtp_start_layer_idx)[0]: info\n'
                 '        for name, info in quantized_layers.items()\n'
                 '    }\n\n\ndef _remap_mtp_weight_name(name: str) -> str | None:\n')
        s = once(s, '    draft_vllm_config = replace(\n',
                 '        quantized_layers = getattr(draft_quant_config, "quantized_layers", None)\n'
                 '        if quantized_layers:\n'
                 '            draft_quant_config.quantized_layers = _remap_quantized_layers(\n'
                 '                quantized_layers, mtp_start_layer_idx\n'
                 '            )\n\n'
                 '    draft_vllm_config = replace(\n')
    compile(s, name, 'exec')
    return s.encode()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path, default=ROOT/'runtime/base/usr/local/lib/python3.12/dist-packages')
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    changes = []
    for rel, expected in SOURCES.items():
        target = args.site/rel
        current = target.read_bytes()
        backup = ROOT/'patches/original'/rel
        original = current if sha(current) == expected else backup.read_bytes()
        if sha(original) != expected:
            raise RuntimeError(f'Original source hash mismatch: {rel}')
        modified = patched_source(rel, original)
        if sha(current) not in {expected, sha(modified)}:
            raise RuntimeError(f'Unrecognized modified source: {rel}')
        changes.append((target, backup, original, modified,
                        {'path': rel, 'original_sha256': expected, 'patched_sha256': sha(modified)}))
    if args.apply:
        for target, backup, original, modified, row in changes:
            backup.parent.mkdir(parents=True, exist_ok=True)
            if backup.exists():
                assert sha(backup.read_bytes()) == row['original_sha256']
            else:
                backup.write_bytes(original)
            temp = target.with_name(target.name+'.arc3-tmp')
            temp.write_bytes(modified)
            os.replace(temp, target)
        (ROOT/'patches/manifest.json').write_text(json.dumps({'patch':'nvidia-ple-mtp-v1','files':[c[4] for c in changes]},indent=2)+'\n')
    print(json.dumps({'applied':args.apply,'files':[c[4] for c in changes]},indent=2))


if __name__ == '__main__':
    main()
