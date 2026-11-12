"""Reversible opt-in quantized main-QSA cache patch for the NVIDIA owner.

Global --kv-cache-dtype stays auto/BF16. ARC3_QSA_KV_FORMAT selects only this
custom backend's storage; its FullAttentionSpec includes every scale byte.
No standard FlashAttention support check or indexer/GDN precision is changed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT=Path(__file__).resolve().parent
REL='vllm/models/qwen3_8_flash_next/nvidia/qsa.py'
EXPECTED='748addc85efaa8f7df940d1245bc900192f92e1f17af8fa774625758600751cb'
STATE_REL='vllm/models/qwen3_8_flash_next/common/qsa_cache.py'
SOURCES={REL:EXPECTED,STATE_REL:'e3460b06cd7ed309e47ad5dfd3d4250890539b912385503133bd98a003f73ba8'}
HELPERS=['arc3_qsa_cache.py','arc3_qsa_profiles.py','arc3_qsa_quant.py','arc3_qsa_nvfp4.py',
         'arc3_qsa_nvfp4_attention.py']


def sha(data):return hashlib.sha256(data).hexdigest()


def once(source,old,new):
    if source.count(old)!=1:raise ValueError('QSA quantization patch anchor mismatch: '+repr(old))
    return source.replace(old,new,1)


def original_source(site, rel):
    current=(site/rel).read_bytes()
    if sha(current)==SOURCES[rel]:
        return current
    original=(ROOT/'patches/qsa-quant-original'/rel).read_bytes()
    assert sha(original)==SOURCES[rel], 'Original QSA source identity mismatch'
    return original


def patched_source(original,rel=REL):
    s=original.decode()
    if rel==STATE_REL:
        s=once(s,'class QSAStateBackend(AttentionBackend):\n',
            'class QSAStateBackend(AttentionBackend):\n')
        anchor='    @staticmethod\n    def get_name() -> str:\n        return "QWEN38_FLASH_NEXT_EXP_QSA_STATE"\n'
        s=once(s,anchor,anchor+'\n'
            '    @classmethod\n'
            '    def customize_spec(cls, spec):\n'
            '        # The alignment probe describes main KV, not this side cache.\n'
            '        import os\n'
            '        fmt = os.environ.get("ARC3_QSA_KV_FORMAT", "bf16")\n'
            '        if fmt == "bf16":\n'
            '            return spec\n'
            '        from arc3_qsa_cache import customize_main_spec\n'
            '        return customize_main_spec(spec, fmt)\n')
        compile(s,rel,'exec');return s.encode()
    assert rel==REL
    s=once(s,'from typing import ClassVar, cast\n','import os\nfrom typing import ClassVar, cast\n')
    s=once(s,'    FullAttentionSpec,\n','    FullAttentionSpec,\n    KVQuantMode,\n')
    s=once(s,'from .indexer_qsa import QSAIndexer\n',
        'from .indexer_qsa import QSAIndexer\n\n'
        '_ARC3_QSA_KV_FORMAT = os.environ.get("ARC3_QSA_KV_FORMAT", "bf16")\n'
        'if _ARC3_QSA_KV_FORMAT not in {"bf16", "fp8_e4m3", "fp8_e5m2", "nvfp4"}:\n'
        '    raise ValueError("Unsupported opt-in QSA cache format")\n')
    s=once(s,'    @staticmethod\n    def get_name() -> str:\n',
        '    @classmethod\n'
        '    def customize_spec(cls, spec):\n'
        '        if _ARC3_QSA_KV_FORMAT == "bf16":\n'
        '            return spec\n'
        '        from arc3_qsa_cache import customize_main_spec\n'
        '        return customize_main_spec(spec, _ARC3_QSA_KV_FORMAT)\n\n'
        '    @staticmethod\n'
        '    def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str="auto"):\n'
        '        if _ARC3_QSA_KV_FORMAT == "bf16":\n'
        '            return FlashAttentionBackend.get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size, cache_dtype_str)\n'
        '        from arc3_qsa_cache import side_bytes\n'
        '        return (num_blocks, num_kv_heads, block_size, 2*side_bytes(head_size, _ARC3_QSA_KV_FORMAT))\n\n'
        '    @staticmethod\n    def get_name() -> str:\n')
    s=once(s,'    def forward_qsa(\n',
        '    def do_kv_cache_update(self, layer, key, value, kv_cache, slot_mapping):\n'
        '        if _ARC3_QSA_KV_FORMAT == "bf16":\n'
        '            return super().do_kv_cache_update(layer, key, value, kv_cache, slot_mapping)\n'
        '        from arc3_qsa_cache import store\n'
        '        return store(key, value, kv_cache, slot_mapping, _ARC3_QSA_KV_FORMAT)\n\n'
        '    def forward_qsa(\n')
    s=once(s,'        token_to_req = token_to_req[:num_tokens]\n',
        '        token_to_req = token_to_req[:num_tokens]\n'
        '        if _ARC3_QSA_KV_FORMAT != "bf16":\n'
        '            from arc3_qsa_cache import attention\n'
        '            attention(query[:num_tokens], kv_cache, logical_indices,\n'
        '                          attn_metadata.block_table, token_to_req, output[:num_tokens], _ARC3_QSA_KV_FORMAT)\n'
        '            return output\n')
    s=once(s,'        tp_size = get_tensor_model_parallel_world_size()\n',
        '        tp_size = get_tensor_model_parallel_world_size()\n'
        '        if _ARC3_QSA_KV_FORMAT != "bf16" and tp_size != 1:\n'
        '            raise NotImplementedError("Initial quantized QSA validation is scoped to TP1")\n')
    s=once(s,'        self.kv_sharing_target_layer_name = None\n',
        '        if _ARC3_QSA_KV_FORMAT != "bf16":\n'
        '            self.kv_cache_torch_dtype = torch.uint8\n'
        '        self.arc3_qsa_kv_format = _ARC3_QSA_KV_FORMAT\n'
        '        self.kv_sharing_target_layer_name = None\n')
    s=once(s,'    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:\n',
        '    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:\n'
        '        if _ARC3_QSA_KV_FORMAT != "bf16":\n'
        '            from arc3_qsa_cache import side_bytes, quant_mode\n'
        '            return FullAttentionSpec(\n'
        '                block_size=vllm_config.cache_config.block_size,\n'
        '                num_kv_heads=self.num_kv_heads, head_size=self.head_dim, head_size_v=self.head_dim,\n'
        '                dtype=torch.uint8, kv_quant_mode=quant_mode(_ARC3_QSA_KV_FORMAT),\n'
        '                state_content_bytes=2*side_bytes(self.head_dim, _ARC3_QSA_KV_FORMAT),\n'
        '            )\n')
    compile(s,REL,'exec');return s.encode()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--site',type=Path,default=ROOT/'runtime/base/usr/local/lib/python3.12/dist-packages')
    p.add_argument('--apply',action='store_true');p.add_argument('--revert',action='store_true')
    p.add_argument('--include-nvfp4',action='store_true',help='Enable NVFP4 launch after its operator checks pass')
    p.add_argument('--include-fp8-scale-placement',action='store_true',help='Enable operator-validated FP8 reader candidate for separate model validation')
    args=p.parse_args();assert not(args.apply and args.revert)
    if args.apply or args.revert:
        from minimal_inference import IDENTITY,owned_snapshot,matches,port_open
        assert not port_open(),'Stop owned serving before changing source'
        if IDENTITY.exists():assert not any(matches(x) for x in owned_snapshot(json.loads(IDENTITY.read_text())))
    manifest=ROOT/'patches/qsa-quant-manifest.json'
    previous=json.loads(manifest.read_text()) if manifest.exists() else None
    previous_files={row['path']:row for row in previous['files']} if previous else {}
    changes=[]
    for rel,expected in SOURCES.items():
        target=args.site/rel;backup=ROOT/'patches/qsa-quant-original'/rel
        current=target.read_bytes()
        original=original_source(args.site,rel)
        patched=patched_source(original,rel)
        allowed={expected,sha(patched)}
        if rel in previous_files:
            assert previous_files[rel]['original_sha256']==expected
            allowed.add(previous_files[rel]['patched_sha256'])
        assert sha(current) in allowed,'Current QSA source is not the expected base or recorded patch: '+rel
        row=dict(path=rel,original_sha256=expected,patched_sha256=sha(patched))
        changes.append((target,backup,original,patched,row))
    if args.apply or args.revert:
        for target,backup,original,patched,row in changes:
            backup.parent.mkdir(parents=True,exist_ok=True)
            if backup.exists():assert sha(backup.read_bytes())==row['original_sha256']
            else:backup.write_bytes(original)
            temp=target.with_name(target.name+'.arc3-tmp');temp.write_bytes(original if args.revert else patched)
            os.replace(temp,target)
        if args.revert:manifest.unlink(missing_ok=True)
        else:manifest.write_text(json.dumps(dict(patch='qsa-quantized-main-v2',files=[c[4] for c in changes],
            helpers={name:sha((ROOT/name).read_bytes()) for name in HELPERS+(['arc3_qsa_fp8_scale_placement.py'] if args.include_fp8_scale_placement else [])},
            fp8_readers=['baseline']+(['scale_placement'] if args.include_fp8_scale_placement else []),
            formats=['fp8_e4m3','fp8_e5m2']+(['nvfp4'] if args.include_nvfp4 else []),
            scope='Main QSA K/V only; scales included in pages; unchanged indexer and GDN'),indent=2)+'\n')
    print(json.dumps(dict(applied=args.apply,reverted=args.revert,files=[c[4] for c in changes]),indent=2))


if __name__=='__main__':main()
