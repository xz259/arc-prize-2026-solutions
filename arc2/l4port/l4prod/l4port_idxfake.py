"""L4 port (2026-10-31, ARC2 - main; lever 4, the indexer dedupe): a MEASUREMENT-ONLY hook, never for serving. With
ARC3_L4PORT_IDX_FAKEHALF=1 the QSA indexer's compressed-key cache spec reports half its row bytes (head_size 128 -> 64), as a
TP-2 parity split of the compressed rows would. The server sizes its KV pool and logs 'GPU KV cache size: N tokens' before
CUDA-graph capture, and capture then fails on the shape mismatch: the token count against the same config without the hook says
whether vLLM's hybrid allocator turns the halved owner into KV capacity at all (page-size unification may pad it away) and how
many lanes the real dedupe could add, before any kernel work. Post-import hook on the model's common.qsa_cache module (no source
edit); one receipt line per process."""
import importlib.abc
import os
import sys

NAME = "vllm.models.qwen3_8_flash_next.common.qsa_cache"
ON = os.environ.get("ARC3_L4PORT_IDX_FAKEHALF") == "1"
STATE = dict(printed=False)


def _patch(module):
    cls = module.QSACompressedKeyCache
    orig = cls.get_kv_cache_spec

    def get_kv_cache_spec(self, vllm_config):
        spec = orig(self, vllm_config)
        half = module.MLAAttentionSpec(block_size=spec.block_size, num_kv_heads=spec.num_kv_heads, head_size=spec.head_size // 2,
                                       dtype=spec.dtype, compress_ratio=spec.compress_ratio)
        if not STATE["printed"]:
            STATE["printed"] = True
            try:
                pb = "page bytes %d -> %d" % (spec.page_size_bytes, half.page_size_bytes)
            except Exception as e:      # noqa: BLE001 (the receipt only)
                pb = "page bytes ? (%r)" % (e,)
            print("[l4port] idxfake: compressed-key cache spec head_size %d -> %d (%s; measurement only)" % (
                spec.head_size, half.head_size, pb), flush=True)
        return half

    cls.get_kv_cache_spec = get_kv_cache_spec


class _Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname != NAME:
            return None
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


def install():
    if not ON:
        return
    if NAME in sys.modules:
        _patch(sys.modules[NAME])
    else:
        sys.meta_path.insert(0, _Finder())
