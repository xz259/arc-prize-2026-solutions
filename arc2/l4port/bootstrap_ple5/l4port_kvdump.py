"""L4 port diagnostic (2026-10-30, ARC2 - main; env ARC3_L4PORT_KVDUMP=1): prints the KV cache layout vLLM derives, to explain why
MTP under TP2 x PP2 cut the 60K lanes from 20.70 to 7.00 on the 4090 (1,244,872 -> 420,896 tokens; TP=4 loses only 8%). Wraps
vllm.v1.core.kv_cache_utils.get_kv_cache_configs (engine core): per global group the spec class, layer count, block size, page
bytes, blocks per request at max_model_len; per worker num_blocks, tensor count and bytes, and each group's layers on that worker.
Read-only: the configs are returned unchanged."""
import importlib.abc
import os
import sys

NAME = "vllm.v1.core.kv_cache_utils"


def _patch(module):
    orig = module.get_kv_cache_configs

    def get_kv_cache_configs(vllm_config, kv_cache_specs, available_memory):
        out = orig(vllm_config, kv_cache_specs, available_memory)
        try:
            from vllm.utils.math_utils import cdiv
        except Exception:
            def cdiv(a, b):
                return -(-a // b)
        try:
            ml = vllm_config.model_config.max_model_len
            print("[l4port] kvdump: workers %d, max_model_len %d, available GiB %s" % (
                len(out), ml, [round(m / 2**30, 3) for m in available_memory]), flush=True)
            for w, (cfg, specs) in enumerate(zip(out, kv_cache_specs)):
                per_req = 0
                for gi, g in enumerate(cfg.kv_cache_groups):
                    s = g.kv_cache_spec
                    b = cdiv(s.max_memory_usage_bytes(vllm_config), s.page_size_bytes)
                    per_req += b
                    if w == 0 or gi < 12:
                        print("[l4port] kvdump: w%d g%d %s layers %d (here %d) block %s page %d B blocks/req %d sample %s" % (
                            w, gi, type(s).__name__, len(g.layer_names), sum(1 for n in g.layer_names if n in specs),
                            getattr(s, "block_size", "-"), s.page_size_bytes, b, ",".join(n.split(".")[-3] if n.count(".") >= 3 else n
                                                                                      for n in g.layer_names[:3])), flush=True)
                tb = sum(t.size for t in cfg.kv_cache_tensors)
                print("[l4port] kvdump: w%d num_blocks %d tensors %d bytes %.3f GiB, blocks/req %d -> concurrency %.2f" % (
                    w, cfg.num_blocks, len(cfg.kv_cache_tensors), tb / 2**30, per_req, cfg.num_blocks / max(per_req, 1)), flush=True)
        except Exception as e:
            print("[l4port] kvdump: failed: %r" % (e,), flush=True)
        return out

    get_kv_cache_configs.__wrapped__ = orig
    module.get_kv_cache_configs = get_kv_cache_configs


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
    if os.environ.get("ARC3_L4PORT_KVDUMP") != "1":
        return
    if NAME in sys.modules:
        _patch(sys.modules[NAME])
    else:
        sys.meta_path.insert(0, _Finder())
