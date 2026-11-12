"""Pinned re-entrancy-safe hook: patch the hyper-connection module and the GDN layer at import time.
Enabled by ARC3_GLUE_FUSION != 0 (parts hc, gemm, gdn or all)."""
import hashlib
import importlib.abc
import sys
import threading
from pathlib import Path

PINS = {
    'vllm.models.qwen3_8_flash_next.nvidia.hyperconnection': '6457124bb3710630bae671848c50864b1d0fc9217135bcec955659e0a0e38f28',
    'vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn': '81b4dcd0952492375c93bffc2cdf45f10b45ab5e117f2e1d949a147d144e64f0',
    'vllm.model_executor.layers.linear': 'fc4e96ba5e36895513b801b99e746e2b07959d402e36d56aae7d8ad3f124cfe7',
}


class Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        m = getattr(self.original, 'create_module', None)
        return m(spec) if m else None

    def exec_module(self, module):
        raw = Path(module.__spec__.origin).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == PINS[module.__name__], module.__name__
        self.original.exec_module(module)
        from . import runtime
        if module.__name__.endswith('hyperconnection'):
            runtime.install_hyperconnection(module)
        elif module.__name__.endswith('.linear'):
            runtime.install_linear(module)
        else:
            runtime.install_gdn(module)

    def __getattr__(self, name):
        return getattr(self.original, name)


class Finder(importlib.abc.MetaPathFinder):
    def __init__(self):
        self._local = threading.local()

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in PINS:
            return None
        active = getattr(self._local, 'active', None)
        if active is None:
            active = self._local.active = set()
        if fullname in active:
            return None
        active.add(fullname)
        try:
            started = False
            for finder in list(sys.meta_path):
                if finder is self:
                    started = True; continue
                if not started or not hasattr(finder, 'find_spec'):
                    continue
                spec = finder.find_spec(fullname, path, target)
                if spec is not None:
                    assert spec.loader is not None
                    spec.loader = Loader(spec.loader)
                    return spec
            return None
        finally:
            active.discard(fullname)


def install():
    assert not set(PINS).intersection(sys.modules), 'glue-fusion hook must precede the serving imports'
    assert not any(isinstance(f, Finder) for f in sys.meta_path)
    sys.meta_path.insert(0, Finder())
