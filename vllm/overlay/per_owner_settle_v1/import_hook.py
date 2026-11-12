"""Pinned re-entrancy-safe hook: patch one_lookahead (before its install() binds _can_schedule into the generated
queue) and the worker runner. Enabled by ARC3_PER_OWNER_SETTLE=1."""
import hashlib
import importlib.abc
import sys
import threading
from pathlib import Path

PINS = {
    'one_lookahead': 'e78460949251b7d805f6625c3640c99bda214530f4435dea0fc23b5ca9255bb2',
    'vllm.v1.worker.gpu.model_runner': 'f08ba0f5ad41c8b9d5145fb1fba115ffd97f35f1e59017602e235ec0823abce8',
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
        if module.__name__ == 'one_lookahead':
            runtime.install_one_lookahead(module)
        else:
            runtime.install_model_runner(module)

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
    assert not set(PINS).intersection(sys.modules), 'per-owner-settle hook must precede the serving imports'
    assert not any(isinstance(f, Finder) for f in sys.meta_path)
    sys.meta_path.insert(0, Finder())
