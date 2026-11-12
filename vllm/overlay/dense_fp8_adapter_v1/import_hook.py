"""Install dense_fp8 as the OUTERMOST wrapper of the serving linear (and embedding) method classes.

Unlike dense_projection_adapter_v1's finder (which resolves the module itself), this finder delegates to the next
finders in sys.meta_path, so any earlier-installed loader wrappers (the dense adapter's) still run first; our install
then wraps whatever they left on the class. Result: dense_fp8.apply -> dense_projection.apply -> original.
"""
import hashlib
import importlib.abc
import os
import sys

LINEAR = 'vllm.model_executor.layers.linear'
EMBEDDING = 'vllm.model_executor.layers.vocab_parallel_embedding'
LINEAR_SHA = 'fc4e96ba5e36895513b801b99e746e2b07959d402e36d56aae7d8ad3f124cfe7'   # same pin as dense_projection_adapter_v1


class Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        m = getattr(self.original, 'create_module', None)
        return m(spec) if m else None

    def exec_module(self, module):
        if module.__name__ == LINEAR:
            raw = open(module.__spec__.origin, 'rb').read()
            assert hashlib.sha256(raw).hexdigest() == LINEAR_SHA, 'serving linear module changed; re-pin'
        self.original.exec_module(module)
        from .runtime import install, install_lm_head
        if module.__name__ == LINEAR:
            install(module)
        else:
            install_lm_head(module)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in (LINEAR, EMBEDDING):
            return None
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


def install():
    assert not {LINEAR, EMBEDDING}.intersection(sys.modules), 'dense-fp8 hook must precede vLLM imports'
    assert not any(isinstance(f, Finder) for f in sys.meta_path)
    sys.meta_path.insert(0, Finder())
