"""Pin and wrap only the serving linear module; compose with the MoE hooks."""
import hashlib
import importlib.abc
import importlib.machinery
import os
from pathlib import Path
import sys

TARGET = 'vllm.model_executor.layers.linear'
SOURCE_SHA = 'fc4e96ba5e36895513b801b99e746e2b07959d402e36d56aae7d8ad3f124cfe7'
AOT_TARGET = 'vllm.compilation.decorators'
AOT_SHA = 'dd7167637a478aeaea3f2410a727ddefc702bda1d9bf632831c2a8b4811ebb1e'
AOT_OLD = '''            cache_dir = os.path.join(
                envs.VLLM_CACHE_ROOT,
                "torch_compile_cache",
                "torch_aot_compile",
'''
AOT_NEW = AOT_OLD.replace('envs.VLLM_CACHE_ROOT', 'os.environ["ARC3_DENSE_AOT_ROOT"]')


def rewrite_aot(raw):
    assert hashlib.sha256(raw).hexdigest() == AOT_SHA
    source = raw.decode()
    assert source.count(AOT_OLD) == 1
    changed = source.replace(AOT_OLD, AOT_NEW)
    assert changed.replace(AOT_NEW, AOT_OLD) == source
    return changed


class Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        method = getattr(self.original, 'create_module', None)
        return method(spec) if method else None

    def exec_module(self, module):
        path = Path(module.__spec__.origin)
        raw = path.read_bytes()
        if module.__name__ == AOT_TARGET:
            root = Path(os.environ['ARC3_DENSE_AOT_ROOT'])
            assert root.is_absolute()
            changed = rewrite_aot(raw)
            exec(compile(changed, str(path), 'exec'), module.__dict__)
            from .runtime import REPORT, save
            REPORT['aot_cache'] = dict(root=str(root), original_sha256=AOT_SHA,
                executed_sha256=hashlib.sha256(changed.encode()).hexdigest(),
                native_autotune_namespace_unchanged=True)
            save()
        else:
            assert hashlib.sha256(raw).hexdigest() == SOURCE_SHA
            self.original.exec_module(module)
            from .runtime import install
            install(module)


class Finder(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname not in (TARGET, AOT_TARGET):
            return None
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        assert spec is not None and spec.loader is not None
        spec.loader = Loader(spec.loader)
        return spec


def install():
    assert not {TARGET,AOT_TARGET}.intersection(sys.modules), 'Dense hook must precede vLLM imports'
    assert not any(isinstance(f, Finder) for f in sys.meta_path)
    sys.meta_path.insert(0, Finder())
