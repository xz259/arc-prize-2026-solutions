"""Delegating meta-path finder: pins the three PLE offload modules and installs the discriminator after they execute.
Delegates to the later finders, so earlier loader wrappers (dense adapters) keep running; the offload subprocess is a
fresh spawn interpreter with the same PYTHONPATH, so sitecustomize installs this hook there too."""
import hashlib
import importlib.abc
import sys
import threading

from .runtime import MODE, PINS, REPORT, install_for, write_receipt

TARGETS = {
    'lookup': ('vllm.v1.ple_offload.worker',),
    'handoff': ('vllm.model_executor.layers.ple_offload_layer', 'vllm.v1.ple_offload.connector'),
    'nowait': ('vllm.model_executor.layers.ple_offload_layer',),
    'timing': ('vllm.v1.ple_offload.connector', 'vllm.v1.ple_offload.worker'),
    'nogpu': ('vllm.model_executor.layers.ple_offload_layer', 'vllm.v1.ple_offload.worker'),
}


class Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        m = getattr(self.original, 'create_module', None)
        return m(spec) if m else None

    def exec_module(self, module):
        raw = open(module.__spec__.origin, 'rb').read()
        digest = hashlib.sha256(raw).hexdigest()
        assert digest == PINS[module.__name__], (module.__name__, digest)
        self.original.exec_module(module)
        try:
            install_for(module.__name__, module)
        except BaseException as e:
            REPORT['errors'].append(f'{module.__name__}: {e!r}')
            write_receipt()
            raise


class Finder(importlib.abc.MetaPathFinder):
    """Re-entrancy-safe: ple_copy_handoff_v1's finder (combined bootstrap) scans every other finder, including this
    one when it sits earlier in sys.meta_path; a nested call for the same module returns None so that finder falls
    through to the path finder, and this finder then wraps the loader it returns (our install runs last/outermost)."""
    def __init__(self, names):
        self.names = set(names)
        self._local = threading.local()

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.names:
            return None
        active = getattr(self._local, 'active', None)
        if active is None:
            active = self._local.active = set()
        if fullname in active:
            return None
        active.add(fullname)
        try:
            return self._find(fullname, path, target)
        finally:
            active.discard(fullname)

    def _find(self, fullname, path, target):
        started = False
        for finder in list(sys.meta_path):
            if finder is self:
                started = True
                continue
            if not started or not hasattr(finder, 'find_spec'):
                continue
            spec = finder.find_spec(fullname, path, target)
            if spec is not None:
                assert spec.loader is not None
                spec.loader = Loader(spec.loader)
                return spec
        return None


def install():
    assert MODE != 'off'
    names = TARGETS[MODE]
    assert not set(names).intersection(sys.modules), 'ple-discriminator hook must precede vLLM imports'
    assert not any(isinstance(f, Finder) for f in sys.meta_path)
    sys.meta_path.insert(0, Finder(names))
    REPORT['hook'] = sorted(names)
    write_receipt()
