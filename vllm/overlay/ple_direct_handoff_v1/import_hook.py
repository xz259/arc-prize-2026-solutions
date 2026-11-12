"""Source-pinned direct-handoff overlay: executes this package's connector.py in place of the runtime connector
module and patches the offload runner in the worker module. Re-entrancy-safe delegating finder (other overlays'
finders scan all of sys.meta_path). Selected by ARC3_PLE_HANDOFF_IMPL=direct in the combined bootstrap."""
import hashlib
import importlib.abc
import json
import os
import sys
import threading
import time
from pathlib import Path

CONNECTOR = 'vllm.v1.ple_offload.connector'
WORKER = 'vllm.v1.ple_offload.worker'
ORIGINAL_SHA = {
    CONNECTOR: 'b911d00cf4a39556d946e86ac34ea70e6230683b22adf5ef14ac299e9ef36206',
    WORKER: 'f93a0a8c40ee3c536e60cb027f5d6e76cde0a51e62b540c82e2c15b2a72b7f5f',
}
SOURCE = Path(__file__).with_name('connector.py')
REPORT = dict(version='arc3-ple-direct-handoff-v1', pid=os.getpid(), installed={}, source_sha256=None, errors=[])


def _receipt():
    root = os.environ.get('ARC3_PLE_DIRECT_RECEIPTS')
    if not root:
        return
    try:
        os.makedirs(root, exist_ok=True)
        REPORT['epoch'] = time.time()
        Path(root, f'worker-{os.getpid()}.json').write_text(json.dumps(REPORT, indent=1) + '\n')
    except Exception as e:
        REPORT['errors'].append(repr(e))


class Loader(importlib.abc.Loader):
    def __init__(self, original):
        self.original = original

    def create_module(self, spec):
        m = getattr(self.original, 'create_module', None)
        return m(spec) if m else None

    def exec_module(self, module):
        raw = Path(module.__spec__.origin).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == ORIGINAL_SHA[module.__name__], module.__name__
        try:
            if module.__name__ == CONNECTOR:
                src = SOURCE.read_bytes(); REPORT['source_sha256'] = hashlib.sha256(src).hexdigest()
                exec(compile(src, str(SOURCE), 'exec'), module.__dict__)
                REPORT['installed'][CONNECTOR] = 'replaced-by-direct-connector'
            else:
                self.original.exec_module(module)
                from . import worker_patch
                worker_patch.install(module)
                REPORT['installed'][WORKER] = 'runner-patched:poll-flag'
        except BaseException as e:
            REPORT['errors'].append(f'{module.__name__}: {e!r}'); _receipt(); raise
        _receipt()

    def __getattr__(self, name):
        return getattr(self.original, name)


class Finder(importlib.abc.MetaPathFinder):
    def __init__(self):
        self._local = threading.local()

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in ORIGINAL_SHA:
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
    assert not {CONNECTOR, WORKER}.intersection(sys.modules), 'ple-direct-handoff hook must precede vLLM imports'
    assert not any(isinstance(f, Finder) for f in sys.meta_path)
    assert Path(os.environ['ARC3_PLE_DIRECT_RECEIPTS']).is_absolute()
    sys.meta_path.insert(0, Finder())
    REPORT['hook'] = [CONNECTOR, WORKER]; _receipt()
