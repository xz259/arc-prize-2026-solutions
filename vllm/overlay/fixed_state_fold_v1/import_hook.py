"""Pinned, re-entrancy-safe import hook: after vllm.v1.core.kv_cache_utils executes, install the fixed-state fold
(ARC3_FIXED_STATE_FOLD = off | A). Receipts per process under ARC3_FIXED_STATE_FOLD_RECEIPTS."""
import hashlib
import importlib.abc
import json
import os
import sys
import threading
import time
from pathlib import Path

TARGET = 'vllm.v1.core.kv_cache_utils'
ORIGINAL_SHA = 'f6643bad29b7b3cb044482b6736a042afa91a4f43a11f034317a8e8029d3a8a4'
MODE = os.environ.get('ARC3_FIXED_STATE_FOLD', 'off')
REPORT = dict(version='arc3-fixed-state-fold-v1', mode=MODE, pid=os.getpid(), installed={}, folded=None, errors=[])


def receipt():
    root = os.environ.get('ARC3_FIXED_STATE_FOLD_RECEIPTS')
    if not root:
        return
    try:
        os.makedirs(root, exist_ok=True)
        REPORT['epoch'] = time.time()
        Path(root, f'proc-{os.getpid()}.json').write_text(json.dumps(REPORT, indent=1) + '\n')
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
        assert hashlib.sha256(raw).hexdigest() == ORIGINAL_SHA, 'kv_cache_utils changed; re-pin'
        self.original.exec_module(module)
        try:
            from . import fold
            fold.install(module, MODE, on_fold=_record_fold)
            REPORT['installed'][TARGET] = f'csa_linear grouping folded: mode {MODE}'
        except BaseException as e:
            REPORT['errors'].append(repr(e)); receipt(); raise
        receipt()

    def __getattr__(self, name):
        return getattr(self.original, name)


def _record_fold(summary):
    REPORT['folded'] = summary; receipt()


class Finder(importlib.abc.MetaPathFinder):
    def __init__(self):
        self._local = threading.local()

    def find_spec(self, fullname, path=None, target=None):
        if fullname != TARGET:
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
    assert MODE != 'off'
    assert TARGET not in sys.modules, 'fixed-state-fold hook must precede vLLM imports'
    assert not any(isinstance(f, Finder) for f in sys.meta_path)
    sys.meta_path.insert(0, Finder())
    REPORT['hook'] = [TARGET]; receipt()
