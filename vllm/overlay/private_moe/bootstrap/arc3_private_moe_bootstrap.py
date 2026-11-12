"""Only two exact source modules are intercepted; original package resolution stays."""
import hashlib
import importlib.abc
import importlib.machinery
import json
from pathlib import Path
import sys


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def settings(path):
    path = Path(path).resolve(strict=True)
    root = path.parent
    require(path.name == 'settings.json', 'Expected immutable settings.json')
    manifest = json.loads((root / 'MANIFEST.json').read_text())
    require('settings.json' in manifest and 'arc3_private_moe_adapter.py' in manifest, 'Incomplete adapter manifest')
    for relative, expected in manifest.items():
        source = (root / relative).resolve(strict=True)
        require(source.is_relative_to(root) and digest(source) == expected, 'Adapter source changed: ' + relative)
    value = json.loads(path.read_text())
    require(value['schema'] == 'arc3.private-moe-serving.settings.v1' and value['mode'] == 3,
            'Only reviewed combined-mode serving candidate')
    require(set(value['targets']) == {'flashinfer.fused_moe.core',
                                    'vllm.model_executor.warmup.kernel_warmup'}, 'Wrong import targets')
    for fullname, row in value['targets'].items():
        source = Path(row['path'])
        require(source.is_absolute() and digest(source) == row['sha256'], 'Runtime source changed: ' + fullname)
    return value, path


class Loader(importlib.machinery.SourceFileLoader):
    def __init__(self, name, path, finder):
        super().__init__(name, str(path))
        self.finder = finder

    def get_code(self, fullname):
        require(fullname == self.name, 'Loader name differs')
        raw = Path(self.path).read_bytes()
        require(hashlib.sha256(raw).hexdigest() == self.finder.settings['targets'][fullname]['sha256'],
                'Runtime module changed after admission')
        return compile(raw, self.path, 'exec', dont_inherit=True)

    def exec_module(self, module):
        try:
            super().exec_module(module)
            state = self.finder.state
            if self.name == 'flashinfer.fused_moe.core':
                state.core_ready(module)
            else:
                state.wrap_warmup(module)
            self.finder.pending.remove(self.name)
            if not self.finder.pending:
                sys.meta_path.remove(self.finder)
        except BaseException as error:
            self.finder.state.record('failed', error=type(error).__name__ + ': ' + str(error),
                                     process_must_exit=True)
            raise


class Finder(importlib.abc.MetaPathFinder):
    def __init__(self, config, state):
        self.settings, self.state = config, state
        self.pending = set(config['targets'])

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in self.settings['targets']:
            return None
        require(target is None and fullname in self.pending, 'Repeated/reloaded module is unsupported')
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        require(spec is not None and isinstance(spec.loader, importlib.machinery.SourceFileLoader),
                'Expected original source loader')
        actual = Path(spec.origin).resolve(strict=True)
        require(actual == Path(self.settings['targets'][fullname]['path']).resolve(strict=True),
                'A different package shadows target module')
        spec.loader = Loader(fullname, actual, self)
        return spec


def install(path):
    config, path = settings(path)
    require(not set(config['targets']).intersection(sys.modules), 'Target was imported before startup adapter')
    require(not any(isinstance(item, Finder) for item in sys.meta_path), 'Adapter already installed')
    # This module is found via the explicit adapter PYTHONPATH entry.
    import arc3_private_moe_adapter as adapter
    require(Path(adapter.__file__).resolve() == path.parent / 'arc3_private_moe_adapter.py', 'Adapter namespace shadowed')
    state = adapter.initialized(config, path)
    finder = Finder(config, state)
    sys.meta_path.insert(0, finder)
    return finder
