"""stage2_v1 'loadskip' part (milestone 2 A10 startup, 2026-10-01): each startup reader reads only the checkpoint tensors
it can use. Exact (the same tensors reach the same load_weights).

Measured (v2 and m2a launches): DefaultModelLoader streams every file of the index and filters by name, so the GPU
worker's target load and the PLE offload worker each read the whole 123.6 GiB checkpoint, concurrently on the FUSE
volume (target 190 s; PLE worker about 4 min, now the critical path).
  - PLE offload worker (_load_weights): keep files with a '.ple.ple_embedding.' tensor (the n-gram table and its
    buffers): shard 1 and model-fp8-mtp-ple, 56.8 GB.
  - GPU worker, target model (the model path, only with a drafter-only folder, so the drafter reads its own folder):
    skip the 'mtp.*' tensors and the offloaded n-gram table rows ('...ngram_embedding.shard_<i>.weight';
    PleOffloadLayer constructors are skipped in the GPU worker, so those tensors have no parameter). A file is dropped
    when all its tensors are skipped; in a kept file the skipped tensors are passed over before they are read
    (weight_utils.should_skip_weight, scoped to that iterator).
Anything else (other models, other roles) is unchanged. Dropped files and skipped tensors are recorded in the receipt.

Fix (06:10; W1 `w4a1e` failed): the table's global FP8 scale ('ngram_embedding.weight_scale') lives in the same file as
the 51 GB table, and the GPU worker needs it. It is kept as '_offload_weight_scale' and sizes the PLE output buffer
(get_offload_output_dtype): without it the buffer was BF16 against the CPU worker's FP8 table, and the first lookup
raised in index_select. The first version dropped files; this one skips tensors, so the scale is read."""
import contextvars
import json
import os

PINS = {'vllm.model_executor.model_loader.default_loader': '9c9d54b1b650bf5affc924ebb8b8c73711e187ae7486cadd9f737bb1279cc7b8',
        'vllm.model_executor.model_loader.weight_utils': '25bb3393105f9a642b7fa58bc277865db9976748be320787f8db2e03b16fd667',
        'vllm.v1.ple_offload.worker': 'f93a0a8c40ee3c536e60cb027f5d6e76cde0a51e62b540c82e2c15b2a72b7f5f'}
STATS = dict(calls=0, filtered=[], tensors_skipped=0)
_ROLE = contextvars.ContextVar('arc3_loadskip_role', default=None)
_ACTIVE = []                                     # skip sets of the target iterators being consumed
PLE = '.ple.ple_embedding.'
TABLE = '.ple.ple_embedding.ngram_embedding.'


def _target_path():
    return os.path.realpath(os.environ.get('ARC3_MODEL', '/workspace/models/Qwen3.8-Flash-Next-NVFP4'))


def target_skips(name):
    """Tensors the GPU worker's target load cannot use: the drafter's and the offloaded table rows (not its scale)."""
    return name.startswith('mtp.') or (TABLE in name and name.split(TABLE, 1)[1].startswith('shard_'))


def plan(files, folder, role):
    """(files kept, tensor names skipped in the kept files) for a role ('ple' or 'target'), from the folder's index."""
    wm = json.load(open(os.path.join(folder, 'model.safetensors.index.json')))['weight_map']
    names = {}
    for k, f in wm.items():
        names.setdefault(f, []).append(k)
    keep, skip = [], set()
    for path in files:
        ks = names.get(os.path.basename(path))
        if ks is None:                                   # not in the index: leave it to the stock behaviour
            keep.append(path)
        elif role == 'ple' and any(PLE in k for k in ks):
            keep.append(path)
        elif role == 'target' and not all(target_skips(k) for k in ks):
            keep.append(path)
            skip.update(k for k in ks if target_skips(k))
    return keep, skip


def select(files, folder, role):
    return plan(files, folder, role)[0]


def _scoped(iterator, skip):
    _ACTIVE.append(skip)
    try:
        yield from iterator
    finally:
        _ACTIVE.remove(skip)
        from . import install
        install.save()


def patch_default_loader(mod):
    cls = mod.DefaultModelLoader
    original = cls._prepare_weights
    original_iterator = cls._get_weights_iterator

    def _prepare_weights(self, model_name_or_path, *args, **kwargs):
        hf_folder, files, use_safetensors = original(self, model_name_or_path, *args, **kwargs)
        self._arc3_loadskip_skip = None
        role = _ROLE.get()
        if role is None and os.environ.get('ARC3_S2_DRAFT_MODEL') and \
                os.path.realpath(str(model_name_or_path)) == _target_path():
            role = 'target'
        if role and use_safetensors and os.path.exists(os.path.join(hf_folder, 'model.safetensors.index.json')):
            keep, skip = plan(files, hf_folder, role)
            assert keep, (role, hf_folder)
            STATS['calls'] += 1
            STATS['filtered'].append(dict(pid=os.getpid(), role=role, kept=len(keep), of=len(files),
                                          dropped=sorted(os.path.basename(f) for f in files if f not in keep),
                                          skip_in_kept=len(skip)))
            from . import install
            install.save()
            files = keep
            self._arc3_loadskip_skip = skip or None
        return hf_folder, files, use_safetensors

    def _get_weights_iterator(self, source):
        iterator = original_iterator(self, source)       # calls _prepare_weights now; its tensors are read lazily
        skip = self.__dict__.pop('_arc3_loadskip_skip', None)
        return _scoped(iterator, skip) if skip else iterator

    _prepare_weights.__wrapped__ = original
    _get_weights_iterator.__wrapped__ = original_iterator
    cls._prepare_weights = _prepare_weights
    cls._get_weights_iterator = _get_weights_iterator


def patch_weight_utils(mod):
    original = mod.should_skip_weight                    # the iterators' global: checked before each tensor is read

    def should_skip_weight(name, *args, **kwargs):
        if _ACTIVE and any(name in s for s in _ACTIVE):
            STATS['tensors_skipped'] += 1
            return True
        return original(name, *args, **kwargs)

    should_skip_weight.__wrapped__ = original
    mod.should_skip_weight = should_skip_weight


def patch_ple_worker(mod):
    owners = [c for c in vars(mod).values() if isinstance(c, type) and '_load_weights' in vars(c)]
    assert len(owners) == 1, [c.__name__ for c in owners]
    cls = owners[0]
    original = cls._load_weights

    def _load_weights(self, *args, **kwargs):
        token = _ROLE.set('ple')
        try:
            return original(self, *args, **kwargs)
        finally:
            _ROLE.reset(token)

    _load_weights.__wrapped__ = original
    cls._load_weights = _load_weights


PATCHERS = {'vllm.model_executor.model_loader.weight_utils': patch_weight_utils,
            'vllm.model_executor.model_loader.default_loader': patch_default_loader,
            'vllm.v1.ple_offload.worker': patch_ple_worker}
