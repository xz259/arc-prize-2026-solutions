"""stage2_v1 memopt: memory parts without numeric effect (stage-2 memory optimisation, 2026-09-29).

  ctxfree    glue_fusion_v1 preallocates its split-K partial/counter buffers on CUDA when vLLM's linear module is
             imported (runtime.install_linear -> kernels.preallocate / kernels_hcf.preallocate), so every process
             that imports that module creates a CUDA context: the API server, the EngineCore and the PLE offload
             helper each hold 678 MiB on this GPU (cuda-init receipts, memdiag). The preallocation is deferred until
             the process has initialised CUDA; the GPU worker runs it right after Worker.init_device, i.e. before
             model loading, compilation and graph capture, so the buffers are the same real tensors as before.
  topkshare  Every QSA owner (12 target layers, the MTP layer) registers its own top-k index buffer
             [max_num_batched_tokens, output_width] int32, 64 MB each. The layer's indexer fills it and the same
             layer's attention consumes it inside the layer call (the eviction remap and the frequency counters run in
             that call too); the MTP drafter reuses its rows across its own consecutive draft steps. One buffer per
             shape serves every owner.

  ctxfree also covers vLLM's bundled FLA ops: flash_linear_attention/ops/utils.py detects the platform at import
             through triton's driver target and torch.cuda device queries, which creates a context in the API server
             and the PLE helper (they import the model module). On CUDA the source is executed with vLLM's NVML-based
             current_platform answers instead (same values: backend 'cuda', name and capability of device 0).
  Result (20:47): the API server and the EngineCore stay CUDA-free (-1.3 GiB). The PLE offload helper needs CUDA
             regardless (its registration receives CUDA tensors by IPC; the GDN constructor queries the current device)
             and keeps its context (746 MiB). expandable_segments:True breaks that IPC here (pidfd_getfd not permitted).

Every patched module is pinned to its source hash (2026-09-29)."""
import os

PINS = {
    'glue_fusion_v1.kernels': 'be5c403461d5ff8cc67d38e96fcc9518795133587c784531e131281e693e5d9a',
    'glue_fusion_v1.kernels_hcf': '0818b0acb6909ee171d782b617b476d612e3c43d70d76c9d8bb70d5906159253',
    'vllm.v1.worker.gpu_worker': '923767cea120a027ca36683bd48b7659eadbfa4cbd4acbcf52a80cee0c0a0ec4',
    'vllm.models.qwen3_8_flash_next.nvidia.qsa': 'a75b7b0fb72761de2fb246ee28dd685054fe6f85112387c7c91efa3c65db52a6',
    'vllm.third_party.flash_linear_attention.ops.utils': 'd6c90d36505e02086954c23ca4dd2fea82a69416d68ff1d0501e16b3148b5b0c',
    'vllm.model_executor.warmup.flashinfer_autotune_cache': 'fcdaf1135e3d9e2be619f423c80c3288397470e85823fc6315b36598ad3297b7',
}
STATS = dict(pid=os.getpid(), prealloc_deferred=0, prealloc_immediate=0, prealloc_ran_deferred=0,
             topk_owners=0, topk_buffers=0, topk_saved_bytes=0)
_DEFERRED = []


# ---------------------------------------------------------------- ctxfree
def patch_glue_kernels(mod):
    original = mod.preallocate

    def preallocate(device):
        import torch
        if torch.cuda.is_initialized():
            STATS['prealloc_immediate'] += 1
            return original(device)
        _DEFERRED.append((original, device))
        STATS['prealloc_deferred'] += 1

    mod.preallocate = preallocate


def run_deferred():
    while _DEFERRED:
        fn, device = _DEFERRED.pop(0)
        fn(device)
        STATS['prealloc_ran_deferred'] += 1
    from .install import save
    save()


def patch_gpu_worker(mod):
    original = mod.Worker.init_device

    def init_device(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        run_deferred()
        return out

    mod.Worker.init_device = init_device


def transform_fla_utils(src):
    reps = [('        return triton.runtime.driver.active.get_current_target().backend\n',
             '        if current_platform.is_cuda():   # stage2_v1 ctxfree: NVML platform, no CUDA context at import\n'
             '            return "cuda"\n'
             '        return triton.runtime.driver.active.get_current_target().backend\n'),
            ('    "NVIDIA H" in torch.cuda.get_device_name(0)\n    or torch.cuda.get_device_capability()[0] >= 9\n',
             '    "NVIDIA H" in current_platform.get_device_name(0)   # stage2_v1 ctxfree: NVML, no CUDA context\n'
             '    or current_platform.get_device_capability(0)[0] >= 9\n')]
    for a, b in reps:
        assert src.count(a) == 1, a
        src = src.replace(a, b)
    STATS['fla_utils_source'] = True
    return src


# ---------------------------------------------------------------- autotune seed (env ARC3_S2_AUTOTUNE_SEED)
def patch_autotune_cache(mod):
    """The FlashInfer autotune cache is keyed by the whole vLLM config hash, so every new (N, KV, parts) configuration
    re-tunes the MoE tactics inside the 8,192-token warmup (extra profiling memory at startup: the KV 14 GiB launch
    ran out of memory there) and picks its own winners (production's and L2m's files agree on 20 of 42 buckets).
    With ARC3_S2_AUTOTUNE_SEED=<autotune_configs.json> that file is installed as this config's cache before the
    warmup reads it: every bucket is cached (no profiling) and the tactics equal the seed's (production's).
    10-16 (two identical servers died at warmup on an empty file): the install is atomic and conditional. The seed's bytes are
    validated (JSON with _metadata and at least one bucket), and written only when the cache file is absent or differs, through a
    temp file + os.replace. A concurrent reader (another server with the same config hash) then always sees a whole file, and a
    corrupt leftover is replaced. Never truncate the cache file in place."""
    original = mod.resolve_flashinfer_autotune_file

    def resolve_flashinfer_autotune_file(runner):
        path = original(runner)
        seed = os.environ.get('ARC3_S2_AUTOTUNE_SEED')
        if seed:
            installed, entries = install_autotune_seed(seed, path)
            STATS['autotune_seeded'] = dict(seed=seed, file=str(path), installed=installed, entries=entries)
        return path

    mod.resolve_flashinfer_autotune_file = resolve_flashinfer_autotune_file


def install_autotune_seed(seed, path):
    """(installed, entries): the seed file's bytes at `path`, atomically, only if `path` is absent or holds other bytes. Raises if
    the seed is not a FlashInfer autotune dict (JSON with _metadata and at least one tactic bucket)."""
    import json
    import tempfile
    from pathlib import Path
    data = Path(seed).read_bytes()
    parsed = json.loads(data)
    if not isinstance(parsed, dict) or '_metadata' not in parsed or len(parsed) < 2:
        raise RuntimeError('memopt: autotune seed %s is not a FlashInfer autotune file' % seed)
    path = Path(path)
    try:
        if path.read_bytes() == data:
            return False, len(parsed)
    except FileNotFoundError:
        pass
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix='.%s.' % path.name, suffix='.seed')
    try:
        with os.fdopen(fd, 'wb') as fh:
            fh.write(data)
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    return True, len(parsed)


# ---------------------------------------------------------------- topkshare
def patch_qsa(mod):
    cls = mod.Qwen3_8FlashNextQSAAttention
    original = cls.register_buffer
    shared = {}

    def register_buffer(self, name, tensor, persistent=True):
        if name == 'topk_indices_buffer' and tensor is not None:
            key = (tuple(tensor.shape), tensor.dtype, str(tensor.device))
            buf = shared.get(key)
            if buf is None:
                buf = shared[key] = tensor
                STATS['topk_buffers'] += 1
            else:
                STATS['topk_saved_bytes'] += tensor.numel() * tensor.element_size()
            STATS['topk_owners'] += 1
            tensor = buf
        return original(self, name, tensor, persistent)

    cls.register_buffer = register_buffer


PATCHERS = {'ctxfree': {'glue_fusion_v1.kernels': patch_glue_kernels, 'glue_fusion_v1.kernels_hcf': patch_glue_kernels,
                        'vllm.v1.worker.gpu_worker': patch_gpu_worker,
                        'vllm.third_party.flash_linear_attention.ops.utils': ('source', transform_fla_utils)},
            'topkshare': {'vllm.models.qwen3_8_flash_next.nvidia.qsa': patch_qsa}}
