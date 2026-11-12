"""stage2_v1 'embedhost' part (milestone 2 outline A1, 2026-10-01): the target's input embedding table in pinned host
memory, read through a UVA view. Exact; frees the table's GPU memory (248,320 x 2,560 BF16 = 1.18 GiB).

The table is read only by row gathers:
  - the target's embed_input_ids, eager in the model runner (one row per scheduled token);
  - the MTP drafter, which shares the same module: vLLM's load_eagle_model replaces the drafter's own embed_tokens
    (and lm_head) with the target's, because Qwen3_8FlashNextMTP does not set has_own_embed_tokens (M2_DECISIONS 0.2).

After GPUModelRunner.load_model (target and drafter loaded, sharing done), the parameter's storage moves to pinned
host memory. The parameter becomes a UVA view of it, as vLLM's UVA offloader does
(vllm.utils.torch_utils.get_accelerator_view_from_cpu_tensor). Every gather then reads the same BF16 rows over PCIe,
so outputs are bit-identical. The whole table is compared chunk by chunk before the swap, and the GPU copy is released
at once (empty_cache), before profile_run can split the freed segment.

Cost: a 5 KB row read over PCIe per gathered token. A decode step at 41 lanes gathers about 164 target rows and
3 x 41 drafter rows; a 4,096-token prefill chunk gathers 20 MB. test_embedhost.py measures it."""
import gc
import time

PINS = {'vllm.v1.worker.gpu.model_runner': 'f08ba0f5ad41c8b9d5145fb1fba115ffd97f35f1e59017602e235ec0823abce8'}
STATS = dict(moved=0, bytes=0, rows=0, shared_with_drafter=None, allocated_freed=None, device_free_gain=None,
             verify_chunks=0, seconds=None)
_KEEP = []          # the pinned host tables (the UVA views do not own them)
CHUNK_ROWS = 16384  # verification chunk: about 80 MB of rows and a 40 MB comparison mask on the GPU


def host_view(param, chunk_rows=CHUNK_ROWS):
    """Move a CUDA parameter's storage to pinned host memory and point the parameter at a UVA view of it.
    The parameter object is kept (every module holding it sees the new storage). Returns (host, chunks verified)."""
    import torch
    from vllm.utils.torch_utils import get_accelerator_view_from_cpu_tensor
    w = param.data
    assert w.is_cuda and w.is_contiguous(), (w.device, w.is_contiguous())
    host = torch.empty(w.shape, dtype=w.dtype, device='cpu', pin_memory=True)
    host.copy_(w)
    torch.cuda.synchronize()
    view = get_accelerator_view_from_cpu_tensor(host)
    assert view.is_cuda and view.shape == w.shape and view.dtype == w.dtype and view.data_ptr() != w.data_ptr()
    chunks = 0
    for s in range(0, w.shape[0], chunk_rows):          # the whole table, through the UVA view, before the swap
        assert torch.equal(view[s:s + chunk_rows], w[s:s + chunk_rows]), f'embedhost: UVA rows {s}+ differ'
        chunks += 1
    _KEEP.append(host)
    param.data = view
    return host, chunks


def move_embedding(runner):
    import torch
    t0 = time.perf_counter()
    model = runner.model
    lm = model.get_language_model() if hasattr(model, 'get_language_model') else model
    emb = lm.model.embed_tokens
    assert type(emb).__name__ == 'VocabParallelEmbedding', type(emb).__name__
    w = emb.weight
    assert w.dtype == torch.bfloat16 and w.dim() == 2 and w.is_cuda, (w.dtype, tuple(w.shape), w.device)
    speculator = getattr(runner, 'speculator', None)
    draft = getattr(getattr(speculator, 'model', None), 'model', None)
    if draft is not None:
        assert getattr(draft, 'embed_tokens', None) is emb, 'embedhost: the drafter does not share the target embedding'
    torch.cuda.synchronize()
    alloc0, free0 = torch.cuda.memory_allocated(), torch.cuda.mem_get_info()[0]
    host, chunks = host_view(w)
    emb._arc3_host_weight = host
    del w
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    alloc1, free1 = torch.cuda.memory_allocated(), torch.cuda.mem_get_info()[0]
    nbytes = host.numel() * host.element_size()
    STATS.update(moved=1, bytes=nbytes, rows=int(host.shape[0]), shared_with_drafter=draft is not None,
                 allocated_freed=alloc0 - alloc1, device_free_gain=free1 - free0, verify_chunks=chunks,
                 seconds=round(time.perf_counter() - t0, 3))
    assert alloc0 - alloc1 >= nbytes, f'embedhost: only {alloc0 - alloc1} of {nbytes} bytes freed (another reference?)'


def patch_model_runner(mod):
    cls = mod.GPUModelRunner
    original = cls.load_model

    def load_model(self, *args, **kwargs):
        out = original(self, *args, **kwargs)
        if not kwargs.get('load_dummy_weights', args[0] if args else False):
            move_embedding(self)
            from . import install
            install.save()
        return out

    load_model.__wrapped__ = original
    cls.load_model = load_model


PATCHERS = {'vllm.v1.worker.gpu.model_runner': patch_model_runner}
