"""stage2_v1 'splitfix' part (milestone 2 outline A4, port of vLLM #58430, 2026-10-01): the worker's profile_run under
the allocator split limit. Exact (allocator placement only); records allocator state around it.

Our snapshot scopes max_split_size_mb=20 around load_model only (gpu_worker.py:555-561, _scoped_allocator_max_split).
profile_run, which still runs with a pinned KV size (determine_available_memory, 604-607), is unscoped: its large
transient activations may split cached segments that later hold live tensors, leaving 'inactive split' bytes no
empty_cache can return (the 09-29 record saw 0.79 GiB; production v2 at rest held 1.26 GiB, M2_DECISIONS 0.4).
Upstream's fix scopes profile_run the same way. This part does that and records memory_stats before and after
profile_run (allocated, reserved, inactive split, segments, device free) in the receipt, so the effect is measured,
not assumed."""

PINS = {'vllm.v1.worker.gpu_worker': '923767cea120a027ca36683bd48b7659eadbfa4cbd4acbcf52a80cee0c0a0ec4'}
STATS = dict(profile_runs=0, before=None, after=None)


def allocator_state():
    import torch
    if not torch.cuda.is_available():
        return None
    torch.cuda.synchronize()
    s = torch.cuda.memory_stats()
    return dict(allocated=s.get('allocated_bytes.all.current', 0), reserved=s.get('reserved_bytes.all.current', 0),
                inactive_split=s.get('inactive_split_bytes.all.current', 0), segments=s.get('segment.all.current', 0),
                device_free=torch.cuda.mem_get_info()[0])


def patch_gpu_worker(mod):
    cls = mod.Worker
    original = cls.determine_available_memory

    def determine_available_memory(self):
        runner = self.model_runner
        stock = runner.profile_run

        def profile_run(*args, **kwargs):
            STATS['before'] = allocator_state()
            with self._scoped_allocator_max_split(max_split_size_mb=20):
                out = stock(*args, **kwargs)
            STATS['after'] = allocator_state()
            STATS['profile_runs'] += 1
            return out

        runner.profile_run = profile_run          # instance attribute: only this call sees the scope
        try:
            return original(self)
        finally:
            runner.__dict__.pop('profile_run', None)
            from . import install
            install.save()

    determine_available_memory.__wrapped__ = original
    cls.determine_available_memory = determine_available_memory


PATCHERS = {'vllm.v1.worker.gpu_worker': patch_gpu_worker}
