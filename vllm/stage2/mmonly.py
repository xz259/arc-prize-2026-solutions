"""stage2_v1 'mmonly' part (2026-10-23, Claude main session 099768dc; V67_DESIGN 282 / 285-287; IE's preferred fix, the user's decision):
vLLM's multimodal processor cache in its 'processor_only' mode in every server process.

Why. With the stock 'lru' mode the API process keeps a metadata-only shadow (P0) of the engine's receiver cache (P1, mm_processor_cache_gb,
default 4 GiB) and sends an image whose hash the shadow holds as the hash alone (input_processor.inject_into_mm_cache / the processor's
get_and_update_item). The two LRUs are updated in different orders in different processes, so they drift: P1 then raises
MultiModalCacheMissError and the engine answers with a retryable empty ERROR output. The rebuilt frames of the rebase arm re-send old,
already-seen images, so they are the exposed requests (10 of the 11 failed sessions of the 10-23 A/B). --mm-processor-cache-gb 0 removes
the drift too, but it also replaces content hashes with per-call uuids (no dedup of identical images; the NLL fingerprint moved +0.00025).

What. 'processor_only' is vLLM's own mode for multi-API-process deployments: the API process keeps the processed items WITH their data (an
LRU of mm_processor_cache_gb) and always sends the data; the engine and the workers keep no receiver cache. Content hashing and the
encoder's dedup are unchanged, so the numerics are unchanged. MultiModalRegistry._get_cache_type is the single decision every cache
constructor reads (processor_cache_from_config in the API process, engine_receiver_cache_from_config in the engine core,
worker_receiver_cache_from_config in the workers): it returns 'processor_only' where vLLM would return 'lru' or 'shm'; None (no
multimodal input, or the cache size 0) stays None.

Receipts: STATS per process (calls, results as 'original->returned' counts, the pid); a process that built a cache shows 'lru->processor_only'.
"""
import os

PINS = {'vllm.multimodal.registry': 'e3eea9b52dc6868791627bd90639c74e0381e1b26f7b6138acde4a8f0bf02330'}
STATS = dict(calls=0, results={}, pid=os.getpid(), errors=0, last_error=None)


def patch_registry(mod):
    cls = mod.MultiModalRegistry
    orig = cls._get_cache_type

    def _get_cache_type(self, vllm_config):
        got = orig(self, vllm_config)
        out = 'processor_only' if got in ('lru', 'shm') else got
        STATS['calls'] += 1
        key = '%s->%s' % (got, out)
        STATS['results'][key] = STATS['results'].get(key, 0) + 1
        try:                                                # the receipt proves the mode per process (API, engine, workers)
            from . import install
            install.save()
        except Exception as exc:                            # noqa: BLE001  receipts are diagnostics; never fail a launch for them
            STATS['errors'] += 1
            STATS['last_error'] = repr(exc)[:200]
        return out

    _get_cache_type.__wrapped__ = orig
    cls._get_cache_type = _get_cache_type


PATCHERS = {'vllm.multimodal.registry': patch_registry}
