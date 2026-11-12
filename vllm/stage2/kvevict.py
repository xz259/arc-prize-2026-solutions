"""stage2_v1 'kvevict' part (milestone 2, 2026-10-01): QSA eviction for FP8 main KV (the winners'-format arm g7w).

The first FP8-KV hour run (g7w1, 14:05) failed at every game's first eviction: production's async eviction
(runs/dense_fp8_v1/engine_eviction_async.py:76, worker_compact_async) admits only NVFP4 main caches, uint8 rows of
296 bytes. The copy it then calls, eviction_storage.compact_cache_rows_, is layout-generic. It reinterprets the cache
as bytes and moves whole token rows ([pages, heads, tokens, row bytes], token axis 2), scale bytes included, with no
decode or requantization.

The FP8 formats (arc3_qsa_quant: per side 256 FP8 bytes and a 4-byte FP32 scale, so 520-byte rows) have the same
layout. This part admits 520-byte uint8 main rows as well, by a source transform of that one condition at import (in
memory; the production file is unchanged). test_kvevict.py runs production's compaction and byte digest on synthetic
520-byte caches.

Scope: only the async path (ARC3_EVICTION_MODE=async, production's default). The fused path keeps its gate
(engine_eviction.py:458); that module is already patched by schedfast, and a source transform cannot chain."""

PINS = {'engine_eviction_async': 'a9bd65dc98300a435be76cb0b40b49d15d021ca3e5550db56c2b8f24b78ef278'}
OLD = 'if item["kind"] == "main" and (cache.dtype != torch.uint8 or cache.shape[-1] != 296):'
NEW = 'if item["kind"] == "main" and (cache.dtype != torch.uint8 or cache.shape[-1] not in (296, 520)):   # kvevict: FP8 rows'
STATS = dict(transformed=0)


def transform(src):
    assert src.count(OLD) == 1, 'kvevict: gate anchor'
    STATS['transformed'] += 1
    return src.replace(OLD, NEW)


PATCHERS = {'engine_eviction_async': ('source', transform)}
