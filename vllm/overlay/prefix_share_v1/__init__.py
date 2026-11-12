"""System-prompt KV sharing (stage 15, 2026-09-27).

ARC3_PREFIX_SHARE (default 0). Every persistent session starts with the same rendered system turn (v5.30: 4,659
tokens), but each held its own copy of the attention KV for it. With the switch on, at every turn boundary
(engine_patch.append_session: the session is paused and settled, nothing of it is in flight) a session whose first
N = ARC3_PREFIX_SHARE_BLOCKS x block_size resident tokens (text only) equal the registered prefix gives up its private
leading attention-group block(s) for the registered shared block(s): ref_cnt + 1 on the shared blocks, the private copies
go back to the pool. The first session to reach a turn boundary registers its own blocks; the registry keeps one
reference, so the shared blocks outlive every session (a later session with the same prefix shares them too). When
only the registry still holds them, a session with another prefix replaces the registration (sessions come and go: the
Kaggle evaluation runs 110 games through fewer lanes).

Scope and invariants:
- Only the full-attention group is shared (QSA main KV and compressed indexer keys, block 2,768 tokens). GDN, ring and
  PLE state stay per session.
- Nobody writes a shared row after the swap: a session's computed frontier is far beyond N, and eviction never moves
  the protected system-prompt rows; engine_eviction refuses any plan whose first moved row of any owner lies inside a
  shared block (shared_blocks / first_moved_row below).
- The worker receives the new block table when the session is re-added at resume (the same path eviction uses).
- Every session computes and holds its own copy during its first turn, so no KV write has to be suppressed; the swap
  replaces that copy with the owner's, which differs at most by batch-composition rounding.
- Sessions whose leading tokens contain a multimodal placeholder, or differ from the registered prefix, are not shared.
"""
import hashlib
import os
import threading
import time

VERSION = 'arc3-prefix-share-v1'
FLAG = 'ARC3_PREFIX_SHARE'
BLOCKS_FLAG = 'ARC3_PREFIX_SHARE_BLOCKS'
_LOCK = threading.Lock()
_REGISTRY = None
_DISABLED = None      # reason string once a classification problem switched sharing off for this engine
_GROUP = {}           # id(kv_cache_config) -> (group id, block size)
STATS = dict(registered=0, replaced=0, swaps=0, freed_blocks=0, already_shared=0, skipped_short=0, skipped_mm=0,
             skipped_mismatch=0, skipped_not_exclusive=0)


class PrefixShareError(RuntimeError):
    pass


def enabled():
    value = os.environ.get(FLAG, '0').strip()
    if value not in ('0', '1'):
        raise PrefixShareError(f'{FLAG} must be 0 or 1')
    return value == '1'


def share_blocks():
    value = os.environ.get(BLOCKS_FLAG, '1').strip()
    if value not in ('1', '2', '3', '4'):
        raise PrefixShareError(f'{BLOCKS_FLAG} must be 1-4')
    return int(value)


def attention_group(kv_cache_config):
    """(group id, block size) of the single group holding a FullAttentionSpec (main KV) owner.

    The stock scheduler normalizes each group to one representative spec; the overlay keeps the physical per-owner
    specs in kv_cache_config._arc3_worker_kv_cache_groups (same order and block sizes, as engine_eviction reads them),
    so those are searched first."""
    key = id(kv_cache_config)
    if key in _GROUP:
        return _GROUP[key]
    from vllm.v1.kv_cache_interface import FullAttentionSpec
    groups = getattr(kv_cache_config, '_arc3_worker_kv_cache_groups', None) or kv_cache_config.kv_cache_groups
    if len(groups) != len(kv_cache_config.kv_cache_groups):
        raise PrefixShareError('worker and scheduler KV cache group counts differ')
    found = []
    for i, g in enumerate(groups):
        spec = g.kv_cache_spec; owners = getattr(spec, 'kv_cache_specs', None)
        owners = list(owners.values()) if isinstance(owners, dict) else [spec]
        if any(type(o) is FullAttentionSpec for o in owners):
            found.append((i, kv_cache_config.kv_cache_groups[i].kv_cache_spec.block_size))
    if len(found) != 1:
        raise PrefixShareError(f'expected one full-attention KV cache group, found {len(found)}')
    _GROUP[key] = found[0]
    return found[0]


def _digest(tokens):
    return hashlib.sha256(','.join(map(str, tokens)).encode()).hexdigest()


def registry():
    """JSON-safe view of the registered shared prefix (None before the first registration)."""
    reg = _REGISTRY
    if reg is None:
        return None if _DISABLED is None else dict(disabled=_DISABLED)
    return dict(disabled=_DISABLED, group_id=reg['group_id'], block_size=reg['block_size'], block_ids=[b.block_id for b in reg['blocks']],
                ref_cnts=[b.ref_cnt for b in reg['blocks']], tokens=len(reg['tokens']), digest=reg['digest'],
                owner=reg['owner'], registered_at=reg['when'])


def on_append(scheduler, session):
    """At a turn boundary, before the session is mutated: register or share its leading attention blocks.

    Returns a receipt dict when something changed, otherwise None. Every check runs before any mutation; the mutation is
    the block-list assignment plus the pool's own ref counting, so a refusal leaves the session untouched."""
    global _REGISTRY, _DISABLED
    if _DISABLED is not None:
        return None
    try:   # classification only: a problem here switches sharing off for this engine instead of failing the append
        if not enabled():
            return None
        manager = scheduler.kv_cache_manager
        coordinator = manager.coordinator
        group_id, block_size = attention_group(coordinator.kv_cache_config)
        k = share_blocks(); n = k * block_size
    except Exception as exc:  # noqa: BLE001 - never take the engine down for an optional memory saving
        _DISABLED = f'{type(exc).__name__}: {exc}'
        print(f'[arc3 prefix share] disabled: {_DISABLED}', flush=True)
        return dict(action='disabled', version=VERSION, reason=_DISABLED)
    ids = session._all_token_ids
    if session.num_computed_tokens < n or len(ids) < n:
        STATS['skipped_short'] += 1
        return None
    if any(feature.mm_position.offset < n for feature in (session.mm_features or ())):
        STATS['skipped_mm'] += 1
        return None
    blocks = coordinator.single_type_managers[group_id].req_to_blocks.get(session.request_id)
    if not blocks or len(blocks) < k:
        return None
    pool = manager.block_pool
    with _LOCK:
        reg = _REGISTRY
        if reg is None:
            if any(b.ref_cnt != 1 or b.block_id == 0 or b.is_null for b in blocks[:k]):
                STATS['skipped_not_exclusive'] += 1
                return None
            prefix = tuple(ids[:n])
            pool.touch(list(blocks[:k]))                       # the registry's own reference
            _REGISTRY = dict(group_id=group_id, block_size=block_size, blocks=list(blocks[:k]), tokens=prefix,
                             digest=_digest(prefix), owner=session.request_id, when=time.time())
            STATS['registered'] += 1
            return dict(action='registered', version=VERSION, group_id=group_id, tokens=n,
                        block_ids=[b.block_id for b in blocks[:k]], digest=_REGISTRY['digest'])
        if reg['group_id'] != group_id or reg['block_size'] != block_size:
            _DISABLED = 'KV cache geometry changed under a registered shared prefix'
            return dict(action='disabled', version=VERSION, reason=_DISABLED)
        shared = reg['blocks']
        if all(a is b for a, b in zip(blocks[:k], shared)):
            STATS['already_shared'] += 1
            return None
        shared_ids = {id(b) for b in shared}               # identity: KVCacheBlock compares field-wise
        if any(id(b) in shared_ids for b in blocks):   # never expected; leave the session untouched
            _DISABLED = 'a session holds a shared block outside the shared prefix slots'
            return dict(action='disabled', version=VERSION, reason=_DISABLED)
        if tuple(ids[:n]) != reg['tokens']:
            if (all(b.ref_cnt == 1 for b in shared)
                    and not any(b.ref_cnt != 1 or b.block_id == 0 or b.is_null for b in blocks[:k])):
                # Churn (2026-09-27, Kaggle: 110 games through ~50 lanes): the registered prefix is held by the registry
                # alone, so no live session uses it; a session with another prefix takes the registry over instead of
                # every later session being refused for good.
                prefix = tuple(ids[:n])
                pool.touch(list(blocks[:k]))
                _REGISTRY = dict(group_id=group_id, block_size=block_size, blocks=list(blocks[:k]), tokens=prefix,
                                 digest=_digest(prefix), owner=session.request_id, when=time.time(), replaced=reg['digest'])
                pool.free_blocks(list(shared))                 # the registry's reference: 1 -> 0, back to the pool
                STATS['replaced'] += 1
                return dict(action='replaced', version=VERSION, group_id=group_id, tokens=n,
                            block_ids=[b.block_id for b in blocks[:k]], released_block_ids=[b.block_id for b in shared],
                            digest=_REGISTRY['digest'])
            STATS['skipped_mismatch'] += 1
            return None
        if any(b.ref_cnt != 1 or b.block_id == 0 or b.is_null for b in blocks[:k]):
            STATS['skipped_not_exclusive'] += 1
            return None
        old = list(blocks[:k])
        pool.touch(list(shared))
        blocks[:k] = shared                                    # in place: the manager's req_to_blocks list
        pool.free_blocks(old)                                  # private copies return to the pool (ref_cnt 1 -> 0)
        STATS['swaps'] += 1; STATS['freed_blocks'] += k
        return dict(action='shared', version=VERSION, group_id=group_id, tokens=n,
                    old_block_ids=[b.block_id for b in old], shared_block_ids=[b.block_id for b in shared],
                    shared_ref_cnts=[b.ref_cnt for b in shared])


def shared_blocks(group_id, blocks):
    """How many leading entries of a request's block list (one KV cache group) are the registered shared blocks."""
    reg = _REGISTRY
    if reg is None or reg['group_id'] != group_id:
        return 0
    shared = reg['blocks']
    if len(blocks) >= len(shared) and all(a is b for a, b in zip(blocks[:len(shared)], shared)):
        return len(shared)
    return 0


def rows_intact(rows, count):
    """True when the kept-row plan keeps rows 0 .. count-1 in place (so no copy touches them)."""
    import numpy as np
    rows = np.asarray(rows, dtype=np.int64)
    return rows.shape[0] >= count and bool(np.array_equal(rows[:count], np.arange(count, dtype=np.int64)))


def reset_for_tests():
    global _REGISTRY, _DISABLED
    _REGISTRY = None; _DISABLED = None; _GROUP.clear()
    for key in STATS:
        STATS[key] = 0
