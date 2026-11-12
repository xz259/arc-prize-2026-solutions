"""stage2_v1 'fqfast' part (2026-10-15; user: "Let's do it now while we are at it"): the API server's QSA frequency step
(qsa_frequency.group_frequencies) without Python-object layouts.

ps1t (prestage validation, 10-15): the step grew 17.7 -> 41.3 ms over the hour, almost all in eviction_qsa.qsa_layout, with
constant inputs (~53K resident positions, ~18.5K descriptor rows, ~1,250 evicted ranges). It follows the API server process's
age, not the session's: at equal process age young (~80-90K logical) and old (>= 200K) sessions are equally slow, and young
ones take 20 ms at 10-25 min and 36 ms at 40-60 min. The function runs synchronously in the API server's event loop. Its
Python-object work is what slows (likely allocator locality or collection costs on a heap growing 2 -> 5 GB): a 53K-element
tuple validated element by element in Python, ~18.5K member tuples, group-end tuples, and a 53K-element list from the compact
snapshot's positions, while the step only needs the layout's group ids.

Here, for compact snapshots (the served form), the same values come from numpy:
- the resident positions are the complement of the evicted ranges, built with one cumulative sum instead of a per-range Python
  loop. The ranges are validated vectorised with production's rule (sorted, disjoint, touching allowed, within the logical
  length); inputs outside the exact-int fast path go to production's compact_positions;
- the stored-prefix digest, the frontier checks and the rows / row-bound checks are production's, in production's order and
  with production's exception types and messages;
- the group ids are the run starts of (positions below the frontier) // 4 below the complete-group frontier (np.unique of a
  nondecreasing array). Nothing else of the layout is built;
- hits matrix, exposure, frequency matrix, lazy tables and the returned dict are production's code on the same values.
Legacy snapshots, and any exception in the fast path, run production's function (its result or its error).
ARC3_S2_FQFAST_VERIFY=N: every N-th call also runs production's function and compares the whole result (NaN-aware). A
difference is counted, and production's result is returned. Per call, STATS time the fast path and, when verifying,
production's function in the same process (the growth comparison). ARC3_S2_FQFAST_SWITCH=<file> containing 'off' restores
production's function (checked at most once a second)."""
import os
import time

import numpy as np

PINS = {'qsa_frequency': 'c9ccb07eb8c4ef75a7f58d62f26cca6da4d7de6b3730e09f96f17766655afb5c'}
VERIFY = int(os.environ.get('ARC3_S2_FQFAST_VERIFY', '0') or 0)
STATS = dict(calls=0, fast=0, legacy=0, fallbacks=0, switch_off=0, fast_ms=0.0, fast_max_ms=0.0, verified=0, mismatches=0,
             orig_ms=0.0, orig_max_ms=0.0, last_error=None, last_mismatch=None, by_age=[])
_SWITCH = dict(path=os.environ.get('ARC3_S2_FQFAST_SWITCH'), t=0.0, off=False)
_TICK = [0]
_T0 = [None]
_AGE_BIN_S = 600.0


def _off():
    s = _SWITCH
    if s['path'] is None:
        return False
    now = time.monotonic()
    if now - s['t'] > 1.0:
        s['t'] = now
        try:
            with open(s['path']) as f:
                s['off'] = f.read().strip() == 'off'
        except OSError:
            s['off'] = False
    return s['off']


def _save(n):
    if n % 50 == 1:
        try:
            from . import install
            install.save(force=False)
        except Exception:                               # noqa: BLE001  receipts are diagnostics
            pass


def _age_bin(fast_ms, orig_ms):
    """Per 10-min bin of process age (from the first call): [calls, fast ms, verified calls, production ms]."""
    now = time.monotonic()
    if _T0[0] is None:
        _T0[0] = now
    k = int((now - _T0[0]) // _AGE_BIN_S)
    bins = STATS['by_age']
    while len(bins) <= k:
        bins.append([0, 0.0, 0, 0.0])
    b = bins[k]
    b[0] += 1
    b[1] += fast_ms
    if orig_ms is not None:
        b[2] += 1
        b[3] += orig_ms


def compact_positions_fast(mod, logical_tokens, evicted):
    """mod.compact_positions(logical_tokens, evicted) for exact-int inputs; other inputs (and every invalid one) run
    production's function, which returns its value or raises its error."""
    if type(logical_tokens) is not int or logical_tokens < 0 or not isinstance(evicted, (list, tuple)):
        return mod.compact_positions(logical_tokens, evicted)
    n = logical_tokens
    k = len(evicted)
    if k == 0:
        return np.arange(n, dtype=np.int64)
    if not all(type(r) in (list, tuple) and len(r) == 2 and type(r[0]) is int and type(r[1]) is int for r in evicted):
        return mod.compact_positions(logical_tokens, evicted)
    arr = np.array(evicted, dtype=np.int64)
    a, b = arr[:, 0], arr[:, 1]
    prev = np.empty(k, dtype=np.int64)
    prev[0] = 0
    prev[1:] = b[:-1]
    if bool((a < prev).any() or (b <= a).any() or (b > n).any()):
        return mod.compact_positions(logical_tokens, evicted)     # production's error
    d = np.zeros(n + 1, dtype=np.int32)
    d[a] += 1                                           # range starts are strictly increasing (a_i >= b_(i-1) > a_(i-1))
    d[b] -= 1                                           # range ends likewise; a touching start / end pair cancels
    return np.flatnonzero(np.cumsum(d[:n]) == 0)        # np.flatnonzero returns int64 (production's dtype)


def group_frequencies_fast(mod, snapshot, *, epoch_start_logical):
    """Production's group_frequencies for a compact snapshot, from numpy arrays."""
    from eviction_storage import EvictionStorageError
    QE, RATIO = mod.QSAFrequencyError, mod.RATIO
    # --- expand_compact_snapshot (positions, the stored prefix proof, the row bound), without the list of positions
    ctx = snapshot['context']
    positions = compact_positions_fast(mod, ctx['logical_tokens'], ctx['evicted'])
    stored = snapshot['stored']
    n = int(stored['length'])
    if n > len(positions) or mod._positions_digest(positions[:n]) != stored['sha256']:
        raise QE('Core frontier does not extend the worker admission context')
    if int(ctx['logical_computed']) < int(stored['logical_computed']):
        raise QE('Core frontier does not extend the worker admission context')
    layout = snapshot.get('hits_layout') or {}
    shape = tuple(int(x) for x in layout.get('shape', ()))
    if len(shape) != 2 or shape[1] != int(snapshot['rows']) or int(snapshot['rows']) > int(ctx['rows_bound']):
        raise QE('Compact snapshot rows disagree with the row bound')
    expanded = dict(slot=snapshot['slot'], rows=int(snapshot['rows']), hits=snapshot['hits'], hits_layout=snapshot['hits_layout'],
                    informative=snapshot['informative'])
    computed = int(ctx['logical_computed'])
    # --- qsa_layout(positions, logical_computed=computed).group_ids (its validation passes by construction: the positions
    #     are np.flatnonzero's strictly increasing nonnegative values)
    if computed < 0:
        raise EvictionStorageError("This prototype supports original four-token groups")
    if len(positions) and int(positions[-1]) > computed:
        raise EvictionStorageError("Only one accepted uncomputed token may follow the frontier")
    complete = computed // RATIO
    m_pos = int(np.searchsorted(positions, computed, side='left'))
    groups = positions[:m_pos] // RATIO
    m = int(np.searchsorted(groups, complete, side='left'))
    g = groups[:m]
    if m:
        first = np.empty(m, dtype=bool)
        first[0] = True
        np.not_equal(g[1:], g[:-1], out=first[1:])
        gids = g[first]                                  # == np.unique(g) for a nondecreasing g
    else:
        gids = np.empty(0, dtype=np.int64)
    gids = gids.astype(np.int64, copy=False)
    ids = gids.tolist()
    # --- production's group_frequencies from here, on the same values
    rows = len(ids)
    hits = mod._hits_matrix(expanded)
    if rows > int(expanded['rows']) or hits.shape[1] < rows:
        raise QE('Snapshot rows do not cover the descriptor layout')
    start = int(epoch_start_logical)
    expo = np.maximum(0, computed - np.maximum(start, (gids + 1) * RATIO))
    exposure = dict(zip(ids, expo.tolist()))
    with np.errstate(divide='ignore', invalid='ignore'):
        matrix = np.where(expo > 0, hits[:, :rows].astype(np.float64) / np.where(expo > 0, expo, 1), np.nan)
    frequency = mod._LazyTables(matrix, list(ids))
    return dict(frequency=frequency, matrix=matrix, group_ids_array=gids, exposure=exposure, layout_rows=rows,
                informative=[int(x) for x in expanded['informative']],
                epoch_start_logical=start, logical_computed=computed,
                epoch_query_positions=max(0, computed - start))


def _same(a, b):
    """Whole-result comparison (NaN-aware matrices; the lazy tables by their ids and matrix)."""
    if set(a) != set(b):
        return 'keys'
    for k in a:
        x, y = a[k], b[k]
        if k == 'frequency':
            if list(x._ids) != list(y._ids) or not np.array_equal(x._matrix, y._matrix, equal_nan=True):
                return 'frequency'
        elif k in ('matrix', 'group_ids_array'):
            if x.dtype != y.dtype or x.shape != y.shape or not np.array_equal(x, y, equal_nan=(k == 'matrix')):
                return k
        elif x != y:
            return k
    return None


def patch_qsa_frequency(mod):
    original = mod.group_frequencies

    def group_frequencies(snapshot, *, epoch_start_logical):
        STATS['calls'] += 1
        if _off():
            STATS['switch_off'] += 1
            return original(snapshot, epoch_start_logical=epoch_start_logical)
        if not (isinstance(snapshot, dict) and snapshot.get('form') == 'compact'):
            STATS['legacy'] += 1
            return original(snapshot, epoch_start_logical=epoch_start_logical)
        t = time.perf_counter()
        try:
            result = group_frequencies_fast(mod, snapshot, epoch_start_logical=epoch_start_logical)
        except Exception as exc:                        # noqa: BLE001  production's result or production's error
            STATS['fallbacks'] += 1
            STATS['last_error'] = repr(exc)[:300]
            return original(snapshot, epoch_start_logical=epoch_start_logical)
        fast_ms = (time.perf_counter() - t) * 1e3
        STATS['fast'] += 1
        STATS['fast_ms'] += fast_ms
        STATS['fast_max_ms'] = max(STATS['fast_max_ms'], fast_ms)
        orig_ms = None
        _TICK[0] += 1
        if VERIFY and _TICK[0] % VERIFY == 0:
            t = time.perf_counter()
            ref = original(snapshot, epoch_start_logical=epoch_start_logical)
            orig_ms = (time.perf_counter() - t) * 1e3
            STATS['verified'] += 1
            STATS['orig_ms'] += orig_ms
            STATS['orig_max_ms'] = max(STATS['orig_max_ms'], orig_ms)
            diff = _same(result, ref)
            if diff is not None:
                STATS['mismatches'] += 1
                STATS['last_mismatch'] = diff
                result = ref
        _age_bin(fast_ms, orig_ms)
        _save(STATS['calls'])
        return result

    group_frequencies.__wrapped__ = original
    mod.group_frequencies = group_frequencies


PATCHERS = {'qsa_frequency': patch_qsa_frequency}
