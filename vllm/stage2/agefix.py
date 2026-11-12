"""stage2_v1 'agefix' part (2026-10-16; the user: "Since the run is 9hrs, let's make sure they are addressed"; IE's review): the
engine core's audit-path work that grows with a session's logical history.

age1 (4 h, K3 32 + 2) per call, first 80 min -> hours 3-4: update_from_output at a turn end audit 1.6 -> 8.5 ms and logical_digest
1.0 -> 7.2 ms; core_add 17 -> 32 ms (audit 3.0 -> 8.0, logical_digest 1.7 -> 5.5); evict_commit 14 -> 26 ms (audit 2.9 -> 7.0,
logical_digest 1.8 -> 5.3). Per-lane throughput fell 52 -> ~49 tok/s from hour 1 to hours 3-4 (tau -1.7%).

Two causes, both in production's evidence path:
1. engine_eviction.install() wraps engine_patch._audit so that every audit event (start, resume, pause, close, preempt,
   evict_prepare, evict) first computes logical_eviction_state, which includes logical_digest(state), and only then calls
   production's _audit, which returns at once when ARC3_PERSISTENT_ENGINE_AUDIT is unset. With the audit off, the digest work is
   wasted; with it on, it is evidence only.
2. engine_eviction._PrefixDigest.hexdigest (the incremental logical digest) checks the hashed prefix on every call with
   current[:n] != self.ids: an O(L) slice copy and compare, L = the session's whole logical history (up to the 200K restart).

The part, exact by construction:
- an outermost wrapper on engine_patch._audit (installed right after engine_eviction.install()): when ARC3_PERSISTENT_ENGINE_AUDIT is
  unset it returns at once, as production's _audit does, without the wrapper's digest and copies (nothing else in the chain has a
  side effect: the eviction wrapper's only other branch is a deferred-resume early return);
- with the audit on, audit-path digests check only the hashed tail [n - ARC3_S2_AGEFIX_TAIL, n) (default 32,768 rows: LogicalState
  rewrites only the current turn's output region above prompt_logical_length, at most a turn's 16K tokens), and the full prefix on
  every ARC3_S2_AGEFIX_FULL_EVERY-th (32) audit digest; a mismatch rehashes from scratch exactly as production does. Every digest
  outside the audit path (evict_begin's comparison with the session service's logical_prefix_sha256) keeps production's full check.
v2 (2026-10-16, IE's review):
- the audit's token digests: production's _audit snapshots list(request._all_token_ids) and its single pool thread json.dumps +
  sha256s it twice (all ids, the computed prefix) holding the GIL ~10-20 ms per event (core_add's cpu/wall 0.65-0.73 in age1).
  engine_patch._audit_pool() now returns a proxy whose submit(_audit_write, destination, item, all_ids, computed) computes both
  digests on the core thread from a per-request incremental chain (sha256 of the same JSON bytes, extended with hasher.copy():
  byte-identical to production's), reset when the request's eviction revision changes (the commit rewrites _all_token_ids) or the
  hashed prefix's tail [n - TAIL, n) differs, with a full compare every FULL_EVERY-th event per request; the pool thread only
  writes the line (production's write code, same key order). Any other submit passes through. Production's _audit (and the
  wrappers around it) is unchanged; the core thread keeps production's list copy (a pointer copy, no int churn).
- ResidentLedger.resident_positions(): production rebuilds np.concatenate(...).tolist() (~50K fresh heap ints) on every call. A
  WeakKeyDictionary keyed by the ledger holds (the evicted list object, its tuple, logical_tokens, positions); a call with the same
  evicted ranges (identity and value) and the same length returns list(cached) (same int objects, a new list callers may mutate);
  a longer length with the same ranges extends the cache by range(old, new) (appended rows are resident); anything else rebuilds.
  The value depends on nothing else (survivor_intervals() reads only logical_tokens and evicted).
- Pickle: production's _PrefixDigest holds a hashlib object, so no path can pickle a LogicalState carrying a digest cache; deepcopy
  (core_snapshot) reconstructs through the class object and works with the subclass below.
v3 (2026-10-17, the 12-h twin's steptimer: evict_begin 100 -> 139 ms, ledger_plan 9.4 -> 12.2 ms per eviction, late sessions with
~1,000 prior evicted ranges and 1,100-1,400 new ones per plan):
- ResidentLedger.commit verifies its plan by re-planning it (fresh = self.plan(plan.ranges); fresh != plan raises): normalize_ranges,
  survivor_intervals and a json + sha256 over every range a second time per eviction. plan() is a pure function of (logical_tokens,
  evicted, ranges) apart from the protected-overlap validation (commit's re-plan passes none). The last plans built are kept with
  their inputs (the ledger's evicted tuple object, logical_tokens, the returned ranges tuple object); commit's re-plan with the same
  objects returns a new EvictionPlan rebuilt from the saved immutable field values, so production's fresh != plan still compares
  every field (a plan altered after it was built still raises). Anything else re-plans as production does.
- gc.freeze() once, ARC3_S2_AGEFIX_GCFREEZE_S seconds after the part installs in a process (0 = off, the default): the objects that
  exist then (model, graphs, modules, the warmup's) leave the collector's generations. Semantics unchanged; garbage cycles alive
  at that moment are never collected (a bounded leak).
- The logical digest's prefix check made exact without the O(L) compare (IE: make it exact rather than relaxed). Every in-place
  write to a LogicalState's logical_ids is LogicalState.paused()'s fast path (del ids[prompt_logical_length:], then extend by the
  output); the resume append and paused() otherwise only extend at the end; the slow path rebinds the list. The cache records the
  list object it hashed and the lowest index paused() may have rewritten since (prompt_logical_length at the call, recorded by a
  wrapper on paused()); the next digest compares only [that index, n) of the hashed prefix, on both the audit and the validation
  path, and the whole prefix when the list object changed or on every FULL_EVERY-th digest per session (the canary). A mismatch
  rehashes from scratch as production does.
- prepare_transaction's O(resident) validations and lookups, vectorized with identical outputs (types included) and production's
  code as the fallback (ARC3_S2_AGEFIX_VEC=0 disables): eviction_storage._positions (the type / sign / strictly-increasing check
  of position tuples: set(map(type, ...)) == {int} and numpy comparisons; ints outside int64 fall back), eviction_qsa's
  descriptor_rows_fast (np.unique of an already sorted array replaced by its linear form), engine_eviction's
  _native_compacted_features (bisect on the strictly increasing keep_indices instead of a dict over every kept row).
Receipt: STATS (audit calls skipped / passed, cheap and full checks, tracked checks / rows / canary full checks, rehashes, chain
digests / resets, resident_positions hits / extensions / rebuilds, plan memo hits / misses / canary runs / mismatches, gc time per
generation and the freeze, foreign-thread calls: nonzero means investigate)."""
import hashlib
import json
import os
import threading
import weakref

EE = 'engine_eviction'
EE_PIN = '1c49513acc9543b6f2392cdcf1f8c6eb172f84f343d853ff2464314f558ae2b5'      # as ctxtail's (chained hooks)
RES = 'residency'
RES_PIN = '307187f8247069d1f7c7a89b9b276c6e0aa14b211b1f47471bb3a4fa8c6adbaf'
ES = 'eviction_storage'
ES_PIN = '2642d861ebe25cffd08786e4712877e0219d2d5820ff2a844562e5208a467824'
EQ = 'eviction_qsa'
EQ_PIN = '568bf16b97130ec9f0ee2d033949b9aeefce4e0d7bcfe509aac6d43dcfedfbbc'      # as install.GUARD_PINS (chained hooks)
PINS = {EE: EE_PIN, RES: RES_PIN, ES: ES_PIN, EQ: EQ_PIN}
CHEAP_AUDIT = os.environ.get('ARC3_S2_AGEFIX_AUDIT_CHAIN', '1') != '0'
RP_CACHE = os.environ.get('ARC3_S2_AGEFIX_RP_CACHE', '1') != '0'
PLAN_MEMO = os.environ.get('ARC3_S2_AGEFIX_PLAN_MEMO', '1') != '0'                    # v3
GCFREEZE_S = float(os.environ.get('ARC3_S2_AGEFIX_GCFREEZE_S', '0') or 0)             # v3: 0 = off
GC_THRESHOLDS = tuple(int(x) for x in os.environ.get('ARC3_S2_AGEFIX_GC_THRESHOLDS', '').split(',') if x.strip()) or None  # 4a
assert GC_THRESHOLDS is None or (len(GC_THRESHOLDS) == 3 and all(x > 0 for x in GC_THRESHOLDS)), 'ARC3_S2_AGEFIX_GC_THRESHOLDS'
PLAN_CANARY = int(os.environ.get('ARC3_S2_AGEFIX_PLAN_CANARY', '8') or 0)               # v3 (IE 10-17): every 8th hit re-plans (gate: 4)
GC_TIMING = os.environ.get('ARC3_S2_AGEFIX_GC_TIMING', '1') != '0'                      # v3 (IE): gc time receipt
TRACK = os.environ.get('ARC3_S2_AGEFIX_TRACK', '1') != '0'                               # v3: the tracked exact prefix check
VEC = os.environ.get('ARC3_S2_AGEFIX_VEC', '1') != '0'                                   # v3: vectorized validations
VEC_VERIFY = os.environ.get('ARC3_S2_AGEFIX_VEC_VERIFY', '0') == '1'                    # v3 gate only: compare every call
_CLEAN = 1 << 62
TAIL = int(os.environ.get('ARC3_S2_AGEFIX_TAIL', '32768') or 32768)
# v3 (IE 10-17): 8 for the first 18-h run (the tracked check's canary, ~1.1 ms per eviction on average); 32 after a clean run
FULL_EVERY = int(os.environ.get('ARC3_S2_AGEFIX_FULL_EVERY', '8') or 8)
STATS = dict(installed=False, audit_skipped=0, audit_passed=0, cheap_checks=0, full_checks_audit=0, full_checks_other=0,
             rehashes=0, foreign_thread_calls=0, chain_digests=0, chain_resets=0, chain_full_checks=0, chain_fallbacks=0,
             track_checks=0, track_rows=0, track_full_checks=0, track_canary_mismatches=0,
             vec_positions_fast=0, vec_positions_fallback=0, vec_descriptor_rows=0, vec_features_fast=0, vec_features_fallback=0, vec_verify_calls=0, vec_verify_mismatches=0, vec_verify_errors=0,
             rp_hits=0, rp_extensions=0, rp_rebuilds=0, plan_memo_hits=0, plan_memo_misses=0, plan_canary_runs=0,
             plan_canary_mismatches=0, gc_frozen=None, gc_freeze_count=None, gc_thresholds=None, gc_collections=[0, 0, 0], gc_seconds=[0.0, 0.0, 0.0],
             errors=0, last_error=None,
             config=dict(full_every=FULL_EVERY, track=TRACK, vec=VEC, plan_memo=PLAN_MEMO, plan_canary=PLAN_CANARY,
                         gc_timing=GC_TIMING, gcfreeze_s=GCFREEZE_S, tail=TAIL, rp_cache=RP_CACHE, audit_chain=CHEAP_AUDIT,
                         vec_verify=VEC_VERIFY, gc_thresholds=GC_THRESHOLDS))
_TLS = threading.local()
_OWNER = []                    # the first thread that digests (the engine core's): the digest objects are not locked


def _wrap_audit(engine_patch):
    inner = engine_patch._audit
    if getattr(inner, '_s2_agefix', False):
        return

    def _audit(scheduler, event, request, **extra):
        if not os.environ.get('ARC3_PERSISTENT_ENGINE_AUDIT'):
            STATS['audit_skipped'] += 1
            return None
        STATS['audit_passed'] += 1
        _TLS.audit = True
        try:
            return inner(scheduler, event, request, **extra)
        finally:
            _TLS.audit = False

    _audit._s2_agefix = True
    _audit.__wrapped__ = inner
    _audit.__name__ = _audit.__qualname__ = '_audit'
    engine_patch._audit = _audit


def _patch_prefix_digest(mod):
    """IE (10-16): the every-FULL_EVERY-th full compare counts per digest object (one per session), so the staleness bound is per
    session and independent of how sessions interleave: production's _PrefixDigest has __slots__, so a subclass with one more
    slot replaces the module's class (logical_digest looks the name up at call time; no instance exists before the patch)."""
    base = mod._PrefixDigest
    import hashlib

    class _PrefixDigest(base):
        __slots__ = ('_s2_n', '_s2_list', '_s2_dirty')

        def __init__(self):
            base.__init__(self)
            self._s2_n = 0
            self._s2_list = None          # v3: the list object last hashed
            self._s2_dirty = 0            # v3: lowest index paused() may have rewritten since (0 = unknown: compare all)

        def hexdigest(self, current, dumps):
            return _hexdigest(self, current, dumps)

    _PrefixDigest.__doc__ = base.__doc__
    _PrefixDigest._s2_base = base

    def _hexdigest(self, current, dumps):
        me = threading.get_ident()
        if not _OWNER:
            _OWNER.append(me)
        elif _OWNER[0] != me:
            STATS['foreign_thread_calls'] += 1
        n = len(self.ids)
        if n:
            if len(current) < n:
                ok = False
            elif TRACK and self._s2_list is current:              # v3: exact, only the region paused() may have rewritten
                self._s2_n += 1
                if self._s2_n % FULL_EVERY == 0:                   # the canary: production's full compare
                    STATS['track_full_checks'] += 1
                    ok = current[:n] == self.ids
                    if not ok:
                        lo = min(self._s2_dirty, n)
                        if current[lo:n] == self.ids[lo:n]:       # the tracked check alone would have missed it
                            STATS['track_canary_mismatches'] += 1
                            STATS['last_error'] = 'untracked write below the dirty index'
                else:
                    lo = min(self._s2_dirty, n)
                    STATS['track_checks'] += 1
                    STATS['track_rows'] += n - lo
                    ok = current[lo:n] == self.ids[lo:n]
            elif getattr(_TLS, 'audit', False):
                self._s2_n += 1
                if self._s2_n % FULL_EVERY == 0:
                    STATS['full_checks_audit'] += 1
                    ok = current[:n] == self.ids
                else:
                    STATS['cheap_checks'] += 1
                    lo = max(0, n - TAIL)
                    ok = current[lo:n] == self.ids[lo:n]
            else:
                STATS['full_checks_other'] += 1
                ok = current[:n] == self.ids
            if not ok:
                STATS['rehashes'] += 1
                self.ids = []
                self.hasher = hashlib.sha256(b'[')
                n = 0
        if len(current) > n:                                   # production's extension, verbatim
            tail = current[n:]
            self.hasher.update((("," if n else "") + dumps(tail)[1:-1]).encode())
            self.ids.extend(tail)
        self._s2_list = current                                # v3: hashed and verified up to len(current)
        self._s2_dirty = _CLEAN
        final = self.hasher.copy()
        final.update(b"]")
        return final.hexdigest()

    mod._PrefixDigest = _PrefixDigest
    if TRACK:
        _patch_paused(mod)
    if VEC:
        _patch_features(mod)


def _patch_paused(mod):
    """v3: before production's paused() rewrites logical_ids[prompt_logical_length:], record that index in the state's digest
    cache (the lowest index any write of this call can touch; the slow path rebinds the list, which the cache detects by identity)."""
    cls = mod.LogicalState
    orig = cls.paused

    def paused(self, request):
        c = self.__dict__.get('_arc3_logical_digest')
        if c is not None and hasattr(c, '_s2_dirty'):
            p = self.prompt_logical_length
            if type(p) is not int or p < 0:
                c._s2_dirty = 0
            elif p < c._s2_dirty:
                c._s2_dirty = p
        return orig(self, request)

    paused.__wrapped__ = orig
    cls.paused = paused


# ------------------------------------------------------------------------------------------------ v2: the audit's token digests
_CHAINS = {}                   # request_id -> [hashed ids copy, hasher at n, n, eviction revision, events]


def _extend(hasher, n, ids):
    hasher.update((("," if n else "") + json.dumps(ids, separators=(",", ":"))[1:-1]).encode())


def _chain_digests(item, all_ids, computed):
    """(tokens_sha256, computed_prefix_sha256) of production's _audit_write, from the request's incremental chain."""
    rid = item.get('request_id')
    rev = (item.get('logical_eviction_state') or {}).get('eviction_revision')
    ch = _CHAINS.get(rid)
    if ch is None or ch[3] != rev:
        if ch is not None:
            STATS['chain_resets'] += 1
        ch = _CHAINS[rid] = [[], hashlib.sha256(b'['), 0, rev, 0]
        if len(_CHAINS) > 1024:                                 # requests that never closed: keep the newest
            for k in list(_CHAINS)[:256]:
                _CHAINS.pop(k, None)
    computed = max(0, min(int(computed), len(all_ids)))        # production slices all_ids[:computed]
    n = ch[2]
    if n:
        ch[4] += 1
        if len(all_ids) < n or computed < n:
            ok = False
        elif ch[4] % FULL_EVERY == 0:
            STATS['chain_full_checks'] += 1
            ok = all_ids[:n] == ch[0]
        else:
            lo = max(0, n - TAIL)
            ok = all_ids[lo:n] == ch[0][lo:n]
        if not ok:
            STATS['chain_resets'] += 1
            ch[0], ch[1], n = [], hashlib.sha256(b'['), 0
    if computed > n:
        part = all_ids[n:computed]
        _extend(ch[1], n, part)
        ch[0].extend(part)
        n = computed
    ch[2] = n
    h = ch[1].copy()
    h.update(b"]")
    comp = h.hexdigest()
    if len(all_ids) > n:
        h = ch[1].copy()
        _extend(h, n, all_ids[n:])
        h.update(b"]")
        tok = h.hexdigest()
    else:
        tok = comp
    if item.get('event') in ('close_released',):
        _CHAINS.pop(rid, None)
    STATS['chain_digests'] += 1
    return tok, comp


def _write_line(destination, item, tok, comp):
    """production's _audit_write after its two digests, verbatim (same key order, same error print)."""
    try:
        item["tokens_sha256"] = tok
        item["computed_prefix_sha256"] = comp
        encoded = (json.dumps(item, separators=(",", ":")) + "\n").encode()
        fd = os.open(destination, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        with os.fdopen(fd, "ab", closefd=True) as stream:
            stream.write(encoded)
    except Exception as exc:  # noqa: BLE001 - evidence sink failures must not take the engine down asynchronously
        print(f"[arc3 engine audit] write failed: {type(exc).__name__}: {exc}", flush=True)


def _patch_audit_pool(engine_patch):
    if getattr(engine_patch._audit_pool, '_s2_agefix', False):
        return
    orig_pool = engine_patch._audit_pool
    orig_write = engine_patch._audit_write

    class _Proxy:
        def submit(self, fn, *args, **kwargs):
            pool = orig_pool()
            if fn is orig_write and len(args) == 4 and not kwargs:
                destination, item, all_ids, computed = args
                try:
                    tok, comp = _chain_digests(item, all_ids, computed)
                except Exception as exc:                        # noqa: BLE001  fall back to production's write
                    STATS['chain_fallbacks'] += 1
                    STATS['last_error'] = repr(exc)[:300]
                    return pool.submit(fn, *args)
                return pool.submit(_write_line, destination, item, tok, comp)
            return pool.submit(fn, *args, **kwargs)

    proxy = _Proxy()

    def _audit_pool():
        return proxy

    _audit_pool._s2_agefix = True
    _audit_pool.__wrapped__ = orig_pool
    engine_patch._audit_pool = _audit_pool


# ------------------------------------------------------------------------------------------------ v2: resident positions
_RP = weakref.WeakKeyDictionary()      # ledger -> [evicted list object, tuple(evicted), logical_tokens, positions]


def patch_residency(mod):
    cls = mod.ResidentLedger
    orig = cls.resident_positions

    def resident_positions(self):
        ev = self.evicted
        n = self.logical_tokens
        e = _RP.get(self)
        if e is not None and e[0] is ev and e[1] == tuple(ev):
            if e[2] == n:
                STATS['rp_hits'] += 1
                return list(e[3])
            if n > e[2] and type(n) is int:
                e[3].extend(range(e[2], n))                    # rows appended above every evicted range are resident
                e[2] = n
                STATS['rp_extensions'] += 1
                return list(e[3])
        positions = orig(self)
        STATS['rp_rebuilds'] += 1
        try:
            _RP[self] = [ev, tuple(ev), n, list(positions)]
        except TypeError:                                       # not weak-referenceable: no cache
            pass
        return positions

    resident_positions.__wrapped__ = orig
    cls.resident_positions = resident_positions


# ------------------------------------------------------------------------------------------------ v3: commit's re-plan
_PLANS = []                    # newest last: (evicted tuple object, logical_tokens, ranges tuple object, saved fields)
_PLAN_KEEP = 16


def patch_plan_memo(mod):
    cls = mod.ResidentLedger
    orig = cls.plan
    plan_cls = mod.EvictionPlan
    names = [f for f in plan_cls.__dataclass_fields__]

    def plan(self, ranges, *, protected=()):
        if not protected:
            # key: the ledger's evicted tuple object (immutable; every change rebinds it), its logical_tokens int (every append
            # changes it) and the ranges tuple object of the built plan: O(1), equal key = identical inputs (IE's version key)
            for ev, n, rg, saved in reversed(_PLANS):
                if ranges is rg and self.evicted is ev and self.logical_tokens == n and type(n) is int:
                    STATS['plan_memo_hits'] += 1
                    memo = plan_cls(**saved)
                    if PLAN_CANARY and STATS['plan_memo_hits'] % PLAN_CANARY == 0:   # IE: a full re-plan as a canary
                        STATS['plan_canary_runs'] += 1
                        fresh = orig(self, ranges, protected=protected)
                        if fresh != memo:
                            STATS['plan_canary_mismatches'] += 1
                            STATS['last_error'] = 'plan memo canary mismatch'
                            return fresh                      # production's value
                    return memo
        STATS['plan_memo_misses'] += 1
        result = orig(self, ranges, protected=protected)
        _PLANS.append((self.evicted, self.logical_tokens, result.ranges, {f: getattr(result, f) for f in names}))
        del _PLANS[:-_PLAN_KEEP]
        return result

    plan.__wrapped__ = orig
    cls.plan = plan


def patch_residency_all(mod):
    if RP_CACHE:
        patch_residency(mod)
    if PLAN_MEMO:
        patch_plan_memo(mod)


# ------------------------------------------------------------------------------------------------ v3: vectorized validations
def _guarded(fn, *a):
    """Gate mode: a comparison that raises is counted (vec_verify_errors), never raised into the engine."""
    try:
        fn(*a)
    except Exception as exc:                                    # noqa: BLE001
        STATS['vec_verify_errors'] += 1
        STATS['last_error'] = ('vec_verify error: %r' % exc)[:300]


def _verify(name, fast, ref):
    """Gate mode (ARC3_S2_AGEFIX_VEC_VERIFY=1): the fast outcome against production's, as ('ok', value, types) or ('err', type,
    message); a mismatch is counted and named in last_error (the fast outcome is still what the caller gets)."""
    STATS['vec_verify_calls'] += 1
    if fast != ref:
        STATS['vec_verify_mismatches'] += 1
        STATS['last_error'] = ('vec_verify %s: %r / %r' % (name, fast, ref))[:300]


def _outcome(fn, *a, **k):
    try:
        r = fn(*a, **k)
    except Exception as exc:                                    # noqa: BLE001
        return ('err', type(exc).__name__, str(exc)), None
    return ('ok', r, _types(r)), r


def _types(x):
    if isinstance(x, (tuple, list)):
        return (type(x).__name__, tuple(sorted({_types(v) for v in x}, key=str)))
    return type(x).__name__


def patch_eviction_storage(mod):
    """eviction_storage._positions with production's memo, error and acceptance set: every element exactly int (bools and
    numpy scalars rejected as production does), the first >= 0 and each > its predecessor; values outside int64 take
    production's path. token_compaction / page_compaction look the name up at call time; eviction_qsa imports it after this
    hook runs (engine_eviction imports eviction_storage first)."""
    orig = mod._positions
    import numpy as np

    def _positions(values, name):
        if type(values) is tuple and mod._VALIDATED.get(id(values)) is values:
            return values
        result = tuple(values)
        try:
            if not result:
                ok = True
            elif set(map(type, result)) == {int}:
                a = np.fromiter(result, dtype=np.int64, count=len(result))
                ok = bool(a[0] >= 0) and bool((a[1:] > a[:-1]).all())
            else:
                ok = False
        except (OverflowError, ValueError):
            STATS['vec_positions_fallback'] += 1
            return orig(result, name)
        if not ok:
            STATS['vec_positions_fallback'] += 1
            return orig(result, name)                          # production raises its own error
        STATS['vec_positions_fast'] += 1
        if VEC_VERIFY:                                         # a list copy: production validates it in full
            _guarded(lambda: _verify('_positions', ('ok', result, _types(result)), _outcome(orig, list(result), name)[0]))
        if len(mod._VALIDATED) >= mod._VALIDATED_LIMIT:         # production's memo, verbatim
            mod._VALIDATED.pop(next(iter(mod._VALIDATED)))
        mod._VALIDATED[id(result)] = result
        return result

    _positions.__wrapped__ = orig
    mod._positions = _positions


def patch_eviction_qsa(mod):
    """descriptor_rows_fast, production's code with np.unique on the sorted (o // 4) and (k // 4) prefixes replaced by the
    linear unique of a sorted array (o and k are strictly increasing by _positions, so the quotients are non-decreasing)."""
    orig = mod.descriptor_rows_fast
    import numpy as np
    Err = mod.EvictionStorageError

    def _uniq(x):
        if len(x) < 2:
            return x.copy()
        return x[np.concatenate((np.ones(1, dtype=bool), x[1:] != x[:-1]))]

    def descriptor_rows_fast(old_positions, kept_positions, *, logical_computed, ratio=4):
        if VEC_VERIFY:
            fast, r = _outcome(_descriptor_rows, old_positions, kept_positions, logical_computed=logical_computed, ratio=ratio)
            _guarded(lambda: _verify('descriptor_rows_fast', fast, _outcome(
                orig, old_positions, kept_positions, logical_computed=logical_computed, ratio=ratio)[0]))
            if fast[0] == 'err':
                return _descriptor_rows(old_positions, kept_positions, logical_computed=logical_computed, ratio=ratio)
            return r
        return _descriptor_rows(old_positions, kept_positions, logical_computed=logical_computed, ratio=ratio)

    def _descriptor_rows(old_positions, kept_positions, *, logical_computed, ratio=4):
        old = mod._positions(old_positions, "Resident original token positions")
        kept = mod._positions(kept_positions, "Resident original token positions")
        if (type(ratio) is not int or ratio != 4 or type(logical_computed) is not int or logical_computed < 0):
            raise Err("This prototype supports original four-token groups")
        if (old and old[-1] > logical_computed) or (kept and kept[-1] > logical_computed):
            raise Err("Only one accepted uncomputed token may follow the frontier")
        o = np.fromiter(old, dtype=np.int64, count=len(old)); k = np.fromiter(kept, dtype=np.int64, count=len(kept))
        if len(k):
            if not len(o):
                raise Err("Compaction introduced new token identities")
            idx = np.searchsorted(o, k)
            if int(idx.max()) >= len(o) or not np.array_equal(o[idx], k):
                raise Err("Compaction introduced new token identities")
        complete = logical_computed // ratio
        go = _uniq(o[:int(np.searchsorted(o, logical_computed, side="left"))] // ratio); go = go[go < complete]
        gk = _uniq(k[:int(np.searchsorted(k, logical_computed, side="left"))] // ratio); gk = gk[gk < complete]
        if len(gk):
            if not len(go):
                raise Err("Compaction introduced a compressed group")
            rows = np.searchsorted(go, gk)
            if int(rows.max()) >= len(go) or not np.array_equal(go[rows], gk):
                raise Err("Compaction introduced a compressed group")
        else:
            rows = gk
        STATS['vec_descriptor_rows'] += 1
        return int(len(go)), tuple(rows.tolist())

    descriptor_rows_fast.__wrapped__ = orig
    mod.descriptor_rows_fast = descriptor_rows_fast


def _patch_features(mod):
    """engine_eviction._native_compacted_features with bisect on plan.keep_indices (a strictly increasing tuple of ints, from
    token_compaction): a feature's rows are all kept iff the count of kept indices inside [offset, offset + length) equals
    length, and its new offset is that count's lower bound; any other shape takes production's path."""
    orig = mod._native_compacted_features
    from bisect import bisect_left
    replace_ = mod.replace

    import dataclasses

    def _feature_key(f):
        p = f.mm_position
        return (type(p.offset).__name__, p.offset, p.length,
                tuple((fld.name, id(getattr(p, fld.name))) for fld in dataclasses.fields(p) if fld.name not in ('offset',)),
                tuple((fld.name, id(getattr(f, fld.name))) for fld in dataclasses.fields(f) if fld.name != 'mm_position'))

    def _native_compacted_features(features, plan):
        if VEC_VERIFY:
            fast, r = _outcome(_features, features, plan)
            key = lambda o: (o[0], [_feature_key(f) for f in o[1]]) if o[0] == 'ok' else o   # noqa: E731
            _guarded(lambda: _verify('_native_compacted_features', key(fast), key(_outcome(orig, features, plan)[0])))
            if fast[0] == 'err':
                return _features(features, plan)
            return r
        return _features(features, plan)

    def _features(features, plan):
        keep = plan.keep_indices
        if type(keep) is not tuple:
            STATS['vec_features_fallback'] += 1
            return orig(features, plan)
        result = []
        for feature in features:
            position = feature.mm_position
            a, n = position.offset, position.length
            if type(a) is not int or type(n) is not int:
                STATS['vec_features_fallback'] += 1
                return orig(features, plan)
            if n <= 0:                                        # production: no rows, nothing present: dropped
                continue
            lo = bisect_left(keep, a)
            hi = bisect_left(keep, a + n, lo)
            if hi - lo == n:
                result.append(replace_(feature, mm_position=replace_(position, offset=lo)))
            elif hi > lo:
                raise mod.EngineEvictionError("Eviction cuts through a multimodal feature")
        STATS['vec_features_fast'] += 1
        return result

    _native_compacted_features.__wrapped__ = orig
    mod._native_compacted_features = _native_compacted_features


# ------------------------------------------------------------------------------------------------ v3: gc timing and gc.freeze
def _gc_timing():
    """IE: the collector's time per generation in this process (a receipt for the long run), via gc.callbacks."""
    import gc
    import time
    start = [0.0]

    def cb(phase, info):
        if phase == 'start':
            start[0] = time.perf_counter()
        else:
            g = info.get('generation', 0)
            if 0 <= g <= 2:
                STATS['gc_collections'][g] += 1
                STATS['gc_seconds'][g] += time.perf_counter() - start[0]

    gc.callbacks.append(cb)

def _gc_freeze_later():
    import gc
    import time

    def run():
        time.sleep(GCFREEZE_S)
        t0 = time.perf_counter()
        gc.freeze()
        STATS['gc_frozen'] = round(time.perf_counter() - t0, 4)
        STATS['gc_freeze_count'] = gc.get_freeze_count()
        if GC_THRESHOLDS:                                       # 4a (IE 10-17): rarer collections after the freeze
            gc.set_threshold(*GC_THRESHOLDS)
            STATS['gc_thresholds'] = list(gc.get_threshold())

    threading.Thread(target=run, name='s2-agefix-gcfreeze', daemon=True).start()


def patch_engine_eviction(mod):
    _patch_prefix_digest(mod)
    if IV:                                                     # 4b (default off)
        try:
            _install_iv(mod)
        except Exception as exc:                               # noqa: BLE001  (production stays in place)
            STATS['iv_errors'] += 1
            STATS['iv_last_error'] = repr(exc)[:300]
    orig_install = mod.install

    def install(*args, **kwargs):
        result = orig_install(*args, **kwargs)
        try:
            import engine_patch
            _wrap_audit(engine_patch)
            if CHEAP_AUDIT:
                _patch_audit_pool(engine_patch)
            STATS['installed'] = True
        except Exception as exc:                                # noqa: BLE001
            STATS['errors'] += 1
            STATS['last_error'] = repr(exc)[:300]
            raise
        return result

    install.__wrapped__ = orig_install
    mod.install = install


# ------------------------------------------------------------------------------------------- 4b: interval-based compaction
IV = os.environ.get('ARC3_S2_AGEFIX_IV', '0') == '1'
IV_CANARY = int(os.environ.get('ARC3_S2_AGEFIX_IV_CANARY', '8') or 0)
IV_VERIFY = os.environ.get('ARC3_S2_AGEFIX_IV_VERIFY', '0') == '1'
STATS.update(iv_installed=False, iv_calls=0, iv_fast=0, iv_fallbacks=0, iv_errors=0, iv_canary_runs=0, iv_canary_mismatches=0,
             iv_verify_calls=0, iv_last_error=None)
STATS['config'].update(iv=IV, iv_canary=IV_CANARY, iv_verify=IV_VERIFY)
_IV_OLD = """    kept=[i for begin,end in logical_plan.survivor_intervals for i in range(begin,end)]
    compact=token_compaction(state.ledger.resident_positions(),kept,
                             old_computed=request.num_computed_tokens,allow_fully_computed=allow_closed_frontier)
    _ki=np.fromiter(compact.keep_indices,dtype=np.int64,count=len(compact.keep_indices))
"""
_IV_NEW = """    _iv=_agefix_iv_compaction(state.ledger,logical_plan,request.num_computed_tokens,allow_closed_frontier)
    if _iv is None:
        kept=[i for begin,end in logical_plan.survivor_intervals for i in range(begin,end)]
        compact=token_compaction(state.ledger.resident_positions(),kept,
                                 old_computed=request.num_computed_tokens,allow_fully_computed=allow_closed_frontier)
        _ki=np.fromiter(compact.keep_indices,dtype=np.int64,count=len(compact.keep_indices))
    else:
        compact,_ki=_iv
"""


def _iv_expand(iv):
    import numpy as np
    lens = iv[:, 1] - iv[:, 0]
    total = int(lens.sum())
    return np.arange(total, dtype=np.int64) + np.repeat(iv[:, 0] - np.concatenate(([0], np.cumsum(lens)[:-1])), lens)


def _iv_fast(es_mod, ledger, logical_plan, old_computed, allow):
    """Production's (TokenCompaction, _ki) from the intervals, or None on anything production would treat differently (it then
    runs and decides). Reads only; the validation memo is written after success."""
    import numpy as np
    if type(allow) is not bool or type(old_computed) is not int:
        return None
    oiv, kiv = ledger.survivor_intervals(), logical_plan.survivor_intervals
    if not oiv or not kiv:
        return None
    o = np.asarray(oiv, dtype=np.int64).reshape(-1, 2)
    k = np.asarray(kiv, dtype=np.int64).reshape(-1, 2)
    if ((o[:, 1] <= o[:, 0]).any() or (k[:, 1] <= k[:, 0]).any() or o[0, 0] < 0 or k[0, 0] < 0
            or (o[1:, 0] < o[:-1, 1]).any() or (k[1:, 0] < k[:-1, 1]).any()):
        return None                                    # not strictly increasing nonnegative positions
    oa, ka = _iv_expand(o), _iv_expand(k)
    R = len(oa)
    if old_computed not in ((R - 1, R) if allow else (R - 1,)) or not R:
        return None
    if not len(ka) or ka[-1] != oa[-1]:
        return None
    idx = np.searchsorted(oa, ka)
    if int(idx[-1]) >= R or not np.array_equal(oa[idx], ka):
        return None
    comp = idx[idx < old_computed]
    if not len(comp):
        return None
    old_t, kept_t = tuple(oa.tolist()), tuple(ka.tolist())
    tc = es_mod.TokenCompaction(old_t, kept_t, tuple(idx.tolist()), tuple(comp.tolist()), old_computed, len(comp))
    for t in (old_t, kept_t):                          # production's _positions memo, as token_compaction leaves it
        if len(es_mod._VALIDATED) >= es_mod._VALIDATED_LIMIT:
            es_mod._VALIDATED.pop(next(iter(es_mod._VALIDATED)))
        es_mod._VALIDATED[id(t)] = t
    return tc, idx


def _iv_production(ee_mod, ledger, logical_plan, old_computed, allow):
    import numpy as np
    kept = [i for begin, end in logical_plan.survivor_intervals for i in range(begin, end)]
    compact = ee_mod.token_compaction(ledger.resident_positions(), kept, old_computed=old_computed, allow_fully_computed=allow)
    return compact, np.fromiter(compact.keep_indices, dtype=np.int64, count=len(compact.keep_indices))


def _iv_same(a, b):
    """Full output equality: every TokenCompaction field by value and container / element type, and _ki by value and dtype."""
    import numpy as np
    (ca, ka), (cb, kb) = a, b
    for f in ('old_positions', 'kept_positions', 'keep_indices', 'computed_keep_indices'):
        x, y = getattr(ca, f), getattr(cb, f)
        if type(x) is not type(y) or x != y or {type(v) for v in x} != {type(v) for v in y}:
            return False
    return (type(ca) is type(cb) and ca.old_computed == cb.old_computed and type(ca.old_computed) is type(cb.old_computed)
            and ca.new_computed == cb.new_computed and type(ca.new_computed) is type(cb.new_computed)
            and ka.dtype == kb.dtype and np.array_equal(ka, kb))


def _install_iv(ee_mod):
    import time
    import eviction_storage as es_mod
    src = open(ee_mod.__file__).read()
    if hashlib.sha256(src.encode()).hexdigest() != EE_PIN:
        raise RuntimeError('agefix 4b: engine_eviction source does not match EE_PIN')
    a = src.index('\ndef prepare_transaction(') + 1
    b = src.index('\ndef ', a + 1) + 1
    fsrc = src[a:b]
    if fsrc.count(_IV_OLD) != 1:
        raise RuntimeError('agefix 4b: compaction anchor not found exactly once')
    orig = ee_mod.prepare_transaction
    calls = [0]

    def _agefix_iv_compaction(ledger, logical_plan, old_computed, allow):
        t0 = time.perf_counter()
        STATS['iv_calls'] += 1
        calls[0] += 1
        try:
            fast = _iv_fast(es_mod, ledger, logical_plan, old_computed, allow)
        except Exception as exc:                                # noqa: BLE001
            STATS['iv_errors'] += 1
            STATS['iv_last_error'] = repr(exc)[:300]
            fast = None
        if fast is None:
            STATS['iv_fallbacks'] += 1
            return None                                         # production's lines run (and raise their own errors)
        STATS['iv_fast'] += 1
        check = IV_VERIFY or (IV_CANARY > 0 and calls[0] % IV_CANARY == 0)
        if check:
            STATS['iv_verify_calls' if IV_VERIFY else 'iv_canary_runs'] += 1
            prod = _iv_production(ee_mod, ledger, logical_plan, old_computed, allow)
            if not _iv_same(fast, prod):
                STATS['iv_canary_mismatches'] += 1
                STATS['iv_last_error'] = 'mismatch at call %d' % calls[0]
                ee_mod._PHASES['iv_compaction'] = ee_mod._PHASES.get('iv_compaction', 0.0) + time.perf_counter() - t0
                return prod
        ee_mod._PHASES['iv_compaction'] = ee_mod._PHASES.get('iv_compaction', 0.0) + time.perf_counter() - t0
        return fast

    ee_mod.__dict__['_agefix_iv_compaction'] = _agefix_iv_compaction
    code = compile(fsrc.replace(_IV_OLD, _IV_NEW), '<agefix4b engine_eviction.prepare_transaction>', 'exec')
    ns = {}
    exec(code, ee_mod.__dict__, ns)
    new = ns['prepare_transaction']
    new.__wrapped__ = orig
    ee_mod.prepare_transaction = new
    STATS['iv_installed'] = True
    return new


PATCHERS = {EE: patch_engine_eviction}
if RP_CACHE or PLAN_MEMO:
    PATCHERS[RES] = patch_residency_all
if VEC:
    PATCHERS[ES] = patch_eviction_storage
    PATCHERS[EQ] = patch_eviction_qsa
if GC_TIMING:
    _gc_timing()
if GCFREEZE_S > 0:
    _gc_freeze_later()
