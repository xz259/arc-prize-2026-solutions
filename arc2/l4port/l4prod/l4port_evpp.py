"""l4port_evpp.py (2026-10-31, ARC2 - main; the user: get the harness working with the PP profile on 4x L4): production's
persistent-session and KV-eviction stack under TP x PP (built for TP2 x PP2; TP > 1 alone takes the same path).

Production assumes one worker in the places below; each is patched after its module is imported (a post-import hook that runs
after stage2_v1's own patchers), the files unchanged. Installed by bootstrap_l4prod when ARC3_L4PORT_EVPP=1.
  1. eviction_runner._limits (worker; stage2's copy and rope2m's wrapper inside): the TP1/PP1 refusal. The inner checks run on
     a view of the runner whose parallel config reports TP = PP = 1; every other check is unchanged.
  2. engine_eviction.preserve_worker_owner_specs (engine core): it refused more than one worker config, and the scheduler's
     config is worker 0's (stage 0's layer names only). The physical owner groups the eviction planner copies from become the
     union over the workers (stage 0's names first; a layer seen on several TP ranks must have equal specs), and
     worker_owner_groups accepts the scheduler's names as their leading subset.
  3. engine_patch.validate_config (the API server's init and every persistent admission in the engine core; stage2's relaxed
     copy during init too): runs on a config view with TP = PP = 1 (the module's class wraps later assignments as well).
  4. persistent_batch_policy.active (engine core): under PP the native batch queue runs (False, so microbatches keep
     pipelining), with the queue capped at PP batches at EngineCore init (vLLM auto-enables async scheduling, and V2 + PP +
     async keeps PP + 1 batches in flight with a decode every PP steps per request: two in-flight steps per request; capped,
     step s is processed before step s + PP is scheduled, so a persistent owner has at most one step in flight, which is what
     production's single-batch policy guarantees at TP 1). The policy asserts the cap. TP alone keeps the policy.
  5. EngineCore.collective_rpc (engine core): calls of production's worker functions (async compaction begin / commit, the
     fused compaction, the two QSA-frequency snapshots; matched by module and name) go to the wrappers below, and the
     per-worker replies are merged into the one reply production checks:
       begin / fused: every worker acknowledges the plan digest, the TP ranks of a stage copied the same layers, and the
         stages together cover the plan's layers exactly once;
       commit: pending while any worker is pending (a worker that already committed answers later polls with its stored
         result), then every worker synchronized on the digest; the receipts of each stage's TP rank 0;
       snapshots: slot, rows, layout and context equal on every worker, the stages' nonzero layers disjoint; counts of each
         stage's lowest TP rank summed over the stages (v2: the TP ranks of a stage are compared and their difference recorded
         instead of refused; l4p37's 15 snapshots all differed on stage 0).
     Other callables pass through (counted, one receipt line each).
  6. Worker wrappers: the compaction runs on the copies of this worker's own layers (its static forward context); the
     worker-process commit wrappers of kvring / draftring still run (resolved through the module attribute, as before).
  7. qsa_frequency.layer_map (worker): under PP the map holds this stage's QSA layers; the counter rows of the other stage's
     layers stay zero.
Not ported: one_lookahead (off: ARC3_PERSISTENT_ONE_LOOKAHEAD unset; its own one-reply checks refuse), prestage (skips itself
when world_size != 1). Receipt lines '[l4port] evpp ...'."""
import functools
import importlib.abc
import sys
import types

VERSION = 'l4port-evpp-v1'
STATS = dict(patched=[], routed={}, unrouted={}, commit_replays=0, commit_pending_polls=0, policy_pp_native=0, errors=0)
_BUSY = set()
_SAID = set()


def _say(key, msg):
    if key in _SAID:
        return
    _SAID.add(key)
    print('[l4port] evpp %s' % msg, flush=True)


class _View:
    """Read-only attribute view of an object with some attributes overridden."""
    __slots__ = ('_o', '_v')

    def __init__(self, obj, **over):
        object.__setattr__(self, '_o', obj)
        object.__setattr__(self, '_v', over)

    def __getattr__(self, name):
        over = object.__getattribute__(self, '_v')
        if name in over:
            return over[name]
        return getattr(object.__getattribute__(self, '_o'), name)


def _sizes(config):
    pc = config.parallel_config
    return int(pc.tensor_parallel_size), int(pc.pipeline_parallel_size)


def _multi(config):
    return _sizes(config) != (1, 1)


def _cfg1(config):
    return _View(config, parallel_config=_View(config.parallel_config, tensor_parallel_size=1, pipeline_parallel_size=1))


# ------------------------------------------------------------------------------------------------ 1. eviction_runner (worker)
def _patch_eviction_runner(mod):
    inner = mod._limits
    if getattr(inner, '_l4port', False):
        return

    @functools.wraps(inner)
    def _limits(runner):
        config = runner.vllm_config
        if not _multi(config):
            return inner(runner)
        _say('limits', 'eviction runner limits under TP %d x PP %d (the TP1/PP1 refusal is skipped)' % _sizes(config))
        return inner(_View(runner, vllm_config=_cfg1(config)))

    _limits._l4port = True
    mod._limits = _limits


# ------------------------------------------------------------------------------------------------ 2. engine_eviction (engine core)
class _Merged(list):
    _l4port_merged = True


def _merge_owner_groups(mod, configs):
    import copy
    import dataclasses
    first = configs[0].kv_cache_groups
    if any(len(c.kv_cache_groups) != len(first) for c in configs):
        raise mod.EngineEvictionError('L4 port: workers have different KV cache group counts')
    merged = _Merged()
    for g, base in enumerate(first):
        spec0 = base.kv_cache_spec
        uniform = getattr(spec0, 'kv_cache_specs', None) is not None
        names, specs = [], {}
        for c in configs:
            grp = c.kv_cache_groups[g]
            spec = grp.kv_cache_spec
            if type(spec) is not type(spec0) or spec.block_size != spec0.block_size:
                raise mod.EngineEvictionError('L4 port: workers disagree on KV cache group %d' % g)
            owners = getattr(spec, 'kv_cache_specs', None)
            if not uniform and spec != spec0:
                # v3 (the reviewer): a non-uniform group is represented by one spec for every stage's layers; require it equal
                raise mod.EngineEvictionError('L4 port: workers disagree on the spec of non-uniform KV cache group %d' % g)
            for name in grp.layer_names:
                own = owners[name] if owners is not None else spec
                if name in specs:
                    if specs[name] != own:
                        raise mod.EngineEvictionError('L4 port: workers disagree on the cache spec of ' + name)
                    continue
                names.append(name)
                specs[name] = own
        spec = dataclasses.replace(spec0, kv_cache_specs={n: specs[n] for n in names}) if uniform else spec0
        merged.append(dataclasses.replace(copy.deepcopy(base), layer_names=list(names), kv_cache_spec=copy.deepcopy(spec)))
    return merged


def _patch_engine_eviction(mod):
    if getattr(mod, '_l4port_evpp', None) == VERSION:
        return
    orig_preserve = mod.preserve_worker_owner_specs
    orig_groups = mod.worker_owner_groups

    def preserve_worker_owner_specs(original):
        stock = orig_preserve(original)

        @functools.wraps(original)
        def generate(configs):
            if len(configs) == 1:
                return stock(configs)
            physical = _merge_owner_groups(mod, configs)
            result = original(configs)
            result._arc3_worker_kv_cache_groups = physical
            mod.worker_owner_groups(result)
            _say('owners', 'eviction owner groups merged over %d workers: layers per group %s (scheduler %s)' % (
                len(configs), [len(g.layer_names) for g in physical], [len(g.layer_names) for g in result.kv_cache_groups]))
            return result
        return generate

    def worker_owner_groups(kv_cache_config):
        groups = getattr(kv_cache_config, '_arc3_worker_kv_cache_groups', None)
        if not getattr(groups, '_l4port_merged', False):
            return orig_groups(kv_cache_config)
        scheduler_groups = kv_cache_config.kv_cache_groups
        if len(groups) != len(scheduler_groups):
            raise mod.EngineEvictionError('Physical per-owner worker cache specs were not preserved')
        for worker, scheduler in zip(groups, scheduler_groups):
            names = list(scheduler.layer_names)
            if (list(worker.layer_names)[:len(names)] != names
                    or worker.kv_cache_spec.block_size != scheduler.kv_cache_spec.block_size):
                raise mod.EngineEvictionError('Worker/scheduler cache group geometry or ownership differs')
            owners = getattr(worker.kv_cache_spec, 'kv_cache_specs', None)
            if owners is not None and (len(owners) != len(worker.layer_names) or set(owners) != set(worker.layer_names)):
                raise mod.EngineEvictionError('Preserved cache owner map does not match group layer names')
        return groups

    mod.preserve_worker_owner_specs = preserve_worker_owner_specs
    mod.worker_owner_groups = worker_owner_groups
    mod._l4port_evpp = VERSION


# ------------------------------------------------------------------------------------------------ 3. engine_patch (API server, core)
def _tolerant_validate(fn):
    @functools.wraps(fn)
    def validate_config(config):
        if not _multi(config):
            return fn(config)
        out = fn(_cfg1(config))
        tp, pp = _sizes(config)
        _say('validate', 'persistent config validated under TP %d x PP %d (the PP1/TP1 refusal is skipped)' % (tp, pp))
        return dict(out, l4port_parallel=dict(tp=tp, pp=pp)) if isinstance(out, dict) else out

    validate_config._l4port = True
    return validate_config


class _EnginePatchModule(types.ModuleType):
    """engine_patch's module class: every function assigned to validate_config (stage2's relaxed copy during the API server's
    init_state, then the restore) is wrapped as well."""

    def __setattr__(self, name, value):
        if name == 'validate_config' and callable(value) and not getattr(value, '_l4port', False):
            value = _tolerant_validate(value)
        types.ModuleType.__setattr__(self, name, value)


def _patch_engine_patch(mod):
    if getattr(type(mod), '__name__', '') == '_EnginePatchModule':
        return
    mod.__class__ = _EnginePatchModule
    mod.validate_config = mod.validate_config


# ------------------------------------------------------------------------------------------------ 4. persistent_batch_policy (core)
def _patch_batch_policy(mod):
    inner = mod.active
    if getattr(inner, '_l4port', False):
        return

    @functools.wraps(inner)
    def active(core, *, is_persistent, paused_status):
        config = core.vllm_config
        tp, pp = _sizes(config)
        if pp > 1:
            size = int(getattr(core, 'batch_queue_size', 1) or 1)
            if size > pp:
                raise mod.BatchPolicyError('L4 port: persistent sessions under PP need the batch queue capped at %d (it is %d)' % (pp, size))
            if not STATS['policy_pp_native']:
                STATS['policy_pp_native'] = 1
                _say('policy', 'single-batch policy off under PP %d: the native batch queue runs, capped at %d batches' % (pp, size))
            return False
        if tp > 1:
            return inner(_View(core, vllm_config=_cfg1(config)), is_persistent=is_persistent, paused_status=paused_status)
        return inner(core, is_persistent=is_persistent, paused_status=paused_status)

    active._l4port = True
    mod.active = active


# ------------------------------------------------------------------------------------------------ 7. qsa_frequency (worker)
def _patch_qsa_frequency(mod):
    inner = mod.layer_map
    if getattr(inner, '_l4port', False):
        return

    @functools.wraps(inner)
    def layer_map(bound_layers):
        from vllm.distributed.parallel_state import get_pp_group
        if get_pp_group().world_size == 1:
            return inner(bound_layers)
        result = {}
        for index, layer_id in enumerate(mod.LAYER_IDS):
            name = mod.CACHE_NAME.format(layer_id)
            layer = bound_layers.get(name)
            if layer is None:
                continue                                         # another pipeline stage's layer
            cache = getattr(layer, 'kv_cache', None)
            if cache is None:
                raise mod.QSAFrequencyError('Missing compressed key cache owner: ' + name)
            pointer = cache.data_ptr()
            if pointer in result or pointer == 0:
                raise mod.QSAFrequencyError('Compressed key caches must have distinct nonzero addresses')
            result[pointer] = index
        if not result:
            raise mod.QSAFrequencyError('L4 port: no compressed key cache owner on this pipeline stage')
        _say('qsa', 'QSA frequency layers on PP rank %d: %s' % (get_pp_group().rank_in_group, sorted(result.values())))
        return result

    layer_map._l4port = True
    mod.layer_map = layer_map


# ------------------------------------------------------------------------------------------------ 6. worker wrappers (RPC targets)
def _ranks():
    from vllm.distributed.parallel_state import get_pp_group, get_tp_group
    return int(get_pp_group().rank_in_group), int(get_tp_group().rank_in_group)


def _local_wire(worker, wire):
    forward = worker.model_runner.vllm_config.compilation_config.static_forward_context
    groups, names = [], []
    for group in wire['groups']:
        copies = [item for item in group['copies'] if item['layer_name'] in forward]
        names.extend(item['layer_name'] for item in copies)
        groups.append(dict(group, copies=copies))
    return dict(wire, groups=groups), names


def l4_worker_compact_async(worker, request_id, wire):
    import engine_eviction_async
    local, names = _local_wire(worker, wire)
    worker.model_runner.__dict__.setdefault('_l4port_commits', {}).pop(request_id, None)
    reply = engine_eviction_async.worker_compact_async(worker, request_id, local)
    pp, tp = _ranks()
    return dict(reply, l4port=dict(pp=pp, tp=tp, layers=names))


def l4_worker_commit_async(worker, request_id):
    import engine_eviction_async
    runner = worker.model_runner
    done = runner.__dict__.setdefault('_l4port_commits', {})
    pending = getattr(runner, '_arc3_async_evictions', None) or {}
    pp, tp = _ranks()
    if request_id not in pending and request_id in done:
        STATS['commit_replays'] += 1
        return dict(done[request_id], l4port=dict(pp=pp, tp=tp, replay=True))
    reply = engine_eviction_async.worker_commit_async(worker, request_id)
    if isinstance(reply, dict) and not reply.get('pending'):
        done[request_id] = reply
        while len(done) > 256:
            done.pop(next(iter(done)))
    return dict(reply, l4port=dict(pp=pp, tp=tp, replay=False))


def l4_worker_compact(worker, request_id, wire):
    import engine_eviction
    local, names = _local_wire(worker, wire)
    reply = engine_eviction.worker_compact(worker, request_id, local)
    pp, tp = _ranks()
    return dict(reply, l4port=dict(pp=pp, tp=tp, layers=names))


def l4_worker_snapshot_compact(worker, request_id, expected_revision, expected_deleted, rows_bound):
    import qsa_frequency
    reply = qsa_frequency.worker_snapshot_compact(worker, request_id, expected_revision, expected_deleted, rows_bound)
    pp, tp = _ranks()
    return dict(reply, l4port=dict(pp=pp, tp=tp))


def l4_worker_snapshot(worker, request_id, resident_positions, logical_computed):
    import qsa_frequency
    reply = qsa_frequency.worker_snapshot(worker, request_id, resident_positions, logical_computed)
    pp, tp = _ranks()
    return dict(reply, l4port=dict(pp=pp, tp=tp))


# ------------------------------------------------------------------------------------------------ 5. merges (engine core)
def _check_count(core, replies):
    world = int(core.vllm_config.parallel_config.world_size)
    if type(replies) is not list or len(replies) != world:
        raise RuntimeError('L4 port: expected %d worker replies, got %r' % (world, None if type(replies) is not list else len(replies)))


def _by_stage(replies):
    stages = {}
    for r in replies:
        meta = r.get('l4port') if isinstance(r, dict) else None
        if not isinstance(meta, dict):
            raise RuntimeError('L4 port: a worker reply lacks its rank tag')
        stages.setdefault(int(meta['pp']), {})[int(meta['tp'])] = r
    return {pp: [tps[t] for t in sorted(tps)] for pp, tps in sorted(stages.items())}


def _coverage(err, replies, wire):
    expected = [item['layer_name'] for group in wire['groups'] for item in group['copies']]
    covered = []
    for pp, rs in _by_stage(replies).items():
        names = rs[0]['l4port']['layers']
        if any(r['l4port']['layers'] != names for r in rs[1:]):
            raise err('L4 port: the TP ranks of pipeline stage %d compacted different layers' % pp)
        covered.extend(names)
    if len(covered) != len(set(covered)) or sorted(covered) != sorted(expected):
        raise err('L4 port: the pipeline stages cover %d of the plan\'s %d layer copies (%d listed)' % (
            len(set(covered) & set(expected)), len(expected), len(covered)))
    return len(covered)


def merge_begin(core, replies, args):
    from engine_eviction import EngineEvictionError
    request_id, wire = args[0], args[1]
    _check_count(core, replies)
    digest = wire.get('plan_digest')
    if any(not isinstance(r, dict) or r.get('plan_digest') != digest or r.get('launched') is not True for r in replies):
        return dict(plan_digest=None, request_id=request_id, launched=False)      # production raises its own error
    layers = _coverage(EngineEvictionError, replies, wire)
    device = 'gpu' if any(r.get('device') == 'gpu' for r in replies) else replies[0].get('device')
    return dict(plan_digest=digest, request_id=request_id, launched=True, layers=layers, device=device,
                l4port=dict(workers=len(replies)))


def merge_commit(core, replies, args):
    request_id = args[0]
    _check_count(core, replies)
    if any(isinstance(r, dict) and r.get('pending') for r in replies):
        STATS['commit_pending_polls'] += 1
        return dict(pending=True, request_id=request_id)
    digests = {r.get('plan_digest') if isinstance(r, dict) else None for r in replies}
    if len(digests) != 1 or any(not isinstance(r, dict) or r.get('worker_synchronized') is not True for r in replies):
        return dict(plan_digest=None, request_id=request_id, worker_synchronized=False)   # production raises
    reps = [rs[0] for rs in _by_stage(replies).values()]
    # v3 (the reviewer, optional hardening): the committed copies cover the begun plan's layers exactly once
    wire = next((e.get('wire') for e in (getattr(core, '_arc3_async_evictions', None) or {}).values()
                 if getattr(e.get('transaction', {}).get('request'), 'request_id', None) == request_id), None)
    if wire is not None:
        expected = sorted(item['layer_name'] for group in wire['groups'] for item in group['copies'])
        got = sorted(c.get('layer_name') for r in reps for c in (r.get('copies') or ()))
        if got != expected:
            STATS['commit_coverage_refusals'] = STATS.get('commit_coverage_refusals', 0) + 1
            print('[l4port] evpp commit coverage: %d committed copies vs %d planned' % (len(got), len(expected)), flush=True)
            return dict(plan_digest=None, request_id=request_id, worker_synchronized=False)   # production raises
    else:
        STATS['commit_coverage_unchecked'] = STATS.get('commit_coverage_unchecked', 0) + 1
    out = dict(plan_digest=digests.pop(), request_id=request_id, worker_synchronized=True,
               copies=[c for r in reps for c in (r.get('copies') or ())],
               byte_verification_enabled=all(bool(r.get('byte_verification_enabled')) for r in replies),
               auxiliary_state_evidence=replies[0].get('auxiliary_state_evidence'), execution=replies[0].get('execution'),
               l4port=dict(workers=len(replies), replays=sum(1 for r in replies if r['l4port'].get('replay'))))
    if 'worker_profile' in reps[0]:
        out['worker_profile'] = reps[0]['worker_profile']
    return out


def merge_fused(core, replies, args):
    from engine_eviction import EngineEvictionError
    request_id, wire = args[0], args[1]
    _check_count(core, replies)
    digest = wire.get('plan_digest')
    if any(not isinstance(r, dict) or r.get('plan_digest') != digest or r.get('worker_synchronized') is not True for r in replies):
        return dict(plan_digest=None, request_id=request_id, worker_synchronized=False)   # production raises
    _coverage(EngineEvictionError, replies, wire)
    reps = [rs[0] for rs in _by_stage(replies).values()]
    out = dict(plan_digest=digest, request_id=request_id, worker_synchronized=True,
               copies=[c for r in reps for c in (r.get('copies') or ())],
               byte_verification_enabled=all(bool(r.get('byte_verification_enabled')) for r in replies),
               auxiliary_state_evidence=replies[0].get('auxiliary_state_evidence'), l4port=dict(workers=len(replies)))
    if 'worker_profile' in reps[0]:
        out['worker_profile'] = reps[0]['worker_profile']
    return out


def _merge_snapshot(core, replies, same):
    import numpy as np
    from qsa_frequency import QSAFrequencyError
    _check_count(core, replies)
    ref = replies[0]
    for r in replies[1:]:
        for key in same:
            if r.get(key) != ref.get(key):
                raise QSAFrequencyError('L4 port: workers disagree on the snapshot\'s %s' % key)
    shape = tuple(int(x) for x in ref['hits_layout']['shape'])
    if ref['hits_layout'].get('dtype') != 'int32' or len(shape) != 2:
        raise QSAFrequencyError('L4 port: unexpected snapshot layout %r' % (ref['hits_layout'],))
    hits = np.zeros(shape, dtype=np.int64)
    informative = np.zeros(len(ref['informative']), dtype=np.int64)
    owner = np.full(shape[0], -1, dtype=np.int64)
    for pp, rs in _by_stage(replies).items():
        h0 = np.frombuffer(rs[0]['hits'], dtype=np.int32).reshape(shape)
        i0 = np.asarray(rs[0]['informative'], dtype=np.int64)
        for r in rs[1:]:
            h = np.frombuffer(r['hits'], dtype=np.int32).reshape(shape)
            i = np.asarray(r['informative'], dtype=np.int64)
            if not np.array_equal(h, h0) or not np.array_equal(i, i0):
                # v2 (l4p37: all 15 snapshots differed between stage 0's TP ranks): the stage's lowest TP rank counts, as one
                # worker would; the size of the difference is recorded (receipts for the first ones, totals in STATS)
                d = np.abs(h.astype(np.int64) - h0.astype(np.int64))
                rec = dict(stage=pp, tp=int(r['l4port']['tp']), hits=int(h0.sum()), abs_diff=int(d.sum()), entries=int((d > 0).sum()),
                           nonzero=int((h0 > 0).sum()), max_diff=int(d.max()) if d.size else 0,
                           layers=[int(x) for x in np.flatnonzero(d.any(axis=1))], informative=[int(x) for x in i0],
                           informative_other=[int(x) for x in i])
                STATS['snapshot_tp_mismatch'] = STATS.get('snapshot_tp_mismatch', 0) + 1
                STATS['snapshot_tp_abs_diff'] = STATS.get('snapshot_tp_abs_diff', 0) + rec['abs_diff']
                STATS['snapshot_tp_hits'] = STATS.get('snapshot_tp_hits', 0) + rec['hits']
                if STATS['snapshot_tp_mismatch'] <= 20:
                    print('[l4port] evpp snapshot TP mismatch %d: %s' % (STATS['snapshot_tp_mismatch'], rec), flush=True)
                # v4 (the reviewer): selector ties stay near 1e-6 of the counts (l4p38); a real divergence between the ranks would not
                if rec['abs_diff'] > 1e-4 * max(1, rec['hits']):
                    raise QSAFrequencyError('L4 port: the TP ranks of stage %d differ by %.2e of the QSA counts (bound 1e-4)'
                                            % (pp, rec['abs_diff'] / max(1, rec['hits'])))
        STATS['snapshot_merges'] = STATS.get('snapshot_merges', 0) + 1
        rows = (h0 != 0).any(axis=1) | (i0 != 0)
        if bool((rows & (owner >= 0)).any()):
            raise QSAFrequencyError('L4 port: two pipeline stages counted the same QSA layer')
        owner[rows] = pp
        hits += h0
        informative += i0
    if hits.size and int(hits.max()) > np.iinfo(np.int32).max:
        raise QSAFrequencyError('L4 port: merged QSA counts overflow int32')
    out = {k: v for k, v in ref.items() if k != 'l4port'}
    out['hits'] = hits.astype(np.int32).tobytes()
    out['informative'] = [int(x) for x in informative]
    return out


def merge_snapshot_compact(core, replies, args):
    return _merge_snapshot(core, replies, ('slot', 'rows', 'hits_layout', 'stored'))


def merge_snapshot(core, replies, args):
    return _merge_snapshot(core, replies, ('slot', 'rows', 'hits_layout', 'context'))


ROUTES = {('engine_eviction_async', 'worker_compact_async'): (l4_worker_compact_async, merge_begin),
          ('engine_eviction_async', 'worker_commit_async'): (l4_worker_commit_async, merge_commit),
          ('engine_eviction', 'worker_compact'): (l4_worker_compact, merge_fused),
          ('qsa_frequency', 'worker_snapshot_compact'): (l4_worker_snapshot_compact, merge_snapshot_compact),
          ('qsa_frequency', 'worker_snapshot'): (l4_worker_snapshot, merge_snapshot)}


def _cap_batch_queue(core):
    """V2 runner + PP + async scheduling keeps PP + 1 batches in flight and lets a request decode again PP steps after its last
    decode (AsyncScheduler's next_decode_eligible_step), so a request can have two steps in flight. A persistent owner must not:
    a second step past its stop would advance its GDN state (production's single-batch policy exists for this). With the queue
    capped at PP, the engine processes step s before it schedules step s + PP, so every request has at most one step in flight
    while the two cadence groups keep the stages pipelined; only async's extra overlap batch is given up. Applied when the
    persistent single-batch policy is configured (ARC3_PERSISTENT_SINGLE_BATCH=1) or ARC3_L4PORT_PP_QUEUE=N asks for a cap
    (core-cell A/Bs)."""
    import collections
    import os
    config = core.vllm_config
    tp, pp = _sizes(config)
    if pp > 1 and not config.scheduler_config.async_scheduling:
        # v3 (the reviewer): the device-side token path, the PP handler and the draft broadcast depend on async scheduling, which
        # the launcher leaves to vLLM's default (None -> True); refuse rather than run a PP engine whose drafts never reach stage 0
        raise RuntimeError('L4 port: PP needs async scheduling on (it resolved False)')
    want = os.environ.get('ARC3_L4PORT_PP_QUEUE', '')
    if want:
        n = int(want)
    elif pp > 1 and config.scheduler_config.async_scheduling and os.environ.get('ARC3_PERSISTENT_SINGLE_BATCH', '0') == '1':
        n = pp
    else:
        return
    size = int(getattr(core, 'batch_queue_size', 1) or 1)
    queue = getattr(core, 'batch_queue', None)
    if queue is None or n >= size:
        _say('queue', 'batch queue %d kept (cap %d)' % (size, n))
        return
    if len(queue):
        raise RuntimeError('L4 port: batch queue not empty at engine init')
    core.batch_queue_size = n
    core.batch_queue = collections.deque(maxlen=n)
    _say('queue', 'batch queue capped %d -> %d (TP %d x PP %d, async scheduling %s)' % (
        size, n, tp, pp, bool(config.scheduler_config.async_scheduling)))


def _install_ppbalance(core):
    """v5 (2026-10-31): l4port_ppbalance.py beside this file when ARC3_L4PORT_PP_BALANCE=1 (the two microbatch parities balanced
    under persistent traffic; l4n3d's step timer: near-empty microbatches took 40% of the span). Off: nothing is loaded."""
    import os
    if os.environ.get('ARC3_L4PORT_PP_BALANCE', '0') != '1':
        return
    import importlib.util
    tp, pp = _sizes(core.vllm_config)
    spec = importlib.util.spec_from_file_location('l4port_ppbalance', os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                                                    'l4port_ppbalance.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    sys.modules['l4port_ppbalance'] = m
    m.install(core, pp, say=lambda msg: print(msg, flush=True))


def _patch_engine_core(mod):
    cls = mod.EngineCore
    if getattr(cls, '_l4port_evpp', None) == VERSION:
        return
    orig_init = cls.__init__

    @functools.wraps(orig_init)
    def __init__(self, *args, **kwargs):
        orig_init(self, *args, **kwargs)
        _cap_batch_queue(self)
        _install_ppbalance(self)

    cls.__init__ = __init__
    orig = cls.collective_rpc

    @functools.wraps(orig)
    def collective_rpc(self, method, timeout=None, args=(), kwargs=None):
        if isinstance(method, str) or not _multi(self.vllm_config):
            return orig(self, method, timeout, args, kwargs)
        key = (getattr(method, '__module__', None), getattr(method, '__name__', None))
        route = ROUTES.get(key)
        if route is None:
            name = '%s.%s' % key
            STATS['unrouted'][name] = STATS['unrouted'].get(name, 0) + 1
            _say('unrouted:' + name, 'worker callable %s is not ported (passed through)' % name)
            return orig(self, method, timeout, args, kwargs)
        wrapper, merge = route
        STATS['routed'][key[1]] = STATS['routed'].get(key[1], 0) + 1
        _say('routed:' + key[1], '%s routed to %d workers (TP %d x PP %d)' % ((key[1], int(self.vllm_config.parallel_config.world_size))
                                                                             + _sizes(self.vllm_config)))
        replies = orig(self, wrapper, timeout, args, kwargs)
        return [merge(self, replies, tuple(args or ()))]

    cls.collective_rpc = collective_rpc
    cls._l4port_evpp = VERSION


# ------------------------------------------------------------------------------------------------ install
PATCHERS = {'eviction_runner': _patch_eviction_runner, 'engine_eviction': _patch_engine_eviction,
            'engine_patch': _patch_engine_patch, 'persistent_batch_policy': _patch_batch_policy,
            'qsa_frequency': _patch_qsa_frequency, 'vllm.v1.engine.core': _patch_engine_core}


def _apply(name, module):
    try:
        PATCHERS[name](module)
        STATS['patched'].append(name)
    except Exception as exc:                                      # noqa: BLE001  reported, then raised: never half-installed
        STATS['errors'] += 1
        print('[l4port] evpp error patching %s: %r' % (name, exc), flush=True)
        raise


class _Finder(importlib.abc.MetaPathFinder):
    _l4port_evpp = VERSION

    def find_spec(self, fullname, path=None, target=None):
        if fullname not in PATCHERS or fullname in _BUSY:
            return None
        _BUSY.add(fullname)
        try:
            for finder in list(sys.meta_path):
                if finder is self or getattr(finder, 'find_spec', None) is None:
                    continue
                spec = finder.find_spec(fullname, path, target)
                if spec is not None:
                    loader = spec.loader
                    orig_exec = loader.exec_module

                    def exec_module(module, _orig=orig_exec, _name=fullname):
                        _orig(module)
                        _apply(_name, module)

                    loader.exec_module = exec_module
                    return spec
            return None
        finally:
            _BUSY.discard(fullname)


def install():
    """Post-import hooks for the modules above (bootstrap_l4prod, after the production chain's head installed stage2_v1's)."""
    if any(getattr(f, '_l4port_evpp', None) == VERSION for f in sys.meta_path):
        return
    for name in PATCHERS:
        if name in sys.modules:
            _apply(name, sys.modules[name])
    sys.meta_path.insert(0, _Finder())
