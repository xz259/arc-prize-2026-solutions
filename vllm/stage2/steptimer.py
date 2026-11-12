"""stage2_v1 steptimer: per-step host phase timings in the engine core and the GPU worker (GAP_PLAN item 1, 2026-09-30).

Off unless ARC3_S2_STEPTIMER_DIR is set (not a part; diagnostics only). Then each process runs a daemon thread that
waits until the target classes are imported and patched by the stage-2 hooks (settle 20 s after first seen), wraps
their methods with time.monotonic() timers, and every 5 s appends the buffered records to
<dir>/steptimer-<pid>.jsonl on local disk; nothing is written on the calling thread. Clock: time.monotonic(), the
clock of the production resume_profile journal, so records align with its resume events across processes.

Records (one per outer call): {'k': kind, 't': start, 'd': seconds, 'p': {phase: seconds}, 'c': {phase: calls}, ...}
  worker  execute_model (phases add_requests, update_requests, prepare_inputs, prepare_attn, ms_add_request,
          evict_stage, rope_init, ms_prepare_attn, gdn_build, shortconv_build, qsa_build; fields: tokens scheduled,
          new requests, their prompt tokens), sample_tokens (phases propose, draft_eager, draft_run_model, draft_attn_meta,
          graph_replay: which drafter passes replay a graph and which run eagerly, with call counts)
  core    schedule (phase allocate_slots; fields: tokens, new requests), core_add (request admission),
          update_from_output (fields of the step it completes: consecutive starts give the step period, unprofiled),
          send_execute / send_sample (executor RPCs)
Nested phase times are inclusive; no GPU reads, no synchronisation."""
import inspect
import json
import os
import sys
import threading
import time

DIR = os.environ.get('ARC3_S2_STEPTIMER_DIR')
CPU = os.environ.get('ARC3_S2_STEPTIMER_CPU') == '1'   # 10-15 (IE, final profile): + thread CPU seconds per outer call ('cpu')
                                                       # and per phase ('pc'), so wall minus CPU separates waiting from work
_BUF = []
_LOCK = threading.Lock()
_CUR = threading.local()
STATS = dict(wrapped=[], records=0, flushes=0, errors=0)

TARGETS = {   # module -> [(class or None for module functions, {name: role}), ...]; role 'outer[:kind]' starts a record,
              # 'phase[:name]' adds its time to the current record (nested phases are inclusive). Only functions called
              # within their own process: never a collective_rpc callable (the RPC serialises it; a wrapper closure
              # holding a thread-local and a lock cannot be pickled).
    'vllm.v1.worker.gpu.model_runner': [('GPUModelRunner', {
        'execute_model': 'outer', 'sample_tokens': 'outer', 'add_requests': 'phase', 'update_requests': 'phase',
        'prepare_inputs': 'phase', 'prepare_attn': 'phase', '_remove_request': 'phase:remove_request'})],
    # v6 (2026-09-30, item 1): the worker's resume admission, component by component.
    'vllm.v1.worker.gpu.states': [('RequestState', {'add_request': 'phase:states_add',
                                                    'apply_staged_writes': 'phase:states_apply'})],
    'vllm.v1.worker.gpu.sample.sampler': [('Sampler', {'add_request': 'phase:sampler_add',
                                                       'apply_staged_writes': 'phase:sampler_apply'})],
    'vllm.v1.worker.gpu.block_table': [('BlockTables', {'append_block_ids': 'phase:block_append'})],
    'vllm.v1.worker.gpu.mm.encoder_cache': [('EncoderCache', {'add_request': 'phase:enc_add'})],
    'vllm.models.qwen3_8_flash_next.nvidia.model_state': [('Qwen3_8FlashNextModelState', {
        'prepare_attn': 'phase:ms_prepare_attn', 'add_request': 'phase:ms_add_request'})],
    'vllm.v1.worker.gpu.mm.rope': [('RopeState', {'init_prefill_positions': 'phase:rope_init',
                                                  'apply_staged_writes': 'phase:rope_apply'})],
    'vllm.models.qwen3_8_flash_next.common.qsa_cache': [('QSAMetadataBuilder', {'build': 'phase:qsa_build'})],
    'vllm.v1.attention.backends.gdn_attn': [('GDNAttentionMetadataBuilder', {'build': 'phase:gdn_build'})],
    'vllm.v1.attention.backends.short_conv_attn': [('PleShortConvAttentionMetadataBuilder', {'build': 'phase:shortconv_build'})],
    'eviction_runner': [(None, {'_stage_request': 'phase:evict_stage', 'validate_context': 'phase:ev_validate',
                                '_mrope_rows': 'phase:ev_mrope', 'historical_virtual_lookup': 'phase:ev_lookup'})],
    'vllm.v1.worker.gpu.spec_decode.autoregressive.speculator': [('AutoRegressiveSpeculator', {
        'propose': 'phase:propose', '_generate_draft': 'phase:draft_eager', '_run_model': 'phase:draft_run_model'})],
    'vllm.v1.worker.gpu.spec_decode.speculator': [('DraftModelSpeculator', {'_build_draft_attn_metadata': 'phase:draft_attn_meta'})],
    'vllm.v1.worker.gpu.cudagraph_utils': [('CudaGraphManager', {'run_fullgraph': 'phase:graph_replay'})],
    'vllm.v1.core.sched.scheduler': [('Scheduler', {'schedule': 'outer', 'update_from_output': 'outer',
                                                    '_update_request_as_session': 'phase:session_append'})],
    'vllm.v1.core.kv_cache_manager': [('KVCacheManager', {'allocate_slots': 'phase:allocate_slots'})],
    'vllm.v1.engine.core': [('EngineCore', {'add_request': 'outer:core_add',
                                            'arc3_qsa_frequency_snapshot_compact': 'outer:snapshot'})],
    'vllm.v1.executor.multiproc_executor': [('MultiprocExecutor', {'execute_model': 'outer:send_execute',
                                                                    'sample_tokens': 'outer:send_sample'})],
    # Resume admission in the engine core (item 1) and asynchronous eviction transactions (item 4: the utilities'
    # durations include the worker round trip).
    'engine_patch': [(None, {'append_session': 'phase:append_session', '_audit': 'phase:audit'})],
    'residency': [('ResidentLedger', {'resident_positions': 'phase:resident_positions', 'plan': 'phase:ledger_plan'})],
    # 2026-10-15 (final profile, IE): the API server's eviction planner and frequency grouping, per call with input sizes, to
    # find what grows with a session's age (plan +0.8 ms, frequency +1.5 ms per eviction of age in g4a / pf1).
    'qsa_frequency': [(None, {'group_frequencies': 'outer:group_frequencies', 'expand_compact_snapshot': 'phase:fq_expand',
                              'compact_positions': 'phase:fq_compact', '_hits_matrix': 'phase:fq_hits',
                              '_positions_digest': 'phase:fq_digest'})],
    'eviction_qsa': [(None, {'qsa_layout': 'phase:fq_layout'})],
    'history_eviction': [(None, {'plan_history_eviction': 'outer:plan_history', '_plan_split': 'phase:pl_split',
                                 '_plan_split_qsa': 'phase:pl_split_qsa', '_hollow_blocks': 'phase:pl_hollow'})],
    'engine_eviction': [(None, {'prepare_transaction': 'phase:prepare_transaction', 'publish_context': 'phase:publish_context',
                                'logical_digest': 'phase:logical_digest'}),
                        ('LogicalState', {'paused': 'phase:paused', 'context': 'phase:context'})],
    'engine_eviction_async': [(None, {'evict_begin': 'outer:evict_begin', 'evict_commit': 'outer:evict_commit'})],
    # 2026-10-15 (item 5, 'prestage'): the worker's pre-stage call (between step messages) and the engine core's send (inside
    # core_add) and reference substitution (inside send_execute).
    'vllm.v1.worker.worker_base': [('WorkerBase', {'arc3_s2_prestage': 'outer:prestage'})],
    'stage2_v1.prestage': [(None, {'on_add': 'phase:prestage_send', 'strip': 'phase:prestage_strip'})],
    # 2026-10-15 ('fqfast'): the numpy frequency step inside group_frequencies (production's phases appear only on verify calls).
    'stage2_v1.fqfast': [(None, {'group_frequencies_fast': 'phase:fqf_fast'})],
}


def _tensor_bytes(obj, depth=0):
    """Bytes of tensors reachable within a few attribute / container levels (multimodal feature payloads)."""
    import torch
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if depth >= 7 or obj is None or isinstance(obj, (int, float, str, bytes)):
        return 0
    if isinstance(obj, dict):
        return sum(_tensor_bytes(v, depth + 1) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return sum(_tensor_bytes(v, depth + 1) for v in obj)
    d = getattr(obj, '__dict__', None)
    return sum(_tensor_bytes(v, depth + 1) for v in d.values()) if d else 0


def _len(x):
    return 0 if x is None else len(x)          # explicit: `x or []` raises on numpy arrays and tensors


def _payload(new):
    """Resume payload composition of scheduled_new_reqs (lengths and byte counts only; nothing is serialised)."""
    out = dict(new_prompt=0, new_prefill=0, new_mm=0, new_mm_bytes=0, ctx_ids=0, ctx_pos=0)
    for r in new:
        out['new_prompt'] += _len(getattr(r, 'prompt_token_ids', None))
        out['new_prefill'] += _len(getattr(r, 'prefill_token_ids', None))
        mm = getattr(r, 'mm_features', None)
        out['new_mm'] += _len(mm)
        out['new_mm_bytes'] += _tensor_bytes(mm)
        extra = getattr(getattr(r, 'sampling_params', None), 'extra_args', None)
        ctx = extra.get('arc3_eviction_context') if isinstance(extra, dict) else None
        if isinstance(ctx, dict):
            out['ctx_ids'] += _len(ctx.get('logical_token_ids'))
            out['ctx_pos'] += _len(ctx.get('resident_positions'))
    return out


def _fields(kind, args, result, kwargs=None):
    try:
        if kind == 'arc3_s2_prestage' and len(args) > 1 and isinstance(args[1], dict):
            p = args[1]
            return dict(ids=_len(p.get('prompt')), pos=_len((p.get('context') or {}).get('resident_positions')),
                        mm=_len(p.get('mm')))
        if kind == 'group_frequencies' and args and isinstance(args[0], dict):
            snap = args[0]; ctx = snap.get('context') or {}
            hits = snap.get('hits')
            return dict(rows=snap.get('rows'), hits_bytes=len(hits) if isinstance(hits, (bytes, bytearray)) else None,
                        logical=ctx.get('logical_tokens'), evicted=_len(ctx.get('evicted')),
                        positions=_len(ctx.get('resident_positions')))
        if kind == 'plan_history_eviction':
            kw = kwargs or {}
            ledger = args[2] if len(args) > 2 else None
            return dict(structural=_len(kw.get('structural')), segments=_len(kw.get('segments')), images=_len(kw.get('images')),
                        logical=getattr(ledger, 'logical_tokens', None), evicted=_len(getattr(ledger, 'evicted', None)))
        if kind == 'execute_model' and len(args) > 1:
            so = args[1]
            new = getattr(so, 'scheduled_new_reqs', None)
            new = [] if new is None else new
            out = dict(tok=getattr(so, 'total_num_scheduled_tokens', None), new=len(new))
            if len(new):
                out.update(_payload(new))
            return out
        if kind == 'update_from_output' and len(args) > 1:    # the step's own scheduler output
            so = args[1]
            new = getattr(so, 'scheduled_new_reqs', None)
            new = [] if new is None else new
            return dict(tok=getattr(so, 'total_num_scheduled_tokens', None), new=len(new))
        if kind == 'schedule' and result is not None:
            new = getattr(result, 'scheduled_new_reqs', None)
            new = [] if new is None else new
            return dict(tok=getattr(result, 'total_num_scheduled_tokens', None), new=len(new))
    except Exception:
        STATS['errors'] += 1
    return {}


def _wrap(fn, kind, role):
    base, _, label = role.partition(':')
    name = label or kind                                   # record kind (outer) or phase name (phase)

    def outer(*args, **kwargs):
        if getattr(_CUR, 'rec', None) is not None:          # re-entrant outer: count as a phase
            return inner(*args, **kwargs)
        rec = dict(k=name, t=time.monotonic(), p={}, c={})
        if CPU:
            rec['pc'] = {}
            c0 = time.thread_time()
        _CUR.rec = rec
        result = None
        try:
            result = fn(*args, **kwargs)
            return result
        finally:
            _CUR.rec = None
            rec['d'] = time.monotonic() - rec['t']
            if CPU:
                rec['cpu'] = time.thread_time() - c0
            rec.update(_fields(kind, args, result, kwargs))
            with _LOCK:
                _BUF.append(rec)

    def inner(*args, **kwargs):
        rec = getattr(_CUR, 'rec', None)
        if rec is None:
            return fn(*args, **kwargs)
        t = time.monotonic()
        c = time.thread_time() if CPU else 0.0
        try:
            return fn(*args, **kwargs)
        finally:
            rec['p'][name] = rec['p'].get(name, 0.0) + time.monotonic() - t
            rec['c'][name] = rec['c'].get(name, 0) + 1
            if CPU and 'pc' in rec:
                rec['pc'][name] = rec['pc'].get(name, 0.0) + time.thread_time() - c

    w = outer if base == 'outer' else inner
    w.__wrapped__ = fn
    w.__name__ = getattr(fn, '__name__', kind)
    return w


def _install_ready():
    done = set()
    seen = {}
    while len(done) < len(TARGETS):
        now = time.monotonic()
        for mod, entries in TARGETS.items():
            if mod in done:
                continue
            m = sys.modules.get(mod)
            if m is None:
                continue
            if now - seen.setdefault(mod, now) < 20:           # let the import-time stage-2 patchers finish
                continue
            for cls_name, methods in entries:
                cls = getattr(m, cls_name, None) if cls_name else m
                if cls is None:
                    continue
                for name, role in methods.items():
                    fn = (cls.__dict__.get(name) if cls_name else None) or getattr(cls, name, None)
                    if not inspect.isfunction(fn) or getattr(fn, '_arc3_steptimer', False):
                        continue
                    w = _wrap(fn, name, role)
                    w._arc3_steptimer = True
                    setattr(cls, name, w)
                    STATS['wrapped'].append(f'{cls_name or mod}.{name}')
            done.add(mod)
        _flush()
        time.sleep(1)


def _flush():
    with _LOCK:
        recs = _BUF[:]
        del _BUF[:]
    if not recs:
        return
    try:
        with open(os.path.join(DIR, f'steptimer-{os.getpid()}.jsonl'), 'a') as f:
            for r in recs:
                f.write(json.dumps(r, separators=(',', ':')) + '\n')
        STATS['records'] += len(recs)
        STATS['flushes'] += 1
    except Exception:
        STATS['errors'] += 1


def _run():
    try:
        _install_ready()
    except Exception:
        STATS['errors'] += 1
    while True:
        time.sleep(5)
        _flush()


def start():
    if not DIR:
        return
    os.makedirs(DIR, exist_ok=True)
    threading.Thread(target=_run, name='arc3-s2-steptimer', daemon=True).start()
