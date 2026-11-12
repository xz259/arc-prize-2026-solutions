"""stage2_v1 'prestage' part (2026-10-15; user: "Let's do it", after IE's design idea from the final profile): a resumed
persistent session is staged on the worker while the previous step runs, instead of on the critical path of its first
step.

Final profile (pf1, 10-15): a resume leaves the GPU idle ~33 ms before its forward (torch trace, 10 mixed steps): the
engine core pickles the resumed request into the step message (send_execute 3.8 ms vs 0.24 for a plain step: ~54K-id
prompt and prefill lists, ~54K resident positions, the new images' data), the worker unpickles it, and add_requests
stages it (ctxtail: validation, MRoPE rows, QSA lookup; 10.3 ms) before the forward starts. The resume's admission
(core_add) is served on the engine core while the previous step runs (the overlap part), 21.9 ms (p50) before its
schedule, and the worker's main thread then sits idle in its receive.

Engine core. Right after a resumed persistent session's append (EngineCore.add_request; the request is WAITING, its
computed count > 0, its eviction context is ctxtail's incremental one), the core puts a fire-and-forget call on the
worker's rpc broadcast queue (output rank NO_REPLY, which no worker has: the worker runs it and answers nothing, so no
future and no response enter the executor's ordered queues): copies of the dense prompt and prefill ids, the computed
count, a copy of the eviction context and the image features as mmstrip's scheduler hook will send them. The copies are
also the core's snapshot. When a step that schedules the request is dispatched (MultiprocExecutor.execute_model), the
scheduler's NewRequestData is compared with the snapshot (id lists and context by value, computed count, image features
field by field); if equal, a copy of the scheduler output is sent in which those four fields are references (the core's
own objects are untouched), resolved by the worker's unpickler (pickle __reduce__) to the objects the worker received, so
every worker reader sees ordinary lists and dicts. A difference sends the full data (counted); the worker then discards
its copy.
Worker. The call is handled on the main thread between step messages (FIFO): it stores the payload first (references must
always resolve), then runs ctxtail.compute (append-only store, validation, MRoPE rows, lookup table; pure CPU, no GPU call)
for a session the worker already stages. ctxtail.stage_tail takes the result only when the request's NewRequestData holds
exactly the received objects (identity) and the computed count agrees, re-checks the eviction revision, and publishes as
before (lookup copy, scalars, eviction state); otherwise production's path runs. Entries are dropped when taken, when the
request is finished or preempted, and beyond a bound (counted; never expected).
Exactness: ARC3_S2_CTXTAIL_VERIFY=N checks every N-th staging, pre-staged or not, against production's full construction.
ARC3_S2_PRESTAGE_SWITCH=<file> containing 'off' stops new pre-stages (checked at most once a second; within-run A/B).
Timing: per taken entry, late = how far the worker's pre-stage ran past the dispatch of the step that needs it (0 = fully
hidden behind the previous step)."""
import copy
import dataclasses
import os
import time

PINS = {'vllm.v1.engine.core': 'f4b1e07b6d91fac1549bc74a5c804e5a88793d663b5dc045c9e455db6550f5ad',
        'vllm.v1.executor.multiproc_executor': 'b911d837e167592acbfc36689808fc18c697d9624eaea527e56f6ea18b7a719d',
        'vllm.v1.worker.gpu.model_runner': 'f08ba0f5ad41c8b9d5145fb1fba115ffd97f35f1e59017602e235ec0823abce8'}
CONTEXT_KEY = 'arc3_eviction_context'
METHOD = 'arc3_s2_prestage'
NO_REPLY = 1 << 20              # an output rank no worker has
MAX_ENTRIES = 256
LATE_EDGES = (0.0, 2.0, 5.0, 10.0, 20.0)     # ms
STATS = dict(
    # engine core
    sent=0, sent_inflight=0, sent_idle=0, send_ms=0.0, skipped_switch=0, skipped_unsupported=0, send_errors=0,
    stripped=0, mismatch=0, mismatch_prompt=0, mismatch_prefill=0, mismatch_computed=0, mismatch_context=0, mismatch_mm=0,
    strip_ms=0.0, snap_dropped=0, last_core_error=None,
    # worker
    staged=0, staged_none=0, stage_errors=0, stage_ms=0.0, last_error=None, entries_max=0, overflow=0, unresolved=0,
    taken=0, take_unmatched=0, take_noresult=0, dropped_finished=0, late_n=0, late_ms=0.0, late_max_ms=0.0,
    late_hist=[0] * (len(LATE_EDGES) + 1))
_SWITCH = dict(path=os.environ.get('ARC3_S2_PRESTAGE_SWITCH'), t=0.0, off=False)
_GEN = [0]
_SNAP = {}                      # engine core: request id -> snapshot of what was sent
_ENTRIES = {}                   # worker: request id -> received payload and staging result


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
    if n % 200 == 1:
        try:
            from . import install
            install.save(force=False)
        except Exception:                               # noqa: BLE001  receipts are diagnostics
            pass


# ------------------------------------------------------------------------------------------------ references
class _Missing:
    """A reference the worker could not resolve (never expected): fails loudly at its first use."""
    def __init__(self, *args):
        self.args = args

    def _fail(self, *_a, **_k):
        raise RuntimeError(f'stage2 prestage: unresolved reference {self.args!r}')

    __len__ = __iter__ = __getitem__ = __bool__ = __eq__ = _fail
    __hash__ = None


def _ref_value(rid, gen, field, t_dispatch):
    """Worker-side unpickling of a reference: the object received with the request's pre-stage."""
    e = _ENTRIES.get(rid)
    if e is None or e['gen'] != gen:
        STATS['unresolved'] += 1
        return _Missing(rid, gen, field)
    if e['t_dispatch'] is None:
        e['t_dispatch'] = t_dispatch
    return e[field]


class _Ref:
    __slots__ = ('rid', 'gen', 'field', 't')

    def __init__(self, rid, gen, field, t):
        self.rid, self.gen, self.field, self.t = rid, gen, field, t

    def __reduce__(self):
        return _ref_value, (self.rid, self.gen, self.field, self.t)


# ------------------------------------------------------------------------------------------------ engine core
def _strip_like_mmstrip(request, features):
    """The image features as mmstrip's NewRequestData hook will send them (data dropped for images wholly before the
    computed frontier of a persistent request); a new list either way (the request's list is never shared)."""
    from . import install
    frontier = request.num_computed_tokens
    if not features or 'mmstrip' not in install.PARTS or not frontier:
        return list(features or ())
    from . import mmstrip
    if not mmstrip._persistent(request):
        return list(features)
    out = []
    for f in features:
        pos = f.mm_position
        out.append(dataclasses.replace(f, data=None) if f.data is not None and pos.offset + pos.length <= frontier else f)
    return out


def _same_features(a, b):
    if a is b:
        return True
    if a is None or b is None or len(a) != len(b):
        return False
    for x, y in zip(a, b):
        if x is y:
            continue
        if not (x.data is y.data and x.mm_position is y.mm_position and x.modality == y.modality
                and x.identifier == y.identifier and x.mm_hash == y.mm_hash):
            return False
    return True


def _copy_context(ctx):
    return {k: (list(v) if type(v) is list and k != 'logical_image_features' else copy.deepcopy(v)) for k, v in ctx.items()}


def on_add(core, req_id):
    """After an ADD: pre-stage a resumed persistent session (see the module docstring)."""
    try:
        request = core.scheduler.requests.get(req_id)
        if request is None or request.status.name != 'WAITING' or not request.num_computed_tokens:
            return
        params = request.sampling_params
        ctx = (getattr(params, 'extra_args', None) or {}).get(CONTEXT_KEY)
        if not isinstance(ctx, dict) or 'arc3_ids_base' not in ctx:
            return
        import engine_patch
        if not engine_patch.is_persistent(request):
            return
        if _off():
            STATS['skipped_switch'] += 1
            return
        ex = core.model_executor
        mq = getattr(ex, 'rpc_broadcast_mq', None)
        if mq is None or getattr(ex, 'world_size', 1) != 1:
            STATS['skipped_unsupported'] += 1
            return
        t0 = time.perf_counter()
        _GEN[0] += 1
        prompt = list(request.prompt_token_ids)
        shared = request._all_token_ids == request.prompt_token_ids
        prefill = prompt if shared else list(request._all_token_ids)
        snap = dict(gen=_GEN[0], request=request, prompt=prompt, prefill=prefill, computed=request.num_computed_tokens,
                    context=_copy_context(ctx), mm=_strip_like_mmstrip(request, request.mm_features))
        payload = dict(req_id=req_id, gen=snap['gen'], prompt=prompt, prefill=None if shared else prefill,
                       computed=snap['computed'], context=snap['context'], mm=snap['mm'])
        inflight = bool(getattr(core, 'batch_queue', None))
        mq.enqueue((METHOD, (payload,), {}, NO_REPLY))
        _SNAP[req_id] = snap
        STATS['sent'] += 1
        STATS['sent_inflight' if inflight else 'sent_idle'] += 1
        STATS['send_ms'] += (time.perf_counter() - t0) * 1e3
        _save(STATS['sent'])
    except Exception as exc:                            # noqa: BLE001  no snapshot: the step sends the full data
        _SNAP.pop(req_id, None)
        STATS['send_errors'] += 1
        STATS['last_core_error'] = repr(exc)[:300]


def _mismatch(nrd, snap):
    if nrd.num_computed_tokens != snap['computed']:
        return 'computed'
    if nrd.prompt_token_ids != snap['prompt']:
        return 'prompt'
    if nrd.prefill_token_ids != snap['prefill']:
        return 'prefill'
    ctx = (getattr(nrd.sampling_params, 'extra_args', None) or {}).get(CONTEXT_KEY)
    if not isinstance(ctx, dict) or ctx != snap['context']:
        return 'context'
    if not _same_features(nrd.mm_features, snap['mm']):
        return 'mm'
    return None


def strip(scheduler_output):
    """At dispatch: the scheduler output to send (a copy with references for pre-staged requests that still match)."""
    if not _SNAP:
        return scheduler_output
    t0 = time.perf_counter()
    try:
        new = scheduler_output.scheduled_new_reqs or []
        out = None
        for i, nrd in enumerate(new):
            snap = _SNAP.pop(nrd.req_id, None)
            if snap is None:
                continue
            reason = _mismatch(nrd, snap)
            if reason is not None:
                STATS['mismatch'] += 1
                STATS['mismatch_' + reason] += 1
                continue
            rid, g, t = nrd.req_id, snap['gen'], time.monotonic()
            sp = copy.copy(nrd.sampling_params)
            sp.extra_args = dict(nrd.sampling_params.extra_args)
            sp.extra_args[CONTEXT_KEY] = _Ref(rid, g, 'context', t)
            if out is None:
                out = list(new)
            out[i] = dataclasses.replace(nrd, prompt_token_ids=_Ref(rid, g, 'prompt', t),
                                         prefill_token_ids=_Ref(rid, g, 'prefill', t), mm_features=_Ref(rid, g, 'mm', t),
                                         sampling_params=sp)
            STATS['stripped'] += 1
        for rid in [r for r, s in _SNAP.items() if s['request'].is_finished() or s['request'].status.name != 'WAITING']:
            del _SNAP[rid]                              # finished, or scheduled without being a new request
            STATS['snap_dropped'] += 1
        if out is None:
            return scheduler_output
        so = copy.copy(scheduler_output)
        so.scheduled_new_reqs = out
        return so
    except Exception as exc:                            # noqa: BLE001  send the scheduler's own output
        STATS['send_errors'] += 1
        STATS['last_core_error'] = repr(exc)[:300]
        return scheduler_output
    finally:
        STATS['strip_ms'] += (time.perf_counter() - t0) * 1e3


def patch_engine_core(mod):
    cls = mod.EngineCore
    original = cls.add_request

    def add_request(self, request, request_wave=0):
        result = original(self, request, request_wave)
        on_add(self, request.request_id)
        return result

    add_request.__wrapped__ = original
    cls.add_request = add_request


def patch_executor(mod):
    cls = mod.MultiprocExecutor
    original = cls.execute_model

    def execute_model(self, scheduler_output, non_block=False):
        return original(self, strip(scheduler_output), non_block)

    execute_model.__wrapped__ = original
    cls.execute_model = execute_model


# ------------------------------------------------------------------------------------------------ worker
def _worker_prestage(self, payload):
    """WorkerBase method, called by name from the rpc broadcast queue (no reply)."""
    t0 = time.monotonic()
    rid = payload['req_id']
    prompt = payload['prompt']
    prefill = payload['prefill'] if payload['prefill'] is not None else list(prompt)
    e = dict(gen=payload['gen'], prompt=prompt, prefill=prefill, computed=payload['computed'], context=payload['context'],
             mm=payload['mm'], result=None, t0=t0, t_end=None, t_dispatch=None)
    _ENTRIES.pop(rid, None)
    _ENTRIES[rid] = e                                   # first: the step's references must always resolve
    if len(_ENTRIES) > MAX_ENTRIES:
        del _ENTRIES[next(iter(_ENTRIES))]
        STATS['overflow'] += 1
    STATS['entries_max'] = max(STATS['entries_max'], len(_ENTRIES))
    try:
        from . import ctxtail
        e['result'] = ctxtail.prestage_compute(self.model_runner, rid, prefill, payload['computed'], payload['context'])
        STATS['staged' if e['result'] is not None else 'staged_none'] += 1
    except Exception as exc:                            # noqa: BLE001  the staging runs production's path instead
        STATS['stage_errors'] += 1
        STATS['last_error'] = repr(exc)[:300]
    e['t_end'] = time.monotonic()
    STATS['stage_ms'] += (e['t_end'] - t0) * 1e3
    _save(STATS['staged'] + STATS['staged_none'] + STATS['stage_errors'])


def take(req_id, new_req_data, context):
    """ctxtail.stage_tail: the pre-staged (checked, rows, delta, table, cutoff, padding) for this NewRequestData, or None."""
    e = _ENTRIES.pop(req_id, None)
    if e is None:
        return None
    if not (new_req_data.prefill_token_ids is e['prefill'] and context is e['context']
            and new_req_data.num_computed_tokens == e['computed']):
        STATS['take_unmatched'] += 1
        return None
    if e['t_dispatch'] is not None:
        late = max(0.0, (e['t_end'] - e['t_dispatch']) * 1e3)
        STATS['late_n'] += 1
        STATS['late_ms'] += late
        STATS['late_max_ms'] = max(STATS['late_max_ms'], late)
        STATS['late_hist'][sum(late > x for x in LATE_EDGES)] += 1
    if e['result'] is None:
        STATS['take_noresult'] += 1
        return None
    STATS['taken'] += 1
    return e['result']


def patch_model_runner(mod):
    from vllm.v1.worker.worker_base import WorkerBase
    WorkerBase.arc3_s2_prestage = _worker_prestage
    cls = mod.GPUModelRunner
    original = cls.finish_requests

    def finish_requests(self, scheduler_output):
        if _ENTRIES:
            for rid in set(scheduler_output.finished_req_ids or ()) | set(scheduler_output.preempted_req_ids or ()):
                if _ENTRIES.pop(rid, None) is not None:
                    STATS['dropped_finished'] += 1
        for nrd in scheduler_output.scheduled_new_reqs or ():
            for v in (nrd.prompt_token_ids, nrd.prefill_token_ids, nrd.mm_features):
                if isinstance(v, _Missing):
                    v._fail()
        return original(self, scheduler_output)

    finish_requests.__wrapped__ = original
    cls.finish_requests = finish_requests


PATCHERS = {'vllm.v1.engine.core': patch_engine_core, 'vllm.v1.executor.multiproc_executor': patch_executor,
            'vllm.v1.worker.gpu.model_runner': patch_model_runner}
