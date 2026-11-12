"""Unvalidated, opt-in numerical candidate: one persistent decode lookahead.

No installation on import. It deliberately drains the whole batch queue and
fences worker streams at each stopping boundary. MM input and actual eviction
are rejected until a later, independently validated candidate expands scope.
"""
from __future__ import annotations

import ast
import copy
import functools
import hashlib
import json
import os
from pathlib import Path

FLAG = 'ARC3_PERSISTENT_ONE_LOOKAHEAD'
MARKER = 'arc3_persistent_one_lookahead'
VERSION = 'arc3-one-lookahead-numerical-v1'
CORE_SHA = 'f4b1e07b6d91fac1549bc74a5c804e5a88793d663b5dc045c9e455db6550f5ad'
_INSTALLED = False


class LookaheadError(RuntimeError):
    pass


def enabled():
    value = os.environ.get(FLAG, '0')
    if value not in ('0', '1'):
        raise LookaheadError(f'{FLAG} must be 0 or 1')
    return value == '1'


def marked_params(params):
    return (getattr(params, 'extra_args', None) or {}).get(MARKER) is True


def marked(request):
    return bool(getattr(request, 'resumable', False)
                and marked_params(request.sampling_params))


def validate_sampling(params):
    extra = getattr(params, 'extra_args', None) or {}
    if MARKER not in extra:
        return
    if type(extra[MARKER]) is not bool:
        raise LookaheadError('One-lookahead marker must be a boolean')
    if extra[MARKER] is False:
        return
    if not enabled():
        raise LookaheadError('Explicit marker requires the isolated candidate flag')
    if extra.get('arc3_persistent_state') is not True:
        raise LookaheadError('One-lookahead requires the persistent marker')
    for key, expected in (('repetition_penalty', 1.0),
                          ('presence_penalty', 0.0), ('frequency_penalty', 0.0)):
        if getattr(params, key, None) != expected:
            raise LookaheadError('Initial candidate requires neutral token penalties')
    if getattr(params, 'bad_words', None) or getattr(params, 'logit_bias', None):
        raise LookaheadError('Initial candidate excludes bad words and logit bias')
    if getattr(params, 'thinking_token_budget', None) is not None:
        raise LookaheadError('Initial candidate excludes mutable thinking-budget state')


def validate_admission(scheduler, request, existing):
    if existing is not None and marked(existing) != marked(request):
        raise LookaheadError('Cannot change the lookahead contract within a session')
    if not marked(request):
        return
    config = scheduler.vllm_config
    if not config.use_v2_model_runner or not config.scheduler_config.async_scheduling:
        raise LookaheadError('One-lookahead requires MRV2 and async scheduling')
    if request.mm_features or getattr(request, 'lora_request', None) is not None:
        raise LookaheadError('Initial numerical candidate excludes MM and LoRA')
    if getattr(request, 'num_preemptions', 0):
        raise LookaheadError('One-lookahead cannot resume a preempted owner')
    if existing is not None and getattr(existing, '_arc3_la_pending', False):
        raise LookaheadError('Prior stop has not drained and been acknowledged')


def validate_append(session, update, delta):
    if marked(session) != marked_params(update.sampling_params):
        raise LookaheadError('Persistent append changed the lookahead marker')
    if not marked(session):
        return
    if getattr(session, '_arc3_la_pending', False):
        raise LookaheadError('Append arrived before stop acknowledgement')
    if not getattr(session, '_arc3_la_fenced', False):
        raise LookaheadError('Append requires the acknowledged worker fence')
    if not delta and len(session._all_token_ids) == session.num_computed_tokens:
        raise LookaheadError('Fully computed EOS boundary requires nonempty append')
    if update.mm_features:
        raise LookaheadError('Initial numerical candidate excludes MM append')


def skip(request):
    if getattr(request, '_arc3_la_pending', False):
        return True
    inflight = request.num_in_flight_tokens
    return bool(inflight and (inflight >= 2
                             or getattr(request, '_arc3_la_last_kind', None) != 'decode'))


def after_schedule_check(request, count):
    start = request.num_computed_tokens
    decode = start >= request.num_prompt_tokens
    if decode and count != 1:
        raise LookaheadError('Lookahead decode must schedule exactly one input token')
    if request.num_in_flight_tokens:
        if (request.num_in_flight_tokens != 1 or not decode or count != 1
                or getattr(request, '_arc3_la_last_kind', None) != 'decode'):
            raise LookaheadError('Only one additional decode step may be outstanding')
    if getattr(request, '_arc3_la_pending', False):
        raise LookaheadError('Attempted scheduling during a stop transaction')
    request._arc3_la_last_kind = 'decode' if decode else 'prefill'
    request._arc3_la_fenced = False


def pending(scheduler):
    return getattr(scheduler, '_arc3_la_transactions', {})


def has_pending_work(scheduler, original):
    # A close/abort of an already paused owner has no queued model step and
    # native has_requests() can be false. Keep the core input loop awake long
    # enough to run the empty-queue fence/free transaction exactly once.
    return bool(pending(scheduler)) or original(scheduler)


def _transaction(scheduler, request):
    transactions = getattr(scheduler, '_arc3_la_transactions', None)
    if transactions is None:
        transactions = scheduler._arc3_la_transactions = {}
    request._arc3_la_pending = True
    return transactions.setdefault(request.request_id,
        dict(request=request, held=None, dropped=[], free=None))


def prepare_stop(scheduler, request):
    if request.streaming_queue and any(x is not None for x in request.streaming_queue):
        raise LookaheadError('Next observation was queued before acknowledged stop')
    if request.num_in_flight_tokens not in (0, 1):
        raise LookaheadError('Stop has more than one remaining decode input')
    if request.num_output_placeholders != request.num_in_flight_tokens:
        raise LookaheadError('Stop reservations are not single-token decode reservations')
    if len(request._all_token_ids)-request.num_computed_tokens not in (0, 1):
        raise LookaheadError('Stop has computed an unaccepted input token')
    _transaction(scheduler, request)


def defer_free(scheduler, request, args, kwargs):
    tx = _transaction(scheduler, request)
    tx['free'] = (args, kwargs)
    return None, None


def consume_discard(scheduler, request, token_ids, *, is_stale=False):
    """Called inside native AsyncScheduler's super() update, after I decrement.

    Returning [] prevents its outer decrement. Settle the one reserved sample
    here, explicitly; do not reuse preemption/stale-output mechanisms.
    """
    tx = pending(scheduler).get(request.request_id)
    if tx is None or tx['held'] is None or tx['free'] is not None:
        raise LookaheadError('Output suppression lacks a pending stopping output')
    if (is_stale or len(token_ids) != 1 or request.num_in_flight_tokens != 0
            or request.num_output_placeholders != 1 or tx['dropped']):
        raise LookaheadError('Unexpected second-result reservation or sample shape')
    request.num_output_placeholders -= 1
    tx['dropped'].extend(token_ids)
    return [], False


def _can_schedule(core):
    return not bool(pending(core.scheduler))


def build_queue(source, namespace, *, serial_policy):
    if hashlib.sha256(source.encode()).hexdigest() != CORE_SHA:
        raise LookaheadError('Unreviewed EngineCore source')
    if serial_policy:
        import persistent_batch_policy as bp
        fn = bp.transform(source)
    else:
        tree = ast.parse(source)
        cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'EngineCore')
        fn = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                                and n.name == 'step_with_batch_queue'))
    anchor = 'self.scheduler.has_requests()'
    found = 0
    for node in fn.body:
        if isinstance(node, ast.If) and anchor in ast.unparse(node.test):
            node.test = ast.BoolOp(op=ast.And(), values=[node.test,
                ast.parse('_arc3_la_can_schedule(self)', mode='eval').body])
            found += 1
    if found != 1:
        raise LookaheadError('Expected one queue scheduling guard')
    scope = dict(namespace, _arc3_la_can_schedule=_can_schedule)
    if serial_policy:
        import engine_patch
        from vllm.v1.request import RequestStatus
        scope[bp.HELPER_NAME] = functools.partial(bp.active,
            is_persistent=engine_patch.is_persistent,
            paused_status=RequestStatus.WAITING_FOR_STREAMING_REQ)
    module = ast.Module(body=[ast.ImportFrom(module='__future__',
        names=[ast.alias(name='annotations')], level=0), fn], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), '<one-lookahead-native-queue>', 'exec'), scope)
    return scope[fn.name]


def worker_settle(worker, wires):
    """Fence, validate the actual frontier, then remove sampled-only metadata.

    Does not call model forward, reset a recurrent state or touch KV storage.
    Numerical equivalence and GDN/cache byte equality still require GPU tests.
    """
    import torch
    runner = worker.model_runner
    torch.cuda.synchronize()
    states = runner.req_states
    prepared = []
    for wire in wires:
        ids, computed = wire['accepted_ids'], wire['computed']
        dropped = wire['dropped']
        if len(ids)-computed not in (0, 1) or len(dropped) not in (0, 1):
            raise LookaheadError('Invalid worker settlement frontier')
        slot = states.req_id_to_index[wire['request_id']]
        actual_computed = int(states.num_computed_tokens.gpu[slot].item())
        actual_total = int(states.total_len.gpu[slot].item())
        actual_ids = states.all_token_ids.gpu[slot, :len(ids)+len(dropped)].cpu().tolist()
        actual_last = int(states.last_sampled_tokens[slot, 0].item())
        if (actual_computed != computed or actual_total != len(ids)+len(dropped)
                or actual_ids != ids+dropped or actual_last != (dropped or ids)[-1]):
            raise LookaheadError('Actual worker token frontier disagrees with accepted stop/drain')
        prepared.append((wire, slot))
    for wire, slot in prepared:
        ids = wire['accepted_ids']
        if wire['dropped']:
            # Zero only the now-inaccessible sampled tail, preserving all
            # accepted token IDs and every computed cache/recurrent input.
            states.all_token_ids.stage_write(slot, len(ids), [0])
            states.total_len.stage_write_elem(slot, len(ids))
            states.last_sampled_tokens[slot, 0] = ids[-1]
        states.num_computed_tokens_np[slot] = wire['computed']
    states.all_token_ids.apply_write()
    states.total_len.apply_write()
    torch.cuda.synchronize()
    return dict(version=VERSION, fenced=True, requests=[dict(
        request_id=w['request_id'], computed=w['computed'], accepted=len(w['accepted_ids']),
        delta=len(w['accepted_ids'])-w['computed'], discarded=len(w['dropped'])) for w in wires])


def _capture_outputs(scheduler, outputs):
    if outputs is None:
        return
    transactions = pending(scheduler)
    for client, container in outputs.items():
        keep = []
        for output in container.outputs:
            tx = transactions.get(output.request_id)
            if tx is None:
                keep.append(output)
            elif output.finish_reason is not None:
                if tx['held'] is not None:
                    raise LookaheadError('Duplicate final output in stop transaction')
                tx['held'] = (client, output)
            elif output.new_token_ids:
                raise LookaheadError('Model-visible output escaped a draining owner')
        container.outputs = keep


def settle_core(core, outputs, original_free, output_class):
    transactions = pending(core.scheduler)
    if not transactions or core.batch_queue:
        return outputs
    # All owners, not just stopped owners, must be drained before touching
    # worker metadata. This intentionally sacrifices overlap at turn stops.
    if any(r.num_in_flight_tokens for r in core.scheduler.requests.values()):
        raise LookaheadError('Empty batch queue still has outstanding owner inputs')
    wires = []
    for tx in transactions.values():
        request = tx['request']
        if tx['free'] is None:
            if request.num_output_placeholders or request.num_stale_output_tokens:
                raise LookaheadError('Stopping owner has unsettled output reservations')
            if tx['held'] is None:
                raise LookaheadError('Stop transaction has no held final output')
            wires.append(dict(request_id=request.request_id,
                accepted_ids=list(request._all_token_ids), computed=request.num_computed_tokens,
                dropped=list(tx['dropped'])))
    replies = core.collective_rpc(worker_settle, timeout=120, args=(wires,))
    if (len(replies) != 1 or replies[0].get('version') != VERSION
            or replies[0].get('fenced') is not True
            or {r['request_id'] for r in replies[0]['requests']} != {w['request_id'] for w in wires}):
        raise LookaheadError('Missing or ambiguous all-stream settlement receipt')
    outputs = {} if outputs is None else outputs
    import engine_patch
    for request_id, tx in list(transactions.items()):
        request = tx['request']
        if tx['free'] is not None:
            # Aborted samples were skipped by native output handling; no
            # resumption is allowed, but their buffers remained owned to here.
            request.num_output_placeholders = 0
            request._arc3_la_allow_free = True
            try:
                original_free(core.scheduler, request, *tx['free'][0], **tx['free'][1])
            finally:
                request._arc3_la_allow_free = False
        else:
            request._arc3_la_fenced = True
            request._arc3_la_last_settlement = dict(
                delta=len(request._all_token_ids)-request.num_computed_tokens,
                discarded_token_ids=list(tx['dropped']), worker_fenced=True)
            engine_patch.assert_settled(request)
            engine_patch._audit(core.scheduler, 'pause', request,
                one_lookahead=copy.deepcopy(request._arc3_la_last_settlement))
        request._arc3_la_pending = False
        held = tx['held']
        if held is not None and request.status.name != 'FINISHED_ABORTED':
            client, output = held
            outputs.setdefault(client, output_class()).outputs.append(output)
        del transactions[request_id]
    return outputs


def core_diagnostic(core, external):
    """Explicit, slow, read-only numerical diagnostic; never a decode hook."""
    import engine_patch
    import engine_eviction
    from one_lookahead_snapshot import worker_snapshot
    if not enabled() or core.batch_queue:
        raise LookaheadError('Numerical snapshot requires enabled candidate and empty queue')
    request = engine_eviction._request_for_external(core.scheduler, external)
    if request.status.name != 'WAITING_FOR_STREAMING_REQ' or getattr(request, '_arc3_la_pending', False):
        raise LookaheadError('Numerical snapshot requires acknowledged pause')
    engine_patch.assert_settled(request)
    state = getattr(request, '_arc3_eviction_state', None)
    if (state is not None and state.deleted_tokens) or request.mm_features:
        raise LookaheadError('Numerical snapshot excludes eviction and multimodal input')
    computed = request.num_computed_tokens
    wire = dict(request_id=request.request_id, computed_tokens=computed,
        accepted_token_ids=list(request._all_token_ids),
        block_ids=[list(group) for group in core.scheduler.kv_cache_manager.get_blocks(
            request.request_id).get_block_ids()], logical_positions=list(range(computed)),
        original_prompt_tokens=request.num_prompt_tokens)
    replies = core.collective_rpc(worker_snapshot, timeout=120, args=(wire,))
    if len(replies) != 1:
        raise LookaheadError('Numerical snapshot requires exactly one worker')
    return dict(version=VERSION, computed_tokens=computed,
        accepted_tokens=len(request._all_token_ids),
        delta=len(request._all_token_ids)-computed,
        settlement=copy.deepcopy(getattr(request, '_arc3_la_last_settlement', None)),
        worker=replies[0])


def install():
    global _INSTALLED
    if not enabled():
        return dict(version=VERSION, installed=False)
    if _INSTALLED:
        return dict(version=VERSION, installed=True)
    import vllm
    runtime = Path(vllm.__file__).resolve().parent
    pins = json.loads(Path(__file__).with_name('runtime-pins.json').read_text())
    for path, digest in pins.items():
        if hashlib.sha256((runtime/path).read_bytes()).hexdigest() != digest:
            raise LookaheadError(f'Unreviewed one-lookahead runtime source: {path}')
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.engine import core as cm
    from vllm.v1.engine import EngineCoreOutputs
    import engine_eviction
    original_update = Scheduler._update_request_with_output
    original_free = Scheduler._free_request
    original_has_requests = Scheduler.has_requests
    original_step = cm.EngineCore.step_with_batch_queue
    source = Path(cm.__file__).read_text()
    serial = getattr(cm.EngineCore, '_arc3_batch_policy', None) is not None
    queue = build_queue(source, cm.__dict__, serial_policy=serial)

    @functools.wraps(original_update)
    def update(self, request, token_ids, is_stale=False):
        if marked(request) and getattr(request, '_arc3_la_pending', False):
            return consume_discard(self, request, token_ids, is_stale=is_stale)
        return original_update(self, request, token_ids, is_stale=is_stale)

    @functools.wraps(original_has_requests)
    def has_requests(self):
        return has_pending_work(self, original_has_requests)

    @functools.wraps(original_step)
    def step(self):
        if any(marked(r) for r in self.scheduler.requests.values()):
            if self.batch_queue_size != 2 or not self.async_scheduling:
                raise LookaheadError('One-lookahead requires exact native queue depth 2')
        outputs, executed = queue(self)
        _capture_outputs(self.scheduler, outputs)
        outputs = settle_core(self, outputs, original_free, EngineCoreOutputs)
        return outputs, executed

    original_prepare = engine_eviction.prepare_transaction
    @functools.wraps(original_prepare)
    def reject_unvalidated_eviction(core, external, payload):
        request = engine_eviction._request_for_external(core.scheduler, external)
        if marked(request):
            raise LookaheadError('Numerical candidate has not validated eviction at delta 0')
        return original_prepare(core, external, payload)

    Scheduler._update_request_with_output = update
    Scheduler.has_requests = has_requests
    cm.EngineCore.step_with_batch_queue = step
    cm.EngineCore.arc3_one_lookahead_diagnostic = core_diagnostic
    engine_eviction.prepare_transaction = reject_unvalidated_eviction
    _INSTALLED = True
    return dict(version=VERSION, installed=True, numerical_candidate=True,
        multimodal=False, eviction=False, neutral_penalties_only=True,
        native_queue_capacity=2, global_drain_at_stops=True)
