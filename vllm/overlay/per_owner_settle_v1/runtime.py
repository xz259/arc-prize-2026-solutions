"""Per-owner settlement v2 for the one-lookahead persistent engine (stage 6 of THROUGHPUT_MEMORY_PLAN_2026-09-19).

Served protocol: a stop blocks all scheduling until the two-deep batch queue is empty, then `worker_settle` fences
the whole device (torch.cuda.synchronize twice) and validates/repairs the owner's slot. Measured ~77-89 ms per settle
event at N39 (two step times of lost overlap + the RPC).

v1 (per-owner readiness, synchronous fence) recovered nothing (44K/N39 staggered: 1,204.6/1,138.1 vs 1,199.1/1,131.9):
under async scheduling the worker routes every reply, RPC replies included, through one FIFO output thread
(multiproc_executor.WorkerProc.handle_output -> async_output_busy_loop -> enqueue_output), and an in-flight step's
AsyncModelRunnerOutput ahead of the reply is only emitted once that step finishes on the GPU. A synchronous fence
therefore parks the core thread until the in-flight step completes, no next step is enqueued, and the GPU idles for
about one step per fence (the 33-38 ms "partial" bubbles in the v1 journal).

v2 keeps v1's readiness rule and every validation/write/audit of the served settlement, and makes the fence
non-blocking:
  scheduler (core): a ready stop transaction issues `worker_settle` with collective_rpc(non_block=True) and stays
      pending; it completes on a later scheduler iteration once the reply is in. The reply is resolved for free when
      the native loop drains an older step future, or explicitly when the fence is the oldest outstanding RPC (its
      reply is already queued right behind the step output the loop just consumed). With an empty batch queue the
      fence is resolved synchronously (nothing is in flight, so the wait is the RPC alone); this is what a keep-pause
      drain (eviction) relies on: no transaction stays pending once the queue is empty.
  worker: unchanged from v1 (owner-scoped event wait on a side stream, repairs applied on the main stream).
Everything else (`prepare_stop`, `consume_discard`, `_capture_outputs`, eviction admission, audits) is unchanged.
"""
import json
import os
import time

STATS = dict(version='arc3-per-owner-settle-v2', pid=os.getpid(), installed={}, settled=0, rpcs=0, deferred=0,
             fences=0, resolved_oldest=0, resolved_empty_queue=0, fence_wait=0, errors=[])


def receipt():
    root = os.environ.get('ARC3_PER_OWNER_SETTLE_RECEIPTS')
    if not root:
        return
    try:
        os.makedirs(root, exist_ok=True); STATS['epoch'] = time.time()
        with open(os.path.join(root, f'proc-{os.getpid()}.json'), 'w') as f:
            json.dump(STATS, f, indent=1)
    except Exception as e:
        STATS['errors'].append(repr(e))


def install_one_lookahead(module):
    LookaheadError = module.LookaheadError
    VERSION = module.VERSION
    pending = module.pending

    def _can_schedule(core):
        return True   # other owners keep stepping during a stop transaction

    def _fences(core):
        table = core.__dict__.get('_arc3_pos_fences')
        if table is None:
            table = core._arc3_pos_fences = {}   # request_id -> (future, frozenset of request_ids in that wire batch)
        return table

    def _resolve(core, fences):
        """Collect fences whose reply is in. Never waits on GPU work: a fence is resolved only when already done,
        when it is the oldest outstanding RPC (reply already emitted behind the consumed step output), or when the
        batch queue is empty (nothing in flight)."""
        executor = core.model_executor
        queue = getattr(executor, 'futures_queue', None)
        idle = not getattr(core, 'batch_queue', None)
        done = {}
        for request_id, (future, ids) in list(fences.items()):
            if not future.done():
                if queue is not None and len(queue) and queue[-1] is future:
                    future.result(); STATS['resolved_oldest'] += 1
                elif idle:
                    future.result(); STATS['resolved_empty_queue'] += 1
                else:
                    STATS['fence_wait'] += 1
                    continue
            done[request_id] = (future, ids)
        checked = set()
        for future, ids in done.values():
            if id(future) in checked:
                continue
            checked.add(id(future))
            replies = future.result()
            if (len(replies) != 1 or replies[0].get('version') != VERSION or replies[0].get('fenced') is not True
                    or {r['request_id'] for r in replies[0]['requests']} != set(ids)):
                raise LookaheadError('Missing or ambiguous owner settlement receipt')
            STATS['rpcs'] += 1
        return done

    def settle_core(core, outputs, original_free, output_class):
        transactions = pending(core.scheduler)
        if not transactions:
            return outputs
        fences = _fences(core)
        # (1) issue fences for transactions that became ready (own in-flight count zero, final output held)
        wires, wired, free_now = [], [], []
        for request_id, tx in transactions.items():
            if request_id in fences:
                continue
            request = tx['request']
            if request.num_in_flight_tokens:
                continue                                   # its last (lookahead) step has not returned yet
            if tx['free'] is None:
                if tx['held'] is None:
                    continue                               # final output not yet captured
                if request.num_output_placeholders or request.num_stale_output_tokens:
                    raise LookaheadError('Stopping owner has unsettled output reservations')
                wires.append(dict(request_id=request.request_id, accepted_ids=list(request._all_token_ids),
                                  computed=request.num_computed_tokens, dropped=list(tx['dropped'])))
                wired.append(request_id)
            else:
                free_now.append(request_id)                # abort/free path needs no worker fence (as served)
        if wires:
            future = core.model_executor.collective_rpc(module.worker_settle, timeout=120, args=(wires,), non_block=True)
            ids = frozenset(wired)
            for request_id in wired:
                fences[request_id] = (future, ids)
            STATS['fences'] += 1
        # (2) complete transactions whose fence reply is in (never waiting on GPU work)
        done = _resolve(core, fences)
        ready = list(done) + free_now
        if not ready:
            STATS['deferred'] += 1
            return outputs
        outputs = {} if outputs is None else outputs
        import engine_patch
        import copy
        for request_id in ready:
            tx = transactions[request_id]
            request = tx['request']
            fences.pop(request_id, None)
            if tx['free'] is not None:
                request.num_output_placeholders = 0
                request._arc3_la_allow_free = True
                try:
                    original_free(core.scheduler, request, *tx['free'][0], **tx['free'][1])
                finally:
                    request._arc3_la_allow_free = False
            else:
                request._arc3_la_fenced = True
                request._arc3_la_last_settlement = dict(
                    delta=len(request._all_token_ids) - request.num_computed_tokens,
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
            STATS['settled'] += 1
        if STATS['settled'] % 500 < len(ready):
            receipt()
        return outputs

    def worker_settle(worker, wires):
        """Owner-scoped fence: wait for the owner's recorded last-step event, validate on a side stream, repair on
        the main stream. Same checks and writes as the served version; no device-wide synchronize."""
        import torch
        runner = worker.model_runner
        states = runner.req_states
        events = getattr(runner, '_arc3_last_step_event', None) or {}
        side = getattr(runner, '_arc3_settle_stream', None)
        if side is None:
            side = runner._arc3_settle_stream = torch.cuda.Stream(device=runner.device)
        main = torch.cuda.current_stream()
        prepared = []
        with torch.cuda.stream(side):
            for wire in wires:
                ids, computed = wire['accepted_ids'], wire['computed']
                dropped = wire['dropped']
                if len(ids) - computed not in (0, 1) or len(dropped) not in (0, 1):
                    raise LookaheadError('Invalid worker settlement frontier')
                slot = states.req_id_to_index[wire['request_id']]
                event = events.get(wire['request_id'])
                if event is None:
                    raise LookaheadError('Stopping owner has no recorded step fence')
                side.wait_event(event)
                actual_computed = int(states.num_computed_tokens.gpu[slot].item())
                actual_total = int(states.total_len.gpu[slot].item())
                actual_ids = states.all_token_ids.gpu[slot, :len(ids) + len(dropped)].cpu().tolist()
                actual_last = int(states.last_sampled_tokens[slot, 0].item())
                if (actual_computed != computed or actual_total != len(ids) + len(dropped)
                        or actual_ids != ids + dropped or actual_last != (dropped or ids)[-1]):
                    raise LookaheadError('Actual worker token frontier disagrees with accepted stop/drain')
                prepared.append((wire, slot))
        for wire, slot in prepared:
            ids = wire['accepted_ids']
            if wire['dropped']:
                states.all_token_ids.stage_write(slot, len(ids), [0])
                states.total_len.stage_write_elem(slot, len(ids))
                states.last_sampled_tokens[slot, 0] = ids[-1]
            states.num_computed_tokens_np[slot] = wire['computed']
        with torch.cuda.stream(main):
            states.all_token_ids.apply_write()
            states.total_len.apply_write()
        for wire, _ in prepared:
            events.pop(wire['request_id'], None)
        return dict(version=VERSION, fenced=True, requests=[dict(
            request_id=w['request_id'], computed=w['computed'], accepted=len(w['accepted_ids']),
            delta=len(w['accepted_ids']) - w['computed'], discarded=len(w['dropped'])) for w in wires])

    module._can_schedule = _can_schedule
    module.settle_core = settle_core
    module.worker_settle = worker_settle
    STATS['installed']['one_lookahead'] = ['_can_schedule', 'settle_core', 'worker_settle']
    receipt()


def install_model_runner(module):
    """Worker: tag every scheduled request with an event recorded after its step on the main stream."""
    R = module.GPUModelRunner
    original = R.execute_model

    def execute_model(self, scheduler_output, *args, **kwargs):
        result = original(self, scheduler_output, *args, **kwargs)
        try:
            import torch
            table = self.__dict__.get('_arc3_last_step_event')
            if table is None:
                table = self._arc3_last_step_event = {}
            ids = list(scheduler_output.num_scheduled_tokens)
            if ids:
                event = torch.cuda.Event(); event.record(torch.cuda.current_stream())
                for rid in ids:
                    table[rid] = event
            if len(table) > 4 * max(1, self.max_num_reqs):
                live = self.req_states.req_id_to_index
                for rid in [r for r in table if r not in live]:
                    del table[rid]
        except Exception as e:
            STATS['errors'].append(f'event: {e!r}')
        return result

    R.execute_model = execute_model
    STATS['installed']['GPUModelRunner.execute_model'] = 'last-step-event'
    receipt()
