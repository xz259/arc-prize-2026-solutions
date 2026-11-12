"""Non-draining eviction transaction (stage C-lite, 2026-09-23).

The fused transaction pauses the scheduler, drains every in-flight batch, plans and copies on the core thread, then
resumes: 250-400 ms during which no lane steps. This variant keeps the same plan (engine_eviction.prepare_transaction
with the engine-wide quiescence checks relaxed to owner-local ones), launches the worker copies on a dedicated CUDA
stream and returns; a second utility commits once the copy event has completed. Other lanes keep stepping throughout:
the paused owner's pages are private (ref_cnt == 1, prefix caching off), no in-flight kernel touches them, and the
owner cannot be scheduled until its next append, which the API holds until commit.

Utilities: arc3_evict_begin(external, payload) -> {begun, plan_digest, ...}; arc3_evict_commit(external) -> {pending}
or the acknowledgement of engine_eviction.evict_resident (with execution="async"). Selected by ARC3_EVICTION_MODE=async;
the fused path stays installed for fallback. On CPU (tests) the copies run inline and commit is immediate.
"""
from __future__ import annotations

import contextlib
import copy
import os
import time

import engine_eviction as ee
from engine_eviction import EngineEvictionError

VERSION = "arc3-eviction-async-v1"
BEGIN = "arc3_evict_begin"
COMMIT = "arc3_evict_commit"


def _pending_map(obj, name):
    table = getattr(obj, name, None)
    if table is None:
        table = {}
        setattr(obj, name, table)
    return table


# ------------------------------------------------------------------------------------------------------ worker side
def worker_compact_async(worker, request_id, wire):
    """Validate the context and cache layouts, launch the row copies on the eviction stream, record an event, return."""
    import torch
    from eviction_runner import validate_context, historical_virtual_lookup, PreparedWorkerEviction, _limits, EvictionRunnerError, trusted_ids
    from eviction_diagnostics import describe_cache
    from eviction_storage import compact_cache_rows_
    import json
    runner = worker.model_runner
    started = time.perf_counter(); profiling = os.environ.get("ARC3_EVICTION_PROFILE", "0") == "1"; phases = {}
    pending = _pending_map(runner, "_arc3_async_evictions")
    if request_id in pending:
        raise EngineEvictionError("An asynchronous compaction is already in flight for this owner")
    limits = _limits(runner)
    checked = validate_context(wire["context"], allow_closed_frontier=wire.get("allow_closed_frontier", False),
                               trusted_ids=trusted_ids(runner._arc3_eviction_states.get(request_id)), **limits)
    if request_id not in runner.req_states.req_id_to_index:
        raise EvictionRunnerError("Eviction target has no resident worker request")
    block_ids = wire["block_ids"]
    if not block_ids or any(not group for group in block_ids):
        raise EvictionRunnerError("Eviction requires explicit retained block ownership")
    table, cutoff, padding = historical_virtual_lookup(checked)
    if cutoff > limits["max_logical_tokens"]:
        raise EvictionRunnerError("QSA virtual mapping exceeds allocated worker capacity")
    # The mrope rows are rebuilt at streaming re-add (_stage_request); they are not needed to copy bytes.
    prepared = PreparedWorkerEviction(request_id, checked, None, None, table, cutoff, padding)
    forward = runner.vllm_config.compilation_config.static_forward_context
    tasks = []
    for group in wire["groups"]:
        for item in group["copies"]:
            cache = forward[item["layer_name"]].kv_cache
            page = item["plan"]
            details = lambda item=item, cache=cache, page=page: {"owner": item["layer_name"], "kind": item["kind"], "cache": describe_cache(cache),
                       "expected_storage_block_size": page.storage_block_size, "old_rows": page.old_rows, "retained_rows": len(page.keep_rows)}   # built only for an error (2026-09-24)
            if not isinstance(cache, torch.Tensor):
                raise EngineEvictionError("Unexpected non-tensor cache owner: " + json.dumps(details(), sort_keys=True))
            axis = 2 if item["kind"] == "main" and cache.dtype == torch.uint8 else 1
            if cache.ndim != 4 or cache.shape[axis] != page.storage_block_size or cache.stride(-1) != 1:
                raise EngineEvictionError("Unexpected physical cache tensor layout: " + json.dumps(details(), sort_keys=True))
            if item["kind"] == "main" and (cache.dtype != torch.uint8 or cache.shape[-1] != 296):
                raise EngineEvictionError("Initial eviction GPU control is NVFP4 main KV only: " + json.dumps(details(), sort_keys=True))
            tasks.append((item, cache, axis))
    phases["prepare_validate_seconds"] = time.perf_counter() - started
    verify = ee._verification_requested(wire) or os.environ.get("ARC3_EVICTION_VERIFY_BYTES", "0") == "1"
    if verify:
        from eviction_validation import stored_rows_digest
    on_gpu = bool(tasks) and tasks[0][1].is_cuda
    stream = None
    if on_gpu:
        stream = getattr(runner, "_arc3_evict_stream", None)
        if stream is None:
            stream = runner._arc3_evict_stream = torch.cuda.Stream(device=tasks[0][1].device)
        # Order after everything already queued on the compute stream (the owner's last step may still be running).
        stream.wait_stream(torch.cuda.current_stream(tasks[0][1].device))
    phase_start = time.perf_counter(); receipts = []; _index_cache = {}
    with (torch.cuda.stream(stream) if on_gpu else contextlib.nullcontext()):
        for item, cache, axis in tasks:
            before = stored_rows_digest(cache, item["plan"], token_axis=axis) if verify else None
            receipt = compact_cache_rows_(cache, item["plan"], token_axis=axis, index_cache=_index_cache)
            if verify:
                after = stored_rows_digest(cache, item["plan"], token_axis=axis, after=True)
                if before != after:
                    raise EngineEvictionError("Actual retained cache bytes changed during compaction")
                receipt["byte_verification"] = {"before": before, "after": after, "equal": True}
            receipts.append({"layer_name": item["layer_name"], **receipt})
        event = None
        if on_gpu:
            event = torch.cuda.Event(); event.record(stream)
    phases["launch_seconds"] = time.perf_counter() - phase_start
    pending[request_id] = dict(event=event, prepared=prepared, receipts=receipts, wire=wire, verify=verify,
                               started=started, phases=phases, profiling=profiling)
    return {"plan_digest": wire["plan_digest"], "request_id": request_id, "launched": True, "layers": len(tasks),
            "device": ("gpu" if on_gpu else "cpu")}


def worker_commit_async(worker, request_id):
    """After the copy event: publish the worker context, compact the counters, return the compaction receipts."""
    import torch
    from eviction_runner import apply_worker_eviction
    runner = worker.model_runner
    pending = _pending_map(runner, "_arc3_async_evictions")
    entry = pending.get(request_id)
    if entry is None:
        raise EngineEvictionError("No asynchronous compaction is in flight for this owner")
    event = entry["event"]
    if event is not None and not event.query():
        return {"pending": True, "request_id": request_id}
    phase_start = time.perf_counter()
    if event is not None:
        torch.cuda.current_stream().wait_event(event)
    apply_worker_eviction(runner, request_id, entry["prepared"])
    counters = getattr(runner.model_state, "_arc3_qsa_frequency", None)
    if counters is not None:
        import qsa_frequency
        keep = qsa_frequency.descriptor_keep_rows_from_wire(entry["wire"])
        if keep is not None:
            qsa_frequency.on_compaction(counters, int(runner.req_states.req_id_to_index[request_id]), keep,
                                        reset_epoch=bool(entry["wire"].get("frequency_epoch_reset", False)))
    phases = entry["phases"]; phases["mapping_publish_seconds"] = time.perf_counter() - phase_start
    phases["total_seconds"] = time.perf_counter() - entry["started"]
    del pending[request_id]
    result = {"plan_digest": entry["wire"]["plan_digest"], "request_id": request_id, "worker_synchronized": True,
              "copies": entry["receipts"], "byte_verification_enabled": entry["verify"],
              "auxiliary_state_evidence": "excluded from compaction; not independently hashed", "execution": "async"}
    if entry["profiling"]:
        result["worker_profile"] = phases
    return result


# ------------------------------------------------------------------------------------------------------ core side
def evict_begin(core, external, payload):
    import engine_patch
    started = time.perf_counter(); profiling = os.environ.get("ARC3_EVICTION_PROFILE", "0") == "1"; phases = {}
    operation = payload.get("operation_id")
    if not isinstance(operation, str) or not operation:
        raise EngineEvictionError("Eviction requires an operation ID")
    pending = _pending_map(core, "_arc3_async_evictions")
    if external in pending:
        raise EngineEvictionError("An asynchronous eviction is already in flight for this session")
    ee._PHASES.clear()
    transaction = ee.prepare_transaction(core, external, payload, drained=False)
    phases["prepare_seconds"] = time.perf_counter() - started
    if profiling:
        phases["prepare_breakdown"] = {k: round(v, 6) for k, v in ee._PHASES.items()}
    phase_start = time.perf_counter()
    request = transaction["request"]; old = transaction["state"]; new = transaction["next_state"]
    compact = transaction["compact"]; logical_plan = transaction["logical_plan"]; groups = transaction["groups"]
    proxy = copy.copy(request); proxy.num_computed_tokens = compact.new_computed
    context = new.context(proxy)
    wire = dict(plan_digest=logical_plan.digest, context=context, allow_closed_frontier=transaction["allow_closed_frontier"],
                verify_eviction=transaction["verify_eviction"],
                block_ids=[[b.block_id for b in group["kept_blocks"]] for group in groups],
                groups=[dict(group_id=g["group_id"], copies=g["copies"]) for g in groups],
                frequency_epoch_reset=payload.get("frequency_epoch_reset", False) is True)
    engine_patch._audit(core.scheduler, "evict_prepare", request, eviction_plan={
        "plan_digest": logical_plan.digest, "logical_ranges": [list(pair) for pair in logical_plan.ranges],
        "resident_before": logical_plan.resident_before, "resident_after": logical_plan.resident_after,
        "frontier": transaction["frontier"], "execution": "async",
        "groups": [{"group_id": g["group_id"], "old_block_ids": [b.block_id for b in g["old_blocks"]],
                    "kept_block_ids": [b.block_id for b in g["kept_blocks"]],
                    "copies": [{"layer_name": item["layer_name"], "kind": item["kind"], "storage_block_size": item["plan"].storage_block_size,
                                "old_rows": item["plan"].old_rows, "retained_rows": len(item["plan"].keep_rows)} for item in g["copies"]]} for g in groups]})
    old.poisoned = True; new.poisoned = True
    phases["wire_and_prepare_audit_seconds"] = time.perf_counter() - phase_start
    phase_start = time.perf_counter()
    try:
        replies = core.collective_rpc(worker_compact_async, timeout=120, args=(request.request_id, wire))
        if (len(replies) != 1 or replies[0].get("plan_digest") != logical_plan.digest or replies[0].get("launched") is not True):
            raise EngineEvictionError("Missing asynchronous worker compaction launch acknowledgement")
    except BaseException:
        old.poisoned = True
        raise
    phases["worker_launch_rpc_seconds"] = time.perf_counter() - phase_start
    phases["begin_total_seconds"] = time.perf_counter() - started
    pending[external] = dict(transaction=transaction, wire=wire, started=started, phases=phases, launch=replies[0])
    return {"begun": True, "plan_digest": logical_plan.digest, "device": replies[0].get("device"),
            "core_profile": (phases if profiling else None)}


def evict_commit(core, external):
    import engine_patch
    profiling = os.environ.get("ARC3_EVICTION_PROFILE", "0") == "1"
    pending = _pending_map(core, "_arc3_async_evictions")
    entry = pending.get(external)
    if entry is None:
        raise EngineEvictionError("No asynchronous eviction is in flight for this session")
    transaction = entry["transaction"]; phases = entry["phases"]
    request = transaction["request"]; old = transaction["state"]; new = transaction["next_state"]
    compact = transaction["compact"]; logical_plan = transaction["logical_plan"]; groups = transaction["groups"]
    phase_start = time.perf_counter()
    try:
        replies = core.collective_rpc(worker_commit_async, timeout=120, args=(request.request_id,))
    except BaseException:
        old.poisoned = True; pending.pop(external, None)
        raise
    reply = replies[0] if len(replies) == 1 and isinstance(replies[0], dict) else None
    if reply is not None and reply.get("pending"):
        phases["commit_polls"] = phases.get("commit_polls", 0) + 1
        return {"pending": True}
    if reply is None or reply.get("plan_digest") != logical_plan.digest or reply.get("worker_synchronized") is not True:
        old.poisoned = True; pending.pop(external, None)
        raise EngineEvictionError("Missing synchronized asynchronous worker compaction acknowledgement")
    phases["worker_commit_rpc_seconds"] = time.perf_counter() - phase_start
    phase_start = time.perf_counter()
    # CPU publication and release happen only after all owning cache layers and position state have completed.
    request.prompt_token_ids[:] = transaction["new_prompt"]
    request.num_prompt_tokens = len(request.prompt_token_ids)
    request._all_token_ids[:] = [new.logical_ids[p] for p in compact.kept_positions]
    request.num_computed_tokens = compact.new_computed
    request.mm_features = transaction["new_features"]
    request.block_hashes.clear()
    request.update_block_hashes()
    request._arc3_eviction_state = new
    managers = core.scheduler.kv_cache_manager.coordinator.single_type_managers
    freed = []
    for group in groups:
        manager = managers[group["group_id"]]
        manager.req_to_blocks[request.request_id][:] = group["kept_blocks"]
        manager.num_cached_block[request.request_id] = 0
        manager._arc3_eviction_states[request.request_id] = new
        freed.extend(group["freed_blocks"])
    core.scheduler.kv_cache_manager.block_pool.free_blocks(freed)
    allocator_receipt = None
    if transaction["verify_eviction"] or os.environ.get("ARC3_EVICTION_VERIFY_ALLOCATOR", "0") == "1":
        from eviction_allocator import verify_allocator_reacquisition
        allocator_receipt = verify_allocator_reacquisition(core.scheduler.kv_cache_manager.block_pool, freed)
    ee.publish_context(request)
    new.poisoned = False
    phases["publish_release_optional_allocator_proof_seconds"] = time.perf_counter() - phase_start
    ack = dict(plan_digest=logical_plan.digest, resident_tokens=len(request._all_token_ids),
               evicted_tokens=logical_plan.removed_tokens, logical_tokens=len(new.logical_ids),
               eviction_revision=new.eviction_revision, frontier=transaction["frontier"],
               freed_blocks_by_group=[[b.block_id for b in g["freed_blocks"]] for g in groups],
               worker_receipts=replies, allocator_reacquisition=allocator_receipt, execution="async")
    if profiling:
        phases["total_before_final_audit_seconds"] = time.perf_counter() - entry["started"]
        ack["core_profile"] = phases
    pending.pop(external, None)
    engine_patch._audit(core.scheduler, "evict", request, eviction=ack)
    return ack


def install(EngineCore=None):
    if EngineCore is None:
        from vllm.v1.engine.core import EngineCore
    if not callable(getattr(EngineCore, "arc3_evict_resident", None)):
        raise EngineEvictionError("Install normal engine eviction before the asynchronous transaction")
    existing = EngineCore.__dict__.get(BEGIN)
    if existing is not None:
        if getattr(existing, "_arc3_eviction_async_version", None) != VERSION:
            raise EngineEvictionError("Refusing to replace an unknown core utility")
        return {"installed": True, "version": VERSION}
    def begin(self, external_request_id, payload):
        return evict_begin(self, external_request_id, payload)
    def commit(self, external_request_id):
        return evict_commit(self, external_request_id)
    begin.__name__ = BEGIN; commit.__name__ = COMMIT
    begin._arc3_eviction_async_version = VERSION; commit._arc3_eviction_async_version = VERSION
    setattr(EngineCore, BEGIN, begin); setattr(EngineCore, COMMIT, commit)
    return {"installed": True, "version": VERSION, "methods": [BEGIN, COMMIT]}
