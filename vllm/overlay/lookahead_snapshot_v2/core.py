"""A stricter replacement for the diagnostic RPC only; no scheduler changes."""
import copy
from .receipt_schema import require, same
from .guards import validate_config, validate_wire
from .layout import validate_specs, validate_block_ids, spec_record
from .provenance import verify_sources


def prepare_wire(core, external, engine_patch, engine_eviction, one_lookahead):
    scheduler = core.scheduler
    validate_config(scheduler.vllm_config)
    require(one_lookahead.enabled(), 'Isolated lookahead diagnostic flag required')
    require(not core.batch_queue and not core.model_executor.futures_queue, 'Outstanding native batch/future')
    require(not one_lookahead.pending(scheduler), 'Pending stop transaction')
    request = engine_eviction._request_for_external(scheduler, external)
    require(scheduler.requests.get(request.request_id) is request, 'Target no longer resident')
    for resident in scheduler.requests.values():
        require(resident.status.name == 'WAITING_FOR_STREAMING_REQ', 'Every resident owner must be paused')
        require(engine_patch.is_persistent(resident), 'Nonpersistent resident owner')
        engine_patch.assert_settled(resident)
        require(not resident.streaming_queue and not getattr(resident, '_arc3_la_pending', False),
                'Unacknowledged stop or append')
        require(resident.num_preemptions == 0 and not resident.mm_features
                and resident.prompt_embeds is None and resident.lora_request is None,
                'Preemption/MM/embedding/LoRA outside text scope')
        state = getattr(resident, '_arc3_eviction_state', None)
        require(state is None or (state.deleted_tokens == 0 and state.eviction_revision == 0
                and state.poisoned is False), 'Evicted/poisoned owner outside scope')
        if one_lookahead.marked(resident):
            require(getattr(resident, '_arc3_la_fenced', False) is True, 'Lookahead pause not acknowledged')
    coord = scheduler.kv_cache_manager.coordinator
    groups = engine_eviction.worker_owner_groups(coord.kv_cache_config)
    specs = validate_specs(groups)
    blocks = [list(g) for g in scheduler.kv_cache_manager.get_blocks(request.request_id).get_block_ids()]
    validate_block_ids(blocks)
    require(len(coord.single_type_managers) == 6, 'Unknown cache manager group count')
    refs = []
    for i, manager in enumerate(coord.single_type_managers):
        owned = manager.req_to_blocks[request.request_id]
        require([b.block_id for b in owned] == blocks[i], 'Core block lists disagree')
        require(all(type(b.block_id) is int and b.block_id > 0 and b.ref_cnt == 1 for b in owned),
                'Null or shared cache page')
        refs.append([dict(block_id=b.block_id, ref_cnt=b.ref_cnt) for b in owned])
    wire = dict(request_id=request.request_id, computed_tokens=request.num_computed_tokens,
        accepted_token_ids=list(request._all_token_ids), block_ids=blocks,
        logical_positions=list(range(request.num_computed_tokens)),
        original_prompt_tokens=request.num_prompt_tokens,
        core_specs={name: spec_record(spec) for name, spec in sorted(specs.items())},
        core_proof=dict(request_id=request.request_id, block_ids=blocks, exclusive_pages=True,
            all_owners_paused=True, queues_empty=True, no_pending_transactions=True, refs=refs))
    validate_wire(wire)
    return request, wire


def core_diagnostic(core, external):
    import engine_patch
    import engine_eviction
    import one_lookahead
    from .worker import worker_snapshot
    _, source_sha = verify_sources()
    request, wire = prepare_wire(core, external, engine_patch, engine_eviction, one_lookahead)
    wire['source_manifest_sha256'] = source_sha
    replies = core.collective_rpc(worker_snapshot, timeout=120, args=(wire,))
    require(type(replies) is list and len(replies) == 1, 'Exactly one worker reply required')
    after_request, after_wire = prepare_wire(core, external, engine_patch, engine_eviction, one_lookahead)
    require(after_request is request and same(after_wire, {k:v for k,v in wire.items()
            if k != 'source_manifest_sha256'}), 'Core frontier/ownership changed during snapshot')
    require(verify_sources()[1] == source_sha, 'Source changed during snapshot')
    return dict(version=one_lookahead.VERSION, computed_tokens=wire['computed_tokens'],
        accepted_tokens=len(wire['accepted_token_ids']),
        delta=len(wire['accepted_token_ids'])-wire['computed_tokens'],
        settlement=copy.deepcopy(getattr(request, '_arc3_la_last_settlement', None)), worker=replies[0])


def install_diagnostic_only():
    """Future isolated stage may call after frozen candidate install; no auto hook."""
    import one_lookahead
    from vllm.v1.engine.core import EngineCore
    verify_sources()
    require(EngineCore.arc3_one_lookahead_diagnostic is one_lookahead.core_diagnostic,
            'Unexpected existing diagnostic method')
    EngineCore.arc3_one_lookahead_diagnostic = core_diagnostic
    return {'installed': True, 'scope': 'diagnostic RPC only; scheduler unchanged'}
