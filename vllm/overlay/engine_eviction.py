"""Opt-in core transaction for original-position, resident-KV compaction.

The public service retains the full logical ledger. Core scheduling uses dense
resident tokens. Worker storage and position hooks must be installed together;
this module refuses a mutation if either side is unavailable. No import-time
runtime mutation occurs.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, replace
import functools
import inspect
import hashlib
import json
from math import ceil
import os
import re
import numpy as np
import time
from pathlib import Path

from residency import ResidentLedger
from eviction_storage import token_compaction, page_compaction, compact_cache_rows_
from eviction_qsa import qsa_layout, descriptor_keep_rows

_PHASES={}   # 2026-09-23: prepare_transaction breakdown for the profile flag (reset per evict_resident)
def _timed(name,fn):
    @functools.wraps(fn)
    def wrapped(*a,**k):
        t=time.perf_counter()
        try:return fn(*a,**k)
        finally:_PHASES[name]=_PHASES.get(name,0.0)+time.perf_counter()-t
    return wrapped
token_compaction=_timed("token_compaction",token_compaction)
qsa_layout=_timed("qsa_layout",qsa_layout)
descriptor_keep_rows=_timed("descriptor_keep_rows",descriptor_keep_rows)
page_compaction=_timed("page_compaction",page_compaction)


CONTEXT_KEY = "arc3_eviction_context"
MARKER = "arc3_managed_eviction"
VERSION = "arc3-managed-eviction-v1"
SOURCE_HASHES = {
    "v1/engine/core.py": "f4b1e07b6d91fac1549bc74a5c804e5a88793d663b5dc045c9e455db6550f5ad",
    "v1/core/single_type_kv_cache_manager.py": "9ec6cbc0c78f30986f294439f02f57a9586ffbf2939d8f5c4fb968a642625ca2",
    "v1/core/kv_cache_coordinator.py": "9cd616a0f6bbd27776aa6bb44c7420c95574f9ffb5e254a8f355937642b8451b",
    "v1/core/block_pool.py": "ddee56dccb2208411b3a035918e917ce8f56a9858471e9ca12b420d5d79bc69c",
    "v1/core/kv_cache_utils.py": "f6643bad29b7b3cb044482b6736a042afa91a4f43a11f034317a8e8029d3a8a4",
    "v1/kv_cache_interface.py": "7452e823367960daf73520f80824aee0d5ed8ad2731d0404705695814d3bcff7",
    "v1/executor/multiproc_executor.py": "b911d837e167592acbfc36689808fc18c697d9624eaea527e56f6ea18b7a719d",
}


class EngineEvictionError(RuntimeError):
    pass


def _page_plan_reuse_enabled():
    value=os.environ.get('ARC3_EFFICIENCY_PAGE_PLAN_REUSE','0')
    if value not in ('0','1'):
        raise EngineEvictionError('ARC3_EFFICIENCY_PAGE_PLAN_REUSE must be 0 or 1')
    return value=='1'


def digest(ids):
    return hashlib.sha256(json.dumps(ids,separators=(",", ":")).encode()).hexdigest()


def _ol_flag(name,default):
    value=os.environ.get(name,default)
    if value not in ("0","1"):
        raise EngineEvictionError(f"{name} must be 0 or 1")
    return value=="1"


class _PrefixDigest:
    """digest() of a growing token list, hashed incrementally (2026-09-25 O(L) fast path). Exact: the hashed prefix is a
    private copy compared with the current list's prefix on every call (C speed); a shorter or rewritten list rehashes."""
    __slots__=("ids","hasher")

    def __init__(self):
        self.ids=[]; self.hasher=hashlib.sha256(b"[")

    def hexdigest(self,current,dumps):
        n=len(self.ids)
        if n and (len(current)<n or current[:n]!=self.ids):
            self.ids=[]; self.hasher=hashlib.sha256(b"["); n=0
        if len(current)>n:
            tail=current[n:]
            self.hasher.update((("," if n else "")+dumps(tail)[1:-1]).encode())
            self.ids.extend(tail)
        final=self.hasher.copy(); final.update(b"]")
        return final.hexdigest()


def logical_digest(state):
    """digest(state.logical_ids); incremental per LogicalState (successors share the cache; it verifies itself)."""
    ids=state.logical_ids
    if not _ol_flag("ARC3_OL_FASTPATH","1") or type(ids) is not list:
        return digest(ids)
    cache=state.__dict__.get("_arc3_logical_digest")
    if cache is None:
        cache=state.__dict__["_arc3_logical_digest"]=_PrefixDigest()
    value=cache.hexdigest(ids,lambda x: json.dumps(x,separators=(",", ":")))
    if _ol_flag("ARC3_OL_FASTPATH_VERIFY","0") and value!=digest(ids):
        raise EngineEvictionError("Incremental logical digest differs from the full digest")
    return value


def _gather(ids,positions):
    import operator
    if not positions:
        return []
    if len(positions)==1:
        return [ids[positions[0]]]
    return list(operator.itemgetter(*positions)(ids))


def marked(request):
    return (getattr(request,"resumable",False)
            and (getattr(request.sampling_params,"extra_args",None) or {}).get(MARKER) is True)


def image_features(features, *, offset=0):
    result=[]
    for feature in features:
        if feature.modality != "image":
            raise EngineEvictionError("Managed eviction currently supports images only")
        grid=feature.data["image_grid_thw"].data.tolist()
        if len(grid)==1 and isinstance(grid[0],list):
            grid=grid[0]
        if len(grid)!=3 or any(type(v) is not int or v<=0 for v in grid):
            raise EngineEvictionError("Invalid image position grid")
        result.append(dict(offset=feature.mm_position.offset+offset,
                           length=feature.mm_position.length,grid_thw=grid))
    return result


@dataclass
class LogicalState:
    logical_ids: list
    ledger: ResidentLedger
    prompt_logical_length: int
    prompt_positions: tuple
    logical_image_features: list
    eviction_revision: int=0
    poisoned: bool=False

    @classmethod
    def create(cls,request):
        ids=list(request.prompt_token_ids)
        return cls(ids,ResidentLedger(len(ids)),len(ids),tuple(range(len(ids))),
                   image_features(request.mm_features))

    def paused(self,request):
        output=list(request._output_token_ids)
        n=self.prompt_logical_length; pp=self.prompt_positions; resident=request._all_token_ids
        if (_ol_flag("ARC3_OL_FASTPATH","1") and type(self.logical_ids) is list and type(resident) is list
                and len(self.logical_ids)>=n and (not pp or pp[-1]<n)):
            # 2026-09-25 O(L) fast path: same check (C-level gather) and the same resulting list contents, updated in
            # place (truncate to the prompt, extend by the output) instead of rebuilding the whole logical list
            k=len(pp)
            if len(resident)!=k+len(output) or resident[k:]!=output or _gather(self.logical_ids,pp)!=resident[:k]:
                raise EngineEvictionError("Logical and resident output ledgers diverged")
            if _ol_flag("ARC3_OL_FASTPATH_VERIFY","0"):
                expected=self.logical_ids[:n]+output
            del self.logical_ids[n:]
            self.logical_ids.extend(output)
            if _ol_flag("ARC3_OL_FASTPATH_VERIFY","0") and self.logical_ids!=expected:
                raise EngineEvictionError("In-place logical ledger differs from the rebuilt one")
            self.ledger.logical_tokens=len(self.logical_ids)
            return
        all_ids=self.logical_ids[:self.prompt_logical_length]+output
        positions=self.prompt_positions+tuple(range(self.prompt_logical_length,len(all_ids)))
        if [all_ids[i] for i in positions] != request._all_token_ids:
            raise EngineEvictionError("Logical and resident output ledgers diverged")
        self.logical_ids=all_ids
        self.ledger.logical_tokens=len(all_ids)

    def context(self,request):
        positions=self.ledger.resident_positions()
        computed=request.num_computed_tokens+self.deleted_tokens
        return dict(logical_token_ids=list(self.logical_ids), resident_positions=positions,
                    logical_computed=computed, logical_image_features=copy.deepcopy(self.logical_image_features),
                    eviction_revision=self.eviction_revision,deleted_tokens=self.deleted_tokens,
                    deleted_groups=self.deleted_groups)

    @property
    def deleted_tokens(self):
        return self.ledger.logical_tokens-self.ledger.resident_tokens

    @property
    def group_padding(self):
        return self.deleted_tokens-4*self.deleted_groups

    @property
    def deleted_groups(self):
        # Intervals are disjoint and adjacency is coalesced. Only groups whose
        # entire original four-member interval is removed lose a descriptor.
        return sum(max(0,end//4-ceil(start/4)) for start,end in self.ledger.evicted)


def publish_context(request):
    state=request._arc3_eviction_state
    params=request.sampling_params
    params.extra_args={**(params.extra_args or {}),CONTEXT_KEY:state.context(request)}


def _shallow_next_state(state):
    """Transaction successor without copy.deepcopy (2026-09-24 stage D): own ledger object (commit rebinds its ranges),
    shared logical id list and image layout (only the installed state ever mutates them; the predecessor is discarded
    at commit or poisoned on failure); prompt_positions and eviction_revision are rebound by the caller."""
    nxt=copy.copy(state)
    nxt.ledger=copy.copy(state.ledger)
    return nxt


def _request_for_external(scheduler,external):
    if type(external) is not str:
        raise EngineEvictionError("Invalid external request ID")
    matches=[r for rid,r in scheduler.requests.items()
             if rid==external or re.fullmatch(re.escape(external)+r"-[0-9a-f]{8}",rid)]
    if len(matches)!=1:
        raise EngineEvictionError("Request identity is absent or ambiguous")
    return matches[0]


def _native_compacted_features(features,plan):
    lookup={old:new for new,old in enumerate(plan.keep_indices)}
    result=[]
    for feature in features:
        position=feature.mm_position
        rows=range(position.offset,position.offset+position.length)
        present=[row in lookup for row in rows]
        if any(present) and not all(present):
            raise EngineEvictionError("Eviction cuts through a multimodal feature")
        if all(present) and present:
            result.append(replace(feature,mm_position=replace(position,offset=lookup[position.offset])))
    return result


def worker_owner_groups(kv_cache_config):
    """Recover physical owner specs lost by stock scheduler normalization."""
    groups=getattr(kv_cache_config,"_arc3_worker_kv_cache_groups",None)
    scheduler_groups=kv_cache_config.kv_cache_groups
    if groups is None or len(groups)!=len(scheduler_groups):
        raise EngineEvictionError("Physical per-owner worker cache specs were not preserved")
    for worker,scheduler in zip(groups,scheduler_groups):
        if (list(worker.layer_names)!=list(scheduler.layer_names)
                or worker.kv_cache_spec.block_size!=scheduler.kv_cache_spec.block_size):
            raise EngineEvictionError("Worker/scheduler cache group geometry or ownership differs")
        owners=getattr(worker.kv_cache_spec,"kv_cache_specs",None)
        if owners is not None and (len(owners)!=len(worker.layer_names)
                                  or set(owners)!=set(worker.layer_names)):
            raise EngineEvictionError("Preserved cache owner map does not match group layer names")
    return groups


def preserve_worker_owner_specs(original):
    """Keep stock scheduler representatives and retain separate physical specs.

The pinned generator collapses UniformTypeKVCacheSpecs to its first member.
That is correct for scheduling because owners share page IDs; it is insufficient
for copying rows when full-attention and compressed owners share that group.
"""
    @functools.wraps(original)
    def generate(configs):
        if len(configs)!=1:
            raise EngineEvictionError("Eviction owner-spec preservation requires one worker")
        physical_groups=copy.deepcopy(configs[0].kv_cache_groups)
        result=original(configs)
        result._arc3_worker_kv_cache_groups=physical_groups
        worker_owner_groups(result)
        return result
    return generate


def _transaction_page_plan(cache, block_ids, keep_rows, *, old_rows, storage_block_size):
    """Reuse fully validated immutable row plans only within this transaction.

    Different owners still receive separate receipts and every owner/layout/
    exclusive-page check remains in prepare_transaction. Mutable row inputs
    and noncanonical scalar types always take the original validation path.
    """
    blocks=tuple(block_ids)
    if (type(keep_rows) is not tuple or type(old_rows) is not int
            or type(storage_block_size) is not int or any(type(b) is not int for b in blocks)):
        return page_compaction(blocks,keep_rows,old_rows=old_rows,storage_block_size=storage_block_size)
    key=(blocks,old_rows,storage_block_size)
    previous=cache.get(key)
    if previous is not None and previous[0] is keep_rows:
        return previous[1]
    plan=page_compaction(blocks,keep_rows,old_rows=old_rows,storage_block_size=storage_block_size)
    cache[key]=(keep_rows,plan)
    return plan



def _verification_requested(payload):
    value=payload.get("verify_eviction",False)
    if type(value) is not bool:
        raise EngineEvictionError("Internal eviction verification flag must be a boolean")
    return value


def prepare_transaction(core,external,payload,drained=True):
    _t_entry=time.perf_counter()
    """All CPU validation and allocation planning before any worker mutation."""
    import engine_patch
    from vllm.v1.kv_cache_interface import (FullAttentionSpec, MLAAttentionSpec,
                                          MambaSpec,CircularBufferSpec,UniformTypeKVCacheSpecs)
    verify_eviction=_verification_requested(payload)
    scheduler=core.scheduler
    if drained:   # fused transaction: whole-engine quiescence; the asynchronous path (stage C) checks the owner only
        if not core.is_scheduler_paused() or getattr(core,"batch_queue",None):
            raise EngineEvictionError("Eviction requires the keep-paused, drained engine")
        for pending in scheduler.requests.values():
            if pending.num_in_flight_tokens:
                raise EngineEvictionError("Outstanding model work remains")
    request=_request_for_external(scheduler,external)
    if not drained and request.num_in_flight_tokens:
        raise EngineEvictionError("Owner has model work in flight")
    state=getattr(request,"_arc3_eviction_state",None)
    if not marked(request) or state is None or state.poisoned:
        raise EngineEvictionError("Request is not an enabled healthy eviction session")
    engine_patch.assert_settled(request)
    if request.status.name!="WAITING_FOR_STREAMING_REQ" or request.num_preemptions:
        raise EngineEvictionError("Eviction requires a non-preempted paused request")
    if request.streaming_queue:
        raise EngineEvictionError("Input is pending at the eviction boundary")
    import one_lookahead
    allow_closed_frontier=one_lookahead.validate_eviction(core,request,drained=drained)
    state.paused(request)
    delta=len(request._all_token_ids)-request.num_computed_tokens
    original_computed=request.num_computed_tokens+state.deleted_tokens
    if delta not in ((0,1) if one_lookahead.marked(request) else (1,)) or original_computed!=len(state.logical_ids)-delta:
        raise EngineEvictionError("Resident/logical computed frontier disagrees before eviction")
    if (payload.get("expected_logical_tokens")!=len(state.logical_ids)
            or payload.get("logical_prefix_sha256")!=logical_digest(state)
            or payload.get("expected_eviction_revision")!=state.eviction_revision
            or payload.get("prior_evicted")!=[list(r) for r in state.ledger.evicted]):
        raise EngineEvictionError("Stale logical identity or eviction revision")
    _PHASES["checks"]=time.perf_counter()-_t_entry;_t=time.perf_counter()
    logical_plan=state.ledger.plan(payload.get("logical_ranges",()),
                                  protected=((max(0,len(state.logical_ids)-4),len(state.logical_ids)),))
    _PHASES["ledger_plan"]=time.perf_counter()-_t
    if (not logical_plan.removed_tokens or payload.get("plan_digest")!=logical_plan.digest
            or payload.get("resident_before")!=logical_plan.resident_before
            or payload.get("resident_after")!=logical_plan.resident_after):
        raise EngineEvictionError("Invalid or mismatched resident eviction plan")
    kept=[i for begin,end in logical_plan.survivor_intervals for i in range(begin,end)]
    compact=token_compaction(state.ledger.resident_positions(),kept,
                             old_computed=request.num_computed_tokens,allow_fully_computed=allow_closed_frontier)
    _ki=np.fromiter(compact.keep_indices,dtype=np.int64,count=len(compact.keep_indices))
    if int(np.count_nonzero(_ki>=request.num_prompt_tokens))!=len(compact.old_positions)-request.num_prompt_tokens:
        raise EngineEvictionError("This phase evicts prior input spans, not the current output")
    _t=time.perf_counter();new_features=_native_compacted_features(request.mm_features,compact);_PHASES["mm_features"]=time.perf_counter()-_t
    from eviction_qsa import descriptor_rows_fast
    _t=time.perf_counter();descriptor_old_rows,descriptor_rows=descriptor_rows_fast(compact.old_positions,compact.kept_positions,logical_computed=original_computed);_PHASES["descriptor_rows"]=time.perf_counter()-_t
    coordinator=scheduler.kv_cache_manager.coordinator
    groups=[];_t_pages=time.perf_counter()
    page_plans={} if _page_plan_reuse_enabled() else None
    for group_id,(group,manager) in enumerate(zip(worker_owner_groups(coordinator.kv_cache_config),
                                                 coordinator.single_type_managers)):
        blocks=tuple(manager.req_to_blocks.get(request.request_id,()))
        import prefix_share_v1   # 2026-09-27: the registered shared prefix block(s) may lead the attention group
        shared=prefix_share_v1.shared_blocks(group_id,blocks)
        if any((b.ref_cnt!=1 and i>=shared) or b.block_id==0 for i,b in enumerate(blocks)):
            raise EngineEvictionError("Cache pages are shared, null, or not exclusively owned")
        spec=group.kv_cache_spec
        owners=spec.kv_cache_specs if isinstance(spec,UniformTypeKVCacheSpecs) else {name:spec for name in group.layer_names}
        copies=[]
        for name,owner in owners.items():
            if isinstance(owner,(MambaSpec,CircularBufferSpec)):
                continue
            if isinstance(owner,MLAAttentionSpec):
                if owner.compress_ratio!=4 or not name.endswith("compressed_key_cache"):
                    raise EngineEvictionError("Unsupported compressed cache owner")
                rows=descriptor_rows; old_rows=descriptor_old_rows; kind="compressed"
            elif isinstance(owner,FullAttentionSpec):
                rows=compact.computed_keep_indices; old_rows=compact.old_computed; kind="main"
            else:
                raise EngineEvictionError("Unreviewed cache owner in eviction group")
            if shared and not prefix_share_v1.rows_intact(rows,shared*owner.storage_block_size):
                raise EngineEvictionError("Eviction would move rows of the shared prefix block")
            if page_plans is None:
                page=page_compaction([b.block_id for b in blocks],rows,old_rows=old_rows,
                                     storage_block_size=owner.storage_block_size)
            else:
                page=_transaction_page_plan(page_plans,[b.block_id for b in blocks],rows,old_rows=old_rows,
                                            storage_block_size=owner.storage_block_size)
            copies.append(dict(layer_name=name,kind=kind,plan=page))
        if copies:
            needed=max(len(copy["plan"].kept_block_ids) for copy in copies)
            # Each layer in a uniform group uses the same page IDs. Do not
            # release a page still required by its compressed or main owner.
            for item in copies:
                p=item["plan"]
                item["plan"]=replace(p,kept_block_ids=p.old_block_ids[:needed],
                                     released_block_ids=p.old_block_ids[needed:])
        else:
            needed=len(blocks)
        groups.append(dict(group_id=group_id,old_blocks=blocks,kept_blocks=blocks[:needed],
                           freed_blocks=blocks[needed:],copies=copies))
    _PHASES["page_plans"]=time.perf_counter()-_t_pages
    new_prompt=np.asarray(request.prompt_token_ids,dtype=np.int64)[_ki[_ki<request.num_prompt_tokens]].tolist()
    _t=time.perf_counter();next_state=_shallow_next_state(state);_PHASES["next_state"]=time.perf_counter()-_t
    next_state.ledger.commit(logical_plan,acknowledged_digest=logical_plan.digest)
    next_state.prompt_positions=tuple(p for p in compact.kept_positions if p<state.prompt_logical_length)
    next_state.eviction_revision+=1
    if (len(compact.kept_positions)-compact.new_computed!=delta
            or compact.new_computed+next_state.deleted_tokens!=original_computed):
        raise EngineEvictionError("Compaction changed the accepted/computed delta or original frontier")
    for manager in coordinator.single_type_managers:
        if getattr(manager,"_arc3_has_compressed",False):
            if logical_plan.resident_after+next_state.group_padding>manager._arc3_table_capacity:
                raise EngineEvictionError("Compacted QSA descriptor demand exceeds page-table capacity")
    if verify_eviction or os.environ.get("ARC3_EVICTION_VERIFY_ALLOCATOR","0")=="1":
        from eviction_allocator import validate_allocator_diagnostic
        validate_allocator_diagnostic(scheduler.kv_cache_manager.block_pool)
    return dict(request=request,state=state,next_state=next_state,logical_plan=logical_plan,
                compact=compact,new_features=new_features,new_prompt=new_prompt,groups=groups,
                frontier=dict(delta=delta,logical_computed=original_computed),
                allow_closed_frontier=allow_closed_frontier,verify_eviction=verify_eviction)


def worker_compact(worker,request_id,wire):
    """Serialized worker RPC: validate all layouts, copy bytes, publish mapping."""
    import torch
    from eviction_runner import prepare_worker_eviction, apply_worker_eviction
    from eviction_diagnostics import describe_cache
    runner=worker.model_runner
    started=time.perf_counter()
    profiling=os.environ.get("ARC3_EVICTION_PROFILE","0")=="1"
    phases={}
    torch.cuda.synchronize()
    phases["initial_fence_seconds"]=time.perf_counter()-started
    phase_start=time.perf_counter()
    prepared=prepare_worker_eviction(runner,request_id,wire["context"],wire["block_ids"],
        allow_closed_frontier=wire.get("allow_closed_frontier",False))
    forward=runner.vllm_config.compilation_config.static_forward_context
    tasks=[]
    for group in wire["groups"]:
        for item in group["copies"]:
            cache=forward[item["layer_name"]].kv_cache
            page=item["plan"]
            details={"owner":item["layer_name"],"kind":item["kind"],"cache":describe_cache(cache),
                     "expected_storage_block_size":page.storage_block_size,
                     "old_rows":page.old_rows,"retained_rows":len(page.keep_rows)}
            if not isinstance(cache,torch.Tensor):
                raise EngineEvictionError("Unexpected non-tensor cache owner: "+json.dumps(details,sort_keys=True))
            axis=2 if item["kind"]=="main" and cache.dtype==torch.uint8 else 1
            if (cache.ndim!=4
                    or cache.shape[axis]!=page.storage_block_size
                    or cache.stride(-1)!=1):
                raise EngineEvictionError("Unexpected physical cache tensor layout: "+json.dumps(details,sort_keys=True))
            if item["kind"]=="main" and (cache.dtype!=torch.uint8 or cache.shape[-1]!=296):
                raise EngineEvictionError("Initial eviction GPU control is NVFP4 main KV only: "+json.dumps(details,sort_keys=True))
            tasks.append((item,cache,axis))
    # A partial mutation cannot be retried: the caller poisons/closes this
    # session if this RPC raises or its outcome is uncertain.
    phases["prepare_validate_seconds"]=time.perf_counter()-phase_start
    receipts=[]
    verify=_verification_requested(wire) or os.environ.get("ARC3_EVICTION_VERIFY_BYTES","0")=="1"
    if verify:
        from eviction_validation import stored_rows_digest
    phase_start=time.perf_counter();_index_cache={}
    for item,cache,axis in tasks:
        before=stored_rows_digest(cache,item["plan"],token_axis=axis) if verify else None
        receipt=compact_cache_rows_(cache,item["plan"],token_axis=axis,index_cache=_index_cache)
        if verify:
            after=stored_rows_digest(cache,item["plan"],token_axis=axis,after=True)
            if before!=after:
                raise EngineEvictionError("Actual retained cache bytes changed during compaction")
            receipt["byte_verification"]={"before":before,"after":after,"equal":True}
        receipts.append({"layer_name":item["layer_name"],**receipt})
    torch.cuda.synchronize()
    phases["copy_and_optional_proofs_seconds"]=time.perf_counter()-phase_start
    phase_start=time.perf_counter()
    # apply_worker_eviction publishes a validated JSON-only CPU context. All
    # preceding cache writes have completed at the fence above; the request's
    # GPU mapping is staged later, on streaming re-add before its next forward.
    apply_worker_eviction(runner,request_id,prepared)
    counters=getattr(runner.model_state,"_arc3_qsa_frequency",None)
    if counters is not None:
        # Statistics follow their descriptors; a history eviction also starts a new epoch.
        import qsa_frequency
        keep=qsa_frequency.descriptor_keep_rows_from_wire(wire)
        if keep is not None:
            qsa_frequency.on_compaction(counters,int(runner.req_states.req_id_to_index[request_id]),keep,
                                        reset_epoch=bool(wire.get("frequency_epoch_reset",False)))
    phases["mapping_publish_seconds"]=time.perf_counter()-phase_start
    phases["total_seconds"]=time.perf_counter()-started
    result={"plan_digest":wire["plan_digest"],"request_id":request_id,
            "worker_synchronized":True,"copies":receipts,
            "byte_verification_enabled":verify,
            "auxiliary_state_evidence":"excluded from compaction; not independently hashed"}
    if profiling:
        result["worker_profile"]=phases
    return result


def evict_resident(core,external,payload):
    import engine_patch
    started=time.perf_counter()
    profiling=os.environ.get("ARC3_EVICTION_PROFILE","0")=="1"
    phases={};_PHASES.clear()
    operation=payload.get("operation_id")
    if not isinstance(operation,str) or not operation:
        raise EngineEvictionError("Eviction requires an operation ID")
    transaction=prepare_transaction(core,external,payload)
    phases["prepare_seconds"]=time.perf_counter()-started
    if profiling:phases["prepare_breakdown"]={k:round(v,6) for k,v in _PHASES.items()}
    phase_start=time.perf_counter()
    request=transaction["request"]; old=transaction["state"]; new=transaction["next_state"]
    compact=transaction["compact"]; logical_plan=transaction["logical_plan"]
    # Construct the worker's post-eviction view without modifying core state.
    proxy=copy.copy(request)
    proxy.num_computed_tokens=compact.new_computed
    context=new.context(proxy)
    groups=transaction["groups"]
    wire=dict(plan_digest=logical_plan.digest,context=context,
              allow_closed_frontier=transaction["allow_closed_frontier"],
              verify_eviction=transaction["verify_eviction"],
              block_ids=[[b.block_id for b in group["kept_blocks"]] for group in groups],
              groups=[dict(group_id=g["group_id"],copies=g["copies"]) for g in groups],
              frequency_epoch_reset=payload.get("frequency_epoch_reset",False) is True)
    engine_patch._audit(core.scheduler,"evict_prepare",request,eviction_plan={
        "plan_digest":logical_plan.digest,"logical_ranges":[list(pair) for pair in logical_plan.ranges],
        "resident_before":logical_plan.resident_before,"resident_after":logical_plan.resident_after,
        "frontier":transaction["frontier"],
        "groups":[{"group_id":g["group_id"],"old_block_ids":[b.block_id for b in g["old_blocks"]],
                   "kept_block_ids":[b.block_id for b in g["kept_blocks"]],
                   "copies":[{"layer_name":item["layer_name"],"kind":item["kind"],
                              "storage_block_size":item["plan"].storage_block_size,
                              "old_rows":item["plan"].old_rows,"retained_rows":len(item["plan"].keep_rows)}
                             for item in g["copies"]]} for g in groups]})
    old.poisoned=True
    new.poisoned=True
    phases["wire_and_prepare_audit_seconds"]=time.perf_counter()-phase_start
    phase_start=time.perf_counter()
    try:
        replies=core.collective_rpc(worker_compact,timeout=120,args=(request.request_id,wire))
        if (len(replies)!=1 or replies[0].get("plan_digest")!=logical_plan.digest
                or replies[0].get("worker_synchronized") is not True):
            raise EngineEvictionError("Missing synchronized worker compaction acknowledgement")
    except BaseException:
        old.poisoned=True
        raise
    phases["worker_rpc_seconds"]=time.perf_counter()-phase_start
    phase_start=time.perf_counter()
    # CPU publication and release happen only after all owning cache layers and
    # position state have completed their worker transaction.
    request.prompt_token_ids[:]=transaction["new_prompt"]
    request.num_prompt_tokens=len(request.prompt_token_ids)
    request._all_token_ids[:]=[new.logical_ids[p] for p in compact.kept_positions]
    request.num_computed_tokens=compact.new_computed
    request.mm_features=transaction["new_features"]
    request.block_hashes.clear()
    request.update_block_hashes()
    request._arc3_eviction_state=new
    managers=core.scheduler.kv_cache_manager.coordinator.single_type_managers
    freed=[]
    for group in groups:
        manager=managers[group["group_id"]]
        manager.req_to_blocks[request.request_id][:]=group["kept_blocks"]
        manager.num_cached_block[request.request_id]=0
        manager._arc3_eviction_states[request.request_id]=new
        freed.extend(group["freed_blocks"])
    core.scheduler.kv_cache_manager.block_pool.free_blocks(freed)
    allocator_receipt=None
    if transaction["verify_eviction"] or os.environ.get("ARC3_EVICTION_VERIFY_ALLOCATOR","0")=="1":
        from eviction_allocator import verify_allocator_reacquisition
        allocator_receipt=verify_allocator_reacquisition(
            core.scheduler.kv_cache_manager.block_pool,freed)
    publish_context(request)
    new.poisoned=False
    phases["publish_release_optional_allocator_proof_seconds"]=time.perf_counter()-phase_start
    ack=dict(plan_digest=logical_plan.digest,resident_tokens=len(request._all_token_ids),
             evicted_tokens=logical_plan.removed_tokens,logical_tokens=len(new.logical_ids),
             eviction_revision=new.eviction_revision,frontier=transaction["frontier"],
             freed_blocks_by_group=[[b.block_id for b in g["freed_blocks"]] for g in groups],
             worker_receipts=replies,allocator_reacquisition=allocator_receipt)
    if profiling:
        phases["total_before_final_audit_seconds"]=time.perf_counter()-started
        ack["core_profile"]=phases
    engine_patch._audit(core.scheduler,"evict",request,eviction=ack)
    return ack


def _attach_manager_states(scheduler,request,state):
    coordinator=scheduler.kv_cache_manager.coordinator
    for group,manager in zip(worker_owner_groups(coordinator.kv_cache_config),
                             coordinator.single_type_managers):
        if not hasattr(manager,"_arc3_eviction_states"):
            manager._arc3_eviction_states={}
            spec=group.kv_cache_spec
            specs=getattr(spec,"kv_cache_specs",None)
            names=list(specs) if specs else list(group.layer_names)
            manager._arc3_has_compressed=any(n.endswith("compressed_key_cache") for n in names)
            manager._arc3_table_capacity=spec.max_num_blocks_per_req(
                scheduler.vllm_config,scheduler.max_model_len)*manager.block_size
        manager._arc3_eviction_states[request.request_id]=state


def _demand_wrapper(original):
    signature=inspect.signature(original)
    @functools.wraps(original)
    def wrapped(self,*args,**kwargs):
        if not getattr(self,"_arc3_has_compressed",False):
            return original(self,*args,**kwargs)
        bound=signature.bind(self,*args,**kwargs)
        rid=bound.arguments["request_id"]
        state=self._arc3_eviction_states.get(rid)
        if state is not None:
            if state.poisoned:
                raise EngineEvictionError("Cannot schedule poisoned eviction state")
            padding=state.group_padding
            for key in ("num_tokens","num_tokens_main_model"):
                if key in bound.arguments:
                    bound.arguments[key]+=padding
            if bound.arguments["num_tokens"]>self._arc3_table_capacity:
                raise EngineEvictionError("QSA virtual group demand exceeds worker page-table capacity")
        return original(*bound.args,**bound.kwargs)
    return wrapped


def validate_turn_capacity(scheduler, *, logical_tokens, resident_tokens,
                           max_tokens, state=None):
    """Reject impossible direct-client turns before changing a paused request."""
    logical_limit=int(scheduler.vllm_config.model_config.hf_text_config.max_position_embeddings)
    if (type(max_tokens) is not int or max_tokens<=0
            or logical_tokens+max_tokens>logical_limit):
        raise EngineEvictionError("Turn exceeds the original model position limit")
    padding=state.group_padding if state is not None else 0
    for manager in scheduler.kv_cache_manager.coordinator.single_type_managers:
        if getattr(manager,"_arc3_has_compressed",False):
            if resident_tokens+max_tokens+padding>manager._arc3_table_capacity:
                raise EngineEvictionError("Turn exceeds the QSA descriptor page-table capacity")


def install():
    """Install after engine_patch.install(); only explicit eviction requests change."""
    import engine_patch
    from vllm.v1.core.sched.scheduler import Scheduler
    from vllm.v1.core.single_type_kv_cache_manager import SingleTypeKVCacheManager
    from vllm.v1.engine.core import EngineCore
    from vllm.v1.engine import core as core_module
    from vllm.v1.core import kv_cache_utils as cache_utils
    import vllm
    if getattr(Scheduler,"_arc3_eviction_patch",None)==VERSION:
        return {"patch":VERSION}
    if getattr(Scheduler,"_arc3_persistent_patch",None)!=engine_patch.PATCH_VERSION:
        raise EngineEvictionError("Install the pinned phase-one lifecycle first")
    runtime_root=Path(vllm.__file__).resolve().parent
    for relative,expected in SOURCE_HASHES.items():
        actual=hashlib.sha256((runtime_root/relative).read_bytes()).hexdigest()
        if actual!=expected:
            raise EngineEvictionError(f"Unreviewed eviction runtime source: {relative} ({actual})")
    old_add=Scheduler.add_request
    old_append=Scheduler._update_request_as_session
    old_stop=Scheduler._handle_stopped_request
    old_free=Scheduler._free_request
    old_audit=engine_patch._audit
    old_generate=core_module.generate_scheduler_kv_cache_config
    if cache_utils.generate_scheduler_kv_cache_config is not old_generate:
        raise EngineEvictionError("Scheduler-config generator alias has an unreviewed modification")

    @functools.wraps(old_add)
    def add(self,request):
        if marked(request) and request.request_id not in self.requests:
            from eviction_runner import graph_mapping_enabled, validate_execution_mode
            if graph_mapping_enabled():
                validate_execution_mode(self.vllm_config)
            elif not self.vllm_config.model_config.enforce_eager:
                raise EngineEvictionError("Initial eviction validation requires eager execution")
            if not self.vllm_config.use_v2_model_runner:
                raise EngineEvictionError("Initial eviction validation requires MRV2")
            validate_turn_capacity(self,logical_tokens=len(request.prompt_token_ids),
                resident_tokens=len(request.prompt_token_ids),max_tokens=request.max_tokens)
            state=LogicalState.create(request)
            request._arc3_eviction_state=state
            _attach_manager_states(self,request,state)
            publish_context(request)
        return old_add(self,request)

    @functools.wraps(old_append)
    def append(self,request,update):
        state=getattr(request,"_arc3_eviction_state",None)
        if state is None:
            return old_append(self,request,update)
        if state.poisoned or not (update.sampling_params.extra_args or {}).get(MARKER):
            raise EngineEvictionError("Missing or unhealthy eviction session state")
        state.paused(request)
        delta=list(update.prompt_token_ids or ())
        old_length=len(state.logical_ids)
        validate_turn_capacity(self,logical_tokens=old_length+len(delta),
            resident_tokens=len(request._all_token_ids)+len(delta),
            max_tokens=update.max_tokens,state=state)
        placements=image_features(update.mm_features or (),offset=old_length)
        # Phase one's append emits its audit immediately. Defer that one event
        # until both logical and dense ledgers describe the newly appended turn.
        request._arc3_defer_resume_audit=True
        try:
            result=old_append(self,request,update)
        finally:
            request._arc3_defer_resume_audit=False
        state.logical_ids.extend(delta)
        state.ledger.append(len(delta))
        state.prompt_logical_length=len(state.logical_ids)
        state.prompt_positions=tuple(state.ledger.resident_positions())
        state.logical_image_features.extend(placements)
        publish_context(request)
        engine_patch._audit(self,"resume",request,append=request._arc3_last_append)
        return result

    @functools.wraps(old_stop)
    def stop(self,request):
        state=getattr(request,"_arc3_eviction_state",None)
        if state is not None:
            state.paused(request)
        return old_stop(self,request)

    @functools.wraps(old_free)
    def free(self,request,*args,**kwargs):
        result=old_free(self,request,*args,**kwargs)
        for manager in self.kv_cache_manager.coordinator.single_type_managers:
            getattr(manager,"_arc3_eviction_states",{}).pop(request.request_id,None)
        return result

    @functools.wraps(old_audit)
    def audit(scheduler,event,request,**extra):
        if event=="resume" and getattr(request,"_arc3_defer_resume_audit",False):
            return
        state=getattr(request,"_arc3_eviction_state",None)
        if state is not None:
            extra["logical_eviction_state"]={
                "logical_tokens":len(state.logical_ids),"logical_sha256":logical_digest(state),
                "eviction_revision":state.eviction_revision,"deleted_tokens":state.deleted_tokens,
                "deleted_groups":state.deleted_groups,"group_padding":state.group_padding,
                "evicted_ranges":[list(r) for r in state.ledger.evicted],"poisoned":state.poisoned}
        return old_audit(scheduler,event,request,**extra)

    Scheduler.add_request=add
    Scheduler._update_request_as_session=append
    Scheduler._handle_stopped_request=stop
    Scheduler._free_request=free
    SingleTypeKVCacheManager.get_num_blocks_to_allocate=_demand_wrapper(
        SingleTypeKVCacheManager.get_num_blocks_to_allocate)
    SingleTypeKVCacheManager.allocate_new_blocks=_demand_wrapper(
        SingleTypeKVCacheManager.allocate_new_blocks)
    EngineCore.arc3_evict_resident=evict_resident
    import qsa_frequency
    EngineCore.arc3_qsa_frequency_snapshot=qsa_frequency.core_snapshot
    EngineCore.arc3_qsa_frequency_snapshot_compact=qsa_frequency.core_snapshot_compact
    patched_generate=preserve_worker_owner_specs(old_generate)
    core_module.generate_scheduler_kv_cache_config=patched_generate
    cache_utils.generate_scheduler_kv_cache_config=patched_generate
    engine_patch._audit=audit
    Scheduler._arc3_eviction_patch=VERSION
    from eviction_runner import graph_mapping_enabled
    graph_mapping=graph_mapping_enabled()
    return {"patch":VERSION,"eager_only":not graph_mapping,
            "graph_mapping":graph_mapping,"source_hashes":dict(SOURCE_HASHES)}
