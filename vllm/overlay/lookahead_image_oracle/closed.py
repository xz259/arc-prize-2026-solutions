"""Metadata-only proof of closed image-owner release; global cache may remain."""
from .receipt_schema import require,same,json_digest,digest_ok
from .image_metadata import validate_images,encoder_descriptors,require_image_flag
from .guards import validate_config
from .provenance import verify_sources

SCHEMA='arc3-one-lookahead-image-closed-v1'


def closed_core_state(core,request_id,images,one_lookahead):
    require(type(request_id) is str and request_id,'Actual native request ID required')
    validate_images(images,2048)
    require_image_flag(images)
    scheduler=core.scheduler;validate_config(scheduler.vllm_config)
    require(one_lookahead.enabled(),'Diagnostic flag required')
    require(not scheduler.requests and not core.batch_queue and not core.model_executor.futures_queue
            and not one_lookahead.pending(scheduler),'All owners must be closed and native work drained')
    manager=scheduler.encoder_cache_manager
    require(request_id not in manager.request_cached_ids
            and all(request_id not in refs for refs in manager.cached.values()),'Closed owner still holds encoder references')
    return dict(request_id=request_id,core_requests_empty=True,queues_empty=True,
        no_pending_transactions=True,owner_input_refs_absent=True,
        entries=[dict(identifier=d['identifier'],cached=d['identifier'] in manager.cached,
            refs=sorted(manager.cached.get(d['identifier'],set())),
            freeable_tokens=manager.freeable.get(d['identifier']),
            pending_free=d['identifier'] in manager.freed) for d in encoder_descriptors(images)])


def closed_worker_state(worker,wire):
    manifest,source_sha=verify_sources()
    require(source_sha==wire['source_manifest_sha256'],'Closed core/worker sources differ')
    runner=worker.model_runner;validate_config(runner.vllm_config)
    require(runner.execute_model_state is None,'Unfinished worker output')
    states=runner.req_states;rid=wire['request_id']
    require(not states.req_id_to_index and not states.index_to_req_id,'Worker owners remain after close')
    cache=runner.encoder_cache
    require(rid not in cache.mm_features,'Closed worker owner still has image association')
    validate_images(wire['image_layout'],2048)
    result=dict(request_id=rid,worker_requests_empty=True,owner_features_absent=True,
        retained_identifiers=sorted(d['identifier'] for d in encoder_descriptors(wire['image_layout'])
                                   if d['identifier'] in cache.encoder_outputs))
    require(verify_sources()[1]==source_sha,'Sources changed during closed proof')
    return result


def core_closed_encoder(core,request_id,image_layout):
    """Call only after closing every owner; never flush, reclaim or edit state."""
    import one_lookahead
    manifest,source_sha=verify_sources()
    before=closed_core_state(core,request_id,image_layout,one_lookahead)
    wire=dict(request_id=request_id,image_layout=image_layout,source_manifest_sha256=source_sha)
    replies=core.collective_rpc(closed_worker_state,timeout=30,args=(wire,))
    require(type(replies) is list and len(replies)==1,'Exactly one closed worker proof required')
    require(same(before,closed_core_state(core,request_id,image_layout,one_lookahead)),
            'Closed encoder ownership changed during proof')
    require(verify_sources()[1]==source_sha,'Sources changed during closed proof')
    result=dict(schema=SCHEMA,complete=True,read_only=True,source_manifest=manifest,
        source_manifest_sha256=source_sha,request_id=request_id,image_layout=image_layout,
        core=before,worker=replies[0])
    validate_closed_receipt(result,request_id=request_id,image_layout=image_layout,source_sha=source_sha)
    return result


def validate_closed_receipt(result,*,request_id,image_layout,source_sha):
    validate_images(image_layout,2048)
    require(digest_ok(source_sha) and type(result) is dict and result.get('schema')==SCHEMA
            and result.get('complete') is True and result.get('read_only') is True,'Incomplete closed proof')
    require(result.get('request_id')==request_id and same(result.get('image_layout'),image_layout),
            'Closed proof belongs to another owner/images')
    require(result.get('source_manifest_sha256')==source_sha
            and type(result.get('source_manifest')) is dict
            and json_digest(result['source_manifest'])==source_sha,'Closed proof source mismatch')
    c,w=result.get('core'),result.get('worker')
    require(type(c) is dict and type(w) is dict and c.get('request_id')==w.get('request_id')==request_id,
            'Missing native closed proof')
    require(all(c.get(k) is True for k in ('core_requests_empty','queues_empty','no_pending_transactions',
                                         'owner_input_refs_absent'))
            and w.get('worker_requests_empty') is True and w.get('owner_features_absent') is True,
            'Native closed-owner release not proved')
    identifiers=[d['identifier'] for d in encoder_descriptors(image_layout)]
    require(type(c.get('entries')) is list and [e.get('identifier') for e in c['entries']]==identifiers
            and all(type(e.get('refs')) is list and request_id not in e['refs'] for e in c['entries']),
            'Closed core encoder reference inventory differs')
    retained=w.get('retained_identifiers')
    require(type(retained) is list and retained==sorted(set(retained))
            and set(retained)<=set(identifiers),'Unknown retained encoder output')
    return result
