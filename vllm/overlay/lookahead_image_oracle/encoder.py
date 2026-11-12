"""Read-only encoder ownership/bytes for fully consumed bounded image fixtures."""
from . import image_metadata as image
from .receipt_schema import require, same, MAX_CHUNK_BYTES
from .layout import hash_chunks


def validate_core_proof(proof,rid,images):
    require(type(proof) is dict and set(proof)=={'images','cached_input_ids','entries'}
            and same(proof['images'],images) and proof['cached_input_ids']==[],
            'Encoder ownership proof does not match image owner')
    descriptors=image.encoder_descriptors(images)
    require(type(proof['entries']) is list and len(proof['entries'])==len(descriptors),
            'Encoder ownership coverage missing')
    for entry,descriptor in zip(proof['entries'],descriptors):
        require(type(entry) is dict and set(entry)=={'identifier','refs','retained','freeable_tokens'}
                and entry['identifier']==descriptor['identifier'] and entry['retained'] is True,
                'Invalid retained encoder ownership')
        refs=entry['refs']
        require(type(refs) is list and all(type(r) is str and r for r in refs)
                and refs==sorted(set(refs)) and rid not in refs,'Target encoder input not released')
        require((bool(refs) and entry['freeable_tokens'] is None)
                or (not refs and type(entry['freeable_tokens']) is int
                    and entry['freeable_tokens']==descriptor['shape'][0]),'Encoder freeable proof differs')
    return proof


def core_encoder(scheduler,request,images):
    manager=getattr(scheduler,'encoder_cache_manager',None);rid=request.request_id
    if not images:
        if manager is not None:
            require(not manager.request_cached_ids.get(rid,set())
                    and all(rid not in refs for refs in manager.cached.values()),'Text owner retains encoder references')
        return dict(images=[],cached_input_ids=[],entries=[])
    require(manager is not None,'Core encoder manager missing')
    require(not manager.request_cached_ids.get(rid,set()),'Consumed image still owns encoder input reference')
    require(all(rid not in refs for refs in manager.cached.values()),'Consumed image still referenced by owner')
    result=[]
    for descriptor in image.encoder_descriptors(images):
        key=descriptor['identifier']
        require(key in manager.cached and key not in manager.freed,'Retained encoder output unavailable or pending free')
        refs=sorted(manager.cached[key]);freeable=manager.freeable.get(key)
        require((bool(refs) and freeable is None) or (not refs and freeable==descriptor['shape'][0]),
                'Encoder reference/freeable bookkeeping disagrees')
        result.append(dict(identifier=key,refs=refs,retained=True,freeable_tokens=freeable))
    return dict(images=images,cached_input_ids=[],entries=result)


def worker_encoder(runner,wire,torch,byte_copy,tensor_identity):
    images=wire.get('image_layout',[])
    cache=getattr(runner,'encoder_cache',None);rid=wire['request_id']
    validate_core_proof(wire.get('encoder_proof'),rid,images)
    if not images:
        require(cache is None or not cache.mm_features.get(rid,[]),'Text owner retains worker image association')
        return dict(outputs=[],views=[],feature_layout=[])
    require(cache is not None and rid in cache.mm_features,'Worker image owner association missing')
    require(same(image.feature_layout(cache.mm_features[rid],wire['original_prompt_tokens']),images),
            'Core/worker image associations disagree')
    config=runner.model_state.model.config
    require(config.vision_config.out_hidden_size==image.ENCODER_HIDDEN
            and not config.vision_config.deepstack_visual_indexes,'Encoder dimensions/deepstack outside scope')
    outputs=[];views=[]
    for descriptor in image.encoder_descriptors(images):
        key=descriptor['identifier'];require(key in cache.encoder_outputs,'Retained worker encoder output missing')
        tensor=cache.encoder_outputs[key]
        require(type(tensor) is torch.Tensor and tensor.is_cuda and tensor.device==runner.device
                and tensor.dtype==torch.bfloat16 and list(tensor.shape)==descriptor['shape']
                and tensor.is_contiguous(),'Unexpected retained encoder output view')
        flat=tensor.view(torch.uint8).view(-1)
        require(flat.numel()==descriptor['bytes'],'Encoder byte count mismatch')
        digest=hash_chunks((byte_copy(flat[i:i+MAX_CHUNK_BYTES],torch)
                            for i in range(0,flat.numel(),MAX_CHUNK_BYTES)),descriptor['bytes'])
        outputs.append(dict(descriptor,sha256=digest));views.append(dict(identifier=key,view=tensor_identity(tensor)))
    require(sum(x['bytes'] for x in outputs)<=image.MAX_ENCODER_BYTES,'Separate encoder readout budget exceeded')
    return dict(outputs=outputs,views=views,feature_layout=images)
