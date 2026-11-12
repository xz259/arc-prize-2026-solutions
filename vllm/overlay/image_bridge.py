"""Explicit diagnostic backend composition; no import-time installation.

Arbitrary stop boundaries here are numerical prepared-input diagnostics. Normal
chat rendering, im_end/tool checks and response parsing are delegated unchanged.
"""
import base64
import copy
import hashlib
import inspect
import json
import re
from pathlib import Path
from types import SimpleNamespace

from image_admission import enabled, image_layout, require, MAX_TOKENS, MAX_IMAGES

HERE = Path(__file__).resolve().parent
KEY = 'image_diagnostic'
SCHEMA = 'arc3-prepared-image-diagnostic-v1'
MODEL = 'nvidia/Qwen3.8-Flash-Next-NVFP4'


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_helpers(endpoint, protocol):
    pins = json.loads((HERE/'BASE.json').read_text())['files']
    for name, module in [('persistent_endpoint.py', endpoint), ('protocol.py', protocol)]:
        require(file_sha(module.__file__) == pins[name], 'Unreviewed bridge helper: '+name)


def verify_runtime():
    import vllm
    runtime = Path(vllm.__file__).resolve().parent
    runtime_pins = json.loads((HERE/'BASE.json').read_text())['runtime_sources']
    for name, sha in runtime_pins.items():
        require(file_sha(runtime/name) == sha, 'Unreviewed renderer/cache runtime: '+name)
    return runtime_pins


def processor_identity(backend, model_root):
    """Read-only identity collector; its digest must be externally admitted."""
    model_root = Path(model_root).resolve()
    pins = json.loads((HERE/'BASE.json').read_text())['model_metadata']
    for name, sha in pins.items():
        require(file_sha(model_root/name) == sha, 'Model/processor metadata changed: '+name)
    runtime_pins = verify_runtime()
    renderer = backend.serving_chat.renderer
    processor = renderer.get_mm_processor()
    readonly = renderer._readonly_mm_processor
    require(readonly is not None and readonly is not processor and type(readonly) is type(processor)
            and type(readonly.info) is type(processor.info), 'Missing/mismatched readonly processor')
    require(type(readonly.cache).__module__ == 'vllm.multimodal.cache'
            and type(readonly.cache).__name__ == 'MultiModalProcessorOnlyCache',
            'Readonly processor must retain actual processed data')
    hf = processor.info.get_hf_processor()
    ip = hf.image_processor
    config = ip.to_dict()
    ro_hf = readonly.info.get_hf_processor()
    require(type(ro_hf) is type(hf) and type(ro_hf.image_processor) is type(ip)
            and ro_hf.image_processor.to_dict() == config, 'Readonly processor config differs')
    require(config.get('patch_size') == 16 and config.get('temporal_patch_size') == 2
            and config.get('merge_size') == 2, 'Unreviewed image patch geometry')
    objects = dict(renderer=renderer, mm_processor=processor, processing_info=processor.info,
                   hf_processor=hf, image_processor=ip, readonly_cache=readonly.cache)
    require(backend.serving_chat.online_renderer.renderer is renderer, 'Online renderer ownership differs')
    classes = {}
    for name, obj in objects.items():
        cls = type(obj)
        path = Path(inspect.getfile(cls)).resolve()
        require(path.suffix == '.py', 'Processor source is not inspectable Python')
        classes[name] = dict(module=cls.__module__, name=cls.__qualname__, source_sha256=file_sha(path))
    return dict(schema='arc3-image-processor-identity-v1', classes=classes,
                image_processor_config=config, model_metadata_sha256=pins,
                runtime_source_sha256=runtime_pins,
                cache_policy=dict(api_process_count=renderer.config.parallel_config._api_process_count,
                    api_process_rank=renderer.config.parallel_config._api_process_rank,
                    data_parallel_size=renderer.config.parallel_config.data_parallel_size,
                    data_parallel_external_lb=renderer.config.parallel_config.data_parallel_external_lb,
                    enable_prefix_caching=renderer.config.cache_config.enable_prefix_caching,
                    mm_processor_cache_gb=renderer.model_config.get_multimodal_config().mm_processor_cache_gb,
                    mm_processor_cache_type=renderer.model_config.get_multimodal_config().mm_processor_cache_type))


def fixture_message(name, *, initial):
    require(name in ('red', 'blue', 'text'), 'Unknown diagnostic fixture')
    text = ('This is a bounded numerical image diagnostic. Count carefully. '*80
            if initial else 'Inspect the current observation carefully.')
    content = [dict(type='text', text=text)]
    png_sha = None
    if name != 'text':
        path = HERE/'fixtures'/(name+'.png')
        expected = json.loads((HERE/'BASE.json').read_text())['fixtures'][path.name]
        require(file_sha(path) == expected, 'PNG fixture changed')
        png_sha = expected
        content.append(dict(type='image_url', image_url=dict(
            url='data:image/png;base64,'+base64.b64encode(path.read_bytes()).decode())))
    return dict(role='user', content=content), png_sha


def tensor_record(tensor):
    import torch
    require(tensor.device.type == 'cpu' and tensor.layout == torch.strided,
            'Processed tensor must be ordinary CPU storage')
    require(tensor.dtype in (torch.float32, torch.float16, torch.bfloat16, torch.int64),
            'Unreviewed processed tensor dtype')
    require(tensor.numel()*tensor.element_size() <= 2*1024*1024, 'Processed tensor exceeds copy bound')
    require(bool(torch.isfinite(tensor).all().item()), 'Nonfinite image processor output')
    raw = tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()
    return dict(dtype=str(tensor.dtype), shape=list(tensor.shape), bytes=len(raw),
                sha256=hashlib.sha256(raw).hexdigest())


def prepared_receipt(prompt, expected_images):
    """Inspect the actual renderer output using the pinned InputProcessor mapping."""
    ids = prompt['prompt_token_ids']
    require(type(ids) is list and ids and len(ids) <= MAX_TOKENS, 'Unbounded prepared token input')
    if not expected_images:
        require(prompt['type'] == 'token' and not prompt.get('mm_placeholders'),
                'Text fixture unexpectedly contains MM data')
        return [], []
    require(prompt['type'] == 'multimodal', 'Image fixture did not produce MM input')
    for key in ('mm_kwargs', 'mm_placeholders', 'mm_hashes'):
        require(set(prompt[key]) == {'image'} and len(prompt[key]['image']) == 1,
                'Expected exactly one processed image: '+key)
    item = prompt['mm_kwargs']['image'][0]
    require(item is not None and set(item) == {'pixel_values', 'image_grid_thw'},
            'Missing/cached-only or unexpected processed image fields')
    pixels = item['pixel_values'].data
    require(list(pixels.shape) == [256, 1536], 'Unexpected patch16 temporal2 RGB pixel tensor')
    identifier = prompt['mm_hashes']['image'][0]
    feature = SimpleNamespace(modality='image', identifier=identifier, data=item,
                              mm_position=prompt['mm_placeholders']['image'][0])
    layout = image_layout([feature], ids)
    records = [dict(identifier=identifier, tensors={k:tensor_record(item[k].data) for k in sorted(item)})]
    return layout, records


def prompt_mapping(prompt):
    """Actual outgoing IDs/hash/spans, without accessing omitted CPU tensor data."""
    require(prompt['type'] == 'multimodal', 'Expected a multimodal mapping')
    for key in ('mm_kwargs', 'mm_placeholders', 'mm_hashes'):
        require(set(prompt[key]) == {'image'} and len(prompt[key]['image']) == 1,
                'Expected exactly one image mapping: '+key)
    span = prompt['mm_placeholders']['image'][0]
    require(span.is_embed is None, 'Image pruning is not admitted')
    return dict(type=prompt['type'], token_ids=list(prompt['prompt_token_ids']),
                hashes=list(prompt['mm_hashes']['image']),
                spans=[dict(offset=span.offset, length=span.length)],
                assistant_tokens_mask=prompt.get('assistant_tokens_mask'))


async def evidence_prompt(serving_chat, protocol, payload, prompt, context, *, initial):
    """Sidecar only. Never replace normal sender-cache omissions in submitted input."""
    prompt_mapping(prompt)
    if prompt['mm_kwargs']['image'][0] is not None:
        return prompt, 'normal_render_actual_payload'
    renderer = serving_chat.renderer
    require(renderer._readonly_mm_processor is not None,
            'Cached omission requires the existing readonly processor')
    require(serving_chat.online_renderer.renderer is renderer, 'Sidecar renderer changed')
    request = protocol._request_factory(copy.deepcopy(payload))
    result = await serving_chat.online_renderer.render_chat(request, skip_mm_cache=True)
    require(isinstance(result, tuple) and len(result) == 2 and len(result[1]) == 1,
            'Readonly image sidecar render failed')
    inspected = result[1][0]
    if not initial:
        inspected = protocol.slice_rendered_delta(inspected, context.anchor_ids, context.separator_ids)
    require(prompt_mapping(inspected) == prompt_mapping(prompt),
            'Readonly image sidecar token/hash/span mapping differs')
    require(inspected['mm_kwargs']['image'][0] is not None, 'Readonly image sidecar omitted data')
    return inspected, 'readonly_render_after_sender_omission'


def descriptor(payload, session, *, initial):
    raw = payload.get('raw')
    marked_initial = isinstance(session.initial.get('raw'), dict) and KEY in session.initial['raw']
    present = isinstance(raw, dict) and KEY in raw
    if not present and not marked_initial:
        return None
    require(enabled(), 'Prepared images require the explicit diagnostic flag')
    require(present and set(raw) == {'token_ids', KEY} and raw['token_ids'] == [],
            'Diagnostic mode is immutable and accepts no caller-rendered tokens')
    value = raw[KEY]
    require(type(value) is dict and set(value) == {'schema', 'fixture', 'prefix_token_ids', 'processor_identity_sha256'}
            and value['schema'] == SCHEMA, 'Wrong prepared-image descriptor')
    require(value['fixture'] in ('red', 'blue', 'text'), 'Unknown fixture')
    prefix = value['prefix_token_ids']
    require(type(prefix) is list and len(prefix) <= 1
            and all(type(i) is int and 0 <= i < 2**31 for i in prefix)
            and (not initial or not prefix), 'Only one actual sampled ID may prefix an append')
    require(not payload.get('continuation') and not payload.get('managed_suffixes')
            and not payload.get('external_messages') and not payload.get('chat'),
            'Prepared numerical input cannot mix chat/continuation/suffix contracts')
    require(session.history_policy.policy == 'off' and session.eviction_revision == 0,
            'Prepared image diagnostics cannot evict history')
    if initial:
        require(type(payload.get('sampling', {}).get('arc3_one_lookahead')) is bool,
                'Reference/candidate marker must be explicit')
    return value


def compose_backend(original, protocol, *, expected_processor_sha256, model_root):
    """Return a diagnostic subclass; caller installs it only in a fresh stage."""
    require(type(expected_processor_sha256) is str and len(expected_processor_sha256) == 64
            and all(c in '0123456789abcdef' for c in expected_processor_sha256),
            'An externally verified processor identity digest is required')

    class ImageDiagnosticBackend(original):
        async def prepare(self, session, payload, *, initial):
            spec = descriptor(payload, session, initial=initial)
            if spec is None:
                return await super().prepare(session, payload, initial=initial)
            require(spec['processor_identity_sha256'] == expected_processor_sha256,
                    'Caller selected a different processor identity')
            identity = processor_identity(self, model_root)
            require(digest(identity) == expected_processor_sha256, 'Actual processor identity is not admitted')
            prior = getattr(session, '_arc3_image_commit', None)
            require((prior is None) == initial, 'Missing or repeated diagnostic start')
            message, png_sha = fixture_message(spec['fixture'], initial=initial)
            if initial:
                chat = dict(model=MODEL, messages=[message], tool_choice='none',
                    add_special_tokens=False, add_generation_prompt=True, continue_final_message=False,
                    chat_template_kwargs=dict(enable_thinking=True, preserve_thinking=True, reasoning_effort='xhigh'))
                rendered = await protocol.render_initial(self.serving_chat, chat)
                prompt, context = rendered.engine_input, rendered.state
                render_payload = chat
                old_layout = []
            else:
                context = prior['context']
                require(protocol._fingerprint(self.serving_chat, context.template_kwargs) == context.renderer_sha256,
                        'Renderer changed between prepared diagnostic turns')
                # Numerical suffix only: no fabricated im_end or parsed tool boundary.
                render_payload = protocol._anchor_payload(context, [message], generate=True)
                _, full = await protocol._render(self.serving_chat, render_payload, protocol._request_factory)
                prompt = protocol.slice_rendered_delta(full, context.anchor_ids, context.separator_ids)
                old_layout = prior['image_layout']
            rendered_ids = list(prompt['prompt_token_ids'])
            inspected, evidence_source = prompt, 'text_no_processed_tensor'
            if spec['fixture'] != 'text':
                inspected, evidence_source = await evidence_prompt(self.serving_chat, protocol,
                    render_payload, prompt, context, initial=initial)
            relative, tensors = prepared_receipt(inspected, spec['fixture'] != 'text')
            prefix = spec['prefix_token_ids']
            if prefix:
                prompt = protocol.slice_rendered_delta(prompt, [], prefix)
            inspected_shifted = protocol.slice_rendered_delta(inspected, [], prefix) if prefix else inspected
            shifted, _ = prepared_receipt(inspected_shifted, spec['fixture'] != 'text')
            if spec['fixture'] != 'text':
                require(prompt_mapping(prompt) == prompt_mapping(inspected_shifted),
                        'Submitted prefix mapping differs from tensor evidence')
            require([dict(x, offset=x['offset']+len(prefix)) for x in relative] == shifted,
                    'Prefix transformed the processed image mapping')
            accepted = len(session.token_ids)
            layout = old_layout+[dict(x, offset=accepted+x['offset']) for x in shifted]
            require(len(layout) <= MAX_IMAGES, 'Cumulative image count exceeds diagnostic scope')
            sanitized = copy.deepcopy(payload)
            sanitized['raw'] = dict(token_ids=list(prompt['prompt_token_ids']))
            turn = await super().prepare(session, sanitized, initial=initial)
            require(type(turn.sampling.get('max_tokens')) is int and turn.sampling['max_tokens'] > 0
                    and accepted+len(turn.token_ids)+turn.sampling['max_tokens'] <= MAX_TOKENS,
                    'Prepared image turn exceeds the one-page oracle bound')
            require(not turn.suffix_records and not turn.evict_observation_ids and not turn.observation_starts,
                    'Prepared image input cannot carry managed deletion metadata')
            turn.prompt = prompt
            receipt = dict(schema=SCHEMA, processor_identity_sha256=expected_processor_sha256,
                fixture=spec['fixture'], png_sha256=png_sha, initial=initial,
                accepted_base=accepted, prior_revision=session.revision,
                rendered_token_ids=rendered_ids, prefix_token_ids=list(prefix),
                input_token_ids=list(prompt['prompt_token_ids']), relative_image_layout=relative,
                image_layout=layout, processed_inputs=tensors, processed_input_evidence_source=evidence_source,
                normal_chat_boundary_claimed=False)
            # Commit only from successful response; rejected preflight does not
            # advance cumulative layout or retained renderer state.
            session._arc3_image_pending = dict(operation=session.current_operation,
                context=context, image_layout=layout, receipt=receipt)
            return turn

        async def response(self, session, result):
            response = await super().response(session, result)
            initial_raw = session.initial.get('raw')
            if not isinstance(initial_raw, dict) or KEY not in initial_raw:
                return response
            pending = getattr(session, '_arc3_image_pending', None)
            require(pending is not None and pending['operation'] == session.current_operation
                    and session.revision == pending['receipt']['prior_revision']+1,
                    'Prepared image response is not bound to the current operation')
            require(session.parse_context is None and response['choices'][0].get('logprobs') == result['logprobs'],
                    'Prepared image diagnostic must retain the raw logprob response')
            response[KEY] = copy.deepcopy(pending['receipt'])
            session._arc3_image_commit = dict(context=pending['context'], image_layout=pending['image_layout'])
            del session._arc3_image_pending
            return response

    return ImageDiagnosticBackend


async def closed_image_receipt(manager, session_id, payload):
    """Read-only literal utility, restricted to the already closed image owner."""
    import one_lookahead
    require(enabled() and one_lookahead.enabled(), 'Image diagnostics are disabled')
    require(type(payload) is dict and set(payload) == {'native_request_id'},
            'Only the previously observed actual native request ID is accepted')
    session = manager.sessions.get(session_id)
    require(session is not None and session.state == 'closed', 'Image owner is not closed')
    require(all(s.state == 'closed' for s in manager.sessions.values()), 'Every local owner must be closed')
    task = session.close_task
    require(task is not None and task.done() and not task.cancelled() and task.exception() is None,
            'Owned close has not completed successfully')
    require(session.current_operation is None and all(op['task'].done() for op in session.operations.values()),
            'An owned operation has not joined')
    raw = session.initial.get('raw')
    commit = getattr(session, '_arc3_image_commit', None)
    require(type(raw) is dict and KEY in raw and commit is not None and commit['image_layout'],
            'No committed prepared-image ownership receipt')
    require(session.history_policy.policy == 'off' and session.eviction_revision == 0,
            'Closed image proof cannot follow eviction')
    pending = getattr(session, '_arc3_image_pending', None)
    require(pending is None or pending['image_layout'] == commit['image_layout'],
            'Uncommitted new images require a separate aborted-input proof')
    rid = payload['native_request_id']
    external = session.engine_request_id
    require(type(rid) is str and type(external) is str and bool(external)
            and re.fullmatch(re.escape(external)+r'(?:-[0-9a-f]{8})?', rid) is not None,
            'Actual native ID does not belong to the external image owner')
    layout = copy.deepcopy(commit['image_layout'])
    result = await manager.backend.engine.engine_core.call_utility_async(
        'arc3_one_lookahead_encoder_closed', rid, layout)
    require(type(result) is dict and result.get('complete') is True and result.get('read_only') is True
            and result.get('request_id') == rid and result.get('image_layout') == layout,
            'Closed native proof changed owner or image layout')
    require(session.state == 'closed' and session._arc3_image_commit['image_layout'] == layout
            and all(s.state == 'closed' for s in manager.sessions.values()),
            'API ownership changed during the closed proof')
    return result


def compose_attach(original):
    def attach(self, app):
        original(self, app)  # All original routes/lifespan are installed literally.
        from fastapi import APIRouter, Request
        from fastapi.responses import JSONResponse
        router = APIRouter(prefix='/arc3/persistent')

        @router.post('/{session_id}/image-encoder-closed')
        async def image_encoder_closed(session_id: str, request: Request):
            try:
                return await closed_image_receipt(request.app.state.arc3_persistent_manager,
                                                  session_id, await request.json())
            except Exception as exc:
                return JSONResponse({'error': type(exc).__name__+': '+str(exc)}, status_code=409)

        app.include_router(router)
    return attach


def install(endpoint, protocol, *, expected_processor_sha256, model_root):
    """Fresh-stage backend and one read-only closed-owner route; scheduler unchanged."""
    require(enabled(), 'Image diagnostic installation requires its flag')
    verify_helpers(endpoint, protocol)
    original = endpoint.VllmBackend
    require(original.__module__ == endpoint.__name__ and original.__name__ == 'VllmBackend',
            'Unexpected or already replaced backend binding')
    original_attach = endpoint.PersistentEndpoint.attach_router
    require(original_attach.__module__ == endpoint.__name__
            and original_attach.__qualname__ == 'PersistentEndpoint.attach_router',
            'Unexpected or already replaced persistent router binding')
    candidate = compose_backend(original, protocol, expected_processor_sha256=expected_processor_sha256,
                                model_root=model_root)
    endpoint.VllmBackend = candidate
    endpoint.PersistentEndpoint.attach_router = compose_attach(original_attach)
    return dict(schema=SCHEMA, installed=True, prepared_input_only=True,
                normal_chat_delegated=True, native_validated=False,
                bounded_image_diagnostic='arc3-lookahead-image-admission-v1',
                max_images=MAX_IMAGES, max_tokens=MAX_TOKENS,
                eviction_admitted=False, production_multimodal_admitted=False)
