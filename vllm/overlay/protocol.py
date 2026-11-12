"""Pinned chat rendering around a raw-token persistent session.

No old generated text is rendered again. Only a synthetic text-only anchor is
used to ask the existing renderer how to frame new external messages; its exact
token prefix is checked then removed before anything reaches the engine.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, is_dataclass, replace
import hashlib
import json
from typing import Any
import uuid

from managed_spans import (SpanError, validate_managed_suffixes,
                           base_messages, resolve_suffix_record, resolve_observation_start)


class ProtocolError(ValueError):
    pass


ANCHOR_MESSAGES = [
    {'role': 'user', 'content': 'ARC3 persistent rendering anchor.'},
    {'role': 'assistant', 'content': ''},
]


def _ids(tokenizer, text):
    result = tokenizer.encode(text, add_special_tokens=False)
    return list(result.ids if hasattr(result, 'ids') else result)


def _decode(tokenizer, ids):
    return tokenizer.decode(list(ids), skip_special_tokens=False)


def _json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     ensure_ascii=False).encode()).hexdigest()


def token_hash(ids):
    """Canonical digest shared by protocol tests; server may use its own codec."""
    return _json_hash(list(ids))


def _check_ids(ids):
    if not isinstance(ids, list) or any(type(x) is not int or x < 0 for x in ids):
        raise ProtocolError('Expected a list of nonnegative token IDs')


def _request_factory(payload):
    from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
    return ChatCompletionRequest.model_validate(payload)


def _effective_kwargs(serving_chat, request):
    return serving_chat._effective_chat_template_kwargs(request)


def _fingerprint(serving_chat, kwargs):
    tokenizer = serving_chat.renderer.tokenizer
    # Immutable model/tokenizer identity is also pinned by the launch manifest;
    # do not serialize the full vocabulary on every append.
    specials = {s: _ids(tokenizer, s) for s in ('<|im_start|>', '<|im_end|>',
                '<think>', '</think>', '<|vision_start|>', '<|image_pad|>', '<|vision_end|>')}
    return _json_hash(dict(template=serving_chat.chat_template,
        content_format=serving_chat.chat_template_content_format,
        kwargs=kwargs, model=serving_chat.model_config.model, special_ids=specials))


@dataclass
class ProtocolState:
    request: Any
    payload: dict
    template_kwargs: dict
    renderer_sha256: str
    # Exact text-only anchor output with add_generation_prompt=False.
    anchor_ids: list[int] = field(default_factory=list)
    separator_ids: list[int] = field(default_factory=list)
    last_finish_reason: str | None = None
    last_output_ids: list[int] = field(default_factory=list)
    assistant_prefix_ids: list[int] = field(default_factory=list)
    pending_tool_call_ids: list[str] = field(default_factory=list)


@dataclass
class InitialRender:
    state: ProtocolState
    engine_input: dict
    request: Any
    suffix_records: list[dict] = field(default_factory=list)
    evict_observation_ids: list[str] = field(default_factory=list)
    observation_starts: list[dict] = field(default_factory=list)


@dataclass
class DeltaRender:
    engine_input: dict
    suffix_records: list[dict] = field(default_factory=list)
    evict_observation_ids: list[str] = field(default_factory=list)
    observation_starts: list[dict] = field(default_factory=list)
    injected_ids: list[int] = field(default_factory=list)   # assistant-turn budget: harness-authored assistant tokens


def _managed_plan(sidecar, messages, **kwargs):
    try:
        return validate_managed_suffixes(sidecar, messages, **kwargs)
    except SpanError as exc:
        raise ProtocolError(str(exc)) from exc


async def _suffix_records(serving_chat, plan, engine_input, messages, render_base):
    records = []
    try:
        for descriptor in plan.observations:
            owns = (descriptor['program_parts'] is not None
                    or descriptor['snapshot_parts'] is not None)
            base = await render_base(base_messages(messages, descriptor)) if owns else None
            records.append(resolve_suffix_record(descriptor, engine_input, base,
                messages, serving_chat.renderer.tokenizer))
    except SpanError as exc:
        raise ProtocolError(str(exc)) from exc
    return records


async def _observation_starts(serving_chat, plan, engine_input, messages,
                              render_preceding, *, prefix_method):
    starts = []
    try:
        for descriptor in plan.observations:
            prefix = await render_preceding(messages[:descriptor['message_index']])
            starts.append(resolve_observation_start(descriptor, engine_input, prefix,
                messages, serving_chat.renderer.tokenizer, prefix_method=prefix_method))
    except SpanError as exc:
        raise ProtocolError(str(exc)) from exc
    return starts


def _validate_start(payload):
    if not isinstance(payload, dict) or not payload.get('messages'):
        raise ProtocolError('Initial chat request requires messages')
    if payload.get('continue_final_message') or not payload.get('add_generation_prompt', True):
        raise ProtocolError('Initial session must start at a fresh assistant prefix')
    if payload.get('add_special_tokens', False):
        raise ProtocolError('Persistent protocol requires add_special_tokens=False')
    if payload.get('chat_template') is not None:
        raise ProtocolError('Use the pinned server template, not a request template')
    if payload.get('truncate_prompt_tokens') is not None or payload.get('kv_transfer_params'):
        raise ProtocolError('Prompt truncation and external KV transfer are unsupported')
    if payload.get('n', 1) != 1 or payload.get('stop'):
        raise ProtocolError('One output with token-ID stops only is supported')
    if payload.get('tool_choice', 'auto') not in ('auto', 'none', None):
        raise ProtocolError('Only auto/none tool choice is supported')
    for message in payload['messages']:
        if any(k.startswith('_arc3_') for k in message):
            raise ProtocolError('Private harness fields must not enter server messages')


async def _render(serving_chat, payload, request_factory):
    request = request_factory(copy.deepcopy(payload))
    rendered = await serving_chat.render_chat_request(request)
    if not isinstance(rendered, tuple) or len(rendered) != 2:
        error = rendered.model_dump() if hasattr(rendered, 'model_dump') else str(rendered)
        raise ProtocolError(f'Server chat renderer rejected input: {error}')
    _, inputs = rendered
    if len(inputs) != 1 or inputs[0].get('type') not in ('token', 'multimodal'):
        raise ProtocolError('Expected one decoder-only token or multimodal input')
    _check_ids(inputs[0].get('prompt_token_ids'))
    return request, inputs[0]


def _anchor_payload(state, external_messages, *, generate):
    payload = copy.deepcopy(state.payload)
    payload['messages'] = copy.deepcopy(ANCHOR_MESSAGES) + copy.deepcopy(external_messages)
    payload['add_generation_prompt'] = generate
    payload['continue_final_message'] = False
    payload['add_special_tokens'] = False
    # This CPU rendering anchor never imposes a generation limit; input
    # rendering itself must not enforce a stale response cap.
    payload.pop('max_tokens', None)
    payload.pop('max_completion_tokens', None)
    return payload


async def render_initial(serving_chat, payload, *, managed_suffixes=None, request_factory=None):
    """Use the normal serving renderer for the first complete chat request."""
    factory = request_factory or _request_factory
    _validate_start(payload)
    plan = _managed_plan(managed_suffixes, payload['messages'], initial=True)
    if plan.observations:
        # All earlier content belongs to the fixed protected prefix. Importing
        # historical users/assistants/tools would make its boundary ambiguous.
        before = payload['messages'][:-1]
        if (len(before) > 1 or any(m.get('role') != 'system' for m in before)
                or sum(m.get('role') == 'user' for m in payload['messages']) != 1):
            raise ProtocolError('Tracked initial input requires one sole observation after an optional system message')
    request, initial = await _render(serving_chat, payload, factory)
    kwargs = _effective_kwargs(serving_chat, request)
    if kwargs.get('add_vision_id') or kwargs.get('arc3_continue_reasoning'):
        raise ProtocolError('Initial persistent control requires unnumbered images and fresh reasoning')
    if kwargs.get('enable_thinking') is False or kwargs.get('preserve_thinking') is False:
        raise ProtocolError('Initial control supports enabled, preserved reasoning only')
    tokenizer = serving_chat.renderer.tokenizer
    prefix = _ids(tokenizer, '<|im_start|>assistant\n<think>\n')
    if initial['prompt_token_ids'][-len(prefix):] != prefix:
        raise ProtocolError('Pinned initial assistant-generation prefix mismatch')
    state = ProtocolState(request=request, payload=copy.deepcopy(payload),
        template_kwargs=copy.deepcopy(kwargs), renderer_sha256=_fingerprint(serving_chat, kwargs),
        separator_ids=_ids(tokenizer, '\n'))
    _, anchor = await _render(serving_chat, _anchor_payload(state, [], generate=False), factory)
    if anchor['type'] != 'token' or anchor.get('mm_placeholders'):
        raise ProtocolError('Synthetic anchor must be text-only')
    state.anchor_ids = list(anchor['prompt_token_ids'])
    end = _ids(tokenizer, '<|im_end|>\n')
    if state.anchor_ids[-len(end):] != end:
        raise ProtocolError('Synthetic assistant anchor must end with im_end and newline')
    async def render_base(messages):
        _, base = await _render(serving_chat, dict(payload, messages=messages), factory)
        return base
    records = await _suffix_records(serving_chat, plan, initial, payload['messages'], render_base)
    async def render_preceding(messages):
        # The pinned template raises "No user query found" for a system-only
        # rendering. A structurally empty query permits the exact same fixed
        # system/instruction/tools prefix; verify and remove its whole envelope.
        probe = dict(payload, messages=copy.deepcopy(messages)+[dict(role='user', content='')],
                     add_generation_prompt=False, continue_final_message=False)
        probe.pop('max_tokens', None)
        probe.pop('max_completion_tokens', None)
        _, rendered = await _render(serving_chat, probe, factory)
        envelope = _ids(tokenizer, '<|im_start|>user\n<|im_end|>\n')
        ids = rendered['prompt_token_ids']
        if not envelope or ids[-len(envelope):] != envelope:
            raise ProtocolError('Initial empty-user prefix probe changed its structural envelope')
        result = dict(rendered, prompt_token_ids=list(ids[:-len(envelope)]))
        result.pop('prompt', None)
        result.pop('prompt_token_offsets', None)
        result.pop('assistant_tokens_mask', None)
        return result
    starts = await _observation_starts(serving_chat, plan, initial, payload['messages'],
        render_preceding, prefix_method='paired_no_generation_empty_user_probe_v1')
    return InitialRender(state=state, engine_input=initial, request=request,
        suffix_records=records, evict_observation_ids=plan.evict_observation_ids,
        observation_starts=starts)


def validate_external_messages(messages):
    if not isinstance(messages, list) or not messages:
        raise ProtocolError('External append requires one or more new tool/user messages')
    for message in messages:
        if not isinstance(message, dict) or message.get('role') not in ('tool', 'user'):
            raise ProtocolError('Append accepts only new external tool/user messages')
        if any(k.startswith('_arc3_') for k in message):
            raise ProtocolError('Private harness fields must not enter server messages')
        if message.get('role') == 'tool' and not isinstance(message.get('tool_call_id'), str):
            raise ProtocolError('Every tool result requires its call ID')
        content = message.get('content')
        if isinstance(content, str):
            continue
        if not isinstance(content, list):
            raise ProtocolError('External content must be text or text/image parts')
        for part in content:
            if not isinstance(part, dict) or part.get('type') not in ('text', 'image_url'):
                raise ProtocolError('Only text and image_url parts are supported')
            if part['type'] == 'text' and not isinstance(part.get('text'), str):
                raise ProtocolError('Text part requires text')
            if part['type'] == 'image_url':
                image = part.get('image_url')
                url = image.get('url') if isinstance(image, dict) else None
                if not isinstance(url, str) or not url.startswith('data:image/png;base64,'):
                    raise ProtocolError('Gameplay images must be self-contained PNG data URLs')


def validate_tool_feedback(state, messages):
    """Validate call/result pairing from the authoritative completed parse.

    No state is consumed here: render/preflight rejection must be reversible.
    The next completed parse replaces pending calls. Source-level diagnostic
    callers supplying explicit output IDs before any parse remain supported.
    """
    if state.last_finish_reason is None:
        return
    pending = set(state.pending_tool_call_ids)
    if len(pending) != len(state.pending_tool_call_ids):
        raise ProtocolError('Prior assistant has duplicate tool call IDs')
    for message in messages:
        if message['role'] == 'tool':
            call_id = message['tool_call_id']
            if call_id not in pending:
                raise ProtocolError('Tool result does not match an unanswered assistant call')
            pending.remove(call_id)
        elif pending:
            raise ProtocolError('New user input would split assistant tool calls from their results')
    if pending:
        raise ProtocolError('All assistant tool calls must be answered before continuing generation')


def slice_rendered_delta(engine_input, anchor_ids, separator_ids):
    """Slice a verified synthetic prefix and shift every expanded image span.

    Image tensors and hashes are preserved by reference. Only the small token
    and placeholder-position structures are copied; this is not KV movement.
    """
    ids = engine_input.get('prompt_token_ids')
    _check_ids(ids)
    cut = len(anchor_ids)
    if ids[:cut] != list(anchor_ids):
        raise ProtocolError('Renderer changed the synthetic token prefix; delta refused')
    if len(ids) == cut:
        raise ProtocolError('Rendered external delta is empty')
    result = dict(engine_input)
    result['prompt_token_ids'] = list(separator_ids) + ids[cut:]
    # These annotations would otherwise refer to the removed synthetic prefix.
    result.pop('prompt', None)
    result.pop('prompt_token_offsets', None)
    if result.get('assistant_tokens_mask') is not None:
        mask = result['assistant_tokens_mask']
        if len(mask) != len(ids):
            raise ProtocolError('Invalid assistant-token mask length')
        result['assistant_tokens_mask'] = [0] * len(separator_ids) + list(mask[cut:])
    if engine_input['type'] == 'multimodal':
        positions = {}
        for modality, spans in engine_input['mm_placeholders'].items():
            if modality != 'image':
                raise ProtocolError('Only image multimodal appends are supported')
            positions[modality] = []
            for span in spans:
                if not is_dataclass(span) or span.offset < cut or span.length <= 0:
                    raise ProtocolError('Image span crosses synthetic prefix or is invalid')
                if span.offset + span.length > len(ids):
                    raise ProtocolError('Image span lies outside rendered input')
                positions[modality].append(replace(span, offset=span.offset-cut+len(separator_ids)))
        result['mm_placeholders'] = positions
    return result


async def render_delta(serving_chat, state, external_messages, *, continuation=False,
                       previous_output_ids=None, previous_finish_reason=None,
                       managed_suffixes=None, request_factory=None, injection_text=None):
    """Render new external content; never re-render any historical assistant.

    The session manager enforces that the session is paused with no forwards in
    flight. Its continuation branch must also enforce positive remaining model
    capacity and a pending final token before allowing an empty EngineInput.
    """
    factory = request_factory or _request_factory
    plan = _managed_plan(managed_suffixes, external_messages, continuation=continuation)
    if _fingerprint(serving_chat, state.template_kwargs) != state.renderer_sha256:
        raise ProtocolError('Model/template renderer changed during persistent session')
    if previous_finish_reason is None:
        previous_finish_reason = state.last_finish_reason
    if previous_output_ids is None:
        previous_output_ids = state.last_output_ids
    if continuation:
        if external_messages or previous_finish_reason != 'length':
            raise ProtocolError('Empty continuation is valid only after a length pause')
        if not previous_output_ids:
            raise ProtocolError('No generated token to continue')
        if injection_text is not None:
            # Assistant-turn budget: harness-authored text appended inside the paused assistant turn.
            if type(injection_text) is not str or not injection_text:
                raise ProtocolError('Assistant injection must be nonempty text')
            ids = _ids(serving_chat.renderer.tokenizer, injection_text)
            if not ids:
                raise ProtocolError('Assistant injection produced no tokens')
            return DeltaRender({'type': 'token', 'prompt_token_ids': list(ids)}, injected_ids=list(ids))
        return DeltaRender({'type': 'token', 'prompt_token_ids': []})
    validate_external_messages(external_messages)
    validate_tool_feedback(state, external_messages)
    tokenizer = serving_chat.renderer.tokenizer
    eos = _ids(tokenizer, '<|im_end|>')
    if previous_output_ids is not None:
        _check_ids(previous_output_ids)
        if previous_output_ids[-len(eos):] != eos:
            raise ProtocolError('External input requires an actual generated im_end boundary')
    _, full = await _render(serving_chat,
        _anchor_payload(state, external_messages, generate=True), factory)
    delta = slice_rendered_delta(full, state.anchor_ids, state.separator_ids)
    async def render_base(messages):
        _, base = await _render(serving_chat, _anchor_payload(state, messages, generate=True), factory)
        return slice_rendered_delta(base, state.anchor_ids, state.separator_ids)
    records = await _suffix_records(serving_chat, plan, delta, external_messages, render_base)
    async def render_preceding(messages):
        _, rendered = await _render(serving_chat,
            _anchor_payload(state, messages, generate=False), factory)
        if not messages:
            if rendered['prompt_token_ids'] != state.anchor_ids or rendered.get('mm_placeholders'):
                raise ProtocolError('Empty external prefix changed the pinned synthetic anchor')
            return dict(type='token', prompt_token_ids=list(state.separator_ids))
        return slice_rendered_delta(rendered, state.anchor_ids, state.separator_ids)
    starts = await _observation_starts(serving_chat, plan, delta, external_messages,
        render_preceding, prefix_method='paired_no_generation_preceding_messages_v1')
    return DeltaRender(delta, suffix_records=records,
                       evict_observation_ids=plan.evict_observation_ids,
                       observation_starts=starts)


def parse_output(serving_chat, state, token_ids, *, finish_reason, stop_reason=None,
                 assistant_prefix_ids=None):
    """Parse raw output through the same Qwen parser as the ordinary endpoint.

    ``token_ids`` always contains only this operation's newly generated tokens.
    If a diagnostic resumes unfinished reasoning, ``assistant_prefix_ids`` gives
    the exact prior raw assistant stream. Parsed fields then describe the whole
    assistant; ``persistent_parse`` makes that scope explicit to the client.
    """
    _check_ids(token_ids)
    prefix = list(state.assistant_prefix_ids if assistant_prefix_ids is None else assistant_prefix_ids)
    _check_ids(prefix)
    all_ids = prefix + token_ids
    tokenizer = serving_chat.renderer.tokenizer
    raw = _decode(tokenizer, all_ids)
    # Match ordinary detokenization: stop-token IDs are returned, but a final
    # EOS is normally omitted from the parser-facing text.
    stop_ids = _ids(tokenizer, '<|im_end|>')
    parse_ids = all_ids[:-len(stop_ids)] if all_ids[-len(stop_ids):] == stop_ids else all_ids
    text = _decode(tokenizer, parse_ids)
    parser_cls = serving_chat.parser_cls
    calls = []
    if parser_cls is None:
        reasoning, content = None, text
    else:
        parser = parser_cls(tokenizer, state.request.tools,
            chat_template_kwargs=state.template_kwargs, model_config=serving_chat.model_config)
        reasoning, content, calls = parser.parse(text, state.request,
            enable_auto_tools=serving_chat.enable_auto_tools,
            model_output_token_ids=all_ids)
    message = dict(role='assistant', content=content, reasoning=reasoning)
    if (calls and state.request.tools and serving_chat.enable_auto_tools
            and state.request.tool_choice in ('auto', None)):
        message['tool_calls'] = [dict(id=getattr(call, 'id', None) or 'call_'+uuid.uuid4().hex,
            type='function', function=dict(name=call.name, arguments=call.arguments)) for call in calls]
    choice = dict(index=0, message=message, token_ids=list(token_ids),
        finish_reason='tool_calls' if message.get('tool_calls') and finish_reason == 'stop' else finish_reason,
        stop_reason=stop_reason)
    if prefix:
        choice['persistent_parse'] = dict(scope='full_assistant', prefix_tokens=len(prefix),
            assistant_token_ids=all_ids, raw_text=raw)
    # This hook runs only after one unambiguously completed engine operation.
    # Rendering/preflight never advances parsing state. Idempotent HTTP retries
    # return the manager's cached response instead of invoking this hook twice.
    state.last_finish_reason = finish_reason
    state.last_output_ids = list(token_ids)
    state.assistant_prefix_ids = all_ids if finish_reason == 'length' else []
    state.pending_tool_call_ids = ([call['id'] for call in message.get('tool_calls', [])]
                                  if finish_reason == 'stop' else [])
    return choice
