"""Exact observation-suffix ownership in processed, operation-local token IDs.

This module never modifies a prompt, KV cache, or logical position. Ownership
comes from an out-of-band content-part descriptor, and token boundaries are
accepted only after a second rendering proves the immutable prefix unchanged.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
import hashlib
import json
import re


class SpanError(ValueError):
    pass


@dataclass(frozen=True)
class ManagedRequest:
    observations: list[dict] = field(default_factory=list)
    evict_observation_ids: list[str] = field(default_factory=list)


def content_hash(content):
    try:
        encoded = json.dumps(content, ensure_ascii=False, sort_keys=True,
                             separators=(',', ':'), allow_nan=False).encode()
    except (TypeError, ValueError, UnicodeError) as exc:
        raise SpanError('Observation content is not canonical JSON') from exc
    return hashlib.sha256(encoded).hexdigest()


def content_parts(content):
    if isinstance(content, str):
        return [{'type': 'text', 'text': content}]
    if not isinstance(content, list) or not content:
        raise SpanError('Observation content requires nonempty text/image parts')
    for part in content:
        if not isinstance(part, dict) or part.get('type') not in ('text', 'image_url'):
            raise SpanError('Managed observations support text/image_url parts only')
        if part['type'] == 'text' and not isinstance(part.get('text'), str):
            raise SpanError('Text content part requires a string')
        if part['type'] == 'image_url':
            value = part.get('image_url')
            url = value.get('url') if isinstance(value, dict) else None
            if not isinstance(url, str) or not url.startswith('data:image/png;base64,'):
                raise SpanError('Managed images require self-contained PNG data URLs')
    return content


def _valid_id(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,160}', value):
        raise SpanError('Invalid managed observation ID')
    return value


def validate_managed_suffixes(sidecar, messages, *, initial=False, continuation=False):
    """Validate semantic ownership without assuming a content/BPE boundary."""
    if sidecar is None:
        return ManagedRequest()
    fields = {'version', 'observations', 'evict_observation_ids'}
    if not isinstance(sidecar, dict) or set(sidecar) != fields:
        raise SpanError('Managed suffix sidecar requires version, observations, evict_observation_ids')
    if type(sidecar['version']) is not int or sidecar['version'] != 1:
        raise SpanError('Unsupported managed suffix sidecar version')
    observations = sidecar['observations']
    evict_ids = sidecar['evict_observation_ids']
    if not isinstance(observations, list) or len(observations) > 1:
        raise SpanError('At most one new real observation is supported per operation')
    if not isinstance(evict_ids, list):
        raise SpanError('evict_observation_ids must be a list')
    for identifier in evict_ids:
        _valid_id(identifier)
    if len(set(evict_ids)) != len(evict_ids):
        raise SpanError('Duplicate evicted observation ID')
    if initial and evict_ids:
        raise SpanError('Initial operation cannot evict an earlier observation')
    if continuation and (observations or evict_ids):
        raise SpanError('Reasoning continuation cannot refresh or evict observation suffixes')
    if evict_ids and not observations:
        raise SpanError('Suffix eviction requires a new real observation descriptor')
    desc_fields = {'message_index', 'observation_id', 'content_sha256',
                   'base_parts_end', 'program_parts', 'snapshot_parts'}
    for desc in observations:
        if not isinstance(desc, dict) or set(desc) != desc_fields:
            raise SpanError('Unexpected or incomplete observation descriptor')
        identifier = _valid_id(desc['observation_id'])
        if identifier in evict_ids:
            raise SpanError('New observation cannot also be evicted')
        index = desc['message_index']
        if (type(index) is not int or not isinstance(messages, list)
                or index != len(messages)-1 or index < 0):
            raise SpanError('Managed observation must be the final newly supplied message')
        message = messages[index]
        if not isinstance(message, dict) or message.get('role') != 'user':
            raise SpanError('Managed observation must identify a user message')
        content = message.get('content')
        parts = content_parts(content)
        if desc['content_sha256'] != content_hash(content):
            raise SpanError('Managed observation content hash mismatch')
        cursor = desc['base_parts_end']
        if type(cursor) is not int or not 0 < cursor <= len(parts):
            raise SpanError('Invalid immutable observation part boundary')
        for key in ('program_parts', 'snapshot_parts'):
            span = desc[key]
            if span is None:
                continue
            if (not isinstance(span, list) or len(span) != 2
                    or any(type(x) is not int for x in span)
                    or span[0] != cursor or not cursor < span[1] <= len(parts)):
                raise SpanError('Program/snapshot parts must form one contiguous terminal suffix')
            if key == 'program_parts' and any(p['type'] != 'text' for p in parts[span[0]:span[1]]):
                raise SpanError('Program index parts must be text')
            cursor = span[1]
        if cursor != len(parts):
            raise SpanError('Observation contains unowned trailing content')
    return ManagedRequest(copy.deepcopy(observations), list(evict_ids))


def base_messages(messages, descriptor):
    result = copy.deepcopy(messages)
    index = descriptor['message_index']
    result[index]['content'] = copy.deepcopy(
        content_parts(result[index]['content'])[:descriptor['base_parts_end']])
    return result


def _ids(tokenizer, text):
    result = tokenizer.encode(text, add_special_tokens=False)
    return list(result.ids if hasattr(result, 'ids') else result)


def _single_id(tokenizer, text):
    result = _ids(tokenizer, text)
    if len(result) != 1:
        raise SpanError('Pinned multimodal boundary must be one special token')
    return result[0]


def _image_locations(messages):
    locations = []
    for message_index, message in enumerate(messages):
        content = message.get('content')
        if not isinstance(content, list):
            continue
        for part_index, part in enumerate(content):
            if isinstance(part, dict) and part.get('type') == 'image_url':
                locations.append((message_index, part_index))
    return locations


def image_inventory(engine_input, messages, tokenizer):
    """Resolve actual image features; never infer them from caption text."""
    ids = engine_input['prompt_token_ids']
    locations = _image_locations(messages)
    placeholders = engine_input.get('mm_placeholders', {})
    hashes = engine_input.get('mm_hashes', {})
    if set(placeholders) - {'image'} or set(hashes) - {'image'}:
        raise SpanError('Managed suffix mapping supports images only')
    spans = placeholders.get('image', [])
    image_hashes = hashes.get('image', [])
    if len(locations) != len(spans) or len(spans) != len(image_hashes):
        raise SpanError('Image parts, expanded placeholders and hashes disagree')
    start_id = _single_id(tokenizer, '<|vision_start|>')
    pad_id = _single_id(tokenizer, '<|image_pad|>')
    end_id = _single_id(tokenizer, '<|vision_end|>')
    result = []
    previous_end = 0
    for (message_index, part_index), span, image_hash in zip(locations, spans, image_hashes):
        offset, length = getattr(span, 'offset', None), getattr(span, 'length', None)
        if (type(offset) is not int or type(length) is not int or length <= 0
                or offset-1 < previous_end or offset+length >= len(ids)):
            raise SpanError('Invalid or overlapping expanded image placeholder')
        if (ids[offset-1] != start_id or ids[offset+length] != end_id
                or any(token != pad_id for token in ids[offset:offset+length])):
            raise SpanError('Expanded image does not match pinned visual-token envelope')
        if not isinstance(image_hash, str) or not image_hash:
            raise SpanError('Expanded image requires a stable multimodal hash')
        result.append(dict(message_index=message_index, part_index=part_index,
            feature_range=[offset, offset+length],
            envelope_range=[offset-1, offset+length+1], mm_hash=image_hash))
        previous_end = offset+length+1
    return result


def resolve_suffix_record(descriptor, full_input, base_input, messages, tokenizer):
    """Prove the one contiguous managed suffix in already-expanded token IDs.

    The final-user-message restriction leaves an exact, pinned closing tail.
    If BPE merges the immutable base with suffix text, the prefix comparison
    refuses ownership instead of deleting a token containing original content.
    """
    record = copy.deepcopy(descriptor)
    record['local_token_range'] = None
    record['image_spans'] = []
    owns = descriptor['program_parts'] is not None or descriptor['snapshot_parts'] is not None
    if not owns:
        return record
    if base_input is None:
        raise SpanError('Owned suffix requires the independently rendered immutable base')
    full_ids = full_input['prompt_token_ids']
    base_ids = base_input['prompt_token_ids']
    tail = _ids(tokenizer, '<|im_end|>\n<|im_start|>assistant\n<think>\n')
    if not tail or full_ids[-len(tail):] != tail or base_ids[-len(tail):] != tail:
        raise SpanError('Managed observation closing tail differs from the pinned template')
    start, end = len(base_ids)-len(tail), len(full_ids)-len(tail)
    if base_ids[:start] != full_ids[:start]:
        raise SpanError('Immutable observation prefix changed across the suffix BPE boundary')
    if start <= 0 or end <= start:
        raise SpanError('Managed suffix must add a nonempty rendered token interval')
    full_images = image_inventory(full_input, messages, tokenizer)
    base_images = image_inventory(base_input, base_messages(messages, descriptor), tokenizer)
    if base_images != full_images[:len(base_images)]:
        raise SpanError('Immutable image features changed during paired suffix rendering')
    for image in full_images:
        left, right = image['envelope_range']
        managed = (image['message_index'] == descriptor['message_index']
                   and image['part_index'] >= descriptor['base_parts_end'])
        if managed:
            if not start <= left < right <= end:
                raise SpanError('Managed image envelope crosses suffix token ownership')
            record['image_spans'].append(image)
        elif right > start:
            raise SpanError('Suffix ownership overlaps an immutable observation/tool image')
    record['local_token_range'] = [start, end]
    return record


def resolve_observation_start(descriptor, full_input, preceding_input, messages,
                              tokenizer, *, prefix_method):
    """Prove an observation-led block start using independent processed IDs.

    preceding_input contains precisely the prior message prefix, without any
    generation prompt. For initial rendering, a verified empty-user probe may
    have been removed because the pinned template refuses system-only input.
    Never infer this boundary by searching user text or estimating image size.
    """
    index = descriptor['message_index']
    before_messages = messages[:index]
    full_ids = full_input['prompt_token_ids']
    prefix_ids = preceding_input['prompt_token_ids']
    start = len(prefix_ids)
    if full_ids[:start] != prefix_ids:
        raise SpanError('Observation prefix changed across processed-token boundary')
    opening = _ids(tokenizer, '<|im_start|>user\n')
    if not opening or full_ids[start:start+len(opening)] != opening:
        raise SpanError('Observation start is not the complete pinned user-role opening')
    before_images = image_inventory(preceding_input, before_messages, tokenizer)
    full_images = image_inventory(full_input, messages, tokenizer)
    if before_images != full_images[:len(before_images)]:
        raise SpanError('Preceding image features changed in observation-prefix render')
    if any(image['envelope_range'][1] > start for image in before_images):
        raise SpanError('Observation start cuts a preceding image envelope')
    if any(image['envelope_range'][0] < start for image in full_images[len(before_images):]):
        raise SpanError('Observation image begins before its user-role boundary')
    return dict(observation_id=descriptor['observation_id'], local_start_token=start,
        proof=dict(method=prefix_method, preceding_token_count=start,
                   prefix_sha256=hashlib.sha256(json.dumps(prefix_ids, separators=(',', ':')).encode()).hexdigest(),
                   opening_token_ids=opening, preceding_image_count=len(before_images),
                   observation_image_count=len(full_images)-len(before_images),
                   includes_generation_prompt=False))
