"""Bounded live chat/MM protocol probe; root executes on the owned test server.

This checks actual image rendering, tool parsing and persisted external appends.
It does not prove KV/MRoPE numerical parity. Python outputs below are explicitly
simulated fixtures from one parsed literal print, never arbitrary execution.
"""
import argparse
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import time
import uuid

import requests
from PIL import Image


MODEL = 'nvidia/Qwen3.8-Flash-Next-NVFP4'
COLORS = {'red': (249, 60, 49), 'blue': (30, 147, 255)}
TOOLS = [{'type': 'function', 'function': {
    'name': 'python',
    'description': 'Report the image color with a single literal print("red") or print("blue").',
    'parameters': {'type': 'object', 'properties': {'code': {'type': 'string'}},
                   'required': ['code']},
}}]


def encode_image(color):
    out = io.BytesIO()
    Image.new('RGB', (256, 256), COLORS[color]).save(out, format='PNG')
    data = out.getvalue()
    return data, {'type': 'image_url', 'image_url': {
        'url': 'data:image/png;base64,'+base64.b64encode(data).decode()}}


def literal_print(code):
    """Evaluate only the shape print(one_literal_string), without exec/eval."""
    parsed = ast.parse(code)
    if len(parsed.body) != 1 or not isinstance(parsed.body[0], ast.Expr):
        raise ValueError('Expected a single print expression')
    call = parsed.body[0].value
    if (not isinstance(call, ast.Call) or not isinstance(call.func, ast.Name)
            or call.func.id != 'print' or len(call.args) != 1 or call.keywords
            or not isinstance(call.args[0], ast.Constant)
            or type(call.args[0].value) is not str):
        raise ValueError('Expected print(one literal string)')
    return call.args[0].value


def check_response(response):
    choices = response.get('choices')
    state = response.get('persistent')
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(state, dict):
        raise ValueError('Missing single choice or persistent receipt')
    ids = state.get('token_ids')
    if not isinstance(ids, list) or not ids or any(type(i) is not int for i in ids):
        raise ValueError('Missing exact emitted token IDs')
    if choices[0].get('token_ids') != ids:
        raise ValueError('Choice and persistent token IDs disagree')
    if response['usage']['completion_tokens'] != len(ids):
        raise ValueError('Chunk usage is not the emitted token delta')
    if state.get('eviction') or state.get('prefix_caching'):
        raise ValueError('Expected no eviction and prefix caching disabled')
    return choices[0], state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-url', default='http://127.0.0.1:1234')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-tokens', type=int, default=2048,
                        help='Per-operation diagnostic cap, not a gameplay setting')
    parser.add_argument('--max-fragments', type=int, default=4)
    parser.add_argument('--timeout', type=float, default=300)
    args = parser.parse_args()
    if args.max_tokens < 1 or args.max_fragments < 1:
        parser.error('Caps must be positive')
    args.output.mkdir(parents=True, exist_ok=False)
    base = args.base_url.rstrip('/')+'/arc3/persistent'
    sid = 'mm-protocol-'+uuid.uuid4().hex[:16]
    http = requests.Session()
    result = {'session_id': sid, 'base_url': base, 'started': time.time(),
              'scope': 'Live renderer/tool/MM append protocol, not numerical state parity',
              'tool_execution': 'Simulated literal print fixture; no arbitrary code executed',
              'diagnostic_max_tokens': args.max_tokens, 'turns': [], 'passed': False}
    index = 0

    def save_result():
        (args.output/'result.json').write_text(json.dumps(result, indent=2)+'\n')

    def call(method, suffix, payload=None):
        nonlocal index
        index += 1
        stem = f'{index:03d}'
        (args.output/f'{stem}-request.json').write_text(json.dumps({
            'method': method, 'url': base+suffix, 'json': payload, 'epoch': time.time()}, indent=2)+'\n')
        started = time.monotonic()
        try:
            response = http.request(method, base+suffix, json=payload, timeout=args.timeout)
        except Exception as exc:
            (args.output/f'{stem}-transport-error.json').write_text(json.dumps({
                'error': f'{type(exc).__name__}: {exc}', 'elapsed_seconds': time.monotonic()-started})+'\n')
            raise
        (args.output/f'{stem}-response.txt').write_text(response.text)
        (args.output/f'{stem}-receipt.json').write_text(json.dumps({
            'status_code': response.status_code, 'elapsed_seconds': time.monotonic()-started})+'\n')
        response.raise_for_status()
        return response.json()

    def finish_assistant(response, logical_turn):
        for fragment in range(args.max_fragments):
            choice, state = check_response(response)
            if state['engine_finish_reason'] != 'length':
                if state['token_ids'][-1] != 248046:
                    raise ValueError('Completed Qwen assistant did not emit im_end')
                return response
            if fragment == args.max_fragments-1:
                raise ValueError('Assistant did not finish within diagnostic fragment budget')
            response = call('POST', '/'+sid+'/append', dict(
                request_id=f'{sid}-turn{logical_turn}-continuation{fragment}',
                expected_revision=state['revision'], expected_prefix_sha256=state['prefix_sha256'],
                external_messages=[], continuation=True,
                sampling=dict(max_tokens=args.max_tokens, temperature=0, top_p=1, top_k=0, min_p=0)))
        raise AssertionError('unreachable')

    previous_tool = None
    previous_state = None
    try:
        result['health_before'] = call('GET', '/health')
        for turn, color in enumerate(('red', 'blue', 'blue')):
            png, part = encode_image(color)
            (args.output/f'turn{turn}-{color}.png').write_bytes(png)
            content = [dict(type='text', text=(
                'Inspect only the latest attached image. What single color fills it? '
                'Call python once with code print("red") or print("blue"). '
                'Do not infer the answer from previous images.')), part]
            user = dict(role='user', content=content)
            if turn == 0:
                chat = dict(model=MODEL, messages=[dict(role='system', content=(
                    'This is a short visual protocol test. Use the python tool to report '
                    'the single solid color filling the latest image, either red or blue. '
                    'Keep the task simple.')), user], tools=TOOLS, tool_choice='auto',
                    temperature=0, top_p=1, top_k=0, min_p=0,
                    max_tokens=args.max_tokens, return_token_ids=True,
                    add_special_tokens=False, add_generation_prompt=True,
                    continue_final_message=False,
                    chat_template_kwargs=dict(enable_thinking=True, reasoning_effort='xhigh', preserve_thinking=True))
                response = call('POST', '/start', dict(session_id=sid, request_id=f'{sid}-start', chat=chat))
            else:
                response = call('POST', '/'+sid+'/append', dict(request_id=f'{sid}-turn{turn}',
                    expected_revision=previous_state['revision'],
                    expected_prefix_sha256=previous_state['prefix_sha256'],
                    external_messages=[previous_tool, user], continuation=False,
                    sampling=dict(max_tokens=args.max_tokens, temperature=0, top_p=1, top_k=0, min_p=0)))
            first_state = response['persistent']
            response = finish_assistant(response, turn)
            choice, state = check_response(response)
            calls = choice['message'].get('tool_calls') or []
            if len(calls) != 1 or calls[0]['function']['name'] != 'python':
                raise ValueError('Expected exactly one parsed python call')
            arguments = json.loads(calls[0]['function']['arguments'])
            reported = literal_print(arguments['code']).lower().strip()
            row = dict(turn=turn, image_color=color, image_sha256=hashlib.sha256(png).hexdigest(),
                reported_color=reported, correct=reported == color, revision=state['revision'],
                appended_prompt_tokens=first_state['appended_prompt_tokens'],
                resident_tokens=state['resident_tokens'], engine_finish_reason=state['engine_finish_reason'],
                code=arguments['code'], continuation_chunks=state['revision']-first_state['revision'])
            result['turns'].append(row)
            save_result()
            if reported != color:
                raise ValueError(f'Latest-image check failed: expected {color}, reported {reported}')
            previous_tool = dict(role='tool', tool_call_id=calls[0]['id'], content=json.dumps({
                'stdout': reported+'\n', 'result': None, 'error': None}))
            previous_state = state
        # Same-size images should have the same token count, but only a new
        # tool/user delta should have been appended on turns two and three.
        if result['turns'][1]['appended_prompt_tokens'] != result['turns'][2]['appended_prompt_tokens']:
            result['delta_note'] = 'Same-size image append lengths differed; inspect text/code/result and receipts.'
        result['passed'] = True
    except BaseException as exc:
        result['error'] = f'{type(exc).__name__}: {exc}'
        try:
            result['status_at_error'] = call('GET', '/'+sid+'/status')
        except Exception as status_error:
            result['status_error'] = f'{type(status_error).__name__}: {status_error}'
    finally:
        try:
            result['close'] = call('POST', '/'+sid+'/close', {})
            result['status_after_close'] = call('GET', '/'+sid+'/status')
        except Exception as exc:
            result['close_error'] = f'{type(exc).__name__}: {exc}'
            result['passed'] = False
        result['finished'] = time.time()
        save_result()
        http.close()
    print(json.dumps({k: result.get(k) for k in ('passed', 'error', 'turns', 'close_error')}, indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
