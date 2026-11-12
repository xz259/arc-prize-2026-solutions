"""CPU protocol checks; GPU/MM arithmetic remains a separate live gate."""
import asyncio
import copy
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

from jinja2.sandbox import ImmutableSandboxedEnvironment
from tokenizers import Tokenizer

from protocol import (ProtocolError, render_initial, render_delta, parse_output,
                      slice_rendered_delta, token_hash)


STACK = Path(__file__).resolve().parents[2]
TOKENIZER = STACK/'evidence/optimization-20260909/kernel-checkpoint-v1/model-metadata/tokenizer.json'


@dataclass(frozen=True)
class Span:
    offset: int
    length: int
    is_embed: object = None


def request(payload):
    return SimpleNamespace(**dict(tools=[], tool_choice='auto') | payload)


class FakeServing:
    """Real template/tokenizer, explicitly fake 64-token image expansion."""
    def __init__(self, tok):
        self.renderer = SimpleNamespace(tokenizer=tok)
        self.chat_template = (STACK/'prolong_chat_template.jinja').read_text()
        self.chat_template_content_format = 'openai'
        self.model_config = SimpleNamespace(model='pinned-test-model')
        self.parser_cls = None
        self.enable_auto_tools = True
        env = ImmutableSandboxedEnvironment()
        env.globals['raise_exception'] = lambda text: (_ for _ in ()).throw(ValueError(text))
        self.template = env.from_string(self.chat_template)
        self.rendered_requests = []
        self.image_lengths = {}

    def _effective_chat_template_kwargs(self, req):
        return dict(enable_thinking=True, preserve_thinking=True, reasoning_effort='xhigh') | getattr(req, 'chat_template_kwargs', {})

    def text(self, req):
        return self.template.render(messages=req.messages, tools=req.tools,
            add_generation_prompt=getattr(req, 'add_generation_prompt', True),
            **self._effective_chat_template_kwargs(req))

    async def render_chat_request(self, req):
        self.rendered_requests.append(copy.deepcopy(req.messages))
        text = self.text(req)
        ids = self.renderer.tokenizer.encode(text, add_special_tokens=False).ids
        images = [part['image_url']['url'] for m in req.messages
                  if isinstance(m.get('content'), list) for part in m['content']
                  if part.get('type') == 'image_url']
        if not images:
            return req.messages, [dict(type='token', prompt_token_ids=ids, prompt=text)]
        pad = self.renderer.tokenizer.token_to_id('<|image_pad|>')
        image_offsets = [i for i, t in enumerate(ids) if t == pad]
        if len(image_offsets) != len(images):
            raise ProtocolError('Fixture image placeholder mismatch')
        out, spans = [], []
        for token in ids:
            if token == pad:
                length = self.image_lengths.get(images[len(spans)], 64)
                spans.append(Span(len(out), length, ('fake-embed-mask',)))
                out.extend([pad] * length)
            else:
                out.append(token)
        return req.messages, [dict(type='multimodal', prompt_token_ids=out,
            mm_placeholders={'image': spans}, mm_kwargs={'image': images},
            mm_hashes={'image': [hashlib.sha256(s.encode()).hexdigest() for s in images]}, prompt=text)]


def image(name='board'):
    return {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,'+name}}


class ProtocolTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Tokenizer.from_file(str(TOKENIZER))

    def setUp(self):
        self.serving = FakeServing(self.tokenizer)
        self.payload = dict(model='pinned-test-model', add_generation_prompt=True,
            add_special_tokens=False, continue_final_message=False, tool_choice='auto',
            tools=[dict(type='function', function=dict(name='python', parameters={}))],
            chat_template_kwargs=dict(enable_thinking=True, reasoning_effort='xhigh', preserve_thinking=True),
            messages=[dict(role='system', content='Solve the game'), dict(role='user', content='Observe')])

    async def start(self, payload=None):
        return await render_initial(self.serving, payload or self.payload, request_factory=request)

    async def delta(self, state, messages, **kwargs):
        result = await render_delta(self.serving, state, messages, request_factory=request,
            previous_output_ids=[self.tokenizer.token_to_id('<|im_end|>')], **kwargs)
        return result.engine_input

    async def test_text_delta_matches_full_template_suffix(self):
        initial = await self.start()
        scenarios = [
            [dict(role='tool', tool_call_id='a', content='{}')],
            [dict(role='tool', tool_call_id='a', content='a'), dict(role='tool', tool_call_id='b', content='b')],
            [dict(role='tool', tool_call_id='a', content='a'), dict(role='user', content='New board')],
            [dict(role='user', content='Recovery')],
            [dict(role='user', content=' αβ中🙂\n{"x": "\\n"} <|im_end|> token text ')],
            [dict(role='user', content='')],
        ]
        canonical = self.payload['messages'] + [dict(role='assistant', content='', reasoning_content='thinking')]
        for messages in scenarios:
            with self.subTest(messages=messages):
                base = request(self.payload | dict(messages=canonical, add_generation_prompt=False))
                full = request(self.payload | dict(messages=canonical+messages))
                _, (base_input,) = await self.serving.render_chat_request(base)
                _, (full_input,) = await self.serving.render_chat_request(full)
                delta = await self.delta(initial.state, messages)
                offset = len(base_input['prompt_token_ids'])
                self.assertEqual(full_input['prompt_token_ids'][:offset], base_input['prompt_token_ids'])
                self.assertEqual(delta['prompt_token_ids'], initial.state.separator_ids + full_input['prompt_token_ids'][offset:])
                text = self.tokenizer.decode(delta['prompt_token_ids'], skip_special_tokens=False)
                self.assertTrue(text.startswith('\n<|im_start|>user\n'))
                self.assertEqual(text.count('ARC3 persistent rendering anchor.'), 0)

    async def test_new_images_only_and_expanded_offsets(self):
        initial_payload = copy.deepcopy(self.payload)
        initial_payload['messages'][-1]['content'] = [dict(type='text', text='Initial'), image('old')]
        initial = await self.start(initial_payload)
        messages = [dict(role='tool', tool_call_id='a', content=[dict(type='text', text='Crop'), image('crop')]),
                    dict(role='user', content=[dict(type='text', text='Now'), image('current'), image('snapshot')])]
        delta = await self.delta(initial.state, messages)
        self.assertEqual(delta['mm_kwargs']['image'], [part['image_url']['url'] for m in messages for part in m['content'] if part['type']=='image_url'])
        self.assertNotIn('old', json.dumps(self.serving.rendered_requests[-1]))
        pad = self.tokenizer.token_to_id('<|image_pad|>')
        for span in delta['mm_placeholders']['image']:
            self.assertEqual(delta['prompt_token_ids'][span.offset:span.offset+span.length], [pad]*64)
            self.assertEqual(span.is_embed, ('fake-embed-mask',))
        self.assertEqual(delta['prompt_token_ids'].count(pad), 192)
        # Compare global offsets to a separately rendered full canonical chat.
        # Its closed-assistant newline is externally appended; the model's
        # accepted prefix ends at im_end, so remove that one newline here.
        previous = initial_payload['messages'] + [dict(role='assistant', content='', reasoning_content='an exact prior assistant')]
        _, (old,) = await self.serving.render_chat_request(request(initial_payload | dict(messages=previous, add_generation_prompt=False)))
        _, (reference,) = await self.serving.render_chat_request(request(initial_payload | dict(messages=previous+messages)))
        self.assertEqual(old['prompt_token_ids'][-1], 198)
        accepted_old_count = len(old['prompt_token_ids']) - 1
        self.assertEqual(old['prompt_token_ids'][:-1] + delta['prompt_token_ids'], reference['prompt_token_ids'])
        self.assertEqual([accepted_old_count+p.offset for p in delta['mm_placeholders']['image']],
                         [p.offset for p in reference['mm_placeholders']['image'][1:]])

    async def test_changed_same_size_image_has_new_hash(self):
        state = (await self.start()).state
        a = await self.delta(state, [dict(role='user', content=[image('red')])])
        b = await self.delta(state, [dict(role='user', content=[image('blue')])])
        self.assertEqual(a['prompt_token_ids'], b['prompt_token_ids'])
        self.assertNotEqual(a['mm_hashes'], b['mm_hashes'])

    async def test_empty_reasoning_continuation_has_no_added_tokens(self):
        state = (await self.start()).state
        delta = await render_delta(self.serving, state, [], continuation=True,
            previous_finish_reason='length', previous_output_ids=[123], request_factory=request)
        self.assertEqual(delta.engine_input, {'type':'token', 'prompt_token_ids':[]})
        for reason in ('stop', None):
            with self.assertRaises(ProtocolError):
                await render_delta(self.serving, state, [], continuation=True,
                    previous_finish_reason=reason, request_factory=request)

    async def test_fail_closed_for_bad_boundaries_and_fields(self):
        state = (await self.start()).state
        for messages in ([dict(role='assistant', content='rewrite')],
                         [dict(role='tool', content='missing ID')],
                         [dict(role='user', content='x', _arc3_observation=True)],
                         [dict(role='user', content=[dict(type='image_url', image_url={'url':'https://external'})])]):
            with self.assertRaises(ProtocolError):
                await self.delta(state, messages)
        with self.assertRaises(ProtocolError):
            await render_delta(self.serving, state, [dict(role='user', content='x')],
                previous_output_ids=[123], request_factory=request)
        self.serving.chat_template += 'changed'
        with self.assertRaises(ProtocolError):
            await self.delta(state, [dict(role='user', content='x')])

    async def test_start_policy_rejections(self):
        for patch in (dict(add_special_tokens=True), dict(n=2), dict(stop=['x']),
                      dict(add_generation_prompt=False), dict(chat_template='override'),
                      dict(truncate_prompt_tokens=10), dict(chat_template_kwargs={'add_vision_id':True})):
            with self.assertRaises(ProtocolError):
                await self.start(self.payload | patch)

    def test_slicer_does_not_mutate_mm_or_prefix(self):
        payload = dict(type='multimodal', prompt_token_ids=[1,2,3,4,5],
            mm_placeholders={'image':[Span(3,2)]}, mm_kwargs={'image':[object()]}, mm_hashes={'image':['hash']})
        delta = slice_rendered_delta(payload, [1,2], [198])
        self.assertEqual(delta['prompt_token_ids'], [198,3,4,5])
        self.assertEqual(delta['mm_placeholders']['image'][0].offset, 2)
        self.assertEqual(payload['mm_placeholders']['image'][0].offset, 3)
        self.assertIs(delta['mm_kwargs'], payload['mm_kwargs'])
        self.assertIs(delta['mm_hashes'], payload['mm_hashes'])
        with self.assertRaises(ProtocolError):
            slice_rendered_delta(payload, [1,8], [198])
        with self.assertRaises(ProtocolError):
            slice_rendered_delta(payload | {'mm_placeholders':{'image':[Span(1,2)]}}, [1,2], [198])

    async def test_parser_uses_exact_ids_and_chunk_local_usage(self):
        state = (await self.start()).state
        seen = []
        class Parser:
            def __init__(self, tok, tools, **kwargs):
                pass
            def parse(self, text, req, **kwargs):
                seen.append((text, kwargs))
                return 'reasoning', None, [SimpleNamespace(id='call_fixed', name='python', arguments='{"code":"1"}')]
        self.serving.parser_cls = Parser
        ids = self.tokenizer.encode('abc\n</think>\n\n<tool_call>\n</tool_call><|im_end|>', add_special_tokens=False).ids
        result = parse_output(self.serving, state, ids, finish_reason='stop', stop_reason=248046)
        self.assertEqual(result['token_ids'], ids)
        self.assertEqual(seen[0][1]['model_output_token_ids'], ids)
        self.assertFalse(seen[0][0].endswith('<|im_end|>'))
        self.assertEqual(result['message']['tool_calls'][0]['id'], 'call_fixed')
        self.assertEqual(result['finish_reason'], 'tool_calls')
        # If a diagnostic splits an assistant, full parsed scope is explicit;
        # the returned generated-token list is still the delta only.
        second = parse_output(self.serving, state, ids[3:], finish_reason='stop', assistant_prefix_ids=ids[:3])
        self.assertEqual(second['token_ids'], ids[3:])
        self.assertEqual(second['persistent_parse']['assistant_token_ids'], ids)
        self.assertEqual(second['persistent_parse']['scope'], 'full_assistant')

    def test_saved_output_roundtrip_counterexamples_remain_ids(self):
        evidence = json.loads((Path(__file__).parent.parent/'persistent_state_20260912/output-token-roundtrip.json').read_text())
        self.assertEqual(evidence['completions'], 352)
        self.assertEqual(evidence['roundtrip_different'], 11)
        # Equality of decoded text must not serve as the prefix guard.
        generated = [22522, 532]
        self.assertNotEqual(token_hash(generated), token_hash(generated+[1]))


if __name__ == '__main__':
    unittest.main()
