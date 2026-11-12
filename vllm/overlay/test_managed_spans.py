"""Actual pinned tokenizer/template; simulated image expansion, no GPU claims."""
import copy
from dataclasses import replace
import unittest

from tokenizers import Tokenizer
from managed_spans import (SpanError, content_hash, validate_managed_suffixes,
                           resolve_suffix_record, image_inventory)
from protocol import ProtocolError, render_initial, render_delta
from test_protocol import FakeServing, TOKENIZER, request, image


def text(value):
    return {'type': 'text', 'text': value}


def descriptor(messages, *, name='obs-00000001', base_end=2, program=True, snapshot=True):
    parts = messages[-1]['content']
    return dict(message_index=len(messages)-1, observation_id=name,
        content_sha256=content_hash(parts), base_parts_end=base_end,
        program_parts=[base_end,base_end+1] if program else None,
        snapshot_parts=[base_end+int(program),len(parts)] if snapshot else None)


def sidecar(desc=None, evict=()):
    return dict(version=1, observations=[] if desc is None else [desc],
                evict_observation_ids=list(evict))


class ManagedSpanTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Tokenizer.from_file(str(TOKENIZER))

    def setUp(self):
        self.serving = FakeServing(self.tokenizer)
        self.messages = [dict(role='system', content='Solve the game'),
            dict(role='user', content=[text('Current board/action information'), image('current'),
                text('\nPrograms: explore, connected_components.\n'),
                text('Recent visual history: log action 12\n'), image('recent'),
                text('\nDistant visual history: log action 2\n'), image('distant')])]
        self.payload = dict(model='pinned-test-model', messages=self.messages,
                            chat_template_kwargs=dict(enable_thinking=True, preserve_thinking=True))

    async def initial(self, messages=None, desc=None):
        messages = self.messages if messages is None else messages
        desc = descriptor(messages) if desc is None else desc
        return await render_initial(self.serving, dict(self.payload, messages=messages),
                                    managed_suffixes=sidecar(desc), request_factory=request)

    async def delta(self, state, messages, managed):
        return await render_delta(self.serving, state, messages, managed_suffixes=managed,
            previous_output_ids=[self.tokenizer.token_to_id('<|im_end|>')], request_factory=request)

    def test_canonical_content_hash_matches_wire_contract(self):
        self.assertEqual(content_hash([{'text':'é🙂', 'type':'text'}]),
                         content_hash([{'type':'text', 'text':'é🙂'}]))
        self.assertNotEqual(content_hash('a'), content_hash([text('a')]))
        with self.assertRaises(SpanError):
            content_hash(float('nan'))

    async def test_initial_span_excludes_original_board_and_framing(self):
        before = copy.deepcopy(self.payload)
        result = await self.initial()
        record, = result.suffix_records
        self.assertEqual(record['observation_id'], 'obs-00000001')
        start,end = record['local_token_range']
        ids = result.engine_input['prompt_token_ids']
        decoded = self.tokenizer.decode(ids[start:end], skip_special_tokens=False)
        self.assertIn('Programs: explore, connected_components.', decoded)
        self.assertIn('log action 12', decoded)
        self.assertIn('log action 2', decoded)
        self.assertNotIn('Current board/action information', decoded)
        self.assertNotIn('<|im_start|>', decoded)
        self.assertEqual(ids[start-1], self.tokenizer.token_to_id('<|vision_end|>'))
        self.assertEqual(ids[end], self.tokenizer.token_to_id('<|im_end|>'))
        self.assertEqual([s['part_index'] for s in record['image_spans']], [4,6])
        self.assertEqual(sum(s['feature_range'][1]-s['feature_range'][0] for s in record['image_spans']),128)
        self.assertEqual(self.payload,before)
        self.assertTrue(all('managed_suffixes' not in str(m) for m in self.serving.rendered_requests))

    async def test_append_span_has_delta_offsets_and_preserves_tool_image(self):
        initial = await self.initial()
        messages = [dict(role='tool', tool_call_id='call1', content=[text('Python crop'),image('tool-crop')]),
                    copy.deepcopy(self.messages[-1])]
        desc = descriptor(messages,name='obs-00000002')
        result = await self.delta(initial.state,messages,sidecar(desc,('obs-00000001',)))
        record, = result.suffix_records
        start,end = record['local_token_range']
        self.assertEqual(result.evict_observation_ids,['obs-00000001'])
        spans = image_inventory(result.engine_input,messages,self.tokenizer)
        self.assertLessEqual(spans[0]['envelope_range'][1],start)
        self.assertLessEqual(spans[1]['envelope_range'][1],start)
        self.assertEqual(record['image_spans'],spans[2:])
        self.assertLess(end,len(result.engine_input['prompt_token_ids']))
        # Only new inputs are rendered; no historical first-board content.
        self.assertEqual(len(self.serving.rendered_requests[-2]),4)
        self.assertEqual(self.serving.rendered_requests[-2][-1]['content'],messages[-1]['content'][:2])
        # The extra block-start proof renders only the preceding tool message,
        # with generation disabled; current-board/suffix image work is absent.
        self.assertEqual(self.serving.rendered_requests[-1][-1],messages[0])

    async def test_variable_visual_expansion_and_duplicate_images(self):
        self.serving.image_lengths = {
            image('current')['image_url']['url']:256,
            image('recent')['image_url']['url']:1024,
            image('distant')['image_url']['url']:64}
        result = await self.initial()
        spans = result.suffix_records[0]['image_spans']
        self.assertEqual([s['feature_range'][1]-s['feature_range'][0] for s in spans],[1024,64])
        duplicate = copy.deepcopy(self.messages)
        duplicate[-1]['content'][6] = image('recent')
        result = await self.initial(duplicate)
        spans = result.suffix_records[0]['image_spans']
        self.assertEqual(spans[0]['mm_hash'],spans[1]['mm_hash'])
        self.assertNotEqual(spans[0]['feature_range'],spans[1]['feature_range'])

    async def test_program_only_snapshot_only_and_zero_suffix(self):
        variants = [([*self.messages[-1]['content'][:3]],True,False),
                    ([*self.messages[-1]['content'][:2],*self.messages[-1]['content'][3:]],False,True),
                    ([*self.messages[-1]['content'][:2]],False,False)]
        for parts,program,snapshot in variants:
            with self.subTest(program=program,snapshot=snapshot):
                messages = [self.messages[0],dict(role='user',content=parts)]
                result = await self.initial(messages, descriptor(messages,program=program,snapshot=snapshot))
                record, = result.suffix_records
                self.assertEqual(record['local_token_range'] is None,not(program or snapshot))
                self.assertEqual(len(record['image_spans']),2 if snapshot else 0)
        messages = [dict(role='user',content=[text('New level'),image('fresh')])]
        desc = descriptor(messages,name='obs-00000002',program=False,snapshot=False)
        initial = await self.initial()
        result = await self.delta(initial.state,messages,sidecar(desc,('obs-00000001',)))
        self.assertIsNone(result.suffix_records[0]['local_token_range'])
        self.assertEqual(result.evict_observation_ids,['obs-00000001'])

    async def test_same_observation_tool_loop_does_no_ownership_work(self):
        initial = await self.initial()
        before = len(self.serving.rendered_requests)
        result = await self.delta(initial.state,[dict(role='tool',tool_call_id='a',content='done')],sidecar())
        self.assertEqual(result.suffix_records,[])
        self.assertEqual(result.evict_observation_ids,[])
        self.assertEqual(len(self.serving.rendered_requests)-before,1)
        before = len(self.serving.rendered_requests)
        continued = await render_delta(self.serving,initial.state,[],continuation=True,
            previous_finish_reason='length',previous_output_ids=[123],managed_suffixes=sidecar(),request_factory=request)
        self.assertEqual(continued.engine_input['prompt_token_ids'],[])
        self.assertEqual(len(self.serving.rendered_requests),before)

    async def test_bpe_boundary_merge_refused_without_mutating_state(self):
        initial = await self.initial()
        state_before = copy.deepcopy(initial.state)
        messages = [dict(role='user',content=[text('a'),text('bc')])]
        desc = descriptor(messages,base_end=1,program=True,snapshot=False)
        with self.assertRaisesRegex(ProtocolError,'BPE boundary'):
            await self.delta(initial.state,messages,sidecar(desc))
        self.assertEqual(initial.state,state_before)

    async def test_forged_hash_and_nonfinal_message_refused_before_rendering(self):
        for patch in ({'content_sha256':'0'*64},{'message_index':0},
                      {'program_parts':[1,3]},{'snapshot_parts':[4,7]},
                      {'program_parts':[2,4],'snapshot_parts':[3,7]},
                      {'base_parts_end':True},{'program_parts':None},
                      {'unexpected':'x'}):
            with self.subTest(patch=patch):
                desc = descriptor(self.messages) | patch
                before = len(self.serving.rendered_requests)
                with self.assertRaises(ProtocolError):
                    await self.initial(desc=desc)
                self.assertEqual(len(self.serving.rendered_requests),before)

    def test_invalid_sidecars(self):
        good = descriptor(self.messages)
        cases = [dict(version=True,observations=[],evict_observation_ids=[]),
                 sidecar(None,('old',)),sidecar(good,('old','old')),
                 sidecar(good,('obs-00000001',)),
                 dict(version=1,observations=[good,good],evict_observation_ids=[]),
                 dict(version=1,observations=[],evict_observation_ids=[],extra=1)]
        for value in cases:
            with self.subTest(value=value),self.assertRaises(SpanError):
                validate_managed_suffixes(value,self.messages)
        with self.assertRaises(SpanError):
            validate_managed_suffixes(sidecar(good,('old',)),self.messages,initial=True)
        with self.assertRaises(SpanError):
            validate_managed_suffixes(sidecar(good),self.messages,continuation=True)

    async def test_image_mapping_detects_changed_hash_span_and_boundary(self):
        initial = await self.initial()
        desc = descriptor(self.messages)
        base_messages = copy.deepcopy(self.messages)
        base_messages[-1]['content'] = base_messages[-1]['content'][:2]
        _, (base,) = await self.serving.render_chat_request(request(dict(self.payload,messages=base_messages)))
        source = initial.engine_input
        variants = []
        changed = copy.deepcopy(source)
        changed['mm_hashes']['image'][0] = 'different'
        variants.append(changed)
        changed = copy.deepcopy(source)
        changed['mm_placeholders']['image'][1] = replace(changed['mm_placeholders']['image'][1],length=63)
        variants.append(changed)
        changed = copy.deepcopy(source)
        changed['prompt_token_ids'][changed['mm_placeholders']['image'][1].offset-1] = 123
        variants.append(changed)
        for value in variants:
            with self.subTest(value=value),self.assertRaises(SpanError):
                resolve_suffix_record(desc,value,base,self.messages,self.tokenizer)


if __name__ == '__main__':
    unittest.main()
