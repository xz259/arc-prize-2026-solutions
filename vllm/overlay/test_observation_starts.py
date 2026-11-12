"""Observation boundaries with actual tokenizer/template and simulated vision.

No source token prose search, runtime import, remote access or model execution.
"""
import copy
from types import SimpleNamespace
import unittest

from tokenizers import Tokenizer
from managed_spans import content_hash, image_inventory, resolve_observation_start, SpanError
from protocol import render_initial, render_delta, parse_output, ProtocolError
from test_protocol import FakeServing, TOKENIZER, image, request


def text(value):
    return dict(type='text',text=value)


def managed(messages, oid='obs-1'):
    content=messages[-1]['content']
    desc=dict(message_index=len(messages)-1, observation_id=oid,
              content_sha256=content_hash(content), base_parts_end=len(content),
              program_parts=None, snapshot_parts=None)
    return dict(version=1,observations=[desc],evict_observation_ids=[])


class ObservationStarts(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.tok=Tokenizer.from_file(str(TOKENIZER))

    def setUp(self):
        self.serving=FakeServing(self.tok)
        self.payload=dict(model='pinned-test-model',
            messages=[dict(role='system',content='Solve the puzzle. Search saved logs after context removal.'),
                      dict(role='user',content=[text('Current board'),image('current')])],
            tools=[dict(type='function',function=dict(name='python',parameters={}))],
            chat_template_kwargs=dict(enable_thinking=True,preserve_thinking=True,reasoning_effort='xhigh'))

    async def start(self,payload=None):
        payload=self.payload if payload is None else payload
        return await render_initial(self.serving,payload,managed_suffixes=managed(payload['messages']),request_factory=request)

    async def append(self,state,messages,*,observe=True):
        return await render_delta(self.serving,state,messages,
            managed_suffixes=managed(messages,'obs-2') if observe else None,
            previous_output_ids=[self.tok.token_to_id('<|im_end|>')],request_factory=request)

    def completed_calls(self,state,ids=('a','b')):
        # This mock stands in for the production parser, called exactly once.
        class Parser:
            def __init__(self,*args,**kwargs): pass
            def parse(self,*args,**kwargs):
                return 'reasoning',None,[SimpleNamespace(id=i,name='python',arguments='{}') for i in ids]
        self.serving.parser_cls=Parser
        output=self.tok.encode('done</think><|im_end|>',add_special_tokens=False).ids
        choice=parse_output(self.serving,state,output,finish_reason='stop')
        self.assertEqual(state.pending_tool_call_ids,[t['id'] for t in choice['message']['tool_calls']])
        return output

    async def test_initial_protects_full_instruction_tool_prefix_and_frame(self):
        result=await self.start()
        start,=result.observation_starts
        n=start['local_start_token']; ids=result.engine_input['prompt_token_ids']
        prefix=self.tok.decode(ids[:n],skip_special_tokens=False)
        self.assertIn('Solve the puzzle.',prefix)
        self.assertIn('# Tools',prefix)
        self.assertIn('Reasoning effort is set to xhigh.',prefix)
        self.assertNotIn('Current board',prefix)
        self.assertEqual(ids[n:n+len(start['proof']['opening_token_ids'])],start['proof']['opening_token_ids'])
        self.assertEqual(start['proof']['observation_image_count'],1)
        self.assertFalse(start['proof']['includes_generation_prompt'])
        self.assertTrue(prefix.endswith('<|im_end|>\n'))
        self.assertNotIn('<|im_start|>assistant',prefix)
        self.assertEqual(self.serving.rendered_requests[-1][-1],dict(role='user',content=''))
        self.assertGreater(image_inventory(result.engine_input,self.payload['messages'],self.tok)[0]['envelope_range'][0],n)

    async def test_initial_without_explicit_system_preserves_generated_tool_prefix(self):
        payload=copy.deepcopy(self.payload)
        payload['messages']=payload['messages'][1:]
        result=await self.start(payload)
        n=result.observation_starts[0]['local_start_token']
        prefix=self.tok.decode(result.engine_input['prompt_token_ids'][:n],skip_special_tokens=False)
        self.assertIn('# Tools',prefix)
        self.assertIn('Reasoning effort is set to xhigh.',prefix)

    async def test_initial_sole_observation_required_before_render(self):
        for preceding in ([dict(role='user',content='old')],
                          [dict(role='assistant',content='old')],
                          [dict(role='tool',tool_call_id='x',content='old')]):
            payload=copy.deepcopy(self.payload)
            payload['messages'][1:1]=preceding
            before=len(self.serving.rendered_requests)
            with self.assertRaisesRegex(ProtocolError,'sole observation'):
                await self.start(payload)
            self.assertEqual(len(self.serving.rendered_requests),before)

    async def test_new_user_owns_role_opening_but_not_prior_separator(self):
        result=await self.start()
        messages=[dict(role='user',content=[text('Another <|im_start|>user\n literal opening'),image('new')])]
        delta=await self.append(result.state,messages)
        start,=delta.observation_starts
        self.assertEqual(start['local_start_token'],len(result.state.separator_ids))
        ids=delta.engine_input['prompt_token_ids']
        self.assertEqual(ids[:start['local_start_token']],result.state.separator_ids)
        self.assertTrue(self.tok.decode(ids[start['local_start_token']:],skip_special_tokens=False).startswith('<|im_start|>user\nAnother'))

    async def test_completed_tool_results_and_images_stay_in_prior_block(self):
        result=await self.start()
        self.completed_calls(result.state)
        self.serving.image_lengths={image('result')['image_url']['url']:256,image('new')['image_url']['url']:1024}
        messages=[dict(role='tool',tool_call_id='a',content=[text('Completed action'),image('result')]),
                  dict(role='tool',tool_call_id='b',content='other result'),
                  dict(role='user',content=[text('New real observation'),image('new')])]
        state_before=copy.deepcopy(result.state)
        delta=await self.append(result.state,messages)
        n=delta.observation_starts[0]['local_start_token']
        ids=delta.engine_input['prompt_token_ids']
        prefix=self.tok.decode(ids[:n],skip_special_tokens=False)
        self.assertIn('Completed action',prefix)
        self.assertIn('other result',prefix)
        self.assertTrue(prefix.endswith('</tool_response><|im_end|>\n'))
        self.assertNotIn('<|im_start|>assistant',prefix)
        images=image_inventory(delta.engine_input,messages,self.tok)
        self.assertLessEqual(images[0]['envelope_range'][1],n)
        self.assertGreater(images[1]['envelope_range'][0],n)
        self.assertEqual(delta.observation_starts[0]['proof']['preceding_image_count'],1)
        self.assertEqual(result.state,state_before)

    async def test_recovery_and_continuation_do_not_open_blocks(self):
        result=await self.start()
        delta=await self.append(result.state,[dict(role='user',content=[text('Recovery: inspect current state')])],observe=False)
        self.assertEqual(delta.observation_starts,[])
        self.assertEqual(delta.suffix_records,[])
        before=len(self.serving.rendered_requests)
        continued=await render_delta(self.serving,result.state,[],continuation=True,
            previous_finish_reason='length',previous_output_ids=[100],request_factory=request)
        self.assertEqual(continued.observation_starts,[])
        self.assertEqual(len(self.serving.rendered_requests),before)

    async def test_recovery_before_new_observation_stays_in_prior_block(self):
        result=await self.start()
        messages=[dict(role='user',content=[text('Recovery, still same observation')]),
                  dict(role='user',content=[text('Actually new board'),image('new')])]
        delta=await self.append(result.state,messages)
        n=delta.observation_starts[0]['local_start_token']
        self.assertIn('Recovery, still same observation',self.tok.decode(delta.engine_input['prompt_token_ids'][:n],skip_special_tokens=False))

    async def test_pending_missing_duplicate_and_split_tool_results_fail_purely(self):
        result=await self.start()
        self.completed_calls(result.state)
        observation=dict(role='user',content=[text('New board'),image('new')])
        variants=[[],[dict(role='tool',tool_call_id='a',content='one')],
                  [dict(role='tool',tool_call_id='a',content='one'),dict(role='tool',tool_call_id='a',content='duplicate')],
                  [dict(role='tool',tool_call_id='wrong',content='bad')],
                  [dict(role='tool',tool_call_id='a',content='one'),observation,dict(role='tool',tool_call_id='b',content='two')]]
        for results in variants:
            state_before=copy.deepcopy(result.state)
            before=len(self.serving.rendered_requests)
            # No ownership descriptor is needed to enforce protocol pairing.
            with self.assertRaises(ProtocolError):
                await self.append(result.state,results+[observation],observe=False)
            self.assertEqual(result.state,state_before)
            self.assertEqual(len(self.serving.rendered_requests),before)

    async def test_duplicate_parsed_call_ids_fail_before_rendering(self):
        result=await self.start()
        self.completed_calls(result.state,ids=('same','same'))
        with self.assertRaisesRegex(ProtocolError,'duplicate'):
            await self.append(result.state,[dict(role='tool',tool_call_id='same',content='one')],observe=False)

    async def test_processed_prefix_and_multimodal_identity_mismatch_rejected(self):
        result=await self.start()
        messages=[dict(role='tool',tool_call_id='x',content=[image('result')]),
                  dict(role='user',content=[text('new'),image('new')])]
        delta=await self.append(result.state,messages)
        n=delta.observation_starts[0]['local_start_token']
        desc=managed(messages)['observations'][0]
        full=delta.engine_input
        prefix=copy.deepcopy(full)
        prefix['prompt_token_ids']=prefix['prompt_token_ids'][:n]
        for key in ('mm_placeholders','mm_hashes','mm_kwargs'):
            prefix[key]['image']=prefix[key]['image'][:1]
        good=resolve_observation_start(desc,full,prefix,messages,self.tok,prefix_method='test')
        self.assertEqual(good['local_start_token'],n)
        bad=copy.deepcopy(prefix);bad['prompt_token_ids'][-1]=42
        with self.assertRaisesRegex(SpanError,'processed-token boundary'):
            resolve_observation_start(desc,full,bad,messages,self.tok,prefix_method='test')
        bad=copy.deepcopy(prefix);bad['mm_hashes']['image'][0]='changed'
        with self.assertRaisesRegex(SpanError,'image features changed'):
            resolve_observation_start(desc,full,bad,messages,self.tok,prefix_method='test')


if __name__=='__main__':
    unittest.main()
