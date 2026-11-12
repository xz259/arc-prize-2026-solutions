"""CPU ledger checks using real tokenizer/template, fake vision expansion/backend."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from tokenizers import Tokenizer
from history_chat_gpu_checks import Ledger,observation,descriptor,sidecar,SYSTEM,TOOLS,parse_tool,audit_chat_events
from protocol import render_initial,render_delta
from raw_gpu_checks import ContractFailure,token_digest
from session_service import SessionManager,TurnInput
from test_protocol import FakeServing,TOKENIZER,request


class Backend:
    def __init__(self):self.prepared=None;self.output_length=40;self.output_finish='stop'
    async def prepare(self,session,payload,*,initial):
        r=self.prepared
        return TurnInput(r.engine_input,dict(max_tokens=256),None,r.suffix_records,
                         r.evict_observation_ids,r.observation_starts)
    async def turn(self,session,turn,*,initial):
        n=min(self.output_length,turn.sampling['max_tokens'])
        stop=self.output_finish=='stop'
        return dict(token_ids=[64]*(n-1)+[248046 if stop else 64],finish_reason=self.output_finish,stop_reason=None)
    async def response(self,session,result):
        return dict(choices=[dict(token_ids=result['token_ids'],message=dict(content='Ready'))])
    async def evict(self,session,plan):
        return dict(plan_digest=plan.digest,resident_tokens=plan.resident_after,evicted_tokens=plan.removed_tokens,
                    logical_tokens=plan.logical_tokens,eviction_revision=session.eviction_revision+1)
    async def close(self,session):pass


class ChatHistoryTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):cls.tokenizer=Tokenizer.from_file(str(TOKENIZER))
    async def asyncSetUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.path=self.root/'service.jsonl';self.path.touch()
        self.serving=FakeServing(self.tokenizer);self.backend=Backend()
        self.manager=SessionManager(self.backend,max_context=40000,max_sessions=1,audit_path=str(self.path))
        self.ledger=Ledger('test',self.path);self.state=None;self.saved=[]
    async def asyncTearDown(self):
        await self.manager.close_all();self.temp.cleanup()

    async def send(self,messages,managed,*,continuation=False):
        initial=self.state is None
        if initial:
            rendered=await render_initial(self.serving,dict(model='fixture',messages=messages,tools=TOOLS),
                managed_suffixes=managed,request_factory=request)
            self.state=rendered.state
        else:
            rendered=await render_delta(self.serving,self.state,messages,managed_suffixes=managed,
                previous_output_ids=[64] if continuation else [248046],
                previous_finish_reason='length' if continuation else 'stop',
                continuation=continuation,request_factory=request)
        self.backend.prepared=rendered
        rid='op'+str(self.ledger.revision+1)
        payload=dict(request_id=rid,continuation=continuation)
        if initial:
            response=await self.manager.start(payload|dict(session_id='test',history_eviction=dict(
                policy='oldest_interactions',fraction=.10,resident_token_budget=1250)))
        else:
            response=await self.manager.append('test',payload|dict(expected_revision=self.ledger.revision,
                expected_prefix_sha256=token_digest(self.ledger.ids)))
        before=copy.deepcopy(self.ledger.__dict__)
        bad=copy.deepcopy(response);bad['persistent']['resident_tokens']+=1
        with self.assertRaises(ContractFailure):self.ledger.accept(bad,rid,managed,continuation=continuation)
        self.assertEqual(self.ledger.ids,before['ids']);self.assertEqual(self.ledger.oracle.__dict__,before['oracle'].__dict__)
        self.ledger.accept(response,rid,managed,continuation=continuation)
        self.saved.append((response,rid,copy.deepcopy(managed),rendered))
        return response

    async def panel(self):
        for number,color in enumerate(('red','blue','blue','red','red'),1):
            user,_,_=observation(color,'blue' if color=='red' else 'red',number)
            desc=descriptor(user,number);messages=[user]
            if number==1:messages=[dict(role='system',content=SYSTEM),user];desc['message_index']=1
            await self.send(messages,sidecar(desc,list(self.ledger.live)))
            frozen=copy.deepcopy(self.ledger.live)
            await self.send([dict(role='tool',tool_call_id='inspect',content=json.dumps(dict(
                inspection_finished=True,current_color=color,next='Call action once with this same current color.')))],sidecar())
            await self.send([dict(role='tool',tool_call_id='action',content=json.dumps(dict(
                action_finished=True,next='Finish with Ready for next board. Do not call any more tools.')))],sidecar())
            self.assertEqual(self.ledger.live,frozen)

    async def test_actual_template_whole_image_blocks_and_retained_holes(self):
        await self.panel()
        rows=self.ledger.checked_operations
        deletions=[r for r in rows if r['history_eviction']['history_removed_tokens']]
        self.assertGreaterEqual(len(deletions),2)
        self.assertGreater(len(self.ledger.ids),self.ledger.resident)
        self.assertGreaterEqual(rows[0]['observation_starts'][0]['local_start_token'],128)
        prefix=rows[0]['observation_starts'][0]['local_start_token']
        self.assertTrue(all(p in self.ledger.oracle.resident_positions for p in range(prefix)))
        self.assertEqual(sum(len(r['observation_starts']) for r in rows),5)
        self.assertEqual([r['history_eviction']['suffix_removed_tokens'] for r in rows if not r['observation_starts']],[0]*10)
        # Surviving current and recap envelopes remain explicit expanded-image tokens.
        current=rows[-3]['suffix_records'][0]['logical_token_range']
        self.assertTrue(all(p in self.ledger.oracle.resident_positions for p in range(*current)))
        self.assertTrue(any(len(r['history_eviction']['history_ranges'])>1 for r in deletions))

    async def test_empty_reasoning_continuation_registers_no_observation(self):
        user,_,_=observation('red','blue',1);desc=descriptor(user,1);desc['message_index']=1
        self.backend.output_finish='length';self.backend.output_length=16
        await self.send([dict(role='system',content=SYSTEM),user],sidecar(desc))
        self.backend.output_finish='stop'
        await self.send([],sidecar(),continuation=True)
        row=self.ledger.checked_operations[-1]
        self.assertEqual(row['delta_tokens'],0);self.assertEqual(row['expected_prefill_tokens'],1)
        self.assertEqual(row['observation_starts'],[])
        self.assertEqual(len(self.ledger.oracle.observations),1)

    async def test_false_block_start_proof_is_rejected(self):
        user,_,_=observation('red','blue',1);desc=descriptor(user,1);desc['message_index']=1
        await self.send([dict(role='system',content=SYSTEM),user],sidecar(desc))
        response,rid,managed,_=self.saved[0]
        rows=[json.loads(x) for x in self.path.read_text().splitlines()]
        row=next(r for r in rows if r['event']=='submit')
        row['observation_starts'][0]['proof']['opening_token_ids']=[248045,882,198]
        self.path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
        fresh=Ledger('test',self.path)
        with self.assertRaisesRegex(ContractFailure,'proven full user-role opening'):fresh.accept(response,rid,managed)
        self.assertEqual(fresh.ids,[])

    def test_current_color_tool_args_must_match(self):
        choice=dict(message=dict(tool_calls=[dict(id='x',function=dict(name='inspect',arguments='{"color":"blue"}'))]))
        self.assertEqual(parse_tool(choice,'inspect','blue')['id'],'x')
        with self.assertRaises(ContractFailure):parse_tool(choice,'inspect','red')

    def test_engine_audit_uses_actual_variable_generation_count(self):
        # Reuse explicit synthetic worker evidence, never mistake it for GPU output.
        from test_raw_history_eviction_checks import RawHistoryTests
        fixture=RawHistoryTests('test_synthetic_owner_byte_and_allocator_audit')
        fixture.setUp()
        try:
            c,_,_=fixture.synthetic_engine_journal()
            result=audit_chat_events(fixture.engine,fixture.root/'service.jsonl',c.sid,c.turns)
            self.assertTrue(result['valid'],result['errors'])
            c.turns[-1]['output_tokens']+=1
            result=audit_chat_events(fixture.engine,fixture.root/'service.jsonl',c.sid,c.turns)
            self.assertIn('only_new_input_and_pending_final_forwarded',[r['check'] for r in result['errors']])
        finally:fixture.tearDown()


if __name__=='__main__':unittest.main()
