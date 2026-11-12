"""Validate live diagnostic accounting with the real template/tokenizer on CPU."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from tokenizers import Tokenizer
from managed_chat_gpu_checks import Ledger,observation,descriptor,sidecar,parse_tool
from protocol import render_initial,render_delta
from raw_gpu_checks import ContractFailure,token_digest
from test_protocol import FakeServing,TOKENIZER,request


class DiagnosticAccountingTests(unittest.IsolatedAsyncioTestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = Tokenizer.from_file(str(TOKENIZER))

    async def test_real_rendered_initial_tool_and_changed_repeated_image_ledgers(self):
        serving = FakeServing(self.tokenizer)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'service.jsonl'
            path.touch()
            ledger = Ledger('fixture',path)
            initial_state = None
            for number,color in enumerate(('red',None,'blue','blue'),1):
                if color is None:
                    messages = [dict(role='tool',tool_call_id='x',content='inspection finished')]
                    managed = sidecar()
                else:
                    user,_,_ = observation(color,'blue' if color=='red' else 'red',number)
                    messages = [user]
                    managed = sidecar(descriptor(user,number),list(ledger.live))
                if initial_state is None:
                    rendered = await render_initial(serving,dict(model='pinned-test-model',messages=messages),
                        managed_suffixes=managed,request_factory=request)
                    initial_state = rendered.state
                else:
                    rendered = await render_delta(serving,initial_state,messages,
                        managed_suffixes=managed,previous_output_ids=[248046],request_factory=request)
                new_ids = rendered.engine_input['prompt_token_ids']
                removed = sum(ledger.live[oid][1]-ledger.live[oid][0]
                              for oid in managed['evict_observation_ids'])
                resident_input = ledger.resident-removed+len(new_ids)
                emitted = [64,248046]
                full = ledger.ids+new_ids+emitted
                ack = dict(registered_observation_ids=[d['observation_id'] for d in managed['observations']],
                           evicted_observation_ids=managed['evict_observation_ids'])
                rid = f'op{number}'
                row = dict(session_id='fixture',event='submit',operation_id=rid,
                    appended_token_ids=new_ids,suffix_records=rendered.suffix_records,
                    managed_suffixes=ack,old_tokens=len(ledger.ids),input_tokens=resident_input)
                with path.open('a') as stream:
                    stream.write(json.dumps(row)+'\n')
                response = dict(choices=[dict(token_ids=emitted,message={})],persistent=dict(
                    session_id='fixture',revision=ledger.revision+1,
                    logical_input_tokens=len(ledger.ids)+len(new_ids),logical_tokens=len(full),
                    resident_tokens=resident_input+len(emitted),appended_prompt_tokens=len(new_ids),
                    prefix_sha256=token_digest(full),evicted_tokens=removed,
                    eviction_revision=ledger.eviction_revision+int(removed>0),eviction=True,
                    prefix_caching=False,engine_finish_reason='stop',managed_suffixes=ack,token_ids=emitted),
                    usage=dict(prompt_tokens=resident_input,completion_tokens=len(emitted),
                               total_tokens=resident_input+len(emitted)))
                bad = copy.deepcopy(response)
                bad['persistent']['resident_tokens'] += 1
                old_ids = list(ledger.ids)
                with self.assertRaises(ContractFailure):
                    ledger.accept(bad,rid,managed)
                self.assertEqual(ledger.ids,old_ids)
                ledger.accept(response,rid,managed)
            self.assertEqual(ledger.eviction_revision,2)
            self.assertEqual(len(ledger.checked_operations),4)
            self.assertEqual(ledger.checked_operations[1]['evicted_tokens'],0)
            self.assertGreater(len(ledger.ids),ledger.resident)

    def test_literal_tool_schema_validation(self):
        choice = {'message':{'tool_calls':[dict(id='one',function=dict(name='inspect',arguments='{"color":"blue"}'))]}}
        self.assertEqual(parse_tool(choice,'inspect','blue')['id'],'one')
        with self.assertRaises(ContractFailure):
            parse_tool(choice,'inspect','red')
        with self.assertRaises(ContractFailure):
            parse_tool(choice,'action','blue')


if __name__ == '__main__':
    unittest.main()
