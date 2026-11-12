"""No-network diagnostic/client checks against real CPU session policy code."""
import asyncio
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest

from raw_history_eviction_checks import (HistoryClient,Oracle,OracleCapacity,intervals,
                                         reject_active_capacity,audit_events)
from raw_gpu_checks import ContractFailure,HTTPFailure,write_json
from session_service import SessionManager,SessionError,TurnInput


class Backend:
    def __init__(self):self.closed=[];self.submissions=[];self.evictions=[]
    async def prepare(self,session,payload,*,initial):
        raw=payload['raw']
        return TurnInput(dict(type='token',prompt_token_ids=raw['token_ids']),dict(payload['sampling']),
                         None,raw['suffix_records'],raw['evict_observation_ids'],raw['observation_starts'])
    async def turn(self,session,turn,*,initial):
        self.submissions.append((session.session_id,list(turn.token_ids)))
        return dict(token_ids=list(range(501,509)),finish_reason='length',stop_reason=None)
    async def response(self,session,result):return dict(choices=[dict(token_ids=result['token_ids'])])
    async def evict(self,session,plan):
        ack=dict(plan_digest=plan.digest,resident_tokens=plan.resident_after,evicted_tokens=plan.removed_tokens,
                 logical_tokens=plan.logical_tokens,eviction_revision=session.eviction_revision+1)
        self.evictions.append(ack);return ack
    async def close(self,session):self.closed.append(session.session_id)


class LocalClient(HistoryClient):
    def __init__(self,*args,manager,loop,**kwargs):
        super().__init__(*args,**kwargs);self.manager=manager;self.loop=loop;self.corrupt=False
    def call(self,method,route,payload=None):
        self.index+=1
        try:
            if route.endswith('/status'):return self.manager.status(self.sid)
            if route.endswith('/start'):task=self.manager.start(copy.deepcopy(payload))
            elif route.endswith('/append'):task=self.manager.append(self.sid,copy.deepcopy(payload))
            elif route.endswith('/close'):task=self.manager.close(self.sid)
            else:raise AssertionError(route)
            response=self.loop.run_until_complete(task)
        except SessionError as e:
            write_json(self.evidence/f'{self.index:03d}-response.json',dict(status=e.status,
                       body=json.dumps(dict(error=dict(message=str(e),history_eviction=e.receipt)))))
            raise HTTPFailure(str(e)) from e
        if self.corrupt:
            response=copy.deepcopy(response);response['persistent']['history_eviction']['history_removed_tokens']+=1
        return response


class RawHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory();self.root=Path(self.temp.name)
        self.loop=asyncio.new_event_loop();self.backend=Backend()
        self.manager=SessionManager(self.backend,max_context=40000,max_sessions=2,audit_path=str(self.root/'service.jsonl'))
        self.engine=self.root/'engine.jsonl';self.engine.write_text('')
        self.client=LocalClient('http://127.0.0.1:1234',self.root,90,'test',deadline=time.monotonic()+60,
                                manager=self.manager,loop=self.loop)
    def tearDown(self):
        self.loop.run_until_complete(self.manager.close_all());self.loop.close();self.temp.cleanup()

    def panel(self):
        c=self.client
        c.submit('one',list(range(3000)),observation=('obs1',128),suffix=[1000,1257])
        c.submit('inspect',list(range(17)))
        c.submit('two',list(range(2500)),observation=('obs2',5),suffix=[800,1063])
        c.submit('three',list(range(2500)),observation=('obs3',5),suffix=[900,1169])
        c.submit('four',list(range(3000)),observation=('obs4',5))
        c.submit('fill',list(range(7000-len(c.oracle.resident_positions)-8)))
        c.submit('continue',[])
        return c

    def test_full_panel_receipts_match_independent_oracle(self):
        c=self.panel()
        self.assertEqual([t['resident_tokens'] for t in c.turns],[3008,3033,5284,4876,5370,7000,4769])
        self.assertEqual([t['history_eviction']['history_removed_tokens'] for t in c.turns],[0,0,0,2653,2245,0,2239])
        self.assertEqual(c.turns[-1]['history_eviction']['selected_block_ids'],['obs3'])
        self.assertEqual(c.turns[-1]['expected_prefill_tokens'],1)
        self.assertEqual(c.turns[-1]['logical_tokens'],12695)
        self.assertEqual(c.turns[-1]['history_state']['active_observation_id'],'obs4')
        self.assertEqual(self.backend.submissions[-1][1],[])
        self.assertEqual(len(self.backend.evictions),4)
        self.assertTrue(all(p in c.oracle.resident_positions for p in range(128)))

    def test_idempotent_append_has_no_additional_model_submission(self):
        c=self.panel();before=len(self.backend.submissions)
        self.assertTrue(c.replay_last()['status_unchanged'])
        self.assertEqual(len(self.backend.submissions),before)

    def test_active_only_rejection_is_pure_and_empty_continuation_works(self):
        c=self.client;c.oracle=Oracle(1024)
        c.submit('active',list(range(900)),observation=('active',128))
        result=reject_active_capacity(c,list(range(200)),self.engine)
        self.assertTrue(result['paused_state_unchanged'])
        self.assertEqual(result['receipt']['eligible_history_tokens'],0)
        c.submit('continue',[])
        self.assertEqual(c.turns[-1]['resident_tokens'],916)
        self.assertEqual(c.turns[-1]['history_state']['active_observation_id'],'active')
        self.assertEqual(len(self.backend.evictions),0)

    def test_bad_policy_receipt_refused(self):
        self.client.corrupt=True
        with self.assertRaisesRegex(ContractFailure,'receipt mismatch'):
            self.client.submit('active',list(range(900)),observation=('active',128))

    def test_new_observation_cannot_make_active_eligible_in_same_plan(self):
        x=Oracle(1024);p=x.prepare(list(range(900)),observation=('a',128));x.commit(p,[1]*8)
        before=copy.deepcopy(x.__dict__)
        with self.assertRaises(OracleCapacity):x.prepare([1]*200,observation=('b',4))
        self.assertEqual(x.__dict__,before)

    def test_intervals_preserve_holes_and_deduplicate(self):
        self.assertEqual(intervals([8,1,2,2,4,5,9]),[[1,3],[4,6],[8,10]])

    def test_audit_fails_on_missing_engine_evidence(self):
        c=self.panel();c.close(c.sid)
        result=audit_events(self.engine,self.root/'service.jsonl',c.sid,c.turns)
        self.assertFalse(result['valid'])
        self.assertIn('single_engine_owner',[r['check'] for r in result['errors']])

    def synthetic_engine_journal(self):
        """Attach explicit synthetic worker proofs to actual CPU service rows."""
        c=self.panel();c.close(c.sid)
        service=[json.loads(x) for x in (self.root/'service.jsonl').read_text().splitlines()]
        rid=service[0]['engine_request_id'];engine=[];blocks=[[1,2,3,4,5],[6]]
        for t in c.turns:
            i=t['turn']
            if t['removed_tokens']:
                sc=next(r for r in service if r['event']=='evict_commit' and r['operation_id']==t['request_id'])
                ack=sc['acknowledgement'];freed=[[blocks[0].pop()],[]]
                planned=[dict(layer_name=f'{kind}-{j}',kind=kind,retained_rows=32) for kind in ('main','compressed') for j in range(12)]
                proof=dict(sha256='a'*64,rows=32,bytes=1024)
                copies=[dict(layer_name=r['layer_name'],copied_rows=32,retained_rows=32,row_bytes=32,
                             byte_verification=dict(before=proof,after=copy.deepcopy(proof),equal=True)) for r in planned]
                ack.update(freed_blocks_by_group=freed,worker_receipts=[dict(worker_synchronized=True,byte_verification_enabled=True,copies=copies)],
                    allocator_reacquisition=dict(kind='actual_allocator_reacquisition_without_forward',attempted=True,
                        released_block_ids=freed[0],reacquired_released_block_ids=freed[0],reacquired_ref_cnt=1,
                        free_count_before=10,free_count_after=10,free_order_before_sha256='b'*64,free_order_after_sha256='b'*64,
                        queue_order_restored=True,other_live_owners_unchanged=True,all_temporary_owners_released=True,model_forward_performed=False))
                engine.append(dict(event='evict_prepare',request_id=rid,eviction_plan=dict(plan_digest=ack['plan_digest'],groups=[dict(copies=planned)])))
                engine.append(dict(event='evict',request_id=rid,eviction=ack,resident_tokens=t['resident_after_eviction'],
                                   tokens_sha256=t['after_eviction_sha256'],computed_tokens=t['resident_after_eviction']-1,block_ids=copy.deepcopy(blocks)))
            engine.append(dict(event='start' if i==0 else 'resume',request_id=rid,turn=i,computed_tokens=t['expected_resume_computed'],block_ids=copy.deepcopy(blocks)))
            engine.append(dict(event='pause',request_id=rid,turn=i,block_ids=copy.deepcopy(blocks),resident_tokens=t['resident_tokens'],
                tokens_sha256=t['resident_sha256'],computed_tokens=t['resident_tokens']-1,computed_prefix_sha256=t['computed_prefix_sha256'],
                in_flight_tokens=0,output_placeholders=0,preemptions=0,logical_eviction_state=dict(logical_sha256=t['logical_sha256']),
                turn_work=dict(prefill_tokens=t['expected_prefill_tokens'],decode_tokens=7,scheduled_tokens=t['expected_prefill_tokens']+7,
                               resume_computed_tokens=t['expected_resume_computed'],minimum_scheduled_position=t['expected_resume_computed'])))
        engine.append(dict(event='close_released',request_id=rid,block_ids=[[],[]],request_still_registered=False))
        self.engine.write_text(''.join(json.dumps(r)+'\n' for r in engine))
        (self.root/'service.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in service))
        return c,engine,service

    def test_synthetic_owner_byte_and_allocator_audit(self):
        c,_,_=self.synthetic_engine_journal()
        result=audit_events(self.engine,self.root/'service.jsonl',c.sid,c.turns)
        self.assertTrue(result['valid'],result['errors'])
        self.assertTrue(result['byte_hashes_observed'])
        self.assertTrue(result['physical_page_release_observed'])

    def test_missing_byte_proof_fails_even_if_enable_flag_true(self):
        c,engine,service=self.synthetic_engine_journal()
        ee=next(r for r in engine if r['event']=='evict')
        del ee['eviction']['worker_receipts'][0]['copies'][0]['byte_verification']
        self.engine.write_text(''.join(json.dumps(r)+'\n' for r in engine))
        # Matching acknowledgements alone cannot establish byte preservation.
        (self.root/'service.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in service))
        result=audit_events(self.engine,self.root/'service.jsonl',c.sid,c.turns)
        self.assertIn('retained_byte_hash_equal',[r['check'] for r in result['errors']])

    def test_allocator_wrong_freed_owner_ids_fail(self):
        c,engine,service=self.synthetic_engine_journal()
        ee=next(r for r in engine if r['event']=='evict')
        ee['eviction']['allocator_reacquisition']['reacquired_released_block_ids']=[9999]
        self.engine.write_text(''.join(json.dumps(r)+'\n' for r in engine))
        (self.root/'service.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in service))
        result=audit_events(self.engine,self.root/'service.jsonl',c.sid,c.turns)
        self.assertIn('freed_pages_reacquired_and_pool_restored_without_forward',[r['check'] for r in result['errors']])

    def test_partial_population_hash_is_not_enough(self):
        c,engine,service=self.synthetic_engine_journal()
        ee=next(r for r in engine if r['event']=='evict')
        proof=ee['eviction']['worker_receipts'][0]['copies'][0]['byte_verification']
        proof['before']=proof['after']=dict(sha256='a'*64,rows=16,bytes=512)
        self.engine.write_text(''.join(json.dumps(r)+'\n' for r in engine))
        (self.root/'service.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in service))
        result=audit_events(self.engine,self.root/'service.jsonl',c.sid,c.turns)
        self.assertIn('byte_proof_covers_all_planned_retained_rows',[r['check'] for r in result['errors']])


if __name__=='__main__':unittest.main()
