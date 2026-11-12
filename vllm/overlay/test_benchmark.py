import asyncio
import copy
import io
from pathlib import Path
import sys
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import benchmark as b
import analyze as a

# Compare the client oracle with the separately implemented, frozen production
# policy, not a second transcription of the client's arithmetic.
FROZEN=Path(__file__).resolve().parent/'frozen-v5-proof'
sys.path.insert(0,str(FROZEN))
from history_eviction import HistoryRegistry,HistoryPolicy
from residency import ResidentLedger


class PolicyBackend:
    def __init__(self):
        self.registry=HistoryRegistry();self.residency=ResidentLedger();self.ids=[];self.revision=0;self.evictions=0
    def respond(self,body):
        raw=body['raw'];delta=raw['token_ids'];count=body['sampling']['max_tokens'];old=len(self.ids)
        if self.revision==0:self.policy=HistoryPolicy.from_payload(body['history_eviction'],40000)
        registration=self.registry.plan_registration(raw['observation_starts'],logical_offset=old,delta_tokens=len(delta))
        decision=self.registry.plan_eviction(self.policy,self.residency,incoming_tokens=len(delta))
        self.residency.commit(decision.eviction_plan,acknowledged_digest=decision.eviction_plan.digest)
        self.registry.commit(registration)
        output=[31+self.revision]*count
        self.ids.extend(delta+output);self.residency.append(len(delta)+count)
        self.revision+=1;self.evictions+=bool(decision.eviction_plan.removed_tokens)
        resident=self.residency.resident_tokens
        response=dict(persistent=dict(session_id=body.get('session_id',self.sid),revision=self.revision,
            resident_tokens=resident,logical_tokens=len(self.ids),logical_input_tokens=old+len(delta),
            appended_prompt_tokens=len(delta),prefix_sha256=b.digest(self.ids),
            evicted_tokens=decision.eviction_plan.removed_tokens,eviction_revision=self.evictions,
            prefix_caching=False,eviction=True,engine_finish_reason='length',token_ids=output,
            history_eviction=decision.receipt),choices=[dict(token_ids=output)],
            usage=dict(prompt_tokens=resident-count,completion_tokens=count,total_tokens=resident))
        return response


def sample(stamp,*,generated=None,prompt=0,running=25,waiting=0,preemptions=0):
    return dict(seconds=stamp,rtt_seconds=.01,metrics=dict(num_requests_running=running,num_requests_waiting=waiting,
        num_preemptions_total=preemptions,prompt_tokens_total=prompt,
        generation_tokens_total=stamp*1000 if generated is None else generated,kv_cache_usage_perc=.8))


def audit_fixture():
    """One resumed turn after eviction; independent explicit event schema."""
    sid='owned';rid='arc3-persistent-owned-aabbccdd';op='operation-1';delta=[19];output=[20,21]
    turn=dict(session_id=sid,turn=1,request_id=op,removed_tokens=4000,
        resident_tokens=36002,resident_input_tokens=36000,resident_sha256='resident',
        computed_prefix_sha256='computed',expected_prefill_tokens=2,output_tokens=2,
        expected_resume_computed=35998,logical_sha256='logical-after',logical_before_sha256='logical-before',
        logical_before=39999,delta_tokens=1,delta_sha256=b.digest(delta),output_sha256=b.digest(output))
    copies=[dict(layer_name=f'layer-{i}.{kind}') for i in range(12)
            for kind in ('main_key_cache','compressed_key_cache')]
    ack=dict(logical_tokens=39999,evicted_tokens=4000,allocator_reacquisition=None,
             worker_receipts=[dict(worker_synchronized=True,byte_verification_enabled=False,copies=copies)])
    engine=[dict(event='resume',request_id=rid,turn=1,computed_tokens=35998),
        dict(event='pause',request_id=rid,turn=1,resident_tokens=36002,tokens_sha256='resident',
            computed_prefix_sha256='computed',computed_tokens=36001,in_flight_tokens=0,output_placeholders=0,preemptions=0,
            turn_work=dict(prefill_tokens=2,decode_tokens=1,scheduled_tokens=3,minimum_scheduled_position=35998,resume_computed_tokens=35998),
            logical_eviction_state=dict(logical_sha256='logical-after')),
        dict(event='evict',request_id=rid,turn=1,eviction=ack),
        dict(event='close_released',request_id=rid,block_ids=[[],[]],request_still_registered=False)]
    service=[dict(event='submit',session_id=sid,operation_id=op,prefix_sha256='logical-before',
                  input_tokens=36000,logical_input_tokens=40000,appended_token_ids=delta),
             dict(event='pause',session_id=sid,operation_id=op,output_token_ids=output),
             dict(event='evict_commit',session_id=sid,operation_id=op,acknowledgement=copy.deepcopy(ack))]
    return engine,service,[turn],[sid]


class BenchmarkTests(unittest.TestCase):
    def call(self,ledger,backend,n,count,observation=None):
        prepared=ledger.prepare([17]*n,count,observation)
        backend.sid=ledger.sid
        response=backend.respond(prepared['body'])
        operation=ledger.accept(prepared,response)
        self.assertEqual(ledger.logical,backend.ids)
        self.assertEqual(len(ledger.logical)-len(ledger.holes),backend.residency.resident_tokens)
        self.assertEqual(ledger.holes,set(range(len(ledger.logical)))-set(backend.residency.resident_positions()))
        return operation

    def test_three_and_six_cycles_against_frozen_policy(self):
        ledger=b.Ledger('owned',0,True);backend=PolicyBackend();operations=[]
        for index,(n,count,start) in enumerate(b.seed_steps('eviction')):
            operations.append(self.call(ledger,backend,n,count,(f'obs-{index:02d}',start)))
        self.assertEqual(len(ledger.logical),39999)
        self.assertEqual(sum(o['delta_tokens'] for o in operations),39989)
        self.assertEqual(sum(o['expected_prefill_tokens'] for o in operations),39998)
        self.assertEqual(len(ledger.observations),10)
        for cycle in range(6):
            op=self.call(ledger,backend,1,3999,(f'cycle-{cycle:02d}',0))
            self.assertEqual(op['removed_tokens'],4000)
            self.assertEqual(op['history_eviction']['selected_block_ids'],[f'obs-{cycle:02d}'])
            self.assertEqual((op['resident_input_tokens'],op['resident_tokens']),(36000,39999))
            self.assertEqual(op['expected_prefill_tokens'],2)
            self.assertEqual(op['expected_resume_computed'],35998)
            self.assertEqual(op['logical_tokens'],39999+4000*(cycle+1))
        self.assertEqual(b.plan('eviction',25,3)['maximum_output_tokens'],300175)

    def test_matched_no_eviction_and_api_pause_controls(self):
        for arm,counts in [('persistent',[3999]),('persistent-api',[999,1000,1000,1000])]:
            ledger=b.Ledger('owned',0);backend=PolicyBackend()
            seed=self.call(ledger,backend,35999,1)
            self.assertEqual(seed['resident_tokens'],36000)
            outputs=[]
            for count in counts:
                op=self.call(ledger,backend,0,count);outputs.append(op)
                self.assertEqual(op['expected_prefill_tokens'],1)
                self.assertEqual(op['removed_tokens'],0)
            self.assertEqual(outputs[-1]['resident_tokens'],39999)
            self.assertEqual(sum(o['output_tokens'] for o in outputs),3999)

    def test_bad_receipt_cannot_advance_client_state(self):
        ledger=b.Ledger('owned',0);backend=PolicyBackend();backend.sid=ledger.sid
        prepared=ledger.prepare([10]*100,1);response=backend.respond(prepared['body'])
        response['persistent']['resident_tokens']+=1
        with self.assertRaisesRegex(b.BenchmarkFailure,'resident_tokens'):
            ledger.accept(prepared,response)
        self.assertEqual(ledger.logical,[]);self.assertEqual(ledger.turn,0)

    def test_full_active_only_block_does_not_silently_evict(self):
        ledger=b.Ledger('owned',0,True);backend=PolicyBackend()
        self.call(ledger,backend,39998,1,('active',128))
        before=copy.deepcopy(ledger)
        with self.assertRaisesRegex(b.BenchmarkFailure,'Insufficient'):
            ledger.prepare([17],3999,('next',0))
        self.assertEqual(ledger,before)

    def test_pure_window_excludes_delayed_resume_guard(self):
        rows=[sample(i) for i in range(41)]
        operation=dict(phase='measured',session_id='owned',turn=1,start_seconds=5,end_seconds=40)
        engine=[dict(event='resume',request_id='arc3-persistent-owned-aabbccdd',turn=1,time=111)]
        guards=a.boundary_guards(engine,[operation],epoch_origin=100)
        self.assertEqual(guards,[(3,13)])
        result=a.pure_decode_windows(rows,concurrency=25,start=0,end=40,boundary_guards=guards)
        self.assertEqual(result['pooled_seconds'],27)
        self.assertEqual(result['tokens_per_second'],1000)
        self.assertEqual(result['windows'][0]['start_seconds'],13)

    def test_pure_window_rejects_prefill_and_partial_occupancy(self):
        rows=[sample(i,prompt=int(i>=10),running=24 if i==25 else 25) for i in range(41)]
        result=a.pure_decode_windows(rows,concurrency=25,start=0,end=40)
        self.assertFalse(result['valid'])

    def test_pure_window_sorts_concurrently_returned_samples(self):
        rows=[sample(i) for i in range(41)]
        rows[10],rows[11]=rows[11],rows[10]
        result=a.pure_decode_windows(rows,concurrency=25,start=0,end=40)
        self.assertEqual(result['pooled_seconds'],40)
        self.assertEqual(result['pooled_generated_tokens'],40000)

    def test_persistent_prompt_counter_semantics_and_inclusive_pause_time(self):
        before=sample(0)['metrics'];after=sample(100,generated=10000)['metrics']
        result=dict(config=dict(arm='eviction',concurrency=25),
            operations=[dict(phase='measured',output_tokens=10000,expected_prefill_tokens=50)],
            measurement=dict(start_seconds=0,end_seconds=100,metrics_before=before,metrics_after=after))
        value=a.summarize(result,[])
        self.assertTrue(value['valid']);self.assertEqual(value['received_tokens_per_second'],100)
        self.assertEqual(value['expected_engine_prompt_counter_delta'],0)
        self.assertEqual(value['expected_scheduled_prefill_tokens'],50)
        self.assertFalse(value['prompt_counter_matches_schedule'])
        result['config']['arm']='stateless'
        self.assertFalse(a.summarize(result,[])['valid'])

    def test_metric_token_counts_are_not_text_lengths(self):
        raw='\n'.join(f'vllm:{key}{{model_name="x"}} {i}' for i,key in enumerate(a.METRICS))
        values=a.metric_values(raw)
        self.assertEqual(values['generation_tokens_total'],4)
        with self.assertRaises(ValueError):a.metric_values('vllm:generation_tokens_total 42')
        self.assertEqual(a.count_at([[1,1],[2,4]],1.5),1)

    def test_dry_run_does_not_enter_execution(self):
        with patch.object(b,'verify_identity',side_effect=AssertionError('should not execute')):
            with redirect_stdout(io.StringIO()):self.assertEqual(b.main(['--arm','eviction']),0)

    def test_proof_off_audit_still_checks_work_and_receipt_identity(self):
        args=audit_fixture()
        self.assertTrue(a.audit_owned(*args,expected_proofs=False)['valid'])
        for mutation,key in [
            (lambda e,s:e[1]['turn_work'].update(prefill_tokens=36000),'no_historical_prefill'),
            (lambda e,s:e[2]['eviction'].update(allocator_reacquisition={'unexpected':True}),'allocator_proof_disabled'),
            (lambda e,s:e[2]['eviction']['worker_receipts'][0].update(byte_verification_enabled=True),'correct_proof_profile'),
            (lambda e,s:s[2]['acknowledgement'].update(evicted_tokens=3999),'service_commits_exact_engine_ack'),
            (lambda e,s:e[2]['eviction']['worker_receipts'][0]['copies'].pop(),'all_cache_owners_acknowledged'),
            (lambda e,s:e[3].update(block_ids=[[17]]),'all_owned_pages_released')]:
            with self.subTest(key=key):
                e,s,ops,sids=copy.deepcopy(args);mutation(e,s)
                report=a.audit_owned(e,s,ops,sids,expected_proofs=False)
                self.assertFalse(report['valid'])
                self.assertIn(key,{row['check'] for row in report['failures']})

    def test_audit_does_not_confuse_neighboring_session_names(self):
        e,s,ops,sids=audit_fixture()
        e.append(dict(event='pause',request_id='arc3-persistent-owned-other-aabbccdd',preemptions=99))
        s.append(dict(event='failed',session_id='owned-other'))
        self.assertTrue(a.audit_owned(e,s,ops,sids,expected_proofs=False)['valid'])


class AsyncFailureTests(unittest.IsolatedAsyncioTestCase):
    async def test_sibling_failure_cancels_and_joins_all_request_tasks(self):
        started=asyncio.Event();cleaned=[]
        async def sibling():
            started.set()
            try:await asyncio.sleep(60)
            finally:cleaned.append(True)
        async def failing():
            await started.wait();raise RuntimeError('failure')
        with self.assertRaisesRegex(RuntimeError,'failure'):
            await b.join_owned([sibling(),failing()])
        self.assertEqual(cleaned,[True])


if __name__=='__main__':unittest.main()
