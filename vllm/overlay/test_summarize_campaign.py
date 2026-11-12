import copy
import json
from pathlib import Path
import tempfile
import unittest

import summarize_campaign as campaign


def fixture():
    operations=[];service=[];engine=[]
    for i in range(2):
        sid=f'owned-{i}';rid=f'op-{i}'
        operations.append(dict(phase='measured',session_id=sid,request_id=rid,turn=10,
            removed_tokens=4000,output_tokens=100,delta_tokens=1,expected_prefill_tokens=2,
            client_prepare_seconds=.01,client_verify_seconds=.02))
        worker=dict(byte_verification_enabled=False,worker_profile=dict(total_seconds=1.8,
            initial_fence_seconds=.1,prepare_validate_seconds=.2,copy_and_optional_proofs_seconds=1.4,mapping_publish_seconds=.1))
        ack=dict(worker_receipts=[worker],core_profile=dict(total_before_final_audit_seconds=2.8,
            prepare_seconds=.5,wire_and_prepare_audit_seconds=.2,worker_rpc_seconds=2,
            publish_release_optional_allocator_proof_seconds=.1))
        service.append(dict(event='evict_commit',session_id=sid,operation_id=rid,acknowledgement=ack,
            endpoint_profile=dict(total_seconds=10,lock_wait_seconds=6,status_seconds=.2,drain_seconds=.3,core_rpc_seconds=3,resume_seconds=.5)))
        engine.append(dict(event='pause',request_id=f'arc3-persistent-{sid}-aabbccdd',turn=10,
            turn_work=dict(prefill_tokens=2,decode_tokens=99,scheduled_tokens=101)))
    return operations,service,engine


class CampaignTests(unittest.TestCase):
    def test_nested_scopes_are_separate_and_only_measured_commits_selected(self):
        ops,service,engine=fixture()
        service.append({**copy.deepcopy(service[0]),'operation_id':'unmeasured-setup'})
        report=campaign.profiles(service,ops)
        self.assertTrue(report['valid']);self.assertEqual(report['measured_commit_count'],2)
        self.assertEqual(report['scopes']['endpoint']['phases']['total_seconds']['sum_seconds'],20)
        self.assertEqual(report['scopes']['core']['phases']['total_before_final_audit_seconds']['sum_seconds'],5.6)
        self.assertEqual(report['scopes']['worker']['phases']['total_seconds']['sum_seconds'],3.6)
        self.assertNotIn('global_stall_seconds',report)
        self.assertEqual(report['byte_proof_modes'],['False'])

    def test_duplicate_or_missing_commit_is_not_a_valid_profile_panel(self):
        ops,service,_=fixture()
        self.assertFalse(campaign.profiles(service+[service[0]],ops)['valid'])
        self.assertFalse(campaign.profiles(service[:1],ops)['valid'])

    def test_nearest_rank_percentile_and_invalid_timing(self):
        result=campaign.stats(range(1,21))
        self.assertEqual(result['p95_seconds'],19)
        self.assertEqual(result['median_seconds'],10.5)
        self.assertEqual(result['sum_seconds'],210)
        with self.assertRaises(ValueError):campaign.stats([float('nan')])
        with self.assertRaises(ValueError):campaign.stats([-1])

    def test_actual_work_uses_exact_owner_and_turn(self):
        ops,_,engine=fixture()
        engine.append({**engine[0],'request_id':'arc3-persistent-owned-0-other-aabbccdd'})
        totals=campaign.work_totals(engine,ops)
        self.assertTrue(totals['valid']);self.assertEqual(totals['actual_prefill_positions'],4)
        self.assertEqual(totals['actual_decode_positions'],198)
        self.assertEqual(totals['actual_scheduled_positions'],202)
        engine.append(engine[0])
        self.assertFalse(campaign.work_totals(engine,ops)['valid'])

    def test_proof_enabled_runs_are_not_presented_as_performance(self):
        ops,service,engine=fixture()
        before=dict(generation_tokens_total=0,prompt_tokens_total=0,num_preemptions_total=0)
        after={**before,'generation_tokens_total':200}
        result=dict(status='passed',config=dict(arm='eviction',concurrency=25,utilization=.95,model='nvidia',expected_proofs=0),
            identity=dict(persistent_experiment=dict(optimization_config=dict(verify_bytes=0,verify_allocator=0))),
            operations=ops,audit=dict(valid=True),measurement=dict(start_seconds=0,end_seconds=10,metrics_before=before,metrics_after=after))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            (root/'metrics.jsonl').write_text('')
            (root/'service-owned.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in service))
            (root/'engine-owned.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in engine))
            (root/'results.json').write_text(json.dumps(result))
            measured=campaign.load_run(root)
            self.assertTrue(measured['performance_eligible'])
            result['identity']['persistent_experiment']['optimization_config']['verify_bytes']=1
            (root/'results.json').write_text(json.dumps(result))
            proof=campaign.load_run(root)
            self.assertFalse(proof['performance_eligible'])
            text=campaign.render([proof])
            self.assertIn('timing values are retained in JSON only',text)
            self.assertNotIn('| total_seconds |',text)
            self.assertNotIn('20.00 |',text)

    def test_stateless_variant_comes_from_wire_payload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);http=root/'http';http.mkdir()
            body=dict(ignore_eos=True,min_tokens=3999)
            row=dict(method='POST',route='/v1/completions',payload=body)
            path=http/'0000.json';path.write_text(json.dumps(row))
            self.assertEqual(campaign.stateless_variant(root,1),'masked-eos-control')
            del body['min_tokens'];path.write_text(json.dumps(row))
            self.assertEqual(campaign.stateless_variant(root,1),'unmasked')
            self.assertEqual(campaign.stateless_variant(root,2),'unverified')
            body['ignore_eos']=False;path.write_text(json.dumps(row))
            self.assertEqual(campaign.stateless_variant(root,1),'unverified')


if __name__=='__main__':unittest.main()
