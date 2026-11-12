"""Synthetic receipt tests; no vLLM, remote work, or recorded code execution."""
import copy
import unittest

from audit_gameplay_eviction import audit, digest, png_identity
from history_eviction import HistoryPolicy,HistoryRegistry
from residency import ResidentLedger


def fixture(history=False, fraction=.10, history_turns=6):
    sid, rid = "synthetic", "arc3-persistent-synthetic-deadbeef"
    engine, service, wire = [], [], []
    logical, positions = [], []
    blocks = [[1, 2], [3]]
    revision = evrev = 0
    policy=HistoryPolicy('oldest_interactions',fraction,20)
    registry=HistoryRegistry();ledger=ResidentLedger()
    content1 = [{"type": "text", "text": "Current board"},
                {"type": "text", "text": "Saved programs"}]
    content2 = [{"type": "text", "text": "Next board"}]
    desc1 = dict(message_index=1, observation_id="obs1", content_sha256=digest(content1),
                 base_parts_end=1, program_parts=[1, 2], snapshot_parts=None)
    desc2 = dict(message_index=0, observation_id="obs2", content_sha256=digest(content2),
                 base_parts_end=1, program_parts=None, snapshot_parts=None)
    for i in range(history_turns if history else 3):
        op = f"{sid}-r{i+1}"
        initial = i == 0
        oldlogical = list(logical)
        desc = [desc1] if i == 0 else [dict(desc2,observation_id=f'obs{i}')] if i >= 2 else []
        retire = ["obs1"] if i == 2 else []
        records = [dict(d, local_token_range=[3, 6] if i == 0 else None, image_spans=[]) for d in desc]
        delta = list(range(i*100, i*100+(10 if i == 0 else 3)))
        output = [1000+i*2, 1001+i*2]
        msgs = ([{"role": "system", "content": "Solve the game"}, {"role": "user", "content": content1}]
                if initial else [{"role": "user", "content": content2}] if i >= 2
                else [{"role": "tool", "tool_call_id": "call1", "content": "Result"}])
        side = dict(version=1, observations=desc, evict_observation_ids=retire)
        body = dict(request_id=op, managed_suffixes=side)
        if initial:
            body.update(session_id=sid, chat=dict(messages=msgs))
            if history:body['history_eviction']=dict(policy=policy.policy,fraction=fraction,resident_token_budget=20)
        else:
            body.update(expected_revision=revision, expected_prefix_sha256=digest(logical),
                        external_messages=msgs, continuation=False)
        wire.append(dict(kind="persistent-request", payload=dict(route="/arc3/persistent/"+("start" if initial else sid+"/append"), body=body)))
        def er(event, **kwargs):
            row = dict(event=event, request_id=rid, turn=i, preemptions=0,
                       block_ids=copy.deepcopy(blocks), in_flight_tokens=0, output_placeholders=0,
                       resident_tokens=len(positions), tokens_sha256=digest([logical[p] for p in positions]),
                       computed_tokens=max(0, len(positions)-1),
                       computed_prefix_sha256=digest([logical[p] for p in positions[:-1]]))
            row.update(kwargs)
            engine.append(row)
            return row
        starts=[]
        if history:
            for d in desc:
                local=2 if initial else 0
                starts.append(dict(observation_id=d['observation_id'],local_start_token=local,
                    proof=dict(method='paired_no_generation_empty_user_probe_v1' if initial else 'paired_no_generation_preceding_messages_v1',
                        preceding_token_count=local,prefix_sha256=digest(delta[:local]),opening_token_ids=delta[local:local+1],
                        includes_generation_prompt=False)))
            registration=registry.plan_registration(starts,logical_offset=len(logical),delta_tokens=len(delta))
            decision=registry.plan_eviction(policy,ledger,pending_suffix_ranges=[[3,6]] if i==2 else [],incoming_tokens=len(delta))
        if (history and decision.eviction_plan.removed_tokens) or (not history and i == 2):
            before = len(positions)
            if history:
                plan=decision.eviction_plan.wire()
                positions=[p for p in positions if not any(a<=p<b for a,b in plan['ranges'])]
                ledger.commit(decision.eviction_plan,acknowledged_digest=plan['digest'])
                service.append(dict(event='evict_submit',session_id=sid,engine_request_id=rid,operation_id=op,
                                    plan=plan,history_eviction=decision.receipt,history_state=registry.snapshot()))
            else:
                positions = [p for p in positions if not 3 <= p < 6]
                plan = dict(ranges=[[3, 6]], resident_before=before, resident_after=len(positions), removed_tokens=3, digest="d"*64)
            proof = dict(sha256="b"*64, bytes=64, rows=8)
            evrev+=1
            ack = dict(plan_digest=plan["digest"], evicted_tokens=plan['removed_tokens'], resident_tokens=len(positions), logical_tokens=len(logical),
                       eviction_revision=evrev, freed_blocks_by_group=[[2] if i==2 else [], []], worker_receipts=[dict(worker_synchronized=True,
                         byte_verification_enabled=True, copies=[dict(copied_rows=8, row_bytes=8,
                             byte_verification=dict(before=proof, after=copy.deepcopy(proof), equal=True))])])
            blocks = [[1], [3]]
            er("evict", eviction=ack)
            service.append(dict(event="evict_commit", session_id=sid, engine_request_id=rid, operation_id=op,
                                plan=plan, acknowledgement=ack,**({'history_eviction':decision.receipt} if history else {})))
        before_append = len(positions)
        computed = before_append-1 if i else 0
        positions.extend(range(len(logical),len(logical)+len(delta)))
        logical.extend(delta)
        if history:
            registry.commit(registration)
            ledger.append(len(delta))
        ownership = dict(registered_observation_ids=[d["observation_id"] for d in desc], evicted_observation_ids=retire)
        sub = dict(event="submit", session_id=sid, engine_request_id=rid, operation_id=op, initial=initial,
                   revision=revision, appended_token_ids=delta, old_tokens=len(oldlogical),
                   logical_input_tokens=len(logical), input_tokens=len(positions), resident_before_append=before_append,
                   prefix_sha256=digest(oldlogical), suffix_records=records, managed_suffixes=ownership)
        if history:sub.update(observation_starts=starts,history_state=registry.snapshot(),history_eviction=decision.receipt)
        service.append(sub)
        er("start" if initial else "resume", computed_tokens=computed, append=dict(pending_prefill_tokens=len(delta)+1))
        input_count, input_logical = len(positions), len(logical)
        positions.extend(range(len(logical), len(logical)+len(output)))
        logical.extend(output)
        if history:ledger.append(len(output))
        revision += 1
        service.append(dict(event="pause", session_id=sid, engine_request_id=rid, operation_id=op, revision=revision,
                            resident_tokens=len(positions), logical_tokens=len(logical), output_token_ids=output,
                            prefix_sha256=digest(logical)))
        er("pause", logical_eviction_state=dict(logical_sha256=digest(logical), deleted_tokens=len(logical)-len(positions)),
           turn_work=dict(prefill_tokens=len(delta)+(not initial), decode_tokens=1,
                          scheduled_tokens=len(delta)+(not initial)+1, minimum_scheduled_position=computed,
                          resume_computed_tokens=computed))
        persistent = dict(session_id=sid, revision=revision, resident_tokens=len(positions), logical_tokens=len(logical),
                          logical_input_tokens=input_logical, appended_prompt_tokens=len(delta), prefix_sha256=digest(logical),
                          prefix_caching=False, eviction=True, eviction_revision=evrev,
                          evicted_tokens=decision.eviction_plan.removed_tokens if history else 3 if i == 2 else 0,
                          managed_suffixes=ownership, token_ids=output)
        if history:persistent.update(history_eviction=decision.receipt,history_state=registry.snapshot())
        message = dict(role="assistant", content="ok")
        if i == 0:
            message["tool_calls"] = [dict(id="call1")]
        wire.append(dict(kind="persistent-response", payload=dict(id=op, choices=[dict(message=message, token_ids=output,
                             finish_reason="tool_calls" if i == 0 else "stop")], persistent=persistent,
                             usage=dict(prompt_tokens=input_count, completion_tokens=len(output), total_tokens=input_count+len(output)))))
    service.append(dict(event="close", session_id=sid, engine_request_id=rid, revision=revision, resident_tokens=len(positions)))
    engine.append(dict(event="close_released", request_id=rid, preemptions=0, block_ids=[[], []], request_still_registered=False))
    wire.append(dict(kind="persistent-close", payload=dict(session_id=sid, state="closed")))
    return engine, service, wire


class AuditTests(unittest.TestCase):
    def test_whole_interaction_evictions_skip_existing_holes(self):
        e,s,w=fixture(history=True)
        r=audit(e,s,w)
        self.assertEqual(r['status'],'pass',r['failures'])
        self.assertEqual(r['totals']['history_evictions'],2)
        self.assertEqual(r['totals']['history_removed_tokens'],17)
        self.assertEqual(r['gates']['whole_interaction_eviction'],'observed')
        decisions=[row['history_eviction'] for row in s if row['event']=='submit' and row['history_eviction']['history_removed_tokens']]
        self.assertEqual(decisions[0]['history_ranges'],[[2,3],[6,17]])
        self.assertEqual(decisions[1]['skipped_zero_resident_block_ids'],['obs1'])

    def test_round_up_can_require_multiple_complete_blocks(self):
        e,s,w=fixture(history=True,fraction=.5)
        r=audit(e,s,w)
        self.assertEqual(r['status'],'pass',r['failures'])
        decisions=[row['history_eviction'] for row in s if row['event']=='submit' and row['history_eviction']['history_removed_tokens']]
        self.assertEqual(decisions[1]['selected_block_ids'],['obs2','obs3'])
        self.assertEqual((decisions[1]['target_tokens'],decisions[1]['history_removed_tokens']),(9,10))

    def test_enabled_but_unobserved_history_is_reported_honestly(self):
        result=audit(*fixture(history=True,history_turns=3))
        self.assertEqual(result['status'],'pass',result['failures'])
        self.assertEqual(result['gates']['whole_interaction_eviction'],'not_observed')
        self.assertEqual(result['totals']['history_removed_tokens'],0)

    def test_history_target_cannot_be_rounded_down(self):
        e,s,w=fixture(history=True)
        decision=next(row['history_eviction'] for row in s if row['event']=='submit' and row['history_eviction']['history_removed_tokens'])
        decision['target_tokens']-=1
        self.assertIn('history_decision_target_tokens',audit(e,s,w)['failed_checks'])

    def test_history_cannot_select_more_recent_or_active_block(self):
        e,s,w=fixture(history=True)
        decision=next(row['history_eviction'] for row in s if row['event']=='submit' and row['history_eviction']['history_removed_tokens'])
        decision['selected_block_ids']=['obs2']
        self.assertIn('history_decision_selected_block_ids',audit(e,s,w)['failed_checks'])

    def test_boundary_proof_and_registry_must_match_actual_delta(self):
        e,s,w=fixture(history=True)
        sub=next(row for row in s if row['event']=='submit')
        sub['observation_starts'][0]['local_start_token']+=1
        failed=audit(e,s,w)['failed_checks']
        self.assertIn('history_paired_prefix_exact_ids',failed)
        self.assertIn('history_registry_exact_committed_blocks',failed)

    def test_existing_holes_cannot_be_deleted_twice(self):
        e,s,w=fixture(history=True)
        decision=next(row['history_eviction'] for row in s if row['event']=='submit' and row['history_eviction']['history_removed_tokens'])
        decision['history_ranges']=[[2,17]]
        self.assertIn('history_decision_history_ranges',audit(e,s,w)['failed_checks'])

    def test_system_prefix_cannot_be_declared_unprotected(self):
        e,s,w=fixture(history=True)
        decision=next(row['history_eviction'] for row in s if row['event']=='submit' and row['history_eviction']['history_removed_tokens'])
        decision['protected_prefix_tokens']=0
        self.assertIn('history_decision_protected_prefix_tokens',audit(e,s,w)['failed_checks'])

    def test_incoming_observation_does_not_close_active_block_early(self):
        e,s,w=fixture(history=True)
        pre=next(row for row in s if row['event']=='evict_submit' and row['history_eviction']['history_removed_tokens'])
        pre['history_state']['blocks'][-1]['end']=999
        self.assertIn('history_selection_uses_preappend_closed_blocks',audit(e,s,w)['failed_checks'])

    def test_history_plan_keeps_exact_prior_holes_and_full_stream(self):
        e,s,w=fixture(history=True)
        plan=next(row['plan'] for row in s if row['event']=='evict_commit' and row['history_eviction']['history_removed_tokens'])
        plan['prior_evicted']=[]
        self.assertIn('history_plan_prior_holes_exact',audit(e,s,w)['failed_checks'])

    def test_policy_cannot_be_changed_on_append(self):
        e,s,w=fixture(history=True)
        req=[row['payload']['body'] for row in w if row['kind']=='persistent-request'][1]
        req['history_eviction']={'policy':'off','fraction':.10}
        self.assertIn('history_policy_only_at_start',audit(e,s,w)['failed_checks'])

    def test_explicit_uncommitted_capacity_stop_is_separate_gate(self):
        e,s,w=fixture()
        op='synthetic-r4'
        receipt=dict(policy='off',fraction=.1,resident_token_budget=40000,triggered=True,
                     eviction_committed=False,capacity_fallback='history_eviction_disabled',
                     resident_tokens_before=19,history_removed_tokens=0)
        s.insert(-1,dict(event='rejected',session_id='synthetic',engine_request_id='arc3-persistent-synthetic-deadbeef',
                         revision=3,operation_id=op,error='SessionError: Context capacity',history_eviction=receipt))
        w.insert(-1,dict(kind='persistent-request',payload=dict(route='/arc3/persistent/synthetic/append',body=dict(
            request_id=op,external_messages=[{'role':'user','content':'Large incoming observation'}]))))
        w.insert(-1,dict(kind='persistent-error',payload=dict(request_id=op,session_id='synthetic',status_code=413,
            uncommitted_capacity_refusal=True,body={'error':{'type':'persistent_session_error','history_eviction':receipt}})))
        result=audit(e,s,w)
        self.assertEqual(result['status'],'pass',result['failures'])
        self.assertEqual(result['totals']['safe_capacity_refusals'],1)
        self.assertEqual(result['gates']['whole_interaction_eviction'],'not_observed')

    def test_positive_fixture(self):
        result = audit(*fixture())
        self.assertEqual(result["status"], "pass", result["failures"])
        self.assertEqual(result["totals"]["committed_evictions"], 1)
        self.assertEqual(result["totals"]["tool_only_appends"], 1)
        self.assertEqual(result["totals"]["freed_pages"], 1)

    def test_detects_historical_refill(self):
        e,s,w = fixture()
        next(r for r in e if r["event"] == "pause" and r["turn"] == 2)["turn_work"]["prefill_tokens"] += 3
        r = audit(e,s,w)
        self.assertIn("no_historical_refill_prefill_tokens", r["failed_checks"])

    def test_detects_mutated_raw_ids(self):
        e,s,w = fixture()
        response = next(r["payload"] for r in w if r["kind"] == "persistent-response")
        response["choices"][0]["token_ids"] = [9,9]
        self.assertIn("wire_choice_exact_emitted_ids", audit(e,s,w)["failed_checks"])

    def test_detects_suffix_refresh_during_tool_loop(self):
        e,s,w = fixture()
        request = [r for r in w if r["kind"] == "persistent-request"][1]
        request["payload"]["body"]["managed_suffixes"]["evict_observation_ids"] = ["bogus"]
        self.assertIn("same_observation_no_eviction", audit(e,s,w)["failed_checks"])

    def test_detects_wrong_shrink_ack(self):
        e,s,w = fixture()
        next(r for r in e if r["event"] == "evict")["eviction"]["resident_tokens"] -= 1
        self.assertIn("ack_resident_tokens", audit(e,s,w)["failed_checks"])

    def test_detects_failed_cleanup(self):
        e,s,w = fixture()
        e[-1]["block_ids"] = [[1],[]]
        self.assertIn("close_frees_all_blocks", audit(e,s,w)["failed_checks"])

    def test_detects_byte_change(self):
        e,s,w = fixture()
        proof = next(r for r in e if r["event"] == "evict")["eviction"]["worker_receipts"][0]["copies"][0]["byte_verification"]
        proof["after"]["sha256"] = "c"*64
        self.assertIn("retained_cache_bytes_equal", audit(e,s,w)["failed_checks"])

    def test_enabled_hash_flag_without_proof_is_not_accepted(self):
        e,s,w=fixture()
        copy_record=next(r for r in e if r['event']=='evict')['eviction']['worker_receipts'][0]['copies'][0]
        del copy_record['byte_verification']
        self.assertIn('enabled_byte_verification_has_actual_proof',audit(e,s,w)['failed_checks'])

    def test_rejects_non_png_without_fetch(self):
        with self.assertRaises(ValueError):
            png_identity(dict(image_url=dict(url="https://invalid.example/image.png")))

    def test_model_only_output_never_used_as_wire(self):
        e,s,w = fixture()
        w = [r for r in w if r["kind"] != "persistent-response"]
        w.append(dict(kind="model-turn", payload=dict(content="looks fine")))
        self.assertIn("completed_pause_has_wire_response", audit(e,s,w)["failed_checks"])

    def test_deadline_cancel_before_engine_submission(self):
        e,s,w = fixture()
        sid, rid = "synthetic", "arc3-persistent-synthetic-deadbeef"
        final = next(r for r in w if r["kind"] == "persistent-response" and r["payload"]["id"] == sid+"-r3")["payload"]["persistent"]
        op = sid+"-r4"
        s.insert(-1, dict(event="submit", session_id=sid, engine_request_id=rid, operation_id=op, initial=False,
                         revision=3, appended_token_ids=[], old_tokens=final["logical_tokens"],
                         logical_input_tokens=final["logical_tokens"], input_tokens=final["resident_tokens"],
                         resident_before_append=final["resident_tokens"], prefix_sha256=final["prefix_sha256"],
                         suffix_records=[], managed_suffixes=dict(registered_observation_ids=[],evicted_observation_ids=[])))
        s.insert(-1,dict(event="failed", session_id=sid, engine_request_id=rid, error="CancelledError()", time=1000))
        w.insert(-1,dict(kind="persistent-request",payload=dict(route="/arc3/persistent/synthetic/append",body=dict(
            request_id=op, expected_revision=3,expected_prefix_sha256=final["prefix_sha256"],external_messages=[],continuation=True,
            managed_suffixes=dict(version=1, observations=[],evict_observation_ids=[])))))
        r=audit(e,s,w,deadline_epoch=1000)
        self.assertEqual(r["status"],"pass",r["failures"])
        self.assertEqual(r["totals"]["unreturned_final_turns"],1)
        self.assertEqual(audit(e,s,w)["status"],"fail")


if __name__ == "__main__":
    unittest.main()
