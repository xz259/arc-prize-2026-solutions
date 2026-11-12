"""CPU contract fixtures for the bounded live diagnostic; no network/model run."""
from __future__ import annotations

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

import raw_managed_eviction_checks as diagnostic


class SimulatedClient(diagnostic.ManagedClient):
    """Independent fake endpoint with logical/physical ledgers and replay cache."""
    def __init__(self):
        super().__init__("http://127.0.0.1:1234",Path("unused"),1,"fixture",deadline=time.monotonic()+60)
        self.server_ids=[]
        self.server_removed=set()
        self.server_live={}
        self.server_revision=0
        self.server_eviction_revision=0
        self.requests=[]
        self.replies={}
        self.outputs=[]
        self.corrupt=None

    def call(self,method,route,payload=None):
        if route.endswith("/status"):
            return {"state":"paused","revision":self.server_revision,"logical_tokens":len(self.server_ids)}
        if route.endswith("/close"):
            return {"state":"closed"}
        key=payload["request_id"]
        if key in self.replies:
            return deepcopy(self.replies[key])
        body=deepcopy(payload)
        self.requests.append(body)
        if self.server_revision:
            assert body["expected_revision"]==self.server_revision
            assert body["expected_prefix_sha256"]==diagnostic.token_digest(self.server_ids)
        raw=body["raw"]
        old=len(self.server_ids)
        removed_before=len(self.server_removed)
        for oid in raw["evict_observation_ids"]:
            start,end=self.server_live.pop(oid)
            self.server_removed.update(range(start,end))
        removed=len(self.server_removed)-removed_before
        self.server_eviction_revision+=bool(removed)
        registered=[]
        for record in raw["suffix_records"]:
            registered.append(record["observation_id"])
            if record["local_token_range"] is not None:
                begin,end=record["local_token_range"]
                self.server_live[record["observation_id"]]=(old+begin,old+end)
        self.server_ids.extend(raw["token_ids"])
        logical_input=len(self.server_ids)
        resident_input=logical_input-len(self.server_removed)
        output=[100+self.server_revision*8+j for j in range(8)]
        self.outputs.append(output)
        self.server_ids.extend(output)
        self.server_revision+=1
        state={"session_id":self.sid,"revision":self.server_revision,
               "resident_tokens":len(self.server_ids)-len(self.server_removed),
               "logical_tokens":len(self.server_ids),"logical_input_tokens":logical_input,
               "prefix_sha256":diagnostic.token_digest(self.server_ids),
               "appended_prompt_tokens":len(raw["token_ids"]),"evicted_tokens":removed,
               "eviction_revision":self.server_eviction_revision,"prefix_caching":False,"eviction":True,
               "engine_finish_reason":"length","token_ids":output,
               "managed_suffixes":{"registered_observation_ids":registered,
                   "evicted_observation_ids":raw["evict_observation_ids"]}}
        result={"persistent":state,"usage":{"prompt_tokens":resident_input,"completion_tokens":8}}
        if self.corrupt:self.corrupt(result)
        self.replies[key]=deepcopy(result)
        return result


def scenario():
    client=SimulatedClient()
    prompt=lambda n,offset=0:list(range(offset,offset+n))
    client.submit("initial",prompt(6143),suffix=[1025,4107],new_observation=True)
    client.submit("same",prompt(17,7000))
    start=128+(-(len(client.logical)+128))%4
    client.submit("aligned",prompt(4101,9000),suffix=[start,start+2784],new_observation=True)
    client.replay_last()
    start=128+(2-(len(client.logical)+128))%4
    client.submit("partial",prompt(4099,15000),suffix=[start,start+2779],new_observation=True)
    client.submit("same2",prompt(7,20000))
    client.submit("no-suffix",prompt(65,22000),new_observation=True)
    return client


def evidence(client):
    engine,service=[],[]
    owner="arc3-persistent-"+client.sid+"-deadbeef"
    main=[]
    next_block=1
    def resize(n):
        nonlocal next_block,main
        target=(n+2767)//2768
        removed=main[target:]
        main=main[:target]
        while len(main)<target:
            main.append(next_block);next_block+=1
        return removed
    def blocks():return [list(main),[900]]
    for i,turn in enumerate(client.turns):
        if turn["removed_tokens"]:
            freed=resize(turn["resident_after_eviction"])
            proof={"sha256":"a"*64,"bytes":2000,"rows":50,"chunk_row_bound":64}
            ack={"plan_digest":str(i)*64,"resident_tokens":turn["resident_after_eviction"],
                 "evicted_tokens":turn["removed_tokens"],"logical_tokens":turn["logical_before"],
                 "eviction_revision":turn["eviction_revision"],"freed_blocks_by_group":[freed,[]],
                 "worker_receipts":[{"plan_digest":str(i)*64,"request_id":owner,
                     "worker_synchronized":True,"byte_verification_enabled":True,
                     "copies":[{"layer_name":"main","copied_rows":50,"row_bytes":40,
                         "byte_verification":{"before":proof,"after":deepcopy(proof),"equal":True}}]}]}
            engine.append({"event":"evict","request_id":owner,"turn":i-1,
                "resident_tokens":turn["resident_after_eviction"],"tokens_sha256":turn["after_eviction_sha256"],
                "computed_tokens":turn["resident_after_eviction"]-1,"in_flight_tokens":0,
                "preemptions":0,"block_ids":blocks(),"eviction":ack})
            service.append({"event":"evict_commit","session_id":client.sid,
                            "operation_id":turn["request_id"],"acknowledgement":deepcopy(ack)})
        engine.append({"event":"start" if i==0 else "resume","request_id":owner,"turn":i,
            "block_ids":blocks(),"append":{"pending_prefill_tokens":turn["delta_tokens"]+1}})
        resize(turn["resident_tokens"])
        engine.append({"event":"pause","request_id":owner,"turn":i,
            "resident_tokens":turn["resident_tokens"],"tokens_sha256":turn["resident_sha256"],
            "computed_prefix_sha256":turn["computed_prefix_sha256"],"computed_tokens":turn["resident_tokens"]-1,
            "in_flight_tokens":0,"output_placeholders":0,"preemptions":0,"block_ids":blocks(),
            "logical_eviction_state":{"logical_tokens":turn["logical_tokens"],"logical_sha256":turn["logical_sha256"]},
            "turn_work":{"prefill_tokens":turn["expected_prefill_tokens"],"decode_tokens":7,
                "scheduled_tokens":turn["expected_prefill_tokens"]+7,
                "minimum_scheduled_position":turn["expected_resume_computed"],
                "resume_computed_tokens":turn["expected_resume_computed"]}})
        service.append({"event":"submit","session_id":client.sid,"operation_id":turn["request_id"],
            "appended_token_ids":client.requests[i]["raw"]["token_ids"],
            "old_tokens":turn["logical_before"],"input_tokens":turn["resident_input_tokens"],
            "logical_input_tokens":turn["logical_before"]+turn["delta_tokens"],
            "prefix_sha256":turn["logical_before_sha256"]})
        service.append({"event":"pause","session_id":client.sid,"operation_id":turn["request_id"],
            "output_token_ids":client.outputs[i],"resident_tokens":turn["resident_tokens"],
            "logical_tokens":turn["logical_tokens"],"prefix_sha256":turn["logical_sha256"]})
    engine.append({"event":"close_released","request_id":owner,"block_ids":[[],[]],"request_still_registered":False})
    return engine,service


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        self.client=scenario()
        self.engine,self.service=evidence(self.client)

    def audit(self):
        with tempfile.TemporaryDirectory() as folder:
            files=[]
            for name,rows in (("engine",self.engine),("service",self.service)):
                path=Path(folder)/name
                path.write_text("\n".join(json.dumps(row) for row in rows)+"\n")
                files.append(path)
            return diagnostic.audit_events(*files,self.client.sid,self.client.turns)

    def test_dry_plan_makes_no_network_or_tokenizer_calls(self):
        with patch("sys.argv",["diagnostic"]), patch("raw_gpu_checks.urlopen",side_effect=AssertionError("network")), \
             patch.object(diagnostic,"run",side_effect=AssertionError("execute")), redirect_stdout(io.StringIO()) as output:
            self.assertEqual(diagnostic.main(),0)
        self.assertEqual(json.loads(output.getvalue())["max_generated_tokens"],56)

    def test_independent_endpoint_ledger_and_alignment(self):
        self.assertEqual(self.client.logical,self.client.server_ids)
        self.assertEqual(len(self.client.requests),6)
        self.assertEqual([t["removed_tokens"] for t in self.client.turns],[0,0,3082,2784,0,2779])
        first,aligned,partial=self.client.evicted
        self.assertNotEqual(first[0]%4,0)
        self.assertEqual((aligned[0]%4,aligned[1]%4),(0,0))
        self.assertEqual(partial[0]%4,2)
        self.assertEqual(self.client.live,{})

    def test_same_observation_sends_no_owner_refresh(self):
        for i in (1,4):
            self.assertEqual(self.client.requests[i]["raw"]["suffix_records"],[])
            self.assertEqual(self.client.requests[i]["raw"]["evict_observation_ids"],[])

    def test_http_detects_wrong_resident_usage(self):
        client=SimulatedClient()
        client.corrupt=lambda result:result["usage"].update(prompt_tokens=999)
        with self.assertRaises(diagnostic.ContractFailure):client.submit("initial",[1,2])

    def test_valid_evictions_audit_exact_ledgers_and_bytes(self):
        result=self.audit()
        self.assertTrue(result["valid"],result["failures"])
        self.assertEqual(len(result["copies"]),3)
        self.assertTrue(all(copy["live_byte_checksum_proof"] for copy in result["copies"]))
        self.assertFalse(result["distinct_request_page_reuse_verified"])

    def test_historical_replay_is_rejected(self):
        pause=next(row for row in self.engine if row["event"]=="pause" and row["turn"]==2)
        pause["turn_work"]["minimum_scheduled_position"]=0
        self.assertIn("only_pending_final_and_new_input_scheduled",[x["check"] for x in self.audit()["failures"]])

    def test_duplicate_retry_submission_is_rejected(self):
        self.engine.append(deepcopy(next(r for r in self.engine if r["event"]=="resume")))
        self.assertFalse(self.audit()["valid"])

    def test_wrong_external_delta_is_rejected(self):
        next(row for row in self.service if row["event"]=="submit")["appended_token_ids"][0]=-3
        self.assertIn("service_exact_external_delta_and_old_prefix",[x["check"] for x in self.audit()["failures"]])

    def test_wrong_byte_hash_is_rejected(self):
        evict=next(row for row in self.engine if row["event"]=="evict")
        evict["eviction"]["worker_receipts"][0]["copies"][0]["byte_verification"]["after"]["sha256"]="b"*64
        self.assertIn("actual_retained_main_and_descriptor_bytes_equal",[x["check"] for x in self.audit()["failures"]])

    def test_missing_worker_fence_is_rejected(self):
        next(row for row in self.engine if row["event"]=="evict")["eviction"]["worker_receipts"][0]["worker_synchronized"]=False
        self.assertIn("worker_synchronized_after_copy",[x["check"] for x in self.audit()["failures"]])

    def test_wrong_freed_owner_page_is_rejected(self):
        next(row for row in self.engine if row["event"]=="evict")["eviction"]["freed_blocks_by_group"][0]=[999]
        self.assertIn("freed_owner_pages_match_block_change",[x["check"] for x in self.audit()["failures"]])

    def test_close_must_unregister_and_release(self):
        self.engine[-1]["request_still_registered"]=True
        self.assertIn("final_close_released_all_owned_pages",[x["check"] for x in self.audit()["failures"]])

    def probe_audit(self,*,reuse=True,owner_closed=False,still_owned=False):
        probe_sid="fixture-distinct"
        req="arc3-persistent-"+probe_sid+"-cafef00d"
        final_eviction=[row for row in self.engine if row["event"]=="evict"][-1]
        page=final_eviction["eviction"]["freed_blocks_by_group"][0][0]
        final_close=self.engine.pop()
        if still_owned:
            self.engine[-1]["block_ids"][0].append(page)
        if owner_closed:self.engine.append(final_close)
        expected={"resident_tokens":1032,"resident_sha256":"c"*64}
        self.engine.extend([
            {"event":"start","request_id":req,"block_ids":[[],[]]},
            {"event":"pause","request_id":req,"resident_tokens":1032,"tokens_sha256":"c"*64,
             "computed_tokens":1031,"in_flight_tokens":0,"preemptions":0,
             "block_ids":[[page if reuse else 9999],[901]]},
            {"event":"close_released","request_id":req,"block_ids":[[],[]],"request_still_registered":False}])
        if not owner_closed:self.engine.append(final_close)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"engine.jsonl"
            path.write_text("\n".join(json.dumps(row) for row in self.engine)+"\n")
            return diagnostic.audit_distinct_page_reuse(path,self.client.sid,probe_sid,expected)

    def test_distinct_request_natural_page_reuse(self):
        result=self.probe_audit()
        self.assertTrue(result["valid"],result)
        self.assertTrue(result["natural_page_reuse_verified"])
        self.assertEqual(len(result["reused_pages"]),1)

    def test_no_natural_page_reuse_does_not_overclaim(self):
        result=self.probe_audit(reuse=False)
        self.assertTrue(result["valid"],result)
        self.assertFalse(result["natural_page_reuse_verified"])

    def test_only_reuse_after_close_is_rejected(self):
        result=self.probe_audit(owner_closed=True)
        self.assertFalse(result["valid"])
        self.assertFalse(result["natural_page_reuse_verified"])

    def test_double_owned_page_is_rejected(self):
        result=self.probe_audit(still_owned=True)
        self.assertFalse(result["valid"])
        self.assertFalse(result["natural_page_reuse_verified"])


if __name__=="__main__":unittest.main()
