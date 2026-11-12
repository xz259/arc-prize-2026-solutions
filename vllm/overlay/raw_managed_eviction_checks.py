"""Explicit, bounded raw HTTP validation of managed suffix KV eviction.

This script never starts/stops a server and never executes recorded model code.
Without --execute it prints the plan and makes no network requests. Runtime
execution requires idle phase-two ownership, campaign locks, host/boot identity,
and readable engine/service audits. It tests state/remap accounting, not equality
to a fresh prefill of a truncated prompt, gameplay quality, or peak throughput.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
from pathlib import Path
import re
import socket
import sys
import time
import traceback
import uuid

from raw_gpu_checks import Client, ContractFailure, prompt_factory, token_digest, write_json


MODE = "persistent-history-eviction-mtp0"
LOCKS = ("public-eval-campaign.lock", "mtp-precision-sweep.lock",
         "bf16-gameplay-ablation.lock", "mtp0-memory-diagnostic.lock")
SOURCE_FILES = ("raw_managed_eviction_checks.py", "raw_gpu_checks.py", "persistent_endpoint.py",
                "session_service.py", "engine_patch.py", "engine_eviction.py", "eviction_runner.py",
                "eviction_storage.py", "eviction_validation.py", "eviction_qsa.py", "_eviction_qsa_gpu.py",
                "residency.py", "suffix_registry.py")


def plan_description():
    return {"mode": MODE, "sessions": 2, "generation_operations": 7,
            "max_generated_tokens": 56, "evictions": 3,
            "cases": ["start 6143-token prompt with nonaligned 3082-token managed span",
                      "17-token inspection append, same observation: no eviction",
                      "4101-token new observation: retire first span; register aligned 2784-token span",
                      "4099-token new observation: retire aligned span; register nonaligned 2779-token span",
                      "7-token inspection append, same observation: no eviction",
                      "65-token new observation without suffix: retire third span",
                      "1024-token distinct request while first owner remains paused: inspect natural freed-page reuse"],
            "retry": "Replay one exact append request ID/body; no second engine submission",
            "output_tokens_per_operation": 8,
            "proof": "Exact HTTP ledgers plus source/engine/service audit; synchronized copies and freed owner pages",
            "excluded_claims": ["fresh truncated-prefill logit parity", "auxiliary-state byte checksums", "gameplay score", "throughput benchmark"]}


def kept_ids(logical, ranges):
    return [token for position, token in enumerate(logical)
            if not any(start <= position < end for start, end in ranges)]


class ManagedClient(Client):
    def __init__(self, *args, deadline, **kwargs):
        super().__init__(*args, **kwargs)
        self.deadline = deadline
        self.logical = []
        self.evicted = []
        self.live = {}
        self.turns = []
        self.last_body = None
        self.last_response = None
        self.eviction_revision = 0
        self.sid = self.unique("owned")

    def call(self, method, route, payload=None):
        cleanup = route.endswith("/close") or route.endswith("/health")
        remaining = self.deadline-time.monotonic()
        if remaining <= 0 and not cleanup:
            raise ContractFailure("Bounded diagnostic deadline reached")
        previous = self.timeout
        self.timeout = min(previous, 20 if cleanup else max(.1, remaining))
        try:
            return super().call(method, route, payload)
        finally:
            self.timeout = previous

    def submit(self, name, delta, *, suffix=None, new_observation=False):
        initial = not self.turns
        old_logical = list(self.logical)
        prior_evicted = list(self.evicted)
        oid = f"observation-{len(self.turns)+1}"
        records = ([{"observation_id": oid, "local_token_range": suffix}]
                   if new_observation else [])
        retire = list(self.live) if records else []
        removed_ranges = [self.live[owner] for owner in retire]
        removed = sum(end-start for start, end in removed_ranges)
        after_ranges = [*prior_evicted, *removed_ranges]
        old_after = kept_ids(old_logical, after_ranges)
        raw = {"token_ids": list(delta), "suffix_records": records,
               "evict_observation_ids": retire}
        body = {"request_id": self.sid+f"-r{len(self.turns)+1}", "raw": raw,
                "sampling": {"temperature": 0, "max_tokens": 8, "ignore_eos": True}}
        if initial:
            body["session_id"] = self.sid
            route = "/arc3/persistent/start"
            self.active.add(self.sid)  # uncertain HTTP starts are still owned
        else:
            body.update(expected_revision=self.last_response["persistent"]["revision"],
                        expected_prefix_sha256=token_digest(old_logical))
            route = f"/arc3/persistent/{self.sid}/append"
        response = self.call("POST", route, body)
        state = response["persistent"]
        output = state["token_ids"]
        if len(output) != 8 or state["engine_finish_reason"] != "length":
            raise ContractFailure("Expected exactly eight capped diagnostic output tokens")
        logical = [*old_logical, *delta, *output]
        resident = kept_ids(logical, after_ranges)
        expected_revision = len(self.turns)+1
        expected_eviction_revision = self.eviction_revision+bool(removed)
        checks = {
            "session": state.get("session_id") == self.sid,
            "revision": state.get("revision") == expected_revision,
            "resident": state.get("resident_tokens") == len(resident),
            "logical": state.get("logical_tokens") == len(logical),
            "logical_digest": state.get("prefix_sha256") == token_digest(logical),
            "appended": state.get("appended_prompt_tokens") == len(delta),
            "eviction_count": state.get("evicted_tokens") == removed,
            "eviction_revision": state.get("eviction_revision") == expected_eviction_revision,
            "prefix_off": state.get("prefix_caching") is False,
            "eviction_enabled": state.get("eviction") is True,
            "ownership": state.get("managed_suffixes") == {
                "registered_observation_ids": [oid] if records else [],
                "evicted_observation_ids": retire},
            "resident_input_usage": response["usage"]["prompt_tokens"] == len(old_after)+len(delta),
            "logical_input_count": state.get("logical_input_tokens") == len(old_logical)+len(delta),
            "output_usage": response["usage"]["completion_tokens"] == len(output),
        }
        if not all(checks.values()):
            raise ContractFailure("HTTP contract failed: "+str({key: value for key,value in checks.items() if not value}))
        turn = {"name": name, "request_id": body["request_id"], "turn": len(self.turns),
                "initial": initial, "delta_tokens": len(delta), "output_tokens": len(output),
                "delta_sha256":token_digest(delta), "output_sha256":token_digest(output),
                "resident_input_tokens": len(old_after)+len(delta), "resident_tokens": len(resident),
                "resident_sha256": token_digest(resident), "computed_prefix_sha256": token_digest(resident[:-1]),
                "logical_tokens": len(logical), "logical_sha256": token_digest(logical),
                "logical_before": len(old_logical), "logical_before_sha256": token_digest(old_logical),
                "resident_before_eviction": len(kept_ids(old_logical, prior_evicted)),
                "resident_after_eviction": len(old_after), "after_eviction_sha256": token_digest(old_after),
                "removed_tokens": removed, "removed_ranges": removed_ranges,
                "eviction_revision": expected_eviction_revision,
                "expected_prefill_tokens": len(delta)+(not initial),
                "expected_resume_computed": 0 if initial else len(old_after)-1,
                "http_checks": checks}
        self.logical, self.evicted = logical, after_ranges
        if records:
            self.live = {} if suffix is None else {oid: (len(old_logical)+suffix[0], len(old_logical)+suffix[1])}
        self.eviction_revision = expected_eviction_revision
        self.turns.append(turn)
        self.last_body, self.last_response = body, response
        return turn

    def replay_last(self):
        before = self.status(self.sid)
        replay = self.call("POST", f"/arc3/persistent/{self.sid}/append", self.last_body)
        after = self.status(self.sid)
        if replay != self.last_response or before != after:
            raise ContractFailure("Idempotent append replay changed state or returned a different result")
        return {"request_id": self.last_body["request_id"], "response_exact": True, "status_unchanged": True}

    def start_reuse_probe(self,delta):
        sid=self.unique("reuseprobe")
        self.active.add(sid)
        response=self.call("POST","/arc3/persistent/start",{
            "session_id":sid,"request_id":sid+"-start",
            "raw":{"token_ids":list(delta),"suffix_records":[],"evict_observation_ids":[]},
            "sampling":{"temperature":0,"max_tokens":8,"ignore_eos":True}})
        state=response["persistent"]
        ids=[*delta,*state["token_ids"]]
        if (state.get("session_id")!=sid or state.get("revision")!=1
                or state.get("resident_tokens")!=len(ids) or state.get("logical_tokens")!=len(ids)
                or state.get("prefix_sha256")!=token_digest(ids) or state.get("evicted_tokens")!=0
                or state.get("eviction_revision")!=0 or len(state["token_ids"])!=8
                or state.get("engine_finish_reason")!="length" or state.get("prefix_caching") is not False
                or state.get("eviction") is not True or response["usage"]["prompt_tokens"]!=len(delta)):
            raise ContractFailure("Distinct-request page-reuse probe ledger mismatch")
        return sid,{"resident_tokens":len(ids),"resident_sha256":token_digest(ids)}


def audit_distinct_page_reuse(engine_path,owner_sid,probe_sid,probe_expected):
    rows=[json.loads(line) for line in Path(engine_path).read_text().splitlines() if line]
    def matching(sid):
        pattern=re.compile(re.escape("arc3-persistent-"+sid)+r"(?:-[0-9a-f]{8})?\Z")
        return [(i,row) for i,row in enumerate(rows) if pattern.fullmatch(row.get("request_id",""))]
    owner,probe=matching(owner_sid),matching(probe_sid)
    evictions=[(i,row) for i,row in owner if row["event"]=="evict"]
    starts=[(i,row) for i,row in probe if row["event"]=="start"]
    pauses=[(i,row) for i,row in probe if row["event"]=="pause"]
    closes=[(i,row) for i,row in probe if row["event"]=="close_released"]
    errors=[]
    if not evictions or len(starts)!=1 or len(pauses)!=1 or len(closes)!=1:
        return {"valid":False,"errors":["Missing distinct-owner lifecycle"],"natural_page_reuse_verified":False}
    e,eviction=evictions[-1]
    s,start=starts[0]; p,pause=pauses[0]; c,close=closes[0]
    if not e<s<p<c:errors.append("Distinct request did not follow final eviction and close in order")
    if (pause.get("resident_tokens")!=probe_expected["resident_tokens"]
            or pause.get("tokens_sha256")!=probe_expected["resident_sha256"]
            or pause.get("computed_tokens")!=probe_expected["resident_tokens"]-1
            or pause.get("in_flight_tokens")!=0 or pause.get("preemptions")!=0):
        errors.append("Distinct request pause ledger/frontier mismatch")
    if any(close.get("block_ids",[[]])) or close.get("request_still_registered") is not False:
        errors.append("Distinct request failed to release and unregister")
    freed=eviction.get("eviction",{}).get("freed_blocks_by_group",[])
    reuse=[{"block_id":block,"freed_owner_group":old_group,"new_owner_group":new_group}
           for old_group,old_blocks in enumerate(freed)
           for new_group,new_blocks in enumerate(pause.get("block_ids",[]))
           for block in old_blocks if block in new_blocks]
    last_owner_pause=[row for i,row in owner if e<i<s and row["event"]=="pause"]
    if len(last_owner_pause)!=1:errors.append("Cannot prove first owner remained paused before second admission")
    elif any(item["block_id"] in group for item in reuse for group in last_owner_pause[0].get("block_ids",[])):
        errors.append("Reused page remained registered to first owner")
    if any(row["event"] in {"close","close_released"} for i,row in owner if i<p):
        errors.append("First owner closed before distinct-request reuse; evidence would only prove close reuse")
    return {"valid":not errors,"errors":errors,"natural_page_reuse_verified":bool(reuse) and not errors,
            "reused_pages":reuse,"freed_blocks_by_group":freed,
            "probe_blocks_by_group":pause.get("block_ids",[]),
            "note":"Only exact page IDs observed on a distinct live request are counted; no pool order manipulation."}


def audit_events(engine_path, service_path, sid, turns):
    def load(path):
        return [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    engine_id = re.compile(re.escape("arc3-persistent-"+sid)+r"(?:-[0-9a-f]{8})?\Z")
    engine = [row for row in load(engine_path) if engine_id.fullmatch(row.get("request_id", ""))]
    service = [row for row in load(service_path) if row.get("session_id") == sid]
    failures, counts = [], {}
    def check(ok, key, detail):
        counts[key] = counts.get(key,0)+1
        if not ok: failures.append({"check":key,"detail":detail})
    check(bool(engine), "engine_events_present", sid)
    check(len({row["request_id"] for row in engine}) == 1, "one_engine_owner", sid)
    starts = [row for row in engine if row["event"] == "start"]
    resumes = [row for row in engine if row["event"] == "resume"]
    pauses = [row for row in engine if row["event"] == "pause"]
    evictions = [row for row in engine if row["event"] == "evict"]
    submits = [row for row in service if row["event"] == "submit"]
    service_pauses = [row for row in service if row["event"] == "pause"]
    commits = [row for row in service if row["event"] == "evict_commit"]
    check(len(starts)==1 and len(resumes)==len(turns)-1 and len(pauses)==len(turns),
          "exact_turn_counts_no_retry_submission", [len(starts),len(resumes),len(pauses)])
    check(len(submits)==len(turns), "service_exact_submission_count", len(submits))
    check(len(evictions)==3 and len(commits)==3, "three_evictions", [len(evictions),len(commits)])
    check(not any(row["event"] in {"failed", "rejected"} for row in service), "no_service_failure", sid)
    previous = None
    eviction_index = 0
    copy_metrics = []
    for i, turn in enumerate(turns):
        submit_matches = [row for row in submits if row.get("operation_id")==turn["request_id"]]
        output_matches = [row for row in service_pauses if row.get("operation_id")==turn["request_id"]]
        check(len(submit_matches)==1 and len(output_matches)==1,"one_service_input_output_per_operation",i)
        if submit_matches and output_matches:
            submit, output = submit_matches[0],output_matches[0]
            check(token_digest(submit.get("appended_token_ids",[]))==turn["delta_sha256"]
                  and submit.get("old_tokens")==turn["logical_before"]
                  and submit.get("input_tokens")==turn["resident_input_tokens"]
                  and submit.get("logical_input_tokens")==turn["logical_before"]+turn["delta_tokens"]
                  and submit.get("prefix_sha256")==turn["logical_before_sha256"],
                  "service_exact_external_delta_and_old_prefix",i)
            check(token_digest(output.get("output_token_ids",[]))==turn["output_sha256"]
                  and output.get("resident_tokens")==turn["resident_tokens"]
                  and output.get("logical_tokens")==turn["logical_tokens"]
                  and output.get("prefix_sha256")==turn["logical_sha256"],
                  "service_exact_pause_output",i)
        matching = [row for row in pauses if row["turn"] == i]
        check(len(matching)==1, "one_pause_per_turn", i)
        if len(matching)!=1: continue
        pause = matching[0]
        for key, expected in (("resident_tokens",turn["resident_tokens"]), ("tokens_sha256",turn["resident_sha256"]),
                              ("computed_prefix_sha256",turn["computed_prefix_sha256"]),
                              ("computed_tokens",turn["resident_tokens"]-1), ("in_flight_tokens",0),
                              ("output_placeholders",0), ("preemptions",0)):
            check(pause.get(key)==expected, "pause_"+key, {"turn":i,"actual":pause.get(key),"expected":expected})
        logical = pause.get("logical_eviction_state", {})
        check(logical.get("logical_sha256")==turn["logical_sha256"] and logical.get("logical_tokens")==turn["logical_tokens"],
              "original_logical_ledger", i)
        work = pause.get("turn_work", {})
        expected_work = {"prefill_tokens":turn["expected_prefill_tokens"],
                         "decode_tokens":turn["output_tokens"]-1,
                         "scheduled_tokens":turn["expected_prefill_tokens"]+turn["output_tokens"]-1,
                         "minimum_scheduled_position":turn["expected_resume_computed"],
                         "resume_computed_tokens":turn["expected_resume_computed"]}
        check(all(work.get(k)==v for k,v in expected_work.items()), "only_pending_final_and_new_input_scheduled",
              {"turn":i,"actual":work,"expected":expected_work})
        if not turn["initial"]:
            resume_matches = [row for row in resumes if row["turn"]==i]
            check(len(resume_matches)==1, "one_resume_per_append", i)
            if resume_matches:
                resume = resume_matches[0]
                check(resume.get("append",{}).get("pending_prefill_tokens")==turn["delta_tokens"]+1,
                      "append_prefill_delta_plus_one", i)
                anchor = previous
                if turn["removed_tokens"]:
                    eviction = evictions[eviction_index] if eviction_index < len(evictions) else {}
                    eviction_index += 1
                    ack = eviction.get("eviction", {})
                    check(eviction.get("resident_tokens")==turn["resident_after_eviction"] and
                          eviction.get("tokens_sha256")==turn["after_eviction_sha256"],
                          "compaction_exact_resident_subsequence", i)
                    check(eviction.get("computed_tokens")==turn["resident_after_eviction"]-1 and
                          eviction.get("in_flight_tokens")==0 and eviction.get("preemptions")==0,
                          "compaction_at_settled_frontier", i)
                    check(ack.get("evicted_tokens")==turn["removed_tokens"] and
                          ack.get("logical_tokens")==turn["logical_before"], "eviction_ack_counts", i)
                    receipts = ack.get("worker_receipts", [])
                    check(len(receipts)==1 and receipts[0].get("worker_synchronized") is True,
                          "worker_synchronized_after_copy", i)
                    copies = [item for receipt in receipts for item in receipt.get("copies", [])]
                    check(bool(copies) and any(item.get("copied_rows",0)>0 for item in copies), "cache_rows_actually_copied", i)
                    check(bool(receipts) and all(r.get("byte_verification_enabled") is True for r in receipts),
                          "live_byte_verification_enabled", i)
                    byte_proofs = [item.get("byte_verification", {}) for item in copies]
                    def valid_byte_proof(proof):
                        before, after = proof.get("before", {}), proof.get("after", {})
                        return (proof.get("equal") is True and before == after
                                and bool(re.fullmatch(r"[0-9a-f]{64}",str(before.get("sha256", ""))))
                                and type(before.get("bytes")) is int and before["bytes"] > 0
                                and type(before.get("rows")) is int and before["rows"] > 0)
                    check(bool(copies) and all(valid_byte_proof(p) for p in byte_proofs),
                          "actual_retained_main_and_descriptor_bytes_equal", i)
                    service_commit = next((row for row in commits if row.get("operation_id")==turn["request_id"]), {})
                    check(service_commit.get("acknowledgement")==ack,
                          "service_commits_exact_engine_ack", i)
                    freed = ack.get("freed_blocks_by_group", [])
                    before = previous.get("block_ids", []) if previous else []
                    after = eviction.get("block_ids", [])
                    check(len(before)==len(after)==len(freed) and all(set(b)-set(a)==set(f) and
                          not(set(a)-set(b)) for b,a,f in zip(before,after,freed)), "freed_owner_pages_match_block_change", i)
                    copy_metrics.append({"turn":i,"freed_blocks_by_group":freed,
                        "reported_copied_rows":sum(item.get("copied_rows",0) for item in copies),
                        "reported_copied_bytes":sum(item.get("copied_rows",0)*item.get("row_bytes",0) for item in copies),
                        "worker_synchronized": bool(receipts) and all(r.get("worker_synchronized") is True for r in receipts),
                        "live_byte_checksum_proof": bool(copies) and all(valid_byte_proof(p) for p in byte_proofs),
                        "retained_bytes_hashed":sum(p.get("before",{}).get("bytes",0) for p in byte_proofs),
                        "retained_layers_hashed":len(byte_proofs)})
                    anchor = eviction
                check(resume.get("block_ids")==anchor.get("block_ids") if anchor else False,
                      "resume_reuses_compacted_pages_before_allocation", i)
        previous = pause
    closes = [row for row in engine if row["event"]=="close_released"]
    check(bool(closes) and all(not any(row.get("block_ids", [[]])) and
                              row.get("request_still_registered") is False for row in closes),
          "final_close_released_all_owned_pages", sid)
    check(any(any(group) for item in copy_metrics for group in item["freed_blocks_by_group"]),
          "at_least_one_physical_page_freed", sid)
    return {"valid":not failures,"checks":counts,"failures":failures,"copies":copy_metrics,
            "distinct_request_page_reuse_verified":False,
            "evidence_limits":"Live receipts require exact retained main KV/scales and compressed-descriptor byte hashes. Auxiliary recurrent/raw-ring state is excluded from compaction but not independently hashed. Freed pages and same-owner resume are checked; reuse by a different request and fresh truncated-prefill semantic parity are not claimed."}


def run(args):
    if socket.gethostname() != args.expected_hostname:
        raise ContractFailure("Host identity mismatch")
    boot = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if boot != args.expected_boot_id:
        raise ContractFailure("Boot identity mismatch")
    for path in (args.engine_events,args.service_events):
        if not Path(path).is_file(): raise ContractFailure("Required audit file missing: "+str(path))
    output = Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    (output/"http").mkdir()
    here = Path(__file__).resolve().parent
    summary = {"status":"starting","plan":plan_description(),"started":time.time(),
               "hostname":socket.gethostname(),"boot_id":boot,
               "source_sha256":{name:hashlib.sha256((here/name).read_bytes()).hexdigest() for name in SOURCE_FILES},
               "audit_files":{"engine":str(Path(args.engine_events).resolve()),"service":str(Path(args.service_events).resolve())}}
    target = output/"results.json"
    write_json(target,summary)
    tag = "raw-evict-"+time.strftime("%Y%m%d-%H%M%S")+"-"+uuid.uuid4().hex[:8]
    client = ManagedClient(args.base_url,output/"http",args.timeout,tag,
                           deadline=time.monotonic()+args.max_seconds)
    summary["session_id"] = client.sid
    with ExitStack() as stack:
        for name in LOCKS:
            handle = stack.enter_context((Path(args.locks_dir)/name).open("a+"))
            fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            health = client.call("GET","/arc3/persistent/health")
            write_json(output/"health-before.json",health)
            if (health.get("mode")!=MODE or health.get("prefix_caching") is not False
                    or health.get("max_context") != args.expected_context):
                raise ContractFailure("Wrong phase-two serving mode/context")
            if any(state.get("state") not in {"closed","error"} for state in health.get("sessions",{}).values()):
                raise ContractFailure("Another resident session owns the server; no diagnostic started")
            if health.get("max_sessions",0)<2:
                raise ContractFailure("Distinct-request reuse check requires admission capacity of two")
            from tokenizers import Tokenizer
            tokenizer_path = Path(args.tokenizer)
            summary["tokenizer_sha256"] = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
            prompt = prompt_factory(Tokenizer.from_file(str(tokenizer_path)))
            client.submit("initial-nonaligned-owner",prompt(6143),suffix=[1025,4107],new_observation=True)
            client.submit("same-observation-inspection",prompt(17,7000))
            start = 128+(-(len(client.logical)+128))%4
            client.submit("retire-first-register-full-groups",prompt(4101,9000),suffix=[start,start+2784],new_observation=True)
            summary["idempotency"] = client.replay_last()
            start = 128+(2-(len(client.logical)+128))%4
            client.submit("retire-full-groups-register-partials",prompt(4099,15000),suffix=[start,start+2779],new_observation=True)
            client.submit("same-observation-after-evictions",prompt(7,20000))
            client.submit("new-observation-without-suffix",prompt(65,22000),suffix=None,new_observation=True)
            summary["turns"] = client.turns
            probe_sid,probe_expected=client.start_reuse_probe(prompt(1024,23000))
            summary["probe_session_id"]=probe_sid
            client.close(probe_sid)
            client.close(client.sid)
            summary["audit"] = audit_events(args.engine_events,args.service_events,client.sid,client.turns)
            summary["distinct_request_reuse"] = audit_distinct_page_reuse(
                args.engine_events,client.sid,probe_sid,probe_expected)
            summary["audit"]["distinct_request_page_reuse_verified"] = summary["distinct_request_reuse"]["natural_page_reuse_verified"]
            if not summary["audit"]["valid"] or not summary["distinct_request_reuse"]["valid"]:
                raise ContractFailure("Source/engine/service audit failed; see results.json")
            summary["status"] = "passed"
        except BaseException as exc:
            summary.update(status="failed",error={"type":type(exc).__name__,"message":str(exc),"traceback":traceback.format_exc()})
            raise
        finally:
            summary["turns"] = client.turns
            summary["cleanup_errors"] = client.cleanup()
            try:
                final = client.call("GET","/arc3/persistent/health")
                write_json(output/"health-after.json",final)
                owned = final.get("sessions",{}).get(client.sid)
                summary["owned_final_state"] = owned
                if client.turns and (owned is None or owned.get("state")!="closed"):
                    summary["status"] = "failed"
                if summary.get("probe_session_id"):
                    probe=final.get("sessions",{}).get(summary["probe_session_id"])
                    summary["probe_final_state"]=probe
                    if probe is None or probe.get("state")!="closed":summary["status"]="failed"
            except BaseException as exc:
                summary["health_after_error"] = repr(exc)
                summary["status"] = "failed"
            summary["ended"] = time.time()
            summary["http_requests"] = client.index
            write_json(target,summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute",action="store_true")
    parser.add_argument("--output")
    parser.add_argument("--base-url",default="http://127.0.0.1:1234")
    parser.add_argument("--tokenizer",default="/workspace/models/Qwen3.8-Flash-Next-NVFP4/tokenizer.json")
    parser.add_argument("--locks-dir",default="/workspace/arc3-inference/runs")
    parser.add_argument("--expected-hostname")
    parser.add_argument("--expected-boot-id")
    parser.add_argument("--engine-events")
    parser.add_argument("--service-events")
    parser.add_argument("--expected-context",type=int,default=40000)
    parser.add_argument("--max-seconds",type=float,default=180)
    parser.add_argument("--timeout",type=float,default=90)
    args = parser.parse_args()
    if not args.execute:
        print(json.dumps(plan_description(),indent=2)); return 0
    if any(getattr(args,key) is None for key in ("output","expected_hostname","expected_boot_id","engine_events","service_events")):
        parser.error("Execution requires output, host/boot identity, and both audit file paths")
    if not 0 < args.timeout <= 180 or not 0 < args.max_seconds <= 600 or args.expected_context < 15000:
        parser.error("Require timeout<=180s, total<=600s, context>=15000")
    try:
        summary = run(args)
        return 0 if summary["status"]=="passed" and not summary["cleanup_errors"] else 1
    except BaseException as exc:
        print(json.dumps({"status":"failed","error":str(exc)}),flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
