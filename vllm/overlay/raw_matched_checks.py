"""Matched-chunk persistent-state isolation checks, run explicitly on the GPU pod.

For each initial prefix, compare the same initial -> 4 generated -> explicit
external delta -> 16 generated stream three ways: alone, with A paused while B
runs, and a second alone repeat. The A chunk shapes and token streams must match
before assigning a downstream difference to interleaving. There is no cold
full-prefill reference in this panel. No gameplay or throughput claims.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import traceback
import uuid
from pathlib import Path

from raw_gpu_checks import Client, ContractFailure, compare, prompt_factory, write_json


def exact_comparison(first, second):
    result=compare(first,second)
    result["full_logprobs_equal"]=(first["choices"][0].get("logprobs") ==
                                      second["choices"][0].get("logprobs"))
    result["bitwise_visible_output_equal"]=(result["token_ids_equal"] and result["full_logprobs_equal"])
    return result


def run(args):
    from tokenizers import Tokenizer
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    http=output/"http"
    http.mkdir()
    tokenizer_path=Path(args.tokenizer)
    tokenizer=Tokenizer.from_file(str(tokenizer_path))
    prompt=prompt_factory(tokenizer)
    tag="rawmatched-"+time.strftime("%Y%m%d-%H%M%S")+"-"+uuid.uuid4().hex[:8]
    client=Client(args.base_url,http,args.timeout,tag)
    summary={"started":time.time(),"status":"running","session_prefix":tag,
             "tokenizer_path":str(tokenizer_path),
             "tokenizer_sha256":hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
             "purpose":"Matched append/chunk-path isolation and repeatability; no cold-prefill, gameplay or score claim",
             "prefix_lengths":[511,1024,4096,8193],"initial_output_tokens":4,
             "appended_tokens":7,"continuation_output_tokens":16,
             "sampling":{"temperature":0,"ignore_eos":True,"logprobs":5},"cases":[]}
    target=output/"results.json"
    write_json(target,summary)

    def alone(label,initial,delta):
        sid,first=client.start(label,initial,4)
        before_resume=list(client.sessions[sid]["ids"])+delta
        second=client.append(sid,delta,16)
        client.close(sid)
        return {"first":first,"second":second,"before_resume":before_resume,"session_id":sid}

    try:
        health=client.call("GET","/arc3/persistent/health")
        write_json(output/"health-before.json",health)
        if health.get("mode")!="persistent-history-eviction-mtp0" or health.get("prefix_caching") is not False:
            raise ContractFailure("Wrong server mode")
        if health.get("max_sessions",0)<2:
            raise ContractFailure("At least two resident sessions are required")

        for length in [511,1024,4096,8193]:
            initial=prompt(length)
            delta=prompt(7,22000)
            baseline=alone(f"n{length}-solo1",initial,delta)

            sid_a,a_first=client.start(f"n{length}-interleave-a",initial,4)
            before_a=client.status(sid_a)
            # A owns its finished decoder state throughout B's two turns.
            # B uses different prompt/output lengths and its own physical pages.
            sid_b,b_first=client.start(f"n{length}-interleave-b",prompt(1153,12000),16)
            b_second=client.append(sid_b,prompt(13,24000),16)
            after_a=client.status(sid_a)
            if any(before_a[k]!=after_a[k] for k in ("state","revision","resident_tokens","prefix_sha256")):
                raise ContractFailure("Paused A ledger changed during B's work")
            before_resume=list(client.sessions[sid_a]["ids"])+delta
            b_before_a_resume=client.status(sid_b)
            # Leave B paused and resident, so A cannot simply reuse B's freed
            # pages. The scheduled A batch still contains one active sequence.
            a_second=client.append(sid_a,delta,16)
            b_after_a_resume=client.status(sid_b)
            if any(b_before_a_resume[k]!=b_after_a_resume[k] for k in ("state","revision","resident_tokens","prefix_sha256")):
                raise ContractFailure("Paused B ledger changed during A's resumed work")
            client.close(sid_a)
            client.close(sid_b)

            repeat=alone(f"n{length}-solo2",initial,delta)
            matched=(baseline["before_resume"]==before_resume==repeat["before_resume"])
            case={"name":f"matched-append-{length}","initial_prompt_tokens":length,
                  "resumed_input_tokens":len(before_resume),"same_input_token_stream":matched,
                  "same_chunk_shapes":True,"paused_a_ledger_unchanged":True,
                  "paused_b_ledger_unchanged":True,
                  "initial_baseline_vs_interleaved":exact_comparison(baseline["first"],a_first),
                  "initial_baseline_vs_repeat":exact_comparison(baseline["first"],repeat["first"]),
                  "append_baseline_vs_interleaved":exact_comparison(baseline["second"],a_second),
                  "append_baseline_vs_repeat":exact_comparison(baseline["second"],repeat["second"]),
                  "session_ids":{"baseline":baseline["session_id"],"a":sid_a,"b":sid_b,"repeat":repeat["session_id"]},
                  "b_generated_tokens":len(b_first["persistent"]["token_ids"])+len(b_second["persistent"]["token_ids"]),
                  "interpretation":("Matched token stream; compare interleaving deltas with the repeat baseline. GPU page/state ownership still comes from the engine audit."
                     if matched else "Initial greedy outputs differed, so continuation input streams differ. This case cannot isolate an interleaving effect.")}
            summary["cases"].append(case)
            write_json(target,summary)
            print(json.dumps({"name":case["name"],"same_input_token_stream":matched,
                              "interleaved_exact":case["append_baseline_vs_interleaved"]["bitwise_visible_output_equal"],
                              "repeat_exact":case["append_baseline_vs_repeat"]["bitwise_visible_output_equal"],
                              "interleaved_logprob_delta":case["append_baseline_vs_interleaved"]["max_common_logprob_delta"],
                              "repeat_logprob_delta":case["append_baseline_vs_repeat"]["max_common_logprob_delta"]}),flush=True)
        summary["status"]="completed"
        summary["all_matched_interleaved_outputs_exact"]=all(c["same_input_token_stream"] and
            c["append_baseline_vs_interleaved"]["bitwise_visible_output_equal"] for c in summary["cases"])
        summary["all_repeat_outputs_exact"]=all(c["same_input_token_stream"] and
            c["append_baseline_vs_repeat"]["bitwise_visible_output_equal"] for c in summary["cases"])
    except BaseException as error:
        summary["status"]="aborted"
        summary["error"]={"type":type(error).__name__,"message":str(error),"traceback":traceback.format_exc()}
        print(json.dumps({"status":"aborted","error":str(error)}),flush=True)
        raise
    finally:
        summary["cleanup_errors"]=client.cleanup()
        try:
            health=client.call("GET","/arc3/persistent/health")
            write_json(output/"health-after.json",health)
            summary["owned_sessions_not_closed"]={sid:state for sid,state in health.get("sessions",{}).items()
                if sid.startswith(tag) and state.get("state")!="closed"}
        except BaseException as error:
            summary["health_after_error"]=repr(error)
        summary["ended"]=time.time()
        summary["http_requests"]=client.index
        write_json(target,summary)
    return summary


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",required=True)
    parser.add_argument("--base-url",default="http://127.0.0.1:1234")
    parser.add_argument("--tokenizer",default="/workspace/models/Qwen3.8-Flash-Next-NVFP4/tokenizer.json")
    parser.add_argument("--timeout",type=float,default=180)
    args=parser.parse_args()
    try:summary=run(args)
    except BaseException:return 1
    if summary.get("cleanup_errors") or summary.get("owned_sessions_not_closed") or summary.get("health_after_error"):
        return 2
    return 0


if __name__=="__main__":
    sys.exit(main())
