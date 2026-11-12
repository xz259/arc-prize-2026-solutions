"""Bounded live raw-token correctness checks for the persistent MTP0 endpoint.

Run explicitly on the GPU server. No invocation, import or --help call launches
work elsewhere. Writes every HTTP request/response before interpreting results.
No retries and no gameplay/score/throughput claims. Numeric differences are
reported, not silently treated as either proof of a bug or proof of correctness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import traceback
import uuid
from pathlib import Path
from urllib.error import HTTPError
from urllib.parse import urlparse
from urllib.request import Request, urlopen


class HTTPFailure(RuntimeError):
    pass


class ContractFailure(RuntimeError):
    pass


def token_digest(ids):
    return hashlib.sha256(json.dumps(list(ids), sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def compare(first, second):
    a, b = first["persistent"]["token_ids"], second["persistent"]["token_ids"]
    common_prefix = 0
    for x, y in zip(a,b):
        if x != y:
            break
        common_prefix += 1
    a_logs = first["choices"][0].get("logprobs") or []
    b_logs = second["choices"][0].get("logprobs") or []
    rows = []
    for index, (left,right) in enumerate(zip(a_logs,b_logs)):
        keys = sorted(set(left) & set(right), key=int)
        finite = [abs(float(left[key]["logprob"])-float(right[key]["logprob"])) for key in keys
                  if math.isfinite(float(left[key]["logprob"])) and math.isfinite(float(right[key]["logprob"]))]
        rows.append({"position":index,"common_top_tokens":len(keys),
                     "max_common_logprob_delta":max(finite) if finite else None,
                     "top_token_sets_match":set(left)==set(right)})
    values = [r["max_common_logprob_delta"] for r in rows if r["max_common_logprob_delta"] is not None]
    return {"token_ids_equal":a==b,"first_mismatch":None if a==b else common_prefix,
            "left_tokens":len(a),"right_tokens":len(b),
            "max_common_logprob_delta":max(values) if values else None,"logprob_rows":rows}


def combined(first, second):
    return {"persistent":{"token_ids": first["persistent"]["token_ids"] + second["persistent"]["token_ids"]},
            "choices":[{"logprobs": (first["choices"][0].get("logprobs") or []) +
                                      (second["choices"][0].get("logprobs") or [])}]}


class Client:
    def __init__(self, base_url, evidence, timeout, prefix):
        parsed = urlparse(base_url)
        if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1","localhost","::1"}:
            raise ValueError("This diagnostic targets a loopback HTTP server only")
        self.base_url=base_url.rstrip("/")
        self.evidence=evidence
        self.timeout=timeout
        self.prefix=prefix
        self.index=0
        self.sessions={}
        self.active=set()

    def call(self, method, route, payload=None):
        self.index+=1
        request_record={"index":self.index,"method":method,"route":route,
                        "payload":payload,"started":time.time()}
        request_file=self.evidence/f"{self.index:03d}-request.json"
        response_file=self.evidence/f"{self.index:03d}-response.json"
        write_json(request_file,request_record)
        data=None if payload is None else json.dumps(payload).encode()
        request=Request(self.base_url+route,data=data,method=method,
                        headers={"Content-Type":"application/json"})
        started=time.monotonic()
        try:
            with urlopen(request,timeout=self.timeout) as response:
                status=response.status
                raw=response.read().decode()
        except HTTPError as error:
            raw=error.read().decode(errors="replace")
            write_json(response_file,{"status":error.code,"body":raw,"elapsed":time.monotonic()-started})
            raise HTTPFailure(f"HTTP {error.code} on {method} {route}: {raw[:2000]}") from error
        except BaseException as error:
            write_json(response_file,{"error":repr(error),"elapsed":time.monotonic()-started})
            raise
        try:
            result=json.loads(raw)
        except ValueError:
            write_json(response_file,{"status":status,"body":raw,"elapsed":time.monotonic()-started})
            raise HTTPFailure(f"Non-JSON response on {route}")
        write_json(response_file,{"status":status,"body":result,"elapsed":time.monotonic()-started})
        if not 200<=status<300:
            raise HTTPFailure(f"HTTP {status} on {route}")
        return result

    def unique(self,label):
        return self.prefix+"-"+label

    @staticmethod
    def sampling(count):
        return {"temperature":0,"max_tokens":count,"ignore_eos":True,"logprobs":5}

    def _record_result(self,sid,old_ids,delta,result,count):
        if "error" in result:
            raise ContractFailure(f"Error result for {sid}: {result['error']}")
        state=result["persistent"]
        output=state["token_ids"]
        if len(output)!=count or state["engine_finish_reason"]!="length":
            raise ContractFailure(f"Expected {count} raw length-capped output tokens for {sid}")
        full=list(old_ids)+list(delta)+list(output)
        if state["resident_tokens"]!=len(full) or state["prefix_sha256"]!=token_digest(full):
            raise ContractFailure(f"Token ledger does not match engine response for {sid}")
        if (state.get("eviction") is not True or state.get("evicted_tokens") != 0
                or state.get("prefix_caching") is not False):
            raise ContractFailure("No-deletion control unexpectedly evicted tokens or enabled prefix caching")
        if state["appended_prompt_tokens"]!=len(delta):
            raise ContractFailure("Reported appended-token count is inconsistent")
        self.sessions[sid]={"ids":full,"state":state}
        return result

    def start(self,label,ids,count):
        sid=self.unique(label)
        if sid in self.sessions:
            raise ValueError("Duplicate local session label")
        # Include uncertain starts in cleanup: a failed HTTP reply can follow
        # successful engine admission. IDs are unique to this invocation.
        self.active.add(sid)
        response=self.call("POST","/arc3/persistent/start",
                           {"session_id":sid,"request_id":sid+"-start",
                            "raw":{"token_ids":list(ids)},"sampling":self.sampling(count)})
        return sid,self._record_result(sid,[],ids,response,count)

    def append(self,sid,delta,count,*,continuation=False):
        prior=self.sessions[sid]
        revision=prior["state"]["revision"]
        response=self.call("POST",f"/arc3/persistent/{sid}/append",
                          {"request_id":sid+f"-append-{revision}","expected_revision":revision,
                           "expected_prefix_sha256":prior["state"]["prefix_sha256"],
                           "raw":{"token_ids":list(delta)},"continuation":continuation,
                           "sampling":self.sampling(count)})
        if response["persistent"]["revision"]!=revision+1:
            raise ContractFailure("Unexpected revision after append")
        return self._record_result(sid,prior["ids"],delta,response,count)

    def status(self,sid):
        return self.call("GET",f"/arc3/persistent/{sid}/status")

    def close(self,sid):
        result=self.call("POST",f"/arc3/persistent/{sid}/close",{})
        if result.get("state")!="closed":
            raise ContractFailure(f"Session did not close: {result}")
        self.active.discard(sid)
        return result

    def fresh(self,label,ids,count):
        sid,result=self.start(label,ids,count)
        self.close(sid)
        return result

    def cleanup(self):
        errors=[]
        for sid in sorted(self.active):
            try:self.close(sid)
            except BaseException as error:errors.append({"session_id":sid,"error":repr(error)})
        return errors


def prompt_factory(tokenizer):
    # Deliberately varying deterministic text exercises the stored sequence and
    # compressor ring without depending on image/chat parser behavior.
    source="".join(f"Entry {index}: position ({index%61}, {(index*7)%61}), color {index%16}. "
                   "Inspect the sequence and retain the earlier entries.\n" for index in range(1500))
    ids=tokenizer.encode(source,add_special_tokens=False).ids
    if len(ids)<25000:
        raise ValueError("Diagnostic prompt source unexpectedly too short")
    def prompt(length,offset=0):
        return list(ids[offset:offset+length])
    return prompt


def run(args):
    from tokenizers import Tokenizer
    output=Path(args.output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    http=output/"http"
    http.mkdir()
    tokenizer_path=Path(args.tokenizer)
    tokenizer=Tokenizer.from_file(str(tokenizer_path))
    prompt=prompt_factory(tokenizer)
    tag="rawgpu-"+time.strftime("%Y%m%d-%H%M%S")+"-"+uuid.uuid4().hex[:8]
    client=Client(args.base_url,http,args.timeout,tag)
    summary={"started":time.time(),"status":"running","session_prefix":tag,
             "tokenizer_path":str(tokenizer_path),
             "tokenizer_sha256":hashlib.sha256(tokenizer_path.read_bytes()).hexdigest(),
             "purpose":"raw-token persistent state correctness diagnostics; no gameplay or score claim",
             "sampling":Client.sampling(None),"cases":[]}
    target=output/"results.json"
    write_json(target,summary)

    def record(case):
        summary["cases"].append(case)
        write_json(target,summary)
        compact={k:v for k,v in case.items() if k not in ("responses","token_ids")}
        print(json.dumps(compact),flush=True)

    try:
        health=client.call("GET","/arc3/persistent/health")
        write_json(output/"health-before.json",health)
        if health.get("mode")!="persistent-history-eviction-mtp0" or health.get("prefix_caching") is not False:
            raise ContractFailure("Wrong server mode")
        if health.get("max_sessions",0)<2:
            raise ContractFailure("At least two resident sessions are needed for isolation checks")

        # First and deliberately smallest live test. A server-side HTTP 500
        # aborts this function before any remaining generation is attempted.
        sid,first=client.start("smoke",prompt(48),4)
        second=client.append(sid,prompt(3,800),4)
        client.close(sid)
        record({"name":"smoke-4-plus-4","status":"passed","session_id":sid,
                "resident_tokens":second["persistent"]["resident_tokens"]})
        if args.smoke_only:
            summary["status"]="smoke_passed"
            return summary

        initial=prompt(511)
        full=client.fresh("full64",initial,64)
        sid,left=client.start("split64",initial,32)
        right=client.append(sid,[],32,continuation=True)
        client.close(sid)
        cold_repeat=client.fresh("full64-repeat",initial,64)
        record({"name":"uninterrupted64-vs32-plus32-empty",
                "comparison":compare(full,combined(left,right)),
                "cold_repeat_control":compare(full,cold_repeat),
                "note":"Same exact input stream; pause is the only split operation. Source/event audit must also verify no historical re-prefill."})

        initial=prompt(1023)
        sid,warm=client.start("append",initial,8)
        delta=prompt(9,2000)
        prefix=client.sessions[sid]["ids"]+delta
        resumed=client.append(sid,delta,1)
        client.close(sid)
        cold=client.fresh("append-cold",prefix,1)
        cold_again=client.fresh("append-cold-repeat",prefix,1)
        record({"name":"explicit-delta-next-token","comparison":compare(resumed,cold),
                "cold_repeat_control":compare(cold,cold_again),
                "prefix_tokens":len(prefix),
                "note":"Fresh full-prefix prefill can differ numerically from retained state; report top-logprob deltas separately."})

        initial_a,initial_b=prompt(769),prompt(777,5000)
        sid_a,a1=client.start("interleave-a",initial_a,8)
        before_a=client.status(sid_a)
        sid_b,b1=client.start("interleave-b",initial_b,8)
        b2=client.append(sid_b,prompt(7,10000),8)
        after_a=client.status(sid_a)
        if any(before_a[k]!=after_a[k] for k in ("state","revision","resident_tokens","prefix_sha256")):
            raise ContractFailure("Session A token ledger changed while only B executed")
        a_delta=prompt(5,15000)
        a_prefix=client.sessions[sid_a]["ids"]+a_delta
        a2=client.append(sid_a,a_delta,1)
        b_prefix=client.sessions[sid_b]["ids"]
        b3=client.append(sid_b,[],1,continuation=True)
        client.close(sid_a)
        client.close(sid_b)
        a_cold=client.fresh("interleave-a-cold",a_prefix,1)
        b_cold=client.fresh("interleave-b-cold",b_prefix,1)
        record({"name":"interleaved-two-session-isolation","paused_ledger_unchanged":True,
                "a_comparison":compare(a2,a_cold),"b_comparison":compare(b3,b_cold),
                "note":"Physical group ownership and unchanged paused-state GPU fingerprints require the engine/worker audit; ledger equality alone is insufficient."})

        lengths=[511,512,513,1023,1024,1025,2047,2048,2049,4095]
        if args.extended_boundaries:
            lengths.extend([4096,4097,8191,8192,8193])
        for length in lengths:
            sid,_=client.start(f"boundary-{length}",prompt(length),3)
            delta=prompt(5,22000)
            prefix=client.sessions[sid]["ids"]+delta
            resumed=client.append(sid,delta,1)
            client.close(sid)
            cold=client.fresh(f"boundary-{length}-cold",prefix,1)
            record({"name":f"boundary-{length}","initial_prompt_tokens":length,
                    "prefix_tokens":len(prefix),"comparison":compare(resumed,cold)})
        summary["status"]="completed"
        summary["interpretation"]=("All API/ledger contracts passed. Inspect numeric comparisons alongside cold-repeat controls and engine cache/work events; completion alone is not proof of cache-state parity.")
    except BaseException as error:
        summary["status"]="aborted"
        summary["error"]={"type":type(error).__name__,"message":str(error),"traceback":traceback.format_exc()}
        print(json.dumps({"status":"aborted","error":str(error)}),flush=True)
        raise
    finally:
        summary["cleanup_errors"]=client.cleanup()
        try:
            final_health=client.call("GET","/arc3/persistent/health")
            write_json(output/"health-after.json",final_health)
            summary["owned_sessions_not_closed"]={sid:state for sid,state in final_health.get("sessions",{}).items()
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
    parser.add_argument("--smoke-only",action="store_true")
    parser.add_argument("--extended-boundaries",action="store_true")
    args=parser.parse_args()
    try:
        summary=run(args)
    except BaseException:
        return 1
    if summary.get("cleanup_errors") or summary.get("owned_sessions_not_closed") or summary.get("health_after_error"):
        return 2
    return 0


if __name__=="__main__":
    sys.exit(main())
