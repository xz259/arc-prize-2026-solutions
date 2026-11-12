"""Bounded three-session persistent correctness gate, not a throughput test.

Five observation rounds force suffix and whole-interaction evictions. Then one
session closes, a new owner reuses its worker slot, and the other two continue.
The identical fixed external inputs can be run under eager and CUDA graphs.
Concurrent batching is observed, not inferred from simultaneous HTTP requests.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import fcntl
import hashlib
import json
from pathlib import Path
import re
import socket
import threading
import time
import traceback
import uuid

from raw_gpu_checks import Client, ContractFailure, prompt_factory, write_json
from raw_history_eviction_checks import HistoryClient, MODE, audit_events, engine_for
from raw_managed_eviction_checks import LOCKS

ROUNDS = 5
INPUT_TOKENS = 2048
OUTPUT_TOKENS = 128
BUDGET = 8192
PREFIX = 128
SUFFIX = [1401,1914]


def plan_description():
    return dict(mode=MODE,max_seconds=300,cleanup_reserve_seconds=25,
        resident_sessions=3,total_distinct_sessions=4,rounds_before_slot_reuse=ROUNDS,
        rounds_after_slot_reuse=ROUNDS,operation_count=30,
        input_tokens_per_observation=INPUT_TOKENS,output_tokens_per_operation=OUTPUT_TOKENS,
        maximum_emitted_tokens=30*OUTPUT_TOKENS,resident_budget=BUDGET,
        fixed_prefix_tokens=PREFIX,managed_suffix_local_range=SUFFIX,
        expected_whole_interaction_evictions=4,expected_suffix_evictions=16,
        expected_graph_batch=dict(mode='FULL',real_requests=3,padded_requests=4),
        no_claims=['throughput','gameplay quality','exact outputs across different concurrent schedules'])


def identity_check(args):
    """Verify the caller-pinned launch and live Linux process without mutation."""
    if socket.gethostname()!=args.expected_hostname:
        raise ContractFailure('Unexpected host')
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if boot!=args.expected_boot_id:
        raise ContractFailure('Unexpected boot identity')
    saved=json.loads(Path(args.candidate_identity).read_text())
    current=json.loads(Path(args.server_identity).read_text())
    if saved['root']!=current['root'] or saved.get('boot_id')!=boot:
        raise ContractFailure('Active server differs from the pinned candidate identity')
    root=saved['root'];proc=Path('/proc')/str(root['pid'])
    fields=(proc/'stat').read_text().rsplit(')',1)[1].strip().split()
    # vLLM changes process titles after launch. Kernel start time, process group
    # and session remain the owned-process identity; cmdline hash is diagnostic.
    if (fields[0]=='Z' or int(fields[19])!=root['start_ticks']
            or int(fields[2])!=root['pgid'] or int(fields[3])!=root['sid']):
        raise ContractFailure('Server process identity changed')
    config=saved.get('persistent_experiment',{}).get('optimization_config',{})
    if (bool(config.get('eager'))!=(args.expected_mode=='eager')
            or config.get('graph_mapping')!=1
            or config.get('verify_bytes')!=1 or config.get('verify_allocator')!=1):
        raise ContractFailure('Require explicitly matched graph mapping and both correctness proofs')
    argv=saved.get('argv',[])
    if ('--enforce-eager' in argv)!=(args.expected_mode=='eager'):
        raise ContractFailure('Actual launch command has the wrong eager/graph mode')
    return saved


def graph_status(client,expected_mode):
    response=client.call('POST','/collective_rpc',
        dict(method='arc3_eviction_graph_status',args=[],timeout=20))
    rows=response.get('results',[])
    if len(rows)!=1:
        raise ContractFailure('Expected one TP1 worker graph diagnostic')
    state=rows[0]
    if (state.get('graph_mapping') is not True or state.get('trace_enabled') is not True
            or state.get('enforce_eager')!=(expected_mode=='eager')):
        raise ContractFailure('Worker graph mapping/trace/mode differs from requested control')
    if expected_mode=='graph':
        if state.get('cudagraph_mode')!='FULL_AND_PIECEWISE' or 4 not in state.get('captured_token_counts',[]):
            raise ContractFailure('Require actual FULL_AND_PIECEWISE capture including size four')
    elif state.get('cudagraph_mode')!='NONE':
        raise ContractFailure('Expected eager NONE mode')
    return state


def slot_for(state,sid):
    pattern=re.compile(re.escape('arc3-persistent-'+sid)+r'(?:-[0-9a-f]{8})?\Z')
    matches=[v for k,v in state['request_slots'].items() if pattern.fullmatch(k)]
    if len(matches)!=1:
        raise ContractFailure('Expected one resident worker slot for '+sid)
    return matches[0]


def batch_delta(before,after,mode):
    key=(('FULL',3,4) if mode=='graph' else ('NONE',3,3))
    def count(state):
        return sum(r['count'] for r in state['prepared_batches']
            if (r['mode'],r['real_requests'],r['padded_requests'])==key)
    return dict(mode=key[0],real_requests=3,padded_requests=key[2],count=count(after)-count(before))


def observation_input(prompt,lane,index):
    # Fixed by lane/round rather than runtime-generated IDs or completion order.
    delta=prompt(INPUT_TOKENS,(lane*3001+index*1103)%20000)
    if len(delta)!=INPUT_TOKENS:
        raise ContractFailure('Tokenizer did not provide the exact planned input length')
    return delta


def parallel_round(jobs):
    """Each job owns a separate client/evidence directory; failures join safely."""
    barrier=threading.Barrier(len(jobs))
    def start(job):
        barrier.wait(timeout=10)
        client,name,delta,observation=job
        return client.submit(name,delta,observation=observation,
                             suffix=SUFFIX if observation else None)
    with ThreadPoolExecutor(max_workers=len(jobs)) as pool:
        futures=[pool.submit(start,job) for job in jobs]
        # Inspect every result, even when an earlier sibling fails.
        outcomes=[];failures=[]
        for future in futures:
            try:outcomes.append(future.result())
            except BaseException as error:failures.append(repr(error))
        if failures:
            raise ContractFailure('Concurrent round failed: '+str(failures))
    return outcomes


def exercise(clients,prompt,*,on_round=None,on_reuse=None):
    a,b,c,d=clients
    for index in range(ROUNDS):
        jobs=[(client,f'initial-round-{index+1}',observation_input(prompt,lane,index),
               (f'obs-{index+1:02d}',PREFIX if index==0 else 0))
              for lane,client in enumerate((a,b,c))]
        parallel_round(jobs)
        if on_round:on_round('before-reuse',index+1)
    # The first history eviction must occur after three completed observations;
    # the fourth is still active immediately before the fifth is appended.
    for client in (a,b,c):
        triggered=[t for t in client.turns if t['history_eviction']['triggered']]
        if len(triggered)!=1 or triggered[0]['turn']!=4:
            raise ContractFailure('Whole-history pressure did not begin on observation five')
    if on_reuse:on_reuse('before-close',b,d)
    b.close(b.sid)
    if on_reuse:on_reuse('after-close',b,d)
    for index in range(ROUNDS):
        parallel_round([
            (a,f'continuation-{index+1}',[],None),
            (d,f'replacement-round-{index+1}',observation_input(prompt,3,index),
             (f'obs-{index+1:02d}',PREFIX if index==0 else 0)),
            (c,f'continuation-{index+1}',[],None)])
        if index==0 and on_reuse:on_reuse('after-replacement',b,d)
        if on_round:on_round('after-reuse',index+1)
    if not any(t['history_eviction']['triggered'] for t in d.turns):
        raise ContractFailure('Replacement owner never exercised whole-history eviction')


def run(args):
    identity=identity_check(args)
    for name in (args.engine_events,args.service_events):
        if not Path(name).is_file():raise ContractFailure('Missing authoritative audit '+name)
    output=Path(args.output).resolve();output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();deadline=started+args.max_seconds-25
    tag='concurrent-eviction-'+uuid.uuid4().hex[:12]
    supervisor_dir=output/'supervisor';supervisor_dir.mkdir()
    supervisor=HistoryClient(args.base_url,supervisor_dir,args.timeout,tag+'-supervisor',deadline=deadline)
    clients=[]
    for lane in 'abcd':
        evidence=output/lane;evidence.mkdir()
        clients.append(HistoryClient(args.base_url,evidence,args.timeout,tag+'-'+lane,
            deadline=deadline,budget=BUDGET,output_tokens=OUTPUT_TOKENS))
    result=dict(status='starting',plan=plan_description(),expected_mode=args.expected_mode,
        started=time.time(),identity=identity,session_ids=[c.sid for c in clients],rounds=[])
    here=Path(__file__).resolve().parent
    result['client_source_sha256']={name:hashlib.sha256((here/name).read_bytes()).hexdigest()
        for name in (Path(__file__).name,'raw_history_eviction_checks.py',
                     'raw_managed_eviction_checks.py','raw_gpu_checks.py')}
    def save():
        result['turns']={c.sid:c.turns for c in clients}
        write_json(output/'results.json',result)
    save()
    with ExitStack() as stack:
        for name in LOCKS:
            stream=stack.enter_context((Path(args.locks_dir)/name).open('a+'))
            fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            health=supervisor.call('GET','/arc3/persistent/health')
            if (health.get('mode')!=MODE or health.get('max_context')!=40000
                    or health.get('max_sessions',0)<3 or health.get('prefix_caching') is not False
                    or any(s.get('state') not in ('closed','error') for s in health.get('sessions',{}).values())):
                raise ContractFailure('Wrong server mode or unrelated resident owner')
            result['graph_before']=graph_status(supervisor,args.expected_mode)
            from tokenizers import Tokenizer
            tokenizer=Path(args.tokenizer)
            result['tokenizer_sha256']=hashlib.sha256(tokenizer.read_bytes()).hexdigest()
            prompt=prompt_factory(Tokenizer.from_file(str(tokenizer)))
            def on_round(phase,index):
                result['rounds'].append(dict(phase=phase,index=index,elapsed=time.monotonic()-started))
                result['status']='running';save()
            def on_reuse(phase,old,new):
                state=graph_status(supervisor,args.expected_mode)
                result.setdefault('slot_reuse',{})[phase]=state
                if phase=='before-close':result['old_worker_slot']=slot_for(state,old.sid)
                elif phase=='after-close':
                    if any(old.sid in key for key in state['request_slots']):
                        raise ContractFailure('Closed owner retains a worker slot')
                else:
                    result['new_worker_slot']=slot_for(state,new.sid)
                    if result['new_worker_slot']!=result['old_worker_slot']:
                        raise ContractFailure('Replacement did not reuse the released state slot')
                save()
            exercise(clients,prompt,on_round=on_round,on_reuse=on_reuse)
            result['graph_after']=graph_status(supervisor,args.expected_mode)
            result['observed_three_session_batches']=batch_delta(result['graph_before'],result['graph_after'],args.expected_mode)
            if result['observed_three_session_batches']['count']<=0:
                raise ContractFailure('Requested three-session execution shape was not observed')
            result['status']='completed_pending_audit'
        except BaseException as error:
            result.update(status='failed',error=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc()))
        finally:
            # No request is closed until all its round threads have joined. Each
            # client closes only its own IDs, including uncertain HTTP starts.
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures=[pool.submit(client.cleanup) for client in clients]
                result['cleanup_errors']=[]
                for client,future in zip(clients,futures):
                    try:result['cleanup_errors'].extend(future.result())
                    except BaseException as error:result['cleanup_errors'].append(dict(session_id=client.sid,error=repr(error)))
            if result['cleanup_errors']:result['status']='failed'
            try:
                final=supervisor.call('GET','/arc3/persistent/health')
                result['final_owned_states']={c.sid:final.get('sessions',{}).get(c.sid) for c in clients}
                for client in clients:
                    state=result['final_owned_states'][client.sid]
                    if client.turns and (state or {}).get('state')!='closed':
                        raise ContractFailure('Owned completed session was not closed')
                result['audits']={c.sid:audit_events(args.engine_events,args.service_events,c.sid,c.turns)
                                  for c in clients if c.turns}
                if result['status']=='completed_pending_audit':
                    if len(result['audits'])!=4 or not all(a['valid'] for a in result['audits'].values()):
                        raise ContractFailure('Exact no-refill/byte/page audit failed')
                    result['status']='passed'
                for c in clients:
                    engine=engine_for(args.engine_events,c.sid)
                    (output/(c.sid+'-engine.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in engine))
                service=[json.loads(line) for line in Path(args.service_events).read_text().splitlines() if line]
                owned={c.sid for c in clients}
                (output/'owned-service.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in service if r.get('session_id') in owned))
            except BaseException as error:
                result.update(status='failed',final_audit_error=repr(error))
            result.update(ended=time.time(),elapsed=time.monotonic()-started,
                          completed_output_tokens=sum(t['output_tokens'] for c in clients for t in c.turns))
            save()
    return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true');parser.add_argument('--output')
    parser.add_argument('--base-url',default='http://127.0.0.1:1234')
    parser.add_argument('--expected-mode',choices=('eager','graph'))
    parser.add_argument('--expected-hostname');parser.add_argument('--expected-boot-id')
    parser.add_argument('--candidate-identity');parser.add_argument('--server-identity',default='/workspace/arc3-inference/server-identity.json')
    parser.add_argument('--engine-events');parser.add_argument('--service-events')
    parser.add_argument('--locks-dir',default='/workspace/arc3-inference/runs')
    parser.add_argument('--tokenizer',default='/workspace/models/Qwen3.8-Flash-Next-NVFP4/tokenizer.json')
    parser.add_argument('--max-seconds',type=float,default=300);parser.add_argument('--timeout',type=float,default=90)
    args=parser.parse_args(argv)
    if not args.execute:
        print(json.dumps(plan_description(),indent=2));return 0
    for key in ('output','expected_mode','expected_hostname','expected_boot_id','candidate_identity','engine_events','service_events'):
        if getattr(args,key) is None:parser.error('Missing explicit '+key)
    if not 60<=args.max_seconds<=300 or not 0<args.timeout<=90:
        parser.error('Require 60..300 seconds with per-call timeout <=90')
    result=run(args)
    print(json.dumps({k:result.get(k) for k in ('status','elapsed','completed_output_tokens','observed_three_session_batches')},indent=2))
    return 0 if result['status']=='passed' else 1


if __name__=='__main__':
    raise SystemExit(main())
