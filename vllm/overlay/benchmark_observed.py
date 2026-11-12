"""Bounded near-40K synthetic scaling client; no network without --execute.

Two graph-counter snapshots are taken outside the measured interval. They
count prepared nonzero batches, not completed GPU work or zero-token calls.

One arm per invocation. Launch/controller ownership remains external. Records
exact token IDs and keeps request evidence buffered during measurement so the
persistent volume is not on the per-turn critical path. All evidence flushes
before exit, including failures. No model code, game code, or runtime edits.
"""
from __future__ import annotations

import argparse
import asyncio
from contextlib import ExitStack
from dataclasses import dataclass,field
import fcntl
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import sys
import time
import traceback
from urllib.parse import urlparse
import uuid

from analyze import metric_values,summarize,audit_owned,boundary_guards

CONTEXT=40000
INPUT=36000
OUTPUT=3999
MODE='persistent-history-eviction-mtp0'
LOCKS=('public-eval-campaign.lock','mtp-precision-sweep.lock','bf16-gameplay-ablation.lock',
       'mtp0-memory-diagnostic.lock','persistent-eviction-scaling.lock')


class BenchmarkFailure(RuntimeError):pass


def digest(ids):
    return hashlib.sha256(json.dumps(list(ids),separators=(',',':')).encode()).hexdigest()


def save(path,value):
    path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2)+'\n');temporary.replace(path)


def sampling(count,lane):
    return dict(temperature=1.,top_p=.95,top_k=20,min_p=0.,repetition_penalty=1.,
                presence_penalty=0.,frequency_penalty=0.,seed=9300+lane,
                max_tokens=count,ignore_eos=True)


def seed_steps(arm):
    if arm=='eviction':
        return [(4127,1,128),*[(3999,1,0) for _ in range(8)],(3870,1,0)]
    if arm in ('persistent','persistent-api'):return [(35999,1,None)]
    return []


def plan(arm,concurrency,cycles):
    measured=OUTPUT*concurrency*(cycles if arm=='eviction' else 1)
    setup=sum(out for _,out,_ in seed_steps(arm))*concurrency
    return dict(arm=arm,concurrency=concurrency,context_limit=CONTEXT,
                measured_context_band=[36000,39999],measured_output_tokens=measured,
                setup_output_tokens=setup,maximum_output_tokens=measured+setup,
                measured_operations_per_lane=cycles if arm=='eviction' else 4 if arm=='persistent-api' else 1,
                setup_steps_per_lane=len(seed_steps(arm)),history_evictions=cycles*concurrency if arm=='eviction' else 0,
                policy_fraction=.10,synthetic_completed_block_tokens=4000,
                note='Four-aligned synthetic blocks are a controlled case; real interaction fragmentation is not modeled. Stateless inclusive time includes initial prefill; persistent arms exclude the shared setup barrier.')


@dataclass
class Ledger:
    sid:str
    lane:int
    eviction:bool=False
    logical:list=field(default_factory=list)
    holes:set=field(default_factory=set)
    observations:list=field(default_factory=list)
    turn:int=0
    eviction_revision:int=0

    def prepare(self,delta,count,observation=None):
        before=len(self.logical)-len(self.holes);removed=set();selected=[]
        triggered=before+len(delta)>=CONTEXT
        target=math.ceil(before/10) if triggered and self.eviction else 0
        required=max(target,before+len(delta)+1-CONTEXT) if triggered else 0
        if triggered:
            if not self.eviction:raise BenchmarkFailure('No-eviction arm reached resident capacity')
            for (oid,start),(_,end) in zip(self.observations,self.observations[1:]):
                points=set(range(start,end))-self.holes
                if not points or max(points)>=len(self.logical)-4:continue
                removed.update(points);selected.append(oid)
                if len(removed)>=required:break
            if len(removed)<required:raise BenchmarkFailure('Insufficient completed synthetic interactions')
        input_tokens=before-len(removed)+len(delta)
        if count<1 or input_tokens+count>CONTEXT:raise BenchmarkFailure('Requested output exceeds resident capacity')
        raw=dict(token_ids=list(delta),suffix_records=[],evict_observation_ids=[],observation_starts=[])
        if observation is not None:
            oid,start=observation
            raw['suffix_records']=[dict(observation_id=oid,local_token_range=None)]
            raw['observation_starts']=[dict(observation_id=oid,local_start_token=start)]
        body=dict(request_id=f'{self.sid}-turn-{self.turn}',raw=raw,sampling=sampling(count,self.lane))
        if self.turn==0:
            body.update(session_id=self.sid,history_eviction=dict(policy='oldest_interactions' if self.eviction else 'off',
                                                                  fraction=.10,resident_token_budget=CONTEXT))
            route='/arc3/persistent/start'
        else:
            body.update(expected_revision=self.turn,expected_prefix_sha256=digest(self.logical),continuation=not delta)
            route=f'/arc3/persistent/{self.sid}/append'
        return dict(body=body,route=route,delta=list(delta),count=count,observation=observation,
                    removed=removed,selected=selected,resident_before=before,resident_input=input_tokens,
                    triggered=triggered,target=target,required=required)

    def accept(self,prepared,response):
        state=response['persistent'];ids=state['token_ids'];count=prepared['count'];old=list(self.logical)
        if (len(ids)!=count or any(type(t) is not int or t<0 for t in ids)
                or response['choices'][0].get('token_ids')!=ids
                or state.get('engine_finish_reason')!='length'):
            raise BenchmarkFailure('Expected exact capped raw output token IDs')
        removed=prepared['removed'];delta=prepared['delta'];new=[*old,*delta,*ids]
        holes=self.holes|removed;resident=[t for i,t in enumerate(new) if i not in holes]
        expected=dict(session_id=self.sid,revision=self.turn+1,resident_tokens=len(resident),
                      logical_tokens=len(new),logical_input_tokens=len(old)+len(delta),
                      appended_prompt_tokens=len(delta),prefix_sha256=digest(new),
                      evicted_tokens=len(removed),eviction_revision=self.eviction_revision+bool(removed),
                      prefix_caching=False,eviction=True)
        for key,value in expected.items():
            if state.get(key)!=value:raise BenchmarkFailure(f'Persistent receipt {key}: {state.get(key)!r} != {value!r}')
        usage=dict(prompt_tokens=prepared['resident_input'],completion_tokens=count,total_tokens=len(resident))
        if response.get('usage')!=usage:raise BenchmarkFailure('Usage disagrees with exact resident input/output counts')
        history=state.get('history_eviction',{})
        hp=dict(policy='oldest_interactions' if self.eviction else 'off',fraction=.10,
                resident_token_budget=CONTEXT,triggered=prepared['triggered'],
                resident_tokens_before=prepared['resident_before'],resident_tokens_after_suffix=prepared['resident_before'],
                incoming_tokens=len(delta),target_tokens=prepared['target'],required_tokens=prepared['required'],
                history_removed_tokens=len(removed),suffix_removed_tokens=0,
                selected_block_ids=prepared['selected'],resident_tokens_after_input=prepared['resident_input'])
        if any(history.get(k)!=v for k,v in hp.items()):
            raise BenchmarkFailure('Independent history receipt mismatch: '+str({k:(history.get(k),v) for k,v in hp.items() if history.get(k)!=v}))
        operation=dict(session_id=self.sid,lane=self.lane,request_id=prepared['body']['request_id'],turn=self.turn,
            delta_tokens=len(delta),output_tokens=len(ids),delta_sha256=digest(delta),output_sha256=digest(ids),
            logical_before=len(old),logical_before_sha256=digest(old),logical_tokens=len(new),logical_sha256=digest(new),
            resident_input_tokens=prepared['resident_input'],resident_tokens=len(resident),resident_sha256=digest(resident),
            computed_prefix_sha256=digest(resident[:-1]),removed_tokens=len(removed),history_eviction=history,
            expected_prefill_tokens=len(delta)+(self.turn!=0),
            expected_resume_computed=0 if self.turn==0 else prepared['resident_input']-len(delta)-1)
        self.logical=new;self.holes=holes
        if prepared['observation'] is not None:
            oid,start=prepared['observation'];self.observations.append((oid,len(old)+start))
        self.turn+=1;self.eviction_revision+=bool(removed)
        return operation


def verify_identity(args):
    if socket.gethostname()!=args.expected_hostname:raise BenchmarkFailure('Host identity mismatch')
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if boot!=args.expected_boot_id:raise BenchmarkFailure('Boot identity mismatch')
    saved=json.loads(Path(args.candidate_identity).read_text());current=json.loads(Path(args.server_identity).read_text())
    if saved['root']!=current['root'] or saved.get('boot_id')!=boot:
        raise BenchmarkFailure('Active server differs from pinned candidate')
    root=saved['root'];fields=(Path('/proc')/str(root['pid'])/'stat').read_text().rsplit(')',1)[1].strip().split()
    if (fields[0]=='Z' or int(fields[19])!=root['start_ticks'] or int(fields[2])!=root['pgid'] or int(fields[3])!=root['sid']):
        raise BenchmarkFailure('Pinned server process no longer matches')
    argv=saved['argv'];config=saved.get('persistent_experiment',{}).get('optimization_config',{})
    for flag,value in (('--max-model-len','40000'),('--max-num-batched-tokens','8192'),('--gpu-memory-utilization',str(args.utilization)),
                       ('--max-num-seqs','30'),('--max-cudagraph-capture-size','32')):
        if flag not in argv or argv[argv.index(flag)+1]!=value:raise BenchmarkFailure('Unexpected launch '+flag)
    capture_sizes=[1,2,4,8,16,24,25,30,32]
    if '--cudagraph-capture-sizes' not in argv:raise BenchmarkFailure('Missing explicit capture sizes')
    capture_index=argv.index('--cudagraph-capture-sizes')+1
    actual=[]
    for value in argv[capture_index:]:
        if value.startswith('--'):break
        actual.append(int(value))
    if actual!=capture_sizes or config.get('capture_sizes')!=capture_sizes:
        raise BenchmarkFailure('Unmatched CUDA graph capture sizes')
    if '--speculative-config' in argv or '--no-enable-prefix-caching' not in argv:raise BenchmarkFailure('Require MTP0 and no prefix cache')
    if ('--enforce-eager' in argv)!=(args.expected_mode=='eager'):raise BenchmarkFailure('Wrong graph/eager launch')
    if saved.get('environment',{}).get('ARC3_QSA_KV_FORMAT')!='nvfp4':raise BenchmarkFailure('Require pinned NVFP4 cache')
    if (config.get('verify_bytes')!=args.expected_proofs or config.get('verify_allocator')!=args.expected_proofs
            or config.get('graph_mapping')!=1 or config.get('max_sessions',0)<args.concurrency):
        raise BenchmarkFailure('Wrong persistent admission, graph mapping, or proof profile')
    code=Path(args.candidate_code).resolve()
    if code!=Path(saved['persistent_experiment']['directory']).resolve():raise BenchmarkFailure('Helper directory is not the served candidate')
    manifest=saved['persistent_experiment']['manifest']
    for name in ('raw_gpu_checks.py','engine_patch.py','engine_eviction.py','eviction_storage.py','session_service.py','persistent_endpoint.py'):
        if hashlib.sha256((code/name).read_bytes()).hexdigest()!=manifest.get(name):raise BenchmarkFailure('Served helper/source drift: '+name)
    return saved


def load_prompt(args):
    path=Path(args.candidate_code)/'raw_gpu_checks.py'
    spec=importlib.util.spec_from_file_location('arc3_scaling_frozen_raw_helpers',path)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    from tokenizers import Tokenizer
    source=module.prompt_factory(Tokenizer.from_file(args.tokenizer))(25000)
    if len(source)!=25000:raise BenchmarkFailure('Frozen helper did not yield the exact token source')
    def prompt(length,lane=0,part=0):
        offset=(lane*97+part*1223)%len(source)
        return (source*((offset+length+len(source)-1)//len(source)))[offset:offset+length]
    return prompt


async def capture_graph_state(request,expected_mode):
    """Read one TP1 worker snapshot outside timing; no per-token instrumentation."""
    response=await request('POST','/collective_rpc',
        dict(method='arc3_eviction_graph_status',args=[],timeout=20))
    workers=response.get('results',[])
    if len(workers)!=1 or not isinstance(workers[0],dict):
        raise BenchmarkFailure('Expected one TP1 graph-state response')
    state=workers[0]
    if (state.get('graph_mapping') is not True or state.get('trace_enabled') is not True
            or state.get('enforce_eager')!=(expected_mode=='eager')):
        raise BenchmarkFailure('Unexpected graph mapping, trace, or execution mode')
    counts=state.get('prepared_batches')
    if not isinstance(counts,list):raise BenchmarkFailure('Missing prepared-batch counters')
    keys=set()
    for row in counts:
        if (not isinstance(row,dict) or not isinstance(row.get('mode'),str)
                or any(type(row.get(k)) is not int for k in ('real_requests','padded_requests','count'))
                or row['real_requests']<1 or row['padded_requests']<row['real_requests'] or row['count']<0):
            raise BenchmarkFailure('Invalid prepared-batch counter')
        key=row['mode'],row['real_requests'],row['padded_requests']
        if key in keys:raise BenchmarkFailure('Duplicate prepared-batch counter')
        keys.add(key)
    return state


async def join_owned(coroutines):
    tasks=[asyncio.create_task(c) for c in coroutines]
    try:return await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:task.cancel()
        await asyncio.gather(*tasks,return_exceptions=True)
        raise


async def run(args):
    identity=verify_identity(args)
    for path in (args.engine_events,args.service_events):
        if not Path(path).is_file():raise BenchmarkFailure('Required event audit missing: '+path)
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    origin=time.monotonic();hard_deadline=origin+args.max_seconds;deadline=hard_deadline-45
    tag='eviction-scale-'+uuid.uuid4().hex[:12]
    ledgers=[Ledger(f'{tag}-lane-{i:02d}',i,args.arm=='eviction') for i in range(args.concurrency)]
    owned=set();http_records=[];operations=[];samples=[];sampler_errors=[]
    result=dict(status='starting',config=vars(args),plan=plan(args.arm,args.concurrency,args.cycles),
                started_epoch=time.time(),identity=identity,
                session_ids=[ledger.sid for ledger in ledgers] if args.arm!='stateless' else [],
                stateless_request_ids=[f'{tag}-stateless-{i:02d}' for i in range(args.concurrency)] if args.arm=='stateless' else [],
                operations=operations,
                source_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                               for name in (Path(__file__).name,'analyze.py')})
    stop=asyncio.Event();measurement_active=False
    def state(stage):
        save(out/'state.json',dict(stage=stage,complete=stage in ('passed','failed'),
            started_epoch=result['started_epoch'],elapsed_seconds=time.monotonic()-origin,
            completed_operations=len(operations),received_tokens=sum(r['output_tokens'] for r in operations),
            deadline_epoch=result['started_epoch']+args.max_seconds,owned_session_ids=sorted(owned),
            planned_session_ids=[ledger.sid for ledger in ledgers] if args.arm!='stateless' else []))
    save(out/'results.json',result);state('starting')
    import httpx
    timeout=httpx.Timeout(args.timeout,connect=15)
    async with httpx.AsyncClient(timeout=timeout,limits=httpx.Limits(max_connections=args.concurrency+8,
                           max_keepalive_connections=args.concurrency+8),trust_env=False) as client:
        async def request(method,route,body=None,*,cleanup=False):
            remaining=(hard_deadline if cleanup else deadline)-time.monotonic()
            if remaining<=0:raise BenchmarkFailure('Bounded benchmark deadline reached')
            record=dict(method=method,route=route,payload=body,started_seconds=time.monotonic()-origin)
            http_records.append(record)
            try:
                response=await client.request(method,args.base_url+route,json=body,
                    timeout=min(args.timeout,20 if cleanup else remaining,remaining))
                record['status']=response.status_code
                try:value=response.json()
                except ValueError:value={'unparsed':response.text}
                record['response']=value;record['finished_seconds']=time.monotonic()-origin
                if not response.is_success:
                    if cleanup and response.status_code==404:return {'absent':True}
                    raise BenchmarkFailure(f'{method} {route}: HTTP {response.status_code}: {str(value)[:1200]}')
                return value
            except BaseException as exc:
                record['error']=repr(exc);record['finished_seconds']=time.monotonic()-origin;raise

        async def metrics():
            t0=time.monotonic();response=await client.get(args.base_url+'/metrics',timeout=10)
            response.raise_for_status();t1=time.monotonic()
            return dict(seconds=(t0+t1)/2-origin,epoch=time.time(),rtt_seconds=t1-t0,
                        metrics=metric_values(response.text))

        async def sample():
            last_progress=0
            while not stop.is_set():
                try:samples.append(await metrics())
                except Exception as exc:sampler_errors.append(repr(exc))
                if time.monotonic()-last_progress>=10:
                    state('measuring' if measurement_active else 'preparing');last_progress=time.monotonic()
                try:await asyncio.wait_for(stop.wait(),timeout=args.sample_interval)
                except asyncio.TimeoutError:pass

        async def submit(ledger,delta,count,phase,observation=None):
            cpu_start=time.monotonic();prepared=ledger.prepare(delta,count,observation)
            if ledger.turn==0:owned.add(ledger.sid)
            sent=time.monotonic()-origin
            response=await request('POST',prepared['route'],prepared['body'])
            received=time.monotonic()-origin;operation=ledger.accept(prepared,response)
            operation.update(phase=phase,start_seconds=sent,end_seconds=received,
                client_prepare_seconds=sent-(cpu_start-origin),client_verify_seconds=time.monotonic()-origin-received)
            operations.append(operation)
            return operation

        async def prepare_lane(ledger):
            for index,(count,output,start) in enumerate(seed_steps(args.arm)):
                observation=(f'obs-{index:02d}',start) if start is not None else None
                await submit(ledger,prompt(count,ledger.lane,index),output,'setup',observation)
            expected=39999 if args.arm=='eviction' else 36000
            if len(ledger.logical)-len(ledger.holes)!=expected:raise BenchmarkFailure('Synthetic seed has wrong resident count')

        async def persistent_lane(ledger):
            counts=[OUTPUT] if args.arm=='persistent' else [999,1000,1000,1000] if args.arm=='persistent-api' else [OUTPUT]*args.cycles
            for index,count in enumerate(counts):
                if args.arm=='eviction':
                    operation=await submit(ledger,prompt(1,ledger.lane,10+index),count,'measured',(f'cycle-{index:02d}',0))
                    if operation['removed_tokens']!=4000 or operation['resident_input_tokens']!=INPUT or operation['resident_tokens']!=39999:
                        raise BenchmarkFailure('Eviction cycle did not preserve the exact 36K..39,999 band')
                else:await submit(ledger,[],count,'measured')

        async def stateless_lane(lane):
            ids=prompt(INPUT,lane);rid=f'{tag}-stateless-{lane:02d}'
            body=dict(model=args.model,prompt=ids,add_special_tokens=False,request_id=rid,
                      **sampling(OUTPUT,lane),min_tokens=OUTPUT,stream=True,return_token_ids=True,
                      stream_options={'include_usage':True},cache_salt=rid)
            raw=dict(route='/v1/completions',method='POST',payload=body,started_seconds=time.monotonic()-origin,
                     token_ids=[],events=[],usage=None,finish_reason=None,response_id=None)
            http_records.append(raw)
            try:
                remaining=deadline-time.monotonic()
                if remaining<=0:raise BenchmarkFailure('Stateless deadline reached')
                async with client.stream('POST',args.base_url+'/v1/completions',json=body,
                                         timeout=min(args.timeout,remaining)) as response:
                    response.raise_for_status();done=False
                    async for line in response.aiter_lines():
                        if time.monotonic()>=deadline:raise BenchmarkFailure('Stateless streaming deadline reached')
                        if not line.startswith('data:'):continue
                        data=line[5:].strip()
                        if data=='[DONE]':done=True;break
                        item=json.loads(data)
                        if item.get('error'):raise BenchmarkFailure(str(item['error']))
                        raw['response_id']=item.get('id',raw['response_id'])
                        if item.get('usage') is not None:raw['usage']=item['usage']
                        for choice in item.get('choices',[]):
                            if choice.get('index')!=0:raise BenchmarkFailure('Unexpected completion choice')
                            emitted=choice.get('token_ids') or []
                            if emitted:
                                raw['token_ids'].extend(emitted)
                                raw['events'].append([time.monotonic()-origin,len(raw['token_ids'])])
                            if choice.get('finish_reason') is not None:raw['finish_reason']=choice['finish_reason']
                usage=raw['usage'] or {}
                if (not done or raw['finish_reason']!='length' or len(raw['token_ids'])!=OUTPUT
                        or usage.get('prompt_tokens')!=INPUT or usage.get('completion_tokens')!=OUTPUT
                        or (usage.get('prompt_tokens_details') or {}).get('cached_tokens',0)):
                    raise BenchmarkFailure('Stateless exact-count/SSE contract failed')
                raw['finished_seconds']=time.monotonic()-origin
                operations.append(dict(phase='measured',lane=lane,request_id=rid,output_tokens=OUTPUT,
                    delta_tokens=INPUT,expected_prefill_tokens=INPUT,resident_input_tokens=INPUT,resident_tokens=39999,
                    start_seconds=raw['started_seconds'],end_seconds=raw['finished_seconds'],
                    delta_sha256=digest(ids),output_sha256=digest(raw['token_ids'])))
            except BaseException as exc:raw['error']=repr(exc);raise

        sampler=None
        with ExitStack() as locks:
            for name in LOCKS:
                handle=locks.enter_context((Path(args.locks_dir)/name).open('a+'))
                fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
            try:
                health=await request('GET','/arc3/persistent/health')
                if (health.get('mode')!=MODE or health.get('max_context')!=CONTEXT
                        or health.get('max_sessions',0)<args.concurrency or health.get('prefix_caching') is not False
                        or any(s.get('state') not in ('closed','error') for s in health.get('sessions',{}).values())):
                    raise BenchmarkFailure('Wrong server mode/admission or unrelated resident owner')
                initial=await metrics()
                if initial['metrics']['num_requests_running'] or initial['metrics']['num_requests_waiting']:
                    raise BenchmarkFailure('Server is not idle')
                prompt=load_prompt(args)
                result['tokenizer_sha256']=hashlib.sha256(Path(args.tokenizer).read_bytes()).hexdigest()
                sampler=asyncio.create_task(sample())
                if args.arm!='stateless':
                    state('seeding');await join_owned(prepare_lane(ledger) for ledger in ledgers)
                result['graph_states']={'before_measurement':await capture_graph_state(request,args.expected_mode)}
                result['graph_counter_scope']='Prepared nonzero attention batches; zero-token worker calls return earlier and are not counted. Snapshots outside measured time; no per-step timestamps or completed-forward proof.'
                before=await metrics();samples.append(before)
                result['setup_finished_seconds']=time.monotonic()-origin
                measurement_active=True;state('measuring')
                start=time.monotonic()-origin
                await join_owned((stateless_lane(i) for i in range(args.concurrency)) if args.arm=='stateless'
                                 else (persistent_lane(ledger) for ledger in ledgers))
                end=time.monotonic()-origin;after=await metrics();samples.append(after)
                result['measurement']=dict(start_seconds=start,end_seconds=end,
                    metrics_before=before['metrics'],metrics_after=after['metrics'])
                result['status']='completed_pending_audit';measurement_active=False
                result['graph_states']['after_measurement']=await capture_graph_state(request,args.expected_mode)
            except BaseException as exc:
                result.update(status='failed',error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
            finally:
                measurement_active=False;state('cleaning');result['cleanup_errors']=[]
                async def close_owned(sid):
                    try:await request('POST',f'/arc3/persistent/{sid}/close',{},cleanup=True)
                    except BaseException as exc:result['cleanup_errors'].append(dict(session_id=sid,error=repr(exc)))
                await asyncio.gather(*(close_owned(sid) for sid in sorted(owned)))
                try:
                    verify_identity(args)
                    final=await request('GET','/arc3/persistent/health',cleanup=True)
                    result['owned_final_states']={sid:final.get('sessions',{}).get(sid) for sid in sorted(owned)}
                    if any(value is not None and value.get('state')!='closed' for value in result['owned_final_states'].values()):
                        raise BenchmarkFailure('An owned persistent session did not close')
                    # Cancellation of owned stateless HTTP streams aborts their
                    # vLLM generators. Initial idle + held campaign locks make
                    # an idle postcondition an independent drain check.
                    drained=False
                    while time.monotonic()<hard_deadline:
                        last=await metrics();samples.append(last)
                        if last['metrics']['num_requests_running']==last['metrics']['num_requests_waiting']==0:
                            drained=True;break
                        await asyncio.sleep(.2)
                    if not drained:raise BenchmarkFailure('Engine did not drain after owned cleanup')
                    result['engine_drained_after_cleanup']=True
                except BaseException as exc:result['cleanup_errors'].append(dict(error=repr(exc)))
                stop.set()
                if sampler is not None:await asyncio.gather(sampler,return_exceptions=True)
                result['actually_attempted_session_ids']=sorted(owned);result['sampler_errors']=sampler_errors
                result['ended_epoch']=time.time();result['operations']=operations
                try:
                    engine=[json.loads(line) for line in Path(args.engine_events).read_text().splitlines() if line]
                    service=[json.loads(line) for line in Path(args.service_events).read_text().splitlines() if line]
                    patterns=[re.compile(re.escape('arc3-persistent-'+sid)+r'(?:-[0-9a-f]{8})?\Z') for sid in owned]
                    engine=[r for r in engine if any(p.fullmatch(r.get('request_id','')) for p in patterns)]
                    service=[r for r in service if r.get('session_id') in owned]
                    (out/'engine-owned.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in engine))
                    (out/'service-owned.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in service))
                    if args.arm!='stateless':
                        result['audit']=audit_owned(engine,service,operations,sorted(owned),expected_proofs=bool(args.expected_proofs))
                        result['decode_boundary_guards']=boundary_guards(engine,operations,epoch_origin=result['started_epoch'])
                        if not result['audit']['valid']:result['status']='failed'
                except BaseException as exc:result['audit_error']=repr(exc);result['status']='failed'
                result['summary']=summarize(result,samples)
                expected=result['plan']['maximum_output_tokens']
                if (result['cleanup_errors'] or sum(r['output_tokens'] for r in operations)!=expected
                        or not result['summary']['valid']):result['status']='failed'
                elif result['status']=='completed_pending_audit':result['status']='passed'
                # Buffered raw evidence is flushed after measurement/cleanup.
                httpdir=out/'http';httpdir.mkdir()
                for index,record in enumerate(http_records):save(httpdir/f'{index:04d}.json',record)
                (out/'metrics.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in samples))
                save(out/'results.json',result);state(result['status'])
    return result


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true');p.add_argument('--arm',choices=('stateless','persistent','persistent-api','eviction'),default='eviction')
    p.add_argument('--concurrency',type=int,default=25);p.add_argument('--cycles',type=int,default=3)
    p.add_argument('--output');p.add_argument('--base-url',default='http://127.0.0.1:1234')
    p.add_argument('--model',default='nvidia/Qwen3.8-Flash-Next-NVFP4')
    p.add_argument('--tokenizer',default='/workspace/models/Qwen3.8-Flash-Next-NVFP4/tokenizer.json')
    p.add_argument('--candidate-code');p.add_argument('--candidate-identity');p.add_argument('--server-identity',default='/workspace/arc3-inference/server-identity.json')
    p.add_argument('--expected-hostname');p.add_argument('--expected-boot-id')
    p.add_argument('--engine-events');p.add_argument('--service-events');p.add_argument('--locks-dir',default='/workspace/arc3-inference/runs')
    p.add_argument('--expected-mode',choices=('eager','graph'),default='graph');p.add_argument('--expected-proofs',type=int,choices=(0,1),default=0)
    p.add_argument('--utilization',type=float,default=.95);p.add_argument('--max-seconds',type=float,default=1500)
    p.add_argument('--timeout',type=float,default=900);p.add_argument('--sample-interval',type=float,default=.5)
    args=p.parse_args(argv)
    if not (1<=args.concurrency<=30 and 1<=args.cycles<=6 and 60<=args.max_seconds<=1500
            and 1<=args.timeout<=1200 and .25<=args.sample_interval<=2):p.error('Settings exceed bounded benchmark limits')
    parsed=urlparse(args.base_url)
    if parsed.scheme!='http' or parsed.hostname not in ('127.0.0.1','localhost','::1'):p.error('Loopback HTTP only')
    if not args.execute:print(json.dumps(plan(args.arm,args.concurrency,args.cycles),indent=2));return 0
    required=('output','candidate_code','candidate_identity','expected_hostname','expected_boot_id','engine_events','service_events')
    if any(getattr(args,k) is None for k in required):p.error('Execution requires fresh output, pinned candidate/source, host/boot, and event audits')
    async def invoke():
        loop=asyncio.get_running_loop();task=asyncio.current_task()
        for sig in (signal.SIGTERM,signal.SIGINT):loop.add_signal_handler(sig,task.cancel)
        return await run(args)
    try:
        result=asyncio.run(invoke());print(json.dumps({'status':result['status'],'summary':result['summary']},indent=2))
        return 0 if result['status']=='passed' else 1
    except BaseException as exc:
        print(json.dumps({'status':'failed','error':repr(exc)}));return 1


if __name__=='__main__':sys.exit(main())
