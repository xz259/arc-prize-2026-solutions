"""Bounded opt-in raw history-eviction diagnostic; dry run performs no IO.

Uses exact IDs from the pinned tokenizer, explicit diagnostic observation starts,
and an independent resident-position oracle. This does not exercise chat/image
rendering or execute model-written programs. Server ownership stays with caller.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import copy
import fcntl
import hashlib
import json
from pathlib import Path
import re
import socket
import time
import traceback
import uuid

from raw_gpu_checks import ContractFailure, HTTPFailure, prompt_factory, token_digest, write_json
from raw_managed_eviction_checks import ManagedClient, LOCKS

MODE='persistent-history-eviction-mtp0'
POLICY=dict(policy='oldest_interactions',fraction=.10,resident_token_budget=7000)
SOURCE_FILES=('raw_history_eviction_checks.py','raw_managed_eviction_checks.py','raw_gpu_checks.py',
              'history_eviction.py','session_service.py','persistent_endpoint.py','engine_eviction.py',
              'eviction_storage.py','eviction_runner.py','residency.py','suffix_registry.py')


def plan_description():
    return dict(mode=MODE, max_seconds=300, sessions=2, maximum_emitted_tokens=72,
        main_budget=7000, main_operations=7, history_deletions=3,
        cases=['initial observation after protected 128-token prefix; 257-token suffix',
               '17-token same-observation append',
               'second observation retires prior suffix and registers a 263-token suffix',
               'third observation forces oldest whole-block eviction across an earlier suffix hole',
               'fourth observation repeats capacity-triggered whole-block eviction',
               'same-observation append fills resident budget exactly with eight generated tokens',
               'empty continuation after length evicts the next complete block, preserving active block',
               'separate 1024-token session rejects oversized incoming observation while only active block exists',
               'eight-token empty continuation proves that rejected session remains usable'],
        exclusions=['real chat block-start proof','images','gameplay quality','throughput',
                    'fresh-shortened-prefill equivalence','independent auxiliary-state byte checks'])


def intervals(positions):
    out=[]
    for p in sorted(set(positions)):
        if out and out[-1][1]==p:out[-1][1]=p+1
        else:out.append([p,p+1])
    return out


class OracleCapacity(Exception):
    pass


class Oracle:
    """Independent set-of-logical-positions oracle; never imports policy code."""
    def __init__(self,budget=7000):
        self.budget=budget;self.logical=[];self.holes=set();self.observations=[];self.live={}
        self.eviction_revision=0;self.turn=0;self.registered_through=0

    @property
    def resident_positions(self):
        return [p for p in range(len(self.logical)) if p not in self.holes]

    def prepare(self,delta,*,observation=None,suffix=None):
        logical=len(self.logical);before=self.resident_positions
        retire=list(self.live) if observation else []
        suffix_positions={p for oid in retire for p in range(*self.live[oid])}
        after_suffix=[p for p in before if p not in suffix_positions]
        resident=len(after_suffix);triggered=resident+len(delta)>=self.budget
        target=(resident+9)//10 if triggered else 0
        capacity_needed=max(0,resident+len(delta)+1-self.budget)
        required=max(target,capacity_needed) if triggered else 0
        selected=[];history=set()
        eligible=[]
        for (oid,start),(_,end) in zip(self.observations,self.observations[1:]):
            points={p for p in after_suffix if start<=p<end}
            if points and all(p<logical-4 for p in points):
                eligible.append((oid,start,end,points))
        if triggered:
            for oid,start,end,points in eligible:
                selected.append(dict(observation_id=oid,logical_range=[start,end],resident_tokens=len(points)))
                history.update(points)
                if len(history)>=required:break
            if len(history)<required:raise OracleCapacity('insufficient_completed_history')
        removed=suffix_positions|history
        after=[p for p in before if p not in removed]
        if len(after)+len(delta)>=self.budget:raise OracleCapacity('no_output_capacity')
        prefix=self.observations[0][1] if self.observations else 0
        receipt=dict(policy='oldest_interactions',fraction=.10,resident_token_budget=self.budget,
            triggered=triggered,fraction_basis='resident_tokens_after_pending_suffix',
            resident_tokens_before=len(before),resident_tokens_after_suffix=resident,incoming_tokens=len(delta),
            target_tokens=target,required_tokens=required,capacity_needed_tokens=capacity_needed,
            history_removed_tokens=len(history),suffix_removed_tokens=len(suffix_positions),
            selected_block_ids=[r['observation_id'] for r in selected],selected_blocks=selected,
            protected_prefix_tokens=prefix,protected_prefix_range=[0,prefix],
            active_observation_id=self.observations[-1][0] if self.observations else None,
            protected_active_range=[self.observations[-1][1],logical] if self.observations else None,
            protected_tail_range=[max(0,logical-4),logical],
            total_removed_tokens=len(removed),resident_tokens_after=len(after),
            resident_tokens_after_input=len(after)+len(delta),minimum_output_capacity=self.budget-len(after)-len(delta),
            combined_ranges=intervals(removed),history_ranges=intervals(history),suffix_ranges=intervals(suffix_positions))
        return dict(before=before,after=after,removed=removed,retire=retire,receipt=receipt,
                    delta=list(delta),observation=observation,suffix=suffix,logical_before=logical)

    def commit(self,prepared,output):
        if prepared['logical_before']!=len(self.logical):raise AssertionError('stale oracle plan')
        delta=prepared['delta'];old=len(self.logical)
        self.holes.update(prepared['removed'])
        self.logical.extend(delta);self.registered_through=len(self.logical)
        observation=prepared['observation']
        if observation:
            oid,start=observation
            self.observations.append((oid,old+start))
            self.live={} if prepared['suffix'] is None else {oid:tuple(old+p for p in prepared['suffix'])}
        self.logical.extend(output)
        self.eviction_revision+=bool(prepared['removed']);self.turn+=1

    def history_state(self):
        return dict(revision=self.turn,registered_through=self.registered_through,
            protected_prefix_tokens=self.observations[0][1] if self.observations else 0,
            active_observation_id=self.observations[-1][0] if self.observations else None,
            blocks=[dict(observation_id=oid,start=start,end=self.observations[i+1][1] if i+1<len(self.observations) else None)
                    for i,(oid,start) in enumerate(self.observations)])


class HistoryClient(ManagedClient):
    def __init__(self,*args,budget=7000,output_tokens=8,**kwargs):
        if type(output_tokens) is not int or not 1 <= output_tokens <= 128:
            raise ValueError('Diagnostic output_tokens must be an integer in 1..128')
        super().__init__(*args,**kwargs);self.oracle=Oracle(budget)
        self.output_tokens=output_tokens

    def build(self,delta,*,observation=None,suffix=None,request_id=None):
        initial=self.oracle.turn==0
        records=[] if observation is None else [dict(observation_id=observation[0],local_token_range=suffix)]
        raw=dict(token_ids=list(delta),suffix_records=records,
                 evict_observation_ids=list(self.oracle.live) if observation else [],
                 observation_starts=[] if observation is None else [dict(observation_id=observation[0],local_start_token=observation[1])])
        body=dict(request_id=request_id or self.sid+f'-r{self.oracle.turn+1}',raw=raw,
                  sampling=dict(temperature=0,max_tokens=self.output_tokens,ignore_eos=True))
        if initial:body.update(session_id=self.sid,history_eviction=POLICY|dict(resident_token_budget=self.oracle.budget))
        else:body.update(expected_revision=self.oracle.turn,expected_prefix_sha256=token_digest(self.oracle.logical),continuation=not delta)
        return body,('/arc3/persistent/start' if initial else f'/arc3/persistent/{self.sid}/append')

    def submit(self,name,delta,*,observation=None,suffix=None):
        prepared=self.oracle.prepare(delta,observation=observation,suffix=suffix)
        old=list(self.oracle.logical);initial=self.oracle.turn==0
        body,route=self.build(delta,observation=observation,suffix=suffix)
        if initial:self.active.add(self.sid)
        response=self.call('POST',route,body);state=response['persistent'];output=state['token_ids']
        if len(output)!=self.output_tokens or state.get('engine_finish_reason')!='length':
            raise ContractFailure(f'Expected {self.output_tokens} capped raw output tokens')
        if response['choices'][0].get('token_ids')!=output:raise ContractFailure('Choice and receipt raw IDs disagree')
        actual=state.get('history_eviction',{})
        differences={k:dict(expected=v,actual=actual.get(k)) for k,v in prepared['receipt'].items() if actual.get(k)!=v}
        if differences:raise ContractFailure('History policy receipt mismatch: '+str(differences))
        self.oracle.commit(prepared,output)
        resident=[self.oracle.logical[p] for p in self.oracle.resident_positions]
        previous_resident=[old[p] for p in prepared['after']]
        expected=dict(session_id=self.sid,revision=self.oracle.turn,resident_tokens=len(resident),logical_tokens=len(self.oracle.logical),
            logical_input_tokens=len(old)+len(delta),appended_prompt_tokens=len(delta),prefix_sha256=token_digest(self.oracle.logical),
            evicted_tokens=len(prepared['removed']),eviction_revision=self.oracle.eviction_revision,prefix_caching=False,eviction=True,
            managed_suffixes=dict(registered_observation_ids=[] if observation is None else [observation[0]],
                                  evicted_observation_ids=prepared['retire']),history_state=self.oracle.history_state())
        for k,v in expected.items():
            if state.get(k)!=v:raise ContractFailure(f'{k}: expected {v}, got {state.get(k)}')
        if response['usage']!=dict(prompt_tokens=len(previous_resident)+len(delta),completion_tokens=self.output_tokens,total_tokens=len(resident)):
            raise ContractFailure('Usage differs from exact resident input/output')
        turn=dict(name=name,request_id=body['request_id'],turn=len(self.turns),initial=initial,
            delta_tokens=len(delta),output_tokens=self.output_tokens,delta_sha256=token_digest(delta),output_sha256=token_digest(output),
            logical_before=len(old),logical_before_sha256=token_digest(old),logical_tokens=len(self.oracle.logical),logical_sha256=token_digest(self.oracle.logical),
            resident_input_tokens=len(previous_resident)+len(delta),resident_tokens=len(resident),resident_sha256=token_digest(resident),
            computed_prefix_sha256=token_digest(resident[:-1]),resident_after_eviction=len(previous_resident),
            after_eviction_sha256=token_digest(previous_resident),removed_tokens=len(prepared['removed']),
            expected_prefill_tokens=len(delta)+(not initial),expected_resume_computed=0 if initial else len(previous_resident)-1,
            history_eviction=actual,history_state=state['history_state'],eviction_revision=self.oracle.eviction_revision)
        self.turns.append(turn);self.logical=list(self.oracle.logical)
        self.last_body,self.last_response=body,response
        return turn


def engine_for(path,sid):
    pattern=re.compile(re.escape('arc3-persistent-'+sid)+r'(?:-[0-9a-f]{8})?\Z')
    return [r for line in Path(path).read_text().splitlines() if line
            and pattern.fullmatch((r:=json.loads(line)).get('request_id',''))]


def audit_events(engine_path,service_path,sid,turns,*,expected_rejections=0):
    engine=engine_for(engine_path,sid)
    service=[r for line in Path(service_path).read_text().splitlines() if line and (r:=json.loads(line)).get('session_id')==sid]
    errors=[];checks={};copies=[]
    def check(ok,key,detail=None):
        checks[key]=checks.get(key,0)+1
        if not ok:errors.append(dict(check=key,detail=detail))
    def one(rows,key,detail):
        check(len(rows)==1,key,detail);return rows[0] if len(rows)==1 else {}
    check(len({r['request_id'] for r in engine})==1,'single_engine_owner')
    check(not any(r['event']=='failed' for r in service),'no_submitted_service_failure')
    check(sum(r['event']=='rejected' for r in service)==expected_rejections,'exact_expected_rejections')
    check(sum(r['event']=='submit' for r in service)==len(turns),'one_submission_per_turn_no_retry_work')
    check(sum(r['event']=='pause' for r in engine)==len(turns),'one_engine_pause_per_turn')
    check(sum(r['event']=='evict' for r in engine)==sum(bool(t['removed_tokens']) for t in turns),'exact_eviction_count')
    previous=None
    for t in turns:
        op=t['request_id'];i=t['turn']
        sub=one([r for r in service if r['event']=='submit' and r.get('operation_id')==op],'one_service_submit',i)
        sp=one([r for r in service if r['event']=='pause' and r.get('operation_id')==op],'one_service_pause',i)
        check(token_digest(sub.get('appended_token_ids',[]))==t['delta_sha256'] and sub.get('old_tokens')==t['logical_before']
              and sub.get('prefix_sha256')==t['logical_before_sha256'],'exact_service_delta_prefix',i)
        check(token_digest(sp.get('output_token_ids',[]))==t['output_sha256'] and sp.get('prefix_sha256')==t['logical_sha256'],
              'exact_service_emitted_ids',i)
        check(sub.get('history_eviction')==t['history_eviction'] and sub.get('history_state')==t['history_state'],'service_policy_and_block_registry_receipt',i)
        entry=one([r for r in engine if r['event']==('start' if t['initial'] else 'resume') and r.get('turn')==i],'one_engine_entry',i)
        ep=one([r for r in engine if r['event']=='pause' and r.get('turn')==i],'one_engine_pause',i)
        for k,v in dict(resident_tokens=t['resident_tokens'],tokens_sha256=t['resident_sha256'],computed_tokens=t['resident_tokens']-1,
                        computed_prefix_sha256=t['computed_prefix_sha256'],in_flight_tokens=0,output_placeholders=0,preemptions=0).items():
            check(ep.get(k)==v,'pause_'+k,i)
        check(ep.get('logical_eviction_state',{}).get('logical_sha256')==t['logical_sha256'],'original_logical_positions_preserved',i)
        decode=t['output_tokens']-1
        expected=dict(prefill_tokens=t['expected_prefill_tokens'],decode_tokens=decode,scheduled_tokens=t['expected_prefill_tokens']+decode,
                      resume_computed_tokens=t['expected_resume_computed'],minimum_scheduled_position=t['expected_resume_computed'])
        check(all(ep.get('turn_work',{}).get(k)==v for k,v in expected.items()),'only_new_input_and_pending_final_forwarded',i)
        check(entry.get('computed_tokens')==t['expected_resume_computed'],'resume_keeps_computed_frontier',i)
        anchor=previous
        if t['removed_tokens']:
            sc=one([r for r in service if r['event']=='evict_commit' and r.get('operation_id')==op],'one_service_eviction_commit',i)
            ack=sc.get('acknowledgement',{})
            ee=one([r for r in engine if r['event']=='evict' and r.get('eviction',{}).get('plan_digest')==ack.get('plan_digest')],'one_engine_eviction',i)
            check(ee.get('eviction')==ack,'exact_engine_service_ack',i)
            check(sc.get('history_eviction')==t['history_eviction'],'eviction_policy_receipt_unchanged',i)
            check(ee.get('resident_tokens')==t['resident_after_eviction'] and ee.get('tokens_sha256')==t['after_eviction_sha256']
                  and ee.get('computed_tokens')==t['resident_after_eviction']-1,'exact_surviving_resident_subsequence',i)
            check(ack.get('evicted_tokens')==t['removed_tokens'] and ack.get('logical_tokens')==t['logical_before']
                  and ack.get('eviction_revision')==t['eviction_revision'],'eviction_ack_counts',i)
            receipts=ack.get('worker_receipts',[])
            check(len(receipts)==1 and receipts[0].get('worker_synchronized') is True,'worker_synchronized',i)
            copy_rows=[item for r in receipts for item in r.get('copies',[])]
            check(bool(copy_rows),'actual_copy_receipts',i)
            check(bool(receipts) and all(r.get('byte_verification_enabled') is True for r in receipts),'retained_byte_verification_enabled',i)
            prepare=one([r for r in engine if r['event']=='evict_prepare'
                         and r.get('eviction_plan',{}).get('plan_digest')==ack.get('plan_digest')],'one_preflight_owner_plan',i)
            planned=[item for g in prepare.get('eviction_plan',{}).get('groups',[]) for item in g.get('copies',[])]
            check(Counter(r.get('kind') for r in planned)=={'main':12,'compressed':12},'all_twelve_main_and_descriptor_layers_planned',i)
            check(Counter(r.get('layer_name') for r in planned)==Counter(r.get('layer_name') for r in copy_rows),
                  'every_planned_cache_owner_has_copy_receipt',i)
            for item in copy_rows:
                proof=item.get('byte_verification') or {};before_hash=proof.get('before',{})
                owner=next((r for r in planned if r.get('layer_name')==item.get('layer_name')), {})
                check(proof.get('equal') is True and before_hash==proof.get('after')
                      and bool(re.fullmatch(r'[0-9a-f]{64}',str(before_hash.get('sha256',''))))
                      and type(before_hash.get('rows')) is int and before_hash['rows']>0
                      and type(before_hash.get('bytes')) is int and before_hash['bytes']>0,
                      'retained_byte_hash_equal',i)
                check(type(item.get('row_bytes')) is int and item['row_bytes']>0
                      and type(before_hash.get('rows')) is int
                      and before_hash['rows']==item.get('retained_rows')==owner.get('retained_rows')
                      and before_hash.get('bytes')==before_hash['rows']*item['row_bytes'],
                      'byte_proof_covers_all_planned_retained_rows',i)
            before=(previous or {}).get('block_ids',[]);after=ee.get('block_ids',[]);freed=ack.get('freed_blocks_by_group',[])
            check(len(before)==len(after)==len(freed) and all(set(a)-set(b)==set(f) and not set(b)-set(a)
                  for a,b,f in zip(before,after,freed)), 'released_pages_match_owner_change',i)
            flat_freed=[p for group in freed for p in group]
            allocation=ack.get('allocator_reacquisition') or {}
            check(allocation.get('kind')=='actual_allocator_reacquisition_without_forward','actual_allocator_receipt',i)
            if flat_freed:
                check(allocation.get('attempted') is True
                      and allocation.get('released_block_ids')==flat_freed
                      and allocation.get('reacquired_released_block_ids')==flat_freed
                      and allocation.get('reacquired_ref_cnt')==1
                      and allocation.get('free_count_before')==allocation.get('free_count_after')
                      and allocation.get('free_order_before_sha256')==allocation.get('free_order_after_sha256')
                      and all(allocation.get(k) is True for k in ('queue_order_restored','other_live_owners_unchanged','all_temporary_owners_released'))
                      and allocation.get('model_forward_performed') is False,
                      'freed_pages_reacquired_and_pool_restored_without_forward',i)
            else:
                check(allocation.get('attempted') is False and allocation.get('reason')=='no_physical_pages_released',
                      'no_page_release_explicitly_reported',i)
            copies.append(dict(turn=i,freed_pages=sum(map(len,freed)),byte_verification_enabled=all(r.get('byte_verification_enabled') is True for r in receipts),
                               copied_rows=sum(item.get('copied_rows',0) for item in copy_rows)))
            anchor=ee
        if not t['initial']:check(entry.get('block_ids')==(anchor or {}).get('block_ids'),'resume_uses_survivor_pages',i)
        previous=ep
    close=one([r for r in engine if r['event']=='close_released'],'one_close_release',sid)
    check(bool(close.get('block_ids')) and not any(close['block_ids']) and close.get('request_still_registered') is False,'all_owned_blocks_released',sid)
    check(not any(r.get('preemptions',0) for r in engine),'no_preemptions')
    return dict(valid=not errors,checks=checks,errors=errors,copies=copies,
                byte_hashes_observed=bool(copies) and all(r['byte_verification_enabled'] for r in copies),
                physical_page_release_observed=any(r['freed_pages'] for r in copies))


def reject_active_capacity(client,delta,engine_path):
    before=client.status(client.sid);events_before=engine_for(engine_path,client.sid)
    body,route=client.build(delta,observation=('incoming-rejected',4),request_id=client.sid+'-capacity-reject')
    try:
        client.call('POST',route,body)
    except HTTPFailure:
        record=json.loads((client.evidence/f'{client.index:03d}-response.json').read_text())
        if record.get('status')!=413:raise ContractFailure('Expected explicit HTTP 413 capacity fallback')
        payload=json.loads(record['body']) if isinstance(record['body'],str) else record['body']
        receipt=payload.get('error',{}).get('history_eviction',{})
        if receipt.get('capacity_fallback')!='insufficient_completed_history' or receipt.get('eviction_committed') is not False:
            raise ContractFailure('Capacity error did not identify an uncommitted whole-block fallback')
    else:raise ContractFailure('Oversized active-only append unexpectedly succeeded')
    after=client.status(client.sid);events_after=engine_for(engine_path,client.sid)
    stable=lambda status:{k:v for k,v in status.items() if k!='operations'}
    if stable(before)!=stable(after) or after.get('state')!='paused' or events_before!=events_after:
        raise ContractFailure('Capacity rejection changed paused state, block registry or engine journal')
    return dict(status=413,receipt=receipt,paused_state_unchanged=True,engine_events_unchanged=True,
                note='Rejected request is retained in HTTP operation audit; resident/protocol state is unchanged.')


def run(args):
    if socket.gethostname()!=args.expected_hostname:raise ContractFailure('Host identity mismatch')
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if boot!=args.expected_boot_id:raise ContractFailure('Boot identity mismatch')
    for path in (args.engine_events,args.service_events):
        if not Path(path).is_file():raise ContractFailure('Missing authoritative audit '+str(path))
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    (out/'main-http').mkdir();(out/'capacity-http').mkdir()
    here=Path(__file__).resolve().parent;tag='raw-history-'+time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]
    deadline=time.monotonic()+args.max_seconds
    main=HistoryClient(args.base_url,out/'main-http',args.timeout,tag,deadline=deadline)
    active=HistoryClient(args.base_url,out/'capacity-http',args.timeout,tag+'-active',deadline=deadline,budget=1024)
    result=dict(status='starting',plan=plan_description(),started=time.time(),hostname=socket.gethostname(),boot_id=boot,
                session_id=main.sid,capacity_session_id=active.sid,
                source_sha256={n:hashlib.sha256((here/n).read_bytes()).hexdigest() for n in SOURCE_FILES})
    target=out/'results.json';write_json(target,result)
    with ExitStack() as stack:
        for name in LOCKS:
            f=stack.enter_context((Path(args.locks_dir)/name).open('a+'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            health=main.call('GET','/arc3/persistent/health');write_json(out/'health-before.json',health)
            if health.get('mode')!=MODE or health.get('max_context')!=args.expected_context or health.get('prefix_caching') is not False:
                raise ContractFailure('Wrong serving mode/context/prefix configuration')
            if any(s.get('state') not in ('closed','error') for s in health.get('sessions',{}).values()):raise ContractFailure('Another resident session owns server')
            from tokenizers import Tokenizer
            tokenizer=Path(args.tokenizer);result['tokenizer_sha256']=hashlib.sha256(tokenizer.read_bytes()).hexdigest()
            prompt=prompt_factory(Tokenizer.from_file(str(tokenizer)))
            main.submit('obs1',prompt(3000),observation=('obs1',128),suffix=[1000,1257])
            main.submit('inspection',prompt(17,4000))
            main.submit('obs2_suffix_hole',prompt(2500,5000),observation=('obs2',5),suffix=[800,1063])
            main.submit('obs3_first_history_eviction',prompt(2500,9000),observation=('obs3',5),suffix=[900,1169])
            result['idempotency']=main.replay_last()
            main.submit('obs4_second_history_eviction',prompt(3000,13000),observation=('obs4',5))
            fill=main.oracle.budget-len(main.oracle.resident_positions)-8
            if fill<1:raise ContractFailure('Unexpected fill plan')
            main.submit('fill_active_to_capacity',prompt(fill,17000))
            main.submit('empty_continuation_third_history_eviction',[])
            main.close(main.sid)
            active.submit('oversized_active',prompt(900,21000),observation=('active',128))
            result['capacity_rejection']=reject_active_capacity(active,prompt(200,23000),args.engine_events)
            active.submit('continuation_after_rejection',[])
            active.close(active.sid)
            result['audit']=audit_events(args.engine_events,args.service_events,main.sid,main.turns)
            result['capacity_audit']=audit_events(args.engine_events,args.service_events,active.sid,active.turns,expected_rejections=1)
            if not result['audit']['valid'] or not result['capacity_audit']['valid']:raise ContractFailure('Authoritative journal audit failed')
            if not result['audit']['physical_page_release_observed']:
                raise ContractFailure('Main panel did not exercise physical page release/reacquisition')
            result['status']='passed'
        except BaseException as exc:
            result.update(status='failed',error=dict(type=type(exc).__name__,message=str(exc),traceback=traceback.format_exc()))
        finally:
            result['turns']=main.turns;result['capacity_turns']=active.turns
            result['cleanup_errors']=main.cleanup()+active.cleanup()
            try:
                final=main.call('GET','/arc3/persistent/health');write_json(out/'health-after.json',final)
                result['owned_final_states']={sid:final.get('sessions',{}).get(sid) for sid in (main.sid,active.sid)}
                for client in (main,active):
                    if client.turns and (result['owned_final_states'][client.sid] or {}).get('state')!='closed':result['status']='failed'
            except BaseException as exc:result.update(status='failed',cleanup_health_error=repr(exc))
            for name,path in (('engine',args.engine_events),('service',args.service_events)):
                try:
                    rows=(engine_for(path,main.sid)+engine_for(path,active.sid)) if name=='engine' else [r for line in Path(path).read_text().splitlines()
                          if line and (r:=json.loads(line)).get('session_id') in (main.sid,active.sid)]
                    (out/(name+'-owned.jsonl')).write_text(''.join(json.dumps(r,separators=(',',':'))+'\n' for r in rows))
                except BaseException as exc:
                    result.setdefault('audit_copy_errors',[]).append(dict(kind=name,error=repr(exc)))
                    result['status']='failed'
            result['ended']=time.time();result['http_requests']=main.index+active.index
            write_json(target,result)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true');p.add_argument('--output')
    p.add_argument('--base-url',default='http://127.0.0.1:1234')
    p.add_argument('--tokenizer',default='/workspace/models/Qwen3.8-Flash-Next-NVFP4/tokenizer.json')
    p.add_argument('--locks-dir',default='/workspace/arc3-inference/runs')
    p.add_argument('--expected-hostname');p.add_argument('--expected-boot-id')
    p.add_argument('--engine-events');p.add_argument('--service-events')
    p.add_argument('--expected-context',type=int,default=40000)
    p.add_argument('--max-seconds',type=float,default=300);p.add_argument('--timeout',type=float,default=90)
    a=p.parse_args()
    if not a.execute:print(json.dumps(plan_description(),indent=2));return 0
    if any(getattr(a,k) is None for k in ('output','expected_hostname','expected_boot_id','engine_events','service_events')):
        p.error('Execution needs fresh output path, explicit host/boot identity and both audit paths')
    if not 0<a.max_seconds<=600 or not 0<a.timeout<=180 or a.expected_context<7000:p.error('Invalid bounded execution settings')
    r=run(a);return 0 if r['status']=='passed' and not r['cleanup_errors'] else 1


if __name__=='__main__':raise SystemExit(main())
