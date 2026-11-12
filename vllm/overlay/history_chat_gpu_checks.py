"""Bounded real-renderer image/tool/whole-observation-eviction control.

Only --execute submits work, and only to a loopback server. No raw image token
IDs are constructed. Tool feedback is a deterministic fixture, not real game
execution. Correct color reports plus ownership/count checks do not establish
numerical KV equivalence or gameplay quality.
"""
from __future__ import annotations

import argparse
from collections import Counter
from contextlib import ExitStack
import fcntl
import re
import base64
import hashlib
import io
import json
from pathlib import Path
import copy
import socket
import traceback
import time
import uuid

from PIL import Image

from managed_spans import content_hash
from raw_gpu_checks import Client, ContractFailure, token_digest, write_json
from raw_history_eviction_checks import Oracle, MODE, engine_for
from raw_managed_eviction_checks import ManagedClient, LOCKS


MODEL = 'nvidia/Qwen3.8-Flash-Next-NVFP4'
COLORS = {'red':(249,60,49), 'blue':(30,147,255)}
TOOLS = [{'type':'function','function':{
    'name':name,
    'description':description,
    'parameters':{'type':'object','properties':{
        'color':{'type':'string','enum':['red','blue']}},'required':['color'],
        'additionalProperties':False}}}
    for name,description in (
        ('inspect','Report the current board color before acting.'),
        ('action','Take the fixture action matching the inspected current board color.'))]
SAMPLING = dict(temperature=0,top_p=1,top_k=0,min_p=0)


def png_part(color):
    buffer = io.BytesIO()
    Image.new('RGB',(256,256),COLORS[color]).save(buffer,format='PNG')
    data = buffer.getvalue()
    return data,{'type':'image_url','image_url':{
        'url':'data:image/png;base64,'+base64.b64encode(data).decode()}}


def observation(current,recap,number):
    current_png,current_part = png_part(current)
    recap_png,recap_part = png_part(recap)
    parts = [
        {'type':'text','text':(
            'A new real board observation. The FIRST image immediately below is the current board. '
            'Identify its solid color and call inspect once. The separate recap is historical.\n')},
        current_part,
        {'type':'text','text':'\nAvailable library programs: inspect_board, choose_matching_action.\n'},
        {'type':'text','text':(
            'Historical recap: the following image is an OLDER board, not the current board. '
            f'Log pointer: fixture history entry {number}.\n')},
        recap_part,
    ]
    return {'role':'user','content':parts},current_png,recap_png


def descriptor(user,number):
    return dict(message_index=0,observation_id=f'obs-{number:08d}',
        content_sha256=content_hash(user['content']),base_parts_end=2,
        program_parts=[2,3],snapshot_parts=[3,5])


def sidecar(desc=None,retire=()):
    return dict(version=1,observations=[] if desc is None else [desc],
                evict_observation_ids=list(retire))


def audit_rows(path,sid):
    rows = []
    with path.open() as stream:
        for line in stream:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue  # unrelated concurrently written trailing record
            if row.get('session_id') == sid:
                rows.append(row)
    return rows


class Ledger:
    """Check receipts against saved exact renderer IDs, not local retokenization."""
    def __init__(self,sid,audit_path,budget=1250):
        self.sid = sid
        self.audit_path = audit_path
        self.ids = []
        self.resident = 0
        self.revision = 0
        self.eviction_revision = 0
        self.live = {}
        self.checked_operations = []
        self.oracle = Oracle(budget)

    def accept(self,response,request_id,managed,*,continuation=False):
        state = response.get('persistent')
        choices = response.get('choices')
        if not isinstance(state,dict) or not isinstance(choices,list) or len(choices) != 1:
            raise ContractFailure('Missing persistent receipt/single choice')
        emitted = state.get('token_ids')
        if (not isinstance(emitted,list) or not emitted
                or any(type(token) is not int for token in emitted)
                or choices[0].get('token_ids') != emitted):
            raise ContractFailure('Raw emitted IDs are missing or disagree')
        rows = audit_rows(self.audit_path,self.sid)
        submissions = [row for row in rows if row.get('event') == 'submit'
                       and row.get('operation_id') == request_id]
        if len(submissions) != 1:
            raise ContractFailure('Require one exact service submit audit for the completed operation')
        submitted = submissions[0]
        new_ids = submitted.get('appended_token_ids')
        if not isinstance(new_ids,list) or any(type(token) is not int for token in new_ids):
            raise ContractFailure('Missing rendered appended input IDs')
        if continuation and new_ids:
            raise ContractFailure('Empty assistant continuation appended unexpected input tokens')
        descriptors = managed['observations']
        retire_ids = managed['evict_observation_ids']
        expected_ack = dict(registered_observation_ids=[d['observation_id'] for d in descriptors],
                            evicted_observation_ids=retire_ids)
        if state.get('managed_suffixes') != expected_ack or submitted.get('managed_suffixes') != expected_ack:
            raise ContractFailure('Managed suffix receipt differs from the explicit operation')
        if retire_ids != (list(self.live) if descriptors else []):
            raise ContractFailure('Diagnostic attempted a non-observation or wrong-owner eviction')
        old_ids = list(self.ids)
        records = submitted.get('suffix_records',[])
        if len(records) != len(descriptors):
            raise ContractFailure('Rendered suffix records disagree with descriptors')
        checked_records = []
        for desc,record in zip(descriptors,records):
            if any(record.get(key) != value for key,value in desc.items()):
                raise ContractFailure('Server changed the semantic suffix descriptor')
            span = record.get('local_token_range')
            if (not isinstance(span,list) or len(span) != 2
                    or any(type(x) is not int for x in span)
                    or not 0 < span[0] < span[1] < len(new_ids)):
                raise ContractFailure('Managed suffix is not an interior expanded input span')
            start,end = span
            # The fixed fixture's immutable base ends with its current board;
            # the closing user framing remains after the complete suffix.
            if new_ids[start-1] != 248054 or new_ids[end] != 248046:
                raise ContractFailure('Managed span overlaps current-board or message framing')
            images = record.get('image_spans',[])
            if len(images) != 1 or images[0].get('part_index') != 4:
                raise ContractFailure('Expected exactly the recap image in managed ownership')
            feature = images[0]['feature_range']
            envelope = images[0]['envelope_range']
            if not (start <= envelope[0] == feature[0]-1 < feature[1] == envelope[1]-1
                    and envelope[1] <= end):
                raise ContractFailure('Recap image envelope escapes the managed suffix')
            if (feature[1]-feature[0] != 64
                    or new_ids[feature[0]:feature[1]] != [248056]*64
                    or new_ids[envelope[0]] != 248053 or new_ids[envelope[1]-1] != 248054):
                raise ContractFailure('Actual renderer did not produce the expected 256px recap envelope')
            if new_ids[:start].count(248056) != 64:
                raise ContractFailure('The separate current-board image is not fully outside the managed suffix')
            checked_records.append(dict(observation_id=record['observation_id'],
                local_token_range=span,logical_token_range=[len(self.ids)+start,len(self.ids)+end],
                recap_image=images[0]))
        starts = submitted.get('observation_starts',[])
        if len(starts) != len(descriptors):
            raise ContractFailure('Exact observation block-start proof is missing or unexpected')
        observation_start = None
        if descriptors:
            start_record = starts[0]
            start = start_record.get('local_start_token')
            proof = start_record.get('proof') or {}
            method = ('paired_no_generation_empty_user_probe_v1' if self.revision == 0
                      else 'paired_no_generation_preceding_messages_v1')
            if (start_record.get('observation_id') != descriptors[0]['observation_id']
                    or type(start) is not int or not 0 <= start < checked_records[0]['local_token_range'][0]
                    or proof.get('method') != method
                    or proof.get('preceding_token_count') != start
                    or proof.get('prefix_sha256') != token_digest(new_ids[:start])
                    or proof.get('opening_token_ids') != [248045,846,198]
                    or new_ids[start:start+3] != [248045,846,198]
                    or proof.get('includes_generation_prompt') is not False
                    or proof.get('preceding_image_count') != 0
                    or proof.get('observation_image_count') != 2):
                raise ContractFailure('Observation is not rooted at the proven full user-role opening')
            if self.revision == 0 and start < 128:
                raise ContractFailure('Initial protected tools/system prefix unexpectedly missing')
            observation_start = (descriptors[0]['observation_id'],start)
        candidate = copy.deepcopy(self.oracle)
        prepared = candidate.prepare(new_ids,observation=observation_start,
            suffix=checked_records[0]['local_token_range'] if checked_records else None)
        receipt = state.get('history_eviction') or {}
        if any(receipt.get(key) != value for key,value in prepared['receipt'].items()):
            raise ContractFailure('History selection differs from independent whole-block oracle')
        if submitted.get('history_eviction') != receipt:
            raise ContractFailure('Service and HTTP history receipts disagree')
        removed = len(prepared['removed'])
        after_eviction = [old_ids[p] for p in prepared['after']]
        candidate.commit(prepared,emitted)
        if state.get('history_state') != candidate.history_state() or submitted.get('history_state') != candidate.history_state():
            raise ContractFailure('History registration differs from exact original block ledger')
        logical_input = len(self.ids)+len(new_ids)
        resident_input = self.resident-removed+len(new_ids)
        full = self.ids+new_ids+emitted
        expected = dict(session_id=self.sid,revision=self.revision+1,
            logical_input_tokens=logical_input,logical_tokens=len(full),
            resident_tokens=resident_input+len(emitted),appended_prompt_tokens=len(new_ids),
            prefix_sha256=token_digest(full),evicted_tokens=removed,
            eviction_revision=self.eviction_revision+int(removed>0),eviction=True,prefix_caching=False)
        if any(state.get(key) != value for key,value in expected.items()):
            raise ContractFailure('Persistent logical/resident receipt arithmetic disagrees with exact saved IDs')
        if submitted.get('old_tokens') != len(self.ids) or submitted.get('input_tokens') != resident_input:
            raise ContractFailure('Submit audit disagrees with retained input size')
        usage = response.get('usage',{})
        if usage != dict(prompt_tokens=resident_input,completion_tokens=len(emitted),
                         total_tokens=resident_input+len(emitted)):
            raise ContractFailure('Response usage differs from resident input and emitted delta')
        if state.get('engine_finish_reason') not in ('length','stop'):
            raise ContractFailure('Unexpected engine finish reason')
        if state['engine_finish_reason'] == 'stop' and emitted[-1] != 248046:
            raise ContractFailure('Completed assistant lacks an actual generated im_end')
        for key in retire_ids:
            del self.live[key]
        for record in checked_records:
            self.live[record['observation_id']] = record['logical_token_range']
        self.oracle = candidate
        self.ids,self.resident = full,expected['resident_tokens']
        self.revision,self.eviction_revision = expected['revision'],expected['eviction_revision']
        retained = [candidate.logical[p] for p in candidate.resident_positions]
        row = dict(request_id=request_id,turn=self.revision-1,initial=self.revision==1,
            delta_tokens=len(new_ids),output_tokens=len(emitted),delta_sha256=token_digest(new_ids),
            output_sha256=token_digest(emitted),logical_before=len(old_ids),logical_before_sha256=token_digest(old_ids),
            logical_sha256=token_digest(full),resident_sha256=token_digest(retained),
            computed_prefix_sha256=token_digest(retained[:-1]),resident_after_eviction=len(after_eviction),
            after_eviction_sha256=token_digest(after_eviction),removed_tokens=removed,
            expected_prefill_tokens=len(new_ids)+(self.revision!=1),
            expected_resume_computed=0 if self.revision==1 else len(after_eviction)-1,
            observation_starts=starts,history_eviction=receipt,history_state=candidate.history_state(),
            appended_prompt_tokens=len(new_ids),
            generated_tokens=len(emitted),logical_tokens=len(full),resident_tokens=self.resident,
            evicted_tokens=removed,eviction_revision=self.eviction_revision,
            engine_finish_reason=state['engine_finish_reason'],suffix_records=checked_records,
            managed_suffixes=expected_ack,continuation=continuation)
        self.checked_operations.append(row)
        return choices[0],state


def parse_tool(choice,name,color):
    calls = choice['message'].get('tool_calls') or []
    if len(calls) != 1 or calls[0]['function']['name'] != name:
        raise ContractFailure(f'Expected exactly one {name} tool call')
    args = json.loads(calls[0]['function']['arguments'])
    if args != {'color':color}:
        raise ContractFailure(f'{name} did not identify the separate current board: {args!r}')
    return calls[0]



def audit_chat_events(engine_path,service_path,sid,turns,*,expected_rejections=0):
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
        expected=dict(prefill_tokens=t['expected_prefill_tokens'],decode_tokens=t['output_tokens']-1,scheduled_tokens=t['expected_prefill_tokens']+t['output_tokens']-1,
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


SYSTEM = (
    'This is a simple visual protocol fixture. Each new real observation contains two images: '
    'FIRST the current board, then an older historical recap. Report the current board only. '
    'For every observation call inspect once with its red/blue color. After inspect feedback, '
    'call action once with the SAME current color. After action feedback, finish with '
    'Ready for next board. Do not call additional tools. The tool results are fixture feedback.')
SEQUENCE = ('red','blue','blue','red','red')


def run(args):
    if socket.gethostname()!=args.expected_hostname:
        raise ContractFailure('Host identity mismatch')
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if boot!=args.expected_boot_id:
        raise ContractFailure('Boot identity mismatch')
    for path in (args.service_audit,args.engine_events):
        if not path.is_file():raise ContractFailure('Missing authoritative audit '+str(path))
    args.output.mkdir(parents=True,exist_ok=False)
    sid='history-mm-'+uuid.uuid4().hex[:16]
    client=ManagedClient(args.base_url,args.output,args.timeout,sid,deadline=time.monotonic()+args.max_seconds)
    ledger=Ledger(sid,args.service_audit,args.resident_budget)
    result=dict(session_id=sid,started=time.time(),passed=False,hostname=socket.gethostname(),boot_id=boot,
        tool_execution='Simulated inspect/action feedback; no arbitrary Python/game execution',
        policy=dict(policy='oldest_interactions',fraction=.10,resident_token_budget=args.resident_budget),
        observations=[],source_sha256={name:hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
            for name in ('history_chat_gpu_checks.py','raw_history_eviction_checks.py','protocol.py','managed_spans.py')})
    operation_count=0

    def save():
        result['checked_operations']=ledger.checked_operations
        write_json(args.output/'result.json',result)

    def send(messages,managed,*,initial=False,continuation=False):
        nonlocal operation_count
        operation_count+=1
        rid=f'{sid}-op{operation_count:03d}'
        if initial:
            chat=dict(model=MODEL,messages=messages,tools=TOOLS,tool_choice='auto',
                max_tokens=args.max_tokens,return_token_ids=True,add_special_tokens=False,
                add_generation_prompt=True,continue_final_message=False,
                chat_template_kwargs=dict(enable_thinking=True,reasoning_effort='xhigh',preserve_thinking=True),**SAMPLING)
            client.active.add(sid)
            response=client.call('POST','/arc3/persistent/start',dict(session_id=sid,request_id=rid,
                chat=chat,managed_suffixes=managed,history_eviction=result['policy']))
        else:
            response=client.call('POST',f'/arc3/persistent/{sid}/append',dict(request_id=rid,
                expected_revision=ledger.revision,expected_prefix_sha256=token_digest(ledger.ids),
                external_messages=messages,continuation=continuation,managed_suffixes=managed,
                sampling=dict(max_tokens=args.max_tokens,**SAMPLING)))
        answer=ledger.accept(response,rid,managed,continuation=continuation)
        save()
        return answer

    def finish(choice,state):
        for _ in range(args.max_fragments-1):
            if state['engine_finish_reason']!='length':break
            choice,state=send([],sidecar(),continuation=True)
        if state['engine_finish_reason']=='length':
            raise ContractFailure('Assistant exceeded bounded continuation budget')
        return choice,state

    with ExitStack() as stack:
        for name in LOCKS:
            lock=stack.enter_context((args.locks_dir/name).open('a+'))
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            result['health_before']=health=client.call('GET','/arc3/persistent/health')
            if health.get('mode')!=MODE or health.get('max_context')!=40000 or health.get('prefix_caching') is not False:
                raise ContractFailure('Wrong mode/context/prefix caching')
            if any(s.get('state') not in ('closed','error') for s in health.get('sessions',{}).values()):
                raise ContractFailure('Another resident session owns the server')
            previous_png=None
            for number,current in enumerate(SEQUENCE,1):
                recap='blue' if current=='red' else 'red'
                user,current_png,recap_png=observation(current,recap,number)
                (args.output/f'obs{number}-current-{current}.png').write_bytes(current_png)
                (args.output/f'obs{number}-recap-{recap}.png').write_bytes(recap_png)
                desc=descriptor(user,number)
                messages=[user]
                if number==1:
                    messages=[dict(role='system',content=SYSTEM),user]
                    desc['message_index']=1
                choice,state=finish(*send(messages,sidecar(desc,list(ledger.live)),initial=number==1))
                inspection=parse_tool(choice,'inspect',current)
                frozen_live=copy.deepcopy(ledger.live)
                loop_start=len(ledger.checked_operations)
                inspect_result=dict(role='tool',tool_call_id=inspection['id'],content=json.dumps({
                    'inspection_finished':True,'current_color':current,
                    'next':'Call action once with this same current color.'}))
                choice,state=finish(*send([inspect_result],sidecar()))
                action=parse_tool(choice,'action',current)
                action_result=dict(role='tool',tool_call_id=action['id'],content=json.dumps({
                    'action_finished':True,'next':'Finish with Ready for next board. Do not call any more tools.'}))
                choice,state=finish(*send([action_result],sidecar()))
                if choice['message'].get('tool_calls') or 'ready' not in (choice['message'].get('content') or '').lower():
                    raise ContractFailure('Expected completed readiness answer after action')
                loop=ledger.checked_operations[loop_start:]
                if ledger.live!=frozen_live or any(r['history_eviction']['suffix_removed_tokens'] for r in loop):
                    raise ContractFailure('Same-observation tool loop changed its managed suffix')
                result['observations'].append(dict(observation=number,current_color=current,recap_color=recap,
                    current_image_sha256=hashlib.sha256(current_png).hexdigest(),
                    recap_image_sha256=hashlib.sha256(recap_png).hexdigest(),inspected_color=current,action_color=current,
                    previous_current_image_equal=previous_png==current_png if previous_png else None,
                    logical_tokens=len(ledger.ids),resident_tokens=ledger.resident,eviction_revision=ledger.eviction_revision,
                    history_deletions_so_far=sum(bool(r['history_eviction']['history_removed_tokens']) for r in ledger.checked_operations)))
                previous_png=current_png
                save()
            if result['observations'][-1]['history_deletions_so_far']<2:
                raise ContractFailure('Panel did not exercise repeated whole-observation deletion')
            # Later changed and repeated images must both be interpreted after a prior history deletion.
            if not all(r['history_deletions_so_far']>0 for r in result['observations'][-2:]):
                raise ContractFailure('Changed/repeated-image coverage did not follow history eviction')
            result['passed']=True
        except BaseException as error:
            result['error']=dict(type=type(error).__name__,message=str(error),traceback=traceback.format_exc())
            try:result['status_at_error']=client.status(sid)
            except BaseException as status_error:result['status_error']=repr(status_error)
        finally:
            result['close_errors']=client.cleanup()
            if result['close_errors']:result['passed']=False
            try:
                result['status_after_close']=client.status(sid)
                if result['status_after_close'].get('state')!='closed':result['passed']=False
                write_json(args.output/'session-service-audit.json',audit_rows(args.service_audit,sid))
                (args.output/'engine-owned.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in engine_for(args.engine_events,sid)))
                if result['passed']:
                    result['audit']=audit_chat_events(args.engine_events,args.service_audit,sid,ledger.checked_operations)
                    if not result['audit']['valid']:result['passed']=False
            except BaseException as audit_error:
                result['audit_error']=repr(audit_error);result['passed']=False
            result['finished']=time.time()
            save()
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--base-url',default='http://127.0.0.1:1234')
    parser.add_argument('--output',type=Path)
    parser.add_argument('--service-audit',type=Path)
    parser.add_argument('--engine-events',type=Path)
    parser.add_argument('--locks-dir',type=Path,default=Path('/workspace/arc3-inference/runs'))
    parser.add_argument('--expected-hostname');parser.add_argument('--expected-boot-id')
    parser.add_argument('--max-tokens',type=int,default=256)
    parser.add_argument('--max-fragments',type=int,default=4)
    parser.add_argument('--resident-budget',type=int,default=1250)
    parser.add_argument('--timeout',type=float,default=90)
    parser.add_argument('--max-seconds',type=float,default=300)
    args=parser.parse_args()
    if not 32<=args.max_tokens<=512 or not 1<=args.max_fragments<=4 or not 1100<=args.resident_budget<=2000:
        parser.error('Diagnostic bounds: output32..512, fragments1..4, residentbudget1100..2000')
    if not 0<args.timeout<=180 or not 0<args.max_seconds<=600:parser.error('Invalid bounded deadline')
    if not args.execute:
        print(json.dumps(dict(execute=False,observations=len(SEQUENCE),current_colors=SEQUENCE,
            resident_budget=args.resident_budget,max_generated_tokens=15*args.max_tokens*args.max_fragments,
            scope='Actual PNG/tool/whole-observation transport and no-refill audit; not gameplay quality'),indent=2))
        return 0
    if any(getattr(args,k) is None for k in ('output','service_audit','engine_events','expected_hostname','expected_boot_id')):
        parser.error('Execution requires fresh output, host/boot identity, and authoritative journal paths')
    result=run(args)
    print(json.dumps({k:result.get(k) for k in ('passed','error','observations','close_errors')},indent=2))
    return 0 if result['passed'] else 1


if __name__=='__main__':raise SystemExit(main())
