"""Bounded real-renderer image/tool/suffix-eviction control.

Only --execute submits work, and only to a loopback server. No raw image token
IDs are constructed. Tool feedback is a deterministic fixture, not real game
execution. Correct color reports plus ownership/count checks do not establish
numerical KV equivalence or gameplay quality.
"""
from __future__ import annotations

import argparse
import base64
import hashlib
import io
import json
from pathlib import Path
import time
import uuid

from PIL import Image

from managed_spans import content_hash
from raw_gpu_checks import Client, ContractFailure, token_digest, write_json


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
    def __init__(self,sid,audit_path):
        self.sid = sid
        self.audit_path = audit_path
        self.ids = []
        self.resident = 0
        self.revision = 0
        self.eviction_revision = 0
        self.live = {}
        self.checked_operations = []

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
        removed = sum(self.live[key][1]-self.live[key][0] for key in retire_ids)
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
        self.ids,self.resident = full,expected['resident_tokens']
        self.revision,self.eviction_revision = expected['revision'],expected['eviction_revision']
        row = dict(request_id=request_id,appended_prompt_tokens=len(new_ids),
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute',action='store_true')
    parser.add_argument('--base-url',default='http://127.0.0.1:1234')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--service-audit',type=Path,required=True)
    parser.add_argument('--max-tokens',type=int,default=1024)
    parser.add_argument('--max-fragments',type=int,default=2)
    parser.add_argument('--timeout',type=float,default=300)
    args = parser.parse_args()
    if not 1 <= args.max_tokens <= 2048 or not 1 <= args.max_fragments <= 4:
        parser.error('Bounded diagnostic requires max-tokens 1..2048 and max-fragments 1..4')
    plan = dict(observations=3,logical_assistants=9,current_colors=['red','blue','blue'],
        expected_suffix_evictions=2,max_generated_tokens=9*args.max_tokens*args.max_fragments,
        scope='Actual image/tool transport, exact suffix ownership and logical/resident receipts; not numerical KV equivalence')
    if not args.execute:
        print(json.dumps(dict(execute=False,plan=plan),indent=2))
        return 0
    if not args.service_audit.is_file():
        parser.error('A readable actual server service-audit file is required')
    args.output.mkdir(parents=True,exist_ok=False)
    sid = 'managed-mm-'+uuid.uuid4().hex[:16]
    client = Client(args.base_url,args.output,args.timeout,sid)
    ledger = Ledger(sid,args.service_audit)
    result = dict(session_id=sid,started=time.time(),plan=plan,passed=False,
                  tool_execution='Simulated inspect/action feedback; no arbitrary Python/game execution',observations=[])
    operation_count = 0

    def save():
        result['checked_operations'] = ledger.checked_operations
        write_json(args.output/'result.json',result)

    def send(messages,managed,*,initial=False,continuation=False):
        nonlocal operation_count
        operation_count += 1
        rid = f'{sid}-op{operation_count:03d}'
        if initial:
            chat = dict(model=MODEL,messages=messages,tools=TOOLS,tool_choice='auto',
                max_tokens=args.max_tokens,return_token_ids=True,add_special_tokens=False,
                add_generation_prompt=True,continue_final_message=False,
                chat_template_kwargs=dict(enable_thinking=True,reasoning_effort='xhigh',preserve_thinking=True),
                **SAMPLING)
            client.active.add(sid)
            response = client.call('POST','/arc3/persistent/start',
                dict(session_id=sid,request_id=rid,chat=chat,managed_suffixes=managed))
        else:
            response = client.call('POST',f'/arc3/persistent/{sid}/append',
                dict(request_id=rid,expected_revision=ledger.revision,
                     expected_prefix_sha256=token_digest(ledger.ids),
                     external_messages=messages,continuation=continuation,
                     managed_suffixes=managed,sampling=dict(max_tokens=args.max_tokens,**SAMPLING)))
        choice,state = ledger.accept(response,rid,managed,continuation=continuation)
        save()
        return choice,state

    def finish(choice,state):
        for _ in range(args.max_fragments-1):
            if state['engine_finish_reason'] != 'length':
                break
            choice,state = send([],sidecar(),continuation=True)
        if state['engine_finish_reason'] == 'length':
            raise ContractFailure('Assistant exceeded the bounded continuation budget')
        return choice,state

    try:
        result['health_before'] = client.call('GET','/arc3/persistent/health')
        previous_png = None
        for number,current in enumerate(('red','blue','blue'),1):
            recap = 'blue' if current == 'red' else 'red'
            user,current_png,recap_png = observation(current,recap,number)
            (args.output/f'obs{number}-current-{current}.png').write_bytes(current_png)
            (args.output/f'obs{number}-recap-{recap}.png').write_bytes(recap_png)
            desc = descriptor(user,number)
            if number == 1:
                messages = [dict(role='system',content=(
                    'This is a simple visual protocol fixture. Each new real observation contains two images: '
                    'FIRST the current board, then an older historical recap. Report the current board only. '
                    'For every observation call inspect once with its red/blue color. After inspect feedback, '
                    'call action once with the SAME current color. After action feedback, finish with '
                    'Ready for next board. Do not call additional tools. The tool results are fixture feedback.')),user]
                desc['message_index'] = 1
            else:
                messages = [user]
            choice,state = finish(*send(messages,sidecar(desc,list(ledger.live)),initial=number==1))
            inspection = parse_tool(choice,'inspect',current)
            before_tool_eviction = ledger.eviction_revision
            inspect_result = dict(role='tool',tool_call_id=inspection['id'],content=json.dumps({
                'inspection_finished':True,'current_color':current,
                'next':'Call action once with this same current color.'}))
            choice,state = finish(*send([inspect_result],sidecar()))
            action = parse_tool(choice,'action',current)
            action_result = dict(role='tool',tool_call_id=action['id'],content=json.dumps({
                'action_finished':True,'next':'Finish with Ready for next board. Do not call any more tools.'}))
            choice,state = finish(*send([action_result],sidecar()))
            if choice['message'].get('tool_calls'):
                raise ContractFailure('Unexpected extra tool after action feedback')
            if 'ready' not in (choice['message'].get('content') or '').lower():
                raise ContractFailure('Expected a completed readiness answer after fixture action')
            if ledger.eviction_revision != before_tool_eviction:
                raise ContractFailure('Same-observation tool loop unexpectedly evicted its suffix')
            row = dict(observation=number,current_color=current,recap_color=recap,
                current_image_sha256=hashlib.sha256(current_png).hexdigest(),
                recap_image_sha256=hashlib.sha256(recap_png).hexdigest(),
                inspected_color=current,action_color=current,
                previous_current_image_equal=previous_png==current_png if previous_png else None,
                logical_tokens=len(ledger.ids),resident_tokens=ledger.resident,
                eviction_revision=ledger.eviction_revision)
            result['observations'].append(row)
            previous_png = current_png
            save()
        if (ledger.eviction_revision != 2
                or result['observations'][1]['previous_current_image_equal'] is not False
                or result['observations'][2]['previous_current_image_equal'] is not True):
            raise ContractFailure('Changed/repeated image or two-eviction coverage is incomplete')
        result['passed'] = True
    except BaseException as error:
        result['error'] = f'{type(error).__name__}: {error}'
        try:
            result['status_at_error'] = client.status(sid)
        except BaseException as status_error:
            result['status_error'] = repr(status_error)
    finally:
        result['close_errors'] = client.cleanup()
        if result['close_errors']:
            result['passed'] = False
        try:
            result['status_after_close'] = client.status(sid)
        except BaseException as status_error:
            result['final_status_error'] = repr(status_error)
        write_json(args.output/'session-service-audit.json',audit_rows(args.service_audit,sid))
        result['finished'] = time.time()
        save()
    print(json.dumps({key:result.get(key) for key in ('passed','error','observations','close_errors')},indent=2))
    return 0 if result['passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
