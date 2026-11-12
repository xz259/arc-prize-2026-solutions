"""Pure accounting and timing analysis for the bounded eviction benchmark."""
from __future__ import annotations

import argparse
from bisect import bisect_right
from collections import Counter
import json
import math
from pathlib import Path
import re


METRICS=('num_requests_running','num_requests_waiting','num_preemptions_total',
         'prompt_tokens_total','generation_tokens_total','kv_cache_usage_perc',
         'prefix_cache_hits_total','prefix_cache_queries_total')


def metric_values(raw):
    values={}
    for line in raw.splitlines():
        if not line or line.startswith('#'):continue
        left,value=line.rsplit(' ',1);key=left.split('{',1)[0]
        if key.startswith('vllm:') and key[5:] in METRICS:
            number=float(value)
            if not math.isfinite(number):raise ValueError('Nonfinite engine metric')
            key=key[5:];values[key]=values.get(key,0.)+number
    if any(k not in values for k in METRICS[:5]):
        raise ValueError('Missing required engine metrics')
    return values


def count_at(events,stamp):
    index=bisect_right(events,[stamp,float('inf')])-1
    return events[index][1] if index>=0 else 0


def pure_decode_windows(samples,*,concurrency,start,end,min_seconds=20,max_gap=2,boundary_guards=()):
    """Counter windows with full observed occupancy and no scheduled prefill.

    The nonstreaming persistent endpoint cannot prove each lane has reached
    38K. Its workload contract bounds every measured request to 36K..39,999.
    Endpoint samples do not prove occupancy remained constant between polls.
    """
    points=sorted((p for p in samples if start<=p['seconds']<=end),key=lambda p:p['seconds'])
    groups=[];group=[]
    for a,b in zip(points,points[1:]):
        am,bm=a['metrics'],b['metrics'];seconds=b['seconds']-a['seconds']
        valid=(0<seconds<=max_gap and max(a['rtt_seconds'],b['rtt_seconds'])<=.5
               and all(m['num_requests_running']==concurrency and m['num_requests_waiting']==0
                       for m in (am,bm))
               and bm['prompt_tokens_total']==am['prompt_tokens_total']
               and bm['num_preemptions_total']==am['num_preemptions_total']
               and bm['generation_tokens_total']>am['generation_tokens_total']
               and not any(a['seconds']<right and left<b['seconds'] for left,right in boundary_guards))
        if valid:
            if not group:group=[a]
            group.append(b)
        elif group:
            groups.append(group);group=[]
    if group:groups.append(group)
    windows=[]
    for group in groups:
        a,b=group[0],group[-1];duration=b['seconds']-a['seconds']
        if duration<min_seconds:continue
        tokens=b['metrics']['generation_tokens_total']-a['metrics']['generation_tokens_total']
        windows.append(dict(start_seconds=a['seconds'],end_seconds=b['seconds'],seconds=duration,
                            generated_tokens=tokens,tokens_per_second=tokens/duration,
                            sample_count=len(group),maximum_rtt_seconds=max(p['rtt_seconds'] for p in group)))
    duration=sum(w['seconds'] for w in windows);tokens=sum(w['generated_tokens'] for w in windows)
    return dict(valid=bool(windows),windows=windows,pooled_seconds=duration,
                pooled_generated_tokens=tokens,tokens_per_second=tokens/duration if duration else None,
                tokens_per_second_per_sequence=tokens/duration/concurrency if duration else None,
                context_contract=[36000,39999],boundary_guard_count=len(boundary_guards),
                occupancy_evidence='Guarded steady-window estimate from sampled engine gauges; no proof of continuous per-step occupancy or all lanes above38K. Persistent prompt counter omits resumed prefill.')


def summarize(result,samples):
    measurement=result.get('measurement')
    if not measurement:return dict(valid=False,reason='No completed measurement interval')
    records=[r for r in result.get('operations',[]) if r.get('phase')=='measured']
    received=sum(r['output_tokens'] for r in records)
    duration=measurement['end_seconds']-measurement['start_seconds']
    before=measurement['metrics_before'];after=measurement['metrics_after']
    generated=after['generation_tokens_total']-before['generation_tokens_total']
    prompt=after['prompt_tokens_total']-before['prompt_tokens_total']
    preemptions=after['num_preemptions_total']-before['num_preemptions_total']
    expected=sum(r['expected_prefill_tokens'] for r in records)
    pure=pure_decode_windows(samples,concurrency=result['config']['concurrency'],
                             start=measurement['start_seconds'],end=measurement['end_seconds'],
                             boundary_guards=result.get('decode_boundary_guards',[]))
    stateless=result['config']['arm']=='stateless'
    expected_counter=expected if stateless else 0
    return dict(valid=duration>0 and received==generated and preemptions==0 and prompt==expected_counter,
                inclusive_seconds=duration,received_tokens=received,engine_generated_tokens=generated,
                received_tokens_per_second=received/duration if duration>0 else None,
                includes_initial_prefill=stateless,
                includes_measured_api_pauses_and_evictions=True,
                engine_prompt_tokens=prompt,expected_scheduled_prefill_tokens=expected,
                expected_engine_prompt_counter_delta=expected_counter,
                prompt_counter_matches_expected_semantics=prompt==expected_counter,
                prompt_counter_matches_schedule=prompt==expected,
                prompt_counter_note='Persistent resume prefill is absent from global prompt counter; use per-turn engine audit for actual scheduled work.',
                preemptions=preemptions,
                max_sampled_kv_usage=max((s['metrics'].get('kv_cache_usage_perc',0) for s in samples),default=None),
                pure_decode=pure)


def audit_owned(engine,service,operations,session_ids,*,expected_proofs):
    """Performance audit: proofs may be off, but ownership/work checks remain."""
    failures=[];counts=Counter();eviction_profiles=[]
    def check(ok,key,detail=None):
        counts[key]+=1
        if not ok:failures.append(dict(check=key,detail=detail))
    for sid in session_ids:
        pattern=re.compile(re.escape('arc3-persistent-'+sid)+r'(?:-[0-9a-f]{8})?\Z')
        rows=[r for r in engine if pattern.fullmatch(r.get('request_id',''))]
        srv=[r for r in service if r.get('session_id')==sid]
        turns=[r for r in operations if r.get('session_id')==sid]
        if not turns:
            check(not rows,'no_engine_owner_for_uncreated_session',sid)
            continue
        check(len({r['request_id'] for r in rows})==1,'one_continuing_engine_owner',sid)
        check(not any(r.get('preemptions',0) for r in rows),'no_owner_preemption',sid)
        check(not any(r['event'] in ('failed','rejected') for r in srv),'no_service_failure',sid)
        check(sum(r['event']=='submit' for r in srv)==len(turns),'one_submission_per_turn',sid)
        check(sum(r['event']=='pause' for r in rows)==len(turns),'one_pause_per_turn',sid)
        evicts=[r for r in rows if r['event']=='evict']
        check(len(evicts)==sum(bool(t['removed_tokens']) for t in turns),'exact_eviction_count',sid)
        for turn in turns:
            index=turn['turn'];op=turn['request_id']
            pauses=[r for r in rows if r['event']=='pause' and r.get('turn')==index]
            entries=[r for r in rows if r['event']==('start' if index==0 else 'resume') and r.get('turn')==index]
            submits=[r for r in srv if r['event']=='submit' and r.get('operation_id')==op]
            sp=[r for r in srv if r['event']=='pause' and r.get('operation_id')==op]
            check(len(pauses)==len(entries)==len(submits)==len(sp)==1,'unique_turn_journals',op)
            if not pauses or not entries:continue
            pause=pauses[0];entry=entries[0]
            for key,value in {'resident_tokens':turn['resident_tokens'],
                              'tokens_sha256':turn['resident_sha256'],
                              'computed_prefix_sha256':turn['computed_prefix_sha256'],
                              'computed_tokens':turn['resident_tokens']-1,
                              'in_flight_tokens':0,'output_placeholders':0,'preemptions':0}.items():
                check(pause.get(key)==value,'pause_'+key,op)
            work=pause.get('turn_work',{});expected={
                'prefill_tokens':turn['expected_prefill_tokens'],'decode_tokens':turn['output_tokens']-1,
                'scheduled_tokens':turn['expected_prefill_tokens']+turn['output_tokens']-1,
                'minimum_scheduled_position':turn['expected_resume_computed'],
                'resume_computed_tokens':turn['expected_resume_computed']}
            check(all(work.get(k)==v for k,v in expected.items()),'no_historical_prefill',dict(op=op,expected=expected,actual=work))
            check(entry.get('computed_tokens')==turn['expected_resume_computed'],'retained_computed_frontier',op)
            check(pause.get('logical_eviction_state',{}).get('logical_sha256')==turn['logical_sha256'],
                  'logical_history_identity',op)
            if submits and sp:
                check(submits[0].get('prefix_sha256')==turn['logical_before_sha256']
                      and submits[0].get('input_tokens')==turn['resident_input_tokens']
                      and submits[0].get('logical_input_tokens')==turn['logical_before']+turn['delta_tokens'],
                      'service_input_prefix',op)
                from hashlib import sha256
                digest=lambda ids:sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()
                check(digest(submits[0].get('appended_token_ids',[]))==turn['delta_sha256']
                      and digest(sp[0].get('output_token_ids',[]))==turn['output_sha256'],
                      'exact_appended_and_generated_ids',op)
        for event in evicts:
            ack=event.get('eviction',{});receipts=ack.get('worker_receipts',[])
            check(len(receipts)==1 and receipts[0].get('worker_synchronized') is True,'fenced_worker_copy',sid)
            copies=[c for r in receipts for c in r.get('copies',[])]
            kinds=Counter('compressed' if c.get('layer_name','').endswith('compressed_key_cache') else 'main' for c in copies)
            check(kinds=={'main':12,'compressed':12},'all_cache_owners_acknowledged',sid)
            check(all(r.get('byte_verification_enabled')==expected_proofs for r in receipts),'correct_proof_profile',sid)
            if not expected_proofs:
                check(ack.get('allocator_reacquisition') is None,'allocator_proof_disabled',sid)
            matching=[t for t in turns if t['removed_tokens'] and t['logical_before']==ack.get('logical_tokens')]
            check(len(matching)==1,'one_turn_per_eviction_ack',sid)
            if matching:
                t=matching[0];commits=[r for r in srv if r['event']=='evict_commit' and r.get('operation_id')==t['request_id']]
                check(len(commits)==1 and commits[0].get('acknowledgement')==ack,'service_commits_exact_engine_ack',t['request_id'])
                check(ack.get('evicted_tokens')==t['removed_tokens'],'ack_matches_http_eviction_count',t['request_id'])
            if expected_proofs:
                check(all((p:=c.get('byte_verification',{})).get('equal') is True
                          and p.get('before')==p.get('after') for c in copies),'retained_bytes_equal',sid)
            eviction_profiles.append(dict(session_id=sid,turn=event.get('turn'),
                 evicted_tokens=ack.get('evicted_tokens'),core=ack.get('core_profile'),
                 worker=[r.get('worker_profile') for r in receipts]))
        closes=[r for r in rows if r['event']=='close_released']
        check(len(closes)==1 and not any(closes[0].get('block_ids',[[]]))
              and closes[0].get('request_still_registered') is False,'all_owned_pages_released',sid)
    return dict(valid=not failures,checks=dict(counts),failures=failures,eviction_profiles=eviction_profiles,
                byte_proofs_required=expected_proofs)


def boundary_guards(engine,operations,*,epoch_origin,guard_seconds=2):
    """Exclude dispatch through authoritative resume+guard, not the whole decode.

    Full-engine pause/drain can delay a later lane well beyond dispatch+2s.
    Matching the engine resume handles that delay. If the audit entry is absent
    conservatively exclude that operation's full HTTP interval.
    """
    guards=[]
    for operation in operations:
        if operation.get('phase')!='measured' or not operation.get('session_id'):continue
        sid=operation['session_id'];pattern=re.compile(re.escape('arc3-persistent-'+sid)+r'(?:-[0-9a-f]{8})?\Z')
        entries=[r for r in engine if r['event']=='resume' and r.get('turn')==operation['turn']
                 and pattern.fullmatch(r.get('request_id',''))]
        start=operation['start_seconds']
        finish=max(start,entries[0]['time']-epoch_origin)+guard_seconds if len(entries)==1 else operation['end_seconds']
        guards.append((max(0,start-guard_seconds),finish))
    return sorted(guards)


def main():
    p=argparse.ArgumentParser();p.add_argument('directory',type=Path);a=p.parse_args()
    result=json.loads((a.directory/'results.json').read_text())
    samples=[json.loads(line) for line in (a.directory/'metrics.jsonl').read_text().splitlines() if line]
    report=summarize(result,samples)
    destination=a.directory/'analysis.json'
    if destination.exists():raise RuntimeError('Analysis already exists; do not overwrite evidence')
    destination.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))


if __name__=='__main__':main()
