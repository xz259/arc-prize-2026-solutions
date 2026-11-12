"""Offline campaign report. Never treats summed request latency as GPU stall."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

import analyze


SCOPES={
    'endpoint':('total_seconds',('lock_wait_seconds','status_seconds','drain_seconds','core_rpc_seconds','resume_seconds')),
    'core':('total_before_final_audit_seconds',('prepare_seconds','wire_and_prepare_audit_seconds','worker_rpc_seconds',
                                           'publish_release_optional_allocator_proof_seconds')),
    'worker':('total_seconds',('initial_fence_seconds','prepare_validate_seconds','copy_and_optional_proofs_seconds','mapping_publish_seconds')),
}
SCOPE_NOTES={
    'endpoint':'Elapsed transaction latency includes lock wait. Concurrent lock waits overlap; its sum is not global stall time.',
    'core':'Nested inside endpoint.core_rpc_seconds. Excludes final engine audit/return transport; never add its total to endpoint total.',
    'worker':'Nested inside core.worker_rpc_seconds. Copy interval includes its ending CUDA fence and optional byte proofs; never add it to parent RPC.',
}


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line]


def stats(values):
    """Nearest-rank p95; totals are transaction-seconds, not global duration."""
    values=sorted(float(x) for x in values)
    if any(not math.isfinite(x) or x<0 for x in values):raise ValueError('Invalid nonnegative timing')
    if not values:return {'count':0,'sum_seconds':None,'median_seconds':None,'p95_seconds':None}
    return dict(count=len(values),sum_seconds=sum(values),median_seconds=statistics.median(values),
                p95_seconds=values[math.ceil(.95*len(values))-1])


def profiles(service,operations):
    """Use service commits once; acknowledgement is also echoed in engine log."""
    measured={(r['session_id'],r['request_id']):r for r in operations
              if r.get('phase')=='measured' and r.get('removed_tokens',0)>0}
    commits=[r for r in service if r.get('event')=='evict_commit'
             and (r.get('session_id'),r.get('operation_id')) in measured]
    occurrences=Counter((r['session_id'],r['operation_id']) for r in commits)
    errors=[]
    if set(occurrences)!=set(measured) or any(n!=1 for n in occurrences.values()):
        errors.append('Measured eviction commits are missing or duplicated; phase statistics are incomplete')
    rows={scope:[] for scope in SCOPES};proof_flags=[]
    for commit in commits:
        ack=commit.get('acknowledgement',{});workers=ack.get('worker_receipts',[])
        if len(workers)!=1:errors.append('Expected exactly one worker receipt per measured eviction')
        if commit.get('endpoint_profile'):rows['endpoint'].append(commit['endpoint_profile'])
        if ack.get('core_profile'):rows['core'].append(ack['core_profile'])
        for worker in workers:
            proof_flags.append(worker.get('byte_verification_enabled'))
            if worker.get('worker_profile'):rows['worker'].append(worker['worker_profile'])
    report={}
    for scope,(total,parts) in SCOPES.items():
        entries=rows[scope];fields=(total,*parts)
        missing=sum(any(k not in row for k in fields) for row in entries)
        if len(entries)!=len(commits) or missing:errors.append(f'{scope} profile missing or incomplete')
        residual=[row[total]-sum(row[k] for k in parts) for row in entries if all(k in row for k in fields)]
        if any(x< -1e-6 for x in residual):errors.append(f'{scope} children exceed containing scope')
        report[scope]=dict(note=SCOPE_NOTES[scope],transaction_count=len(entries),
            phases={k:stats(row[k] for row in entries if k in row) for k in fields},
            # Clamp only sub-microsecond timer roundoff, after checking it.
            uninstrumented_between_sibling_scopes=stats(max(0,x) for x in residual))
    return dict(expected_measured_evictions=len(measured),measured_commit_count=len(commits),
                valid=not errors,errors=errors,byte_proof_modes=sorted({str(x) for x in proof_flags}),
                scopes=report,
                aggregation_note='Phase sums are separate aggregates of nested transaction scopes. They are not additive across scopes, and no sum is labeled global GPU stall.')


def work_totals(engine,operations):
    measured=[r for r in operations if r.get('phase')=='measured' and r.get('session_id')]
    values=[];errors=[]
    for op in measured:
        pattern=re.compile(re.escape('arc3-persistent-'+op['session_id'])+r'(?:-[0-9a-f]{8})?\Z')
        pauses=[r for r in engine if r.get('event')=='pause' and r.get('turn')==op['turn']
                and pattern.fullmatch(r.get('request_id',''))]
        if len(pauses)!=1:errors.append(f'Missing/duplicate turn-work receipt: {op["request_id"]}');continue
        work=pauses[0].get('turn_work',{})
        if not all(k in work for k in ('prefill_tokens','decode_tokens','scheduled_tokens')):
            errors.append('Incomplete turn-work receipt');continue
        values.append(work)
    return dict(valid=not errors,errors=errors,measured_turns=len(measured),audited_turns=len(values),
                actual_prefill_positions=sum(v['prefill_tokens'] for v in values),
                actual_decode_positions=sum(v['decode_tokens'] for v in values),
                actual_scheduled_positions=sum(v['scheduled_tokens'] for v in values))


def stateless_variant(directory,expected_requests):
    """Classify saved wire payloads, not filenames or an inferred client version."""
    payloads=[]
    for path in sorted((directory/'http').glob('*.json')):
        row=json.loads(path.read_text())
        if row.get('method')=='POST' and row.get('route')=='/v1/completions':
            payloads.append(row.get('payload',{}))
    if len(payloads)!=expected_requests or not all(p.get('ignore_eos') is True for p in payloads):
        return 'unverified'
    values={p.get('min_tokens',0) for p in payloads}
    if values=={0}:return 'unmasked'
    if values=={3999}:return 'masked-eos-control'
    return 'mixed-or-unexpected'


def load_run(directory):
    directory=Path(directory).resolve();result=json.loads((directory/'results.json').read_text())
    if result.get('config',{}).get('arm') not in ('stateless','persistent','persistent-api','eviction'):
        raise ValueError(f'{directory}: not a scaling benchmark result; proof gates are not throughput arms')
    samples=read_jsonl(directory/'metrics.jsonl')
    summary=analyze.summarize(result,samples)
    operations=result.get('operations',[]);measured=[r for r in operations if r.get('phase')=='measured']
    service=read_jsonl(directory/'service-owned.jsonl') if (directory/'service-owned.jsonl').exists() else []
    engine=read_jsonl(directory/'engine-owned.jsonl') if (directory/'engine-owned.jsonl').exists() else []
    profile=profiles(service,operations)
    stateful=result['config']['arm']!='stateless'
    work=work_totals(engine,operations) if stateful else None
    optimization=result.get('identity',{}).get('persistent_experiment',{}).get('optimization_config',{})
    proofs_off=(result['config'].get('expected_proofs')==0 and optimization.get('verify_bytes')==0
                and optimization.get('verify_allocator')==0 and not any(p!='False' for p in profile['byte_proof_modes']))
    eligible=(result.get('status')=='passed' and summary.get('valid') and proofs_off
              and (not stateful or (result.get('audit',{}).get('valid') and work['valid'])))
    if not profile['valid'] and result['config']['arm']=='eviction':eligible=False
    identity=result.get('identity',{});manifest=identity.get('persistent_experiment',{}).get('manifest',{})
    signature=dict(server_root=identity.get('root'),source_manifest=manifest,
                   client_hashes=result.get('source_sha256'),tokenizer=result.get('tokenizer_sha256'),
                   concurrency=result['config']['concurrency'],utilization=result['config']['utilization'],
                   model=result['config']['model'],optimization=optimization)
    variant=None if stateful else stateless_variant(directory,result['config']['concurrency'])
    return dict(directory=str(directory),arm=result['config']['arm'],concurrency=result['config']['concurrency'],
        status=result.get('status'),performance_eligible=bool(eligible),proofs_off=proofs_off,
        stateless_variant=variant,primary_comparison_eligible=bool(eligible and (stateful or variant=='unmasked')),
        signature_sha256=hashlib.sha256(json.dumps(signature,sort_keys=True).encode()).hexdigest(),
        received_setup_tokens=sum(r['output_tokens'] for r in operations if r.get('phase')=='setup'),
        measured_new_input_tokens=sum(r['delta_tokens'] for r in measured),
        measured_output_tokens=sum(r['output_tokens'] for r in measured),
        measured_evicted_tokens=sum(r.get('removed_tokens',0) for r in measured),
        summary=summary,engine_work=work,eviction_profiles=profile,
        client_measured_prepare=stats(r['client_prepare_seconds'] for r in measured if 'client_prepare_seconds' in r),
        client_measured_verify=stats(r['client_verify_seconds'] for r in measured if 'client_verify_seconds' in r),
        client_timing_note='Ledger preparation/verification is inside inclusive time. Recorded client wall durations are not a measure of GPU idle time.',
        artifacts_sha256={name:hashlib.sha256((directory/name).read_bytes()).hexdigest()
            for name in ('results.json','metrics.jsonl','engine-owned.jsonl','service-owned.jsonl') if (directory/name).exists()})


def render(runs):
    def number(value):return '—' if value is None else f'{value:,.2f}'
    primary=[r for r in runs if r['arm']!='stateless' or r.get('stateless_variant')=='unmasked']
    secondary=[r for r in runs if r not in primary]
    lines=['# Near-40K eviction campaign summary','',
        '| Arm | N | Eligible | Output tokens | Inclusive s | Inclusive tok/s | Guarded tok/s | Guarded s | Preemptions |',
        '| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for run in primary:
        s=run['summary'];p=s.get('pure_decode',{});eligible=run['performance_eligible']
        lines.append(f'| {run["arm"]} | {run["concurrency"]} | {eligible} | {run["measured_output_tokens"]:,} | '
            f'{number(s.get("inclusive_seconds"))} | {number(s.get("received_tokens_per_second") if eligible else None)} | '
            f'{number(p.get("tokens_per_second") if eligible else None)} | {number(p.get("pooled_seconds"))} | {s.get("preemptions","—")} |')
    lines+=['','Stateless inclusive time includes initial prefill. Persistent inclusive time starts after the seed barrier and includes all measured API pauses, evictions, pending-token prefills and tail latency. Guarded rates are sampled occupancy estimates over 36,000–39,999 tokens, not the historical strict late-38K subset. They are not substitutes for the inclusive rate.','']
    if not any(r['arm']=='stateless' for r in primary):
        lines+=['Corrected unmasked stateless baseline is not yet included. The original masked-EOS run is a secondary control.','']
    if secondary:
        lines+=['## Secondary stateless controls','',
                '| Variant | Inclusive tok/s | Guarded tok/s | Primary comparison |',
                '| --- | ---: | ---: | --- |']
        for run in secondary:
            s=run['summary'];eligible=run['performance_eligible']
            lines.append(f'| {run.get("stateless_variant")} | {number(s.get("received_tokens_per_second") if eligible else None)} | '
                         f'{number(s.get("pure_decode",{}).get("tokens_per_second") if eligible else None)} | Excluded |')
        lines+=['','Variant is verified from saved HTTP payloads. `min_tokens=3999` adds EOS masking despite `ignore_eos=True`; persistent arms and the corrected stateless baseline use no minimum-token mask.','']
    if len({r['signature_sha256'] for r in primary})>1:
        lines+=['**Configuration warning:** candidate/client/tokenizer/occupancy signatures differ. Do not interpret cross-run rate differences as a matched causal comparison.','']
    for run in runs:
        lines += [f'## {run["arm"]}: {Path(run["directory"]).name}','',f'Status: {run["status"]}; performance eligible: {run["performance_eligible"]}; proofs off: {run["proofs_off"]}.',
                  f'New measured input: {run["measured_new_input_tokens"]:,}; measured output: {run["measured_output_tokens"]:,}; seed output: {run["received_setup_tokens"]:,}; removed resident tokens: {run["measured_evicted_tokens"]:,}.']
        if run['engine_work']:
            w=run['engine_work'];lines.append(f'Engine work: {w["actual_prefill_positions"]:,} actual prefill positions, {w["actual_decode_positions"]:,} decode positions, {w["actual_scheduled_positions"]:,} total scheduled positions; {w["audited_turns"]}/{w["measured_turns"]} measured turns audited.')
        lines+=['',run['client_timing_note']]
        for name in ('client_measured_prepare','client_measured_verify'):
            s=run[name];lines.append(f'{name}: sum {number(s["sum_seconds"])} s; median {number(None if s["median_seconds"] is None else s["median_seconds"]*1000)} ms; p95 {number(None if s["p95_seconds"] is None else s["p95_seconds"]*1000)} ms.')
        p=run['eviction_profiles']
        if p['expected_measured_evictions']:
            lines+=['',f'Eviction profiles: {p["measured_commit_count"]}/{p["expected_measured_evictions"]} commits; valid: {p["valid"]}.',p['aggregation_note']]
            if not run['proofs_off']:
                lines+=['Proofs are enabled or unverified: timing values are retained in JSON only and are not presented as performance measurements.'];continue
            for scope,entry in p['scopes'].items():
                lines+=['',f'**{scope} scope.** {entry["note"]}','',
                        '| Phase | Samples | Sum transaction s | Median ms | p95 ms |','| --- | ---: | ---: | ---: | ---: |']
                for name,s in entry['phases'].items():
                    lines.append(f'| {name} | {s["count"]} | {number(s["sum_seconds"])} | {number(None if s["median_seconds"] is None else s["median_seconds"]*1000)} | {number(None if s["p95_seconds"] is None else s["p95_seconds"]*1000)} |')
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directories',nargs='+',type=Path)
    parser.add_argument('--output-prefix',type=Path,required=True)
    args=parser.parse_args();jsonpath=Path(str(args.output_prefix)+'.json');mdpath=Path(str(args.output_prefix)+'.md')
    if jsonpath.exists() or mdpath.exists():parser.error('Refusing to overwrite an existing campaign report')
    runs=[load_run(path) for path in args.directories]
    primary=[r for r in runs if r['arm']!='stateless' or r.get('stateless_variant')=='unmasked']
    report=dict(runs=runs,matched_configuration=len({r['signature_sha256'] for r in primary})<=1,
                analyzer_sha256=hashlib.sha256(Path(analyze.__file__).read_bytes()).hexdigest(),
                summary_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    jsonpath.write_text(json.dumps(report,indent=2)+'\n');mdpath.write_text(render(runs))
    print(mdpath)


if __name__=='__main__':main()
