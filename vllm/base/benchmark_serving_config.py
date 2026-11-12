"""Bounded FIFO replay of fixed Duck inputs for serving configuration comparisons.

Natural stopping and forced output length are separate measurements. The long
synthetic panel tests pressure, not puzzle quality. No game actions are taken.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import threading
import time
import uuid

import requests

from game_allocation import Allocation
from gameplay_config import GameplayConfig, tokenization_payload
from minimal_inference import ROOT, MODEL_NAME, HOST, PORT, IDENTITY, matches, owned_snapshot, save
from profile_monitor import Monitor

BASE = f'http://{HOST}:{PORT}'


def post(route, payload):
    r = requests.post(BASE+route, json=payload, timeout=600)
    r.raise_for_status()
    return r.json()


def get_metrics():
    r = requests.get(BASE+'/metrics', timeout=20)
    r.raise_for_status()
    return r.text


def metric_values(text):
    result = {}
    for line in text.splitlines():
        if not line or line.startswith('#'): continue
        name = line.split('{', 1)[0].split(' ', 1)[0]
        if name.endswith(('_total', '_sum', '_count')):
            result[name] = result.get(name, 0.)+float(line.rsplit(' ', 1)[1])
    return result


def idle():
    deadline = time.monotonic()+45
    while True:
        text = get_metrics()
        vals = [float(x.split()[-1]) for x in text.splitlines()
                if x.startswith(('vllm:num_requests_running{', 'vllm:num_requests_waiting{'))]
        if len(vals) == 2 and not any(vals): return text
        if time.monotonic() > deadline: raise TimeoutError('Expected idle engine')
        time.sleep(.2)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--slots', default='8', help='Distinct admission settings to compare')
    p.add_argument('--streams', type=int, choices=[8,16], default=8,
                   help='16 repeats the eight fixed streams twice with independent request identities')
    p.add_argument('--reference', type=Path, help='Prior correctness.json for cross-config checks')
    p.add_argument('--cross-config-policy', choices=['record', 'strict'], default='record',
                   help='Record numerical differences by default; strict is a diagnostic gate, not a correctness oracle')
    p.add_argument('--natural-rounds', type=int, default=2)
    p.add_argument('--correctness-only', action='store_true')
    p.add_argument('--expected-qsa', choices=['exact', 'radix'], default='exact')
    p.add_argument('--eager', action='store_true', help='Require eager mode for a same-process selector diagnostic')
    p.add_argument('--expected-kv-format',choices=['bf16','fp8_e4m3','fp8_e5m2','nvfp4'],default='bf16')
    p.add_argument('--expected-kv-gib',type=float,default=8)
    p.add_argument('--expected-context',type=int,default=32768)
    p.add_argument('--expected-prefill-profile',choices=['stock','tile32'],default='stock')
    p.add_argument('--expected-fp8-reader',choices=['baseline','scale_placement'],default='baseline')
    args = p.parse_args()
    slots = [int(x) for x in args.slots.split(',')]
    assert len(slots) == len(set(slots)) and all(1 <= x <= args.streams for x in slots)
    assert args.natural_rounds >= 1
    args.output.mkdir(parents=True, exist_ok=False)
    identity = json.loads(IDENTITY.read_text())
    assert matches(identity['root'])
    identity['processes'] = owned_snapshot(identity); save(IDENTITY, identity)
    argv = identity['argv']
    assert argv[argv.index('--kv-cache-memory-bytes')+1] == str(int(args.expected_kv_gib*2**30))
    assert argv[argv.index('--max-model-len')+1] == str(args.expected_context)
    assert identity['environment'].get('ARC3_QSA_KV_FORMAT','bf16')==args.expected_kv_format
    assert identity['environment'].get('ARC3_QSA_FP8_READER','baseline')==args.expected_fp8_reader
    assert identity['environment'].get('ARC3_QSA_PREFILL_PROFILE','stock')==args.expected_prefill_profile
    engine_sequences = int(argv[argv.index('--max-num-seqs')+1])
    assert engine_sequences in [8,16] and max(slots) <= engine_sequences
    assert '--no-enable-prefix-caching' in argv
    assert ('--enforce-eager' in argv) == args.eager
    if args.eager:
        runtime_mode=post('/collective_rpc', dict(method='arc3_determinism',args=['status'],timeout=30))
        def modes(x):
            if isinstance(x,dict):
                if 'qsa' in x and 'moe' in x:yield x
                else:
                    for v in x.values():yield from modes(v)
            elif isinstance(x,list):
                for v in x:yield from modes(v)
        live_modes=list(modes(runtime_mode))
        assert len(live_modes)==1 and live_modes[0]['qsa']==args.expected_qsa and live_modes[0]['moe']=='unfused'
    else:
        assert identity['environment']['ARC3_QSA_MODE'] == args.expected_qsa
    assert identity['environment']['ARC3_MOE_FINALIZE'] == 'unfused'
    assert identity['environment']['ARC3_MTP_MIN_P'] == '1'
    raw = (ROOT/'replay-panel-v1.json').read_bytes(); panel = json.loads(raw)
    for name in ['benchmark_serving_config.py', 'game_allocation.py', 'gameplay_config.py',
                 'minimal_inference.py', 'profile_monitor.py', 'server-identity.json']:
        (args.output/name).write_bytes((ROOT/name).read_bytes())
    (args.output/'panel.json').write_bytes(raw)
    config = GameplayConfig()
    inputs = [[config.chat_payload(MODEL_NAME, step['request']['messages'],
               step['request']['tools'], max_tokens=1024, seed=1000+10*i+j)
               for j, step in enumerate(panel['streams'][i%8]['steps'])] for i in range(args.streams)]
    counts = [[post('/tokenize', tokenization_payload(x))['count'] for x in group] for group in inputs]
    assert len(inputs) == args.streams and all(len(x) == 3 for x in inputs)
    assert max(x for group in counts for x in group)+1024 <= 32768
    settings = dict(slots=slots, engine_sequences=engine_sequences, streams=args.streams,
        stream_expansion='fixed eight streams repeated twice when streams=16; independent IDs/seeds; prefix off',
        gameplay=config.manifest(), qsa_mode=args.expected_qsa, eager=args.eager,
        kv_format=args.expected_kv_format, kv_gib=args.expected_kv_gib,context=args.expected_context,
        fp8_reader=args.expected_fp8_reader,
        cross_config_policy=args.cross_config_policy,
        panel_sha256=hashlib.sha256(raw).hexdigest(), prompt_counts=counts,
        natural_max_tokens=1024, forced_tokens=512, pressure_input_tokens=28000,
        pressure_output_tokens=256, seed_schedule='1000+10*stream+step, unchanged across rounds/configurations',
        warmup='full natural replay for each slot count before its measured work',
        scope='fixed-input serving and synthetic pressure; not gameplay quality')
    save(args.output/'settings.json', settings)
    progress = dict(stage='correctness', started_epoch=time.time())
    def state(stage, **fields):
        progress.update(stage=stage, **fields); save(args.output/'progress.json', progress)
        print(json.dumps(progress), flush=True)

    # Repeated greedy probes catch gross startup/model-state regressions, and
    # record cross-configuration first-position differences without silently
    # treating stochastic continuation changes as a correctness test.
    correctness = []
    for i in [0, 2, 5, 7]:
        group = []
        for repeat in range(2):
            idle()
            payload = dict(inputs[i][-1], temperature=0., min_p=0., max_tokens=32,
                           logprobs=True, top_logprobs=20, cache_salt=uuid.uuid4().hex)
            response = post('/v1/chat/completions', payload)
            assert response['usage']['prompt_tokens'] == counts[i][-1]
            group.append(response)
        a = group[0]['choices'][0]['logprobs']['content']
        b = group[1]['choices'][0]['logprobs']['content']
        same = a == b
        correctness.append(dict(stream=i, repeated_logprobs_exact=same, responses=group))
    save(args.output/'correctness.json', correctness)
    assert all(x['repeated_logprobs_exact'] for x in correctness), 'Within-config repeatability failed'
    if args.expected_kv_format.startswith('fp8_'):
        # Real correctness requests have exercised attention even if a future
        # runtime chooses not to execute this reader during startup warmup.
        reader=post('/collective_rpc',dict(method='arc3_qsa_reader',timeout=30))
        save(args.output/'reader-status.json',reader)
        assert len(reader['results'])==1
        resolved=reader['results'][0]
        assert resolved['fp8_reader']==args.expected_fp8_reader and resolved['calls'][args.expected_fp8_reader]>0
    if args.reference:
        ref = json.loads(args.reference.read_text()); cross = []
        for before, after in zip(ref, correctness):
            assert before['stream'] == after['stream']
            a = before['responses'][0]['choices'][0]['logprobs']['content']
            b = after['responses'][0]['choices'][0]['logprobs']['content']
            aa = {x['token']:x['logprob'] for x in a[0]['top_logprobs']}
            bb = {x['token']:x['logprob'] for x in b[0]['top_logprobs']}
            common = aa.keys() & bb.keys()
            cross.append(dict(stream=after['stream'], first_token_match=a[0]['token']==b[0]['token'],
                common_top_tokens=len(common), max_first_logprob_diff=max(abs(aa[k]-bb[k]) for k in common),
                continuation_tokens_match=[x['token'] for x in a]==[x['token'] for x in b]))
        save(args.output/'cross-config-correctness.json', cross)
        if args.cross_config_policy == 'strict':
            assert all(x['first_token_match'] and x['common_top_tokens'] == 20 and
                       x['max_first_logprob_diff'] < .02 for x in cross), 'Cross-config numerical screening threshold exceeded'

    if args.correctness_only:
        state('complete', correctness_only=True, finished_epoch=time.time())
        return

    seed = post('/tokenize', dict(model=MODEL_NAME, prompt='Reference: amber is 17, cobalt is 29, jade is 43.\n'))['tokens']
    pressure = []
    for i in range(args.streams):
        lead = post('/tokenize', dict(model=MODEL_NAME, prompt=f'Independent stream {i}.\n'))['tokens']
        tokens = lead+(seed*((28000-len(lead)+len(seed)-1)//len(seed)))[:28000-len(lead)]
        pressure.append([dict(model=MODEL_NAME, prompt=tokens, temperature=1., top_p=.95,
            top_k=20, min_p=.02, seed=2000+i, max_tokens=256, min_tokens=256, ignore_eos=True)])
    save(args.output/'pressure-inputs.json', pressure)

    rounds = []
    def run(kind, admission, number, warm=False):
        label = f'{kind}-c{admission}-r{number}'; dest = args.output/label
        dest.mkdir(); before = idle(); (dest/'metrics-before.prom').write_text(before)
        data = pressure if kind == 'pressure' else inputs
        alloc = Allocation([str(i) for i in range(args.streams)], slots=admission, token_cap=100000,
                           deadline=time.time()+1200, output=dest)
        lock = threading.Lock(); rows = []
        monitor = Monitor(dest, BASE, IDENTITY).start()
        def stream(i, steps):
            for j, source in enumerate(steps):
                payload = dict(source, cache_salt=uuid.uuid4().hex)
                if kind == 'forced': payload.update(max_tokens=512, min_tokens=512, ignore_eos=True)
                with alloc.admit(str(i)) as ticket:
                    started = time.monotonic()
                    response = post('/v1/completions' if kind == 'pressure' else '/v1/chat/completions', payload)
                    seconds = time.monotonic()-started
                    usage = response['usage']
                    assert usage['prompt_tokens'] == (28000 if kind == 'pressure' else counts[i][j])
                    if kind != 'natural': assert usage['completion_tokens'] == (256 if kind=='pressure' else 512)
                    assert not (usage.get('prompt_tokens_details') or {}).get('cached_tokens',0)
                    alloc.record(str(i), dict(completion_tokens=usage['completion_tokens'],
                        charged_tokens=usage['completion_tokens'], prompt_tokens=usage['prompt_tokens'],
                        cached_tokens=0, serving_seconds=seconds))
                    row = dict(stream=i, step=j, seconds=seconds, queue_seconds=ticket['queue_seconds'], response=response)
                    with lock:
                        rows.append(row)
                        with (dest/'responses.jsonl').open('a') as f: f.write(json.dumps(row)+'\n')
            alloc.finish(str(i))
        try:
            start = time.monotonic()
            with ThreadPoolExecutor(max_workers=args.streams) as pool:
                jobs = [pool.submit(stream, i, group) for i, group in enumerate(data)]
                for job in jobs: job.result()
            seconds = time.monotonic()-start
        finally:
            monitor.stop(); alloc.save()
        after = idle(); (dest/'metrics-after.prom').write_text(after)
        a, b = metric_values(before), metric_values(after)
        delta = {k:b[k]-a.get(k,0) for k in b}
        tokens = sum(x['response']['usage']['completion_tokens'] for x in rows)
        def avg(name):
            n=delta.get('vllm:'+name+'_count',0)
            return delta.get('vllm:'+name+'_sum',0)/n if n else None
        drafted=delta.get('vllm:spec_decode_num_draft_tokens_total',0)
        r=dict(kind=kind, slots=admission, round=number, warmup=warm, seconds=seconds,
            requests=len(rows), output_tokens=tokens, tokens_per_second=tokens/seconds,
            preemptions=delta.get('vllm:num_preemptions_total'),
            acceptance=delta.get('vllm:spec_decode_num_accepted_tokens_total',0)/drafted if drafted else None,
            mean_ttft_seconds=avg('time_to_first_token_seconds'),
            mean_prefill_seconds=avg('request_prefill_time_seconds'),
            mean_engine_queue_seconds=avg('request_queue_time_seconds'),
            mean_admission_queue_seconds=sum(x['queue_seconds'] for x in rows)/len(rows),
            metrics_delta=delta, valid=len(rows)==args.streams*(1 if kind=='pressure' else 3))
        assert r['valid']; save(dest/'result.json', r)
        rounds.append(r); save(args.output/'result.json',dict(valid=True, rounds=rounds))
        print(json.dumps({k:v for k,v in r.items() if k!='metrics_delta'}), flush=True)

    for c in slots:
        state('warmup', slots=c); run('natural', c, -1, warm=True)
    for number in range(args.natural_rounds):
        for c in (slots if number%2==0 else list(reversed(slots))):
            state('natural', slots=c, round=number); run('natural',c,number)
    for c in reversed(slots):
        state('forced', slots=c); run('forced',c,0)
        state('pressure', slots=c); run('pressure',c,0)
    state('complete', finished_epoch=time.time())


if __name__ == '__main__': main()
