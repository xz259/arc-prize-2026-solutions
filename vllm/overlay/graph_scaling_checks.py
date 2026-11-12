"""Bounded FULL25/FULL30 persistent correctness gate at near-40K residency.

No model launch or remote transport. --execute targets only the supplied owned
loopback server. Byte and allocator proofs must be enabled; this is not a speed
measurement. Dependencies may be flat beside this file or selected explicitly
with --candidate-dir (the local working candidate is the final fallback).
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import fcntl
import hashlib
import json
from pathlib import Path
import sys
import threading
import time
import traceback
from urllib.parse import urlparse
import uuid


def helper_directory(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--candidate-dir')
    preliminary, _ = parser.parse_known_args(argv)
    if preliminary.candidate_dir:
        candidates = [Path(preliminary.candidate_dir).resolve()]
    else:
        here = Path(__file__).resolve().parent
        candidates = [here, here.parent / 'persistent_eviction_optimization_20260913']
    for path in candidates:
        if (path / 'raw_history_eviction_checks.py').is_file():
            return path
    raise ImportError('Supply --candidate-dir containing the pinned persistent correctness helpers')


HELPERS = helper_directory()
sys.path.insert(0, str(HELPERS))
from raw_concurrent_eviction_checks import identity_check, graph_status, slot_for
from raw_gpu_checks import ContractFailure, prompt_factory, write_json
from raw_history_eviction_checks import HistoryClient, MODE, audit_events, engine_for
from raw_managed_eviction_checks import LOCKS

CONTEXT = 40000
INPUT_LENGTHS = (5000, 10000, 10000, 10000, 3500)
WARM_OUTPUT = 8
PRE_OUTPUT = 1024
EVICT_INPUT = 4000
EVICT_OUTPUT = 1024
POST_OUTPUT = 512
SUFFIX = [1401, 1914]
PREFIX = 128
TOKENS_PER_OWNER = len(INPUT_LENGTHS) * WARM_OUTPUT + PRE_OUTPUT + EVICT_OUTPUT + POST_OUTPUT


def plan_description(concurrencies=(25, 30)):
    return dict(mode=MODE, concurrencies=list(concurrencies), sequential_cohorts=True,
        context_limit=CONTEXT, maximum_generated_tokens=TOKENS_PER_OWNER * sum(concurrencies),
        operations_per_owner=8, input_lengths_per_observation=list(INPUT_LENGTHS) + [EVICT_INPUT],
        output_tokens=dict(warm_observation=WARM_OUTPUT, pre_eviction_burst=PRE_OUTPUT,
                           eviction_append=EVICT_OUTPUT, post_eviction_burst=POST_OUTPUT),
        expected_residency=dict(after_warmup=38540, before_eviction=39564,
            after_eviction_input=38171, after_eviction_output=39195, final=39707),
        expected_eviction=dict(transactions_per_owner=1, suffix_tokens=513,
            oldest_interaction_tokens=4880, total_removed_tokens=5393,
            selected_block_ids=['obs-01']),
        fixed_prefix_tokens=PREFIX, managed_suffix_local_range=SUFFIX,
        required_graph_batches=[dict(mode='FULL', real_requests=n, padded_requests=n)
                                for n in concurrencies],
        required_proofs=['before/after retained bytes', 'allocator release/reacquisition',
            'exact external and resident ledgers', 'no historical re-prefill',
            'all live owners have disjoint pages within each cache group',
            'FULLN observed before and after eviction, plus completed response/engine audit'],
        maximum_seconds=2400, cleanup_reserve_seconds_per_cohort=45,
        exclusions=['throughput', 'gameplay quality', 'exact cross-schedule model outputs',
                    'new padded-row or slot-reuse coverage'])


class ScalingClient(HistoryClient):
    """Reuse the exact independent oracle/auditor with explicit larger caps."""
    def set_output_tokens(self, count):
        if type(count) is not int or count not in (WARM_OUTPUT, POST_OUTPUT, PRE_OUTPUT):
            raise ValueError('Scaling diagnostics permit only fixed 8, 512 or 1024-token outputs')
        self.output_tokens = count


def exact_batch_delta(before, after, count):
    def value(state):
        return sum(row['count'] for row in state['prepared_batches']
            if (row['mode'], row['real_requests'], row['padded_requests']) == ('FULL', count, count))
    delta = value(after) - value(before)
    if delta < 0:
        raise ContractFailure('Worker graph counters reset during the owned control')
    return dict(mode='FULL', real_requests=count, padded_requests=count, count=delta)


def assert_resident_slots(state, clients):
    slots = [slot_for(state, client.sid) for client in clients]
    if len(set(slots)) != len(clients):
        raise ContractFailure('Different live owners share a worker state slot')
    return dict(zip((client.sid for client in clients), slots))


def assert_owner_pages(engine_path, clients):
    """All owners are paused at this snapshot; compare their latest ledgers."""
    owners = {}
    snapshots = {}
    for client in clients:
        pauses = [row for row in engine_for(engine_path, client.sid) if row['event'] == 'pause']
        if len(pauses) != len(client.turns) or not pauses:
            raise ContractFailure('Missing completed pause snapshot for ' + client.sid)
        pause = pauses[-1]
        snapshots[client.sid] = pause['block_ids']
        for group, blocks in enumerate(pause['block_ids']):
            for block in blocks:
                key = (group, block)
                if key in owners and owners[key] != client.sid:
                    raise ContractFailure(f'Live cache page {key} belongs to multiple sessions')
                owners[key] = client.sid
    return dict(valid=True, snapshots=snapshots, distinct_group_page_pairs=len(owners))


def parallel_submit(clients, name, deltas, *, output_tokens, observation=None, suffix=None):
    if len(clients) != len(deltas):
        raise ValueError('Each owner needs exactly one independent input')
    barrier = threading.Barrier(len(clients))
    def invoke(client, delta):
        client.set_output_tokens(output_tokens)
        barrier.wait(timeout=30)
        return client.submit(name, delta, observation=observation, suffix=suffix)
    failures = []
    results = []
    with ThreadPoolExecutor(max_workers=len(clients)) as pool:
        futures = [pool.submit(invoke, client, delta) for client, delta in zip(clients, deltas)]
        # Never abandon sibling HTTP calls before cleanup closes their sessions.
        for future in futures:
            try:
                results.append(future.result())
            except BaseException as error:
                failures.append(repr(error))
    if failures:
        raise ContractFailure('Concurrent scaling operation failed: ' + str(failures))
    return results


def exercise(clients, prompt, *, on_phase=None):
    def phase(name):
        if on_phase:
            on_phase(name)
    for index, count in enumerate(INPUT_LENGTHS):
        deltas = [prompt(count, (lane * 701 + index * 1301) % 20000)
                  for lane in range(len(clients))]
        if any(len(delta) != count for delta in deltas):
            raise ContractFailure('Tokenizer produced the wrong planned input length')
        parallel_submit(clients, f'warm-observation-{index+1}', deltas,
            output_tokens=WARM_OUTPUT, observation=(f'obs-{index+1:02d}', PREFIX if index == 0 else 0),
            suffix=SUFFIX if index == len(INPUT_LENGTHS)-1 else None)
        phase(f'warm-{index+1}')
    phase('before-pre-burst')
    parallel_submit(clients, 'full-concurrency-before-eviction', [[] for _ in clients],
                    output_tokens=PRE_OUTPUT)
    phase('after-pre-burst')
    deltas = [prompt(EVICT_INPUT, (lane * 701 + 9109) % 20000) for lane in range(len(clients))]
    if any(len(delta) != EVICT_INPUT for delta in deltas):
        raise ContractFailure('Tokenizer produced the wrong eviction input length')
    parallel_submit(clients, 'one-combined-history-suffix-eviction', deltas,
                    output_tokens=EVICT_OUTPUT, observation=('obs-06', 0))
    phase('before-post-burst')
    parallel_submit(clients, 'full-concurrency-after-eviction', [[] for _ in clients],
                    output_tokens=POST_OUTPUT)
    phase('after-post-burst')
    for client in clients:
        if len(client.turns) != 8:
            raise ContractFailure('Incorrect operation count')
        evictions = [turn for turn in client.turns if turn['removed_tokens']]
        if len(evictions) != 1:
            raise ContractFailure('Expected exactly one eviction transaction per owner')
        turn = evictions[0]
        receipt = turn['history_eviction']
        if (turn['turn'] != 6 or turn['removed_tokens'] != 5393
                or receipt['history_removed_tokens'] != 4880
                or receipt['suffix_removed_tokens'] != 513
                or receipt['selected_block_ids'] != ['obs-01']):
            raise ContractFailure('Unexpected combined eviction policy outcome')
        expected = {4:38540, 5:39564, 6:39195, 7:39707}
        if any(client.turns[index]['resident_tokens'] != value for index, value in expected.items()):
            raise ContractFailure('Unexpected near-limit resident lengths')


def check_launch(args):
    try:
        address = urlparse(args.base_url)
        local = (address.scheme == 'http' and address.hostname in ('127.0.0.1', 'localhost')
                 and address.port == 1234 and address.path in ('', '/')
                 and not address.query and not address.fragment
                 and address.username is None and address.password is None)
    except ValueError:
        local = False
    if not local:
        raise ContractFailure('Scaling control requires http://127.0.0.1:1234 or http://localhost:1234')
    args.expected_mode = 'graph'
    identity = identity_check(args)
    argv = identity.get('argv', [])
    def option(name):
        if argv.count(name) != 1:
            raise ContractFailure('Require explicit unique launch option ' + name)
        index = argv.index(name) + 1
        values = []
        while index < len(argv) and not argv[index].startswith('--'):
            values.append(argv[index]); index += 1
        return values
    captures = [int(value) for value in option('--cudagraph-capture-sizes')]
    if not {25, 30} <= set(captures):
        raise ContractFailure('Explicit native FULL25/FULL30 capture sizes are missing')
    if option('--max-num-seqs') != ['30'] or option('--max-model-len') != ['40000']:
        raise ContractFailure('Require the reviewed 30-request / 40K server ceilings')
    manifest = identity.get('persistent_experiment', {}).get('manifest', {})
    for name in ('raw_concurrent_eviction_checks.py', 'raw_history_eviction_checks.py',
                 'raw_managed_eviction_checks.py', 'raw_gpu_checks.py'):
        if name not in manifest:
            raise ContractFailure('Pinned launch manifest is missing correctness helper ' + name)
        actual = hashlib.sha256((HELPERS / name).read_bytes()).hexdigest()
        if actual != manifest[name]:
            raise ContractFailure('Correctness helper differs from the pinned launch: ' + name)
    return identity


def run_cohort(args, output, count, prompt, *, hard_deadline):
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = min(hard_deadline - 45, started + args.cohort_seconds - 45)
    tag = f'graph-scale-{count}-' + uuid.uuid4().hex[:10]
    supervisor_dir = output / 'supervisor'; supervisor_dir.mkdir()
    supervisor = HistoryClient(args.base_url, supervisor_dir, args.timeout, tag + '-supervisor', deadline=deadline)
    clients = []
    for lane in range(count):
        folder = output / f'lane-{lane:02d}'; folder.mkdir()
        clients.append(ScalingClient(args.base_url, folder, args.timeout, tag + f'-lane-{lane:02d}',
                                     deadline=deadline, budget=CONTEXT))
    result = dict(status='starting', concurrency=count, started=time.time(),
                  session_ids=[client.sid for client in clients], phases=[])
    def save():
        result['turns'] = {client.sid:client.turns for client in clients}
        write_json(output / 'results.json', result)
    try:
        health = supervisor.call('GET', '/arc3/persistent/health')
        if (health.get('mode') != MODE or health.get('max_context') != CONTEXT
                or health.get('max_sessions') != 30 or health.get('prefix_caching') is not False
                or any(state.get('state') not in ('closed', 'error')
                       for state in health.get('sessions', {}).values())):
            raise ContractFailure('Wrong server mode/limits or unrelated live session')
        state = graph_status(supervisor, 'graph')
        if not {25, 30} <= set(state.get('captured_token_counts', [])):
            raise ContractFailure('Worker does not report captured sizes 25 and 30')
        result['graph_initial'] = state
        def phase(name):
            result['phases'].append(dict(name=name, elapsed=time.monotonic()-started))
            if name in ('before-pre-burst', 'after-pre-burst', 'before-post-burst', 'after-post-burst'):
                state = graph_status(supervisor, 'graph')
                result.setdefault('graph_states', {})[name] = state
                result.setdefault('owner_slots', {})[name] = assert_resident_slots(state, clients)
                result.setdefault('owner_pages', {})[name] = assert_owner_pages(args.engine_events, clients)
                if name.startswith('after-'):
                    label = 'pre' if name == 'after-pre-burst' else 'post'
                    delta = exact_batch_delta(result['graph_states'][f'before-{label}-burst'], state, count)
                    result.setdefault('observed_full_batches', {})[label] = delta
                    if delta['count'] <= 0:
                        raise ContractFailure(f'FULL{count} not observed during completed {label}-eviction burst')
            result['status'] = 'running'; save()
        exercise(clients, prompt, on_phase=phase)
        result['status'] = 'completed_pending_audit'
    except BaseException as error:
        result.update(status='failed', error=dict(type=type(error).__name__, message=str(error),
                                                  traceback=traceback.format_exc()))
    finally:
        # parallel_submit joined every outstanding call before this point.
        with ThreadPoolExecutor(max_workers=count) as pool:
            futures = [pool.submit(client.cleanup) for client in clients]
            result['cleanup_errors'] = []
            for client, future in zip(clients, futures):
                try:
                    result['cleanup_errors'].extend(future.result())
                except BaseException as error:
                    result['cleanup_errors'].append(dict(session_id=client.sid, error=repr(error)))
        if result['cleanup_errors']:
            result['status'] = 'failed'
        try:
            health = supervisor.call('GET', '/arc3/persistent/health')
            result['final_owned_states'] = {client.sid:health.get('sessions', {}).get(client.sid) for client in clients}
            if any(client.turns and (result['final_owned_states'][client.sid] or {}).get('state') != 'closed'
                   for client in clients):
                raise ContractFailure('An owned completed session was not closed')
            result['audits'] = {client.sid:audit_events(args.engine_events, args.service_events,
                                  client.sid, client.turns) for client in clients if client.turns}
            if result['status'] == 'completed_pending_audit':
                if (len(result['audits']) != count
                        or not all(audit['valid'] and len(audit['copies']) == 1
                                   for audit in result['audits'].values())):
                    raise ContractFailure('Exact byte/page/ledger/no-refill audit failed')
                result['status'] = 'passed'
            for client in clients:
                rows = engine_for(args.engine_events, client.sid)
                (output / (client.sid + '-engine.jsonl')).write_text(''.join(json.dumps(row)+'\n' for row in rows))
            owned = {client.sid for client in clients}
            rows = [json.loads(line) for line in Path(args.service_events).read_text().splitlines() if line]
            (output / 'owned-service.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows if row.get('session_id') in owned))
        except BaseException as error:
            result.update(status='failed', final_audit_error=repr(error))
        result.update(ended=time.time(), elapsed=time.monotonic()-started,
            completed_output_tokens=sum(turn['output_tokens'] for client in clients for turn in client.turns))
        save()
    return result


def run(args):
    identity = check_launch(args)
    for path in (args.engine_events, args.service_events):
        if not Path(path).is_file():
            raise ContractFailure('Missing authoritative audit ' + path)
    output = Path(args.output).resolve(); output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic(); deadline = started + args.max_seconds
    result = dict(status='starting', plan=plan_description(args.concurrency), identity=identity,
        started=time.time(), cohorts=[], helper_directory=str(HELPERS),
        client_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        helper_sha256={name:hashlib.sha256((HELPERS/name).read_bytes()).hexdigest()
            for name in ('raw_concurrent_eviction_checks.py', 'raw_history_eviction_checks.py',
                         'raw_managed_eviction_checks.py', 'raw_gpu_checks.py')})
    write_json(output/'results.json', result)
    with ExitStack() as stack:
        for name in LOCKS:
            handle = stack.enter_context((Path(args.locks_dir)/name).open('a+'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        from tokenizers import Tokenizer
        tokenizer = Path(args.tokenizer)
        result['tokenizer_sha256'] = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
        prompt = prompt_factory(Tokenizer.from_file(str(tokenizer)))
        for count in args.concurrency:
            if time.monotonic() >= deadline-90:
                result.update(status='failed', error='Insufficient bounded time to start the next cohort'); break
            cohort = run_cohort(args, output/f'concurrency-{count}', count, prompt, hard_deadline=deadline)
            result['cohorts'].append({key:cohort.get(key) for key in
                ('status', 'concurrency', 'elapsed', 'completed_output_tokens', 'observed_full_batches', 'error', 'final_audit_error')})
            result['status'] = 'running'; write_json(output/'results.json', result)
            if cohort['status'] != 'passed':
                result['status'] = 'failed'; break
        else:
            result['status'] = 'passed'
    result.update(ended=time.time(), elapsed=time.monotonic()-started)
    write_json(output/'results.json', result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--candidate-dir'); parser.add_argument('--execute', action='store_true')
    parser.add_argument('--output'); parser.add_argument('--concurrency', type=int, nargs='+', default=[25, 30])
    parser.add_argument('--base-url', default='http://127.0.0.1:1234')
    parser.add_argument('--expected-hostname'); parser.add_argument('--expected-boot-id')
    parser.add_argument('--candidate-identity'); parser.add_argument('--server-identity', default='/workspace/arc3-inference/server-identity.json')
    parser.add_argument('--engine-events'); parser.add_argument('--service-events')
    parser.add_argument('--locks-dir', default='/workspace/arc3-inference/runs')
    parser.add_argument('--tokenizer', default='/workspace/models/Qwen3.8-Flash-Next-NVFP4/tokenizer.json')
    parser.add_argument('--max-seconds', type=float, default=2400)
    parser.add_argument('--cohort-seconds', type=float, default=1200)
    parser.add_argument('--timeout', type=float, default=300)
    args = parser.parse_args(argv)
    if args.concurrency not in ([25], [30], [25, 30]):
        parser.error('Supported cohorts are 25, 30, or 25 followed by 30')
    if not args.execute:
        print(json.dumps(plan_description(args.concurrency), indent=2)); return 0
    for key in ('output', 'expected_hostname', 'expected_boot_id', 'candidate_identity', 'engine_events', 'service_events'):
        if getattr(args, key) is None:
            parser.error('Missing explicit ' + key)
    if not (120 <= args.max_seconds <= 2400 and 90 <= args.cohort_seconds <= 1200 and 0 < args.timeout <= 300):
        parser.error('Require total<=2400s, cohort<=1200s and per-call timeout<=300s')
    result = run(args)
    print(json.dumps({key:result.get(key) for key in ('status', 'elapsed', 'cohorts')}, indent=2))
    return 0 if result['status'] == 'passed' else 1


if __name__ == '__main__':
    raise SystemExit(main())
