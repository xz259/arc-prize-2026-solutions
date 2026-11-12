"""Opt-in bounded original-position diagnostic: 48,128 logical / <=7,000 resident.

Reuses the signed-off raw history client, independent Oracle, and byte/work audit.
No server launch, shutdown, game execution, or fresh-shortened-prefill parity claim.
Without --execute, print the plan only. All runtime work stays on a verified idle
loopback server under the existing campaign locks; cleanup closes only our session.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
from pathlib import Path
import socket
import time
import traceback
import uuid

from raw_history_eviction_checks import (
    HistoryClient, MODE, SOURCE_FILES as HISTORY_SOURCE_FILES, audit_events,
    engine_for, intervals,
)
from raw_managed_eviction_checks import LOCKS
from raw_gpu_checks import ContractFailure, prompt_factory, token_digest, write_json

CONTEXT = 40000
BUDGET = 7000
OBSERVATIONS = 16
INPUT_PER_OBSERVATION = 3000
OUTPUT_PER_OBSERVATION = 8
FIXED_PREFIX = 128
LOGICAL_TARGET = OBSERVATIONS * (INPUT_PER_OBSERVATION + OUTPUT_PER_OBSERVATION)
LOGICAL_CEILING = 50000
SOURCE_FILES = tuple(dict.fromkeys((Path(__file__).name, *HISTORY_SOURCE_FILES,
    'engine_patch.py', 'eviction_validation.py', 'eviction_qsa.py', '_eviction_qsa_gpu.py')))


def plan_description():
    return dict(mode=MODE, max_seconds=600, sessions=1,
        server_context=CONTEXT, resident_budget=BUDGET,
        observations=OBSERVATIONS, input_tokens_per_observation=INPUT_PER_OBSERVATION,
        output_tokens_per_observation=OUTPUT_PER_OBSERVATION,
        fixed_prefix_tokens=FIXED_PREFIX, logical_target=LOGICAL_TARGET,
        logical_ceiling_exclusive=LOGICAL_CEILING, expected_history_evictions=14,
        expected_final_resident=6144, maximum_emitted_tokens=128,
        proof=['Exact full logical stream exceeds the real 40K server context',
               'Resident input and output stay within the diagnostic 7K pool',
               'Original position journal and retained subsequence remain exact',
               'Only pending-final plus new input receives prefill; no history replay',
               'Repeated synchronized cache copies and physical page release',
               'Fixed prefix and current observation survive; exact retry adds no work'],
        exclusions=['direct GPU register position trace', 'gameplay quality', 'throughput',
                    'fresh truncated-prefill equality', 'independent auxiliary-state byte hashes'])


def exercise(client, prompt, *, on_progress=None):
    """Run the fixed panel with exact bounded inputs; injectable transport for CPU tests."""
    if client.oracle.budget != BUDGET or client.turns:
        raise ContractFailure('Long-position panel needs a fresh 7K session')
    retry = None
    for i in range(OBSERVATIONS):
        # prompt_factory guarantees >=25K source IDs; these slices end below23K.
        delta = prompt(INPUT_PER_OBSERVATION, (i * 1103) % 20000)
        if len(delta) != INPUT_PER_OBSERVATION:
            raise ContractFailure('Tokenizer source did not supply the exact planned delta')
        planned = len(client.oracle.logical) + len(delta) + OUTPUT_PER_OBSERVATION
        if planned >= LOGICAL_CEILING:
            raise ContractFailure('Long-position diagnostic would exceed its logical bound')
        client.submit(f'observation_{i+1:02d}', delta,
                      observation=(f'long-obs-{i+1:02d}', FIXED_PREFIX if i == 0 else 0))
        if len(client.oracle.resident_positions) > BUDGET:
            raise ContractFailure('Resident output exceeded diagnostic budget')
        if i == 13:  # first completed response beyond40K; exercise exact retry there.
            retry = client.replay_last()
        if on_progress is not None:
            on_progress(client.turns)
    if len(client.oracle.logical) != LOGICAL_TARGET:
        raise ContractFailure('Final logical stream differs from the bounded plan')
    return dict(idempotency=retry, logical_tokens=len(client.oracle.logical),
                resident_tokens=len(client.oracle.resident_positions))


def audit_long_positions(engine_path, service_path, sid, turns):
    """Extend the existing physical/cache audit with independent original-index checks."""
    base = audit_events(engine_path, service_path, sid, turns)
    checks = {}; errors = []
    def check(ok, key, detail=None):
        checks[key] = checks.get(key, 0) + 1
        if not ok:
            errors.append(dict(check=key, detail=detail))

    engine = engine_for(engine_path, sid)
    service = [r for line in Path(service_path).read_text().splitlines() if line
               and (r := json.loads(line)).get('session_id') == sid]
    logical = []; removed = set(); prefix = None; crossed = []
    scheduled_original_max = -1
    check(len(turns) == OBSERVATIONS, 'complete_sixteen_observation_panel')
    for t in turns:
        op = t['request_id']; n = t['turn']
        submits = [r for r in service if r['event'] == 'submit' and r.get('operation_id') == op]
        pauses = [r for r in service if r['event'] == 'pause' and r.get('operation_id') == op]
        eps = [r for r in engine if r['event'] == 'pause' and r.get('turn') == n]
        if len(submits) != 1 or len(pauses) != 1 or len(eps) != 1:
            check(False, 'complete_position_journal', n)
            continue
        sub, pause, ep = submits[0], pauses[0], eps[0]
        delta = sub.get('appended_token_ids', [])
        output = pause.get('output_token_ids', [])
        check(len(delta) == INPUT_PER_OBSERVATION and len(output) == OUTPUT_PER_OBSERVATION,
              'exact_bounded_delta_and_output', n)
        check(len(logical) == t['logical_before'] and token_digest(logical) == t['logical_before_sha256'],
              'historical_logical_stream_unchanged', n)
        for start, end in t['history_eviction']['combined_ranges']:
            check(0 <= start < end <= len(logical), 'eviction_uses_existing_original_positions', n)
            points = set(range(start, end))
            check(not (points & removed), 'previous_holes_not_removed_twice', n)
            removed.update(points)
        logical.extend(delta); logical.extend(output)
        if prefix is None:
            prefix = logical[:FIXED_PREFIX]
        positions = [p for p in range(len(logical)) if p not in removed]
        resident = [logical[p] for p in positions]
        state = ep.get('logical_eviction_state', {})
        full_groups = sum(all(p in removed for p in range(4*g, 4*g+4))
                          for g in {p//4 for p in removed})
        expected_state = dict(logical_tokens=len(logical), logical_sha256=token_digest(logical),
                              eviction_revision=t['eviction_revision'], deleted_tokens=len(removed),
                              deleted_groups=full_groups, group_padding=len(removed)-4*full_groups,
                              evicted_ranges=intervals(removed), poisoned=False)
        check(all(state.get(k) == v for k, v in expected_state.items()),
              'exact_original_position_ledger', n)
        check(ep.get('resident_tokens') == len(resident) and ep.get('tokens_sha256') == token_digest(resident),
              'original_indices_select_exact_resident_stream', n)
        check(len(resident) <= BUDGET and t['resident_input_tokens'] <= BUDGET,
              'resident_within_diagnostic_capacity', n)
        check(logical[:FIXED_PREFIX] == prefix and not any(p in removed for p in range(FIXED_PREFIX)),
              'fixed_prefix_survives_all_evictions', n)
        active = t['history_state']['blocks'][-1]
        check(active['end'] is None and not any(active['start'] <= p < len(logical) for p in removed),
              'current_observation_survives_all_evictions', n)
        check(len(positions) >= 2 and positions[-1] == len(logical)-1,
              'pending_final_token_keeps_original_frontier', n)
        computed = ep.get('computed_tokens')
        if isinstance(computed, int) and 0 < computed <= len(positions):
            original_max = positions[computed-1]
            check(original_max == len(logical)-2,
                  'computed_dense_frontier_maps_to_original_frontier', n)
            scheduled_original_max = max(scheduled_original_max, original_max)
        else:
            check(False, 'computed_dense_frontier_maps_to_original_frontier', n)
        if len(logical) > CONTEXT:
            crossed.append(dict(turn=n, logical_tokens=len(logical), resident_tokens=len(resident),
                                logical_input_tokens=len(logical)-len(output)))
        check(len(logical) < LOGICAL_CEILING, 'original_stream_below_diagnostic_ceiling', n)
    check(len(logical) == LOGICAL_TARGET, 'exact_48128_logical_tokens')
    check(len(crossed) == 3, 'three_completed_turns_beyond_40000')
    check(scheduled_original_max == LOGICAL_TARGET-2 and scheduled_original_max >= CONTEXT,
          'computed_original_positions_exceed_configured_context')
    check(sum(bool(t['removed_tokens']) for t in turns) == 14, 'fourteen_history_evictions')
    physical = sum(c.get('freed_pages', 0) > 0 for c in base['copies'])
    check(physical >= 2, 'multiple_physical_page_release_transactions')
    return dict(valid=base['valid'] and not errors, physical_audit=base,
                checks=checks, errors=errors, logical_tokens=len(logical),
                resident_tokens=len(logical)-len(removed), removed_tokens=len(removed),
                highest_computed_original_position=scheduled_original_max,
                completed_turns_beyond_context=crossed,
                physical_release_transactions=physical,
                expected_total_prefill_tokens=sum(t['expected_prefill_tokens'] for t in turns),
                expected_total_decode_tokens=7*len(turns),
                evidence_limits=plan_description()['exclusions'])


def run(args):
    if socket.gethostname() != args.expected_hostname:
        raise ContractFailure('Host identity mismatch')
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    if boot != args.expected_boot_id:
        raise ContractFailure('Boot identity mismatch')
    for path in (args.engine_events, args.service_events):
        if not Path(path).is_file():
            raise ContractFailure('Missing authoritative audit '+str(path))
    out = Path(args.output).resolve(); out.mkdir(parents=True, exist_ok=False)
    http = out/'http'; http.mkdir()
    here = Path(__file__).resolve().parent
    tag = 'raw-history-long-'+time.strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]
    client = HistoryClient(args.base_url, http, args.timeout, tag,
                           deadline=time.monotonic()+args.max_seconds)
    result = dict(status='starting', plan=plan_description(), started=time.time(),
                  hostname=socket.gethostname(), boot_id=boot, session_id=client.sid,
                  source_sha256={n:hashlib.sha256((here/n).read_bytes()).hexdigest() for n in SOURCE_FILES})
    target = out/'results.json'; write_json(target, result)
    with ExitStack() as stack:
        for name in LOCKS:
            f = stack.enter_context((Path(args.locks_dir)/name).open('a+'))
            fcntl.flock(f, fcntl.LOCK_EX|fcntl.LOCK_NB)
        try:
            health = client.call('GET', '/arc3/persistent/health')
            write_json(out/'health-before.json', health)
            if health.get('mode') != MODE or health.get('max_context') != CONTEXT or health.get('prefix_caching') is not False:
                raise ContractFailure('Wrong serving mode/40K context/prefix configuration')
            if any(s.get('state') not in ('closed', 'error') for s in health.get('sessions', {}).values()):
                raise ContractFailure('Another resident session owns server')
            from tokenizers import Tokenizer
            tokenizer = Path(args.tokenizer)
            result['tokenizer_sha256'] = hashlib.sha256(tokenizer.read_bytes()).hexdigest()
            prompt = prompt_factory(Tokenizer.from_file(str(tokenizer)))
            def progress(turns):
                result.update(status='running', turns=turns)
                write_json(target, result)
            result['panel'] = exercise(client, prompt, on_progress=progress)
            client.close(client.sid)
            result['audit'] = audit_long_positions(args.engine_events, args.service_events, client.sid, client.turns)
            if not result['audit']['valid']:
                raise ContractFailure('Authoritative long-position audit failed')
            result['status'] = 'passed'
        except BaseException as exc:
            result.update(status='failed', error=dict(type=type(exc).__name__, message=str(exc), traceback=traceback.format_exc()))
        finally:
            result['turns'] = client.turns
            result['cleanup_errors'] = client.cleanup()
            try:
                final = client.call('GET', '/arc3/persistent/health')
                write_json(out/'health-after.json', final)
                result['owned_final_state'] = final.get('sessions', {}).get(client.sid)
                if client.active or (client.turns and (result['owned_final_state'] or {}).get('state') != 'closed'):
                    result['status'] = 'failed'
            except BaseException as exc:
                result.update(status='failed', cleanup_health_error=repr(exc))
            for name, path in (('engine', args.engine_events), ('service', args.service_events)):
                try:
                    rows = engine_for(path, client.sid) if name == 'engine' else [r for line in Path(path).read_text().splitlines()
                        if line and (r := json.loads(line)).get('session_id') == client.sid]
                    (out/(name+'-owned.jsonl')).write_text(''.join(json.dumps(r, separators=(',', ':'))+'\n' for r in rows))
                except BaseException as exc:
                    result.setdefault('audit_copy_errors', []).append(dict(kind=name, error=repr(exc)))
                    result['status'] = 'failed'
            result['ended'] = time.time(); result['http_requests'] = client.index
            write_json(target, result)
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute', action='store_true'); p.add_argument('--output')
    p.add_argument('--base-url', default='http://127.0.0.1:1234')
    p.add_argument('--tokenizer', default='/workspace/models/Qwen3.8-Flash-Next-NVFP4/tokenizer.json')
    p.add_argument('--locks-dir', default='/workspace/arc3-inference/runs')
    p.add_argument('--expected-hostname'); p.add_argument('--expected-boot-id')
    p.add_argument('--engine-events'); p.add_argument('--service-events')
    p.add_argument('--expected-context', type=int, default=CONTEXT)
    p.add_argument('--max-seconds', type=float, default=600); p.add_argument('--timeout', type=float, default=90)
    args = p.parse_args(argv)
    if not args.execute:
        print(json.dumps(plan_description(), indent=2)); return 0
    if any(getattr(args, key) is None for key in ('output', 'expected_hostname', 'expected_boot_id', 'engine_events', 'service_events')):
        p.error('Execution needs fresh output, explicit host/boot identity, and both audit paths')
    if not 0 < args.max_seconds <= 600 or not 0 < args.timeout <= 180 or args.expected_context != CONTEXT:
        p.error('Require max-seconds<=600, timeout<=180, and the real server context40000')
    result = run(args)
    return 0 if result['status'] == 'passed' and not result['cleanup_errors'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
