"""Replay and profile one fixed FP8 E4M3 baseline on the owned idle server.

Requires the current reader candidate to have completed and its controller to
have exited. Restarts with the baseline reader and profiling configured, checks
the preserved FP8 MoE plans, then captures an eight-way MTP3 decode window.
The replay remains the unprofiled throughput measurement. This controller does
not promote kernels or change context/MTP defaults, and leaves the server alive
and idle on success. Each helper has a hard process-group wall-time bound.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import time
from urllib.request import Request, urlopen

from minimal_inference import HOST, PORT, ROOT, IDENTITY, matches, owned_snapshot, process, save
from run_quantized_serving_candidate import verify_storage

BASE = f'http://{HOST}:{PORT}'
BOOT_ID = Path('/proc/sys/kernel/random/boot_id')
SERVING_TIMEOUT = 4800
PROFILE_TIMEOUT = 420
LIKELIHOOD_TIMEOUT = 1200


def same_server(expected):
    """Reject a replaced server before issuing any diagnostic/recovery request."""
    current = json.loads(IDENTITY.read_text())
    assert current['boot_id'] == expected['boot_id'] == BOOT_ID.read_text().strip()
    assert current['root'] == expected['root'] and matches(expected['root'])
    assert current['argv'] == expected['argv']
    assert current['log_dir'] == expected['log_dir']
    return current


def request_metrics(expected, path):
    same_server(expected)
    with urlopen(BASE+'/metrics', timeout=10) as response:
        raw = response.read().decode()
    path.write_text(raw)
    counts = {}
    for name in ['num_requests_running', 'num_requests_waiting']:
        rows = [float(line.split()[-1]) for line in raw.splitlines()
                if line.startswith('vllm:'+name+'{')]
        assert len(rows) == 1, f'Expected one TP1 metric for {name}: {rows}'
        counts[name] = rows[0]
    return counts


def wait_idle(expected, output, label, timeout=45):
    deadline = time.monotonic()+timeout
    while True:
        counts = request_metrics(expected, output/(label+'-metrics.txt'))
        if not any(counts.values()):
            return counts
        if time.monotonic() >= deadline:
            raise TimeoutError('Owned server did not drain: '+str(counts))
        time.sleep(.25)


def kill_helper_group(child, saved):
    """Signal only the isolated helper session; the vLLM server has its own."""
    assert saved['pid'] == saved['pgid'] == saved['sid'] == child.pid
    snapshots = []
    for sig, grace in [(signal.SIGTERM, 5), (signal.SIGKILL, 5)]:
        rows = [process(int(path.name)) for path in Path('/proc').iterdir() if path.name.isdigit()]
        group = [row for row in rows if row and row['state'] != 'Z' and
                 row['pgid'] == saved['pgid'] and row['sid'] == saved['sid']]
        assert all(row['start_ticks'] >= saved['start_ticks'] for row in group)
        if not group:
            break
        # Never target a server session even if an unexpected launcher changes
        # how its children are created. Preserve the failure for manual review.
        if IDENTITY.exists():
            live = json.loads(IDENTITY.read_text())
            assert live['root']['sid'] != saved['sid'], 'Helper unexpectedly owns the server session'
        snapshots.append(dict(signal=int(sig), processes=group))
        try:
            os.killpg(saved['pgid'], sig)
        except ProcessLookupError:
            pass
        deadline = time.monotonic()+grace
        while time.monotonic() < deadline and any(matches(row) for row in group):
            time.sleep(.1)
    child.wait(timeout=5)
    assert not any(matches(row) for row in group), 'Helper descendants survived the timeout'
    return snapshots


def verify_tactics(server_log, expected_raw, output):
    text = server_log.read_text()
    paths = re.findall(r'Using FlashInfer autotune cache file: (\S+)', text)
    assert len(set(paths)) == 1, paths
    path = Path(paths[0])
    assert f'configs from {path}' in text, 'Saved MoE plans were not loaded at startup'
    raw = path.read_bytes()
    assert json.loads(raw) == json.loads(expected_raw), 'MoE tactics differ from the preserved FP8 reference'
    (output/'moe-tactics-used.json').write_bytes(raw)
    result = dict(passed=True, path=str(path), sha256=hashlib.sha256(raw).hexdigest(),
                  scope='Loaded cache and saved post-startup plans match the preserved FP8 reference')
    save(output/'moe-tactics-verification.json', result)
    return result


def validate_profile_server(identity, output):
    argv = identity['argv']
    for flag, expected in [('--max-model-len', '32768'), ('--max-num-seqs', '8'),
                           ('--max-num-batched-tokens', '8192'),
                           ('--max-cudagraph-capture-size', '32'),
                           ('--kv-cache-memory-bytes', str(8*2**30))]:
        assert argv[argv.index(flag)+1] == expected, flag
    assert '--enforce-eager' not in argv and '--no-enable-prefix-caching' in argv
    speculative = json.loads(argv[argv.index('--speculative-config')+1])
    assert speculative == {'method':'mtp', 'num_speculative_tokens':3}
    for key, value in dict(ARC3_QSA_KV_FORMAT='fp8_e4m3', ARC3_QSA_FP8_READER='baseline',
                           ARC3_QSA_MODE='radix', ARC3_TOPK_THREADS='512', ARC3_TOPK_CACHE_ROW='1',
                           ARC3_MOE_FINALIZE='unfused', ARC3_MTP_MIN_P='1').items():
        assert identity['environment'].get(key) == value, key
    profiler = json.loads(argv[argv.index('--profiler-config')+1])
    assert profiler['profiler'] == 'torch' and profiler['ignore_frontend']
    assert profiler['max_iterations'] == 8 and profiler['delay_iterations'] == 4
    assert not profiler['torch_profiler_with_stack'] and not profiler['torch_profiler_with_memory']
    assert Path(profiler['torch_profiler_dir']).resolve() == (output/'serving/profile').resolve()


def inspect_shared_experts(expected, output):
    same_server(expected)
    payload = json.dumps(dict(method='inspect_shared_experts', timeout=30)).encode()
    with urlopen(Request(BASE+'/collective_rpc', data=payload,
                         headers={'Content-Type':'application/json'}), timeout=45) as response:
        raw = response.read()
    (output/'shared-experts-response.json').write_bytes(raw)
    report = json.loads(raw)
    assert len(report['results']) == 1, 'Expected one TP1 worker response'
    models = report['results'][0]['models']
    assert len(models) == 2 and {row['role'] for row in models} == {'main','draft'}
    assert all(row['present'] and row['moe_layers'] for row in models), 'Missing main or draft MoE inventory'
    return {row['role']:len(row['moe_layers']) for row in models}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--previous-stage', type=Path, required=True)
    parser.add_argument('--tactic-reference', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, default=ROOT/'runs/quant-serving-e4m3-v1')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    for name in ['previous_stage','tactic_reference','baseline','output']:
        setattr(args, name, getattr(args, name).resolve())
    args.output.mkdir(parents=True, exist_ok=False)
    state = dict(stage='preflight', complete=False, controller=process(os.getpid()),
                 started_epoch=time.time(), runs=[],
                 previous_stage=str(args.previous_stage), baseline=str(args.baseline),
                 serving_timeout_seconds=SERVING_TIMEOUT, profile_timeout_seconds=PROFILE_TIMEOUT,
                 likelihood_timeout_seconds=LIKELIHOOD_TIMEOUT)
    expected_server = None
    profile_attempted = False

    def update(**fields):
        state.update(**fields)
        save(args.output/'state.json', state)
        print(json.dumps(state), flush=True)

    def command(label, argv, timeout):
        update(stage=label)
        entry = dict(label=label, argv=argv, timeout_seconds=timeout, started_epoch=time.time())
        state['runs'].append(entry)
        with (args.output/(label+'.log')).open('w') as log:
            child = subprocess.Popen(argv, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True)
            helper = process(child.pid)
            assert helper and helper['pgid'] == helper['sid'] == child.pid
            entry['process'] = helper
            update()
            try:
                entry['returncode'] = child.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                entry['timed_out'] = True
                entry['timeout_cleanup'] = kill_helper_group(child, helper)
                entry['returncode'] = child.returncode
                raise TimeoutError(label+' exceeded its hard wall-time bound')
            finally:
                entry['finished_epoch'] = time.time()
                update()
        if entry['returncode']:
            raise RuntimeError(label+' failed; inspect preserved evidence')

    try:
        update()
        source_hashes = {}
        for name in [Path(__file__).name, 'profile_decode_cost.py', 'run_config_candidate.py',
                     'benchmark_serving_config.py', 'minimal_inference.py', 'gameplay_config.py',
                     'run_quantized_serving_candidate.py', 'probe_multimodal_likelihood.py',
                     'worker_diagnostics.py']:
            raw = (ROOT/name).read_bytes()
            (args.output/name).write_bytes(raw)
            source_hashes[name] = hashlib.sha256(raw).hexdigest()
        save(args.output/'source-sha256.json', source_hashes)
        previous = json.loads((args.previous_stage/'state.json').read_text())
        assert previous['stage'] == 'complete' and not matches(previous['controller'])
        assert json.loads((args.previous_stage/'public/validation.json').read_text())['valid']
        assert json.loads((args.previous_stage/'moe-tactics-verification.json').read_text())['passed']
        baseline = json.loads((args.baseline/'state.json').read_text())
        assert baseline['stage'] == 'complete' and not matches(baseline['controller'])
        reference = args.baseline/'serving/benchmark/correctness.json'
        assert json.loads(reference.read_text())
        likelihood = args.baseline/'multimodal-likelihood-v1'
        assert json.loads((likelihood/'result.json').read_text())['complete']
        for name in ['panel.json','result.json']:
            (args.output/('multimodal-reference-'+name)).write_bytes((likelihood/name).read_bytes())
        expected_raw = args.tactic_reference.read_bytes()
        assert json.loads(expected_raw) == json.loads((args.previous_stage/'moe-tactics-used.json').read_bytes())
        (args.output/'moe-tactics-reference.json').write_bytes(expected_raw)
        save(args.output/'prerequisite-state.json', previous)
        (args.output/'reference-correctness.json').write_bytes(reference.read_bytes())
        previous_server = json.loads((args.previous_stage/'serving/server-identity.json').read_text())
        same_server(previous_server)
        wait_idle(previous_server, args.output, 'preflight')
        save(args.output/'previous-server.json', previous_server)
        command('serving', ['python3','-B','run_config_candidate.py',
            '--batch','8192','--mtp','3','--sequences','8','--streams','8','--slots','8',
            '--context','32768','--kv-gib','8','--kv-format','fp8_e4m3','--fp8-reader','baseline',
            '--selector','radix','--topk-threads','512','--topk-cache-row','1','--graph-size','32',
            '--profile','--reference',str(reference),'--output',str(args.output/'serving')], SERVING_TIMEOUT)
        assert json.loads((args.output/'serving/stage.json').read_text())['stage'] == 'complete'
        expected_server = json.loads((args.output/'serving/server-identity.json').read_text())
        assert expected_server['root'] != previous_server['root'], 'Expected a fresh server'
        same_server(expected_server)
        validate_profile_server(expected_server, args.output)
        save(args.output/'server-identity.json', expected_server)
        update(tactics=verify_tactics(args.output/'serving/server.log', expected_raw, args.output))
        save(args.output/'storage-verification.json', verify_storage(
            args.output/'serving/benchmark/forced-c8-r0/memory-after.json', 'fp8_e4m3'))
        wait_idle(expected_server, args.output, 'before-likelihood')
        command('multimodal_likelihood', ['venv-duck/bin/python','-B','probe_multimodal_likelihood.py',
            '--panel',str(likelihood/'panel.json'),'--reference',str(likelihood/'result.json'),
            '--output',str(args.output/'multimodal-likelihood-v1')], LIKELIHOOD_TIMEOUT)
        assert json.loads((args.output/'multimodal-likelihood-v1/result.json').read_text())['complete']
        wait_idle(expected_server, args.output, 'before-profile')
        update(stage='inspect_shared_experts', shared_expert_inventory=inspect_shared_experts(expected_server, args.output))
        wait_idle(expected_server, args.output, 'after-shared-inspection')
        profile_attempted = True
        command('decode_profile', ['venv-duck/bin/python','-B','profile_decode_cost.py',
                                  '--output',str(args.output/'decode-profile')], PROFILE_TIMEOUT)
        assert json.loads((args.output/'decode-profile/state.json').read_text())['complete']
        same_server(expected_server)
        idle_counts = wait_idle(expected_server, args.output, 'complete')
        final = dict(expected_server, processes=owned_snapshot(expected_server))
        save(args.output/'server-identity-after.json', final)
        update(stage='complete', complete=True, finished_epoch=time.time(), server_alive=True,
               idle_metrics=idle_counts,
               scope='Fresh fixed-config unprofiled replay plus diagnostic decode attribution; no kernel promotion')
    except Exception as exc:
        error = f'{type(exc).__name__}: {exc}'
        if profile_attempted and expected_server is not None:
            # The helper normally stops profiling itself. Repeat on failure or
            # timeout, but only if this exact owned server still exists.
            try:
                same_server(expected_server)
                with urlopen(Request(BASE+'/stop_profile', data=b''), timeout=30) as response:
                    recovery = dict(status=response.status, body=response.read().decode())
                save(args.output/'profile-recovery.json', recovery)
                save(args.output/'recovery-idle.json', wait_idle(expected_server, args.output, 'recovery'))
            except Exception as recovery_exc:
                save(args.output/'profile-recovery-error.json', dict(error=f'{type(recovery_exc).__name__}: {recovery_exc}'))
        update(stage='failed', error=error, finished_epoch=time.time())
        raise
    finally:
        if expected_server is not None:
            for name in ['server.log','launch.json']:
                path = Path(expected_server['log_dir'])/name
                if path.exists():
                    (args.output/name).write_bytes(path.read_bytes())


if __name__ == '__main__':
    main()
