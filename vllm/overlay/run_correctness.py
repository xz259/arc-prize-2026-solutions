"""Run three bounded correctness controls, stopping at the first failure.

Does not start, stop or modify the serving process. Gameplay is a separate gate.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--candidate', type=Path, required=True)
    args = parser.parse_args()
    if not args.execute:
        print('Plan: wait for the selected idle candidate, then matched/no-deletion, raw eviction, image/tool eviction controls.')
        return
    root = Path('/workspace/arc3-inference')
    here = args.candidate.resolve()
    assert here.is_relative_to(root/'runs')
    assert socket.gethostname() == '9b2fd24de2c6'
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert boot == '1bd8be96-58e2-4485-8fd1-c0e99dc12e74'
    manifest = json.loads((here/'overlay-manifest.json').read_text())
    for name, expected in manifest.items():
        assert hashlib.sha256((here/name).read_bytes()).hexdigest() == expected, name
    identity = json.loads((root/'server-identity.json').read_text())
    assert identity['persistent_experiment']['directory'] == str(here)
    config=json.loads((here/'optimization-config.json').read_text())
    assert ('--enforce-eager' in identity['argv']) == bool(config['eager'])
    output = here/'live-controls-v1'
    output.mkdir(exist_ok=False)
    summary = dict(started=time.time(), state='waiting_for_health', identity=identity, steps=[])
    def save():
        (output/'controller.json').write_text(json.dumps(summary, indent=2)+'\n')
    save()
    try:
        deadline = time.monotonic()+1500
        last_report = 0
        while True:
            proc = Path('/proc')/str(identity['root']['pid'])
            assert proc.exists(), 'Candidate exited during startup'
            cmdline = (proc/'cmdline').read_bytes()
            assert hashlib.sha256(cmdline).hexdigest() == identity['root']['cmdline_sha256'], 'Candidate identity changed/exited'
            try:
                with urlopen('http://127.0.0.1:1234/arc3/persistent/health', timeout=5) as response:
                    health = json.load(response)
            except OSError:
                health = None
            if health is not None:
                assert health['mode'] == 'persistent-history-eviction-mtp0'
                assert health['prefix_caching'] is False
                assert all(s['state'] in ('closed', 'error') for s in health['sessions'].values())
                break
            if time.monotonic() >= deadline:
                raise TimeoutError('Candidate startup exceeded bounded wait')
            if time.monotonic()-last_report >= 30:
                print(json.dumps(dict(state='waiting_for_health', elapsed=time.time()-summary['started'])), flush=True)
                last_report = time.monotonic()
            time.sleep(3)
        summary['health_before'] = health
        with urlopen(Request('http://127.0.0.1:1234/collective_rpc',
                             data=json.dumps({'method': 'arc3_eviction_layout'}).encode(),
                             headers={'Content-Type': 'application/json'}), timeout=30) as response:
            layouts = json.load(response)
        (output/'bound-cache-layout.json').write_text(json.dumps(layouts, indent=2)+'\n')
        assert layouts.get('results') and all(
            row.get('read_only') is True and row.get('tensor_values_read') is False
            for row in layouts['results']), 'Missing read-only worker-layout receipt'
        common = ['bash', str(root/'runtime_python.sh')]
        commands = [
            ('matched', [str(here/'raw_matched_checks.py'), '--output', str(output/'matched')], 360),
            ('raw_eviction', [str(here/'raw_managed_eviction_checks.py'), '--execute',
                '--output', str(output/'raw'), '--expected-hostname', socket.gethostname(),
                '--expected-boot-id', boot, '--engine-events', str(here/'engine-events.jsonl'),
                '--service-events', str(here/'service-events.jsonl'),
                '--max-seconds', '300', '--timeout', '180'], 360),
            ('image_eviction', [str(here/'managed_chat_gpu_checks.py'), '--execute',
                '--output', str(output/'images'), '--service-audit', str(here/'service-events.jsonl'),
                '--timeout', '180'], 600),
        ]
        commands += [
            ('history_eviction', [str(here/'raw_history_eviction_checks.py'), '--execute',
                '--output', str(output/'history'), '--expected-hostname', socket.gethostname(),
                '--expected-boot-id', boot, '--engine-events', str(here/'engine-events.jsonl'),
                '--service-events', str(here/'service-events.jsonl'), '--max-seconds', '300'], 360),
            ('long_positions', [str(here/'raw_history_long_position_checks.py'), '--execute',
                '--output', str(output/'long'), '--expected-hostname', socket.gethostname(),
                '--expected-boot-id', boot, '--engine-events', str(here/'engine-events.jsonl'),
                '--service-events', str(here/'service-events.jsonl'), '--max-seconds', '600'], 660),
        ]
        if config.get('graph_trace'):
            commands.append(('concurrent', [str(here/'raw_concurrent_eviction_checks.py'),
                '--execute', '--output', str(output/'concurrent'),
                '--expected-hostname', socket.gethostname(), '--expected-boot-id', boot,
                '--engine-events', str(here/'engine-events.jsonl'),
                '--service-events', str(here/'service-events.jsonl'),
                '--expected-mode', 'eager' if config['eager'] else 'graph',
                '--candidate-identity', str(here/'candidate-server.json'),
                '--max-seconds', '300'], 360))
        for name, command, timeout in commands:
            summary['state'] = name
            save()
            print(json.dumps(dict(state=name)), flush=True)
            step = dict(name=name, started=time.time(), argv=common+command)
            with (output/(name+'.log')).open('w') as log:
                result = subprocess.run(common+command, cwd=root, stdout=log, stderr=subprocess.STDOUT,
                                        timeout=timeout, env=dict(os.environ, VLLM_PLUGINS=''))
            step.update(ended=time.time(), returncode=result.returncode)
            summary['steps'].append(step)
            save()
            print(json.dumps(step), flush=True)
            if result.returncode:
                raise RuntimeError(name+' failed; inspect its preserved log and receipts')
            if name == 'matched':
                matched = json.loads((output/'matched/results.json').read_text())
                assert matched['all_matched_interleaved_outputs_exact'] and matched['all_repeat_outputs_exact'], 'Matched no-deletion control differed'
        summary['state'] = 'passed'
    except BaseException as error:
        summary.update(state='failed', error=repr(error))
        raise
    finally:
        summary['ended'] = time.time()
        # Timed-out children cannot execute their finally blocks. Close only
        # this controller's recorded session prefixes, never another owner.
        cleanup = []
        for relative in ('matched/results.json', 'raw/results.json', 'images/result.json', 'history/results.json', 'long/results.json'):
            path = output/relative
            if not path.exists():
                continue
            result = json.loads(path.read_text())
            prefixes = [result.get('session_prefix'), result.get('session_id'), result.get('probe_session_id')]
            # Requests are persisted before transmission, so even a timeout
            # during probe admission leaves an exact ownership record.
            owned_ids = set()
            for request_path in path.parent.rglob('*-request.json'):
                request_record = json.loads(request_path.read_text())
                payload = request_record.get('payload') or {}
                if request_record.get('route') == '/arc3/persistent/start' and payload.get('session_id'):
                    owned_ids.add(payload['session_id'])
            try:
                with urlopen('http://127.0.0.1:1234/arc3/persistent/health', timeout=10) as response:
                    final = json.load(response)
                for sid, state in final['sessions'].items():
                    owned = sid in owned_ids or any(prefix and sid.startswith(prefix) for prefix in prefixes)
                    if owned and state['state'] not in ('closed', 'error'):
                        with urlopen(Request('http://127.0.0.1:1234/arc3/persistent/'+sid+'/close',
                                             data=b'{}', headers={'Content-Type':'application/json'}), timeout=60) as response:
                            cleanup.append(json.load(response))
            except Exception as error:
                cleanup.append(dict(error=repr(error)))
        summary['controller_cleanup'] = cleanup
        save()


if __name__ == '__main__':
    main()
