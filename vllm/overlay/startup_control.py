"""Bounded startup plus read-only memory observation of one pinned candidate."""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time
import urllib.request


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--candidate-dir', type=Path, required=True)
    parser.add_argument('--execute', action='store_true')
    args = parser.parse_args()
    if not args.execute:
        print('No execution without --execute'); return
    root = Path('/workspace/arc3-inference')
    here = args.candidate_dir.resolve()
    assert here.parent == root/'runs'
    assert socket.gethostname() == '3832f8f789f9'
    boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert boot == '1bd8be96-58e2-4485-8fd1-c0e99dc12e74'
    for name, sha in json.loads((here/'overlay-manifest.json').read_text()).items():
        assert hashlib.sha256((here/name).read_bytes()).hexdigest() == sha, name
    state_path = here/'startup-state.json'
    assert not state_path.exists()
    locks = ExitStack()
    try:
        for name in ('public-eval-campaign.lock','mtp-precision-sweep.lock',
                     'bf16-gameplay-ablation.lock','mtp0-memory-diagnostic.lock',
                     'persistent-eviction-scaling.lock'):
            handle = locks.enter_context((root/'runs'/name).open('a+'))
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BaseException:
        locks.close(); raise
    fields = Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')', 1)[1].split()
    state = dict(stage='starting_empty_pod', complete=False, started_epoch=time.time(),
        campaign_deadline_epoch=time.time()+1800, hostname=socket.gethostname(),
        boot_id=boot, controller=dict(pid=os.getpid(), state=fields[0],
            pgid=int(fields[2]), sid=int(fields[3]), start_ticks=int(fields[19])))
    def save():
        temp = state_path.with_suffix('.tmp')
        temp.write_text(json.dumps(state, indent=2)+'\n'); temp.replace(state_path)
    def fetch(route, body=None):
        request = urllib.request.Request('http://127.0.0.1:1234'+route,
            data=None if body is None else json.dumps(body).encode(),
            headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request, timeout=20) as response:
            return json.load(response)
    save()
    with (here/'gpu-watch.log').open('x') as log:
        observer = subprocess.Popen(['python3', str(here/'monitor_candidate_gpu_memory.py'),
            '--state', str(state_path), '--output', str(here/'gpu-watch'),
            '--max-seconds', '1920'], stdout=log, stderr=subprocess.STDOUT)
    state['observer_pid'] = observer.pid; save()
    try:
        env = dict(os.environ, ARC3_ROOT=str(root))
        for action in ('start',):
            state['stage'] = 'stopping_previous' if action=='stop-baseline' else 'startup'
            save()
            with (here/f'{action}.log').open('x') as log:
                subprocess.run(['bash', str(root/'runtime_python.sh'),
                    str(here/'launch_scaling.py'), action], env=env, stdout=log,
                    stderr=subprocess.STDOUT, check=True, timeout=240)
        identity = json.loads((here/'candidate-server.json').read_text())
        state['server'] = identity['root']; save()
        last = 0
        while time.time() < state['campaign_deadline_epoch']:
            pid = identity['root']['pid']
            actual = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
            assert actual[0] != 'Z' and int(actual[19]) == identity['root']['start_ticks']
            try:
                health = fetch('/arc3/persistent/health')
                ready = health.get('state') == 'ready'
            except (OSError, ValueError):
                ready = False
            if ready:
                assert health['max_context'] == 40000 and health['max_sessions'] == 30
                assert health['prefix_caching'] is False
                assert not any(s['state'] not in ('closed','error') for s in health['sessions'].values())
                (here/'health-ready.json').write_text(json.dumps(health, indent=2)+'\n')
                for name, method in [('graph-ready','arc3_eviction_graph_status'),
                                     ('memory-ready','arc3_memory_snapshot')]:
                    result = fetch('/collective_rpc', {'method':method})
                    (here/f'{name}.json').write_text(json.dumps(result, indent=2)+'\n')
                state['stage'] = 'ready'; state['complete'] = True; break
            if time.time()-last >= 30:
                print(json.dumps({'stage':'startup', 'elapsed':round(time.time()-state['started_epoch'])}), flush=True)
                last = time.time()
            time.sleep(3)
        else:
            raise TimeoutError('Candidate startup exceeded 30-minute bound')
    except BaseException as exc:
        state.update(stage='failed', error=repr(exc)); raise
    finally:
        state['ended_epoch'] = time.time(); save()
        try:
            try:
                observer.wait(timeout=10)
            except subprocess.TimeoutExpired:
                observer.terminate(); observer.wait(timeout=10)
            summary=json.loads((here/'gpu-watch/summary.json').read_text())
            state['observer_returncode']=observer.returncode
            if (observer.returncode or not summary.get('complete') or summary.get('error')
                    or not summary.get('overall',{}).get('samples')
                    or not summary.get('stages',{}).get('startup',{}).get('samples')):
                state.update(stage='failed',memory_observer_error='Missing or failed startup memory observation')
        except Exception as exc:
            state.update(stage='failed',memory_observer_error=repr(exc))
        finally:
            locks.close();save()
        print(json.dumps(state), flush=True)
    if state['stage']!='ready':raise SystemExit(1)


if __name__ == '__main__':
    main()
