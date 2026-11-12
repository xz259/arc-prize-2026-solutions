"""One bounded, source-pinned actual-gameplay profile with owned cleanup."""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import fcntl
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time
from urllib.request import Request, urlopen

from profile_observer import HTTPObserver

ROOT = Path('/workspace/arc3-inference')
HOST = '9b2fd24de2c6'
BOOT = '1bd8be96-58e2-4485-8fd1-c0e99dc12e74'
SMOKE_GAMES = 'ls20-9607627b,ft09-0d8bbf25,vc33-5430563c,tr87-cd924810'


def save(path, value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def fetch(route, body=None, timeout=15):
    req=Request('http://127.0.0.1:1234'+route,
        data=None if body is None else json.dumps(body).encode(),
        headers={'Content-Type':'application/json'})
    with urlopen(req,timeout=timeout) as response:return json.load(response)


def identity(pid):
    f=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    return dict(pid=pid,state=f[0],ppid=int(f[1]),pgid=int(f[2]),sid=int(f[3]),start_ticks=int(f[19]))


def same(saved):
    try:current=identity(saved['pid'])
    except (FileNotFoundError,ProcessLookupError):return False
    return current['state']!='Z' and all(current[k]==saved[k] for k in ('pid','pgid','sid','start_ticks'))


def owned_processes(saved):
    """Read identities of the verified process tree, without command arguments."""
    if not same(saved):
        raise RuntimeError('Cannot observe a changed process identity')
    table={}
    for path in Path('/proc').iterdir():
        if path.name.isdigit():
            try:row=identity(int(path.name))
            except (FileNotFoundError,ProcessLookupError):continue
            if row['state']!='Z':table[row['pid']]=row
    selected={saved['pid']}
    while True:
        expanded=selected | {pid for pid,row in table.items() if row['ppid'] in selected}
        if expanded==selected:break
        selected=expanded
    return [row for pid,row in table.items() if pid in selected or row['sid']==saved['sid']]


def attempted_sessions(output):
    """Only request IDs persisted by this harness before its HTTP start call."""
    found=set()
    for path in sorted((output/'raw-harness').glob('*.jsonl')):
        for line in path.read_text().splitlines():
            try:row=json.loads(line)
            except ValueError:continue  # A killed writer may leave a trailing partial line.
            payload=row.get('payload',{})
            if row.get('kind')=='persistent-request' and payload.get('route')=='/arc3/persistent/start':
                sid=payload.get('body',{}).get('session_id')
                if isinstance(sid,str) and sid and all(c.isalnum() or c in '_.-' for c in sid):
                    found.add(sid)
    return found


def validate_server_profile(profile, server_root):
    assert profile['installed'] and profile['pid']==server_root['pid']
    assert profile['attempted']==profile['written'] and profile['written']>0
    assert all(profile[k]==0 for k in ('dropped','metadata_errors','unmatched_output_events'))
    assert profile['loop_lag_tasks']==1
    assert profile['availability'] and all(profile['availability'].values())
    assert profile['active_output_turns']==0


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--candidate-dir',type=Path,required=True)
    p.add_argument('--contract',choices=('smoke','public25'),required=True)
    p.add_argument('--execute',action='store_true')
    args=p.parse_args()
    if not args.execute:return
    here=args.candidate_dir.resolve()
    assert here.parent==ROOT/'runs' and socket.gethostname()==HOST
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip()==BOOT
    for name,sha in json.loads((here/'overlay-manifest.json').read_text()).items():
        assert hashlib.sha256((here/name).read_bytes()).hexdigest()==sha,name
    server=json.loads((here/'candidate-server.json').read_text())
    current=json.loads((ROOT/'server-identity.json').read_text())
    assert current['root']==server['root'] and same(server['root'])
    assert server['persistent_experiment']['phase']=='gameplay-profile-v1'
    config=server['persistent_experiment']['optimization_config']
    assert config['gameplay_profile']==1 and config['eager']==0
    assert config['verify_bytes']==config['verify_allocator']==0
    label='gameplay-smoke' if args.contract=='smoke' else 'gameplay15'
    duration=150 if args.contract=='smoke' else 900
    output=here/label
    assert not output.exists()
    state_path=here/(label+'-controller.json');assert not state_path.exists()
    overlay=ROOT/'runs/prolong-v5.22-release-20260912-v1/overlay'
    release_manifest=overlay.parent/'source-sha256.json'
    locks=ExitStack()
    for name in ('public-eval-campaign.lock','mtp-precision-sweep.lock',
                 'bf16-gameplay-ablation.lock','mtp0-memory-diagnostic.lock',
                 'persistent-eviction-scaling.lock'):
        handle=locks.enter_context((ROOT/'runs'/name).open('a+'))
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
    health=fetch('/arc3/persistent/health')
    assert health['state']=='ready' and health['max_sessions']==30 and health['max_context']==40000
    assert not health['prefix_caching']
    assert all(s['state'] in ('closed','error') for s in health['sessions'].values())
    previous=set(health['sessions'])
    server_profile=fetch('/arc3/persistent/profile-status')
    validate_server_profile(server_profile,server['root'])
    save(here/(label+'-server-profile-before.json'),server_profile)
    argv=[str(ROOT/'venv-duck/bin/python'),'-B',str(here/'run_profiled_gameplay.py'),
        '--harness-root',str(overlay),'--harness-manifest',str(release_manifest),
        '--profile-contract',args.contract,'--persistent-base-url','http://127.0.0.1:1234',
        '--games',SMOKE_GAMES if args.contract=='smoke' else 'all',
        '--total-seconds',str(duration),'--replicas','1','--context','40000','--context-profile','nvfp4',
        '--expected-sequences','30','--expected-mtp','0','--expected-gpu-memory-utilization','0.95',
        '--active-games','0','--request-slots','0','--clock-mode','gameplay','--shutdown-margin','0',
        '--no-token-limit','--no-response-cap','--response-headroom-fraction','0.10',
        '--temperature','1','--min-p','0','--reasoning-effort','xhigh','--image-scale','4','--stdout-token-limit','1024',
        '--program-index','--snapshot','--kv-eviction-policy','oldest_interactions',
        '--kv-evict-fraction','0.10','--output',str(output),'--execute']
    state=dict(stage='preparing',complete=False,hostname=HOST,boot_id=BOOT,
        controller=identity(os.getpid()),server=server['root'],argv=argv,
        started_epoch=time.time(),campaign_deadline_epoch=time.time()+duration+360,
        previous_session_ids=sorted(previous),cleanup_errors=[],contract=args.contract)
    save(state_path,state)
    observer=None; metrics=None; child=None
    try:
        save(here/(label+'-graph-before.json'),fetch('/collective_rpc',
            {'method':'arc3_eviction_graph_status','timeout':20},timeout=25))
        with (here/(label+'-gpu-watch.log')).open('x') as log:
            observer=subprocess.Popen(['python3',str(here/'monitor_candidate_gpu_memory.py'),
                '--state',str(state_path),'--output',str(here/(label+'-gpu-watch')),
                '--max-seconds',str(duration+480)],stdout=log,stderr=subprocess.STDOUT)
        process_observation=dict(server,processes=owned_processes(server['root']))
        metrics=HTTPObserver(here/(label+'-metrics.jsonl'),process_observation,max_seconds=duration+480).start()
        deps=[here,overlay,ROOT/'duck/src/ARC3-Inference',ROOT/'duck/src/tufa-arc-agi-framework/src',ROOT]
        env=dict(os.environ,ARC3_ROOT=str(ROOT),PYTHONPATH=os.pathsep.join(map(str,deps)),
            MPLCONFIGDIR=str(here/'harness-mpl'),MPLBACKEND='Agg',HF_HUB_OFFLINE='1',
            TRANSFORMERS_OFFLINE='1',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1',VLLM_PLUGINS='')
        with (here/(label+'.log')).open('x') as log:
            child=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state.update(stage='running',client=identity(child.pid));save(state_path,state)
        metrics.track_processes([state['client']])
        while child.poll() is None:
            if time.time()>=state['campaign_deadline_epoch']:raise TimeoutError('Bounded gameplay controller expired')
            if not same(server['root']):raise RuntimeError('Owned server process disappeared')
            settings=output/'run_settings.json'
            if settings.exists():
                try:
                    run=json.loads(settings.read_text())
                    if run.get('gameplay_started_epoch'):
                        state.update(stage='gameplay',gameplay_started_epoch=run['gameplay_started_epoch'],
                                     gameplay_deadline_epoch=run['deadline_epoch'])
                except ValueError:pass  # Runner replaces settings during arming.
            state['updated_epoch']=time.time();save(state_path,state)
            time.sleep(2)
        state['returncode']=child.returncode
        assert child.returncode==0,'Gameplay client failed; preserve logs'
        for name in ('validation.json','persistent-validation.json','harness-profile-status.json'):
            result=json.loads((output/name).read_text())
            assert result['valid'],name
        result=json.loads((output/'profiled-gameplay-result.json').read_text())
        assert result['status']=='passed' and result['profile']['dropped']==0
        state['stage']='passed'
    except BaseException as exc:
        state.update(stage='failed',error=repr(exc))
    finally:
        if child is not None and child.poll() is None:
            try:
                if same(state['client']):
                    try:os.killpg(child.pid,signal.SIGTERM)
                    except ProcessLookupError:pass
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        if same(state['client']):
                            try:os.killpg(child.pid,signal.SIGKILL)
                            except ProcessLookupError:pass
                        child.wait(timeout=10)
            except Exception as exc:
                state['cleanup_errors'].append({'child_cleanup_error':repr(exc)})
        try:
            health=fetch('/arc3/persistent/health')
            observed=set(health['sessions'])-previous
            attempted=attempted_sessions(output)
            new=observed & attempted
            # Even noncooperating traffic that ignores the experiment locks is
            # left intact: this run must have persisted its own start request.
            state['attempted_session_ids']=sorted(attempted)
            state['unowned_new_session_ids']=sorted(observed-attempted)
            if observed-attempted:
                state['cleanup_errors'].append({'error':'Unowned new sessions; left untouched'})
            def close(sid):
                if health['sessions'][sid]['state']=='closed':return
                try:fetch('/arc3/persistent/'+sid+'/close',{},timeout=15)
                except Exception as exc:return dict(session_id=sid,error=repr(exc))
            with ThreadPoolExecutor(max_workers=8) as pool:
                state['cleanup_errors'].extend(e for e in pool.map(close,sorted(new)) if e)
            final=fetch('/arc3/persistent/health')
            state['new_session_ids']=sorted(new)
            state['final_owned_states']={s:final['sessions'][s]['state'] for s in sorted(new)}
            if any(v!='closed' for v in state['final_owned_states'].values()):
                state['cleanup_errors'].append({'error':'Unclosed owned session'})
            save(here/(label+'-health-after.json'),final)
            server_profile=fetch('/arc3/persistent/profile-status')
            save(here/(label+'-server-profile-after.json'),server_profile)
            validate_server_profile(server_profile,server['root'])
            save(here/(label+'-graph-after.json'),fetch('/collective_rpc',
                {'method':'arc3_eviction_graph_status','timeout':20},timeout=25))
        except Exception as exc:state['cleanup_errors'].append({'error':repr(exc)})
        if metrics is not None:state['metrics_observer']=metrics.stop()
        if state['cleanup_errors']:state['stage']='failed'
        state.update(complete=True,ended_epoch=time.time());save(state_path,state)
        if observer is not None:
            try:observer.wait(timeout=10)
            except subprocess.TimeoutExpired:observer.terminate();observer.wait(timeout=10)
            state['observer_returncode']=observer.returncode
            try:
                mem=json.loads((here/(label+'-gpu-watch')/'summary.json').read_text())
                if observer.returncode or not mem.get('complete') or mem.get('error') or not mem.get('overall',{}).get('samples'):
                    state.update(stage='failed',memory_observer_error='Missing/incomplete observer')
            except (OSError,ValueError) as exc:
                state.update(stage='failed',memory_observer_error=repr(exc))
        if metrics is not None:
            observed=state['metrics_observer']
            if (not observed['complete'] or observed['successful_samples']<2
                    or observed.get('first_success_epoch',float('inf'))>state.get('gameplay_started_epoch',0)+5
                    or observed.get('last_success_epoch',0)<state.get('gameplay_deadline_epoch',float('inf'))-5):
                state.update(stage='failed',metrics_observer_error='Missing/incomplete observer or gameplay coverage')
        save(state_path,state);locks.close()
    print(json.dumps({k:v for k,v in state.items() if k not in ('argv','previous_session_ids')},indent=2),flush=True)
    if state['stage']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
