"""Run one hash-pinned local validation client with bounded memory observation.

The job is a reviewed JSON file in the candidate directory. This controller
does not launch or stop the model server. Client code owns experiment locks;
only session IDs from persisted start requests are eligible for final cleanup.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
import re
from pathlib import Path
import signal
import socket
import subprocess
import time
import urllib.error
import urllib.request


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--candidate-dir',type=Path,required=True)
    parser.add_argument('--job',required=True)
    parser.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if not args.execute:
        print('No execution without --execute');return
    root=Path('/workspace/arc3-inference');here=args.candidate_dir.resolve()
    assert here.parent==root/'runs'
    assert socket.gethostname()=='9b2fd24de2c6'
    boot=Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    assert boot=='1bd8be96-58e2-4485-8fd1-c0e99dc12e74'
    job_path=here/args.job;assert job_path.parent==here
    job=json.loads(job_path.read_text())
    label=job['label'];assert label and all(c.isalnum() or c in '-_' for c in label)
    client=here/job['client'];assert client.parent==here and client.suffix=='.py'
    assert hashlib.sha256(client.read_bytes()).hexdigest()==job['client_sha256']
    for name,sha in job.get('helper_sha256',{}).items():
        f=here/name;assert f.parent==here
        assert hashlib.sha256(f.read_bytes()).hexdigest()==sha,name
    timeout=job['timeout_seconds'];assert type(timeout) is int and 120<=timeout<=3000
    output=here/label;assert not output.exists()
    argv=job['argv'];assert all(isinstance(v,str) for v in argv)
    assert argv.count('--execute')==1 and argv.count('--output')==1
    assert argv[argv.index('--output')+1]==str(output)
    saved=json.loads((here/'candidate-server.json').read_text())
    current=json.loads((root/'server-identity.json').read_text())
    assert saved['root']==current['root'] and saved['boot_id']==boot
    def fetch(route,body=None,wait=20):
        request=urllib.request.Request('http://127.0.0.1:1234'+route,
            data=None if body is None else json.dumps(body).encode(),
            headers={'Content-Type':'application/json'})
        with urllib.request.urlopen(request,timeout=wait) as response:return json.load(response)
    health=fetch('/arc3/persistent/health')
    assert health['state']=='ready' and not health['prefix_caching']
    assert all(s['state'] in ('closed','error') for s in health['sessions'].values())
    pid=saved['root']['pid']
    fields=Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()
    assert fields[0]!='Z' and int(fields[19])==saved['root']['start_ticks']
    environ=dict(item.split('=',1) for item in Path(f'/proc/{pid}/environ').read_bytes().decode().split('\0') if '=' in item)
    initialized=[]
    for key,name in [('ARC3_PERSISTENT_ENGINE_AUDIT','engine-events.jsonl'),
                     ('ARC3_PERSISTENT_SERVICE_AUDIT','service-events.jsonl')]:
        path=here/name;assert environ[key]==str(path)
        if not path.exists():path.open('x').close();initialized.append(name)
    own=Path(f'/proc/{os.getpid()}/stat').read_text().rsplit(')',1)[1].split()
    state_path=here/(label+'-controller.json');assert not state_path.exists()
    state=dict(stage='client_start',complete=False,started_epoch=time.time(),
        campaign_deadline_epoch=time.time()+timeout+120,hostname=socket.gethostname(),boot_id=boot,
        controller=dict(pid=os.getpid(),state=own[0],pgid=int(own[2]),sid=int(own[3]),start_ticks=int(own[19])),
        job_sha256=hashlib.sha256(job_path.read_bytes()).hexdigest(),job=job,
        initialized_empty_audit_files=initialized,cleanup_errors=[])
    def save():
        tmp=state_path.with_suffix('.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(state_path)
    save()
    with (here/(label+'-gpu-watch.log')).open('x') as log:
        observer=subprocess.Popen(['python3',str(here/'monitor_candidate_gpu_memory.py'),
            '--state',str(state_path),'--output',str(here/(label+'-gpu-watch')),
            '--max-seconds',str(timeout+180)],stdout=log,stderr=subprocess.STDOUT)
    child=None
    try:
        command=['bash',str(root/'runtime_python.sh'),str(client),*argv]
        state['command']=command
        with (here/(label+'.log')).open('x') as log:
            child=subprocess.Popen(command,env=dict(os.environ,ARC3_ROOT=str(root),VLLM_PLUGINS=''),
                stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        state['client_pid']=child.pid;state['stage']='running';save()
        deadline=time.monotonic()+timeout;last=0
        while child.poll() is None:
            if time.monotonic()>=deadline:raise TimeoutError('Bounded client deadline expired')
            if time.monotonic()-last>=30:
                progress={}
                result_path=output/'results.json'
                if result_path.exists():
                    try:
                        result=json.loads(result_path.read_text())
                        progress={k:result.get(k) for k in ('status','stage','completed_output_tokens') if k in result}
                        cohorts=result.get('cohorts',[])
                        progress['completed_cohorts']=[{k:c.get(k) for k in ('concurrency','status')} for c in cohorts]
                    except (ValueError,OSError):pass
                client_state=output/'state.json'
                if client_state.exists():
                    try:
                        current_stage=json.loads(client_state.read_text()).get('stage','running')
                        state['stage']='client-'+str(current_stage)
                    except (OSError,ValueError):pass
                state['progress']=progress;state['updated_epoch']=time.time();save()
                print(json.dumps({'stage':state['stage'],'elapsed':round(time.time()-state['started_epoch']),'progress':progress}),flush=True)
                last=time.monotonic()
            time.sleep(2)
        state['returncode']=child.returncode
        if child.returncode:raise RuntimeError('Client failed; inspect preserved log/results')
        result=json.loads((output/'results.json').read_text())
        assert result['status']=='passed',result['status']
        state['stage']='passed'
    except BaseException as exc:
        state.update(stage='failed',error=repr(exc))
    finally:
        if child is not None and child.poll() is None:
            try:
                os.killpg(child.pid,signal.SIGTERM)
                try:child.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=10)
            except ProcessLookupError:pass
            except Exception as exc:
                state['cleanup_errors'].append({'process_error':repr(exc)})
        ids=set()
        # A benchmark buffers full HTTP evidence until its timing ends. Its
        # explicit planned-owner registry is persisted before any start call.
        for path in [*output.rglob('*-request.json'), *output.rglob('results.json')]:
            try:
                item=json.loads(path.read_text());body=item.get('payload')
                if path.name=='results.json':
                    found=item.get('session_ids',[])
                elif item.get('route','').endswith('/start') and isinstance(body,dict):
                    found=[body['session_id']] if body.get('session_id') else []
                else:found=[]
                assert isinstance(found,list)
                for sid in found:
                    assert isinstance(sid,str) and re.fullmatch(r'[A-Za-z0-9_.-]{1,200}',sid)
                    ids.add(sid)
            except Exception as exc:
                state['cleanup_errors'].append(dict(evidence=str(path),error=repr(exc)))
        def close(sid):
            try:
                fetch('/arc3/persistent/'+sid+'/close',{},wait=5)
                return None
            except urllib.error.HTTPError as exc:
                # Planned owners may never have reached the server on failure.
                return None if exc.code==404 else dict(session_id=sid,error=repr(exc))
            except Exception as exc:return dict(session_id=sid,error=repr(exc))
        with ThreadPoolExecutor(max_workers=8) as pool:
            state['cleanup_errors'].extend(error for error in pool.map(close,sorted(ids)) if error)
        state['cleanup_recovered_session_ids']=sorted(ids)
        try:
            health=fetch('/arc3/persistent/health')
            state['final_owned_states']={sid:health.get('sessions',{}).get(sid,{}).get('state') for sid in ids}
            if any(v not in ('closed',None) for v in state['final_owned_states'].values()):
                state['cleanup_errors'].append({'error':'An owned session is not closed'})
        except Exception as exc:state['cleanup_errors'].append({'health_error':repr(exc)})
        if state['cleanup_errors']:state['stage']='failed'
        state['complete']=True;state['ended_epoch']=time.time();save()
        try:
            try:observer.wait(timeout=10)
            except subprocess.TimeoutExpired:
                observer.terminate();observer.wait(timeout=10)
        except Exception as exc:
            state.update(stage='failed',memory_observer_error=repr(exc))
        state['observer_returncode']=observer.returncode
        summary_path=here/(label+'-gpu-watch')/'summary.json'
        try:summary=json.loads(summary_path.read_text()) if summary_path.exists() else {}
        except (OSError,ValueError) as exc:
            summary={};state['memory_observer_error']=repr(exc)
        if observer.returncode or not summary.get('complete') or summary.get('error') or not summary.get('overall',{}).get('samples'):
            state.update(stage='failed',memory_observer_error='Missing or failed memory observation')
        save();print(json.dumps(state),flush=True)
    if state['stage']!='passed':raise SystemExit(1)


if __name__=='__main__':main()
