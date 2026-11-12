"""Own one five-minute, two-game persistent validation and its final cleanup."""
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

ROOT = Path('/workspace/arc3-inference')
HERE = Path(__file__).resolve().parent
CLIENT = HERE/'harness-client-v1'
OVERLAY = ROOT/'runs/prolong-v5.22-release-20260912-v1/overlay'

def request(path, payload=None):
    data = None if payload is None else json.dumps(payload).encode()
    with urlopen(Request('http://127.0.0.1:1234'+path, data=data,
                         headers={'Content-Type':'application/json'}), timeout=20) as r:
        return json.load(r)

def save(name, value):
    (HERE/name).write_text(json.dumps(value, indent=2)+'\n')

def main():
    assert socket.gethostname() == 'd1896a7c21ee'
    assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == '4ccd74e4-d6d3-4ae0-9081-4cfab45143a2'
    locks=[]
    for name in ('public-eval-campaign.lock','mtp-precision-sweep.lock',
                 'bf16-gameplay-ablation.lock','mtp0-memory-diagnostic.lock'):
        lock=(ROOT/'runs'/name).open('a+')
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);locks.append(lock)
    for name, expected in json.loads((CLIENT/'manifest.json').read_text()).items():
        assert hashlib.sha256((CLIENT/name).read_bytes()).hexdigest()==expected,name
    for name, expected in json.loads((OVERLAY.parent/'source-sha256.json').read_text()).items():
        assert hashlib.sha256((OVERLAY/name).read_bytes()).hexdigest()==expected,name
    health=request('/arc3/persistent/health')
    assert health['prefix_caching'] is False and health['max_context']==40000
    assert health['mode']=='persistent-history-eviction-mtp0'
    identity=json.loads((ROOT/'server-identity.json').read_text())
    assert identity['persistent_experiment']['phase']=='oldest-interaction-eviction-mtp0'
    assert '--enforce-eager' in identity['argv']
    assert all(s['state'] in ('closed','error') for s in health['sessions'].values())
    previous_sessions=set(health['sessions'])
    output=HERE/'gameplay-5min-v1'
    assert not output.exists()
    argv=[str(ROOT/'venv-duck/bin/python'),'-B',str(CLIENT/'run_persistent_gameplay.py'),
          '--persistent-base-url','http://127.0.0.1:1234','--games','ls20-9607627b,ft09-0d8bbf25',
          '--total-seconds','300','--context','40000','--expected-sequences','36','--expected-mtp','0',
          '--context-profile','nvfp4','--expected-gpu-memory-utilization','0.95','--active-games','0',
          '--request-slots','0','--clock-mode','gameplay','--shutdown-margin','0','--no-token-limit',
          '--no-response-cap','--response-headroom-fraction','0.10','--min-p','0',
          '--reasoning-effort','xhigh','--image-scale','4','--stdout-token-limit','1024',
          '--kv-eviction-policy','oldest_interactions','--kv-evict-fraction','0.10',
          '--output',str(output),'--execute']
    deps=[OVERLAY,ROOT/'duck/src/ARC3-Inference',ROOT/'duck/src/tufa-arc-agi-framework/src',ROOT]
    env=dict(os.environ,ARC3_ROOT=str(ROOT),PYTHONPATH=os.pathsep.join(map(str,deps)),
             MPLCONFIGDIR=str(HERE/'harness-mpl'),MPLBACKEND='Agg',HF_HUB_OFFLINE='1',
             TRANSFORMERS_OFFLINE='1',PYTHONNOUSERSITE='1',PYTHONDONTWRITEBYTECODE='1')
    record={'started':time.time(),'argv':argv,'server':json.loads((ROOT/'server-identity.json').read_text()),
            'previous_session_ids':sorted(previous_sessions),'stage':'starting'}
    save('gameplay-controller.json',record)
    child=None
    try:
        with (HERE/'gameplay-stdout.log').open('w') as log:
            child=subprocess.Popen(argv,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,
                                   start_new_session=True)
            record.update(pid=child.pid,stage='running');save('gameplay-controller.json',record)
            record['returncode']=child.wait(timeout=480)
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid,signal.SIGTERM)
            try:child.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=15)
        # A killed Python client cannot run its finally block. Close only session
        # IDs created during this exclusively owned experiment, and record it.
        cleanup=[]
        try:
            final=request('/arc3/persistent/health')
            for sid in set(final['sessions'])-previous_sessions:
                if final['sessions'][sid]['state'] not in ('closed','error'):
                    cleanup.append(request('/arc3/persistent/'+sid+'/close',{}))
            final=request('/arc3/persistent/health')
            record['new_sessions']={sid:s for sid,s in final['sessions'].items() if sid not in previous_sessions}
            record['unclosed_sessions']={sid:s for sid,s in record['new_sessions'].items() if s['state']!='closed'}
        except Exception as exc:record['cleanup_error']=repr(exc)
        record.update(stage='finished',ended=time.time(),controller_cleanup=cleanup)
        save('gameplay-controller.json',record)
    print(json.dumps({k:v for k,v in record.items() if k not in ('argv','server','previous_session_ids')},indent=2))
    if record.get('returncode')!=0 or record.get('cleanup_error') or record.get('unclosed_sessions'):
        raise RuntimeError('Gameplay control needs review; see controller record')

if __name__=='__main__':main()
