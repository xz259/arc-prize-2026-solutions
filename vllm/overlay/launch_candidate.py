"""Run on the existing pod through runtime_python.sh; preserve baseline identity.

Commands: plan, stop-baseline, start, status. Baseline is the validated phase-two
server, not the original stateless deployment. This never edits the extracted
runtime. Candidate modules and entry-point metadata are an explicit PYTHONPATH
overlay inherited by spawned engine and worker processes.
"""
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import sys
from urllib.request import urlopen

ROOT = Path('/workspace/arc3-inference')
HERE = Path(__file__).resolve().parent
BASELINE = HERE / 'baseline-server.json'
assert socket.gethostname() == 'd1896a7c21ee'
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == '4ccd74e4-d6d3-4ae0-9081-4cfab45143a2'
assert HERE.is_relative_to(ROOT / 'runs')
if not BASELINE.exists():
    prior = json.loads((ROOT / 'server-identity.json').read_text())
    assert prior['root']['pid'] == 15554 and prior['root']['start_ticks'] == 171381368
    assert prior['root']['cmdline_sha256'] == '71f1005ef7a3edab2fd5dc167af9cb7f8430e8bd7f4d7390dcfbfb08bbb1553b'
    assert prior['persistent_experiment']['phase'] == 'managed-suffix-eviction-mtp0'
    shutil.copy2(ROOT / 'server-identity.json', BASELINE)
old = json.loads(BASELINE.read_text())
assert '--no-enable-prefix-caching' in old['argv']
assert '--speculative-config' not in old['argv']
os.environ.update(old['environment'])
os.environ.update(ARC3_ROOT=str(ROOT), ARC3_CONTEXT='40000', ARC3_CONCURRENCY='36',
                  ARC3_KV_GIB='0', ARC3_GPU_MEMORY_UTILIZATION='0.95', ARC3_GRAPH_SIZE='40',
                  ARC3_MEMORY_DIAGNOSTICS='1', ARC3_MTP_TOKENS='0', ARC3_PREFIX_CACHE='0')
sys.path.insert(0, str(ROOT))
import minimal_inference as m

cmd = sys.argv[-1]
if cmd == 'stop-baseline':
    current = json.loads(m.IDENTITY.read_text())
    assert current['root'] == old['root'], 'Current server is not the saved baseline'
    with urlopen('http://127.0.0.1:1234/arc3/persistent/health', timeout=10) as response:
        health = json.load(response)
    assert all(s['state'] in ('closed', 'error') for s in health['sessions'].values()), 'Resident session still owned'
    with urlopen('http://127.0.0.1:1234/metrics', timeout=10) as response:
        metrics = response.read().decode()
    for line in metrics.splitlines():
        if line.startswith(('vllm:num_requests_running{', 'vllm:num_requests_waiting{')):
            assert float(line.rsplit(' ', 1)[1]) == 0, 'Active requests: ' + line
    m.stop()
elif cmd in ('plan', 'start'):
    overlay_manifest = json.loads((HERE / 'overlay-manifest.json').read_text())
    for name, checksum in overlay_manifest.items():
        assert hashlib.sha256((HERE/name).read_bytes()).hexdigest() == checksum, name
    os.environ.update(
        PYTHONPATH=str(HERE) + ':' + os.environ['PYTHONPATH'],
        VLLM_PLUGINS='arc3_persistent_engine,arc3_persistent',
        ARC3_PERSISTENT_ENABLE='1', ARC3_PERSISTENT_MAX_SESSIONS='4',
        ARC3_EVICTION_VERIFY_BYTES='1', ARC3_EVICTION_VERIFY_ALLOCATOR='1',
        ARC3_PERSISTENT_ENGINE_AUDIT=str(HERE/'engine-events.jsonl'),
        ARC3_PERSISTENT_SERVICE_AUDIT=str(HERE/'service-events.jsonl'))
    candidate_argv = list(old['argv'])
    if '--enforce-eager' not in candidate_argv:
        candidate_argv.append('--enforce-eager')
    # Phase-two overlay must not remain in the import search path: all candidate
    # modules are pinned together, with stock custom runtime modules after them.
    prior_overlay = old['persistent_experiment']['directory']
    paths = [str(HERE), str(ROOT), *[p for p in os.environ['PYTHONPATH'].split(os.pathsep)
                                   if p and p != prior_overlay]]
    os.environ['PYTHONPATH'] = os.pathsep.join(dict.fromkeys(paths))
    m.server_command = lambda: candidate_argv
    if cmd == 'plan':
        print(json.dumps({'argv': m.server_command(), 'overlay': str(HERE),
                          'manifest': overlay_manifest, 'baseline_root': old['root']}, indent=2))
    else:
        m.start()
        current = json.loads(m.IDENTITY.read_text())
        current['persistent_experiment'] = {'directory': str(HERE), 'manifest': overlay_manifest,
                                            'phase': 'oldest-interaction-eviction-mtp0', 'prefix_caching': False,
                                            'eager_correctness_candidate': True,
                                            'phase2_reference_root': old['root']}
        current['prolong_template'] = old.get('prolong_template')
        current['overlay_environment'] = {key: os.environ[key] for key in (
            'PYTHONPATH', 'VLLM_PLUGINS', 'ARC3_PERSISTENT_ENABLE',
            'ARC3_PERSISTENT_MAX_SESSIONS', 'ARC3_EVICTION_VERIFY_BYTES',
            'ARC3_EVICTION_VERIFY_ALLOCATOR', 'ARC3_PERSISTENT_ENGINE_AUDIT',
            'ARC3_PERSISTENT_SERVICE_AUDIT')}
        m.save(m.IDENTITY, current)
        m.save(HERE/'candidate-server.json', current)
elif cmd == 'status':
    m.status()
else:
    raise ValueError(cmd)
