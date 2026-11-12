"""Immutable candidate launch on the verified existing pod; run via runtime_python.sh."""
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
assert socket.gethostname() == '9b2fd24de2c6'
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == '1bd8be96-58e2-4485-8fd1-c0e99dc12e74'
assert HERE.is_relative_to(ROOT / 'runs')
config = json.loads((HERE/'optimization-config.json').read_text())
manifest = json.loads((HERE/'overlay-manifest.json').read_text())
for name, expected in manifest.items():
    assert hashlib.sha256((HERE/name).read_bytes()).hexdigest() == expected, name
os.environ.update(
    ARC3_ROOT=str(ROOT), ARC3_CONTEXT='40000', ARC3_CONCURRENCY='30',
    ARC3_KV_GIB='0', ARC3_GPU_MEMORY_UTILIZATION='0.95', ARC3_GRAPH_SIZE='32',
    ARC3_BATCHED_TOKENS='8192', ARC3_MTP_TOKENS='0', ARC3_PREFIX_CACHE='0',
    ARC3_EAGER=str(config['eager']), ARC3_QSA_MODE='radix', ARC3_QSA_KV_FORMAT='nvfp4',
    ARC3_QSA_FP8_READER='baseline', ARC3_MOE_FINALIZE='unfused',
    ARC3_TOPK_THREADS='512', ARC3_TOPK_CACHE_ROW='1', ARC3_QSA_PREFILL_PROFILE='stock',
    ARC3_SKIP_REDUNDANT_TOPK='1', ARC3_MTP_MIN_P='1', ARC3_MEMORY_DIAGNOSTICS='1',
    VLLM_USE_DEEP_GEMM='1', VLLM_MOE_USE_DEEP_GEMM='0',
    VLLM_DISABLE_SHARED_EXPERTS_STREAM='0', VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD='256',
    VLLM_PLUGINS='arc3_persistent_engine,arc3_persistent',
    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
    ARC3_PERSISTENT_ENABLE='1', ARC3_PERSISTENT_MAX_SESSIONS=str(config['max_sessions']),
    ARC3_EVICTION_VERIFY_BYTES=str(config['verify_bytes']),
    ARC3_EVICTION_VERIFY_ALLOCATOR=str(config['verify_allocator']),
    ARC3_EVICTION_GRAPH_MAPPING=str(config['graph_mapping']),
    ARC3_EVICTION_GRAPH_TRACE=str(config.get('graph_trace',0)),
    ARC3_EVICTION_PROFILE='1',
    ARC3_PERSISTENT_ENGINE_AUDIT=str(HERE/'engine-events.jsonl'),
    ARC3_PERSISTENT_SERVICE_AUDIT=str(HERE/'service-events.jsonl'),
)
sys.path.insert(0, str(ROOT))
os.environ['PYTHONPATH'] = str(HERE)+':'+str(ROOT)+':'+os.environ['PYTHONPATH']
import minimal_inference as m

TEMPLATE = ROOT/'runs/reasoning-continuation-tools-20260911-v1/prolong_chat_template_v2.jinja'
assert hashlib.sha256(TEMPLATE.read_bytes()).hexdigest() == 'c31085d9cfd7e81fffa8bf128e13ad319cbc50b73109547ab05f43dfe58f535f'
base_server_command = m.server_command
def server_command():
    argv = base_server_command()
    argv[argv.index('--chat-template')+1] = str(TEMPLATE)
    return argv
m.server_command = server_command

action = sys.argv[1]
if action == 'stop-baseline':
    identity = json.loads(m.IDENTITY.read_text())
    expected = json.loads((HERE/'expected-baseline.json').read_text())
    assert identity['root'] == expected['root'], 'Current server is not the pinned baseline'
    with urlopen('http://127.0.0.1:1234/metrics', timeout=10) as response:
        metrics = response.read().decode()
    for line in metrics.splitlines():
        if line.startswith(('vllm:num_requests_running{', 'vllm:num_requests_waiting{')):
            assert float(line.rsplit(' ',1)[1]) == 0, line
    if identity.get('persistent_experiment'):
        with urlopen('http://127.0.0.1:1234/arc3/persistent/health',timeout=10) as response:
            health=json.load(response)
        assert all(s['state'] in ('closed','error') for s in health['sessions'].values())
    shutil.copy2(m.IDENTITY, HERE/'previous-server.json')
    m.stop()
elif action == 'start':
    assert not (HERE/'candidate-server.json').exists(), 'Immutable launch already exists'
    assert not m.port_open(), 'Port occupied'
    m.save(HERE/'requested-config.json', dict(argv=m.server_command(), config=config,
        environment={k:v for k,v in os.environ.items() if k.startswith(('ARC3_','VLLM_'))
                     or k in ('PYTHONPATH','OMP_NUM_THREADS','MKL_NUM_THREADS')}))
    m.start()
    identity = json.loads(m.IDENTITY.read_text())
    identity['persistent_experiment'] = dict(directory=str(HERE), manifest=manifest,
        phase='eviction-optimization-v2', prefix_caching=False,
        eager_correctness_candidate=bool(config['eager']), optimization_config=config)
    identity['prolong_template'] = dict(path=str(TEMPLATE), continuation_sha256=hashlib.sha256(TEMPLATE.read_bytes()).hexdigest())
    m.save(m.IDENTITY, identity)
    m.save(HERE/'candidate-server.json', identity)
elif action == 'status':
    m.status()
else:
    raise ValueError(action)
