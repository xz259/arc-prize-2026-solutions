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
assert socket.gethostname() == '263819044b0b'
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == '4c1eccbd-6c61-455d-b472-8a9de7d4db6c'
assert HERE.is_relative_to(ROOT / 'runs')
config = json.loads((HERE/'optimization-config.json').read_text())
# Stage 6/7 verification server: ARC3_STAGE_VERIFY=1 turns on the eviction byte/allocator proofs the live controls audit (controls only; not for throughput).
if os.environ.get('ARC3_STAGE_VERIFY', '0') == '1':
    config['verify_bytes'] = 1; config['verify_allocator'] = 1
if os.environ.get('ARC3_STAGE4_MAX_SEQS'):
    config['max_sessions'] = int(os.environ['ARC3_STAGE4_MAX_SEQS'])   # stage 4: clients require max_sessions >= concurrency
for key in ('ARC3_PROFILE_DIR', 'ARC3_PROFILE_STEPS', 'ARC3_DENSE_EXPERIMENT', 'ARC3_PRIVATE_MOE_SERVING', 'ARC3_PLE_HANDOFF_RECEIPTS', 'ARC3_ENGINE_TIMING_ENABLE', 'ARC3_PERSISTENT_ONE_LOOKAHEAD', 'ARC3_PERSISTENT_ONE_LOOKAHEAD_IMAGE_DIAGNOSTIC'):
    assert not os.environ.get(key), ('Unexpected inherited experiment', key)
# Bounded torch profiler for a step-level kernel profile (2026-09-21): set only through the stage knob, after the guard above.
if os.environ.get('ARC3_STAGE_PROFILE_DIR'):
    os.environ['ARC3_PROFILE_DIR'] = os.environ['ARC3_STAGE_PROFILE_DIR']; os.environ['ARC3_PROFILE_STEPS'] = os.environ.get('ARC3_STAGE_PROFILE_STEPS', '8')
manifest = json.loads((HERE/'overlay-manifest.json').read_text())
for name, expected in manifest.items():
    assert hashlib.sha256((HERE/name).read_bytes()).hexdigest() == expected, name
assert os.environ.get('ARC3_STAGE_KV_FORMAT', 'nvfp4') in ('bf16', 'fp8_e4m3', 'fp8_e5m2', 'nvfp4'), os.environ.get('ARC3_STAGE_KV_FORMAT')
os.environ.update(
    ARC3_ROOT=str(ROOT), ARC3_CONTEXT=os.environ.get('ARC3_STAGE4_CONTEXT', '40000'), ARC3_CONCURRENCY=os.environ.get('ARC3_STAGE4_MAX_SEQS', '30'),
    ARC3_KV_GIB=os.environ.get('ARC3_STAGE4_KV_GIB', '0'), ARC3_GPU_MEMORY_UTILIZATION=os.environ.get('ARC3_STAGE4_UTIL', '0.95'), ARC3_GRAPH_SIZE=os.environ.get('ARC3_STAGE4_GRAPH_SIZE', '32'),
    ARC3_BATCHED_TOKENS=os.environ.get('ARC3_STAGE2_BATCHED_TOKENS', '8192'), ARC3_MTP_TOKENS='0', ARC3_PREFIX_CACHE='0',  # stage 2: 4096 candidate
    ARC3_EAGER=str(config['eager']), ARC3_QSA_MODE='radix', ARC3_QSA_KV_FORMAT=os.environ.get('ARC3_STAGE_KV_FORMAT', 'nvfp4'),  # stage 11: KV format NLL comparison
    ARC3_QSA_FP8_READER='baseline', ARC3_MOE_FINALIZE='unfused',
    ARC3_TOPK_THREADS='512', ARC3_TOPK_CACHE_ROW='1', ARC3_QSA_PREFILL_PROFILE='stock',
    ARC3_SKIP_REDUNDANT_TOPK='1', ARC3_MTP_MIN_P='1', ARC3_MEMORY_DIAGNOSTICS='1',
    VLLM_USE_DEEP_GEMM='1', VLLM_MOE_USE_DEEP_GEMM='0',
    VLLM_DISABLE_SHARED_EXPERTS_STREAM='0', VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD='256',
    VLLM_PLUGINS='arc3_persistent_engine,arc3_persistent',
    ARC3_PERSISTENT_ONE_LOOKAHEAD='1',
    ARC3_PERSISTENT_ONE_LOOKAHEAD_IMAGE_DIAGNOSTIC='1',
    OMP_NUM_THREADS='1', MKL_NUM_THREADS='1',
    ARC3_PERSISTENT_ENABLE='1', ARC3_PERSISTENT_MAX_SESSIONS=os.environ.get('ARC3_STAGE4_MAX_SEQS', str(config['max_sessions'])),
    ARC3_EVICTION_VERIFY_BYTES=str(config['verify_bytes']),
    ARC3_EVICTION_VERIFY_ALLOCATOR=str(config['verify_allocator']),
    ARC3_EVICTION_GRAPH_MAPPING=str(config['graph_mapping']),
    ARC3_EVICTION_GRAPH_TRACE=str(config.get('graph_trace',0)),
    ARC3_EVICTION_PROFILE=str(config.get('profile',1)),
    ARC3_EVICTION_FUSED_CONTROL=str(config.get('fused_control',0)),
    ARC3_EFFICIENCY_IDENTITY_LOOKUP=str(config.get('identity_lookup',0)),
    ARC3_EFFICIENCY_VALIDATED_CONTEXT_COPY=str(config.get('validated_context_copy',0)),
    ARC3_EFFICIENCY_PAGE_PLAN_REUSE=str(config.get('page_plan_reuse',0)),
    ARC3_EFFICIENCY_SKIP_UNCHANGED_SCALARS=str(config.get('skip_unchanged_scalars',0)),
    ARC3_PERSISTENT_SINGLE_BATCH=str(config.get('single_batch',0)),
    ARC3_RESUME_PROFILE=str(config.get('resume_profile',0)),
    ARC3_RESUME_PROFILE_PATH=str(HERE/'resume-profile'),
    ARC3_GAMEPLAY_PROFILE=str(config.get('gameplay_profile',0)),
    ARC3_GAMEPLAY_PROFILE_PATH=str(HERE/'gameplay-server-profile'),
    ARC3_PERSISTENT_ENGINE_AUDIT=str(HERE/'engine-events.jsonl'),
    ARC3_PERSISTENT_SERVICE_AUDIT=str(HERE/'service-events.jsonl'),
)
if config.get('pin_v2_runner',0):
    # Native startup validation still checks all MRV2 unsupported features.
    # Subsequent property reads need not rescan installed plugin entry points.
    os.environ['VLLM_USE_V2_MODEL_RUNNER']='1'
sys.path.insert(0, str(ROOT))
os.environ['PYTHONPATH'] = str(HERE)+':'+str(ROOT)+':'+os.environ['PYTHONPATH']
# Install only in fresh child interpreters; normal runtime wrapper is already active.
ADAPTER = HERE/'private_moe'
os.environ.update(ARC3_PRIVATE_MOE_SERVING=os.environ.get('ARC3_STAGE4_PRIVATE_MOE', '1'),
    ARC3_PRIVATE_MOE_SETTINGS=str(ADAPTER/'settings.json'),
    ARC3_PRIVATE_MOE_RECEIPTS=str(HERE/'private-startup'))
os.environ['PYTHONPATH'] = str(ADAPTER/'bootstrap')+':'+str(ADAPTER)+':'+os.environ['PYTHONPATH']
# Common measured projection tuning; explicitly compose MoE startup.
os.environ.update(ARC3_DENSE_EXPERIMENT='1',
    ARC3_DENSE_RECEIPTS=str(HERE/'dense-startup'),
    ARC3_DENSE_AOT_ROOT=str(HERE/'aot-cache'),
    ARC3_DENSE_CHAIN_BOOTSTRAP=str(ADAPTER/'bootstrap'/'sitecustomize.py'))
os.environ['PYTHONPATH'] = str(HERE/'dense_projection_adapter_v1'/'bootstrap')+':'+os.environ['PYTHONPATH']
os.environ['ARC3_PLE_HANDOFF_RECEIPTS'] = str(HERE/'ple-handoff')
os.environ['PYTHONPATH'] = str(HERE/'combined-bootstrap')+':'+os.environ['PYTHONPATH']
# Dense FP8 weight-only (stage 1 of THROUGHPUT_MEMORY_PLAN_2026-09-19). Tier is inherited from the launching shell:
# off (byte-identical composition to qsa_frequency_eviction_v1), inventory (record only), A, AB2, AB2C; lm_head via ARC3_DENSE_FP8_LM_HEAD=1.
FP8 = HERE/'dense_fp8_adapter_v1'
os.environ.setdefault('ARC3_DENSE_FP8_TIER', 'off')
assert os.environ['ARC3_DENSE_FP8_TIER'] in ('off', 'inventory', 'A', 'AB2', 'AB2C', 'A_qkvz', 'A_out', 'A_qkv'), os.environ['ARC3_DENSE_FP8_TIER']
os.environ['ARC3_DENSE_FP8_FAKE'] = os.environ.get('ARC3_STAGE_FP8_FAKE', 'off'); assert os.environ['ARC3_DENSE_FP8_FAKE'] in ('off', 'e4m3', 'e5m2')  # stage 13
os.environ.update(ARC3_DENSE_FP8_RECEIPTS=str(HERE/'dense-fp8-startup'),
    ARC3_DENSE_FP8_CHAIN_BOOTSTRAP=str(HERE/'combined-bootstrap'/'sitecustomize.py'))
os.environ['PYTHONPATH'] = str(FP8/'bootstrap')+':'+os.environ['PYTHONPATH']
# Stage 5 handoff implementation: copy (served Sept 16 acknowledgement fix) | direct (ple_direct_handoff_v1: model-thread
# staging + polled flag, no request thread / acknowledgement / per-step ZMQ). Receipts + cache tag on direct.
os.environ.setdefault('ARC3_PLE_HANDOFF_IMPL', 'copy')
assert os.environ['ARC3_PLE_HANDOFF_IMPL'] in ('copy', 'direct', 'hostpoll'), os.environ['ARC3_PLE_HANDOFF_IMPL']
if os.environ['ARC3_PLE_HANDOFF_IMPL'] == 'direct':
    os.environ['ARC3_PLE_DIRECT_RECEIPTS'] = str(HERE/'ple-direct-startup')
if os.environ['ARC3_PLE_HANDOFF_IMPL'] == 'hostpoll':
    os.environ['ARC3_PLE_HOSTPOLL_RECEIPTS'] = str(HERE/'ple-hostpoll-startup')
    os.environ['ARC3_PLE_HOSTPOLL_BUILD'] = str(HERE/'ple_hostpoll_v1'/'build')   # precompiled polling kernel
# Stage 5 PLE critical-path discriminator (THROUGHPUT-ONLY diagnostic): off | lookup | handoff; chains the dense-fp8 bootstrap.
PLED = HERE/'ple_discriminator_v1'
os.environ.setdefault('ARC3_PLE_DISCRIMINATOR', 'off')
assert os.environ['ARC3_PLE_DISCRIMINATOR'] in ('off', 'lookup', 'handoff', 'nowait', 'timing', 'nogpu'), os.environ['ARC3_PLE_DISCRIMINATOR']
if os.environ['ARC3_PLE_DISCRIMINATOR'] != 'off':
    os.environ.update(ARC3_PLE_DISCRIMINATOR_RECEIPTS=str(HERE/'ple-discriminator-startup'),
        ARC3_PLE_DISC_CHAIN_BOOTSTRAP=str(FP8/'bootstrap'/'sitecustomize.py'))
    os.environ['PYTHONPATH'] = str(PLED/'bootstrap')+':'+os.environ['PYTHONPATH']
# Stage 3 fixed-state fold (ring group + PLE state share one block per request): off | A. Outermost bootstrap link.
FOLD = HERE/'fixed_state_fold_v1'
os.environ.setdefault('ARC3_FIXED_STATE_FOLD', 'off')
assert os.environ['ARC3_FIXED_STATE_FOLD'] in ('off', 'A', 'B'), os.environ['ARC3_FIXED_STATE_FOLD']
if os.environ['ARC3_FIXED_STATE_FOLD'] != 'off':
    _head = os.environ['PYTHONPATH'].split(':')[0]   # the current outermost bootstrap directory
    os.environ.update(ARC3_FIXED_STATE_FOLD_RECEIPTS=str(HERE/'fold-startup'), ARC3_FOLD_CHAIN_BOOTSTRAP=os.path.join(_head, 'sitecustomize.py'))
    os.environ['PYTHONPATH'] = str(FOLD/'bootstrap')+':'+os.environ['PYTHONPATH']
# Stage 6/7 boundary timing journal (measure-first; byte-identical serving): 0 | 1. Outermost bootstrap link.
BT = HERE/'boundary_timing_v1'
os.environ.setdefault('ARC3_BOUNDARY_TIMING', '0')
assert os.environ['ARC3_BOUNDARY_TIMING'] in ('0', '1'), os.environ['ARC3_BOUNDARY_TIMING']
if os.environ['ARC3_BOUNDARY_TIMING'] == '1':
    _head = os.environ['PYTHONPATH'].split(':')[0]
    os.environ.update(ARC3_BOUNDARY_TIMING_RECEIPTS=str(HERE/'boundary-timing'), ARC3_BT_CHAIN_BOOTSTRAP=os.path.join(_head, 'sitecustomize.py'))
    os.environ['PYTHONPATH'] = str(BT/'bootstrap')+':'+os.environ['PYTHONPATH']
# Stage 6 per-owner settlement (settle a stopping lane on its own fence; others keep stepping): 0 | 1. Outermost link.
POS = HERE/'per_owner_settle_v1'
os.environ.setdefault('ARC3_PER_OWNER_SETTLE', '0')
assert os.environ['ARC3_PER_OWNER_SETTLE'] in ('0', '1'), os.environ['ARC3_PER_OWNER_SETTLE']
if os.environ['ARC3_PER_OWNER_SETTLE'] == '1':
    _head = os.environ['PYTHONPATH'].split(':')[0]
    os.environ.update(ARC3_PER_OWNER_SETTLE_RECEIPTS=str(HERE/'per-owner-settle'), ARC3_POS_CHAIN_BOOTSTRAP=os.path.join(_head, 'sitecustomize.py'))
    os.environ['PYTHONPATH'] = str(POS/'bootstrap')+':'+os.environ['PYTHONPATH']
# Stage 10 glue fusion (fused HC up-site GEMM, shape-keyed small GEMMs, GDN copy removal): 0 | all | parts. Outermost link.
GF = HERE/'glue_fusion_v1'
os.environ.setdefault('ARC3_GLUE_FUSION', '0')
assert os.environ['ARC3_GLUE_FUSION'] == '0' or os.environ['ARC3_GLUE_FUSION'] == 'all' or set(os.environ['ARC3_GLUE_FUSION'].split(',')) <= {'hc', 'gemm', 'gdn', 'hcf', 'splitk'}, os.environ['ARC3_GLUE_FUSION']
if os.environ['ARC3_GLUE_FUSION'] != '0':
    _head = os.environ['PYTHONPATH'].split(':')[0]
    os.environ['ARC3_GF_CHAIN_BOOTSTRAP'] = os.path.join(_head, 'sitecustomize.py')
    os.environ['PYTHONPATH'] = str(GF/'bootstrap')+':'+os.environ['PYTHONPATH']
# 2026-09-29 (user: adopt the MTP0 changes): mtp0x_v1 = fp8w dense (plain E4M3 layout, fp8w Triton GEMM with
# deterministic split-K; replaces the Marlin repack) and the candidate decode sampler (MRv2, every top_k <= 31).
# Default fp8w,sampler; ARC3_MTP0X=0 restores the v5.31 dense path and sampling. Outermost bootstrap link.
MX = HERE/'mtp0x_v1'
os.environ.setdefault('ARC3_MTP0X', 'fp8w,sampler')
assert os.environ['ARC3_MTP0X'] == '0' or set(os.environ['ARC3_MTP0X'].split(',')) <= {'fp8w', 'sampler'}, os.environ['ARC3_MTP0X']
if os.environ['ARC3_MTP0X'] != '0':
    _head = os.environ['PYTHONPATH'].split(':')[0]
    os.environ.update(ARC3_MTP0X_CHAIN_BOOTSTRAP=os.path.join(_head, 'sitecustomize.py'), ARC3_MTP0X_RECEIPTS=str(HERE/'mtp0x-startup'))
    os.environ['PYTHONPATH'] = str(MX/'bootstrap')+':'+os.environ['PYTHONPATH']
# 2026-09-27 (user): the adopted gameplay configuration is the launch default; an exported value still overrides.
# Observation pin on; QSA image rank q75 for observation and tool-output boards, q60 for recap boards, animation
# frames and other images (fallback q60); async eviction; rebase on; O(L) fast paths on, self-verification off.
# 2026-09-27 stage 15: system-prompt KV sharing (ARC3_PREFIX_SHARE), on since the 75-min validation run (18.78).
# 2026-09-27 evening (user): observation boards q75, every other image kind q50 (fallback q50).
for _k, _v in (('ARC3_PIN_OBSERVATION', '1'), ('ARC3_QSA_IMAGE_RANK', 'q50'),
               ('ARC3_QSA_IMAGE_RANK_BY_KIND', 'observation=q75,tool=q50,recap=q50,animation=q50,other=q50'),
               ('ARC3_EVICTION_MODE', 'async'), ('ARC3_REBASE', '1'), ('ARC3_OL_FASTPATH', '1'), ('ARC3_OL_FASTPATH_VERIFY', '0'),
               ('ARC3_PREFIX_SHARE', '1')):
    os.environ.setdefault(_k, _v)
# Compiled artifacts are traced with the weight layout of the selected tier: keep one AOT/compile cache per tier so a
# cached graph (BF16 weight as a graph input) is never loaded on top of Marlin-packed weights (stale-artifact failure 2026-09-19 18:36).
CACHE_TAG = ('tier-' + os.environ['ARC3_DENSE_FP8_TIER'] + '-lmh' + os.environ.get('ARC3_DENSE_FP8_LM_HEAD', '0') + '-bt' + os.environ['ARC3_BATCHED_TOKENS']
             + '-n' + os.environ['ARC3_CONCURRENCY'] + '-u' + os.environ['ARC3_GPU_MEMORY_UTILIZATION'] + '-pm' + os.environ.get('ARC3_STAGE4_PRIVATE_MOE', '1')
             + ('-c' + os.environ['ARC3_CONTEXT'] if os.environ['ARC3_CONTEXT'] != '40000' else '')
             + ('-ple' + os.environ['ARC3_PLE_DISCRIMINATOR'] if os.environ['ARC3_PLE_DISCRIMINATOR'] != 'off' else '')
             + ('-pledirect' if os.environ['ARC3_PLE_HANDOFF_IMPL'] == 'direct' else '')
             + ('-plehostpoll' if os.environ['ARC3_PLE_HANDOFF_IMPL'] == 'hostpoll' else '')
             + ('-fold' + os.environ['ARC3_FIXED_STATE_FOLD'] if os.environ['ARC3_FIXED_STATE_FOLD'] != 'off' else '')
             + ('-pos' if os.environ['ARC3_PER_OWNER_SETTLE'] == '1' else '')
             + ('-gf' + os.environ['ARC3_GLUE_FUSION'].replace(',', '') if os.environ['ARC3_GLUE_FUSION'] != '0' else '')
             + ('-kv' + os.environ['ARC3_QSA_KV_FORMAT'] if os.environ['ARC3_QSA_KV_FORMAT'] != 'nvfp4' else '')
             + ('-fake' + os.environ['ARC3_DENSE_FP8_FAKE'] if os.environ['ARC3_DENSE_FP8_FAKE'] != 'off' else ''))
if os.environ['ARC3_MTP0X'] != '0':
    CACHE_TAG += '-x' + os.environ['ARC3_MTP0X'].replace(',', '')   # fp8w: plain E4M3 weights, not Marlin-packed
os.environ['ARC3_DENSE_AOT_ROOT'] = str(HERE/('aot-cache-' + CACHE_TAG))
assert int(os.environ['ARC3_BATCHED_TOKENS']) in (4096, 8192), os.environ['ARC3_BATCHED_TOKENS']
import minimal_inference as m

TEMPLATE = ROOT/'runs/reasoning-continuation-tools-20260911-v1/prolong_chat_template_v529.jinja'
assert hashlib.sha256(TEMPLATE.read_bytes()).hexdigest() == '86cef302280cac60c0148ada70ecd30d5084e5c0c4d65ace8755a855b5176a3d'
base_server_command = m.server_command
def server_command():
    argv = base_server_command()
    i = argv.index('--worker-extension-cls') + 1
    assert argv[i] == 'worker_diagnostics.WorkerDiagnostics'
    argv[i] = 'arc3_host_diagnostics.WorkerDiagnostics'
    argv[argv.index('--chat-template')+1] = str(TEMPLATE)
    BASE_SIZES = [1, 2, 4, 8, 16, 24, 25, 30, 32]
    extra = [int(x) for x in os.environ.get('ARC3_STAGE4_CAPTURE_EXTRA', '').split(',') if x]
    sizes = sorted(set(BASE_SIZES + extra))
    assert config['capture_sizes'] in (BASE_SIZES, sizes), config['capture_sizes']   # server_command() runs twice per launch
    config['capture_sizes'] = sizes   # identity must record the effective list (clients compare it)
    assert max(sizes) <= int(os.environ['ARC3_GRAPH_SIZE']) and int(os.environ['ARC3_CONCURRENCY']) in sizes, (sizes, os.environ['ARC3_GRAPH_SIZE'], os.environ['ARC3_CONCURRENCY'])
    assert '--cudagraph-capture-sizes' not in argv
    argv.extend(['--cudagraph-capture-sizes', *map(str, sizes)])
    assert '--compilation-config' not in argv
    argv.extend(['--compilation-config', json.dumps({'cache_dir':str(HERE/('compiled-cache-' + CACHE_TAG))})])
    return argv
m.server_command = server_command

action = sys.argv[1]
assert action in ('start', 'status'), 'Selected launcher never stops a predecessor'
if action == 'stop-baseline':
    identity = json.loads(m.IDENTITY.read_text())
    expected = json.loads((HERE/'expected-baseline.json').read_text())
    assert identity['root'] == expected['root'], 'Current server is not the pinned baseline'
    with urlopen('http://127.0.0.1:1234/metrics', timeout=10) as response:
        metrics = response.read().decode()
    observed = set()
    required = {'vllm:num_requests_running', 'vllm:num_requests_waiting'}
    for line in metrics.splitlines():
        name = line.split('{', 1)[0].split(' ', 1)[0]
        if name in required:
            observed.add(name)
            assert float(line.rsplit(' ',1)[1]) == 0, line
    assert observed == required, 'Missing idle engine metrics'
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
        phase=config['label'], prefix_caching=False,
        eager_correctness_candidate=bool(config['eager']), optimization_config=config)
    identity['prolong_template'] = dict(path=str(TEMPLATE), continuation_sha256=hashlib.sha256(TEMPLATE.read_bytes()).hexdigest())
    m.save(m.IDENTITY, identity)
    m.save(HERE/'candidate-server.json', identity)
elif action == 'status':
    m.status()
else:
    raise ValueError(action)
