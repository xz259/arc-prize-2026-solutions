"""k_launch_scaling_s2i.py (2026-10-24, mk_klaunch.py; the Kaggle port): launch_scaling_s2i.py 4fb44cbd with every check that
would stop the launch logging instead ('[k_launch] ...' on stderr); the composition is unchanged. mk_klaunch.py lists the checks.

Instance launcher (2026-10-11, corpus v7 generation: two servers, one per GPU): launch_scaling_s2.py (776fc10c) with
per-instance state. ARC3_S2_INSTANCE (name), ARC3_S2_GPU (physical GPU index) and ARC3_PORT are required; the identity
file, the service and engine audit logs and the launch receipts live in S2/instances/<name>/, never the production
identity (ROOT/server-identity.json); the server process sees only its GPU. Overlay, AOT/compile caches and the per-PID
stage-2 receipts are shared with launch_scaling_s2.py. Actions: start, status, stop (this instance's processes only).
Production files (minimal_inference.py, runs/dense_fp8_v1) are read, never modified.

Stage-2 copy of runs/dense_fp8_v1/launch_scaling.py (sha f75af54fd789): identical composition, plus MTP > 0
(ARC3_S2_MTP_TOKENS, default 3) and the stage2_v1 overlay link (ARC3_STAGE2 parts) as the outermost bootstrap link.
Production code is read in place and verified by its overlay manifest; caches, receipts and audit logs go to this
directory. Verify CUDA graphs take token counts from ARC3_S2_CAPTURE. Run via runtime_python.sh."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from urllib.request import urlopen
import importlib.metadata
import time


def _klog(msg):   # k (the Kaggle port): a check that would stop the launch logs, and the launch continues
    sys.stderr.write('[k_launch %s] %s\n' % (time.strftime('%H:%M:%S', time.gmtime()), msg))
    sys.stderr.flush()

ROOT = Path('/workspace/arc3-inference')
HERE = Path('/workspace/arc3-inference/runs/dense_fp8_v1')   # stage-2 copy: production overlay, read-only
S2D = Path(__file__).resolve().parent   # stage 2: overlay link, caches, receipts, audits
INSTANCE, GPU = os.environ['ARC3_S2_INSTANCE'], os.environ['ARC3_S2_GPU']
assert INSTANCE.replace('-', '').replace('_', '').isalnum() and GPU in ('0', '1'), (INSTANCE, GPU)
assert os.environ.get('ARC3_PORT', '').isdigit(), 'ARC3_PORT must be set for an instance launch'
INST = S2D/'instances'/INSTANCE   # identity, audits, launch receipts of this instance
INST.mkdir(parents=True, exist_ok=True)
if socket.gethostname() != '2a159825c76b':   # k: the pod's host and boot pins log only
    _klog('host %s, not the pod (the host pin logs only)' % socket.gethostname())
try:
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != 'e27af87d-a1a1-40d4-99c6-f972803e152b':
        _klog('boot id differs from the pod\'s (logs only)')
except OSError as _kexc:
    _klog('boot id unreadable: %r' % _kexc)
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
try:   # k: the overlay manifest logs the files that differ or are missing
    manifest = json.loads((HERE/'overlay-manifest.json').read_text())
except Exception as _kexc:  # noqa: BLE001
    manifest = {}
    _klog('overlay manifest unreadable: %r' % _kexc)
_kbad = []
for name, expected in manifest.items():
    try:
        if hashlib.sha256((HERE/name).read_bytes()).hexdigest() != expected:
            _kbad.append(name)
    except OSError as _kexc:
        _kbad.append('%s (%s)' % (name, _kexc.__class__.__name__))
_klog('overlay manifest: %d files, %d differ or missing%s' % (len(manifest), len(_kbad), (': ' + ', '.join(_kbad[:20])) if _kbad else ''))
assert os.environ.get('ARC3_STAGE_KV_FORMAT', 'nvfp4') in ('bf16', 'fp8_e4m3', 'fp8_e5m2', 'nvfp4'), os.environ.get('ARC3_STAGE_KV_FORMAT')
os.environ.update(
    ARC3_ROOT=str(ROOT), ARC3_CONTEXT=os.environ.get('ARC3_STAGE4_CONTEXT', '40000'), ARC3_CONCURRENCY=os.environ.get('ARC3_STAGE4_MAX_SEQS', '30'),
    ARC3_KV_GIB=os.environ.get('ARC3_STAGE4_KV_GIB', '0'), ARC3_GPU_MEMORY_UTILIZATION=os.environ.get('ARC3_STAGE4_UTIL', '0.95'), ARC3_GRAPH_SIZE=os.environ.get('ARC3_STAGE4_GRAPH_SIZE', '32'),
    ARC3_BATCHED_TOKENS=os.environ.get('ARC3_STAGE2_BATCHED_TOKENS', '8192'), ARC3_MTP_TOKENS=os.environ.get('ARC3_S2_MTP_TOKENS', '3'), ARC3_PREFIX_CACHE='0',  # stage 2: 4096 candidate
    ARC3_EAGER=str(config['eager']), ARC3_QSA_MODE='radix', ARC3_QSA_KV_FORMAT=os.environ.get('ARC3_STAGE_KV_FORMAT', 'nvfp4'),  # stage 11: KV format NLL comparison
    ARC3_QSA_FP8_READER='baseline', ARC3_MOE_FINALIZE='unfused',
    ARC3_TOPK_THREADS='512', ARC3_TOPK_CACHE_ROW='1', ARC3_QSA_PREFILL_PROFILE='stock',
    ARC3_SKIP_REDUNDANT_TOPK='1', ARC3_MTP_MIN_P='1', ARC3_MEMORY_DIAGNOSTICS='1',
    VLLM_USE_DEEP_GEMM='1', VLLM_MOE_USE_DEEP_GEMM='0',
    VLLM_DISABLE_SHARED_EXPERTS_STREAM=os.environ.get('ARC3_S2_DISABLE_SHARED_EXPERTS_STREAM', '0'), VLLM_SHARED_EXPERTS_STREAM_TOKEN_THRESHOLD='256',  # milestone 2 diagnostic (06:40)
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
    ARC3_RESUME_PROFILE_PATH=str(S2D/'resume-profile'),
    ARC3_GAMEPLAY_PROFILE=str(config.get('gameplay_profile',0)),
    ARC3_GAMEPLAY_PROFILE_PATH=str(S2D/'gameplay-server-profile'),
    ARC3_PERSISTENT_ENGINE_AUDIT=str(INST/'engine-events.jsonl'),
    ARC3_PERSISTENT_SERVICE_AUDIT=str(INST/'service-events.jsonl'),
)
# 2026-10-16 (user: "make it a Boolean so it can be switched on and off"; "keep the log" = default on): ARC3_ENGINE_AUDIT=0 (or
# false / off / no) turns the engine audit log off: its path is not exported, so production's engine_patch._audit returns at once
# and stage2_v1/agefix skips the eviction wrapper's digest work. 1 (default) keeps engine-events.jsonl.
if os.environ.get('ARC3_ENGINE_AUDIT', '1').strip().lower() in ('0', 'false', 'off', 'no'):
    os.environ.pop('ARC3_PERSISTENT_ENGINE_AUDIT', None)
# k (the user 10-24 via IE: diagnostics off in a competition rerun): ARC3_K_SERVICE_AUDIT=0 leaves the service audit's path unset, so
# session_service writes no service-events.jsonl (audit_path None: no writes); unset or 1 = on, as launch_scaling_s2i.py
if os.environ.get('ARC3_K_SERVICE_AUDIT', '1').strip().lower() in ('0', 'false', 'off', 'no'):
    os.environ.pop('ARC3_PERSISTENT_SERVICE_AUDIT', None)
# v7.5 C11 / C12 (IE; the user 10-23, review Part 2): the gameplay and resume profilers through an environment override of the
# read-only optimization-config.json: ARC3_S2_GAMEPLAY_PROFILE / ARC3_S2_RESUME_PROFILE = 0 | 1 (unset: the config's value)
for _v75_k, _v75_v in (('ARC3_S2_GAMEPLAY_PROFILE', 'ARC3_GAMEPLAY_PROFILE'), ('ARC3_S2_RESUME_PROFILE', 'ARC3_RESUME_PROFILE')):
    if os.environ.get(_v75_k) is not None:
        if os.environ[_v75_k] not in ('0', '1'):
            raise SystemExit('%s must be 0 or 1, not %r' % (_v75_k, os.environ[_v75_k]))
        os.environ[_v75_v] = os.environ[_v75_k]
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
    ARC3_PRIVATE_MOE_RECEIPTS=str(S2D/'private-startup'))
os.environ['PYTHONPATH'] = str(ADAPTER/'bootstrap')+':'+str(ADAPTER)+':'+os.environ['PYTHONPATH']
# Common measured projection tuning; explicitly compose MoE startup.
os.environ.update(ARC3_DENSE_EXPERIMENT='1',
    ARC3_DENSE_RECEIPTS=str(S2D/'dense-startup'),
    ARC3_DENSE_AOT_ROOT=str(S2D/'aot-cache'),
    ARC3_DENSE_CHAIN_BOOTSTRAP=str(ADAPTER/'bootstrap'/'sitecustomize.py'))
os.environ['PYTHONPATH'] = str(HERE/'dense_projection_adapter_v1'/'bootstrap')+':'+os.environ['PYTHONPATH']
os.environ['ARC3_PLE_HANDOFF_RECEIPTS'] = str(S2D/'ple-handoff')
os.environ['PYTHONPATH'] = str(HERE/'combined-bootstrap')+':'+os.environ['PYTHONPATH']
# Dense FP8 weight-only (stage 1 of THROUGHPUT_MEMORY_PLAN_2026-09-19). Tier is inherited from the launching shell:
# off (byte-identical composition to qsa_frequency_eviction_v1), inventory (record only), A, AB2, AB2C; lm_head via ARC3_DENSE_FP8_LM_HEAD=1.
FP8 = HERE/'dense_fp8_adapter_v1'
os.environ.setdefault('ARC3_DENSE_FP8_TIER', 'off')
assert os.environ['ARC3_DENSE_FP8_TIER'] in ('off', 'inventory', 'A', 'AB2', 'AB2C', 'A_qkvz', 'A_out', 'A_qkv', 'A_in'), os.environ['ARC3_DENSE_FP8_TIER']
os.environ['ARC3_DENSE_FP8_FAKE'] = os.environ.get('ARC3_STAGE_FP8_FAKE', 'off'); assert os.environ['ARC3_DENSE_FP8_FAKE'] in ('off', 'e4m3', 'e5m2')  # stage 13
os.environ.update(ARC3_DENSE_FP8_RECEIPTS=str(S2D/'dense-fp8-startup'),
    ARC3_DENSE_FP8_CHAIN_BOOTSTRAP=str(HERE/'combined-bootstrap'/'sitecustomize.py'))
os.environ['PYTHONPATH'] = str(FP8/'bootstrap')+':'+os.environ['PYTHONPATH']
# Stage 5 handoff implementation: copy (served Sept 16 acknowledgement fix) | direct (ple_direct_handoff_v1: model-thread
# staging + polled flag, no request thread / acknowledgement / per-step ZMQ). Receipts + cache tag on direct.
os.environ.setdefault('ARC3_PLE_HANDOFF_IMPL', 'copy')
assert os.environ['ARC3_PLE_HANDOFF_IMPL'] in ('copy', 'direct', 'hostpoll'), os.environ['ARC3_PLE_HANDOFF_IMPL']
if os.environ['ARC3_PLE_HANDOFF_IMPL'] == 'direct':
    os.environ['ARC3_PLE_DIRECT_RECEIPTS'] = str(S2D/'ple-direct-startup')
if os.environ['ARC3_PLE_HANDOFF_IMPL'] == 'hostpoll':
    os.environ['ARC3_PLE_HOSTPOLL_RECEIPTS'] = str(S2D/'ple-hostpoll-startup')
    os.environ['ARC3_PLE_HOSTPOLL_BUILD'] = str(HERE/'ple_hostpoll_v1'/'build')   # precompiled polling kernel
# Stage 5 PLE critical-path discriminator (THROUGHPUT-ONLY diagnostic): off | lookup | handoff; chains the dense-fp8 bootstrap.
PLED = HERE/'ple_discriminator_v1'
os.environ.setdefault('ARC3_PLE_DISCRIMINATOR', 'off')
assert os.environ['ARC3_PLE_DISCRIMINATOR'] in ('off', 'lookup', 'handoff', 'nowait', 'timing', 'nogpu'), os.environ['ARC3_PLE_DISCRIMINATOR']
if os.environ['ARC3_PLE_DISCRIMINATOR'] != 'off':
    os.environ.update(ARC3_PLE_DISCRIMINATOR_RECEIPTS=str(S2D/'ple-discriminator-startup'),
        ARC3_PLE_DISC_CHAIN_BOOTSTRAP=str(FP8/'bootstrap'/'sitecustomize.py'))
    os.environ['PYTHONPATH'] = str(PLED/'bootstrap')+':'+os.environ['PYTHONPATH']
# Stage 3 fixed-state fold (ring group + PLE state share one block per request): off | A. Outermost bootstrap link.
FOLD = HERE/'fixed_state_fold_v1'
os.environ.setdefault('ARC3_FIXED_STATE_FOLD', 'off')
assert os.environ['ARC3_FIXED_STATE_FOLD'] in ('off', 'A', 'B'), os.environ['ARC3_FIXED_STATE_FOLD']
if os.environ['ARC3_FIXED_STATE_FOLD'] != 'off':
    _head = os.environ['PYTHONPATH'].split(':')[0]   # the current outermost bootstrap directory
    os.environ.update(ARC3_FIXED_STATE_FOLD_RECEIPTS=str(S2D/'fold-startup'), ARC3_FOLD_CHAIN_BOOTSTRAP=os.path.join(_head, 'sitecustomize.py'))
    os.environ['PYTHONPATH'] = str(FOLD/'bootstrap')+':'+os.environ['PYTHONPATH']
# Stage 6/7 boundary timing journal (measure-first; byte-identical serving): 0 | 1. Outermost bootstrap link.
BT = HERE/'boundary_timing_v1'
os.environ.setdefault('ARC3_BOUNDARY_TIMING', '0')
assert os.environ['ARC3_BOUNDARY_TIMING'] in ('0', '1'), os.environ['ARC3_BOUNDARY_TIMING']
if os.environ['ARC3_BOUNDARY_TIMING'] == '1':
    _head = os.environ['PYTHONPATH'].split(':')[0]
    os.environ.update(ARC3_BOUNDARY_TIMING_RECEIPTS=str(S2D/'boundary-timing'), ARC3_BT_CHAIN_BOOTSTRAP=os.path.join(_head, 'sitecustomize.py'))
    os.environ['PYTHONPATH'] = str(BT/'bootstrap')+':'+os.environ['PYTHONPATH']
# Stage 6 per-owner settlement (settle a stopping lane on its own fence; others keep stepping): 0 | 1. Outermost link.
POS = HERE/'per_owner_settle_v1'
os.environ.setdefault('ARC3_PER_OWNER_SETTLE', '0')
assert os.environ['ARC3_PER_OWNER_SETTLE'] in ('0', '1'), os.environ['ARC3_PER_OWNER_SETTLE']
if os.environ['ARC3_PER_OWNER_SETTLE'] == '1':
    _head = os.environ['PYTHONPATH'].split(':')[0]
    os.environ.update(ARC3_PER_OWNER_SETTLE_RECEIPTS=str(S2D/'per-owner-settle'), ARC3_POS_CHAIN_BOOTSTRAP=os.path.join(_head, 'sitecustomize.py'))
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
    os.environ.update(ARC3_MTP0X_CHAIN_BOOTSTRAP=os.path.join(_head, 'sitecustomize.py'), ARC3_MTP0X_RECEIPTS=str(S2D/'mtp0x-startup'))
    os.environ['PYTHONPATH'] = str(MX/'bootstrap')+':'+os.environ['PYTHONPATH']
# Stage 2 (2026-09-29): stage2_v1 = MTP > 0 on the production composition (guards, rollback, verify path). Outermost.
S2 = S2D/'stage2_v1'
os.environ.setdefault('ARC3_STAGE2', 'guards')
assert os.environ['ARC3_STAGE2'] == '0' or set(os.environ['ARC3_STAGE2'].split(',')) <= {'guards', 'rollback', 'wy', 'vsampler', 'qsaprofile', 'pagepad', 'ctxfree', 'topkshare', 'pleconv', 'persist', 'draftqsa', 'schedfast', 'mmstrip', 'overlap', 'draftgraph', 'attnprof', 'resumefast', 'deadlinefix', 'topkfix', 'pwgraph', 'embedhost', 'splitfix', 'b1inc', 'drafthead', 'loadskip', 'marlindet', 'kvfq', 'kvevict', 'headonly', 'kvrot', 'idxfq', 'seloverlap', 'ckptfetch', 'draftwin', 'rope2m', 'rebaseobs', 'agestat', 'agecap', 'hollowgroup', 'sessmem', 'promptmove', 'ctxtail', 'foldc', 'padbudget', 'replayfx', 'dflashtaps', 'precmode', 'kvwin', 'poolpress', 'stallguard', 'kvring', 'prestage', 'fqfast', 'stepev', 'draftring', 'agefix', 'sumpin', 'rebase74', 'mmonly', 'kvq4', 'eprune'}, os.environ['ARC3_STAGE2']
if os.environ['ARC3_STAGE2'] != '0':
    _head = os.environ['PYTHONPATH'].split(':')[0]
    os.environ.update(ARC3_S2_CHAIN_BOOTSTRAP=os.path.join(_head, 'sitecustomize.py'), ARC3_S2_RECEIPTS=str(S2D/'receipts'))
    os.environ['PYTHONPATH'] = str(S2/'bootstrap')+':'+str(S2D)+':'+os.environ['PYTHONPATH']
# Stage-2 memory work: allocator settings (runtime_python.sh exports expandable_segments:False).
if os.environ.get('ARC3_S2_ALLOC_CONF'):
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = os.environ['ARC3_S2_ALLOC_CONF']
# 2026-09-27 (user): the adopted gameplay configuration is the launch default; an exported value still overrides.
# Observation pin on; QSA image rank q75 for observation and tool-output boards, q60 for recap boards, animation
# frames and other images (fallback q60); async eviction; rebase on; O(L) fast paths on, self-verification off.
# 2026-09-27 stage 15: system-prompt KV sharing (ARC3_PREFIX_SHARE), on since the 75-min validation run (18.78).
# 2026-09-27 evening (user): observation boards q75, every other image kind q50 (fallback q50).
# 2026-10-03 22:50 (user): hollow rule at 0.75 (a turn block goes whole once 75% of its content is evicted; production 0.8).
for _k, _v in (('ARC3_PIN_OBSERVATION', '1'), ('ARC3_QSA_IMAGE_RANK', 'q50'),
               ('ARC3_QSA_IMAGE_RANK_BY_KIND', 'observation=q75,tool=q50,recap=q50,animation=q50,other=q50'),
               ('ARC3_EVICTION_MODE', 'async'), ('ARC3_REBASE', '1'), ('ARC3_OL_FASTPATH', '1'), ('ARC3_OL_FASTPATH_VERIFY', '0'),
               ('ARC3_PREFIX_SHARE', '1'), ('ARC3_HOLLOW_FRACTION', '0.75')):
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
# 2026-10-10 22:30 (job 403c): the dflashtaps capture buffer [n_taps, rows, hidden] is traced into the AOT graph, so a
# non-default ARC3_S2_TAPS needs its own cache (a 16-tap export loaded the 9-tap artifact: size 16 != 9); default unchanged
if 'dflashtaps' in os.environ['ARC3_STAGE2'] and os.environ.get('ARC3_S2_TAPS', '').replace(' ', '') not in ('', '2,4,12,13,24,35,36,44,46'):
    CACHE_TAG += '-t' + hashlib.sha256(os.environ['ARC3_S2_TAPS'].replace(' ', '').encode()).hexdigest()[:8]
if os.environ.get('ARC3_S2_GDN_STATE'):  # scaled16 (10-15): the GDN page grows (row scales), so its own compile / AOT cache
    CACHE_TAG += '-gs' + os.environ['ARC3_S2_GDN_STATE']
if 'eprune' in os.environ['ARC3_STAGE2'].split(','):   # 2026-10-26: one compile / AOT cache per prune set (the file's bytes)
    _ep = os.environ.get('ARC3_S2_EPRUNE_FILE') or 'eprune_k16.json'   # eprune's default: the bottom-16 set
    _ep = str(S2D/'stage2_v1'/_ep) if os.sep not in _ep else _ep
    CACHE_TAG += '-ep' + hashlib.sha256(Path(_ep).read_bytes()).hexdigest()[:8]
if os.environ.get('ARC3_S2_PRECMODE'):   # precision audit (10-15): one compile / AOT cache per numerics mode set
    CACHE_TAG += '-prec' + os.environ['ARC3_S2_PRECMODE'].replace(',', '')
_HOST_ONLY = {'prestage', 'fqfast', 'stepev', 'agefix', 'sumpin'}   # 2026-10-17: + sumpin (API server); 2026-10-15: host-side parts (no traced code) are not part of the compile / AOT cache tag
_TAG_PARTS = ','.join(x for x in os.environ['ARC3_STAGE2'].split(',') if x not in _HOST_ONLY)
_S2TAG = _TAG_PARTS.replace(',', '') if os.environ['ARC3_STAGE2'] != '0' else ''
if len('compiled-cache-' + CACHE_TAG + '-s2mtp' + os.environ['ARC3_MTP_TOKENS'] + '-' + _S2TAG) > 240:   # milestone 2 (07:05): path component limit (255)
    _S2TAG = 'p' + hashlib.sha256(_TAG_PARTS.encode()).hexdigest()[:16]   # long part lists: hashed (exact list in the identity)
CACHE_TAG += '-s2mtp' + os.environ['ARC3_MTP_TOKENS'] + ('-' + _S2TAG if _S2TAG else '')
os.environ['ARC3_DENSE_AOT_ROOT'] = str(S2D/('aot-cache-' + CACHE_TAG))
assert int(os.environ['ARC3_BATCHED_TOKENS']) in (4096, 8192), os.environ['ARC3_BATCHED_TOKENS']
if os.environ.get('ARC3_PY_PREPEND'):   # k: the file tracer first on the server processes' path (unset: nothing)
    os.environ['PYTHONPATH'] = os.environ['ARC3_PY_PREPEND'] + ':' + os.environ['PYTHONPATH']
import minimal_inference as m
m.IDENTITY = INST/'server-identity.json'   # this instance's server only (start / status / stop)
assert m.PORT == int(os.environ['ARC3_PORT']), (m.PORT, os.environ['ARC3_PORT'])
_stock_server_environment = m.server_environment
def _instance_server_environment():
    env, updates = _stock_server_environment()
    assert env['CUDA_VISIBLE_DEVICES'] == updates['CUDA_VISIBLE_DEVICES'] == '0', updates['CUDA_VISIBLE_DEVICES']
    env['CUDA_VISIBLE_DEVICES'] = updates['CUDA_VISIBLE_DEVICES'] = GPU   # the record shows the effective device
    return env, updates
m.server_environment = _instance_server_environment
_stock_inspect_model = m.inspect_model
def _variant_inspect_model():
    tv = m.MODEL/'ARC3_TARGET_VIEW.json'
    if not tv.exists():
        return _stock_inspect_model()
    raw = (m.MODEL/'config.json').read_bytes()
    q = json.loads(raw).get('quantization_config', {})
    index = json.loads((m.MODEL/'model.safetensors.index.json').read_text())
    shards = sorted(set(index['weight_map'].values()))
    return {'model': str(m.MODEL), 'revision': None, 'config_sha256': hashlib.sha256(raw).hexdigest(),
            'quantization': q.get('quant_algo') or q.get('quant_method'), 'shards': len(shards),
            'shard_bytes': sum((m.MODEL/p).stat().st_size for p in shards), 'target_view': json.loads(tv.read_text())}
m.inspect_model = _variant_inspect_model

TEMPLATE = ROOT/'runs/reasoning-continuation-tools-20260911-v1/prolong_chat_template_v529.jinja'
try:   # k: the template's identity logs only
    if hashlib.sha256(TEMPLATE.read_bytes()).hexdigest() != '86cef302280cac60c0148ada70ecd30d5084e5c0c4d65ace8755a855b5176a3d':
        _klog('chat template %s differs from its pinned sha256' % TEMPLATE)
except OSError as _kexc:
    _klog('chat template %s unreadable: %r' % (TEMPLATE, _kexc))
base_server_command = m.server_command
def server_command():
    argv = base_server_command()
    i = argv.index('--worker-extension-cls') + 1
    assert argv[i] == 'worker_diagnostics.WorkerDiagnostics'
    argv[i] = 'arc3_host_diagnostics.WorkerDiagnostics'
    argv[argv.index('--chat-template')+1] = str(TEMPLATE)
    BASE_SIZES = [1, 2, 4, 8, 16, 24, 25, 30, 32]
    if int(os.environ['ARC3_MTP_TOKENS']):
        # Verify graphs: token counts (MRv2 rounds a uniform decode batch up to a multiple of 1 + k).
        sizes = sorted({int(x) for x in os.environ['ARC3_S2_CAPTURE'].split(',') if x})
        assert config['capture_sizes'] in (BASE_SIZES, sizes), config['capture_sizes']   # server_command() runs twice per launch
        config['capture_sizes'] = sizes
        assert max(sizes) <= int(os.environ['ARC3_GRAPH_SIZE']), (sizes, os.environ['ARC3_GRAPH_SIZE'])
    else:
        extra = [int(x) for x in os.environ.get('ARC3_STAGE4_CAPTURE_EXTRA', '').split(',') if x]
        sizes = sorted(set(BASE_SIZES + extra))
        assert config['capture_sizes'] in (BASE_SIZES, sizes), config['capture_sizes']   # server_command() runs twice per launch
        config['capture_sizes'] = sizes   # identity must record the effective list (clients compare it)
        assert max(sizes) <= int(os.environ['ARC3_GRAPH_SIZE']) and int(os.environ['ARC3_CONCURRENCY']) in sizes, (sizes, os.environ['ARC3_GRAPH_SIZE'], os.environ['ARC3_CONCURRENCY'])
    assert '--cudagraph-capture-sizes' not in argv
    argv.extend(['--cudagraph-capture-sizes', *map(str, sizes)])
    if int(os.environ['ARC3_MTP_TOKENS']):
        assert '--mamba-ssm-cache-dtype' not in argv
        argv.extend(['--mamba-ssm-cache-dtype', os.environ.get('ARC3_S2_SSM_DTYPE', 'float16')])   # production's effective GDN state dtype; precision work 10-15: ARC3_S2_SSM_DTYPE=float16
    if os.environ.get('ARC3_S2_DRAFT_MODEL'):   # milestone 2 A10: drafter-only folder (same tensors, verified)
        _i = argv.index('--speculative-config') + 1
        _spec = json.loads(argv[_i])
        assert 'model' not in _spec, _spec
        try:   # k: the drafter view's check logs only
            _view = json.loads((Path(os.environ['ARC3_S2_DRAFT_MODEL'])/'ARC3_DRAFT_VIEW.json').read_text())
            _tv = m.MODEL/'ARC3_TARGET_VIEW.json'   # a config-only target variant names its base checkpoint
            _base = json.loads(_tv.read_text())['base_source'] if _tv.exists() else str(m.MODEL)
            if not (_view.get('verified') and _view['source'] == _base):
                _klog('drafter view: source %r, target base %r, verified %r' % (_view.get('source'), _base, bool(_view.get('verified'))))
        except Exception as _kexc:  # noqa: BLE001
            _klog('drafter view unreadable: %r' % _kexc)
        _spec['model'] = os.environ['ARC3_S2_DRAFT_MODEL']
        if os.environ.get('ARC3_S2_DRAFT_QUANTIZATION'):   # MTP drafts inherit the target's method otherwise (speculative.py:809)
            _spec['quantization'] = os.environ['ARC3_S2_DRAFT_QUANTIZATION']
        argv[_i] = json.dumps(_spec)
    if os.environ.get('ARC3_S2_QUANTIZATION'):   # milestone 2 B1b: e.g. inc for the Intel AutoRound experts
        argv[argv.index('--quantization') + 1] = os.environ['ARC3_S2_QUANTIZATION']
    if os.environ.get('ARC3_S2_MOE_BACKEND'):   # milestone 2 diagnostic (06:45): explicit MoE backend, e.g. humming
        assert '--moe-backend' not in argv
        argv += ['--moe-backend', os.environ['ARC3_S2_MOE_BACKEND']]
    if os.environ.get('ARC3_S2_LONG_PREFILL'):                    # 2026-10-04 promptmove gate panel: first chunk = system turn
        assert '--long-prefill-token-threshold' not in argv
        argv.extend(['--long-prefill-token-threshold', str(int(os.environ['ARC3_S2_LONG_PREFILL']))])
    if os.environ.get('ARC3_S2_WATERMARK'):                       # overcommit (10-14): blocks kept free from waiting (resume) admissions
        assert '--watermark' not in argv
        argv.extend(['--watermark', str(float(os.environ['ARC3_S2_WATERMARK']))])
    if os.environ.get('ARC3_S2_MM_CACHE_GB'):                     # 10-23 flag 3 (V67_DESIGN 282 / 285): 0 = no processor cache, no P0/P1 drift
        assert '--mm-processor-cache-gb' not in argv
        argv.extend(['--mm-processor-cache-gb', str(float(os.environ['ARC3_S2_MM_CACHE_GB']))])
    assert '--compilation-config' not in argv
    argv.extend(['--compilation-config', json.dumps({'cache_dir':str(S2D/('compiled-cache-' + CACHE_TAG))})])
    return argv
m.server_command = server_command



def _k_start():
    """k: minimal_inference.start (6cb3de51) with each check logging instead of raising; the launch itself is unchanged."""
    if m.port_open():
        _klog('port %d is occupied (continuing)' % m.PORT)
    if m.IDENTITY.exists():
        try:
            if any(m.matches(p) for p in m.owned_snapshot(json.loads(m.IDENTITY.read_text()))):
                _klog('prior owned server processes still live (continuing)')
        except Exception as exc:  # noqa: BLE001
            _klog('prior identity unreadable: %r' % exc)
    try:
        model = m.inspect_model()
    except Exception as exc:  # noqa: BLE001
        model = dict(model=str(m.MODEL), error=repr(exc)[:300])
        _klog('inspect_model: %r' % exc)
    try:
        if importlib.metadata.version('vllm') != m.VLLM_VERSION:
            _klog('vllm %s, not the pinned %s' % (importlib.metadata.version('vllm'), m.VLLM_VERSION))
    except Exception as exc:  # noqa: BLE001
        _klog('vllm version unreadable: %r' % exc)
    site = m.ROOT/'runtime/base/usr/local/lib/python3.12/dist-packages'

    def rows_ok(name, d):
        bad = []
        for row in d.get('files', []):
            try:
                if m.sha((site/row['path']).read_bytes()) != row['patched_sha256']:
                    bad.append(row['path'])
            except OSError:
                bad.append(row['path'] + ' (missing)')
        if bad:
            _klog('%s: %d of %d patched files differ: %s' % (name, len(bad), len(d.get('files', [])), ', '.join(bad[:10])))

    def load(rel):
        p = m.ROOT/rel
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text())
        except Exception as exc:  # noqa: BLE001
            _klog('%s unreadable: %r' % (rel, exc))
            return None

    def helper(rel, want, what):
        try:
            if m.sha((m.ROOT/rel).read_bytes()) != want:
                _klog('%s identity mismatch: %s' % (what, rel))
        except OSError as exc:
            _klog('%s unreadable: %s %r' % (what, rel, exc))

    patch = load('patches/manifest.json')
    if patch is None:
        _klog('patches/manifest.json missing')
    else:
        rows_ok('runtime patch', patch)
    diagnostic_patch = load('patches/determinism-manifest.json')
    if diagnostic_patch is not None:
        rows_ok('determinism patch', diagnostic_patch)
        helper('arc3_determinism.py', diagnostic_patch.get('helper_sha256'), 'determinism helper')
        if os.environ.get('ARC3_QSA_PREFILL_PROFILE', 'stock') != 'stock' and diagnostic_patch.get('patch') != 'qsa-moe-diagnostics-v2':
            _klog('optional sparse-prefill hook is not installed')
    elif m.QSA_MODE != 'native' or m.MOE_FINALIZE != 'fused':
        _klog('requested determinism hooks are not installed')
    radix_build = None
    if m.QSA_MODE == 'radix':
        radix_build = load('kernels/build/latest.json')
        if radix_build is None:
            _klog('kernels/build/latest.json missing')
        else:
            for name, expected in (radix_build.get('sources') or {}).items():
                helper('kernels/' + name, expected, 'radix selector source')
            library_path = Path(radix_build.get('library', ''))
            if not library_path.is_absolute():
                library_path = m.ROOT/library_path
            try:
                if m.sha(library_path.read_bytes()) != radix_build.get('library_sha256'):
                    _klog('radix selector library identity mismatch')
            except OSError as exc:
                _klog('radix selector library unreadable: %r' % exc)
            try:
                radix_build['wrapper_sha256'] = m.sha((m.ROOT/'arc3_topk_radix.py').read_bytes())
            except OSError as exc:
                _klog('arc3_topk_radix.py unreadable: %r' % exc)
    qsa_quant_patch = load('patches/qsa-quant-manifest.json')
    if qsa_quant_patch is not None:
        rows_ok('QSA cache patch', qsa_quant_patch)
        for name, expected in (qsa_quant_patch.get('helpers') or {'arc3_qsa_quant.py': qsa_quant_patch.get('helper_sha256')}).items():
            helper(name, expected, 'QSA cache helper')
        if m.QSA_KV_FORMAT != 'bf16' and m.QSA_KV_FORMAT not in qsa_quant_patch.get('formats', []):
            _klog('QSA format %s is not in the installed patch manifest' % m.QSA_KV_FORMAT)
    elif m.QSA_KV_FORMAT != 'bf16':
        _klog('quantized QSA storage requested without the scoped runtime patch')
    if m.FP8_READER != 'baseline' and (qsa_quant_patch is None or m.FP8_READER not in qsa_quant_patch.get('fp8_readers', [])):
        _klog('FP8 reader %s is not in the installed patch manifest' % m.FP8_READER)
    min_p_patch = load('patches/mtp-min-p-manifest.json')
    if min_p_patch is not None:
        rows_ok('MTP min-p patch', min_p_patch)
    elif m.MTP_MIN_P:
        _klog('MTP min-p requested but the patch is not installed')
    stamp = time.strftime('%Y%m%d-%H%M%S')
    for n in range(100):
        run = m.ROOT/'logs'/(stamp if n == 0 else '%s-%d' % (stamp, n))
        try:
            run.mkdir(parents=True, exist_ok=False)
            break
        except FileExistsError:
            continue
    else:
        raise RuntimeError('no free log directory for ' + stamp)
    env, updates = m.server_environment()
    argv = m.server_command()
    record = {'model': model, 'argv': argv, 'environment': updates, 'patch': patch, 'diagnostic_patch': diagnostic_patch,
              'min_p_patch': min_p_patch, 'radix_build': radix_build, 'qsa_quant_patch': qsa_quant_patch,
              'started_epoch': time.time(), 'log_dir': str(run),
              'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    m.save(run/'launch.json', record)
    with (run/'server.log').open('w') as log:
        child = subprocess.Popen(argv, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    root = m.process(child.pid)
    if root is None or root['pgid'] != child.pid:
        _klog('the server did not establish an owned session (root %r)' % (root,))
        root = root or dict(pid=child.pid, state='?', ppid=os.getpid(), pgid=child.pid, sid=child.pid, start_ticks=0, cmdline_sha256='')
    identity = {**record, 'root': root, 'processes': [root]}
    m.save(m.IDENTITY, identity)
    print(json.dumps({'pid': child.pid, 'log_dir': str(run), 'mtp_tokens': m.MTP_TOKENS}, indent=2))


action = sys.argv[1]
assert action in ('start', 'status', 'stop'), 'Instance launcher: start, status or stop (this instance only)'
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
    if (INST/'candidate-server.json').exists():   # k: logs only
        _klog('%s exists (an earlier launch of this instance; continuing)' % (INST/'candidate-server.json'))
    m.save(INST/'requested-config.json', dict(argv=m.server_command(), config=config,
        environment={k:v for k,v in os.environ.items() if k.startswith(('ARC3_','VLLM_'))
                     or k in ('PYTHONPATH','OMP_NUM_THREADS','MKL_NUM_THREADS')}))
    _k_start()   # k: minimal_inference.start with its checks logging
    identity = json.loads(m.IDENTITY.read_text())
    identity['persistent_experiment'] = dict(directory=str(HERE), manifest=manifest,
        phase=config['label'], prefix_caching=False,
        eager_correctness_candidate=bool(config['eager']), optimization_config=config)
    identity['prolong_template'] = dict(path=str(TEMPLATE), continuation_sha256=hashlib.sha256(TEMPLATE.read_bytes()).hexdigest())
    m.save(m.IDENTITY, identity)
    identity['stage2'] = dict(directory=str(S2D), parts=os.environ['ARC3_STAGE2'], mtp_tokens=int(os.environ['ARC3_MTP_TOKENS']),
                              alloc_conf=os.environ.get('PYTORCH_CUDA_ALLOC_CONF'), draft_model=os.environ.get('ARC3_S2_DRAFT_MODEL'),
                              pycache=os.environ.get('ARC3_S2_PYCACHE'), target_model=str(m.MODEL),
                              quantization=os.environ.get('ARC3_S2_QUANTIZATION'), moe_backend=os.environ.get('ARC3_S2_MOE_BACKEND'), shared_experts_stream_disabled=os.environ.get('ARC3_S2_DISABLE_SHARED_EXPERTS_STREAM'), production_launcher_sha256='f75af54fd789bac02121e6a7bd46479b91a727837606f37f87dc29305637fd4e',
                              instance=dict(name=INSTANCE, gpu=GPU, port=m.PORT, directory=str(INST), launcher='launch_scaling_s2i.py'))
    m.save(m.IDENTITY, identity)
    m.save(INST/'candidate-server.json', identity)
elif action == 'status':
    m.status()
elif action == 'stop':
    m.stop()
else:
    raise ValueError(action)
