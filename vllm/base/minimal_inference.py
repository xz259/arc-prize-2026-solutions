"""Explicit NVIDIA launcher for the prepared custom runtime; no watchdog/install.

Use runtime_python.sh. `plan` prints; `start` launches an owned server; `status`
checks it; `smoke` generates; `stop` stops its exact processes.
"""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
from urllib.request import Request, urlopen

# Baseline parameters; environment overrides support later controlled sweeps.
ROOT = Path(os.environ.get('ARC3_ROOT', Path(__file__).resolve().parent))
MODEL = Path(os.environ.get('ARC3_MODEL', '/workspace/models/Qwen3.8-Flash-Next-NVFP4'))
MODEL_REVISION = 'fc694b54fb0174e0913e6adf86691ef85a4ead47'
CONFIG_SHA256 = 'deef67a61f3311faf051b23dc4192f442c7fee4f9cd2f38cbcbe4da55c763a80'
VLLM_VERSION = '0.1.dev20073+g8e685d198'
MODEL_NAME = 'nvidia/Qwen3.8-Flash-Next-NVFP4'
HOST = '127.0.0.1'
PORT = int(os.environ.get('ARC3_PORT', '1234'))
CONTEXT = int(os.environ.get('ARC3_CONTEXT', '32768'))
CONCURRENCY = int(os.environ.get('ARC3_CONCURRENCY', '8'))
BATCHED_TOKENS = int(os.environ.get('ARC3_BATCHED_TOKENS', '8192'))
KV_GIB = float(os.environ.get('ARC3_KV_GIB', '8'))
# Explicit sizing takes precedence. Set ARC3_KV_GIB=0 to use this fallback.
GPU_MEMORY_UTILIZATION = float(os.environ.get('ARC3_GPU_MEMORY_UTILIZATION', '0.90'))
GRAPH_SIZE = int(os.environ.get('ARC3_GRAPH_SIZE', '32'))
MTP_TOKENS = int(os.environ.get('ARC3_MTP_TOKENS', '0'))
PREFIX_CACHE = os.environ.get('ARC3_PREFIX_CACHE', '0') == '1'
EAGER = os.environ.get('ARC3_EAGER', '0') == '1'
QSA_MODE = os.environ.get('ARC3_QSA_MODE', 'exact')
QSA_KV_FORMAT = os.environ.get('ARC3_QSA_KV_FORMAT', 'bf16')
FP8_READER = os.environ.get('ARC3_QSA_FP8_READER', 'baseline')
MOE_FINALIZE = os.environ.get('ARC3_MOE_FINALIZE', 'unfused')
MTP_MIN_P = os.environ.get('ARC3_MTP_MIN_P', '1') == '1'  # correctness tested; measured cost accepted
SAMPLER_TOPK_GUARD = os.environ.get('ARC3_SKIP_REDUNDANT_TOPK', '0')
# 2026-09-28: optional vLLM draft sampling / rejection method for MTP (unset = the vLLM defaults greedy / standard,
# and the speculative-config JSON is unchanged).
MTP_DRAFT_SAMPLE = os.environ.get('ARC3_MTP_DRAFT_SAMPLE', '')
MTP_REJECTION = os.environ.get('ARC3_MTP_REJECTION', '')
IDENTITY = ROOT/'server-identity.json'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name+'.tmp')
    temp.write_text(json.dumps(value, indent=2)+'\n')
    os.replace(temp, path)


def server_command():
    cmd = [sys.executable, '-m', 'vllm.entrypoints.cli.main', 'serve', str(MODEL),
           '--served-model-name', MODEL_NAME, '--host', HOST, '--port', str(PORT),
           '--load-format', 'safetensors', '--dtype', 'bfloat16',
           '--quantization', 'modelopt', '--tensor-parallel-size', '1',
           '--distributed-executor-backend', 'mp',
           '--max-model-len', str(CONTEXT), '--max-num-seqs', str(CONCURRENCY),
           '--max-num-batched-tokens', str(BATCHED_TOKENS),
           '--async-scheduling', '--enable-chunked-prefill',
           '--enable-prefix-caching' if PREFIX_CACHE else '--no-enable-prefix-caching',
           '--enable-auto-tool-choice', '--tool-call-parser', 'qwen3_coder',
           '--reasoning-parser', 'qwen3', '--chat-template', str(MODEL/'chat_template.jinja'),
           '--enable-prompt-tokens-details', '--enable-per-request-metrics',
           '--no-enable-log-requests', '--disable-uvicorn-access-log']
    if KV_GIB < 0 or not 0 < GPU_MEMORY_UTILIZATION <= 1:
        raise ValueError('KV GiB must be nonnegative and GPU utilization in (0, 1]')
    if KV_GIB:
        cmd += ['--kv-cache-memory-bytes', str(int(KV_GIB*1024**3))]
    else:
        cmd += ['--gpu-memory-utilization', str(GPU_MEMORY_UTILIZATION)]
    if GRAPH_SIZE:
        cmd += ['--max-cudagraph-capture-size', str(GRAPH_SIZE)]
    if EAGER:
        cmd += ['--enforce-eager']
    if PREFIX_CACHE:
        cmd += ['--mamba-cache-mode', 'align']
    if MTP_TOKENS:
        spec = {'method':'mtp','num_speculative_tokens':MTP_TOKENS}
        if MTP_DRAFT_SAMPLE:
            assert MTP_DRAFT_SAMPLE in ('greedy', 'probabilistic'), MTP_DRAFT_SAMPLE
            spec['draft_sample_method'] = MTP_DRAFT_SAMPLE
        if MTP_REJECTION:
            assert MTP_REJECTION in ('standard', 'block'), MTP_REJECTION
            spec['rejection_sample_method'] = MTP_REJECTION
        cmd += ['--speculative-config', json.dumps(spec)]
    profile_dir = os.environ.get('ARC3_PROFILE_DIR')
    if profile_dir:
        cmd += ['--profiler-config', json.dumps({
            'profiler':'torch', 'torch_profiler_dir':profile_dir,
            'torch_profiler_with_stack':False, 'torch_profiler_with_memory':False,
            'ignore_frontend':True, 'delay_iterations':4,
            'max_iterations':int(os.environ.get('ARC3_PROFILE_STEPS','8'))})]
    if os.environ.get('ARC3_MEMORY_DIAGNOSTICS') == '1':
        cmd += ['--worker-extension-cls', 'worker_diagnostics.WorkerDiagnostics']
    return cmd


def server_environment():
    env = os.environ.copy()
    for key in list(env):
        if key.startswith('VLLM_RADIXARK_'):
            env.pop(key)
    updates = {'VLLM_PLE_CPU_OFFLOAD':'1', 'VLLM_PLE_OFFLOAD_READY_TIMEOUT':'1800',
               'HF_HUB_OFFLINE':'1', 'HF_DATASETS_OFFLINE':'1', 'TRANSFORMERS_OFFLINE':'1',
               'CUDA_DEVICE_ORDER':'PCI_BUS_ID', 'CUDA_VISIBLE_DEVICES':'0',
               'TORCH_CUDA_ARCH_LIST':'12.0', 'PYTHONHASHSEED':'0',
               'OMP_NUM_THREADS':os.environ.get('OMP_NUM_THREADS','1'),
               'MKL_NUM_THREADS':os.environ.get('MKL_NUM_THREADS','1')}
    if QSA_MODE not in {'native', 'exact', 'radix'} or MOE_FINALIZE not in {'fused', 'unfused'}:
        raise ValueError('Invalid QSA selector or MoE finalizer mode')
    updates.update(ARC3_QSA_MODE=QSA_MODE, ARC3_MOE_FINALIZE=MOE_FINALIZE)
    if QSA_KV_FORMAT not in {'bf16', 'fp8_e4m3', 'fp8_e5m2', 'nvfp4'}:
        raise ValueError('Unsupported experimental QSA cache format')
    updates['ARC3_QSA_KV_FORMAT'] = QSA_KV_FORMAT
    if FP8_READER not in {'baseline','scale_placement'} or (FP8_READER != 'baseline' and QSA_KV_FORMAT not in {'fp8_e4m3','fp8_e5m2'}):
        raise ValueError('FP8 reader choice is invalid for this cache format')
    updates['ARC3_QSA_FP8_READER'] = FP8_READER
    threads = os.environ.get('ARC3_TOPK_THREADS', '256')
    cache_row = os.environ.get('ARC3_TOPK_CACHE_ROW', '1')
    if threads not in {'128', '256', '512', '1024'} or cache_row not in {'0', '1'}:
        raise ValueError('Invalid radix selector launch settings')
    updates.update(ARC3_TOPK_THREADS=threads, ARC3_TOPK_CACHE_ROW=cache_row)
    prefill_profile = os.environ.get('ARC3_QSA_PREFILL_PROFILE', 'stock')
    if prefill_profile not in {'stock','tile32'}:
        raise ValueError('Unsupported sparse-prefill profile')
    if prefill_profile != 'stock' and QSA_KV_FORMAT != 'bf16':
        raise ValueError('The optional sparse-prefill profile has only been implemented for BF16')
    updates['ARC3_QSA_PREFILL_PROFILE'] = prefill_profile
    updates['ARC3_MTP_MIN_P'] = '1' if MTP_MIN_P else '0'
    if SAMPLER_TOPK_GUARD not in {'0', '1'}:
        raise ValueError('Sampler top-k guard must be 0 or 1')
    updates['ARC3_SKIP_REDUNDANT_TOPK'] = SAMPLER_TOPK_GUARD
    env.update(updates)
    if os.environ.get('ARC3_MEMORY_DIAGNOSTICS') == '1':
        # The server remains bound to 127.0.0.1; only the local profiling run
        # enables the development RPC endpoint used by this extension.
        updates['VLLM_SERVER_DEV_MODE'] = '1'
        env.update(updates)
    return env, updates


def inspect_model():
    raw = (MODEL/'config.json').read_bytes()
    if sha(raw) != CONFIG_SHA256:
        raise RuntimeError('Model config changed; review checkpoint identity first')
    config = json.loads(raw)
    index = json.loads((MODEL/'model.safetensors.index.json').read_text())
    shards = sorted(set(index['weight_map'].values()))
    return {'model':str(MODEL), 'revision':MODEL_REVISION, 'config_sha256':sha(raw),
            'quantization':config['quantization_config']['quant_algo'],
            'shards':len(shards), 'shard_bytes':sum((MODEL/p).stat().st_size for p in shards)}


def process(pid):
    try:
        p = Path(f'/proc/{pid}')
        fields = (p/'stat').read_text().rsplit(')',1)[1].split()
        return {'pid':int(pid), 'state':fields[0], 'ppid':int(fields[1]),
                'pgid':int(fields[2]), 'sid':int(fields[3]), 'start_ticks':int(fields[19]),
                'cmdline_sha256':sha((p/'cmdline').read_bytes())}
    except (FileNotFoundError, ProcessLookupError):
        return None


def matches(saved):
    live = process(saved['pid'])
    # vLLM sets process titles after initialization, changing /proc/cmdline.
    # PID + kernel start time + isolated session/group identify this process
    # across that title change; the recorded cmdline hash is diagnostic only.
    return live is not None and live['state'] != 'Z' and all(
        live[k] == saved[k] for k in ['pid','start_ticks','pgid','sid'])


def owned_snapshot(identity):
    if not matches(identity['root']):
        return [r for r in identity.get('processes',[]) if matches(r)]
    table = {int(p.name):process(int(p.name)) for p in Path('/proc').iterdir() if p.name.isdigit()}
    selected = {identity['root']['pid']}
    while True:
        expanded = selected | {pid for pid,r in table.items() if r and r['ppid'] in selected}
        if expanded == selected:
            break
        selected = expanded
    return [r for pid,r in table.items() if r and (pid in selected or r['sid']==identity['root']['sid']) and r['state']!='Z']


def port_open():
    try:
        with socket.create_connection((HOST,PORT),timeout=1):
            return True
    except OSError:
        return False


def request(path, payload=None, timeout=10):
    body = None if payload is None else json.dumps(payload).encode()
    req = Request(f'http://{HOST}:{PORT}{path}',data=body,headers={'Content-Type':'application/json'})
    with urlopen(req,timeout=timeout) as response:
        data = response.read()
        return json.loads(data) if data else {'status':response.status}


def start():
    if port_open():
        raise RuntimeError(f'Port {PORT} is occupied')
    if IDENTITY.exists():
        old = json.loads(IDENTITY.read_text())
        if any(matches(p) for p in owned_snapshot(old)):
            raise RuntimeError('Prior owned server processes still live; use stop first')
    model = inspect_model()
    if importlib.metadata.version('vllm') != VLLM_VERSION:
        raise RuntimeError('Use runtime_python.sh with the pinned runtime')
    patch = json.loads((ROOT/'patches/manifest.json').read_text())
    site = ROOT/'runtime/base/usr/local/lib/python3.12/dist-packages'
    for row in patch['files']:
        if sha((site/row['path']).read_bytes()) != row['patched_sha256']:
            raise RuntimeError(f'Runtime patch identity mismatch: {row["path"]}')
    diagnostic_patch = None
    diagnostic_manifest = ROOT/'patches/determinism-manifest.json'
    if diagnostic_manifest.exists():
        diagnostic_patch = json.loads(diagnostic_manifest.read_text())
        for row in diagnostic_patch['files']:
            if sha((site/row['path']).read_bytes()) != row['patched_sha256']:
                raise RuntimeError(f'Diagnostic patch identity mismatch: {row["path"]}')
        if sha((ROOT/'arc3_determinism.py').read_bytes()) != diagnostic_patch['helper_sha256']:
            raise RuntimeError('Diagnostic helper identity mismatch')
        if (os.environ.get('ARC3_QSA_PREFILL_PROFILE','stock') != 'stock'
                and diagnostic_patch['patch'] != 'qsa-moe-diagnostics-v2'):
            raise RuntimeError('Optional sparse-prefill hook is not installed')
    elif QSA_MODE != 'native' or MOE_FINALIZE != 'fused':
        raise RuntimeError('Requested determinism hooks are not installed')
    radix_build = None
    if QSA_MODE == 'radix':
        radix_build = json.loads((ROOT/'kernels/build/latest.json').read_text())
        for name, expected in radix_build['sources'].items():
            if sha((ROOT/'kernels'/name).read_bytes()) != expected:
                raise RuntimeError('Radix selector build/source mismatch: '+name)
        library_path = Path(radix_build['library'])
        if not library_path.is_absolute():
            library_path = ROOT/library_path
        if sha(library_path.read_bytes()) != radix_build['library_sha256']:
            raise RuntimeError('Radix selector library identity mismatch')
        radix_build['wrapper_sha256'] = sha((ROOT/'arc3_topk_radix.py').read_bytes())
    qsa_quant_patch = None
    qsa_quant_manifest = ROOT/'patches/qsa-quant-manifest.json'
    if qsa_quant_manifest.exists():
        qsa_quant_patch = json.loads(qsa_quant_manifest.read_text())
        for row in qsa_quant_patch['files']:
            if sha((site/row['path']).read_bytes()) != row['patched_sha256']:
                raise RuntimeError('QSA cache patch identity mismatch: '+row['path'])
        helpers=qsa_quant_patch.get('helpers') or {'arc3_qsa_quant.py':qsa_quant_patch['helper_sha256']}
        for name, expected in helpers.items():
            if sha((ROOT/name).read_bytes()) != expected:
                raise RuntimeError('QSA cache helper identity mismatch: '+name)
        if QSA_KV_FORMAT != 'bf16' and QSA_KV_FORMAT not in qsa_quant_patch['formats']:
            raise RuntimeError('QSA format is not enabled in the installed patch manifest')
    elif QSA_KV_FORMAT != 'bf16':
        raise RuntimeError('Quantized QSA storage requested without the scoped runtime patch')
    if FP8_READER != 'baseline' and (qsa_quant_patch is None or FP8_READER not in qsa_quant_patch.get('fp8_readers', [])):
        raise RuntimeError('FP8 reader is not enabled in the installed patch manifest')
    min_p_patch = None
    min_p_manifest = ROOT/'patches/mtp-min-p-manifest.json'
    if min_p_manifest.exists():
        min_p_patch = json.loads(min_p_manifest.read_text())
        for row in min_p_patch['files']:
            if sha((site/row['path']).read_bytes()) != row['patched_sha256']:
                raise RuntimeError(f'MTP min-p patch identity mismatch: {row["path"]}')
    elif MTP_MIN_P:
        raise RuntimeError('MTP min-p support requested but patch is not installed')
    stamp = time.strftime('%Y%m%d-%H%M%S')   # 2026-10-20 (user-approved): two servers starting in one second collided here
    for n in range(100):                           # (FileExistsError, 10-18 and 10-20); mkdir is atomic, so a loser takes the next suffix
        run = ROOT/'logs'/(stamp if n == 0 else '%s-%d' % (stamp, n))
        try:
            run.mkdir(parents=True,exist_ok=False)
            break
        except FileExistsError:
            continue
    else:
        raise RuntimeError('no free log directory for ' + stamp)
    env, updates = server_environment()
    argv = server_command()
    record = {'model':model, 'argv':argv, 'environment':updates, 'patch':patch, 'diagnostic_patch':diagnostic_patch,
              'min_p_patch':min_p_patch, 'radix_build':radix_build, 'qsa_quant_patch':qsa_quant_patch,
              'started_epoch':time.time(), 'log_dir':str(run),
              'boot_id':Path('/proc/sys/kernel/random/boot_id').read_text().strip()}
    save(run/'launch.json',record)
    with (run/'server.log').open('w') as log:
        child = subprocess.Popen(argv,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    root = process(child.pid)
    if root is None or root['pgid'] != child.pid:
        raise RuntimeError('Server did not establish an owned session')
    identity = {**record,'root':root,'processes':[root]}
    save(IDENTITY,identity)
    print(json.dumps({'pid':child.pid,'log_dir':str(run),'mtp_tokens':MTP_TOKENS},indent=2))


def status():
    identity = json.loads(IDENTITY.read_text())
    identity['processes'] = owned_snapshot(identity)
    save(IDENTITY,identity)
    result = {'root_alive':matches(identity['root']),'processes':identity['processes'],'log_dir':identity['log_dir']}
    for route in ['/health','/v1/models']:
        try:
            result[route] = request(route,timeout=5)
        except Exception as e:
            result[route] = {'error':str(e)}
    print(json.dumps(result,indent=2))


def stop():
    identity = json.loads(IDENTITY.read_text())
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != identity['boot_id']:
        raise RuntimeError('Host boot identity changed; refusing to signal')
    records = owned_snapshot(identity)
    for sig, grace in [(signal.SIGTERM,30),(signal.SIGKILL,10)]:
        for p in records:
            if matches(p):
                try:
                    os.kill(p['pid'],sig)
                except ProcessLookupError:
                    pass
        deadline = time.monotonic()+grace
        while time.monotonic()<deadline and any(matches(p) for p in records):
            time.sleep(.5)
        if not any(matches(p) for p in records):
            break
    survivors = [p for p in records if matches(p)]
    result = {'stopped':not survivors and not port_open(),'survivors':survivors,'epoch':time.time()}
    save(Path(identity['log_dir'])/'stop.json',result)
    print(json.dumps(result,indent=2))
    if not result['stopped']:
        raise RuntimeError('Owned server did not stop completely')


def smoke():
    payload = {'model':MODEL_NAME,'messages':[{'role':'user','content':'What is 2 + 2? Answer briefly.'}],
               'temperature':0,'max_tokens':64,'chat_template_kwargs':{'enable_thinking':False}}
    started = time.monotonic()
    result = {'seconds':None,'response':request('/v1/chat/completions',payload,timeout=180)}
    result['seconds'] = time.monotonic()-started
    identity = json.loads(IDENTITY.read_text())
    save(Path(identity['log_dir'])/'text-smoke.json',result)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',nargs='?',default='plan',choices=['plan','inspect','start','status','smoke','stop'])
    action = parser.parse_args().action
    if action == 'plan':
        print(json.dumps({'command':server_command(),'environment':server_environment()[1]},indent=2))
    elif action == 'inspect':
        print(json.dumps(inspect_model(),indent=2))
    else:
        globals()[action]()
