"""mk_klaunch.py S2 (2026-10-24, Claude main session 099768dc; the Kaggle port, KAGGLE_PORT_PLAN_2026-10-24.md section 2; the user 10-24:
"It should run without any guards because a block fails the submission and we get zero"). Writes S2/k_launch_scaling_s2i.py from
S2/launch_scaling_s2i.py (md5-checked) and S2/k_runtime_python.sh from /workspace/arc3-inference/runtime_python.sh (md5-checked).
k_launch_scaling_s2i.py composes the server exactly as launch_scaling_s2i.py does; only the checks that would stop the launch log
instead ('[k_launch] ...' on stderr) and the launch continues:
  - the pod's host and boot pins (hostname 2a159825c76b, the boot id);
  - the production overlay's sha256 manifest (298 files): the files that differ or are missing are listed;
  - the chat template's sha256; the drafter view (verified, the same base checkpoint as the target view);
  - start: an existing candidate-server.json, an occupied port;
  - minimal_inference.start (6cb3de51) as _k_start: the prior-identity, vLLM version, runtime patch / determinism / radix / QSA /
    min-p manifests and the session-ownership checks log; the log directory, environment, argv, launch and identity are unchanged.
ARC3_PY_PREPEND (unset by default) goes first on the server processes' PYTHONPATH (the file tracer, filetrace/sitecustomize.py).
ARC3_K_SERVICE_AUDIT=0 (unset by default) turns the service audit off (the competition rerun's diagnostics-off set).
k_runtime_python.sh = runtime_python.sh with its three runtime checks logging and ARC3_PY_PREPEND first on PYTHONPATH when set."""
import hashlib
import os
import sys

S2 = os.path.abspath(sys.argv[1])
SRC = os.path.join(S2, 'launch_scaling_s2i.py')
RT = '/workspace/arc3-inference/runtime_python.sh'
assert hashlib.md5(open(SRC, 'rb').read()).hexdigest() == '4fb44cbd64770de12cbf3fa8f4df97b6', 'launch_scaling_s2i.py changed'
assert hashlib.md5(open(RT, 'rb').read()).hexdigest() == '901b0dbcc99650fd6c4c17541f6b99a4', 'runtime_python.sh changed'
MI = '/workspace/arc3-inference/minimal_inference.py'
assert hashlib.md5(open(MI, 'rb').read()).hexdigest() == '6cb3de51ac8f69eb790f5d974d0b24a1', 'minimal_inference.py changed'


def rep(s, a, b, what):
    assert s.count(a) == 1, '%s: anchor found %d times' % (what, s.count(a))
    return s.replace(a, b)


s = open(SRC).read()
s = rep(s, '"""Instance launcher (2026-10-11,', '"""k_launch_scaling_s2i.py (2026-10-24, mk_klaunch.py; the Kaggle port): launch_scaling_s2i.py 4fb44cbd with every check that\n'
        'would stop the launch logging instead (\'[k_launch] ...\' on stderr); the composition is unchanged. mk_klaunch.py lists the checks.\n\n'
        'Instance launcher (2026-10-11,', 'docstring')
s = rep(s, 'from urllib.request import urlopen\n', '''from urllib.request import urlopen
import importlib.metadata
import time


def _klog(msg):   # k (the Kaggle port): a check that would stop the launch logs, and the launch continues
    sys.stderr.write('[k_launch %s] %s\\n' % (time.strftime('%H:%M:%S', time.gmtime()), msg))
    sys.stderr.flush()
''', 'imports')
s = rep(s, '''assert socket.gethostname() == '2a159825c76b'
assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == 'e27af87d-a1a1-40d4-99c6-f972803e152b'
''', '''if socket.gethostname() != '2a159825c76b':   # k: the pod's host and boot pins log only
    _klog('host %s, not the pod (the host pin logs only)' % socket.gethostname())
try:
    if Path('/proc/sys/kernel/random/boot_id').read_text().strip() != 'e27af87d-a1a1-40d4-99c6-f972803e152b':
        _klog('boot id differs from the pod\\'s (logs only)')
except OSError as _kexc:
    _klog('boot id unreadable: %r' % _kexc)
''', 'host pins')
s = rep(s, '''manifest = json.loads((HERE/'overlay-manifest.json').read_text())
for name, expected in manifest.items():
    assert hashlib.sha256((HERE/name).read_bytes()).hexdigest() == expected, name
''', '''try:   # k: the overlay manifest logs the files that differ or are missing
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
''', 'overlay')
s = rep(s, '''import minimal_inference as m
''', '''if os.environ.get('ARC3_PY_PREPEND'):   # k: the file tracer first on the server processes' path (unset: nothing)
    os.environ['PYTHONPATH'] = os.environ['ARC3_PY_PREPEND'] + ':' + os.environ['PYTHONPATH']
import minimal_inference as m
''', 'prepend')
s = rep(s, '''assert hashlib.sha256(TEMPLATE.read_bytes()).hexdigest() == '86cef302280cac60c0148ada70ecd30d5084e5c0c4d65ace8755a855b5176a3d'
''', '''try:   # k: the template's identity logs only
    if hashlib.sha256(TEMPLATE.read_bytes()).hexdigest() != '86cef302280cac60c0148ada70ecd30d5084e5c0c4d65ace8755a855b5176a3d':
        _klog('chat template %s differs from its pinned sha256' % TEMPLATE)
except OSError as _kexc:
    _klog('chat template %s unreadable: %r' % (TEMPLATE, _kexc))
''', 'template')
s = rep(s, '''        _view = json.loads((Path(os.environ['ARC3_S2_DRAFT_MODEL'])/'ARC3_DRAFT_VIEW.json').read_text())
        _tv = m.MODEL/'ARC3_TARGET_VIEW.json'   # a config-only target variant names its base checkpoint
        _base = json.loads(_tv.read_text())['base_source'] if _tv.exists() else str(m.MODEL)
        assert _view.get('verified') and _view['source'] == _base, (_view.get('source'), _base, _view.get('verified'))
''', '''        try:   # k: the drafter view's check logs only
            _view = json.loads((Path(os.environ['ARC3_S2_DRAFT_MODEL'])/'ARC3_DRAFT_VIEW.json').read_text())
            _tv = m.MODEL/'ARC3_TARGET_VIEW.json'   # a config-only target variant names its base checkpoint
            _base = json.loads(_tv.read_text())['base_source'] if _tv.exists() else str(m.MODEL)
            if not (_view.get('verified') and _view['source'] == _base):
                _klog('drafter view: source %r, target base %r, verified %r' % (_view.get('source'), _base, bool(_view.get('verified'))))
        except Exception as _kexc:  # noqa: BLE001
            _klog('drafter view unreadable: %r' % _kexc)
''', 'drafter view')
s = rep(s, '''action = sys.argv[1]
''', '''

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
''', 'k_start')
s = rep(s, '''    assert not (INST/'candidate-server.json').exists(), 'Immutable launch already exists'
    assert not m.port_open(), 'Port occupied'
''', '''    if (INST/'candidate-server.json').exists():   # k: logs only
        _klog('%s exists (an earlier launch of this instance; continuing)' % (INST/'candidate-server.json'))
''', 'start asserts')
s = rep(s, '''        environment={k:v for k,v in os.environ.items() if k.startswith(('ARC3_','VLLM_'))
                     or k in ('PYTHONPATH','OMP_NUM_THREADS','MKL_NUM_THREADS')}))
    m.start()
''', '''        environment={k:v for k,v in os.environ.items() if k.startswith(('ARC3_','VLLM_'))
                     or k in ('PYTHONPATH','OMP_NUM_THREADS','MKL_NUM_THREADS')}))
    _k_start()   # k: minimal_inference.start with its checks logging
''', 'start call')
s = rep(s, 'import shutil\nimport socket\n', 'import shutil\nimport socket\nimport subprocess\n', 'subprocess import')
s = rep(s, '''if os.environ.get('ARC3_ENGINE_AUDIT', '1').strip().lower() in ('0', 'false', 'off', 'no'):
    os.environ.pop('ARC3_PERSISTENT_ENGINE_AUDIT', None)
''', '''if os.environ.get('ARC3_ENGINE_AUDIT', '1').strip().lower() in ('0', 'false', 'off', 'no'):
    os.environ.pop('ARC3_PERSISTENT_ENGINE_AUDIT', None)
# k (the user 10-24 via IE: diagnostics off in a competition rerun): ARC3_K_SERVICE_AUDIT=0 leaves the service audit's path unset, so
# session_service writes no service-events.jsonl (audit_path None: no writes); unset or 1 = on, as launch_scaling_s2i.py
if os.environ.get('ARC3_K_SERVICE_AUDIT', '1').strip().lower() in ('0', 'false', 'off', 'no'):
    os.environ.pop('ARC3_PERSISTENT_SERVICE_AUDIT', None)
''', 'service audit switch')
out = os.path.join(S2, 'k_launch_scaling_s2i.py')
with open(out + '.tmp', 'w') as fh:
    fh.write(s)
os.replace(out + '.tmp', out)

r = open(RT).read()
r = rep(r, '''test -f "$ARC3_ROOT/runtime/extraction.json"
test -x "$arc3_python"
"$arc3_python" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"] == "complete", "Runtime extraction is incomplete"' "$ARC3_ROOT/runtime/extraction.json"
''', '''# k_runtime_python.sh (2026-10-24, mk_klaunch.py; the Kaggle port): runtime_python.sh 901b0dbc with its runtime checks logging
# (the launch continues) and ARC3_PY_PREPEND (unset by default: the file tracer) first on PYTHONPATH
test -f "$ARC3_ROOT/runtime/extraction.json" || echo "[k_runtime_python] no $ARC3_ROOT/runtime/extraction.json (continuing)" >&2
test -x "$arc3_python" || echo "[k_runtime_python] $arc3_python is not executable (continuing)" >&2
"$arc3_python" -c 'import json,sys; assert json.load(open(sys.argv[1]))["status"] == "complete", "Runtime extraction is incomplete"' "$ARC3_ROOT/runtime/extraction.json" \\
  || echo "[k_runtime_python] the runtime extraction record is not complete (continuing)" >&2
''', 'runtime checks')
r = rep(r, '''export PYTHONPATH="$arc3_site/nvidia_cutlass_dsl/dsl_packages:$arc3_site"
''', '''export PYTHONPATH="${ARC3_PY_PREPEND:+$ARC3_PY_PREPEND:}$arc3_site/nvidia_cutlass_dsl/dsl_packages:$arc3_site"
''', 'runtime prepend')
r = rep(r, 'ARC3_ROOT="${ARC3_ROOT:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)}"', 'ARC3_ROOT="${ARC3_ROOT:-/workspace/arc3-inference}"', 'root')
rout = os.path.join(S2, 'k_runtime_python.sh')
with open(rout + '.tmp', 'w') as fh:
    fh.write(r)
os.chmod(rout + '.tmp', 0o755)
os.replace(rout + '.tmp', rout)
for p in (out, rout):
    print(hashlib.md5(open(p, 'rb').read()).hexdigest()[:8], p)
