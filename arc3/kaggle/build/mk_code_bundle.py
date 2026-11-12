"""mk_code_bundle.py OUT_DIR REPORT.json [REPORT.json ...] (2026-10-24, Claude main session 099768dc; the Kaggle port, the arc3-code model,
KAGGLE_PORT_PLAN_2026-10-24.md section 1). Builds OUT_DIR/code.tar.zst: the stack's code and compile caches, as the pod has them, for the
Kaggle notebook to extract at / (the members are relative to /, so the pod's absolute paths come back as they are).
  - every file the traced launches read (trace_report.py's code and cache classes, REPORT.json from the pod rehearsal of kaggle_run.py),
    with each symlink on the way stored as a link (the pod's /workspace -> /mnt/shared links), so Path(__file__).resolve() and every
    cache key see the pod's paths;
  - whole directories as a safety net for files a trace can miss (sandbox imports run with -I -S; lazy imports): the harness, stage2_v1,
    s2/lib, the production overlay's manifest files, the bytecode cache, the AOT / compile caches of the listed profiles, the small
    runtime caches, the custom games' Python, the Kaggle launcher kit;
  - excluded: the runtime and venv archives' trees, the models, the games archive's trees, run outputs.
OUT_DIR/CODE_MANIFEST.json lists the members (path, kind, bytes) and the archive's sha256. Run at idle priority after the rehearsal.
--override MEMBER=SOURCE (2026-10-25, the final code while fin74 serves n76): the archive's member MEMBER (an absolute path the bundle
already holds as a file) gets SOURCE's content; the tree itself is untouched. After the archive is written, every overridden member and
one untouched control file are read back from it and md5-compared (an assert); the manifest records each override."""
import hashlib
import json
import os
import subprocess
import sys
import time

OUT = os.path.abspath(sys.argv[1])
REPORTS, OVERRIDES = [], {}
_a = sys.argv[2:]
while _a:
    x = _a.pop(0)
    if x == '--override':
        m_, _, src_ = _a.pop(0).partition('=')
        assert m_.startswith('/') and os.path.isfile(src_), ('--override MEMBER=SOURCE', m_, src_)
        OVERRIDES[m_] = os.path.abspath(src_)
    else:
        REPORTS.append(x)
S2 = '/workspace/kvring_dev/fin74/s2'
ROOT = '/workspace/arc3-inference'
EXCLUDE = ('/mnt/shared/arc3-inference/runtime/', '/mnt/shared/arc3-inference/venv-runtime/', '/mnt/shared/arc3-inference/venv-duck/',
           '/mnt/shared/arc3-inference/python/', '/mnt/shared/models/', '/workspace/models/', '/mnt/shared/arc3-inference/pub25cc30/',
           '/mnt/shared/arc3-inference/all223/', '/workspace/kvring_dev/fin74/runs/', S2 + '/instances/', S2 + '/receipts/',
           S2 + '/client_mtp/', '/workspace/arc3-inference/logs/', '/mnt/shared/arc3-inference/logs/', '/proc/', '/dev/', '/sys/', '/tmp/')
EXCLUDE_EXACT = {'/workspace/models', '/workspace/arc3-inference/runtime', '/workspace/arc3-inference/venv-runtime',
                 '/workspace/arc3-inference/venv-duck', '/workspace/arc3-inference/python', '/workspace/arc3-inference/pub25cc30',
                 '/workspace/arc3-inference/all223',   # links into the other archives: the notebook recreates them
                 '/workspace/kvring_dev/fin74/games_a.txt', '/workspace/kvring_dev/fin74/games_110.txt'}   # in games.tar.zst: one writer per file
SKIP_PARTS = ('/__pycache__/', '.pre-v7', '.new7', '.tmp')
TAGS = ['tier-off-lmh0-bt4096-n29-u0.95-pm0-c80204-plehostpoll-foldB-pos-gfgemmgdnhcfsplitk-xfp8wsampler-gsscaled16-s2mtp4-p6f3baeae81dc94a6',
        'tier-off-lmh0-bt4096-n27-u0.95-pm0-c89516-plehostpoll-foldB-pos-gfgemmgdnhcfsplitk-xfp8wsampler-gsscaled16-s2mtp4-p6f3baeae81dc94a6',
        'tier-off-lmh0-bt4096-n25-u0.95-pm0-c98828-plehostpoll-foldB-pos-gfgemmgdnhcfsplitk-xfp8wsampler-gsscaled16-s2mtp4-p6f3baeae81dc94a6']   # 98K (the user, after the 23:42:05 UTC read: 80K vs 98K on the LB)
if os.environ.get('CODE_TAGS'):                # 2026-10-25 (the user: 80K and 92K only): the cache tags to ship, comma-separated
    TAGS = [t.strip() for t in os.environ['CODE_TAGS'].split(',') if t.strip()]
    for t in TAGS:
        assert os.path.isdir(S2 + '/aot-cache-' + t) and os.path.isdir(S2 + '/compiled-cache-' + t), ('no caches for tag', t)
WHOLE = [S2 + '/harness_v6', S2 + '/stage2_v1', S2 + '/lib', '/workspace/kvring_dev/fin74/pycache', ROOT + '/cache/runtime/torchinductor',
         ROOT + '/cache/runtime/flashinfer', ROOT + '/cache/runtime/cuda', ROOT + '/cache/runtime/torch', ROOT + '/cache/runtime/vllm/flashinfer_autotune_cache',
         ROOT + '/cache/runtime/vllm/deep_gemm', ROOT + '/cache/runtime/vllm/modelinfos', ROOT + '/copycat30/py', '/workspace/kvring_dev/kaggle']
WHOLE += [S2 + '/aot-cache-' + t for t in TAGS] + [S2 + '/compiled-cache-' + t for t in TAGS]
WHOLE += [ROOT + '/runs/stage0-reference-20260919-v1/nll-panel-cold', S2 + '/nll-panels']   # the NLL panel's inputs (37 MB)
FILES = [S2 + '/k_launch_scaling_s2i.py', S2 + '/k_runtime_python.sh', S2 + '/launch_scaling_s2i.py', S2 + '/draft_vocab_32k.json',
         S2 + '/kvlut_means_dkv74drafter_e27b02d9.pt', ROOT + '/runtime_python.sh', ROOT + '/minimal_inference.py',
         ROOT + '/runs/dense_fp8_v1/overlay-manifest.json', ROOT + '/runs/dense_fp8_v1/optimization-config.json',
         # the optional NLL panel (kaggle_run.py nll_panel; its probe runs untraced, PYTHONPATH=ROOT): nll_panel_i.sh, the instance probe
         # wrapper, the probe and its import closure (modulefinder, 10-24), the stage-0 panel (WHOLE below)
         S2 + '/nll_panel_i.sh', S2 + '/probe_instance.py', ROOT + '/runs/stage0-reference-20260919-v1/probe_multimodal_likelihood_stage0.py',
         ROOT + '/benchmark_serving_config.py', ROOT + '/game_allocation.py', ROOT + '/gameplay_config.py', ROOT + '/profile_monitor.py',
         ROOT + '/run_decode_profile_candidate.py', ROOT + '/run_quantized_serving_candidate.py']
members, kinds, why = {}, {}, {}


def excluded(p):
    return p in EXCLUDE_EXACT or p.startswith(EXCLUDE) or any(x in p for x in SKIP_PARTS)


def add(p, reason, depth=0):
    """p (absolute) and every symlink on its way: links as links, then the file at the end."""
    if depth > 40 or excluded(p):
        return
    parts = p.strip('/').split('/')
    cur = ''
    for i, comp in enumerate(parts):
        cur = cur + '/' + comp
        if excluded(cur):
            return
        if os.path.islink(cur):
            members[cur], kinds[cur] = 'link', 'link'
            why.setdefault(cur, reason)
            t = os.readlink(cur)
            t = t if t.startswith('/') else os.path.normpath(os.path.join(os.path.dirname(cur), t))
            rest = '/'.join(parts[i + 1:])
            add(t + ('/' + rest if rest else ''), reason, depth + 1)
            return
    if os.path.isfile(p):
        members[p], kinds[p] = 'file', 'file'
        why.setdefault(p, reason)


def add_tree(d, reason):
    if not os.path.exists(d):
        print('  missing (skipped): %s' % d)
        return
    add(d, reason)                          # the links on the way to the directory
    real = os.path.realpath(d)
    for root, dirs, files in os.walk(real, followlinks=False):
        dirs[:] = [x for x in dirs if x != '__pycache__' or root.startswith('/workspace/kvring_dev/fin74/pycache')]
        for name in files + [x for x in dirs if os.path.islink(os.path.join(root, x))]:
            add(os.path.join(root, name), reason)


t0 = time.time()
for rp in REPORTS:
    rep = json.load(open(rp))
    for cls in ('code', 'cache', 'home'):
        for it in rep['classes'].get(cls, []):
            if cls == 'home' and '/.cache/' not in it['path'] and '/.triton/' not in it['path']:
                continue
            add(it.get('opened_as') or it['path'], 'trace:' + cls)
            add(it['path'], 'trace:' + cls)
    for it in rep['classes'].get('cache', []):        # a Triton cache entry: its whole directory (the trace sees what it opened)
        if '/cache/runtime/triton/' in it['path']:
            add_tree(os.path.dirname(it['path']), 'triton-entry')
for d in WHOLE:
    add_tree(d, 'whole')
for f in FILES:
    add(f, 'listed')
man = json.load(open(ROOT + '/runs/dense_fp8_v1/overlay-manifest.json'))
for name in man:
    add(ROOT + '/runs/dense_fp8_v1/' + name, 'overlay-manifest')
os.makedirs(OUT, exist_ok=True)
rows = sorted(members)
for m_ in OVERRIDES:
    assert kinds.get(m_) == 'file', ('an override must replace a file the bundle holds', m_, kinds.get(m_))
    why[m_] = 'override:' + why[m_]
OV_ROOT = os.path.join(OUT, 'override_root')            # the overriding files at their members' paths (tar -C in the list)
if OVERRIDES:
    import shutil
    if os.path.isdir(OV_ROOT):
        shutil.rmtree(OV_ROOT)                          # this builder's own staging from an earlier build
    for m_, src_ in OVERRIDES.items():
        os.makedirs(os.path.dirname(OV_ROOT + m_), exist_ok=True)
        shutil.copy2(src_, OV_ROOT + m_)


def src_of(p):
    return OV_ROOT + p if p in OVERRIDES else p


size = sum(os.path.getsize(src_of(p)) for p in rows if kinds[p] == 'file')
lst = os.path.join(OUT, 'code_members.txt')
with open(lst + '.tmp', 'w') as fh:
    fh.write(''.join(p.lstrip('/') + '\n' for p in rows if p not in OVERRIDES))
    if OVERRIDES:                                       # GNU tar: a -C line in a -T list changes the directory for the names after it
        fh.write('-C' + OV_ROOT + '\n' + ''.join(p.lstrip('/') + '\n' for p in rows if p in OVERRIDES))
os.replace(lst + '.tmp', lst)
arc = os.path.join(OUT, 'code.tar.zst')
with open(arc + '.tmp', 'wb') as fh:
    tar = subprocess.Popen(['tar', '-C', '/', '--no-recursion', '--numeric-owner', '-cf', '-', '-T', lst], stdout=subprocess.PIPE)
    z = subprocess.Popen(['zstd', '-T16', '-3', '-q', '-c'], stdin=tar.stdout, stdout=fh)
    tar.stdout.close()
    rz, rt = z.wait(), tar.wait()
assert rt == 0 and rz == 0, ('tar rc', rt, 'zstd rc', rz)
os.replace(arc + '.tmp', arc)
h = hashlib.sha256()
with open(arc, 'rb') as fh:
    for b in iter(lambda: fh.read(64 << 20), b''):
        h.update(b)


def md5f(p):
    return hashlib.md5(open(p, 'rb').read()).hexdigest()


def md5_member(p):
    z_ = subprocess.Popen(['zstd', '-dc', arc], stdout=subprocess.PIPE)
    t_ = subprocess.run(['tar', '-xOf', '-', p.lstrip('/')], stdin=z_.stdout, capture_output=True)
    z_.stdout.close()
    z_.wait()
    return hashlib.md5(t_.stdout).hexdigest() if t_.returncode == 0 else 'tar rc %d' % t_.returncode


readback = {}
ctl = next((p for p in rows if kinds[p] == 'file' and p not in OVERRIDES and p.endswith('/harness_v6/v62/session_ext.py')), None)
for p in list(OVERRIDES) + ([ctl] if ctl else []):
    want, got = md5f(src_of(p)), md5_member(p)
    readback[p] = dict(want=want[:8], got=got[:8], source=src_of(p) if p in OVERRIDES else p,
                       tree=md5f(p)[:8])
    assert got == want, ('read back from the archive', p, got, want)
by = {}
for p in rows:
    k = why[p].split(':')[0]
    by[k] = by.get(k, 0) + 1
info = dict(built=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), builder='mk_code_bundle.py (Claude main session 099768dc)', tags=TAGS,
            reports=REPORTS, members=len(rows), files=sum(1 for p in rows if kinds[p] == 'file'), links=sum(1 for p in rows if kinds[p] == 'link'),
            file_bytes=size, archive=dict(path=arc, bytes=os.path.getsize(arc), sha256=h.hexdigest()), by_reason=by,
            overrides={m_: dict(source=src_, md5=md5f(src_)[:8], tree_md5=md5f(m_)[:8]) for m_, src_ in OVERRIDES.items()},
            readback=readback,
            extract='zstd -dc code.tar.zst | tar -C / -xf -   (members are relative to /)',
            rows=[dict(path=p, kind=kinds[p], why=why[p], bytes=os.path.getsize(src_of(p)) if kinds[p] == 'file' else None,
                       target=os.readlink(p) if kinds[p] == 'link' else None) for p in rows])
with open(os.path.join(OUT, 'CODE_MANIFEST.json.tmp'), 'w') as fh:
    json.dump(info, fh, indent=1)
os.replace(os.path.join(OUT, 'CODE_MANIFEST.json.tmp'), os.path.join(OUT, 'CODE_MANIFEST.json'))
print('code.tar.zst: %d members (%d files, %d links), %.1f MB of files -> %.1f MB archive, sha256 %s, %.0f s; by reason %s' % (
    len(rows), info['files'], info['links'], size / 1e6, info['archive']['bytes'] / 1e6, h.hexdigest()[:16], time.time() - t0, by))
for p, r in readback.items():
    print('  read back %s: archive %s, source %s (tree %s)%s' % (p, r['got'], r['want'], r['tree'], ' OVERRIDE' if p in OVERRIDES else ' control'))
