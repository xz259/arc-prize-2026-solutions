"""build_kaggle_stage.py ITEM [ITEM ...] (2026-10-24, Claude main session 099768dc; KAGGLE_PORT_PLAN_2026-10-24.md section 1): builds
the staged upload folders under /workspace/kaggle_stage/<item>/ for the user's upload notebook (kaggle_upload_v2.ipynb). Each folder
holds the files to upload plus MANIFEST.json (each file's bytes and sha256; for an archive also its member count, its source paths and
the extraction root). Archives are tar streams compressed by zstd (-T16 -3), keeping symlinks, directory symlinks, permissions and the
absolute layout (members are relative to /, so `tar -C <root> -xf` recreates /mnt/shared/... under <root>). Run at idle priority.
Items:
  smoke    a file, a file symlink, an exec-bit script, a nested directory and a small .tar.zst holding the same (the mechanics test);
  runtime  runtime.tar.zst (/mnt/shared/arc3-inference/runtime: the serving runtime, 27 GB) and venvs.tar.zst (venv-runtime, venv-duck);
  games    games.tar.zst: pub25cc30 (the core 55), all223 (the 110-game pool's environment files) and the game lists in fin74;
  bundle   the static Kaggle model (2026-10-24, the user's forum notes: big files as Kaggle Models, which Kaggle places on NFS, and fewer
           mounts): drafter/ (the fr1cfinal files), aux/ (the KV means and the draft vocabulary), and the runtime, venvs and games archives,
           all as file symlinks in real subfolders (kagglehub follows file links, not directory links), <= 50 files.
A folder that exists is refused (FORCE=1 replaces it)."""
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

STAGE = '/workspace/kaggle_stage'
FORCE = os.environ.get('FORCE') == '1'
ITEMS = {
    'runtime': {'runtime.tar.zst': ['mnt/shared/arc3-inference/runtime'],
                'venvs.tar.zst': ['mnt/shared/arc3-inference/venv-runtime', 'mnt/shared/arc3-inference/venv-duck',
                                  'mnt/shared/arc3-inference/python']},   # the uv standalone CPython 3.12.12 both venvs' bin/python link to
    'games': {'games.tar.zst': ['mnt/shared/arc3-inference/pub25cc30', 'mnt/shared/arc3-inference/all223',
                                'workspace/kvring_dev/fin74/games_a.txt', 'workspace/kvring_dev/fin74/games_110.txt']},
}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for b in iter(lambda: fh.read(64 << 20), b''):
            h.update(b)
    return h.hexdigest()


def pipe2(cmd1, cmd2, stdout):
    """cmd1 | cmd2 > stdout with BOTH return codes checked (the Kaggle-upload review, A: no /bin/sh pipeline without pipefail)."""
    a = subprocess.Popen(cmd1, stdout=subprocess.PIPE)
    b = subprocess.Popen(cmd2, stdin=a.stdout, stdout=stdout)
    a.stdout.close()
    rb, ra = b.wait(), a.wait()
    assert ra == 0 and rb == 0, (cmd1[0], ra, cmd2[0], rb)


POD_LINKS = ['/workspace/arc3-inference/' + x for x in ('python', 'runtime', 'venv-runtime', 'venv-duck', 'pub25cc30', 'all223')]


def links_report(members):
    """Every symlink under the members: targets outside the archived set (after the pod's /workspace -> /mnt/shared links) and the
    pod links the notebook must recreate (the review, C)."""
    roots = ['/' + m for m in members]
    inside = lambda q: any(q == r or q.startswith(r.rstrip('/') + '/') for r in roots)   # noqa: E731
    external, via_pod, n = [], {}, 0
    for r in roots:
        for d, dirs, names in os.walk(r):
            for name in dirs + names:
                q = os.path.join(d, name)
                if not os.path.islink(q):
                    continue
                n += 1
                t = os.readlink(q)
                ta = os.path.normpath(t if os.path.isabs(t) else os.path.join(d, t))
                if inside(ta):
                    continue
                real = os.path.realpath(q)
                if inside(real):
                    for pl in POD_LINKS:
                        if ta == pl or ta.startswith(pl + '/'):
                            via_pod[pl] = os.readlink(pl) if os.path.islink(pl) else None
                    continue
                external.append(dict(link=q, target=t, resolves=real, exists=os.path.exists(q)))
    return dict(symlinks=n, external=len(external), external_samples=external[:50], pod_links_needed=via_pod)


def archive(out, members):
    for m in members:
        assert os.path.lexists('/' + m), 'missing: /' + m
        assert not os.path.islink('/' + m), 'a member is a symlink (tar would store only the link): /' + m
    t0 = time.time()
    with open(out + '.tmp', 'wb') as fh:
        tar = subprocess.Popen(['tar', '-C', '/', '--numeric-owner', '-cf', '-'] + members, stdout=subprocess.PIPE)
        z = subprocess.Popen(['zstd', '-T16', '-3', '-q', '-c'], stdin=tar.stdout, stdout=fh)
        tar.stdout.close()
        rz, rt = z.wait(), tar.wait()
    assert rt == 0 and rz == 0, ('tar rc', rt, 'zstd rc', rz)
    os.replace(out + '.tmp', out)
    with tempfile.TemporaryFile() as lst:
        pipe2(['zstd', '-dc', '-q', out], ['tar', '-tf', '-'], lst)   # the archive decodes and lists to the end
        lst.seek(0)
        n = sum(1 for _ in lst)
    info = dict(members=members, entries=n, seconds=round(time.time() - t0, 1),
                extract='zstd -dc <file> | tar -C <root> -xf -   (members are relative to /)', links=links_report(members))
    for m in members:
        cfg = '/' + m + '/pyvenv.cfg'
        if os.path.exists(cfg):
            info.setdefault('pyvenv', {})[m] = dict(l.strip().split(' = ', 1) for l in open(cfg) if ' = ' in l)
    print('   %s: %d entries, %d symlinks, %d external, pod links needed %s' % (os.path.basename(out), n, info['links']['symlinks'],
                                                                             info['links']['external'], sorted(info['links']['pod_links_needed'])), flush=True)
    return info


def build_smoke(d):
    src = os.path.join(d, 'src')
    os.makedirs(os.path.join(src, 'nested', 'dir'))
    with open(os.path.join(src, 'hello.txt'), 'w') as fh:
        fh.write('hello from the arc3 upload smoke test\n')
    os.symlink('hello.txt', os.path.join(src, 'link_to_hello.txt'))
    with open(os.path.join(src, 'run.sh'), 'w') as fh:
        fh.write('#!/bin/sh\necho exec bit ok\n')
    os.chmod(os.path.join(src, 'run.sh'), 0o755)
    with open(os.path.join(src, 'nested', 'dir', 'deep.txt'), 'w') as fh:
        fh.write('nested ok\n')
    os.symlink('nested', os.path.join(src, 'dirlink'))
    with open(os.path.join(d, 'smoke.tar.zst'), 'wb') as fh:
        pipe2(['tar', '-C', src, '-cf', '-', '.'], ['zstd', '-q', '-c'], fh)
    for name in os.listdir(src):                                    # the loose copies (kagglehub's own walk decides what goes up)
        p = os.path.join(src, name)
        shutil.move(p, os.path.join(d, name))
    os.rmdir(src)
    return {'smoke.tar.zst': dict(members=['.'], extract='zstd -dc smoke.tar.zst | tar -C <dir> -xf -; expect hello.txt, link_to_hello.txt '
                                           '(a symlink), run.sh (mode 755), nested/dir/deep.txt, dirlink (a directory symlink)')}


BUNDLE = {'drafter': '/workspace/models/Qwen3.8-Flash-Next-mtp-draft-nvfp4a16-fr1cfinal',
          'aux': ['/workspace/kvring_dev/fin74/s2/kvlut_means_dkv74drafter_e27b02d9.pt', '/workspace/kvring_dev/fin74/s2/draft_vocab_32k.json'],
          'archives': [STAGE + '/runtime/runtime.tar.zst', STAGE + '/runtime/venvs.tar.zst', STAGE + '/games/games.tar.zst']}


def build_bundle(d):
    os.makedirs(os.path.join(d, 'drafter'))
    for name in sorted(os.listdir(BUNDLE['drafter'])):
        src = os.path.join(BUNDLE['drafter'], name)
        assert os.path.isfile(src) and not os.path.islink(src), src
        os.symlink(src, os.path.join(d, 'drafter', name))
    os.makedirs(os.path.join(d, 'aux'))
    for src in BUNDLE['aux']:
        os.symlink(src, os.path.join(d, 'aux', os.path.basename(src)))
    info = {}
    for src in BUNDLE['archives']:
        os.symlink(src, os.path.join(d, os.path.basename(src)))
        man = json.load(open(os.path.join(os.path.dirname(src), 'MANIFEST.json')))
        info[os.path.basename(src)] = dict(man['archives'][os.path.basename(src)], built=man['built'], sha256=man['files'][os.path.basename(src)]['sha256'])
    return info


for item in sys.argv[1:]:
    assert item in ('smoke', 'bundle') + tuple(ITEMS), item
    d = os.path.join(STAGE, item)
    if os.path.exists(d):
        assert FORCE, 'exists: %s (FORCE=1 replaces it)' % d
        shutil.rmtree(d)
    os.makedirs(d)
    print('[%s] %s ...' % (time.strftime('%H:%M:%S', time.gmtime()), item), flush=True)
    info = (build_smoke(d) if item == 'smoke' else build_bundle(d) if item == 'bundle'
            else {name: archive(os.path.join(d, name), members) for name, members in ITEMS[item].items()})
    files = {}
    for root, dirs, names in os.walk(d):
        for name in names:
            p = os.path.join(root, name)
            rel = os.path.relpath(p, d)
            link = os.readlink(p) if os.path.islink(p) else None
            known = (info.get(name) or {}).get('sha256') if item == 'bundle' else None     # the archives' sha256 from their own manifests
            files[rel] = dict(bytes=os.stat(p).st_size, symlink=link, mode=oct(os.stat(p).st_mode & 0o777),
                              sha256=known or (sha256(p) if (item == 'bundle' or link is None) else None))
    man = dict(item=item, built=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), files=files, archives=info,
               builder='build_kaggle_stage.py (Claude main session 099768dc)')
    with open(os.path.join(d, 'MANIFEST.json'), 'w') as fh:
        json.dump(man, fh, indent=1)
    tot = sum(f['bytes'] for f in files.values())
    print('[%s] %s: %d files, %.2f GB; %s' % (time.strftime('%H:%M:%S', time.gmtime()), item, len(files), tot / 1e9,
                                             {k: (v.get('entries'), v.get('seconds')) for k, v in info.items()}), flush=True)
