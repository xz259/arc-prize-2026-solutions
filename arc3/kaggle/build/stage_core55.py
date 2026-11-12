"""stage_core55.py OUT_DIR (2026-10-25, Claude main session 099768dc; the user: "We should upload the core 55 games in a single dataset for
kaggle testing (Save and run all cells to do the 4.5h run of 55 games)"; "He uses select games from public-25 but we want to stress test
using core 55"). Stages the Kaggle dataset arc3-core55: core55.tar.zst (members relative to /: mnt/shared/arc3-inference/pub25cc30, the 55
core games' environment files = the public 25 + the copycat 30, and workspace/kvring_dev/fin74/games_a.txt, their order on the pod) and
MANIFEST.json (sha256, members, entries, the 55 game ids). The tree is 113 files, over kagglehub's 50-file rule, hence one archive. The
copycat games' helper modules (arc3-inference/copycat30/py) ride in the arc3-code archive with the client's PYTHONPATH."""
import hashlib
import json
import os
import subprocess
import sys
import time

OUT = os.path.abspath(sys.argv[1])
MEMBERS = ['mnt/shared/arc3-inference/pub25cc30', 'workspace/kvring_dev/fin74/games_a.txt']
for m in MEMBERS:
    assert os.path.exists('/' + m) and not os.path.islink('/' + m), m
games = open('/workspace/kvring_dev/fin74/games_a.txt').read().strip().split(',')
envs = sorted(os.listdir('/mnt/shared/arc3-inference/pub25cc30/environment_files'))
assert len(games) == 55 and len(envs) == 55 and sorted(g.split('-')[0] for g in games) == envs, (len(games), len(envs))
os.makedirs(OUT, exist_ok=True)
arc = os.path.join(OUT, 'core55.tar.zst')
t0 = time.time()
with open(arc + '.tmp', 'wb') as fh:
    tar = subprocess.Popen(['tar', '-C', '/', '--numeric-owner', '-cf', '-'] + MEMBERS, stdout=subprocess.PIPE)
    z = subprocess.Popen(['zstd', '-T8', '-3', '-q', '-c'], stdin=tar.stdout, stdout=fh)
    tar.stdout.close()
    rz, rt = z.wait(), tar.wait()
assert rt == 0 and rz == 0, ('tar rc', rt, 'zstd rc', rz)
os.replace(arc + '.tmp', arc)
lst = subprocess.run("zstd -dc '%s' | tar -tf -" % arc, shell=True, capture_output=True, text=True)
assert lst.returncode == 0, lst.stderr[-300:]
h = hashlib.sha256(open(arc, 'rb').read()).hexdigest()
man = dict(item='core55', built=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()), builder='stage_core55.py (Claude main session 099768dc)',
           files={'core55.tar.zst': dict(bytes=os.path.getsize(arc), sha256=h, symlink=None, mode='0o644')},   # the upload notebook's plan() schema (10-25 fix)
           members=MEMBERS, entries=len(lst.stdout.split()),
           extract='zstd -dc core55.tar.zst | tar -C / -xf -   (members are relative to /)', games=games,
           environments_dir='/mnt/shared/arc3-inference/pub25cc30/environment_files (= /workspace/arc3-inference/pub25cc30/environment_files)')
with open(os.path.join(OUT, 'MANIFEST.json.tmp'), 'w') as fh:
    json.dump(man, fh, indent=1)
os.replace(os.path.join(OUT, 'MANIFEST.json.tmp'), os.path.join(OUT, 'MANIFEST.json'))
print('core55.tar.zst: %d entries, %.1f MB, sha256 %s, %.0f s; 55 games, first %s' % (man['entries'], os.path.getsize(arc) / 1e6, h[:16],
                                                                                    time.time() - t0, games[:3]))
