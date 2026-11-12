"""Side-pod re-pin (2026-10-12, Claude fork 5d20b23b): the host guards of the side pod's PRIVATE copies only.

The side pod (container 6bc94744da90, boot 0902dc44-...; 2 x RTX PRO 6000 Blackwell Server Edition, driver 595.91.07, 2 x
EPYC 9535, 377 GB container limit) mounts the shared volume at /mnt/shared and keeps /workspace on its container disk
(build_sidepod_workspace.sh). runs/dense_fp8_v1 and the stage-2 tree under /workspace there are private copies of the main
pod's, pinned to the main pod (container a8da6c56441e, boot 4c1eccbd-...; re-pin v17). This is v17's procedure (host
guards only: dense_fp8_v1 launch_scaling.py asserts, private_moe/settings.json, the inner private_moe maps, gameplay-canary-
pins.json, overlay-manifest.json, LINEAGE; the stage-2 launcher S2/launch_scaling_s2.py asserts) from the main pod's
identity to the side pod's, with one addition: every file it writes must be a regular file on the container disk under
/workspace (not a link, not on the shared volume), and it runs only on the side pod. The main pod's files are never
touched. Idempotent per file."""
import hashlib, json, os, re, socket, time
from pathlib import Path

SIDE_HOST = '6bc94744da90'
OLD_HOST, OLD_BOOT = 'a8da6c56441e', '4c1eccbd-6c61-455d-b472-8a9de7d4db6c'
ROOT = Path('/workspace/arc3-inference/runs')
S2 = ROOT / 'stage0-reference-20260919-v1/stage12/harness_revision_v529/spec_decode_stage2'
host = socket.gethostname(); boot = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
assert host == SIDE_HOST, f'not the side pod: {host}'
assert (host, boot) != (OLD_HOST, OLD_BOOT)
WS_DEV = os.stat('/workspace').st_dev
assert os.stat('/mnt/shared').st_dev != WS_DEV, '/workspace is on the shared volume'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest(); when = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())


def private(p):
    p = Path(p)
    real = os.path.realpath(p)
    probe = real if os.path.exists(real) else os.path.dirname(real)   # a new file: its directory
    assert real.startswith('/workspace/') and os.stat(probe).st_dev == WS_DEV, f'not on the container disk: {p} -> {real}'
    for part in [p, *p.parents]:   # no link anywhere on the path below /workspace
        if str(part) in ('/', '/workspace'):
            break
        assert not part.is_symlink(), f'link on the path: {part}'
    return p


def put(p, text):
    private(p).write_text(text)


for d in (ROOT / 'dense_fp8_v1', S2):
    private(d)
print(f'host={host} boot={boot}'); report = {}
HOST_RX = r"assert socket\.gethostname\(\) == '([0-9a-f]+)'"
BOOT_RX = r"assert Path\('/proc/sys/kernel/random/boot_id'\)\.read_text\(\)\.strip\(\) == '([0-9a-f-]+)'"


def repin_asserts(ls):
    t = ls.read_text()
    mh = re.search(HOST_RX, t); mb = re.search(BOOT_RX, t)
    assert mh and mb, ls
    if (mh.group(1), mb.group(1)) == (OLD_HOST, OLD_BOOT):
        t = t.replace(mh.group(0), f"assert socket.gethostname() == '{host}'").replace(mb.group(0), f"assert Path('/proc/sys/kernel/random/boot_id').read_text().strip() == '{boot}'")
        put(ls, t)
        return dict(hostname=(OLD_HOST, host), boot_id=(OLD_BOOT, boot))
    assert (mh.group(1), mb.group(1)) == (host, boot), (ls, mh.group(1), mb.group(1))
    return None


# --- dense_fp8_v1 (private copy), as v17
D = ROOT / 'dense_fp8_v1'; changed = {}
c = repin_asserts(D / 'launch_scaling.py')
if c:
    changed['launch_scaling.py'] = c
st = D / 'private_moe' / 'settings.json'; raw = st.read_text(); doc = json.loads(raw)
if doc['expected_hostname'] == OLD_HOST:
    assert doc['expected_boot_id'] == OLD_BOOT and raw.count(f'"{OLD_HOST}"') == 1 and raw.count(f'"{OLD_BOOT}"') == 1
    put(st, raw.replace(f'"{OLD_HOST}"', f'"{host}"').replace(f'"{OLD_BOOT}"', f'"{boot}"'))
    changed['private_moe/settings.json'] = dict(hostname=(OLD_HOST, host), boot_id=(OLD_BOOT, boot))
else:
    assert doc['expected_hostname'] == host and doc['expected_boot_id'] == boot
inner = []
for cand in sorted((D / 'private_moe').glob('*.json')):
    if cand.name == 'settings.json': continue
    doc = json.loads(cand.read_text()); rows = doc.get('installed_files', doc) if isinstance(doc, dict) else {}
    if isinstance(rows, dict) and 'settings.json' in rows and rows['settings.json'] != sha(D / 'private_moe' / 'settings.json'):
        rows['settings.json'] = sha(D / 'private_moe' / 'settings.json'); put(cand, json.dumps(doc, indent=2, sort_keys=True) + '\n'); inner.append(cand.name)
files = ['launch_scaling.py', 'private_moe/settings.json'] + [f'private_moe/{n}' for n in inner]
for map_rel, field in (('gameplay-canary-pins.json', 'installed_files'), ('overlay-manifest.json', None)):
    path = D / map_rel; doc = json.loads(path.read_text()); rows = doc[field] if field else doc
    for f in files:
        if f in rows: rows[f] = sha(D / f)
    if map_rel == 'overlay-manifest.json': rows['gameplay-canary-pins.json'] = sha(D / 'gameplay-canary-pins.json')
    put(path, json.dumps(doc, indent=2, sort_keys=True) + '\n')
outer = json.loads((D / 'overlay-manifest.json').read_text()); bad = [n for n, d in outer.items() if not (D / n).is_file() or sha(D / n) != d]
print(f'== dense_fp8_v1 (private copy): changed={list(changed)} inner_maps={inner}; overlay verifies {len(outer)-len(bad)}/{len(outer)}; bad={bad}'); assert not bad
import py_compile; py_compile.compile(str(private(D / 'launch_scaling.py')), doraise=True)
if changed:
    l = D / 'LINEAGE.json'; doc = json.loads(l.read_text())
    doc.setdefault('deliberate_repins', []).append(dict(file=list(changed), changed=changed, when=when, reason='side pod private copy (Claude fork 5d20b23b; container ' + host + ', 2 x RTX PRO 6000 Blackwell Server Edition, the shared volume at /mnt/shared, /workspace on the container disk); host guards only; the main pod copy is unchanged'))
    put(l, json.dumps(doc, indent=2) + '\n')
report['dense_fp8_v1'] = changed or 'already on the side-pod identity'

# --- stage-2 launcher (private copy)
c = repin_asserts(S2 / 'launch_scaling_s2.py')
py_compile.compile(str(private(S2 / 'launch_scaling_s2.py')), doraise=True)
report['S2/launch_scaling_s2.py'] = c or 'already on the side-pod identity'
report['skipped'] = ['S2/launch_scaling_s2i.py (pinned to 951008c62fb8 since v16; not used by the replay path)',
                     'dense_fp8_v1 control scripts (not used by the replay path)']
put(S2 / 'REPIN_SIDEPOD_20261012.json', json.dumps(dict(host=host, boot=boot, old=(OLD_HOST, OLD_BOOT), when=when, report=report), indent=2) + '\n')
print(json.dumps(report, indent=1))
