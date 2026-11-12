"""reshard_ple.py SRC_VIEW OUT_DIR [--per-file N] [--limit-files K] (2026-10-24, Claude main session 099768dc; PLE_RESHARD_PLAN_2026-10-15.md
step 1; the user 10-24: "Sure ... Can we also reshard the model to speed up the vLLM server start time?"): the target view with its PLE
table (one 102.4 GB safetensors file, model-00016, 128 ngram_embedding shard tensors) split into files of N tensors each (default 7,
~5.6 GB), every other file linked unchanged (symlinks to the same targets as the source view), a new model.safetensors.index.json and
ARC3_TARGET_VIEW.json. Byte-exact: each output tensor's bytes are copied from the source range (pread / write, 64 MiB chunks) and
sha256-hashed on the way; a verification pass re-reads every output tensor and compares its sha256, dtype, shape and length with the
source's. Page cache bounded: posix_fadvise(DONTNEED) on the copied source ranges and on each output file after fsync. Run it at idle
priority (ionice -c3 nice -n 19). --limit-files K writes only the first K output files (a test; no index or view file).
Exit 0 only when every check passes; the report is OUT_DIR/RESHARD_REPORT.json."""
import argparse
import hashlib
import json
import os
import struct
import sys
import time

CHUNK = 64 << 20
ap = argparse.ArgumentParser()
ap.add_argument('src')
ap.add_argument('out')
ap.add_argument('--per-file', type=int, default=7)
ap.add_argument('--limit-files', type=int, default=0)
a = ap.parse_args()
SRC, OUT = os.path.abspath(a.src), os.path.abspath(a.out)
assert os.path.isdir(SRC), SRC
assert not os.path.exists(OUT) or not os.listdir(OUT), 'OUT_DIR must be new or empty: %s' % OUT
os.makedirs(OUT, exist_ok=True)
ix = json.load(open(os.path.join(SRC, 'model.safetensors.index.json')))
wm = ix['weight_map']
big = {}
for k, f in wm.items():
    if '.ple.ple_embedding.ngram_embedding.' in k:
        big.setdefault(f, []).append(k)
assert len(big) == 1, 'expected one PLE table file: %s' % sorted(big)
(PFILE, pnames), = big.items()
src_path = os.path.join(SRC, PFILE)


def header(path):
    with open(path, 'rb') as fh:
        n = struct.unpack('<Q', fh.read(8))[0]
        h = json.loads(fh.read(n))
    return h, 8 + n


H, base = header(src_path)
meta = H.pop('__metadata__', None)
assert set(H) == set(pnames), ('the PLE file holds other tensors too', sorted(set(H) - set(pnames))[:5])
order = sorted(H, key=lambda k: H[k]['data_offsets'][0])          # the source's byte order
groups = [order[i:i + a.per_file] for i in range(0, len(order), a.per_file)]
nf = len(groups)
names = ['model-ple-%05d-of-%05d.safetensors' % (i + 1, nf) for i in range(nf)]
if a.limit_files:
    groups, names = groups[:a.limit_files], names[:a.limit_files]
print('source %s: %d tensors, %.1f GB; %d output files of <= %d tensors%s' % (
    PFILE, len(order), (H[order[-1]]['data_offsets'][1]) / 1e9, nf, a.per_file, (' (test: first %d)' % len(groups)) if a.limit_files else ''),
      flush=True)
report = dict(source_view=SRC, ple_file=PFILE, tensors=len(order), per_file=a.per_file, files=nf, limit_files=a.limit_files, outputs={},
              started=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
src_hash = {}
sfd = os.open(src_path, os.O_RDONLY)
t0 = time.time()
for name, grp in zip(names, groups):
    hdr, off = {}, 0
    for k in grp:
        b, e = H[k]['data_offsets']
        hdr[k] = dict(dtype=H[k]['dtype'], shape=H[k]['shape'], data_offsets=[off, off + (e - b)])
        off += e - b
    if meta is not None:
        hdr['__metadata__'] = meta
    hj = json.dumps(hdr, separators=(',', ':')).encode()
    hj += b' ' * ((8 - (len(hj) % 8)) % 8)                            # the data section starts 8-byte aligned
    dst = os.path.join(OUT, name)
    dfd = os.open(dst + '.tmp', os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    os.write(dfd, struct.pack('<Q', len(hj)) + hj)
    for k in grp:
        b, e = H[k]['data_offsets']
        h = hashlib.sha256()
        pos, end = base + b, base + e
        while pos < end:
            buf = os.pread(sfd, min(CHUNK, end - pos), pos)
            assert buf, 'short read at %d' % pos
            h.update(buf)
            mv = memoryview(buf)
            while mv:
                w = os.write(dfd, mv)
                mv = mv[w:]
            os.posix_fadvise(sfd, pos, len(buf), os.POSIX_FADV_DONTNEED)
            pos += len(buf)
        src_hash[k] = h.hexdigest()
    os.fsync(dfd)
    os.posix_fadvise(dfd, 0, 0, os.POSIX_FADV_DONTNEED)
    os.close(dfd)
    os.replace(dst + '.tmp', dst)
    report['outputs'][name] = dict(tensors=len(grp), bytes=os.path.getsize(dst))
    print('[%s] wrote %s: %d tensors, %.2f GB, %.0f s since start' % (time.strftime('%H:%M:%S', time.gmtime()), name, len(grp),
                                                                     os.path.getsize(dst) / 1e9, time.time() - t0), flush=True)
os.close(sfd)

# verification: re-read every output tensor
bad = []
for name, grp in zip(names, groups):
    oh, obase = header(os.path.join(OUT, name))
    oh.pop('__metadata__', None)
    if set(oh) != set(grp):
        bad.append((name, 'tensor set'))
        continue
    fd = os.open(os.path.join(OUT, name), os.O_RDONLY)
    for k in grp:
        b, e = oh[k]['data_offsets']
        if (oh[k]['dtype'], oh[k]['shape'], e - b) != (H[k]['dtype'], H[k]['shape'], H[k]['data_offsets'][1] - H[k]['data_offsets'][0]):
            bad.append((k, 'meta'))
            continue
        h, pos, end = hashlib.sha256(), obase + b, obase + e
        while pos < end:
            buf = os.pread(fd, min(CHUNK, end - pos), pos)
            h.update(buf)
            os.posix_fadvise(fd, pos, len(buf), os.POSIX_FADV_DONTNEED)
            pos += len(buf)
        if h.hexdigest() != src_hash[k]:
            bad.append((k, 'sha256'))
    os.close(fd)
report['verify'] = dict(checked=sum(len(g) for g in groups), bad=bad)
report['sha256'] = src_hash
if not a.limit_files and not bad:
    for f in sorted(set(wm.values()) - {PFILE}) + sorted(x for x in os.listdir(SRC) if not x.endswith('.safetensors')
                                                          and x not in ('model.safetensors.index.json', 'ARC3_TARGET_VIEW.json')):
        os.symlink(os.path.realpath(os.path.join(SRC, f)), os.path.join(OUT, f))
    nwm = {k: f for k, f in wm.items() if f != PFILE}
    for name, grp in zip(names, groups):
        for k in grp:
            nwm[k] = name
    assert set(nwm) == set(wm) and len(nwm) == len(wm)
    nix = dict(ix, weight_map=dict(sorted(nwm.items())))
    nix.setdefault('metadata', {})['total_size'] = ix.get('metadata', {}).get('total_size')
    with open(os.path.join(OUT, 'model.safetensors.index.json.tmp'), 'w') as fh:
        json.dump(nix, fh, indent=2)
    os.replace(os.path.join(OUT, 'model.safetensors.index.json.tmp'), os.path.join(OUT, 'model.safetensors.index.json'))
    view = json.load(open(os.path.join(SRC, 'ARC3_TARGET_VIEW.json')))
    view.update(kind='intel-ple%d' % nf, resharded_from=SRC,
                change='%s (the PLE table, %d tensors) split into %d files of <= %d tensors (reshard_ple.py; byte-exact, sha256-verified)' % (
                    PFILE, len(order), nf, a.per_file))
    with open(os.path.join(OUT, 'ARC3_TARGET_VIEW.json.tmp'), 'w') as fh:
        json.dump(view, fh, indent=1)
    os.replace(os.path.join(OUT, 'ARC3_TARGET_VIEW.json.tmp'), os.path.join(OUT, 'ARC3_TARGET_VIEW.json'))
report['finished'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
report['seconds'] = round(time.time() - t0, 1)
with open(os.path.join(OUT, 'RESHARD_REPORT.json'), 'w') as fh:
    json.dump(report, fh, indent=1)
print('verify: %d tensors checked, %d bad%s; %.0f s' % (report['verify']['checked'], len(bad), (': %s' % bad[:5]) if bad else '', report['seconds']))
sys.exit(1 if bad else 0)
