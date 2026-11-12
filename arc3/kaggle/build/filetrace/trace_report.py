"""trace_report.py TRACE_DIR [OUT.json [EXTRA_OUTPUT_PREFIX]] (2026-10-24, Claude main session 099768dc; the Kaggle port's file tracer): merges the traces that
filetrace/sitecustomize.py wrote (trace-<pid>.txt) and any /proc maps snapshots (maps-*.txt: one mapped path per line) into the file
set a launch reads. Each path is resolved through symlinks and classified as runtime (the runtime and venvs archives, the standalone
CPython), model (the model folders), output (written by the run: fin74/runs, s2 instances / receipts / client_mtp / nll-*, the server
logs, /tmp, /proc, /dev), cache (the compile / AOT / runtime caches and the bytecode cache), home (under /root: caches read from HOME),
system (/usr, /lib, /etc outside the archives) or code (everything else under /workspace and /mnt/shared). Code and cache files go into
the Kaggle code bundle. Prints the counts per class and per top directory; OUT.json gets the full lists (path as opened -> resolved).
22:1x fix: the production overlay and the chat template live under arc3-inference/runs/, which is code, not run output."""
import collections
import glob
import json
import os
import sys

D = sys.argv[1]
RUNTIME = ('/mnt/shared/arc3-inference/runtime/', '/mnt/shared/arc3-inference/venv-runtime/', '/mnt/shared/arc3-inference/venv-duck/',
           '/mnt/shared/arc3-inference/python/')
MODEL = ('/mnt/shared/models/', '/workspace/models/')
S2 = '/workspace/kvring_dev/fin74/s2/'
OUTPUT_PREFIX = ('/tmp/', '/proc/', '/dev/', '/sys/', '/run/', '/var/', '/kaggle/working/', '/workspace/kvring_dev/fin74/runs/',
                 S2 + 'instances/', S2 + 'receipts/', S2 + 'client_mtp/', S2 + 'nll-', '/workspace/arc3-inference/logs/',
                 '/mnt/shared/arc3-inference/logs/')
CACHE_PREFIX = (S2 + 'aot-cache-', S2 + 'compiled-cache-', '/workspace/arc3-inference/cache/', '/mnt/shared/arc3-inference/cache/',
                '/workspace/kvring_dev/fin74/pycache/')

events = collections.defaultdict(set)              # resolved path -> kinds
opened_as = {}
procs = []
for f in sorted(glob.glob(os.path.join(D, 'trace-*.txt'))):
    for line in open(f, errors='replace'):
        line = line.rstrip('\n')
        if line.startswith('#argv'):
            procs.append(line[6:][:200])
            continue
        kind, _, path = line.partition(' ')
        real = os.path.realpath(path)
        events[real].add(kind)
        opened_as.setdefault(real, path)
for f in sorted(glob.glob(os.path.join(D, 'maps-*.txt'))):
    for line in open(f, errors='replace'):
        p = line.strip()
        if p.startswith('/'):
            real = os.path.realpath(p)
            events[real].add('mapped')
            opened_as.setdefault(real, p)


def klass(p):
    if p.startswith(RUNTIME) or any(p.startswith(r.rstrip('/')) and p == r.rstrip('/') for r in RUNTIME):
        return 'runtime'
    if p.startswith(MODEL):
        return 'model'
    if p.startswith(OUTPUT_PREFIX) or (len(sys.argv) > 3 and p.startswith(sys.argv[3])):
        return 'output'
    if p.startswith(CACHE_PREFIX):
        return 'cache'
    if p.startswith('/root/'):
        return 'home'
    if p.startswith(('/workspace/', '/mnt/shared/')):
        return 'code'
    return 'system'


cls = collections.defaultdict(list)
for p in sorted(events):
    if os.path.isdir(p):
        continue
    cls[klass(p)].append(p)
print('trace %s: %d processes, %d paths (files only): %s' % (D, len(procs), sum(len(v) for v in cls.values()),
                                                         {k: len(v) for k, v in sorted(cls.items())}))
top = collections.Counter()
for p in cls['code']:
    parts = p.split('/')
    top['/'.join(parts[:5])] += 1
print('code files by directory:')
for k, n in top.most_common(40):
    print('  %5d  %s' % (n, k))
missing = [p for p in cls['code'] if not os.path.exists(p)]
print('code paths that do not exist now (written and removed, or probed): %d' % len(missing))
sysx = collections.Counter('/'.join(p.split('/')[:3]) for p in cls['system'])
print('system paths by prefix: %s' % dict(sysx.most_common(12)))
if len(sys.argv) > 2:
    out = dict(trace_dir=D, processes=procs, classes={k: [dict(path=p, opened_as=opened_as.get(p), kinds=sorted(events[p])) for p in v]
                                                      for k, v in cls.items()})
    with open(sys.argv[2] + '.tmp', 'w') as fh:
        json.dump(out, fh, indent=1)
    os.replace(sys.argv[2] + '.tmp', sys.argv[2])
    print('wrote', sys.argv[2])
