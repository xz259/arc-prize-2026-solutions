"""CPU test of mtp_resume_sessions.py (2026-10-13, Claude fork 5d20b23b) on synthetic inputs: seven sessions covering
done, buffered (same pod), replay after a crash, never started, failed (skipped and listed), a driver count above the
corpus count (extra segment -> replay) and a session in flight. Only a temporary directory is written.
usage: python test_resume_sessions_cpu.py"""
import json
import os
import subprocess
import sys
import tempfile

D = os.path.dirname(os.path.abspath(__file__))
T = tempfile.mkdtemp()
sid = lambda i: 'arc3-%032x' % i
ns = {1: 150000, 2: 70000, 3: 70000, 4: 10000, 5: 5000, 6: 65536, 7: 140000}
os.makedirs(T + '/streams')
with open(T + '/streams/sessions.jsonl', 'w') as f:
    for i, n in ns.items():
        f.write(json.dumps(dict(session_id=sid(i), run='r', n=n)) + '\n')
json.dump([dict(session_id=sid(i), run='r') for i in ns], open(T + '/list.json', 'w'))
with open(T + '/driver.jsonl', 'w') as f:
    for i, st, tok in ((1, 'ok', 150000), (2, 'ok', 70000), (3, 'ok', 70000), (5, 'error: no recorded turn', 1234), (6, 'ok', 65600)):
        f.write(json.dumps(dict(session_id=sid(i), status=st, stream_tokens=tok)) + '\n')
    f.write('{"session_id": "partial li')                                  # a cut last line is ignored
cons = ['%s_s%04d' % (sid(1), k) for k in range(3)] + ['%s_s0000' % sid(i) for i in (2, 3, 6, 7)]
json.dump(dict(step=100, stream=dict(consumed=cons, trained_w={'%s_s0001' % sid(3): [0, 4000]})), open(T + '/state.pt.json', 'w'))
os.makedirs(T + '/shm')
for i, k in ((2, 1), (7, 1)):
    open(T + '/shm/arc3-persistent-%s-abcd1234_s%04d.safetensors' % (sid(i), k), 'w').close()
r = subprocess.run([sys.executable, D + '/mtp_resume_sessions.py', '--sessions', T + '/list.json', '--sidecar', T + '/state.pt.json',
                    '--driver', T + '/driver.jsonl', '--stream-dir', T + '/shm', '--streams', T + '/streams', '--out', T + '/resume.json'],
                   capture_output=True, text=True)
print(r.stdout.strip(), r.stderr.strip()[-500:])
assert r.returncode == 0
out = [s['session_id'] for s in json.load(open(T + '/resume.json'))]
summ = json.load(open(T + '/resume.summary.json'))
want = [sid(i) for i in (3, 4, 6, 7)]
assert out == want, (out, want)
assert summ['done'] == 1 and summ['buffered'] == 1 and summ['failed_skipped'] == 1 and summ['failed'][0]['session_id'] == sid(5), summ
assert summ['failed_tokens'] == 1234 and summ['consumed_segments'] == 7 and summ['partial_segments'] == 1, summ
print('resume-sessions test ok (%s)' % T)
