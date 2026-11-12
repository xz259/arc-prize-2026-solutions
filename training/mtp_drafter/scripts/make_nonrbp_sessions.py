"""make_nonrbp_sessions.py (2026-10-14, Claude fork 5d20b23b; user 10-14 ~22:4x: "take the corpus and exclude the redbluepill
games ... anneal from 0.1x to 0", 2,000 steps): the session list of the fr1n anneal. The fr1 training sessions
(jobs/valrbp50v/train_sessions.json) whose game is not a redbluepill game (redbluepill252/games.txt, game ids from
jobs/valrbp50v/game_map.json), shuffled (seed), then the shortest prefix whose 16K windows (window_bounds per 65,536-row
segment, min window 2,048, as sp_fullrun.sh counts them) reach STEPS x windows per step x (1 + margin): the replay then
finishes its list close to the trainer's last step (little left over in /dev/shm), and a shortfall from failed sessions
only ends the stream a few steps early, where the cosine LR is ~0. Writes OUT (the list) and OUT.summary.json.
usage: python make_nonrbp_sessions.py OUT [--steps 2000] [--per-step 4] [--margin 0.005] [--seed 20261014]"""
import argparse
import collections
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from train_mtp_sp import window_bounds  # noqa: E402

SP = '/mnt/shared/arc3-inference/sidepod_20261012'
RBP = '/mnt/shared/arc3-inference/redbluepill252/games.txt'
STREAMS = '/workspace/arc3-inference/dflash_streams_20261011/sessions.jsonl'

ap = argparse.ArgumentParser()
ap.add_argument('out')
ap.add_argument('--steps', type=int, default=2000)
ap.add_argument('--per-step', type=int, default=4)
ap.add_argument('--margin', type=float, default=0.005)
ap.add_argument('--seed', type=int, default=20261014)
ap.add_argument('--window', type=int, default=16384)
ap.add_argument('--num-steps', type=int, default=4)
ap.add_argument('--min-window', type=int, default=2048)
a = ap.parse_args()
rbp = {t.strip() for t in open(RBP).read().split(',') if t.strip()}
gm = json.load(open(SP + '/jobs/valrbp50v/game_map.json'))
train = json.load(open(SP + '/jobs/valrbp50v/train_sessions.json'))
n = {}
for line in open(STREAMS):
    d = json.loads(line)
    n[d['session_id']] = int(d.get('n') or 0)


def windows(sid):
    m = n.get(sid, 0)
    return sum(len(window_bounds(min(65536, m - s0), a.window, a.num_steps, a.min_window)) for s0 in range(0, m, 65536))


keep = [s for s in train if gm.get(s['session_id']) not in rbp]
random.Random(a.seed).shuffle(keep)
need = int(a.steps * a.per_step * (1 + a.margin)) + 1
out, nw, tok = [], 0, 0
for s in keep:
    if nw >= need:
        break
    out.append(s)
    nw += windows(s['session_id'])
    tok += n.get(s['session_id'], 0)
if nw < need:
    raise SystemExit('the non-redbluepill corpus has only %d windows, %d needed' % (nw, need))
json.dump(out, open(a.out, 'x'), indent=0)
summ = dict(sessions=len(out), windows=nw, windows_needed=need, stream_tokens=tok, steps=a.steps, per_step=a.per_step,
            seed=a.seed, nonrbp_sessions_available=len(keep), rbp_sessions_excluded=len(train) - len(keep),
            runs=dict(collections.Counter(s['run'] for s in out)), games=len({gm.get(s['session_id']) for s in out}))
json.dump(summ, open(a.out + '.summary.json', 'x'), indent=1)
print(json.dumps(summ))
