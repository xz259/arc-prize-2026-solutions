"""select_val_next55.py (2026-10-22, Claude session 099768dc; from select_val_rbp.py d9bd7b99): the MTP refit's validation set and training
list. Validation = the user's holdout, gen110/next55 (10-21: "The holdout will use the next 55 matched"): per game one window of WIN stream
tokens drawn as select_val_rbp.py draws them (a session chosen with probability proportional to its length, the start uniform in it),
from VAL_STREAMS (extract_streams.py over the runs that played those games, the 18-h 222-game pair p18on01 / p18off01, with "games" =
next55; older transcripts are fine: the replay recomputes the teacher on the current stack). Training = every session in TRAIN_STREAMS
(the refit_r1 runs) whose game is not a next55 game (games recur across runs: excluded by game), shuffled with SEED. Writes into OUT/:
  sessions.json      replay list up to each window end (stop_tokens), for the validation export
  keep_ctx16k.json   ARC3_S2_TAPS_KEEP ranges [start - 16,384, end)
  manifest.json      per window: run, session, game, start, end, n, depth
  pack/segments.jsonl, pack/rows.jsonl   the scorer's windows (input_ids; loss_mask = role 1 on rows [1,024, WIN - 16))
  train_sessions.json + train_sessions.summary.json   the full run's replay list ({run, raw, session_id, tokens})
  game_map.json      session -> game for every session of both stream sets
usage: python3 select_val_next55.py OUT --val-streams DIR --train-streams DIR --games next55_games.txt [--win 16384] [--seed 20261022]"""
import argparse
import collections
import json
import os
import random

import numpy as np

HEAD, TAIL = 1024, 16


def game_of(raw, cache={}):
    if raw not in cache:
        g = None
        with open(raw, errors='replace') as fh:
            for line in fh:
                if line.strip().startswith('{'):
                    g = (json.loads(line).get('game_id') or '').split('-')[0]
                    break
        cache[raw] = g
    return cache[raw]


def window_mask(role, start, end):
    m = (role[start:end] == 1).astype(int)
    m[:HEAD] = 0
    m[end - start - TAIL:] = 0
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--val-streams', required=True)
    ap.add_argument('--train-streams', required=True)
    ap.add_argument('--games', required=True)
    ap.add_argument('--win', type=int, default=16384)
    ap.add_argument('--seed', type=int, default=20261022)
    a = ap.parse_args()
    games = sorted({g.strip().split('-')[0] for g in open(a.games).read().replace('\n', ',').split(',') if g.strip()})
    assert len(games) == 55, len(games)
    vs = [json.loads(l) for l in open(os.path.join(a.val_streams, 'sessions.jsonl'))]
    ts = [json.loads(l) for l in open(os.path.join(a.train_streams, 'sessions.jsonl'))]
    gm = {s['session_id']: game_of(s['raw']) for s in vs + ts}
    held = set(games)
    by_game = collections.defaultdict(list)
    for s in vs:
        if gm[s['session_id']] in held and s['n'] - 1 >= a.win:
            by_game[gm[s['session_id']]].append(s)
    missing = [g for g in games if g not in by_game]
    rng = random.Random(a.seed)
    os.makedirs(os.path.join(a.out, 'pack'), exist_ok=False)
    json.dump(gm, open(os.path.join(a.out, 'game_map.json'), 'x'), indent=0)
    sess, keep, man, seg_rows, rec_rows = [], {}, [], [], []
    for gi, g in enumerate(g for g in games if g in by_game):
        cands = by_game[g]
        s = cands[rng.choices(range(len(cands)), weights=[c['n'] for c in cands])[0]]
        start = rng.randrange(0, s['n'] - 1 - a.win + 1)
        end = start + a.win
        sess.append(dict(run=s['run'], raw=s['raw'], session_id=s['session_id'], tokens=s['tokens'], stop_tokens=end))
        keep[s['session_id']] = [[max(0, start - 16384), end]]
        man.append(dict(run=s['run'], session_id=s['session_id'], game=g, start=start, end=end, n=s['n'], depth=round(start / s['n'], 3),
                        sessions_of_game=len(cands)))
        z = np.load(os.path.join(a.val_streams, s['run'], s['session_id'] + '.npz'))
        ids, role = z['ids'], z['role']
        m = window_mask(role, start, end)
        seg_rows.append(dict(idx=gi, session_id=s['session_id'], run=s['run'], start=start, end=end, k=0, loss_tokens=int(m.sum()),
                             status='ok-window'))
        rec_rows.append(dict(input_ids=[int(x) for x in ids[start:end]], loss_mask=[int(x) for x in m]))
    json.dump(sess, open(os.path.join(a.out, 'sessions.json'), 'x'), indent=1)
    json.dump(keep, open(os.path.join(a.out, 'keep_ctx16k.json'), 'x'), indent=0)
    json.dump(dict(win=a.win, seed=a.seed, games=len(games), windows=man, games_without_a_window=missing),
              open(os.path.join(a.out, 'manifest.json'), 'x'), indent=1)
    with open(os.path.join(a.out, 'pack', 'segments.jsonl'), 'x') as f:
        f.writelines(json.dumps(r) + '\n' for r in seg_rows)
    with open(os.path.join(a.out, 'pack', 'rows.jsonl'), 'x') as f:
        f.writelines(json.dumps(r) + '\n' for r in rec_rows)
    train = [dict(run=s['run'], raw=s['raw'], session_id=s['session_id'], tokens=s['tokens']) for s in ts if gm[s['session_id']] not in held]
    drop = [s for s in ts if gm[s['session_id']] in held]
    random.Random(a.seed).shuffle(train)
    json.dump(train, open(os.path.join(a.out, 'train_sessions.json'), 'x'), indent=1)
    n_of = {s['session_id']: s['n'] for s in ts}
    summ = dict(val_windows=len(sess), val_games=len(games), games_without_a_window=missing,
                val_loss_tokens=sum(r['loss_tokens'] for r in seg_rows), replayed_tokens=sum(s['stop_tokens'] for s in sess),
                train_sessions=len(train), train_stream_tokens=sum(n_of[s['session_id']] for s in train), dropped_sessions=len(drop),
                dropped_stream_tokens=sum(s['n'] for s in drop), train_runs=dict(collections.Counter(s['run'] for s in train)),
                val_runs=dict(collections.Counter(s['run'] for s in sess)))
    json.dump(summ, open(os.path.join(a.out, 'train_sessions.summary.json'), 'x'), indent=1)
    print(json.dumps(summ))


if __name__ == '__main__':
    main()
