"""select_val_rbp.py (2026-10-13, Claude fork 5d20b23b): the MTP full run's validation set (user 10-13 03:4x: "take 50 games
from the redbluepill set; reserve the other, higher-quality games for training"; the untrained baseline is scored on it at
the full run's start). 50 redbluepill games drawn at random (seed) among the redbluepill games that have corpus v7 sessions
long enough for a window; per game one window of WIN stream tokens, as select_val_v7.py draws them (a session chosen with
probability proportional to its length, the window start uniform in the session). Every session of the 50 games leaves the
full run's training list. Writes into OUT/:
  sessions.json      replay list up to each window end (stop_tokens), for the MTP export
  keep_ctx16k.json   ARC3_S2_TAPS_KEEP ranges [start - 16,384, end) (16K of context for the MTP's attention)
  manifest.json      per window: run, session, game, start, end, n, depth
  pack/segments.jsonl, pack/rows.jsonl   the evaluator's windows in the cache_val50 format: input_ids, and loss_mask = stream
                     role 1 on window rows [HEAD 1,024, WIN - 16) (the val50 pack's anchor-free head and last DFlash block)
  train_sessions.json   the full run's replay list: corpus v7 (Phase 6 sessions_driver.json, no chunk indices) minus every
                     session of the 50 games, shuffled; and train_sessions.summary.json
  game_map.json      session -> game for all corpus v7 sessions
Self-check (before writing anything): this rule reproduces the old val50 pack's masks and ids for its first 5 windows.
usage: python3 select_val_rbp.py OUT [--n 50] [--win 16384] [--seed 20261013]"""
import argparse
import collections
import json
import os
import random

import numpy as np

P6 = '/workspace/arc3-inference/dflash_phase6_20261011v/sessions_driver.json'
ST = '/workspace/arc3-inference/dflash_streams_20261011'
RBP = '/mnt/shared/arc3-inference/redbluepill252/games.txt'
OLD = '/workspace/arc3-inference/dflash_packed_20261011/cache_val50'


def game_of(raw):
    with open(raw) as fh:
        for line in fh:
            if line.strip().startswith('{'):
                return json.loads(line).get('game_id')
    return None


def stream(run, sid):
    z = np.load(os.path.join(ST, run, sid + '.npz'))
    return z['ids'], z['role']


HEAD, TAIL = 1024, 16


def window_mask(role, start, end):
    m = (role[start:end] == 1).astype(int)
    m[:HEAD] = 0
    m[end - start - TAIL:] = 0
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--win', type=int, default=16384)
    ap.add_argument('--seed', type=int, default=20261013)
    a = ap.parse_args()
    segs = [json.loads(l) for l in open(OLD + '/segments.jsonl')]
    recs = [json.loads(l) for l in open(OLD + '/rows.jsonl')]
    for sg, rc in list(zip(segs, recs))[:5]:                 # the mask rule against the old val50 pack
        ids, role = stream(sg['run'], sg['session_id'])
        assert list(ids[sg['start']:sg['end']]) == rc['input_ids'], 'ids'
        assert list(window_mask(role, sg['start'], sg['end'])) == rc['loss_mask'], 'mask convention'
    os.makedirs(os.path.join(a.out, 'pack'), exist_ok=False)
    items = json.load(open(P6))
    n_of = {}
    for line in open(ST + '/sessions.jsonl'):
        d = json.loads(line)
        n_of[d['session_id']] = d['n']
    gm = {it['session_id']: game_of(it['raw']) for it in items}
    json.dump(gm, open(os.path.join(a.out, 'game_map.json'), 'x'), indent=0)
    rbp = {t.strip() for t in open(RBP).read().split(',') if t.strip()}
    by_game = collections.defaultdict(list)
    for it in items:
        n = it.get('stop_tokens') or n_of[it['session_id']]
        if gm[it['session_id']] in rbp and n - 1 >= a.win:
            by_game[gm[it['session_id']]].append((it, n))
    rng = random.Random(a.seed)
    games = sorted(by_game)
    chosen = sorted(rng.sample(games, a.n))
    sess, keep, man, seg_rows, rec_rows = [], {}, [], [], []
    for gi, g in enumerate(chosen):
        cands = by_game[g]
        it, n = cands[rng.choices(range(len(cands)), weights=[c[1] for c in cands])[0]]
        start = rng.randrange(0, n - 1 - a.win + 1)
        end = start + a.win
        sess.append(dict(run=it['run'], raw=it['raw'], session_id=it['session_id'], tokens=it['tokens'], stop_tokens=end))
        keep[it['session_id']] = [[max(0, start - 16384), end]]
        man.append(dict(run=it['run'], session_id=it['session_id'], game=g, start=start, end=end, n=n,
                        depth=round(start / n, 3), sessions_of_game=len(cands)))
        ids, role = stream(it['run'], it['session_id'])
        m = window_mask(role, start, end)
        seg_rows.append(dict(idx=gi, session_id=it['session_id'], run=it['run'], start=start, end=end, k=0,
                             loss_tokens=int(m.sum()), status='ok-window'))
        rec_rows.append(dict(input_ids=[int(x) for x in ids[start:end]], loss_mask=[int(x) for x in m]))
    json.dump(sess, open(os.path.join(a.out, 'sessions.json'), 'x'), indent=1)
    json.dump(keep, open(os.path.join(a.out, 'keep_ctx16k.json'), 'x'), indent=0)
    json.dump(dict(win=a.win, seed=a.seed, n=a.n, rbp_games_eligible=len(games), windows=man),
              open(os.path.join(a.out, 'manifest.json'), 'x'), indent=1)
    with open(os.path.join(a.out, 'pack', 'segments.jsonl'), 'x') as f:
        f.writelines(json.dumps(r) + '\n' for r in seg_rows)
    with open(os.path.join(a.out, 'pack', 'rows.jsonl'), 'x') as f:
        f.writelines(json.dumps(r) + '\n' for r in rec_rows)
    held = set(chosen)
    train = [{k: v for k, v in it.items() if k != 'first_chunk'} for it in items if gm[it['session_id']] not in held]
    drop = [it for it in items if gm[it['session_id']] in held]
    random.Random(a.seed).shuffle(train)
    json.dump(train, open(os.path.join(a.out, 'train_sessions.json'), 'x'), indent=1)
    tok = lambda L: sum(it.get('stop_tokens') or n_of[it['session_id']] for it in L)
    summ = dict(val_windows=len(sess), val_games=len(chosen), rbp_games_eligible=len(games),
                val_loss_tokens=sum(r['loss_tokens'] for r in seg_rows), replayed_tokens=sum(s['stop_tokens'] for s in sess),
                train_sessions=len(train), train_stream_tokens=tok(train), dropped_sessions=len(drop),
                dropped_stream_tokens=tok(drop), runs_of_dropped=dict(collections.Counter(it['run'] for it in drop)))
    json.dump(summ, open(os.path.join(a.out, 'train_sessions.summary.json'), 'x'), indent=1)
    print(json.dumps(summ))


if __name__ == '__main__':
    main()
