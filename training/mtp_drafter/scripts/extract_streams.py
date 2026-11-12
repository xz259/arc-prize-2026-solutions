"""extract_streams.py (2026-10-10, Phase 6 streaming): one pass over the training runs' token logs -> one compact file per
session, so per-chunk packing never rereads the logs. Per session (<out>/<run>/<session_id>.npz):
  ids    int32 [n]  the stream (submit.appended_token_ids then pause.output_token_ids, per operation in order)
  role   int8  [n]  0 context, 1 assistant target (training mask train=true for that pause's revision), 2 assistant excluded
  op_end int64 [k]  stream length after each completed operation (the driver's `replayed` count k truncates the stream)
  ev_l, ev_r int64  evict_commit logical_tokens and removed_tokens (served positions: t - sum r for l - 1 <= t)
plus <out>/sessions.jsonl: {run, session_id, raw, tokens, n, ops} for every session with at least one completed operation
(raw = the session's raw-harness game file, needed by the replay driver).
usage: extract_streams.py TRAINING_RUNS.json OUT_DIR [--truncate TRUNCATIONS.json]
2026-10-22 (MTP refit, Claude session 099768dc): S2 = fin73/s2 on 2a159825c76b; a run may carry "tree" (the stage-2 tree holding its
client_mtp/<label>/raw-harness; default S2) and "games" (a list of game ids, or a path to a comma / newline separated list: only
sessions whose raw-harness game file belongs to one of them are extracted; the token-log pass skips the others before parsing).
2026-10-11: --truncate (corpus_truncations_*.json) keeps only the first keep_responses completed operations of each listed
session (and the evictions inside them); sessions.jsonl records truncated_from_ops."""
import collections
import glob
import gzip
import json
import os
import re
import sys

import numpy as np

S2 = '/workspace/kvring_dev/fin73/s2'
runs_doc = json.load(open(sys.argv[1]))
runs = runs_doc['runs'] if isinstance(runs_doc, dict) else runs_doc
runs = [r for r in runs if r['label'] not in ('v62a167p', 'v62a167q')]       # short test runs, not trained on
out = sys.argv[2]
trunc = {}
if len(sys.argv) > 4 and sys.argv[3] == '--truncate':
    for g in json.load(open(sys.argv[4]))['truncated_games'].values():
        for lab_, ss in g['sessions'].items():
            for sid_, s in ss.items():
                trunc[(lab_, sid_)] = int(s['keep_responses'])
os.makedirs(out, exist_ok=False)
index = open(os.path.join(out, 'sessions.jsonl'), 'x')
sid_re = re.compile(r'"session_id": ?"([^"]+)"')
tot = collections.Counter()
for r in runs:
    lab = r['label']
    tokens = '%s/tokens/%s.jsonl.gz' % (r['corpus'], lab)
    mask = '%s/index/%s/train_20261007.jsonl.gz' % (r['corpus'], lab)
    train = {}
    with gzip.open(mask, 'rt') as f:
        for line in f:
            d = json.loads(line)
            train[(d['session_id'], d['revision'])] = bool(d['train'])
    games = r.get('games')
    if isinstance(games, str):
        games = [g.strip() for g in open(games).read().replace('\n', ',').split(',') if g.strip()]
    games = {g.split('-')[0] for g in games} if games else None
    raw = {}
    for p in sorted(glob.glob('%s/client_mtp/%s/raw-harness/game-*.jsonl' % (r.get('tree', S2), lab))):
        if games is not None:
            g0 = None
            for line in open(p, errors='replace'):
                if line.strip().startswith('{'):
                    g0 = (json.loads(line).get('game_id') or '').split('-')[0]
                    break
            if g0 not in games:
                continue
        for line in open(p, errors='replace'):
            if '"persistent-request"' in line[:120] and '"/arc3/persistent/start"' in line:
                sid = json.loads(line)['payload']['body'].get('session_id')
                if sid:
                    raw[sid] = p
    ops = collections.defaultdict(list)
    evs = collections.defaultdict(list)
    with gzip.open(tokens, 'rt') as f:
        for line in f:
            m = sid_re.search(line)
            if not m:
                continue
            if games is not None and m.group(1) not in raw:
                continue
            if '"submit"' not in line[:300] and '"pause"' not in line[:300] and '"evict_commit"' not in line[:300]:
                continue
            d = json.loads(line)
            ev = d.get('event')
            if ev in ('submit', 'pause'):
                ops[d['session_id']].append(d)
            elif ev == 'evict_commit':
                plan = d.get('plan') or {}
                if plan.get('removed_tokens'):
                    evs[d['session_id']].append((int(plan['logical_tokens']), int(plan['removed_tokens'])))
    os.makedirs(os.path.join(out, lab), exist_ok=True)
    for sid, lst in ops.items():
        ids, role, op_end = [], [], []
        pending = None
        for d in lst:
            if d['event'] == 'submit':
                pending = d
            elif pending is not None:
                a, o = pending['appended_token_ids'], d['output_token_ids']
                ids += a
                role += [0] * len(a)
                ids += o
                role += [1 if train.get((sid, d.get('revision')), False) else 2] * len(o)
                op_end.append(len(ids))
                pending = None
        full_ops = len(op_end)
        k = trunc.get((lab, sid))
        if k is not None and len(op_end) > k:                    # cut at the listed operation (winning point)
            n = op_end[k - 1]
            tot['truncated_tokens'] += len(ids) - n
            tot['truncated_sessions'] += 1
            ids, role, op_end = ids[:n], role[:n], op_end[:k]
            evs[sid] = [e for e in evs[sid] if e[0] <= n]
        if not op_end or sid not in raw:
            tot['skipped_sessions'] += 1
            continue
        el = [e[0] for e in evs[sid]]
        er = [e[1] for e in evs[sid]]
        np.savez(os.path.join(out, lab, sid + '.npz'), ids=np.array(ids, dtype=np.int32), role=np.array(role, dtype=np.int8),
                 op_end=np.array(op_end, dtype=np.int64), ev_l=np.array(el, dtype=np.int64), ev_r=np.array(er, dtype=np.int64))
        index.write(json.dumps(dict(run=lab, session_id=sid, raw=raw[sid], tokens=tokens, n=len(ids), ops=len(op_end),
                                    **({'truncated_from_ops': full_ops} if full_ops != len(op_end) else {}))) + '\n')
        tot['sessions'] += 1
        tot['tokens'] += len(ids)
        tot['run:' + lab] += len(ids)
    print(lab, tot['run:' + lab], flush=True)
index.close()
print(json.dumps(dict(tot)))
