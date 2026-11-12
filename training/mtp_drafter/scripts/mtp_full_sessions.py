"""mtp_full_sessions.py (2026-10-13, Claude fork 5d20b23b): the MTP full run's replay list (user 10-13 03:1x: the whole corpus
v7, one pass, minus every session of the 50 val50 games so val50 stays an unseen-game measure). Phase 6's
sessions_driver.json (2,595 corpus v7 sessions) without its chunk indices (the MTP stream gates on RAM only), the sessions
whose game (the raw-harness log's game_id) is a val50 game dropped, shuffled (seed) so concurrent sessions mix runs and
games; stream tokens per session from the Phase 6 streams index. Writes OUT (the driver's SESSIONS.json) and OUT.summary.json.
usage: python3 mtp_full_sessions.py OUT [--seed 20261013]"""
import argparse
import collections
import json
import random

P6 = '/workspace/arc3-inference/dflash_phase6_20261011v/sessions_driver.json'
VAL = '/workspace/arc3-inference/dflash_packed_20261011/cache_val50/segments.jsonl'
STREAMS = '/workspace/arc3-inference/dflash_streams_20261011/sessions.jsonl'


def game_of(raw):
    with open(raw) as fh:
        for line in fh:
            if line.strip().startswith('{'):
                d = json.loads(line)
                for k in ('game_id', 'game', 'env_id', 'task_id'):
                    if k in d:
                        return str(d[k])
                return None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('out')
    ap.add_argument('--seed', type=int, default=20261013)
    a = ap.parse_args()
    items = json.load(open(P6))
    n = {}
    for line in open(STREAMS):
        d = json.loads(line)
        n[d['session_id']] = d['n']
    raw_of = {it['session_id']: it['raw'] for it in items}
    val_sids = [json.loads(l)['session_id'] for l in open(VAL)]
    val_games = {game_of(raw_of[s]) for s in val_sids}
    assert None not in val_games and len(val_games) == 50, len(val_games)
    keep, drop = [], []
    for it in items:
        it = {k: v for k, v in it.items() if k != 'first_chunk'}
        (drop if game_of(it['raw']) in val_games else keep).append(it)
    random.Random(a.seed).shuffle(keep)
    json.dump(keep, open(a.out, 'x'), indent=1)
    runs = collections.Counter()
    for it in keep:
        runs[it['run']] += n[it['session_id']]
    summ = dict(sessions=len(keep), stream_tokens=sum(n[it['session_id']] for it in keep), dropped_sessions=len(drop),
                dropped_stream_tokens=sum(n[it['session_id']] for it in drop), val_games=len(val_games), seed=a.seed,
                runs=dict(sorted(runs.items())))
    json.dump(summ, open(a.out + '.summary.json', 'x'), indent=1)
    print(json.dumps({k: v for k, v in summ.items() if k != 'runs'}))


if __name__ == '__main__':
    main()
