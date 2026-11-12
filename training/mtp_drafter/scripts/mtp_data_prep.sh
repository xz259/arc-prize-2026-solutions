#!/bin/bash
# mtp_data_prep.sh (2026-10-22, MTP refit; Claude session 099768dc): the refit's data on the main pod 2a159825c76b, CPU only, after the
# ctx78 corpus chain (manifest v8 with refit_r1 on the volume):
#   1. TRAINING_RUNS.json from manifest v8's refit_r1 runs: corpus = the run's corpus on disk, tree = the stage-2 tree holding its
#      client_mtp/<label>/raw-harness (checked);
#   2. extract_streams.py -> STREAMS (/workspace/mtp_refit_r1/streams);
#   3. the holdout's source: the 18-h 222-game pair p18on01 / p18off01 restored from archive_20261018/p18_222 (corpus + raw-harness) into
#      /workspace/restore_p18 (once), then extract_streams.py over them with games = next55 -> VSTREAMS (/workspace/mtp_refit_r1/streams_val);
#   4. select_val_next55.py -> SP/jobs/valnext55 (validation windows, keep ranges, pack, the shuffled training list).
# Log lines '[HH:MM:SS] STEP n ...', ending 'MTP DATA PREP DONE' or 'MTP DATA PREP FAILED: ...'.
# usage: setsid nohup bash mtp_data_prep.sh > /workspace/mtp_refit_r1/data_prep.log 2>&1 < /dev/null &
set -u
[ "$(hostname)" = 2a159825c76b ] || { echo "not the main pod"; exit 9; }
SP=/mnt/shared/arc3-inference/mtp_refit_20261022
W=/workspace/mtp_refit_r1
MAN=/mnt/shared/arc3-inference/corpus_m2_collection_20261019/manifest.json
P18=/mnt/shared/arc3-inference/archive_20261018/p18_222
RP=/workspace/restore_p18
log() { echo "[$(date -u +%T)] $*"; }
fail() { log "MTP DATA PREP FAILED: $*"; exit 1; }
mkdir -p $W
python3 -c "import json; m = json.load(open('$MAN')); assert 'refit_r1' in m, 'no refit_r1'; print(m['by'], m['written_utc'])" || fail "manifest v8 with refit_r1 missing"
log "STEP 1 TRAINING_RUNS.json from $MAN"
python3 - $MAN $W/training_runs.json <<'PYEOF' || fail "training runs"
import json, os, sys
m = json.load(open(sys.argv[1]))
rows = {r['run']: r for r in m['runs']}
TREE = {'kag71': '/workspace/kvring_dev/kag71/s2', 'hab70': '/workspace/kvring_dev/hab70/s2', 'rnk72': '/workspace/kvring_dev/rnk72/s2',
        'fin73': '/workspace/kvring_dev/fin73/s2', 'ctx74': '/workspace/kvring_dev/fin73/s2', 'ctx78': '/workspace/kvring_dev/fin73/s2',
        'ctx75': '/workspace/restore_final_20261021/kvring_dev/fin73/s2', 'rex69': '/workspace/kvring_dev/rex69/s2', 'rab69': '/workspace/kvring_dev/rab69/s2'}
out = []
for run in m['refit_r1']['runs']:
    r = rows[run]
    corpus = r['corpus_disk']
    tree = TREE[run[:5]]
    for p in ('%s/tokens/%s.jsonl.gz' % (corpus, run), '%s/index/%s/train_20261007.jsonl.gz' % (corpus, run), '%s/client_mtp/%s/raw-harness' % (tree, run)):
        assert os.path.exists(p), 'missing %s' % p
    out.append(dict(label=run, corpus=corpus, tree=tree, train_tokens=r['train_tokens']))
json.dump(dict(runs=out, refit_r1=m['refit_r1']), open(sys.argv[2], 'w'), indent=1)
print('%d runs, %.2fM training tokens: %s' % (len(out), sum(o['train_tokens'] for o in out) / 1e6, ' '.join(o['label'] for o in out)))
PYEOF
log "STEP 2 extract_streams -> $W/streams"
[ -e $W/streams ] && fail "$W/streams exists"
nice -n 5 python3 -B $SP/scripts/extract_streams.py $W/training_runs.json $W/streams > $W/extract_streams.log 2>&1 || fail "extract_streams: $(tail -n 3 $W/extract_streams.log | tr '\n' ' ')"
log "STEP 2 done: $(tail -n 1 $W/extract_streams.log | cut -c1-300)"
log "STEP 3 p18 restore + holdout streams"
if [ ! -d $RP/corpus_p18_20261018/tokens ]; then
  [ -e $RP ] && fail "$RP exists without the p18 corpus"
  mkdir -p $RP
  for a in corpus_p18_20261018 client_mtp_p18on01 client_mtp_p18off01; do
    nice -n 10 tar -C $RP -I zstd -xf $P18/$a.tar.zst || fail "restore $a rc $?"
  done
fi
for L in p18on01 p18off01; do [ -d $RP/client_mtp/$L/raw-harness ] && [ -s $RP/corpus_p18_20261018/tokens/$L.jsonl.gz ] || fail "p18 restore incomplete: $L"; done
python3 - $RP $SP/jobs/next55_games.txt $W/val_runs.json <<'PYEOF' || fail "val runs"
import json, sys
rp, games, out = sys.argv[1:4]
json.dump(dict(runs=[dict(label=L, corpus=rp + '/corpus_p18_20261018', tree=rp, games=games) for L in ('p18on01', 'p18off01')]), open(out, 'w'), indent=1)
PYEOF
[ -e $W/streams_val ] && fail "$W/streams_val exists"
nice -n 5 python3 -B $SP/scripts/extract_streams.py $W/val_runs.json $W/streams_val > $W/extract_streams_val.log 2>&1 || fail "extract_streams val: $(tail -n 3 $W/extract_streams_val.log | tr '\n' ' ')"
log "STEP 3 done: $(tail -n 1 $W/extract_streams_val.log | cut -c1-300)"
log "STEP 4 select_val_next55 -> $SP/jobs/valnext55"
[ -e $SP/jobs/valnext55 ] && fail "$SP/jobs/valnext55 exists"
nice -n 5 python3 -B $SP/scripts/select_val_next55.py $SP/jobs/valnext55 --val-streams $W/streams_val --train-streams $W/streams \
  --games $SP/jobs/next55_games.txt > $W/select_val.log 2>&1 || fail "select_val_next55: $(tail -n 3 $W/select_val.log | tr '\n' ' ')"
log "STEP 4 done: $(tail -n 1 $W/select_val.log | cut -c1-400)"
log "MTP DATA PREP DONE"
