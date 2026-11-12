#!/bin/bash
# sp_replay_speedtest.sh (2026-10-13, Claude fork 5d20b23b; side pod only): before the full run (user 10-13: "try faster replay
# before the full training"; it pays on a 4-GPU pod, 1 replay + 3 trainers): the same ~1.5M stream tokens (the first
# sessions of the full run's list, each capped at 300K tokens) replayed twice on GPU 0 at the full run's driver concurrency
# 4, storing the full run's slots plus the final state (9-13): BASE = production (prefill chunk BT 4,096, K 3 draft tokens),
# FAST = BT 16,384 and K 1 (Phase 6 design item 2; drafting is wasted work in a forced replay). Reports stream tokens/s per
# arm (driver time, server start excluded) and compares the features (compare_replay_arms.py: argmax agreement, KL, NLL of
# the recorded tokens, stream error). Outputs on the container disk under /workspace/replaytest_20261013/ (~37 GB per arm).
# Waits for GPU 0 (after sp_export_valrbp.sh).
# usage: setsid nohup bash sp_replay_speedtest.sh > LOG 2>&1 < /dev/null &      (env: FAST_BT 16384, FAST_K 1)
set -u
[ "$(hostname)" = 6bc94744da90 ] || { echo "not the side pod"; exit 9; }
S2=/workspace/arc3-inference/runs/stage0-reference-20260919-v1/stage12/harness_revision_v529/spec_decode_stage2
SP=/mnt/shared/arc3-inference/sidepod_20261012
PY=/workspace/arc3-inference/venv-speculators/bin/python
OUTD=/workspace/replaytest_20261013
SESS=$SP/jobs/replaytest_sessions.json
until grep -q "valrbp export done" $SP/logs/export_valrbp_20261013.log 2>/dev/null; do sleep 60; done
cd $S2 || exit 1
[ "$(df --output=fstype . | tail -n 1 | tr -d ' ')" = overlay ] || { echo "not the private stage-2 copy"; exit 1; }
[ -e $OUTD ] && { echo "exists: $OUTD"; exit 1; }
[ -s $SESS ] || python3 - $SP/jobs/valrbp50v/train_sessions.json $SESS <<'PYEOF' || exit 1
import json, sys
items = json.load(open(sys.argv[1]))
n = {}
for line in open('/workspace/arc3-inference/dflash_streams_20261011/sessions.jsonl'):
    d = json.loads(line); n[d['session_id']] = d['n']
pick, tot = [], 0
for it in items:
    t = min(300000, it.get('stop_tokens') or n[it['session_id']])
    pick.append(dict(it, stop_tokens=t)); tot += t
    if tot >= 1500000:
        break
json.dump(pick, open(sys.argv[2], 'x'), indent=1)
print('replay test: %d sessions, %d stream tokens' % (len(pick), tot))
PYEOF
mkdir -p $OUTD
for ARMDEF in "base 4096 3" "fast ${FAST_BT:-16384} ${FAST_K:-1}"; do
  set -- $ARMDEF
  NAME=$1; BTV=$2; KV=$3; LAB=replaytest_$NAME
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
  [ "$m" -lt 2000 ] || { echo "GPU 0 busy ($m MiB)"; exit 1; }
  [ -e client_mtp/$LAB ] && { echo "exists: client_mtp/$LAB"; exit 1; }
  mkdir -p $OUTD/$NAME
  echo "[$(date -u +%T)] replay arm $NAME: BT $BTV, K $KV, driver concurrency 4"
  ARC3_S2_REPLAY=1 ARC3_S2_TAPS_DIR=$OUTD/$NAME REPLAY_SESSIONS=$SESS ARC3_S2_TAPS_MTP=1 ARC3_S2_TAPS_STORE=9,10,11,12,13 \
    ARC3_S2_TAPS_SEGMENT=65536 ARC3_S2_TAPS_HEAD=0 ARC3_S2_TAPS_PREFIX=arc3-persistent- ARC3_PERSISTENT_TURN_TIMEOUT=3600 \
    CONCURRENCY=4 ARM=run_replay_arm.sh ARC3_S2_PMOVE_VERIFY=1 ARC3_S2_AGECAP=96000 CTX=64684 N=32 KVG=19.01 BASIS=52684 \
    TAIL=12000 FRAC=0.2 LABEL=$LAB KEEP=32342 PREFIX_SHARE=0 SECS=3600 LC_BT=$BTV MTPK=$KV \
    EXTRA_PARTS=agestat,agecap,hollowgroup,sessmem,rope2m,ctxtail,promptmove,foldc,dflashtaps,replayfx bash lc_run3.sh
  echo "[$(date -u +%T)] lc_run3 rc $?"
  python3 - client_mtp/$LAB/driver.jsonl $NAME <<'PYEOF'
import collections, json, sys
c = collections.Counter(); tok = 0; secs = None
for line in open(sys.argv[1]):
    d = json.loads(line)
    if 'session_id' in d:
        c[d['status'].split(':')[0][:30]] += 1; tok += d.get('stream_tokens') or 0
    elif 'done' in d:
        secs = d.get('seconds')
print('arm %s: %s, stream tokens %d, driver seconds %s, %.0f tokens/s' % (sys.argv[2], dict(c), tok, secs, tok / secs if secs else 0))
PYEOF
  (cd /workspace/arc3-inference && bash runtime_python.sh minimal_inference.py stop)
  sleep 30
done
echo "[$(date -u +%T)] compare the arms' features (GPU 0)"
cd $SP/scripts && PYTHONDONTWRITEBYTECODE=1 $PY -u compare_replay_arms.py $OUTD/base $OUTD/fast --device cuda:0 --out $SP/results/replay_speedtest_compare.json 2>&1 | tail -n 3
echo "[$(date -u +%T)] replay speed test done"
