#!/bin/bash
# sp_replay_fast8k.sh (2026-10-13, Claude fork 5d20b23b; side pod only): the faster-replay arm of sp_replay_speedtest.sh again,
# within the production launcher's limit (launch_scaling_s2.py accepts batched tokens 4,096 or 8,192 only; the 16,384 arm
# failed at 07:02): BT 8,192 and K 1 (drafting is wasted work in a forced replay), the same 8 sessions / 1.5M stream tokens
# and slots as the base arm (8,445 tokens/s at 06:55), then compare_replay_arms.py base vs fast8k (argmax agreement, KL,
# NLL of the recorded tokens). On GPU 0 after round 3 (sp_round3_gpu0.sh), before the full run; stops its server at the end.
# Writes /workspace/replaytest_20261013/fast8k (~38 GB, container disk) and results/replay_speedtest_compare_fast8k.json.
# usage: bash start_detached.sh logs/replay_fast8k_20261013.log sp_replay_fast8k.sh
set -u
[ "$(hostname)" = 6bc94744da90 ] || { echo "not the side pod"; exit 9; }
S2=/workspace/arc3-inference/runs/stage0-reference-20260919-v1/stage12/harness_revision_v529/spec_decode_stage2
SP=/mnt/shared/arc3-inference/sidepod_20261012
PY=/workspace/arc3-inference/venv-speculators/bin/python
OUTD=/workspace/replaytest_20261013
SESS=$SP/jobs/replaytest_sessions.json
NAME=fast8k; LAB=replaytest_$NAME
until grep -q "round 3 gpu0 done" $SP/logs/round3_gpu0.log 2>/dev/null; do sleep 60; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)" -lt 2000 ]; do sleep 30; done
cd $S2 || exit 1
[ "$(df --output=fstype . | tail -n 1 | tr -d ' ')" = overlay ] || { echo "not the private stage-2 copy"; exit 1; }
[ -d $OUTD/base ] && [ -s $SESS ] || { echo "no base arm / session list"; exit 1; }
[ -e client_mtp/$LAB ] && { echo "exists: client_mtp/$LAB"; exit 1; }
[ -e $OUTD/$NAME ] && { echo "exists: $OUTD/$NAME"; exit 1; }
FREE=$(df --output=avail -BG /workspace | tail -n 1 | tr -dc 0-9)
[ "$FREE" -gt 110 ] || { echo "container disk: ${FREE} GB free, need > 110 (38 GB arm + the full run's 62 GB state staging)"; exit 1; }
mkdir -p $OUTD/$NAME
echo "[$(date -u +%T)] replay arm $NAME: BT 8192, K 1, driver concurrency 4"
ARC3_S2_REPLAY=1 ARC3_S2_TAPS_DIR=$OUTD/$NAME REPLAY_SESSIONS=$SESS ARC3_S2_TAPS_MTP=1 ARC3_S2_TAPS_STORE=9,10,11,12,13 \
  ARC3_S2_TAPS_SEGMENT=65536 ARC3_S2_TAPS_HEAD=0 ARC3_S2_TAPS_PREFIX=arc3-persistent- ARC3_PERSISTENT_TURN_TIMEOUT=3600 \
  CONCURRENCY=4 ARM=run_replay_arm.sh ARC3_S2_PMOVE_VERIFY=1 ARC3_S2_AGECAP=96000 CTX=64684 N=32 KVG=19.01 BASIS=52684 \
  TAIL=12000 FRAC=0.2 LABEL=$LAB KEEP=32342 PREFIX_SHARE=0 SECS=3600 LC_BT=8192 MTPK=1 \
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
print('arm %s: %s, stream tokens %d, driver seconds %s, %.0f tokens/s (base: 8445)' % (sys.argv[2], dict(c), tok, secs, tok / secs if secs else 0))
PYEOF
(cd /workspace/arc3-inference && bash runtime_python.sh minimal_inference.py stop)
sleep 30
echo "[$(date -u +%T)] compare base vs $NAME features (GPU 0)"
cd $SP/scripts && PYTHONDONTWRITEBYTECODE=1 $PY -u compare_replay_arms.py $OUTD/base $OUTD/$NAME --device cuda:0 \
  --out $SP/results/replay_speedtest_compare_$NAME.json 2>&1 | tail -n 3
echo "[$(date -u +%T)] replay fast8k done"
