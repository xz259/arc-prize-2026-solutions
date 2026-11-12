#!/bin/bash
# sp_export_valrbp.sh (2026-10-13, Claude fork 5d20b23b; side pod only): the MTP features of the full run's validation set
# (jobs/valrbp50v: 50 redbluepill games, select_val_rbp.py; user 10-13: the baseline is scored on it at the full run's
# start), exported as sp_export_val50mtp.sh did for val50: the 50 sessions replayed up to each window end on the production
# replay path (N 32, KV 19.01 GiB, K 3, BT 4,096) with dflashtaps' MTP option (already in the private copy), keep ranges
# [start - 16,384, end), stored slots 0,3,4,5,8 (taps), 9 (final), 10-13 (streams) so mtp_eval_val50.py reads it unchanged.
# Waits for GPU 0 (after sp_speed_tests5.sh); stops the server at the end. Output: features/$LAB on the volume (~25 GB).
# usage: setsid nohup bash sp_export_valrbp.sh > LOG 2>&1 < /dev/null &
set -u
[ "$(hostname)" = 6bc94744da90 ] || { echo "not the side pod"; exit 9; }
S2=/workspace/arc3-inference/runs/stage0-reference-20260919-v1/stage12/harness_revision_v529/spec_decode_stage2
SP=/mnt/shared/arc3-inference/sidepod_20261012
LAB=${LAB:-sp_valrbp50mtp}
V=$SP/jobs/valrbp50v
FEAT=$SP/features/$LAB
until grep -q "speed tests 5 done" $SP/logs/speed_tests5_20261013.log 2>/dev/null; do sleep 60; done
cd $S2 || exit 1
[ "$(df --output=fstype . | tail -n 1 | tr -d ' ')" = overlay ] || { echo "not the private stage-2 copy"; exit 1; }
grep -q "ARC3_S2_TAPS_MTP" stage2_v1/dflashtaps.py || { echo "dflashtaps without the MTP option"; exit 1; }
[ -e $FEAT ] && { echo "exists: $FEAT"; exit 1; }
[ -e client_mtp/$LAB ] && { echo "exists: client_mtp/$LAB"; exit 1; }
[ -s $V/sessions.json ] && [ -s $V/keep_ctx16k.json ] || { echo "no valrbp50v selection"; exit 1; }
m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)
[ "$m" -lt 2000 ] || { echo "GPU 0 busy ($m MiB)"; exit 1; }
md5sum stage2_v1/dflashtaps.py replay_driver.py lc_run3.sh run_replay_arm.sh launch_scaling_s2.py | cut -c1-8,33-
mkdir -p $FEAT
echo "[$(date -u +%T)] export $LAB: valrbp50v with MTP slots, features $FEAT"
export ARC3_S2_REPLAY=1 ARC3_S2_TAPS_DIR=$FEAT REPLAY_SESSIONS=$V/sessions.json ARC3_S2_TAPS_KEEP=$V/keep_ctx16k.json \
  ARC3_S2_TAPS_MTP=1 ARC3_S2_TAPS_STORE=0,3,4,5,8,9,10,11,12,13 ARC3_S2_TAPS_PREFIX=arc3-persistent- \
  CONCURRENCY=32 ARC3_PERSISTENT_TURN_TIMEOUT=3600
ARM=run_replay_arm.sh ARC3_S2_PMOVE_VERIFY=1 ARC3_S2_AGECAP=96000 CTX=64684 N=32 KVG=19.01 BASIS=52684 TAIL=12000 FRAC=0.2 \
  LABEL=$LAB KEEP=32342 PREFIX_SHARE=0 SECS=7200 \
  EXTRA_PARTS=agestat,agecap,hollowgroup,sessmem,rope2m,ctxtail,promptmove,foldc,dflashtaps,replayfx bash lc_run3.sh
echo "[$(date -u +%T)] lc_run3 rc $?"
[ -s client_mtp/$LAB/driver.jsonl ] && python3 - client_mtp/$LAB/driver.jsonl <<'PYEOF'
import collections, json, sys
c = collections.Counter(); tok = 0
for line in open(sys.argv[1]):
    d = json.loads(line)
    if 'session_id' in d:
        c[d['status'].split(':')[0][:40]] += 1; tok += d.get('stream_tokens') or 0
    else:
        print('total', d)
print(dict(c), 'stream tokens', tok)
PYEOF
ls $FEAT | wc -l | sed 's/^/feature files: /'; du -sh --apparent-size $FEAT
cd /workspace/arc3-inference && bash runtime_python.sh minimal_inference.py stop
sleep 30
echo "[$(date -u +%T)] vllm processes left: $(ps -eo args | grep -v grep | grep -c vllm.entrypoints); GPU 0 $(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader)"
echo "[$(date -u +%T)] valrbp export done"
