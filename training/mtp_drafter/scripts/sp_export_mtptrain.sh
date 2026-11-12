#!/bin/bash
# sp_export_mtptrain.sh (2026-10-12, Claude fork 5d20b23b; side pod only): SIDE_POD_PLAN section 2(b), the MTP
# fine-tune data for the quick estimate (user 10-12 ~21:5x: step 4 is a quick estimate on a ~20M-token subset, not the
# full fine-tune). Whole corpus-v7 sessions (Phase 6's sessions_driver.json entries, stop_tokens kept), none of the 50
# validation sessions, drawn at random (seed 20261012) until ~TOKENS stream tokens; replayed exactly on the production
# replay path (job 407's settings, concurrency 32) with dflashtaps' MTP option (deployed in the private copy by
# sp_export_val50mtp.sh); stored slots 9 (final state) and 10-13 (the 4 pre-final-mixer streams): 25.6 KB per row, written
# in segments of 65,536 rows without overlap (ARC3_S2_TAPS_SEGMENT, HEAD 0) so a long session never sits in host memory.
# Output: sidepod_20261012/features/$LAB on the volume (~0.5 TB at 20M tokens). Roles (assistant mask) come from the
# Phase 6 streams (dflash_streams_20261011) at training time (train_mtp.py --streams).
# usage: setsid nohup bash sp_export_mtptrain.sh > LOG 2>&1 < /dev/null &      (env: LAB, TOKENS)
set -u
[ "$(hostname)" = 6bc94744da90 ] || { echo "not the side pod"; exit 9; }
S2=/workspace/arc3-inference/runs/stage0-reference-20260919-v1/stage12/harness_revision_v529/spec_decode_stage2
SP=/mnt/shared/arc3-inference/sidepod_20261012
W=/workspace/arc3-inference/dflash_work_20261009
P6=/workspace/arc3-inference/dflash_phase6_20261011v
LAB=${LAB:-sp_mtptrain20m}
TOKENS=${TOKENS:-20000000}
FEAT=$SP/features/$LAB
SESS=$SP/jobs/$LAB.sessions.json
cd $S2 || exit 1
[ "$(df --output=fstype . | tail -n 1 | tr -d ' ')" = overlay ] || { echo "not the private stage-2 copy"; exit 1; }
grep -q "ARC3_S2_TAPS_MTP" stage2_v1/dflashtaps.py || { echo "dflashtaps without the MTP option"; exit 1; }
[ -e $FEAT ] && { echo "exists: $FEAT"; exit 1; }
[ -e client_mtp/$LAB ] && { echo "exists: client_mtp/$LAB"; exit 1; }
md5sum stage2_v1/dflashtaps.py replay_driver.py lc_run3.sh run_replay_arm.sh launch_scaling_s2.py | cut -c1-8,33-
mkdir -p $FEAT $SP/jobs
[ -s $SESS ] || python3 - $P6/sessions_driver.json $P6/plan.json $W/val50_20261011/val50_sessions.json $SESS $TOKENS <<'PYEOF' || exit 1
import json, random, sys
drv, plan, val, out, budget = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], int(sys.argv[5])
n = {s['session_id']: s['n'] for s in json.load(open(plan))['sessions']}
vs = {s['session_id'] for s in json.load(open(val))}
cand = [s for s in json.load(open(drv)) if s['session_id'] not in vs]
random.Random(20261012).shuffle(cand)
pick, tot = [], 0
for s in cand:
    t = s.get('stop_tokens') or n[s['session_id']]
    if tot >= budget:
        break
    pick.append(s); tot += t
json.dump(pick, open(out, 'x'), indent=1)
runs = {}
for s in pick:
    runs[s['run']] = runs.get(s['run'], 0) + (s.get('stop_tokens') or n[s['session_id']])
print('sessions', len(pick), 'of', len(cand), 'stream tokens', tot, 'runs', dict(sorted(runs.items())))
PYEOF
echo "[$(date -u +%T)] export $LAB: $(python3 -c "import json; print(len(json.load(open('$SESS'))))" ) sessions, features $FEAT"
export ARC3_S2_REPLAY=1 ARC3_S2_TAPS_DIR=$FEAT REPLAY_SESSIONS=$SESS ARC3_S2_TAPS_MTP=1 ARC3_S2_TAPS_STORE=9,10,11,12,13 \
  ARC3_S2_TAPS_SEGMENT=65536 ARC3_S2_TAPS_HEAD=0 ARC3_S2_TAPS_PREFIX=arc3-persistent- CONCURRENCY=32 \
  ARC3_PERSISTENT_TURN_TIMEOUT=3600
ARM=run_replay_arm.sh ARC3_S2_PMOVE_VERIFY=1 ARC3_S2_AGECAP=96000 CTX=64684 N=32 KVG=19.01 BASIS=52684 TAIL=12000 FRAC=0.2 \
  LABEL=$LAB KEEP=32342 PREFIX_SHARE=0 SECS=21600 \
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
