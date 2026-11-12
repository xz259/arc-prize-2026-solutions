#!/bin/bash
# sp_bringup.sh (2026-10-12, Claude fork 5d20b23b; side pod only): bring-up gate, SIDE_POD_PLAN sections 1.3-1.4.
# Job 407's launch on the side pod (production replay path: N 32, KV 19.01 GiB, K 3, BT 4,096, the Phase 6 parts with
# dflashtaps + replayfx) from the side pod's PRIVATE stage-2 copy, lc_run3's NLL panel gate (must print IDENTICAL against
# 0.34739071770720314), then the exact replay of one validation session (default: the shortest, arc3-cec13fad..., run
# v62gen402a, up to its window end 20,623, keep [4,239, 20,623)) with features on the container disk, for a bitwise
# comparison with job 407's file (sp_compare_features.py). The server is stopped at the end.
# usage: setsid nohup bash sp_bringup.sh > LOG 2>&1 < /dev/null &      (env: LAB, SID)
set -u
[ "$(hostname)" = 6bc94744da90 ] || { echo "not the side pod"; exit 9; }
S2=/workspace/arc3-inference/runs/stage0-reference-20260919-v1/stage12/harness_revision_v529/spec_decode_stage2
SP=/mnt/shared/arc3-inference/sidepod_20261012
W=/workspace/arc3-inference/dflash_work_20261009
LAB=${LAB:-sp_bringup1}
SID=${SID:-arc3-cec13fad6f0c4ec4a69cac2f3fb49aa4}
FEAT=/workspace/sidepod_local/features/$LAB
cd $S2 || exit 1
[ "$(df --output=fstype . | tail -n 1 | tr -d ' ')" = overlay ] || { echo "not the private stage-2 copy"; exit 1; }
[ -e $FEAT ] && { echo "exists: $FEAT"; exit 1; }
[ -e client_mtp/$LAB ] && { echo "exists: client_mtp/$LAB"; exit 1; }
grep -q "served_position_ids=torch.from_numpy" stage2_v1/dflashtaps.py && grep -q "remap_tool_ids" replay_driver.py \
  || { echo "writer or driver not deployed"; exit 1; }
md5sum stage2_v1/dflashtaps.py replay_driver.py lc_run3.sh run_replay_arm.sh launch_scaling_s2.py launch_s2.sh nll_panel.sh | cut -c1-8,33-
mkdir -p $FEAT $SP/jobs
SESS=$SP/jobs/$LAB.sessions.json
python3 - $W/val50_20261011/val50_sessions.json $SID $SESS <<'PYEOF' || exit 1
import json, sys
s = [x for x in json.load(open(sys.argv[1])) if x['session_id'] == sys.argv[2]]
assert len(s) == 1, sys.argv[2]
json.dump(s, open(sys.argv[3], 'x'), indent=1)
print('session', s[0]['session_id'], s[0]['run'], 'stop_tokens', s[0]['stop_tokens'])
PYEOF
echo "[$(date -u +%T)] bring-up $LAB on $(hostname): launch + NLL panel + replay of $SID"
export ARC3_S2_REPLAY=1 ARC3_S2_TAPS_DIR=$FEAT REPLAY_SESSIONS=$SESS ARC3_S2_TAPS_KEEP=$W/val50_20261011/val50_keep.json \
  CONCURRENCY=32 ARC3_PERSISTENT_TURN_TIMEOUT=3600
ARM=run_replay_arm.sh ARC3_S2_PMOVE_VERIFY=1 ARC3_S2_AGECAP=96000 CTX=64684 N=32 KVG=19.01 BASIS=52684 TAIL=12000 FRAC=0.2 \
  LABEL=$LAB KEEP=32342 PREFIX_SHARE=0 SECS=3600 \
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
ls -la $FEAT | grep -v _warmup_
cd /workspace/arc3-inference && bash runtime_python.sh minimal_inference.py stop
sleep 30
echo "[$(date -u +%T)] vllm processes left: $(ps -eo args | grep -v grep | grep -c vllm.entrypoints); GPU 0 $(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader)"
