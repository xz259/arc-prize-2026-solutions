#!/bin/bash
# sp_export_val50mtp.sh (2026-10-12, Claude fork 5d20b23b; side pod only): SIDE_POD_PLAN section 2(a), the validation
# export with the MTP drafter's input. Job 407's replay of the 50 validation sessions (val50, up to each window end; Phase 6
# server settings N 32, KV 19.01 GiB, K 3, BT 4,096) from the side pod's private stage-2 copy, with
#   - the MTP option of dflashtaps (mtp_ft_20261011/apply_dflashtaps_mtp.py on 31a39ca9, deployed in the PRIVATE copy only;
#     the original is kept as stage2_v1/dflashtaps.py.31a39ca9 and restored by the next bring-up if needed): slots = the 9
#     union taps (2,4,12,13,24,35,36,44,46), the final state, then the 4 pre-final-mixer streams (the MTP's input);
#   - stored slots 0,3,4,5,8 (taps 2,13,24,35,46 = Phase 6's u5), 9 (final), 10-13 (streams): 10 slots, 51.2 KB per row;
#   - keep ranges [start - 16,384, end) per window (16K of context for the MTP's attention; job 145: a 16K MTP window is
#     within 0.1% of production): <= 32,768 rows per session, ~84 GB, on the volume under sidepod_20261012/features.
# usage: setsid nohup bash sp_export_val50mtp.sh > LOG 2>&1 < /dev/null &
set -u
[ "$(hostname)" = 6bc94744da90 ] || { echo "not the side pod"; exit 9; }
S2=/workspace/arc3-inference/runs/stage0-reference-20260919-v1/stage12/harness_revision_v529/spec_decode_stage2
SP=/mnt/shared/arc3-inference/sidepod_20261012
W=/workspace/arc3-inference/dflash_work_20261009
KIT=/workspace/arc3-inference/mtp_ft_20261011
LAB=${LAB:-sp_val50mtp}
FEAT=$SP/features/$LAB
KEEPX=$SP/jobs/val50_keep_ctx16k.json
cd $S2 || exit 1
[ "$(df --output=fstype . | tail -n 1 | tr -d ' ')" = overlay ] || { echo "not the private stage-2 copy"; exit 1; }
[ -e $FEAT ] && { echo "exists: $FEAT"; exit 1; }
[ -e client_mtp/$LAB ] && { echo "exists: client_mtp/$LAB"; exit 1; }
# the MTP option, in the private copy only
if ! grep -q "ARC3_S2_TAPS_MTP" stage2_v1/dflashtaps.py; then
  [ "$(md5sum stage2_v1/dflashtaps.py | cut -c1-8)" = 31a39ca9 ] || { echo "unexpected dflashtaps.py"; exit 1; }
  [ -e stage2_v1/dflashtaps.py.31a39ca9 ] || cp -p stage2_v1/dflashtaps.py stage2_v1/dflashtaps.py.31a39ca9
  python3 $KIT/apply_dflashtaps_mtp.py stage2_v1/dflashtaps.py.31a39ca9 stage2_v1/dflashtaps.py.mtp.tmp || exit 1
  mv stage2_v1/dflashtaps.py.mtp.tmp stage2_v1/dflashtaps.py
fi
md5sum stage2_v1/dflashtaps.py stage2_v1/dflashtaps.py.31a39ca9 replay_driver.py lc_run3.sh run_replay_arm.sh launch_scaling_s2.py | cut -c1-8,33-
mkdir -p $FEAT $SP/jobs
[ -s $KEEPX ] || python3 - $W/val50_20261011/val50_keep.json $KEEPX <<'PYEOF' || exit 1
import json, sys
k = json.load(open(sys.argv[1]))
out = {sid: [[max(0, a - 16384), b] for a, b in r] for sid, r in k.items()}
assert all(len(r) == 1 for r in out.values())
json.dump(out, open(sys.argv[2], 'x'), indent=0)
print('keep ranges', len(out), 'sessions,', sum(b - a for r in out.values() for a, b in r), 'rows')
PYEOF
echo "[$(date -u +%T)] export $LAB: val50 with MTP slots, keep $KEEPX, features $FEAT"
export ARC3_S2_REPLAY=1 ARC3_S2_TAPS_DIR=$FEAT REPLAY_SESSIONS=$W/val50_20261011/val50_sessions.json ARC3_S2_TAPS_KEEP=$KEEPX \
  ARC3_S2_TAPS_MTP=1 ARC3_S2_TAPS_STORE=0,3,4,5,8,9,10,11,12,13 ARC3_S2_TAPS_PREFIX=arc3-persistent- \
  CONCURRENCY=32 ARC3_PERSISTENT_TURN_TIMEOUT=3600
ARM=run_replay_arm.sh ARC3_S2_PMOVE_VERIFY=1 ARC3_S2_AGECAP=96000 CTX=64684 N=32 KVG=19.01 BASIS=52684 TAIL=12000 FRAC=0.2 \
  LABEL=$LAB KEEP=32342 PREFIX_SHARE=0 SECS=10800 \
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
ls $FEAT | wc -l | sed 's/^/feature files: /'; du -sh $FEAT
cd /workspace/arc3-inference && bash runtime_python.sh minimal_inference.py stop
sleep 30
echo "[$(date -u +%T)] vllm processes left: $(ps -eo args | grep -v grep | grep -c vllm.entrypoints); GPU 0 $(nvidia-smi -i 0 --query-gpu=memory.used --format=csv,noheader)"
