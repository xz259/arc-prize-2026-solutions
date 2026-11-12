#!/bin/bash
# sp_export_valnext55.sh (2026-10-22, MTP refit; Claude session 099768dc; from sidepod sp_export_valrbp.sh 52c84742): the MTP features of
# the refit's validation set (jobs/valnext55: the user's holdout gen110/next55, select_val_next55.py): the sessions replayed up to each
# window end on the CURRENT serving stack (fin73/s2 at ctx78 arm b's config, mtp_replay_start.sh MODE=val: dflashtaps MTP option, keep
# ranges [start - 16,384, end), stored slots 0,3,4,5,8 (taps), 9 (final), 10-13 (streams), so mtp_eval_val50.py reads it unchanged), on
# GPU G (default 0), port 1377 + G. The launcher stops its server at the end. Output: FEAT (default /workspace/mtp_refit_r1/features/
# sp_valnext55mtp, container disk).
# usage: [G=0] [DRY=1] setsid nohup bash sp_export_valnext55.sh > LOG 2>&1 < /dev/null &
set -u
[ "$(hostname)" = 2a159825c76b ] || { echo "not the main pod"; exit 9; }
S2=/workspace/kvring_dev/fin73/s2
SP=/mnt/shared/arc3-inference/mtp_refit_20261022
K=/workspace/kvring_dev/mtp73
LAB=${LAB:-sp_valnext55mtp}
V=$SP/jobs/valnext55
FEAT=${FEAT:-/workspace/mtp_refit_r1/features/$LAB}
VSESS=${VSESS:-$V/sessions.json}             # 10-22: a subset list (the re-export of sessions an earlier export failed)
G=${G:-0}; P=$((1377 + G))
cd $S2 || exit 1
[ "$(df --output=fstype . | tail -n 1 | tr -d ' ')" = overlay ] || { echo "not the private stage-2 copy"; exit 1; }
grep -q "ARC3_S2_TAPS_MTP" stage2_v1/dflashtaps.py || { echo "dflashtaps without the MTP option"; exit 1; }
[ -e $FEAT ] && { echo "exists: $FEAT"; exit 1; }
[ -e client_mtp/$LAB ] && { echo "exists: client_mtp/$LAB"; exit 1; }
[ -s $VSESS ] && [ -s $V/keep_ctx16k.json ] || { echo "no valnext55 selection"; exit 1; }
md5sum stage2_v1/dflashtaps.py stage2_v1/replayfx.py replay_driver.py lc_fin.sh $K/mtp_replay_ps.sh $K/mtp_replay_start.sh $K/keep_replay_arm_i.sh | cut -c1-8,33-
echo "[$(date -u +%T)] export $LAB: valnext55 ($(python3 -c "import json,sys; print(len(json.load(open(sys.argv[1]))))" $VSESS) windows) with MTP slots, features $FEAT, GPU $G"
if [ -n "${DRY:-}" ]; then DRYRUN=1 MODE=val KEEPF=$V/keep_ctx16k.json bash $K/mtp_replay_start.sh $LAB $G $P $VSESS $FEAT 7200 ${CONC:-16} 2>&1 | tail -n 25; exit 0; fi
MODE=val KEEPF=$V/keep_ctx16k.json bash $K/mtp_replay_start.sh $LAB $G $P $VSESS $FEAT 7200 ${CONC:-16}
echo "[$(date -u +%T)] mtp_replay_start rc $?; $(grep -E ' DONE |lc_run3i rc' /workspace/kvring_dev/fin73/runs/$LAB/status.txt 2>/dev/null | tail -n 2 | tr '\n' ' ')"
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
echo "[$(date -u +%T)] valnext55 export done"
