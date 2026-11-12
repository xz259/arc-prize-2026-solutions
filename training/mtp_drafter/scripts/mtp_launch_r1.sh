#!/bin/bash
# mtp_launch_r1.sh (2026-10-22, MTP refit; Claude session 099768dc; the user's approval 10-22: launch the full run unattended if the smokes
# pass, with the early check; handoff 05:41 UTC): after the next55 export is complete (run_export_b.sh 'EXPORT B DONE' with 55 feature
# files; OVERDUE 08:00 UTC):
#   1. the baselines on the holdout, in parallel: the base MTP head (MTP3, H=base) on GPU 0, fr1's head (head_step10142) on GPU 1
#      (sp_score_head.sh: production emulation, served experts, live tau K 3-5 at T 1 / 0.6);
#   2. the full run r1 (sp_fullrun.sh: replay GPU 0, trainer GPU 1): fresh from fr1's state (--init-from state_stop_step10142.pt), LR 4e-5,
#      STEPS 7,388 (DRY at 05:59), WARMUP 369 (5%), DECAY_START 1,847 (+20%), served_kl -> lk, loss T 0.6, Muon, nonexpert, served experts,
#      K4 1/.69/.40/.18, CKPT 225, KEEP 750, EVAL 375, baselines skipped (step 1);
#   3. early_check.sh (steps 750 / 1,125 vs step 0).
# Log: SP/logs/mtp_launch_r1.log
set -u
SP=/mnt/shared/arc3-inference/mtp_refit_20261022; SPOLD=/mnt/shared/arc3-inference/sidepod_20261012
F=/workspace/mtp_refit_r1/features/sp_valnext55mtp
log() { echo "[$(date -u +%T)] $*"; }
DL=$(date -u -d '2026-10-22 08:00:00' +%s)
until grep -q "EXPORT B DONE" $SP/logs/export_valnext55_b.log 2>/dev/null; do
  [ "$(date +%s)" -ge "$DL" ] && { log "OVERDUE: no EXPORT B DONE by 08:00 UTC: nothing launched"; exit 1; }
  sleep 20
done
N=$(ls $F | grep -c safetensors)
[ "$N" -ge 55 ] || { log "the export holds $N feature files (55 wanted): nothing launched"; exit 1; }
log "export complete: $N files; baselines on GPU 0 (MTP3) and GPU 1 (fr1)"
cd $SP/scripts
H=base TAG=mtp3_base DEV=cuda:0 SETS=valnext55 bash sp_score_head.sh > $SP/logs/score_mtp3_base.log 2>&1 &
P0=$!
H=$SPOLD/mtp_ft/fr1_final/head_step10142.safetensors TAG=fr1 DEV=cuda:1 SETS=valnext55 bash sp_score_head.sh > $SP/logs/score_fr1.log 2>&1 &
P1=$!
wait $P0; R0=$?; wait $P1; R1=$?
log "baselines rc $R0 / $R1"
grep -h -E "prob_srv_T(10|06)|K [345]|tau" $SP/logs/score_mtp3_base.log $SP/logs/score_fr1.log | head -n 30
for g in 0 1; do for i in $(seq 1 30); do [ -z "$(nvidia-smi -i $g --query-compute-apps=pid --format=csv,noheader)" ] && break; sleep 10; done; done
log "launching the full run r1"
RUN=r1 LR=4e-5 OPT=muon TRAIN=nonexpert MOE=grouped NK=4 SW=1,0.69,0.40,0.18 LOSS=served_kl CLOSS=lk LOSST=0.6 \
  STEPS=7388 WARMUP=369 DECAY_START=1847 KEEP=750 CKPT=225 EVAL=375 SKIP_BASELINE=1 \
  EXTRA="--init-from $SPOLD/mtp_ft/fr1/state_stop_step10142.pt" \
  bash start_detached.sh $SP/logs/fullrun_r1_a1.log sp_fullrun.sh
sleep 5
RUN=r1 bash start_detached.sh $SP/logs/early_check_r1.log early_check.sh
log "MTP LAUNCH DONE (full run log logs/fullrun_r1_a1.log; early check logs/early_check_r1.log)"
