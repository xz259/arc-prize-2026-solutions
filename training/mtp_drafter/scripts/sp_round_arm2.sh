#!/bin/bash
# sp_round_arm2.sh (2026-10-13, Claude fork 5d20b23b; side pod only): sp_round_arm.sh with early stopping (user 10-13: "stop
# early if the early sign is not good"): if the trainer stopped on its stop file (trainer_status "stopped"), the arm ends
# without scoring. Otherwise identical: train (sp_mtp_ft2.sh), score heads 1,500 and 1,800 on val50 (EXPERTS), and the
# BF16 check for trained experts. (A separate copy: sp_round_arm.sh is in use by running arms.) An optional
# jobs/arm_overrides/<RUN>.env is sourced first (queued arms can be changed before they start).
# usage: RUN=... LR=... DEV=cuda:N [TRAIN OPT ROUTER NK SW WINDOW ACCUM LOSS CLOSS ACCEPT EXTRA EXPERTS MOE STEPS] bash sp_round_arm2.sh
set -u
[ "$(hostname)" = 6bc94744da90 ] || { echo "not the side pod"; exit 9; }
SP=/mnt/shared/arc3-inference/sidepod_20261012
RUN=${RUN:?run}; LR=${LR:?lr}; DEV=${DEV:?device}
OV=$SP/jobs/arm_overrides/$RUN.env                     # optional per-arm overrides (may rename RUN, change the loss)
[ -s $OV ] && { echo "[$(date -u +%T)] overrides from $OV: $(tr '\n' ' ' < $OV)"; . $OV; }
G=${DEV#cuda:}
cd $SP/scripts
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G)" -lt 2000 ]; do sleep 30; done
TS=$(date -u +%Y%m%dT%H%M%S)
echo "[$(date -u +%T)] arm $RUN on $DEV: training (log logs/mtp_ft_${RUN}_$TS.log)"
MTP_MOE=${MOE:-loop} RUN=$RUN LR=$LR TRAIN=${TRAIN:-all} DEV=$DEV NK=${NK:-4} SW=${SW:-1,0.69,0.40,0.18} WINDOW=${WINDOW:-4096} \
  ACCUM=${ACCUM:-4} ROUTER=${ROUTER:-1.0} SAVE_AT=1500,1800 STEPS=${STEPS:-1800} LOSS=${LOSS:-served_kl} CLOSS=${CLOSS:-lk} \
  ACCEPT="${ACCEPT:-}" EXTRA="--val-sessions 2 --val-windows 24 --prefetch 2 --optimizer ${OPT:-adamw} ${EXTRA:-}" \
  bash sp_mtp_ft2.sh > $SP/logs/mtp_ft_${RUN}_$TS.log 2>&1
echo "[$(date -u +%T)] trainer done: $(tail -n 1 $SP/logs/mtp_ft_${RUN}_$TS.log)"
if grep -q '"stopped"' $SP/mtp_ft/$RUN/trainer_status.json 2>/dev/null; then
  echo "[$(date -u +%T)] stopped early: $(cat $SP/mtp_ft/$RUN/trainer_status.json); no scoring"
  echo "[$(date -u +%T)] arm $RUN done"; exit 0
fi
RUN=$RUN STEPS="1500 1800" EXPERTS=${EXPERTS:-requant} DEV=$DEV bash sp_eval_heads.sh
if [ "${TRAIN:-all}" = all ]; then
  RUN=$RUN STEPS="1800" EXPERTS=bf16 SUFFIX=_bf16 DEV=$DEV bash sp_eval_heads.sh
fi
echo "[$(date -u +%T)] arm $RUN done"
