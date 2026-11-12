#!/bin/bash
# sp_round_arm.sh (2026-10-13, Claude fork 5d20b23b; side pod only): one arm of recipe rounds 2-3 on one GPU, end to end (as
# sp_round1_arm.sh, generalised): wait for the GPU, train (sp_mtp_ft2.sh with the arm's settings; trainer 881c9624: served
# target once per row, prefetch 2), then score heads 1,500 and 1,800 on val50 (production emulation; EXPERTS requant for
# trained experts, served for frozen ones) and, when the experts were trained, head 1,800 again with BF16 experts.
# usage: RUN=r2_noexp_lr4e5 LR=4e-5 DEV=cuda:1 TRAIN=noexperts ROUTER=0.3 EXPERTS=served setsid nohup bash sp_round_arm.sh > LOG 2>&1 < /dev/null &
#        (env: TRAIN all, OPT adamw, ROUTER 1.0, NK 4, SW 1,0.69,0.40,0.18, WINDOW 4096, ACCUM 4, LOSS served_kl, CLOSS lk,
#         ACCEPT "", EXTRA "", EXPERTS requant, MOE loop, STEPS 1800)
set -u
[ "$(hostname)" = 6bc94744da90 ] || { echo "not the side pod"; exit 9; }
SP=/mnt/shared/arc3-inference/sidepod_20261012
RUN=${RUN:?run}; LR=${LR:?lr}; DEV=${DEV:?device}
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
RUN=$RUN STEPS="1500 1800" EXPERTS=${EXPERTS:-requant} DEV=$DEV bash sp_eval_heads.sh
if [ "${TRAIN:-all}" = all ]; then
  RUN=$RUN STEPS="1800" EXPERTS=bf16 SUFFIX=_bf16 DEV=$DEV bash sp_eval_heads.sh
fi
echo "[$(date -u +%T)] arm $RUN done"
