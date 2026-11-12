#!/bin/bash
# sp_round1_arm.sh (2026-10-13, Claude fork 5d20b23b; side pod only): one round-1 arm of the MTP recipe sweep on one GPU,
# end to end: wait for the speed tests to leave the GPUs, train (sp_mtp_ft2.sh: all parameters, K 4 unroll with weights
# 1 / 0.69 / 0.40 / 0.18, 4,096-row windows x 4 = 16K rows per step, served KL with an LK-hybrid cooldown, 1,500 + 300
# steps, 5% warmup, session-disjoint trainer validation), then score heads 1,500 and 1,800 on val50 (production
# emulation, experts re-quantised to NVFP4) and head 1,800 again with BF16 experts (IE's per-arm re-quantisation check).
# usage: RUN=r1_all_k4_lr4e5 LR=4e-5 DEV=cuda:0 setsid nohup bash sp_round1_arm.sh > LOG 2>&1 < /dev/null &
set -u
[ "$(hostname)" = 6bc94744da90 ] || { echo "not the side pod"; exit 9; }
SP=/mnt/shared/arc3-inference/sidepod_20261012
RUN=${RUN:?run}; LR=${LR:?lr}; DEV=${DEV:?device}
cd $SP/scripts
while pgrep -f '[s]p_speed_tests' >/dev/null; do sleep 15; done
TS=$(date -u +%Y%m%dT%H%M%S)
echo "[$(date -u +%T)] round-1 arm $RUN on $DEV: training (log logs/mtp_ft_${RUN}_$TS.log)"
RUN=$RUN LR=$LR TRAIN=all DEV=$DEV NK=4 SW=1,0.69,0.40,0.18 WINDOW=4096 ACCUM=4 ROUTER=1.0 SAVE_AT=1500,1800 \
  EXTRA="--val-sessions 2 --val-windows 24" bash sp_mtp_ft2.sh > $SP/logs/mtp_ft_${RUN}_$TS.log 2>&1
echo "[$(date -u +%T)] trainer done: $(tail -n 1 $SP/logs/mtp_ft_${RUN}_$TS.log)"
RUN=$RUN STEPS="1500 1800" EXPERTS=requant DEV=$DEV bash sp_eval_heads.sh
RUN=$RUN STEPS="1800" EXPERTS=bf16 SUFFIX=_bf16 DEV=$DEV bash sp_eval_heads.sh
echo "[$(date -u +%T)] arm $RUN done"
