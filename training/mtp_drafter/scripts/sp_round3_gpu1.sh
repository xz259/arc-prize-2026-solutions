#!/bin/bash
# sp_round3_gpu1.sh (2026-10-13, Claude fork 5d20b23b; side pod only): round 3 on GPU 1 after round 2's two arms there:
# r3_k4_w16k_lr4e5, K 4 at 16K windows (--ckpt-steps), 16K rows per step (16,384 x 1), LR 4e-5: the full run's shape
# (user 10-13: training windows 16K, never shorter), against round 1's r1_all_k4_lr4e5 (4K x 4) and r3_k5_w16k_lr4e5.
# usage: bash start_detached.sh logs/round3_gpu1.log sp_round3_gpu1.sh
set -u
[ "$(hostname)" = 6bc94744da90 ] || { echo "not the side pod"; exit 9; }
SP=/mnt/shared/arc3-inference/sidepod_20261012
cd $SP/scripts
until grep -q "round 2 gpu1 done" $SP/logs/round2_gpu1.log 2>/dev/null; do sleep 60; done
echo "[$(date -u +%T)] round 2 gpu1 done: round 3 arm on GPU 1"
RUN=r3_k4_w16k_lr4e5 LR=4e-5 DEV=cuda:1 WINDOW=16384 ACCUM=1 EXTRA="--ckpt-steps" \
  bash sp_round_arm.sh > $SP/logs/round3_r3_k4_w16k_lr4e5.log 2>&1
echo "[$(date -u +%T)] r3_k4_w16k_lr4e5: $(tail -n 1 $SP/logs/round3_r3_k4_w16k_lr4e5.log)"
echo "[$(date -u +%T)] round 3 gpu1 done"
