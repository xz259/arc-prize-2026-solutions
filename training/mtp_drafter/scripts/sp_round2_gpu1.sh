#!/bin/bash
# sp_round2_gpu1.sh (2026-10-13, Claude fork 5d20b23b; side pod only): round 2 on GPU 1 after sp_speed_tests6.sh, one arm at
# a time (sp_round_arm.sh: 4K x 4, K 4, served KL + LK cooldown, 1,500 + 300 steps, heads scored on val50):
#   1. r2_noexp_lr4e5: routed experts frozen (BF16), router at 0.3x LR, scored with the served NVFP4 experts;
#   2. r2_all_k4_lr5e5: all parameters at 5e-5 (the user's FastMTP default for the full run) against round 1's 4e-5.
# Rounds 2-3 run at 4e-5 so every comparison has a same-LR reference (round 1's r1_all_k4_lr4e5).
# usage: bash start_detached.sh logs/round2_gpu1.log sp_round2_gpu1.sh
set -u
[ "$(hostname)" = 6bc94744da90 ] || { echo "not the side pod"; exit 9; }
SP=/mnt/shared/arc3-inference/sidepod_20261012
cd $SP/scripts
until grep -q "speed tests 6 done" $SP/logs/speed_tests6_20261013.log 2>/dev/null; do sleep 60; done
echo "[$(date -u +%T)] speed tests 6 done: round 2 on GPU 1"
RUN=r2_noexp_lr4e5 LR=4e-5 DEV=cuda:1 TRAIN=noexperts ROUTER=0.3 EXPERTS=served bash sp_round_arm.sh > $SP/logs/round2_r2_noexp_lr4e5.log 2>&1
echo "[$(date -u +%T)] r2_noexp_lr4e5: $(tail -n 1 $SP/logs/round2_r2_noexp_lr4e5.log)"
RUN=r2_all_k4_lr5e5 LR=5e-5 DEV=cuda:1 bash sp_round_arm.sh > $SP/logs/round2_r2_all_k4_lr5e5.log 2>&1
echo "[$(date -u +%T)] r2_all_k4_lr5e5: $(tail -n 1 $SP/logs/round2_r2_all_k4_lr5e5.log)"
echo "[$(date -u +%T)] round 2 gpu1 done"
