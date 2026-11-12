#!/bin/bash
# sp_round5_gpu1.sh (2026-10-13, Claude fork 5d20b23b; side pod only): the full run's combination at round scale on GPU 1 after
# round 3 there: r3_muon_k4_w16k_lr4e5, Muon (round 2's winner, +0.014 at 4K) with 16K windows (16,384 x 1, --ckpt-steps),
# K 4, LR 4e-5, 1,500 + 300 steps, against r3_k4_w16k_lr4e5 (AdamW, same shape): does Muon's gain hold at 16K?
# usage: bash start_detached.sh logs/round5_gpu1.log sp_round5_gpu1.sh
set -u
[ "$(hostname)" = 6bc94744da90 ] || { echo "not the side pod"; exit 9; }
SP=/mnt/shared/arc3-inference/sidepod_20261012
cd $SP/scripts
until grep -q "round 3 gpu1 done" $SP/logs/round3_gpu1.log 2>/dev/null; do sleep 60; done
echo "[$(date -u +%T)] round 3 gpu1 done: Muon at 16K on GPU 1"
RUN=r3_muon_k4_w16k_lr4e5 LR=4e-5 DEV=cuda:1 OPT=muon WINDOW=16384 ACCUM=1 EXTRA="--ckpt-steps" \
  bash sp_round_arm.sh > $SP/logs/round5_r3_muon_k4_w16k_lr4e5.log 2>&1
echo "[$(date -u +%T)] r3_muon_k4_w16k_lr4e5: $(tail -n 1 $SP/logs/round5_r3_muon_k4_w16k_lr4e5.log)"
echo "[$(date -u +%T)] round 5 gpu1 done"
