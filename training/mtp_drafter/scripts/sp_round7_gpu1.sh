#!/bin/bash
# sp_round7_gpu1.sh (2026-10-13, Claude fork 5d20b23b; side pod only): loss ablation (user 10-13: ablate LK-hybrid, stop early if
# the early sign is not good), one arm of a pair in the full run's shape with the router frozen (Muon, 16K windows with
# --ckpt-steps, K 4, grouped MoE, frozen routed experts and router, LR 4e-5, 1,500 + 300 steps), against
# r6_muon_w16k_noexp_rfrozen (served KL + LK cooldown): r7_lk_through, LK-hybrid throughout (the stable phase too), served KL nowhere. Early stop by sp_early_stop.sh. GPU 1 after round 6.
# usage: bash start_detached.sh logs/round7_gpu1.log sp_round7_gpu1.sh
set -u
[ "$(hostname)" = 6bc94744da90 ] || { echo "not the side pod"; exit 9; }
SP=/mnt/shared/arc3-inference/sidepod_20261012
cd $SP/scripts
until grep -q "round 6 gpu1 done" $SP/logs/round6_gpu1.log 2>/dev/null; do sleep 60; done
echo "[$(date -u +%T)] round 6 gpu1 done: r7_lk_through on GPU 1"
RUN=r7_lk_through REF=r6_muon_w16k_noexp_rfrozen bash start_detached.sh $SP/logs/early_stop_r7_lk_through.log sp_early_stop.sh
RUN=r7_lk_through LR=4e-5 DEV=cuda:1 OPT=muon TRAIN=nonexpert EXPERTS=served MOE=grouped WINDOW=16384 ACCUM=1 LOSS=lk CLOSS=lk EXTRA="--ckpt-steps" \
  bash sp_round_arm2.sh > $SP/logs/round7_r7_lk_through.log 2>&1
echo "[$(date -u +%T)] r7_lk_through: $(tail -n 1 $SP/logs/round7_r7_lk_through.log)"
echo "[$(date -u +%T)] round 7 gpu1 done"
