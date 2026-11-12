#!/bin/bash
# sp_round6_gpu0.sh (2026-10-13, Claude fork 5d20b23b; side pod only): the last ablation before the full run (user 10-13), one arm
# of a pair in the full run's shape (Muon on the 2D non-expert matrices, 16K windows with --ckpt-steps, K 4, grouped MoE,
# LR 4e-5 for comparability with the other arms, 1,500 + 300 steps): r6_muon_w16k_noexp_rfrozen, frozen routed experts and frozen router (user 10-13: routing changes are small and can be absorbed by the other parameters).
# Its pair: r6_muon_w16k_noexp_r03 (GPU 1, router 0.3x) vs r6_muon_w16k_noexp_rfrozen (GPU 0, router frozen); scored with the
# served NVFP4 experts on val50. GPU 0 after round 5 there.
# usage: bash start_detached.sh logs/round6_gpu0.log sp_round6_gpu0.sh
set -u
[ "$(hostname)" = 6bc94744da90 ] || { echo "not the side pod"; exit 9; }
SP=/mnt/shared/arc3-inference/sidepod_20261012
cd $SP/scripts
until grep -q "round 5 gpu0 done" $SP/logs/round5_gpu0.log 2>/dev/null; do sleep 60; done
echo "[$(date -u +%T)] round 5 gpu0 done: r6_muon_w16k_noexp_rfrozen on GPU 0"
RUN=r6_muon_w16k_noexp_rfrozen LR=4e-5 DEV=cuda:0 OPT=muon TRAIN=nonexpert ROUTER=0.3 EXPERTS=served MOE=grouped WINDOW=16384 ACCUM=1 EXTRA="--ckpt-steps" \
  bash sp_round_arm.sh > $SP/logs/round6_r6_muon_w16k_noexp_rfrozen.log 2>&1
echo "[$(date -u +%T)] r6_muon_w16k_noexp_rfrozen: $(tail -n 1 $SP/logs/round6_r6_muon_w16k_noexp_rfrozen.log)"
echo "[$(date -u +%T)] round 6 gpu0 done"
