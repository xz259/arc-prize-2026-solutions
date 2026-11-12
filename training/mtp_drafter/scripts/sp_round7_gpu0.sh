#!/bin/bash
# sp_round7_gpu0.sh (2026-10-13, Claude fork 5d20b23b; side pod only): loss ablation (user 10-13: ablate LK-hybrid, stop early if
# the early sign is not good), one arm of a pair in the full run's shape with the router frozen (Muon, 16K windows with
# --ckpt-steps, K 4, grouped MoE, frozen routed experts and router, LR 4e-5, 1,500 + 300 steps), against
# r6_muon_w16k_noexp_rfrozen (served KL + LK cooldown): r7_acceptw, served KL with accepted-path weights w_k = prod_{j<k} min(1, q'_j / p'_j) (detached), LK cooldown. Early stop by sp_early_stop.sh. GPU 0 after round 6.
# usage: bash start_detached.sh logs/round7_gpu0.log sp_round7_gpu0.sh
set -u
[ "$(hostname)" = 6bc94744da90 ] || { echo "not the side pod"; exit 9; }
SP=/mnt/shared/arc3-inference/sidepod_20261012
cd $SP/scripts
until grep -q "round 6 gpu0 done" $SP/logs/round6_gpu0.log 2>/dev/null; do sleep 60; done
echo "[$(date -u +%T)] round 6 gpu0 done: r7_acceptw on GPU 0"
RUN=r7_acceptw REF=r6_muon_w16k_noexp_rfrozen bash start_detached.sh $SP/logs/early_stop_r7_acceptw.log sp_early_stop.sh
RUN=r7_acceptw LR=4e-5 DEV=cuda:0 OPT=muon TRAIN=nonexpert EXPERTS=served MOE=grouped WINDOW=16384 ACCUM=1 LOSS=served_kl CLOSS=lk ACCEPT=--accept-weights EXTRA="--ckpt-steps" \
  bash sp_round_arm2.sh > $SP/logs/round7_r7_acceptw.log 2>&1
echo "[$(date -u +%T)] r7_acceptw: $(tail -n 1 $SP/logs/round7_r7_acceptw.log)"
echo "[$(date -u +%T)] round 7 gpu0 done"
