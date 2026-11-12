#!/bin/bash
# sp_round4_gpu0.sh (2026-10-13, Claude fork 5d20b23b; side pod only): the experts-only arm (user 10-13: run it if frozen
# experts is competitive; r2_noexp_lr4e5 came within 0.006 of all parameters at K 4): r4_experts_lr4e5, --train experts
# (routed experts, router and shared expert train; attention, input projections, norms, hyper-connections frozen),
# 4K x 4, K 4, served KL + LK cooldown, LR 4e-5, 1,500 + 300 steps, scored with re-quantised experts. Completes the
# 4K triplet all (r1_all_k4_lr4e5) / frozen experts (r2_noexp_lr4e5) / experts only. GPU 0 after sp_replay_fast8k.sh.
# usage: bash start_detached.sh logs/round4_gpu0.log sp_round4_gpu0.sh
set -u
[ "$(hostname)" = 6bc94744da90 ] || { echo "not the side pod"; exit 9; }
SP=/mnt/shared/arc3-inference/sidepod_20261012
cd $SP/scripts
until grep -q "replay fast8k done" $SP/logs/replay_fast8k_20261013.log 2>/dev/null; do sleep 60; done
echo "[$(date -u +%T)] replay fast8k done: experts-only arm on GPU 0"
RUN=r4_experts_lr4e5 LR=4e-5 DEV=cuda:0 TRAIN=experts bash sp_round_arm.sh > $SP/logs/round4_r4_experts_lr4e5.log 2>&1
echo "[$(date -u +%T)] r4_experts_lr4e5: $(tail -n 1 $SP/logs/round4_r4_experts_lr4e5.log)"
echo "[$(date -u +%T)] round 4 gpu0 done"
