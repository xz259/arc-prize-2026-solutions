#!/bin/bash
# sp_round2_gpu0.sh (2026-10-13, Claude fork 5d20b23b; side pod only): round 2 on GPU 0 after the GPU 0 chain (speed tests 5
# -> valrbp export -> replay speed test, which stops its server): r2_muon_lr4e5 (sp_round_arm.sh with OPT=muon:
# torch.optim.Muon on the 2D non-expert matrices, match-RMS LR, momentum 0.95, weight decay 0; AdamW on the rest),
# compared with round 1's r1_all_k4_lr4e5 (AdamW, same LR and recipe).
# usage: bash start_detached.sh logs/round2_gpu0.log sp_round2_gpu0.sh
set -u
[ "$(hostname)" = 6bc94744da90 ] || { echo "not the side pod"; exit 9; }
SP=/mnt/shared/arc3-inference/sidepod_20261012
cd $SP/scripts
until grep -q "replay speed test done" $SP/logs/replay_speedtest_20261013.log 2>/dev/null; do sleep 60; done
echo "[$(date -u +%T)] replay speed test done: round 2 on GPU 0"
RUN=r2_muon_lr4e5 LR=4e-5 DEV=cuda:0 OPT=muon bash sp_round_arm.sh > $SP/logs/round2_r2_muon_lr4e5.log 2>&1
echo "[$(date -u +%T)] r2_muon_lr4e5: $(tail -n 1 $SP/logs/round2_r2_muon_lr4e5.log)"
echo "[$(date -u +%T)] round 2 gpu0 done"
