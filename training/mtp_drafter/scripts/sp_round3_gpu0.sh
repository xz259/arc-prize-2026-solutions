#!/bin/bash
# sp_round3_gpu0.sh (2026-10-13, Claude fork 5d20b23b; side pod only): round 3 on GPU 0 after round 2's Muon arm:
#   1. test_moe_grouped_fp32.py (9ff96de9): grouped vs loop MoE gradients against an FP32 loop reference (adoption check for
#      the full run; the 06:22 run failed on a test bug), and their times;
#   2. r3_k5_w16k_lr4e5: K 5 at 16K windows (auxiliary step 5: weights 1,0.69,0.40,0.18,0.12; --ckpt-steps), 16K rows per step
#      (16,384 x 1), LR 4e-5, against r3_k4_w16k_lr4e5 (GPU 1) on K 4 live tau (user 10-13: K 5 only if it costs no extra
#      wall time: 4,050 vs 4,317 rows/s per trainer, both replay-bound on 4 GPUs).
# usage: bash start_detached.sh logs/round3_gpu0.log sp_round3_gpu0.sh
set -u
[ "$(hostname)" = 6bc94744da90 ] || { echo "not the side pod"; exit 9; }
SP=/mnt/shared/arc3-inference/sidepod_20261012
PY=/workspace/arc3-inference/venv-speculators/bin/python
cd $SP/scripts
export PYTHONDONTWRITEBYTECODE=1
until grep -q "round 2 gpu0 done" $SP/logs/round2_gpu0.log 2>/dev/null; do sleep 60; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 0)" -lt 2000 ]; do sleep 30; done
echo "[$(date -u +%T)] round 2 gpu0 done: grouped-MoE accuracy check"
$PY -u test_moe_grouped_fp32.py --device cuda:0 2>&1 | grep -vi "warn" | grep -v "^\s*$" | tail -n 12
echo "[$(date -u +%T)] round 3 arm on GPU 0"
RUN=r3_k5_w16k_lr4e5 LR=4e-5 DEV=cuda:0 NK=5 SW=1,0.69,0.40,0.18,0.12 WINDOW=16384 ACCUM=1 EXTRA="--ckpt-steps" \
  bash sp_round_arm.sh > $SP/logs/round3_r3_k5_w16k_lr4e5.log 2>&1
echo "[$(date -u +%T)] r3_k5_w16k_lr4e5: $(tail -n 1 $SP/logs/round3_r3_k5_w16k_lr4e5.log)"
echo "[$(date -u +%T)] round 3 gpu0 done"
