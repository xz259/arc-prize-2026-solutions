#!/bin/bash
# sp_round5_gpu0.sh (2026-10-13, Claude fork 5d20b23b; side pod only): the rerun of round 3's K 5 at 16K arm (r3_k5_w16k_lr4e5
# ran out of memory at step ~135: torch.compile hit its recompile limit on the 9th padded window length and flex attention
# fell back to the eager dense backward; fixed in train_mtp_sp.py cf6ac662: 2,048-row padding, recompile limit 64,
# test_flex_buckets_gpu.py). r3_k5_w16k_lr4e5b: K 5 (weights 1,0.69,0.40,0.18,0.12), 16,384 x 1, --ckpt-steps, LR 4e-5,
# AdamW, 1,500 + 300 steps, against r3_k4_w16k_lr4e5 on K 4 live tau. GPU 0 after the experts-only arm.
# usage: bash start_detached.sh logs/round5_gpu0.log sp_round5_gpu0.sh
set -u
[ "$(hostname)" = 6bc94744da90 ] || { echo "not the side pod"; exit 9; }
SP=/mnt/shared/arc3-inference/sidepod_20261012
cd $SP/scripts
until grep -q "round 4 gpu0 done" $SP/logs/round4_gpu0.log 2>/dev/null; do sleep 60; done
echo "[$(date -u +%T)] round 4 gpu0 done: K 5 at 16K rerun on GPU 0"
RUN=r3_k5_w16k_lr4e5b LR=4e-5 DEV=cuda:0 NK=5 SW=1,0.69,0.40,0.18,0.12 WINDOW=16384 ACCUM=1 EXTRA="--ckpt-steps" \
  bash sp_round_arm.sh > $SP/logs/round5_r3_k5_w16k_lr4e5b.log 2>&1
echo "[$(date -u +%T)] r3_k5_w16k_lr4e5b: $(tail -n 1 $SP/logs/round5_r3_k5_w16k_lr4e5b.log)"
echo "[$(date -u +%T)] round 5 gpu0 done"
