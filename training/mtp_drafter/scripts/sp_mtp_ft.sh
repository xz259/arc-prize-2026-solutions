#!/bin/bash
# sp_mtp_ft.sh (2026-10-12, Claude fork 5d20b23b; side pod only): SIDE_POD_PLAN section 4 as a quick estimate (user 10-12
# ~21:5x: not the full fine-tune). train_mtp_sp.py from the production BF16 head on the 2(b) export (MTP input, ~20M
# rows), K = 3 unroll with step weights 1 / 0.8 / 0.64, CE + KL to the target (kl-weight 1), AdamW lr 2e-5 WSD (20% decay)
# with 100 warmup, windows of 8,192 rows, RoPE on the raw stream index, heads saved every 250 steps; each saved head is
# then scored on val50 by mtp_eval_val50.py (served-like experts: requant) against the baseline.
# usage: setsid nohup bash sp_mtp_ft.sh > LOG 2>&1 < /dev/null &      (env: RUN, STEPS, LR, TRAIN, KLW, DEV, DATA)
set -u
[ "$(hostname)" = 6bc94744da90 ] || { echo "not the side pod"; exit 9; }
SP=/mnt/shared/arc3-inference/sidepod_20261012
R=/workspace/arc3-inference
M=/workspace/models/Qwen3.8-Flash-Next-W4A16-intel
RUN=${RUN:-ft_all_lr2e5}
OUT=$SP/mtp_ft/$RUN
DATA=${DATA:-$SP/features/sp_mtptrain20m}
[ -e $OUT ] && { echo "exists: $OUT"; exit 1; }
mkdir -p $SP/mtp_ft
echo "[$(date -u +%T)] MTP fine-tune $RUN: steps ${STEPS:-1500}, lr ${LR:-2e-5}, train ${TRAIN:-all}, kl ${KLW:-1.0}, data $DATA ($(ls $DATA | wc -l) files)"
cd $SP/scripts && PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $R/venv-speculators/bin/python -u train_mtp_sp.py --data $DATA \
  --streams $R/dflash_streams_20261011 --mtp-file $M/model_extra_tensors.safetensors --target-dir $M --out $OUT \
  --val-files 8 --steps ${STEPS:-1500} --lr ${LR:-2e-5} --warmup 100 --schedule wsd --num-steps 3 --step-weights 1,0.8,0.64 \
  --kl-weight ${KLW:-1.0} --window ${WINDOW:-4096} --train ${TRAIN:-all} --eval-every ${EVAL:-500} --save-every ${SAVE:-500} --attention flex \
  --device ${DEV:-cuda:1} --pos-key position_ids
echo "[$(date -u +%T)] trainer rc $?"
