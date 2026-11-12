#!/bin/bash
# sp_mtp_ft2.sh (2026-10-13, Claude fork 5d20b23b; side pod only): one arm of the MTP recipe sweep agreed with Inference
# Engineering (IE_REPLY_MTP_RECIPE_2026-10-13.md): train_mtp_sp.py on the 2(b) export with the served-pair loss on the
# 32K subset (KL(p'_in || q_in); LK-hybrid in the cooldown), step weights 1 / 0.60 / 0.26, ~16K rows per step, WSD with 5%
# warmup and a 300-step cooldown after 1,500 stable steps, compiled flex attention and fused AdamW; experts frozen (arm A,
# router at 0.3x) or all. Heads at 900 / 1,200 / 1,500 / 1,800 (trained tensors only when the experts are frozen); scoring by sp_eval_heads.sh (EXPERTS served for arm A).
# usage: RUN=sw_A_lr4e5 LR=4e-5 TRAIN=noexperts DEV=cuda:0 setsid nohup bash sp_mtp_ft2.sh > LOG 2>&1 < /dev/null &
#        (env: NK 3, STEPS 1800, DECAY 0.1667, WARMUP 90, WINDOW 16384, ACCUM 1, LOSS served_kl, CLOSS lk, SW 1,0.6,0.26,
#         ACCEPT "" or --accept-weights, ROUTER 0.3, SAVE_AT 900,1200,1500,1800, EVAL 300, EXTRA "")
set -u
[ "$(hostname)" = 6bc94744da90 ] || { echo "not the side pod"; exit 9; }
SP=/mnt/shared/arc3-inference/sidepod_20261012
R=/workspace/arc3-inference
M=/workspace/models/Qwen3.8-Flash-Next-W4A16-intel
RUN=${RUN:?run}; LR=${LR:?lr}; TRAIN=${TRAIN:?train}; DEV=${DEV:?device}
OUT=$SP/mtp_ft/$RUN
[ -e $OUT ] && { echo "exists: $OUT"; exit 1; }
echo "[$(date -u +%T)] MTP sweep arm $RUN: lr $LR, train $TRAIN, window ${WINDOW:-16384} x accum ${ACCUM:-1}, loss ${LOSS:-served_kl} / cooldown ${CLOSS:-lk}, steps ${STEPS:-1800} (decay ${DECAY:-0.1667}), weights ${SW:-1,0.6,0.26} ${ACCEPT:-}, $DEV"
cd $SP/scripts && PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $R/venv-speculators/bin/python -u train_mtp_sp.py \
  --data $SP/features/sp_mtptrain20m --streams $R/dflash_streams_20261011 --mtp-file $M/model_extra_tensors.safetensors \
  --target-dir $M --out $OUT --val-files 8 --steps ${STEPS:-1800} --lr $LR --warmup ${WARMUP:-90} --schedule wsd \
  --decay-ratio ${DECAY:-0.1667} --num-steps ${NK:-3} --step-weights ${SW:-1,0.6,0.26} ${ACCEPT:-} --loss ${LOSS:-served_kl} \
  --cooldown-loss ${CLOSS:-lk} --window ${WINDOW:-16384} --accum ${ACCUM:-1} --train $TRAIN --router-lr-scale ${ROUTER:-0.3} \
  --eval-every ${EVAL:-300} --save-at ${SAVE_AT:-900,1200,1500,1800} --save-trainable-only --attention flex --compile-flex --fused-adam --device $DEV \
  --pos-key position_ids ${EXTRA:-}
echo "[$(date -u +%T)] trainer rc $?"
