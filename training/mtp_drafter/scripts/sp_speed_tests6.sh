#!/bin/bash
# sp_speed_tests6.sh (2026-10-13, Claude fork 5d20b23b; side pod only): step times of the full-run shapes the user asked
# about (K 4 / K 5 x 4K / 16K windows), all parameters, 16K rows per step, the round-2 trainer (served target once per
# row, prefetch 2, loop MoE), on GPU 1 once its round-1 arm (training + scoring) is done and before round 2's arm there.
# K 5 and 16K windows need --ckpt-steps for memory; K 4 at 4K x 4 runs with and without it (the checkpointing overhead).
# K 5 uses the auxiliary-step weights (K 4 serving weights + 0.12). 30 steps each, timed over steps 11-30.
# Writes only mtp_ft/speedtest6_* and its own log.
# usage: setsid nohup bash sp_speed_tests6.sh > logs/speed_tests6_20261013.log 2>&1 < /dev/null &
set -u
[ "$(hostname)" = 6bc94744da90 ] || { echo "not the side pod"; exit 9; }
SP=/mnt/shared/arc3-inference/sidepod_20261012
R=/workspace/arc3-inference
M=/workspace/models/Qwen3.8-Flash-Next-W4A16-intel
PY=$R/venv-speculators/bin/python
cd $SP/scripts
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q "arm r1_all_k4_lr1e4 done" $SP/logs/round1_r1_all_k4_lr1e4.log 2>/dev/null; do sleep 60; done
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1)" -lt 2000 ]; do sleep 30; done
echo "[$(date -u +%T)] GPU 1 free: step times K 4 / K 5 x 4K / 16K windows"
for CFG in "k4_w4k 4 1,0.69,0.40,0.18 4096 4 -" \
           "k4_w4k_ckpt 4 1,0.69,0.40,0.18 4096 4 --ckpt-steps" \
           "k5_w4k_ckpt 5 1,0.69,0.40,0.18,0.12 4096 4 --ckpt-steps" \
           "k4_w16k_ckpt 4 1,0.69,0.40,0.18 16384 1 --ckpt-steps" \
           "k5_w16k_ckpt 5 1,0.69,0.40,0.18,0.12 16384 1 --ckpt-steps"; do
  read -r NAME NK SW WIN ACC CK <<< "$CFG"
  [ "$CK" = - ] && CK=""
  OUT=$SP/mtp_ft/speedtest6_$NAME
  [ -e $OUT ] && { echo "exists: $OUT"; continue; }
  echo "[$(date -u +%T)] trainer timing $NAME"
  MTP_MOE=loop $PY -u train_mtp_sp.py --data $SP/features/sp_mtptrain20m --streams $R/dflash_streams_20261011 \
    --mtp-file $M/model_extra_tensors.safetensors --target-dir $M --out $OUT --val-files 8 --steps 30 --lr 4e-5 --warmup 5 \
    --schedule wsd --num-steps $NK --step-weights $SW --loss served_kl --window $WIN --accum $ACC --train all $CK \
    --compile-flex --fused-adam --prefetch 2 --eval-every 0 --save-every 0 --attention flex --device cuda:1 \
    --pos-key position_ids 2>&1 | grep -E "^train windows|Traceback|Error|out of memory" | head -5
  $PY - $OUT <<'PYEOF'
import json, os, sys
p = sys.argv[1] + '/log.jsonl'
rows = [json.loads(l) for l in open(p) if '"loss"' in l] if os.path.exists(p) else []
if len(rows) >= 30:
    per = (rows[29]['sec'] - rows[9]['sec']) / 20
    print('  %s: %.2f s/step over steps 11-30 = %.0f rows/s (round 1: 6.0 s/step); peak %.1f GiB; kl1 %.3f' % (
        sys.argv[1].rsplit('/', 1)[1], per, 16384 / per, rows[-1]['mem_gb'], rows[-1]['kl_step1']))
else:
    print('  %s: only %d steps logged' % (sys.argv[1].rsplit('/', 1)[1], len(rows)))
PYEOF
done
echo "[$(date -u +%T)] speed tests 6 done"
