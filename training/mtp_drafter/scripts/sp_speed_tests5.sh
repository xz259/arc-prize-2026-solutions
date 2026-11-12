#!/bin/bash
# sp_speed_tests5.sh (2026-10-13, Claude fork 5d20b23b; side pod only): trainer speedups before round 2, on GPU 0 once its
# round-1 arm (training + scoring) is done: (1) test_moe_grouped_fp32.py: grouped vs loop MoE gradients against an FP32
# loop reference, their time at 4,096 rows K 4, and the window load time; (2) 30-step timings of the round-1 recipe
# (all parameters, K 4, 4K x 4) with the served target computed once per row (train_mtp_sp.py dc0b8b9a) and prefetch 2,
# with the loop MoE and with the grouped MoE. Round 1 measured 6.0 s/step with the previous trainer.
# usage: setsid nohup bash sp_speed_tests5.sh > LOG 2>&1 < /dev/null &
set -u
[ "$(hostname)" = 6bc94744da90 ] || { echo "not the side pod"; exit 9; }
SP=/mnt/shared/arc3-inference/sidepod_20261012
R=/workspace/arc3-inference
M=/workspace/models/Qwen3.8-Flash-Next-W4A16-intel
PY=$R/venv-speculators/bin/python
cd $SP/scripts
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
until grep -q "arm r1_all_k4_lr4e5 done" $SP/logs/round1_r1_all_k4_lr4e5.log 2>/dev/null; do sleep 60; done
echo "[$(date -u +%T)] GPU 0 free: grouped-MoE accuracy and timing"
$PY -u test_moe_grouped_fp32.py --device cuda:0 2>&1 | grep -vi "warn"
for CFG in "srvonce_loop loop" "srvonce_grouped grouped"; do
  set -- $CFG
  OUT=$SP/mtp_ft/speedtest5_$1
  [ -e $OUT ] && { echo "exists: $OUT"; continue; }
  echo "[$(date -u +%T)] trainer timing $1 (MoE $2)"
  MTP_MOE=$2 $PY -u train_mtp_sp.py --data $SP/features/sp_mtptrain20m --streams $R/dflash_streams_20261011 \
    --mtp-file $M/model_extra_tensors.safetensors --target-dir $M --out $OUT --val-files 8 --steps 30 --lr 4e-5 --warmup 5 \
    --schedule wsd --num-steps 4 --step-weights 1,0.69,0.40,0.18 --loss served_kl --window 4096 --accum 4 --train all \
    --compile-flex --fused-adam --prefetch 2 --eval-every 0 --save-every 0 --attention flex --device cuda:0 \
    --pos-key position_ids 2>&1 | grep -E "^train windows|Traceback|Error|out of memory" | head -5
  $PY - $OUT <<'PYEOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1] + '/log.jsonl') if '"loss"' in l]
if len(rows) >= 30:
    per = (rows[29]['sec'] - rows[9]['sec']) / 20
    print('  %s: %.2f s/step over steps 11-30 = %.0f rows/s (round 1: 6.0 s/step); peak %.1f GiB; kl1 %.3f' % (
        sys.argv[1].rsplit('/', 1)[1], per, 16384 / per, rows[-1]['mem_gb'], rows[-1]['kl_step1']))
else:
    print('  only %d steps logged' % len(rows))
PYEOF
done
echo "[$(date -u +%T)] speed tests 5 done"
