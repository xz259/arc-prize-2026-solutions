#!/bin/bash
# sp_speed_tests.sh (2026-10-13, Claude fork 5d20b23b; side pod only): before the recipe sweep, on GPU 0 once the head
# scorer is done: (1) compiled vs eager flex attention (test_flex_compiled.py: agreement, time, memory at 4K / 8K / 16K
# rows); (2) 30-step trainer runs without checkpoints (--save-every 0, --eval-every 0) to measure seconds per window and
# peak memory: experts frozen + served KL + accepted-path weights + compiled flex + fused AdamW at 4,096 and 16,384-row
# windows, and all parameters at 16,384 (does it fit). Outputs: logs/speed_tests_20261013.log, mtp_ft/speedtest_* (log only).
# usage: setsid nohup bash sp_speed_tests.sh > LOG 2>&1 < /dev/null &
set -u
[ "$(hostname)" = 6bc94744da90 ] || { echo "not the side pod"; exit 9; }
SP=/mnt/shared/arc3-inference/sidepod_20261012
R=/workspace/arc3-inference
M=/workspace/models/Qwen3.8-Flash-Next-W4A16-intel
PY=$R/venv-speculators/bin/python
cd $SP/scripts
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f '[s]p_eval_heads.sh' >/dev/null; do sleep 30; done
echo "[$(date -u +%T)] GPU 0 free; flex test"
$PY -u test_flex_compiled.py --device cuda:0 2>&1 | grep -v Warning
for CFG in "noexp_w4k noexperts 4096" "noexp_w16k noexperts 16384" "all_w16k all 16384"; do
  set -- $CFG
  OUT=$SP/mtp_ft/speedtest2_$1
  [ -e $OUT ] && { echo "exists: $OUT"; continue; }
  echo "[$(date -u +%T)] trainer speed test $1 (train $2, window $3)"
  $PY -u train_mtp_sp.py --data $SP/features/sp_mtptrain20m --streams $R/dflash_streams_20261011 \
    --mtp-file $M/model_extra_tensors.safetensors --target-dir $M --out $OUT --val-files 8 --steps 30 --lr 4e-5 --warmup 5 \
    --schedule wsd --num-steps 3 --step-weights 1,0.6,0.26 --loss served_kl --accept-weights --window $3 --train $2 \
    --router-lr-scale 0.3 --compile-flex --fused-adam --eval-every 0 --save-every 0 --attention flex --device cuda:0 \
    --pos-key position_ids 2>&1 | grep -E "^train windows|Traceback|Error|out of memory" | head -5
  $PY - $OUT <<'PYEOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1] + '/log.jsonl') if '"loss"' in l]
if len(rows) >= 30:
    a = json.loads(open(sys.argv[1] + '/args.json').read())
    per = (rows[29]['sec'] - rows[9]['sec']) / 20
    print('  %s: %.2f s/step over steps 11-30 (window %d, accum %d) = %.0f rows/s; last: kl1 %.3f alpha1 %.3f w2 %.3f w3 %.3f load_entropy %.3f peak %.1f GiB' % (
        sys.argv[1].rsplit('/', 1)[1], per, a['window'], a['accum'], a['window'] * a['accum'] / per, rows[-1]['kl_step1'],
        rows[-1]['alpha_step1'], rows[-1]['w_step2'], rows[-1]['w_step3'], rows[-1]['load_entropy'], rows[-1]['mem_gb']))
else:
    print('  only %d steps logged' % len(rows))
PYEOF
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -1
done
echo "[$(date -u +%T)] speed tests done"
