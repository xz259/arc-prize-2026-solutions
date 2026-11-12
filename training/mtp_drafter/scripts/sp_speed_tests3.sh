#!/bin/bash
# sp_speed_tests3.sh (2026-10-13, Claude fork 5d20b23b; side pod only): K 5 unroll timings and peak memory for the round-1
# arms (IE 02:2x: train the served depth), 30-step trainer runs without checkpoints on GPU 0: CFGS entries
# "tag train window accum K" (default: all parameters at 8K x 2 and 16K x 1, experts frozen at 16K), step weights from
# d tau / d alpha_k at the fine-tuned head's conditionals. Outputs: mtp_ft/speedtest3_* (log only).
# usage: setsid nohup bash sp_speed_tests3.sh > LOG 2>&1 < /dev/null &
set -u
[ "$(hostname)" = 6bc94744da90 ] || { echo "not the side pod"; exit 9; }
SP=/mnt/shared/arc3-inference/sidepod_20261012
R=/workspace/arc3-inference
M=/workspace/models/Qwen3.8-Flash-Next-W4A16-intel
PY=$R/venv-speculators/bin/python
cd $SP/scripts
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while pgrep -f '[s]p_speed_tests.sh|[s]p_eval_heads.sh' >/dev/null; do sleep 20; done
echo "[$(date -u +%T)] GPU 0 free"
CFGS=${CFGS:-"all_k5_w8k all 8192 2 5;all_k5_w16k all 16384 1 5;noexp_k5_w16k noexperts 16384 1 5"}
IFS=';' read -ra LIST <<< "$CFGS"
for CFG in "${LIST[@]}"; do
  set -- $CFG
  SW=1,0.6,0.26; [ $5 = 5 ] && SW=1,0.73,0.48,0.28,0.12; [ $5 = 4 ] && SW=1,0.69,0.40,0.18
  OUT=$SP/mtp_ft/speedtest3_$1
  [ -e $OUT ] && { echo "exists: $OUT"; continue; }
  echo "[$(date -u +%T)] trainer speed test $1 (train $2, window $3 x accum $4, K $5)"
  $PY -u train_mtp_sp.py --data $SP/features/sp_mtptrain20m --streams $R/dflash_streams_20261011 \
    --mtp-file $M/model_extra_tensors.safetensors --target-dir $M --out $OUT --val-files 8 --steps 30 --lr 4e-5 --warmup 5 \
    --schedule wsd --num-steps $5 --step-weights $SW --loss served_kl --window $3 --accum $4 --train $2 \
    --router-lr-scale 0.3 --compile-flex --fused-adam --eval-every 0 --save-every 0 --attention flex --device ${DEV:-cuda:0} \
    --pos-key position_ids 2>&1 | grep -E "^train windows|Traceback|Error|out of memory" | head -5
  $PY - $OUT <<'PYEOF'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1] + '/log.jsonl') if '"loss"' in l]
if len(rows) >= 30:
    a = json.loads(open(sys.argv[1] + '/args.json').read())
    per = (rows[29]['sec'] - rows[9]['sec']) / 20
    print('  %s: %.2f s/step over steps 11-30 (window %d, accum %d) = %.0f rows/s; last: kl1 %.3f alpha1 %.3f w2 %.3f w3 %.3f load_entropy %.3f peak %.1f GiB' % (
        sys.argv[1].rsplit('/', 1)[1], per, a['window'], a['accum'], a['window'] * a['accum'] / per, rows[-1]['kl_step1'],
        rows[-1]['alpha_step1'], rows[-1].get('w_step2', 1), rows[-1].get('w_step3', 1), rows[-1]['load_entropy'], rows[-1]['mem_gb']))
else:
    print('  only %d steps logged' % len(rows))
PYEOF
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | head -1
done
echo "[$(date -u +%T)] speed tests done"
