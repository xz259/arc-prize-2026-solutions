#!/bin/bash
# sp_queue_after_val50.sh (2026-10-12, Claude fork 5d20b23b; side pod only): once the val50 MTP export (2(a)) has finished
# and stopped its server, start the 2(b) training-subset export on GPU 0 (sp_export_mtptrain.sh, detached) and run the
# GPU 1 evaluations in sequence:
#   1. mtp_eval_val50.py: the production MTP3 baseline (served experts, QSA, lutc KV, K 5) on all 50 windows (+ anchors)
#   2. dflash_eval_window.py on the side export (taps 0-4, final 5): DFlash 15K at window 512, seeds 1-4 (matched data)
#   3. sp_compare_val50.py: side export vs job 407 (statistical equivalence)
#   4. probe_features.py: the feature probes (section 5)
#   5. mtp_eval_val50.py --attention dense, then --experts bf16 (QSA and quantization effects)
# usage: setsid nohup bash sp_queue_after_val50.sh > LOG 2>&1 < /dev/null &
set -u
[ "$(hostname)" = 6bc94744da90 ] || { echo "not the side pod"; exit 9; }
SP=/mnt/shared/arc3-inference/sidepod_20261012
PY=/workspace/arc3-inference/venv-speculators/bin/python
EXPLOG=${EXPLOG:?the val50 export log}
F=$SP/features/sp_val50mtp
R=$SP/results
log() { echo "[$(date -u +%T)] $*"; }
until grep -q "processes left" $EXPLOG; do
  pgrep -f '[s]p_export_val50mtp.sh' >/dev/null || { log "export script gone without its end line"; break; }
  sleep 30
done
grep -E "ok-window|stream tokens|feature files|processes left" $EXPLOG | tail -n 4
n=$(ls $F | wc -l)
log "val50 export files: $n"
[ "$n" = 50 ] || { log "expected 50 feature files; stopping the queue"; exit 1; }
L2B=$SP/logs/sp_mtptrain20m_$(date -u +%Y%m%dT%H%M%S).log
setsid nohup bash $SP/scripts/sp_export_mtptrain.sh > $L2B 2>&1 < /dev/null &
log "2(b) export started on GPU 0: $L2B"
cd $SP/scripts
export PYTHONDONTWRITEBYTECODE=1
log "1. MTP baseline"
$PY -u mtp_eval_val50.py --features $F --out $R/mtp_base_val50.json --save-anchors $R/mtp_base_val50_anchors.pt --device cuda:1 --K 5
log "2. DFlash 15K on the side export"
CUDA_VISIBLE_DEVICES=1 $PY -u dflash_eval_window.py --ckpt /workspace/arc3-inference/dflash_work_20261009/ckpt_v7r2e/epoch0_step15000 \
  --features $F --slots 0,1,2,3,4 --final 5 --windows 512 --seeds 1 2 3 4 --out $R/dflash15k_side_val50.json --device cuda:0
log "3. side vs main features"
$PY -u sp_compare_val50.py $F --device cuda:1 --out $R/equiv_side_vs_407.json
log "4. probes"
$PY -u probe_features.py --features $F --out $R/probes_val50.json --device cuda:1
log "5. MTP dense attention, BF16 experts"
$PY -u mtp_eval_val50.py --features $F --out $R/mtp_dense_val50.json --device cuda:1 --K 5 --attention dense
$PY -u mtp_eval_val50.py --features $F --out $R/mtp_bf16experts_val50.json --device cuda:1 --K 5 --experts bf16
log "queue done"
