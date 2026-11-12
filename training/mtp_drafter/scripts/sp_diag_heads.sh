#!/bin/bash
# sp_diag_heads.sh (2026-10-13, Claude fork 5d20b23b; side pod only): where the fine-tuned head loses slots 2-5 on val50.
# The served emulation (experts re-quantized to NVFP4, QSA, lutc KV) shows the step-1000 head below the baseline at slots
# 2-5 (untruncated: -1.4 / -2.6 / -3.4 / -4.0 points), while the trainer's own validation (BF16 experts, dense attention,
# BF16 KV, held-out training-corpus windows) shows slots 2-3 up. Scores one head of the run with its BF16 experts (QSA and
# lutc KV as served: isolates the re-quantization), then BF16 experts with dense attention and BF16 KV (the trainer's
# setting on val50: isolates attention / KV; what remains is data and context length), each followed by live_tau.
# GPU 1, after the trainer exits.
# usage: RUN=ft_all_lr2e5d N=2500 setsid nohup bash sp_diag_heads.sh > LOG 2>&1 < /dev/null &
set -u
[ "$(hostname)" = 6bc94744da90 ] || { echo "not the side pod"; exit 9; }
SP=/mnt/shared/arc3-inference/sidepod_20261012
PY=/workspace/arc3-inference/venv-speculators/bin/python
RUN=${RUN:-ft_all_lr2e5d}
N=${N:-2500}
H=$SP/mtp_ft/$RUN/head_step$N.safetensors
F=$SP/features/sp_val50mtp
cd $SP/scripts
export PYTHONDONTWRITEBYTECODE=1
while pgrep -f '[t]rain_mtp_sp.py' >/dev/null; do sleep 60; done
[ -s $H ] || { echo "[$(date -u +%T)] no $H"; exit 1; }
echo "[$(date -u +%T)] trainer gone; diagnostics on $H"
for V in "bf16 qsa lutc" "bf16 dense bf16"; do
  set -- $V
  TAG=${RUN}_step${N}_$1_$2_$3
  echo "[$(date -u +%T)] scoring experts $1, attention $2, kv $3"
  $PY -u mtp_eval_val50.py --features $F --head $H --experts $1 --attention $2 --kv $3 --device cuda:1 --K 5 \
    --out $SP/results/mtp_diag_$TAG.json --save-anchors $SP/results/mtp_diag_${TAG}_anchors.pt 2>&1 | grep -E "^prob_srv_T|^done|Traceback|Error"
  $PY live_tau.py $SP/results/mtp_diag_${TAG}_anchors.pt --modes prob_srv_T10 prob_srv_T06 --K 3 4
done
echo "[$(date -u +%T)] diagnostics done"
