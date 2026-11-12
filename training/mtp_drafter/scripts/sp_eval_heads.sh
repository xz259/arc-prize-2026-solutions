#!/bin/bash
# sp_eval_heads.sh (2026-10-12, Claude fork 5d20b23b; side pod only): scores each head the quick MTP fine-tune saves
# (head_stepN.safetensors) on val50 as the baseline was scored (mtp_eval_val50.py: QSA, lutc KV, drafthead subset +
# DRAFTTRUNC, probabilistic; experts re-quantized to NVFP4 as export_mtp_drafter.py would serve them), then its
# live-process tau (live_tau.py). GPU 0 (the trainer is on GPU 1).
# usage: RUN=ft_all_lr2e5c setsid nohup bash sp_eval_heads.sh > LOG 2>&1 < /dev/null &   (env: STEPS "250 500 ...", EXPERTS requant|served|bf16, DEV)
set -u
[ "$(hostname)" = 6bc94744da90 ] || { echo "not the side pod"; exit 9; }
SP=/mnt/shared/arc3-inference/sidepod_20261012
PY=/workspace/arc3-inference/venv-speculators/bin/python
RUN=${RUN:?run}
D=$SP/mtp_ft/$RUN; SFX=${SUFFIX:-}
F=$SP/features/sp_val50mtp
cd $SP/scripts
export PYTHONDONTWRITEBYTECODE=1
for N in ${STEPS:-250 500 750 1000 1250 1500 1750 2000}; do
  H=$D/head_step$N.safetensors
  until [ -s $H ] && grep -q "saved head_step$N " $SP/logs/mtp_ft_${RUN#ft_}_*.log 2>/dev/null; do
    pgrep -f '[t]rain_mtp_sp.py' >/dev/null || { [ -s $H ] || { echo "[$(date -u +%T)] trainer gone before step $N"; exit 0; }; break; }
    sleep 60
  done
  echo "[$(date -u +%T)] scoring $H"
  $PY -u mtp_eval_val50.py --features $F --head $H --experts ${EXPERTS:-requant} --device ${DEV:-cuda:0} --K 5 \
    --out $SP/results/mtp_${RUN}${SFX}_step${N}.json --save-anchors $SP/results/mtp_${RUN}${SFX}_step${N}_anchors.pt 2>&1 | grep -E "^(prob_srv|greedy_srv)_T|^done|Traceback|Error"
  $PY live_tau.py $SP/results/mtp_${RUN}${SFX}_step${N}_anchors.pt --modes prob_srv_T10 prob_srv_T06 --K 3 4 5
done
echo "[$(date -u +%T)] all heads scored"
