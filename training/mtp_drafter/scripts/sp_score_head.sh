#!/bin/bash
# 2026-10-22 MTP refit port (Claude session 099768dc; from 2fe24f23): host 2a159825c76b; SP = mtp_refit_20261022; the default set is the
# next55 holdout (valnext55: features VFN, pack SP/jobs/valnext55/pack); valrbp50v / val50 read fr1's root (SPOLD); H=base scores
# the untrained MTP head (no --head); SCORE_EXTRA is passed to the scorer (its defaults: lutc with the dkv73 means, draftwin 16,384).
# sp_score_head.sh (2026-10-14, Claude fork 5d20b23b; side pod only): score one head file exactly as the milestone branches
# do (sp_branch_cooldown.sh: production emulation, served NVFP4 experts, live renewal tau K 3-5 at T 1 / 0.6), for heads
# made by state_to_head.py from in-decay states (single or averaged) and the final head. Writes only results/<TAG>_*.
# usage: H=<head.safetensors> TAG=<name> DEV=cuda:1 [SETS="valrbp50v val50"] [SWEEP=k:p,...] [ISHARE=0|1] [DRY=1] \
#          bash start_detached.sh logs/score_<TAG>.log sp_score_head.sh
set -u
[ "$(hostname)" = 2a159825c76b ] || { echo "not the main pod"; exit 9; }
SP=/mnt/shared/arc3-inference/mtp_refit_20261022
SPOLD=/mnt/shared/arc3-inference/sidepod_20261012
VFN=${VFN:-/workspace/mtp_refit_r1/features/sp_valnext55mtp}
PY=/workspace/arc3-inference/venv-speculators/bin/python
H=${H:?head (a head file, or base = the untrained MTP head)}; TAG=${TAG:?tag}; DEV=${DEV:?device}; SETS=${SETS:-valnext55}; SWEEP=${SWEEP:-}
if [ "$H" = base ]; then HD=""; else [ -s $H ] || { echo "no head $H"; exit 1; }; HD="--head $H"; fi
cd $SP/scripts
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
run() { if [ -n "${DRY:-}" ]; then echo "DRY: $*"; else eval "$@"; fi; }
echo "[$(date -u +%T)] scoring $H as $TAG on $DEV (sets $SETS${SWEEP:+, sweep $SWEEP}${ISHARE:+, index share $ISHARE})"
for VS in $SETS; do
  case $VS in
    valnext55) VF="--features $VFN --pack $SP/jobs/valnext55/pack" ;;
    valrbp50v) VF="--features $SPOLD/features/sp_valrbp50mtp --pack $SPOLD/jobs/valrbp50v/pack" ;;
    *)         VF="--features $SPOLD/features/sp_val50mtp" ;;
  esac
  T=${TAG}_${VS}_served
  TS=""; [ $VS != val50 ] && [ -n "$SWEEP" ] && TS="--trunc-sweep $SWEEP"
  run "$PY -u mtp_eval_val50.py $VF $HD --experts served --device $DEV $TS --K 5 ${SCORE_EXTRA:-} --out $SP/results/$T.json \
    --save-anchors $SP/results/${T}_anchors.pt 2>&1 | grep -E '^prob_srv_T|^done|Traceback|Error'"
  run "$PY live_tau.py $SP/results/${T}_anchors.pt --modes prob_srv_T10 prob_srv_T06 --K 3 4 5"
  [ -n "$TS" ] && run "$PY live_tau.py $SP/results/${T}_anchors.pt --modes $(echo $SWEEP | tr ',' '\n' | \
    awk -F: '{printf "prob_srv_T10_k%dp%02d ", $1, $2 * 100 + 0.5}') --K 4"
done
if [ "${ISHARE:-0}" = 1 ]; then
  T=${TAG}_valnext55_ishare
  run "$PY -u mtp_eval_val50.py --features $VFN --pack $SP/jobs/valnext55/pack $HD --experts served \
    --device $DEV --index-share --K 5 --out $SP/results/$T.json --save-anchors $SP/results/${T}_anchors.pt 2>&1 \
    | grep -E '^prob_srv_T|^done|Traceback|Error'"
  run "$PY live_tau.py $SP/results/${T}_anchors.pt --modes prob_srv_T10 prob_srv_T06 --K 3 4 5"
fi
echo "[$(date -u +%T)] scoring $TAG done"
