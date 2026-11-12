#!/bin/bash
# sp_branch_cooldown.sh (2026-10-13, Claude fork 5d20b23b; side pod only): a milestone measurement of the MTP full run (user
# 10-13 04:0x: anneal by branched cooldowns, not by a fixed date): from a kept stable-phase state (state_keep_stepN.pt) a
# short cooldown branch (B steps, the LR decaying from the stable value to 0 over the branch, the run's cooldown loss) on a
# fixed dataset (the 20M-token export: 55 sessions of 54 games, game-stratified by construction; the same for every
# milestone, so branches compare), then the branch head scored on both validation sets (valrbp50v, the full run's, and
# val50, the rounds', so the two stay comparable; production emulation; live tau K 3-5, T 1 / 0.6) with its experts
# re-quantised to NVFP4 (and as BF16 when the experts train). IE 10-13: ~500-step branches at the full batch (16,384 x 3
# rows), equal length at every milestone; hard stop when a milestone's cooled tau falls > ~0.5% below the previous one.
# The main run is not touched; the branch writes only its own directory mtp_ft/<RUN>_branch<N>.
# usage: RUN=fr1 N=9000 DEV=cuda:1 setsid nohup bash sp_branch_cooldown.sh > LOG 2>&1 < /dev/null &
#        (env: B 375, plus the run's recipe as in sp_fullrun.sh: LR, TRAIN, OPT, NK, SW, LOSS, CLOSS, ACCEPT, WINDOW 16384,
#         ACCUM 4, MINWIN 2048, TOKW 1, ROUTER, MOE grouped, CKPTSTEPS --ckpt-steps, EXTRA, EXPW (served for frozen experts, else source),
#         UNFREEZE 0 / 1 (a state from after the escape hatch: experts trained, scored requant and BF16), EXPERT_LR 0.5,
#         EXPERT_WARMUP 200, SWEEP (draft truncations K:P on valrbp50v), ISHARE 1 (an index-share pass); WAIT_GPU=1 waits)
set -u
[ "$(hostname)" = 6bc94744da90 ] || { echo "not the side pod"; exit 9; }
SP=/mnt/shared/arc3-inference/sidepod_20261012
R=/workspace/arc3-inference
M=/workspace/models/Qwen3.8-Flash-Next-W4A16-intel
PY=$R/venv-speculators/bin/python
RUN=${RUN:?run}; N=${N:?step}; DEV=${DEV:?device}; LR=${LR:?lr}
B=${B:-375}                      # 1,500 windows at 4 per step (the planned 500 x 3)
SWEEP=${SWEEP-10:0.95,40:0.95,20:0.90,20:0.98}   # draft-truncation sweep on valrbp50v (user 10-13: if cheap); "" = off
TW=""; [ "${TOKW:-1}" = 1 ] && TW="--token-weighted"
case "${TRAIN:-all}" in noexperts|nonexpert) EXPW=${EXPW:-served};; *) EXPW=${EXPW:-source};; esac
UNF=""; [ "${UNFREEZE:-0}" = 1 ] && UNF="--unfreeze-experts --expert-lr-scale ${EXPERT_LR:-0.5} --expert-warmup ${EXPERT_WARMUP:-200}"
STATE=$SP/mtp_ft/$RUN/state_keep_step$N.pt
OUT=$SP/mtp_ft/${RUN}_branch$N${SUFFIX:-}                # SUFFIX: a rerun beside a failed branch's directory
G=${DEV#cuda:}
cd $SP/scripts
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
[ -s $STATE ] || { echo "no $STATE"; exit 1; }
[ -e $OUT ] && { echo "exists: $OUT"; exit 1; }
if [ -n "${WAIT_GPU:-}" ]; then until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $G)" -lt 2000 ]; do sleep 30; done; fi
echo "[$(date -u +%T)] branch cooldown of $RUN at step $N: $B steps on sp_mtptrain20m, $DEV"
MTP_MOE=${MOE:-grouped} $PY -u train_mtp_sp.py --data $SP/features/sp_mtptrain20m --val-files 0 --streams $R/dflash_streams_20261011 \
  --mtp-file $M/model_extra_tensors.safetensors --target-dir $M --out $OUT --resume $STATE \
  --steps $((N + B)) --decay-start $((N + 1)) --lr $LR --warmup 0 --schedule wsd \
  --num-steps ${NK:-4} --step-weights ${SW:-1,0.69,0.40,0.18} ${ACCEPT:-} --loss ${CLOSS:-lk} --cooldown-loss ${CLOSS:-lk} \
  --window ${WINDOW:-16384} --accum ${ACCUM:-4} --min-window ${MINWIN:-2048} $TW --train ${TRAIN:-all} --expert-weights $EXPW $UNF --router-lr-scale ${ROUTER:-1.0} --optimizer ${OPT:-adamw} \
  --eval-every 0 --save-at $((N + B)) --save-trainable-only --attention flex --compile-flex --fused-adam --prefetch 2 \
  --device $DEV --pos-key position_ids ${CKPTSTEPS---ckpt-steps} ${EXTRA:-}
RC=$?; echo "[$(date -u +%T)] branch trainer rc $RC"
H=$OUT/head_step$((N + B)).safetensors
[ -s $H ] || { echo "no branch head"; exit 1; }
for VS in valrbp50v val50; do
  if [ $VS = valrbp50v ]; then VF="--features $SP/features/sp_valrbp50mtp --pack $SP/jobs/valrbp50v/pack"; else VF="--features $SP/features/sp_val50mtp"; fi
  for EX in ${EXPERTS:-requant bf16}; do
    [ -z "$UNF" ] && case "${TRAIN:-all}" in noexperts|nonexpert) [ "$EX" = bf16 ] && continue; EX=served;; esac
    [ $VS = val50 ] && [ "$EX" = bf16 ] && continue                 # the BF16 check once, on valrbp50v
    TAG=${RUN}_branch${N}${SUFFIX:-}_${VS}_$EX
    TS=""; [ $VS = valrbp50v ] && [ -n "$SWEEP" ] && TS="--trunc-sweep $SWEEP"     # draft truncations from the same pass
    $PY -u mtp_eval_val50.py $VF --head $H --experts $EX --device $DEV $TS \
      --K 5 --out $SP/results/$TAG.json --save-anchors $SP/results/${TAG}_anchors.pt 2>&1 | grep -E "^prob_srv_T|^done|Traceback|Error"
    $PY live_tau.py $SP/results/${TAG}_anchors.pt --modes prob_srv_T10 prob_srv_T06 --K 3 4 5
    [ -n "$TS" ] && $PY live_tau.py $SP/results/${TAG}_anchors.pt --modes $(echo $SWEEP | tr ',' '\n' | \
      awk -F: '{printf "prob_srv_T10_k%dp%02d ", $1, $2 * 100 + 0.5}') --K 4
  done
done
if [ "${ISHARE:-1}" = 1 ] && [ -z "$UNF" ]; then          # IE 10-13: index_share_for_mtp_iteration emulated (valrbp50v)
  TAG=${RUN}_branch${N}${SUFFIX:-}_valrbp50v_ishare
  $PY -u mtp_eval_val50.py --features $SP/features/sp_valrbp50mtp --pack $SP/jobs/valrbp50v/pack --head $H --experts served \
    --device $DEV --index-share --K 5 --out $SP/results/$TAG.json --save-anchors $SP/results/${TAG}_anchors.pt 2>&1 \
    | grep -E "^prob_srv_T|^done|Traceback|Error"
  $PY live_tau.py $SP/results/${TAG}_anchors.pt --modes prob_srv_T10 prob_srv_T06 --K 3 4 5
fi
echo "[$(date -u +%T)] branch $RUN step $N done"
