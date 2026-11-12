#!/bin/bash
# sp_fullrun.sh (2026-10-13, Claude fork 5d20b23b; side pod only): the MTP full run, launch (attempt 1) or resume (attempt K).
# User 10-13: the whole corpus v7 minus the 50 redbluepill validation games, one pass, 16K windows, recipe from the rounds
# and Inference Engineering's review; 2 GPUs as a head start (1 replay + 1 trainer, accum 3), moved to a 4-GPU pod
# (1 replay + 3 trainers, accum 1) by a graceful stop and a resume, so rows per step (48K), LR and schedule stay the same.
#   1. replay (GPU 0, sp_fullrun_replay.sh): the session list streamed as 4-stream features into /dev/shm/mtpfull_<RUN>/features;
#   2. attempt 1: the untrained baseline on valrbp50v (production emulation, live tau) on the first trainer GPU;
#   3. trainer (train_mtp_sp.py 65b8c544+): --state-local (states on the container disk first, copied to the volume),
#      rolling states every CKPT steps, milestones every KEEP steps, the pre-decay state, the stop file <out>/STOP,
#      stdout to the container disk (the health logger mirrors it to logs/);
#   4. the health logger (sp_fullrun_health.sh): one JSON line a minute.
# Control files: mtp_ft/<RUN>_ctl/run_a<K>.env (every setting a resume must reuse) and latest.env; processes carry
# MTP_RUN=<RUN> (sp_fullrun_ctl.sh finds them by it, never by a name pattern).
# usage: launch: RUN=fr1 LR=5e-5 [recipe env] bash start_detached.sh logs/fullrun_fr1_a1.log sp_fullrun.sh
#        resume: prepared by `bash sp_fullrun_ctl.sh resume RUN` (ATTEMPT, RESUME, SESS, TRAINER_ONLY, REPLAY_LAB from it)
#        recipe env (defaults): TRAIN all, OPT adamw, NK 4, SW 1,0.69,0.40,0.18, LOSS served_kl, CLOSS lk, ACCEPT "",
#        WINDOW 16384, ACCUM 4 (user 10-13: 4 windows per step, 2 per trainer after the move), MINWIN 2048, TOKW 1, ROUTER 1.0, MOE loop, CKPTSTEPS --ckpt-steps, STEPS (computed: 98% of the windows /
#        windows per step), WARMUP 5%, DECAY 0.2, DECAY_START -1, CKPT 225, KEEP 2250 (milestones every ~147M tokens), EVAL 375, POOL 8, EXTRA "",
#        TRAIN_GPUS 1 (e.g. 1,2,3: torchrun), REPLAY_GPUS 0 (0,1: two replay instances, each on half of the session list,
#        sp_fullrun_replay.sh instance mode; the 4-GPU layout REPLAY_GPUS=0,1 TRAIN_GPUS=2,3), SKIP_BASELINE; DRY=1 prints the settings (and STEPS) and launches nothing;
#        EXPW (frozen routed experts' weights: served, the production NVFP4 values, for TRAIN noexperts|nonexpert, else
#        source; user 10-13), UNFREEZE 0 (1: the escape hatch at a resume, --unfreeze-experts), EXPERT_LR 0.5, EXPERT_WARMUP 200.
#        REPLAY_GPU_REPLACE=1 (10-14): a new run's replay may start on a GPU held by a finished run's idle replay server
#        (launch_s2.sh stops it first); the trainer GPUs must still be free.
set -u
[ "$(hostname)" = 6bc94744da90 ] || { echo "not the side pod"; exit 9; }
SP=/mnt/shared/arc3-inference/sidepod_20261012
R=/workspace/arc3-inference
M=/workspace/models/Qwen3.8-Flash-Next-W4A16-intel
PY=$R/venv-speculators/bin/python
RUN=${RUN:?run}; LR=${LR:?lr}; ATTEMPT=${ATTEMPT:-1}
FEAT=/dev/shm/mtpfull_$RUN/features
OUT=$SP/mtp_ft/$RUN
CTL=$SP/mtp_ft/${RUN}_ctl
LOC=/workspace/mtpfull_state_$RUN
VFEAT=$SP/features/sp_valrbp50mtp
VPACK=$SP/jobs/valrbp50v/pack
SESS=${SESS:-$SP/jobs/valrbp50v/train_sessions.json}
LAB=mtpfull_$RUN; [ "$ATTEMPT" -gt 1 ] && LAB=mtpfull_${RUN}_r$ATTEMPT
[ -n "${TRAINER_ONLY:-}" ] && LAB=${REPLAY_LAB:?replay label of the running replay}
DONEF=$FEAT/DONE_$LAB
cd $SP/scripts
export MTP_RUN=$RUN PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date -u +%T)] $*"; }
alive() {   # processes of this run (by MTP_RUN in their environment), excluding this shell and its parents
  for p in /proc/[0-9]*; do
    pid=${p#/proc/}
    [ "$pid" = $$ ] || [ "$pid" = $PPID ] && continue
    tr '\0' '\n' 2>/dev/null < $p/environ | grep -qx "MTP_RUN=$RUN" && echo "$pid $(tr '\0' ' ' 2>/dev/null < $p/cmdline | cut -c1-100)"
  done
}
TRAIN_GPUS=${TRAIN_GPUS:-1}
IFS=',' read -ra TG <<< "$TRAIN_GPUS"
REPLAY_GPUS=${REPLAY_GPUS:-0}                # one replay on GPU 0 (production launch), or 0,1: two instances (user 10-13)
IFS=',' read -ra RG <<< "$REPLAY_GPUS"
[ ${#RG[@]} -gt 1 ] || [ "$REPLAY_GPUS" = 0 ] || { log "a single replay runs on GPU 0 (lc_run3.sh)"; exit 1; }
GPUS="${TG[*]}"; [ -z "${TRAINER_ONLY:-}" ] && GPUS="${RG[*]} $GPUS"
for g in $GPUS; do
  [ -n "${DRY:-}" ] && break                               # DRY=1: print the plan, launch nothing
  if [ -n "${REPLAY_GPU_REPLACE:-}" ] && [ -z "${TRAINER_ONLY:-}" ] && [[ " ${RG[*]} " == *" $g "* ]]; then
    log "GPU $g: in use by the previous run's idle replay server ($(nvidia-smi --query-gpu=memory.used --format=csv,noheader -i $g)); the replay launch (launch_s2.sh) replaces it (REPLAY_GPU_REPLACE, user 10-14)"
    continue
  fi
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  [ "$m" -lt 2000 ] || { log "GPU $g busy ($m MiB)"; exit 1; }
done
[ -d $VFEAT ] && [ -s $VPACK/rows.jsonl ] || { log "no valrbp50 export / pack"; exit 1; }
if [ "$ATTEMPT" = 1 ]; then
  [ -e $OUT ] && { log "exists: $OUT"; exit 1; }
  [ -e /dev/shm/mtpfull_$RUN ] && { log "exists: /dev/shm/mtpfull_$RUN"; exit 1; }
  [ -e $CTL ] && { log "exists: $CTL"; exit 1; }
  RES=""
else
  RESUME=${RESUME:?state to resume from}
  [ -s $RESUME ] && [ -s $RESUME.json ] || { log "no state or sidecar: $RESUME"; exit 1; }
  RES="--resume $RESUME"
  [ -e $CTL/run_a$ATTEMPT.env ] && { log "exists: $CTL/run_a$ATTEMPT.env"; exit 1; }
  A=$(alive | grep -v "sp_fullrun.sh" || true)
  if [ -n "${TRAINER_ONLY:-}" ]; then
    echo "$A" | grep -q "train_mtp_sp" && { log "a trainer of $RUN is alive:"; echo "$A"; exit 1; }
  else
    [ -n "$A" ] && { log "processes of $RUN are alive:"; echo "$A"; exit 1; }
  fi
fi
[ -z "${TRAINER_ONLY:-}${DRY:-}" ] && pgrep -f "[v]llm.entrypoints.cli.main serve" > /dev/null && { log "a server is running"; exit 1; }
NR=${#TG[@]}
WINDOW=${WINDOW:-16384}; ACCUM=${ACCUM:-4}; NK=${NK:-4}
MINWIN=${MINWIN:-2048}; TOKW=${TOKW:-1}          # user 10-13: no segment-tail stubs; every trained position weighs the same
TW=""; [ "$TOKW" = 1 ] && TW="--token-weighted"
case "${TRAIN:-all}" in noexperts|nonexpert) EXPW=${EXPW:-served};; *) EXPW=${EXPW:-source};; esac
UNFREEZE=${UNFREEZE:-0}; EXPERT_LR=${EXPERT_LR:-0.5}; EXPERT_WARMUP=${EXPERT_WARMUP:-200}
UNF=""; [ "$UNFREEZE" = 1 ] && UNF="--unfreeze-experts --expert-lr-scale $EXPERT_LR --expert-warmup $EXPERT_WARMUP"
if [ -z "${STEPS:-}" ]; then      # 98% of the stream's windows / windows per step (2% for failed sessions and rank imbalance)
  STEPS=$($PY - $SESS $R/dflash_streams_20261011/sessions.jsonl $WINDOW $NK $((ACCUM * NR)) $MINWIN <<'PYEOF'
import json, math, sys
sys.path.insert(0, '.')
from train_mtp_sp import window_bounds
want = {s['session_id'] for s in json.load(open(sys.argv[1]))}
W, K, per, mw = int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6])
nw = 0
for line in open(sys.argv[2]):
    d = json.loads(line)
    if d['session_id'] in want:
        n = int(d.get('n') or 0)
        for s0 in range(0, n, 65536):
            nw += len(window_bounds(min(65536, n - s0), W, K, mw))
print(int(0.98 * nw / per))
PYEOF
)
fi
WARMUP=${WARMUP:-$((STEPS / 20))}
if [ -n "${DRY:-}" ]; then CTL=$(mktemp -d); LOC=$CTL/loc; fi
mkdir -p $CTL $LOC
cat > $CTL/run_a$ATTEMPT.env <<EOF
RUN=$RUN
ATTEMPT=$ATTEMPT
LR=$LR
TRAIN=${TRAIN:-all}
OPT=${OPT:-adamw}
NK=$NK
SW=${SW:-1,0.69,0.40,0.18}
LOSS=${LOSS:-served_kl}
CLOSS=${CLOSS:-lk}
ACCEPT="${ACCEPT:-}"
WINDOW=$WINDOW
ACCUM=$ACCUM
NRANKS=$NR
TRAIN_GPUS=$TRAIN_GPUS
REPLAY_GPUS=$REPLAY_GPUS
ROUTER=${ROUTER:-1.0}
MOE=${MOE:-loop}
CKPTSTEPS=${CKPTSTEPS---ckpt-steps}
STEPS=$STEPS
WARMUP=$WARMUP
DECAY=${DECAY:-0.2}
DECAY_START=${DECAY_START:--1}
CKPT=${CKPT:-225}
KEEP=${KEEP:-2250}
EVAL=${EVAL:-375}
POOL=${POOL:-8}
EXTRA="${EXTRA:-}"
EXPW=$EXPW
MINWIN=$MINWIN
TOKW=$TOKW
UNFREEZE=$UNFREEZE
EXPERT_LR=$EXPERT_LR
EXPERT_WARMUP=$EXPERT_WARMUP
SESS=$SESS
REPLAY_LAB=$LAB
RESUME=${RESUME:-}
TRAINER_ONLY=${TRAINER_ONLY:-}
STARTED=$(date -u +%Y-%m-%dT%H:%M:%SZ)
EOF
cp $CTL/run_a$ATTEMPT.env $CTL/latest.env.tmp && mv $CTL/latest.env.tmp $CTL/latest.env
log "full run $RUN attempt $ATTEMPT: lr $LR train ${TRAIN:-all} opt ${OPT:-adamw} K $NK weights ${SW:-1,0.69,0.40,0.18} loss ${LOSS:-served_kl}/${CLOSS:-lk} rows ${WINDOW}x${ACCUM}x$NR steps $STEPS (warmup $WARMUP, decay ${DECAY:-0.2}) moe ${MOE:-loop} min-window $MINWIN token-weighted $TOKW experts $EXPW${UNF:+ (unfrozen, $EXPERT_LR x after $EXPERT_WARMUP)} ckpt ${CKPT:-225} keep ${KEEP:-2250}; sessions $SESS; replay label $LAB${TRAINER_ONLY:+ (running replay)}; ${RES:-fresh}"
NRG=${#RG[@]}
if [ -z "${TRAINER_ONLY:-}" ] && [ $NRG -gt 1 ]; then   # two-replay layout: every NRG-th session of the stratified order
  $PY - $SESS $CTL/sessions_a$ATTEMPT $NRG <<'PYEOF'
import json, sys
s, n = json.load(open(sys.argv[1])), int(sys.argv[3])
for i in range(n):
    json.dump(s[i::n], open('%s_i%d.json' % (sys.argv[2], i), 'w'))
    print('replay instance %d: %d sessions' % (i, len(s[i::n])))
PYEOF
fi
if [ -n "${DRY:-}" ]; then log "DRY: settings in $CTL/run_a$ATTEMPT.env:"; cat $CTL/run_a$ATTEMPT.env; exit 0; fi
if [ -z "${TRAINER_ONLY:-}" ]; then
  AT=$([ "$ATTEMPT" -gt 1 ] && echo $ATTEMPT)
  if [ $NRG = 1 ]; then
    RUN=$RUN SESS=$SESS ATTEMPT=$AT bash start_detached.sh $SP/logs/fullrun_replay_$LAB.log sp_fullrun_replay.sh
    log "replay side started (log logs/fullrun_replay_$LAB.log)"
  else
    for i in "${!RG[@]}"; do
      RUN=$RUN SESS=$CTL/sessions_a${ATTEMPT}_i$i.json ATTEMPT=$AT INST=$i NINST=$NRG GPU=${RG[$i]} PORT=$((1300 + ${RG[$i]})) \
        bash start_detached.sh $SP/logs/fullrun_replay_${LAB}_i$i.log sp_fullrun_replay.sh
      log "replay instance $i started on GPU ${RG[$i]}, port $((1300 + ${RG[$i]})) (log logs/fullrun_replay_${LAB}_i$i.log)"
    done
  fi
fi
RUN=$RUN ATTEMPT=$ATTEMPT bash start_detached.sh $SP/logs/fullrun_health_${RUN}_a$ATTEMPT.log sp_fullrun_health.sh
if [ "$ATTEMPT" = 1 ] && [ -z "${SKIP_BASELINE:-}" ]; then
  log "baseline (untrained head) on valrbp50v, GPU ${TG[0]}"
  $PY -u mtp_eval_val50.py --features $VFEAT --pack $VPACK --experts served --device cuda:${TG[0]} --K 5 \
    --out $SP/results/fullrun_${RUN}_baseline_valrbp50.json --save-anchors $SP/results/fullrun_${RUN}_baseline_valrbp50_anchors.pt 2>&1 \
    | grep -E "^(prob_srv|greedy_srv)_T|^done|Traceback|Error"
  $PY live_tau.py $SP/results/fullrun_${RUN}_baseline_valrbp50_anchors.pt --modes prob_srv_T10 prob_srv_T06 --K 3 4 5
fi
if [ $NR -gt 1 ]; then
  LAUNCH="$R/venv-speculators/bin/torchrun --standalone --nproc_per_node $NR train_mtp_sp.py --devices $TRAIN_GPUS"
else
  LAUNCH="$PY -u train_mtp_sp.py --device cuda:${TG[0]}"
fi
TLOG=$LOC/trainer_a$ATTEMPT.log
log "trainer (GPUs $TRAIN_GPUS), log $TLOG (mirrored to logs/mtp_ft_${RUN}_trainer_a$ATTEMPT.log)"
MTP_MOE=${MOE:-loop} $LAUNCH --data $VFEAT --val-files 50 --val-windows 24 --streams $R/dflash_streams_20261011 \
  --mtp-file $M/model_extra_tensors.safetensors --target-dir $M --out $OUT --state-local $LOC \
  --stream-dir $FEAT --stream-done $DONEF --stream-pool ${POOL:-8} \
  --steps $STEPS --lr $LR --warmup $WARMUP --schedule wsd --decay-ratio ${DECAY:-0.2} --decay-start ${DECAY_START:--1} \
  --num-steps $NK --step-weights ${SW:-1,0.69,0.40,0.18} ${ACCEPT:-} --loss ${LOSS:-served_kl} --cooldown-loss ${CLOSS:-lk} \
  --window $WINDOW --accum $ACCUM --min-window $MINWIN $TW --train ${TRAIN:-all} --expert-weights $EXPW $UNF \
  --router-lr-scale ${ROUTER:-1.0} \
  --optimizer ${OPT:-adamw} --eval-every ${EVAL:-375} --save-every 0 --ckpt-every ${CKPT:-225} --keep-every ${KEEP:-2250} \
  --attention flex --compile-flex --fused-adam --prefetch 2 --pos-key position_ids ${CKPTSTEPS---ckpt-steps} $RES ${EXTRA:-} \
  > $TLOG 2>&1
RC=$?
log "trainer rc $RC; status $(cat $OUT/trainer_status.json 2>/dev/null)"
