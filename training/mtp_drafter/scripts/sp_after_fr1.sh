#!/bin/bash
# sp_after_fr1.sh (2026-10-14, Claude fork 5d20b23b; side pod only; user 10-14 ~22:4x: after fr1, anneal from 0.1x to 0 on the
# corpus without the redbluepill games, 2,000 steps; in the morning checkpoint averaging and the step-conditioned round on
# that corpus, judged live on the public-25 games). Runs detached:
#   1. waits for fr1's trainer to finish (status finished / stream_ended) and its launcher (attempt 3) to exit;
#   2. keeps fr1's final state as mtp_ft/fr1/state_final_step<N>.pt;
#   3. promotes dev_stepemb/ (train_mtp_sp.py with --init-from / --step-emb, mtp_eval_val50.py, step_emb.py, sp_fullrun.sh with
#      REPLAY_GPU_REPLACE; CPU-tested; previous versions kept as <file>.pre_fr1n);
#   4. launches fr1n: replay of jobs/nonrbp/sessions_fr1n_2000.json on GPU 0 (its launch replaces fr1's idle replay server),
#      trainer on GPU 1 from fr1's final state (--init-from: weights only, fresh optimizer), LR 5e-6 (0.1x of fr1's peak),
#      50 warmup steps, cosine to 0 at step 2,000, LK throughout, K 4, 4 x 16K, KEEP 250, EVAL 250, CKPT 225;
#   5. once fr1n trains: scores fr1's final head (valrbp50v, truncation sweep, index share) and the tail average
#      (11,250 / 12,000 / 12,750 / final) on GPU 1 beside fr1n's trainer (val50 is left out: 49 of its 50 sessions are in
#      fr1's training list).
# Writes mtp_ft/fr1/state_final_step*, scripts/ (step 3), mtp_ft/fr1n*, mtp_ft/fr1_final/, results/fr1_final*, results/fr1_avg4*,
# logs/. Kills nothing itself (launch_s2.sh, the production launch, stops the idle server it replaces).
# usage: bash start_detached.sh logs/after_fr1.log sp_after_fr1.sh      DRY=1 bash sp_after_fr1.sh: the plan only
set -u
[ "$(hostname)" = 6bc94744da90 ] || { echo "not the side pod"; exit 9; }
SP=/mnt/shared/arc3-inference/sidepod_20261012
PY=/workspace/arc3-inference/venv-speculators/bin/python
O=$SP/mtp_ft/fr1
LIST=$SP/jobs/nonrbp/sessions_fr1n_2000.json
cd $SP/scripts
log() { echo "[$(date -u +%T)] $*"; }
[ -s $LIST ] || { log "no session list $LIST"; exit 1; }
for f in train_mtp_sp.py mtp_eval_val50.py step_emb.py sp_fullrun.sh; do [ -s dev_stepemb/$f ] || { log "no dev_stepemb/$f"; exit 1; }; done
if [ -z "${DRY:-}" ]; then
  log "waiting for fr1 to finish"
  until grep -qE '"(finished|stream_ended)"' $O/trainer_status.json 2>/dev/null; do sleep 60; done
  log "fr1: $(cat $O/trainer_status.json)"
  for i in $(seq 1 60); do               # fr1's launcher runs two lines after its trainer; sp_fullrun.sh is replaced below
    busy=0
    for p in /proc/[0-9]*; do
      tr '\0' '\n' 2>/dev/null < $p/environ | grep -qx "MTP_RUN=fr1" || continue
      tr '\0' ' ' 2>/dev/null < $p/cmdline | grep -q "sp_fullrun.sh" && busy=1
    done
    [ $busy = 0 ] && break
    sleep 20
  done
  [ $busy = 0 ] || { log "fr1's launcher still running after 20 min; stopping"; exit 1; }
fi
read -r FSTEP FSTATE <<< "$($PY - $O <<'PYEOF'
import glob, json, sys
best = (-1, 'none')
for p in glob.glob(sys.argv[1] + '/state_*.pt'):
    try:
        s = json.load(open(p + '.json'))['step']
    except Exception:
        continue
    if s > best[0]:
        best = (s, p)
print(best[0], best[1])
PYEOF
)"
log "fr1 final state: $FSTATE (step $FSTEP)"
F=$O/state_final_step$FSTEP.pt
ENVS="RUN=fr1n LR=5e-6 TRAIN=nonexpert OPT=muon NK=4 SW=1,0.69,0.40,0.18 LOSS=lk CLOSS=lk WINDOW=16384 ACCUM=4 MINWIN=2048 TOKW=1 \
ROUTER=1.0 MOE=grouped STEPS=2000 WARMUP=50 DECAY_START=51 CKPT=225 KEEP=250 EVAL=250 POOL=8 EXPW=served TRAIN_GPUS=1 REPLAY_GPUS=0 \
SESS=$LIST SKIP_BASELINE=1 REPLAY_GPU_REPLACE=1"
if [ -n "${DRY:-}" ]; then
  log "DRY: the launch with the dev launcher (state $F would be the final one):"
  env $ENVS EXTRA="--init-from $F" DRY=1 bash dev_stepemb/sp_fullrun.sh 2>&1 | tail -48
  exit 0
fi
[ "$FSTEP" -ge 13000 ] || { log "final state at step $FSTEP, expected >= 13000; stopping"; exit 1; }
if [ ! -s $F.json ]; then
  cp $FSTATE $F.tmp && mv $F.tmp $F && cp $FSTATE.json $F.json.tmp && mv $F.json.tmp $F.json || { log "could not keep the final state"; exit 1; }
fi
for f in train_mtp_sp.py mtp_eval_val50.py step_emb.py sp_fullrun.sh; do
  [ -e $f ] && [ ! -e $f.pre_fr1n ] && cp $f $f.pre_fr1n
  cp dev_stepemb/$f $f.new && mv $f.new $f || { log "could not promote $f"; exit 1; }
done
log "promoted dev_stepemb: $(md5sum train_mtp_sp.py mtp_eval_val50.py step_emb.py sp_fullrun.sh | cut -c1-8 | tr '\n' ' ')"
env $ENVS EXTRA="--init-from $F" bash start_detached.sh $SP/logs/fullrun_fr1n_a1.log sp_fullrun.sh
log "fr1n launched (logs/fullrun_fr1n_a1.log); waiting for its first training step"
TL=/workspace/mtpfull_state_fr1n/trainer_a1.log
for i in $(seq 1 60); do grep -q '"kind"' $TL 2>/dev/null && break; sleep 60; done
log "fr1n: $(grep '"kind"' $TL 2>/dev/null | tail -1 | cut -c1-200)"
mkdir -p $SP/mtp_ft/fr1_final
H=$SP/mtp_ft/fr1_final/head_step$FSTEP.safetensors
[ -s $H ] || PYTHONDONTWRITEBYTECODE=1 $PY state_to_head.py $H $F || { log "no final head"; exit 1; }
H=$H TAG=fr1_final DEV=cuda:1 SETS=valrbp50v SWEEP=10:0.95,40:0.95,20:0.90,20:0.98 ISHARE=1 bash sp_score_head.sh
A=$SP/mtp_ft/fr1_final/head_avg4_11250_12000_12750_$FSTEP.safetensors
[ -s $A ] || PYTHONDONTWRITEBYTECODE=1 $PY state_to_head.py $A $O/state_keep_step11250.pt $O/state_keep_step12000.pt \
  $O/state_keep_step12750.pt $F || { log "no average"; exit 1; }
H=$A TAG=fr1_avg4 DEV=cuda:1 SETS=valrbp50v bash sp_score_head.sh
log "after-fr1 chain done"
