#!/bin/bash
# sp_after_stop.sh (2026-10-14, Claude fork 5d20b23b; side pod only; user 10-14 ~23:5x: stop fr1 at 10k, hand the pod over): waits
# for fr1's graceful stop (trainer status "stopped" at step >= N), keeps that state as mtp_ft/fr1/state_stop_step<N>.pt, makes its
# head (state_to_head.py) and scores it once on the CURRENT numerics (valrbp50v, served experts, live tau K 3-5) on GPU 1: the
# old-stack side of the tau-loss measurement for the fixed-stack refit. Writes only those files, results/fr1_stop<N>_*, logs/.
# usage: N=10000 bash start_detached.sh logs/after_stop_fr1.log sp_after_stop.sh
set -u
[ "$(hostname)" = 6bc94744da90 ] || { echo "not the side pod"; exit 9; }
SP=/mnt/shared/arc3-inference/sidepod_20261012
PY=/workspace/arc3-inference/venv-speculators/bin/python
O=$SP/mtp_ft/fr1
N=${N:?step}
cd $SP/scripts
log() { echo "[$(date -u +%T)] $*"; }
log "waiting for fr1's stop at step >= $N"
until $PY -c "import json, sys; s = json.load(open('$O/trainer_status.json')); sys.exit(0 if s.get('status') == 'stopped' and s.get('step', 0) >= $N else 1)" 2>/dev/null; do sleep 20; done
ST=$($PY -c "import json; print(json.load(open('$O/trainer_status.json'))['state'])")
STEP=$($PY -c "import json; print(json.load(open('$ST.json'))['step'])")
log "fr1 stopped: $(cat $O/trainer_status.json); state $ST (step $STEP)"
F=$O/state_stop_step$STEP.pt
if [ ! -s $F.json ]; then
  cp $ST $F.tmp && mv $F.tmp $F && cp $ST.json $F.json.tmp && mv $F.json.tmp $F.json || { log "could not keep the state"; exit 1; }
fi
mkdir -p $SP/mtp_ft/fr1_final
H=$SP/mtp_ft/fr1_final/head_step$STEP.safetensors
[ -s $H ] || PYTHONDONTWRITEBYTECODE=1 $PY state_to_head.py $H $F || { log "no head"; exit 1; }
H=$H TAG=fr1_stop$STEP DEV=cuda:1 SETS=valrbp50v bash sp_score_head.sh
RC=$?; log "after-stop scoring finished (rc $RC)"
