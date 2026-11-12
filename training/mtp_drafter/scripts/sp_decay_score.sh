#!/bin/bash
# sp_decay_score.sh (2026-10-14, Claude fork 5d20b23b; side pod only): an in-decay score of the full run (user plan 10-14:
# the decay runs to the end; raw heads scored where the LR is below half of peak, stopping stays with the user). Waits for
# state_keep_step<N>.pt and its sidecar, makes its head with state_to_head.py (mtp_ft/<RUN>_decay_heads/), and scores it
# with sp_score_head.sh on DEV beside the trainer (default valrbp50v, live tau K 3-5). Writes only those two places.
# usage: RUN=fr1 N=9000 [DEV=cuda:1] [SETS=valrbp50v] bash start_detached.sh logs/decay_score_fr1_9000.log sp_decay_score.sh
set -u
[ "$(hostname)" = 6bc94744da90 ] || { echo "not the side pod"; exit 9; }
SP=/mnt/shared/arc3-inference/sidepod_20261012
PY=/workspace/arc3-inference/venv-speculators/bin/python
RUN=${RUN:?run}; N=${N:?step}; DEV=${DEV:-cuda:1}
STATE=$SP/mtp_ft/$RUN/state_keep_step$N.pt
echo "[$(date -u +%T)] waiting for $STATE"
until [ -s $STATE.json ] && [ -s $STATE ]; do sleep 60; done
sleep 30                                             # the sidecar follows the state; let both settle
cd $SP/scripts
mkdir -p $SP/mtp_ft/${RUN}_decay_heads
H=$SP/mtp_ft/${RUN}_decay_heads/head_step$N.safetensors
PYTHONDONTWRITEBYTECODE=1 $PY state_to_head.py $H $STATE || { echo "[$(date -u +%T)] no head (rc $?)"; exit 1; }
H=$H TAG=${RUN}_decay$N DEV=$DEV SETS=${SETS:-valrbp50v} bash sp_score_head.sh
RC=$?; echo "[$(date -u +%T)] decay score $N finished (rc $RC)"
