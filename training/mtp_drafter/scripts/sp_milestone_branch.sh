#!/bin/bash
# sp_milestone_branch.sh (2026-10-13, Claude fork 5d20b23b; side pod only): waits for the full run's milestone state
# (state_keep_step<N>.pt and its sidecar on the volume), then runs sp_branch_cooldown.sh for it with the run's own recipe
# (from mtp_ft/<RUN>_ctl/latest.env; rows per step = ACCUM x NRANKS on the branch's one GPU), on DEV (default the trainer's
# first GPU, shared with the trainer: ~37 + ~37 GB of 96). The branch writes only mtp_ft/<RUN>_branch<N> and results/.
# usage: RUN=fr1 N=2250 [DEV=cuda:1] bash start_detached.sh logs/milestone_fr1_2250.log sp_milestone_branch.sh
set -u
[ "$(hostname)" = 6bc94744da90 ] || { echo "not the side pod"; exit 9; }
SP=/mnt/shared/arc3-inference/sidepod_20261012
RUN=${RUN:?run}; N=${N:?milestone step}
STATE=$SP/mtp_ft/$RUN/state_keep_step$N.pt
echo "[$(date -u +%T)] waiting for $STATE"
until [ -s $STATE.json ] && [ -s $STATE ]; do sleep 60; done
. $SP/mtp_ft/${RUN}_ctl/latest.env
DEV=${DEV:-cuda:${TRAIN_GPUS%%,*}}
echo "[$(date -u +%T)] milestone $N of $RUN: branch on $DEV (rows per step $((ACCUM * NRANKS)) x $WINDOW)"
cd $SP/scripts
RUN=$RUN N=$N DEV=$DEV LR=$LR TRAIN=$TRAIN OPT=$OPT NK=$NK SW=$SW LOSS=$LOSS CLOSS=$CLOSS ACCEPT="$ACCEPT" WINDOW=$WINDOW \
  ACCUM=$((ACCUM * NRANKS)) ROUTER=$ROUTER MOE=$MOE CKPTSTEPS="$CKPTSTEPS" EXPW=${EXPW:-} UNFREEZE=${UNFREEZE:-0} \
  EXPERT_LR=${EXPERT_LR:-0.5} EXPERT_WARMUP=${EXPERT_WARMUP:-200} MINWIN=${MINWIN:-2048} TOKW=${TOKW:-1} \
  SUFFIX=${SUFFIX:-} bash sp_branch_cooldown.sh
RC=$?; echo "[$(date -u +%T)] milestone $N branch finished (rc $RC)"
