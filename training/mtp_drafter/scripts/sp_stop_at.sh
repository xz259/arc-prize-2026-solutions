#!/bin/bash
# sp_stop_at.sh (2026-10-14, Claude fork 5d20b23b; side pod only; fixed 2026-10-15): a graceful stop of RUN's trainer right after
# step N, for a trainer started without --stop-at (new runs: pass --stop-at N to train_mtp_sp.py instead, in-process).
# 10-15 fix: the first version grepped the trainer's stdout log for step N-1, which prints every 10th step only, so it never
# fired. Now: the per-step log mtp_ft/<RUN>/log.jsonl (every step), an "at or past" condition (step >= N-1), and a deadline:
# if step N-1 is not reached by the expected time (measured step time) + 15 min, it prints OVERDUE and exits 2.
# Then `sp_fullrun_ctl.sh stop RUN` (the STOP file: the trainer finishes the step in flight, saves a full state, exits).
# CHECK=1: print the current step and the deadline, request nothing. usage: RUN=fr1 N=10000 bash start_detached.sh LOG sp_stop_at.sh
set -u
[ "$(hostname)" = 6bc94744da90 ] || { echo "not the side pod"; exit 9; }
SP=/mnt/shared/arc3-inference/sidepod_20261012
RUN=${RUN:?run}; N=${N:?step}
LJ=$SP/mtp_ft/$RUN/log.jsonl
cur() { tail -c 400000 $LJ 2>/dev/null | grep -oE '^\{"step": [0-9]+, "loss"' | tail -1 | grep -oE '[0-9]+'; }
sps() { tail -c 400000 $LJ 2>/dev/null | grep -oE '"step_s": [0-9.]+' | tail -20 | awk '{s += $2; n++} END {print (n ? int(s / n + 1) : 12)}'; }
S0=$(cur); [ -n "$S0" ] || { echo "no per-step records in $LJ"; exit 1; }
[ "$S0" -lt "$N" ] || { echo "already at step $S0 >= $N"; exit 1; }
DL=$(( $(date -u +%s) + (N - 1 - S0) * $(sps) + 900 ))
echo "[$(date -u +%T)] step $S0, waiting for step >= $((N - 1)) in $LJ; deadline $(date -u -d @$DL +%T) (step time ~$(sps) s)"
[ -n "${CHECK:-}" ] && exit 0
until S=$(cur); [ "${S:-0}" -ge $((N - 1)) ]; do
  [ $(date -u +%s) -gt $DL ] && { echo "[$(date -u +%T)] OVERDUE: step ${S:-?} < $((N - 1)) at the deadline"; exit 2; }
  sleep 2
done
[ "$S" -eq $((N - 1)) ] && sleep 3                 # past the stop check of step N-1: the stop lands after step N
echo "[$(date -u +%T)] step $S logged; requesting the stop"
cd $SP/scripts && bash sp_fullrun_ctl.sh stop $RUN 30
