#!/bin/bash
# early_check.sh (2026-10-22, MTP refit; Claude session 099768dc): the user's early check for the refit's loss temperature 0.6 (approved
# with the unattended launch, 10-22): the trainer's validation tau_srv06 (val_tau_srv06 in OUT/log.jsonl) at steps 750 and 1,125 against
# the step-0 eval of the starting head (fr1's weights). If it is more than 1% below the start at BOTH steps, the run is stopped gracefully
# (sp_fullrun_ctl.sh stop RUN: the trainer saves at the end of its step and exits) and the user decides (revert to 1.0 or continue);
# otherwise PASS. Reads only log.jsonl (>= conditions on the step; the last record per step wins: resume duplicates). OVERDUE line if
# step 1,125's eval is missing 6 h after the start. Log: SP/logs/early_check_<RUN>.log
# usage: RUN=r1 setsid nohup bash early_check.sh > SP/logs/early_check_r1.log 2>&1 < /dev/null &
set -u
SP=/mnt/shared/arc3-inference/mtp_refit_20261022
RUN=${RUN:?run}
L=$SP/mtp_ft/$RUN/log.jsonl
log() { echo "[$(date -u +%T)] $*"; }
DL=$(( $(date +%s) + 6 * 3600 ))
val() {   # val STEP -> the last val_tau_srv06 logged at that step, or empty
  python3 - "$L" "$1" <<'PYEOF'
import json, sys
v = None
try:
    for line in open(sys.argv[1]):
        if '"val_tau_srv06"' not in line:
            continue
        d = json.loads(line)
        if int(d.get('step', -1)) == int(sys.argv[2]):
            v = d['val_tau_srv06']
except FileNotFoundError:
    pass
print('' if v is None else repr(float(v)))
PYEOF
}
log "early check for $RUN: tau_srv06 at steps 750 and 1,125 vs step 0 ($L); OVERDUE at $(date -u -d @$DL +%T)"
while :; do
  T0=$(val 0); T750=$(val 750); T1125=$(val 1125)
  [ -n "$T0" ] && [ -n "$T750" ] && [ -n "$T1125" ] && break
  [ "$(date +%s)" -ge "$DL" ] && { log "OVERDUE: evals step 0 '$T0', 750 '$T750', 1125 '$T1125' after 6 h"; exit 1; }
  sleep 60
done
VERDICT=$(python3 -c "
t0, a, b = float('$T0'), float('$T750'), float('$T1125')
ra, rb = a / t0 - 1, b / t0 - 1
print('STOP' if ra < -0.01 and rb < -0.01 else 'PASS', '%+.2f%% at 750, %+.2f%% at 1,125 (start %.4f, 750 %.4f, 1,125 %.4f)' % (100 * ra, 100 * rb, t0, a, b))")
log "tau_srv06: $VERDICT"
case "$VERDICT" in
  STOP*) log "EARLY CHECK STOP: more than 1% below the start at both steps; graceful stop (sp_fullrun_ctl.sh stop $RUN); the user decides"
         cd $SP/scripts && bash sp_fullrun_ctl.sh stop $RUN ;;
  *)     log "EARLY CHECK PASS" ;;
esac
