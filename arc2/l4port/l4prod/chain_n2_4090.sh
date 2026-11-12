#!/usr/bin/env bash
# chain n2, 4090 (2026-10-31, ARC2 - main): replaces chain_n1_4090.sh (its PID stopped while l4p39 ran, kills.log; l4p39 continues) to
# put the reviewer's TP-identity check before the long run. Steps: wait for try l4p39; its pass line (check_l4prod.py --verify); the
# gate line 'smoke gate PASS' (or 'chain n1 4090 ended' on a failure) appended to chain_n1_4090.log, which the L4 chain waits on;
# tpident_pp.sh tpid1 (pre-stated: IDENTICAL -> continue; DIFFER or no records -> stop before any long run); twin_pp.sh twin39 (the
# PP side); try l4p40 = one pass over the 120 evaluation tasks with ARC3_STAGE_VERIFY=1 and its pass line (>= 50 evictions).
# Ends 'chain n2 4090 ended'.
set -uo pipefail
D=/mnt/shared/arc2/l4port; P4=$D/l4prod; LD=$D/logs; L=$LD/chain_n2_4090.log; N1=$LD/chain_n1_4090.log; S2=/workspace/kvring_dev/fin74/s2
log() { echo "[$(date -u +'%F %T')] $*" >> $L; }
gate() { echo "[$(date -u +'%F %T')] $1" >> $N1; }
stop() { log "STOP: $*"; log "chain n2 4090 ended"; exit 1; }
check() { python3 $P4/check_l4prod.py $S2/instances/$1 "$(cat $S2/instances/$1/last_logdir 2>/dev/null)/server.log" $LD/try_l4prod_$1.log "${@:2}" 2>&1; }
log "start (pid $$); tpident $(md5sum < $P4/l4port_tpident.py | cut -c1-8), tpident_pp $(md5sum < $P4/tpident_pp.sh | cut -c1-8), compare $(md5sum < $P4/compare_tpident.py | cut -c1-8), bootstrap $(md5sum < $D/bootstrap_l4prod/sitecustomize.py | cut -c1-8), check $(md5sum < $P4/check_l4prod.py | cut -c1-8)"
DL=$(( $(date +%s) + 3 * 3600 ))
until grep -q -E '^\[[0-9: -]+\] try l4p39 ended$' $LD/try_l4prod_l4p39.log 2>/dev/null; do
  [ "$(date +%s)" -ge "$DL" ] && { gate "chain n1 4090 ended"; stop "OVERDUE waiting for l4p39"; }
  sleep 30
done
R=$(check l4p39 --verify)
log "l4p39 pass line: $(echo "$R" | tr '\n' ' ' | cut -c1-1500)"
if ! echo "$R" | head -n 1 | grep -q '^PASS'; then gate "chain n1 4090 ended"; stop "l4p39 failed its pass line"; fi
gate "smoke gate PASS"
log "smoke gate PASS written to chain_n1_4090.log"
sleep 20
bash $P4/tpident_pp.sh tpid1 --drop-parts ctxfree > $LD/tpid1.out 2>&1
ID=$(grep -h -o -E 'identity: [A-Z ]+' $LD/tpident_tpid1.log | tail -n 1)
log "tpid1: $(grep -E 'identity:|records:|bench rc|server wrapper' $LD/tpident_tpid1.log | cut -c22- | tr '\n' '|' | cut -c1-1500)"
echo "$ID" | grep -q 'IDENTICAL' || stop "TP identity not shown ($ID): no long run until the non-identical path is found"
sleep 20
bash $P4/twin_pp.sh twin39 --drop-parts ctxfree > $LD/twin39.out 2>&1
log "twin39: $(grep -E 'server wrapper|tf_score|PP rerun|STOP' $LD/twin_twin39.log | cut -c22- | tr '\n' '|' | cut -c1-1200)"
sleep 20
bash $P4/try_l4prod_4090.sh l4p40 gp4090_l4p40 pass --drop-parts ctxfree --set ARC3_STAGE_VERIFY=1 > $LD/try_l4p40.out 2>&1
R=$(check l4p40 --verify --min-evicts 50)
log "l4p40 pass line: $(echo "$R" | tr '\n' ' ' | cut -c1-1500)"
log "l4p40 harness: $(grep -E 'gp4090_l4p40 (exit|tail)' $LD/try_l4prod_l4p40.log | cut -c22- | tr '\n' ' ' | cut -c1-600)"
log "chain n2 4090 ended"
