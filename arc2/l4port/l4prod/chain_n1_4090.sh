#!/usr/bin/env bash
# chain n1, 4090 (2026-10-31, ARC2 - main; the user approved the revised plan: REVIEW_PACKET_2026-10-31.md section 7). After try
# l4p38 (the tolerant snapshot merge): its pass line; l4p39 = the 8-task smoke with ARC3_STAGE_VERIFY=1 (byte + allocator checks of
# every compaction on every worker; evpp v3, draftring_pp v2, bootstrap v5 = drafter head v2, split 26/22) and its pre-stated pass line
# (check_l4prod.py --verify: harness exit 0 + score; no PersistentStateError / EngineDeadError / unrouted callable; the queue-cap and
# policy receipts; >= 1 eviction, all merged receipts synchronized, submits == commits; no failed frequency snapshot; every eviction
# byte- and allocator-verified; 0 preemptions). PASS -> 'smoke gate PASS' (the L4 chain waits for it), the twin's PP side
# (twin_pp.sh: tf_score twice with SR off), then l4p40 = one pass over the 120 evaluation tasks with verification (function and
# persistent-traffic timeline, not a quality read; pass line with >= 50 evictions). Ends 'chain n1 4090 ended'.
set -uo pipefail
D=/mnt/shared/arc2/l4port; P4=$D/l4prod; LD=$D/logs; L=$LD/chain_n1_4090.log; S2=/workspace/kvring_dev/fin74/s2
log() { echo "[$(date -u +'%F %T')] $*" >> $L; }
stop() { log "STOP: $*"; log "chain n1 4090 ended"; exit 1; }
check() { python3 $P4/check_l4prod.py $S2/instances/$1 "$(cat $S2/instances/$1/last_logdir 2>/dev/null)/server.log" $LD/try_l4prod_$1.log "${@:2}" 2>&1; }
log "start (pid $$); try_4090 $(md5sum < $P4/try_l4prod_4090.sh | cut -c1-8), check $(md5sum < $P4/check_l4prod.py | cut -c1-8), twin $(md5sum < $P4/twin_pp.sh | cut -c1-8), evpp $(md5sum < $P4/l4port_evpp.py | cut -c1-8), draftring_pp $(md5sum < $P4/l4port_draftring_pp.py | cut -c1-8), bootstrap $(md5sum < $D/bootstrap_l4prod/sitecustomize.py | cut -c1-8)"
DL=$(( $(date +%s) + 3 * 3600 ))
until grep -q -E '^\[[0-9: -]+\] try l4p38 ended$' $LD/try_l4prod_l4p38.log 2>/dev/null; do
  [ "$(date +%s)" -ge "$DL" ] && stop "OVERDUE waiting for l4p38"
  sleep 30
done
log "l4p38: $(check l4p38 | tr '\n' ' ' | cut -c1-1200)"
log "l4p38 TP-mismatch receipts: $(grep -h -o '\[l4port\] evpp snapshot TP mismatch [0-9]*: {[^}]*}' "$(cat $S2/instances/l4p38/last_logdir)/server.log" 2>/dev/null | head -n 3 | tr '\n' ' ' | cut -c1-900)"
sleep 20
bash $P4/try_l4prod_4090.sh l4p39 smoke_l4p39 smoke --drop-parts ctxfree --set ARC3_STAGE_VERIFY=1 > $LD/try_l4p39.out 2>&1
R=$(check l4p39 --verify)
log "l4p39 pass line: $(echo "$R" | tr '\n' ' ' | cut -c1-1500)"
echo "$R" | head -n 1 | grep -q '^PASS' || stop "l4p39 failed its pass line"
log "smoke gate PASS"
bash $P4/twin_pp.sh twin39 --drop-parts ctxfree > $LD/twin39.out 2>&1
log "twin39: $(grep -E 'server wrapper|tf_score|PP rerun|STOP' $LD/twin_twin39.log | tr '\n' ' ' | cut -c1-1200)"
bash $P4/try_l4prod_4090.sh l4p40 gp4090_l4p40 pass --drop-parts ctxfree --set ARC3_STAGE_VERIFY=1 > $LD/try_l4p40.out 2>&1
R=$(check l4p40 --verify --min-evicts 50)
log "l4p40 pass line: $(echo "$R" | tr '\n' ' ' | cut -c1-1500)"
log "l4p40 harness: $(grep -E 'gp4090_l4p40 (exit|tail)' $LD/try_l4prod_l4p40.log | tr '\n' ' ' | cut -c1-600)"
log "chain n1 4090 ended"
