#!/usr/bin/env bash
# chain b2, the second 4x L4 pod (2026-10-31 15:5x, ARC2 - main): the long real-traffic validation of the adopted configuration,
# after chain b1 ends on this pod. l4v2a = the Kaggle driver (kaggle_run_arc2.py b590ae16: the next package's harness flags,
# thresholds 80K / 80K, budget 200K, max passes 4) in competition mode (the submission's server environment: diagnostics off)
# plus the allocator retry observer (ARC3_S2_MEMDIAG_RETRY=1, polling counters only), over the 120 evaluation tasks for 3 h:
#   sizing  26/22 + explicit KV 4.33 GiB (29 lanes, pbx26), the microbatch balancer (l4v1c), batched tokens 2048, the train120 +
#           eval120 expert set, util 0.89, MAX_SEQS 34, the l4c12 capture list;
#   gates   0 allocator retries (receipts), 0 preemptions, no engine death, the per-GPU peaks (nvidia-smi every 10 s), the
#           submission written (120 tasks); the score from the driver's submission against the evaluation solutions.
# Ends 'chain b2 l4 ended'.
set -uo pipefail
D=/mnt/shared/arc2/l4port; P4=$D/l4prod; LD=$D/logs_l4; L=$LD/chain_b2_l4.log; S2=/workspace/kvring_dev/fin74/s2
K=/mnt/shared/arc2/kaggle; KR=$K/rehearsal; OD=/mnt/shared/arc2/official/data
NS=$D/sidepod/prune_sets_hope/eprune_freq_k128.json
log() { echo "[$(date -u +'%F %T')] $*" >> $L; }
stop() { log "STOP: $*"; log "chain b2 l4 ended"; exit 1; }
log "start pod $(hostname) boot $(cat /proc/sys/kernel/random/boot_id) (pid $$); driver $(md5sum < $K/kaggle_run_arc2.py | cut -c1-8), ppbalance $(md5sum < $P4/l4port_ppbalance.py | cut -c1-8), set $(md5sum < $NS | cut -c1-8)"
[ "$(md5sum < $K/kaggle_run_arc2.py | cut -c1-8)" = b590ae16 ] || stop "the driver is not b590ae16"
[ "$(md5sum < $P4/l4port_ppbalance.py | cut -c1-8)" = 54f7802a ] || stop "l4port_ppbalance.py is not v2 54f7802a"
DL=$(( $(date +%s) + 2 * 3600 ))
until grep -q -E '^\[[0-9: -]+\] chain b1 l4 ended$' $LD/chain_b1_l4.log 2>/dev/null; do
  [ "$(date +%s)" -ge "$DL" ] && stop "OVERDUE waiting for chain b1's end"
  sleep 30
done
n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); [ "$n" = 0 ] || stop "$n GPU processes after chain b1"
log "chain b1 ended; starting"
sleep 20
LB=l4v2a; SMI=$LD/smi_$LB.csv; SMIS=$LD/smi_$LB.stop
SETS="{\"ARC3_STAGE4_UTIL\": \"0.89\", \"ARC3_STAGE4_MAX_SEQS\": \"34\", \"ARC3_S2_CAPTURE\": \"1,2,4,5,8,10,16,20,34,40,80,170\", \"ARC3_L4PORT_EVPP\": \"1\", \"ARC3_STAGE2_BATCHED_TOKENS\": \"2048\", \"ARC3_S2_EPRUNE_FILE\": \"$NS\", \"ARC3_L4PORT_PP_SPLIT\": \"26,22\", \"ARC3_STAGE4_KV_GIB\": \"4.33\", \"ARC3_L4PORT_PP_BALANCE\": \"1\"}"
python3 - "$SETS" > $KR/$LB.json <<PY
import json, sys, time
print(json.dumps({"label": "$LB", "notebook_start": time.time(), "budget_s": 10800, "teardown_margin_s": 150, "server_wait_s": 2400,
                  "competition": True, "extra_server_env": {"ARC3_S2_MEMDIAG_RETRY": "1"}, "l4_sets": json.loads(sys.argv[1]),
                  "challenges": "$OD/arc-agi_evaluation_challenges.json", "solutions": "$OD/arc-agi_evaluation_solutions.json",
                  "submission_path": "$KR/${LB}_submission.json", "runs_dir": "/mnt/shared/arc2/runs", "status_every_s": 300}, indent=1))
PY
( while [ ! -e $SMIS ]; do nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits >> $SMI 2>/dev/null; sleep 10; done ) &
log "$LB starting (driver b590ae16, competition mode + the retry observer, 120 evaluation tasks, budget 10800 s; 26/22, KV 4.33, balancer)"
timeout 13000 python3 $K/kaggle_run_arc2.py $KR/$LB.json > $KR/${LB}_driver.out 2>&1
rc=$?
touch $SMIS; sleep 12
log "$LB driver rc $rc: $(grep -E 'server healthy|KV sizing|harness started|harness exited|DIED|submission written|final metrics|done,|exited before|relaunch' $KR/${LB}_driver.out | cut -c1-240 | tr '\n' '|' | cut -c1-2000)"
log "$LB status: $(grep -o -E 'status [0-9.]+ h: server [A-Za-z]+; running [0-9.None]+ waiting [0-9.None]+, ([0-9]+ tok/s|-), KV [0-9.None]+, preemptions [0-9.None]+' $KR/${LB}_driver.out | sed -E 's/status ([0-9.]+) h: server ([A-Za-z]+); running ([0-9.None]+) waiting ([0-9.None]+), ([0-9]+ tok.s|-), KV ([0-9.None]{1,5})[0-9]*, preemptions ([0-9.None]+)/\1h \3r \5 kv\6 p\7/' | tr '\n' '|' | cut -c1-3000)"
log "$LB peaks MiB: $(python3 - $SMI <<'PY'
import sys
peak = {}
for line in open(sys.argv[1]):
    try:
        i, m = (int(x) for x in line.split(","))
    except ValueError:
        continue
    peak[i] = max(peak.get(i, 0), m)
print(" ".join("%d:%d" % kv for kv in sorted(peak.items())) or "-")
PY
)"
log "$LB retries: $(find $S2/receipts -name 'retries-*' -newer $KR/$LB.json 2>/dev/null | wc -l) files; oom: $(find $S2/receipts -name 'oom-*' -newer $KR/$LB.json 2>/dev/null | wc -l)"
log "$LB score: $(python3 - $KR/${LB}_submission.json $OD/arc-agi_evaluation_solutions.json <<'PY'
import json, sys
sub, sol = json.load(open(sys.argv[1])), json.load(open(sys.argv[2]))
total = 0.0
for tid, outs in sol.items():
    att = sub.get(tid) or []
    right = sum(1 for i, g in enumerate(outs) if i < len(att) and g in (att[i].get("attempt_1"), att[i].get("attempt_2")))
    total += right / len(outs)
print("%.2f of %d tasks (submission with %d tasks)" % (total, len(sol), len(sub)))
PY
)"
log "$LB harness: $(tail -n 2 /mnt/shared/arc2/runs/$LB.out 2>/dev/null | tr '\n' ' ' | cut -c1-500)"
log "chain b2 l4 ended"
