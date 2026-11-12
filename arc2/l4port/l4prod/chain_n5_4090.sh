#!/usr/bin/env bash
# chain n5, 4090 (2026-10-31 11:0x, ARC2 - main): the Kaggle driver (kaggle_run_arc2.py 6a316849, the package's) with Kaggle's
# harness settings (2 passes + the phase-3 pass, rank, admission) over the 120 evaluation tasks for 6 h on the 4090 under the L4
# ballast (1,530 MiB, the try_l4prod_4090.sh sizing: util 0.85, 16 sequences, capture to 80, split 26/22): the phase-3 code path and
# the driver over hours on PP (the L4 runs used one-pass settings), and a 2-pass score of the PP path beside the RTX runs (60K
# prune 128 A_in: 104.67 with the new freq set; this run uses the package's training-ranked set). Label kg4090a. Afterwards the
# pass line (check_l4prod.py v2, >= 50 evictions), allocator retries and peaks. Ends 'chain n5 4090 ended'.
set -uo pipefail
D=/mnt/shared/arc2/l4port; LD=$D/logs; L=$LD/chain_n5_4090.log; P4=$D/l4prod; K=/mnt/shared/arc2/kaggle; S2=/workspace/kvring_dev/fin74/s2
OD=/mnt/shared/arc2/official/data; KR=$K/rehearsal
log() { echo "[$(date -u +'%F %T')] $*" >> $L; }
stop() { log "STOP: $*"; log "chain n5 4090 ended"; exit 1; }
log "start pod $(hostname) (pid $$); driver $(md5sum < $K/kaggle_run_arc2.py | cut -c1-8), check $(md5sum < $P4/check_l4prod.py | cut -c1-8)"
until grep -q -E '^\[[0-9: -]+\] chain n4 4090 ended$' $LD/chain_n4_4090.log 2>/dev/null; do sleep 30; done
n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); [ "$n" = 0 ] || stop "$n GPU processes"
[ -e $S2/instances/kg4090a ] && stop "instance kg4090a exists"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
STOPF=$LD/ballast_kg4090a.stop; BR=$LD/ballast_kg4090a.json
bash /workspace/arc3-inference/runtime_python.sh $D/ballast.py --mib 1530 --hold-s 30000 --stop-file $STOPF --report $BR > $LD/ballast_kg4090a.out 2>&1 &
for i in $(seq 1 60); do [ -f $BR ] && break; sleep 2; done
[ -f $BR ] || { touch $STOPF; stop "ballast did not report"; }
log "ballast: $(head -c 300 $BR)"
python3 - > $KR/kg4090a.json <<PY
import json, time
print(json.dumps({"label": "kg4090a", "notebook_start": time.time(), "budget_s": 21900, "teardown_margin_s": 300, "server_wait_s": 2400,
                  "l4_sets": {"ARC3_STAGE4_UTIL": "0.85", "ARC3_STAGE4_MAX_SEQS": "16", "ARC3_S2_CAPTURE": "1,2,4,5,8,10,16,20,40,80",
                              "ARC3_L4PORT_EVPP": "1", "ARC3_L4PORT_PP_SPLIT": "26,22"},
                  "challenges": "$OD/arc-agi_evaluation_challenges.json", "solutions": "$OD/arc-agi_evaluation_solutions.json",
                  "submission_path": "$KR/kg4090a_submission.json", "runs_dir": "/mnt/shared/arc2/runs", "status_every_s": 600}, indent=1))
PY
log "kg4090a starting (driver, Kaggle harness settings, 120 evaluation tasks, budget 21900 s, 4090 sizing + ballast)"
timeout 23500 python3 $K/kaggle_run_arc2.py $KR/kg4090a.json > $LD/kg4090a_driver.out 2>&1
log "kg4090a driver rc $?: $(grep -E 'server healthy|KV sizing|host RAM|harness started|harness exited|DIED|relaunch|submission written|harness score|final metrics|server stop|done,|exited before' $LD/kg4090a_driver.out | cut -c1-260 | tr '\n' '|' | cut -c1-2500)"
TL=$KR/kg4090a_trylog.txt
python3 - /mnt/shared/arc2/runs/kg4090a/score.json > $TL 2>/dev/null <<'PY'
import json, sys, time
d = json.load(open(sys.argv[1]))
print("[%s] kg4090a exit 0: score %.2f outputs %d/%d wall %.0f s tok/s %.1f" % (time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime()), d["score"],
      d["outputs_right"], d["outputs"], d["wall_seconds"], d.get("tok_per_s", 0)))
PY
grep -h -o 'final metrics: .*' $LD/kg4090a_driver.out | sed 's/^final metrics: /[metrics] metrics: /' >> $TL
log "kg4090a pass line: $(python3 $P4/check_l4prod.py $S2/instances/kg4090a "$(cat $S2/instances/kg4090a/last_logdir 2>/dev/null)/server.log" $TL --min-evicts 50 2>&1 | tr '\n' ' ' | cut -c1-1500)"
log "kg4090a score_by_passes: $(python3 -c "import json; d=json.load(open('/mnt/shared/arc2/runs/kg4090a/score.json')); print({k: d.get(k) for k in ('score', 'score_without_reviews', 'score_by_passes', 'outputs_right', 'outputs', 'tok_per_s')})" 2>&1 | cut -c1-600)"
log "kg4090a retries: $(find $S2/receipts -name 'retries-*' -newer $KR/kg4090a.json 2>/dev/null | wc -l) files"
touch $STOPF; sleep 10
log "chain n5 4090 ended"
