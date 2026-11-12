#!/usr/bin/env bash
# twin_pp.sh LABEL [arc2_server_l4.py args...] (2026-10-31, ARC2 - main; the reviewer's exact-path twin, the user approved the plan):
# the PP side of the PP-vs-TP 1 teacher-forced twin on the 4090. The production profile g60f8inp128 on TP2 x PP2 (as try_l4prod.sh:
# the 4090 sizing and ballast, evpp, split 26/22) with the GDN state's stochastic rounding off (ARC3_S2_GDN_STATE=scaled16; the TP 1
# side on the RTX takes the same override), then tf_score over the 32 short streams (prompt_logprobs 20; prefill path, so MTP's
# drafts cannot enter) twice (the PP rerun floor), then the stop. The TP 1 side and the comparison (tf_cmp2) follow on the RTX.
# Reading (the reviewer, stated before the run): per-token logprobs agree with TP 1 to ~1e-3 (reduction order); a wrong layer
# partition, head-hook row, ring renumber or compaction mapping shows as >= 0.01 at specific positions. Log logs/twin_LABEL.log.
set -uo pipefail
D=/mnt/shared/arc2/l4port; LD=$D/logs; LB=$1; shift; L=$LD/twin_$LB.log; P4=$D/l4prod; SP=$D/sidepod
S2=/workspace/kvring_dev/fin74/s2; KD=$SP/KDEF_sp20.json; PROF=g60f8inp128; PORT=1390
log() { echo "[$(date -u +'%F %T')] $*" >> $L; }
log "twin $LB start pod $(hostname) (pid $$); wrapper args [$*]"
n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); [ "$n" = 0 ] || { log "STOP: $n GPU processes"; log "twin $LB ended"; exit 1; }
[ -e $S2/instances/$LB ] && { log "STOP: instance $LB exists"; log "twin $LB ended"; exit 1; }
log "bootstrap_l4prod $(md5sum < $D/bootstrap_l4prod/sitecustomize.py | cut -c1-8), evpp $(md5sum < $P4/l4port_evpp.py | cut -c1-8), streams $(md5sum < $SP/tf_streams.json | cut -c1-8), tf_score $(md5sum < $SP/tf_score.py | cut -c1-8)"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
STOPF=$LD/ballast_$LB.stop; BR=$LD/ballast_$LB.json
bash /workspace/arc3-inference/runtime_python.sh $D/ballast.py --mib 1530 --hold-s 14400 --stop-file $STOPF --report $BR > $LD/ballast_$LB.out 2>&1 &
for i in $(seq 1 60); do [ -f $BR ] && break; sleep 2; done
[ -f $BR ] || { log "STOP: ballast did not report"; touch $STOPF; log "twin $LB ended"; exit 1; }
python3 $P4/arc2_server_l4.py --label $LB --gpu 0,1,2,3 --port $PORT --profile $PROF --kdef $KD --health-timeout-s 3600 \
  --set ARC3_STAGE4_UTIL=0.85 --set ARC3_STAGE4_MAX_SEQS=16 --set ARC3_S2_CAPTURE=1,2,4,5,8,10,16,20,40,80 \
  --set ARC3_L4PORT_EVPP=1 --set ARC3_L4PORT_PP_SPLIT=26,22 --set ARC3_S2_GDN_STATE=scaled16 "$@" > $LD/server_$LB.out 2>&1
rc=$?
log "server wrapper exit $rc: $(grep -E 'SERVER READY|STOP|healthy after|exited before' $LD/server_$LB.out | tail -n 2 | tr '\n' ' ' | cut -c1-400)"
SL=$(cat $S2/instances/$LB/last_logdir 2>/dev/null)/server.log
log "GDN state receipts: $(grep -h -o -i 'scaled16[a-z]*[^ ]*' $SL 2>/dev/null | sort | uniq -c | head -n 4 | tr '\n' ' ' | cut -c1-300)"
if [ $rc = 0 ]; then
  for k in a b; do
    python3 $SP/tf_score.py $SP/tf_streams.json $LD/tf_twin_${LB}_$k.json $PORT 20 > $LD/tf_twin_${LB}_$k.out 2>&1
    log "tf_score $k rc $?: $(tail -n 1 $LD/tf_twin_${LB}_$k.out | cut -c1-200)"
  done
  python3 $SP/tf_cmp2.py $LD/tf_twin_${LB}_a.json $LD/tf_twin_${LB}_b.json 2>&1 | head -n 12 | while read -r row; do log "PP rerun: $row"; done
fi
python3 $P4/arc2_server_l4.py --label $LB --gpu 0,1,2,3 --port $PORT --profile $PROF --kdef $KD --stop >> $LD/server_$LB.out 2>&1
log "server stop rc $?"
touch $STOPF; sleep 10
log "twin $LB ended"
