#!/usr/bin/env bash
# tpident_pp.sh LABEL [arc2_server_l4.py args...] (2026-10-31, ARC2 - main; the reviewer's identity check before any long run): the
# production profile on TP2 x PP2 (4090 sizing and ballast, evpp, split 26/22) with l4port_tpident recording the first 400 model calls
# of every worker (stage outputs, logits, sampled tokens), a short decode bench (8 lanes, 3,000 + 2,000 tokens), the stop, then
# compare_tpident.py. Reading (stated before the run): IDENTICAL = the TP ranks of each stage are bitwise equal where vLLM needs it
# and the QSA count difference is the selector's tie-break (no fix); DIFFER = stop before any long run and find the non-identical
# path. Log logs/tpident_LABEL.log.
set -uo pipefail
D=/mnt/shared/arc2/l4port; LD=$D/logs; LB=$1; shift; L=$LD/tpident_$LB.log; P4=$D/l4prod; SP=$D/sidepod
S2=/workspace/kvring_dev/fin74/s2; KD=$SP/KDEF_sp20.json; PROF=g60f8inp128; PORT=1390; OUT=$LD/tpident_$LB
log() { echo "[$(date -u +'%F %T')] $*" >> $L; }
log "tpident $LB start pod $(hostname) (pid $$); tpident $(md5sum < $P4/l4port_tpident.py | cut -c1-8), compare $(md5sum < $P4/compare_tpident.py | cut -c1-8), bootstrap $(md5sum < $D/bootstrap_l4prod/sitecustomize.py | cut -c1-8)"
n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); [ "$n" = 0 ] || { log "STOP: $n GPU processes"; log "tpident $LB ended"; exit 1; }
[ -e $S2/instances/$LB ] && { log "STOP: instance $LB exists"; log "tpident $LB ended"; exit 1; }
mkdir -p $OUT
export CUDA_DEVICE_ORDER=PCI_BUS_ID
STOPF=$LD/ballast_$LB.stop; BR=$LD/ballast_$LB.json
bash /workspace/arc3-inference/runtime_python.sh $D/ballast.py --mib 1530 --hold-s 14400 --stop-file $STOPF --report $BR > $LD/ballast_$LB.out 2>&1 &
for i in $(seq 1 60); do [ -f $BR ] && break; sleep 2; done
[ -f $BR ] || { log "STOP: ballast did not report"; touch $STOPF; log "tpident $LB ended"; exit 1; }
python3 $P4/arc2_server_l4.py --label $LB --gpu 0,1,2,3 --port $PORT --profile $PROF --kdef $KD --health-timeout-s 3600 \
  --set ARC3_STAGE4_UTIL=0.85 --set ARC3_STAGE4_MAX_SEQS=16 --set ARC3_S2_CAPTURE=1,2,4,5,8,10,16,20,40,80 \
  --set ARC3_L4PORT_EVPP=1 --set ARC3_L4PORT_PP_SPLIT=26,22 --set ARC3_L4PORT_TPIDENT=400 --set ARC3_L4PORT_TPIDENT_DIR=$OUT "$@" > $LD/server_$LB.out 2>&1
rc=$?
log "server wrapper exit $rc: $(grep -E 'SERVER READY|STOP|healthy after|exited before' $LD/server_$LB.out | tail -n 2 | tr '\n' ' ' | cut -c1-300)"
SL=$(cat $S2/instances/$LB/last_logdir 2>/dev/null)/server.log
log "receipts: $(grep -h -o '\[l4port\] tpident[^(]*' $SL 2>/dev/null | sort | uniq -c | tr '\n' ' ' | cut -c1-300)"
if [ $rc = 0 ]; then
  timeout 1800 python3 $D/w3_rtbench2.py $PORT 8 $LD/rtbench_$LB.json 3000 2000 - 60240 > $LD/rtbench_$LB.out 2>&1
  log "bench rc $?: $(grep -E '^rtbench:' $LD/rtbench_$LB.out | tail -n 1 | cut -c1-200)"
fi
python3 $P4/arc2_server_l4.py --label $LB --gpu 0,1,2,3 --port $PORT --profile $PROF --kdef $KD --stop >> $LD/server_$LB.out 2>&1
log "server stop rc $?"
touch $STOPF; sleep 10
log "records: $(ls $OUT | wc -l) files, $(cat $OUT/*.jsonl 2>/dev/null | wc -l) lines"
log "identity: $(python3 $P4/compare_tpident.py $OUT 2>&1 | tr '\n' ' ' | cut -c1-1200)"
log "tpident $LB ended"
