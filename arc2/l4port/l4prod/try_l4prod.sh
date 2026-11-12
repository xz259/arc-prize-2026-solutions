#!/usr/bin/env bash
# try_l4prod.sh LABEL SMOKE_RUN [arc2_server_l4.py args...] (2026-10-31, ARC2 - main): one production-path PP launch on the 4090
# (the L4 port; the chain33 body, parameterized): GPUs idle, the L4 launcher copy in place, the L4 ballast, arc2_server_l4.py with
# the 4090 sizing (util 0.85, 16 seqs, capture to 80) plus the given args, receipts and errors; if healthy a decode bench and an
# 8-task harness smoke (SMOKE_RUN under /mnt/shared/arc2/runs; '-' skips it); then the stop. Log logs/try_l4prod_LABEL.log,
# ending 'try LABEL ended'.
set -uo pipefail
D=/mnt/shared/arc2/l4port; LD=$D/logs; LB=$1; R0=$2; shift 2; L=$LD/try_l4prod_$LB.log; P4=$D/l4prod
S2=/workspace/kvring_dev/fin74/s2; RUNS=/mnt/shared/arc2/runs; H=/mnt/shared/arc2/harness_v2l
KD=$D/sidepod/KDEF_sp20.json; PROF=g60f8inp128; PORT=1390
log() { echo "[$(date -u +'%F %T')] $*" >> $L; }
log "try $LB start pod $(hostname) (pid $$); wrapper args [$*]"
n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); [ "$n" = 0 ] || { log "STOP: $n GPU processes"; log "try $LB ended"; exit 1; }
# the L4 launcher copy beside the production launcher (kaggle_run's S2)
if [ ! -f $S2/k_launch_scaling_l4.py ] || ! cmp -s $P4/k_launch_scaling_l4.py $S2/k_launch_scaling_l4.py; then
  [ -f $S2/k_launch_scaling_l4.py ] && { log "STOP: $S2/k_launch_scaling_l4.py differs from $P4's; not overwriting"; log "try $LB ended"; exit 1; }
  cp $P4/k_launch_scaling_l4.py $S2/k_launch_scaling_l4.py
fi
log "launcher $(md5sum < $S2/k_launch_scaling_l4.py | cut -c1-8), server wrapper $(md5sum < $P4/arc2_server_l4.py | cut -c1-8), bootstrap_l4prod $(md5sum < $D/bootstrap_l4prod/sitecustomize.py | cut -c1-8), KDEF_sp20 $(md5sum < $KD | cut -c1-8), kaggle_run $(md5sum < /workspace/kvring_dev/kaggle/kaggle_run.py | cut -c1-8)"
[ -e $S2/instances/$LB ] && { log "STOP: instance $LB exists; use a fresh label"; log "try $LB ended"; exit 1; }
# ballast: the L4's 23,034 MiB on the 4090's 24,564
export CUDA_DEVICE_ORDER=PCI_BUS_ID
STOPF=$LD/ballast_$LB.stop; BR=$LD/ballast_$LB.json
bash /workspace/arc3-inference/runtime_python.sh $D/ballast.py --mib 1530 --hold-s 14400 --stop-file $STOPF --report $BR > $LD/ballast_$LB.out 2>&1 &
BPID=$!
for i in $(seq 1 60); do [ -f $BR ] && break; sleep 2; done
[ -f $BR ] || { log "STOP: ballast did not report"; touch $STOPF; log "try $LB ended"; exit 1; }
log "ballast pid $BPID: $(tr -d '\n' < $BR | cut -c1-200)"
python3 $P4/arc2_server_l4.py --label $LB --gpu 0,1,2,3 --port $PORT --profile $PROF --kdef $KD --health-timeout-s 3600 \
  --set ARC3_STAGE4_UTIL=0.85 --set ARC3_STAGE4_MAX_SEQS=16 --set ARC3_S2_CAPTURE=1,2,4,5,8,10,16,20,40,80 "$@" > $LD/server_$LB.out 2>&1
rc=$?
log "server wrapper exit $rc: $(grep -E 'SERVER READY|STOP|healthy after|exited before' $LD/server_$LB.out | tail -n 2 | tr '\n' ' ' | cut -c1-500)"
SL=$(cat $S2/instances/$LB/last_logdir 2>/dev/null)/server.log
log "server log $SL; errors: $(grep -h -E 'Error|Traceback|assert' $SL 2>/dev/null | grep -v -E 'max_frames|min_frames|warn' | sort | uniq -c | sort -rn | head -n 6 | tr '\n' ' ' | cut -c1-900)"
log "receipts: $(grep -h -o -E '\[l4port\] [^(]{0,120}' $SL 2>/dev/null | sort | uniq -c | tr '\n' ' ' | cut -c1-900)"
log "KV: $(grep -h -o -E 'Available KV cache memory: [0-9.]+ GiB|Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x|GPU KV cache size: [0-9,]+ tokens' $SL 2>/dev/null | sort -u | tr '\n' ' ')"
if [ $rc = 0 ]; then
  N=$(python3 $D/w3_lanes.py $SL 16)
  log "bench at $N lanes"
  timeout 3600 python3 $D/w3_rtbench2.py $PORT $N $LD/rtbench_$LB.json 44000 16000 - 60240 > $LD/rtbench_$LB.out 2>&1
  log "bench rc $?: $(grep -E '^rtbench:' $LD/rtbench_$LB.out | tail -n 1 | cut -c1-300)"
  log "decode-only windows: $(python3 $D/w3_decode_windows.py $SL 40 3 2>&1 | tail -n +2 | tr '\n' '|' | cut -c1-700)"
  KVT=$(grep -h -m1 -o -E 'GPU KV cache size: [0-9,]+ tokens' $SL | grep -o -E '[0-9,]+' | tr -d ',')
  FCP=$(python3 -c "print(round(755 * int('${KVT:-0}') / 2148573))")
  R=$R0
  if [ "$R" = - ]; then log "smoke skipped (no run name)"; elif [ -e $RUNS/$R ]; then log "smoke skipped: $RUNS/$R exists"; else
    log "$R starting (8 evaluation tasks, 8 lanes, one pass, fc-pool $FCP)"
    ( cd $H && ARC2_ARC3_HARNESS_ROOT=$S2/harness_v6 LOCAL_ANALYZER_TEMPERATURE=0.6 timeout 2400 /workspace/arc3-inference/venv-duck/bin/python -m arc2.runner \
      --out $RUNS/$R --base-url http://127.0.0.1:$PORT --tokenizer /workspace/models/Qwen3.8-Flash-Next-W4A16-intel/tokenizer.json \
      --profile $PROF --kdef $KD --limit 8 --policy rank --lanes 8 --hard-lanes 4 --passes 1 --valid-target 1 --max-passes 1 \
      --no-call keep_going --idle-limit 6 --repeat-limit 3 --observations turns --reply-caps v2c --cue-from 120000 --cap-ext 4096 \
      --marker v2e --recall-at 120000 --limit-at 100000 --wrap-at 120000 --idle-skip-status --notice 0 --temperature 0.6 \
      --admit-below 0.85 --admit-forecast --fc-pool $FCP --evict-fraction 0.2 --note-guard --prompt v2j --cap-close 32000 \
      --budget 160000 --deadline-s 1500 > $RUNS/$R.out 2>&1 )
    log "$R exit $?: $(python3 -c "import json; d=json.load(open('$RUNS/$R/score.json')); print('score %.2f outputs %d/%d wall %.0f s tok/s %.1f' % (d['score'], d['outputs_right'], d['outputs'], d['wall_seconds'], d.get('tok_per_s', 0)))" 2>&1 | tail -n 1)"
    log "$R tail: $(tail -n 3 $RUNS/$R.out | tr '\n' ' ' | cut -c1-400)"
  fi
  log "server errors after: $(grep -h -E 'Error|Traceback' $SL 2>/dev/null | grep -v -E 'max_frames|min_frames' | sort | uniq -c | sort -rn | head -n 4 | tr '\n' ' ' | cut -c1-600)"
fi
python3 $P4/arc2_server_l4.py --label $LB --gpu 0,1,2,3 --port $PORT --profile $PROF --kdef $KD --stop >> $LD/server_$LB.out 2>&1
log "server stop rc $?"
touch $STOPF; sleep 10
log "GPUs after: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
log "try $LB ended"
