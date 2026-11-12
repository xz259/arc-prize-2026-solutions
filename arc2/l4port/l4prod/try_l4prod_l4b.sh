#!/usr/bin/env bash
# try_l4prod_l4.sh LABEL RUN MODE [arc2_server_l4.py args...] (2026-10-31, ARC2 - main; the user: the production path on the L4,
# replacing l4c13): one production-path TP2 x PP2 launch on the 4x L4 pod (no ballast), the L4 sizing of the best core cell (l12e:
# util 0.89, 34 seqs, the l4c12 capture list, split 26/22, host embeddings, drafter head v2 via bootstrap_l4prod v5, evpp on),
# receipts, a decode bench at the KV-sized lanes, then MODE: smoke = the 8-task harness smoke (as try_l4prod.sh); pass = one pass
# over the 120 evaluation tasks at the bench's lane count (persistent traffic: the production throughput read); '-' = none. Then the
# stop. Log logs_l4/try_l4prod_l4_LABEL.log, ending 'try LABEL ended'. Derived from try_l4prod.sh (the 4090 version).
# l4b (2026-10-31 07:5x, ARC2 - main; the lane experiments): try_l4prod_l4.sh (cb08f557) with the bench's lane cap from env
# LANES_CAP (default 34, as before), for runs with ARC3_STAGE4_MAX_SEQS below 34.
# PROFILE_WINDOW=1 (default 0): w3_profile_window.py in the background during the bench (the decode-window torch profile;
# pass --set ARC3_STAGE_PROFILE_DIR=DIR --set ARC3_STAGE_PROFILE_STEPS=N, the production launcher's bounded profiler).
set -uo pipefail
D=/mnt/shared/arc2/l4port; LD=$D/logs_l4; LB=$1; R0=$2; MODE=$3; shift 3; L=$LD/try_l4prod_l4_$LB.log; P4=$D/l4prod
S2=/workspace/kvring_dev/fin74/s2; RUNS=/mnt/shared/arc2/runs; H=/mnt/shared/arc2/harness_v2l
KD=$D/sidepod/KDEF_sp20.json; PROF=g60f8inp128; PORT=1390
log() { echo "[$(date -u +'%F %T')] $*" >> $L; }
log "try $LB start pod $(hostname) (pid $$); mode $MODE; wrapper args [$*]"
n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); [ "$n" = 0 ] || { log "STOP: $n GPU processes"; log "try $LB ended"; exit 1; }
# the L4 launcher copy beside the production launcher (kaggle_run's S2)
if [ ! -f $S2/k_launch_scaling_l4.py ] || ! cmp -s $P4/k_launch_scaling_l4.py $S2/k_launch_scaling_l4.py; then
  [ -f $S2/k_launch_scaling_l4.py ] && { log "STOP: $S2/k_launch_scaling_l4.py differs from $P4's; not overwriting"; log "try $LB ended"; exit 1; }
  cp $P4/k_launch_scaling_l4.py $S2/k_launch_scaling_l4.py
fi
log "launcher $(md5sum < $S2/k_launch_scaling_l4.py | cut -c1-8), server wrapper $(md5sum < $P4/arc2_server_l4.py | cut -c1-8), bootstrap_l4prod $(md5sum < $D/bootstrap_l4prod/sitecustomize.py | cut -c1-8), KDEF_sp20 $(md5sum < $KD | cut -c1-8), kaggle_run $(md5sum < /workspace/kvring_dev/kaggle/kaggle_run.py | cut -c1-8)"
[ -e $S2/instances/$LB ] && { log "STOP: instance $LB exists; use a fresh label"; log "try $LB ended"; exit 1; }
export CUDA_DEVICE_ORDER=PCI_BUS_ID
STOPF=$LD/ballast_none_$LB.stop
python3 $P4/arc2_server_l4.py --label $LB --gpu 0,1,2,3 --port $PORT --profile $PROF --kdef $KD --health-timeout-s 3600 \
  --set ARC3_STAGE4_UTIL=0.89 --set ARC3_STAGE4_MAX_SEQS=34 --set ARC3_S2_CAPTURE=${CAPS:-1,2,4,5,8,10,16,20,34,40,80,170} \
  --set ARC3_L4PORT_EVPP=1 --set ARC3_L4PORT_PP_SPLIT=26,22 "$@" > $LD/server_$LB.out 2>&1
rc=$?
log "server wrapper exit $rc: $(grep -E 'SERVER READY|STOP|healthy after|exited before' $LD/server_$LB.out | tail -n 2 | tr '\n' ' ' | cut -c1-500)"
SL=$(cat $S2/instances/$LB/last_logdir 2>/dev/null)/server.log
log "server log $SL; errors: $(grep -h -E 'Error|Traceback|assert' $SL 2>/dev/null | grep -v -E 'max_frames|min_frames|warn' | sort | uniq -c | sort -rn | head -n 6 | tr '\n' ' ' | cut -c1-900)"
log "receipts: $(grep -h -o -E '\[l4port\] [^(]{0,120}' $SL 2>/dev/null | sort | uniq -c | tr '\n' ' ' | cut -c1-900)"
log "KV: $(grep -h -o -E 'Available KV cache memory: [0-9.]+ GiB|Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x|GPU KV cache size: [0-9,]+ tokens' $SL 2>/dev/null | sort -u | tr '\n' ' ')"
if [ $rc = 0 ]; then
  N=$(python3 $D/w3_lanes.py $SL ${LANES_CAP:-34})
  log "bench at $N lanes"
  if [ "${PROFILE_WINDOW:-0}" = 1 ]; then   # l4b: the decode-window torch profile (needs --set ARC3_STAGE_PROFILE_DIR=...)
    python3 $D/w3_profile_window.py $PORT $N 20 > $LD/profile_window_$LB.out 2>&1 &
    log "profile window armed (w3_profile_window.py, $N lanes, 20 s hold)"
  fi
  timeout 3600 python3 $D/w3_rtbench2.py $PORT $N $LD/rtbench_$LB.json 44000 16000 - 60240 > $LD/rtbench_$LB.out 2>&1
  log "bench rc $?: $(grep -E '^rtbench:' $LD/rtbench_$LB.out | tail -n 1 | cut -c1-300)"
  log "decode-only windows: $(python3 $D/w3_decode_windows.py $SL 40 3 2>&1 | tail -n +2 | tr '\n' '|' | cut -c1-700)"
  KVT=$(grep -h -m1 -o -E 'GPU KV cache size: [0-9,]+ tokens' $SL | grep -o -E '[0-9,]+' | tr -d ',')
  FCP=$(python3 -c "print(round(755 * int('${KVT:-0}') / 2148573))")
  if [ "$MODE" = smoke ]; then
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
  elif [ "$MODE" = pass ]; then
  R=$R0
  if [ "$R" = - ]; then log "smoke skipped (no run name)"; elif [ -e $RUNS/$R ]; then log "smoke skipped: $RUNS/$R exists"; else
    log "$R starting (120 evaluation tasks, $N lanes, one pass, fc-pool $FCP)"
    ( cd $H && ARC2_ARC3_HARNESS_ROOT=$S2/harness_v6 LOCAL_ANALYZER_TEMPERATURE=0.6 timeout 18000 /workspace/arc3-inference/venv-duck/bin/python -m arc2.runner \
      --out $RUNS/$R --base-url http://127.0.0.1:$PORT --tokenizer /workspace/models/Qwen3.8-Flash-Next-W4A16-intel/tokenizer.json \
      --profile $PROF --kdef $KD --limit 120 --policy rank --lanes $N --hard-lanes $((N / 2)) --passes 1 --valid-target 1 --max-passes 1 \
      --no-call keep_going --idle-limit 6 --repeat-limit 3 --observations turns --reply-caps v2c --cue-from 120000 --cap-ext 4096 \
      --marker v2e --recall-at 120000 --limit-at 100000 --wrap-at 120000 --idle-skip-status --notice 0 --temperature 0.6 \
      --admit-below 0.85 --admit-forecast --fc-pool $FCP --evict-fraction 0.2 --note-guard --prompt v2j --cap-close 32000 \
      --budget 160000 --deadline-s 16000 > $RUNS/$R.out 2>&1 )
    log "$R exit $?: $(python3 -c "import json; d=json.load(open('$RUNS/$R/score.json')); print('score %.2f outputs %d/%d wall %.0f s tok/s %.1f' % (d['score'], d['outputs_right'], d['outputs'], d['wall_seconds'], d.get('tok_per_s', 0)))" 2>&1 | tail -n 1)"
    log "$R tail: $(tail -n 3 $RUNS/$R.out | tr '\n' ' ' | cut -c1-400)"
  fi
  else
    log "no harness run (mode $MODE)"
  fi
  log "metrics: $(curl -s --max-time 20 http://127.0.0.1:$PORT/metrics | grep -E '^vllm:num_preemptions_total' | head -n 2 | tr '\n' ' ')"
  log "server errors after: $(grep -h -E 'Error|Traceback' $SL 2>/dev/null | grep -v -E 'max_frames|min_frames' | sort | uniq -c | sort -rn | head -n 4 | tr '\n' ' ' | cut -c1-600)"
fi
python3 $P4/arc2_server_l4.py --label $LB --gpu 0,1,2,3 --port $PORT --profile $PROF --kdef $KD --stop >> $LD/server_$LB.out 2>&1
log "server stop rc $?"
touch $STOPF; sleep 10
log "GPUs after: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | tr '\n' ' ')"
log "try $LB ended"
