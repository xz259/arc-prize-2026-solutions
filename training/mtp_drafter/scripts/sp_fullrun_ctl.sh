#!/bin/bash
# 2026-10-22 MTP refit port (Claude session 099768dc; from 569941e0): host 2a159825c76b, SP = mtp_refit_20261022, S2 = fin73/s2, the
# replay processes include lc_fin.sh / mtp_replay_* / p6_replay_arm_i.sh, the resume list is built from the valnext55 training list.
# The pod-move hints below still name the side pod's launchers: the refit replay is stopped through its own launcher (mtp_replay_ps.sh).
# sp_fullrun_ctl.sh (2026-10-13, Claude fork 5d20b23b; side pod only): one control script for the MTP full run (Phase 6
# supervision guide, principle 2). Finds the run's processes by MTP_RUN=<RUN> in their environment, never by a name
# pattern. Kills nothing and deletes nothing.
#   status RUN            run settings (latest.env), last health line, trainer status, newest states, live processes
#   stop RUN [MIN]        graceful stop: creates <out>/STOP; the trainer saves a full state at the end of its step, copies it
#                         to the volume and exits; waits up to MIN minutes (default 60) for "stopped". The replay keeps
#                         running (a same-pod trainer restart needs no replay); stopping the replay server and driver for a
#                         pod move is the user's step (printed).
#   resume RUN [GPUS]     newest complete state (sidecars), then:
#                         - replay alive and the trainer stopped gracefully: restart the trainer only (TRAINER_ONLY);
#                         - nothing alive: build the resume list (mtp_resume_sessions.py: sessions with segments neither
#                           consumed nor waiting in /dev/shm; failed sessions skipped and listed), start attempt K+1 (replay of
#                           the list + trainer from the state), the recipe from latest.env; GPUS (e.g. 1,2,3) changes the
#                           trainer GPUs with ACCUM rescaled so rows per step stay the same;
#                         - otherwise refuses (a crashed trainer with a live replay: stop the replay first).
#                         NEW_<VAR> overrides (LR ROUTER DECAY_START STEPS WARMUP CKPT KEEP EVAL UNFREEZE EXPERT_LR
#                         EXPERT_WARMUP REPLAY_GPUS LRFLOOR LOSST (10-22: LOSST, LRFLOOR, VOCAB, DRAFTCFG, STREAMS, VFEAT, VPACK, VALN
#                         now travel with a resume; they were dropped before); the 4-GPU pod: NEW_REPLAY_GPUS=0,1 ... resume RUN 2,3); DECAY_NOW=1 [COOLDOWN=N] starts the final cooldown at the resumed step.
#                         NEW_UNFREEZE=1 (the experts' escape hatch) first keeps the resumed state as state_frozen_step<N>.pt;
#                         revert: FROM=<that state> NEW_UNFREEZE=0 with nothing alive (the sessions trained after it are
#                         replayed again). FROM=<state> resumes from that state instead of the newest (full restart only),
#                         except FROM=<milestone state> DECAY_NOW=1 (user 10-14: at a plateau, anneal from an earlier
#                         milestone): a trainer-only restart is allowed; the stretch trained after that state is abandoned.
#                         "Newest state" = the newest written by the current attempt (sidecar time >= its start), else the
#                         state it started from, so an abandoned stretch's higher-step states are never resumed.
#                         DRY=1 prints the launch instead of running it.
# usage: bash sp_fullrun_ctl.sh status|stop|resume RUN [arg]
set -u
[ "$(hostname)" = 2a159825c76b ] || { echo "not the main pod"; exit 9; }
SP=${SP_ROOT:-/mnt/shared/arc3-inference/mtp_refit_20261022}          # SP_ROOT: tests only
PY=/workspace/arc3-inference/venv-speculators/bin/python
S2=/workspace/kvring_dev/fin73/s2
CMD=${1:?status|stop|resume}; RUN=${2:?run}; ARG=${3:-}
OUT=$SP/mtp_ft/$RUN; CTL=$SP/mtp_ft/${RUN}_ctl; LOC=${LOC_ROOT:-/workspace}/mtpfull_state_$RUN; FEAT=/dev/shm/mtpfull_$RUN/features
[ -s $CTL/latest.env ] || { echo "no $CTL/latest.env"; exit 1; }
. $CTL/latest.env
cd $SP/scripts || exit 1
procs() {   # "pid kind cmdline" for every process of the run
  for p in /proc/[0-9]*; do
    pid=${p#/proc/}; [ "$pid" = $$ ] && continue
    tr '\0' '\n' 2>/dev/null < $p/environ | grep -qx "MTP_RUN=$RUN" || continue
    c=$(tr '\0' ' ' 2>/dev/null < $p/cmdline)
    case "$c" in
      *train_mtp_sp.py*) k=trainer;; *replay_driver*|*lc_run3*|*lc_fin*|*mtp_replay*|*phase6_replay*|*p6_replay_arm*|*sp_fullrun_replay*) k=replay;;
      *vllm.entrypoints*) k=server;; *sp_fullrun_health*) k=health;; *sp_fullrun.sh*) k=launcher;; *) k=other;;
    esac
    echo "$pid $k $(echo "$c" | cut -c1-90)"
  done
}
newest_state() {   # step path of the newest complete state of the current attempt (else the state it started from)
  $PY - $OUT "${STARTED:-}" "${RESUME:-}" <<'PYEOF'
import glob, json, os, sys, time
out, started, resume = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    t0 = time.strptime(started, '%Y-%m-%dT%H:%M:%SZ')
except ValueError:
    t0 = None                                 # no attempt time (tests, old env): every state counts
best = None
for p in glob.glob(os.path.join(out, 'state_*.pt.json')):
    try:
        d = json.load(open(p)); f = p[:-5]
        if os.path.getsize(f) != d['bytes']:
            continue
        if t0 is not None and 'time' in d and time.strptime(d['time'], '%Y-%m-%dT%H:%M:%SZ') < t0:
            continue                          # written before this attempt: an earlier attempt's (maybe abandoned) state
        if best is None or d['step'] > best[0]:
            best = (d['step'], f)
    except Exception:
        pass
if best is None and resume and os.path.exists(resume + '.json'):
    best = (json.load(open(resume + '.json'))['step'], resume)    # nothing written yet: the state this attempt started from
print('%d %s' % best if best else '0 none')
PYEOF
}
case $CMD in
status)
  echo "== $RUN attempt $ATTEMPT (started $STARTED): lr $LR K $NK rows ${WINDOW}x${ACCUM}x$NRANKS steps $STEPS gpus $TRAIN_GPUS replay $REPLAY_LAB"
  echo "== trainer status: $(cat $OUT/trainer_status.json 2>/dev/null)"
  echo "== newest state on the volume: $(newest_state)"
  echo "== local scratch: $(cat $LOC/state_scratch.pt.json 2>/dev/null | cut -c1-120)"
  echo "== health: $(tail -n 1 $LOC/health.jsonl 2>/dev/null)"
  echo "== processes:"; procs
  ;;
stop)
  P=$(procs | awk '$2 == "trainer"')
  [ -n "$P" ] || { echo "no trainer process of $RUN"; exit 1; }
  touch $OUT/STOP
  echo "[$(date -u +%T)] stop requested ($OUT/STOP); waiting up to ${ARG:-60} min for the state"
  for i in $(seq 1 $((${ARG:-60} * 6))); do
    S=$(cat $OUT/trainer_status.json 2>/dev/null)
    case "$S" in *'"stopped"'*) echo "[$(date -u +%T)] $S"; break;; esac
    sleep 10
  done
  case "$(cat $OUT/trainer_status.json 2>/dev/null)" in *'"stopped"'*) ;; *) echo "not stopped yet: check status"; exit 1;; esac
  echo "same-pod restart: bash sp_fullrun_ctl.sh resume $RUN"
  IFS=',' read -ra RGS <<< "${REPLAY_GPUS:-0}"
  if [ ${#RGS[@]} -gt 1 ]; then      # two-replay layout: each instance's own server (never minimal_inference.py stop)
    echo "pod move: the user stops each replay instance's server:"
    for i in "${!RGS[@]}"; do
      echo "  cd /workspace/arc3-inference && ARC3_S2_INSTANCE=${REPLAY_LAB}_i$i ARC3_S2_GPU=${RGS[$i]} ARC3_PORT=$((1300 + ${RGS[$i]})) \\"
      echo "    bash runtime_python.sh $S2/launch_scaling_s2i.py stop"
    done
    echo "  then the replay driver / lc_run3i processes listed here, by PID; on the new pod: NEW_REPLAY_GPUS=0,1 resume $RUN 2,3"
  else
    echo "pod move: the user stops the replay (cd /workspace/arc3-inference && bash runtime_python.sh minimal_inference.py stop;"
    echo "          then the replay driver / lc_run3 processes listed here, by PID), then on the new pod: NEW_REPLAY_GPUS=0,1 resume $RUN 2,3"
  fi
  procs | awk '$2 != "trainer"'
  ;;
resume)
  P=$(procs)
  echo "$P" | awk '$2 == "trainer"' | grep -q . && { echo "a trainer of $RUN is alive:"; echo "$P"; exit 1; }
  read -r STEP STATE <<< "$(newest_state)"
  [ "$STATE" != none ] || { echo "no complete state of $RUN on the volume"; exit 1; }
  if [ -n "${FROM:-}" ]; then          # an older state (the revert after an unfreeze trial): its later segments are gone
    [ -s "$FROM" ] && [ -s "$FROM.json" ] || { echo "no state or sidecar: $FROM"; exit 1; }
    if [ -z "${DECAY_NOW:-}" ]; then
      echo "$P" | awk '$2 == "replay" || $2 == "server"' | grep -q . && { echo "FROM needs a full restart: stop the replay first"; exit 1; }
    fi
    STEP=$($PY -c "import json, sys; print(json.load(open(sys.argv[1]))['step'])" "$FROM.json"); STATE=$FROM
    echo "from $STATE (step $STEP) instead of the newest state"
  fi
  ST=$(cat $OUT/trainer_status.json 2>/dev/null)
  NEXT=$((ATTEMPT + 1))
  # overrides at a resume (user / IE 10-13): NEW_<VAR> for LR ROUTER DECAY_START STEPS WARMUP CKPT KEEP EVAL (e.g. halve the
  # router LR on the entropy guard); DECAY_NOW=1 starts the final cooldown at the resumed step, over COOLDOWN steps
  # (default 20% of STEPS), the user's manual anneal at a plateau
  PREV_UNFREEZE=${UNFREEZE:-0}
  for v in LR ROUTER DECAY_START STEPS WARMUP CKPT KEEP EVAL UNFREEZE EXPERT_LR EXPERT_WARMUP REPLAY_GPUS LRFLOOR LOSST STREAMS; do   # STREAMS: 10-22 fr1c (the eval sessions' roles)
    eval "nv=\${NEW_$v:-}"; [ -n "$nv" ] && { echo "override $v: ${!v:-} -> $nv"; eval "$v=$nv"; }
  done
  case " $EXTRA " in *" --init-from "*)   # 10-22 (sk1): the trainer refuses --init-from with --resume; the state carries the weights
    EXTRA=$(echo "$EXTRA" | sed -E 's/--init-from +[^ ]+//; s/  +/ /g; s/^ +//; s/ +$//')
    echo "resume: --init-from dropped from EXTRA (the state carries the weights): '$EXTRA'";;
  esac
  if [ "${UNFREEZE:-0}" = 1 ] && [ "$PREV_UNFREEZE" != 1 ]; then      # keep the last frozen state for the revert
    KS=$OUT/state_frozen_step$STEP.pt
    if [ -n "${DRY:-}" ]; then echo "DRY: would keep $STATE as $KS"
    elif [ ! -e $KS ]; then
      cp $STATE $KS.tmp && mv $KS.tmp $KS && cp $STATE.json $KS.json || { echo "could not keep the pre-unfreeze state"; exit 1; }
      echo "pre-unfreeze state kept: $KS"
    fi
  fi
  if [ -n "${DECAY_NOW:-}" ]; then
    CD=${COOLDOWN:-$((STEPS / 5))}; DECAY_START=$((STEP + 1)); STEPS=$((STEP + CD))
    echo "decay now: cosine decay from step $DECAY_START over $CD steps (run ends at step $STEPS)"
  fi
  NG=${ARG:-$TRAIN_GPUS}; NNR=$(echo $NG | tr ',' '\n' | wc -l)
  ROWS=$((ACCUM * NRANKS)); [ $((ROWS % NNR)) = 0 ] || { echo "windows per step $ROWS not divisible by $NNR ranks"; exit 1; }
  NACC=$((ROWS / NNR))
  COMMON="RUN=$RUN ATTEMPT=$NEXT RESUME=$STATE LR=$LR TRAIN=$TRAIN OPT=$OPT NK=$NK SW=$SW LOSS=$LOSS CLOSS=$CLOSS ACCEPT='$ACCEPT' \
WINDOW=$WINDOW ACCUM=$NACC TRAIN_GPUS=$NG ROUTER=$ROUTER MOE=$MOE CKPTSTEPS='$CKPTSTEPS' STEPS=$STEPS WARMUP=$WARMUP DECAY=$DECAY \
DECAY_START=$DECAY_START CKPT=$CKPT KEEP=$KEEP EVAL=$EVAL POOL=$POOL EXTRA='$EXTRA' EXPW=${EXPW:-} UNFREEZE=${UNFREEZE:-0} \
EXPERT_LR=${EXPERT_LR:-0.5} EXPERT_WARMUP=${EXPERT_WARMUP:-200} MINWIN=${MINWIN:-0} TOKW=${TOKW:-0} REPLAY_GPUS=${REPLAY_GPUS:-0} \
LOSST=${LOSST:-1.0} LRFLOOR=${LRFLOOR:-0} VALN=${VALN:-55}${VOCAB:+ VOCAB=$VOCAB}${DRAFTCFG:+ DRAFTCFG=$DRAFTCFG}${STREAMS:+ STREAMS=$STREAMS}\
${VFEAT:+ VFEAT=$VFEAT}${VPACK:+ VPACK=$VPACK}"   # 10-22 (r2): the refit's settings travel with a resume (LOSST / LRFLOOR were dropped)
  if echo "$P" | awk '$2 == "replay" || $2 == "server"' | grep -q .; then
    case "$ST" in *'"stopped"'*) ;; *) echo "replay alive but the trainer did not stop gracefully ($ST):"; \
      echo "segments trained after step $STEP are gone; stop the replay first (user), then resume"; exit 1;; esac
    L="$COMMON TRAINER_ONLY=1 REPLAY_LAB=$REPLAY_LAB SESS=$SESS"
    echo "[$(date -u +%T)] trainer-only restart from step $STEP ($STATE), replay $REPLAY_LAB continues"
  else
    RL=$CTL/resume_a$NEXT.json
    DRV=$(ls $SP/logs/driver_mtpfull_${RUN}*.jsonl 2>/dev/null | tr '\n' ' ')
    $PY mtp_resume_sessions.py --sessions ${SESS:?no SESS in latest.env} --sidecar $STATE.json --driver $DRV \
      --stream-dir $FEAT --out $RL || exit 1
    L="$COMMON SESS=$RL"
    echo "[$(date -u +%T)] attempt $NEXT from step $STEP ($STATE): replay $RL, trainer GPUs $NG (accum $NACC)"
  fi
  if [ -n "${DRY:-}" ]; then echo "DRY: env $L bash start_detached.sh $SP/logs/fullrun_${RUN}_a$NEXT.log sp_fullrun.sh"; exit 0; fi
  cd $SP/scripts && eval "env $L bash start_detached.sh $SP/logs/fullrun_${RUN}_a$NEXT.log sp_fullrun.sh"
  ;;
*) echo "unknown command $CMD"; exit 1;;
esac
