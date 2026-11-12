#!/bin/bash
# pe_r1.sh (2026-10-22, Claude session 099768dc): the MTP refit's phase E for run r1, as pre-stated in MTP_REFIT_PLAN_2026-10-22.md
# ("Phase E, pre-stated", 10:57 UTC 10-22, before any r1 result past step 1,590):
#   1. waits for r1's trainer_status.json 'finished' (authoritative; OVERDUE line at 04:30 UTC 10-23, gives up at 10:00 UTC 10-23); any
#      other final status (stopped / stream_ended / failed) is logged and the chain exits: the user decides;
#   2. heads on the CPU (state_to_head.py): final = the end state (the rolling slot whose sidecar step = STEPS); tail3 = the uniform FP64
#      average of state_keep_step6000, state_keep_step6750 and the end state (each sidecar's step checked);
#   3. scores both on next55 (sp_score_head.sh: served experts, live renewal tau K 3-5, T 1 / 0.6): final on GPU 0, tail3 on GPU 1, in
#      parallel once each GPU has no compute process (the replay server stops itself at the end of its list); if GPU 0 is still busy 30 min
#      after the trainer ended, both run on GPU 1 one after the other;
#   4. summary (results/phase_e_r1_summary.txt): the two heads against fr1 and MTP3 (their 10-22 score logs, same set); the choice by
#      IE's 10-14 rule: tail3 if its live tau at K 4, T 0.6 is >= 1.002 x final's, else final;
#   5. stitches the chosen head onto the noisc drafter (the source of production's fr1 drafter) -> mtp_ft/r1_final/drafter_r1<choice>
#      (stitch_mtp_head.py re-reads and compares every tensor bitwise).
# Kills nothing, deletes nothing; the production-server check (startup, log-prob identity, a short gameplay tau) is a separate step.
# usage: bash start_detached.sh logs/pe_r1.log pe_r1.sh
#   DRY=1: the plan on today's data, nothing scored or stitched: END_STEP / KEEP_STEPS override the states (e.g. END_STEP=1575
#   KEEP_STEPS="750 1500"), the heads go to HD (default /workspace/pe_r1_dry), the summary parses the existing fr1 / MTP3 logs.
set -u
[ "$(hostname)" = 2a159825c76b ] || { echo "not the main pod"; exit 9; }
SP=/mnt/shared/arc3-inference/mtp_refit_20261022
O=$SP/mtp_ft/r1
PY=/workspace/arc3-inference/venv-speculators/bin/python
SRC=/workspace/models/Qwen3.8-Flash-Next-mtp-draft-nvfp4a16-noisc
STEPS=$(sed -n 's/^STEPS=\([0-9][0-9]*\)$/\1/p' $SP/mtp_ft/r1_ctl/latest.env)
[ -n "$STEPS" ] || { echo "no STEPS in latest.env"; exit 2; }
END_STEP=${END_STEP:-$STEPS}; KEEP_STEPS=${KEEP_STEPS:-6000 6750}
HD=${HD:-$([ -n "${DRY:-}" ] && echo /workspace/pe_r1_dry || echo $O/heads)}
SUM=$([ -n "${DRY:-}" ] && echo $HD/summary_dry.txt || echo $SP/results/phase_e_r1_summary.txt)
cd $SP/scripts
log() { echo "[$(date -u +%T)] $*"; }
OVERDUE=$(date -u -d '2026-10-23 04:30:00' +%s); GIVEUP=$(date -u -d '2026-10-23 10:00:00' +%s); OD=0
log "phase E for r1: STEPS $STEPS, end state step $END_STEP, kept $KEEP_STEPS, heads -> $HD${DRY:+ (DRY)}"

# 1. the trainer's final status
if [ -z "${DRY:-}" ]; then
  while :; do
    ST=$($PY -c "import json; print(json.load(open('$O/trainer_status.json')).get('status', ''))" 2>/dev/null)
    case "$ST" in
      finished) break ;;
      stopped|stream_ended|failed|error) log "r1 ended with status '$ST' (not 'finished'): phase E stops here; the user decides"; exit 3 ;;
    esac
    NOW=$(date +%s)
    [ $NOW -ge $OVERDUE ] && [ $OD = 0 ] && { log "OVERDUE: r1 not finished by 04:30 UTC (status '$ST')"; OD=1; }
    [ $NOW -ge $GIVEUP ] && { log "GIVING UP: r1 not finished by 10:00 UTC (status '$ST')"; exit 4; }
    sleep 60
  done
  TEND=$(date +%s)
  log "r1 finished: $(cat $O/trainer_status.json)"
fi

# 2. heads
state_at() {   # state_at STEP -> the volume state whose sidecar step is STEP (kept states by name, the end state among the rolling slots)
  local s=$1 f
  for f in $O/state_keep_step$s.pt $O/state_a.pt $O/state_b.pt; do
    [ -s $f ] && [ -s $f.json ] || continue
    [ "$($PY -c "import json; print(json.load(open('$f.json'))['step'])")" = "$s" ] && { echo $f; return 0; }
  done
  return 1
}
FEND=$(state_at $END_STEP) || { log "FAILED: no state with step $END_STEP"; exit 5; }
TAILS=""
for s in $KEEP_STEPS; do f=$(state_at $s) || { log "FAILED: no state with step $s"; exit 5; }; TAILS="$TAILS $f"; done
log "end state $FEND; tail states$TAILS"
mkdir -p $HD
HF=$HD/head_r1_final_step$END_STEP.safetensors; HT=$HD/head_r1_tail3.safetensors
for pair in "$HF|$FEND" "$HT|$TAILS $FEND"; do
  out=${pair%%|*}; ins=${pair#*|}
  if [ -s $out ]; then log "exists: $out"; continue; fi
  nice -n 10 $PY state_to_head.py $out $ins > $HD/state_to_head.log 2>&1; rc=$?
  tail -n 3 $HD/state_to_head.log
  [ $rc = 0 ] && [ -s $out ] || { log "FAILED: state_to_head.py $out (rc $rc)"; exit 6; }
  log "head $out ($(du -h $out | cut -f1), md5 $(md5sum < $out | cut -c1-8))"
done

# 3. scores
busy() { nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .; }
if [ -n "${DRY:-}" ]; then
  log "DRY: would score (GPU 0 busy $(busy 0), GPU 1 busy $(busy 1)):"
  for t in "r1final|$HF|0" "r1tail3|$HT|1"; do IFS='|' read -r tag h g <<< "$t"; DRY=1 H=$h TAG=$tag DEV=cuda:$g bash sp_score_head.sh | sed 's/^/  /'; done
  LF=$SP/logs/score_fr1.log; LT=$SP/logs/score_fr1.log   # parse test on existing logs
else
  LF=$SP/logs/score_r1final.log; LT=$SP/logs/score_r1tail3.log
  until [ "$(busy 1)" = 0 ]; do [ $(( $(date +%s) - TEND )) -gt 3600 ] && { log "FAILED: GPU 1 still busy 60 min after the trainer"; exit 7; }; sleep 30; done
  W0=0; until [ "$(busy 0)" = 0 ]; do [ $(( $(date +%s) - TEND )) -gt 1800 ] && { W0=1; break; }; sleep 30; done
  if [ $W0 = 0 ]; then
    log "scoring final on GPU 0 and tail3 on GPU 1"
    H=$HF TAG=r1final DEV=cuda:0 bash sp_score_head.sh > $LF 2>&1 & P0=$!
    H=$HT TAG=r1tail3 DEV=cuda:1 bash sp_score_head.sh > $LT 2>&1 & P1=$!
    wait $P0; R0=$?; wait $P1; R1=$?
  else
    log "GPU 0 still busy 30 min after the trainer ($(busy 0) processes): both on GPU 1"
    H=$HF TAG=r1final DEV=cuda:1 bash sp_score_head.sh > $LF 2>&1; R0=$?
    H=$HT TAG=r1tail3 DEV=cuda:1 bash sp_score_head.sh > $LT 2>&1; R1=$?
  fi
  log "scores done (rc $R0 / $R1)"
fi

# 4. summary and the choice (anchored parses of 'prob_srv_T06   K 4: ... | live renewal tau 3.4840 (ratio ...')
CH=$($PY - "$SUM" "$LF" "$LT" $SP/logs/score_fr1.log $SP/logs/score_mtp3_base.log <<'PYEOF'
import re, sys
out, logs = sys.argv[1], dict(zip(['r1 final', 'r1 tail3', 'fr1', 'MTP3'], sys.argv[2:]))
pat = re.compile(r'^prob_srv_(T10|T06)\s+K (\d): uniform-anchor L ([0-9.]+) \| live renewal tau ([0-9.]+) ')
tau = {}
for name, path in logs.items():
    d = {}
    try:
        for line in open(path):
            m = pat.match(line)
            if m:
                d[(m.group(1), int(m.group(2)))] = float(m.group(4))
    except FileNotFoundError:
        pass
    tau[name] = d
need = [(t, k) for t in ('T10', 'T06') for k in (3, 4, 5)]
lines = ['phase E r1: next55 live renewal tau (sp_score_head.sh, served experts)',
         '%-9s %s' % ('head', '  '.join('%s K%d' % (t, k) for t, k in need))]
for name, d in tau.items():
    lines.append('%-9s %s' % (name, '  '.join(('%6.4f' % d[x]) if x in d else '   n/a' for x in need)))
f, t = tau['r1 final'].get(('T06', 4)), tau['r1 tail3'].get(('T06', 4))
if f is None or t is None:
    lines.append('CHOICE: none (a score is missing)'); choice = 'none'
else:
    choice = 'tail3' if t >= 1.002 * f else 'final'
    lines.append('rule (IE 10-14): tail3 if K 4 T 0.6 tau >= 1.002 x final: tail3 %.4f vs final %.4f (%+.2f%%) -> CHOICE: %s'
                 % (t, f, 100 * (t / f - 1), choice))
    for ref in ('fr1', 'MTP3'):
        r = tau[ref].get(('T06', 4))
        if r:
            lines.append('  chosen vs %s at K 4 T 0.6: %+.2f%%' % (ref, 100 * ((t if choice == 'tail3' else f) / r - 1)))
open(out, 'w').write('\n'.join(lines) + '\n')
print(choice)
PYEOF
)
cat $SUM | sed 's/^/  /'
case "$CH" in final) HC=$HF ;; tail3) HC=$HT ;; *) log "FAILED: no choice ('$CH')"; exit 8 ;; esac

# 5. stitch
DOUT=$SP/mtp_ft/r1_final/drafter_r1$CH
if [ -n "${DRY:-}" ]; then
  log "DRY: would stitch: $PY stitch_mtp_head.py $SRC $HC $DOUT --note 'r1 $CH ...'"
  log "DRY DONE"; exit 0
fi
[ -e $DOUT ] && { log "FAILED: $DOUT exists"; exit 9; }
mkdir -p $SP/mtp_ft/r1_final
NOTE="r1 $CH (refit 10-22: fresh from fr1 10,142, LR 4e-5, loss T 0.6, refit_r1 corpus); next55 $(grep -o 'CHOICE.*' $SUM | head -n 1)"
nice -n 10 $PY stitch_mtp_head.py $SRC $HC $DOUT --note "$NOTE" > $HD/stitch.log 2>&1; rc=$?
tail -n 5 $HD/stitch.log
[ $rc = 0 ] && [ -s $DOUT/model-mtp-draft.safetensors ] || { log "FAILED: stitch (rc $rc)"; exit 10; }
log "PHASE E DONE: choice $CH, head $HC, drafter $DOUT; next: the production-server check (separate step)"
