#!/bin/bash
# sk1_chain.sh (2026-10-22, Claude main session 099768dc; the order the user approved after IE's review, V67_DESIGN 262: phase E ->
# the s_k round -> stitch -> KV means -> live core-55 -> v7.3): the s_k round unattended, from fr1c's phase E to its paired gate.
#   1. waits for 'PHASE E DONE' on logs/pe_fr1c.log (new lines only; a phase-E FAILED / GIVING UP / 'ended with status' line stops
#      the chain); OVERDUE alarm PE_OVERDUE_AT, gives up PE_GIVEUP_AT;
#   2. waits for both GPUs idle (no compute processes) and no vLLM server (<= 60 min, else stops);
#   3. sk1_launch.sh (HEAD = the phase-E head);
#   4. waits for sk1's trainer_status 'finished' (any other final status stops the chain: the user decides; note that a graceful
#      stop + resume ends this chain too: relaunch it with SKIP_LAUNCH=1); OVERDUE launch + 4 h, gives up launch + 9 h;
#   5. the head from the end state (state_to_head.py) -> mtp_ft/sk1/heads/head_sk1_final_step1875.safetensors;
#   6. its next55 score (sp_score_head.sh, TAG sk1final) on an idle GPU;
#   7. the PAIRED gate vs the base head's anchors (fr1c<choice>, scored by pe_run.sh): paired_live_tau.py -> results/sk1_gate.txt
#      / .json, plus the per-eval trail from log.jsonl (val_stepemb_rms, alpha_srv06_step1..4, tau_srv06): the ||s_k|| growth label
#      (LINEAR: the last constant-phase increment 1,125 -> 1,500 >= 0.7 x the first 375 -> 750; PLATEAU: <= 0.3 x; else SLOWING).
# Stops there: the serving part, its tests, the stitch and the means refit are separate steps (the user).
# DRY=1 HEAD=<head>: sk1_launch.sh DRY and the gate step on existing files (BASE_ANCHORS / CAND_ANCHORS); nothing launched.
set -u
[ "$(hostname)" = 2a159825c76b ] || { echo "not the main pod"; exit 9; }
SP=/mnt/shared/arc3-inference/mtp_refit_20261022; O=$SP/mtp_ft/sk1; PL=$SP/logs/pe_fr1c.log
PY=/workspace/arc3-inference/venv-speculators/bin/python
PE_OVERDUE_AT=${PE_OVERDUE_AT:-2026-10-23 07:00:00}; PE_GIVEUP_AT=${PE_GIVEUP_AT:-2026-10-23 12:00:00}
cd $SP/scripts
log() { echo "[$(date -u +%T)] $*"; }
busy() { nvidia-smi -i $1 --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -c .; }
gate() {   # gate BASE_ANCHORS CAND_ANCHORS OUT_PREFIX
  $PY paired_live_tau.py $1 $2 --json $3.json > $3.txt 2>&1 || { log "FAILED: paired_live_tau.py (see $3.txt)"; return 1; }
  $PY - $O/log.jsonl >> $3.txt <<'PYEOF'
import json, sys
ev = []
try:
    for line in open(sys.argv[1]):
        d = json.loads(line)
        if 'val_tau_srv06' in d:
            ev.append(d)
except OSError:
    pass
print('trainer evals (next55, trainer metric): step, tau_srv06, tau_srv, alpha_srv06 steps 1-4, ||s_k|| rms rows 2-4')
for d in ev:
    print('  %5d  %.4f  %.4f  %s  %s' % (d['step'], d['val_tau_srv06'], d.get('val_tau_srv', float('nan')),
          ' '.join('%.4f' % d.get('val_alpha_srv06_step%d' % k, float('nan')) for k in (1, 2, 3, 4)),
          ' '.join('%.5f' % x for x in d.get('val_stepemb_rms', []))))
r = {d['step']: sum(d.get('val_stepemb_rms', [0])) / max(1, len(d.get('val_stepemb_rms', [0]))) for d in ev}
if all(s in r for s in (375, 750, 1125, 1500)):
    first, last = r[750] - r[375], r[1500] - r[1125]
    lab = 'LINEAR (LR-limited: extend before concluding)' if last >= 0.7 * first else ('PLATEAU' if last <= 0.3 * first else 'SLOWING')
    print('||s_k|| growth: 375->750 %+.5f, 1125->1500 %+.5f -> %s' % (first, last, lab))
else:
    print('||s_k|| growth: evals at 375 / 750 / 1125 / 1500 not all present')
PYEOF
  grep -E "^GATE|^\|\|s_k\|\| growth" $3.txt | sed 's/^/  /'
}

if [ -n "${DRY:-}" ]; then
  HEAD=${HEAD:?DRY needs HEAD} DRY=1 bash sk1_launch.sh 2>&1 | tail -n 40
  [ -n "${BASE_ANCHORS:-}" ] && [ -n "${CAND_ANCHORS:-}" ] && { log "DRY gate on $BASE_ANCHORS vs $CAND_ANCHORS"; gate $BASE_ANCHORS $CAND_ANCHORS /tmp/sk1_gate_dry; cat /tmp/sk1_gate_dry.txt; }
  log "DRY DONE"; exit 0
fi

if [ -z "${SKIP_LAUNCH:-}" ]; then
  # 1. phase E
  L0=$(wc -l < $PL 2>/dev/null || echo 0); OD=0
  log "waiting for fr1c's phase E (pe_fr1c.log from line $((L0 + 1)); OVERDUE $PE_OVERDUE_AT, gives up $PE_GIVEUP_AT)"
  while :; do
    NEW=$(tail -n +$((L0 + 1)) $PL 2>/dev/null)
    echo "$NEW" | grep -q "PHASE E DONE" && break
    echo "$NEW" | grep -q -E "FAILED|GIVING UP|ended with status" && { log "phase E did not finish: $(echo "$NEW" | grep -E 'FAILED|GIVING UP|ended with status' | tail -n 1)"; exit 2; }
    NOW=$(date -u +%s)
    [ $OD = 0 ] && [ $NOW -ge $(date -u -d "$PE_OVERDUE_AT" +%s) ] && { log "OVERDUE: no PHASE E DONE by $PE_OVERDUE_AT UTC"; OD=1; }
    [ $NOW -ge $(date -u -d "$PE_GIVEUP_AT" +%s) ] && { log "GIVING UP: no PHASE E DONE by $PE_GIVEUP_AT UTC"; exit 4; }
    sleep 60
  done
  DL=$(tail -n +$((L0 + 1)) $PL | grep "PHASE E DONE" | tail -n 1); log "phase E: $DL"
  HEAD=$(echo "$DL" | grep -oP 'head \K[^,]+'); CH=$(echo "$DL" | grep -oP 'choice \K[a-z0-9]+')
  [ -s "$HEAD" ] || { log "FAILED: no head in '$DL'"; exit 5; }
  # 2. idle GPUs, no server
  for i in $(seq 1 120); do
    [ "$(busy 0)" = 0 ] && [ "$(busy 1)" = 0 ] && ! pgrep -f "[v]llm.entrypoints.cli.main serve" > /dev/null && break
    [ $i = 120 ] && { log "FAILED: GPUs / server not idle 60 min after phase E (GPU 0 $(busy 0), GPU 1 $(busy 1) processes)"; exit 6; }
    sleep 30
  done
  # 3. launch
  HEAD=$HEAD bash sk1_launch.sh || { log "FAILED: sk1_launch.sh"; exit 7; }
  echo "CH=$CH" > $SP/jobs/sk1/base_choice.env
  for i in $(seq 1 90); do      # the launcher's own checks fail inside the detached process: catch that within 45 min
    [ -s $O/trainer_status.json ] && break     # written after the model load and the step-0 eval
    if grep -q -E "exists:|busy|a server is running|no validation|not the main pod|are alive|is alive|trainer rc|Traceback" \
         $SP/logs/fullrun_sk1_a1.log 2>/dev/null; then
      log "FAILED: the launch stopped before the trainer ran:"; tail -n 5 $SP/logs/fullrun_sk1_a1.log; exit 7
    fi
    [ $i = 90 ] && { log "FAILED: no trainer_status.json 45 min after the launch"; tail -n 5 $SP/logs/fullrun_sk1_a1.log; exit 7; }
    sleep 30
  done
  log "sk1 trainer up: $(cat $O/trainer_status.json)"
else
  CH=$(sed -n 's/^CH=//p' $SP/jobs/sk1/base_choice.env)
fi
# 4. sk1's end
T0=$(date -u +%s); OD=0
log "waiting for sk1 (base fr1c $CH; OVERDUE +4 h, gives up +9 h)"
while :; do
  ST=$($PY -c "import json; print(json.load(open('$O/trainer_status.json')).get('status', ''))" 2>/dev/null)
  case "$ST" in
    finished) break ;;
    stopped|stream_ended|failed|error) log "sk1 ended with status '$ST' (not 'finished'): the chain stops; the user decides"; exit 3 ;;
  esac
  NOW=$(date -u +%s)
  [ $OD = 0 ] && [ $((NOW - T0)) -ge 14400 ] && { log "OVERDUE: sk1 not finished 4 h after the launch (status '$ST')"; OD=1; }
  [ $((NOW - T0)) -ge 32400 ] && { log "GIVING UP: sk1 not finished 9 h after the launch"; exit 4; }
  sleep 60
done
TEND=$(date -u +%s); log "sk1 finished: $(cat $O/trainer_status.json)"
# 5. head
FEND=""
for f in $O/state_keep_step1875.pt $O/state_a.pt $O/state_b.pt; do
  [ -s $f ] && [ -s $f.json ] && [ "$($PY -c "import json; print(json.load(open('$f.json'))['step'])")" = 1875 ] && { FEND=$f; break; }
done
[ -n "$FEND" ] || { log "FAILED: no state with step 1875"; exit 5; }
mkdir -p $O/heads; HF=$O/heads/head_sk1_final_step1875.safetensors
nice -n 10 $PY state_to_head.py $HF $FEND > $O/heads/state_to_head.log 2>&1 && [ -s $HF ] || { log "FAILED: state_to_head.py"; exit 6; }
log "head $HF (md5 $(md5sum < $HF | cut -c1-8)); step_emb rows: $($PY -c "import sys; sys.path.insert(0, '.'); import step_emb as SE; print(SE.head_rows('$HF'))")"
BH=$(sed -n 's/^EXTRA=".*--init-from \([^ "]*\).*/\1/p' $SP/mtp_ft/sk1_ctl/run_a1.env)
$PY - $BH $HF <<'PYEOF' | sed 's/^/  /'
import sys
from safetensors.torch import load_file
b, c = load_file(sys.argv[1]), load_file(sys.argv[2])
diff = [n for n in b if n not in c or not __import__('torch').equal(b[n], c[n])]
extra = sorted(set(c) - set(b))
print('trunk vs the base head (%s): %d tensors, %s; extra in sk1: %s' % (sys.argv[1], len(b), 'IDENTICAL' if not diff else 'DIFFER: %s' % diff[:4], extra))
PYEOF
# 6. score on an idle GPU
G=""
for i in $(seq 1 120); do
  for g in 1 0; do [ "$(busy $g)" = 0 ] && { G=$g; break; }; done
  [ -n "$G" ] && break; sleep 30
done
[ -n "$G" ] || { log "FAILED: no idle GPU 60 min after sk1"; exit 7; }
log "scoring sk1final on GPU $G"
H=$HF TAG=sk1final DEV=cuda:$G bash sp_score_head.sh > $SP/logs/score_sk1final.log 2>&1 || { log "FAILED: sp_score_head.sh (rc $?)"; exit 8; }
grep -E "live renewal" $SP/logs/score_sk1final.log | sed 's/^/  /'
# 7. paired gate
BA=$SP/results/fr1c${CH}_valnext55_served_anchors.pt; CA=$SP/results/sk1final_valnext55_served_anchors.pt
[ -s $BA ] && [ -s $CA ] || { log "FAILED: anchors missing ($BA, $CA)"; exit 9; }
gate $BA $CA $SP/results/sk1_gate || exit 10
log "SK1 CHAIN DONE: results/sk1_gate.txt; next (the user): the serving part's GPU tests, the stitch, the means refit"
