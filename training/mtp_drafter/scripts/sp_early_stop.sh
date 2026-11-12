#!/bin/bash
# sp_early_stop.sh (2026-10-13, Claude fork 5d20b23b; side pod only): early stop of a loss-ablation arm (user 10-13: stop early if
# the early sign is not good). Pre-stated rule: compare the arm's trainer validation tau_srv (served pair, T 1, the product
# over K 4 steps, the same held-out windows for every arm) with the reference arm's at steps 600 and 900; if the arm trails
# by more than 0.02 at both, create the arm's stop file (the trainer saves a state and exits; sp_round_arm2.sh then skips
# scoring). Creates only that stop file. usage: RUN=r7_... REF=r6_... bash start_detached.sh logs/early_stop_RUN.log sp_early_stop.sh
set -u
SP=/mnt/shared/arc3-inference/sidepod_20261012
RUN=${RUN:?run}; REF=${REF:?reference run}; MARGIN=${MARGIN:-0.02}; ORIG=$RUN
OV=$SP/jobs/arm_overrides/$RUN.env; [ -s $OV ] && . $OV       # the arm may have been renamed before it started
taus() {   # step tau_srv lines of a run's trainer log
  f=$(ls -t $SP/logs/mtp_ft_${1}_*.log 2>/dev/null | head -n 1); [ -n "$f" ] || return
  python3 - $f <<'PYEOF'
import ast, re, sys
for line in open(sys.argv[1]):
    m = re.match(r'step (\d+) val (\{.*\})', line)
    if m:
        d = ast.literal_eval(re.sub(r'\bnan\b', 'None', m.group(2)))
        if 'tau_srv' in d:
            print(m.group(1), d['tau_srv'])
PYEOF
}
echo "[$(date -u +%T)] early-stop watch: $RUN vs $REF (margin $MARGIN at steps 600 and 900)"
for i in $(seq 1 600); do
  A=$(taus $RUN); B=$(taus $REF)
  a6=$(echo "$A" | awk '$1==600{print $2}'); a9=$(echo "$A" | awk '$1==900{print $2}')
  b6=$(echo "$B" | awk '$1==600{print $2}'); b9=$(echo "$B" | awk '$1==900{print $2}')
  if [ -n "$a9" ] && [ -n "$b9" ] && [ -n "$a6" ] && [ -n "$b6" ]; then
    V=$(python3 -c "print(int($a6 < $b6 - $MARGIN and $a9 < $b9 - $MARGIN))")
    echo "[$(date -u +%T)] step 600: $a6 vs $b6; step 900: $a9 vs $b9 -> $([ $V = 1 ] && echo STOP || echo continue)"
    [ $V = 1 ] && touch $SP/mtp_ft/$RUN/STOP && echo "[$(date -u +%T)] stop file created for $RUN"
    exit 0
  fi
  grep -q "arm $RUN done" $SP/logs/round7_$ORIG.log 2>/dev/null && { echo "arm ended before step 900"; exit 0; }
  sleep 120
done
