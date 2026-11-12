#!/usr/bin/env bash
# chain n1, L4 (2026-10-31, ARC2 - main; the user: replace l4c13 with the production path on the L4). After l4c12 ends and the
# 4090's chain n1 logs 'smoke gate PASS': the production path on the 4x L4 (try_l4prod_l4.sh l4n1, mode pass: lanes from the KV
# sizing, a decode bench, one pass over the 120 evaluation tasks at the bench's lane count = the persistent-traffic throughput read;
# no byte verification here, that is the 4090's). Capture list (pre-stated rule): l12ec's shorter list if it gave more lanes than
# l12e and a median decode-only tok/s >= l12e's - 1%, else l12e's. Pass line: check_l4prod.py with >= 50 evictions. Ends
# 'chain n1 l4 ended'. v2: the production-path split scan (24/24, 25/23) before the pass (see below).
set -uo pipefail
D=/mnt/shared/arc2/l4port; P4=$D/l4prod; LD=$D/logs_l4; L=$LD/chain_n1_l4.log; S2=/workspace/kvring_dev/fin74/s2
log() { echo "[$(date -u +'%F %T')] $*" >> $L; }
stop() { log "STOP: $*"; log "chain n1 l4 ended"; exit 1; }
log "start (pid $$); try_l4 $(md5sum < $P4/try_l4prod_l4.sh | cut -c1-8), check $(md5sum < $P4/check_l4prod.py | cut -c1-8)"
DL=$(( $(date +%s) + 6 * 3600 ))
until grep -q -E '^\[[0-9: -]+\] l4c12 ended$' $LD/l4c12.log 2>/dev/null; do
  [ "$(date +%s)" -ge "$DL" ] && stop "OVERDUE waiting for l4c12"
  sleep 30
done
log "l4c12: $(grep -E '^\[[0-9: -]+\] summary:' $LD/l4c12.log | tail -n 1 | cut -c22-400)"
until grep -q -E '^\[[0-9: -]+\] (smoke gate PASS|chain n1 4090 ended)$' $D/logs/chain_n1_4090.log 2>/dev/null; do
  [ "$(date +%s)" -ge "$DL" ] && stop "OVERDUE waiting for the 4090 smoke gate"
  sleep 30
done
grep -q -E '^\[[0-9: -]+\] smoke gate PASS$' $D/logs/chain_n1_4090.log || stop "the 4090 smoke gate did not pass"
lanes() { grep -h -o -E "trial $1: lanes [0-9]+" $LD/l4c12.log | grep -o -E '[0-9]+$' | tail -n 1; }
tps() { python3 - "$LD/w3_$1.out" <<'PY'
import re, sys
try:
    t = open(sys.argv[1]).read()
except OSError:
    print(0); sys.exit()
best = (0, 0.0)
for m in re.finditer(r"running\s+(\d+):\s+(\d+) windows, median\s+([0-9.]+) tok/s", t):
    best = max(best, (int(m.group(2)), float(m.group(3))))
print(best[1])
PY
}
LE=$(lanes l12e); LC=$(lanes l12ec); TE=$(tps l12e); TC=$(tps l12ec)
CAPS=1,2,4,5,8,10,16,20,34,40,80,170
python3 -c "import sys; sys.exit(0 if int('${LC:-0}') > int('${LE:-0}') and float('$TC') >= float('$TE') * 0.99 else 1)" && CAPS=1,2,4,5,8,10,16,20,40,80
log "capture list $CAPS (l12e lanes ${LE:-?} tok/s $TE; l12ec lanes ${LC:-?} tok/s $TC)"
# v2 (2026-10-31 06:4x): split scan on the production path first. On the 4090 the production path at 26/22 sized 2.98 GiB of KV
# (20.22x; stage 0 binding: 15.94 vs 15.06 GiB in use) vs 4.04 GiB (24.18x) at 24/24: draftring takes the drafter's KV out of
# stage 1's pool, so the core's 26/22 over-shifts. Pre-stated rule: the split with more KV-sized lanes wins; a tie goes to the
# higher median decode-only tok/s; then the one-pass run at that split.
SPL=24,24; BL=0; BT=0
for sp in 24,24 25,23; do
  lb=l4n0$(echo $sp | tr -d ',')
  sleep 20
  CAPS=$CAPS bash $P4/try_l4prod_l4.sh $lb - - --drop-parts ctxfree --set ARC3_L4PORT_PP_SPLIT=$sp > $LD/try_$lb.out 2>&1
  conc=$(grep -h -o -E 'Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x' $LD/try_l4prod_l4_$lb.log | tail -n 1 | grep -o -E '[0-9.]+x$' | tr -d x)
  t=$(grep -h -o -E 'running +[0-9]+: +[0-9]+ windows, median +[0-9.]+ tok/s' $LD/try_l4prod_l4_$lb.log | tail -n 1 | grep -o -E 'median +[0-9.]+' | grep -o -E '[0-9.]+$')
  log "split $sp ($lb): concurrency ${conc:-?}x, decode-only median ${t:-?} tok/s; $(grep -E 'bench rc' $LD/try_l4prod_l4_$lb.log | cut -c22-220)"
  ln=$(python3 -c "print(int(float('${conc:-0}')))")
  if [ "$ln" -gt "$BL" ] || { [ "$ln" -eq "$BL" ] && python3 -c "import sys; sys.exit(0 if float('${t:-0}') > float('$BT') else 1)"; }; then
    SPL=$sp; BL=$ln; BT=${t:-0}
  fi
done
log "split $SPL chosen ($BL lanes, $BT tok/s)"
sleep 20
CAPS=$CAPS bash $P4/try_l4prod_l4.sh l4n1 gl4_l4n1 pass --drop-parts ctxfree --set ARC3_L4PORT_PP_SPLIT=$SPL > $LD/try_l4n1.out 2>&1
R=$(python3 $P4/check_l4prod.py $S2/instances/l4n1 "$(cat $S2/instances/l4n1/last_logdir 2>/dev/null)/server.log" $LD/try_l4prod_l4_l4n1.log --min-evicts 50 2>&1)
log "l4n1 pass line: $(echo "$R" | tr '\n' ' ' | cut -c1-1500)"
log "l4n1: $(grep -E 'KV:|bench rc|decode-only|gl4_l4n1 (exit|tail)' $LD/try_l4prod_l4_l4n1.log | cut -c22- | tr '\n' '|' | cut -c1-1500)"
log "chain n1 l4 ended"
