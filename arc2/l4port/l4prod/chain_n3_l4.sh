#!/usr/bin/env bash
# chain n3, L4 (2026-10-31 07:0x, ARC2 - main; replaces chain_n1_l4.sh v2, which idled the L4 until the 4090 smoke gate): the
# production-path split scan runs now; only the long pass waits for the 4090's 'smoke gate PASS'. Capture list: chain n1's rule
# (l12ec's shorter list only if it gave more lanes than l12e at >= 99% of its tok/s; it did not, so the full list).
# Split scan (pre-stated rule): 24/24 first, with nvidia-smi sampled every 10 s; the stage with the higher peak memory in use binds
# (all workers get the same number of KV blocks), so the next split moves one layer away from it (stage 0 binds -> 23/25, stage 1
# binds -> 25/23); a third split continues in the same direction only if the second gave more lanes. The split with more KV-sized
# lanes (floor of vLLM's maximum concurrency) wins; a tie goes to the higher median decode-only tok/s. Then the one-pass run
# (try_l4prod_l4.sh l4n3 gl4_l4n3 pass) at the winner, gated on the 4090 smoke; check_l4prod.py with >= 50 evictions.
# Ends 'chain n3 l4 ended'.
set -uo pipefail
D=/mnt/shared/arc2/l4port; P4=$D/l4prod; LD=$D/logs_l4; L=$LD/chain_n3_l4.log; S2=/workspace/kvring_dev/fin74/s2
log() { echo "[$(date -u +'%F %T')] $*" >> $L; }
stop() { log "STOP: $*"; log "chain n3 l4 ended"; exit 1; }
log "start pod $(hostname) (pid $$); try_l4 $(md5sum < $P4/try_l4prod_l4.sh | cut -c1-8), check $(md5sum < $P4/check_l4prod.py | cut -c1-8)"
n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); [ "$n" = 0 ] || stop "$n GPU processes"
CAPS=1,2,4,5,8,10,16,20,34,40,80,170
declare -A CONC TPS
run_split() {  # $1 = a,b ; sets CONC[$1], TPS[$1], logs the per-GPU peak memory in use
  local sp=$1 lb=l4s$(echo $1 | tr -d ',') smi=$LD/smi_l4s$(echo $1 | tr -d ',').csv stopf=$LD/smi_l4s$(echo $1 | tr -d ',').stop
  ( while [ ! -e $stopf ]; do nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits >> $smi 2>/dev/null; sleep 10; done ) &
  CAPS=$CAPS bash $P4/try_l4prod_l4.sh $lb - - --drop-parts ctxfree --set ARC3_L4PORT_PP_SPLIT=$sp > $LD/try_$lb.out 2>&1
  touch $stopf; sleep 12
  local tl=$LD/try_l4prod_l4_$lb.log
  CONC[$sp]=$(grep -h -o -E 'Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x' $tl | tail -n 1 | grep -o -E '[0-9.]+x$' | tr -d x)
  TPS[$sp]=$(grep -h -o -E 'running +[0-9]+: +[0-9]+ windows, median +[0-9.]+ tok/s' $tl | tail -n 1 | grep -o -E 'median +[0-9.]+' | grep -o -E '[0-9.]+$')
  PEAK=$(python3 - $smi <<'PY'
import sys
peak = {}
for line in open(sys.argv[1]):
    try:
        i, m = (int(x) for x in line.split(","))
    except ValueError:
        continue
    peak[i] = max(peak.get(i, 0), m)
print(" ".join("%d:%d" % kv for kv in sorted(peak.items())))
PY
)
  log "split $sp ($lb): concurrency ${CONC[$sp]:-?}x, decode-only median ${TPS[$sp]:-?} tok/s; peak MiB in use $PEAK; $(grep -E 'bench rc|server wrapper exit' $tl | cut -c22-260 | tr '\n' ' ')"
}
lanes_of() { python3 -c "print(int(float('${CONC[$1]:-0}')))"; }
better() {  # $1 beats $2 ?
  local a b; a=$(lanes_of $1); b=$(lanes_of $2)
  [ "$a" -gt "$b" ] && return 0
  [ "$a" -eq "$b" ] && python3 -c "import sys; sys.exit(0 if float('${TPS[$1]:-0}') > float('${TPS[$2]:-0}') else 1)"
}
run_split 24,24
BIND=$(echo "$PEAK" | python3 -c "
import sys
p = dict(tuple(int(x) for x in kv.split(':')) for kv in sys.stdin.read().split())
s0 = max(p.get(0, 0), p.get(1, 0)); s1 = max(p.get(2, 0), p.get(3, 0))
print(0 if s0 >= s1 else 1)")
[ "$(lanes_of 24,24)" -gt 0 ] || stop "24/24 gave no KV sizing (see try_l4s2424.out)"
if [ "$BIND" = 0 ]; then NEXT=(23,25 22,26); else NEXT=(25,23 26,22); fi
log "24/24: stage $BIND binds; next ${NEXT[0]} (then ${NEXT[1]} if it gains lanes)"
BEST=24,24
sleep 20; run_split ${NEXT[0]}
if better ${NEXT[0]} $BEST; then
  BEST=${NEXT[0]}
  if [ "$(lanes_of ${NEXT[0]})" -gt "$(lanes_of 24,24)" ]; then
    sleep 20; run_split ${NEXT[1]}
    better ${NEXT[1]} $BEST && BEST=${NEXT[1]}
  fi
fi
log "split $BEST chosen ($(lanes_of $BEST) lanes, ${TPS[$BEST]:-?} tok/s)"
DL=$(( $(date +%s) + 4 * 3600 ))
until grep -q -E '^\[[0-9: -]+\] (smoke gate PASS|chain n1 4090 ended)$' $D/logs/chain_n1_4090.log 2>/dev/null; do
  [ "$(date +%s)" -ge "$DL" ] && stop "OVERDUE waiting for the 4090 smoke gate"
  sleep 30
done
grep -q -E '^\[[0-9: -]+\] smoke gate PASS$' $D/logs/chain_n1_4090.log || stop "the 4090 smoke gate did not pass"
log "4090 smoke gate PASS seen; l4n3 starting at split $BEST"
sleep 20
CAPS=$CAPS bash $P4/try_l4prod_l4.sh l4n3 gl4_l4n3 pass --drop-parts ctxfree --set ARC3_L4PORT_PP_SPLIT=$BEST > $LD/try_l4n3.out 2>&1
R=$(python3 $P4/check_l4prod.py $S2/instances/l4n3 "$(cat $S2/instances/l4n3/last_logdir 2>/dev/null)/server.log" $LD/try_l4prod_l4_l4n3.log --min-evicts 50 2>&1)
log "l4n3 pass line: $(echo "$R" | tr '\n' ' ' | cut -c1-1500)"
log "l4n3: $(grep -E 'KV:|bench rc|decode-only|gl4_l4n3 (exit|tail)' $LD/try_l4prod_l4_l4n3.log | cut -c22- | tr '\n' '|' | cut -c1-1500)"
log "chain n3 l4 ended"
