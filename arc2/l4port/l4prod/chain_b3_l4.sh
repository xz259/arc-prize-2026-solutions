#!/usr/bin/env bash
# chain b3, L4 pod B (2026-10-31 18:3x, ARC2 - main; the user: "Go ahead in that order", lever 2 = the capture-list trim; "only use
# [the pods] for the listed items"). With the microbatch balancer a step schedules at most ceil(lanes / 2) requests (15 at 29
# lanes), so the 34-request graphs (34 drafter tokens, 170 verify tokens) are never replayed at steady state; vLLM's graph pool
# is sized by the largest graph (pbs30: 170 -> 150 tokens freed ~50 MiB per worker). Under explicit KV sizing any freed memory
# converts to KV. Each arm is one try_l4prod_l4c.sh launch with the decode bench (mode '-', nvidia-smi every 10 s), at the
# adopted config: 26/22, the balancer, batched tokens 2048, the train120 + eval120 k128 set, MAX_SEQS 34:
#   pbc7  the control: the full capture list (1,...,34,40,80,170), KV 4.33 GiB (pbx26 + the balancer; pbx26 ran before the
#         balancer was adopted, so its peaks and tok/s are not this arm's twin);
#   pbt1  the trimmed list 1,2,4,5,8,10,16,20,40,80 (largest graph 16 requests = 80 verify tokens), KV 4.33 GiB;
#   pbt2  the trimmed list at KV 4.33 + dKV, dKV = min over GPUs (pbc7's binding peak - pbt1's peak) / 1024, floored to 0.01 GiB:
#         the trim's freed memory moved into KV, keeping the control's bench peak (and so the real-traffic margin l4v2a saw);
#         skipped when dKV < 0.02 GiB or an earlier arm failed.
# Read: per-worker CUDAGraph GiB, peaks, lanes (KV tokens / 60,240), steady decode-only tok/s, preemptions. Pass for pbt2: every
# peak <= pbc7's binding peak + 10 MiB, 0 preemptions, tok/s within noise of pbc7 at more lanes. Ends 'chain b3 l4 ended'.
set -uo pipefail
D=/mnt/shared/arc2/l4port; P4=$D/l4prod; LD=$D/logs_l4; L=$LD/chain_b3_l4.log; S2=/workspace/kvring_dev/fin74/s2
NS=$D/sidepod/prune_sets_hope/eprune_freq_k128.json
log() { echo "[$(date -u +'%F %T')] $*" >> $L; }
stop() { log "STOP: $*"; log "chain b3 l4 ended"; exit 1; }
idle() { n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); [ "$n" = 0 ] || stop "$n GPU processes before $1"; }
log "start pod $(hostname) boot $(cat /proc/sys/kernel/random/boot_id) (pid $$); try_l4c $(md5sum < $P4/try_l4prod_l4c.sh | cut -c1-8), ppbalance $(md5sum < $P4/l4port_ppbalance.py | cut -c1-8), server wrapper $(md5sum < $P4/arc2_server_l4.py | cut -c1-8), set $(md5sum < $NS | cut -c1-8)"
[ "$(md5sum < $P4/try_l4prod_l4c.sh | cut -c1-8)" = 0b0d483c ] || stop "try_l4prod_l4c.sh is not 0b0d483c"
[ "$(md5sum < $P4/l4port_ppbalance.py | cut -c1-8)" = 54f7802a ] || stop "l4port_ppbalance.py is not 54f7802a"
for lb in pbc7 pbt1 pbt2; do [ -e $S2/instances/$lb ] && stop "instance $lb exists"; [ -e $LD/try_l4prod_l4_$lb.log ] && stop "log of $lb exists"; done
FULL=1,2,4,5,8,10,16,20,34,40,80,170; TRIM=1,2,4,5,8,10,16,20,40,80
A="--drop-parts ctxfree --set ARC3_L4PORT_PP_SPLIT=26,22 --set ARC3_L4PORT_PP_BALANCE=1 --set ARC3_STAGE2_BATCHED_TOKENS=2048 --set ARC3_S2_EPRUNE_FILE=$NS"
read_mem() {  # $1 label: KV tokens, concurrency, per-worker memory from the server log (the bench branch logs no usage lines)
  python3 - "$(cat $S2/instances/$1/last_logdir 2>/dev/null)/server.log" <<'PY'
import re, sys
try:
    t = open(sys.argv[1], errors="replace").read()
except OSError:
    t = ""
kv = re.findall(r"GPU KV cache size: ([0-9,]+) tokens", t)
cc = re.findall(r"Maximum concurrency for [0-9,]+ tokens per request: ([0-9.]+)x", t)
w = {}
W = r"\((Worker_PP\d_TP\d)[^)]*\)[^\n]*?"
for tag, a in re.findall(W + r"Model loading took ([0-9.]+) GiB", t):
    w.setdefault(tag, {})["load"] = a
for tag, a, b, c in re.findall(W + r"Actual usage is ([0-9.]+) GiB for consumed memory \(weights \+ non-torch\), "
                               r"([0-9.]+) GiB for peak activation, and ([0-9.]+) GiB for CUDAGraph", t):
    w.setdefault(tag, {}).update(cons=a, peak=b, graph=c)
for tag, a in re.findall(W + r"Graph capturing finished in [0-9]+ secs, took ([0-9.]+) GiB", t):
    w.setdefault(tag, {})["cap"] = a
print("KV %s tokens, %sx; %s" % (kv[-1] if kv else "?", cc[-1] if cc else "?", "; ".join(
    "%s load %s cons %s peak %s graph %s capture %s" % (k[7:], v.get("load", "?"), v.get("cons", "?"), v.get("peak", "?"),
                                                       v.get("graph", "?"), v.get("cap", "?")) for k, v in sorted(w.items()))))
PY
}
peaks() {  # $1 label: the max MiB per GPU from the sampler file
  python3 - $LD/smi_$1.csv <<'PY'
import sys
peak = {}
try:
    for line in open(sys.argv[1]):
        try:
            i, m = (int(x) for x in line.split(","))
        except ValueError:
            continue
        peak[i] = max(peak.get(i, 0), m)
except OSError:
    pass
print(" ".join("%d:%d" % kv for kv in sorted(peak.items())) or "-")
PY
}
tps() {  # $1 label: the steady decode-only median of the group with the most windows
  python3 - $LD/try_l4prod_l4_$1.log <<'PY'
import re, sys
best = (0, 0.0)
for m in re.finditer(r"running\s+(\d+):\s+(\d+) windows, median\s+([0-9.]+) tok/s", open(sys.argv[1]).read()):
    best = max(best, (int(m.group(2)), float(m.group(3))))
print(best[1])
PY
}
benched() { grep -q -E 'bench rc 0' $LD/try_l4prod_l4_$1.log 2>/dev/null; }
run() {  # $1 label, $2 capture list, then --set / --drop-parts args
  local lb=$1 caps=$2; shift 2
  local smi=$LD/smi_$lb.csv stopf=$LD/smi_$lb.stop
  idle "$lb"
  sleep 20
  ( while [ ! -e $stopf ]; do nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits >> $smi 2>/dev/null; sleep 10; done ) &
  env NOBENCH=0 LANES_CAP=34 CAPS=$caps bash $P4/try_l4prod_l4c.sh $lb - - "$@" > $LD/try_$lb.out 2>&1
  touch $stopf; sleep 12
  log "$lb (caps $caps; $*): $(read_mem $lb); peaks MiB $(peaks $lb); $(grep -E 'bench at|bench rc|server wrapper exit' $LD/try_l4prod_l4_$lb.log | cut -c22-240 | tr '\n' ' ')"
  log "$lb steady decode-only median $(tps $lb) tok/s; preemptions $(grep -h -o -E 'num_preemptions_total\{[^}]*\} [0-9.e+]+' $LD/try_l4prod_l4_$lb.log | tail -n 1 | awk '{print $NF}')"
}
run pbc7 $FULL $A --set ARC3_STAGE4_KV_GIB=4.33
benched pbc7 || stop "the control pbc7 did not bench"
run pbt1 $TRIM $A --set ARC3_STAGE4_KV_GIB=4.33
benched pbt1 || stop "pbt1 (the trimmed list) did not bench"
DKV=$(python3 - $LD/smi_pbc7.csv $LD/smi_pbt1.csv <<'PY'
import math, sys
def peaks(p):
    pk = {}
    for line in open(p):
        try:
            i, m = (int(x) for x in line.split(","))
        except ValueError:
            continue
        pk[i] = max(pk.get(i, 0), m)
    return pk
c, t = peaks(sys.argv[1]), peaks(sys.argv[2])
target = max(c.values())
d = min(target - t[i] for i in t)
print("%.2f" % (math.floor(max(d, 0) / 1024.0 * 100) / 100.0))
PY
)
log "dKV $DKV GiB (pbc7's binding peak minus pbt1's peaks, the smallest over GPUs)"
python3 -c "import sys; sys.exit(0 if float('$DKV') >= 0.02 else 1)" || stop "dKV $DKV GiB < 0.02: the trim frees too little to move KV; pbt2 skipped"
KV2=$(python3 -c "print('%.2f' % (4.33 + float('$DKV')))")
run pbt2 $TRIM $A --set ARC3_STAGE4_KV_GIB=$KV2
log "summary: pbc7 $(grep -h -o -E '[0-9.]+x' <<< "$(read_mem pbc7)" | head -n 1) $(tps pbc7) tok/s peaks $(peaks pbc7) | pbt1 $(tps pbt1) tok/s peaks $(peaks pbt1) | pbt2 (KV $KV2) $(grep -h -o -E '[0-9.]+x' <<< "$(read_mem pbt2)" | head -n 1) $(tps pbt2) tok/s peaks $(peaks pbt2)"
log "chain b3 l4 ended"
