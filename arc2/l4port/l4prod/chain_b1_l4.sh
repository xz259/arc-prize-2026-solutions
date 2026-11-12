#!/usr/bin/env bash
# chain b1, the second 4x L4 pod (2026-10-31 13:2x, ARC2 - main; the user: "increase lanes as many as possible, increase throughput";
# the user reserved a second L4 pod). Runs after podb_setup.sh (the clone of pod A's serving trees). Each step is one
# try_l4prod_l4c.sh launch (mode '-', nvidia-smi sampled every 10 s), the new expert set, batched tokens 2048:
#   pbc6   the control: E6's config (25/23, explicit KV 4.25 GiB) with the bench, against pod A's l4e6kv (28.83x, 788.3 tok/s,
#          peaks 21,480 / 21,446 / 22,133 / 22,131 MiB): same KV tokens expected, tok/s and peaks close;
#   pbx26  26/22 + explicit KV 4.33 GiB (E6's rule on E5's bench; 29 lanes predicted) with the bench; passes with every peak
#          <= 22,200 MiB and 0 preemptions;
#   the memory bisection at E4's util-sized config (25/23, util 0.89), memory only (NOBENCH=1): pbm0 (nothing more dropped),
#   pbm1 kvring, pbm3 draftring, pbm4 persist + prestage + resumefast + overlap + rollback, pbm5 fqfast + hollowgroup + wy +
#   marlindet + pleconv + agestat, pbs30 MAX_SEQS 30 (capture list ...,30,40,80,150). A start that fails is logged and the chain
#   goes on. Not bisected: rope2m (the profile already runs ROPE_ROWS 1,048,576 = the stock M-RoPE table, all rows FP64: no
#   extension to save; the user, 14:2x: no extension needed, keep FP64), mmstrip / mmonly (memory savers).
#   pbd6 (after the control): memdiag at E6's config (ARC3_S2_MEMDIAG=2000000), a short bench (P 44000, D 2000), memdiag's
#   per-worker report: the allocation sites of the run-time memory above weights + KV + graphs (attribution before bisection).
# Each line: KV tokens, concurrency and, per worker (PP / TP), the model load, consumed, peak activation and graphs; the summary:
# each memory run's KV tokens against pbm0, in lanes of 60,240 tokens. Ends 'chain b1 l4 ended'.
set -uo pipefail
D=/mnt/shared/arc2/l4port; P4=$D/l4prod; LD=$D/logs_l4; L=$LD/chain_b1_l4.log; S2=/workspace/kvring_dev/fin74/s2
NS=$D/sidepod/prune_sets_hope/eprune_freq_k128.json
log() { echo "[$(date -u +'%F %T')] $*" >> $L; }
stop() { log "STOP: $*"; log "chain b1 l4 ended"; exit 1; }
idle() { n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); [ "$n" = 0 ] || stop "$n GPU processes before $1"; }
log "start pod $(hostname) boot $(cat /proc/sys/kernel/random/boot_id) (pid $$); try_l4c $(md5sum < $P4/try_l4prod_l4c.sh | cut -c1-8), set $(md5sum < $NS | cut -c1-8); GPUs $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader | sort | uniq -c | tr '\n' ' ')"
grep -q 'POD B SETUP DONE' $D/clone/podb_setup_$(hostname).log 2>/dev/null || stop "podb_setup has not passed on $(hostname)"
CAPS=1,2,4,5,8,10,16,20,34,40,80,170
BT="--set ARC3_STAGE2_BATCHED_TOKENS=2048 --set ARC3_S2_EPRUNE_FILE=$NS"
read_mem() {  # $1 label: KV tokens, concurrency, per-worker memory from the try log
  python3 - $LD/try_l4prod_l4_$1.log <<'PY'
import re, sys
t = open(sys.argv[1], errors="replace").read()
kv = re.findall(r"GPU KV cache size: ([0-9,]+) tokens", t)
cc = re.findall(r"Maximum concurrency for [0-9,]+ tokens per request: ([0-9.]+)x", t)
w = {}
for tag, a in re.findall(r"(Worker_PP\d_TP\d) Model loading took ([0-9.]+) GiB", t):
    w.setdefault(tag, {})["load"] = a
for tag, a, b, c in re.findall(r"(Worker_PP\d_TP\d) Actual usage is ([0-9.]+) GiB for consumed memory \(weights \+ non-torch\), "
                               r"([0-9.]+) GiB for peak activation, and ([0-9.]+) GiB for CUDAGraph", t):
    w.setdefault(tag, {}).update(cons=a, peak=b, graph=c)
print("KV %s tokens, %sx; %s" % (kv[-1] if kv else "?", cc[-1] if cc else "?", "; ".join(
    "%s load %s cons %s peak %s graph %s" % (k[7:], v.get("load", "?"), v.get("cons", "?"), v.get("peak", "?"), v.get("graph", "?"))
    for k, v in sorted(w.items()))))
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
run() {  # $1 label, $2 1 = memory only, $3 capture list, then --set / --drop-parts args
  local lb=$1 nb=$2 caps=$3; shift 3
  local smi=$LD/smi_$lb.csv stopf=$LD/smi_$lb.stop
  idle "$lb"
  sleep 20
  ( while [ ! -e $stopf ]; do nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits >> $smi 2>/dev/null; sleep 10; done ) &
  env ${RUNENV:-} NOBENCH=$nb LANES_CAP=34 CAPS=$caps bash $P4/try_l4prod_l4c.sh $lb - - "$@" > $LD/try_$lb.out 2>&1
  touch $stopf; sleep 12
  log "$lb ($*): $(read_mem $lb); peaks MiB $(peaks $lb); $(grep -E 'bench rc|server wrapper exit|memreport' $LD/try_l4prod_l4_$lb.log | cut -c22-240 | tr '\n' ' ')"
  [ "$nb" = 1 ] || log "$lb steady decode-only median $(tps $lb) tok/s; $(grep -h -o -E 'preemptions [0-9.]+' $LD/try_l4prod_l4_$lb.log | tail -n 1)"
}
E4="--drop-parts ctxfree --set ARC3_L4PORT_PP_SPLIT=25,23 $BT"
run pbc6 0 $CAPS --drop-parts ctxfree --set ARC3_L4PORT_PP_SPLIT=25,23 --set ARC3_STAGE4_KV_GIB=4.25 $BT
RUNENV="MEMREPORT=1 BENCH_D=2000" run pbd6 0 $CAPS --drop-parts ctxfree --set ARC3_L4PORT_PP_SPLIT=25,23 --set ARC3_STAGE4_KV_GIB=4.25 $BT --set ARC3_S2_MEMDIAG=2000000
run pbx26 0 $CAPS --drop-parts ctxfree --set ARC3_L4PORT_PP_SPLIT=26,22 --set ARC3_STAGE4_KV_GIB=4.33 $BT
run pbm0 1 $CAPS $E4
run pbm1 1 $CAPS --drop-parts ctxfree,kvring --set ARC3_L4PORT_PP_SPLIT=25,23 $BT
run pbm3 1 $CAPS --drop-parts ctxfree,draftring --set ARC3_L4PORT_PP_SPLIT=25,23 $BT
run pbm4 1 $CAPS --drop-parts ctxfree,persist,prestage,resumefast,overlap,rollback --set ARC3_L4PORT_PP_SPLIT=25,23 $BT
run pbm5 1 $CAPS --drop-parts ctxfree,fqfast,hollowgroup,wy,marlindet,pleconv,agestat --set ARC3_L4PORT_PP_SPLIT=25,23 $BT
run pbs30 1 1,2,4,5,8,10,16,20,30,40,80,150 $E4 --set ARC3_STAGE4_MAX_SEQS=30
log "summary (KV tokens against pbm0, lanes of 60,240): $(python3 - $LD <<'PY'
import re, sys
def kv(lb):
    try:
        m = re.findall(r"GPU KV cache size: ([0-9,]+) tokens", open("%s/try_l4prod_l4_%s.log" % (sys.argv[1], lb), errors="replace").read())
        return int(m[-1].replace(",", "")) if m else None
    except OSError:
        return None
b = kv("pbm0")
out = []
for lb in ("pbm1", "pbm3", "pbm4", "pbm5", "pbs30"):
    k = kv(lb)
    out.append("%s %s" % (lb, "start failed" if k is None or b is None else "%+d tokens (%+.2f lanes)" % (k - b, (k - b) / 60240.0)))
print("pbm0 %s tokens; " % b + "; ".join(out))
PY
)"
log "chain b1 l4 ended"
