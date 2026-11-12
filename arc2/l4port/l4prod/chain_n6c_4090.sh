#!/usr/bin/env bash
# chain n6c, 4090 (2026-10-31 19:1x, ARC2 - main): ph2 found the TP broadcast's payload bug (ForkingPickler.dumps returns a memoryview;
# connector 2e84b011 broadcasts bytes) and the second CUDA touch in the offload's structure build (the Marlin workspace's SM-count
# query): worker_patch_tp 6068c033 spawns the offload with only the last GPU visible (its context leaves stage 0) and leaves
# registration if its worker exits. One arm, ph3 (no no-GPU gate). chain n6b's header follows.
# chain n6b, 4090 (2026-10-31 19:0x, ARC2 - main; chain n6's ph1 / ph1b stopped at the eviction runner's TP1/PP1 guard: the 4090's
# try_l4prod.sh does not pass ARC3_L4PORT_EVPP=1, added here; ph1's no-GPU spawn caught a CUDA init in the offload process's
# model-structure build (the GDN gated norm, device=current_device()): worker_patch_tp e22e17cb builds it with current_device() =
# meta. Labels ph2 / ph2b. Original header follows.
# chain n6, 4090 (2026-10-31 19:0x, ARC2 - main; the user: "Let's do it" = lever 1, the PLE handoff without the PLE offload
# process's CUDA contexts; "Go ahead in that order"; the pods only for the listed items; the 4090 is sm_89 like the L4). Steps:
#   1. build production's polling kernel source for sm_89 into l4port/ple_hostpoll_tp/build_sm89 (build_sm89.py, the runtime
#      python, TORCH_CUDA_ARCH_LIST=8.9) and run its poll / timeout self-test;
#   2. ph1: one try_l4prod.sh launch (the 4090 sizing under the L4 ballast; decode bench + the 8-task smoke) with bootstrap_l4prod2
#      (v7), ARC3_PLE_HANDOFF_IMPL=hostpoll, ARC3_L4PORT_PLE_HPTP=1 and the verify mode (ARC3_L4PORT_PLE_HP_VERIFY=1), plus the
#      adopted parts (26/22, the balancer, batched tokens 2048, the k128 set). While it runs, every 60 s: the GPU processes
#      (pid, MiB, comm); after it: the host-poll receipts (offload: handled, torn reads, skipped seq, l4port_tp; workers: the
#      connector source sha; verify: checked / mismatched / skipped per rank).
# Pass (functional, before any L4 throughput arm): healthy, bench rc 0, the smoke done, checked > 0 and 0 mismatched on both
# stage-0 TP ranks, no poll timeout, no GPU process besides the four workers and the ballast. Ends 'chain n6 4090 ended'.
set -uo pipefail
D=/mnt/shared/arc2/l4port; LD=$D/logs; L=$LD/chain_n6c_4090.log; P4=$D/l4prod; T=$D/ple_hostpoll_tp; S2=/workspace/kvring_dev/fin74/s2
NS=$D/sidepod/prune_sets_hope/eprune_freq_k128.json; HR=$S2/ple-hostpoll-startup
log() { echo "[$(date -u +'%F %T')] $*" >> $L; }
stop() { log "STOP: $*"; log "chain n6c 4090 ended"; exit 1; }
m() { md5sum < "$1" | cut -c1-8; }
log "start pod $(hostname) boot $(cat /proc/sys/kernel/random/boot_id) (pid $$); connector $(m $T/connector.py), worker_patch_tp $(m $T/worker_patch_tp.py), install $(m $T/install.py), build $(m $T/build_sm89.py), bootstrap v7 $(m $D/bootstrap_l4prod2/sitecustomize.py), try $(m $P4/try_l4prod.sh), launcher $(m $S2/k_launch_scaling_l4.py)"
[ "$(m $T/connector.py)" = 2e84b011 ] || stop "connector.py is not 2e84b011"
[ "$(m $T/worker_patch_tp.py)" = 6068c033 ] || stop "worker_patch_tp.py is not 6068c033"
[ "$(m $T/install.py)" = 7ae0f9d7 ] || stop "install.py is not 7ae0f9d7"
[ "$(sha256sum < $T/build_sm89/arc3_ple_hostpoll_v1.so | cut -c1-16)" = d9ae205cef27481d ] || stop "the sm_89 build is not d9ae205c"
[ "$(m $D/bootstrap_l4prod2/sitecustomize.py)" = 651f331e ] || stop "bootstrap_l4prod2 is not 651f331e"
[ "$(m $P4/try_l4prod.sh)" = f6922c94 ] || stop "try_l4prod.sh is not f6922c94"
n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); [ "$n" = 0 ] || stop "$n GPU processes"
for lb in ph3; do [ -e $S2/instances/$lb ] && stop "instance $lb exists"; [ -e /mnt/shared/arc2/runs/${lb}smoke ] && stop "run ${lb}smoke exists"; done
# 1. the sm_89 build
if [ ! -f $T/build_sm89/arc3_ple_hostpoll_v1.so ]; then
  bash /workspace/arc3-inference/runtime_python.sh $T/build_sm89.py $T/build_sm89 > $T/build_sm89.log 2>&1
  log "build rc $?: $(tail -n 1 $T/build_sm89.log | cut -c1-300)"
fi
[ -f $T/build_sm89/arc3_ple_hostpoll_v1.so ] || { log "build log tail: $(tail -n 8 $T/build_sm89.log | tr '\n' '|' | cut -c1-1500)"; stop "no sm_89 build"; }
log "build: $(ls -la $T/build_sm89/ | awk 'NR>1 {print $5, $9}' | tr '\n' ' '); so sha256 $(sha256sum $T/build_sm89/arc3_ple_hostpoll_v1.so | cut -c1-16); elf $(/workspace/arc3-inference/runtime/base/usr/local/cuda-13.0/bin/cuobjdump --list-elf $T/build_sm89/arc3_ple_hostpoll_v1.so 2>&1 | head -n 2 | tr '\n' ' ')"
# 2. ph1 (+ the reviewer's gate: the offload process spawned with no visible GPU); ph1b without that gate if ph1 does not come up
mkdir -p $HR
run_ph() {  # $1 label, then extra --set args
  local lb=$1; shift
  [ -e $S2/instances/$lb ] && { log "$lb skipped: instance exists"; return 1; }
  touch $LD/${lb}_t0.ref
  local GP=$LD/gpuprocs_$lb.txt GS=$LD/gpuprocs_$lb.stop
  ( while [ ! -e $GS ]; do
      echo "@ $(date -u +%T)"
      nvidia-smi --query-compute-apps=pid,used_memory,gpu_bus_id --format=csv,noheader 2>/dev/null | while IFS=, read -r p mem bus; do
        p=$(echo $p | tr -d ' '); echo "$p,$mem,$bus,$(cat /proc/$p/comm 2>/dev/null),$(tr '\0' ' ' < /proc/$p/cmdline 2>/dev/null | cut -c1-90)"
      done
      sleep 60
    done ) > $GP 2>&1 &
  log "$lb starting (hostpoll-tp + verify $*, the 4090 sizing + ballast, bench + the 8-task smoke ${lb}smoke)"
  bash $P4/try_l4prod.sh $lb ${lb}smoke --drop-parts ctxfree --set ARC3_L4PORT_PP_SPLIT=26,22 --set ARC3_L4PORT_PP_BALANCE=1 \
    --set ARC3_STAGE2_BATCHED_TOKENS=2048 --set ARC3_S2_EPRUNE_FILE=$NS --set ARC3_PY_PREPEND=$D/bootstrap_l4prod2 \
    --set ARC3_PLE_HANDOFF_IMPL=hostpoll --set ARC3_L4PORT_PLE_HPTP=1 --set ARC3_L4PORT_PLE_HP_VERIFY=1 --set ARC3_L4PORT_EVPP=1 "$@" > $LD/try_$lb.out 2>&1
  touch $GS; sleep 5
  log "$lb try log: $(grep -E 'server wrapper exit|errors:|KV:|bench at|bench rc|smoke|exit [0-9]|server stop' $LD/try_l4prod_$lb.log | cut -c22-260 | tr '\n' '|' | cut -c1-2500)"
  log "$lb GPU processes (pid: max MiB, comm, cmdline): $(python3 - $GP <<'PY2'
import sys
mx, info = {}, {}
for line in open(sys.argv[1], errors="replace"):
    parts = line.rstrip("\n").split(",", 4)
    if len(parts) < 5 or not parts[0].isdigit():
        continue
    pid, mem = parts[0], int("".join(c for c in parts[1] if c.isdigit()) or 0)
    mx[pid] = max(mx.get(pid, 0), mem); info[pid] = (parts[3], parts[4][:70])
print("; ".join("%s: %d MiB, %s, %s" % (p, mx[p], info[p][0], info[p][1]) for p in sorted(mx)) or "-")
PY2
)"
  log "$lb host-poll receipts: $(python3 - $HR $LD/${lb}_t0.ref <<'PY2'
import glob, json, os, sys
t0 = os.path.getmtime(sys.argv[2])
out = []
for p in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    if os.path.getmtime(p) < t0:
        continue
    try:
        d = json.load(open(p))
    except Exception as e:
        out.append("%s unreadable %r" % (os.path.basename(p), e)); continue
    keys = ("role", "tp_rank", "forwards", "checked", "mismatched", "skipped", "handled", "batches", "torn_reads", "skipped_seq",
            "fasthash", "l4port_tp", "installed", "source_sha256", "errors")
    out.append("%s %s" % (os.path.basename(p), json.dumps({k: d[k] for k in keys if k in d})[:400]))
print(" || ".join(out) or "none")
PY2
)"
  local SL=$(cat $S2/instances/$lb/last_logdir 2>/dev/null)/server.log
  log "$lb server log hostpoll lines: $(grep -h -o -E 'PleOffload\(hostpoll[^)]*\)[^|]{0,160}|host-poll[^|]{0,160}' $SL 2>/dev/null | sort | uniq -c | head -n 12 | tr '\n' '|' | cut -c1-2000)"
  log "$lb offload failure lines: $(grep -h -E 'PleOffloadWorker|PLE offload' $SL 2>/dev/null | grep -i -E 'error|fail|Traceback|CUDA' | head -n 5 | cut -c1-300 | tr '\n' '|')"
  grep -q -E 'server wrapper exit 0' $LD/try_l4prod_$lb.log
}
run_ph ph3 || log "ph3 did not come up"
log "chain n6c 4090 ended"
