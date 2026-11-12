#!/usr/bin/env bash
# chain n9, 4090 (2026-10-31 21:1x, ARC2 - main; lever 4, the indexer dedupe v2): (1) idx_shard/t_merge_gpu.py on GPU 0: the Triton
# composite-key kernels against their torch twins and exact_topk (bitwise), and their timing against v1's sort merge (eager and in
# a CUDA graph; STOP unless ALL PASS); (2) pis2 = pis1 (chain n8: ph3's config + the shard, verify every 4th eager selection) with
# bootstrap_l4prod5 (v10: l4port_idxshard_v2.py, the sharded compaction and digest, merge keys) and ARC3_EVICTION_VERIFY_BYTES=1
# (production's before / after digests on every compacted owner, the sharded ones gathered over TP), the bench and the 8-task smoke.
# Read: KV tokens (pis1 1,589,812), bench tok/s against pis1 (1,426.0, 11.2 ms per step) and ph3 (1,528.3, 10.5 ms), verify counts
# (real = 0), the sharded compaction receipts, eviction errors (none expected), the smoke. Ends 'chain n9 4090 ended'.
set -uo pipefail
D=/mnt/shared/arc2/l4port; LD=$D/logs; L=$LD/chain_n9_4090.log; P4=$D/l4prod; S2=/workspace/kvring_dev/fin74/s2
NS=$D/sidepod/prune_sets_hope/eprune_freq_k128.json; RP=/workspace/arc3-inference/runtime_python.sh
log() { echo "[$(date -u +'%F %T')] $*" >> $L; }
stop() { log "STOP: $*"; log "chain n9 4090 ended"; exit 1; }
m() { md5sum < "$1" | cut -c1-8; }
log "start pod $(hostname) (pid $$); idxshard v2 $(m $P4/l4port_idxshard_v2.py), merge v2 $(m $D/idx_shard/merge_v2.py), t_merge_gpu $(m $D/idx_shard/t_merge_gpu.py), bootstrap v10 $(m $D/bootstrap_l4prod5/sitecustomize.py), install $(m $D/ple_hostpoll_tp/install.py), try $(m $P4/try_l4prod.sh)"
[ "$(m $P4/l4port_idxshard_v2.py)" = 1db8edad ] || stop "l4port_idxshard_v2.py is not 1db8edad"
[ "$(m $D/idx_shard/merge_v2.py)" = e4195a48 ] || stop "merge_v2.py is not e4195a48"
[ "$(m $D/idx_shard/t_merge_gpu.py)" = f0eccb8c ] || stop "t_merge_gpu.py is not f0eccb8c"
[ "$(m $D/bootstrap_l4prod5/sitecustomize.py)" = bfa8bde0 ] || stop "bootstrap_l4prod5 is not bfa8bde0"
[ "$(m $D/ple_hostpoll_tp/install.py)" = 7f856114 ] || stop "install.py is not 7f856114"
[ "$(m $P4/try_l4prod.sh)" = f6922c94 ] || stop "try_l4prod.sh is not f6922c94"
n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); [ "$n" = 0 ] || stop "$n GPU processes"
lb=pis2
[ -e $S2/instances/$lb ] && stop "instance $lb exists"
[ -e /mnt/shared/arc2/runs/${lb}smoke ] && stop "run ${lb}smoke exists"
[ -e $LD/t_merge_gpu_4090.out ] && stop "t_merge_gpu_4090.out exists"
log "GPU unit test (t_merge_gpu.py on GPU 0)"
CUDA_VISIBLE_DEVICES=0 timeout 1200 bash $RP $D/idx_shard/t_merge_gpu.py > $LD/t_merge_gpu_4090.out 2>&1; rc=$?
log "t_merge_gpu rc $rc: $(grep -E 'device|ALL PASS|FAIL|timing|Error' $LD/t_merge_gpu_4090.out | cut -c1-240 | tr '\n' '|')"
grep -q "ALL PASS" $LD/t_merge_gpu_4090.out || stop "t_merge_gpu did not pass (rc $rc)"
n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); [ "$n" = 0 ] || stop "$n GPU processes after the unit test"
log "$lb starting (pis1's config with the shard v2: merge keys, the sharded compaction; verify every 4th eager selection + eviction bytes; bench + the 8-task smoke)"
bash $P4/try_l4prod.sh $lb ${lb}smoke --drop-parts ctxfree --set ARC3_L4PORT_PP_SPLIT=26,22 --set ARC3_L4PORT_PP_BALANCE=1 \
  --set ARC3_STAGE2_BATCHED_TOKENS=2048 --set ARC3_S2_EPRUNE_FILE=$NS --set ARC3_PY_PREPEND=$D/bootstrap_l4prod5 \
  --set ARC3_PLE_HANDOFF_IMPL=hostpoll --set ARC3_L4PORT_PLE_HPTP=1 --set ARC3_L4PORT_EVPP=1 \
  --set ARC3_L4PORT_IDX_SHARD=1 --set ARC3_L4PORT_IDX_SHARD_VERIFY=4 --set ARC3_L4PORT_IDX_SHARD_MERGE=keys \
  --set ARC3_EVICTION_VERIFY_BYTES=1 > $LD/try_$lb.out 2>&1
SL=$(cat $S2/instances/$lb/last_logdir 2>/dev/null)/server.log
log "$lb try log: $(grep -E 'server wrapper exit|errors:|KV:|bench at|bench rc|decode-only|smoke|exit [0-9]|server stop' $LD/try_l4prod_$lb.log | cut -c22-300 | tr '\n' '|' | cut -c1-2500)"
log "$lb decode-only line: $(grep -h -A1 'decode-only windows' $LD/try_l4prod_$lb.log | grep -o -E 'running +[0-9]+: [^|]{0,120}' | head -n 2 | tr '\n' '|')"
log "$lb shard receipts: $(grep -h -o -E '\[l4port\] idxshard: (compressed-key spec|sharded indexer forward|sharded compaction)[^|]{0,200}' $SL 2>/dev/null | sort | uniq -c | sort -rn | head -n 8 | tr '\n' '|' | cut -c1-2000)"
log "$lb last verify lines: $(grep -h -o -E '\[l4port\] idxshard: verify rank [0-9]: [^|]{0,160}' $SL 2>/dev/null | tail -n 4 | tr '\n' '|')"
log "$lb salt: $(grep -h -o -E 'VLLM_BUILD_URL[^ ]{0,120}' $SL 2>/dev/null | sort -u | head -n 3 | tr '\n' '|')"
log "$lb eviction lines: $(grep -h -c -E 'arc3_evict|EngineEviction|compaction' $SL 2>/dev/null) total; errors: $(grep -h -E 'EngineEvictionError|retained cache bytes changed|idxshard: compressed cache|unsupported sharded|Traceback' $SL 2>/dev/null | sort | uniq -c | sort -rn | head -n 6 | cut -c1-300 | tr '\n' '|')"
log "$lb smoke evictions: $(python3 - /mnt/shared/arc2/runs/${lb}smoke/results.jsonl <<'PY'
import json, sys
n = e = t = 0
try:
    for line in open(sys.argv[1]):
        r = json.loads(line); n += 1
        v = int(r.get("evicted_tokens") or 0); e += v > 0; t += v
except OSError as x:
    print("no results (%s)" % x.__class__.__name__); sys.exit()
print("%d attempts, %d with evictions, %d evicted tokens" % (n, e, t))
PY
)"
log "chain n9 4090 ended"
