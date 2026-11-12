#!/usr/bin/env bash
# chain n7, 4090 (2026-10-31 20:1x, ARC2 - main; lever 4, the indexer dedupe; the user: keep going down the items): the capacity test
# before any kernel work. pif1 = chain n6c's ph3 config (the 4090 sizing under the L4 ballast, bootstrap v7's host-poll TP handoff,
# the adopted parts) with bootstrap_l4prod3 (v8) and ARC3_L4PORT_IDX_FAKEHALF=1: the QSA indexer's compressed-key cache spec at half
# its row bytes (as a TP-2 parity split of the compressed rows would be). The server logs its KV size, then fails at CUDA-graph
# capture (the shapes no longer match; expected). Read: 'GPU KV cache size' against ph3's 1,448,379 tokens (24.04x, KV 3.54 GiB):
# a ratio near (296 + 64) / (296 + 32) = 1.098 per QSA-layer token means the hybrid allocator turns the halved owner into KV;
# ~1.0 means page-size unification pads it away (then the dedupe buys no lanes as specified). Ends 'chain n7 4090 ended'.
set -uo pipefail
D=/mnt/shared/arc2/l4port; LD=$D/logs; L=$LD/chain_n7_4090.log; P4=$D/l4prod; S2=/workspace/kvring_dev/fin74/s2
NS=$D/sidepod/prune_sets_hope/eprune_freq_k128.json
log() { echo "[$(date -u +'%F %T')] $*" >> $L; }
stop() { log "STOP: $*"; log "chain n7 4090 ended"; exit 1; }
m() { md5sum < "$1" | cut -c1-8; }
log "start pod $(hostname) (pid $$); idxfake $(m $P4/l4port_idxfake.py), bootstrap v8 $(m $D/bootstrap_l4prod3/sitecustomize.py), try $(m $P4/try_l4prod.sh)"
[ "$(m $P4/l4port_idxfake.py)" = 5f493d42 ] || stop "l4port_idxfake.py is not 5f493d42"
[ "$(m $D/bootstrap_l4prod3/sitecustomize.py)" = b0b59e48 ] || stop "bootstrap_l4prod3 is not b0b59e48"
[ "$(m $P4/try_l4prod.sh)" = f6922c94 ] || stop "try_l4prod.sh is not f6922c94"
n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); [ "$n" = 0 ] || stop "$n GPU processes"
lb=pif1
[ -e $S2/instances/$lb ] && stop "instance $lb exists"
log "$lb starting (ph3's config + the half-row compressed-key spec; expected to fail at graph capture)"
bash $P4/try_l4prod.sh $lb - --drop-parts ctxfree --set ARC3_L4PORT_PP_SPLIT=26,22 --set ARC3_L4PORT_PP_BALANCE=1 \
  --set ARC3_STAGE2_BATCHED_TOKENS=2048 --set ARC3_S2_EPRUNE_FILE=$NS --set ARC3_PY_PREPEND=$D/bootstrap_l4prod3 \
  --set ARC3_PLE_HANDOFF_IMPL=hostpoll --set ARC3_L4PORT_PLE_HPTP=1 --set ARC3_L4PORT_EVPP=1 --set ARC3_L4PORT_IDX_FAKEHALF=1 > $LD/try_$lb.out 2>&1
SL=$(cat $S2/instances/$lb/last_logdir 2>/dev/null)/server.log
log "$lb KV: $(grep -h -o -E 'Available KV cache memory: [0-9.]+ GiB|Maximum concurrency for [0-9,]+ tokens per request: [0-9.]+x|GPU KV cache size: [0-9,]+ tokens' $SL 2>/dev/null | sort -u | tr '\n' ' ')"
log "$lb receipts: $(grep -h -o -E '\[l4port\] idxfake[^|]{0,200}' $SL 2>/dev/null | sort | uniq -c | tr '\n' ' ' | cut -c1-600)"
log "$lb page-size lines: $(grep -h -o -E '(Padding|padding)[^|]{0,160}|page size[^|]{0,120}' $SL 2>/dev/null | sort | uniq -c | head -n 6 | tr '\n' '|' | cut -c1-900)"
log "$lb end: $(grep -E 'server wrapper exit' $LD/try_l4prod_$lb.log | cut -c22-300)"
log "chain n7 4090 ended"
