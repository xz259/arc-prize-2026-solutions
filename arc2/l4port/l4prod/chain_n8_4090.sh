#!/usr/bin/env bash
# chain n8, 4090 (2026-10-31 20:4x, ARC2 - main; lever 4, the indexer dedupe, first functional test): pis1 = chain n6c's ph3 config
# (the 4090 sizing under the L4 ballast, the host-poll TP handoff, the adopted parts) with bootstrap_l4prod4 (v9) and
# ARC3_L4PORT_IDX_SHARD=1 + the verify mode (ARC3_L4PORT_IDX_SHARD_VERIFY=4: every 4th eager selection call is checked against
# exact_topk on the all-gathered full logits), the decode bench and the 8-task smoke. Compaction of a sharded owner refuses loudly
# (not adapted yet), so an eviction in this run is a planned stop, not a pass. Read: KV tokens against ph3's 1,448,379 (expect
# ~x1.098), the shard receipts (spec, forward, verify: checks / rows / differ, tie-only, real), bench rc and tok/s against ph3
# (1,528.3 at 16 lanes), the smoke, any refusal. Pass: real = 0 with checks > 0, bench rc 0. Ends 'chain n8 4090 ended'.
set -uo pipefail
D=/mnt/shared/arc2/l4port; LD=$D/logs; L=$LD/chain_n8_4090.log; P4=$D/l4prod; S2=/workspace/kvring_dev/fin74/s2
NS=$D/sidepod/prune_sets_hope/eprune_freq_k128.json
log() { echo "[$(date -u +'%F %T')] $*" >> $L; }
stop() { log "STOP: $*"; log "chain n8 4090 ended"; exit 1; }
m() { md5sum < "$1" | cut -c1-8; }
log "start pod $(hostname) (pid $$); idxshard $(m $P4/l4port_idxshard.py), merge $(m $D/idx_shard/merge.py), bootstrap v9 $(m $D/bootstrap_l4prod4/sitecustomize.py), install $(m $D/ple_hostpoll_tp/install.py), try $(m $P4/try_l4prod.sh)"
[ "$(m $P4/l4port_idxshard.py)" = 005b59a0 ] || stop "l4port_idxshard.py is not 005b59a0"
[ "$(m $D/idx_shard/merge.py)" = eff973ff ] || stop "merge.py is not eff973ff"
[ "$(m $D/bootstrap_l4prod4/sitecustomize.py)" = dce24e45 ] || stop "bootstrap_l4prod4 is not dce24e45"
[ "$(m $P4/try_l4prod.sh)" = f6922c94 ] || stop "try_l4prod.sh is not f6922c94"
n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); [ "$n" = 0 ] || stop "$n GPU processes"
lb=pis1
[ -e $S2/instances/$lb ] && stop "instance $lb exists"
[ -e /mnt/shared/arc2/runs/${lb}smoke ] && stop "run ${lb}smoke exists"
log "$lb starting (ph3's config + the indexer shard + verify every 4th eager selection; bench + the 8-task smoke)"
bash $P4/try_l4prod.sh $lb ${lb}smoke --drop-parts ctxfree --set ARC3_L4PORT_PP_SPLIT=26,22 --set ARC3_L4PORT_PP_BALANCE=1 \
  --set ARC3_STAGE2_BATCHED_TOKENS=2048 --set ARC3_S2_EPRUNE_FILE=$NS --set ARC3_PY_PREPEND=$D/bootstrap_l4prod4 \
  --set ARC3_PLE_HANDOFF_IMPL=hostpoll --set ARC3_L4PORT_PLE_HPTP=1 --set ARC3_L4PORT_EVPP=1 \
  --set ARC3_L4PORT_IDX_SHARD=1 --set ARC3_L4PORT_IDX_SHARD_VERIFY=4 > $LD/try_$lb.out 2>&1
SL=$(cat $S2/instances/$lb/last_logdir 2>/dev/null)/server.log
log "$lb try log: $(grep -E 'server wrapper exit|errors:|KV:|bench at|bench rc|decode-only|smoke|exit [0-9]|server stop' $LD/try_l4prod_$lb.log | cut -c22-300 | tr '\n' '|' | cut -c1-2500)"
log "$lb shard receipts: $(grep -h -o -E '\[l4port\] idxshard: [^|]{0,200}' $SL 2>/dev/null | sort | uniq -c | sort -rn | head -n 12 | tr '\n' '|' | cut -c1-2500)"
log "$lb last verify lines: $(grep -h -o -E '\[l4port\] idxshard: verify rank [0-9]: [^|]{0,160}' $SL 2>/dev/null | tail -n 4 | tr '\n' '|')"
log "$lb refusals / errors: $(grep -h -E 'idxshard: compaction|unexpected compressed key cache|Traceback|Error' $SL 2>/dev/null | grep -v -E 'min_frames|max_frames' | sort | uniq -c | sort -rn | head -n 6 | cut -c1-300 | tr '\n' '|')"
log "chain n8 4090 ended"
