#!/usr/bin/env bash
# chain n4, 4090 (2026-10-31 10:4x, ARC2 - main): two checks on the free 4090.
#   ck0     the Kaggle notebook's new default CKPTFETCH False on the PP production path: try_l4prod_4090.sh with
#           --set ARC3_S2_CKPTFETCH_THREADS=0 (vLLM's own loader in every process), mode '-': the server must come up healthy and
#           bench; its load time is logged beside l4p40's (ckptfetch on). The pod's weights sit on the shared FUSE volume, not
#           Kaggle's NFS: a functional check, not Kaggle's timing.
#   twin44  the PP rerun floor with deterministic cuBLAS (CUBLAS_WORKSPACE_CONFIG=:4096:8; VPTOPK=0, DHEAD_K=20 as twin42): twin42 /
#           twin43 showed the PP path not reproducible run to run even for sequential single requests (34-38% of positions >= 0.01),
#           with vptopk and the MoE finalize ruled out; cuBLAS's split-K heuristics on sm_89 are the next suspect. Reading
#           (pre-stated): a vs b bitwise equal (or far below 34%) -> cuBLAS is the source. Diagnostic only.
# Ends 'chain n4 4090 ended'.
set -uo pipefail
D=/mnt/shared/arc2/l4port; LD=$D/logs; L=$LD/chain_n4_4090.log; P4=$D/l4prod; SP=$D/sidepod
log() { echo "[$(date -u +'%F %T')] $*" >> $L; }
log "start pod $(hostname) (pid $$); try_4090 $(md5sum < $P4/try_l4prod_4090.sh | cut -c1-8), twin_pp $(md5sum < $P4/twin_pp.sh | cut -c1-8)"
n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l); [ "$n" = 0 ] || { log "STOP: $n GPU processes"; log "chain n4 4090 ended"; exit 1; }
sleep 20
bash $P4/try_l4prod_4090.sh ck0 - - --drop-parts ctxfree --set ARC3_S2_CKPTFETCH_THREADS=0 > $LD/try_ck0.out 2>&1
log "ck0 (CKPTFETCH_THREADS=0): $(grep -E 'server wrapper exit|KV:|bench rc' $LD/try_l4prod_ck0.log | cut -c22-300 | tr '\n' '|' | cut -c1-900)"
log "ck0 load: $(grep -h -o -E 'Model loading took [0-9.]+ GiB memory and [0-9.]+ seconds' "$(cat /workspace/kvring_dev/fin74/s2/instances/ck0/last_logdir 2>/dev/null)/server.log" | tr '\n' '|' | cut -c1-500); l4p40 (ckptfetch on): $(grep -h -o -E 'Model loading took [0-9.]+ GiB memory and [0-9.]+ seconds' "$(cat /workspace/kvring_dev/fin74/s2/instances/l4p40/last_logdir 2>/dev/null)/server.log" | tr '\n' '|' | cut -c1-500)"
sleep 30
bash $P4/twin_pp.sh twin44 --drop-parts ctxfree --set ARC3_L4PORT_VPTOPK=0 --set ARC3_L4PORT_DHEAD_K=20 --set CUBLAS_WORKSPACE_CONFIG=:4096:8 > $LD/twin_twin44.out 2>&1
log "twin44: $(grep -E 'server wrapper exit|tf_score [ab] rc|PP rerun' $LD/twin_twin44.log | cut -c22-260 | tr '\n' '|' | cut -c1-1500)"
[ -f $LD/tf_twin_twin44_a.json ] && [ -f $LD/tf_twin_twin44_b.json ] && log "twin44 a vs b: $(python3 $SP/s27_twin_cmp2.py $LD/tf_twin_twin44_a.json $LD/tf_twin_twin44_b.json 2>&1 | head -n 8 | tr '\n' '|' | cut -c1-1200)"
log "chain n4 4090 ended"
