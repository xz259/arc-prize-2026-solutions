#!/bin/bash
# decomp_fr1c.sh (2026-10-23, Claude main session 099768dc; IE 03:29 UTC; the user: "yes, start the decomposition on GPU 1"): the serving-side
# drafter gap of the adopted fr1c final head (offline trainer tau ~3.55 vs live renewal tau 3.4921 at K 4, T 0.6), one factor at a time
# from production emulation (sp_score_head.sh = served NVFP4 experts, lutc drafter KV with the dkv73 means, draftwin 16,384, QSA attention),
# plus all-ideal. Same scorer, set (next55), head and K as phase E; live renewal tau K 3-5 at T 1 / 0.6 (live_tau.py). The baseline is
# phase E's results/fr1cfinal_valnext55_served (not rerun). Writes results/decomp_fr1c_<cfg>_valnext55{.json,_anchors.pt,_live.txt}
# and results/decomp_fr1c_summary.txt. Refuses a busy GPU.
set -u
[ "$(hostname)" = 2a159825c76b ] || { echo "not the main pod"; exit 9; }
SP=/mnt/shared/arc3-inference/mtp_refit_20261022; cd $SP/scripts
PY=/workspace/arc3-inference/venv-speculators/bin/python
HF=$SP/mtp_ft/fr1c/heads/head_fr1c_final_step13125.safetensors; DEV=${DEV:-cuda:1}
VF="--features /workspace/mtp_refit_r1/features/sp_valnext55mtp --pack $SP/jobs/valnext55/pack"
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date -u +%T)] $*"; }
[ "$(md5sum < $HF | cut -c1-8)" = d9922a38 ] || { log "head md5 is not d9922a38"; exit 2; }
G=${DEV#cuda:}; n=$(nvidia-smi -i $G --query-compute-apps=pid --format=csv,noheader | grep -c .)
[ "$n" = 0 ] || { log "GPU $G has $n compute processes: not running"; exit 3; }
CFGS="experts_bf16|--experts bf16 --kv lutc --draftwin 16384 --attention qsa
kv_bf16|--experts served --kv bf16 --draftwin 16384 --attention qsa
window_full|--experts served --kv lutc --draftwin 0 --attention qsa
attn_dense|--experts served --kv lutc --draftwin 16384 --attention dense
all_ideal|--experts bf16 --kv bf16 --draftwin 0 --attention dense"
while IFS="|" read -r name args; do
  T=decomp_fr1c_${name}_valnext55
  log "scoring $name ($args) on $DEV"
  nice -n 5 $PY -u mtp_eval_val50.py $VF --head $HF $args --device $DEV --K 5 --out $SP/results/$T.json \
    --save-anchors $SP/results/${T}_anchors.pt > $SP/logs/$T.log 2>&1 || { log "FAILED: $name (see logs/$T.log)"; tail -n 3 $SP/logs/$T.log; continue; }
  $PY live_tau.py $SP/results/${T}_anchors.pt --modes prob_srv_T10 prob_srv_T06 --K 3 4 5 > $SP/results/${T}_live.txt 2>&1
  log "$name done: $(tr '\n' ' ' < $SP/results/${T}_live.txt | cut -c1-240)"
done <<< "$CFGS"
$PY live_tau.py $SP/results/fr1cfinal_valnext55_served_anchors.pt --modes prob_srv_T10 prob_srv_T06 --K 3 4 5 \
  > $SP/results/decomp_fr1c_base_live.txt 2>&1
{ echo "decomposition of fr1c final (next55 live renewal tau; base = production emulation, phase E)"
  echo "== base"; cat $SP/results/decomp_fr1c_base_live.txt
  for c in experts_bf16 kv_bf16 window_full attn_dense all_ideal; do
    echo "== $c"; cat $SP/results/decomp_fr1c_${c}_valnext55_live.txt 2>/dev/null
  done; } > $SP/results/decomp_fr1c_summary.txt
log "DECOMP DONE: results/decomp_fr1c_summary.txt"
