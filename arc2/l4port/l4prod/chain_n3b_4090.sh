#!/usr/bin/env bash
# chain n3b, 4090 (2026-10-31 10:1x, ARC2 - main): chain_n3_4090.sh after it ends, with ARC3_L4PORT_DHEAD_K=20 set explicitly (twin40
# failed at engine init: the drafter-head hook takes its K from vptopk's K by default, so VPTOPK=0 gave K=0 and a reshape of 0
# elements in l4port_dheadtp.compute_logits); labels twin42 / twin43. chain n3's text follows.
# chain n3, 4090 (2026-10-31 08:0x, ARC2 - main; the helper's twin read: twin39's PP side ran with vptopk on (prompt logprobs over
# the candidates only) and is not run-to-run reproducible, |dlp| >= 0.01 at ~36% of positions, while TP 1 on the RTX is bitwise
# reproducible; so PP vs TP 1 sits inside PP's own floor and cannot localize a port bug). After chain n2 ends (l4p40): two PP twins
# on the 4090 (twin_pp.sh: scaled16 without SR, tf_score twice on the 32 streams, then the stop), both with vptopk off:
#   twin40  ARC3_L4PORT_VPTOPK=0 (exact prompt logprobs; MoE finalize 'fused' as the L4 path);
#   twin41  ARC3_L4PORT_VPTOPK=0 + ARC3_L4_MOE_FINALIZE=unfused (production's value on the RTX: the fused finalize may sum with
#           atomics, a candidate source of run-to-run differences).
# For each: its a vs b floor and a vs the RTX's TP 1 side (tf_twin_tp1_a.json) by sidepod/s27_twin_cmp2.py (27d5a2e3). Reading
# (pre-stated): a twin whose a vs b is bitwise equal (or far below twin39's 36% >= 0.01) has deterministic base logits; then its
# PP vs TP 1 is read on statistics (the kernels differ across architectures). Diagnostic only: no adoption from it. Ends
# 'chain n3 4090 ended'.
set -uo pipefail
D=/mnt/shared/arc2/l4port; LD=$D/logs; L=$LD/chain_n3b_4090.log; P4=$D/l4prod; SP=$D/sidepod
log() { echo "[$(date -u +'%F %T')] $*" >> $L; }
log "start pod $(hostname) (pid $$); twin_pp $(md5sum < $P4/twin_pp.sh | cut -c1-8), cmp2 $(md5sum < $SP/s27_twin_cmp2.py | cut -c1-8)"
DL=$(( $(date +%s) + 8 * 3600 ))
until grep -q -E '^\[[0-9: -]+\] chain n3 4090 ended$' $LD/chain_n3_4090.log 2>/dev/null; do
  [ "$(date +%s)" -ge "$DL" ] && { log "STOP: OVERDUE waiting for chain n3"; log "chain n3b 4090 ended"; exit 1; }
  sleep 60
done
for t in "twin42 --set ARC3_L4PORT_VPTOPK=0 --set ARC3_L4PORT_DHEAD_K=20" "twin43 --set ARC3_L4PORT_VPTOPK=0 --set ARC3_L4PORT_DHEAD_K=20 --set ARC3_L4_MOE_FINALIZE=unfused"; do
  set -- $t; lb=$1; shift
  sleep 30
  bash $P4/twin_pp.sh $lb --drop-parts ctxfree "$@" > $LD/twin_$lb.out 2>&1
  log "$lb ($*): $(grep -E 'server wrapper exit|tf_score [ab] rc|PP rerun' $LD/twin_$lb.log | cut -c22-260 | tr '\n' '|' | cut -c1-1500)"
  for pair in "a b" "a tp1"; do
    set -- $pair
    A=$LD/tf_twin_${lb}_$1.json; B=$LD/tf_twin_${lb}_$2.json; [ "$2" = tp1 ] && B=$LD/tf_twin_tp1_a.json
    [ -f $A ] && [ -f $B ] && log "$lb $1 vs $2: $(python3 $SP/s27_twin_cmp2.py $A $B 2>&1 | head -n 8 | tr '\n' '|' | cut -c1-1200)"
  done
done
log "chain n3b 4090 ended"
