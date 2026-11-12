#!/bin/bash
# mtp_launch_r2.sh (2026-10-22, MTP refit; Claude session 099768dc; the user 10-22 after r1 was stopped at step 3,831: "starting with a
# fresh MTP drafter, 5% warmup to 4e-5 and anneal immediately after to 0.01x"; the question: plasticity of fr1's head vs data diversity).
# Waits for both GPUs to be free (r1's replay server stopped; the gdn32 unit test done: 'GDN32 TEST DONE'; OVERDUE 19:00 UTC), then:
#   1. the full run r2 (sp_fullrun.sh b1dbf386 + train_mtp_sp.py 74151aa6 with --lr-floor): the untrained MTP head (no --init-from), LR
#      4e-5, STEPS 7,388, WARMUP 369 (5%), DECAY_START 369 (cosine right after the warmup), LRFLOOR 0.01 (ends at 4e-7), served_kl -> lk
#      at the decay start, loss T 0.6, Muon, nonexpert, served experts, K4 1/.69/.40/.18, CKPT 225, KEEP 750, EVAL 375, baseline skipped
#      (the trainer's step-0 eval is the untrained head); data = refit_r1 (streams_all, the same session list), replay GPU 0, trainer GPU 1;
#   2. phase E for r2 (pe_run.sh d731cd50 RUN=r2: final vs tail3 = 6,000 / 6,750 / end, IE's +0.2% rule, stitch).
# Log: SP/logs/mtp_launch_r2.log
set -u
SP=/mnt/shared/arc3-inference/mtp_refit_20261022
log() { echo "[$(date -u +%T)] $*"; }
DL=$(date -u -d '2026-10-22 19:00:00' +%s); od=0
log "waiting: GDN32 TEST DONE and both GPUs without compute processes"
until grep -q "GDN32 TEST DONE" /workspace/kvring_dev/gdn32/after_r1.log 2>/dev/null \
      && [ -z "$(nvidia-smi -i 0 --query-compute-apps=pid --format=csv,noheader)" ] \
      && [ -z "$(nvidia-smi -i 1 --query-compute-apps=pid --format=csv,noheader)" ]; do
  [ "$(date +%s)" -ge "$DL" ] && [ $od = 0 ] && { log "OVERDUE: GPUs or the gdn32 test not done by 19:00 UTC (still waiting)"; od=1; }
  sleep 20
done
log "$(grep 'GDN32 TEST DONE' /workspace/kvring_dev/gdn32/after_r1.log | tail -n 1); both GPUs free: launching r2"
cd $SP/scripts
RUN=r2 LR=4e-5 OPT=muon TRAIN=nonexpert MOE=grouped NK=4 SW=1,0.69,0.40,0.18 LOSS=served_kl CLOSS=lk LOSST=0.6 \
  STEPS=7388 WARMUP=369 DECAY_START=369 LRFLOOR=0.01 KEEP=750 CKPT=225 EVAL=375 SKIP_BASELINE=1 \
  bash start_detached.sh $SP/logs/fullrun_r2_a1.log sp_fullrun.sh
sleep 5
RUN=r2 OVERDUE_AT="2026-10-23 15:30:00" GIVEUP_AT="2026-10-23 21:00:00" \
  NOTE_DESC="fresh MTP3 head, LR 4e-5, 5% warmup, cosine to 0.01x, loss T 0.6, refit_r1 corpus" \
  bash start_detached.sh $SP/logs/pe_r2.log pe_run.sh
log "R2 LAUNCH DONE (full run log logs/fullrun_r2_a1.log; phase E logs/pe_r2.log)"
