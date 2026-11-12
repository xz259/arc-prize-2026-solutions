#!/bin/bash
# sk1_launch.sh (2026-10-22, Claude main session 099768dc; the user's "Yes to all three" after IE's review, V67_DESIGN 262): the
# step-conditioned round, form A. RUN sk1 = fr1c's phase-E head (HEAD; default: the head on pe_fr1c.log's PHASE E DONE line) with
# per-step vectors s_2..s_4 (step_emb.py: zero init, one H vector on all hc streams after the fuse, steps >= 2) trained ALONE:
#   - trunk FROZEN (--trunk-lr-scale 0: every other tensor keeps the served head's values); s_k in its own AdamW group (no weight
#     decay) at 10 x LR = 6e-5. Sizing (fuse_rms.py on fr1's head, 8 next55 windows): the fuse output's per-coordinate RMS p50 is 0.400;
#     6.1e-5 x 1,640.5 effective full-LR steps = 0.10 = 25% of it (IE: reach ~20-30%);
#   - WSD: warmup 94, constant to 1,500, cosine to 0 at 1,875 (IE: cosine over the last 20%); LK throughout, loss T 1.0 (fr1c's
#     objective); K 4, step weights 1 / .69 / .40 / .18; 16K x 1 window per step (ACCUM 1); Muon for the (frozen) trunk as fr1c;
#   - data jobs/sk1/sessions.json: 104 refit-corpus sessions (seed 20261023; all 13 runs; current stack, T 0.6 generation; unseen by
#     fr1 / fr1c), 31.6M stream tokens, 1,970 windows (1,875 steps use 95%); replay GPU 0, trainer GPU 1;
#   - eval next55 every 375 steps (step 0 = the base head): tau_srv06 / tau_srv, alpha_srv06_step1..4, val_stepemb_rms (train_mtp_sp.py
#     a17329a9). The gate is the paired next55 score after the run (pe_sk1.sh, paired_live_tau.py).
# A resume (sp_fullrun_ctl.sh 0dd2666d) drops --init-from from EXTRA and keeps the step-embedding flags. DRY=1: sp_fullrun.sh DRY.
set -u
SP=/mnt/shared/arc3-inference/mtp_refit_20261022
cd $SP/scripts
if [ -z "${HEAD:-}" ]; then
  HEAD=$(grep -oP 'PHASE E DONE: choice \S+, head \K[^,]+' $SP/logs/pe_fr1c.log 2>/dev/null | tail -n 1)
fi
[ -s "${HEAD:-}" ] || { echo "no head (HEAD='${HEAD:-}')"; exit 1; }
if [ -n "${DRY:-}" ]; then L="bash sp_fullrun.sh"; else L="bash start_detached.sh $SP/logs/fullrun_sk1_a1.log sp_fullrun.sh"; fi
echo "[$(date -u +%T)] sk1: init $HEAD (md5 $(md5sum < $HEAD | cut -c1-8))${DRY:+ (DRY)}"
RUN=sk1 ATTEMPT=1 SESS=$SP/jobs/sk1/sessions.json \
  LR=6e-6 STEPS=1875 WARMUP=94 DECAY_START=1500 LRFLOOR=0 \
  OPT=muon TRAIN=nonexpert MOE=grouped NK=4 SW=1,0.69,0.40,0.18 LOSS=lk CLOSS=lk LOSST=1.0 ACCUM=1 \
  CKPT=125 KEEP=375 EVAL=375 SKIP_BASELINE=1 \
  STREAMS=/workspace/mtp_refit_r1/streams_all \
  DRAFTCFG=/workspace/models/Qwen3.8-Flash-Next-mtp-draft-nvfp4a16-noisc/config.json \
  EXTRA="--step-emb --step-emb-lr-scale 10 --trunk-lr-scale 0 --init-from $HEAD" \
  $L
