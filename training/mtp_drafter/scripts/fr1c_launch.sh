#!/bin/bash
# fr1c_launch.sh (2026-10-22, Claude session 099768dc; PREPARED, run only on the user's call at r2's decision point): the user's option
# "go back and finish the fr1 finetune with the new precision", with the schedule bottoming out at 1e-6 (the user 10-22). Run fr1c =
# fr1 continued: a full --resume of fr1's stop state (state_stop_step10142.pt: FP32 masters, Muon + AdamW moments, RNG, step 10,142,
# the stream position: 10,904 consumed segments) with fr1's own recipe (LR 5e-5, warmup 658, decay from 4,501, served_kl -> lk, loss
# T 1.0, Muon, nonexpert, served experts, K4 1/.69/.40/.18, 16K windows x 4) and --lr-floor 0.02 (cosine to 1e-6 at step 13,164: 3,022
# steps, LR 1.43e-5 at the resume step instead of 1.36e-5). Data: jobs/fr1c/resume.json (mtp_resume_sessions.py: 631 sessions,
# 214.7M stream tokens; 17 that failed in fr1's replay skipped), replayed on the CURRENT stack (fin73/s2 at ctx78 arm b's config, the
# new precision); fr1's streams index for the role masks (/workspace/arc3-inference/dflash_streams_20261011, restored 10-22); eval on
# next55 (as r1 / r2). Needs both GPUs free (r2 stopped first). Log: SP/logs/fullrun_fr1c_a2.log
set -u
SP=/mnt/shared/arc3-inference/mtp_refit_20261022; SPOLD=/mnt/shared/arc3-inference/sidepod_20261012
cd $SP/scripts
RUN=fr1c ATTEMPT=2 RESUME=$SPOLD/mtp_ft/fr1/state_stop_step10142.pt SESS=$SP/jobs/fr1c/resume.json \
  LR=5e-5 OPT=muon TRAIN=nonexpert MOE=grouped NK=4 SW=1,0.69,0.40,0.18 LOSS=served_kl CLOSS=lk LOSST=1.0 \
  STEPS=13164 WARMUP=658 DECAY_START=4501 LRFLOOR=0.02 KEEP=750 CKPT=225 EVAL=375 SKIP_BASELINE=1 \
  STREAMS=/workspace/arc3-inference/dflash_streams_20261011 \
  DRAFTCFG=/workspace/models/Qwen3.8-Flash-Next-mtp-draft-nvfp4a16-noisc/config.json \
  bash start_detached.sh $SP/logs/fullrun_fr1c_a2.log sp_fullrun.sh
