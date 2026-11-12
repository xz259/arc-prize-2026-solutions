#!/bin/bash
# fr1c_launch2.sh (2026-10-22, Claude session 099768dc; the user after r2 was stopped at step 426: "game diversity is a problem here.
# Let's just stop it and do the fr1 continuation plan"; "I don't want to waste any data so I want to just blend them together for the
# last stretch"; the schedule bottoms out at 1e-6). Run fr1c = fr1 continued: a full --resume of fr1's state_stop_step10142.pt (FP32
# masters, Muon + AdamW moments, RNG, step 10,142, stream position) with fr1's recipe (Muon, nonexpert, served experts, K4
# 1/.69/.40/.18, 16K windows x 4, LK loss, loss T 1.0). Data: jobs/fr1c/blend.json (747 sessions, 236.9M stream tokens: the 631 that
# mtp_resume_sessions.py lists for fr1's stop state + fr1's own holdout, the 115 sessions of its 50 redbluepill validation games + the
# one session whose replay failed with an engine error; shuffled with seed 20261022; the 16 'no-recorded-turn' sessions have no model
# turns), replayed on the current stack (fin73/s2 at ctx78 arm b's config: the new precision). Schedule: a fresh cosine from fr1's LR at
# its stop step (1.3569e-5) to 1e-6 over all of it: LR 1.3569e-5, DECAY_START 10,143, STEPS 13,703 (10,142 + 3,561 = 98% of the 14,538
# unconsumed windows / 4), LRFLOOR 0.0737. Eval on next55. Log: SP/logs/fullrun_fr1c_a2.log. DRY=1: sp_fullrun.sh DRY in the foreground.
set -u
SP=/mnt/shared/arc3-inference/mtp_refit_20261022; SPOLD=/mnt/shared/arc3-inference/sidepod_20261012
cd $SP/scripts
if [ -n "${DRY:-}" ]; then L="bash sp_fullrun.sh"; else L="bash start_detached.sh $SP/logs/fullrun_fr1c_a2.log sp_fullrun.sh"; fi
RUN=fr1c ATTEMPT=2 RESUME=$SPOLD/mtp_ft/fr1/state_stop_step10142.pt SESS=$SP/jobs/fr1c/blend.json \
  LR=1.3569e-5 STEPS=13703 WARMUP=658 DECAY_START=10143 LRFLOOR=0.0737 \
  OPT=muon TRAIN=nonexpert MOE=grouped NK=4 SW=1,0.69,0.40,0.18 LOSS=served_kl CLOSS=lk LOSST=1.0 \
  KEEP=750 CKPT=225 EVAL=375 SKIP_BASELINE=1 \
  STREAMS=/workspace/arc3-inference/dflash_streams_20261011 \
  DRAFTCFG=/workspace/models/Qwen3.8-Flash-Next-mtp-draft-nvfp4a16-noisc/config.json \
  $L
