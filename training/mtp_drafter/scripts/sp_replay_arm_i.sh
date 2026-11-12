#!/bin/bash
# sp_replay_arm_i.sh (2026-10-13, Claude fork 5d20b23b; the MTP full run's two-replay layout, user 10-13: 2 replays +
# 2 trainers on the 4-GPU pod): S2's phase6_replay_arm.sh (2026-10-10) for one replay instance, run by lc_run3i.sh (which
# passes PORT): the server check is this port's /health (the other instance's server must not satisfy it, as in
# run_v62i_arm.sh) and the replay driver gets --base-url on PORT. Everything else as phase6_replay_arm.sh: the driver walks
# the session list at P6_CONCURRENCY, pausing session starts while the RAM directory is over P6_SHM_LIMIT_GB (both
# instances gate on the same directory). Outputs to S2/client_mtp/<label>. LABEL SECONDS [GAMES (ignored)].
set -e
LABEL=${1:?label}; SECS=${2:?seconds}
X=/workspace/arc3-inference/runs/stage0-reference-20260919-v1/stage12/harness_revision_v529/spec_decode_stage2
R=/workspace/arc3-inference
OUT=$X/client_mtp/$LABEL
PORT=${PORT:?port}
[ -e "$OUT" ] && { echo "output exists: $OUT"; exit 1; }
[ "$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 http://127.0.0.1:$PORT/health)" = 200 ] || { echo "no server on port $PORT"; exit 1; }
mkdir -p $OUT "${P6_FEATURES:?features dir}"
exec timeout $SECS $R/venv-duck/bin/python -u $X/replay_driver.py ${P6_SESSIONS:?sessions json} $OUT/driver.jsonl \
  --base-url http://127.0.0.1:$PORT --concurrency ${P6_CONCURRENCY:-16} --timeout 3700 --shm-dir $P6_FEATURES \
  --shm-limit-gb ${P6_SHM_LIMIT_GB:-110} --progress-file ${P6_PROGRESS:?progress file} --windows ${P6_WINDOWS:?windows json} \
  --lookahead-chunks 2 ${P6_ALIAS_SUFFIX:+--alias-suffix=$P6_ALIAS_SUFFIX}
