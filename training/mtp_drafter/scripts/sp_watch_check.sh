#!/bin/bash
# sp_watch_check.sh (2026-10-13, Claude fork 5d20b23b; side pod only): one status poll for the local watcher. Prints
# "FAIL <log>: <line>" for failure signatures in the current queue logs, "DONE <line>" for every milestone line, and a
# one-line GPU summary. Read-only; matches no process names (memory pod-process-control).
# usage: bash sp_watch_check.sh
L=/mnt/shared/arc3-inference/sidepod_20261012/logs
LOGS="speed_tests5_20261013.log speed_tests6_20261013.log export_valrbp_20261013.log replay_speedtest_20261013.log
      round2_gpu0.log round2_gpu1.log round3_gpu0.log round3_gpu1.log replay_fast8k_20261013.log round4_gpu0.log round5_gpu0.log round5_gpu1.log round6_gpu0.log round6_gpu1.log round7_gpu0.log round7_gpu1.log"
for f in $LOGS $(cd $L && ls mtp_ft_r2_*.log mtp_ft_r3_*.log mtp_ft_r4_*.log round2_r2_*.log round3_r3_*.log round4_r4_*.log round5_r3_*.log round6_r6_*.log mtp_ft_r6_*.log round7_r7_*.log mtp_ft_r7_*.log early_stop_r7_*.log 2>/dev/null); do
  [ -f $L/$f ] || continue
  grep -E "Traceback|OutOfMemoryError|CUDA out of memory|trainer rc [1-9]|rc [1-9][0-9]*$|identity mismatch|Killed" $L/$f 2>/dev/null \
    | head -n 2 | sed "s|^|FAIL $f: |" | cut -c1-220
  grep -hE "speed tests [0-9]+ done|valrbp export done|replay speed test done|replay fast8k done|arm r[0-9]_[a-z0-9_]+ done|round [0-9] gpu[01] done|stop file created|stopped early" \
    $L/$f 2>/dev/null | sed 's/^\[[0-9:]*\] //; s/^/DONE /'
done | sort -u
echo "GPU $(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader | tr '\n' ';') $(date -u +%T)"
