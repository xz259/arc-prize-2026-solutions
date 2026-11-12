#!/bin/bash
# start_detached.sh (2026-10-13, Claude fork 5d20b23b; side pod only): start a script fully detached (own session, no
# terminal, stdin from /dev/null) so the ssh call that starts it returns at once (memory pod-process-control: setsid
# inside a script file, never after && in an ssh one-liner). Prints the new process id.
# usage: bash start_detached.sh LOG SCRIPT [ARGS...]      (environment is inherited)
set -u
LOG=${1:?log}; shift
setsid nohup bash "$@" > "$LOG" 2>&1 < /dev/null &
echo "started pid $! -> $LOG"
