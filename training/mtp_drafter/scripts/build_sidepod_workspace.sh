#!/bin/bash
# build_sidepod_workspace.sh (2026-10-12, Claude fork 5d20b23b; run on the side pod only).
# The side pod mounts the shared network volume at /mnt/shared and keeps /workspace on its own container disk. This
# script builds that /workspace: private copies of the trees that are pinned to one host or hold the running server's
# state (runs/dense_fp8_v1, the stage-2 harness tree without its run outputs, the runtime cache), fresh private
# directories for runtime writes (logs, the HF datasets cache), and absolute links to /mnt/shared for everything else
# (models, venvs, runtime, corpus and Phase 6 data). Every script keeps its /workspace paths; whatever the side pod
# creates lands on its own disk. Directories of the main run that side-pod tooling reads (dflash_work_20261009,
# dflash_phase6_20261011v, client_mtp) are link farms, so a new file created there stays local. Overwriting an
# existing linked file would still reach the shared volume: side-pod scripts use their own names and paths.
# Creates directories, links and copies only; never deletes, never overwrites. Re-runnable (RESUME=1 continues copies).
# usage: HOST_EXPECT=<side pod hostname> [BW=150000 KiB/s] [RESUME=1] bash build_sidepod_workspace.sh
set -euo pipefail
SH=/mnt/shared; WS=/workspace; A=arc3-inference
H=runs/stage0-reference-20260919-v1/stage12/harness_revision_v529
BW=${BW:-150000}
[ "$(hostname)" = "${HOST_EXPECT:?set HOST_EXPECT to the side pod hostname}" ] || { echo "hostname $(hostname) is not $HOST_EXPECT"; exit 1; }
mountpoint -q $SH || { echo "$SH is not a mount point"; exit 1; }
[ "$(df --output=fstype $WS | tail -n 1 | tr -d ' ')" = overlay ] || { echo "$WS is not on the container disk"; exit 1; }
[ -s $SH/$A/server-identity.json ] && [ -d $SH/$A/runs/dense_fp8_v1 ] || { echo "$SH does not look like the shared volume"; exit 1; }

farm() {   # farm SRC DST [EXCEPT...]: DST is a real directory; each entry of SRC is linked by absolute path, except those named
  local src=$1 dst=$2 e n x skip; shift 2
  mkdir -p "$dst"
  for e in "$src"/* "$src"/.[!.]*; do
    [ -e "$e" ] || [ -L "$e" ] || continue
    n=${e##*/}; skip=0
    for x in "$@"; do [ "$n" = "$x" ] && skip=1; done
    [ $skip = 1 ] && continue
    [ -e "$dst/$n" ] || [ -L "$dst/$n" ] || ln -s "$e" "$dst/$n"
  done
  echo "[$(date -u +%T)] farm $dst: $(find "$dst" -maxdepth 1 -type l | wc -l) links"
}
copy() {   # copy SRC DST [rsync options]: rate-capped; a non-empty DST only with RESUME=1
  local src=$1 dst=$2; shift 2
  if [ -d "$dst" ] && [ -n "$(ls -A "$dst")" ] && [ "${RESUME:-0}" != 1 ]; then echo "$dst is not empty (RESUME=1 continues)"; exit 1; fi
  mkdir -p "$dst"
  echo "[$(date -u +%T)] copy $src -> $dst"
  rsync -a --bwlimit=$BW "$@" "$src/" "$dst/"
  echo "[$(date -u +%T)] copied $(du -sh "$dst" | cut -f1)"
}

echo "[$(date -u +%T)] build on $(hostname); shared $SH; container disk $(df -h --output=avail $WS | tail -n 1)"
[ -e $WS/models ] || ln -s $SH/models $WS/models
farm $SH/$A $WS/$A runs cache logs server-identity.json __pycache__ hf_datasets_cache_p6 dflash_work_20261009 dflash_phase6_20261011v
mkdir -p $WS/$A/logs $WS/$A/hf_datasets_cache_p6
farm $SH/$A/dflash_work_20261009 $WS/$A/dflash_work_20261009 __pycache__
farm $SH/$A/dflash_phase6_20261011v $WS/$A/dflash_phase6_20261011v __pycache__
farm $SH/$A/runs $WS/$A/runs dense_fp8_v1 stage0-reference-20260919-v1 __pycache__
farm $SH/$A/runs/stage0-reference-20260919-v1 $WS/$A/runs/stage0-reference-20260919-v1 stage12 __pycache__
farm $SH/$A/runs/stage0-reference-20260919-v1/stage12 $WS/$A/runs/stage0-reference-20260919-v1/stage12 harness_revision_v529 __pycache__
copy $SH/$A/runs/dense_fp8_v1 $WS/$A/runs/dense_fp8_v1
copy $SH/$A/$H $WS/$A/$H --exclude=/spec_decode_stage2/client_mtp
farm $SH/$A/$H/spec_decode_stage2/client_mtp $WS/$A/$H/spec_decode_stage2/client_mtp
copy $SH/$A/cache $WS/$A/cache
echo "[$(date -u +%T)] build done; container disk free $(df -h --output=avail $WS | tail -n 1)"
