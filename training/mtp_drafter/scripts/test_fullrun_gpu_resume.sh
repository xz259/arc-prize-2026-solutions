#!/bin/bash
# test_fullrun_gpu_resume.sh (2026-10-13, Claude fork 5d20b23b; side pod only; IE 10-13 item e2): the full run's stop / resume
# across trainer counts on the GPUs, once, before committing. Real model, 16K windows, --ckpt-steps, compiled flex,
# grouped MoE, Muon (2D non-expert) + AdamW (rest), served KL, 2 windows per step throughout, on a 10-segment stream of
# symlinks to the 20M export (the trainer deletes only the links; the export is untouched). The full run's recipe (10-13):
# routed experts and router frozen (--train nonexpert) at the served NVFP4 weights, 2,048-row minimum windows, token-weighted
# loss; Muon and AdamW groups both exercised.
#   1. one trainer (cuda:1, accum 2) until the step-6 state, stop file -> "stopped";
#   2. two trainers (torchrun, cuda:0,1, accum 1) resumed from that state, stop after the next state;
#   3. one trainer (cuda:0, accum 2) resumed, runs to the end of the stream.
# Checks: steps 1..N each logged once, the logged LR equals the schedule at every step, statuses stopped / stopped / end,
# no traceback. Needs both GPUs free. Writes only /workspace/mtp_gputest_<time>/ (container disk, small).
# usage: bash test_fullrun_gpu_resume.sh
set -u
[ "$(hostname)" = 6bc94744da90 ] || { echo "not the side pod"; exit 9; }
SP=/mnt/shared/arc3-inference/sidepod_20261012
R=/workspace/arc3-inference
M=/workspace/models/Qwen3.8-Flash-Next-W4A16-intel
PY=$R/venv-speculators/bin/python
TR=$R/venv-speculators/bin/torchrun
T=/workspace/mtp_gputest_$(date -u +%Y%m%dT%H%M%S)
cd $SP/scripts
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True MTP_MOE=grouped
for g in 0 1; do
  m=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i $g)
  [ "$m" -lt 2000 ] || { echo "GPU $g busy ($m MiB)"; exit 1; }
done
mkdir -p $T/stream
n=0
for f in $(ls $SP/features/sp_mtptrain20m/*_s0000.safetensors | head -n 10); do ln -s $f $T/stream/$(basename $f); n=$((n + 1)); done
echo done > $T/stream/DONE
echo "[$(date -u +%T)] $n segments linked into $T/stream"
COMMON="--data $SP/features/sp_valrbp50mtp --val-files 50 --streams $R/dflash_streams_20261011 --mtp-file $M/model_extra_tensors.safetensors \
  --target-dir $M --out $T/run --state-local $T/local --stream-dir $T/stream --stream-done $T/stream/DONE --stream-pool 2 \
  --steps 40 --lr 4e-5 --warmup 4 --schedule wsd --decay-ratio 0.2 --num-steps 4 --step-weights 1,0.69,0.40,0.18 \
  --loss served_kl --cooldown-loss lk --window 16384 --min-window 2048 --token-weighted --train nonexpert --expert-weights served --optimizer muon \
  --eval-every 0 --save-every 0 --ckpt-every 3 --attention flex --compile-flex --fused-adam --prefetch 2 --pos-key position_ids --ckpt-steps"
stop_at() {   # sidecar step: touch STOP once a state sidecar shows that step or later
  for i in $(seq 1 900); do
    s=$($PY -c "import glob, json; print(max([json.load(open(p))['step'] for p in glob.glob('$T/run/state_*.pt.json')] or [0]))" 2>/dev/null)
    [ "${s:-0}" -ge $1 ] && { touch $T/run/STOP; echo "  stop requested after the step-$s state"; return; }
    sleep 2
  done
}
state() { $PY -c "import json; print(json.load(open('$T/run/trainer_status.json'))['state'])"; }
echo "[$(date -u +%T)] phase 1: one trainer"
( stop_at 6 ) & S=$!
$PY -u train_mtp_sp.py $COMMON --device cuda:1 --accum 2 > $T/p1.txt 2>&1; echo "  rc $?"; wait $S
grep -h "stopped by" $T/p1.txt | sed 's/^/  /'
ST=$(state); echo "  state $ST"
echo "[$(date -u +%T)] phase 2: two trainers (torchrun)"
P=$($PY -c "import json; print(json.load(open('$T/run/trainer_status.json'))['step'])")
( stop_at $((P + 3)) ) & S=$!
$TR --standalone --nproc_per_node 2 train_mtp_sp.py $COMMON --devices 0,1 --accum 1 --resume $ST > $T/p2.txt 2>&1; echo "  rc $?"; wait $S
grep -h "resumed from\|stopped by" $T/p2.txt | sed 's/^/  /'
ST=$(state); echo "  state $ST"
echo "[$(date -u +%T)] phase 3: one trainer to the end"
$PY -u train_mtp_sp.py $COMMON --device cuda:0 --accum 2 --resume $ST > $T/p3.txt 2>&1; echo "  rc $?"
grep -h "resumed from\|stream ended" $T/p3.txt | sed 's/^/  /'
$PY - $T <<'PYEOF'
import json, math, sys
sys.path.insert(0, '.')
import train_mtp_sp as TM
T = sys.argv[1]
rows = [json.loads(l) for l in open(T + '/run/log.jsonl') if '"loss"' in l]
steps = [r['step'] for r in rows]
assert steps == list(range(1, len(steps) + 1)), steps
bad = [(r['step'], r['lr']) for r in rows if abs(r['lr'] - TM.lr_at(r['step'], 40, 4e-5, 4, 'wsd', 0.2)) > 1e-12]
assert not bad, bad
assert all(math.isfinite(r['loss']) for r in rows)
st = json.load(open(T + '/run/trainer_status.json'))
print('  %d steps, LR on schedule at every step, losses finite (first %.3f, last %.3f), final status %s' % (
    len(rows), rows[0]['loss'], rows[-1]['loss'], st['status']))
for p in ('p1', 'p2', 'p3'):
    assert 'Traceback' not in open('%s/%s.txt' % (T, p)).read(), p
PYEOF
RC=$?
[ $RC = 0 ] && echo "[$(date -u +%T)] gpu resume test ok ($T)" || echo "[$(date -u +%T)] gpu resume test FAILED ($T)"
