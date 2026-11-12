#!/bin/bash
# CPU test of train_mtp_sp.py's graceful stop and resume on a stream (2026-10-13, Claude fork 5d20b23b; the full run's
# stop / resume / pod move). A fake replay writes 6 segments of 64 rows (temporary name, then rename) into a stream
# directory; windows of 32 rows, accum 2 -> 18 windows = 9 steps when every window is trained exactly once.
# (1) single process with --state-local: a stop file created after the step-4 state stops the run at the end of a step; checks the
#     status file (stopped), the renamed stop file, the state and its sidecar on the "volume" (run dir) with the stream
#     state, the same size as the local scratch copy. Then a "resume replay" re-writes all 6 segments under another request
#     suffix (a pod move replays sessions again) and the run resumes from the stopped state: consumed segments that
#     reappear are deleted unseen, duplicates of unfinished segments are trained once, and the run ends with 9 steps in
#     all, every window once, nothing left in the stream directory.
# (2) two ranks (torchrun, gloo): a stop file after the step-2 state stops both ranks in the same step; resumed as one process,
#     the run finishes the stream with 9 steps in all and nothing left. Only the test's own temporary directories are touched.
set -e
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); PY=${PY:-python3}; TR=${TORCHRUN:-torchrun}; T=$(mktemp -d)
export MTP_KIT=${MTP_KIT:-/workspace/arc3-inference/mtp_ft_20261011} PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2
cat > $T/writer.py <<'PYEOF'
import os, sys, time, torch
from safetensors.torch import save_file
d, n, delay, suffix, done = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4], sys.argv[5]
os.makedirs(d, exist_ok=True)
for i in range(n):
    rows = 64
    g = torch.Generator().manual_seed(i)                       # the same rows on every replay attempt
    tmp = os.path.join(d, '.seg%d_%s.tmp' % (i, suffix))
    save_file(dict(token_ids=torch.randint(0, 50, (rows,), generator=g), position_ids=torch.arange(rows) + 1000 * i,
                   served_position_ids=torch.arange(rows), mtp_hidden_states=torch.randn(rows, 32, generator=g)), tmp)
    os.rename(tmp, os.path.join(d, 'arc3-persistent-arc3-%032x-%s_s%04d.safetensors' % (i // 2, suffix, i % 2)))
    time.sleep(delay)
open(os.path.join(d, done), 'w').write('done\n')
PYEOF
mkdir -p $T/val; $PY $T/writer.py $T/val 1 0 v DONE > /dev/null; rm -f $T/val/DONE
COMMON="--tiny --device cpu --attention reference --data $T/val --val-files 1 --mtp-file none --target-dir none --eval-every 0 \
  --window 32 --accum 2 --lr 1e-3 --warmup 2 --loss served_kl --prefetch 2 --ckpt-every 4 --save-every 0 --stream-pool 2 --steps 100"
stop_after() {   # run_dir sidecar: create the stop file once that state's sidecar exists (the log is flushed every 10 steps)
  for i in $(seq 1 600); do
    if [ -f $1/$2 ]; then touch $1/STOP; return; fi
    sleep 0.2
  done
}
check() {   # stream run expected_steps
  $PY - $1 $2 $3 <<'PYEOF'
import json, os, sys
stream, run, want = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = [json.loads(l) for l in open(run + '/log.jsonl') if '"loss"' in l]
steps = [r['step'] for r in rows]
left = [n for n in os.listdir(stream) if n.endswith('.safetensors')]
st = json.load(open(run + '/trainer_status.json'))
print('  %s: %d step records (unique %d, last %d), %d files left, status %s' % (os.path.basename(run), len(steps),
      len(set(steps)), steps[-1], len(left), st['status']))
assert not left and steps[-1] == want and sorted(set(steps)) == list(range(1, want + 1)), (left, steps)
assert st['status'] in ('stream_ended', 'finished'), st
PYEOF
}
# (1) single process: stop, re-replay under another suffix, resume
$PY $T/writer.py $T/s1 6 2 x DONE & W1=$!          # paced: the trainer waits for segments, so a stop lands mid-stream
( stop_after $T/run1 state_a.pt.json ) & S1=$!
$PY $D/train_mtp_sp.py $COMMON --out $T/run1 --stream-dir $T/s1 --stream-done $T/s1/DONE --state-local $T/local1 > $T/out1a.txt 2>&1 || { cat $T/out1a.txt; exit 1; }
wait $S1; wait $W1
grep -h "stopped by\|state .* on the volume" $T/out1a.txt | sed 's/^/  /'
$PY - $T/run1 $T/local1 <<'PYEOF'
import glob, json, os, sys
run, local = sys.argv[1], sys.argv[2]
st = json.load(open(run + '/trainer_status.json'))
assert st['status'] == 'stopped' and not st['copy_errors'], st
assert not os.path.exists(run + '/STOP') and os.path.exists(run + '/STOP.done_step%d' % st['step']), os.listdir(run)
side = json.load(open(st['state'] + '.json'))
assert side['step'] == st['step'] and os.path.getsize(st['state']) == side['bytes'] == os.path.getsize(local + '/state_scratch.pt')
assert 'stream' in side and side['stream']['consumed'] is not None, side.keys()
print('  stopped at step %d: state %s (%d bytes), sidecar consumed %s' % (st['step'], os.path.basename(st['state']),
      side['bytes'], side['stream']['consumed']))
PYEOF
STATE=$($PY -c "import json,sys; print(json.load(open(sys.argv[1]))['state'])" $T/run1/trainer_status.json)
$PY $T/writer.py $T/s1 6 0.3 y DONE2 & W2=$!          # the resume replay: every session again, other request suffix
$PY $D/train_mtp_sp.py $COMMON --out $T/run1 --stream-dir $T/s1 --stream-done $T/s1/DONE2 --state-local $T/local1 --resume $STATE > $T/out1b.txt 2>&1 || { cat $T/out1b.txt; exit 1; }
wait $W2
grep -h "resumed from\|stream ended" $T/out1b.txt | sed 's/^/  /'
check $T/s1 $T/run1 9
# (2) two ranks: stop after step 2, resume as one process
$PY $T/writer.py $T/s2 6 2 x DONE & W3=$!
( stop_after $T/run2 state_a.pt.json ) & S2=$!
$TR --standalone --nproc_per_node 2 $D/train_mtp_sp.py $COMMON --ckpt-every 2 --out $T/run2 --stream-dir $T/s2 --stream-done $T/s2/DONE > $T/out2a.txt 2>&1 || { cat $T/out2a.txt; exit 1; }
wait $S2; wait $W3
grep -h "stopped by" $T/out2a.txt | sed 's/^/  /'
STATE=$($PY -c "import json,sys; d=json.load(open(sys.argv[1])); assert d['status']=='stopped', d; print(d['state'])" $T/run2/trainer_status.json)
$PY $D/train_mtp_sp.py $COMMON --out $T/run2 --stream-dir $T/s2 --stream-done $T/s2/DONE --resume $STATE > $T/out2b.txt 2>&1 || { cat $T/out2b.txt; exit 1; }
grep -h "resumed from\|stream ended" $T/out2b.txt | sed 's/^/  /'
$PY - $T/s2 $T/run2 <<'PYEOF'
import json, os, sys
stream, run = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(run + '/log.jsonl') if '"loss"' in l]
left = [n for n in os.listdir(stream) if n.endswith('.safetensors')]
print('  %s: %d step records, last %d, %d files left' % (os.path.basename(run), len(rows), rows[-1]['step'], len(left)))
assert not left
PYEOF
echo "stop/resume test ok; scratch left in $T (no deletions outside the test's own temporary directories)"
