#!/bin/bash
# CPU test of train_mtp_sp.py --stream-dir (2026-10-13, Claude fork 5d20b23b): a fake replay writes 6 segment files of 64 rows
# (temporary name, then rename, as dflashtaps does) one per second into a stream directory, then a DONE file. Checks:
# (1) an uninterrupted run trains every window once (18 windows, accum 2 -> 9 steps), deletes every segment and ends at the
# stream's end; (2) a run stopped at step 3 and resumed from its state finishes the same stream: 9 steps in total, every
# segment consumed and deleted. Only the test's own temporary directories are touched.
set -e
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); PY=${PY:-python3}; T=$(mktemp -d)
export MTP_KIT=${MTP_KIT:-/workspace/arc3-inference/mtp_ft_20261011} PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES=""
mkdir -p $T/val
cat > $T/writer.py <<'PYEOF'
import os, sys, time, torch
from safetensors.torch import save_file
d, n, delay = sys.argv[1], int(sys.argv[2]), float(sys.argv[3])
os.makedirs(d, exist_ok=True)
for i in range(n):
    rows = 64
    tmp = os.path.join(d, '.seg%d.tmp' % i)
    save_file(dict(token_ids=torch.randint(0, 50, (rows,)), position_ids=torch.arange(rows) + 1000 * i,
                   served_position_ids=torch.arange(rows), mtp_hidden_states=torch.randn(rows, 32)), tmp)
    os.rename(tmp, os.path.join(d, 'arc3-persistent-arc3-%032x-x_s%04d.safetensors' % (i // 2, i % 2)))
    time.sleep(delay)
open(os.path.join(d, 'DONE'), 'w').write('done\n')
PYEOF
$PY $T/writer.py $T/val 1 0 > /dev/null; rm -f $T/val/DONE
COMMON="--tiny --device cpu --attention reference --data $T/val --val-files 1 --mtp-file none --target-dir none --eval-every 0 \
  --window 32 --accum 2 --lr 1e-3 --warmup 2 --loss served_kl --prefetch 2 --ckpt-every 2 --save-every 0 --stream-pool 3"
check() {   # dir steps
  $PY - $1 $2 $3 <<'PYEOF'
import json, os, sys
stream, run, want = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = [json.loads(l) for l in open(run + '/log.jsonl') if '"loss"' in l]
left = [n for n in os.listdir(stream) if n.endswith('.safetensors')]
print('  %s: %d steps logged (last %d), %d segment files left, consumed %d' % (os.path.basename(run), len(rows),
      rows[-1]['step'] if rows else 0, len(left), rows[-1]['stream_consumed'] if rows else 0))
assert not left and rows[-1]['step'] == want and rows[-1]['stream_consumed'] == 6, (left, rows[-1]['step'] if rows else None)
PYEOF
}
# (1) uninterrupted
$PY $T/writer.py $T/s1 6 1 & W1=$!
$PY $D/train_mtp_sp.py $COMMON --out $T/run1 --stream-dir $T/s1 --stream-done $T/s1/DONE --steps 100 > $T/out1.txt 2>&1 || { cat $T/out1.txt; exit 1; }
wait $W1
grep -h "stream ended" $T/out1.txt
check $T/s1 $T/run1 9
# (2) stop at step 3, resume
$PY $T/writer.py $T/s2 6 1 & W2=$!
$PY $D/train_mtp_sp.py $COMMON --out $T/run2 --stream-dir $T/s2 --stream-done $T/s2/DONE --steps 3 > $T/out2a.txt 2>&1 || { cat $T/out2a.txt; exit 1; }
STATE=$(ls -t $T/run2/state_*.pt | head -1)
$PY $D/train_mtp_sp.py $COMMON --out $T/run2 --stream-dir $T/s2 --stream-done $T/s2/DONE --steps 100 --resume $STATE > $T/out2b.txt 2>&1 || { cat $T/out2b.txt; exit 1; }
wait $W2
grep -h "resumed from\|stream ended" $T/out2b.txt
check $T/s2 $T/run2 9
echo "stream test ok; scratch left in $T (no deletions outside the test's own temporary directories)"
