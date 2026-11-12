#!/bin/bash
# CPU test of train_mtp_sp.py's data parallel path (2026-10-13, Claude fork 5d20b23b; for the 4-GPU pod: 1 replay + 3 trainers):
# (1) allreduce_grads on 2 gloo ranks with rank-dependent gradients and divisors gives the sum / divisor on both ranks;
# (2) torchrun with 2 ranks on a fake stream of 8 segments: the ranks split the segments, every segment is consumed and
#     deleted (ranks without data at the end add zero gradients), the run ends at the stream's end;
# (3) a 2-rank run stopped at step 2 and resumed as a single process finishes the same stream (the union stream state
#     carries across world sizes). Only the test's own temporary directories are touched.
set -e
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); PY=${PY:-python3}; TR=${TORCHRUN:-torchrun}; T=$(mktemp -d)
export MTP_KIT=${MTP_KIT:-/workspace/arc3-inference/mtp_ft_20261011} PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2
cat > $T/ar.py <<'PYEOF'
import os, sys, torch, torch.distributed as dist
sys.path.insert(0, sys.argv[1])
import train_mtp_sp as T
dist.init_process_group('gloo')
r = dist.get_rank()
ps = [torch.nn.Parameter(torch.zeros(5)), torch.nn.Parameter(torch.zeros(1 << 24)), torch.nn.Parameter(torch.zeros(3))]
ps[0].grad = torch.full((5,), float(r + 1)); ps[1].grad = torch.full((1 << 24,), float(10 * (r + 1)))
ps[2].grad = None if r == 1 else torch.full((3,), 6.0)            # rank 1 has no gradient for this one
T.allreduce_grads(ps, 2.0)
ok = torch.allclose(ps[0].grad, torch.full((5,), 1.5)) and torch.allclose(ps[1].grad[:4], torch.full((4,), 15.0)) \
    and torch.allclose(ps[2].grad, torch.full((3,), 3.0))
print('rank %d allreduce_grads %s' % (r, 'ok' if ok else 'WRONG %s %s %s' % (ps[0].grad, ps[1].grad[:4], ps[2].grad)), flush=True)
assert ok
dist.destroy_process_group()
PYEOF
$TR --standalone --nproc_per_node 2 $T/ar.py $D 2>&1 | grep -E "allreduce_grads|Error" | sed 's/^/  /'
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
mkdir -p $T/val; $PY $T/writer.py $T/val 1 0 > /dev/null; rm -f $T/val/DONE
COMMON="--tiny --device cpu --attention reference --data $T/val --val-files 1 --mtp-file none --target-dir none --eval-every 0 \
  --window 32 --accum 2 --lr 1e-3 --warmup 2 --loss served_kl --prefetch 2 --ckpt-every 2 --save-every 0 --stream-pool 2"
check() {
  $PY - $1 $2 <<'PYEOF'
import json, os, sys
stream, run = sys.argv[1], sys.argv[2]
rows = [json.loads(l) for l in open(run + '/log.jsonl') if '"loss"' in l]
left = [n for n in os.listdir(stream) if n.endswith('.safetensors')]
print('  %s: %d steps logged (last %d), %d segment files left' % (os.path.basename(run), len(rows), rows[-1]['step'], len(left)))
assert not left, left
PYEOF
}
$PY $T/writer.py $T/s1 8 0.5 & W1=$!
$TR --standalone --nproc_per_node 2 $D/train_mtp_sp.py $COMMON --out $T/run1 --stream-dir $T/s1 --stream-done $T/s1/DONE --steps 100 > $T/out1.txt 2>&1 || { cat $T/out1.txt; exit 1; }
wait $W1
grep -h "stream ended" $T/out1.txt | sed 's/^/  /'
check $T/s1 $T/run1
$PY $T/writer.py $T/s2 8 0.5 & W2=$!
$TR --standalone --nproc_per_node 2 $D/train_mtp_sp.py $COMMON --out $T/run2 --stream-dir $T/s2 --stream-done $T/s2/DONE --steps 2 > $T/out2a.txt 2>&1 || { cat $T/out2a.txt; exit 1; }
STATE=$(ls -t $T/run2/state_*.pt | head -1)
$PY $D/train_mtp_sp.py $COMMON --out $T/run2 --stream-dir $T/s2 --stream-done $T/s2/DONE --steps 100 --resume $STATE > $T/out2b.txt 2>&1 || { cat $T/out2b.txt; exit 1; }
wait $W2
grep -h "resumed from\|stream ended" $T/out2b.txt | sed 's/^/  /'
check $T/s2 $T/run2
echo "ddp test ok; scratch left in $T (no deletions outside the test's own temporary directories)"
