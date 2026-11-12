#!/bin/bash
# CPU test (2026-10-14, Claude fork 5d20b23b; the first milestone's branch failed: "list indices must be integers or slices,
# not tuple"): a milestone branch cooldown resumes a STREAM-mode full-run state in FILES mode (the fixed export). The state's
# order holds prefetched stream windows (tuples); the branch must start a fresh order from its seed. Checks: the stream
# state's order has tuples; two branches from it run to the end with a head; their per-step losses are identical (the same
# data in the same order at every milestone). Only the test's own temporary directory is touched.
set -e
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); PY=${PY:-python3}; T=$(mktemp -d)
export MTP_KIT=${MTP_KIT:-/workspace/arc3-inference/mtp_ft_20261011} PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2
cat > $T/writer.py <<'PYEOF'
import os, sys, torch
from safetensors.torch import save_file
d = sys.argv[1]; os.makedirs(d, exist_ok=True)
for i in range(6):
    g = torch.Generator().manual_seed(i); rows = 64
    save_file(dict(token_ids=torch.randint(0, 50, (rows,), generator=g), position_ids=torch.arange(rows) + 1000 * i,
                   served_position_ids=torch.arange(rows), mtp_hidden_states=torch.randn(rows, 32, generator=g)),
              os.path.join(d, 'arc3-persistent-arc3-%032x-x_s%04d.safetensors' % (i // 2, i % 2)))
if len(sys.argv) > 2:
    open(os.path.join(d, 'DONE'), 'w').write('done\n')
PYEOF
$PY $T/writer.py $T/files                                    # the branch's fixed data (the stream deletes its own copies)
mkdir -p $T/val; cp $T/files/*_s0000.safetensors $T/val/
C="--tiny --device cpu --attention reference --mtp-file none --target-dir none --eval-every 0 --window 32 --accum 2 --lr 1e-2 \
  --loss served_kl --cooldown-loss lk --save-every 0 --train nonexpert --token-weighted --prefetch 2"
$PY $D/train_mtp_sp.py $C --data $T/val --val-files 1 --steps 100 --warmup 0 --ckpt-every 2 --stream-pool 2 --out $T/run \
  --stream-dir $T/s --stream-done $T/s/DONE > $T/stream.txt 2>&1 < /dev/null &
P=$!; $PY $T/writer.py $T/s done; wait $P || { cat $T/stream.txt; exit 1; }
$PY - $T <<'PYEOF'
import sys, torch
st = torch.load(sys.argv[1] + '/run/state_a.pt', map_location='cpu', weights_only=False)
assert any(not isinstance(i, int) for i in st['order']), ('no stream windows in the state order', st['order'][:4])
print('  stream state at step %d holds %d stream windows in its order' % (st['step'], len(st['order'])))
open(sys.argv[1] + '/step', 'w').write(str(st['step']))
PYEOF
S=$(cat $T/step)
for b in 1 2; do
  $PY $D/train_mtp_sp.py $C --data $T/files --val-files 0 --resume $T/run/state_a.pt --decay-start $((S + 1)) --steps $((S + 4)) \
    --warmup 0 --schedule wsd --save-at $((S + 4)) --save-trainable-only --out $T/branch$b > $T/branch$b.txt 2>&1 \
    || { cat $T/branch$b.txt; exit 1; }
  [ -s $T/branch$b/head_step$((S + 4)).safetensors ] || { cat $T/branch$b.txt; echo "no branch head"; exit 1; }
done
$PY - $T <<'PYEOF'
import json, sys
T = sys.argv[1]
l = [[r['loss'] for r in map(json.loads, open('%s/branch%d/log.jsonl' % (T, b))) if 'loss' in r] for b in (1, 2)]
assert len(l[0]) == 4 and l[0] == l[1], l
print('  two branches from the stream state: 4 steps each, identical losses %s' % [round(x, 4) for x in l[0]])
PYEOF
echo "branch-from-stream test ok; scratch left in $T"
