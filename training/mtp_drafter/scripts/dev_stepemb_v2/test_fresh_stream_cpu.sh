#!/bin/bash
# CPU test (2026-10-14, Claude fork 5d20b23b): --fresh-stream. A stream-mode run consumes 6 segments and keeps its state; the
# same segment names are then replayed again (the fixed-teacher refit re-replays sessions the state already trained). Resumed
# without --fresh-stream, the state's consumed set deletes them unseen (no new step: why the option exists); with --fresh-stream
# the resumed run (model, optimizers, step, RNG restored) trains on all of them. Only the test's temporary directory is touched.
set -e
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); PY=${PY:-python3}; T=$(mktemp -d)
export MTP_KIT=${MTP_KIT:-/workspace/arc3-inference/mtp_ft_20261011} PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2
cat > $T/writer.py <<'PYEOF'
import os, sys, torch
from safetensors.torch import save_file
d = sys.argv[1]; os.makedirs(d, exist_ok=True)
for i in range(6):
    g = torch.Generator().manual_seed(i + int(sys.argv[2])); rows = 64
    save_file(dict(token_ids=torch.randint(0, 50, (rows,), generator=g), position_ids=torch.arange(rows) + 1000 * i,
                   served_position_ids=torch.arange(rows), mtp_hidden_states=torch.randn(rows, 32, generator=g)),
              os.path.join(d, 'arc3-persistent-arc3-%032x-x_s%04d.safetensors' % (i // 2, i % 2)))
open(os.path.join(d, 'DONE'), 'w').write('done\n')
PYEOF
$PY $T/writer.py $T/files 0; mkdir -p $T/val; cp $T/files/*_s0000.safetensors $T/val/
C="--tiny --device cpu --attention reference --mtp-file none --target-dir none --eval-every 0 --window 32 --accum 2 --lr 1e-2 \
  --loss served_kl --cooldown-loss lk --save-every 0 --train nonexpert --token-weighted --prefetch 2 --data $T/val --val-files 1 \
  --steps 100 --warmup 0 --ckpt-every 2 --stream-pool 2"
$PY $T/writer.py $T/s1 0
$PY $D/train_mtp_sp.py $C --out $T/run --stream-dir $T/s1 --stream-done $T/s1/DONE > $T/a.txt 2>&1 || { cat $T/a.txt; exit 1; }
S=$(ls $T/run/state_end_step*.pt | head -1); N=$(basename $S .pt | sed 's/state_end_step//')
echo "  first run: stream ended after step $N ($S)"
$PY $T/writer.py $T/s2 100                          # the same segment names, new contents (the fixed teacher's features)
$PY $D/train_mtp_sp.py $C --out $T/old --stream-dir $T/s2 --stream-done $T/s2/DONE --resume $S > $T/b.txt 2>&1 || { cat $T/b.txt; exit 1; }
nb=$(grep -c '"kind"' $T/old/log.jsonl 2>/dev/null || true)
[ "${nb:-0}" = 0 ] || { echo "resumed without --fresh-stream trained $nb steps on re-replayed segments"; exit 1; }
echo "  resumed without --fresh-stream: 0 steps, the re-replayed segments were dropped as consumed"
$PY $T/writer.py $T/s3 100
$PY $D/train_mtp_sp.py $C --out $T/new --stream-dir $T/s3 --stream-done $T/s3/DONE --resume $S --fresh-stream > $T/c.txt 2>&1 \
  || { cat $T/c.txt; exit 1; }
grep -q "fresh stream" $T/c.txt && grep -q "resumed from .* at step $N" $T/c.txt || { cat $T/c.txt; exit 1; }
$PY - $T $N <<'PYEOF'
import json, sys
T, N = sys.argv[1], int(sys.argv[2])
a = [r['step'] for r in map(json.loads, open(T + '/run/log.jsonl')) if 'kind' in r]
c = [r['step'] for r in map(json.loads, open(T + '/new/log.jsonl')) if 'kind' in r]
assert c and c[0] == N + 1 and len(c) == len(a), (a, c)
print('  --fresh-stream: resumed at step %d, trained %d steps (the first run trained %d)' % (N, len(c), len(a)))
PYEOF
if $PY $D/train_mtp_sp.py $C --out $T/bad --data $T/files --fresh-stream > $T/d.txt 2>&1; then echo "--fresh-stream without --resume ran"; exit 1; fi
echo "fresh-stream test ok; scratch left in $T"
