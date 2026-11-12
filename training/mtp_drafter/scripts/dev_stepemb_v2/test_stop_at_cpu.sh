#!/bin/bash
# CPU test (2026-10-15, Claude fork 5d20b23b): --stop-at N stops the trainer right after step N in-process, like the stop file:
# status "stopped" at N, a full state at N, no step after N, and the run resumes from that state. Only a temporary dir is touched.
set -e
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); PY=${PY:-python3}; T=$(mktemp -d)
export MTP_KIT=${MTP_KIT:-/workspace/arc3-inference/mtp_ft_20261011} PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2
$PY - $T/files <<'PYEOF'
import os, sys, torch
from safetensors.torch import save_file
d = sys.argv[1]; os.makedirs(d, exist_ok=True)
for i in range(6):
    g = torch.Generator().manual_seed(i); rows = 64
    save_file(dict(token_ids=torch.randint(0, 50, (rows,), generator=g), position_ids=torch.arange(rows) + 1000 * i,
                   served_position_ids=torch.arange(rows), mtp_hidden_states=torch.randn(rows, 32, generator=g)),
              os.path.join(d, 'arc3-persistent-arc3-%032x-x_s%04d.safetensors' % (i // 2, i % 2)))
PYEOF
C="--tiny --device cpu --attention reference --mtp-file none --target-dir none --eval-every 0 --window 32 --accum 2 --lr 1e-2 \
  --loss served_kl --train nonexpert --token-weighted --data $T/files --val-files 0 --warmup 0 --save-every 0 --ckpt-every 3"
$PY $D/train_mtp_sp.py $C --steps 10 --stop-at 4 --out $T/run > $T/a.txt 2>&1 || { cat $T/a.txt; exit 1; }
$PY - $T <<'PYEOF'
import glob, json, sys
T = sys.argv[1]
st = json.load(open(T + '/run/trainer_status.json'))
steps = [r['step'] for r in map(json.loads, open(T + '/run/log.jsonl')) if 'kind' in r]
assert st['status'] == 'stopped' and st['step'] == 4 and max(steps) == 4, (st, steps)
saved = {json.load(open(p))['step'] for p in glob.glob(T + '/run/state_*.pt.json')}
assert 4 in saved, saved
print('  --stop-at 4: status stopped at 4, state at 4 (states %s), last trained step %d' % (sorted(saved), max(steps)))
PYEOF
grep -q "stopped by --stop-at at step 4" $T/a.txt || { cat $T/a.txt; exit 1; }
S=$(grep -l '"step": 4' $T/run/state_*.pt.json | head -1); S=${S%.json}
$PY $D/train_mtp_sp.py $C --steps 6 --resume $S --out $T/run2 > $T/b.txt 2>&1 || { cat $T/b.txt; exit 1; }
grep -q "resumed from .* at step 4" $T/b.txt || { cat $T/b.txt; exit 1; }
echo "  resumed from the stop state at step 4 and finished at 6"
echo "stop-at test ok; scratch left in $T"
