#!/bin/bash
# CPU test (2026-10-13, Claude fork 5d20b23b): with --optimizer muon the router stays on AdamW at --router-lr-scale (IE 10-13)
# and the other 17 2D matrices are one Muon group; a 2-step frozen-experts run saves a state with Muon x 1.0, router x 0.3. Only the
# test's own temporary directory is touched.
set -e
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); PY=${PY:-python3}; T=$(mktemp -d)
export MTP_KIT=${MTP_KIT:-/workspace/arc3-inference/mtp_ft_20261011} PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2
$PY - $T <<'PYEOF'
import os, sys, torch
from safetensors.torch import save_file
for d, n in ((sys.argv[1] + '/val', 1), (sys.argv[1] + '/s', 4)):
    os.makedirs(d, exist_ok=True)
    for i in range(n):
        g = torch.Generator().manual_seed(i)
        save_file(dict(token_ids=torch.randint(0, 50, (64,), generator=g), position_ids=torch.arange(64) + 1000 * i,
                       served_position_ids=torch.arange(64), mtp_hidden_states=torch.randn(64, 32, generator=g)),
                  os.path.join(d, 'arc3-persistent-arc3-%032x-x_s%04d.safetensors' % (i, 0)))
open(sys.argv[1] + '/s/DONE', 'w').write('done\n')
PYEOF
$PY $D/train_mtp_sp.py --tiny --device cpu --attention reference --data $T/val --val-files 1 --mtp-file none --target-dir none \
  --eval-every 0 --window 32 --accum 2 --lr 1e-3 --warmup 0 --loss served_kl --save-every 0 --steps 2 --ckpt-every 2 \
  --optimizer muon --train noexperts --router-lr-scale 0.3 --out $T/run --stream-dir $T/s --stream-done $T/s/DONE > $T/out.txt 2>&1 \
  || { cat $T/out.txt; exit 1; }
grep -h "^Muon:" $T/out.txt | sed 's/^/  /'
$PY - $T <<'PYEOF'
import sys, torch
st = torch.load(sys.argv[1] + '/run/state_a.pt', map_location='cpu', weights_only=False)
mg, ag = st['opts'][0]['param_groups'], st['opts'][1]['param_groups']
m = [(g['lr'], g['lr_scale'], len(g['params'])) for g in mg]
w = sorted((g['lr'], g['lr_scale'], len(g['params'])) for g in ag)
print('  Muon groups (lr, scale, tensors):', m, '; AdamW groups:', w)
assert len(mg) == 1 and m[0][1] == 1.0 and m[0][2] == 17 and abs(m[0][0] - 1e-3) < 1e-12, m
assert any(abs(x[0] - 3e-4) < 1e-12 and x[1] == 0.3 and x[2] == 1 for x in w), w
print('muon / router group test ok')
PYEOF
