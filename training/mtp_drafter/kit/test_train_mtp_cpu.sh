#!/bin/bash
# CPU smoke test of train_mtp.py on a tiny random model and synthetic MTP export files (2026-10-11).
set -e
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); PY=${PY:-python3}; T=$(mktemp -d)
$PY - $T <<'PYEOF'
import sys, torch
from safetensors.torch import save_file
d = sys.argv[1]
for i in range(4):
    n = 64
    extra = dict(mtp_hidden_states=torch.randn(n, 32)) if i % 2 else dict(hidden_states=torch.randn(n, 3, 16))  # both layouts
    save_file(dict(token_ids=torch.randint(0, 50, (n,)), position_ids=torch.arange(n), served_position_ids=torch.arange(n) * 2,
                   **extra), '%s/arc3-persistent-arc3-%032x-x.safetensors' % (d, i))
PYEOF
CUDA_VISIBLE_DEVICES="" $PY $D/train_mtp.py --tiny --device cpu --attention reference --data $T --mtp-file none --target-dir none \
  --out $T/run --steps 6 --eval-every 3 --save-every 6 --window 32 --val-files 1 --lr 1e-3 --warmup 2 > $T/out.txt 2>&1 || { cat $T/out.txt; exit 1; }
$PY - $T <<'PYEOF'
import json, os, sys
d = sys.argv[1]
rows = [json.loads(l) for l in open(d + '/run/log.jsonl')]
tr = [r for r in rows if 'loss' in r]; va = [r for r in rows if 'val_tau_est' in r]
assert len(tr) == 6 and len(va) == 3 and os.path.exists(d + '/run/head_step6.safetensors'), (len(tr), len(va))
print('train_mtp smoke: 6 steps, loss %.3f -> %.3f, val tau_est %.3f -> %.3f, alpha_step1 %.3f, checkpoint written: ok' % (
      tr[0]['loss'], tr[-1]['loss'], va[0]['val_tau_est'], va[-1]['val_tau_est'], va[-1]['val_alpha_step1']))
PYEOF
echo "scratch left in $T (no deletions on the pod)"
