#!/bin/bash
# CPU test of train_mtp_sp.py stop / resume (2026-10-13, Claude fork 5d20b23b): an 8-step run with full states every 4 steps
# (prefetch 2, Muon + AdamW, served KL, WSD decay from step 6) against a run resumed from its step-4 state: the step-8 heads
# must be identical (model, both optimizers, RNG, data order incl. prefetched windows restored), and the kept stable-phase
# state must exist. Scratch stays in a mktemp dir.
set -e
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); PY=${PY:-python3}; T=$(mktemp -d)
export MTP_KIT=${MTP_KIT:-/workspace/arc3-inference/mtp_ft_20261011} PYTHONDONTWRITEBYTECODE=1
$PY - $T <<'PYEOF'
import sys, torch
from safetensors.torch import save_file
d = sys.argv[1]
for i in range(4):
    n = 96
    save_file(dict(token_ids=torch.randint(0, 50, (n,)), position_ids=torch.arange(n), served_position_ids=torch.arange(n),
                   mtp_hidden_states=torch.randn(n, 32)), '%s/arc3-persistent-arc3-%032x-x.safetensors' % (d, i))
PYEOF
COMMON="--tiny --device cpu --attention reference --data $T --mtp-file none --target-dir none --steps 8 --eval-every 0 --window 32 \
  --val-files 1 --lr 1e-3 --warmup 2 --decay-start 6 --loss served_kl --optimizer muon --prefetch 2 --save-at 8 --ckpt-every 4"
CUDA_VISIBLE_DEVICES="" $PY $D/train_mtp_sp.py $COMMON --out $T/full > $T/out_full.txt 2>&1 || { cat $T/out_full.txt; exit 1; }
CUDA_VISIBLE_DEVICES="" $PY $D/train_mtp_sp.py $COMMON --out $T/resumed --resume $T/full/state_a.pt > $T/out_res.txt 2>&1 || { cat $T/out_res.txt; exit 1; }
grep -h "resumed from\|saved state" $T/out_full.txt $T/out_res.txt
$PY - $T <<'PYEOF'
import sys, torch
from safetensors.torch import load_file
d = sys.argv[1]
a, b = load_file(d + '/full/head_step8.safetensors'), load_file(d + '/resumed/head_step8.safetensors')
assert set(a) == set(b)
diff = max(float((a[k].float() - b[k].float()).abs().max()) for k in a)
import os
assert os.path.exists(d + '/full/state_stable_step5.pt'), 'stable-phase state missing'
assert diff == 0.0, diff
print('resume test: step-8 heads identical over %d tensors (max diff %.1e); stable-phase state kept: ok' % (len(a), diff))
PYEOF
echo "scratch left in $T (no deletions on the pod)"
