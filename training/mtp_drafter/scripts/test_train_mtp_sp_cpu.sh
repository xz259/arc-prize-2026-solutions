#!/bin/bash
# CPU smoke test of train_mtp_sp.py's 10-13 options on the kit's tiny random model and synthetic MTP export files
# (Claude fork 5d20b23b): (1) the kit's defaults (cekl), (2) served_kl with accepted-path weights, accumulation 2,
# experts frozen (router at 0.3x), an LK-hybrid cooldown over the last half; (3) served_kl with Muon. Scratch stays in a
# mktemp dir.
set -e
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); PY=${PY:-python3}; T=$(mktemp -d)
export MTP_KIT=${MTP_KIT:-/workspace/arc3-inference/mtp_ft_20261011} PYTHONDONTWRITEBYTECODE=1
$PY - $T <<'PYEOF'
import sys, torch
from safetensors.torch import save_file
d = sys.argv[1]
for i in range(4):
    n = 64
    extra = dict(mtp_hidden_states=torch.randn(n, 32)) if i % 2 else dict(hidden_states=torch.randn(n, 3, 16))
    save_file(dict(token_ids=torch.randint(0, 50, (n,)), position_ids=torch.arange(n), served_position_ids=torch.arange(n) * 2,
                   **extra), '%s/arc3-persistent-arc3-%032x-x.safetensors' % (d, i))
PYEOF
COMMON="--tiny --device cpu --attention reference --data $T --mtp-file none --target-dir none --steps 6 --eval-every 3 --save-every 6 --window 32 --val-files 1 --lr 1e-3 --warmup 2"
CUDA_VISIBLE_DEVICES="" $PY $D/train_mtp_sp.py $COMMON --out $T/run_cekl > $T/out_cekl.txt 2>&1 || { cat $T/out_cekl.txt; exit 1; }
CUDA_VISIBLE_DEVICES="" $PY $D/train_mtp_sp.py $COMMON --out $T/run_srv --loss served_kl --accept-weights --accum 2 \
  --train noexperts --router-lr-scale 0.3 --cooldown-loss lk --decay-ratio 0.5 --step-weights 1,0.6,0.26 --save-at 6 --save-trainable-only --prefetch 2 --ckpt-steps > $T/out_srv.txt 2>&1 || { cat $T/out_srv.txt; exit 1; }
CUDA_VISIBLE_DEVICES="" $PY $D/train_mtp_sp.py $COMMON --out $T/run_muon --loss served_kl --optimizer muon --step-weights 1,0.6,0.26 \
  > $T/out_muon.txt 2>&1 || { cat $T/out_muon.txt; exit 1; }
grep -h '^Muon:' $T/out_muon.txt
$PY - $T <<'PYEOF'
import json, os, sys
d = sys.argv[1]
for run in ('run_cekl', 'run_srv', 'run_muon'):
    rows = [json.loads(l) for l in open('%s/%s/log.jsonl' % (d, run))]
    tr = [r for r in rows if 'loss' in r]; va = [r for r in rows if 'val_tau_est' in r]
    assert len(tr) == 6 and len(va) == 3 and os.path.exists('%s/%s/head_step6.safetensors' % (d, run)), (run, len(tr), len(va))
    print('%s: kinds %s, loss %.3f -> %.3f, val tau_est %.3f -> %.3f, tau_srv %.3f, mout_step1 %.3f, load_entropy %.3f' % (
        run, sorted({r['kind'] for r in tr}), tr[0]['loss'], tr[-1]['loss'], va[0]['val_tau_est'], va[-1]['val_tau_est'],
        va[-1]['val_tau_srv'], va[-1]['val_mout_step1'], tr[-1]['load_entropy']))
    if run == 'run_srv':
        assert {r['kind'] for r in tr} == {'served_kl', 'lk'}, 'cooldown switch'
        assert all(0 <= r['w_step3'] <= r['w_step2'] <= r['w_step1'] + 1e-9 for r in tr), 'accepted-path weights'
        print('  w_step1..3 (last step): %.3f %.3f %.3f; alpha_step1 %.3f; kl_step1 %.3f' % (
            tr[-1]['w_step1'], tr[-1]['w_step2'], tr[-1]['w_step3'], tr[-1]['alpha_step1'], tr[-1]['kl_step1']))
import torch
from safetensors.torch import load_file
a = load_file(d + '/run_srv/head_step6.safetensors')
print('noexperts head: %d tensors saved' % len(a))
PYEOF
echo "smoke ok; scratch left in $T (no deletions on the pod)"
