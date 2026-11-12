#!/bin/bash
# CPU test (2026-10-14, Claude fork 5d20b23b): state_to_head.py. A tiny frozen-experts run saves trainable-only heads and
# kept states at steps 2 and 4; the head converted from state_keep_step4 must equal the trainer's head_step4 bit for bit
# (same names, same BF16 values); the average of the two states must equal the BF16 of the FP32 mean; mismatched states
# are refused. Only the test's own temporary directory is touched.
set -e
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); PY=${PY:-python3}; T=$(mktemp -d)
export MTP_KIT=${MTP_KIT:-/workspace/arc3-inference/mtp_ft_20261011} PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2
$PY - $T/files <<'PYEOF'
import os, sys, torch
from safetensors.torch import save_file
d = sys.argv[1]; os.makedirs(d, exist_ok=True)
for i in range(4):
    g = torch.Generator().manual_seed(i); rows = 64
    save_file(dict(token_ids=torch.randint(0, 50, (rows,), generator=g), position_ids=torch.arange(rows) + 1000 * i,
                   served_position_ids=torch.arange(rows), mtp_hidden_states=torch.randn(rows, 32, generator=g)),
              os.path.join(d, 'arc3-persistent-arc3-%032x-x_s%04d.safetensors' % (i // 2, i % 2)))
PYEOF
$PY $D/train_mtp_sp.py --tiny --device cpu --attention reference --mtp-file none --target-dir none --eval-every 0 --window 32 \
  --accum 2 --lr 1e-2 --loss served_kl --train nonexpert --token-weighted --data $T/files --val-files 0 --steps 4 --warmup 0 \
  --ckpt-every 2 --keep-every 2 --save-every 2 --save-trainable-only --out $T/run > $T/train.txt 2>&1 || { cat $T/train.txt; exit 1; }
for f in head_step2.safetensors head_step4.safetensors state_keep_step2.pt state_keep_step4.pt; do
  [ -s $T/run/$f ] || { cat $T/train.txt; echo "missing $f"; exit 1; }
done
$PY $D/state_to_head.py $T/one.safetensors $T/run/state_keep_step4.pt
$PY $D/state_to_head.py $T/avg.safetensors $T/run/state_keep_step2.pt $T/run/state_keep_step4.pt
$PY - $T <<'PYEOF'
import sys, torch
from safetensors.torch import load_file
T = sys.argv[1]
ref, one, avg = load_file(T + '/run/head_step4.safetensors'), load_file(T + '/one.safetensors'), load_file(T + '/avg.safetensors')
assert sorted(ref) == sorted(one), (sorted(set(ref) ^ set(one)))
assert all(torch.equal(ref[n], one[n]) for n in ref), [n for n in ref if not torch.equal(ref[n], one[n])][:4]
a = torch.load(T + '/run/state_keep_step2.pt', map_location='cpu', weights_only=False)['model']
b = torch.load(T + '/run/state_keep_step4.pt', map_location='cpu', weights_only=False)['model']
for n in avg:
    k = n[len('mtp.'):]
    want = ((a[k].double() + b[k].double()) / 2).to(torch.bfloat16)
    assert torch.equal(avg[n], want), n
moved = sum(int(not torch.equal(a[n[4:]], b[n[4:]])) for n in avg)
assert moved > 0, 'the two states are identical: the test would not see an average'
print('  head from state = trainer head (%d tensors, bitwise); average of 2 states exact (%d tensors moved)' % (len(ref), moved))
st = torch.load(T + '/run/state_keep_step2.pt', map_location='cpu', weights_only=False)
st['meta'] = dict(st.get('meta', {}), expert_weights='other'); torch.save(st, T + '/bad.pt')
PYEOF
if $PY $D/state_to_head.py $T/bad.safetensors $T/run/state_keep_step4.pt $T/bad.pt > $T/bad.txt 2>&1; then
  echo "a state with different meta was averaged"; exit 1
fi
grep -q "different state meta" $T/bad.txt || { cat $T/bad.txt; exit 1; }
echo "state-to-head test ok; scratch left in $T"
