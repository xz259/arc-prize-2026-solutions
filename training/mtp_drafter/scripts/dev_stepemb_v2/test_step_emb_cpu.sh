#!/bin/bash
# CPU test (2026-10-14, Claude fork 5d20b23b): form A step embeddings (step_emb.py; train_mtp_sp.py --step-emb /
# --trunk-lr-scale / --init-from; mtp_eval_val50.py heads with step embeddings). Tiny model, K 4, Muon, served KL,
# per-step activation checkpoints (the full-run recipe's unroll). Checks:
#  1. without --step-emb the dev trainer's losses equal the deployed trainer's bit for bit (the step-aware unroll is a no-op);
#  2. with --step-emb the first step's loss equals (zero init), and the head carries mtp.step_emb [3, H], every row trained;
#  3. --trunk-lr-scale 0 leaves every trunk tensor at its initial value while the step embeddings train;
#  4. --init-from a state / a head restores the tensors (step embeddings included); a step-embedding state into a model
#     without them is refused;
#  5. the scorer's loading: step embeddings installed before the non-strict load and filled; fuse_k adds s_k for steps
#     2-4 only (step 1 and steps beyond the trained rows unchanged).
# Only the test's own temporary directory is touched.
set -e
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); P=$(dirname "$D"); PY=${PY:-python3}; T=$(mktemp -d)
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
C="--tiny --device cpu --attention reference --mtp-file none --target-dir none --eval-every 0 --window 32 --accum 2 --lr 1e-2 \
  --loss served_kl --train nonexpert --token-weighted --num-steps 4 --step-weights 1,0.69,0.40,0.18 --optimizer muon --ckpt-steps \
  --data $T/files --val-files 0 --steps 3 --warmup 0 --save-every 3 --save-trainable-only --ckpt-every 3"
run() { local dir=$1 name=$2; shift 2; $PY $dir/train_mtp_sp.py $C --out $T/$name "$@" > $T/$name.txt 2>&1 || { cat $T/$name.txt; exit 1; }; }
run $P prod
run $D base
run $D emb --step-emb
run $D held --step-emb --trunk-lr-scale 0
S=$(ls $T/held/state_*.pt | head -1)
$PY - $T $D $S <<'PYEOF'
import json, os, sys, torch
T, D, S = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, D); sys.path.insert(0, os.environ['MTP_KIT'])
from safetensors.torch import load_file
loss = lambda n: [r['loss'] for r in map(json.loads, open('%s/%s/log.jsonl' % (T, n))) if 'loss' in r]
prod, base, emb = loss('prod'), loss('base'), loss('emb')
assert len(prod) == 3 and prod == base, (prod, base)
assert emb[0] == base[0] and emb[1:] != base[1:], (emb, base)
print('  1-2. default path = deployed trainer (losses %s); step embeddings: same first loss, then they train' % [round(x, 4) for x in base])
he, hh = load_file(T + '/emb/head_step3.safetensors'), load_file(T + '/held/head_step3.safetensors')
se = he['mtp.step_emb']
assert se.shape[0] == 3 and all(float(r.abs().sum()) > 0 for r in se), se
import test_qwen38_mtp_cpu as TQ
import train_mtp_sp as TR
import step_emb as SE
import qwen38_mtp as Q
init = dict(TQ.tiny_model().named_parameters())
trunk = [n for n in hh if n != 'mtp.step_emb']
assert trunk and all(torch.equal(hh[n], init[n[4:]].detach().to(torch.bfloat16)) for n in trunk), \
    [n for n in trunk if not torch.equal(hh[n], init[n[4:]].detach().to(torch.bfloat16))][:4]
assert float(hh['mtp.step_emb'].abs().sum()) > 0
print('  3. --trunk-lr-scale 0: %d trunk tensors unchanged, step embeddings trained' % len(trunk))
st = torch.load(S, map_location='cpu', weights_only=False)
m = TQ.tiny_model(seed=5); SE.install(m, 3)
n = TR.init_weights(m, S)
named = dict(m.named_parameters())
assert n == len(st['model']) and all(torch.equal(named[k], v) for k, v in st['model'].items())
m2 = TQ.tiny_model(seed=5); SE.install(m2, 3)
TR.init_weights(m2, T + '/held/head_step3.safetensors')
assert torch.equal(m2.step_emb.detach().to(torch.bfloat16), hh['mtp.step_emb'])
try:
    TR.init_weights(TQ.tiny_model(seed=5), S); raise AssertionError('a step-embedding state loaded into a model without them')
except SystemExit as e:
    assert 'step embeddings' in str(e), e
print('  4. --init-from: state (%d tensors) and head restored; refused without --step-emb' % n)
ms = TQ.tiny_model(seed=7).to(torch.bfloat16)
rows = SE.head_rows(T + '/held/head_step3.safetensors')
SE.install(ms, rows, dtype=torch.bfloat16)
filled = Q.load_from_checkpoint(ms, T + '/held/head_step3.safetensors', None, strict=False)
assert rows == 3 and 'step_emb' in filled and torch.equal(ms.step_emb.detach(), hh['mtp.step_emb'])
h = torch.randn(5, 32).to(torch.bfloat16); tok = torch.randint(0, 50, (5,))
with torch.no_grad():
    base_x = ms.fuse(h, tok)
    assert torch.equal(SE.fuse_k(ms, h, tok, 1), base_x) and torch.equal(SE.fuse_k(ms, h, tok, 5), base_x)
    for k in (2, 3, 4):
        d = (SE.fuse_k(ms, h, tok, k) - base_x).float().unflatten(-1, (2, 16))
        want = ms.step_emb[k - 2].float()
        assert torch.allclose(d, want.expand_as(d), atol=0.02), k
print('  5. scorer loading: step embeddings filled; fuse_k adds s_k on every stream for steps 2-4 only')
PYEOF
echo "step-embedding test ok; scratch left in $T"
