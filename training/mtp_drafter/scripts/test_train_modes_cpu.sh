#!/bin/bash
# CPU test of train_mtp_sp.py --train modes (2026-10-13, Claude fork 5d20b23b; the experts-only ablation): (1) apply_train_mode
# on the tiny model gives the expected trainable sets for all / noexperts / nonexpert / experts; (2) a 2-step run with
# --train experts changes only the MoE block (routed experts, router, shared expert) against a 2-step run at LR 0 (the
# initial weights), while a 2-step --train all run also changes attention. Only the test's own temporary directories are
# touched.
set -e
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); PY=${PY:-python3}; T=$(mktemp -d)
export MTP_KIT=${MTP_KIT:-/workspace/arc3-inference/mtp_ft_20261011} PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2
$PY - $D <<'PYEOF'
import sys
sys.path.insert(0, sys.argv[1])
import train_mtp_sp as T
import test_qwen38_mtp_cpu as K
def trainable(mode):
    m = K.tiny_model(); T.apply_train_mode(m, mode, tiny=True)
    return {n for n, p in m.named_parameters() if p.requires_grad}
base = trainable('all')
ex = trainable('experts')
assert ex and all(n.startswith('layers.0.mlp.') for n in ex), ex
assert {'layers.0.mlp.w_gate', 'layers.0.mlp.w_up', 'layers.0.mlp.w_down', 'layers.0.mlp.gate.weight'} <= ex, ex
assert any('shared_expert.' in n for n in ex) and 'layers.0.mlp.shared_expert_gate.weight' in ex, ex
assert ex == {n for n in base if n.startswith('layers.0.mlp.')}, (ex, base)
ne = trainable('noexperts'); assert not any(n.startswith('layers.0.mlp.w_') for n in ne) and 'layers.0.mlp.gate.weight' in ne
nn = trainable('nonexpert'); assert not any(n.startswith('layers.0.mlp.w_') for n in nn) and 'layers.0.mlp.gate.weight' not in nn
assert 'layers.0.self_attn.q_proj.weight' in base and 'layers.0.self_attn.q_proj.weight' not in ex
print('  modes: all %d, experts %d, noexperts %d, nonexpert %d trainable tensors' % (len(base), len(ex), len(ne), len(nn)))
PYEOF
cat > $T/writer.py <<'PYEOF'
import os, sys, torch
from safetensors.torch import save_file
d = sys.argv[1]; os.makedirs(d, exist_ok=True)
for i in range(4):
    g = torch.Generator().manual_seed(i); rows = 64
    save_file(dict(token_ids=torch.randint(0, 50, (rows,), generator=g), position_ids=torch.arange(rows) + 1000 * i,
                   served_position_ids=torch.arange(rows), mtp_hidden_states=torch.randn(rows, 32, generator=g)),
              os.path.join(d, 'arc3-persistent-arc3-%032x-x_s%04d.safetensors' % (i // 2, i % 2)))
open(os.path.join(d, 'DONE'), 'w').write('done\n')
PYEOF
mkdir -p $T/val; $PY $T/writer.py $T/val; rm -f $T/val/DONE $T/val/*_s0001.safetensors
COMMON="--tiny --device cpu --attention reference --data $T/val --val-files 1 --mtp-file none --target-dir none --eval-every 0 \
  --window 32 --accum 2 --warmup 0 --loss served_kl --save-every 0 --save-at 2 --steps 2 --stream-pool 1"
for RUNDEF in "zero all 0" "ex experts 1e-2" "all all 1e-2"; do
  read -r NAME MODE LR <<< "$RUNDEF"
  $PY $T/writer.py $T/s_$NAME
  $PY $D/train_mtp_sp.py $COMMON --train $MODE --lr $LR --out $T/run_$NAME --stream-dir $T/s_$NAME --stream-done $T/s_$NAME/DONE > $T/out_$NAME.txt 2>&1 \
    || { cat $T/out_$NAME.txt; exit 1; }
done
$PY - $T <<'PYEOF'
import sys
from safetensors.torch import load_file
T = sys.argv[1]
z, e, a = (load_file('%s/run_%s/head_step2.safetensors' % (T, n)) for n in ('zero', 'ex', 'all'))
diff = lambda x, n: float((x[n].float() - z[n].float()).abs().max())
moe = [n for n in z if '.mlp.' in n and 'mlp_hyper' not in n]
attn = [n for n in z if 'self_attn' in n and 'indexer' not in n]
assert moe and attn, (list(z)[:10])
assert all(diff(e, n) == 0 for n in attn), [(n, diff(e, n)) for n in attn]
assert any(diff(e, n) > 0 for n in moe), [(n, diff(e, n)) for n in moe]
other = [n for n in z if n not in moe and 'embed' not in n and 'lm_head' not in n]
assert all(diff(e, n) == 0 for n in other), [(n, diff(e, n)) for n in other if diff(e, n) > 0]
assert any(diff(a, n) > 0 for n in attn), 'all-parameter run left attention unchanged'
print('  experts run: %d MoE tensors changed, %d attention and %d other tensors unchanged; all run changed attention' % (
    sum(diff(e, n) > 0 for n in moe), len(attn), len(other)))
PYEOF
echo "train-modes test ok; scratch left in $T"
