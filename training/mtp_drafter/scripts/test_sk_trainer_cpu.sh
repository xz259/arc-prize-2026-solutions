#!/bin/bash
# test_sk_trainer_cpu.sh (2026-10-22, Claude main session 099768dc; the step-conditioned round's CPU prep, V67_DESIGN 262):
# the refit trainer with the eval-time ||s_k|| log (IE: growth vs plateau at every eval). Tiny model, K 4, Muon, the round's
# recipe shape (LK, accum 1, --step-emb --trunk-lr-scale 0). Checks:
#  1. without --step-emb the new trainer's losses equal the deployed trainer's (DEPLOYED, default the refit's scripts/) bit for bit;
#  2. with --step-emb --trunk-lr-scale 0 and evals: every eval line carries val_stepemb_rms [3]; the step-0 eval reads 0
#     (zero init) and the last one > 0; the trunk tensors stay at their initial values; s_k trains;
#  3. the step-0 eval of the step-embedding run equals the plain run's step-0 eval (zero init = the base head).
# Usage: bash test_sk_trainer_cpu.sh NEW_TRAINER_DIR (holding train_mtp_sp.py + step_emb.py). Only a temporary directory is written.
set -e
N=${1:?dir with the new train_mtp_sp.py}; PY=${PY:-/workspace/arc3-inference/venv-speculators/bin/python}
DEPLOYED=${DEPLOYED:-/mnt/shared/arc3-inference/mtp_refit_20261022/scripts}
T=$(mktemp -d)
export MTP_KIT=${MTP_KIT:-/workspace/arc3-inference/mtp_ft_20261011} PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2
$PY - $T/files $T/val <<'PYEOF'
import os, sys, torch
from safetensors.torch import save_file
for d, n0, n in ((sys.argv[1], 0, 4), (sys.argv[2], 10, 1)):
    os.makedirs(d, exist_ok=True)
    for i in range(n0, n0 + n):
        g = torch.Generator().manual_seed(i); rows = 64
        save_file(dict(token_ids=torch.randint(0, 50, (rows,), generator=g), position_ids=torch.arange(rows) + 1000 * i,
                       served_position_ids=torch.arange(rows), mtp_hidden_states=torch.randn(rows, 32, generator=g)),
                  os.path.join(d, 'arc3-persistent-arc3-%032x-x_s%04d.safetensors' % (i // 2, i % 2)))
PYEOF
C="--tiny --device cpu --attention reference --mtp-file none --target-dir none --window 32 --accum 1 --lr 1e-3 \
  --loss lk --cooldown-loss lk --train nonexpert --token-weighted --num-steps 4 --step-weights 1,0.69,0.40,0.18 --optimizer muon \
  --ckpt-steps --steps 4 --warmup 0 --save-every 4 --save-trainable-only --ckpt-every 4"
run() { local dir=$1 name=$2; shift 2; $PY $dir/train_mtp_sp.py $C --out $T/$name "$@" > $T/$name.txt 2>&1 || { tail -n 30 $T/$name.txt; exit 1; }; }
run $DEPLOYED prod --data $T/files --val-files 0 --eval-every 0
run $N new --data $T/files --val-files 0 --eval-every 0
run $N plainev --data $T/val $T/files --val-files 1 --eval-every 2
run $N sk --data $T/val $T/files --val-files 1 --eval-every 2 --step-emb --step-emb-lr-scale 10 --trunk-lr-scale 0
$PY - $T <<'PYEOF'
import json, os, sys, torch
T = sys.argv[1]
sys.path.insert(0, os.environ['MTP_KIT'])
from safetensors.torch import load_file
rows = lambda n: [json.loads(l) for l in open('%s/%s/log.jsonl' % (T, n))]
loss = lambda n: [r['loss'] for r in rows(n) if 'loss' in r]
prod, new = loss('prod'), loss('new')
assert len(prod) == 4 and prod == new, (prod, new)
print('  1. default path = the deployed trainer bit for bit (losses %s)' % [round(x, 5) for x in new])
ev = [r for r in rows('sk') if any(k.startswith('val_') for k in r)]
pv = [r for r in rows('plainev') if any(k.startswith('val_') for k in r)]
assert ev and all(len(r.get('val_stepemb_rms', [])) == 3 for r in ev), ev
assert not any('val_stepemb_rms' in r for r in pv)
assert ev[0]['step'] == 0 and ev[0]['val_stepemb_rms'] == [0.0, 0.0, 0.0], ev[0]
assert all(x > 0 for x in ev[-1]['val_stepemb_rms']), ev[-1]
print('  2. evals at steps %s carry val_stepemb_rms; step 0 %s, last %s' % ([r['step'] for r in ev], ev[0]['val_stepemb_rms'],
                                                                         [round(x, 6) for x in ev[-1]['val_stepemb_rms']]))
import test_qwen38_mtp_cpu as TQ
init = dict(TQ.tiny_model().named_parameters())
hh = load_file(T + '/sk/head_step4.safetensors')
trunk = [n for n in hh if n != 'mtp.step_emb']
assert trunk and all(torch.equal(hh[n], init[n[4:]].detach().to(torch.bfloat16)) for n in trunk)
assert float(hh['mtp.step_emb'].abs().sum()) > 0
print('     --trunk-lr-scale 0: %d trunk tensors unchanged; s_k trained' % len(trunk))
k0 = [k for k in ev[0] if k.startswith('val_') and k != 'val_stepemb_rms']
same = [k for k in k0 if ev[0][k] == pv[0].get(k)]
assert pv[0]['step'] == 0 and same == k0, [(k, ev[0][k], pv[0].get(k)) for k in k0 if ev[0][k] != pv[0].get(k)][:4]
print('  3. step-0 eval with zero s_k = the plain head\'s step-0 eval (%d metrics equal)' % len(k0))
PYEOF
echo "sk trainer test ok; scratch left in $T"
