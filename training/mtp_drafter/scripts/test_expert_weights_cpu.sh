#!/bin/bash
# CPU test of train_mtp_sp.py --expert-weights served and --unfreeze-experts (2026-10-13, Claude fork 5d20b23b; user 10-13:
# the frozen routed experts run forward at their test-time weights; the unfreeze escape hatch in reserve), on the tiny model
# with a served file of its own (NVFP4 by export_mtp_drafter.nvfp4_quantize):
#   1. load_served_experts puts the dequantized served values (BF16-rounded, as mtp_eval_val50.py --experts served) into the
#      frozen experts and touches nothing else; torch's load_state_dict restores a saved lr_scale, load_state re-applies the
#      run's own (the router-guard resume);
#   2. runs: served frozen 4 steps (states at 2, 4); a resume with source experts fails; a resume with --unfreeze-experts
#      (warmup 2, 0.5x) and a halved router scale trains the experts with the warmup and the new router LR; resuming that
#      state without the flag fails, with it continues; resuming the step-4 state without it (revert) trains frozen.
# Only the test's own temporary directory is touched.
set -e
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); PY=${PY:-python3}; T=$(mktemp -d)
export MTP_KIT=${MTP_KIT:-/workspace/arc3-inference/mtp_ft_20261011} PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2
$PY - $D $T <<'PYEOF'
import json, os, sys, torch
sys.path.insert(0, sys.argv[1])
import train_mtp_sp as T
import test_qwen38_mtp_cpu as K
from export_mtp_drafter import nvfp4_quantize, nvfp4_dequantize
from safetensors.torch import save_file
d = os.path.join(sys.argv[2], 'served'); os.makedirs(d)
src = K.tiny_model(seed=7)                          # a different model: the served experts differ from the tiny source
mlp, out, ref = src.layers[0].mlp, {}, {}
for e in range(mlp.cfg.num_experts):
    for proj, attr in (('gate_proj', 'w_gate'), ('up_proj', 'w_up'), ('down_proj', 'w_down')):
        w = getattr(mlp, attr).data[e].float()
        blk = 16 if w.shape[1] % 16 == 0 else 8       # the tiny down_proj has 8 inputs
        pk, sc, s2 = nvfp4_quantize(w, block=blk)
        b = 'mtp.layers.0.mlp.experts.%d.%s.' % (e, proj)
        out.update({b + 'weight': pk, b + 'weight_scale': sc, b + 'weight_scale_2': s2})
        ref[(attr, e)] = nvfp4_dequantize(pk, sc, s2, block=blk).to(torch.bfloat16).float()
save_file(out, os.path.join(d, 'model-mtp-draft.safetensors'))
json.dump({}, open(os.path.join(d, 'config.json'), 'w'))
m = K.tiny_model(); before = {n: p.detach().clone() for n, p in m.named_parameters()}
T.apply_train_mode(m, 'noexperts', tiny=True)
n = T.load_served_experts(m, d)
mm = m.layers[0].mlp
assert n == 3 * mm.cfg.num_experts, n
for (attr, e), r in ref.items():
    assert torch.equal(getattr(mm, attr).data[e].float(), r), (attr, e)
changed = {k for k, p in m.named_parameters() if not torch.equal(p.detach(), before[k])}
assert changed == {'layers.0.mlp.w_gate', 'layers.0.mlp.w_up', 'layers.0.mlp.w_down'}, changed
rel = float((mm.w_gate.data.float() - before['layers.0.mlp.w_gate']).norm() / before['layers.0.mlp.w_gate'].norm())
print('  load_served_experts: %d matrices equal to the dequantized served values, only the routed experts changed' % n)
# torch restores saved group hyperparameters (the resume bug for --router-lr-scale); load_state re-applies the run's own
p = torch.nn.Parameter(torch.zeros(3))
o1 = torch.optim.AdamW([dict(params=[p], lr_scale=0.3, role='router')], lr=1.0)
p.grad = torch.ones(3); o1.step()
o2 = torch.optim.AdamW([dict(params=[p], lr_scale=0.15, role='router')], lr=1.0)
o2.load_state_dict(o1.state_dict())
print('  torch load_state_dict keeps the saved lr_scale: %s' % (o2.param_groups[0]['lr_scale'] == 0.3))
o3 = torch.optim.AdamW([dict(params=[p], lr_scale=0.15, role='router')], lr=1.0)
st = dict(model={}, opts=[o1.state_dict()], rng=dict(python=__import__('random').getstate(), torch=torch.get_rng_state(), cuda=None))
T.load_state(st, torch.nn.Linear(1, 1), [o3])
assert o3.param_groups[0]['lr_scale'] == 0.15 and len(o3.state) == 1, o3.param_groups[0]
print('  load_state: the run\'s lr_scale 0.15 kept, optimizer moments restored')
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
PYEOF
$PY $T/writer.py $T/data
C="--tiny --device cpu --attention reference --data $T/data --val-files 1 --mtp-file none --target-dir none --eval-every 0 \
  --window 32 --accum 2 --warmup 0 --lr 1e-2 --schedule wsd --decay-ratio 0.25 --loss served_kl --save-every 0 --stream-pool 1 \
  --train noexperts --router-lr-scale 0.3 --draft-config $T/served/config.json --ckpt-every 2"
run() { local name=$1; shift; $PY $D/train_mtp_sp.py $C "$@" --out $T/$name > $T/$name.txt 2>&1; }
run a --expert-weights served --steps 4 || { cat $T/a.txt; exit 1; }
cp $T/a/state_b.pt $T/s4.pt                                        # step 4, frozen (served)
run b --expert-weights source --steps 8 --resume $T/s4.pt && { echo "source resume of a served state did not fail"; exit 1; }
grep -q "the state ran with --expert-weights served" $T/b.txt || { cat $T/b.txt; exit 1; }
run c --expert-weights served --steps 8 --resume $T/s4.pt --unfreeze-experts --expert-warmup 2 --expert-lr-scale 0.5 \
  --router-lr-scale 0.15 || { cat $T/c.txt; exit 1; }
cp $T/c/state_a.pt $T/s6.pt                                        # step 6, experts training since step 5
run d --expert-weights served --steps 8 --resume $T/s6.pt && { echo "resume of an unfrozen state without the flag did not fail"; exit 1; }
grep -q "resume with --unfreeze-experts" $T/d.txt || { cat $T/d.txt; exit 1; }
run e --expert-weights served --steps 8 --resume $T/s6.pt --unfreeze-experts --expert-warmup 2 --expert-lr-scale 0.5 \
  --router-lr-scale 0.15 || { cat $T/e.txt; exit 1; }
run f --expert-weights served --steps 8 --resume $T/s4.pt || { cat $T/f.txt; exit 1; }   # revert: frozen again
$PY - $D $T <<'PYEOF'
import json, sys, torch
sys.path.insert(0, sys.argv[1])
import train_mtp_sp as TM
T = sys.argv[2]
rows = lambda r: {x['step']: x for x in map(json.loads, open('%s/%s/log.jsonl' % (T, r))) if 'loss' in x}
a, c, e, f = rows('a'), rows('c'), rows('e'), rows('f')
lr = lambda s: TM.lr_at(s, 8, 1e-2, 0, 'wsd', 0.25)
for s in (5, 6, 7, 8):
    assert abs(c[s]['lr_router'] - 0.15 * lr(s)) < 1e-15, (s, c[s]['lr_router'], lr(s))
    assert abs(c[s]['lr_experts'] - 0.5 * lr(s) * min(1.0, (s - 4) / 2)) < 1e-15, (s, c[s]['lr_experts'])
assert all(abs(a[s]['lr_router'] - 0.3 * TM.lr_at(s, 4, 1e-2, 0, 'wsd', 0.25)) < 1e-15 for s in a)
assert all('lr_experts' not in a[s] for s in a) and all('lr_experts' not in f[s] for s in f), 'experts trained while frozen'
assert set(e) == {7, 8} and all(abs(e[s]['lr_experts'] - c[s]['lr_experts']) < 1e-15 for s in e), (e, c)
s4, s6, s8 = (torch.load('%s/%s' % (T, p), map_location='cpu', weights_only=False) for p in ('s4.pt', 's6.pt', 'c/state_b.pt'))
assert s4['meta'] == dict(train='noexperts', expert_weights='served', unfreeze_at=0), s4['meta']
assert s6['meta']['unfreeze_at'] == 5 and len(s4['opts']) == 1 and len(s6['opts']) == 2, (s6['meta'], len(s4['opts']))
assert not any(k.startswith('layers.0.mlp.w_') for k in s4['model']) and 'layers.0.mlp.w_gate' in s6['model']
assert s6['model']['layers.0.mlp.w_gate'].dtype == torch.float32
assert not torch.equal(s6['model']['layers.0.mlp.w_gate'], s8['model']['layers.0.mlp.w_gate']), 'experts did not train'
print('  runs: served frozen; source resume refused; unfrozen from step 5 (expert LR warmup %.1e -> %.1e, router 0.15x '
      'applied); unfrozen state refused without the flag, continued with it; revert to frozen ok' % (
      c[5]['lr_experts'], c[6]['lr_experts']))
PYEOF
echo "expert-weights test ok; scratch left in $T"
