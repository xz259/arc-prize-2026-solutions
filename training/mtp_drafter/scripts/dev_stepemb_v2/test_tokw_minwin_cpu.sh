#!/bin/bash
# CPU test of train_mtp_sp.py --min-window and --token-weighted (2026-10-13, Claude fork 5d20b23b; user 10-13: drop the
# segment-tail stubs, weigh every trained position of a step the same):
#   1. window_bounds: a 65,536-row segment at 16K windows, K 4 ends in a 20-row window; --min-window 2048 drops it and
#      keeps the label coverage of the other windows; Windows and StreamWindows cut alike;
#   2. token weighting on the tiny model: per-window backward with norm + token_scale_grads equals the gradient of
#      (sum of the windows' position losses) / (their positions), and differs from the per-window mean when the windows'
#      position counts differ;
#   3. the same with two data-parallel ranks (gloo, one window each) equals the single process;
#   4. trainer runs (tiny, accum 2): stream and files, --token-weighted --min-window: finite losses, ntok logged, the
#      dropped windows never trained (window counts), every segment consumed.
# Only the test's own temporary directory is touched.
set -e
D=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd); PY=${PY:-python3}; T=$(mktemp -d)
export MTP_KIT=${MTP_KIT:-/workspace/arc3-inference/mtp_ft_20261011} PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2
cat > $T/tw.py <<'PYEOF'
import os, sys, torch
sys.path.insert(0, os.environ['TEST_D'])
import train_mtp_sp as TM
import test_qwen38_mtp_cpu as K


def windows(seed=3):
    g = torch.Generator().manual_seed(seed)
    out = []
    for L, dens in ((24, 0.9), (16, 0.25), (20, 0.6)):
        h = torch.randn(L, K.TINY.hc_count * K.TINY.hidden_size, generator=g)
        tok = torch.randint(0, K.TINY.vocab_size, (L,), generator=g)
        pos = torch.arange(L) * 2 + 5
        mask = (torch.rand(L, generator=g) < dens).long()
        out.append((h, tok, pos, mask))
    return out


def setup():
    m = K.tiny_model(seed=1)
    sub, t2d = TM.load_subset('tiny', m.lm_head.weight.shape[0], 'cpu')
    return m, sub, t2d, m.lm_head.weight.index_select(0, sub).contiguous().detach()     # as in the trainer (frozen head)


def win_loss(m, wsub, t2d, w, norm):
    h, tok, pos, mask = w
    s = m.forward_steps(h, tok, pos, num_steps=3, attention='reference')
    return TM.step_losses_served(m, s, h, tok, mask, [1.0, 0.7, 0.4], 'served_kl', wsub, t2d, norm=norm)


def grads(m):
    return {n: p.grad.detach().clone() for n, p in m.named_parameters() if p.grad is not None}
PYEOF
export TEST_D=$D
$PY - $T <<'PYEOF'
import os, sys, torch
sys.path.insert(0, sys.argv[1])
from tw import TM, K, windows, setup, win_loss, grads
# 1. window bounds
b0, b1 = TM.window_bounds(65536, 16384, 4), TM.window_bounds(65536, 16384, 4, 2048)
assert b0[-1] == (65516, 65536) and len(b0) == 5 and b1 == b0[:4], (b0, b1)
cover = lambda bs: sorted(t for a, b in bs for t in range(a, b - 5))
assert cover(b1) == list(range(0, 65516)), 'positions of the kept windows not covered once'   # T - K - 1 per window
assert TM.window_bounds(1500, 16384, 4, 2048) == [] and TM.window_bounds(3000, 16384, 4, 2048) == [(0, 3000)]
print('  window_bounds: 65,536 rows -> 5 windows (last 20 rows), 4 with --min-window 2048; positions 0..65,515 covered once (the stub had 15)')
# 2. token weighting vs the direct reference
m, sub, t2d, wsub = setup()
ws = windows()
params = [p for p in m.parameters() if p.requires_grad]
m.zero_grad(); tot, n = 0.0, 0.0
for w in ws:
    l, met = win_loss(m, wsub, t2d, w, 1.0); tot = tot + l; n += met['_n']
(tot / n).backward(); ref = grads(m)
m.zero_grad(); nt = 0.0
for w in ws:
    l, met = win_loss(m, wsub, t2d, w, 32.0); l.backward(); nt += met['_n']
TM.token_scale_grads(params, nt, 32.0, 1); got = grads(m)
err = max(float((got[k] - ref[k]).abs().max() / ref[k].abs().max().clamp_min(1e-12)) for k in ref)
assert err < 1e-5, err
m.zero_grad()
for w in ws:
    l, _ = win_loss(m, wsub, t2d, w, None); (l / len(ws)).backward()
old = grads(m)
diff = max(float((old[k] - ref[k]).abs().max() / ref[k].abs().max().clamp_min(1e-12)) for k in ref)
assert diff > 1e-3, diff
print('  token weighting: equals the direct token-weighted gradient (max rel err %.1e); per-window mean differs (%.2f)' % (err, diff))
torch.save(dict(ref=ref, n=n), os.path.join(sys.argv[1], 'ref.pt'))
PYEOF
cat > $T/ddp.py <<'PYEOF'
import os, sys, torch
import torch.distributed as dist
import torch.multiprocessing as mp
sys.path.insert(0, sys.argv[1])


def run(rank, T):
    os.environ.update(MASTER_ADDR='127.0.0.1', MASTER_PORT=os.environ['TEST_PORT'])
    dist.init_process_group('gloo', rank=rank, world_size=2)
    from tw import TM, windows, setup, win_loss, grads
    m, sub, t2d, wsub = setup()
    ws = windows()
    mine = ws[:2] if rank == 0 else ws[2:]               # uneven on purpose: rank 0 two windows, rank 1 one
    params = [p for p in m.parameters() if p.requires_grad]
    m.zero_grad(); nt = 0.0
    for w in mine:
        l, met = win_loss(m, wsub, t2d, w, 32.0); l.backward(); nt += met['_n']
    n = TM.token_scale_grads(params, nt, 32.0, 2)
    if rank == 0:
        r = torch.load(os.path.join(T, 'ref.pt'))
        got = grads(m)
        err = max(float((got[k] - r['ref'][k]).abs().max() / r['ref'][k].abs().max().clamp_min(1e-12)) for k in r['ref'])
        assert abs(n - r['n']) < 1e-6 and err < 1e-5, (n, r['n'], err)
        print('  two ranks (2 + 1 windows): equal to the single process (max rel err %.1e, positions %.0f)' % (err, n))
    dist.destroy_process_group()


if __name__ == '__main__':
    mp.spawn(run, args=(sys.argv[1],), nprocs=2)
PYEOF
TEST_PORT=$((29600 + RANDOM % 300)) $PY $T/ddp.py $T
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
$PY $T/writer.py $T/files; rm -f $T/files/DONE
C="--tiny --device cpu --attention reference --mtp-file none --target-dir none --eval-every 0 --window 32 --accum 2 --warmup 0 \
  --lr 1e-2 --loss served_kl --save-every 0 --stream-pool 1 --train noexperts --token-weighted"
# 64-row files at window 32, K 3: windows (0,32) (28,60) (56,64); --min-window 16 drops the 8-row tails
$PY $D/train_mtp_sp.py $C --data $T/files --val-files 1 --steps 3 --out $T/run_files --min-window 16 > $T/files.txt 2>&1 || { cat $T/files.txt; exit 1; }
grep -q "train windows 6 (3 files)" $T/files.txt || { grep "train windows" $T/files.txt; echo "file windows not filtered"; exit 1; }
$PY $D/train_mtp_sp.py $C --data $T/val --val-files 1 --steps 100 --out $T/run_stream --min-window 16 \
  --stream-dir $T/s --stream-done $T/s/DONE > $T/stream.txt 2>&1 < /dev/null &
P=$!; $PY $T/writer.py $T/s; wait $P || { cat $T/stream.txt; exit 1; }
$PY - $T <<'PYEOF'
import json, math, os, sys
T = sys.argv[1]
for r, want in (('run_files', 3), ('run_stream', 4)):
    rows = [json.loads(l) for l in open('%s/%s/log.jsonl' % (T, r)) if '"loss"' in l]
    assert len(rows) == want and all(math.isfinite(x['loss']) and x['ntok'] > 0 for x in rows), (r, rows)
left = [n for n in os.listdir(T + '/s') if n.endswith('.safetensors')]
assert not left, left
print('  runs: files 6 windows of 9 kept (3 steps), stream 8 windows in 4 steps, every segment consumed; ntok logged '
      '(e.g. %.0f), losses finite' % rows[0]['ntok'])
PYEOF
echo "token-weighting / min-window test ok; scratch left in $T"
