"""G1 for kvring_kernels (item 4a; KVRING_DESIGN_2026-10-15.md): ring content and addressing, the write mask, rejected-draft rewrites,
wrap-around offsets, negative boundaries, the floor, no-ring requests, the lutc split bitwise against kvlut_kernels.attention, and the
two-source output against a float64 reference built independently (y from the original x in torch; lutc decoded in torch).
Run on a GPU in the runtime: bash runtime_python.sh stage2_v1/test_kvring_kernels.py (cwd = the stage-2 dir)."""
import math
import os
import sys

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.dirname(HERE), '/workspace/arc3-inference']
from stage2_v1 import kvlut_kernels as KL      # noqa: E402
from stage2_v1 import kvring_kernels as KR     # noqa: E402

dev = 'cuda'
torch.manual_seed(1234)
KVH, QH, D, PAGE, N, TOPK = 2, 32, 256, 3104, 256, 2048
G = QH // KVH
FAILS = []


def check(name, ok, detail=''):
    print(('PASS ' if ok else 'FAIL ') + name + (' ' + detail if detail else ''))
    if not ok:
        FAILS.append(name)


def hadamard():
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < 256:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    return h / 16.0                                   # orthonormal, symmetric


HM = hadamard().to(dev)


def rot_y(x, mu):
    """y = H(x - mu) in FP32 like the store kernels, returned as BF16 (the ring's content)."""
    y = (x.float() - mu[None]).double() @ HM.T
    return y.float().to(torch.bfloat16)


LUTP = torch.tensor(KL.LUTP, dtype=torch.float32, device=dev)
LEVELS = torch.cat([-LUTP.flip(0), LUTP])            # codes 0..15


def lutc_decode(cache, slots):
    """Decoded lutc rows (rotated domain) of physical slots: FP32 [n, KVH, 2 sides, 256] -> BF16 like the reader."""
    pages, offs = slots // PAGE, slots % PAGE
    rows = cache[pages, :, offs]                      # [n, KVH, 296] uint8
    out = []
    for side in range(2):
        b = rows[..., side * 148:(side + 1) * 148]
        codes = torch.stack([b[..., :128] & 15, b[..., :128] >> 4], -1).reshape(*b.shape[:-1], 256).long()
        block = b[..., 128:144].contiguous().view(torch.float8_e4m3fn).float()        # [n, KVH, 16]
        head = b[..., 144:148].contiguous().view(torch.float32)                      # [n, KVH, 1]
        val = LEVELS[codes] * block.repeat_interleave(16, -1) * head
        out.append(val.to(torch.bfloat16))
    return out[0], out[1]


def make_request(S, q_len, ring_id, off, floor=-1, draft_rewrite=False):
    return dict(S=S, q_len=q_len, ring_id=ring_id, off=off, floor=floor, draft_rewrite=draft_rewrite)


REQS = [make_request(100, 4, 3, 0),                   # fewer than N rows: everything recent
        make_request(257, 4, 0, 5),                   # boundary 0
        make_request(5000, 4, 1, 37, draft_rewrite=True),
        make_request(7000, 1, 6, 255),                # wrap-around
        make_request(3000, 1024, 2, 129),             # a chunk longer than N (write mask)
        make_request(4000, 4, -1, 0),                 # no ring: all lutc
        make_request(6000, 4, 4, 11, floor=6000 - 1 - 100)]   # floor: only the newest 100 rows from the ring
R = len(REQS)
RING_SLOTS = 8

kmean = torch.randn(KVH, D, device=dev) * 0.3
kmean[:, :64] = 0                                     # K means are zero on the rotary dims
vmean = torch.randn(KVH, D, device=dev) * 0.3
vmean_rot = torch.randn(KVH, D, device=dev) * 0.1     # any vector: the reference adds the same one

pages_needed = [math.ceil(r['S'] / PAGE) for r in REQS]
num_blocks = sum(pages_needed) + 5
cache = torch.zeros(num_blocks, KVH, PAGE, 296, dtype=torch.uint8, device=dev)
perm = torch.randperm(num_blocks, device='cpu').tolist()
width = max(pages_needed) + 1
block_table = torch.full((R, width), -1, dtype=torch.int32, device=dev)
X = []
p = 0
for i, r in enumerate(REQS):
    phys = perm[p:p + pages_needed[i]]
    p += pages_needed[i]
    block_table[i, :len(phys)] = torch.tensor(phys, dtype=torch.int32)
    xk = (torch.randn(r['S'], KVH, D, device=dev) + kmean[None]).to(torch.bfloat16)
    xv = (torch.randn(r['S'], KVH, D, device=dev) + vmean[None]).to(torch.bfloat16)
    X.append((xk, xv))
    rows = torch.arange(r['S'], device=dev)
    slots = (block_table[i, (rows // PAGE)].long() * PAGE + rows % PAGE).to(torch.int64)
    KL.store(xk, xv, cache, slots, kmean.contiguous(), vmean.contiguous())
    r['slots'] = slots

ring_k = torch.full((RING_SLOTS * N, KVH, D), float('nan'), dtype=torch.bfloat16, device=dev)
ring_v = torch.full_like(ring_k, float('nan'))
slot_rows = torch.full((RING_SLOTS * N,), -1, dtype=torch.int32, device=dev)


def write_steps(i, r):
    """The request's history written to the ring in steps: 4,096-row prefill chunks, then the final step's q_len rows; with
    draft_rewrite, a step of 4 garbage rows at the final positions is written first and then rewritten."""
    xk, xv = X[i]
    meta = torch.tensor([[r['ring_id'], r['off'], r['floor']]], dtype=torch.int32, device=dev)
    S, q = r['S'], r['q_len']
    bounds = list(range(0, S - q, 4096)) + [S - q]
    chunks = [(a, min(a + 4096, S - q)) for a in bounds[:-1]] + [(S - q, S)]
    for a, b in chunks:
        if b <= a:
            continue
        if r['draft_rewrite'] and (a, b) == (S - q, S):
            gk, gv = torch.randn_like(xk[a:b].float()).to(torch.bfloat16), torch.randn_like(xv[a:b].float()).to(torch.bfloat16)
            KR.ring_store(gk, gv, ring_k, ring_v, torch.zeros(b - a, dtype=torch.int32, device=dev),
                          torch.tensor([0, b - a], dtype=torch.int32, device=dev), torch.tensor([b], dtype=torch.int32, device=dev),
                          meta, kmean.contiguous(), vmean.contiguous(), n=N, slot_rows=slot_rows)
        KR.ring_store(xk[a:b], xv[a:b], ring_k, ring_v, torch.zeros(b - a, dtype=torch.int32, device=dev),
                      torch.tensor([0, b - a], dtype=torch.int32, device=dev), torch.tensor([b], dtype=torch.int32, device=dev),
                      meta, kmean.contiguous(), vmean.contiguous(), n=N, slot_rows=slot_rows)


for i, r in enumerate(REQS):
    write_steps(i, r)
torch.cuda.synchronize()

# ring content: every window row in its slot, bitwise equal to y computed independently
used = set()
for i, r in enumerate(REQS):
    if r['ring_id'] < 0:
        continue
    xk, xv = X[i]
    lo = max(0, r['S'] - N)
    rows = torch.arange(lo, r['S'], device=dev)
    slots = r['ring_id'] * N + (rows + r['off']) % N
    used.update(slots.tolist())
    yk, yv = rot_y(xk[lo:], kmean), rot_y(xv[lo:], vmean)
    # y here is computed in float64 (the kernel in FP32 Triton dots), so a value may land one BF16 ulp apart; a wrong row or slot
    # would be off by the values' own magnitude
    dk = (ring_k[slots].float() - yk.float()).abs() / (yk.float().abs() + 1e-2)
    dv = (ring_v[slots].float() - yv.float()).abs() / (yv.float().abs() + 1e-2)
    exact = (torch.eq(ring_k[slots], yk).float().mean().item() + torch.eq(ring_v[slots], yv).float().mean().item()) / 2
    check('ring content req %d (S %d, off %d)' % (i, r['S'], r['off']), dk.max().item() < 1e-2 and dv.max().item() < 1e-2,
          'max rel %.3g / %.3g, bitwise-equal share %.5f' % (dk.max().item(), dv.max().item(), exact))
untouched = [s for s in range(RING_SLOTS * N) if s not in used]
check('no writes outside the window slots', bool(torch.isnan(ring_k[untouched].float()).all() and torch.isnan(ring_v[untouched].float()).all()))

# the final step: every request's q_len query rows, attention over selected rows
rows_meta, idx_rows, q_start = [], [], [0]
for i, r in enumerate(REQS):
    for t in range(r['q_len']):
        pos = r['S'] - r['q_len'] + t
        cand = torch.arange(pos + 1, device=dev)
        if pos + 1 > TOPK:
            recent = cand[-64:]
            rest = cand[:-64][torch.randperm(pos + 1 - 64, device=dev)[:TOPK - 64]]
            cand = torch.cat([rest, recent])
        cand = cand[torch.randperm(cand.numel(), device=dev)]
        row = torch.full((TOPK,), -1, dtype=torch.int32, device=dev)
        row[:cand.numel()] = cand.to(torch.int32)
        idx_rows.append(row)
        rows_meta.append((i, pos))
    q_start.append(q_start[-1] + r['q_len'])
indices = torch.stack(idx_rows)
T = indices.shape[0]
token_to_req = torch.tensor([m[0] for m in rows_meta], dtype=torch.int32, device=dev)
seq_lens = torch.tensor([r['S'] for r in REQS], dtype=torch.int32, device=dev)
ring_meta = torch.tensor([[r['ring_id'], r['off'], r['floor']] for r in REQS], dtype=torch.int32, device=dev)
query = torch.randn(T, QH, D, device=dev).to(torch.bfloat16)


def run_ring(rows):
    o = torch.zeros(len(rows), QH, D, dtype=torch.bfloat16, device=dev)          # production: BF16 attention output
    KR.attention(query[rows], cache, indices[rows], block_table, token_to_req[rows], seq_lens, ring_k, ring_v, ring_meta, o,
                 vmean=vmean_rot, n=N)
    return o


def run_plain(rows, idx):
    o = torch.zeros(len(rows), QH, D, dtype=torch.bfloat16, device=dev)
    KL.attention(query[rows], cache, idx[rows], block_table, token_to_req[rows], o, vmean=vmean_rot)
    return o


def boundary(i):
    r = REQS[i]
    return max(r['S'] - 1 - N, r['floor']) if r['ring_id'] >= 0 else 1 << 40


ALL = torch.arange(T, device=dev)
DECODE = torch.tensor([t for t, (i, _) in enumerate(rows_meta) if REQS[i]['q_len'] <= 4], device=dev)   # small batch: split-K
has_window = torch.tensor([bool((indices[t][indices[t] >= 0] > boundary(i)).any()) for t, (i, _) in enumerate(rows_meta)],
                          device=dev)
out = run_ring(ALL)
torch.cuda.synchronize()

# rows without window rows (no ring, or nothing selected above the boundary): bitwise today's reader, whatever the split choice
plain = run_plain(ALL, indices)
nw = ~has_window
check('rows without window rows bitwise = kvlut_kernels.attention (%d rows, %d-row batch)' % (int(nw.sum()), T),
      torch.equal(out[nw], plain[nw]), 'max diff %.3g' % (out[nw].float() - plain[nw].float()).abs().max().item())
masked = indices.clone()
for t, (i, pos) in enumerate(rows_meta):
    masked[t][masked[t] > boundary(i)] = -1
for name, rows in (('full batch', ALL), ('decode batch', DECODE)):
    o1 = torch.zeros(len(rows), QH, D, dtype=torch.bfloat16, device=dev)
    KR.attention(query[rows], cache, masked[rows], block_table, token_to_req[rows], seq_lens, ring_k, ring_v, ring_meta, o1,
                 vmean=vmean_rot, n=N)
    o2 = run_plain(rows, masked)
    check('masked lists bitwise = kvlut_kernels.attention (%s)' % name, torch.equal(o1, o2),
          'max diff %.3g' % (o1.float() - o2.float()).abs().max().item())


def reference(t):
    i, pos = rows_meta[t]
    r = REQS[i]
    sel = indices[t][indices[t] >= 0].long()
    rec = sel > boundary(i)
    xk, xv = X[i]
    kk = torch.empty(sel.numel(), KVH, D, dtype=torch.bfloat16, device=dev)
    vv = torch.empty_like(kk)
    if (~rec).any():
        dk, dv = lutc_decode(cache, r['slots'][sel[~rec]])
        kk[~rec], vv[~rec] = dk, dv
    if rec.any():
        kk[rec], vv[rec] = rot_y(xk[sel[rec]], kmean), rot_y(xv[sel[rec]], vmean)
    q = query[t].double()
    res = []
    for h in range(KVH):
        s_ = q[h * G:(h + 1) * G] @ kk[:, h].double().T / 16.0
        res.append(torch.softmax(s_, -1) @ vv[:, h].double() + vmean_rot[h].double())
    return torch.cat(res, 0)


for name, rows, o in (('full batch (one lutc split: direct result + lse)', ALL, out),
                      ('decode batch (lutc split-K partials)', DECODE, run_ring(DECODE))):
    err, scale = 0.0, 0.0
    for k, t in enumerate(rows.tolist()):
        if not has_window[t]:
            continue
        ref = reference(t)
        err = max(err, (ref - o[k].double()).abs().max().item())
        scale = max(scale, ref.abs().max().item())
    check('two-source output vs float64 reference, %s' % name, err / max(scale, 1e-9) < 2e-2,
          'max abs err %.3g, rel %.3g' % (err, err / max(scale, 1e-9)))

# Shift invariance (IE): a compaction removes D rows below the window and renumbers the rest (row -> row - D, off -> off + D); a query
# that selects only window rows must read the same slots, bitwise the same output (no lutc rows, so the page table does not matter).
r2 = REQS[2]
S0, ridx, off0 = r2['S'], 2, r2['off']
win = torch.arange(S0 - N, S0, dtype=torch.int32, device=dev)
qs = torch.randn(1, QH, D, device=dev).to(torch.bfloat16)


def ring_only(shift):
    idx = torch.full((1, TOPK), -1, dtype=torch.int32, device=dev)
    idx[0, :N] = (win - shift)[torch.randperm(N, device=dev)] if False else (win - shift)
    sl = seq_lens.clone()
    sl[ridx] = S0 - shift
    meta = ring_meta.clone()
    meta[ridx, 1] = (off0 + shift) % N
    o = torch.zeros(1, QH, D, dtype=torch.bfloat16, device=dev)
    KR.attention(qs, cache, idx, block_table, torch.tensor([ridx], dtype=torch.int32, device=dev), sl, ring_k, ring_v, meta, o,
                 vmean=vmean_rot, n=N)
    return o


base_out = ring_only(0)
for shift in (4, 300, 1000, 4096 + 12):            # 300: (row + off) wraps differently past N; whole 4-row groups as compactions are
    check('shift invariance: rows - %d, off + %d -> same ring slots, bitwise' % (shift, shift),
          torch.equal(ring_only(shift), base_out))

# G2a tooling self-test: shadow-addressed merge and content / boundary checks must read 0 on a correct ring and flag a corruption
SH = 8192
shadow_k = torch.full((RING_SLOTS * SH, KVH, D), float('nan'), dtype=torch.bfloat16, device=dev)
shadow_v = torch.full_like(shadow_k, float('nan'))
for i, r in enumerate(REQS):
    if r['ring_id'] < 0:
        continue
    xk, xv = X[i]
    meta1 = torch.tensor([[r['ring_id'], r['off'], r['floor']]], dtype=torch.int32, device=dev)
    for a in range(0, r['S'], 4096):
        b = min(a + 4096, r['S'])
        KR.shadow_store(xk[a:b], xv[a:b], shadow_k, shadow_v, torch.zeros(b - a, dtype=torch.int32, device=dev),
                        torch.tensor([0, b - a], dtype=torch.int32, device=dev), torch.tensor([b], dtype=torch.int32, device=dev),
                        meta1, kmean.contiguous(), vmean.contiguous(), SH, b - a)


def debug_run(rows, expected):
    counters = torch.zeros(8, dtype=torch.int64, device=dev)
    first = torch.zeros(8, dtype=torch.int32, device=dev)
    o = torch.zeros(len(rows), QH, D, dtype=torch.bfloat16, device=dev)
    KR.attention(query[rows], cache, indices[rows], block_table, token_to_req[rows], seq_lens, ring_k, ring_v, ring_meta, o,
                 vmean=vmean_rot, n=N, debug=dict(shadow_k=shadow_k, shadow_v=shadow_v, sh=SH, expected=expected,
                                                  counters=counters, first=first, slot_rows=slot_rows))
    torch.cuda.synchronize()
    return o, counters.tolist(), first.tolist()


good_expected = seq_lens.clone()
o_dbg, c, f = debug_run(DECODE, good_expected)
check('G2a tooling: correct ring -> 0 content / seq-len / output / slot-row mismatches',
      c[0] == 0 and c[1] == 0 and c[3] == 0 and c[5] == 0 and c[2] > 0, 'counters %s' % c[:6])
check('G2a tooling: debug path leaves the output unchanged', torch.equal(o_dbg, run_ring(DECODE)))
r2 = REQS[2]
bad_slot = r2['ring_id'] * N + ((r2['S'] - 1) + r2['off']) % N             # the newest row of request 2
saved = ring_k[bad_slot].clone()
ring_k[bad_slot] = ring_k[bad_slot] + 1.0
bad_expected = seq_lens.clone()
bad_expected[3] -= 1                                                          # kernels' seq_len above the scheduler's count
_, c, f = debug_run(DECODE, bad_expected)
ring_k[bad_slot] = saved
check('G2a tooling: a corrupted slot and a wrong seq_len are flagged', c[0] >= 1 and c[1] >= 1 and c[3] > 0 and f[4] == 1
      and f[0] == 2 and f[1] == r2['S'] - 1, 'counters %s, first %s' % (c[:4], f[:5]))

print('G1 %s: %d failures' % ('PASS' if not FAILS else 'FAIL', len(FAILS)))
sys.exit(1 if FAILS else 0)
