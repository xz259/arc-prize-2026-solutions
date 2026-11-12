"""sp_compare_val50.py (2026-10-12, Claude fork 5d20b23b): the side pod's val50 export against the main pod's (job 407,
dflash_features_20261011/cacheval50) on the window rows of every session. The replay is not bitwise reproducible across
scheduling (bring-up 22:07: one session replayed alone vs job 407's 50 at concurrency 32: assistant-token argmax
agreement 0.939, KL 0.035 nats; promptmove's moves and batch composition depend on the schedule), so the gate is
statistical: per session and overall, on assistant tokens, the target's argmax agreement, KL(main || side) and both NLLs,
plus the relative error per shared slot (taps 2, 13, 24, 35, 46 and the final state).
usage: python sp_compare_val50.py SIDE_FEATURES_DIR [--main DIR] [--device cuda:1] [--out OUT.json]"""
import argparse
import json
import os

import torch
from safetensors import safe_open

MAIN = '/workspace/arc3-inference/dflash_features_20261011/cacheval50'
PACK = '/workspace/arc3-inference/dflash_packed_20261011/cache_val50'
TARGET = '/workspace/models/Qwen3.8-Flash-Next-W4A16-intel'
SIDE_SLOTS = (0, 1, 2, 3, 4, 5)        # side export: taps 2,13,24,35,46, final (then the 4 MTP streams)
MAIN_SLOTS = (0, 3, 4, 5, 8, 9)        # job 407: taps 2,4,12,13,24,35,36,44,46, final


def by_sid(d):
    return {f.split('-', 2)[2].rsplit('-', 1)[0]: os.path.join(d, f) for f in os.listdir(d)
            if f.startswith('arc3-persistent-') and f.endswith('.safetensors')}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('side')
    ap.add_argument('--main', default=MAIN)
    ap.add_argument('--device', default='cuda:1')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    dev = torch.device(a.device)
    wm = json.load(open(os.path.join(TARGET, 'model.safetensors.index.json')))['weight_map']
    W = safe_open(os.path.join(TARGET, wm['lm_head.weight']), 'pt').get_tensor('lm_head.weight').to(dev)
    segs = [json.loads(l) for l in open(os.path.join(PACK, 'segments.jsonl'))]
    rows = [json.loads(l) for l in open(os.path.join(PACK, 'rows.jsonl'))]
    side, main_ = by_sid(a.side), by_sid(a.main)
    tot = dict(n=0.0, agree=0.0, kl=0.0, nll_side=0.0, nll_main=0.0)
    slot_rel = [[] for _ in SIDE_SLOTS]
    per = {}
    for seg, rec in zip(segs, rows):
        sid = seg['session_id']
        if sid not in side or sid not in main_:
            print('missing', sid, sid in side, sid in main_); continue
        fs, fm = safe_open(side[sid], 'pt'), safe_open(main_[sid], 'pt')
        hs, hm = fs.get_tensor('hidden_states'), fm.get_tensor('hidden_states')
        ns, nm = hs.shape[0], hm.shape[0]
        w = ns - nm                                                     # side rows start 16,384 earlier (or at 0)
        ts, tm = fs.get_tensor('token_ids')[w:], fm.get_tensor('token_ids')
        assert nm == len(rec['input_ids']) and torch.equal(ts, tm), sid
        mask = torch.tensor(rec['loss_mask'], dtype=torch.bool, device=dev)
        for i, (si, mi) in enumerate(zip(SIDE_SLOTS, MAIN_SLOTS)):
            x, y = hs[w:, si].float(), hm[:, mi].float()
            slot_rel[i].append(((x - y).norm(dim=-1) / y.norm(dim=-1).clamp_min(1e-6)).median().item())
        xs, xm = hs[w:, SIDE_SLOTS[-1]].to(dev), hm[:, MAIN_SLOTS[-1]].to(dev)
        ids = tm.to(dev)
        d = dict(n=0.0, agree=0.0, kl=0.0, nll_side=0.0, nll_main=0.0)
        for s in range(0, nm - 1, 1024):
            e = min(nm - 1, s + 1024)
            m = mask[s + 1:e + 1]
            if not m.any():
                continue
            ls, lm = (xs[s:e][m] @ W.t()).float().log_softmax(-1), (xm[s:e][m] @ W.t()).float().log_softmax(-1)
            nxt = ids[s + 1:e + 1][m]
            d['n'] += float(m.sum()); d['agree'] += float((ls.argmax(-1) == lm.argmax(-1)).sum())
            d['kl'] += float((lm.exp() * (lm - ls)).sum())
            d['nll_side'] += float(-ls.gather(1, nxt[:, None]).sum()); d['nll_main'] += float(-lm.gather(1, nxt[:, None]).sum())
        for k in tot:
            tot[k] += d[k]
        n = max(1.0, d['n'])
        per[sid] = dict(run=seg['run'], tokens=d['n'], agree=d['agree'] / n, kl=d['kl'] / n, nll_side=d['nll_side'] / n, nll_main=d['nll_main'] / n)
        print('%s %-12s assistant %5d: agree %.4f KL %.4f NLL side %.4f main %.4f' % (sid, seg['run'], d['n'], per[sid]['agree'],
              per[sid]['kl'], per[sid]['nll_side'], per[sid]['nll_main']), flush=True)
    n = max(1.0, tot['n'])
    summ = dict(sessions=len(per), assistant_tokens=tot['n'], agree=tot['agree'] / n, kl=tot['kl'] / n,
                nll_side=tot['nll_side'] / n, nll_main=tot['nll_main'] / n,
                slot_rel_median={name: sorted(v)[len(v) // 2] for name, v in zip(('tap2', 'tap13', 'tap24', 'tap35', 'tap46', 'final'), slot_rel) if v})
    print(json.dumps(summ, indent=1))
    if a.out:
        json.dump(dict(summary=summ, per_session=per), open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
