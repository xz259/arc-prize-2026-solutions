"""compare_replay_arms.py (2026-10-13, Claude fork 5d20b23b): are the features of a faster replay (larger prefill chunk, fewer
draft tokens) equivalent to the production replay's? Two exports of the same sessions (segment files, stored slots: final,
then the 4 MTP streams), compared on assistant positions (stream role 1; up to --max-rows per session) as the side pod's
replay equivalence was judged (sp_compare_val50.py; replays are not bitwise reproducible across schedules): the target's
argmax agreement, KL(base || fast) and both NLLs of the recorded next token from lm_head(final), and the relative error of
the 4 streams (the MTP's input).
usage: python compare_replay_arms.py BASE_DIR FAST_DIR [--device cuda:0] [--out OUT.json]"""
import argparse
import collections
import json
import os
import re

import numpy as np
import torch
from safetensors import safe_open

TARGET = '/workspace/models/Qwen3.8-Flash-Next-W4A16-intel'
ST = '/workspace/arc3-inference/dflash_streams_20261011'
SID = re.compile(r'(arc3-[0-9a-f]{32})')


def sessions(d):
    out = collections.defaultdict(list)
    for f in sorted(os.listdir(d)):
        m = SID.search(f)
        if m and f.endswith('.safetensors'):
            out[m.group(1)].append(os.path.join(d, f))
    return out


def load(files):
    hs, ids, pos = [], [], []
    for f in files:                                    # segments in name order (_s0000, _s0001, ...)
        with safe_open(f, 'pt') as fh:
            hs.append(fh.get_tensor('hidden_states')); ids.append(fh.get_tensor('token_ids')); pos.append(fh.get_tensor('position_ids'))
    return torch.cat(hs), torch.cat(ids), torch.cat(pos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('base'); ap.add_argument('fast')
    ap.add_argument('--device', default='cuda:0'); ap.add_argument('--out', default='')
    ap.add_argument('--max-rows', type=int, default=40000)
    a = ap.parse_args()
    dev = torch.device(a.device)
    idx = json.load(open(os.path.join(TARGET, 'model.safetensors.index.json')))['weight_map']
    key = next(k for k in idx if k.endswith('lm_head.weight'))
    with safe_open(os.path.join(TARGET, idx[key]), 'pt') as fh:
        W = fh.get_tensor(key).to(dev, torch.bfloat16)
    runs = {}
    for line in open(ST + '/sessions.jsonl'):
        d = json.loads(line)
        runs[d['session_id']] = d['run']
    B, F = sessions(a.base), sessions(a.fast)
    tot = collections.Counter()
    per = []
    g = torch.Generator().manual_seed(0)
    for sid in sorted(set(B) & set(F)):
        hb, ib, pb = load(B[sid]); hf, if_, pf = load(F[sid])
        n = min(len(ib), len(if_))
        assert torch.equal(ib[:n], if_[:n]) and torch.equal(pb[:n], pf[:n]), sid
        role = np.load(os.path.join(ST, runs[sid], sid + '.npz'))['role']
        rows = torch.nonzero(torch.from_numpy(role[pb[1:n].numpy()] == 1)).squeeze(1)      # row r predicts an assistant token r + 1
        if len(rows) > a.max_rows:
            rows = rows[torch.randperm(len(rows), generator=g)[:a.max_rows]].sort().values
        s = collections.Counter()
        for c in range(0, len(rows), 4096):
            r = rows[c:c + 4096]
            lb = (hb[r, 0].to(dev, torch.bfloat16) @ W.t()).float().log_softmax(-1)
            lf = (hf[r, 0].to(dev, torch.bfloat16) @ W.t()).float().log_softmax(-1)
            nxt = ib[r + 1].to(dev)
            s['n'] += len(r)
            s['agree'] += float((lb.argmax(-1) == lf.argmax(-1)).sum())
            s['kl'] += float((lb.exp() * (lb - lf)).sum())
            s['nll_b'] -= float(lb.gather(1, nxt[:, None]).sum()); s['nll_f'] -= float(lf.gather(1, nxt[:, None]).sum())
            xb, xf = hb[r, 1:5].float(), hf[r, 1:5].float()
            s['st_err'] += float((xb - xf).norm(dim=-1).sum() / 4); s['st_norm'] += float(xb.norm(dim=-1).sum() / 4)
        tot.update(s)
        per.append(dict(session=sid, rows=int(s['n']), agree=s['agree'] / s['n'], kl=s['kl'] / s['n'],
                        nll_base=s['nll_b'] / s['n'], nll_fast=s['nll_f'] / s['n']))
        print('%s rows %6d agree %.4f KL %.4f NLL base %.4f fast %.4f' % (sid, s['n'], s['agree'] / s['n'], s['kl'] / s['n'],
              s['nll_b'] / s['n'], s['nll_f'] / s['n']), flush=True)
    summ = dict(sessions=len(per), rows=int(tot['n']), agree=tot['agree'] / tot['n'], kl=tot['kl'] / tot['n'],
                nll_base=tot['nll_b'] / tot['n'], nll_fast=tot['nll_f'] / tot['n'], stream_rel_err=tot['st_err'] / tot['st_norm'])
    print('overall', json.dumps(summ), flush=True)
    if a.out:
        json.dump(dict(summary=summ, per_session=per), open(a.out, 'w'), indent=1)


if __name__ == '__main__':
    main()
