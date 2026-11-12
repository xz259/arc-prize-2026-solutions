"""dflash_dense_anchors.py (2026-10-12, Claude fork 5d20b23b): DFlash's per-anchor slot acceptance at EVERY valid anchor
of the val50 windows, for the live-process (renewal) acceptance length (live_tau.py). The validation metric averages over
uniformly drawn anchors; live rounds start only after the previous round's accepted run, so easy stretches get fewer
rounds and the uniform mean overstates live tau, more at larger K (MTP3: renewal 2.796 = the live gen402 run's 2.797 vs
the uniform 2.931; K 5: ratio 0.909). Same model and data path as dflash_eval_window.py (one window per forward, all
anchors: select_anchors replaced by every row with loss_mask 1 outside the last block), eval_accept_modes' per-row
acceptance captured from its aggregate() call (per anchor, slots 1..15, times the aligned mask).
Output: torch file {session id: {anchors [A] (row in the window), slot_mask [A, 15], <mode> [A, 15]}} in live_tau.py's
format, plus the uniform-anchor L_K as a check against the sampled validation.
usage: python dflash_dense_anchors.py --ckpt DIR --out OUT.pt [--features DIR --slots 0,1,2,3,4 --final 5] [--device cuda:0]"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault('ARC3_VERIFIER_NORM', 'identity')
os.environ.setdefault('ARC3_VOCAB_CHUNK', '3072')
SPEC = '/workspace/arc3-inference/speculators_p6c_20261010'
W = '/workspace/arc3-inference/dflash_work_20261009'
sys.path[:0] = [SPEC + '/src', SPEC + '/hs_connectors/src', W, os.path.dirname(os.path.abspath(__file__))]

import torch  # noqa: E402

import dflash_eval_window as dew  # noqa: E402

MODES = ('sel_srv_T10', 'sel_srv_T06', 'prob_srv_T10', 'prob_srv_T06', 'greedy_srv_T10', 'ceil_srv_T10')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--features', default='')
    ap.add_argument('--slots', default='0,3,4,5,8')
    ap.add_argument('--final', type=int, default=-1)
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--sessions', type=int, default=0)
    a = ap.parse_args()
    import eval_accept_modes as eam
    eam.TEMPS = (0.6, 1.0)
    eam.KS = (1, 2, 3, 4, 5, 6, 7, 8, 15)
    eam.SLOTS = (1, 2, 3, 4, 5, 6, 7, 8)
    import speculators.models.dflash.core as dcore
    from speculators.models.dflash2 import DFlash2DraftModel
    tail = eam.install_row_hook()
    assert tail is not None
    tail.ROW_HOOK = eam
    cap = {}
    orig_agg = eam.aggregate

    def aggregate(acc, loss_mask, block_size, start_pos):
        n = next(iter(acc.values())).shape[0]
        nb = n // block_size
        vb = loss_mask[0].float().view(nb, block_size)[:, start_pos:]
        cap['slot_mask'] = vb.bool().cpu()
        for m in MODES:
            cap[m] = (acc[m].view(nb, block_size)[:, start_pos:] * vb).float().cpu()
        return orig_agg(acc, loss_mask, block_size, start_pos)

    eam.aggregate = aggregate

    def select_all(loss_mask, num_anchors, block_size):
        valid = loss_mask.bool().clone()
        valid[:, -block_size:] = False
        idx = torch.nonzero(valid.squeeze(0), as_tuple=False).squeeze(-1)
        cap['anchors'] = idx.cpu()
        return idx, torch.ones_like(idx, dtype=torch.bool)

    dcore.select_anchors = select_all
    dev = torch.device(a.device)
    rows = dew.load_rows(a.features or None, [int(x) for x in a.slots.split(',')], a.final)
    if a.sessions:
        rows = rows[:a.sessions]
    model = DFlash2DraftModel.from_pretrained(a.ckpt).to(torch.bfloat16)
    model.load_verifier_weights()
    model = model.to(dev).eval()
    out, tot = {}, {}
    t0 = time.time()
    for r in rows:
        n = r['input_ids'].shape[0]
        kw = dict(hidden_states=r['hidden_states'].unsqueeze(0).to(torch.bfloat16), input_ids=r['input_ids'].unsqueeze(0),
                  loss_mask=r['loss_mask'].unsqueeze(0), position_ids=r['position_ids'].unsqueeze(0),
                  verifier_last_hidden_states=r['verifier_last_hidden_states'].unsqueeze(0).to(torch.bfloat16),
                  document_ids=torch.zeros(1, n, dtype=torch.long))
        valid = r['loss_mask'].bool().clone(); valid[-model.block_size:] = False
        na = int(valid.sum())
        cap.clear()
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            model(**{k: v.to(dev) for k, v in kw.items()}, max_anchors=na)
        d = {k: v for k, v in cap.items()}
        assert d['anchors'].shape[0] == na and d['sel_srv_T10'].shape[0] == na, (na, d['anchors'].shape)
        out[r['sid']] = d
        for m in MODES:
            pref = d[m].cumprod(1)
            blk = d['slot_mask'].any(1).float()
            t = tot.setdefault(m, [0.0] * 9 + [0.0])
            for k in range(1, 9):
                t[k] += float(((1 + pref[:, :k].sum(1)) * blk).sum())
            t[9] += float(blk.sum())
        print('%s: %d anchors, %.0f s cumulative' % (r['sid'], na, time.time() - t0), flush=True)
    torch.save(out, a.out)
    for m in MODES:
        t = tot[m]
        print('%-14s uniform-anchor L1-8 %s' % (m, [round(t[k] / t[9], 4) for k in range(1, 9)]), flush=True)
    print('saved', a.out, flush=True)


if __name__ == '__main__':
    main()
