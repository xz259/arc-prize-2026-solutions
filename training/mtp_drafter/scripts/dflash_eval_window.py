"""dflash_eval_window.py (2026-10-12, Claude fork 5d20b23b): SIDE_POD_PLAN section 6(a) (Inference Engineering 10-12:
evaluate the Phase 6 checkpoint at windows 1,024 and 2,048 on val50; evaluation only, no training). A standalone copy of
Phase 6's validation pass: the checkpoint's DFlash 2 drafter (speculators_p6c with the ARC3 patches: verifier norm
identity, chunked vocabulary tail with eval_accept_modes' row hook) on the val50 rows read from the main pod's job 407
files (taps 2, 13, 24, 35, 46 = slots 0, 3, 4, 5, 8; the final state = the last slot; positions from the file, re-offset
per sample), packed 4 rows per 65,536-token batch with document ids as the trainer's collate does, 1,536 anchors per
batch at a fixed seed per batch (identical anchors for every window: a matched comparison). Each --window rewrites the
drafter's sliding window (all 10 layers are sliding_attention); 512 is the control and must land near the trainer's
logged 15K value (sel_srv T 1.0 L8 3.183: same population of anchors, different draws). --seeds repeats the pass with
other anchor draws to shrink the anchor noise.
usage: python dflash_eval_window.py --ckpt DIR --out OUT.json [--windows 512 1024 2048] [--seeds 1 2] [--device cuda:1]"""
import argparse
import json
import os
import sys
import time

os.environ.setdefault('ARC3_VERIFIER_NORM', 'identity')
os.environ.setdefault('ARC3_VOCAB_CHUNK', '3072')
SPEC = '/workspace/arc3-inference/speculators_p6c_20261010'
W = '/workspace/arc3-inference/dflash_work_20261009'
sys.path[:0] = [SPEC + '/src', SPEC + '/hs_connectors/src', W]

import torch  # noqa: E402
from safetensors import safe_open  # noqa: E402

PACK = '/workspace/arc3-inference/dflash_packed_20261011/cache_val50'
SLOTS = [0, 3, 4, 5, 8]
MAXLEN, ANCHORS = 65536, 1536


def load_rows(features=None, slots=SLOTS, final=-1):
    """The 50 validation rows: from the main pod's pack (job 407 files, default) or a side-pod export directory
    (--features; rows [start - 16,384, end) per session: the window is the last 16,384 rows; --slots / --final pick
    the taps and the final state, e.g. 0,1,2,3,4 and 5 for sp_val50mtp)."""
    segs = [json.loads(l) for l in open(os.path.join(PACK, 'segments.jsonl'))]
    recs = [json.loads(l) for l in open(os.path.join(PACK, 'rows.jsonl'))]
    files = None
    if features:
        files = {f.split('-', 2)[2].rsplit('-', 1)[0]: os.path.join(features, f) for f in os.listdir(features)
                 if f.startswith('arc3-persistent-') and f.endswith('.safetensors')}
    rows = []
    for i, (seg, rec) in enumerate(zip(segs, recs)):
        path = files[seg['session_id']] if files else os.path.join(PACK, 'hs_%d.safetensors' % seg['idx'])
        f = safe_open(path, 'pt')
        hs = f.get_tensor('hidden_states')
        ids = f.get_tensor('token_ids')
        pos = f.get_tensor('position_ids').long()
        w = ids.shape[0] - len(rec['input_ids'])
        hs, ids, pos = hs[w:], ids[w:], pos[w:]
        assert torch.equal(ids, torch.tensor(rec['input_ids'])), seg
        rows.append(dict(sid=seg['session_id'], hidden_states=hs[:, list(slots)].flatten(1), verifier_last_hidden_states=hs[:, final],
                         input_ids=ids, loss_mask=torch.tensor(rec['loss_mask']), position_ids=pos - pos[0]))
    return rows


def batches(rows):
    out = []
    for b0 in range(0, len(rows), 4):
        grp = rows[b0:b0 + 4]
        d = {}
        for key in ('hidden_states', 'verifier_last_hidden_states', 'input_ids', 'loss_mask', 'position_ids'):
            first = grp[0][key]
            buf = torch.zeros((MAXLEN, *first.shape[1:]), dtype=torch.bfloat16 if 'hidden' in key else first.dtype)
            o = 0
            for r in grp:
                n = r[key].shape[0]; buf[o:o + n] = r[key]; o += n
            d[key] = buf.unsqueeze(0)
        lengths = [r['input_ids'].shape[0] for r in grp]
        doc = torch.repeat_interleave(torch.arange(len(lengths)), torch.tensor(lengths))
        d['document_ids'] = torch.cat([doc, -torch.ones(MAXLEN - doc.shape[0], dtype=torch.long)]).unsqueeze(0)
        out.append(d)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--windows', type=int, nargs='+', default=[512, 1024, 2048])
    ap.add_argument('--seeds', type=int, nargs='+', default=[1])
    ap.add_argument('--device', default='cuda:1')
    ap.add_argument('--features', default='', help='a side-pod export instead of the main pod pack')
    ap.add_argument('--slots', default='0,3,4,5,8')
    ap.add_argument('--final', type=int, default=-1)
    a = ap.parse_args()
    import eval_accept_modes as eam
    eam.TEMPS = (0.6, 1.0)
    eam.KS = (1, 2, 3, 4, 5, 6, 7, 8, 15)            # L5-L7 too: K = 6 parity and the per-slot conditional view
    eam.SLOTS = (1, 2, 3, 4, 5, 6, 7, 8)
    import speculators.models.dflash2.core as core
    from speculators.models.dflash2 import DFlash2DraftModel
    tail = eam.install_row_hook()
    assert tail is not None, 'patch 6 tail not found'
    tail.ROW_HOOK = eam
    orig = core.compute_metrics

    def compute_metrics(**kw):                       # the non-chunked path (unused with ARC3_VOCAB_CHUNK > 0)
        loss, m = orig(**kw)
        m.update(eam.extra_from_kwargs(kw))
        return loss, m

    core.compute_metrics = compute_metrics
    dev = torch.device(a.device)
    t0 = time.time()
    rows = load_rows(a.features or None, [int(x) for x in a.slots.split(',')], a.final)
    bs = batches(rows)
    print('val rows %d, batches %d, loaded in %.0f s' % (len(rows), len(bs), time.time() - t0), flush=True)
    results = {}
    for win in a.windows:
        model = DFlash2DraftModel.from_pretrained(a.ckpt).to(torch.bfloat16)
        assert model.uses_sliding_window_attn and not model.uses_full_attn, 'expected all-sliding layers'
        model.sliding_window = win                  # _build_attention_mask reads it for the sliding-window mask
        model.config.transformer_layer_config.sliding_window = win
        model.load_verifier_weights()
        model = model.to(dev).eval()
        tot = {}
        t1 = time.time()
        for seed in a.seeds:
            for bi, b in enumerate(bs):
                torch.manual_seed(seed * 1000 + bi)
                kw = {k: v.to(dev) for k, v in b.items()}
                with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
                    out = model(**kw, max_anchors=ANCHORS)
                metrics = out[2]
                for k, v in metrics.items():
                    if k.startswith('xa_'):
                        tot[k] = tot.get(k, 0.0) + float(v)
        res = {}
        for k in tot:
            if k.endswith('_sum'):
                base = k[:-4]
                res[base] = round(tot[k] / max(1.0, tot.get(base + '_total', 1.0)), 4)
        results[str(win)] = res
        print('window %d (%.0f s): sel_srv T10 L1-8 %s | T06 L8 %s | prob_srv T10 L8 %s' % (win, time.time() - t1,
              [res.get('xa_sel_srv_T10_L%d' % k) for k in (1, 2, 3, 4, 5, 6, 7, 8)], res.get('xa_sel_srv_T06_L8'),
              res.get('xa_prob_srv_T10_L8')), flush=True)
        json.dump(dict(args=vars(a), results=results), open(a.out, 'w'), indent=1)
        del model
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
