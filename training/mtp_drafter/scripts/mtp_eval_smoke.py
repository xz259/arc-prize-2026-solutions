"""mtp_eval_smoke.py (2026-10-12, Claude fork 5d20b23b): shape/runtime smoke test of mtp_eval_val50.py on a synthetic
session (random streams with realistic scale, consecutive positions, a random assistant mask), plus the QSA emulation
against dense on a context of <= 2,048 rows (must agree exactly: every row is selected there).
usage: python mtp_eval_smoke.py [--device cuda:1] [--experts served|bf16]"""
import argparse
import json
import os
import sys
import tempfile
import time

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mtp_eval_val50 as E


def synth(path, N, start, end, seed=0):
    g = torch.Generator().manual_seed(seed)
    hs = (torch.randn(N, 10, 2560, generator=g) * 0.5).to(torch.bfloat16)
    ids = torch.randint(0, 150000, (N,), generator=g)
    pos = torch.arange(end - N, end, dtype=torch.long)
    save_file(dict(hidden_states=hs, token_ids=ids, position_ids=pos, served_position_ids=pos.clone()), path)
    w = end - start
    mask = (torch.rand(w, generator=g) < 0.6).long()
    mask[:min(w, 64)] = 0
    return dict(session_id='arc3-smoke', run='smoke', start=start, end=end), dict(input_ids=ids[N - w:].tolist(), loss_mask=mask.tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--device', default='cuda:1')
    ap.add_argument('--experts', default='served')
    a = ap.parse_args()
    args = argparse.Namespace(target='/workspace/models/Qwen3.8-Flash-Next-W4A16-intel',
                              draft='/workspace/models/Qwen3.8-Flash-Next-mtp-draft-nvfp4a16-noisc',
                              vocab='/workspace/arc3-inference/runs/stage0-reference-20260919-v1/stage12/harness_revision_v529/spec_decode_stage2/draft_vocab_32k.json',
                              kit='/workspace/arc3-inference/mtp_ft_20261011', experts=a.experts, attention='qsa',
                              pos_key='position_ids', K=5, chunk=4096, qchunk=256)
    dev = torch.device(a.device)
    t0 = time.time()
    Q, model, cfg, sub, wsub = E.load_model(args, dev)
    print('loaded %.0f s, torch %s, subset %d' % (time.time() - t0, torch.__version__, len(sub)), flush=True)
    with tempfile.TemporaryDirectory() as d:
        # (1) short context: QSA == dense exactly (<= 512 complete groups)
        seg, rec = synth(os.path.join(d, 's.safetensors'), 1800, 1000, 1800, seed=1)
        out = {}
        for att in ('qsa', 'dense'):
            args.attention = att
            r, (acc, sm, an) = E.run_session(Q, model, cfg, sub, wsub, os.path.join(d, 's.safetensors'), seg, rec, args, dev)
            out[att] = acc
        diff = max(float((out['qsa'][k] - out['dense'][k]).abs().max()) for k in out['qsa'])
        print('short context: anchors %d, max |qsa - dense| over modes %.3g (expect 0)' % (len(an), diff), flush=True)
        # (2) long context: 32,768 rows, QSA selection active; timing
        args.attention = 'qsa'
        seg, rec = synth(os.path.join(d, 'l.safetensors'), 32768, 16384, 32768, seed=2)
        t1 = time.time()
        r, (acc, sm, an) = E.run_session(Q, model, cfg, sub, wsub, os.path.join(d, 'l.safetensors'), seg, rec, args, dev)
        torch.cuda.synchronize(dev)
        s = E.summarize(r, args.K)
        print('long context: anchors %d, %.0f s, peak GPU %.1f GB; prob_srv_T10 %s' % (len(an), time.time() - t1,
              torch.cuda.max_memory_allocated(dev) / 2**30, json.dumps(s['prob_srv_T10'])), flush=True)


if __name__ == '__main__':
    main()
