"""test_served_experts_pod.py (2026-10-13, Claude fork 5d20b23b; side pod, CPU, real checkpoints): the trainer's frozen routed
experts with --expert-weights served (train_mtp_sp.py: the kit model built as the trainer builds it, apply_train_mode
noexperts, load_served_experts) are bitwise the tensors that mtp_eval_val50.py --experts served scores (its load_model), and
every other MoE tensor (router, shared expert and its gate) matches the scorer's BF16 values. Also prints how far the served
experts are from the BF16 source the rounds trained with.
usage: python test_served_experts_pod.py [--trainer-dir DIR] (default: this file's directory)"""
import argparse
import json
import os
import sys

import torch

ap = argparse.ArgumentParser()
ap.add_argument('--trainer-dir', default=os.path.dirname(os.path.abspath(__file__)))
ap.add_argument('--scripts', default='/mnt/shared/arc3-inference/sidepod_20261012/scripts')
a = ap.parse_args()
sys.path.insert(0, a.scripts)
sys.path.insert(0, a.trainer_dir)
import train_mtp_sp as T  # noqa: E402  (puts MTP_KIT on the path)
import mtp_eval_val50 as E  # noqa: E402

TARGET = '/workspace/models/Qwen3.8-Flash-Next-W4A16-intel'
DRAFT = '/workspace/models/Qwen3.8-Flash-Next-mtp-draft-nvfp4a16-noisc'
print('trainer %s, scorer %s' % (T.__file__, E.__file__), flush=True)
cfg = T.M.MTPConfig.from_text_config(json.load(open(os.path.join(DRAFT, 'config.json')))['text_config'])
m = T.M.Qwen38MTP(cfg)
T.M.load_from_checkpoint(m, os.path.join(TARGET, 'model_extra_tensors.safetensors'), TARGET, strict=True)
T.apply_train_mode(m, 'noexperts')
mm = m.layers[0].mlp
src = {k: getattr(mm, k).data.clone() for k in ('w_gate', 'w_up', 'w_down')}
n = T.load_served_experts(m, DRAFT)
print('trainer: %d served matrices loaded' % n, flush=True)
ns = argparse.Namespace(kit=os.environ.get('MTP_KIT', '/workspace/arc3-inference/mtp_ft_20261011'), target=TARGET, draft=DRAFT,
                        head='', experts='served', vocab=T.DRAFT_VOCAB)
_, em, _, _, _ = E.load_model(ns, torch.device('cpu'))
emm = em.layers[0].mlp
for k in ('w_gate', 'w_up', 'w_down'):
    t, e = getattr(mm, k).data, getattr(emm, k).data
    assert t.dtype == e.dtype == torch.bfloat16 and t.shape == e.shape, (k, t.dtype, e.dtype, t.shape, e.shape)
    assert torch.equal(t, e), (k, float((t.float() - e.float()).abs().max()))
    rel = float((t.float() - src[k].float()).norm() / src[k].float().norm())
    print('%s: bitwise equal to the scorer; served vs BF16 source: relative difference %.4f, %.1f%% of entries differ' % (
          k, rel, 100.0 * float((t != src[k]).float().mean())), flush=True)
other = {nm: p for nm, p in mm.named_parameters() if not nm.startswith('w_')}
for nm, p in other.items():
    q = dict(emm.named_parameters())[nm]
    assert torch.equal(p.data.to(torch.bfloat16), q.data), nm
print('router, shared expert and its gate: equal to the scorer (%d tensors)' % len(other))
print('served experts test ok')
