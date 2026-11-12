"""test_ckpt_steps_cpu.py (2026-10-13, Claude fork 5d20b23b): train_mtp_sp.forward_steps_ckpt (activation checkpoint per
draft step) against the kit's forward_steps on the tiny random model, K 5, reference attention: outputs and every
parameter gradient of a fixed random projection loss must match (fp32, CPU), and the routed-expert load is counted once.
usage: MTP_KIT=... python test_ckpt_steps_cpu.py"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_mtp_sp as T  # noqa: E402

M = T.M
sys.path.insert(0, os.environ.get('MTP_KIT', '/workspace/arc3-inference/mtp_ft_20261011'))
import test_qwen38_mtp_cpu as TT  # noqa: E402


def run(fwd, model, h0, tok, pos, K, projs):
    model.zero_grad(set_to_none=True)
    mlp = model.layers[0].mlp
    mlp._load = torch.zeros(mlp.cfg.num_experts, dtype=torch.long)
    s = fwd(model, h0, tok, pos, num_steps=K, attention='reference')
    loss = sum((x * p).sum() for x, p in zip(s, projs))
    T.TRACK[0] = False
    loss.backward()
    T.TRACK[0] = True
    return [x.detach() for x in s], {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}, mlp._load.clone()


def main():
    model = TT.tiny_model().train()
    h0, tok, pos = TT.tiny_inputs(T=24)
    K = 5
    L = tok.shape[0] - K - 1
    g = torch.Generator().manual_seed(3)
    projs = [torch.randn(L, TT.TINY.hidden_size, generator=g) for _ in range(K)]
    a = run(M.Qwen38MTP.forward_steps, model, h0, tok, pos, K, projs)
    b = run(T.forward_steps_ckpt, model, h0, tok, pos, K, projs)
    out = max(float((x - y).abs().max()) for x, y in zip(a[0], b[0]))
    grad = max(float((a[1][n] - b[1][n]).abs().max() / a[1][n].abs().max().clamp_min(1e-12)) for n in a[1])
    assert set(a[1]) == set(b[1]), 'gradient sets differ'
    assert torch.equal(a[2], b[2]), ('load counted differently', a[2], b[2])
    assert out < 1e-5 and grad < 1e-5, (out, grad)
    print('forward_steps_ckpt == forward_steps (K %d): max |out diff| %.2e, max rel grad diff %.2e over %d tensors, load %d: ok'
          % (K, out, grad, len(a[1]), int(a[2].sum())))


if __name__ == '__main__':
    main()
