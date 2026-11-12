"""test_served_loss_cpu.py (2026-10-13, Claude fork 5d20b23b): train_mtp_sp.step_losses_served (served target computed
once per window row, chunked, under checkpoints) against a plain per-step reference (the target recomputed for every
step, no checkpoint, no chunking) on the tiny random model, K 4, for served_kl / tv / lk with and without the
accepted-path weights: loss, every parameter gradient and the metrics must match (fp32, CPU).
usage: MTP_KIT=... python test_served_loss_cpu.py"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import train_mtp_sp as T  # noqa: E402

M = T.M
sys.path.insert(0, os.environ.get('MTP_KIT', '/workspace/arc3-inference/mtp_ft_20261011'))
import test_qwen38_mtp_cpu as TT  # noqa: E402


def reference(model, samples, h0, tokens, mask_all, weights, kind, wsub, t2d, temp, accept):
    K, L = len(samples), samples[0].shape[0]
    total, w, met = 0.0, torch.ones(L), {}
    for k, s in enumerate(samples, start=1):
        lab = tokens[k + 1:L + k + 1]
        m = mask_all[k + 1:L + k + 1].float()
        y = (s @ wsub.t()).float()
        with torch.no_grad():
            pp, pi = T.served_target(model.lm_head(model.teacher_hidden(h0[k:L + k])).float() / temp)
            pos = t2d[pi]; ins = pos >= 0; ppi = pp * ins
            pin = ppi / ppi.sum(1, keepdim=True).clamp_min(1e-12)
            lab_pos = t2d[lab]; p_lab = (pp * (pi == lab[:, None])).sum(1)
        lq = torch.log_softmax(y / temp, -1).gather(1, pos.clamp_min(0))
        kl = torch.where(pin > 0, pin * (torch.log(pin.clamp_min(1e-30)) - lq), torch.zeros_like(lq)).sum(1)
        alpha = torch.minimum(ppi, lq.exp() * ins).sum(1)
        lam = torch.exp(-3.0 * alpha.detach())
        row = {'served_kl': kl, 'tv': 1.0 - alpha, 'lk': lam * kl + (1.0 - lam) * (1.0 - alpha)}[kind]
        wr = m * w if accept else m
        total = total + weights[k - 1] * (row * wr).sum() / m.sum().clamp(min=1)
        with torch.no_grad():
            qd, qs = T.served_draft(y, temp)
            q_lab = (qd * (qs == lab_pos[:, None])).sum(1) * (lab_pos >= 0)
            w = w * torch.where(p_lab > 0, (q_lab / p_lab.clamp_min(1e-12)).clamp(max=1.0), torch.zeros_like(p_lab))
        met['kl_step%d' % k] = float((kl.detach() * m).sum() / m.sum().clamp(min=1))
    return total, met


def grads(model, loss):
    model.zero_grad(set_to_none=True)
    loss.backward()
    return {n: p.grad.clone() for n, p in model.named_parameters() if p.grad is not None}


def main():
    model = TT.tiny_model().train()
    h0, tok, pos = TT.tiny_inputs(T=40)
    K = 4
    sub, t2d = T.load_subset('tiny', model.lm_head.weight.shape[0], 'cpu')
    wsub = model.lm_head.weight.index_select(0, sub).contiguous()
    mask = (torch.arange(tok.shape[0]) % 5 != 0).long()
    weights = [1.0, 0.69, 0.40, 0.18]
    worst = 0.0
    for kind in ('served_kl', 'tv', 'lk'):
        for accept in (False, True):
            s = model.forward_steps(h0, tok, pos, num_steps=K, attention='reference')
            ref, rmet = reference(model, s, h0, tok, mask, weights, kind, wsub, t2d, 1.0, accept)
            g_ref = grads(model, ref)
            s = model.forward_steps(h0, tok, pos, num_steps=K, attention='reference')
            new, nmet = T.step_losses_served(model, s, h0, tok, mask, weights, kind, wsub, t2d, 1.0, accept, chunk=7)
            g_new = grads(model, new)
            dl = abs(float(new) - float(ref))
            dg = max(float((g_new[n] - g_ref[n]).abs().max() / g_ref[n].abs().max().clamp_min(1e-12)) for n in g_ref)
            dm = max(abs(nmet['kl_step%d' % k] - rmet['kl_step%d' % k]) for k in range(1, K + 1))
            assert set(g_new) == set(g_ref) and dl < 1e-5 and dg < 1e-4 and dm < 1e-5, (kind, accept, dl, dg, dm)
            worst = max(worst, dg)
            print('%-9s accept %-5s loss %.6f vs %.6f, max rel grad diff %.1e, kl metric diff %.1e: ok' % (kind, accept, float(new), float(ref), dg, dm))
    print('step_losses_served == per-step reference (chunk 7, K %d), worst rel grad diff %.1e' % (K, worst))


if __name__ == '__main__':
    main()
