"""CPU tests for qwen38_mtp.py (2026-10-11). Tiny random configs except the checkpoint tests, which read the real
safetensors header (and, with --real, load the real BF16 head on the CPU and run a short forward).
usage: python test_qwen38_mtp_cpu.py [--ckpt MODEL_EXTRA_TENSORS --target-dir DIR --draft-config CONFIG.json] [--real]"""
import argparse
import json
import os
import sys
import tempfile

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen38_mtp as M  # noqa: E402

TINY = M.MTPConfig(hidden_size=16, hc_count=2, hc_lowrank=4, num_heads=4, num_kv_heads=2, head_dim=8, rotary_dim=4,
                   rope_theta=1e4, num_experts=8, top_k=2, moe_intermediate=8, shared_intermediate=8, vocab_size=50,
                   indexer_heads=2, indexer_kv_heads=1, indexer_head_dim=4)


def tiny_model(seed=0):
    torch.manual_seed(seed)
    m = M.Qwen38MTP(TINY).float()
    with torch.no_grad():
        for n, p in m.named_parameters():
            p.normal_(0, 0.3 if 'norm' not in n else 0.1)
    return m


def tiny_inputs(T=12, seed=1):
    g = torch.Generator().manual_seed(seed)
    h0 = torch.randn(T, TINY.hc_count * TINY.hidden_size, generator=g)
    tokens = torch.randint(0, TINY.vocab_size, (T,), generator=g)
    positions = torch.arange(T) * 2 + 5          # gaps, as after evictions
    return h0, tokens, positions


def ref_hc_mix_combine(x, hc_norm_w, down_w, up_w, inj_w, hc, H, eps):
    """vLLM common/hyperconnection.py GatedResidual (mix, then combine of a block output), verbatim formulas."""
    xf = x.float()
    g = xf.unflatten(-1, (hc, H))
    xn = ((g * torch.rsqrt(g.square().mean(-1, keepdim=True) + eps)).flatten(-2) * (1.0 + hc_norm_w.float())).to(x.dtype)
    gate = F.silu(F.linear(xn, down_w) / hc)
    gate = torch.sigmoid(F.linear(gate, up_w)).unflatten(-1, (hc, H))
    mixed = (gate * xn.unflatten(-1, (hc, H))).mean(dim=-2)
    block_out = torch.tanh(mixed)                                             # any block
    inj = 2.0 * torch.sigmoid(F.linear(xn, inj_w) / hc)
    out = x.unflatten(-1, (hc, H)) + block_out.unsqueeze(-2) * inj.unsqueeze(-1)
    return mixed, out.flatten(-2)


def test_hc_matches_reference():
    m = tiny_model()
    hcm = m.layers[0].attn_hyper_connection
    x = torch.randn(7, TINY.hc_count * TINY.hidden_size)
    mixed, inj = hcm.mix(x)
    out = hcm.combine(x, torch.tanh(mixed), inj)
    rm, ro = ref_hc_mix_combine(x, hcm.hc_norm.weight, hcm.input_mix_weight_down.weight, hcm.input_mix_weight_up.weight,
                                hcm.block_inject_weight.weight, TINY.hc_count, TINY.hidden_size, TINY.rms_eps)
    assert torch.allclose(mixed, rm, atol=1e-6) and torch.allclose(out, ro, atol=1e-6)
    print('hc mix / combine == vLLM reference formulas: ok')


def test_multistep_matches_incremental():
    m = tiny_model()
    h0, tokens, positions = tiny_inputs()
    with torch.no_grad():
        a = m.forward_steps(h0, tokens, positions, num_steps=3)
        b = m.incremental_reference(h0, tokens, positions, num_steps=3)
    diffs = [float((x - y).abs().max()) for x, y in zip(a, b)]
    assert max(diffs) < 1e-5, diffs
    print('batched multi-step == row-by-row serving simulation (steps 1-3): max diff %.2e: ok' % max(diffs))


def test_causality():
    m = tiny_model()
    h0, tokens, positions = tiny_inputs(T=16)
    K, cut = 3, 10
    with torch.no_grad():
        a = m.forward_steps(h0, tokens, positions, num_steps=K)
        h2, t2 = h0.clone(), tokens.clone()
        h2[cut:] += 1.0
        t2[cut:] = (t2[cut:] + 7) % TINY.vocab_size
        b = m.forward_steps(h2, t2, positions, num_steps=K)
    # step k at row t reads h_t and x_{t+1..t+k} plus earlier rows: rows t + k < cut are unaffected
    for k in range(1, K + 1):
        rows = cut - k
        assert torch.allclose(a[k - 1][:rows], b[k - 1][:rows], atol=1e-6), k
        assert not torch.allclose(a[k - 1][rows:], b[k - 1][rows:]), k
    print('causality (no row sees later tokens or states; later rows do change): ok')


def test_loss_masking_and_teacher():
    m = tiny_model()
    h0, tokens, positions = tiny_inputs(T=14)
    mask = torch.ones(14, dtype=torch.long)
    mask[8] = 0
    with torch.no_grad():
        s = m.forward_steps(h0, tokens, positions, num_steps=2)
        l1, met = m.step_losses(s, h0, tokens, mask, kl_weight=0.5, chunk=3)
        t2 = tokens.clone()
        t2[8] = (t2[8] + 1) % TINY.vocab_size                  # a masked label: only enters as an input for later steps
        s2 = m.forward_steps(h0, t2, positions, num_steps=2)
        # rows whose inputs include x_8 change; compare the loss restricted to rows before them
        mask2 = mask.clone(); mask2[8:] = 0
        la, _ = m.step_losses(s, h0, tokens, mask2)
        lb, _ = m.step_losses(s2, h0, t2, mask2)
    assert torch.isfinite(l1) and abs(float(la) - float(lb)) < 1e-6
    assert set(met) == {'ce_step1', 'top1_step1', 'kl_step1', 'ce_step2', 'top1_step2', 'kl_step2'}
    print('loss masking and teacher KL (finite, chunked, masked labels inert): ok')


def test_flex_matches_reference():
    if not torch.cuda.is_available():
        print('flex attention: skipped (no GPU here)')
        return
    m = tiny_model().cuda()
    h0, tokens, positions = (t.cuda() for t in tiny_inputs(T=40))
    with torch.no_grad():
        a = m.forward_steps(h0, tokens, positions, attention='reference')
        b = m.forward_steps(h0, tokens, positions, attention='flex')
    d = max(float((x - y).abs().max()) for x, y in zip(a, b))
    assert d < 1e-3, d
    print('flex attention path == reference: %.2e: ok' % d)


def test_save_load_roundtrip():
    m = tiny_model()
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, 'mtp.safetensors')
        n = M.save_mtp_checkpoint(m, p, dtype=torch.float32)
        m2 = M.Qwen38MTP(TINY).float()
        filled = M.load_from_checkpoint(m2, p, None, strict=True)
    for (a, pa), (b, pb) in zip(m.named_parameters(), m2.named_parameters()):
        if a.startswith(('embed_tokens', 'lm_head', 'target_mixer')):
            continue
        assert a == b and torch.equal(pa, pb), a
    print('save -> load round trip under the checkpoint names (%d tensors, %d filled): ok' % (n, len(filled)))


def test_checkpoint_mapping(ckpt, draft_config):
    """Every tensor of the real BF16 MTP file maps to a module parameter of the same shape and every trainable
    parameter is covered (header only)."""
    import struct
    tc = json.load(open(draft_config))['text_config']
    cfg = M.MTPConfig.from_text_config(tc)
    with torch.device('meta'):
        m = M.Qwen38MTP(cfg)
    params = dict(m.named_parameters())
    f = open(ckpt, 'rb')
    n = struct.unpack('<Q', f.read(8))[0]
    hdr = json.loads(f.read(n))
    hdr.pop('__metadata__', None)
    seen, experts = set(), {}
    for name, info in hdr.items():
        mod, e = M.checkpoint_to_module_name(name)
        assert mod in params, (name, mod)
        shape = tuple(params[mod].shape[1:]) if e is not None else tuple(params[mod].shape)
        assert shape == tuple(info['shape']), (name, shape, info['shape'])
        if e is None:
            seen.add(mod)
        else:
            experts.setdefault(mod, set()).add(e)
    trainable = [k for k in params if not k.startswith(('embed_tokens', 'lm_head', 'target_mixer'))]
    missing = [k for k in trainable if k not in seen and k not in experts]
    assert not missing, missing
    assert all(len(v) == cfg.num_experts for v in experts.values()), {k: len(v) for k, v in experts.items()}
    total = sum(p.numel() for k, p in params.items() if k in trainable)
    print('real checkpoint: %d tensors map onto the module (%d stacked expert banks x %d experts); config %s; trainable '
          'head %.3fB parameters: ok' % (len(hdr), len(experts), cfg.num_experts,
                                       {k: getattr(cfg, k) for k in ('hc_count', 'rotary_dim', 'rope_theta', 'top_k')}, total / 1e9))
    return cfg


def test_real_forward(ckpt, target_dir, cfg):
    torch.manual_seed(0)
    m = M.Qwen38MTP(cfg).to(torch.bfloat16)
    M.load_from_checkpoint(m, ckpt, target_dir, strict=True)
    T = 10
    h0 = torch.randn(T, cfg.hc_count * cfg.hidden_size).to(torch.bfloat16)
    tokens = torch.randint(0, cfg.vocab_size, (T,))
    with torch.no_grad():
        s = m.forward_steps(h0, tokens, torch.arange(T), num_steps=3)
        loss, met = m.step_losses(s, h0, tokens, torch.ones(T, dtype=torch.long), kl_weight=1.0)
    assert all(torch.isfinite(x.float()).all() for x in s) and torch.isfinite(loss)
    print('real BF16 head on CPU: 3 steps x %d rows finite; loss %.3f (random states, not meaningful): ok' % (s[0].shape[0], float(loss)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt'); ap.add_argument('--target-dir'); ap.add_argument('--draft-config')
    ap.add_argument('--real', action='store_true')
    a = ap.parse_args()
    test_hc_matches_reference()
    test_multistep_matches_incremental()
    test_causality()
    test_loss_masking_and_teacher()
    test_save_load_roundtrip()
    test_flex_matches_reference()
    if a.ckpt:
        cfg = test_checkpoint_mapping(a.ckpt, a.draft_config)
        if a.real:
            test_real_forward(a.ckpt, a.target_dir, cfg)
    print('all tests passed')
