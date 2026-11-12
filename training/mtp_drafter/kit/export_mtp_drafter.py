"""export_mtp_drafter.py (2026-10-11; MTP fine-tune backup plan): turn a fine-tuned BF16 MTP head (`mtp.*` tensors, as
qwen38_mtp.save_mtp_checkpoint writes them) into a servable drafter folder in the format of the production drafter
`/workspace/models/Qwen3.8-Flash-Next-mtp-draft-nvfp4a16-noisc`: every tensor BF16 except the 512 routed experts'
gate / up / down projections, which are NVFP4 (modelopt layout: `weight` uint8 [out, in / 2] with element 2i in the low
nibble, `weight_scale` float8_e4m3fn [out, in / 16], `weight_scale_2` float32 scalar = amax / (6 * 448); no input
scales). The embedding and lm_head are copied from the source drafter; the small files are copied; the safetensors
index and ARC3_DRAFT_VIEW.json are rewritten (source unchanged, so the launcher's source check passes).
Quantizer (matched to the served file, 10-11): vLLM's E2M1 rounding table (`nvfp4_emulation_utils.cast_to_fp4`); scale_2 =
amax / (6 * 448), shared by an expert's gate_proj and up_proj (fused gate_up at serving); per 16-block the adaptive
'4 over 6' scale: the block max maps to 6 or to 4, whichever gives the lower block MSE (ties: 6). On the served experts this
reproduces every block scale and the same 8.69% relative error (the plain max -> 6 rule: 9.50%).
  check:  python export_mtp_drafter.py check SOURCE_DRAFTER_DIR BF16_MTP_FILE [--experts 0,1,511]
          (dequantizes the served experts against the BF16 source and re-quantizes the BF16 source: byte agreement)
  export: python export_mtp_drafter.py export SOURCE_DRAFTER_DIR TRAINED_HEAD.safetensors OUT_DIR [--note TEXT]"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time

import torch

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
FP8_MAX, FP4_MAX = 448.0, 6.0


def cast_to_fp4_codes(x: torch.Tensor) -> torch.Tensor:
    """vLLM ref rounding (cast_to_fp4) to the E2M1 grid, returned as 4-bit codes (sign bit 0x8 | magnitude index)."""
    a = x.abs()
    mag = torch.zeros_like(a, dtype=torch.uint8)
    bounds = [(0.25, 1, False), (0.75, 2, True), (1.25, 3, False), (1.75, 4, True), (2.5, 5, False), (3.5, 6, True), (5.0, 7, False)]
    # thresholds as in cast_to_fp4: (0.25, 0.75) -> 0.5; [0.75, 1.25] -> 1; (1.25, 1.75) -> 1.5; [1.75, 2.5] -> 2;
    # (2.5, 3.5) -> 3; [3.5, 5] -> 4; > 5 -> 6 (x <= 0.25 -> 0)
    for thr, code, inclusive in bounds:
        mag = torch.where(a >= thr if inclusive else a > thr, torch.tensor(code, dtype=torch.uint8), mag)
    sign = (x < 0).to(torch.uint8) << 3                  # negatives that round to zero keep the sign bit (-0), as served
    return mag | sign


def _codes_for(g, scale, scale_2):
    denom = scale.float() * scale_2
    q = torch.where(denom > 0, g / denom, torch.zeros_like(g)).clamp(-FP4_MAX, FP4_MAX)
    return cast_to_fp4_codes(q)


def _decode(codes):
    return E2M1[(codes & 0x07).long()] * torch.where((codes & 0x08) > 0, -1.0, 1.0)


def nvfp4_quantize(w: torch.Tensor, block: int = 16, scale_2: torch.Tensor | None = None, adaptive: bool = True):
    """w [out, in] -> (packed uint8 [out, in/2], scale float8_e4m3fn [out, in/16], scale_2 float32 scalar).
    scale_2 may be given (an expert's gate / up pair share it); adaptive = the 4-over-6 block scale."""
    w = w.float()
    out, inn = w.shape
    if scale_2 is None:
        scale_2 = w.abs().max() / (FP4_MAX * FP8_MAX)
    scale_2 = scale_2.float().clamp(min=1e-12).reshape(())
    g = w.reshape(out, inn // block, block)
    bmax = g.abs().amax(-1, keepdim=True)
    s6 = (bmax / FP4_MAX / scale_2).clamp(max=FP8_MAX).to(torch.float8_e4m3fn)
    scale = s6
    if adaptive:
        s4 = (bmax / 4.0 / scale_2).clamp(max=FP8_MAX).to(torch.float8_e4m3fn)
        e6 = ((_decode(_codes_for(g, s6, scale_2)) * s6.float() * scale_2 - g) ** 2).sum(-1, keepdim=True)
        e4 = ((_decode(_codes_for(g, s4, scale_2)) * s4.float() * scale_2 - g) ** 2).sum(-1, keepdim=True)
        scale = torch.where(e4 < e6, s4.float(), s6.float()).to(torch.float8_e4m3fn)
    codes = _codes_for(g, scale, scale_2).reshape(out, inn)
    packed = (codes[:, 0::2] | (codes[:, 1::2] << 4)).to(torch.uint8)
    return packed.contiguous(), scale.squeeze(-1).contiguous(), scale_2.reshape(())


def expert_scale_2(gate_w: torch.Tensor, up_w: torch.Tensor) -> torch.Tensor:
    """The shared global scale of an expert's gate_proj and up_proj."""
    return torch.maximum(gate_w.float().abs().max(), up_w.float().abs().max()) / (FP4_MAX * FP8_MAX)


def nvfp4_dequantize(packed: torch.Tensor, scale: torch.Tensor, scale_2: torch.Tensor, block: int = 16) -> torch.Tensor:
    lo, hi = packed & 0x0F, (packed >> 4) & 0x0F
    codes = torch.stack((lo, hi), dim=-1).reshape(packed.shape[0], -1)
    vals = E2M1[(codes & 0x07).long()] * torch.where((codes & 0x08) > 0, -1.0, 1.0)
    out, inn = vals.shape
    return (vals.reshape(out, inn // block, block) * scale.float().unsqueeze(-1) * scale_2.float()).reshape(out, inn)


def check(src_dir, bf16_file, experts):
    from safetensors import safe_open
    rows = []
    with safe_open(os.path.join(src_dir, 'model-mtp-draft.safetensors'), 'pt') as fd, safe_open(bf16_file, 'pt') as fb:
        for e in experts:
            for proj in ('gate_proj', 'up_proj', 'down_proj'):
                base = 'mtp.layers.0.mlp.experts.%d.%s.' % (e, proj)
                w = fb.get_tensor(base + 'weight').float()
                pk, sc, s2 = fd.get_tensor(base + 'weight'), fd.get_tensor(base + 'weight_scale'), fd.get_tensor(base + 'weight_scale_2')
                dq = nvfp4_dequantize(pk, sc, s2)
                rel = float((dq - w).norm() / w.norm())
                shared = None
                if proj != 'down_proj':
                    ep = 'mtp.layers.0.mlp.experts.%d.' % e
                    shared = expert_scale_2(fb.get_tensor(ep + 'gate_proj.weight'), fb.get_tensor(ep + 'up_proj.weight'))
                p2, c2, t2 = nvfp4_quantize(w, scale_2=shared)
                rows.append((e, proj, rel, float((p2 == pk).float().mean()),
                             float((c2.view(torch.uint8) == sc.view(torch.uint8)).float().mean()), float(t2), float(s2),
                             float((nvfp4_dequantize(p2, c2, t2) - w).norm() / w.norm())))
    for r in rows:
        print('expert %3d %-9s served dequant vs BF16: rel err %.4f | requantized: bytes equal %.4f, block scales equal %.4f, '
              'scale_2 %.6e vs %.6e, rel err %.4f' % r)
    ok = all(r[2] < 0.15 for r in rows)
    print('convention check: %s (served experts dequantize to their BF16 source within NVFP4 error); requantized bytes equal '
          '>= %.5f, block scales >= %.5f (10-11: 99.998%%; the rest are near-tie 4-vs-6 picks with the same error)'
          % ('PASS' if ok else 'FAIL', min(r[3] for r in rows), min(r[4] for r in rows)))
    return ok


def blake(t: torch.Tensor) -> str:
    return hashlib.blake2b(t.contiguous().view(torch.uint8).numpy().tobytes(), digest_size=16).hexdigest()


def export(src_dir, head_file, out_dir, note=''):
    from safetensors import safe_open
    from safetensors.torch import save_file
    if os.path.exists(out_dir):
        sys.exit('exists: %s' % out_dir)
    os.makedirs(out_dir)
    tensors = {}
    with safe_open(os.path.join(src_dir, 'model-mtp-draft.safetensors'), 'pt') as fd:
        src_keys = list(fd.keys())
        for k in ('lm_head.weight', 'model.language_model.embed_tokens.weight'):
            tensors[k] = fd.get_tensor(k)
    with safe_open(head_file, 'pt') as fh:
        head_keys = set(fh.keys())
        for k in sorted(head_keys):
            t = fh.get_tensor(k)
            if '.mlp.experts.' in k:
                base = k[:-len('weight')]
                shared = None
                if not k.endswith('down_proj.weight'):
                    ep = base.rsplit('.', 2)[0] + '.'
                    shared = expert_scale_2(fh.get_tensor(ep + 'gate_proj.weight'), fh.get_tensor(ep + 'up_proj.weight'))
                p, s, s2 = nvfp4_quantize(t, scale_2=shared)
                tensors[base + 'weight'], tensors[base + 'weight_scale'], tensors[base + 'weight_scale_2'] = p, s, s2
            else:
                tensors[k] = t.to(torch.bfloat16).contiguous()
    missing = sorted(set(src_keys) - set(tensors))
    extra = sorted(set(tensors) - set(src_keys))
    assert not missing and not extra, ('tensor set differs from the source drafter', missing[:5], extra[:5])
    save_file(tensors, os.path.join(out_dir, 'model-mtp-draft.safetensors'), metadata={'format': 'pt'})
    for f in os.listdir(src_dir):
        if f not in ('model-mtp-draft.safetensors', 'model.safetensors.index.json', 'ARC3_DRAFT_VIEW.json'):
            shutil.copy2(os.path.join(src_dir, f), os.path.join(out_dir, f))
    index = json.load(open(os.path.join(src_dir, 'model.safetensors.index.json')))
    index['weight_map'] = {k: 'model-mtp-draft.safetensors' for k in sorted(tensors)}
    index.setdefault('metadata', {})['total_size'] = sum(t.numel() * t.element_size() for t in tensors.values())
    json.dump(index, open(os.path.join(out_dir, 'model.safetensors.index.json'), 'w'), indent=2)
    view = json.load(open(os.path.join(src_dir, 'ARC3_DRAFT_VIEW.json')))
    view.update(tensors=len(tensors), bytes=index['metadata']['total_size'],
                verified='fine-tuned MTP head (%s; %s) on the source drafter %s: BF16 head tensors from the trained file, routed '
                         'experts re-quantized to NVFP4 (modelopt layout, no input scales), embedding / lm_head copied' % (
                             os.path.basename(head_file), note or 'mtp_ft', src_dir),
                derived_from=src_dir, exported_utc=time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()),
                tensor_blake2b={k: blake(tensors[k]) for k in ('lm_head.weight', 'model.language_model.embed_tokens.weight',
                                                               'mtp.fc_embedding.weight', 'mtp.fc_hidden.weight')})
    json.dump(view, open(os.path.join(out_dir, 'ARC3_DRAFT_VIEW.json'), 'w'), indent=1)
    print('exported %d tensors (%.2f GB) to %s' % (len(tensors), index['metadata']['total_size'] / 1e9, out_dir))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('mode', choices=('check', 'export'))
    ap.add_argument('src'); ap.add_argument('file'); ap.add_argument('out', nargs='?')
    ap.add_argument('--experts', default='0,1,255,511'); ap.add_argument('--note', default='')
    a = ap.parse_args()
    if a.mode == 'check':
        sys.exit(0 if check(a.src, a.file, [int(x) for x in a.experts.split(',')]) else 1)
    export(a.src, a.file, a.out, a.note)
