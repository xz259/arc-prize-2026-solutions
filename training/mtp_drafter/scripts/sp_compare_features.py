"""sp_compare_features.py (2026-10-12, Claude fork 5d20b23b): bitwise comparison of two dflashtaps feature files of the same
session (side-pod replay vs the main pod's job 407 export). Gate for SIDE_POD_PLAN 1.4: same tensor names, shapes and dtypes,
and identical bytes. If not identical, reports per tensor the count of differing elements and the max abs / relative
difference, so a pod difference can be told from a rounding-level one.
usage: python sp_compare_features.py A.safetensors B.safetensors   (run with venv-speculators' python: torch + safetensors)"""
import sys

import torch
from safetensors import safe_open


def main(a_path, b_path):
    with safe_open(a_path, 'pt') as fa, safe_open(b_path, 'pt') as fb:
        ka, kb = set(fa.keys()), set(fb.keys())
        print('A', a_path, len(ka), 'tensors;', 'B', b_path, len(kb), 'tensors')
        if ka != kb:
            print('KEY MISMATCH: only A', sorted(ka - kb)[:10], 'only B', sorted(kb - ka)[:10])
        ma, mb = fa.metadata() or {}, fb.metadata() or {}
        if ma != mb:
            print('metadata differs:', {k: (ma.get(k), mb.get(k)) for k in set(ma) | set(mb) if ma.get(k) != mb.get(k)})
        same = 0; diff = []
        for k in sorted(ka & kb):
            x, y = fa.get_tensor(k), fb.get_tensor(k)
            if x.shape != y.shape or x.dtype != y.dtype:
                diff.append((k, f'shape/dtype {tuple(x.shape)} {x.dtype} vs {tuple(y.shape)} {y.dtype}')); continue
            xb = x.contiguous().view(torch.uint8) if x.element_size() == 1 else x.contiguous().view(torch.int16 if x.element_size() == 2 else torch.int32 if x.element_size() == 4 else torch.int64)
            yb = y.contiguous().view(xb.dtype)
            if torch.equal(xb, yb):
                same += 1; continue
            n = int((xb != yb).sum())
            if x.is_floating_point():
                d = (x.float() - y.float()).abs(); scale = y.float().abs().max().clamp_min(1e-30)
                diff.append((k, f'{n} of {x.numel()} elements differ; max abs {float(d.max()):.3e}; max abs / max |B| {float(d.max() / scale):.3e}'))
            else:
                diff.append((k, f'{n} of {x.numel()} elements differ (integer tensor)'))
        print(f'identical tensors {same} of {len(ka & kb)}')
        for k, msg in diff:
            print('  DIFF', k, msg)
        ok = ka == kb and not diff
        print('GATE', 'IDENTICAL' if ok else 'NOT IDENTICAL')
        return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main(*sys.argv[1:3]))
