"""state_to_head.py (2026-10-14, Claude fork 5d20b23b): a scoreable head from full-run states, on the CPU, without
loading the model. One state: its trained tensors (FP32 masters) as a trainable-only head, exactly what
train_mtp_sp.save_head(trainable_only=True) writes (BF16, 'mtp.' + name, embed_tokens / lm_head / target_mixer left out),
which mtp_eval_val50.py overlays on the base. Several states: their uniform average (IE 10-14: checkpoint averaging over
the late decay states; all states must come from one run with the same trained tensors and the same frozen-expert
source). usage: python state_to_head.py OUT.safetensors STATE [STATE ...]"""
import sys

import torch
from safetensors.torch import save_file

SKIP = ('embed_tokens', 'lm_head', 'target_mixer')


def main(out, paths):
    acc, keys, steps, meta = None, None, [], None
    for p in paths:
        st = torch.load(p, map_location='cpu', weights_only=False)
        m = {n: t for n, t in st['model'].items() if not n.startswith(SKIP)}
        if keys is None:
            keys, meta = sorted(m), st.get('meta', {})
            acc = {n: m[n].double().clone() for n in keys}
        else:
            if sorted(m) != keys:
                raise SystemExit('%s: different trained tensors' % p)
            if st.get('meta', {}) != meta:
                raise SystemExit('%s: different state meta %s vs %s' % (p, st.get('meta'), meta))
            for n in keys:
                acc[n] += m[n].double()
        steps.append(st['step'])
        del st, m
    head = {'mtp.' + n: (acc[n] / len(paths)).to(torch.bfloat16).contiguous() for n in keys}
    src = 'state_to_head.py: %s of steps %s' % ('uniform average' if len(paths) > 1 else 'state', steps)
    save_file(head, out, metadata={'format': 'pt', 'source': src})
    print('%s: %d tensors, %.1fM parameters (%s)' % (out, len(head), sum(t.numel() for t in head.values()) / 1e6, src))


if __name__ == '__main__':
    if len(sys.argv) < 3:
        raise SystemExit(__doc__)
    main(sys.argv[1], sys.argv[2:])
