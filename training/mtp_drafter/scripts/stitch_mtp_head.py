"""stitch_mtp_head.py (2026-10-15, Claude fork 5d20b23b): a servable drafter folder for a fine-tuned MTP head trained with the
production drafter's routed experts and router frozen (fr1 recipe: 25 trained BF16 tensors, state_to_head.py heads). The source
drafter folder's model-mtp-draft.safetensors is rewritten with the head file's tensors replaced; every other tensor (NVFP4 routed
experts, router, embedding, lm_head, indexer, ...) is copied byte for byte. export_mtp_drafter.py export needs the full BF16 tensor
set and re-quantizes the experts, so it does not apply to frozen-expert heads.
Checks: every head tensor exists in the source with the same shape and dtype (no other name may appear); the written file is re-read
and every tensor compared: head tensors bitwise equal to the head file, all others bitwise equal to the source. The small files are
copied, the index is rewritten from the written file, and ARC3_DRAFT_VIEW.json keeps `source` (the launcher's check) and gains
`stitched` (head file, its metadata, the replaced names with blake2b of the old and new bytes, UTC time).
usage: python stitch_mtp_head.py SOURCE_DRAFTER_DIR HEAD.safetensors OUT_DIR [--note TEXT]     (refuses if OUT_DIR exists)"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time

import torch
from safetensors import safe_open
from safetensors.torch import save_file

FILE = 'model-mtp-draft.safetensors'


def u8(t):
    """Raw bytes as a flat uint8 tensor (0-dim scalars included)."""
    return t.contiguous().reshape(-1).view(torch.uint8)


def blake(t):
    return hashlib.blake2b(u8(t).numpy().tobytes(), digest_size=16).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src'); ap.add_argument('head'); ap.add_argument('out'); ap.add_argument('--note', default='')
    a = ap.parse_args()
    if os.path.exists(a.out):
        sys.exit('exists: %s' % a.out)
    t0 = time.time()
    with safe_open(a.head, 'pt') as fh:
        head = {k: fh.get_tensor(k) for k in fh.keys()}
        head_meta = fh.metadata() or {}
    with safe_open(os.path.join(a.src, FILE), 'pt') as fs:
        src_meta = fs.metadata() or {'format': 'pt'}
        keys = list(fs.keys())
        missing = sorted(set(head) - set(keys))
        assert not missing, ('head tensors absent from the source drafter', missing)
        tensors, replaced = {}, {}
        for k in keys:
            t = fs.get_tensor(k)
            if k in head:
                h = head[k]
                assert h.shape == t.shape and h.dtype == t.dtype, (k, tuple(h.shape), h.dtype, tuple(t.shape), t.dtype)
                replaced[k] = dict(old=blake(t), new=blake(h), changed=not torch.equal(u8(h), u8(t)))
                t = h
            tensors[k] = t.contiguous()
    os.makedirs(a.out)
    save_file(tensors, os.path.join(a.out, FILE), metadata=src_meta)
    # re-read and compare every tensor
    bad = []
    with safe_open(os.path.join(a.out, FILE), 'pt') as fo, safe_open(os.path.join(a.src, FILE), 'pt') as fs:
        assert sorted(fo.keys()) == sorted(keys), 'tensor set changed'
        for k in keys:
            o = fo.get_tensor(k)
            ref = head[k] if k in head else fs.get_tensor(k)
            if not (o.dtype == ref.dtype and o.shape == ref.shape and torch.equal(u8(o), u8(ref))):
                bad.append(k)
    assert not bad, ('written tensors differ', bad[:5])
    for f in os.listdir(a.src):
        if f not in (FILE, 'model.safetensors.index.json', 'ARC3_DRAFT_VIEW.json'):
            shutil.copy2(os.path.join(a.src, f), os.path.join(a.out, f))
    index = json.load(open(os.path.join(a.src, 'model.safetensors.index.json')))
    index['weight_map'] = {k: FILE for k in sorted(keys)}
    index.setdefault('metadata', {})['total_size'] = sum(t.numel() * t.element_size() for t in tensors.values())
    with open(os.path.join(a.out, 'model.safetensors.index.json'), 'w') as f:
        json.dump(index, f, indent=2)
    view = json.load(open(os.path.join(a.src, 'ARC3_DRAFT_VIEW.json')))
    view['stitched'] = dict(head_file=os.path.abspath(a.head), head_metadata=head_meta, derived_from=os.path.abspath(a.src),
                            note=a.note, utc=time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime()), replaced=replaced,
                            rule='head tensors replaced (BF16, bitwise); every other tensor byte-identical to derived_from')
    with open(os.path.join(a.out, 'ARC3_DRAFT_VIEW.json'), 'w') as f:
        json.dump(view, f, indent=1)
    nchg = sum(r['changed'] for r in replaced.values())
    print('stitched %s: %d tensors, %d replaced (%d changed bytes), all others byte-identical to %s; written file re-read and '
          'verified; %.2f GB, %.0f s' % (a.out, len(keys), len(replaced), nchg, a.src, index['metadata']['total_size'] / 1e9,
                                         time.time() - t0))


if __name__ == '__main__':
    main()
