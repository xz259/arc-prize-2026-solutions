"""Reversible, hash-guarded QSA/MoE diagnostic hooks for the pinned runtime.

Does not enable new serving defaults. The FlashInfer cache key distinguishes
finalizer modes, preventing incompatible GEMM tactics during the controlled A/B.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SOURCES = {
    'vllm/models/qwen3_8_flash_next/nvidia/ops/qsa.py': 'c4ffe3674cafa0ce2dabc39a39f0ddbb4b594bc358ad210ffce9d04383350c7f',
    'vllm/model_executor/layers/fused_moe/experts/flashinfer_cutlass_moe.py': '52887cec5fe628e0a4e7793deb546b40f6c29f7362b3b436c10ca36934ab9826',
    'flashinfer/fused_moe/core.py': '1fc7f2b942b253837e554b2a4c128e196a19a48679c4a11cdd7677e0cccfd5ca',
}


def sha(data):
    return hashlib.sha256(data).hexdigest()


def once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Patch anchor mismatch: ' + repr(old))
    return source.replace(old, new, 1)


def patched_source(path, original):
    s = original.decode()
    if path.endswith('/qsa.py'):
        s = once(s, 'from __future__ import annotations\n',
                 'from __future__ import annotations\n\nimport arc3_determinism\n')
        s = once(s, '        topk_op(logits, visible_blocks, blocks, topk_workspace, block_topk, columns)\n',
                 '        arc3_determinism.select_topk(topk_op, logits, visible_blocks, blocks, topk_workspace, block_topk, columns)\n')
        s = once(s, '    num_tiles = triton.cdiv(logical_indices.shape[1], block_n)\n',
                 '    block_n, target_splits, partial_warps = arc3_determinism.sparse_prefill_profile(\n'
                 '        q.shape[0], block_n, target_splits, partial_warps)\n'
                 '    num_tiles = triton.cdiv(logical_indices.shape[1], block_n)\n')
    elif path.endswith('/flashinfer_cutlass_moe.py'):
        s = 'import arc3_determinism\n' + s
        s = once(s, '        _ = flashinfer_cutlass_fused_moe(\n',
                 '        _ = flashinfer_cutlass_fused_moe(\n            use_fused_finalize=arc3_determinism.fused_finalize(),\n')
    else:
        s = once(s, '            self.fused_moe_runner = MoERunner.runner_dict[instance_key]\n',
                 '            self.fused_moe_runner = MoERunner.runner_dict[instance_key]\n\n'
                 '        def get_cache_key_extras(self, inputs):\n'
                 '            return super().get_cache_key_extras(inputs) + (self.use_fused_finalize,)\n')
    compile(s, path, 'exec')
    return s.encode()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--site', type=Path, default=ROOT/'runtime/base/usr/local/lib/python3.12/dist-packages')
    p.add_argument('--apply', action='store_true')
    p.add_argument('--revert', action='store_true')
    args = p.parse_args()
    assert not (args.apply and args.revert)
    if args.apply or args.revert:
        from minimal_inference import IDENTITY, owned_snapshot, matches, port_open
        assert not port_open(), 'Stop the owned server before changing source'
        if IDENTITY.exists():
            assert not any(matches(x) for x in owned_snapshot(json.loads(IDENTITY.read_text())))
    manifest = ROOT/'patches/determinism-manifest.json'
    previous = json.loads(manifest.read_text()) if manifest.exists() else None
    previous_files = {r['path']:r for r in previous['files']} if previous else {}
    changes = []
    for rel, expected in SOURCES.items():
        target = args.site/rel
        backup = ROOT/'patches/determinism-original'/rel
        current = target.read_bytes()
        original = current if sha(current) == expected else backup.read_bytes()
        assert sha(original) == expected, rel
        patched = patched_source(rel, original)
        allowed = {expected, sha(patched)}
        if rel in previous_files:
            assert previous_files[rel]['original_sha256'] == expected
            allowed.add(previous_files[rel]['patched_sha256'])
        assert sha(current) in allowed, rel
        changes.append((target, backup, original, patched, dict(path=rel, original_sha256=expected, patched_sha256=sha(patched))))
    if args.apply or args.revert:
        for target, backup, original, patched, row in changes:
            backup.parent.mkdir(parents=True, exist_ok=True)
            if backup.exists():
                assert sha(backup.read_bytes()) == row['original_sha256']
            else:
                backup.write_bytes(original)
            temporary = target.with_name(target.name + '.arc3-tmp')
            temporary.write_bytes(original if args.revert else patched)
            os.replace(temporary, target)
        if args.revert:
            manifest.unlink(missing_ok=True)
        else:
            manifest.write_text(json.dumps(dict(patch='qsa-moe-diagnostics-v2', files=[c[4] for c in changes],
                helper_sha256=sha((ROOT/'arc3_determinism.py').read_bytes())), indent=2)+'\n')
    print(json.dumps(dict(applied=args.apply, reverted=args.revert, files=[c[4] for c in changes]), indent=2))


if __name__ == '__main__':
    main()
