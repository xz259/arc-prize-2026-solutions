"""Reversible import-only sampler patch; the startup flag defaults off.

The guarded implementation must be the exact source that passed the isolated
operator screen. The stock Triton kernel is preserved. Stop serving before
applying or reverting; application does not launch a server or promote a flag.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BUNDLE = ROOT/'staged/sampler_guard_integration_v1'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def prepare(root, site, bundle):
    manifest = json.loads((bundle/'source-manifest.json').read_text())
    for name, expected in manifest['files'].items():
        path = (bundle/name).resolve()
        assert path.is_relative_to(bundle.resolve())
        assert sha(path.read_bytes()) == expected, name
    for name in ['arc3_sampler_topk_guard.py', 'arc3_sampler_topk_guard_impl.py',
                 'arc3_sampler_guard_diagnostics.py']:
        assert sha((root/name).read_bytes()) == manifest['files'][name], name
    assert manifest['files']['arc3_sampler_topk_guard_impl.py'] == manifest['screened_implementation_sha256']
    stock = manifest['original_kernel_source']
    assert sha((site/stock['path']).read_bytes()) == stock['sha256'], 'Stock kernel changed'
    original = (bundle/'topk_topp_sampler_original.py').read_bytes()
    patched = (bundle/'topk_topp_sampler_patched.py').read_bytes()
    record = manifest['original_runtime_source']
    target = site/record['path']
    assert sha(original) == record['sha256']
    assert sha(patched) == manifest['patched_runtime_source']['sha256']
    assert sha(target.read_bytes()) in {sha(original), sha(patched)}, 'Runtime dispatcher changed'
    compile(original, str(target), 'exec')
    compile(patched, str(target), 'exec')
    return target, original, patched, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--site', type=Path, default=ROOT/'runtime/base/usr/local/lib/python3.12/dist-packages')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--apply', action='store_true')
    mode.add_argument('--revert', action='store_true')
    args = parser.parse_args()
    target, original, patched, manifest = prepare(ROOT, args.site, BUNDLE)
    result = dict(applied=args.apply, reverted=args.revert,
                  path=manifest['original_runtime_source']['path'],
                  original_sha256=sha(original), patched_sha256=sha(patched),
                  flag=manifest['environment_flag'], default=manifest['default_flag'],
                  implementation_sha256=manifest['screened_implementation_sha256'])
    if args.apply or args.revert:
        from minimal_inference import IDENTITY, matches, owned_snapshot, port_open
        identity = json.loads(IDENTITY.read_text())
        assert identity['boot_id'] == Path('/proc/sys/kernel/random/boot_id').read_text().strip()
        assert not port_open() and not matches(identity['root']), 'Stop the owned server before changing source'
        assert not any(matches(row) for row in owned_snapshot(identity)), 'Owned workers remain live'
        backup = ROOT/'patches/sampler-topk-guard-original'/result['path']
        backup.parent.mkdir(parents=True, exist_ok=True)
        if backup.exists():
            assert backup.read_bytes() == original
        else:
            with backup.open('xb') as handle:
                handle.write(original)
        temporary = target.with_name(target.name+'.arc3-sampler-tmp')
        with temporary.open('xb') as handle:
            handle.write(original if args.revert else patched)
        os.replace(temporary, target)
        assert target.read_bytes() == (original if args.revert else patched)
        result['server_identity'] = identity['root']
        result['runtime_source_after_sha256'] = sha(target.read_bytes())
        record = ROOT/'patches/sampler-topk-guard-manifest.json'
        record.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
