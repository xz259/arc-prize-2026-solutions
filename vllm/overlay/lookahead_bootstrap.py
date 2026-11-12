"""Isolated early-gameplay canary: unchanged scheduler, no image diagnostics."""
import hashlib
import json
import os
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
_receipt = None


def install():
    global _receipt
    pins = json.loads((HERE/'gameplay-canary-pins.json').read_text())
    for name, digest in pins['installed_files'].items():
        if hashlib.sha256((HERE/name).read_bytes()).hexdigest() != digest:
            raise RuntimeError('Unreviewed gameplay-canary file: '+name)
    if (os.environ.get('ARC3_PERSISTENT_ONE_LOOKAHEAD') != '1'
            or os.environ.get('ARC3_PERSISTENT_ONE_LOOKAHEAD_IMAGE_DIAGNOSTIC') != '1'
            or os.environ.get('ARC3_PERSISTENT_SINGLE_BATCH') != '1'
            or os.environ.get('ARC3_ENGINE_TIMING_ENABLE')):
        raise RuntimeError('Wrong isolated gameplay-canary configuration')
    if any(name == 'image_bridge' or name.startswith('lookahead_image_oracle')
           or name.startswith('lookahead_snapshot_v2') for name in sys.modules):
        raise RuntimeError('Prepared-image bridge/state oracle must remain uninstalled')
    if _receipt is not None:
        return _receipt
    import engine_patch
    original = engine_patch.install()
    _receipt = dict(schema='arc3-lookahead-gameplay-canary-bootstrap-v1', pid=os.getpid(),
        engine_installation=original, parent_manifest=pins['parent_manifest'],
        source_only_at_preparation=True, early_gameplay_canary=True,
        prepared_image_bridge=False, image_state_oracle=False, marked_eviction=True,
        native_context_and_mm_limits=True)
    out = HERE/'lookahead-gameplay-startup'
    out.mkdir(exist_ok=True)
    with (out/f'process-{os.getpid()}.json').open('x') as handle:
        json.dump(_receipt, handle, indent=2)
        handle.write('\n')
    return _receipt
