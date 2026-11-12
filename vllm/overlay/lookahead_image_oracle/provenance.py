"""Exact source bytes, verified at each untimed snapshot boundary."""
import hashlib
import importlib
import json
from pathlib import Path
import sys
from .receipt_schema import require, json_digest, verify_inventory


def verify_sources():
    own = Path(__file__).resolve().parent
    manifest = json.loads((own/'SOURCE_MANIFEST.json').read_text())
    require(manifest['schema'] == 'arc3-lookahead-snapshot-sources-v1', 'Wrong source manifest')
    for name, digest in manifest['own_files'].items():
        path = own/name
        require(path.parent == own and hashlib.sha256(path.read_bytes()).hexdigest() == digest,
                'Unreviewed oracle source: '+name)
    import vllm
    runtime = Path(vllm.__file__).resolve().parent
    for name, digest in manifest['runtime_files'].items():
        path = runtime/name
        require(path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() == digest,
                'Unreviewed native runtime source: '+name)
    for name, digest in manifest['helper_modules'].items():
        module = importlib.import_module(name)
        path = Path(module.__file__).resolve()
        require(path.suffix == '.py' and hashlib.sha256(path.read_bytes()).hexdigest() == digest,
                'Unreviewed active helper: '+name)
    require(sys.byteorder == 'little', 'Byte oracle requires little-endian host')
    verify_inventory()
    return manifest, json_digest(manifest)
