"""Pure contracts for explicit private-module identity and comparison order."""
import ast
import hashlib
import importlib
import importlib.util
import inspect
import json
from pathlib import Path
import textwrap

from .nvrtc_dependency import verify_nvrtc_header
from .link_dependency import verify_link_dependencies

HERE = Path(__file__).resolve().parent
INIT_FIELDS = ('x_dtype', 'weight_dtype', 'output_dtype',
               'use_deepseek_fp8_block_scale', 'use_w4_group_scaling',
               'use_mxfp8_act_scaling', 'use_packed_weights', 'use_fused_finalize')
BASELINE = (17, 37)
LOADER_MANIFEST_SHA = '435ec2533fb8eb569df62ad38f24298a0eeea86199f6a779b502a7d1a31d8e00'
PREPARER_SHA = '8d08c15beaaa21f5534cfff74eef3deef169a8c66184f2c22a0c8e5f6b98ee14'
OVERLAY_RECEIPT_SHA = 'f23e4168777cbc52c03cf0fc4c5f7cec4d48d77c18cb3ef2660a251f8815c3f8'
OVERLAY_FILES = {
    'fused_moe/cutlass_backend/cutlass_fused_moe_kernels.cuh':
        '791bfb02c9d77615ea1e683056855f156bfaf143a70f57cd76a1908da6c99f95',
    'fused_moe/cutlass_backend/flashinfer_cutlass_fused_moe_binding.cu':
        '4ebe336349e768e4a8bea1be6f7d0c50400f4f3d73d64fab34e19f63b112eec3',
}


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def ast_sha(obj):
    code = textwrap.dedent(inspect.getsource(obj))
    return hashlib.sha256(ast.dump(ast.parse(code), include_attributes=False).encode()).hexdigest()


def dependencies():
    # Legacy AST values are preserved as historical metadata only. The actual
    # packaged defining modules and their exports are the current authority.
    spec = importlib.util.find_spec('moe_one_layer_replay_v3')
    require(spec is not None and spec.origin is not None, 'Pinned loader-v3 helper authority is required')
    root = Path(spec.origin).resolve().parent
    require(sha(root / 'MANIFEST.json') == LOADER_MANIFEST_SHA, 'Helper authority manifest differs')
    for relative, expected in json.loads((root / 'MANIFEST.json').read_text()).items():
        path = (root / relative).resolve(strict=True)
        require(path.is_relative_to(root) and sha(path) == expected, 'Helper authority source changed')
    authority = importlib.import_module('moe_one_layer_replay_v3.contracts')
    require(Path(authority.__file__).resolve() == root / 'contracts.py', 'Helper authority namespace collision')
    exports = json.loads((root / 'HELPER_EXPORTS.json').read_text())
    required = set(json.loads((HERE / 'REQUIRED_HELPERS.json').read_text()))
    require(required <= set(exports), 'Required helper mapping absent from current authority')
    package = importlib.util.find_spec('arc3_moe_probe')
    require(package is not None and package.origin is not None, 'Frozen helper package is absent')
    helper_root = Path(package.origin).resolve().parent.parent
    authority.helper_preflight({'helper_root': str(helper_root)})
    modules = {name: importlib.import_module(name) for name in {key.split(':')[0] for key in exports}}
    verified = authority.verify_helper_exports(modules, str(helper_root))
    return {name.split(':')[1]: verified[name.split(':')[1]] for name in required}


def init_args(record):
    runner = record['source_runner']
    require(runner.fused_moe_runner is record['raw_runner'], 'Captured native runner identity differs')
    values = tuple(getattr(runner, key) for key in INIT_FIELDS)
    require(all(value is False for value in values[3:]), 'Unsupported scaling/packed/atomic finalization path')
    return values


def inventory(raw):
    first, second, total = (int(raw.get_gemm1_tactic_count()), int(raw.get_gemm2_tactic_count()),
                            int(raw.get_tactic_num()))
    require((first, second, total) == (20, 20, 40), 'Native tactic enumeration differs from baseline')
    occupancy = [int(raw.get_tactic_occupancy(i)) for i in range(total)]
    require(all(v >= 0 for v in occupancy), 'Invalid occupancy response')
    return dict(stage_counts=[first, second], occupancy=occupancy,
                scope='Interface/config parity only; positive occupancy does not prove tile dispatch',
                executed_pair=list(BASELINE))


def validate_config(config):
    c = dict(config)
    require(c.get('execute') is True, 'Explicit bounded execution required')
    require(c.get('input_boundary', 'bf16') in ('bf16', 'packed'), 'Unknown input boundary')
    c.setdefault('input_boundary', 'bf16')
    c.setdefault('rounds', 8); c.setdefault('calls_per_graph', 4); c.setdefault('replays', 4)
    c.setdefault('max_seconds', 900); c.setdefault('max_workspace_bytes', 2 << 30)
    require(type(c['rounds']) is int and c['rounds'] in (4, 8, 12, 16), 'Balanced4-sample cycles required')
    require(type(c['calls_per_graph']) is int and 1 <= c['calls_per_graph'] <= 8, 'Capture call bound exceeded')
    require(type(c['replays']) is int and 1 <= c['replays'] <= 16, 'Replay bound exceeded')
    require(0 < c['max_seconds'] <= 900 and 0 < c['max_workspace_bytes'] <= 2 << 30,
            'Runtime/workspace bound exceeded')
    for key in ('prepared_root', 'build_result', 'private_library', 'output'):
        p = Path(c[key]); require(p.is_absolute(), 'Explicit absolute path required: ' + key)
        c[key] = str(p.resolve())
    for key in ('build_result_sha256', 'private_library_sha256'):
        require(isinstance(c[key], str) and len(c[key]) == 64, 'Explicit source receipt required: ' + key)
    require(not Path(c['output']).exists(), 'Validation output must be fresh')
    return c


def verify_build(c):
    root = Path(c['prepared_root']); receipt = Path(c['build_result']); library = Path(c['private_library'])
    require(receipt.is_relative_to(root), 'Build receipt outside preparation')
    require(sha(receipt) == c['build_result_sha256'], 'Build receipt changed')
    built = json.loads(receipt.read_text())
    require(built['status'] == 'built-not-loaded-or-validated' and built['cleanup']['complete'],
            'Build did not complete/drain cleanly')
    require(built['preparation_unchanged'] and built['ninja_unchanged'] and
            not built['storage_ceiling_exceeded'], 'Build provenance/resource gate failed')
    require(Path(built['library']['path']).resolve() == library and
            built['library']['sha256'] == c['private_library_sha256'] == sha(library), 'Private library identity differs')
    require(library.is_relative_to(root / 'build') and not library.is_symlink(), 'Unexpected private library path')
    require(sha(root / 'PREPARE_RESULT.json') == built['prepared_receipt_sha256'], 'Prepared source receipt changed')
    plan = json.loads((root / 'PREPARE_RESULT.json').read_text())
    require(plan['script_sha256'] == PREPARER_SHA and
            plan['overlay_receipt_sha256'] == OVERLAY_RECEIPT_SHA,
            'Private build is not the reviewed routing-v4 materialization')
    for relative, expected in OVERLAY_FILES.items():
        require(sha(root / 'csrc' / relative) == expected, 'Private native overlay changed')
    require(Path(plan['native_library_path']).resolve() == library, 'Prepared native target differs')
    require(sha(plan['ninja_path']) == plan['ninja_sha256'], 'Prepared Ninja changed')
    nvrtc = verify_nvrtc_header(plan, root)
    require(built.get('nvrtc_header_unchanged') is True and built.get('nvrtc_header_verified') == nvrtc,
            'Build did not preserve its pinned isolated NVRTC header')
    link = verify_link_dependencies(plan, root)
    require(built.get('link_libraries_unchanged') is True and built.get('link_libraries_verified') == link,
            'Build did not preserve pinned CUDA link dependencies')
    require(len(plan['private_sources']) == 96, 'Not the complete reviewed translation-unit inventory')
    for item in plan['private_sources']:
        require(sha(item['path']) == item['sha256'], 'Prepared compilation source changed')
    include = root / 'csrc/nv_internal/tensorrt_llm'
    require(include.is_dir(), 'Private DeepGEMM include root absent')
    return built, plan, include


def mapped_paths(path=Path('/proc/self/maps')):
    result = set()
    for line in Path(path).read_text().splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) == 6 and fields[5].startswith('/'):
            require(not fields[5].endswith(' (deleted)') or 'arc3_private_moe_sm120_' not in fields[5],
                    'Deleted private library remains mapped')
            result.add(fields[5])
    return result


def schedule(rounds):
    require(rounds % 4 == 0, 'Timing must cross both banks with both orders')
    return [((i // 2) % 2, bool(i % 2)) for i in range(rounds)]


def validate_session(session, identity, n):
    require(n in (25, 30), 'Only declared N25/N30 session cohorts')
    if session is not None:
        require(session['identity'] == identity and session['pending_n'] is None and
                n not in session['completed_n'] and len(session['completed_n']) == 1,
                'Only one clean N25 and one clean N30 cohort may share the same private session')


def assert_exact(byte_equal, actual, expected, label):
    require(byte_equal(actual, expected), label + ': output differs byte-for-byte')
