"""Exact isolated NVRTC header dependency; no compilation or native loading."""
import hashlib
from pathlib import Path

HEADER_SHA = '9e4415b5ff5a2c58ec2b1a02eefcc743187c94ce48b00743c68092efe8c0f86e'
HEADER_BYTES = 44887


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_nvrtc_header(plan, prepared):
    """Recheck original installed bytes and the sole isolated include file."""
    prepared = Path(prepared).resolve()
    row = plan['nvrtc_header']
    require(set(row) == {'source_path', 'destination_path', 'sha256', 'bytes', 'cuda_nvrtc_version'}, 'NVRTC receipt fields differ')
    require(row['sha256'] == HEADER_SHA and row['bytes'] == HEADER_BYTES, 'NVRTC header identity differs')
    require(row['cuda_nvrtc_version'] == '13.0.88', 'NVRTC version differs from the verified NVCC build')
    source, target = Path(row['source_path']), Path(row['destination_path'])
    require(source.is_absolute() and target.is_absolute(), 'NVRTC paths must be absolute')
    require(source.parts[-4:] == ('nvidia', 'cu13', 'include', 'nvrtc.h'), 'NVRTC installed source path differs')
    require(not source.is_symlink() and source.is_file(), 'NVRTC source must be a regular installed file')
    include = prepared / 'include-nvrtc'
    require(not include.is_symlink() and include.is_dir() and target == include / 'nvrtc.h',
            'NVRTC isolated destination differs')
    require(list(include.iterdir()) == [target] and not target.is_symlink() and target.is_file(),
            'NVRTC include directory must contain only its pinned header')
    for path in (source, target):
        require(path.stat().st_size == HEADER_BYTES and digest(path) == HEADER_SHA, 'NVRTC header bytes changed')
    require(plan['include_dirs'].count(str(include)) == 1 and str(source.parent) not in plan['include_dirs'],
            'NVRTC include must use only the isolated header directory')
    ninja = Path(plan['ninja_path']).read_text()
    require('-I' + str(include) in ninja, 'Effective Ninja lacks the isolated NVRTC include')
    return dict(row, isolated_include=str(include), exact=True)
