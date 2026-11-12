"""Inactive experimental exact QSA selector; loaded only when explicitly called.

The shared library is built separately and identified by a SHA-256 manifest.
The selector uses the caller's current CUDA stream and launches no host transfer.
Only top-512 float32 rows are supported by this initial isolated implementation.
"""
import ctypes
import hashlib
import json
from pathlib import Path

import torch

_library = None
_initialized_devices = set()


def load():
    global _library
    if _library is None:
        manifest = json.loads((Path(__file__).parent/'kernels/build/latest.json').read_text())
        path = Path(manifest['library'])
        if not path.is_absolute():
            path = Path(__file__).parent/path
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['library_sha256']
        library = ctypes.CDLL(str(path))
        library.arc3_radix_initialize.argtypes = [ctypes.c_int]
        library.arc3_radix_initialize.restype = ctypes.c_int
        library.arc3_radix_topk.argtypes = [ctypes.c_void_p]*3 + [ctypes.c_int]*2 + [
            ctypes.c_int64, ctypes.c_int, ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        library.arc3_radix_topk.restype = ctypes.c_int
        library.arc3_radix_error_string.argtypes = [ctypes.c_int]
        library.arc3_radix_error_string.restype = ctypes.c_char_p
        _library = library
    return _library


def _check(code):
    if code:
        raise RuntimeError(load().arc3_radix_error_string(code).decode())


def radix_topk(logits, lengths, output, k, columns, threads=256, cache_row=True):
    if k != 512 or threads not in [128,256,512,1024]:
        raise ValueError('Experimental selector requires k=512 and a supported thread count')
    if not (logits.is_cuda and logits.device == lengths.device == output.device):
        raise ValueError('All selector tensors must share one CUDA device')
    if (logits.dtype != torch.float32 or lengths.dtype != torch.int32 or output.dtype != torch.int32
            or logits.ndim != 2 or lengths.ndim != 1 or output.ndim != 2):
        raise ValueError('Unsupported selector tensor dtype or rank')
    rows = logits.shape[0]
    if (not lengths.is_contiguous() or not output.is_contiguous() or logits.stride(1) != 1
            or lengths.numel() != rows or output.shape != (rows,k)
            or not 0 <= columns <= logits.shape[1] or logits.stride(0) < columns):
        raise ValueError('Unsupported selector shapes or strides')
    library = load()
    with torch.cuda.device(logits.device):
        device = logits.device.index
        if device not in _initialized_devices:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError('Initialize the selector before CUDA graph capture')
            _check(library.arc3_radix_initialize(device))
            _initialized_devices.add(device)
        _check(library.arc3_radix_topk(logits.data_ptr(), lengths.data_ptr(), output.data_ptr(),
            rows, columns, logits.stride(0), threads, device,
            torch.cuda.current_stream().cuda_stream, int(cache_row)))
