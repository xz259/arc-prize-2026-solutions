"""Measured SM120 launch policy for the experimental quantized sparse readers.

Tile64/two-warps from the BF16 reader spills badly with dequantization. Tile16
and four warps keep the measured kernels spill-free. Use one compiler stage for
decode and NVFP4; FP8 large prefills favor two. This changes no cache bytes.
"""


def quantized_sparse_profile(base_programs, block_m, fmt='fp8'):
    small_limit=8 if block_m<=8 else 4
    if base_programs<=small_limit:
        splits=64
    elif base_programs<32:
        splits=32
    elif base_programs<=256:
        splits=16
    elif base_programs<=512:
        splits=4
    else:
        splits=1
    return 16,splits,4,1 if base_programs<=512 or fmt=='nvfp4' else 2
