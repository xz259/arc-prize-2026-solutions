"""Build production's PLE host-poll polling kernel (ple_hostpoll_v1/poll_kernel.py, md5-pinned, unchanged) for sm_89 into
BUILD_DIR (L4 port, 2026-10-31, ARC2 - main; W2's recipe). TORCH_CUDA_ARCH_LIST=8.9 as the L4 launcher sets it for the server,
so the server's load_inline finds this build up to date. Run with the runtime python: runtime_python.sh build_sm89.py BUILD_DIR"""
import hashlib
import importlib.util
import os
import sys
import time

SRC = "/workspace/arc3-inference/runs/dense_fp8_v1/ple_hostpoll_v1/poll_kernel.py"
SRC_MD5 = "6b7ff35b712fc3bb305cba2f2463a6cb"
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9"
build = sys.argv[1]
assert hashlib.md5(open(SRC, "rb").read()).hexdigest() == SRC_MD5, "production poll_kernel.py changed"
spec = importlib.util.spec_from_file_location("poll_kernel_sm89", SRC)
pk = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pk)
t = time.time()
mod = pk.load(build_dir=build, verbose=False)
import torch
flag = torch.zeros(1, dtype=torch.int32).share_memory_()   # as the connector's flags: shared, registered portable + mapped
assert int(torch.cuda.cudart().cudaHostRegister(flag.data_ptr(), 4, 3)) == 0 and flag.is_pinned()
expected = torch.zeros(1, dtype=torch.int32, device="cuda")
timed_out = torch.zeros(1, dtype=torch.int32, device="cuda")
mod.arc3_poll_host_flag(flag, expected, timed_out, 1000)   # flag == expected: returns at once
torch.cuda.synchronize()
expected.fill_(7)
mod.arc3_poll_host_flag(flag, expected, timed_out, 1000)   # never equal: times out after 1,000 spins
torch.cuda.synchronize()
print("ple_hostpoll sm_89 built in %.1f s; device %s; poll ok, timeout path %d" % (time.time() - t, torch.cuda.get_device_name(0),
                                                                                 int(timed_out.item())))
