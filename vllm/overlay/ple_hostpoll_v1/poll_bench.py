"""GPU micro-benchmark of the polling kernel: (1) correctness — the kernel returns only after the CPU publishes the
expected value; (2) cost when the flag is already set, eager and inside a captured CUDA graph, vs cuStreamWaitValue32
on an already-set device flag; (3) wake latency after a CPU store."""
import json, sys, threading, time
import torch
from cuda.bindings import driver as cuda_driver
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from poll_kernel import load


def chk(r, what):
    e = r[0] if isinstance(r, tuple) else r
    assert e.value == 0, (what, e)


def main():
    dev = torch.device('cuda', 0); torch.cuda.init(); torch.zeros(1, device=dev)
    mod = load(verbose=False)
    flag = torch.zeros(1, dtype=torch.int32).share_memory_()
    chk(cuda_driver.cuMemHostRegister(flag.data_ptr(), 4, cuda_driver.CU_MEMHOSTREGISTER_PORTABLE | cuda_driver.CU_MEMHOSTREGISTER_DEVICEMAP), 'register')
    assert flag.is_pinned()
    fv = flag.numpy(); expected = torch.zeros(1, dtype=torch.int32, device=dev); timed_out = torch.zeros(1, dtype=torch.int32, device=dev)
    out = {}
    # (1) correctness: kernel must block until the CPU publishes
    expected.fill_(7); torch.cuda.synchronize()
    done = torch.cuda.Event(); mod.arc3_poll_host_flag(flag, expected, timed_out, 10_000_000); done.record()
    time.sleep(0.05); assert not done.query(), 'kernel finished before the flag was published'
    t0 = time.perf_counter(); fv[0] = 7; done.synchronize(); out['wake_after_store_us'] = (time.perf_counter() - t0) * 1e6
    assert int(timed_out.item()) == 0
    # (2) eager cost when already satisfied
    torch.cuda.synchronize(); t0 = time.perf_counter()
    for _ in range(1000): mod.arc3_poll_host_flag(flag, expected, timed_out, 10_000_000)
    torch.cuda.synchronize(); out['eager_satisfied_us_per_call'] = (time.perf_counter() - t0) * 1e3
    # graph with the kernel between two matmuls vs memop wait vs nothing
    a = torch.randn(2048, 2048, device=dev, dtype=torch.bfloat16); b = torch.randn(2048, 2048, device=dev, dtype=torch.bfloat16)
    dflag = torch.ones(1, dtype=torch.int32, device=dev)
    def memop_wait():
        chk(cuda_driver.cuStreamWaitValue32(cuda_driver.CUstream(torch.cuda.current_stream().cuda_stream), cuda_driver.CUdeviceptr(dflag.data_ptr()), 1,
                                             cuda_driver.CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ.value), 'wait')
    variants = {'none': lambda: None, 'poll_kernel': lambda: mod.arc3_poll_host_flag(flag, expected, timed_out, 10_000_000), 'memop_wait': memop_wait}
    for name, op in variants.items():
        s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3): c = a @ b; op(); d = c @ b
        torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=s):
            c = a @ b; op(); d = c @ b
        torch.cuda.synchronize(); g.replay(); torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(300): g.replay()
        torch.cuda.synchronize(); out[f'graph_{name}_us_per_replay'] = (time.perf_counter() - t0) / 300 * 1e6
    print(json.dumps(out, indent=1))
    if len(sys.argv) > 1: json.dump(out, open(sys.argv[1], 'w'), indent=1)


if __name__ == '__main__':
    main()
