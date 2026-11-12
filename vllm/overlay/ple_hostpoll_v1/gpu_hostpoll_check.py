"""GPU end-to-end check of the host-poll transport with the real polling kernel: parent = GPU side (shared pinned
request flag / result buffer / result flag, registered like the connector; a captured CUDA graph containing the
poll + H2D op between two matmuls), child = offload side (polls the request flag, writes rows = f(seq) into the
shared result buffer, publishes the result flag). Every replay must see exactly the rows of its own seq."""
import json, multiprocessing as mp, sys, time
import torch
from cuda.bindings import driver as cuda_driver
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from poll_kernel import load


def chk(r, what):
    e = r[0] if isinstance(r, tuple) else r
    assert e.value == 0, (what, e)


def child(req, result, rflag, ready, n):
    rv = req.numpy(); res = result; fv = rflag.numpy(); last = 0; ready.put('ready'); t0 = time.time(); served = 0
    while served < n and time.time() - t0 < 60:
        seq = int(rv[3])
        if seq == last:
            time.sleep(0.00005); continue
        nt = int(rv[0]); last = seq
        res[:nt].fill_(seq % 251)      # rows of this seq
        res[nt:nt + 1].fill_(-1)       # a guard row the GPU must not copy
        fv[0] = seq; served += 1


def main():
    mp.set_start_method('spawn'); dev = torch.device('cuda', 0); torch.cuda.init(); torch.zeros(1, device=dev)
    mod = load()
    n = 400; rows = 4096; width = 256
    req = torch.zeros(4, dtype=torch.int32).share_memory_()
    result = torch.zeros(rows, width, dtype=torch.int32).share_memory_()
    rflag = torch.zeros(1, dtype=torch.int32).share_memory_()
    for buf, flags in ((req, cuda_driver.CU_MEMHOSTREGISTER_PORTABLE), (result, cuda_driver.CU_MEMHOSTREGISTER_PORTABLE),
                       (rflag, cuda_driver.CU_MEMHOSTREGISTER_PORTABLE | cuda_driver.CU_MEMHOSTREGISTER_DEVICEMAP)):
        chk(cuda_driver.cuMemHostRegister(buf.data_ptr(), buf.numel() * buf.element_size(), flags), 'register'); assert buf.is_pinned()
    expected = torch.zeros(1, dtype=torch.int32, device=dev); timed_out = torch.zeros(1, dtype=torch.int32, device=dev)
    gpu_out = torch.zeros(rows, width, dtype=torch.int32, device=dev)
    flag_dev = torch.zeros(4, dtype=torch.int32, device=dev); slots = torch.zeros(2, 4, dtype=torch.int32, pin_memory=True)
    a = torch.randn(1024, 1024, device=dev, dtype=torch.bfloat16)
    ready = mp.Queue(); p = mp.Process(target=child, args=(req, result, rflag, ready, n)); p.start(); assert ready.get(timeout=120) == 'ready'
    nt = 37
    def step_ops():
        c = a @ a
        mod.arc3_poll_host_flag(rflag, expected, timed_out, 75_000_000)
        gpu_out[:nt].copy_(result[:nt], non_blocking=True)
        d = c @ a
        return d
    # capture (dummy pass-through: expected := current flag)
    s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
    expected.copy_(rflag, non_blocking=True); torch.cuda.synchronize()
    with torch.cuda.stream(s):
        for _ in range(2): step_ops()
    torch.cuda.current_stream().wait_stream(s); torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g, stream=s): step_ops()
    torch.cuda.synchronize()
    d2h = torch.cuda.Stream(); bad = []; lat = []
    for seq in range(1, n + 1):
        t0 = time.perf_counter()
        ev = torch.cuda.Event(); ev.record(torch.cuda.current_stream())
        with torch.cuda.stream(d2h):                  # staging + request flag (as the connector does)
            d2h.wait_event(ev); slot = slots[seq & 1]; slot[0] = nt; slot[1] = 1; slot[2] = 0; slot[3] = seq
            flag_dev.copy_(slot, non_blocking=True); req.copy_(flag_dev, non_blocking=True)
        expected.fill_(seq)                            # model stream, before the graph
        g.replay(); torch.cuda.synchronize(); lat.append(time.perf_counter() - t0)
        got = gpu_out[:nt + 1].cpu()
        if not bool((got[:nt] == seq % 251).all()) or int(got[nt, 0]) != 0:
            bad.append((seq, int(got[0, 0]), int(got[nt, 0])))
        if int(timed_out.item()) != 0: bad.append((seq, 'timed_out')); break
    p.join(30)
    out = dict(steps=n, bad=bad[:5], n_bad=len(bad), step_us_mean=sum(lat) / len(lat) * 1e6, step_us_max=max(lat) * 1e6, timed_out=int(timed_out.item()))
    print(json.dumps(out, indent=1))
    if len(sys.argv) > 1: json.dump(out, open(sys.argv[1], 'w'), indent=1)
    assert not bad, out


if __name__ == '__main__':
    main()
