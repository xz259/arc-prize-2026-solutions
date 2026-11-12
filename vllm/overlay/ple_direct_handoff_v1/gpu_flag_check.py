"""GPU micro-check of the direct-handoff transport: a shared pinned int32[4] flag, three staging D2H copies and the
16-byte flag DMA on one stream; a child process polls the flag and must observe seq only after the staged data.
Exercises the exact primitives the connector uses (share_memory_, cuMemHostRegister, non_blocking copies)."""
import json, multiprocessing as mp, os, sys, time
import torch
from cuda.bindings import driver as cuda_driver


def child(flag, inp, ack, results, n, ready):
    view = flag.numpy(); iv = inp.numpy(); av = ack.numpy(); last = 0; seen = []; torn = 0; ready.put('ready'); t0 = time.time()
    while len(seen) < n and time.time() - t0 < 30:
        seq = int(view[3])
        if seq == last:
            time.sleep(0.00005); continue
        nt = int(view[0]); seq2 = int(view[3])
        if seq2 != seq: torn += 1; continue
        ok = bool((iv[:nt] == seq).all())      # staged inputs of step seq must already be visible
        seen.append((seq, nt, ok)); last = seq; av[0] = seq   # consumed: the producer may stage the next step
    results.put(dict(seen=seen, torn=torn))


def main():
    mp.set_start_method('spawn')
    dev = torch.device('cuda', 0)
    torch.cuda.init(); torch.zeros(1, device=dev)   # the driver context must exist before cuMemHostRegister (as in serving)
    flag = torch.zeros(4, dtype=torch.int32).share_memory_()
    inp = torch.zeros(8192, dtype=torch.int32).share_memory_(); ack = torch.zeros(1, dtype=torch.int32).share_memory_(); akv = ack.numpy()
    for buf in (flag, inp):
        r = cuda_driver.cuMemHostRegister(buf.data_ptr(), buf.numel() * 4, cuda_driver.CU_MEMHOSTREGISTER_PORTABLE)
        assert (r[0] if isinstance(r, tuple) else r).value == 0, r
        assert buf.is_pinned()
    flag_dev = torch.zeros(4, dtype=torch.int32, device=dev); slots = torch.zeros(2, 4, dtype=torch.int32, pin_memory=True)
    src = torch.zeros(8192, dtype=torch.int32, device=dev); stream = torch.cuda.Stream(device=dev)
    results = mp.Queue(); ready = mp.Queue(); n = 2000
    p = mp.Process(target=child, args=(flag, inp, ack, results, n, ready)); p.start(); assert ready.get(timeout=120) == 'ready'
    waits = []
    lat = []
    for seq in range(1, n + 1):
        nt = 8 + (seq % 500) * 16
        src[:nt].fill_(seq)                       # "input preparation" on the default stream
        ev = torch.cuda.Event(); ev.record(torch.cuda.current_stream(dev))
        t0 = time.perf_counter()
        with torch.cuda.stream(stream):
            stream.wait_event(ev)
            inp[:nt].copy_(src[:nt], non_blocking=True)
            slot = slots[seq & 1]; slot[0] = nt; slot[1] = 1; slot[2] = 0; slot[3] = seq
            flag_dev.copy_(slot, non_blocking=True)
            flag.copy_(flag_dev, non_blocking=True)
        lat.append(time.perf_counter() - t0)
        t1 = time.perf_counter()
        while int(akv[0]) != seq:                 # back-pressure as in serving (consumer must finish step t first)
            if time.perf_counter() - t1 > 5: raise SystemExit(f'consumer never acknowledged seq {seq}')
        waits.append(time.perf_counter() - t1)
    p.join(40); res = results.get(timeout=5)
    seen = res['seen']; bad = [s for s in seen if not s[2]]
    out = dict(steps=n, observed=len(seen), in_order=[s[0] for s in seen] == sorted(s[0] for s in seen), missed=n - len(seen),
               data_visible_on_every_seq=not bad, torn=res['torn'], enqueue_us_mean=sum(lat) / len(lat) * 1e6, enqueue_us_max=max(lat) * 1e6,
               publish_to_consume_us_mean=sum(waits) / len(waits) * 1e6, publish_to_consume_us_p90=sorted(waits)[int(0.9 * len(waits))] * 1e6)
    print(json.dumps(out, indent=1))
    if len(sys.argv) > 1: json.dump(out, open(sys.argv[1], 'w'), indent=1)
    assert not bad and res['torn'] == 0 and len(seen) == n, out


if __name__ == '__main__':
    main()
