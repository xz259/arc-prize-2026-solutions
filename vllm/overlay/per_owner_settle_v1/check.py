"""CPU gates for per_owner_settle v2 with fakes: (1) settle_core issues a NON-BLOCKING fence for owners whose own
in-flight count is zero and whose final output is held, completes them only once the reply is in (done, oldest
outstanding RPC, or empty batch queue), never waits on an in-flight step, runs the abort free path at once, and
sends exactly the ready stop wires; (2) _can_schedule is True while transactions pend; (3) worker_settle waits on
the owner's recorded event on a side stream, reads only the owner's slot, stages the same repairs and applies them
on the main stream, without any device-wide synchronize; (4) execute_model tags scheduled requests with an event."""
import os, sys, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from per_owner_settle_v1 import runtime as rt


class LookaheadError(RuntimeError): pass
class Req:
    def __init__(self, rid, inflight=0, placeholders=0, stale=0, ids=(1, 2, 3), computed=3, status='FINISHED_STOPPED'):
        self.request_id = rid; self.num_in_flight_tokens = inflight; self.num_output_placeholders = placeholders; self.num_stale_output_tokens = stale
        self._all_token_ids = list(ids); self.num_computed_tokens = computed; self.status = types.SimpleNamespace(name=status); self._arc3_la_pending = True


RPC_LOG = []


def make_module():
    RPC_LOG.clear()
    m = types.SimpleNamespace(LookaheadError=LookaheadError, VERSION='V', pending=lambda sched: sched._tx, worker_settle=lambda w, x: None, __name__='one_lookahead')
    rt.install_one_lookahead(m); return m, RPC_LOG


import collections
from concurrent.futures import Future


class FakeFuture(Future):
    """Stands in for multiproc_executor.FutureWrapper: result() blocks on the worker reply; here the test decides
    whether the reply is available, and a result() call while it is not counts as the forbidden GPU wait."""
    def __init__(self, fq, reply, name):
        super().__init__(); self.fq = fq; self.reply = reply; self.name = name; self.available = False; fq.appendleft(self)
    def result(self, timeout=None):
        if not self.done():
            assert self.available, f'blocking wait on {self.name} while its reply is not available'
            while self.fq and self.fq[-1] is not self: self.fq.pop()   # drain older futures, as FutureWrapper does
            if self.fq: self.fq.pop()
            self.set_result(self.reply)
        return super().result()


def core_with(tx, fq=None, batch_queue=('inflight-step',)):
    sched = types.SimpleNamespace(_tx=tx)
    fq = collections.deque() if fq is None else fq
    core = types.SimpleNamespace(scheduler=sched, batch_queue=list(batch_queue), __dict__={})
    def rpc(fn, timeout, args, non_block=False):
        assert fn.__name__ == 'worker_settle' and non_block is True and timeout == 120
        ids = [w['request_id'] for w in args[0]]; RPC_LOG.append(('rpc_nb', ids))
        return FakeFuture(fq, [dict(version='V', fenced=True, requests=[dict(request_id=i) for i in ids])], 'fence:' + ','.join(ids))
    core.model_executor = types.SimpleNamespace(collective_rpc=rpc, futures_queue=fq)
    core.collective_rpc = lambda *a, **k: (_ for _ in ()).throw(AssertionError('v2 must not use the blocking core RPC'))
    return core, fq


class Outs:
    def __init__(self): self.outputs = []


def test_settle_core():
    m, calls = make_module()
    sys.modules['engine_patch'] = types.SimpleNamespace(assert_settled=lambda r: None, _audit=lambda *a, **k: RPC_LOG.append(('audit', a[2].request_id)))
    a = Req('a', inflight=0); b = Req('b', inflight=1); c = Req('c', inflight=0, status='FINISHED_ABORTED')
    tx = {'a': dict(request=a, held=('client', 'out-a'), dropped=[9], free=None),
          'b': dict(request=b, held=('client', 'out-b'), dropped=[], free=None),
          'c': dict(request=c, held=None, dropped=[], free=((), {}))}
    freed = []
    core, fq = core_with(tx)
    step_k2 = FakeFuture(fq, 'step-k+2', 'step-k+2')            # older in-flight step future (the one the fence queues behind)
    # (A) a is ready: its fence is issued non-blocking; c (abort free) completes at once; a stays pending, output held
    out = m.settle_core(core, None, lambda sched, req, *x, **k: freed.append(req.request_id), Outs)
    assert calls == [('rpc_nb', ['a'])], calls
    assert set(tx) == {'a', 'b'} and a._arc3_la_pending and freed == ['c'] and not c._arc3_la_pending
    assert out == {} or 'client' not in out, out
    fence_a = fq[0]; assert fence_a.name == 'fence:a' and fq[-1] is step_k2
    # (B) fence not done and not the oldest RPC: deferred, no wait
    step_k3 = FakeFuture(fq, 'step-k+3', 'step-k+3')
    assert m.settle_core(core, 'same', None, Outs) == 'same' and calls == [('rpc_nb', ['a'])] and rt.STATS['fence_wait'] >= 1
    # (C) the native loop consumed step k+2 (its future is gone): the fence is the oldest outstanding RPC -> resolved now, transaction completes
    fq.remove(step_k2); fence_a.available = True
    out = m.settle_core(core, None, None, Outs)
    assert calls[-1] == ('audit', 'a') and 'a' not in tx and not a._arc3_la_pending and a._arc3_la_fenced
    assert out['client'].outputs == ['out-a'] and a._arc3_la_last_settlement == dict(delta=0, discarded_token_ids=[9], worker_fenced=True)
    assert rt.STATS['resolved_oldest'] >= 1 and fence_a not in fq and step_k3 in fq
    # (D) a fence already resolved by a newer future's drain (done() True) completes without any result() wait
    b.num_in_flight_tokens = 0
    m.settle_core(core, None, None, Outs); fence_b = fq[0]; assert fence_b.name == 'fence:b' and 'b' in tx
    fence_b.set_result(fence_b.reply); fq.remove(fence_b)
    out = m.settle_core(core, None, None, Outs); assert not tx and out['client'].outputs == ['out-b'] and calls[-1] == ('audit', 'b')
    # (E) empty batch queue: nothing in flight, the fence is resolved synchronously in the same call (keep-pause drains rely on this)
    d = Req('d', inflight=0); tx = {'d': dict(request=d, held=('c', 'o'), dropped=[], free=None)}
    core, fq = core_with(tx, batch_queue=())
    older = FakeFuture(fq, 'x', 'older-done'); older.set_result('x')   # an older, already-resolved RPC ahead of the fence: not the oldest, so only the empty-queue rule applies
    orig_rpc = core.model_executor.collective_rpc
    def rpc_avail(*a, **k):
        f = orig_rpc(*a, **k); f.available = True; return f
    core.model_executor.collective_rpc = rpc_avail
    out = m.settle_core(core, None, None, Outs); assert not tx and out['c'].outputs == ['o'] and rt.STATS['resolved_empty_queue'] >= 1
    # (F) a receipt naming the wrong owner is rejected
    e = Req('e', inflight=0); tx = {'e': dict(request=e, held=('c', 'o'), dropped=[], free=None)}
    core, fq = core_with(tx, batch_queue=())
    def rpc_bad(fn, timeout, args, non_block=False):
        f = FakeFuture(fq, [dict(version='V', fenced=True, requests=[dict(request_id='zzz')])], 'bad'); f.available = True; return f
    core.model_executor.collective_rpc = rpc_bad
    try: m.settle_core(core, None, None, Outs); raise SystemExit('bad receipt accepted')
    except LookaheadError: pass
    # (G) unsettled reservations are rejected before any fence
    g = Req('g', inflight=0, placeholders=1); tx = {'g': dict(request=g, held=('c', 'o'), dropped=[], free=None)}
    core, fq = core_with(tx)
    try: m.settle_core(core, None, None, Outs); raise SystemExit('unsettled reservations accepted')
    except LookaheadError: pass
    assert not fq, 'no fence issued for a rejected transaction'
    assert m._can_schedule(core_with({'x': 1})[0]) is True


class FakeStream:
    def __init__(self, name, log): self.name = name; self.log = log
    def wait_event(self, ev): self.log.append(('wait', self.name, ev.name))
class FakeEvent:
    def __init__(self, name): self.name = name
    def record(self, stream): pass
class Tensor:
    def __init__(self, rows, log, name): self.rows = rows; self.log = log; self.name = name
    def __getitem__(self, key):
        idx = key if isinstance(key, int) else key[0]; self.log.append(('read', self.name, idx)); r = self.rows[idx]
        if isinstance(key, tuple): r = r[key[1]]   # slice -> sub-list, int -> element
        return types.SimpleNamespace(item=lambda: r, cpu=lambda: types.SimpleNamespace(tolist=lambda: list(r)))
    def __setitem__(self, key, v): self.log.append(('set', self.name, key, v))
class Staged:
    def __init__(self, gpu, log, name): self.gpu = gpu; self.log = log; self.name = name
    def stage_write(self, slot, start, x): self.log.append(('stage', self.name, slot, start, list(x)))
    def stage_write_elem(self, slot, x): self.log.append(('stage_elem', self.name, slot, x))
    def apply_write(self): self.log.append(('apply', self.name, CUR[0]))
CUR = ['main']


def test_worker_settle():
    import torch as real_torch
    log = []
    class Fake:
        cuda = types.SimpleNamespace(current_stream=lambda: FakeStream('main', log), Stream=lambda device=None: FakeStream('side', log),
                                     synchronize=lambda: log.append(('SYNC',)), Event=lambda: FakeEvent('new'))
        @staticmethod
        def stream(s):
            class C:
                def __enter__(self_): CUR[0] = s.name; log.append(('enter', s.name))
                def __exit__(self_, *a): CUR[0] = 'main'
            return C()
    Fake.cuda.stream = Fake.stream
    sys.modules['torch'] = Fake
    try:
        m, _ = make_module()
        states = types.SimpleNamespace(req_id_to_index={'a': 5, 'z': 6},
                                       num_computed_tokens=types.SimpleNamespace(gpu=Tensor({5: 3, 6: 99}, log, 'computed')),
                                       total_len=Staged(Tensor({5: 4, 6: 99}, log, 'total'), log, 'total'),
                                       all_token_ids=Staged(Tensor({5: [1, 2, 3, 9], 6: [7]}, log, 'ids'), log, 'ids'),
                                       last_sampled_tokens=Tensor({5: [9], 6: [0]}, log, 'last'), num_computed_tokens_np={})
        runner = types.SimpleNamespace(req_states=states, device='dev', _arc3_last_step_event={'a': FakeEvent('ev-a'), 'z': FakeEvent('ev-z')})
        worker = types.SimpleNamespace(model_runner=runner)
        r = m.worker_settle(worker, [dict(request_id='a', accepted_ids=[1, 2, 3], computed=3, dropped=[9])])
        assert r['fenced'] and r['requests'][0]['discarded'] == 1
        assert ('SYNC',) not in log, 'no device-wide synchronize'
        assert log[:2] == [('enter', 'side'), ('wait', 'side', 'ev-a')], log[:3]
        assert all(x[2] == 5 for x in log if x[0] == 'read'), 'only the owner slot is read'
        assert ('stage', 'ids', 5, 3, [0]) in log and ('stage_elem', 'total', 5, 3) in log and ('set', 'last', (5, 0), 3) in log
        applies = [x for x in log if x[0] == 'apply']; assert applies == [('apply', 'ids', 'main'), ('apply', 'total', 'main')], applies
        assert states.num_computed_tokens_np[5] == 3 and 'a' not in runner._arc3_last_step_event and 'z' in runner._arc3_last_step_event
        try: m.worker_settle(worker, [dict(request_id='z', accepted_ids=[7], computed=1, dropped=[])]); raise SystemExit('bad frontier accepted')
        except LookaheadError: pass
    finally:
        sys.modules['torch'] = real_torch


def test_execute_model_events():
    import torch
    class R:
        max_num_reqs = 4; req_states = types.SimpleNamespace(req_id_to_index={'a': 0})
        def execute_model(self, so, *a, **k): return 'ok'
    mod = types.SimpleNamespace(GPUModelRunner=R, __name__='mr'); rt.install_model_runner(mod)
    r = R(); so = types.SimpleNamespace(num_scheduled_tokens={'a': 1, 'b': 1}, finished_req_ids=set())
    if not torch.cuda.is_available():
        print('execute_model event test skipped (no CUDA)'); return
    assert r.execute_model(so) == 'ok' and set(r._arc3_last_step_event) == {'a', 'b'} and r._arc3_last_step_event['a'] is r._arc3_last_step_event['b']


if __name__ == '__main__':
    test_settle_core(); test_worker_settle(); test_execute_model_events(); print('cpu gates ok', rt.STATS['installed'])
