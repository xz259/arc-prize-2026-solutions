"""CPU gates for the PLE discriminator: patched-method semantics on stand-ins, receipts, dispatch by mode, and (with
--import) a real import of the pinned vLLM modules under the hook. Run: python check.py [--import VLLM_PARENT_DIR]."""
import json, os, sys, tempfile, time, types
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault('ARC3_PLE_DISCRIMINATOR', 'off')
tmp = tempfile.mkdtemp(); os.environ['ARC3_PLE_DISCRIMINATOR_RECEIPTS'] = tmp
import torch
from ple_discriminator_v1 import runtime as rt


class Sem:
    def __init__(self): self.calls = []
    def wait_reset(self, s): self.calls.append(('wait_reset', s))
    def signal(self, s): self.calls.append(('signal', s))
class Stream:
    def __init__(self): self.sync = 0
    def synchronize(self): self.sync += 1
class Target:
    def __init__(self): self.copy_stream = Stream(); self.sem = Sem()
class Req:
    def __init__(self, dp): self.dp_rank = dp; self.num_tokens = 4; self.num_reqs = 2
class Worker:
    def __init__(self):
        self._worker_targets = {0: {'L0': [Target(), Target()], 'L1': [Target()]}}
        self._layers = {'L0': object(), 'L1': object()}; self._input_bufs = {}; self._pinned_bufs = {}
    def _handle_requests(self, requests): raise AssertionError('original handler must not run')


def test_worker():
    mod = types.SimpleNamespace(PleOffloadRunner=Worker); rt.install_worker(mod)
    w = Worker(); w._handle_requests([Req(0), Req(0), Req(7)])   # duplicate and unknown dp are skipped
    for layer in ('L0', 'L1'):
        for t in w._worker_targets[0][layer]:
            assert t.copy_stream.sync == 1 and t.sem.calls == [('wait_reset', t.copy_stream), ('signal', t.copy_stream)], t.sem.calls
    assert rt.REPORT['handled_requests'] == 1 and Worker._arc3_ple_discriminator == 'lookup'
    assert Worker._arc3_original_handle_requests is not Worker._handle_requests
    try: rt.install_worker(mod); raise SystemExit('double install accepted')
    except AssertionError: pass


class Layer:
    _is_cpu_offloaded = True
    def __init__(self): self._gpu_output_buffer = torch.arange(32.).reshape(8, 4)
    def forward_impl(self, hidden_states, input_ids, *a, **k): return torch.full((input_ids.shape[0], 4), -1.)
    def forward(self, *a, **k): raise AssertionError('original forward must not run')


def test_layer():
    mod = types.SimpleNamespace(PleOffloadLayer=Layer); rt.install_layer(mod)
    l = Layer(); out = l.forward(torch.zeros(3, 4), torch.tensor([5, 6, 7]), 'qsl', 'ctx')
    assert out.shape == (3, 4) and torch.equal(out, l._gpu_output_buffer[:3]) and out.data_ptr() == l._gpu_output_buffer.data_ptr()
    l2 = Layer(); l2._is_cpu_offloaded = False
    assert torch.equal(l2.forward(torch.zeros(2, 4), torch.tensor([1, 2])), torch.full((2, 4), -1.))
    assert Layer._arc3_ple_discriminator == rt.MODE   # marker records the active mode


class Connector:
    def __init__(self): self.tp_rank = 0; self._request_queue = types.SimpleNamespace(put_nowait=lambda r: (_ for _ in ()).throw(AssertionError('no request')))
    def _launch(self, num_reqs, num_tokens): raise AssertionError('original launch must not run')


def test_connector():
    mod = types.SimpleNamespace(PleOffloadConnector=Connector); rt.install_connector(mod)
    c = Connector(); assert c._launch(2, 4) is None and rt.REPORT['skipped_launches'] == 1
    assert Connector._arc3_ple_discriminator == 'handoff'


def test_worker_nogpu():
    calls = []
    class Layer:
        def forward_impl(self, ids, ids2, qsl, ctx, output_buffer=None): calls.append(('lookup', len(ids), output_buffer)); return output_buffer
    class Buf(list):
        def __getitem__(self, k): return Buf(list.__getitem__(self, k)) if isinstance(k, slice) else list.__getitem__(self, k)
        def clamp_min_(self, v): calls.append(('clamp', v)); return self
    class W:
        def __init__(self):
            self._worker_targets = {0: {'L0': [Target()]}}; self._layers = {'L0': Layer()}; self._clamp_input_ids = True
            self._input_bufs = {0: types.SimpleNamespace(input_ids_buf=Buf([1, 2, 3, 4]), query_start_loc_buf=[0, 2, 4], ngram_context_buf=None)}
            self._pinned_bufs = {0: {'L0': 'pinned'}}
        def _handle_requests(self, r): raise AssertionError('original must not run')
    rt.install_worker_nogpu(types.SimpleNamespace(PleOffloadRunner=W)); w = W(); w._handle_requests([Req(0)])
    assert calls == [('clamp', 0), ('lookup', 4, 'pinned')], calls
    t = w._worker_targets[0]['L0'][0]; assert t.sem.calls == [] and t.copy_stream.sync == 0, 'nogpu must not touch streams or semaphores'


def test_dispatch_and_receipt():
    saved = rt.MODE
    for mode, name, expect in (('lookup', 'vllm.v1.ple_offload.worker', 'PleOffloadRunner'),
                               ('handoff', 'vllm.model_executor.layers.ple_offload_layer', 'PleOffloadLayer'),
                               ('handoff', 'vllm.v1.ple_offload.connector', 'PleOffloadConnector'),
                               ('lookup', 'vllm.v1.ple_offload.connector', None), ('off', 'vllm.v1.ple_offload.worker', None),
                               ('nowait', 'vllm.model_executor.layers.ple_offload_layer', 'PleOffloadLayer'), ('nowait', 'vllm.v1.ple_offload.connector', None),
                               ('nogpu', 'vllm.v1.ple_offload.worker', 'PleOffloadRunner'), ('nogpu', 'vllm.model_executor.layers.ple_offload_layer', 'PleOffloadLayer')):
        rt.MODE = mode
        class Fresh:
            def _handle_requests(self, r): pass
            def forward(self, *a): pass
            def _launch(self, *a): pass
        mod = types.SimpleNamespace(**{expect or 'X': Fresh}); rt.install_for(name, mod)
        assert bool(getattr(Fresh, '_arc3_ple_discriminator', False)) == (expect is not None), (mode, name)
    rt.MODE = saved
    rt.write_receipt(); files = [f for f in os.listdir(tmp) if f.endswith('.json')]
    assert len(files) == 1 and json.load(open(os.path.join(tmp, files[0])))['installed'], files


def test_timing():
    class Conn:
        def _launch(self, a, b): time.sleep(0.0005); return 'ok'
        def _process_request(self, r, s): return None
        def _copy_cuda_inputs(self, r): return None
    class Runner:
        def _handle_requests(self, r): return 7
    rt.install_timing_connector(types.SimpleNamespace(PleOffloadConnector=Conn)); rt.install_timing_worker(types.SimpleNamespace(PleOffloadRunner=Runner))
    c = Conn(); r = Runner()
    for _ in range(3): assert c._launch(1, 2) == 'ok'
    for _ in range(2001): r._handle_requests([])
    s = {k: v.summary() for k, v in rt.TIMERS.items()}
    assert s['model_thread_launch_incl_ack']['count'] == 3 and s['model_thread_launch_incl_ack']['mean_us'] >= 400, s
    assert s['offload_process_handle_lookup_h2d']['count'] == 2001 and rt.REPORT.get('timing'), 'timing receipt not written at 2000'
    assert Conn._arc3_ple_discriminator == 'timing' and Runner._arc3_ple_discriminator == 'timing'


def test_real_import(parent):
    """Import the pinned modules through the hook in this interpreter (mode from the environment). With
    ARC3_PLE_HANDOFF_RECEIPTS set, the served ple_copy_handoff_v1 hook is installed first (as the combined bootstrap
    does), so the two finders' interaction and the patched connector source are exercised."""
    sys.path.insert(0, parent)
    import importlib, inspect
    if os.environ.get('ARC3_PLE_HANDOFF_RECEIPTS'):
        from ple_copy_handoff_v1.import_hook import install as install_handoff
        install_handoff()
    from ple_discriminator_v1.import_hook import install, TARGETS
    install()
    for name in TARGETS[rt.MODE]:
        m = importlib.import_module(name)
        cls = {'vllm.v1.ple_offload.worker': 'PleOffloadRunner', 'vllm.v1.ple_offload.connector': 'PleOffloadConnector',
               'vllm.model_executor.layers.ple_offload_layer': 'PleOffloadLayer'}[name]
        assert getattr(m, cls)._arc3_ple_discriminator == rt.MODE, name
        if name.endswith('.connector') and os.environ.get('ARC3_PLE_HANDOFF_RECEIPTS'):
            assert '_input_copy_submitted' in inspect.getsource(m.PleOffloadConnector.__init__), 'served connector source not loaded'
            assert any(f.startswith('worker-') for f in os.listdir(os.environ['ARC3_PLE_HANDOFF_RECEIPTS'])), 'handoff receipt missing'
    print('real import ok:', rt.MODE, sorted(rt.REPORT['installed']), 'with ple_copy_handoff' if os.environ.get('ARC3_PLE_HANDOFF_RECEIPTS') else 'alone')


if __name__ == '__main__':
    if len(sys.argv) > 2 and sys.argv[1] == '--import':
        test_real_import(sys.argv[2])
    else:
        test_worker(); test_layer(); test_connector(); test_dispatch_and_receipt(); test_timing(); test_worker_nogpu(); print('cpu gates ok')
