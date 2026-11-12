"""CPU gates for ple_hostpoll_v1: connector launch path (staging order + expected-seq publish on the model stream,
dummy-run pass-through, timeout surfacing), offload handler (gather into the shared result, flag published last),
registration (second message via ForkingPickler), placeholder patch (op called with the right tensors, no wait)."""
import ast
import sys
import threading
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
SOURCE = HERE / 'connector.py'


def connector_methods(torch, names):
    tree = ast.parse(SOURCE.read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'PleOffloadConnector')
    functions = [x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name in names]
    code = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                            ast.ClassDef(name='Connector', bases=[], keywords=[], body=functions, decorator_list=[])], type_ignores=[])
    ns = dict(torch=torch, PleDirectRequest=lambda **kw: SimpleNamespace(**kw))
    exec(compile(ast.fix_missing_locations(code), str(SOURCE), 'exec'), ns)
    return ns['Connector']


class Fake:
    def __init__(self, name, log): self.name = name; self.log = log
    def __getitem__(self, key): return _Slot(self, key) if isinstance(key, int) else Fake(f'{self.name}[{key}]', self.log)
    def copy_(self, source, non_blocking=False): self.log.append(('copy', self.name, getattr(source, 'name', source), non_blocking))
    def fill_(self, v): self.log.append(('fill', self.name, v))
    def zero_(self): self.log.append(('zero', self.name))
    def max(self): return 0


class _Slot:
    def __init__(self, parent, key): self.parent = parent; self.name = f'{parent.name}[{key}]'
    def __setitem__(self, i, v): self.parent.log.append(('store', self.name, i, v))


class ConnectorTest(unittest.TestCase):
    def make(self, cuda=True, tp_rank=0):
        log = []
        class Event:
            def __init__(self, name): self.name = name
            def record(self, stream): log.append(('record', self.name, stream))
        class Stream:
            def wait_event(self, event): log.append(('wait-ready', event.name))
        torch = SimpleNamespace(accelerator=SimpleNamespace(device_index=lambda x: nullcontext()),
                                cuda=SimpleNamespace(stream=lambda x: nullcontext(), current_stream=lambda x: 'main',
                                                     nvtx=SimpleNamespace(range=lambda x: nullcontext())))
        c = connector_methods(torch, ('_launch', '_copy_cuda_inputs', '_copy_cpu_inputs', 'prepare_forward', 'signal_dummy_outputs', 'release_outputs'))()
        c.device = SimpleNamespace(index=0); c.tp_rank = tp_rank; c.dp_rank = 0; c._uses_cuda_inputs = cuda
        c._d2h_stream = Stream(); c._input_ready_event = Event('input_ready'); c._d2h_done_event = Event('d2h_done')
        c._request_seq = 0; c._flag_dev = Fake('flag_dev', log); c._flag_src = Fake('flag_src', log); c._request_flag_buf = Fake('flag_shared', log)
        c._expected_dev = {'L': Fake('expected_L', log)}; c._result_flags = {'L': Fake('result_flag_L', log)}
        c._timed_out_dev = {'L': Fake('timed_out_L', log)}; c._timed_out_host = Fake('timed_out_host', log)
        c._layers = {'L': SimpleNamespace(_gpu_output_buffer=Fake('gpu_out_L', log))}
        for name in ('input_ids', 'query_start_loc', 'ngram_context'):
            setattr(c, '_' + name + '_buf', Fake(name, log)); setattr(c, '_' + name + '_source', Fake('source_' + name, log))
        return c, log

    def test_launch_order_and_expected_publish(self):
        c, log = self.make(); c.prepare_forward(4, 9, dummy_run=False)
        kinds = [x[0] for x in log]
        self.assertEqual(kinds[:2], ['record', 'wait-ready'])
        copies = [x for x in log if x[0] == 'copy']
        self.assertEqual([x[1] for x in copies], ['input_ids[slice(None, 9, None)]', 'query_start_loc[slice(None, 5, None)]',
                                                  'ngram_context[slice(None, 4, None)]', 'flag_dev', 'flag_shared'])
        self.assertEqual(log[-1], ('fill', 'expected_L', 1), 'expected seq published last, after the staging enqueue')
        self.assertLess([i for i, x in enumerate(log) if x[0] == 'record' and x[1] == 'd2h_done'][0], len(log) - 1)
        c.prepare_forward(2, 3, dummy_run=False); self.assertEqual(log[-1], ('fill', 'expected_L', 2))

    def test_dummy_run_passes_the_poll_without_a_cpu_result(self):
        c, log = self.make(); c.prepare_forward(4, 16, dummy_run=True)
        self.assertEqual(log, [('zero', 'gpu_out_L[slice(None, 16, None)]'), ('copy', 'expected_L', 'result_flag_L', True)])
        self.assertEqual(c._request_seq, 0, 'dummy runs do not consume sequence numbers')

    def test_release_checks_previous_timeout_and_queues_mirror(self):
        c, log = self.make(); c.release_outputs()
        self.assertEqual(log, [('copy', 'timed_out_host[slice(0, 1, None)]', 'timed_out_L', True)])
        class Bad(Fake):
            def max(self): return 1
        c._timed_out_host = Bad('timed_out_host', log)
        with self.assertRaisesRegex(RuntimeError, 'timed out'): c.release_outputs()

    def test_non_publisher(self):
        c, log = self.make(tp_rank=1); c._launch(3, 5); self.assertEqual(log, [])


class WorkerTest(unittest.TestCase):
    def setUp(self):
        import torch
        from ple_hostpoll_v1 import worker_patch
        self.torch = torch; self.wp = worker_patch
        class Layer:
            def __init__(self, calls): self.calls = calls
            def forward_impl(self, ids, ids2, qsl, ctx, output_buffer=None):
                output_buffer[: len(ids)] = 7; self.calls.append(('lookup', len(ids), int(output_buffer[0, 0]))); return output_buffer
        class Runner:
            def __init__(self, vllm_config):
                self.vllm_config = vllm_config; self.calls = []; self._layers = {'L': Layer(self.calls)}; self._clamp_input_ids = False
                self.layer_names = ['L']
                self._input_bufs = {0: SimpleNamespace(input_ids_buf=torch.arange(8, dtype=torch.int32), query_start_loc_buf=torch.tensor([0, 4, 8]), ngram_context_buf=None)}
            def accept_registrations(self, pull_socket, num_workers): self.accepted = num_workers
            def busy_loop(self, a, b): raise AssertionError('original loop must not run')
        self.module = SimpleNamespace(PleOffloadRunner=Runner, torch=torch, logger=SimpleNamespace(info=lambda *a, **k: None))
        self.Runner = worker_patch.install(self.module)

    def _runner(self):
        r = self.Runner(SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=1)))
        result = self.torch.zeros(8, 4, dtype=self.torch.int32).share_memory_(); flag = self.torch.zeros(1, dtype=self.torch.int32).share_memory_()
        r._hp_result_bufs = {0: {'L': result}}; r._hp_result_flags = {0: {'L': flag.numpy()}}
        return r, result, flag

    def test_handler_gathers_then_publishes_flag(self):
        r, result, flag = self._runner()
        r._handle_hostpoll([(0, 6, 2, 41)])
        self.assertEqual(r.calls, [('lookup', 6, 7)]); self.assertEqual(int(flag[0]), 41); self.assertTrue(bool((result[:6] == 7).all()))

    def test_poll_loop_serves_requests(self):
        import time
        r, result, flag = self._runner()
        req = self.torch.zeros(4, dtype=self.torch.int32).share_memory_(); r._hp_request_flags = {0: req}; stop = threading.Event()
        t = threading.Thread(target=r.busy_loop, args=(None, stop)); t.start(); time.sleep(0.02)
        req[0] = 3; req[1] = 1; req[3] = 1; time.sleep(0.02); self.assertEqual(int(flag[0]), 1)
        req[0] = 5; req[1] = 2; req[3] = 2; time.sleep(0.02); self.assertEqual(int(flag[0]), 2)
        stop.set(); t.join(2); self.assertEqual([c[1] for c in r.calls], [3, 5])

    def test_accept_registrations_second_message(self):
        from multiprocessing.reduction import ForkingPickler
        import torch.multiprocessing as torch_mp
        r = self.Runner(SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=1)))
        flag = self.torch.zeros(4, dtype=self.torch.int32).share_memory_(); rb = self.torch.zeros(8, 4, dtype=self.torch.int32).share_memory_(); rf = self.torch.zeros(1, dtype=self.torch.int32).share_memory_()
        strategy = torch_mp.get_sharing_strategy(); torch_mp.set_sharing_strategy('file_system')
        try: msgs = [ForkingPickler.dumps(PleHostpollRegistration(0, flag, {'L': rb}, {'L': rf}))]
        finally: torch_mp.set_sharing_strategy(strategy)
        r.accept_registrations(SimpleNamespace(recv=lambda: bytes(msgs.pop(0))), 1)
        rf[0] = 9; self.assertEqual(int(r._hp_result_flags[0]['L'][0]), 9, 'flag view must alias the shared storage')
        rb[2, 1] = 5; self.assertEqual(int(r._hp_result_bufs[0]['L'][2, 1]), 5)


class PleHostpollRegistration:   # module-level for pickling; the patch checks the class name
    def __init__(self, dp_rank, buf, result_bufs, result_flags):
        self.dp_rank = dp_rank; self.request_flag_buf = buf; self.result_bufs = result_bufs; self.result_flags = result_flags


class LayerPatchTest(unittest.TestCase):
    def test_forward_calls_op_and_returns_gpu_view(self):
        import torch
        from ple_hostpoll_v1 import layer_patch
        calls = []
        def fake_register(op_name, op_func, mutates_args, fake_impl):
            calls.append(('register', op_name, tuple(mutates_args)))
            fake_register.func = op_func
        class L:
            _is_cpu_offloaded = False
            def forward_impl(self, hs, ids, *a, **k): return 'impl'
        module = SimpleNamespace(PleOffloadLayer=L, direct_register_custom_op=fake_register)
        layer_patch._KERNEL = SimpleNamespace(arc3_poll_host_flag=lambda *a: calls.append(('poll', a[3])))
        # torch.ops.vllm.<name> would not exist without a real registration: route the op to the impl function
        real_ops = torch.ops
        try:
            torch.ops = SimpleNamespace(vllm=SimpleNamespace(arc3_ple_hostpoll_wait_copy=lambda *a: fake_register.func(*a)))
            layer_patch.install(module)
        finally:
            torch.ops = real_ops
        self.assertEqual(calls[0], ('register', 'arc3_ple_hostpoll_wait_copy', ('gpu_output_buffer', 'timed_out_dev')))
        l = L(); gpu = torch.zeros(8, 4); host = torch.ones(8, 4)
        l.setup_hostpoll_offload(gpu, host, 'flag', 'expected', 'timed_out')
        out = l.forward(torch.zeros(3, 4), torch.tensor([1, 2, 3]))
        self.assertEqual(calls[1], ('poll', layer_patch.MAX_SPINS)); self.assertTrue(torch.equal(out, torch.ones(3, 4))); self.assertEqual(out.data_ptr(), gpu.data_ptr())
        self.assertTrue(torch.equal(gpu[3:], torch.zeros(5, 4)), 'only num_tokens rows copied')
        self.assertIsNone(l.release_offloaded_output()); self.assertEqual(L().forward(None, None), 'impl')


if __name__ == '__main__':
    unittest.main(verbosity=1)
