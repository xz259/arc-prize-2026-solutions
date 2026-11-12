"""CPU gates for ple_direct_handoff_v1: (1) the connector's launch path executes the actual patched methods with fakes
and must enqueue record -> wait-ready -> three staging copies -> flag H2D -> flag D2H -> done-record, on the model
thread, with monotonic seq and alternating slots; (2) the offload runner's poll loop decodes flags in order, counts
skipped sequence numbers, ignores stale flags and stops on shutdown; (3) registration accepts the second message."""
import ast
import pickle
import sys
import threading
import time
import unittest
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
SOURCE = HERE / 'connector.py'


def connector_methods(torch):
    tree = ast.parse(SOURCE.read_text())
    cls = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'PleOffloadConnector')
    functions = [x for x in cls.body if isinstance(x, ast.FunctionDef) and x.name in ('_launch', '_copy_cuda_inputs', '_copy_cpu_inputs')]
    req = next(x for x in tree.body if isinstance(x, ast.ClassDef) and x.name == 'PleDirectRequest')
    code = ast.Module(body=[ast.ImportFrom(module='__future__', names=[ast.alias(name='annotations')], level=0),
                            ast.ClassDef(name='Connector', bases=[], keywords=[], body=functions, decorator_list=[])], type_ignores=[])
    ns = dict(torch=torch, PleDirectRequest=lambda **kw: SimpleNamespace(**kw))
    exec(compile(ast.fix_missing_locations(code), str(SOURCE), 'exec'), ns)
    return ns['Connector']


class Fake:
    """Fake tensor recording copies; getitem returns a named view."""
    def __init__(self, name, log):
        self.name = name; self.log = log; self.store = {}
    def __getitem__(self, key):
        return Fake(f'{self.name}[{key}]', self.log) if not isinstance(key, int) else _Slot(self, key)
    def copy_(self, source, non_blocking=False):
        self.log.append(('copy', self.name, getattr(source, 'name', source), non_blocking, threading.current_thread().name))
    def __setitem__(self, key, value):
        self.store[key] = value; self.log.append(('store', self.name, key, value))


class _Slot:
    def __init__(self, parent, key): self.parent = parent; self.key = key; self.name = f'{parent.name}[{key}]'
    def __setitem__(self, i, v): self.parent.log.append(('store', self.name, i, v))


class ConnectorLaunchTest(unittest.TestCase):
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
        c = connector_methods(torch)(); c.device = SimpleNamespace(index=0); c.tp_rank = tp_rank; c.dp_rank = 0
        c._uses_cuda_inputs = cuda; c._d2h_stream = Stream(); c._input_ready_event = Event('input_ready'); c._d2h_done_event = Event('d2h_done')
        c._request_seq = 0; c._flag_dev = Fake('flag_dev', log); c._flag_src = Fake('flag_src', log); c._request_flag_buf = Fake('flag_shared', log)
        for name in ('input_ids', 'query_start_loc', 'ngram_context'):
            setattr(c, '_' + name + '_buf', Fake(name, log)); setattr(c, '_' + name + '_source', Fake('source_' + name, log))
        return c, log

    def test_cuda_launch_order_seq_and_slots(self):
        c, log = self.make()
        c._launch(4, 9)
        kinds = [(x[0], x[1]) for x in log]
        self.assertEqual(kinds[0], ('record', 'input_ready')); self.assertEqual(log[0][2], 'main')
        self.assertEqual(kinds[1], ('wait-ready', 'input_ready'))
        copies = [x for x in log if x[0] == 'copy']
        self.assertEqual([x[1] for x in copies], ['input_ids[slice(None, 9, None)]', 'query_start_loc[slice(None, 5, None)]',
                                                  'ngram_context[slice(None, 4, None)]', 'flag_dev', 'flag_shared'])
        self.assertTrue(all(x[3] is True for x in copies), 'all copies must be non_blocking')
        self.assertEqual(copies[3][2], 'flag_src[1]'); self.assertEqual(copies[4][2], 'flag_dev')
        stores = [x for x in log if x[0] == 'store']
        self.assertEqual([(x[2], x[3]) for x in stores], [(0, 9), (1, 4), (2, 0), (3, 1)])
        self.assertEqual(log[-1][:2], ('record', 'd2h_done'))
        self.assertTrue(all(x[-1] == threading.current_thread().name for x in copies), 'copies enqueued on the calling (model) thread')
        c._launch(2, 3); c._launch(1, 1)
        seqs = [x[3] for x in log if x[0] == 'store' and x[2] == 3]; self.assertEqual(seqs, [1, 2, 3])
        slots = [x[2] for x in log if x[0] == 'copy' and x[1] == 'flag_dev']; self.assertEqual(slots, ['flag_src[1]', 'flag_src[0]', 'flag_src[1]'])

    def test_non_publisher_and_cpu_path(self):
        c, log = self.make(tp_rank=1); c._launch(3, 5); self.assertEqual(log, []); self.assertEqual(c._request_seq, 0)
        c, log = self.make(cuda=False); c._launch(3, 5)
        self.assertEqual([x[1] for x in log if x[0] == 'copy'][:3], ['input_ids[slice(None, 5, None)]', 'query_start_loc[slice(None, 4, None)]', 'ngram_context[slice(None, 3, None)]'])
        stores = [(x[1], x[2], x[3]) for x in log if x[0] == 'store']
        self.assertEqual(stores, [('flag_shared', 0, 5), ('flag_shared', 1, 3), ('flag_shared', 3, 1)], 'seq stored last on the CPU path')

    def test_null_optional_ngram(self):
        c, log = self.make(); c._ngram_context_buf = None; c._ngram_context_source = None; c._launch(4, 4)
        self.assertEqual([x[1] for x in log if x[0] == 'copy'], ['input_ids[slice(None, 4, None)]', 'query_start_loc[slice(None, 5, None)]', 'flag_dev', 'flag_shared'])


class PleDirectFlagRegistration:   # module-level so pickle can serialize it; the name is what the patch checks
    def __init__(self, dp_rank, buf): self.dp_rank = dp_rank; self.request_flag_buf = buf


class RunnerPollTest(unittest.TestCase):
    def setUp(self):
        import numpy as np, torch
        from ple_direct_handoff_v1 import worker_patch
        self.np = np; self.torch = torch; self.wp = worker_patch
        class Runner:
            def __init__(self, vllm_config): self.vllm_config = vllm_config; self.handled = []
            def accept_registrations(self, pull_socket, num_workers): self.accepted = num_workers
            def busy_loop(self, pull_socket, shutdown_event): raise AssertionError('original loop must not run')
            def _handle_requests(self, requests): self.handled.append([(r.dp_rank, r.num_tokens, r.num_reqs) for r in requests])
        Req = lambda **kw: SimpleNamespace(**kw)
        self.module = SimpleNamespace(PleOffloadRunner=Runner, PleOffloadRequest=Req, torch=torch,
                                      logger=SimpleNamespace(info=lambda *a, **k: None, warning=lambda *a, **k: None))
        worker_patch.RECEIPT.update(handled=0, batches=0, skipped_seq=0, torn_reads=0, idle_polls=0)
        self.Runner = worker_patch.install(self.module)

    def test_poll_decodes_in_order_and_counts_skips(self):
        r = self.Runner(SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=1)))
        flag = self.torch.zeros(4, dtype=self.torch.int32).share_memory_(); flag[3] = 7   # stale record from before the loop
        r._direct_flag_bufs = {0: flag}; stop = threading.Event()
        t = threading.Thread(target=r.busy_loop, args=(None, stop), name='offload'); t.start()
        time.sleep(0.02); self.assertEqual(r.handled, [], 'stale flag must not be served')
        def publish(nt, nr, seq): flag[0] = nt; flag[1] = nr; flag[3] = seq; time.sleep(0.01)
        publish(9, 4, 8); publish(660, 1, 9); publish(8192, 12, 11)   # 10 skipped
        stop.set(); t.join(2); self.assertFalse(t.is_alive())
        self.assertEqual(r.handled, [[(0, 9, 4)], [(0, 660, 1)], [(0, 8192, 12)]])
        self.assertEqual(self.wp.RECEIPT['skipped_seq'], 1); self.assertGreater(self.wp.RECEIPT['idle_polls'], 0)

    def test_accept_registrations_second_message(self):
        r = self.Runner(SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=1)))
        flag = self.torch.zeros(4, dtype=self.torch.int32).share_memory_()
        from multiprocessing.reduction import ForkingPickler
        import torch.multiprocessing as torch_mp
        strategy = torch_mp.get_sharing_strategy(); torch_mp.set_sharing_strategy('file_system')
        try: msgs = [ForkingPickler.dumps(PleDirectFlagRegistration(0, flag))]   # the connector's exact transport
        finally: torch_mp.set_sharing_strategy(strategy)
        sock = SimpleNamespace(recv=lambda: bytes(msgs.pop(0)))
        r.accept_registrations(sock, 1); self.assertEqual(r.accepted, 1)
        got = r._direct_flag_bufs[0]; self.assertTrue(got.is_shared()); flag[3] = 42; self.assertEqual(int(got[3]), 42, 'must alias the shared storage')
        bad = SimpleNamespace(recv=lambda: pickle.dumps(SimpleNamespace(dp_rank=0)))
        r2 = self.Runner(SimpleNamespace(parallel_config=SimpleNamespace(data_parallel_size=1)))
        with self.assertRaises(RuntimeError): r2.accept_registrations(bad, 1)


if __name__ == '__main__':
    unittest.main(verbosity=1)
