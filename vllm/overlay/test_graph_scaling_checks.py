"""Local policy/client checks; no model, GPU or HTTP requests."""
import asyncio
from contextlib import redirect_stdout
import copy
import hashlib
import io
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import graph_scaling_checks as panel
from raw_gpu_checks import ContractFailure
from session_service import SessionManager
from test_raw_history_eviction_checks import Backend


class CountedBackend(Backend):
    async def turn(self, session, turn, *, initial):
        self.submissions.append((session.session_id, list(turn.token_ids)))
        await asyncio.sleep(0)
        return dict(token_ids=[501] * turn.sampling['max_tokens'],
                    finish_reason='length', stop_reason=None)


class LocalClient(panel.ScalingClient):
    def __init__(self, *args, manager, loop, **kwargs):
        super().__init__(*args, **kwargs)
        self.manager, self.loop = manager, loop

    def call(self, method, route, payload=None):
        async def invoke():
            if route.endswith('/start'):
                return await self.manager.start(copy.deepcopy(payload))
            if route.endswith('/append'):
                return await self.manager.append(self.sid, copy.deepcopy(payload))
            if route.endswith('/close'):
                return await self.manager.close(self.sid)
            raise AssertionError(route)
        return asyncio.run_coroutine_threadsafe(invoke(), self.loop).result(timeout=90)


class ScalingChecksTests(unittest.TestCase):
    def test_dry_run_does_not_execute(self):
        with patch.object(panel, 'run', side_effect=AssertionError('execution')):
            with redirect_stdout(io.StringIO()) as stream:
                self.assertEqual(panel.main([]), 0)
        plan = json.loads(stream.getvalue())
        self.assertEqual(plan['concurrencies'], [25, 30])
        self.assertEqual(plan['maximum_generated_tokens'], 143000)
        self.assertEqual(plan['expected_residency']['final'], 39707)

    def test_real_policy_supports_both_complete_full_concurrency_workloads(self):
        for count in (25, 30):
            with self.subTest(count=count):
                loop = asyncio.new_event_loop()
                thread = threading.Thread(target=loop.run_forever, daemon=True); thread.start()
                backend = CountedBackend()
                manager = SessionManager(backend, max_context=40000, max_sessions=30)
                clients = [LocalClient('http://127.0.0.1:1234', Path('unused'), 1,
                    f'local-{count}-{lane}', deadline=time.monotonic()+120,
                    budget=40000, manager=manager, loop=loop) for lane in range(count)]
                phases = []
                try:
                    panel.exercise(clients, lambda n, offset=0:[100 + offset % 10] * n,
                                   on_phase=phases.append)
                    self.assertEqual(len(backend.submissions), count * 8)
                    self.assertEqual(len(backend.evictions), count)
                    self.assertEqual(phases[-4:], ['before-pre-burst', 'after-pre-burst',
                                                  'before-post-burst', 'after-post-burst'])
                    for client in clients:
                        self.assertEqual(sum(turn['output_tokens'] for turn in client.turns), 2600)
                        self.assertEqual([turn['output_tokens'] for turn in client.turns],
                                         [8, 8, 8, 8, 8, 1024, 1024, 512])
                        self.assertEqual([turn['expected_prefill_tokens'] for turn in client.turns],
                                         [5000, 10001, 10001, 10001, 3501, 1, 4001, 1])
                        self.assertEqual(client.turns[6]['history_eviction']['selected_block_ids'], ['obs-01'])
                        self.assertEqual(client.turns[6]['resident_input_tokens'], 38171)
                        self.assertEqual(client.turns[-1]['resident_tokens'], 39707)
                        self.assertTrue(all(position in client.oracle.resident_positions for position in range(128)))
                finally:
                    asyncio.run_coroutine_threadsafe(manager.close_all(), loop).result(timeout=90)
                    loop.call_soon_threadsafe(loop.stop); thread.join(timeout=5); loop.close()

    def test_batch_delta_requires_exact_full_real_and_padded_shape(self):
        before = {'prepared_batches':[dict(mode='FULL', real_requests=25, padded_requests=25, count=4)]}
        after = {'prepared_batches':[
            dict(mode='FULL', real_requests=25, padded_requests=25, count=15),
            dict(mode='FULL', real_requests=25, padded_requests=30, count=900),
            dict(mode='PIECEWISE', real_requests=25, padded_requests=25, count=900),
            dict(mode='FULL', real_requests=30, padded_requests=30, count=6)]}
        self.assertEqual(panel.exact_batch_delta(before, after, 25)['count'], 11)
        self.assertEqual(panel.exact_batch_delta(before, after, 30)['count'], 6)
        with self.assertRaisesRegex(ContractFailure, 'reset'):
            panel.exact_batch_delta(after, before, 25)

    def test_output_extension_is_explicit_and_bounded(self):
        client = panel.ScalingClient('http://localhost:1234', Path('unused'), 1, 'bound',
                                     deadline=time.monotonic()+30, budget=40000)
        for count in (8, 512, 1024):
            client.set_output_tokens(count)
            body, _ = client.build([2]*10, observation=('first', 1))
            self.assertEqual(body['sampling']['max_tokens'], count)
        for count in (0, 128, 2048, True, 512.0):
            with self.subTest(count=count), self.assertRaises(ValueError):
                client.set_output_tokens(count)

    def test_parallel_error_joins_siblings_before_cleanup(self):
        seen = []
        lock = threading.Lock()
        class Client:
            def __init__(self, index): self.index = index
            def set_output_tokens(self, count): pass
            def submit(self, *args, **kwargs):
                with lock: seen.append(self.index)
                if self.index == 0: raise RuntimeError('intentional')
                return self.index
        with self.assertRaisesRegex(ContractFailure, 'intentional'):
            panel.parallel_submit([Client(i) for i in range(4)], 'test', [[]]*4, output_tokens=8)
        self.assertEqual(sorted(seen), [0, 1, 2, 3])

    def test_pages_cannot_overlap_across_live_owners_within_group(self):
        clients = [SimpleNamespace(sid='one', turns=[{}]), SimpleNamespace(sid='two', turns=[{}])]
        rows = {'one':[dict(event='pause', block_ids=[[1, 2], [7]])],
                'two':[dict(event='pause', block_ids=[[3, 4], [8]])]}
        with patch.object(panel, 'engine_for', side_effect=lambda path, sid:rows[sid]):
            self.assertTrue(panel.assert_owner_pages('unused', clients)['valid'])
            rows['two'][0]['block_ids'][1] = [7]
            with self.assertRaisesRegex(ContractFailure, 'multiple'):
                panel.assert_owner_pages('unused', clients)

    def test_explicit_capture_and_ceiling_requirements(self):
        identity = dict(argv=['vllm', '--cudagraph-capture-sizes', '1', '4', '25', '30', '32',
                              '--max-num-seqs', '30', '--max-model-len', '40000'],
            persistent_experiment=dict(manifest={name:hashlib.sha256((panel.HELPERS/name).read_bytes()).hexdigest()
                for name in ('raw_concurrent_eviction_checks.py','raw_history_eviction_checks.py',
                             'raw_managed_eviction_checks.py','raw_gpu_checks.py')}))
        args = SimpleNamespace(base_url='http://127.0.0.1:1234')
        with patch.object(panel, 'identity_check', return_value=identity):
            self.assertIs(panel.check_launch(args), identity)
            removed = identity['persistent_experiment']['manifest'].pop('raw_gpu_checks.py')
            with self.assertRaisesRegex(ContractFailure, 'manifest is missing'):
                panel.check_launch(args)
            identity['persistent_experiment']['manifest']['raw_gpu_checks.py'] = removed
            identity['argv'].remove('25')
            with self.assertRaisesRegex(ContractFailure, 'capture'):
                panel.check_launch(args)

    def test_execution_rejects_nonlocal_or_wrong_port_before_identity_access(self):
        for address in ('https://127.0.0.1:1234', 'http://127.0.0.1:8000',
                        'http://example.com:1234', 'http://localhost:1234/extra',
                        'http://localhost:1234?x=1', 'http://user@localhost:1234'):
            with self.subTest(address=address), patch.object(panel, 'identity_check',
                    side_effect=AssertionError('identity accessed')):
                with self.assertRaisesRegex(ContractFailure, 'requires http'):
                    panel.check_launch(SimpleNamespace(base_url=address))


if __name__ == '__main__':
    unittest.main()
