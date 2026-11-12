import concurrent.futures
import copy
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gameplay_harness_profile as profile
import persistent_harness_adapter as adapter
from test_persistent_harness_adapter import AdapterTests
from test_harness_v2 import call, response
import test_prolong as harness_fixtures

# The sealed overlay intentionally excludes model assets. Change only the CPU
# fixture's tokenizer lookup, retaining the reviewed production module imports.
STACK = Path(__file__).resolve().parents[3]
class LocatedProlongFixtures(harness_fixtures.ProlongTests):
    make_agent = profile.rebind(harness_fixtures.ProlongTests.make_agent,
                               __file__=str(STACK/'test_prolong.py'))
make_fixture_agent = profile.rebind(AdapterTests.make_agent,
    fixtures=SimpleNamespace(ProlongTests=LocatedProlongFixtures))


class ProfileTests(unittest.TestCase):
    def tearDown(self):
        profile.deactivate()

    def test_lazy_journal_and_thread_local_correlations(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'created-later'/'profile.jsonl'
            recorder = profile.activate(path)
            self.assertFalse(path.parent.exists())
            path.parent.mkdir()
            barrier = threading.Barrier(2)
            def work(i):
                with profile.context(game_id=f'g{i}', operation_id=f'op{i}', receipt={}):
                    with profile.span('parent'):
                        barrier.wait(timeout=5)
                        with profile.span('child'):
                            profile.emit('point')
            with concurrent.futures.ThreadPoolExecutor(2) as pool:
                list(pool.map(work, range(2)))
            rows = [json.loads(x) for x in path.read_text().splitlines()]
            self.assertEqual(len(rows), 6)
            self.assertTrue(recorder.status()['valid'])
            for row in rows:
                self.assertEqual(row['game_id'][-1], row['operation_id'][-1])
                self.assertNotIn('receipt', row)
            children = [r for r in rows if r.get('phase') == 'child']
            parents = {r['span_id']:r for r in rows if r.get('phase') == 'parent'}
            for child in children:
                parent = parents[child['parent_span_id']]
                self.assertEqual(parent['game_id'], child['game_id'])
                self.assertLessEqual(child['seconds'], parent['seconds'])

    def test_profile_io_failure_does_not_change_function_or_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            recorder = profile.activate(Path(tmp)/'missing'/'profile.jsonl')
            self.assertEqual(profile.timed(lambda: 23, 'work')(), 23)
            with self.assertRaisesRegex(ValueError, 'original'):
                with profile.span('failure'):
                    raise ValueError('original')
            self.assertEqual(recorder.status()['dropped'], 2)
            self.assertFalse(recorder.status()['valid'])

    def test_local_rebinding_preserves_super_and_does_not_patch_globals(self):
        class Base:
            def value(self): return 2
        class Child(Base):
            def value(self): return super().value()+len([1])
        original = Child.value
        class Profiled(Child):
            value = profile.rebind(original)
        self.assertEqual(Profiled().value(), 3)
        self.assertIs(Child.value, original)
        self.assertIs(Profiled.value.__closure__, original.__closure__)
        with self.assertRaises(ValueError):
            profile.rebind(original, typo=lambda: None)

    def test_http_parse_received_usage_survives_nested_context(self):
        classes = profile.build_classes()
        response_obj = classes['response']()
        response_obj._content = json.dumps(dict(usage=dict(completion_tokens=2),
            persistent=dict(token_ids=[1,2]))).encode()
        with tempfile.TemporaryDirectory() as tmp:
            profile.activate(Path(tmp)/'profile.jsonl')
            with profile.context(receipt={}) as ctx:
                with profile.span('outer'):
                    self.assertEqual(response_obj.json()['usage']['completion_tokens'], 2)
                self.assertEqual(ctx['receipt']['received_completion_tokens'], 2)
        self.assertIsNone(profile.known_received(dict(usage=dict(completion_tokens=3),persistent=dict(token_ids=[1,2]))))

    def test_actual_python_loop_payloads_and_images_unchanged(self):
        replies = [response([call('inspect', "workspace.run_code('grid.py')\nprint('diagnostic')\ndisplay_grid([[1,2],[3,4]])")]),
                   response([call('done', "print('done')")])]
        def run(tmp, enabled):
            agent, path, transport = make_fixture_agent(AdapterTests(),tmp,copy.deepcopy(replies))
            if enabled:
                classes = profile.build_classes()
                agent.__class__ = classes['agent']
                # Fixture eagerly initialized state; production uses the profiled lazy initializer.
                agent._persistent_external_state = classes['managed_state']()
                agent.workspace.__class__ = classes['workspace']
                profile.activate(Path(tmp)/'profile.jsonl')
            agent._tool_steps = 2
            agent.token_counter.count = Mock(side_effect=AssertionError('historical tokenizer refill'))
            with patch.object(agent, '_chat_transport', side_effect=AssertionError('stateless transport')):
                result = agent.analyze(path, 0, valid_actions=['RIGHT'])
            agent._close_chat_transport()
            wire = copy.deepcopy(transport.calls)
            # Only the run workspace/tool description absolute paths differ.
            wire = json.loads(json.dumps(wire).replace(tmp, '<WORKSPACE>'))
            return result, wire, transport
        with tempfile.TemporaryDirectory() as a, tempfile.TemporaryDirectory() as b:
            original = run(a, False)
            measured = run(b, True)
            self.assertEqual(original[1], measured[1])
            self.assertEqual([x[0] for x in measured[1]], ['start','append'])
            self.assertNotIn('chat', measured[1][1][1])
            self.assertTrue(measured[1][1][1]['external_messages'])
            rows = [json.loads(x) for x in (Path(b)/'profile.jsonl').read_text().splitlines()]
            phases = {r.get('phase') for r in rows}
            self.assertTrue({'python_subprocess','output_filter','output_render','raw_journal',
                             'persistent_stage','managed_suffix','response_normalization'} <= phases)
            completions = [r for r in rows if r['event'] == 'completion_result']
            self.assertEqual(len(completions), 2)
            self.assertTrue(all(r['accepted'] and r['completion_tokens'] > 0 for r in completions))
            tools = [r for r in rows if r.get('phase') == 'tool_dispatch']
            self.assertEqual([r['model_tool_call_id'] for r in tools], ['inspect','done'])
            self.assertEqual(measured[2].closed, ['fixture-session'])

    def test_classes_preserve_original_methods_and_solver_deepcopy(self):
        before = adapter.PersistentProlongAgent._chat_completion
        classes = profile.build_classes()
        self.assertIs(adapter.PersistentProlongAgent._chat_completion, before)
        solver = classes['solver'].__new__(classes['solver'])
        solver.example = dict(a=[1,2])
        cloned = copy.deepcopy(solver)
        self.assertEqual(cloned.example, solver.example)
        self.assertIsNot(cloned.example, solver.example)


if __name__ == '__main__': unittest.main()
