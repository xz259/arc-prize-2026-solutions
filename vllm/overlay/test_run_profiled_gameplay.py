import builtins
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import gameplay_harness_profile as profile
import run_profiled_gameplay as runner

STACK = Path(__file__).resolve().parents[3]
RELEASE = STACK/'evidence/prolong-v522-output-20260912/smoke15/verified/runs/prolong-v5.22-release-20260912-v1'


def argv(contract='public25'):
    return ['--total-seconds','900' if contract=='public25' else '150', '--replicas','1',
        '--expected-mtp','0','--expected-sequences','30','--context','40000',
        '--context-profile','nvfp4','--expected-gpu-memory-utilization','0.95',
        '--active-games','0','--request-slots','0','--clock-mode','gameplay',
        '--shutdown-margin','0','--min-p','0','--reasoning-effort','xhigh',
        '--image-scale','4','--stdout-token-limit','1024','--temperature','1',
        '--response-headroom-fraction','0.10','--no-token-limit','--no-response-cap',
        '--games','all' if contract=='public25' else 'ls20-9607627b,ft09-0d8bbf25,vc33-5430563c,tr87-cd924810',
        '--output','/tmp/not-created-by-validation']


def settings(n=25):
    return dict(original_game_ids=[f'g{i}' for i in range(n)],
        game_instances=[{} for _ in range(n)], request_admission='all_games',
        request_slots=n,game_concurrency=n,initial_total_generated_tokens=None,
        context_policy={'response_cap':None},program_index=True,snapshot=True,
        gameplay={},output=None)


class RunnerTests(unittest.TestCase):
    def tearDown(self): profile.deactivate()

    def test_explicit_contracts_and_no_hidden_concurrency_or_token_caps(self):
        for contract in ('public25','smoke'):
            self.assertEqual(runner.validate_contract(argv(contract),contract),argv(contract))
        for extra in (['--token-cap','100'],['--no-snapshot'],['--request-slots','3'],
                      ['--response-cap','4096'],['--expected-kv-gib','8']):
            with self.assertRaises(ValueError): runner.validate_contract(argv()+extra,'public25')
        for option, value in (('--total-seconds','901'),('--expected-sequences','25'),
                              ('--min-p','0.02'),('--games','g0,g1,g2,g3')):
            args=argv();args[args.index(option)+1]=value
            with self.assertRaises(ValueError):runner.validate_contract(args,'public25')
        with self.assertRaises(ValueError):runner.validate_contract(argv('smoke'),'public25')

    def test_resolved_all_games_checks_actual_population(self):
        runner.validate_resolved(settings(),'public25')
        runner.validate_resolved(settings(4),'smoke')
        for mutate in (lambda s:s.update(request_slots=8),lambda s:s.update(game_concurrency=8),
                       lambda s:s['original_game_ids'].__setitem__(0,'g1'),
                       lambda s:s['context_policy'].update(response_cap=4096),
                       lambda s:s.update(snapshot=False)):
            row=settings();mutate(row)
            with self.assertRaises(ValueError):runner.validate_resolved(row,'public25')

    def test_reviewed_111_file_manifest_and_adapter_source_pins(self):
        receipt=runner.verify_sources(RELEASE/'overlay',RELEASE/'source-sha256.json')
        self.assertEqual(receipt['harness_files'],111)
        with tempfile.TemporaryDirectory() as tmp:
            wrong=Path(tmp)/'manifest.json';wrong.write_text('{}')
            with self.assertRaisesRegex(ValueError,'manifest'):
                runner.verify_sources(RELEASE/'overlay',wrong)

    def test_archive_keeps_required_original_and_new_sources(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp);(output/'code-sha256.json').write_text('{}')
            runner.inherited.archive_experiment_sources(output)
            runner.archive_profile_sources(output)
            receipt=json.loads((output/'code-sha256.json').read_text())
            self.assertEqual(len(receipt),12)
            self.assertTrue((output/'implementation/persistent-eviction/HARNESS_HISTORY_DESIGN.md').exists())
            self.assertTrue((output/'implementation/gameplay-profile/gameplay_harness_profile.py').exists())

    def test_function_binding_keeps_clock_order_and_dry_plan_does_not_activate(self):
        def main(remaining):
            print(json.dumps(EXAMPLE))
            if '--execute' in remaining:
                return arm_run_clock(EXAMPLE,SOLVER)
        # These deliberately scoped globals mimic the original runner's callsites.
        scope=dict(main.__globals__,EXAMPLE=settings(),SOLVER=object(),arm_run_clock=lambda *a:None)
        from types import FunctionType
        fake_main=FunctionType(main.__code__,scope,main.__name__,main.__defaults__,main.__closure__)
        base_import=builtins.__import__
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'not-yet-created'
            scope['EXAMPLE']['output']=str(output)
            def arm(settings,solver,**kw):
                self.assertTrue((output/'implementation/gameplay-profile/run_profiled_gameplay.py').exists())
                self.assertIsNotNone(profile._RECORDER)
                settings.update(gameplay_started_epoch=100.0,deadline_epoch=1000.0,total_seconds=900)
                return 1000.0
            fake=SimpleNamespace(main=fake_main,arm_run_clock=arm)
            options=SimpleNamespace(harness_root=RELEASE/'overlay',harness_manifest=RELEASE/'source-sha256.json',
                persistent_base_url='http://127.0.0.1:1234',profile_contract='public25')
            state={}
            bound=runner.make_bound_runner(fake,options=options,history_config={'policy':'oldest_interactions','fraction':.1},
                                           source_receipt={},state=state)
            with contextlib.redirect_stdout(io.StringIO()) as capture:
                bound([])
            self.assertFalse(output.exists())
            self.assertIsNone(profile._RECORDER)
            self.assertIn('external message deltas',capture.getvalue())
            output.mkdir();(output/'code-sha256.json').write_text('{}')
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(bound(['--execute']),1000.0)
            self.assertIs(builtins.__import__,base_import)
            rows=[json.loads(x) for x in (output/'gameplay-harness-profile.jsonl').read_text().splitlines()]
            self.assertEqual(rows[0]['event'],'run_clock')
            self.assertEqual(rows[0]['deadline_epoch'],1000.0)
            self.assertNotIn('arm_run_clock',fake_main.__globals__['arm_run_clock'].__name__)

    def test_contract_is_not_implemented_by_bypassing_old_scope(self):
        with patch.object(runner.inherited,'validate_scope',side_effect=AssertionError('old scope should not run')):
            runner.validate_contract(argv(),'public25')
        raw=Path(runner.__file__).read_text()
        self.assertNotIn('inherited.validate_scope =',raw)

    def test_clock_failure_after_activation_preserves_original_exception(self):
        with tempfile.TemporaryDirectory() as tmp:
            output=Path(tmp)/'run'
            args=argv();args[args.index('--output')+1]=str(output)
            args += ['--execute','--profile-contract','public25','--persistent-base-url','http://127.0.0.1:1234',
                     '--harness-root',str(RELEASE/'overlay'),'--harness-manifest',str(RELEASE/'source-sha256.json')]
            def make(fake,*,options,history_config,source_receipt,state):
                def fail(remaining):
                    output.mkdir()
                    # arm() now publishes settings before activating the journal.
                    state['settings']=settings()
                    state['recorder']=profile.activate(output/'gameplay-harness-profile.jsonl')
                    raise ValueError('original clock failure')
                return fail
            with patch.object(runner,'load_reviewed_runner',return_value=SimpleNamespace(HOST='127.0.0.1',PORT=1234)), \
                 patch.object(runner,'make_bound_runner',side_effect=make):
                with self.assertRaisesRegex(ValueError,'original clock failure'):
                    runner.main(args)
            result=json.loads((output/'profiled-gameplay-result.json').read_text())
            self.assertEqual(result['status'],'failed')
            self.assertEqual(result['error_type'],'ValueError')
            self.assertIsNone(result['gameplay_started_epoch'])
            self.assertIsNone(profile._RECORDER)


if __name__=='__main__':unittest.main()
