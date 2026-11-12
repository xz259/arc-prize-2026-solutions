"""CPU position proofs; actual source MRoPE with a numpy-only torch shim."""
import ast
import copy
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from eviction_runner import (EvictionRunnerError,validate_context,original_mrope_rows,
    historical_virtual_lookup,prepare_worker_eviction,apply_worker_eviction,_assert_active,
    verify_sources, graph_mapping_enabled)


RUNTIME_SOURCE = Path(os.environ.get('ARC3_RUNTIME_SOURCE',
    str(Path(__file__).parent.parent/'persistent_state_20260912/source-snapshot/vllm')))
SOURCE = RUNTIME_SOURCE/'model_executor/models/qwen3_vl.py'


def source_position_model():
    """Execute the two pinned pure-CPU source methods, without importing vLLM."""
    module = ast.parse(SOURCE.read_text())
    cls = next(n for n in module.body if isinstance(n,ast.ClassDef)
               and n.name == 'Qwen3VLForConditionalGeneration')
    names = ('_iter_mm_grid_hw','_get_mrope_input_positions')
    functions = []
    for name in names:
        fn = copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name == name))
        fn.decorator_list = []
        functions.append(fn)
    tree = ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*functions],type_ignores=[])
    namespace = {'np':np,'torch':SimpleNamespace(from_numpy=lambda value:value)}
    exec(compile(ast.fix_missing_locations(tree),str(SOURCE),'exec'),namespace)
    holder = SimpleNamespace(_iter_mm_grid_hw=namespace['_iter_mm_grid_hw'])
    namespace['Qwen3VLForConditionalGeneration'] = holder
    config = SimpleNamespace(video_token_id=248057,vision_start_token_id=248053,
                             vision_end_token_id=248054,vision_config=SimpleNamespace(spatial_merge_size=2))
    return SimpleNamespace(config=config,get_mrope_input_positions=lambda ids,features:
        namespace['_get_mrope_input_positions'](ids,features,config))


def context(length=50,deleted=range(7,13),computed=None):
    computed = length-1 if computed is None else computed
    deleted = set(deleted)
    positions = [p for p in range(length) if p not in deleted]
    groups = {p//4 for p in positions if p < 4*(computed//4)}
    return dict(logical_token_ids=list(range(length)),resident_positions=positions,
        logical_computed=computed,eviction_revision=int(bool(deleted)),deleted_tokens=len(deleted),
        deleted_groups=computed//4-len(groups),logical_image_features=[])


LIMITS = dict(max_resident_tokens=100,max_logical_tokens=300,ngram_context_len=2)


def runner_fixture():
    model = source_position_model()
    state = type('Qwen3_8FlashNextModelState',(),{})()
    state.rope_state = object()
    state.model = model
    state.max_model_len = 100
    state.ngram_context_len = 2
    state._align_mode = False
    state.mm_pruner = None
    config = SimpleNamespace(compilation_config=SimpleNamespace(cudagraph_mode=SimpleNamespace(name='NONE')),
        model_config=SimpleNamespace(enforce_eager=True,hf_text_config=SimpleNamespace(max_position_embeddings=300)),
        parallel_config=SimpleNamespace(tensor_parallel_size=1,pipeline_parallel_size=1),
        num_speculative_tokens=0,speculative_config=None)
    return SimpleNamespace(vllm_config=config,model_state=state,_arc3_eviction_states={},
                           req_states=SimpleNamespace(req_id_to_index={'game':0}))


class RunnerPositionTests(unittest.TestCase):
    def test_source_hashes_match_reviewed_snapshot(self):
        self.assertEqual(len(verify_sources(RUNTIME_SOURCE)),14 if graph_mapping_enabled() else 10)

    def test_context_preserves_logical_ids_and_frontiers(self):
        value = context()
        dense = [value['logical_token_ids'][p] for p in value['resident_positions']]
        validated = validate_context(value,dense_ids=dense,dense_computed=len(dense)-1,**LIMITS)
        self.assertEqual(validated,value)
        self.assertIsNot(validated,value)
        with self.assertRaises(EvictionRunnerError):
            validate_context(value,dense_ids=dense+[3],**LIMITS)
        with self.assertRaises(EvictionRunnerError):
            validate_context(value,dense_computed=len(dense),**LIMITS)

    def test_contiguous_appended_tail_and_original_ple_history(self):
        value = context(length=60,computed=49)
        validated = validate_context(value,**LIMITS)
        self.assertEqual(validated['resident_positions'][-11:],list(range(49,60)))
        # A newly sampled token or appended tool/image input has original
        # scalar ordinal equal to the dense ordinal plus deleted_tokens.
        for dense,logical in enumerate(value['resident_positions']):
            if logical >= value['logical_computed']:
                self.assertEqual(dense+value['deleted_tokens'],logical)
        with self.assertRaisesRegex(EvictionRunnerError,'PLE'):
            validate_context(context(deleted=[47]),**LIMITS)
        with self.assertRaisesRegex(EvictionRunnerError,'contiguous tail'):
            validate_context(context(length=60,computed=49,deleted=[52]),**LIMITS)

    def test_actual_multimodal_rope_columns_survive_image_deletion(self):
        value = context()
        value['logical_image_features'] = [dict(offset=8,length=4,grid_thw=[1,4,4]),
                                           dict(offset=20,length=4,grid_thw=[1,4,4])]
        model = source_position_model()
        rows,delta = original_mrope_rows(model,value)
        original = context(deleted=[])
        original['logical_image_features'] = value['logical_image_features']
        full,original_delta = original_mrope_rows(model,original)
        self.assertEqual(rows,[[r[p] for p in value['resident_positions']] for r in full])
        self.assertEqual(delta,original_delta+value['deleted_tokens'])
        # The outer model's image MRoPE is not the inner LM's flat arange.
        self.assertNotEqual(full[0],list(range(50)))
        self.assertNotEqual(full[1],full[2])
        logical_next = 50
        dense_next = len(value['resident_positions'])
        self.assertEqual(dense_next+delta,logical_next+original_delta)

    def test_new_image_positions_use_original_prior_images(self):
        value = context(length=70,computed=49)
        value['logical_image_features'] = [dict(offset=8,length=4,grid_thw=[1,4,4]),
                                           dict(offset=52,length=4,grid_thw=[1,4,4])]
        rows,_ = original_mrope_rows(source_position_model(),value)
        retained_index = value['resident_positions'].index(52)
        # The newly supplied image starts after the first original image's
        # compressed MRoPE advance, although that first image is no longer resident.
        self.assertEqual(rows[0][retained_index],50)

    def test_virtual_group_slots_include_holes_and_future_analytic_tail(self):
        value = validate_context(context(),**LIMITS)
        table,cutoff,padding = historical_virtual_lookup(value)
        groups = sorted({p//4 for p in value['resident_positions']})
        lookup = {p:i for i,p in enumerate(value['resident_positions'])}
        expected = [lookup.get(g*4+r,-1) for g in groups for r in range(4)][:cutoff]
        self.assertEqual(table,expected)
        self.assertIn(-1,table)
        self.assertEqual(padding,2)
        self.assertEqual(cutoff-padding,len(value['resident_positions']))
        for offset in range(20):
            self.assertEqual(cutoff+offset-padding,len(value['resident_positions'])+offset)

    def test_prepare_then_publish_has_no_existing_state_mutation(self):
        runner = runner_fixture()
        prepared = prepare_worker_eviction(runner,'game',context(),[[1],[2]])
        self.assertEqual(runner._arc3_eviction_states,{})
        result = apply_worker_eviction(runner,'game',prepared)
        self.assertTrue(result['awaiting_readd'])
        with self.assertRaisesRegex(EvictionRunnerError,'before streaming re-add'):
            _assert_active(runner,SimpleNamespace(req_ids=['game']))
        _assert_active(runner,SimpleNamespace(req_ids=['other']))

    def test_preflight_guards_fail_before_publication(self):
        patches = [dict(deleted_tokens=9),dict(deleted_groups=9),dict(logical_computed=50),
                   dict(resident_positions=[1,1]),dict(eviction_revision=-1),
                   dict(logical_image_features=[dict(offset=9,length=4,grid_thw=[2,4,4])])]
        for patch in patches:
            runner = runner_fixture()
            with self.subTest(patch=patch),self.assertRaises(EvictionRunnerError):
                prepare_worker_eviction(runner,'game',context()|patch,[[1]])
            self.assertEqual(runner._arc3_eviction_states,{})
        runner = runner_fixture()
        runner.vllm_config.model_config.enforce_eager = False
        with self.assertRaises(EvictionRunnerError):
            prepare_worker_eviction(runner,'game',context(),[[1]])

    def test_bad_image_grid_length_refused(self):
        value = context()
        value['logical_image_features'] = [dict(offset=8,length=4,grid_thw=[1,6,4])]
        with self.assertRaisesRegex(EvictionRunnerError,'grid and visual token length'):
            original_mrope_rows(source_position_model(),value)


class TorchWorkerBufferTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import torch
        except ImportError:
            raise unittest.SkipTest('PyTorch unavailable in local harness-only environment')
        cls.torch = torch

    def test_staged_cpu_tensors_preserve_original_mrope_and_slot_maps(self):
        from eviction_runner import _allocate_buffers,_stage_request
        runner = runner_fixture()
        runner.model_state.max_num_reqs = 2
        runner.model_state.device = self.torch.device('cpu')
        _allocate_buffers(runner)
        value = context()
        data = SimpleNamespace(req_id='game',
            prefill_token_ids=[value['logical_token_ids'][p] for p in value['resident_positions']],
            num_computed_tokens=len(value['resident_positions'])-1)
        rows,delta = _stage_request(runner,1,data,value)
        buffers = runner.model_state._arc3_eviction_buffers
        table,cutoff,padding = historical_virtual_lookup(value)
        self.assertEqual(buffers.lookup[1,:cutoff].tolist(),table)
        self.assertEqual(buffers.cutoffs.tolist(),[0,cutoff])
        self.assertEqual(buffers.padding.tolist(),[0,padding])
        self.assertEqual(buffers.deleted.tolist(),[0,6])
        self.assertEqual(rows,[value['resident_positions']]*3)
        self.assertEqual(delta,6)
        bad = copy.deepcopy(value)
        bad['logical_token_ids'][0] = 1000
        changed = copy.deepcopy(data)
        changed.prefill_token_ids[0] = 1000
        old_table = buffers.lookup.clone()
        with self.assertRaisesRegex(EvictionRunnerError,'prefix was rewritten'):
            _stage_request(runner,1,changed,bad)
        self.assertTrue(self.torch.equal(old_table,buffers.lookup))


if __name__ == '__main__':
    unittest.main()
