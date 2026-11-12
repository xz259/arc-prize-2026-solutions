"""Opt-in graph metadata tests, including replay across deletion/slot changes.

Tensor tests run on CPU when torch is available. CUDA tests capture only the
address-remapping kernel; full-model graph parity remains a separate live gate.
"""
import copy
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from eviction_runner import (EvictionRunnerError, _allocate_buffers,
    _stage_graph_batch, graph_mapping_enabled, validate_execution_mode,
    verify_sources)
from eviction_qsa import QSASelectionMapping, remap_selected_indices_
from test_eviction_runner import runner_fixture, RUNTIME_SOURCE

try:
    import torch
except ImportError:
    torch = None


class GraphGuardTests(unittest.TestCase):
    def test_default_guard_keeps_eager_requirement(self):
        with patch.dict(os.environ, {'ARC3_EVICTION_GRAPH_MAPPING': '0'}):
            config = runner_fixture().vllm_config
            self.assertTrue(validate_execution_mode(config)['eager_only'])
            config.model_config.enforce_eager = False
            with self.assertRaisesRegex(EvictionRunnerError, 'enforce-eager'):
                validate_execution_mode(config)

    def test_opt_in_accepts_matched_eager_and_reviewed_full_piecewise_only(self):
        with patch.dict(os.environ, {'ARC3_EVICTION_GRAPH_MAPPING': '1'}):
            config = runner_fixture().vllm_config
            self.assertTrue(validate_execution_mode(config)['graph_mapping'])
            config.model_config.enforce_eager = False
            config.compilation_config.cudagraph_mode.name = 'FULL_AND_PIECEWISE'
            self.assertFalse(validate_execution_mode(config)['eager_only'])
            for mode in ('FULL', 'PIECEWISE', 'FULL_DECODE_ONLY', 'NONE'):
                config.compilation_config.cudagraph_mode.name = mode
                with self.subTest(mode=mode), self.assertRaises(EvictionRunnerError):
                    validate_execution_mode(config)
            self.assertEqual(len(verify_sources(RUNTIME_SOURCE)), 14)

    def test_invalid_flag_fails_closed(self):
        with patch.dict(os.environ, {'ARC3_EVICTION_GRAPH_MAPPING': 'true'}):
            with self.assertRaises(EvictionRunnerError):
                graph_mapping_enabled()


@unittest.skipIf(torch is None, 'PyTorch unavailable in local harness environment')
class GraphBufferTests(unittest.TestCase):
    def buffers(self, device='cpu'):
        runner = runner_fixture()
        runner.model_state.max_num_reqs = 4
        runner.model_state.device = torch.device(device)
        with patch.dict(os.environ, {'ARC3_EVICTION_GRAPH_MAPPING': '1'}):
            _allocate_buffers(runner)
        return runner.model_state._arc3_eviction_buffers

    def batch(self, slots, lengths, device='cpu', padded=4):
        return SimpleNamespace(num_reqs=len(slots), num_reqs_after_padding=padded,
            idx_mapping=torch.tensor(slots, dtype=torch.int64, device=device),
            idx_mapping_np=np.array(slots, dtype=np.intp),
            seq_lens=torch.tensor(lengths, dtype=torch.int32, device=device))

    def test_stable_pointers_and_reordered_padded_rows_after_eviction(self):
        buffers = self.buffers()
        first = _stage_graph_batch(buffers, self.batch([2,0], [40,30,999,999]),
                                   for_capture=False, padded=True)
        self.assertEqual(first.slots.tolist(), [2,0,-1,-1])
        self.assertEqual(first.virtual_seq_lens.tolist(), [40,30,0,0])
        pointers = tuple(t.data_ptr() for t in (first.slots,first.raw_seq_lens,first.virtual_seq_lens))
        buffers.deleted.copy_(torch.tensor([8,0,3,0]))
        buffers.padding.copy_(torch.tensor([0,0,3,0]))
        second = _stage_graph_batch(buffers, self.batch([0,2,1], [30,40,20,999]),
                                    for_capture=False, padded=True)
        self.assertEqual(second.slots.tolist(), [0,2,1,-1])
        self.assertEqual(second.raw_seq_lens.tolist(), [38,43,20,0])
        self.assertEqual(second.virtual_seq_lens.tolist(), [30,43,20,0])
        self.assertEqual(pointers, tuple(t.data_ptr() for t in
            (second.slots,second.raw_seq_lens,second.virtual_seq_lens)))
        # Same base addresses also cover a smaller, unpadded eager/prefill batch.
        third = _stage_graph_batch(buffers, self.batch([2], [41], padded=1),
                                   for_capture=False, padded=False)
        self.assertEqual(third.raw_seq_lens.tolist(), [44])
        self.assertEqual(third.virtual_seq_lens.data_ptr(), pointers[2])

    def test_capture_does_not_inherit_live_request_shifts(self):
        buffers = self.buffers()
        buffers.deleted.fill_(100)
        buffers.padding.fill_(3)
        capture = _stage_graph_batch(buffers, self.batch([0,1], [1,1,777,777]),
                                     for_capture=True, padded=True)
        self.assertEqual(capture.slots.tolist(), [-1,-1,-1,-1])
        self.assertEqual(capture.raw_seq_lens.tolist(), [1,1,0,0])
        self.assertEqual(capture.virtual_seq_lens.tolist(), [1,1,0,0])

    def test_bad_batch_rejected_before_graph_buffer_mutation(self):
        buffers = self.buffers()
        for slots in ([4], [-1]):
            before = buffers.batch_slots.clone()
            with self.subTest(slots=slots), self.assertRaisesRegex(EvictionRunnerError, 'outside'):
                _stage_graph_batch(buffers, self.batch(slots,[1,0,0,0]),
                                   for_capture=False,padded=True)
            self.assertTrue(torch.equal(before,buffers.batch_slots))

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), 'CUDA unavailable')
    def test_captured_remap_replays_first_eviction_reorder_and_slot_reuse(self):
        device = 'cuda'
        buffers = self.buffers(device)
        batch = self.batch([0,1], [12,12,0,0], device)
        active = _stage_graph_batch(buffers,batch,for_capture=True,padded=True)
        # A graph built before the first eviction must still contain the remap
        # and read all future table, cutoff, padding, length and slot updates.
        query = torch.tensor([11,11,-1,-1], device=device, dtype=torch.int64)
        mapping = QSASelectionMapping(query,active.virtual_seq_lens,buffers.lookup,
            buffers.cutoffs,buffers.padding,active.slots)
        requests = torch.tensor([0,1,2,3], dtype=torch.int32, device=device)
        source = torch.tensor([[0,1,2,3,4,5,6,10,-1]]*4, dtype=torch.int32, device=device)
        output = source.clone()
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                output.copy_(source)
                remap_selected_indices_(output, requests, mapping)
        torch.cuda.current_stream().wait_stream(stream)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output.copy_(source)
            remap_selected_indices_(output, requests, mapping)

        def check(slots,lengths):
            _stage_graph_batch(buffers,self.batch(slots,lengths,device),
                               for_capture=False,padded=True)
            expected = source.cpu()
            cpu_mapping = QSASelectionMapping(query.cpu(),active.virtual_seq_lens.cpu(),
                buffers.lookup.cpu(),buffers.cutoffs.cpu(),buffers.padding.cpu(),active.slots.cpu())
            remap_selected_indices_(expected,requests.cpu(),cpu_mapping)
            graph.replay()
            torch.cuda.synchronize()
            self.assertTrue(torch.equal(output.cpu(),expected))

        check([0,1],[12,12,0,0])
        buffers.deleted[0] = 2
        buffers.padding[0] = 2
        buffers.cutoffs[0] = 6
        buffers.lookup[0,:6].copy_(torch.tensor([0,-1,1,-1,2,3],device=device,dtype=torch.int32))
        check([0,1],[10,12,0,0])
        check([1,0],[12,10,0,0])
        # A request leaves. Its physical state slot is reused by an untouched
        # request; resetting cutoff=0 makes its stale historical table invisible.
        buffers.deleted[0] = 0
        buffers.padding[0] = 0
        buffers.cutoffs[0] = 0
        check([0],[12,999,999,999])
        # Request count grows again and the previously evicted request can live
        # in a different state slot, with the same captured graph addresses.
        buffers.deleted[3] = 2
        buffers.padding[3] = 2
        buffers.cutoffs[3] = 6
        buffers.lookup[3,:6].copy_(buffers.lookup[0,:6])
        check([3,1,0],[10,12,12,0])


if __name__ == '__main__':
    unittest.main()
