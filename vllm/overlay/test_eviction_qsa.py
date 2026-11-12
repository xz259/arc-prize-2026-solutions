"""Group semantics and optional CUDA remap parity; no model serving required."""
import importlib.util
import random
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from eviction_qsa import (QSASelectionMapping, compacted_prefix, descriptor_keep_rows,
                          expand_selected_groups, install_qsa_selector_wrapper,
                          new_group_storage_rows, qsa_layout, remap_selected_indices_,
                          virtual_query_position)
from eviction_storage import EvictionStorageError


def stock_expand(selected, virtual_query, virtual_length, token_topk):
    visible = min((virtual_query+1)//4, virtual_length//4, token_topk//4)
    if len(selected) != visible:
        raise AssertionError("Fixture must supply stock selected count")
    result = [4*group+member for group in selected for member in range(4)]
    result += list(range((virtual_query+1)//4*4, virtual_query+1))
    return result + [-1]*(token_topk+3-len(result))


class GroupMappingTests(unittest.TestCase):
    def test_no_eviction_is_identity_including_future_tail(self):
        for end in range(1, 65):
            prefix = compacted_prefix(range(end+1), logical_computed=end)
            self.assertEqual(prefix.deleted_tokens, 0)
            self.assertEqual(prefix.deleted_groups, 0)
            self.assertEqual(prefix.group_padding, 0)
            self.assertEqual([prefix.remap(i) for i in range(end+40)], list(range(end+40)))

    def test_full_and_partial_group_deletion_keeps_original_members(self):
        positions = (0, 1, 2, 3, 8, 10, 11, 13, 14, 16)
        prefix = compacted_prefix(positions, logical_computed=16)
        self.assertEqual(prefix.compressed_group_ids, (0, 2, 3))
        self.assertEqual(prefix.virtual_to_resident, (0,1,2,3,4,-1,5,6,-1,7,8,-1))
        self.assertEqual((prefix.deleted_tokens, prefix.deleted_groups, prefix.group_padding), (7, 1, 3))
        self.assertEqual(prefix.remap(12), 9)
        self.assertEqual(prefix.remap(-1), -1)

    def test_partial_open_group_keeps_holes_before_future_members(self):
        prefix = compacted_prefix((0, 1, 2, 3, 6), logical_computed=6)
        self.assertEqual(prefix.virtual_to_resident, (0,1,2,3,-1,-1))
        self.assertEqual(prefix.remap(6), 4)
        self.assertEqual(prefix.remap(7), 5)
        completed = qsa_layout((0,1,2,3,6,7,8), logical_computed=8)
        self.assertEqual(completed.members[-1], (-1,-1,4,5))

    def test_selector_equivalence_for_random_partial_evictions_and_queries(self):
        rng = random.Random(471)
        checked = 0
        for _ in range(100):
            end = rng.randrange(4, 100)
            kept = tuple(i for i in range(end) if i == 0 or rng.random() > .45)+(end,)
            layout = qsa_layout(kept, logical_computed=end)
            prefix = compacted_prefix(kept, logical_computed=end)
            for query in kept[:-1]:
                virtual = virtual_query_position(layout, query)
                self.assertEqual((virtual+1)//4, layout.visible_groups(query))
                self.assertEqual(virtual%4, query%4)
                for budget in (4, 8, 16):
                    selected = rng.sample(range(layout.visible_groups(query)),
                                          min(layout.visible_groups(query), budget//4))
                    expanded = stock_expand(selected, virtual, prefix.virtual_cutoff, budget)
                    actual = tuple(prefix.remap(index) for index in expanded)
                    expected = expand_selected_groups(layout, selected, query_position=query, token_topk=budget)
                    self.assertEqual(actual, expected)
                    self.assertTrue(all(index < 0 or kept[index] <= query for index in actual))
                    checked += 1
        self.assertGreater(checked, 5000)

    def test_future_chunk_uses_constant_offsets_without_rebuilding_table(self):
        rng = random.Random(599)
        for end in range(1, 40):
            kept = tuple(i for i in range(end) if i == 0 or rng.random() > .6)+(end,)
            prefix = compacted_prefix(kept, logical_computed=end)
            extended = kept+tuple(range(end+1, end+21))
            layout = qsa_layout(extended, logical_computed=end+20)
            slots = new_group_storage_rows(prefix.compressed_group_ids,
                       old_logical_computed=end, query_positions=range(end, end+20))
            for offset, query in enumerate(range(end, end+20)):
                dense = len(kept)-1+offset
                virtual = dense+prefix.group_padding
                self.assertEqual(virtual, query-4*prefix.deleted_groups)
                self.assertEqual(virtual, virtual_query_position(layout, query))
                self.assertEqual(prefix.remap(virtual), dense)
                self.assertEqual(slots[offset], virtual//4 if query%4 == 3 else -1)
                selected = list(range(min(layout.visible_groups(query), 4)))
                actual = tuple(prefix.remap(index) for index in stock_expand(
                    selected, virtual, end+20-4*prefix.deleted_groups, 16))
                self.assertEqual(actual, expand_selected_groups(layout, selected,
                                  query_position=query, token_topk=16))

    def test_descriptor_rows_are_gathered_without_regrouping(self):
        before = qsa_layout(range(21), logical_computed=20)
        after = qsa_layout((0, 2, 7, 8, 9, 10, 11, 17, 20), logical_computed=20)
        self.assertEqual(descriptor_keep_rows(before, after), (0, 1, 2, 4))
        self.assertEqual(after.group_ids, (0, 1, 2, 4))

    def test_repeated_evictions_recompute_cumulative_offsets(self):
        first = compacted_prefix((0,1,2,3,8,9,10,11,15,16), logical_computed=16)
        second = compacted_prefix((0,1,2,3,10,11,16), logical_computed=16)
        self.assertGreater(second.deleted_tokens, first.deleted_tokens)
        self.assertGreater(second.deleted_groups, first.deleted_groups)
        self.assertEqual(second.remap(second.virtual_cutoff), second.resident_computed)

    def test_invalid_frontier_and_removed_query_rejected(self):
        with self.assertRaises(EvictionStorageError):
            compacted_prefix((0, 1, 3), logical_computed=4)
        layout = qsa_layout((0,2,3,4), logical_computed=4)
        with self.assertRaises(EvictionStorageError):
            virtual_query_position(layout, 1)
        with self.assertRaises(EvictionStorageError):
            new_group_storage_rows((0,), old_logical_computed=4, query_positions=(5,6))

    def test_wrapper_without_mapping_preserves_exact_original_call(self):
        calls = []
        result = object()
        def original(*args):
            calls.append(args)
            return result
        module = SimpleNamespace(qsa_select_paged_tokens=original)
        wrapper = install_qsa_selector_wrapper(lambda: None, module=module)
        values = [object() for _ in range(9)]
        self.assertIs(wrapper(*values), result)
        self.assertEqual(calls, [tuple(values)])
        self.assertIs(install_qsa_selector_wrapper(module=module), wrapper)

    def test_wrapper_preserves_selector_and_only_substitutes_address_metadata(self):
        class Buffer:
            def __init__(self, shape): self.shape = shape
            def __getitem__(self, index): return self
        mapping = QSASelectionMapping(Buffer((3,)), Buffer((2,)), Buffer((2, 20)))
        calls = []
        result = object()
        def original(*args): calls.append(args); return result
        module = SimpleNamespace(qsa_select_paged_tokens=original)
        wrapper = install_qsa_selector_wrapper(lambda: mapping, module=module)
        q, keys, pages, requests, old_positions, old_lengths, out = Buffer((3,8,128)), object(), object(), object(), object(), Buffer((2,)), object()
        with patch("eviction_qsa.remap_selected_indices_", side_effect=lambda indices, req, mp: indices) as remap:
            self.assertIs(wrapper(q, keys, pages, requests, old_positions, old_lengths, 2048, 4, out), result)
        self.assertEqual(calls[0], (q, keys, pages, requests, mapping.virtual_query_positions,
                                    mapping.virtual_seq_lens, 2048, 4, out))
        remap.assert_called_once_with(result, requests, mapping)


try:
    import torch
except ImportError:
    torch = None


@unittest.skipIf(torch is None, "PyTorch unavailable in local harness-only environment")
class TensorMappingTests(unittest.TestCase):
    def mapping(self, device):
        # Current request order [stable slot 1, stable slot 0, invalid].
        return QSASelectionMapping(
            torch.tensor([10,8,-1], device=device),
            torch.tensor([11,9,0], device=device),
            torch.tensor([[0,1,-1,2,3,-1], [0,-1,-1,1,2,3]], device=device, dtype=torch.int32),
            torch.tensor([6,6], device=device), torch.tensor([2,2], device=device),
            torch.tensor([1,0,-1], device=device))

    def test_cpu_reference_handles_deleted_holes_future_tail_and_invalid_rows(self):
        mapping = self.mapping("cpu")
        indices = torch.tensor([[0,1,5,6,10,11,-1], [0,2,3,5,8,9,-1], [0,1,2,3,4,5,6]], dtype=torch.int32)
        requests = torch.tensor([0,1,-1], dtype=torch.int32)
        result = remap_selected_indices_(indices, requests, mapping)
        self.assertIs(result, indices)
        self.assertEqual(result.tolist(), [[0,-1,3,4,8,-1,-1], [0,-1,2,-1,6,-1,-1], [-1]*7])

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_parity_for_padded_noncontiguous_indices_and_slot_reordering(self):
        generator = torch.Generator().manual_seed(771)
        base = torch.randint(-2, 14, (3, 4110), generator=generator, dtype=torch.int32)
        cpu = base[:, ::2]
        gpu_base = base.cuda()
        gpu = gpu_base[:, ::2]
        before = gpu_base.clone()
        requests = torch.tensor([0,1,-1], dtype=torch.int64)
        remap_selected_indices_(cpu, requests, self.mapping("cpu"))
        remap_selected_indices_(gpu, requests.cuda(), self.mapping("cuda"))
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(gpu.cpu(), cpu))
        self.assertTrue(torch.equal(gpu_base[:,1::2], before[:,1::2]))

    @unittest.skipUnless(torch is not None and torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_full_table_mode_matches_cpu_without_analytic_tail(self):
        def mapping(device):
            return QSASelectionMapping(torch.tensor([3,3], device=device),
                   torch.tensor([4,4], device=device),
                   torch.tensor([[0,-1,1,2], [0,1,-1,2]], dtype=torch.int32, device=device))
        cpu = torch.tensor([[0,1,2,3,4,-1], [0,1,2,3,5,-1]], dtype=torch.int32)
        gpu = cpu.cuda()
        requests = torch.tensor([0,1], dtype=torch.int32)
        remap_selected_indices_(cpu, requests, mapping("cpu"))
        remap_selected_indices_(gpu, requests.cuda(), mapping("cuda"))
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(cpu, gpu.cpu()))


if __name__ == "__main__":
    unittest.main()
