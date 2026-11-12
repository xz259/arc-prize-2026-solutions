"""Phase 2 tests: CPU reference semantics, off-path invariance, fused kernel vs reference."""
import sys, unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from eviction_qsa import QSASelectionMapping, remap_selected_indices_
import qsa_frequency as qf

TOPK = 8            # token_topk -> 2 selected groups; width = TOPK + 3 tail slots
LAYER = 5
CAP = 16


def fixture(device='cpu'):
    t = lambda v, dt=torch.int32: torch.tensor(v, dtype=dt, device=device)
    table = torch.arange(0, 40, dtype=torch.int32, device=device).repeat(2, 1)   # identity, then holes
    table[1, 0] = -1                                    # slot 1: first member of descriptor 0 deleted
    mapping = QSASelectionMapping(
        virtual_query_positions=t([23, 7, 29, 23]),     # rows: informative, NOT, informative, informative
        virtual_seq_lens=t([24, 30]),                     # per request
        virtual_to_resident=table,
        virtual_cutoffs=t([28, 20]),                      # per slot: slot0 tail from 28, slot1 from 20
        group_padding=t([2, 1]),
        request_to_state_slot=t([1, 0]))                  # request0 -> slot1, request1 -> slot0
    token_to_req = t([0, 1, 1, 0])
    indices = t([
        [0, 1, 2, 3,   8, 9, 10, 11,   20, 21, 22],       # drows 0,2 ; tail cols excluded
        [0, 1, 2, 3,   4, 5, 6, 7,     -1, -1, -1],       # non-informative row: nothing
        [4, 5, 6, 7,   12, 13, 14, 15, 28, 29, -1],       # drows 1,3 ; tails excluded
        [36, 37, 38, 39, -1, -1, -1, -1, -1, -1, -1],     # invalid (>= seq_len) and padding: nothing, but row is informative
    ])
    counters = SimpleNamespace(hits=torch.zeros((2, 12, CAP), dtype=torch.int32, device=device),
                               informative=torch.zeros((2, 12), dtype=torch.int32, device=device),
                               enabled=torch.ones((1,), dtype=torch.int32, device=device),
                               capacity=CAP, slots=2, layer_index={}, layer_ids=qf.LAYER_IDS)
    return mapping, token_to_req, indices, counters


def expected_hits(device='cpu'):
    h = torch.zeros((2, 12, CAP), dtype=torch.int32, device=device)
    h[1, LAYER, 0] += 1; h[1, LAYER, 2] += 1          # row 0 (slot 1)
    h[0, LAYER, 1] += 1; h[0, LAYER, 3] += 1          # row 2 (slot 0)
    inf = torch.zeros((2, 12), dtype=torch.int32, device=device)
    inf[1, LAYER] = 2                                  # rows 0 and 3
    inf[0, LAYER] = 1                                  # row 2 only (row 1 not informative)
    return h, inf


def fixture_tail(device='cpu'):
    """2026-09-19 live defect: an identity owner (cutoff 0) counted nothing and post-cutoff
    selections were dropped as 'analytic tail'. Both must count with drow = virtual // 4."""
    t = lambda v, dt=torch.int32: torch.tensor(v, dtype=dt, device=device)
    table = torch.arange(0, 40, dtype=torch.int32, device=device).repeat(2, 1)
    mapping = QSASelectionMapping(
        virtual_query_positions=t([23, 23]),
        virtual_seq_lens=t([24, 24]),
        virtual_to_resident=table,
        virtual_cutoffs=t([0, 8]),                      # slot0: no deletions (identity); slot1: table up to 8, shifted tail beyond
        group_padding=t([0, 1]),
        request_to_state_slot=t([0, 1]))
    token_to_req = t([0, 1])
    indices = t([
        [0, 1, 2, 3,   8, 9, 10, 11,   20, 21, 22],    # identity owner: drows 0 and 2 must count
        [8, 9, 10, 11, 0, 1, 2, 3,     20, 21, 22],    # tail group (>= cutoff 8) inside top-k: drows 2 and 0 must count
    ])
    counters = SimpleNamespace(hits=torch.zeros((2, 12, CAP), dtype=torch.int32, device=device),
                               informative=torch.zeros((2, 12), dtype=torch.int32, device=device),
                               enabled=torch.ones((1,), dtype=torch.int32, device=device),
                               capacity=CAP, slots=2, layer_index={}, layer_ids=qf.LAYER_IDS)
    return mapping, token_to_req, indices, counters


def expected_hits_tail(device='cpu'):
    h = torch.zeros((2, 12, CAP), dtype=torch.int32, device=device)
    h[0, LAYER, 0] += 1; h[0, LAYER, 2] += 1
    h[1, LAYER, 2] += 1; h[1, LAYER, 0] += 1
    inf = torch.zeros((2, 12), dtype=torch.int32, device=device)
    inf[0, LAYER] = 1; inf[1, LAYER] = 1
    return h, inf


class TailOwnerTests(unittest.TestCase):
    def test_identity_and_shifted_tail_owners_count(self):
        mapping, req, idx, c = fixture_tail()
        plain = remap_selected_indices_(idx.clone(), req, mapping)
        counted = remap_selected_indices_(idx.clone(), req, mapping, c, layer=LAYER, token_topk=TOPK)
        self.assertTrue(torch.equal(plain, counted))
        h, inf = expected_hits_tail()
        self.assertTrue(torch.equal(c.hits, h), f'hits:\n{c.hits[:, LAYER]}'); self.assertTrue(torch.equal(c.informative, inf))
        self.assertEqual(plain[0].tolist()[:8], list(range(0, 4)) + list(range(8, 12)))     # identity owner remaps to itself
        self.assertEqual(plain[1].tolist()[:8], [7, 8, 9, 10, 0, 1, 2, 3])                 # tail shifted by padding 1, table for the rest

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_gpu_kernel_counts_identity_and_tail_owners(self):
        try:
            torch.zeros(1, device='cuda')
        except Exception as error:                      # noqa: BLE001
            self.skipTest('CUDA context unavailable beside the loaded server: ' + repr(error))
        from _eviction_qsa_gpu import remap_indices_cuda_
        cm, creq, cidx, cc = fixture_tail()
        cpu_plain = remap_selected_indices_(cidx.clone(), creq, cm)
        qf.reference_update(cc, cidx, creq, cm, layer=LAYER, token_topk=TOPK)
        gm, greq, gidx, gc = fixture_tail('cuda')
        g_plain = remap_indices_cuda_(gidx.clone(), greq, gm)
        g_counted = remap_indices_cuda_(gidx.clone(), greq, gm, gc, LAYER, TOPK)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(g_plain.cpu(), cpu_plain)); self.assertTrue(torch.equal(g_counted.cpu(), cpu_plain))
        h, inf = expected_hits_tail()
        self.assertTrue(torch.equal(gc.hits.cpu(), h), f'GPU hits:\n{gc.hits.cpu()[:, LAYER]}'); self.assertTrue(torch.equal(gc.informative.cpu(), inf))


class ReferenceTests(unittest.TestCase):
    def test_reference_counts_designated_slots_only(self):
        mapping, req, idx, c = fixture()
        qf.reference_update(c, idx, req, mapping, layer=LAYER, token_topk=TOPK)
        h, inf = expected_hits()
        self.assertTrue(torch.equal(c.hits, h)); self.assertTrue(torch.equal(c.informative, inf))

    def test_disabled_flag_blocks_all_updates(self):
        mapping, req, idx, c = fixture(); c.enabled.zero_()
        qf.reference_update(c, idx, req, mapping, layer=LAYER, token_topk=TOPK)
        self.assertEqual(int(c.hits.sum()), 0); self.assertEqual(int(c.informative.sum()), 0)

    def test_hole_in_first_member_still_counts_the_descriptor(self):
        mapping, req, idx, c = fixture()
        self.assertEqual(int(mapping.virtual_to_resident[1, 0]), -1)
        qf.reference_update(c, idx, req, mapping, layer=LAYER, token_topk=TOPK)
        self.assertEqual(int(c.hits[1, LAYER, 0]), 1)

    def test_capacity_bound(self):
        mapping, req, idx, c = fixture(); c.capacity = 2; c.hits = c.hits[:, :, :2].clone()
        qf.reference_update(c, idx, req, mapping, layer=LAYER, token_topk=TOPK)
        self.assertEqual(int(c.hits.sum()), 2)          # drows 0,1 kept; 2,3 dropped

    def test_layer_map_requires_distinct_registered_caches(self):
        class Cache:  # data_ptr stand-in
            def __init__(self, p): self.p = p
            def data_ptr(self): return self.p
        bound = {qf.CACHE_NAME.format(l): SimpleNamespace(kv_cache=Cache(1000 + i)) for i, l in enumerate(qf.LAYER_IDS)}
        m = qf.layer_map(bound)
        self.assertEqual(sorted(m.values()), list(range(12)))
        bound[qf.CACHE_NAME.format(7)].kv_cache.p = 1000
        with self.assertRaises(qf.QSAFrequencyError): qf.layer_map(bound)
        del bound[qf.CACHE_NAME.format(3)]
        with self.assertRaises(qf.QSAFrequencyError): qf.layer_map(bound)


class LazyLayerMapTests(unittest.TestCase):
    def test_layer_map_resolves_on_first_call_not_at_allocation(self):
        class Cache:
            def __init__(self, p): self.p = p
            def data_ptr(self): return self.p
        bound = {}
        c = qf.allocate(torch, slots=2, capacity=CAP, device='cpu', bound_layers=lambda: bound)   # caches absent: must not raise
        self.assertEqual(c.layer_index, {})
        for i, l in enumerate(qf.LAYER_IDS):
            bound[qf.CACHE_NAME.format(l)] = SimpleNamespace(kv_cache=Cache(5000 + i))         # caches appear later
        self.assertEqual(qf.layer_for_cache(c, Cache(5000 + 3)), 3)
        self.assertEqual(len(c.layer_index), 12)
        with self.assertRaises(qf.QSAFrequencyError): qf.layer_for_cache(c, Cache(1))
        with self.assertRaises(qf.QSAFrequencyError): qf.allocate(torch, slots=2, capacity=CAP, device='cpu', bound_layers={})


class IntegrationTests(unittest.TestCase):
    def test_cpu_remap_unchanged_by_counters(self):
        mapping, req, idx, c = fixture()
        plain = remap_selected_indices_(idx.clone(), req, mapping)
        counted = remap_selected_indices_(idx.clone(), req, mapping, c, layer=LAYER, token_topk=TOPK)
        self.assertTrue(torch.equal(plain, counted))
        h, inf = expected_hits()
        self.assertTrue(torch.equal(c.hits, h)); self.assertTrue(torch.equal(c.informative, inf))
        self.assertEqual(int(plain[3, 0]), -1)          # invalid virtual remapped to -1
        self.assertEqual(int(plain[0, 8]), 20 - 1)      # tail: virtual - padding[slot1]

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA unavailable')
    def test_gpu_kernel_matches_reference_and_off_path(self):
        try:
            torch.zeros(1, device='cuda')
        except Exception as error:                      # noqa: BLE001
            self.skipTest('CUDA context unavailable beside the loaded server: ' + repr(error))
        from _eviction_qsa_gpu import remap_indices_cuda_
        cm, creq, cidx, cc = fixture()
        cpu_plain = remap_selected_indices_(cidx.clone(), creq, cm)
        qf.reference_update(cc, cidx, creq, cm, layer=LAYER, token_topk=TOPK)
        gm, greq, gidx, gc = fixture('cuda')
        g_plain = remap_indices_cuda_(gidx.clone(), greq, gm)
        g_counted = remap_indices_cuda_(gidx.clone(), greq, gm, gc, LAYER, TOPK)
        torch.cuda.synchronize()
        self.assertTrue(torch.equal(g_plain.cpu(), cpu_plain), 'off-path GPU remap differs from CPU reference')
        self.assertTrue(torch.equal(g_counted.cpu(), cpu_plain), 'counted GPU remap changed indices')
        self.assertTrue(torch.equal(gc.hits.cpu(), cc.hits), f'hits differ:\n{gc.hits.cpu()[:, LAYER]}\n{cc.hits[:, LAYER]}')
        self.assertTrue(torch.equal(gc.informative.cpu(), cc.informative))
        # repeat: counts accumulate exactly, indices stay stable
        remap_indices_cuda_(gidx.clone(), greq, gm, gc, LAYER, TOPK); torch.cuda.synchronize()
        self.assertTrue(torch.equal(gc.hits.cpu(), cc.hits * 2))


if __name__ == '__main__':
    unittest.main(verbosity=1)
