"""Phase 3 tests: counter lifecycle helpers (pure CPU)."""
import sys, unittest
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
import qsa_frequency as qf


def counters(slots=3, layers=12, cap=8):
    c = SimpleNamespace(hits=torch.zeros((slots, layers, cap), dtype=torch.int32),
                        informative=torch.zeros((slots, layers), dtype=torch.int32),
                        enabled=torch.zeros((1,), dtype=torch.int32), capacity=cap, slots=slots)
    for s in range(slots):
        c.hits[s] = torch.arange(cap, dtype=torch.int32).repeat(layers, 1) + 100 * s
        c.informative[s] = 7 + s
    return c


class LifecycleTests(unittest.TestCase):
    def test_fresh_admission_clears_only_that_slot_and_arms_flag(self):
        c = counters(); qf.on_fresh_admission(c, 1)
        self.assertEqual(int(c.hits[1].sum()), 0); self.assertEqual(int(c.informative[1].sum()), 0)
        self.assertNotEqual(int(c.hits[0].sum()), 0); self.assertNotEqual(int(c.hits[2].sum()), 0)
        self.assertEqual(int(c.enabled[0]), 1)
        with self.assertRaises(qf.QSAFrequencyError): qf.on_fresh_admission(c, 3)

    def test_compaction_gathers_descriptor_rows_in_new_order(self):
        c = counters(); before = c.hits[2].clone()
        qf.on_compaction(c, 2, (0, 3, 5), reset_epoch=False)
        self.assertTrue(torch.equal(c.hits[2][:, :3], before[:, [0, 3, 5]]))
        self.assertEqual(int(c.hits[2][:, 3:].sum()), 0)
        self.assertEqual(int(c.informative[2, 0]), 9)                 # preserved
        self.assertTrue(torch.equal(c.hits[1], counters().hits[1]))   # other slots untouched

    def test_history_eviction_resets_epoch(self):
        c = counters(); qf.on_compaction(c, 0, (0, 1), reset_epoch=True)
        self.assertEqual(int(c.hits[0].sum()), 0); self.assertEqual(int(c.informative[0].sum()), 0)
        self.assertNotEqual(int(c.hits[1].sum()), 0)

    def test_compaction_rejects_bad_rows(self):
        c = counters()
        for rows in ((0, 0), (0, 99), (-1,)):
            with self.assertRaises(qf.QSAFrequencyError): qf.on_compaction(c, 0, rows, reset_epoch=False)
        qf.on_compaction(c, 0, (), reset_epoch=False); self.assertEqual(int(c.hits[0].sum()), 0)

    def test_keep_rows_from_wire_requires_agreement(self):
        plan = lambda rows: SimpleNamespace(keep_rows=rows)
        wire = dict(groups=[dict(copies=[dict(kind='main', plan=plan((9,))), dict(kind='compressed', plan=plan((0, 2)))]),
                            dict(copies=[dict(kind='compressed', plan=plan((0, 2)))])])
        self.assertEqual(qf.descriptor_keep_rows_from_wire(wire), (0, 2))
        wire['groups'][1]['copies'][0]['plan'] = plan((0, 3))
        with self.assertRaises(qf.QSAFrequencyError): qf.descriptor_keep_rows_from_wire(wire)
        self.assertIsNone(qf.descriptor_keep_rows_from_wire(dict(groups=[dict(copies=[dict(kind='main', plan=plan((1,)))])])))


class SnapshotFrontierTests(unittest.TestCase):
    """2026-09-19 live defect: the worker's stored admission context has logical_computed == 0 for a
    fresh owner, so a layout built from it rejects every real snapshot. The core now supplies the
    paused frontier; the worker cross-checks and reads the counters for that layout."""

    def worker(self, stored_positions, stored_computed, slot=1):
        c = counters(slots=3, cap=20000)
        runner = SimpleNamespace(model_state=SimpleNamespace(_arc3_qsa_frequency=c), device='cpu',
                                 req_states=SimpleNamespace(req_id_to_index={'r': slot}),
                                 _arc3_eviction_states={'r': dict(resident_positions=list(stored_positions), logical_computed=stored_computed,
                                                                  awaiting_readd=False, req_index=slot, eviction_revision=2, deleted_tokens=0)})
        return SimpleNamespace(model_runner=runner), c

    def test_worker_uses_core_frontier_not_admission_context(self):
        w, c = self.worker(range(36000), 0)                      # fresh owner as admitted
        positions, computed = list(range(36020)), 36019          # paused after a 20-token turn, one accepted uncomputed token
        r = qf.worker_snapshot(w, 'r', positions, computed)
        self.assertEqual(r['rows'], 36019 // 4)
        self.assertEqual(qf._hits_matrix(r).shape[1], r['rows']); self.assertEqual(r['context']['logical_computed'], computed)
        self.assertEqual(r['context']['eviction_revision'], 2)
        with self.assertRaises(qf.QSAFrequencyError):           # the old construction: admission frontier
            qf.worker_snapshot(w, 'r', list(range(36000)), 0)
        with self.assertRaises(qf.QSAFrequencyError):           # core positions must extend the stored ones
            qf.worker_snapshot(w, 'r', [0, 2, 3], 3)
        with self.assertRaises(qf.QSAFrequencyError):           # two uncomputed tokens are rejected as a fallback, not a crash
            qf.worker_snapshot(w, 'r', list(range(36020)), 36018)

    def test_worker_after_readd_accepts_extended_positions(self):
        stored = [p for p in range(36000) if not 4000 <= p < 8000]
        w, c = self.worker(stored, 36000)
        positions = stored + list(range(36000, 36100))
        r = qf.worker_snapshot(w, 'r', positions, 36099)
        self.assertEqual(r['context']['resident_positions'], positions)

    def test_core_builds_paused_context_on_a_private_copy(self):
        import copy, sys
        class Ledger:
            def __init__(self): self.logical_tokens = 36000; self.resident_tokens = 36000; self.evicted = []
            def resident_positions(self): return list(range(self.logical_tokens))
        class State:
            def __init__(self):
                self.logical_ids = list(range(36000)); self.prompt_logical_length = 36000; self.ledger = Ledger()
                self.eviction_revision = 0; self.poisoned = False
            def paused(self, request):
                self.logical_ids = self.logical_ids[:self.prompt_logical_length] + list(request._output_token_ids)
                self.ledger.logical_tokens = len(self.logical_ids)
            def context(self, request):
                return dict(resident_positions=self.ledger.resident_positions(), logical_computed=request.num_computed_tokens,
                            deleted_tokens=0, eviction_revision=self.eviction_revision)
        state = State(); live = copy.deepcopy(state.__dict__['logical_ids']), state.ledger.logical_tokens
        request = SimpleNamespace(request_id='arc3-r-deadbeef', status=SimpleNamespace(name='WAITING_FOR_STREAMING_REQ'), num_in_flight_tokens=0,
                                  _output_token_ids=list(range(20)), num_computed_tokens=36019, _arc3_eviction_state=state)
        calls = []
        def rpc(fn, timeout, args):
            calls.append((fn, args))
            return [dict(slot=1, rows=args[2] // 4, hits=[], informative=[], context=dict(resident_positions=args[1], logical_computed=args[2], eviction_revision=0, deleted_tokens=0))]
        core = SimpleNamespace(scheduler=SimpleNamespace(requests={'arc3-r-deadbeef': request}), collective_rpc=rpc)
        sys.modules['engine_eviction'] = SimpleNamespace(_request_for_external=lambda s, e: s.requests['arc3-r-deadbeef'])
        try:
            r = qf.core_snapshot(core, 'arc3-r')
        finally:
            del sys.modules['engine_eviction']
        self.assertEqual(calls[0][0], qf.worker_snapshot); self.assertEqual(calls[0][1], ('arc3-r-deadbeef', list(range(36020)), 36019))
        self.assertEqual(r['rows'], 36019 // 4)
        self.assertEqual((state.logical_ids, state.ledger.logical_tokens), (list(range(36000)), 36000))   # live state untouched


if __name__ == '__main__':
    unittest.main(verbosity=1)
