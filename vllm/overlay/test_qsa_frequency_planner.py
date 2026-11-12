"""Phase 4 tests: frequency scoring and the ranked whole-interaction planner (pure CPU)."""
import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from history_eviction import HistoryPolicy, HistoryRegistry, HistoryValidationError, HistoryCapacityError
from residency import ResidentLedger
import qsa_frequency as qf

L = 12


def setup(starts, logical, *, budget=None, fraction=0.20, policy="qsa_frequency", protect=0.0):
    registry = HistoryRegistry()
    registry.commit(registry.plan_registration(
        [{"observation_id": f"obs{i}", "local_start_token": s} for i, s in enumerate(starts)],
        logical_offset=0, delta_tokens=logical))
    ledger = ResidentLedger(logical, ())
    config = HistoryPolicy.from_payload({"policy": policy, "fraction": fraction, "resident_token_budget": budget or logical,
                                         "protect_recent_fraction": protect}, max_context=max(logical, budget or 1))
    return registry, ledger, config


def freq_from_ranges(logical, ranges_to_value, default=1.0, none_ranges=()):
    """frequency dict where original group g gets the value of the range containing 4g."""
    table = {}
    for g in range(logical // 4 + 1):
        p = g * 4
        v = default
        for (a, b), value in ranges_to_value.items():
            if a <= p < b: v = value
        for a, b in none_ranges:
            if a <= p < b: v = None
        table[g] = v
    return dict(frequency=[dict(table) for _ in range(L)], exposure={}, informative=[10] * L,
                epoch_query_positions=10, epoch_start_logical=0, logical_computed=logical)


def selected(decision):
    return decision.receipt["selected_block_ids"]


class PolicyTests(unittest.TestCase):
    def test_policy_accepts_new_fields_and_validates(self):
        p = HistoryPolicy.from_payload({"policy": "qsa_frequency", "fraction": 0.2, "protect_recent_fraction": 0.2}, max_context=40000)
        self.assertEqual((p.policy, p.protect_recent_fraction), ("qsa_frequency", 0.2))
        for bad in ({"policy": "qsa_frequency", "protect_recent_fraction": 1.0}, {"policy": "qsa_frequency", "protect_recent_fraction": -0.1},
                    {"policy": "qsa_frequency", "unknown": 1}):
            with self.assertRaises(HistoryValidationError): HistoryPolicy.from_payload(bad, max_context=40000)
        self.assertEqual(HistoryPolicy.from_payload({"policy": "oldest_interactions"}, max_context=40000).protect_recent_fraction, 0.0)


class ScoringTests(unittest.TestCase):
    def test_group_frequencies_uses_analytic_exposure(self):
        computed = 4096
        rows = computed // 4
        hits = [[0] * rows for _ in range(L)]; hits[3] = list(range(rows))
        snap = dict(rows=rows, hits=hits, informative=[2048] * L,
                    context=dict(resident_positions=list(range(computed)), logical_computed=computed, eviction_revision=0, deleted_tokens=0))
        f = qf.group_frequencies(snap, epoch_start_logical=2048)
        self.assertEqual(f["epoch_query_positions"], 2048)
        self.assertEqual(f["exposure"][0], 2048)                        # old group: whole epoch
        self.assertEqual(f["exposure"][600], computed - 2404)            # born mid-epoch: 4*601 = 2404
        self.assertAlmostEqual(f["frequency"][3][0], 0 / 2048)
        self.assertAlmostEqual(f["frequency"][3][600], 600 / (computed - 2404))
        self.assertEqual(f["frequency"][0][600], 0.0)
        last = max(f["exposure"])
        self.assertTrue(all(v >= 0 for v in f["exposure"].values()))
        self.assertLessEqual(last, rows - 1)

    def test_score_blocks_splits_boundary_group_and_normalizes_by_length(self):
        f = freq_from_ranges(100, {(0, 40): 2.0, (40, 100): 0.5})
        # block A: tokens [10, 42) crosses group 10 (positions 40..43): 2 tokens at 0.5, 30 at 2.0
        # block B: tokens [42, 62): all 0.5
        eligible = [(type("O", (), {"observation_id": "A", "start": 10})(), 42, ((10, 42),), 32),
                    (type("O", (), {"observation_id": "B", "start": 42})(), 62, ((42, 62),), 20)]
        scores, coverage, ok = qf.score_blocks(eligible, f)
        self.assertAlmostEqual(scores[0], (30 * 2.0 + 2 * 0.5) / 32)
        self.assertAlmostEqual(scores[1], 0.5)
        self.assertTrue(ok); self.assertEqual(coverage, [1.0, 1.0])

    def test_coverage_counts_zero_exposure_groups(self):
        f = freq_from_ranges(100, {}, none_ranges=((0, 30),))
        eligible = [(type("O", (), {"observation_id": "A", "start": 0})(), 40, ((0, 40),), 40)]
        scores, coverage, ok = qf.score_blocks(eligible, f)
        self.assertAlmostEqual(coverage[0], 8 / 40)   # groups 0..7 blanked -> positions 0..31 uncovered
        self.assertFalse(ok)


class RankedPlannerTests(unittest.TestCase):
    STARTS = [0, 1000, 2000, 3000, 4000]; LOGICAL = 5000

    def plan(self, frequency, **kw):
        registry, ledger, config = setup(self.STARTS, self.LOGICAL, **kw)
        return registry.plan_eviction(config, ledger, incoming_tokens=1, frequency=frequency)

    def test_lowest_frequency_block_is_selected_not_oldest(self):
        f = freq_from_ranges(self.LOGICAL, {(0, 1000): 5.0, (1000, 2000): 4.0, (2000, 3000): 0.1, (3000, 4000): 2.0})
        d = self.plan(f)
        self.assertEqual(selected(d), ["obs2"])
        self.assertEqual(d.receipt["effective_policy"], "qsa_frequency"); self.assertIsNone(d.receipt["fallback_reason"])
        self.assertEqual([b["observation_id"] for b in d.receipt["block_scores"]], ["obs0", "obs1", "obs2", "obs3"])
        chron = self.plan(None, policy="oldest_interactions")
        self.assertEqual(selected(chron), ["obs0"])                      # control unchanged

    def test_ties_resolve_oldest_first(self):
        d = self.plan(freq_from_ranges(self.LOGICAL, {}, default=1.0))
        self.assertEqual(selected(d), ["obs0"])

    def test_selection_continues_until_target_in_score_order(self):
        f = freq_from_ranges(self.LOGICAL, {(0, 1000): 5.0, (1000, 2000): 0.2, (2000, 3000): 0.1, (3000, 4000): 4.0})
        d = self.plan(f, fraction=0.35)                                   # target 1750 -> two blocks
        self.assertEqual(selected(d), ["obs2", "obs1"])

    def test_recent_protection_shields_newest_whole_blocks(self):
        f = freq_from_ranges(self.LOGICAL, {(3000, 4000): 0.01})           # obs3 would be the pick
        d = self.plan(f, protect=0.4)                                     # 2000 protected: active(1000)+obs3(1000)
        self.assertNotIn("obs3", selected(d))
        self.assertEqual(d.receipt["recent_protected_block_ids"], ["obs3"])
        self.assertIsNone(d.receipt["fallback_reason"])

    def test_recent_protection_relaxes_rather_than_failing(self):
        f = freq_from_ranges(self.LOGICAL, {})
        d = self.plan(f, protect=0.95)                                    # shields everything eligible
        self.assertEqual(d.receipt["fallback_reason"], "recent_protection_relaxed")
        self.assertTrue(selected(d))

    def test_missing_snapshot_falls_back_chronologically_with_reason(self):
        d = self.plan(None)
        self.assertEqual(selected(d), ["obs0"])
        self.assertEqual(d.receipt["effective_policy"], "oldest_interactions")
        self.assertEqual(d.receipt["fallback_reason"], "no_frequency_snapshot")

    def test_low_coverage_falls_back_with_reason(self):
        f = freq_from_ranges(self.LOGICAL, {(2000, 3000): 0.1}, none_ranges=((0, 2000),))
        d = self.plan(f)
        self.assertEqual(d.receipt["fallback_reason"], "insufficient_exposure_coverage")
        self.assertEqual(selected(d), ["obs0"])

    def test_oldest_interactions_ignores_frequency_kwarg(self):
        f = freq_from_ranges(self.LOGICAL, {(0, 1000): 5.0, (2000, 3000): 0.1})
        a = self.plan(None, policy="oldest_interactions"); b = self.plan(f, policy="oldest_interactions")
        self.assertEqual(a.eviction_plan, b.eviction_plan); self.assertNotIn("block_scores", b.receipt)

    def test_not_triggered_means_no_ranking_work(self):
        registry, ledger, config = setup(self.STARTS, self.LOGICAL, budget=6000)
        d = registry.plan_eviction(config, ledger, incoming_tokens=1, frequency=freq_from_ranges(self.LOGICAL, {}))
        self.assertFalse(d.receipt["triggered"]); self.assertEqual(d.eviction_plan.removed_tokens, 0)
        self.assertNotIn("block_scores", d.receipt)


if __name__ == "__main__":
    unittest.main(verbosity=1)
