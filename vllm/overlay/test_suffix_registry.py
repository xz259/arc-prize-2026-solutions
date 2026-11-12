import unittest
from residency import ResidencyError
from suffix_registry import SuffixRegistry


def record(oid, start=10, end=30):
    return dict(observation_id=oid, local_token_range=[start, end])


class RegistryTests(unittest.TestCase):
    def test_first_observation_inspection_then_replacement(self):
        state = SuffixRegistry()
        initial = state.plan([record("obs1")], [], logical_offset=0, delta_tokens=50)
        self.assertEqual(state.live, {})
        state.commit(initial)
        inspection = state.plan([], [], logical_offset=100, delta_tokens=5)
        self.assertEqual(state.eviction_ranges(inspection), [])
        state.commit(inspection)
        later = state.plan([record("obs2", 20, 40)], ["obs1"], logical_offset=150, delta_tokens=70)
        self.assertEqual(state.eviction_ranges(later), [[10, 30]])
        self.assertEqual(later.acknowledgement(), dict(registered_observation_ids=["obs2"], evicted_observation_ids=["obs1"]))
        state.commit(later)
        self.assertEqual(state.live, {"obs2": (170, 190)})

    def test_new_observation_without_suffix_retires_previous(self):
        state = SuffixRegistry()
        state.commit(state.plan([record("a")], [], logical_offset=0, delta_tokens=40))
        state.commit(state.plan([dict(observation_id="b", local_token_range=None)], ["a"], logical_offset=40, delta_tokens=10))
        self.assertEqual(state.live, {})
        state.commit(state.plan([record("c")], [], logical_offset=50, delta_tokens=40))
        self.assertEqual(state.live, {"c": (60, 80)})

    def test_arbitrary_old_or_missing_owner_cannot_be_evicted(self):
        state = SuffixRegistry()
        state.commit(state.plan([record("a")], [], logical_offset=0, delta_tokens=40))
        for records, retire in [([], ["a"]), ([record("b")], []),
                                ([record("b")], ["wrong"]), ([record("a")], ["a"])]:
            with self.assertRaises(ResidencyError):
                state.plan(records, retire, logical_offset=50, delta_tokens=50)
        self.assertEqual(state.live, {"a": (10, 30)})

    def test_token_span_cannot_reach_outside_new_input(self):
        for start, end in [(-1, 4), (0, 51), (4, 4), (True, 9)]:
            with self.assertRaises(ResidencyError):
                SuffixRegistry().plan([record("a", start, end)], [], logical_offset=20, delta_tokens=50)

    def test_stale_transition_cannot_commit_twice(self):
        state = SuffixRegistry()
        plan = state.plan([record("a")], [], logical_offset=0, delta_tokens=50)
        state.commit(plan)
        with self.assertRaises(ResidencyError):
            state.commit(plan)


if __name__ == "__main__":
    unittest.main()
