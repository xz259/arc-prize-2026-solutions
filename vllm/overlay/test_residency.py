import dataclasses
import random
import unittest

from residency import ResidentLedger, ResidencyError


class ResidencyTests(unittest.TestCase):
    def test_arbitrary_ranges_preserve_absolute_identity(self):
        ledger = ResidentLedger(20)
        plan = ledger.plan([[3, 8], [10, 13]], protected=[[0, 3], [17, 20]])
        self.assertEqual(ledger.resident_tokens, 20)  # preparation is pure
        self.assertEqual(plan.resident_after, 12)
        ledger.commit(plan, acknowledged_digest=plan.digest)
        self.assertEqual(ledger.resident_positions(), [0, 1, 2, 8, 9, *range(13, 20)])
        ledger.append(3)
        self.assertEqual(ledger.logical_tokens, 23)
        self.assertEqual(ledger.resident_tokens, 15)
        self.assertEqual(ledger.resident_positions()[-3:], [20, 21, 22])

    def test_overlaps_and_protected_tokens_are_rejected_without_mutation(self):
        ledger = ResidentLedger(20, [[3, 8]])
        before = (ledger.logical_tokens, ledger.evicted)
        for ranges, protected in [([[2, 4]], []), ([[0, 2]], [[0, 3]]),
                                  ([[9, 12], [11, 14]], []), ([[19, 21]], [])]:
            with self.assertRaises(ResidencyError):
                ledger.plan(ranges, protected=protected)
            self.assertEqual((ledger.logical_tokens, ledger.evicted), before)

    def test_exact_engine_ack_and_current_revision_required(self):
        ledger = ResidentLedger(20)
        plan = ledger.plan([[3, 8]])
        with self.assertRaises(ResidencyError):
            ledger.commit(plan, acknowledged_digest="wrong")
        altered = dataclasses.replace(plan, resident_after=19)
        with self.assertRaises(ResidencyError):
            ledger.commit(altered, acknowledged_digest=altered.digest)
        ledger.append(1)
        with self.assertRaises(ResidencyError):
            ledger.commit(plan, acknowledged_digest=plan.digest)

    def test_adjacent_removals_coalesce_and_retry_is_not_a_second_eviction(self):
        ledger = ResidentLedger(20)
        plan = ledger.plan([[4, 6], [2, 4]])
        self.assertEqual(plan.ranges, ((2, 6),))
        ledger.commit(plan, acknowledged_digest=plan.digest)
        with self.assertRaises(ResidencyError):
            ledger.commit(plan, acknowledged_digest=plan.digest)
        second = ledger.plan([[6, 8]])
        ledger.commit(second, acknowledged_digest=second.digest)
        self.assertEqual(ledger.evicted, ((2, 8),))

    def test_random_eviction_and_append_matches_boolean_reference(self):
        rng = random.Random(927)
        ledger, kept = ResidentLedger(), []
        for _ in range(150):
            amount = rng.randrange(1, 15)
            ledger.append(amount)
            kept.extend([True] * amount)
            choices = [i for i, live in enumerate(kept) if live]
            selected = sorted(rng.sample(choices, min(len(choices), rng.randrange(1, 10))))
            plan = ledger.plan([[i, i+1] for i in selected])
            ledger.commit(plan, acknowledged_digest=plan.digest)
            for i in selected:
                kept[i] = False
            self.assertEqual(ledger.resident_positions(), [i for i, live in enumerate(kept) if live])
            self.assertEqual(ledger.resident_tokens, sum(kept))

    def test_invalid_scalar_types_are_not_silently_coerced(self):
        for value in [True, -1, 1.5, "3"]:
            with self.assertRaises(ResidencyError):
                ResidentLedger(value)
        ledger = ResidentLedger(5)
        with self.assertRaises(ResidencyError):
            ledger.plan([[False, 3]])


if __name__ == "__main__":
    unittest.main()
