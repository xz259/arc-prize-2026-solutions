import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import scalar_publication as pub


class Tensor:
    def __init__(self, rows, dtype, device, journal):
        self.values = [0]*rows
        self.dtype, self.device, self.shape = dtype, device, (rows,)
        self.journal, self.fail = journal, False
    def data_ptr(self):
        return id(self)
    def stride(self):
        return (1,)
    def __getitem__(self, index):
        raise AssertionError("Helper must never read scalar data from GPU")
    def __setitem__(self, index, value):
        self.journal.append((self, index, value))
        self.values[index] = value
        if self.fail:
            raise RuntimeError("Native scalar submission failed after a possible write")


class Torch:
    int64 = "int64"
    def __init__(self):
        self.journal, self.allocations = [], []
    def zeros(self, rows, *, dtype, device):
        tensor = Tensor(rows, dtype, device, self.journal)
        self.allocations.append(tensor)
        return tensor


def make(rows=3):
    torch = Torch()
    buffers = NS(**pub.allocate_zero_fields(torch, rows, device="cuda:0"))
    return torch, buffers


def publish(buffers, slot, values):
    with pub.publish_scalars(buffers, slot, *values):
        pass


class ScalarTests(unittest.TestCase):
    def setUp(self):
        env = patch.dict(os.environ, {pub.FLAG: "1"})
        env.start()
        self.addCleanup(env.stop)
        self.torch, self.buffers = make()

    def fields(self):
        return tuple(getattr(self.buffers, name).values for name in pub.FIELDS)

    def test_factory_owns_exact_zero_allocation_and_stable_layout(self):
        self.assertEqual(len(self.torch.allocations), 3)
        for name, allocated in zip(pub.FIELDS, self.torch.allocations):
            self.assertIs(getattr(self.buffers, name), allocated)
            self.assertEqual(allocated.dtype, "int64")
            self.assertEqual(allocated.stride(), (1,))
        refs = tuple(getattr(self.buffers, name) for name in pub.FIELDS)
        publish(self.buffers, 0, (8, 0, 40000))
        self.assertEqual(refs, tuple(getattr(self.buffers, name) for name in pub.FIELDS))

    def test_identity_admission_and_readd_do_not_write(self):
        publish(self.buffers, 1, (0, 0, 0))
        publish(self.buffers, 1, (0, 0, 0))
        self.assertEqual(self.torch.journal, [])
        self.assertEqual(pub.status(self.buffers)["counts"]["skipped_writes"], 6)

    def test_first_deletion_repeat_and_cutoff_growth(self):
        publish(self.buffers, 0, (6, 2, 38000))
        self.assertEqual(len(self.torch.journal), 3)
        publish(self.buffers, 0, (6, 2, 38000))
        self.assertEqual(len(self.torch.journal), 3)
        publish(self.buffers, 0, (6, 2, 39000))
        self.assertEqual(len(self.torch.journal), 4)
        self.assertIs(self.torch.journal[-1][0], self.buffers.cutoffs)

    def test_repeated_holes_change_required_scalars(self):
        for values in ((3, 3, 39003), (9, 1, 38001), (13, 1, 37001)):
            publish(self.buffers, 2, values)
        self.assertEqual([values[2] for values in self.fields()], [13, 1, 37001])
        self.assertEqual(len(self.torch.journal), 8)

    def test_slot_reuse_by_new_persistent_or_stateless_owner(self):
        publish(self.buffers, 1, (5, 1, 40001))
        # Logical owner removal does not change the physical slot shadow.
        publish(self.buffers, 1, (0, 0, 0))
        self.assertEqual(len(self.torch.journal), 6)
        publish(self.buffers, 1, (0, 0, 0))
        self.assertEqual(len(self.torch.journal), 6)
        self.assertEqual([values[1] for values in self.fields()], [0, 0, 0])

    def test_partial_scalar_failure_forces_complete_republication(self):
        self.buffers.padding.fail = True
        with self.assertRaisesRegex(RuntimeError, "Native scalar"):
            publish(self.buffers, 0, (7, 3, 40003))
        self.assertEqual(len(self.torch.journal), 2)
        self.assertIsNone(getattr(self.buffers, pub.STATE_ATTR).slots[0])
        self.buffers.padding.fail = False
        publish(self.buffers, 0, (7, 3, 40003))
        self.assertEqual(len(self.torch.journal), 5)
        self.assertEqual(pub.status(self.buffers)["counts"]["unknown_republications"], 1)

    def test_lookup_failure_also_leaves_shadow_unknown(self):
        failure = RuntimeError("lookup copy failed")
        with self.assertRaises(RuntimeError) as caught:
            with pub.publish_scalars(self.buffers, 0, 8, 0, 39999):
                raise failure
        self.assertIs(caught.exception, failure)
        self.assertIsNone(getattr(self.buffers, pub.STATE_ATTR).slots[0])
        before = len(self.torch.journal)
        publish(self.buffers, 0, (8, 0, 39999))
        self.assertEqual(len(self.torch.journal)-before, 3)

    def test_failed_body_after_skips_forces_zero_republication(self):
        with self.assertRaises(ValueError):
            with pub.publish_scalars(self.buffers, 0, 0, 0, 0):
                raise ValueError("body")
        self.assertEqual(len(self.torch.journal), 0)
        publish(self.buffers, 0, (0, 0, 0))
        self.assertEqual(len(self.torch.journal), 3)

    def test_explicit_external_write_invalidation(self):
        pub.invalidate(self.buffers, 1)
        self.buffers.deleted[1] = 10
        self.torch.journal.clear()
        publish(self.buffers, 1, (0, 0, 0))
        self.assertEqual(len(self.torch.journal), 3)
        pub.invalidate(self.buffers)
        self.assertEqual(pub.status(self.buffers)["known_slots"], 0)

    def test_replacing_tensor_is_rejected_before_any_write(self):
        self.buffers.cutoffs = self.torch.zeros(3, dtype="int64", device="cuda:0")
        with self.assertRaisesRegex(pub.ScalarPublicationError, "identity/layout"):
            publish(self.buffers, 0, (4, 0, 40000))
        self.assertEqual(self.torch.journal, [])

    def test_shadow_cannot_be_seeded_from_arbitrary_live_tensors(self):
        buffers = NS(**{name: getattr(self.buffers, name) for name in pub.FIELDS})
        with self.assertRaisesRegex(pub.ScalarPublicationError, "allocate_zero_fields"):
            publish(buffers, 0, (0, 0, 0))

    def test_invalid_slot_values_and_nested_publication_fail_before_writes(self):
        for slot, values in ((3, (0, 0, 0)), (-1, (0, 0, 0)), (0, (0, -1, 1)), (0, (0, 0, True))):
            with self.assertRaises(pub.ScalarPublicationError):
                publish(self.buffers, slot, values)
        with self.assertRaisesRegex(pub.ScalarPublicationError, "Nested"):
            with pub.publish_scalars(self.buffers, 0, 0, 0, 0):
                publish(self.buffers, 0, (0, 0, 0))
        self.assertEqual(self.torch.journal, [])
        self.assertIsNone(getattr(self.buffers, pub.STATE_ATTR).slots[0])

    def test_control_mode_keeps_unconditional_writes_and_invalidates_old_shadow(self):
        with patch.dict(os.environ, {pub.FLAG: "0"}):
            torch, buffers = make()
            self.assertFalse(pub.status(buffers)["initialized"])
            publish(buffers, 0, (0, 0, 0))
            publish(buffers, 0, (0, 0, 0))
            self.assertEqual(len(torch.journal), 6)
            publish(self.buffers, 0, (0, 0, 0))
        before = len(self.torch.journal)
        publish(self.buffers, 0, (0, 0, 0))
        self.assertEqual(len(self.torch.journal)-before, 3)

    def test_randomized_sequences_match_unconditional_reference(self):
        with patch.dict(os.environ, {pub.FLAG: "0"}):
            reference_torch, reference = make()
        rng = random.Random(713)
        for _ in range(150):
            slot = rng.randrange(3)
            deleted = rng.randrange(15)
            values = (deleted, deleted % 4, rng.randrange(35000, 40000))
            if rng.randrange(4) == 0:
                values = (0, 0, 0)
            with patch.dict(os.environ, {pub.FLAG: "0"}):
                publish(reference, slot, values)
            publish(self.buffers, slot, values)
            self.assertEqual(self.fields(), tuple(getattr(reference, name).values for name in pub.FIELDS))
        self.assertLessEqual(len(self.torch.journal), len(reference_torch.journal))

    def test_invalid_flag_rejected(self):
        with patch.dict(os.environ, {pub.FLAG: "yes"}), self.assertRaises(pub.ScalarPublicationError):
            make()


if __name__ == "__main__":
    unittest.main()
