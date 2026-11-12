"""CPU ownership/transaction tests; no serving, GPU or recorded code execution."""
import copy
import hashlib
import json
import unittest

from append_only_history import AppendOnlyViolation, observation_base
from managed_suffix_history import (ManagedSuffixMessageState, canonical_json,
                                    describe_observation)
from prolong_program_suffix import BASE_KEY as PROGRAM_BASE
from prolong_snapshots import BASE_KEY as SNAPSHOT_BASE
from test_append_only_history import observation, workspace


class ManagedSuffixTests(unittest.TestCase):
    def setUp(self):
        self.ws = workspace()
        self.state = ManagedSuffixMessageState()
        self.initial = [dict(role="system", content="Solve the game."), observation(self.ws)]

    def stage(self, messages=None, request_id="r1", **kwargs):
        return self.state.stage(self.initial if messages is None else messages,
                request_id=request_id, workspace=self.ws,
                renderer_fingerprint=kwargs.get("renderer_fingerprint", "renderer-v1"))

    def receipt(self, prepared):
        sidecar = prepared.sidecar()
        return dict(registered_observation_ids=[d["observation_id"] for d in sidecar["observations"]],
                    evicted_observation_ids=sidecar["evict_observation_ids"])

    def commit(self, prepared, **kwargs):
        self.state.commit(prepared, accepted_input_tokens=kwargs.get("tokens", 500),
                          managed_suffixes=kwargs.get("receipt", self.receipt(prepared)))

    def first(self):
        prepared = self.stage()
        self.commit(prepared)
        return prepared

    def test_initial_exact_content_parts_and_digest(self):
        prepared = self.stage()
        self.assertEqual(prepared.sidecar()["evict_observation_ids"], [])
        descriptor = prepared.sidecar()["observations"][0]
        message = prepared.messages()[1]
        self.assertEqual(descriptor["message_index"], 1)
        self.assertEqual(descriptor["base_parts_end"], 2)
        self.assertEqual(descriptor["program_parts"], [2, 3])
        self.assertEqual(descriptor["snapshot_parts"], [3, len(message["content"])])
        self.assertEqual(descriptor["content_sha256"], hashlib.sha256(canonical_json(message["content"]).encode()).hexdigest())

    def test_no_control_fields_or_private_metadata_in_wire_messages(self):
        prepared = self.stage()
        wire = json.dumps(prepared.external_messages())
        for forbidden in ("_arc3_", "observation_id", "managed_suffixes", "base_parts_end"):
            self.assertNotIn(forbidden, wire)
        self.assertIn("PROGRAMS", wire)

    def test_snapshot_and_program_can_be_disabled_independently(self):
        for program, snapshot in ((True, False), (False, True), (False, False)):
            state = ManagedSuffixMessageState(program_enabled=program, snapshot_enabled=snapshot)
            prepared = state.stage(self.initial, request_id="r1", workspace=self.ws,
                                   renderer_fingerprint="x")
            descriptor = prepared.sidecar()["observations"][0]
            self.assertEqual(descriptor["program_parts"] is not None, program)
            self.assertEqual(descriptor["snapshot_parts"] is not None, snapshot)
            if snapshot:
                self.assertEqual(descriptor["snapshot_parts"][0], 2)

    def test_string_observation_without_suffix_keeps_wire_string(self):
        self.state = ManagedSuffixMessageState(program_enabled=False, snapshot_enabled=False)
        self.initial[-1]["content"] = "Original observation."
        prepared = self.stage()
        self.assertEqual(prepared.external_messages()[-1]["content"], "Original observation.")
        self.assertEqual(prepared.sidecar()["observations"][0]["base_parts_end"], 1)

    def test_real_new_observation_retires_previous_owner_only(self):
        first = self.first()
        self.ws.program_index.value = "PROGRAMS: updated"
        second = self.stage([*first.messages(), dict(role="tool", tool_call_id="t", content="ok"),
                             observation(self.ws, identity="state-2")], "r2")
        self.assertEqual(second.sidecar()["evict_observation_ids"], ["obs-00000001"])
        self.assertEqual(second.sidecar()["observations"][0]["observation_id"], "obs-00000002")
        self.assertEqual(second.sidecar()["observations"][0]["message_index"], 1)
        self.assertEqual(second.messages()[:2], first.messages())
        self.assertIn("updated", json.dumps(second.messages()[-1]["content"]))
        self.assertEqual(self.state.live_observation_ids(), ["obs-00000001"])
        self.commit(second)
        self.assertEqual(self.state.live_observation_ids(), ["obs-00000002"])
        third = self.stage([*second.messages(), observation(self.ws, identity="state-3")], "r3")
        self.assertEqual(third.sidecar()["evict_observation_ids"], ["obs-00000002"])

    def test_inspection_defers_library_and_preserves_suffixes(self):
        first = self.first()
        self.ws.program_index.value = "PROGRAMS: inspection_helper"
        second = self.stage([*first.messages(), dict(role="tool", tool_call_id="t", content="computed")], "r2")
        self.assertEqual(second.sidecar(), dict(version=1, observations=[], evict_observation_ids=[]))
        self.assertEqual(second.messages()[:2], first.messages())
        self.assertNotIn("inspection_helper", json.dumps(second.messages()))
        self.commit(second)
        self.assertEqual(self.state.live_observation_ids(), ["obs-00000001"])

    def test_empty_input_continuation_has_no_suffix_change(self):
        first = self.first()
        second = self.stage(first.messages(), "r2")
        self.assertEqual(second.external_messages(), [])
        self.assertEqual(second.sidecar(), dict(version=1, observations=[], evict_observation_ids=[]))

    def test_non_observation_user_does_not_trigger_eviction(self):
        first = self.first()
        second = self.stage([*first.messages(), dict(role="user", content="Tool recovery notice")], "r2")
        self.assertEqual(second.sidecar()["evict_observation_ids"], [])
        self.assertEqual(second.sidecar()["observations"], [])

    def test_duplicate_board_state_gets_distinct_observation_identity(self):
        first = self.first()
        second = self.stage([*first.messages(), self.initial[-1]], "r2")
        self.assertEqual(second.sidecar()["observations"][0]["observation_id"], "obs-00000002")
        self.assertEqual(second.sidecar()["evict_observation_ids"], ["obs-00000001"])

    def test_pending_retry_is_immutable_after_workspace_updates(self):
        first = self.stage()
        expected = first.sidecar(), first.external_messages()
        self.ws.program_index.value = "PROGRAMS: changed after submit"
        retry = self.stage()
        self.assertIs(retry, first)
        self.assertEqual((retry.sidecar(), retry.external_messages()), expected)
        self.assertIs(self.stage(first.messages()), first)

    def test_retry_rejects_changed_original_content_and_renderer(self):
        self.stage()
        changed = copy.deepcopy(self.initial)
        changed[-1]["content"][0]["text"] = "Different board"
        with self.assertRaises(AppendOnlyViolation):
            self.stage(changed)
        with self.assertRaises(AppendOnlyViolation):
            self.stage(renderer_fingerprint="changed")

    def test_different_pending_request_is_refused(self):
        self.stage()
        with self.assertRaises(AppendOnlyViolation):
            self.stage(request_id="r2")

    def test_rejection_keeps_old_owner_and_requires_new_request_id(self):
        first = self.first()
        second = self.stage([*first.messages(), observation(self.ws, identity="two")], "r2")
        self.state.discard_rejected(second, server_rejected=True)
        self.assertEqual(self.state.live_observation_ids(), ["obs-00000001"])
        self.assertEqual(self.state.committed_messages(), first.messages())
        with self.assertRaises(AppendOnlyViolation):
            self.commit(second)
        with self.assertRaises(AppendOnlyViolation):
            self.stage([*first.messages(), observation(self.ws, identity="two")], "r2")
        third = self.stage([*first.messages(), observation(self.ws, identity="two")], "r3")
        self.assertEqual(third.sidecar()["observations"][0]["observation_id"], "obs-00000002")

    def test_uncertain_transport_cannot_discard_pending(self):
        pending = self.stage()
        with self.assertRaises(AppendOnlyViolation):
            self.state.discard_rejected(pending, server_rejected=False)
        self.assertIs(self.stage(), pending)

    def test_commit_validates_exact_ownership_acknowledgement(self):
        first = self.stage()
        for receipt in (None, {}, dict(registered_observation_ids=[], evicted_observation_ids=[]),
                        dict(registered_observation_ids=["obs-00000001"], evicted_observation_ids=["unknown"])):
            with self.assertRaises(AppendOnlyViolation):
                self.commit(first, receipt=receipt)
            self.assertEqual(self.state.committed_messages(), [])
            self.assertEqual(self.state.live_observation_ids(), [])
        self.commit(first)

    def test_double_commit_same_receipt_is_idempotent(self):
        first = self.first()
        self.commit(first)
        with self.assertRaises(AppendOnlyViolation):
            self.commit(first, tokens=501)
        second = self.stage([*first.messages(), observation(self.ws, identity="two")], "r2")
        self.assertEqual(second.sidecar()["observations"][0]["observation_id"], "obs-00000002")

    def test_last_committed_retry_is_identical_until_next_operation(self):
        first = self.first()
        self.assertIs(self.stage(first.messages()), first)
        second = self.stage([*first.messages(), dict(role="tool", content="x")], "r2")
        self.commit(second)
        with self.assertRaises(AppendOnlyViolation):
            self.stage(first.messages())

    def test_empty_new_suffix_still_retires_previous_suffix(self):
        self.state = ManagedSuffixMessageState(snapshot_enabled=False)
        first = self.first()
        self.ws.program_index.value = ""
        second = self.stage([*first.messages(), observation(self.ws, identity="two")], "r2")
        self.assertEqual(second.sidecar()["evict_observation_ids"], ["obs-00000001"])
        self.assertEqual(second.sidecar()["observations"][0]["program_parts"], None)
        self.assertEqual(self.receipt(second)["registered_observation_ids"], ["obs-00000002"])
        self.commit(second)
        self.assertEqual(self.state.live_observation_ids(), [])

    def test_old_suffix_rewrap_restores_submitted_copy(self):
        first = self.first()
        raw = [first.messages()[0], observation_base(first.messages()[1]),
               dict(role="tool", content="inspection")]
        later = self.stage(raw, "r2")
        self.assertEqual(later.messages()[1], first.messages()[1])
        self.assertEqual(later.sidecar()["evict_observation_ids"], [])

    def test_changed_old_original_or_system_is_rejected(self):
        first = self.first()
        for index in (0, 1):
            changed = first.messages()
            changed[index]["content"] = "rewritten"
            with self.assertRaises(AppendOnlyViolation):
                self.stage(changed, "r2")

    def test_bad_private_base_is_not_used_as_eviction_boundary(self):
        first = self.stage()
        for key in (PROGRAM_BASE, SNAPSHOT_BASE):
            bad = first.messages()[1]
            bad[key] = "forged prefix"
            with self.assertRaises(AppendOnlyViolation):
                describe_observation(bad, message_index=1, observation_id="obs")

    def test_no_shared_mutable_sidecar_message_state(self):
        first = self.stage()
        first.sidecar()["observations"][0]["base_parts_end"] = 1000
        first.external_messages()[-1]["content"] = "mutated"
        self.assertEqual(first.sidecar()["observations"][0]["base_parts_end"], 2)
        self.assertNotEqual(first.external_messages()[-1]["content"], "mutated")

    def test_multiple_unseen_observations_are_refused(self):
        with self.assertRaises(AppendOnlyViolation):
            self.stage([*self.initial, observation(self.ws, identity="two")])

    def test_assistant_input_is_not_external_delta(self):
        with self.assertRaises(AppendOnlyViolation):
            self.stage([*self.initial, dict(role="assistant", content="parsed view")])

    def test_cross_session_commit_refused(self):
        first = self.stage()
        other = ManagedSuffixMessageState()
        with self.assertRaises(AppendOnlyViolation):
            other.commit(first, accepted_input_tokens=500, managed_suffixes=self.receipt(first))

    def test_authoritative_token_count_required(self):
        first = self.stage()
        for value in (True, -1, None, "500"):
            with self.assertRaises(AppendOnlyViolation):
                self.commit(first, tokens=value)


if __name__ == "__main__":
    unittest.main()
