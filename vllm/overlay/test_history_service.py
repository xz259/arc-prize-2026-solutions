"""Whole-interaction history planning at the actual session transaction boundary."""
import unittest
from copy import deepcopy

from session_service import SessionError, SessionManager
from test_suffix_service import Backend as SuffixBackend


class Backend(SuffixBackend):
    async def prepare(self, session, payload, *, initial):
        turn = await super().prepare(session, payload, initial=initial)
        turn.observation_starts = deepcopy(payload.get("starts", []))
        return turn


class HistorySessions(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.backend = Backend()
        self.manager = SessionManager(self.backend, max_context=80, max_sessions=2)

    async def asyncTearDown(self):
        await self.manager.close_all()

    def observation(self, number, **updates):
        record = {"observation_id": f"obs{number}", "local_token_range": [10, 12]}
        payload = {"ids": list(range(20)), "records": [record],
                   "starts": [{"observation_id": f"obs{number}",
                               "local_start_token": 4 if number == 1 else 2,
                               "proof": {"method": "synthetic_exact_tokens"}}]}
        if number == 1:
            payload.update(session_id="a", request_id="start", history_eviction={
                "policy": "oldest_interactions", "fraction": .10})
        else:
            payload.update(request_id=f"obs{number}", expected_revision=number-1,
                           retire=[f"obs{number-1}"])
        return {**payload, **updates}

    async def fill(self):
        await self.manager.start(self.observation(1))
        await self.manager.append("a", self.observation(2))
        await self.manager.append("a", self.observation(3))
        await self.manager.append("a", {"request_id": "tool", "expected_revision": 3,
                                         "ids": list(range(20))})
        self.assertEqual(self.manager.get("a").residency.resident_tokens, 80)

    def snapshot(self):
        s = self.manager.get("a")
        return (list(s.token_ids), s.residency.wire() if hasattr(s.residency, "wire") else
                (s.residency.logical_tokens, s.residency.evicted),
                deepcopy(s.suffix_registry.live), s.history_registry.snapshot(), s.revision)

    async def test_empty_raw_continuation_evicts_oldest_complete_block_without_refill(self):
        await self.fill()
        before = list(self.manager.get("a").token_ids)
        reply = await self.manager.append("a", {"request_id": "continue", "expected_revision": 4,
                                                "ids": [], "continuation": True})
        p = reply["persistent"]
        self.assertEqual(p["history_eviction"]["selected_block_ids"], ["obs1"])
        self.assertEqual(p["history_eviction"]["target_tokens"], 8)
        self.assertEqual(p["history_eviction"]["history_removed_tokens"], 17)
        self.assertEqual(p["resident_tokens"], 64)
        self.assertEqual(self.manager.get("a").token_ids, before+[101])
        self.assertEqual(self.backend.submissions[-1][1], [])
        self.assertEqual(self.backend.submissions[-1][2]["max_tokens"], 17)
        self.assertEqual(self.backend.evictions[-1].ranges, ((4,10),(12,23)))
        self.assertEqual(p["history_state"]["active_observation_id"], "obs3")

    async def test_combined_suffix_and_history_skips_holes_and_rounds_up_for_append(self):
        await self.fill()
        reply = await self.manager.append("a", self.observation(4, expected_revision=4))
        h = reply["persistent"]["history_eviction"]
        self.assertEqual(h["suffix_removed_tokens"], 2)
        self.assertEqual(h["resident_tokens_after_suffix"], 78)
        self.assertEqual(h["required_tokens"], 19)
        self.assertEqual(h["selected_block_ids"], ["obs1", "obs2"])
        self.assertEqual(h["history_removed_tokens"], 36)
        self.assertEqual(reply["persistent"]["history_state"]["active_observation_id"], "obs4")
        self.assertEqual(reply["persistent"]["resident_tokens"], 63)
        self.assertEqual(self.backend.submissions[-1][1], list(range(20)))

    async def test_no_early_eviction_or_4096_response_cap(self):
        reply = await self.manager.start(self.observation(1))
        self.assertFalse(reply["persistent"]["history_eviction"]["triggered"])
        self.assertEqual(self.backend.submissions[0][2]["max_tokens"], 60)
        self.assertEqual(self.backend.evictions, [])

    async def test_current_block_cannot_be_closed_by_unsubmitted_observation(self):
        await self.manager.start(self.observation(1, history_eviction={
            "policy": "oldest_interactions", "resident_token_budget": 30}))
        before = self.snapshot()
        with self.assertRaises(SessionError) as error:
            await self.manager.append("a", self.observation(2))
        self.assertEqual(error.exception.status, 413)
        self.assertEqual(error.exception.receipt["eligible_history_tokens"], 0)
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(self.manager.get("a").state, "paused")
        self.assertEqual(self.backend.evictions, [])

    async def test_preflight_failure_cannot_commit_either_registry(self):
        await self.fill()
        before = self.snapshot()
        self.backend.preflight_error = True
        n = len(self.backend.evictions)
        with self.assertRaisesRegex(SessionError, "before mutation"):
            await self.manager.append("a", self.observation(4, expected_revision=4))
        self.assertEqual(self.snapshot(), before)
        self.assertEqual(len(self.backend.evictions), n)

    async def test_bad_copy_ack_poisoned_without_registering_new_observation(self):
        await self.fill()
        before = self.manager.get("a").history_registry.snapshot()
        self.backend.bad_ack = True
        with self.assertRaisesRegex(RuntimeError, "acknowledgement"):
            await self.manager.append("a", self.observation(4, expected_revision=4))
        self.assertEqual(self.manager.get("a").history_registry.snapshot(), before)
        self.assertEqual(self.manager.get("a").state, "error")
        self.assertIn("a", self.backend.closed)

    async def test_policy_is_start_only_and_exact_retry_is_idempotent(self):
        body = self.observation(1)
        first = await self.manager.start(body)
        self.assertEqual(await self.manager.start(body), first)
        with self.assertRaisesRegex(SessionError, "immutable"):
            await self.manager.append("a", {"request_id": "bad", "expected_revision": 1,
                                             "ids": [2], "history_eviction": {"policy": "off"}})
        self.assertEqual(len(self.backend.submissions), 1)

    async def test_invalid_policy_rejected_before_admission(self):
        with self.assertRaises(SessionError) as error:
            await self.manager.start(self.observation(1, history_eviction={"fraction": 0}))
        self.assertEqual(error.exception.status, 400)
        self.assertEqual(self.manager.sessions, {})

    async def test_missing_or_mismatched_boundary_rejected_before_submission(self):
        for starts in ([], [{"observation_id": "other", "local_start_token": 4}]):
            manager = SessionManager(self.backend, max_context=80, max_sessions=2)
            with self.assertRaisesRegex(SessionError, "observation starts"):
                await manager.start(self.observation(1, starts=starts))
        self.assertEqual(self.backend.submissions, [])


if __name__ == "__main__":
    unittest.main()
