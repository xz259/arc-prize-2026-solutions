import asyncio
import unittest

from session_service import SessionError, SessionManager, TurnInput
from test_session_service import FakeBackend


class Backend(FakeBackend):
    def __init__(self):
        super().__init__()
        self.evictions = []
        self.events = []
        self.bad_ack = False
        self.preflight_error = False

    async def prepare(self, session, payload, *, initial):
        turn = await super().prepare(session, payload, initial=initial)
        turn.suffix_records = payload.get("records", [])
        turn.evict_observation_ids = payload.get("retire", [])
        return turn

    async def preflight(self, session, turn, *, initial):
        self.events.append("preflight")
        if self.preflight_error:
            raise SessionError("Input rejected before mutation", 400)

    async def evict(self, session, plan):
        self.events.append("evict")
        self.evictions.append(plan)
        return {"plan_digest": "wrong" if self.bad_ack else plan.digest,
                "resident_tokens": plan.resident_after,
                "evicted_tokens": plan.removed_tokens,
                "logical_tokens": plan.logical_tokens,
                "eviction_revision": session.eviction_revision+1}

    async def turn(self, session, turn, *, initial):
        self.events.append("turn")
        return await super().turn(session, turn, initial=initial)


class SuffixSessions(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.backend = Backend()
        self.manager = SessionManager(self.backend, max_context=20, max_sessions=2)
        await self.manager.start({"session_id": "a", "request_id": "start",
                                  "ids": list(range(16)),
                                  "records": [{"observation_id": "obs1", "local_token_range": [4, 10]}]})

    async def asyncTearDown(self):
        await self.manager.close_all()

    def replacement(self, **changes):
        return {"request_id": "replace", "expected_revision": 1,
                "ids": list(range(20, 28)),
                "records": [{"observation_id": "obs2", "local_token_range": [2, 4]}],
                "retire": ["obs1"], **changes}

    async def test_resident_budget_allows_logical_history_larger_than_context(self):
        result = await self.manager.append("a", self.replacement())
        self.assertEqual(result["persistent"]["resident_tokens"], 20)
        self.assertEqual(result["persistent"]["logical_tokens"], 26)
        self.assertEqual(result["usage"]["prompt_tokens"], 19)
        self.assertEqual(result["persistent"]["evicted_tokens"], 6)
        self.assertEqual(self.backend.events, ["preflight", "turn", "preflight", "evict", "turn"])
        self.assertEqual(self.backend.submissions[-1][1], list(range(20, 28)))
        session = self.manager.get("a")
        self.assertEqual(session.token_ids, [*range(16), 101, *range(20, 28), 101])
        self.assertEqual(session.residency.resident_positions(), [*range(4), *range(10, 26)])
        self.assertEqual(session.suffix_registry.live, {"obs2": (19, 21)})

    async def test_retry_does_not_evict_or_generate_twice(self):
        payload = self.replacement()
        result = await self.manager.append("a", payload)
        self.assertEqual(await self.manager.append("a", payload), result)
        self.assertEqual(len(self.backend.evictions), 1)
        self.assertEqual(len(self.backend.submissions), 2)

    async def test_inspection_does_not_refresh_or_evict_suffix(self):
        result = await self.manager.append("a", {"request_id": "inspect", "expected_revision": 1, "ids": [22]})
        self.assertEqual(self.backend.evictions, [])
        self.assertEqual(result["persistent"]["managed_suffixes"],
                         {"registered_observation_ids": [], "evicted_observation_ids": []})
        self.assertEqual(self.manager.get("a").suffix_registry.live, {"obs1": (4, 10)})

    async def test_insufficient_capacity_does_not_evict_first_and_fail_later(self):
        with self.assertRaisesRegex(SessionError, "capacity"):
            await self.manager.append("a", self.replacement(ids=list(range(15))))
        self.assertEqual(self.backend.evictions, [])
        self.assertEqual(self.manager.get("a").residency.evicted, ())
        self.assertEqual(self.manager.get("a").state, "paused")

    async def test_renderer_or_engine_input_preflight_failure_preserves_cache(self):
        self.backend.preflight_error = True
        with self.assertRaisesRegex(SessionError, "before mutation"):
            await self.manager.append("a", self.replacement())
        self.assertEqual(self.backend.evictions, [])
        self.assertEqual(self.manager.get("a").suffix_registry.live, {"obs1": (4, 10)})

    async def test_uncertain_engine_ack_aborts_owned_session_without_retry(self):
        self.backend.bad_ack = True
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "acknowledgement"):
                await self.manager.append("a", self.replacement())
        self.assertEqual(len(self.backend.evictions), 1)
        self.assertEqual(len(self.backend.submissions), 1)
        self.assertIn("a", self.backend.closed)
        self.assertEqual(self.manager.get("a").state, "error")

    async def test_new_observation_without_suffix_still_retires_old_one(self):
        result = await self.manager.append("a", self.replacement(
            records=[{"observation_id": "obs2", "local_token_range": None}]))
        self.assertEqual(result["persistent"]["managed_suffixes"],
                         {"registered_observation_ids": ["obs2"], "evicted_observation_ids": ["obs1"]})
        self.assertEqual(self.manager.get("a").suffix_registry.live, {})

    async def test_missing_retirement_is_rejected_before_mutation(self):
        with self.assertRaisesRegex(SessionError, "preceding"):
            await self.manager.append("a", self.replacement(retire=[]))
        self.assertEqual(self.backend.evictions, [])

    async def test_original_position_limit_rejects_before_cache_mutation(self):
        self.manager.max_logical_tokens = 25
        with self.assertRaisesRegex(SessionError, "logical position limit"):
            await self.manager.append("a", self.replacement())
        self.assertEqual(self.backend.evictions, [])
        self.assertEqual(self.manager.get("a").state, "paused")

    async def test_output_budget_also_respects_original_position_limit(self):
        self.manager.max_logical_tokens = 24
        await self.manager.append("a", self.replacement(ids=list(range(20, 26))))
        self.assertEqual(self.backend.submissions[-1][2]["max_tokens"], 1)


if __name__ == "__main__":
    unittest.main()
