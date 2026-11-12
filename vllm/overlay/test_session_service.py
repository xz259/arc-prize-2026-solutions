import asyncio
import unittest

from session_service import SessionError, SessionManager, TurnInput


class FakeBackend:
    def __init__(self):
        self.submissions = []
        self.closed = []
        self.gate = None
        self.fail = False

    async def prepare(self, session, payload, *, initial):
        return TurnInput({"type": "token", "prompt_token_ids": payload.get("ids", [])},
                         dict(payload.get("sampling", {})))

    async def turn(self, session, turn, *, initial):
        self.submissions.append((session.session_id, list(turn.token_ids), dict(turn.sampling)))
        if self.gate:
            await self.gate.wait()
        if self.fail:
            raise RuntimeError("Engine failed after submission")
        return {"token_ids": [101], "finish_reason": "length", "stop_reason": None}

    async def response(self, session, result):
        return {"choices": [{"message": {"role": "assistant", "content": "answer"}}]}

    async def close(self, session):
        self.closed.append(session.session_id)


class Sessions(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.backend = FakeBackend()
        self.manager = SessionManager(self.backend, max_context=20, max_sessions=2, turn_timeout=1)

    async def asyncTearDown(self):
        await self.manager.close_all()

    async def start(self, sid="game", **kw):
        return await self.manager.start({"session_id": sid, "request_id": "start", "ids": [1, 2], **kw})

    async def test_append_uses_exact_generated_tokens_and_only_new_input(self):
        first = await self.start()
        second = await self.manager.append("game", {"request_id": "next", "expected_revision": 1, "ids": [3]})
        self.assertEqual(self.manager.get("game").token_ids, [1, 2, 101, 3, 101])
        self.assertEqual([s[1] for s in self.backend.submissions], [[1, 2], [3]])
        self.assertEqual(second["usage"], {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5})
        self.assertEqual(first["persistent"]["resident_tokens"], 3)
        self.assertEqual([s[2]["max_tokens"] for s in self.backend.submissions], [18, 16])

    async def test_retry_replays_result_without_double_submission(self):
        payload = {"session_id": "game", "request_id": "start", "ids": [1, 2]}
        first = await self.manager.start(payload)
        self.assertEqual(await self.manager.start(payload), first)
        self.assertEqual(len(self.backend.submissions), 1)
        with self.assertRaisesRegex(SessionError, "different input"):
            await self.manager.start({**payload, "ids": [2, 1]})

    async def test_http_cancellation_does_not_cancel_engine_operation(self):
        self.backend.gate = asyncio.Event()
        payload = {"session_id": "game", "request_id": "start", "ids": [1]}
        caller = asyncio.create_task(self.manager.start(payload))
        while not self.backend.submissions:
            await asyncio.sleep(0)
        caller.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await caller
        self.backend.gate.set()
        result = await self.manager.start(payload)
        self.assertEqual(result["persistent"]["revision"], 1)
        self.assertEqual(len(self.backend.submissions), 1)

    async def test_inflight_append_rejected_and_close_cleans_up(self):
        self.backend.gate = asyncio.Event()
        caller = asyncio.create_task(self.start())
        while not self.backend.submissions:
            await asyncio.sleep(0)
        with self.assertRaisesRegex(SessionError, "in flight"):
            await self.manager.append("game", {"request_id": "bad", "expected_revision": 0, "ids": [3]})
        await self.manager.close("game")
        with self.assertRaises(asyncio.CancelledError):
            await caller
        self.assertEqual(self.manager.status("game")["state"], "closed")
        self.assertIn("game", self.backend.closed)

    async def test_cumulative_capacity_rejects_without_mutation(self):
        await self.start(ids=list(range(18)))
        before = list(self.manager.get("game").token_ids)
        with self.assertRaisesRegex(SessionError, "Context capacity"):
            await self.manager.append("game", {"request_id": "overflow", "expected_revision": 1, "ids": [4]})
        self.assertEqual(self.manager.get("game").token_ids, before)
        self.assertEqual(len(self.backend.submissions), 1)
        self.assertEqual(self.manager.get("game").state, "paused")

    async def test_noinput_continuation_preserves_prefix_and_cap(self):
        await self.start()
        await self.manager.append("game", {"request_id": "continue", "expected_revision": 1,
                                           "ids": [], "continuation": True})
        self.assertEqual(self.backend.submissions[-1][1], [])
        self.assertEqual(self.backend.submissions[-1][2]["max_tokens"], 17)

    async def test_empty_or_false_continuations_rejected(self):
        await self.start()
        for name, fields in [("empty", {}), ("nonempty", {"continuation": True, "ids": [3]})]:
            with self.assertRaises(SessionError):
                await self.manager.append("game", {"request_id": name, "expected_revision": 1, **fields})
        self.manager.get("game").finish_reason = "stop"
        with self.assertRaisesRegex(SessionError, "length pause"):
            await self.manager.append("game", {"request_id": "stop", "expected_revision": 1,
                                               "continuation": True})

    async def test_stale_revision_or_digest_does_not_submit(self):
        await self.start()
        for fields in ({"expected_revision": 0}, {"expected_revision": 1, "expected_prefix_sha256": "bad"}):
            with self.assertRaises(SessionError):
                await self.manager.append("game", {"request_id": "bad", "ids": [3], **fields})
        self.assertEqual(len(self.backend.submissions), 1)

    async def test_engine_failure_aborts_and_is_never_retried(self):
        self.backend.fail = True
        for _ in range(2):
            with self.assertRaisesRegex(RuntimeError, "Engine failed"):
                await self.start()
        self.assertEqual(len(self.backend.submissions), 1)
        self.assertEqual(self.manager.get("game").state, "error")
        self.assertIn("game", self.backend.closed)

    async def test_paused_sessions_count_toward_admission_and_close_releases(self):
        await self.start("a")
        await self.start("b")
        with self.assertRaisesRegex(SessionError, "admission"):
            await self.start("c")
        await self.manager.close("a")
        await self.start("c")

    async def test_diagnostic_max_tokens_is_clamped_to_real_capacity(self):
        await self.start(sampling={"max_tokens": 100})
        self.assertEqual(self.backend.submissions[-1][2]["max_tokens"], 18)

    async def test_timeout_closes_owned_engine_session(self):
        self.backend.gate = asyncio.Event()
        self.manager.turn_timeout = 0.01
        with self.assertRaises(asyncio.TimeoutError):
            await self.start()
        self.assertIn("game", self.backend.closed)
        self.assertEqual(self.manager.get("game").state, "error")


if __name__ == "__main__":
    unittest.main()
