"""Exercise the real endpoint cancellation fence and fused ack boundary."""
import asyncio
import os
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from test_endpoint_transaction import EndpointTransactionTests


class FusedEndpointTests(EndpointTransactionTests):
    # Legacy tests run unmodified with the feature disabled. These added tests
    # target behavior spanning endpoint cancellation and core utility completion.
    async def test_fused_exact_ack_and_separate_profile(self):
        backend, core, session, plan, _ = self.setup_backend()
        ack = {"plan_digest": plan.digest, "unchanged_receipt": [1, 2]}
        calls = []
        async def utility(name, rid, payload):
            calls.append((name, rid, payload))
            return {"ack": ack, "control_profile": {"drain_seconds": 0.1}}
        core.call_utility_async = utility
        with patch.dict(os.environ, ARC3_EVICTION_FUSED_CONTROL="1", ARC3_EVICTION_PROFILE="1"):
            result = await backend.evict(session, plan)
        self.assertIs(result, ack)
        self.assertEqual(core.events, [])
        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][:2], ("arc3_evict_resident_transaction", "engine-s"))
        self.assertEqual(calls[0][2]["plan_digest"], plan.digest)
        self.assertEqual(session.eviction_endpoint_profile["core_control"], {"drain_seconds": 0.1})

    async def test_fused_repeated_cancellation_waits_for_core_cleanup(self):
        backend, core, session, plan, _ = self.setup_backend()
        entered, release = asyncio.Event(), asyncio.Event()
        completed = []
        async def utility(*args):
            entered.set()
            await release.wait()
            completed.append("resumed")
            return {"ack": {"plan_digest": plan.digest}, "control_profile": {}}
        core.call_utility_async = utility
        with patch.dict(os.environ, ARC3_EVICTION_FUSED_CONTROL="1"):
            task = asyncio.create_task(backend.evict(session, plan))
            await asyncio.wait_for(entered.wait(), 1)
            for _ in range(2):
                task.cancel()
                await self.ticks()
                self.assertFalse(task.done())
            release.set()
            with self.assertRaises(asyncio.CancelledError):
                await asyncio.wait_for(task, 1)
        self.assertEqual(completed, ["resumed"])
        self.assertEqual(core.events, [])

    async def test_fused_error_does_not_issue_unowned_resume(self):
        backend, core, session, plan, _ = self.setup_backend()
        async def utility(*args):
            raise RuntimeError("already paused outside transaction")
        core.call_utility_async = utility
        with patch.dict(os.environ, ARC3_EVICTION_FUSED_CONTROL="1"):
            with self.assertRaisesRegex(RuntimeError, "already paused"):
                await backend.evict(session, plan)
        self.assertEqual(core.events, [])


if __name__ == "__main__":
    unittest.main()
