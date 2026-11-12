"""CPU concurrency/cancellation tests for the real endpoint eviction fence."""
import asyncio
import importlib.util
import os
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from residency import ResidentLedger
from session_service import Session, SessionError, SessionManager, token_digest


def load_backend():
    spec = importlib.util.spec_from_file_location("arc3_endpoint_transaction_fixture",
                                                Path(__file__).with_name("persistent_endpoint.py"))
    module = importlib.util.module_from_spec(spec)
    # Backend tests need no HTTP framework. The imported Request name is only
    # used by route annotations; none of those routes are instantiated here.
    if importlib.util.find_spec("fastapi") is None:
        with patch.dict(sys.modules, {"fastapi": SimpleNamespace(Request=object)}):
            spec.loader.exec_module(module)
    else:
        spec.loader.exec_module(module)
    return module.VllmBackend


VllmBackend = load_backend()


class Core:
    def __init__(self):
        self.events = []
        self.paused = False
        self.pause_entered = asyncio.Event()
        self.pause_release = asyncio.Event()
        self.pause_release.set()
        self.rpc_entered = asyncio.Event()
        self.rpc_release = asyncio.Event()
        self.rpc_release.set()
        self.rpc_cancelled = False
        self.pause_error = None
        self.rpc_error = None
        self.resume_error = None

    async def is_scheduler_paused_async(self):
        self.events.append("status")
        return self.paused

    async def pause_scheduler_async(self, *, mode, clear_cache):
        if mode != "keep" or clear_cache is not False:
            raise AssertionError("Eviction must preserve the cache while draining")
        self.events.append("pause-enter")
        self.pause_entered.set()
        await self.pause_release.wait()
        self.paused = True
        if self.pause_error:
            raise self.pause_error
        self.events.append("pause-drained")

    async def call_utility_async(self, name, request_id, payload):
        if not self.paused or "pause-drained" not in self.events:
            raise AssertionError("RPC before the worker drain")
        if name != "arc3_evict_resident":
            raise AssertionError("Wrong utility")
        self.events.append(("rpc-enter", request_id, payload))
        self.rpc_entered.set()
        try:
            await self.rpc_release.wait()
        except asyncio.CancelledError:
            self.rpc_cancelled = True
            self.events.append("rpc-cancelled")
            raise
        if self.rpc_error:
            raise self.rpc_error
        self.events.append(("rpc-completed", request_id))
        return {"plan_digest": payload["plan_digest"]}

    async def resume_scheduler_async(self):
        self.events.append("resume-enter")
        if self.resume_error:
            raise self.resume_error
        self.paused = False
        self.events.append("resumed")


class EndpointTransactionTests(unittest.IsolatedAsyncioTestCase):
    def setup_backend(self):
        core = Core()
        aborted = []
        async def abort(request):
            aborted.append(request)
            core.events.append(("abort", request))
        backend = VllmBackend.__new__(VllmBackend)
        backend.engine = SimpleNamespace(engine_core=core, abort=abort)
        backend.eviction_lock = asyncio.Lock()
        session = Session("s", "engine-s", {})
        session.token_ids = list(range(24))
        session.current_operation = "operation-2"
        session.state = "paused"
        session.residency = ResidentLedger(24)
        plan = session.residency.plan([[4, 8]], protected=[[20,24]])
        return backend, core, session, plan, aborted

    async def ticks(self, count=5):
        for _ in range(count):
            await asyncio.sleep(0)

    async def test_pause_drain_rpc_resume_order_and_exact_plan(self):
        backend, core, session, plan, _ = self.setup_backend()
        result = await backend.evict(session, plan)
        self.assertEqual(core.events[:3], ["status", "pause-enter", "pause-drained"])
        rpc = core.events[3]
        self.assertEqual(rpc[:2], ("rpc-enter", "engine-s"))
        self.assertEqual(rpc[2]["logical_prefix_sha256"], token_digest(session.token_ids))
        self.assertEqual(rpc[2]["logical_ranges"], [[4,8]])
        self.assertEqual(rpc[2]["plan_digest"], plan.digest)
        self.assertEqual(core.events[4:], [("rpc-completed", "engine-s"), "resume-enter", "resumed"])
        self.assertEqual(result, {"plan_digest": plan.digest})
        self.assertFalse(core.paused)

    async def test_rpc_waits_for_pause_drain_acknowledgement(self):
        backend, core, session, plan, _ = self.setup_backend()
        core.pause_release.clear()
        task = asyncio.create_task(backend.evict(session, plan))
        await asyncio.wait_for(core.pause_entered.wait(), 1)
        self.assertFalse(core.rpc_entered.is_set())
        core.pause_release.set()
        await asyncio.wait_for(task, 1)

    async def test_profile_covers_resume_without_changing_transaction_order(self):
        backend, core, session, plan, _ = self.setup_backend()
        with patch.dict(os.environ, {"ARC3_EVICTION_PROFILE": "1"}):
            result = await backend.evict(session, plan)
        self.assertEqual(result, {"plan_digest": plan.digest})
        profile = session.eviction_endpoint_profile
        self.assertEqual(set(profile), {"lock_wait_seconds", "status_seconds",
            "drain_seconds", "core_rpc_seconds", "resume_seconds", "total_seconds"})
        self.assertTrue(all(value >= 0 for value in profile.values()))
        self.assertGreaterEqual(profile["total_seconds"], sum(
            value for key, value in profile.items() if key != "total_seconds"))
        self.assertEqual(core.events[-2:], ["resume-enter", "resumed"])
        self.assertEqual(result["plan_digest"], plan.digest)

    async def test_already_paused_scheduler_is_not_owned_or_resumed(self):
        backend, core, session, plan, _ = self.setup_backend()
        core.paused = True
        with self.assertRaisesRegex(SessionError, "already paused"):
            await backend.evict(session, plan)
        self.assertEqual(core.events, ["status"])
        self.assertTrue(core.paused)

    async def test_pause_failure_attempts_resume_and_never_runs_rpc(self):
        backend, core, session, plan, _ = self.setup_backend()
        core.pause_error = RuntimeError("drain failed")
        with self.assertRaisesRegex(RuntimeError, "drain failed"):
            await backend.evict(session, plan)
        self.assertFalse(core.rpc_entered.is_set())
        self.assertEqual(core.events[-2:], ["resume-enter", "resumed"])
        self.assertFalse(core.paused)

    async def test_rpc_failure_resumes_before_propagating(self):
        backend, core, session, plan, _ = self.setup_backend()
        core.rpc_error = RuntimeError("copy failed")
        with self.assertRaisesRegex(RuntimeError, "copy failed"):
            await backend.evict(session, plan)
        self.assertEqual(core.events[-2:], ["resume-enter", "resumed"])
        self.assertFalse(core.paused)

    async def test_resume_failure_is_explicit(self):
        backend, core, session, plan, _ = self.setup_backend()
        core.resume_error = RuntimeError("resume failed")
        with self.assertRaisesRegex(RuntimeError, "resume failed"):
            await backend.evict(session, plan)
        self.assertTrue(core.paused)

    async def test_first_cancellation_waits_for_rpc_and_resume(self):
        backend, core, session, plan, _ = self.setup_backend()
        core.rpc_release.clear()
        task = asyncio.create_task(backend.evict(session, plan))
        await asyncio.wait_for(core.rpc_entered.wait(), 1)
        task.cancel()
        await self.ticks()
        self.assertFalse(task.done())
        self.assertFalse(core.rpc_cancelled)
        self.assertNotIn("resume-enter", core.events)
        core.rpc_release.set()
        with self.assertRaises(asyncio.CancelledError):
            await asyncio.wait_for(task, 1)
        self.assertEqual(core.events[-2:], ["resume-enter", "resumed"])

    async def test_repeated_cancellation_cannot_cancel_fenced_rpc(self):
        backend, core, session, plan, _ = self.setup_backend()
        core.rpc_release.clear()
        task = asyncio.create_task(backend.evict(session, plan))
        await asyncio.wait_for(core.rpc_entered.wait(), 1)
        try:
            task.cancel()
            await self.ticks()
            task.cancel()
            await self.ticks()
            self.assertFalse(core.rpc_cancelled, "Second cancellation reached the fenced worker utility")
            self.assertFalse(task.done())
            self.assertNotIn("resume-enter", core.events)
        finally:
            core.rpc_release.set()
            await asyncio.wait_for(asyncio.gather(task, return_exceptions=True), 1)
        self.assertIn(("rpc-completed", "engine-s"), core.events)
        self.assertEqual(core.events[-2:], ["resume-enter", "resumed"])

    async def test_two_game_evictions_serialize_entire_fence(self):
        backend, core, first, plan, _ = self.setup_backend()
        core.rpc_release.clear()
        second = Session("s2", "engine-s2", {})
        second.token_ids = list(range(24))
        second.current_operation = "op2"
        tasks = [asyncio.create_task(backend.evict(first, plan))]
        await asyncio.wait_for(core.rpc_entered.wait(), 1)
        tasks.append(asyncio.create_task(backend.evict(second, plan)))
        await self.ticks()
        self.assertEqual(core.events.count("pause-enter"), 1)
        core.rpc_release.set()
        await asyncio.wait_for(asyncio.gather(*tasks), 1)
        self.assertEqual(core.events.count("pause-enter"), 2)
        first_resumed = core.events.index("resumed")
        second_status = core.events.index("status", 1)
        self.assertLess(first_resumed, second_status)

    async def test_session_close_waits_for_eviction_before_aborting_cache_owner(self):
        backend, core, session, plan, aborted = self.setup_backend()
        core.rpc_release.clear()
        manager = SessionManager(backend, max_context=40000, max_sessions=2)
        manager.sessions[session.session_id] = session
        operation = asyncio.create_task(backend.evict(session, plan))
        session.operations[session.current_operation] = {"digest": "fixture", "task": operation}
        await asyncio.wait_for(core.rpc_entered.wait(), 1)
        close_task = asyncio.create_task(manager.close(session.session_id))
        try:
            await self.ticks()
            self.assertEqual(aborted, [], "Session close aborted/free'd cache during eviction")
            self.assertFalse(close_task.done())
        finally:
            core.rpc_release.set()
            await asyncio.wait_for(asyncio.gather(operation, close_task, return_exceptions=True), 1)
        self.assertLess(core.events.index("resumed"), core.events.index(("abort", "engine-s")))
        self.assertEqual(session.state, "closed")

    async def test_timeout_then_cancelled_and_repeated_close_keeps_one_live_fence(self):
        backend, core, session, plan, aborted = self.setup_backend()
        core.rpc_release.clear()
        manager = SessionManager(backend, max_context=40000, max_sessions=2)
        manager.sessions[session.session_id] = session
        eviction = asyncio.create_task(backend.evict(session, plan))
        operation = asyncio.create_task(asyncio.wait_for(eviction, timeout=.005))
        session.operations[session.current_operation] = {"digest": "fixture", "task": operation}
        await asyncio.wait_for(core.rpc_entered.wait(), 1)
        async def wait_for_timeout_cancel():
            while not eviction.cancelling():
                await asyncio.sleep(0)
        await asyncio.wait_for(wait_for_timeout_cancel(), 1)
        first_close = asyncio.create_task(manager.close(session.session_id))
        await self.ticks()
        shared_close = session.close_task
        first_close.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await first_close
        second_close = asyncio.create_task(manager.close(session.session_id))
        try:
            await self.ticks()
            self.assertIs(session.close_task, shared_close)
            self.assertFalse(shared_close.cancelled())
            self.assertFalse(core.rpc_cancelled)
            self.assertFalse(second_close.done())
            self.assertEqual(aborted, [])
        finally:
            core.rpc_release.set()
            await asyncio.wait_for(asyncio.gather(eviction, operation, second_close,
                                                 return_exceptions=True), 1)
        self.assertEqual(aborted, ["engine-s"])
        self.assertEqual(core.events.count("pause-enter"), 1)
        self.assertEqual(core.events.count("resumed"), 1)
        self.assertLess(core.events.index("resumed"), core.events.index(("abort", "engine-s")))
        self.assertEqual(session.state, "closed")

    async def test_cancelled_waiting_eviction_remains_serialized_and_resumes(self):
        backend, core, first, plan, _ = self.setup_backend()
        core.rpc_release.clear()
        second = Session("s2", "engine-s2", {})
        second.token_ids = list(range(24))
        second.current_operation = "op2"
        active = asyncio.create_task(backend.evict(first, plan))
        await asyncio.wait_for(core.rpc_entered.wait(), 1)
        queued = asyncio.create_task(backend.evict(second, plan))
        await self.ticks()
        queued.cancel()
        await self.ticks()
        self.assertFalse(queued.done())
        self.assertEqual(core.events.count("pause-enter"), 1)
        core.rpc_release.set()
        results = await asyncio.wait_for(asyncio.gather(active, queued, return_exceptions=True), 1)
        self.assertIsInstance(results[1], asyncio.CancelledError)
        self.assertEqual(core.events.count("resumed"), 2)
        self.assertLess(core.events.index("resumed"), core.events.index("status", 1))


if __name__ == "__main__":
    unittest.main()
