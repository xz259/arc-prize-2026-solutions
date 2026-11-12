"""CPU ordering/lifecycle tests; no model, CUDA, sockets, or remote execution."""
import ast
import hashlib
import json
import logging
import os
from pathlib import Path
import queue
import threading
import unittest
from concurrent.futures import Future
from functools import partial
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import control_transaction as control


WAKEUP = object()


def types():
    class Core:
        def __init__(self):
            self.paused = False
            self.input_queue = queue.Queue()
            self.pending = None
            self.events = []
            self.pause_error = None
            self.error_after_pause = False
            self.resume_error = None
            self.eviction_error = None
            self.ack = {"plan_digest": "abc", "nested": {"receipt": [1, 2]}}

        def is_scheduler_paused(self):
            return self.paused

        def pause_scheduler(self, mode="abort", clear_cache=True):
            self.events.append(("base-pause", mode, clear_cache))
            if self.pause_error and not self.error_after_pause:
                raise self.pause_error
            self.paused = True
            if self.pause_error:
                raise self.pause_error
            return self.pending

        def resume_scheduler(self):
            self.events.append(("base-resume",))
            if self.resume_error:
                raise self.resume_error
            self.paused = False

        def arc3_evict_resident(self, request_id, payload):
            self.events.append(("evict", request_id, payload["operation_id"]))
            if not self.paused:
                raise AssertionError("eviction outside pause")
            if self.eviction_error:
                raise self.eviction_error
            return self.ack

    class Proc(Core):
        def pause_scheduler(self, mode="abort", clear_cache=True):
            self.events.append(("proc-pause", mode, clear_cache))
            return super().pause_scheduler(mode, clear_cache)

        def resume_scheduler(self):
            self.events.append(("proc-resume",))
            return super().resume_scheduler()

    return Core, Proc


class ControlTests(unittest.TestCase):
    def setUp(self):
        self.Core, self.Proc = types()
        self.flag = patch.dict(os.environ, {control.FLAG: "1"})
        self.flag.start()
        self.addCleanup(self.flag.stop)
        control.install(self.Core, wakeup_type=WAKEUP)
        self.core = self.Proc()
        self.payload = {"operation_id": "operation-1", "logical_ranges": [[2, 5]]}

    def start(self, core=None):
        return getattr(core or self.core, control.METHOD)("owner", self.payload)

    def assert_clean(self, *, wakeup=True):
        self.assertFalse(self.core.paused)
        self.assertIsNone(getattr(self.core, control._LEASE, None))
        if wakeup:
            self.assertEqual(self.core.input_queue.get_nowait(), (WAKEUP, None))
        self.assertTrue(self.core.input_queue.empty())

    def test_immediate_exact_ack_separate_numeric_profile(self):
        before = json.dumps(self.core.ack, sort_keys=True)
        result = self.start().result()
        self.assertIs(result["ack"], self.core.ack)
        self.assertEqual(json.dumps(self.core.ack, sort_keys=True), before)
        self.assertTrue(all(type(v) in (int, float) and v >= 0
                            for v in result["control_profile"].values()))
        self.assertEqual(result["control_profile"]["deferred_drain"], 0)
        self.assertEqual([e[0] for e in self.core.events],
                         ["proc-pause", "base-pause", "evict", "proc-resume", "base-resume"])
        self.assertEqual(self.core.events[0][1:], ("keep", False))
        self.assert_clean()

    def test_delayed_drain_cannot_cancel_final_control_future(self):
        self.core.pending = Future()
        result = self.start()
        self.assertFalse(result.done())
        self.assertFalse(result.cancel())
        self.assertEqual([e[0] for e in self.core.events], ["proc-pause", "base-pause"])
        self.core.pending.set_result(None)
        self.assertEqual(result.result()["control_profile"]["deferred_drain"], 1)
        self.assert_clean()

    def test_pending_lease_rejects_external_base_and_override_pause_resume(self):
        self.core.pending = Future()
        result = self.start()
        before = list(self.core.events)
        for call in (lambda: self.core.pause_scheduler(mode="abort", clear_cache=True),
                     lambda: self.core.resume_scheduler(),
                     lambda: self.Core.pause_scheduler(self.core),
                     lambda: self.Core.resume_scheduler(self.core),
                     lambda: self.start()):
            with self.assertRaises(control.ControlTransactionError):
                call()
            self.assertEqual(self.core.events, before)
            self.assertTrue(self.core.paused)
        self.core.pending.set_result(None)
        self.assertIs(result.result()["ack"], self.core.ack)
        self.assert_clean()

    def test_existing_external_pause_is_untouched(self):
        self.core.paused = True
        with self.assertRaises(control.ControlTransactionError):
            self.start()
        self.assertEqual(self.core.events, [])
        self.assertTrue(self.core.paused)
        self.assertIsNone(getattr(self.core, control._LEASE, None))

    def test_other_instance_is_not_blocked_by_lease(self):
        self.core.pending = Future()
        result = self.start()
        other = self.Proc()
        other.pause_scheduler()
        other.resume_scheduler()
        self.assertFalse(other.paused)
        self.core.pending.set_result(None)
        result.result()
        self.assert_clean()

    def test_pause_errors_before_and_after_pause(self):
        for after in (False, True):
            with self.subTest(after=after):
                self.core = self.Proc()
                error = RuntimeError("pause failed")
                self.core.pause_error = error
                self.core.error_after_pause = after
                with self.assertRaises(RuntimeError) as caught:
                    self.start().result()
                self.assertIs(caught.exception, error)
                self.assertNotIn("evict", [e[0] for e in self.core.events])
                self.assert_clean(wakeup=after)

    def test_drain_failure_resumes_without_mutation(self):
        self.core.pending = Future()
        result = self.start()
        error = RuntimeError("drain failed")
        self.core.pending.set_exception(error)
        with self.assertRaises(RuntimeError) as caught:
            result.result()
        self.assertIs(caught.exception, error)
        self.assertNotIn("evict", [e[0] for e in self.core.events])
        self.assert_clean()

    def test_eviction_error_resumes_preserving_original_error(self):
        error = RuntimeError("existing utility rejected stale digest")
        self.core.eviction_error = error
        with self.assertRaises(RuntimeError) as caught:
            self.start().result()
        self.assertIs(caught.exception, error)
        self.assert_clean()

    def test_resume_failure_retains_lease_and_primary_error(self):
        self.core.eviction_error = ValueError("copy failed and poisoned owner")
        self.core.resume_error = RuntimeError("resume failed")
        with self.assertRaises(control.ControlCleanupError) as caught:
            self.start().result()
        self.assertIs(caught.exception.primary, self.core.eviction_error)
        self.assertIs(caught.exception.cleanup, self.core.resume_error)
        self.assertTrue(self.core.paused)
        self.assertIsNotNone(getattr(self.core, control._LEASE, None))
        with self.assertRaises(control.ControlTransactionError):
            self.core.resume_scheduler()

    def test_wrong_thread_drain_fails_closed_without_mutation_or_resume(self):
        self.core.pending = Future()
        result = self.start()
        thread = threading.Thread(target=lambda: self.core.pending.set_result(None))
        thread.start()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        with self.assertRaises(control.ControlThreadError):
            result.result()
        self.assertTrue(self.core.paused)
        self.assertIsNotNone(getattr(self.core, control._LEASE, None))
        self.assertEqual([e[0] for e in self.core.events], ["proc-pause", "base-pause"])

    def test_lost_lease_never_resumes_another_owner(self):
        self.core.pending = Future()
        result = self.start()
        foreign = object()
        setattr(self.core, control._LEASE, foreign)
        self.core.pending.set_result(None)
        with self.assertRaises(control.ControlTransactionError):
            result.result()
        self.assertIs(getattr(self.core, control._LEASE), foreign)
        self.assertTrue(self.core.paused)
        self.assertEqual([e[0] for e in self.core.events], ["proc-pause", "base-pause"])

    def test_second_completed_transaction_allowed(self):
        self.start().result()
        self.assert_clean()
        self.payload["operation_id"] = "operation-2"
        self.start().result()
        self.assertEqual(sum(e[0] == "evict" for e in self.core.events), 2)
        self.assert_clean()

    def test_flag_zero_leaves_classes_and_legacy_utility_unchanged(self):
        Core, Proc = types()
        before = (Core.pause_scheduler, Proc.pause_scheduler, Core.arc3_evict_resident)
        with patch.dict(os.environ, {control.FLAG: "0"}):
            self.assertFalse(control.install(Core, wakeup_type=WAKEUP)["installed"])
        self.assertEqual(before, (Core.pause_scheduler, Proc.pause_scheduler, Core.arc3_evict_resident))
        self.assertFalse(hasattr(Core, control.METHOD))

    def test_install_idempotent_and_legacy_calls_work_without_lease(self):
        before = self.Core.pause_scheduler
        control.install(self.Core, wakeup_type=WAKEUP)
        self.assertIs(before, self.Core.pause_scheduler)
        self.core.pause_scheduler(mode="abort", clear_cache=True)
        self.core.resume_scheduler()
        self.assertEqual(self.core.events[0], ("proc-pause", "abort", True))

    def test_invalid_identity_rejected_before_pause(self):
        for value in ({}, {"operation_id": ""}, {"operation_id": 7}):
            with self.subTest(value=value), self.assertRaises(control.ControlTransactionError):
                getattr(self.core, control.METHOD)("owner", value)
        self.assertFalse(self.core.events)


class NativeIdleLoopTests(unittest.TestCase):
    def test_native_deferred_pause_callback_resumes_without_frontend_traffic(self):
        source = (Path(__file__).resolve().parents[2] / "persistent_state_20260912" /
                  "source-snapshot/vllm/v1/engine/core.py")
        if not source.exists():
            self.skipTest("Pinned native source snapshot unavailable; synthetic tests still run")
        self.assertEqual(hashlib.sha256(source.read_bytes()).hexdigest(),
                         "f4b1e07b6d91fac1549bc74a5c804e5a88793d663b5dc045c9e455db6550f5ad")
        tree = ast.parse(source.read_text())
        proc = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EngineCoreProc")
        names = {"pause_scheduler", "_pause_complete", "has_work", "_process_input_queue",
                 "_notify_idle_state_callbacks"}
        methods = [n for n in proc.body if isinstance(n, ast.FunctionDef) and n.name in names]
        self.assertEqual({n.name for n in methods}, names)
        namespace = {"Future": Future, "partial": partial, "queue": queue, "Any": Any,
                     "PauseState": SimpleNamespace(PAUSED_ALL=1, PAUSED_NEW=2),
                     "logger": logging.getLogger(__name__), "DEBUG": logging.DEBUG}
        exec(compile("from __future__ import annotations\n" + "\n\n".join(
            ast.unparse(n) for n in methods), str(source), "exec"), namespace)
        Core, Proc = types()
        for name in names:
            setattr(Proc, name, namespace[name])
        Core.is_scheduler_paused = lambda self: bool(self.scheduler.pause_state)
        def resume(self):
            self.events.append(("native-resume",))
            self.scheduler.pause_state = 0
        Proc.resume_scheduler = resume
        def evict(self, rid, payload):
            self.assertions.append(not self.batch_queue and self.scheduler.pause_state == 1)
            return self.ack
        Core.arc3_evict_resident = evict
        with patch.dict(os.environ, {control.FLAG: "1"}):
            control.install(Core, wakeup_type=WAKEUP)
        core = Proc()
        core.scheduler = SimpleNamespace(pause_state=0)
        core.scheduler.set_pause_state = lambda value: setattr(core.scheduler, "pause_state", value)
        core.scheduler.has_requests = lambda: core.scheduler.pause_state == 0
        core.engines_running = False
        core.batch_queue = ["one prior queued model step"]
        core._idle_state_callbacks = []
        core.process_input_queue_block = True
        core.is_running = lambda: True
        core.aborts_queue = queue.Queue()
        core._handle_client_request = lambda kind, payload: core.events.append(("wakeup", kind))
        core.assertions = []
        output = queue.Queue()
        def core_thread():
            try:
                result = getattr(core, control.METHOD)("owner", {"operation_id": "native-1"})
                if result.done():
                    result.result()
                    raise AssertionError("Queued work must delay the pause")
                # Stand in for completion of the already-scheduled model work.
                core.batch_queue.clear()
                # This exact native method invokes idle callbacks, then can
                # block on input_queue.get unless the callback sent WAKEUP.
                core._process_input_queue()
                output.put(result.result())
            except BaseException as error:
                output.put(error)
        thread = threading.Thread(target=core_thread)
        thread.start()
        thread.join(2)
        was_stuck = thread.is_alive()
        if was_stuck:
            core.input_queue.put((WAKEUP, None))
            thread.join(2)
        self.assertFalse(was_stuck, "Resume depended on unrelated subsequent frontend traffic")
        value = output.get(timeout=1)
        if isinstance(value, BaseException):
            raise value
        self.assertIs(value["ack"], core.ack)
        self.assertEqual(core.assertions, [True])
        self.assertEqual(core.events, [("native-resume",), ("wakeup", WAKEUP)])
        self.assertEqual(value["control_profile"]["callback_thread_verified"], 1)
        self.assertIsNone(getattr(core, control._LEASE, None))


if __name__ == "__main__":
    unittest.main()
