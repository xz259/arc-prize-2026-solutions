import ast
from collections import deque
from concurrent.futures import Future
from contextlib import contextmanager
import copy
import enum
import os
from pathlib import Path
import threading
from types import SimpleNamespace as NS
from typing import cast
import unittest
from unittest.mock import patch

import persistent_batch_policy as policy


SOURCE = Path(os.environ.get("ARC3_RUNTIME_SOURCE", str(Path(__file__).resolve().parents[2]/
    "persistent_state_20260912/source-snapshot/vllm")))/"v1/engine/core.py"


class Status(enum.IntEnum):
    RUNNING = 0
    PAUSED = 1
    FINISHED = 2


class Request:
    def __init__(self, name, *, persistent=True, cap=10, eos=None, structured=False):
        self.name, self.persistent, self.cap, self.eos = name, persistent, cap, eos
        self.structured, self.status = structured, Status.RUNNING
        self.num_in_flight_tokens = self.accepted = self.computed = self.max_inflight = 0
    def is_finished(self):
        return self.status == Status.FINISHED


class Scheduler:
    def __init__(self, core):
        self.core, self.requests, self.paused = core, {}, False
        self.batches = []
    def has_requests(self):
        return not self.paused and any(req.status == Status.RUNNING for req in self.requests.values())
    def schedule(self, throttle):
        pending = False
        ids = []
        for req in self.requests.values():
            if self.paused or req.status != Status.RUNNING:
                continue
            # Existing per-request guard: never removed by the new core policy.
            if req.persistent and req.num_in_flight_tokens:
                continue
            pending |= req.structured and req.num_in_flight_tokens > 0
            req.num_in_flight_tokens += 1
            req.max_inflight = max(req.max_inflight, req.num_in_flight_tokens)
            ids.append(req.name)
        self.batches.append(tuple(ids))
        return NS(total_num_scheduled_tokens=len(ids), pending_structured_output_tokens=pending,
                  ids=ids)
    def get_grammar_bitmask(self, output):
        self.core.trace.append("grammar")
        return None
    def update_from_output(self, scheduled, output):
        self.core.trace.append("update")
        for name in output.ids:
            req = self.requests.get(name)
            if req is None:
                continue
            req.num_in_flight_tokens -= 1
            req.computed += 1
            req.accepted += 1
            stop = req.accepted >= req.cap or req.accepted == req.eos
            if stop:
                req.status = Status.PAUSED if req.persistent else Status.FINISHED
                if req.persistent:
                    assert req.num_in_flight_tokens == 0
                    # One initial pending prompt token + generated tokens.
                    assert 1+req.accepted-req.computed == 1
        return {0: NS(ids=tuple(output.ids))}


class Executor:
    def __init__(self, core):
        self.core, self.last, self.sample_error, self.execute_error = core, None, None, None
        self.executed = []
    def execute_model(self, output, non_block):
        self.core.trace.append("execute")
        self.executed.append(tuple(output.ids))
        self.last = output
        result = Future()
        if self.execute_error is not None:
            result.set_exception(self.execute_error)
        else:
            result.set_result(NS(ids=[]) if not output.ids else None)
        return result
    def sample_tokens(self, grammar, non_block=True):
        self.core.trace.append("sample")
        result = Future()
        if self.sample_error is not None:
            result.set_exception(self.sample_error)
        elif self.execute_error is not None:
            result.set_result(None)
        else:
            result.set_result(NS(ids=list(self.last.ids)))
        return result


class Core:
    def __init__(self):
        self.trace, self.pending_aborts = [], []
        self.scheduler = Scheduler(self)
        self.model_executor = Executor(self)
        self.batch_queue, self.batch_queue_size = deque(maxlen=2), 2
        self.is_ec_consumer, self.is_pooling_model, self.check_for_draft_tokens = True, False, False
        self.vllm_config = NS(parallel_config=NS(tensor_parallel_size=1,
            pipeline_parallel_size=1, data_parallel_size=1), num_speculative_tokens=0)
    def _should_throttle_prefills(self):
        return False
    @contextmanager
    def log_error_detail(self, output):
        yield
    @contextmanager
    def capture_iteration_details(self, output):
        yield None
    def _process_aborts_queue(self):
        self.trace.append("aborts")
        for name in self.pending_aborts:
            self.scheduler.requests.pop(name, None)
        self.pending_aborts.clear()
    def _attach_iteration_details(self, outputs, details):
        self.trace.append("attach")


def methods():
    source = SOURCE.read_text()
    namespace = dict(Future=Future, ModelRunnerOutput=object, cast=cast)
    changed = policy.build(source, namespace, lambda core: policy.active(core,
        is_persistent=lambda req: req.persistent, paused_status=Status.PAUSED))
    tree = ast.parse(source)
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "EngineCore")
    fn = copy.deepcopy(next(n for n in cls.body if isinstance(n, ast.FunctionDef)
                            and n.name == "step_with_batch_queue"))
    module = ast.Module(body=[ast.ImportFrom(module="__future__",
        names=[ast.alias(name="annotations")], level=0), fn], type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), "native_queue", "exec"), namespace)
    return namespace[fn.name], changed


def add(core, prefix, number, **kwargs):
    result = [Request(f"{prefix}{i}", **kwargs) for i in range(number)]
    core.scheduler.requests.update({req.name: req for req in result})
    return result


class QueueTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        native, changed = methods()
        cls.native, cls.changed = staticmethod(native), staticmethod(changed)
    def test_real_native_two_plus_twenty_three_phase_lock_then_coalescing(self):
        base = Core()
        add(base, "a", 2)
        self.native(base)
        add(base, "b", 23)
        for _ in range(4):
            self.native(base)
        self.assertEqual(list(map(len, base.scheduler.batches)), [2, 23, 2, 23, 2])
        updated = Core()
        requests = add(updated, "a", 2)
        self.changed(updated)
        requests += add(updated, "b", 23)
        for _ in range(4):
            self.changed(updated)
        self.assertEqual(list(map(len, updated.scheduler.batches)), [2, 25, 25, 25, 25])
        self.assertEqual(len(updated.batch_queue), 0)
        self.assertEqual(updated.batch_queue.maxlen, 2)
        self.assertTrue(all(req.max_inflight == 1 for req in requests))
    def test_existing_queue_drains_before_new_persistent_owners_are_scheduled(self):
        core = Core()
        add(core, "a", 2)
        self.native(core)
        add(core, "b", 23)
        count = len(core.scheduler.batches)
        outputs, executed = self.changed(core)
        self.assertEqual(len(core.scheduler.batches), count)
        self.assertFalse(executed)  # No fabricated model-executed flag for a drain.
        self.assertEqual(len(outputs[0].ids), 2)
        self.changed(core)
        self.assertEqual(len(core.scheduler.batches[-1]), 25)
    def test_single_cohort_has_no_zero_token_filler_rpc(self):
        core = Core()
        add(core, "a", 25)
        for _ in range(5):
            _, executed = self.changed(core)
            self.assertTrue(executed)
        self.assertEqual(list(map(len, core.model_executor.executed)), [25]*5)
    def test_eos_and_output_cap_preserve_accepted_frontier(self):
        core = Core()
        eos = add(core, "eos", 1, cap=20, eos=2)[0]
        cap = add(core, "cap", 1, cap=4)[0]
        for _ in range(4):
            self.changed(core)
        self.assertEqual((eos.accepted, cap.accepted), (2, 4))
        self.assertEqual((eos.status, cap.status), (Status.PAUSED, Status.PAUSED))
        for req in (eos, cap):
            self.assertEqual(req.num_in_flight_tokens, 0)
            self.assertEqual(1+req.accepted-req.computed, 1)
    def test_ordinary_only_and_paused_persistent_keep_native_overlap(self):
        for paused in (False, True):
            a, b = Core(), Core()
            for core in (a, b):
                add(core, "o", 3, persistent=False)
                if paused:
                    add(core, "p", 1)[0].status = Status.PAUSED
            for _ in range(4):
                out_a = self.native(a)
                out_b = self.changed(b)
                self.assertEqual(out_a[1], out_b[1])
                self.assertEqual(a.scheduler.batches, b.scheduler.batches)
                self.assertEqual(a.trace, b.trace)
            self.assertEqual(len(b.batch_queue), 1)
    def test_mixed_ordinary_and_active_persistent_share_one_batch(self):
        core = Core()
        requests = add(core, "o", 2, persistent=False)+add(core, "p", 3)
        for _ in range(4):
            self.changed(core)
        self.assertEqual(list(map(len, core.scheduler.batches)), [5]*4)
        self.assertTrue(all(req.max_inflight == 1 for req in requests))
    def test_keep_pause_drains_existing_queue_and_schedules_nothing(self):
        core = Core()
        req = add(core, "p", 1)[0]
        self.native(core)
        core.scheduler.paused = True
        _, executed = self.changed(core)
        self.assertFalse(executed)
        self.assertFalse(core.batch_queue)
        self.assertEqual(req.num_in_flight_tokens, 0)
        self.assertEqual(len(core.scheduler.batches), 1)
        self.assertEqual(self.changed(core), (None, False))
    def test_empty_allpaused_and_finished_do_not_activate_policy(self):
        core = Core()
        self.assertEqual(self.changed(core), (None, False))
        add(core, "p", 1)[0].status = Status.PAUSED
        add(core, "f", 1)[0].status = Status.FINISHED
        self.assertEqual(self.changed(core), (None, False))
        self.assertEqual(core.model_executor.executed, [])
    def test_abort_precedes_native_output_update_and_other_owners_survive(self):
        core = Core()
        add(core, "a", 2)
        core.pending_aborts = ["a0"]
        self.changed(core)
        self.assertNotIn("a0", core.scheduler.requests)
        self.assertEqual(core.scheduler.requests["a1"].accepted, 1)
        self.assertLess(core.trace.index("aborts"), core.trace.index("update"))
    def test_native_sample_and_execute_failures_propagate_exactly(self):
        for kind in ("sample_error", "execute_error"):
            core = Core()
            add(core, "p", 1)
            failure = RuntimeError(kind)
            setattr(core.model_executor, kind, failure)
            with self.assertRaises(RuntimeError) as caught:
                self.changed(core)
            self.assertIs(caught.exception, failure)
            self.assertNotIn("update", core.trace)
    def test_ordinary_deferred_structured_sampling_path_is_unchanged(self):
        a, b = Core(), Core()
        for core in (a, b):
            add(core, "o", 2, persistent=False, structured=True)
        for _ in range(4):
            self.native(a); self.changed(b)
            self.assertEqual(a.trace, b.trace)
            self.assertEqual(a.scheduler.batches, b.scheduler.batches)
    def test_new_persistent_owner_drains_existing_deferred_ordinary_batch(self):
        core = Core()
        ordinary = add(core, "o", 2, persistent=False, structured=True)
        self.changed(core)
        self.changed(core)
        # Native deferred sampling has consumed the older output and left the
        # grammar-dependent output pending in the existing queue.
        self.assertEqual(len(core.batch_queue), 1)
        self.assertTrue(all(req.num_in_flight_tokens == 1 for req in ordinary))
        self.assertEqual(core.trace[-2:], ["grammar", "sample"])
        persistent = add(core, "p", 1)[0]
        previous_batches = list(core.scheduler.batches)
        result, executed = self.changed(core)
        self.assertFalse(executed)
        self.assertEqual(core.scheduler.batches, previous_batches)
        self.assertEqual(result[0].ids, ("o0", "o1"))
        self.assertTrue(all(req.num_in_flight_tokens == 0 for req in ordinary))
        self.assertEqual(persistent.accepted, 0)
        self.changed(core)
        self.assertEqual(core.scheduler.batches[-1], ("o0", "o1", "p0"))
        self.assertEqual(persistent.max_inflight, 1)
        self.assertEqual(len(core.batch_queue), 0)
    def test_delayed_future_blocks_then_processes_aborts_without_extra_submission(self):
        core = Core(); requests = add(core, "p", 2)
        entered = threading.Event()
        class ObservedFuture(Future):
            def result(self, timeout=None):
                entered.set()
                return super().result(timeout)
        pending = ObservedFuture()
        core.model_executor.sample_tokens = lambda *args, **kwargs: pending
        outputs, errors = [], []
        def invoke():
            try:
                outputs.append(self.changed(core))
            except BaseException as exc:
                errors.append(exc)
        worker = threading.Thread(target=invoke, daemon=True)
        worker.start()
        try:
            self.assertTrue(entered.wait(1), "native Future.result was not reached")
            self.assertTrue(worker.is_alive())
            self.assertEqual(core.scheduler.batches, [("p0", "p1")])
            self.assertEqual([req.num_in_flight_tokens for req in requests], [1, 1])
            core.pending_aborts.append("p0")
        finally:
            pending.set_result(NS(ids=["p0", "p1"]))
            worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertEqual(errors, [])
        self.assertTrue(outputs[0][1])
        self.assertNotIn("p0", core.scheduler.requests)
        self.assertEqual(requests[1].accepted, 1)
        self.assertFalse(core.batch_queue)
    def test_unsupported_pipeline_rejected_before_schedule(self):
        core = Core(); add(core, "p", 1)
        core.vllm_config.parallel_config.pipeline_parallel_size = 2
        with self.assertRaises(policy.BatchPolicyError):
            self.changed(core)
        self.assertEqual(core.model_executor.executed, [])


class SourceTests(unittest.TestCase):
    def test_only_two_conditions_and_one_boolean_differ(self):
        source = SOURCE.read_text()
        changed = policy.transform(source)
        cls = next(n for n in ast.parse(source).body if isinstance(n, ast.ClassDef) and n.name == "EngineCore")
        native = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "step_with_batch_queue")
        self.assertEqual(ast.unparse(changed.body.pop(1)),
                         f"{policy.POLICY_NAME} = {policy.HELPER_NAME}(self)")
        for node in ast.walk(changed):
            if isinstance(node, ast.If) and isinstance(node.test, ast.BoolOp):
                if any(isinstance(n, ast.Name) and n.id == policy.POLICY_NAME for n in ast.walk(node.test)):
                    node.test = node.test.values[0]
        self.assertEqual(ast.dump(changed, include_attributes=False), ast.dump(native, include_attributes=False))
    def test_changed_native_source_fails_closed(self):
        with self.assertRaises(policy.BatchPolicyError):
            policy.transform(SOURCE.read_text()+"\n")
    def test_disabled_install_does_not_import_vllm(self):
        with patch.dict(os.environ, {policy.FLAG: "0"}):
            self.assertFalse(policy.install()["installed"])


if __name__ == "__main__":
    unittest.main()
