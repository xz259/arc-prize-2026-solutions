import asyncio
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import resume_profile as p


def request(rid="owner", tokens=4, persistent=True):
    return NS(request_id=rid, req_id=rid, prompt_token_ids=list(range(tokens)),
              prefill_token_ids=list(range(tokens)), _all_token_ids=list(range(tokens)),
              num_computed_tokens=tokens-1, _arc3_persistent_turn=0,
              resumable=persistent, sampling_params=NS(extra_args={"arc3_persistent_state": persistent}))


def fixture():
    trace = []
    ep, ee, er = NS(), NS(), NS()
    ep.is_persistent = lambda r: r.resumable and r.sampling_params.extra_args["arc3_persistent_state"]

    class LogicalState:
        def __init__(self):
            self.logical_ids = [0, 1, 2, 3]
            self.eviction_revision = 0
        def paused(self, req):
            trace.append("paused")
        def context(self, req):
            trace.append("context")
            return dict(logical_token_ids=self.logical_ids)
    ee.LogicalState = LogicalState
    def publish(req):
        req.sampling_params.extra_args["arc3_eviction_context"] = req._arc3_eviction_state.context(req)
    ee.publish_context = publish
    def append(scheduler, req, update, **kwargs):
        trace.append("append")
        req._all_token_ids.extend(update.prompt_token_ids)
        req._arc3_persistent_turn += 1
    ep.append_session = append
    ep._audit = lambda *a, **kw: trace.append("audit")

    class Scheduler:
        def __init__(self):
            self.requests = {}
        def add_request(self, req):
            trace.append("scheduler_add")
            if req.request_id in self.requests:
                return self._update_request_as_session(self.requests[req.request_id], req)
            req._arc3_eviction_state = LogicalState()
            self.requests[req.request_id] = req
            ee.publish_context(req)
            ep._audit(self, "start", req)
            return "new"
        def _update_request_as_session(self, req, update):
            req._arc3_eviction_state.paused(req)
            ep.append_session(self, req, update)
            ee.publish_context(req)
            ep._audit(self, "resume", req)
            return "updated"
    class Core:
        def __init__(self):
            self.scheduler = Scheduler()
        def add_request(self, req, request_wave=0):
            trace.append("core_add")
            return self.scheduler.add_request(req)
        def preprocess_add_request(self, req):
            trace.append("preprocess")
            return req, 0
    class RequestState:
        def __init__(self):
            self.req_id_to_index = {}
        def add_request(self, req_id, **kwargs):
            self.req_id_to_index[req_id] = 2
            trace.append("token_add")
        def apply_staged_writes(self):
            trace.append("token_apply")
    er._stage_request = lambda *a: trace.append("eviction_stage")
    class ModelState:
        def add_request(self, index, data):
            er._stage_request(self, index, data)
            trace.append("original_model_add")
        def apply_staged_writes(self):
            trace.append("model_apply")
    class Runner:
        def __init__(self):
            self.req_states, self.model_state = RequestState(), ModelState()
        def add_requests(self, scheduler_output):
            for req in scheduler_output.scheduled_new_reqs:
                self.req_states.add_request(req_id=req.req_id)
                self.model_state.add_request(2, req)
            if scheduler_output.scheduled_new_reqs:
                self.req_states.apply_staged_writes()
                self.model_state.apply_staged_writes()
            return "worker-result"
    class Frontend:
        async def add_request_async(self, req):
            trace.append("frontend_before")
            await asyncio.sleep(0)
            trace.append("frontend_after")
            return "sent"
    class Executor:
        def collective_rpc(self, method, timeout=None, args=(), kwargs=None, non_block=False,
                           unique_reply_rank=None, kv_output_aggregator=None, ec_output_aggregator=None):
            trace.append(("executor", method, non_block))
            class Future:
                def result(self):
                    raise AssertionError("Observer must not wait on or read native future")
            return Future()
    return dict(scheduler=Scheduler, core=Core, engine_patch=ep, engine_eviction=ee,
                runner=Runner, model_state=ModelState, request_state=RequestState,
                eviction_runner=er, frontend=Frontend, executor=Executor), trace


class ProfileTests(unittest.TestCase):
    def setUp(self):
        self.args, self.trace = fixture()
        self.events = []
        self.writer = patch.object(p, "_write", self.events.append)
        self.writer.start()
        self.addCleanup(self.writer.stop)
        p.install_targets(**self.args)

    def test_nested_core_one_record_return_and_state_preserved(self):
        core = self.args["core"]()
        self.assertEqual(core.add_request(request()), "new")
        self.events.clear()
        self.assertEqual(core.add_request(request(tokens=2), request_wave=7), "updated")
        self.assertEqual(len(self.events), 1)
        event = self.events[0]
        self.assertEqual(event["event"], "core_add")
        self.assertEqual(event["before"]["input_tokens"], 2)
        self.assertEqual(event["after"]["resident_tokens"], 6)
        self.assertEqual(event["after"]["turn"], 1)
        for name in ("scheduler_add", "core_update_session", "core_append_session",
                     "core_logical_paused", "core_publish_context", "core_logical_context", "core_audit"):
            self.assertEqual(event["phases"][name]["calls"], 1)
            self.assertGreaterEqual(event["phases"][name]["seconds"], 0)
        self.assertIsNone(event["error_type"])
        self.assertLessEqual(event["mono_start"], event["mono_end"])
        self.assertIsNone(p._CURRENT.get())

    def test_scheduler_only_has_outer_event(self):
        self.args["scheduler"]().add_request(request())
        self.assertEqual(self.events[0]["event"], "scheduler_add")

    def test_preprocess_is_distinct_from_scheduler_entry(self):
        req = request()
        self.assertEqual(self.args["core"]().preprocess_add_request(req), (req, 0))
        self.assertEqual(self.events[0]["event"], "core_preprocess_add")
        self.assertEqual(self.events[0]["phases"], {})

    def test_nonpersistent_core_is_untimed(self):
        self.args["core"]().add_request(request(persistent=False))
        self.assertEqual(self.events, [])

    def test_pause_and_audit_outside_append_are_untimed(self):
        self.args["engine_eviction"].LogicalState().paused(request())
        self.args["engine_patch"]._audit(None, "pause", request())
        self.assertEqual(self.events, [])

    def test_worker_one_batch_records_new_and_readded_and_native_calls(self):
        runner = self.args["runner"]()
        runner.req_states.req_id_to_index["old"] = 9
        output = NS(scheduled_new_reqs=[request("new"), request("old")])
        self.assertEqual(runner.add_requests(output), "worker-result")
        event = self.events[0]
        self.assertEqual([r["readded"] for r in event["before"]], [False, True])
        self.assertEqual([r["slot"] for r in event["after"]], [2, 2])
        for name in ("worker_model_state_add", "worker_eviction_stage", "worker_token_state_add"):
            self.assertEqual(event["phases"][name]["calls"], 2)
        self.assertEqual(event["phases"]["worker_token_apply"]["calls"], 1)
        self.assertEqual(self.trace.count("original_model_add"), 2)
        self.assertNotIn("prompt_token_ids", json.dumps(event))

    def test_empty_decode_batch_skips_events(self):
        self.assertEqual(self.args["runner"]().add_requests(NS(scheduled_new_reqs=[])), "worker-result")
        self.assertEqual(self.events, [])

    def test_worker_native_keyword_signature(self):
        runner = self.args["runner"]()
        self.assertEqual(runner.add_requests(scheduler_output=NS(scheduled_new_reqs=[request()])), "worker-result")
        self.assertEqual(self.events[0]["event"], "worker_add_batch")

    def test_frontend_metadata_error_does_not_change_native_result(self):
        with patch.object(p, "_request_info", side_effect=ValueError("metadata")):
            self.assertEqual(asyncio.run(self.args["frontend"]().add_request_async(request())), "sent")
        self.assertEqual(self.events[0]["before"], {"profile_metadata_error": "ValueError"})

    def test_exception_identity_and_context_reset(self):
        failure = RuntimeError("native failure")
        def fail(self, req):
            raise failure
        wrapped = p._outer(fail, "test", lambda *a: True,
                           lambda *a: {}, lambda *a: {})
        with self.assertRaises(RuntimeError) as caught:
            wrapped(None, request())
        self.assertIs(caught.exception, failure)
        self.assertEqual(self.events[0]["error_type"], "RuntimeError")
        self.assertIsNone(p._CURRENT.get())

    def test_observer_failure_does_not_override_success(self):
        with patch.object(p, "_write", side_effect=OSError("disk")):
            self.assertEqual(self.args["core"]().add_request(request()), "new")
        self.assertIsNone(p._CURRENT.get())

    def test_metadata_error_is_recorded_without_changing_native_result(self):
        def bad(*a):
            raise ValueError("metadata only")
        fn = p._outer(lambda: "result", "test", lambda: True, bad, bad)
        self.assertEqual(fn(), "result")
        self.assertEqual(self.events[0]["before"], {"profile_metadata_error": "ValueError"})
        self.assertIsNone(self.events[0]["error_type"])

    def test_executor_preserves_future_no_wait_and_only_new_nonblocking_batches(self):
        executor = self.args["executor"]()
        populated, empty = NS(scheduled_new_reqs=[request()]), NS(scheduled_new_reqs=[])
        future = executor.collective_rpc("execute_model", args=(populated,), non_block=True,
                                        unique_reply_rank=0)
        self.assertTrue(callable(future.result))
        self.assertEqual(self.events[0]["event"], "executor_enqueue_new")
        self.events.clear()
        executor.collective_rpc("execute_model", args=(empty,), non_block=True)
        executor.collective_rpc("execute_model", args=(populated,), non_block=False)
        executor.collective_rpc("sample_tokens", args=(populated,), non_block=True)
        self.assertEqual(self.events, [])
        executor.collective_rpc("execute_model", None, (populated,), None, True, 0, None, None)
        self.assertEqual(len(self.events), 1)

    def test_frontend_async_events_are_isolated(self):
        async def run():
            frontend = self.args["frontend"]()
            return await asyncio.gather(*(frontend.add_request_async(request(str(i))) for i in range(3)))
        self.assertEqual(asyncio.run(run()), ["sent"]*3)
        self.assertEqual(len(self.events), 3)
        self.assertEqual({e["before"]["request_id"] for e in self.events}, {"0", "1", "2"})
        self.assertIsNone(p._CURRENT.get())

    def test_signature_and_wrapped_name_preserved(self):
        import inspect
        method = self.args["core"].add_request
        self.assertEqual(str(inspect.signature(method)), "(self, req, request_wave=0)")
        self.assertEqual(method.__name__, "add_request")


class InstallationTests(unittest.TestCase):
    def test_disabled_never_imports_runtime_or_requires_path(self):
        with patch.dict(os.environ, {p.FLAG: "0"}), patch.dict(sys.modules, {"engine_patch": None}):
            self.assertFalse(p.install()["installed"])

    def test_bad_flag_and_path_rejected_before_imports(self):
        for env in ({p.FLAG: "yes"}, {p.FLAG: "1", p.PATH: "relative"}):
            with patch.dict(os.environ, env), self.assertRaises(ValueError):
                p.install()

    def test_journal_is_per_pid_no_token_payload(self):
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {p.PATH: str(Path(temp)/"profile")}):
            before = p._STATS["written"]
            p._write({"event": "test", "seconds": 0.1})
            journal = Path(p.status()["journal"])
            self.assertEqual(json.loads(journal.read_text()), {"event": "test", "seconds": 0.1})
            self.assertEqual(p._STATS["written"], before+1)
            self.assertEqual(journal.stat().st_mode & 0o777, 0o600)

    def test_concurrent_native_threads_keep_complete_json_lines(self):
        from concurrent.futures import ThreadPoolExecutor
        with tempfile.TemporaryDirectory() as temp, patch.dict(os.environ, {p.PATH: str(Path(temp)/"profile")}):
            with ThreadPoolExecutor(max_workers=2) as executor:
                list(executor.map(lambda i: p._write({"i": i, "padding": "x"*12000}), range(20)))
            records = [json.loads(line) for line in Path(p.status()["journal"]).read_text().splitlines()]
            self.assertEqual(sorted(r["i"] for r in records), list(range(20)))


if __name__ == "__main__":
    unittest.main()
