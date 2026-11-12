"""CPU contract tests for the isolated persistent-state runtime patch.

These tests exercise exact token/MM bookkeeping, mutation-free rejection,
source guards, actual hook installation and asynchronous boundary invariants.
They do not substitute for the required CUDA QSA/GDN state continuation tests.
"""
from __future__ import annotations

import ast
import copy
import enum
import hashlib
import json
import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from types import ModuleType, SimpleNamespace as NS
import sys
import unittest
from unittest.mock import patch

import engine_patch as ep

ROOT = Path(os.environ.get("ARC3_RUNTIME_SOURCE",str(
    Path(__file__).resolve().parent.parent / "persistent_state_20260912" / "source-snapshot" / "vllm")))


class Status(enum.IntEnum):
    WAITING = 0
    RUNNING = 1
    WAITING_FOR_STREAMING_REQ = 2
    FINISHED_STOPPED = 3
    FINISHED_ABORTED = 4


class OutputKind(enum.IntEnum):
    DELTA = 1
    CUMULATIVE = 2


def params(**changes):
    data = dict(extra_args={ep.MARKER: True}, n=1, output_kind=OutputKind.DELTA,
                detokenize=False, stop=[], structured_outputs=None, max_tokens=7)
    data.update(changes)
    return NS(**data)


def config(**changes):
    data = dict(speculative_config=None, num_speculative_tokens=0,
                cache_config=NS(enable_prefix_caching=False),
                parallel_config=NS(pipeline_parallel_size=1, tensor_parallel_size=1, data_parallel_size=1),
                model_config=NS(runner_type="generate", max_model_len=100),
                scheduler_config=NS(async_scheduling=True), use_v2_model_runner=True)
    data.update(changes)
    return NS(**data)


@dataclass
class Position:
    offset: int
    length: int


@dataclass
class Feature:
    mm_position: Position
    identifier: str = "image"


class Request:
    def __init__(self, tokens=(10, 11), output=(20, 21, 99), *, request_id="game", resumable=True):
        self.request_id = request_id
        self.prompt_token_ids = list(tokens)
        self.num_prompt_tokens = len(tokens)
        self._output_token_ids = list(output)
        self._all_token_ids = list(tokens) + list(output)
        self.num_computed_tokens = len(self._all_token_ids) - 1
        self.num_in_flight_tokens = 0
        self.num_output_placeholders = 0
        self.num_stale_output_tokens = 0
        self.spec_token_ids = []
        self.num_preemptions = 0
        self.sampling_params = params()
        self.max_tokens = 99
        self.stop_reason = 1234
        self.status = Status.WAITING_FOR_STREAMING_REQ
        self.streaming_queue = deque()
        self.mm_features = []
        self.prompt_embeds = None
        self.prompt_is_token_ids = None
        self.is_prefill_chunk = False
        self.next_decode_eligible_step = 0
        self.use_structured_output = False
        self.pooling_params = None
        self.arrival_time = 0.0
        self.resumable = resumable
        self.hash_updates = 0
        self.events = []

    @property
    def num_tokens(self):
        return len(self._all_token_ids)

    def update_block_hashes(self):
        self.hash_updates += 1

    def record_event(self, event):
        self.events.append(event)


@dataclass
class Update:
    prompt_token_ids: list[int]
    mm_features: list[Feature] | None = None
    sampling_params: object = None
    max_tokens: int = 7
    arrival_time: float = 5.0

    def __post_init__(self):
        if self.sampling_params is None:
            self.sampling_params = params(max_tokens=self.max_tokens)

    @classmethod
    def from_request(cls, request):
        if not request.resumable:
            return None
        return cls(request.prompt_token_ids, request.mm_features,
                   request.sampling_params, request.max_tokens, request.arrival_time)


def scheduler():
    return NS(max_model_len=100, num_waiting_for_streaming_input=1, log_stats=True)


def apply(sched, req, delta):
    ep.append_session(sched, req, delta, statuses=Status, queued_event="QUEUED")


class AppendTests(unittest.TestCase):
    def test_retains_final_token_without_claiming_it_was_computed(self):
        req, sched = Request(), scheduler()
        original_ids = req._all_token_ids
        original_prompt = req.prompt_token_ids
        original_output = req._output_token_ids
        apply(sched, req, Update([30, 31]))
        self.assertIs(req._all_token_ids, original_ids)
        self.assertIs(req.prompt_token_ids, original_prompt)
        self.assertIs(req._output_token_ids, original_output)
        self.assertEqual(req._all_token_ids, [10, 11, 20, 21, 99, 30, 31])
        self.assertEqual(req.prompt_token_ids, req._all_token_ids)
        self.assertEqual(req._output_token_ids, [])
        self.assertEqual(req.num_computed_tokens, 4)
        self.assertEqual(req.num_prompt_tokens - req.num_computed_tokens, 3)
        self.assertEqual(req.max_tokens, 7)
        self.assertIsNone(req.stop_reason)
        self.assertEqual(req.events, ["QUEUED"])
        self.assertEqual(sched.num_waiting_for_streaming_input, 0)

    def test_empty_delta_resumes_exactly_one_residual_token(self):
        req = Request(output=(42,))
        apply(scheduler(), req, Update([]))
        self.assertEqual(req.prompt_token_ids, [10, 11, 42])
        self.assertEqual(req.num_computed_tokens, 2)
        self.assertEqual(req._arc3_last_append["pending_prefill_tokens"], 1)
        self.assertEqual(req._arc3_last_append["appended_tokens"], 0)

    def test_multimodal_offset_includes_all_generated_tokens(self):
        req = Request()
        old_feature = Feature(Position(0, 1), "old")
        new_feature = Feature(Position(1, 2), "new")
        req.mm_features = [old_feature]
        delta = Update([30, 31, 32], [new_feature])
        apply(scheduler(), req, delta)
        self.assertEqual(req.mm_features[1].mm_position, Position(6, 2))
        self.assertEqual(new_feature.mm_position, Position(1, 2))
        self.assertIs(req.mm_features[0], old_feature)
        self.assertIsNot(req.mm_features[1], new_feature)

    def test_repeated_turns_do_not_duplicate_prior_output(self):
        req, sched = Request(), scheduler()
        apply(sched, req, Update([30]))
        req._output_token_ids.extend([40, 41])
        req._all_token_ids.extend([40, 41])
        req.num_computed_tokens = req.num_tokens - 1
        req.status = Status.WAITING_FOR_STREAMING_REQ
        sched.num_waiting_for_streaming_input = 1
        apply(sched, req, Update([50]))
        self.assertEqual(req.prompt_token_ids, [10, 11, 20, 21, 99, 30, 40, 41, 50])
        self.assertEqual(req._arc3_persistent_turn, 2)

    def test_rejection_is_atomic(self):
        cases = [
            ("inflight", lambda r,s,u: setattr(r,"num_in_flight_tokens",1)),
            ("placeholder", lambda r,s,u: setattr(r,"num_output_placeholders",1)),
            ("stale", lambda r,s,u: setattr(r,"num_stale_output_tokens",1)),
            ("spec", lambda r,s,u: r.spec_token_ids.append(77)),
            ("preempted", lambda r,s,u: setattr(r,"num_preemptions",1)),
            ("computed", lambda r,s,u: setattr(r,"num_computed_tokens",r.num_tokens)),
            ("context", lambda r,s,u: setattr(s,"max_model_len",r.num_tokens+len(u.prompt_token_ids))),
            ("max_tokens", lambda r,s,u: setattr(u,"max_tokens",12)),
            ("output_room", lambda r,s,u: setattr(s,"max_model_len",10)),
            ("not_paused", lambda r,s,u: setattr(r,"status",Status.RUNNING)),
            ("queued", lambda r,s,u: r.streaming_queue.append(Update([80]))),
            ("counter", lambda r,s,u: setattr(s,"num_waiting_for_streaming_input",0)),
            ("prefix", lambda r,s,u: r.prompt_token_ids.__setitem__(0,999)),
            ("output", lambda r,s,u: r._output_token_ids.__setitem__(0,999)),
            ("image", lambda r,s,u: setattr(u,"mm_features",[Feature(Position(1,10))])),
        ]
        for name, mutate in cases:
            with self.subTest(name=name):
                req, sched, update = Request(), scheduler(), Update([30])
                mutate(req, sched, update)
                before = copy.deepcopy((req.__dict__, sched.__dict__, update.__dict__))
                with self.assertRaises(ep.PersistentStateError):
                    apply(sched, req, update)
                self.assertEqual((req.__dict__, sched.__dict__, update.__dict__), before)

    def test_interleaved_game_state_is_untouched(self):
        first, second = Request(request_id="a"), Request((70,), (80, 81), request_id="b")
        before = copy.deepcopy(second.__dict__)
        apply(scheduler(), first, Update([30]))
        self.assertEqual(second.__dict__, before)


class SourceAndConfigTests(unittest.TestCase):
    def test_fresh_runtime_hashes_match(self):
        self.assertEqual(ep.verify_sources(ROOT), ep.SOURCE_HASHES)

    def test_modified_source_is_rejected(self):
        with TemporaryDirectory() as directory:
            base = Path(directory)
            for name in ep.SOURCE_HASHES:
                path = base / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes((ROOT / name).read_bytes())
            path = base / "v1/request.py"
            path.write_bytes(path.read_bytes() + b"\n# drift\n")
            with self.assertRaisesRegex(ep.PersistentStateError, "Unreviewed runtime source"):
                ep.verify_sources(base)

    def test_schedule_guard_is_single_and_immediately_after_request_lookup(self):
        source = (ROOT / "v1/core/sched/scheduler.py").read_text()
        tree = ep._patched_schedule_ast(source)
        calls = [n for n in ast.walk(tree) if isinstance(n,ast.Call)
                 and isinstance(n.func,ast.Name) and n.func.id=="_arc3_persistent_skip_inflight"]
        self.assertEqual(len(calls), 1)
        matched = [node for node in ast.walk(tree) if isinstance(node, ast.While)
                   and any(isinstance(x,ast.If) and "_arc3_persistent_skip_inflight" in ast.unparse(x.test) for x in node.body)]
        self.assertEqual(ast.unparse(matched[0].body[0]), "request = self.running[req_index]")
        self.assertIn("_arc3_persistent_skip_inflight", ast.unparse(matched[0].body[1]))
        with self.assertRaises(ep.PersistentStateError):
            ep._patched_schedule_ast(source.replace("request = self.running[req_index]", "request = self.running[0]"))

    def test_supported_runners_are_recorded(self):
        for use_v2 in (False, True):
            self.assertEqual(ep.validate_config(config(use_v2_model_runner=use_v2))["model_runner"], "v2" if use_v2 else "v1")

    def test_unsupported_configs_are_rejected(self):
        invalid = [config(speculative_config=object()), config(num_speculative_tokens=1),
                   config(cache_config=NS(enable_prefix_caching=True)),
                   config(parallel_config=NS(pipeline_parallel_size=2,tensor_parallel_size=1)),
                   config(parallel_config=NS(pipeline_parallel_size=1,tensor_parallel_size=2)),
                   config(kv_transfer_config=object()), config(ec_transfer_config=object())]
        for item in invalid:
            with self.assertRaises(ep.PersistentStateError): ep.validate_config(item)

    def test_raw_sampling_contract(self):
        ep.validate_sampling(params())
        for change in ({"detokenize":True}, {"n":2}, {"stop":["stop"]},
                       {"output_kind":OutputKind.CUMULATIVE}, {"max_tokens":None},
                       {"structured_outputs":object()}, {"extra_args":{}}):
            with self.assertRaises(ep.PersistentStateError): ep.validate_sampling(params(**change))


# The installation tests execute the actual pinned stock lifecycle methods with
# lightweight CPU Request/queue/cache stand-ins. No vLLM import or CUDA runtime is
# required; the wrappers and marked/unmarked dispatch are the real patch code.
def stock_method(name, globals_):
    tree = ast.parse((ROOT / "v1/core/sched/scheduler.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "Scheduler")
    method = copy.deepcopy(next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name==name))
    method.decorator_list=[]
    code = compile(ast.fix_missing_locations(ast.Module(body=[method], type_ignores=[])), "pinned-scheduler:"+name, "exec", flags=__import__("__future__").annotations.compiler_flag)
    namespace = dict(globals_)
    exec(code,namespace)
    return namespace[name]


class HookInstallationTests(unittest.TestCase):
    def setUp(self):
        self.old_receipt = ep._INSTALL_RECEIPT
        ep._INSTALL_RECEIPT = None
        sm = ModuleType("vllm.v1.core.sched.scheduler")
        sm.RequestStatus=Status
        sm.EngineCoreEventType=NS(QUEUED="QUEUED")
        sm.StreamingUpdate=Update
        sm.deque=deque
        sm.replace=__import__("dataclasses").replace
        class Scheduler:
            def __init__(self):
                self.requests={}
                self.waiting=[]
                self.skipped_waiting=[]
                self.running=[]
                self.max_model_len=100
                self.num_waiting_for_streaming_input=0
                self.vllm_config=config()
                self.log_stats=False
                self.connector=None
                self.finished_req_ids=set()
                self.cache_blocks={}
                self.kv_cache_manager=NS(get_blocks=lambda rid: NS(get_block_ids=lambda: self.cache_blocks.get(rid, ([],[]))))
            def _enqueue_waiting_request(self, request):
                self.waiting.append(request)
            def finish_requests(self, request_id, status):
                request=self.requests[request_id]
                request.status=status
                self._free_request(request)
            def _free_request(self,request,*args,**kwargs):
                self.requests.pop(request.request_id,None)
                self.cache_blocks.pop(request.request_id,None)
                self.finished_req_ids.add(request.request_id)
                return None
            def _preempt_request(self,request,*args,**kwargs):
                request.num_preemptions+=1
                request.num_computed_tokens=0
                self.cache_blocks.pop(request.request_id,None)
            def _update_after_schedule(self,output):
                for rid,count in output.num_scheduled_tokens.items():
                    request=self.requests[rid]
                    request.num_computed_tokens+=count
                    request.num_in_flight_tokens+=count
        sm.Scheduler=Scheduler
        for name in ("add_request", "_handle_stopped_request", "_update_request_as_session"):
            setattr(Scheduler,name,stock_method(name, sm.__dict__))
        # This method is replaced by the compiled pinned-source schedule hook.
        Scheduler.schedule=lambda self,*a,**kw:None
        class InputProcessor:
            def process_inputs(self, request_id, prompt, params, *, resumable=False):
                self._validate_prompt_len(len(prompt),"decoder")
                if request_id=="raise": raise ValueError("fixture")
                return list(prompt)
            def _validate_prompt_len(self, prompt_len,prompt_type):
                if prompt_len==0: raise ValueError("empty")
        im=ModuleType("vllm.v1.engine.input_processor")
        im.InputProcessor=InputProcessor
        modules={name:ModuleType(name) for name in ("vllm","vllm.v1","vllm.v1.core","vllm.v1.core.sched","vllm.v1.engine")}
        modules["vllm"].__file__=str(ROOT/"__init__.py")
        modules["vllm.v1.core.sched"].scheduler=sm
        modules[sm.__name__]=sm
        modules[im.__name__]=im
        self.module_patch=patch.dict(sys.modules,modules)
        self.module_patch.start()
        # These remain phase-one lifecycle regression tests. The additional
        # phase-two component ordering and transaction hooks have separate tests.
        self.component_patch=patch.object(ep,"_install_eviction_components",lambda receipt:receipt)
        self.component_patch.start()
        self.Scheduler=Scheduler
        self.InputProcessor=InputProcessor
        self.receipt=ep.install()
        self.sched=Scheduler()

    def tearDown(self):
        self.component_patch.stop()
        self.module_patch.stop()
        ep._INSTALL_RECEIPT=self.old_receipt

    def admit(self):
        request=Request(output=())
        request.num_computed_tokens=0
        request.max_tokens=7
        request.status=Status.WAITING
        self.sched.add_request(request)
        self.sched.cache_blocks[request.request_id]=([11,12],[20])
        return request

    def finish_turn(self,request):
        request._output_token_ids.extend([30,99])
        request._all_token_ids.extend([30,99])
        request.num_computed_tokens=request.num_tokens-1
        request.status=Status.FINISHED_STOPPED
        self.assertFalse(self.sched._handle_stopped_request(request))

    def test_install_is_idempotent(self):
        function=self.Scheduler.schedule
        self.assertEqual(ep.install(),self.receipt)
        self.assertIs(self.Scheduler.schedule,function)

    def test_stock_pause_keeps_all_cache_group_ownership(self):
        request=self.admit()
        self.finish_turn(request)
        self.assertIn(request.request_id,self.sched.requests)
        self.assertEqual(self.sched.cache_blocks[request.request_id],([11,12],[20]))
        self.assertEqual(request.status,Status.WAITING_FOR_STREAMING_REQ)
        self.assertEqual(self.sched.num_waiting_for_streaming_input,1)
        self.assertNotIn(request.request_id,self.sched.finished_req_ids)

    def test_marked_append_and_close_use_stock_lifecycle(self):
        request=self.admit()
        self.finish_turn(request)
        update=Request((70,),(),request_id=request.request_id)
        update.max_tokens=7
        self.sched.add_request(update)
        self.assertIs(self.sched.requests[request.request_id],request)
        self.assertEqual(request.prompt_token_ids,[10,11,30,99,70])
        self.assertEqual(self.sched.cache_blocks[request.request_id],([11,12],[20]))
        # Closing while paused uses the existing abort/free path.
        request.status=Status.WAITING_FOR_STREAMING_REQ
        self.sched.num_waiting_for_streaming_input=1
        self.sched.add_request(Request((0,),(),request_id=request.request_id,resumable=False))
        self.assertNotIn(request.request_id,self.sched.requests)
        self.assertNotIn(request.request_id,self.sched.cache_blocks)
        self.assertIn(request.request_id,self.sched.finished_req_ids)

    def test_no_early_append_queue_or_marker_loss(self):
        request=self.admit()
        for marker in (True,False):
            update=Request((70,),(),request_id=request.request_id)
            update.max_tokens=7
            if not marker: update.sampling_params.extra_args={}
            with self.assertRaises(ep.PersistentStateError):self.sched.add_request(update)
        self.assertFalse(request.streaming_queue)

    def test_async_inflight_step_cannot_cross_pause(self):
        request=self.admit()
        request.status=Status.RUNNING
        self.sched._update_after_schedule(NS(num_scheduled_tokens={request.request_id:2}))
        self.assertEqual(request.num_in_flight_tokens,2)
        self.assertTrue(ep.should_skip_inflight(request))
        with self.assertRaises(ep.PersistentStateError):
            self.sched._update_after_schedule(NS(num_scheduled_tokens={request.request_id:1}))
        with self.assertRaises(ep.PersistentStateError):self.sched._handle_stopped_request(request)
        # Acknowledge the step and sample its final token before pausing.
        request.num_in_flight_tokens=0
        request._output_token_ids.append(99)
        request._all_token_ids.append(99)
        request.status=Status.FINISHED_STOPPED
        self.assertFalse(self.sched._handle_stopped_request(request))
        self.assertEqual(request.num_tokens-request.num_computed_tokens,1)
        self.assertEqual(request._arc3_turn_work["prefill_tokens"],2)

    def test_guard_does_not_change_unmarked_or_nonstreaming_requests(self):
        for resumable, marked in ((False,True),(True,False)):
            request=Request(resumable=resumable)
            request.num_in_flight_tokens=1
            if not marked:request.sampling_params.extra_args={}
            self.assertFalse(ep.should_skip_inflight(request))

    def test_empty_delta_allowance_is_narrow_and_context_is_restored(self):
        processor=self.InputProcessor()
        self.assertEqual(processor.process_inputs("resume",[],params(),resumable=True),[])
        for resumable,p in ((False,params()),(True,params(extra_args={}))):
            with self.assertRaises(ValueError):processor.process_inputs("ordinary",[],p,resumable=resumable)
        with self.assertRaises(ValueError):processor.process_inputs("raise",[],params(),resumable=True)
        with self.assertRaises(ValueError):processor._validate_prompt_len(0,"decoder")
        with self.assertRaises(ValueError):processor._validate_prompt_len(0,"encoder")
        fresh=Request((),())
        fresh.max_tokens=7
        with self.assertRaises(ep.PersistentStateError):self.sched.add_request(fresh)

    def test_audit_is_event_based_and_reports_work_not_apc_hits(self):
        with TemporaryDirectory() as directory:
            path=Path(directory)/"audit.jsonl"
            with patch.dict(os.environ,{"ARC3_PERSISTENT_ENGINE_AUDIT":str(path)}):
                request=self.admit()
                self.sched._update_after_schedule(NS(num_scheduled_tokens={request.request_id:2}))
                self.assertEqual(len(path.read_text().splitlines()),1)
                request.num_in_flight_tokens=0
                request._all_token_ids.append(99)
                request._output_token_ids.append(99)
                request.status=Status.FINISHED_STOPPED
                self.sched._handle_stopped_request(request)
                update=Request((70,),(),request_id=request.request_id)
                update.max_tokens=7
                self.sched.add_request(update)
            entries=[json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual([e["event"] for e in entries],["start","pause","resume"])
            self.assertEqual(entries[1]["block_ids"],[[11,12],[20]])
            self.assertEqual(entries[1]["turn_work"]["prefill_tokens"],2)
            self.assertEqual(entries[2]["append"]["pending_prefill_tokens"],2)
            self.assertEqual(entries[1]["tokens_sha256"],hashlib.sha256(b"[10,11,99]").hexdigest())


if __name__ == "__main__":
    unittest.main()
