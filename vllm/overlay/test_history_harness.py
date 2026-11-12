"""History policy wiring on the real local PRO-LONG loop; CPU fixtures only."""
import copy
from contextlib import redirect_stdout, redirect_stderr
import io
import json
import tempfile
import unittest
from unittest.mock import patch

from gameplay_config import ContextPolicy
from append_only_history import AppendOnlyViolation
from persistent_harness_adapter import (HISTORY_MEMORY_REMINDER, PersistentProlongAgent,
    PersistentProtocolError, configure_persistent_solver, history_eviction_config)
from prolong_duck import ProlongSolver
import test_prolong as fixtures
from test_harness_v2 import call,response
import test_persistent_harness_adapter as adapter_fixture
from test_persistent_harness_adapter import FixtureTransport
from run_persistent_gameplay import apply_persistent_settings,main


POLICY={"policy":"oldest_interactions","fraction":0.10}


class HistoryAdapterTests(unittest.TestCase):
    def make(self,tmp,replies,config=None):
        return adapter_fixture.AdapterTests().make_agent(tmp,replies,history_eviction=POLICY if config is None else config)

    def test_start_only_policy_and_once_pinned_system_reminder(self):
        replies=[response([call("first","print(1)")]),response([call("second","print(2)")])]
        with tempfile.TemporaryDirectory() as tmp:
            agent,path,transport=self.make(tmp,replies)
            agent._tool_steps=2
            pinned=agent._system_prompt
            agent.analyze(path,0,valid_actions=["RIGHT"])
            start,append=[body for _,body in transport.calls]
            self.assertEqual(start["history_eviction"],POLICY)
            self.assertNotIn("history_eviction",start["chat"])
            self.assertNotIn("history_eviction",append)
            self.assertEqual(start["chat"]["messages"][0]["content"],pinned)
            self.assertEqual(pinned.count(HISTORY_MEMORY_REMINDER),1)
            self.assertEqual(agent._system_prompt,pinned)
            self.assertEqual([m["role"] for m in append["external_messages"]],["tool"])
            self.assertNotIn(HISTORY_MEMORY_REMINDER,json.dumps(append))
            self.assertEqual(append["managed_suffixes"]["observations"],[])

    def test_off_preserves_production_system_text(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent,path,transport=self.make(tmp,[response()],{"policy":"off","fraction":.10})
            self.assertNotIn(HISTORY_MEMORY_REMINDER,agent._system_prompt)
            before=agent._system_prompt
            agent._chat_completion(adapter_fixture.AdapterTests().messages(agent,path),tools=[])
            self.assertEqual(transport.calls[0][1]["chat"]["messages"][0]["content"],before)
            self.assertEqual(transport.calls[0][1]["history_eviction"]["policy"],"off")

    def test_external_config_mutation_cannot_change_session(self):
        config=copy.deepcopy(POLICY)
        with tempfile.TemporaryDirectory() as tmp:
            agent,path,transport=self.make(tmp,[response()],config)
            config.update(policy="off",fraction=.5)
            messages=adapter_fixture.AdapterTests().messages(agent,path)
            first,_=agent._stage_persistent_messages(messages,[],request_id="staged")
            second,_=agent._stage_persistent_messages(messages,[],request_id="staged")
            self.assertEqual(first.sidecar(),second.sidecar())
            self.assertEqual(first.messages(),second.messages())
            self.assertEqual(json.loads(agent._persistent_history_eviction_json),POLICY)

    def test_policy_receipt_mismatch_closes_session(self):
        for bad in ({"policy":"off"},{"fraction":True},{"fraction":.2},
                    {"resident_token_budget":39000}):
            with self.subTest(bad=bad),tempfile.TemporaryDirectory() as tmp:
                agent,path,transport=self.make(tmp,[response()])
                def corrupt(payload):
                    payload["persistent"]["history_eviction"].update(bad)
                    return payload
                transport.transform=corrupt
                with self.assertRaises(PersistentProtocolError):
                    agent._chat_completion(adapter_fixture.AdapterTests().messages(agent,path),tools=[])
                self.assertEqual(transport.closed,["fixture-session"])
                self.assertEqual(agent._persistent_revision,0)

    def test_missing_receipt_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent,path,transport=self.make(tmp,[response()])
            def corrupt(payload):
                del payload["persistent"]["history_eviction"]
                return payload
            transport.transform=corrupt
            with self.assertRaises(PersistentProtocolError):
                agent._chat_completion(adapter_fixture.AdapterTests().messages(agent,path),tools=[])

    def test_proven_uncommitted_capacity_refusal_has_no_token_charge(self):
        receipt=dict(POLICY,resident_token_budget=40000,triggered=True,
                     eviction_committed=False,capacity_fallback="insufficient_completed_history")
        error=PersistentProtocolError("413 Client Error",status_code=413,error_payload={
            "error":{"type":"persistent_session_error","history_eviction":receipt}})
        with tempfile.TemporaryDirectory() as tmp:
            agent,path,transport=self.make(tmp,[error])
            with self.assertRaises(PersistentProtocolError):
                agent._chat_completion(adapter_fixture.AdapterTests().messages(agent,path),tools=[])
            row=agent.allocation.rows["game"]
            self.assertEqual(row["uncertain_requests"],0)
            self.assertEqual(row["generated_tokens"],0)
            self.assertEqual(row["charged_tokens"],0)
            self.assertEqual(row["stop_reason"],"persistent_capacity")
            self.assertEqual(transport.closed,["fixture-session"])
            event=next(e for e in adapter_fixture.AdapterTests().raw_events(tmp) if e["kind"]=="persistent-error")
            self.assertTrue(event["payload"]["uncommitted_capacity_refusal"])
            self.assertEqual(event["payload"]["body"],error.error_payload)

    def test_unproven_413_keeps_uncertain_accounting(self):
        for receipt in (None,dict(POLICY,resident_token_budget=40000,triggered=True,
                                 eviction_committed=True,capacity_fallback="insufficient_completed_history")):
            error=PersistentProtocolError("413 Client Error",status_code=413,error_payload={
                "error":{"type":"persistent_session_error","history_eviction":receipt}})
            with self.subTest(receipt=receipt),tempfile.TemporaryDirectory() as tmp:
                agent,path,transport=self.make(tmp,[error])
                with self.assertRaises(PersistentProtocolError):
                    agent._chat_completion(adapter_fixture.AdapterTests().messages(agent,path),tools=[])
                self.assertEqual(agent.allocation.rows["game"]["uncertain_requests"],1)
                self.assertGreater(agent.allocation.rows["game"]["charged_tokens"],0)

    def test_diagnostic_budget_is_not_harness_context_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent,path,transport=self.make(tmp,[response()],dict(POLICY,resident_token_budget=8192))
            agent._chat_completion(adapter_fixture.AdapterTests().messages(agent,path),tools=[])
            self.assertEqual(agent.context_policy.context_limit,40000)
            self.assertEqual(transport.calls[0][1]["history_eviction"]["resident_token_budget"],8192)

    def test_history_eviction_receipt_does_not_duplicate_reasoning(self):
        replies=[response(reasoning="One ",raw="One ",length=True),
                 response([call("done","print(1)")],reasoning="two")]
        with tempfile.TemporaryDirectory() as tmp:
            agent,path,transport=self.make(tmp,replies)
            agent._tool_steps=2
            def evicted(payload):
                if payload["persistent"]["revision"]==2:
                    payload["persistent"]["history_eviction"].update(triggered=True,target_tokens=4000,history_removed_tokens=4500)
                return payload
            transport.transform=evicted
            agent.analyze(path,0,valid_actions=["RIGHT"])
            body=transport.calls[1][1]
            self.assertTrue(body["continuation"])
            self.assertEqual(body["external_messages"],[])
            self.assertNotIn("history_eviction",body)
            last=next(m for m in reversed(agent._history_messages) if m["role"]=="assistant")
            self.assertEqual(last["reasoning"],"One two")

    def test_late_system_rewrite_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent,path,transport=self.make(tmp,[response()])
            messages=adapter_fixture.AdapterTests().messages(agent,path)
            agent._chat_completion(messages,tools=[])
            messages[0]["content"]=agent._persistent_original_system_prompt
            with self.assertRaises(AppendOnlyViolation):
                agent._stage_persistent_messages(messages,[],request_id="next")
            self.assertEqual(len(transport.calls),1)

    def test_lazy_initialization_upgrades_only_unsubmitted_base(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent,path=fixtures.ProlongTests().make_agent(tmp)
            agent._context_policy=ContextPolicy(40000,4000,response_cap=None)
            agent.context_limit=40000
            original=agent._system_prompt
            messages=adapter_fixture.AdapterTests().messages(agent,path)
            transport=FixtureTransport(agent.output_decoder,[response()])
            agent.__class__=PersistentProlongAgent
            agent._persistent_history_eviction_requested=copy.deepcopy(POLICY)
            agent._persistent_transport_factory=lambda _:transport
            agent._chat_completion(messages,tools=[])
            self.assertEqual(messages[0]["content"],original+"\n\n"+HISTORY_MEMORY_REMINDER)
            self.assertEqual(messages[0]["content"],agent._system_prompt)

    def test_normal_solver_prompt_configuration_pins_reminder_early(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent,path=fixtures.ProlongTests().make_agent(tmp)
            agent.__class__=PersistentProlongAgent
            agent._persistent_history_eviction_requested=copy.deepcopy(POLICY)
            agent.configure_corrected_prompts()
            self.assertTrue(agent._system_prompt.endswith(HISTORY_MEMORY_REMINDER))
            self.assertEqual(agent._system_prompt.count(HISTORY_MEMORY_REMINDER),1)
            self.assertEqual(adapter_fixture.AdapterTests().messages(agent,path)[0]["content"],agent._system_prompt)

    def test_invalid_policy_is_rejected(self):
        for config in ({"policy":"importance","fraction":.1},{"policy":"off","fraction":False},
                       dict(POLICY,fraction=float('nan')),dict(POLICY,fraction=1),
                       dict(POLICY,resident_token_budget=0),dict(POLICY,unexpected=True)):
            with self.subTest(config=config),self.assertRaises(ValueError):history_eviction_config(config)

    def test_solver_pins_config_without_changing_production_class(self):
        source=copy.deepcopy(POLICY)
        solver=object.__new__(ProlongSolver)
        configure_persistent_solver(solver,base_url="http://local",history_eviction=source)
        source["fraction"]=.5
        self.assertEqual(json.loads(solver.persistent_history_eviction_json),POLICY)
        self.assertNotIn("persistent_history_eviction_json",ProlongSolver.__dict__)


class HistoryRunnerTests(unittest.TestCase):
    BASE=["--persistent-base-url","http://localhost:1234","--games","a,b","--total-seconds","300",
          "--clock-mode","gameplay","--no-token-limit","--no-response-cap","--output","/tmp/no-run"]

    def test_default_and_off_settings(self):
        on=apply_persistent_settings({},base_url="http://local")["persistent_state"]
        off=apply_persistent_settings({},base_url="http://local",history_eviction={"policy":"off","fraction":.2})["persistent_state"]
        self.assertEqual(on["history_eviction"],POLICY)
        self.assertEqual(off["memory_reminder"],"disabled")
        self.assertFalse(on["text_truncation"])

    def test_invalid_cli_fails_before_heavy_import_or_execution(self):
        for flags in (["--kv-evict-fraction","nan"],["--kv-evict-fraction","1"],
                      ["--kv-evict-fraction",".1","--kv-evict-fraction",".2"],
                      ["--kv-eviction-policy","importance"]):
            with self.subTest(flags=flags),patch.dict('sys.modules',{'run_prolong':None}), \
                 redirect_stdout(io.StringIO()),redirect_stderr(io.StringIO()),self.assertRaises(SystemExit):main(self.BASE+flags)


if __name__=='__main__':unittest.main()
