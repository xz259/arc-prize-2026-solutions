"""Real PRO-LONG loop with an injectable CPU-only session-server fixture."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import Mock, patch

import requests

from append_only_history import AppendOnlyViolation
from game_allocation import Allocation
from gameplay_config import ContextPolicy
from history_budget import ContextBudgetError
from persistent_harness_adapter import (HTTPPersistentTransport, PersistentProlongAgent,
                                       PersistentProlongSolver, PersistentProtocolError,
                                       configure_persistent_solver, enable_persistent_state)
from prolong_duck import ProlongAgent, ProlongSolver
from prolong_program_suffix import BASE_KEY as PROGRAM_BASE
from prolong_snapshots import BASE_KEY as SNAPSHOT_BASE
import test_prolong as fixtures
from test_harness_v2 import call, response


class FixtureTransport:
    def __init__(self, decoder, replies):
        self.decoder = decoder
        self.replies = list(replies)
        self.calls = []
        self.closed = []
        self.shutdown_count = 0
        self.revision = 0
        self.assistant_ids = []
        self.session_id = None
        self.transform = lambda payload: payload

    def _reply(self, *, continuation=False):
        reply = self.replies.pop(0)
        if isinstance(reply, BaseException):
            raise reply
        self.revision += 1
        ids = getattr(reply, "token_ids", None)
        if ids is None:
            ids = self.decoder.encode(reply.raw_text, add_special_tokens=False).ids
        previous = self.assistant_ids if continuation else []
        self.assistant_ids = previous + ids
        prompt = 100 + self.revision * 10
        persistent = dict(session_id=self.session_id, revision=self.revision,
                          resident_tokens=prompt+len(ids), appended_prompt_tokens=10,
                          prefix_sha256=hashlib.sha256(str(self.revision).encode()).hexdigest(),
                          token_ids=ids)
        policy=copy.deepcopy(self.calls[0][1]["history_eviction"])
        policy.setdefault("resident_token_budget",40000)
        persistent["history_eviction"]=dict(policy,triggered=False,target_tokens=0,history_removed_tokens=0)
        sidecar = self.calls[-1][1]["managed_suffixes"]
        persistent["managed_suffixes"] = dict(
            registered_observation_ids=[d["observation_id"] for d in sidecar["observations"]],
            evicted_observation_ids=sidecar["evict_observation_ids"])
        choice = dict(message=copy.deepcopy(reply.message), finish_reason=reply.finish_reason, token_ids=ids)
        if continuation:
            raw = self.decoder.decode(self.assistant_ids, skip_special_tokens=False)
            choice["message"]["reasoning"] = raw.split("</think>", 1)[0]
            choice["persistent_parse"] = dict(scope="full_assistant", prefix_tokens=len(previous),
                                               assistant_token_ids=self.assistant_ids, raw_text=raw)
        return self.transform(dict(choices=[choice],
                            usage=dict(prompt_tokens=prompt, completion_tokens=len(ids),
                                       total_tokens=prompt+len(ids)), persistent=persistent))

    def start(self, payload, *, timeout):
        self.session_id = payload["session_id"]
        self.calls.append(("start", copy.deepcopy(payload)))
        return self._reply()

    def append(self, session_id, payload, *, timeout):
        if session_id != self.session_id:
            raise AssertionError("Wrong session")
        if payload["expected_revision"] != self.revision:
            raise AssertionError("Wrong revision")
        self.calls.append(("append", copy.deepcopy(payload)))
        return self._reply(continuation=payload["continuation"])

    def close(self, session_id, *, timeout):
        self.closed.append(session_id)
        return dict(closed=True, session_id=session_id)

    def shutdown(self):
        self.shutdown_count += 1


def latest_observation(messages):
    return next(m for m in reversed(messages) if m.get("_arc3_observation"))


class AdapterTests(unittest.TestCase):
    def make_agent(self, tmp, replies, *, history_eviction=None):
        agent, path = fixtures.ProlongTests().make_agent(tmp)
        agent._context_policy = ContextPolicy(40000, 4000, response_cap=None)
        agent.context_limit = 40000
        agent._timeout = 10
        agent._prolong_snapshot_enabled = True
        transport = FixtureTransport(agent.output_decoder, replies)
        enable_persistent_state(agent, transport=transport, session_id="fixture-session",
                                history_eviction=history_eviction)
        return agent, path, transport

    def raw_events(self, tmp):
        path = Path(tmp) / "raw-harness/game-000.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()]

    def messages(self, agent, path):
        frame, _ = __import__("inference.agent.runtime_state", fromlist=["load_runtime_state"]).load_runtime_state(path)
        obs = agent._build_user_message("Current board", frame)
        obs["_arc3_state_key"] = "one"
        return [dict(role="system", content=agent._system_prompt), obs]

    def test_actual_tool_loop_sends_one_initial_history_then_tool_delta(self):
        replies = [response([call("first", "def inspect_pair(x): return x * 2\nprint(inspect_pair(3))")]),
                   response([call("second", "print('done')")])]
        with tempfile.TemporaryDirectory() as tmp:
            agent, path, transport = self.make_agent(tmp, replies)
            agent._tool_steps = 2
            # Any inherited full-history tokenization or ordinary HTTP path would
            # fail immediately. Actual Python tool execution remains enabled.
            agent.token_counter.count = Mock(side_effect=AssertionError("stateless count called"))
            with patch.object(agent, "_chat_transport", side_effect=AssertionError("stateless HTTP called")):
                outcome = agent.analyze(path, 0, valid_actions=["RIGHT"])
            self.assertFalse(outcome.step_executed)
            self.assertEqual([op for op, _ in transport.calls], ["start", "append"])
            initial = transport.calls[0][1]["chat"]["messages"]
            delta = transport.calls[1][1]["external_messages"]
            self.assertEqual([m["role"] for m in delta], ["tool"])
            self.assertEqual(transport.calls[1][1]["managed_suffixes"],
                             dict(version=1, observations=[], evict_observation_ids=[]))
            self.assertNotIn("managed_suffixes", transport.calls[0][1]["chat"])
            self.assertNotIn("observation_id", json.dumps(initial))
            self.assertNotIn("_arc3_", json.dumps(transport.calls))
            self.assertNotIn("inspect_pair", json.dumps(initial[-1]["content"]))
            self.assertIn("inspect_pair", agent.workspace.program_index.text())
            local_obs = latest_observation(agent._history_messages)
            self.assertNotIn("inspect_pair", json.dumps(local_obs["content"]))
            self.assertIn(PROGRAM_BASE, local_obs)
            self.assertIn(SNAPSHOT_BASE, local_obs)
            self.assertGreater(agent.generated_tokens, 0)
            self.assertEqual(agent.generated_tokens, agent.allocation.rows["game"]["generated_tokens"])
            self.assertEqual(agent._persistent_revision, 2)
            events = self.raw_events(tmp)
            self.assertEqual(sum(e["kind"] == "persistent-request" for e in events), 2)
            self.assertEqual(sum(e["kind"] == "persistent-response" for e in events), 2)
            self.assertEqual(sum(e["kind"] == "model-turn" for e in events), 2)
            agent._close_chat_transport()
            self.assertEqual(transport.closed, ["fixture-session"])

    def test_new_observation_gets_updated_library_without_retiring_old_copy(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent, path, transport = self.make_agent(tmp, [response(), response()])
            messages = self.messages(agent, path)
            first = agent._chat_completion(messages, tools=[])
            old = copy.deepcopy(messages[1])
            with patch.object(agent.workspace.program_index, "text", return_value="PROGRAMS: later_helper"):
                new = dict(agent._build_user_message("Next board", None), _arc3_state_key="two")
                messages += [dict(role="assistant", content=first.message["content"],
                                  reasoning=first.message["reasoning"]), new]
                agent._chat_completion(messages, tools=[])
            self.assertEqual(messages[1], old)
            self.assertIn("later_helper", json.dumps(messages[-1]["content"]))
            delta = transport.calls[-1][1]["external_messages"]
            self.assertEqual([m["role"] for m in delta], ["user"])
            self.assertNotIn("assistant", [m["role"] for m in delta])
            self.assertEqual(transport.calls[-1][1]["managed_suffixes"]["evict_observation_ids"],
                             ["obs-00000001"])
            self.assertEqual(agent._persistent_external_state.live_observation_ids(), ["obs-00000002"])

    def test_repeated_unfinished_reasoning_uses_empty_input_continuation(self):
        replies = [response(reasoning="First ", raw="First ", length=True),
                   response(reasoning="second ", raw="second ", length=True),
                   response([call("done", "print(1)")], reasoning="third")]
        with tempfile.TemporaryDirectory() as tmp:
            agent, path, transport = self.make_agent(tmp, replies)
            agent._tool_steps = 3
            agent.analyze(path, 0, valid_actions=["RIGHT"])
            self.assertEqual(len(transport.calls), 3)
            for _, body in transport.calls[1:]:
                self.assertEqual(body["external_messages"], [])
                self.assertEqual(body["managed_suffixes"],
                                 dict(version=1, observations=[], evict_observation_ids=[]))
                self.assertTrue(body["continuation"])
            assistant = next(m for m in reversed(agent._history_messages) if m["role"] == "assistant")
            self.assertEqual(assistant["reasoning"], "First second third")
            self.assertEqual(agent._persistent_revision, 3)

    def test_normalized_assistant_prose_never_reenters_wire(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent, path, transport = self.make_agent(tmp, [response(), response()])
            messages = self.messages(agent, path)
            agent._chat_completion(messages, tools=[])
            messages += [dict(role="assistant", content="rewritten parsed view", reasoning="normalized"),
                         dict(role="user", content="Use a tool.", _arc3_recovery=True)]
            agent._chat_completion(messages, tools=[])
            self.assertEqual(transport.calls[-1][1]["external_messages"],
                             [dict(role="user", content="Use a tool.")])

    def test_split_unicode_tokens_are_reconstructed_from_cumulative_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent, path, transport = self.make_agent(tmp, [])
            symbol = "𐍈"
            ids = agent.output_decoder.encode(symbol, add_special_tokens=False).ids
            self.assertGreater(len(ids), 1, "Fixture requires a split UTF-8 code point")
            first = response(reasoning=agent.output_decoder.decode(ids[:1]),
                             raw=agent.output_decoder.decode(ids[:1]), length=True)
            first.token_ids = ids[:1]
            second = response(reasoning=agent.output_decoder.decode(ids[1:]),
                              raw=agent.output_decoder.decode(ids[1:]), length=True)
            second.token_ids = ids[1:]
            transport.replies = [first, second]
            agent._tool_steps = 2
            agent.analyze(path, 0, valid_actions=["RIGHT"])
            assistant = agent._history_messages[-1]
            self.assertEqual(assistant["reasoning"], symbol)
            self.assertEqual(agent._persistent_assistant_token_ids, ids)

    def test_cumulative_assistant_receipt_mismatch_fails_before_dispatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent, path, transport = self.make_agent(tmp, [
                response(reasoning="First ", raw="First ", length=True),
                response(reasoning="next", raw="next", length=True)])
            messages = self.messages(agent, path)
            agent._chat_completion(messages, tools=[])
            messages += [dict(role="assistant", content=None, reasoning="First ", _arc3_open_reasoning=True)]
            def corrupt(payload):
                payload["choices"][0]["persistent_parse"]["assistant_token_ids"] = [42]
                return payload
            transport.transform = corrupt
            with self.assertRaisesRegex(PersistentProtocolError, "Cumulative assistant"):
                agent._chat_completion(messages, tools=[])

    def test_no_input_after_normal_stop_refused(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent, path, transport = self.make_agent(tmp, [response()])
            messages = self.messages(agent, path)
            agent._chat_completion(messages, tools=[])
            with self.assertRaisesRegex(AppendOnlyViolation, "verified length pause"):
                agent._chat_completion(messages, tools=[])
            self.assertEqual(len(transport.calls), 1)
            self.assertEqual(transport.closed, ["fixture-session"])

    def test_system_or_tools_change_rejected_before_server_append(self):
        for change in ("system", "tools"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as tmp:
                agent, path, transport = self.make_agent(tmp, [response()])
                messages = self.messages(agent, path)
                agent._chat_completion(messages, tools=[])
                messages += [dict(role="tool", tool_call_id="c", content="result")]
                if change == "system":
                    messages[0]["content"] += " amended"
                with self.assertRaises(AppendOnlyViolation):
                    agent._chat_completion(messages, tools=[dict(type="function")] if change == "tools" else [])
                self.assertEqual(len(transport.calls), 1)

    def test_no_implicit_context_trim_or_force_reduction(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent, path, transport = self.make_agent(tmp, [])
            messages = self.messages(agent, path)
            original = copy.deepcopy(messages)
            for method in (agent._trim_messages_for_context, agent._force_reduce_messages):
                with self.assertRaises(ContextBudgetError):
                    method(messages)
            self.assertEqual(messages, original)
            self.assertFalse(transport.calls)

    def test_server_capacity_rejection_closes_session_without_client_trimming(self):
        error = PersistentProtocolError("422 Client Error | capacity exceeded")
        with tempfile.TemporaryDirectory() as tmp:
            agent, path, transport = self.make_agent(tmp, [response(), error])
            messages = self.messages(agent, path)
            agent._chat_completion(messages, tools=[])
            previous = agent._persistent_external_state.committed_messages()
            messages += [dict(role="tool", tool_call_id="c", content="result")]
            with self.assertRaises(PersistentProtocolError):
                agent._chat_completion(messages, tools=[])
            self.assertEqual(agent._persistent_external_state.committed_messages(), previous)
            self.assertEqual(agent._persistent_revision, 1)
            self.assertEqual(transport.closed, ["fixture-session"])
            self.assertEqual(agent.allocation.rows["game"]["uncertain_requests"], 0)

    def test_logical_history_above_context_uses_resident_capacity(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent, path, transport = self.make_agent(tmp, [response()])
            def long_logical(payload):
                payload["persistent"]["logical_tokens"] = 60000
                payload["usage"]["logical_input_tokens"] = 59900
                return payload
            transport.transform = long_logical
            result = agent._chat_completion(self.messages(agent, path), tools=[])
            self.assertLess(result.persistent["resident_tokens"], 40000)
            self.assertEqual(result.persistent["logical_tokens"], 60000)
            self.assertEqual(agent._persistent_revision, 1)

    def test_wrong_eviction_ack_does_not_commit_new_owner(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent, path, transport = self.make_agent(tmp, [response(), response()])
            messages = self.messages(agent, path)
            agent._chat_completion(messages, tools=[])
            previous = agent._persistent_external_state.committed_messages()
            messages += [dict(agent._build_user_message("Next board", None), _arc3_state_key="two")]
            def wrong_ack(payload):
                payload["persistent"]["managed_suffixes"]["evicted_observation_ids"] = []
                return payload
            transport.transform = wrong_ack
            with self.assertRaisesRegex(AppendOnlyViolation, "acknowledgement"):
                agent._chat_completion(messages, tools=[])
            self.assertEqual(agent._persistent_external_state.committed_messages(), previous)
            self.assertEqual(agent._persistent_external_state.live_observation_ids(), ["obs-00000001"])
            self.assertEqual(agent._persistent_revision, 1)
            self.assertEqual(transport.closed, ["fixture-session"])

    def test_lost_response_is_not_retried_or_committed_and_owned_session_closes(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent, path, transport = self.make_agent(tmp, [requests.Timeout("lost response")])
            with self.assertRaises(requests.Timeout):
                agent._chat_completion(self.messages(agent, path), tools=[])
            self.assertEqual(len(transport.calls), 1)
            self.assertEqual(transport.closed, ["fixture-session"])
            self.assertEqual(agent._persistent_revision, 0)
            self.assertEqual(agent._persistent_external_state.committed_messages(), [])
            self.assertEqual(agent.allocation.rows["game"]["uncertain_requests"], 1)
            self.assertEqual(agent.allocation.rows["game"]["generated_tokens"], 0)

    def test_corrupt_receipt_is_rejected_and_received_usage_is_accounted(self):
        for key, value in (("session_id", "other"), ("revision", 9),
                           ("prefix_sha256", "bad"), ("resident_tokens", 40001),
                           ("token_ids", [99]), ("appended_prompt_tokens", -1)):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as tmp:
                agent, path, transport = self.make_agent(tmp, [response()])
                def corrupt(payload):
                    payload["persistent"][key] = value
                    return payload
                transport.transform = corrupt
                with self.assertRaises(PersistentProtocolError):
                    agent._chat_completion(self.messages(agent, path), tools=[])
                self.assertEqual(agent._persistent_revision, 0)
                self.assertGreater(agent.allocation.rows["game"]["generated_tokens"], 0)
                self.assertEqual(agent.generated_tokens, agent.allocation.rows["game"]["generated_tokens"])
                self.assertEqual(transport.closed, ["fixture-session"])

    def test_uncapped_gameplay_omits_artificial_response_limit(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent, path, transport = self.make_agent(tmp, [response()])
            agent.allocation = Allocation(["game"], slots=1, deadline=time.time()+30,
                                          output=tmp, unlimited_tokens=True)
            agent._chat_completion(self.messages(agent, path), tools=[])
            self.assertNotIn("max_tokens", transport.calls[0][1]["chat"])

    def test_finite_remaining_budget_still_caps_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent, path, transport = self.make_agent(tmp, [response()])
            agent._chat_completion(self.messages(agent, path), tools=[])
            self.assertEqual(transport.calls[0][1]["chat"]["max_tokens"], 1000)

    def test_close_is_idempotent_and_no_request_needs_no_server_close(self):
        with tempfile.TemporaryDirectory() as tmp:
            agent, path, transport = self.make_agent(tmp, [])
            agent.close_persistent_state()
            agent.close_persistent_state()
            self.assertFalse(transport.closed)
            self.assertTrue(agent._persistent_closed)

    def test_solver_configuration_is_local_and_explicit(self):
        solver = object.__new__(ProlongSolver)
        result = configure_persistent_solver(solver, base_url="http://127.0.0.1:1234")
        self.assertIs(result, solver)
        self.assertIs(type(solver), PersistentProlongSolver)
        self.assertIs(ProlongSolver._play_one.__globals__["AllocatedAgent"], ProlongAgent)
        self.assertIs(PersistentProlongSolver._play_one.__globals__["AllocatedAgent"], PersistentProlongAgent)


class HTTPTransportTests(unittest.TestCase):
    def test_routes_payloads_and_response_close(self):
        response = Mock()
        response.json.return_value = {"ok": True}
        http = Mock()
        http.post.return_value = response
        http.get.return_value = response
        transport = HTTPPersistentTransport("http://localhost:1234/v1", http=http)
        transport.start({"session_id": "s"}, timeout=3)
        transport.append("s/1", {"expected_revision": 1}, timeout=3)
        transport.status("s/1")
        transport.close("s/1")
        self.assertEqual(http.post.call_args_list[0].args[0], "http://localhost:1234/arc3/persistent/start")
        self.assertEqual(http.post.call_args_list[1].args[0], "http://localhost:1234/arc3/persistent/s%2F1/append")
        self.assertEqual(http.get.call_args.args[0], "http://localhost:1234/arc3/persistent/s%2F1/status")
        self.assertEqual(response.close.call_count, 4)

    def test_http_error_preserves_status_and_body_and_closes_response(self):
        response = Mock(text="capacity exceeded")
        response.raise_for_status.side_effect = requests.HTTPError("422 Client Error")
        http = Mock()
        http.post.return_value = response
        transport = HTTPPersistentTransport("http://localhost:1234", http=http)
        with self.assertRaisesRegex(PersistentProtocolError, "422 Client Error.*capacity exceeded"):
            transport.start({}, timeout=3)
        response.close.assert_called_once()


class EntrypointTests(unittest.TestCase):
    def test_bounded_scope_accepts_specific_games_and_adds_mtp_zero(self):
        from run_persistent_gameplay import validate_scope
        argv = ["--games", "a,b", "--total-seconds", "300", "--clock-mode", "gameplay",
                "--no-token-limit", "--no-response-cap", "--output", "/tmp/example"]
        self.assertEqual(validate_scope(argv)[-2:], ["--expected-mtp", "0"])

    def test_unbounded_or_unsupported_scope_is_rejected(self):
        from run_persistent_gameplay import validate_scope
        base = ["--games", "a,b", "--total-seconds", "300", "--clock-mode", "gameplay",
                "--no-token-limit", "--no-response-cap", "--output", "/tmp/example"]
        for flag, value in (("--games", "all"), ("--games", "a,a"),
                            ("--total-seconds", "601"), ("--total-seconds", "nan"),
                            ("--clock-mode", "entry"), ("--expected-mtp", "1"),
                            ("--replicas", "2")):
            argv = list(base)
            if flag in argv:
                argv[argv.index(flag)+1] = value
            else:
                argv += [flag, value]
            with self.subTest(flag=flag, value=value), self.assertRaises(ValueError):
                validate_scope(argv)

    def test_settings_make_unused_text_headroom_and_persistent_wire_explicit(self):
        from run_persistent_gameplay import apply_persistent_settings
        settings = apply_persistent_settings({}, base_url="http://local")
        self.assertFalse(settings["persistent_state"]["text_truncation"])
        self.assertFalse(settings["persistent_state"]["response_headroom_applied"])
        self.assertIn("external message deltas", settings["transport"])

    def test_persistent_validation_requires_matching_close_receipts(self):
        from run_persistent_gameplay import write_persistent_validation
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root/"raw-harness").mkdir()
            (root/"allocation-summary.json").write_text(json.dumps(dict(
                games=[dict(game_id="g", stop_reason=None)], accounting_valid=True, active_requests=0)))
            path = root/"raw-harness/game-000.jsonl"
            start = dict(kind="persistent-request", payload=dict(route="/arc3/persistent/start",
                                                                 body=dict(session_id="s")))
            path.write_text(json.dumps(start)+"\n")
            result = write_persistent_validation(root, runner_completed=True)
            self.assertFalse(result["valid"])
            self.assertEqual(result["unclosed_sessions"], ["s"])
            with path.open("a") as stream:
                stream.write(json.dumps(dict(kind="persistent-close",
                                             payload=dict(session_id="s", state="closed")))+"\n")
            self.assertTrue(write_persistent_validation(root, runner_completed=True)["valid"])


if __name__ == "__main__":
    unittest.main()
