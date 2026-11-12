"""Opt-in PRO-LONG client for persistent serving with managed-suffix eviction.

No import patches production classes. Call enable_persistent_state explicitly on
a newly constructed ProlongAgent. The session server owns raw assistant token
IDs, positional state and hybrid caches; parsed assistants remain dispatch/log
views only. Suffix ownership travels out of band. No capacity rejection is
repaired by text trimming, and no observation is rewritten after submission.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import time
from urllib.parse import quote
import uuid

import requests

from append_only_history import AppendOnlyViolation
from managed_suffix_history import ManagedSuffixMessageState
from game_allocation import AllocationStopped
from gameplay_config import wire_messages
from history_budget import ContextBudgetError
from inference.agent import tool_agent as duck_agent
from prolong_duck import ProlongAgent, ProlongSession, ProlongSolver
from prolong_continuation import open_tail
from prolong_sandbox import with_globals
from prolong_snapshot_config import snapshot_enabled
from scheduled_duck import AllocatedSolver


SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty",
                   "presence_penalty", "frequency_penalty", "max_tokens", "seed")

HISTORY_MEMORY_REMINDER = (
    "Memory reminder: older completed observations and exchanges may no longer be present "
    "in your attention context. Recover earlier findings, programs, board states, and actions "
    "from the program library and game log when needed. Continue solving from the current "
    "board and the retained conversation.")


def history_eviction_config(value=None):
    """Validate a private session policy; no process-global environment setting."""
    if value is None:
        value = {"policy": "off", "fraction": 0.10}
    if not isinstance(value, dict) or set(value)-{"policy", "fraction", "resident_token_budget", "protect_recent_fraction"}:
        raise ValueError("Unknown history-eviction configuration fields")
    policy, fraction = value.get("policy"), value.get("fraction")
    if (policy not in {"off", "oldest_interactions", "qsa_frequency"} or type(fraction) not in (int, float)
            or not math.isfinite(fraction) or not 0 < fraction < 1):
        raise ValueError("Require off/oldest_interactions/qsa_frequency and a finite eviction fraction in (0,1)")
    config = {"policy": policy, "fraction": float(fraction)}
    if "resident_token_budget" in value:
        budget = value["resident_token_budget"]
        if type(budget) is not int or budget <= 0:
            raise ValueError("Diagnostic resident token budget must be a positive integer")
        config["resident_token_budget"] = budget
    if "protect_recent_fraction" in value:
        recent = value["protect_recent_fraction"]
        if type(recent) not in (int, float) or not math.isfinite(recent) or not 0 <= recent < 1:
            raise ValueError("Recent-protection fraction must be finite and in [0, 1)")
        config["protect_recent_fraction"] = float(recent)
    return config


def _pin_memory_reminder(agent, config):
    original = agent._system_prompt
    if not isinstance(original, str):
        raise TypeError("Persistent history control requires a text system prompt")
    if hasattr(agent, "_persistent_pinned_system_prompt"):
        if original != agent._persistent_pinned_system_prompt:
            raise AppendOnlyViolation("Pinned persistent system prompt changed")
        return
    agent._persistent_original_system_prompt = original
    agent._persistent_pinned_system_prompt = original + (
        "\n\n" + HISTORY_MEMORY_REMINDER if config["policy"] in ("oldest_interactions", "qsa_frequency") else "")
    agent._system_prompt = agent._persistent_pinned_system_prompt


class PersistentProtocolError(requests.RequestException):
    def __init__(self, message, *, status_code=None, error_payload=None):
        super().__init__(message)
        self.status_code = status_code
        self.error_payload = copy.deepcopy(error_payload)


class HTTPPersistentTransport:
    """One game owns this synchronous HTTP connection pool and session ID."""

    def __init__(self, base_url, *, headers=None, http=None):
        self.base_url = base_url.rstrip("/").removesuffix("/v1")
        self.headers = dict(headers or {})
        self.http = http or requests.Session()

    @staticmethod
    def session_route(session_id, operation):
        return "/arc3/persistent/" + quote(session_id, safe="") + "/" + operation

    def _post(self, route, payload, timeout):
        response = self.http.post(self.base_url + route, json=payload,
                                  headers=self.headers, timeout=timeout)
        try:
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                try:
                    error_payload=response.json()
                except ValueError:
                    error_payload=None
                raise PersistentProtocolError(f"{exc} | response: {response.text.strip()}",
                    status_code=response.status_code,error_payload=error_payload) from exc
            return response.json()
        finally:
            response.close()

    def start(self, payload, *, timeout):
        return self._post("/arc3/persistent/start", payload, timeout)

    def append(self, session_id, payload, *, timeout):
        return self._post(self.session_route(session_id, "append"), payload, timeout)

    def close(self, session_id, *, timeout=15):
        return self._post(self.session_route(session_id, "close"), {}, timeout)

    def status(self, session_id, *, timeout=15):
        response = self.http.get(self.base_url + self.session_route(session_id, "status"),
                                 headers=self.headers, timeout=timeout)
        try:
            response.raise_for_status()
            return response.json()
        finally:
            response.close()

    def shutdown(self):
        self.http.close()


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def enable_persistent_state(agent, *, transport=None, base_url=None, session_id=None,
                            history_eviction=None):
    """Explicitly enable the experiment on a fresh game agent, never globally."""
    if type(agent) is not ProlongAgent:
        raise TypeError("Persistent control expects a fresh, exact ProlongAgent")
    if getattr(agent, "_history_messages", None):
        raise ValueError("Enable persistence before the game's first model request")
    config = history_eviction_config(history_eviction)
    agent.__class__ = PersistentProlongAgent
    _initialize_persistent_state(agent, transport=transport, base_url=base_url, session_id=session_id,
                                 history_eviction=config)
    return agent


def _initialize_persistent_state(agent, *, transport=None, base_url=None, session_id=None,
                                 history_eviction=None):
    config = history_eviction_config(history_eviction)
    if config.get("resident_token_budget", agent.context_policy.context_limit) > agent.context_policy.context_limit:
        raise ValueError("Diagnostic resident budget exceeds the harness context limit")
    agent._persistent_history_eviction_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
    _pin_memory_reminder(agent, config)
    agent._persistent_transport = transport or HTTPPersistentTransport(
        base_url or agent._model.base_url, headers=agent._headers())
    agent._persistent_session_id = session_id or "arc3-" + uuid.uuid4().hex
    agent._persistent_revision = 0
    agent._persistent_prefix_sha256 = None
    agent._persistent_external_state = ManagedSuffixMessageState(
        program_enabled=getattr(agent, "_prolong_program_index_enabled", None)
            if type(getattr(agent, "_prolong_program_index_enabled", None)) is bool
            else getattr(getattr(agent, "workspace", None), "program_index_enabled", True),
        snapshot_enabled=snapshot_enabled(getattr(agent, "_prolong_snapshot_enabled", None)))
    agent._persistent_attempted = False
    agent._persistent_closed = False
    agent._persistent_last_finish_reason = None
    agent._persistent_assistant_token_ids = []


class PersistentProlongAgent(ProlongAgent):
    def _uncommitted_capacity_refusal(self, exc):
        if not isinstance(exc, PersistentProtocolError) or exc.status_code != 413:
            return False
        error = exc.error_payload.get("error") if isinstance(exc.error_payload, dict) else None
        if not isinstance(error, dict) or error.get("type") != "persistent_session_error":
            return False
        receipt=error.get("history_eviction")
        expected=json.loads(self._persistent_history_eviction_json)
        return (isinstance(receipt,dict) and receipt.get("eviction_committed") is False
                and receipt.get("capacity_fallback") in ("insufficient_completed_history","history_eviction_disabled")
                and receipt.get("triggered") is True and receipt.get("policy")==expected["policy"]
                and type(receipt.get("fraction")) in (int,float) and receipt["fraction"]==expected["fraction"]
                and type(receipt.get("resident_token_budget")) is int
                and receipt["resident_token_budget"]==expected.get("resident_token_budget",self.context_policy.context_limit))

    def configure_corrected_prompts(self, *args, **kwargs):
        if hasattr(self, "_persistent_external_state"):
            raise AppendOnlyViolation("Cannot reconfigure prompts after persistent initialization")
        super().configure_corrected_prompts(*args, **kwargs)
        _pin_memory_reminder(self, history_eviction_config(
            getattr(self, "_persistent_history_eviction_requested", None)))

    def _chat_completion(self, messages, *, tools, request_timeout_seconds=None):
        if not hasattr(self, "_persistent_external_state"):
            factory = getattr(self, "_persistent_transport_factory", None)
            _initialize_persistent_state(self, transport=factory(self) if factory else None,
                                         base_url=getattr(self, "_persistent_base_url", None),
                                         history_eviction=getattr(self, "_persistent_history_eviction_requested", None))
        return super()._chat_completion(messages, tools=tools,
                                        request_timeout_seconds=request_timeout_seconds)

    def _refresh_program_index(self, messages):
        # The inherited construction and tool loop both call this method.
        # Construction stays raw; suffixes are materialized once in staging.
        return messages

    def _trim_messages_for_context(self, messages, **kwargs):
        self._persistent_stop("persistent_capacity")
        raise ContextBudgetError("Persistent suffix-eviction control cannot trim history; close or rebuild explicitly")

    def _force_reduce_messages(self, messages, **kwargs):
        return self._trim_messages_for_context(messages, **kwargs)

    def _persistent_stop(self, reason):
        with self.allocation.cv:
            row = self.allocation.rows[self.game_id]
            row["stop_reason"] = row["stop_reason"] or reason

    def _stage_persistent_messages(self, messages, tools, *, request_id):
        systems = [i for i,m in enumerate(messages) if m.get("role") == "system"]
        if systems != [0]:
            raise AppendOnlyViolation("Persistent input requires one leading pinned system message")
        current = messages[0].get("content")
        pinned = self._persistent_pinned_system_prompt
        if self._system_prompt != pinned:
            raise AppendOnlyViolation("Persistent system prompt changed outside the session")
        if current != pinned:
            # Lazy initialization can run after initial messages were built.
            # Only that initial, unsubmitted base is upgraded; later input must
            # already contain the identical pinned prompt.
            if self._persistent_revision or current != self._persistent_original_system_prompt:
                raise AppendOnlyViolation("Previously submitted system message changed")
            messages[0] = dict(messages[0], content=pinned)
        positions = [i for i, message in enumerate(messages)
                     if message.get("role") in {"system", "user", "tool"}]
        if any(m.get("role") not in {"system", "user", "tool", "assistant"} for m in messages):
            raise AppendOnlyViolation("Unexpected message role in persistent harness history")
        external = [messages[i] for i in positions]
        settings = dict(model=self._model.model_id, tools=tools,
                        chat_template_kwargs=self.gameplay.template_kwargs(),
                        add_special_tokens=False,
                        snapshot_enabled=self._persistent_external_state.snapshot_enabled,
                        program_enabled=self._persistent_external_state.program_enabled,
                        history_eviction=json.loads(self._persistent_history_eviction_json))
        staged = self._persistent_external_state.stage(external, request_id=request_id, workspace=self.workspace,
                                                       renderer_fingerprint=_fingerprint(settings))
        prepared_external = staged.messages()
        for position, message in zip(positions, prepared_external):
            messages[position] = message
        return staged, staged.external_messages()

    def _persistent_result(self, payload, *, continuation=False):
        choices = payload.get("choices")
        persistent = payload.get("persistent")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(persistent, dict):
            raise PersistentProtocolError("Persistent response lacks its single choice/state receipt")
        if persistent.get("session_id") != self._persistent_session_id:
            raise PersistentProtocolError("Persistent response session identity mismatch")
        if persistent.get("revision") != self._persistent_revision + 1:
            raise PersistentProtocolError("Persistent response revision is not the expected successor")
        prefix = persistent.get("prefix_sha256")
        if not isinstance(prefix, str) or len(prefix) != 64 or any(c not in "0123456789abcdef" for c in prefix):
            raise PersistentProtocolError("Persistent response lacks a valid exact-token prefix digest")
        for key in ("resident_tokens", "appended_prompt_tokens"):
            if type(persistent.get(key)) is not int or persistent[key] < 0:
                raise PersistentProtocolError(f"Persistent response has invalid {key}")
        if persistent["resident_tokens"] > self.context_policy.context_limit:
            raise PersistentProtocolError("Persistent resident count exceeds configured capacity")
        expected_history = json.loads(self._persistent_history_eviction_json)
        receipt = persistent.get("history_eviction")
        if (not isinstance(receipt, dict) or receipt.get("policy") != expected_history["policy"]
                or type(receipt.get("fraction")) not in (int, float)
                or receipt["fraction"] != expected_history["fraction"]):
            raise PersistentProtocolError("History-eviction receipt differs from the pinned session policy")
        expected_budget = expected_history.get("resident_token_budget", self.context_policy.context_limit)
        if (type(receipt.get("resident_token_budget")) is not int
                or receipt["resident_token_budget"] != expected_budget):
            raise PersistentProtocolError("History-eviction resident-budget receipt differs from session settings")
        choice = choices[0]
        ids = choice.get("token_ids", persistent.get("token_ids"))
        if (not isinstance(ids, list) or any(type(token) is not int or token < 0 for token in ids)
                or ("token_ids" in persistent and persistent["token_ids"] != ids)):
            raise PersistentProtocolError("Missing or inconsistent exact generated token IDs")
        usage = payload.get("usage") or {}
        if (type(usage.get("prompt_tokens")) is not int or usage["prompt_tokens"] < 0
                or usage.get("completion_tokens") != len(ids)):
            raise PersistentProtocolError("Persistent usage does not match its exact token ledger")
        if not isinstance(choice.get("message"), dict):
            raise PersistentProtocolError("Persistent response lacks a parsed dispatch message")
        result = duck_agent._ChatCompletionResult(message=copy.deepcopy(choice["message"]),
            finish_reason=str(choice.get("finish_reason", "") or ""), usage=usage)
        started = time.monotonic()
        parse = choice.get("persistent_parse")
        previous = self._persistent_assistant_token_ids if continuation else []
        all_ids = previous + ids
        if parse is not None:
            if (not isinstance(parse, dict) or parse.get("scope") != "full_assistant"
                    or parse.get("prefix_tokens") != len(previous)
                    or parse.get("assistant_token_ids") != all_ids):
                raise PersistentProtocolError("Cumulative assistant parse does not match exact generated IDs")
            raw = self.output_decoder.decode(all_ids, skip_special_tokens=False)
            if parse.get("raw_text") != raw:
                raise PersistentProtocolError("Cumulative assistant raw text does not match token IDs")
            result.raw_text = raw
        elif continuation:
            raise PersistentProtocolError("Continuation lacks its cumulative assistant parse receipt")
        else:
            result.raw_text = self.output_decoder.decode(ids, skip_special_tokens=False)
        result.output_verification_error = None
        result.output_decode_seconds = time.monotonic() - started
        result.persistent = copy.deepcopy(persistent)
        result.persistent_assistant_token_ids = all_ids
        return result

    def _allocated_completion(self, messages, tools, admission):
        """Keep existing allocation/dispatch accounting, replace stateless HTTP."""
        started = time.monotonic()
        original_max = self._max_output_tokens
        reservation = admission["reserved_tokens"]
        sent = False
        known_usage = None
        self._transport_timings = {}
        record = dict(request=admission["request"], charged_tokens=0, completion_tokens=None,
                      prompt_tokens=None, cached_tokens=None, uncertain=False,
                      requested_max_tokens=None, finish_reason=None,
                      transport="arc3-persistent", session_id=self._persistent_session_id)
        try:
            if self._persistent_closed:
                raise PersistentProtocolError("Persistent game session was already closed")
            # A finite per-game allowance still constrains output; otherwise the
            # server uses available context and no artificial response cap.
            self._max_output_tokens = (reservation if self.context_policy.response_cap is not None
                                       or reservation < self.context_policy.context_limit else None)
            request_id = f"{self._persistent_session_id}-r{admission['request']}"
            staged, newly_external = self._stage_persistent_messages(messages, tools, request_id=request_id)
            display_payload = self._request_payload(messages, tools)
            if self._persistent_revision == 0:
                continuation = False
                if any(m.get("role") == "assistant" for m in messages):
                    raise AppendOnlyViolation("Initial persistent control cannot import parsed assistant history")
                route = "/arc3/persistent/start"
                body = dict(session_id=self._persistent_session_id, request_id=request_id,
                            chat=display_payload,
                            history_eviction=json.loads(self._persistent_history_eviction_json))
            else:
                if any(m.get("role") == "system" for m in newly_external):
                    raise AppendOnlyViolation("Cannot append a new system message")
                continuation = not newly_external
                if continuation and self._persistent_last_finish_reason != "length":
                    raise AppendOnlyViolation("Empty-input continuation requires a verified length pause")
                if continuation and not open_tail(messages):
                    raise AppendOnlyViolation("Length continuation requires the harness's open reasoning tail")
                route = HTTPPersistentTransport.session_route(self._persistent_session_id, "append")
                body = dict(request_id=request_id, expected_revision=self._persistent_revision,
                            expected_prefix_sha256=self._persistent_prefix_sha256,
                            external_messages=newly_external,
                            sampling={key: display_payload[key] for key in SAMPLING_FIELDS if key in display_payload},
                            continuation=continuation)
            # This field is transport control, never model-visible content or
            # a chat-template keyword. Historical audit messages remain intact.
            body["managed_suffixes"] = staged.sidecar()
            # The standard transcript remains a readable dispatch/history view.
            # Exact wire differences are preserved separately, never mislabeled
            # as a full-history request actually submitted to the session server.
            self._record_prepared_request(messages, tools, payload=display_payload)
            self._append_raw_evidence("persistent-request", dict(route=route, body=body))
            record["request_preparation_seconds"] = time.monotonic() - started
            record["requested_max_tokens"] = self._max_output_tokens
            record["appended_messages"] = len(newly_external)
            timeout = min(self._timeout or 900, self.allocation.deadline - time.time())
            if timeout <= 0:
                raise AllocationStopped("global_deadline")
            self.allocation.mark_submitted(self.game_id)
            self._persistent_attempted = sent = True
            before_transport = time.monotonic()
            try:
                if self._persistent_revision == 0:
                    payload = self._persistent_transport.start(body, timeout=timeout)
                else:
                    payload = self._persistent_transport.append(self._persistent_session_id, body, timeout=timeout)
            finally:
                self._transport_timings["http_roundtrip_seconds"] = time.monotonic() - before_transport
            self._append_raw_evidence("persistent-response", payload)
            if isinstance(payload, dict):
                usage = payload.get("usage") or {}
                if type(usage.get("completion_tokens")) is int and usage["completion_tokens"] >= 0:
                    known_usage = usage
            result = self._persistent_result(payload, continuation=continuation)
            if result.usage["completion_tokens"] > reservation:
                raise PersistentProtocolError("Persistent output exceeded admitted token allowance")
            self._persistent_external_state.commit(staged,
                accepted_input_tokens=result.usage["prompt_tokens"],
                managed_suffixes=result.persistent.get("managed_suffixes"))
            self._persistent_revision = result.persistent["revision"]
            self._persistent_prefix_sha256 = result.persistent["prefix_sha256"]
            self._persistent_last_finish_reason = result.finish_reason
            self._persistent_assistant_token_ids = result.persistent_assistant_token_ids
            if continuation:
                # The server parsed exact cumulative assistant IDs, including any
                # split Unicode byte tokens. The inherited loop is about to
                # replace this open message by prefix + result.raw_text. Empty
                # only that local accumulator so its new view is the complete
                # cumulative parse, never duplicated or decoded by fragments.
                messages[-1] = dict(messages[-1], reasoning="")
                self._append_raw_evidence("persistent-cumulative-dispatch", dict(
                    session_id=self._persistent_session_id, revision=self._persistent_revision,
                    assistant_token_count=len(result.persistent_assistant_token_ids)))
            usage = result.usage
            record.update(completion_tokens=usage["completion_tokens"], charged_tokens=usage["completion_tokens"],
                          prompt_tokens=usage["prompt_tokens"], finish_reason=result.finish_reason,
                          usage=usage, persistent=result.persistent,
                          output_decode_seconds=result.output_decode_seconds)
            return result
        except BaseException as exc:
            if known_usage is not None:
                self._accumulate_usage_tokens(known_usage)
            capacity_refusal = known_usage is None and self._uncommitted_capacity_refusal(exc)
            if isinstance(exc, PersistentProtocolError) and exc.error_payload is not None:
                self._append_raw_evidence("persistent-error",dict(
                    session_id=self._persistent_session_id,request_id=f"{self._persistent_session_id}-r{admission['request']}",
                    status_code=exc.status_code,body=exc.error_payload,
                    uncommitted_capacity_refusal=capacity_refusal))
            validation = isinstance(exc, requests.RequestException) and any(
                marker in str(exc) for marker in ("400 Client Error", "409 Client Error", "422 Client Error"))
            uncertain = sent and known_usage is None and not validation and not capacity_refusal
            completed = known_usage["completion_tokens"] if known_usage is not None else None
            reason = ("global_deadline" if time.time() >= self.allocation.deadline
                      else "persistent_capacity" if capacity_refusal else "persistent_request_error")
            self._persistent_stop(reason)
            record.update(error=f"{type(exc).__name__}: {exc}", uncertain=uncertain,
                          completion_tokens=completed, failure_kind=reason,
                          charged_tokens=reservation if uncertain else completed or 0)
            # Conservative experiment: do not guess at acceptance or retry model
            # output. Closing the owned session prevents orphaned cache state.
            self.close_persistent_state()
            raise
        finally:
            self._max_output_tokens = original_max
            record["serving_seconds"] = time.monotonic() - started
            record["transport_timings"] = dict(self._transport_timings)
            self.allocation.record(self.game_id, record)

    def close_persistent_state(self):
        if not hasattr(self, "_persistent_external_state") or self._persistent_closed:
            return
        if self._persistent_attempted:
            try:
                response = self._persistent_transport.close(self._persistent_session_id, timeout=15)
                if (not isinstance(response, dict)
                        or response.get("session_id") != self._persistent_session_id
                        or not (response.get("state") == "closed" or response.get("closed") is True)):
                    raise PersistentProtocolError("Owned session close was not acknowledged")
            except Exception as exc:
                self._append_raw_evidence("persistent-close-error", dict(error=str(exc),
                    session_id=self._persistent_session_id))
                return
            self._append_raw_evidence("persistent-close", response)
        self._persistent_closed = True

    def _close_chat_transport(self):
        try:
            self.close_persistent_state()
        finally:
            transport = getattr(self, "_persistent_transport", None)
            if transport is not None:
                transport.shutdown()
            super()._close_chat_transport()


class PersistentProlongSolver(ProlongSolver):
    def _make_analyzer(self, *args, **kwargs):
        agent = super()._make_analyzer(*args, **kwargs)
        agent._persistent_base_url = self.persistent_base_url
        agent._persistent_transport_factory = getattr(self, "persistent_transport_factory", None)
        agent._persistent_history_eviction_requested = json.loads(self.persistent_history_eviction_json)
        return agent

    _play_one = with_globals(AllocatedSolver._play_one,
                            AllocatedAgent=PersistentProlongAgent, AllocatedSession=ProlongSession)


def configure_persistent_solver(solver, *, base_url, transport_factory=None, history_eviction=None):
    """Call after production configure_solver on a new, not-yet-run solver."""
    if type(solver) is not ProlongSolver:
        raise TypeError("Configure the regular PRO-LONG solver first")
    config = history_eviction_config(history_eviction)
    solver.__class__ = PersistentProlongSolver
    solver.persistent_base_url = base_url
    solver.persistent_transport_factory = transport_factory
    solver.persistent_history_eviction_json = json.dumps(config, sort_keys=True, separators=(",", ":"))
    return solver
