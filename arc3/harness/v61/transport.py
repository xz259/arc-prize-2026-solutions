"""Harness v6 persistent-session transport (spec §3, §5.7, §5.8; BUILD_PLAN component 1).

A standalone client for the server's append-only sessions (/arc3/persistent/start | append | close), adapted from the
control harness's persistent adapter (prolong-v529 client, persistent_harness_adapter.py f48e6ee3 and
persistent_observations.py), keeping its protocol checks:
  - the external-message ledger: previously submitted system/user/tool messages never change; each operation sends only
    its new external messages; every opener is registered as an observation (one per operation) through the
    managed_suffixes sidecar;
  - revision / prefix-digest chaining, exact generated token ids, usage = len(token ids), the pinned eviction policy;
  - per-completion caps (soft / hard / check) with the control's cue texts, repetition detector and closes.
v6 changes (Harness Design 04:20, accepted):
  - finish_reason 'length' is a PAUSE: the assistant stays pending and the next request is an empty-input continuation
    (with a cap injection when one is due); nothing external may be appended while it is pending;
  - capacity refusal (HTTP 413 persistent_session_error with eviction_committed false, any reason) is surfaced as
    CapacityRefusal with the admissible append size I_max = B - 1 - R + E (None when the 413 has no receipt), for the
    agent's one HC1b retry; the session is unchanged by a refusal;
  - no memory-reminder pin: the static system prompt is sent as given.
Receipts kept for the agent: merged eviction ranges of the current position frame (reset at a rebase), the
observation blocks (history_state), and rebased-away observation ids.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import time
import uuid
from dataclasses import dataclass, field
from urllib.parse import quote

import requests

SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty",
                   "presence_penalty", "frequency_penalty", "max_tokens", "seed")
EXTERNAL_ROLES = frozenset({"system", "user", "tool"})

TURN_BUDGET_CUE = ("\n\nThis turn's reasoning budget is spent; the game itself continues and the next turn has a fresh "
                   "budget. I will conclude now with the best plan so far and act.\n</think>\n\n")
TURN_REPEAT_CUE = ("\n\nI was repeating the same tokens; that tally carries no information and has been removed. "
                   "I will compute what I need in code instead of enumerating here, and act now on the best plan so far."
                   "\n</think>\n\n")
TURN_REPEAT_CUE_CONTENT = ("\n\nThat repetition carried no information and has been removed. I will make the tool call now.\n\n")
TURN_BUDGET_CLOSE = "This turn was closed at its hard reasoning budget; the game continues in the next turn."
TURN_REPEAT_CLOSE = "This turn was closed after repeated token loops; the game continues in the next turn."
TURN_BUDGET_RECOVERY = ("The previous assistant turn was closed by the harness at its reasoning budget. That is not an error "
                        "and the game continues. Start a fresh turn now: act on the best plan so far with one python tool "
                        "call, keeping the reasoning short.")
TURN_REPEAT_RECOVERY = ("The previous assistant turn was closed by the harness because its reasoning fell into a repeated "
                        "token loop. The game continues. Start a fresh turn now: compute any tallies in code, keep the "
                        "reasoning short, and act with one python tool call.")


class AppendOnlyViolation(ValueError):
    """A request would change previously submitted history or break the operation order."""


class PersistentProtocolError(requests.RequestException):
    def __init__(self, message, *, status_code=None, error_payload=None):
        super().__init__(message)
        self.status_code = status_code
        self.error_payload = copy.deepcopy(error_payload)


class CapacityRefusal(Exception):
    """HTTP 413: the append does not fit after the protected prefix and tail. The session is unchanged."""

    def __init__(self, message, *, receipt=None, i_max=None):
        super().__init__(message)
        self.receipt = receipt
        self.i_max = i_max


# ----------------------------------------------------------------------------------------------- per-completion caps
def turn_caps_from_env():
    """(soft, hard, check) from ARC3_ASSISTANT_TURN_CAPS='soft,hard' (token counts) and ARC3_ASSISTANT_TURN_CHECK."""
    raw = os.environ.get("ARC3_ASSISTANT_TURN_CAPS", "8000,12000").strip()
    soft, hard = (int(x) for x in raw.split(","))
    if hard <= soft:
        raise ValueError("ARC3_ASSISTANT_TURN_CAPS hard cap must exceed the soft cap")
    check = int(os.environ.get("ARC3_ASSISTANT_TURN_CHECK", "8000"))
    if check < 256:
        raise ValueError("ARC3_ASSISTANT_TURN_CHECK must be >= 256")
    return soft, hard, check


def repetition_onset(ids, *, min_tokens=40, max_period=8):
    """Index where a repeating suffix (period <= max_period tokens, >= min_tokens long) begins, else None."""
    n = len(ids)
    for period in range(1, max_period + 1):
        if n < min_tokens or n < 2 * period:
            continue
        k = 0
        while n - 1 - k - period >= 0 and ids[n - 1 - k] == ids[n - 1 - k - period]:
            k += 1
        if k + period >= min_tokens:
            return n - (k + period)
    return None


def turn_cap_decision(cumulative, think_closed, soft, hard, cued, *, check=0, repeat_onset=None, strikes=0):
    """(max_tokens or None, assistant_injection or None, tail_eviction or None) for the next request of a completion.
    Copied from the control adapter (identical semantics)."""
    def cap(limit):
        if limit is None:
            return check or None
        return min(limit, check) if check else limit
    if repeat_onset is not None:
        tail = {"assistant_offset": repeat_onset, "reason": "repeated_tokens"}
        if strikes >= 1 or (hard and cumulative >= hard):
            text = ("" if think_closed else "\n</think>\n\n") + TURN_REPEAT_CLOSE
            return 1, {"text": text, "kind": "repeat_close", "close": True}, tail
        cue = TURN_REPEAT_CUE_CONTENT if think_closed else TURN_REPEAT_CUE
        limit = (hard - cumulative) if hard else None
        return cap(limit), {"text": cue, "kind": "repeat_cue", "close": False}, tail
    if hard and cumulative >= hard:
        text = ("" if think_closed else "\n</think>\n\n") + TURN_BUDGET_CLOSE
        return 1, {"text": text, "kind": "hard_close", "close": True}, None
    if soft and not cued and cumulative >= soft and not think_closed:
        return cap((hard - cumulative) if hard else None), {"text": TURN_BUDGET_CUE, "kind": "soft_budget", "close": False}, None
    if soft and not cued and cumulative < soft:
        return cap(soft - cumulative), None, None
    if hard:
        return cap(hard - cumulative), None, None
    return cap(None), None, None


# -------------------------------------------------------------------------------------------- external-message ledger
def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def wire_messages(messages):
    """Local identities (keys starting with _arc3_) never become chat API fields."""
    return [{k: v for k, v in m.items() if not k.startswith("_arc3_")} for m in messages]


def _parts(content):
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list) or any(not isinstance(p, dict) for p in content):
        raise AppendOnlyViolation("Observation content must be text or content-part dictionaries")
    return content


def is_observation(message):
    return message.get("role") == "user" and message.get("_arc3_observation") is True


def describe_observation(message, *, message_index, observation_id):
    content = message.get("content")
    parts = _parts(content)
    return dict(message_index=message_index, observation_id=observation_id,
                content_sha256=hashlib.sha256(canonical_json(content).encode()).hexdigest(),
                base_parts_end=len(parts), program_parts=None, snapshot_parts=None)


@dataclass
class Staged:
    request_id: str
    messages_json: str
    delta_json: str
    sidecar_json: str
    observation_ids: list = field(default_factory=list)

    def external_messages(self):
        return json.loads(self.delta_json)

    def sidecar(self):
        return json.loads(self.sidecar_json)


class ExternalMessageLedger:
    """Append-only ledger of the session's external messages (control semantics, one pending operation)."""

    def __init__(self):
        self._messages_json = "[]"
        self._observation_count = 0
        self._pending = None

    def committed(self):
        return json.loads(self._messages_json)

    def stage(self, messages, *, request_id):
        if any(m.get("role") not in EXTERNAL_ROLES for m in messages):
            raise AppendOnlyViolation("The ledger accepts only external messages")
        if self._pending is not None and self._pending.request_id != request_id:
            raise AppendOnlyViolation("Resolve the pending operation before preparing a new request")
        identity = canonical_json(messages)
        if self._pending is not None:
            if identity != self._pending.messages_json:
                raise AppendOnlyViolation("Retry changed the operation's messages")
            return self._pending
        current, previous = json.loads(identity), self.committed()
        if len(current) < len(previous):
            raise AppendOnlyViolation("Previously submitted messages were removed")
        for i, old in enumerate(previous):
            if current[i] != old:
                raise AppendOnlyViolation(f"Previously submitted message {i} changed")
        additions = current[len(previous):]
        obs = [describe_observation(m, message_index=i, observation_id=f"obs-{self._observation_count + 1:08d}")
               for i, m in enumerate(additions) if is_observation(m)]
        if len(obs) > 1:
            raise AppendOnlyViolation("Only one new observation may be submitted per operation")
        sidecar = dict(version=1, observations=obs, evict_observation_ids=[])
        self._pending = Staged(request_id, identity, canonical_json(wire_messages(additions)), canonical_json(sidecar),
                               [o["observation_id"] for o in obs])
        return self._pending

    def commit(self, staged, managed_suffixes):
        expected = dict(registered_observation_ids=staged.observation_ids, evicted_observation_ids=[])
        if (not isinstance(managed_suffixes, dict)
                or any(managed_suffixes.get(k) != v for k, v in expected.items())):
            raise AppendOnlyViolation("Service observation acknowledgement differs from the prepared operation")
        if staged is not self._pending:
            raise AppendOnlyViolation("Prepared operation is stale")
        self._messages_json = staged.messages_json
        self._observation_count += len(staged.observation_ids)
        self._pending = None

    def discard_rejected(self, staged):
        """A definitive pre-admission rejection (413 before admission) discards the pending plan."""
        if staged is not self._pending:
            raise AppendOnlyViolation("Cannot discard a nonpending operation")
        self._pending = None


class HTTPPersistentTransport:
    def __init__(self, base_url, *, headers=None):
        self.base_url = base_url.rstrip("/").removesuffix("/v1")
        self.headers = dict(headers or {})
        # 2026-10-03 step 2: two games crashed on a reused keep-alive connection that the server had closed while idle
        # (RemoteDisconnected / ConnectionResetError, no retry for POST). One connection per request avoids the race.
        self.headers.setdefault("Connection", "close")
        self.http = requests.Session()

    @staticmethod
    def session_route(session_id, operation):
        return "/arc3/persistent/" + quote(session_id, safe="") + "/" + operation

    def _post(self, route, payload, timeout):
        response = self.http.post(self.base_url + route, json=payload, headers=self.headers, timeout=timeout)
        try:
            try:
                response.raise_for_status()
            except requests.HTTPError as exc:
                try:
                    error_payload = response.json()
                except ValueError:
                    error_payload = None
                raise PersistentProtocolError(f"{exc} | response: {response.text.strip()[:2000]}",
                                              status_code=response.status_code, error_payload=error_payload) from exc
            return response.json()
        finally:
            response.close()

    def start(self, payload, *, timeout):
        return self._post("/arc3/persistent/start", payload, timeout)

    def append(self, session_id, payload, *, timeout):
        return self._post(self.session_route(session_id, "append"), payload, timeout)

    def close(self, session_id, *, timeout=15):
        return self._post(self.session_route(session_id, "close"), {}, timeout)

    def shutdown(self):
        self.http.close()


def _merge_ranges(ranges):
    out = []
    for a, b in sorted((int(a), int(b)) for a, b in ranges if int(b) > int(a)):
        if out and a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


@dataclass
class Completion:
    message: dict
    finish_reason: str
    usage: dict
    token_ids: list            # this request's generated ids
    assistant_token_ids: list  # cumulative ids of the (possibly continued) assistant
    pending: bool              # paused (length) with the assistant still open
    closed_by: str | None      # 'hard_close' | 'repeat_close' when the harness closed it
    persistent: dict
    seconds: float


class PersistentSession:
    """One game's server session. `generate(external, ...)` appends the new external messages (or continues a paused
    assistant when `external` is empty) and returns a Completion. Not thread-safe; one per game."""

    def __init__(self, *, base_url, model, tools, template_kwargs, sampling, history_eviction, im_end_ids,
                 decode, log=None, timeout=900.0):
        self.transport = HTTPPersistentTransport(base_url)
        self.model = model
        self.tools = tools
        self.template_kwargs = dict(template_kwargs)
        self.sampling = dict(sampling)
        self.history_eviction = dict(history_eviction)
        self.im_end_ids = tuple(im_end_ids)
        self.decode = decode            # token ids -> raw text (the served tokenizer)
        self.log = log or (lambda kind, payload: None)
        self.timeout = timeout
        self.session_id = "arc3-" + uuid.uuid4().hex
        self.revision = 0
        self.prefix_sha256 = None
        self.ledger = ExternalMessageLedger()
        self.messages = []              # every external message submitted (plus local keys)
        self.pending = False
        self.assistant_ids = []
        self.assistant_raw = ""
        self.last_finish = None
        self.soft, self.hard, self.check = turn_caps_from_env()
        self.cued = False
        self.strikes = 0
        self.closed = False
        self.started = False
        self.request_count = 0
        self.evicted_ranges = []        # merged logical ranges removed in the current position frame
        self.blocks = []                # history_state blocks of the last receipt
        self.rebased_away = set()       # observation ids dropped or cut by a position rebase
        self.rebases = 0
        self.observation_spans = {}     # observation id -> [start, end) logical span when known
        self.last_receipt = None

    # ----------------------------------------------------------------------------------------------------- requests
    def generate(self, external):
        """Append `external` (list of new user/tool messages; [] = continue the paused assistant) and generate."""
        if self.closed:
            raise PersistentProtocolError("Persistent game session was already closed")
        continuation = not external
        if continuation and not (self.pending and self.last_finish == "length"):
            raise AppendOnlyViolation("Empty-input continuation requires a verified length pause")
        if external and self.pending:
            raise AppendOnlyViolation("Cannot insert external messages before the paused assistant's im_end")
        cumulative = len(self.assistant_ids) if self.pending else 0
        think_closed = "</think>" in self.assistant_raw if self.pending else False
        limit, injection, tail = turn_cap_decision(
            cumulative, think_closed, self.soft, self.hard, self.cued, check=self.check,
            repeat_onset=repetition_onset(self.assistant_ids) if self.pending else None, strikes=self.strikes)
        if not self.pending:
            injection = tail = None
        sampling = dict(self.sampling)
        if limit is not None:
            sampling["max_tokens"] = limit
        self.request_count += 1
        request_id = f"{self.session_id}-r{self.request_count}"
        candidate = self.messages + list(external)
        externals = [m for m in candidate if m.get("role") in EXTERNAL_ROLES]
        staged = self.ledger.stage(externals, request_id=request_id)
        if not self.started:
            chat = dict(model=self.model, messages=wire_messages(candidate), tools=self.tools, stream=False,
                        **{k: v for k, v in sampling.items() if k in SAMPLING_FIELDS},
                        reasoning_effort=self.template_kwargs.get("reasoning_effort"),
                        chat_template_kwargs=self.template_kwargs, add_generation_prompt=True,
                        continue_final_message=False, add_special_tokens=False, tool_choice="auto",
                        return_token_ids=True, arc3_one_lookahead=True)
            route = "/arc3/persistent/start"
            body = dict(session_id=self.session_id, request_id=request_id, chat=chat,
                        history_eviction=self.history_eviction)
        else:
            route = HTTPPersistentTransport.session_route(self.session_id, "append")
            body = dict(request_id=request_id, expected_revision=self.revision,
                        expected_prefix_sha256=self.prefix_sha256, external_messages=staged.external_messages(),
                        sampling={k: v for k, v in sampling.items() if k in SAMPLING_FIELDS},
                        continuation=continuation)
            if injection is not None and continuation:
                body["assistant_injection"] = injection
                self.log("persistent-injection", dict(session_id=self.session_id, kind=injection["kind"],
                                                      close=injection["close"], cumulative_assistant_tokens=cumulative,
                                                      soft_cap=self.soft, hard_cap=self.hard, strikes=self.strikes))
                if injection["kind"] in ("soft_budget", "repeat_cue"):
                    self.cued = True
                if injection["kind"] == "repeat_cue":
                    self.strikes += 1
        body["managed_suffixes"] = staged.sidecar()
        self.log("persistent-request", dict(route=route, body=body))
        t0 = time.monotonic()
        try:
            payload = (self.transport.start(body, timeout=self.timeout) if not self.started
                       else self.transport.append(self.session_id, body, timeout=self.timeout))
        except PersistentProtocolError as exc:
            refusal = self._capacity_refusal(exc)
            self.log("persistent-error", dict(session_id=self.session_id, request_id=request_id,
                                              status_code=exc.status_code, body=exc.error_payload,
                                              capacity_refusal=refusal is not None))
            if refusal is not None:
                self.ledger.discard_rejected(staged)   # 413 before admission: the session is unchanged
                if injection is not None and injection["kind"] in ("soft_budget", "repeat_cue"):
                    self.cued = False                  # the cue never reached the model
                raise refusal from exc
            raise
        seconds = time.monotonic() - t0
        self.log("persistent-response", payload)
        result = self._parse(payload, continuation=continuation, seconds=seconds)
        self.ledger.commit(staged, result.persistent.get("managed_suffixes"))
        self.messages = candidate
        self.started = True
        self.revision = result.persistent["revision"]
        self.prefix_sha256 = result.persistent["prefix_sha256"]
        self.last_finish = result.finish_reason
        self.pending = result.pending
        self.assistant_ids = result.assistant_token_ids
        self.assistant_raw = self.decode(result.assistant_token_ids) if result.pending else ""
        self._note_receipt(result.persistent, staged, external)
        if not result.pending:
            self.cued = False
            self.strikes = 0
        return result

    def _capacity_refusal(self, exc):
        if exc.status_code != 413:
            return None
        error = exc.error_payload.get("error") if isinstance(exc.error_payload, dict) else None
        if not isinstance(error, dict) or error.get("type") != "persistent_session_error":
            return None
        receipt = error.get("history_eviction")
        if isinstance(receipt, dict):
            if receipt.get("eviction_committed") is not False:
                return None
            try:
                b = int(receipt["resident_token_budget"])
                r = int(receipt.get("resident_tokens_after_suffix", receipt.get("resident_tokens_before")))
                e = int(receipt.get("eligible_history_tokens") or 0)
                i_max = b - 1 - r + e
            except (KeyError, TypeError, ValueError):
                i_max = None
            return CapacityRefusal(error.get("message", "capacity"), receipt=receipt, i_max=i_max)
        return CapacityRefusal(error.get("message", "capacity"), receipt=None, i_max=None)

    def _parse(self, payload, *, continuation, seconds):
        choices = payload.get("choices")
        persistent = payload.get("persistent")
        if not isinstance(choices, list) or len(choices) != 1 or not isinstance(persistent, dict):
            raise PersistentProtocolError("Persistent response lacks its single choice/state receipt")
        if persistent.get("session_id") != self.session_id:
            raise PersistentProtocolError("Persistent response session identity mismatch")
        if persistent.get("revision") != self.revision + 1:
            raise PersistentProtocolError("Persistent response revision is not the expected successor")
        prefix = persistent.get("prefix_sha256")
        if not isinstance(prefix, str) or len(prefix) != 64:
            raise PersistentProtocolError("Persistent response lacks a valid exact-token prefix digest")
        receipt = persistent.get("history_eviction")
        if (not isinstance(receipt, dict) or receipt.get("policy") != self.history_eviction["policy"]
                or receipt.get("fraction") != self.history_eviction["fraction"]):
            raise PersistentProtocolError("History-eviction receipt differs from the pinned session policy")
        choice = choices[0]
        ids = choice.get("token_ids", persistent.get("token_ids"))
        if not isinstance(ids, list) or any(type(t) is not int or t < 0 for t in ids):
            raise PersistentProtocolError("Missing or inconsistent exact generated token IDs")
        usage = payload.get("usage") or {}
        if type(usage.get("prompt_tokens")) is not int or usage.get("completion_tokens") != len(ids):
            raise PersistentProtocolError("Persistent usage does not match its exact token ledger")
        if not isinstance(choice.get("message"), dict):
            raise PersistentProtocolError("Persistent response lacks a parsed dispatch message")
        injected = persistent.get("injected_token_ids") or []
        previous = (self.assistant_ids + list(injected)) if continuation else []
        all_ids = previous + list(ids)
        parse = choice.get("persistent_parse")
        if parse is not None:
            if (not isinstance(parse, dict) or parse.get("scope") != "full_assistant"
                    or parse.get("prefix_tokens") != len(previous) or parse.get("assistant_token_ids") != all_ids):
                raise PersistentProtocolError("Cumulative assistant parse does not match exact generated IDs")
        elif continuation:
            raise PersistentProtocolError("Continuation lacks its cumulative assistant parse receipt")
        finish = str(choice.get("finish_reason", "") or "")
        closed = tuple(all_ids[-len(self.im_end_ids):]) == self.im_end_ids
        if finish == "length":
            if closed:
                raise PersistentProtocolError("Length receipt contains terminal im_end; do not resume or dispatch")
            pending = True
        else:
            if not closed:
                raise PersistentProtocolError("Completed assistant lacks actual generated im_end")
            pending = False
        inj = persistent.get("assistant_injection") or {}
        closed_by = inj.get("kind") if (not pending and inj.get("close")) else None
        return Completion(message=copy.deepcopy(choice["message"]), finish_reason=finish, usage=usage,
                          token_ids=list(ids), assistant_token_ids=all_ids, pending=pending, closed_by=closed_by,
                          persistent=persistent, seconds=seconds)

    # ----------------------------------------------------------------------------------------------------- receipts
    def _note_receipt(self, persistent, staged, external):
        self.last_receipt = persistent
        rebase = persistent.get("rebase")
        if isinstance(rebase, dict):
            self.rebased_away.update(i for i in (rebase.get("dropped_observation_ids") or []) if isinstance(i, str))
            if isinstance(rebase.get("partial_observation_id"), str):
                self.rebased_away.add(rebase["partial_observation_id"])
            self.evicted_ranges = []
            self.rebases += 1
        blocks = (persistent.get("history_state") or {}).get("blocks")
        if isinstance(blocks, list):
            self.blocks = blocks
        receipt = persistent.get("history_eviction") or {}
        if not isinstance(rebase, dict) and receipt.get("history_removed_tokens"):
            self.evicted_ranges = _merge_ranges(list(self.evicted_ranges) + list(receipt.get("history_ranges") or []))

    def span_evicted(self, start, end):
        """Tokens of the logical span [start, end) evicted in the current frame."""
        total = 0
        for a, b in self.evicted_ranges:
            total += max(0, min(b, end) - max(a, start))
        return total

    def close(self):
        if self.closed:
            return
        if self.started:
            try:
                response = self.transport.close(self.session_id, timeout=15)
                self.log("persistent-close", response)
            except Exception as exc:  # noqa: BLE001
                self.log("persistent-close-error", dict(error=str(exc), session_id=self.session_id))
        self.closed = True
        self.transport.shutdown()
