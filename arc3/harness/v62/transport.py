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
import threading
import time
import uuid
from dataclasses import dataclass, field
from urllib.parse import quote

import requests

SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty",
                   "presence_penalty", "frequency_penalty", "max_tokens", "seed")
EXTERNAL_ROLES = frozenset({"system", "user", "tool"})

# round 5i (user 10-07 ~05:05: two cues of different urgency): the closing cue names what is left for the reply before the
# hard close ({left} = hard - soft); the earlier nudge leaves the reasoning open and only encourages concluding
TURN_BUDGET_CUE = ("\n\nThis turn's reasoning budget is spent, and about {left:,} tokens remain for my reply before the turn is "
                   "closed; the game itself continues and the next turn has a fresh budget. I will conclude now with the "
                   "best plan so far and act.\n</think>\n\n")
# round 5i fix (391 live, +18 min): 3 of the first 13 nudged replies ended right after the nudge with no text and no call
# (0 such replies in 386 and 390); the nudge now names the next step, the python call
TURN_BUDGET_NUDGE = ("\n\nThis turn's reasoning has used {used:,} tokens. I should conclude with the best plan so far and act "
                     "on it with a python call.\n\n")
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
# v7.6 CUE_MARK (apply_v76.py; the user 10-24: "Your proposal (Harness:) looks good"): ARC3_V76_CUE_MARK=1 marks the in-message
# injections (the 12K cue, the repeat cues, the 16K and repeat closes) as harness notes, in parentheses and in the harness's voice,
# instead of sentences in the model's voice; unset or 0 keeps the texts above. The recovery messages are user messages (unchanged);
# the nudge (off: two caps) is unchanged. The corpus builders drop injections by the receipts' injected_token_ids, not by text.
_V76_CUE_MARK = os.environ.get('ARC3_V76_CUE_MARK', '0')
if _V76_CUE_MARK not in ('0', '1'):
    raise RuntimeError('ARC3_V76_CUE_MARK must be 0 or 1, not %r' % _V76_CUE_MARK)
V76_CUE_MARK = _V76_CUE_MARK == '1'
if V76_CUE_MARK:
    TURN_BUDGET_CUE = ("\n\n(Harness: this turn's reasoning budget is spent; about {left:,} tokens remain for the reply before the turn "
                       "is closed. The game continues and the next turn has a fresh budget. Conclude now with the best plan so far "
                       "and act.)\n</think>\n\n")
    TURN_REPEAT_CUE = ("\n\n(Harness: repeated tokens were removed; they carried no information. Compute what you need in code "
                       "instead of enumerating here, and act now on the best plan so far.)\n</think>\n\n")
    TURN_REPEAT_CUE_CONTENT = "\n\n(Harness: the repetition was removed; it carried no information. Make the tool call now.)\n\n"
    TURN_BUDGET_CLOSE = "(Harness: this turn was closed at its hard reasoning budget; the game continues in the next turn.)"
    TURN_REPEAT_CLOSE = "(Harness: this turn was closed after repeated token loops; the game continues in the next turn.)"

# v7.9 KAGGLE K1 (C1a; apply_v79.py): ARC3_KAGGLE_SERVER_WAIT_S > 0 (Kaggle: the client starts while the server loads): until this
# process has seen the server's /health answer 200, a request first waits for it (one poller at a time, every 5 s, at most that many
# seconds; then it goes out as before). The requests that waited go out ARC3_V62_START_STAGGER s apart (the first 25, as the pod's
# staggered game starts). After the first 200 nothing changes. 0 or unset (default): off.
_K79_WAIT_S = float(os.environ.get('ARC3_KAGGLE_SERVER_WAIT_S', '0') or 0)
_K79_STAGGER_S = float(os.environ.get('ARC3_V62_START_STAGGER', '3') or 0)
_K79_UP = threading.Event()
_K79_POLL = threading.Lock()
_K79_REL = threading.Lock()
_K79_STATE = dict(released=0, first_up=None)
# v7.9b KAGGLE K5 (apply_v79b.py): ARC3_KAGGLE_KEEPALIVE_S > 0 (gateway runs): while K1 waits, at most one keep-alive RESET per period
# across all waiting games, done by a waiting game on itself (its session handed over by the agent through _K79_TLS.session), straight
# to the game API so the model-facing history is untouched; the gateway closes a scorecard idle for 15 min. 0 or unset: off.
_K79_KEEPALIVE_S = float(os.environ.get('ARC3_KAGGLE_KEEPALIVE_S', '0') or 0)
_K79_TLS = threading.local()
_K79_KA_LOCK = threading.Lock()
_K79_KA = dict(next=None, n=0, games=[])


def _k79_keepalive():
    if _K79_KEEPALIVE_S <= 0:
        return
    sess = getattr(_K79_TLS, 'session', None)
    if sess is None:
        return
    now = time.monotonic()
    with _K79_KA_LOCK:
        if _K79_KA['next'] is None:
            _K79_KA['next'] = now + _K79_KEEPALIVE_S
            return
        if now < _K79_KA['next']:
            return
        _K79_KA['next'] = now + _K79_KEEPALIVE_S
        _K79_KA['n'] += 1
        n = _K79_KA['n']
    gid = None
    try:
        import arcengine
        game = sess.game
        gid = getattr(getattr(game, 'game_run', None), 'game_id', None)
        game.execute_action(arcengine.ActionInput(id=arcengine.GameAction.RESET, data={}), generated_tokens=0, uncached_input_tokens=0)
        _K79_KA['games'].append(gid)
        print('v79 keepalive: RESET %d on %s while the server loads' % (n, gid), flush=True)
    except Exception as exc:  # noqa: BLE001  (a failed keep-alive must not stop the wait)
        print('v79 keepalive: RESET %d on %s failed: %r' % (n, gid, exc), flush=True)


def _k79_wait_server(base_url):
    if _K79_WAIT_S <= 0 or _K79_UP.is_set():
        return
    deadline = time.monotonic() + _K79_WAIT_S
    while not _K79_UP.is_set() and time.monotonic() < deadline:
        with _K79_POLL:
            if _K79_UP.is_set():
                break
            try:
                ok = requests.get(base_url + "/health", timeout=5).status_code == 200
            except requests.RequestException:
                ok = False
            if ok:
                _K79_STATE['first_up'] = time.time()
                _K79_UP.set()
                break
        _k79_keepalive()                              # v7.9b K5: off unless ARC3_KAGGLE_KEEPALIVE_S > 0
        _K79_UP.wait(5.0)
    if _K79_UP.is_set() and _K79_STAGGER_S > 0:
        with _K79_REL:
            k = _K79_STATE['released']
            _K79_STATE['released'] = k + 1
        time.sleep(_K79_STAGGER_S * min(k, 24))


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
def _caps_values():
    values = [int(x) for x in os.environ.get("ARC3_ASSISTANT_TURN_CAPS", "8000,12000").strip().split(",")]
    if len(values) not in (2, 3):
        raise ValueError("ARC3_ASSISTANT_TURN_CAPS must be 'soft,hard' or 'nudge,soft,hard'")
    return values


def turn_caps_from_env():
    """(soft, hard, check) from ARC3_ASSISTANT_TURN_CAPS='soft,hard' or 'nudge,soft,hard' (token counts) and
    ARC3_ASSISTANT_TURN_CHECK."""
    soft, hard = _caps_values()[-2:]
    if hard <= soft:
        raise ValueError("ARC3_ASSISTANT_TURN_CAPS hard cap must exceed the soft cap")
    check = int(os.environ.get("ARC3_ASSISTANT_TURN_CHECK", "8000"))
    if check < 256:
        raise ValueError("ARC3_ASSISTANT_TURN_CHECK must be >= 256")
    return soft, hard, check


def turn_nudge_from_env():
    """The nudge point of ARC3_ASSISTANT_TURN_CAPS='nudge,soft,hard' (0 for 'soft,hard'): when the reasoning reaches it,
    a note in the model's voice that leaves the reasoning open; the closing cue still comes at the soft cap."""
    values = _caps_values()
    if len(values) == 2:
        return 0
    nudge, soft = values[0], values[1]
    if not 0 < nudge < soft:
        raise ValueError("ARC3_ASSISTANT_TURN_CAPS nudge must be above 0 and below the soft cap")
    return nudge


def turn_call_extension_from_env():
    """Tokens a tool call that is open at the hard cap may still use to finish (user 10-16: "12k/16k should allow ongoing tool
    call to finish"; adopted without A/B). ARC3_ASSISTANT_TURN_CALL_EXT, default 4096; 0 restores the hard close at the cap."""
    ext = int(os.environ.get("ARC3_ASSISTANT_TURN_CALL_EXT", "4096"))
    if ext < 0:
        raise ValueError("ARC3_ASSISTANT_TURN_CALL_EXT must be >= 0")
    return ext


def tool_call_open(raw):
    """True when the reply after the reasoning has opened a tool call (<tool_call>) that is not closed yet (</tool_call>)."""
    if not raw or "</think>" not in raw:
        return False
    text = raw.rsplit("</think>", 1)[-1]
    at = text.rfind("<tool_call>")
    return at >= 0 and text.find("</tool_call>", at) < 0


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


def turn_cap_decision(cumulative, think_closed, soft, hard, cued, *, check=0, repeat_onset=None, strikes=0, nudge=0,
                      nudged=False, call_open=False, call_extension=0):
    """(max_tokens or None, assistant_injection or None, tail_eviction or None) for the next request of a completion.
    Copied from the control adapter (identical semantics); `nudge` (0 = off) adds the open nudge below the soft cap.
    10-16: at the hard cap, a tool call opened after the reasoning (`call_open`) continues without an injection for up to
    `call_extension` more tokens, so a nearly finished call is not cut (the 12K cue already waits: it is sent only while the
    reasoning is open); past the extension, or with no open call, the hard close as before. Repetition handling is unchanged."""
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
        if call_open and think_closed and cumulative < hard + call_extension:
            return cap(hard + call_extension - cumulative), None, None
        text = ("" if think_closed else "\n</think>\n\n") + TURN_BUDGET_CLOSE
        return 1, {"text": text, "kind": "hard_close", "close": True}, None
    if soft and not cued and cumulative >= soft and not think_closed:
        return cap((hard - cumulative) if hard else None), {"text": TURN_BUDGET_CUE.format(left=hard - soft),
                                                              "kind": "soft_budget", "close": False}, None
    if nudge and soft and not nudged and not cued and not think_closed and nudge <= cumulative < soft:
        text = TURN_BUDGET_NUDGE.format(used=nudge)
        return cap(soft - cumulative), {"text": text, "kind": "soft_nudge", "close": False}, None
    if nudge and soft and not nudged and not cued and cumulative < nudge:
        return cap(nudge - cumulative), None, None
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
        _k79_wait_server(self.base_url)            # v7.9 KAGGLE K1: off unless ARC3_KAGGLE_SERVER_WAIT_S > 0
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
    cues: tuple = ()           # round 5i: the injection kinds sent into this assistant message so far


RESUME_RETRIES = 60             # v7.4 (IE 10-23): a resume append refused by the server's admission (429) is retried as is
RESUME_RETRY_S = 2.0


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
        self.nudge = turn_nudge_from_env()
        self.call_ext = turn_call_extension_from_env()
        self.call_extensions = 0        # hard-cap extensions granted to open tool calls (log: persistent-call-extension)
        self.cued = False
        self.nudged = False
        self.message_cues = []          # injection kinds sent into the current assistant message
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
        self.message_obs = []           # v7.4: each message's observation id (None: not an observation), as the ledger numbers them
        self.suspended = False          # v7.4: suspended on the server; the next generate resumes it (with a rebase)
        self.suspends = 0
        self.trims = 0                  # v7.4: message / ledger trims after rebases
        self.resume_retries = 0         # v7.4: 429 retries of resume appends (the agent's receipt reads and resets it)

    # ----------------------------------------------------------------------------------------------------- requests
    def generate(self, external, rebase=False):
        """Append `external` (list of new user/tool messages; [] = continue the paused assistant) and generate. v7.4: rebase=True
        asks the server to rebase on this append (ignored on a start and on a continuation)."""
        if self.closed:
            raise PersistentProtocolError("Persistent game session was already closed")
        continuation = not external
        if continuation and not (self.pending and self.last_finish == "length"):
            raise AppendOnlyViolation("Empty-input continuation requires a verified length pause")
        if external and self.pending:
            raise AppendOnlyViolation("Cannot insert external messages before the paused assistant's im_end")
        cumulative = len(self.assistant_ids) if self.pending else 0
        think_closed = "</think>" in self.assistant_raw if self.pending else False
        call_open = tool_call_open(self.assistant_raw) if self.pending else False
        limit, injection, tail = turn_cap_decision(
            cumulative, think_closed, self.soft, self.hard, self.cued, check=self.check,
            repeat_onset=repetition_onset(self.assistant_ids) if self.pending else None, strikes=self.strikes,
            nudge=self.nudge, nudged=self.nudged, call_open=call_open, call_extension=self.call_ext)
        extending = bool(self.pending and call_open and self.hard and cumulative >= self.hard and injection is None
                         and tail is None)
        if extending:
            self.call_extensions += 1
            self.log("persistent-call-extension", dict(session_id=self.session_id, cumulative_assistant_tokens=cumulative,
                                                       hard_cap=self.hard, extension=self.call_ext, max_tokens=limit))
        if not self.pending:
            injection = tail = None
            self.message_cues = []
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
            pins = [i for i, m in enumerate(candidate) if m.get("_arc3_pin_parts")]   # v6.5 C5 (arm A)
            if pins:
                body["arc3_summary_pin"] = dict(message_index=pins[0], parts_end=int(candidate[pins[0]]["_arc3_pin_parts"]))
                if candidate[pins[0]].get("_arc3_pin_split"):        # v7.7 (apply_v77.py): the text pin / board pin split
                    body["arc3_summary_pin"]["parts_split"] = int(candidate[pins[0]]["_arc3_pin_split"])
        else:
            route = HTTPPersistentTransport.session_route(self.session_id, "append")
            body = dict(request_id=request_id, expected_revision=self.revision,
                        expected_prefix_sha256=self.prefix_sha256, external_messages=staged.external_messages(),
                        sampling={k: v for k, v in sampling.items() if k in SAMPLING_FIELDS},
                        continuation=continuation)
            if rebase and not continuation:         # v7.4: the server rebases on this append (the rebase arm)
                body["arc3_rebase"] = True
            if injection is not None and continuation:
                body["assistant_injection"] = injection
                self.log("persistent-injection", dict(session_id=self.session_id, kind=injection["kind"],
                                                      close=injection["close"], cumulative_assistant_tokens=cumulative,
                                                      soft_cap=self.soft, hard_cap=self.hard, strikes=self.strikes,
                                                      nudge_cap=self.nudge))
                if injection["kind"] in ("soft_budget", "repeat_cue"):
                    self.cued = True
                if injection["kind"] == "soft_nudge":
                    self.nudged = True
                if injection["kind"] == "repeat_cue":
                    self.strikes += 1
                self.message_cues.append(injection["kind"])
        body["managed_suffixes"] = staged.sidecar()
        self.log("persistent-request", dict(route=route, body=body))
        t0 = time.monotonic()
        try:
            payload = (self.transport.start(body, timeout=self.timeout) if not self.started
                       else self._append_resume(body) if self.suspended          # v7.4: a resume retries a 429
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
                if injection is not None and injection["kind"] == "soft_nudge":
                    self.nudged = False
                if injection is not None and continuation and self.message_cues:
                    self.message_cues.pop()
                raise refusal from exc
            raise
        seconds = time.monotonic() - t0
        self.log("persistent-response", payload)
        result = self._parse(payload, continuation=continuation, seconds=seconds)
        result.cues = tuple(self.message_cues)
        if extending:                   # the receipt: did the open call finish inside the extension (stop) or pause again
            self.log("persistent-call-extension-end", dict(session_id=self.session_id, finish_reason=result.finish_reason,
                                                           pending=result.pending,
                                                           assistant_tokens=len(result.assistant_token_ids)))
        self.ledger.commit(staged, result.persistent.get("managed_suffixes"))
        self.messages = candidate
        obs_ids = list(staged.observation_ids)       # v7.4: at most one observation per operation (the ledger's rule)
        self.message_obs = self.message_obs + [obs_ids.pop(0) if (is_observation(m) and obs_ids) else None for m in external]
        self.suspended = False
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
            self.nudged = False
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
        if isinstance(rebase, dict):
            self._trim_after_rebase(persistent)

    def _trim_after_rebase(self, persistent):
        """v7.4 (the user's growing-history point): after a position rebase the messages and the append ledger keep only the
        prefix (the messages before the first observation) and every observation block (the observation message and the
        messages after it, to the next observation) whose observation the server's registry still lists, so a session that
        lives for a whole game stays bounded. The ledger is trimmed identically: the next operation's prefix check covers
        these messages, and the observation ids keep counting."""
        blocks = (persistent.get("history_state") or {}).get("blocks")
        if not isinstance(blocks, list) or len(self.message_obs) != len(self.messages) or self.ledger._pending is not None:
            self.log("persistent-trim-skipped", dict(session_id=self.session_id, blocks=isinstance(blocks, list),
                                                     aligned=len(self.message_obs) == len(self.messages)))
            return
        kept = {b.get("observation_id") for b in blocks if isinstance(b, dict)}
        keep, current = [], None
        for oid in self.message_obs:
            current = oid if oid is not None else current
            keep.append(current is None or current in kept)
        if all(keep):
            return
        n0 = len(self.messages)
        self.messages = [m for m, k in zip(self.messages, keep) if k]
        self.message_obs = [o for o, k in zip(self.message_obs, keep) if k]
        self.ledger._messages_json = canonical_json([m for m in self.messages if m.get("role") in EXTERNAL_ROLES])
        self.trims += 1
        self.log("persistent-trim", dict(session_id=self.session_id, messages_before=n0, messages_after=len(self.messages),
                                          observations_kept=sum(1 for o in self.message_obs if o is not None)))

    def span_evicted(self, start, end):
        """Tokens of the logical span [start, end) evicted in the current frame."""
        total = 0
        for a, b in self.evicted_ranges:
            total += max(0, min(b, end) - max(a, start))
        return total

    def _append_resume(self, body):
        """v7.4 (IE 10-23): the resume append of a suspended session. A 429 means the server's unsuspended sessions fill its
        admission limit for a moment (a finished game's close not yet seen when this game's lane came back); the server changed
        nothing, so the same request is sent again every RESUME_RETRY_S s, at most RESUME_RETRIES times."""
        for i in range(RESUME_RETRIES):
            try:
                return self.transport.append(self.session_id, body, timeout=self.timeout)
            except PersistentProtocolError as exc:
                if exc.status_code != 429 or i == RESUME_RETRIES - 1:
                    raise
                self.resume_retries += 1
                self.log("persistent-resume-retry", dict(session_id=self.session_id, attempt=i + 1))
                time.sleep(RESUME_RETRY_S)

    def suspend(self):
        """v7.4 (the rebase arm's park): the server suspends this session: its engine request and KV are released, its token
        history stays; the next generate() resumes it (the server rebases the frame on that append). Returns the server's receipt."""
        if self.closed or not self.started:
            raise PersistentProtocolError("Only a started, open session can be suspended")
        if self.pending:
            raise AppendOnlyViolation("Cannot suspend a paused assistant (length pause)")
        payload = self.transport.append(self.session_id, dict(arc3_suspend=True, expected_revision=self.revision), timeout=120)
        self.suspended = True
        self.suspends += 1
        self.log("persistent-suspend", payload)
        return payload

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
