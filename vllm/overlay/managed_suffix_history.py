"""Out-of-band suffix ownership; no token mapping or engine mutations here."""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json

from append_only_history import (AppendOnlyMessageState, AppendOnlyViolation,
                                PreparedHistory, observation_base)
from gameplay_config import wire_messages
from prolong_program_suffix import BASE_KEY as PROGRAM_BASE
from prolong_snapshots import BASE_KEY as SNAPSHOT_BASE


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _parts(content):
    if isinstance(content, str):
        return [{"type": "text", "text": content}]
    if not isinstance(content, list) or any(not isinstance(part, dict) for part in content):
        raise AppendOnlyViolation("Observation content must be text or content-part dictionaries")
    return content


def _is_observation(message):
    return message.get("role") == "user" and message.get("_arc3_observation") is True


def describe_observation(message, *, message_index, observation_id):
    """Locate exact semantic content parts using only validated private bases.

    The server resolves these indices against actual rendered token IDs. A
    client-side content boundary is never claimed to be a token boundary.
    """
    if not _is_observation(message):
        raise AppendOnlyViolation("Suffix descriptors require a real observation")
    if type(message_index) is not int or message_index < 0:
        raise AppendOnlyViolation("Invalid operation-local message index")
    if not isinstance(observation_id, str) or not observation_id:
        raise AppendOnlyViolation("Missing observation identity")
    base = observation_base(message)
    content = message["content"]
    parts, base_parts = _parts(content), _parts(base["content"])
    if parts[:len(base_parts)] != base_parts:
        raise AppendOnlyViolation("Observation content does not begin with its immutable base")
    base_end = len(base_parts)
    snapshot_start = len(_parts(message[SNAPSHOT_BASE])) if SNAPSHOT_BASE in message else len(parts)
    program = None
    if PROGRAM_BASE in message:
        if _parts(message[PROGRAM_BASE]) != base_parts or snapshot_start <= base_end:
            raise AppendOnlyViolation("Program suffix does not follow the immutable observation")
        program = [base_end, snapshot_start]
    elif snapshot_start != base_end:
        raise AppendOnlyViolation("Unowned content appears before the snapshot suffix")
    snapshot = None
    if SNAPSHOT_BASE in message:
        if (parts[:snapshot_start] != _parts(message[SNAPSHOT_BASE])
                or snapshot_start >= len(parts)):
            raise AppendOnlyViolation("Snapshot suffix does not follow its explicit base")
        snapshot = [snapshot_start, len(parts)]
    return dict(message_index=message_index, observation_id=observation_id,
                content_sha256=hashlib.sha256(canonical_json(content).encode()).hexdigest(),
                base_parts_end=base_end, program_parts=program, snapshot_parts=snapshot)


def _input_identity(messages):
    # Restored or freshly decorated views of the same observation are accepted;
    # only the first prepared content is submitted for this operation.
    return canonical_json([observation_base(message) if _is_observation(message)
                           else message for message in messages])


@dataclass(frozen=True)
class PreparedSuffixHistory:
    _owner: object = field(repr=False, compare=False)
    _revision: int = field(repr=False)
    _history: PreparedHistory = field(repr=False)
    request_id: str
    _input_json: str = field(repr=False)
    _delta_json: str = field(repr=False)
    _sidecar_json: str = field(repr=False)

    @property
    def renderer_fingerprint(self):
        return self._history.renderer_fingerprint

    def messages(self):
        return self._history.messages()

    def wire_messages(self):
        return self._history.wire_messages()

    def external_messages(self):
        return json.loads(self._delta_json)

    def sidecar(self):
        return json.loads(self._sidecar_json)


class ManagedSuffixMessageState:
    """Keep audit history append-only while retiring older suffix ownership.

    Only one pending request is allowed per game. This is an ownership planner,
    not a KV cache implementation. Rejections must be explicit before retrying
    a changed operation; uncertain transport results cannot discard a plan.
    """

    def __init__(self, *, program_enabled=True, snapshot_enabled=True):
        self._history = AppendOnlyMessageState(program_enabled=program_enabled,
                                                snapshot_enabled=snapshot_enabled)
        self.program_enabled = program_enabled
        self.snapshot_enabled = snapshot_enabled
        self._owner = object()
        self._revision = 0
        self._observation_count = 0
        self._live_ids = []
        self._pending = None
        self._last = None
        self._last_receipt = None
        self._retired_request_ids = set()

    def committed_messages(self):
        return self._history.committed_messages()

    def live_observation_ids(self):
        return list(self._live_ids)

    def stage(self, messages, *, request_id, workspace, renderer_fingerprint):
        if not isinstance(request_id, str) or not request_id or len(request_id) > 256:
            raise AppendOnlyViolation("Require a nonempty request ID of at most 256 characters")
        if not isinstance(messages, list) or any(not isinstance(m, dict) for m in messages):
            raise AppendOnlyViolation("messages must be a list of dictionaries")
        if any(m.get("role") not in {"system", "user", "tool"} for m in messages):
            raise AppendOnlyViolation("Ownership planning accepts only external messages")
        identity = _input_identity(messages)
        if self._pending is not None and self._pending.request_id != request_id:
            raise AppendOnlyViolation("Resolve the pending operation before preparing a new request")
        cached = self._pending if self._pending is not None else self._last
        if cached is not None and cached.request_id == request_id:
            if (identity != cached._input_json
                    and identity != _input_identity(cached.messages())):
                raise AppendOnlyViolation("Retry changed the operation's message bases or tool feedback")
            if renderer_fingerprint != cached.renderer_fingerprint:
                raise AppendOnlyViolation("Retry changed the rendering configuration")
            return cached
        if request_id in self._retired_request_ids:
            raise AppendOnlyViolation("The request ID belongs to an older resolved operation")
        prepared = self._history.stage(messages, workspace=workspace,
                                        renderer_fingerprint=renderer_fingerprint)
        old_count = len(self.committed_messages())
        additions = prepared.messages()[old_count:]
        observations = [describe_observation(message, message_index=index,
                          observation_id=f"obs-{self._observation_count + 1:08d}")
                        for index, message in enumerate(additions) if _is_observation(message)]
        # Phase-one staging rejects multiple new real observations; never build
        # unseen earlier suffixes from the latest workspace state.
        if len(observations) > 1:
            raise AppendOnlyViolation("Only one new observation may be submitted per operation")
        sidecar = dict(version=1, observations=observations,
                       evict_observation_ids=list(self._live_ids) if observations else [])
        candidate = PreparedSuffixHistory(self._owner, self._revision, prepared,
                    request_id, identity, canonical_json(wire_messages(additions)),
                    canonical_json(sidecar))
        self._pending = candidate
        return candidate

    def _validate_owner(self, prepared):
        if not isinstance(prepared, PreparedSuffixHistory) or prepared._owner is not self._owner:
            raise AppendOnlyViolation("Prepared suffix operation belongs to another session")

    def commit(self, prepared, *, accepted_input_tokens, managed_suffixes):
        self._validate_owner(prepared)
        if type(accepted_input_tokens) is not int or accepted_input_tokens < 0:
            raise AppendOnlyViolation("Require the authoritative accepted input token count")
        sidecar = prepared.sidecar()
        expected = dict(registered_observation_ids=[o["observation_id"] for o in sidecar["observations"]],
                        evicted_observation_ids=sidecar["evict_observation_ids"])
        if (not isinstance(managed_suffixes, dict)
                or any(managed_suffixes.get(key) != value for key, value in expected.items())):
            raise AppendOnlyViolation("Service suffix acknowledgement differs from the prepared ownership operation")
        receipt = canonical_json(dict(input_tokens=accepted_input_tokens, **expected))
        if prepared is self._last:
            if receipt != self._last_receipt:
                raise AppendOnlyViolation("Repeated acknowledgement changed the accepted result")
            return
        if prepared is not self._pending or prepared._revision != self._revision:
            raise AppendOnlyViolation("Prepared suffix operation is stale or was rejected")
        self._history.commit(prepared._history, accepted_input_tokens=accepted_input_tokens)
        if sidecar["observations"]:
            self._live_ids = [o["observation_id"] for o in sidecar["observations"]
                              if o["program_parts"] is not None or o["snapshot_parts"] is not None]
            self._observation_count += len(sidecar["observations"])
        if self._last is not None:
            self._retired_request_ids.add(self._last.request_id)
        self._last, self._last_receipt = prepared, receipt
        self._pending = None
        self._revision += 1

    def discard_rejected(self, prepared, *, server_rejected):
        """Only a definitive pre-admission rejection may discard a pending plan."""
        self._validate_owner(prepared)
        if server_rejected is not True or prepared is not self._pending:
            raise AppendOnlyViolation("Cannot discard an uncertain or nonpending operation")
        self._retired_request_ids.add(prepared.request_id)
        self._pending = None
