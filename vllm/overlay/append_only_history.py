"""Local-only append-only observation policy; no serving or eviction hooks.

The serving adapter owns token IDs, hybrid state and request acknowledgements.
This module prevents accidental historical message rewriting before integration.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, replace
import json
from typing import Callable, Sequence

from gameplay_config import wire_messages
from prolong_program_suffix import BASE_KEY as PROGRAM_BASE, refresh_program_suffix
from prolong_snapshots import BASE_KEY as SNAPSHOT_BASE, remove_snapshot_suffix, refresh_snapshot_suffix


class AppendOnlyViolation(ValueError):
    """A request would change previously submitted history."""


class CapacityExceeded(AppendOnlyViolation):
    """The no-eviction control must stop or explicitly rebuild its session."""

    def __init__(self, input_tokens: int, token_limit: int, required_free_tokens: int):
        self.input_tokens = input_tokens
        self.token_limit = token_limit
        self.required_free_tokens = required_free_tokens
        super().__init__(
            f"Append-only capacity exceeded: {input_tokens} input tokens + "
            f"{required_free_tokens} required free tokens > {token_limit}. "
            "No history was trimmed; stop or explicitly rebuild the session."
        )


def _json(value):
    # JSON is the actual message domain and avoids sharing nested mutable arrays.
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(",", ":"), allow_nan=False)


def _is_observation(message):
    return message.get("role") == "user" and message.get("_arc3_observation") is True


def observation_base(message):
    """Undo only explicit private wrappers, in their construction order."""
    current = copy.deepcopy(message)
    # Never trust a private base that no longer matches the actual observation's
    # prefix. Only managed suffix content may be discarded/replaced here.
    for key in (SNAPSHOT_BASE, PROGRAM_BASE):
        if key not in current:
            continue
        base = current[key]
        prefix = [{"type": "text", "text": base}] if isinstance(base, str) else base
        content = current.get("content")
        if (not isinstance(prefix, list) or not isinstance(content, list)
                or len(content) <= len(prefix) or content[:len(prefix)] != prefix):
            raise AppendOnlyViolation("Managed suffix base no longer matches observation content")
        current["content"] = base
        del current[key]
    clean = remove_snapshot_suffix([current])
    return refresh_program_suffix(clean, "", enabled=False)[0]


def decorate_observation(message, prefix, workspace, *, program_enabled, snapshot_enabled):
    """Use v5.22 builders while discarding their historical-removal side effect.

    The full prefix is provided for the optional retained-boundary sampler. Only
    the new observation is taken from the builders' return value. Their current
    default sampling remains the unrestricted 6+6 policy.
    """
    base = observation_base(message)
    combined = remove_snapshot_suffix(copy.deepcopy([*prefix, base]))
    combined = refresh_program_suffix(combined, workspace.program_index.text(),
                                      enabled=program_enabled)
    combined = refresh_snapshot_suffix(combined, workspace, enabled=snapshot_enabled)
    return combined[-1]


@dataclass(frozen=True)
class PreparedHistory:
    """A candidate to commit only after its request is accepted by the server."""

    _owner: object = field(repr=False, compare=False)
    _revision: int = field(repr=False)
    _messages_json: str = field(repr=False)
    renderer_fingerprint: str
    input_tokens: int | None
    newly_added_messages: int

    def messages(self):
        """A fresh local/audit copy including private construction fields."""
        return json.loads(self._messages_json)

    def wire_messages(self):
        return wire_messages(self.messages())


class AppendOnlyMessageState:
    """Per-game no-eviction control, deliberately absent from production defaults."""

    def __init__(self, *, program_enabled=True, snapshot_enabled=True):
        if type(program_enabled) is not bool or type(snapshot_enabled) is not bool:
            raise ValueError("Suffix options must be booleans")
        self.program_enabled = program_enabled
        self.snapshot_enabled = snapshot_enabled
        self._owner = object()
        self._revision = 0
        self._messages_json = "[]"
        self._renderer_fingerprint = None

    def committed_messages(self):
        return json.loads(self._messages_json)

    def prepare(self, messages, *, workspace, count: Callable, token_limit: int,
                renderer_fingerprint: str, required_free_tokens: int = 1):
        if (type(token_limit) is not int or token_limit <= 0
                or type(required_free_tokens) is not int
                or not 0 <= required_free_tokens < token_limit):
            raise ValueError("Require positive token limit and valid free-token requirement")
        staged = self.stage(messages, workspace=workspace,
                            renderer_fingerprint=renderer_fingerprint)
        # Count a detached copy: counting must never become an implicit mutation.
        input_tokens = count(staged.messages())
        if type(input_tokens) is not int or input_tokens < 0:
            raise ValueError("The serving renderer must return a nonnegative integer token count")
        if input_tokens + required_free_tokens > token_limit:
            raise CapacityExceeded(input_tokens, token_limit, required_free_tokens)
        return replace(staged, input_tokens=input_tokens)

    def stage(self, messages, *, workspace, renderer_fingerprint: str):
        """Freeze a candidate without inventing a client-side resident count.

        Stateful transport uses this form: the server validates capacity against
        its authoritative token ledger and returns the accepted logical input
        count. Commit then requires that explicit returned count.
        """
        if not isinstance(renderer_fingerprint, str) or not renderer_fingerprint:
            raise ValueError("A nonempty rendering configuration fingerprint is required")
        if (self._renderer_fingerprint is not None
                and renderer_fingerprint != self._renderer_fingerprint):
            raise AppendOnlyViolation("Serving tokenizer/template/tools/image configuration changed")
        if not isinstance(messages, list) or any(not isinstance(m, dict) for m in messages):
            raise ValueError("messages must be a list of dictionaries")
        current = json.loads(_json(messages))
        previous = self.committed_messages()
        if len(current) < len(previous):
            raise AppendOnlyViolation("Previously submitted messages were removed")
        for index, old in enumerate(previous):
            candidate = current[index]
            if _is_observation(old):
                # A current suffix builder may already have rewrapped this
                # observation; only its immutable original is accepted as input.
                # Always restore the first committed version, including suffixes.
                unchanged = observation_base(candidate) == observation_base(old)
                if unchanged:
                    current[index] = old
            else:
                unchanged = candidate == old
            if not unchanged:
                raise AppendOnlyViolation(f"Previously submitted message {index} changed")
        new_observations = [i for i in range(len(previous), len(current))
                            if _is_observation(current[i])]
        if len(new_observations) > 1:
            raise AppendOnlyViolation("Cannot reconstruct multiple unseen observations from one workspace")
        if new_observations:
            index = new_observations[0]
            current[index] = decorate_observation(
                current[index], current[:index], workspace,
                program_enabled=self.program_enabled, snapshot_enabled=self.snapshot_enabled)
        return PreparedHistory(self._owner, self._revision, _json(current),
                               renderer_fingerprint, None,
                               len(current) - len(previous))

    def commit(self, prepared: PreparedHistory, *, accepted_input_tokens=None):
        if not isinstance(prepared, PreparedHistory) or prepared._owner is not self._owner:
            raise AppendOnlyViolation("Prepared history belongs to a different game/session")
        if prepared._revision != self._revision:
            raise AppendOnlyViolation("Prepared history is stale or already committed")
        if prepared.input_tokens is None and (type(accepted_input_tokens) is not int
                                              or accepted_input_tokens < 0):
            raise AppendOnlyViolation("Staged history needs the server's accepted input token count")
        self._messages_json = prepared._messages_json
        self._renderer_fingerprint = prepared.renderer_fingerprint
        self._revision += 1


def assert_token_append(acknowledged: Sequence[int], candidate: Sequence[int]):
    """Return newly appended exact token IDs, refusing any acknowledged rewrite.

    The acknowledged sequence must include actual generated token IDs. Neither
    decoded strings nor normalized assistant messages are an acceptable source.
    This check alone does not validate positions, embeddings or recurrent state.
    """
    old, new = tuple(acknowledged), tuple(candidate)
    if any(type(token) is not int or token < 0 for token in (*old, *new)):
        raise ValueError("Expected nonnegative integer token IDs")
    if len(new) < len(old):
        raise AppendOnlyViolation("Acknowledged token stream was shortened")
    for index, token in enumerate(old):
        if new[index] != token:
            raise AppendOnlyViolation(f"Acknowledged token prefix changed at position {index}")
    return new[len(old):]
