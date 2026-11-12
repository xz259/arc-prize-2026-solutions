"""Local stand-in for the transport's PersistentSession (runner --fake): scripted replies, no server.

Each reply is ("code", source) for one python call, ("args", raw) for one python call with these raw arguments (v2q), ("final",
text) for a reply without a tool call, or ("pause", n)
for a length pause of n tokens before the next scripted reply. The default script defines an identity transform,
checks it, submits it and finishes, so a dry run exercises the kernel, the turn loop, the logs, the submission file
and the scorer end to end.
"""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field

DEFAULT_SCRIPT = [
    ("pause", 300),
    ("code", "def transform(grid):\n    return grid.copy()\n\ncheck(transform)"),
    ("code", "history()\nprograms()"),
    ("code", "submit(transform)"),
    ("final", "The answer is recorded."),
]


@dataclass
class FakeCompletion:
    message: dict
    finish_reason: str
    usage: dict
    token_ids: list
    assistant_token_ids: list
    pending: bool
    closed_by: str | None = None
    persistent: dict = field(default_factory=dict)
    seconds: float = 0.0
    cues: tuple = ()


class FakeSession:
    def __init__(self, script=None, *, log=None, tokens_per_reply=400):
        self.script = list(script if script is not None else DEFAULT_SCRIPT)
        self.log = log or (lambda kind, payload: None)
        self.tokens_per_reply = tokens_per_reply
        self.session_id = "fake-" + uuid.uuid4().hex
        self.evicted_ranges = []
        self.pending = False
        self.closed = False
        self.calls = 0
        self.prompt = 0
        self.received = []

    def generate(self, external):
        if self.closed:
            raise RuntimeError("session closed")
        if external and self.pending:
            raise RuntimeError("cannot append while a reply is pending")
        self.received.append(external)
        self.prompt += sum(len(json.dumps(m)) // 4 for m in external)
        self.log("persistent-request", dict(fake=True, external=len(external)))
        step = self.script.pop(0) if self.script else ("final", "Nothing more to do.")
        if step[0] == "pause":
            self.pending = True
            return FakeCompletion(message={"role": "assistant", "content": ""}, finish_reason="length",
                                  usage=dict(prompt_tokens=self.prompt, completion_tokens=step[1]),
                                  token_ids=[0] * step[1], assistant_token_ids=[0] * step[1], pending=True)
        self.pending = False
        n = self.tokens_per_reply
        if step[0] in ("code", "args"):
            self.calls += 1
            raw = json.dumps({"code": step[1]}) if step[0] == "code" else step[1]
            msg = {"role": "assistant", "content": "",
                   "tool_calls": [{"id": f"call_{self.calls}", "type": "function",
                                   "function": {"name": "python", "arguments": raw}}]}
            finish = "tool_calls"
        else:
            msg = {"role": "assistant", "content": step[1]}
            finish = "stop"
        return FakeCompletion(message=msg, finish_reason=finish, usage=dict(prompt_tokens=self.prompt, completion_tokens=n),
                              token_ids=[0] * n, assistant_token_ids=[0] * n, pending=False)

    def close(self):
        self.closed = True
