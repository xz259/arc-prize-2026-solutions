"""Imports of ARC-3 modules (never copies), checked against shared_modules.json.

Only the persistent-session transport is shared in v0. Its root (the ARC-3 harness_v6 directory) is, in order:
ARC2_ARC3_HARNESS_ROOT, the pod path of the extracted arc3-code archive, the local v12 snapshot. A file whose sha256
differs from the pin is refused unless ARC2_ALLOW_UNPINNED=1 (then the mismatch is reported in the run settings).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import threading
from pathlib import Path

HARNESS = Path(__file__).resolve().parent.parent                       # arc2_stack/harness
PINS = json.loads((HARNESS / "shared_modules.json").read_text())["modules"]
ROOTS = [
    os.environ.get("ARC2_ARC3_HARNESS_ROOT"),
    "/workspace/kvring_dev/fin74/s2/harness_v6",
    str(HARNESS.parent.parent / "inference_stack/code_snapshot_20261028/bundle/workspace/kvring_dev/fin74/s2/harness_v6"),
]


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def locate(rel):
    for root in ROOTS:
        if root and (Path(root) / rel).is_file():
            return Path(root) / rel
    raise FileNotFoundError(f"{rel} not found under any ARC-3 harness root: {[r for r in ROOTS if r]}")


def load_shared(rel, module_name):
    path = locate(rel)
    sha = _sha256(path)
    pinned = PINS.get(rel)
    status = "pinned" if sha == pinned else "UNPINNED"
    if sha != pinned and os.environ.get("ARC2_ALLOW_UNPINNED") != "1":
        raise RuntimeError(f"{path}: sha256 {sha} differs from the pin {pinned} (set ARC2_ALLOW_UNPINNED=1 to override)")
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module            # dataclasses resolve annotations through sys.modules
    spec.loader.exec_module(module)
    module.ARC2_SHARED = dict(path=str(path), sha256=sha, status=status)
    return module


def load_transport():
    return load_shared("v62/transport.py", "arc2_arc3_transport")


# Per-reply cues and caps (ARC-3's 12K cue, 16K close, open-call extension, repetition cue / close). Run 1 holds them off
# (the user, 2026-10-29: examine the minimal harness's outputs first): requests stay bounded at `request_tokens` (a length
# pause the agent continues in place), nothing is injected into a reply, and the repetition detector is off. 'arc3'
# restores ARC-3 production's values with the ARC-2 cue wording.
ARC3_CAPS_ENV = {"ARC3_ASSISTANT_TURN_CAPS": "12000,16000", "ARC3_ASSISTANT_TURN_CHECK": "8000",
                 "ARC3_ASSISTANT_TURN_CALL_EXT": "4096", "ARC3_V76_CUE_MARK": "0"}
CAPS_ENV_KEYS = tuple(ARC3_CAPS_ENV)


CAP_STATE = threading.local()          # v2: the agent writes (used, budget) before each generate; the status cue reads it


def v2_decision(original, *, close, ext, status_every, status_text):
    """v2 replies (the user, 2026-10-29): no soft cue; every `status_every` tokens of a reply, while its reasoning is
    open and no tool call is open, an in-reply status note (the reply's tokens and the puzzle's spent budget); the hard
    close at `close` with ARC-3's open-call extension; repetition handling is ARC-3's (cue, then close)."""
    def decide(cumulative, think_closed, soft, hard, cued, *, check=0, repeat_onset=None, strikes=0, nudge=0,
               nudged=False, call_open=False, call_extension=0):
        if repeat_onset is not None or cumulative >= close:
            return original(cumulative, think_closed, 0, close, cued, check=check, repeat_onset=repeat_onset,
                            strikes=strikes, call_open=call_open, call_extension=ext)
        nxt = min(close, (cumulative // status_every + 1) * status_every) if status_every else close
        limit = min(nxt - cumulative, check) if check else nxt - cumulative
        injection = None
        if status_every and cumulative > 0 and cumulative % status_every == 0 and not think_closed and not call_open:
            used = getattr(CAP_STATE, "used", None)
            injection = {"text": status_text.format(reply=cumulative, used=cumulative if used is None else used,
                                                    budget=getattr(CAP_STATE, "budget", None) or 0),
                         "kind": "soft_nudge", "close": False}
        return limit, injection, None
    return decide


def wrap_text(texts, used, budget, wrap_at):
    """The wrap-up reminder (the user, 2026-10-29: from wrap_at tokens spent, more urgent as the pass budget nears): the most
    urgent tier of texts['wrap_tiers'] ([max tokens left, text]) that applies, or '' below wrap_at."""
    if not budget or wrap_at is None or used < wrap_at:
        return ""
    left = max(0, budget - used)
    for max_left, text in sorted(texts["wrap_tiers"], key=lambda t: t[0]):
        if left <= max_left:
            return " " + text.format(left=left)
    return ""


def spent_text(texts, used, budget, limit_at):
    """v2d (the user, 2026-10-29): the tokens spent, with the budget named only from limit_at tokens spent."""
    if limit_at is None:
        return ""
    return texts["spent_limit_v2d" if used >= limit_at else "spent_v2d"].format(used=used, budget=budget)


def v2c_decision(original, *, close, ext, status_every, status_text, texts, wrap_at, cue_from=0):
    """v2c (the user, 2026-10-29): as v2_decision, but the status note falls every `status_every` tokens of the PASS
    (CAP_STATE.used: the pass's tokens at this request, this reply's included) and carries the wrap-up reminder from
    `wrap_at`; requests end on those pass-total boundaries. cue_from (v2d): no status notes before this many tokens of the
    pass. CAP_STATE.cue_from / CAP_STATE.wrap_at, when set, override both for the calling thread's attempt (2026-10-30:
    review sessions run with the reviewer experiment's 45,000 in the same runner as the passes' 120,000)."""
    def decide(cumulative, think_closed, soft, hard, cued, *, check=0, repeat_onset=None, strikes=0, nudge=0,
               nudged=False, call_open=False, call_extension=0):
        if repeat_onset is not None or cumulative >= close:
            return original(cumulative, think_closed, 0, close, cued, check=check, repeat_onset=repeat_onset,
                            strikes=strikes, call_open=call_open, call_extension=ext)
        used = getattr(CAP_STATE, "used", None)
        budget = getattr(CAP_STATE, "budget", None) or 0
        cf = getattr(CAP_STATE, "cue_from", None)
        cf = cue_from if cf is None else cf
        wa = getattr(CAP_STATE, "wrap_at", None)
        wa = wrap_at if wa is None else wa
        total = cumulative if used is None else used
        limit = close - cumulative
        nxt = (total // status_every + 1) * status_every if status_every else 0
        if status_every and nxt >= cf:
            limit = min(limit, nxt - total)
        if check:
            limit = min(limit, check)
        injection = None
        if (status_every and cumulative > 0 and total % status_every == 0 and total >= cf and not think_closed
                and not call_open):
            injection = {"text": status_text.format(used=total, budget=budget,
                                                    wrap=wrap_text(texts, total, budget, wa)),
                         "kind": "soft_nudge", "close": False}
        return max(1, limit), injection, None
    return decide


def configure_transport(transport, *, texts, reply_caps="off", request_tokens=8000, custom=(24000, 32000, 4096),
                        v2=None):
    """Apply the ARC-2 cue wording and the reply-caps mode; returns the settings record. Call before any session is
    constructed (PersistentSession reads the caps from the environment in its constructor). 'custom' is ARC-3's
    mechanism (cue, close, open-call extension, repetition cue / close) at the given (cue, close, extension) values."""
    from .prompts import apply_transport_texts
    replaced = apply_transport_texts(transport, texts)
    if reply_caps == "off":
        os.environ.update(ARC3_ASSISTANT_TURN_CAPS="1000000,1000001", ARC3_ASSISTANT_TURN_CHECK=str(int(request_tokens)),
                          ARC3_ASSISTANT_TURN_CALL_EXT="0", ARC3_V76_CUE_MARK="0")

        def no_caps(cumulative, think_closed, soft, hard, cued, *, check=0, **_):
            return (check or None), None, None
        transport.turn_cap_decision = no_caps
        transport.repetition_onset = lambda ids, **_: None
    elif reply_caps == "arc3":
        for k, v in ARC3_CAPS_ENV.items():
            os.environ.setdefault(k, v)
    elif reply_caps == "v2":
        v2 = dict(v2 or {})
        close, ext = int(v2.get("close", 32000)), int(v2.get("ext", 4096))
        os.environ.update(ARC3_ASSISTANT_TURN_CAPS=f"{close - 1},{close}", ARC3_ASSISTANT_TURN_CHECK=str(int(request_tokens)),
                          ARC3_ASSISTANT_TURN_CALL_EXT=str(ext), ARC3_V76_CUE_MARK="0")
        transport.turn_cap_decision = v2_decision(transport.turn_cap_decision, close=close, ext=ext,
                                                  status_every=int(v2.get("status_every", 12000)),
                                                  status_text=texts[v2.get("status_key", "status_cue")])
    elif reply_caps == "v2c":
        v2 = dict(v2 or {})
        close, ext = int(v2.get("close", 32000)), int(v2.get("ext", 4096))
        os.environ.update(ARC3_ASSISTANT_TURN_CAPS=f"{close - 1},{close}", ARC3_ASSISTANT_TURN_CHECK=str(int(request_tokens)),
                          ARC3_ASSISTANT_TURN_CALL_EXT=str(ext), ARC3_V76_CUE_MARK="0")
        transport.turn_cap_decision = v2c_decision(transport.turn_cap_decision, close=close, ext=ext,
                                                   status_every=int(v2.get("status_every", 10000)),
                                                   status_text=texts["status_cue_v2c"], texts=texts,
                                                   wrap_at=v2.get("wrap_at"), cue_from=int(v2.get("cue_from") or 0))
    elif reply_caps == "custom":
        cue, close, ext = (int(x) for x in custom)
        if not 0 < cue < close:
            raise ValueError(f"custom caps need 0 < cue < close, got {custom!r}")
        os.environ.update(ARC3_ASSISTANT_TURN_CAPS=f"{cue},{close}", ARC3_ASSISTANT_TURN_CHECK=str(int(request_tokens)),
                          ARC3_ASSISTANT_TURN_CALL_EXT=str(ext), ARC3_V76_CUE_MARK="0")
    else:
        raise ValueError(f"reply_caps must be 'off', 'arc3', 'custom', 'v2' or 'v2c', not {reply_caps!r}")
    return dict(reply_caps=reply_caps, request_tokens=int(request_tokens), texts_replaced=replaced,
                env={k: os.environ.get(k) for k in CAPS_ENV_KEYS})
