"""Opt-in, sequential sessions with exact managed-suffix KV eviction.

No text-history reconstruction, prefix cache, or implicit retry.  The
manager is engine independent so ownership/cancellation can be exercised on CPU.
The backend is the only component allowed to pass inputs to the inference engine.
"""
from __future__ import annotations

import asyncio
import bisect
import hashlib
import json
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from residency import ResidentLedger, ResidencyError
from suffix_registry import SuffixRegistry
from history_eviction import HistoryPolicy, HistoryRegistry, HistoryValidationError, HistoryCapacityError, ObservationStart


class SessionError(ValueError):
    def __init__(self, message: str, status: int = 409, *, receipt=None):
        super().__init__(message)
        self.status = status
        self.receipt = receipt


def digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     ensure_ascii=False).encode()).hexdigest()


def token_digest(ids: list[int]) -> str:
    return digest(ids)


class _PrefixDigest:
    """token_digest() of a growing token list, hashed incrementally (2026-09-25 O(L) fast path). Exact: the hashed prefix
    is a private copy compared with the current list's prefix on every call; a shorter or rewritten list rehashes."""
    __slots__ = ("ids", "hasher")

    def __init__(self):
        self.ids = []; self.hasher = hashlib.sha256(b"[")

    def hexdigest(self, current):
        n = len(self.ids)
        if n and (len(current) < n or current[:n] != self.ids):
            self.ids = []; self.hasher = hashlib.sha256(b"["); n = 0
        if len(current) > n:
            tail = current[n:]
            self.hasher.update((("," if n else "") + json.dumps(tail, sort_keys=True, separators=(",", ":"),
                                                                ensure_ascii=False)[1:-1]).encode())
            self.ids.extend(tail)
        final = self.hasher.copy(); final.update(b"]")
        return final.hexdigest()


def _ol_flag(name, default):
    value = os.environ.get(name, default)
    if value not in ("0", "1"):
        raise RuntimeError(f"{name} must be 0 or 1")
    return value == "1"


def session_token_digest(session) -> str:
    """token_digest() of the session token list; incremental per session (rebase replaces the list and rehashes once)."""
    ids = session.token_ids
    if not _ol_flag("ARC3_OL_FASTPATH", "1") or type(ids) is not list:
        return token_digest(ids)
    cache = session.__dict__.get("_arc3_token_digest")
    if cache is None:
        cache = session.__dict__["_arc3_token_digest"] = _PrefixDigest()
    value = cache.hexdigest(ids)
    if _ol_flag("ARC3_OL_FASTPATH_VERIFY", "0") and value != token_digest(ids):
        raise RuntimeError("Incremental session token digest differs from the full digest")
    return value


def history_state(session):
    """History registry snapshot plus, per observation block, the state of the observation's own board (2026-09-27):
    board_resident (the block's first image labelled 'observation' is fully resident) and text_evicted_fraction (evicted
    share of the observation's text before that board); both None when the block has no labelled board. The persistent
    client recaps only observations whose board has left the context (ARC3_RECAP_BOARD_RULE), so the recap never repeats
    a board that is still in view."""
    state = session.history_registry.snapshot()
    evicted = list(session.residency.evicted); ends = [b for _, b in evicted]
    def overlap(a, b):
        i = bisect.bisect_right(ends, a); total = 0
        while i < len(evicted) and evicted[i][0] < b:
            total += max(0, min(b, evicted[i][1]) - max(a, evicted[i][0])); i += 1
        return total
    kinds = session.image_kinds or {}
    boards = sorted((int(a), int(b)) for a, b in (session.images or []) if kinds.get(int(a)) == "observation")
    starts = [a for a, _ in boards]
    for block in state["blocks"]:
        start, end = block["start"], block["end"]
        i = bisect.bisect_left(starts, start)
        board = boards[i] if i < len(boards) and (end is None or boards[i][0] < end) else None
        if board is None:
            block.update(board_resident=None, text_evicted_fraction=None)
            continue
        text = board[0] - start
        block.update(board_resident=overlap(*board) == 0,
                     text_evicted_fraction=round(overlap(start, board[0]) / text, 4) if text > 0 else None)
    return state


@dataclass
class TurnInput:
    prompt: dict
    sampling: dict
    parse_context: Any = None
    suffix_records: list = field(default_factory=list)
    evict_observation_ids: list = field(default_factory=list)
    observation_starts: list = field(default_factory=list)
    injected_ids: list = field(default_factory=list)   # assistant-turn budget: harness-authored assistant tokens
    injection: dict | None = None
    tail_eviction: dict | None = None                  # harness-requested removal of a degenerate run in the paused turn

    @property
    def token_ids(self):
        return self.prompt["prompt_token_ids"]


@dataclass
class Session:
    session_id: str
    engine_request_id: str
    initial: dict
    revision: int = 0
    token_ids: list[int] = field(default_factory=list)
    state: str = "new"
    finish_reason: str | None = None
    operations: dict = field(default_factory=dict)
    current_operation: str | None = None
    input_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    output_future: asyncio.Future | None = None
    collector: asyncio.Task | None = None
    parse_context: Any = None
    error: str | None = None
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    residency: ResidentLedger = field(default_factory=ResidentLedger)
    suffix_registry: SuffixRegistry = field(default_factory=SuffixRegistry)
    history_policy: HistoryPolicy = field(default_factory=HistoryPolicy)
    history_registry: HistoryRegistry = field(default_factory=HistoryRegistry)
    eviction_revision: int = 0
    frequency_epoch_start: int = 0
    frequency_epoch_reset_pending: bool = False
    eviction_endpoint_profile: dict | None = None
    close_task: asyncio.Task | None = None
    segments: list = field(default_factory=list)   # stage 12: (kind, start, end) turn segments
    structural: list = field(default_factory=list)   # stage 12 split unit: logical positions of chat-template control tokens
    images: list = field(default_factory=list)       # logical [start, end) spans of <|vision_start|>..<|vision_end|> (evicted whole or not at all)
    image_kinds: dict = field(default_factory=dict)  # image start -> observation | recap | tool | animation | other (2026-09-27)
    open_image: int | None = None                    # a vision span opened in an earlier append
    open_tool_call: int | None = None                # structure-preserving QSA (2026-09-24): an open <tool_call> position
    prepare_profile: dict = field(default_factory=dict)   # API-side snapshot/frequency/plan timings of the last append
    parts: list = field(default_factory=list)        # eviction-NLL probe: (logical start, rendered multimodal part) per image-bearing append
    probe_semaphore: Any = None
    rebases: int = 0                                 # position rebase (2026-09-24): completed frame switches
    rebase_info: dict | None = None                  # the rebase performed by the current turn (response/audit), else None




# ---------------------------------------------------------------- eviction-NLL probe (2026-09-23)
def nll_probe_rate():
    value = os.environ.get("ARC3_EVICTION_NLL_PROBE", "0")
    if not value.isdigit():
        raise RuntimeError("ARC3_EVICTION_NLL_PROBE must be a nonnegative integer")
    return int(value)


def nll_probe_tokens():
    value = os.environ.get("ARC3_EVICTION_NLL_PROBE_TOKENS", "1024")
    if not value.isdigit() or int(value) < 1:
        raise RuntimeError("ARC3_EVICTION_NLL_PROBE_TOKENS must be a positive integer")
    return int(value)


def build_probe_prompt(token_ids, survivors, gen_end, images, parts, suffix_ids=()):
    """The holed context as a fresh engine input: resident tokens of token_ids[:gen_end] (survivors = the ledger's
    survivor intervals) plus suffix_ids, with every surviving image's processed inputs copied from the rendered part
    that carried it and its placeholder offset shifted into the holed stream. Raises if an image cannot be mapped."""
    from dataclasses import replace
    holed = []; base = {}   # base[(a, b)] = holed index of logical position a
    for a, b in survivors:
        b = min(b, gen_end)
        if a >= b:
            continue
        base[(a, b)] = len(holed); holed.extend(token_ids[a:b])
    items, hashes, ranges, container = [], [], [], None
    parts = sorted(parts, key=lambda p: p[0])
    for s, e in images:
        s, e = int(s), int(e)
        if s >= gen_end:
            continue
        hit = next(((a, b) for (a, b) in base if a <= s and e <= b), None)
        if hit is None:
            if any(a < e and s < b for (a, b) in base):
                raise RuntimeError(f"image [{s},{e}) is partially resident")
            continue                                                   # evicted whole
        # parts are (logical start, appended length, rendered prompt): the length is recorded at append time because the
        # initial prompt's token list is shared with vLLM's per-request state and grows as inputs stream in.
        part = next((p for p in parts if p[0] <= s < p[0] + p[1]), None)
        if part is None or part[2].get("type") != "multimodal":
            raise RuntimeError(f"no rendered multimodal part covers image at {s}")
        start, _, prompt = part
        spans = list(prompt["mm_placeholders"].get("image", ()))
        k = next((i for i, sp in enumerate(spans) if start + sp.offset == s + 1), None)
        if k is None or spans[k].length != e - s - 2:
            raise RuntimeError(f"no placeholder matches image [{s},{e}) in the part at {start}")
        new_offset = base[hit] + (s - hit[0]) + 1
        ranges.append(replace(spans[k], offset=new_offset))
        items.append(prompt["mm_kwargs"]["image"][k]); hashes.append(prompt["mm_hashes"]["image"][k])
        container = container or type(prompt["mm_kwargs"])
    ids = list(holed) + list(suffix_ids)
    if not items:
        return {"type": "token", "prompt_token_ids": ids}, len(holed), 0
    return {"type": "multimodal", "prompt_token_ids": ids, "mm_kwargs": container({"image": items}),
            "mm_hashes": {"image": hashes}, "mm_placeholders": {"image": ranges}}, len(holed), len(items)


def merge_prompts(prefix, delta):
    """Concatenate two rendered engine inputs (the rebuilt holed prefix and the pending delta): token ids appended,
    the delta's image placeholders shifted by the prefix length, image inputs and hashes appended in order."""
    from dataclasses import replace
    ids = list(prefix["prompt_token_ids"]) + list(delta["prompt_token_ids"]); off = len(prefix["prompt_token_ids"])
    items, hashes, ranges, container = [], [], [], None
    for part, shift in ((prefix, 0), (delta, off)):
        if part.get("type") != "multimodal":
            continue
        container = container or type(part["mm_kwargs"])
        items += list(part["mm_kwargs"]["image"]); hashes += list(part["mm_hashes"]["image"])
        ranges += [replace(r, offset=r.offset + shift) for r in part["mm_placeholders"]["image"]]
    if not items:
        return {"type": "token", "prompt_token_ids": ids}
    return {"type": "multimodal", "prompt_token_ids": ids, "mm_kwargs": container({"image": items}),
            "mm_hashes": {"image": hashes}, "mm_placeholders": {"image": ranges}}


def _mean_nll_sampled(rows, ids):
    """rows: the collector's per-token {str(token): {logprob, rank}} for the sampled tokens (logprobs=0)."""
    vals = []
    for row, tok in zip(rows, ids):
        entry = row.get(str(tok)) if isinstance(row, dict) else None
        if entry is None:
            return None
        vals.append(-float(entry["logprob"]))
    return (sum(vals) / len(vals)) if vals else None


def _mean_nll_prompt(prompt_logprobs, ids, offset):
    """prompt_logprobs: vLLM PromptLogprobs (list of {token_id: Logprob} | None); score ids at prompt positions offset.."""
    vals = []
    for j, tok in enumerate(ids):
        row = prompt_logprobs[offset + j] if offset + j < len(prompt_logprobs) else None
        entry = row.get(tok) if isinstance(row, dict) else None
        if entry is None:
            return None
        vals.append(-float(getattr(entry, "logprob", entry)))
    return (sum(vals) / len(vals)) if vals else None


class SessionManager:
    def __init__(self, backend, *, max_context: int, max_sessions: int,
                 audit_path: str | None = None, turn_timeout: float = 600,
                 max_logical_tokens: int = 262144):
        self.backend = backend
        self.max_context = max_context
        if type(max_logical_tokens) is not int or max_logical_tokens < max_context:
            raise ValueError("Logical position limit must cover the resident context limit")
        self.max_logical_tokens = max_logical_tokens
        self.max_sessions = max_sessions
        self.turn_timeout = turn_timeout
        self.sessions: dict[str, Session] = {}
        self.lock = asyncio.Lock()
        self.audit_path = Path(audit_path) if audit_path else None
        if self.audit_path:
            self.audit_path.parent.mkdir(parents=True, exist_ok=True)

    CONTROL_TOKENS = ("<|im_start|>", "<|im_end|>", "<think>", "</think>", "<tool_call>", "</tool_call>",
                      "<tool_response>", "</tool_response>")

    def _control_ids(self):
        """{token id: token} for the chat-template control tokens; resolved once from the backend tokenizer."""
        cached = getattr(self, "_control_ids_cache", None)
        if cached is None:
            cached = {}
            tokenizer = getattr(self.backend, "tokenizer", None)
            unknown = getattr(tokenizer, "unk_token_id", None)
            for token in self.CONTROL_TOKENS:
                try: ident = tokenizer.convert_tokens_to_ids(token) if tokenizer is not None else None
                except Exception: ident = None
                if type(ident) is int and ident >= 0 and ident != unknown: cached[ident] = token
            self._control_ids_cache = cached
        return cached

    IMAGE_TOKENS = ("<|vision_start|>", "<|vision_end|>")

    def _image_ids(self):
        cached = getattr(self, "_image_ids_cache", None)
        if cached is None:
            cached = {}
            tokenizer = getattr(self.backend, "tokenizer", None)
            unknown = getattr(tokenizer, "unk_token_id", None)
            for token in self.IMAGE_TOKENS:
                try: ident = tokenizer.convert_tokens_to_ids(token) if tokenizer is not None else None
                except Exception: ident = None
                if type(ident) is int and ident >= 0 and ident != unknown: cached[ident] = token
            self._image_ids_cache = cached
        return cached

    def _mark_images(self, session, ids, offset):
        """Index image spans in newly committed ids (O(len(ids))); a span may open in one append and close in a later one."""
        table = self._image_ids()
        if len(table) != 2: return
        state = {}
        for k, ident in enumerate(ids):
            token = table.get(ident)
            if token == "<|vision_start|>": session.open_image = offset + k
            elif token == "<|vision_end|>" and session.open_image is not None:
                session.images.append([session.open_image, offset + k + 1])
                session.image_kinds[session.open_image] = self._image_kind(session, session.open_image, state)
                session.open_image = None


    def _image_kind(self, session, start, state):
        """Label a committed image span (per-kind QSA image rank, 2026-09-27): the segment it arrived in and the text
        before it. state carries, per observation segment, the last scanned position and whether "Recap: the" was seen."""
        seg = None
        for kind, a, b in reversed(session.segments):
            if a <= start < b:
                seg = (kind, a); break
            if b <= start:
                break
        tokenizer = getattr(self.backend, "tokenizer", None)
        def text(a, b):
            try:
                return tokenizer.decode(session.token_ids[max(0, a):b]) if tokenizer is not None and b > a else ""
            except Exception:
                return ""
        if seg is None:
            return "other"
        if seg[0] == "tool":   # fix 1: the caption lies between the previous image (or the segment start) and this one
            prev_end = seg[1]
            for a, b in reversed(getattr(session, "images", None) or []):
                if b <= start:
                    prev_end = max(prev_end, int(b)); break
            before = text(max(prev_end, start - 2048), start)
            if "Animation, action event" in before:
                return "animation"
            return "tool" if "Board image" in before else "other"
        if seg[0] == "observation":
            last, seen = state.get(seg[1], (seg[1], False))
            seen = seen or "Recap: the" in text(last, start)
            state[seg[1]] = (start, seen)
            return "recap" if seen else "observation"
        return "other"

    def _wrapper_ids(self):
        """Token ids used to delimit a tool call's formatting (structure-preserving QSA (2026-09-24)); {} without a tokenizer."""
        cached = getattr(self, "_wrapper_ids_cache", None)
        if cached is None:
            cached = {}
            tokenizer = getattr(self.backend, "tokenizer", None)
            try:
                enc = lambda t: tokenizer.encode(t, add_special_tokens=False)
                cached = {"param": enc("</parameter>")[1], "func": enc("</function>")[1], "close_slash": enc("</parameter>")[0],
                          "newline": enc("\n")[0]}
                self._wrapper_tokenizer = tokenizer
            except Exception:
                cached = {}
            self._wrapper_ids_cache = cached
        return cached

    def _ends_with_gt(self, ident):
        tok = getattr(self, "_wrapper_tokenizer", None)
        memo = self.__dict__.setdefault("_gt_memo", {})
        if ident not in memo:
            try: memo[ident] = tok.decode([ident]).rstrip().endswith(">")
            except Exception: memo[ident] = False
        return memo[ident]

    def _tool_call_wrappers(self, toks, s, e):
        """Positions of a tool call's formatting in toks[s..e] (s: <tool_call>, e: </tool_call>): header through the
        parameter tag's line, footer from the closing tag before the code's end; the code body stays evictable."""
        w = self._wrapper_ids()
        if not w:
            return list(range(s, e + 1))
        h = None; lim = min(e, s + 24)
        for i in range(s + 1, lim):
            if toks[i] == w["param"]:
                for j in range(i + 1, min(lim, i + 8)):
                    if self._ends_with_gt(toks[j]):
                        h = j; break
                break
        if h is None:                                  # no parameter tag: through the function tag's line
            for i in range(s + 1, lim):
                if toks[i] == w["func"]:
                    for j in range(i + 1, min(lim, i + 8)):
                        if self._ends_with_gt(toks[j]):
                            h = j; break
                    break
        h = s if h is None else h
        if h + 1 < e and toks[h + 1] == w["newline"]:
            h += 1
        f = e
        for i in range(e - 1, max(h, e - 16), -1):     # last '</' followed by the parameter word, else the last '</'
            if toks[i] == w["close_slash"]:
                if i + 1 < e and toks[i + 1] == w["param"]:
                    f = i; break
                f = i if f == e else f
        if f > h + 1 and toks[f - 1] == w["newline"]:
            f -= 1
        return list(range(s, h + 1)) + list(range(max(f, h + 1), e + 1))

    def _mark_tool_calls(self, session, ids, offset):
        """Index tool-call formatting when a </tool_call> closes (the call may span pauses). O(len(ids))."""
        table = self._control_ids()
        rev = {tok: ident for ident, tok in table.items()}
        open_id, close_id = rev.get("<tool_call>"), rev.get("</tool_call>")
        if open_id is None or close_id is None:
            return
        marks = []
        for k, ident in enumerate(ids):
            if ident == open_id:
                session.open_tool_call = offset + k
            elif ident == close_id and session.open_tool_call is not None:
                marks.extend(self._tool_call_wrappers(session.token_ids, session.open_tool_call, offset + k))
                session.open_tool_call = None
        if marks:
            from bisect import bisect_left
            out = session.structural; j = bisect_left(out, min(marks))
            out[j:] = sorted(set(out[j:]) | set(marks))

    def _mark_structural(self, session, ids, offset):
        """Index control-token positions in newly committed ids (O(len(ids))); the role word and newline after a turn opener too."""
        table = self._control_ids()
        if not table: return
        out = session.structural; n = len(ids)
        for k, ident in enumerate(ids):
            if ident in table:
                out.append(offset + k)
                if table[ident] == "<|im_start|>":
                    out.extend(offset + k + j for j in (1, 2) if k + j < n)

    # ------------------------------------------------------------------ position rebase (2026-09-24)
    @staticmethod
    def _rebase_enabled():
        return os.environ.get("ARC3_REBASE", "1") == "1"

    def _rebase_threshold(self):
        at = os.environ.get("ARC3_REBASE_AT")
        if at:
            return int(at)
        return self.max_logical_tokens - int(os.environ.get("ARC3_REBASE_MARGIN", "24576"))   # 2026-09-25: 24K default

    def _rebase_due(self, session, new_ids, *, initial, continuation):
        return bool(self._rebase_enabled() and not initial and not continuation and new_ids
                    and session.history_policy.policy != "oldest_recompute" and session.history_registry.observations
                    and len(session.token_ids) + len(new_ids) >= self._rebase_threshold())

    def _record_part(self, session, start, length, prompt):
        """Rendered multimodal input of an image-bearing append; only the last 2 x context tokens can be rebuilt."""
        session.parts.append((start, length, prompt))
        if not nll_probe_rate() and session.history_policy.policy != "oldest_recompute":
            horizon = len(session.token_ids) - 2 * self.max_context   # token_ids already include this append
            session.parts = [q for q in session.parts if q[0] + q[1] > horizon]

    @staticmethod
    def _rebase_window(session, target):
        """(w, kind, tail_start): earliest observation start (else assistant start) w with P + (end - w) <= target and
        w <= the pair-guarded protected-tail boundary (the planner's rule for the split unit)."""
        end = len(session.token_ids); obs = session.history_registry.observations; P = obs[0].start
        protect = int(getattr(session.history_policy, "protect_recent_tokens", 0) or 0)
        hend = max(P, end - protect); owner = None; prev = None
        for kind, a, b in session.segments:
            if kind == "assistant": prev = a
            if a < hend < b or (a == hend and kind != "assistant"):
                owner = prev; break
            if a > hend: break
        tail = max(P, owner) if owner is not None and owner < hend else hend
        fits = lambda w: P + (end - w) <= target
        for kind, cands in (("interaction", [o.start for o in obs]),
                            ("assistant", [a for k, a, b in session.segments if k == "assistant" and a >= P])):
            ok = [w for w in cands if P <= w <= tail and fits(w)]
            if ok:
                return min(ok), kind, tail
        raise RuntimeError(f"rebase window does not fit: prefix {P}, protected tail from {tail}, end {end}, target {target}")

    async def _rebase(self, session, turn, local_history_tokens=0):
        """Rebuild the session as prefix + verbatim recent window in a new coordinate frame and restart the engine request."""
        started = time.monotonic()
        if session.open_image is not None:
            raise RuntimeError("rebase with an open image span")
        open_call = getattr(session, "open_tool_call", None)   # review fix 2026-09-25: remapped below, never a refusal
        old_len = len(session.token_ids); resident_now = session.residency.resident_tokens
        obs = session.history_registry.observations; P = obs[0].start
        # user 2026-09-24: aim for the split-A length (the policy basis), so about a tail's worth of new tokens is generated,
        # with fresh QSA statistics, before the first eviction; fall back to the current resident count if A cannot hold
        # the protected tail
        basis = int(getattr(session.history_policy, "target_basis_tokens", 0) or 0); fallback = False
        target = basis or resident_now
        try:
            w, kind, tail = self._rebase_window(session, target)
        except RuntimeError:
            if target == resident_now:
                raise
            target = resident_now; fallback = True
            w, kind, tail = self._rebase_window(session, target)
        keep = [(0, P), (w, old_len)] if w > P else [(0, old_len)]
        prompt, resident, n_images = build_probe_prompt(session.token_ids, keep, old_len,
                                                        [tuple(x) for x in session.images], list(session.parts), ())
        new_len = P + old_len - w
        if resident != new_len or len(prompt["prompt_token_ids"]) != new_len:
            raise RuntimeError(f"rebase prompt has {resident} tokens, window expects {new_len}")
        shift = w - P
        def m(q):                       # old position -> new position (None when dropped)
            return q if q < P else (q - shift if q >= w else None)
        ev = session.residency.evicted
        restored = sum(max(0, min(b, old_len) - max(a, w)) for a, b in ev)
        segments = []
        for sk, a, b in session.segments:
            if a < P:
                segments.append([sk, a, min(b, P) if w > P else b])
            elif a >= w:
                segments.append([sk, a - shift, b - shift])
        images = [[a - shift, b - shift] for a, b in session.images if a >= w] + [[a, b] for a, b in session.images if b <= P]
        images.sort()
        image_kinds = {(a - shift if a >= w else a): session.image_kinds.get(a, "other")   # 2026-09-27: labels follow the spans
                       for a, b in session.images if a >= w or b <= P}
        structural = [m(q) for q in session.structural if m(q) is not None]
        session.open_tool_call = m(open_call) if open_call is not None else None
        kept_obs = [o for o in obs if o.start >= w]
        owner_obs = [o for o in obs if o.start < w]
        partial = owner_obs[-1].observation_id if (owner_obs and kind == "assistant") else None
        dropped = [o.observation_id for o in owner_obs if o.observation_id != partial]
        new_obs = ([ObservationStart(partial, P)] if partial else []) + [ObservationStart(o.observation_id, o.start - shift) for o in kept_obs]
        if not new_obs or new_obs[0].start != P or any(x.start >= y.start for x, y in zip(new_obs, new_obs[1:])):
            raise RuntimeError(f"rebase observation registry would not start at the prefix or not increase: {new_obs[:3]}")
        reg = session.history_registry
        reg.observations = tuple(new_obs); reg.registered_through = new_len; reg.revision += 1
        sreg = session.suffix_registry
        for oid, (a, b) in list(sreg.live.items()):
            if a < w:
                raise RuntimeError(f"live managed suffix {oid} [{a},{b}) lies before the rebase window")
            sreg.live[oid] = (a - shift, b - shift)
        for oid, rec in sreg.records.items():
            span = rec.get("logical_token_range")
            if span is not None:
                rec["logical_token_range"] = (span[0] - shift, span[1] - shift) if span[0] >= w else None
        session.token_ids = list(prompt["prompt_token_ids"])
        session.residency = ResidentLedger(new_len)
        session.segments = segments; session.images = images; session.structural = structural; session.image_kinds = image_kinds
        session.parts = [(0, new_len, prompt)] if prompt.get("type") == "multimodal" else []
        session.frequency_epoch_start = new_len; session.frequency_epoch_reset_pending = False
        session.eviction_revision = 0
        session.rebases += 1
        turn.prompt = prompt; turn.recompute = True
        await self.backend.restart_request(session)
        info = {"rebase": session.rebases, "from_logical_tokens": old_len, "to_logical_tokens": new_len,
                "prefix_tokens": P, "window_start": w, "boundary": kind, "protected_tail_start": tail,
                "resident_before": resident_now, "target_tokens": target, "target_fallback": fallback,
                "restored_tokens": restored, "net_dropped_tokens": resident_now + local_history_tokens - new_len,
                "local_history_tokens": local_history_tokens, "stale_tool_call": open_call,
                "dropped_observation_ids": dropped, "partial_observation_id": partial, "images": n_images,
                "restart_seconds": time.monotonic() - started, "engine_request_id": session.engine_request_id}
        session.rebase_info = info
        self.audit("rebase", session, operation_id=session.current_operation, **info)
        return info

    async def _recompute_restart(self, session, plan, turn):
        """oldest_recompute: rebuild the engine request from the holed logical stream (the plan's survivors, i.e. the same
        removed ranges as the compaction policies) and fold the pending delta into its first prefill. No KV compaction;
        the acknowledgement is local. The caller then runs the turn as an initial request on the new id."""
        gen_end = len(session.token_ids)
        prefix, resident, n_images = build_probe_prompt(session.token_ids, list(plan.survivor_intervals), gen_end,
                                                        [tuple(x) for x in session.images], list(session.parts), ())
        if resident != plan.resident_after:
            raise RuntimeError(f"recompute prefix has {resident} tokens, plan expects {plan.resident_after}")
        turn.delta_prompt = turn.prompt
        turn.prompt = merge_prompts(prefix, turn.prompt); turn.recompute = True
        started = time.monotonic()
        await self.backend.restart_request(session)
        return {"plan_digest": plan.digest, "resident_tokens": plan.resident_after, "evicted_tokens": plan.removed_tokens,
                "logical_tokens": len(session.token_ids), "eviction_revision": session.eviction_revision + 1,
                "execution": "recompute", "prefill_tokens": len(turn.prompt["prompt_token_ids"]), "images": n_images,
                "restart_seconds": time.monotonic() - started, "engine_request_id": session.engine_request_id}

    def _schedule_nll_probe(self, session, decision, plan, turn, result, gen_end, ids):
        """Capture everything the probe needs now (the session moves on), then run it in the background."""
        try:
            K = nll_probe_tokens(); T = list(ids[:K])
            snap = dict(token_ids=list(session.token_ids[:gen_end]), survivors=list(session.residency.survivor_intervals()),
                        gen_end=gen_end, images=[tuple(x) for x in session.images], parts=list(session.parts), T=T,
                        generated=len(ids), finish=result.get("finish_reason"), own_rows=list(result.get("logprobs") or [])[:len(T)],
                        sampling={k: turn.sampling[k] for k in ("temperature", "top_p", "top_k", "min_p", "repetition_penalty",
                                                                 "presence_penalty", "frequency_penalty") if k in turn.sampling},
                        eviction_revision=session.eviction_revision, evicted_tokens=plan.removed_tokens,
                        policy=(decision.receipt or {}).get("effective_policy"), requested_policy=(decision.receipt or {}).get("policy"),
                        resident_after=plan.resident_after, operation_id=session.current_operation)
            if session.probe_semaphore is None:
                session.probe_semaphore = getattr(self, "_probe_semaphore", None) or asyncio.Semaphore(2); self._probe_semaphore = session.probe_semaphore
            task = asyncio.create_task(self._nll_probe(session, snap))
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
        except Exception as exc:  # noqa: BLE001
            self.audit("evict_nll_probe_failed", session, operation_id=session.current_operation, stage="schedule", error=f"{type(exc).__name__}: {exc}")

    async def _nll_probe(self, session, snap):
        probe = getattr(self.backend, "probe", None)
        if probe is None:
            self.audit("evict_nll_probe_failed", session, operation_id=snap["operation_id"], stage="backend", error="backend has no probe()"); return
        started = time.monotonic(); tag = f"{session.session_id}-r{snap['eviction_revision']}"
        try:
            async with session.probe_semaphore:
                T = snap["T"]
                if not T:
                    raise RuntimeError("no generated tokens to score")
                prompt_a, resident, n_images = build_probe_prompt(snap["token_ids"], snap["survivors"], snap["gen_end"], snap["images"], snap["parts"], T)
                t0 = time.monotonic()
                res_a = await probe(prompt_a, {"max_tokens": 1, "temperature": 0.0, "top_p": 1.0, "top_k": -1, "min_p": 0.0, "prompt_logprobs": 0}, tag=tag + "-a")
                t_a = time.monotonic() - t0
                nll_rc_ev = _mean_nll_prompt(res_a.get("prompt_logprobs") or [], T, resident)
                prompt_b, _, _ = build_probe_prompt(snap["token_ids"], snap["survivors"], snap["gen_end"], snap["images"], snap["parts"], ())
                t0 = time.monotonic()
                res_b = await probe(prompt_b, {**snap["sampling"], "max_tokens": len(T), "seed": 8200, "logprobs": 0}, tag=tag + "-b")
                t_b = time.monotonic() - t0
                rc_ids = list(res_b.get("token_ids") or [])
                nll_rc_rc = _mean_nll_sampled(res_b.get("logprobs") or [], rc_ids)
                nll_ev_ev = _mean_nll_sampled(snap["own_rows"], T)
                self.audit("evict_nll_probe", session, operation_id=snap["operation_id"], eviction_revision=snap["eviction_revision"],
                           policy=snap["policy"], requested_policy=snap["requested_policy"], evicted_tokens=snap["evicted_tokens"],
                           resident_tokens=resident, resident_after=snap["resident_after"], images=n_images, scored_tokens=len(T),
                           generated_tokens=snap["generated"], generated_finish=snap["finish"], sampled_tokens=len(rc_ids),
                           sampled_finish=res_b.get("finish_reason"), nll_ev_ev=nll_ev_ev, nll_rc_ev=nll_rc_ev, nll_rc_rc=nll_rc_rc,
                           excess_state=(nll_rc_ev - nll_ev_ev) if (nll_rc_ev is not None and nll_ev_ev is not None) else None,
                           excess_sample=(nll_rc_ev - nll_rc_rc) if (nll_rc_ev is not None and nll_rc_rc is not None) else None,
                           probe_a_seconds=t_a, probe_b_seconds=t_b, total_seconds=time.monotonic() - started)
        except Exception as exc:  # noqa: BLE001
            self.audit("evict_nll_probe_failed", session, operation_id=snap["operation_id"], stage="probe", error=f"{type(exc).__name__}: {exc}"[:400])

    def audit(self, event, session, **fields):
        if self.audit_path:
            row = {"time": time.time(), "event": event,
                   "session_id": session.session_id, "revision": session.revision,
                   "engine_request_id": session.engine_request_id, **fields}
            try:
                line = json.dumps(row, ensure_ascii=False) + "\n"   # encode now: fields may be mutated later
            except Exception as exc:  # noqa: BLE001
                line = json.dumps({"time": row["time"], "event": event, "session_id": session.session_id, "audit_error": f"{type(exc).__name__}: {exc}"}) + "\n"
            self._audit_pool().submit(self._audit_write, line)

    def _audit_pool(self):
        """Single background writer (2026-09-23): the network-volume append leaves the event loop; order kept."""
        pool = getattr(self, "_audit_executor", None)
        if pool is None:
            from concurrent.futures import ThreadPoolExecutor
            pool = self._audit_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="arc3-service-audit")
        return pool

    def _audit_write(self, line):
        try:
            with self.audit_path.open("a") as f:
                f.write(line)
        except Exception as exc:  # noqa: BLE001
            print(f"[arc3 service audit] write failed: {type(exc).__name__}: {exc}", flush=True)

    @staticmethod
    def valid_id(value):
        if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,160}", value):
            raise SessionError("Invalid session/request ID", 400)
        return value

    def get(self, session_id):
        try:
            return self.sessions[self.valid_id(session_id)]
        except KeyError:
            raise SessionError("Unknown session", 404) from None

    async def start(self, payload):
        sid = self.valid_id(payload.get("session_id"))
        request_id = self.valid_id(payload.get("request_id"))
        try:
            policy = HistoryPolicy.from_payload(payload.get("history_eviction"), self.max_context)
        except HistoryValidationError as exc:
            raise SessionError(str(exc), 400) from exc
        async with self.lock:
            session = self.sessions.get(sid)
            if session is None:
                active = sum(s.state not in ("closed", "error") for s in self.sessions.values())
                if active >= self.max_sessions:
                    raise SessionError("Resident session admission limit reached", 429)
                session = Session(sid, "arc3-persistent-" + sid, dict(payload))
                session.history_policy = policy
                self.sessions[sid] = session
            elif request_id not in session.operations:
                raise SessionError("Session already exists; use append or a new ID")
        return await self._operation(session, request_id, payload, initial=True)

    async def append(self, sid, payload):
        if "history_eviction" in payload:
            raise SessionError("History eviction policy is immutable and is configured only at start", 400)
        return await self._operation(self.get(sid), self.valid_id(payload.get("request_id")),
                                     payload, initial=False)

    async def _operation(self, session, request_id, payload, *, initial):
        body_digest = digest(payload)
        async with session.lock:
            previous = session.operations.get(request_id)
            if previous:
                if previous["digest"] != body_digest:
                    raise SessionError("Request ID reused with different input")
                task = previous["task"]
            else:
                if session.state in ("closed", "error"):
                    raise SessionError("Session is " + session.state)
                if session.current_operation:
                    raise SessionError("Another turn is in flight")
                if not initial and payload.get("expected_revision") != session.revision:
                    raise SessionError("Stale session revision")
                if payload.get("expected_prefix_sha256") not in (None, session_token_digest(session)):
                    raise SessionError("Token prefix digest mismatch")
                if initial != (session.revision == 0):
                    raise SessionError("Invalid start/append lifecycle")
                session.current_operation = request_id
                task = asyncio.create_task(self._run_turn(session, payload, initial=initial))
                # Consume failures even if the HTTP caller disappears; retries
                # retrieve this same task, never duplicate the engine submission.
                task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
                session.operations[request_id] = {"digest": body_digest, "task": task}
        return await asyncio.shield(task)

    async def _run_turn(self, session, payload, *, initial):
        submitted = False
        start_time = time.monotonic()
        try:
            continuation = bool(payload.get("continuation", False))
            if continuation and (initial or session.finish_reason != "length"):
                raise SessionError("Empty continuation requires an existing length pause", 400)
            turn = await self.backend.prepare(session, payload, initial=initial)
            new_ids = list(turn.token_ids)
            if not new_ids and not continuation:
                raise SessionError("Empty append must explicitly request continuation", 400)
            if new_ids and continuation and not turn.injected_ids:
                raise SessionError("Continuation cannot also append input", 400)
            if turn.injected_ids and (not continuation or list(turn.injected_ids) != new_ids
                                      or not session.segments or session.segments[-1][0] != "assistant"):
                raise SessionError("Assistant injection requires a paused assistant turn", 400)
            if session.residency.logical_tokens != len(session.token_ids):
                raise RuntimeError("Logical token ledger and residency accounting diverged")
            session.rebase_info = None
            rebase_due = self._rebase_due(session, new_ids, initial=initial, continuation=continuation)
            try:
                transition = session.suffix_registry.plan(
                    turn.suffix_records, turn.evict_observation_ids,
                    logical_offset=len(session.token_ids), delta_tokens=len(new_ids))
                ranges = session.suffix_registry.eviction_ranges(transition)
                tail_range = None
                if turn.tail_eviction is not None:
                    # Degenerate-run removal: [assistant start + offset, frontier guard) inside the paused assistant turn.
                    seg = session.segments[-1] if session.segments else None
                    a = (seg[1] + turn.tail_eviction["assistant_offset"]) if seg else None
                    b = len(session.token_ids) - 4
                    if (not continuation or seg is None or seg[0] != "assistant" or seg[2] != len(session.token_ids)
                            or a is None or a < seg[1] or a >= b):
                        raise SessionError("tail_eviction range must lie inside the paused assistant turn before the frontier guard", 400)
                    tail_range = (a, b); ranges = list(ranges) + [tail_range]
                starts = turn.observation_starts
                if session.history_policy.policy != "off":
                    if (not isinstance(starts, list) or any(not isinstance(r, dict) for r in starts)
                            or [r.get("observation_id") for r in starts] !=
                            [r.get("observation_id") for r in turn.suffix_records]
                            or (initial and len(starts) != 1)):
                        raise HistoryValidationError("History policy requires matching proven observation starts")
                history_transition = session.history_registry.plan_registration(
                    starts, logical_offset=len(session.token_ids), delta_tokens=len(new_ids))
                # Plan on committed blocks only. The incoming observation closes
                # its predecessor only after this operation commits, so pending
                # tool results cannot be deleted before they have been computed.
                frequency = None; prepare_profile = {}
                hpolicy = session.history_policy
                if (hpolicy.policy == "qsa_frequency"
                        and session.residency.resident_tokens + len(new_ids) >= hpolicy.resident_token_budget):
                    # Only a possible trigger pays for the counter transfer; the planner recomputes exactly.
                    snapshot_fn = getattr(self.backend, "frequency_snapshot", None)
                    if snapshot_fn is not None:
                        from qsa_frequency import QSAFrequencyError, group_frequencies
                        try:
                            _t = time.perf_counter()
                            snapshot = await snapshot_fn(session)
                            prepare_profile["snapshot_seconds"] = time.perf_counter() - _t; _t = time.perf_counter()
                            frequency = group_frequencies(snapshot, epoch_start_logical=session.frequency_epoch_start)
                            prepare_profile["frequency_seconds"] = time.perf_counter() - _t
                        except QSAFrequencyError as exc:
                            self.audit("frequency_snapshot_failed", session, operation_id=session.current_operation,
                                       error=str(exc))
                _t = time.perf_counter()
                decision = session.history_registry.plan_eviction(
                    session.history_policy, session.residency,
                    pending_suffix_ranges=ranges, incoming_tokens=len(new_ids), frequency=frequency,
                    segments=session.segments, structural=session.structural, images=session.images,
                    incoming_observation=bool(starts),   # 2026-09-26: a new observation releases the pinned one
                    image_kinds=[session.image_kinds.get(a, "other") for a, _ in session.images])
                prepare_profile["plan_seconds"] = time.perf_counter() - _t
                session.prepare_profile = prepare_profile
                session.frequency_epoch_reset_pending = (hpolicy.policy == "qsa_frequency"
                                                          and bool(decision.history_ranges))
                plan = decision.eviction_plan
            except (ResidencyError, HistoryValidationError) as exc:
                raise SessionError(str(exc), 400) from exc
            except HistoryCapacityError as exc:
                raise SessionError("Context capacity: " + str(exc), 413, receipt=exc.receipt) from exc
            logical_input_count = len(session.token_ids) + len(new_ids)
            input_count = plan.resident_after + len(new_ids)
            capacity = session.history_policy.resident_token_budget - input_count
            if capacity < 1:
                raise SessionError("Context capacity reached after planned eviction", 413)
            logical_capacity = self.max_logical_tokens - (input_count if rebase_due else logical_input_count)
            if logical_capacity < 1:
                raise SessionError("Original logical position limit reached; eviction does not reset RoPE positions", 413)
            capacity = min(capacity, logical_capacity)
            # Normally consume all remaining context, with no artificial response
            # cap. Explicit max_tokens is only a caller-requested diagnostic cap.
            requested = turn.sampling.get("max_tokens")
            if requested is not None and (type(requested) is not int or requested < 1):
                raise SessionError("max_tokens must be positive or null", 400)
            turn.sampling["max_tokens"] = min(requested, capacity) if requested else capacity
            # Finish all renderer/input/sampler preflight before the first cache
            # mutation. Once mutation begins, any uncertainty aborts this session.
            preflight = getattr(self.backend, "preflight", None)
            if preflight is not None:
                await preflight(session, turn, initial=initial)
            if session.state == "closed":
                raise SessionError("Session closed during input preparation")
            if plan.removed_tokens:
                submitted = True
                self.audit("evict_submit", session, operation_id=session.current_operation,
                           eviction_revision=session.eviction_revision,
                           prefix_sha256=session_token_digest(session),
                           retired_observation_ids=list(transition.retire_ids), plan=plan.wire(),
                           history_eviction=decision.receipt,
                           history_state=history_state(session))
                if session.history_policy.policy == "oldest_recompute":
                    ack = await asyncio.wait_for(self._recompute_restart(session, plan, turn), timeout=self.turn_timeout)
                elif rebase_due:   # the rebase below discards the engine request; acknowledge the plan locally
                    ack = {"plan_digest": plan.digest, "resident_tokens": plan.resident_after, "evicted_tokens": plan.removed_tokens,
                           "logical_tokens": len(session.token_ids), "eviction_revision": session.eviction_revision + 1,
                           "execution": "rebase"}
                else:
                    ack = await asyncio.wait_for(self.backend.evict(session, plan), timeout=self.turn_timeout)
                if (not isinstance(ack, dict) or ack.get("plan_digest") != plan.digest
                        or any(type(ack.get(k)) is not int for k in
                               ("resident_tokens", "evicted_tokens", "logical_tokens", "eviction_revision"))
                        or ack.get("resident_tokens") != plan.resident_after
                        or ack.get("evicted_tokens") != plan.removed_tokens
                        or ack.get("logical_tokens") != len(session.token_ids)
                        or ack.get("eviction_revision") != session.eviction_revision+1):
                    raise RuntimeError("Engine eviction acknowledgement differs from the prepared plan")
                session.residency.commit(plan, acknowledged_digest=ack["plan_digest"])
                session.eviction_revision += 1
                if session.frequency_epoch_reset_pending:
                    session.frequency_epoch_start = len(session.token_ids)
                    session.frequency_epoch_reset_pending = False
                self.audit("evict_commit", session, operation_id=session.current_operation,
                           acknowledgement=ack, plan=plan.wire(), history_eviction=decision.receipt,
                           endpoint_profile={**(session.eviction_endpoint_profile or {}), **getattr(session, "prepare_profile", {})})
            # Registry/ledger updates below commit the prepared input together.
            # Any unexpected failure after this point poisons the owned session.
            submitted = True
            session.suffix_registry.commit(transition)
            session.history_registry.commit(history_transition)
            session.state = "running"
            session.parse_context = turn.parse_context
            self.audit("submit", session, operation_id=session.current_operation,
                       initial=initial, old_tokens=len(session.token_ids),
                       appended_token_ids=new_ids, input_tokens=input_count,
                       logical_input_tokens=logical_input_count,
                       resident_before_append=session.residency.resident_tokens,
                       suffix_records=turn.suffix_records,
                       observation_starts=turn.observation_starts,
                       history_state=history_state(session),
                       history_eviction=decision.receipt,
                       managed_suffixes=transition.acknowledgement(),
                       eviction_revision=session.eviction_revision,
                       prefix_sha256=session_token_digest(session),
                       max_tokens=turn.sampling["max_tokens"])
            _old = len(session.token_ids)
            if new_ids and turn.injected_ids:   # assistant-turn budget: the injected text extends the paused assistant turn
                if session.segments[-1][2] == _old: session.segments[-1][2] = _old + len(new_ids)
                else: session.segments.append(["assistant", _old, _old + len(new_ids)])
                turn.parse_context.assistant_prefix_ids.extend(new_ids)
            elif new_ids:   # stage 12: segment the delta (tool output before the observation start, then the observation)
                _obs = None
                if isinstance(starts, list) and starts and isinstance(starts[0], dict):
                    _obs = starts[0].get("local_start_token")
                if initial:
                    session.segments.append(["observation", _old, _old + len(new_ids)])
                elif type(_obs) is int and 0 <= _obs < len(new_ids):
                    if _obs > 0: session.segments.append(["tool", _old, _old + _obs])
                    session.segments.append(["observation", _old + _obs, _old + len(new_ids)])
                else:
                    session.segments.append(["tool", _old, _old + len(new_ids)])
            session.token_ids.extend(new_ids)
            if not continuation:   # review fix 2026-09-25: an unclosed <tool_call> dies with its assistant turn
                session.open_tool_call = None
            self._mark_structural(session, new_ids, _old); self._mark_tool_calls(session, new_ids, _old); self._mark_images(session, new_ids, _old)
            if new_ids and (nll_probe_rate() or session.history_policy.policy == "oldest_recompute" or self._rebase_enabled()) and turn.prompt.get("type") == "multimodal":
                self._record_part(session, _old, len(new_ids), getattr(turn, "delta_prompt", turn.prompt))
            probe_now = bool(nll_probe_rate() and plan.removed_tokens and decision.history_ranges
                             and session.eviction_revision % nll_probe_rate() == 0)
            # (sampled-token logprobs are requested uniformly for every turn while probing: vLLM's per-request
            # logprobs processor is fixed at session start and asserts if a later turn changes it)
            session.residency.append(len(new_ids))
            submitted = True
            if rebase_due:
                local = int(decision.receipt.get("history_removed_tokens") or 0) if plan.removed_tokens else 0
                await asyncio.wait_for(self._rebase(session, turn, local), timeout=self.turn_timeout)
                # the rebuilt frame is smaller: re-admit output against it (never above the caller's max_tokens)
                logical_input_count = input_count = len(session.token_ids)
                capacity = min(session.history_policy.resident_token_budget - input_count, self.max_logical_tokens - input_count)
                turn.sampling["max_tokens"] = min(requested, capacity) if requested else capacity
            result = await asyncio.wait_for(self.backend.turn(session, turn, initial=initial or getattr(turn, "recompute", False)),
                                            timeout=self.turn_timeout)
            ids = list(result["token_ids"])
            if not ids or len(ids) > turn.sampling["max_tokens"]:
                raise RuntimeError("Invalid emitted token count at pause")
            if result["finish_reason"] not in ("stop", "length"):
                raise RuntimeError("Unsupported engine finish reason: " + str(result["finish_reason"]))
            _gen = len(session.token_ids)
            if continuation and session.segments and session.segments[-1][0] == "assistant" and session.segments[-1][2] == _gen:
                session.segments[-1][2] = _gen + len(ids)   # a length continuation extends the same assistant turn
            else:
                session.segments.append(["assistant", _gen, _gen + len(ids)])
            session.token_ids.extend(ids)
            self._mark_structural(session, ids, _gen); self._mark_tool_calls(session, ids, _gen); self._mark_images(session, ids, _gen)
            session.residency.append(len(ids))
            if probe_now:
                self._schedule_nll_probe(session, decision, plan, turn, result, _gen, ids)
            session.finish_reason = result["finish_reason"]
            session.revision += 1
            session.state = "paused"
            response = await self.backend.response(session, result)
            response.update(usage={"prompt_tokens": input_count, "completion_tokens": len(ids),
                                   "total_tokens": input_count + len(ids)})
            response["persistent"] = {
                "session_id": session.session_id, "revision": session.revision,
                "resident_tokens": session.residency.resident_tokens,
                "logical_tokens": len(session.token_ids),
                "max_logical_tokens": self.max_logical_tokens,
                "logical_input_tokens": logical_input_count,
                "appended_prompt_tokens": len(new_ids),
                "token_ids": ids, "prefix_sha256": session_token_digest(session),
                "engine_finish_reason": session.finish_reason, "prefix_caching": False,
                "eviction": True, "eviction_revision": session.eviction_revision,
                "evicted_tokens": plan.removed_tokens,
                "managed_suffixes": transition.acknowledgement(),
                "history_eviction": decision.receipt,
                "history_state": history_state(session),
                "segments": len(session.segments),
                "structural": len(session.structural),
                "images": len(session.images),
                "injected_token_ids": list(new_ids) if turn.injected_ids else [],
                "assistant_injection": turn.injection,
                "tail_eviction": (dict(turn.tail_eviction, range=list(tail_range), tokens=tail_range[1] - tail_range[0])
                                  if tail_range is not None else None),
                "elapsed_seconds": time.monotonic() - start_time,
                "rebase": session.rebase_info,
            }
            self.audit("pause", session, operation_id=session.current_operation,
                       output_token_ids=ids, resident_tokens=session.residency.resident_tokens,
                       logical_tokens=len(session.token_ids),
                       finish_reason=session.finish_reason,
                       prefix_sha256=session_token_digest(session),
                       elapsed_seconds=time.monotonic() - start_time)
            return response
        except BaseException as exc:
            # Once submitted, an uncertain engine outcome must never be retried
            # as a new append. Abort the owned session and make failure explicit.
            if submitted:
                session.state = "error"
                session.error = f"{type(exc).__name__}: {exc}"
                await self.backend.close(session)
            elif initial:
                session.state = "error"
                session.error = f"{type(exc).__name__}: {exc}"
            self.audit("rejected" if not submitted else "failed", session,
                       operation_id=session.current_operation,
                       error=f"{type(exc).__name__}: {exc}",
                       history_eviction=getattr(exc, "receipt", None))
            raise
        finally:
            session.current_operation = None

    async def close(self, sid):
        session = self.get(sid)
        async with session.lock:
            if session.close_task is not None:
                task = session.close_task
            elif session.state == "closed":
                return {"session_id": sid, "state": "closed"}
            else:
                session.state = "closed"
                task = asyncio.create_task(self._close_session(session))
                session.close_task = task
                task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
        return await asyncio.shield(task)

    async def _close_session(self, session):
        pending = [op["task"] for op in session.operations.values() if not op["task"].done()]
        for task in pending:
            task.cancel()
        if pending:
            # An eviction drains its shielded worker transaction before the
            # cancelled operation returns. Do not free its cache while it copies.
            await asyncio.gather(*pending, return_exceptions=True)
        await self.backend.close(session)
        session.state = "closed"
        self.audit("close", session, resident_tokens=session.residency.resident_tokens,
                   logical_tokens=len(session.token_ids))
        return {"session_id": session.session_id, "state": "closed"}

    def status(self, sid):
        session = self.get(sid)
        operations = {}
        for name, op in session.operations.items():
            task = op["task"]
            if not task.done():
                operations[name] = {"state": "running"}
            elif task.cancelled():
                operations[name] = {"state": "cancelled"}
            elif task.exception():
                operations[name] = {"state": "error", "error": str(task.exception())}
            else:
                operations[name] = {"state": "complete", "result": task.result()}
        return {"session_id": sid, "state": session.state, "revision": session.revision,
                "resident_tokens": session.residency.resident_tokens,
                "logical_tokens": len(session.token_ids),
                "eviction_revision": session.eviction_revision,
                "history_policy": {"policy": session.history_policy.policy,
                                   "fraction": session.history_policy.fraction,
                                   "resident_token_budget": session.history_policy.resident_token_budget},
                "history_state": history_state(session),
                "prefix_sha256": session_token_digest(session),
                "error": session.error, "operations": operations}

    async def close_all(self):
        await asyncio.gather(*(self.close(sid) for sid in self.sessions), return_exceptions=True)
