"""One ARC-2 attempt: a persistent session on the server, a python kernel, and the turn loop (v0, minimal).

Loop: send the new external messages (system + opener first, then tool results), let the server generate (a length
pause is continued in place; the transport injects its per-reply cues and closes), run every python call in the
kernel, return each call's output as a tool message. The attempt ends when the model replies without a tool call after
an answer is recorded, when it does so twice without an answer (one harness note in between), when the generated-token
budget is spent, or on a session error. Without a recorded answer, the best-checked function still defined is applied
to the test inputs as a harness fallback, reported separately.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field

from .prompts import opener_message, system_message
from .shared import spent_text, wrap_text

MARKUP = ("<tool_call>", "<function=", "</tool_call>")


@dataclass
class AttemptResult:
    task_id: str
    end: str = ""                      # finished | no_answer | budget | deadline | capacity | protocol_error | error
    generated_tokens: int = 0
    requests: int = 0
    replies: int = 0
    tool_calls: int = 0
    kernel_restarts: int = 0
    kernel_timeouts: int = 0
    cues: dict = field(default_factory=dict)
    closes: dict = field(default_factory=dict)
    nudges: dict = field(default_factory=dict)
    checks: list = field(default_factory=list)          # {tokens, call, fn, passed, total}
    first_full_pass_tokens: int | None = None
    submits: list = field(default_factory=list)         # {tokens, call, attempt, fn, train_passed, train_total}
    submissions: dict = field(default_factory=dict)     # {attempt: outputs}
    fallback: dict | None = None
    notice_at_tokens: int | None = None
    max_prompt_tokens: int = 0
    evicted_tokens: int = 0
    seconds: float = 0.0
    error: str | None = None
    earlier: list = field(default_factory=list)       # v2j: earlier valid answers that later submits replaced

    def to_json(self):
        return asdict(self)


def _user(text, observation=False):
    msg = {"role": "user", "content": text}
    if observation:                       # a turn boundary for the server's history registry (one per operation)
        msg["_arc3_observation"] = True
    return msg


def _tool(call_id, text):
    return {"role": "tool", "tool_call_id": call_id, "content": text}


def _count(d, key):
    d[key] = d.get(key, 0) + 1


def correction_text(kernel, texts):
    """The correction for an unverified attempt-1 answer (harness facts), or None when it is verified or absent."""
    subs = [r for r in kernel.records if r["kind"] == "submit" and r["attempt"] == 1]
    if not subs or subs[-1].get("verified_by"):
        return None
    last = subs[-1]
    if last["fn"] == "grids":
        return texts["correction_grids"]
    checks = [r for r in kernel.records if r["kind"] == "check" and r["fn"] == last["fn"]]
    failing = ", ".join(f"pair {i}" for i, k in enumerate(checks[-1]["results"]) if k != "pass") if checks else "not checked"
    return texts["correction_function"].format(fn=last["fn"], passed=last.get("train_passed"), total=last.get("train_total"),
                                               failing=failing or "none")


def valid_saved(kernel):
    """The user's rule (2026-10-29): a saved final answer is valid when it is code passing every demonstration
    (verified_by) or a grid answer."""
    for a in (1, 2):
        subs = [r for r in kernel.records if r["kind"] == "submit" and r["attempt"] == a]
        if subs and (subs[-1]["fn"] == "grids" or subs[-1].get("verified_by")):
            return True
    return False


def earlier_answers(kernel):
    """v2j (B, non-destructive submit): valid answers (verified code or grids) that a later submit replaced, most recent
    first, distinct from each other and from the final answers. Not described to the model; slots.pass_answers uses the
    first to fill an empty attempt-2 slot."""
    subs = [r for r in kernel.records if r["kind"] == "submit"]
    final = {}
    for r in subs:
        final[r["attempt"]] = r
    finals = [r["outputs"] for r in final.values()]
    out, seen = [], []
    for r in reversed(subs):
        if any(r is f for f in final.values()):
            continue
        if not (r.get("verified_by") or r["fn"] == "grids"):
            continue
        if r["outputs"] in finals or r["outputs"] in seen:
            continue
        seen.append(r["outputs"])
        out.append(dict(attempt=r["attempt"], fn=r["fn"], outputs=r["outputs"], verified=bool(r.get("verified_by"))))
    return out


def keep_going_text(kernel, texts, used, budget, wrap="", key="keep_going", spent=""):
    detail = ""
    subs = [r for r in kernel.records if r["kind"] == "submit" and r["attempt"] == 1]
    if subs and subs[-1]["fn"] != "grids" and not subs[-1].get("verified_by"):
        last = subs[-1]
        checks = [r for r in kernel.records if r["kind"] == "check" and r["fn"] == last["fn"]]
        failing = (", ".join(f"pair {i}" for i, k in enumerate(checks[-1]["results"]) if k != "pass") if checks
                   else "not checked")
        detail = texts["keep_going_detail_function"].format(fn=last["fn"], passed=last.get("train_passed"),
                                                            total=last.get("train_total"), failing=failing or "none")
    return texts[key].format(detail=detail, used=used, budget=budget, wrap=wrap, spent=spent)


def run_attempt(task, session, kernel, texts, *, budget, notice, log=None, fmt="letters", pin=True, deadline=None,
                observe_turns=False, messages=None, answered=None, nudge_key="nudge_no_submission",
                notice_key="budget_notice", fallback=True, corrections=0, no_call="nudge_once", idle_limit=6,
                repeat_limit=0, cap_state=None, prompt_key="system", reply_cap=None, recovery_budget_key="recovery_budget",
                image_cell=None, idle_skip_status=False, marker_key="turn_marker", wrap_at=None, limit_at=None,
                text_keys=None, recall_at=None, progress=None, note_guard=False, invalid_recover=0, invalid_limit=0):
    """One attempt (a solver, or with messages / answered / nudge_key / notice_key another session kind such as the
    judge). corrections: how many times an unverified attempt-1 answer is answered with correction_text() before the
    attempt may finish with it.
    v2 (the user, 2026-10-29): no_call='keep_going' answers every reply without a tool call, until a valid solution is
    saved (valid_saved), with keep_going_text (the budget spent, and the failing pairs of a saved function); more than
    idle_limit such replies in a row end the attempt ('stalled'). repeat_limit: that many repetition closes end the
    attempt ('degenerate'). cap_state: a threading.local the in-reply status cue reads (used, budget).
    idle_skip_status (v2b, 2026-10-29): a reply without a tool call that carried an in-reply status note does not count
    toward idle_limit (run03a: 84% of status notes ended their reply within one token).
    v2q (the user, 2026-11-01: "Can we add a recovery (and maybe stop and park in the queue if it does not recover)?"):
    replies whose tool calls all fail (an unknown tool or unreadable arguments: nothing runs) count in a row; at
    invalid_recover of them texts['invalid_call_recovery'] follows the tool results; at invalid_limit the attempt ends
    ('stalled'): its saved answers are kept and the task takes its next pass from the queue (g7a: one pass sent 746 empty
    python calls after its 12th eviction, until the token budget; healthy runs have at most 3 in a row). 0 = off."""
    log = log or (lambda kind, payload: None)
    res = AttemptResult(task.task_id)
    t0 = time.monotonic()
    external = list(messages) if messages is not None else [
        system_message(texts, budget=budget, notice=notice, fmt=fmt, key=prompt_key, reply_cap=reply_cap,
                       image_cell=image_cell),
        opener_message(task, texts, pin=pin, fmt=fmt, image_cell=image_cell)]
    answered = answered or (lambda: bool(kernel.submissions()))
    noticed = False
    nudged = False
    corrected = 0
    idle = 0
    invalid = 0                                  # v2q: replies in a row whose calls all failed
    turns = 0

    def generate(ext):
        if cap_state is not None:
            cap_state.used, cap_state.budget = res.generated_tokens, budget
        return session.generate(ext)

    def account(c):
        res.requests += 1
        res.generated_tokens += len(c.token_ids)
        if progress is not None:                # length-aware admission (runner.Forecast): tokens and resident size
            try:
                progress(res.generated_tokens, int((getattr(c, "persistent", None) or {}).get("resident_tokens") or 0))
            except Exception:                   # noqa: BLE001 - bookkeeping never ends an attempt
                pass
        res.max_prompt_tokens = max(res.max_prompt_tokens, int((c.usage or {}).get("prompt_tokens") or 0))
        for kind in getattr(c, "cues", ()) or ():
            _count(res.cues, kind)

    try:
        late = (lambda: deadline is not None and time.monotonic() >= deadline)   # the run's wall-clock deadline
        recalled = []

        def wrap_now():
            """v2c: the wrap-up reminder in every harness message from wrap_at; v2e: once, from recall_at, the history()
            reminder (the user, 2026-10-29)."""
            w = wrap_text(texts, res.generated_tokens, budget, wrap_at)
            if recall_at is not None and not recalled and res.generated_tokens >= recall_at:
                recalled.append(res.generated_tokens)
                w += " " + texts["history_reminder"]
            return w
        spent_now = (lambda: spent_text(texts, res.generated_tokens, budget, limit_at))   # v2d: every user turn
        keys = dict(dict(keep_going="keep_going", recovery_repeat="recovery_repeat", nudge_format="nudge_format",
                         unknown_tool="unknown_tool"), **(text_keys or {}))
        while True:
            if res.generated_tokens >= budget:
                res.end = "budget"
                break
            if late():
                res.end = "deadline"
                break
            completion = generate(external)
            external = []
            account(completion)
            while completion.pending and res.generated_tokens < budget and not late():
                completion = generate([])
                account(completion)
            if completion.pending:                       # the budget or the run's deadline ran out inside a reply
                res.end = "budget" if res.generated_tokens >= budget else "deadline"
                break
            res.replies += 1
            msg = completion.message or {}
            calls = msg.get("tool_calls") or []
            if completion.closed_by:
                _count(res.closes, completion.closed_by)
                if repeat_limit and res.closes.get("repeat_close", 0) >= repeat_limit:
                    res.end = "degenerate"
                    break
                key = keys["recovery_repeat"] if completion.closed_by == "repeat_close" else recovery_budget_key
                external = [_user(texts[key].format(reply_cap=reply_cap or 0, wrap=wrap_now(), spent=spent_now()),
                                  observe_turns)]
                continue
            if not calls:
                content = str(msg.get("content") or "")
                if any(m in content for m in MARKUP):
                    _count(res.nudges, "format")
                    external = [_user(texts[keys["nudge_format"]].format(detail="the call markup was incomplete",
                                                                          wrap=wrap_now(), spent=spent_now()),
                                      observe_turns)]
                    continue
                if no_call == "keep_going":
                    if valid_saved(kernel):
                        cues = getattr(completion, "cues", ()) or ()
                        if note_guard and "soft_nudge" in cues and len(completion.token_ids) <= 64:
                            # v2j (A): the reply ended right after an injected note; that is not a finish
                            _count(res.nudges, "note_continue")
                            external = [_user(texts["note_ended_continue"].format(spent=spent_now(), wrap=wrap_now()),
                                              observe_turns)]
                            continue
                        res.end = "finished"
                        break
                    if not (idle_skip_status and "soft_nudge" in (getattr(completion, "cues", ()) or ())):
                        idle += 1
                    if idle > idle_limit:
                        res.end = "stalled"
                        break
                    _count(res.nudges, "keep_going")
                    external = [_user(keep_going_text(kernel, texts, res.generated_tokens, budget, wrap_now(),
                                                      keys["keep_going"], spent_now()), observe_turns)]
                    continue
                if answered():
                    fix = correction_text(kernel, texts) if corrected < corrections else None
                    if fix is not None:
                        corrected += 1
                        _count(res.nudges, "correction")
                        external = [_user(fix, observe_turns)]
                        continue
                    res.end = "finished"
                    break
                if not nudged:
                    nudged = True
                    _count(res.nudges, "no_submission")
                    external = [_user(texts[nudge_key], observe_turns)]
                    continue
                res.end = "no_answer"
                break
            idle = 0
            tool_msgs = []
            ran = 0
            for call in calls:
                fn = call.get("function") or {}
                name = str(fn.get("name", "")).strip()
                cid = call.get("id", "")
                if name != "python":
                    tool_msgs.append(_tool(cid, texts[keys["unknown_tool"]].format(name=name)))
                    continue
                raw = fn.get("arguments", "{}")
                try:
                    args = json.loads(raw) if isinstance(raw, str) else dict(raw)
                    code = args["code"]
                    if not isinstance(code, str):
                        raise TypeError("code must be a string")
                except Exception as exc:                 # noqa: BLE001 - reported to the model as a fact
                    tool_msgs.append(_tool(cid, texts["bad_arguments"].format(detail=f"{type(exc).__name__}: {exc}")))
                    continue
                r = kernel.run(code)
                ran += 1
                res.tool_calls += 1
                res.kernel_timeouts += int(r.timed_out)
                for ev in r.events:
                    if ev["kind"] == "check":
                        res.checks.append(dict(tokens=res.generated_tokens, call=ev.get("call"), fn=ev["fn"],
                                               passed=ev["passed"], total=ev["total"]))
                        if ev["passed"] == ev["total"] and res.first_full_pass_tokens is None:
                            res.first_full_pass_tokens = res.generated_tokens
                    elif ev["kind"] == "submit":
                        res.submits.append(dict(tokens=res.generated_tokens, call=ev.get("call"), attempt=ev["attempt"],
                                                fn=ev["fn"], train_passed=ev.get("train_passed"),
                                                train_total=ev.get("train_total"), verified_by=ev.get("verified_by")))
                log("tool", dict(call=kernel.calls, tokens=res.generated_tokens, seconds=round(r.seconds, 3),
                                 error=r.error, timed_out=r.timed_out, restarted=r.restarted, code=code,
                                 text=r.text, events=r.events))
                tool_msgs.append(_tool(cid, r.text))
            recovery = None
            if ran:
                invalid = 0
            else:                                        # v2q: every call of this reply failed, nothing ran
                invalid += 1
                if invalid_limit and invalid >= invalid_limit:
                    _count(res.nudges, "invalid_stop")
                    res.end = "stalled"
                    break
                if invalid_recover and invalid == invalid_recover:
                    _count(res.nudges, "invalid_recovery")
                    recovery = texts["invalid_call_recovery"].format(n=invalid)
            note = None
            if not noticed and budget - res.generated_tokens <= notice:
                noticed = True
                res.notice_at_tokens = res.generated_tokens
                note = texts[notice_key].format(left=max(0, budget - res.generated_tokens))
            if observe_turns:                            # v1: each tool-result append closes a turn (evictable history)
                turns += 1
                used = res.generated_tokens
                marker = texts[marker_key].format(turn=turns, used=used, budget=budget, wrap=wrap_now(),
                                                  spent=spent_now())
                tool_msgs.append(_user(marker + (" " + note if note else "") + (" " + recovery if recovery else ""),
                                       observation=True))
            elif note or recovery:                       # v2q: the recovery note in the same message as the budget notice
                tool_msgs.append(_user(" ".join(x for x in (note, recovery) if x)))
            external = tool_msgs
    except Exception as exc:                             # noqa: BLE001 - the attempt ends; the run continues
        cls = type(exc).__name__
        res.end = "capacity" if cls == "CapacityRefusal" else (
            "protocol_error" if cls in ("PersistentProtocolError", "AppendOnlyViolation") else "error")
        res.error = f"{cls}: {exc}"[:2000]
        log("attempt-error", dict(error=res.error))
    finally:
        res.kernel_restarts = kernel.restarts
        res.evicted_tokens = sum(b - a for a, b in (getattr(session, "evicted_ranges", None) or []))
        res.submissions = kernel.submissions()
        res.earlier = earlier_answers(kernel)
        if not res.submissions and fallback:
            res.fallback = kernel.fallback()
        res.seconds = round(time.monotonic() - t0, 3)
    return res
