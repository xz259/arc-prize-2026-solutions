"""The reviewer session (the user's design, 2026-10-30; JUDGE_EXPERIMENT_2026-10-29.md, revised): a fresh session sees the
puzzle as a solver does, then the answers two independent attempts saved (each with its rule explanation, its program when
it has one, and its test outputs), and saves its own attempts 1 and 2 with submit(...), reused or new. The kernel gets the
same answers as `previous` (prelude, payload key "previous").
"""
from __future__ import annotations

import json
import random

from .agent import run_attempt
from .kernel import Kernel
from .prompts import opener_message, system_message
from .tasks import grid_shape, grid_text


def last_submit_events(log_path):
    """{attempt: the last submit event} from a pass log (the rule explanation lives there)."""
    out = {}
    try:
        with open(log_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r.get("kind") != "tool":
                    continue
                for ev in (r.get("payload") or {}).get("events") or []:
                    if ev.get("kind") == "submit":
                        out[ev.get("attempt")] = ev
    except OSError:
        pass
    return out


def previous_from_rows(p1, p2, log_path_of, seed):
    """The answers of passes 1 and 2 as the reviewer experiment built them (analysis/review_offline.py build_cases): A and
    B in a seeded random order, each with its attempts 1 and 2 when saved, the rule explanation from the pass log's last
    submit event, its program and its check. Returns (previous, {label: pass number})."""
    order = [p1, p2]
    if random.Random(seed).random() < 0.5:
        order = [p2, p1]
    previous = []
    for label, r in zip("AB", order):
        ev = last_submit_events(log_path_of(r["pass_no"]))
        subs = {int(k): v for k, v in (r.get("submissions") or {}).items()}
        progs = {str(k): v for k, v in (r.get("programs") or {}).items()}
        for a in (1, 2):
            if not subs.get(a):
                continue
            recs = [x for x in r.get("submits") or [] if x.get("attempt") == a]
            last = recs[-1] if recs else {}
            prog = progs.get(str(a)) or {}
            frame = frame_kind(r.get("frame"))     # v2o: a pass on a re-presented copy: its program runs only in that frame
            previous.append(dict(label=label, attempt=a, outputs=subs[a], source="" if frame else (prog.get("source") or ""),
                                 explanation=(ev.get(a) or {}).get("explanation") or "",
                                 verified=bool(last.get("verified_by")), fn=last.get("fn"),
                                 passed=last.get("train_passed"), total=last.get("train_total"), pass_no=r["pass_no"],
                                 frame=frame))
    return previous, {"A": order[0]["pass_no"], "B": order[1]["pass_no"]}


def frame_kind(frame):
    """v2o: None for a pass on the puzzle itself; 'colour' or 'colour_symmetry' for a pass on a re-presented copy
    (runner --pass2-frame; the row's frame key)."""
    if not frame:
        return None
    return "colour" if frame.get("symmetry") in (None, "identity") else "colour_symmetry"


def answer_kind(texts, verified, fn, passed, total, frame=None):
    if verified:
        kind = texts["review_kind_verified"]
    elif fn == "grids" or fn is None:
        kind = texts["review_kind_grids"]
    else:
        kind = texts["review_kind_partial"].format(passed=passed, total=total)
    if frame:                                   # v2o (the reviewer session, 2026-11-01): the kind says why no program follows
        kind = texts["review_kind_framed" if frame == "colour" else "review_kind_framed_symmetry"].format(kind=kind)
    return kind


def _cap(text, chars):
    if len(text) <= chars:
        return text
    head = int(chars * 2 / 3)
    tail = chars - head
    return text[:head] + f"\n[... {len(text) - head - tail:,} characters omitted ...]\n" + text[-tail:]


def review_block(previous, texts, fmt="letters", source_chars=8000):
    """The answers as text: per answer its kind and explanation, its program (when any), and its test output grids."""
    blocks = [texts["review_head"]]
    for p in previous:
        blocks.append(texts["review_answer"].format(label=p["label"], attempt=p["attempt"], kind=p["kind_text"],
                                                    explanation=(p.get("explanation") or "").strip()
                                                    or texts["review_no_explanation"]))
        if p.get("source"):
            blocks.append(texts["review_program"].format(label=p["label"], attempt=p["attempt"],
                                                         source=_cap(p["source"], source_chars)))
        for j, g in enumerate(p["outputs"]):
            if g is None:
                continue
            h, w = grid_shape(g)
            blocks.append(texts["review_output"].format(label=p["label"], attempt=p["attempt"], j=j, h=h, w=w,
                                                        grid=grid_text(g, fmt)))
    blocks.append(texts["review_tail_framed" if any(p.get("frame") for p in previous) else "review_tail"])
    return "\n\n".join(blocks)


def review_messages(task, previous, texts, *, budget, fmt="letters", reply_cap=None, pin=True):
    system = system_message(texts, budget=budget, notice=0, fmt=fmt, key="system_review_v2j", reply_cap=reply_cap)
    opener = opener_message(task, texts, pin=pin, fmt=fmt)
    opener["content"] = list(opener["content"]) + [{"type": "text", "text": "\n\n" + review_block(previous, texts, fmt)}]
    return [system, opener]


def review_payload(task, previous, fmt="letters"):
    return dict(task.sandbox_payload(fmt), previous=[
        {k: p.get(k) for k in ("label", "attempt", "outputs", "source", "explanation")} for p in previous])


def run_review(task, previous, session, texts, *, budget=60_000, fmt="letters", log=None, kernel_kwargs=None, kernel=None,
               **kw):
    """One review: (AttemptResult, kernel). The caller closes the kernel (after reading its programs). kernel: an unstarted
    Kernel built from review_payload (the runner registers it for cleanup first); built here when None."""
    kernel = kernel or Kernel(review_payload(task, previous, fmt), log=log, **(kernel_kwargs or {}))
    kernel.start()
    res = run_attempt(task, session, kernel, texts, budget=budget, notice=0, log=log, fmt=fmt,
                      messages=review_messages(task, previous, texts, budget=budget, fmt=fmt,
                                               reply_cap=kw.get("reply_cap")), **kw)
    return res, kernel
