"""The judge (v1; the user's proposal, 2026-10-29): when a puzzle has two or more distinct candidate answers, a fresh
session reviews them and ranks them with pick(...). The ranking orders the submission slots.

A candidate is a full answer (one grid per test input) with harness facts: its program text when a program verifiably
produced it (program.verify on the demonstrations and test inputs), and that program's demonstration passes. The order in
which candidates are shown is the caller's (shuffle them to measure the judge without position effects).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .agent import run_attempt
from .kernel import Kernel
from .prompts import grid_blocks
from .tasks import LETTER_NAMES, LETTERS, grid_shape, grid_text


@dataclass
class Candidate:
    outputs: list                    # one grid (colour codes) per test input
    source: str = ""                 # program text; '' for grids without a verified program
    passed: int | None = None        # that program's demonstration passes
    total: int | None = None
    origins: list = field(default_factory=list)      # harness bookkeeping, never shown (e.g. "run01a:a1")


def provenance(c, texts):
    if not c.source:
        return texts["judge_prov_grids"]
    if c.passed == c.total:
        return texts["judge_prov_code_all"].format(total=c.total)
    return texts["judge_prov_code_part"].format(passed=c.passed, total=c.total)


def _cap(text, chars):
    if len(text) <= chars:
        return text
    head = int(chars * 2 / 3)
    tail = chars - head
    return text[:head] + f"\n[... {len(text) - head - tail:,} characters omitted ...]\n" + text[-tail:]


def judge_messages(task, cands, texts, *, budget, notice, fmt="letters", pin=True, source_chars=8000):
    system = texts["judge_system"].format(
        grid_format=texts[f"grid_format_{fmt}"].format(legend=LETTER_NAMES),
        grid_python=texts[f"grid_python_{fmt}"].format(order=" ".join(LETTERS)), budget=budget, notice=notice)
    n_test = len(task.test)
    head = texts["judge_head"].format(task_id=task.task_id, n_train=len(task.train), n_test=n_test,
                                      inputs="input" if n_test == 1 else "inputs", n_cand=len(cands))
    blocks = []
    for k, c in enumerate(cands):
        blocks.append(texts["judge_candidate"].format(k=k, provenance=provenance(c, texts)))
        for j, g in enumerate(c.outputs):
            h, w = grid_shape(g)
            blocks.append(texts["judge_candidate_output"].format(k=k, j=j, h=h, w=w, grid=grid_text(g, fmt)))
        if c.source:
            blocks.append(texts["judge_candidate_program"].format(k=k, source=_cap(c.source, source_chars)))
    opener = {"role": "user", "content": [
        {"type": "text", "text": "\n\n".join([head] + grid_blocks(task, texts, fmt))},
        {"type": "text", "text": "\n\n" + "\n\n".join(blocks)},
        {"type": "text", "text": "\n\n" + texts["judge_tail"]}], "_arc3_observation": True}
    if pin:
        opener["_arc3_pin_parts"] = 1
    return [{"role": "system", "content": system}, opener]


def judge_payload(task, cands, fmt="letters"):
    return dict(task.sandbox_payload(fmt), candidates=[{"outputs": c.outputs, "source": c.source} for c in cands])


def run_judge(task, cands, session, texts, *, budget=32_000, notice=4_000, fmt="letters", log=None, deadline=None,
              kernel_kwargs=None, observe_turns=False):
    """One review. Returns (AttemptResult, order): order lists candidate indices, the one judged most likely right
    first (None without a recorded pick)."""
    kernel = Kernel(judge_payload(task, cands, fmt), log=log, **(kernel_kwargs or {}))
    try:
        kernel.start()
        res = run_attempt(task, session, kernel, texts, budget=budget, notice=notice, log=log, fmt=fmt,
                          deadline=deadline, observe_turns=observe_turns,
                          messages=judge_messages(task, cands, texts, budget=budget, notice=notice, fmt=fmt),
                          answered=lambda: any(r["kind"] == "pick" for r in kernel.records),
                          nudge_key="judge_nudge", notice_key="judge_budget_notice", fallback=False)
        picks = [r for r in kernel.records if r["kind"] == "pick"]
        return res, (list(picks[-1]["order"]) if picks else None)
    finally:
        kernel.close()
