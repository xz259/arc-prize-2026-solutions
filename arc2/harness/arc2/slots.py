"""Submission slots and pass bands from several passes of one task (v1: the user's two-pass proposal, 2026-10-29, and
the reviewer's slot rule, PLAN B.7 item 4).

A pass contributes its final attempt-1 and attempt-2 answers (or, without either, its fallback, unverified). An answer is
verified when the kernel's own check says so (the submit record's `verified_by`: an all-pass function, or a function
that passes every demonstration and returns exactly the submitted grids). Rows from v0 runs have no `verified_by`; for
them a function answer passing every demonstration counts as verified and a grid answer does not.

Per test input, each distinct grid is ranked by: more verified votes, more valid votes (pass_valid's rule per answer:
verified code or a grid answer; a function failing some demonstration and a fallback are not valid), more votes (passes
that recorded it in either attempt), the judge's rank of the full answers containing it (when a judge ran), then first
appearance. The first two fill attempt_1 and attempt_2. Under the user's rule (2026-10-29: a task is submitted once it
has two valid solutions) the two valid solutions therefore take the two slots; when they agree, slot 2 takes the next
distinct grid.
"""
from __future__ import annotations


def pass_answers(row):
    """[(slot, outputs, verified)] of one pass row (AttemptResult JSON plus 'submissions' / 'fallback')."""
    subs = {int(k): v for k, v in (row.get("submissions") or {}).items()}
    out = []
    for a in (1, 2):
        if a not in subs or not subs[a]:
            continue
        recs = [s for s in row.get("submits") or [] if s.get("attempt") == a]
        last = recs[-1] if recs else {}
        if "verified_by" in last:
            verified = bool(last.get("verified_by"))
        else:                                                        # v0 rows
            verified = last.get("fn") not in (None, "grids") and last.get("train_passed") == last.get("train_total")
        out.append((a, subs[a], verified))
    if 2 not in subs and out and row.get("earlier"):   # v2j: an earlier, replaced valid answer fills an empty slot 2
        e = row["earlier"][0]
        out.append((2, e["outputs"], bool(e.get("verified"))))
    if not out and (row.get("fallback") or {}).get("outputs"):
        out.append((0, row["fallback"]["outputs"], False))
    return out


def _answer_valid(row, slot, verified):
    """One recorded answer under the user's rule: verified code, or a grid answer (slot 0, the fallback, is never valid)."""
    if verified:
        return True
    if not slot:
        return False
    recs = [s for s in row.get("submits") or [] if s.get("attempt") == slot]
    return bool(recs) and recs[-1].get("fn") == "grids"


def pass_valid(row):
    """The user's rule (2026-10-29): a pass produced a valid solution when a recorded answer is code that passes every
    demonstration (verified), or a grid answer (any recorded grid is a valid grid). A function failing some
    demonstration, a fallback, or nothing is not valid. Only each slot's final submission counts."""
    for a in (1, 2):
        recs = [s for s in row.get("submits") or [] if s.get("attempt") == a]
        if not recs:
            continue
        last = recs[-1]
        if last.get("fn") == "grids" or last.get("verified_by"):
            return True
        if "verified_by" not in last and last.get("fn") and last.get("train_passed") == last.get("train_total"):
            return True                                              # v0 rows: an all-pass function
    return False


def full_answers(rows):
    """Distinct full answers over passes, in first-appearance order: [{outputs, votes, verified_votes, origins}]."""
    found = []
    for p, row in enumerate(rows):
        for slot, outs, ver in pass_answers(row):
            f = next((x for x in found if x["outputs"] == outs), None)
            if f is None:
                f = dict(outputs=outs, votes=0, verified_votes=0, origins=[])
                found.append(f)
            f["origins"].append(f"p{p + 1}:a{slot}" if slot else f"p{p + 1}:fallback")
    for f in found:
        passes = {o.split(":")[0] for o in f["origins"]}
        f["votes"] = len(passes)
        f["verified_votes"] = len({f"p{p + 1}" for p, row in enumerate(rows) for slot, outs, ver in pass_answers(row)
                                   if ver and outs == f["outputs"]})
    return found


def ranked_grids(rows, i, judge_order=None):
    """Distinct grids for test input i, best first. judge_order: indices into full_answers(rows), most likely first."""
    fulls = full_answers(rows)
    jrank = {k: r for r, k in enumerate(judge_order or [])}
    grids = []
    for p, row in enumerate(rows):
        for slot, outs, ver in pass_answers(row):
            if i >= len(outs) or outs[i] is None:
                continue
            g = next((x for x in grids if x["grid"] == outs[i]), None)
            if g is None:
                g = dict(grid=outs[i], passes=set(), verified=set(), valid=set(), first=len(grids),
                         judge=len(jrank) + 1, slot=3)
                grids.append(g)
            g["passes"].add(p)
            g["slot"] = min(g["slot"], slot if slot > 0 else 3)
            if ver:
                g["verified"].add(p)
            if _answer_valid(row, slot, ver):
                g["valid"].add(p)
    for k, f in enumerate(fulls):
        if k in jrank and i < len(f["outputs"]):
            for g in grids:
                if g["grid"] == f["outputs"][i]:
                    g["judge"] = min(g["judge"], jrank[k])
    # equal votes: a pass's attempt-1 answer before any attempt-2 answer (2026-10-29, review of 581f7754: by first
    # appearance alone, pass 1's attempt 2 outranked pass 2's exact attempt 1; replay +1 to +6 outputs per run, never lower)
    grids.sort(key=lambda g: (-len(g["verified"]), -len(g["valid"]), -len(g["passes"]), g["judge"], g["slot"],
                              g["first"]))
    return grids


def slot_outputs(rows, n_tests, judge_order=None):
    """{1: [grid per test input], 2: [...]} for tasks.submission_entry (None where no grid is available)."""
    a1, a2 = [], []
    for i in range(n_tests):
        g = ranked_grids(rows, i, judge_order)
        a1.append(g[0]["grid"] if g else None)
        a2.append(g[1]["grid"] if len(g) > 1 else None)
    return {1: a1, 2: a2}


def reviewed_outputs(outs, review_submissions):
    """The reviewer pass's P3 rule (the user, 2026-10-30; fixed before the second run's read, rev12a / rev12b): slot 1
    keeps the votes' top grid; slot 2 becomes the reviewer's attempt-1 grid when it differs from slot 1, else its attempt-2
    grid when that differs, else the votes' second grid. outs: slot_outputs(...); review_submissions: the review row's
    submissions ({attempt: [grid per test input]})."""
    out = {1: list(outs[1]), 2: list(outs[2])}
    subs = {int(k): v for k, v in (review_submissions or {}).items()}

    def at(x, i):
        return x[i] if isinstance(x, list) and i < len(x) else None
    for i, s1 in enumerate(out[1]):
        r1, r2 = at(subs.get(1), i), at(subs.get(2), i)
        if r1 is not None and r1 != s1:
            out[2][i] = r1
        elif r2 is not None and r2 != s1:
            out[2][i] = r2
    return out


def band(rows):
    """Pass priority for a task from its finished passes (lower runs first):
    0 no answer at all, 1 no verified answer, 2 verified answers disagree, 3 one verified answer from a single pass,
    4 one verified answer from two or more passes (agreement)."""
    fulls = full_answers(rows)
    if not fulls:
        return 0
    ver = [f for f in fulls if f["verified_votes"]]
    if not ver:
        return 1
    if len(ver) > 1:
        return 2
    return 3 if ver[0]["verified_votes"] < 2 else 4
