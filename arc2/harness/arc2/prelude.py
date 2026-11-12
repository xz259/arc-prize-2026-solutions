"""Executed in every attempt's kernel namespace at start, and again after a kernel restart (kernel_child.py).

Model-facing names: np, train, test, check, submit, note, history, source, show (and programs, described only in the v1
prompt). Everything else starts with _arc2 / _ARC2. Grids are numpy arrays in the attempt's format: single-character colour symbols (ARC-3's letters, the default) or
integers 0-9 (digits). A function may return either form, a list of row strings, or nested lists; every output is
normalized to colour codes 0-9 before comparison and submission. Expected outputs and inputs come from private copies,
so a model that edits `train` or `test` cannot change what is checked or submitted. Results are facts only.
"""
import contextlib as _arc2_contextlib
import copy as _arc2_copy
import io as _arc2_io

import numpy as np

_ARC2_FMT = _ARC2_TASK.get("format", "letters")
_ARC2_LETTERS = _ARC2_TASK.get("letters", "BbRNYgMOSr")
_ARC2_CODE = {ch: i for i, ch in enumerate(_ARC2_LETTERS)}
_ARC2_SYMBOLS = _ARC2_LETTERS if _ARC2_FMT == "letters" else "0123456789"
_ARC2_DIFF_CELLS = 8


def _arc2_view(codes):
    """Colour codes -> the model-facing array (letters or ints)."""
    codes = np.asarray(codes, dtype=int)
    return np.array(list(_ARC2_LETTERS))[codes] if _ARC2_FMT == "letters" else codes.copy()


_ARC2_TRAIN = [{"input": np.array(p["input"], dtype=int), "output": np.array(p["output"], dtype=int)}
               for p in _ARC2_TASK["train"]]
_ARC2_TEST = [np.array(p["input"], dtype=int) for p in _ARC2_TASK["test"]]
train = [{"input": _arc2_view(p["input"]), "output": _arc2_view(p["output"])} for p in _ARC2_TRAIN]
test = [{"input": _arc2_view(t)} for t in _ARC2_TEST]          # ARC's own task format, like train


def _arc2_codes(x):
    """(colour-code int array, None) for a valid ARC grid in any accepted form, else (None, reason)."""
    if isinstance(x, str):
        x = x.splitlines()
    if isinstance(x, (list, tuple)) and x and all(isinstance(r, str) for r in x):
        x = [list(r) for r in x]
    try:
        a = np.asarray(x)
    except Exception as exc:            # ragged nesting and the like
        return None, f"not convertible to a grid ({type(exc).__name__})"
    if a.dtype == object:
        return None, "not a rectangular 2-D grid"
    if a.ndim == 1 and a.dtype.kind == "U" and a.size and len({len(r) for r in a}) == 1 and len(a[0]) > 1:
        a = np.array([list(r) for r in a])
    if a.ndim != 2:
        return None, f"not a 2-D grid (got shape {a.shape})"
    h, w = a.shape
    if not (1 <= h <= 30 and 1 <= w <= 30):
        return None, f"size {h}x{w} is outside 1-30 per side"
    if a.dtype.kind == "U":
        bad = sorted({str(v) for v in np.unique(a)} - set(_ARC2_CODE))
        if bad:
            return None, f"unknown symbols {bad[:5]} (allowed: {' '.join(_ARC2_LETTERS)})"
        return np.vectorize(_ARC2_CODE.get, otypes=[int])(a), None
    if a.dtype.kind == "f":
        if not (np.all(np.isfinite(a)) and np.all(a == np.round(a))):
            return None, f"cells are not whole numbers (dtype {a.dtype})"
        a = a.astype(int)
    if a.dtype.kind not in "iub":
        return None, f"cells are neither colour symbols nor integers (dtype {a.dtype})"
    if a.min() < 0 or a.max() > 9:
        return None, "cell values outside 0-9"
    return a.astype(int), None


def _arc2_sym(code):
    return _ARC2_SYMBOLS[int(code)]


def _arc2_name(fn):
    return getattr(fn, "__name__", type(fn).__name__)


def _arc2_apply(fn, codes):
    try:
        out = fn(_arc2_view(codes))
    except Exception as exc:
        return None, f"error: {type(exc).__name__}: {exc}"
    g, why = _arc2_codes(out)
    if g is None:
        return None, f"invalid output: {why}"
    return g, None


def _arc2_apply_hidden(fn, i):
    """v2j (2026-10-29, review of abc82100 / faa9f03d): demonstration i run with train holding only the OTHER pairs and
    test empty, so a function that reads its answers from train or special-cases the test inputs does not verify.
    Functions that do not read train or test are unaffected."""
    g = globals()
    saved = g.get("train"), g.get("test")
    g["train"] = [{"input": _arc2_view(p["input"]), "output": _arc2_view(p["output"])}
                  for k, p in enumerate(_ARC2_TRAIN) if k != i]
    g["test"] = []
    try:
        return _arc2_apply(fn, _ARC2_TRAIN[i]["input"])
    finally:
        g["train"], g["test"] = saved


def _arc2_reads_test(fn, _seen=None):
    """v2j2 (the reviewer, the user's OK, 2026-10-29): True when fn's code, or a function it calls by global name, reads
    the global test or the harness's own data (_ARC2_*). A solver has no reason to read the test inputs, so such a function
    is not counted as verified. Literal test coordinates and earlier aliases are not caught."""
    seen = set() if _seen is None else _seen
    code = getattr(fn, "__code__", None)
    if code is None or id(code) in seen:
        return False
    seen.add(id(code))
    stack, names = [code], set()
    while stack:
        c = stack.pop()
        names.update(c.co_names)
        stack.extend(x for x in c.co_consts if hasattr(x, "co_names"))
    if "test" in names or any(n.startswith("_ARC2") for n in names):
        return True
    g = globals()
    return any(_arc2_reads_test(g[n], seen) for n in names if callable(g.get(n)) and getattr(g[n], "__globals__", None) is g
               and n not in ("check", "submit", "note", "history", "programs", "source", "show"))


def _arc2_event(rec):
    rec["call"] = _ARC2_CALL[0]
    _ARC2_STATE["log"].append(rec)
    _ARC2_EVENTS.append(rec)


def _arc2_compare(g, exp):
    """('pass' | 'shape' | 'cells', text), with cells shown in the attempt's symbols and (row, col) coordinates."""
    if g.shape != exp.shape:
        return "shape", f"wrong shape: expected {exp.shape[0]}x{exp.shape[1]}, got {g.shape[0]}x{g.shape[1]}"
    bad = np.argwhere(g != exp)
    if len(bad) == 0:
        return "pass", "pass"
    first = "; ".join(f"({r},{c}) expected {_arc2_sym(exp[r, c])} got {_arc2_sym(g[r, c])}"
                      for r, c in bad[:_ARC2_DIFF_CELLS])
    return "cells", f"{len(bad)} of {exp.size} cells differ; first: {first}"


def check(fn):
    """Run fn on every demonstration input; report per pair whether the output is exactly right, else how it differs.
    Also reports what fn returns for the test inputs. Returns True when every demonstration pair passes."""
    name = _arc2_name(fn)
    kinds, lines = [], []
    hidden_note = (" (it passes only when the function can read this pair from train, or the test inputs from test; a "
                   "pair is checked with only the other pairs in train and test empty)")
    for i, pair in enumerate(_ARC2_TRAIN):
        g, why = _arc2_apply_hidden(fn, i)
        if g is None:
            kind, text = ("error" if why.startswith("error") else "invalid"), why
        else:
            kind, text = _arc2_compare(g, pair["output"])
        if kind != "pass":                        # reviewer 10-29: explain hidden-pair failures, errors included
            g2, _ = _arc2_apply(fn, pair["input"])
            if g2 is not None and _arc2_compare(g2, pair["output"])[0] == "pass":
                text += hidden_note
        kinds.append(kind)
        lines.append(f"pair {i}: {text}")
    tests = []                                    # v2h (2026-10-29): the predicted test grids, as the prompt says
    for j, t in enumerate(_ARC2_TEST):
        g, why = _arc2_apply(fn, t)
        if g is None:
            tests.append(f"test input {j}: {why}")
        else:
            tests.append(f"test input {j} -> {g.shape[0]}x{g.shape[1]}:\n"
                         + "\n".join("".join(_arc2_sym(v) for v in row) for row in g))
    passed = kinds.count("pass")
    reads = _arc2_reads_test(fn)
    if reads:                                     # v2j2: a function that reads test does not count as passing (v2n wording)
        lines.append(f"does not count as passing: {name} reads the test inputs (test)")
    print("\n".join([f"check({name}): {passed} of {len(_ARC2_TRAIN)} demonstration pairs pass."] + lines + tests))
    _arc2_event({"kind": "check", "fn": name, "passed": passed, "total": len(_ARC2_TRAIN), "results": kinds,
                 "reads_test": reads})
    return passed == len(_ARC2_TRAIN) and not reads


def submit(answer, attempt=1, explanation=None):
    """Record the answer for every test input: a function (applied to each test input) or the grids themselves (a list
    with one grid per test input; a single grid when there is one test input), with an optional explanation of the rule
    in words. attempt=2 records a second, different answer; the puzzle counts as solved if either attempt is exactly
    right. A later submit replaces that attempt."""
    if attempt not in (1, 2):
        print("submit: attempt must be 1 or 2; nothing recorded.")
        return False
    outs = []
    if callable(answer):
        name = _arc2_name(answer)
        for j, t in enumerate(_ARC2_TEST):
            g, why = _arc2_apply(answer, t)
            if g is None:
                print(f"submit({name}): nothing recorded; test input {j}: {why}")
                return False
            outs.append(g)
        passed = 0
        for i, p in enumerate(_ARC2_TRAIN):
            g, _ = _arc2_apply_hidden(answer, i)
            passed += int(g is not None and _arc2_compare(g, p["output"])[0] == "pass")
        train_note = f"; demonstrations: {passed} of {len(_ARC2_TRAIN)} pass"
    else:
        name, passed, train_note = "grids", None, ""
        single, _ = _arc2_codes(answer)
        seq = [answer] if (single is not None and len(_ARC2_TEST) == 1) else answer
        try:
            seq = list(seq)
        except TypeError:
            print("submit: pass a function or a list with one grid per test input; nothing recorded.")
            return False
        if len(seq) != len(_ARC2_TEST):
            print(f"submit: {len(seq)} grids given for {len(_ARC2_TEST)} test inputs; nothing recorded.")
            return False
        for j, x in enumerate(seq):
            g, why = _arc2_codes(x)
            if g is None:
                print(f"submit: test input {j}: {why}; nothing recorded.")
                return False
            outs.append(g)
    reads = callable(answer) and _arc2_reads_test(answer)
    if reads:
        train_note += "; does not count as passing: it reads the test inputs (test)"
    verified = ((name if passed == len(_ARC2_TRAIN) and not reads else None) if callable(answer)
                else _arc2_verified_by(outs))
    rec = {"kind": "submit", "attempt": attempt, "fn": name, "outputs": [o.tolist() for o in outs],
           "train_passed": passed, "train_total": len(_ARC2_TRAIN), "verified_by": verified}
    if explanation is not None:
        rec["explanation"] = str(explanation)[:_ARC2_NOTE_CHARS]
    _arc2_event(rec)
    shapes = ", ".join(f"test input {j} -> {o.shape[0]}x{o.shape[1]}" for j, o in enumerate(outs))
    print(f"Recorded attempt {attempt} ({name}): {shapes}{train_note}.")
    for j, o in enumerate(outs):                  # v2j: the recorded grids, as check() shows them (v2h)
        print(f"test input {j} -> {o.shape[0]}x{o.shape[1]}:\n" + "\n".join("".join(_arc2_sym(v) for v in row) for row in o))
    return True


def _arc2_verified_by(outs):
    """Harness only: the latest-checked defined function that passes every demonstration now and returns exactly these
    test grids (its prints are discarded), else None. A grid answer counts as verified only through this check."""
    names = []
    for r in reversed(_ARC2_STATE["log"]):
        if r["kind"] == "check" and r["passed"] == r["total"] and r["fn"] not in names:
            names.append(r["fn"])
    with _arc2_contextlib.redirect_stdout(_arc2_io.StringIO()):
        for n in names:
            f = globals().get(n)
            if not callable(f) or _arc2_reads_test(f):
                continue
            ok = True
            for i, p in enumerate(_ARC2_TRAIN):
                g, _ = _arc2_apply_hidden(f, i)
                if g is None or _arc2_compare(g, p["output"])[0] != "pass":
                    ok = False
                    break
            for t, o in zip(_ARC2_TEST, outs):
                if not ok:
                    break
                g, _ = _arc2_apply(f, t)
                ok = g is not None and g.shape == o.shape and bool(np.all(g == o))
            if ok:
                return n
    return None


_ARC2_NOTE_CHARS = 4000


def note(text):
    """Save a note for this puzzle: a finding, a hypothesis tested, anything worth keeping. history() lists the notes."""
    text = str(text)
    _arc2_event({"kind": "note", "text": text[:_ARC2_NOTE_CHARS]})
    n = sum(1 for r in _ARC2_STATE["log"] if r["kind"] == "note")
    print(f"Saved note {n}" + (f" (cut to {_ARC2_NOTE_CHARS} characters)." if len(text) > _ARC2_NOTE_CHARS else "."))
    return True


def history():
    """Print this puzzle's record so far, oldest first (call numbers count python calls): checks, submissions, notes, and
    the functions defined with their latest check result."""
    rows = []
    for r in _ARC2_STATE["log"]:
        if r["kind"] == "check":
            rows.append(f"call {r['call']}: check({r['fn']}) {r['passed']} of {r['total']} pass ({', '.join(r['results'])})")
        elif r["kind"] == "submit":
            extra = "" if r["train_passed"] is None else f", demonstrations {r['train_passed']} of {r['train_total']}"
            if r.get("verified_by") and r["fn"] == "grids":
                extra += f", the outputs of {r['verified_by']}, which passes every demonstration"
            rows.append(f"call {r['call']}: submit attempt {r['attempt']} ({r['fn']}{extra})"
                        + (f"; explanation: {r['explanation']}" if r.get("explanation") else ""))
        elif r["kind"] == "note":
            rows.append(f"call {r['call']}: note: {r['text']}")
    latest = {}
    for r in _ARC2_STATE["log"]:
        if r["kind"] == "check":
            latest[r["fn"]] = r
    fns = []
    for n, v in list(globals().items()):
        code = getattr(v, "__code__", None)
        if code is None or not str(code.co_filename).startswith("<call "):
            continue
        r = latest.get(n)
        fns.append(f"{n} ({code.co_filename[1:-1]}; " + (f"latest check {r['passed']} of {r['total']})" if r else "not checked)"))
    if fns:
        rows.append("functions defined: " + "; ".join(fns))
    print("\n".join(rows) if rows else "Nothing recorded yet.")


def programs():
    """Print the functions defined so far (in python calls) with their latest check result."""
    latest = {}
    for r in _ARC2_STATE["log"]:
        if r["kind"] == "check":
            latest[r["fn"]] = r
    rows = []
    for n, v in list(globals().items()):
        code = getattr(v, "__code__", None)
        if code is None or not str(code.co_filename).startswith("<call "):
            continue
        r = latest.get(n)
        status = f"latest check {r['passed']} of {r['total']}" if r else "not checked"
        doc = (getattr(v, "__doc__", None) or "").strip().split("\n")[0][:100]
        rows.append(f"{n}: defined in {code.co_filename[1:-1]}; {status}" + (f"; {doc}" if doc else ""))
    print("\n".join(rows) if rows else "No functions defined yet.")


def source(name):
    """Print the source of a function, class or UPPER_CASE constant defined in an earlier python call."""
    text = _ARC2_SOURCES.get(name)
    print(text if text is not None else f"No saved source for {name!r}.")


def show(grid):
    """Print a grid as rows of symbols, with its size."""
    g, why = _arc2_codes(grid)
    if g is None:
        print(f"show: {why}")
        return
    print(f"{g.shape[0]}x{g.shape[1]}")
    print("\n".join("".join(_arc2_sym(v) for v in row) for row in g))


def _arc2_fallback():
    """Harness only (not described to the model): the best-checked function still defined, applied to the test inputs.
    Used when an attempt ends without a submission."""
    best = None
    for r in _ARC2_STATE["log"]:
        if r["kind"] == "check" and callable(globals().get(r["fn"])) and (best is None or r["passed"] >= best["passed"]):
            best = r
    if best is None:
        return
    outs = []
    for t in _ARC2_TEST:
        g, _ = _arc2_apply(globals()[best["fn"]], t)
        if g is None:
            return
        outs.append(g.tolist())
    _arc2_event({"kind": "fallback", "fn": best["fn"], "passed": best["passed"], "total": best["total"], "outputs": outs})


# ------------------------------------------------------------------ review sessions only (payload key "previous")
if _ARC2_TASK.get("previous") is not None:
    # the reviewer (the user's design, 2026-10-30): the two earlier attempts' answers, with programs and explanations
    previous = [{"label": p["label"], "attempt": p["attempt"],
                 "outputs": [_arc2_view(g) if g is not None else None for g in p["outputs"]],
                 "source": p.get("source") or "", "explanation": p.get("explanation") or ""}
                for p in _ARC2_TASK["previous"]]
    _ARC2_RESERVED_EXTRA = ("previous",)


# ------------------------------------------------------------------ judge sessions only (payload key "candidates")
if _ARC2_TASK.get("candidates") is not None:
    _ARC2_CANDIDATES = [{"outputs": [np.array(g, dtype=int) for g in c["outputs"]], "source": c.get("source") or ""}
                        for c in _ARC2_TASK["candidates"]]
    candidates = [{"outputs": [_arc2_view(g) for g in c["outputs"]], "source": c["source"]} for c in _ARC2_CANDIDATES]
    _ARC2_RESERVED_EXTRA = ("candidates", "pick")

    def pick(*order):
        """Record your ranking of the candidates: indices into candidates, the one most likely right first. Give at
        least one index, each at most once. Picking again replaces the ranking."""
        n = len(_ARC2_CANDIDATES)
        try:
            order = [int(i) for i in order]
        except (TypeError, ValueError):
            print("pick: pass candidate indices (integers); nothing recorded.")
            return False
        if not order or len(set(order)) != len(order) or any(not 0 <= i < n for i in order):
            print(f"pick: give one or more distinct indices from 0 to {n - 1}; nothing recorded.")
            return False
        _arc2_event({"kind": "pick", "order": order})
        print(f"Recorded ranking: {', '.join(map(str, order))}.")
        return True
