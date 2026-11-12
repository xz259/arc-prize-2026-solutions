"""Programs of an attempt, rebuilt from its python calls (harness side; the model is not involved).

The kernel's restart replay keeps the top-level imports, functions, classes and UPPER_CASE constants of the calls that
completed (kernel_child._library_entries), latest definition last. library() applies the same rule to a log's calls.
program_source() returns what one function needs: the imports, plus the definitions it reaches by name (transitively),
in definition order. verify() runs that text in a fresh kernel and reports the demonstration passes and the test outputs.
"""
from __future__ import annotations

import ast
import json

from .kernel import Kernel
from .kernel_child import _library_entries


def library(codes):
    """codes: source texts of completed python calls, oldest first -> {name: entry} in replay order."""
    lib = {}
    for code in codes:
        try:
            tree = ast.parse(code)
        except SyntaxError:
            continue
        for entry in _library_entries(code, tree):
            lib.pop(entry["name"], None)
            lib[entry["name"]] = entry
    return lib


def log_library(log_rows, upto_call=None):
    """The library of an attempt's task log (kind 'tool' rows) up to and including python call `upto_call`."""
    codes = []
    for r in log_rows:
        if r.get("kind") != "tool":
            continue
        p = r["payload"]
        if upto_call is not None and int(p.get("call") or 0) > upto_call:
            break
        if p.get("error") is None and not p.get("timed_out"):
            codes.append(p.get("code") or "")
    return library(codes)


def program_source(lib, name):
    """Imports plus the definitions `name` reaches by name, in replay order; None when `name` is not defined."""
    if name not in lib or lib[name]["kind"] != "def":
        return None
    need, stack = set(), [name]
    while stack:
        n = stack.pop()
        if n in need or n not in lib or lib[n]["kind"] == "import":
            continue
        need.add(n)
        try:
            tree = ast.parse(lib[n]["source"])
        except SyntaxError:
            continue
        stack.extend(sub.id for sub in ast.walk(tree) if isinstance(sub, ast.Name))
    imports = [e["source"] for e in lib.values() if e["kind"] == "import"]
    defs = [e["source"] for k, e in lib.items() if k in need]
    return "\n".join(imports) + ("\n\n" if imports else "") + "\n\n".join(defs)


VERIFY = """
import json as _arc2_json
_arc2_v = {{"passed": 0, "total": len(_ARC2_TRAIN), "outputs": [], "error": None}}
try:
    _arc2_f = globals()[{name!r}]
    for _p in _ARC2_TRAIN:
        _g, _ = _arc2_apply(_arc2_f, _p["input"])
        _arc2_v["passed"] += int(_g is not None and _arc2_compare(_g, _p["output"])[0] == "pass")
    for _t in _ARC2_TEST:
        _g, _why = _arc2_apply(_arc2_f, _t)
        _arc2_v["outputs"].append(None if _g is None else _g.tolist())
except Exception as _e:
    _arc2_v["error"] = f"{{type(_e).__name__}}: {{_e}}"
print("ARC2VERIFY " + _arc2_json.dumps(_arc2_v))
"""


def verify(payload, source, name, *, timeout_s=60.0, mem_mb=4096):
    """Run a program text in a fresh kernel on a task payload: {passed, total, outputs, error}."""
    k = Kernel(payload, timeout_s=timeout_s, mem_mb=mem_mb, output_chars=10 ** 7)
    try:
        k.start()
        r = k.run(source, internal=True)
        if r.error:
            return dict(passed=0, total=len(payload["train"]), outputs=[], error=f"definition failed: {r.error[:300]}")
        r = k.run(VERIFY.format(name=name), internal=True)
        for line in (r.stdout or "").splitlines():
            if line.startswith("ARC2VERIFY "):
                return json.loads(line[len("ARC2VERIFY "):])
        return dict(passed=0, total=len(payload["train"]), outputs=[], error=(r.error or "no result")[:300])
    finally:
        k.close()
