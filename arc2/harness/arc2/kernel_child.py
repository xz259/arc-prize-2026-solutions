"""ARC-2 kernel child: one persistent Python namespace per attempt, driven by kernel.py over stdin/stdout JSON lines.

Requests:  {"kind": "init", "task": payload, "prelude": path, "log": [...], "library": [entry, ...]}
           {"kind": "exec", "code": source, "internal": bool}
Response:  {"ok", "stdout", "omitted", "error", "events", "library", "reserved", "replay_failed", "seconds", "calls"}
The model's prints go to a captured buffer; the protocol uses the process's real stdout only. A call's code is compiled
under the file name "<call N>" (registered with linecache), so tracebacks show the model's own lines and programs()
can tell the model's functions from the harness's.
"""
import ast
import contextlib
import io
import json
import linecache
import os
import sys
import time
import traceback

OUTPUT_CHARS = int(os.environ.get("ARC2_KERNEL_CAPTURE_CHARS", "400000"))
RESERVED = ("np", "train", "test", "check", "submit", "note", "history", "programs", "source", "show")


def _limits():
    mb = int(os.environ.get("ARC2_KERNEL_MEM_MB", "4096"))
    if mb <= 0:
        return
    try:
        import resource
        resource.setrlimit(resource.RLIMIT_AS, (mb << 20, mb << 20))
    except (ImportError, ValueError, OSError):
        pass                                    # not enforceable here (e.g. macOS); the parent's timeout still applies


def _traceback(exc):
    frames = [f for f in traceback.extract_tb(exc.__traceback__) if str(f.filename).startswith("<call ")]
    lines = []
    for f in frames[-4:]:
        where = f"{f.filename[1:-1]}, line {f.lineno}" + ("" if f.name == "<module>" else f", in {f.name}")
        lines.append(f"  {where}" + (f": {f.line.strip()}" if f.line else ""))
    tail = "".join(traceback.format_exception_only(type(exc), exc)).strip()
    return ("Traceback (most recent call last):\n" + "\n".join(lines) + "\n" + tail) if lines else tail


def _library_entries(code, tree):
    """Top-level imports, functions, classes and UPPER_CASE assignments of a call that ran to completion."""
    out = []
    for node in tree.body:
        src = ast.get_source_segment(code, node)
        if src is None:
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            out.append({"name": node.name, "kind": "def", "source": src})
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            out.append({"name": "import:" + src, "kind": "import", "source": src})
        elif isinstance(node, ast.Assign) and node.targets and all(
                isinstance(t, ast.Name) and t.id.isupper() for t in node.targets):
            out.append({"name": node.targets[0].id, "kind": "const", "source": src})
    return out


def _execute(code, ns, filename):
    linecache.cache[filename] = (len(code), None, code.splitlines(True), filename)
    tree = ast.parse(code, filename)
    full = ast.parse(code, filename)
    last = None
    if tree.body and isinstance(tree.body[-1], ast.Expr):
        last = ast.Expression(tree.body.pop().value)
    exec(compile(tree, filename, "exec"), ns)
    if last is not None:
        value = eval(compile(last, filename, "eval"), ns)
        if value is not None:
            print(repr(value))
    return full


def main():
    _limits()
    proto = sys.stdout
    sys.stdout = io.StringIO()                  # stray prints outside a call never reach the protocol channel
    ns = {"__name__": "__arc2__", "__builtins__": __builtins__}
    calls = [0]
    for line in sys.stdin:
        req = json.loads(line)
        events, library, reserved, replay_failed = [], [], [], []
        buf = io.StringIO()
        error = None
        t0 = time.monotonic()
        if req["kind"] == "init":
            ns.update(_ARC2_TASK=req["task"], _ARC2_STATE={"log": list(req.get("log") or [])},
                      _ARC2_EVENTS=events, _ARC2_CALL=calls, _ARC2_SOURCES={})
            try:
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    with open(req["prelude"]) as f:
                        exec(compile(f.read(), "<arc2>", "exec"), ns)
                for entry in req.get("library") or []:   # replay after a restart, in definition order
                    try:
                        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                            exec(compile(entry["source"], f"<call {entry.get('call', 0)}>", "exec"), ns)
                        if entry["kind"] in ("def", "const"):
                            ns["_ARC2_SOURCES"][entry["name"]] = entry["source"]
                    except BaseException as exc:          # noqa: BLE001 - reported, never fatal
                        replay_failed.append(f"{entry['name']}: {type(exc).__name__}")
                calls[0] = int(req.get("calls") or 0)
            except BaseException as exc:                  # noqa: BLE001
                error = "init failed: " + "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))[-2000:]
        else:
            internal = bool(req.get("internal"))
            if not internal:
                calls[0] += 1
            ns["_ARC2_EVENTS"] = events
            saved = {k: ns.get(k) for k in RESERVED + tuple(ns.get("_ARC2_RESERVED_EXTRA") or ())}
            filename = "<harness>" if internal else f"<call {calls[0]}>"
            try:
                with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                    tree = _execute(req["code"], ns, filename)
                if not internal:
                    library = _library_entries(req["code"], tree)
                    for entry in library:
                        entry["call"] = calls[0]
                        if entry["kind"] in ("def", "const"):
                            ns["_ARC2_SOURCES"][entry["name"]] = entry["source"]
            except SystemExit:
                error = "SystemExit was raised; the call stopped there."
            except KeyboardInterrupt:
                error = "KeyboardInterrupt"
            except BaseException as exc:                  # noqa: BLE001 - the model's errors are facts
                error = _traceback(exc)
            for k, v in saved.items():                    # helper names stay bound to the harness's objects
                if ns.get(k) is not v and v is not None:
                    ns[k] = v
                    reserved.append(k)
        text = buf.getvalue()
        omitted = max(0, len(text) - OUTPUT_CHARS)
        resp = dict(ok=error is None, stdout=text[:OUTPUT_CHARS], omitted=omitted, error=error, events=events,
                    library=library, reserved=reserved, replay_failed=replay_failed, seconds=round(time.monotonic() - t0, 4), calls=calls[0])
        proto.write(json.dumps(resp, default=str) + "\n")
        proto.flush()


if __name__ == "__main__":
    main()
