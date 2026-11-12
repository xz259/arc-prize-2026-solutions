"""ARC-2 attempt kernel (parent side): a persistent Python process per attempt with a wall-clock limit per call.

A call that exceeds the limit, or a child that dies, is answered with a fact and the kernel is restarted: the prelude is
re-run, the check/submit log restored, and the model's saved definitions (top-level imports, functions, classes and
UPPER_CASE constants of calls that completed) are replayed in order. Variables outside those are lost, and the tool
result says so. Tool text is capped (head and tail kept) like ARC-3's tool output.
"""
from __future__ import annotations

import json
import os
import queue
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHILD = HERE / "kernel_child.py"
PRELUDE = HERE / "prelude.py"


@dataclass
class KernelResult:
    text: str                       # the tool message content
    stdout: str = ""
    error: str | None = None
    events: list = field(default_factory=list)
    seconds: float = 0.0
    timed_out: bool = False
    restarted: bool = False
    reserved: list = field(default_factory=list)


class KernelDied(RuntimeError):
    pass


class Kernel:
    def __init__(self, payload, *, timeout_s=60.0, mem_mb=4096, output_chars=12000, python=None, log=None,
                 texts=None):
        self.payload = payload
        self.timeout_s = float(timeout_s)
        self.mem_mb = int(mem_mb)
        self.output_chars = int(output_chars)
        self.python = python or sys.executable
        self.log = log or (lambda kind, payload: None)
        self.texts = texts or {}
        self.proc = None
        self.lines = None
        self.library = []           # replayable entries, in definition order (a redefinition moves to the end)
        self.records = []           # check / submit / fallback records (restored after a restart)
        self.calls = 0
        self.restarts = 0

    # ------------------------------------------------------------------------------------------------ process
    def _spawn(self):
        env = dict(os.environ, ARC2_KERNEL_MEM_MB=str(self.mem_mb), PYTHONUNBUFFERED="1",
                   OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
        env.pop("LD_PRELOAD", None)
        self.proc = subprocess.Popen([self.python, "-u", str(CHILD)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env,
                                     start_new_session=True)
        self.lines = queue.Queue()
        out = self.proc.stdout

        def pump():
            for line in out:
                self.lines.put(line)
            self.lines.put(None)
        threading.Thread(target=pump, daemon=True).start()

    def _request(self, obj, timeout):
        try:
            self.proc.stdin.write(json.dumps(obj) + "\n")
            self.proc.stdin.flush()
        except (BrokenPipeError, OSError) as exc:
            raise KernelDied(f"the Python process was not accepting input ({type(exc).__name__})") from exc
        try:
            line = self.lines.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError
        if line is None:
            code = self.proc.poll()
            raise KernelDied(f"the Python process exited (code {code})")
        return json.loads(line)

    def _kill(self):
        if self.proc is None:
            return
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            self.proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            pass
        self.proc = None

    def start(self):
        self._spawn()
        resp = self._request(dict(kind="init", task=self.payload, prelude=str(PRELUDE), log=self.records,
                                  library=self.library, calls=self.calls), timeout=60)
        if not resp.get("ok"):
            raise RuntimeError(f"kernel init failed: {resp.get('error')}")
        return resp

    def restart(self, reason):
        self._kill()
        self.restarts += 1
        resp = self.start()
        self.log("kernel-restart", dict(reason=reason, restarts=self.restarts, replayed=len(self.library),
                                        replay_failed=resp.get("replay_failed")))
        return resp

    def close(self):
        self._kill()

    # ------------------------------------------------------------------------------------------------ calls
    def _remember(self, resp):
        for entry in resp.get("library") or []:
            self.library = [e for e in self.library if e["name"] != entry["name"]] + [entry]
        for rec in resp.get("events") or []:
            self.records.append(rec)

    def run(self, code, *, internal=False):
        t0 = time.monotonic()
        try:
            resp = self._request(dict(kind="exec", code=code, internal=internal), timeout=self.timeout_s)
        except TimeoutError:
            if not internal:
                self.calls += 1
            resp = self.restart("timeout")
            return self._lost(f"The call ran longer than {self.timeout_s:.0f} s and was stopped.", resp, t0, timed_out=True)
        except KernelDied as exc:
            if not internal:
                self.calls += 1
            resp = self.restart(str(exc))
            return self._lost(f"The call ended abnormally: {exc} (for example, out of memory).", resp, t0)
        self.calls = int(resp.get("calls", self.calls))
        self._remember(resp)
        text = self._format(resp)
        return KernelResult(text=text, stdout=resp.get("stdout", ""), error=resp.get("error"),
                            events=resp.get("events") or [], seconds=time.monotonic() - t0,
                            reserved=resp.get("reserved") or [])

    def _lost(self, what, resp, t0, timed_out=False):
        names = [e["name"] for e in self.library if e["kind"] in ("def", "const")]
        restored = (f" Restored definitions: {', '.join(names)}." if names else "")
        failed = resp.get("replay_failed") or []
        error = (f"{what} The Python state was reset: variables from earlier calls are gone; the preloaded names and "
                 f"the check/submit log were restored.{restored}"
                 + (f" Not restored (error on replay): {', '.join(failed)}." if failed else ""))
        text = json.dumps({"tool": "python", "error": error}, indent=2)
        return KernelResult(text=text, error=error, seconds=time.monotonic() - t0, timed_out=timed_out, restarted=True)

    def _format(self, resp):
        """ARC-3's python tool message shape: {"tool": "python", "returncode": 0, "stdout": ...} on success,
        {"tool": "python", "error": ..., "stdout": ...} on an error, as indented JSON; long output keeps its head and tail."""
        payload = {"tool": "python"}
        out = resp.get("stdout") or ""
        truncated = False
        if resp.get("omitted"):
            out += f"\n[... {resp['omitted']:,} more characters of output were not captured ...]"
            truncated = True
        if len(out) > self.output_chars:
            head = int(self.output_chars * 2 / 3)
            tail = self.output_chars - head
            out = out[:head] + f"\n[... {len(out) - head - tail:,} characters omitted ...]\n" + out[-tail:]
            truncated = True
        if resp.get("error"):
            payload["error"] = resp["error"]
        else:
            payload["returncode"] = 0
        if out.strip():
            payload["stdout"] = out.rstrip("\n")
        if truncated:
            payload["truncated"] = True
            payload["truncation_note"] = (f"Tool output was cut to about {self.output_chars:,} characters; print less, "
                                          f"or slice what you need.")
        if resp.get("reserved"):
            payload["note"] = (f"{', '.join(resp['reserved'])}: preloaded name(s) bound back to the harness's objects; "
                               f"use other names for your own.")
        return json.dumps(payload, indent=2)

    # ------------------------------------------------------------------------------------------------ harness
    def submissions(self):
        """{attempt: outputs} from the latest submit per attempt slot."""
        slots = {}
        for rec in self.records:
            if rec["kind"] == "submit":
                slots[rec["attempt"]] = rec["outputs"]
        return slots

    def fallback(self):
        """The best-checked function still defined, applied to the test inputs (None if there is none)."""
        try:
            res = self.run("_arc2_fallback()", internal=True)
        except Exception as exc:                    # noqa: BLE001 - a fallback failure only means no fallback
            self.log("kernel-fallback-error", dict(error=repr(exc)))
            return None
        for rec in res.events:
            if rec["kind"] == "fallback":
                return rec
        return None
