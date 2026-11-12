"""kernel_ext.py (2026-10-25, Claude main session 099768dc; the user: "apply the alias edits since it is free? Then we A/B the current
harness + updated / model preferred alias vs the same harness but with a stateful kernel ... We do need to make sure it is robust";
"What about 60s ...": the wall-clock backstop). Installed by run_v6.py after lazy_history.install(), so it patches the final bootstrap.

Always (both arms):
  - the names the model guesses (the 9-h n76 runs' NameError / AttributeError census): last_action_timeline = last_animation_timeline;
    history entries .level / .step; transitions .level / .step / .level_completed / .animation_frames; FrameView.frame (the view
    itself). Non-data descriptors, so code that assigns those attributes still can;
  - the wall-clock backstop 120 -> 60 s (no call in the n76 runs took more than 34 s); the model-facing text states it.
ARC3_V80_STATEFUL=1 (arm B): a persistent sandbox worker per conversation (the agent's server session object), so Python variables
persist between calls as in a notebook. The worker runs the same restricted bootstrap in a loop:
  - one namespace per worker: the runtime globals (current_frame, history, action, ...) are re-injected and `result` is reset at every
    call; the per-call closures (action latches, history caches, animation views) are rebuilt as before;
  - the per-call CPU limit by ITIMER_PROF (RLIMIT_CPU counts the whole process): ComputeLimitError at the limit (the worker and its
    variables survive), os._exit(152) 5 s later if the snippet swallowed it (reported as the hard CPU kill);
  - the program library as before: restored every call from the host's sources, collected every call, its dependencies judged against
    the provided names only (a library function never depends on a variable, so it still loads in a fresh worker);
  - a worker that died, hit the 60-s wall backstop, or was evicted is replaced at the next call, whose output starts with a note;
  - a new conversation (a new session object) gets a fresh worker; workers exit on stdin EOF (the client's death); at most
    ARC3_V80_KERNEL_CAP live workers (LRU among idle ones, default 48), idle ones killed after ARC3_V80_KERNEL_IDLE_S (5,400 s);
    RLIMIT_AS 2 GiB per worker; the worker's stderr is drained continuously (a full pipe would block it).
v2 (2026-10-25, IE review review74/v80_ie/KERNEL_V80_REVIEW_ie_2026-10-25.md):
  - F1: the seven action exception classes are one object per worker and `action` is one stable proxy calling this call's action, so
    a reference the model keeps across calls (`act = action`, `ERRS = (InvalidActionError, ...)`) matches and gets fresh latches;
  - F2: a worker that cannot start (OSError: pids, memory, disk) runs that call in the per-call sandbox instead of raising;
  - F3: a worker whose resident memory exceeds ARC3_V80_KERNEL_RSS_MB (768) after a call is replaced (with the note); IDLE_S doubled;
  - F5: install() asserts both bootstraps fit one argv string (Linux: 131,072 bytes);
  - F12: the .level / .step aliases read the history entry's metadata first (no frame fetch); the counters are logged every 2,000 calls.
v3 (2026-10-25, the k81s A/B's early check: two kernel deaths on MemoryError in one game, the handler itself out of memory):
  - RLIMIT_AS 4 GiB per worker, as the per-call sandbox (v1/v2 had 2 GiB);
  - a 32 MB address-space reserve per worker, released first by the error handler and the fatal path, so a MemoryError is reported
    with its traceback and the fatal message is never lost (re-made at the next call's start);
  - the per-call hard CPU bound enforced by the host as well: the worker's CPU time (/proc/<pid>/stat) checked every second, killed at
    the limit + 6 s (the per-call sandbox: RLIMIT_CPU's SIGKILL at the limit + 5 s); the in-worker exit 152 at + 5 s needs a bytecode
    boundary, so a long C-level operation (a huge sort, a GC pass) deferred it to the 60-s wall.
v4 (2026-10-25, the user adopted the kernel (a notebook boolean, default ON) and the final polishes from the k81s A/B):
  - both arms: a 32-MB reserve released the moment a MemoryError leaves the snippet (before any cleanup), so the error is reported;
    `globals` / `locals` / `vars` and `ImportError` / `ModuleNotFoundError` allowed (the user 10-26: a model's import fallback catches); `copy` / `bisect` preloaded; FrameView.split / .splitlines (the board's text);
    action_trace lines that also read as the per-action record (line["action"], .get("verdict"), .interior_changed["n"]), from
    the host's per-action facts (step_env -> action_trace_details, passed through the compact result);
  - the stateful kernel: each call compiled under its own filename (F6: tracebacks into functions from earlier calls show their
    own source lines); the v3 handler reserve folded into the shared one; the cgroup's pids headroom logged (and the cap lowered
    if it is small).
"""
import atexit
import collections
import json
import logging
import os
import queue
import shutil
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid

from inference.agent import python_tool_sandbox as sb
from inference.agent import tool_agent as ta
from inference.utils.grid_utils import ARC_COLOR_CHARS
from v62 import sandbox_ext

log = logging.getLogger(__name__)
STATEFUL = os.environ.get('ARC3_V80_STATEFUL', '0') == '1'
KERNEL_CAP = int(os.environ.get('ARC3_V80_KERNEL_CAP', '48'))
IDLE_S = float(os.environ.get('ARC3_V80_KERNEL_IDLE_S', '5400'))
RSS_MB = float(os.environ.get('ARC3_V80_KERNEL_RSS_MB', '768'))
WALL_S = 60
AS_BYTES = int(os.environ.get('ARC3_V80_KERNEL_AS_MB', '4096')) * 1024 ** 2   # v3: as the per-call sandbox (4294967296)
RESERVE_BYTES = 32 * 1024 ** 2    # v3: released first when an error is handled
CPU_TOTAL_S = 6 * 3600
HARD_EXIT = 152
CPU_GRACE_S = 6                    # v3: the host's hard CPU bound, seconds past the call's limit
ARGV_MAX = 120_000                 # IE F5: one argv string is at most 131,072 bytes on Linux (MAX_ARG_STRLEN)
LOG_EVERY = 2000
try:
    _PAGE = os.sysconf('SC_PAGE_SIZE')
    _TICK = float(os.sysconf('SC_CLK_TCK'))
except (AttributeError, ValueError, OSError):
    _PAGE, _TICK = 4096, 100.0
STATS = collections.Counter()
STATEFUL_BOOTSTRAP = None
_PER_CALL = None
_INSTALLED = []


def _rep(text, old, new, what):
    n = text.count(old)
    assert n == 1, 'kernel_ext: anchor %r found %d times' % (what, n)
    return text.replace(old, new)


# ----------------------------------------------------------------------------------------------------------- the aliases (both arms)
_ALIAS_CLASSES = '''
class _V80Alias:
    """v8.0: a read-only alias the model guesses; a non-data descriptor, so assigning the attribute still works."""
    def __init__(self, fn):
        self._fn = fn

    def __get__(self, obj, cls=None):
        return self if obj is None else self._fn(obj)


def _v80_transition_animation(t):
    h, i = t.__dict__.get("_v6_h"), t.__dict__.get("_v6_i")
    return h[i].animation_frames if h is not None and i is not None else []


def _v80_meta_or_frame(entry, key):
    meta = getattr(entry, "__dict__", {}).get("_v6_meta")     # a lazy entry's metadata (the host's frame.level / .step)
    value = meta.get(key) if isinstance(meta, dict) else None
    return value if value is not None else getattr(entry.frame, key, None)


def _v80_transition_attr(t, key):
    h, i = t.__dict__.get("_v6_h"), t.__dict__.get("_v6_i")
    if h is not None and i is not None:
        return _v80_meta_or_frame(h[i], key)
    return getattr(t.after_frame, key, None)


FrameView.frame = _V80Alias(lambda self: self)
HistoryEntryView.level = _V80Alias(lambda self: _v80_meta_or_frame(self, "level"))
HistoryEntryView.step = _V80Alias(lambda self: _v80_meta_or_frame(self, "step"))
TransitionView.level = _V80Alias(lambda self: _v80_transition_attr(self, "level"))
TransitionView.step = _V80Alias(lambda self: _v80_transition_attr(self, "step"))
TransitionView.level_completed = _V80Alias(lambda self: bool((self.result or {}).get("level_completed")))
TransitionView.animation_frames = _V80Alias(_v80_transition_animation)
FrameView.split = _V80Alias(lambda self: self.ascii.split)              # v8.1: a board read as its text
FrameView.splitlines = _V80Alias(lambda self: self.ascii.splitlines)


class _V81TraceLine(str):
    """v8.1: one action_trace line. It prints and compares as before, and it also reads as the per-action record
    (line["action"], line.get("verdict"), line.interior_changed["n"]); string indexing and methods are unchanged."""

    def __new__(cls, text, record=None):
        obj = str.__new__(cls, text)
        obj.__dict__["_v81"] = dict(record or {})
        return obj

    def __getitem__(self, key):
        if isinstance(key, str):
            return self.__dict__["_v81"][key]
        return str.__getitem__(self, key)

    def get(self, key, default=None):
        return self.__dict__["_v81"].get(key, default)

    def keys(self):
        return self.__dict__["_v81"].keys()

    def values(self):
        return self.__dict__["_v81"].values()

    def items(self):
        return self.__dict__["_v81"].items()

    def __getattr__(self, name):
        record = self.__dict__.get("_v81") or {}
        if name in record:
            return record[name]
        raise AttributeError("'str' object has no attribute %r" % name)


def _v81_wrap_trace(result):
    """v8.1: the action_trace lines of an action result as _V81TraceLine records, with the host's per-action facts
    (action_trace_details: one dict per executed action), which are then dropped from the result."""
    if not isinstance(result, dict):
        return result
    facts = result.pop("action_trace_details", None)
    lines = result.get("action_trace")
    if not isinstance(lines, list):
        return result
    facts = facts if isinstance(facts, list) else []
    out = []
    for i, line in enumerate(lines):
        if not isinstance(line, str) or isinstance(line, _V81TraceLine):
            out.append(line)
            continue
        head, sep, verdict = line.partition(" -> ")
        pos, sep2, name = head.partition(": ")
        record = {"position": int(pos) if sep2 and pos.strip().isdigit() else i + 1,
                  "action": name if sep2 else head, "verdict": verdict}
        fact = facts[i] if i < len(facts) and isinstance(facts[i], dict) else None
        if facts:                                       # without the host's facts, executed is unknown: not stated
            record["executed"] = fact is not None
        if fact is not None:
            record.update(fact)
            record["result"] = {k: fact.get(k) for k in ("gameplay_changed", "level_completed", "game_over")}
        out.append(_V81TraceLine(line, record))
    result["action_trace"] = out
    return result


def _frame_from_payload(payload):
'''


def aliases(text):
    text = _rep(text, '\ndef _frame_from_payload(payload):\n', _ALIAS_CLASSES, 'def _frame_from_payload')
    text = _rep(text, '        runtime_globals["last_animation_timeline"] = LazyTimelineDict("timeline", {})\n',
                '        runtime_globals["last_animation_timeline"] = LazyTimelineDict("timeline", {})\n'
                '        runtime_globals["last_action_timeline"] = runtime_globals["last_animation_timeline"]   # v8.0 alias\n',
                'last_animation_timeline')
    return text


_RESERVE_FN = '''
_V81_RESERVE = []


def _v81_reserve():
    """v8.1: 32 MB of address space held during a call (calloc: no pages touched), released the moment a MemoryError leaves the
    snippet, so the report can be made (at the cap the unwinding otherwise stalls until the CPU kill)."""
    if not _V81_RESERVE:
        try:
            _V81_RESERVE.append(bytes(__RESERVE__))
        except MemoryError:
            pass


class ComputeLimitError(BaseException):
'''


def common(text):
    """v8.1, both arms: the MemoryError reserve, globals / locals / vars, copy / bisect, the action_trace records."""
    text = _rep(text, '\nclass ComputeLimitError(BaseException):\n', _RESERVE_FN.replace('__RESERVE__', str(RESERVE_BYTES)),
                'class ComputeLimitError')
    text = _rep(text, '    _set_limits(timeout_seconds)\n', '    _set_limits(timeout_seconds)\n    _v81_reserve()\n', '_set_limits call')
    text = _rep(text, '                exec(compile(_tree, "<python_tool>", "exec"), runtime_globals, runtime_globals)\n',
                '                try:\n'
                '                    exec(compile(_tree, "<python_tool>", "exec"), runtime_globals, runtime_globals)\n'
                '                except MemoryError:\n'
                '                    _V81_RESERVE.clear()        # v8.1: memory for the report, before any cleanup runs\n'
                '                    raise\n', 'exec')
    text = _rep(text, '                _value = eval(compile(_last_expr, "<python_tool>", "eval"), runtime_globals, runtime_globals)\n',
                '                try:\n'
                '                    _value = eval(compile(_last_expr, "<python_tool>", "eval"), runtime_globals, runtime_globals)\n'
                '                except MemoryError:\n'
                '                    _V81_RESERVE.clear()\n'
                '                    raise\n', 'eval')
    text = _rep(text, '\ndef _safe_import(name, globals=None, locals=None, fromlist=(), level=0):\n',
                '\nSAFE_BUILTINS = set(SAFE_BUILTINS) | {"globals", "locals", "vars", "ImportError", "ModuleNotFoundError"}   # v8.1: the model\'s own namespace; an import fallback can catch\n\n\n'
                'def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):\n', '_safe_import')
    text = _rep(text, '            runtime_globals.setdefault(_v70h_k, _v70h_m)\n',
                '            runtime_globals.setdefault(_v70h_k, _v70h_m)\n'
                '    import copy as _v81_copy, bisect as _v81_bisect          # v8.1: allowed modules the model uses without importing\n'
                '    runtime_globals.setdefault("copy", _v81_copy)\n'
                '    runtime_globals.setdefault("bisect", _v81_bisect)\n', 'v70h preload')
    text = _rep(text, '        action_result = reply.get("action_result") or {}\n',
                '        action_result = reply.get("action_result") or {}\n        _v81_wrap_trace(action_result)\n', 'action result')
    text = _rep(text, '        action_result.pop("action_echo", None)\n',
                '        action_result.pop("action_echo", None)\n        _v81_wrap_trace(action_result)\n', 'last_action_call_result')
    return text


# ------------------------------------------------------------------------------------------------- the stateful bootstrap (arm B)
_LOOP = '''
_V80 = {"ns": None, "provided": set(), "calls": 0, "limits": False, "cpu": 10, "fired": 0, "fn": "<python_tool>"}


def _v80_prof(signum, frame):
    import signal as _s
    if _V80["fired"] == 0:
        _V80["fired"] = 1
        _s.setitimer(_s.ITIMER_PROF, 5.0)
        raise ComputeLimitError("This call used its %d seconds of computation and was stopped here. It may "
                                "already have executed actions: inspect the current state before retrying." % _V80["cpu"])
    os._exit(__HARD_EXIT__)


def _v80_limits(timeout_seconds):
    import signal as _s
    if not _V80["limits"]:
        _V80["limits"] = True
        try:
            os.nice(10)
        except Exception:
            pass
        if resource is not None:
            for limit, value in ((getattr(resource, "RLIMIT_AS", None), __AS_BYTES__),
                                 (getattr(resource, "RLIMIT_FSIZE", None), 1_000_000),
                                 (getattr(resource, "RLIMIT_NOFILE", None), 32),
                                 (getattr(resource, "RLIMIT_CPU", None), __CPU_TOTAL__)):
                if limit is None:
                    continue
                try:
                    resource.setrlimit(limit, (value, value))
                except (OSError, ValueError):
                    pass
        _s.signal(_s.SIGPROF, _v80_prof)
    _V80["cpu"] = int(timeout_seconds)
    _V80["fired"] = 0
    _s.setitimer(_s.ITIMER_PROF, float(timeout_seconds))


def _v80_disarm():
    try:
        import signal as _s
        _s.setitimer(_s.ITIMER_PROF, 0)
    except Exception:
        pass


def _v80_release():
    _v80_disarm()
    _V81_RESERVE.clear()                                   # v3 / v8.1: the handler's memory (a MemoryError leaves none)


def _v80_proxy_action():
    def action(actions):                                   # IE F1: one object per worker; calls this call's action
        return _V80["action_impl"](actions)
    action.__qualname__ = "action"
    return action


_v80_action = _v80_proxy_action()


def main():
    while True:
        try:
            initial = _recv()
        except Exception:
            return
        if initial.get("type") == "shutdown":
            return
        try:
            _v80_call(initial)
        except BaseException as exc:
            _v80_release()
            try:
                _send({"type": "error", "error": "The Python kernel failed (%s); it is restarted at the next call."
                       % exc.__class__.__name__, "stdout": "", "action_results": [], "v80_fatal": True})
            except Exception:
                pass
            os._exit(3)
        _v80_disarm()
        _V80["calls"] += 1


if __name__ == "__main__":
    main()'''


def stateful(text):
    text = _rep(text, 'def main():\n    initial = _recv()\n', 'def _v80_call(initial):\n', 'def main')
    text = _rep(text, '    _set_limits(timeout_seconds)\n', '    _v80_limits(timeout_seconds)\n', '_set_limits call')
    text = _rep(text, '''    runtime_globals = {
        "__builtins__": {
            name: getattr(builtins, name)
            for name in SAFE_BUILTINS
        },
        "result": None,
    }
    runtime_globals["__builtins__"]["__import__"] = _safe_import
''', '''    if _V80["ns"] is None:                                  # v8.0 stateful kernel: one namespace per worker
        _V80["ns"] = {
            "__builtins__": {
                name: getattr(builtins, name)
                for name in SAFE_BUILTINS
            },
            "result": None,
        }
        _V80["ns"]["__builtins__"]["__import__"] = _safe_import
    runtime_globals = _V80["ns"]
    runtime_globals["__builtins__"] = {name: getattr(builtins, name) for name in SAFE_BUILTINS}   # v8.1 (IE): every call, so a
    runtime_globals["__builtins__"]["__import__"] = _safe_import                                 # cleared namespace stays restricted
    runtime_globals["result"] = None
    _v80_pre = {k: id(v) for k, v in runtime_globals.items()}
''', 'runtime_globals')
    text = _rep(text, '    _available_names = set(runtime_globals) | set(runtime_globals["__builtins__"])\n', '''    # v8.0: the names this call's setup provided (on a worker's first call, all of them); the library's dependencies are
    # judged against these only, never against variables from earlier calls
    _V80["provided"] |= ({k for k, v in runtime_globals.items() if _v80_pre.get(k) != id(v)} if _V80["calls"]
                         else set(runtime_globals))
    _available_names = (_V80["provided"] & set(runtime_globals)) | set(runtime_globals["__builtins__"])
''', '_available_names')
    # IE F1: the classes are hoisted to one object per worker (a kept class or tuple of them still matches what action raises and what
    # the handler below names); `action` and the except tuple are defined after this point, so both use the hoisted classes
    text = _rep(text, '''    runtime_globals["ActionCapError"] = ActionCapError
    runtime_globals["InvalidActionError"] = InvalidActionError
''', '''    runtime_globals["ActionCapError"] = ActionCapError
    runtime_globals["InvalidActionError"] = InvalidActionError
    _v80_c = _V80.setdefault("classes", {})                 # v8.0 (IE F1): one object per class per worker
    KnownNoOpActionError = _v80_c.setdefault("KnownNoOpActionError", KnownNoOpActionError)
    KnownDeathActionError = _v80_c.setdefault("KnownDeathActionError", KnownDeathActionError)
    StaleStateActionError = _v80_c.setdefault("StaleStateActionError", StaleStateActionError)
    RepeatedActionInStateError = _v80_c.setdefault("RepeatedActionInStateError", RepeatedActionInStateError)
    TerminalStateActionError = _v80_c.setdefault("TerminalStateActionError", TerminalStateActionError)
    ActionCapError = _v80_c.setdefault("ActionCapError", ActionCapError)
    InvalidActionError = _v80_c.setdefault("InvalidActionError", InvalidActionError)
    for _v80_k, _v80_v in _v80_c.items():
        runtime_globals[_v80_k] = _v80_v
''', 'the action classes')
    text = _rep(text, '    runtime_globals["action"] = action\n',
                '    _V80["action_impl"] = action                          # v8.0 (IE F1): the namespace holds one stable proxy, so a\n'
                '    runtime_globals["action"] = _v80_action               # kept `act = action` calls this call\'s action\n',
                'runtime_globals action')
    text = _rep(text, '''        _send(
            {
                "type": "final",''', '''        _v80_disarm()
        _send(
            {
                "type": "final",''', 'final send')
    text = _rep(text, "    ) as exc:\n        # The three refusals derive from BaseException so a model's own",
                "    ) as exc:\n        _v80_release()\n        # The three refusals derive from BaseException so a model's own",
                'except handler')
    # v8.1 (F6): each call compiled under its own filename, registered in linecache, so a traceback into a function defined in
    # an earlier call shows that call's source line (one shared "<python_tool>" entry showed the current call's line)
    text = _rep(text, '        try:\n            import linecache as _v62_linecache\n',
                '        _V80["fn"] = "<python_tool#%d>" % _V80["calls"]          # v8.1 (F6)\n'
                '        try:\n            import linecache as _v62_linecache\n'
                '            _v62_linecache.cache[_V80["fn"]] = (len(_code_text), None, _code_text.splitlines(True), _V80["fn"])\n'
                '            _v62_linecache.cache.pop("<python_tool#%d>" % (_V80["calls"] - 300), None)\n', 'linecache')
    text = _rep(text, 'compile(_tree, "<python_tool>", "exec")', 'compile(_tree, _V80["fn"], "exec")', 'compile exec')
    text = _rep(text, 'compile(_last_expr, "<python_tool>", "eval")', 'compile(_last_expr, _V80["fn"], "eval")', 'compile eval')
    text = _rep(text, "and value.__code__.co_filename == '<python_tool>'", 'and value.__code__.co_filename == _V80.get("fn", "<python_tool>")',
                'retention filename')
    text = _rep(text, 'frame.filename in ("<python_tool>", "<retained>")',
                '(frame.filename == "<retained>" or frame.filename.startswith("<python_tool"))', 'sanitizer frames')
    text = _rep(text, 'if frame.filename == "<python_tool>" and (frame.line or "").strip():',
                'if frame.filename.startswith("<python_tool") and (frame.line or "").strip():', 'sanitizer line')
    loop = (_LOOP.replace('__HARD_EXIT__', str(HARD_EXIT)).replace('__AS_BYTES__', str(AS_BYTES)).replace('__RESERVE__', str(RESERVE_BYTES))
            .replace('__CPU_TOTAL__', str(CPU_TOTAL_S)))
    text = _rep(text, '\nif __name__ == "__main__":\n    main()', loop, '__main__')
    compile(text, '<v80_stateful_bootstrap>', 'exec')
    return text


# ------------------------------------------------------------------------------------------------------------- the host side
class _Kernel:
    def __init__(self):
        self.dir = tempfile.mkdtemp(prefix='arc3_kernel_')
        self.proc = None
        try:
            self.proc = subprocess.Popen([sys.executable, '-I', '-S', '-c', STATEFUL_BOOTSTRAP], stdin=subprocess.PIPE,
                                         stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding='utf-8',
                                         cwd=self.dir, env=sb._sandbox_env(), start_new_session=True)
            self.q = queue.Queue()
            self.err = collections.deque(maxlen=64)
            self.busy = False
            self.last = time.monotonic()
            self.calls = 0
            threading.Thread(target=self._out, daemon=True).start()
            threading.Thread(target=self._errs, daemon=True).start()
        except BaseException:                               # IE F2: nothing is left behind by a start that failed
            if self.proc is not None:
                try:
                    os.killpg(self.proc.pid, signal.SIGKILL)
                except OSError:
                    pass
                try:
                    self.proc.wait(timeout=5)
                except Exception:
                    pass
            shutil.rmtree(self.dir, ignore_errors=True)
            raise
        STATS['started'] += 1

    def _out(self):
        try:
            for line in self.proc.stdout:
                self.q.put(line)
        except Exception:
            pass
        self.q.put(None)

    def _errs(self):
        try:
            for line in self.proc.stderr:
                self.err.append(line)
        except Exception:
            pass

    def alive(self):
        return self.proc.poll() is None

    def kill(self, why):
        STATS['killed_' + why] += 1
        try:
            os.killpg(self.proc.pid, signal.SIGKILL)
        except OSError:
            try:
                self.proc.kill()
            except OSError:
                pass
        try:
            self.proc.wait(timeout=5)
        except Exception:
            pass
        shutil.rmtree(self.dir, ignore_errors=True)


_KERNELS = {}          # (agent token, session token) -> _Kernel
_BY_AGENT = {}         # agent token -> its current key
_DIED = set()          # keys whose worker died / timed out / was evicted: the next call's output starts with a note
_LOCK = threading.Lock()


def _token(obj):
    try:
        t = obj.__dict__.get('_v80_token')
        if t is None:
            t = obj.__dict__['_v80_token'] = uuid.uuid4().hex
        return t
    except Exception:
        return 'id%x' % id(obj)


def _key():
    agent = getattr(sandbox_ext.CURRENT, 'agent', None)
    if agent is None:
        return None
    return (_token(agent), _token(getattr(agent, '_v6_session', None) or agent))


def _evict(now):
    for key, k in list(_KERNELS.items()):
        if not k.busy and (now - k.last > IDLE_S or not k.alive()):
            _KERNELS.pop(key).kill('idle' if k.alive() else 'dead')
            _DIED.add(key)
    idle = sorted((k.last, key) for key, k in _KERNELS.items() if not k.busy)
    while len(_KERNELS) > KERNEL_CAP and idle:
        _, key = idle.pop(0)
        _KERNELS.pop(key).kill('cap')
        _DIED.add(key)


def _acquire(key):
    with _LOCK:
        now = time.monotonic()
        prev = _BY_AGENT.get(key[0])
        if prev is not None and prev != key and prev in _KERNELS:    # a new conversation of this game: its old worker goes
            _KERNELS.pop(prev).kill('conversation')
        _BY_AGENT[key[0]] = key
        k = _KERNELS.get(key)
        if k is not None and not k.alive():
            _KERNELS.pop(key).kill('dead')
            _DIED.add(key)
            k = None
        note = ''
        if k is None:
            k = _KERNELS[key] = _Kernel()
            if key in _DIED:
                _DIED.discard(key)
                note = TEXTS_KERNEL_RESTARTED
        k.busy = True
        k.last = now
        _evict(now)
        return k, note


def _rss_mb(pid):
    try:
        with open('/proc/%d/statm' % pid) as fh:
            return int(fh.read().split()[1]) * _PAGE / 1048576.0
    except (OSError, ValueError, IndexError):
        return 0.0


def _cpu_s(pid):
    try:
        with open('/proc/%d/stat' % pid) as fh:
            f = fh.read()
        f = f[f.rindex(')') + 2:].split()
        return (int(f[11]) + int(f[12])) / _TICK                # utime + stime
    except (OSError, ValueError, IndexError):
        return None


def _release(key, k, dead):
    why = 'died'
    if not dead:
        rss = _rss_mb(k.proc.pid)
        if rss > STATS['rss_max_mb']:
            STATS['rss_max_mb'] = int(rss)
        if RSS_MB > 0 and rss > RSS_MB:                    # IE F3: a persistent worker keeps what the model keeps
            dead, why = True, 'memory'
    with _LOCK:
        k.busy = False
        k.last = time.monotonic()
        k.calls += 1
        if dead:
            if _KERNELS.get(key) is k:
                _KERNELS.pop(key)
            k.kill(why)
            _DIED.add(key)
        STATS['released'] += 1
        n, live = STATS['released'], len(_KERNELS)
    if n % LOG_EVERY == 0:
        log.warning('v80 kernel stats after %d calls (%d live workers): %s', n, live, dict(sorted(STATS.items())))


def run_stateful(*, code, timeout_seconds, initial_state, action_handler, animation_handler=None, kept_functions=None,
                 retain_imports=False, repair_hints=False, **kwargs):
    key = _key()
    if key is None:                                   # no game context (tests, tools): the per-call sandbox
        STATS['per_call_fallback'] += 1
        return _PER_CALL(code=code, timeout_seconds=timeout_seconds, initial_state=initial_state, action_handler=action_handler,
                         animation_handler=animation_handler, kept_functions=kept_functions, retain_imports=retain_imports,
                         repair_hints=repair_hints, **kwargs)
    try:
        k, note = _acquire(key)
    except (OSError, RuntimeError, subprocess.SubprocessError):     # IE F2: no worker (pids, memory, disk): this call per-call
        STATS['start_failed'] += 1
        log.warning('v80 kernel: a worker did not start; this call runs in the per-call sandbox', exc_info=True)
        return _PER_CALL(code=code, timeout_seconds=timeout_seconds, initial_state=initial_state, action_handler=action_handler,
                         animation_handler=animation_handler, kept_functions=kept_functions, retain_imports=retain_imports,
                         repair_hints=repair_hints, **kwargs)
    dead = False
    host_action_results = []

    def done(res):
        if note:
            res['stdout'] = note + '\n' + (res.get('stdout') or '')
        return res
    try:
        try:
            sb._send_json_line(k.proc.stdin, {
                "code": code, "kept_functions": list(kept_functions or []), "persistence_enabled": kept_functions is not None,
                "retain_imports": retain_imports, "repair_hints": repair_hints, "animation_enabled": animation_handler is not None,
                "timeout_seconds": timeout_seconds, "sandbox_cwd": k.dir, "state": initial_state,
                "color_chars": ARC_COLOR_CHARS})
        except (OSError, ValueError):
            dead = True
            return done({"error": "Sandbox process could not start.", "stdout": "", "action_results": []})
        deadline = time.monotonic() + max(1, int(timeout_seconds))
        _cpu = (initial_state or {}).get("v62_cpu_seconds")          # the worker's own rule for its CPU limit
        cpu_cap = max(1, int(_cpu)) if _cpu else max(1, int(timeout_seconds))
        cpu0 = _cpu_s(k.proc.pid)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                dead = True
                STATS['wall_timeout'] += 1
                return done({"error": "Tool timed out after %ss" % timeout_seconds, "stdout": "",
                             "action_results": list(host_action_results)})
            if cpu0 is not None:                                     # v3: the hard CPU bound, enforced from outside
                used = _cpu_s(k.proc.pid)
                if used is not None and used - cpu0 > cpu_cap + CPU_GRACE_S:
                    dead = True
                    STATS['cpu_host_kill'] += 1
                    return done({"error": sandbox_ext.CPU_KILLED, "stdout": "", "action_results": list(host_action_results)})
            try:
                line = k.q.get(timeout=min(remaining, 1.0))
            except queue.Empty:
                continue
            if line is None:                              # the worker exited
                dead = True
                try:
                    k.proc.wait(timeout=5)
                except Exception:
                    pass
                rc = k.proc.returncode
                if rc == HARD_EXIT:
                    STATS['cpu_hard_kill'] += 1
                    err = sandbox_ext.CPU_KILLED
                else:
                    STATS['died_rc_%s' % rc] += 1
                    log.warning('stateful kernel died (rc %s); stderr tail:\n%s', rc, ''.join(k.err)[-4000:])
                    err = "Sandbox process exited unexpectedly."
                return done({"error": err, "stdout": "", "action_results": list(host_action_results)})
            try:
                message = json.loads(line)
            except json.JSONDecodeError:
                dead = True
                return done({"error": "Sandbox process returned an invalid response.", "stdout": "",
                             "action_results": list(host_action_results)})
            kind = str(message.get("type", "")).strip()
            if kind == "animation":
                if animation_handler is None:
                    sb._send_json_line(k.proc.stdin, {"type": "animation_error",
                                                      "error": "Animation frames are not available in this session."})
                    continue
                try:
                    payload = animation_handler(dict(message.get("request") or {}))
                except Exception:  # noqa: BLE001
                    sb._send_json_line(k.proc.stdin, {"type": "animation_error", "error": "animation failed in sandbox host."})
                    continue
                sb._send_json_line(k.proc.stdin, {"type": "animation_result", "animation": payload})
                continue
            if kind == "action":
                try:
                    result = action_handler(list(message.get("actions") or []), stale_after=message.get("stale_after"))
                except Exception:  # noqa: BLE001
                    sb._send_json_line(k.proc.stdin, {"type": "action_error", "error": "action failed in sandbox host."})
                    continue
                raw = result.get("action_result") or {}
                if isinstance(raw, dict):
                    host_action_results.append(dict(raw))
                sb._send_json_line(k.proc.stdin, {"type": "action_result", "action_result": raw, "state": result.get("state") or {}})
                continue
            if kind in ("final", "error"):
                if message.get("v80_fatal"):
                    dead = True
                    STATS['fatal'] += 1
                STATS['calls'] += 1
                out = {
                    "stdout": str(message.get("stdout", "") or ""),
                    "result": message.get("result"),
                    "error": str(message.get("error", "") or ""),
                    "action_results": list(message.get("action_results") or host_action_results),
                    "keepable_functions": list(message.get("keepable_functions") or []),
                }
                if kept_functions is not None:
                    out["retention_rejected"] = message.get("retention_rejected", [])
                return done(out)
            dead = True
            return done({"error": "Sandbox process returned an unknown message type.", "stdout": "",
                         "action_results": list(host_action_results)})
    except (OSError, ValueError, BrokenPipeError):
        dead = True
        return done({"error": "Sandbox process exited unexpectedly.", "stdout": "", "action_results": list(host_action_results)})
    finally:
        _release(key, k, dead)


def shutdown():
    with _LOCK:
        for key, k in list(_KERNELS.items()):
            _KERNELS.pop(key).kill('shutdown')
    if STATS['released']:
        try:
            log.warning('v80 kernel stats at shutdown: %s', dict(sorted(STATS.items())))
        except Exception:
            pass


TEXTS_KERNEL_RESTARTED = ("[The Python kernel was restarted, so variables from earlier calls are gone; your library programs are "
                          "loaded.]")
SYSTEM_OLD = ("- Python variables reset between calls, but your functions persist, and so do top-level constants named in UPPER_CASE "
              "whose values are literals (numbers, strings, or lists, tuples, sets and dicts of them).")
SYSTEM_NEW = ("- Python variables persist between calls within this conversation, as in a notebook: a name you assign stays defined "
              "in later calls (a new conversation starts with none). The runtime globals (`current_frame`, `history`, `valid_actions` "
              "and the others) are refreshed at the start of every call; values you computed from an earlier board are not. Your "
              "functions persist for the whole game, and so do top-level constants named in UPPER_CASE whose values are literals "
              "(numbers, strings, or lists, tuples, sets and dicts of them).")
NAMEERROR_NEW = ("This name is not defined in this conversation's Python kernel: variables persist only from earlier calls of this "
                 "conversation (a new conversation or a kernel restart starts empty); programs in your library and UPPER_CASE "
                 "constants bound to literals persist for the whole game.")


TRACE_KEYS = ("interior_changed", "gameplay_changed", "level_completed", "game_over", "board_changed", "reverted")


def _trace_details(session, payload):
    """v8.1: the per-action facts of this action(...) call, one dict per executed action, from the session's v61_call_payloads
    (step_env extends it with this call's executed payloads, in order, before it returns)."""
    n = int(payload.get("executed_count") or 0)
    recs = (getattr(session, "__dict__", {}) or {}).get("v61_call_payloads") or []
    if n <= 0 or len(recs) < n:
        return None
    return [{k: r[k] for k in TRACE_KEYS if k in r} for r in recs[-n:]]


_TRACE_SIDE = {}        # id(step_env's payload) -> its per-action facts, until the agent compacts that payload


def _after_step(session, payload):
    """step_env's payload in hand: its per-action facts into the side table (the payload itself unchanged)."""
    try:
        if isinstance(payload, dict) and isinstance(payload.get("action_trace"), list):
            details = _trace_details(session, payload)
            if details is not None:
                if len(_TRACE_SIDE) > 256:              # never in normal use (one entry per lane at most)
                    _TRACE_SIDE.clear()
                _TRACE_SIDE[id(payload)] = (list(payload["action_trace"]), details)
    except Exception:  # noqa: BLE001
        STATS['trace_details_failed'] += 1


def _after_compact(payload, compact):
    """The agent's compact result for that payload: the facts as action_trace_details (the sandbox wraps and drops them)."""
    try:
        entry = _TRACE_SIDE.pop(id(payload), None)
        if entry is not None and isinstance(compact, dict) and compact.get("action_trace"):
            if list(compact["action_trace"]) == entry[0]:   # IE: an id reused by another payload never carries its facts
                compact["action_trace_details"] = entry[1]
            else:
                STATS['trace_details_stale'] += 1
    except Exception:  # noqa: BLE001
        STATS['trace_details_failed'] += 1
    return compact


def _wrap_host():
    """v8.1, both arms: step_env computes the per-action facts beside its payload (a side table keyed by the payload object, so the
    payload itself, which the prolong logs record, is unchanged); the agent's compact result for that payload carries them to the
    sandbox as action_trace_details."""
    from inference.framework import solver as sv
    from v62.agent import V6Agent
    cls = getattr(sv, '_HarnessGameSession', None)
    step0 = getattr(cls, 'step_env', None)
    if step0 is not None and not getattr(step0, '_v81', False):
        def step_env(self, arguments):
            payload = step0(self, arguments)
            _after_step(self, payload)
            return payload
        step_env._v81 = True
        cls.step_env = step_env
    comp0 = V6Agent._compact_action_result
    if not getattr(comp0, '_v81', False):
        def _compact_action_result(self, payload):
            return _after_compact(payload, comp0(self, payload))
        _compact_action_result._v81 = True
        V6Agent._compact_action_result = _compact_action_result


def _pids():
    """The cgroup's pids limit and use (v2, then v1), or (None, None)."""
    for d in ('/sys/fs/cgroup', '/sys/fs/cgroup/pids'):
        try:
            with open(os.path.join(d, 'pids.max')) as fh:
                mx = fh.read().strip()
            with open(os.path.join(d, 'pids.current')) as fh:
                return mx, int(fh.read().strip())
        except (OSError, ValueError):
            continue
    return None, None


def install():
    """Run once, after lazy_history.install() (run_v6.py)."""
    global STATEFUL_BOOTSTRAP, _PER_CALL, KERNEL_CAP
    if _INSTALLED:
        return
    _INSTALLED.append(1)
    from v62 import agent as A
    sb._SANDBOX_BOOTSTRAP = common(aliases(sb._SANDBOX_BOOTSTRAP))
    _wrap_host()
    assert len(sb._SANDBOX_BOOTSTRAP.encode('utf-8')) < ARGV_MAX, \
        'kernel_ext: the per-call bootstrap is %d bytes (one argv string)' % len(sb._SANDBOX_BOOTSTRAP.encode('utf-8'))
    sandbox_ext.WALL_BACKSTOP = WALL_S
    assert A.TEXTS['timeout_wall'].count('120 seconds') == 1, 'kernel_ext: timeout_wall anchor'
    A.TEXTS['timeout_wall'] = A.TEXTS['timeout_wall'].replace('120 seconds', '%d seconds' % WALL_S)
    if not STATEFUL:
        return
    STATEFUL_BOOTSTRAP = stateful(sb._SANDBOX_BOOTSTRAP)
    assert len(STATEFUL_BOOTSTRAP.encode('utf-8')) < ARGV_MAX, \
        'kernel_ext: the stateful bootstrap is %d bytes (one argv string)' % len(STATEFUL_BOOTSTRAP.encode('utf-8'))
    assert A.TEXTS['system'].count(SYSTEM_OLD) == 1, 'kernel_ext: system-prompt anchor'
    A.TEXTS['system'] = A.TEXTS['system'].replace(SYSTEM_OLD, SYSTEM_NEW)
    A.TEXTS['nameerror_note'] = NAMEERROR_NEW
    A.TEXTS['v80_kernel_restarted'] = TEXTS_KERNEL_RESTARTED
    mx, cur = _pids()                     # v8.1 (IE F3): each worker is a process plus two host threads
    if mx is not None and mx != 'max' and cur is not None:
        headroom = int(mx) - cur
        if headroom < 20 * KERNEL_CAP:
            KERNEL_CAP = max(4, headroom // 20)
    log.warning('v8.1 kernel: pids max %s, current %s; worker cap %d, idle %d s, RSS bound %d MB', mx, cur, KERNEL_CAP, IDLE_S,
                RSS_MB)
    _PER_CALL = sandbox_ext._original_run
    sandbox_ext._original_run = run_stateful
    atexit.register(shutdown)
