"""Harness v6 sandbox extensions (spec §5.2, §5.4, §5.5; BUILD_PLAN 5-7; v6.1: preloaded modules, start frames, library levels), applied to dfranzen's sandbox at import:
  - seeds grid.py and objects.py (v6/seeds, the control's helpers; they accept frames with .ascii and frame dicts),
    executed into every call's namespace before the library is restored (a library function of the same name wins);
  - programs(query=None) and read_code(name) over this game's library (name, signature, doc, step added, source);
  - history[i].animation_frames: the frames action i passed through, fetched lazily from the host's animation side
    store (empty when the action did not animate), as views like current_frame.
Host side: run_sandboxed_python is wrapped to add initial_state['v6'] (library + seeds) and to answer
'history_frames' animation requests from the calling agent's game session (thread-local current agent).
"""
import os
import signal
import subprocess
import threading
import time
from pathlib import Path

from inference.agent import python_tool_sandbox as sb
from inference.agent import tool_agent as ta
from inference.utils.animation import build_animation_frames

# round 5m (item 124, user 10-08): show_board(cells, label) in the sandbox (ARC3_V62_SHOW_BOARD=0 turns it off)
SHOW_BOARD = os.environ.get('ARC3_V62_SHOW_BOARD', '1') == '1'
# v6.6 item C (IE-agreed): tools.py joins the seeds when ARC3_V66_SEEDS=1 (SEED1, user 10-17: on by default in v6.6)
SEED_FILES = ('grid.py', 'objects.py') + (('tools.py',) if os.environ.get('ARC3_V66_SEEDS', '1') == '1' else ())   # SEED1
# v7.0 harness revision (apply_v70h.py): item 14's history tools join the seeds (they use tools.py's board_key); item 13's
# preloaded names (json, random, Counter, deque, defaultdict) are set in each call's namespace (_INSERT, from initial_state)
V70H_ACCESSORS = os.environ.get('ARC3_V70H_ACCESSORS', '0') == '1'
V70H_ALIASES = os.environ.get('ARC3_V70H_ALIASES', '0') == '1'
if V70H_ACCESSORS:
    if 'tools.py' not in SEED_FILES:
        raise RuntimeError('ARC3_V70H_ACCESSORS=1 needs the tools.py seeds (ARC3_V66_SEEDS=1)')
    SEED_FILES = SEED_FILES + ('history70.py',)
SEEDS = '\n\n'.join((Path(__file__).with_name('seeds') / n).read_text() for n in SEED_FILES)
V75_POLISH = os.environ.get('ARC3_V75_POLISH', '0') == '1'   # v7.5 POLISH (agent.V75_POLISH; variants.v75_polish checks the value)
V77 = os.environ.get('ARC3_V77', '0') == '1'   # v7.7 (apply_v77.py; variants.v77 checks the value): summaries(query), frame_diff


def _v75_sub(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError('v7.5: anchor not found exactly once: %r' % old[:70])
    return text.replace(old, new)


# v7.5 TOOL-4 (the user 10-23, review Part 4; q74: 77 and 79 refusals per run, 64% of them within 40 x 40): crop refuses windows over
# 1,600 cells (40 x 40, or strips such as 25 x 64) instead of over 32 x 32; its error ends with a period (TXT-10)
V75_CROP = (
    ('refuses windows over 32 x 32 (use show_board).',
     'refuses windows over 1,600 cells (40 x 40, or strips such as 25 x 64; use show_board).'),
    ('    if r1 - r0 + 1 > 32 or c1 - c0 + 1 > 32:\n'
     '        raise ValueError("window %dx%d is larger than 32 x 32: crop a smaller window, or call show_board for a whole board"\n'
     '                         % (r1 - r0 + 1, c1 - c0 + 1))\n',
     '    if (r1 - r0 + 1) * (c1 - c0 + 1) > 1600:\n'
     '        raise ValueError("window %dx%d has %d cells, over the 1,600-cell limit (40 x 40, or strips such as 25 x 64): crop a "\n'
     '                         "smaller window, or call show_board for a whole board." % (r1 - r0 + 1, c1 - c0 + 1,\n'
     '                                                                                     (r1 - r0 + 1) * (c1 - c0 + 1)))\n'),
)
if V75_POLISH and 'tools.py' in SEED_FILES:
    for _v75_old, _v75_new in V75_CROP:
        SEEDS = _v75_sub(SEEDS, _v75_old, _v75_new)
CURRENT = threading.local()

_ANCHOR = '        _refresh_state(initial.get("state") or {})\n'
_INSERT = '''        _v6 = (initial.get("state") or {}).get("v6") or {}
        import collections as _v61_collections, itertools as _v61_itertools, math as _v61_math
        import re as _v61_re, heapq as _v61_heapq
        for _v61_k, _v61_m in (("collections", _v61_collections), ("itertools", _v61_itertools), ("math", _v61_math),
                               ("re", _v61_re), ("heapq", _v61_heapq)):
            runtime_globals.setdefault(_v61_k, _v61_m)       # v6.1 item 8: preloaded modules
        if _v6.get("v70h_aliases"):                            # v7.0 item 13: the names model code most often used unimported
            import json as _v70h_json, random as _v70h_random
            for _v70h_k, _v70h_m in (("json", _v70h_json), ("random", _v70h_random),
                                     ("Counter", _v61_collections.Counter), ("deque", _v61_collections.deque),
                                     ("defaultdict", _v61_collections.defaultdict)):
                runtime_globals.setdefault(_v70h_k, _v70h_m)

        def _v6_history_frames(index):
            _send({"type": "animation", "request": {"kind": "history_frames", "index": int(index)}})
            reply = _recv()
            if reply.get("type") != "animation_result":
                return []
            return [FrameView(ascii=i.get("ascii", ""), step=i.get("step", 0), level=i.get("level", 0),
                              shape=i.get("shape", (0, 0)), grid=i.get("grid", ()))
                    for i in (reply.get("animation") or [])]

        def _v6_animation_frames(self):
            cached = self.__dict__.get("_v6_frames")
            if cached is None:
                idx = self.__dict__.get("_v6_index")
                cached = _v6_history_frames(idx) if (idx is not None and initial.get("animation_enabled")) else []
                self.__dict__["_v6_frames"] = cached
            return cached

        HistoryEntryView.animation_frames = property(_v6_animation_frames)
        _v6_refresh_original = _refresh_state

        class _V61FrameRef:
            """A history frame by index, resolved on first attribute access (the lazy history store fetches old
            frames over the host channel; resolving at every refresh would fetch after each action)."""

            def __init__(self, index):
                self.__dict__["_v61_i"] = int(index)
                self.__dict__["_v61_f"] = None

            def _v61_frame(self):
                if self.__dict__["_v61_f"] is None:
                    self.__dict__["_v61_f"] = (runtime_globals.get("history") or [])[self.__dict__["_v61_i"]].frame
                return self.__dict__["_v61_f"]

            def __getattr__(self, name):
                return getattr(self._v61_frame(), name)

            def __repr__(self):
                return repr(self._v61_frame())

            __str__ = __repr__

        def _refresh_state(payload):
            _v6_refresh_original(payload)
            _v61_hist = runtime_globals.get("history") or []
            _v61_lvl = _v61_att = 0
            if getattr(_v61_hist, "_v6_paged", False):    # round 5k: indices set by the payload, starts from the host
                _v61_lvl, _v61_att = _v61_hist._v6_level_start, _v61_hist._v6_attempt_start
            else:
                for _v6_i, _v6_e in enumerate(_v61_hist):
                    try:
                        _v6_e.__dict__["_v6_index"] = _v6_e.__dict__.get("_v6_ref", _v6_i)   # the host index (run B hides entries)
                    except Exception:
                        pass
                    _v61_r = getattr(_v6_e, "result", None) or {}
                    if _v61_r.get("level_completed"):
                        _v61_lvl = _v61_att = _v6_i
                    elif _v61_r.get("action_name") == "RESET" or _v61_r.get("automatic"):   # round 5: no inferred restarts
                        _v61_att = _v6_i
            if _v61_hist:                                     # v6.1 item 2g: start frames, fetched on first use
                runtime_globals["level_start_frame"] = _V61FrameRef(_v61_lvl)
                runtime_globals["attempt_start_frame"] = _V61FrameRef(max(_v61_lvl, _v61_att))

        _v6_library = list(_v6.get("library") or [])
        _v61_archive = list(_v6.get("archive") or [])

        def programs(query=None, archived=False):
            """This game's program library: name, signature, doc, the level it was last written on, step added (query
            filters by name or doc). archived=True lists programs set aside when a level was started again; they are
            not loaded, but read_code(name) shows them and you can redefine any of them."""
            q = str(query or "").lower()
            return [dict(name=e.get("name"), signature=e.get("signature"), doc=e.get("doc"), level=e.get("level"),
                         step_added=e.get("step"))
                    for e in (_v61_archive if archived else _v6_library)
                    if not q or q in str(e.get("name", "")).lower() or q in str(e.get("doc", "")).lower()]

        def read_code(name):
            """The source of a library function."""
            for e in _v6_library + _v61_archive:
                if e.get("name") == name:
                    return e.get("source")
            raise KeyError("no program named %r; programs() lists the library" % (name,))

        def save_summary(text):
            """Save a summary when the harness asks for one (at a level completion, and before a level restarts); prints
            what was saved."""
            if not isinstance(text, str):
                raise TypeError("save_summary(text) takes the summary as one string (str), not %s"
                                % type(text).__name__)
            _send({"type": "animation", "request": {"kind": "save_summary", "text": text}})
            reply = _recv()
            if reply.get("type") != "animation_result":
                raise RuntimeError(str(reply.get("error") or "the summary was not saved"))
            print((reply.get("animation") or {}).get("note") or "")

        _v67x_sums = _v6.get("summaries")

        def summaries():
            """Your saved level summaries as {level: text}: a copy of those saved before this call (a summary saved in
            this call appears from the next call on)."""
            _send({"type": "animation", "request": {"kind": "summaries", "n": len(_v67x_sums or [])}})
            _recv()                                           # v6.7x SUMLOOKUP: the harness counts the call
            return {int(level): str(text) for level, text in (_v67x_sums or [])}

        def _v62_symbol(value):
            if isinstance(value, str):
                if len(value) != 1:
                    raise ValueError
                return value
            index = int(value)
            if not 0 <= index < len(COLOR_CHARS):
                raise ValueError
            return COLOR_CHARS[index]

        def show_board(cells, label=None):
            """Add a board (a frame, rows of symbols or ASCII text, as grid_cells accepts) to this call's result as an
            image captioned with `label`; at most 4 per call, and a board already shown in the call is not repeated.
            Prints the image's caption."""
            source = cells
            if hasattr(source, "ascii"):
                source = source.ascii
            elif isinstance(source, dict) and "ascii" in source:
                source = source["ascii"]
            if isinstance(source, str):
                source = [] if source == "(empty grid)" else source.splitlines()
            try:
                rows = [row if isinstance(row, str) else "".join(_v62_symbol(v) for v in row) for row in source]
            except (TypeError, ValueError, IndexError):
                raise ValueError("show_board(cells) takes a frame, rows of symbols or ASCII text") from None
            if (not rows or len(rows) > 64 or len(set(len(row) for row in rows)) != 1 or not 0 < len(rows[0]) <= 64
                    or any(ch not in COLOR_CHARS for row in rows for ch in row)):
                raise ValueError("show_board(cells) takes rows of equal length, at most 64 x 64, of the board's colour "
                                 "symbols")
            _send({"type": "animation", "request": {"kind": "show_board", "rows": rows,
                                                    "label": None if label is None else str(label)[:80]}})
            reply = _recv()
            if reply.get("type") != "animation_result":
                raise RuntimeError(str(reply.get("error") or "the board was not shown"))
            print((reply.get("animation") or {}).get("note") or "")

        if _v6:
            runtime_globals["programs"] = programs
            runtime_globals["read_code"] = read_code
            runtime_globals["save_summary"] = save_summary            # round 5l (item 108)
            if _v6.get("summaries") is not None:
                runtime_globals["summaries"] = summaries              # v6.7x SUMLOOKUP (user decision 53)
            if _v6.get("show_board"):
                runtime_globals["show_board"] = show_board            # round 5m (item 124)
            if _v6.get("seeds"):
                import time as _v66_time
                runtime_globals["_tool_clock"] = _v66_time.monotonic   # v6.6 C: the seeds' bfs budget (imports are restricted)
                exec(compile(_v6["seeds"], "<seeds>", "exec"), runtime_globals, runtime_globals)
'''
if V75_POLISH:                                   # v7.5 TXT-6 (the user 10-23): save_summary's docstring says what the texts say
    _INSERT = _v75_sub(_INSERT, 'and before a level restarts); prints\n            what was saved.',
                       'and before the harness\n            starts a new conversation); prints what was saved.')
# v7.7 (C, the user 10-24): summaries(query) returns only the saved summaries' lines that contain the query text (ignoring case) as
# {level: [lines]}; summaries() is unchanged. The docstring is set after the definition (no nested triple quotes in _INSERT).
V77_SUM_OLD = '''        def summaries():
            """Your saved level summaries as {level: text}: a copy of those saved before this call (a summary saved in
            this call appears from the next call on)."""
            _send({"type": "animation", "request": {"kind": "summaries", "n": len(_v67x_sums or [])}})
            _recv()                                           # v6.7x SUMLOOKUP: the harness counts the call
            return {int(level): str(text) for level, text in (_v67x_sums or [])}
'''
V77_SUM_NEW = '''        def summaries(query=None):
            _send({"type": "animation", "request": {"kind": "summaries", "n": len(_v67x_sums or []),
                                                    "query": query is not None}})
            _recv()                                           # v6.7x SUMLOOKUP: the harness counts the call
            out = {int(level): str(text) for level, text in (_v67x_sums or [])}
            if query is None:
                return out
            q = str(query).lower()
            hits = {level: [line for line in text.split("\\n") if q in line.lower()] for level, text in out.items()}
            return {level: lines for level, lines in hits.items() if lines}

        summaries.__doc__ = ("Your saved level summaries as {level: text}: a copy of those saved before this call (a summary "
                             "saved in this call appears from the next call on). summaries(query) returns only their lines that "
                             "contain the query text, ignoring case, as {level: [lines]}.")
'''


def _v77_sub(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError('v7.7: anchor not found exactly once: %r' % old[:70])
    return text.replace(old, new)


if V77:
    _INSERT = _v77_sub(_INSERT, V77_SUM_OLD, V77_SUM_NEW)
_INSERT = '\n'.join(l[4:] if l.startswith('        ') else l for l in _INSERT.split('\n'))
_ANCHOR_D = _ANCHOR[4:]
if _ANCHOR_D not in sb._SANDBOX_BOOTSTRAP:
    raise RuntimeError('sandbox bootstrap changed: dedented anchor not found')
sb._SANDBOX_BOOTSTRAP = sb._SANDBOX_BOOTSTRAP.replace(_ANCHOR_D, _INSERT + _ANCHOR_D, 1)
if V77:                                           # v7.7 (2): frame_diff pairs exact in-place recolours first, in the host module and
    from inference.utils import frame_diff as _v77_frame_diff          # in the sandbox bootstrap's inlined copy
    _v77_frame_diff.INPLACE_FIRST = True
    sb._SANDBOX_BOOTSTRAP = _v77_sub(sb._SANDBOX_BOOTSTRAP, '\nINPLACE_FIRST = False\n', '\nINPLACE_FIRST = True\n')

_original_run = ta.run_sandboxed_python


WALL_BACKSTOP = 120     # v6.2: seconds of wall time before the host stops a call (its compute is bounded by CPU time)


def _run_sandboxed_python(*, code, timeout_seconds, initial_state, action_handler, animation_handler=None, **kwargs):
    agent = getattr(CURRENT, 'agent', None)
    initial_state = dict(initial_state or {})
    initial_state['v62_cpu_seconds'] = int(timeout_seconds)
    _EXIT.cpu_seconds, _EXIT.started, _EXIT.returncode = int(timeout_seconds), time.monotonic(), None   # item 123
    timeout_seconds = max(int(timeout_seconds), WALL_BACKSTOP)
    if agent is not None:
        initial_state = dict(initial_state or {})
        initial_state['v6'] = dict(library=agent._v6_library_payload(), archive=agent._v61_archive_payload(),
                                   seeds=SEEDS, show_board=SHOW_BOARD, v70h_aliases=V70H_ALIASES)
        sums = agent._v67x_summaries_payload() if hasattr(agent, '_v67x_summaries_payload') else None
        if sums is not None:                                   # v6.7x SUMLOOKUP: the saved summaries at this call's start
            initial_state['v6']['summaries'] = sums
        inner = animation_handler

        def handler(request):
            if str(request.get('kind') or '') == 'history_frames':
                return agent._v6_history_frames(int(request.get('index', -1)))
            if str(request.get('kind') or '') == 'save_summary':          # round 5l (item 108)
                return agent._v62_save_summary(request.get('text'))
            if str(request.get('kind') or '') == 'summaries' and hasattr(agent, '_v67x_summaries_call'):
                return agent._v67x_summaries_call(request)               # v6.7x SUMLOOKUP: the call's receipt
            if str(request.get('kind') or '') == 'show_board' and SHOW_BOARD:   # round 5m (item 124)
                return agent._v62_show_board(request)
            return inner(request) if inner is not None else {}
        animation_handler = handler
    return _original_run(code=code, timeout_seconds=timeout_seconds, initial_state=initial_state,
                         action_handler=action_handler, animation_handler=animation_handler, **kwargs)


ta.run_sandboxed_python = _run_sandboxed_python

# round 5m (item 123; run 396: 22 calls killed at the hard CPU limit read "Sandbox process exited unexpectedly."): a
# sandbox process that died on its own by SIGKILL or SIGXCPU (the kernel's CPU limit; patches.CPU_MARGIN s after the
# stated limit) after at least the stated limit of wall time, or whose stderr shows the limit's error, is reported as the
# compute limit (agent: TEXTS['compute_limit_lost']). The exit status is recorded only when the child exited by itself;
# a child the host kills (wall-clock backstop, invalid response) takes other branches and is not reported here.
CPU_KILLED = 'Sandbox process stopped at its CPU limit.'
_EXIT = threading.local()
_original_wait = sb._wait_for_process_exit
_original_sanitize = sb._sanitize_host_error_text


def _wait_for_process_exit(process, *, timeout=1.0):
    _EXIT.returncode = None
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        return _original_wait(process, timeout=timeout)       # still running: the original kills it (not recorded)
    except OSError:
        return None
    _EXIT.returncode = process.returncode
    return None


def _sanitize_host_error_text(text):
    rc, _EXIT.returncode = getattr(_EXIT, 'returncode', None), None
    message = _original_sanitize(text)                         # logs the child's stderr, as before
    elapsed = time.monotonic() - (getattr(_EXIT, 'started', None) or time.monotonic())
    if ((rc in (-signal.SIGKILL, -signal.SIGXCPU) and elapsed >= (getattr(_EXIT, 'cpu_seconds', None) or 10))
            or 'ComputeLimitError' in str(text or '')):
        return CPU_KILLED
    return message


sb._wait_for_process_exit = _wait_for_process_exit
sb._sanitize_host_error_text = _sanitize_host_error_text


def host_history_frames(session, index):
    """Frames of history entry `index` from the game session's animation side store (v6.patches)."""
    from v62.patches import unpack
    record = (getattr(session, 'v6_anim', None) or {}).get(index)
    if not record:
        return []
    return build_animation_frames(dict(record, chain=[unpack(g) for g in record['chain']]))
