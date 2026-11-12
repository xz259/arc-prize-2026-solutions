"""Lazy history store (spec §5.4; overnight plan §4.4), behind ARC3_V6_LAZY_HISTORY=1 (default off: not used in the
2026-10-03 screens, so their arms stay comparable).

dfranzen's runtime state costs O(actions) per action and per Python call:
  - after every executed action the solver rewrites tool_runtime_state.json with every history frame
    (indent=2: ~63 KB per entry; 2,000 entries = 126 MB, ~2.5 s to write);
  - every call (and every action inside a call) reloads that file and sends every frame to the sandbox as ascii + grid
    (~20 KB per entry), where views are built for all of them.
Over a 9-hour game (thousands of actions) this alone exceeds the 10-s sandbox limit.

Here:
  host     the runtime state lives in memory (a registry keyed by the state path, holding the session's own history
           list); a one-frame stub file keeps the path's existence checks and artifact globbing working.
  payload  history entries carry action, result, a frame reference (the host index) and the frame's step / level /
           shape; only the last EAGER_TAIL entries carry their frame.
  sandbox  history[i].frame is fetched on first access over the existing host channel (kind 'history_frame_window':
           entries i-7 .. i+8, cached for the call, so forward and backward passes both batch), transitions read
           frames through history lazily, and
           str()/repr() use the metadata, so printing history fetches nothing.
The model-facing API is unchanged: history[i].action / .frame / .result / .animation_frames, transitions,
previous_frame, last_action_frame, frame_diff().
Import after v6.agent and v6.sandbox_ext (run_v6.py does)."""
import json
import os
from pathlib import Path

ENABLED = os.environ.get('ARC3_V6_LAZY_HISTORY', '0') == '1'
EAGER_TAIL = 4
WINDOW = 8
# round 5k (392b, user 10-07 ~18:50: python time per call grew with the game's history, 0.28 -> 0.89 s for action-only
# calls past 1,600 actions; the payload still carried every entry's action, result and metadata, sent and parsed per call
# and after every action, and the sandbox walked every entry per refresh): only the last HISTORY_TAIL entries are sent
# with their records; older positions are stubs whose records arrive in PAGE-sized pages on first access; the level and
# attempt starts come from the host; a forward scan fetches frames SEQ_WINDOW at a time. ARC3_V6_HISTORY_PAGED=0 restores
# the previous payload.
PAGED = os.environ.get('ARC3_V6_HISTORY_PAGED', '1') == '1'
HISTORY_TAIL = int(os.environ.get('ARC3_V6_HISTORY_TAIL', '128'))
PAGE = 256
SEQ_WINDOW = 64
_REGISTRY = {}                     # str(state path) -> (current_frame, history list owned by the game session)
STATS = dict(writes=0, loads=0, window_requests=0, frames_sent=0, meta_requests=0, meta_sent=0)

_SANDBOX_INSERT = '''    _v6_fcache = {}
    _v6_flast = [-10]

    def _v6_fetch(index):
        index = int(index)
        hit = _v6_fcache.get(index)
        if hit is not None:
            return hit
        if index == _v6_flast[0] + 1:                    # round 5k: a forward scan gets a larger window
            lo, hi = index, index + %(seq)d - 1
        else:
            lo, hi = max(0, index - %(win)d + 1), index + %(win)d
        _send({"type": "animation", "request": {"kind": "history_frame_window", "lo": lo, "hi": hi}})
        reply = _recv()
        if reply.get("type") == "animation_result":
            for item in reply.get("animation") or []:
                try:
                    _v6_fcache[int(item.get("index"))] = _frame_from_payload(item)
                except Exception:
                    pass
        _v6_flast[0] = hi
        return _v6_fcache.get(index)

    _v6_mcache = {}

    def _v6_record(pos):
        pos = int(pos)
        hit = _v6_mcache.get(pos)
        if hit is not None:
            return hit
        lo = (pos // %(page)d) * %(page)d
        _send({"type": "animation", "request": {"kind": "history_meta_window", "lo": lo, "hi": lo + %(page)d - 1}})
        reply = _recv()
        if reply.get("type") == "animation_result":
            for item in reply.get("animation") or []:
                try:
                    _v6_mcache[int(item.get("index"))] = item
                except Exception:
                    pass
        return _v6_mcache.get(pos) or {}

    def _v6_meta_str(entry):
        meta = entry.__dict__.get("_v6_meta")
        if meta is None:
            return str(getattr(entry, "frame", None))
        shape = meta.get("shape") or [0, 0]
        return "AsciiFrameView(level=%%s, step=%%s, shape=%%sx%%s)" %% (meta.get("level"), meta.get("step"), shape[0], shape[1])

    class _V6LazyEntry(HistoryEntryView):
        def __init__(self, *, action, ref, meta, result=None, frame=None):
            self.action = action
            self.result = dict(result) if isinstance(result, dict) else {}
            self.__dict__["_v6_ref"] = int(ref)
            self.__dict__["_v6_meta"] = dict(meta or {})
            if frame is not None:
                _v6_fcache[int(ref)] = frame

        @property
        def frame(self):
            return _v6_fetch(self.__dict__["_v6_ref"])

        def __str__(self):
            return "AsciiHistoryEntryView(action=%%r, frame=%%s)" %% (self.action, _v6_meta_str(self))

        __repr__ = __str__

    class _V6StubEntry(HistoryEntryView):
        """round 5k: a position before the eager tail; its record arrives with its page on first access."""
        def __init__(self, pos, ref):
            self.__dict__["_v6_pos"] = int(pos)
            self.__dict__["_v6_ref"] = int(ref)
            self.__dict__["_v6_index"] = int(ref)

        @property
        def action(self):
            return str(_v6_record(self.__dict__["_v6_pos"]).get("action", ""))

        @property
        def result(self):
            r = self.__dict__.get("_v6_result")
            if r is None:
                r = _v6_record(self.__dict__["_v6_pos"]).get("result")
                r = dict(r) if isinstance(r, dict) else {}
                self.__dict__["_v6_result"] = r
            return r

        @property
        def frame(self):
            return _v6_fetch(self.__dict__["_v6_ref"])

        def __str__(self):
            meta = _v6_record(self.__dict__["_v6_pos"]).get("meta") or {}
            shape = meta.get("shape") or [0, 0]
            return ("AsciiHistoryEntryView(action=%%r, frame=AsciiFrameView(level=%%s, step=%%s, shape=%%sx%%s))"
                    %% (self.action, meta.get("level"), meta.get("step"), shape[0], shape[1]))

        __repr__ = __str__

    class _V6History(list):
        """round 5k: the full-length history list of a paged payload (stubs, then the eager tail)."""
        _v6_paged = True

    class _V6LazyTransition(TransitionView):
        def __init__(self, history, index):
            self.__dict__["_v6_h"] = history
            self.__dict__["_v6_i"] = index

        @property
        def action(self):
            a = self.__dict__.get("_v6_action")
            if a is None:
                a = str(getattr(self.__dict__["_v6_h"][self.__dict__["_v6_i"]], "action", "") or "").strip()
                self.__dict__["_v6_action"] = a
            return a

        @property
        def result(self):
            r = self.__dict__.get("_v6_result")
            if r is None:
                e = self.__dict__["_v6_h"][self.__dict__["_v6_i"]]
                r = dict(e.result) if isinstance(getattr(e, "result", None), dict) else {}
                self.__dict__["_v6_result"] = r
            return r

        @property
        def before_frame(self):
            i = self.__dict__["_v6_i"]
            return self.__dict__["_v6_h"][i - 1].frame if i > 0 else None

        @property
        def after_frame(self):
            return self.__dict__["_v6_h"][self.__dict__["_v6_i"]].frame

        @property
        def frame(self):
            return self.after_frame

        def __str__(self):
            h, i = self.__dict__["_v6_h"], self.__dict__["_v6_i"]
            return ("ActionTransitionView(action=%%r, before_frame=%%s, after_frame=%%s)"
                    %% (self.action, _v6_meta_str(h[i - 1]) if i > 0 else None, _v6_meta_str(h[i])))

        __repr__ = __str__

    def _v6_history_from_payload(payload):
        if isinstance(payload, dict) and payload.get("v6_paged"):     # round 5k
            n, t = int(payload.get("n") or 0), int(payload.get("tail_from") or 0)
            items = _V6History()
            for p0, h0, count in payload.get("runs") or []:
                for k in range(min(int(count), max(0, t - int(p0)))):
                    items.append(_V6StubEntry(int(p0) + k, int(h0) + k))
            for entry in payload.get("entries") or []:
                fp = entry.get("frame")
                e = _V6LazyEntry(action=str(entry.get("action", "")), ref=entry["frame_ref"], meta=entry.get("meta"),
                                 result=entry.get("result"), frame=_frame_from_payload(fp) if isinstance(fp, dict) else None)
                e.__dict__["_v6_index"] = int(entry["frame_ref"])
                items.append(e)
            items._v6_tail_from = t
            items._v6_empty = set(int(x) for x in payload.get("empty_action") or [])
            items._v6_level_start = int(payload.get("level_start") or 0)
            items._v6_attempt_start = int(payload.get("attempt_start") or 0)
            return items
        items = []
        for entry in payload or []:
            if not isinstance(entry, dict):
                continue
            if "frame_ref" in entry:
                fp = entry.get("frame")
                items.append(_V6LazyEntry(action=str(entry.get("action", "")), ref=entry["frame_ref"], meta=entry.get("meta"),
                                          result=entry.get("result"), frame=_frame_from_payload(fp) if isinstance(fp, dict) else None))
            else:
                items.append(HistoryEntryView(action=str(entry.get("action", "")), frame=_frame_from_payload(entry.get("frame")),
                                              result=entry.get("result")))
        return items

    def _v6_transitions_from_history(history):
        if getattr(history, "_v6_paged", False):        # round 5k: stubs are judged by the host's empty-action list
            t, empty = history._v6_tail_from, history._v6_empty
            return [_V6LazyTransition(history, i) for i in range(len(history))
                    if (i not in empty if i < t else str(getattr(history[i], "action", "") or "").strip())]
        return [_V6LazyTransition(history, i) for i, e in enumerate(history) if str(getattr(e, "action", "") or "").strip()]

    globals()["_history_from_payload"] = _v6_history_from_payload
    globals()["_transitions_from_history"] = _v6_transitions_from_history
''' % dict(win=WINDOW, seq=SEQ_WINDOW, page=PAGE)
# v7.5 TOOL-3 (the user 10-23, review Part 4; q74: 19 "'_V6LazyEntry' / '_V6StubEntry' object has no attribute 'step'"): history
# entries have `.step`, the step of their board (= `.frame.step`; the paged entries read it from their record's meta, no frame fetch)
_V75_STEP_INSERT = '''    HistoryEntryView.step = property(lambda self: getattr(self.frame, "step", None))

    def _v75_meta_step(meta, entry):
        step = (meta or {}).get("step")
        return step if step is not None else getattr(entry.frame, "step", None)

    _V6LazyEntry.step = property(lambda self: _v75_meta_step(self.__dict__.get("_v6_meta"), self))
    _V6StubEntry.step = property(lambda self: _v75_meta_step(_v6_record(self.__dict__["_v6_pos"]).get("meta"), self))
'''
if os.environ.get('ARC3_V75_POLISH', '0') == '1':
    _SANDBOX_INSERT += _V75_STEP_INSERT


def _mem_write(path, *, current_frame, history):
    from inference.agent.runtime_state import frame_to_payload
    _REGISTRY[str(path)] = (current_frame, history)
    STATS['writes'] += 1
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + '.tmp')
    tmp.write_text(json.dumps(dict(v6_lazy=True, history_len=len(history), current_frame=frame_to_payload(current_frame))))
    tmp.replace(p)


def _make_load(original):
    def _mem_load(path):
        hit = _REGISTRY.get(str(path))
        if hit is None:
            return original(path)
        STATS['loads'] += 1
        frame, history = hit
        return frame, list(history)
    return _mem_load


def _item(history_entries, i, pos, with_frame):
    from inference.agent import tool_agent as ta
    entry = history_entries[i]
    f = entry.frame
    rows = len(f.grid)
    result = dict(getattr(entry, 'result', {}) or {})
    if 'score' in result:                              # v6.1 item 10: the model reads levels_completed
        result['levels_completed'] = result.pop('score')
    item = {'action': entry.action, 'frame_ref': i, 'index': pos, 'result': result,
            'meta': {'step': f.step, 'level': f.level, 'shape': [rows, len(f.grid[0]) if rows else 0]}}
    if with_frame:
        item['frame'] = ta._ascii_frame_view_payload(f)
    return item


def _visible(history_entries):
    """round 5k: the host indices the calling agent's sandbox sees: entries with a frame, outside the run-B hidden ranges
    (the paged payload applies the scope itself; install_history_scope passes it through)."""
    from v62 import sandbox_ext
    agent = getattr(sandbox_ext.CURRENT, 'agent', None)
    hidden = list(getattr(agent, '_v61_hidden', None) or []) if agent is not None else []
    return [i for i, e in enumerate(history_entries) if getattr(e, 'frame', None) is not None
            and not any(lo <= i < hi for lo, hi in hidden)]


def lazy_history_payload(history_entries):
    if not PAGED:                                      # the previous payload: every entry's record, the last frames
        out, n = [], len(history_entries)
        for i, entry in enumerate(history_entries):
            if getattr(entry, 'frame', None) is None:
                continue
            out.append(_item(history_entries, i, len(out), i >= n - EAGER_TAIL))
        return out
    vis = _visible(history_entries)
    n = len(vis)
    t = max(0, n - HISTORY_TAIL)
    level_start = attempt_start = 0                    # sandbox_ext's start frames, found here instead of per refresh
    for pos, i in enumerate(vis):
        r = getattr(history_entries[i], 'result', None) or {}
        if r.get('level_completed'):
            level_start = attempt_start = pos
        elif r.get('action_name') == 'RESET' or r.get('automatic'):
            attempt_start = pos
    runs = []                                          # [first position, first host index, count]
    for pos, i in enumerate(vis):
        if runs and runs[-1][1] + runs[-1][2] == i:
            runs[-1][2] += 1
        else:
            runs.append([pos, i, 1])
    return {'v6_paged': 1, 'n': n, 'tail_from': t, 'runs': runs,
            'entries': [_item(history_entries, vis[pos], pos, pos >= n - EAGER_TAIL) for pos in range(t, n)],
            'empty_action': [pos for pos in range(t) if not str(getattr(history_entries[vis[pos]], 'action', '') or '').strip()],
            'level_start': level_start, 'attempt_start': attempt_start}


def meta_window(agent, lo, hi):
    """Host side of 'history_meta_window' (round 5k): the records (with 'index' = sandbox position) of positions lo..hi."""
    game = getattr(getattr(agent, '_step_env_callback', None), '__self__', None)
    hist = getattr(game, 'history_entries', None) or []
    vis = _visible(hist)
    STATS['meta_requests'] += 1
    out = [_item(hist, vis[pos], pos, False) for pos in range(max(0, int(lo)), min(int(hi), len(vis) - 1) + 1)]
    STATS['meta_sent'] += len(out)
    return out


def frame_window(agent, lo, hi):
    """Host side of 'history_frame_window': frame payloads (with 'index') of history entries lo..hi."""
    from inference.agent import tool_agent as ta
    game = getattr(getattr(agent, '_step_env_callback', None), '__self__', None)
    hist = getattr(game, 'history_entries', None) or []
    STATS['window_requests'] += 1
    out = []
    for i in range(max(0, int(lo)), min(int(hi), len(hist) - 1) + 1):
        p = ta._ascii_frame_view_payload(hist[i].frame)
        if p is not None:
            p['index'] = i
            out.append(p)
    STATS['frames_sent'] += len(out)
    return out


def install():
    if not ENABLED:
        return False
    from inference.agent import python_tool_sandbox as sb
    from inference.agent import runtime_state as rs
    from inference.agent import tool_agent as ta
    from v62 import agent as v6agent
    from v62 import sandbox_ext
    try:
        from inference.framework import solver as sv           # needs arcengine (absent in local tests)
    except ImportError:
        sv = None
    original_load = rs.load_runtime_state
    mem_load = _make_load(original_load)
    for mod in (sv, rs):
        if mod is not None:
            mod.write_runtime_state = _mem_write
    for mod in (ta, rs, v6agent):
        mod.load_runtime_state = mem_load
    ta._ascii_history_view_payload = lazy_history_payload
    anchor = '    _refresh_state(initial.get("state") or {})\n'
    if anchor not in sb._SANDBOX_BOOTSTRAP:
        raise RuntimeError('sandbox bootstrap changed: lazy-history anchor not found')
    sb._SANDBOX_BOOTSTRAP = sb._SANDBOX_BOOTSTRAP.replace(anchor, _SANDBOX_INSERT + anchor, 1)
    outer = ta.run_sandboxed_python                      # sandbox_ext's wrapper

    def _run(*, code, timeout_seconds, initial_state, action_handler, animation_handler=None, **kwargs):
        agent = getattr(sandbox_ext.CURRENT, 'agent', None)
        inner = animation_handler

        def handler(request):
            if str(request.get('kind') or '') == 'history_frame_window' and agent is not None:
                return frame_window(agent, request.get('lo', 0), request.get('hi', -1))
            if str(request.get('kind') or '') == 'history_meta_window' and agent is not None:      # round 5k
                return meta_window(agent, request.get('lo', 0), request.get('hi', -1))
            return inner(request) if inner is not None else []
        return outer(code=code, timeout_seconds=timeout_seconds, initial_state=initial_state,
                     action_handler=action_handler, animation_handler=handler, **kwargs)
    ta.run_sandboxed_python = _run
    return True
