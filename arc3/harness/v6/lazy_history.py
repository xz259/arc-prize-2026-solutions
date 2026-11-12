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
_REGISTRY = {}                     # str(state path) -> (current_frame, history list owned by the game session)
STATS = dict(writes=0, loads=0, window_requests=0, frames_sent=0)

_SANDBOX_INSERT = '''    _v6_fcache = {}

    def _v6_fetch(index):
        index = int(index)
        hit = _v6_fcache.get(index)
        if hit is not None:
            return hit
        _send({"type": "animation", "request": {"kind": "history_frame_window", "lo": max(0, index - %(win)d + 1), "hi": index + %(win)d}})
        reply = _recv()
        if reply.get("type") == "animation_result":
            for item in reply.get("animation") or []:
                try:
                    _v6_fcache[int(item.get("index"))] = _frame_from_payload(item)
                except Exception:
                    pass
        return _v6_fcache.get(index)

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

    class _V6LazyTransition(TransitionView):
        def __init__(self, history, index):
            entry = history[index]
            self.action = str(getattr(entry, "action", "") or "").strip()
            self.result = dict(entry.result) if isinstance(getattr(entry, "result", None), dict) else {}
            self.__dict__["_v6_h"] = history
            self.__dict__["_v6_i"] = index

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
        return [_V6LazyTransition(history, i) for i, e in enumerate(history) if str(getattr(e, "action", "") or "").strip()]

    globals()["_history_from_payload"] = _v6_history_from_payload
    globals()["_transitions_from_history"] = _v6_transitions_from_history
''' % dict(win=WINDOW)


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


def lazy_history_payload(history_entries):
    from inference.agent import tool_agent as ta
    out, n = [], len(history_entries)
    for i, entry in enumerate(history_entries):
        f = getattr(entry, 'frame', None)
        if f is None:
            continue
        rows = len(f.grid)
        item = {'action': entry.action, 'frame_ref': i, 'result': dict(getattr(entry, 'result', {}) or {}),
                'meta': {'step': f.step, 'level': f.level, 'shape': [rows, len(f.grid[0]) if rows else 0]}}
        if i >= n - EAGER_TAIL:
            item['frame'] = ta._ascii_frame_view_payload(f)
        out.append(item)
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
    from v6 import agent as v6agent
    from v6 import sandbox_ext
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
            return inner(request) if inner is not None else []
        return outer(code=code, timeout_seconds=timeout_seconds, initial_state=initial_state,
                     action_handler=action_handler, animation_handler=handler, **kwargs)
    ta.run_sandboxed_python = _run
    return True
