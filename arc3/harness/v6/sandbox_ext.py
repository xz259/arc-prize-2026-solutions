"""Harness v6 sandbox extensions (spec §5.2, §5.4, §5.5; BUILD_PLAN 5-7), applied to dfranzen's sandbox at import:
  - seeds grid.py and objects.py (v6/seeds, the control's helpers; they accept frames with .ascii and frame dicts),
    executed into every call's namespace before the library is restored (a library function of the same name wins);
  - programs(query=None) and read_code(name) over this game's library (name, signature, doc, step added, source);
  - history[i].animation_frames: the frames action i passed through, fetched lazily from the host's animation side
    store (empty when the action did not animate), as views like current_frame.
Host side: run_sandboxed_python is wrapped to add initial_state['v6'] (library + seeds) and to answer
'history_frames' animation requests from the calling agent's game session (thread-local current agent).
"""
import threading
from pathlib import Path

from inference.agent import python_tool_sandbox as sb
from inference.agent import tool_agent as ta
from inference.utils.animation import build_animation_frames

SEEDS = '\n\n'.join((Path(__file__).with_name('seeds') / n).read_text() for n in ('grid.py', 'objects.py'))
CURRENT = threading.local()

_ANCHOR = '        _refresh_state(initial.get("state") or {})\n'
_INSERT = '''        _v6 = (initial.get("state") or {}).get("v6") or {}

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

        def _refresh_state(payload):
            _v6_refresh_original(payload)
            for _v6_i, _v6_e in enumerate(runtime_globals.get("history") or []):
                try:
                    _v6_e.__dict__["_v6_index"] = _v6_i
                except Exception:
                    pass

        _v6_library = list(_v6.get("library") or [])

        def programs(query=None):
            """This game's program library: name, signature, doc, step added (query filters by name or doc)."""
            q = str(query or "").lower()
            return [dict(name=e.get("name"), signature=e.get("signature"), doc=e.get("doc"), step_added=e.get("step"))
                    for e in _v6_library
                    if not q or q in str(e.get("name", "")).lower() or q in str(e.get("doc", "")).lower()]

        def read_code(name):
            """The source of a library function."""
            for e in _v6_library:
                if e.get("name") == name:
                    return e.get("source")
            raise KeyError("no program named %r; programs() lists the library" % (name,))

        if _v6:
            runtime_globals["programs"] = programs
            runtime_globals["read_code"] = read_code
            if _v6.get("seeds"):
                exec(compile(_v6["seeds"], "<seeds>", "exec"), runtime_globals, runtime_globals)
'''
_INSERT = '\n'.join(l[4:] if l.startswith('        ') else l for l in _INSERT.split('\n'))
_ANCHOR_D = _ANCHOR[4:]
if _ANCHOR_D not in sb._SANDBOX_BOOTSTRAP:
    raise RuntimeError('sandbox bootstrap changed: dedented anchor not found')
sb._SANDBOX_BOOTSTRAP = sb._SANDBOX_BOOTSTRAP.replace(_ANCHOR_D, _INSERT + _ANCHOR_D, 1)

_original_run = ta.run_sandboxed_python


def _run_sandboxed_python(*, code, timeout_seconds, initial_state, action_handler, animation_handler=None, **kwargs):
    agent = getattr(CURRENT, 'agent', None)
    if agent is not None:
        initial_state = dict(initial_state or {})
        initial_state['v6'] = dict(library=agent._v6_library_payload(), seeds=SEEDS)
        inner = animation_handler

        def handler(request):
            if str(request.get('kind') or '') == 'history_frames':
                return agent._v6_history_frames(int(request.get('index', -1)))
            return inner(request) if inner is not None else {}
        animation_handler = handler
    return _original_run(code=code, timeout_seconds=timeout_seconds, initial_state=initial_state,
                         action_handler=action_handler, animation_handler=animation_handler, **kwargs)


ta.run_sandboxed_python = _run_sandboxed_python


def host_history_frames(session, index):
    """Frames of history entry `index` from the game session's animation side store (v6.patches)."""
    from v6.patches import unpack
    record = (getattr(session, 'v6_anim', None) or {}).get(index)
    if not record:
        return []
    return build_animation_frames(dict(record, chain=[unpack(g) for g in record['chain']]))
