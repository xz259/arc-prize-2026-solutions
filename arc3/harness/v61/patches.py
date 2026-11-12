"""Harness v6.1 port deltas applied to dfranzen's modules at import (v6 patches + the v6.1 revision; see
session_ext.py and HARNESS_REVISION_ITEMS_2026-10-05.md). Imported by run_v6.py after the environment is set.
  1. Game session: session_ext's _execute_action (REVERTED, level-start, edge-band and MOUSE-hit labels) and step_env
     (v6.1 stops, per-call cap, stable schema), wrapped here with the attempt bookkeeping (A2 / A3 / v6.1 item 3:
     actions and no-ops per attempt; an attempt also ends at a level-start restart or a voluntary RESET) and the
     animation side store. The v6 transient-pixel exemption is replaced by the REVERTED label (item 2b).
  2. Verdicts: REVERTED; LEVEL RESTARTED for a game over and for a level-start restart; the MOUSE hit appended.
  3. Sandbox: nice 10 and RLIMIT_AS (v6); v6.1: the stale-state refusal fires once (2e); an invalid action raises
     InvalidActionError (2f); the per-call cap raises ActionCapError (2d); a single REVERTED action does not arm the
     stale flag (2b); collections / itertools / math / re / heapq are preloaded (8); level_start_frame and
     attempt_start_frame are runtime globals (2g).
"""
import os

from inference.agent import python_tool_sandbox as sb
from inference.framework import solver as sv

from v61 import session_ext

AS_GIB = float(os.environ.get('ARC3_V6_SANDBOX_AS_GIB', '4'))

# 1. game session ----------------------------------------------------------------------------------------------------
_base_execute = session_ext.install()


def _pack(grid):
    return (len(grid), len(grid[0]) if grid else 0, bytes(v for row in grid for v in row))


def unpack(packed):
    rows, cols, data = packed
    return [list(data[r * cols:(r + 1) * cols]) for r in range(rows)]


def _end_attempt(self, level):
    """Record the attempt that just ended (length and no-ops) for A3's lengths and the v6.1 counters."""
    lengths = self.__dict__.setdefault('v6_level_attempts', {}).setdefault(level, [])
    lengths.append(self.__dict__.get('v6_attempt_actions', 0))
    self.__dict__.setdefault('v61_level_attempt_noops', {}).setdefault(level, []).append(
        self.__dict__.get('v61_attempt_noops', 0))
    self.__dict__['v6_attempt_actions'] = 0
    self.__dict__['v61_attempt_noops'] = 0
    # run B: attempts are counted from the latest conversation restart (decision 5)
    return len(lengths) - (self.__dict__.get('v61_attempt_offset') or {}).get(level, 0)


def _execute_action(self, action, **kwargs):
    payload = _base_execute(self, action, **kwargs)
    if not payload.get('automatic'):
        self.__dict__['v6_attempt_actions'] = self.__dict__.get('v6_attempt_actions', 0) + 1
        if payload.get('gameplay_changed') is False and not payload.get('reverted'):
            self.__dict__['v61_attempt_noops'] = self.__dict__.get('v61_attempt_noops', 0) + 1
        if payload.get('level_completed'):
            self.__dict__['v6_attempt_actions'] = 0
            self.__dict__['v61_attempt_noops'] = 0
        elif payload.get('game_over'):
            payload['v6_attempt'] = _end_attempt(self, payload.get('level'))
        elif payload.get('v61_level_start') == 'restart' or payload.get('action_name') == 'RESET':
            payload['v6_attempt'] = _end_attempt(self, payload.get('level'))
            if payload.get('action_name') == 'RESET':
                key = 'v61_harness_resets' if self.__dict__.get('v61_harness_reset') else 'v61_voluntary_resets'
                self.__dict__[key] = self.__dict__.get(key, 0) + 1
    chain = payload.get('animation_chain') or []
    if len(chain) > 2 and self.history_entries:   # animation side store (spec §5.4), compact: one byte per cell
        store = self.__dict__.setdefault('v6_anim', {})
        store[len(self.history_entries) - 1] = dict(chain=[_pack(g) for g in chain], step=payload.get('action_num'),
                                                    level=payload.get('level'))
    return payload


sv._HarnessGameSession._execute_action = _execute_action

# 2. verdicts --------------------------------------------------------------------------------------------------------
_original_verdict = sv._action_verdict


def _action_verdict(item):
    if item.get('game_over'):
        verdict = 'LEVEL RESTARTED (attempt %s ended here)' % item.get('v6_attempt', '?')
    elif item.get('run_complete') or item.get('done') or item.get('level_completed'):
        verdict = _original_verdict(item)
    elif item.get('v61_level_start') == 'restart':
        verdict = 'LEVEL RESTARTED (attempt %s ended here; the board returned to the level start)' % item.get('v6_attempt', '?')
    elif item.get('v61_level_start') == 'soft':
        verdict = 'BACK AT LEVEL START (the board area matches the level\'s starting board)'
    elif item.get('reverted'):
        verdict = 'REVERTED (animated, then the board returned to its previous state)'
    else:
        verdict = _original_verdict(item)
    hit = session_ext.hit_text(item.get('v61_hit'))
    return verdict + (' | ' + hit if hit else '')


sv._action_verdict = _action_verdict

# 2e: the stale refusal fires once (the sandbox clears its flag before raising); the detail says so -----------------
from types import SimpleNamespace  # noqa: E402

_original_stale_detail = sv._stale_detail
_STALE_OLD = "(StaleStateActionError can be caught if a snippet has good reason to keep acting through no-ops.)"


def _stale_detail(action, stale_after):
    text = _original_stale_detail(action, stale_after)
    return text.replace(_STALE_OLD, "This refusal happens once: the next action call executes.")


assert _STALE_OLD in _original_stale_detail(SimpleNamespace(id=SimpleNamespace(name='ACTION1'), data={}), 'UP'), \
    'stale detail anchor'
sv._stale_detail = _stale_detail

# 3. sandbox -----------------------------------------------------------------------------------------------------------
_OLD = """    for limit, value in (
        (getattr(resource, "RLIMIT_CPU", None), cpu_limit),"""
_NEW = """    try:
        os.nice(10)
    except Exception:
        pass
    for limit, value in (
        (getattr(resource, "RLIMIT_AS", None), %d),
        (getattr(resource, "RLIMIT_CPU", None), cpu_limit),""" % int(AS_GIB * (1 << 30))


def _ded(text):
    """The bootstrap string is the sandbox source dedented by 4 spaces; anchors here are written at file indentation."""
    return '\n'.join(l[4:] if l.startswith('    ') else l for l in text.split('\n'))


def _patch(old, new, dedent=True):
    if dedent:
        old, new = _ded(old), _ded(new)
    if sb._SANDBOX_BOOTSTRAP.count(old) != 1:
        raise RuntimeError('sandbox bootstrap changed: anchor not found exactly once: %r' % old[:80])
    sb._SANDBOX_BOOTSTRAP = sb._SANDBOX_BOOTSTRAP.replace(old, new, 1)


_patch(_OLD, _NEW, dedent=False)
assert 'import os' in sb._SANDBOX_BOOTSTRAP.split('def _set_limits')[0]

# 2e: the stale-state refusal fires once (dfranzen's comment: "a model that has re-read current_frame may legitimately
# want to proceed"; without clearing, every later call in the snippet was refused and catching could not help)
_patch('''            if action_result.get("stop_reason") == "stale_state":
                raise StaleStateActionError(''',
       '''            if action_result.get("stop_reason") == "stale_state":
                del _stale_state[:]
                raise StaleStateActionError(''')
# 2d / 2f: per-call cap and invalid actions raise; BaseException for the cap (a loop's `except Exception` must not
# swallow it), Exception for an invalid action (the snippet may catch it and choose another action)
_patch('''            for _flag in ("run_complete", "done", "level_completed", "game_over"):
                if action_result.get(_flag):''',
       '''            if action_result.get("stop_reason") == "call_cap":
                _echo_cap = action_result.pop("action_echo", None)
                if _echo_cap:
                    print(_echo_cap)
                raise ActionCapError(str(action_result.get("stop_detail") or "per-call action limit reached"))
            if not action_result.get("executed") and action_result.get("error"):
                raise InvalidActionError(str(action_result.get("error")))
            for _flag in ("run_complete", "done", "level_completed", "game_over"):
                if action_result.get(_flag):''')
_patch('''        runtime_globals["TerminalStateActionError"] = TerminalStateActionError
''',
       '''        runtime_globals["TerminalStateActionError"] = TerminalStateActionError

        class ActionCapError(BaseException):
            pass  # v6.1: this call reached the per-call action limit; the next turn continues

        class InvalidActionError(Exception):
            pass  # v6.1: the action is not valid now; nothing was executed

        runtime_globals["ActionCapError"] = ActionCapError
        runtime_globals["InvalidActionError"] = InvalidActionError
''')
# 2d: the snippet handler reports the cap like his guard refusals (BaseException classes must be named there, or the
# child process dies and the model only sees "exited unexpectedly"; found in the 2026-10-05 dry run)
_patch('''            TerminalStateActionError,
            Exception,
        ) as exc:''',
       '''            TerminalStateActionError,
            ActionCapError,
            Exception,
        ) as exc:''')
# 2b: a single REVERTED action does not arm the stale flag; two in a row (stop_reason 'reverted') do
_patch('''            if gameplay_changed is False and action_result.get("executed"):''',
       '''            if (gameplay_changed is False and action_result.get("executed")
                    and (not action_result.get("reverted") or action_result.get("stop_reason") == "reverted")):''')
