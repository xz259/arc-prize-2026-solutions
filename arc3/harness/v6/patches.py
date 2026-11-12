"""Harness v6 port deltas applied to dfranzen's modules at import (spec §5.2, §5.3; BUILD_PLAN 5, 9). Imported by
run_v6.py after the environment is set and before any game starts.
  1. Transient-pixel exemption: an executed action whose animation has transient cells INSIDE the board area (outside
     the 4-cell edge band) is never a NO-OP. _execute_action marks it interior_transient (and gameplay_changed /
     no_op so the batch no-op stop does not fire); _action_verdict returns 'effect (animated; see the animation note)'.
  2. Sandbox limits: nice 10 and RLIMIT_AS (ARC3_V6_SANDBOX_AS_GIB, default 4) in every sandbox subprocess, on top of
     his RLIMIT_CPU / FSIZE / NOFILE.
"""
import os

from inference.agent import python_tool_sandbox as sb
from inference.framework import solver as sv
from inference.utils.animation import _transient_cells

BORDER = sv.NOOP_GUARD_BORDER_WIDTH
AS_GIB = float(os.environ.get('ARC3_V6_SANDBOX_AS_GIB', '4'))

# 1. the exemption -------------------------------------------------------------------------------------------------
_original_verdict = sv._action_verdict


def _action_verdict(item):
    if item.get('interior_transient') and not (item.get('game_over') or item.get('run_complete') or item.get('done')
                                               or item.get('level_completed')):
        return 'effect (animated; see the animation note)'
    if item.get('game_over') and os.environ.get('ARC3_V6_A13', '1') == '1':      # A2 wording (A1-A3 default on)
        return 'LEVEL RESTARTED (attempt %s ended here)' % item.get('v6_attempt', '?')
    return _original_verdict(item)


sv._action_verdict = _action_verdict
_original_execute = sv._HarnessGameSession._execute_action


def _interior(cells, rows, cols):
    return [(r, c) for r, c in cells if BORDER <= r < rows - BORDER and BORDER <= c < cols - BORDER]


def _pack(grid):
    return (len(grid), len(grid[0]) if grid else 0, bytes(v for row in grid for v in row))


def unpack(packed):
    rows, cols, data = packed
    return [list(data[r * cols:(r + 1) * cols]) for r in range(rows)]


def _execute_action(self, action, **kwargs):
    payload = _original_execute(self, action, **kwargs)
    # attempts per level (A2's attempt number, A3's attempt lengths): actions since the level started or last restarted
    if not payload.get('automatic'):
        self.__dict__['v6_attempt_actions'] = self.__dict__.get('v6_attempt_actions', 0) + 1
        if payload.get('level_completed'):
            self.__dict__['v6_attempt_actions'] = 0
        elif payload.get('game_over'):
            lengths = self.__dict__.setdefault('v6_level_attempts', {}).setdefault(payload.get('level'), [])
            lengths.append(self.__dict__['v6_attempt_actions'])
            payload['v6_attempt'] = len(lengths)
            self.__dict__['v6_attempt_actions'] = 0
    chain = payload.get('animation_chain') or []
    if len(chain) > 2 and self.history_entries:   # animation side store (spec §5.4), compact: one byte per cell
        store = self.__dict__.setdefault('v6_anim', {})
        store[len(self.history_entries) - 1] = dict(chain=[_pack(g) for g in chain], step=payload.get('action_num'),
                                                    level=payload.get('level'))
    if chain and payload.get('gameplay_changed') is False:
        rows = len(chain[-1])
        cols = len(chain[-1][0]) if rows else 0
        inner = _interior(_transient_cells(chain), rows, cols)
        if inner:
            payload['interior_transient'] = len(inner)
            payload['gameplay_changed'] = True
            payload['no_op'] = False
            if self.history_entries and isinstance(self.history_entries[-1].result, dict):
                self.history_entries[-1].result.update(interior_transient=len(inner), gameplay_changed=True,
                                                       no_op=False)
    return payload


sv._HarnessGameSession._execute_action = _execute_action

# 2. sandbox limits ------------------------------------------------------------------------------------------------
_OLD = """    for limit, value in (
        (getattr(resource, "RLIMIT_CPU", None), cpu_limit),"""
_NEW = """    try:
        os.nice(10)
    except Exception:
        pass
    for limit, value in (
        (getattr(resource, "RLIMIT_AS", None), %d),
        (getattr(resource, "RLIMIT_CPU", None), cpu_limit),""" % int(AS_GIB * (1 << 30))
if _OLD not in sb._SANDBOX_BOOTSTRAP:
    raise RuntimeError('sandbox bootstrap changed: limits patch anchor not found')
sb._SANDBOX_BOOTSTRAP = sb._SANDBOX_BOOTSTRAP.replace(_OLD, _NEW, 1)
assert 'import os' in sb._SANDBOX_BOOTSTRAP.split('def _set_limits')[0]
