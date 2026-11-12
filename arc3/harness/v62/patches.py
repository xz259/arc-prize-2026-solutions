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

from v62 import session_ext

AS_GIB = float(os.environ.get('ARC3_V6_SANDBOX_AS_GIB', '4'))
V77 = os.environ.get('ARC3_V77', '0') == '1'   # v7.7 (apply_v77.py): board_shift names a shift of the whole interior
# round 5m (item 123): seconds of CPU between the stated limit (ComputeLimitError) and the kernel's hard kill; room for
# the call's bookkeeping and its report after the limit (the wall-clock backstop is unchanged)
CPU_MARGIN = 5

# 1. game session ----------------------------------------------------------------------------------------------------
_base_execute = session_ext.install()


def _pack(grid):
    return (len(grid), len(grid[0]) if grid else 0, bytes(v for row in grid for v in row))


def unpack(packed):
    rows, cols, data = packed
    return [list(data[r * cols:(r + 1) * cols]) for r in range(rows)]


def _end_attempt(self, level, cause=None):
    """Record the attempt that just ended (length and no-ops) for A3's lengths and the v6.1 counters. Round 5l (item 109,
    A4 / B4): and how it ended ('game_over', 'reset', or 'restart' for the harness's level restart)."""
    lengths = self.__dict__.setdefault('v6_level_attempts', {}).setdefault(level, [])
    lengths.append(self.__dict__.get('v6_attempt_actions', 0))
    self.__dict__.setdefault('v61_level_attempt_noops', {}).setdefault(level, []).append(
        self.__dict__.get('v61_attempt_noops', 0))
    self.__dict__.setdefault('v62_level_attempt_ends', {}).setdefault(level, []).append(cause)
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
            payload['v6_attempt'] = _end_attempt(self, payload.get('level'), 'game_over')
        elif payload.get('action_name') == 'RESET':          # round 5: no inferred restarts
            harness = bool(self.__dict__.get('v61_harness_reset'))
            # round 5l (item 109, A4; 392b g518: 16 of 44 ledger lines were 1-action phantom attempts): a RESET on a board
            # already at the level's start ends no attempt (its action counts in the attempt that goes on)
            # round 5l review (Fable E3): the harness restart ends an attempt only if it had actions besides that RESET
            if (harness and self.__dict__.get('v6_attempt_actions', 0) > 1) or (payload.get('board_changed') and not harness):
                payload['v6_attempt'] = _end_attempt(self, payload.get('level'), 'restart' if harness else 'reset')
            key = 'v61_harness_resets' if harness else 'v61_voluntary_resets'
            self.__dict__[key] = self.__dict__.get(key, 0) + 1
        if payload.get('start_return'):                      # round 5l (item 109, A5): an event, not an attempt boundary
            returns = self.__dict__.setdefault('v62_level_start_returns', {})
            returns[payload.get('level')] = returns.get(payload.get('level'), 0) + 1
    chain = payload.get('animation_chain') or []
    if len(chain) > 2 and self.history_entries:   # animation side store (spec §5.4), compact: one byte per cell
        store = self.__dict__.setdefault('v6_anim', {})
        store[len(self.history_entries) - 1] = dict(chain=[_pack(g) for g in chain], step=payload.get('action_num'),
                                                    level=payload.get('level'))
    return payload


sv._HarnessGameSession._execute_action = _execute_action

# 2. verdicts --------------------------------------------------------------------------------------------------------
_original_verdict = sv._action_verdict


def _span(name, ab):
    return '%s %d' % (name, ab[0]) if ab[0] == ab[1] else '%ss %d-%d' % (name, ab[0], ab[1])


def _cells(n):
    """round 5c (B3 cd82, vc33: "1 cells")."""
    return '%d cell%s' % (n, '' if n == 1 else 's')


def _v77_colors(region):
    """v7.7 (lever 4a): a region's most common colour changes: 'b->Y 80, Y->b 70'."""
    return ', '.join('%s->%s %d' % (f, t, k) for f, t, k in region.get('colors') or [])


def _v77_region(region):
    """v7.7 (lever 4a): 'rows 4-20, cols 9-23 (150 cells: b->Y 80, Y->b 70)'."""
    return '%s, %s (%s: %s)' % (_span('row', region['rows']), _span('col', region['cols']), _cells(region['n']), _v77_colors(region))


def _action_verdict(item):
    if item.get('game_over') and not item.get('automatic') and \
            str(item.get('action_display') or item.get('action_name') or '') == 'RESET':
        # round 5l (A1, 392b rw01): a RESET that ends in GAME OVER is not followed by an automatic restart
        verdict = 'GAME OVER (reported right after this RESET; no automatic restart follows a RESET)'
    elif item.get('game_over'):
        verdict = 'LEVEL RESTARTED (attempt %s ended here)' % item.get('v6_attempt', '?')
    elif item.get('run_complete') or item.get('done') or item.get('level_completed'):
        verdict = _original_verdict(item)
    elif item.get('reverted'):
        n = (item.get('v61_edge') or {}).get('n') or 0      # round 5l (Fable 2.2): the band fact on the interior verdict
        verdict = ('REVERTED (animated, then the interior returned to its previous state'           # round 5f (E5)
                   + ('; %s of the outer band changed)' % _cells(n) if n else ')'))
    elif item.get('gameplay_changed') is False:          # round 5 (user: no priors): the rule as a measurement
        # round 5l (A3, user 10-07 23:51 / 10-08 00:31; 392b gh14, dw01, lq01, ac02, kd01, bq01, pc01, td05: real moves in
        # the outer band read NO-OP): NO-OP means nothing on the board changed; a change confined to the outer band has
        # its own label
        n = (item.get('v61_edge') or {}).get('n') or 0
        if n or item.get('board_changed'):
            verdict = ('BAND-ONLY CHANGE (no change in the interior' + ('; %s of the outer band changed)' % _cells(n)
                                                                         if n else '; the outer band changed)'))
        else:
            verdict = 'NO-OP (nothing on the board changed)'
    else:
        verdict = _original_verdict(item)
    ic = item.get('interior_changed')                # round 5d (user): where an 'effect' changed the interior
    if ic and ic.get('n') and verdict.startswith('effect'):
        regs = ic.get('regions') if V77 else None    # v7.7 (the user 10-24, lever 4a): the separate regions and their colours
        if regs and (len(regs) > 1 or ic.get('more')):
            more = int(ic.get('more') or 0)
            verdict += ' | %s changed in the interior in %d separate regions: %s%s' % (
                _cells(ic['n']), len(regs) + more, '; '.join(_v77_region(x) for x in regs),
                ('; and %d smaller region%s (%s)' % (more, '' if more == 1 else 's', _cells(int(ic.get('more_cells') or 0)))
                 if more else ''))
        elif regs:
            verdict += ' | %s changed in the interior (%s, %s: %s)' % (_cells(ic['n']), _span('row', ic['rows']),
                                                                       _span('col', ic['cols']), _v77_colors(regs[0]))
        else:
            verdict += ' | %s changed in the interior (%s, %s)' % (_cells(ic['n']), _span('row', ic['rows']), _span('col', ic['cols']))
        k = item.get('band_connected')               # round 5h (G3): one changed region crossing into the outer band
        if k:
            verdict += '; %s of the outer band also changed and %s connected to them' % (_cells(k), 'is' if k == 1 else 'are')
    since = item.get('actions_since_start')
    since_text = (', %d action%s after the board was last at the level\'s start' % (since, '' if since == 1 else 's')
                  if since else '')
    if item.get('near_level_start') and not item.get('level_start_match'):
        verdict += (' | the interior now differs from the level\'s starting board in only %s, with no GAME OVER reported%s'
                    % (_cells(item['near_level_start']), since_text))
    at = item.get('level_start_match')
    if at and not (item.get('game_over') or item.get('level_completed') or item.get('run_complete')):
        # round 5c (B3 su15, dc22, g50t: "the board inside the 4-cell edge band" was read as the band itself)
        # round 5d (C50 g027 T270: "the whole board ...; N cells of the outer band also returned" read as two events;
        # g035, g014, g023: a return to the start without a GAME OVER was read as a teleport or a reset key)
        verdict += ' | %s now matches the level\'s starting board, with no GAME OVER reported%s%s' % (
            'the whole board' if at == 'full' else 'the interior', since_text,
            ('; %s of the outer band also returned to %s' % (
                _cells(item['edge_cells_restored']),
                'its starting value' if item['edge_cells_restored'] == 1 else 'their starting values'))
            if item.get('edge_cells_restored') and at != 'full' else '')
    if item.get('repeat_in_call'):                   # round 5b: a fact, nothing refused
        k = item['repeat_in_call']
        verdict += (' | already executed from this board earlier in this call (%d time%s; boards compared on the '
                    'interior)' % (k, '' if k == 1 else 's'))
    sh = item.get('board_shift')                     # round 5l (C1): a measurement, no "scroll"
    if sh:
        verdict += (' | a shift of the %s by %d row%s and %d col%s (down and right positive) explains %d of the %d '
                    'changed cells' % ('whole interior' if V77 else 'board content',          # v7.7: a shift of the whole interior
                                       sh['dy'], '' if abs(sh['dy']) == 1 else 's', sh['dx'], '' if abs(sh['dx']) == 1 else 's',
                                       sh['explained'], sh['changed']))
    back = item.get('undo_restored')                 # round 5l (C3)
    if back:
        verdict += ' | the board now equals the board before %s at step %s' % (back.get('action'), back.get('step'))
    hit = session_ext.hit_text(item.get('v61_hit'))
    return verdict + (' | ' + hit if hit else '')


sv._action_verdict = _action_verdict

# 2e: the stale refusal fires once (the sandbox clears its flag before raising); the detail says so -----------------
from types import SimpleNamespace  # noqa: E402

_original_stale_detail = sv._stale_detail
_STALE_OLD = "(StaleStateActionError can be caught if a snippet has good reason to keep acting through no-ops.)"


def _stale_detail(action, stale_after, reverted=False):
    text = _original_stale_detail(action, stale_after)
    # round 5l (A3 option b, user 10-08 00:31): the refusal follows only an action that changed nothing on the board
    for old in ("was a no-op - it produced no change inside the board area -",
                "was a no-op - it changed nothing outside the board border -"):
        text = text.replace(old, "was a NO-OP - nothing on the board changed -")
    text = text.replace("work out why nothing moved", "work out why nothing changed")
    if reverted:              # round 5h (Fable Q3, 386 by35: the second of two REVERTED actions was called a no-op)
        text = text.replace("was a NO-OP - nothing on the board changed -",
                            "was REVERTED - it animated, then the board returned to its previous state -")
        text = text.replace("work out why nothing changed", "work out why the board returned to its previous state")
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
        (getattr(resource, "RLIMIT_CPU", None), cpu_limit - 1 + %d),""" % (int(AS_GIB * (1 << 30)), CPU_MARGIN)


RESET_REFUSAL_NOTE = ("This action() call executed nothing; actions executed earlier in this snippet are not undone.")


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

# v6.2 fix (run A: 60 "10-second limit" errors, 40 of them in two whole-pod stalls at 4 and 10 minutes, where every call's
# wall clock ran out together and its output was lost): the snippet's limit is its own CPU time. A soft RLIMIT_CPU raises
# TimeoutError inside the snippet (its output so far is kept) at the stated limit; the hard limit is 1 s later. The host's wall-clock
# deadline becomes a backstop (sandbox_ext: WALL_BACKSTOP); the CPU seconds come from the state ("v62_cpu_seconds").
# round 5m (item 123; run 396: 22 calls in 7 games were killed at the hard limit, the model saw only "Sandbox process exited
# unexpectedly" and lost its printed output): the call's bookkeeping after the error (re-checking the program library)
# could need more than the 1-s margin. The hard limit is now CPU_MARGIN s after the stated limit (the generic loop above
# sets it), and the error is raised once: the kernel repeats SIGXCPU every second up to the hard limit, and a second
# raise would land in that bookkeeping. A process still killed is reported by the host (sandbox_ext, item 123).
_patch("""            try:
                resource.setrlimit(limit, (value, value))
            except (OSError, ValueError):
                pass


    def _normalize_actions(actions):""", """            try:
                resource.setrlimit(limit, (value, value))
            except (OSError, ValueError):
                pass
        try:
            import signal as _v62_signal
            _v62_xcpu_raised = []

            def _v62_xcpu(signum, frame):
                if _v62_xcpu_raised:
                    return                # round 5m (item 123): once; the rest of the margin is for the report
                _v62_xcpu_raised.append(1)
                raise ComputeLimitError("This call used its %d seconds of computation and was stopped here. It may "
                                        "already have executed actions: inspect the current state before retrying."
                                        % (cpu_limit - 1))
            _v62_signal.signal(_v62_signal.SIGXCPU, _v62_xcpu)
            resource.setrlimit(resource.RLIMIT_CPU, (cpu_limit - 1, cpu_limit - 1 + __CPU_MARGIN__))   # soft at the stated limit
        except Exception:
            pass


    def _normalize_actions(actions):""".replace('__CPU_MARGIN__', str(CPU_MARGIN)))
_patch("""    def _set_limits(timeout_seconds):""", """    class ComputeLimitError(BaseException):
        pass  # v6.2: the snippet used its CPU seconds (a BaseException, so `except Exception` in a snippet cannot swallow it)


    def _set_limits(timeout_seconds):""")
_patch("""        timeout_seconds = max(1, int(initial.get("timeout_seconds", 30)))
""", """        timeout_seconds = max(1, int(initial.get("timeout_seconds", 30)))
        _v62_cpu = (initial.get("state") or {}).get("v62_cpu_seconds")        # v6.2: the CPU-time limit
        if _v62_cpu:
            timeout_seconds = max(1, int(_v62_cpu))
""")

# round 5l (A1, 392b rw01): after a game over the sandbox's latch said "The level has been reset to its start", which is
# false when the game over came from a RESET (no automatic restart follows one)
_patch('''            "game_over": (
                "the previous action reported game_over. The level has been reset to its "
                "start, so the board you planned against is gone and this action was not "
                "executed. The next turn begins from the new state."
            ),''',
       '''            "game_over": (
                "the previous action reported game_over, so the board you planned against "
                "is gone and this action was not executed. The next turn reports the new "
                "state."
            ),''')
# round 5d (C50 g013 7 of 7 cases, g017 T226, g047 T42): dfranzen's RESET-not-first refusal says "This call executed
# no actions." although 1-13 actions had run earlier in the snippet; it refers to the refused action() call only
# Round 5l (B7, user 10-07 23:51: the rule stays, its text says what to do; Fable 3.15): how many actions this Python call
# already executed, that they stand, and where RESET goes
RESET_FIRST_TEXT = ("RESET must be the first game action of a Python call. This action() call executed nothing: RESET was "
                    "not first in its list. Put RESET first in the list, or end this call and put RESET first in the "
                    "next one.",
                    "RESET must be the first game action of a Python call. This action() call executed nothing; the %d "
                    "action(s) executed earlier in this call stand. To restart the level, end this call and put RESET "
                    "first in the next one.")
_patch('''            if action_result.get("stop_reason") == "reset_not_first":
                raise RuntimeError(str(action_result.get("stop_detail")))''',
       '''            if action_result.get("stop_reason") == "reset_not_first":
                _v62_k = sum(max(1, int(_r.get("executed_count") or 1)) for _r in action_results if _r.get("executed"))
                raise RuntimeError(RESET_FIRST_TEXT[1] % _v62_k if _v62_k else RESET_FIRST_TEXT[0])''')
_patch('''        runtime_globals["TerminalStateActionError"] = TerminalStateActionError
''',
       '''        runtime_globals["TerminalStateActionError"] = TerminalStateActionError
        RESET_REFUSAL_NOTE = %r
        RESET_FIRST_TEXT = %r
''' % (RESET_REFUSAL_NOTE, RESET_FIRST_TEXT))
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
            ComputeLimitError,
            Exception,
        ) as exc:''')
# 2b: a single REVERTED action does not arm the stale flag; two in a row (stop_reason 'reverted') do
_patch('''            if gameplay_changed is False and action_result.get("executed"):''',
       '''            if action_result.get("executed"):          # round 5h: the flag describes the latest executed call only
                del _stale_state[:]
            # round 5l (A3 option b, user 10-08 00:31): only an action after which nothing on the board changed arms it
            # (a change confined to the outer band, BAND-ONLY CHANGE, does not)
            if (gameplay_changed is False and action_result.get("executed")
                    and action_result.get("last_board_changed") is False
                    and (not action_result.get("reverted") or action_result.get("stop_reason") == "reverted")
                    and not str((action_result.get("executed_actions") or [action_result.get("action_display") or ""])[-1]
                                ).startswith("RESET")):''')
# v6.2 (run A: m0r0 9 of 10, cn04 4 of 6, tn36 3 of 3, lp85 2 of 2 stale refusals false, all after a no-op click in a probe
# loop; m0r0 T200 then concluded "MOUSE is inert"): a click or a RESET that changed nothing leaves every position as it
# was, so it does not make the snippet's later targets stale; only a no-op of a move or SPACE arms the refusal.
# Round 5h (Fable Q3, 386 by35: 12 x the same no-op click in one attempt): a no-op click arms the flag too, and the host
# (session_ext.step_env) then refuses only the same click; any other next action clears it. Each executed call replaces
# the flag, so it always names the latest executed action (it used to keep the first arming action of the snippet).

# v6.2 round 3 made a LEVEL RESTARTED stop end the snippet; round 5 (user: no priors) removed inferred restarts, so
# that terminal state is gone with them.

# v6.2 round 3 (b) (A2 vc33: an IndexError in `b[2][0]` was blamed on `history[140]` for three turns): tracebacks show
# the snippet's source line under each frame
_patch('''        user_frames = [frame for frame in extracted if frame.filename == "<python_tool>"]''',
       '''        user_frames = [frame for frame in extracted if frame.filename in ("<python_tool>", "<retained>")]''')
_patch('''        for frame in user_frames or extracted[-1:]:
            lines.append(f'  File "<python_tool>", line {frame.lineno}, in {frame.name}')''',
       '''        _v62_quals, _v62_tb = [], exc.__traceback__   # round 5h (G7): each frame's qualified name
        while _v62_tb is not None:
            _v62_quals.append(getattr(_v62_tb.tb_frame.f_code, "co_qualname", "") or "")
            _v62_tb = _v62_tb.tb_next
        _v62_qual = {id(f): q for f, q in zip(extracted, _v62_quals)}
        for frame in user_frames or extracted[-1:]:
            if frame.filename == "<retained>":   # round 4 (A3 tu93 T57, T92; s5i5): frames inside library functions
                # round 5h (G7, 386 gy0t: `Library function <genexpr>`, read_code("<genexpr>")): a generator expression,
                # lambda or nested function is named with the library function that contains it
                _v62_top = (_v62_qual.get(id(frame)) or frame.name).split(".")[0]
                if _v62_top == frame.name:
                    lines.append(f'  Library function {frame.name}, line {frame.lineno} of its source '
                                 f'(read_code("{frame.name}") shows it)')
                elif not _v62_top.startswith("<"):
                    lines.append(f'  {frame.name} in library function {_v62_top}, line {frame.lineno} of its source '
                                 f'(read_code("{_v62_top}") shows it)')
                else:
                    lines.append(f'  {frame.name} in a library function, line {frame.lineno} of its source')
                continue
            lines.append(f'  File "<python_tool>", line {frame.lineno}, in {frame.name}')
            if frame.filename == "<python_tool>" and (frame.line or "").strip():
                lines.append("    " + frame.line.strip()[:200])''')
_patch('''            _code_text = str(initial.get("code", ""))
''', '''            _code_text = str(initial.get("code", ""))
            try:
                import linecache as _v62_linecache
                _v62_linecache.cache["<python_tool>"] = (len(_code_text), None, _code_text.splitlines(True),
                                                         "<python_tool>")
            except Exception:
                pass
''')
# round 5h (G7, 386 lx20 x5, ly85: "Action 1 must be a string or a dict" for Python index 0): positions stay 1-based, as in
# the batch stops, and say so
for _v62_old in ('f"Action {index} is empty."', 'f"Action {index} is missing an `action` field."',
                 'f"Action {index} uses legacy MOUSE x/y fields; use row and col."',
                 'f"Action {index} must be a string or a dict."'):
    _patch(_v62_old, _v62_old.replace('f"Action {index} ', 'f"The action at position {index} (1-based) '), dedent=False)
# round 5l (C4, 392b g513: an action string copied from the history display was refused): action() accepts the display
# form of a click, "MOUSE(row=R, col=C)"
_C4_OLD = '''            if isinstance(item, str):
                action_name = item.strip()
'''
_C4_NEW = _C4_OLD + r'''                _v62_m = __import__("re").match(r"^MOUSE\s*\(\s*row\s*=\s*(-?\d+)\s*,\s*col\s*=\s*(-?\d+)\s*\)$",
                                                action_name, __import__("re").I)
                if _v62_m:
                    normalized.append({"action": "MOUSE", "row": int(_v62_m.group(1)), "col": int(_v62_m.group(2))})
                    continue
'''
_patch(_C4_OLD, _C4_NEW)
# round 5h (Fable Q4.4, 386 kx59 T260, sy25 T299, fx09, cy82, tx36: `print(type(last_animation_timeline))` showed
# `_LazyAnimation`; the timeline was sliced and iterated as a list): the lazy views are named for what they hold, the
# timeline has the rest of the dict interface, and a non-key index says what the object is
_patch('''            runtime_globals["last_animation_frames"] = _LazyAnimation("frames", [])
            runtime_globals["last_animation_timeline"] = _LazyAnimation("timeline", {})''',
       '''            class LazyFrameList(_LazyAnimation):
                """last_animation_frames: a list of frame views, fetched from the harness when first read."""

            class LazyTimelineDict(_LazyAnimation):
                """last_animation_timeline: a dict ('action', 'frames', 'steps'; only 'error' when the latest action
                did not animate), fetched from the harness when first read."""

                def items(self):
                    return self._get().items()

                def values(self):
                    return self._get().values()

                def __getitem__(self, key):
                    if not isinstance(key, str):
                        raise TypeError("last_animation_timeline is a dict with the keys %s, not a list; its steps are "
                                        "last_animation_timeline['steps']" % sorted(self._get()))
                    return self._get()[key]

            runtime_globals["last_animation_frames"] = LazyFrameList("frames", [])
            runtime_globals["last_animation_timeline"] = LazyTimelineDict("timeline", {})''')
# v7.5 TOOL-1 (the user 10-23, review Part 4; q74: 24 and 15 "Object of type LazyTimelineDict is not JSON serializable" in 19 and 13
# games): json.dumps of the lazy animation views writes their contents (the encoder's default returns the fetched value); any other
# object is refused as before. Installed once per sandbox process, after the views are created.
if os.environ.get('ARC3_V75_POLISH', '0') == '1':
    _patch('''            runtime_globals["last_animation_timeline"] = LazyTimelineDict("timeline", {})''',
           '''            runtime_globals["last_animation_timeline"] = LazyTimelineDict("timeline", {})
            json._v75_lazy = _LazyAnimation
            if not getattr(json.JSONEncoder.default, "_v75", False):
                _v75_json_default0 = json.JSONEncoder.default

                def _v75_json_default(self, obj):
                    if isinstance(obj, json._v75_lazy):
                        return obj._get()
                    return _v75_json_default0(self, obj)

                _v75_json_default._v75 = True
                json.JSONEncoder.default = _v75_json_default''')
