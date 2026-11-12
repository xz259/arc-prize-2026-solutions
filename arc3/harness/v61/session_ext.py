"""Harness v6.1 game-session layer: modified copies of dfranzen's _HarnessGameSession._execute_action and .step_env
(src/inference/framework/solver.py @ 10882e3, Apache-2.0, see LICENSE.dfranzen), installed by v61/patches.py.
Items (HARNESS_REVISION_ITEMS_2026-10-05.md):
  2b REVERTED: an executed action whose animation had transient interior cells and whose settled interior equals the
     board before it is labelled reverted (gameplay_changed False, no_op False). A batch stops at the 2nd REVERTED
     action in a row (stop_reason 'reverted'); a single revert neither stops the batch nor arms the stale-state flag.
  2c level start: when the full frame (edge included) returns to the level's starting frame from a frame whose edge
     band differed from the start's (a bar refilled), the action is labelled v61_level_start='restart' (an attempt
     ended without a game-over flag; the batch stops, the patches wrapper counts an attempt); when only the interior
     matches, or the edge never differed (a game without a bar, where a walk back to the start gives the identical
     frame), 'soft' (the batch stops, nothing is counted). UNDO and RESET never trigger it. The level start is the first frame of the game, of each level, and the
     frame after any RESET.
  2d per-call cap: a python call executes at most CALL_CAP actions (the agent resets the per-call counter); a batch is
     cut at the cap (stop_reason 'call_cap') and a call already at the cap is refused (executed False).
  2g stable schema: every step_env result carries executed_actions, action_trace, gameplay_changed_per_action and
     stop_reason (None when nothing stopped it).
  3  edge band: per action, the cells changed in the 4-cell edge band (count, colour transitions, rows / cols).
  5  MOUSE hit: what the click landed on in the board before it (component colour, size, bbox, id in that board's
     segmentation; 'background' when the component covers >= 1/4 of the board).
Everything else is his code unchanged.
"""
from __future__ import annotations

import copy
import os
from collections import Counter
from typing import Any

import arcengine

from inference.framework import solver as sv
from inference.utils.animation import _transient_cells
from inference.utils.grid_utils import ARC_COLOR_CHARS

CALL_CAP = int(os.environ.get('ARC3_V61_CALL_CAP', '40'))
BORDER = sv.NOOP_GUARD_BORDER_WIDTH
EXTRA_RESULT_KEYS = ('reverted', 'interior_transient', 'v61_level_start', 'v61_edge', 'v61_hit')


def _sym(v):
    return ARC_COLOR_CHARS[v] if 0 <= v < len(ARC_COLOR_CHARS) else str(v)


def _interior(grid, border=BORDER):
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    if border <= 0 or rows <= 2 * border or cols <= 2 * border:
        return tuple(tuple(r) for r in grid)
    return tuple(tuple(r[border:cols - border]) for r in grid[border:rows - border])


def _interior_transient(chain, rows, cols, border=BORDER):
    return [(r, c) for r, c in _transient_cells(chain)
            if border <= r < rows - border and border <= c < cols - border]


def level_start_label(start, before, after):
    """2c: 'restart' when the whole frame returned to the level start and the edge band (the bar) had differed from
    the start's before the action; 'soft' when only the board area returned (or the frame matches in a game whose
    edge never differed, e.g. a walk back in a game without a bar); else None. The caller has checked after != before."""
    if after == start and edge_delta(start, before):
        return 'restart'
    if _interior(after) == _interior(start) and _interior(before) != _interior(start):
        return 'soft'
    return None

def edge_delta(before, after, border=BORDER):
    """Cells changed inside the edge band: dict(n, tr=[[from, to, count]...], rows=[...], cols=[...]) or None."""
    rows = len(after)
    cols = len(after[0]) if rows else 0
    if not before or len(before) != rows or border <= 0:
        return None
    tr, rs, cs, n = Counter(), set(), set(), 0
    for r in range(rows):
        edge_row = r < border or r >= rows - border
        a, b = before[r], after[r]
        for c in range(cols):
            if (edge_row or c < border or c >= cols - border) and a[c] != b[c]:
                n += 1
                tr[(_sym(a[c]), _sym(b[c]))] += 1
                rs.add(r)
                cs.add(c)
    if not n:
        return None
    return dict(n=n, tr=[[f, t, k] for (f, t), k in tr.most_common(3)], rows=sorted(rs)[:6], cols=sorted(cs)[:6],
                nrows=len(rs), ncols=len(cs))


def component_at(grid, row, col):
    """(id, colour value, cells) of the 4-connected same-colour component containing (row, col); ids follow
    segmentation.segment_layer (row-major order of each component's first cell)."""
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    if not (0 <= row < rows and 0 <= col < cols):
        return None
    label = [[-1] * cols for _ in range(rows)]
    nid = 0
    for sr in range(rows):
        for sc in range(cols):
            if label[sr][sc] != -1:
                continue
            v = grid[sr][sc]
            cells = [(sr, sc)]
            label[sr][sc] = nid
            stack = [(sr, sc)]
            while stack:
                r, c = stack.pop()
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if 0 <= nr < rows and 0 <= nc < cols and label[nr][nc] == -1 and grid[nr][nc] == v:
                        label[nr][nc] = nid
                        stack.append((nr, nc))
                        cells.append((nr, nc))
            if label[row][col] == nid:
                return nid, v, cells
            nid += 1
    return None


def mouse_hit(grid, row, col, border=BORDER):
    found = component_at(grid, row, col)
    if found is None:
        return dict(kind='outside')
    cid, v, cells = found
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    r0, r1 = min(r for r, _ in cells), max(r for r, _ in cells)
    c0, c1 = min(c for _, c in cells), max(c for _, c in cells)
    in_edge = row < border or row >= rows - border or col < border or col >= cols - border
    kind = 'background' if len(cells) * 4 >= rows * cols else 'component'
    return dict(kind=kind, id=cid, color=_sym(v), cells=len(cells), bbox=[r0, c0, r1 + 1, c1 + 1], edge=in_edge)


def hit_text(hit):
    if not hit:
        return ''
    if hit.get('kind') == 'outside':
        return 'hit: outside the board'
    where = ' in the edge band' if hit.get('edge') else ''
    if hit['kind'] == 'background':
        return 'hit: background (%s, %d cells)%s' % (hit['color'], hit['cells'], where)
    r0, c0, r1, c1 = hit['bbox']
    return 'hit: component %d of the board before it (%s, %d cells, rows %d-%d, cols %d-%d)%s' % (
        hit['id'], hit['color'], hit['cells'], r0, r1 - 1, c0, c1 - 1, where)


def execute_action(self, action, *, batch_index, batch_size, generated_tokens=None, flush_viewer_payload=True,
                   automatic=False):
    """His _execute_action with the v6.1 labels computed before the history entry is written."""
    previous_grid = sv._grid_from_state(self.game.current_state)
    previous_completed = int(self.game.current_state.levels_completed)
    if 'v61_level_start' not in self.__dict__:
        self.__dict__['v61_level_start'] = previous_grid            # the game's first frame
    if generated_tokens is None:
        current_tokens = sv._analyzer_reported_tokens(self.analyzer)
        generated_tokens = max(0, current_tokens - self.token_baseline)
        self.token_baseline = current_tokens

    new_state = self.game.execute_action(action, generated_tokens=generated_tokens, uncached_input_tokens=0)
    self.last_engine_action = action.id.name
    action_display = sv._format_action_display(action.id.name, dict(action.data))
    current_frame = sv.Frame(grid=sv._grid_from_state(new_state), step=self.action_count,
                             level=sv._level_number(self.game))

    completed = int(new_state.levels_completed)
    reward = float(completed - previous_completed) / max(1.0, float(self.game.number_of_levels))
    raw_state = new_state.raw.state
    new_grid = sv._grid_from_state(new_state)
    board_changed = previous_grid != new_grid
    gameplay_changed = sv._interior_gameplay_changed(previous_grid, new_grid)
    no_op = gameplay_changed is False
    animation_summary = None
    chain = None
    if sv.ANIMATION_ENABLED and action.id.name != "RESET":
        chain = sv._animation_chain(previous_grid, new_state)
        animation_summary = sv.summarize_animation(chain)
        self.animation_record = (
            {"action_display": action_display, "chain": chain, "summary": animation_summary,
             "step": self.action_count, "level": sv._level_number(self.game)}
            if animation_summary else None
        )
    level_completed = bool(new_state.just_won_level and raw_state != arcengine.GameState.WIN)
    game_over = raw_state == arcengine.GameState.GAME_OVER
    payload = {
        "executed": True,
        "action_num": self.action_count,
        "level": sv._level_number(self.game),
        "score": completed,
        "reward": reward,
        "state": raw_state.name,
        "valid_actions": sv.to_model_actions(sv._engine_action_names(self.game)),
        "board_changed": board_changed,
        "gameplay_changed": gameplay_changed,
        "no_op": no_op,
        "done": raw_state == arcengine.GameState.WIN,
        "level_completed": level_completed,
        "game_over": game_over,
        "run_complete": raw_state == arcengine.GameState.WIN,
        "action_name": action.id.name,
        "action_data": (sv._model_mouse_action_data(action.data) if action.id == arcengine.GameAction.ACTION6
                        else dict(action.data)),
        "action_display": action_display,
        "batch_index": batch_index,
        "batch_size": batch_size,
        **self.timing_payload(),
    }
    if animation_summary:
        payload["animation"] = animation_summary
        payload["animation_chain"] = chain
    payload["automatic"] = automatic

    # ---- v6.1 labels ------------------------------------------------------------------------------------------
    rows = len(new_grid)
    cols = len(new_grid[0]) if rows else 0
    terminal = game_over or level_completed or payload["run_complete"]
    if chain and gameplay_changed is False and not terminal:                       # 2b
        inner = _interior_transient(chain, rows, cols)
        if inner:
            payload["reverted"] = True
            payload["interior_transient"] = len(inner)
            payload["no_op"] = False
    start = self.__dict__.get('v61_level_start')
    if action.id.name == 'RESET' or level_completed:
        self.__dict__['v61_level_start'] = new_grid                                 # a new level / attempt start
    elif (not terminal and not automatic and action.id.name not in ('ACTION7',) and start is not None
          and new_grid != previous_grid):                                           # 2c
        label = level_start_label(start, previous_grid, new_grid)
        if label:
            payload["v61_level_start"] = label
    if not automatic and action.id.name != 'RESET':
        edge = edge_delta(previous_grid, new_grid)                                  # 3
        if edge:
            payload["v61_edge"] = edge
    if action.id == arcengine.GameAction.ACTION6:                                    # 5
        data = sv._model_mouse_action_data(action.data)
        payload["v61_hit"] = mouse_hit(previous_grid, int(data.get("row", -1)), int(data.get("col", -1)))
    # -----------------------------------------------------------------------------------------------------------

    result_keys = (
        "executed", "action_num", "level", "score", "state", "valid_actions",
        "board_changed", "gameplay_changed", "no_op", "done", "level_completed",
        "game_over", "run_complete", "action_name", "action_data", "action_display",
        "animation", "automatic",
    ) + EXTRA_RESULT_KEYS
    individual_result = copy.deepcopy({k: payload[k] for k in result_keys if k in payload})
    self.history_entries.append(sv.HistoryEntry(action=action_display, frame=current_frame, result=individual_result))
    self.write_runtime_state()
    self._append_action_viewer_event(payload, current_frame)
    if flush_viewer_payload:
        self.write_viewer_payload()
    return payload


STOP_REASONS_V61 = ('reverted', 'level_restarted', 'level_start', 'call_cap')


def step_env(self, arguments):
    """His step_env with the v6.1 batch stops, the per-call cap and a stable result schema."""
    if str(arguments.get("query") or "").strip() == "animation":
        return {"executed": False, "query": "animation", "record": self.animation_record if sv.ANIMATION_ENABLED else None}
    requested_actions, error = self._normalize_actions(arguments)
    if error is not None or requested_actions is None:
        return _stable(self._error_payload(error or "Could not parse action request."))
    if self.should_stop() or sv._is_engine_game_over(self.game):
        return _stable(self._terminal_payload(requested_actions))

    executed_payloads: list[dict[str, Any]] = []
    total_reward = 0.0
    stop_reason: str | None = None
    batch_size = len(requested_actions)
    requested_displays = [sv._format_action_display(a.id.name, dict(a.data)) for a in requested_actions]

    guard_hook = getattr(self, "action_guard_hook", None)
    blocked_detail: str | None = None
    blocked_kind: str | None = None
    stale_after = (arguments.get("stale_after")
                   if sv.STALE_STATE_BLOCK and sv._guards_active(sv._level_number(self.game)) else None)

    allowance = None                                                                # 2d
    if CALL_CAP > 0:
        allowance = CALL_CAP - int(self.__dict__.get('v61_call_executed', 0))
        if allowance <= 0:
            detail = ("This call already executed %d actions, the per-call limit; nothing more was executed. End "
                      "this call: the next turn continues from the current board." % CALL_CAP)
            return _stable({**self._error_payload(detail), "executed": False, "stop_reason": "call_cap",
                            "stop_detail": detail, "requested_count": batch_size, "executed_count": 0,
                            "stopped_early": True, "requested_actions": requested_displays,
                            "skipped_actions": list(requested_displays)})
    revert_run = int(self.__dict__.get('v61_call_revert_run', 0))

    for batch_index, action in enumerate(requested_actions, start=1):
        if self.should_stop():
            stop_reason = "stopped"
            break
        if allowance is not None and len(executed_payloads) >= allowance:
            stop_reason = "call_cap"
            break
        if stale_after and batch_index == 1 and sv._stale_repeat(stale_after, action):
            blocked_kind = "stale_state"
            blocked_detail = sv._stale_detail(action, stale_after)
            stop_reason = blocked_kind
            break
        if guard_hook is not None:
            verdict = guard_hook(
                sv._grid_from_state(self.game.current_state), action.id.name, sv._guard_action_data(action),
                sv._level_number(self.game),
                [(later.id.name, sv._guard_action_data(later)) for later in requested_actions[batch_index - 1:]],
            )
            if verdict:
                blocked_kind, blocked_detail = verdict
                stop_reason = blocked_kind
                break
        if stale_after and batch_index == 1:
            blocked_kind = "stale_state"
            blocked_detail = sv._stale_detail(action, stale_after)
            stop_reason = blocked_kind
            break
        stale_after = None
        if action.id.value not in self.game.current_state.available_actions:
            message = f"{sv._format_action_display(action.id.name, dict(action.data))} is not valid right now."
            if executed_payloads:
                stop_reason = "invalid_action"
                break
            return _stable({**self._error_payload(message), "stop_reason": "invalid_action",
                            "requested_actions": requested_displays, "skipped_actions": list(requested_displays)})
        try:
            payload = self._execute_action(action, batch_index=batch_index, batch_size=batch_size,
                                           flush_viewer_payload=False)
        except Exception as exc:
            if executed_payloads:
                stop_reason = "action_error"
                break
            return _stable(self._error_payload(f"{type(exc).__name__}: {exc}"))
        executed_payloads.append(payload)
        self.__dict__['v61_call_executed'] = int(self.__dict__.get('v61_call_executed', 0)) + 1
        total_reward += float(payload.get("reward", 0.0) or 0.0)

        if payload.get("run_complete"):
            stop_reason = "run_complete"
            break
        if payload.get("game_over"):
            stop_reason = "game_over"
            break
        if payload.get("level_completed"):
            stop_reason = "level_completed"
            break
        lvl_start = payload.get("v61_level_start")
        if lvl_start:                                                                # 2c
            stop_reason = "level_restarted" if lvl_start == 'restart' else "level_start"
            break
        if payload.get("reverted"):                                                  # 2b
            revert_run += 1
            if revert_run >= 2:
                stop_reason = "reverted"
                break
            continue
        revert_run = 0
        if (sv.BATCH_NOOP_BLOCK and sv.NOOP_GUARD_BORDER_WIDTH > 0 and batch_size >= 2
                and not payload.get("gameplay_changed", True)):
            stop_reason = "no_op_action"
            break
    self.__dict__['v61_call_revert_run'] = revert_run
    if stop_reason is None and allowance is not None and len(executed_payloads) < batch_size \
            and len(executed_payloads) >= allowance:
        stop_reason = "call_cap"

    if not executed_payloads:
        if blocked_detail is not None:
            return _stable({
                **self._error_payload(blocked_detail), "executed": False, "stop_reason": blocked_kind,
                "stop_detail": blocked_detail, "requested_count": batch_size, "executed_count": 0,
                "stopped_early": True, "requested_actions": requested_displays,
                "skipped_actions": list(requested_displays),
                "blocked_action": requested_displays[0] if requested_displays else None,
            })
        return _stable(self._error_payload("No action was executed."))

    final_payload = dict(executed_payloads[-1])
    final_payload["reward"] = total_reward
    final_payload["last_reward"] = executed_payloads[-1].get("reward", 0.0)
    final_payload["batched"] = batch_size > 1
    final_payload["requested_count"] = batch_size
    final_payload["executed_count"] = len(executed_payloads)
    final_payload["requested_actions"] = requested_displays
    final_payload["executed_actions"] = [str(i.get("action_display") or i.get("action_name") or "")
                                         for i in executed_payloads]
    final_payload["board_changed"] = any(bool(i.get("board_changed")) for i in executed_payloads)
    final_payload["stopped_early"] = len(executed_payloads) < batch_size
    final_payload["stop_reason"] = stop_reason
    if final_payload["stopped_early"]:
        final_payload["skipped_actions"] = requested_displays[len(executed_payloads):]
    for item in executed_payloads:
        if item.get("game_over"):
            final_payload["fatal_action"] = str(item.get("action_display") or "")
            break
    final_payload["action_trace"] = sv._build_action_trace(requested_displays, executed_payloads, blocked_kind)
    final_payload["gameplay_changed_per_action"] = [bool(i.get("gameplay_changed", True)) for i in executed_payloads]
    final_payload["reverted_per_action"] = [bool(i.get("reverted")) for i in executed_payloads]
    self.__dict__.setdefault('v61_call_payloads', []).extend(
        {k: i[k] for k in ("action_display", "gameplay_changed", "reverted", "v61_edge", "v61_hit", "v61_level_start",
                           "game_over", "automatic") if k in i} for i in executed_payloads)   # read by the agent
    if sv.ACTION_ECHO:
        echo = sv._build_action_echo(requested_displays, executed_payloads, stop_reason)
        if echo:
            final_payload["action_echo"] = echo
    if blocked_detail is not None:
        final_payload["stop_detail"] = blocked_detail
        final_payload["blocked_action"] = requested_displays[len(executed_payloads)]
    if stop_reason == "no_op_action":
        noop_index = len(executed_payloads)
        final_payload["no_op_action_index"] = noop_index
        final_payload["no_op_action"] = final_payload["executed_actions"][-1]
        skipped = requested_displays[noop_index:]
        final_payload["stop_detail"] = (
            f"Action {noop_index} of {batch_size} ({final_payload['executed_actions'][-1]!r}) EXECUTED but produced no "
            f"change inside the board area - only the outer {sv.NOOP_GUARD_BORDER_WIDTH} cells, where a timer or "
            "remaining-steps bar usually sits, could have moved. "
            f"Actions {noop_index + 1}-{batch_size} ({skipped!r}) were SKIPPED and did not execute. Indices are 1-based "
            "positions in your submitted list. "
            f"Net effect: only the first {max(0, noop_index - 1)} executed action(s) changed the board; the final "
            "executed action moved nothing. Do NOT update your position estimate by counting submitted moves - "
            "re-locate the player from current_frame before planning. The per-action verdicts are listed in "
            "action_trace. This stop is about one action from one board state, not about batch size - batching is "
            "still the right approach, and the stop is what keeps a wrong movement model from costing the whole sequence."
        )
    elif stop_reason in STOP_REASONS_V61:
        skipped = requested_displays[len(executed_payloads):]
        final_payload["stop_detail"] = _v61_stop_detail(stop_reason, final_payload["executed_actions"], skipped)
    self.write_viewer_payload()
    return final_payload


def _v61_stop_detail(reason, executed, skipped):
    tail = (" The remaining %d action(s) were not executed: %s." % (len(skipped), ', '.join(skipped))) if skipped else ''
    if reason == 'reverted':
        return ("Stopped after two REVERTED actions in a row (%s): each animated and left the board as it was, which "
                "usually means those moves are blocked.%s" % (', '.join(executed[-2:]), tail))
    if reason == 'level_restarted':
        return ("%s ended the attempt without a game-over flag: the whole board, edge included, returned to this "
                "level's starting state.%s" % (executed[-1], tail))
    if reason == 'level_start':
        return ("After %s the board area matches this level's starting board (a return to the start, or a restart "
                "the edge does not show).%s" % (executed[-1], tail))
    return "This call reached the limit of %d executed actions.%s" % (CALL_CAP, tail)


def _stable(payload):
    """2g: the keys every action(...) result carries."""
    payload.setdefault("executed_actions", [])
    payload.setdefault("action_trace", [])
    payload.setdefault("gameplay_changed_per_action", [])
    payload.setdefault("stop_reason", None)
    return payload


def install():
    sv._HarnessGameSession.step_env = step_env
    return execute_action          # the base the v61 patches wrapper calls
