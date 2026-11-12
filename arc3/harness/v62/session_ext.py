"""Harness v6.1 game-session layer: modified copies of dfranzen's _HarnessGameSession._execute_action and .step_env
(src/inference/framework/solver.py @ 10882e3, Apache-2.0, see LICENSE.dfranzen), installed by v61/patches.py.
Items (HARNESS_REVISION_ITEMS_2026-10-05.md):
  2b REVERTED: an executed action whose animation had transient interior cells and whose settled interior equals the
     board before it is labelled reverted (gameplay_changed False, no_op False). A batch stops at the 2nd REVERTED
     action in a row (stop_reason 'reverted'); a single revert neither stops the batch nor arms the stale-state flag.
  2c level start (v6.2 round 5, user 2026-10-05: no priors): when the board inside the edge band returns to the
     level's starting board (the first frame of the game, of each level, or after a RESET), the action carries the facts
     v62_at_start ('board' when the whole frame matches, else 'interior') and v62_restored (edge cells that returned to
     their starting values). Nothing stops, no attempt is counted. Earlier rounds inferred restarts from a refill
     (level_start_label, restored_spread; kept below for the record, unused).
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
import re
from collections import Counter
from typing import Any

import arcengine

from inference.framework import solver as sv
from inference.utils.animation import _transient_cells
from inference.utils.grid_utils import ARC_COLOR_CHARS

CALL_CAP = int(os.environ.get('ARC3_V61_CALL_CAP', '40'))
# v6.9 REX HIDE_RESET (decision 49; apply_rex69.py item R69HR): arm B refuses a model RESET and repeats a restart that ended in GAME
# OVER (_hr69_step); off (default): nothing changes
HIDE_RESET69 = os.environ.get('ARC3_V69_HIDE_RESET', '0') == '1'
V70H_RESET0 = os.environ.get('ARC3_V70H_RESET0', '0') == '1'   # v7.0 item 19c (apply_v70h.py): a RESET in a 0-action attempt
BORDER = sv.NOOP_GUARD_BORDER_WIDTH
V77 = os.environ.get('ARC3_V77', '0') == '1'   # v7.7 (apply_v77.py): board_shift only for a shift of the whole interior
REFILL_MIN = 4                     # v6.2: edge cells back toward the start in one action that mark a refill
SPREAD_MIN = 3                     # v6.2 round 3: a restart undoes the edge changes of >= 3 earlier actions at once
EXTRA_RESULT_KEYS = ('reverted', 'interior_transient', 'level_start_match', 'edge_cells_restored', 'repeat_in_call',
                     'v61_edge', 'v61_hit', 'interior_changed', 'near_level_start', 'actions_since_start', 'band_connected',
                     'start_return', 'board_shift', 'undo_restored')
SHIFT_MAX = 16                     # round 5l (C1, Fable 3.13): shifts tried, |dy|, |dx| <= 16
SHIFT_MIN_CELLS = 64               # below this many changed cells, frame_diff's `moved` already covers displacement
SHIFT_MIN_SHARE = 0.9              # the share of changed cells one shift must explain
UNDO_LOOKBACK = 200                # round 5l (C3): history entries searched for the board an UNDO restored
NEAR_START = 16                    # round 5d: interior cells from the level's starting board that count as "only N cells"
NEAR_REARM = 64                    # round 5f (Fable S1): the near line fires again only after the interior differed
                                   # from the starting board in more than this many cells (g014: 65 lines from a
                                   # sprite hovering across the 16-cell line)


def _sym(v):
    return ARC_COLOR_CHARS[v] if 0 <= v < len(ARC_COLOR_CHARS) else str(v)


def _interior(grid, border=BORDER):
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    if border <= 0 or rows <= 2 * border or cols <= 2 * border:
        return tuple(tuple(r) for r in grid)
    return tuple(tuple(r[border:cols - border]) for r in grid[border:rows - border])


def _interior_changes(a, b, border=BORDER):
    """Round 5d: (count, [r0, r1], [c0, c1]) of the interior cells that differ between two boards; None when their shapes
    differ."""
    rows = len(a)
    cols = len(a[0]) if rows else 0
    if len(b) != rows or (rows and len(b[0]) != cols):
        return None
    cells = [(r, c) for r in range(border, rows - border) for c in range(border, cols - border) if a[r][c] != b[r][c]]
    if not cells:
        return 0, None, None
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return len(cells), [min(rs), max(rs)], [min(cs), max(cs)]


def interior_regions(a, b, border=BORDER, limit=3, colors=2):
    """v7.7 (the user 10-24, lever 4a: per-action change clusters with their colours): the interior cells two boards differ in,
    split into 8-connected regions, largest first: (the first `limit` as dicts n, rows [r0, r1], cols [c0, c1], colors [[from,
    to, count], ...] with the `colors` most common colour changes as board symbols; the number of the other regions; their
    cells). A measurement, no reading of the game."""
    rows = len(a)
    cols = len(a[0]) if rows else 0
    if len(b) != rows or (rows and len(b[0]) != cols):
        return [], 0, 0

    def sym(v):                                          # a board value as its symbol (numpy integers included)
        try:
            i = int(v)
        except (TypeError, ValueError):
            return str(v)
        return ARC_COLOR_CHARS[i] if 0 <= i < len(ARC_COLOR_CHARS) else str(v)
    order = [(r, c) for r in range(border, rows - border) for c in range(border, cols - border) if a[r][c] != b[r][c]]
    left = set(order)
    regions = []
    for seed in order:                                   # row-major seeds: each region starts at its first cell
        if seed not in left:
            continue
        left.discard(seed)
        stack, cells = [seed], [seed]
        while stack:
            r, c = stack.pop()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    q = (r + dr, c + dc)
                    if q in left:
                        left.discard(q)
                        stack.append(q)
                        cells.append(q)
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        moves = Counter((sym(a[r][c]), sym(b[r][c])) for r, c in cells)
        regions.append(dict(n=len(cells), rows=[min(rs), max(rs)], cols=[min(cs), max(cs)],
                            colors=[[f, t, k] for (f, t), k in sorted(moves.items(), key=lambda kv: (-kv[1], kv[0]))[:colors]]))
    regions.sort(key=lambda x: (-x['n'], x['rows'][0], x['cols'][0]))
    rest = regions[limit:]
    return regions[:limit], len(rest), sum(x['n'] for x in rest)


def band_connected(a, b, border=BORDER):
    """Round 5h (G3, 386 sx25, sy25, rx1l, wy30, lx52, tx36; Fable: a fact, the split itself unchanged): the changed cells of
    the outer band that are edge-connected, through changed cells, to a changed interior cell (one changed region that
    crosses the band's boundary); 0 when none are, None when the boards' shapes differ."""
    rows = len(a)
    cols = len(a[0]) if rows else 0
    if len(b) != rows or (rows and len(b[0]) != cols):
        return None
    changed = {(r, c) for r in range(rows) for c in range(cols) if a[r][c] != b[r][c]}
    inner = lambda r, c: border <= r < rows - border and border <= c < cols - border
    seen = {rc for rc in changed if inner(*rc)}
    stack = list(seen)
    while stack:
        r, c = stack.pop()
        for q in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if q in changed and q not in seen:
                seen.add(q)
                stack.append(q)
    return sum(1 for rc in seen if not inner(*rc))


def board_shift(a, b, max_shift=SHIFT_MAX, min_cells=SHIFT_MIN_CELLS, min_share=SHIFT_MIN_SHARE):
    """Round 5l (C1; 392b sp01, wr01, ng01, g305, g306: scrolling views reported only as large changed-cell counts; Fable
    3.13): when one translation of the board content explains at least `min_share` of the changed cells (at least
    `min_cells` of them), dict(dy, dx, explained, changed): new[r][c] == old[r - dy][c - dx] for the explained cells (dy
    down, dx right). A measurement, no reading of the game; None otherwise."""
    try:
        import numpy as np
        from numpy.lib.stride_tricks import sliding_window_view
        A = np.asarray(a, dtype=np.int16)
        B = np.asarray(b, dtype=np.int16)
    except Exception:
        return None
    if A.ndim != 2 or A.shape != B.shape:
        return None
    mask = A != B
    n = int(mask.sum())
    if n < min_cells:
        return None
    h, w = A.shape
    s = int(max_shift)
    padded = np.full((h + 2 * s, w + 2 * s), -1, dtype=np.int16)
    padded[s:s + h, s:s + w] = A
    view = sliding_window_view(padded, (h, w))               # view[i, j][r, c] = A[r + i - s, c + j - s]
    counts = ((view == B) & mask).sum(axis=(2, 3))
    counts[s, s] = 0
    i, j = np.unravel_index(int(counts.argmax()), counts.shape)
    k = int(counts[i, j])
    if k < min_share * n:
        return None
    if V77:                     # v7.7 (the user 10-24: "Apply fixes"; review B3, wa30:5): the rest of the interior must agree with the
        dy, dx = int(s - i), int(s - j)              # shift too, where both a cell and its source lie in the interior (at most
        bi = int(BORDER)                             # (1 - min_share) * n disagree), else a part of the board moved: None
        Ai, Bi = A[bi:h - bi, bi:w - bi], B[bi:h - bi, bi:w - bi]
        hh, ww = Ai.shape
        r0, r1, c0, c1 = max(0, dy), min(hh, hh + dy), max(0, dx), min(ww, ww + dx)
        if r1 <= r0 or c1 <= c0:
            return None
        if int((Ai[r0 - dy:r1 - dy, c0 - dx:c1 - dx] != Bi[r0:r1, c0:c1]).sum()) > (1.0 - min_share) * n:
            return None
    return dict(dy=int(s - i), dx=int(s - j), explained=k, changed=n)


def undo_restored(entries, new_grid, lookback=UNDO_LOOKBACK):
    """Round 5l (C3, 392b wk01: ~3 turns and a RESET to find what an UNDO restored): the latest earlier board of the current
    attempt equal to the board after the UNDO: dict(step, action) of the action that board preceded, else None."""
    rows = list(entries or [])
    last = len(rows) - 1                                         # rows[-1] is the board the UNDO acted on
    for j in range(last - 1, max(-1, last - 1 - lookback), -1):
        r = getattr(rows[j + 1], 'result', None) or {}
        if r.get('action_name') == 'RESET' or r.get('automatic') or r.get('level_completed'):
            break                                                # the attempt began after rows[j]
        frame = getattr(rows[j], 'frame', None)
        if frame is not None and getattr(frame, 'grid', None) == new_grid:
            nxt = rows[j + 1]
            return dict(step=getattr(getattr(nxt, 'frame', None), 'step', None), action=str(getattr(nxt, 'action', '') or '?'))
    return None


def _start_facts(d, payload, restart, automatic, terminal, new_grid, start):
    """Round 5d: `actions_since_start` on the start-match and near-start lines, and `near_level_start` (0 < N <=
    NEAR_START interior cells from the level's starting board; round 5f: once per excursion of more than NEAR_REARM cells). Round 5e (381 review): the count
    runs from the last board whose interior equals the level's starting board, or from the RESET / level start, as its
    text says; a near line used to restart it, so a later line counted from that line."""
    if restart:
        d['v62_since_start'] = 0
        d['v62_start_diff'] = 0
        d['v62_near_armed'] = False
        d['v62_far_armed'] = False
        return
    if automatic:
        return
    d['v62_since_start'] = d.get('v62_since_start', 0) + 1
    if terminal:
        return
    ch = _interior_changes(new_grid, start)
    diff = ch[0] if ch else None
    if payload.get("level_start_match"):
        payload["actions_since_start"] = d['v62_since_start']
    elif diff is not None and 0 < diff <= NEAR_START and d.get('v62_near_armed'):
        payload["near_level_start"] = diff
        payload["actions_since_start"] = d['v62_since_start']
        d['v62_near_armed'] = False
    if diff is not None and diff > NEAR_REARM:
        d['v62_near_armed'] = True               # round 5f: armed again only by a real excursion
        d['v62_far_armed'] = True                # round 5l (A5): cleared only by a return to the start or a RESET
    if diff == 0 or payload.get("level_start_match"):
        d['v62_since_start'] = 0
        d['v62_near_armed'] = False
        d['v62_far_armed'] = False
    d['v62_start_diff'] = diff


def _interior_transient(chain, rows, cols, border=BORDER):
    return [(r, c) for r, c in _transient_cells(chain)
            if border <= r < rows - border and border <= c < cols - border]


def level_start_label(start, before, after, spread=None):
    """2c: 'restart' when the whole frame returned to the level start and the edge band (the bar) had differed from
    the start's before the action; 'soft' when only the board area returned (or the frame matches in a game whose
    edge never differed, e.g. a walk back in a game without a bar); else None. The caller has checked after != before.
    v6.2 round 3 (A2: 11 false LEVEL RESTARTED in vc33, lf52, re86, tr87, m0r0 and ar25, where a press undid the
    previous one, a piece or cursor left the band again, or a walk ended back at the start): a restart also needs
    `spread` >= SPREAD_MIN, the number of distinct earlier actions whose edge-band changes this action undid. A restart
    refills what many moves used up at once (A2 replay: ls20 21-43, g50t 4-5); an undo or a walk back restores what one
    or two actions changed (1-2 in all 11)."""
    many = spread is not None and spread >= SPREAD_MIN
    if after == start and edge_delta(start, before) and many:
        return 'restart'
    if _interior(after) == _interior(start) and _interior(before) != _interior(start):
        # v6.2 fix (run A, ls20: 12 budget deaths labelled soft): an edge band that moves back toward its start state by
        # REFILL_MIN cells or more in this action (a bar refill) marks a restart, even when another edge cell (lives, a
        # counter) keeps the frame from matching exactly
        d0 = (edge_delta(start, before) or {}).get('n', 0)
        d1 = (edge_delta(start, after) or {}).get('n', 0)
        if d0 - d1 >= REFILL_MIN and many:
            return 'restart'
        return 'soft'
    return None


def _band(rows, cols, border=BORDER):
    for r in range(rows):
        for c in (range(cols) if (r < border or r >= rows - border)
                  else tuple(range(min(border, cols))) + tuple(range(max(border, cols - border), cols))):
            yield r, c


def restored_spread(start, before, after, last):
    """The number of distinct earlier actions (indices in `last`, edge cell -> the last action that changed it on this
    level) whose changes to the edge cells this action returned to their level-start values; None when the frames'
    shapes differ."""
    try:
        rows, cols = len(after), len(after[0]) if after else 0
        if (len(start), len(before)) != (rows, rows) or (rows and (len(start[0]), len(before[0])) != (cols, cols)):
            return None
        return len({last.get(rc) for rc in _band(rows, cols)
                    if before[rc[0]][rc[1]] != start[rc[0]][rc[1]] and after[rc[0]][rc[1]] == start[rc[0]][rc[1]]}
                   - {None})
    except (IndexError, TypeError):
        return None


def note_edge_changes(last, k, before, after):
    """Record action k as the last change of every edge cell it changed."""
    try:
        rows, cols = len(after), len(after[0]) if after else 0
        if len(before) != rows or (rows and len(before[0]) != cols):
            return
        for r, c in _band(rows, cols):
            if before[r][c] != after[r][c]:
                last[(r, c)] = k
    except (IndexError, TypeError):
        pass

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


def _enclosing(grid, row, col, exclude=None, max_box=None, any_size=False):
    """v6.2: the smallest non-background component whose bounding box contains (row, col): (area, id, value, cells,
    bbox). v6.2 round 3: `exclude` skips the clicked component; `max_box` bounds the box area. Round 5c: `any_size`
    drops the 1/4-of-the-board filter (the relative rule in mouse_hit needs no threshold)."""
    rows = len(grid)
    cols = len(grid[0]) if rows else 0
    label = [[-1] * cols for _ in range(rows)]
    best, nid = None, 0
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
            if any_size or len(cells) * 4 < rows * cols:
                r0, r1 = min(r for r, _ in cells), max(r for r, _ in cells)
                c0, c1 = min(c for _, c in cells), max(c for _, c in cells)
                if r0 <= row <= r1 and c0 <= col <= c1 and nid != exclude:
                    area = (r1 - r0 + 1) * (c1 - c0 + 1)
                    if (max_box is None or area <= max_box) and (best is None or area < best[0]):
                        best = (area, nid, v, len(cells), [r0, c0, r1 + 1, c1 + 1])
            nid += 1
    return best


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
    hit = dict(kind=kind, id=cid, color=_sym(v), cells=len(cells), bbox=[r0, c0, r1 + 1, c1 + 1], edge=in_edge)
    # v6.2 (b): run A r11l, a click on background inside an object's box. Round 5c (user: add the relative box fact; B3
    # r11l: 17 of 19 NO-OP clicks fell near a moving token, the hit text naming only the floor; ~40 actions): the
    # component with the smallest bounding box containing the click, when that box is smaller than the hit component's
    # box. A relative rule, no size threshold (round 3's 1/4 and 1/16 were chosen on public games).
    enc = _enclosing(grid, row, col, exclude=cid, max_box=(r1 - r0 + 1) * (c1 - c0 + 1) - 1, any_size=True)
    if enc:
        hit['inside'] = dict(id=enc[1], color=_sym(enc[2]), cells=enc[3], bbox=enc[4])
    return hit


def hit_text(hit):
    if not hit:
        return ''
    if hit.get('kind') == 'outside':
        return 'hit: outside the board'
    # v6.2 round 5b (user: no priors; audit item 3): the component hit, as measured, for every click. No "background"
    # label (a >= 25% share) and no "inside the bounding box" clause (thresholds chosen on public games); no band tag.
    r0, c0, r1, c1 = hit['bbox']
    text = 'hit: component %d of the board before it (%s, %d cells, rows %d-%d, cols %d-%d)' % (
        hit['id'], hit['color'], hit['cells'], r0, r1 - 1, c0, c1 - 1)
    inside = hit.get('inside')
    if inside:                                        # round 5c: the relative box fact (see mouse_hit)
        q0, d0, q1, d1 = inside['bbox']
        text += '; inside the bounding box of component %d (%s, %d cells, rows %d-%d, cols %d-%d)' % (
            inside['id'], inside['color'], inside['cells'], q0, q1 - 1, d0, d1 - 1)
    return text


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
    # round 5h (Fable Q4.2, 386 fy09 T95): the action that wins the game completes its last level too
    level_completed = bool(new_state.just_won_level or raw_state == arcengine.GameState.WIN)
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
    start = self.__dict__.get('v61_level_start')                  # the level's (or the latest RESET's) first board
    if action.id.name == 'RESET' or level_completed:
        self.__dict__['v61_level_start'] = new_grid                                 # a new level / attempt start
    elif (not terminal and not automatic and start is not None and new_grid != previous_grid
          and _interior(new_grid) == _interior(start) and _interior(previous_grid) != _interior(start)):
        # v6.2 round 5 (user 2026-10-05 ~13:50: no priors; "a game may require going back to the start"): facts only.
        # The board inside the edge band (or the whole board) now matches the level's starting board, and how many edge
        # cells returned to their starting values. No stop, no "attempt ended", no attempt counted.
        payload["level_start_match"] = 'full' if new_grid == start else 'interior'
        if new_grid == start and self.__dict__.get('v62_far_armed'):
            # round 5l (item 109, A5; 392b g019: >= 17 silent resets inside calls, >= 63 actions then ran on the fresh
            # level; ws03 misread it): the whole board is back at the level's starting board without a RESET, after the
            # interior had differed from it in more than NEAR_REARM cells. A fact and a batch stop, not an attempt boundary
            payload["start_return"] = True
        rows_, cols_ = len(new_grid), len(new_grid[0]) if new_grid else 0
        try:
            payload["edge_cells_restored"] = sum(1 for r, c in _band(rows_, cols_)
                                                 if previous_grid[r][c] != start[r][c] and new_grid[r][c] == start[r][c])
        except (IndexError, TypeError):
            pass
    # round 5d (user 2026-10-05, C50 review): facts for ambient change and for returns to the level's start without a
    # GAME OVER. `interior_changed`: how many interior cells this action changed and where (g001, g040, g136: a ticker or
    # blinkers made every action "effect"). `near_level_start`: the interior is only N cells from the level's starting
    # board after being further (g019, g006: a counter kept the exact match from firing). `actions_since_start`: the
    # actions since the board was last at the level's start, on either fact (g016: a 44-action budget).
    if not automatic:                    # round 5h (G4): on every executed action, n = 0 when nothing changed
        ch = _interior_changes(previous_grid, new_grid) if gameplay_changed else None
        if ch and ch[0]:
            payload["interior_changed"] = dict(n=ch[0], rows=ch[1], cols=ch[2])
            if V77:                      # v7.7 (lever 4a): the separate changed regions with their colour changes
                regs, more, more_cells = interior_regions(previous_grid, new_grid)
                payload["interior_changed"]["regions"] = regs
                if more:
                    payload["interior_changed"].update(more=more, more_cells=more_cells)
            if not terminal and action.id.name != 'RESET' and edge_delta(previous_grid, new_grid):
                k = band_connected(previous_grid, new_grid)                         # round 5h (G3)
                if k:
                    payload["band_connected"] = k
        else:
            payload["interior_changed"] = dict(n=0, rows=None, cols=None)
    _start_facts(self.__dict__, payload, action.id.name == 'RESET' or level_completed or start is None, automatic,
                 terminal, new_grid, start)
    if not automatic and action.id.name != 'RESET':
        # v6.2 round 5b (user + Fable: the repeat-state refusal is off; its detection stays as a fact a loop can test):
        # how many times this Python call already executed this action from this board (compared inside the edge band)
        seen = self.__dict__.setdefault('v62_call_seen', {})
        key = (_interior(previous_grid), action_display)
        if seen.get(key):
            payload["repeat_in_call"] = seen[key]
        seen[key] = seen.get(key, 0) + 1
    if not automatic and action.id.name != 'RESET':
        edge = edge_delta(previous_grid, new_grid)                                  # 3
        if edge:
            payload["v61_edge"] = edge
        if not terminal and board_changed:                                       # round 5l (C1)
            shift = board_shift(previous_grid, new_grid)
            if shift:
                payload["board_shift"] = shift
        if action.id.name == 'ACTION7' and board_changed and not terminal:      # round 5l (C3): what UNDO restored
            back = undo_restored(self.history_entries, new_grid)
            if back:
                payload["undo_restored"] = back
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
        return _stable(self._error_payload(_position_text(self, error) or "Could not parse action request."))
    if self.should_stop():
        return _stable(self._terminal_payload(requested_actions))
    if HIDE_RESET69:                                  # v6.9 REX HIDE_RESET (arm B): before the GAME OVER refusal and every action
        refused = _hr69_step(self, requested_actions)
        if refused is not None:
            return _stable(refused)
    if sv._is_engine_game_over(self.game) and not (requested_actions and requested_actions[0].id.name == 'RESET'):
        # round 5l (A1, 392b rw01: ~1,980 turns executed nothing): a RESET that itself ended in GAME OVER gets no
        # automatic restart (the solver's loop guard), and every action used to be refused here, RESET included. The
        # engine resets in any state (arcengine base_game.perform_action handles RESET first), so a RESET as the call's
        # first action executes below; anything else is refused with the reason
        return _stable(game_over_refusal(self, requested_actions))
    if (V70H_RESET0 and requested_actions and requested_actions[0].id.name == 'RESET'   # v7.0 item 19c: the attempt has executed
            and not self.__dict__.get('v6_attempt_actions', 0) and not sv._is_engine_game_over(self.game)):   # no action
        return _stable(_v70h_reset0_refusal(self, requested_actions))

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
    v65 = self.__dict__.get('v65_stats')                      # v6.5 C6: the agent's per-game stats dict (receipts)
    if (not sv.STALE_STATE_BLOCK and v65 is not None and arguments.get("stale_after") and requested_actions
            and sv._guards_active(sv._level_number(self.game)) and not self.should_stop()):
        would = arguments.get("stale_after")                   # the arm the sandbox sent: counted, not refused
        if str(would).startswith('MOUSE') and not sv._stale_repeat(would, requested_actions[0]):
            would = None
        if would:
            v65['stale_would_block'] = v65.get('stale_would_block', 0) + 1
    noop_run = 0                                               # v6.5 C6: consecutive NO-OPs in this batch

    allowance = None                                                                # 2d
    if CALL_CAP > 0:
        allowance = CALL_CAP - int(self.__dict__.get('v61_call_executed', 0))
        if allowance <= 0:
            detail = ("Not executed: this Python call (the whole snippet) has already executed %d actions, the limit "
                      "for one Python call; the limit counts every action(...) call in the snippet together. End this "
                      "call: the next turn continues from the current board, and the next Python call may execute up "
                      "to %d actions again." % (CALL_CAP, CALL_CAP))
            return _stable({**self._error_payload(detail), "executed": False, "stop_reason": "call_cap",
                            "stop_detail": detail, "requested_count": batch_size, "executed_count": 0,
                            "stopped_early": True, "requested_actions": requested_displays,
                            "skipped_actions": list(requested_displays)})
    revert_run = int(self.__dict__.get('v61_call_revert_run', 0))
    revert_pair = None
    revert_prev = self.__dict__.get('v61_call_revert_prev')          # v6.2: the previous REVERTED action of this call

    # round 5h (Fable Q3, 386 by35: 12 x the same no-op click): after a click that changed nothing only the same click is
    # refused; the refusal names a REVERTED arming action as such (it said "no-op")
    if stale_after and str(stale_after).startswith('MOUSE') and requested_actions \
            and not sv._stale_repeat(stale_after, requested_actions[0]):
        stale_after = None
    last = self.history_entries[-1] if getattr(self, 'history_entries', None) else None
    stale_reverted = bool(stale_after and ((getattr(last, 'result', None) or {}).get('reverted')))

    for batch_index, action in enumerate(requested_actions, start=1):
        if self.should_stop():
            stop_reason = "stopped"
            break
        if allowance is not None and len(executed_payloads) >= allowance:
            stop_reason = "call_cap"
            break
        if stale_after and batch_index == 1 and sv._stale_repeat(stale_after, action):
            blocked_kind = "stale_state"
            blocked_detail = _c6r_stale_detail(action, stale_after, reverted=stale_reverted)
            if v65 is not None:                                # C6R receipt
                v65["stale_refusals"] = v65.get("stale_refusals", 0) + 1
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
            blocked_detail = _c6r_stale_detail(action, stale_after, reverted=stale_reverted)
            if v65 is not None:                                # C6R receipt
                v65["stale_refusals"] = v65.get("stale_refusals", 0) + 1
            stop_reason = blocked_kind
            break
        stale_after = None
        if action.id.value not in self.game.current_state.available_actions:
            # round 5l (B2, 392b: UNDO assumed available in most games without it; the prompt described it unconditionally)
            message = (f"{sv._format_action_display(action.id.name, dict(action.data))} is not among the valid actions in "
                       "this observation; nothing was executed.")
            try:                                   # round 4 (A2/A3 ls20: InvalidActionError named no valid actions)
                names = [n for n in sv.to_model_actions(sv._engine_action_names(self.game)) if n != 'RESET']
                if names:
                    message += " Valid actions now: %s." % ', '.join(names)
            except Exception:
                pass
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
        if payload.get("start_return"):                     # round 5l (item 109, A5): the game's own return to the start
            stop_reason = "level_start"
            break
        if payload.get("reverted"):                                                  # 2b
            noop_run = 0                                   # v6.5 C6: a REVERTED action breaks a NO-OP streak
            revert_run += 1
            shown = str(payload.get("action_display") or payload.get("action_name") or "?")
            if revert_run >= 2:
                stop_reason = "reverted"
                revert_pair = [revert_prev or "?", shown]
                revert_prev = shown       # round 5h (G7, 386 fx09: a run of 3-5 named its first and last): the last two
                break
            revert_prev = shown
            continue
        revert_run = 0
        revert_prev = None
        # round 5d (C50 g009 T71-T119 x4, g036 4 of 8 no-op stops, g046 T345, T479): a RESET on the level's starting
        # board changes nothing by definition; it no longer ends the batch it opens
        if (sv.BATCH_NOOP_BLOCK and sv.NOOP_GUARD_BORDER_WIDTH > 0 and batch_size >= 2
                and not payload.get("gameplay_changed", True) and payload.get("action_name") != "RESET"):
            if not _v65_batch():
                stop_reason = "no_op_action"
                if v65 is not None and batch_index < batch_size:      # C6R receipts: stops that skipped actions
                    key = 'band_only_stops' if payload.get("board_changed") else 'noop_stops'
                    v65[key] = v65.get(key, 0) + 1
                break
            if payload.get("board_changed"):              # v6.5 C6: a BAND-ONLY CHANGE does not stop the batch
                noop_run = 0
                if v65 is not None and batch_index < batch_size:
                    v65['band_only_continued'] = v65.get('band_only_continued', 0) + 1
            else:                                          # a NO-OP: the second in a row stops the batch
                noop_run += 1
                if noop_run >= 2:
                    stop_reason = "no_op_action"
                    if v65 is not None:
                        v65['noop_second_stops'] = v65.get('noop_second_stops', 0) + 1
                    break
                if v65 is not None and batch_index < batch_size:
                    v65['noop_first_continued'] = v65.get('noop_first_continued', 0) + 1
        else:
            noop_run = 0
    self.__dict__['v61_call_revert_run'] = revert_run
    if v65 is not None and executed_payloads:                 # v6.5 C6: executed actions per game (receipt)
        v65['executed_actions'] = v65.get('executed_actions', 0) + len(executed_payloads)
    self.__dict__['v61_call_revert_prev'] = revert_prev
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
    # round 5l (A3 option b): whether the LAST executed action changed any cell; the sandbox arms the stale refusal only
    # when it did not (board_changed above covers the whole call)
    final_payload["last_board_changed"] = bool(executed_payloads[-1].get("board_changed"))
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
    if revert_pair:
        final_payload["reverted_pair"] = revert_pair
    self.__dict__.setdefault('v61_call_payloads', []).extend(
        {k: i[k] for k in ("action_display", "board_changed", "gameplay_changed", "reverted", "v61_edge", "v61_hit",
                           "level_start_match",
                           "edge_cells_restored", "repeat_in_call", "interior_changed", "near_level_start",
                           "actions_since_start", "band_connected", "start_return", "board_shift", "undo_restored",
                           "game_over", "automatic", "level_completed", "run_complete", "done", "v6_attempt")
                           + (("action_num",) if V77 else ()) if k in i}   # v7.8 F1: the step of each action (V77)
        for i in executed_payloads)   # read by the agent (v6.2: also the per-action verdicts of the opener)
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
        band = (executed_payloads[-1].get('v61_edge') or {}).get('n') or 0
        k_before = max(0, noop_index - 1)
        final_payload["stop_detail"] = (_v65_stop_detail(final_payload, executed_payloads, noop_index, batch_size, skipped)
                                        if _v65_batch() else _c6r_stop_detail(final_payload, noop_index, batch_size,
                                                                              skipped, band, k_before))
    elif stop_reason in STOP_REASONS_V61:
        skipped = requested_displays[len(executed_payloads):]
        shown = list(revert_pair) if (stop_reason == 'reverted' and revert_pair) else final_payload["executed_actions"]
        final_payload["stop_detail"] = _v61_stop_detail(stop_reason, shown, skipped)
    self.write_viewer_payload()
    return final_payload


GAME_OVER_REFUSAL = ("Not executed: the game reports GAME OVER, and no automatic restart followed because the last "
                     "executed action was a RESET. RESET is the only action that executes now: put it first in the action "
                     "list.")


def game_over_refusal(session, requested_actions):
    """Round 5l (A1): the refusal of a non-RESET call while the game reports GAME OVER (his terminal payload, with the
    reason and the one action that executes)."""
    payload = session._terminal_payload(requested_actions)
    payload["error"] = GAME_OVER_REFUSAL
    payload["valid_actions"] = ["RESET"]
    payload["skipped_actions"] = [sv._format_action_display(a.id.name, dict(a.data)) for a in requested_actions]
    return payload


def _v70h_reset0_refusal(session, requested_actions):
    """v7.0 item 19c (ARC3_V70H_RESET0): a model action(...) call whose first action is RESET while the current attempt has executed
    no action (the level is at its starting state: the level's start, an automatic restart or a RESET just before): refused before
    anything executes, the call's other actions included (executed False, the text as `error`: the sandbox raises
    InvalidActionError as for an invalid action). Receipts in the agent's per-game stats (the session's v65_stats link)."""
    from v62 import variants as _va
    v65 = session.__dict__.get('v65_stats')
    shown = [sv._format_action_display(a.id.name, dict(a.data)) for a in requested_actions]
    if v65 is not None:
        v65['v70h_reset0_refusals'] = v65.get('v70h_reset0_refusals', 0) + 1
        try:
            lv = str(sv._level_number(session.game))
        except Exception:  # noqa: BLE001
            lv = '?'
        by = v65.setdefault('v70h_reset0_refusals_by_level', {})
        by[lv] = by.get(lv, 0) + 1
    text = (_va.V70H_RESET0_REFUSAL if len(requested_actions) == 1
            else _va.V70H_RESET0_REFUSAL_BATCH.replace('{n}', str(len(requested_actions))))
    return {**session._error_payload(text), "executed": False, "stop_reason": "reset_zero", "requested_count": len(requested_actions),
            "executed_count": 0, "stopped_early": True, "requested_actions": shown, "skipped_actions": list(shown)}


def _hr69_valid(session):
    """v6.9 REX HIDE_RESET: the valid actions as the model sees them (EXPOSE_RESET off: RESET is not among them)."""
    try:
        return [n for n in sv.to_model_actions(sv._engine_action_names(session.game)) if n != 'RESET']
    except Exception:  # noqa: BLE001
        return []


def _hr69_step(session, requested_actions):
    """v6.9 REX HIDE_RESET (arm B), a model action(...) call: (1) the engine reports GAME OVER (possible only after a RESET that
    itself ended in GAME OVER, which dfranzen's solver does not repeat; in arm B that RESET was the harness's): the harness repeats
    the automatic restart (the game rule's RESET) and refuses the call; (2) the call's list holds RESET: refused before anything
    executes, the call's other actions included; (3) otherwise None (the call goes on as in arm A). The refusal carries executed
    False and the text as `error`, so the sandbox raises InvalidActionError as for an invalid action. Receipts in the agent's
    per-game stats (the session's v65_stats link)."""
    from v62 import variants as _va
    v65 = session.__dict__.get('v65_stats')
    shown = [sv._format_action_display(a.id.name, dict(a.data)) for a in requested_actions]
    if sv._is_engine_game_over(session.game):
        session._execute_auto_reset()
        if v65 is not None:
            v65['reset_hidden_gameover_restarts'] = v65.get('reset_hidden_gameover_restarts', 0) + 1
        text, kind = _va.HR69_GAME_OVER_REFUSAL, 'hr69_game_over'
    else:
        n = sum(1 for a in requested_actions if a.id.name == 'RESET')
        if not n:
            return None
        if v65 is not None:
            v65['reset_hidden_refusals'] = v65.get('reset_hidden_refusals', 0) + 1
            v65['reset_hidden_resets'] = v65.get('reset_hidden_resets', 0) + n
            try:
                lv = str(sv._level_number(session.game))
            except Exception:  # noqa: BLE001
                lv = '?'
            by = v65.setdefault('reset_hidden_refusals_by_level', {})
            by[lv] = by.get(lv, 0) + 1
        text = (_va.HR69_RESET_REFUSAL if len(requested_actions) == 1
                else _va.HR69_RESET_REFUSAL_BATCH.replace('{n}', str(len(requested_actions))))
        names = _hr69_valid(session)
        if names:
            text += _va.HR69_VALID_NOW.replace('{valid}', ', '.join(names))
        kind = 'reset_hidden'
    return {**session._error_payload(text), "executed": False, "stop_reason": kind, "requested_count": len(requested_actions),
            "executed_count": 0, "stopped_early": True, "requested_actions": shown, "skipped_actions": list(shown)}


def _position_text(session, error):
    """Round 5h (G7, 386 lx20 x5, ly85: "Unknown action at index 2" read as a Python index, no valid names given): positions
    stay 1-based, as in the batch stops, and say so; an unknown action lists the valid ones."""
    if not isinstance(error, str):
        return error
    text = re.sub(r'^Unknown action at index (\d+): (.*)$', r'Unknown action at position \1 (1-based) of the list: \2.', error)
    text = re.sub(r'^MOUSE action at index (\d+) ', r'MOUSE action at position \1 (1-based) ', text)
    text = re.sub(r'^Action (\d+) must be a JSON object\.', r'The action at position \1 (1-based) must be a JSON object.', text)
    if text.startswith('Unknown action'):
        try:
            names = [n for n in sv.to_model_actions(sv._engine_action_names(session.game)) if n != 'RESET']
            if names:
                text += ' Valid actions now: %s.' % ', '.join(names)
        except Exception:
            pass
    return text


def _v61_stop_detail(reason, executed, skipped):
    tail = (" The remaining %d action(s) were not executed: %s." % (len(skipped), ', '.join(skipped))) if skipped else ''
    if reason == 'level_start':                    # round 5l (item 109, A5)
        return ("The whole board, outer band included, returned to this level's starting board after %s, with no GAME OVER "
                "reported and no RESET; the interior had differed from it in more than %d cells.%s"
                % (executed[-1] if executed else 'the last action', NEAR_REARM, tail))
    if reason == 'reverted':                       # round 5: facts only (no "a blocked move" reading)
        return ("Stopped after two REVERTED actions in a row (%s): each animated and then left the interior as it was.%s"
                % (', '.join(executed[-2:]), tail))
    return ("This Python call (the whole snippet) reached the limit of %d executed actions for one Python call; the limit "
            "counts every action(...) call in the snippet together, and the next Python call may execute up to %d "
            "again.%s" % (CALL_CAP, CALL_CAP, tail))


def _v65_batch():
    import os
    return os.environ.get('ARC3_V65_BATCH', '1') == '1'


def _v65_stop_detail(final_payload, executed_payloads, noop_index, batch_size, skipped):
    """v6.5 C6: the batch stopped at its second consecutive NO-OP; the interior count comes from the per-action verdicts
    (earlier actions may be BAND-ONLY changes or the first NO-OP)."""
    done = final_payload["executed_actions"]
    changed = sum(1 for i in executed_payloads if i.get("gameplay_changed", True))
    return (f"Actions {noop_index - 1} and {noop_index} of {batch_size} ({done[-2]!r}, {done[-1]!r}) both EXECUTED but "
            "changed nothing on the board (NO-OP twice in a row), so the batch stopped. "
            + (("Action %d (%r) was SKIPPED and did not execute. " % (noop_index + 1, skipped[0]) if len(skipped) == 1 else
                f"Actions {noop_index + 1}-{batch_size} ({skipped!r}) were SKIPPED and did not execute. ")
               + "Indices are 1-based positions in your submitted list. " if skipped else "")
            + (f"{changed} of the {noop_index} executed actions changed cells in the interior. " if changed else
               "No executed action of this batch changed anything in the interior. ")
            + "Re-read current_frame before planning rather than counting submitted actions. The per-action verdicts "
              "are listed in action_trace.")


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


def _c6r_stale_detail(action, stale_after, reverted=False):
    """C6R (user + IE, 2026-10-17): the stale refusal as a harness fact. It follows an action that changed nothing on the
    board (the sandbox arms it on last_board_changed False): a NO-OP, or a REVERTED action."""
    shown = sv._format_action_display(action.id.name, dict(action.data))
    verdict, rule = (("was REVERTED (it animated, then the board returned to its previous state)", "after a REVERTED action")
                     if reverted else ("was a NO-OP (nothing on the board changed)", "after a NO-OP"))
    return (f"{shown} was not executed. The previous action in this snippet, {stale_after}, {verdict}. The harness, not the "
            f"game, refuses the first action call {rule} in the same snippet, because positions and paths computed on the "
            f"assumption that {stale_after} had changed the board no longer describe it, and acting on them would waste "
            "actions. The refusal happens once; the next action call executes.")


def _c6r_stop_detail(final_payload, noop_index, batch_size, skipped, band, k_before):
    """C6R (user + IE, 2026-10-17): v6.4's first-no-change batch stop, stated as a harness fact; the harness-rule sentence
    only when actions were skipped."""
    act = final_payload['executed_actions'][-1]
    if band:
        head = (f"Action {noop_index} of {batch_size} ({act!r}) executed but changed only {band} cell{'' if band == 1 else 's'} "
                "of the outer band (BAND-ONLY CHANGE), nothing in the interior. ")
        why = ("This is a harness rule, not a game event: a batch stops at an action that changed nothing in the interior, "
               "because the actions after it were planned for an interior change that did not happen, and executing them "
               "would waste actions. The executed action and its band change stand. ")
    else:
        head = f"Action {noop_index} of {batch_size} ({act!r}) executed but changed nothing on the board (NO-OP). "
        why = ("This is a harness rule, not a game event: a batch stops at an action that produced no state change, because "
               "the actions after it were planned on the assumption that it would, and executing them on that assumption "
               "would waste actions. ")
    if skipped:
        sk = ("Action %d (%r) was not executed. " % (noop_index + 1, skipped[0]) if len(skipped) == 1 else
              f"Actions {noop_index + 1}-{batch_size} ({skipped!r}) were not executed. ")
        sk += "Indices are 1-based positions in your submitted list. "
    else:
        why = sk = ''
    tail = ("No executed action of this batch changed anything in the interior. " if not k_before else
            "Only the first executed action changed anything in the interior. " if k_before == 1 else
            f"Only the first {k_before} executed actions changed anything in the interior. ")
    return head + why + sk + tail + "The per-action verdicts are listed in action_trace."
