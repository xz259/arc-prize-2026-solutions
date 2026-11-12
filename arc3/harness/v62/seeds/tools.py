"""Generic analysis helpers (harness v6.6 C, agreed with IE 2026-10-17). Coordinates are (row, col); bounds are inclusive, as in
frame_diff and segmentation. Facts only: nothing here knows a game. A library function of the same name replaces a helper."""


def _t_rows(frame_or_ascii):
    """Rows of symbols from a public frame, recorded frame dictionary, or ASCII text."""
    source = frame_or_ascii
    if hasattr(source, "ascii"):
        source = source.ascii
    elif isinstance(source, dict):
        source = source["ascii"]
    if isinstance(source, str):
        source = [] if source == "(empty grid)" else source.splitlines()
    rows = tuple(tuple(row) for row in source)
    if rows and any(len(row) != len(rows[0]) for row in rows):
        raise ValueError("Grid rows must have equal lengths")
    return rows


def crop(frame_or_ascii, r0, c0, r1, c1):
    """ASCII of rows r0..r1 and cols c0..c1 (inclusive) with row and column numbers; refuses windows over 32 x 32 (use show_board)."""
    rows = _t_rows(frame_or_ascii)
    if not rows:
        return "(empty grid)"
    for v in (r0, c0, r1, c1):
        if type(v) is not int:
            raise TypeError("crop bounds must be integers")
    r0, c0 = max(0, r0), max(0, c0)
    r1, c1 = min(len(rows) - 1, r1), min(len(rows[0]) - 1, c1)
    if r1 < r0 or c1 < c0:
        raise ValueError("empty window after clipping to the board")
    if r1 - r0 + 1 > 32 or c1 - c0 + 1 > 32:
        raise ValueError("window %dx%d is larger than 32 x 32: crop a smaller window, or call show_board for a whole board"
                         % (r1 - r0 + 1, c1 - c0 + 1))
    cols = range(c0, c1 + 1)
    head = ["    " + "".join(str(c // 10) for c in cols), "    " + "".join(str(c % 10) for c in cols)]
    return "\n".join(head + ["%3d " % r + "".join(rows[r][c0:c1 + 1]) for r in range(r0, r1 + 1)])


def _t_components(rows, connectivity):
    """Same-symbol connected components as row-major cell lists (4-connectivity matches `segmentation`)."""
    seen = set()
    offsets = ((-1, 0), (0, -1), (0, 1), (1, 0))
    if connectivity == 8:
        offsets += ((-1, -1), (-1, 1), (1, -1), (1, 1))
    out = []
    for r in range(len(rows)):
        for c in range(len(rows[0]) if rows else 0):
            if (r, c) in seen:
                continue
            sym = rows[r][c]
            stack, comp = [(r, c)], []
            seen.add((r, c))
            while stack:
                y, x = stack.pop()
                comp.append((y, x))
                for dy, dx in offsets:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < len(rows) and 0 <= nx < len(rows[0]) and (ny, nx) not in seen and rows[ny][nx] == sym:
                        seen.add((ny, nx))
                        stack.append((ny, nx))
            out.append((sym, sorted(comp)))
    return out


def objects(frame_or_ascii, colors=None, min_size=1, connectivity=4):
    """Connected same-symbol patches as dicts (color, size, inclusive bbox, center, cells), largest first; 4-connected ones are exactly the segmentation nodes."""
    if connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    if type(min_size) is not int or min_size < 1:
        raise ValueError("min_size must be a positive integer")
    wanted = None if colors is None else set(colors)
    out = []
    for sym, cells in _t_components(_t_rows(frame_or_ascii), connectivity):
        if (wanted is not None and sym not in wanted) or len(cells) < min_size:
            continue
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        out.append(dict(color=sym, size=len(cells), bbox=(min(rs), min(cs), max(rs), max(cs)),
                        center=(round(sum(rs) / len(rs), 2), round(sum(cs) / len(cs), 2)), cells=cells))
    out.sort(key=lambda o: (-o["size"], o["bbox"][0], o["bbox"][1]))
    return out


def describe(frame_or_ascii, top=20, skip_largest=True, connectivity=4):
    """One line per object (color, size, inclusive bbox), largest first; the largest uniform region by area is omitted unless skip_largest=False."""
    objs = objects(frame_or_ascii, connectivity=connectivity)
    lines = []
    if skip_largest and objs:
        o = objs.pop(0)
        lines.append("largest region omitted: %s, %d cells, bbox %s" % (o["color"], o["size"], o["bbox"]))
    for o in objs[:top]:
        lines.append("%s size %d bbox %s" % (o["color"], o["size"], o["bbox"]))
    if len(objs) > top:
        lines.append("... and %d more (objects() returns all)" % (len(objs) - top))
    return "\n".join(lines)


def bfs(start, neighbors, is_goal, max_nodes=100000, max_seconds=6.0):
    """Breadth-first search over hashable states; neighbors(state) yields (move, next_state); returns (moves or None, nodes, stop) with stop in goal / exhausted / max_nodes / time."""
    from collections import deque
    try:
        clock = _tool_clock                                   # the harness's monotonic clock (imports are restricted)
    except NameError:
        clock = None
    try:
        hash(start)
    except TypeError:
        raise TypeError("bfs states must be hashable (use tuples, not lists or dicts)") from None
    if is_goal(start):
        return [], 0, "goal"
    t0 = clock() if clock else 0.0
    parent = {start: None}
    queue = deque([start])
    nodes = 0
    while queue:
        if nodes >= max_nodes:
            return None, nodes, "max_nodes"
        if clock and clock() - t0 > max_seconds:
            return None, nodes, "time"
        state = queue.popleft()
        nodes += 1
        for move, nxt in neighbors(state):
            if nxt in parent:
                continue
            parent[nxt] = (state, move)
            if is_goal(nxt):
                path, cur = [], nxt
                while parent[cur] is not None:
                    cur, mv = parent[cur]
                    path.append(mv)
                return path[::-1], nodes, "goal"
            queue.append(nxt)
    return None, nodes, "exhausted"


def board_key(frame_or_ascii, region=None):
    """Hashable key of the interior (rows and cols 4-59 of a 64 x 64 frame, the harness's band convention) or of an inclusive region (r0, c0, r1, c1)."""
    rows = _t_rows(frame_or_ascii)
    if not rows:
        return ()
    r0, c0, r1, c1 = region if region is not None else (4, 4, len(rows) - 5, len(rows[0]) - 5)
    return tuple("".join(row[c0:c1 + 1]) for row in rows[r0:r1 + 1])


def _t_runs(seq):
    """Run-length list of (symbol, count)."""
    out = []
    for s in seq:
        if out and out[-1][0] == s:
            out[-1] = (s, out[-1][1] + 1)
        else:
            out.append((s, 1))
    return out


def band(frame_or_ascii, width=4):
    """The outer band (width 4: rows 0-3 and 60-63, cols 0-3 and 60-63 of a 64 x 64 frame) as symbol runs per line: dict top, bottom, left, right."""
    rows = _t_rows(frame_or_ascii)
    if not rows:
        return {}
    h, w = len(rows), len(rows[0])
    return dict(top=[_t_runs(rows[r]) for r in range(width)],
                bottom=[_t_runs(rows[r]) for r in range(h - width, h)],
                left=[_t_runs([rows[r][c] for r in range(width, h - width)]) for c in range(width)],
                right=[_t_runs([rows[r][c] for r in range(width, h - width)]) for c in range(w - width, w)])


def band_diff(before, after, width=4, limit=64):
    """Changed cells of the outer band between two frames: count, a bounded (row, col, old, new) sample and the count per side."""
    a, b = _t_rows(before), _t_rows(after)
    if len(a) != len(b) or (a and len(a[0]) != len(b[0])):
        raise ValueError("Cannot compare grids with different shapes")
    h, w = len(a), len(a[0]) if a else 0
    sides = dict(top=0, bottom=0, left=0, right=0)
    sample, n = [], 0
    for r in range(h):
        for c in range(w):
            side = ("top" if r < width else "bottom" if r >= h - width else "left" if c < width else "right" if c >= w - width
                    else None)
            if side is None or a[r][c] == b[r][c]:
                continue
            n += 1
            sides[side] += 1
            if len(sample) < limit:
                sample.append((r, c, a[r][c], b[r][c]))
    return dict(changed=n, changes=sample, per_side=sides, truncated=len(sample) < n)
