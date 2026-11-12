"""Pure helpers for visible grids. Coordinates are (row, col), never image pixels."""


def grid_cells(frame_or_ascii):
    """Return immutable rectangular rows from a public frame, recorded frame, or ASCII."""
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


def color_counts(frame_or_ascii):
    """Count visible symbols; makes no assumption about their gameplay meaning."""
    from collections import Counter
    return dict(Counter(cell for row in grid_cells(frame_or_ascii) for cell in row))


def find_cells(frame_or_ascii, values):
    """Return row-major (row, col) positions whose symbols are in values."""
    wanted = set(values)
    return [(r, c) for r, row in enumerate(grid_cells(frame_or_ascii))
            for c, value in enumerate(row) if value in wanted]


def bounding_box(cells):
    """Return (row_min, col_min, row_stop, col_stop), exclusive stops; empty -> None."""
    cells = list(cells)
    if not cells:
        return None
    return (min(p[0] for p in cells), min(p[1] for p in cells),
            max(p[0] for p in cells) + 1, max(p[1] for p in cells) + 1)


def grid_diff(before, after, limit=64):
    """Count all changed cells; return a bounded row-major sample and full change bbox.

    Each change is (row, col, old_symbol, new_symbol). Different shapes raise
    ValueError. Counts/bbox cover the full grid even when the sample is truncated.
    """
    if type(limit) is not int or limit < 0:
        raise ValueError("limit must be a nonnegative integer")
    before, after = grid_cells(before), grid_cells(after)
    if len(before) != len(after) or any(len(a) != len(b) for a, b in zip(before, after)):
        raise ValueError("Cannot compare grids with different shapes")
    changed, sample = [], []
    for r, (old_row, new_row) in enumerate(zip(before, after)):
        for c, (old, new) in enumerate(zip(old_row, new_row)):
            if old != new:
                changed.append((r, c))
                if len(sample) < limit:
                    sample.append((r, c, old, new))
    return dict(changed=len(changed), changes=sample, bbox=bounding_box(changed),
                truncated=len(sample) < len(changed))


def neighbors4(cell, shape):
    """In-bounds orthogonal neighbors, ordered up/left/right/down; ignores obstacles."""
    r, c = cell
    rows, cols = shape
    if not (0 <= r < rows and 0 <= c < cols):
        raise ValueError("Starting cell must be inside shape")
    return [(nr, nc) for nr, nc in ((r - 1, c), (r, c - 1), (r, c + 1), (r + 1, c))
            if 0 <= nr < rows and 0 <= nc < cols]


def connected_components(cells, connectivity=4):
    """Group selected integer (row, col) coordinates by 4- or 8-neighbor adjacency.

    Pass coordinates, not a raw grid or mask: use find_cells(frame, values), or
    [(r,c) for r,row in enumerate(mask) for c,on in enumerate(row) if on].
    All supplied cells may connect regardless of color; select each color
    separately when needed. This is geometry, not object or movement semantics.

    Returns a list of components, each a row-major list of (row, col) tuples.
    Components are ordered by their first cell. Duplicate cells are ignored;
    empty input returns []. Input is not modified; negative coordinates work.
    """
    from operator import index
    if type(connectivity) is not int or connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    remaining = set()
    for cell in cells:
        r, c = cell
        if isinstance(r, bool) or isinstance(c, bool):
            raise TypeError("cells must contain integer (row, col) coordinates, not a mask")
        remaining.add((index(r), index(c)))
    offsets = ((-1, 0), (0, -1), (0, 1), (1, 0))
    if connectivity == 8:
        offsets += ((-1, -1), (-1, 1), (1, -1), (1, 1))
    components = []
    for start in sorted(remaining):
        if start not in remaining:
            continue
        remaining.remove(start)
        stack, component = [start], []
        while stack:
            r, c = stack.pop()
            component.append((r, c))
            for dr, dc in offsets:
                neighbor = (r + dr, c + dc)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        components.append(sorted(component))
    return components
