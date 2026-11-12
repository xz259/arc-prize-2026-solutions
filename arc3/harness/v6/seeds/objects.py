"""Locate same-symbol objects and track one across frames. Coordinates are (row, col).

Templates: the logic is game-independent; which symbol and size identify the
piece you control is discovered per game. Once known, save a short wrapper such as

    def player(frame):
        "Controlled piece: symbol b, 24 cells."
        return find_object(frame, 'b', size=24)

so later calls and the program index carry the identification. Self-contained:
grid.py does not need to be loaded first.
"""


def _cells(frame_or_ascii):
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


def _components(cells, connectivity):
    """Same grouping as grid.py connected_components; duplicated so this file stands alone."""
    if type(connectivity) is not int or connectivity not in (4, 8):
        raise ValueError("connectivity must be 4 or 8")
    remaining = set(cells)
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


def find_objects(frame_or_ascii, color, size=None, connectivity=4):
    """Connected same-symbol objects of color, optionally with exactly size cells.

    color is one symbol or several (any symbol in it matches, as in find_cells).
    Returns a list of objects, each a row-major list of (row, col) cells, ordered
    by first cell; [] when none match. Geometry only: an object here is a
    connected patch of equal symbols, which need not be one game entity.
    """
    if size is not None and (type(size) is not int or size < 1):
        raise ValueError("size must be None or a positive integer")
    wanted = set(color)
    cells = [(r, c) for r, row in enumerate(_cells(frame_or_ascii))
             for c, value in enumerate(row) if value in wanted]
    return [group for group in _components(cells, connectivity)
            if size is None or len(group) == size]


def find_object(frame_or_ascii, color, size=None, index=0, connectivity=4):
    """The index-th matching object from find_objects (row-major order), or None.

    Negative index counts from the end. With several matching objects, index is
    row-major order within each frame separately; use size or a unique symbol to
    track one piece reliably.
    """
    if type(index) is not int:
        raise TypeError("index must be an integer")
    objects = find_objects(frame_or_ascii, color, size, connectivity)
    return objects[index] if -len(objects) <= index < len(objects) else None


def object_position(frame_or_ascii, color, size=None, index=0, connectivity=4):
    """Top-left (row_min, col_min) of find_object(...), or None when absent."""
    obj = find_object(frame_or_ascii, color, size, index, connectivity)
    if obj is None:
        return None
    return (min(r for r, _ in obj), min(c for _, c in obj))


def object_move(before, after, color, size=None, index=0, connectivity=4):
    """(d_row, d_col) of the tracked object's top-left from before to after.

    None when the object is absent in either frame. Frames may differ in shape.
    A zero move means the object stayed put or changed shape in place; compare
    find_object results directly when shape matters.
    """
    start = object_position(before, color, size, index, connectivity)
    end = object_position(after, color, size, index, connectivity)
    if start is None or end is None:
        return None
    return (end[0] - start[0], end[1] - start[1])
