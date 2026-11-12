"""Benchmark (2026-10-08, user: "the length of the game in progress should not affect python call speed"): the wall time of
a trivial python call through the real sandbox (v62 patches and retention installed) as the retained program library grows.
Each call is a fresh sandbox process that re-executes every kept program (restore_functions) and re-parses the snippet for
the library (collect_functions). Synthetic programs of two sizes (~15 and ~60 lines); no game state.
usage (pod copy): PYTHONPATH=src:. python3 tests/bench_library_restore.py"""
import os
import statistics
import time

os.environ.update({'MULTIMODAL_CONTEXT': 'current_grid', 'MULTIMODAL_UPSCALE': '8', 'ARC3_PERSISTENT_FUNCTIONS': '1',
                   'ARC3_PERSISTENT_FUNCTIONS_SCOPE': 'game', 'ARC3_PERSISTENT_FUNCTIONS_IMPORTS': '1',
                   'ARC3_ANIMATION': '1', 'LOCAL_ANALYZER_TOOL_TIMEOUT': '30'})
import v62.patches  # noqa: E402,F401
import v62.retention as r  # noqa: E402
r.install()
from inference.agent import python_tool_sandbox as sb  # noqa: E402

SMALL = '''def f_{i}(grid, target=None):
    h, w = len(grid), len(grid[0])
    out = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == target:
                out.append((r, c))
    best = None
    for (r, c) in out:
        d = abs(r - h // 2) + abs(c - w // 2)
        if best is None or d < best[0]:
            best = (d, r, c)
    return best
'''
BIG = SMALL.replace('def f_{i}(', 'def g_{i}(').replace('    return best\n', '''    seen = set()
    stack = [best[1:]] if best else []
    while stack:
        r, c = stack.pop()
        if (r, c) in seen or not (0 <= r < h and 0 <= c < w):
            continue
        seen.add((r, c))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            stack.append((r + dr, c + dc))
    regions = {}
    for r in range(h):
        for c in range(w):
            v = grid[r][c]
            regions.setdefault(v, []).append((r, c))
    sizes = sorted((len(cells), v) for v, cells in regions.items())
    if not sizes:
        return None
    small = sizes[0]
    large = sizes[-1]
    centre = (sum(p[0] for p in regions[large[1]]) / large[0], sum(p[1] for p in regions[large[1]]) / large[0])
    path = []
    cur = best[1:] if best else (0, 0)
    for _ in range(64):
        if abs(cur[0] - centre[0]) < 1 and abs(cur[1] - centre[1]) < 1:
            break
        step = (1 if centre[0] > cur[0] else -1 if centre[0] < cur[0] else 0,
                1 if centre[1] > cur[1] else -1 if centre[1] < cur[1] else 0)
        cur = (cur[0] + step[0], cur[1] + step[1])
        path.append(cur)
    return dict(best=best, seen=len(seen), small=small, large=large, path=path)
''')


def call(kept):
    t = time.monotonic()
    out = sb.run_sandboxed_python(code='print(1)', timeout_seconds=30, initial_state={}, action_handler=lambda *a, **k: {},
                                  kept_functions=kept)
    dt = time.monotonic() - t
    assert '1' in str(out.get('stdout', out)), out
    return dt


print('programs   small (~15 lines)   big (~60 lines)   (median of 3, seconds per trivial call)')
for k in (0, 50, 100, 200, 300):
    row = []
    for tpl in (SMALL, BIG):
        kept = [tpl.replace("{i}", str(i)) for i in range(k)]
        row.append(statistics.median(call(kept) for _ in range(3)))
    print('%8d   %10.3f          %10.3f' % (k, row[0], row[1]))
print('no persistence (kept_functions=None): %.3f s' % statistics.median(call(None) for _ in range(3)))
