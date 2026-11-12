"""v6.6 item C (seeds/tools.py; agreed with IE 2026-10-17): the helpers in an import-restricted namespace, objects() against the
runtime's segmentation on random frames, bfs's four stops, board_key under band-only changes, band runs and band_diff, the
prompt switch, and real sandbox calls (seeds loaded, `_tool_clock` injected). Scaffolding: test_v64_fixes.py's prelude.
usage (dev copy of harness_v6): PYTHONPATH=src:. python3 tests/test_v66_seeds.py"""
import builtins
import json
import os
import random
import time
from pathlib import Path

os.environ['ARC3_V66_SEEDS'] = '1'
_src = open(Path(__file__).with_name('test_v64_fixes.py')).read()
exec(compile(_src.split('\ndef run_notes_turn')[0], 'test_v64_fixes.py (prelude)', 'exec'), globals())
import v62.sandbox_ext as SX  # noqa: E402
import v62.variants as V  # noqa: E402

TOOLS = (Path(A.__file__).with_name('seeds') / 'tools.py').read_text()
SAFE = {'bisect', 'collections', 'copy', 'fractions', 'functools', 'heapq', 'itertools', 'json', 'math', 'operator', 'random',
        're', 'statistics', 'string'}


def _safe_import(name, globals=None, locals=None, fromlist=(), level=0):
    if str(name).split('.')[0] not in SAFE:
        raise ImportError("Module '%s' is not allowed in the sandbox." % name)
    return builtins.__import__(name, globals, locals, fromlist, level)


def namespace(clock=True):
    ns = {'__builtins__': dict(vars(builtins), __import__=_safe_import)}
    if clock:
        ns['_tool_clock'] = time.monotonic
    exec(compile(TOOLS, '<seeds>', 'exec'), ns, ns)
    return ns


T = namespace()
check('SEED tools.py executes under the restricted imports', all(k in T for k in ('crop', 'objects', 'describe', 'bfs', 'board_key',
                                                                                  'band', 'band_diff')), True)
rng = random.Random(7)
SYM = 'WwgGcBMPRbSYOrNp'


def rand_board(h=64, w=64, blobs=25):
    g = [['B'] * w for _ in range(h)]
    for _ in range(blobs):
        r, c = rng.randrange(h), rng.randrange(w)
        hh, ww, s = rng.randint(1, 9), rng.randint(1, 9), rng.choice(SYM)
        for y in range(r, min(h, r + hh)):
            for x in range(c, min(w, c + ww)):
                if rng.random() < 0.85:
                    g[y][x] = s
    return '\n'.join(''.join(row) for row in g)


# --- crop
B0 = rand_board()
rows = B0.split('\n')
out = T['crop'](B0, 4, 8, 6, 12).split('\n')
check('crop: two header lines (tens, ones) and three numbered rows', (len(out), out[0][4:], out[1][4:], out[2][:4], out[2][4:]),
      (5, '00111', '89012', '  4 ', rows[4][8:13]))
check('crop: clipped to the board', T['crop'](B0, -3, -3, 1, 1).split('\n')[2:], ['  0 ' + rows[0][:2], '  1 ' + rows[1][:2]])
try:
    T['crop'](B0, 0, 0, 40, 10)
    msg = None
except ValueError as e:
    msg = str(e)
check('crop: refuses a window over 32 x 32, naming show_board', (msg or '').startswith('window 41x11 is larger than 32 x 32')
      and 'show_board' in (msg or ''), True)
try:
    T['crop'](B0, 0.5, 0, 3, 3)
    te = False
except TypeError:
    te = True
check('crop: non-integer bounds raise TypeError', te, True)

# --- objects vs segmentation: compared inside a real sandbox call below (the host Frame has no segmentation)
objs = T['objects'](B0, colors='R', min_size=3)
check('objects: colour and min_size filters, largest first, centre inside the bbox',
      (all(o['color'] == 'R' and o['size'] >= 3 for o in objs), [o['size'] for o in objs] == sorted([o['size'] for o in objs], reverse=True),
       all(o['bbox'][0] <= o['center'][0] <= o['bbox'][2] and o['bbox'][1] <= o['center'][1] <= o['bbox'][3] for o in objs)),
      (True, True, True))
d = T['describe'](B0, top=5).split('\n')
check('describe: the largest region omitted line, then at most 5 patches and a remainder line',
      (d[0].startswith('largest region omitted: '), len(d) in (6, 7), d[-1].startswith('... and ') or len(d) == 6), (True, True, True))

# --- bfs
WALLS = {(1, c) for c in range(0, 9)} | {(3, c) for c in range(1, 10)}


def nb(s):
    r, c = s
    for mv, (dr, dc) in (('U', (-1, 0)), ('D', (1, 0)), ('L', (0, -1)), ('R', (0, 1))):
        n = (r + dr, c + dc)
        if 0 <= n[0] < 5 and 0 <= n[1] < 10 and n not in WALLS:
            yield mv, n


path, nodes, stop = T['bfs']((0, 0), nb, lambda s: s == (4, 0))
pos = (0, 0)
for mv in path or []:
    pos = dict(nb(pos))[mv]
check('bfs: goal reached by a shortest path (length 22) that replays', (stop, len(path or []), pos), ('goal', 22, (4, 0)))
check('bfs: unreachable goal -> exhausted', T['bfs']((0, 0), nb, lambda s: s == (9, 9))[::2], (None, 'exhausted'))
check('bfs: max_nodes stop', T['bfs']((0, 0), nb, lambda s: s == (4, 0), max_nodes=5), (None, 5, 'max_nodes'))


def slow(s):
    t = time.monotonic()
    while time.monotonic() - t < 0.002:
        pass
    return [('+', s + 1)]


check('bfs: time stop', T['bfs'](0, slow, lambda s: False, max_seconds=0.05)[2], 'time')
check('bfs: no clock in the namespace -> no time stop (max_nodes still applies)', namespace(clock=False)['bfs'](0, slow, lambda s: False,
                                                                                                          max_nodes=20, max_seconds=0.0)[2],
      'max_nodes')
check('bfs: start already a goal', T['bfs']((1, 1), nb, lambda s: True), ([], 0, 'goal'))
try:
    T['bfs']([0, 0], nb, lambda s: False)
    te = None
except TypeError as e:
    te = str(e)
check('bfs: unhashable start -> TypeError naming tuples', 'hashable' in (te or ''), True)

# --- board_key, band, band_diff
g = [list(r) for r in rows]
for c in range(10, 20):
    g[1][c] = 'Y' if g[1][c] != 'Y' else 'O'
g[62][5] = 'S' if g[62][5] != 'S' else 'R'
B1 = '\n'.join(''.join(r) for r in g)
check('board_key: unchanged by band-only changes; region override', (T['board_key'](B0) == T['board_key'](B1),
                                                                     len(T['board_key'](B0)), len(T['board_key'](B0)[0]),
                                                                     T['board_key'](B0, region=(0, 0, 1, 2))),
      (True, 56, 56, (rows[0][:3], rows[1][:3])))
g2 = [list(r) for r in rows]
g2[30][30] = 'W' if g2[30][30] != 'W' else 'p'
check('board_key: an interior change changes it', T['board_key'](B0) != T['board_key']('\n'.join(''.join(r) for r in g2)), True)
bd = T['band'](B0)
check('band: four lines per side; runs rebuild the lines', ([len(bd[k]) for k in ('top', 'bottom', 'left', 'right')],
                                                           ''.join(s * n for s, n in bd['top'][2]) == rows[2],
                                                           ''.join(s * n for s, n in bd['left'][0]) == ''.join(rows[r][0] for r in range(4, 60))),
      ([4, 4, 4, 4], True, True))
dd = T['band_diff'](B0, B1)
check('band_diff: 11 band cells (10 top, 1 bottom)', (dd['changed'], dd['per_side'], dd['truncated']),
      (11, dict(top=10, bottom=1, left=0, right=0), False))

# --- prompt switch
RAW = json.load(open(Path(A.__file__).with_name('texts.json')))['texts']
t_on = A.TEXTS
os.environ['ARC3_V66_SEEDS'] = '0'
t_off = V.apply(json.loads(json.dumps(RAW)))
os.environ['ARC3_V66_SEEDS'] = '1'
S_on, S_off = t_on['system'], t_off['system']
check('prompt: tools.py documented only with the switch on', ('  - tools.py: `crop(frame, r0, c0, r1, c1)`' in S_on, 'tools.py' in S_off),
      (True, False))
check('prompt: bounding-box convention, override rule and bfs pointer',
      ("grid.py's `bounding_box` uses exclusive stops, while tools.py, `frame_diff` and `segmentation` use inclusive bounds." in S_on,
       'A library function of the same name replaces a starter helper.' in S_on,
       'instead of stepping by hand; the starter `bfs` runs one over your own step function.' in S_on), (True, True, True))
check('prompt: no other text differs', sorted(k for k in set(t_on) | set(t_off) if t_on.get(k) != t_off.get(k)), ['system'])

# --- real sandbox calls
check('sandbox: tools.py among the seed files', SX.SEED_FILES, ('grid.py', 'objects.py', 'tools.py'))
state = Path(tempfile.mkdtemp()) / 'game_runtime_state.json'
game = FakeGame(state)
code = ("print(bfs(0, lambda s: [(1, s + 1)] if s < 3 else [], lambda s: s == 3))\n"
        "print(crop(current_frame, 4, 4, 5, 6).split(chr(10))[2])\n"
        "def slow_nb(s):\n    x = 0\n    for i in range(3000):\n        x += i\n    return [(1, s + 1)]\n"
        "print(bfs(0, slow_nb, lambda s: False, max_seconds=0.05)[2])\n"
        "print(len(objects(current_frame)), board_key(current_frame) == board_key(current_frame))")
agent, sess, _ = new_agent([(code, 100), ('action(["UP"])', 50)])
agent.analyze(state, game.step, valid_actions=['UP', 'LEFT'], step_env=game.step_env, should_stop=lambda: False)
tool_msgs = [m for m in sess.sent[1] if m.get('role') == 'tool']
text = tool_msgs[-1]['content'] if tool_msgs else ''
text = text if isinstance(text, str) else json.dumps(text)
check('sandbox: bfs, crop, the clock-based time stop and objects run in a real call',
      ("([1, 1, 1], 3, 'goal')" in text, '  4 ' in text, 'time' in text, '1 True' in text), (True, True, True, True))

# objects (connectivity 4) against the sandbox's own segmentation, on random frames placed in the game's history
state2 = Path(tempfile.mkdtemp()) / 'game_runtime_state.json'
game2 = FakeGame(state2)


def rand_grid():
    g = [[0] * 64 for _ in range(64)]
    for _ in range(rng.randint(5, 40)):
        r, c, hh, ww, v = rng.randrange(64), rng.randrange(64), rng.randint(1, 9), rng.randint(1, 9), rng.randrange(16)
        for y in range(r, min(64, r + hh)):
            for x in range(c, min(64, c + ww)):
                if rng.random() < 0.85:
                    g[y][x] = v
    return g


for i in range(1, 9):
    game2.history.append(HistoryEntry(action='UP', frame=Frame(grid=rand_grid(), step=i, level=1),
                                      result=dict(executed=True, gameplay_changed=True, board_changed=True, action_num=i, level=1,
                                                  action_name='UP', action_display='UP', valid_actions=['UP', 'LEFT'],
                                                  level_completed=False)))
game2.step = 8
write_runtime_state(state2, current_frame=game2.history[-1].frame, history=game2.history)
code2 = ("ok = 0\n"
         "for h in history[1:]:\n"
         "    f = h.frame\n"
         "    seg = sorted((n['color'], n['pixels'], (min(p[0] for p in n['boundary']), min(p[1] for p in n['boundary']), "
         "max(p[0] for p in n['boundary']), max(p[1] for p in n['boundary']))) for n in f.segmentation['nodes'])\n"
         "    obj = sorted((o['color'], o['size'], o['bbox']) for o in objects(f))\n"
         "    ok += seg == obj\n"
         "print('SEGMATCH', ok, len(history) - 1)")
agent2, sess2b, _ = new_agent([(code2, 100), ('action(["UP"])', 50)])
agent2.analyze(state2, game2.step, valid_actions=['UP', 'LEFT'], step_env=game2.step_env, should_stop=lambda: False)
tm = [m for m in sess2b.sent[1] if m.get('role') == 'tool']
t2 = tm[-1]['content'] if tm else ''
t2 = t2 if isinstance(t2, str) else json.dumps(t2)
check('objects (connectivity 4) equals the sandbox segmentation on 8 random frames (colour, size, inclusive bbox)',
      'SEGMATCH 8 8' in t2, True)
print('FAILS:', fails)
