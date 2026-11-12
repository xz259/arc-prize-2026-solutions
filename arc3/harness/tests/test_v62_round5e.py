"""v6.2 round 5e unit tests (381 review): `actions_since_start` counts from the last board whose interior equals the
level's starting board (or the RESET / level start), as its text says; a near-start line no longer restarts the count."""
import os
for k, v in dict(ARC3_V61_TEXTS='1', ARC3_V6_A13='1', ARC3_V61_SUMMARY='1', ARC3_NEW_CHANGED_PROMPTS='1',
                 ARC3_ACTION_INFO='1', ARC3_REPEAT_STATE_GUARD='0', ARC3_V61_RESTART_TOKENS='150000', ARC3_V6_A3='1').items():
    os.environ.setdefault(k, v)
import v62.session_ext as se
fails = []


def check(name, got, want):
    ok = got == want
    print('%-62s -> %-48r %s' % (name, got, 'OK' if ok else 'WRONG (want %r)' % (want,)))
    if not ok:
        fails.append(name)


START = [[0] * 32 for _ in range(32)]                 # a 32 x 32 board: the interior is rows and columns 4-27
INTERIOR = [(r, c) for r in range(4, 28) for c in range(4, 28)]
# round 5f: excursions of 80 cells (> NEAR_REARM 64) arm the near line; the counts below are round 5e's semantics


def board(n, band=False):
    g = [r[:] for r in START]
    for r, c in INTERIOR[:n]:
        g[r][c] = 5
    if band:
        g[0][0] = 7
    return g


def step(d, n, restart=False, match=False, automatic=False, terminal=False, band=False):
    p = {'level_start_match': 'full'} if match else {}
    se._start_facts(d, p, restart, automatic, terminal, board(n, band), START)
    p.pop('level_start_match', None)
    return p


d = {}
step(d, 0, restart=True)
check('away', step(d, 80), {})
check('near after being further: counted from the RESET', step(d, 8), dict(near_level_start=8, actions_since_start=2))
check('away again', step(d, 80), {})
check('second near line: from the RESET, not the first near line', step(d, 8), dict(near_level_start=8, actions_since_start=4))
check('exact return', step(d, 0, match=True), dict(actions_since_start=5))
check('away after the return', step(d, 80), {})
check('near: counted from the exact return', step(d, 8), dict(near_level_start=8, actions_since_start=2))

d = {}
step(d, 0, restart=True)
step(d, 0)
step(d, 0, band=True)                                 # actions that leave the interior at the start are not counted
step(d, 80)
check('idle actions at the start are not counted (away + return)', step(d, 0, match=True), dict(actions_since_start=2))

d = {}
step(d, 0, restart=True)
step(d, 80)
check('an automatic entry is not an action', step(d, 80, automatic=True), {})
check('count after it', step(d, 0, match=True), dict(actions_since_start=2))
step(d, 80)
step(d, 80, terminal=True)                            # GAME OVER: counted, no facts
step(d, 0, restart=True, automatic=True)              # its automatic restart restarts the count
step(d, 80)
check('after a GAME OVER restart', step(d, 8), dict(near_level_start=8, actions_since_start=2))

d = {}
step(d, 0, restart=True)
check('near without being further first: no line', step(d, 8), {})
step(d, 80)
check('17 cells is not near', step(d, 17), {})
check('16 cells is near', step(d, 16), dict(near_level_start=16, actions_since_start=4))

print('FAILS:', fails)
