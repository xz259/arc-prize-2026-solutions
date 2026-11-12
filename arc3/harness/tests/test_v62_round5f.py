"""v6.2 round 5f unit tests (Fable's final review, user 2026-10-06 ~04:30): REVERTED names the interior (E5); the
near-start line fires once per excursion of more than 64 cells (S1); the edge line no longer repeats the band's
definition (S2)."""
import os
for k, v in dict(ARC3_V61_TEXTS='1', ARC3_V6_A13='1', ARC3_V61_SUMMARY='1', ARC3_NEW_CHANGED_PROMPTS='1',
                 ARC3_ACTION_INFO='1', ARC3_REPEAT_STATE_GUARD='0', ARC3_V61_RESTART_TOKENS='150000', ARC3_V6_A3='1').items():
    os.environ.setdefault(k, v)
import v62.patches  # noqa: F401
import v62.retention as r; r.install()
import v62.agent as A
import v62.session_ext as se
from inference.framework import solver as sv
fails = []


def check(name, got, want):
    ok = got == want
    print('%-62s -> %-48r %s' % (name, got if not isinstance(got, str) else got[:48], 'OK' if ok else 'WRONG (want %r)' % (want,)))
    if not ok:
        fails.append(name)


# E5: REVERTED compares the interior
S = A.TEXTS['system']
check('system: REVERTED names the interior', 'REVERTED (it animated, then the interior returned to its previous state)' in S, True)
check('system: no "board returned to its previous state"', 'board returned to its previous state' in S, False)
check('animation note names the interior', 'then the interior returned to its previous state (REVERTED)' in A.TEXTS['animation'], True)
check('revert stop names the interior', 'each animated and then left the interior as it was' in A.TEXTS['revert_stop'], True)
check('verdict', sv._action_verdict(dict(reverted=True, gameplay_changed=False)),
      'REVERTED (animated, then the interior returned to its previous state)')
check('stop detail', 'left the interior as it was' in se._v61_stop_detail('reverted', ['UP', 'UP'], []), True)
# S2: the edge line
check('edge line', A.TEXTS['edge_line'], 'Outer band in your previous call: {text}.')
check('system still defines the outer band by rows and columns', 'the outer band (rows 0-3 and 60-63, columns 0-3 and 60-63)' in S, True)
# S1: the near-start re-arm
check('system documents the re-arm', 'more than 64 cells since the board was last that close' in S, True)
START = [[0] * 32 for _ in range(32)]
INTERIOR = [(rr, c) for rr in range(4, 28) for c in range(4, 28)]


def board(n):
    g = [row[:] for row in START]
    for rr, c in INTERIOR[:n]:
        g[rr][c] = 5
    return g


def step(d, n, restart=False, match=False):
    p = {'level_start_match': 'full'} if match else {}
    se._start_facts(d, p, restart, False, False, board(n), START)
    p.pop('level_start_match', None)
    return p


d = {}
step(d, 0, restart=True)
step(d, 30)
check('excursion of 30 cells (<= 64) does not arm', step(d, 8), {})
step(d, 80)
check('excursion of 80 arms: the line fires', step(d, 8), dict(near_level_start=8, actions_since_start=4))
step(d, 20)
check('hovering 8 -> 20 -> 8 after it: no second line', step(d, 8), {})
step(d, 70)
check('another real excursion: fires again', step(d, 8), dict(near_level_start=8, actions_since_start=8))
step(d, 0, match=True)
step(d, 30)
check('after an exact return, no line without a real excursion', step(d, 8), {})
print('FAILS:', fails)
