"""Round 5m item 125 (user 2026-10-09: refit restart_value from run 396): the constants, and the ordering the allocator
relies on within equal swap-out counts (a 7-level game, as the 10-05 worked example): with a long horizon the lower stuck
level ranks first (more levels ahead), with a horizon of one level the higher one (more weight now); p no longer depends on
the level. v6.3 item D (user 2026-10-13, "adopted tentatively"): constants 0.41 / 0.30 / 0.78 / 0.92 and LEFT_FACTOR 0.7
on p with <= 2 levels left (the worked-example values updated). usage (pod copy): PYTHONPATH=src:. python3 tests/test_v62_round5m_restartvalue.py"""
import os
for k, v in dict(ARC3_V61_TEXTS='1', ARC3_V6_A13='1', ARC3_V61_SUMMARY='1', ARC3_NEW_CHANGED_PROMPTS='1',
                 ARC3_ACTION_INFO='1', ARC3_REPEAT_STATE_GUARD='0', ARC3_V61_RESTART_TOKENS='150000', ARC3_V6_A3='1',
                 ARC3_GUARDS_FROM_LEVEL='2', ARC3_BATCH_NOOP_BLOCK='1', ARC3_STALE_STATE_BLOCK='1',
                 ARC3_ANIMATION='1').items():
    os.environ.setdefault(k, v)
import v62.patches  # noqa: E402,F401
import v62.retention as r  # noqa: E402
r.install()
import v62.agent as A  # noqa: E402
fails = []


def check(name, got, want):
    ok = got == want
    print('%-72s -> %-28r %s' % (name, got, 'OK' if ok else 'WRONG (want %r)' % (want,)))
    if not ok:
        fails.append(name)


check('125 / v6.3 D constants', (A.RESTART_P, A.STUCK_C, A.NEXT_C, A.DECAY_C, A.LEFT_FACTOR), (0.41, 0.30, 0.78, 0.92, 0.7))
long_s = 10 ** 6                                       # every level playable
one_s = A.LEVEL_TOKENS / A.LANE_TOKS * 1.01            # the stuck level plus one
v = lambda L, s: round(A.restart_value(L, 7, s), 4)
check('125 long horizon, 7 levels: L2 0.2441, L5 0.1640', (v(2, long_s), v(5, long_s)), (0.2441, 0.164))
check('125   ... the lower stuck level ranks first', v(2, long_s) > v(5, long_s), True)
check('125 horizon of one level: L2 0.0430, L5 0.0905', (v(2, one_s), v(5, one_s)), (0.043, 0.0905))
check('125   ... the higher stuck level ranks first', v(5, one_s) > v(2, one_s), True)
check('125 p is the same at L4 and L5 (no level step; both one level left)', round(A.restart_value(4, 4, 0) / (0.30 * 4 / 10), 4) ==
      round(A.restart_value(5, 5, 0) / (0.30 * 5 / 15), 4), True)
check('v6.3 D: p x 0.7 with <= 2 levels left (L6 and L7 of 7, long horizon)', (v(6, long_s), v(7, long_s)), (0.0744, 0.0215))
check('v6.3 D: the factor applies with 2 left, not with 3', (round(A.restart_value(6, 7, 0) / (0.30 * 6 / 28), 3),
      round(A.restart_value(5, 7, 0) / (0.30 * 5 / 28), 3)), (0.287, 0.41))
check('v6.3 D: 8-level order at a long horizon: L2 > L1 > L3 > ... > L8', sorted(range(1, 9), key=lambda L: -A.restart_value(L, 8, long_s)),
      [2, 1, 3, 4, 5, 6, 7, 8])
check('125 a snapshot priority stays in its swap band', 0 <= A._V62Snapshot(3, 7, 1).priority(0.0, long_s) % 1000 <= 999, True)
print('FAILS:', fails)
print('test_v62_round5m_restartvalue', 'PASS' if not fails else 'FAIL')
