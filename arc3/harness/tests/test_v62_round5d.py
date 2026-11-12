"""v6.2 round 5d unit tests (380 g047 crash): a session error restarts the level in place instead of ending the game
(capped per game), the fake session's error injection, and the runner's graceful-stop handler (source check)."""
import inspect
import os
for k, v in dict(ARC3_V61_TEXTS='1', ARC3_V6_A13='1', ARC3_V61_SUMMARY='1', ARC3_NEW_CHANGED_PROMPTS='1',
                 ARC3_ACTION_INFO='1', ARC3_REPEAT_STATE_GUARD='0', ARC3_V61_RESTART_TOKENS='150000', ARC3_V6_A3='1').items():
    os.environ.setdefault(k, v)
import requests
import v62.patches  # noqa
import v62.retention as r; r.install()
import v62.agent as A
from v62.transport import CapacityRefusal, PersistentProtocolError
fails = []


def check(name, got, want):
    ok = got == want
    print('%-55s -> %-50r %s' % (name, got if not isinstance(got, str) else got[:50], 'OK' if ok else 'WRONG (want %r)' % (want,)))
    if not ok:
        fails.append(name)


def make_agent(exc):
    agent = object.__new__(A.V6Agent)
    agent._v6_stats, agent._logs, agent._calls = {}, [], []
    agent._v6_log = lambda kind, payload: agent._logs.append((kind, payload))

    def failing(state_path, action_num, **kw):
        agent._calls.append(kw)
        raise exc
    agent._v62_analyze = failing
    return agent


# 1. the g047 error: a retryable failure and an in-place restart on the next call, up to the cap
agent = make_agent(PersistentProtocolError("Completed assistant lacks actual generated im_end"))
res = A.V6Agent.analyze(agent, 'sp', 5, valid_actions=['UP'], should_stop=None)
check('session error -> retryable failure', (res.retryable_failure, res.step_executed), (True, False))
check('flag set for the in-place restart', agent._v62_force_restart, True)
check('logged and counted', (agent._logs[0][0], agent._logs[0][1]['n'], agent._logs[0][1]['restart'],
                             agent._v6_stats['protocol_restarts']), ('v62-session-error', 1, True, 1))
check('keyword arguments passed through', agent._calls[0], dict(valid_actions=['UP'], should_stop=None))
for _ in range(A.PROTOCOL_RESTARTS - 1):
    A.V6Agent.analyze(agent, 'sp', 5)
try:
    A.V6Agent.analyze(agent, 'sp', 5)
    raised = False
except PersistentProtocolError:
    raised = True
check('past the cap the error ends the game as before', (raised, agent._logs[-1][1]['restart'],
                                                          agent._v6_stats['protocol_restarts']), (True, False, A.PROTOCOL_RESTARTS + 1))
# 2. the other session errors are covered; an ordinary harness bug is not
for exc in (requests.ConnectionError('Connection aborted'), CapacityRefusal('capacity'), A.AppendOnlyViolation('order')):
    agent = make_agent(exc)
    res = A.V6Agent.analyze(agent, 'sp', 1)
    check('covered: %s' % type(exc).__name__, (res.retryable_failure, agent._v62_force_restart), (True, True))
agent = make_agent(KeyError('bug'))
try:
    A.V6Agent.analyze(agent, 'sp', 1)
    raised = False
except KeyError:
    raised = True
check('a harness bug still propagates', raised, True)
# 3. the forced restart skips the swap-out and the late gate
src = inspect.getsource(A.V6Agent._v62_analyze)
check('forced restart: no swap-out, no late gate', ("forced = getattr(self, '_v62_force_restart', False)" in src
                                                    and 'if forced or due:' in src   # round 5l (item 111)
                                                    and 'if not forced and self._v62_swap_out' in src), True)
# 4. fake-session injection (dry runs)
os.environ['ARC3_V62_FAKE_PROTOCOL_ERROR'] = '1'
import importlib
import v62.fake_session as fs
importlib.reload(fs)
try:
    fs.FakeSession(seed=1).generate([])
    raised = False
except PersistentProtocolError:
    raised = True
check('fake session injects the g047 error', raised, True)
# 5. runner: SIGTERM / SIGUSR1 go through taaf's request_stop
src = open(os.path.join(os.path.dirname(A.__file__), 'run_v6.py')).read()
check('runner maps SIGTERM / SIGUSR1 to request_stop', all(x in src for x in (
    'loop.add_signal_handler(sig, stop, sig.name)', 'bm.request_stop()', 'except asyncio.CancelledError:')), True)
# 6. Rule Lk capture (C50 g014, g023, g029, g043, g001)
class D:
    pass


def cap(texts, n, turns=2):
    d = D(); d._v6_rule_capture = n; d._v62_rule_turns = turns; d._v6_rules = {}; d._v6_stats = {}; d._v6_rules_dirty = False
    A.V6Agent._v6_capture_rule(d, texts)
    return d._v6_rules.get(n), d._v6_rule_capture


real = 'Rule L1:\n- the blue token moves one 4-cell tile per press\n- walls block; the goal is the orange slot'
check('g014: empty printed body does not close; reasoning read', cap(['', 'Rule L1:\n\nmore', real], 1),
      ('- the blue token moves one 4-cell tile per press\n- walls block; the goal is the orange slot', 1))   # round 5l:
      # a summary found only in the reasoning is provisional (the capture stays open for one in a reply)
check('g023: stub alone keeps the capture open', cap(['Rule L1: ...\nmore thinking'], 1), (None, 1))
check('g023: stub then real in one text -> the real one', cap(['Rule L1: ...\n\n' + real], 1)[0],
      '- the blue token moves one 4-cell tile per press\n- walls block; the goal is the orange slot')
check('g029: printed placeholder, reasoning summary', cap(['', 'Rule L7 recap: see above', real.replace('L1', 'L7')], 7)[0],
      '- the blue token moves one 4-cell tile per press\n- walls block; the goal is the orange slot')
six = 'Rule L1:\n1. alpha moves right\n2. beta moves left\n3. gamma blocks\n4. delta opens\n5. epsilon fills\n6. GOAL: orange funnel slot'
check('g043: six items kept (10 since round 5g)', cap([six], 1)[0].endswith('6. GOAL: orange funnel slot'), True)
check('g001: parenthesised header, no trailing paren', cap(['(Rule L1: the bar fills when the ring closes around the key)'], 1)[0],
      'the bar fills when the ring closes around the key')
check('last capture turn without a summary closes it', cap(['nothing here'], 1, turns=1), (None, None))
src = inspect.getsource(A.V6Agent._v61_takeover_text)
check('g040: restart opener shows every completed level', ('[-5:]' in src, 'if lvl < int(level)]' in src), (False, True))
# 7. restart archive keeps earlier-level versions (C50 g016, g020, g022, g042)
ag = object.__new__(A.V6Agent)
ag._v6_library_order = ['bar', 'tilemap', 'probe']
ag._kept_functions = {'bar': 'def bar():\n    return 2\n', 'tilemap': 'def tilemap():\n    return 1\n', 'probe': 'def probe():\n    return 3\n'}
ag._v61_library_levels = {'bar': 2, 'tilemap': 1, 'probe': 2}
ag._v6_library_steps = {'bar': 90, 'tilemap': 10, 'probe': 95}
ag._v61_archive = {}
ag._v62_pre_level = {'bar': (1, 'def bar():\n    return 1\n', 12)}
arch, rest = A.V6Agent._v62_archive_level_programs(ag, 2)
check('redefined L1 program restored, new L2 program archived', (arch, rest), (['probe'], ['bar']))
check('restored source and level', (ag._kept_functions['bar'], ag._v61_library_levels['bar'], ag._v6_library_steps['bar']),
      ('def bar():\n    return 1\n', 1, 12))
check('untouched L1 program kept; library order', (sorted(ag._kept_functions), ag._v6_library_order), (['bar', 'tilemap'], ['bar', 'tilemap']))
check('archive holds only the new L2 program', sorted(ag._v61_archive), ['probe'])
# 8. RESET-not-first refusal text (C50 g013, g017, g047)
from inference.agent import python_tool_sandbox as sbx
boot = sbx._SANDBOX_BOOTSTRAP
check('sandbox rewrites the RESET refusal sentence', ('RESET_FIRST_TEXT' in boot   # round 5l (B7): the whole text
                                                     and v62.patches.RESET_FIRST_TEXT[1] in boot), True)
# 9. wording (C50 g007, g023: revert stop "the batch"; g008, g018, g023: "levels 1")
check('revert stop ends the action() list', 'that action() list stopped there' in A.TEXTS['revert_stop'], True)
lb = object.__new__(A.V6Agent)
lb._v6_library_order = ['f', 'g']; lb._kept_functions = {'f': 'def f(): pass', 'g': 'def g(): pass'}
lb._v61_library_levels = {'f': 1, 'g': 1}
check('library count, one level', 'written on level 1;' in A.V6Agent._v61_library_count_line(lb), True)
lb._v61_library_levels = {'f': 1, 'g': 2}
check('library count, two levels', 'written on levels 1-2;' in A.V6Agent._v61_library_count_line(lb), True)
# 10. round 5d wording: the interior by rows and columns, restarts without a GAME OVER, edge-line counts
S5 = A.TEXTS['system']
check('system: interior defined once by rows and columns', S5.count('The interior is rows 4-59 and columns 4-59'), 1)
check('system: no "more than 4 cells" left', 'more than 4 cells' in S5, False)
check('system: restart entries only after GAME OVER', 'A return to the level\'s starting board without a GAME OVER has no entry' in S5, True)
check('no-op stop sentence', A._NOOP_LAST in A.TEXTS['noop_stop'] and 'in the interior' in A._NOOP_LAST, True)
check('edge line (round 5f: the band is defined in the system prompt only)', A.TEXTS['edge_line'], 'Outer band in your previous call: {text}.')   # superseded 2026-10-06 (round 5f S2)
et = A._edge_text([dict(v61_edge=dict(n=10, tr=[('a', 'b', 5), ('c', 'd', 3), ('e', 'f', 2)], rows=[0], cols=[1, 2]), gameplay_changed=True)])
check('edge line counts add up', 'a->b x5, c->d x3, other colour changes x2' in et, True)
# 11. user decisions 2026-10-05 (C50 review): interior change facts, near-start and since-start facts, refusals
from inference.framework import solver as sv
check('effect: interior count and span', sv._action_verdict(dict(gameplay_changed=True, interior_changed=dict(n=6, rows=[2, 5], cols=[2, 5]))),
      'effect | 6 cells changed in the interior (rows 2-5, cols 2-5)')
check('effect: one cell', sv._action_verdict(dict(gameplay_changed=True, interior_changed=dict(n=1, rows=[7, 7], cols=[9, 9]))),
      'effect | 1 cell changed in the interior (row 7, col 9)')
check('near start', sv._action_verdict(dict(gameplay_changed=True, near_level_start=2, actions_since_start=44)),
      "effect | the interior now differs from the level's starting board in only 2 cells, with no GAME OVER reported, 44 "
      "actions after the board was last at the level's start")
check('start match with count', sv._action_verdict(dict(gameplay_changed=True, level_start_match='interior', edge_cells_restored=3,
                                                         actions_since_start=44)),
      "effect | the interior now matches the level's starting board, with no GAME OVER reported, 44 actions after the board "
      "was last at the level's start; 3 cells of the outer band also returned to their starting values")
import v62.session_ext as se
g0 = [[0] * 16 for _ in range(16)]
g1 = [r[:] for r in g0]; g1[5][6] = 1; g1[9][10] = 2; g1[1][1] = 3
check('_interior_changes ignores the band', se._interior_changes(g0, g1), (2, [5, 9], [6, 10]))
check('_interior_changes: shapes differ', se._interior_changes(g0, [[0] * 8]), None)
txt = A._refusal_text([dict(executed=False, stop_reason='reset_not_first'), dict(executed=False, stop_reason='stale_state'),
                       dict(executed=False, stop_reason='reset_not_first')])
check('refusal text', txt, '3 action() calls in this Python call were refused and executed nothing: RESET was not the first '
                           'executed action of the snippet (2); the previous action changed nothing on the board '
                           '(stale-state refusal) (1).')   # round 5l (item 102)
ag = object.__new__(A.V6Agent)
pl = {}
A.V6Agent._record_retained_functions(ag, {'action_results': [dict(executed=False, stop_reason='reset_not_first'),
                                                             dict(executed=True, stop_reason=None)]}, pl)
check('refusal stated in the tool result', pl.get('refused_action_calls', '').startswith('1 action() call in this Python call was refused'), True)
check('system documents the new fields', all(k in A.TEXTS['system'] for k in ('`interior_changed`', '`actions_since_start`', '`near_level_start`')), True)
src5 = inspect.getsource(A.V6Agent._v62_analyze)
check('a game that waited for a lane past the end ends cancelled, not crashed',
      ('return ta.AnalyzerTurnResult(step_executed=False)' in src5, 'return None                                   # the run ended' in src5), (True, False))
print('FAILS:', fails)
