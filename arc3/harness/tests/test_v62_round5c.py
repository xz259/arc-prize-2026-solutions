"""v6.2 round 5c unit tests (B3 review: Rule Lk capture boundaries, band wording, no-op stop sentence, stale wording,
segmentation wording, plurals)."""
import json
import os
for k, v in dict(ARC3_V61_TEXTS='1', ARC3_V6_A13='1', ARC3_V61_SUMMARY='1', ARC3_NEW_CHANGED_PROMPTS='1',
                 ARC3_ACTION_INFO='1', ARC3_REPEAT_STATE_GUARD='0', ARC3_V61_RESTART_TOKENS='150000', ARC3_V6_A3='1').items():
    os.environ.setdefault(k, v)
import v62.patches  # noqa
import v62.retention as r; r.install()
import v62.agent as A
from types import SimpleNamespace
from inference.framework import solver as sv
fails = []


def check(name, got, want):
    ok = got == want
    print('%-55s -> %-50r %s' % (name, got if not isinstance(got, str) else got[:50], 'OK' if ok else 'WRONG (want %r)' % (want,)))
    if not ok:
        fails.append(name)


class D:
    pass


def cap(texts, n):
    d = D(); d._v6_rule_capture = n; d._v62_rule_turns = 1; d._v6_rules = {}; d._v6_stats = {}; d._v6_rules_dirty = False
    A.V6Agent._v6_capture_rule(d, texts)
    return d._v6_rules.get(n)


# 1. capture boundaries
check('next-level planning cut (ka59 form)',
      cap(['Rule L1: - Two rings; the selected one moves 3 cells.\n- Goal: fill the boxes. So yes: fill all boxes. Now level 2. '
           'The board: mostly gray.'], 1),
      '- Two rings; the selected one moves 3 cells.\n- Goal: fill the boxes. So yes: fill all boxes.')   # round 5g: list lines
check('next-level cut on its own line (su15 form)',
      cap(['Rule L1: token moves along the rope (tested).\nLevel 2: A blue ball at centre, no socket.'], 1),
      'token moves along the rope (tested).')
check('cut keeps "(tested)" before "Now level" (bp35 form)',
      cap(['Rule L1: 5. Goal: reach the plus -> level complete. (tested) Now level 2: - 3 purple blocks'], 1),
      '5. Goal: reach the plus -> level complete. (tested)')
check('earlier levels are not cut', cap(['Rule L3: arrows move 3 cells (tested on level 1 and level 2)'], 3),
      'arrows move 3 cells (tested on level 1 and level 2)')
check('"level 12" is not level 2', cap(['Rule L1: arrows move the token\n- seen on level 12 too, same arrows'], 1),
      'arrows move the token\n- seen on level 12 too, same arrows')   # round 5d: under 20 letters skipped; 5g: list lines
check('ten following lines kept (round 5g; 5d had 8)', cap(['Rule L3: alpha\n- bravo\n- charlie\n- delta\n- echo\n- foxtrot\n- golf\n- hotel\n- india\n- juliet'], 3),
      'alpha\n- bravo\n- charlie\n- delta\n- echo\n- foxtrot\n- golf\n- hotel\n- india\n- juliet')
check('untagged "Rule summary (5 lines):" (ls20)', cap(['Rule summary (5 lines):\n- block moves 5 px\n- walls block'], 5),
      '- block moves 5 px\n- walls block')
check('tagged header preferred over untagged', cap(['Rule summary: generic words for the untagged form', '',
                                                    'Rule L5: tagged words that state the level five rule'], 5),
      'tagged words that state the level five rule')
check('"Maybe the rule:" still no match', cap(['Maybe the rule: no'], 2), None)
long = 'Rule L2: ' + ' '.join('word%d' % i for i in range(6000))     # round 5g: the cap is 4,000 tokens (estimate here)
got = cap([long], 2)
check('long summary cut at a word, marked', (got.endswith(' ...'), (len(got) + 3) // 4 <= A.SUMMARY_TOKENS, got[-5:-4] != ' '), (True, True, True))
calls = [{'function': {'name': 'python', 'arguments': json.dumps({'code': "print('Rule L2: arrows move 3 cells')\n"
                                                                            "print('- SPACE cycles')\n"
                                                                            "print('tokens', W, 'id')\nprint('more')"})}}]
check('_printed_strings: computed print gives a blank line', A._printed_strings(calls),
      "Rule L2: arrows move 3 cells\n- SPACE cycles\n\nmore")
check('printed summary ends at the computed print (re86)', cap(['', A._printed_strings(calls), ''], 2),
      'arrows move 3 cells\n- SPACE cycles')

# 2. verdict wording
check('level-start fact (interior)', sv._action_verdict(dict(gameplay_changed=True, level_start_match='interior', edge_cells_restored=1)),
      "effect | the interior now matches the level's starting board, with no GAME OVER reported; 1 cell of the outer "
      "band also returned to its starting value")   # round 5d wording
check('level-start fact (full, plural)', sv._action_verdict(dict(gameplay_changed=True, level_start_match='full', edge_cells_restored=84)),
      "effect | the whole board now matches the level's starting board, with no GAME OVER reported")   # round 5d: no band clause
check('repeat fact', sv._action_verdict(dict(gameplay_changed=True, repeat_in_call=1)),
      'effect | already executed from this board earlier in this call (1 time; boards compared on the interior)')   # round 5d
check('NO-OP singular cell', sv._action_verdict(dict(gameplay_changed=False, v61_edge=dict(n=1))),
      'BAND-ONLY CHANGE (no change in the interior; 1 cell of the outer band changed)')   # round 5d; 5l (item 102)
check('NO-OP plural cells', sv._action_verdict(dict(gameplay_changed=False, v61_edge=dict(n=22))),
      'BAND-ONLY CHANGE (no change in the interior; 22 cells of the outer band changed)')   # round 5d; 5l (item 102)

# 3. texts
T = A.TEXTS
S = T['system']
check('no-op stop: no "moved anything"', 'moved anything' in T['noop_stop'], False)
check('no-op stop anchor kept', A._NOOP_LAST in T['noop_stop'], True)
check('no-op first wording', 'nothing in the batch moved' in A._NOOP_FIRST, False)
stale = sv._stale_detail(SimpleNamespace(id=SimpleNamespace(name='ACTION1'), data={}), 'UP')
check('stale: no "why nothing moved"', 'why nothing moved' in stale, False)
check('stale: measurement wording', 'work out why nothing changed' in stale, True)   # round 5d; 5l (item 102)
check('segmentation is an attribute (system)', '`.segmentation` is an attribute, not a method' in S, True)
check('segmentation reminder', 'use `current_frame.segmentation` (an attribute) first' in T['reminder'], True)
check('fields doc wording', 'The interior is rows 4-59 and columns 4-59' in S and 'outer 4-cell band' not in S, True)   # round 5d
check('revert stop names the Python call', T['revert_stop'].startswith('Two REVERTED actions in a row in this Python call'), True)
d = D()
line = A.V6Agent._v61_stop_line(d, dict(kind='reverted', pair=['MOUSE(row=1, col=2)', 'MOUSE(row=3, col=4)'],
                                         executed=['MOUSE(row=3, col=4)'], skipped=[]), None, 1)
check('revert stop, nothing skipped', line.endswith('the second was the last action of its action() list, so nothing was '
      'skipped.'), True)   # round 5d wording; 5l (item 107)
d._v61_attempts = lambda session, level: []
check('step line says whole run', A.V6Agent._v61_step_line(d, 85, 2, 'MOUSE', None),
      'Step 85 (actions in the whole run), level 2, attempt 1. Valid actions: MOUSE.')
# 4. death ledger (user: neutralise), MOUSE relative box fact (user: add), late-restart gate (user: under 10 min)
import ast, inspect
import inference.agent.tool_agent as ta
import v62.session_ext as se
src = inspect.getsource(ta.ToolAgent._death_ledger_lines)
import textwrap
lits = [n.value for n in ast.walk(ast.parse(textwrap.dedent(src))) if isinstance(n, ast.Constant) and isinstance(n.value, str)]
advice = next(x for x in lits if x.startswith('Do NOT replay'))
check('ledger advice anchor matches the source', advice.startswith(A.V6Agent._LEDGER_ADVICE), True)
orig = ta.ToolAgent._death_ledger_lines
ta.ToolAgent._death_ledger_lines = lambda self, cf, he, ps: [
    'Known fatal continuations FROM THE CURRENT BOARD STATE (this level has 2 failed attempt(s) recorded; the game is deterministic):',
    '- 3 action(s): UP, UP, LEFT -> GAME OVER (attempt(s) 7, 9)', advice]
try:
    agent = object.__new__(A.V6Agent)
    out = A.V6Agent._death_ledger_lines(agent, None, [], None)
finally:
    ta.ToolAgent._death_ledger_lines = orig
check('ledger: determinism header kept', out[0].endswith('recorded on this level):'), True)   # round 5l (B3): replaced
check('ledger: game-over numbers', out[1], '- 3 action(s): UP, UP, LEFT -> GAME OVER (game over #7, #9 of this game)')
check('ledger: no imperative, no example', ('Do NOT' in out[2], 'held items' in out[2], out[2].endswith('`death_ledger`.')), (False, False, True))
print(out[2])
g = [[0] * 16 for _ in range(16)]
for rr in range(4, 9):
    for cc in range(4, 9):
        if rr in (4, 8) or cc in (4, 8):
            g[rr][cc] = 2
g[8][6] = 0                                            # an open ring: its inside is the same floor component
t_in = se.hit_text(se.mouse_hit(g, 6, 6))
check('MOUSE: floor click inside a ring box names the ring', 'inside the bounding box of component' in t_in and '15 cells, rows 4-8, cols 4-8)' in t_in, True)
print(t_in)
check('MOUSE: click on the ring itself, no smaller box', 'inside the bounding box' in se.hit_text(se.mouse_hit(g, 4, 6)), False)
check('MOUSE: floor click outside every smaller box', 'inside the bounding box' in se.hit_text(se.mouse_hit(g, 12, 12)), False)
agent = object.__new__(A.V6Agent)
agent._v6_stats = {}; agent._v61_level_tokens = 160000; logs = []
agent._v6_log = lambda kind, payload: logs.append((kind, payload))
agent._v61_session = lambda: SimpleNamespace(timing_payload=lambda: {'time_remaining_seconds': 300})
check('restart gate: 5 min left -> no restart', (A.V6Agent._v62_restart_time_ok(agent, 5), A.V6Agent._v62_restart_time_ok(agent, 5)), (False, False))
check('restart gate: counted and logged once per level', (agent._v6_stats.get('restarts_skipped_late'), len(logs), logs[0][0]), (1, 1, 'v62-restart-skipped'))
agent._v61_session = lambda: SimpleNamespace(timing_payload=lambda: {'time_remaining_seconds': 840})
check('restart gate: 14 min left -> no restart (15-min gate)', A.V6Agent._v62_restart_time_ok(agent, 7), False)
# v6.3 item C (user 2026-10-12): the gate is 25 min (ARC3_V62_RESTART_MIN_LEFT 1500); 15 min no longer restarts
agent._v61_session = lambda: SimpleNamespace(timing_payload=lambda: {'time_remaining_seconds': 900})
check('restart gate: 15 min left -> no restart (25-min gate, v6.3 C)', A.V6Agent._v62_restart_time_ok(agent, 6), False)
agent._v61_session = lambda: SimpleNamespace(timing_payload=lambda: {'time_remaining_seconds': 1500})
check('restart gate: 25 min left -> restart', A.V6Agent._v62_restart_time_ok(agent, 8), True)
print('FAILS:', fails)
