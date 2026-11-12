"""v6.2 round 5h unit tests (run 386 copycat-30 review, Fable's reply and the user's decisions, 2026-10-06 ~23:00; V62_BUILD
items 80-): turn budget and yield defaults, the restart clause (D1), attempts from the level's start (D2), game-wide
game-over numbers (G6), summary capture (G1), programs kept on rebinding (G2), the band-crossing fact (G3), fact
locations (G4), the refused-call footnote (G5), cosmetic labels (G7), G8, the no-change line (G10), the stale refusal of a
repeated click and after REVERTED, level_completed on a win, the animation timeline, the blocked-builtin note."""
import ast
import builtins
import inspect
import os
import traceback
import types
from types import SimpleNamespace
for k, v in dict(ARC3_V61_TEXTS='1', ARC3_V6_A13='1', ARC3_V61_SUMMARY='1', ARC3_NEW_CHANGED_PROMPTS='1',
                 ARC3_ACTION_INFO='1', ARC3_REPEAT_STATE_GUARD='0', ARC3_V61_RESTART_TOKENS='150000', ARC3_V6_A3='1',
                 ARC3_GUARDS_FROM_LEVEL='2', ARC3_BATCH_NOOP_BLOCK='1', ARC3_STALE_STATE_BLOCK='1',
                 ARC3_ANIMATION='1').items():
    os.environ.setdefault(k, v)
import v62.patches  # noqa: F401
import v62.retention as r
PATCHED_RF = r.install()
import v62.agent as A
import v62.run_v6 as RUN
import v62.session_ext as se
import v62.transport as T
from inference.agent import python_tool_sandbox as sb
from inference.agent.runtime_state import Frame
from inference.framework import solver as sv
fails = []


def check(name, got, want):
    ok = got == want
    print('%-66s -> %-44r %s' % (name, got if not isinstance(got, str) else got[:44], 'OK' if ok else 'WRONG (want %r)' % (want,)))
    if not ok:
        fails.append(name)


S = A.TEXTS['system']
BOOT = sb._SANDBOX_BOOTSTRAP

# --- defaults: turn caps (D3) and the yield ------------------------------------------------------------------------
check('yield default 4,096 (user 10-07 ~02:15; 8,192 in 387b only)', A.YIELD_TOKENS, 4096)
# item 127 (user 2026-10-09: the production harness): the runner default is the production 12K cue + 16K cap; the
# checks below still run round 5h's no-cue configuration explicitly
check('runner default caps 12000,16000', "os.environ.setdefault('ARC3_ASSISTANT_TURN_CAPS', '12000,16000')" in inspect.getsource(RUN), True)
os.environ['ARC3_ASSISTANT_TURN_CAPS'] = '0,16000'
check('caps parse with soft 0', T.turn_caps_from_env()[:2], (0, 16000))
check('no soft cue past 8K', T.turn_cap_decision(9000, False, 0, 16000, False, check=8000), (7000, None, None))
check('hard close at 16K', T.turn_cap_decision(16000, False, 0, 16000, False, check=8000)[1]['kind'], 'hard_close')

# --- texts ------------------------------------------------------------------------------------------------------------
check('D1 restart clause', A._fill(A.TEXTS['takeover_restart'], level=3, step=120),
      'This conversation begins after a RESET of level 3 at step 120; your earlier attempts on this level are not in this '
      'conversation or in `history`.')   # round 5l (item 109)
check('D1 in the takeover text', "TEXTS['takeover_restart'] if hidden else TEXTS['takeover_restart_shown']" in   # 5l review
      inspect.getsource(A.V6Agent._v61_takeover_text), True)
check('G8 summary promise', 'The saved summary is shown to you again if this game restarts a level.' in A.TEXTS['levelup'],
      True)   # round 5l (item 108)
check('end-of-run line', '- No closing report is expected: keep playing until the harness ends the game, however little '
      'run time is left.' in S, True)
check('G4 where the facts are', "These facts are in each executed action's result: its `history` entry's `.result`, and for "
      "the last executed action of a call also the top level of `last_action_call_result`, which has no per-action result "
      "records (its per-action lists are `executed_actions`, `action_trace` and `gameplay_changed_per_action`)." in S, True)
check('G4 interior_changed always', "Every executed action has `interior_changed`" in S and 'n is 0 when none did)' in S, True)
check('History note', "The entry of an action that completed a level holds the next level's first board." in S, True)
check('G3 documented (v6.3 G sub-bullet)', '`band_connected` (how many band cells): when changed cells of the outer band connect' in S, True)
check('stale: same click refused', 'after a click that changed nothing only the same click (same cell) is refused' in S, True)
check('stale: old exemption gone', 'A click or RESET that changed nothing does not trigger this.' in S, False)
check('timeline documented as a dict', '`last_animation_timeline` is a dict: `action` (a string), `frames` (the number of '
      'frames, an int) and `steps` (a list,' in S, True)   # round 5l (item 112)
check('no-change line', A.TEXTS['no_change'], 'No cell differs from the board at your previous turn.')
check('system prompt size (chars)', 18000 < len(S) < 21000, True)   # round 5l: ~+1,000 characters

# --- D2 / G6: restart bookkeeping -------------------------------------------------------------------------------------
src = inspect.getsource(A.V6Agent._v61_restart)
check('D2: no attempt offset set at a restart', "v61_attempt_offset'" in src, False)
check('G6: the ledger total is not reset', "'_death_ledger_total'" in src, False)


# --- G1: summary capture ----------------------------------------------------------------------------------------------
class D:
    pass


def agent_for_capture(n=3, turns=2):
    d = D(); d._v6_rule_capture = n; d._v62_rule_turns = turns; d._v6_rules = {}; d._v6_stats = {}; d._v6_rules_dirty = False
    return d


def cap(texts, n=3, turns=2):
    d = agent_for_capture(n, turns)
    A.V6Agent._v6_capture_rule(d, texts)
    return d._v6_rules.get(n)


items = ['- the red block is pushed by the player', '- walls stop the block']
check('G1 header line ending in ":" is all header', cap('Rule L3: mechanics so far:\n' + '\n'.join(items)), '\n'.join(items))
check('G1 kx59 quoted tag in a header', cap('Rule summary (for the required "Rule L3:" lines):\n' + '\n'.join(items)),
      '\n'.join(items))
check('G1 one-line summaries merged', cap('Rule L3: the red block is pushed by the player.\n\nRule L3: walls stop the block '
      'from moving.'), '- the red block is pushed by the player.\n- walls stop the block from moving.')
check('G1 repeated lines merged once', cap('Rule L3: the red block is pushed by the player.\n\nRule L3: the red block is '
      'pushed by the player.'), '- the red block is pushed by the player.')
check('G1 repeated level tag stripped', cap('Rule L3: L3: the red block is pushed onto the green pad.'),
      'the red block is pushed onto the green pad.')
check('G1 tag stripped from a list item', cap('Rule L3:\n- L3: the red block is pushed by the player\n- walls stop it'),
      '- the red block is pushed by the player\n- walls stop it')
d = agent_for_capture()
A.V6Agent._v6_capture_rule(d, ['Rule L3: the red block is pushed onto the green pad.'], provisional=True)
check('G1 provisional capture stored, capture still open', (d._v6_rules.get(3), d._v6_rule_capture, d._v62_rule_turns),
      ('the red block is pushed onto the green pad.', 3, 2))
A.V6Agent._v6_capture_rule(d, ['Rule L3:\n' + '\n'.join(items)])
check('G1 the post-level-up summary overrides it and closes',
      (d._v6_rules.get(3), d._v6_rule_capture, d._v6_stats.get('rules_recorded'), d._v6_stats.get('rules_from_clearing_turn')),
      ('\n'.join(items), None, 1, 1))
d = agent_for_capture()
A.V6Agent._v6_capture_rule(d, ['no summary here'], provisional=True)
check('G1 provisional miss changes nothing', (d._v6_rules, d._v6_rule_capture, d._v62_rule_turns), ({}, 3, 2))
words = lambda text: len(text.split())
got = A._cap_rule('- ' + ' '.join(['w'] * 5000) + '\n- short item', words)
check('G1 cap: one item over the budget is cut at a word (review)', (got.startswith('- w w'), got.endswith(' ...'),
                                                                    words(got) <= A.SUMMARY_TOKENS), (True, True, True))
check('G1 provisional capture armed at the level-up', 'self._v6_capture_rule(self._v62_last_texts, provisional=True)' in
      inspect.getsource(A.V6Agent._build_user_prompt), True)

# --- G2: a saved program survives a rebinding of its name -------------------------------------------------------------
RF = {}
exec(compile(PATCHED_RF, '<retained_functions v61>', 'exec'), RF)
AVAILABLE = set(dir(builtins)) | {'action', 'current_frame', 'history'}
PREV = {'cc': 'def cc(x):\n    return x + 1', 'dd': 'def dd(x):\n    return cc(x) * 2'}


def collect(code):
    ns = {'__builtins__': builtins}
    for s in PREV.values():
        exec(compile(s, '<retained>', 'exec'), ns)
    exec(compile(code, '<python_tool>', 'exec'), ns)
    kept, rejected = RF['collect_functions'](code, dict(PREV), ns, AVAILABLE)
    return kept, {x['name']: x['reason'] for x in rejected}


kept, rej = collect('for cc in range(3):\n    pass\ny = dd')
check('G2 loop variable: cc and its dependent kept', (sorted(kept), rej.get('cc')), (['cc', 'dd'], '__v62_kept_rebound__'))
kept, rej = collect('del cc')
check('G2 deleted name kept with the note', ('cc' in kept, rej.get('cc')), (True, '__v62_kept_rebound__'))
kept, rej = collect('def cc(x):\n    return x * 3')
check('G2 a new def still replaces it', (kept.get('cc'), 'cc' in rej), ('def cc(x):\n    return x * 3', False))
kept, rej = collect('y = dd(2)')
check('G2 untouched programs kept silently', (sorted(kept), rej), (['cc', 'dd'], {}))
check('G2 note text', A._v6_retention_notes('Your function cc was not retained: __v62_kept_rebound__.'),
      '`cc` was bound to another value or deleted in this call without a new definition; the saved program `cc` is kept '
      'for later calls.')

# --- G3: the band-crossing fact ---------------------------------------------------------------------------------------
N = 16
a = [[0] * N for _ in range(N)]
b = [row[:] for row in a]
for rr in range(2, 7):
    b[rr][6] = 5                     # rows 2-3 in the band, 4-6 in the interior: one region
b[0][12] = 7                         # a band change elsewhere
b[2][9] = 7                          # diagonal to nothing changed in the interior
check('G3 band cells connected to the interior change', se.band_connected(a, b), 2)
check('G3 none when the regions are apart', se.band_connected(a, [[7 if (rr, c) == (0, 0) else 0 for c in range(N)]
                                                                   for rr in range(N)]), 0)
check('G3 in the result keys', 'band_connected' in se.EXTRA_RESULT_KEYS, True)
check('G3 computed for non-terminal interior changes', 'k = band_connected(previous_grid, new_grid)' in
      inspect.getsource(se.execute_action), True)
v = sv._action_verdict(dict(gameplay_changed=True, interior_changed=dict(n=3, rows=[4, 6], cols=[6, 6]), band_connected=2))
check('G3 verdict', v, 'effect | 3 cells changed in the interior (rows 4-6, col 6); 2 cells of the outer band also changed '
      'and are connected to them')
e = A._edge_text([dict(v61_edge=dict(n=3, tr=[['0', '5', 2], ['0', '7', 1]], rows=[0, 2, 3], cols=[6, 12]),
                       gameplay_changed=True, band_connected=2)])
check('G3 outer-band line', e.endswith('; 2 of the changed edge cells are connected to cells that changed in the interior'), True)
check('G3 in the compact result', "'band_connected'" in inspect.getsource(A.V6Agent._compact_action_result), True)

# --- G5: the refused-call footnote ------------------------------------------------------------------------------------
t = A._refusal_text([dict(stop_reason=None, requested_actions=['UP', 'D'],
                          error="Unknown action at position 2 (1-based) of the list: 'D'. Valid actions now: UP, DOWN.")])
check('G5 footnote names the call and the message', t.endswith("The first was UP, D: Unknown action at position 2 (1-based) "
                                                               "of the list: 'D'. Valid actions now: UP, DOWN."), True)
check('G5 footnote: a message without a final stop gets one', A._refusal_text([dict(stop_reason='stale_state',
      requested_actions=['UP'], stop_detail='not executed')]).endswith('The first was UP: not executed.'), True)


# --- step_env with a scripted session: stale refusals, the REVERTED pair, the no-op stop, positions -----------------
class Session:
    def __init__(self, payloads=()):
        self.game = SimpleNamespace(current_state=SimpleNamespace(available_actions=[1, 2, 3, 4, 5, 6]))
        self.history_entries, self.payloads, self.executed = [], list(payloads), []

    def _normalize_actions(self, arguments):
        return sv._HarnessGameSession._normalize_actions(self, arguments)

    def should_stop(self):
        return False

    def _error_payload(self, message):
        return {'executed': False, 'error': message}

    def write_viewer_payload(self):
        pass

    def _execute_action(self, action, *, batch_index, batch_size, flush_viewer_payload=True):
        p = dict(self.payloads.pop(0) if self.payloads else dict(gameplay_changed=True))
        disp = sv._format_action_display(action.id.name, dict(action.data))
        p.update(executed=True, action_display=disp, action_name=action.id.name)
        self.executed.append(disp)
        self.history_entries.append(SimpleNamespace(action=disp, result=p))
        return p


sv._level_number = lambda game: 2
sv._is_engine_game_over = lambda game: False
sv._engine_action_names = lambda game: ['ACTION1', 'ACTION2', 'ACTION3', 'ACTION4', 'ACTION5', 'ACTION6']
MOUSE = lambda rr, c: dict(action='MOUSE', row=rr, col=c)


def step(session, actions, stale=None):
    return se.step_env(session, dict(actions=actions, stale_after=stale))


s = Session()
res = step(s, [MOUSE(5, 5)], 'MOUSE(row=5, col=5)')
check('stale: the same click is refused', (res.get('stop_reason'), s.executed), ('stale_state', []))
check('stale: its detail names the click', res.get('stop_detail', '').startswith('MOUSE(row=5, col=5) was not executed.'), True)
s = Session()
res = step(s, [MOUSE(6, 6)], 'MOUSE(row=5, col=5)')
check('stale: another cell executes', (res.get('stop_reason'), s.executed), (None, ['MOUSE(row=6, col=6)']))
s = Session()
res = step(s, [dict(action='UP')], 'MOUSE(row=5, col=5)')
check('stale: a move after a no-op click executes', s.executed, ['UP'])
s = Session()
res = step(s, [dict(action='DOWN')], 'UP')
check('stale: after a move no-op any action is refused (unchanged)', (res.get('stop_reason'), s.executed), ('stale_state', []))
check('stale: no-op wording', 'was a NO-OP - nothing on the board changed -' in res.get('stop_detail', ''), True)   # round 5l
s = Session()
s.history_entries.append(SimpleNamespace(action='UP', result=dict(reverted=True, gameplay_changed=False)))
res = step(s, [dict(action='DOWN')], 'UP')
check('stale after REVERTED: wording', ('was REVERTED - it animated, then the board returned to its previous state -'   # round 5l
                                        in res.get('stop_detail', ''), 'no-op' in res.get('stop_detail', '')), (True, False))
check('stale after REVERTED: what to work out', 'work out why the board returned to its previous state' in   # round 5l
      res.get('stop_detail', ''), True)

REV = dict(gameplay_changed=False, reverted=True)
s = Session([REV, REV, REV])
res1 = step(s, [dict(action='UP'), dict(action='DOWN')])
res2 = step(s, [dict(action='LEFT')])
check('G7 REVERTED pair in one call', res1.get('reverted_pair'), ['UP', 'DOWN'])
check('G7 a third REVERTED names the last two', res2.get('reverted_pair'), ['DOWN', 'LEFT'])
check('G7 stop detail names the last two', '(DOWN, LEFT)' in res2.get('stop_detail', ''), True)

s = Session([dict(gameplay_changed=True), dict(gameplay_changed=False)])
res = step(s, [dict(action='UP'), dict(action='DOWN')])
check('G7 no-op stop at the end: no empty skip line', (res.get('stop_reason'), 'SKIPPED' in res.get('stop_detail', '')),
      ('no_op_action', False))
s = Session([dict(gameplay_changed=True), dict(gameplay_changed=False)])
res = step(s, [dict(action='UP'), dict(action='DOWN'), dict(action='LEFT')])
check('G7 no-op stop with a skip keeps it', "Action 3 ('LEFT') was SKIPPED" in res.get('stop_detail', ''), True)   # round 5l: singular

s = Session()
res = step(s, [dict(action='UP'), dict(action='D')])
check('G7 unknown action: 1-based position and the valid actions', res.get('error'),
      "Unknown action at position 2 (1-based) of the list: 'D'. Valid actions now: UP, DOWN, LEFT, RIGHT, SPACE, MOUSE.")
check('G7 MOUSE without row', se._position_text(s, 'MOUSE action at index 1 requires integer row and col arguments.'),
      'MOUSE action at position 1 (1-based) requires integer row and col arguments.')
check('G7 sandbox messages say position (1-based)', ('f"The action at position {index} (1-based) must be a string or a '
                                                     'dict."' in BOOT, 'f"Action {index} ' in BOOT), (True, False))

# --- the sandbox's stale arming: MOUSE no-ops arm it, each executed call replaces it ---------------------------------
check('arming: every executed call clears the flag first', 'if action_result.get("executed"):          # round 5h' in BOOT
      and '    del _stale_state[:]\n' in BOOT and 'if (gameplay_changed is False and action_result.get("executed")' in BOOT,
      True)   # round 5l: comment lines between
check('arming: only RESET exempt', ('.startswith("RESET"))' in BOOT, '.startswith(("MOUSE", "RESET"))' in BOOT), (True, False))

# --- G7: generator frames in library functions -------------------------------------------------------------------------
tree = ast.parse(BOOT)
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_sanitize_exception')
NS = {'traceback': traceback}
exec(compile(ast.Module(body=[fn], type_ignores=[]), '<bootstrap>', 'exec'), NS)
LIB = {}
exec(compile('def total(xs):\n    return sum(1 / x for x in xs)\n', '<retained>', 'exec'), LIB)
try:
    exec(compile('total([1, 0])\n', '<python_tool>', 'exec'), dict(LIB))
except ZeroDivisionError as exc:
    tb_text = NS['_sanitize_exception'](exc)
check('G7 genexpr frame named with its library function',
      '  <genexpr> in library function total, line 2 of its source (read_code("total") shows it)' in tb_text, True)
check('G7 the library function frame as before', '  Library function total, line 2 of its source' in tb_text, True)
check('G7 no read_code("<genexpr>")', 'read_code("<genexpr>")' in tb_text, False)

# --- G10: the no-change line ------------------------------------------------------------------------------------------
ag = object.__new__(A.V6Agent)
grid = [[0] * 8 for _ in range(8)]
ag._v6_prev_opener_grid = [row[:] for row in grid]
op = A.V6Agent._v6_opener(ag, 'text', 'same', Frame(grid=grid, step=3, level=1), [])
texts = [p.get('text') for p in op['content'] if isinstance(p, dict) and p.get('type') == 'text']
check('G10 unchanged board: the line, no change caption', (A.TEXTS['no_change'] in texts,
                                                         any('Change since your previous turn' in (x or '') for x in texts)), (True, False))
ag._v6_prev_opener_grid = [[1] * 8 for _ in range(8)]
op = A.V6Agent._v6_opener(ag, 'text', 'same', Frame(grid=grid, step=4, level=1), [])
texts = [p.get('text') for p in op['content'] if isinstance(p, dict) and p.get('type') == 'text']
check('G10 changed board: no such line', A.TEXTS['no_change'] in texts, False)
ag._v6_prev_opener_grid = [row[:] for row in grid]
op = A.V6Agent._v6_opener(ag, 'text', 'levelup', Frame(grid=grid, step=5, level=2), [])
texts = [p.get('text') for p in op['content'] if isinstance(p, dict) and p.get('type') == 'text']
check('G10 not on a level-up opener', A.TEXTS['no_change'] in texts, False)

# --- level_completed on the winning action; the timeline; the NameError note; the bootstrap compiles -------------------
check('WIN action completes its level', 'level_completed = bool(new_state.just_won_level or raw_state == arcengine.GameState.WIN)'
      in inspect.getsource(se.execute_action), True)
check('verdict on a win stays RUN COMPLETE', sv._action_verdict(dict(level_completed=True, run_complete=True, done=True)),
      'RUN COMPLETE')
check('timeline view classes', ('class LazyTimelineDict(_LazyAnimation):' in BOOT, 'class LazyFrameList(_LazyAnimation):' in BOOT,
                                'runtime_globals["last_animation_timeline"] = LazyTimelineDict("timeline", {})' in BOOT),
      (True, True, True))
check('blocked builtin note', A._v6_error_text("Traceback (most recent call last):\nNameError: name 'globals' is not defined"),
      "Traceback (most recent call last):\nNameError: name 'globals' is not defined\n`globals` is a Python builtin that this "
      "sandbox does not provide.")
check('a lost variable keeps the persistence note', A._v6_error_text("NameError: name 'grid2' is not defined").endswith(
    A.TEXTS['nameerror_note']), True)
try:
    compile(BOOT, '<bootstrap>', 'exec')
    ok = True
except SyntaxError as exc:
    ok = repr(exc)
check('patched sandbox bootstrap compiles', ok, True)
# --- review fix after 387b: a recovered malformed call is named in its tool result ------------------------------------
check('recovered note text', A.TEXTS['markup_recovered'].startswith('Note: this tool call was malformed (the python tool takes '
                                                                     'one argument, `code`'), True)
check('note first in a text result', A._v62_with_note('{"tool": "python"}', 'N'), 'N\n{"tool": "python"}')
check('note first in a result with images', A._v62_with_note([dict(type='text', text='x'), dict(type='image_url')], 'N')[0],
      dict(type='text', text='N'))
src = inspect.getsource(A.V6Agent._v62_analyze)
check('both recovery paths mark the call', src.count("call['_v62_recovered'] = True"), 2)
# round 5i (item 98): the note may name the missing tag (superseded check)
check('the note goes into that call\'s tool result', "content = _v62_with_note(content, note)" in src, True)
check('levelup: header line first, items enumerated', 'save a summary with save_summary(text): a text that begins with the '
      'line "Rule L{level-1}:" and then enumerates, in at most 10 items, the game mechanics established so far and what cleared level '
      '{level-1}. Keep the summary within 4,000 tokens; a longer one is truncated.' in A.TEXTS['levelup'], True)
check('levelup: no "one item per line"', 'one item per line' in A.TEXTS['levelup'], False)
check('capture: an inline enumeration after the header', cap('Rule L3:\n(1) the red block slides until it hits a wall, (2) keys '
      'open doors of their colour, (3) the exit clears the level.'), '(1) the red block slides until it hits a wall, (2) keys open '
      'doors of their colour, (3) the exit clears the level.')
check('levelup: no "bullet list" wording left', 'bullet list' in A.TEXTS['levelup'], False)
for form in ('- ', '1. ', '2) ', '* '):
    got = cap('Rule L3:\n' + '\n'.join(form.replace('1', str(i + 1)).replace('2', str(i + 1)) + 'the red block moves %d cells right' % i
                                      for i in range(3)))
    check('capture after the header line, items as %r' % form, (got or '').count('\n'), 2)
print('FAILS:', fails)
