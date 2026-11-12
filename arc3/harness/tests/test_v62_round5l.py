"""v6.2 round 5l unit tests (the 392b audit, Fable's review and the user's decisions, 2026-10-08; V62_BUILD items 101-).
usage (pod copy): PYTHONPATH=src:. python3 tests/test_v62_round5l.py"""
import inspect
import os
from types import SimpleNamespace
for k, v in dict(ARC3_V61_TEXTS='1', ARC3_V6_A13='1', ARC3_V61_SUMMARY='1', ARC3_NEW_CHANGED_PROMPTS='1',
                 ARC3_ACTION_INFO='1', ARC3_REPEAT_STATE_GUARD='0', ARC3_V61_RESTART_TOKENS='150000', ARC3_V6_A3='1',
                 ARC3_GUARDS_FROM_LEVEL='2', ARC3_BATCH_NOOP_BLOCK='1', ARC3_STALE_STATE_BLOCK='1', ARC3_DEATH_LEDGER='1',
                 EXPOSE_RESET='on', EXPOSE_UNDO='on', ONLY_RESET_LEVELS='true', ARC3_ANIMATION='1').items():
    os.environ.setdefault(k, v)
import arcengine  # noqa: E402
import v62.patches  # noqa: E402,F401
import v62.retention as r  # noqa: E402
PATCHED_RF = r.install()
import v62.agent as A  # noqa: E402
import v62.session_ext as se  # noqa: E402
from inference.agent import python_tool_sandbox as sb  # noqa: E402
from inference.agent.runtime_state import Frame  # noqa: E402
from inference.framework import solver as sv  # noqa: E402
fails = []


def check(name, got, want):
    ok = got == want
    shown = got if not isinstance(got, str) else got[:60]
    print('%-70s -> %-62r %s' % (name, shown, 'OK' if ok else 'WRONG (want %r)' % (want,)))
    if not ok:
        fails.append(name)


S = A.TEXTS['system']
BOOT = sb._SANDBOX_BOOTSTRAP
_real_game_over = sv._is_engine_game_over
_real_level_number = sv._level_number


# --- scripted game session (step_env) ---------------------------------------------------------------------------------
class Session:
    """step_env's collaborators: a game whose GAME_OVER flag a RESET clears, scripted payloads for other actions."""

    def __init__(self, payloads=(), over=False, reset_dies=False):
        self.game = SimpleNamespace(over=over, current_state=SimpleNamespace(
            available_actions=[0, 1, 2, 3, 4, 5, 6], levels_completed=1,
            raw=SimpleNamespace(state=arcengine.GameState.GAME_OVER if over else arcengine.GameState.NOT_FINISHED)))
        self.history_entries, self.payloads, self.executed = [], list(payloads), []
        self.reset_dies, self.action_count = reset_dies, 7

    def _normalize_actions(self, arguments):
        return sv._HarnessGameSession._normalize_actions(self, arguments)

    def _terminal_payload(self, requested_actions):
        return sv._HarnessGameSession._terminal_payload(self, requested_actions)

    def timing_payload(self):
        return {}

    def should_stop(self):
        return False

    def _error_payload(self, message):
        return {'executed': False, 'error': message}

    def write_viewer_payload(self):
        pass

    def _set_over(self, over):
        self.game.over = over
        self.game.current_state.raw.state = arcengine.GameState.GAME_OVER if over else arcengine.GameState.NOT_FINISHED

    def _execute_action(self, action, *, batch_index, batch_size, flush_viewer_payload=True):
        disp = sv._format_action_display(action.id.name, dict(action.data))
        if action.id.name == 'RESET':
            self._set_over(self.reset_dies)
            p = dict(gameplay_changed=True, game_over=self.reset_dies)
        else:
            p = dict(self.payloads.pop(0) if self.payloads else dict(gameplay_changed=True))
        p.update(executed=True, action_display=disp, action_name=action.id.name)
        self.executed.append(disp)
        self.history_entries.append(SimpleNamespace(action=disp, result=p))
        return p


sv._level_number = lambda game: 2
sv._is_engine_game_over = lambda game: bool(getattr(game, 'over', False))
sv._engine_action_names = lambda game: ['ACTION1', 'ACTION2', 'ACTION3', 'ACTION4', 'ACTION5', 'ACTION6']


def step(session, actions, stale=None):
    return se.step_env(session, dict(actions=actions, stale_after=stale))


# === 101 (A1): a RESET in GAME_OVER executes ============================================================================
s = Session(over=True)
res = step(s, [dict(action='UP')])
check('101 GAME_OVER: a non-RESET call executes nothing', (res.get('executed'), s.executed), (False, []))
check('101 GAME_OVER: the refusal names RESET', (res.get('error'), res.get('valid_actions')), (se.GAME_OVER_REFUSAL, ['RESET']))
check('101 GAME_OVER: the refusal raises in the snippet (error, not executed)', bool(res.get('error')) and not res.get('executed'),
      True)
s = Session(over=True)
res = step(s, [dict(action='RESET'), dict(action='UP')])
check('101 GAME_OVER: RESET first executes, the batch continues', (s.executed, s.game.over), (['RESET', 'UP'], False))
s = Session(over=True, reset_dies=True)
res = step(s, [dict(action='RESET'), dict(action='UP')])
check('101 a RESET that dies again ends the batch (game_over stop)', (s.executed, res.get('stop_reason')), (['RESET'], 'game_over'))
s = Session(over=True)
res = step(s, [dict(action='UP'), dict(action='RESET')])
check('101 RESET not first: refused like any other action', (s.executed, res.get('error')), ([], se.GAME_OVER_REFUSAL))
s = Session()
s.should_stop = lambda: True
res = step(s, [dict(action='RESET')])
check('101 stopping: the terminal payload as before', (s.executed, res.get('error')),
      ([], 'No action was executed because the current game state is terminal or stopping.'))
check('101 the automatic RESET keeps its loop guard', 'and self.last_engine_action != "RESET"' in
      inspect.getsource(sv._HarnessGameSession._play_inner), True)
check('101 verdict: a RESET that ends in GAME OVER', sv._action_verdict(dict(game_over=True, action_display='RESET')),
      'GAME OVER (reported right after this RESET; no automatic restart follows a RESET)')
check('101 verdict: other game overs unchanged', sv._action_verdict(dict(game_over=True, action_display='UP', v6_attempt=3)),
      'LEVEL RESTARTED (attempt 3 ended here)')
check('101 verdict: an automatic RESET that dies is a restart verdict',
      sv._action_verdict(dict(game_over=True, action_display='RESET', automatic=True, v6_attempt=2)).startswith('LEVEL RESTARTED'),
      True)
check('101 sandbox latch no longer claims a reset', ('The level has been reset to its' in BOOT,
                                                     'the previous action reported game_over, so the board you planned against' in BOOT),
      (False, True))


class AgentSess:
    rebased_away, blocks = set(), []

    def span_evicted(self, a, b):
        return 0


class GameSess:
    def __init__(self, over):
        self.game = SimpleNamespace(over=over)
        self.v6_level_attempts = {}

    def step_env(self, arguments):
        return {}

    def timing_payload(self):
        return {'time_remaining_seconds': 3600}


def agent_with(game_session):
    a = A.V6Agent(session=AgentSess(), logger=lambda k, p: None, count_tokens=lambda t: max(1, len(t) // 4), model='M',
                  base_url='http://x/v1', provider='vllm')
    a._v6_started = True
    a._step_env_callback = game_session.step_env
    return a


board = [[0] * 64 for _ in range(64)]
f = Frame(grid=board, step=40, level=3)
a = agent_with(GameSess(over=True))
summ = dict(executed_actions=['RESET'], executed_count=1, level=3, game_over=True, fatal_action='RESET',
            v61_actions=[dict(action_display='RESET', game_over=True)])
text, kind = a._build_user_prompt(40, valid_actions=['ACTION1', 'ACTION2'], current_frame=f, history_entries=[],
                                  previous_step_summary=summ)
check('101 opener after a RESET that died: the no-restart line', A.TEXTS['game_over_no_restart'] in text, True)
check('101 opener: no restart block, kind same', ('has restarted from its initial state' in text, kind), (False, 'same'))
check('101 opener: valid actions say RESET only', 'Valid actions: RESET only (the game reports GAME OVER).' in text, True)
check('101 opener: the trace shows the RESET verdict', 'RESET -> GAME OVER (reported right after this RESET' in text, True)
text, kind = a._build_user_prompt(40, valid_actions=['ACTION1'], current_frame=f, history_entries=[], previous_step_summary=None)
check('101 opener after a refused call in GAME_OVER', (A.TEXTS['game_over_no_restart'] in text, 'You are still on level' in text),
      (True, False))
a = agent_with(GameSess(over=False))
text, kind = a._build_user_prompt(40, valid_actions=['ACTION1'], current_frame=f, history_entries=[], previous_step_summary=None)
check('101 opener outside GAME_OVER: no RESET-only line; v6.3 G: no same-level line either', ('You are still on level 3.' in text, 'RESET only' in text, 'level 3, attempt' in text), (False, False, True))

# === 102 (A3 option b + the band sums) ==================================================================================
V = sv._action_verdict
check('102 NO-OP: nothing on the board changed', V(dict(gameplay_changed=False, board_changed=False)),
      'NO-OP (nothing on the board changed)')
check('102 BAND-ONLY CHANGE with the band count', V(dict(gameplay_changed=False, board_changed=True, v61_edge=dict(n=3))),
      'BAND-ONLY CHANGE (no change in the interior; 3 cells of the outer band changed)')
check('102 BAND-ONLY CHANGE without an edge record (board_changed only)', V(dict(gameplay_changed=False, board_changed=True)),
      'BAND-ONLY CHANGE (no change in the interior; the outer band changed)')
check('102 REVERTED carries the band fact', V(dict(gameplay_changed=False, reverted=True, v61_edge=dict(n=1))),
      'REVERTED (animated, then the interior returned to its previous state; 1 cell of the outer band changed)')
check('102 REVERTED without band change unchanged', V(dict(gameplay_changed=False, reverted=True)),
      'REVERTED (animated, then the interior returned to its previous state)')
BAND = dict(gameplay_changed=False, board_changed=True, v61_edge=dict(n=2))
NOOP = dict(gameplay_changed=False, board_changed=False)
s = Session([dict(gameplay_changed=True, board_changed=True), BAND])
res = step(s, [dict(action='UP'), dict(action='DOWN'), dict(action='LEFT')])
check('102 batch stop at a BAND-ONLY CHANGE (interior basis kept)', (s.executed, res.get('stop_reason')),
      (['UP', 'DOWN'], 'no_op_action'))
check('102 stop detail names the label', 'EXECUTED but changed nothing in the interior (BAND-ONLY CHANGE: 2 cells of the '
      'outer band changed).' in res.get('stop_detail', ''), True)
check('102 stop detail: one skipped action, singular', "Action 3 ('LEFT') was SKIPPED and did not execute." in
      res.get('stop_detail', ''), True)
check('102 stop detail: actions before it', 'Only the first 1 executed action changed anything in the interior.' in
      res.get('stop_detail', ''), True)
check('102 last_board_changed: the band changed', res.get('last_board_changed'), True)
s = Session([NOOP])
res = step(s, [dict(action='UP'), dict(action='DOWN')])
check('102 NO-OP stop text and last_board_changed', ('(NO-OP: nothing on the board changed)' in res.get('stop_detail', ''),
      'No executed action of this batch changed anything in the interior.' in res.get('stop_detail', ''),
      res.get('last_board_changed')), (True, True, False))
check('102 per-call records carry board_changed', 'board_changed' in (s.__dict__.get('v61_call_payloads') or [{}])[0], True)
check('102 sandbox arms the stale flag only when nothing on the board changed',
      'and action_result.get("last_board_changed") is False' in BOOT, True)
check('102 stale detail wording', sv._stale_detail(SimpleNamespace(id=SimpleNamespace(name='ACTION1'), data={}), 'UP').count(
      'was a NO-OP - nothing on the board changed -'), 1)
check('102 stale detail after REVERTED', 'was REVERTED - it animated, then the board returned to its previous state -' in
      sv._stale_detail(SimpleNamespace(id=SimpleNamespace(name='ACTION1'), data={}), 'UP', reverted=True), True)
check('102 refusal reason text', A.REFUSAL_REASONS['stale_state'], 'the previous action changed nothing on the board '
      '(stale-state refusal)')
check('102 lengths text', A.V6Agent._v61_lengths_text([(23, 5), (1, 1)]),
      '23 actions (5 changed nothing in the interior), 1 action (1 changed nothing in the interior)')
e = A._edge_text([dict(v61_edge=dict(n=10, tr=[['a', 'b', 4], ['c', 'd', 2], ['e', 'f', 1]], rows=[0], cols=[1, 2]),
                       gameplay_changed=False, board_changed=True)])
check('102 band sums: remainder = total - top 2 (10 - 6)', 'a->b x4, c->d x2, other colour changes x4' in e, True)
check('102 band line names BAND-ONLY CHANGE', e.endswith('1 of them changed nothing in the interior (BAND-ONLY CHANGE)'), True)
for name, frag in (('102 prompt: band-only sentence', 'their verdict is BAND-ONLY CHANGE, not NO-OP.'),
                   ('102 prompt: verdict list', 'NO-OP (executed, but nothing on the board changed), BAND-ONLY CHANGE '
                                            '(executed; only cells of the outer band changed), REVERTED'),
                   ('102 prompt: batch stop', 'changes nothing in the interior (a NO-OP or a BAND-ONLY CHANGE)'),
                   ('102 prompt: stale refusal after a NO-OP', 'after a NO-OP (nothing on the board changed), the next '
                                                              'action(...) call')):
    check(name, frag in S, True)
check('102 prompt: the old NO-OP test sentence is gone', 'the NO-OP test ignores those cells' in S, False)
check('102 noop_stop opener', A.TEXTS['noop_stop'].startswith('Your batch was stopped: {action}'), True)

# === 103-106 (B1 run clock, B2 UNDO, B3 ledger header, B7 RESET-first text) ============================================
check('103 duration: hours and minutes', [A._duration_text(x) for x in (42000, 3600, 3569, 61, 20)],
      ['11 h 40 min', '1 h 0 min', '59 min', '1 min', 'less than 1 min'])
check('103 run clock line', A.V6Agent._v61_clock_line(None, GameSess(over=False)),
      'Run time left: 1 h 0 min (the whole run, not an attempt timer).')
check('104 UNDO conditioned', '- UNDO, when it is among the valid actions, reverses a previous action' in S, True)
check('104 RESET bullet no longer prefers UNDO', 'If UNDO can restore a winnable state, prefer it.' in S, False)
s = Session()
s.game.current_state.available_actions = [0, 1, 2, 3, 4]
res = step(s, [dict(action='UNDO')])
check('104 an action not offered: the refusal', res.get('error', '').startswith(
      'UNDO is not among the valid actions in this observation; nothing was executed. Valid actions now: '), True)


class Ledger(A.V6Agent):
    pass


lg = object.__new__(Ledger)
base = ['Known fatal continuations FROM THE CURRENT BOARD STATE (this level has 3 failed attempt(s) recorded; the game is '
        'deterministic):', '- 2 action(s): UP, LEFT -> GAME OVER (attempt(s) 4, 7)',
        A.V6Agent._LEDGER_ADVICE + 'The full ledger is available in python as `death_ledger`.']
orig = A.ta.ToolAgent._death_ledger_lines
A.ta.ToolAgent._death_ledger_lines = lambda self, *a: list(base)
out = Ledger._death_ledger_lines(lg, None, [], None)
A.ta.ToolAgent._death_ledger_lines = orig
check('105 ledger header: the observation and its basis', out[0], 'Continuations that ended in GAME OVER from a board '
      'identical to the current one (whole board compared; 3 game overs recorded on this level):')
check('105 ledger ids', out[1], '- 2 action(s): UP, LEFT -> GAME OVER (game over #4, #7 of this game)')
check('105 ledger advice: whole board, no "interior"', (out[2].startswith('Each listed continuation ended in GAME OVER when '
      'played from a board identical to this one;'), 'interior' in out[2]), (True, False))
check('105 no "deterministic" anywhere in the ledger lines', any('deterministic' in x for x in out), False)
check('106 RESET-first texts in the sandbox', ('RESET_FIRST_TEXT = ' in BOOT, 'raise RuntimeError(RESET_FIRST_TEXT[1] % _v62_k '
      'if _v62_k else RESET_FIRST_TEXT[0])' in BOOT), (True, True))
check('106 RESET-first text after executed actions', __import__('v62.patches', fromlist=['x']).RESET_FIRST_TEXT[1] % 3,
      'RESET must be the first game action of a Python call. This action() call executed nothing; the 3 action(s) executed '
      'earlier in this call stand. To restart the level, end this call and put RESET first in the next one.')
check('106 refused-call footnote for RESET', A._refusal_text([dict(stop_reason='reset_not_first', requested_actions=['RESET'],
      error='RESET must be the first executed game action in a Python snippet. This call executed no actions.')]),
      '1 action() call in this Python call was refused and executed nothing: RESET was not the first executed action of the '
      'snippet (1). The first was RESET: RESET must be the first game action of a Python call; that action() call executed '
      'nothing.')

# === 107 (A7: small accuracy items) =====================================================================================
a = agent_with(GameSess(over=False))
text, cut = a._trim_tool_text('x' * 20000)
check('107 one over-long line is cut, not dropped', (cut, text.split('\n')[0].endswith(' [... line cut]'),
                                                     len(text.split('\n')[0]) > 1000), (True, True, True))
pl = {}
A.V6Agent._record_retained_functions(a, dict(action_results=[dict(executed=False, stop_reason='invalid_action',
                                                                  requested_actions=['UNDO'], error='UNDO is not among ...')]), pl)
check('107 an invalid-action refusal is tallied', pl.get('refused_action_calls', '').startswith(
      '1 action() call in this Python call was refused and executed nothing: an action not among the valid actions (1).'), True)
check('107 call cap and game over named', (A.REFUSAL_REASONS['call_cap'], A.REFUSAL_REASONS['game_over']),
      ('the limit of 40 executed actions per Python call', 'the game reported GAME OVER (only RESET executes then)'))
gs = GameSess(over=False)
gs.__dict__['v61_call_payloads'] = [dict(action_display='UP', gameplay_changed=True, board_changed=True),
                                    dict(action_display='LEFT', game_over=True, board_changed=True)]
a = agent_with(gs)
summ = A.V6Agent._summarize_step_sequence(a, [dict(executed=True, executed_count=1, executed_actions=['UP'], action_num=5,
                                                    level=2, gameplay_changed=True)])
check('107 ComputeLimitError: the host records complete the list', (summ['executed_actions'], summ['executed_count'],
      summ['game_over'], summ['fatal_action']), (['UP', 'LEFT'], 2, True, 'LEFT'))
summ = A.V6Agent._summarize_step_sequence(a, [dict(executed=False, error='ComputeLimitError ...')])
check('107 nothing in the snippet list: a summary from the host records', (summ or {}).get('executed_actions'), ['UP', 'LEFT'])
check('107 malformed-call note says how much it recovered', 'the harness recovered 3 lines of code from the message and ran '
      'it.' in A._fill(A.TEXTS['markup_recovered'], lines=A._lines_text(3)), True)
check('107 takeover: "level 1 completed"', A._fill(A.TEXTS['takeover'], level=2, done='level 1'),
      'The game is at level 2 (level 1 completed).')
check('107 takeover computes the completed levels', "done=('level 1' if int(level) == 2 else 'levels 1-%d'" in
      inspect.getsource(A.V6Agent._v61_takeover_text), True)
line = A.V6Agent._v61_stop_line(a, dict(kind='reverted', executed=['UP'], skipped=[], pair=['DOWN', 'UP']), None, 2)
check('107 two REVERTED, the second ended its list', line, 'Two REVERTED actions in a row in this Python call (DOWN, then UP): '
      'each animated and then left the interior as it was; the second was the last action of its action() list, so nothing '
      'was skipped.')
src = inspect.getsource(A.V6Agent._v62_analyze)
check('107 g514 kept as designed: a cue holds the yield for its own reply (round 5i, user)',
      "held = (sorted(set(getattr(result, 'cues', ()) or ()) & YIELD_HELD_BY)" in src, True)   # v6.3: + the no-action warning

# === 108 (S1: save_summary; A2 fallback fixes; B5) ======================================================================
a = agent_with(GameSess(over=False))
a._v6_stats['completions'] = 3
check('108 nothing requested: nothing saved', a._v62_save_summary('Rule L2:\n- the red block moves when pushed'),
      dict(saved=False, note=A.TEXTS['summary_not_requested']))
a._v62_fn_window = (2, 5)
a._v6_rule_capture = 2
r1 = a._v62_save_summary('Rule L2:\n- the red block moves when pushed\n- walls stop it')
check('108 saved: tag stripped, stored for the requested level', (r1['saved'], a._v6_rules.get(2), sorted(a._v62_rule_fn),
      a._v6_rule_capture), (True, '- the red block moves when pushed\n- walls stop it', [2], None))
check('108 confirmation text', r1['note'].startswith('Saved your summary of level 2 (') and r1['note'].endswith(
      'tokens). It is shown to you again if this game restarts a level.'), True)
r2 = a._v62_save_summary('Rule L2:\n- the red block moves one cell per push\n- walls stop it')
check('108 the latest call wins and says so', (a._v6_rules[2].startswith('- the red block moves one cell'),
      r2['note'].endswith(A.TEXTS['summary_replaced'])), (True, True))
check('108 counters', (a._v6_stats.get('rules_via_function'), a._v6_stats.get('rules_recorded')), (1, 1))
check('108 the call is marked (no-action / yield exemption)', a._v62_saved_in_call, True)
check('108 too short', a._v62_save_summary('Rule L2: ok')['note'], A.TEXTS['summary_too_short'])
a._v6_stats['completions'] = 6
check('108 the window closes after RULE_TURNS completions', a._v62_save_summary('Rule L2:\n- a long enough summary text '
      'here')['saved'], False)
a._v6_rule_capture = 2
A.V6Agent._v6_capture_rule(a, ['Rule L2: a different summary written as prose in the reply.'])
check('108 after a save the regex capture does not override it', (a._v6_rules[2].startswith('- the red block moves one'),
      a._v6_rule_capture), (True, None))
check('108 sandbox function and host dispatch', ('def save_summary(text):' in BOOT, 'runtime_globals["save_summary"] = '
      'save_summary' in BOOT, "agent._v62_save_summary(request.get('text'))" in inspect.getsource(
      __import__('v62.sandbox_ext', fromlist=['x']))), (True, True, True))
check('108 level-up request names the function, same call', "In your next python call, together with that call's other code, "
      'save a summary with save_summary(text): a text that begins with the line "Rule L{level-1}:"' in A.TEXTS['levelup'], True)
check('108 system prompt documents it', '- `save_summary(text)` saves a summary when the harness asks for one' in S, True)
src = inspect.getsource(A.V6Agent._v62_analyze)
check('108 a save-only call is neither idle nor a yield trigger', ('elif self._v62_saved_in_call:' in src,
      'self._turn_generated_tokens >= YIELD_TOKENS and not saved:' in src), (True, True))


class Cap:
    pass


def cap(texts, n=3, turns=2):
    d = Cap(); d._v6_rule_capture = n; d._v62_rule_turns = turns; d._v6_rules = {}; d._v6_stats = {}
    d._v6_rules_dirty = False
    A.V6Agent._v6_capture_rule(d, texts)
    return d._v6_rules.get(n), d._v6_rule_capture, d._v62_rule_turns


check('108 fallback: g531, a header inside the list body is not a second summary',
      cap(['Rule L3 summary:\n1. the red block is pushed by the player\n2. Rule L3 detail: walls stop the block from moving\n'
           '3. the goal is the green pad'])[0],
      '1. the red block is pushed by the player\n2. Rule L3 detail: walls stop the block from moving\n3. the goal is the green pad')
check('108 fallback: one "- Rule Lk:" line per level (387b, 390)',
      cap(['- Rule L2: the blue key opens the blue door\n- Rule L3: the red block is pushed by the player'])[0],
      'the red block is pushed by the player')
check('108 fallback: a header written as the first list item stays (387b, 390)',
      cap(['Summary:\n- Rule L3:\n- the red block is pushed by the player\n- walls stop the block'])[0],
      '- the red block is pushed by the player\n- walls stop the block')
check('108 fallback: a quoted header is not one', cap(['I will write "Rule L3: the red block is pushed by the player" '
                                                       'later'])[0], None)
check('108 fallback: the kx59 header (quote after "Rule") still captured', cap(['Rule summary (for the required "Rule L3:" '
      'lines):\n- the red block is pushed by the player\n- walls stop the block'])[0],
      '- the red block is pushed by the player\n- walls stop the block')
check('108 fallback: prose cut only where a sentence names the next level', cap(['Rule L3: the block is pushed as in level 4 '
      'rules (seen earlier), walls stop it. Level 4 has a new key.'])[0],
      'the block is pushed as in level 4 rules (seen earlier), walls stop it.')
check('108 fallback: a reasoning-only summary is provisional', cap(['', '', 'Rule L3: the draft from the reasoning text '
      'here.']), ('the draft from the reasoning text here.', 3, 1))
check('108 fallback: a line contained in another is dropped', cap(['Rule L3: walls stop the red block when pushed.\n\n'
      'Rule L3: walls stop the red block when pushed from the left.'])[0],
      'walls stop the red block when pushed from the left.')   # one line left: kept as written
check('108 fallback: repeated list lines dropped', cap(['Rule L3:\n- walls stop the moving block\n- the key opens the doors\n'
      '- walls stop the moving block'])[0], '- walls stop the moving block\n- the key opens the doors')

# === 109 (S3: attempt record; A4 phantom attempts, A5 silent return, B4 restart facts, ledger kept) ======================
import v62.patches as P  # noqa: E402
_real_base = P._base_execute


class Rec:
    def __init__(self):
        self.history_entries = []


def run_actions(seq):
    """seq of payload dicts through the patches wrapper (attempt bookkeeping only)"""
    g = Rec()
    it = iter(seq)
    P._base_execute = lambda self, action, **kw: dict(next(it))
    out = [P._execute_action(g, SimpleNamespace(id=SimpleNamespace(name='X'))) for _ in seq]
    P._base_execute = _real_base
    return g, out


g, out = run_actions([dict(action_name='UP', level=3, board_changed=True, gameplay_changed=True),
                      dict(action_name='RESET', level=3, board_changed=True),
                      dict(action_name='RESET', level=3, board_changed=False),
                      dict(action_name='UP', level=3, board_changed=True, gameplay_changed=True),
                      dict(action_name='DOWN', level=3, game_over=True, board_changed=True)])
check('109 a RESET that changed nothing opens no attempt', (g.v6_level_attempts[3], g.v62_level_attempt_ends[3]),
      ([2, 3], ['reset', 'game_over']))
check('109 the no-op RESET is not an attempt boundary', 'v6_attempt' in out[2], False)
g, out = run_actions([dict(action_name='UP', level=2, board_changed=True, gameplay_changed=True, start_return=True)])
check('109 a silent return is an event, not a boundary', (g.__dict__.get('v62_level_start_returns'),
      g.__dict__.get('v6_level_attempts')), ({2: 1}, None))
g = Rec()
g.v61_harness_reset = True
g.v6_attempt_actions = 2
P._base_execute = lambda self, action, **kw: dict(action_name='RESET', level=4, board_changed=False)
P._execute_action(g, SimpleNamespace(id=SimpleNamespace(name='RESET')))
P._base_execute = _real_base
check('109 the harness restart ends an attempt that had actions', g.v62_level_attempt_ends, {4: ['restart']})
s = Session([dict(gameplay_changed=True, board_changed=True, start_return=True, level_start_match='full')])
res = step(s, [dict(action='UP'), dict(action='LEFT'), dict(action='DOWN')])
check('109 A5: the batch stops at the return', (s.executed, res.get('stop_reason')), (['UP'], 'level_start'))
check('109 A5: stop detail', res.get('stop_detail', '').startswith("The whole board, outer band included, returned to this "
      "level's starting board after UP, with no GAME OVER reported and no RESET; the interior had differed from it in more "
      "than 64 cells. The remaining 2 action(s) were not executed: LEFT, DOWN."), True)
d, start = {}, [[0] * 64 for _ in range(64)]
far = [[1 if 10 <= r < 30 and 10 <= c < 30 else 0 for c in range(64)] for r in range(64)]
se._start_facts(d, {}, False, False, False, far, start)
check('109 A5: a far excursion arms', d.get('v62_far_armed'), True)
se._start_facts(d, dict(level_start_match='full'), False, False, False, start, start)
check('109 A5: the return clears it', d.get('v62_far_armed'), False)
check('109 A5: execute_action marks the return only when armed', "if new_grid == start and self.__dict__.get('v62_far_armed'):"
      in inspect.getsource(se.execute_action), True)
gs = GameSess(over=False)
gs.__dict__['v62_level_attempt_ends'] = {3: ['game_over', 'game_over', 'reset', 'restart']}
gs.__dict__['v62_level_start_returns'] = {3: 2}
a = agent_with(gs)
a._v62_restart_archived = (3, 2)
check('109 B4: restart facts', a._v62_restart_facts(gs, 3), 'Before this conversation, this level had 4 attempts: 2 ended in '
      'GAME OVER, 1 ended with your RESET and 1 ended at a harness restart. The whole board returned to the level\'s starting '
      'board without a RESET 2 times. 2 programs written on this level were set aside at the restart and are not loaded.')
src = inspect.getsource(A.V6Agent._v61_restart)
check('109 B4: the ledger is kept across a restart', ("'_death_ledger_attempts'" in src, "'_death_ledger_index'" in src),
      (False, False))
src = inspect.getsource(A.V6Agent._v61_takeover_text)
check('109 B4: the takeover shows the facts and the ledger', ('self._v62_restart_facts(session, level)' in src,
      'self._death_ledger_lines(current_frame, history_entries, None)' in src), (True, True))
check('109 takeover texts', (A.TEXTS['takeover_restart'].endswith('your earlier attempts on this level are not in this '
      'conversation or in `history`.'), 'The step counter counts actions in the whole run' in A.TEXTS['takeover_history']),
      (True, True))
check('109 A4: untried actions after a restart are per conversation', "'Actions not yet executed on this level in this "
      "conversation:'" in inspect.getsource(A.V6Agent._v6_resumption_text), True)
check('109 A5 in the system prompt', "and when the whole board returns to the level's starting board without a RESET after "
      "its interior had differed from it in more than 64 cells." in S, True)

# === 110 (S4: versioned library; A6) ====================================================================================
import builtins  # noqa: E402
RF = {}
exec(compile(PATCHED_RF, '<retained_functions v61>', 'exec'), RF)
AVAILABLE = set(dir(builtins)) | {'action', 'current_frame', 'history'}
PREV = {'cc': 'def cc(x):\n    return x + 1', 'dd': 'def dd(x):\n    return cc(x) * 2'}


def collect(code):
    ns = {'__builtins__': builtins}
    for src in PREV.values():
        exec(compile(src, '<retained>', 'exec'), ns)
    exec(compile(code, '<python_tool>', 'exec'), ns)
    kept, rejected = RF['collect_functions'](code, dict(PREV), ns, AVAILABLE)
    return kept, {x['name']: x['reason'] for x in rejected}


kept, rej = collect('y = 5\ndef cc(x):\n    return x + y')
check('110 a redefinition failing validation keeps the earlier version', (kept.get('cc'), rej.get('cc', '')[:22], 'dd' in kept),
      (PREV['cc'], '__v62_kept_previous__:', True))
check('110 its reason is kept for the note', 'unavailable dependencies: y' in rej.get('cc', ''), True)
kept, rej = collect('def cc(x):\n    return 5\ncc = 3')
check('110 a redefinition then rebound keeps the earlier version', (kept.get('cc'), rej.get('cc', '').startswith(
      '__v62_kept_previous__: the new definition did not execute')), (PREV['cc'], True))
kept, rej = collect('@staticmethod\ndef cc(x):\n    return 7')
check('110 a decorated redefinition keeps the earlier version', (kept.get('cc'), rej.get('cc', '').startswith(
      '__v62_kept_previous__:')), (PREV['cc'], True))
kept, rej = collect('def cc(x):\n    return x * 3')
check('110 a valid redefinition still replaces it', (kept.get('cc'), 'cc' in rej), ('def cc(x):\n    return x * 3', False))
kept, rej = collect('def ee(x):\n    return x + Z')
check('110 a new program failing validation is still rejected as before', ('ee' in kept, rej.get('ee', '').startswith(
      'unavailable dependencies')), (False, True))
check('110 note text', A._v6_retention_notes('Your function cc was not retained: __v62_kept_previous__: unavailable '
      'dependencies: Y. Pass snippet-specific data as arguments or redefine it if needed.'),
      'The new definition of `cc` was not saved (unavailable dependencies: Y); the earlier version of `cc` is kept in the '
      'program library.')
a = agent_with(GameSess(over=False))
a._v6_library_order, a._v61_library_levels, a._v6_library_steps = ['show', 'once'], {'show': 3, 'once': 3}, {'show': 50, 'once': 51}
a._kept_functions = {'show': 'def show():\n    return 3', 'once': 'def once():\n    return 0'}
a._v62_library_log = [dict(name='show', level=1, step=5, source='def show():\n    return 1', status='saved'),
                      dict(name='show', level=2, step=20, source='def show():\n    return 2', status='saved'),
                      dict(name='show', level=3, step=50, source='def show():\n    return 3', status='saved'),
                      dict(name='once', level=3, step=51, source='def once():\n    return 0', status='saved')]
archived, restored = a._v62_archive_level_programs(3)
check('110 restart restores the latest earlier-level version from the log', (restored, archived, a._kept_functions.get('show'),
      a._v61_library_levels.get('show')), (['show'], ['once'], 'def show():\n    return 2', 2))
check('110 the log records the restore and the archive', [e['status'] for e in a._v62_library_log[-2:]], ['restored', 'archived'])

# === 112 (C1 board shift, C3 UNDO restored, C4 MOUSE display form, C5 same report, B6 types and "did you mean") ==========
import ast  # noqa: E402
import random  # noqa: E402
rnd = random.Random(5)
g0 = [[rnd.randrange(10) for _ in range(64)] for _ in range(64)]
g1 = [[(g0[r - 2][c - 3] if r >= 2 and c >= 3 else 0) for c in range(64)] for r in range(64)]
check('112 C1 a shift of the content (down 2, right 3)', {k: v for k, v in (se.board_shift(g0, g1) or {}).items()
                                                         if k in ('dy', 'dx')}, dict(dy=2, dx=3))
g2 = [row[:] for row in g0]
for r in range(10, 14):
    for c in range(10, 14):
        g2[r][c] = (g0[r][c] + 1) % 10
check('112 C1 a local change: no shift fact', se.board_shift(g0, g2), None)
check('112 C1 verdict', sv._action_verdict(dict(gameplay_changed=True, board_shift=dict(dy=2, dx=-1, explained=3900,
      changed=4000))).endswith('| a shift of the board content by 2 rows and -1 col (down and right positive) explains 3900 '
      'of the 4000 changed cells'), True)
E = lambda grid, step, action, **res: SimpleNamespace(frame=SimpleNamespace(grid=grid, step=step), action=action, result=res)
b0, b1, b2 = [[0]], [[1]], [[2]]
rows = [E(b0, 10, 'RESET', action_name='RESET'), E(b1, 11, 'UP', action_name='UP'), E(b2, 12, 'LEFT', action_name='LEFT')]
check('112 C3 UNDO back to an earlier board of the attempt', se.undo_restored(rows, b1), dict(step=12, action='LEFT'))
check('112 C3 not across the attempt start', se.undo_restored([E(b1, 9, 'UP'), E(b0, 10, 'RESET', action_name='RESET'),
                                                               E(b2, 11, 'LEFT')], b1), None)
check('112 C3 verdict', sv._action_verdict(dict(gameplay_changed=True, undo_restored=dict(step=12, action='LEFT'))).endswith(
      '| the board now equals the board before LEFT at step 12'), True)
tree = ast.parse(BOOT)
fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == '_normalize_actions')
NS = {}
exec(compile(ast.Module(body=[fn], type_ignores=[]), '<bootstrap>', 'exec'), NS)
check('112 C4 the history display form of a click', NS['_normalize_actions'](['MOUSE(row=5, col=7)', 'UP']),
      [dict(action='MOUSE', row=5, col=7), dict(action='UP')])
t1 = 'Your previous call executed 2 action(s): RESET, UP.\nPer action: 1: RESET -> effect.\nStep 40 (actions in the whole run), ' \
     'level 3, attempt 2. Valid actions: UP.\nRun time left: 2 h 0 min (the whole run, not an attempt timer).'
t2 = t1.replace('Step 40', 'Step 42').replace('2 h 0 min', '1 h 59 min')
check('112 C5 same report: same key apart from step and clock', A._opener_key(t1) == A._opener_key(t2), True)
check('112 C5 collapsed text keeps the step and clock lines', A._collapse_opener(t2).split('\n'),
      [A.TEXTS['same_report'], 'Step 42 (actions in the whole run), level 3, attempt 2. Valid actions: UP.',
       'Run time left: 1 h 59 min (the whole run, not an attempt timer).'])
check('112 C5 wired for same-kind openers', 'text = _collapse_opener(text)' in inspect.getsource(A.V6Agent._v62_analyze), True)
check('112 B6 did you mean (provided)', A._v6_error_text("NameError: name 'last_animation_timelines' is not defined").endswith(
      'Did you mean `last_animation_timeline`?'), True)
check('112 B6 did you mean (library)', A._v6_error_text("NameError: name 'find_blok' is not defined", ('find_block',)).endswith(
      'Did you mean `find_block` (in your program library)?'), True)
check('112 B6 otherwise the persistence note', A._v6_error_text("NameError: name 'xyz' is not defined").endswith(
      A.TEXTS['nameerror_note']), True)
check('112 B6 timeline types in the prompt', '`last_animation_timeline` is a dict: `action` (a string), `frames` (the number '
      'of frames, an int) and `steps` (a list,' in S, True)
check('112 C1/C3 documented', ('`board_shift`' in S, '`undo_restored`' in S), (True, True))

# === 113 (S5 template boundary values; class 10 documented facts) =======================================================
import re  # noqa: E402
NOUNS = r'(?:actions|cells|programs|attempts|tokens|lines|times|conversations|game overs|items|calls|rows|cols|frames)'
BAD = [re.compile(r'(?<![\d.,])1 %s\b' % NOUNS), re.compile(r'\b(?:[Ll]evels|[Aa]ctions|rows|cols) (\d+)-\1\b')]


def lint(text):
    return [m.group(0) for rx in BAD for m in rx.finditer(text or '')]


renders = []
for n, m in ((0, 0), (1, 1), (2, 1), (1, 0)):
    renders.append(A.V6Agent._v61_lengths_text([(n, m)]))
renders.append(A._edge_text([dict(v61_edge=dict(n=1, tr=[['a', 'b', 1]], rows=[0], cols=[5]), gameplay_changed=False,
                                  board_changed=True)]))
renders.append(A._refusal_text([dict(stop_reason='stale_state', requested_actions=['UP'])]))
renders.append(A._refusal_text([dict(stop_reason='stale_state'), dict(stop_reason='invalid_action')]))
gs = GameSess(over=False)
gs.__dict__['v62_level_attempt_ends'] = {2: ['game_over']}
gs.__dict__['v62_level_start_returns'] = {2: 1}
a = agent_with(gs)
a._v62_restart_archived = (2, 1)
renders.append(a._v62_restart_facts(gs, 2))
for skipped in ([], ['LEFT'], ['LEFT', 'DOWN']):
    renders.append(A.V6Agent._v61_stop_line(a, dict(kind='reverted', executed=['UP'], skipped=skipped, pair=['DOWN', 'UP']), None, 2))
    renders.append(A.V6Agent._v61_stop_line(a, dict(kind='level_start', executed=['UP'], skipped=skipped), None, 2))
    renders.append(A.V6Agent._v61_stop_line(a, dict(kind='call_cap', executed=['UP'], skipped=skipped), None, 2))
for scripted, acts in (([BAND], 2), ([dict(gameplay_changed=True, board_changed=True), NOOP], 3),
                       ([dict(gameplay_changed=True, board_changed=True)] * 2 + [NOOP], 4)):
    s = Session(list(scripted))
    renders.append(step(s, [dict(action='UP')] * acts).get('stop_detail', ''))
renders += [A._duration_text(x) for x in (59, 61, 3600, 3660)] + [A._lines_text(1)]
a._v6_library_order, a._kept_functions, a._v61_library_levels = ['f'], {'f': 'def f():\n    return 1'}, {'f': 2}
renders.append(a._v61_library_count_line())
renders.append(V(dict(gameplay_changed=False, board_changed=True, v61_edge=dict(n=1))))
renders.append(V(dict(gameplay_changed=True, interior_changed=dict(n=1, rows=[5, 5], cols=[6, 6]), band_connected=1)))
renders.append(A._fill(A.TEXTS['takeover'], level=2, done='level 1'))
a._v62_attempt_notes = {2: ['- note one about the level', '- note two about the level', '- note three']}
renders.append('\n'.join(a._v62_notes_lines(2)))
check('113 the lint flags what it should', lint('has 1 programs; levels 3-3; 1 action; 11 cells'), ['1 programs', 'levels 3-3'])
found = [(i, x) for i, r_ in enumerate(renders) for x in lint(r_)]
check('113 no "1 <plural>" and no one-element ranges in %d rendered texts' % len(renders), found, [])
check('113 the library line, one program', renders[renders.index(a._v61_library_count_line())].startswith(
      'Your program library has 1 program, written on level 2; `programs()` lists it'), True)
DOCUMENTED = ('level_start_match', 'edge_cells_restored', 'repeat_in_call', 'interior_changed', 'band_connected',
              'actions_since_start', 'near_level_start', 'start_return', 'board_shift', 'undo_restored')
INTERNAL = ('reverted', 'interior_transient', 'v61_edge', 'v61_hit')
check('113 every documented fact key is in the results', [k for k in DOCUMENTED if k not in se.EXTRA_RESULT_KEYS], [])
check('113 every result key is documented or internal', [k for k in se.EXTRA_RESULT_KEYS if k not in DOCUMENTED + INTERNAL], [])
check('113 every documented key is named in the prompt', [k for k in DOCUMENTED if '`%s`' % k not in S], [])
check('113 the compact result carries the documented keys', [k for k in DOCUMENTED if "'%s'" % k not in
                                                           inspect.getsource(A.V6Agent._compact_action_result)], [])

# === user 10-08 ~01:55: no deferral (D2), the 4K yield not held by model-voice cues, rows 12-14 reworded ================
src = inspect.getsource(A.V6Agent._v62_analyze)
check('D2 (user): no attempt deferral in the trigger', ('RESTART_DEFER_TOKENS' in src, 'restart-deferred' in src), (False, False))
check('yield (user): model-voice cues hold nothing by default', A.YIELD_HELD_BY, frozenset())
check('yield: the breaker note still holds it', "+ (['breaker_note'] if breaker else [])" in src, True)
check('row 12: no step bar assumed', ('including any step or time counter' in S, 'including the step or time bar' in S),
      (True, False))
check('row 13: UNDO gives back anything spent', 'whether it gives back anything spent depends on the game.' in S, True)
check('row 14: the game-over exception', '(except after a RESET that ended in GAME OVER; the turn\'s message then says so)' in S,
      True)

# === Fable's code review (2026-10-08): E1-E3, T1-T5, the late gate, notes from any completion, two summary windows =====
a = agent_with(GameSess(over=False))
B0 = [[0] * 64 for _ in range(64)]
B1 = [row[:] for row in B0]; B1[30][30] = 3
H = lambda act, grid, lvl=2, **res: SimpleNamespace(action=act, frame=Frame(grid=grid, step=0, level=lvl), result=res)
a._death_ledger_record([H('RESET', B0, automatic=True, action_name='RESET'), H('UP', B1, action_name='UP'),
                        H('RESET', B1, game_over=True, action_name='RESET')])
check('E1 a model RESET that died records no fatal continuation', (a._death_ledger_attempts, a._v6_stats.get(
      'ledger_reset_deaths_skipped')), ([], 1))
a._death_ledger_record([H('RESET', B0, action_name='RESET'), H('UP', B1, game_over=True, action_name='UP'),
                        H('RESET', B0, automatic=True, action_name='RESET')])
check('E1 a normal death is still recorded', len(a._death_ledger_attempts), 1)
src_r, src_a = inspect.getsource(A.V6Agent._v61_restart), inspect.getsource(A.V6Agent._v62_analyze)
check('E2 the notes request is cleared at a restart and a level change', ('self._v62_attempt_request, self._v62_attempt_saved = '
      'None, False' in src_r, 'self._v62_attempt_request = None                   # round 5l review' in src_a), (True, True))
g = Rec()
g.v61_harness_reset = True
P._base_execute = lambda self, action, **kw: dict(action_name='RESET', level=4, board_changed=False)
P._execute_action(g, SimpleNamespace(id=SimpleNamespace(name='RESET')))
check('E3 a harness restart on an attempt without actions records nothing', (g.__dict__.get('v62_level_attempt_ends'),
      g.__dict__.get('v6_attempt_actions')), (None, 1))
g.v6_attempt_actions = 3
P._execute_action(g, SimpleNamespace(id=SimpleNamespace(name='RESET')))
P._base_execute = _real_base
check('E3 with actions it ends the attempt', g.v62_level_attempt_ends, {4: ['restart']})
check('late gate: after the notes turn the restart follows', "if self._v62_summary_state == 'done' and (budget_hit or noact_hit)" in src_a, True)   # v6.3 B
a = agent_with(GameSess(over=False))
a._v62_attempt_request, a._v62_attempt_saved = 3, False
a._v62_end_summary_turn([['Notes L3:\n- the red key opens the blue door\n- the timer refills at the pad', '', ''],
                         ['', '', 'continuing']])
check('notes from an earlier completion of the turn (a cut message)', a._v62_attempt_notes.get(3),
      ['- the red key opens the blue door\n- the timer refills at the pad'])
a = agent_with(GameSess(over=False))
a._v6_stats['completions'] = 5
a._v62_fn_window, a._v62_fn_windows = (3, 12), {2: 10, 3: 12}
a._v62_save_summary('Rule L2:\n- the blue block slides until it hits a wall\n- the goal is the pad')
check('two quick level-ups: a "Rule L2:" save goes to level 2', (sorted(a._v6_rules), a._v6_rules.get(2, '')[:12]),
      ([2], '- the blue b'))
check('T1 the automatic restart that died', A._game_over_text([H('RESET', B0, automatic=True, game_over=True)]),
      A.TEXTS['game_over_auto_reset'])
check('T1 the model RESET that died', A._game_over_text([H('RESET', B0, game_over=True)]), A.TEXTS['game_over_no_restart'])
check('T2 the takeover without hidden history', A.TEXTS['takeover_restart_shown'].endswith('are not in this conversation.'), True)
check('T3 collapse at most 3 in a row; never with the ledger or GAME OVER', ('and run < 3' in src_a,
      "'Continuations that ended in GAME OVER' in text" in src_a), (True, True))
check('T4 no action reminder above the notes request', src_a.count("text.replace(TEXTS['reminder'], '')"), 2)
a = agent_with(GameSess(over=False))
text, cut = a._trim_tool_text('x' * 20000)
check('T5 a cut line says so', ('a line over the limit was cut' in text, '0 lines omitted' in text), (True, False))

sv._is_engine_game_over, sv._level_number = _real_game_over, _real_level_number
print('FAILS:', fails)
