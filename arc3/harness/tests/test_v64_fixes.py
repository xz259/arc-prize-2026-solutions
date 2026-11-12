"""v6.4 fixes from the 12-h audit (SYNTHESIS_12H.md A1-A5, A7, B1-B3), through agent.analyze with the real sandbox where the
path needs it (pattern of test_v62_round5l_loop.py). usage (dev copy of harness_v6): PYTHONPATH=src:. python3 tests/test_v64_fixes.py"""
import json
import os
import tempfile
from pathlib import Path

os.environ.update({'MULTIMODAL_CONTEXT': 'current_grid', 'MULTIMODAL_UPSCALE': '8', 'ARC3_PERSISTENT_FUNCTIONS': '1',
                   'ARC3_PERSISTENT_FUNCTIONS_SCOPE': 'game', 'ARC3_PERSISTENT_FUNCTIONS_IMPORTS': '1',
                   'ARC3_ANIMATION': '1', 'LOCAL_ANALYZER_TOOL_TIMEOUT': '10', 'ARC3_ACTION_ECHO': '1',
                   'ARC3_V6_YIELD_TOKENS': '4096'})
for k, v in dict(ARC3_V61_TEXTS='1', ARC3_V6_A13='1', ARC3_V61_SUMMARY='1', ARC3_NEW_CHANGED_PROMPTS='1',
                 ARC3_ACTION_INFO='1', ARC3_REPEAT_STATE_GUARD='0', ARC3_V61_RESTART_TOKENS='150000', ARC3_V6_A3='1',
                 ARC3_GUARDS_FROM_LEVEL='2', ARC3_BATCH_NOOP_BLOCK='1', ARC3_STALE_STATE_BLOCK='1').items():
    os.environ.setdefault(k, v)
import v62.patches  # noqa: E402,F401
import v62.retention as r  # noqa: E402
r.install()
from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state  # noqa: E402
import v62.agent as A  # noqa: E402
import v62.transport as T  # noqa: E402
from v62.transport import Completion  # noqa: E402
fails = []


def check(name, got, want):
    ok = got == want
    print('%-66s -> %-44r %s' % (name, got if not isinstance(got, str) else got[:44], 'OK' if ok else 'WRONG (want %r)' % (want,)))
    if not ok:
        fails.append(name)


class FakeSession:
    """script items: (code or None for a reply without a tool call, tokens, cues, reasoning)"""

    def __init__(self, script):
        self.script, self.sent = list(script), []
        self.rebased_away, self.blocks, self.closed = set(), [], False

    def generate(self, external):
        self.sent.append(json.loads(json.dumps(external, default=str)))
        item = self.script.pop(0)
        code, tokens = item[0], item[1]
        cues = item[2] if len(item) > 2 else ()
        reasoning = item[3] if len(item) > 3 else 'thinking'
        content = item[4] if len(item) > 4 else ''
        calls = ([dict(id='call-%d' % len(self.sent), type='function',
                       function=dict(name='python', arguments=json.dumps(dict(code=code))))] if code is not None else [])
        msg = dict(role='assistant', content=content, reasoning_content=reasoning, tool_calls=calls)
        return Completion(message=msg, finish_reason='tool_calls' if calls else 'stop',
                          usage=dict(prompt_tokens=10, completion_tokens=tokens), token_ids=[1] * tokens,
                          assistant_token_ids=[1] * tokens, pending=False, closed_by=None, persistent={}, seconds=0.0,
                          cues=tuple(cues))

    def span_evicted(self, a, b):
        return 0

    def close(self):
        self.closed = True


class FakeGame:
    def __init__(self, state_path):
        self.state_path, self.step, self.level = state_path, 0, 1
        self.history = [HistoryEntry(action='', frame=Frame(grid=[[0] * 64 for _ in range(64)], step=0, level=1))]
        self.v6_anim = {}
        write_runtime_state(self.state_path, current_frame=self.history[-1].frame, history=self.history)

    def _add(self, name, result):
        self.step += 1
        done = name == 'UP' and self.level == 1
        result = dict(result, action_num=self.step, level=self.level, action_name=name, action_display=name,
                      valid_actions=['UP', 'RIGHT', 'LEFT'], level_completed=done)
        if done:
            self.level = 2
        self.history.append(HistoryEntry(action=name, frame=Frame(grid=[[0] * 64 for _ in range(64)], step=self.step,
                                                                 level=self.level), result=result))
        write_runtime_state(self.state_path, current_frame=self.history[-1].frame, history=self.history)
        return result

    def step_env(self, arguments):
        if arguments.get('query') == 'animation':
            return {'record': None}
        executed = [self._add(a.get('action'), dict(executed=True, gameplay_changed=True, board_changed=True))
                    for a in arguments.get('actions') or []]
        self.__dict__['v6_attempt_actions'] = self.__dict__.get('v6_attempt_actions', 0) + len(executed)
        last = dict(executed[-1]) if executed else {}
        last.update(executed=bool(executed), executed_count=len(executed),
                    executed_actions=[e['action_display'] for e in executed], requested_count=len(executed))
        return last

    def _execute_action(self, action, *, batch_index, batch_size, **kw):      # the restart's RESET
        self.__dict__['v6_attempt_actions'] = 0
        return self._add('RESET', dict(executed=True, gameplay_changed=True, board_changed=True))


def new_agent(script, script2=None, log=None):
    sess = FakeSession(script)
    sess2 = FakeSession(script2 or [('action(["LEFT"])', 100)])
    agent = A.V6Agent(session=sess, logger=lambda k, p: (log if log is not None else []).append((k, p)),
                      count_tokens=lambda t: max(1, len(t) // 4), model='M', base_url='http://x/v1', provider='vllm',
                      session_factory=lambda: sess2)
    return agent, sess, sess2


def run_notes_turn(notes_script, title):
    """A turn crossing a 500-token restart budget, then the notes turn with `notes_script`, then the restart."""
    A.RESTART_TOKENS = 500
    state = Path(tempfile.mkdtemp()) / 'game_runtime_state.json'
    game = FakeGame(state)
    log = []
    agent, sess, sess2 = new_agent([('action(["RIGHT"])', 600)] + notes_script, log=log)
    go = lambda: agent.analyze(state, game.step, valid_actions=['RIGHT', 'LEFT'], step_env=game.step_env,
                               should_stop=lambda: False)
    go()                    # crosses the budget
    go()                    # the notes turn (and its retry, if any)
    go()                    # the restart
    first = [m for m in sess2.sent[0] if m.get('role') == 'user'][-1]['content'][0]['text']
    return agent, sess, log, first


notes_ok = 'save_summary("Notes L1:\\n- the red block moves right one cell per RIGHT\\n- the goal is the green pad")'

# === A1: a refused action in the notes turn, then the retry saves the notes ===========================================
agent, sess, log, first = run_notes_turn([('action(["RIGHT"])', 200), (notes_ok, 150)], 'refused')
retries = [p for k, p in log if k == 'v64-attempt-notes-retry']
check('A1 refused action -> one retry, saved', [(p.get('reason'), p.get('outcome')) for p in retries],
      [('an action call was refused', 'saved')])
retry_msg = sess.sent[-1][-1]
check('A1 retry: tool result then the user note', (sess.sent[-1][-2].get('role'), retry_msg.get('role'),
                                                  retry_msg.get('content', '').startswith('No notes were saved')),
      ('tool', 'user', True))
check('A1 notes saved by the retry', agent._v62_attempt_notes.get(1),
      ['- the red block moves right one cell per RIGHT\n- the goal is the green pad'])
check('A1 the restart shows them', '- the red block moves right one cell per RIGHT' in first, True)

# === A1: a SyntaxError in the save, then the retry ====================================================================
agent, sess, log, first = run_notes_turn([('save_summary("""Notes L1:\\n- unclosed', 200), (notes_ok, 150)], 'syntax')
check('A1 syntax error -> reason names it', [p.get('reason') for k, p in log if k == 'v64-attempt-notes-retry'],
      ['the python call had a syntax error at line 1'])
check('A1 syntax error: saved on retry', bool(agent._v62_attempt_notes.get(1)), True)

# === A1: a reply without a python call, then the retry =================================================================
agent, sess, log, first = run_notes_turn([(None, 200), (notes_ok, 150)], 'nocall')
check('A1 no call -> retry', [p.get('reason') for k, p in log if k == 'v64-attempt-notes-retry'], ['the reply had no python call'])
check('A1 no call: saved on retry', bool(agent._v62_attempt_notes.get(1)), True)

# === A1: two failures -> exactly one retry, then the restart without notes ============================================
agent, sess, log, first = run_notes_turn([('action(["RIGHT"])', 200), ('action(["LEFT"])', 150)], 'twice')
check('A1 at most one retry, outcome none', [p.get('outcome') for k, p in log if k == 'v64-attempt-notes-retry'], ['none'])
check('A1 then the restart without notes', (agent._v62_attempt_notes.get(1), A.TEXTS['attempt_notes_none'] in first,
                                           agent._v6_stats.get('restarts')), (None, True, 1))

# === A1: a saved note -> no retry ======================================================================================
agent, sess, log, first = run_notes_turn([(notes_ok, 200)], 'saved')
check('A1 no retry after a save', (len([1 for k, _ in log if k == 'v64-attempt-notes-retry']), bool(agent._v62_attempt_notes.get(1))),
      (0, True))

# === A1: a "Notes L1:" block in the reply text (capture fallback) -> no retry ==========================================
agent, sess, log, first = run_notes_turn([(None, 200, (), 'thinking', 'Notes L1:\n- the block moves right\n- the goal is green')],
                                         'capture')
check('A1 no retry when the text capture finds notes', (len([1 for k, _ in log if k == 'v64-attempt-notes-retry']),
                                                         bool(agent._v62_attempt_notes.get(1))), (0, True))

# === A2: a later save does not replace a level's summary once an action executed on the next level ====================
A.RESTART_TOKENS = 150000
state = Path(tempfile.mkdtemp()) / 'game_runtime_state.json'
game = FakeGame(state)
log = []
agent, sess, sess2 = new_agent([('action(["UP"])', 300),
                                ('save_summary("Rule L1:\\n- UP clears the level when pressed once\\n- the board is static")\n'
                                 'action(["LEFT"])', 300),
                                ('save_summary("- level 2 notes: the board has a red block on the left side")\n'
                                 'action(["LEFT"])', 300)],
                         log=log)
go = lambda: agent.analyze(state, game.step, valid_actions=['UP', 'LEFT'], step_env=game.step_env, should_stop=lambda: False)
go()
go()
go()
check('A2 the first save is the L1 summary, not replaced after L2 actions', agent._v6_rules.get(1),
      '- UP clears the level when pressed once\n- the board is static')
saved = [p for k, p in log if k == 'v62-summary-saved']
check('A2 one save logged, one refusal', (len(saved), [p.get('tag') for k, p in log if k == 'v64-summary-not-replaced']),
      (1, [None]))

# A2: a save tagged with the window's level after acting replaces (the model's own attribution, IE)
state = Path(tempfile.mkdtemp()) / 'game_runtime_state.json'
game = FakeGame(state)
agent, sess, sess2 = new_agent([('action(["UP"])', 300),
                                ('save_summary("Rule L1:\\n- UP clears the level when pressed once\\n- first")\naction(["LEFT"])',
                                 300),
                                ('save_summary("Rule L1:\\n- UP clears the level when pressed once\\n- second version")\n'
                                 'action(["LEFT"])', 300)])
go = lambda: agent.analyze(state, game.step, valid_actions=['UP', 'LEFT'], step_env=game.step_env, should_stop=lambda: False)
go()
go()
go()
check('A2 a tagged "Rule L1:" re-save still replaces', agent._v6_rules.get(1),
      '- UP clears the level when pressed once\n- second version')

# A2: act first, then a first save -> accepted
state = Path(tempfile.mkdtemp()) / 'game_runtime_state.json'
game = FakeGame(state)
agent, sess, sess2 = new_agent([('action(["UP"])', 300), ('action(["LEFT"])', 300),
                                ('save_summary("Rule L1:\\n- UP clears the level when pressed once\\n- nothing else moves")\n'
                                 'action(["LEFT"])', 300)])
go = lambda: agent.analyze(state, game.step, valid_actions=['UP', 'LEFT'], step_env=game.step_env, should_stop=lambda: False)
go()
go()
go()
check('A2 a first save after acting is still accepted', agent._v6_rules.get(1),
      '- UP clears the level when pressed once\n- nothing else moves')

# === A2 (IE review): a tag naming another level is never stored as the open window's level, whatever the action state ====
def save_case(existing, text):
    ag, _, _ = new_agent([('action(["RIGHT"])', 10)])
    ag._v62_attempt_request = None
    ag._v6_stats['completions'] = 10
    ag._v62_fn_window, ag._v62_fn_windows = (2, 99), {2: 99}          # L2's window open; L1's closed
    ag._v6_rules = dict(existing)
    ag._v62_rule_fn = set(existing)
    return ag, ag._v62_save_summary(text)


body = '\n- the green bar shrinks by one cell per move\n- the exit opens when it is empty'
for existing, tag, want_note in (({1: '- old L1', 2: '- old L2'}, 'Rule L1:', 'summary_tag_closed'),   # v6.5 A2r: L1 has one
                                 ({1: '- old L1'}, 'Rule L1:', 'summary_tag_closed'),
                                 ({2: '- old L2'}, 'Rule L3:', 'summary_tag_open_level'), ({}, 'Rule L3:', 'summary_tag_open_level')):
    ag, res = save_case(existing, tag + body)
    check('A2 %s with%s an L2 summary: refused, L2 kept' % (tag, '' if existing else 'out'),
          (res.get('saved'), res.get('note') == A._fill(A.TEXTS[want_note], level=int(tag[6])), ag._v6_rules.get(2)),
          (False, True, existing.get(2)))
ag, res = save_case({2: '- old L2'}, 'Rule L2:' + body)
check('A2 a tag naming the window level replaces (before any action)', (res.get('saved'), ag._v6_rules.get(2)),
      (True, body.strip('\n')))
ag, res = save_case({}, body)
check('A2 an untagged first save is stored for the window level', (res.get('saved'), ag._v6_rules.get(2)),
      (True, body.strip('\n')))
check('A2 tag refusals counted', ag._v6_stats.get('summary_tag_refused'), None)

# === A3: the N1 nudge does not quote the last line after an injected cue ================================================
cue = T.TURN_BUDGET_CUE.split('\n</think>')[0].strip().replace('{left:,}', '3,950')
for last, want in (('Now I will press RIGHT twice and check the board.', True), (cue, False)):
    state = Path(tempfile.mkdtemp()) / 'game_runtime_state.json'
    game = FakeGame(state)
    agent, sess, sess2 = new_agent([(None, 300, (), 'some analysis\n' + last), ('action(["RIGHT"])', 100)])
    agent.analyze(state, game.step, valid_actions=['RIGHT', 'LEFT'], step_env=game.step_env, should_stop=lambda: False)
    nudge = sess.sent[1][-1]['content']
    check('A3 N1 quotes a plan line (%s)' % ('cue' if last == cue else 'model'), 'Your reasoning ended with' in nudge, want)
check('A3 receipt', agent._v6_stats.get('n1_quote_skipped'), 1)

# === A4, B1, B2: texts =================================================================================================
check('A4 same_report says what was compared', A.TEXTS['same_report'],
      "Your previous call's report was identical to the one before it, so it is not repeated here.")
s = 'At the restart, programs you saved on this level are set aside'
check('B1 the notes request names the archive (tokens / noact / position)',
      tuple(s in A.TEXTS[k] for k in ('attempt_summary_request', 'attempt_summary_request_noact',
                                      'attempt_summary_request_position')), (True, True, True))
if os.environ.get('V64_B2') == '1':
    check('B2 the 12K cue ends on the call', T.TURN_BUDGET_CUE.split('\n</think>')[0].endswith('act now with one python call.'),
          True)

# === A5: recovered code must compile ===================================================================================
check('A5 compiles: code', A._v64_compile_error('x = 1\nprint(x)'), None)
check('A5 compiles: a tag fragment', (A._v64_compile_error('<parameter=code\nprint(1)') or '').split(' at')[0], 'SyntaxError')
check('A5 compiles: an unclosed string', (A._v64_compile_error('save_summary("""Notes') or '').split(' at')[0], 'SyntaxError')

# === A7: the restart opener names completed levels without a summary ====================================================
agent, sess, sess2 = new_agent([('action(["RIGHT"])', 10)])
agent._v6_rules = {1: '- rule one'}
text = agent._v61_takeover_text(3, 40, 'UP, LEFT', 'R', 'Run time left: 1 h', None)
check('A7 L1 shown, L2 named as missing', ('- rule one' in text, '- L2: no summary was saved for this level.' in text),
      (True, True))

# === B3: no "That is fine" while a no-action warning stands ============================================================
agent._v62_resume_error, agent._v62_noact_warned = False, 1
check('B3 key with a warning', agent._v62_resumption_key(), 'resumption_noact')
check('B3 text without "That is fine"', 'That is fine' in A.TEXTS['resumption_noact'], False)
agent._v62_noact_warned = 0
check('B3 key without a warning', agent._v62_resumption_key(), 'resumption')
check('A7 receipt', agent._v6_stats.get('opener_missing_summaries'), 1)
print('FAILS:', fails)
