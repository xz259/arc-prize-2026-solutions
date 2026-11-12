"""v6.3 items B, C, E, F (user 2026-10-12/13; HARNESS_V63_PLAN): the no-action token budget through agent.analyze with the
real sandbox: the counter accumulates over non-acting calls and resumes, resets on an executed action and at a level change;
warnings once per stretch at each threshold (the first with the untried list and the limit), taking the slot of a nudge or a
breaker note and holding the token yield; the hard limit ends the turn, the notes turn names the reason, the swap-out and
restart follow with reason 'noact'; the 25-minute rule skips both and the attempt continues; the 8-call line is off; the
levels-complete end fires only for a game whose completions reached number_of_levels without a WIN; the defaults (1500 s,
32K / 48K, 64K, turn check 12K). usage (pod copy): PYTHONPATH=src:. python3 tests/test_v63_noact_budget.py"""
import inspect
import json
import os
import tempfile
from pathlib import Path

os.environ.update({'MULTIMODAL_CONTEXT': 'current_grid', 'MULTIMODAL_UPSCALE': '8', 'ARC3_PERSISTENT_FUNCTIONS': '1',
                   'ARC3_PERSISTENT_FUNCTIONS_SCOPE': 'game', 'ARC3_PERSISTENT_FUNCTIONS_IMPORTS': '1',
                   'ARC3_ANIMATION': '1', 'LOCAL_ANALYZER_TOOL_TIMEOUT': '10', 'ARC3_ACTION_ECHO': '1'})
for k, v in dict(ARC3_V61_TEXTS='1', ARC3_V6_A13='1', ARC3_V61_SUMMARY='1', ARC3_NEW_CHANGED_PROMPTS='1',
                 ARC3_ACTION_INFO='1', ARC3_REPEAT_STATE_GUARD='0', ARC3_V61_RESTART_TOKENS='150000', ARC3_V6_A3='1',
                 ARC3_GUARDS_FROM_LEVEL='2', ARC3_BATCH_NOOP_BLOCK='1', ARC3_STALE_STATE_BLOCK='1').items():
    os.environ.setdefault(k, v)
import arcengine  # noqa: E402
import v62.patches  # noqa: E402,F401
import v62.retention as r  # noqa: E402
r.install()
from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state  # noqa: E402
import v62.agent as A  # noqa: E402
import v62.run_v6 as RUN  # noqa: E402
from v62.transport import Completion  # noqa: E402
fails = []


def check(name, got, want):
    ok = got == want
    print('%-82s -> %-40r %s' % (name, got if not isinstance(got, str) else got[:40], 'OK' if ok else 'WRONG (want %r)' % (want,)))
    if not ok:
        fails.append(name)


# --- defaults (items B, C, F) ---------------------------------------------------------------------------------------
check('defaults: RESTART_MIN_LEFT 1500 s (item C)', A.RESTART_MIN_LEFT, 1500.0)
check('defaults: warnings at 32K and 48K, limit 64K, 8-call line off (item B)', (A.NOACT_WARN, A.NOACT_TOKENS, A.NO_ACTION_CALLS), ((32000, 48000), 64000, 0))
check('defaults: texts present', all(k in A.TEXTS for k in ('noact_warn_1', 'noact_warn_2', 'noact_limit_clause', 'attempt_summary_request_noact')), True)
check('defaults: run_v6 turn check 12,000 (item F)', "setdefault('ARC3_ASSISTANT_TURN_CHECK', '12000')" in inspect.getsource(RUN.main), True)
check('defaults: levels-complete end on (item E)', A.LEVELS_COMPLETE_END, True)


class FakeSession:
    """script items: (code, tokens) for a python call, or dict(text=..., tokens=...) for a reply without a tool call."""

    def __init__(self, script):
        self.script, self.sent = list(script), []
        self.rebased_away, self.blocks, self.closed = set(), [], False

    def generate(self, external):
        self.sent.append(json.loads(json.dumps(external, default=str)))
        item = self.script.pop(0)
        if isinstance(item, dict):
            tokens = item['tokens']
            msg = dict(role='assistant', content=item['text'], reasoning_content='thinking')
        else:
            code, tokens = item
            msg = dict(role='assistant', content='', reasoning_content='thinking',
                       tool_calls=[dict(id='call-%d' % len(self.sent), type='function',
                                        function=dict(name='python', arguments=json.dumps(dict(code=code))))])
        return Completion(message=msg, finish_reason='tool_calls' if 'tool_calls' in msg else 'stop',
                          usage=dict(prompt_tokens=10, completion_tokens=tokens), token_ids=[1] * tokens,
                          assistant_token_ids=[1] * tokens, pending=False, closed_by=None, persistent={}, seconds=0.0)

    def span_evicted(self, a, b):
        return 0

    def close(self):
        self.closed = True


class Raw:
    def __init__(self, state):
        self.state = state


class CurState:
    def __init__(self, state):
        self.raw = Raw(state)


class Run:
    def __init__(self, n):
        self.number_of_levels, self.levels_completed, self.state, self.solver_note, self.final_score = n, 0, 'playing', None, None


class TG:
    """the taaf Game of the fake session (item E)"""

    def __init__(self, n=5, state=arcengine.GameState.NOT_FINISHED):
        self.number_of_levels, self.game_run, self.current_state, self.finished = n, Run(n), CurState(state), 0

    def finish_game(self):
        self.finished += 1
        if self.game_run.state == 'playing':
            self.game_run.state = 'gave_up'
        self.game_run.final_score = 1.0


class FakeGame:
    remaining = 20000.0

    def __init__(self, state_path, n_levels=5):
        self.state_path, self.step, self.level = state_path, 0, 1
        self.game = TG(n_levels)
        self.history = [HistoryEntry(action='', frame=Frame(grid=[[0] * 64 for _ in range(64)], step=0, level=1))]
        self.v6_anim = {}
        write_runtime_state(self.state_path, current_frame=self.history[-1].frame, history=self.history)

    def _add(self, name, result):
        self.step += 1
        done = name == 'UP' and self.level == 1
        result = dict(result, action_num=self.step, level=self.level, action_name=name, action_display=name,
                      valid_actions=['RIGHT', 'LEFT', 'UP'], level_completed=done)
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

    def timing_payload(self):
        return dict(time_remaining_seconds=self.remaining)


VALID = ['RIGHT', 'LEFT', 'UP']


def setup(script, script2=(), remaining=20000.0, n_levels=5):
    A.ta._priority_gate = lambda: None             # restarts in place (no allocator gate)
    A.RESTART_TOKENS, A.RESTART_NOTES = 10 ** 9, True
    state = Path(tempfile.mkdtemp()) / 'game_runtime_state.json'
    game = FakeGame(state, n_levels)
    game.remaining = remaining
    log = []
    sess, sess2 = FakeSession(script), FakeSession(script2)
    agent = A.V6Agent(session=sess, logger=lambda k, p: log.append((k, p)), count_tokens=lambda t: max(1, len(t) // 4),
                      model='M', base_url='http://x/v1', provider='vllm', session_factory=lambda: sess2)
    go = lambda: agent.analyze(state, game.step, valid_actions=VALID, step_env=game.step_env, should_stop=lambda: False)
    return game, sess, sess2, agent, log, go


def user_msgs(sent):
    return [m['content'] for batch in sent for m in batch if m.get('role') == 'user' and isinstance(m.get('content'), str)]


# === 1. counter, both warnings, the yield hold, the limit, the notes turn, the restart ================================
A.NOACT_WARN, A.NOACT_TOKENS, A.YIELD_TOKENS = (300, 450), 600, 300
game, sess, sess2, agent, log, go = setup(
    [('print(1)', 200), ('print(2)', 150), ('print(3)', 50), ('print(4)', 100), ('print(5)', 150),
     ('save_summary("Notes L1:\\n- nothing executed; the board is static")', 100)],
    [('action(["LEFT"])', 100)])
r1 = go()
w = [p for k, p in log if k == 'v62-noact-warning']
check('1 first warning at 350 (>= 300) with the untried list; the yield held by it', (
    [(x['which'], x['tokens'], x['threshold'], x['untried']) for x in w],
    [p['by'] for k, p in log if k == 'v62-yield-held']), ([(1, 350, 300, ['RIGHT', 'LEFT', 'UP'])], [['noact_warning']]))
texts = user_msgs(sess.sent)
check('1 ... the text names the count, the limit and the untried actions', any(
    'generated 350 tokens in this conversation since your last executed action' in t and 'If you reach 600 tokens' in t
    and 'in this conversation: RIGHT, LEFT, UP.' in t and 'restored as soon as an action executes' in t for t in texts), True)
check('1 the turn then yielded at 400 turn tokens (no hold without a new warning)', (r1.yielded_control, [k for k, _ in log].count('v6-yield')), (True, 1))
r2 = go()                                   # resume: 100 -> 500 (second warning), 150 -> 650 (the limit)
w = [p for k, p in log if k == 'v62-noact-warning']
check('1 second warning at 500 (>= 450), once', [(x['which'], x['tokens']) for x in w], [(1, 350), (2, 500)])
check('1 ... its text', any('500 of 600 tokens generated without an executed action' in t for t in user_msgs(sess.sent)), True)
check('1 the limit at 650 ended the turn (resume pending), no restart yet', (r2.step_executed, agent._v6_resume, agent._v62_noact_tokens,
      agent._v6_stats.get('restarts')), (False, True, 650, None))
r3 = go()                                   # the notes turn
st = [p for k, p in log if k == 'v62-summary-turn']
check('1 notes turn: reason noact, counts logged', [(x['reason'], x['noact_tokens'], x['tokens_on_level']) for x in st], [('noact', 650, 650)])
req = sess.sent[-1][-1]['content'][0]['text']
check('1 ... the request names the reason and the no-action count', ('generated 650 tokens on level 1 without executing an action, so the level restarts' in req,
      'save_summary(text)' in req), (True, True))
check('1 ... notes saved', agent._v62_attempt_notes.get(1), ['- nothing executed; the board is static'])
r4 = go()                                   # the restart (in place: no gate) and the new conversation acts
check('1 restart: reason noact in the restart event and the noact-restart event', (
    [p.get('reason') for k, p in log if k == 'v61-restart'], [(p['level'], p['tokens']) for k, p in log if k == 'v62-noact-restart']),
    (['noact'], [(1, 750)]))
check('1 ... counters', (agent._v6_stats.get('noact_warnings_1'), agent._v6_stats.get('noact_warnings_2'), agent._v6_stats.get('noact_restarts'),
      agent._v6_stats.get('restarts'), agent._v6_stats.get('summary_turns'), agent._v6_stats.get('attempt_notes_saved')), (1, 1, 1, 1, 1, 1))
check('1 ... the new conversation acted and the stretch counter is 0', (r4.step_executed, sess.closed, agent._v62_noact_tokens, agent._v62_noact_warned), (True, True, 0, 0))
first = [m for m in sess2.sent[0] if m.get('role') == 'user'][-1]['content'][0]['text']
check('1 ... its first observation shows the notes', '- nothing executed; the board is static' in first, True)

# === 2. an executed action restores the budget; a new stretch warns again =============================================
A.YIELD_TOKENS = 10 ** 9
game, sess, sess2, agent, log, go = setup([('print(1)', 350), ('action(["RIGHT"])', 10), ('print(2)', 320), ('action(["LEFT"])', 10)])
go(); go()
check('2 warning in each stretch; reset by the action', ([p['tokens'] for k, p in log if k == 'v62-noact-warning'],
      agent._v6_stats.get('noact_warnings_1'), agent._v62_noact_tokens), ([350, 320], 2, 0))

# === 3. the last 25 minutes: no notes turn, no restart; the attempt continues in place =================================
game, sess, sess2, agent, log, go = setup([('print(1)', 700), ('print(2)', 10), ('print(3)', 10), ('action(["RIGHT"])', 10)], remaining=1000.0)
r1 = go()
check('3 the limit broke the turn', (r1.step_executed, agent._v6_resume), (False, True))
r2 = go()
sk = [p for k, p in log if k == 'v62-restart-skipped']
check('3 skipped late: logged with the no-action count; no notes turn; the turn played on and acted', (
    [(x['remaining_s'], x['noact_tokens']) for x in sk], [k for k, _ in log].count('v62-summary-turn'), r2.step_executed,
    agent._v6_stats.get('restarts_skipped_late'), agent._v6_stats.get('restarts')), ([(1000, 700)], 0, True, 1, None))

# === 4. a reply without a tool call: the warning takes the nudge's slot ================================================
game, sess, sess2, agent, log, go = setup([dict(text='I am still thinking.', tokens=320), ('action(["RIGHT"])', 10)])
go()
check('4 the warning replaced the N1 nudge', (user_msgs(sess.sent)[-1].startswith('You have generated 320 tokens'),
      agent._v6_stats.get('nudges_replaced'), agent._v6_stats.get('nudges')), (True, 1, {}))

# === 5. the breaker note gives way to the warning ====================================================================
game, sess, sess2, agent, log, go = setup([('1/0', 100), ('1/0', 250), ('action(["RIGHT"])', 10)])
go()
check('5 the second identical error coincided with the warning: breaker replaced, warning sent', (
    agent._v6_stats.get('breaker_replaced'), agent._v6_stats.get('breaker_notes'),
    user_msgs(sess.sent)[-1].startswith('You have generated 350 tokens')), (1, None, True))

# === 6. a level change starts a new stretch ===========================================================================
game, sess, sess2, agent, log, go = setup([('action(["UP"])', 50), ('print(2)', 10), ('action(["RIGHT"])', 10)])
go()
agent._v62_noact_tokens, agent._v62_noact_warned = 500, 2
go()
check('6 level 2: the stretch counter and warnings were reset (then 10 tokens, then the action)', (game.level, agent._v6_stats.get('noact_warnings_1'), agent._v62_noact_tokens), (2, None, 0))

# === 7. the 8-call line is off =========================================================================================
game, sess, sess2, agent, log, go = setup([('print(%d)' % i, 5) for i in range(10)] + [('action(["RIGHT"])', 5)])
go()
check('7 ten non-acting calls: no 8-call line, counter kept', ([k for k, _ in log].count('v61-no-action-line'), agent._v6_stats.get('no_action_lines')), (0, None))

# === 8. item G: opener token instrumentation ==========================================================================
op = [p for k, p in log if k == 'v6-opener']
check('8 opener events carry text_tokens and pending_tool_tokens', (all(isinstance(p.get('text_tokens'), int) and p['text_tokens'] > 0 for p in op),
      all('pending_tool_tokens' in p for p in op)), (True, True))

# === 9. item E: the levels-complete end ================================================================================
game, sess, sess2, agent, log, go = setup([('action(["RIGHT"])', 10), ('action(["RIGHT"])', 10), ('action(["RIGHT"])', 10)], n_levels=2)
go()                                         # started
game.game.game_run.levels_completed = 1
r = go()
check('9 completions below number_of_levels: nothing happens', (r.step_executed, agent._v6_stats.get('levels_complete_ends'), game.game.finished), (True, None, 0))
game.game.game_run.levels_completed = 2
game.game.current_state = CurState(arcengine.GameState.WIN)
r = go()
check('9 all levels and WIN: nothing happens (the solver ends the game itself)', (r.step_executed, agent._v6_stats.get('levels_complete_ends'), game.game.finished), (True, None, 0))
game.game.current_state = CurState(arcengine.GameState.NOT_FINISHED)
n_sent = len(sess.sent)
r = go()
ev = [p for k, p in log if k == 'v63-levels-complete-end']
check('9 all levels, NOT_FINISHED: the game is ended, scored, session closed, no model call', (
    r.step_executed, game.game.finished, game.game.game_run.state, sess.closed, len(sess.sent) == n_sent,
    agent._v6_stats.get('levels_complete_ends'), [(p['levels_completed'], p['n_levels'], p['engine_state']) for p in ev],
    (game.game.game_run.solver_note or '').startswith('v6.3 item E')), (False, 1, 'gave_up', True, True, 1, [(2, 2, 'NOT_FINISHED')], True))
A.LEVELS_COMPLETE_END = False
game, sess, sess2, agent, log, go = setup([('action(["RIGHT"])', 10), ('action(["RIGHT"])', 10)], n_levels=1)
go()
game.game.game_run.levels_completed = 1
go()
check('9 switch off: no end', (game.game.finished, agent._v6_stats.get('levels_complete_ends')), (0, None))
A.LEVELS_COMPLETE_END = True
print('FAILS:', fails)
print('test_v63_noact_budget', 'PASS' if not fails else 'FAIL')
