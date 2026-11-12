"""v6.2 round 5l loop tests (V62_BUILD items 108 and 111) through agent.analyze with the real sandbox: save_summary stores the
requested level summary; at the token restart an attempt in progress plays on (D2), then a notes turn (D1) refuses actions,
saves the notes, and the restarted conversation's first observation shows them. usage (pod copy):
PYTHONPATH=src:. python3 tests/test_v62_round5l_loop.py"""
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
from v62.transport import Completion  # noqa: E402
fails = []


def check(name, got, want):
    ok = got == want
    print('%-66s -> %-44r %s' % (name, got if not isinstance(got, str) else got[:44], 'OK' if ok else 'WRONG (want %r)' % (want,)))
    if not ok:
        fails.append(name)


class FakeSession:
    def __init__(self, script):
        self.script, self.sent = list(script), []
        self.rebased_away, self.blocks, self.closed = set(), [], False

    def generate(self, external):
        self.sent.append(json.loads(json.dumps(external, default=str)))
        code, tokens = self.script.pop(0)
        msg = dict(role='assistant', content='', reasoning_content='thinking',
                   tool_calls=[dict(id='call-%d' % len(self.sent), type='function',
                                    function=dict(name='python', arguments=json.dumps(dict(code=code))))])
        return Completion(message=msg, finish_reason='tool_calls', usage=dict(prompt_tokens=10, completion_tokens=tokens),
                          token_ids=[1] * tokens, assistant_token_ids=[1] * tokens, pending=False, closed_by=None,
                          persistent={}, seconds=0.0)

    def span_evicted(self, a, b):
        return 0

    def close(self):
        self.closed = True


class FakeGame:
    def __init__(self, state_path):
        self.state_path, self.step = state_path, 0
        self.history = [HistoryEntry(action='', frame=Frame(grid=[[0] * 64 for _ in range(64)], step=0, level=1))]
        self.v6_anim = {}
        write_runtime_state(self.state_path, current_frame=self.history[-1].frame, history=self.history)

    def _add(self, name, result):
        self.step += 1
        result = dict(result, action_num=self.step, level=1, action_name=name, action_display=name,
                      valid_actions=['RIGHT', 'LEFT'])
        self.history.append(HistoryEntry(action=name, frame=Frame(grid=[[0] * 64 for _ in range(64)], step=self.step,
                                                                 level=1), result=result))
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


# === item 108: save_summary through the sandbox, for the requested level ==============================================
state = Path(tempfile.mkdtemp()) / 'game_runtime_state.json'
game = FakeGame(state)
log = []
sess = FakeSession([('save_summary("Rule L1:\\n- the red block moves right by one cell\\n- walls stop it")\n'
                     'action(["RIGHT"])', 200)])
agent = A.V6Agent(session=sess, logger=lambda k, p: log.append((k, p)), count_tokens=lambda t: max(1, len(t) // 4),
                  model='M', base_url='http://x/v1', provider='vllm')
agent._v62_fn_window = (1, 99)
res = agent.analyze(state, 0, valid_actions=['RIGHT', 'LEFT'], step_env=game.step_env, should_stop=lambda: False)
check('108 the summary saved through the sandbox', agent._v6_rules.get(1),
      '- the red block moves right by one cell\n- walls stop it')
check('108 the same call acted', (res.step_executed, game.step), (True, 1))
check('108 logged with its path', [p.get('kind') for k, p in log if k == 'v62-summary-saved'], ['level'])

# === item 108 through a real level-up: the opener asks, the next call's save_summary stores the summary ================
class LevelGame(FakeGame):
    level = 1

    def _add(self, name, result):
        self.step += 1
        done = name == 'UP' and self.level == 1
        result = dict(result, action_num=self.step, level=self.level, action_name=name, action_display=name,
                      valid_actions=['UP', 'LEFT'], level_completed=done)
        if done:
            self.level = 2
        self.history.append(HistoryEntry(action=name, frame=Frame(grid=[[0] * 64 for _ in range(64)], step=self.step,
                                                                 level=self.level), result=result))
        write_runtime_state(self.state_path, current_frame=self.history[-1].frame, history=self.history)
        return result


state = Path(tempfile.mkdtemp()) / 'game_runtime_state.json'
game = LevelGame(state)
log = []
sess = FakeSession([('action(["UP"])', 300), ('save_summary("Rule L1:\\n- UP clears the level when pressed once\\n- the board '
                                              'is static otherwise")\naction(["LEFT"])', 300)])
agent = A.V6Agent(session=sess, logger=lambda k, p: log.append((k, p)), count_tokens=lambda t: max(1, len(t) // 4),
                  model='M', base_url='http://x/v1', provider='vllm')
go = lambda: agent.analyze(state, game.step, valid_actions=['UP', 'LEFT'], step_env=game.step_env, should_stop=lambda: False)
go()
go()
opener = sess.sent[1][-1]['content'][0]['text']
check('108 level-up: the opener asks for save_summary', 'save a summary with save_summary(text)' in opener, True)
check('108 level-up: the summary is stored for level 1', agent._v6_rules.get(1),
      '- UP clears the level when pressed once\n- the board is static otherwise')
check('108 level-up: counted as via function', agent._v6_stats.get('rules_via_function'), 1)

# === item 111: the notes turn (D1) right after the turn that crossed the budget (D2, user 10-08), then the restart ======
A.RESTART_TOKENS = 500
state = Path(tempfile.mkdtemp()) / 'game_runtime_state.json'
game = FakeGame(state)
log = []
notes_code = ('save_summary("Notes L1:\\n- the red block moves right one cell per RIGHT\\n- the goal is the green pad")\n'
              'action(["RIGHT"])')
sess = FakeSession([('action(["RIGHT"])', 600), (notes_code, 200)])
sess2 = FakeSession([('action(["LEFT"])', 100)])
agent = A.V6Agent(session=sess, logger=lambda k, p: log.append((k, p)), count_tokens=lambda t: max(1, len(t) // 4),
                  model='M', base_url='http://x/v1', provider='vllm', session_factory=lambda: sess2)
go = lambda: agent.analyze(state, game.step, valid_actions=['RIGHT', 'LEFT'], step_env=game.step_env, should_stop=lambda: False)
r1 = go()                                   # 600 tokens on the level: the turn that crossed the budget ends acting
game.__dict__['v6_attempt_actions'] = 3     # an attempt in progress does not delay the restart (D2, user 10-08)
r3 = go()                                   # the notes turn
check('111 D2: no deferral, the notes turn comes next', ([k for k, _ in log].count('v62-restart-deferred'),
      [k for k, _ in log].count('v62-summary-turn')), (0, 1))
opener = sess.sent[1][-1]['content'][0]['text']
check('111 D1: the notes turn asks for notes', 'so the level restarts after this turn' in opener and
      'save_summary(text): a text that begins with the line "Notes L1:"' in opener, True)
check('111 D1: actions are not executed in it', (r3.step_executed, game.step), (False, 1))
check('111 D1: the notes are saved (header stripped)', agent._v62_attempt_notes.get(1),
      ['- the red block moves right one cell per RIGHT\n- the goal is the green pad'])
r4 = go()                                   # the restart: a new conversation; its first observation shows the notes
first = [m for m in sess2.sent[0] if m.get('role') == 'user'][-1]['content'][0]['text']
check('111 restart: new session, RESET executed', (sess.closed, game.history[-2].action), (True, 'RESET'))
check('111 restart: the notes in the first observation', (A.TEXTS['attempt_notes_header'] in first,
      '- the red block moves right one cell per RIGHT' in first), (True, True))
check('111 counters', (agent._v6_stats.get('restart_deferrals'), agent._v6_stats.get('summary_turns'),
                       agent._v6_stats.get('attempt_notes_saved'), agent._v6_stats.get('restarts')), (None, 1, 1, 1))
print('FAILS:', fails)
