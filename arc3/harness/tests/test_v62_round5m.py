"""v6.2 round 5m (user 2026-10-08): the notes turn before a position restart, through agent.analyze with the real sandbox.
At POSITION_RESTART logical tokens, a conversation with at least POSITION_NOTES_MIN tokens on the level first gets the notes
turn (the position wording; actions not executed; notes saved), then the restart (RESET, a new conversation whose first
observation shows the notes). Below the minimum, or with a session error pending, the restart is immediate as before.
usage (pod copy): PYTHONPATH=src:. python3 tests/test_v62_round5m.py"""
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
        self.last_receipt = {}

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


A.RESTART_TOKENS = 10 ** 9                  # no token-budget restart in these tests
NOTES = ('save_summary("Notes L1:\\n- the red block moves right one cell per RIGHT\\n- the goal is the green pad")\n'
         'action(["RIGHT"])')


def run(min_tokens, force_error=False):
    A.POSITION_NOTES_MIN = min_tokens
    state = Path(tempfile.mkdtemp()) / 'game_runtime_state.json'
    game = FakeGame(state)
    log = []
    sess = FakeSession([('action(["RIGHT"])', 600), (NOTES, 200)])
    sess2 = FakeSession([('action(["LEFT"])', 100)])
    agent = A.V6Agent(session=sess, logger=lambda k, p: log.append((k, p)), count_tokens=lambda t: max(1, len(t) // 4),
                      model='M', base_url='http://x/v1', provider='vllm', session_factory=lambda: sess2)
    go = lambda: agent.analyze(state, game.step, valid_actions=['RIGHT', 'LEFT'], step_env=game.step_env,
                               should_stop=lambda: False)
    go()                                                        # 600 tokens on the level
    sess.last_receipt = {'logical_tokens': A.POSITION_RESTART}  # the session has reached the position limit
    if force_error:
        agent._v62_force_restart = True                         # a session error is pending as well
    return agent, game, log, sess, sess2, go


# === notes first: >= POSITION_NOTES_MIN tokens on the level ===========================================================
agent, game, log, sess, sess2, go = run(500)
r2 = go()                                                       # the notes turn
opener = sess.sent[1][-1]['content'][0]['text']
check('5m notes turn before the position restart', [p.get('reason') for k, p in log if k == 'v62-summary-turn'], ['position'])
check('5m the position wording', ('reached its length limit, so level 1 restarts after this turn' in opener,
                                 'This conversation has generated' in opener), (True, False))
check('5m actions are not executed in it', (r2.step_executed, game.step), (False, 1))
check('5m the notes are saved', agent._v62_attempt_notes.get(1),
      ['- the red block moves right one cell per RIGHT\n- the goal is the green pad'])
check('5m no restart yet', (sess.closed, agent._v6_stats.get('position_restarts')), (False, None))
go()                                                            # the restart
first = [m for m in sess2.sent[0] if m.get('role') == 'user'][-1]['content'][0]['text']
check('5m restart: new session, RESET executed', (sess.closed, game.history[-2].action), (True, 'RESET'))
check('5m restart: the notes in the first observation', (A.TEXTS['attempt_notes_header'] in first,
      '- the red block moves right one cell per RIGHT' in first), (True, True))
check('5m counters', tuple(agent._v6_stats.get(k) for k in ('position_restarts', 'position_notes_turns', 'summary_turns',
                                                            'attempt_notes_saved', 'restarts')), (1, 1, 1, 1, 1))

# === below the minimum: the restart is immediate =========================================================================
agent, game, log, sess, sess2, go = run(10 ** 9)
go()
first = [m for m in sess2.sent[0] if m.get('role') == 'user'][-1]['content'][0]['text']
check('5m below the minimum: no notes turn, immediate restart', ([k for k, _ in log].count('v62-summary-turn'), sess.closed,
                                                               game.history[-2].action), (0, True, 'RESET'))
check('5m below the minimum: no notes lines', (A.TEXTS['attempt_notes_header'] in first,
                                               A.TEXTS['attempt_notes_none'] in first), (False, False))

# === a session error pending: the restart is immediate ===================================================================
agent, game, log, sess, sess2, go = run(500, force_error=True)
go()
check('5m session error: no notes turn, immediate restart', ([k for k, _ in log].count('v62-summary-turn'), sess.closed),
      (0, True))
print('FAILS:', fails)
