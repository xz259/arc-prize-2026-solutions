"""v6.2 round 5i loop test (user 2026-10-07 ~04:50: the soft-cap cues take priority over the yield; one automatic message
at a time; V62_BUILD item 96): scripted completions through the real v6.2 turn loop (fake session and game, the real
sandbox). A reply that carried a cue (the nudge, the closing cue, a repeat cue) or got a breaker note does not yield even
past YIELD_TOKENS: its tool result goes back alone and the yield waits for the next non-acting call.
usage (pod copy): PYTHONPATH=src:. python3 tests/test_v62_round5i_loop.py"""
import json
import os
os.environ.setdefault('ARC3_V62_YIELD_HELD_BY', 'soft_nudge,soft_budget,repeat_cue')   # round 5l: the 5i setting
import os
import sys
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
from v62.agent import V6Agent  # noqa: E402
import v62.agent as A  # noqa: E402
from v62.transport import Completion  # noqa: E402

FAIL = []


def check(name, ok, detail=''):
    print('%-70s %s %s' % (name, 'PASS' if ok else 'FAIL', detail))
    if not ok:
        FAIL.append(name)


def grid(fill=0, mark=None):
    g = [[fill] * 64 for _ in range(64)]
    if mark:
        rr, c, v = mark
        g[rr][c] = v
    return g


class FakeSession:
    def __init__(self, script):
        self.script, self.sent = list(script), []
        self.rebased_away, self.blocks, self.evicted = set(), [], []

    def generate(self, external):
        self.sent.append(json.loads(json.dumps(external, default=str)))
        code, tokens, cues = self.script.pop(0)
        msg = dict(role='assistant', content='', reasoning_content='thinking',
                   tool_calls=[dict(id='call-%d' % len(self.sent), type='function',
                                    function=dict(name='python', arguments=json.dumps(dict(code=code))))])
        return Completion(message=msg, finish_reason='stop', usage=dict(prompt_tokens=10, completion_tokens=tokens),
                          token_ids=[1] * tokens, assistant_token_ids=[1] * tokens, pending=False, closed_by=None,
                          persistent={}, seconds=0.0, cues=tuple(cues))

    def span_evicted(self, a, b):
        return 0


class FakeGame:
    def __init__(self, state_path):
        self.state_path, self.col, self.step = state_path, 10, 0
        self.history = [HistoryEntry(action='', frame=Frame(grid=grid(0, (30, self.col, 9)), step=0, level=1))]
        self.v6_anim = {}
        self.write()

    def write(self):
        write_runtime_state(self.state_path, current_frame=self.history[-1].frame, history=self.history)

    def step_env(self, arguments):
        if arguments.get('query') == 'animation':
            return {'record': None}
        executed = []
        for a in arguments.get('actions') or []:
            name = a.get('action')
            before = self.history[-1].frame.grid
            if name == 'RIGHT':
                self.col += 1
            self.step += 1
            g = grid(0, (30, self.col, 9))
            res = dict(executed=True, action_num=self.step, level=1, gameplay_changed=before != g, board_changed=before != g,
                       no_op=before == g, action_name=name, action_display=name, valid_actions=['RIGHT', 'LEFT'])
            self.history.append(HistoryEntry(action=name, frame=Frame(grid=g, step=self.step, level=1), result=res))
            executed.append(res)
        self.write()
        last = dict(executed[-1]) if executed else {}
        last.update(executed=bool(executed), executed_count=len(executed),
                    executed_actions=[e['action_display'] for e in executed], requested_count=len(executed))
        return last


def roles(msgs):
    return [m['role'] for m in msgs]


def main():
    check('held-by set', A.YIELD_HELD_BY == frozenset({'soft_nudge', 'soft_budget', 'repeat_cue'}), A.YIELD_HELD_BY)
    tmp = Path(tempfile.mkdtemp())
    state = tmp / 'game_runtime_state.json'
    game = FakeGame(state)
    log = []
    script = [
        ("action(['RIGHT'])", 300, ()),                        # turn 1: acts
        ("print(len(history))", 5000, ('soft_budget',)),       # turn 2: past 4,096 with the closing cue -> held
        ("print(len(history) + 1)", 300, ()),                  # no cue -> the yield
        ("print(len(history) + 2)", 4500, ('soft_nudge',)),    # resumed turn: past 4,096 with the nudge -> held
        ("print(len(history) + 3)", 4500, ('soft_nudge', 'soft_budget')),   # both cues -> held
        ("action(['RIGHT'])", 200, ()),                        # acts
        ("print(undefined_name_x)", 2500, ()),                 # turn 3: an error
        ("print(undefined_name_x)", 2500, ()),                 # the same error: breaker note, 5,000 tokens -> held
        ("print(len(history) + 4)", 200, ('repeat_cue',)),     # a repeat cue -> held
        ("print(len(history) + 5)", 200, ('hard_close',)),     # other kinds do not hold -> the yield
        ("action(['LEFT'])", 200, ()),
    ]
    sess = FakeSession(script)
    agent = V6Agent(session=sess, logger=lambda k, p: log.append((k, p)),
                    count_tokens=lambda t: max(1, len(t) // 4), model='M', base_url='http://x/v1', provider='vllm')
    kw = dict(valid_actions=['RIGHT', 'LEFT'], step_env=game.step_env, should_stop=lambda: False)
    r1 = agent.analyze(state, 0, **kw)
    check('turn 1 acts', r1.step_executed, r1)
    r2 = agent.analyze(state, 1, **kw)
    check('turn 2: after the cued reply the tool result goes back alone', roles(sess.sent[2]) == ['tool'], roles(sess.sent[2]))
    check('turn 2: the next non-acting reply yields', r2.yielded_control and not r2.step_executed, r2)
    held = [p for k, p in log if k == 'v62-yield-held']
    check('held once, by the closing cue, at 5,000 turn tokens', [(p['by'], p['turn_tokens']) for p in held] ==
          [(['soft_budget'], 5000)], [(p['by'], p['turn_tokens']) for p in held])
    r3 = agent.analyze(state, 1, **kw)
    check('resumption opener after the yield', roles(sess.sent[3]) == ['tool', 'user'] and sess.sent[3][1]['content'][0][
        'text'].startswith('Your last Python call executed no actions.'), roles(sess.sent[3]))
    check('nudge and both cues hold: tool results alone', (roles(sess.sent[4]), roles(sess.sent[5])) == (['tool'], ['tool']),
          (roles(sess.sent[4]), roles(sess.sent[5])))
    check('resumed turn acts', r3.step_executed, r3)
    r4 = agent.analyze(state, 2, **kw)
    sent_b = sess.sent[8] if len(sess.sent) > 8 else []
    check('breaker note held the yield: tool result + note, no resumption', roles(sent_b) == ['tool', 'user'] and
          sent_b[1]['content'].startswith('The last 2 python calls returned the same error'), roles(sent_b))
    check('repeat cue held: tool result alone', roles(sess.sent[9]) == ['tool'] if len(sess.sent) > 9 else False)
    check('a hard-close kind does not hold: the yield', r4.yielded_control and not r4.step_executed, r4)
    held = [p['by'] for k, p in log if k == 'v62-yield-held']
    check('held by: cue, nudge, both, breaker note, repeat cue', held == [['soft_budget'], ['soft_nudge'],
          ['soft_budget', 'soft_nudge'], ['breaker_note'], ['repeat_cue']], held)
    stats = (agent._v6_stats['yields'], agent._v6_stats.get('yields_held'))
    check('stats: yields 2, held 5', stats == (2, 5), stats)
    r5 = agent.analyze(state, 2, **kw)
    check('after the second yield: resumption, then acts', roles(sess.sent[10]) == ['tool', 'user'] and r5.step_executed,
          roles(sess.sent[10]))
    check('all scripted replies used', (len(sess.sent), sess.script) == (11, []), (len(sess.sent), len(sess.script)))
    print('TEST_V62_ROUND5I_LOOP', 'PASS' if not FAIL else 'FAIL ' + ', '.join(FAIL))
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
