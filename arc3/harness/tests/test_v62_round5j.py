"""v6.2 round 5i follow-up tests (391 live, 2026-10-07 ~08:05; V62_BUILD item 98): a recovered malformed python call's
note names the tag its markup lacked and gives the required format (wy30 66, s5i5 26 of 26, wa30 13 recovered calls in
series under the generic note); item 99: a message closed by the harness with a complete tool call gets its result. Synthetic markup only. usage (pod copy): PYTHONPATH=src:. python3 tests/test_v62_round5j.py"""
import json
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
import v62.agent as A  # noqa: E402
from v62.transport import Completion  # noqa: E402
fails = []


def check(name, got, want):
    ok = got == want
    print('%-66s -> %-44r %s' % (name, got if not isinstance(got, str) else got[:44], 'OK' if ok else 'WRONG (want %r)' % (want,)))
    if not ok:
        fails.append(name)


R = 'thinking</think>\n\n'
D = A.call_defect
check('closers missing (wy30, wa30)', D(R + '<tool_call>\n<function=python>\n<parameter=code>\nprint(1)\n<|im_end|>'),
      'it ended without </parameter>, </function> and </tool_call>')
check('opener missing (s5i5)', D(R + '<tool_call>\n<function=python>\nprint(1)\n</parameter>\n</function>\n</tool_call>'),
      'the code was not preceded by <parameter=code>')
check('another parameter name', D(R + '<tool_call>\n<function=python>\n<parameter=command>\nprint(1)\n</parameter>\n'
                                      '</function>\n</tool_call>'), 'its parameter tag was not <parameter=code>')
check('one closer missing', D(R + '<tool_call>\n<function=python>\n<parameter=code>\nprint(1)\n</parameter>\n</function>\n'),
      'it ended without </tool_call>')
check('two closers missing', D(R + '<tool_call>\n<function=python>\n<parameter=code>\nprint(1)\n</parameter>\n'),
      'it ended without </function> and </tool_call>')
check('well formed: none', D(R + '<tool_call>\n<function=python>\n<parameter=code>\nprint(1)\n</parameter>\n</function>\n'
                                 '</tool_call>'), None)
check('markup only inside the reasoning: none', D('<tool_call>\n<function=python>\nprint(1)\n</think>\n\ntext'), None)
check('no text: none', (D(''), D(None)), (None, None))
check('the last call is judged', D(R + '<tool_call>\n<function=python>\n<parameter=code>\nx\n</parameter>\n</function>\n'
                                       '</tool_call>\n<tool_call>\n<function=python>\nprint(2)\n<|im_end|>'),
      'the code was not preceded by <parameter=code>')
FMT = ('Write each python call in the required format: <tool_call>, <function=python>, <parameter=code>, your code, '
       '</parameter>, </function>, </tool_call>.')
check('generic note ends with the format', A.TEXTS['markup_recovered'].endswith(' ' + FMT), True)
check('defect note', A._fill(A.TEXTS['markup_recovered_defect'], what='it ended without </tool_call>', lines='2 lines'),
      'Note: this tool call was malformed: it ended without </tool_call>, so it carried no `code` argument. The harness '
      'recovered 2 lines of code from the message and ran it. ' + FMT)   # round 5l (item 107)


# --- through the loop: the recovered call's tool result starts with the named defect --------------------------------------
class FakeSession:
    def __init__(self, script):
        self.script, self.sent, self.raw = list(script), [], ''
        self.rebased_away, self.blocks, self.evicted = set(), [], []

    def decode(self, ids):
        return self.raw

    def generate(self, external):
        self.sent.append(json.loads(json.dumps(external, default=str)))
        args, raw, tokens = self.script.pop(0)
        self.raw = raw
        msg = dict(role='assistant', content='', reasoning_content='thinking',
                   tool_calls=[dict(id='call-%d' % len(self.sent), type='function',
                                    function=dict(name='python', arguments=args))])
        return Completion(message=msg, finish_reason='stop', usage=dict(prompt_tokens=10, completion_tokens=tokens),
                          token_ids=[1] * tokens, assistant_token_ids=[1] * tokens, pending=False, closed_by=None,
                          persistent={}, seconds=0.0)

    def span_evicted(self, a, b):
        return 0


class FakeGame:
    def __init__(self, state_path):
        self.state_path, self.step = state_path, 0
        self.history = [HistoryEntry(action='', frame=Frame(grid=[[0] * 64 for _ in range(64)], step=0, level=1))]
        self.v6_anim = {}
        write_runtime_state(self.state_path, current_frame=self.history[-1].frame, history=self.history)

    def step_env(self, arguments):
        if arguments.get('query') == 'animation':
            return {'record': None}
        executed = []
        for a in arguments.get('actions') or []:
            self.step += 1
            res = dict(executed=True, action_num=self.step, level=1, gameplay_changed=False, board_changed=False, no_op=True,
                       action_name=a.get('action'), action_display=a.get('action'), valid_actions=['RIGHT', 'LEFT'])
            self.history.append(HistoryEntry(action=a.get('action'), frame=Frame(grid=[[0] * 64 for _ in range(64)],
                                                                                  step=self.step, level=1), result=res))
            executed.append(res)
        write_runtime_state(self.state_path, current_frame=self.history[-1].frame, history=self.history)
        last = dict(executed[-1]) if executed else {}
        last.update(executed=bool(executed), executed_count=len(executed),
                    executed_actions=[e['action_display'] for e in executed], requested_count=len(executed))
        return last


state = Path(tempfile.mkdtemp()) / 'game_runtime_state.json'
game = FakeGame(state)
log = []
sess = FakeSession([
    ('{}', R + '<tool_call>\n<function=python>\n<parameter=code>\nprint(len(history))\n<|im_end|>', 300),
    ('{}', R + '<tool_call>\n<function=python>\nprint(len(history) + 1)\n</parameter>\n</function>\n</tool_call>', 300),
    (json.dumps(dict(code="action(['RIGHT'])")), '', 200),
])
agent = A.V6Agent(session=sess, logger=lambda k, p: log.append((k, p)), count_tokens=lambda t: max(1, len(t) // 4),
                  model='M', base_url='http://x/v1', provider='vllm')
res = agent.analyze(state, 0, valid_actions=['RIGHT', 'LEFT'], step_env=game.step_env, should_stop=lambda: False)
t1, t2 = sess.sent[1][0]['content'], sess.sent[2][0]['content']
check('first recovered call: note names the missing closers', t1.split('\n')[0],
      'Note: this tool call was malformed: it ended without </parameter>, </function> and </tool_call>, so it carried no '
      '`code` argument. The harness recovered 1 line of code from the message and ran it. ' + FMT)   # round 5l (item 107)
check('its code ran', json.loads(t1.split('\n', 1)[1]).get('stdout', '').strip(), '1')
check('second: note names the missing opener', t2.split('\n')[0].startswith('Note: this tool call was malformed: the code was '
                                                                            'not preceded by <parameter=code>,'), True)
check('reparse log carries the defect', [p.get('defect') for k, p in log if k == 'v61-reparse'],
      ['it ended without </parameter>, </function> and </tool_call>', 'the code was not preceded by <parameter=code>'])
check('the turn ends acting', res.step_executed, True)


# --- item 99: a message closed by the harness that already holds a complete tool call gets its call's result -----------------
class ClosingSession(FakeSession):
    def generate(self, external):
        self.sent.append(json.loads(json.dumps(external, default=str)))
        args, closed_by, tokens = self.script.pop(0)
        calls = [dict(id='call-%d' % len(self.sent), type='function', function=dict(name='python', arguments=args))] if args else []
        msg = dict(role='assistant', content='', reasoning_content='thinking', tool_calls=calls)
        return Completion(message=msg, finish_reason='tool_calls' if calls else 'stop', usage=dict(prompt_tokens=10,
                          completion_tokens=tokens), token_ids=[1] * tokens, assistant_token_ids=[1] * tokens, pending=False,
                          closed_by=closed_by, persistent={}, seconds=0.0)


state2 = Path(tempfile.mkdtemp()) / 'game_runtime_state.json'
game2 = FakeGame(state2)
log2 = []
sess2 = ClosingSession([
    (json.dumps(dict(code='print(len(history))')), 'hard_close', 300),     # closed, with a complete non-acting call
    (None, 'hard_close', 300),                                             # closed, no call: the recovery note as before
    (json.dumps(dict(code="action(['RIGHT'])")), None, 200),
])
agent2 = A.V6Agent(session=sess2, logger=lambda k, p: log2.append((k, p)), count_tokens=lambda t: max(1, len(t) // 4),
                   model='M', base_url='http://x/v1', provider='vllm')
res2 = agent2.analyze(state2, 0, valid_actions=['RIGHT', 'LEFT'], step_env=game2.step_env, should_stop=lambda: False)
check('closed with a call: its result follows, no user message', [m['role'] for m in sess2.sent[1]], ['tool'])
check('the call ran', json.loads(sess2.sent[1][0]['content']).get('stdout', '').strip(), '1')
check('closed without a call: the recovery note', ([m['role'] for m in sess2.sent[2]], sess2.sent[2][0]['content'][:40]),
      (['user'], 'The previous assistant turn was closed b'))
check('closes counted, the closed-with-call logged', (agent2._v6_stats['closes'], agent2._v6_stats.get('closed_with_calls'),
      [p['calls'] for k, p in log2 if k == 'v62-closed-with-calls']), (2, 1, [1]))
check('the turn ends acting', res2.step_executed, True)
print('FAILS:', fails)
sys.exit(1 if fails else 0)
