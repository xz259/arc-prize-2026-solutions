"""Local simulation of the v6 turn loop (no server, no arcengine): scripted completions through a fake persistent
session, a fake game behind step_env, and the real sandbox subprocess (dfranzen's, with the v6 extensions).
Checks: first opener (system + opener, one image), acting call ends the turn, the tool result is sent with the next
opener, N1 nudge, yield after >= 2,048 turn tokens with a resumption opener, length pause -> empty continuation,
programs()/read_code()/seeds/history[i].animation_frames in the sandbox, the program-list reminder after a save.
usage: PYTHONPATH=src:. python3 tests/test_agent_loop.py"""
import json
import os
import sys
import tempfile
from pathlib import Path

os.environ.update({'MULTIMODAL_CONTEXT': 'current_grid', 'MULTIMODAL_UPSCALE': '8', 'ARC3_PERSISTENT_FUNCTIONS': '1',
                   'ARC3_PERSISTENT_FUNCTIONS_SCOPE': 'game', 'ARC3_PERSISTENT_FUNCTIONS_IMPORTS': '1',
                   'ARC3_ANIMATION': '1', 'LOCAL_ANALYZER_TOOL_TIMEOUT': '10', 'ARC3_ACTION_ECHO': '1',
                   'ARC3_V6_YIELD_TOKENS': '2048'})   # the loop test exercises the yield (default off since 2026-10-03)

from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state  # noqa: E402
from v6.agent import V6Agent  # noqa: E402
from v6.transport import Completion  # noqa: E402

FAIL = []


def check(name, ok, detail=''):
    print('%-70s %s %s' % (name, 'PASS' if ok else 'FAIL', detail))
    if not ok:
        FAIL.append(name)


def grid(fill=0, mark=None):
    g = [[fill] * 64 for _ in range(64)]
    if mark:
        r, c, v = mark
        g[r][c] = v
    return g


class FakeSession:
    def __init__(self, script):
        self.script = list(script)
        self.sent = []
        self.rebased_away, self.blocks, self.evicted = set(), [], []

    def generate(self, external):
        self.sent.append(json.loads(json.dumps(external, default=str)))
        kind, data = self.script.pop(0)
        if kind == 'pause':
            return Completion(message={}, finish_reason='length', usage=dict(prompt_tokens=10, completion_tokens=data),
                              token_ids=[1] * data, assistant_token_ids=[1] * data, pending=True, closed_by=None,
                              persistent={}, seconds=0.0)
        if kind == 'text':
            msg = dict(role='assistant', content=data[0], reasoning_content=data[1])
            return Completion(message=msg, finish_reason='stop', usage=dict(prompt_tokens=10, completion_tokens=data[2]),
                              token_ids=[1] * data[2], assistant_token_ids=[1] * data[2], pending=False, closed_by=None,
                              persistent={}, seconds=0.0)
        code, tokens = data
        msg = dict(role='assistant', content='', reasoning_content='thinking',
                   tool_calls=[dict(id='call-%d' % len(self.sent), type='function',
                                    function=dict(name='python', arguments=json.dumps(dict(code=code))))])
        return Completion(message=msg, finish_reason='stop', usage=dict(prompt_tokens=10, completion_tokens=tokens),
                          token_ids=[1] * tokens, assistant_token_ids=[1] * tokens, pending=False, closed_by=None,
                          persistent={}, seconds=0.0)

    def span_evicted(self, a, b):
        return 0


class FakeGame:
    """Minimal step_env: RIGHT moves a marker; the result payload mimics the solver's."""

    def __init__(self, state_path):
        self.state_path = state_path
        self.col = 10
        self.step = 0
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


def main():
    tmp = Path(tempfile.mkdtemp())
    state = tmp / 'game_runtime_state.json'
    game = FakeGame(state)
    log = []
    script = [
        ('call', ("print(current_frame.shape, len(grid_cells(current_frame)))\naction(['RIGHT'])", 900)),
        ('text', ('I will now think about the board.', 'Let me call python to move right.', 300)),
        ('call', ('def probe_right(n):\n    """Move right n times."""\n    return n\nprint(programs(), len(history))', 2100)),
        ('pause', 8000),
        ('call', ("print(history[-1].animation_frames, read_code('probe_right')[:20])\naction(['RIGHT','RIGHT'])", 500)),
    ]
    sess = FakeSession(script)
    agent = V6Agent(session=sess, logger=lambda k, p: log.append((k, p)),
                    count_tokens=lambda t: max(1, len(t) // 4), model='M', base_url='http://x/v1', provider='vllm')
    r1 = agent.analyze(state, 0, valid_actions=['RIGHT', 'LEFT'], step_env=game.step_env, should_stop=lambda: False)
    s0 = sess.sent[0]
    check('first request: system + opener', [m['role'] for m in s0] == ['system', 'user'], [m['role'] for m in s0])
    imgs = [p for p in s0[1]['content'] if p.get('type') == 'image_url']
    check('first opener: one board image (no change image)', len(imgs) == 1, len(imgs))
    check('first opener text', s0[1]['content'][0]['text'].startswith('The game has started: level 1, step 0.'),
          s0[1]['content'][0]['text'][:60])
    check('acting call ends the turn', r1.step_executed and len(sess.sent) == 1)
    r2 = agent.analyze(state, 1, valid_actions=['RIGHT', 'LEFT'], step_env=game.step_env, should_stop=lambda: False)
    s1 = sess.sent[1]
    check('second append: tool result + opener', [m['role'] for m in s1] == ['tool', 'user'], [m['role'] for m in s1])
    tool = json.loads(s1[0]['content'])
    check('tool result carries the action result', "'executed': True" in tool.get('stdout', ''), tool.get('stdout', '')[:120])
    check('seeds available (grid_cells)', '64' in tool.get('stdout', ''), tool.get('stdout', '')[:60])
    op = s1[1]['content'][0]['text']
    check('opener summary', op.startswith('Your previous call executed 1 action(s): RIGHT'), op[:80])
    check('opener same-level status + step line', 'You are still on level 1.' in op and 'Step 1, level 1.' in op)
    imgs = [p for p in s1[1]['content'] if p.get('type') == 'image_url']
    check('opener: board + change image', len(imgs) == 2, len(imgs))
    s2 = sess.sent[2]
    check('N1 nudge after a reply without a tool call', s2[0]['role'] == 'user' and s2[0]['content'].startswith(
        'You did not call the python tool'), s2[0]['content'][:60])
    check('N1 quotes the plan line', 'Your reasoning ended with: "Let me call python to move right."' in s2[0]['content'])
    check('turn 2 yields after >= 2,048 tokens without acting', r2.yielded_control and not r2.step_executed,
          (r2.yielded_control, r2.step_executed))
    r3 = agent.analyze(state, 1, valid_actions=['RIGHT', 'LEFT'], step_env=game.step_env, should_stop=lambda: False)
    s3 = sess.sent[3]
    check('yield: tool result + resumption opener', [m['role'] for m in s3] == ['tool', 'user'], [m['role'] for m in s3])
    t3 = json.loads(s3[0]['content'])
    check('programs() empty within the defining call (library updates after the call)', '[]' in t3.get('stdout', ''),
          t3.get('stdout', '')[:80])
    check('retention note in the programs field', 'probe_right' in json.dumps(t3.get('programs', '')), t3.get('programs'))
    check('resumption opener text', s3[1]['content'][0]['text'].startswith('Your last Python call executed no actions.'))
    check('program list in a later opener or reminder pending', agent._v6_library_dirty or agent._v6_reminder is not None)
    s4 = sess.sent[4]
    check('length pause -> empty continuation', s4 == [], s4)
    check('resumed turn ends acting', r3.step_executed, r3)
    r4_out = []
    sess.script.append(('call', ("print(history[-1].animation_frames, read_code('probe_right')[:20])", 100)))
    sess.script.append(('call', ("action(['LEFT'])", 100)))
    r4 = agent.analyze(state, 4, valid_actions=['RIGHT', 'LEFT'], step_env=game.step_env, should_stop=lambda: False)
    t5 = json.loads(sess.sent[-1][0]['content'])
    check('read_code and animation_frames in a later call', 'def probe_right' in t5.get('stdout', '') and
          t5.get('stdout', '').startswith('[]'), t5.get('stdout', '')[:100])
    op4 = sess.sent[-2][1]['content'][0]['text']
    check('program list block in the next regular opener', 'Your program library (1 programs' in op4 and
          'probe_right(n): Move right n times.' in op4, op4[:300])
    print('stats', agent._v6_stats)
    check('yield counted', agent._v6_stats['yields'] == 1, agent._v6_stats['yields'])
    print('TEST_AGENT_LOOP', 'PASS' if not FAIL else 'FAIL ' + ', '.join(FAIL))
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
