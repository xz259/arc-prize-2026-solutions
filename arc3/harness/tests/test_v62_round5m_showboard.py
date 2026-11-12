"""v6.2 round 5m item 124 (user 2026-10-08: "We instruct the model not to print the full board. But what if it wants to see
the simulated effects on the board?" ... "Let's add it."): show_board(cells, label) through agent.analyze with the real
sandbox. A board the code built is added to the call's tool result as an image like the board image (8x, 512 px for 64 x
64), captioned "Board image j: show_board" (the server's 'tool' image kind); a repeat in the same call is named; at most 4
per call, shared with converted printed boards (a printed copy of a shown board is named); invalid input raises in the
snippet; the system prompt carries the two lines once; ARC3_V62_SHOW_BOARD=0 removes the function and the lines.
usage (pod copy): PYTHONPATH=src:. python3 tests/test_v62_round5m_showboard.py"""
import base64
import json
import os
import struct
import subprocess
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
from v62 import sandbox_ext  # noqa: E402
from v62.transport import Completion  # noqa: E402
fails = []


def check(name, got, want):
    ok = got == want
    print('%-72s -> %-40r %s' % (name, got if not isinstance(got, str) else got[:40], 'OK' if ok else 'WRONG (want %r)' % (want,)))
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
        last = dict(executed[-1]) if executed else {}
        last.update(executed=bool(executed), executed_count=len(executed),
                    executed_actions=[e['action_display'] for e in executed], requested_count=len(executed))
        return last


A.RESTART_TOKENS = 10 ** 9


def turn(code):
    """One step: the snippet `code` (no action), then an action; returns the tool message of the snippet, the snippet's
    stdout / error and the agent's log."""
    state = Path(tempfile.mkdtemp()) / 'game_runtime_state.json'
    game = FakeGame(state)
    log = []
    sess = FakeSession([(code, 100), ('action(["RIGHT"])', 50)])
    agent = A.V6Agent(session=sess, logger=lambda k, p: log.append((k, p)), count_tokens=lambda t: max(1, len(t) // 4),
                      model='M', base_url='http://x/v1', provider='vllm', session_factory=lambda: None)
    agent.analyze(state, game.step, valid_actions=['RIGHT', 'LEFT'], step_env=game.step_env, should_stop=lambda: False)
    tool = [m for m in sess.sent[1] if m.get('role') == 'tool'][-1]
    parts = tool['content'] if isinstance(tool['content'], list) else [dict(type='text', text=tool['content'])]
    body = json.loads(parts[0]['text'])
    return parts, body, log


def png_size(part):
    raw = base64.b64decode(part['image_url']['url'].split(',', 1)[1])
    return struct.unpack('>II', raw[16:24])


def images(parts):
    return [p for p in parts if p.get('type') == 'image_url']


def captions(parts):
    """The text part right before each image (the server labels an image 'tool' when "Board image" precedes it)."""
    return [parts[i - 1].get('text', '') for i, p in enumerate(parts) if p.get('type') == 'image_url' and i]


SIM = ('B = [list(row) for row in grid_cells(current_frame)]\nB[5][7] = "R"\nB[6][7] = "R"\n'
       'show_board(B, "after RIGHT")\nshow_board(B)\nprint("done")\n')
parts, body, log = turn(SIM)
out = str(body.get('stdout'))
check('124 a simulated board is shown as one image', len(images(parts)), 1)
check('  ... at the board image scale (512 x 512 px for 64 x 64)', png_size(images(parts)[0]), (512, 512))
check('  ... its caption names it, with the label', captions(parts)[0], '[Board image 1: show_board, "after RIGHT", 64x64.]')
check('  ... the caption carries "Board image" (server image kind tool)', all('Board image' in c for c in captions(parts)), True)
check('  ... show_board printed its caption in the output', '[Board image 1: show_board, "after RIGHT", 64x64.]' in out, True)
check('  ... the repeat in the same call is named, not shown', '[Same board as image 1 above.]' in out, True)
check('  ... the snippet went on', 'done' in out and not body.get('error'), True)
rec = [p for k, p in log if k == 'tool-output'][0]
check('  ... the tool-output record counts it as shown, not converted', (rec.get('shown_boards'), rec.get('converted_boards')), (1, 0))

FIVE = ''.join('B%d = [list(row) for row in grid_cells(current_frame)]\nB%d[0][%d] = "R"\nshow_board(B%d, "v%d")\n'
               % (i, i, i, i, i) for i in range(5))
parts, body, log = turn(FIVE)
out = str(body.get('stdout'))
check('124 five boards: four images', len(images(parts)), 4)
check('  ... the fifth is not shown, and says why', '[Board not shown: this call already has 4 board images.]' in out, True)

SHARED = ('show_board(current_frame)\nB = [list(row) for row in grid_cells(current_frame)]\nB[1][1] = "R"\nshow_board(B)\n'
          'C = [list(row) for row in grid_cells(current_frame)]\nC[2][2] = "R"\nshow_board(C)\n'
          'print(current_frame.ascii)\nprint("next")\nD = [list(row) for row in grid_cells(current_frame)]\nD[3][3] = "R"\n'
          'print("\\n".join("".join(r) for r in D))\nprint("next")\nE = [list(row) for row in grid_cells(current_frame)]\n'
          'E[4][4] = "R"\nprint("\\n".join("".join(r) for r in E))\n')   # separate runs: consecutive rows are one board
parts, body, log = turn(SHARED)
out = str(body.get('stdout'))
check('124 printed boards share the budget: 3 shown + 1 converted', len(images(parts)), 4)
check('  ... a printed copy of a shown board is named', '[Same board as image 1 above.]' in out, True)
check('  ... the printed board over the budget is counted as omitted', 'more printed boards omitted' in json.dumps(body), True)

parts, body, log = turn('show_board([[1, 2], [3]])\n')
check('124 invalid input raises in the snippet', 'show_board(cells) takes' in str(body.get('error')), True)
check('  ... and shows nothing', len(images(parts)), 0)
parts, body, log = turn('show_board([[0] * 8 for _ in range(8)], "ints")\n')
check('124 colour indices 0-15 and small grids are accepted', (len(images(parts)), '8x8' in str(body.get('stdout'))), (1, True))

S = A.TEXTS['system']
check('124 / v6.3 G the no-print rule points to show_board (once, with the label and the no-repeat rule)',
      S.count('To look at a whole board your code built, call `show_board(cells, label=None)` (a frame, rows of symbols or ASCII '
              'text, as `grid_cells` accepts): it is added to this call\'s result as an image captioned with `label`, like the '
              'board image; at most 4 per call, and a board already shown in the call is not repeated.'), 1)
check('124 / v6.3 G the separate helper line is gone (it split the starter-helper list)', S.count('- `show_board(cells, label=None)` adds a board'), 0)
check('v6.3 G objects.py follows grid.py in the starter list', S.find('  - grid.py:') < S.find('  - objects.py:') and
      'show_board' not in S[S.find('  - grid.py:'):S.find('  - objects.py:')], True)
check('124 the reminder is unchanged', 'never print a whole board' in A.TEXTS['reminder'] and 'show_board' not in A.TEXTS['reminder'], True)
check('124 show_board is a provided name (NameError hints)', 'show_board' in A.PROVIDED_NAMES, True)

OFF = r'''
import os, sys
os.environ.update(ARC3_V62_SHOW_BOARD="0", ARC3_V61_TEXTS="1", ARC3_V6_A13="1", ARC3_V61_SUMMARY="1",
                  ARC3_NEW_CHANGED_PROMPTS="1", ARC3_ACTION_INFO="1", ARC3_V6_A3="1", ARC3_PERSISTENT_FUNCTIONS="1",
                  ARC3_PERSISTENT_FUNCTIONS_SCOPE="game", ARC3_ANIMATION="1", LOCAL_ANALYZER_TOOL_TIMEOUT="10")
import v62.patches, v62.retention as r
r.install()
import v62.agent as A
from v62 import sandbox_ext as se
from inference.agent import tool_agent as ta
class Dummy:
    def _v6_library_payload(self): return []
    def _v61_archive_payload(self): return []
se.CURRENT.agent = Dummy()
out = ta.run_sandboxed_python(code="show_board([[0]])", timeout_seconds=10, initial_state={}, action_handler=lambda *a, **k: {},
                              kept_functions=[], retain_imports=True, repair_hints=True)
print(int("show_board" in A.TEXTS["system"]), int(se.SHOW_BOARD), int("show_board" in A.PROVIDED_NAMES),
      int("name 'show_board' is not defined" in str(out.get("error"))))
'''
res = subprocess.run([sys.executable, '-c', OFF], capture_output=True, text=True, env=dict(os.environ, ARC3_V62_SHOW_BOARD='0'))
check('124 off switch: no prompt lines, no flag, no name, NameError in the sandbox', res.stdout.strip().split('\n')[-1],
      '0 0 0 1')
print('FAILS:', fails)
print('test_v62_round5m_showboard', 'PASS' if not fails else 'FAIL')
