"""Lazy history store checks (spec §5.4; run with ARC3_V6_LAZY_HISTORY=1): the model-facing API is unchanged and old
frames arrive on access. A 300-entry game behind step_env; one Python call reads old and recent frames, transitions,
previous_frame, frame_diff, str(history[i]) (no fetch), iterates every frame, then acts; a second call sees the new
entry. Also: payload size against dfranzen's eager payload, and no state file rewrite of the full history.
usage: ARC3_V6_LAZY_HISTORY=1 PYTHONPATH=src:. python3 tests/test_lazy_history.py"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

assert os.environ.get('ARC3_V6_LAZY_HISTORY') == '1'
os.environ.update({'MULTIMODAL_CONTEXT': 'current_grid', 'MULTIMODAL_UPSCALE': '8', 'ARC3_PERSISTENT_FUNCTIONS': '1',
                   'ARC3_PERSISTENT_FUNCTIONS_SCOPE': 'game', 'ARC3_PERSISTENT_FUNCTIONS_IMPORTS': '1',
                   'ARC3_ANIMATION': '1', 'LOCAL_ANALYZER_TOOL_TIMEOUT': '10', 'ARC3_ACTION_ECHO': '1'})
from inference.agent import runtime_state as rs  # noqa: E402
from inference.agent import tool_agent as ta  # noqa: E402
from inference.agent.runtime_state import Frame, HistoryEntry  # noqa: E402
eager_payload = ta._ascii_history_view_payload
from v6.agent import V6Agent  # noqa: E402
from v6 import lazy_history  # noqa: E402
from v6.transport import Completion  # noqa: E402

assert lazy_history.install()
FAIL = []
N = 300


def check(name, ok, detail=''):
    print('%-72s %s %s' % (name, 'PASS' if ok else 'FAIL', detail))
    if not ok:
        FAIL.append(name)


def grid_for(i):
    g = [[0] * 64 for _ in range(64)]
    g[i % 64][(i // 64) % 64] = 1 + i % 9
    return g


class FakeSession:
    def __init__(self, codes):
        self.codes, self.sent = list(codes), []
        self.rebased_away, self.blocks = set(), []

    def generate(self, external):
        self.sent.append(json.loads(json.dumps(external, default=str)))
        code = self.codes.pop(0)
        msg = dict(role='assistant', content='', reasoning_content='t',
                   tool_calls=[dict(id='c%d' % len(self.sent), type='function',
                                    function=dict(name='python', arguments=json.dumps(dict(code=code))))])
        return Completion(message=msg, finish_reason='stop', usage=dict(prompt_tokens=10, completion_tokens=50),
                          token_ids=[1] * 50, assistant_token_ids=[1] * 50, pending=False, closed_by=None,
                          persistent={}, seconds=0.0)

    def span_evicted(self, a, b):
        return 0


class FakeGame:
    def __init__(self, state_path):
        self.state_path = state_path
        self.history_entries = [HistoryEntry(action='' if i == 0 else 'RIGHT', frame=Frame(grid=grid_for(i), step=i, level=1),
                                             result={} if i == 0 else dict(executed=True, action_num=i))
                                for i in range(N)]
        self.v6_anim = {}
        rs.write_runtime_state(self.state_path, current_frame=self.history_entries[-1].frame, history=self.history_entries)

    def step_env(self, arguments):
        executed = []
        for a in arguments.get('actions') or []:
            i = len(self.history_entries)
            res = dict(executed=True, action_num=i, level=1, gameplay_changed=True, board_changed=True, no_op=False,
                       action_name=a.get('action'), action_display=a.get('action'), valid_actions=['RIGHT'])
            self.history_entries.append(HistoryEntry(action=a.get('action'), frame=Frame(grid=grid_for(i), step=i, level=1),
                                                     result=res))
            executed.append(res)
        rs.write_runtime_state(self.state_path, current_frame=self.history_entries[-1].frame, history=self.history_entries)
        last = dict(executed[-1]) if executed else {}
        last.update(executed=bool(executed), executed_count=len(executed),
                    executed_actions=[e['action_display'] for e in executed], requested_count=len(executed))
        return last


CODE1 = '''
h = history
print('LEN', len(h))
print('OLD', h[5].frame._grid[5][0], h[130].frame._grid[130 % 64][2])
print('NEW', h[-1].frame._grid[(len(h) - 1) % 64][((len(h) - 1) // 64) % 64])
print('PREV', previous_frame is not None and previous_frame._grid == h[-2].frame._grid)
print('TRANS', len(transitions), transitions[9].before_frame._grid == h[9].frame._grid, transitions[9].after_frame._grid == h[10].frame._grid)
print('STR', str(h[3]))
d = frame_diff(h[1].frame, h[2].frame)
print('DIFF', d.get('changed_cell_count'))
print('ALL', sum(1 for e in h if e.frame is not None))
print('ANIM', h[-1].animation_frames)
'''
CODE2 = "action(['RIGHT'])"
CODE3 = "print('LEN2', len(history), history[-1].frame._grid[%d][%d])" % (N % 64, (N // 64) % 64)
CODE4 = "action(['RIGHT'])"


def main():
    tmp = Path(tempfile.mkdtemp())
    state = tmp / 'game_tool_runtime_state.json'
    game = FakeGame(state)
    check('state stub file small (no full-history rewrite)', state.stat().st_size < 50_000, state.stat().st_size)
    lazy = len(json.dumps(lazy_history.lazy_history_payload(game.history_entries)))
    eager = len(json.dumps(eager_payload(game.history_entries)))
    check('sandbox payload at %d entries: lazy %.0f KB vs eager %.0f KB' % (N, lazy / 1e3, eager / 1e3), lazy * 10 < eager)
    sess = FakeSession([CODE1, CODE2, CODE3, CODE4])
    agent = V6Agent(session=sess, logger=lambda k, p: None, count_tokens=lambda t: max(1, len(t) // 4), model='M',
                    base_url='http://x/v1', provider='vllm')
    w0 = lazy_history.STATS['window_requests']
    t0 = time.time()
    agent.analyze(state, N, valid_actions=['RIGHT'], step_env=game.step_env, should_stop=lambda: False)
    dt = time.time() - t0
    out = json.loads(sess.sent[1][0]['content']).get('stdout', '')
    lines = dict(l.split(' ', 1) for l in out.strip().split('\n') if ' ' in l)
    check('len(history)', lines.get('LEN') == str(N), lines.get('LEN'))
    check('old frames fetched lazily and correct', lines.get('OLD') == '%d %d' % (1 + 5 % 9, 1 + 130 % 9), lines.get('OLD'))
    check('newest frame (eager tail)', lines.get('NEW') == str(1 + (N - 1) % 9), lines.get('NEW'))
    check('previous_frame is history[-2].frame', lines.get('PREV') == 'True', lines.get('PREV'))
    check('transitions lazy and aligned', lines.get('TRANS') == '%d True True' % (N - 1), lines.get('TRANS'))
    check('str(history[i]) from metadata', (lines.get('STR') or '').startswith("AsciiHistoryEntryView(action='RIGHT', frame=AsciiFrameView(level=1, step=3, shape=64x64)"), lines.get('STR'))
    check('frame_diff on lazy frames', lines.get('DIFF') == '2', lines.get('DIFF'))
    check('iterating every frame works', lines.get('ALL') == str(N), lines.get('ALL'))
    check('animation_frames still available', lines.get('ANIM') == '[]', lines.get('ANIM'))
    reqs = lazy_history.STATS['window_requests'] - w0
    check('window requests ~ N/9 for a full forward pass (%d)' % reqs, reqs <= N // 9 + 10, reqs)
    agent.analyze(state, N + 1, valid_actions=['RIGHT'], step_env=game.step_env, should_stop=lambda: False)
    out3 = json.loads(sess.sent[3][0]['content']).get('stdout', '')
    check('second call sees the new entry', out3.strip() == 'LEN2 %d %d' % (N + 1, 1 + N % 9), out3.strip())
    print('first call (with one full pass over %d frames): %.2f s; stats %s' % (N, dt, lazy_history.STATS))
    print('TEST_LAZY_HISTORY', 'PASS' if not FAIL else 'FAIL ' + ', '.join(FAIL))
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
