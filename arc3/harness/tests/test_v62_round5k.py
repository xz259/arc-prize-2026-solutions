"""v6.2 round 5k tests (392b, user 2026-10-07 ~18:50: python time per call grew with the game's history; V62_BUILD item
100): paged history records. A game of N entries behind step_env, with a hidden range as after a run-B restart and level
/ restart markers in the results; the same probe code runs with ARC3_V6_HISTORY_PAGED off (the previous payload) and on,
and its output must match: lengths, actions, results, old and recent frames, str(entry), slices across the hidden range,
transitions, previous_frame, level / attempt start frames, a full scan of frames and records, animation_frames; an action
inside a call extends history; payload size and call time at 7,000 entries for both.
usage (pod copy): PYTHONPATH=src:. python3 tests/test_v62_round5k.py"""
import json
import os
import sys
import tempfile
import time
from pathlib import Path

os.environ.update({'MULTIMODAL_CONTEXT': 'current_grid', 'MULTIMODAL_UPSCALE': '8', 'ARC3_PERSISTENT_FUNCTIONS': '1',
                   'ARC3_PERSISTENT_FUNCTIONS_SCOPE': 'game', 'ARC3_PERSISTENT_FUNCTIONS_IMPORTS': '1',
                   'ARC3_ANIMATION': '1', 'LOCAL_ANALYZER_TOOL_TIMEOUT': '30', 'ARC3_ACTION_ECHO': '1',
                   'ARC3_V6_YIELD_TOKENS': '4096', 'ARC3_V6_LAZY_HISTORY': '1'})
for k, v in dict(ARC3_V61_TEXTS='1', ARC3_V6_A13='1', ARC3_V61_SUMMARY='1', ARC3_NEW_CHANGED_PROMPTS='1',
                 ARC3_ACTION_INFO='1', ARC3_REPEAT_STATE_GUARD='0', ARC3_V61_RESTART_TOKENS='150000', ARC3_V6_A3='1',
                 ARC3_GUARDS_FROM_LEVEL='2', ARC3_BATCH_NOOP_BLOCK='1', ARC3_STALE_STATE_BLOCK='1').items():
    os.environ.setdefault(k, v)
import v62.patches  # noqa: E402,F401
import v62.retention as r  # noqa: E402
r.install()
from inference.agent import runtime_state as rs  # noqa: E402
from inference.agent import tool_agent as ta  # noqa: E402
from inference.agent.runtime_state import Frame, HistoryEntry  # noqa: E402
import v62.agent as A  # noqa: E402
from v62 import lazy_history  # noqa: E402
from v62.transport import Completion  # noqa: E402

assert lazy_history.install()
A.install_history_scope()
FAIL = []


def check(name, ok, detail=''):
    print('%-74s %s %s' % (name, 'PASS' if ok else 'FAIL', '' if ok else str(detail)[:300]))
    if not ok:
        FAIL.append(name)


def grid_for(i):
    g = [[0] * 64 for _ in range(64)]
    g[i % 64][(i // 64) % 64] = 1 + i % 9
    return g


def result_for(i):
    if i == 0:
        return {}
    r = dict(executed=True, action_num=i, level=1 + i // 1000, gameplay_changed=bool(i % 3), board_changed=bool(i % 3),
             no_op=not i % 3, action_name='RIGHT' if i % 2 else 'LEFT', action_display='RIGHT' if i % 2 else 'LEFT',
             changed_cells=i % 50, interior_changed=i % 7, valid_actions=['RIGHT', 'LEFT', 'UP', 'DOWN'],
             note='x' * 120)
    if i in (1000, 2000):
        r['level_completed'] = True
    if i == 2500:
        r['action_name'] = 'RESET'
    if i == 2600:
        r['automatic'] = True
    return r


class FakeSession:
    def __init__(self, codes):
        self.codes, self.sent = list(codes), []
        self.rebased_away, self.blocks = set(), []

    def generate(self, external):
        self.sent.append(json.loads(json.dumps(external, default=str)))
        code = self.codes.pop(0) if self.codes else "action(['RIGHT'])"     # ends the turn once the script is done
        msg = dict(role='assistant', content='', reasoning_content='t',
                   tool_calls=[dict(id='c%d' % len(self.sent), type='function',
                                    function=dict(name='python', arguments=json.dumps(dict(code=code))))])
        return Completion(message=msg, finish_reason='stop', usage=dict(prompt_tokens=10, completion_tokens=50),
                          token_ids=[1] * 50, assistant_token_ids=[1] * 50, pending=False, closed_by=None,
                          persistent={}, seconds=0.0)

    def span_evicted(self, a, b):
        return 0


class FakeGame:
    def __init__(self, state_path, n):
        self.state_path = state_path
        self.history_entries = [HistoryEntry(action='' if i == 0 else ('RIGHT' if i % 2 else 'LEFT'),
                                             frame=Frame(grid=grid_for(i), step=i, level=1 + i // 1000), result=result_for(i))
                                for i in range(n)]
        self.v6_anim = {}
        rs.write_runtime_state(self.state_path, current_frame=self.history_entries[-1].frame, history=self.history_entries)

    def step_env(self, arguments):
        if arguments.get('query') == 'animation':
            return {'record': None}
        executed = []
        for a in arguments.get('actions') or []:
            i = len(self.history_entries)
            res = dict(executed=True, action_num=i, level=1, gameplay_changed=True, board_changed=True, no_op=False,
                       action_name=a.get('action'), action_display=a.get('action'), valid_actions=['RIGHT', 'LEFT'])
            self.history_entries.append(HistoryEntry(action=a.get('action'), frame=Frame(grid=grid_for(i), step=i, level=1),
                                                     result=res))
            executed.append(res)
        rs.write_runtime_state(self.state_path, current_frame=self.history_entries[-1].frame, history=self.history_entries)
        last = dict(executed[-1]) if executed else {}
        last.update(executed=bool(executed), executed_count=len(executed),
                    executed_actions=[e['action_display'] for e in executed], requested_count=len(executed))
        return last


PROBE = '''
import json
h = history
o = {}
o["n"] = len(h)
o["is_list"] = isinstance(h, list)
o["head"] = [e.action for e in h[:4]]
o["tail"] = [e.action for e in h[-3:]]
o["res0"] = h[0].result
o["res_mid"] = h[len(h) // 2].result
o["res_old"] = h[700].result
o["old_frame"] = [h[700].frame.step, h[700].frame._grid[700 % 64][(700 // 64) % 64]]
o["tail_frame"] = [h[-1].frame.step, h[-2].frame.step]
o["str_old"] = str(h[5])
o["repr_tail"] = repr(h[-2])
o["slice_hidden"] = [[e.action, e.frame.step] for e in h[1196:1204]]
o["n_trans"] = len(transitions)
t = transitions[1500]
o["trans"] = [t.action, t.result.get("action_num"), t.before_frame.step, t.after_frame.step, str(t)]
o["last_trans"] = [last_transition.action, last_transition.after_frame.step, last_transition.before_frame.step]
o["prev"] = previous_frame.step
o["level_start"] = level_start_frame.step
o["attempt_start"] = attempt_start_frame.step
o["diff"] = frame_diff(h[700].frame, h[701].frame).get("changed_cell_count")
o["scan_steps"] = sum(e.frame.step for e in h)
o["scan_records"] = sum(1 for e in h if e.result.get("level_completed")) + sum(len(e.action) for e in h)
o["anim"] = h[-1].animation_frames
print(json.dumps(o, sort_keys=True))
'''
ACT = "for _ in range(3):\n    action(['RIGHT'])\nprint(json.dumps([len(history), history[-1].action, history[-1].frame.step, history[-4].frame.step, len(transitions)]))"


def run(paged, n, codes, hidden=None):
    """Runs the scripted calls (each turn ends at an acting call); returns each call's stdout and seconds, in order."""
    lazy_history.PAGED = paged
    state = Path(tempfile.mkdtemp()) / 'game_tool_runtime_state.json'
    game = FakeGame(state, n)
    sess = FakeSession(codes)
    log = []
    agent = A.V6Agent(session=sess, logger=lambda k, p: log.append((k, p)), count_tokens=lambda t: max(1, len(t) // 4),
                      model='M', base_url='http://x/v1', provider='vllm')
    if hidden:
        agent._v61_hidden = list(hidden)
    for _ in range(len(codes) + 2):
        if not sess.codes:
            break
        agent.analyze(state, len(game.history_entries), valid_actions=['RIGHT', 'LEFT'], step_env=game.step_env,
                      should_stop=lambda: False)
    outs, times = [], []
    for k, p in log:
        if k == 'tool-output':
            c = p.get('content')
            c = c if isinstance(c, str) else ' '.join(x.get('text', '') for x in c if isinstance(x, dict))
            try:
                outs.append(json.loads(c[c.find('{'):]).get('stdout', '').strip())
            except Exception:  # noqa: BLE001
                outs.append(c[:300])
            times.append(float(p.get('seconds') or 0))
    return game, outs[:len(codes)], times[:len(codes)]


def main():
    N = 3000
    hidden = [(1200, 1300)]
    _, ref, _ = run(False, N, [PROBE, ACT, 'print(1)'], hidden)
    _, new, _ = run(True, N, [PROBE, ACT, 'print(1)'], hidden)
    try:
        a, b = json.loads(ref[0]), json.loads(new[0])
    except Exception as exc:  # noqa: BLE001
        a, b = {'error': str(exc), 'ref': ref[:1]}, {'new': new[:1]}
    check('probe ran (reference)', 'n' in a, a)
    for k in sorted(set(a) | set(b)):
        check('paged == previous payload: %s' % k, a.get(k) == b.get(k), (a.get(k), b.get(k)))
    check('hidden range left out (length %d)' % b.get('n', -1), b.get('n') == N - 100, b.get('n'))
    check('action inside a call extends history (paged == previous)', ref[1:2] == new[1:2] and len(new) > 1, (ref[1:2], new[1:2]))
    # payload size and call time at 7,000 entries
    M = 7000
    lazy_history.PAGED = False
    g = FakeGame(Path(tempfile.mkdtemp()) / 's.json', M)
    old = len(json.dumps(ta._ascii_history_view_payload(g.history_entries)))
    lazy_history.PAGED = True
    new_size = len(json.dumps(ta._ascii_history_view_payload(g.history_entries)))
    check('payload at %d entries: paged %.0f KB vs previous %.0f KB' % (M, new_size / 1e3, old / 1e3), new_size * 10 < old)
    simple = ['print(len(history))', ACT]
    _, o1, t_old = run(False, M, simple)
    _, o2, t_new = run(True, M, simple)
    print('call times at %d entries: print(len(history)) previous %.2f s, paged %.2f s; 3 actions previous %.2f s, paged %.2f s'
          % (M, t_old[0], t_new[0], t_old[1], t_new[1]))
    check('same outputs at %d entries' % M, o1 == o2, (o1, o2))
    check('paged call faster at %d entries' % M, t_new[0] < t_old[0] and t_new[1] < t_old[1], (t_old, t_new))
    print('stats', lazy_history.STATS)
    print('TEST_V62_ROUND5K', 'PASS' if not FAIL else 'FAIL ' + ', '.join(FAIL))
    return 1 if FAIL else 0


if __name__ == '__main__':
    sys.exit(main())
