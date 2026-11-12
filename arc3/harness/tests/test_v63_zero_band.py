"""v6.3 item A (user 2026-10-12: "Let's just put the level 0 at the end of the queue ... ranked by their number of restarts"):
a game swapped out with no level completed sorts below every never-started game and every swapped-out game with >= 1 level,
whatever the swap counts; within the band fewest restarts first, then restart_value; ARC3_V62_ZERO_LEVEL_BAND=0 restores
v6.2's values; the swap-out logs the band and counts zero-band swaps. usage (pod copy): PYTHONPATH=src:. python3 tests/test_v63_zero_band.py"""
import json
import os
import tempfile
from pathlib import Path

for k, v in dict(ARC3_V61_TEXTS='1', ARC3_V6_A13='1', ARC3_V61_SUMMARY='1', ARC3_NEW_CHANGED_PROMPTS='1',
                 ARC3_ACTION_INFO='1', ARC3_REPEAT_STATE_GUARD='0', ARC3_V61_RESTART_TOKENS='150000', ARC3_V6_A3='1',
                 ARC3_GUARDS_FROM_LEVEL='2', ARC3_BATCH_NOOP_BLOCK='1', ARC3_STALE_STATE_BLOCK='1', ARC3_ANIMATION='1',
                 MULTIMODAL_CONTEXT='current_grid', MULTIMODAL_UPSCALE='8', ARC3_PERSISTENT_FUNCTIONS='1',
                 ARC3_PERSISTENT_FUNCTIONS_SCOPE='game', ARC3_PERSISTENT_FUNCTIONS_IMPORTS='1').items():
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
    print('%-78s -> %-30r %s' % (name, got, 'OK' if ok else 'WRONG (want %r)' % (want,)))
    if not ok:
        fails.append(name)


long_s = 10 ** 6
P = lambda level, n, swaps: A._V62Snapshot(level, n, swaps).priority(0.0, long_s)
untrimmed = A.ta._PRIORITY_UNTRIMMED_BASE - 100 * 221          # the last of 222 never-started games
check('A defaults: ZERO_LEVEL_BAND on, band constant 100,000', (A.ZERO_LEVEL_BAND, A._ZERO_BAND), (True, 100000))
check('A a 0-level game after 1 swap sorts below a 2-level game after 5 swaps', P(1, 7, 1) < P(3, 7, 5), True)
check('A ... and below a 1-level game after 50 swaps', P(1, 7, 1) < P(2, 7, 50), True)
check('A ... and below every never-started game', P(1, 7, 1) < untrimmed, True)
check('A the middle band is unchanged (level 3, 1 swap, 7 levels)', P(3, 7, 1), 1_000_000 - 1000 + min(999, int(round(1000 * A.restart_value(3, 7, long_s)))))
check('A within the band: fewer restarts first', P(1, 7, 1) > P(1, 7, 2) > P(1, 7, 3), True)
check('A within the band, equal restarts: the higher restart_value first (7-level vs 3-level game at a long horizon)',
      (P(1, 7, 1) > P(1, 3, 1)) == (A.restart_value(1, 7, long_s) > A.restart_value(1, 3, long_s)) and P(1, 7, 1) != P(1, 3, 1), True)
check('A band value = 100,000 - 1,000 x restarts + value', P(1, 7, 2), 100_000 - 2000 + min(999, int(round(1000 * A.restart_value(1, 7, long_s)))))
check('A level None counts as level 1', A._V62Snapshot(None, 7, 1).zero_band, True)
A.ZERO_LEVEL_BAND = False
check('A switch off: v6.2 value for a 0-level game', P(1, 7, 1), 1_000_000 - 1000 + min(999, int(round(1000 * A.restart_value(1, 7, long_s)))))
A.ZERO_LEVEL_BAND = True

# === the swap-out logs the band and counts zero-band swaps (a recording gate; in-place restart afterwards) ==============
class FakeGate:
    def __init__(self):
        self._waiting, self._clock_deadline, self.calls = [], 0.0, []

    def handover(self, priority, snapshot=None):
        self.calls.append((priority, snapshot))


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


class Run:
    number_of_levels, levels_completed, state = 5, 0, 'playing'


class TG:
    number_of_levels = 5
    game_run = Run()


class FakeGame:
    def __init__(self, state_path, level=1):
        self.state_path, self.step, self.level = state_path, 0, level
        self.game = TG()
        self.history = [HistoryEntry(action='', frame=Frame(grid=[[0] * 64 for _ in range(64)], step=0, level=level))]
        self.v6_anim = {}
        write_runtime_state(self.state_path, current_frame=self.history[-1].frame, history=self.history)

    def _add(self, name, result):
        self.step += 1
        result = dict(result, action_num=self.step, level=self.level, action_name=name, action_display=name,
                      valid_actions=['RIGHT', 'LEFT'])
        self.history.append(HistoryEntry(action=name, frame=Frame(grid=[[0] * 64 for _ in range(64)], step=self.step,
                                                                 level=self.level), result=result))
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

    def _execute_action(self, action, *, batch_index, batch_size, **kw):      # the restart's RESET
        return self._add('RESET', dict(executed=True, gameplay_changed=True, board_changed=True))

    def timing_payload(self):
        return dict(time_remaining_seconds=20000.0)


def run_swap(level):
    gate = FakeGate()
    A.ta._priority_gate = lambda: gate
    A.RESTART_TOKENS, A.RESTART_NOTES = 500, False
    state = Path(tempfile.mkdtemp()) / 'game_runtime_state.json'
    game = FakeGame(state, level=level)
    log = []
    sess = FakeSession([('action(["RIGHT"])', 600)])
    sess2 = FakeSession([('action(["LEFT"])', 100)])
    agent = A.V6Agent(session=sess, logger=lambda k, p: log.append((k, p)), count_tokens=lambda t: max(1, len(t) // 4),
                      model='M', base_url='http://x/v1', provider='vllm', session_factory=lambda: sess2)
    go = lambda: agent.analyze(state, game.step, valid_actions=['RIGHT', 'LEFT'], step_env=game.step_env, should_stop=lambda: False)
    go()                                       # 600 tokens on the level
    go()                                       # the budget: swap-out (recorded) then the in-place restart
    return gate, agent, log


gate, agent, log = run_swap(level=1)
swap = [p for k, p in log if k == 'v62-swap']
check('A swap-out of a 0-level game: band zero, reason tokens, logged', [(s['band'], s['reason'], s['swaps_on_level']) for s in swap], [('zero', 'tokens', 1)])
check('A ... the gate got the bottom-band priority', (len(gate.calls), gate.calls[0][0] < 200_000, gate.calls[0][1].zero_band), (1, True, True))
check('A ... counted', (agent._v6_stats.get('zero_band_swaps'), agent._v6_stats.get('swaps'), agent._v6_stats.get('restarts')), (1, 1, 1))
check('A ... the restart logs its reason', [p.get('reason') for k, p in log if k == 'v61-restart'], ['tokens'])
gate, agent, log = run_swap(level=3)
swap = [p for k, p in log if k == 'v62-swap']
check('A swap-out at level 3: band middle, not counted as zero-band', ([s['band'] for s in swap], agent._v6_stats.get('zero_band_swaps')), (['middle'], None))
check('A ... the gate got a swap-band priority', gate.calls[0][0] > 900_000, True)
print('FAILS:', fails)
print('test_v63_zero_band', 'PASS' if not fails else 'FAIL')
