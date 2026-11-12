"""test_v67x_gamebudget.py (2026-10-18, Claude 099768dc, implementation subagent): apply_v67x.py's item GAMEBUDGET
(ARC3_V67_GAME_POOL, user decision 55) in v62/solver.py; the agent runs on test_v67_soft.py's fakes with the real sandbox, under the
solver's own stop check (the patched per-game token limit), as dfranzen's _play_inner and the analyzer call it:
  0 the switch: no budget unless ARC3_V67_GAME_POOL > 0; run_v6's explicit default, record, install and summary line;
  1 the caps at the start: pool // games each, the rounding remainder unallocated; the start events;
  2 a game stops at its cap: at the first check at or past it (acting turns; mid-turn before the next request); the overshoot
    (at most the last request's tokens); the solver note, the exhaustion event, the receipts; the count is the agent's
    generated tokens = the model-turn completion tokens over all conversations; restarts off (the per-level budget 0): past
    200K on a level, no restart and no swap-out; restarts on: a budget restart, the count spans both conversations;
  3 redistribution through the play wrapper: a won game's leftover in equal shares to the games not yet finished, unstarted
    included, exhausted ones excluded; a crashed game's leftover likewise; a run-end finish and an exhausted game release
    nothing; caps only grow; many games finishing at once from parallel threads keep the sums;
  4 the receipts: the per-game stats, the events, the run-level summary (pool, caps, leftover, overshoot);
  5 the switch off: the patched token limit equals dfranzen's (no limit, reached, not reached); the play wrapper adds nothing.
usage (dev copy of harness_v6): PYTHONPATH=src:. python3 tests/test_v67x_gamebudget.py"""
import json
import os
import random
import threading
import tempfile  # noqa: F401  (the fakes' harness)
from pathlib import Path
from types import SimpleNamespace

_src = open(Path(__file__).with_name('test_v67_soft.py')).read()
ENV = eval(_src.split('ENV = ', 1)[1].split('\n\n', 1)[0])
FAKES = _src.split("FAKES = r'''", 1)[1].split("'''", 1)[0]
os.environ.update(ENV)
for _k in [k for k in os.environ if k.startswith('ARC3_V67_')]:
    os.environ.pop(_k)
import v62.patches  # noqa: E402,F401
import v62.retention as r  # noqa: E402
r.install()
import v62.agent as A  # noqa: E402
import v62.solver as SV  # noqa: E402
from inference.framework import solver as sv  # noqa: E402
exec(compile(FAKES, 'test_v67_soft.py (FAKES)', 'exec'), globals())
fails = []


def check(name, got, want):
    ok = got == want
    print('%-84s -> %-40r %s' % (name, got if not isinstance(got, str) else got[:40], 'OK' if ok else 'WRONG (want %r)' % (want,)))
    if not ok:
        fails.append(name)


def events(log, kind):
    return [p for k, p in log if k == kind]


class Writer:
    """run_v6's writer(gid) stand-in: every event, per game, in order"""

    def __init__(self):
        self.lines, self.lock = [], threading.Lock()

    def __call__(self, gid):
        def log(kind, payload):
            with self.lock:
                self.lines.append((gid, kind, dict(payload)))
        return log

    def of(self, gid=None, reason=None):
        return [(g, p) for g, k, p in self.lines if (gid is None or g == gid) and (reason is None or p.get('reason') == reason)]


class Gate:
    _waiting, _clock_deadline = [], None

    def __init__(self):
        self.calls = []

    def handover(self, priority, snapshot):
        self.calls.append(priority)


class FakeSolver:
    max_generated_tokens_per_game = None

    def __init__(self, soft=None):
        self.soft = soft

    def soft_time_remaining_seconds(self):
        return self.soft


def session(gid, analyzer, solver=None):
    """the attributes of dfranzen's _HarnessGameSession that the GAMEBUDGET hooks read"""
    s = SimpleNamespace(game=SimpleNamespace(game_run=SimpleNamespace(game_id=gid, state='playing', solver_note=None)),
                        analyzer=analyzer, game_token_baseline=0, stop_event=threading.Event(), solver=solver or FakeSolver(),
                        game_index=0)
    s.runtime_limit_reached = lambda: False
    return s


def play(agent, game, fs, turns=100):
    """dfranzen's _play_inner loop on the fakes: should_stop (the patched token limit) at every loop boundary and, through the
    analyzer, before every model request"""
    stop = lambda: SV._token_limit_reached(fs)   # noqa: E731
    n = 0
    while not stop() and n < turns:
        game.auto_reset()
        agent.analyze(game.state_path, len(game.history_entries) - 1, valid_actions=VALID, step_env=game.step_env,
                      should_stop=stop)
        n += 1
    return n


def completions(log):
    return sum(int((p.get('usage') or {}).get('completion_tokens') or 0) for p in events(log, 'model-turn'))


R_, L_ = 'action(["RIGHT"])', 'action(["LEFT"])'
SV.STAGGER = 0

# === 0. the switch ======================================================================================================
check('0 off by default: no budget installed', (SV.gamebudget_install(['a', 'b']), SV.GAMEBUDGET), (None, None))
RUN = open(Path(A.__file__).with_name('run_v6.py')).read()
check('0 run_v6: the explicit default, the record, the install before the run, the summary before v6-stats is written',
      ("os.environ.setdefault('ARC3_V67_GAME_POOL', '0')" in RUN,
       "'ARC3_V67_GAME_POOL'," in RUN.split('variants=')[1].split('history_eviction=')[0],
       RUN.index('v67x_solver.gamebudget_install(games, writer)') < RUN.index('deadline = datetime.fromtimestamp'),
       RUN.index("stats['GAMEBUDGET'] = v67x_solver.GAMEBUDGET.summary()") < RUN.index("(out / 'v6-stats.json').write_text")),
      (True, True, True, True))
check('0 the hook is installed on dfranzen\'s session class', (sv._HarnessGameSession.token_limit_reached is SV._token_limit_reached,
      sv._HarnessGameSession.play is SV._play), (True, True))
import v62.variants as V  # noqa: E402
RAW = json.load(open(Path(A.__file__).with_name('texts.json')))['texts']


def render(**env):
    keep = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        return V.apply(json.loads(json.dumps(RAW)))
    except RuntimeError as exc:
        return 'ERROR: %s' % str(exc)[:60]
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


with_restarts = render(ARC3_V61_RESTART_TOKENS='200000')
check('0 per-level budget off in a pool run: the texts render, identical to the restart runs\' (the restart sentence kept)',
      (render(ARC3_V61_RESTART_TOKENS='0', ARC3_V67_GAME_POOL='90000') == with_restarts,
       render(ARC3_V61_RESTART_TOKENS='200000', ARC3_V67_GAME_POOL='90000') == with_restarts,
       'The saved summary is shown to you again if this game restarts a level.' in with_restarts['levelup']), (True, True, True))
check('0 without a pool the per-level budget off still fails at the round-5l anchor, as in the deployed tree (unchanged)',
      str(render(ARC3_V61_RESTART_TOKENS='0'))[:44], 'ERROR: variant anchor not found exactly once')

# === 1. the caps at the start ============================================================================================
os.environ['ARC3_V67_GAME_POOL'] = '1000'
w = Writer()
gb = SV.gamebudget_install(['a', 'b', 'c', 'b'], w)
os.environ.pop('ARC3_V67_GAME_POOL')
gb.start('a')
check('1 the caps: 1,000 // 3 games (a duplicate listed once), the remainder unallocated; the start event',
      (SV.GAMEBUDGET is gb, gb.games, gb.cap, gb.unallocated, [(g, p['cap'], p['pool'], p['games']) for g, p in w.of(reason='start')]),
      (True, ['a', 'b', 'c'], {'a': 333, 'b': 333, 'c': 333}, 1, [('a', 333, 1000, 3)]))
SV.GAMEBUDGET = None


# === 2. a game stops at its cap ===========================================================================================
def budget_run(scripts, cap, per_level=0, gate=None, turns=100):
    game, sessions, agent, log, go = harness(scripts, budget=per_level)
    if gate is not None:
        A.ta._priority_gate = lambda: gate
    w = Writer()
    SV.GAMEBUDGET = SV.GameBudget(cap, ['g1'], w)
    fs = session('g1', agent)
    SV.GAMEBUDGET.start('g1')
    n = play(agent, game, fs, turns)
    SV._gamebudget_finish(fs, SV.GAMEBUDGET, 'g1')
    A.ta._priority_gate = lambda: None
    out = dict(game=game, sessions=sessions, agent=agent, log=log, w=w, fs=fs, turns=n, budget=SV.GAMEBUDGET)
    SV.GAMEBUDGET = None
    return out


alt = lambda k, t: [(R_ if i % 2 == 0 else L_, t) for i in range(k)]   # noqa: E731
x = budget_run([alt(12, 700)], 5000)
ex = [p for g, p in x['w'].of(reason='exhausted')]
check('2 acting turns: stops at the first check past 5,000 (8 turns, 5,600 tokens; overshoot 600 < one request\'s 700)',
      (x['turns'], x['agent'].generated_tokens, len(events(x['log'], 'model-turn')), [(p['tokens'], p['cap'], p['overshoot']) for p in ex]),
      (8, 5600, 8, [(5600, 5000, 600)]))
check('2 the solver note records budget_exhausted; the receipts in the agent\'s stats',
      (x['fs'].game.game_run.solver_note, {k: x['agent']._v6_stats.get(k) for k in ('game_cap_final', 'game_tokens_used',
       'game_budget_exhausted', 'redistributed_in', 'redistributed_out', 'game_budget_end')}),
      ('budget_exhausted: tokens=5600 cap=5000', dict(game_cap_final=5000, game_tokens_used=5600, game_budget_exhausted=True,
                                                     redistributed_in=0, redistributed_out=0, game_budget_end='exhausted')))
check('2 the count is the agent\'s generated tokens = the model-turn completion tokens', x['agent'].generated_tokens,
      completions(x['log']))
x = budget_run([[('print(%d)' % i, 700) for i in range(6)] + alt(4, 100)], 2000)
check('2 mid-turn: the analyzer stops before the 4th request of a turn (2,100 tokens; overshoot 100), no action',
      (x['turns'], x['agent'].generated_tokens, len(events(x['log'], 'model-turn')), x['game'].engine,
       [(p['tokens'], p['overshoot']) for g, p in x['w'].of(reason='exhausted')]), (1, 2100, 3, [], [(2100, 100)]))
gate = Gate()
x = budget_run([alt(8, 40000)], 250000, per_level=0, gate=gate)
check('2 restarts off (per-level budget 0): 280K on one level, no restart, no notes turn, no swap-out; stops at 250K',
      (x['turns'], x['agent'].generated_tokens, x['agent']._v61_level_tokens, x['agent']._v6_stats.get('restarts'),
       x['agent']._v6_stats.get('summary_turns'), gate.calls, events(x['log'], 'v62-swap'),
       [p['overshoot'] for g, p in x['w'].of(reason='exhausted')]),
      (7, 280000, 280000, None, None, [], [], [30000]))
NOTES = 'save_summary("Notes L1:\\n- RIGHT and LEFT move the marker one cell\\n- nothing else changes")'
gate = Gate()
x = budget_run([alt(3, 40000) + [(NOTES, 500)], alt(4, 40000)], 200000, per_level=100000, gate=gate)
check('2 restarts on (per-level budget 100K): a budget restart (and its swap-out); the count spans both conversations',
      ([(p['reason'], p['tokens_on_level']) for p in events(x['log'], 'v61-restart')], len(gate.calls), x['agent'].generated_tokens,
       completions(x['log']), x['agent']._v61_level_tokens, [(p['tokens'], p['overshoot']) for g, p in x['w'].of(reason='exhausted')]),
      ([('tokens', 120500)], 1, 200500, 200500, 80000, [(200500, 500)]))


# === 3. redistribution through the play wrapper ===========================================================================
def analyzer():
    return SimpleNamespace(generated_tokens=0, _v6_stats={}, _v6_session=None)


def stub_play(tokens, state=None, check_cap=False, stop=False):
    """the game's play as the stub of dfranzen's play(): it generates `tokens`, may end in `state`, may run into its cap (the
    solver's stop check) or be stopped at the run's end"""
    def fn(self):
        self.analyzer.generated_tokens = tokens
        if check_cap:
            SV._token_limit_reached(self)
        if state:
            self.game.game_run.state = state
        if stop:
            self.stop_event.set()
    return fn


real_original_play = SV._original_play
w = Writer()
gb = SV.GameBudget(4000, ['a', 'b', 'c', 'd', 'e'], w)
SV.GAMEBUDGET = gb
games = {g: session(g, analyzer()) for g in 'abcde'}
for g, fn in (('a', stub_play(300, 'won')), ('b', stub_play(1000, check_cap=True)), ('c', stub_play(425, 'crashed')),
              ('d', stub_play(100, stop=True))):
    SV._original_play = fn
    SV._play(games[g])
SV._original_play = real_original_play
SV.GAMEBUDGET = None
check('3 a (won, 300 of 800): its 500 left goes to b, c, d, e (e unstarted), 125 each',
      [(g, p['source'], p['amount'], p['cap']) for g, p in w.of(reason='redistribution') if p['source'] == 'a'],
      [('b', 'a', 125, 925), ('c', 'a', 125, 925), ('d', 'a', 125, 925), ('e', 'a', 125, 925)])
check('3 b exhausts at 925 (1,000 tokens): no release; c (crashed, 425 of 925) gives its 500 to d and e only',
      ([(g, p['tokens'], p['cap'], p['overshoot']) for g, p in w.of(reason='exhausted')],
       [(g, p['amount'], p['cap']) for g, p in w.of(reason='redistribution') if p['source'] == 'c']),
      ([('b', 1000, 925, 75)], [('d', 250, 1175), ('e', 250, 1175)]))
check('3 d ends at the run\'s end (100 tokens): nothing released; e never started and received 375',
      ([(g, p['how'], p['tokens'], p['cap'], p['released'], p['receivers']) for g, p in w.of(reason='finish')], gb.received['e'],
       gb.cap['e'], 'e' in gb.ended),
      ([('a', 'won', 300, 800, 500, 4), ('b', 'exhausted', 1000, 925, 0, 0), ('c', 'crashed', 425, 925, 500, 2),
        ('d', 'run_end', 100, 1175, 0, 0)], 375, 1175, False))
check('3 the receipts in each game\'s stats', {g: games[g].analyzer._v6_stats for g in 'abcd'}, {
    'a': dict(game_cap_final=800, game_tokens_used=300, game_budget_exhausted=False, redistributed_in=0, redistributed_out=500,
              game_budget_end='won'),
    'b': dict(game_cap_final=925, game_tokens_used=1000, game_budget_exhausted=True, redistributed_in=125, redistributed_out=0,
              game_budget_end='exhausted'),
    'c': dict(game_cap_final=925, game_tokens_used=425, game_budget_exhausted=False, redistributed_in=125, redistributed_out=500,
              game_budget_end='crashed'),
    'd': dict(game_cap_final=1175, game_tokens_used=100, game_budget_exhausted=False, redistributed_in=375, redistributed_out=0,
              game_budget_end='run_end')})
check('3 the start events (d and e: caps grown while waiting)', [(g, p['cap'], p['received']) for g, p in w.of(reason='start')],
      [('a', 800, 0), ('b', 925, 125), ('c', 925, 125), ('d', 1175, 375)])
check('3 caps only grow (every cap event of every game is non-decreasing)',
      [g for g in 'abcde' if [p['cap'] for gg, k, p in w.lines if gg == g and p.get('reason') in ('start', 'redistribution')]
       != sorted(p['cap'] for gg, k, p in w.lines if gg == g and p.get('reason') in ('start', 'redistribution'))], [])
summary = gb.summary()
check('4 the run-level summary: pool, caps (conserved), tokens, leftover never used, overshoot, redistributed, exhausted',
      {k: summary[k] for k in ('pool', 'games', 'base_cap', 'unallocated', 'caps_sum', 'tokens_used', 'leftover_unused',
                               'overshoot', 'redistributed', 'exhausted')},
      dict(pool=4000, games=5, base_cap=800, unallocated=0, caps_sum=4000, tokens_used=1825, leftover_unused=2250,
           overshoot=75, redistributed=1000, exhausted=1))
check('4 the summary\'s per-game numbers (e: not finished)', summary['per_game']['e'],
      dict(cap=1175, used=0, received=375, released=0, end='not_finished', exhausted=False))
check('4 the accounting closes: tokens used - overshoot + leftover = caps', summary['tokens_used'] - summary['overshoot']
      + summary['leftover_unused'], summary['caps_sum'])

# thread safety: 60 games finishing from 12 threads at once (won with leftovers, exhausted, run end), with cap checks
rng = random.Random(7)
ids = ['g%02d' % i for i in range(60)]
gb = SV.GameBudget(600000, ids, Writer())
plan = {g: (rng.choice(['won', 'crashed', 'gave_up', 'run_end', 'exhausted']), rng.randrange(0, 9000)) for g in ids}
barrier = threading.Barrier(12)
errors = []


def worker(chunk):
    try:
        barrier.wait()
        for g in chunk:
            how, spent = plan[g]
            if how == 'exhausted':
                gb.reached(g, 10 ** 7)
            for _ in range(3):
                gb.reached(rng.choice(ids), 5000)
            gb.finish(g, 10 ** 7 if how == 'exhausted' else spent, how)
    except Exception as exc:  # noqa: BLE001
        errors.append(repr(exc))


threads = [threading.Thread(target=worker, args=(ids[i::12],)) for i in range(12)]
for t in threads:
    t.start()
for t in threads:
    t.join()
s = gb.summary()
check('3 parallel finishes: no error; caps conserved; released = received (all games ended); every game ended once',
      (errors, s['caps_sum'], sum(gb.released.values()) == sum(gb.received.values()), len(gb.ended),
       all(gb.cap[g] >= 10000 for g in ids)), ([], 600000, True, 60, True))

# === 5. the switch off ===================================================================================================
SV.GAMEBUDGET = None
same = []
for limit, tokens in ((None, 10 ** 6), (1000, 1500), (1000, 500)):
    fs = session('a', SimpleNamespace(generated_tokens=tokens), FakeSolver())
    fs.solver.max_generated_tokens_per_game = limit
    same.append((SV._token_limit_reached(fs), SV._original_token_limit(fs)))
check('5 off: the patched token limit equals dfranzen\'s (no limit, reached, not reached)', same,
      [(False, False), (True, True), (False, False)])
fs = session('a', analyzer())
SV._original_play = stub_play(500, 'won')
SV._play(fs)
SV._original_play = real_original_play
check('5 off: the play wrapper adds no stats, no note', (fs.analyzer._v6_stats, fs.game.game_run.solver_note), ({}, None))
print('FAILS:', fails)
