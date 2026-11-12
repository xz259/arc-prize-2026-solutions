"""test_v67x_ext200.py (2026-10-18, Claude 099768dc, implementation subagent): apply_v67x.py's item EXT200 (ARC3_V67_EXT200,
user decision 48), through agent.analyze with the real sandbox on test_v67_soft.py's fakes:
  0 the switches: the defaults (0 = off, 50000, 20261018), the range guard, the env read in a fresh process; run_v6's explicit
    defaults, the record in run_settings.json and the run-end close;
  1 the assignment: u is the documented stable hash of (seed, game, level, attempt) in [0, 1) (the same in a fresh process);
    the fraction holds over 2,000 keys (+-3%); the seed matters;
  2 the extend arm: no notes turn at 200K, the attempt plays on, the normal notes turn and restart at 250K, the swap-out at the
    new limit, the receipts and both events; the next attempt's own assignment (attempt 2);
  3 the restart arm: the notes turn and restart at 200K exactly as with the switch off (identical traffic and actions; events
    and stats differ only by the ext200 receipts);
  4 a clear during the extension (receipts; the next level's budget is 200K again); a clear the acting turn did not report;
  5 the late window: 'late' logged once and nothing changes (traffic identical to the switch off), its outcome at the clear;
    an extended attempt that reaches 250K in the late window plays on (as today) and closes at the run's end;
  6 the budget text: the loop-rule block states 250,000 in the extend arm (a block shown after the extension, and a standing
    block repeated in a refusal), 200,000 in the late arm; no other text or format carries the budget;
  7 no assignment at a position notes turn, a session-error restart or a pending notes turn / restart; the no-action limit
    reached at the assignment turn restarts the extended attempt for 'noact';
  8 the switch off is byte-identical to the deployed v6.7 base (v62/*.pre-v67x, a subprocess on each tree), with the v6.7
    switches off and with the core switches on (LOOP, TEXTS67, NOTES, SCORE); with the switch on and every assignment in the
    restart or late arm, the traffic, actions, records, library and texts are identical and only the ext200 events and stats
    are added.
usage (dev copy of harness_v6): PYTHONPATH=src:. python3 tests/test_v67x_ext200.py"""
import copy
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace  # noqa: F401  (the fakes' harness)

_src = open(Path(__file__).with_name('test_v67_soft.py')).read()
ENV = eval(_src.split('ENV = ', 1)[1].split('\n\n', 1)[0])
FAKES = _src.split("FAKES = r'''", 1)[1].split("'''", 1)[0]
os.environ.update(ENV)
for _k in ('ARC3_V67_EXT200', 'ARC3_V67_EXT200_TOKENS', 'ARC3_V67_EXT200_SEED', 'ARC3_V67_RESTART_KIND'):
    os.environ.pop(_k, None)
os.environ['ARC3_V67_LOOP'] = '1'                       # the loop-rule texts (section 6); A.LOOP is set per run
import v62.patches  # noqa: E402,F401
import v62.retention as r  # noqa: E402
r.install()
import v62.variants as V  # noqa: E402
import v62.agent as A  # noqa: E402
from inference.agent.runtime_state import load_runtime_state  # noqa: E402
exec(compile(FAKES, 'test_v67_soft.py (FAKES)', 'exec'), globals())
fails = []
ROOT = Path(A.__file__).resolve().parents[1]


def check(name, got, want):
    ok = got == want
    print('%-84s -> %-40r %s' % (name, got if not isinstance(got, str) else got[:40], 'OK' if ok else 'WRONG (want %r)' % (want,)))
    if not ok:
        fails.append(name)


def user_text(batch):
    m = [x for x in batch if x.get('role') == 'user'][-1]
    c = m['content']
    return c if isinstance(c, str) else '\n'.join(p.get('text', '') for p in c if p.get('type') == 'text')


def events(log, kind):
    return [p for k, p in log if k == kind]


def tool_errors(batch):
    """the error fields of a sent batch's tool results (JSON payloads)"""
    out = []
    for m in batch:
        if m.get('role') == 'tool' and isinstance(m.get('content'), str):
            try:
                out.append(json.loads(m['content']).get('error') or '')
            except (TypeError, ValueError, AttributeError):
                pass
    return '\n'.join(out)


class Gate:
    _waiting, _clock_deadline = [], None

    def __init__(self):
        self.calls = []

    def handover(self, priority, snapshot):
        self.calls.append(priority)


GID, SEED = 'tst-ext200', 20261018
R_, L_, U_ = 'action(["RIGHT"])', 'action(["LEFT"])', 'action(["UP"])'
EXT_EVENTS = ('v67-ext200', 'v67-ext200-outcome')
LATE = lambda: dict(time_remaining_seconds=600.0)       # noqa: E731  (below RESTART_MIN_LEFT: the late gate)


def h(scripts, frac=1.0, ext=50000, seed=SEED, loop=False, gate=None, budget=200000, **kw):
    """harness() with the EXT200 knobs, the loop switch and the game id; an optional allocator gate"""
    out = harness(scripts, budget=budget, EXT200=frac, EXT200_TOKENS=ext, EXT200_SEED=seed, LOOP=loop, **kw)
    out[0].game.game_run.game_id = GID
    if gate is not None:
        A.ta._priority_gate = lambda: gate
    return out


def plain(log, drop=()):
    """the log as comparable JSON (tool-output timings dropped), without the event kinds in drop"""
    out = []
    for k, p in log:
        if k in drop:
            continue
        p = json.loads(json.dumps(p, default=str))
        if k == 'tool-output':
            p.pop('seconds', None)
        out.append([k, p])
    return out


def nonext_stats(stats):
    return {k: v for k, v in json.loads(json.dumps(stats, default=str)).items() if not k.startswith('ext200_')}


def outcome(p):
    return (p['arm'], p['cleared'], p['ended'], p['reason'], p['tokens_on_level'], p['tokens_since'], p['actions_since'])


def ext_stats(agent):
    return tuple(agent._v6_stats.get(k) for k in ('ext200_extend', 'ext200_restart', 'ext200_late', 'ext200_extend_cleared',
                                                   'ext200_extend_tokens'))


# === 0. the switches ===================================================================================================
check('0 off by default; the knobs', (A.EXT200, A.EXT200_TOKENS, A.EXT200_SEED), (0.0, 50000, 20261018))
PROBE = r'''
import json, os, sys
os.environ.update(json.loads(sys.argv[1]))
import v62.patches
import v62.retention as r
r.install()
import v62.agent as A
print('PROBE ' + json.dumps(dict(ext=A.EXT200, tokens=A.EXT200_TOKENS, seed=A.EXT200_SEED,
                                 u=[A._v67x_u(A.EXT200_SEED, 'ls20-9607627b', 1, k) for k in (1, 2, 3)])))
'''


def probe(**env):
    e = dict(os.environ, PYTHONPATH='%s:%s' % (ROOT / 'src', ROOT))
    e.update(env)
    p = subprocess.run([sys.executable, '-c', PROBE, json.dumps(env)], env=e, cwd=str(ROOT), capture_output=True, text=True,
                       timeout=600)
    line = [x for x in p.stdout.splitlines() if x.startswith('PROBE ')]
    return p.returncode, (json.loads(line[-1][6:]) if line else None), p.stderr


with ThreadPoolExecutor(2) as pool:
    ok_run, bad_run = pool.map(lambda env: probe(**env), [dict(ARC3_V67_EXT200='0.5', ARC3_V67_EXT200_TOKENS='40000',
                                                               ARC3_V67_EXT200_SEED='7'), dict(ARC3_V67_EXT200='1.5')])
check('0 a fresh process reads the switches; its u equals this process\'s',
      (ok_run[0], ok_run[1] and (ok_run[1]['ext'], ok_run[1]['tokens'], ok_run[1]['seed'],
                                 ok_run[1]['u'] == [A._v67x_u(7, 'ls20-9607627b', 1, k) for k in (1, 2, 3)])),
      (0, (0.5, 40000, 7, True)))
check('0 a fraction outside [0, 1] refuses to start', (bad_run[0] != 0, 'ARC3_V67_EXT200 must be a fraction in [0, 1]' in bad_run[2]),
      (True, True))
RUN = open(Path(A.__file__).with_name('run_v6.py')).read()
check('0 run_v6: the explicit defaults, the three switches recorded, the run-end close before the stats are written',
      ("os.environ.setdefault('ARC3_V67_EXT200', '0')" in RUN, "os.environ.setdefault('ARC3_V67_EXT200_TOKENS', '50000')" in RUN,
       "os.environ.setdefault('ARC3_V67_EXT200_SEED', '20261018')" in RUN,
       "'ARC3_V67_EXT200', 'ARC3_V67_EXT200_TOKENS', 'ARC3_V67_EXT200_SEED'," in RUN.split('variants=')[1].split('history_eviction=')[0],
       RUN.index('agent._v67x_run_end()') < RUN.index("stats = {gid: a._v6_stats for gid, a in agents.items()}")
       < RUN.index("(out / 'v6-stats.json').write_text")), (True,) * 5)

# === 1. the assignment's draw ==========================================================================================
u = A._v67x_u(SEED, 'ls20-9607627b', 3, 2)
ref = (int.from_bytes(hashlib.sha256(b'20261018|ls20-9607627b|3|2').digest()[:8], 'big') >> 11) / 2.0 ** 53
check('1 u: the documented formula (sha256 of seed|game|level|attempt, top 53 bits), in [0, 1), stable',
      (u == ref, 0.0 <= u < 1.0, A._v67x_u(SEED, 'ls20-9607627b', 3, 2) == u), (True, True, True))
KEYS = [('g%03d' % g, lvl, att) for g in range(100) for lvl in (1, 2, 3, 4) for att in (1, 2, 3, 4, 5)]
check('1 2,000 distinct keys', len(set(KEYS)), 2000)
shares = {f: sum(A._v67x_u(SEED, *k) < f for k in KEYS) / len(KEYS) for f in (0.1, 0.25, 0.5, 0.75, 0.9)}
check('1 the share extended is the fraction +-3%% (0.1, 0.25, 0.5, 0.75, 0.9): %s' % shares,
      [f for f, s in shares.items() if abs(s - f) > 0.03], [])
check('1 fraction 0 extends none, 1 extends all', (sum(A._v67x_u(SEED, *k) < 0.0 for k in KEYS),
                                                   sum(A._v67x_u(SEED, *k) < 1.0 for k in KEYS)), (0, 2000))
diff = sum((A._v67x_u(SEED, *k) < 0.5) != (A._v67x_u(SEED + 1, *k) < 0.5) for k in KEYS) / len(KEYS)
check('1 another seed reassigns about half of the keys (%.3f)' % diff, 0.4 < diff < 0.6, True)

# === 2. the extend arm: plays on past 200K, the normal notes turn and restart at 250K; the next attempt's assignment ======
conv1 = [(PROG + R_, 120000), (R_, 90000), (R_, 30000), (R_, 20000), (NOTES, 500)]
conv2 = [(R_, 100), (R_, 150000), (R_, 60000), (R_, 100)]
gate = Gate()
game, (s1, s2), agent, log, go = h([conv1, conv2], frac=1.0, gate=gate)
go()
go()
s1.last_receipt, s1.session_id, s1.request_count = {'logical_tokens': 345678}, 'arc3-fake', 2
go()
check('2 the assignment at 210K: extend; no notes turn; the attempt plays on',
      (A.TEXTS['attempt_summary_request'].split('{tokens}')[0] in user_text(s1.sent[2]), agent._v67x_budget(1),
       agent._v6_stats.get('summary_turns'), game.engine), (False, 250000, None, ['RIGHT'] * 3))
ev = events(log, 'v67-ext200')
check('2 the v67-ext200 event: its fields', sorted(ev[0]), sorted(
      ['game', 'level', 'attempt', 'u', 'arm', 'tokens_on_level', 'budget', 'logical', 'step', 'requests', 'turns', 'obs_index',
       'session_id', 'session_requests', 'conversation', 'game_attempt', 'actions_on_level', 'noact_tokens', 'seed', 'fraction']))
check('2 the v67-ext200 event: the values (the checkpoint after request 2 / opener 2 of conversation 1)',
      {k: ev[0][k] for k in ev[0]}, dict(game=GID, level=1, attempt=1, u=A._v67x_u(SEED, GID, 1, 1), arm='extend',
                                          tokens_on_level=210000, budget=250000, logical=345678, step=2, requests=2, turns=2,
                                          obs_index=2, session_id='arc3-fake', session_requests=2, conversation=1, game_attempt=1,
                                          actions_on_level=2, noact_tokens=0, seed=SEED, fraction=1.0))
go()
go()
check('2 at 250K: the normal notes turn (the request states the tokens), once',
      (A._fill(A.TEXTS['attempt_summary_request'], level=1, tokens='260,000') in user_text(s1.sent[4]),
       [(p['tokens_on_level'], p['reason']) for p in events(log, 'v62-summary-turn')]), (True, [(260000, 'tokens')]))
go()
check('2 the restart follows the notes turn: hard, reason tokens, the swap-out at the new limit',
      ([(p['reason'], p['kind'], p['tokens_on_level']) for p in events(log, 'v61-restart')], len(gate.calls),
       [(p['reason'], p['tokens_on_level']) for p in events(log, 'v62-swap')], game.engine[4:6]),
      ([('tokens', 'hard', 260500)], 1, [('tokens', 260500)], ['RESET', 'RIGHT']))
check('2 the outcome event: its fields', sorted(events(log, 'v67-ext200-outcome')[0]), sorted(
      ['game', 'level', 'attempt', 'arm', 'cleared', 'ended', 'reason', 'tokens_on_level', 'tokens_since', 'actions_since',
       'budget']))
check('2 the outcome: extend, not cleared, at the restart (tokens 260,500 with the notes turn; 50,500 and 2 actions since)',
      [outcome(p) for p in events(log, 'v67-ext200-outcome')], [('extend', False, 'restart', 'tokens', 260500, 50500, 2)])
check('2 the next attempt starts at the 200K budget', (agent.__dict__.get('_v67x_ext'), agent._v67x_budget(1),
                                                       agent._v61_level_tokens), (None, 200000, 100))
for _ in range(3):
    go()
ev = events(log, 'v67-ext200')
check('2 the next attempt reaching 200K: its own assignment (attempt 2, conversation 2, its own u)',
      [(p['attempt'], p['conversation'], p['arm'], p['tokens_on_level'], p['u']) for p in ev[1:]],
      [(2, 2, 'extend', 210100, A._v67x_u(SEED, GID, 1, 2))])
check('2 receipts (extend 2, restart 0, late 0, cleared 0, the extension tokens), one restart, one notes turn',
      (ext_stats(agent), agent._v6_stats.get('restarts'), agent._v6_stats.get('summary_turns')),
      ((2, 0, 0, 0, [50500]), 1, 1))

# === 3. the restart arm: unchanged =====================================================================================
SCRIPT3 = [[(PROG + R_, 120000), (R_, 90000), (R_, 60), (NOTES, 500)], [(R_, 100), (R_, 100)]]
u1 = A._v67x_u(SEED, GID, 1, 1)
runs3 = {}
for name, frac in (('restart', u1 / 2), ('off', 0.0)):
    game, sessions, agent, log, go = h(copy.deepcopy(SCRIPT3), frac=frac, gate=Gate())
    for _ in range(5):
        go()
    runs3[name] = (game, sessions, agent, log)
game, (s1, s2), agent, log = runs3['restart']
check('3 the assignment at 210K: restart (u >= the fraction); the notes turn at once with the 210,000 request',
      ([p['arm'] for p in events(log, 'v67-ext200')], A._fill(A.TEXTS['attempt_summary_request'], level=1, tokens='210,000')
       in user_text(s1.sent[2])), (['restart'], True))
check('3 the outcome: at the restart (the notes turn\'s 560 tokens since, no action)',
      [outcome(p) for p in events(log, 'v67-ext200-outcome')], [('restart', False, 'restart', 'tokens', 210560, 560, 0)])
offg, offs, offa, offlog = runs3['off']
check('3 identical to the switch off: model traffic, game actions, attempt records, library',
      ([x.sent for x in sessions] == [x.sent for x in offs], game.engine == offg.engine,
       {k: game.__dict__.get(k) for k in ('v6_level_attempts', 'v62_level_attempt_ends', 'v67_level_restarts')} ==
       {k: offg.__dict__.get(k) for k in ('v6_level_attempts', 'v62_level_attempt_ends', 'v67_level_restarts')},
       sorted(agent._kept_functions) == sorted(offa._kept_functions)), (True, True, True, True))
check('3 events and stats identical apart from the ext200 receipts', (plain(log, EXT_EVENTS) == plain(offlog),
      nonext_stats(agent._v6_stats) == nonext_stats(offa._v6_stats), ext_stats(agent), ext_stats(offa),
      [k for k, _ in offlog if k in EXT_EVENTS]), (True, True, (0, 1, 0, 0, []), (None,) * 5, []))

# === 4. a clear during the extension ===================================================================================
game, (s1,), agent, log, go = h([[(PROG + R_, 120000), (R_, 90000), (U_, 1000), (R_, 100)]], frac=1.0)
for _ in range(3):
    go()
check('4 the clear ends the attempt: cleared, at the clear (211,000 tokens; 1,000 and 1 action since)',
      [outcome(p) for p in events(log, 'v67-ext200-outcome')], [('extend', True, 'clear', None, 211000, 1000, 1)])
check('4 receipts: extend 1, cleared 1, the extension tokens', ext_stats(agent), (1, 0, 0, 1, [1000]))
go()
check('4 the next level: no open attempt, the 200K budget, no second outcome', (agent.__dict__.get('_v67x_ext'),
      agent._v67x_budget(2), agent._v61_level_tokens, len(events(log, 'v67-ext200-outcome')), game.level), (None, 200000, 100, 1, 2))
game, (s1,), agent, log, go = h([[(PROG + R_, 120000), (R_, 90000), (U_, 1000), (R_, 100)]], frac=1.0)
agent._v67x_cleared = lambda: None                      # a clear the acting turn does not report
for _ in range(4):
    go()
check('4 a clear the acting turn did not report: closed at the level change (cleared, the same tokens)',
      ([outcome(p) for p in events(log, 'v67-ext200-outcome')], agent._v67x_budget(2)),
      ([('extend', True, 'level_change', None, 211000, 1000, 1)], 200000))

# === 5. the late window ================================================================================================
SCRIPT5 = [[(PROG + R_, 120000), (R_, 90000), (R_, 5000), (U_, 1000), (R_, 100)]]
runs5 = {}
for name, frac in (('late', 1.0), ('off', 0.0)):
    game, sessions, agent, log, go = h(copy.deepcopy(SCRIPT5), frac=frac)
    go()
    go()
    game.timing_payload = LATE
    for _ in range(3):
        go()
    runs5[name] = (game, sessions, agent, log)
game, sessions, agent, log = runs5['late']
offg, offs, offa, offlog = runs5['off']
check('5 in the late window: one assignment, late (fraction 1); the restart skipped; the receipts',
      ([(p['arm'], p['tokens_on_level']) for p in events(log, 'v67-ext200')], len(events(log, 'v62-restart-skipped')),
       ext_stats(agent), agent._v6_stats.get('restarts_skipped_late'), agent._v6_stats.get('summary_turns'),
       agent._v6_stats.get('restarts')), ([('late', 210000)], 1, (0, 0, 1, 0, []), 1, None, None))
check('5 nothing changes: traffic, actions, events and stats identical to the switch off (ext200 receipts aside)',
      ([x.sent for x in sessions] == [x.sent for x in offs], game.engine == offg.engine, plain(log, EXT_EVENTS) == plain(offlog),
       nonext_stats(agent._v6_stats) == nonext_stats(offa._v6_stats)), (True, True, True, True))
check('5 the late attempt\'s outcome at its clear', [outcome(p) for p in events(log, 'v67-ext200-outcome')],
      [('late', True, 'clear', None, 216000, 6000, 2)])
game, (s1,), agent, log, go = h([[(PROG + R_, 120000), (R_, 90000), (R_, 50000), (R_, 100), (R_, 100)]], frac=1.0)
for _ in range(3):
    go()
game.timing_payload = LATE
go()
go()
check('5 an extended attempt past 250K in the late window plays on (no notes turn, no restart), as today',
      (agent._v6_stats.get('summary_turns'), agent._v6_stats.get('restarts'), agent._v6_stats.get('restarts_skipped_late'),
       events(log, 'v67-ext200-outcome'), game.engine), (None, None, 1, [], ['RIGHT'] * 5))
agent._v67x_run_end()
check('5 the run\'s end closes it (run_v6): run_end, not cleared; the extension tokens',
      ([outcome(p) for p in events(log, 'v67-ext200-outcome')], ext_stats(agent)),
      ([('extend', False, 'run_end', None, 260200, 50200, 3)], (1, 0, 0, 0, [50200])))
agent._v67x_run_end()
check('5 the run-end close runs once', len(events(log, 'v67-ext200-outcome')), 1)
game, (s1,), agent, log, go = h([[(R_, 120000), (R_, 90000), (R_, 100)]], frac=1.0)
for _ in range(3):
    go()
game.game.game_run.levels_completed = 1                 # a clear no turn reported (e.g. a WIN on the last action)
agent._v67x_run_end()
check('5 the run\'s end with the level completed in the game run: cleared',
      [outcome(p)[:3] for p in events(log, 'v67-ext200-outcome')], [('extend', True, 'run_end')])

# === 6. the budget text ==================================================================================================
def block(t, T):
    return A._fill(A.TEXTS['loop_block'], n=10, p=2, k=5, a=2, A=10, t=t, T=T)


CYC = [(R_, 200000)] + [(L_, 100), (R_, 100)] * 4 + [(L_, 100), (R_ + '  # next', 100)]
for name, late, want in (('extend', False, '250,000'), ('late', True, '200,000')):
    game, (s1,), agent, log, go = h([copy.deepcopy(CYC)], frac=1.0, loop=True)
    go()
    if late:
        game.timing_payload = LATE
    for _ in range(10):
        go()
    check('6 %s arm: the loop-rule block (5th cycle report) states the budget %s' % (name, want),
          ([p['arm'] for p in events(log, 'v67-ext200')], user_text(s1.sent[10]).startswith(block('200,900', want) + '\n')),
          ([name], True))
conv = [(R_, 100), (L_, 100)] * 5 + [(R_ + '  # continue', 199500), (L_, 100), (L_ + '  # other', 100)]
game, (s1,), agent, log, go = h([conv], frac=1.0, loop=True)
for _ in range(12):
    go()
check('6 a block shown before 200K states 200,000; repeated in a refusal after the extension it states 250,000',
      (user_text(s1.sent[10]).startswith(block('1,000', '200,000') + '\n'), [p['arm'] for p in events(log, 'v67-ext200')],
       'This call is identical to call 2 of the detected cycle and was not executed.\n' + block('1,000', '250,000')
       in tool_errors(s1.sent[12]), '200,000 tokens' in tool_errors(s1.sent[12]), agent._v6_stats.get('loop_refused')),
      (True, ['extend'], True, False, 1))
RAW = json.load(open(Path(A.__file__).with_name('texts.json')))['texts']


def build(**env):
    keep = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        return V.apply(json.loads(json.dumps(RAW)))
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


ENVS = {'production': dict(ARC3_V67_LOOP='0'),
        'core': dict(ARC3_V67_LOOP='1', ARC3_V67_TEXTS='1', ARC3_V67_NOTES='1', ARC3_V67_SCORE_TEXT='1'),
        'core_refresh_soft': dict(ARC3_V67_LOOP='1', ARC3_V67_TEXTS='1', ARC3_V67_NOTES='1', ARC3_V67_SCORE_TEXT='1',
                                  ARC3_V67_REFRESH_LOGICAL='500000', ARC3_V67_RESTART_KIND='soft'),
        'clock': dict(ARC3_V65_CLOCK='1', ARC3_V67_LOOP='1')}
found = {}
for name, env in ENVS.items():
    t = build(**env)
    found[name] = ([k for k, v in t.items() if '{T}' in str(v)],
                   [k for k, v in t.items() if any(x in str(v) for x in ('200,000', '200000', '200K', '250,000', '250000'))])
check('6 no other text carries the budget: {T} only in loop_block, no literal budget (4 environments)', found,
      {'production': ([], []), 'core': (['loop_block'], []), 'core_refresh_soft': (['loop_block'], []), 'clock': (['loop_block'], [])})
SRC = open(A.__file__).read()
check('6 the only format of the budget into a text is the loop-rule block\'s, from the attempt\'s budget',
      [l.strip() for l in SRC.split('\n') if 'RESTART_TOKENS' in l and 'format(' in l],
      ["t=format(level_tokens, ','), T=format(self._v67x_budget(level) if EXT200 else RESTART_TOKENS, ','))"])
check('6 the no-action cues name the no-action limit, not the level budget', ('{limit}' in A.TEXTS['noact_limit_clause'],
      "limit = format(NOACT_TOKENS, ',') if NOACT_TOKENS else ''" in SRC), (True, True))

# === 7. not an assignment point; the no-action limit at the assignment turn ============================================
game, (s1, s2), agent, log, go = h([[(R_, 120000), (R_, 90000), (NOTES, 100)], [(R_, 100)]], frac=1.0)
go()
go()
s1.last_receipt = {'logical_tokens': A.POSITION_RESTART}
go()
go()
check('7 the position notes turn at 210K: no assignment; the position restart',
      (events(log, 'v67-ext200'), [p['reason'] for p in events(log, 'v61-restart')], ext_stats(agent)),
      ([], ['position'], (None,) * 5))
game, (s1, s2), agent, log, go = h([[(R_, 300), ('print(1)', 300), 'ERROR'], [(R_, 50)]], frac=1.0, budget=500)
for _ in range(3):
    go()
check('7 a session error after the budget was crossed inside a turn: the forced restart is no assignment',
      (events(log, 'v67-ext200'), [(p['reason'], p['tokens_on_level']) for p in events(log, 'v61-restart')],
       agent._v6_stats.get('protocol_restarts')), ([], [('tokens', 600)], 1))
game, (s1, s2), agent, log, go = h([[(R_, 120000), (R_, 90000), 'ERROR'], [(R_, 100)]], frac=1.0)
for _ in range(4):
    go()
check('7 a session error in the extension: the forced restart ends the attempt (reason forced)',
      ([p['arm'] for p in events(log, 'v67-ext200')], [outcome(p) for p in events(log, 'v67-ext200-outcome')],
       [p['reason'] for p in events(log, 'v61-restart')]), (['extend'], [('extend', False, 'restart', 'forced', 210000, 0, 0)],
                                                            ['forced']))
game, (s1,), agent, log, go = h([[(R_, 210000), (R_, 100)]], frac=1.0)
go()
cf, he = load_runtime_state(game.state_path)
agent._step_env_callback = game.step_env
blocked = []
for attr, value in (('_v62_summary_state', 'requested'), ('_v62_summary_state', 'done'), ('_v62_restart_reason', 'loop'),
                    ('_v62_restart_reason', 'noact'), ('_v6_started', False), ('_v61_level_tokens', 199999)):
    keep = getattr(agent, attr)
    setattr(agent, attr, value)
    agent._v67x_assign(1, cf, he)
    blocked.append((attr, value, bool(events(log, 'v67-ext200'))))
    setattr(agent, attr, keep)
agent._v67x_assign(1, cf, he)
agent._v67x_assign(1, cf, he)
agent._step_env_callback = None
check('7 no assignment with a notes turn or restart pending, before the first turn, or below the budget',
      [b for b in blocked if b[2]], [])
check('7 then one assignment; a second for the same attempt is impossible', len(events(log, 'v67-ext200')), 1)
game, (s1, s2), agent, log, go = h([[(R_, 140000), ('print(1)', 70000), (NOTES, 100)], [(R_, 100)]], frac=1.0, warn=())
for _ in range(4):
    go()
check('7 the no-action limit reached at the assignment turn: extended, then the no-action notes turn and restart',
      ([p['arm'] for p in events(log, 'v67-ext200')], [p['reason'] for p in events(log, 'v62-summary-turn')],
       [outcome(p) for p in events(log, 'v67-ext200-outcome')], agent._v6_stats.get('noact_restarts')),
      (['extend'], ['noact'], [('extend', False, 'restart', 'noact', 210100, 100, 0)], 1))

# === 8. switch-off identity with the deployed v6.7 base; the restart / late arms add receipts only =======================
IDENT = r'''
import hashlib
import json
import os
import sys
import tempfile  # noqa: F401  (the fakes' harness)
from pathlib import Path  # noqa: F401
from types import SimpleNamespace  # noqa: F401
os.environ.update(json.loads(sys.argv[1]))
import v62.patches  # noqa: E402,F401
import v62.retention as r  # noqa: E402
r.install()
import v62.variants as V  # noqa: E402
import v62.agent as A  # noqa: E402
exec(sys.argv[2])


def norm(x):
    if isinstance(x, dict):
        if x.get('type') == 'image_url':
            return dict(type='image_url', sha1=hashlib.sha1(json.dumps(x, sort_keys=True).encode()).hexdigest())
        return {k: norm(v) for k, v in x.items()}
    if isinstance(x, list):
        return [norm(v) for v in x]
    return x


def collect(game, sessions, agent, log):
    events = []
    for k, p in log:
        p = json.loads(json.dumps(p, default=str))
        if k == 'tool-output':
            p.pop('seconds', None)
        events.append([k, norm(p)])
    keys = ('v6_level_attempts', 'v61_level_attempt_noops', 'v62_level_attempt_ends', 'v6_attempt_actions', 'v62_since_start',
            'v62_level_start_returns', 'v61_harness_resets', 'v61_voluntary_resets', 'v67_level_restarts')
    return dict(sent=[norm(s.sent) for s in sessions], engine=game.engine, events=events,
                stats=json.loads(json.dumps(agent._v6_stats, default=str)),
                records=json.loads(json.dumps({k: game.__dict__.get(k) for k in keys})),
                library=sorted(agent._kept_functions), archive=sorted(agent._v61_archive), hidden=agent._v61_hidden)


def run(scripts, turns, before=None, **kw):
    game, sessions, agent, log, go = harness(scripts, **kw)
    for i in range(turns):
        if before and i in before:
            before[i](game, sessions, agent)
        go()
    return collect(game, sessions, agent, log)


R_, L_, U_ = 'action(["RIGHT"])', 'action(["LEFT"])', 'action(["UP"])'
NOTES_L = 'save_summary("Notes L1:\\n- RIGHT and LEFT move the marker one cell\\n- the calls repeated a cycle")'
SUM = 'save_summary("Rule L1:\\n- UP completes the level\\n- RIGHT moves the marker")\naction(["RIGHT"])'
NOTES2 = 'save_summary("Notes L2:\\n- RIGHT moves the marker on level 2\\n- UP clears a level")'
late = lambda g, ss, a: setattr(g, 'timing_payload', lambda: dict(time_remaining_seconds=600.0))   # noqa: E731
runs = {}
# a budget restart: a program, a death (the ledger), the notes turn with a refused action and its retry, the takeover
runs['budget'] = run([[(PROG + R_, 100), ('action(["DOWN"])', 100), ('action(["RIGHT", "RIGHT"])', 400), (L_, 50), (NOTES, 80)],
                      [(LOOK + R_, 50), (R_, 50)]], 6, budget=500)
# level-ups with summaries, a budget restart on level 2 with notes, a clear after it
runs['levels'] = run([[(PROG + R_, 40), (U_, 40), (SUM, 40), (R_, 500), (L_, 30), (NOTES2, 40)], [(R_, 30), (U_, 30), (R_, 30)]],
                     8, budget=500)
# the late window: the budget reached with 600 s left (the restart skipped), then a clear
runs['late'] = run([[(R_, 300), (R_, 300), (R_, 50), (U_, 50), (R_, 50)]], 5, budget=500, before={2: late})
# no-action cues and the no-action restart
runs['noact'] = run([[('print(1)', 150), ('print(2)', 100), ('print(3)', 100), ('print(4)', 100), (NOTES, 50)], [(R_, 50)]], 3,
                    warn=(100, 200, 300), limit=400)
# a session error in the notes turn after the budget; the forced restart
runs['forced'] = run([[(R_, 600), 'ERROR'], [(R_, 50)]], 3, budget=500)
# the position limit with the budget reached: the position notes turn and restart
runs['position'] = run([[(R_, 600), (NOTES, 80)], [(R_, 50), (R_, 50)]], 4, budget=500, pos_min=500,
                       before={1: lambda g, ss, a: setattr(ss[0], 'last_receipt', {'logical_tokens': A.POSITION_RESTART})})
# a cycling stretch (C1 lines; with LOOP on: the block with the budget sentence, two refusals, the loop soft restart)
runs['cycle'] = run([[(R_, 50), (L_, 50)] * 5 + [(R_, 40), (L_ + '   \n', 40), (NOTES_L, 60)], [(R_, 50)]], 12, budget=200000)
# two budget restarts with lazy history (the hidden-history texts)
real_lazy = A._lazy_enabled
A._lazy_enabled = lambda: True
runs['twice'] = run([[(R_, 600), (NOTES, 50)], [(R_, 600), (NOTES, 50)], [(R_, 50)]], 5, budget=500)
A._lazy_enabled = real_lazy
raw = json.load(open(os.path.join(os.path.dirname(A.__file__), 'texts.json')))['texts']
texts = {}
for name, env in (('production', {}), ('v65_batch', {'ARC3_V65_BATCH': '1', 'ARC3_STALE_STATE_BLOCK': '0'}),
                  ('control_sco0_seeds0', {'ARC3_V66_SCO': '0', 'ARC3_V66_SEEDS': '0'}), ('no_reset', {'EXPOSE_RESET': 'off'}),
                  ('clock', {'ARC3_V65_CLOCK': '1'}), ('soft', {'ARC3_V67_RESTART_KIND': 'soft'}),
                  ('refresh', {'ARC3_V67_REFRESH_LOGICAL': '500000'})):
    keep = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    texts[name] = V.apply(json.loads(json.dumps(raw)))
    for k, v in keep.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
print('IDENT-JSON ' + json.dumps(dict(runs=runs, texts=texts, agent_texts=A.TEXTS, tool=A.TOOL), sort_keys=True, default=str))
'''
CHANGED = ('v62/agent.py', 'v62/variants.py', 'v62/run_v6.py', 'v62/sandbox_ext.py')   # all files apply_v67x.py patches
missing = [f for f in CHANGED if not (ROOT / (f + '.pre-v67x')).exists()]
check('8 the deployed v6.7 base copies (v62/*.pre-v67x, written by apply_v67x.py) are present', missing, [])
if not missing:
    base = Path(tempfile.mkdtemp()) / 'v67_base'
    shutil.copytree(ROOT / 'v62', base / 'v62', ignore=shutil.ignore_patterns('__pycache__', '*.pre-*', '*.tmp'))
    for f in CHANGED:
        shutil.copyfile(ROOT / (f + '.pre-v67x'), base / f)
    env_clean = {k: v for k, v in os.environ.items() if not k.startswith('ARC3_V67_')}
    CORE = dict(ARC3_V67_LOOP='1', ARC3_V67_TEXTS='1', ARC3_V67_NOTES='1', ARC3_V67_SCORE_TEXT='1')
    VARIANTS = {'off': {}, 'core': CORE, 'core_on': dict(CORE, ARC3_V67_EXT200='1e-09')}

    def run_tree(job):
        tree, variant = job
        env = dict(env_clean, PYTHONPATH='%s:%s' % (ROOT / 'src', tree))
        p = subprocess.run([sys.executable, '-c', IDENT, json.dumps(dict(ENV, **VARIANTS[variant])), FAKES], env=env, cwd=str(tree),
                           capture_output=True, text=True, timeout=1500)
        line = [x for x in p.stdout.splitlines() if x.startswith('IDENT-JSON ')]
        if p.returncode or not line:
            print('identity run failed in %s %s (rc %s):\n%s' % (tree, variant, p.returncode, p.stderr[-3000:]))
            return None
        return json.loads(line[-1][len('IDENT-JSON '):])
    jobs = [(base, 'off'), (ROOT, 'off'), (base, 'core'), (ROOT, 'core'), (ROOT, 'core_on')]
    with ThreadPoolExecutor(len(jobs)) as pool:
        outs = dict(zip([(('base' if t == base else 'new'), v) for t, v in jobs], pool.map(run_tree, jobs)))
    check('8 the five identity runs completed', [k for k, v in outs.items() if v is None], [])
    CMP = ('sent', 'engine', 'events', 'stats', 'records', 'library', 'archive', 'hidden')
    if all(v is not None for v in outs.values()):
        for variant in ('off', 'core'):
            a, b = outs[('base', variant)], outs[('new', variant)]
            check('8 %s: the texts in %d environments, the agent\'s TEXTS and tool identical' % (variant, len(a['texts'])),
                  ([k for k in a['texts'] if a['texts'][k] != b['texts'][k]], a['agent_texts'] == b['agent_texts'],
                   a['tool'] == b['tool']), ([], True, True))
            for name in a['runs']:
                check('8 %s %s: traffic, actions, events, stats, records, library identical' % (variant, name),
                      [k for k in CMP if a['runs'][name][k] != b['runs'][name][k]], [])
            check('8 %s: no ext200 event or stat with the switch off' % variant,
                  [n for n, x in b['runs'].items() if any(e[0] in EXT_EVENTS for e in x['events'])
                   or any(k.startswith('ext200_') for k in x['stats'])], [])
        a, b = outs[('base', 'core')], outs[('new', 'core_on')]
        check('8 core_on (fraction 1e-9): the texts identical', ([k for k in a['texts'] if a['texts'][k] != b['texts'][k]],
                                                                 a['agent_texts'] == b['agent_texts']), ([], True))
        for name in a['runs']:
            x, y = a['runs'][name], dict(b['runs'][name])
            y['events'] = [e for e in y['events'] if e[0] not in EXT_EVENTS]
            y['stats'] = {k: v for k, v in y['stats'].items() if not k.startswith('ext200_')}
            check('8 core_on %s: identical to the base apart from the ext200 events and stats' % name,
                  [k for k in CMP if x[k] != y[k]], [])
        got = {n: ([(p['arm'], p['level'], p['attempt']) for k, p in x['events'] if k == 'v67-ext200'],
                   [(p['arm'], p['cleared'], p['ended'], p['reason']) for k, p in x['events'] if k == 'v67-ext200-outcome'])
               for n, x in b['runs'].items()}
        check('8 core_on: the assignments and outcomes the runs should have (restart / late arms only)', got, {
            'budget': ([('restart', 1, 1)], [('restart', False, 'restart', 'tokens')]),
            'levels': ([('restart', 2, 1)], [('restart', False, 'restart', 'tokens')]),
            'late': ([('late', 1, 1)], [('late', True, 'clear', None)]),
            'noact': ([], []), 'forced': ([('restart', 1, 1)], [('restart', False, 'restart', 'tokens')]),
            'position': ([], []), 'cycle': ([], []),
            'twice': ([('restart', 1, 1), ('restart', 1, 2)], [('restart', False, 'restart', 'tokens')] * 2)})
        rn = outs[('new', 'core')]['runs']
        check('8 the runs took the intended paths (restarts; the late skip; the loop block, refusals and loop restart)',
              ([rn[n]['stats'].get('restarts') for n in ('budget', 'levels', 'late', 'noact', 'forced', 'position', 'cycle', 'twice')],
               rn['late']['stats'].get('restarts_skipped_late'), rn['cycle']['stats'].get('loop_rung1'),
               rn['cycle']['stats'].get('loop_refused'), rn['cycle']['stats'].get('loop_soft_restarts'),
               rn['budget']['stats'].get('notes_retries'), rn['noact']['stats'].get('noact_restarts'),
               rn['position']['stats'].get('position_notes_turns'), rn['forced']['stats'].get('protocol_restarts')),
              ([1, 1, None, 1, 1, 1, 1, 2], 1, 1, 2, 1, 1, 1, 1, 1))
print('FAILS:', fails)
