"""test_v67_loop.py (2026-10-18, Claude 099768dc): apply_v67.py's item LOOP (ARC3_V67_LOOP=1, the loop ladder), under the HARD
restart switch (the default), through agent.analyze with the real sandbox on test_v67_soft.py's fakes. IE's conditions:
  1 detector units: periods 1-8 (C1 keeps 4), k >= 3, the consecutive count (5 / 8), the reset rules, resumptions;
  2 the rungs on a scripted session: C1's line, the block at the 5th (once per episode), the refusal of an identical call
    (trailing whitespace aside; different code runs), the 2nd refused repeat -> the notes turn mid-turn -> a soft restart
    from the current board although ARC3_V67_RESTART_KIND=hard, with the loop opener line; other restarts stay hard;
  3 rung 2 at the 8th consecutive (the notes turn at the opener), then rung 3 (hard, RESET) when it recurs within 200K;
  4 caps: rung 3 after 2 loop soft restarts; parking at the 2nd loop hard restart (the lowest priority, also re-scored);
  5 receipts and events; the LOOPS stdout lines and their wiring in run_v6;
  the switch off is byte-identical to v66g: tests/test_v67_ident.py (subprocess on the v66g base) and the suite.
usage (dev copy of harness_v6): PYTHONPATH=src:. python3 tests/test_v67_loop.py"""
import copy
import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace

_src = open(Path(__file__).with_name('test_v67_soft.py')).read()
ENV = eval(_src.split('ENV = ', 1)[1].split('\n\n', 1)[0])
FAKES = _src.split("FAKES = r'''", 1)[1].split("'''", 1)[0]
os.environ.update(ENV)
os.environ.pop('ARC3_V67_RESTART_KIND', None)            # the hard switch: rung 2 must still be soft
os.environ['ARC3_V67_LOOP'] = '1'
import v62.patches  # noqa: E402,F401
import v62.retention as r  # noqa: E402
r.install()
import v62.variants as V  # noqa: E402
import v62.agent as A  # noqa: E402
from inference.agent.runtime_state import load_runtime_state  # noqa: E402
exec(compile(FAKES, 'test_v67_soft.py (FAKES)', 'exec'), globals())
fails = []


def check(name, got, want):
    ok = got == want
    print('%-84s -> %-40r %s' % (name, got if not isinstance(got, str) else got[:40], 'OK' if ok else 'WRONG (want %r)' % (want,)))
    if not ok:
        fails.append(name)


def user_text(batch):
    m = [x for x in batch if x.get('role') == 'user'][-1]
    c = m['content']
    return c if isinstance(c, str) else '\n'.join(p.get('text', '') for p in c if p.get('type') == 'text')


def tool_text(batch):
    return '\n'.join(m['content'] if isinstance(m['content'], str) else
                     '\n'.join(p.get('text', '') for p in m['content'] if p.get('type') == 'text')
                     for m in batch if m.get('role') == 'tool')


def events(log, kind):
    return [p for k, p in log if k == kind]


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


RAW = json.load(open(Path(A.__file__).with_name('texts.json')))['texts']
R_, L_ = 'action(["RIGHT"])', 'action(["LEFT"])'

# === 0. switch and texts ================================================================================================
check('0 switch on, the hard restart switch, the knobs', (A.LOOP, A.RESTART_KIND, A.LOOP_PMAX, A.LOOP_RUNG1, A.LOOP_RUNG2,
      A.LOOP_HARD_TOKENS, A.LOOP_SOFT_CAP, A.LOOP_PARK_AT, A.LOOP_REPORT_S), (True, 'hard', 8, 5, 8, 200000, 2, 2, 1800.0))
off = build(ARC3_V67_LOOP='0')
new = sorted(set(A.TEXTS) - set(off))
check('0 LOOP adds its texts and the soft texts, nothing else changes',
      (new, {k: v for k, v in A.TEXTS.items() if k not in new} == off),
      (sorted(['attempt_summary_request_loop', 'attempt_summary_request_loop_again', 'attempt_summary_request_loop_hard',
               'loop_block', 'loop_refused', 'takeover_loop', 'attempt_summary_request_noact_soft',
               'attempt_summary_request_position_soft', 'attempt_summary_request_soft', 'noact_limit_clause_soft',
               'noact_warn_2_soft', 'summary_turn_reason_soft', 'summary_turn_refusal_soft', 'takeover_soft',
               'takeover_soft_history', 'takeover_soft_history_cut']), True))
check('0 the loop requests keep the notes instructions (soft remainder for rung 2, hard for rung 3)',
      (A.TEXTS['attempt_summary_request_loop'].endswith('Your programs stay loaded, including those saved on this level.'),
       A.TEXTS['attempt_summary_request_loop_again'].endswith('rather than relying on them.'),
       'Notes L{level}:' in A.TEXTS['attempt_summary_request_loop']), (True, True, True))

# === 1. detector units ==================================================================================================
blocks = {p: [(('A%d' % i,), 'report %d' % i) for i in range(p)] for p in range(1, 9)}
check('1 periods 1-8 with k = 3 found at pmax 8', [A._v65_loop(blocks[p] * 3, 8) for p in range(1, 9)],
      [(p, 3) for p in range(1, 9)])
check('1 two repeats are not a cycle', [A._v65_loop(blocks[p] * 2, 8) for p in range(1, 9)], [(0, 0)] * 8)
check('1 C1 (pmax 4) misses periods 5-8', [A._v65_loop(blocks[p] * 3) for p in range(5, 9)], [(0, 0)] * 4)
check('1 smallest period wins (p 2 inside p 4)', A._v65_loop(blocks[2] * 6, 8), (2, 6))


def drive(agent, game, seq, kinds=None):
    """feed (acts, report, code) items through _v67_loop_line; returns per item (consecutive, block?, C1 line?, notes?)"""
    cf, he = load_runtime_state(game.state_path)
    agent._step_env_callback = game.step_env
    out = []
    for i, (acts, report, code) in enumerate(seq):
        kind = (kinds or {}).get(i, 'same')
        agent._last_step_summary = dict(executed_actions=list(acts))
        agent._v67_last_code = code
        base = report + '\nStep 9 (actions in the whole run), level 1, attempt 1. Valid actions: RIGHT.\n' + A.TEXTS['reminder']
        text, notes = agent._v67_loop_line(base, base, kind, False, 1, game.step_env, cf, he)
        st = agent._v67_loop
        out.append((st['consecutive'], text.startswith('Loop rule (harness):'), 'Python calls' in text.split('\nStep')[0]
                    and not text.startswith('Loop rule'), notes))
    agent._step_env_callback = None
    return out


game, (sess,), agent, log, go = harness([[(R_, 10)]], budget=200000)
p6 = [(('A%d' % i,), 'report %d' % i, 'code %d' % i) for i in range(6)]
res = drive(agent, game, p6 * 4 + p6[:1])            # 25 signatures, period 6: cycle reports from the 18th
cons = [c for c, _, _, _ in res]
check('1 period 6: the count starts at the 18th signature (3 blocks) and runs 1..8', cons[16:], [0, 1, 2, 3, 4, 5, 6, 7, 8])
check('1 rung 0 (C1 line) at 1-4 and 6-7, the block at the 5th only', [(c, b, l) for c, b, l, _ in res[17:24]],
      [(1, False, True), (2, False, True), (3, False, True), (4, False, True), (5, True, False), (6, False, True),
       (7, False, True)])
check('1 the 8th: the notes turn (rung 2) at the opener; soft under the hard switch',
      (res[24][3], agent._v67_pending.get('kind'), agent._v67_pending.get('loop', {}).get('rung'), agent._v62_restart_reason,
       agent._v62_summary_state), (True, 'soft', 2, 'loop', 'requested'))
game, (sess,), agent, log, go = harness([[(R_, 10)]], budget=200000)
odd = (('Z',), 'something else', 'z')
res = drive(agent, game, p6 * 3 + p6[:3] + [odd] + p6[3:] + p6 * 3, kinds={22: 'resume'})
cons = [c for c, _, _, _ in res]
check('1 a non-cycle turn resets the count; a resumption neither counts nor resets', (cons[17:22], cons[22], cons[23]),
      ([1, 2, 3, 4, 0], 0, 0))
game, (sess,), agent, log, go = harness([[(R_, 10)]], budget=200000)
res = drive(agent, game, p6 * 3 + p6[:2] + [p6[2]] + p6[3:], kinds={20: 'takeover'})
check('1 a takeover (new conversation) resets the detector', [c for c, _, _, _ in res][17:], [1, 2, 3, 0, 0, 0, 0])

# === 2. the rungs on a scripted session: rung 1, the refusal, the 2nd refused repeat -> soft restart (hard switch) =========
NOTES_L = 'save_summary("Notes L1:\\n- RIGHT and LEFT move the marker one cell\\n- the calls repeated a cycle")'
conv1 = [(R_, 50), (L_, 50)] * 5 + [(R_, 40), (L_ + '   \n', 40), (NOTES_L, 60)]
game, (sess, sess2), agent, log, go = harness([conv1, [(R_, 50)]], budget=200000)
for _ in range(12):
    go()
opener11 = user_text(sess.sent[10])
block = A._fill(A.TEXTS['loop_block'], n=10, p=2, k=5, a=2, A=10, t='500', T='200,000')
check('2 rung 1 at the 5th cycle report: the block opens the report', opener11.startswith(block + '\n'), True)
check('2 C1 lines before it (openers 7-10), with the period up to 8', [('repeated the same cycle of 2 calls' in user_text(sess.sent[i]))
                                                                       for i in range(6, 10)], [True] * 4)
ref1, ref2 = tool_text(sess.sent[11]), tool_text(sess.sent[12])
check('2 the identical call (and the whitespace-padded one) not executed; the block repeated',
      ('This call is identical to call 1 of the detected cycle and was not executed.' in ref1,
       'This call is identical to call 2 of the detected cycle and was not executed.' in ref2, 'Loop rule (harness):' in ref1),
      (True, True, True))
req = user_text(sess.sent[12])
check('2 the 2nd refused repeat: the loop notes request in the same turn',
      req, A._fill(A.TEXTS['attempt_summary_request_loop'], n=10, p=2, level=1))
first = user_text(sess2.sent[0])
check('2 the restart: soft (no RESET) under the hard switch, the notes and the loop opener line',
      (game.engine.count('RESET'), len(game.engine), [(p['reason'], p['kind']) for p in events(log, 'v61-restart')],
       A._fill(A.TEXTS['takeover_loop'], level=1, step=10, n=10, p=2, count=10) in first,
       '- RIGHT and LEFT move the marker one cell' in first, A.TEXTS['takeover_soft_history'] in first),
      (0, 11, [('loop', 'soft')], True, True, True))
check('2 receipts', tuple(agent._v6_stats.get(k) for k in ('loop_reports', 'loop_episodes', 'loop_rung1', 'loop_refused',
      'loop_soft_restarts', 'loop_hard_restarts', 'loop_cycle_actions', 'loop_max_consecutive', 'summary_turns')),
      (5, 1, 1, 2, 1, None, 5, 5, 1))
check('2 events: v67-loop rungs 0,0,0,0,1,2; two refusals; the soft restart event',
      ([p['rung'] for p in events(log, 'v67-loop')], [(p['call'], p['refused'], p['p']) for p in events(log, 'v67-loop-refused')],
       [(p['reason'], p['level'], p['step']) for p in events(log, 'v67-soft-restart')]),
      ([0, 0, 0, 0, 1, 2], [(1, 1, 2), (2, 2, 2)], [('loop', 1, 10)]))
check('2 the game record carries the loop', [(x['kind'], x['reason'], x['loop']['rung'], x['loop']['n'], x['loop']['p'])
                                             for x in game.v67_level_restarts[1]], [('soft', 'loop', 2, 10, 2)])
check('2 the new conversation starts a fresh detector and budget', (agent._v67_loop['consecutive'], len(agent._v67_loop['sigs']),
      agent._v61_level_tokens), (0, 0, 50))
# different code runs; trailing whitespace is stripped
conv = [(R_, 50), (L_, 50)] * 5 + [(R_ + '  # probe', 40), (L_ + '\t\n\n', 40), (L_ + '  # other', 40), (NOTES_L, 40)]
game, (sess, sess2), agent, log, go = harness([conv, [(R_, 50)]], budget=200000)
for _ in range(12):
    go()
check('2 different code runs (a comment), a whitespace-padded copy is refused, then other code runs',
      (game.engine[10:], tool_text(sess.sent[12]).count('was not executed'), agent._v6_stats.get('loop_refused')),
      (['RIGHT', 'LEFT'], 1, 1))
# the other restarts stay hard under the hard switch
game, (sess, sess2), agent, log, go = harness([[(R_, 600), (NOTES_L, 40)], [(R_, 50)]], budget=500)
for _ in range(3):
    go()
check('2 a budget restart with LOOP on, hard switch: hard (RESET), no soft line',
      ([(p['reason'], p['kind']) for p in events(log, 'v61-restart')], game.engine.count('RESET'),
       'continues level' in user_text(sess2.sent[0])), ([('tokens', 'hard')], 1, False))

# === 3. rung 2 at the 8th consecutive (no repeat of identical code), then rung 3 (hard) within 200K =======================
cyc = lambda tag: [('%s  # %s%d' % (R_ if i % 2 == 0 else L_, tag, i), 50) for i in range(13)]
game, (sess, sess2, sess3), agent, log, go = harness([cyc('a') + [(NOTES_L, 40)], cyc('b') + [(NOTES_L, 40)], [(R_, 50)]],
                                                     budget=200000)
for _ in range(14):
    go()
op14 = user_text(sess.sent[13])
check('3 the 8th consecutive: the opener carries C1\'s line and the loop notes request (no refusal: the code differs)',
      ('repeated the same cycle of 2 calls' in op14, A._fill(A.TEXTS['attempt_summary_request_loop'], n=12, p=2, level=1) in op14,
       agent._v6_stats.get('loop_refused'), agent._v6_stats.get('loop_rung1')), (True, True, None, 1))
for _ in range(15):
    go()
op14b = user_text(sess2.sent[13])
check('3 the loop again within 200K tokens: rung 3, the hard request, a RESET, the hard takeover',
      (A._fill(A.TEXTS['attempt_summary_request_loop_again'], n=12, p=2, level=1) in op14b,
       [(p['reason'], p['kind'], p['hard_why']) for p in events(log, 'v61-restart')], game.engine.count('RESET'),
       'This conversation begins after a RESET of level 1' in user_text(sess3.sent[0])),
      (True, [('loop', 'soft', None), ('loop', 'hard', None)], 1, True))
check('3 receipts', tuple(agent._v6_stats.get(k) for k in ('loop_episodes', 'loop_rung1', 'loop_soft_restarts',
                                                            'loop_hard_restarts', 'loop_parks')), (2, 2, 1, 1, None))
check('3 v67-loop trigger events: rung 2 then 3 (since = tokens after the soft restart)',
      [(p['rung'], p.get('since')) for p in events(log, 'v67-loop') if p['rung'] >= 2], [(2, None), (3, 650)])

# === 4. caps: rung 3 after 2 loop soft restarts; parking at the 2nd loop hard restart ====================================
def trigger_with(records, level_tokens):
    game, (sess,), agent, log, go = harness([[(R_, 10)]], budget=200000)
    go()
    game.__dict__['v67_level_restarts'] = {1: copy.deepcopy(records)}
    agent._v61_level_tokens = level_tokens
    cf, he = load_runtime_state(game.state_path)
    agent._step_env_callback = game.step_env
    req = agent._v67_loop_trigger(1, game.step_env, cf, he, dict(n=10, p=2, k=5))
    agent._step_env_callback = None
    return agent._v67_pending['loop']['rung'], agent._v67_pending['kind'], req.split(',')[0]


soft = dict(kind='soft', reason='loop', tokens=300000)
check('4 rung 2 without a loop soft restart', trigger_with([], 300000)[:2], (2, 'soft'))
check('4 rung 2 when the loop soft restart was over 200K tokens ago', trigger_with([soft], 250000)[:2], (2, 'soft'))
check('4 rung 3 within 200K tokens of a loop soft restart', trigger_with([soft, dict(kind='hard', reason='tokens', tokens=50000)],
                                                                         100000)[:2], (3, 'hard'))
check('4 rung 3 after 2 loop soft restarts (cap)', trigger_with([soft, soft], 900000)[:2], (3, 'hard'))


class Gate:
    _waiting, _clock_deadline = [], None

    def __init__(self):
        self.calls = []

    def handover(self, priority, snapshot):
        self.calls.append((priority, snapshot))


for prior, want in ((0, (False, None)), (1, (True, 1))):
    game, (sess,), agent, log, go = harness([[(R_, 10)]], budget=200000)
    go()
    gate = Gate()
    A.ta._priority_gate = lambda: gate
    game.__dict__['v67_level_restarts'] = {1: [dict(kind='hard', reason='loop', tokens=1000)] * prior}
    agent._v67_pending = dict(level=1, kind='hard', why=None, loop=dict(rung=3, n=10, p=2, k=5))
    agent._step_env_callback = game.step_env
    agent._v62_swap_out(game.history_entries[-1].frame, 'loop')
    agent._step_env_callback = None
    A.ta._priority_gate = lambda: None
    prio, snap = gate.calls[-1]
    rescored = A._gate_snapshot_priority(SimpleNamespace(_clock_deadline=None), snap, 0.0)
    swap = events(log, 'v62-swap')[-1]
    check('4 park with %d earlier loop hard restart(s): parked, lowest priority (also re-scored), band, receipt' % prior,
          (snap.parked, agent._v6_stats.get('loop_parks')) + ((prio, rescored, swap['band']) if prior else ()),
          want + ((-1_000_000_000, -1_000_000_000, 'parked') if prior else ()))

# === 5. the LOOPS lines =================================================================================================
stats = {'g1': dict(loop_episodes=2, loop_rung1=1, loop_refused=2, loop_soft_restarts=1, loop_cycle_actions=40, loop_reports=9),
         'g2': dict(loop_episodes=1, loop_reports=3, loop_cycle_actions=3), 'g3': dict(turns=5)}
tot, games = A.loops_totals(stats)
check('5 totals and the games with a rung 1-3 event', (tot, sorted(games)), (dict(loop_episodes=3, loop_rung1=1, loop_refused=2,
      loop_soft_restarts=1, loop_hard_restarts=0, loop_cycle_actions=43), ['g1']))
check('5 the stdout line', A.loops_line(tot, 3725), 'LOOPS t=01:02: episodes 3, rung1 1, refused 2, soft 1, hard 0, cycle actions 43')
check('5 a game line', A.loops_game_line('g1', games['g1']),
      'LOOPS game g1: episodes 2, rung1 1, refused 2, soft 1, hard 0, cycle actions 40')
RUN = open(Path(A.__file__).with_name('run_v6.py')).read()
check('5 run_v6: default, record, the 30-min reporter thread, v6-stats LOOPS, the last stdout line',
      ("os.environ.setdefault('ARC3_V67_LOOP', '0')" in RUN, "'ARC3_V67_LOOP'," in RUN,
       "threading.Thread(target=loops_reporter, name='v67-loops', daemon=True).start()" in RUN,
       "stats['LOOPS'] = dict(totals=loops_tot, games=loops_games)" in RUN,
       RUN.rstrip().split("print('Saved:'")[1].count('loops_line') == 1), (True,) * 5)
print('FAILS:', fails)
