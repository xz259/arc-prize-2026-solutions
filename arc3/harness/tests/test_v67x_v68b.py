"""test_v67x_v68b.py (2026-10-18, Claude 099768dc, implementation subagent): apply_v67x.py's items V68B (the LOOP_FRAME fix agreed
by IE after the loop audit review67_loops) and V68P (ARC3_V68_REFRESH_PROGRAMS, built off), through agent.analyze with the real sandbox on test_v67_soft.py's fakes, in the planned
run's shape (LOOP with a hard rung 2, every V68 switch but NOTES_NEG on, LOOP_FRAME on, the fallback at its default 16):
  0 the knob: ARC3_V68_LOOP_FALLBACK defaults to 16 with LOOP_FRAME and 0 without; it refuses without LOOP_FRAME or at or below
    the rung-2 count; run_v6 sets and records it; _v68_ends_ok on synthetic boards (identical ends, period 2 at k >= 4 only,
    an entry board that differs, a missing board), against the V68 starts form;
  1 a true cycle escalates at the 8th consecutive report (the count path; no refusal, nothing blocked);
  2 a period-2 alternation (the interior toggles every repetition; p 1) escalates at 8; the V68 starts form fails on its boards;
  3 an entry board that differs while the ends are identical escalates (the refusal path at the 5th report, where the window
    begins at the entry board, as sf10f g533 L4); the V68 starts form fails on its boards;
  4 a band-counter-only change escalates at 8 (whole boards differ, interiors equal);
  5 a walk (new interiors, identical normalised reports, identical code): neither refused nor escalated through 14 consecutive
    reports; the receipts loop_refusal_frame_blocked (every identical call after the block) and loop_frame_blocked (8..14);
  6 a repeated-death walk (RIGHT then DOWN every call, new interiors): the fallback at the 16th consecutive report, a hard loop
    restart (RESET, new conversation), the v68-loop-fallback event with p, k, consecutive, level, attempt and 16 GAME OVERs;
  7 a refused then frame-blocked episode keeps counting: one refusal while the condition holds (a period-2 stall), then a walk
    with identical calls executed; the fallback at 16;
  8 LOOP_FRAME and REFRESH_PROGRAMS off: identical to the tree after V68 and V68C (V68C: the loop block's text under
    LOOP_TEXTS; rebuilt from the base by apply_v67x.py's items through V68 plus V68C, md5-checked), a subprocess on each tree (test_v67x_v68.py's identity script), with every switch off and with
    every V68 switch but LOOP_FRAME on in the planned run's base config (a refresh included);
  9 REFRESH_PROGRAMS (item V68P; the user's selection at 18:02 UTC, the final rule from the refresh reviewer's data): off by
    default, refuses without the refresh or below 0, defaults FREQ 0 / RECENT 8; the selection on a synthetic library
    (recency by the last call, or the last writing if never called; a rewrite after a call does not count; same-turn ties by
    the call order; FREQ > 0 by count with ties to the more recent; nothing left out; full ties; empty); the counting (word
    boundary, def lines, turn and sequence); on: the selection right after the count line in the refresh opener (a program
    never called; a called one; the most-called header with FREQ 1); off: no selection, the opener otherwise identical.
usage (dev copy of harness_v6): PYTHONPATH=src:. python3 -B tests/test_v67x_v68b.py"""
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
IDENT = open(Path(__file__).with_name('test_v67x_v68.py')).read().split("IDENT = r'''", 1)[1].split("'''", 1)[0]
os.environ.update(ENV)
for _k in [k for k in os.environ if k.startswith(('ARC3_V67_', 'ARC3_V68_'))]:
    os.environ.pop(_k)
PLANNED = dict(ARC3_V67_LOOP='1', ARC3_V67_LOOP_RUNG2_KIND='hard', ARC3_V67_REFRESH_LOGICAL='250000', ARC3_V67_SUMLOOKUP='1')
V68_ON = dict(ARC3_V68_LEDGER='1', ARC3_V68_NEVER_TYPES='1', ARC3_V68_LOOP_TEXTS='1', ARC3_V68_LOOP_FRAME='1',
              ARC3_V68_REFRESH_TEXTS='1', ARC3_V68_NO_SCORE='1')
os.environ.update(PLANNED)
os.environ.update(V68_ON)
import v62.patches  # noqa: E402,F401
import v62.retention as r  # noqa: E402
r.install()
import v62.variants as V  # noqa: E402,F401
import v62.agent as A  # noqa: E402
exec(compile(FAKES, 'test_v67_soft.py (FAKES)', 'exec'), globals())
fails = []
ROOT = Path(A.__file__).resolve().parents[1]


def check(name, got, want):
    ok = got == want
    print('%-88s -> %-40r %s' % (name, got if not isinstance(got, str) else got[:40], 'OK' if ok else 'WRONG (want %r)' % (want,)))
    if not ok:
        fails.append(name)


def texts_of(batch, role=None):
    out = []
    for m in batch:
        if role and m.get('role') != role:
            continue
        c = m.get('content')
        if m.get('role') == 'tool' and isinstance(c, str):
            try:
                c = json.loads(c).get('error') or ''
            except (TypeError, ValueError, AttributeError):
                pass
        out.append(c if isinstance(c, str) else '\n'.join(p.get('text', '') for p in (c or []) if p.get('type') == 'text'))
    return '\n'.join(out)


def events(log, kind):
    return [p for k, p in log if k == kind]


R_, L_ = 'action(["RIGHT"])', 'action(["LEFT"])'
RD = 'action(["RIGHT", "DOWN"])'
NOTES_L = 'save_summary("Notes L1:\\n- RIGHT and LEFT move the marker one cell\\n- the calls repeated a cycle")'
real_board = board
MODE, CLOCK = ['plain'], [0]


def board(pos):  # noqa: F811  (the fakes' board, per scenario mode; CLOCK counts the boards drawn)
    m = MODE[0]
    if m == 'toggle':                                   # the interior alternates between two boards (marker at column 10 / 12)
        g = [[0] * 64 for _ in range(64)]
        g[30][10 + 2 * (pos % 2)] = 3
    else:
        g = real_board(pos)
    if m == 'clock':                                    # an interior pixel that moves at every board drawn: every board new
        g[50][10 + (CLOCK[0] % 40)] = 7
    if m == 'entry' and CLOCK[0] == 0:                  # the initial board only: one more interior cell (the entry board)
        g[30][50] = 4
    if m == 'band':                                     # a pixel in the outer band that moves at every board drawn
        g[0][CLOCK[0] % 64] = 5
    CLOCK[0] += 1
    return g


class Auto:
    """A FakeSession that answers by content: the loop rule's notes request -> save_summary; else code_fn(self, external)."""

    def __init__(self, code_fn):
        self.code_fn, self.sent, self.n = code_fn, [], 0
        self.rebased_away, self.blocks, self.closed, self.last_receipt = set(), [], False, {}

    def generate(self, external):
        self.sent.append(json.loads(json.dumps(external, default=str)))
        code = NOTES_L if 'restarts after this turn' in texts_of(external, 'user') else self.code_fn(self, external)
        self.n += 1
        calls = [dict(id='call-%d' % len(self.sent), type='function', function=dict(name='python', arguments=json.dumps(dict(code=code))))]
        msg = dict(role='assistant', content='', reasoning_content='thinking', tool_calls=calls)
        return Completion(message=msg, finish_reason='tool_calls', usage=dict(prompt_tokens=10, completion_tokens=50),
                          token_ids=[1] * 50, assistant_token_ids=[1] * 50, pending=False, closed_by=None, persistent={},
                          seconds=0.0)

    def span_evicted(self, a, b):
        return 0

    def close(self):
        self.closed = True


SNAP = []
_trigger = A.V6Agent._v67_loop_trigger


def _snap_trigger(self, level, step_env, current_frame, history_entries, cyc=None, fallback=False):
    st = self.__dict__.get('_v67_loop') or {}
    SNAP.append(dict(starts=list(st.get('starts') or []), frame=st.get('frame'), sigs=list(st.get('sigs') or []),
                     consecutive=st.get('consecutive'), fallback=fallback))
    return _trigger(self, level, step_env, current_frame, history_entries, cyc, fallback)


A.V6Agent._v67_loop_trigger = _snap_trigger


def scenario(mode, code_fn, turns):
    """One game on FakeGame with an Auto session (the first conversation) and a FakeSession for the next; runs until the second
    conversation has a request or for `turns` turns. Returns (game, agent, log, first, second)."""
    MODE[0], CLOCK[0] = mode, 0
    del SNAP[:]
    A.ta._priority_gate = lambda: None
    A.RESTART_TOKENS, A.RESTART_NOTES, A.POSITION_NOTES_MIN = 200000, True, 50000
    A.NOACT_WARN, A.NOACT_TOKENS, A.REFRESH = (16000, 32000, 48000), 64000, False
    state = Path(tempfile.mkdtemp()) / 'game_runtime_state.json'
    game = FakeGame(state)
    log, first, second = [], Auto(code_fn), FakeSession([(R_, 50)] * 4)
    queue = [second]
    agent = A.V6Agent(session=first, logger=lambda k, p: log.append((k, p)), count_tokens=lambda t: max(1, len(t) // 4),
                      model='M', base_url='http://x/v1', provider='vllm', session_factory=lambda: queue.pop(0))
    first.agent = agent
    for _ in range(turns):
        game.auto_reset()
        agent.analyze(state, len(game.history_entries) - 1, valid_actions=VALID, step_env=game.step_env, should_stop=lambda: False)
        if second.sent:
            break
    return game, agent, log, first, second


def rung2(log):
    return [(p.get('rung'), p.get('consecutive'), p.get('kind'), p.get('fallback')) for p in events(log, 'v67-loop') if p.get('rung') in (2, 3)]


def restarts(log):
    return [(p.get('reason'), p.get('kind')) for p in events(log, 'v61-restart')]


def loopstats(agent):
    return {k: agent._v6_stats.get(k) for k in ('loop_refused', 'loop_frame_blocked', 'loop_refusal_frame_blocked', 'loop_fallback')}


def starts_form(starts, p, k):
    """the V68 form (as built before V68B): the boards at the START of each of the last k repetitions identical"""
    n = len(starts)
    heads = [starts[n - k * p + i * p] for i in range(k)] if p > 0 and k >= 2 and n >= k * p else []
    return bool(heads) and heads[0] is not None and all(h == heads[0] for h in heads)


def at_trigger(snap):
    """(p, k, the V68 starts form, the corrected form) on a trigger's snapshot"""
    p, k = A._v65_loop(snap['sigs'], A.LOOP_PMAX)
    return p, k, starts_form(snap['starts'], p, k), A._v68_ends_ok(snap['starts'], snap['frame'], p, k)


NONE4 = dict(loop_refused=None, loop_frame_blocked=None, loop_refusal_frame_blocked=None, loop_fallback=None)

# === 0. the knob and the condition ==========================================================================================
check('0 this process: LOOP_FRAME on, the fallback at its default 16', (A.LOOP_FRAME68, A.LOOP_FALLBACK68), (True, 16))
base_env = dict(os.environ, PYTHONPATH='%s:%s' % (ROOT / 'src', ROOT), PYTHONDONTWRITEBYTECODE='1')


def probe(**env):
    e = dict(base_env)
    for k, v in env.items():
        if v is None:
            e.pop(k, None)
        else:
            e[k] = v
    q = subprocess.run([sys.executable, '-B', '-c', 'import v62.patches, v62.retention as r; r.install(); import v62.agent as A; '
                        'print("FALLBACK", A.LOOP_FALLBACK68)'], env=e, cwd=str(ROOT), capture_output=True, text=True, timeout=300)
    out = [x for x in q.stdout.splitlines() if x.startswith('FALLBACK ')]
    return out[-1] if out else 'ERROR ' + (q.stderr.strip().splitlines() or ['?'])[-1][:120]


check('0 the fallback: 0 without LOOP_FRAME; refuses without LOOP_FRAME, at 8, at -1; 0 and 9 accepted',
      (probe(ARC3_V68_LOOP_FRAME=None), probe(ARC3_V68_LOOP_FRAME=None, ARC3_V68_LOOP_FALLBACK='16').startswith('ERROR RuntimeError: ARC3_V68_LOOP_FALLBACK needs'),
       probe(ARC3_V68_LOOP_FALLBACK='8').startswith('ERROR RuntimeError: ARC3_V68_LOOP_FALLBACK must be 0 or exceed'),
       probe(ARC3_V68_LOOP_FALLBACK='-1').startswith('ERROR RuntimeError'), probe(ARC3_V68_LOOP_FALLBACK='0'),
       probe(ARC3_V68_LOOP_FALLBACK='9')),
      ('FALLBACK 0', True, True, True, 'FALLBACK 0', 'FALLBACK 9'))
run_src = open(ROOT / 'v62' / 'run_v6.py').read()
check('0 run_v6: the explicit default (16 with LOOP_FRAME, else 0) and the record',
      ("os.environ.setdefault('ARC3_V68_LOOP_FALLBACK', '16' if os.environ.get('ARC3_V68_LOOP_FRAME') == '1' else '0')" in run_src,
       run_src.count("'ARC3_V68_LOOP_FALLBACK',\n")), (True, 1))
X, Y, Z = 'x', 'y', 'z'
check('0 _v68_ends_ok: identical ends; the entry board ignored; period 2 at k >= 4 only; a missing end; too few starts',
      (A._v68_ends_ok([Z, X, X, X], X, 1, 4), starts_form([Z, X, X, X], 1, 4), A._v68_ends_ok([X, Y, X, Y], X, 1, 4),
       A._v68_ends_ok([X, Y, X], Y, 1, 3), A._v68_ends_ok([X, Y, X, Y, X], Y, 1, 5), starts_form([X, Y, X, Y], 1, 4),
       A._v68_ends_ok([X, X, None, X], X, 1, 4), A._v68_ends_ok([X, X], X, 1, 3), A._v68_ends_ok([X, X, X], None, 1, 3),
       A._v68_ends_ok([Z, Y, X, Y, X, Y], X, 2, 3), A._v68_ends_ok([Z, Y, X, Y, X, Y, X, Y], X, 2, 4)),
      (True, False, True, False, True, False, False, False, False, True, True))

# === 1. a true cycle escalates at 8 =========================================================================================
game, agent, log, s1, s2 = scenario('plain', lambda s, ext: (R_ if s.n % 2 == 0 else L_) + '  # c%d' % s.n, 30)
check('1 a true cycle (RIGHT / LEFT, codes varied): the hard rung 2 at the 8th report, no refusal, nothing blocked',
      (rung2(log), restarts(log), loopstats(agent), bool(s2.sent)), ([(2, 8, 'hard', None)], [('loop', 'hard')], NONE4, True))

# === 2. a period-2 alternation ==============================================================================================
game, agent, log, s1, s2 = scenario('toggle', lambda s, ext: R_ + '  # t%d' % s.n, 30)
check('2 a period-2 alternation (p 1, the interior toggles): the hard rung 2 at 8, nothing blocked',
      (rung2(log), restarts(log), loopstats(agent)), ([(2, 8, 'hard', None)], [('loop', 'hard')], NONE4))
check('2 at the trigger: p 1, k 10; the V68 starts form fails, the corrected form holds (period 2)', at_trigger(SNAP[-1]) if SNAP else None,
      (1, 10, False, True))

# === 3. an entry board that differs =========================================================================================
game, agent, log, s1, s2 = scenario('entry', lambda s, ext: R_ if s.n % 2 == 0 else L_, 30)
check('3 an entry board that differs, ends identical (p 2, identical codes): two refusals, the hard rung 2 at the 5th report',
      (rung2(log), restarts(log), loopstats(agent)), ([(2, 5, 'hard', None)], [('loop', 'hard')], dict(NONE4, loop_refused=2)))
check('3 at the trigger (the 2nd refusal): p 2, k 5, the window from the entry board; the starts form fails, the corrected holds',
      (at_trigger(SNAP[-1]), len(SNAP[-1]['starts']), SNAP[-1]['starts'][0] != SNAP[-1]['starts'][2]) if SNAP else None,
      ((2, 5, False, True), 10, True))

# === 4. a band-counter-only change ==========================================================================================
game, agent, log, s1, s2 = scenario('band', lambda s, ext: (R_ if s.n % 2 == 0 else L_) + '  # b%d' % s.n, 30)
whole = [json.dumps(e.frame.grid) for e in game.history_entries[2:12:2]]
inner = [json.dumps(A._v68_interior(e.frame)) for e in game.history_entries[2:12:2]]
check('4 a band-counter-only change: the hard rung 2 at 8; whole boards differ, interiors equal; nothing blocked',
      (rung2(log), restarts(log), len(set(whole)) > 1, len(set(inner)), loopstats(agent)),
      ([(2, 8, 'hard', None)], [('loop', 'hard')], True, 1, NONE4))

# === 5. a walk ==============================================================================================================
game, agent, log, s1, s2 = scenario('clock', lambda s, ext: R_, 17)
blocked = events(log, 'v68-loop-frame-blocked')
check('5 a walk (new interiors, identical reports and code): no refusal, no escalation through 14 reports; 17 RIGHTs executed',
      (agent._v67_loop['consecutive'], agent._v6_stats.get('loop_rung1'), agent._v6_stats.get('loop_refused'), rung2(log),
       restarts(log), [p['reason'] for p in events(log, 'v62-summary-turn')], game.engine.count('RIGHT'), len(game.engine)),
      (14, 1, None, [], [], [], 17, 17))
check('5 receipts: every identical call after the block executed (10); triggers blocked at 8..14 (7), with their events',
      (agent._v6_stats.get('loop_refusal_frame_blocked'), agent._v6_stats.get('loop_frame_blocked'),
       [(p['p'], p['consecutive'], p['refused']) for p in blocked], agent._v6_stats.get('loop_fallback')),
      (10, 7, [(1, c, 0) for c in range(8, 15)], None))

# === 6. a repeated-death walk: the fallback ================================================================================
game, agent, log, s1, s2 = scenario('clock', lambda s, ext: RD, 60)
fb = events(log, 'v68-loop-fallback')
check('6 a repeated-death walk: the fallback at the 16th report, a hard loop restart (RESET), no refusal',
      (rung2(log), restarts(log), 'RESET' in game.engine, agent._v6_stats.get('loop_refused'), agent._v6_stats.get('loop_fallback'),
       bool(s2.sent)), ([(2, 16, 'hard', True)], [('loop', 'hard')], True, None, 1, True))
check('6 the fallback event: p 1, consecutive 16, level 1, 16 GAME OVERs in the window; triggers blocked at 8..15',
      ([(p['p'], p['consecutive'], p['level'], p['gameovers'], sorted(p)) for p in fb], agent._v6_stats.get('loop_frame_blocked'),
       game.engine.count('DOWN') >= 16),
      ([(1, 16, 1, 16, ['attempt', 'consecutive', 'gameovers', 'k', 'level', 'p'])], 8, True))

# === 7. a refused, then frame-blocked episode keeps counting ================================================================
def stall_then_walk(s, ext):
    if 'This call is identical to call' in texts_of(ext, 'tool'):
        MODE[0] = 'clock'                               # after the one refusal the boards become new: a walk from here on
        return R_ + '  # varied once'
    return R_


game, agent, log, s1, s2 = scenario('toggle', stall_then_walk, 60)
fb = events(log, 'v68-loop-fallback')
check('7 one refusal while the condition held, then identical calls executed: the fallback at 16, a hard loop restart',
      (agent._v6_stats.get('loop_refused'), agent._v6_stats.get('loop_refusal_frame_blocked', 0) > 0, rung2(log), restarts(log),
       [(p['consecutive'], p['gameovers']) for p in fb], agent._v6_stats.get('loop_frame_blocked')),
      (1, True, [(2, 16, 'hard', True)], [('loop', 'hard')], [(16, 0)], 8))

# === 8. LOOP_FRAME off: identical to the tree after V68 ======================================================================
A.V6Agent._v67_loop_trigger = _trigger
V68_STAGE = {'v62/agent.py': '9e00aa5436b064b57fca550778a97f59', 'v62/variants.py': '2b7f567b81f3bfdf25f5d777be271506',   # + V68C
             'v62/run_v6.py': 'c3a66e582de5249609ab3330cc3075d0', 'v62/sandbox_ext.py': '9ed872bacacce2a9deda14582df66d4a',
             'v62/solver.py': '5494cedf3f4ee2d117130eb0e757ef10'}
prev = Path(tempfile.mkdtemp()) / 'v68_stage'
shutil.copytree(ROOT / 'v62', prev / 'v62', ignore=shutil.ignore_patterns('__pycache__', '*.pre-*', '*.tmp', '*.new'))
for f in V68_STAGE:
    shutil.copyfile(ROOT / (f + '.pre-v67x'), prev / f)
p = subprocess.run([sys.executable, '-B', str(ROOT.parent / 'apply_v67x.py'), str(prev), 'EXT200,SUMLOOKUP,GAMEBUDGET,LOOPHARD,V68,V68C'],
                   capture_output=True, text=True, timeout=600, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
md5 = {f: hashlib.md5(open(prev / f, 'rb').read()).hexdigest() for f in V68_STAGE}
check('8 the tree after V68 + V68C (the loop text), rebuilt by apply_v67x.py, has the recorded md5s', (p.returncode, md5 == V68_STAGE),
      (0, True))
env_clean = {k: v for k, v in os.environ.items() if not k.startswith(('ARC3_V67_', 'ARC3_V68_'))}
VARIANTS = {'all_off': {}, 'v68_but_frame': dict(PLANNED, ARC3_V62_ZERO_LEVEL_BAND='0', ARC3_V67_GAME_POOL='1000000',
                                                 ARC3_V61_RESTART_TOKENS='0', **{k: v for k, v in V68_ON.items()
                                                                                 if k != 'ARC3_V68_LOOP_FRAME'})}


def run_tree(job):
    tree, variant = job
    env = dict(env_clean, PYTHONPATH='%s:%s' % (ROOT / 'src', tree), PYTHONDONTWRITEBYTECODE='1')
    q = subprocess.run([sys.executable, '-B', '-c', IDENT, json.dumps(dict(ENV, **VARIANTS[variant])), FAKES], env=env,
                       cwd=str(tree), capture_output=True, text=True, timeout=1500)
    line = [x for x in q.stdout.splitlines() if x.startswith('IDENT-JSON ')]
    if q.returncode or not line:
        print('identity run failed in %s %s (rc %s):\n%s' % (tree, variant, q.returncode, q.stderr[-3000:]))
        return None
    return json.loads(line[-1][len('IDENT-JSON '):])


jobs = [(prev, v) for v in VARIANTS] + [(ROOT, v) for v in VARIANTS]
with ThreadPoolExecutor(len(jobs)) as pool:
    outs = dict(zip([(('prev' if t == prev else 'new'), v) for t, v in jobs], pool.map(run_tree, jobs)))
check('8 the four identity runs completed', [k for k, v in outs.items() if v is None], [])
CMP = ('sent', 'engine', 'events', 'stats', 'records', 'library')
if all(v is not None for v in outs.values()):
    for variant in VARIANTS:
        a, b = outs[('prev', variant)], outs[('new', variant)]
        check('8 %s: the texts and the agent\'s TEXTS identical' % variant, (a['texts'] == b['texts'],
                                                                           a['agent_texts'] == b['agent_texts']), (True, True))
        for name in a['runs']:
            check('8 %s %s: traffic, actions, events, stats, records, library identical' % (variant, name),
                  [k for k in CMP if a['runs'][name][k] != b['runs'][name][k]], [])
    rn = outs[('new', 'v68_but_frame')]['runs']
    check('8 the V68-but-frame runs took their paths (a refresh, the loop restarts of the cycle and of the walk)',
          (rn['refresh']['stats'].get('refreshes'), [e[1]['reason'] for e in rn['rung2']['events'] if e[0] == 'v61-restart'],
           [e[1]['reason'] for e in rn['walk']['events'] if e[0] == 'v61-restart']), (1, ['loop'], ['loop']))

# === 9. REFRESH_PROGRAMS (item V68P; the user's selection at 18:02 UTC, the final rule from the refresh reviewer's data) =====
check('9 off by default; refuses without the refresh and below 0; defaults FREQ 0, RECENT 8',
      (A.REFRESH_PROGRAMS68, probe(ARC3_V68_REFRESH_PROGRAMS='1', ARC3_V67_REFRESH_LOGICAL='0').startswith(
          'ERROR RuntimeError: ARC3_V68_REFRESH_PROGRAMS=1 needs the refresh'),
       probe(ARC3_V68_REFRESH_PROG_FREQ='-1').startswith('ERROR RuntimeError: ARC3_V68_REFRESH_PROG_FREQ and _RECENT'),
       probe(ARC3_V68_REFRESH_PROGRAMS='1'), (A.PROG_FREQ68, A.PROG_RECENT68)),
      (False, True, True, 'FALLBACK 16', (0, 8)))
SEL = A._v68_select_programs
NAMES = list('abcdefg')
COUNTS = dict(a=4, b=4, d=1, g=7)
CALL = dict(a=(5, 10), b=(5, 12), d=(9, 30), g=(2, 3))
DEF = dict(a=(11, 40), b=(1, 2), c=(8, 25), d=(3, 5), e=(7, 20), f=(0, 0), g=(1, 2))
check('9 recency (FREQ 0): the last call; never called, the last writing; a rewrite after a call does not count; same turn by '
      'the call order; nothing left out', (SEL(NAMES, COUNTS, CALL, DEF, 0, 8), SEL(NAMES, COUNTS, CALL, DEF, 0, 3)),
      (([], ['d', 'c', 'e', 'b', 'a', 'g', 'f']), ([], ['d', 'c', 'e'])))
check('9 FREQ 2 / RECENT 2: by count, the tie a / b to the more recent b, then the most recent of the rest; a full tie to the '
      'later program; empty', (SEL(NAMES, COUNTS, CALL, DEF, 2, 2), SEL(['x', 'y'], {}, dict(x=(3, 5), y=(3, 5)), {}, 0, 8),
                              SEL([], {}, {}, {}, 0, 8)),
      ((['g', 'b'], ['d', 'c']), ([], ['y', 'x']), ([], [])))
game, (sess, sess2), agent, log, go = harness([[(R_, 30)], [(R_, 30)]], budget=200000)
agent._kept_functions = {'foo': 'def foo(x):\n    """f"""\n    return x', 'foo_bar': 'def foo_bar():\n    return 1'}
for turn, code in ((3, 'def foo(x):\n    return x'), (4, 'print(foo(1))'), (4, 'foo_bar()'), (6, 'def baz():\n    return foo(2)\nbaz()')):
    agent._v6_stats['turns'] = turn
    agent._v68_count_program_refs(code)
st = agent._v68_prog
check('9 counting: a call names a program by word boundary beyond its def lines; (turn, sequence) of the last call / writing',
      (st['counts'], st['last_call'], st['last_def']),
      ({'foo': 2, 'foo_bar': 1, 'baz': 1}, {'foo': (6, 4), 'foo_bar': (4, 3), 'baz': (6, 4)}, {'foo': (3, 1), 'baz': (6, 4)}))
U_ = 'action(["UP"])'
SUM1 = 'save_summary("Rule L1:\\n- UP completes the level\\n- RIGHT moves the marker one cell")'
MODE[0] = 'plain'
A.TEXTS.update(V._v68_refresh_programs_texts({}))      # its texts, as an import with the switch on renders them


def refresh_run(on, used, freq=0):
    A.REFRESH_PROGRAMS68, A.PROG_FREQ68 = on, freq
    first = [(PROG + R_, 40)] + ([('print(double(2))\n' + R_, 40)] if used else []) + [(U_, 40), (SUM1, 60)]
    game, (sess, sess2), agent, log, go = harness([first, [(R_, 30), (R_, 30)]], budget=200000, REFRESH=True)
    for i in range(len(first) + 1):
        if i == len(first) - 1:
            sess.last_receipt = {'logical_tokens': 600000}
        go()
    A.REFRESH_PROGRAMS68, A.PROG_FREQ68 = False, 0
    return agent, texts_of(sess2.sent[0], 'user') if sess2.sent else '', log


HDR = 'Your most recent programs, by your last call to each (or its writing, if not yet called) (1 of 1):'
HDR_MOST = ('Your most called programs, then your most recent ones by your last call to each (or its writing, if not yet called) '
            '(1 of 1):')
PTR = '`programs()` lists all 1, and `read_code(name)` shows any source.'
agent, opener_new, log = refresh_run(True, False)
count = agent._v61_library_count_line()
check('9 on, double written and never called: listed right after the count line (header, program_list\'s line, pointer)',
      (len(events(log, 'v67-refresh')), (count + '\n' + HDR + '\n- double(x): twice x\n' + PTR) in opener_new,
       [(p['library'], p['most'], p['recent'], p['counts']) for p in events(log, 'v68-refresh-programs')],
       agent._v6_stats.get('refresh_program_lists')), (1, True, [(1, 0, 1, [0])], 1))
agent, opener_used, log = refresh_run(True, True)
agent2, opener_most, log2 = refresh_run(True, True, freq=1)
check('9 on, double called once: recent only at FREQ 0; with FREQ 1 the most-called header (count 1)',
      ([(p['most'], p['recent'], p['counts']) for p in events(log, 'v68-refresh-programs')], HDR in opener_used,
       [(p['most'], p['recent'], p['counts']) for p in events(log2, 'v68-refresh-programs')], HDR_MOST in opener_most),
      ([(0, 1, [1])], True, [(1, 0, [1])], True))
agent, opener_off, log = refresh_run(False, False)
check('9 off: the count line, no selection; the opener otherwise identical',
      (len(events(log, 'v67-refresh')), count in opener_off, HDR in opener_off, events(log, 'v68-refresh-programs'),
       opener_new.replace('\n' + HDR + '\n- double(x): twice x\n' + PTR, '') == opener_off), (1, True, False, [], True))
print('FAILS:', fails)
