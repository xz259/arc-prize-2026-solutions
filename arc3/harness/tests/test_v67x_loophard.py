"""test_v67x_loophard.py (2026-10-18, Claude 099768dc, implementation subagent): apply_v67x.py's item LOOPHARD
(ARC3_V67_LOOP_RUNG2_KIND, user decision 50: no soft restarts), through agent.analyze with the real sandbox on test_v67_soft.py's
fakes (ARC3_V67_LOOP=1):
  0 the switch: default soft; case-insensitive; an invalid value refuses; run_v6's explicit default and record;
  1 rung 2 under hard: the loop-rule block states the hard restart; two refused repeats; the hard loop request mid-turn; the
    restart is hard: a RESET to the level's first board, the hard takeover with the notes, no soft line; receipts, the record;
  2 rung 3 under hard: the loop again within 200K tokens of the rung-2 restart gives the 'again' request (hard), and the 2nd
    loop hard restart on the level parks the game (parking as built);
  3 the default (soft): the same script gives today's soft rung 2 (no RESET, the loop opener line, the soft block);
  4 switch-off identity with the deployed v6.7 base (v62/*.pre-v67x, a subprocess on each tree), with LOOP off and on.
usage (dev copy of harness_v6): PYTHONPATH=src:. python3 tests/test_v67x_loophard.py"""
import copy
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
for _k in [k for k in os.environ if k.startswith('ARC3_V67_')]:
    os.environ.pop(_k)
os.environ['ARC3_V67_LOOP'] = '1'
os.environ['ARC3_V67_LOOP_RUNG2_KIND'] = 'hard'
import v62.patches  # noqa: E402,F401
import v62.retention as r  # noqa: E402
r.install()
import v62.variants as V  # noqa: E402
import v62.agent as A  # noqa: E402
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


def tool_errors(batch):
    out = []
    for m in batch:
        if m.get('role') == 'tool' and isinstance(m.get('content'), str):
            try:
                out.append(json.loads(m['content']).get('error') or '')
            except (TypeError, ValueError, AttributeError):
                pass
    return '\n'.join(out)


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


class Gate:
    _waiting, _clock_deadline = [], None

    def __init__(self):
        self.calls = []

    def handover(self, priority, snapshot):
        self.calls.append((priority, snapshot))


RAW = json.load(open(Path(A.__file__).with_name('texts.json')))['texts']
R_, L_ = 'action(["RIGHT"])', 'action(["LEFT"])'
NOTES_L = 'save_summary("Notes L1:\\n- RIGHT and LEFT move the marker one cell\\n- the calls repeated a cycle")'
CONV2 = [(R_, 50), (L_, 50)] * 5 + [(R_, 40), (L_ + '   \n', 40), (NOTES_L, 60)]
HARD_SENTENCE = "the harness restarts this level: a RESET, then a new conversation from the level's first board with your notes."
SOFT_SENTENCE = "the harness ends this conversation and starts a new one from the current board with your notes, without a RESET."

# === 0. the switch ======================================================================================================
kinds = []
for value in (None, 'hard', 'HARD ', 'soft', ''):
    keep = os.environ.get('ARC3_V67_LOOP_RUNG2_KIND')
    if value is None:
        os.environ.pop('ARC3_V67_LOOP_RUNG2_KIND', None)
    else:
        os.environ['ARC3_V67_LOOP_RUNG2_KIND'] = value
    kinds.append(V.loop_rung2_kind())
    os.environ['ARC3_V67_LOOP_RUNG2_KIND'] = keep
os.environ['ARC3_V67_LOOP_RUNG2_KIND'] = 'medium'
try:
    V.loop_rung2_kind()
    refused = False
except RuntimeError:
    refused = True
os.environ['ARC3_V67_LOOP_RUNG2_KIND'] = 'hard'
check('0 the switch: default soft, hard (any case), an empty value soft; an invalid value refuses',
      (kinds, refused, A.LOOP_RUNG2_KIND), (['soft', 'hard', 'hard', 'soft', 'soft'], True, 'hard'))
RUN = open(Path(A.__file__).with_name('run_v6.py')).read()
check('0 run_v6: the explicit default soft and the recorded switch',
      ("os.environ.setdefault('ARC3_V67_LOOP_RUNG2_KIND', 'soft')" in RUN,
       "'ARC3_V67_LOOP_RUNG2_KIND'," in RUN.split('variants=')[1].split('history_eviction=')[0]), (True, True))
soft_texts = build(ARC3_V67_LOOP_RUNG2_KIND='soft')
check('0 the texts: under hard only loop_block changes, and only its last sentence',
      (sorted(k for k in set(A.TEXTS) | set(soft_texts) if A.TEXTS.get(k) != soft_texts.get(k)),
       A.TEXTS['loop_block'] == soft_texts['loop_block'].replace(SOFT_SENTENCE, HARD_SENTENCE),
       A.TEXTS['loop_block'].endswith(HARD_SENTENCE)), (['loop_block'], True, True))

# === 1. rung 2 under hard ================================================================================================
game, (sess, sess2), agent, log, go = harness([copy.deepcopy(CONV2), [(R_, 50)]], budget=200000)
for _ in range(12):
    go()
block = A._fill(A.TEXTS['loop_block'], n=10, p=2, k=5, a=2, A=10, t='500', T='200,000')
check('1 the block at the 5th cycle report states the hard restart', (user_text(sess.sent[10]).startswith(block + '\n'),
      HARD_SENTENCE in block, 'without a RESET' in block), (True, True, False))
check('1 the two repeats refused with the (hard) block',
      ('This call is identical to call 1 of the detected cycle and was not executed.' in tool_errors(sess.sent[11]),
       'This call is identical to call 2 of the detected cycle and was not executed.' in tool_errors(sess.sent[12]),
       HARD_SENTENCE in tool_errors(sess.sent[11])), (True, True, True))
check('1 the 2nd refused repeat: the hard loop request in the same turn',
      user_text(sess.sent[12]), A._fill(A.TEXTS['attempt_summary_request_loop_hard'], n=10, p=2, level=1))
first = user_text(sess2.sent[0])
check('1 the restart is hard: one RESET, to the level\'s first board, the hard takeover with the notes, no soft line',
      ([(p['reason'], p['kind']) for p in events(log, 'v61-restart')], game.engine.count('RESET'), len(game.engine),
       game.history_entries[-2].action, game.history_entries[-2].frame.grid == board(0),
       'This conversation begins after a RESET of level 1' in first, '- RIGHT and LEFT move the marker one cell' in first,
       'there was no RESET' in first, 'continues level 1' in first),
      ([('loop', 'hard')], 1, 12, 'RESET', True, True, True, False, False))
check('1 receipts: a loop hard restart (no soft), the record carries rung 2, no soft-restart event',
      (agent._v6_stats.get('loop_hard_restarts'), agent._v6_stats.get('loop_soft_restarts'), agent._v6_stats.get('hard_restarts'),
       [(x['kind'], x['reason'], x['loop']['rung']) for x in game.v67_level_restarts[1]], events(log, 'v67-soft-restart')),
      (1, None, 1, [('hard', 'loop', 2)], []))
check('1 the v67-loop trigger event: rung 2, kind hard', [(p['rung'], p.get('kind')) for p in events(log, 'v67-loop')
                                                          if p['rung'] >= 2], [(2, 'hard')])

# === 2. rung 3 under hard, parking at the 2nd loop hard restart =========================================================
cyc = lambda tag: [('%s  # %s%d' % (R_ if i % 2 == 0 else L_, tag, i), 50) for i in range(13)]   # noqa: E731
gate = Gate()
game, (sess, sess2, sess3), agent, log, go = harness([cyc('a') + [(NOTES_L, 40)], cyc('b') + [(NOTES_L, 40)], [(R_, 50)]],
                                                     budget=200000)
A.ta._priority_gate = lambda: gate
for _ in range(29):
    go()
A.ta._priority_gate = lambda: None
check('2 rung 2 at the 8th consecutive report: the hard request at the opener',
      A._fill(A.TEXTS['attempt_summary_request_loop_hard'], n=12, p=2, level=1) in user_text(sess.sent[13]), True)
check('2 the loop again within 200K tokens: rung 3 (the \'again\' request), hard',
      (A._fill(A.TEXTS['attempt_summary_request_loop_again'], n=12, p=2, level=1) in user_text(sess2.sent[13]),
       [(p['rung'], p.get('since'), p.get('kind')) for p in events(log, 'v67-loop') if p['rung'] >= 2]),
      (True, [(2, None, 'hard'), (3, 650, 'hard')]))
check('2 two hard restarts (two RESETs); the 2nd loop hard restart parks the game (lowest priority), as built',
      ([(p['reason'], p['kind']) for p in events(log, 'v61-restart')], game.engine.count('RESET'),
       [snap.parked for _, snap in gate.calls], gate.calls[-1][0], agent._v6_stats.get('loop_parks'),
       [p['loop_hard_restarts'] for p in events(log, 'v67-loop-park')],
       'This conversation begins after a RESET of level 1' in user_text(sess3.sent[0])),
      ([('loop', 'hard'), ('loop', 'hard')], 2, [False, True], -1_000_000_000, 1, [2], True))

# === 3. the default (soft) is unchanged ==================================================================================
hard_texts, A.TEXTS, A.LOOP_RUNG2_KIND = A.TEXTS, soft_texts, 'soft'
game, (sess, sess2), agent, log, go = harness([copy.deepcopy(CONV2), [(R_, 50)]], budget=200000)
for _ in range(12):
    go()
first = user_text(sess2.sent[0])
check('3 soft (default): the soft block, the soft loop request, a soft restart (no RESET), the loop opener line',
      (user_text(sess.sent[10]).startswith(A._fill(soft_texts['loop_block'], n=10, p=2, k=5, a=2, A=10, t='500', T='200,000')),
       user_text(sess.sent[12]) == A._fill(soft_texts['attempt_summary_request_loop'], n=10, p=2, level=1),
       [(p['reason'], p['kind']) for p in events(log, 'v61-restart')], game.engine.count('RESET'),
       A._fill(soft_texts['takeover_loop'], level=1, step=10, n=10, p=2, count=10) in first),
      (True, True, [('loop', 'soft')], 0, True))
A.TEXTS, A.LOOP_RUNG2_KIND = hard_texts, 'hard'

# === 4. switch-off identity with the deployed v6.7 base ===================================================================
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


class Gate:
    _waiting, _clock_deadline = [], None

    def __init__(self):
        self.calls = []

    def handover(self, priority, snapshot):
        self.calls.append([priority, bool(getattr(snapshot, 'parked', False))])


def run(scripts, turns, **kw):
    game, sessions, agent, log, go = harness(scripts, **kw)
    gate = Gate()
    A.ta._priority_gate = lambda: gate
    for _ in range(turns):
        go()
    A.ta._priority_gate = lambda: None
    events = []
    for k, p in log:
        p = json.loads(json.dumps(p, default=str))
        if k == 'tool-output':
            p.pop('seconds', None)
        if k == 'v62-swap':
            p.pop('waited_s', None)
        events.append([k, norm(p)])
    stats = json.loads(json.dumps(agent._v6_stats, default=str))
    stats.pop('swap_wait_s', None)
    keys = ('v6_level_attempts', 'v62_level_attempt_ends', 'v6_attempt_actions', 'v67_level_restarts')
    return dict(sent=[norm(s.sent) for s in sessions], engine=game.engine, events=events, stats=stats,
                records=json.loads(json.dumps({k: game.__dict__.get(k) for k in keys})), gate=gate.calls,
                library=sorted(agent._kept_functions), archive=sorted(agent._v61_archive))


R_, L_ = 'action(["RIGHT"])', 'action(["LEFT"])'
NOTES_L = 'save_summary("Notes L1:\\n- RIGHT and LEFT move the marker one cell\\n- the calls repeated a cycle")'
cyc = lambda tag: [('%s  # %s%d' % (R_ if i % 2 == 0 else L_, tag, i), 50) for i in range(13)]   # noqa: E731
runs = {}
runs['rung2'] = run([[(R_, 50), (L_, 50)] * 5 + [(R_, 40), (L_ + '   \n', 40), (NOTES_L, 60)], [(R_, 50)]], 12, budget=200000)
if os.environ.get('ARC3_V67_LOOP') == '1':          # with LOOP off the script would run out (no loop restart)
    runs['rung3'] = run([cyc('a') + [(NOTES_L, 40)], cyc('b') + [(NOTES_L, 40)], [(R_, 50)]], 29, budget=200000)
runs['budget'] = run([[(PROG + R_, 100), (L_, 450), (L_, 50), (NOTES, 80)], [(R_, 50), (L_, 50)]], 5, budget=500)
print('IDENT-JSON ' + json.dumps(dict(runs=runs, agent_texts=A.TEXTS), sort_keys=True, default=str))
'''
CHANGED = ('v62/agent.py', 'v62/variants.py', 'v62/run_v6.py', 'v62/sandbox_ext.py', 'v62/solver.py')
missing = [f for f in CHANGED if not (ROOT / (f + '.pre-v67x')).exists()]
check('4 the deployed v6.7 base copies (v62/*.pre-v67x, written by apply_v67x.py) are present', missing, [])
if not missing:
    base = Path(tempfile.mkdtemp()) / 'v67_base'
    shutil.copytree(ROOT / 'v62', base / 'v62', ignore=shutil.ignore_patterns('__pycache__', '*.pre-*', '*.tmp'))
    for f in CHANGED:
        shutil.copyfile(ROOT / (f + '.pre-v67x'), base / f)
    env_clean = {k: v for k, v in os.environ.items() if not k.startswith('ARC3_V67_')}
    VARIANTS = {'loop_off': {}, 'loop_on': dict(ARC3_V67_LOOP='1')}

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
    jobs = [(base, 'loop_off'), (ROOT, 'loop_off'), (base, 'loop_on'), (ROOT, 'loop_on')]
    with ThreadPoolExecutor(len(jobs)) as pool:
        outs = dict(zip([(('base' if t == base else 'new'), v) for t, v in jobs], pool.map(run_tree, jobs)))
    check('4 the four identity runs completed', [k for k, v in outs.items() if v is None], [])
    CMP = ('sent', 'engine', 'events', 'stats', 'records', 'gate', 'library', 'archive')
    if all(v is not None for v in outs.values()):
        for variant in VARIANTS:
            a, b = outs[('base', variant)], outs[('new', variant)]
            check('4 %s: the agent\'s TEXTS identical' % variant, a['agent_texts'] == b['agent_texts'], True)
            for name in a['runs']:
                check('4 %s %s: traffic, actions, events, stats, records, swaps, library identical' % (variant, name),
                      [k for k in CMP if a['runs'][name][k] != b['runs'][name][k]], [])
        rn = outs[('new', 'loop_on')]['runs']
        check('4 the LOOP-on runs took the intended paths (a soft rung 2; rung 3 hard; a budget restart)',
              ([(e[1]['reason'], e[1]['kind']) for e in rn['rung2']['events'] if e[0] == 'v61-restart'],
               [(e[1]['reason'], e[1]['kind']) for e in rn['rung3']['events'] if e[0] == 'v61-restart'],
               [(e[1]['reason'], e[1]['kind']) for e in rn['budget']['events'] if e[0] == 'v61-restart']),
              ([('loop', 'soft')], [('loop', 'soft'), ('loop', 'hard')], [('tokens', 'hard')]))
print('FAILS:', fails)
