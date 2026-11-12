"""test_v67x_v68.py (2026-10-18, Claude 099768dc, implementation subagent): apply_v67x.py's item V68 (the v6.8 texts, IE agreement
2026-10-18), through agent.analyze with the real sandbox on test_v67_soft.py's fakes, in the planned run's shape (LOOP with a hard
rung 2, REFRESH, every V68 switch but NOTES_NEG on):
  0 the switches: defaults off; run_v6's explicit defaults and record; the refusals (NO_SCORE without SCO or with SCORE_TEXT,
    LOOP_TEXTS without LOOP or a hard rung 2, LOOP_FRAME without LOOP, NOTES_NEG with NOTES);
  1 NO_SCORE: exactly the v6.6 scoring bullet removed; the score mentions left are the candidate-scoring ones;
  2 LEDGER: v6.7's ledger (its texts) in the level-up, restart and refresh openers, none of the other NOTES parts;
  3 NEVER_TYPES: the never-executed line in restart openers only (not level-up or refresh), no distinct-boards line;
  4 NOTES_NEG (built; the user skipped it at 16:40 UTC, it stays off): its clause, derived by the loop requests;
  5 LOOP_TEXTS: the block (exact, {level} filled), the rung-2 hard request and loop_again (exact), the takeover's loop line
    after a loop restart only, loop_refused unchanged;
  6 LOOP_FRAME (as corrected by V68B, 17:4x UTC: the interiors at the end of each repetition; refusals only while the
    condition holds): a walk (identical normalised reports and code, frames differing) is neither refused nor escalated
    (receipts loop_refusal_frame_blocked and loop_frame_blocked, the event); the same walk escalates without LOOP_FRAME; a
    cycle whose frames differ only in the outer band escalates (test_v67x_v68b.py has the V68B cases);
  7 REFRESH_TEXTS: the summary-turn request and the opener line (exact), the pinned summary and the ledger follow, the refusal
    unchanged;
  8 all V68 switches off: identical to the tree before V68 (built from the base by apply_v67x.py's earlier items, md5-checked),
    a subprocess on each tree, with every switch off and in the planned run's base config.
usage (dev copy of harness_v6): PYTHONPATH=src:. python3 -B tests/test_v67x_v68.py"""
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace  # noqa: F401  (the fakes' harness)

_src = open(Path(__file__).with_name('test_v67_soft.py')).read()
ENV = eval(_src.split('ENV = ', 1)[1].split('\n\n', 1)[0])
FAKES = _src.split("FAKES = r'''", 1)[1].split("'''", 1)[0]
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
import v62.variants as V  # noqa: E402
import v62.agent as A  # noqa: E402
exec(compile(FAKES, 'test_v67_soft.py (FAKES)', 'exec'), globals())
fails = []
ROOT = Path(A.__file__).resolve().parents[1]
RAW = json.load(open(Path(A.__file__).with_name('texts.json')))['texts']


def check(name, got, want):
    ok = got == want
    print('%-84s -> %-40r %s' % (name, got if not isinstance(got, str) else got[:40], 'OK' if ok else 'WRONG (want %r)' % (want,)))
    if not ok:
        fails.append(name)


def user_text(batch):
    m = [x for x in batch if x.get('role') == 'user'][-1]
    c = m['content']
    return c if isinstance(c, str) else '\n'.join(p.get('text', '') for p in c if p.get('type') == 'text')


def all_text(batch):
    out = []
    for m in batch:
        c = m.get('content')
        out.append(c if isinstance(c, str) else '\n'.join(p.get('text', '') for p in (c or []) if p.get('type') == 'text'))
    return '\n'.join(out)


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
    """V.apply with these environment changes (a value None removes the variable)"""
    keep = {k: os.environ.get(k) for k in env}
    for k, v in env.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
    try:
        return V.apply(json.loads(json.dumps(RAW)))
    except RuntimeError as exc:
        return 'ERROR: %s' % exc
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


OFF = {k: None for k in V68_ON}
R_, L_, U_ = 'action(["RIGHT"])', 'action(["LEFT"])', 'action(["UP"])'
NOTES_L = 'save_summary("Notes L1:\\n- RIGHT and LEFT move the marker one cell\\n- the calls repeated a cycle")'
NOTES2 = 'save_summary("Notes L2:\\n- RIGHT moves the marker on level 2\\n- UP clears a level")'
SUM1 = 'save_summary("Rule L1:\\n- UP completes the level\\n- RIGHT moves the marker one cell")'

# === 0. the switches ======================================================================================================
check('0 this process: the V68 agent switches on', (A.LEDGER68, A.NEVER68, A.LOOP_TEXTS68, A.LOOP_FRAME68), (True,) * 4)
RUN = open(Path(A.__file__).with_name('run_v6.py')).read()
names = ('NO_SCORE', 'LEDGER', 'NEVER_TYPES', 'NOTES_NEG', 'LOOP_TEXTS', 'LOOP_FRAME', 'REFRESH_TEXTS')
check('0 run_v6: an explicit default 0 and the record for each of the seven switches',
      ([n for n in names if "os.environ.setdefault('ARC3_V68_%s', '0')" % n not in RUN],
       [n for n in names if "'ARC3_V68_%s'," % n not in RUN.split('variants=')[1].split('history_eviction=')[0]]), ([], []))
REFUSALS = ((dict(ARC3_V66_SCO='0'), 'ERROR: ARC3_V68_NO_SCORE=1 needs ARC3_V66_SCO=1'),
            (dict(ARC3_V67_SCORE_TEXT='1'), 'ERROR: ARC3_V68_NO_SCORE=1 needs ARC3_V66_SCO=1'),
            (dict(ARC3_V67_LOOP=None), 'ERROR: ARC3_V68_LOOP_TEXTS=1 needs ARC3_V67_LOOP=1'),
            (dict(ARC3_V67_LOOP_RUNG2_KIND='soft'), 'ERROR: ARC3_V68_LOOP_TEXTS=1 needs ARC3_V67_LOOP=1'),
            (dict(ARC3_V68_NOTES_NEG='1', ARC3_V67_NOTES='1'), 'ERROR: ARC3_V68_NOTES_NEG=1 edits the v6.6 notes sentence'))
check('0 the refusals: NO_SCORE without SCO / with SCORE_TEXT; LOOP_TEXTS without LOOP / soft; NOTES_NEG with NOTES',
      [str(build(**e)).startswith(prefix) for e, prefix in REFUSALS], [True] * 5)
probe = subprocess.run([sys.executable, '-B', '-c', 'import v62.patches, v62.retention as r; r.install(); import v62.agent'],
                       env=dict(os.environ, ARC3_V67_LOOP='0', ARC3_V68_LOOP_TEXTS='0', ARC3_V68_LOOP_FRAME='1',
                                PYTHONDONTWRITEBYTECODE='1', PYTHONPATH='%s:%s' % (ROOT / 'src', ROOT)),
                       cwd=str(ROOT), capture_output=True, text=True, timeout=600)
check('0 LOOP_FRAME without LOOP refuses to start', (probe.returncode != 0, 'ARC3_V68_LOOP_FRAME=1 needs the loop ladder' in probe.stderr),
      (True, True))

# === 1. NO_SCORE ===========================================================================================================
BULLET = ("- A cleared level scores in full when it takes no more actions than a hidden per-level baseline, and less the further "
          "it goes over it; later levels count more; an uncleared level scores 0.\n")
with_score = build(ARC3_V68_NO_SCORE=None)
check('1 NO_SCORE: exactly the v6.6 scoring bullet removed, nothing else changes',
      (BULLET in with_score['system'], A.TEXTS['system'] == with_score['system'].replace(BULLET, ''),
       [k for k in set(A.TEXTS) | set(with_score) if k != 'system' and A.TEXTS.get(k) != with_score.get(k)]), (True, True, []))
import re  # noqa: E402
mentions = {k: [m.group(0) for m in re.finditer(r'(?i)[^.\n]*\b(?:scor\w*|baseline\w*)[^.\n]*', str(v))] for k, v in A.TEXTS.items()}
mentions = {k: v for k, v in mentions.items() if v}
check('1 the score mentions left (planned config, NO_SCORE on): candidate scoring only, in the system prompt and the reminder',
      sorted(mentions), ['reminder', 'system'])
check('1 ... the system line is "score or search candidate actions"; the reminder\'s "Score or search candidate actions"',
      (['score or search candidate actions' in x for x in mentions['system']], ['Score or search candidate actions' in x
                                                                                for x in mentions['reminder']]), ([True], [True]))

# === 2-3. LEDGER and NEVER_TYPES ==========================================================================================
notes_texts = build(ARC3_V67_NOTES='1', **OFF)
check('2 the ledger and never texts are NOTES\'s, and none of NOTES\'s other texts exist',
      ([A.TEXTS[k] == notes_texts[k] for k in ('notes_ledger', 'notes_ledger_first', 'notes_never')],
       [k for k in ('notes_action_count', 'notes_coverage') if k in A.TEXTS],
       'Confirmed effects:' in A.TEXTS['attempt_summary_request']), ([True] * 3, [], False))
conv1 = [(R_, 40), (U_, 40), ('save_summary("Rule L1:\\n- UP completes the level\\n- RIGHT moves the marker")\n' + R_, 40),
         (R_, 600), (NOTES2, 40)]
game, (sess, sess2), agent, log, go = harness([conv1, [(R_, 30), (R_, 30)]], budget=500, REFRESH=False)
for _ in range(6):
    go()
LEDGER1 = A._fill(A.TEXTS['notes_ledger'], k=1, action='UP', actions='RIGHT')
levelup, take = user_text(sess.sent[2]), user_text(sess2.sent[0])
check('2 the level-up opener: the ledger line; no never line', (LEDGER1 in levelup, 'never executed on this level' in levelup),
      (True, False))
check('2-3 the restart opener: the ledger line and the never line (LEFT, UP, DOWN never executed on level 2)',
      (LEDGER1 in take, A._fill(A.TEXTS['notes_never'], names='LEFT, UP, DOWN') in take), (True, True))
check('2-3 none of the other NOTES parts: no action count, no distinct boards, the notes request without headers',
      ('Actions so far on this level' in take, 'Distinct boards' in take, 'Confirmed effects' in user_text(sess.sent[4])),
      (False, False, False))
check('2-3 receipts: ledger_lines (level-up and restart), never_lines (restart)', (agent._v6_stats.get('ledger_lines'),
      agent._v6_stats.get('never_lines'), agent._v6_stats.get('coverage_lines')), (2, 1, None))
from inference.agent.runtime_state import load_runtime_state  # noqa: E402
cf, he = load_runtime_state(game.state_path)
agent._step_env_callback = game.step_env
check('3 no never line when every valid type was executed on the level (valid RIGHT only: level 2 executed RIGHT)',
      (agent._v67_coverage_lines(2, 'RIGHT', he, never_only=True), agent._v68_restart_lines(2, 'RIGHT', he)), ([], [LEDGER1]))
agent._step_env_callback = None

# === 4. NOTES_NEG (built, off) =============================================================================================
neg = build(ARC3_V68_NOTES_NEG='1')
CLAUSE = 'and the hypotheses still open, noting, for anything you concluded does not work, the step where it was tested.'
check('4 NOTES_NEG (off by default; the user kept v6.6 verbatim): off here; on, its clause in the three requests and the loop '
      'requests', ('hypotheses still open, noting' in A.TEXTS['attempt_summary_request'],
                   [CLAUSE in neg[k] for k in ('attempt_summary_request', 'attempt_summary_request_noact',
                                               'attempt_summary_request_position', 'attempt_summary_request_loop_hard',
                                               'attempt_summary_request_loop_again')]), (False, [True] * 5))

# === 5. LOOP_TEXTS ==========================================================================================================
BLOCK = ("Loop rule (harness): your last {n} Python calls repeated one cycle of {p} call(s) {k} times: each cycle executed the "
         "same actions, and the reports were identical apart from numbers and the listed continuations that ended in GAME OVER. "
         "The game changes only when an action executes; it does not need actions to keep running. If the cycle continues, "
         "level {level} restarts from its first board (a RESET) in a new conversation with your notes.")   # V68C (18:3x UTC)
LEAD2 = ("The last {n} Python calls repeated one cycle of {p} call(s), so level {level} restarts after this turn (loop rule): a "
         "RESET, then a new conversation from the level's first board.")
LEAD3 = ("The last {n} Python calls repeated one cycle again after a loop-rule restart, so level {level} restarts after this "
         "turn: a RESET, then a new conversation from the level's first board.")
REST = A.TEXTS['attempt_summary_request'][A.TEXTS['attempt_summary_request'].index(' Actions are not executed in this turn.'):]
before = build(ARC3_V68_LOOP_TEXTS=None)
check('5 the texts: the block exact; the rung-2 and rung-3 requests = the leads + the v6.6 hard request\'s remainder; '
      'loop_refused unchanged; the takeover line',
      (A.TEXTS['loop_block'] == BLOCK, A.TEXTS['attempt_summary_request_loop_hard'] == LEAD2 + REST,
       A.TEXTS['attempt_summary_request_loop_again'] == LEAD3 + REST, A.TEXTS['loop_refused'] == before['loop_refused'],
       A.TEXTS['takeover_loop_hard'], 'score' in A.TEXTS['loop_block'] or 'tokens' in A.TEXTS['loop_block']),
      (True, True, True, True, 'The previous conversation on this level ended under the loop rule.', False))
CONV2 = [(R_, 50), (L_, 50)] * 5 + [(R_, 40), (L_ + '   \n', 40), (NOTES_L, 60)]
game, (sess, sess2), agent, log, go = harness([copy.deepcopy(CONV2), [(R_, 50)]], budget=200000, REFRESH=False)
for _ in range(12):
    go()
block = A._fill(BLOCK, n=10, p=2, k=5, level=1)
first = user_text(sess2.sent[0])
check('5 the block at the 5th cycle report ({level} filled) and in the refusals; the hard request mid-turn (exact)',
      (user_text(sess.sent[10]).startswith(block + '\n'), block in tool_errors(sess.sent[11]),
       user_text(sess.sent[12]) == A._fill(LEAD2 + REST, n=10, p=2, level=1)), (True, True, True))
check('5-6 a true cycle (identical interior frames) escalates: one RESET, the hard takeover with the loop line',
      ([(p['reason'], p['kind']) for p in events(log, 'v61-restart')], game.engine.count('RESET'),
       'This conversation begins after a RESET of level 1' in first, A.TEXTS['takeover_loop_hard'] in first,
       agent._v6_stats.get('loop_frame_blocked')), ([('loop', 'hard')], 1, True, True, None))
check('3 the loop restart\'s opener (level 1) has the never line (UP, DOWN)',
      A._fill(A.TEXTS['notes_never'], names='UP, DOWN') in first, True)
game, (sess, sess2), agent, log, go = harness([[(R_, 600), (NOTES_L, 40)], [(R_, 30)]], budget=500, REFRESH=False)
for _ in range(3):
    go()
check('5 a budget restart\'s takeover has no loop line', ([(p['reason']) for p in events(log, 'v61-restart')],
      A.TEXTS['takeover_loop_hard'] in user_text(sess2.sent[0])), (['tokens'], False))

# === 6. LOOP_FRAME ==========================================================================================================
walk = [(R_, 50)] * 7 + [(R_, 50), (R_, 50), (R_ + '  # w1', 50), (R_ + '  # w2', 50), (R_ + '  # w3', 50), (R_ + '  # w4', 50)]
walk13 = [(R_, 50)] * 13                 # V68B (17:4x UTC): the corrected form; the walk's identical calls execute
game, (sess, sess2), agent, log, go = harness([copy.deepcopy(walk13), [(R_, 50)]], budget=200000, REFRESH=False)
for _ in range(13):
    go()
blocked = events(log, 'v68-loop-frame-blocked')
check('6 a walk (identical normalised reports and code, frames differing): rung 1, no refusal, no escalation (13 RIGHTs)',
      (agent._v6_stats.get('loop_rung1'), agent._v6_stats.get('loop_refused'), events(log, 'v61-restart'),
       [p['reason'] for p in events(log, 'v62-summary-turn')], game.engine.count('RESET'), len(game.engine)),
      (1, None, [], [], 0, 13))
check('6 the receipts (identical calls executed after the block; one blocked trigger per report from the 8th) and the event',
      (agent._v6_stats.get('loop_refusal_frame_blocked'), agent._v6_stats.get('loop_frame_blocked'), sorted(blocked[0]),
       [(p['p'], p['k'], p['consecutive'], p['level'], p['attempt']) for p in blocked]),
      (6, 3, ['attempt', 'consecutive', 'k', 'level', 'p', 'refused'], [(1, 10, 8, 1, 1), (1, 11, 9, 1, 1), (1, 12, 10, 1, 1)]))
A.LOOP_FRAME68 = False
game, (sess, sess2), agent, log, go = harness([copy.deepcopy(walk), [(R_, 50)]], budget=200000, REFRESH=False)
for _ in range(9):
    go()
check('6 the same walk without LOOP_FRAME escalates (the 2nd refusal: a hard loop restart)',
      [(p['reason'], p['kind']) for p in events(log, 'v61-restart')], [('loop', 'hard')])
A.LOOP_FRAME68 = True
real_board, band = board, [0]


def board(pos):  # noqa: F811  (the fakes' board with an outer-band pixel that moves at every board drawn)
    g = real_board(pos)
    g[0][band[0] % 64] = 5
    band[0] += 1
    return g


game, (sess, sess2), agent, log, go = harness([copy.deepcopy(CONV2), [(R_, 50)]], budget=200000, REFRESH=False)
for _ in range(12):
    go()
starts = [e.frame.grid for e in game.history_entries[:11:2]]          # the board before each RIGHT (each cycle's start)
board = real_board
check('6 a cycle whose frames differ only in the outer band escalates (whole boards differ, interiors equal)',
      ([(p['reason'], p['kind']) for p in events(log, 'v61-restart')], len({json.dumps(g) for g in starts}) > 1,
       len({json.dumps(A._v68_interior(SimpleNamespace(grid=g))) for g in starts}), agent._v6_stats.get('loop_frame_blocked')),
      ([('loop', 'hard')], True, 1, None))

# === 7. REFRESH_TEXTS =======================================================================================================
A.REFRESH = True
game, (sess, sess2), agent, log, go = harness([[(PROG + R_, 40), (U_, 40), (SUM1, 60)], [(R_, 30), (R_, 30)]], budget=200000)
go()
go()
sess.last_receipt = {'logical_tokens': 600000}
for _ in range(2):
    go()
REQ = ("This conversation ends after this turn, and a new conversation starts at level {level}'s first board, the current board "
       "(no RESET; a harness step, not a game event). Actions are not executed in this turn. In one python call, save your "
       "summary of level {prev} with save_summary(text) as asked above; it is shown in the new conversation, and your programs "
       "stay loaded.")
LINE = ("This conversation starts at level {level}'s first board. The harness started a new conversation at this level-up; no "
        "RESET was done.")
op = user_text(sess.sent[2])
newconv = user_text(sess2.sent[0])
check('7 the texts exact; the refusal unchanged', (A.TEXTS['refresh_summary_request'] == REQ, A.TEXTS['takeover_refresh'] == LINE,
      A.TEXTS['refresh_turn_refusal'] == build(ARC3_V68_REFRESH_TEXTS=None)['refresh_turn_refusal']), (True, True, True))
check('7 the summary turn\'s request; the new conversation: the pinned summary first, the line, the ledger, no never line',
      (A._fill(REQ, level=2, prev=1) in op, agent._v6_stats.get('refreshes'), newconv.startswith('Your summary of level 1'),
       A._fill(LINE, level=2) in newconv, A._fill(A.TEXTS['notes_ledger'], k=1, action='UP', actions='RIGHT') in newconv,
       'never executed' in newconv, newconv.index('- UP completes the level') < newconv.index(A._fill(LINE, level=2))),
      (True, 1, True, True, True, False, True))

# === 8. all V68 switches off: identical to the tree before V68 ============================================================
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


def run(scripts, turns, before=None, **kw):
    game, sessions, agent, log, go = harness(scripts, **kw)
    for i in range(turns):
        if before and i in before:
            before[i](game, sessions, agent)
        go()
    events = []
    for k, p in log:
        p = json.loads(json.dumps(p, default=str))
        if k == 'tool-output':
            p.pop('seconds', None)
        events.append([k, norm(p)])
    keys = ('v6_level_attempts', 'v62_level_attempt_ends', 'v6_attempt_actions', 'v67_level_restarts')
    return dict(sent=[norm(s.sent) for s in sessions], engine=game.engine, events=events,
                stats=json.loads(json.dumps(agent._v6_stats, default=str)),
                records=json.loads(json.dumps({k: game.__dict__.get(k) for k in keys})), library=sorted(agent._kept_functions))


R_, L_, U_ = 'action(["RIGHT"])', 'action(["LEFT"])', 'action(["UP"])'
NOTES_L = 'save_summary("Notes L1:\\n- RIGHT and LEFT move the marker one cell\\n- the calls repeated a cycle")'
NOTES2 = 'save_summary("Notes L2:\\n- RIGHT moves the marker on level 2\\n- UP clears a level")'
SUM = 'save_summary("Rule L1:\\n- UP completes the level\\n- RIGHT moves the marker")\naction(["RIGHT"])'
runs = {}
runs['levels'] = run([[(PROG + R_, 40), (U_, 40), (SUM, 40), (R_, 500), (L_, 30), (NOTES2, 40)], [(R_, 30), (U_, 30), (R_, 30)]],
                     8, budget=500)
runs['refresh'] = run([[(PROG + R_, 40), (U_, 40), ('save_summary("Rule L1:\\n- UP completes the level\\n- RIGHT moves")', 60),
                        (R_, 40), (R_, 40)], [(R_, 30), (R_, 30)]], 4, budget=200000,
                      before={2: lambda g, ss, a: setattr(ss[0], 'last_receipt', {'logical_tokens': 600000})})
if os.environ.get('ARC3_V67_LOOP') == '1':
    runs['rung2'] = run([[(R_, 50), (L_, 50)] * 5 + [(R_, 40), (L_ + '   \n', 40), (NOTES_L, 60)], [(R_, 50)]], 12, budget=200000)
    runs['walk'] = run([[(R_, 50)] * 9 + [(R_ + '  # w%d' % i, 50) for i in range(4)] + [(NOTES_L, 50)], [(R_, 50), (R_, 50)]],
                       10, budget=200000)
raw = json.load(open(os.path.join(os.path.dirname(A.__file__), 'texts.json')))['texts']
print('IDENT-JSON ' + json.dumps(dict(runs=runs, agent_texts=A.TEXTS, texts=V.apply(json.loads(json.dumps(raw)))),
                                  sort_keys=True, default=str))
'''
GB_STAGE = {'v62/agent.py': '07da14de92d734ed74d1aa29e04aaafb', 'v62/variants.py': 'd7cffaa319f62693969189ba4a88f14e',
            'v62/run_v6.py': '85b66ab270fde1dc21ee917cae4ca084', 'v62/sandbox_ext.py': '9ed872bacacce2a9deda14582df66d4a',
            'v62/solver.py': '5494cedf3f4ee2d117130eb0e757ef10'}
prev = Path(tempfile.mkdtemp()) / 'prev_stage'
shutil.copytree(ROOT / 'v62', prev / 'v62', ignore=shutil.ignore_patterns('__pycache__', '*.pre-*', '*.tmp', '*.new'))
for f in GB_STAGE:
    shutil.copyfile(ROOT / (f + '.pre-v67x'), prev / f)
p = subprocess.run([sys.executable, '-B', str(ROOT.parent / 'apply_v67x.py'), str(prev), 'EXT200,SUMLOOKUP,GAMEBUDGET,LOOPHARD'],
                   capture_output=True, text=True, timeout=600, env=dict(os.environ, PYTHONDONTWRITEBYTECODE='1'))
md5 = {f: hashlib.md5(open(prev / f, 'rb').read()).hexdigest() for f in GB_STAGE}
check('8 the tree before V68, rebuilt by apply_v67x.py\'s earlier items, has the recorded md5s', (p.returncode, md5 == GB_STAGE),
      (0, True))
env_clean = {k: v for k, v in os.environ.items() if not k.startswith(('ARC3_V67_', 'ARC3_V68_'))}
VARIANTS = {'all_off': {}, 'planned_base': dict(PLANNED, ARC3_V62_ZERO_LEVEL_BAND='0', ARC3_V67_GAME_POOL='1000000',
                                                ARC3_V61_RESTART_TOKENS='0')}


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
    rn = outs[('new', 'planned_base')]['runs']
    check('8 the planned-base runs took their paths (a refresh, a loop restart, the walk\'s loop restart without LOOP_FRAME)',
          (rn['refresh']['stats'].get('refreshes'), [e[1]['reason'] for e in rn['rung2']['events'] if e[0] == 'v61-restart'],
           [e[1]['reason'] for e in rn['walk']['events'] if e[0] == 'v61-restart'],
           [e[1]['reason'] for e in rn['levels']['events'] if e[0] == 'v61-restart']), (1, ['loop'], ['loop'], ['tokens']))
print('FAILS:', fails)
