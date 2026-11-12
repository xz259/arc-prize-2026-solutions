"""test_v67_notes.py (2026-10-18, Claude 099768dc): apply_v67.py's items NOTES (ARC3_V67_NOTES=1) and SCORE (ARC3_V67_SCORE_TEXT=1):
  NOTES: the notes request's three headers (token, no-action, position requests; the soft and loop requests derive them);
         the level-clear ledger built from a fake history (8 actions, the attempt start, the first action of an attempt, the
         cap at 10 levels) and shown at a level-up; the action count, ledger and coverage facts in a restart opener; the
         coverage caps; the opener-size receipts; the switch off adds nothing;
  SCORE: the prompt diff only (with ARC3_V66_SCO=1; inert with SCO 0).
The switches off are byte-identical to v66g: tests/test_v67_ident.py and the suite.
usage (dev copy of harness_v6): PYTHONPATH=src:. python3 tests/test_v67_notes.py"""
import json
import os
import tempfile  # noqa: F401  (the fakes' harness)
from pathlib import Path
from types import SimpleNamespace

_src = open(Path(__file__).with_name('test_v67_soft.py')).read()
ENV = eval(_src.split('ENV = ', 1)[1].split('\n\n', 1)[0])
FAKES = _src.split("FAKES = r'''", 1)[1].split("'''", 1)[0]
os.environ.update(ENV)
os.environ['ARC3_V67_NOTES'] = '1'
import v62.patches  # noqa: E402,F401
import v62.retention as r  # noqa: E402
r.install()
import v62.variants as V  # noqa: E402
import v62.agent as A  # noqa: E402
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


OLD_ITEMS = ('and then gives, in at most 10 items, what you established on this level, what you tried and what happened, and the '
             'hypotheses still open.')
NEW_ITEMS = ('and then has three parts under these headers: "Confirmed effects:" (what actions did, with the step numbers where you '
             'saw it), "Untested ideas:" and "Negative claims:" (each claim that something does not work or cannot be reached, with '
             'the step number of the board state it was tested in); at most 10 items in all.')
REQS = ('attempt_summary_request', 'attempt_summary_request_noact', 'attempt_summary_request_position')

# === NOTES texts ===========================================================================================================
on, off = build(ARC3_V67_NOTES='1'), build(ARC3_V67_NOTES='0')
new_keys = sorted(set(on) - set(off))
check('the request format: the three headers in the token, no-action and position requests',
      [on[k] == off[k].replace(OLD_ITEMS, NEW_ITEMS) and off[k].count(OLD_ITEMS) == 1 for k in REQS], [True] * 3)
check('nothing else changes; the new opener texts', (sorted(k for k in off if on[k] != off[k]), new_keys),
      (sorted(REQS), ['notes_action_count', 'notes_coverage', 'notes_ledger', 'notes_ledger_first', 'notes_never']))
both = build(ARC3_V67_NOTES='1', ARC3_V67_RESTART_KIND='soft', ARC3_V67_LOOP='1')
check('the soft and loop requests derive the headers', [NEW_ITEMS in both[k] for k in ('attempt_summary_request_soft',
      'attempt_summary_request_noact_soft', 'attempt_summary_request_position_soft', 'attempt_summary_request_loop',
      'attempt_summary_request_loop_again')], [True] * 5)
check('no directive to re-test', [w for w in ('re-test', 'retest', 'test again', 'try again') if w in json.dumps(
      {k: on[k] for k in new_keys + list(REQS)}).lower()], [])

# === the ledger from a fake history ===========================================================================================
H = lambda act, **res: SimpleNamespace(action=act, result=dict(res), frame=SimpleNamespace(grid=[[0]], level=1, step=0))
hist = [H('')] + [H('RIGHT'), H('DOWN'), H('RIGHT'), H('UP', level_completed=True)]
hist += [H('LEFT')] * 3 + [H('RESET', action_name='RESET')] + [H('RIGHT'), H('LEFT')] * 5 + [H('MOUSE(row=5, col=7)', level_completed=True)]
hist += [H('DOWN', game_over=True), H('RESET', automatic=True, action_name='RESET'), H('SPACE', level_completed=True)]
game, (sess,), agent, log, go = harness([[('action(["RIGHT"])', 10)]])
lines = agent._v67_ledger_lines(hist)
check('the ledger: 8 actions at most, the attempt start (after the RESET), the first action of an attempt', lines,
      ['Level 1 was completed by UP after: RIGHT, DOWN, RIGHT.',
       'Level 2 was completed by MOUSE(row=5, col=7) after: RIGHT, LEFT, RIGHT, LEFT, RIGHT, LEFT, RIGHT, LEFT.',
       'Level 3 was completed by SPACE, the first action of its attempt.'])
many = [H('')] + [x for k in range(12) for x in (H('RIGHT'), H('UP', level_completed=True))]
lines = agent._v67_ledger_lines(many)
check('the ledger cap: the last 10 completed levels', (len(lines), lines[0], lines[-1]),
      (10, 'Level 3 was completed by UP after: RIGHT.', 'Level 12 was completed by UP after: RIGHT.'))
check('receipt ledger_lines', agent._v6_stats.get('ledger_lines'), 13)

# === coverage: distinct boards per valid type, the never-executed types (capped) ============================================
B = lambda v: SimpleNamespace(grid=[[v]], level=1, step=0)
cov = [SimpleNamespace(action='', result={}, frame=B(0))]
for act, board_after in (('RIGHT', 1), ('LEFT', 0), ('RIGHT', 1), ('LEFT', 0), ('RIGHT', 2), ('MOUSE(row=1, col=2)', 3),
                         ('MOUSE(row=4, col=4)', 3)):
    cov.append(SimpleNamespace(action=act, result={}, frame=B(board_after)))
lines = agent._v67_coverage_lines(1, 'UP, DOWN, LEFT, RIGHT, MOUSE', cov)
check('coverage: distinct boards each type was executed from; never-executed types', lines,
      ['Distinct boards (whole board compared) each action type was executed from on this level: LEFT 1, RIGHT 1, MOUSE 2.',
       'Valid action types never executed on this level: UP, DOWN.'])
wide = ', '.join('A%d' % i for i in range(14))
check('coverage cap: at most 10 types', agent._v67_coverage_lines(1, wide, cov)[-1].count(','), 9)

# === in openers: the ledger at a level-up; the count, ledger and coverage after a restart; the opener sizes ==================
game, (sess, sess2), agent, log, go = harness([[('action(["RIGHT"])', 40), ('action(["UP"])', 40), ('action(["RIGHT"])', 600),
                                                ('save_summary("Notes L2:\\n- Confirmed effects: RIGHT moves (step 3)")', 40)],
                                               [('action(["RIGHT"])', 30)]], budget=500)
for _ in range(5):
    go()
check('the level-up opener has the ledger line', 'Level 1 was completed by UP after: RIGHT.' in user_text(sess.sent[2]), True)
first = user_text(sess2.sent[0])
check('the restart opener: the action count (taaf counter), the ledger, the coverage facts',
      ("Actions so far on this level: 2; all of them count for this level's score." in first,
       'Level 1 was completed by UP after: RIGHT.' in first,
       'Distinct boards (whole board compared) each action type was executed from on this level: RIGHT 1.' in first,
       'Valid action types never executed on this level: LEFT, UP, DOWN.' in first), (True, True, True, True))
check('the notes request asks for the three parts', '"Confirmed effects:"' in user_text(sess.sent[3]), True)
st = agent._v6_stats
check('opener-size receipts per kind; the v6-opener chars field',
      ([k for k in ('openers_first', 'openers_same', 'openers_levelup', 'openers_takeover') if not st.get(k)],
       all(st.get('opener_chars_' + k, 0) > 0 and st.get('opener_tokens_' + k, 0) > 0 for k in ('first', 'levelup', 'takeover')),
       all('chars' in p for k, p in log if k == 'v6-opener'), st.get('coverage_lines')), ([], True, True, 2))
A.NOTES = False
game, (sess,), agent, log, go = harness([[('action(["RIGHT"])', 40), ('action(["UP"])', 40), ('action(["RIGHT"])', 40)]])
for _ in range(3):
    go()
check('switch off: no ledger, no size receipts, no chars field', ('was completed by' in user_text(sess.sent[2]),
      agent._v6_stats.get('openers_levelup'), any('chars' in p for k, p in log if k == 'v6-opener')), (False, None, False))
A.NOTES = True

# === SCORE: the prompt diff only ==============================================================================================
OLD_SCO = ("- A cleared level scores in full when it takes no more actions than a hidden per-level baseline, and less the further "
           "it goes over it; later levels count more; an uncleared level scores 0.\n")
NEW_SCO = ("- A cleared level scores 100 x (baseline / actions)^2, capped at 115, where the baseline is a hidden per-level number of "
           "actions; every action on the level counts, including actions before a RESET or a GAME OVER. A game's score is the "
           "average of its level scores weighted by level number (later levels count more), capped at 100 times the weighted share "
           "of its levels that are cleared. An uncleared level scores 0.\n")
s_on, s_off = build(ARC3_V67_SCORE_TEXT='1'), build(ARC3_V67_SCORE_TEXT='0')
check('SCORE: exactly the scoring sentence replaced, nothing else', (s_on['system'] == s_off['system'].replace(OLD_SCO, NEW_SCO),
      s_off['system'].count(OLD_SCO), sorted(k for k in s_on if s_on[k] != s_off.get(k)), set(s_on) == set(s_off)),
      (True, 1, ['system'], True))
check('SCORE is inert with ARC3_V66_SCO=0 (the control keeps v6.5\'s line)',
      build(ARC3_V67_SCORE_TEXT='1', ARC3_V66_SCO='0') == build(ARC3_V67_SCORE_TEXT='0', ARC3_V66_SCO='0'), True)
RUN = open(Path(A.__file__).with_name('run_v6.py')).read()
check('run_v6: the NOTES and SCORE defaults and records', ("os.environ.setdefault('ARC3_V67_NOTES', '0')" in RUN,
      "os.environ.setdefault('ARC3_V67_SCORE_TEXT', '0')" in RUN, "'ARC3_V67_NOTES', 'ARC3_V67_SCORE_TEXT'," in RUN),
      (True, True, True))
print('FAILS:', fails)
