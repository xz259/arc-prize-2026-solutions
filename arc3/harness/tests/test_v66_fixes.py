"""v6.6 fixes (apply_v66.py), on the test scaffolding of test_v64_fixes.py (its prelude up to run_notes_turn is executed here):
SCO (the scoring fact replaces the action-minimising objective line), NA3 (the 16K facts-only cue before v6.5's two), RUN.
usage (dev copy of harness_v6): PYTHONPATH=src:. python3 tests/test_v66_fixes.py"""
import json
from pathlib import Path

_src = open(Path(__file__).with_name('test_v64_fixes.py')).read()
exec(compile(_src.split('\ndef run_notes_turn')[0], 'test_v64_fixes.py (prelude)', 'exec'), globals())
import v62.variants as V  # noqa: E402

RAW = json.load(open(Path(A.__file__).with_name('texts.json')))['texts']
OLD = '- Use as few in-game actions as possible while staying reliable.'
NEW = ('- A cleared level scores in full when it takes no more actions than a hidden per-level baseline, and less the further it '
       'goes over it; later levels count more; an uncleared level scores 0.')

# === SCO =================================================================================================================
S = A.TEXTS['system']
check('SCO default: the scoring fact in the Objective section', (NEW in S, OLD in S, S.index(NEW) < S.index('## Each turn')),
      (True, False, True))
check('SCO default: it follows the WIN line and precedes the levels bullet',
      (S.index('WIN means the whole game is solved.') < S.index(NEW) < S.index('- Levels usually build on')), True)
os.environ['ARC3_V66_TEXTS'] = '0'
t0 = V.apply(json.loads(json.dumps(RAW)))
os.environ['ARC3_V66_TEXTS'] = '1'
t1 = V.apply(json.loads(json.dumps(RAW)))
check('SCO off: the v6.5 line, no fact, no noact_warn_0', (OLD in t0['system'], NEW in t0['system'], 'noact_warn_0' in t0),
      (True, False, False))
check('SCO on vs off: the system texts differ only in that line', t1['system'].replace(NEW, OLD) == t0['system'], True)
check('SCO on vs off: only system and noact_warn_0 differ', sorted(k for k in set(t0) | set(t1) if t0.get(k) != t1.get(k)),
      ['noact_warn_0', 'system'])
check('SCO the cue text: facts only (no limit clause, no advice)', (t1['noact_warn_0'].count('{k}'), '{untried}' in t1['noact_warn_0'],
                                                                    'limit' in t1['noact_warn_0'], 'must' in t1['noact_warn_0']),
      (1, True, False, False))

# === NA3 =================================================================================================================
check('NA3 default thresholds 16K / 32K / 48K, restart 64K', (A.NOACT_WARN, A.NOACT_TOKENS), ((16000, 32000, 48000), 64000))
VALID = ['UP', 'RIGHT', 'LEFT']


def run_cues(thresholds, ks):
    A.NOACT_WARN = tuple(thresholds)
    log = []
    ag, _, _ = new_agent([('action(["RIGHT"])', 10)], log=log)
    ag._v62_noact_tokens, ag._v62_noact_warned = 0, 0
    out = []
    for k in ks:
        ag._v62_noact_tokens = k
        out.append(ag._v62_noact_check([], VALID))
    stats = {kk: v for kk, v in ag._v6_stats.items() if kk.startswith('noact_warnings')}
    logs = [(p['which'], p['threshold']) for kk, p in log if kk == 'v62-noact-warning']
    return out, stats, logs


def kind(r):
    kd, text = r
    if kd != 'warn':
        return kd
    if text.startswith('You have generated') and 'If you reach' not in text:
        return 'cue0'
    if text.startswith('You have generated') and 'If you reach' in text:
        return 'warn1'
    if 'must execute an action' in text:
        return 'warn2'
    return '?'


out, stats, logs = run_cues((16000, 32000, 48000), [15999, 16000, 17000, 32500, 33000, 48000, 60000, 64000])
check('NA3 three cues: none, 16K facts, -, 32K limit clause, -, 48K must-act, -, limit',
      [kind(r) for r in out], [None, 'cue0', None, 'warn1', None, 'warn2', None, 'limit'])
check('NA3 the 16K text: facts and the untried actions, nothing else', out[1][1],
      'You have generated 16,000 tokens in this conversation since your last executed action. Actions not yet executed on this '
      'level in this conversation: UP, RIGHT, LEFT.')
check('NA3 counters per threshold and the log', (stats, logs),
      ({'noact_warnings_1': 1, 'noact_warnings_2': 1, 'noact_warnings_3': 1}, [(1, 16000), (2, 32000), (3, 48000)]))
out, stats, logs = run_cues((16000, 32000, 48000), [40000])
check('NA3 a jump past 16K and 32K at once: one cue, the 32K text', ([kind(r) for r in out], logs), (['warn1'], [(2, 32000)]))
out, stats, logs = run_cues((32000, 48000), [16000, 32000, 48000, 64000])
check('NA3 v6.5 thresholds: v6.5 behaviour (none, limit-clause cue, must-act, limit)', [kind(r) for r in out],
      [None, 'warn1', 'warn2', 'limit'])
check('NA3 v6.5 thresholds: v6.5 counter keys', stats, {'noact_warnings_1': 1, 'noact_warnings_2': 1})
saved = A.TEXTS.pop('noact_warn_0')
out, stats, logs = run_cues((16000, 32000, 48000), [16000])
check('NA3 three thresholds without the v6.6 text (ARC3_V66_TEXTS=0): the 16K cue falls back to the limit-clause text',
      [kind(r) for r in out], ['warn1'])
A.TEXTS['noact_warn_0'] = saved
A.NOACT_WARN = (16000, 32000, 48000)

# === RUN =================================================================================================================
R = open(Path(A.__file__).with_name('run_v6.py')).read()
check('RUN explicit defaults and the recorded switch', ("os.environ.setdefault('ARC3_V66_TEXTS', '1')" in R,
                                                        "os.environ.setdefault('ARC3_V62_NOACT_WARN', '16000,32000,48000')" in R,
                                                        "'ARC3_V66_TEXTS'" in R.split("'ARC3_V62_RESTART_MIN_LEFT'")[1]), (True, True, True))
print('FAILS:', fails)
