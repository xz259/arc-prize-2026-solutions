"""test_v66g.py (2026-10-18, Claude 099768dc): apply_v66g.py's ARC3_V66_SCO switch, on the test scaffolding of test_v64_fixes.py.
The control of the v6.6 pair (ARC3_V66_SCO=0, ARC3_V66_TEXTS=1, the default 16K / 32K / 48K cues, seeds on) must differ from v6.6
only in the objective line, and must keep the 16K facts-only cue.
usage (dev copy of harness_v6): PYTHONPATH=src:. python3 tests/test_v66g.py"""
import json
from pathlib import Path

_src = open(Path(__file__).with_name('test_v64_fixes.py')).read()
exec(compile(_src.split('\ndef run_notes_turn')[0], 'test_v64_fixes.py (prelude)', 'exec'), globals())
import v62.variants as V  # noqa: E402

RAW = json.load(open(Path(A.__file__).with_name('texts.json')))['texts']
OLD = '- Use as few in-game actions as possible while staying reliable.'
NEW = ('- A cleared level scores in full when it takes no more actions than a hidden per-level baseline, and less the further it '
       'goes over it; later levels count more; an uncleared level scores 0.')
TOOLS = 'tools.py: `crop('


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


t66 = build(ARC3_V66_TEXTS='1', ARC3_V66_SCO='1', ARC3_V66_SEEDS='1')
tc = build(ARC3_V66_TEXTS='1', ARC3_V66_SCO='0', ARC3_V66_SEEDS='1')
t65 = build(ARC3_V66_TEXTS='0', ARC3_V66_SCO='1', ARC3_V66_SEEDS='1')
check('v6.6 default: the scoring fact, not the old line; the 16K cue text; the tools.py line',
      (NEW in t66['system'], OLD in t66['system'], 'noact_warn_0' in t66, TOOLS in t66['system']), (True, False, True, True))
check('control (SCO 0): the old line, not the scoring fact; the 16K cue text; the tools.py line',
      (OLD in tc['system'], NEW in tc['system'], 'noact_warn_0' in tc, TOOLS in tc['system']), (True, False, True, True))
check('control vs v6.6: only the system text differs, and only in that line',
      (sorted(k for k in set(tc) | set(t66) if tc.get(k) != t66.get(k)), tc['system'].replace(OLD, NEW) == t66['system']),
      (['system'], True))
check('control 16K cue text identical to v6.6', tc['noact_warn_0'] == t66['noact_warn_0'], True)
check('control system text = the TEXTS=0 system text (SCO is the only system change of the v6.6 texts)', tc['system'] == t65['system'], True)
check('the switch has no effect with ARC3_V66_TEXTS=0', build(ARC3_V66_TEXTS='0', ARC3_V66_SCO='0', ARC3_V66_SEEDS='1') == t65, True)
# the 16K cue in the control: the facts-only text at 16K, the limit clause at 32K, must-act at 48K
VALID = ['UP', 'RIGHT', 'LEFT']
A.TEXTS = tc
A.NOACT_WARN = (16000, 32000, 48000)
log = []
ag, _, _ = new_agent([('action(["RIGHT"])', 10)], log=log)
ag._v62_noact_tokens, ag._v62_noact_warned = 0, 0
kinds = []
for k in (15999, 16000, 32000, 48000):
    ag._v62_noact_tokens = k
    r = ag._v62_noact_check([], VALID)
    kd, text = r
    kinds.append(kd if kd != 'warn' else ('cue0' if 'If you reach' not in text and text.startswith('You have generated')
                                          else 'warn1' if 'If you reach' in text else 'warn2' if 'must execute an action' in text else '?'))
check('control cues: none, 16K facts-only, 32K limit clause, 48K must-act', kinds, [None, 'cue0', 'warn1', 'warn2'])
R = open(Path(A.__file__).with_name('run_v6.py')).read()
check('RUN: explicit default and the recorded switch', ("os.environ.setdefault('ARC3_V66_SCO', '1')" in R,
                                                       "'ARC3_V66_SCO', 'ARC3_V66_TEXTS', 'ARC3_V66_SEEDS', 'ARC3_V65_BATCH')}," in R), (True, True))
print('FAILS:', fails)
