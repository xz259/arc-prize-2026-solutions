"""test_v67_texts.py (2026-10-18, Claude 099768dc): apply_v67.py's item TEXTS67 (ARC3_V67_TEXTS=1): the text diffs exactly as
specified, every occurrence: the turn-based bullet, the closing-report bullet (with and without v6.5's CLOCK clause) and the
Never-give-up clause, the budget-restored sentence of noact_warn_1 / noact_warn_2 (and the soft variant derived from it),
the resumption's token count (rendered by the agent, with its receipt); nothing else changes. The switch off is
byte-identical to v66g: tests/test_v67_ident.py and the suite.
usage (dev copy of harness_v6): PYTHONPATH=src:. python3 tests/test_v67_texts.py"""
import json
import os
import tempfile  # noqa: F401  (the fakes' harness)
from pathlib import Path
from types import SimpleNamespace  # noqa: F401

_src = open(Path(__file__).with_name('test_v67_soft.py')).read()
ENV = eval(_src.split('ENV = ', 1)[1].split('\n\n', 1)[0])
FAKES = _src.split("FAKES = r'''", 1)[1].split("'''", 1)[0]
os.environ.update(ENV)
os.environ['ARC3_V67_TEXTS'] = '1'
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


C3_OLD = ("- The game is turn-based: it advances only when one of your actions executes, and nothing in the game changes while you "
          "think or between your calls.\n")
C3_NEW = ("- The game is turn-based: it advances only when one of your actions executes, nothing in the game changes while you "
          "think or between your calls, and a game never ends or times out for lack of actions; only the harness ends a "
          "conversation at its token budget, or a game.\n")
END_OLD = "- No closing report is expected: keep playing until the harness ends the game."
END_NEW = "- Continue working on the level until the harness ends the game; no closing report is expected."
END_OLD_CLOCK = "- No closing report is expected: keep playing until the harness ends the game, however little run time is left."
END_NEW_CLOCK = ("- Continue working on the level until the harness ends the game, however little run time is left; no closing "
                 "report is expected.")
NGU_OLD, NGU_NEW = "- Never give up: only the harness ends a game. When you are stuck,", "- Never give up. When you are stuck,"
B1, B2 = " The budget is restored as soon as an action executes.", " The budget is restored as soon as one action executes."
FINE, COUNT = "That is fine: it computed or inspected state.", "{t} tokens generated since the last executed action."

for label, env in (('production', {}), ('clock', {'ARC3_V65_CLOCK': '1'}), ('v65 batch', {'ARC3_V65_BATCH': '1',
                   'ARC3_STALE_STATE_BLOCK': '0'}), ('soft + loop', {'ARC3_V67_RESTART_KIND': 'soft', 'ARC3_V67_LOOP': '1'})):
    on, off = build(ARC3_V67_TEXTS='1', **env), build(ARC3_V67_TEXTS='0', **env)
    changed = sorted(k for k in set(on) | set(off) if on.get(k) != off.get(k))
    want_changed = ['noact_warn_1', 'noact_warn_2', 'resumption', 'system'] + (['noact_warn_2_soft'] if 'ARC3_V67_LOOP' in env else [])
    check('%s: exactly these texts change' % label, changed, sorted(want_changed))
    eo, en = (END_OLD_CLOCK, END_NEW_CLOCK) if 'ARC3_V65_CLOCK' in env else (END_OLD, END_NEW)
    check('%s: system = the three replacements, each once' % label,
          (off['system'].count(C3_OLD), off['system'].count(eo), off['system'].count(NGU_OLD),
           on['system'] == off['system'].replace(C3_OLD, C3_NEW).replace(eo, en).replace(NGU_OLD, NGU_NEW)), (1, 1, 1, True))
    check('%s: the cues and the resumption' % label,
          (on['noact_warn_1'] == off['noact_warn_1'].replace(B1, ''), off['noact_warn_1'].count(B1),
           on['noact_warn_2'] == off['noact_warn_2'].replace(B2, ''), off['noact_warn_2'].count(B2),
           on['noact_warn_2_nolimit'] == off['noact_warn_2_nolimit'], on['resumption'] == off['resumption'].replace(FINE, COUNT),
           (on['resumption_error'], on['resumption_noact']) == (off['resumption_error'], off['resumption_noact'])),
          (True, 1, True, 1, True, True, True))
    texts = json.dumps(on)
    check('%s: no occurrence left; "only the harness ends" once' % label,
          (texts.count('keep playing'), texts.count('budget is restored'), texts.count('That is fine'),
           on['system'].count('only the harness ends'), on['system'].count('no closing report is expected'),
           on['system'].count('No closing report')), (0, 0, 0, 1, 1, 0))
    if 'ARC3_V67_LOOP' in env:
        check('%s: the soft must-act cue derived without the sentence' % label,
              on['noact_warn_2_soft'] == off['noact_warn_2_soft'].replace(B2, ''), True)

# the agent renders the count and keeps its counters
game, (sess,), agent, log, go = harness([[('action(["RIGHT"])', 10)]], warn=(100, 200, 300), limit=400)
go()
cf, he = load_runtime_state(game.state_path)
agent._step_env_callback = game.step_env
agent._v62_noact_tokens, agent._v62_noact_warned, agent._v62_resume_error = 12345, 0, False
text = agent._v6_resumption_text(cf, VALID, he)
check('the resumption: the count, not "That is fine"; one receipt',
      ('Your last Python call executed no actions. 12,345 tokens generated since the last executed action. The board is '
       'unchanged.' in text, 'That is fine' in text, agent._v6_stats.get('resumption_counts')), (True, False, 1))
agent._v62_noact_warned = 1
text = agent._v6_resumption_text(cf, VALID, he)
check('the no-action resumption is unchanged (no count, its own counter once)',
      (text.startswith('Your last Python call executed no actions. The board is unchanged') and 'tokens generated' not in text,
       agent._v6_stats.get('resumption_counts'),
       agent._v6_stats.get('resumption_noact')), (True, 1, 1))
cues = []
for k, w in ((250, 1), (350, 2)):
    agent._v62_noact_tokens, agent._v62_noact_warned = k, w
    cues.append(agent._v62_noact_check(he, VALID)[1])
agent._step_env_callback = None
check('the rendered cues have no budget sentence', ([('budget is restored' in c) for c in cues],
      cues[0].startswith('You have generated 250 tokens'), cues[1].startswith('350 of 400 tokens')), ([False, False], True, True))
RUN = open(Path(A.__file__).with_name('run_v6.py')).read()
check('run_v6: the explicit default and the recorded switch', ("os.environ.setdefault('ARC3_V67_TEXTS', '0')" in RUN,
      "'ARC3_V67_LOOP', 'ARC3_V67_TEXTS'," in RUN), (True, True))
print('FAILS:', fails)
