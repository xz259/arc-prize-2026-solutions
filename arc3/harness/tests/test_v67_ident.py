"""test_v67_ident.py (2026-10-18, Claude 099768dc): apply_v67.py with every v6.7 switch off (the defaults) is byte-identical to
the v66g base. The same scripted runs execute in a subprocess on the base tree (the patcher's md5-checked copies
v62/*.pre-v67) and on the patched tree, and everything the model receives (every message, images hashed), the game actions,
the events (the documented new keys of the v61-restart event aside), the stats (hard_restarts aside), the attempt records
and the library must match; the rendered texts must match in several environments. The runs cover the paths the v6.7
items touch: a cycling run (C1 lines, periods 2 and 6), model RESETs (with and without actions before them, after a GAME
OVER), level-ups (openers, summaries), a budget restart with a refused action and a no-action stretch with its cues.
test_v67_soft.py section 7 runs the restart paths the same way.
usage (dev copy of harness_v6): PYTHONPATH=src:. python3 tests/test_v67_ident.py"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_src = open(Path(__file__).with_name('test_v67_soft.py')).read()
ENV = eval(_src.split('ENV = ', 1)[1].split('\n\n', 1)[0])
FAKES = _src.split("FAKES = r'''", 1)[1].split("'''", 1)[0]
fails = []


def check(name, got, want):
    ok = got == want
    print('%-84s -> %-40r %s' % (name, got if not isinstance(got, str) else got[:40], 'OK' if ok else 'WRONG (want %r)' % (want,)))
    if not ok:
        fails.append(name)


IDENT = r'''
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
os.environ.update(json.loads(sys.argv[1]))
import v62.patches  # noqa: E402,F401
import v62.retention as r  # noqa: E402
r.install()
import v62.variants as V  # noqa: E402
import v62.agent as A  # noqa: E402
exec(sys.argv[2])
NEW_KEYS = ('kind', 'hard_why', 'actions_on_level')


def norm(x):
    if isinstance(x, dict):
        if x.get('type') == 'image_url':
            return dict(type='image_url', sha1=hashlib.sha1(json.dumps(x, sort_keys=True).encode()).hexdigest())
        return {k: norm(v) for k, v in x.items()}
    if isinstance(x, list):
        return [norm(v) for v in x]
    return x


def collect(game, sessions, agent, log):
    events, new = [], []
    for k, p in log:
        p = json.loads(json.dumps(p, default=str))
        if k == 'tool-output':
            p.pop('seconds', None)
        if k == 'v61-restart':
            new.append({x: p.pop(x) for x in NEW_KEYS if x in p})
        events.append([k, norm(p)])
    stats = json.loads(json.dumps(agent._v6_stats, default=str))
    new.append(dict(hard_restarts=stats.pop('hard_restarts', None)))
    keys = ('v6_level_attempts', 'v61_level_attempt_noops', 'v62_level_attempt_ends', 'v6_attempt_actions', 'v62_since_start',
            'v62_level_start_returns', 'v61_harness_resets', 'v61_voluntary_resets')
    return dict(sent=[norm(s.sent) for s in sessions], engine=game.engine, events=events, stats=stats, new=new,
                records=json.loads(json.dumps({k: game.__dict__.get(k) for k in keys})),
                library=sorted(agent._kept_functions), archive=sorted(agent._v61_archive))


def run(scripts, turns, before=None, **kw):
    game, sessions, agent, log, go = harness(scripts, **kw)
    for i in range(turns):
        if before and i in before:
            before[i](game, sessions, agent)
        go()
    return collect(game, sessions, agent, log)


R_, L_ = 'action(["RIGHT"])', 'action(["LEFT"])'
runs = {}
# a cycling run: period 2 for 14 turns (C1 lines from the 7th report), then a period-6 stretch
runs['cycle2'] = run([[(R_, 40), (L_, 40)] * 7 + [(R_, 40)]], 15, budget=200000)
p6 = ['action(["RIGHT"])', 'action(["RIGHT"])', 'action(["RIGHT"])', 'action(["LEFT"])', 'action(["LEFT"])', 'action(["LEFT"])']
runs['cycle6'] = run([[(c, 30) for c in p6 * 4]], 24, budget=200000)
# model RESETs: after actions, at the level's start (no-op RESET), after a GAME OVER (the automatic restart)
runs['resets'] = run([[(R_, 40), ('action(["RESET"])', 40), ('action(["RESET"])', 40), ('action(["RIGHT", "RIGHT"])', 40),
                       ('action(["DOWN"])', 40), ('action(["RESET", "RIGHT"])', 40), (R_, 40)]], 7, budget=200000)
# level-ups with summaries (save_summary at the level-up), a program, a budget restart on level 2 with notes
SUM = 'save_summary("Rule L1:\\n- UP completes the level\\n- RIGHT moves the marker")\naction(["RIGHT"])'
NOTES2 = 'save_summary("Notes L2:\\n- RIGHT moves the marker on level 2\\n- UP clears a level")'
runs['levels'] = run([[(PROG + 'action(["RIGHT"])', 40), ('action(["UP"])', 40), (SUM, 40), ('action(["RIGHT"])', 500),
                       ('action(["LEFT"])', 30), (NOTES2, 40)], [(R_, 30), ('action(["UP"])', 30), (R_, 30)]], 8, budget=500)
# a level-up at 600K logical positions (REFRESH off: the level-up turn as before)
runs['levelup_long'] = run([[(R_, 40), ('action(["UP"])', 40), (SUM, 40), (R_, 40)]], 4, budget=200000,
                           before={2: lambda g, ss, a: setattr(ss[0], 'last_receipt', {'logical_tokens': 600000})})
# a no-action stretch: the three cues, then the action
runs['noact'] = run([[('print(1)', 150), ('print(2)', 100), ('print(3)', 100), (R_, 20), (R_, 20)]], 2,
                    warn=(100, 200, 300), limit=400)
raw = json.load(open(Path(A.__file__).with_name('texts.json')))['texts']
texts = {}
for name, env in (('production', {}), ('v65_batch', {'ARC3_V65_BATCH': '1', 'ARC3_STALE_STATE_BLOCK': '0'}),
                  ('control_sco0_seeds0', {'ARC3_V66_SCO': '0', 'ARC3_V66_SEEDS': '0'}), ('v66_texts_off', {'ARC3_V66_TEXTS': '0'}),
                  ('no_reset', {'EXPOSE_RESET': 'off'}), ('lean', {'ARC3_V6_LEAN': '1'}), ('a45', {'ARC3_V6_A45': '1'}),
                  ('clock', {'ARC3_V65_CLOCK': '1'}), ('summary_off', {'ARC3_V61_SUMMARY': '0'})):
    keep = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    texts[name] = V.apply(json.loads(json.dumps(raw)))
    for k, v in keep.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
print('IDENT-JSON ' + json.dumps(dict(runs=runs, texts=texts, agent_texts=A.TEXTS, system=A.TOOL), sort_keys=True,
                                  default=str))
'''

root = Path(__file__).resolve().parents[1]
BASE = ('v62/agent.py', 'v62/variants.py', 'v62/run_v6.py', 'v62/session_ext.py', 'v62/sandbox_ext.py')
missing = [f for f in BASE if not (root / (f + '.pre-v67')).exists()]
check('the v66g base copies (v62/*.pre-v67, written by apply_v67.py) are present', missing, [])
if not missing:
    base = Path(tempfile.mkdtemp()) / 'v66g_base'
    shutil.copytree(root / 'v62', base / 'v62', ignore=shutil.ignore_patterns('__pycache__', '*.pre-*', '*.tmp'))
    for f in BASE:
        shutil.copyfile(root / (f + '.pre-v67'), base / f)
    env_off = {k: v for k, v in os.environ.items() if not k.startswith('ARC3_V67_')}

    def run_tree(tree):
        env = dict(env_off, PYTHONPATH='%s:%s' % (root / 'src', tree))
        p = subprocess.run([sys.executable, '-c', IDENT, json.dumps(ENV), FAKES], env=env, cwd=str(tree), capture_output=True,
                           text=True, timeout=1200)
        line = [x for x in p.stdout.splitlines() if x.startswith('IDENT-JSON ')]
        if p.returncode or not line:
            print('identity run failed in %s (rc %s):\n%s' % (tree, p.returncode, p.stderr[-3000:]))
            return None
        return json.loads(line[-1][len('IDENT-JSON '):])
    out_base, out_new = run_tree(base), run_tree(root)
    check('both identity runs completed', (out_base is not None, out_new is not None), (True, True))
    if out_base and out_new:
        check('texts: the switches off render v66g\'s texts in %d environments' % len(out_base['texts']),
              [k for k in out_base['texts'] if out_base['texts'][k] != out_new['texts'][k]], [])
        check('texts: the agent\'s TEXTS and tool identical', (out_base['agent_texts'] == out_new['agent_texts'],
                                                             out_base['system'] == out_new['system']), (True, True))
        keys = ('sent', 'engine', 'events', 'stats', 'records', 'library', 'archive')
        for name in out_base['runs']:
            a, b = out_base['runs'][name], out_new['runs'][name]
            check('%s: model traffic, game actions, events, stats, records identical' % name, [k for k in keys if a[k] != b[k]], [])
            check('%s: the new restart keys: every restart hard, no hard_why; hard_restarts = restarts' % name,
                  (all(x.get('kind') == 'hard' and x.get('hard_why') is None for x in b['new'][:-1]),
                   b['new'][-1]['hard_restarts'] == b['stats'].get('restarts'), a['new'][-1]['hard_restarts']), (True, True, None))
        rn = out_new['runs']
        check('the runs took the intended paths (C1 lines; model RESETs; a restart after two level-ups; the three cues)',
              (rn['cycle2']['stats'].get('loop_lines', 0) > 0, rn['cycle6']['stats'].get('loop_lines', 0) > 0,
               (rn['resets']['stats'].get('voluntary_resets') or 0) >= 2, rn['levels']['stats'].get('restarts'),
               rn['levels']['stats'].get('rules_recorded'), [rn['noact']['stats'].get('noact_warnings_%d' % i) for i in (1, 2, 3)]),
              (True, True, True, 1, 1, [1, 1, 1]))
print('FAILS:', fails)
