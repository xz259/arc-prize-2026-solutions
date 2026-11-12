"""test_v67x_sumlookup.py (2026-10-18, Claude 099768dc, implementation subagent): apply_v67x.py's item SUMLOOKUP
(ARC3_V67_SUMLOOKUP, user decision 53), through agent.analyze with the real sandbox on test_v67_soft.py's fakes:
  0 the switch: default off; run_v6's explicit default and record; the texts guard;
  1 the accessor: summaries() returns the saved level summaries as {level: text}, as stored (a truncated one included); a
    summary saved in a call appears from the next call on; the result is a copy (mutating it changes nothing); the call's
    initial state carries the snapshot; with the switch off the sandbox has no summaries and the call's state is unchanged;
  2 the texts: under the switch, the system-prompt bullet after save_summary's and the pointer in the two not-shown lines,
    nothing else, in four environments; the restart openers on levels 3 and 4 (singular, plural) carry the pointer;
  3 the receipts: summaries_calls counts every call; the v67-sumlookup event (level, n_summaries) for the first call in each
    conversation;
  4 the switch off is byte-identical to the deployed v6.7 base (v62/*.pre-v67x, a subprocess on each tree), with the v6.7
    switches off and with the core switches on: model traffic, game actions, events, stats, records, library, the sandbox's
    per-call state, the texts.
usage (dev copy of harness_v6): PYTHONPATH=src:. python3 tests/test_v67x_sumlookup.py"""
import copy  # noqa: F401
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
for _k in [k for k in os.environ if k.startswith('ARC3_V67_')]:
    os.environ.pop(_k)
os.environ['ARC3_V67_SUMLOOKUP'] = '1'                  # the switch-on texts and accessor in this process
import v62.patches  # noqa: E402,F401
import v62.retention as r  # noqa: E402
r.install()
import v62.variants as V  # noqa: E402
import v62.agent as A  # noqa: E402
import v62.sandbox_ext as SBX  # noqa: E402
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


def printed(log):
    """{tag: rest} from the stdout lines 'TAG rest' of every python call's result (the tool-output events' JSON content)"""
    out = {}
    for p in events(log, 'tool-output'):
        try:
            stdout = json.loads(p.get('content') or '').get('stdout') or ''
        except (TypeError, ValueError, AttributeError):
            continue
        for line in stdout.splitlines():
            tag, _, rest = line.partition(' ')
            out.setdefault(tag, rest)
    return out


STATES = []
_orig_run = SBX._original_run


def _spy(**kw):
    STATES.append(copy.deepcopy(((kw.get('initial_state') or {}).get('v6') or {}).get('summaries', 'ABSENT')))
    return _orig_run(**kw)


SBX._original_run = _spy
R_, U_ = 'action(["RIGHT"])', 'action(["UP"])'


def rule(k, extra=''):
    return 'save_summary("Rule L%d:\\n- UP completes the level\\n- RIGHT moves the marker on level %d")\n%s' % (k, k, extra)


def h(scripts, on=True, levels=5, **kw):
    """harness() with the switch and a game of `levels` levels"""
    out = harness(scripts, SUMLOOKUP=on, **kw)
    out[0].game = _TG(levels)
    return out


# === 0. the switch =======================================================================================================
RUN = open(Path(A.__file__).with_name('run_v6.py')).read()
check('0 the switch on in this process; the agent and variants agree', (A.SUMLOOKUP, V.sumlookup_on()), (True, True))
check('0 run_v6: the explicit default and the recorded switch',
      ("os.environ.setdefault('ARC3_V67_SUMLOOKUP', '0')" in RUN,
       "'ARC3_V67_SUMLOOKUP'," in RUN.split('variants=')[1].split('history_eviction=')[0]), (True, True))
check('0 the switch needs the v6.1 texts (the guard)', "ARC3_V67_SUMLOOKUP=1 needs the v6.1 texts" in open(A.__file__).read(),
      True)

# === 1. the accessor =====================================================================================================
LONG = ('save_summary("Rule L2:\\n" + "\\n".join("- item %d: the blocks slide until they meet a wall or another block" % i '
        'for i in range(900)))')
PROBE = 'import json\nprint("{tag}", json.dumps(sorted(summaries().items())))\n'
conv = [(R_, 40), (U_, 40),
        (PROBE.format(tag='A') + rule(1) + '\n' + PROBE.format(tag='B'), 40),          # saved in this call: not visible yet
        (PROBE.format(tag='C') + 's = summaries()\ns[1] = "changed"\ns[7] = "new"\ns.clear()\n' + PROBE.format(tag='D') + R_, 40),
        (U_, 40), (LONG, 40),
        ('s = summaries()[2]\nprint("E", len(s), sum(map(ord, s)) % 1000003, s.endswith("- ..."))\n' + R_, 40)]
STATES.clear()
game, (sess,), agent, log, go = h([conv])
for _ in range(5):
    go()
got = printed(log)
stored1 = agent._v6_rules.get(1)
check('1 before any save: {}; saved in the same call: still {} (visible from the next call)', (got.get('A'), got.get('B')),
      ('[]', '[]'))
check('1 the next call: {1: the summary as stored}', json.loads(got.get('C') or 'null'), [[1, stored1]])
check('1 a copy: mutating it changes nothing for the next summaries()', json.loads(got.get('D') or 'null'), [[1, stored1]])
stored2 = agent._v6_rules.get(2) or ''
check('1 a truncated summary is returned as stored (length, checksum, the cut marker)',
      (got.get('E'), stored2.endswith('- ...')), ('%d %d True' % (len(stored2), sum(map(ord, stored2)) % 1000003), True))
check('1 the call\'s initial state carries the snapshot ([[level, text]]) of the summaries saved before it',
      STATES, [[], [], [], [[1, stored1]], [[1, stored1]], [[1, stored1]], [[1, stored1], [2, stored2]]])
check('1 the host\'s summaries were not touched by the mutation', sorted(agent._v6_rules), [1, 2])
HAS = 'try:\n    summaries\n    print("S", True)\nexcept NameError:\n    print("S", False)\n'
game, (sess,), agent, log, go = h([[(HAS + R_, 40)]], on=False)
STATES.clear()
go()
check('1 switch off (agent): no summaries in the sandbox, no snapshot in the call\'s state',
      (printed(log).get('S'), STATES, game.engine), ('False', ['ABSENT'], ['RIGHT']))

# === 2. the texts ========================================================================================================
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


BULLET = "- `summaries()` returns your saved level summaries as {level: text}.\n"
ENVS = {'production': {}, 'core': dict(ARC3_V67_LOOP='1', ARC3_V67_TEXTS='1', ARC3_V67_NOTES='1', ARC3_V67_SCORE_TEXT='1'),
        'soft_refresh': dict(ARC3_V67_RESTART_KIND='soft', ARC3_V67_REFRESH_LOGICAL='500000'),
        'v65_texts_off': dict(ARC3_V65_TEXTS='0')}
diffs = {}
for name, env in ENVS.items():
    on, off = build(ARC3_V67_SUMLOOKUP='1', **env), build(ARC3_V67_SUMLOOKUP='0', **env)
    keys = sorted(k for k in set(on) | set(off) if on.get(k) != off.get(k))
    exact = (on['system'] == off['system'].replace('at other times it saves nothing.\n', 'at other times it saves nothing.\n' + BULLET)
             and on['system'].count(BULLET) == 1
             and on.get('v65_summaries_not_shown') == (None if 'v65_summaries_not_shown' not in off else
                                                       off['v65_summaries_not_shown'][:-1] + '; `summaries()` returns them.')
             and on.get('v65_summary_not_shown') == (None if 'v65_summary_not_shown' not in off else
                                                     off['v65_summary_not_shown'][:-1] + '; `summaries()` returns it.'))
    diffs[name] = (keys, exact)
check('2 under the switch only the system bullet and the two not-shown lines change, exactly (4 environments)', diffs, {
    'production': (['system', 'v65_summaries_not_shown', 'v65_summary_not_shown'], True),
    'core': (['system', 'v65_summaries_not_shown', 'v65_summary_not_shown'], True),
    'soft_refresh': (['system', 'v65_summaries_not_shown', 'v65_summary_not_shown'], True),
    'v65_texts_off': (['system'], True)})
check('2 the bullet follows the save_summary bullet', A.TEXTS['system'].split('\n')[[i for i, l in enumerate(
      A.TEXTS['system'].split('\n')) if l.startswith('- `save_summary(text)`')][0] + 1], BULLET.rstrip('\n'))
NOTES = 'save_summary("Notes L{k}:\\n- the marker moves right one cell per RIGHT\\n- UP completes the level")'
# level 4: summaries of 1-3 saved; a budget restart on level 4; the takeover names levels 1 and 2 as not shown
conv4 = [(R_, 40), (U_, 40), (rule(1, R_), 40), (U_, 40), (rule(2, R_), 40), (U_, 40),
         (rule(3, 'x = summaries()\n' + R_), 40), (R_, 600), (NOTES.format(k=4), 60)]
game, (sess, sess2), agent, log, go = h([conv4, [('x = summaries()\ny = summaries()\n' + R_, 50), (R_, 50)]], budget=500)
for _ in range(10):
    go()
take4 = user_text(sess2.sent[0])
check('2 level 4\'s restart opener: "Summaries of levels 1 and 2 are not shown ...; `summaries()` returns them."',
      (game.level, A._fill(A.TEXTS['v65_summaries_not_shown'], levels='1 and 2') in take4,
       'Summaries of levels 1 and 2 are not shown in this conversation; `summaries()` returns them.' in take4),
      (4, True, True))
check('3 receipts: every call counted; the first call in each conversation logged (level, n_summaries)',
      (agent._v6_stats.get('summaries_calls'), [(p['level'], p['n_summaries'], p['conversation'], p['calls'])
                                               for p in events(log, 'v67-sumlookup')]), (3, [(4, 2, 1, 1), (4, 3, 2, 2)]))
check('3 the event\'s fields', sorted(events(log, 'v67-sumlookup')[0]), ['calls', 'conversation', 'level', 'n_summaries'])
conv3 = [(R_, 40), (U_, 40), (rule(1, R_), 40), (U_, 40), (rule(2, R_), 40), (R_, 600), (NOTES.format(k=3), 60)]
game, (sess, sess2), agent, log, go = h([conv3, [(R_, 50)]], budget=500)
for _ in range(8):
    go()
take3 = user_text(sess2.sent[0])
check('2 level 3\'s restart opener: "The summary of level 1 is not shown ...; `summaries()` returns it."',
      (game.level, 'The summary of level 1 is not shown in this conversation; `summaries()` returns it.' in take3,
       agent._v6_stats.get('summaries_calls'), events(log, 'v67-sumlookup')), (3, True, None, []))

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
import v62.sandbox_ext as SBX  # noqa: E402
exec(sys.argv[2])
STATES = []
_orig_run = SBX._original_run


def _spy(**kw):
    v6 = (kw.get('initial_state') or {}).get('v6') or {}
    STATES.append(hashlib.sha1(json.dumps(v6, sort_keys=True, default=str).encode()).hexdigest())
    return _orig_run(**kw)


SBX._original_run = _spy


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
    out = dict(sent=[norm(s.sent) for s in sessions], engine=game.engine, events=events,
               stats=json.loads(json.dumps(agent._v6_stats, default=str)),
               records=json.loads(json.dumps({k: game.__dict__.get(k) for k in keys})),
               library=sorted(agent._kept_functions), archive=sorted(agent._v61_archive), hidden=agent._v61_hidden,
               rules=json.loads(json.dumps(agent._v6_rules)), states=list(STATES))
    STATES.clear()
    return out


def run(scripts, turns, levels=5, **kw):
    game, sessions, agent, log, go = harness(scripts, **kw)
    game.game = _TG(levels)
    for _ in range(turns):
        go()
    return collect(game, sessions, agent, log)


R_, U_ = 'action(["RIGHT"])', 'action(["UP"])'


def rule(k, extra=''):
    return 'save_summary("Rule L%d:\\n- UP completes the level\\n- RIGHT moves the marker on level %d")\n%s' % (k, k, extra)


NOTES = 'save_summary("Notes L{k}:\\n- the marker moves right one cell per RIGHT\\n- UP completes the level")'
LONG = ('save_summary("Rule L2:\\n" + "\\n".join("- item %d: the blocks slide until they meet a wall or another block" % i '
        'for i in range(900)))')
runs = {}
# three summaries, a budget restart on level 4 (the plural not-shown line), a probe of the sandbox's globals
runs['level4'] = run([[(R_, 40), (U_, 40), (rule(1, R_), 40), (U_, 40), (rule(2, R_), 40), (U_, 40), (rule(3, R_), 40), (R_, 600),
                       (NOTES.format(k=4), 60)],
                      [('try:\n    summaries\n    print("S", True)\nexcept NameError:\n    print("S", False)\n' + R_, 50), (R_, 50)]],
                     11, budget=500)
# two summaries, a budget restart on level 3 (the singular line)
runs['level3'] = run([[(R_, 40), (U_, 40), (rule(1, R_), 40), (U_, 40), (rule(2, R_), 40), (R_, 600), (NOTES.format(k=3), 60)],
                      [(R_, 50)]], 8, budget=500)
# a truncated summary; a save in a non-acting call; an unrequested save
runs['long'] = run([[(R_, 40), (U_, 40), (rule(1), 40), (R_, 40), (U_, 40), (LONG, 40), (R_, 40), (rule(5, R_), 40)]], 6)
raw = json.load(open(os.path.join(os.path.dirname(A.__file__), 'texts.json')))['texts']
texts = {}
for name, env in (('production', {}), ('control_sco0_seeds0', {'ARC3_V66_SCO': '0', 'ARC3_V66_SEEDS': '0'}),
                  ('no_reset', {'EXPOSE_RESET': 'off'}), ('clock', {'ARC3_V65_CLOCK': '1'}),
                  ('v65_texts_off', {'ARC3_V65_TEXTS': '0'}), ('soft', {'ARC3_V67_RESTART_KIND': 'soft'})):
    keep = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        texts[name] = V.apply(json.loads(json.dumps(raw)))
    except RuntimeError as exc:                 # an environment the variant does not support (e.g. TEXTS67 without the
        texts[name] = 'ERROR: %s' % exc         # v6.5 texts) fails the same way in both trees
    for k, v in keep.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
print('IDENT-JSON ' + json.dumps(dict(runs=runs, texts=texts, agent_texts=A.TEXTS, seeds=SBX.SEEDS), sort_keys=True,
                                  default=str))
'''
CHANGED = ('v62/agent.py', 'v62/variants.py', 'v62/run_v6.py', 'v62/sandbox_ext.py')
missing = [f for f in CHANGED if not (ROOT / (f + '.pre-v67x')).exists()]
check('4 the deployed v6.7 base copies (v62/*.pre-v67x, written by apply_v67x.py) are present', missing, [])
if not missing:
    base = Path(tempfile.mkdtemp()) / 'v67_base'
    shutil.copytree(ROOT / 'v62', base / 'v62', ignore=shutil.ignore_patterns('__pycache__', '*.pre-*', '*.tmp'))
    for f in CHANGED:
        shutil.copyfile(ROOT / (f + '.pre-v67x'), base / f)
    env_clean = {k: v for k, v in os.environ.items() if not k.startswith('ARC3_V67_')}
    CORE = dict(ARC3_V67_LOOP='1', ARC3_V67_TEXTS='1', ARC3_V67_NOTES='1', ARC3_V67_SCORE_TEXT='1')
    VARIANTS = {'off': {}, 'core': CORE}

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
    jobs = [(base, 'off'), (ROOT, 'off'), (base, 'core'), (ROOT, 'core')]
    with ThreadPoolExecutor(len(jobs)) as pool:
        outs = dict(zip([(('base' if t == base else 'new'), v) for t, v in jobs], pool.map(run_tree, jobs)))
    check('4 the four identity runs completed', [k for k, v in outs.items() if v is None], [])
    CMP = ('sent', 'engine', 'events', 'stats', 'records', 'library', 'archive', 'hidden', 'rules', 'states')
    if all(v is not None for v in outs.values()):
        for variant in ('off', 'core'):
            a, b = outs[('base', variant)], outs[('new', variant)]
            check('4 %s: the texts in %d environments, the agent\'s TEXTS, the seeds identical' % (variant, len(a['texts'])),
                  ([k for k in a['texts'] if a['texts'][k] != b['texts'][k]], a['agent_texts'] == b['agent_texts'],
                   a['seeds'] == b['seeds']), ([], True, True))
            for name in a['runs']:
                check('4 %s %s: traffic, actions, events, stats, records, library, summaries, sandbox state identical'
                      % (variant, name), [k for k in CMP if a['runs'][name][k] != b['runs'][name][k]], [])
        rn = outs[('new', 'off')]['runs']
        check('4 the runs took the intended paths (summaries saved, a truncated one, the restarts, the not-shown lines)',
              (sorted(rn['level4']['rules']), sorted(rn['level3']['rules']), rn['long']['rules'].get('2', '').endswith('- ...'),
               rn['level4']['stats'].get('restarts'), rn['level3']['stats'].get('restarts'),
               'Summaries of levels 1 and 2 are not shown in this conversation.' in json.dumps(rn['level4']['sent'][1]),
               'The summary of level 1 is not shown in this conversation.' in json.dumps(rn['level3']['sent'][1]),
               'S False' in json.dumps(rn['level4']['sent'][1]), len(rn['level4']['states'])),
              (['1', '2', '3'], ['1', '2'], True, 1, 1, True, True, True, 11))
print('FAILS:', fails)
