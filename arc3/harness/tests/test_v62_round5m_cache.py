"""Round 5m item 122 (user 10-08: "Let's try the cache"): the per-program analysis carried across python calls, through the
real wrapped sandbox path (tool_agent.run_sandboxed_python with sandbox_ext and retention installed). A first call fills
the host store; a second call with the same library is seeded; the results (kept programs, rejections, output) equal an
unseeded call; the 'v62' analysis never reaches the host's library bookkeeping; and a trivial call with 200 programs of
~60 lines is faster seeded than unseeded. Synthetic programs plus edge cases; counts only.
usage (pod copy): PYTHONPATH=src:. python3 tests/test_v62_round5m_cache.py"""
import json
import os
import statistics
import time

os.environ.update({'MULTIMODAL_CONTEXT': 'current_grid', 'MULTIMODAL_UPSCALE': '8', 'ARC3_PERSISTENT_FUNCTIONS': '1',
                   'ARC3_PERSISTENT_FUNCTIONS_SCOPE': 'game', 'ARC3_PERSISTENT_FUNCTIONS_IMPORTS': '1',
                   'ARC3_ANIMATION': '1', 'LOCAL_ANALYZER_TOOL_TIMEOUT': '30'})
import v62.patches  # noqa: E402,F401
import v62.retention as r  # noqa: E402
r.install()
from inference.agent import tool_agent as ta  # noqa: E402

_bench = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bench_library_restore.py')).read()
_tpl = {}
exec(_bench[_bench.index("SMALL = '''"):_bench.index('def call(')], _tpl)
BIG = _tpl['BIG']
fails = []


def check(name, got, want):
    ok = got == want
    print('%-70s -> %-30r %s' % (name, got if not isinstance(got, str) else got[:30], 'OK' if ok else 'WRONG (want %r)' % (want,)))
    if not ok:
        fails.append(name)


def call(kept, code='print(1)'):
    out = ta.run_sandboxed_python(code=code, timeout_seconds=30, initial_state={}, action_handler=lambda *a, **k: {},
                                  kept_functions=list(kept), retain_imports=True, repair_hints=True)
    return out


def view(out):
    return json.dumps([sorted((i['name'], i['source']) for i in out.get('keepable_functions') or []),
                       out.get('retention_rejected'), out.get('stdout'), out.get('error')], sort_keys=True, default=str)


EDGE = ['def a(x):\n    return x + 1\n', 'import math\ndef b(x):\n    return math.sqrt(x)\n',
        'def c(x):\n    return a(x) * 2\n', 'LIMIT = 7\n', 'def k(x):\n    return LIMIT + x\n',
        'def h(x):\n    return missing_name(x)\n']
SNIPPET = 'def d(x):\n    return c(x) + k(x)\nprint(d(3))\n'
lib = EDGE + [BIG.replace('{i}', str(i)) for i in range(40)]

r.ANALYSIS.clear()
first = call(lib, SNIPPET)
check('first call: analysis harvested for the kept programs', r.STATS['harvested'] > 40, True)
check('first call: no v62 key reaches the host bookkeeping',
      any('v62' in i for i in first.get('keepable_functions') or []), False)
kept_sources = [i['source'] for i in first.get('keepable_functions') or []]
seeded_before = r.STATS['seeded']
second = call(kept_sources, SNIPPET)
check('second call: every kept program with a record was seeded', r.STATS['seeded'] - seeded_before >= 40, True)
saved = dict(r.ANALYSIS)
r.ANALYSIS.clear()
unseeded = call(kept_sources, SNIPPET)
r.ANALYSIS.update(saved)
check('seeded and unseeded calls: identical kept programs, rejections and output', view(second), view(unseeded))
check('the snippet ran with library programs (d(3) = 18)', '18' in str(second.get('stdout')), True)
bad = dict(saved)
for h in list(bad)[:5]:
    bad[h] = {'n': 1, 'r': {'None': 'garbage'}}              # malformed entries are ignored, not trusted
r.ANALYSIS.clear(); r.ANALYSIS.update(bad)
check('malformed seed entries are ignored', view(call(kept_sources, SNIPPET)), view(unseeded))
r.ANALYSIS.clear(); r.ANALYSIS.update(saved)

big = [BIG.replace('{i}', str(i)) for i in range(200)]
r.ANALYSIS.clear()
call(big)                                                    # fills the store
t_seed, t_cold = [], []
for _ in range(3):
    t = time.monotonic(); call(big); t_seed.append(time.monotonic() - t)
    keep = dict(r.ANALYSIS); r.ANALYSIS.clear()
    t = time.monotonic(); call(big); t_cold.append(time.monotonic() - t)
    r.ANALYSIS.clear(); r.ANALYSIS.update(keep)
print('200 programs of ~60 lines, trivial call: seeded %.3f s, unseeded %.3f s (median of 3)' % (
    statistics.median(t_seed), statistics.median(t_cold)))
check('seeded is faster', statistics.median(t_seed) < statistics.median(t_cold), True)
print('FAILS:', fails)
