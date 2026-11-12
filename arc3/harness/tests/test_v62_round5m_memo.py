"""Equivalence test of round 5m item 121 (the per-call memo of the library analysis): restore_functions and
collect_functions of the sandbox's retained_functions (with v62/retention's EDITS) give identical results with and without
the memo, on random pools of real program sources (top-level defs and UPPER_CASE constants taken from a run's model code)
and on synthetic edge cases (trailing whitespace variants, decorators, annotations, imports, syntax errors, global stores,
missing dependencies). Prints counts only, never the sources.
usage (pod copy): PYTHONPATH=src:. python3 tests/test_v62_round5m_memo.py [raw-harness glob] [pools]"""
import ast
import copy
import glob
import json
import os
import random
import sys

os.environ.update({'ARC3_PERSISTENT_FUNCTIONS': '1', 'ARC3_PERSISTENT_FUNCTIONS_SCOPE': 'game',
                   'ARC3_PERSISTENT_FUNCTIONS_IMPORTS': '1'})
import v62.retention as r  # noqa: E402

GLOB = sys.argv[1] if len(sys.argv) > 1 else ''
POOLS = int(sys.argv[2]) if len(sys.argv) > 2 else 60
NEW = [e for e in r.EDITS if '_v62_parse' in e[1]]
OLD_EDITS = [e for e in r.EDITS if e not in NEW]


def build(memo):
    text = r._source_text()
    for old, new in (r.EDITS if memo else OLD_EDITS):
        assert text.count(old) == 1, old[:60]
        text = text.replace(old, new, 1)
    if memo:
        text = text.rstrip('\n') + '\n' + r.MEMO
    ns = {'__name__': 'rf_%s' % memo}
    exec(compile(text, '<retained_functions v61>', 'exec'), ns)
    return ns


def corpus():
    out = []
    for f in sorted(glob.glob(GLOB))[:400] if GLOB else []:
        for line in open(f, errors='replace'):
            if '"model-turn"' not in line[:120]:
                continue
            for tc in (json.loads(line)['payload'].get('message') or {}).get('tool_calls') or []:
                try:
                    code = json.loads(tc['function']['arguments']).get('code') or ''
                    tree = ast.parse(code)
                except Exception:
                    continue
                for node in tree.body:
                    if isinstance(node, (ast.FunctionDef, ast.Assign)):
                        seg = ast.get_source_segment(code, node)
                        if seg:
                            out.append(seg)
    return out


SYN = ['def a(x):\n    return x + 1\n', 'def a(x):\n    return x + 1', 'def a(x):\n    return x + 1\n\n\n',
       'import math\ndef b(x):\n    return math.sqrt(x)\n', 'from math import sqrt as s\ndef c(x):\n    return s(x)\n',
       '@staticmethod\ndef d(x):\n    return x\n', 'def e(x: int):\n    return x\n', 'def f(x=[]):\n    return x\n',
       'def g(x):\n    global G\n    G = x\n', 'def h(x):\n    return missing(x)\n', 'def i(x):\n    return a(x)\n',
       'def j(x:\n    return x\n', 'LIMIT = 7\n', 'LIMIT = [1, 2, 3]\n', 'lower = 3\n', 'def k(x):\n    return LIMIT + x\n',
       'import os, sys\ndef m():\n    return os.sep + sys.platform\n', '  def n(x):\n      return x\n', 'x = 1\ndef p():\n    return 1\n']


def run(ns, sources, snippet, retain_imports, export=None):
    g = {'__builtins__': __builtins__}
    avail = set(g) | set(dir(__builtins__)) | {'np', 'history', 'current_frame', 'action'}
    pool, rej = ns['restore_functions'](list(sources), g, avail, retain_imports=retain_imports, repair_hints=True)
    try:
        exec(compile(snippet, '<python_tool>', 'exec'), g, g)
    except Exception:
        pass
    kept, rej2 = ns['collect_functions'](snippet, pool, g, avail, retain_imports=retain_imports, repair_hints=True)
    if export is not None:                                   # item 122: what this call would hand to the host
        for src in kept.values():
            export.update((ns['_v62_export_item'](src) or {}).get('v62') or {})
    return json.dumps([sorted(pool.items()), rej, sorted(kept.items()), rej2], sort_keys=True, default=str)


A, B = build(False), build(True)
real = corpus()
rng = random.Random(0)
same = differ = 0
cases = [(SYN, 'def q(x):\n    return a(x)\n')]
for _ in range(POOLS if real else 0):
    pool = rng.sample(real, min(len(real), rng.randint(5, 150)))
    snippet = '\n'.join(rng.sample(real, min(len(real), rng.randint(0, 4))))
    cases.append((pool, snippet))
seeded_same = seeded_differ = seeded_hits = 0
for pool, snippet in cases:
    for retain in (True, False):
        export = {}
        a, b = run(A, copy.deepcopy(pool), snippet, retain), run(B, copy.deepcopy(pool), snippet, retain, export)
        same += a == b
        differ += a != b
        C = build(True)                                      # item 122: a fresh call seeded with the exported analysis
        C['_v62_seed'](export)
        c = run(C, copy.deepcopy(pool), snippet, retain)
        seeded_hits += sum(1 for k in C['_v62_cache'] if k[0] == 'record')
        seeded_same += a == c
        seeded_differ += a != c
print('real sources %d; cases %d x 2 retain modes: identical %d, different %d' % (len(real), len(cases), same, differ))
print('seeded (item 122): identical %d, different %d; records in seeded calls %d' % (seeded_same, seeded_differ, seeded_hits))
print('test_v62_round5m_memo', 'PASS' if differ == 0 and same > 0 and seeded_differ == 0 and seeded_same > 0 else 'FAIL')
