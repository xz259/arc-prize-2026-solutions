"""Profile (2026-10-08) of one call's library work in the sandbox's code (retained_functions with v62/retention's EDITS):
restore_functions and collect_functions on N kept programs of ~60 lines, in-process with cProfile.
usage (pod copy): PYTHONPATH=src:. python3 tests/profile_library_restore.py [N]"""
import cProfile, io, os, pstats, sys, time
sys.argv += [] if len(sys.argv) > 1 else ['200']
N = int(sys.argv[1])
os.environ.update({'ARC3_PERSISTENT_FUNCTIONS': '1', 'ARC3_PERSISTENT_FUNCTIONS_SCOPE': 'game', 'ARC3_PERSISTENT_FUNCTIONS_IMPORTS': '1'})
import v62.patches  # noqa: E402,F401
import v62.retention as r  # noqa: E402
from inference.agent import python_tool_sandbox as sb  # noqa: E402
original = r._source_text()
r.install()
start = sb._SANDBOX_BOOTSTRAP.find('def _import_statements')
patched = None
for old, new in r.EDITS:
    original = original.replace(old, new, 1)
patched = original
if os.environ.get('MEMO', '1') == '1' and hasattr(r, 'MEMO'):
    patched = patched.rstrip('\n') + '\n' + r.MEMO      # round 5m (item 121)
ns = {'__name__': 'retained_functions_profile'}
exec(compile(patched, '<retained_functions v61>', 'exec'), ns)
_bench = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bench_library_restore.py')).read()
_tpl = {}
exec(_bench[_bench.index("SMALL = '''"):_bench.index('def call(')], _tpl)   # the program templates only
BIG = _tpl['BIG']
kept = [BIG.replace('{i}', str(i)) for i in range(N)]
g = {'__builtins__': __builtins__}
avail = set(g) | set(dir(__builtins__)) | {'np', 'history', 'current_frame', 'action'}
pr = cProfile.Profile()
t = time.monotonic(); pr.enable()
pool, rej = ns['restore_functions'](kept, g, avail, retain_imports=True, repair_hints=True)
t1 = time.monotonic()
kept2, rej2 = ns['collect_functions']('print(1)', pool, g, avail, retain_imports=True, repair_hints=True)
pr.disable(); t2 = time.monotonic()
print('N %d: restore %.3f s, collect %.3f s, kept %d, rejected %d' % (N, t1 - t, t2 - t1, len(kept2), len(rej) + len(rej2)))
s = io.StringIO(); pstats.Stats(pr, stream=s).sort_stats('cumulative').print_stats(18); print(s.getvalue()[:4000])
