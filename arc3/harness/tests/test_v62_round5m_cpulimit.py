"""Round 5m item 123 (user 10-08 ~19:30, "Sure. Let's do it."): the 10-s compute limit through the real wrapped sandbox
path (v62.patches, v62.retention, v62.sandbox_ext via v62.agent). Run 396 had 22 calls killed at the hard CPU limit: the
model saw "Sandbox process exited unexpectedly." and lost the call's printed output, because the bookkeeping after the
limit's error (re-checking the program library) could need more than the 1-s margin to the hard limit.
Checks: a call that runs past the limit gets the limit's error with its printed output, with no library and with 600
programs of ~60 lines; the error is raised once (a snippet that catches it finishes); a process still killed at the hard
limit (the error swallowed and the loop resumed, or a long C-level call) is reported as the compute limit
(TEXTS['compute_limit_lost']); another death keeps "exited unexpectedly"; normal calls are unchanged; the hard limit is
CPU_MARGIN s after the stated one. Outcomes and times only.
usage (pod copy): PYTHONPATH=src:. python3 tests/test_v62_round5m_cpulimit.py"""
import os
import time

for k, v in dict(ARC3_V61_TEXTS='1', ARC3_V6_A13='1', ARC3_V61_SUMMARY='1', ARC3_NEW_CHANGED_PROMPTS='1',
                 ARC3_ACTION_INFO='1', ARC3_REPEAT_STATE_GUARD='0', ARC3_V61_RESTART_TOKENS='150000', ARC3_V6_A3='1',
                 ARC3_GUARDS_FROM_LEVEL='2', ARC3_BATCH_NOOP_BLOCK='1', ARC3_STALE_STATE_BLOCK='1', ARC3_DEATH_LEDGER='1',
                 EXPOSE_RESET='on', EXPOSE_UNDO='on', ONLY_RESET_LEVELS='true', ARC3_ANIMATION='1',
                 MULTIMODAL_CONTEXT='current_grid', MULTIMODAL_UPSCALE='8', ARC3_PERSISTENT_FUNCTIONS='1',
                 ARC3_PERSISTENT_FUNCTIONS_SCOPE='game', ARC3_PERSISTENT_FUNCTIONS_IMPORTS='1',
                 LOCAL_ANALYZER_TOOL_TIMEOUT='10').items():                 # as test_v62_round5l.py, plus the library
    os.environ.setdefault(k, v)
import arcengine  # noqa: E402,F401
import v62.patches as patches  # noqa: E402
import v62.retention as r  # noqa: E402
r.install()
import v62.agent as A  # noqa: E402  (imports v62.sandbox_ext: the CPU-seconds wrapper and the host's exit reporting)
from v62 import sandbox_ext as se  # noqa: E402
from inference.agent import python_tool_sandbox as sb  # noqa: E402
from inference.agent import tool_agent as ta  # noqa: E402

_bench = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bench_library_restore.py')).read()
_tpl = {}
exec(_bench[_bench.index("SMALL = '''"):_bench.index('def call(')], _tpl)
BIG = _tpl['BIG']
fails = []


def check(name, got, want):
    ok = got == want
    print('%-78s -> %-24r %s' % (name, got if not isinstance(got, str) else got[:24], 'OK' if ok else 'WRONG (want %r)' % (want,)))
    if not ok:
        fails.append(name)


def call(code, kept=()):
    t = time.monotonic()
    out = ta.run_sandboxed_python(code=code, timeout_seconds=10, initial_state={}, action_handler=lambda *a, **k: {},
                                  kept_functions=list(kept), retain_imports=True, repair_hints=True)
    return out, time.monotonic() - t


def kind(out):
    err = str(out.get('error') or '')
    if 'ComputeLimitError' in err or 'seconds of computation and was stopped here' in err:
        return 'limit error'
    if err == se.CPU_KILLED:
        return 'killed at the limit'
    if 'exited unexpectedly' in err:
        return 'exited unexpectedly'
    return 'ok' if not err else 'other'


LOOP = 'print("marker-1")\nx = 0\nwhile True:\n    x += 1\n'
check('the bootstrap sets the hard CPU limit CPU_MARGIN s after the stated one',
      'cpu_limit - 1 + %d' % patches.CPU_MARGIN in sb._SANDBOX_BOOTSTRAP and patches.CPU_MARGIN == 5, True)

out, w = call(LOOP)
check('loop past the limit, no library: the limit error', kind(out), 'limit error')
check('  ... with the printed output', 'marker-1' in str(out.get('stdout')), True)
check('  ... at about the stated limit (10-12 s wall)', 10 <= w < 12, True)
print('    wall %.1f s' % w)

lib = [BIG.replace('{i}', str(i)) for i in range(600)]
r.ANALYSIS.clear()
out, w = call(LOOP, lib)
check('loop past the limit, 600 programs, cold cache: the limit error', kind(out), 'limit error')
check('  ... with the printed output', 'marker-1' in str(out.get('stdout')), True)
check('  ... and the library returned (keepable programs)', len(out.get('keepable_functions') or []) >= 600, True)
print('    wall %.1f s' % w)

# the sandbox's builtins have no BaseException / SystemExit, so a bare `except:` is the only way a snippet catches the limit
check('the sandbox has no BaseException name (a bare except is the only catch)',
      "name 'BaseException' is not defined" in str(call('try:\n    raise ValueError()\nexcept BaseException:\n    pass\n')[0]
                                                   .get('error')), True)
ONCE = ('print("marker-1")\ntry:\n    x = 0\n    while True:\n        x += 1\nexcept:\n    pass\n'
        'y = 0\nfor i in range(3000000):\n    y += i\nprint("marker-2")\n')
out, w = call(ONCE)
check('the error caught by the snippet: raised once, the snippet finishes', kind(out), 'ok')
check('  ... with both outputs', 'marker-1' in str(out.get('stdout')) and 'marker-2' in str(out.get('stdout')), True)
print('    wall %.1f s' % w)

SWALLOW = 'print("marker-1")\nwhile True:\n    try:\n        x = 0\n        while True:\n            x += 1\n    except:\n        pass\n'
out, w = call(SWALLOW)
check('the error swallowed and the loop resumed: killed, reported as the limit', kind(out), 'killed at the limit')
check('  ... the agent shows the compute-limit text', A._v6_error_text(out.get('error')), A.TEXTS['compute_limit_lost'])
check('  ... at the hard limit (15-17 s wall)', 15 <= w < 17, True)
print('    wall %.1f s' % w)

CCALL = 'print("marker-1")\nsum(range(10 ** 11))\n'
out, w = call(CCALL)
check('a long C-level call past the hard limit: reported as the limit', kind(out), 'killed at the limit')
check('  ... at the hard limit (15-17 s wall)', 15 <= w < 17, True)
print('    wall %.1f s' % w)

# the host's reporting of a dead child, on real processes (the None branch: wait, then sanitize)
import signal  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402


def death(argv, kill=False, elapsed=0.0, stderr=''):
    p = subprocess.Popen(argv)
    if kill:
        time.sleep(0.3)
        p.send_signal(signal.SIGKILL)
    time.sleep(0.5)
    se._EXIT.started, se._EXIT.cpu_seconds = time.monotonic() - elapsed, 10
    sb._wait_for_process_exit(p)
    return sb._sanitize_host_error_text(stderr)


UNEXPECTED = 'Sandbox process exited unexpectedly.'
check('a child exiting with status 3 keeps "exited unexpectedly"', death([sys.executable, '-c', 'raise SystemExit(3)'],
                                                                        elapsed=30), UNEXPECTED)
check('a SIGKILL before the stated limit keeps "exited unexpectedly"',
      death([sys.executable, '-c', 'import time; time.sleep(30)'], kill=True, elapsed=2), UNEXPECTED)
check('a SIGKILL after the stated limit is the compute limit',
      death([sys.executable, '-c', 'import time; time.sleep(30)'], kill=True, elapsed=16), se.CPU_KILLED)
check('stderr with the limit error is the compute limit',
      death([sys.executable, '-c', 'raise SystemExit(1)'], elapsed=1, stderr='Traceback ...\nComputeLimitError: ...'),
      se.CPU_KILLED)
check('the agent leaves "exited unexpectedly" unchanged', A._v6_error_text(UNEXPECTED), UNEXPECTED)

out, w = call('print(sum(range(10)))\n')
check('a normal call is unchanged', (kind(out), str(out.get('stdout')).strip()), ('ok', '45'))

check('wall-clock backstop text unchanged', A._v6_error_text('Tool timed out after 120s'), A.TEXTS['timeout_wall'])
print('compute_limit_lost text: %d chars' % len(A.TEXTS['compute_limit_lost']))
print('FAILS:', fails)
print('test_v62_round5m_cpulimit', 'PASS' if not fails else 'FAIL')
