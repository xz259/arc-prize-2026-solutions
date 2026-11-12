"""A timed-out or crashed sandbox call leaves the program library unchanged (S1 finding, 2026-10-03).
usage: PYTHONPATH=src:. python3 tests/test_library_timeout.py"""
import os
import sys

os.environ.update({'ARC3_PERSISTENT_FUNCTIONS': '1', 'ARC3_PERSISTENT_FUNCTIONS_SCOPE': 'game'})
from v6.agent import V6Agent  # noqa: E402


class Sess:
    rebased_away, blocks = set(), []

    def span_evicted(self, a, b):
        return 0


a = V6Agent(session=Sess(), logger=lambda k, p: None, count_tokens=lambda t: 1, model='M', base_url='http://x/v1',
            provider='vllm')
a._step_env_callback = None
src = 'def f(a):\n    """doc"""\n    return a\n'
p1 = {}
a._record_retained_functions({'keepable_functions': [{'name': 'f', 'source': src}], 'retention_rejected': []}, p1)
ok1 = 'f' in a._kept_functions and a._v6_library_order == ['f']
p2 = {}
a._record_retained_functions({'error': 'Tool timed out after 10s', 'stdout': '', 'action_results': []}, p2)
ok2 = 'f' in a._kept_functions and a._v6_library_order == ['f'] and 'programs' not in p2
p3 = {}
a._record_retained_functions({'keepable_functions': [], 'retention_rejected': []}, p3)   # a finished call that kept none
ok3 = 'f' not in a._kept_functions and '`f` was removed from the library' in p3.get('programs', '')   # spec wording (10-03)
print('saved after a normal call', 'PASS' if ok1 else 'FAIL')
print('unchanged after a timeout (no intake, no note)', 'PASS' if ok2 else 'FAIL', p2)
print('a finished call still updates the library', 'PASS' if ok3 else 'FAIL')
print('TEST_LIBRARY_TIMEOUT', 'PASS' if ok1 and ok2 and ok3 else 'FAIL')
sys.exit(0 if ok1 and ok2 and ok3 else 1)
