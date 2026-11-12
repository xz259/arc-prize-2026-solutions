"""Spec texts wired over dfranzen's (2026-10-03): the Python timeout error and the program-library retention notes.
usage: PYTHONPATH=src:. python3 tests/test_texts_wiring.py"""
from v6 import agent

T = agent.TEXTS
assert agent._v6_error_text('Tool timed out after 10s') == T['timeout']
assert agent._v6_error_text('Python syntax error: x') == 'Python syntax error: x'
notes = '\n'.join([
    'Retained your function state(f) for later Python calls in this game.',
    'Retained your function plan(grid, k=(1, 2)) for later Python calls on this level.',
    'Your function helper was not retained: it reads snippet-local data. Pass snippet-specific data as arguments or redefine it if needed.',
    'Your function bad was not retained: decorated functions are not kept.',
    'Your previous function old is no longer retained; redefine it if needed.',
    'Your retained functions were cleared: the sandbox restarted.',
    'Your retained functions were cleared after a retention error; redefine any you need.',
    'Something else stays as it is.'])
got = agent._v6_retention_notes(notes).split('\n')
want = ['Saved `state(f)` to the program library.',
        'Saved `plan(grid, k=(1, 2))` to the program library.',
        '`helper` was not saved: it reads snippet-local data. Pass snippet-specific data as arguments or redefine it if needed.',
        '`bad` was not saved: decorated functions are not kept.',
        '`old` was removed from the library: it was no longer defined after this call; redefine it if needed.',
        'Your program library was cleared: the sandbox restarted.',
        'Your program library was cleared after a retention error; redefine any programs you need.',
        'Something else stays as it is.']
for g, w in zip(got, want):
    assert g == w, (g, w)
print('timeout:', T['timeout'])
print('\n'.join(got))
assert agent._v6_valid_display(['RESET', 'UP', 'DOWN']) == 'UP, DOWN'
assert agent._v6_valid_display(['UP']) == 'UP' and agent._v6_valid_display([]) == 'none'
print('valid actions shown:', agent._v6_valid_display(['RESET', 'UP', 'DOWN', 'MOUSE']))
print('TEST_TEXTS_WIRING PASS')
