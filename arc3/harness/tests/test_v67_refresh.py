"""test_v67_refresh.py (2026-10-18, Claude 099768dc): apply_v67.py's item REFRESH (ARC3_V67_REFRESH_LOGICAL=500000), through
agent.analyze with the real sandbox on test_v67_soft.py's fakes; IE's conditions (SPEC UPDATE 3 and the 07:4x amendments):
  - a level-up with the session's logical length >= the threshold: the completed level's summary turn first (actions
    refused), then a new conversation at the new level's first board: no RESET, the fact line, the summary shown, the
    programs kept, the level's budget fresh, receipts (refreshes, the logical length, the summary turn, tokens to the first
    action, the level started), no attempt, no swap-out;
  - the summary already saved: no extra turn; the summary turn without a summary: the missing line;
  - below the threshold: unchanged (the same traffic as with the switch off); only at level-ups;
  - it pre-empts the position restart at a level-up;
  - switch off: byte-identical (tests/test_v67_ident.py runs a level-up at 600K positions on both trees; the suite).
usage (dev copy of harness_v6): PYTHONPATH=src:. python3 tests/test_v67_refresh.py"""
import copy
import json
import os
import tempfile  # noqa: F401  (the fakes' harness)
from pathlib import Path
from types import SimpleNamespace  # noqa: F401

_src = open(Path(__file__).with_name('test_v67_soft.py')).read()
ENV = eval(_src.split('ENV = ', 1)[1].split('\n\n', 1)[0])
FAKES = _src.split("FAKES = r'''", 1)[1].split("'''", 1)[0]
os.environ.update(ENV)
os.environ['ARC3_V67_REFRESH_LOGICAL'] = '500000'
import v62.patches  # noqa: E402,F401
import v62.retention as r  # noqa: E402
r.install()
import v62.agent as A  # noqa: E402
exec(compile(FAKES, 'test_v67_soft.py (FAKES)', 'exec'), globals())
fails = []


def check(name, got, want):
    ok = got == want
    print('%-84s -> %-40r %s' % (name, got if not isinstance(got, str) else got[:40], 'OK' if ok else 'WRONG (want %r)' % (want,)))
    if not ok:
        fails.append(name)


def user_text(batch):
    m = [x for x in batch if x.get('role') == 'user'][-1]
    c = m['content']
    return c if isinstance(c, str) else '\n'.join(p.get('text', '') for p in c if p.get('type') == 'text')


def tool_text(batch):
    return '\n'.join(m['content'] if isinstance(m['content'], str) else
                     '\n'.join(p.get('text', '') for p in m['content'] if p.get('type') == 'text')
                     for m in batch if m.get('role') == 'tool')


def events(log, kind):
    return [p for k, p in log if k == kind]


class Gate:
    _waiting, _clock_deadline = [], None

    def __init__(self):
        self.calls = []

    def handover(self, priority, snapshot):
        self.calls.append(priority)


R_, UP = 'action(["RIGHT"])', 'action(["UP"])'
SUM1 = 'save_summary("Rule L1:\\n- UP completes the level\\n- RIGHT moves the marker one cell")'


def run(conv1, conv2, logical, turns_before=2, after=2, rules=None, refresh=True):
    A.REFRESH = refresh
    game, sessions, agent, log, go = harness([conv1, conv2], budget=200000)
    gate = Gate()
    A.ta._priority_gate = lambda: gate
    for _ in range(turns_before):
        go()
    sessions[0].last_receipt = {'logical_tokens': logical}
    if rules:
        agent._v6_rules.update(rules)
    for _ in range(after):
        go()
    A.ta._priority_gate = lambda: None
    A.REFRESH = True
    return game, sessions, agent, log, gate


check('switch and threshold', (A.REFRESH, A.REFRESH_LOGICAL, 'takeover_refresh' in A.TEXTS), (True, 500000, True))

# === 1. the summary turn, then the refresh =================================================================================
game, (sess, sess2), agent, log, gate = run([(PROG + R_, 40), (UP, 40), (SUM1, 60)], [(R_, 30), (R_, 30)], 600000, after=3)
op = user_text(sess.sent[2])
check('1 the level-up turn is the summary turn: the level-up text and the refresh request',
      (A.TEXTS['levelup'].replace('{level-1}', '1').replace('{level}', '2').split('\n')[0][:60] in op,
       A._fill(A.TEXTS['refresh_summary_request'], logical='600,000', level=2, prev=1) in op), (True, True))
check('1 the summary saved in it', agent._v6_rules.get(1), '- UP completes the level\n- RIGHT moves the marker one cell')
first = user_text(sess2.sent[0])
line = A._fill(A.TEXTS['takeover_refresh'], level=2, done='level 1', logical='600,000')
check('1 the new conversation: the fact line, the summary shown, no restart lines',
      (line in first, "Your summary of level 1, saved when you completed it" in first, '- UP completes the level' in first,
       'RESET of level' in first, 'continues level' in first, 'In `history`' in first, 'The game is at level 2' in first),
      (True, True, True, False, False, False, True))
check('1 no RESET, the old session closed, no attempt, no swap-out', (game.engine, sess.closed, game.__dict__.get('v6_level_attempts'),
      game.__dict__.get('v62_level_attempt_ends'), gate.calls, agent._v6_stats.get('restarts'), agent._v6_stats.get('swaps')),
      (['RIGHT', 'UP', 'RIGHT', 'RIGHT'], True, None, None, [], None, None))
check('1 the library kept; the level budget fresh (the summary turn not counted)', (sorted(agent._kept_functions), agent._v61_archive,
      agent._v61_level_tokens), (['double'], {}, 60))
check('1 receipts', tuple(agent._v6_stats.get(k) for k in ('refreshes', 'refresh_logical', 'refresh_levels', 'refresh_summary_turns',
                                                          'refresh_tokens_to_action')), (1, [600000], [2], 1, [30]))
ev = events(log, 'v67-refresh')
check('1 the v67-refresh event', [(p['level'], p['logical'], p['conversation'], p['tokens_prev_level'], p['summary_turn'],
                                   p['summary_saved'], p['completed']) for p in ev], [(2, 600000, 1, 80, True, True, [1])])
check('1 the first-action event', [(p['level'], p['tokens']) for p in events(log, 'v67-refresh-first-action')], [(2, 30)])
check('1 the summary-turn event names the reason', [p.get('reason') for p in events(log, 'v62-summary-turn')], ['refresh'])

check('1 no retry when the first reply saves', (agent._v6_stats.get('refresh_summary_retries'),
      [p['summary_retry'] for p in ev], [p for k, p in log if k == 'v64-attempt-notes-retry']), (None, [None], []))

# === 2. the summary turn without a summary: refused, one retry (IE 08:12), still none, the refresh, the missing line =========
RETRY = A._fill(A.TEXTS['refresh_summary_retry'], reason='an action call was refused', level=1)
game, (sess, sess2), agent, log, gate = run([(R_, 40), (UP, 40), (R_, 50), (R_, 50)], [(R_, 30)], 600000, after=2)
check('2 the action in the summary turn is refused with the refresh refusal and reason',
      (A.TEXTS['refresh_turn_refusal'] in json.dumps([p for k, p in log if k == 'tool-output']),
       A.TEXTS['refresh_turn_reason'] in json.dumps([p for k, p in log if k == 'tool-output']), game.engine),
      (True, True, ['RIGHT', 'UP', 'RIGHT']))
check('2 exactly one retry, then the refresh anyway; the receipts and both events', (
      [m['content'] for b in sess.sent for m in b if m.get('role') == 'user' and isinstance(m.get('content'), str)], len(sess.sent),
      agent._v6_stats.get('refresh_summary_retries'), agent._v6_stats.get('refreshes'),
      [(p['summary_retry'], p['summary_saved']) for p in events(log, 'v67-refresh')],
      [(p['kind'], p['level'], p['outcome']) for p in events(log, 'v64-attempt-notes-retry')], agent._v6_stats.get('notes_retries')),
      ([RETRY], 4, 1, 1, [('an action call was refused', False)], [('refresh', 1, 'none')], None))
check('2 the new conversation says no summary was saved for level 1', A._fill(A.TEXTS['v65_summary_missing'], level=1) in
      user_text(sess2.sent[0]), True)

# === 2b. the summary saved on the retry ==========================================================================================
game, (sess, sess2), agent, log, gate = run([(R_, 40), (UP, 40), (R_, 50), (SUM1, 50)], [(R_, 30)], 600000, after=2)
check('2b saved on the retry: the summary shown, the receipts and the outcome', (
      agent._v6_rules.get(1), '- UP completes the level' in user_text(sess2.sent[0]), agent._v6_stats.get('refresh_summary_retries'),
      [(p['summary_retry'], p['summary_saved']) for p in events(log, 'v67-refresh')],
      [(p['kind'], p['outcome']) for p in events(log, 'v64-attempt-notes-retry')], game.engine.count('RESET')),
      ('- UP completes the level\n- RIGHT moves the marker one cell', True, 1, [('an action call was refused', True)],
       [('refresh', 'saved')], 0))

# === 3. the summary saved before the check: no extra turn ======================================================================
game, (sess, sess2), agent, log, gate = run([(R_, 40), (UP, 40)], [(R_, 30)], 600000, after=1, rules={1: '- saved earlier'})
check('3 refresh at once: no summary turn, the old conversation got no level-up turn',
      (len(sess.sent), agent._v6_stats.get('refresh_summary_turns'), [p['summary_turn'] for p in events(log, 'v67-refresh')],
       '- saved earlier' in user_text(sess2.sent[0])), (2, None, [False], True))

# === 4. below the threshold: unchanged; the same traffic as with the switch off; only at level-ups =========================
runs = {}
for name, refresh in (('on', True), ('off', False)):
    runs[name] = run([(R_, 40), (UP, 40), (SUM1 + '\n' + R_, 40), (R_, 40)], [(R_, 30)], 400000, after=2, refresh=refresh)
check('4 below the threshold: no refresh; traffic identical to the switch off',
      (runs['on'][2]._v6_stats.get('refreshes'), runs['on'][1][0].sent == runs['off'][1][0].sent, runs['on'][1][1].sent),
      (None, True, []))
game, (sess, sess2), agent, log, gate = run([(R_, 40), (R_, 40), (R_, 40)], [(R_, 30)], 900000, turns_before=1, after=2)
check('4 no level-up: no refresh even at 900K positions', (agent._v6_stats.get('refreshes'), len(sess2.sent)), (None, 0))

# === 5. it pre-empts the position restart at a level-up ==========================================================================
game, (sess, sess2), agent, log, gate = run([(R_, 40), (UP, 40), (SUM1, 60)], [(R_, 30)], A.POSITION_RESTART, after=2)
check('5 at the position limit: the summary turn and the refresh, no position restart, no RESET',
      (agent._v6_stats.get('refreshes'), agent._v6_stats.get('position_restarts'), agent._v6_stats.get('position_notes_turns'),
       game.engine.count('RESET'), [p['logical'] for p in events(log, 'v67-refresh')]), (1, None, None, 0, [A.POSITION_RESTART]))

# === 6. run_v6 ===================================================================================================================
RUN = open(Path(A.__file__).with_name('run_v6.py')).read()
check('6 run_v6: the explicit default and the recorded switch', ("os.environ.setdefault('ARC3_V67_REFRESH_LOGICAL', '0')" in RUN,
      "'ARC3_V67_REFRESH_LOGICAL'," in RUN), (True, True))
print('FAILS:', fails)
