"""v6.5 fixes (apply_v65.py), on the test scaffolding of test_v64_fixes.py (its prelude up to run_notes_turn is executed here):
A2r, RF, CLK, C3, C1, OPN (+ the C5 client field), TRN, RUN. C6 is in test_v65_c6.py (session_ext level).
usage (dev copy of harness_v6): PYTHONPATH=src:. python3 tests/test_v65_fixes.py"""
import json
from pathlib import Path

_src = open(Path(__file__).with_name('test_v64_fixes.py')).read()
exec(compile(_src.split('\ndef run_notes_turn')[0], 'test_v64_fixes.py (prelude)', 'exec'), globals())

# === A2r (IE 10-17): a late tag naming a completed level without a summary is stored as that level =====================
body = '\n- the green bar shrinks by one cell per move\n- the exit opens when it is empty'


def save_case(existing, text):
    ag, _, _ = new_agent([('action(["RIGHT"])', 10)])
    ag._v62_attempt_request = None
    ag._v6_stats['completions'] = 10
    ag._v62_fn_window, ag._v62_fn_windows = (2, 99), {2: 99}          # L2's window open; L1's closed
    ag._v6_rules = dict(existing)
    ag._v62_rule_fn = set(existing)
    log = []
    ag._v6_log = lambda k, p: log.append((k, p))
    return ag, ag._v62_save_summary(text), log


ag, res, log = save_case({}, 'Rule L1:' + body)
check('A2r late L1 tag, no L1 summary: stored as L1, L2 untouched', (res.get('saved'), ag._v6_rules.get(1), ag._v6_rules.get(2)),
      (True, body.strip('\n'), None))
check('A2r receipt and log (level, window, since)', (ag._v6_stats.get('summary_late_tagged'),
                                                    [(p['level'], p['window'], p['since']) for k, p in log
                                                     if k == 'v65-summary-late-tagged']), (1, [(1, 2, 9)]))
check('A2r the note names level 1', res.get('note', '').startswith('Saved your summary of level 1 ('), True)
res2 = ag._v62_save_summary(body.replace('green', 'blue'))
check('A2r the L2 window stays open (an untagged save stores L2)', (res2.get('saved'), ag._v6_rules.get(2)),
      (True, body.replace('green', 'blue').strip('\n')))
ag, res, log = save_case({1: '- old L1'}, 'Rule L1:' + body)
check('A2r late L1 tag, L1 summary exists: refused, L1 kept', (res.get('saved'), ag._v6_rules.get(1),
                                                               res.get('note') == A._fill(A.TEXTS['summary_tag_closed'], level=1)),
      (False, '- old L1', True))
check('A2r that refusal is counted, no late save', (ag._v6_stats.get('summary_tag_refused'), ag._v6_stats.get('summary_late_tagged')),
      (1, None))
ag, res, log = save_case({2: '- old L2'}, 'Rule L1:' + body)
check('A2r late L1 tag with an L2 summary: L1 stored, L2 kept', (res.get('saved'), ag._v6_rules.get(1), ag._v6_rules.get(2)),
      (True, body.strip('\n'), '- old L2'))
ag, res, log = save_case({}, 'Rule L3:' + body)
check('A2r an L3 tag in the L2 window: refused (not completed)', (res.get('saved'), ag._v6_rules.get(3)), (False, None))
ag, res, log = save_case({}, 'Rule L0:' + body)
check('A2r an L0 tag: refused', (res.get('saved'), 0 in ag._v6_rules), (False, False))

# === RF ===================================================================================================================
check('RF constants', (A.RESTART_P, A.STUCK_C, A.NEXT_C, A.DECAY_C, A.LEFT_FACTOR), (0.41, 0.27, 0.78, 0.94, 0.8))

# === RUN: run_v6 defaults ================================================================================================
import v62.run_v6 as RV  # noqa: E402
rv_src = open(RV.__file__).read()
check('RUN stale block off, LANE_TOKS 50', (RV.ENV['ARC3_STALE_STATE_BLOCK'], RV.ENV['ARC3_V62_LANE_TOKS']), ('0', '50'))
check('RUN check default 8000', "os.environ.setdefault('ARC3_ASSISTANT_TURN_CHECK', '8000')" in rv_src, True)

# === CLK, C3: texts ======================================================================================================
S = A.TEXTS['system']
check('CLK _OPENER_KEEP has no clock', A._OPENER_KEEP, ('Step ', 'Reminder:'))
check('CLK system: no run time', ('time left' in S, 'run time' in S), (False, False))
check('CLK system: closing-report bullet', '- No closing report is expected: keep playing until the harness ends the game.\n' in S,
      True)
check('C3 turn-based bullet', '- The game is turn-based: it advances only when one of your actions executes, and nothing in the '
      'game changes while you think or between your calls.' in S, True)
check('C3 removed-turns bullet', '; this never ends the conversation, and this system message and the latest board always remain.'
      in S and 'A harness restart of the level is normally announced one turn ahead, with a request for notes.' in S, True)
check('C6 batch bullet tail (IE)', 'a stop concerns two actions in a row on one board.' in S, True)

# === C1: signatures and the detector =====================================================================================
rep = ('Your previous call executed 2 action(s): UP, SPACE.\nPer action: 1: UP -> effect | 24 cells changed in the interior '
       '(rows 12-39, cols 26-51)\nStep 120, level 5 (attempt 7). Valid actions: UP, DOWN\nReminder: call action(...)')
rep2 = rep.replace('24 cells', '25 cells').replace('Step 120', 'Step 121').replace('attempt 7', 'attempt 8')
led = ('\nContinuations that ended in GAME OVER from a board identical to the current one (whole board compared; 3 game overs '
       'recorded on this level):\n- 4 action(s): UP, UP, LEFT, SPACE -> GAME OVER (game over #3 of this game)\nEach listed '
       'continuation ended in GAME OVER when played from a board identical to this one; state not shown on the board can differ.')
check('C1 norm: digits ignored', A._v65_report_sig(rep, 'norm') == A._v65_report_sig(rep2, 'norm'), True)
check('C1 norm: the death ledger ignored', A._v65_report_sig(rep + led, 'norm') == A._v65_report_sig(rep2, 'norm'), True)
check('C1 exact: digits matter', A._v65_report_sig(rep, 'exact') == A._v65_report_sig(rep2, 'exact'), False)
check('C1 norm: Step / Reminder lines dropped', 'Step' in A._v65_report_sig(rep, 'norm'), False)
x, y, z = ('a',), ('b',), ('c',)
check('C1 detector p=1 k=3', A._v65_loop([y, x, x, x]), (1, 3))
check('C1 detector p=3 k=4 (smallest period)', A._v65_loop([x, y, z] * 4), (3, 4))
check('C1 detector: two repeats are not a loop', A._v65_loop([z, x, x]), (0, 0))
check('C1 detector p=2', A._v65_loop([z, x, y, x, y, x, y]), (2, 3))
check('C1 detector: p=1 preferred over p=2', A._v65_loop([x] * 6), (1, 6))

state = Path(tempfile.mkdtemp()) / 'game_runtime_state.json'
game = FakeGame(state)
log = []
agent, sess, sess2 = new_agent([('action(["RIGHT"])', 100)] * 6, log=log)
for _ in range(6):
    agent.analyze(state, game.step, valid_actions=['UP', 'RIGHT', 'LEFT'], step_env=game.step_env, should_stop=lambda: False)
texts = [m['content'][0]['text'] for sent in sess.sent for m in sent if m.get('role') == 'user' and isinstance(m.get('content'), list)]
lines = [t for t in texts if 'Python calls executed the same actions' in t]
check('C1 e2e: the loop line appears once 3 identical turns repeat', len(lines) >= 1, True)
if lines:
    ln = [l for l in lines[0].split('\n') if 'Python calls' in l][0]
    check('C1 e2e: wording and position (before the Step line)', (ln.startswith('The last 3 Python calls executed the same actions, '
                                                                                'and their reports were identical apart from numbers'),
                                                                  lines[0].split('\n').index(ln) <
                                                                  next(i for i, l in enumerate(lines[0].split('\n')) if l.startswith('Step '))),
          (True, True))
check('C1 receipts', (agent._v6_stats.get('loop_lines', 0) >= 1, agent._v6_stats.get('loop_games'), agent._v6_stats.get('loop_sig')),
      (True, 1, 'norm'))
check('CLK e2e: no run-time line in any opener', any('Run time left' in t for t in texts), False)

# === OPN + C5 client: a restart through the agent loop ===================================================================


def restart_case(pin):
    A.RESTART_TOKENS = 500
    A.V65_PIN = pin
    st = Path(tempfile.mkdtemp()) / 'game_runtime_state.json'
    g = FakeGame(st)
    lg = []
    ag, s1, s2 = new_agent([('action(["UP"])', 100),
                            ('save_summary("Rule L1:\\n- UP clears the level when pressed once\\n- the board is static")\n'
                             'action(["RIGHT"])', 600),
                            ('save_summary("Notes L2:\\n- RIGHT moves the red block one cell\\n- the goal is the green pad")', 150)],
                           log=lg)
    for _ in range(4):
        ag.analyze(st, g.step, valid_actions=['UP', 'RIGHT', 'LEFT'], step_env=g.step_env, should_stop=lambda: False)
    return ag, s2, lg


ag, s2, lg = restart_case(False)
first = [m for m in s2.sent[0] if m.get('role') == 'user'][-1] if s2.sent else {}
parts = [p for p in (first.get('content') or []) if p.get('type') == 'text']
check('OPN restart opener: two text parts (section, rest)', len(parts) >= 2, True)
if len(parts) >= 2:
    sec, rest = parts[0]['text'], parts[1]['text']
    check('OPN section: newest notes first, then the L1 summary', (sec.startswith(A.TEXTS['attempt_notes_header'] + '\n'),
                                                                  '- RIGHT moves the red block one cell' in sec,
                                                                  'Your summary of level 1, saved when you completed it' in sec,
                                                                  sec.index('RIGHT moves') < sec.index('Your summary of level 1'),
                                                                  sec.endswith('\n')), (True, True, True, True, True))
    check('OPN rest: takeover text, no summaries list', (rest.startswith('The game is at level 2'),
                                                         A.TEXTS['restart_summaries'] in rest, 'not shown in this conversation' in rest),
          (True, False, False))
check('OPN arm B: no pin field', '_arc3_pin_parts' in first, False)
check('OPN receipts (section, no pin request)', (ag._v6_stats.get('opener_sections'), ag._v6_stats.get('pins_requested')), (1, None))
ag, s2, lg = restart_case(True)
first = [m for m in s2.sent[0] if m.get('role') == 'user'][-1] if s2.sent else {}
check('OPN arm A: the opener carries the pin request', (first.get('_arc3_pin_parts'), ag._v6_stats.get('pins_requested')), (1, 1))
A.V65_PIN = False

ag, _, _ = new_agent([('action(["RIGHT"])', 10)])
ag._v6_rules = {1: '- a', 2: '- bb', 4: '- dddd', 5: '- e five'}
ag._v62_attempt_notes = {}
ag._v62_notes_asked = None
check('OPN rest: the exact level list', ag._v65_opener_rest(6), ['Summaries of levels 1, 2 and 4 are not shown in this conversation.'])
check('OPN rest: one level, singular', ag._v65_opener_rest(3), ['The summary of level 1 is not shown in this conversation.'])
check('OPN rest: previous level missing -> A7 line', ag._v65_opener_rest(4)[-1], 'No summary was saved for level 3.')
A.V65_PIN_CAP = 5
lead = ag._v65_opener_lead(6)
check('OPN fit: nothing fits -> no section, items after it, counted', (ag._v65_section, lead.startswith('Your summary of level 5'),
                                                                        ag._v6_stats.get('pin_skipped_none_fit')), (None, True, 1))
A.V65_PIN_CAP = 2000

# === TRN: the /start body ================================================================================================
from v62 import transport as tp  # noqa: E402
sent = []


class FakeHTTP:
    def start(self, b, timeout):
        sent.append(('start', b))
        return self._resp(b['session_id'])

    def append(self, sid, b, timeout):
        sent.append(('append', b))
        return self._resp(sid)

    def close(self, sid, timeout=15):
        return {}

    def shutdown(self):
        pass

    def _resp(self, sid):
        obs = sent[-1][1]['managed_suffixes']['observations']
        return dict(choices=[dict(message=dict(role='assistant', content='x'), finish_reason='stop', token_ids=[5, 248046])],
                    usage=dict(prompt_tokens=10, completion_tokens=2),
                    persistent=dict(session_id=sid, revision=1, prefix_sha256='a' * 64,
                                    history_eviction=dict(policy='qsa_frequency', fraction=0.1),
                                    managed_suffixes=dict(registered_observation_ids=[o['observation_id'] for o in obs],
                                                          evicted_observation_ids=[])))


chats = {}
for pin in (True, False):
    sent.clear()
    he = dict(fraction=0.1, policy='qsa_frequency', protect_recent_tokens=5484, target_basis_tokens=30000, unit='split')
    s = tp.PersistentSession(base_url='http://x', model='m', tools=[{'t': 1}], template_kwargs={'reasoning_effort': 'xhigh'},
                             sampling=dict(temperature=1.0, top_p=0.95, top_k=20, max_tokens=12000),
                             history_eviction=he, im_end_ids=(248046,), decode=lambda ids: '')
    s.transport = FakeHTTP()
    user = dict(role='user', content=[dict(type='text', text='SEC\n'), dict(type='text', text='rest')], _arc3_observation=True)
    if pin:
        user['_arc3_pin_parts'] = 1
    s.generate([dict(role='system', content='S'), user])
    b0 = sent[0][1]
    check('TRN %s: arc3_summary_pin' % ('arm A' if pin else 'arm B'), b0.get('arc3_summary_pin'),
          dict(message_index=1, parts_end=1) if pin else None)
    check('TRN %s: no private key in chat messages' % ('arm A' if pin else 'arm B'), '_arc3_' in json.dumps(b0['chat']['messages']),
          False)
    chats[pin] = json.dumps(b0['chat'], sort_keys=True)
check('TRN arms A and B send byte-identical chat payloads (same prompt tokens; only arc3_summary_pin differs)',
      chats[True] == chats[False], True)
print('FAILS:', fails)
