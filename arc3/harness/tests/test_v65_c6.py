"""v6.5 C6 (apply_v65.py): the batch stop at the second consecutive NO-OP, BAND-ONLY / REVERTED / RESET breaking the streak,
the stale arm counted instead of refused, the stop detail, the receipts; through session_ext.step_env with the scripted session
of test_v62_round5l.py. usage (dev copy of harness_v6): PYTHONPATH=src:. python3 tests/test_v65_c6.py"""
import os
from types import SimpleNamespace
for k, v in dict(ARC3_V61_TEXTS='1', ARC3_V6_A13='1', ARC3_V61_SUMMARY='1', ARC3_NEW_CHANGED_PROMPTS='1',
                 ARC3_ACTION_INFO='1', ARC3_REPEAT_STATE_GUARD='0', ARC3_V61_RESTART_TOKENS='150000', ARC3_V6_A3='1',
                 ARC3_GUARDS_FROM_LEVEL='2', ARC3_BATCH_NOOP_BLOCK='1', ARC3_STALE_STATE_BLOCK='0', ARC3_DEATH_LEDGER='1',
                 EXPOSE_RESET='on', EXPOSE_UNDO='on', ONLY_RESET_LEVELS='true', ARC3_ANIMATION='1').items():
    os.environ.setdefault(k, v)
import arcengine  # noqa: E402
import v62.patches  # noqa: E402,F401
import v62.retention as r  # noqa: E402
r.install()
import v62.agent as A  # noqa: E402
import v62.session_ext as se  # noqa: E402
from inference.framework import solver as sv  # noqa: E402
fails = []


def check(name, got, want):
    ok = got == want
    print('%-70s -> %-50r %s' % (name, got if not isinstance(got, str) else got[:50], 'OK' if ok else 'WRONG (want %r)' % (want,)))
    if not ok:
        fails.append(name)


class Session:
    """step_env's collaborators with scripted payloads (as test_v62_round5l.py), plus the v6.5 receipts dict."""

    def __init__(self, payloads=()):
        self.game = SimpleNamespace(over=False, current_state=SimpleNamespace(
            available_actions=[0, 1, 2, 3, 4, 5, 6], levels_completed=1,
            raw=SimpleNamespace(state=arcengine.GameState.NOT_FINISHED)))
        self.history_entries, self.payloads, self.executed = [], list(payloads), []
        self.action_count = 7
        self.__dict__['v65_stats'] = {}

    def _normalize_actions(self, arguments):
        return sv._HarnessGameSession._normalize_actions(self, arguments)

    def _terminal_payload(self, requested_actions):
        return sv._HarnessGameSession._terminal_payload(self, requested_actions)

    def timing_payload(self):
        return {}

    def should_stop(self):
        return False

    def _error_payload(self, message):
        return {'executed': False, 'error': message}

    def write_viewer_payload(self):
        pass

    def _execute_action(self, action, *, batch_index, batch_size, flush_viewer_payload=True):
        disp = sv._format_action_display(action.id.name, dict(action.data))
        p = dict(self.payloads.pop(0) if self.payloads else dict(gameplay_changed=True, board_changed=True))
        p.update(executed=True, action_display=disp, action_name=action.id.name)
        self.executed.append(disp)
        self.history_entries.append(SimpleNamespace(action=disp, result=p))
        return p


sv._level_number = lambda game: 2
sv._is_engine_game_over = lambda game: bool(getattr(game, 'over', False))
sv._engine_action_names = lambda game: ['ACTION1', 'ACTION2', 'ACTION3', 'ACTION4', 'ACTION5', 'ACTION6']
EFFECT = dict(gameplay_changed=True, board_changed=True)
NOOP = dict(gameplay_changed=False, board_changed=False)
BAND = dict(gameplay_changed=False, board_changed=True, v61_edge=dict(n=3))
REVERT = dict(gameplay_changed=False, board_changed=False, reverted=True)


def run(payloads, actions=None, stale=None):
    s = Session(payloads)
    acts = actions or [dict(action=a) for a in ['UP', 'DOWN', 'LEFT', 'RIGHT'][:len(payloads)]]
    return s, se.step_env(s, dict(actions=acts, stale_after=stale))


check('setup: stale refusal off, v6.5 batch rule on', (bool(sv.STALE_STATE_BLOCK), se._v65_batch()), (False, True))

s, res = run([NOOP, EFFECT, EFFECT, EFFECT])
check('one NO-OP does not stop the batch', (res.get('executed_count'), res.get('stop_reason')), (4, None))
check('  counted as noop_first_continued', s.__dict__['v65_stats'].get('noop_first_continued'), 1)

s, res = run([NOOP, NOOP, EFFECT, EFFECT])
check('two NO-OPs in a row stop the batch at the second', (res.get('executed_count'), res.get('stop_reason'),
                                                         res.get('skipped_actions')), (2, 'no_op_action', ['LEFT', 'RIGHT']))
check('  stop detail names both and the interior count', (res.get('stop_detail') or '').startswith(
    "Actions 1 and 2 of 4 ('UP', 'DOWN') both EXECUTED but changed nothing on the board (NO-OP twice in a row)"), True)
check('  no interior change sentence', 'No executed action of this batch changed anything in the interior.' in res['stop_detail'],
      True)
check('  receipts', (s.__dict__['v65_stats'].get('noop_second_stops'), s.__dict__['v65_stats'].get('executed_actions')), (1, 2))
check('  no_op_action_index is the second NO-OP', res.get('no_op_action_index'), 2)

s, res = run([EFFECT, NOOP, NOOP, EFFECT])
check('stop at action 3 after one effect', (res.get('executed_count'), res.get('stop_reason')), (3, 'no_op_action'))
check('  "1 of the 3 executed actions changed cells in the interior"',
      '1 of the 3 executed actions changed cells in the interior.' in res.get('stop_detail', ''), True)

s, res = run([BAND, BAND, BAND, EFFECT])
check('BAND-ONLY changes do not stop the batch', (res.get('executed_count'), res.get('stop_reason')), (4, None))
check('  band_only_continued counts the 3 (not last)', s.__dict__['v65_stats'].get('band_only_continued'), 3)

s, res = run([NOOP, BAND, NOOP, EFFECT])
check('NO-OP, BAND-ONLY, NO-OP: the band breaks the streak', (res.get('executed_count'), res.get('stop_reason')), (4, None))

s, res = run([NOOP, REVERT, NOOP, EFFECT])
check('NO-OP, REVERTED, NO-OP: a revert breaks the streak', (res.get('executed_count'), res.get('stop_reason')), (4, None))

s, res = run([NOOP, dict(gameplay_changed=False, board_changed=False), EFFECT],
             actions=[dict(action='UP'), dict(action='RESET'), dict(action='DOWN')])
check('a RESET that changed nothing is not a NO-OP for the streak', (res.get('stop_reason'), s.executed),
      (None, ['UP', 'RESET', 'DOWN']))

s, res = run([EFFECT, EFFECT, NOOP, NOOP])
check('second NO-OP as the last action: stop_reason set, nothing skipped', (res.get('executed_count'), res.get('stop_reason'),
                                                                          res.get('skipped_actions')), (4, 'no_op_action', None))

s, res = run([NOOP], actions=[dict(action='UP')])
check('a single action never stops or counts', (res.get('stop_reason'), s.__dict__['v65_stats'].get('noop_first_continued')),
      (None, None))

s, res = run([NOOP, NOOP], stale='UP')
check('stale arm: executed, not refused', (res.get('executed'), res.get('stop_reason') != 'stale_state', s.executed[:1]),
      (True, True, ['UP']))
check('  stale_would_block counted', s.__dict__['v65_stats'].get('stale_would_block'), 1)

s, res = run([EFFECT], actions=[dict(action='MOUSE', row=5, col=5)], stale='MOUSE 4 4')
check('stale arm after a different click: not counted', s.__dict__['v65_stats'].get('stale_would_block'), None)

real = se._v65_batch
se._v65_batch = lambda: False
s, res = run([BAND, EFFECT, EFFECT])
check('ARC3_V65_BATCH=0: v6.4 rule (a BAND-ONLY CHANGE stops)', (res.get('executed_count'), res.get('stop_reason')),
      (1, 'no_op_action'))
check('  v6.4 stop detail text', (res.get('stop_detail') or '').startswith('Action 1 of 3'), True)
se._v65_batch = real

t = A.TEXTS
check('noop_stop text: second NO-OP in a row', 'was the second action in a row that executed but changed nothing on the board'
      in t['noop_stop'], True)
check('_NOOP_LAST anchor in noop_stop', A._NOOP_LAST in t['noop_stop'], True)
check('system: the new batch sentence', 'A batch stops at the second consecutive action that changes nothing on the board (NO-OP); '
      'a BAND-ONLY CHANGE does not stop it; the remaining actions are not executed.' in t['system'], True)
check('system: no StaleStateActionError sentence', 'StaleStateActionError' in t['system'], False)
check('system: the InvalidActionError sentence stays',
      '- An action that is not valid now raises InvalidActionError and executes nothing.' in t['system'], True)
print('FAILS:', fails)
