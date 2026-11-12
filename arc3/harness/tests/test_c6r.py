"""C6R (apply_v66.py, user + IE 2026-10-17): v6.4's first-no-change batch stop and stale refusal as the defaults, stated as harness
rules with facts only. It covers the stop details, the stale refusal (NO-OP and REVERTED), the receipts, the system-prompt bullets
and noop_stop, v6.5's behaviour with ARC3_V65_BATCH=1, and the run_v6 defaults. step_env runs on the scripted session of
test_v65_c6.py. usage (dev copy of harness_v6): PYTHONPATH=src:. python3 tests/test_c6r.py"""
import json
import os
from pathlib import Path
from types import SimpleNamespace
for k, v in dict(ARC3_V61_TEXTS='1', ARC3_V6_A13='1', ARC3_V61_SUMMARY='1', ARC3_NEW_CHANGED_PROMPTS='1',
                 ARC3_ACTION_INFO='1', ARC3_REPEAT_STATE_GUARD='0', ARC3_V61_RESTART_TOKENS='150000', ARC3_V6_A3='1',
                 ARC3_GUARDS_FROM_LEVEL='2', ARC3_BATCH_NOOP_BLOCK='1', ARC3_STALE_STATE_BLOCK='1', ARC3_V65_BATCH='0',
                 ARC3_DEATH_LEDGER='1', EXPOSE_RESET='on', EXPOSE_UNDO='on', ONLY_RESET_LEVELS='true', ARC3_ANIMATION='1').items():
    os.environ[k] = v
import arcengine  # noqa: E402
import v62.patches  # noqa: E402,F401
import v62.retention as r  # noqa: E402
r.install()
import v62.agent as A  # noqa: E402
import v62.session_ext as se  # noqa: E402
import v62.variants as V  # noqa: E402
from inference.framework import solver as sv  # noqa: E402
fails = []


def check(name, got, want):
    ok = got == want
    print('%-70s -> %-50r %s' % (name, got if not isinstance(got, str) else got[:50], 'OK' if ok else 'WRONG (want %r)' % (want,)))
    if not ok:
        fails.append(name)


class Session:
    def __init__(self, payloads=(), history=()):
        self.game = SimpleNamespace(over=False, current_state=SimpleNamespace(
            available_actions=[0, 1, 2, 3, 4, 5, 6], levels_completed=1,
            raw=SimpleNamespace(state=arcengine.GameState.NOT_FINISHED)))
        self.history_entries, self.payloads, self.executed = list(history), list(payloads), []
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


def run(payloads, actions=None, stale=None, history=()):
    s = Session(payloads, history)
    acts = actions or [dict(action=a) for a in ['UP', 'DOWN', 'LEFT', 'RIGHT'][:len(payloads)]]
    return s, se.step_env(s, dict(actions=acts, stale_after=stale))


check('setup: stale block on, v6.4 batch rule', (sv.STALE_STATE_BLOCK, se._v65_batch()), (True, False))

# --- batch stops ------------------------------------------------------------------------------------------------------
s, res = run([EFFECT, NOOP, EFFECT, EFFECT])
check('NO-OP at 2 of 4 stops the batch', (s.executed, res.get('stop_reason'), res.get('skipped_actions')),
      (['UP', 'DOWN'], 'no_op_action', ['LEFT', 'RIGHT']))
check('NO-OP stop detail (IE text)', res.get('stop_detail'),
      "Action 2 of 4 ('DOWN') executed but changed nothing on the board (NO-OP). This is a harness rule, not a game event: a "
      "batch stops at an action that produced no state change, because the actions after it were planned on the assumption "
      "that it would, and executing them on that assumption would waste actions. Actions 3-4 (['LEFT', 'RIGHT']) were not "
      "executed. Indices are 1-based positions in your submitted list. Only the first executed action changed anything in "
      "the interior. The per-action verdicts are listed in action_trace.")
check('NO-OP receipt', s.__dict__['v65_stats'].get('noop_stops'), 1)
s, res = run([EFFECT, BAND, EFFECT, EFFECT])
check('BAND-ONLY at 2 of 4 stops the batch', (s.executed, res.get('stop_reason')), (['UP', 'DOWN'], 'no_op_action'))
check('BAND-ONLY stop detail (IE text)', res.get('stop_detail'),
      "Action 2 of 4 ('DOWN') executed but changed only 3 cells of the outer band (BAND-ONLY CHANGE), nothing in the interior. "
      "This is a harness rule, not a game event: a batch stops at an action that changed nothing in the interior, because the "
      "actions after it were planned for an interior change that did not happen, and executing them would waste actions. The "
      "executed action and its band change stand. Actions 3-4 (['LEFT', 'RIGHT']) were not executed. Indices are 1-based "
      "positions in your submitted list. Only the first executed action changed anything in the interior. The per-action "
      "verdicts are listed in action_trace.")
check('BAND-ONLY receipt', (s.__dict__['v65_stats'].get('band_only_stops'), s.__dict__['v65_stats'].get('noop_stops')), (1, None))
s, res = run([EFFECT, NOOP])
check('NO-OP as the last of 2: nothing skipped, no rule sentence, no stop receipt',
      ('harness rule' in (res.get('stop_detail') or ''), res.get('skipped_actions'), s.__dict__['v65_stats'].get('noop_stops')),
      (False, None, None))
s, res = run([NOOP], actions=[dict(action='UP')])
check('a single action that changes nothing is not stopped', res.get('stop_reason'), None)
s, res = run([NOOP, EFFECT, EFFECT])
check('NO-OP first: only it executes', (s.executed, res.get('stop_reason')), (['UP'], 'no_op_action'))
check('  "No executed action of this batch changed anything in the interior."',
      'No executed action of this batch changed anything in the interior.' in res.get('stop_detail', ''), True)

# --- stale refusal ----------------------------------------------------------------------------------------------------
s, res = run([EFFECT], actions=[dict(action='DOWN')], stale='UP')
check('stale: the first call after a NO-OP is refused', (s.executed, res.get('stop_reason'), res.get('executed')),
      ([], 'stale_state', False))
check('stale refusal text (IE)', res.get('stop_detail'),
      "DOWN was not executed. The previous action in this snippet, UP, was a NO-OP (nothing on the board changed). The harness, "
      "not the game, refuses the first action call after a NO-OP in the same snippet, because positions and paths computed on "
      "the assumption that UP had changed the board no longer describe it, and acting on them would waste actions. The "
      "refusal happens once; the next action call executes.")
check('stale receipt', s.__dict__['v65_stats'].get('stale_refusals'), 1)
s, res = run([EFFECT], actions=[dict(action='DOWN')], stale='UP',
             history=[SimpleNamespace(action='UP', result={'reverted': True})])
check('stale after a REVERTED action: the REVERTED text', res.get('stop_detail'),
      "DOWN was not executed. The previous action in this snippet, UP, was REVERTED (it animated, then the board returned to "
      "its previous state). The harness, not the game, refuses the first action call after a REVERTED action in the same "
      "snippet, because positions and paths computed on the assumption that UP had changed the board no longer describe it, "
      "and acting on them would waste actions. The refusal happens once; the next action call executes.")
s, res = run([EFFECT], actions=[dict(action='MOUSE', row=5, col=5)], stale='MOUSE(row=4, col=7)')
check('stale: a click elsewhere after a no-change click executes', (s.executed, res.get('stop_reason')),
      (['MOUSE(row=5, col=5)'], None))

# --- texts ------------------------------------------------------------------------------------------------------------
RAW = json.load(open(Path(A.__file__).with_name('texts.json')))['texts']
t = V.apply(json.loads(json.dumps(RAW)))
S = t['system']
check('system: the harness batch bullet', "- The harness (not the game) stops a batch at the first action that changes nothing in "
      "the interior (a NO-OP or a BAND-ONLY CHANGE), because the actions after it were planned for an interior change that did "
      "not happen, and executing them would waste actions; that action executes, the remaining actions are not executed, and "
      "the stop is reported. Re-locate from `current_frame`" in S, True)
check('system: the v6.4 bullet tail kept (single-action note, one action on one board)',
      ('so test its `gameplay_changed`' in S, 'a stop concerns one action on one board.' in S), (True, True))
check('system: the harness stale bullet', "the harness (not the game) refuses the next action(...) call in the same snippet once "
      "with StaleStateActionError, because positions and paths computed on the assumption that the action had changed the "
      "board no longer describe it, and acting on them would waste actions: re-read the state first; a later call proceeds." in S,
      True)
check('noop_stop: the harness text, no re-locate advice',
      (t['noop_stop'].startswith('Your batch was stopped by the harness, not by the game: {action} (position {p} of {n})'),
       'Re-locate' in t['noop_stop'], '{m}' in t['noop_stop'] and '{skipped}' in t['noop_stop']), (True, False, True))
check('noop_last unchanged (no stop)', 'Re-locate from `current_frame` instead of counting moves.' in t['noop_last'], True)
os.environ['ARC3_V65_BATCH'], os.environ['ARC3_STALE_STATE_BLOCK'] = '1', '0'
t5 = V.apply(json.loads(json.dumps(RAW)))
check('ARC3_V65_BATCH=1, block off: v6.5 texts (second NO-OP bullet, no stale bullet, no C6R text)',
      ('second consecutive action that changes nothing on the board' in t5['system'], 'StaleStateActionError' in t5['system'],
       'The harness (not the game)' in t5['system'], 'harness, not by the game' in t5['noop_stop']), (True, False, False, False))
os.environ['ARC3_V65_BATCH'], os.environ['ARC3_STALE_STATE_BLOCK'] = '0', '1'

# --- run_v6 -----------------------------------------------------------------------------------------------------------
R = open(Path(A.__file__).with_name('run_v6.py')).read()
check('run_v6: stale block on, memory key restored, V65_BATCH 0, recorded',
      ("'ARC3_STALE_STATE_BLOCK': '1', 'ARC3_MEMORY_SECTIONS': 'off'," in R, "# v6.5 C6 'ARC3_MEMORY_SECTIONS'" in R,
       "os.environ.setdefault('ARC3_V65_BATCH', '0')" in R, "'ARC3_V66_SEEDS', 'ARC3_V65_BATCH')}," in R), (True, False, True, True))
print('FAILS:', fails)
