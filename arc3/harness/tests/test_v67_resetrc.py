"""test_v67_resetrc.py (2026-10-18, Claude 099768dc): apply_v67.py's item RESETRC (ARC3_V67_RESET_RECEIPT=1; scope conversation
by default, MIN_ACTIONS 10), on a scripted game session through the real session_ext.step_env and the attempt bookkeeping
(v62.patches), with the agent in sandbox_ext's thread-local as during a Python call. IE's conditions:
  - the refusal conditions (GAME OVER, too few actions, the second call; the scopes: conversation re-armed by a new
    conversation or level, attempt re-armed by a new attempt);
  - the batch stop (the actions after the held RESET not executed and reported);
  - the message filled correctly (n from the per-level counter, k, UNDO validity from the valid actions);
  - the counters and the outcomes (reissued, cleared, died, harness restart, run end);
  - the system-prompt bullet only under the switch; switch off = byte-identical (also test_v67_ident.py and the suite).
usage (dev copy of harness_v6): PYTHONPATH=src:. python3 tests/test_v67_resetrc.py"""
import json
import os
from pathlib import Path
from types import SimpleNamespace

_src = open(Path(__file__).with_name('test_v67_soft.py')).read()
ENV = eval(_src.split('ENV = ', 1)[1].split('\n\n', 1)[0])
os.environ.update(ENV)
os.environ.update(ARC3_V67_RESET_RECEIPT='1', EXPOSE_UNDO='on')
import arcengine  # noqa: E402
import v62.patches as P  # noqa: E402
import v62.retention as r  # noqa: E402
r.install()
import v62.agent as A  # noqa: E402
import v62.session_ext as se  # noqa: E402
if not hasattr(se, 'RESET_RECEIPT'):                     # RESETRC is parked: not in apply_v67.py's DEFAULT items
    print('SKIP: RESETRC is not applied (apply_v67.py DIR SOFT,LOOP,TEXTS67,...,RESETRC applies it)')
    print('FAILS: []')
    raise SystemExit(0)
import v62.sandbox_ext as sx  # noqa: E402
import v62.variants as V  # noqa: E402
from inference.framework import solver as sv  # noqa: E402
fails = []


def check(name, got, want):
    ok = got == want
    print('%-84s -> %-40r %s' % (name, got if not isinstance(got, str) else got[:40], 'OK' if ok else 'WRONG (want %r)' % (want,)))
    if not ok:
        fails.append(name)


class Session:
    """a game session for step_env: UP completes the level, DOWN ends the attempt in GAME OVER, others change the board"""

    def __init__(self, undo=True):
        self.game = SimpleNamespace(over=False, level=1, names=['ACTION1', 'ACTION2', 'ACTION3', 'ACTION4'] + (['ACTION7'] if undo else []),
                                    current_state=SimpleNamespace(available_actions=list(range(8)), levels_completed=0,
                                                                  raw=SimpleNamespace(state=arcengine.GameState.NOT_FINISHED)),
                                    game_run=SimpleNamespace(actions_per_level=[0, 0, 0]))
        self.history_entries, self.executed, self.action_count = [], [], 0
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

    def _base(self, action, *, batch_index=1, batch_size=1, flush_viewer_payload=True, automatic=False):
        name = action.id.name
        disp = sv._format_action_display(name, dict(action.data))
        self.game.game_run.actions_per_level[self.game.level - 1] += 1
        self.action_count += 1
        over, done = name == 'ACTION2', name == 'ACTION1'
        if done:
            self.game.level += 1
        self.game.over = over
        p = dict(executed=True, action_display=disp, action_name=name, level=self.game.level, gameplay_changed=True,
                 board_changed=True, game_over=over, level_completed=done, automatic=automatic)
        self.history_entries.append(SimpleNamespace(action=disp, result=p, frame=None))
        self.executed.append(disp)
        return p

    def _execute_action(self, action, **kw):
        return P._execute_action(self, action, **kw)

    def auto_reset(self):
        if self.game.over:
            self._execute_action(SimpleNamespace(id=SimpleNamespace(name='RESET'), data={}), batch_index=1, batch_size=1,
                                 automatic=True)
            self.game.over = False


P._base_execute = Session._base
sv._level_number = lambda game: game.level
sv._is_engine_game_over = lambda game: bool(game.over)
sv._engine_action_names = lambda game: list(game.names)


class Quiet:
    rebased_away, blocks = set(), []

    def span_evicted(self, a, b):
        return 0


def setup(undo=True):
    log = []
    agent = A.V6Agent(session=Quiet(), logger=lambda k, p: log.append((k, p)), count_tokens=lambda t: max(1, len(t) // 4),
                      model='M', base_url='http://x/v1', provider='vllm')
    sx.CURRENT.agent = agent
    return Session(undo), agent, log


def step(s, acts):
    """one Python call's action(...) list (the per-call counters reset, as _run_python_tool does)"""
    s.__dict__['v61_call_executed'] = 0
    return se.step_env(s, dict(actions=[dict(action=a) for a in acts]))


def receipts(log):
    return [p for k, p in log if k == 'v67-reset-receipt']


check('switch, scope and minimum (defaults)', (se.RESET_RECEIPT, se.RESET_RECEIPT_SCOPE, se.RESET_RECEIPT_MIN), (True, 'conversation', 10))

# --- the refusal conditions ----------------------------------------------------------------------------------------
s, agent, log = setup()
step(s, ['RIGHT'] * 9)
res = step(s, ['RESET'])
check('9 actions in the attempt: the RESET executes (no hold)', (res.get('executed'), s.executed[-1], agent._v6_stats.get('reset_receipts')),
      (True, 'RESET', None))
step(s, ['RIGHT'] * 10)
res = step(s, ['RESET', 'UP', 'DOWN'])
msg = res.get('error') or ''
check('10 actions: held; nothing executed; the actions after it reported', (res.get('executed'), res.get('stop_reason'),
      res.get('skipped_actions'), s.executed.count('RESET'), agent._v6_stats.get('reset_receipts')),
      (False, 'reset_receipt', ['RESET', 'UP', 'DOWN'], 1, 1))
check('the message, filled (k 10 of this attempt, n 20 on the level, UNDO valid)', msg,
      "RESET was not executed. This is a harness rule, not a game event: the first RESET on a level in each conversation, in "
      "an attempt with at least 10 executed actions, is held once and reported. A RESET returns this level to its first board "
      "and undoes the effect of the 10 actions of this attempt; all 20 actions taken on this level so far stay counted for the "
      "level's score, and a RESET adds one more. UNDO is among this game's valid actions. A second RESET call in this attempt "
      "executes. The 2 actions after it in the list were not executed: UP, DOWN.")
res = step(s, ['RESET'])
check('the second RESET in the attempt executes: outcome reissued', (res.get('executed'), s.executed[-1],
      [(p['outcome'], p['n'], p['k'], p['undo_valid'], p['actions_after']) for p in receipts(log)],
      agent._v6_stats.get('resets_reissued'), agent._v6_stats.get('resets_withdrawn')), (True, 'RESET', [('reissued', 20, 10, True, 1)], 1, None))
step(s, ['RIGHT'] * 12)
res = step(s, ['RESET'])
check('scope conversation: no second hold on this level in this conversation (a new attempt)', (res.get('executed'),
      agent._v6_stats.get('reset_receipts')), (True, 1))
step(s, ['RIGHT'] * 12)
agent._v61_restarts += 1                                    # a new conversation (a harness restart) re-arms it
res = step(s, ['RESET'])
check('scope conversation: a new conversation re-arms it', (res.get('executed'), res.get('stop_reason')), (False, 'reset_receipt'))
step(s, ['UP'])                                             # the level clears: outcome cleared
check('the level cleared while pending: outcome cleared (withdrawn)', ([p['outcome'] for p in receipts(log)],
      agent._v6_stats.get('resets_withdrawn')), (['reissued', 'cleared'], 1))
step(s, ['RIGHT'] * 10)
res = step(s, ['RESET'])
check('a new level re-arms it (same conversation); UNDO counted', (res.get('stop_reason'), 'all 10 actions taken on this level' in
      res.get('error', '')), ('reset_receipt', True))
step(s, ['DOWN'])                                           # GAME OVER while pending: outcome died
s.auto_reset()
check('a GAME OVER while pending: outcome died', [p['outcome'] for p in receipts(log)][-1], 'died')
s.game.over = True
step(s, ['RIGHT'] * 0 or ['RESET'])
check('at GAME OVER the RESET executes (no hold)', (s.executed[-1], agent._v6_stats.get('reset_receipts')), ('RESET', 3))
s.game.over = False

# --- outcomes at the harness restart and the run end; no UNDO --------------------------------------------------------------
s, agent, log = setup(undo=False)
step(s, ['RIGHT'] * 10)
res = step(s, ['RESET'])
check('UNDO not among the valid actions', 'UNDO is not among this game\'s valid actions.' in res.get('error', ''), True)
agent._v67_rr_close('harness_restart', s.action_count)
s2, agent2, log2 = setup()
step(s2, ['RIGHT'] * 10)
step(s2, ['RESET'])
agent2._v67_rr_close('run_end')
check('harness restart and run end close the receipt (withdrawn)', ([p['outcome'] for p in receipts(log)], [p['outcome'] for p in
      receipts(log2)], agent._v6_stats.get('resets_withdrawn'), agent2._v6_stats.get('resets_withdrawn')),
      (['harness_restart'], ['run_end'], 1, 1))
check('the receipt event fields', sorted(receipts(log)[0]), sorted(['level', 'n', 'k', 'undo_valid', 'scope', 'key', 'step', 'outcome',
                                                                    'actions_after']))
src = open(Path(A.__file__)).read()
check('the agent closes a pending receipt at a harness restart', "self._v67_rr_close('harness_restart'," in src, True)

# --- scope attempt ---------------------------------------------------------------------------------------------------
se.RESET_RECEIPT_SCOPE = 'attempt'
s, agent, log = setup()
step(s, ['RIGHT'] * 10)
first = step(s, ['RESET'])
again = step(s, ['RESET'])
step(s, ['RIGHT'] * 10)
third = step(s, ['RESET'])
check('scope attempt: held, re-issued, then held again in the next attempt', (first.get('stop_reason'), again.get('executed'),
      third.get('stop_reason'), 'the first RESET of an attempt with at least 10 executed actions is held once' in first.get('error', '')),
      ('reset_receipt', True, 'reset_receipt', True))
se.RESET_RECEIPT_SCOPE = 'conversation'

# --- the switch off: RESET executes as before ------------------------------------------------------------------------------
se.RESET_RECEIPT = False
s, agent, log = setup()
step(s, ['RIGHT'] * 15)
res = step(s, ['RESET'])
check('switch off: the RESET executes, no receipt', (res.get('executed'), agent._v6_stats.get('reset_receipts')), (True, None))
se.RESET_RECEIPT = True
sx.CURRENT.agent = None

# --- texts: the bullet, the refusal reason ---------------------------------------------------------------------------------
RAW = json.load(open(Path(A.__file__).with_name('texts.json')))['texts']


def build(**env):
    keep = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    try:
        return V.apply(json.loads(json.dumps(RAW)))
    finally:
        for k, v in keep.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


BUL_C = ("- The harness (not the game) holds the first RESET on each level in each conversation once, when the attempt has at least "
         "10 executed actions, and reports the level's action count; calling RESET again executes it.")
BUL_A = ("- The harness (not the game) holds the first RESET of each attempt with at least 10 executed actions once and reports the "
         "level's action count; calling RESET again executes it.")
on, off = build(ARC3_V67_RESET_RECEIPT='1'), build(ARC3_V67_RESET_RECEIPT='0')
batch = [l for l in off['system'].split('\n') if l.startswith('- The harness (not the game) stops a batch')][0]
check('the bullet: only under the switch, right after the batch-stop bullet; nothing else changes',
      (on['system'] == off['system'].replace(batch, batch + '\n' + BUL_C), sorted(k for k in on if on[k] != off.get(k))),
      (True, ['system']))
check('the bullet with scope attempt', BUL_A in build(ARC3_V67_RESET_RECEIPT='1', ARC3_V67_RESET_RECEIPT_SCOPE='attempt')['system'], True)
v65 = build(ARC3_V67_RESET_RECEIPT='1', ARC3_V65_BATCH='1', ARC3_STALE_STATE_BLOCK='0')['system']
check('the bullet after v6.5\'s batch bullet too', v65.split(BUL_C)[0].rstrip().split('\n')[-1].startswith('- A batch stops at'), True)
check('the held RESET in refused_action_calls', A._refusal_text([dict(stop_reason='reset_receipt', requested_actions=['RESET'],
      error='RESET was not executed.')]).split(': ')[1].split(' (1)')[0], A.REFUSAL_REASONS['reset_receipt'])
RUN = open(Path(A.__file__).with_name('run_v6.py')).read()
check('run_v6: defaults, record, the run-end close', ("os.environ.setdefault('ARC3_V67_RESET_RECEIPT', '0')" in RUN,
      "os.environ.setdefault('ARC3_V67_RESET_RECEIPT_SCOPE', 'conversation')" in RUN,
      "os.environ.setdefault('ARC3_V67_RESET_RECEIPT_MIN_ACTIONS', '10')" in RUN, "'ARC3_V67_RESET_RECEIPT_MIN_ACTIONS'," in RUN,
      "a._v67_rr_close('run_end')" in RUN), (True,) * 5)
print('FAILS:', fails)
