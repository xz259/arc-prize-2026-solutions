"""test_v67_soft.py (2026-10-18, Claude 099768dc): apply_v67.py's item SOFT (ARC3_V67_RESTART_KIND), through agent.analyze with
the real sandbox, on the scaffolding of test_v64_fixes.py / test_v63_noact_budget.py; the reviewer's (IE) conditions:
  1 a soft restart on a scripted session: no RESET, the board kept, the programs kept (not archived), the notes shown, the
    opener's facts true for soft (no first-board claim) and unchanged for hard;
  2 attempt accounting with kind soft: the attempt record, the death ledger and the start-board counters untouched by the
    restart, the attempt going on across the two conversations; the hard twin for contrast;
  3 the position-limit path under soft (notes first, and immediate): no RESET, a new session, no second restart;
  4 hard under soft: GAME OVER, an explicit kind='hard', the dead-board rules (GAME OVERs from the board, the window);
  5 the receipts, the restart event and the game's per-level restart record;
  6 the switch off (default): the rendered texts and a scripted run (budget restart with a death, a program, a refused
    action and the notes retry; no-action cues and restart; position restarts; a session error; a second restart with
    hidden history) byte-identical to the v66g base, run in a subprocess on each tree (the base: the patcher's md5-checked
    copies v62/*.pre-v67); the existing restart tests run in the suite.
usage (dev copy of harness_v6): PYTHONPATH=src:. python3 tests/test_v67_soft.py"""
import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

ENV = {'MULTIMODAL_CONTEXT': 'current_grid', 'MULTIMODAL_UPSCALE': '8', 'ARC3_PERSISTENT_FUNCTIONS': '1',
       'ARC3_PERSISTENT_FUNCTIONS_SCOPE': 'game', 'ARC3_PERSISTENT_FUNCTIONS_IMPORTS': '1', 'ARC3_ANIMATION': '1',
       'LOCAL_ANALYZER_TOOL_TIMEOUT': '10', 'ARC3_ACTION_ECHO': '1', 'ARC3_V6_YIELD_TOKENS': '4096',
       'ARC3_V61_TEXTS': '1', 'ARC3_V6_A13': '1', 'ARC3_V61_SUMMARY': '1', 'ARC3_NEW_CHANGED_PROMPTS': '1',
       'ARC3_ACTION_INFO': '1', 'ARC3_REPEAT_STATE_GUARD': '0', 'ARC3_V61_RESTART_TOKENS': '200000', 'ARC3_V6_A3': '1',
       'ARC3_GUARDS_FROM_LEVEL': '2', 'ARC3_BATCH_NOOP_BLOCK': '1', 'ARC3_STALE_STATE_BLOCK': '1', 'ARC3_V65_BATCH': '0',
       'ARC3_DEATH_LEDGER': '1', 'EXPOSE_RESET': 'on', 'ARC3_V61_RESTART_HISTORY': 'level'}

# the scripted session and game, shared by the in-process tests and the identity script (exec'd in both)
FAKES = r'''
import arcengine as _ae
import v62.patches as P
import v62.session_ext as se
from inference.agent.runtime_state import Frame, HistoryEntry, write_runtime_state
from v62.transport import AppendOnlyViolation, Completion
VALID = ['RIGHT', 'LEFT', 'UP', 'DOWN']


def board(pos):
    """the scripted game's board: a marker at row 30, column 10 + 2 * pos (interior)"""
    g = [[0] * 64 for _ in range(64)]
    g[30][10 + 2 * pos] = 3
    return g


class FakeSession:
    """script items: (code or None for a reply without a call, generated tokens); 'ERROR' raises a session error."""

    def __init__(self, script):
        self.script, self.sent = list(script), []
        self.rebased_away, self.blocks, self.closed, self.last_receipt = set(), [], False, {}

    def generate(self, external):
        self.sent.append(json.loads(json.dumps(external, default=str)))
        item = self.script.pop(0)
        if item == 'ERROR':
            raise AppendOnlyViolation('scripted session error')
        code, tokens = item
        calls = ([dict(id='call-%d' % len(self.sent), type='function',
                       function=dict(name='python', arguments=json.dumps(dict(code=code))))] if code is not None else [])
        msg = dict(role='assistant', content='', reasoning_content='thinking', tool_calls=calls)
        return Completion(message=msg, finish_reason='tool_calls' if calls else 'stop',
                          usage=dict(prompt_tokens=10, completion_tokens=tokens), token_ids=[1] * tokens,
                          assistant_token_ids=[1] * tokens, pending=False, closed_by=None, persistent={}, seconds=0.0)

    def span_evicted(self, a, b):
        return 0

    def close(self):
        self.closed = True


class _Raw:
    state = _ae.GameState.NOT_FINISHED


class _State:
    def __init__(self):
        self.raw = _Raw()


class _Run:
    def __init__(self, n):
        self.number_of_levels, self.levels_completed, self.state = n, 0, 'playing'
        self.actions_per_level = [0] * n


class _TG:
    def __init__(self, n):
        self.number_of_levels, self.current_state, self.game_run = n, _State(), _Run(n)


class FakeGame:
    """The game session of the restart paths. Its engine: RIGHT / LEFT move the marker, UP completes the level, DOWN ends
    the attempt in GAME OVER (the board unchanged), RESET returns to the level's first board; the game run counts actions
    per level as taaf does. Actions and the restart's RESET go through v62.patches._execute_action (the attempt record),
    the start-board facts are kept as session_ext.execute_action keeps them, and auto_reset() is the solver's automatic
    restart after a GAME OVER (solver._play_inner, before each analyzer call)."""

    def __init__(self, state_path, n_levels=3):
        self.state_path, self.level, self.pos = state_path, 1, 0
        self.game = _TG(n_levels)
        self.last_engine_action = None
        self.history_entries = [HistoryEntry(action='', frame=Frame(grid=board(0), step=0, level=1))]
        self.v6_anim, self.engine = {}, []
        self._write()

    def _write(self):
        write_runtime_state(self.state_path, current_frame=self.history_entries[-1].frame, history=self.history_entries)

    def timing_payload(self):
        return dict(time_remaining_seconds=20000.0)

    def _base(self, action, *, batch_index=1, batch_size=1, automatic=False, **kw):
        name = action.id.name
        before = self.history_entries[-1].frame.grid
        over, done, lvl0 = name == 'DOWN', name == 'UP', self.level
        if name in ('RESET', 'UP'):
            self.pos = 0
        elif name == 'RIGHT':
            self.pos += 1
        elif name == 'LEFT':
            self.pos = max(0, self.pos - 1)
        if done:
            self.level += 1
            self.game.game_run.levels_completed += 1
        grid = board(self.pos)
        self.game.current_state.raw.state = _ae.GameState.GAME_OVER if over else _ae.GameState.NOT_FINISHED
        self.game.game_run.actions_per_level[lvl0 - 1] += 1
        self.last_engine_action = name
        step = len(self.history_entries)
        payload = dict(executed=True, action_num=step, level=self.level, action_name=name, action_display=name,
                       board_changed=grid != before, gameplay_changed=grid != before, no_op=grid == before, game_over=over,
                       level_completed=done, run_complete=False, done=False, automatic=automatic,
                       valid_actions=list(VALID), score=self.level - 1, state='GAME_OVER' if over else 'NOT_FINISHED')
        start = self.__dict__.get('v61_level_start')
        if name == 'RESET' or done:
            self.__dict__['v61_level_start'] = grid
        se._start_facts(self.__dict__, payload, name == 'RESET' or done or start is None, automatic, over or done, grid,
                        start)
        self.history_entries.append(HistoryEntry(action=name, frame=Frame(grid=grid, step=step, level=self.level),
                                                 result=dict(payload)))
        self._write()
        self.engine.append(name + (' (automatic)' if automatic else ''))
        return payload

    def _execute_action(self, action, **kw):
        return P._execute_action(self, action, **kw)

    def step_env(self, arguments):
        if arguments.get('query') == 'animation':
            return {'record': None}
        acts, done = list(arguments.get('actions') or []), []
        for i, a in enumerate(acts, 1):
            name = a.get('action') if isinstance(a, dict) else str(a)
            p = self._execute_action(SimpleNamespace(id=SimpleNamespace(name=name), data={}), batch_index=i,
                                     batch_size=len(acts))
            done.append(p)
            if p.get('game_over') or p.get('level_completed'):
                break
        last = dict(done[-1]) if done else {}
        last.update(executed=bool(done), executed_count=len(done), executed_actions=[p['action_display'] for p in done],
                    requested_count=len(acts))
        return last

    def auto_reset(self):
        if self.game.current_state.raw.state == _ae.GameState.GAME_OVER and self.last_engine_action != 'RESET':
            self._execute_action(SimpleNamespace(id=SimpleNamespace(name='RESET'), data={}), batch_index=1, batch_size=1,
                                 automatic=True)


P._base_execute = FakeGame._base


def harness(scripts, budget=10 ** 9, pos_min=50000, warn=(16000, 32000, 48000), limit=64000, **knobs):
    """an agent on FakeGame with one FakeSession per conversation; knobs set module values (RESTART_KIND, DEAD_*)"""
    A.ta._priority_gate = lambda: None
    A.RESTART_TOKENS, A.RESTART_NOTES, A.POSITION_NOTES_MIN = budget, True, pos_min
    A.NOACT_WARN, A.NOACT_TOKENS = tuple(warn), limit
    for k, v in knobs.items():
        setattr(A, k, v)
    state = Path(tempfile.mkdtemp()) / 'game_runtime_state.json'
    game = FakeGame(state)
    log, sessions = [], [FakeSession(s) for s in scripts]
    queue = list(sessions[1:])
    agent = A.V6Agent(session=sessions[0], logger=lambda k, p: log.append((k, p)), count_tokens=lambda t: max(1, len(t) // 4),
                      model='M', base_url='http://x/v1', provider='vllm', session_factory=lambda: queue.pop(0))

    def go():
        game.auto_reset()
        return agent.analyze(state, len(game.history_entries) - 1, valid_actions=VALID, step_env=game.step_env,
                             should_stop=lambda: False)
    return game, sessions, agent, log, go


NOTES = 'save_summary("Notes L1:\\n- the marker moves right one cell per RIGHT\\n- DOWN ends the attempt")'
PROG = 'def double(x):\n    """twice x"""\n    return 2 * x\n'
LOOK = 'print("LIB", sorted(p["name"] for p in programs()))\nprint("START", attempt_start_frame.step, level_start_frame.step)\n'
'''

# the switch-off identity script: the same scripted runs on a tree, as JSON (images hashed, timings dropped)
IDENT = r'''
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
os.environ.update(json.loads(sys.argv[1]))
import v62.patches  # noqa: E402,F401
import v62.retention as r  # noqa: E402
r.install()
import v62.variants as V  # noqa: E402
import v62.agent as A  # noqa: E402
exec(sys.argv[2])
NEW_KEYS = ('kind', 'hard_why', 'actions_on_level')


def norm(x):
    if isinstance(x, dict):
        if x.get('type') == 'image_url':
            return dict(type='image_url', sha1=hashlib.sha1(json.dumps(x, sort_keys=True).encode()).hexdigest())
        return {k: norm(v) for k, v in x.items()}
    if isinstance(x, list):
        return [norm(v) for v in x]
    return x


def collect(game, sessions, agent, log):
    events, v67 = [], []
    for k, p in log:
        p = json.loads(json.dumps(p, default=str))
        if k == 'tool-output':
            p.pop('seconds', None)
        if k == 'v61-restart':
            v67.append({x: p.pop(x) for x in NEW_KEYS if x in p})
        events.append([k, norm(p)])
    stats = json.loads(json.dumps(agent._v6_stats, default=str))
    v67.append(dict(hard_restarts=stats.pop('hard_restarts', None)))
    keys = ('v6_level_attempts', 'v61_level_attempt_noops', 'v62_level_attempt_ends', 'v6_attempt_actions', 'v62_since_start',
            'v62_level_start_returns', 'v61_harness_resets', 'v61_voluntary_resets')
    return dict(sent=[norm(s.sent) for s in sessions], engine=game.engine, events=events, stats=stats, v67=v67,
                records=json.loads(json.dumps({k: game.__dict__.get(k) for k in keys})),
                library=sorted(agent._kept_functions), archive=sorted(agent._v61_archive), hidden=agent._v61_hidden)


runs = {}
# 1. budget restart: a program, a death (ledger), the notes turn with a refused action and its retry, the takeover, two turns
game, sessions, agent, log, go = harness([[(PROG + 'action(["RIGHT"])', 100), ('action(["DOWN"])', 100),
                                           ('action(["RIGHT", "RIGHT"])', 400), ('action(["LEFT"])', 50), (NOTES, 80)],
                                          [(LOOK + 'action(["RIGHT"])', 50), ('action(["RIGHT"])', 50)]], budget=500)
for _ in range(6):
    go()
runs['budget'] = collect(game, sessions, agent, log)
# 2. no-action cues (16K-style facts cue, the limit clause, must-act) and the no-action restart
game, sessions, agent, log, go = harness([[('print(1)', 150), ('print(2)', 100), ('print(3)', 100), ('print(4)', 100),
                                           (NOTES, 50)], [('action(["RIGHT"])', 50)]], warn=(100, 200, 300), limit=400)
for _ in range(3):
    go()
runs['noact'] = collect(game, sessions, agent, log)
# 3. position limit with the notes turn first, then the next conversation goes on
game, sessions, agent, log, go = harness([[('action(["RIGHT"])', 600), (NOTES, 80)],
                                          [('action(["RIGHT"])', 50), ('action(["RIGHT"])', 50)]], pos_min=500)
go()
sessions[0].last_receipt = {'logical_tokens': A.POSITION_RESTART}
for _ in range(3):
    go()
runs['position_notes'] = collect(game, sessions, agent, log)
# 4. position limit below the notes minimum: an immediate restart
game, sessions, agent, log, go = harness([[('action(["RIGHT"])', 600)], [('action(["RIGHT"])', 50)]], pos_min=10 ** 9)
go()
sessions[0].last_receipt = {'logical_tokens': A.POSITION_RESTART}
go()
runs['position_now'] = collect(game, sessions, agent, log)
# 5. a session error: the forced restart
game, sessions, agent, log, go = harness([[('action(["RIGHT"])', 100), 'ERROR'], [('action(["RIGHT"])', 50)]])
for _ in range(3):
    go()
runs['forced'] = collect(game, sessions, agent, log)
# 6. two budget restarts with lazy history on (the hidden-history texts)
real_lazy = A._lazy_enabled
A._lazy_enabled = lambda: True
game, sessions, agent, log, go = harness([[('action(["RIGHT"])', 600), (NOTES, 50)], [('action(["RIGHT"])', 600), (NOTES, 50)],
                                          [('action(["RIGHT"])', 50)]], budget=500)
for _ in range(5):
    go()
A._lazy_enabled = real_lazy
runs['twice'] = collect(game, sessions, agent, log)
raw = json.load(open(Path(A.__file__).with_name('texts.json')))['texts']
texts = {}
for name, env in (('production', {}), ('v65_batch', {'ARC3_V65_BATCH': '1', 'ARC3_STALE_STATE_BLOCK': '0'}),
                  ('control_sco0_seeds0', {'ARC3_V66_SCO': '0', 'ARC3_V66_SEEDS': '0'}), ('v66_texts_off', {'ARC3_V66_TEXTS': '0'}),
                  ('no_reset', {'EXPOSE_RESET': 'off'})):
    keep = {k: os.environ.get(k) for k in env}
    os.environ.update(env)
    texts[name] = V.apply(json.loads(json.dumps(raw)))
    for k, v in keep.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v
print('IDENT-JSON ' + json.dumps(dict(runs=runs, texts=texts, agent_texts=A.TEXTS), sort_keys=True, default=str))
'''

os.environ.update(ENV)
os.environ['ARC3_V67_RESTART_KIND'] = 'soft'            # the soft texts and switch for the in-process tests
import v62.patches  # noqa: E402,F401
import v62.retention as r  # noqa: E402
r.install()
import v62.variants as V  # noqa: E402
import v62.agent as A  # noqa: E402
from inference.agent.runtime_state import load_runtime_state  # noqa: E402
from inference.agent.vision_context import current_grid_image_part  # noqa: E402
exec(compile(FAKES, 'test_v67_soft.py (FAKES)', 'exec'), globals())
import arcengine  # noqa: E402
fails = []


def h(scripts, kind='soft', dead=3, window=0, **kw):
    """harness() with the v6.7 knobs always set (RESTART_KIND, DEAD_GAMEOVERS, DEAD_WINDOW)"""
    return harness(scripts, RESTART_KIND=kind, DEAD_GAMEOVERS=dead, DEAD_WINDOW=window, **kw)


def check(name, got, want):
    ok = got == want
    print('%-84s -> %-40r %s' % (name, got if not isinstance(got, str) else got[:40], 'OK' if ok else 'WRONG (want %r)' % (want,)))
    if not ok:
        fails.append(name)


def user_text(batch):
    """all text parts of the last user message of a sent batch"""
    m = [x for x in batch if x.get('role') == 'user'][-1]
    c = m['content']
    return c if isinstance(c, str) else '\n'.join(p.get('text', '') for p in c if p.get('type') == 'text')


def tool_text(batch):
    out = []
    for m in batch:
        if m.get('role') == 'tool':
            c = m['content']
            out.append(c if isinstance(c, str) else '\n'.join(p.get('text', '') for p in c if p.get('type') == 'text'))
    return '\n'.join(out)


def board_url(batch):
    parts = [x for x in batch if x.get('role') == 'user'][-1]['content']
    i = next(i for i, p in enumerate(parts) if p.get('type') == 'text' and p.get('text') == 'Current board.')
    return parts[i + 1]['image_url']['url']


def restarts(log):
    return [p for k, p in log if k == 'v61-restart']


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


RAW = json.load(open(Path(A.__file__).with_name('texts.json')))['texts']
TEXTS_SOFT, TEXTS_OFF = A.TEXTS, build(ARC3_V67_RESTART_KIND='hard')
FALSE_FOR_SOFT = ('first board', 'a RESET, then', 'set aside', 'level restarts', 'level resets', 'RESET of level')
REC_KEYS = ('v6_level_attempts', 'v61_level_attempt_noops', 'v62_level_attempt_ends', 'v6_attempt_actions',
            'v62_level_start_returns', 'v61_harness_resets')
START_KEYS = ('v61_level_start', 'v62_since_start', 'v62_start_diff', 'v62_near_armed', 'v62_far_armed')

# === 0. switch, defaults, texts ========================================================================================
check('0 module switch and knobs (soft in this process; defaults 3 / 0)', (A.RESTART_KIND, A.DEAD_GAMEOVERS, A.DEAD_WINDOW),
      ('soft', 3, 0))
soft_keys = sorted(set(TEXTS_SOFT) - set(TEXTS_OFF))      # (v6.1's level_start_soft is an older, unrelated text)
check('0 the soft texts', soft_keys, ['attempt_summary_request_noact_soft', 'attempt_summary_request_position_soft',
                                      'attempt_summary_request_soft', 'noact_limit_clause_soft', 'noact_warn_2_soft',
                                      'summary_turn_reason_soft', 'summary_turn_refusal_soft', 'takeover_soft',
                                      'takeover_soft_history', 'takeover_soft_history_cut'])
check('0 the soft switch only adds those texts (every other text identical to the switch off)',
      ({k: v for k, v in TEXTS_SOFT.items() if k not in soft_keys} == TEXTS_OFF, set(TEXTS_OFF) <= set(TEXTS_SOFT)), (True, True))
check('0 the texts the agent swaps have a soft variant',
      [k for k in ('attempt_summary_request', 'attempt_summary_request_noact', 'attempt_summary_request_position',
                   'summary_turn_refusal', 'noact_limit_clause', 'noact_warn_2') if k + '_soft' not in soft_keys], [])
check('0 no soft text claims a RESET, the first board, a level restart or programs set aside',
      [(k, w) for k in soft_keys for w in FALSE_FOR_SOFT if w in TEXTS_SOFT[k]], [])
check('0 the soft refusal starts like the hard one (the notes retry recognises it)',
      TEXTS_SOFT['summary_turn_refusal_soft'][:40] == TEXTS_SOFT['summary_turn_refusal'][:40], True)
try:
    build(ARC3_V67_RESTART_KIND='medium')
    bad = 'no error'
except ValueError as exc:
    bad = str(exc)
check('0 an unknown kind is refused', bad, 'ARC3_V67_RESTART_KIND must be hard or soft')
R = open(Path(A.__file__).with_name('run_v6.py')).read()
check('0 run_v6: explicit defaults, recorded; the earlier anchors kept',
      ("os.environ.setdefault('ARC3_V67_RESTART_KIND', 'hard')" in R, "os.environ.setdefault('ARC3_V67_DEAD_GAMEOVERS', '3')" in R,
       "os.environ.setdefault('ARC3_V67_DEAD_WINDOW', '0')" in R,
       "'ARC3_V67_RESTART_KIND', 'ARC3_V67_DEAD_GAMEOVERS', 'ARC3_V67_DEAD_WINDOW'," in R,
       "'ARC3_V66_SCO', 'ARC3_V66_TEXTS', 'ARC3_V66_SEEDS', 'ARC3_V65_BATCH')}," in R), (True,) * 5)

# === 1. a soft budget restart: no RESET, the board, the programs and the notes kept; the opener's facts ==================
SCRIPT1 = [[(PROG + 'action(["RIGHT"])', 600), ('action(["LEFT"])', 100), (NOTES, 80)],
           [(LOOK + 'action(["RIGHT"])', 50), ('action(["RIGHT"])', 50)]]
game, (sess, sess2), agent, log, go = h(copy.deepcopy(SCRIPT1), budget=500)
go()                                                    # 600 tokens: over the 500 budget
board_at_restart = game.history_entries[-1].frame
go()                                                    # the notes turn: a refused action, the retry, the notes
req = user_text(sess.sent[1])
check('1 the notes request: the soft wording, no RESET / first board / set aside',
      (A._fill(A.TEXTS['attempt_summary_request_soft'], level=1, tokens='600') in req,
       [w for w in FALSE_FOR_SOFT if w in req]), (True, []))
refused = tool_text(sess.sent[2])
check('1 the refused action in the notes turn: the soft refusal and reason',
      (A.TEXTS['summary_turn_refusal_soft'] in refused, A.TEXTS['summary_turn_reason_soft'] in refused,
       'level restarts' in refused, user_text(sess.sent[2]).startswith('No notes were saved (an action call was refused)')),
      (True, True, False, True))
check('1 the notes saved by the retry', agent._v62_attempt_notes.get(1),
      ['- the marker moves right one cell per RIGHT\n- DOWN ends the attempt'])
go()                                                    # the soft restart and the new conversation's first call
go()                                                    # its second call (the first call's output arrives)
first = user_text(sess2.sent[0])
check('1 no RESET executed; the history kept', (game.engine, [e.action for e in game.history_entries]),
      (['RIGHT', 'RIGHT', 'RIGHT'], ['', 'RIGHT', 'RIGHT', 'RIGHT']))
check('1 the old session closed, the new one used', (sess.closed, len(sess2.sent)), (True, 2))
check('1 the takeover shows the current board (the board at the restart)',
      board_url(sess2.sent[0]) == current_grid_image_part(board_at_restart)['image_url']['url'], True)
want1 = A._fill(A.TEXTS['takeover_soft'], level=1, step=1, tokens='780', n=1)
check('1 the takeover: continues level 1 from the current board, 780 tokens, no RESET, 1 action so far', want1 in first, True)
check('1 ... the notes (as after a hard restart), the programs line, the history line',
      (A.TEXTS['attempt_notes_header'] in first, '- the marker moves right one cell per RIGHT' in first,
       '1 program written on this level is still loaded.' in first, A.TEXTS['takeover_soft_history'] in first,
       'Step 1 (actions in the whole run), level 1, attempt 1.' in first), (True,) * 5)
check('1 ... no hard claims: no RESET of the level, first board, programs set aside, hidden attempts',
      [w for w in FALSE_FOR_SOFT + ('not in this conversation', 'begins at the current board') if w in first], [])
look = tool_text(sess2.sent[1])
check('1 the program still loaded (programs() in the new conversation); the start frames not moved',
      ("LIB ['double']" in look, 'START 0 0' in look, sorted(agent._kept_functions), agent._v61_archive, agent._v61_hidden),
      (True, True, ['double'], {}, []))
ev = restarts(log)
check('1 the restart event', [(p['kind'], p['hard_why'], p['actions_on_level'], p['reason'], p['archived'], p['hidden'])
                              for p in ev], [('soft', None, 1, 'tokens', [], None)])
check('1 receipts: soft_restarts 1, no hard_restarts, restarts 1',
      tuple(agent._v6_stats.get(k) for k in ('soft_restarts', 'hard_restarts', 'restarts')), (1, None, 1))
check('1 the game record', game.__dict__.get('v67_level_restarts'),
      {1: [dict(kind='soft', why=None, reason='tokens', step=1, tokens=780, actions_on_level=1, attempts_ended=0)]})

# the hard kind on the same script: today's takeover, unchanged; the soft texts do not touch it
runs = {}
for name, texts in (('hard', TEXTS_SOFT), ('hard_texts_off', TEXTS_OFF)):
    A.TEXTS = texts
    game, (sess, sess2), agent, log, go = h(copy.deepcopy(SCRIPT1), budget=500, kind='hard')
    for _ in range(4):
        go()
    runs[name] = (game, sess, sess2, agent, log)
A.TEXTS = TEXTS_SOFT
game, sess, sess2, agent, log = runs['hard']
first = user_text(sess2.sent[0])
check('1 hard: the RESET, the hard request and refusal', (game.engine,
      A._fill(A.TEXTS['attempt_summary_request'], level=1, tokens='600') in user_text(sess.sent[1]),
      A.TEXTS['summary_turn_refusal'] in tool_text(sess.sent[2])), (['RIGHT', 'RESET', 'RIGHT', 'RIGHT'], True, True))
check('1 hard: the takeover as today (RESET at step 2, the program set aside), no soft line',
      (A._fill(A.TEXTS['takeover_restart_shown'], level=1, step=2) in first,
       '1 program written on this level was set aside at the restart and is not loaded.' in first,
       'continues level' in first, A.TEXTS['takeover_soft_history'] in first), (True, True, False, False))
check('1 hard: the library and the start frames', ("LIB []" in tool_text(sess2.sent[1]), 'START 2 0' in tool_text(sess2.sent[1]),
      sorted(agent._v61_archive)), (True, True, ['double']))
check('1 hard: event and receipts', ([(p['kind'], p['hard_why'], p['actions_on_level']) for p in restarts(log)],
      agent._v6_stats.get('hard_restarts'), agent._v6_stats.get('soft_restarts')), ([('hard', None, 1)], 1, None))
other = runs['hard_texts_off']
check('1 hard: the model traffic is the same with the switch-off texts (the soft texts change nothing for hard)',
      (sess.sent == other[1].sent, sess2.sent == other[2].sent, game.engine == other[0].engine), (True, True, True))

# === 2. attempt accounting with kind soft: the attempt record, the ledger and the start board stay; the attempt goes on ===
SCRIPT2 = [[('action(["RIGHT"])', 100), ('action(["DOWN"])', 100), ('action(["RIGHT", "RIGHT"])', 400), (NOTES, 80)],
           [(LOOK + 'action(["RIGHT"])', 50), ('action(["DOWN"])', 50), ('action(["RIGHT"])', 50)]]


def snapshot(game, agent):
    return dict(records=copy.deepcopy({k: game.__dict__.get(k) for k in REC_KEYS + START_KEYS}),
                ledger=copy.deepcopy((agent._death_ledger_attempts, agent._death_ledger_index, agent._death_ledger_total,
                                      agent._death_ledger_level)),
                hist=len(game.history_entries), attempt=len((game.__dict__.get('v6_level_attempts') or {}).get(1, [])) + 1)


def run2(kind):
    game, sessions, agent, log, go = h(copy.deepcopy(SCRIPT2), budget=500, kind=kind)
    snaps, real = [], agent._v61_restart

    def wrapped(*a, **k):
        snaps.append(snapshot(game, agent))
        out = real(*a, **k)
        snaps.append(snapshot(game, agent))
        return out
    agent._v61_restart = wrapped
    for _ in range(7):
        go()
    return game, sessions, agent, log, snaps


game, (sess, sess2), agent, log, snaps = run2('soft')
before, after = snaps
check('2 the restart had a death on record (ledger 1, attempts [2], 2 actions in the attempt)',
      (before['ledger'][2], before['records']['v6_level_attempts'], before['records']['v6_attempt_actions']), (1, {1: [2]}, 2))
check('2 soft: the attempt record unchanged by the restart (%s)' % ', '.join(REC_KEYS),
      [k for k in REC_KEYS if after['records'][k] != before['records'][k]], [])
check('2 soft: the start-board counters unchanged (%s)' % ', '.join(START_KEYS),
      ([k for k in START_KEYS if after['records'][k] != before['records'][k]], before['records']['v62_since_start'],
       before['records']['v61_level_start'] == board(0)), ([], 2, True))
check('2 soft: the death ledger unchanged; no history entry added', (after['ledger'] == before['ledger'], after['hist'] - before['hist']),
      (True, 0))
first = user_text(sess2.sent[0])
check('2 soft: the takeover: attempt 2 goes on; one earlier attempt; 5 actions counted (2 + the automatic RESET + 2)',
      ('level 1, attempt 2.' in first, 'Before the current attempt, this level had 1 attempt: 1 ended in GAME OVER.' in first,
       '(5 so far)' in first), (True, True, True))
check('2 soft: the sandbox start frames: the attempt still starts at the automatic restart (entry 3)',
      'START 3 0' in tool_text(sess2.sent[1]), True)
check('2 soft: the attempt spans the two conversations (2 + 2 actions before the second GAME OVER)',
      (game.v6_level_attempts, game.v62_level_attempt_ends, game.engine.count('RESET')), ({1: [2, 4]}, {1: ['game_over', 'game_over']}, 0))
check('2 soft: the second death joins the same ledger', agent._death_ledger_total, 2)
check('2 soft: the record', [(x['kind'], x['attempts_ended'], x['actions_on_level']) for x in game.v67_level_restarts[1]],
      [('soft', 1, 5)])
game, (sess, sess2), agent, log, snaps = run2('hard')
before, after = snaps
check('2 hard (contrast): the RESET ends attempt 2 and resets the start counters',
      (after['records']['v6_level_attempts'], after['records']['v62_level_attempt_ends'], before['records']['v62_since_start'],
       after['records']['v62_since_start'], after['hist'] - before['hist']), ({1: [2, 3]}, {1: ['game_over', 'restart']}, 2, 0, 1))
check('2 hard: the death ledger kept as well', after['ledger'] == before['ledger'], True)

# === 3. the position limit under soft ===================================================================================
game, (sess, sess2), agent, log, go = h([[('action(["RIGHT"])', 600), (NOTES, 80)],
                                         [('action(["RIGHT"])', 50), ('action(["RIGHT"])', 50)]], pos_min=500)
go()
sess.last_receipt = {'logical_tokens': A.POSITION_RESTART}
go()                                                    # the notes turn (position wording, soft)
req = user_text(sess.sent[1])
check('3 notes first: the soft position request', (A.TEXTS['attempt_summary_request_position_soft'].split(':')[0] in req,
      'This conversation has reached its length limit, so it ends after this turn' in req, [w for w in FALSE_FOR_SOFT if w in req]),
      (True, True, []))
go()                                                    # the position restart: soft
go()                                                    # the next turn: no second restart (a new session)
first = user_text(sess2.sent[0])
check('3 no RESET; the new session; one restart; the notes and the soft line',
      (game.engine, sess.closed, len(sess2.sent), agent._v6_stats.get('restarts'),
       '- the marker moves right one cell per RIGHT' in first, 'continues level 1 from the current board' in first),
      (['RIGHT', 'RIGHT', 'RIGHT'], True, 2, 1, True, True))
check('3 receipts and event', (agent._v6_stats.get('position_restarts'), agent._v6_stats.get('position_notes_turns'),
      agent._v6_stats.get('soft_restarts'), [(p['reason'], p['kind']) for p in restarts(log)]), (1, 1, 1, [('position', 'soft')]))
game, (sess, sess2), agent, log, go = h([[('action(["RIGHT"])', 600)], [('action(["RIGHT"])', 50)]], pos_min=10 ** 9)
go()
sess.last_receipt = {'logical_tokens': A.POSITION_RESTART}
go()                                                    # an immediate position restart (below the notes minimum)
first = user_text(sess2.sent[0])
check('3 immediate: no notes turn, no RESET, soft, no notes lines',
      ([k for k, _ in log].count('v62-summary-turn'), game.engine, [(p['reason'], p['kind']) for p in restarts(log)],
       A.TEXTS['attempt_notes_header'] in first, A.TEXTS['attempt_notes_none'] in first, 'continues level 1' in first),
      (0, ['RIGHT', 'RIGHT'], [('position', 'soft')], False, False, True))

# === 4. hard under soft =================================================================================================
# (a) the game reports GAME OVER (a RESET that ended in one: no automatic restart follows)
game, (sess, sess2), agent, log, go = h([[(PROG + 'action(["RIGHT"])', 600), (NOTES, 80)], [('action(["RIGHT"])', 50)]],
                                        budget=500)
go()
game.game.current_state.raw.state, game.last_engine_action = arcengine.GameState.GAME_OVER, 'RESET'
go()                                                    # the notes turn: hard (GAME OVER)
req = user_text(sess.sent[1])
go()                                                    # the restart: RESET
first = user_text(sess2.sent[0])
check('4a GAME OVER: the hard request, the RESET, the hard takeover, the program set aside',
      (A._fill(A.TEXTS['attempt_summary_request'], level=1, tokens='600') in req, game.engine.count('RESET'),
       A._fill(A.TEXTS['takeover_restart_shown'], level=1, step=2) in first, 'continues level' in first,
       sorted(agent._v61_archive)), (True, 1, True, False, ['double']))
check('4a receipts and event', (agent._v6_stats.get('hard_restarts'), agent._v6_stats.get('hard_under_soft_game_over'),
      agent._v6_stats.get('soft_restarts'), [(p['kind'], p['hard_why']) for p in restarts(log)]), (1, 1, None, [('hard', 'game_over')]))
# (b) a caller's explicit kind='hard' (a later loop-ladder rung); kind=None without a notes turn decides at the restart (soft)
for kind, want in (('hard', ('hard', 'explicit', 1, 1, None)), (None, ('soft', None, 0, None, 1))):
    game, (sess, sess2), agent, log, go = h([[('action(["RIGHT"])', 100)], [('action(["RIGHT"])', 50)]])
    go()
    agent._step_env_callback = game.step_env
    cf, he = load_runtime_state(game.state_path)
    agent._v61_restart(game.state_path, cf, he, reason='tokens', kind=kind)
    agent._step_env_callback = None
    p = restarts(log)[-1]
    check('4b kind=%s: %s; receipts' % (kind, want[0]), (p['kind'], p['hard_why'], game.engine.count('RESET'),
          agent._v6_stats.get('hard_under_soft_explicit'), agent._v6_stats.get('soft_restarts')), want)
# (c) dead board: GAME OVERs in this conversation from a board identical to the current one
DEATHS = [('action(["DOWN"])', 100), ('action(["DOWN"])', 100), ('action(["DOWN"])', 400), (NOTES, 80)]
for label, script, dead, want in (('3 deaths from the board, threshold 3', DEATHS, 3, ('hard', 'dead_gameovers')),
                                  ('3 deaths, threshold 4', DEATHS, 4, ('soft', None)),
                                  ('3 deaths, rule off (0)', DEATHS, 0, ('soft', None)),
                                  ('3 deaths, then a move to a board none passed through',
                                   DEATHS[:2] + [('action(["DOWN"])', 100), ('action(["RIGHT"])', 300), (NOTES, 80)], 3,
                                   ('soft', None))):
    game, (sess, sess2), agent, log, go = h([list(script), [('action(["RIGHT"])', 50)]], budget=500, dead=dead)
    for _ in range(len(script) + 1):
        go()
    req = user_text(sess.sent[len(script) - 1])
    check('4c %s: %s, stated so in the notes request' % (label, want[0]),
          ([(x['kind'], x['hard_why']) for x in restarts(log)], game.engine.count('RESET (automatic)'), game.engine.count('RESET'),
           'so it ends after this turn' in req, 'so the level restarts after this turn' in req,
           agent._v6_stats.get('hard_under_soft_dead_gameovers')),
          ([want], 3, 1 if want[0] == 'hard' else 0, want[0] == 'soft', want[0] == 'hard', 1 if want[0] == 'hard' else None))
# (d) dead board: no new interior state over the last W executed actions on the level
for label, acts, window, want in (('cycling, window 3', '["RIGHT", "LEFT", "RIGHT", "LEFT"]', 3, ('hard', 'dead_window')),
                                  ('new states, window 3', '["RIGHT", "RIGHT", "RIGHT", "RIGHT"]', 3, ('soft', None)),
                                  ('cycling, window off (0)', '["RIGHT", "LEFT", "RIGHT", "LEFT"]', 0, ('soft', None)),
                                  ('cycling, window 5 > 4 actions', '["RIGHT", "LEFT", "RIGHT", "LEFT"]', 5, ('soft', None))):
    game, (sess, sess2), agent, log, go = h([[('action(%s)' % acts, 600), (NOTES, 80)], [('action(["RIGHT"])', 50)]],
                                            budget=500, window=window)
    for _ in range(3):
        go()
    check('4d %s: %s' % (label, want[0]), ([(x['kind'], x['hard_why']) for x in restarts(log)], game.engine.count('RESET'),
          agent._v6_stats.get('hard_under_soft_dead_window')), ([want], 1 if want[0] == 'hard' else 0, 1 if want[0] == 'hard' else None))

# === 5. the no-action cues state the restart the limit would bring ======================================================
game, (sess,), agent, log, go = h([[('action(["RIGHT"])', 10)]], warn=(100, 200, 300), limit=400)
go()
agent._step_env_callback = game.step_env
cf, he = load_runtime_state(game.state_path)
cues = []
for over in (False, True):
    game.game.current_state.raw.state = arcengine.GameState.GAME_OVER if over else arcengine.GameState.NOT_FINISHED
    for k in (250, 350):
        agent._v62_noact_tokens, agent._v62_noact_warned = k, (1 if k == 250 else 2)
        cues.append(agent._v62_noact_check(he, VALID)[1])
agent._step_env_callback = None
check('5 soft board: the limit clause and the must-act cue say the conversation continues from the current board',
      (A._fill(A.TEXTS['noact_limit_clause_soft'], limit='400') in cues[0], cues[1] == A._fill(A.TEXTS['noact_warn_2_soft'], k='350', limit='400'),
       [w for w in FALSE_FOR_SOFT if w in cues[0] + cues[1]]), (True, True, []))
check('5 GAME OVER (a hard restart would follow): the hard texts', (A._fill(A.TEXTS['noact_limit_clause'], limit='400') in cues[2],
      cues[3] == A._fill(A.TEXTS['noact_warn_2'], k='350', limit='400')), (True, True))

# === 6. a soft restart after a hard one on the same level, lazy history on: the history line names the cut ==============
real_lazy = A._lazy_enabled
A._lazy_enabled = lambda: True
game, (sess, sess2, sess3), agent, log, go = h([[('action(["RIGHT"])', 100)], [('action(["RIGHT"])', 600), (NOTES, 50)],
                                                [('action(["RIGHT"])', 50)]], budget=500)
go()
agent._step_env_callback = game.step_env
cf, he = load_runtime_state(game.state_path)
agent._v61_restart(game.state_path, cf, he, reason='tokens', kind='hard')     # an explicit hard restart first
agent._step_env_callback = None
hidden = list(agent._v61_hidden)
for _ in range(3):
    go()
A._lazy_enabled = real_lazy
first = user_text(sess3.sent[0])
check("6 the hard restart hid the level's entries before its RESET; the soft one hid nothing more",
      (hidden, list(agent._v61_hidden), [(p['kind'], p['hidden']) for p in restarts(log)]),
      ([(0, 2)], [(0, 2)], [('hard', [0, 2]), ('soft', None)]))
check('6 the soft takeover names where `history` begins (the RESET at step 2)',
      (A._fill(A.TEXTS['takeover_soft_history_cut'], what='the game', reset_step=2) in first, A.TEXTS['takeover_soft_history'] in first),
      (True, False))

# === 7. the switch off: byte-identical to the v66g base (subprocess on each tree) =========================================
root = Path(A.__file__).resolve().parents[1]
CHANGED = ('v62/agent.py', 'v62/variants.py', 'v62/run_v6.py')
missing = [f for f in CHANGED if not (root / (f + '.pre-v67')).exists()]
check('7 the v66g base copies (v62/*.pre-v67, written by apply_v67.py) are present', missing, [])
if not missing:
    base = Path(tempfile.mkdtemp()) / 'v66g_base'
    shutil.copytree(root / 'v62', base / 'v62', ignore=shutil.ignore_patterns('__pycache__', '*.pre-*', '*.tmp'))
    for f in CHANGED:
        shutil.copyfile(root / (f + '.pre-v67'), base / f)
    env_off = {k: v for k, v in os.environ.items() if not k.startswith('ARC3_V67_')}

    def run_tree(tree):
        env = dict(env_off, PYTHONPATH='%s:%s' % (root / 'src', tree))
        p = subprocess.run([sys.executable, '-c', IDENT, json.dumps(ENV), FAKES], env=env, cwd=str(tree), capture_output=True,
                           text=True, timeout=1200)
        line = [x for x in p.stdout.splitlines() if x.startswith('IDENT-JSON ')]
        if p.returncode or not line:
            print('identity run failed in %s (rc %s):\n%s' % (tree, p.returncode, p.stderr[-3000:]))
            return None
        return json.loads(line[-1][len('IDENT-JSON '):])
    out_base, out_new = run_tree(base), run_tree(root)
    check('7 both identity runs completed', (out_base is not None, out_new is not None), (True, True))
    if out_base and out_new:
        check('7 texts: the switch off renders v66g\'s texts in %d environments' % len(out_base['texts']),
              [k for k in out_base['texts'] if out_base['texts'][k] != out_new['texts'][k]], [])
        check('7 texts: the agent\'s TEXTS identical', out_base['agent_texts'] == out_new['agent_texts'], True)
        for name in out_base['runs']:
            a, b = out_base['runs'][name], out_new['runs'][name]
            same = [k for k in ('sent', 'engine', 'events', 'stats', 'records', 'library', 'archive', 'hidden') if a[k] == b[k]]
            check('7 %s: model traffic, game actions, events, stats, records identical' % name,
                  [k for k in ('sent', 'engine', 'events', 'stats', 'records', 'library', 'archive', 'hidden') if k not in same], [])
            new = b['v67']
            check('7 %s: the new restart keys: every restart hard (no hard_why); hard_restarts = restarts' % name,
                  (all(x.get('kind') == 'hard' and x.get('hard_why') is None for x in new[:-1]),
                   new[-1]['hard_restarts'] == b['stats'].get('restarts'), a['v67'][-1]['hard_restarts']), (True, True, None))
        check('7 the identity runs restarted (budget 1, noact 1, position 1 + 1, forced 1, twice 2)',
              [out_new['runs'][n]['stats'].get('restarts') for n in ('budget', 'noact', 'position_notes', 'position_now', 'forced',
                                                                    'twice')], [1, 1, 1, 1, 1, 2])
        rn = out_new['runs']
        sent_text = lambda name, i: json.dumps(rn[name]['sent'][i])
        check('7 the identity runs took the intended paths (notes retry and ledger; 3 no-action cues; position notes; '
              'session error; hidden history)',
              (rn['budget']['stats'].get('notes_retries'), 'Continuations that ended in GAME OVER' in sent_text('budget', 1),
               'refused_action_calls' in sent_text('budget', 0),
               [rn['noact']['stats'].get('noact_warnings_%d' % i) for i in (1, 2, 3)], rn['noact']['stats'].get('noact_restarts'),
               rn['position_notes']['stats'].get('position_notes_turns'), rn['position_now']['stats'].get('position_restarts'),
               rn['forced']['stats'].get('protocol_restarts'), 'In `history` and `transitions`' in sent_text('twice', 2)),
              (1, True, True, [1, 1, 1], 1, 1, 1, 1, True))
print('FAILS:', fails)
