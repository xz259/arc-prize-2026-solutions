"""Harness v6.0 agent (spec §2, §5, §6; BUILD_PLAN component 2): dfranzen's ToolAgent (10882e3) with the turn loop
replaced for our append-only persistent sessions and the model-facing texts replaced by the spec's (v6/texts.json).

Kept from his agent (inherited, unmodified): the Python tool (_run_python_tool: sandbox, action(), guards, echo,
retained-function intake, step summaries, death/diff/animation bookkeeping), markup recovery, the degeneracy check.
Replaced:
  analyze()            one turn on the server session: [pending tool result +] opener -> generate; length = pause
                       (continue in place); an acting call ends the turn; yield after >= YIELD_TOKENS generated in
                       the turn without an action (resumption opener appended with the non-acting tool result);
                       recovery messages appended (N1, N3, N4, N5; TURN_BUDGET/REPEAT_RECOVERY after a close);
                       HC1b on a capacity refusal; no trimming, no rollback, no summaries or memory sections.
  _build_user_prompt   the §6.3 opener (summary, trace, status, no-op stop, animation, program list, step line,
                       standing reminder) and its images (8x board first, then the change image; restart: three)
  _tools / system      §6.1 / §6.2
  tool payload         §5.6: JSON, 2,048-token middle-truncated text (his renderer with our cap and marker)
The program-list reminder is scheduled by the exact rule (logical span of the carrying opener's list against the
merged eviction ranges; a rebase that drops or cuts its observation also re-arms it).

v6.1 (HARNESS_REVISION_ITEMS_2026-10-05.md, user go-ahead 2026-10-05):
  1   a python call whose arguments lack `code` is re-parsed from the decoded assistant text; an explicit format error
      when nothing is recovered; after 2 identical tool errors in a row a corrective message, after 3 also a yield
  2   per-call counters reset at every python call (cap, REVERTED run); the opener reports the v6.1 stops
  2h  dfranzen's death ledger recorded and its lines shown (his opener did that; ours replaces his)
  3   attempt in the step line; attempt lengths with no-ops at restarts; the edge band of the previous call
  4   the restart block says whether the fatal action changed the interior
  10  run-clock line; `score` shown as `levels_completed`
  11  after 8 python calls in a row without an action: a line with the actions not yet executed on this level
  12  no program list; library entries carry their level; a count line at each new level
  6   the level-completion mechanics summary is asked for and captured in both arms; run B
      (ARC3_V61_RESTART_TOKENS > 0) restarts a level after that many generated tokens on it: RESET, a new server
      session, the system prompt, the level's first board and the summaries; the level's programs are archived; the
      death ledger, the attempt counter and the no-action counter start again from the restart (user: restart fresh)
"""
from __future__ import annotations

import json
import os
import re
import time
import zlib
from pathlib import Path
from typing import Any

from inference.agent import tool_agent as ta
from inference.agent.runtime_state import load_runtime_state
from inference.agent.vision_context import current_grid_image_part, diff_image_part, frame_to_png_data_url
from inference.utils.animation import _bbox_phrase, _transient_cells
from inference.utils import retained_functions as _rf


def function_signature(source):
    """Resolved at call time: v61.retention makes it constant-aware after this module is imported."""
    return _rf.function_signature(source)

from v61 import sandbox_ext
from v61 import tool_output
from v61.transport import (CapacityRefusal, TURN_BUDGET_RECOVERY, TURN_REPEAT_RECOVERY)

from v61 import variants

TEXTS = variants.apply(json.load(open(Path(__file__).with_name('texts.json')))['texts'])
TOOL = {"type": "function", "function": {
    "name": "python",
    "description": ("Run Python code against the live game. Inspect the runtime state, compute, and execute real game "
                    "actions with action(...)."),
    "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python source to run."}},
                   "required": ["code"]}}}
YIELD_TOKENS = int(os.environ.get("ARC3_V6_YIELD_TOKENS", "4096"))   # 0 = off; default 4,096 since 2026-10-03 ~16:42 (user)
TOOL_TEXT_TOKENS = int(os.environ.get("ARC3_V6_TOOL_TEXT_TOKENS", "2048"))
# B2 animation push (spec §8; user lean: on): after an acting call, up to ANIM_MAX frames of the call's last animated
# action at 4x, only when that action has transient cells inside the board area, attached to the acting call's tool
# result. Captions "Animation, action event ..." put them in the server's 'animation' image kind (tool segment), the
# kind the control's frames had and the per-kind eviction rank was tuned with.
ANIM_PUSH = os.environ.get("ARC3_V6_ANIM_PUSH", "0") == "1"
# Recap (old harness v5.29, ARC3_V6_RECAP=1): when the opener that carried the last recap has left the context and newer
# boards have left it too, the next opener re-shows the RECAP_COUNT most recent boards whose opener's board image is no
# longer resident (server receipt: history_state.blocks[].board_resident; a rebase that dropped the observation counts),
# oldest first, at 4x, each captioned with step, level, attempt and the actions executed after it.
RECAP_COUNT, RECAP_SCALE = 10, 4
ANIM_MAX, ANIM_SCALE, ANIM_BORDER = 6, 4, 4       # border = the solver's NOOP_GUARD_BORDER_WIDTH (4-cell edge band)
PLAN_WORDS = re.compile(r"\b(call|run|execute|try|probe|act|press|move|click|let me|i will|i'll|now)\b", re.I)
RESTART_TOKENS = int(os.environ.get("ARC3_V61_RESTART_TOKENS", "0"))   # v6.1 item 6 (run B); 0 = off (run A)
NO_ACTION_CALLS = int(os.environ.get("ARC3_V61_NO_ACTION_CALLS", "8"))  # v6.1 item 11
SUMMARY_CHARS = 900                                                     # v6.1 item 6: captured summary length
V61_STOPS = ('reverted', 'level_restarted', 'level_start', 'call_cap')
_FORMAT_ERROR = ("Your python call had no `code` argument, so nothing ran. The call must contain the parameter tag "
                 "<parameter=code> before your code and </parameter> after it, inside <function=python>.")
_BREAKER = ("The last {k} python calls returned the same error: {error} Do not repeat that call. Emit one python "
            "tool call in the required format: <tool_call>, <function=python>, <parameter=code>, your code, "
            "</parameter>, </function>, </tool_call>.")


_PARAM_TAG = re.compile(r'^\s*<parameter=[^>\n]*>\n?')


def recover_code(raw):
    """v6.1 item 1: the code of the last python tool call in the decoded assistant text, when the server parser
    returned no `code`: a missing <parameter=code> opening tag, a wrong parameter name (<parameter=python>,
    <parameter=command>), or a body left unclosed before <|im_end|>. Returns None when nothing usable remains."""
    if not raw:
        return None
    text = raw.rsplit('</think>', 1)[-1]
    at = text.rfind('<tool_call>')
    if at < 0:
        return None
    block = text[at + len('<tool_call>'):]
    for stop in ('</tool_call>', '<|im_end|>'):
        end = block.find(stop)
        if end >= 0:
            block = block[:end]
    fn = block.find('<function=python>')
    if fn < 0:
        return None
    body = block[fn + len('<function=python>'):]
    end = body.rfind('</function>')
    if end >= 0:
        body = body[:end]
    body = _PARAM_TAG.sub('', body.lstrip('\n'), count=1)
    body = body.rstrip()
    if body.endswith('</parameter>'):
        body = body[:-len('</parameter>')]
    code = body.strip('\n')
    return code if code.strip() else None


def code_from_arguments(parsed):
    """v6.1 item 1: a python call whose single argument came under another name ({"command": "..."})."""
    if not isinstance(parsed, dict) or str(parsed.get('code') or '').strip():
        return None
    values = [v for v in parsed.values() if isinstance(v, str) and v.strip()]
    return values[0] if len(values) == 1 else None


def _levels_text(levels):
    """[1, 2, 3, 5] -> '1-3 and 5'"""
    vals = sorted({int(v) for v in levels if isinstance(v, int)})
    if not vals:
        return '?'
    runs, start, prev = [], vals[0], vals[0]
    for v in vals[1:] + [None]:
        if v is not None and v == prev + 1:
            prev = v
            continue
        runs.append(str(start) if start == prev else '%d-%d' % (start, prev))
        if v is not None:
            start = prev = v
    return runs[0] if len(runs) == 1 else ', '.join(runs[:-1]) + ' and ' + runs[-1]


def _edge_text(actions):
    """v6.1 item 3: the edge band over one call's executed actions (exact facts; the model fits the rate)."""
    rows = [a for a in actions if not a.get('automatic')]
    changed = [a for a in rows if a.get('v61_edge')]
    if not changed:
        return ''
    total = sum(a['v61_edge']['n'] for a in changed)
    tr = {}
    rs, cs = set(), set()
    for a in changed:
        e = a['v61_edge']
        for f, t, k in e.get('tr') or []:
            tr[(f, t)] = tr.get((f, t), 0) + k
        rs.update(e.get('rows') or [])
        cs.update(e.get('cols') or [])
    top = sorted(tr.items(), key=lambda kv: -kv[1])[:2]
    trs = ', '.join('%s->%s x%d' % (f, t, k) for (f, t), k in top)
    where = ('row %s' % sorted(rs)[0]) if len(rs) == 1 else ('col %s' % sorted(cs)[0]) if len(cs) == 1 else 'several rows'
    noop = sum(1 for a in changed if a.get('gameplay_changed') is False and not a.get('reverted'))
    text = '%d of %d executed actions changed %d edge cells (%s; %s)' % (len(changed), len(rows), total, trs, where)
    if noop:
        text += ('; %d of them %s NO-OP%s inside the board' % (noop, 'was a' if noop == 1 else 'were', '' if noop == 1 else 's'))
    return text


def _fill(template, **kw):
    out = template
    for k, v in kw.items():
        out = out.replace('{' + k + '}', str(v))
    return out


_RET_SAVED = re.compile(r'^Retained your function ([A-Za-z_]\w*)\((.*)\) for later Python calls [^.]*\.$')
_RET_REJECTED = re.compile(r'^Your function (\S+) was not retained: (.*?)\.(?: (.*))?$')
_RET_REMOVED = re.compile(r'^Your previous function (\S+) is no longer retained; redefine it if needed\.$')
_RET_CLEARED = re.compile(r'^Your retained functions were cleared: (.*)\.$')


def _v6_retention_notes(text):
    """dfranzen's retention notes in the spec's program-library wording (2026-10-03, user: wire the spec texts)."""
    out = []
    for line in str(text).split('\n'):
        m = _RET_SAVED.match(line)
        if m:
            if m.group(2) == 'constant':                     # v6.1 item 8 (retention.py reports NAME(constant))
                out.append('Saved the constant `%s` to the program library.' % m.group(1))
            else:
                out.append(_fill(TEXTS['retention_saved'], name=m.group(1), params=m.group(2)))
            continue
        m = _RET_REJECTED.match(line)
        if m:
            out.append(_fill(TEXTS['retention_rejected'], name=m.group(1), reason=m.group(2), hint=m.group(3) or '').rstrip())
            continue
        m = _RET_REMOVED.match(line)
        if m:
            out.append(_fill(TEXTS['retention_removed'], name=m.group(1),
                             reason='it was no longer defined after this call; redefine it if needed'))
            continue
        m = _RET_CLEARED.match(line)
        if m:
            out.append('Your program library was cleared: %s.' % m.group(1))
            continue
        if line.startswith('Your retained functions were cleared after a retention error'):
            out.append('Your program library was cleared after a retention error; redefine any programs you need.')
            continue
        out.append(line)
    return '\n'.join(out)


def _v6_valid_display(valid_actions):
    """The valid actions as listed to the model: RESET left out (user 2026-10-03 ~19:45: the system prompt says it is an
    option; listing it in every observation would encourage it). The executable set is unchanged."""
    return ', '.join(a for a in ta._normalize_valid_actions(valid_actions) if a != 'RESET') or 'none'


def _v6_error_text(err):
    """The spec's timeout text for the sandbox's bare timeout error (it may already have executed actions)."""
    if isinstance(err, str) and err.startswith('Tool timed out after'):
        return TEXTS['timeout']
    return err


def _docline(source):
    m = re.search(r'^\s*def [^\n]*\n\s*(?:"""|\'\'\')(.*?)(?:"""|\'\'\'|\n)', source, re.S)
    return (m.group(1).strip().split('\n')[0] if m else '').strip()


_original_history_view = ta._ascii_history_view_payload


def _history_view_v61(history_entries):
    """item 10: history results name `score` levels_completed, as last_action_call_result does."""
    payload = _original_history_view(history_entries)
    for item in payload:
        r = item.get('result')
        if isinstance(r, dict) and 'score' in r:
            r['levels_completed'] = r.pop('score')
    return payload


ta._ascii_history_view_payload = _history_view_v61


_NOOP_LAST = "It is the last action in the executed list above; only the actions before it moved anything."
_NOOP_NOT_LAST = ("It is executed action %d of %d in this call; the actions after it came from later action(...) "
                  "calls in the same snippet.")
assert _NOOP_LAST in TEXTS['noop_stop'], 'noop_stop anchor'


class V6Agent(ta.ToolAgent):
    """One game's agent. `session` is a v6.transport.PersistentSession owned by this game; `logger` writes the
    raw-harness evidence (kind, payload)."""

    def __init__(self, *, session, logger, count_tokens, session_factory=None, **kwargs):
        super().__init__(**kwargs)
        self._system_prompt = TEXTS['system']
        self._v6_session = session
        self._v61_session_factory = session_factory    # run B: a fresh server session per restart
        self._v61_library_levels: dict[str, int] = {}   # item 12: level each program was last written on
        self._v61_archive: dict[str, dict] = {}         # item 6: programs of a restarted level, by name
        self._v61_level = None                          # level of the latest opener
        self._v61_level_tokens = 0                      # item 6: tokens generated on the level since its start / restart
        self._v61_tokens_level = None
        self._v61_noact_calls = 0                       # item 11: python calls in a row without an action
        self._v61_noact_level_from = None               # history index the untried-actions scan starts at
        self._v61_last_error = None                     # item 1: breaker
        self._v61_error_streak = 0
        self._v61_restarts = 0
        self._v6_log = logger
        self._v6_count = count_tokens            # text -> token count (served tokenizer)
        self._v6_pending_tools: list[dict[str, Any]] = []
        self._v6_started = False
        self._v6_prev_opener_grid = None
        self._v6_prev_opener_level = None
        self._v6_resume = False
        self._v6_library_order: list[str] = []   # names, least recently added/changed first
        self._v6_library_dirty = False
        self._v6_reminder = None                 # dict(obs_index, offset, length) of the carrying opener's list
        self._v6_obs_count = 0
        self._v6_pending_list = None
        self._v6_library_steps: dict[str, int] = {}
        self._v6_tool_images = []
        self._v6_rules: dict[int, str] = {}       # A4: level -> recorded rule text
        self._v6_obs_records = {}                 # recap: obs index -> step, level, attempt, history length, board
        self._v6_recap_shown = None               # recap: dict(turn, last) of the latest recap
        self._v6_push_note = False                # B2: frames of the last executed action were pushed this turn
        self._v6_rules_dirty = False
        self._v6_rules_reminder = None            # carrier of the rules block (as _v6_reminder)
        self._v6_rule_capture = None              # level whose rule the next completion may state
        self._v6_pending_rules = None
        self._v6_stats = dict(turns=0, requests=0, yields=0, nudges={}, closes=0, capacity_refusals=0,
                              reminder_appends=0, completions=0, continuations=0)

    # ------------------------------------------------------------------------------------------ texts and tools
    def _tools(self, state_path):
        return [TOOL]

    def _v6_carrier_evicted(self, r):
        """Exact rule: a carried block is gone when any of its logical tokens was evicted (current frame) or a rebase
        dropped or cut its observation. Unresolvable (no block for the observation yet): not evicted."""
        obs_id = f"obs-{r['obs_index']:08d}"
        if obs_id in self._v6_session.rebased_away:
            return True
        start = None
        for b in self._v6_session.blocks:
            if isinstance(b, dict) and b.get('observation_id') == obs_id and type(b.get('start')) is int:
                start = b['start']
        if start is None:
            return False
        a = start + r['offset']
        return self._v6_session.span_evicted(a, a + r['length']) > 0

    def _v6_rules_due(self):
        if not (variants.a45() and self._v6_rules):
            return False
        if self._v6_rules_dirty or self._v6_rules_reminder is None:
            return True
        return self._v6_carrier_evicted(self._v6_rules_reminder)

    def _v6_rules_block(self):
        lines = [TEXTS['rules_block']]
        for lvl in sorted(self._v6_rules)[-5:]:            # A4: the 5 most recent levels (HD review)
            lines.append(f"- L{lvl}: {self._v6_rules[lvl]}")
        return '\n'.join(lines)

    def _v6_reminder_due(self):
        """v6.1 item 12 (user 2026-10-05): no program list in openers; programs() lists the library with levels and a
        count line opens each new level."""
        return False

    def _v61_library_count_line(self):
        names = [n for n in self._v6_library_order if n in self._kept_functions]
        if not names:
            return ''
        levels = [self._v61_library_levels.get(n) for n in names]
        return _fill(TEXTS['library_count'], n=len(names), levels=_levels_text([l for l in levels if l is not None]))

    def _v6_program_list(self):
        names = [n for n in self._v6_library_order if n in self._kept_functions]
        total = len(names)
        shown = names[-10:]
        lines = [_fill(TEXTS['program_list'].split('\n')[0], total=total).replace('{min(10,total)}', str(min(10, total)))]
        for n in shown:
            src = self._kept_functions[n]
            try:
                sig = function_signature(src)
            except Exception:  # noqa: BLE001
                sig = n + '(...)'
            doc = _docline(src)
            lines.append(f"- {sig}: {doc}" if doc else f"- {sig}")
        if total > 10:
            lines.append(_fill(TEXTS['program_list'].split('\n')[-1], **{'total-10': total - 10}))
        return '\n'.join(lines)

    def _v6_summary_lines(self, s):
        names = [str(a) for a in (s.get('executed_actions') or [])]
        n = s.get('executed_count') or len(names)
        tmpl = TEXTS['summary_trace'].split('\n')
        lines = [tmpl[0].replace('{n}', str(n)).replace('{a1}, {a2}, …', ', '.join(names))]
        trace = list(s.get('game_over_trace') or [])
        if not trace:
            for stop in (s.get('no_op_stops') or []) + (s.get('guard_stops') or []):
                trace = list(stop.get('trace') or [])
                if trace:
                    break
        if not trace and n and n > 1:
            trace = self._turn_action_trace([dict(executed_actions=names, gameplay_changed=s.get('gameplay_changed'))])
        if trace and variants.a13():                     # A2 everywhere the model reads a verdict (HD review)
            session = getattr(self._step_env_callback, '__self__', None)
            ended = len(self._v61_attempts(session, s.get('level'))) or '?'
            trace = [t.replace('GAME OVER (attempt ended immediately after this action)',
                               'LEVEL RESTARTED (attempt %s ended here)' % ended) for t in trace]
        if trace:
            lines.append('Per action: ' + '; '.join(trace) + '.')
        return lines

    # ------------------------------------------------------------------------------------------ v6.1 opener parts
    def _v61_session(self):
        return getattr(self._step_env_callback, '__self__', None)

    def _v61_attempts(self, session, level):
        """[(actions, no-ops)] of this level's ended attempts, counted from the latest restart (decision 5)."""
        lengths = list((getattr(session, 'v6_level_attempts', None) or {}).get(level, []))
        noops = list((getattr(session, 'v61_level_attempt_noops', None) or {}).get(level, []))
        noops += [0] * (len(lengths) - len(noops))
        offset = (getattr(session, '__dict__', {}).get('v61_attempt_offset') or {}).get(level, 0)
        return list(zip(lengths, noops))[offset:]

    @staticmethod
    def _v61_lengths_text(attempts):
        return ', '.join('%d actions (%d no-op%s)' % (n, m, '' if m == 1 else 's') for n, m in attempts[-5:]) or '?'

    @staticmethod
    def _v61_fatal_interior(history_entries):
        entries = list(history_entries or [])
        r = (getattr(entries[-2], 'result', None) or {}) if len(entries) >= 2 else {}
        changed = r.get('gameplay_changed')
        return 'changed' if changed is True else 'did not change' if changed is False else 'may have changed'

    def _v61_clock_line(self, session=None):
        session = session or self._v61_session()
        try:
            remaining = session.timing_payload().get('time_remaining_seconds')
        except Exception:  # noqa: BLE001
            remaining = None
        if remaining is None:
            return ''
        return _fill(TEXTS['run_clock'], minutes=int(round(float(remaining) / 60.0)))

    def _v61_step_line(self, step, level, valid, session):
        k = len(self._v61_attempts(session, level)) + 1
        return f"Step {step}, level {level}, attempt {k}. Valid actions: {valid}."

    def _v61_stop_line(self, stop, session, level):
        kind = stop.get('kind')
        executed = [str(a) for a in (stop.get('executed') or [])]
        skipped = [str(a) for a in (stop.get('skipped') or [])]
        common = dict(m=len(skipped), skipped=', '.join(skipped) or 'none',
                      skipped_part=(' The remaining %d action(s) were skipped: %s.' % (len(skipped), ', '.join(skipped))
                                    if skipped else ''))
        pos, total = stop.get('pos'), stop.get('total')
        if pos and total and pos < total:             # a later action(...) call in the same snippet continued
            common['skipped_part'] += (' This stop came at executed action %d of %d; the actions after it came from '
                                       'later action(...) calls in the same snippet.' % (pos, total))
        if kind == 'reverted':
            a = executed[-2:] if len(executed) >= 2 else (['?'] + executed)[-2:]
            return _fill(TEXTS['revert_stop'], a1=a[0], a2=a[1], **common)
        if kind == 'level_restarted':
            attempts = self._v61_attempts(session, level)
            return _fill(TEXTS['hazard_restart'], action=executed[-1] if executed else '?',
                         n=attempts[-1][0] if attempts else '?', **common)
        if kind == 'level_start':
            return _fill(TEXTS['level_start_soft'], action=executed[-1] if executed else '?', **common)
        if kind == 'call_cap':
            return _fill(TEXTS['call_cap'], **common)
        return ''

    def _v61_untried(self, history_entries, valid_actions):
        """Valid actions (by type) not executed on this level since its start or the latest restart."""
        entries = list(history_entries or [])
        start = 0
        for i, e in enumerate(entries):
            if (getattr(e, 'result', None) or {}).get('level_completed'):
                start = i + 1
        if self._v61_noact_level_from is not None:
            start = max(start, self._v61_noact_level_from)
        done = {m.group(0) for e in entries[start:] for m in [re.match(r'[A-Z]+', str(getattr(e, 'action', '') or ''))] if m}
        valid = [a.strip() for a in _v6_valid_display(valid_actions).split(',') if a.strip() and a.strip() != 'none']
        return [a for a in valid if a not in done]

    def _build_user_prompt(self, action_num, *, valid_actions=None, current_frame=None, history_entries=None,
                           previous_step_summary=None, **_):
        """The §6.3 opener text (images are added by _v6_opener). Returns (text, kind) with kind in
        first | takeover | same | levelup | restart."""
        valid = _v6_valid_display(valid_actions)
        step = getattr(current_frame, 'step', action_num)
        level = getattr(current_frame, 'level', 1)
        self._v61_level = level
        session = self._v61_session()
        reminder = '' if variants.lean() else TEXTS['reminder']
        s = previous_step_summary if (previous_step_summary and not previous_step_summary.get('stale')) else None
        try:                                    # 2h: his ledger records a death when the previous sequence ended in one
            self._death_ledger_observe(history_entries, previous_step_summary)
        except Exception:  # noqa: BLE001
            pass
        clock = self._v61_clock_line(session)
        if not self._v6_started:
            if getattr(self, '_v61_takeover', False):
                return self._v61_takeover_text(level, step, valid, reminder, clock, session), 'takeover'
            text = _fill(TEXTS['first_turn'], valid_actions=valid).replace('{STANDING REMINDER}', reminder)
            lines = [l for l in text.split('\n') if l.strip() and not l.startswith('[image]')]
            if clock:
                lines.insert(1, clock)
            return '\n'.join(lines), 'first'
        parts = []
        kind = 'same'
        if s:
            parts += self._v6_summary_lines(s)
            manual = ta.ToolAgent._manual_reset_context_line(s)     # EXPOSE_RESET only (dfranzen's line)
            if manual:
                parts.append(manual)
            if s.get('game_over'):
                kind = 'restart'
                attempts = self._v61_attempts(session, level)
                body = _fill(TEXTS['restart'], fatal_action=s.get('fatal_action') or 'the last executed action',
                             level=level, next_attempt=len(attempts) + 1, k=len(attempts),
                             lengths=self._v61_lengths_text(attempts),
                             fatal_interior=self._v61_fatal_interior(history_entries))
                if len(attempts) > 5:                       # A3 (HD review): the list is capped at the last 5
                    body = body.replace('; the attempts lasted ', '; the last 5 attempts lasted ')
                parts.append('\n'.join(l for l in body.split('\n') if not l.startswith('[image]')))
            elif s.get('level_transition'):
                kind = 'levelup'
                if variants.summary() or variants.a45():
                    self._v6_rule_capture = int(level) - 1
                parts.append(TEXTS['levelup'].replace('{level-1}', str(int(level) - 1)).replace('{level}', str(level)))
                count = self._v61_library_count_line()
                if count:
                    parts.append(count)
            else:
                parts.append(f"You are still on level {level}.")
            for stop in (s.get('no_op_stops') or [])[:1]:
                skipped = [str(x) for x in (stop.get('skipped') or [])]
                idx = stop.get('index')                   # 1-based position in its batch (v6 added 1 again)
                line = _fill(TEXTS['noop_stop'], action=stop.get('action'), p=idx if isinstance(idx, int) else '?',
                             n=len(skipped) + (idx if isinstance(idx, int) else 1), m=len(skipped),
                             skipped=', '.join(skipped) or 'none')
                pos, total = stop.get('pos'), stop.get('total')
                if pos and total and pos < total:         # a later action(...) call in the same snippet continued
                    line = line.replace(_NOOP_LAST, _NOOP_NOT_LAST % (pos, total))
                parts.append(line)
            for stop in (s.get('v61_stops') or [])[:1]:
                line = self._v61_stop_line(stop, session, level)
                if line:
                    parts.append(line)
            edge = _edge_text(s.get('v61_actions') or [])
            if edge:
                parts.append(_fill(TEXTS['edge_line'], text=edge))
            anim = s.get('animation') or {}
            if anim.get('transient_pixels'):
                bbox = anim.get('transient_bbox')
                where = _bbox_phrase(bbox)
                lines = TEXTS['animation'].split('\n')
                pick = 2 if s.get('game_over') else (1 if s.get('gameplay_changed') is False else 0)
                line = _fill(lines[pick], action=s.get('animation_action') or '?', frames=anim.get('frames'),
                             k=anim.get('transient_pixels'), where=where)
                if self._v6_push_note:                       # B2: the frames of this action were pushed (HD review)
                    line += ' Its frames are shown above.'
                parts.append(line)
        else:
            parts.append(f"You are still on level {level}.")
        self._v6_pending_list = None                        # item 12: no program list
        if self._v6_rules_due():
            rb = self._v6_rules_block()
            parts.append(rb)
            self._v6_pending_rules = rb
        else:
            self._v6_pending_rules = None
        if ta._get_env_bool("ARC3_DEATH_LEDGER", False):    # 2h: fatal continuations from this exact board
            try:
                parts += self._death_ledger_lines(current_frame, history_entries, previous_step_summary)
            except Exception:  # noqa: BLE001
                pass
        parts.append(self._v61_step_line(step, level, valid, session))
        parts.append(clock)
        parts.append(reminder)
        self._v6_push_note = False
        return '\n'.join(p for p in parts if p), kind

    def _v61_takeover_text(self, level, step, valid, reminder, clock, session):
        """Run B restart: the first observation of the new conversation (no restart notice; user decision)."""
        parts = [_fill(TEXTS['takeover'], level=level, prev=int(level) - 1) if int(level) > 1 else TEXTS['takeover_l1']]
        rules = [(lvl, r) for lvl, r in sorted(self._v6_rules.items()) if lvl < int(level)][-5:]
        if rules:
            parts.append(TEXTS['restart_summaries'])
            parts += [f"- L{lvl}: {r}" for lvl, r in rules]
        count = self._v61_library_count_line()
        if count:
            parts.append(count)
        parts.append(self._v61_step_line(step, level, valid, session))
        parts.append(clock)
        parts.append(reminder)
        return '\n'.join(p for p in parts if p)

    def _v6_resumption_text(self, current_frame, valid_actions, history_entries=None):
        valid = _v6_valid_display(valid_actions)
        step, level = getattr(current_frame, 'step', '?'), getattr(current_frame, 'level', '?')
        body = _fill(TEXTS['resumption'], step=step, level=level,
                     valid_actions=valid).replace('{STANDING REMINDER}', '' if variants.lean() else TEXTS['reminder'])
        body = body.replace('{; the latest sequence executed this turn was {seq}}', '')
        lines = [l for l in body.split('\n') if l.strip() and not l.startswith('[image]')]
        session = self._v61_session()
        extra = []
        if self._v61_noact_calls >= NO_ACTION_CALLS:              # item 11
            untried = self._v61_untried(history_entries, valid_actions)
            extra.append(_fill(TEXTS['no_action'], k=self._v61_noact_calls,
                               untried=', '.join(untried) if untried else 'none: every valid action has been tried'))
            self._v6_stats['no_action_lines'] = self._v6_stats.get('no_action_lines', 0) + 1
            self._v6_log('v61-no-action-line', dict(calls=self._v61_noact_calls, untried=untried))
        out = []
        for l in lines:
            if l.startswith('Step ') and ', level ' in l and 'Valid actions:' in l and isinstance(level, int):
                out += extra
                out.append(self._v61_step_line(step, level, valid, session))
                clock = self._v61_clock_line(session)
                if clock:
                    out.append(clock)
            else:
                out.append(l)
        return '\n'.join(out)

    def _v6_opener(self, text, kind, current_frame, history_entries):
        """The user message: text, then images (board first)."""
        content = [dict(type='text', text=text)]
        board = current_grid_image_part(current_frame)
        if kind == 'restart':
            captions = [l[len('[image] '):] for l in TEXTS['restart'].split('\n') if l.startswith('[image]')]
            death = history_entries[-2].frame if len(history_entries) >= 2 else None
            before = history_entries[-3].frame if len(history_entries) >= 3 else None
            imgs = [board, current_grid_image_part(death) if death is not None else None,
                    diff_image_part(before, death) if (before is not None and death is not None) else None]
            for cap, img in zip(captions, imgs):
                if img is not None:
                    content += [dict(type='text', text=cap), img]
        else:
            content += [dict(type='text', text='Current board.'), board]
            prev = self._v6_prev_opener_grid
            changed = prev is not None and prev != current_frame.grid
            if kind not in ('first', 'levelup', 'takeover') and changed:
                from inference.agent.runtime_state import Frame
                d = diff_image_part(Frame(grid=prev, step=0, level=0), current_frame)
                if d is not None:
                    content += [dict(type='text', text=('Change since your previous turn: changed cells in their new '
                                                        'colour, everything else dark navy.')), d]
        self._v6_prev_opener_grid = current_frame.grid
        self._v6_prev_opener_level = current_frame.level
        return {"role": "user", "content": content, "_arc3_observation": True}

    # ----------------------------------------------------------------------------------------------- tool output
    def _trim_tool_text(self, text):
        """§5.6: at most TOOL_TEXT_TOKENS tokens, middle-truncated (first 2/3, last 1/3), with the §6.4 marker."""
        n = self._v6_count(text)
        if n <= TOOL_TEXT_TOKENS:
            return text, False
        lines = text.split('\n')
        head, tail = [], []
        budget_head, budget_tail = int(TOOL_TEXT_TOKENS * 2 / 3), TOOL_TEXT_TOKENS // 3 - 64
        used = 0
        for line in lines:
            c = self._v6_count(line) + 1
            if used + c > budget_head:
                break
            head.append(line)
            used += c
        used = 0
        for line in reversed(lines[len(head):]):
            c = self._v6_count(line) + 1
            if used + c > budget_tail:
                break
            tail.insert(0, line)
            used += c
        k = len(lines) - len(head) - len(tail)
        omitted = n - self._v6_count('\n'.join(head + tail))
        marker = _fill(TEXTS['truncation_marker'], k=k, t=max(0, omitted))
        return '\n'.join(head + [marker] + tail), True

    def _run_python_tool(self, state_path, arguments):
        sandbox_ext.CURRENT.agent = self
        game = self._v61_session()
        if game is not None:                          # items 2b / 2d / 3: per-call counters and details
            game.__dict__['v61_call_executed'] = 0
            game.__dict__['v61_call_revert_run'] = 0
            game.__dict__['v61_call_payloads'] = []
        try:
            return super()._run_python_tool(state_path, arguments)
        finally:
            sandbox_ext.CURRENT.agent = None

    def _summarize_step_sequence(self, action_results):
        summary = super()._summarize_step_sequence(action_results)
        if summary is None:
            return None
        game = self._v61_session()
        summary['v61_actions'] = list((getattr(game, '__dict__', {}) or {}).get('v61_call_payloads') or [])
        stops, noop_pos, done = [], [], 0
        for item in action_results or []:
            executed = item.get('executed_actions') or []
            done += len(executed)
            if item.get('stop_reason') in V61_STOPS:
                stops.append(dict(kind=item.get('stop_reason'), executed=executed, skipped=item.get('skipped_actions') or [],
                                  detail=item.get('stop_detail'), pos=done))
            elif item.get('stop_reason') == 'no_op_action' and executed:
                noop_pos.append(done)
        for stop in stops:                            # where the stop fell among this call's executed actions
            stop['total'] = done
        summary['v61_stops'] = stops
        noop_stops = summary.get('no_op_stops') or []
        if len(noop_stops) == len(noop_pos):
            for stop, pos in zip(noop_stops, noop_pos):
                stop['pos'], stop['total'] = pos, done
        stops = self._v6_stats.setdefault('stops', {})            # measurement: stops by kind
        for item in action_results or []:
            reason = item.get('stop_reason')
            if reason:
                stops[reason] = stops.get(reason, 0) + 1
        d = getattr(game, '__dict__', {}) or {}
        for key in ('v61_voluntary_resets', 'v61_harness_resets'):
            if key in d:
                self._v6_stats[key[4:]] = d[key]
        return summary

    def _compact_action_result(self, payload):
        compact = super()._compact_action_result(payload)
        if 'reverted' in payload:                     # 2b: read by the sandbox's stale-arm rule
            compact['reverted'] = bool(payload['reverted'])
        compact.setdefault('executed_actions', [])    # 2g: stable keys
        compact.setdefault('action_trace', list(payload.get('action_trace') or []))
        compact.setdefault('gameplay_changed_per_action', list(payload.get('gameplay_changed_per_action') or []))
        compact['stop_reason'] = payload.get('stop_reason')
        if 'score' in compact:                        # item 10: `score` is the number of completed levels
            compact['levels_completed'] = compact.pop('score')
        return compact

    def _v6_library_payload(self):
        out = []
        for name in self._v6_library_order:
            src = self._kept_functions.get(name)
            if src is None:
                continue
            try:
                sig = function_signature(src)
            except Exception:  # noqa: BLE001
                sig = name + '(...)'
            out.append(dict(name=name, signature=sig, doc=_docline(src), step=self._v6_library_steps.get(name),
                            level=self._v61_library_levels.get(name), source=src))
        return out

    def _v61_archive_payload(self):
        return [dict(name=n, signature=e.get('signature'), doc=e.get('doc'), level=e.get('level'), step=e.get('step'),
                     source=e.get('source'), archived=True) for n, e in self._v61_archive.items()]

    def _v6_history_frames(self, index):
        session = getattr(self._step_env_callback, '__self__', None)
        return sandbox_ext.host_history_frames(session, index) if session is not None else []

    def _render_tool_payload(self, payload, *, truncate_fields=()):
        """§5.6: printed full boards -> 4x images (deduplicated, <= 4), then his JSON rendering with our text cap."""
        state = dict(seen={}, budget=tool_output.MAX_IMAGES, omitted=0)
        captions = dict(board=TEXTS['board_caption'], same=TEXTS['board_same'])
        images = []
        payload = dict(payload)
        if 'error' in payload:
            payload['error'] = _v6_error_text(payload['error'])
        for f in truncate_fields:
            v = payload.get(f)
            if isinstance(v, str) and v:
                new, imgs = tool_output.convert(v, self._v6_count, captions, state)
                payload[f] = new
                images += imgs
        if state['omitted']:
            payload['boards_note'] = TEXTS['board_more'].replace('{m}', str(state['omitted']))
        self._v6_tool_images = images
        return super()._render_tool_payload(payload, truncate_fields=truncate_fields)

    def _record_retained_functions(self, sandbox_result, payload):
        if 'keepable_functions' not in (sandbox_result or {}):
            # The sandbox did not finish (timeout or crash): no intake, and the library stays as it was (spec §5.5).
            # dfranzen's retention would read the missing list as empty and drop every program (S1, 2026-10-03: 13
            # timeouts in long games each wiped the library).
            return
        before = dict(self._kept_functions)
        super()._record_retained_functions(sandbox_result, payload)
        after = self._kept_functions
        for name, src in after.items():
            if before.get(name) != src:
                if name in self._v6_library_order:
                    self._v6_library_order.remove(name)
                self._v6_library_order.append(name)
                self._v6_library_dirty = True
                session = getattr(self._step_env_callback, '__self__', None)
                self._v6_library_steps[name] = getattr(session, 'action_count', None)
                self._v61_library_levels[name] = self._v61_level
                self._v61_archive.pop(name, None)            # redefined: no longer archived
        for name in list(self._v6_library_order):
            if name not in after:
                self._v6_library_order.remove(name)
                self._v6_library_dirty = True
        if 'function_retention' in payload:          # spec field name and wording
            payload['programs'] = _v6_retention_notes(payload.pop('function_retention'))

    # ----------------------------------------------------------------------------------------------- the turn loop
    def _v6_degenerate(self, text):
        if not text or len(text) < 500:
            return False
        return len(zlib.compress(text.encode())) / len(text.encode()) < 0.01 or ta._is_degenerate_text(text)

    def _v6_generate(self, external):
        """generate with HC1b: one retry with the tool results replaced by N6; then a minimal notice."""
        try:
            return self._v6_session.generate(external)
        except CapacityRefusal as refusal:
            self._v6_stats['capacity_refusals'] += 1
            self._v6_log('v6-capacity-refusal', dict(i_max=refusal.i_max, receipt=refusal.receipt))
            retry = []
            for m in external:
                if m.get('role') == 'tool':
                    n = self._v6_count(json.dumps(m.get('content'), ensure_ascii=False))
                    m = dict(m, content=_fill(TEXTS['N6'], n=n))
                retry.append(m)
            try:
                return self._v6_session.generate(retry)
            except CapacityRefusal:
                self._v6_stats['capacity_refusals'] += 1
                minimal = [dict(m, content='Output omitted.') if m.get('role') == 'tool' else
                           (dict(m, content=[p for p in m['content'] if p.get('type') == 'text'])
                            if isinstance(m.get('content'), list) else m) for m in external]
                return self._v6_session.generate(minimal)

    def analyze(self, state_path, action_num, valid_actions=None, step_env=None, transcript_path=None,
                analysis_step=None, transcript_updated=None, request_timeout_seconds=None, should_stop=None):
        if not state_path.exists():
            return None
        self._ensure_session(state_path)
        self._step_env_callback = step_env
        self._current_valid_actions = ta._normalize_valid_actions(valid_actions)
        current_frame, history_entries = load_runtime_state(state_path)
        level_now = getattr(current_frame, 'level', None)          # item 6: tokens on the level
        if level_now != self._v61_tokens_level:
            self._v61_tokens_level = level_now
            self._v61_level_tokens = 0
            self._v61_noact_level_from = None
        if RESTART_TOKENS and self._v6_started and self._v61_level_tokens >= RESTART_TOKENS:
            current_frame, history_entries = self._v61_restart(state_path, current_frame, history_entries)
        step_executed = False
        stop_reason = None
        try:
            if self._v6_resume:
                text = self._v6_resumption_text(current_frame, valid_actions, history_entries)
                content = [dict(type='text', text=text), dict(type='text', text='Current board (unchanged).'),
                           current_grid_image_part(current_frame)]
                opener = {"role": "user", "content": content, "_arc3_observation": True}
                kind = 'resume'
                self._v6_resume = False
                self._v6_pending_list = None
            else:
                text, kind = self._build_user_prompt(action_num, valid_actions=valid_actions, current_frame=current_frame,
                                                     history_entries=history_entries,
                                                     previous_step_summary=self._last_step_summary)
                opener = self._v6_opener(text, kind, current_frame, history_entries)
                if kind == 'takeover':
                    self._v61_takeover = False
            external = list(self._v6_pending_tools) + [opener]
            if not self._v6_started:
                external = [{"role": "system", "content": self._system_prompt}] + external
            self._v6_pending_tools = []
            self._v6_obs_count += 1
            if variants.recap():
                recap = self._v6_recap_parts()
                if recap:
                    opener['content'] = list(opener['content']) + recap
                self._v6_record_observation(current_frame)
            if getattr(self, '_v6_pending_list', None):
                self._v6_note_list(opener, self._v6_obs_count)
            if getattr(self, '_v6_pending_rules', None):
                self._v6_note_rules(opener, self._v6_obs_count)
            self._v6_stats['turns'] += kind != 'resume'
            self._v6_log('v6-opener', dict(kind=kind, obs_index=self._v6_obs_count, text=text,
                                           images=sum(1 for p in opener['content'] if p.get('type') == 'image_url')))
            self._turn_generated_tokens = 0
            while True:
                if should_stop is not None and should_stop():
                    stop_reason = 'stop_requested'
                    if external:                       # never leave a tool result unsent in a live session
                        self._v6_pending_tools = [m for m in external if m.get('role') == 'tool']
                    break
                result = self._v6_generate(external)
                self._v6_started = True
                external = []
                self._v6_stats['requests'] += 1
                self._accumulate_usage_tokens(result.usage)
                self._v61_level_tokens += int((result.usage or {}).get('completion_tokens') or 0)
                self._v6_log('model-turn', dict(usage=result.usage, finish_reason=result.finish_reason,
                                                pending=result.pending, closed_by=result.closed_by,
                                                message=result.message, seconds=round(result.seconds, 3),
                                                turn_tokens=self._turn_generated_tokens))
                if result.pending:                    # length pause: continue in place
                    self._v6_stats['continuations'] += 1
                    continue
                self._v6_stats['completions'] += 1
                msg = result.message
                if self._v6_rule_capture is not None:
                    self._v6_capture_rule(ta._extract_reasoning_text(msg) + '\n' +
                                          ta._normalize_message_content(msg.get('content', '')))
                if result.closed_by:                  # harness hard close / repeat close: recovery in the next append
                    self._v6_stats['closes'] += 1
                    rec = TURN_REPEAT_RECOVERY if result.closed_by == 'repeat_close' else TURN_BUDGET_RECOVERY
                    external = [{"role": "user", "content": rec}]
                    continue
                reasoning = ta._extract_reasoning_text(msg)
                content = ta._normalize_message_content(msg.get('content', ''))
                tool_calls = json.loads(json.dumps(msg.get('tool_calls') or []))
                markup = ta._contains_tool_call_markup(reasoning, content)
                if not tool_calls and markup:
                    tool_calls = ta._recover_tool_calls_from_markup(reasoning, content)
                    if tool_calls:
                        self._v6_log('v6-markup-recovered', dict(count=len(tool_calls)))
                for call in tool_calls:                       # item 1: a python call without `code`
                    fn = call.get('function') or {}
                    if str(fn.get('name', '')).strip() != 'python':
                        continue
                    raw_args = fn.get('arguments', '{}')
                    try:
                        parsed = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                    except json.JSONDecodeError:
                        continue                              # malformed JSON: N3 below
                    if isinstance(parsed, dict) and str(parsed.get('code') or '').strip():
                        continue
                    decode = getattr(self._v6_session, 'decode', None)
                    ids = getattr(result, 'assistant_token_ids', None)
                    raw_text = decode(ids) if (decode is not None and ids) else ''
                    code = (code_from_arguments(parsed) or recover_code(raw_text)
                            or recover_code((reasoning or '') + '\n' + (content or '')))
                    if code:
                        fn['arguments'] = json.dumps({'code': code})
                        call['function'] = fn
                        self._v6_stats['reparsed'] = self._v6_stats.get('reparsed', 0) + 1
                        self._v6_log('v61-reparse', dict(recovered=True, chars=len(code)))
                    else:
                        call['_v61_format_error'] = True
                        self._v6_stats['reparse_failed'] = self._v6_stats.get('reparse_failed', 0) + 1
                        self._v6_log('v61-reparse', dict(recovered=False))
                malformed = []
                for call in tool_calls:
                    raw = (call.get('function') or {}).get('arguments', '{}')
                    if isinstance(raw, str):
                        try:
                            json.loads(raw)
                        except json.JSONDecodeError as exc:
                            malformed.append(str(exc))
                if malformed or not tool_calls:
                    if malformed:
                        key, nudge = 'N3', TEXTS['N3']
                    elif markup:
                        key, nudge = 'N4', TEXTS['N4']
                    elif self._v6_degenerate(content) or self._v6_degenerate(reasoning):
                        key, nudge = 'N5', TEXTS['N5']
                    else:
                        key, nudge = 'N1', TEXTS['N1']
                        last = [l.strip() for l in (reasoning or content or '').split('\n') if l.strip()]
                        if last and PLAN_WORDS.search(last[-1]) and len(last[-1]) <= 300:
                            nudge += ' ' + _fill(TEXTS['N1_plan'], last_line=last[-1])
                    self._v6_stats['nudges'][key] = self._v6_stats['nudges'].get(key, 0) + 1
                    self._v6_log('v6-nudge', dict(kind=key))
                    external = [{"role": "user", "content": nudge}]
                    continue
                tool_messages = []
                acted = False
                for i, call in enumerate(tool_calls):
                    if acted:
                        tool_messages.append({"role": "tool", "tool_call_id": call.get('id', ''),
                                              "content": json.dumps({"tool": "python", "error": (
                                                  "Not executed: an earlier call in this reply executed actions, "
                                                  "which ended the turn.")})})
                        continue
                    if call.get('_v61_format_error'):
                        tool_messages.append({"role": "tool", "tool_call_id": call.get('id', ''),
                                              "content": json.dumps({"tool": "python", "error": _FORMAT_ERROR})})
                        self._v61_noact_calls += 1
                        continue
                    fn = call.get('function') or {}
                    raw = fn.get('arguments', '{}')
                    args = json.loads(raw) if isinstance(raw, str) else (raw if isinstance(raw, dict) else {})
                    t0 = time.monotonic()
                    self._v6_tool_images = []
                    game = getattr(self._step_env_callback, '__self__', None)
                    n0 = len(getattr(game, 'history_entries', None) or [])
                    dispatch = self._dispatch_tool(state_path, str(fn.get('name', '')).strip(), args)
                    imgs, self._v6_tool_images = self._v6_tool_images, []
                    self._v6_log('tool-output', dict(seconds=round(time.monotonic() - t0, 3), acted=dispatch.step_executed,
                                                     history_len=len(history_entries), content=dispatch.content,
                                                     converted_boards=len(imgs)))
                    content = dispatch.content
                    if imgs:
                        self._v6_stats['converted_boards'] = self._v6_stats.get('converted_boards', 0) + len(imgs)
                        content = [dict(type='text', text=dispatch.content)]
                        for cap, part in imgs:
                            content += [dict(type='text', text=cap), part]
                    if ANIM_PUSH and dispatch.step_executed:
                        push = self._v6_anim_push(game, n0)
                        if push:
                            if isinstance(content, str):
                                content = [dict(type='text', text=content)]
                            content = content + push
                    tool_messages.append({"role": "tool", "tool_call_id": call.get('id', ''), "content": content})
                    if dispatch.step_executed:
                        acted = True
                        self._v61_noact_calls = 0
                    else:
                        self._v61_noact_calls += 1
                if acted:
                    step_executed = True
                    self._v6_pending_tools = tool_messages     # sent with the next opener
                    break
                breaker = self._v61_breaker(tool_messages)
                if breaker == 'yield':
                    self._v6_stats['breaker_yields'] = self._v6_stats.get('breaker_yields', 0) + 1
                    self._v6_pending_tools = tool_messages
                    self._v6_resume = True
                    stop_reason = 'yield'
                    break
                if breaker:
                    self._v6_stats['breaker_notes'] = self._v6_stats.get('breaker_notes', 0) + 1
                    tool_messages = tool_messages + [{"role": "user", "content": breaker}]
                if YIELD_TOKENS and self._turn_generated_tokens >= YIELD_TOKENS:
                    self._v6_stats['yields'] += 1
                    self._v6_log('v6-yield', dict(turn_tokens=self._turn_generated_tokens))
                    self._v6_pending_tools = tool_messages
                    self._v6_resume = True
                    stop_reason = 'yield'
                    break
                external = tool_messages
        finally:
            self._step_env_callback = None
            self._current_valid_actions = []
        return ta.AnalyzerTurnResult(step_executed=step_executed, reasoning='',
                                     yielded_control=stop_reason == 'yield')

    # ----------------------------------------------------------------------------------------------- v6.1 helpers
    @staticmethod
    def _v61_tool_error(message):
        content = message.get('content')
        if isinstance(content, list):
            content = next((p.get('text') for p in content if isinstance(p, dict) and p.get('type') == 'text'), '')
        try:
            data = json.loads(content) if isinstance(content, str) else {}
        except (TypeError, json.JSONDecodeError):
            return None
        err = data.get('error') if isinstance(data, dict) else None
        return str(err).strip()[:400] if err else None

    def _v61_breaker(self, tool_messages):
        """item 1: the same tool error in consecutive rounds -> a corrective note at 2, a yield at 3."""
        errors = [self._v61_tool_error(m) for m in tool_messages if m.get('role') == 'tool']
        error = errors[0] if errors and all(e and e == errors[0] for e in errors) else None
        if error and error == self._v61_last_error:
            self._v61_error_streak += 1
        else:
            self._v61_error_streak = 1 if error else 0
        self._v61_last_error = error
        if not error or self._v61_error_streak < 2:
            return None
        self._v6_log('v61-breaker', dict(streak=self._v61_error_streak, error=error[:200]))
        if self._v61_error_streak >= 3:
            return 'yield'
        if 'code' in error and ('non-empty' in error or 'no `code`' in error):
            return _fill(_BREAKER, k=self._v61_error_streak, error=error)
        return ('The last %d python calls returned the same error: %s Do not repeat that call: change the code, '
                'or inspect the state first.' % (self._v61_error_streak, error))

    def _v61_restart(self, state_path, current_frame, history_entries):
        """item 6, run B: RESET the level and start a new conversation (system prompt, the level's first board, the
        completion summaries). The level's programs are archived; ledger, attempts and no-action counts start again."""
        game = self._v61_session()
        if game is None:
            return current_frame, history_entries
        import arcengine
        level = getattr(current_frame, 'level', None)
        tokens = self._v61_level_tokens
        step = getattr(current_frame, 'step', None)
        game.__dict__['v61_harness_reset'] = True
        game.__dict__['v61_call_executed'] = 0
        try:
            game._execute_action(arcengine.ActionInput(id=arcengine.GameAction.RESET, data={}), batch_index=1,
                                 batch_size=1)
        finally:
            game.__dict__['v61_harness_reset'] = False
        for key in ('v61_voluntary_resets', 'v61_harness_resets'):      # stats, also when no action call follows
            if key in game.__dict__:
                self._v6_stats[key[4:]] = game.__dict__[key]
        try:
            self._v6_session.close()
        except Exception:  # noqa: BLE001
            pass
        if self._v61_session_factory is not None:
            self._v6_session = self._v61_session_factory()
        for name, value in (('_v6_started', False), ('_v6_pending_tools', []), ('_v6_prev_opener_grid', None),
                            ('_v6_prev_opener_level', None), ('_v6_resume', False), ('_v6_reminder', None),
                            ('_v6_rules_reminder', None), ('_v6_pending_list', None), ('_v6_pending_rules', None),
                            ('_v6_obs_count', 0), ('_v6_obs_records', {}), ('_v6_recap_shown', None),
                            ('_v6_push_note', False), ('_v6_rule_capture', None), ('_last_step_summary', None),
                            ('_last_action_call_result', None), ('_v61_noact_calls', 0), ('_v61_last_error', None),
                            ('_v61_error_streak', 0), ('_death_ledger_attempts', []), ('_death_ledger_index', {}),
                            ('_death_ledger_total', 0)):
            setattr(self, name, value)
        attempts = (getattr(game, 'v6_level_attempts', None) or {}).get(level, [])
        game.__dict__.setdefault('v61_attempt_offset', {})[level] = len(attempts)   # decision 5: count from the restart
        archived = []
        for name in list(self._v6_library_order):
            if self._v61_library_levels.get(name) == level and name in self._kept_functions:
                src = self._kept_functions.pop(name)
                try:
                    sig = function_signature(src)
                except Exception:  # noqa: BLE001
                    sig = name + '(...)'
                self._v61_archive[name] = dict(signature=sig, doc=_docline(src), level=level,
                                               step=self._v6_library_steps.get(name), source=src)
                self._v6_library_order.remove(name)
                archived.append(name)
        self._v61_level_tokens = 0
        self._v61_restarts += 1
        self._v61_takeover = True
        current_frame, history_entries = load_runtime_state(state_path)
        self._v61_noact_level_from = len(history_entries)
        self._v6_stats['restarts'] = self._v6_stats.get('restarts', 0) + 1
        self._v6_log('v61-restart', dict(level=level, step=step, tokens_on_level=tokens, attempts_before=len(attempts),
                                         archived=archived, summaries=sorted(self._v6_rules)))
        return current_frame, history_entries

    def _v6_anim_push(self, game, n0):
        """B2: content parts for the last animated action among history entries n0.. (this call's actions), or [].
        Pushed only when its transient cells (changed and changed back) include cells inside the board area. Frames:
        the intermediate ones (the settled board is the current board) that show a transient cell differently from the
        settled board; at most ANIM_MAX, evenly spaced."""
        store = getattr(game, 'v6_anim', None) or {}
        n1 = len(getattr(game, 'history_entries', None) or [])
        entries = game.history_entries
        auto = lambda i: isinstance(getattr(entries[i], 'result', None), dict) and entries[i].result.get('automatic')
        executed = [i for i in range(n0, n1) if not auto(i)]
        idx = next((i for i in reversed(executed) if i in store), None)
        if idx is None:
            return []
        rec = store[idx]
        chain = []
        for rows, cols, data in rec['chain']:
            chain.append([list(data[r * cols:(r + 1) * cols]) for r in range(rows)])
        final = chain[-1]
        rows, cols = len(final), (len(final[0]) if final else 0)
        inner = [(r, c) for r, c in _transient_cells(chain)
                 if ANIM_BORDER <= r < rows - ANIM_BORDER and ANIM_BORDER <= c < cols - ANIM_BORDER]
        if not inner:
            return []
        frames = chain[1:]                                       # engine frames; the last is the settled board
        n = len(frames)
        shown = [k for k, f in enumerate(frames[:-1])
                 if any(r < len(f) and c < len(f[r]) and f[r][c] != final[r][c] for r, c in inner)] or list(range(n - 1))
        if not shown:
            return []
        if len(shown) > ANIM_MAX:
            shown = [shown[round(j * (len(shown) - 1) / (ANIM_MAX - 1))] for j in range(ANIM_MAX)]
        entry = game.history_entries[idx]
        action = getattr(entry, 'action', None) or rec.get('step')
        where = _bbox_phrase([min(r for r, _ in inner), min(c for _, c in inner),
                              max(r for r, _ in inner), max(c for _, c in inner)])
        from inference.agent.runtime_state import Frame
        parts = [dict(type='text', text='Animation of %s: frames %d..%d of %d, oldest first; cells changed and changed '
                                         'back%s; the settled board is the current board.' % (
                                             action, shown[0] + 1, shown[-1] + 1, n, where))]
        for k in shown:
            parts += [dict(type='text', text='Animation, action event %s: frame %d of %d' % (rec.get('step'), k + 1, n)),
                      {"type": "image_url", "image_url": {"url": frame_to_png_data_url(
                          Frame(grid=frames[k], step=0, level=0), upscale=ANIM_SCALE)}}]
        self._v6_push_note = bool(executed) and idx == executed[-1]
        self._v6_stats['anim_pushes'] = self._v6_stats.get('anim_pushes', 0) + 1
        self._v6_stats['anim_frames'] = self._v6_stats.get('anim_frames', 0) + len(shown)
        self._v6_log('v6-anim-push', dict(index=idx, frames=len(shown), of=n, transient=len(inner)))
        return parts

    def _v6_record_observation(self, frame):
        game = getattr(self._step_env_callback, '__self__', None)
        grid = getattr(frame, 'grid', None) or []
        level = getattr(frame, 'level', None)
        self._v6_obs_records[self._v6_obs_count] = dict(
            step=getattr(frame, 'step', None), level=level,
            attempt=len((getattr(game, 'v6_level_attempts', None) or {}).get(level, [])) + 1,
            hist=len(getattr(game, 'history_entries', None) or []),
            board=(len(grid), len(grid[0]) if grid else 0, bytes(v for row in grid for v in row)))

    def _v6_board_left(self, t, blocks):
        oid = f"obs-{t:08d}"
        if oid in self._v6_session.rebased_away:
            return True
        b = blocks.get(oid)
        return b is not None and b.get('board_resident') is False

    def _v6_recap_parts(self):
        """The old harness's recap rule (v5.29 _recap_parts) on the server's board residency."""
        records = self._v6_obs_records
        cur = self._v6_obs_count
        blocks = {b.get('observation_id'): b for b in (self._v6_session.blocks or []) if isinstance(b, dict)}
        left = [t for t in sorted(records) if t < cur and self._v6_board_left(t, blocks)]
        shown = self._v6_recap_shown
        if not left or (shown is not None and (not self._v6_board_left(shown['turn'], blocks)
                                               or left[-1] <= shown['last'])):
            return []
        chosen = left[-RECAP_COUNT:]
        self._v6_recap_shown = dict(turn=cur, last=chosen[-1])
        game = getattr(self._step_env_callback, '__self__', None)
        hist = getattr(game, 'history_entries', None) or []
        order = sorted(records)
        steps = [records[t]['step'] for t in chosen]
        span = 'step %s' % steps[0] if len(chosen) == 1 else 'steps %s-%s' % (steps[0], steps[-1])
        parts = [dict(type='text', text=(
            'Recap: the %d most recent boards no longer in your context (%s), oldest first. These are the boards as '
            'they were at those turns; the current board is the image above.' % (len(chosen), span)))]
        from inference.agent.runtime_state import Frame
        previous = None
        restarted = 'level restarted' if variants.a13() else 'game over'
        for t in chosen:
            rec = records[t]
            later = [u for u in order if u > t]
            stop = records[later[0]]['hist'] if later else len(hist)
            acts = [e for e in hist[rec['hist']:stop]]
            head = 'Step %s (level %s, attempt %s)' % (rec['step'], rec['level'], rec['attempt'])
            if not acts:
                caption = head + ' -> no action before the next turn'
            else:
                labels = [str(getattr(e, 'action', '') or '?') for e in acts]
                results = [getattr(e, 'result', None) or {} for e in acts]
                groups = []
                for lab in labels:
                    if groups and groups[-1][0] == lab:
                        groups[-1][1] += 1
                    else:
                        groups.append([lab, 1])
                body = ', '.join('%s x%d' % (g, n) if n > 1 else g for g, n in groups)
                changed = sum(1 for r in results if r.get('board_changed'))
                body += '; unchanged' if not changed else '; board changed %d/%d' % (changed, len(results))
                flags = [f for f, k in (('level cleared', 'level_completed'), (restarted, 'game_over'),
                                        ('game complete', 'run_complete')) if any(r.get(k) for r in results)]
                caption = head + ' -> ' + body + ('; ' + ', '.join(flags) if flags else '')
            rows, cols, data = rec['board']
            if rec['board'] == previous:
                parts.append(dict(type='text', text=caption + ' (same board as the previous entry)'))
            else:
                grid = [list(data[r * cols:(r + 1) * cols]) for r in range(rows)]
                parts += [dict(type='text', text=caption),
                          {"type": "image_url", "image_url": {"url": frame_to_png_data_url(
                              Frame(grid=grid, step=0, level=0), upscale=RECAP_SCALE)}}]
            previous = rec['board']
        self._v6_stats['recaps'] = self._v6_stats.get('recaps', 0) + 1
        self._v6_stats['recap_boards'] = self._v6_stats.get('recap_boards', 0) + len(chosen)
        self._v6_log('v6-recap', dict(obs_index=cur, chosen=chosen, steps=steps))
        return parts

    def _v6_capture_rule(self, text):
        """A4: the lines that state the solved level's rule ('Rule L{n}:' and up to 4 following lines, to a blank)."""
        n, self._v6_rule_capture = self._v6_rule_capture, None
        lines = text.split('\n')
        tag = f'Rule L{n}:'
        for i, line in enumerate(lines):
            at = line.find(tag)
            if at < 0:
                continue
            body = [line[at + len(tag):].strip()]
            for extra in lines[i + 1:i + 5]:
                if not extra.strip():
                    break
                body.append(extra.strip())
            rule = ' '.join(b for b in body if b)[:SUMMARY_CHARS]
            if rule:
                self._v6_rules[n] = rule
                self._v6_rules_dirty = True
                self._v6_stats['rules_recorded'] = self._v6_stats.get('rules_recorded', 0) + 1
            return

    def _v6_note_rules(self, opener, obs_index):
        text = opener['content'][0]['text']
        block = self._v6_pending_rules
        at = text.find(block)
        if at < 0:
            return
        self._v6_rules_reminder = dict(obs_index=obs_index, offset=self._v6_count('<|im_start|>user\n' + text[:at]),
                                       length=self._v6_count(block))
        self._v6_rules_dirty = False

    def _v6_note_list(self, opener, obs_index):
        """Record where the program list sits inside the opener (token offset from the observation start)."""
        text = opener['content'][0]['text']
        plist = self._v6_pending_list
        at = text.find(plist)
        if at < 0:
            return
        prefix = '<|im_start|>user\n' + text[:at]
        self._v6_reminder = dict(obs_index=obs_index, offset=self._v6_count(prefix), length=self._v6_count(plist))
        self._v6_library_dirty = False
        self._v6_stats['reminder_appends'] += 1
