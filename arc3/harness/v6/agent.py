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
from inference.utils.retained_functions import function_signature

from v6 import sandbox_ext
from v6 import tool_output
from v6.transport import (CapacityRefusal, TURN_BUDGET_RECOVERY, TURN_REPEAT_RECOVERY)

from v6 import variants

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


class V6Agent(ta.ToolAgent):
    """One game's agent. `session` is a v6.transport.PersistentSession owned by this game; `logger` writes the
    raw-harness evidence (kind, payload)."""

    def __init__(self, *, session, logger, count_tokens, **kwargs):
        super().__init__(**kwargs)
        self._system_prompt = TEXTS['system']
        self._v6_session = session
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
        if not self._kept_functions:
            return False
        if self._v6_library_dirty:
            return True
        r = self._v6_reminder
        if r is None:
            return True
        return self._v6_carrier_evicted(r)

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
            ended = len((getattr(session, 'v6_level_attempts', None) or {}).get(s.get('level'), [])) or '?'
            trace = [t.replace('GAME OVER (attempt ended immediately after this action)',
                               'LEVEL RESTARTED (attempt %s ended here)' % ended) for t in trace]
        if trace:
            lines.append('Per action: ' + '; '.join(trace) + '.')
        return lines

    def _build_user_prompt(self, action_num, *, valid_actions=None, current_frame=None, history_entries=None,
                           previous_step_summary=None, **_):
        """The §6.3 opener text (images are added by _v6_opener). Returns (text, kind) with kind in
        first | same | levelup | restart."""
        valid = _v6_valid_display(valid_actions)
        step = getattr(current_frame, 'step', action_num)
        level = getattr(current_frame, 'level', 1)
        reminder = '' if variants.lean() else TEXTS['reminder']
        s = previous_step_summary if (previous_step_summary and not previous_step_summary.get('stale')) else None
        if not self._v6_started:
            text = _fill(TEXTS['first_turn'], valid_actions=valid).replace('{STANDING REMINDER}', reminder)
            text = '\n'.join(l for l in text.split('\n') if l.strip() and not l.startswith('[image]'))
            return text, 'first'
        parts = []
        kind = 'same'
        if s:
            parts += self._v6_summary_lines(s)
            manual = ta.ToolAgent._manual_reset_context_line(s)     # EXPOSE_RESET only (dfranzen's line)
            if manual:
                parts.append(manual)
            if s.get('game_over'):
                kind = 'restart'
                session = getattr(self._step_env_callback, '__self__', None)
                attempts = (getattr(session, 'v6_level_attempts', None) or {}).get(level, [])
                body = _fill(TEXTS['restart'], fatal_action=s.get('fatal_action') or 'the last executed action',
                             level=level, next_attempt=len(attempts) + 1, k=len(attempts),
                             lengths=', '.join(str(a) for a in attempts[-5:]) or '?')
                if len(attempts) > 5:                       # A3 (HD review): the list is capped at the last 5
                    body = body.replace('; the attempts lasted ', '; the last 5 attempts lasted ')
                parts.append('\n'.join(l for l in body.split('\n') if not l.startswith('[image]')))
            elif s.get('level_transition'):
                kind = 'levelup'
                if variants.a45():
                    self._v6_rule_capture = int(level) - 1
                parts.append(TEXTS['levelup'].replace('{level-1}', str(int(level) - 1)).replace('{level}', str(level)))
            else:
                parts.append(f"You are still on level {level}.")
            for stop in (s.get('no_op_stops') or [])[:1]:
                skipped = [str(x) for x in (stop.get('skipped') or [])]
                idx = stop.get('index')
                parts.append(_fill(TEXTS['noop_stop'], action=stop.get('action'), p=(idx + 1) if isinstance(idx, int) else '?',
                                   n=len(skipped) + ((idx + 1) if isinstance(idx, int) else 1), m=len(skipped),
                                   skipped=', '.join(skipped) or 'none'))
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
        if self._v6_reminder_due():
            plist = self._v6_program_list()
            parts.append(plist)
            self._v6_pending_list = plist
        else:
            self._v6_pending_list = None
        if self._v6_rules_due():
            rb = self._v6_rules_block()
            parts.append(rb)
            self._v6_pending_rules = rb
        else:
            self._v6_pending_rules = None
        parts.append(f"Step {step}, level {level}. Valid actions: {valid}.")
        parts.append(reminder)
        self._v6_push_note = False
        return '\n'.join(p for p in parts if p), kind

    def _v6_resumption_text(self, current_frame, valid_actions):
        valid = _v6_valid_display(valid_actions)
        body = _fill(TEXTS['resumption'], step=getattr(current_frame, 'step', '?'), level=getattr(current_frame, 'level', '?'),
                     valid_actions=valid).replace('{STANDING REMINDER}', '' if variants.lean() else TEXTS['reminder'])
        body = body.replace('{; the latest sequence executed this turn was {seq}}', '')
        return '\n'.join(l for l in body.split('\n') if l.strip() and not l.startswith('[image]'))

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
            if kind not in ('first', 'levelup') and changed:
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
        try:
            return super()._run_python_tool(state_path, arguments)
        finally:
            sandbox_ext.CURRENT.agent = None

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
                            source=src))
        return out

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
        step_executed = False
        stop_reason = None
        try:
            if self._v6_resume:
                text = self._v6_resumption_text(current_frame, valid_actions)
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
                if acted:
                    step_executed = True
                    self._v6_pending_tools = tool_messages     # sent with the next opener
                    break
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
            rule = ' '.join(b for b in body if b)[:600]
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
