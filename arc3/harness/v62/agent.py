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

import ast
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
    """Resolved at call time: v62.retention makes it constant-aware after this module is imported."""
    return _rf.function_signature(source)

from v62 import sandbox_ext
from v62 import tool_output
import requests
from v62.transport import (AppendOnlyViolation, CapacityRefusal, TURN_BUDGET_RECOVERY, TURN_REPEAT_RECOVERY)
from v62 import transport as _v64_tr      # v6.4 A3: the injected texts' final sentences

from v62 import variants

TEXTS = variants.apply(json.load(open(Path(__file__).with_name('texts.json')))['texts'])
TOOL = {"type": "function", "function": {
    "name": "python",
    "description": ("Run Python code against the live game. Inspect the runtime state, compute, and execute real game "
                    "actions with action(...)."),
    "parameters": {"type": "object", "properties": {"code": {"type": "string", "description": "Python source to run."}},
                   "required": ["code"]}}}
YIELD_TOKENS = int(os.environ.get("ARC3_V6_YIELD_TOKENS", "4096"))   # 0 = off; 4,096 since 2026-10-03 ~16:42 (user);
                                                                      # 8,192 in run 387b only (user 10-07 ~02:15: back to 4,096)
# round 5i: cues in a reply that held the yield; round 5l (user 10-08 ~01:55: "The 12K cue is not a user prompt. In that case we
# will allow the 4k yield"): none by default, since the cues are injections into the model's own reply and only user-turn
# prompts (the breaker note) take priority over the yield; ARC3_V62_YIELD_HELD_BY=soft_nudge,soft_budget,repeat_cue restores 5i
YIELD_HELD_BY = frozenset(x.strip() for x in os.environ.get('ARC3_V62_YIELD_HELD_BY', '').split(',') if x.strip())
TOOL_TEXT_TOKENS = int(os.environ.get("ARC3_V6_TOOL_TEXT_TOKENS", "3072"))   # 10-25 (the user): 2,048 -> 3,072
for _k, _v in list(TEXTS.items()):                                           # 10-25: the texts state the cap in force
    if isinstance(_v, str) and "about 2,048 tokens" in _v:
        TEXTS[_k] = _v.replace("about 2,048 tokens", "about {:,} tokens".format(TOOL_TEXT_TOKENS))
# B2 animation push (spec §8; user lean: on): after an acting call, up to ANIM_MAX frames of the call's last animated
# action at 8x (4x until 10-25), only when that action has transient cells inside the board area, attached to the acting call's tool
# result. Captions "Animation, action event ..." put them in the server's 'animation' image kind (tool segment), the
# kind the control's frames had and the per-kind eviction rank was tuned with.
ANIM_PUSH = os.environ.get("ARC3_V6_ANIM_PUSH", "0") == "1"
# Recap (old harness v5.29, ARC3_V6_RECAP=1): when the opener that carried the last recap has left the context and newer
# boards have left it too, the next opener re-shows the RECAP_COUNT most recent boards whose opener's board image is no
# longer resident (server receipt: history_state.blocks[].board_resident; a rebase that dropped the observation counts),
# oldest first, at 8x (4x until 10-25), each captioned with step, level, attempt and the actions executed after it.
RECAP_COUNT, RECAP_SCALE = 10, 8   # 10-25 (the user): every image at 8x (was 4)
ANIM_MAX, ANIM_SCALE, ANIM_BORDER = 6, 8, 4       # border = the solver's NOOP_GUARD_BORDER_WIDTH (4-cell edge band)
def _v64_tail(text):
    """v6.4 A3: the final sentence of an injected text (before its </think>)."""
    body = str(text).split('</think>')[0].strip()
    return body.rsplit('. ', 1)[-1].strip()


_V64_INJECTED_TAILS = tuple(_v64_tail(getattr(_v64_tr, k)) for k in ('TURN_BUDGET_CUE', 'TURN_BUDGET_NUDGE', 'TURN_REPEAT_CUE')
                            if getattr(_v64_tr, k, None))
PLAN_WORDS = re.compile(r"\b(call|run|execute|try|probe|act|press|move|click|let me|i will|i'll|now)\b", re.I)
RESTART_TOKENS = int(os.environ.get("ARC3_V61_RESTART_TOKENS", "0"))   # v6.1 item 6 (run B); 0 = off (run A)
RESTART_MIN_LEFT = float(os.environ.get("ARC3_V62_RESTART_MIN_LEFT", "1500"))  # round 5c (user; B3 ka59, ls20, re86
                             # restarted with 4-7 min left): no level restart with less run time left (s); 15 min (user
                             # 2026-10-05 ~21:20: the late phase includes the submission's end margin; was 10 min);
                             # v6.3 item C (user 2026-10-12): 25 min (a budget restart clears its level within 25K / 50K /
                             # 100K tokens with p 0.02 / 0.08 / 0.19; restarts granted with < 30 min left solved 2 of 27)
NO_ACTION_CALLS = int(os.environ.get("ARC3_V61_NO_ACTION_CALLS", "0"))  # v6.1 item 11; v6.3 item B (user 2026-10-12):
                             # off by default (0), replaced by the token warnings below; the counter stays for the logs
# v6.3 item B (user 2026-10-12, HARNESS_V63_PLAN): generated tokens (reasoning included) since the last executed action
# in the conversation. Warnings the first time the count crosses each NOACT_WARN threshold (explicit about the budget and
# that acting restores it; they carry the untried-actions list and take precedence over a nudge, a breaker note and the
# token yield); at NOACT_TOKENS the conversation ends as at the 200K budget (notes turn, swap-out, restart). 0 / empty = off.
# Evidence (396 / 404a / 404b): first warnings 137 / 57 / 74, second 34 / 13 / 15, restarts 12 / 3 / 5 (0.3-0.9% of tokens)
NOACT_WARN = tuple(sorted(int(x) for x in os.environ.get("ARC3_V62_NOACT_WARN", "16000,32000,48000").split(',') if x.strip()))
# v6.6 (user 10-17): a facts-only cue at 16K before v6.5's two (32K facts with the limit clause, 48K must-act); restart at 64K
NOACT_TOKENS = int(os.environ.get("ARC3_V62_NOACT_TOKENS", "64000"))
# v6.3 item A (user 2026-10-12): a game swapped out with no level completed re-queues below every other waiting game
# (never-started games and games with >= 1 level first), ordered by fewest restarts, then restart_value, then FIFO;
# an ordering rule, not an exclusion (it gets its restarts in open lanes). 0 restores v6.2's single swap band.
ZERO_LEVEL_BAND = os.environ.get("ARC3_V62_ZERO_LEVEL_BAND", "1") == "1"
# v6.3 item E (user 2026-10-13: "This shouldn't happen on the official games but it can be a fallback. Just make sure it
# does not cause unintended problems"): a game whose level completions reach number_of_levels while the engine still
# reports NOT_FINISHED ends (the completed levels are scored; taaf's step() would assert on the next action anyway)
LEVELS_COMPLETE_END = os.environ.get("ARC3_V63_LEVELS_COMPLETE_END", "1") == "1"
# round 5l (item 111): D1 (user 10-07 23:51): at the token restart one turn asks for notes on the abandoned attempt
# (save_summary), shown after the restart. D2 (user 10-08 ~01:55: "end it as soon as the current assistant turn ends"): no
# deferral; the budget is checked between turns, so the turn that crossed it has ended (a level-up resets the count)
RESTART_NOTES = os.environ.get("ARC3_V62_RESTART_NOTES", "1") == "1"
ATTEMPT_NOTES_TOKENS = 8000      # all notes on one level shown after a restart (newest kept; Fable 5.5)
V65_OPENER = os.environ.get('ARC3_V65_OPENER', '1') == '1'     # v6.5 OPENER (user via IE 10-17)
V65_PIN = os.environ.get('ARC3_V65_PIN', '0') == '1'           # v6.5 C5 (arm A): the opener's leading section is pinned
V65_PIN_CAP = int(os.environ.get('ARC3_V65_PIN_CAP', '2000'))  # tokens of the section (user via IE)
POSITION_RESTART = int(os.environ.get("ARC3_V62_POSITION_RESTART", "1950000"))  # round 5g (user 2026-10-06): restart the
                             # level before a session's logical positions reach the server's RoPE rows (2,097,152), so the
                             # server never rebases (a full re-prefill); 0 = off. Never reached so far (max 1.28M in 7 runs)
POSITION_NOTES_MIN = int(os.environ.get("ARC3_V62_POSITION_NOTES_MIN", "50000"))  # round 5m (user 2026-10-08): before a
                             # position restart, the notes turn of the token restart when this conversation has generated at
                             # least this many tokens on the level (the RoPE margin above POSITION_RESTART, ~147K positions,
                             # leaves room for it); 0 = never
PROTOCOL_RESTARTS = int(os.environ.get("ARC3_V62_PROTOCOL_RESTARTS", "3"))  # round 5d (380 g047 crashed on a turn that
                             # ended on <|endoftext|>): a session error restarts the level in place, at most this many
                             # times per game; the next one ends the game as before
SUMMARY_CHARS = 1200      # v6.1 item 6: captured summary length (characters); replaced by SUMMARY_TOKENS in round 5g
SUMMARY_TOKENS = int(os.environ.get('ARC3_V62_SUMMARY_TOKENS', '4000'))   # round 5g (user): a summary over this many
                          # tokens (served tokenizer) is truncated; the level-up message states the budget
RULE_TURNS = 8            # v6.2: completions after a level-up in which the "Rule Lk" summary is looked for (round 3,
                          # A2 ft09: the L2 summary came in the 5th; was 4)
RULE_LINES = 10           # round 5d (C50 g043: a 6-item summary lost its goal line at 5; the next-level cut of round 5c
                          # now ends run-on summaries); round 5c had 5 (B3: 7 games ran on into next-level planning);
                          # round 5g (user): the summary is a bullet list of up to 10 items, so 10 items (or prose lines)
                          # are kept after the header
# v6.1 item 6 (user 2026-10-05 ~03:00 / ~03:15 UTC): after a restart the new conversation's `history` omits the restarted
# level's earlier attempts (earlier levels and the board after the RESET stay), and the programs archived at restarts are
# not shown (programs(archived=True) is empty, read_code does not find them); 'full' keeps both (sub-point ii as first agreed)
RESTART_HISTORY = os.environ.get("ARC3_V61_RESTART_HISTORY", "level")
# v6.7 SOFT (agreed with IE 2026-10-18; apply_v67.py; arm B of the next A/B): the kind of a harness restart (token budget,
# no-action limit, position limit, session error). hard (default) = today's: a RESET, a new conversation from the level's
# first board, the level's programs archived and its earlier attempts hidden. soft = only the conversation restarts (a new
# session and the per-conversation state) from the current board: no RESET, the programs and the history stay, the
# attempt goes on. Under soft a restart is still hard when the game reports GAME OVER, when the caller asks for hard, or when
# a dead-board rule fires: DEAD_GAMEOVERS GAME OVERs in this conversation on this level came from a board identical to the
# current one, or none of the last DEAD_WINDOW executed actions on the level reached a new interior state (0 turns a rule off)
RESTART_KIND = variants.restart_kind()
DEAD_GAMEOVERS = int(os.environ.get('ARC3_V67_DEAD_GAMEOVERS', '3'))
DEAD_WINDOW = int(os.environ.get('ARC3_V67_DEAD_WINDOW', '0'))
# v6.7 LOOP (agreed with IE 2026-10-18; apply_v67.py; core of both arms): the loop ladder on C1's signatures, periods up to
# LOOP_PMAX. Consecutive cycle reports in a conversation: 1-4 the C1 line (rung 0); the LOOP_RUNG1-th the loop-rule block, after
# which a call whose code repeats one of the cycle's calls is not executed (rung 1); the LOOP_RUNG2-th, or the 2nd refused
# repeat, a notes turn and a soft restart from the current board, also under the hard switch (rung 2); a hard restart when that
# recurs on the level within LOOP_HARD_TOKENS of a loop soft restart or after LOOP_SOFT_CAP of them (rung 3); the
# LOOP_PARK_AT-th loop hard restart on a level parks the game (its swap-out queues it below every other game)
LOOP = variants.loop_on()
LOOP_PMAX = int(os.environ.get('ARC3_V67_LOOP_PMAX', '8'))
LOOP_RUNG1 = int(os.environ.get('ARC3_V67_LOOP_RUNG1', '5'))
LOOP_RUNG2 = int(os.environ.get('ARC3_V67_LOOP_RUNG2', '8'))
LOOP_HARD_TOKENS = int(os.environ.get('ARC3_V67_LOOP_HARD_TOKENS', '200000'))
LOOP_SOFT_CAP, LOOP_PARK_AT = 2, 2
LOOP_REPORT_S = float(os.environ.get('ARC3_V67_LOOP_REPORT_S', '1800'))   # the LOOPS stdout line's period (run_v6)
# v6.7x LOOPHARD (user decision 50: no soft restarts; apply_v67x.py): the kind of the ladder's rung-2 restart, soft (the default,
# as built) | hard: a RESET and a new conversation from the level's first board with the notes (attempt_summary_request_loop_hard,
# the hard takeover). Rung 3 then follows the level's earlier loop restarts of that kind; parking is unchanged
LOOP_RUNG2_KIND = variants.loop_rung2_kind()
# v6.8 texts (the IE agreement, 2026-10-18; apply_v67x.py): the measured v6.6 base plus game facts, each switch off by default.
# The text switches live in variants (ARC3_V68_NO_SCORE, _NOTES_NEG, _REFRESH_TEXTS and the texts of the others); here the
# agent's parts: ARC3_V68_LEDGER (the v6.7 level-clear ledger without the other NOTES parts: level-up, restart and refresh
# openers), ARC3_V68_NEVER_TYPES (restart openers: the valid action types never executed on the level), ARC3_V68_LOOP_TEXTS
# (the loop-rule block's facts and {level}; the hard takeover's loop line), ARC3_V68_LOOP_FRAME (a loop restart, rung 2 or 3,
# also needs the same interior board at the start of each repetition of the cycle)
LEDGER68 = variants.v68_ledger()
NEVER68 = variants.v68_never_types()
LOOP_TEXTS68 = variants.v68_loop_texts()
LOOP_FRAME68 = variants.v68_loop_frame()
if LOOP_FRAME68 and not LOOP:
    raise RuntimeError('ARC3_V68_LOOP_FRAME=1 needs the loop ladder (ARC3_V67_LOOP=1)')
# v6.8b LOOP_FRAME fix (IE, after the loop audit review67_loops, 2026-10-18; apply_v67x.py item V68B; all under
# ARC3_V68_LOOP_FRAME=1): the corrected frame condition (_v68_ends_ok: the interior at the END of each of the last k repetitions
# identical, or with a period of 2 repetitions when k >= 4) gates rung 2 / 3 and, at the moment of the call, rung 1's refusals
# (a cycle call is executed when it fails); ARC3_V68_LOOP_FALLBACK consecutive cycle reports (default 16 with LOOP_FRAME, 0 = off)
# bring a hard rung 2 without the frame check. Frame-blocked triggers and refusals do not end the episode.
LOOP_FALLBACK68 = int(os.environ.get('ARC3_V68_LOOP_FALLBACK', '16' if LOOP_FRAME68 else '0') or 0)
if LOOP_FALLBACK68 and not LOOP_FRAME68:
    raise RuntimeError('ARC3_V68_LOOP_FALLBACK needs ARC3_V68_LOOP_FRAME=1')
if LOOP_FALLBACK68 < 0 or (LOOP_FALLBACK68 and LOOP_FALLBACK68 <= LOOP_RUNG2):
    raise RuntimeError('ARC3_V68_LOOP_FALLBACK must be 0 or exceed the rung-2 count (%d): %d' % (LOOP_RUNG2, LOOP_FALLBACK68))
if (LEDGER68 or NEVER68) and 'notes_ledger' not in TEXTS:    # the ledger texts come with the v6.1 texts (variants.r61)
    raise RuntimeError('ARC3_V68_LEDGER / ARC3_V68_NEVER_TYPES need the v6.1 texts (ARC3_V61_TEXTS=1, ARC3_V6_A13=1)')


def _v68_interior(frame):
    """v6.8 LOOP_FRAME: a board's interior as compared for a loop restart: the whole board as the death ledger compares it,
    without the outer band (NOOP_GUARD_BORDER_WIDTH cells, as the dead-window rule crops it); None without a board."""
    grid = getattr(frame, 'grid', None)
    if not grid:
        return None
    from inference.framework import solver as _sv
    b = int(getattr(_sv, 'NOOP_GUARD_BORDER_WIDTH', 4) or 0)
    return tuple(tuple(row[b:len(row) - b]) for row in grid[b:len(grid) - b])


def _v68_ends_ok(starts, current, p, k):
    """v6.8b LOOP_FRAME (the loop audit's corrected form, cond67.py I_ends_q2): True when the interiors at the END of each of the
    last k repetitions of the p-call cycle are identical, or (k >= 4) repeat with a period of 2 repetitions. starts[j] is the
    interior acting turn j started from, so a repetition's end is the next repetition's start and the last one's is current
    (the interior now). False when a needed board is missing."""
    n = len(starts)
    if p <= 0 or k < 2 or n < k * p:
        return False
    ends = [starts[n - k * p + (i + 1) * p] for i in range(k - 1)] + [current]
    if any(e is None for e in ends):
        return False
    return all(e == ends[0] for e in ends) or (k >= 4 and all(ends[i] == ends[i + 2] for i in range(k - 2)))
# v6.7 REFRESH (variant B of the next A/B; IE SPEC UPDATE 3 with the 07:4x amendments): at a level-up in a conversation whose
# logical length (the persistent session's server-reported logical position count, last_receipt['logical_tokens']) is at least
# REFRESH_LOGICAL, the conversation ends and a new one starts at the new level's first board: no RESET, no attempt, no
# swap-out; the library and the summaries carry over. The completed level's summary turn runs first when its summary is not
# saved yet (actions are not executed in it). 0 = off.
REFRESH_LOGICAL = int(os.environ.get('ARC3_V67_REFRESH_LOGICAL', '0'))
REFRESH = REFRESH_LOGICAL > 0
# v6.8 REFRESH_PROGRAMS (IE: built off; the user, 18:02 UTC: yes, a selection; the final rule from the refresh reviewer's data:
# recency alone; apply_v67x.py item V68P): in the refresh opener only (the new conversation after a REFRESH), right after the
# library count line, the ARC3_V68_REFRESH_PROG_RECENT (default 8) most recent programs, by the turn of each one's last call (an
# executed python call whose code names it) or, if never called, of its last writing (a def line); ARC3_V68_REFRESH_PROG_FREQ
# (default 0) programs by call count may come first (ties to the more recent). program_list's per-program format. The call
# counts and turns cover the whole game (with the switch on).
REFRESH_PROGRAMS68 = variants.v68_refresh_programs()
PROG_FREQ68 = int(os.environ.get('ARC3_V68_REFRESH_PROG_FREQ', '0'))
PROG_RECENT68 = int(os.environ.get('ARC3_V68_REFRESH_PROG_RECENT', '8'))
if REFRESH_PROGRAMS68 and not REFRESH:
    raise RuntimeError('ARC3_V68_REFRESH_PROGRAMS=1 needs the refresh (ARC3_V67_REFRESH_LOGICAL > 0)')
if REFRESH_PROGRAMS68 and 'refresh_program_header' not in TEXTS:   # its texts come with the refresh texts (variants)
    raise RuntimeError('ARC3_V68_REFRESH_PROGRAMS needs the v6.1 texts (ARC3_V61_TEXTS=1, ARC3_V6_A13=1)')
if PROG_FREQ68 < 0 or PROG_RECENT68 < 0:
    raise RuntimeError('ARC3_V68_REFRESH_PROG_FREQ and _RECENT must be >= 0')
# v6.9 CLEAR_STEP (V67_DESIGN decisions 82 and 87; the user's GO 2026-10-18 22:54 UTC; IE's rulings 23:0x; apply_v67x.py item
# V69CS): in a new conversation's opener (a restart takeover of any reason or kind, or a refresh) at level L >= 2, after the
# opener text and before the current board (which stays the last image), the clearing step of level L - 1 from the game
# session's record (v62.solver keeps it at the clear): a line, the board the clearing action was executed from and the first
# board it produced (the first raw frame that differs from that board and is not level L's first board; worded "the first
# changed board" when that is not the action's frame 0); without such a frame, the fallback line and the board before only.
# Standard board images (the opener's renderer). Not in level-up openers inside a conversation, nor in the GAME OVER restart
# opener inside a conversation. Receipts clear_step_shown / _fallback / _missing / _index; the v69-clear-step event.
CLEAR_STEP69 = variants.v69_clear_step()
if CLEAR_STEP69 and 'clearstep_line' not in TEXTS:          # its texts come with the v6.1 texts (variants.r61)
    raise RuntimeError('ARC3_V69_CLEAR_STEP=1 needs the v6.1 texts (ARC3_V61_TEXTS=1, ARC3_V6_A13=1)')
if CLEAR_STEP69:
    from inference.agent.vision_context import current_grid_image_enabled as _v69_images_on
    if not _v69_images_on():
        raise RuntimeError('ARC3_V69_CLEAR_STEP=1 needs the board images (MULTIMODAL_CONTEXT=current_grid)')
if CLEAR_STEP69 and variants.recap():   # IE 10-19 (point 5): the recap reads the server's board_resident (_v6_board_left)
    raise RuntimeError("ARC3_V69_CLEAR_STEP=1 cannot run with the recap (ARC3_V6_RECAP=1): the server's history_state "
                       "board_resident follows an observation's first observation image, which in a restart or refresh "
                       "opener with the clearing-step block is the clear's before board, not the current board. Turn one "
                       "off until board_resident skips the block's images.")
# v6.9 INLEVEL (V67_DESIGN decisions 94-99: the user's adoption 2026-10-19 00:53 UTC, IE's design points; apply_v67x.py item
# V69IR): ARC3_V69_INLEVEL_REFRESH = a logical position threshold (0 = off, the default; the next run 500000). Within a level
# (never at the level-up turn, which the level-up refresh covers), when this conversation's logical position (the persistent
# session's server receipt, last_receipt['logical_tokens'], as the level-up refresh reads it) has reached the threshold at a
# mid-level opener and no harness restart, notes turn or level-up refresh is due in that turn (the budget, no-action, position
# and loop restarts, a session error and the level-up refresh keep precedence), that turn becomes a notes turn (the
# attempt-notes request in its in-level form; actions are refused) and the next turn start opens a new conversation at the
# CURRENT board: no RESET, the attempt goes on, no swap-out or park, no archive, nothing hidden (the level-up refresh's path).
# The game reporting GAME OVER (only RESET executes) defers the trigger. The new opener (a takeover): the pinned lead (the newest
# attempt note, the previous level's summary), the in-level line, the library count line with the program list
# (ARC3_V68_REFRESH_PROGRAMS), the clearing-step block (level >= 2, ARC3_V69_CLEAR_STEP), the current board last. It re-arms with
# every conversation (a new one starts low). Receipts: inlevel_refreshes, the lists inlevel_refresh_logical / _levels /
# _level_tokens / _tokens_to_action, inlevel_notes_turns, inlevel_deferred_game_over, inlevel_dropped; the events
# v69-inlevel-notes-turn, v69-inlevel-refresh, v69-inlevel-refresh-first-action, v69-inlevel-dropped; the game session's
# per-level record v69_inlevel_refreshes.
INLEVEL69 = variants.v69_inlevel_refresh()
if INLEVEL69 and 'takeover_inlevel' not in TEXTS:          # its texts come with the v6.1 texts (variants.r61)
    raise RuntimeError('ARC3_V69_INLEVEL_REFRESH needs the v6.1 texts (ARC3_V61_TEXTS=1, ARC3_V6_A13=1)')
if INLEVEL69 and float(os.environ.get('ARC3_V67_EXT200', '0') or 0):
    raise RuntimeError('ARC3_V69_INLEVEL_REFRESH cannot run with ARC3_V67_EXT200 (its extension is per conversation on a level)')
# v6.9 ACT-REV shadow (V67_DESIGN decisions 91, 95, 97 and 99; the revisit review's detector: reports/review68_revisit section 4,
# pod review68/revisit combined2.py 'actrev'; apply_v67x.py item V69AS): ARC3_V69_ACTREV_SHADOW=1 runs the executed-action cycle
# detector in shadow mode: it logs and never acts (no model-facing change). Per conversation, the executed actions of each acting
# report (the step summary's executed_actions, which its report line "Your previous call executed ..." lists); a cycle report =
# the smallest period p <= ACTREV_PMAX69 (8) whose last k*p reports repeat k >= 3 times; consecutive cycle reports count up, a
# non-cycle report ends the episode, a conversation's first opener resets the detector. The novelty check: no counted action
# among the last ACTREV_REPS69 (2) repetitions' actions (those of the last 2p acting reports) reached an interior (rows and
# columns 4..59, exact) not seen earlier on the level (all attempts and conversations, the level's first board included;
# automatic RESETs are not counted). The would-fire: the first report at consecutive >= ACTREV_K69 (12) whose check finds no new
# board, once per episode. From the ACTREV_LOG_FROM69-th (8) consecutive report to the would-fire, every report logs its check
# (v69-actrev-shadow: game, level, step, logical position, consecutive, period, k, the window and its new boards, the check,
# would_fire, the active guard rung); an episode that reached ACTREV_LOG_FROM69 logs its end (v69-actrev-shadow-end). Receipts
# actrev_shadow_fires / _checks / _episodes8 / _max_consecutive / _errors.
ACTREV69 = variants.v69_actrev_shadow()
ACTREV_K69, ACTREV_PMAX69, ACTREV_REPS69, ACTREV_LOG_FROM69 = 12, 8, 2, 8


def _v69_interior_key(grid, border=4):
    """v6.9 ACT-REV: a board's exact interior identity: blake2b-8 of the cells inside the outer band (the revisit review's
    extraction, review67 extract.py board_hashes: rows and columns 4..59 of a 64 x 64 board); a board too small for the band
    whole."""
    import hashlib
    import itertools
    rows = len(grid or ())
    cols = max((len(r) for r in grid), default=0) if rows else 0
    try:
        if rows > 2 * border and cols > 2 * border:
            data = bytes(itertools.chain.from_iterable(r[border:cols - border] for r in grid[border:rows - border]))
        else:
            data = bytes(itertools.chain.from_iterable(grid or ()))
    except (TypeError, ValueError):
        data = repr([list(r) for r in grid or ()]).encode()
    return hashlib.blake2b(data, digest_size=8).hexdigest()


class _V69Novelty:
    """v6.9 ACT-REV: level-new interiors over one game's history, fed in history order (entry i = history index i). A level
    starts at the first entry with its level (the level's first board, its seed: never counted); per level the interior keys
    seen so far, every entry's included; an entry is counted-new when it is not an automatic RESET and its key was not seen
    earlier on the level. cum[i] = the counted-new entries among the first i (the revisit review's rv rows: flag 1 of mask m0,
    counted = not automatic and not a seed)."""

    def __init__(self):
        self.level, self.seen, self.cum = object(), set(), [0]

    def __len__(self):
        return len(self.cum) - 1

    def feed(self, level, key, automatic):
        if level != self.level:
            self.level, self.seen = level, {key}
            self.cum.append(self.cum[-1])
            return
        new = key not in self.seen
        self.seen.add(key)
        self.cum.append(self.cum[-1] + (1 if new and not automatic else 0))

    def new_between(self, lo, hi):
        """the counted-new entries with history index in [lo, hi)"""
        return self.cum[hi] - self.cum[lo]


class _V69ActRev:
    """v6.9 ACT-REV: the executed-action cycle detector of one conversation (the revisit review's ACT-REV, combined2.py's actrev
    over review67 lib67's per-conversation acting reports). reset(n) at a conversation's first opener (n = the history length
    there); report(acts, n, novelty) at each acting report (acts: its executed actions; n: the history length at its opener).
    Returns {} or a dict with 'ended' (an episode that reached log_from ended: its peak count and the count at its would-fire)
    and 'check' (a report at consecutive >= log_from before the episode's would-fire: p, k, the count, the window [lo, hi) of
    the last reps repetitions' actions, its counted-new boards, would_fire, first = the episode's first check)."""

    def __init__(self, k_fire=12, pmax=8, reps=2, log_from=8):
        self.k_fire, self.pmax, self.reps, self.log_from = k_fire, pmax, reps, log_from
        self.acts, self.starts, self.prev_n = [], [], 0
        self.cons, self.peak, self.fired = 0, 0, None

    def _end(self):
        out = dict(peak=self.peak, fired=self.fired) if self.peak >= self.log_from else None
        self.cons, self.peak, self.fired = 0, 0, None
        return out

    def reset(self, n):
        ended = self._end()
        self.acts, self.starts, self.prev_n = [], [], n
        return dict(ended=ended) if ended else {}

    def report(self, acts, n, novelty):
        self.acts.append(acts)
        self.starts.append(self.prev_n)
        self.prev_n = n
        del self.acts[:-512], self.starts[:-512]
        p, k = _v65_loop(self.acts, self.pmax)
        if k < 3:
            ended = self._end()
            return dict(ended=ended) if ended else {}
        self.cons += 1
        self.peak = max(self.peak, self.cons)
        if self.fired is not None or self.cons < self.log_from:
            return {}
        lo = self.starts[len(self.acts) - self.reps * p]
        new = novelty.new_between(lo, n)
        fire = self.cons >= self.k_fire and new == 0
        if fire:
            self.fired = self.cons
        return dict(check=dict(p=p, k=k, consecutive=self.cons, lo=lo, hi=n, new=new, would_fire=fire,
                               first=self.cons == self.log_from))


def _v68_select_programs(names, counts, last_call, last_def, freq, recent):
    """v6.8 REFRESH_PROGRAMS: (the most called, the most recent) program names. names: the library in its order; counts: name ->
    executed python calls whose code names it; last_call / last_def: name -> (turn, sequence number) of its last call / last
    writing. Recency: the last call, or the last writing for a program never called (then the library order, the later
    first). The most called: the top `freq` called programs by count (ties: the more recent first); then the `recent` most
    recent of the rest."""
    order = {n: i for i, n in enumerate(names)}
    rec = lambda n: (last_call[n] if n in last_call else last_def.get(n, (-1, -1))) + (order[n],)   # noqa: E731
    most = sorted((n for n in names if counts.get(n, 0) > 0), key=lambda n: (counts[n],) + rec(n), reverse=True)[:freq]
    rest = sorted((n for n in names if n not in most), key=rec, reverse=True)[:recent]
    return most, rest
# v6.7 NOTES (core; IE spec): the restart notes in three parts, the level-clear ledger in level-start and restart openers, the
# level's action count and capped coverage facts in restart openers, the opener-size receipts
NOTES = variants.notes_on()
if NOTES and 'notes_ledger' not in TEXTS:            # the notes texts come with the v6.1 texts (variants.r61)
    raise RuntimeError('ARC3_V67_NOTES=1 needs the v6.1 texts (ARC3_V61_TEXTS=1, ARC3_V6_A13=1)')
if REFRESH and 'takeover_refresh' not in TEXTS:      # the refresh texts come with the v6.1 texts (variants.r61)
    raise RuntimeError('ARC3_V67_REFRESH_LOGICAL needs the v6.1 texts (ARC3_V61_TEXTS=1, ARC3_V6_A13=1)')
_LOOP_KEYS = ('loop_episodes', 'loop_rung1', 'loop_refused', 'loop_soft_restarts', 'loop_hard_restarts', 'loop_cycle_actions')
if LOOP and 'takeover_loop' not in TEXTS:            # the ladder's texts come with the v6.1 texts (variants.r61)
    raise RuntimeError('ARC3_V67_LOOP=1 needs the v6.1 texts (ARC3_V61_TEXTS=1, ARC3_V6_A13=1)')
if RESTART_KIND == 'soft' and 'takeover_soft' not in TEXTS:    # the soft texts come with the v6.1 texts (variants.r61)
    raise RuntimeError('ARC3_V67_RESTART_KIND=soft needs the v6.1 texts (ARC3_V61_TEXTS=1, ARC3_V6_A13=1)')
# v6.7x SUMLOOKUP (user decision 53, 2026-10-18; apply_v67x.py): the sandbox's read-only summaries() returns a copy of the saved
# level summaries ({level: text}, _v6_rules as stored) as they stood at the start of the python call; its two texts (variants)
# and receipts (summaries_calls, the v67-sumlookup event). 0 = off.
SUMLOOKUP = variants.sumlookup_on()
if SUMLOOKUP and '`summaries()`' not in TEXTS['system']:   # the texts come with the v6.1 texts (variants.r61)
    raise RuntimeError('ARC3_V67_SUMLOOKUP=1 needs the v6.1 texts (ARC3_V61_TEXTS=1, ARC3_V6_A13=1)')
# v6.7x EXT200 (user decision 48, 2026-10-18; apply_v67x.py; log-only, no judge in the loop): where the notes turn of a token-budget
# restart (reason 'tokens': RESTART_TOKENS generated on the level in one conversation) would be decided, the attempt (that
# conversation on the level) is assigned once by u = _v67x_u(EXT200_SEED, game, level, its index on the level): u < EXT200 extends
# it (its budget on the level rises once by EXT200_TOKENS; the notes turn and the restart or park follow at the new limit as
# today), else the restart is unchanged; when the late gate skips the restart the assignment is 'late' and nothing changes. Not
# for no-action, position, loop, session-error or refresh restarts. Receipts: the ext200_* stats, the v67-ext200 and
# v67-ext200-outcome events. 0 = off.
EXT200 = float(os.environ.get('ARC3_V67_EXT200', '0') or 0)
EXT200_TOKENS = int(os.environ.get('ARC3_V67_EXT200_TOKENS', '50000'))
EXT200_SEED = int(os.environ.get('ARC3_V67_EXT200_SEED', '20261018'))
if not (0.0 <= EXT200 <= 1.0) or EXT200_TOKENS < 0:
    raise RuntimeError('ARC3_V67_EXT200 must be a fraction in [0, 1] and ARC3_V67_EXT200_TOKENS at least 0')
# v6.9 RAB RESTART_ACROSS (the RESET A/B, V67_DESIGN decisions 105 and 117; rab69 apply_rab69.py item R69RT; IE's constraint):
# ARC3_V69_RESTART_ACROSS=1 makes the per-level budget restart (ARC3_V61_RESTART_TOKENS) count the tokens generated on the level
# since it began or the harness last restarted it, over all its conversations (an in-level refresh or a park starts a new
# conversation without ending that count; a level-up, a level-up refresh and every harness restart start it again), so in-level
# refreshes cannot starve it. Its notes request states that count when it spans more than one conversation
# (attempt_summary_request_across). At a turn start with an in-level refresh or park pending (its notes turn done) and the count
# at the budget, the restart wins: the pending refresh is dropped and its notes turn serves as the restart's. Receipts:
# attempt_tokens in v61-restart / v62-summary-turn, restart_notes_across, inlevel_superseded, the v69-inlevel-superseded event.
RESTART_ACROSS69 = os.environ.get('ARC3_V69_RESTART_ACROSS', '0') == '1'
if RESTART_ACROSS69 and EXT200:
    raise RuntimeError('ARC3_V69_RESTART_ACROSS cannot run with ARC3_V67_EXT200 (its budget extension is per conversation)')
# v6.9 RAB PARK (the allocation rung; the user's decisions 111 and 117; apply_rab69.py item R69PK): ARC3_V69_PARK_TOKENS > 0.
# Once a level has used that many tokens (the count over conversations above) since its first in-level refresh in that count, or
# since the game's last rung firing on it, without being cleared, and a NEVER-STARTED game waits for a lane (simalloc69, decision
# 127: parking pays only then; games swapped out at a restart and parked games do not count), the next mid-level opener becomes the
# in-level notes turn (the in-level refresh's texts; it wins over the logical trigger at the same
# opener); at the next turn start the game gives up its lane (the session closed) and queues in the parked band (below every
# other game); once it holds a lane again, a new conversation continues the level from the current board as after an in-level
# refresh (no RESET). Without a waiting never-started game at that turn start the lane is kept (park_skipped); the refresh follows.
# The game session and the agent stay in place while the thread waits, so their records carry over (the v69-park event's
# fingerprints before and after). A pending park loses to the budget restart. Needs the in-level refresh. Receipts:
# park_rung_fires, parks, park_skipped, park_wait_s, park_levels, park_records_changed; the v69-park event.
PARK69 = int(os.environ.get('ARC3_V69_PARK_TOKENS', '0') or 0)
if PARK69 < 0:
    raise ValueError('ARC3_V69_PARK_TOKENS must be 0 (off) or a positive token count')
if PARK69 and not INLEVEL69:
    raise RuntimeError('ARC3_V69_PARK_TOKENS needs the in-level refresh (ARC3_V69_INLEVEL_REFRESH > 0)')
# v7.0 SWAP70 (V67_DESIGN decisions 145-152; apply_swap70.py): the expectation-based swap-out's scaffolding, each switch off by
# default (PLAN.md, CHANGES_swap70.md). SOFT_CAP: the pool cap binds only while an eligible game waits (a park with trigger 'cap'
# in the over-cap band); INDEX (+ INDEX_FILE): the index at every turn boundary (v62/swap70.py), INDEX_SHADOW logs it; INDEX_ADMIT:
# the queue ordered by index; INDEX_SWAP: the index swap rule's park (trigger 'index'); NO_HARD_RESTART: every harness hard
# restart becomes an in-level refresh with its trigger's name.
from v70 import swap70 as _s70        # harness_v6/v70/swap70.py (outside v62/: the older suites stage v62/)


def _v70_switch(name):
    v = os.environ.get(name, '0')
    if v not in ('0', '1'):
        raise RuntimeError('%s must be 0 or 1, not %r' % (name, v))
    return v == '1'


V70_SOFTCAP = _v70_switch('ARC3_V70_SOFT_CAP')
V70_INDEX = _v70_switch('ARC3_V70_INDEX')
V70_SHADOW = _v70_switch('ARC3_V70_INDEX_SHADOW')
V70_ADMIT = _v70_switch('ARC3_V70_INDEX_ADMIT')
V70_SWAP = _v70_switch('ARC3_V70_INDEX_SWAP')
V70_NHR = _v70_switch('ARC3_V70_NO_HARD_RESTART')
V70_PARKS = V70_SWAP or V70_SOFTCAP                     # the triggers that park through the rung's path
V70_ON = V70_SOFTCAP or V70_INDEX or V70_NHR
V70_C = None
if V70_INDEX:
    _v70_file = os.environ.get('ARC3_V70_INDEX_FILE', '').strip()
    if not _v70_file:
        raise RuntimeError('ARC3_V70_INDEX=1 needs ARC3_V70_INDEX_FILE (the index constants; v62/swap70.py SCHEMA)')
    V70_C = _s70.load(_v70_file)
if (V70_SHADOW or V70_ADMIT or V70_SWAP) and not V70_INDEX:
    raise RuntimeError('ARC3_V70_INDEX_SHADOW / _ADMIT / _SWAP need ARC3_V70_INDEX=1')
if V70_SWAP and not V70_ADMIT:
    raise RuntimeError('ARC3_V70_INDEX_SWAP=1 needs ARC3_V70_INDEX_ADMIT=1 (a parked game returns by its index)')
if V70_SWAP and PARK69:
    raise RuntimeError('ARC3_V70_INDEX_SWAP=1 replaces the rung: ARC3_V69_PARK_TOKENS must be 0')
if (V70_PARKS or V70_NHR) and not INLEVEL69:
    raise RuntimeError('ARC3_V70_SOFT_CAP / _INDEX_SWAP / _NO_HARD_RESTART need the in-level refresh (ARC3_V69_INLEVEL_REFRESH > 0)')
if V70_SOFTCAP and int(os.environ.get('ARC3_V67_GAME_POOL', '0') or 0) <= 0:
    raise RuntimeError('ARC3_V70_SOFT_CAP=1 needs the run\'s pool (ARC3_V67_GAME_POOL > 0)')
# v7.0 HARNESS REVISION (apply_v70h.py; V67_DESIGN 163-182; PRIORITIES items 10-17, 19b, 19c): each switch off by default; with all
# of them off the tree behaves as swap70's. Arm B of the harness A/B turns them on.
V70H_LEDGER = _v70_switch('ARC3_V70H_ATTEMPT_LEDGER')       # 10: the attempt ledger at every attempt end
V70H_ILOPEN = _v70_switch('ARC3_V70H_IL_OPENER')            # 11: the in-level refresh opener (last actions, first board, change)
V70H_ILOPEN_TOKENS = int(os.environ.get('ARC3_V70H_IL_OPENER_TOKENS', '3000') or 3000)
V70H_ILOPEN_KMAX = int(os.environ.get('ARC3_V70H_IL_OPENER_KMAX', '100') or 100)
V70H_NOCLOCK = _v70_switch('ARC3_V70H_NO_CLOCK')            # 12: no run clock in model-visible action results
V70H_ALIASES = _v70_switch('ARC3_V70H_ALIASES')             # 13: preloaded names (sandbox_ext) and ImportError notes
V70H_FACTS = _v70_switch('ARC3_V70H_ACTION_FACTS')          # 14: the per-action verdict line also for a single action
V70H_ACCESSORS = _v70_switch('ARC3_V70H_ACCESSORS')         # 14: history tools in the sandbox (sandbox_ext)
V70H_BLEDGER = _v70_switch('ARC3_V70H_BOARD_LEDGER')        # 15: earlier actions from an identical board; the revisit fact
V70H_REFUSAL1 = _v70_switch('ARC3_V70H_REFUSAL_ONCE')       # 16: a refused call's message stated once
V70H_ACTREV = _v70_switch('ARC3_V70H_ACTREV')               # 17: ACT-REV replaces the 16-report fallback
V70H_RESIDENT = _v70_switch('ARC3_V70H_RESIDENT_POINTER')   # 19b: a text pointer for a resident unchanged board at a resumption
V70H_RESET0 = _v70_switch('ARC3_V70H_RESET0')               # 19c: a RESET in a 0-action attempt is refused (session_ext)
# v7.2 item 11 REDESIGN (apply_v72.py; the user 10-20, V67_DESIGN 201-202): the in-level refresh opener's block shows boards of
# the current attempt (at most V72_ILB_BOARDS besides the current board) and its last V72_ILB_ACTIONS actions, not item 11's list.
V72_ILB = _v70_switch('ARC3_V72_IL_BOARDS')
V72_ILB_BOARDS = int(os.environ.get('ARC3_V73_ILB_BOARDS', '5') or 5)      # v7.3: settable (v7.2's 5 by default)
V72_ILB_ACTIONS = int(os.environ.get('ARC3_V73_ILB_ACTIONS', '5') or 5)
# v7.1 ALLOCATION (apply_v71.py; the user's rules, V67_DESIGN 188-190): ARC3_V71_ALLOC=1 replaces the in-level refresh's logical
# trigger, the rung and the budget. The level's generated tokens in this conversation (_v61_level_tokens) reaching
# ARC3_V71_LEVEL_TOKENS make the next mid-level opener the in-level notes turn; at the next turn start the game parks (the parked
# band: after the never-started games, FIFO) while any game waits and ARC3_V71_GUARD_S seconds of the run are left, else the
# refresh follows in place; every NO_HARD_RESTART refresh parks the same way.
V71 = _v70_switch('ARC3_V71_ALLOC')
V71_TOKENS = int(os.environ.get('ARC3_V71_LEVEL_TOKENS', '200000') or 0)
V71_GUARD_S = float(os.environ.get('ARC3_V71_GUARD_S', '1800') or 0)
if V71:
    if not INLEVEL69:
        raise RuntimeError('ARC3_V71_ALLOC=1 needs the in-level refresh machinery (ARC3_V69_INLEVEL_REFRESH > 0)')
    if PARK69 or V70_SOFTCAP or V70_INDEX:
        raise RuntimeError('ARC3_V71_ALLOC=1 replaces the rung, the soft caps and the index: ARC3_V69_PARK_TOKENS, '
                           'ARC3_V70_SOFT_CAP and ARC3_V70_INDEX must be 0')
    if int(os.environ.get('ARC3_V67_GAME_POOL', '0') or 0) > 0:
        raise RuntimeError('ARC3_V71_ALLOC=1 runs without a budget: ARC3_V67_GAME_POOL must be unset or 0')
    if V71_TOKENS <= 0 or V71_GUARD_S < 0:
        raise RuntimeError('ARC3_V71_LEVEL_TOKENS must be > 0 and ARC3_V71_GUARD_S >= 0')
if V70H_ACTREV and not (ACTREV69 and LOOP and LOOP_FRAME68):
    raise RuntimeError('ARC3_V70H_ACTREV needs ARC3_V69_ACTREV_SHADOW=1, ARC3_V67_LOOP=1 and ARC3_V68_LOOP_FRAME=1')
if V70H_ILOPEN and not INLEVEL69:
    raise RuntimeError('ARC3_V70H_IL_OPENER needs the in-level refresh (ARC3_V69_INLEVEL_REFRESH > 0)')
if V72_ILB and not V70H_ILOPEN:
    raise RuntimeError('ARC3_V72_IL_BOARDS replaces item 11\'s block: it needs ARC3_V70H_IL_OPENER=1')
# v7.2b (apply_v72b.py; the user 10-20, V67_DESIGN 204): REFRESH_MIN_LEFT: no level-up and no in-level refresh in the run's last
# ARC3_V72_REFRESH_MIN_LEFT seconds (0 = off); exempt: the loop guard (incl. ACT-REV), a session error, the position trigger, the
# no-action stall; it replaces RESTART_MIN_LEFT for the triggers NO_HARD_RESTART turns into refreshes. PARK_RANK: the order of
# V71's parked band: fifo (v7.1), restart (restart_value, ARC3_V72_RV's constants), index (ARC3_V72_PARK_INDEX_FILE's index),
# interleave (alternating per parked admission, ARC3_V72_PARK_FIRST first); never-started games stay first.
V72_RML = float(os.environ.get('ARC3_V72_REFRESH_MIN_LEFT', '0') or 0)
V72_RANK = os.environ.get('ARC3_V72_PARK_RANK', 'fifo') or 'fifo'
V72_FIRST = os.environ.get('ARC3_V72_PARK_FIRST', 'restart') or 'restart'
V72_RV = dict(p=0.486, stuck=0.213, next=0.792, decay=0.734, left=0.79, level_tokens=62910.0, lane_toks=55.4)   # refit 10-20
for _kv in [x for x in os.environ.get('ARC3_V72_RV', '').split(',') if x.strip()]:
    _k, _eq, _v = _kv.partition('=')
    if not _eq or _k.strip() not in V72_RV:
        raise RuntimeError('ARC3_V72_RV: %r is not key=value with a key in %s' % (_kv, sorted(V72_RV)))
    V72_RV[_k.strip()] = float(_v)
if not (0 < V72_RV['p'] <= 1 and 0 < V72_RV['left'] <= 1 and min(V72_RV['stuck'], V72_RV['next'], V72_RV['decay']) >= 0
        and V72_RV['level_tokens'] > 0 and V72_RV['lane_toks'] > 0):
    raise RuntimeError('ARC3_V72_RV out of range: %r' % (V72_RV,))
V72_C = None
_v72_file = os.environ.get('ARC3_V72_PARK_INDEX_FILE', '').strip()
if _v72_file:
    V72_C = _s70.load(_v72_file)
if V72_RML < 0:
    raise RuntimeError('ARC3_V72_REFRESH_MIN_LEFT must be >= 0')
if V72_RML > 0 and not V70_NHR:
    raise RuntimeError('ARC3_V72_REFRESH_MIN_LEFT replaces the late restart rule where restarts are refreshes: it needs '
                       'ARC3_V70_NO_HARD_RESTART=1')
if V72_RANK not in ('fifo', 'restart', 'index', 'interleave'):
    raise RuntimeError('ARC3_V72_PARK_RANK must be fifo, restart, index or interleave, not %r' % V72_RANK)
if V72_FIRST not in ('restart', 'index'):
    raise RuntimeError('ARC3_V72_PARK_FIRST must be restart or index, not %r' % V72_FIRST)
if V72_RANK != 'fifo' and not V71:
    raise RuntimeError('ARC3_V72_PARK_RANK orders the v7.1 parked band: it needs ARC3_V71_ALLOC=1')
if V72_RANK in ('index', 'interleave') and V72_C is None:
    raise RuntimeError('ARC3_V72_PARK_RANK=%s needs ARC3_V72_PARK_INDEX_FILE (the index constants)' % V72_RANK)
# v7.2c (apply_v72c.py; the user 10-20, V67_DESIGN 210): PARK_BANDS: the parked band ordered first by the game's park count on its
# current level (fewer parks first; a level-up resets the count), then by ARC3_V72_PARK_RANK within a band (fifo: the gate's sequence).
V72_BANDS = _v70_switch('ARC3_V72_PARK_BANDS')
if V72_BANDS and not V71:
    raise RuntimeError('ARC3_V72_PARK_BANDS orders the v7.1 parked band: it needs ARC3_V71_ALLOC=1')
V72_PARKX = V72_RANK != 'fifo' or V72_BANDS          # the parked entry and the pump (v7.2b's ranker and / or v7.2c's bands)
V72_BAND_W = 1_000_000                               # priority per park-count band (above any band's size)
V72_BAND_MAX = 99
# v7.3 (apply_v73.py; the user 10-22, V73_DRAFT.md with IE's review): ARC3_V73=1: the v7.1 tokens trigger parks only (the notes
# turn and the park while ANOTHER game waits and ARC3_V71_GUARD_S seconds are left; otherwise no notes turn and no refresh: the
# conversation continues, the trigger armed); the strict in-level refresh at ARC3_V73_STRICT_LOGICAL logical positions (its notes
# turn, then the park while another game waits, else in place; none in the last ARC3_V71_GUARD_S seconds). ARC3_V73_CLEAR_BOARDS:
# the clearing-step block's boards of the cleared level's final attempt (0 = v6.9's pair only). ARC3_V73_ILB_*: above.
V73 = _v70_switch('ARC3_V73')
V73_STRICT = int(os.environ.get('ARC3_V73_STRICT_LOGICAL', '750000') or 0)
V73_CLEAR_BOARDS = int(os.environ.get('ARC3_V73_CLEAR_BOARDS', '0') or 0)
V73_NO_LEVELUP = _v70_switch('ARC3_V73_NO_LEVELUP_REFRESH')   # the user 10-22: no level-up refreshes (the strict cap alone)
if (V73 or V73_CLEAR_BOARDS) and 'v73_kept' not in TEXTS:
    raise RuntimeError('ARC3_V73 / ARC3_V73_CLEAR_BOARDS need the v7.3 texts (the v7.0 texts: an ARC3_V70H switch on)')
if V73 and not V71:
    raise RuntimeError('ARC3_V73=1 changes the v7.1 trigger: it needs ARC3_V71_ALLOC=1')
if V73 and V73_STRICT <= 0:
    raise RuntimeError('ARC3_V73_STRICT_LOGICAL must be > 0 with ARC3_V73=1')
if V73_NO_LEVELUP and not V73:
    raise RuntimeError('ARC3_V73_NO_LEVELUP_REFRESH=1 needs ARC3_V73=1 (its strict in-level cap replaces the level-up refresh)')
if V72_ILB_BOARDS < 2 or V72_ILB_ACTIONS < 1:
    raise RuntimeError('ARC3_V73_ILB_BOARDS must be >= 2 and ARC3_V73_ILB_ACTIONS >= 1')
if V73_CLEAR_BOARDS and (V73_CLEAR_BOARDS < 3 or not CLEAR_STEP69):
    raise RuntimeError('ARC3_V73_CLEAR_BOARDS must be 0 or >= 3 (the pair plus the attempt) and needs ARC3_V69_CLEAR_STEP=1')
# v7.4 (apply_v74.py; the user's rebase arm 10-22, REBASE_PROPOSAL_2026-10-22.md, V67_DESIGN 263-270): ARC3_REBASE_MODE=rebase: position
# rebases replace the in-level refreshes of a long conversation and of a park's return. The rebase point (logical >= ARC3_V74_REBASE_AT,
# the tokens trigger not due): the level's tokens <= ARC3_V74_RESET_MAX: the in-place rebase and the count reset; above: the park
# (suspended; the return rebases); the run's last ARC3_V71_GUARD_S seconds: in place without the reset. Parks suspend the server
# session; the return continues the conversation (the stint's counts reset); no notes turn before a park.
REBASE_MODE = os.environ.get('ARC3_REBASE_MODE', 'refresh')   # the user 10-23: one toggle (the server part reads it too)
if REBASE_MODE not in ('refresh', 'rebase'):
    raise RuntimeError('ARC3_REBASE_MODE must be refresh or rebase')
V74 = REBASE_MODE == 'rebase'
V74_AT = int(os.environ.get('ARC3_V74_REBASE_AT', '500000') or 0)
V74_RESET_MAX = int(os.environ.get('ARC3_V74_RESET_MAX', '100000') or 0)
if V74 and not (V73 and V71):
    raise RuntimeError('ARC3_REBASE_MODE=rebase replaces v7.3\'s strict cap and returns: it needs ARC3_V73=1 and ARC3_V71_ALLOC=1')
if V74 and not INLEVEL69:
    raise RuntimeError('ARC3_REBASE_MODE=rebase needs ARC3_V69_INLEVEL_REFRESH > 0 (the in-level hook runs its triggers)')
if V74 and V74_AT <= 0:
    raise RuntimeError('ARC3_V74_REBASE_AT must be > 0 with ARC3_REBASE_MODE=rebase')
# v7.6 HYBRID (apply_v76.py; the user 10-23 / 10-24, V67_DESIGN 297-300, 305 and 308): ARC3_V76_HYBRID=1 with ARC3_REBASE_MODE=rebase:
# the tokens trigger refreshes stuck levels (v7.1: its notes turn, then the park while another game waits, else in place; the park
# closes the session; none in the run's last ARC3_V71_GUARD_S seconds); the rebase point (logical >= ARC3_V74_REBASE_AT) at an action
# boundary becomes a summary turn (actions refused), then the next turn's first append rebases (the level's count kept) while the
# level's tokens are below ARC3_V76_REFRESH_AT; at or above it the in-level refresh (notes turn, park or in place) instead; in the
# run's last V76_LATE_S seconds (no refresh) the rebase whatever the count; none in the last ARC3_V76_REBASE_MIN_LEFT seconds. After a
# model RESET the reset ledger's line ends with the RESET reminder (v76_reset_line).
V76 = _v70_switch('ARC3_V76_HYBRID')
V76_REFRESH_AT = int(os.environ.get('ARC3_V76_REFRESH_AT', '150000') or 0)
V76_MIN_LEFT = float(os.environ.get('ARC3_V76_REBASE_MIN_LEFT', '600') or 0)
V76_FRAME_TEXT = '%dK' % round(int(os.environ.get('ARC3_S2_REBASE_CAP', '60000') or 60000) / 1000.0)   # the server's frame cap
V76_LATE_S = max(V71_GUARD_S, V72_RML)       # no refresh in the run's last seconds (the v7.1 guard, the v7.2b REFRESH_MIN_LEFT)
if V76 and not V74:
    raise RuntimeError('ARC3_V76_HYBRID=1 needs ARC3_REBASE_MODE=rebase (the transport\'s rebase flag and the server\'s rebuilt frame)')
if V76 and not 0 < V76_REFRESH_AT <= V71_TOKENS:
    raise RuntimeError('ARC3_V76_REFRESH_AT must be in (0, ARC3_V71_LEVEL_TOKENS]')
if V76 and not 0 <= V76_MIN_LEFT <= V71_GUARD_S:
    raise RuntimeError('ARC3_V76_REBASE_MIN_LEFT must be in [0, ARC3_V71_GUARD_S]')
if V76 and not ('v76_summary_request' in TEXTS and 'v76_reset_line' in TEXTS):
    raise RuntimeError('ARC3_V76_HYBRID=1 needs the v7.6 texts (variants.v76_hybrid)')
# v7.7 (apply_v77.py; the user 10-24, V67_DESIGN 312: the efficiency pass's levers 1-2): ARC3_V77=1: (A) the level-up summary has no
# 10-item limit (a list in a reply keeps all its items; the 4,000-token cap stays); (B) the level-up opener shows the previous level's
# saved summary to update; (C) summaries(query) searches the saved summaries (sandbox_ext); the board_shift fact only for a shift of
# the whole interior (session_ext); frame_diff pairs exact in-place recolours first (frame_diff, set by sandbox_ext). Unset or 0: v7.6.
V77 = _v70_switch('ARC3_V77')
RULE_ITEMS77 = 1000                     # v7.7 (A): a list summary's items kept by the capture (all; the token cap applies at the save)
V77_SNAP = 9                            # v7.7 (the user 10-24): the refresh opener's boards: the attempt's first + 8 (+ the current)
if V77 and not all(k in TEXTS for k in ('v77_prev_summary', 'v77_prev_missing', 'v77_attempt_steps', 'v77_notes_prev_summary',
                                         'v77_ilb_range', 'v77_ilb_head', 'v77_ilb_head_start')):
    raise RuntimeError('ARC3_V77=1 needs the v7.7 texts (variants.v77)')
if V77 and not SUMLOOKUP:
    raise RuntimeError('ARC3_V77=1 needs ARC3_V67_SUMLOOKUP=1 (summaries() in the sandbox)')
# v7.5 POLISH (the user 10-23: the polish review's decisions, REVIEW_2026-10-23.md Parts 3-5; BUILD_PLAN_polish75.md H1-H13): the text
# batch (TXT-1/2/3/4/6/8/9/10), the tool fixes (TOOL-1 to TOOL-4) and RUN-1 / RUN-2 behind one switch; unset or 0 leaves v7.4 as it is.
# variants.py, sandbox_ext.py, patches.py and lazy_history.py read the same switch at import.
V75_POLISH = variants.v75_polish()
# TOOL-2: the runtime functions a model may call as a tool by mistake (the provided ones and the starter helpers of the seeds)
_V75_RUNTIME_FUNCTIONS = frozenset(
    ('action', 'frame_diff', 'programs', 'read_code', 'save_summary', 'summaries', 'show_board', 'death_ledger', 'grid_cells',
     'color_counts', 'find_cells', 'bounding_box', 'grid_diff', 'connected_components', 'find_objects', 'find_object',
     'object_position', 'object_move')) | frozenset(re.findall(r'(?m)^def ([a-z]\w*)\(', sandbox_ext.SEEDS))


def _v75_and(items):
    # v7.5 TXT-4: '12', '12 and 7', '12, 7 and 30'
    items = [str(x) for x in items]
    return items[0] if len(items) == 1 else ', '.join(items[:-1]) + ' and ' + items[-1]


def _v75_period(text):
    # v7.5 TXT-10: a quoted error ends with a period (or ! / ?) before the note that follows it
    text = str(text).rstrip()
    return text if text.endswith(('.', '!', '?')) else text + '.'


def _v67x_u(seed, game, level, attempt):
    """v6.7x EXT200: the attempt's stable draw in [0, 1): sha256 of 'seed|game|level|attempt', its first 8 bytes read
    big-endian, the top 53 bits / 2**53 (reproducible offline from the v67-ext200 event's fields)."""
    import hashlib
    digest = hashlib.sha256(('%s|%s|%s|%s' % (seed, game, level, attempt)).encode()).digest()
    return (int.from_bytes(digest[:8], 'big') >> 11) / float(2 ** 53)


def _v67x_game_id(game):
    """v6.7x EXT200: the hash's game id (the taaf game run's game_id, else the game's own)."""
    tg = getattr(game, 'game', None)
    return getattr(getattr(tg, 'game_run', None), 'game_id', None) or getattr(tg, 'game_id', None)


def _v67_state(grid):
    """v6.7 SOFT: a board as the death ledger identifies it (ta._interior_state_hash's crop, before hashing: the whole board
    unless ARC3_STATE_IDENTITY_BORDER is set)."""
    b = ta._state_identity_border()
    rows, cols = len(grid), max((len(r) for r in grid), default=0)
    if rows > 2 * b and cols > 2 * b:
        return tuple(tuple(r[b:cols - b]) for r in grid[b:rows - b])
    return tuple(tuple(r) for r in grid)


def _v67_norm_code(code):
    """v6.7 LOOP: a call's code as the identical-call refusal compares it (trailing whitespace stripped, per line and at the
    end)."""
    return '\n'.join(line.rstrip() for line in str(code or '').rstrip().split('\n'))


def loops_totals(stats_by_game):
    """v6.7 LOOP (Kaggle visibility): (the totals over the games, {game: its loop stats} for the games with a rung 1-3
    event)."""
    games = {g: s for g, s in (stats_by_game or {}).items() if isinstance(s, dict)}
    tot = {k: sum(int(s.get(k) or 0) for s in games.values()) for k in _LOOP_KEYS}
    keys = _LOOP_KEYS + ('loop_reports', 'loop_cycle_tokens', 'loop_parks')
    hit = {g: {k: s[k] for k in keys if s.get(k)} for g, s in sorted(games.items())
           if any(s.get(k) for k in ('loop_rung1', 'loop_refused', 'loop_soft_restarts', 'loop_hard_restarts'))}
    return tot, hit


def loops_line(tot, seconds):
    """v6.7 LOOP: the stdout line LOOPS t=<hh:mm>: episodes E, rung1 X, refused Y, soft S, hard H, cycle actions A."""
    h, m = divmod(max(0, int(seconds)) // 60, 60)
    return ('LOOPS t=%02d:%02d: episodes %d, rung1 %d, refused %d, soft %d, hard %d, cycle actions %d'
            % (h, m, tot.get('loop_episodes', 0), tot.get('loop_rung1', 0), tot.get('loop_refused', 0),
               tot.get('loop_soft_restarts', 0), tot.get('loop_hard_restarts', 0), tot.get('loop_cycle_actions', 0)))


def loops_game_line(gid, s):
    """v6.7 LOOP: one game's line in the final summary."""
    return ('LOOPS game %s: episodes %d, rung1 %d, refused %d, soft %d, hard %d, cycle actions %d'
            % (gid, s.get('loop_episodes', 0), s.get('loop_rung1', 0), s.get('loop_refused', 0),
               s.get('loop_soft_restarts', 0), s.get('loop_hard_restarts', 0), s.get('loop_cycle_actions', 0)))
# v6.2 allocator (user 2026-10-05 ~03:40-03:55; M2_DECISIONS "allocator design agreed"). With dfranzen's priority gate on
# (ARC3_MAX_ACTIVE_STREAMS = lanes), a game that spends RESTART_TOKENS on a level without completing it gives up its lane
# (its session is closed, freeing the KV) and re-queues; it comes back with the restart. Never-started games are admitted
# first (his untrimmed band, in game order); swapped-out games by fewest swap-outs on their current level, then the
# expected score of a restart (restart_value), then FIFO. With nothing better waiting the handover returns at once: the
# in-place restart (run B's mechanism).
LANE_TOKS = float(os.environ.get("ARC3_V62_LANE_TOKS", "45"))           # generated tokens/s per lane (EV horizon)
LEVEL_TOKENS = float(os.environ.get("ARC3_V62_LEVEL_TOKENS", "60000"))   # tokens per further level (EV horizon)
_SWAP_BAND = 1_000_000                       # below his untrimmed band (2,000,000 - 100 x game index)
_ZERO_BAND = 100_000                         # v6.3 item A: swapped out with no level completed (below the swap band)


# round 5m (item 125, user 2026-10-09 "Refit now"; run 396's 225 restarts, M2_DECISIONS 10-09): restart_value's constants,
# refit from the 10-05 proxies. p: a budget restart solves its level in the new conversation 0.44 (L <= 4) / 0.45 (L > 4) of
# the resolved cases (was 0.85 / 0.75, the general level solve rate); the stuck level's official efficiency 0.33 (public 25,
# levels solved after a restart: mean level score 32.8, unchanged); the next level after a level solved only after a
# restart 0.69 (61 / 89; was 0.4); further levels x0.83 (910 / 1,090 after a level solved in its first conversation; was 0.85)
# v6.3 item D (user 2026-10-13 20:22 UTC, "adopted tentatively"; HARNESS_V63_PLAN D): refit on run 396's public 25 +
# Additional 166 (172 budget restarts, 1,033 solved levels) and cross-validated over 11 corpus runs. p = KM P(solve <= 200K)
# 0.41 (the 0.45 counted resolved cases only), x LEFT_FACTOR 0.7 when at most 2 levels are left (pooled KM <= 150K: 0.25 /
# 0.19 with 1 / 2 left vs 0.33 with >= 3); the stuck level's official efficiency 0.30 (public-25 restart solves over four
# runs: mean 30.1, n 41); the next level after a restart solve 0.78 (KM <= 400K: 0.80; pooled 0.93 public / 0.77 Additional);
# further levels x0.92 (first-conversation continuation; the 0.83 took won games' final levels as "next not solved").
# LANE_TOKS is set at deployment to the measured per-lane rate; the table is re-measured on long runs of the final stack.
# v6.5 RF (12-h audit AUDIT_12H_RESULTS.md, user 10-17 ~03:2x: adopted): LEFT_FACTOR 0.8, STUCK_C 0.27, DECAY_C 0.94 (and
# LANE_TOKS 50 in run_v6); RESTART_P, NEXT_C and LEVEL_TOKENS unchanged
RESTART_P = 0.41
STUCK_C, NEXT_C, DECAY_C = 0.27, 0.78, 0.94
LEFT_FACTOR = 0.8                 # on p when n - level + 1 <= 2


def restart_value(level, n_levels, remaining_s):
    """Expected share of the game's score from a restart at `level` (official weights j / W, W = n(n+1)/2): p x the sum,
    over the levels playable in the remaining time, of c_j x j / W; p = RESTART_P (x LEFT_FACTOR with <= 2 levels left),
    c = STUCK_C for the stuck level (its stalled actions count against its efficiency), NEXT_C for the next, then x DECAY_C
    per further level (refit from run 396, item 125; v6.3 item D)."""
    L = max(1, int(level or 1))
    n = max(L, int(n_levels) if n_levels else 8)
    h = int(max(0.0, float(remaining_s)) * LANE_TOKS / LEVEL_TOKENS)
    total, c = 0.0, STUCK_C
    for j in range(L, min(n, L + h) + 1):
        total += c * j
        c = NEXT_C if j == L else c * DECAY_C
    p = RESTART_P * (LEFT_FACTOR if n - L + 1 <= 2 else 1.0)
    return p * total / (n * (n + 1) / 2.0)


class _V62Snapshot:
    """A swapped-out game's queue entry; the gate re-scores it whenever it fills a freed lane (ARC3_PRIORITY_REFRESH_QUEUE)."""

    def __init__(self, level, n_levels, swaps):
        self.level, self.n_levels, self.swaps = level, n_levels, swaps
        self.parked = False                             # v6.7 LOOP: parked at a loop hard restart (below every band)

    @property
    def zero_band(self):
        """v6.3 item A: swapped out with no level completed (the current level is 1)."""
        return ZERO_LEVEL_BAND and int(self.level or 1) <= 1

    def priority(self, now, deadline):
        if getattr(self, 'parked', False):              # v6.7 LOOP: the lowest priority, also when the gate re-scores
            return -1_000_000_000
        remaining = (deadline - now) if deadline is not None else 0.0
        value = restart_value(self.level, self.n_levels, remaining)
        # swap-outs dominate (1,000 per swap, no cap: equal counts only compete on value), value in 0..999, FIFO by the
        # gate's sequence number; v6.3 item A: the zero-level band sits below the swap band with the same ordering
        base = _ZERO_BAND if self.zero_band else _SWAP_BAND
        return base - int(self.swaps) * 1000 + min(999, int(round(1000 * value)))


class _V70Snapshot(_V62Snapshot):
    """v7.0 INDEX_ADMIT / SOFT_CAP: a waiting game's queue entry, re-scored at every pump: band 'index' by its index (a never-
    started game from the level-1 prior of its number of levels; a parked or swapped-out game from its state with the parked
    flag), 'overcap' (a soft-cap bind) below every 'index' entry, 'lowest' (a loop park) below all. Without the index (SOFT_CAP
    alone) the band orders and the gate's sequence number breaks ties (FIFO)."""
    BANDS = {'index': 0, 'overcap': -10 ** 15, 'lowest': -10 ** 18}

    def __init__(self, state, band='index', gid=None):
        _V62Snapshot.__init__(self, state.get('level') or 1, state.get('n_levels'), 0)
        self.state, self.band, self.gid = dict(state), band, gid
        self.parked = band == 'lowest'

    def index_gate(self, gate, now):
        if V70_C is None:
            return 0.0
        dl = getattr(gate, '_clock_deadline', None) if gate is not None else None
        t0 = getattr(gate, '_clock_start', None) if gate is not None else None
        return _s70.index(V70_C, dict(self.state, time_left_s=None if dl is None else max(0.0, dl - now),
                                      elapsed_s=0.0 if t0 is None else max(0.0, now - t0)))

    def priority_gate(self, gate, now):
        return self.BANDS[self.band] + int(round(1e12 * self.index_gate(gate, now)))

    def priority(self, now, deadline):
        return self.priority_gate(ta._priority_gate(), now)


_original_snapshot_priority = ta._PriorityGate._snapshot_priority


def _gate_snapshot_priority(self, snapshot, now):
    if isinstance(snapshot, _V70Snapshot):                     # v7.0: scored by the index within its band
        return snapshot.priority_gate(self, now)
    if isinstance(snapshot, _V62Snapshot):
        return snapshot.priority(now, self._clock_deadline)
    return _original_snapshot_priority(self, snapshot, now)


ta._PriorityGate._snapshot_priority = _gate_snapshot_priority


def _v70_never_snapshot():
    """v7.0 INDEX_ADMIT: a never-started game's entry: level 1 and the game's number of levels, read from taaf's game run (TAAF
    starts the game before the solver's play() queues it; the play wrapper below puts the session in swap70.TLS)"""
    sess = getattr(_s70.TLS, 'session', None)
    run = getattr(getattr(sess, 'game', None), 'game_run', None)
    return _V70Snapshot(dict(level=1, n_levels=getattr(run, 'number_of_levels', None), never_started=True), 'index',
                        getattr(run, 'game_id', None))


_original_gate_acquire = ta._PriorityGate.acquire
_original_gate_pump = ta._PriorityGate._pump


def _v70_gate_acquire(self, priority):
    """v7.0 INDEX_ADMIT: the never-started game queues with its V70 snapshot (scored by the level-1 prior at every pump)"""
    with self._cond:
        self._enqueue_and_wait(priority, _v70_never_snapshot())


def _v70_gate_pump(self):
    """v7.0 INDEX_ADMIT: before a free lane is handed out, every V70 entry is re-scored (also without
    ARC3_PRIORITY_REFRESH_QUEUE), then the gate's own pump; each admission is noted (swap70.ADMITS: game, kind, index, N, time
    left, the competing entries at the top of the queue)"""
    snaps, before, top, now = None, None, None, None
    if self._free > 0 and self._waiting:
        import heapq
        now = time.monotonic()
        self._waiting = [(-_gate_snapshot_priority(self, self._snapshots[t], now), t)
                         if isinstance(self._snapshots.get(t), _V70Snapshot) else (p, t) for p, t in self._waiting]
        heapq.heapify(self._waiting)
        snaps, before, top = dict(self._snapshots), set(self._admitted), sorted(self._waiting)[:6]
    _original_gate_pump(self)
    if snaps is not None:
        dl = getattr(self, '_clock_deadline', None)
        comp = [dict(game=getattr(snaps.get(t), 'gid', None), priority=-p) for p, t in top]
        for t in [x for x in self._admitted if x not in before]:
            sn = snaps.get(t)
            if isinstance(sn, _V70Snapshot):
                _s70.ADMITS.note(dict(epoch=round(time.time(), 1), game=sn.gid, n_levels=sn.state.get('n_levels'),
                                      kind='never' if sn.state.get('never_started') else sn.band, level=sn.state.get('level'),
                                      index=sn.index_gate(self, now), time_left_s=None if dl is None else round(dl - now, 1),
                                      competing=comp, waiting=len(self._waiting) + 1))


if V70_ADMIT:
    ta._PriorityGate.acquire = _v70_gate_acquire
    ta._PriorityGate._pump = _v70_gate_pump


_V72_BAND = -1_000_000_000                     # the parked band (as _V62Snapshot.parked); a parked entry = the band + its rank
_V72_STATE = dict(n=0)                         # PARK_RANK: parked admissions so far (the interleave's phase)


def _v72_restart_value(level, n_levels, remaining_s, c=None):
    """v7.2b PARK_RANK: restart_value's formula (v6.2 / v6.3 item D) with ARC3_V72_RV's constants: p (x left with <= 2 levels
    left) x the sum over the levels playable in the remaining time (remaining_s x lane_toks / level_tokens further levels) of
    c_j x j / W, c = stuck for the stuck level, next for the next, x decay per further level; W = n(n+1)/2."""
    c = c or V72_RV
    L = max(1, int(level or 1))
    n = max(L, int(n_levels) if n_levels else 8)
    h = int(max(0.0, float(remaining_s)) * c['lane_toks'] / c['level_tokens'])
    total, cj = 0.0, c['stuck']
    for j in range(L, min(n, L + h) + 1):
        total += cj * j
        cj = c['next'] if j == L else cj * c['decay']
    p = c['p'] * (c['left'] if n - L + 1 <= 2 else 1.0)
    return p * total / (n * (n + 1) / 2.0)


def _v72_picker():
    """v7.2b PARK_RANK: the score that orders the parked band at the next parked admission"""
    if V72_RANK != 'interleave':
        return V72_RANK
    return V72_FIRST if _V72_STATE['n'] % 2 == 0 else ('index' if V72_FIRST == 'restart' else 'restart')


class _V72ParkSnapshot(_V62Snapshot):
    """v7.2b PARK_RANK: a parked game's queue entry (v7.1's parked band, below every never-started game) with its state at the
    park (the index's inputs); the gate's pump scores it at every admission and sets v72_prio (the band + its rank by the
    picker); the gate's re-score (ARC3_PRIORITY_REFRESH_QUEUE) reads the same value."""

    def __init__(self, state, gid, level, n_levels, swaps, band_k=1):
        _V62Snapshot.__init__(self, level, n_levels, swaps)
        self.band_k = int(band_k)                       # v7.2c: the game's park count on this level (this park included)
        self.parked = True
        self.state, self.gid = dict(state), gid
        self.t_park = time.monotonic()
        self.v72_prio = _V72_BAND
        self.v72_admit = None

    def v72_score(self, gate, now):
        """(restart_value, index or None) now: the time left and the time run from the gate's shared clock"""
        dl = getattr(gate, '_clock_deadline', None) if gate is not None else None
        t0 = getattr(gate, '_clock_start', None) if gate is not None else None
        tl = None if dl is None else max(0.0, dl - now)
        el = 0.0 if t0 is None else max(0.0, now - t0)
        rv = _v72_restart_value(self.level, self.n_levels, 0.0 if tl is None else tl)
        ix = None if V72_C is None else _s70.index(V72_C, dict(self.state, time_left_s=tl, elapsed_s=el))
        return rv, ix

    def priority(self, now, deadline):
        return self.v72_prio


def _v72_gate_pump(self):
    """v7.2b PARK_RANK, under the gate's lock: before a free lane is handed out, every parked entry (_V72ParkSnapshot) is scored
    by restart_value and by the index and ranked by this admission's picker within the parked band (ties: the gate's sequence);
    never-started games keep their band above it; then the previous pump. Each parked admission is noted (swap70.ADMITS kind
    'v72_parked'; the snapshot's v72_admit, which the game's v69-park event carries): the picker, the time left, both scores and
    both ranks of every parked candidate; the picker advances (interleave)."""
    pend = None
    if self._free > 0 and self._waiting:
        park = [(t, self._snapshots[t]) for _, t in self._waiting if isinstance(self._snapshots.get(t), _V72ParkSnapshot)]
        if park:
            import heapq
            now = time.monotonic()
            picker = _v72_picker()
            rows = []
            for t, sn in park:
                rv, ix = sn.v72_score(self, now)
                rows.append(dict(t=t, sn=sn, rv=rv, ix=ix))
            r_rv = {r['t']: i for i, r in enumerate(sorted(rows, key=lambda r: (-r['rv'], r['t'])))}
            r_ix = ({r['t']: i for i, r in enumerate(sorted(rows, key=lambda r: (-r['ix'], r['t'])))}
                    if all(r['ix'] is not None for r in rows) else {})
            if V72_BANDS:                                   # v7.2c: bands by park count, the picker (fifo: the sequence) within a band
                key = {'restart': lambda r: (-r['rv'], r['t']), 'index': lambda r: (-(r['ix'] or 0.0), r['t'])}.get(picker, lambda r: r['t'])
                byk = {}
                for r in rows:
                    byk.setdefault(min(int(r['sn'].band_k), V72_BAND_MAX), []).append(r)
                for k, rr in byk.items():
                    for i, r in enumerate(sorted(rr, key=key)):
                        r['rank_band'] = i
                        r['sn'].v72_prio = _V72_BAND + V72_BAND_W * (V72_BAND_MAX + 1 - k) + (len(rr) - i)
            else:
                rank = r_rv if picker == 'restart' else r_ix
                for r in rows:
                    r['sn'].v72_prio = _V72_BAND + (len(rows) - rank[r['t']])
            self._waiting = [((-self._snapshots[t].v72_prio, t) if isinstance(self._snapshots.get(t), _V72ParkSnapshot) else (q, t))
                             for q, t in self._waiting]
            heapq.heapify(self._waiting)
            pend = (picker, now, rows, r_rv, r_ix, set(self._admitted), len(self._waiting))
    _v72_prev_pump(self)
    if pend is None:
        return
    picker, now, rows, r_rv, r_ix, before, queued = pend
    got = sorted((r for r in rows if r['t'] in self._admitted and r['t'] not in before), key=lambda r: -r['sn'].v72_prio)
    if not got:
        return
    dl = getattr(self, '_clock_deadline', None)
    t0 = getattr(self, '_clock_start', None)
    cands = [dict(game=r['sn'].gid, level=r['sn'].level, n_levels=r['sn'].n_levels,
                  tokens_on_level=r['sn'].state.get('tokens_on_level'), actions_on_level=r['sn'].state.get('actions_on_level'),
                  cleared=len(r['sn'].state.get('cleared') or []), waited_s=round(now - r['sn'].t_park, 1), rv=r['rv'], ix=r['ix'],
                  rank_rv=r_rv.get(r['t']), rank_ix=r_ix.get(r['t']),
                  **({'band': r['sn'].band_k, 'rank_band': r.get('rank_band')} if V72_BANDS else {})) for r in sorted(rows, key=lambda r: r['t'])]
    for r in got:
        _V72_STATE['n'] += 1
        row = dict(epoch=round(time.time(), 1), kind='v72_parked', n=_V72_STATE['n'], picker=picker, game=r['sn'].gid,
                   level=r['sn'].level, time_left_s=None if dl is None else round(max(0.0, dl - now), 1),
                   elapsed_s=None if t0 is None else round(max(0.0, now - t0), 1), rv=r['rv'], ix=r['ix'],
                   rank_rv=r_rv.get(r['t']), rank_ix=r_ix.get(r['t']), parked=len(rows), waiting=queued,
                   newest=r['t'] == self._seq, waited_s=round(now - r['sn'].t_park, 1), candidates=cands,
                   **({'band': r['sn'].band_k, 'rank_band': r.get('rank_band')} if V72_BANDS else {}))
        r['sn'].v72_admit = row
        _s70.ADMITS.note(row)


_v72_prev_pump = ta._PriorityGate._pump
if V72_PARKX:
    ta._PriorityGate._pump = _v72_gate_pump

if V70_ON:                          # v7.0: the solver-side hooks, installed here (v62/solver.py unchanged): the play wrapper always,
    from v62 import solver as _v62s  # SOFT_CAP's budget methods, receipts and check with SOFT_CAP
    from inference.framework import solver as _fsv

    def _v70_soft_over(self, gid, spent):
        """v7.0 SOFT_CAP (the solver's check): the count at or past the cap does not end the game; the first pass is logged
        (v67-gamebudget 'soft_over') and kept"""
        with self.lock:
            cap = self.cap.get(gid)
            if cap is None or gid in self.ended or int(spent) < cap:
                return False
            so = self.__dict__.setdefault('soft_over_at', {})
            if gid not in so:
                so[gid] = (int(spent), cap)
                self._log(gid, reason='soft_over', tokens=int(spent), cap=cap)
            return True

    def _v70_soft_bind(self, gid, spent):
        """v7.0 SOFT_CAP (the agent, an eligible game waiting): the reserve first if still held (the first-cap rule); True
        when the game is still at or past its cap (the bind: 'soft_bind' event and count)"""
        with self.lock:
            cap = self.cap.get(gid)
            if cap is None or int(spent) < cap:
                return False
            if self.reserve_left > 0:
                self._release_reserve(gid, int(spent), cap)
                cap = self.cap.get(gid)
                if int(spent) < cap:
                    return False
            binds = self.__dict__.setdefault('soft_binds', {})
            binds[gid] = binds.get(gid, 0) + 1
            self._log(gid, reason='soft_bind', tokens=int(spent), cap=cap, over=int(spent) - cap, binds=binds[gid])
            return True

    _v62s.GameBudget.soft_over = _v70_soft_over
    _v62s.GameBudget.soft_bind = _v70_soft_bind
    if V70_SOFTCAP:
        _v70_finish0, _v70_summary0 = _v62s.GameBudget.finish, _v62s.GameBudget.summary

        def _v70_finish(self, gid, spent, how):
            out = _v70_finish0(self, gid, spent, how)
            if out is not None:                         # v7.0 SOFT_CAP receipts: the tokens past the cap, the first pass, binds
                with self.lock:
                    out.update(soft_cap_overrun_tokens=max(0, int(spent) - int(out['game_cap_final'])),
                               soft_cap_over_at=list((self.__dict__.get('soft_over_at') or {}).get(gid) or []) or None,
                               soft_cap_binds_budget=(self.__dict__.get('soft_binds') or {}).get(gid, 0))
            return out

        def _v70_summary(self):
            out = _v70_summary0(self)
            with self.lock:
                out['soft_cap'] = dict(games_over=len(self.__dict__.get('soft_over_at') or {}),
                                       binds=sum((self.__dict__.get('soft_binds') or {}).values()),
                                       beyond_base_caps=sum(max(0, p['used'] - self.base) for p in out['per_game'].values()))
            return out

        def _v70_token_limit_reached(self):
            """v7.0 SOFT_CAP: the pool cap never ends the game here (the agent's bind parks it); dfranzen's own limit stays"""
            budget, run = _v62s.GAMEBUDGET, self.game.game_run
            if budget is not None and run is not None:
                budget.soft_over(run.game_id, _v62s._gamebudget_spent(self))
            return _v62s._original_token_limit(self)

        _v62s.GameBudget.finish, _v62s.GameBudget.summary = _v70_finish, _v70_summary
        _fsv._HarnessGameSession.token_limit_reached = _v70_token_limit_reached
    _v70_play0 = _fsv._HarnessGameSession.play

    def _v70_play(self):
        """v7.0: this thread's session for INDEX_ADMIT's never-started entry (TAAF has started the game: its number of
        levels is known); a finished game leaves the registry's running set and drops its claim"""
        gid = getattr(getattr(self.game, 'game_run', None), 'game_id', None)
        _s70.TLS.session = self
        try:
            return _v70_play0(self)
        finally:
            if gid is not None:
                _s70.REGISTRY.publish(gid, running=False, finished=True)
                _s70.CLAIMS.drop(gid)

    _fsv._HarnessGameSession.play = _v70_play
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


def _v64_compile_error(code):
    """v6.4 A5: None when recovered code compiles (the sandbox runs the same interpreter), else "Type at line N"."""
    try:
        compile(code, '<recovered>', 'exec')
        return None
    except SyntaxError as exc:
        return '%s at line %s' % (type(exc).__name__, exc.lineno)
    except ValueError as exc:
        return type(exc).__name__


def code_from_arguments(parsed):
    """v6.1 item 1: a python call whose single argument came under another name ({"command": "..."})."""
    if not isinstance(parsed, dict) or str(parsed.get('code') or '').strip():
        return None
    values = [v for v in parsed.values() if isinstance(v, str) and v.strip()]
    return values[0] if len(values) == 1 else None


def call_defect(raw):
    """round 5i (item 98; 391: recovered calls ran in series in which each call dropped the same tag, wy30 66, s5i5 26 of
    26, wa30 13, and the generic note did not stop them): what the last python call's markup lacked, read from the
    decoded text after the reasoning; None when it is none of these."""
    if not raw:
        return None
    text = raw.rsplit('</think>', 1)[-1]
    at = text.rfind('<tool_call>')
    if at < 0:
        return None
    fn = text.find('<function=python>', at)
    if fn < 0:
        return None
    body = text[fn + len('<function=python>'):]
    if '<parameter=' not in body:
        return 'the code was not preceded by <parameter=code>'
    if '<parameter=code>' not in body:
        return 'its parameter tag was not <parameter=code>'
    missing = [tag for tag in ('</parameter>', '</function>', '</tool_call>') if tag not in body]
    if missing:
        return 'it ended without ' + (missing[0] if len(missing) == 1 else ', '.join(missing[:-1]) + ' and ' + missing[-1])
    return None


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
    # v6.2 (run A, tu93): the refill at a level completion is not a budget step
    changed = [a for a in rows if a.get('v61_edge') and not a.get('level_completed')]
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
    # round 5d (C50 g021, g034: counts did not add up); round 5l (A7, Fable 1.5; 392b: 106 of 160 lines still did not):
    # each action's record keeps only its 3 most common transitions, so the remainder is the total minus the 2 shown
    rest = total - sum(k for _, k in top)
    if rest > 0:
        trs += ', other colour changes x%d' % rest
    # round 4 (A2/A3 lp85: a single changed cell in a column bar printed "row N", read as a top-edge bar)
    if len(rs) == 1 and len(cs) == 1:
        where = 'row %s, col %s' % (sorted(rs)[0], sorted(cs)[0])
    elif len(rs) == 1:
        where = 'row %s' % sorted(rs)[0]
    elif len(cs) == 1:
        where = 'col %s' % sorted(cs)[0]
    else:
        where = 'several rows and columns'
    noop = sum(1 for a in changed if a.get('gameplay_changed') is False and not a.get('reverted'))
    text = '%d of %d executed actions changed %d edge cells (%s; %s)' % (len(changed), len(rows), total, trs, where)
    if noop:                         # round 5l (A3 option b): these changed the band only (BAND-ONLY CHANGE), not NO-OPs
        text += '; %d of them changed nothing in the interior (BAND-ONLY CHANGE)' % noop
    joined = sum(a.get('band_connected') or 0 for a in changed)      # round 5h (G3)
    if joined:
        text += ('; %d of the changed edge cells %s connected to cells that changed in the interior'
                 % (joined, 'is' if joined == 1 else 'are'))
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
        if ' was not retained: __v62_kept_previous__: ' in line and line.startswith('Your function '):
            # round 5l (item 110, A6): a new definition that was not saved; the earlier version stays in the library
            name, why = line[len('Your function '):].split(' was not retained: __v62_kept_previous__: ', 1)
            for hint in (' Pass snippet-specific data as arguments or redefine it if needed.',):
                why = why[:-len(hint)] if why.endswith(hint) else why
            out.append(_fill(TEXTS['retention_kept_previous'], name=name, reason=why.rstrip('.')))
            continue
        m = _RET_SAVED.match(line)
        if m:
            if m.group(2) == 'constant':                     # v6.1 item 8 (retention.py reports NAME(constant))
                out.append('Saved the constant `%s` to the program library.' % m.group(1))
            else:
                out.append(_fill(TEXTS['retention_saved'], name=m.group(1), params=m.group(2)))
            continue
        m = _RET_REJECTED.match(line)
        if m and m.group(2) == '__v62_kept_rebound__' and 'retention_kept' in TEXTS:   # round 5h (G2)
            out.append(_fill(TEXTS['retention_kept'], name=m.group(1)))
            continue
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


_PY_BUILTINS = frozenset(dir(__import__('builtins')))


PROVIDED_NAMES = ('current_frame', 'previous_frame', 'history', 'transitions', 'last_transition', 'last_action',
                  'last_action_frame', 'last_action_call_result', 'level_start_frame', 'attempt_start_frame', 'valid_actions',
                  'action', 'frame_diff', 'last_animation_frames', 'last_animation_timeline', 'programs', 'read_code',
                  'save_summary', 'death_ledger', 'grid_cells', 'color_counts', 'find_cells', 'bounding_box', 'grid_diff',
                  'connected_components', 'find_objects', 'find_object', 'object_position', 'object_move',
                  'StaleStateActionError', 'InvalidActionError', 'ActionCapError', 'TerminalStateActionError') + (
    ('show_board',) if sandbox_ext.SHOW_BOARD else ())                                  # round 5m (item 124)


def _v6_error_text(err, library=()):
    """The spec's timeout text for the sandbox's bare timeout error (it may already have executed actions)."""
    if isinstance(err, str) and err.startswith('Tool timed out after'):   # v6.2: only the wall-clock backstop now
        return TEXTS['timeout_wall']
    if err == sandbox_ext.CPU_KILLED:                                     # round 5m (item 123)
        return TEXTS.get('compute_limit_lost', err)
    m = re.search(r"(?m)^NameError: name '([^']+)' is not defined", err) if isinstance(err, str) else None
    if m:
        if m.group(1) in _PY_BUILTINS and 'blocked_builtin_note' in TEXTS:
            # round 5h (Fable Q4.5, 386 lx85 T80: `globals`): a builtin the sandbox leaves out, not a lost variable
            return err + '\n' + _fill(TEXTS['blocked_builtin_note'], name=m.group(1))
        # round 5l (B6, 392b tq62, ce01 and many games: wrong guesses of provided names): a close provided or library name
        import difflib
        close = difflib.get_close_matches(m.group(1), list(PROVIDED_NAMES) + list(library or ()), n=1, cutoff=0.8)
        if close:
            return err + '\n' + _fill(TEXTS['nameerror_close'], name=close[0],
                                      where='' if close[0] in PROVIDED_NAMES else ' (in your program library)')
        # v6.2 round 3 (b) (A2 ft09: 4 of 6 errors, m0r0, tu93): the reason a name from an earlier call is missing
        return err + '\n' + TEXTS['nameerror_note']
    m = re.search(r"(?m)^ImportError: Module '([^']+)' is not allowed in the sandbox\.", err) if (
        V70H_ALIASES and isinstance(err, str)) else None
    if m:                                             # v7.0 item 13: the most frequent refused imports (the census) get a fact
        key = _V70H_IMPORT_NOTES.get(m.group(1).split('.')[0])
        if key:
            return err + '\n' + _fill(TEXTS[key], name=m.group(1))
    if V75_POLISH and isinstance(err, str):          # v7.5 TOOL-3: subscripting a frame view or a history entry
        m = re.search(r"(?m)^TypeError: '(\w+)' object is not subscriptable", err)
        if m and m.group(1) == 'FrameView':
            return err + '\n' + TEXTS['v75_frame_subscript']
        if m and m.group(1) in ('HistoryEntryView', '_V6LazyEntry', '_V6StubEntry'):
            return err + '\n' + TEXTS['v75_entry_subscript']
    return err


_V70H_IMPORT_NOTES = {'tools': 'v70h_import_helpers', 'grid': 'v70h_import_helpers', 'objects': 'v70h_import_helpers',
                      'programs': 'v70h_import_library', 'library': 'v70h_import_library', 'inspect': 'v70h_import_inspect',
                      'time': 'v70h_import_time', 'history': 'v70h_import_global', 'transitions': 'v70h_import_global',
                      'current_frame': 'v70h_import_global', 'previous_frame': 'v70h_import_global'}


def _printed_strings(tool_calls):
    """v6.2 round 3: the text of print(...) calls of a message's python code, one per line (a summary the model wrote
    with print). Round 5c (B3 re86, sk48: summaries ending in "tokens W id", "shape step level id"): only prints whose
    arguments are all string literals; a print of computed values gives a blank line, which ends a captured summary."""
    out = []
    for call in tool_calls or []:
        try:
            args = (call.get('function') or {}).get('arguments')
            code = (json.loads(args) if isinstance(args, str) else (args or {})).get('code') or ''
            tree = ast.parse(code)
        except Exception:  # noqa: BLE001
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'print':
                parts = [a.value for a in node.args if isinstance(a, ast.Constant) and isinstance(a.value, str)]
                out.append(' '.join(parts) if parts and len(parts) == len(node.args) else '')
    return '\n'.join(out)


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


def _lazy_enabled():
    from v62 import lazy_history
    return lazy_history.ENABLED


def install_history_scope():
    """Run B: the sandbox's history leaves out the calling agent's hidden ranges. Lazy history only (its entries carry
    the host index as frame_ref); run_v6.py installs this after lazy_history.install()."""
    inner = ta._ascii_history_view_payload

    def payload(history_entries):
        out = inner(history_entries)
        if isinstance(out, dict):                # round 5k: the paged payload applies the hidden ranges itself
            return out
        agent = getattr(sandbox_ext.CURRENT, 'agent', None)
        hidden = getattr(agent, '_v61_hidden', None) if agent is not None else None
        if not hidden or not _lazy_enabled():
            return out
        return [item for item in out
                if not any(lo <= int(item.get('frame_ref', -1)) < hi for lo, hi in hidden)]
    ta._ascii_history_view_payload = payload


# round 5c (B3 vc33 L4 T38: "moved anything" contradicted a NO-OP that changed 22 cells of the outer band)
_NOOP_LAST = "It is the last action in the executed list above; only the actions before it changed anything in the interior."
_NOOP_NOT_LAST = ("It is executed action %d of %d in this call; the actions after it came from later action(...) "
                  "calls in the same snippet.")
_NOOP_FIRST = "It was the first action of the batch, so no action in the batch changed anything in the interior."
if variants.v65_batch():           # v6.5 C6: the stopping NO-OP follows another NO-OP; earlier actions may be BAND-ONLY
    _NOOP_LAST = "It is the last action in the executed list above."
assert _NOOP_LAST in TEXTS['noop_stop'], 'noop_stop anchor'


_BULLET = re.compile(r'^(?:[-*\u2022\u2013]|\d{1,2}[.)])\s+')   # round 5g: a list item ('- ', '* ', '1. ', '2) ')


def _rule_body(text, m, nxt, nxt_item=None):
    """The summary after a Rule header match: the header line's remainder and up to RULE_LINES following lines, to a
    blank line or the sentence that names the next level. '' when the body is empty, a placeholder ('...', 'TBD') or
    shorter than 20 letters (round 5d), so the capture keeps looking.
    Round 5g (user: a bullet list of up to 10 items): when the body is a list (the line after the header, after any blank
    lines when the header line is empty or ends with ':', is an item: '- ', '* ', '1. ', '2) '), its lines are kept as
    written, up to RULE_LINES items; blank lines inside the list are skipped when another item follows; the first line
    that is not an item ends the list. In a list the next-level cut is `nxt_item`: the next level named at the start of
    a sentence ('... fill all boxes. Now level 2. The board ...', B3 ka59) ends the item and the list there, while a
    mid-sentence mention stays. A non-empty header remainder is kept as the first line."""
    lines = text[m.end():].split('\n')
    if re.fullmatch(r'\s*(?:\.{2,}|\u2026|tbd|todo|-+)?\s*\)?\s*', lines[0], re.I) and lines[0].strip():
        return ''                                      # 'Rule L1: ...' and the like: a stub, not the summary
    head = lines[0].strip().strip('*').strip()
    if head.endswith(':'):          # round 5h (G1, 386 kx59: 'Rule summary (for the required "Rule L6:" lines):'): the
        head = ''                   # whole line is the header; the body starts on the next line
    j = 1
    if not head:
        while j < len(lines) and not lines[j].strip():
            j += 1
    if j < len(lines) and _BULLET.match(lines[j].strip()) and (j == 1 or not head or head.endswith(':')):
        rx = nxt_item or nxt

        def cut(line):                                 # (kept part, the list ends here)
            mb = _BULLET.match(line)
            body = line[mb.end():] if mb else line
            hit = rx.search(body)
            if not hit:
                return line, False
            pre = body[:hit.start()]
            c = max(pre.rfind('.'), pre.rfind('!'), pre.rfind('?'), pre.rfind(')'))
            kept = pre[:c + 1].strip() if c >= 0 else ''
            return ((line[:mb.end()] + kept) if (mb and kept) else kept), True

        out, items, k = [], 0, j
        if head:
            kept, stop = cut(head)
            if kept:
                out.append(kept)
                items += bool(_BULLET.match(kept))
            if stop:
                k = len(lines)
        while k < len(lines) and items < (RULE_ITEMS77 if V77 else RULE_LINES):   # v7.7 (A): all the list's items
            line = lines[k].strip()
            if not line:
                k2 = k + 1
                while k2 < len(lines) and not lines[k2].strip():
                    k2 += 1
                if k2 < len(lines) and _BULLET.match(lines[k2].strip()):
                    k = k2
                    continue
                break
            if not _BULLET.match(line):
                break
            kept, stop = cut(line)
            if kept and _BULLET.sub('', kept).strip():
                out.append(kept)
                items += 1
            if stop:
                break
            k += 1
        rule = '\n'.join(out)
        return rule if len(re.findall(r'[A-Za-z]', rule)) >= 20 else ''
    body = []
    start = 0 if head else j        # round 5h: after an empty header remainder the prose starts past any blank lines
    for i, line in enumerate(lines[start:start + 1 + RULE_LINES]):
        line = line.strip()
        if i and not line:
            break
        hit = (nxt_item or nxt).search(line)
        if hit:
            head = line[:hit.start()]
            if nxt_item is None:
                cut = max(head.rfind('. '), head.rfind('! '), head.rfind('? '), head.rfind(') '))
                head = head[:cut + 1] if cut >= 0 else ''
            # round 5l (A2, 392b g512: ten items cut to one sentence at "Level 5"): in prose too, only the next level named
            # at the start of a sentence ends the summary; the match starts after the previous sentence's end
            body.append(head.strip())
            break
        body.append(line.strip('*').strip() if i == 0 else line)
    rule = ' '.join(b for b in body if b)
    if rule.endswith(')') and rule.count(')') > rule.count('('):     # '(Rule L1: ... )' (C50 g001)
        rule = rule[:-1].rstrip()
    return rule if len(re.findall(r'[A-Za-z]', rule)) >= 20 else ''


def _cap_rule(rule, count=None):
    """Round 5g (user: at most 4,000 tokens per summary, truncated beyond): `count` is the served tokenizer's text -> tokens
    (an estimate of 4 characters per token without it). A list keeps whole items and ends with '- ...'; prose is cut at a
    word and marked ' ...' (round 5c)."""
    count = count or (lambda text: (len(text) + 3) // 4)
    if count(rule) <= SUMMARY_TOKENS:
        return rule
    if '\n' in rule:
        kept = []
        for line in rule.split('\n'):
            if count('\n'.join(kept + [line, '- ...'])) > SUMMARY_TOKENS:
                break
            kept.append(line)
        if kept:
            return '\n'.join(kept + ['- ...'])
        rule = rule.split('\n')[0]              # round 5h review: a first item over the budget is cut like prose
    words = rule.split(' ')
    lo, hi = 0, len(words)                      # the longest word prefix that fits with the marker
    while lo < hi:
        mid = (lo + hi + 1) // 2
        if count(' '.join(words[:mid]) + ' ...') <= SUMMARY_TOKENS:
            lo = mid
        else:
            hi = mid - 1
    return ' '.join(words[:lo]) + ' ...'


def _strip_level_tag(rule, n):
    """Round 5h (G1, 386 dy22, lx20, rx1l: "- L4: - L4:"): a summary that repeats its own level tag at its start."""
    lines = rule.split('\n')
    lines[0] = re.sub(r'(?i)^((?:[-*\u2022]|\d{1,2}[.)])\s+)?(?:rule\s*)?L\s*%d\s*[:.)\-]\s*' % int(n), lambda m: m.group(1) or '',
                      lines[0]).strip()
    rule = '\n'.join(l for l in lines if l.strip())
    return rule if len(re.findall(r'[A-Za-z]', rule)) >= 20 else ''


def _norm_line(line):
    return re.sub(r'\W+', ' ', _BULLET.sub('', line.strip())).strip().lower()


def _merge_rule_lines(found):
    """Round 5h (G1, 386 cx04: 1 of 5 "Rule L1:" lines kept; gx0t 1 of 4): several one-line summaries of the same level
    in one text are one list (in order, without repeats, up to RULE_LINES); otherwise the last substantive one. Round 5l
    (A2, 392b rv63: six overlapping duplicates): a line contained in another is dropped, and so is a repeated line."""
    if len(found) > 1:              # round 5l (A2, 392b g531): a body inside a longer one (an item of the list) goes
        keys = [_norm_line(f.replace('\n', ' ')) for f in found]
        found = [f for i, f in enumerate(found)
                 if not any(j != i and len(keys[j]) > len(keys[i]) and keys[i] in keys[j] for j in range(len(found)))]
    if len(found) > 1 and all('\n' not in f for f in found):
        items = list(dict.fromkeys(found))
        items = [x for x in items if not any(_norm_line(x) != _norm_line(y) and _norm_line(x) in _norm_line(y)
                                             for y in items)][:(RULE_ITEMS77 if V77 else RULE_LINES)]   # v7.7 (A)
        return '\n'.join(f if _BULLET.match(f) else '- ' + f for f in items)
    seen, out = set(), []
    for line in found[-1].split('\n'):
        key = _norm_line(line)
        if key and key in seen:
            continue
        seen.add(key)
        out.append(line)
    return '\n'.join(out)


def _header_ok(header, text='', start=0):
    """Round 5l (A2, 392b yd01: a quoted header fragment): a summary header is not inside quotes. List items stay headers
    (387b, 390: one "- Rule Lk: ..." line per level); g531's mid-list match is dropped in _merge_rule_lines instead."""
    h = header.lstrip()
    pre = re.split(r'(?i)\b(?:rules?|level)\b', h, 1)[0]
    pre = _BULLET.sub('', pre)
    return not re.search('["\'\u201c\u201d\u2018\u2019`]', pre)


def _rule_display(lvl, rule):
    """Round 5g: one level's summary in the restart opener; a list shows its items indented under '- L{n}:'."""
    if '\n' not in rule:
        return f"- L{lvl}: {rule}"
    lines = rule.split('\n')
    if _BULLET.match(lines[0]):
        return '\n'.join([f"- L{lvl}:"] + ['  ' + l for l in lines])
    return '\n'.join([f"- L{lvl}: {lines[0]}"] + ['  ' + l for l in lines[1:]])


REFUSAL_REASONS = {
    'summary_turn': 'actions are not executed in the turn before a level restart',   # round 5l (item 111)
    # round 5l (A7, Fable 3.12; 392b fn01: 3 of 9 refused calls tallied): refusals that carry an error and a stop reason
    'invalid_action': 'an action not among the valid actions',
    'call_cap': 'the limit of %d executed actions per Python call' % variants.CALL_CAP,
    'game_over': 'the game reported GAME OVER (only RESET executes then)',
    'reset_not_first': 'RESET was not the first executed action of the snippet',
    'stale_state': 'the previous action changed nothing on the board (stale-state refusal)',   # round 5l (A3 option b)
    'known_death': 'a recorded fatal continuation',
    'repeated_action_in_state': 'the same action from the same board',
}


# v6.9 REX HIDE_RESET (decision 49; apply_rex69.py item R69HR): arm B (ARC3_V69_HIDE_RESET=1) needs EXPOSE_RESET off (run_v6 sets it);
# its refusal reasons. Off (default): nothing changes.
HIDE_RESET69 = variants.v69_hide_reset()
if HIDE_RESET69:
    if ta.reset_exposed():
        raise RuntimeError('ARC3_V69_HIDE_RESET=1 needs EXPOSE_RESET off (run_v6 sets it off with the switch)')
    REFUSAL_REASONS['reset_hidden'] = TEXTS['hr69_reason_reset']
    REFUSAL_REASONS['hr69_game_over'] = TEXTS['hr69_reason_game_over']
    REFUSAL_REASONS['game_over'] = TEXTS['hr69_reason_game_over_stop']
if V70H_RESET0:                                           # v7.0 item 19c: the reason of a refused RESET in a 0-action attempt
    if HIDE_RESET69 or not ta.reset_exposed():
        raise RuntimeError('ARC3_V70H_RESET0 needs RESET exposed (EXPOSE_RESET on, ARC3_V69_HIDE_RESET=0)')
    REFUSAL_REASONS['reset_zero'] = TEXTS['v70h_reason_reset0']


def _refusal_text(refused, reasons=None, shown=None):
    """Round 5d: '2 action() calls in this Python call were refused and executed nothing: <reason> (1), ...'. v7.0 item 16
    (ARC3_V70H_REFUSAL_ONCE): `shown` is the call's error text; when it already holds the first refused call's message, the
    message is not repeated here."""
    counts = {}
    for r in refused:
        why = (reasons or REFUSAL_REASONS).get(r.get('stop_reason')) or 'an invalid action'   # v6.7: soft notes turn
        counts[why] = counts.get(why, 0) + 1
    n = len(refused)
    text = '%d action() call%s in this Python call %s refused and executed nothing: %s.' % (
        n, '' if n == 1 else 's', 'was' if n == 1 else 'were', '; '.join('%s (%d)' % kv for kv in counts.items()))
    # round 5h (G5, 386 ly52: 537 refusals read as "the game refuses MOUSE"): the first refused call and its message
    first = refused[0]
    acts = first.get('requested_actions') or []
    msg = ' '.join(str(first.get('error') or first.get('stop_detail') or '').split())
    if first.get('stop_reason') == 'reset_not_first' and msg:   # round 5l (B7): his text says "This call executed no actions."
        msg = 'RESET must be the first game action of a Python call; that action() call executed nothing.'
    if shown and msg and msg in ' '.join(str(shown).split()):   # v7.0 item 16: the error states it already
        msg = ''
    if len(msg) > 300:
        msg = msg[:300].rsplit(' ', 1)[0] + ' ...'
    if acts or msg:
        text += ' The first %s%s' % (('was ' + ', '.join(acts[:3]) + ('' if len(acts) <= 3 else ', ...')) if acts else 'one',
                                      (': ' + msg) if msg else '')
        text += '' if text.endswith(('.', '!', '?')) else '.'
    return text


def _v62_summary_from_records(recs, level):
    """Round 5l (A7): a step summary from the host's per-action records of one Python call."""
    shown = [str(r.get('action_display') or '?') for r in recs]
    return dict(start_action_num=None, end_action_num=None, executed_count=len(recs), executed_actions=shown, level=level,
                level_transition=any(bool(r.get('level_completed')) for r in recs),
                run_complete=any(bool(r.get('run_complete')) for r in recs),
                game_over=any(bool(r.get('game_over')) for r in recs), game_over_trace=[],
                fatal_action=next((shown[i] for i, r in enumerate(recs) if r.get('game_over')), ''),
                board_changed=any(bool(r.get('board_changed')) for r in recs),
                gameplay_changed=any(bool(r.get('gameplay_changed')) for r in recs),
                stop_reason=None, no_op_stops=[], animation=None, animation_action=None)


V65_CLOCK = os.environ.get('ARC3_V65_CLOCK', '0') == '1'    # v6.5 CLOCK (user via IE 10-17): the run clock is not shown
_OPENER_KEEP = ('Step ', 'Reminder:') + (('Run time left:',) if V65_CLOCK else ())


def _opener_key(text):
    """Round 5l (C5): an opener's report without its step, clock and reminder lines."""
    body = [l.strip() for l in str(text or '').split('\n') if l.strip() and not l.startswith(_OPENER_KEEP)]
    return '\n'.join(body) or None


def _collapse_opener(text):
    lines = str(text or '').split('\n')
    return '\n'.join([TEXTS['same_report']] + [l for l in lines if l.startswith(_OPENER_KEEP)])


_V65_LEDGER = ('Continuations that ended in GAME OVER', 'Each listed continuation ended in GAME OVER')


def _v65_report_sig(text, mode):
    """v6.5 C1: a report's signature. exact: _opener_key; norm: also without the death ledger (its header, entries and
    note) and with every digit run replaced by '#'; acts: nothing (the actions alone)."""
    if mode == 'acts':
        return ''
    if mode == 'exact':
        return _opener_key(text)
    out = []
    for l in str(text or '').split('\n'):
        l = l.strip()
        if not l or l.startswith(_OPENER_KEEP) or l.startswith(_V65_LEDGER):
            continue
        n = re.sub(r'[0-9]+', '#', l)
        if re.match(r'- # action\(s\): .* -> GAME OVER', n):       # a ledger entry
            continue
        out.append(n)
    return '\n'.join(out)


def _v65_loop(sigs, pmax=4):
    """(p, k): the smallest period p in 1..pmax whose last block of p signatures repeats k >= 3 times at the end; (0, 0).
    v6.7 LOOP: pmax LOOP_PMAX (8) under ARC3_V67_LOOP=1; C1's 4 otherwise."""
    for p in range(1, pmax + 1):
        if len(sigs) < 3 * p:
            continue
        block, k = sigs[-p:], 1
        while len(sigs) >= (k + 1) * p and sigs[-(k + 1) * p:len(sigs) - k * p] == block:
            k += 1
        if k >= 3:
            return p, k
    return 0, 0


def _lines_text(n):
    return '%d line%s' % (n, '' if n == 1 else 's')


def _game_over_text(history_entries):
    """Round 5l review (Fable T1): GAME OVER at a turn start after the model's RESET, or after the automatic restart's."""
    entries = list(history_entries or [])
    last = (getattr(entries[-1], 'result', None) or {}) if entries else {}
    return TEXTS['game_over_auto_reset'] if last.get('automatic') else TEXTS['game_over_no_restart']


def _duration_text(seconds):
    """Round 5l (B1, 392b: "Run time left: N min" read as seconds in g502, g507, g518, g522, g001, ...): hours and minutes,
    each with its unit."""
    m = max(0, int(round(float(seconds) / 60.0)))
    if m < 1:
        return 'less than 1 min'
    h, m = divmod(m, 60)
    return ('%d h %d min' % (h, m)) if h else ('%d min' % m)


def _v62_game_over_now(session):
    """Round 5l (A1): the engine reports GAME OVER at the start of a turn only after a RESET that ended in one (the solver
    restarts the level automatically after any other game over)."""
    try:
        from inference.framework import solver as _sv
        return session is not None and bool(_sv._is_engine_game_over(session.game))
    except Exception:  # noqa: BLE001
        return False


def _v62_with_note(content, note):
    """Round 5h review (387b: recovered malformed calls ran in unbroken series, bx35 23, sy80 16, dy22 15, sxi5 9; 386 sx25
    12; the recovery was silent, so the model kept copying its own markup): the note goes first in that call's tool result."""
    if isinstance(content, list):
        return [dict(type='text', text=note)] + list(content)
    return note + '\n' + str(content)


class _NoActions:
    """Round 5l (item 111): the summary turn's step_env: queries pass, actions are refused (the level restarts after it)."""

    def __init__(self, inner, refusal='summary_turn_refusal'):
        self.__self__ = getattr(inner, '__self__', None)
        self._inner = inner
        self._v67_refusal = refusal                     # v6.7 SOFT: the refusal's TEXTS key (the soft notes turn's)

    def __call__(self, arguments):
        if str((arguments or {}).get('query') or '').strip() == 'animation':
            return self._inner(arguments)
        return {"executed": False, "error": TEXTS[self._v67_refusal], "stop_reason": "summary_turn",
                "executed_actions": [], "action_trace": [], "gameplay_changed_per_action": [],
                "requested_actions": [str((a or {}).get('action', a)) if isinstance(a, dict) else str(a)
                                      for a in ((arguments or {}).get('actions') or [])]}


_NOTES_NEVER = re.compile(r'(?!x)x')


def _capture_notes(texts, n):
    """Round 5l (item 111): the fallback for a summary turn without save_summary: a "Notes L{n}:" block in the reply text
    or printed strings (not the reasoning)."""
    pattern = r'(?im)^[^\n]{0,40}\bnotes?\b[^\n:]{0,30}\bL\s*%d\b[^\n:]{0,40}:' % int(n)
    for text in list(texts or [])[:2]:
        for m in re.finditer(pattern, text or ''):
            if not _header_ok(m.group(0), text or '', m.start()):
                continue
            body = _rule_body(text, m, _NOTES_NEVER)
            if body:
                return body
    return ''


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
        self._v62_noact_tokens = 0                      # v6.3 item B: generated tokens since the last executed action
        self._v62_noact_warned = 0                      # v6.3 item B: warnings sent in the current stretch
        self._v62_restart_reason = None                 # v6.3: 'tokens' | 'noact' | 'position' of the pending restart
        self._v62_noact_late = False                    # v6.3 item C: the no-action limit fell in the last 25 min
        self._v61_last_error = None                     # item 1: breaker
        self._v61_error_streak = 0
        self._v61_restarts = 0
        self._v61_hidden = []                         # host history index ranges [lo, hi) left out of the sandbox's history
        self._v61_hidden_level = None
        self._v62_swaps = {}                          # level -> swap-outs on it (allocator queue order)
        self._v62_rule_turns = 0                      # completions left to look for the level-up summary
        # v6.2 fix (an unforeseen error found in run A, 2026-10-05): decision 3's repeat-state guard (2a) never ran in v6.1.
        # His guard hook returns at once without the NoopRepeatGuard store, which only ARC3_NOOP_REPEAT_GUARD or
        # ARC3_DEATH_REPEAT_GUARD create (both off by decision), and our analyze() did not attach the hook. With
        # ARC3_REPEAT_STATE_GUARD on, create the store here with the known-no-op and known-death refusals left off
        # (_noop_guard_noops_enabled / _noop_guard_deaths_enabled stay False), so only the repeat-state check runs.
        if self._noop_repeat_guard is None and ta._repeat_state_guard_enabled():
            from inference.agent.noop_repeat_guard import NoopRepeatGuard
            self._noop_repeat_guard = NoopRepeatGuard()
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
        self._v62_fn_window = None                # round 5l (item 108): (level, last completion) save_summary may fill
        self._v62_rule_fn = set()                 # levels whose summary came through save_summary
        self._v62_saved_in_call = False           # this Python call saved a summary (yield / no-action exemption)
        self._v62_attempt_request = None          # round 5l (item 111, D1): the level whose attempt summary is asked for
        self._v62_attempt_notes = {}              # level -> [notes the model saved at the end of an abandoned conversation]
        self._v62_library_log = []                # round 5l (item 110, S4): every library change, append-only
        self._v62_summary_state = None            # round 5l (item 111): None | 'requested' | 'done' at a token restart
        self._v64_notes_retry = None              # v6.4 A1 (12-h audit): the reason of this notes turn's one retry
        self._v62_notes_asked = None              # the level whose last restart had a summary turn
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
        known = sorted({l for l in levels if isinstance(l, int)})     # round 5d (C50 g008, g018, g023: "levels 1")
        return _fill(TEXTS['library_count'], n=len(names), s='' if len(names) == 1 else 's',   # round 5l (item 113)
                     them='it' if len(names) == 1 else 'them',
                     their='its' if len(names) == 1 else 'their', ls='' if len(names) == 1 else 's',   # v7.5 TXT-3 (the text has them)
                     levels=('level ' if len(known) == 1 else 'levels ') + _levels_text(known))

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
        recs = list(s.get('v61_actions') or [])
        trace = []
        if recs and len(recs) == len(names) and (n > 1 or V70H_FACTS or any(r.get('game_over') or r.get('level_start_match') or r.get('repeat_in_call')
                                                              or r.get('reverted') for r in recs)):
            # v6.2 fix (run A: sp80, cn04, ka59, lf52, lp85, dc22, r11l, s5i5, sc25): every action's own verdict from the
            # per-action records of all the call's action(...) calls. Before, the line came from one stop's batch trace
            # or from the call's single gameplay_changed flag, so no-ops read "effect", a batch ending in REVERTED read
            # NO-OP throughout, and REVERTED / BACK AT LEVEL START never showed.
            from inference.framework import solver as _sv
            trace = ['%d: %s -> %s' % (i + 1, names[i], _sv._action_verdict(r)) for i, r in enumerate(recs)]
        if not trace:
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
        # round 5l (A3 option b): the count is of actions that changed nothing in the interior (NO-OP or BAND-ONLY CHANGE)
        return ', '.join('%d action%s (%d changed nothing in the interior)' % (n, '' if n == 1 else 's', m)
                         for n, m in attempts[-5:]) or '?'

    @staticmethod
    def _v62_fatal_position(history_entries):
        """v6.2 round 5b (Fable; facts only): where the attempt's last action stands in the attempt, by kind and overall
        ("the 5th SPACE and the 23rd action of the attempt"). The attempt starts after the latest level completion, RESET
        or automatic restart before it. '' when unknown."""
        entries = list(history_entries or [])
        if len(entries) < 2:
            return ''
        names = []
        for e in reversed(entries[:-1]):                  # entries[-1] is the automatic restart
            r = getattr(e, 'result', None) or {}
            if r.get('level_completed') or r.get('automatic') or r.get('action_name') == 'RESET':
                break
            names.append(re.match(r'[A-Z0-9]+', str(getattr(e, 'action', '') or '')))
        names = [m.group(0) for m in names if m]
        if not names:
            return ''
        kind, n = names[0], len(names)
        k = sum(1 for x in names if x == kind)

        def nth(i):
            return '%d%s' % (i, 'th' if 10 <= i % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(i % 10, 'th'))
        return ' It was the %s %s and the %s action of the attempt.' % (nth(k), kind, nth(n))

    @staticmethod
    def _v61_fatal_interior(history_entries):
        entries = list(history_entries or [])
        r = (getattr(entries[-2], 'result', None) or {}) if len(entries) >= 2 else {}
        changed = r.get('gameplay_changed')
        return 'changed' if changed is True else 'did not change' if changed is False else 'may have changed'

    def _v61_clock_line(self, session=None):
        if not V65_CLOCK:                   # v6.5 CLOCK: no run-time budget in any observation (harness timing unchanged)
            return ''
        session = session or self._v61_session()
        try:
            remaining = session.timing_payload().get('time_remaining_seconds')
        except Exception:  # noqa: BLE001
            remaining = None
        if remaining is None:
            return ''
        return _fill(TEXTS['run_clock'], left=_duration_text(remaining))

    def _v62_restart_time_ok(self, level):
        """Round 5c: a level restart (and its swap-out) only with at least RESTART_MIN_LEFT seconds of run time left; the
        first skip on a level is logged and counted."""
        try:
            remaining = self._v61_session().timing_payload().get('time_remaining_seconds')
        except Exception:  # noqa: BLE001
            return True
        if remaining is None or float(remaining) >= RESTART_MIN_LEFT:
            return True
        if getattr(self, '_v62_late_skip_level', None) != level:
            self._v62_late_skip_level = level
            self._v6_stats['restarts_skipped_late'] = self._v6_stats.get('restarts_skipped_late', 0) + 1
            self._v6_log('v62-restart-skipped', dict(level=level, remaining_s=round(float(remaining)),
                                                     tokens_on_level=self._v61_level_tokens,
                                                     noact_tokens=getattr(self, '_v62_noact_tokens', 0)))
        return False

    _LEDGER_ADVICE = ("Do NOT replay a listed continuation exactly - vary the route or action at or before its first step. "
                      "(Advisory: fatality can depend on state not visible in the board interior, e.g. bar level or held "
                      "items; if you are certain the situation differs, state why before proceeding.) ")

    def _death_ledger_record(self, history_entries):
        """Round 5l review (Fable E1): his recorder takes entries[-2] as the death whenever entries[-1] is a RESET (the
        automatic restart). After a model RESET that itself ended in GAME OVER no automatic RESET follows, so entries[-2]
        is the previous call's last action, which did not die; nothing is recorded then."""
        entries = list(history_entries or [])
        if entries:
            r = getattr(entries[-1], 'result', None) or {}
            if str(getattr(entries[-1], 'action', '') or '').upper() == 'RESET' and r.get('game_over') \
                    and not r.get('automatic'):
                self._v6_stats['ledger_reset_deaths_skipped'] = self._v6_stats.get('ledger_reset_deaths_skipped', 0) + 1
                return None
        return super()._death_ledger_record(history_entries)

    def _death_ledger_lines(self, current_frame, history_entries, previous_step_summary):
        """Round 5c (user: neutralise; B3 g50t, ka59, sc25, tn36, wa30, r11l read the imperative as the "do not repeat
        that path" prior round 5 removed): the recorded continuations and the game's documented determinism stay; the
        imperative and the "bar level or held items" example go. The ids are the game's game-over numbers (B3 tu93:
        "(attempt(s) 7)" beside "1 failed attempt(s)")."""
        lines = super()._death_ledger_lines(current_frame, history_entries, previous_step_summary)
        out = []
        for line in lines:
            m = re.match(r'Known fatal continuations FROM THE CURRENT BOARD STATE \(this level has (\d+) failed attempt\(s\) '
                         r'recorded; the game is deterministic\):$', line)
            if m:
                # round 5l (B3, 392b ar02, lc01, te01, rw01, g539: "the game is deterministic" misled where hidden
                # per-attempt state differed; Fable 3.8 updates 386-G9): the observation only, and its basis (the ledger
                # hashes the whole frame: ARC3_STATE_IDENTITY_BORDER is unset, default 0)
                k = int(m.group(1))
                line = ('Continuations that ended in GAME OVER from a board identical to the current one (whole board '
                        'compared; %d game over%s recorded on this level):' % (k, '' if k == 1 else 's'))
            if line.startswith(self._LEDGER_ADVICE):
                # round 5l (class 10, Fable 1.10): the board compared is the whole board, not the interior
                line = ("Each listed continuation ended in GAME OVER when played from a board identical to this one; "
                        "state not shown on the board can differ between attempts. " + line[len(self._LEDGER_ADVICE):])
            line = re.sub(r'\(attempt\(s\) ([0-9][0-9, ]*)\)',
                          lambda m: '(game over %s of this game)' % ', '.join('#' + x.strip() for x in m.group(1).split(',')),
                          line)
            out.append(line)
        return out

    def _v61_step_line(self, step, level, valid, session):
        k = len(self._v61_attempts(session, level)) + 1
        if _v62_game_over_now(session):                 # round 5l (A1): only RESET executes in GAME OVER
            # v6.9 REX HIDE_RESET (arm B): no model RESET; the harness repeats the restart at the next action call
            valid = TEXTS['hr69_valid_game_over'] if HIDE_RESET69 else 'RESET only (the game reports GAME OVER)'
        # round 5c (B3 vc33 T40, cd82; A2 cd82: the step count was read as per level)
        whole = 'game' if V75_POLISH else 'run'         # v7.5 TXT-2: the step counts the whole game's actions
        return f"Step {step} (actions in the whole {whole}), level {level}, attempt {k}. Valid actions: {valid}."

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
            pair = [str(x) for x in (stop.get('pair') or [])]          # v6.2: the pair can span action() calls
            a = pair if len(pair) == 2 else (executed[-2:] if len(executed) >= 2 else (['?'] + executed)[-2:])
            if not skipped:                           # round 5c (B3 sc25 T28: single-action calls, nothing skipped)
                # round 5l (A7, Fable 3.12; 392b, many games): the second was the last action of its action() list
                return _fill(TEXTS['revert_stop_last'], a1=a[0], a2=a[1], **common)
            return _fill(TEXTS['revert_stop'], a1=a[0], a2=a[1], **common)
        if kind == 'level_restarted':
            attempts = self._v61_attempts(session, level)
            line = _fill(TEXTS['hazard_restart'], action=executed[-1] if executed else '?',
                         n=attempts[-1][0] if attempts else '?', **common)
            if len(attempts) >= 2:                    # v6.2 round 3 (b) (A2 ls20: budget restarts read as hazards)
                line += ' ' + _fill(TEXTS['restart_lengths'], lengths=self._v61_lengths_text(attempts))
            return line
        if kind == 'level_start':                     # round 5l (item 109, A5)
            return _fill(TEXTS['level_start_return'], action=executed[-1] if executed else '?', **common)
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
                return self._v61_takeover_text(level, step, valid, reminder, clock, session, current_frame,
                                               history_entries), 'takeover'
            text = _fill(TEXTS['first_turn'], valid_actions=valid).replace('{STANDING REMINDER}', reminder)
            lines = [l for l in text.split('\n') if l.strip() and not l.startswith('[image]')]
            if clock:
                lines.insert(1, clock)
            return '\n'.join(lines), 'first'
        parts = []
        kind = 'same'
        game_over_now = _v62_game_over_now(session)
        if s:
            parts += self._v6_summary_lines(s)
            manual = ta.ToolAgent._manual_reset_context_line(s)     # EXPOSE_RESET only (dfranzen's line)
            if manual:
                parts.append(manual)
            if V70H_LEDGER:                             # v7.0 item 10: a model RESET that ended an attempt: the attempt facts
                ledger = self._v70h_reset_ledger(session, level, s)
                if ledger:
                    parts.append(ledger)
            if game_over_now:
                # round 5l (A1, 392b rw01): the attempt's last action was a RESET that ended in GAME OVER; no automatic
                # restart followed, so the restart block (its "the level has restarted", history[-1] the automatic
                # restart) would be false. Review (Fable T1): when that RESET was the automatic restart, say so
                parts.append(_game_over_text(history_entries))
            elif s.get('game_over'):
                kind = 'restart'
                attempts = self._v61_attempts(session, level)
                body = _fill(TEXTS['restart'], fatal_action=s.get('fatal_action') or 'the last executed action',
                             level=level, next_attempt=len(attempts) + 1, k=len(attempts),
                             lengths=(self._v70h_lengths_text(session, level) if V70H_LEDGER     # v7.0 item 10: how each
                                      else self._v61_lengths_text(attempts)),                     # attempt ended, the total
                             fatal_interior=self._v61_fatal_interior(history_entries),
                             fatal_position=self._v62_fatal_position(history_entries))
                if len(attempts) > 5:                       # A3 (HD review): the list is capped at the last 5
                    body = body.replace('; the attempts lasted ', '; the last 5 attempts lasted ')
                parts.append('\n'.join(l for l in body.split('\n') if not l.startswith('[image]')))
                if V77:                                     # v7.7 (the user 10-24, lever 5): the ended attempt's step range
                    st77 = self._v77_steps_line(len(attempts), attempts[-1][0] if attempts else 0,
                                                self._v77_last_step(s, 'game_over', history_entries, -2))
                    if st77:
                        parts.append(st77)
            elif s.get('level_transition'):
                kind = 'levelup'
                if variants.summary() or variants.a45():
                    self._v6_rule_capture = int(level) - 1
                    self._v62_rule_turns = RULE_TURNS
                    # round 5l (item 108, S1): save_summary may store this level's summary for RULE_TURNS completions
                    self._v62_fn_window = (int(level) - 1, self._v6_stats.get('completions', 0) + RULE_TURNS)
                    self._v64_fn_window_hist = self._v64_game_hist_len()    # v6.4 A2: executed entries at the level-up
                    self._v62_fn_windows = {k: v for k, v in (getattr(self, '_v62_fn_windows', None) or {}).items()
                                            if v >= self._v6_stats.get('completions', 0)}
                    self._v62_fn_windows[int(level) - 1] = self._v62_fn_window[1]   # review: a later "Rule Lk:" save
                    if getattr(self, '_v62_last_texts', None):     # round 5h (G1 iv, 386 sy25): the summary may come
                        self._v6_capture_rule(self._v62_last_texts, provisional=True)   # in the clearing turn
                parts.append(TEXTS['levelup'].replace('{level-1}', str(int(level) - 1)).replace('{level}', str(level)))
                if V77:                                 # v7.7 (B, the user 10-24): the previous level's summary, to update
                    parts += self._v77_prev_lines(int(level) - 2)
                count = self._v61_library_count_line()
                if count:
                    parts.append(count)
                if NOTES or LEDGER68:                       # v6.7 NOTES / v6.8 LEDGER: the level-clear ledger at a level start
                    parts += self._v67_ledger_lines(history_entries)
            # v6.3 item G (user 2026-10-14: drop): no "You are still on level N." line; the step line states the level
            for stop in (s.get('no_op_stops') or [])[:1]:
                skipped = [str(x) for x in (stop.get('skipped') or [])]
                idx = stop.get('index')                   # 1-based position in its batch (v6 added 1 again)
                line = _fill(TEXTS['noop_stop'], action=stop.get('action'), p=idx if isinstance(idx, int) else '?',
                             n=len(skipped) + (idx if isinstance(idx, int) else 1), m=len(skipped),
                             skipped=', '.join(skipped) or 'none')
                pos, total = stop.get('pos'), stop.get('total')
                if pos and total and pos < total:         # a later action(...) call in the same snippet continued
                    line = line.replace(_NOOP_LAST, _NOOP_NOT_LAST % (pos, total))
                elif idx == 1:                            # v6.2 round 3 (b) (A2 sp80 15 of 29, s5i5, sc25)
                    line = line.replace(_NOOP_LAST, _NOOP_FIRST)
                if not skipped:                           # v6.2 (b): nothing was cut (run A: about 65 such paragraphs)
                    line = TEXTS['noop_last'].replace('{action}', str(stop.get('action')))
                parts.append(line)
            for stop in (s.get('v61_stops') or [])[-1:]:      # round 4 (A3 m0r0 T213): the latest stop, not the first
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
        elif game_over_now:                                  # round 5l (A1): a refused call or a yield in GAME OVER
            parts.append(_game_over_text(history_entries))
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
        if V70H_BLEDGER:                                    # v7.0 item 15: earlier actions from an identical board; the revisit
            try:
                parts += self._v70h_board_ledger_lines(current_frame, history_entries, valid_actions, death=any(
                    p and 'Continuations that ended in GAME OVER' in p for p in parts))
            except Exception as exc:  # noqa: BLE001
                self._v6_stats['v70h_board_ledger_errors'] = self._v6_stats.get('v70h_board_ledger_errors', 0) + 1
                if self._v6_stats['v70h_board_ledger_errors'] == 1:
                    self._v6_log('v70h-board-ledger-error', dict(error='%s: %s' % (type(exc).__name__, str(exc)[:200])))
        parts.append(self._v61_step_line(step, level, valid, session))
        parts.append(clock)
        parts.append(reminder)
        self._v6_push_note = False
        return '\n'.join(p for p in parts if p), kind

    def _v61_takeover_text(self, level, step, valid, reminder, clock, session, current_frame=None, history_entries=None):
        """Run B restart: the first observation of the new conversation (no restart notice; user decision). v6.5 OPENER:
        the leading section (the newest attempt note, the previous level's summary) comes first; see _v65_opener_lead."""
        section = self._v65_opener_lead(int(level)) if V65_OPENER else ''
        text = self._v61_takeover_body(level, step, valid, reminder, clock, session, current_frame, history_entries)
        return section + text

    def _v65_opener_lead(self, level):
        """v6.5 OPENER / C5 (user via IE 10-17): the restart opener's leading items, the current level's newest attempt note
        and the previous level's summary, in that order. Those that fit V65_PIN_CAP tokens together form the section (the
        opener's first text part; pinned in arm A); an item that does not fit follows the section, unpinned. Both arms
        compute the same order. Returns the leading text (section, then the items that did not fit), each item ending in a
        newline; the section's text is kept for _v6_opener."""
        count = getattr(self, '_v6_count', None) or (lambda t: (len(t) + 3) // 4)
        items = []
        notes = list(self._v62_attempt_notes.get(level) or [])
        prev = self._v6_rules.get(level - 1) if level > 1 else None
        fresh77 = None
        if V77:                     # v7.7 (the user 10-24): the notes saved for this conversation (they update the last level-up
            seen77 = self.__dict__.setdefault('_v77_notes_seen', {})      # summary), else that summary; decided once per
            conv77 = self._v69_ir_conv()                                  # conversation (IE 10-24: a second build of the same
            rec77 = seen77.get(level)                                     # opener gives the same lead)
            if rec77 is not None and rec77[1] == conv77:
                fresh77 = rec77[2]
            else:
                fresh77 = len(notes) > (int(rec77[0]) if rec77 else 0)
                seen77[level] = (len(notes), conv77, fresh77)
                key77 = ('v77_lead_notes' if fresh77 else 'v77_lead_old_notes' if notes   # the user 10-24: the level's older
                         else 'v77_lead_summary')                                       # notes before the previous summary
                self._v6_stats[key77] = self._v6_stats.get(key77, 0) + 1
            self._v77_lead_summary = not notes
        if notes:
            items.append(TEXTS['attempt_notes_header'] + '\n' + notes[-1])
        if prev and (fresh77 is None or not notes):
            items.append(_fill(TEXTS['v65_prev_summary'], level=level - 1) + '\n' + _rule_display(level - 1, prev))
        section, unfit, used = [], [], 0
        for item in items:
            c = count(item)
            if used + c <= (10 ** 9 if V77 and V65_PIN else V65_PIN_CAP):   # v7.7: the text pin holds every text part
                section.append(item)
                used += c
            else:
                unfit.append(item)
        sec = ''.join(x + '\n' for x in section)
        self._v65_section = sec or None
        if items and not section:
            self._v6_stats['pin_skipped_none_fit'] = self._v6_stats.get('pin_skipped_none_fit', 0) + 1
        if unfit:
            self._v6_stats['pin_unfit_items'] = self._v6_stats.get('pin_unfit_items', 0) + len(unfit)
        if section:
            self._v6_stats['opener_sections'] = self._v6_stats.get('opener_sections', 0) + 1
            self._v6_stats['opener_section_tokens'] = self._v6_stats.get('opener_section_tokens', 0) + used
        self._v6_log('v65-opener-lead', dict(level=level, items=len(items), section=len(section), unfit=len(unfit),
                                            tokens=used, pin=bool(V65_PIN and section)))
        return sec + ''.join(x + '\n' for x in unfit)

    def _v65_opener_rest(self, level):
        """v6.5 OPENER: in place of the list of summaries: one line naming the levels whose stored summaries are not shown,
        the A7 line for the previous level only, and the attempt-notes-none line."""
        out = []
        others = sorted(lvl for lvl in self._v6_rules if lvl < level - 1 or (V77 and lvl == level - 1
                                                                          and not self.__dict__.get('_v77_lead_summary', True)))
        if others:
            names = [str(x) for x in others]
            levels = names[0] if len(names) == 1 else ', '.join(names[:-1]) + ' and ' + names[-1]
            out.append(_fill(TEXTS['v65_summary_not_shown' if len(names) == 1 else 'v65_summaries_not_shown'],
                             levels=levels))
        if level > 1 and (level - 1) not in self._v6_rules:
            out.append(_fill(TEXTS['v65_summary_missing'], level=level - 1))
            self._v6_stats['opener_missing_summaries'] = self._v6_stats.get('opener_missing_summaries', 0) + 1
            self._v6_log('v64-opener-missing-summaries', dict(level=int(level), missing=[level - 1]))
        if not (self._v62_attempt_notes.get(level) or []) and self._v62_notes_asked == level:
            out.append(TEXTS['attempt_notes_none'])
        return out

    def _v61_takeover_body(self, level, step, valid, reminder, clock, session, current_frame=None, history_entries=None):
        # round 5l (A7, 392b gf01: "levels 1-1 completed")
        parts = [_fill(TEXTS['takeover'], level=level, done=('level 1' if int(level) == 2 else 'levels 1-%d' % (int(level) - 1)))
                 if int(level) > 1 else TEXTS['takeover_l1']]
        soft = self._v67_takeover_lines(session, level, step)   # v6.7 SOFT: the soft restart's facts ([] after hard)
        parts += soft
        if 'takeover_restart' in TEXTS and not soft:    # round 5h (user, D1): the restart is stated
            hidden = self._v61_hidden_level == level and _lazy_enabled()   # round 5l review (Fable T2)
            parts.append(_fill(TEXTS['takeover_restart'] if hidden else TEXTS['takeover_restart_shown'], level=level,
                               step=step))
            if LOOP_TEXTS68 and self._v68_loop_takeover(level):   # v6.8 LOOP_TEXTS: the restart was the loop rule's
                parts.append(TEXTS['takeover_loop_hard'])
            facts = self._v62_restart_facts(session, level)          # round 5l (B4, A4)
            if facts:
                parts.append(facts)
        if V65_OPENER:                                       # v6.5 OPENER: the summaries' line and the A7 / notes lines
            parts += self._v65_opener_rest(int(level))
        # round 5d (C50 g040: L1 and L2 were dropped silently at L8): every completed level's summary, not the 5 newest
        rules = [(lvl, r) for lvl, r in sorted(self._v6_rules.items()) if lvl < int(level) and not V65_OPENER]
        missing = ([lvl for lvl in range(1, int(level)) if lvl not in self._v6_rules]
                   if 'restart_summary_missing' in TEXTS and not V65_OPENER else [])
        if rules or missing:                                 # v6.4 A7 (12-h audit, sf06f g024: L3 left out silently)
            parts.append(TEXTS['restart_summaries'])
            shown = dict((lvl, _rule_display(lvl, r)) for lvl, r in rules)
            shown.update((lvl, _fill(TEXTS['restart_summary_missing'], level=lvl)) for lvl in missing)
            parts += [shown[lvl] for lvl in sorted(shown)]  # round 5g: lists shown as indented items
            if missing:
                self._v6_stats['opener_missing_summaries'] = self._v6_stats.get('opener_missing_summaries', 0) + len(missing)
                self._v6_log('v64-opener-missing-summaries', dict(level=int(level), missing=missing))
        if not V65_OPENER:
            parts += self._v62_notes_lines(int(level))       # round 5l (item 111, D1)
        if NOTES:                                        # v6.7 NOTES: the action count, the ledger and the coverage facts
            parts += self._v67_notes_restart_lines(int(level), valid, history_entries)
        elif LEDGER68 or NEVER68:                        # v6.8 LEDGER / NEVER_TYPES, without the other NOTES parts
            parts += self._v68_restart_lines(int(level), valid, history_entries)
        count = self._v61_library_count_line()
        if count:
            parts.append(count)
            # v6.9 INLEVEL: the in-level refresh's opener carries the program list as the level-up refresh's does
            if REFRESH_PROGRAMS68 and (getattr(self, '_v67_takeover', None) or {}).get('kind') in ('refresh', 'inlevel'):
                parts += self._v68_refresh_program_lines(level)   # v6.8 REFRESH_PROGRAMS: the selection after the count line
        if soft:                                         # v6.7 SOFT: where this level's entries begin in `history`
            parts.append(self._v67_history_line(level, history_entries))
        elif self._v61_hidden_level == level and _lazy_enabled():
            parts.append(TEXTS['takeover_history'] if int(level) > 1 else TEXTS['takeover_history_l1'])
        if _v62_game_over_now(session):                 # round 5l (A1): the restart's own RESET ended in GAME OVER
            parts.append(TEXTS['game_over_harness_reset'])
        if current_frame is not None and ta._get_env_bool("ARC3_DEATH_LEDGER", False):
            try:                                        # round 5l (B4, user): the ledger, kept across the restart
                parts += self._death_ledger_lines(current_frame, history_entries, None)
            except Exception:  # noqa: BLE001
                pass
        parts.append(self._v61_step_line(step, level, valid, session))
        parts.append(clock)
        parts.append(reminder)
        return '\n'.join(p for p in parts if p)

    def _v62_restart_facts(self, session, level):
        """Round 5l (B4, A4; 392b fr01, g501, g503, g305): how the level's earlier attempts ended, silent returns to its
        starting board, and the programs set aside, from the game session's attempt record."""
        d = getattr(session, '__dict__', {}) or {}
        ends = list((d.get('v62_level_attempt_ends') or {}).get(level, []))
        out = []
        if ends:
            how = [('%d ended in GAME OVER' % ends.count('game_over')) if ends.count('game_over') else '',
                   ('%d ended with your RESET' % ends.count('reset')) if ends.count('reset') else '',
                   ('%d ended at a harness restart' % ends.count('restart')) if ends.count('restart') else '']
            how = [h for h in how if h]
            out.append('Before this conversation, this level had %d attempt%s: %s.' % (
                len(ends), '' if len(ends) == 1 else 's', ', '.join(how[:-1]) + (' and ' if len(how) > 1 else '') + how[-1]))
        returns = (d.get('v62_level_start_returns') or {}).get(level, 0)
        if returns:
            out.append("The whole board returned to the level's starting board without a RESET %d time%s." % (
                returns, '' if returns == 1 else 's'))
        lv, k = getattr(self, '_v62_restart_archived', (None, 0))
        if lv == level and k:
            out.append('%d program%s written on this level %s set aside at the restart and %s not loaded.' % (
                k, '' if k == 1 else 's', 'was' if k == 1 else 'were', 'is' if k == 1 else 'are'))
        return ' '.join(out)

    def _v6_resumption_text(self, current_frame, valid_actions, history_entries=None):
        valid = _v6_valid_display(valid_actions)
        step, level = getattr(current_frame, 'step', '?'), getattr(current_frame, 'level', '?')
        key67 = self._v62_resumption_key()                       # (once: it counts resumption_noact)
        body = _fill(TEXTS[key67], step=step, level=level, valid_actions=valid,
                     t=format(int(getattr(self, '_v62_noact_tokens', 0) or 0), ',')   # v6.7 TEXTS67: the count
                     ).replace('{STANDING REMINDER}', '' if variants.lean() else TEXTS['reminder'])
        if '{t}' in TEXTS[key67]:                                 # v6.7 TEXTS67 receipt
            self._v6_stats['resumption_counts'] = self._v6_stats.get('resumption_counts', 0) + 1
        body = body.replace('{; the latest sequence executed this turn was {seq}}', '')
        lines = [l for l in body.split('\n') if l.strip() and not l.startswith('[image]')]
        session = self._v61_session()
        extra = []
        if NO_ACTION_CALLS and self._v61_noact_calls >= NO_ACTION_CALLS:   # item 11 (v6.3: off unless set)
            untried = self._v61_untried(history_entries, valid_actions)
            line = _fill(TEXTS['no_action'], k=self._v61_noact_calls, untried=', '.join(untried) or '-')
            if self._v61_noact_level_from is not None:        # round 5l (A4): counted from this conversation's start
                line = line.replace('Actions not yet executed on this level:',
                                    'Actions not yet executed on this level in this conversation:')
            if not untried:                                   # v6.2 (b): no vacuous "every valid action has been tried"
                line = line.split(' Actions not yet executed')[0]
            extra.append(line)
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
        """The user message: text, then images (board first). v6.5 OPENER / C5: a restart opener's leading section is its own
        first text part in both arms (the template joins the parts, so the tokens are the same); arm A marks it for the
        server's pin (_arc3_pin_parts, sent with /start by the transport)."""
        sec = self.__dict__.pop('_v65_section', None)
        pin = False
        two77 = V77 and V65_PIN and kind == 'takeover'      # v7.7 (the user 10-24): two pins, the text parts and the boards
        split77 = None
        if sec and kind == 'takeover' and text.startswith(sec) and len(text) > len(sec):
            content = [dict(type='text', text=sec), dict(type='text', text=text[len(sec):])]
            pin = V65_PIN and not two77
            if pin:
                self._v6_stats['pins_requested'] = self._v6_stats.get('pins_requested', 0) + 1
        else:
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
            split77 = len(content)                           # v7.7: the board pin's first part (the text parts precede it)
            if CLEAR_STEP69 and kind == 'takeover' and not (    # v6.9 CLEAR_STEP: the previous level's clearing step (v7.7:
                    V77 and (self.__dict__.get('_v67_takeover') or {}).get('kind') == 'inlevel'):   # not in a mid-level opener)
                content += (self._v73_clear_parts(current_frame, history_entries) if V73_CLEAR_BOARDS   # v7.3: with boards
                            else self._v69_clear_step_parts(current_frame))                            # of its final attempt
            if (V70H_ILOPEN and kind == 'takeover'           # v7.0 item 11: an in-level refresh's opener (also the NO_HARD_RESTART
                    and (self.__dict__.get('_v67_takeover') or {}).get('kind') == 'inlevel'):   # and park returns): the block
                il77 = self._v70h_il_parts(current_frame, history_entries)
                n77 = int(self.__dict__.pop('_v77_ilb_text', 0) or 0)   # v7.7: the block's leading text parts join the text
                if V77 and split77 == len(content):                    # section (no clearing block before the block)
                    split77 += n77
                content += il77
            content += [dict(type='text', text='Current board.'), board]
            prev = self._v6_prev_opener_grid
            changed = prev is not None and prev != current_frame.grid
            if kind not in ('first', 'levelup', 'takeover') and changed:
                from inference.agent.runtime_state import Frame
                d = diff_image_part(Frame(grid=prev, step=0, level=0), current_frame)
                if d is not None:
                    content += [dict(type='text', text=('Change since your previous turn: changed cells in their new '
                                                        'colour, everything else dark navy.')), d]
            elif kind not in ('first', 'levelup', 'takeover') and prev is not None and 'no_change' in TEXTS:
                content += [dict(type='text', text=TEXTS['no_change'])]       # round 5h (G10, 386 gx0t)
        self._v6_prev_opener_grid = current_frame.grid
        self._v6_prev_opener_level = current_frame.level
        ok77 = bool(two77 and split77 is not None and 1 <= split77 < len(content)
                    and all(isinstance(p, dict) and p.get('type') == 'text' for p in content[:split77]))
        if ok77:                                            # v7.7: a clean token boundary between the two pins
            self._v77_split_boundary(content, split77)
        if (V73 or V73_CLEAR_BOARDS) and kind in ('first', 'takeover'):   # v7.3: images per opener kind
            self._v73_opener_receipt(kind, content)
        if ok77:                    # v7.7 (the user 10-24: "pin the text part and images separately"): the server pins the text
            self._v6_stats['v77_pins_split'] = self._v6_stats.get('v77_pins_split', 0) + 1      # parts [0, split) and the board
            return {"role": "user", "content": content, "_arc3_observation": True, "_arc3_pin_parts": len(content),   # parts
                    "_arc3_pin_split": split77}                                                 # [split, end) as two pins
        if two77:                                           # no valid split (not expected): no pin, counted
            self._v6_stats['v77_pin_split_invalid'] = self._v6_stats.get('v77_pin_split_invalid', 0) + 1
        if pin:
            return {"role": "user", "content": content, "_arc3_observation": True, "_arc3_pin_parts": 1}
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
                if not head:                  # round 5l (A7, 392b dw01): a single over-long line is cut, not dropped
                    head.append(self._v62_cut_line(line, budget_head - 16))
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
        marker = _fill(TEXTS['truncation_marker'] if k else TEXTS['truncation_line_cut'], k=k, t=max(0, omitted))
        return '\n'.join(head + [marker] + tail), True

    def _v62_save_summary(self, text):
        """Round 5l (item 108, S1; Fable 2.1): the sandbox's save_summary(text). Stored verbatim (its own "Rule Lk:" tag
        stripped, capped like a captured summary), for the level whose completion asked for it, or as the attempt notes
        while the restart's summary turn is pending (item 111); the latest call wins. Returns the note save_summary
        prints."""
        text = str(text or '').strip()
        count = getattr(self, '_v6_count', None)
        req = getattr(self, '_v62_attempt_request', None)
        win = getattr(self, '_v62_fn_window', None)
        if req is None and (win is None or self._v6_stats.get('completions', 0) > win[1]):
            self._v6_stats['summaries_unrequested'] = self._v6_stats.get('summaries_unrequested', 0) + 1
            return dict(saved=False, note=TEXTS['summary_not_requested'])
        n = req if req is not None else win[0]
        tag = None                   # v6.4 A2: also read below
        if req is None:              # round 5l review (Fable): a "Rule Lk:" tag naming another level whose window is open
            tag = re.match(r'(?i)\s*(?:rule\s*)?L\s*(\d+)\s*:', text)
            open_ = {k for k, v in (getattr(self, '_v62_fn_windows', None) or {}).items()
                     if v >= self._v6_stats.get('completions', 0)}
            if tag and int(tag.group(1)) in open_:
                n = int(tag.group(1))
        body = (_strip_level_tag(text, n) if req is None
                else re.sub(r'(?i)^\s*notes?\b[^\n:]{0,40}:[ \t]*\n?', '', text, count=1).strip())
        if len(re.findall(r'[A-Za-z]', body or '')) < 20:
            return dict(saved=False, note=TEXTS['summary_too_short'])
        capped = _cap_rule(body, count)
        tokens = count(capped) if count else (len(capped) + 3) // 4
        cut = ' (cut to %s tokens)' % format(SUMMARY_TOKENS, ',') if capped != body else ''
        self._v62_saved_in_call = True
        if req is not None:                                   # item 111 (D1): notes from the abandoned conversation
            notes = self._v62_attempt_notes.setdefault(n, [])
            replaced = bool(notes) and getattr(self, '_v62_attempt_saved', False)
            if replaced:
                notes[-1] = capped
            else:
                notes.append(capped)
            self._v62_attempt_saved = True
            self._v6_stats['attempt_notes_saved'] = self._v6_stats.get('attempt_notes_saved', 0) + (not replaced)
            self._v6_log('v62-summary-saved', dict(kind='attempt', level=n, tokens=tokens, replaced=replaced))
            k69 = ('summary_saved_attempt_inlevel' if INLEVEL69 and (self.__dict__.get('_v69_ir') or {}).get('pending')
                   == 'notes' else 'summary_saved_attempt')   # v6.9 INLEVEL: no restart follows its notes turn
            return dict(saved=True, note=_fill(TEXTS[k69], level=n, tokens=tokens, cut=cut))
        if req is None and tag and int(tag.group(1)) != n and 'summary_tag_closed' in TEXTS:
            # v6.4 A2, IE review: a tag is the model's own attribution, so a tag naming a level without an open window is
            # never stored as another level (s5i5's "Rule L1" fell through to L2's window), whatever the action state
            k = int(tag.group(1))
            if 1 <= k < n and k not in self._v6_rules:
                # v6.5 A2r (IE 10-17, gm06): a completed level with no summary takes the text as its first summary, late
                return self._v65_save_late(text, k, n, count)
            self._v6_stats['summary_tag_refused'] = self._v6_stats.get('summary_tag_refused', 0) + 1
            self._v6_log('v64-summary-tag-refused', dict(tag=k, window=n))
            return dict(saved=False, note=_fill(TEXTS['summary_tag_closed' if k < n else 'summary_tag_open_level'], level=k,
                                                window=n))   # v7.8 F2: the open window's level (V77's text)
        if (req is None and n in self._v6_rules and tag is None
                and self._v64_acted_since_levelup() and 'summary_not_replaced' in TEXTS):
            # v6.4 A2 (12-h audit, sf10f s5i5 T50: a save tagged "Rule L1" during L3 replaced the L2 summary and showed as
            # L2 in the restart opener; IE: a tag naming this level stays allowed, as does the first save): an untagged
            # save does not replace an existing summary once an action has executed on the next level
            self._v6_stats['summary_replace_refused'] = self._v6_stats.get('summary_replace_refused', 0) + 1
            self._v6_log('v64-summary-not-replaced', dict(level=n, tag=int(tag.group(1)) if tag else None))
            return dict(saved=False, note=_fill(TEXTS['summary_not_replaced'], level=n, next=n + 1))
        replaced = n in self._v62_rule_fn                     # an earlier save_summary for this level
        if not replaced:
            self._v6_stats['rules_via_function'] = self._v6_stats.get('rules_via_function', 0) + 1
        if n not in self._v6_rules:
            self._v6_stats['rules_recorded'] = self._v6_stats.get('rules_recorded', 0) + 1
        self._v62_rule_fn.add(n)
        self._v6_rules[n] = capped
        self._v6_rules_dirty = True
        if self._v6_rule_capture == n:                        # the function's text wins over any later capture
            self._v6_rule_capture = None
        self._v6_log('v62-summary-saved', dict(kind='level', level=n, tokens=tokens, replaced=replaced))
        return dict(saved=True, note=_fill(TEXTS['summary_saved'], level=n, tokens=tokens, cut=cut)
                    + (' ' + TEXTS['summary_replaced'] if replaced else ''))

    def _v65_save_late(self, text, k, n, count):
        """v6.5 A2r: store a save tagged for completed level k (no summary yet) as k's first summary, while level n's window
        is open; that window stays open for n's own summary."""
        body = _strip_level_tag(text, k)
        if len(re.findall(r'[A-Za-z]', body or '')) < 20:
            return dict(saved=False, note=TEXTS['summary_too_short'])
        capped = _cap_rule(body, count)
        tokens = count(capped) if count else (len(capped) + 3) // 4
        cut = ' (cut to %s tokens)' % format(SUMMARY_TOKENS, ',') if capped != body else ''
        done = self._v6_stats.get('completions', 0)
        self._v6_stats['summary_late_tagged'] = self._v6_stats.get('summary_late_tagged', 0) + 1
        self._v6_stats['rules_via_function'] = self._v6_stats.get('rules_via_function', 0) + 1
        self._v6_stats['rules_recorded'] = self._v6_stats.get('rules_recorded', 0) + 1
        self._v62_rule_fn.add(k)
        self._v6_rules[k] = capped
        self._v6_rules_dirty = True
        if self._v6_rule_capture == k:
            self._v6_rule_capture = None
        self._v6_log('v65-summary-late-tagged', dict(level=k, window=n, completions=done, since=max(0, done - k),
                                                     tokens=tokens))
        return dict(saved=True, note=_fill(TEXTS['summary_saved'], level=k, tokens=tokens, cut=cut))

    def _v65_loop_line(self, full, text, kind):
        """v6.5 C1 (user via IE 10-17): each acting turn's signature is (its executed actions, its report); when the smallest
        period p <= 4 with the last k*p signatures k identical repeats has k >= 3, the report states it (nothing is refused or
        stopped). Per conversation: a takeover or first turn starts a new history; a resumption is not an acting turn."""
        mode = variants.loop_sig()
        if kind in ('first', 'takeover'):
            self._v65_sigs = []
        if mode == 'off' or kind in ('first', 'takeover', 'resume'):
            return text
        acts = tuple(str(a) for a in ((self._last_step_summary or {}).get('executed_actions') or ()))
        sigs = self.__dict__.setdefault('_v65_sigs', [])
        sigs.append((acts, _v65_report_sig(full, mode)))
        del sigs[:-512]
        p, k = _v65_loop(sigs)
        self._v6_stats['loop_sig'] = mode
        if k < 3:
            return text
        line = (_fill(TEXTS['v65_loop_1'], k=k) if p == 1 else _fill(TEXTS['v65_loop_p'], n=k * p, p=p, k=k))
        self._v6_stats['loop_lines'] = self._v6_stats.get('loop_lines', 0) + 1
        self._v6_stats['loop_max_k'] = max(self._v6_stats.get('loop_max_k', 0), k)
        self._v6_stats['loop_games'] = 1
        self._v6_log('v65-loop-line', dict(p=p, k=k, mode=mode))
        lines = text.split('\n')
        at = next((i for i, l in enumerate(lines) if l.startswith('Step ')), len(lines))
        return '\n'.join(lines[:at] + [line] + lines[at:])

    def _v64_game_hist_len(self):
        """v6.4 A2: the game's executed-entry count (the solver's history_entries; `history` in the loop tests)."""
        game = getattr(getattr(self, '_step_env_callback', None), '__self__', None)
        return len(getattr(game, 'history_entries', None) or getattr(game, 'history', None) or [])

    def _v64_acted_since_levelup(self):
        """v6.4 A2: True when an action has executed since the last level-up opened the summary window."""
        at = getattr(self, '_v64_fn_window_hist', None)
        return at is not None and self._v64_game_hist_len() > at

    def _v62_show_board(self, request):
        """Round 5m (item 124, user 10-08: the model is told never to print a whole board, so a board its code built,
        such as a simulated one, had no advertised view): the sandbox's show_board(cells, label). The board is kept for
        this call's result as an image like the board image (MULTIMODAL_UPSCALE), captioned "Board image j: show_board"
        (the server's 'tool' image kind, as converted boards); at most tool_output.MAX_IMAGES per call, shared with
        converted printed boards; a board already shown in the call is named instead. Returns the note show_board prints."""
        import hashlib
        from inference.agent.runtime_state import Frame
        from inference.utils.grid_utils import ARC_COLOR_CHARS
        rows = request.get('rows')
        sym = {c: i for i, c in enumerate(ARC_COLOR_CHARS)}
        if (not isinstance(rows, list) or not rows or len(rows) > 64 or any(not isinstance(r, str) for r in rows)
                or len({len(r) for r in rows}) != 1 or not 0 < len(rows[0]) <= 64
                or any(ch not in sym for r in rows for ch in r)):
            return {'note': TEXTS['show_board_invalid']}
        grid = [[sym[ch] for ch in r] for r in rows]
        digest = hashlib.sha256(repr(grid).encode()).hexdigest()   # as tool_output.convert: a printed copy is named
        shown = self.__dict__.setdefault('_v62_shown', [])
        for j, (h, _, _) in enumerate(shown, 1):
            if h == digest:
                return {'note': TEXTS['board_same'].replace('{j}', str(j))}
        if len(shown) >= tool_output.MAX_IMAGES:
            return {'note': _fill(TEXTS['show_board_full'], m=tool_output.MAX_IMAGES)}
        label = request.get('label')
        cap = _fill(TEXTS['show_board_caption'], j=len(shown) + 1, rows=len(grid), cols=len(grid[0]),
                    label=(', "%s"' % str(label).replace('"', "'")[:80]) if label not in (None, '') else '')
        part = {"type": "image_url", "image_url": {"url": frame_to_png_data_url(Frame(grid=grid, step=0, level=0))}}
        shown.append((digest, cap, part))
        self._v6_stats['shown_boards'] = self._v6_stats.get('shown_boards', 0) + 1
        return {'note': cap}

    def _v62_end_summary_turn(self, completions):
        """Round 5l (item 111): close the notes turn; without a save_summary call, a "Notes L{n}:" block of the reply
        (review: of any completion of the turn, the latest first, so a message the 16K close cut still counts)."""
        n = self._v62_attempt_request
        if n is not None and not getattr(self, '_v62_attempt_saved', False):
            notes = next((x for x in (_capture_notes(t, n) for t in reversed(list(completions or []))) if x), '')
            if notes:
                self._v62_attempt_notes.setdefault(n, []).append(_cap_rule(notes, getattr(self, '_v6_count', None)))
                self._v6_stats['attempt_notes_fallback'] = self._v6_stats.get('attempt_notes_fallback', 0) + 1
                self._v6_log('v62-summary-saved', dict(kind='attempt', level=n, via='capture'))
        if getattr(self, '_v64_notes_retry', None):          # v6.4 A1: the retry's outcome (receipt)
            outcome = ('saved' if getattr(self, '_v62_attempt_saved', False)
                       else 'captured' if n is not None and n in getattr(self, '_v62_attempt_notes', {}) else 'none')
            self._v6_stats['notes_retry_' + outcome] = self._v6_stats.get('notes_retry_' + outcome, 0) + 1
            self._v6_log('v64-attempt-notes-retry', dict(level=n, reason=self._v64_notes_retry, outcome=outcome))
            self._v64_notes_retry = None
        self._v62_attempt_request = None
        self._v62_summary_state = 'done'

    def _v64_retry_text(self, completions, reason):
        """v6.4 A1 (12-h audit: sf10f g525's save raised SyntaxError, tu93's reply called action() and was refused; both
        restarts opened with no notes, g525 then spent ~92K tokens re-deriving the level; agreed with IE): the text of the
        one more request when the notes turn would end with neither a save_summary call nor a "Notes L{n}:" block over its
        replies so far; None otherwise (also after the one retry)."""
        if (self._v62_attempt_request is None and 'refresh_summary_retry' in TEXTS
                and (self.__dict__.get('_v67_refresh_state') or {}).get('pending') == 'summary'):
            return self._v67_refresh_retry(reason)      # v6.7 REFRESH (IE 08:12): its summary turn's one retry
        n = self._v62_attempt_request
        if (n is None or getattr(self, '_v62_attempt_saved', False) or getattr(self, '_v64_notes_retry', None)
                or 'summary_retry' not in TEXTS):
            return None
        if any(_capture_notes(t, n) for t in reversed(list(completions or []))):
            return None
        self._v64_notes_retry = reason
        self._v6_stats['notes_retries'] = self._v6_stats.get('notes_retries', 0) + 1
        return _fill(TEXTS['summary_retry'], reason=reason, level=n)

    @staticmethod
    def _v64_notes_reason(tool_messages):
        """v6.4 A1: why the notes turn's calls saved nothing, from their results (IE: the error type and line)."""
        text = ' '.join(m['content'] if isinstance(m.get('content'), str) else
                        ' '.join(p.get('text', '') for p in m.get('content') or [] if isinstance(p, dict))
                        for m in tool_messages or [])
        if TEXTS['summary_turn_refusal'][:40] in text:
            return 'an action call was refused'
        syn = re.search(r'Python syntax error: .*?line (\d+)\)', text)    # the sandbox's syntax-error report
        if syn:
            return 'the python call had a syntax error at line %s' % syn.group(1)
        err = re.search(r'\b([A-Z][A-Za-z]*(?:Error|Exception))\b', text)
        if err:
            line = re.search(r'line (\d+)', text[err.start():err.start() + 400]) or re.search(r'line (\d+)', text)
            return 'the python call raised %s%s' % (err.group(1), ' at line %s' % line.group(1) if line else '')
        return 'the call did not call save_summary(text)'

    def _v62_notes_lines(self, level):
        """Round 5l (item 111, D1; Fable 5.4-5.5): the notes of the level's abandoned conversations, labelled as the model's
        own, newest kept within ATTEMPT_NOTES_TOKENS."""
        notes = list(self._v62_attempt_notes.get(level) or [])
        if not notes:
            return [TEXTS['attempt_notes_none']] if self._v62_notes_asked == level else []
        count = getattr(self, '_v6_count', None) or (lambda t: (len(t) + 3) // 4)
        keep, used = [], 0
        for note in reversed(notes):
            c = count(note)
            if keep and used + c > ATTEMPT_NOTES_TOKENS:
                break
            keep.insert(0, note)
            used += c
        lines = [TEXTS['attempt_notes_header'] if len(notes) == 1 else TEXTS['attempt_notes_header_many']]
        if len(notes) > len(keep):
            k = len(notes) - len(keep)
            lines.append('(Notes from %d earlier conversation%s on this level are omitted.)' % (k, '' if k == 1 else 's'))
        for i, note in enumerate(keep):
            if len(keep) > 1:
                lines.append('Conversation %d of %d:' % (len(notes) - len(keep) + i + 1, len(notes)))
            lines.append(note)
        return lines

    def _v62_cut_line(self, line, budget):
        """Round 5l (A7): the longest prefix of `line` within `budget` tokens, marked as cut."""
        lo, hi = 0, len(line)
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if self._v6_count(line[:mid]) <= budget:
                lo = mid
            else:
                hi = mid - 1
        return line[:lo] + ' [... line cut]'

    def _run_python_tool(self, state_path, arguments):
        sandbox_ext.CURRENT.agent = self
        game = self._v61_session()
        if game is not None:                          # items 2b / 2d / 3: per-call counters and details
            game.__dict__['v61_call_executed'] = 0
            game.__dict__['v61_call_revert_run'] = 0
            game.__dict__['v61_call_revert_prev'] = None
            game.__dict__['v61_call_payloads'] = []
            game.__dict__['v62_call_seen'] = {}           # round 5b: repeat_in_call is per Python call
        self.__dict__['_v62_guard_first'] = {}                # round 4: per snippet, like his repeat-state store
        self._v62_saved_in_call = False                       # round 5l (item 108)
        try:
            return super()._run_python_tool(state_path, arguments)
        finally:
            sandbox_ext.CURRENT.agent = None

    def _summarize_step_sequence(self, action_results):
        summary = super()._summarize_step_sequence(action_results)
        game = self._v61_session()
        recs = list((getattr(game, '__dict__', {}) or {}).get('v61_call_payloads') or [])
        if summary is None and recs:
            summary = _v62_summary_from_records(recs, getattr(self, '_v61_level', None))
        if summary is None:
            return None
        if len(recs) > len(summary.get('executed_actions') or []):
            # round 5l (A7, Fable 3.12; 392b pi01): the host executed an action whose reply the sandbox never read (the
            # call's CPU limit stopped it while it waited), so the snippet's list missed it; the host's records are complete
            fixed = _v62_summary_from_records(recs, summary.get('level'))
            for key in ('executed_actions', 'executed_count', 'game_over', 'level_transition', 'run_complete',
                        'fatal_action', 'board_changed', 'gameplay_changed'):
                summary[key] = fixed[key]
            self._v6_stats['records_reconciled'] = self._v6_stats.get('records_reconciled', 0) + 1
        summary['v61_actions'] = recs
        stops, noop_pos, done = [], [], 0
        for item in action_results or []:
            executed = item.get('executed_actions') or []
            done += len(executed)
            if item.get('stop_reason') in V61_STOPS:
                stops.append(dict(kind=item.get('stop_reason'), executed=executed, skipped=item.get('skipped_actions') or [],
                                  detail=item.get('stop_detail'), pos=done, pair=item.get('reverted_pair')))
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

    def _dispatch_tool(self, state_path, name, arguments):
        result = super()._dispatch_tool(state_path, name, arguments)
        if name != 'python':                          # round 4 (A3 re86 T255, T316; bp35 T271): a tool named `action`
            import dataclasses
            text = "Unknown tool: %s. The only tool is python: call action([...]) inside its code." % name
            if V75_POLISH and name in _V75_RUNTIME_FUNCTIONS and name != 'action':   # v7.5 TOOL-2: name the called function
                text = _fill(TEXTS['v75_unknown_tool'], name=name)
                self._v6_stats['v75_unknown_tool_named'] = self._v6_stats.get('v75_unknown_tool_named', 0) + 1
            result = dataclasses.replace(result, content=json.dumps({"error": text}, indent=2))
        return result

    def _compact_action_result(self, payload):
        compact = super()._compact_action_result(payload)
        if V70H_NOCLOCK:                              # v7.0 item 12: the run clock leaves the model-visible result (super() read
            compact.pop('run_elapsed_seconds', None)  # it for the game's own clock first)
            compact.pop('time_remaining_seconds', None)
        if 'reverted' in payload:                     # 2b: read by the sandbox's stale-arm rule
            compact['reverted'] = bool(payload['reverted'])
        for key in ('level_start_match', 'edge_cells_restored', 'repeat_in_call',   # round 5b: facts a loop can test
                    'interior_changed', 'near_level_start', 'actions_since_start',   # round 5d
                    'band_connected',                                                 # round 5h (G3)
                    'start_return', 'board_shift', 'undo_restored'):                 # round 5l (A5, C1, C3)
            if payload.get(key):
                compact[key] = payload[key]
        if payload.get('reverted_pair'):              # v6.2 round 3 (A2: dc22 11/11, lf52, sc25 stops showed "?"): the
            compact['reverted_pair'] = list(payload['reverted_pair'])   # compact result dropped it; the stop line reads it
        if 'last_board_changed' in payload:           # round 5l (A3 option b): read by the sandbox's stale-arm rule
            compact['last_board_changed'] = bool(payload['last_board_changed'])
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
        if RESTART_HISTORY == 'level':               # user ~03:15: the restarted level's programs are hidden as well
            return []
        return [dict(name=n, signature=e.get('signature'), doc=e.get('doc'), level=e.get('level'), step=e.get('step'),
                     source=e.get('source'), archived=True) for n, e in self._v61_archive.items()]

    def _v6_history_frames(self, index):
        session = getattr(self._step_env_callback, '__self__', None)
        return sandbox_ext.host_history_frames(session, index) if session is not None else []

    def _render_tool_payload(self, payload, *, truncate_fields=()):
        """§5.6: printed full boards -> 4x images (deduplicated, <= 4), then his JSON rendering with our text cap."""
        # round 5m (item 124): the call's show_board images first; printed boards share the per-call image budget and
        # a printed copy of a shown board is named
        shown = self.__dict__.pop('_v62_shown', None) or []
        self._v62_last_shown = len(shown)
        state = dict(seen={h: j for j, (h, _, _) in enumerate(shown, 1)}, budget=tool_output.MAX_IMAGES - len(shown),
                     omitted=0)
        captions = dict(board=TEXTS['board_caption'], same=TEXTS['board_same'])
        images = [(cap, part) for _, cap, part in shown]
        payload = dict(payload)
        if 'error' in payload:
            payload['error'] = _v6_error_text(payload['error'], tuple(self._kept_functions))
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
        # round 5d (user; C50 g013 ~23K tokens, g046): a refused action() call raises in the snippet, and a try/except
        # around it hid the refusal; the tool result states it whether or not the code caught it
        refused = [r for r in ((sandbox_result or {}).get('action_results') or [])
                   if isinstance(r, dict) and not r.get('executed')
                   and (r.get('stop_reason') in REFUSAL_REASONS or (r.get('error') and not r.get('stop_reason')))]
        if refused:
            payload['refused_action_calls'] = _refusal_text(refused, self._v67_reasons(),   # v6.7 SOFT; v7.0 item 16
                                                            shown=payload.get('error') if V70H_REFUSAL1 else None)
            if V70H_REFUSAL1 and payload.get('error'):
                self._v6_stats['v70h_refusals_with_error'] = self._v6_stats.get('v70h_refusals_with_error', 0) + 1
            if V70H_RESET0:                           # v7.0 item 19c: each refused 0-action RESET, one event
                for r in refused:
                    if r.get('stop_reason') == 'reset_zero':
                        self._v6_log('v70h-reset0-refused', dict(level=self._v61_level,
                                                                 actions=list(r.get('skipped_actions') or [])))
            if HIDE_RESET69:                          # v6.9 REX HIDE_RESET (arm B): the refused model RESETs, one event each
                for r in refused:
                    if r.get('stop_reason') in ('reset_hidden', 'hr69_game_over'):
                        self._v6_log('v69-reset-refused', dict(level=self._v61_level, kind=r.get('stop_reason'),
                                                               actions=list(r.get('skipped_actions') or [])))
        if 'keepable_functions' not in (sandbox_result or {}):
            # The sandbox did not finish (timeout or crash): no intake, and the library stays as it was (spec §5.5).
            # dfranzen's retention would read the missing list as empty and drop every program (S1, 2026-10-03: 13
            # timeouts in long games each wiped the library).
            return
        before = dict(self._kept_functions)
        super()._record_retained_functions(sandbox_result, payload)
        if payload.get('function_retention'):        # round 4 (A3 cn04: 224 repeated notes, ~10K tokens; ar25, re86)
            noted = self.__dict__.setdefault('_v62_const_noted', {}).setdefault(self._v61_level, set())
            keep = []
            for line in str(payload['function_retention']).split('\n'):
                m = re.match(r'Your function (\S+) was not retained: only an UPPER_CASE name bound to a literal', line)
                if m:
                    if m.group(1) in noted:
                        continue
                    noted.add(m.group(1))
                keep.append(line)
            if keep:
                payload['function_retention'] = '\n'.join(keep)
            else:
                payload.pop('function_retention', None)
        after = self._kept_functions
        session = getattr(self._step_env_callback, '__self__', None)
        log = self.__dict__.setdefault('_v62_library_log', [])
        for name, src in after.items():                     # round 5l (item 110, S4): the versioned log
            if before.get(name) != src:
                log.append(dict(name=name, level=self._v61_level, step=getattr(session, 'action_count', None), source=src,
                                status='saved'))
        for name in before:
            if name not in after:
                log.append(dict(name=name, level=self._v61_level, step=getattr(session, 'action_count', None),
                                status='removed'))
        for name, src in after.items():
            if before.get(name) != src:
                prev_level = self._v61_library_levels.get(name)
                if name in before and prev_level is not None and prev_level != self._v61_level:
                    # round 5d (C50 g016, g020, g022, g042): the version from before this level, restored if the level
                    # restarts (the restart archived the redefined program, and its earlier-level version was lost)
                    self.__dict__.setdefault('_v62_pre_level', {})[name] = (prev_level, before[name],
                                                                            self._v6_library_steps.get(name))
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
        """v7.4: a pending rebase flag (the rebase point, a return, an untaken rebase park) rides on this request (its first
        external append) and the server's rebase receipt is recorded; in refresh mode the HC1b path unchanged."""
        if not V74:
            return self._v6_generate_hc1b(external)
        flag = self.__dict__.pop('_v74_rebase_next', None) if external else None
        result = self._v6_generate_hc1b(external, **({'rebase': True} if flag else {}))
        self._v74_after(result, flag)
        return result

    def _v6_generate_hc1b(self, external, **kw):
        """generate with HC1b: one retry with the tool results replaced by N6; then a minimal notice."""
        try:
            return self._v6_session.generate(external, **kw)
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
                return self._v6_session.generate(retry, **kw)
            except CapacityRefusal:
                self._v6_stats['capacity_refusals'] += 1
                minimal = [dict(m, content='Output omitted.') if m.get('role') == 'tool' else
                           (dict(m, content=[p for p in m['content'] if p.get('type') == 'text'])
                            if isinstance(m.get('content'), list) else m) for m in external]
                return self._v6_session.generate(minimal, **kw)

    def analyze(self, state_path, action_num, **kwargs):
        """Round 5d: a session error (a protocol or transport failure, an append-contract violation, a capacity refusal
        that survived its retries) no longer crashes the game. The broken session is dropped and the next call restarts
        the level in place (RESET and a new conversation, the same path as the token restart, without a swap-out or the
        late gate), at most PROTOCOL_RESTARTS times per game."""
        try:
            return self._v62_analyze(state_path, action_num, **kwargs)
        except (requests.RequestException, AppendOnlyViolation, CapacityRefusal) as exc:
            n = self._v6_stats['protocol_restarts'] = self._v6_stats.get('protocol_restarts', 0) + 1
            self._v6_log('v62-session-error', dict(error='%s: %s' % (type(exc).__name__, str(exc)[:300]), n=n,
                                                   cap=PROTOCOL_RESTARTS, restart=n <= PROTOCOL_RESTARTS))
            if n > PROTOCOL_RESTARTS:
                raise
            self._v62_force_restart = True
            return ta.AnalyzerTurnResult(step_executed=False, retryable_failure=True)

    def _v62_analyze(self, state_path, action_num, valid_actions=None, step_env=None, transcript_path=None,
                     analysis_step=None, transcript_updated=None, request_timeout_seconds=None, should_stop=None):
        if not state_path.exists():
            return None
        self._ensure_session(state_path)
        self._step_env_callback = step_env
        _v65_game = getattr(step_env, "__self__", None)          # v6.5 C6: the session's batch / stale receipts
        if _v64_tr._K79_KEEPALIVE_S > 0 and _v65_game is not None:   # v7.9b K5: this game's session for the transport's keep-alive
            _v64_tr._K79_TLS.session = _v65_game
        if _v65_game is not None:
            _v65_game.__dict__['v65_stats'] = self._v6_stats
        if self._noop_repeat_guard is not None:                   # v6.2 fix: his analyze() attaches the guard hook
            solver_obj = getattr(step_env, "__self__", None)
            if solver_obj is not None:
                solver_obj.action_guard_hook = self._action_guard_hook
        self._current_valid_actions = ta._normalize_valid_actions(valid_actions)
        current_frame, history_entries = load_runtime_state(state_path)
        level_now = getattr(current_frame, 'level', None)          # item 6: tokens on the level
        if level_now != self._v61_tokens_level:
            self._v61_tokens_level = level_now
            if EXT200 and self.__dict__.get('_v67x_ext'):     # v6.7x EXT200: a level change (a clear) that the acting
                self._v67x_end(True, self._v61_level_tokens, 'level_change')   # turn did not report ends the attempt
            self._v67_prev_level_tokens = self._v61_level_tokens   # v6.7 REFRESH receipt: tokens on the completed level
            self._v61_level_tokens = 0
            if RESTART_ACROSS69 or PARK69:                     # v6.9 RAB: a level change starts the level's count and the rung
                self._v69_att = 0
                self.__dict__.pop('_v69_pk', None)
            if V70_ON:                                         # v7.0: the cleared level's clock and actions; a new count
                self._v70_level_change(level_now, history_entries)
            self._v61_noact_level_from = None
            self._v62_summary_state = None                     # round 5l (item 111)
            self._v62_attempt_request = None                   # round 5l review (Fable E2)
            self._v62_noact_tokens, self._v62_noact_warned = 0, 0   # v6.3 item B: a level change starts a new stretch
        if V76:                                                 # v7.6 HYBRID: after the rebase's summary turn this turn's first
            self._v76_turn(level_now)                           # append rebases
        if LEVELS_COMPLETE_END and self._v6_started and self._v63_levels_complete_end(level_now):
            return ta.AnalyzerTurnResult(step_executed=False)   # v6.3 item E: the game ended; the solver sees it
        if V70_ON:                                              # v7.0: the turn boundary (index, registry, marks)
            self._v70_turn(current_frame, history_entries, level_now)
        if REFRESH:                                             # v6.7 REFRESH: at a level-up of a long conversation (before the
            if not self._v6_started:                            # position checks, which it pre-empts)
                self._v67_conv_level = level_now
            if self._v67_refresh_gate(level_now) == 'now':
                current_frame, history_entries = self._v67_refresh(state_path, current_frame, history_entries)
        if INLEVEL69 and self._v69_ir_gate(level_now) == 'now':   # v6.9 INLEVEL: after its notes turn (pre-empts the
            if not (RESTART_ACROSS69 and self._v69_rt_supersede(level_now, current_frame, history_entries)):   # (v6.9 RAB:
                _ir73 = self.__dict__.get('_v69_ir') or {}              # the budget restart wins; first)
                _pk73 = bool((PARK69 or V70_PARKS or V71) and _ir73.get('park') and self._v69_pk_park(current_frame))
                if _pk73 and should_stop is not None and should_stop():
                    return ta.AnalyzerTurnResult(step_executed=False)   # the run ended while this game was parked
                if V74 and not V76:                              # v7.4: the return / an untaken park: no new conversation
                    current_frame, history_entries = self._v74_return(state_path, current_frame, history_entries, _ir73,
                                                                      _pk73)
                elif V73 and not V76 and not _pk73 and _ir73.get('trigger') == 'tokens':   # v7.3: nobody took the lane: no
                                                                 # refresh (v7.6 HYBRID: the refresh in place, as v7.1)
                    self._v73_keep(level_now, current_frame)
                else:
                    current_frame, history_entries = self._v69_ir_refresh(state_path, current_frame,   # position checks and
                                                                          history_entries)    # the restart decisions)
        position_notes = self._v62_position_notes_first()       # round 5m: the notes turn before a position restart
        position_restart = False if position_notes else self._v62_position_check()  # round 5g: sets _v62_force_restart
        forced = getattr(self, '_v62_force_restart', False)     # round 5d: after a session error, in place
        if EXT200 and not (forced or position_notes):      # v6.7x EXT200: the attempt's one assignment at its token budget
            self._v67x_assign(level_now, current_frame, history_entries)
        budget67x = self._v67x_budget(level_now) if EXT200 else RESTART_TOKENS   # v6.7x EXT200: this attempt's budget
        budget_hit = bool(RESTART_TOKENS and (self._v69_att_tokens() if RESTART_ACROSS69 else self._v61_level_tokens)
                          >= budget67x)              # v6.9 RAB RESTART_ACROSS: the level's count over conversations
        noact_hit = bool(NOACT_TOKENS and self._v62_noact_tokens >= NOACT_TOKENS)   # v6.3 item B: the same path
        if V72_RML:                                             # v7.2b: REFRESH_MIN_LEFT replaces the late restart rule: the
            if (self._v6_started and budget_hit and self._v62_summary_state is None   # budget trigger is skipped late in
                    and self._v72_late('budget', level_now)):   # the run (before its notes turn); the no-action stall is
                budget_hit = False                              # exempt (the user 10-20)
            time_ok = True
        else:
            time_ok = self._v62_restart_time_ok(level_now) if (self._v6_started and (budget_hit or noact_hit)) else True
        due = bool(self._v6_started and (budget_hit or noact_hit) and time_ok)
        self._v62_noact_late = bool(noact_hit and not time_ok)   # item C: the attempt continues in place (no more breaks)
        if self._v62_summary_state == 'done' and (budget_hit or noact_hit):
            due = True                  # round 5l review (Fable): the model was told the level restarts; the late gate
                                        # is not re-applied between the notes turn and the restart
        if LOOP and self._v62_restart_reason == 'loop' and (self._v62_summary_state == 'done' or not RESTART_NOTES):
            due = True                  # v6.7 LOOP: rung 2 / 3 after its notes turn (at once without notes); no late gate
        summary_turn = False
        if (due or position_notes) and not forced and self._v62_summary_state is None and RESTART_NOTES:
            due, summary_turn = False, True                     # round 5l (item 111, D1): first the notes turn
            self._v62_summary_state = 'requested'
            self._v62_restart_reason = 'position' if position_notes else ('tokens' if budget_hit else 'noact')
            kind67, why67 = self._v67_restart_kind(current_frame, history_entries)   # v6.7 SOFT: decided with the request,
            self._v67_pending = dict(level=level_now, kind=kind67, why=why67)         # which states it (no action runs before)
        if V70_NHR and (forced or due):                         # v7.0 NO_HARD_RESTART: an in-level refresh instead
            current_frame, history_entries = self._v70_nhr_refresh(state_path, current_frame, history_entries, forced,
                                                                   position_restart, budget_hit, noact_hit, should_stop)
            if V71 and self.__dict__.pop('_v71_stopped', False):   # v7.1: the run ended while this game was parked
                return ta.AnalyzerTurnResult(step_executed=False)
            forced = due = False
        if forced or due:
            reason = self._v62_restart_reason if (not forced and self._v62_restart_reason) else (
                'position' if position_restart else 'tokens' if budget_hit else 'noact' if noact_hit else 'forced')
            if not forced and self._v62_swap_out(current_frame, reason) and should_stop is not None and should_stop():
                # the run ended while this game waited for a lane; round 5d (dry run v62dry37 ls20): None made the solver
                # raise "Analyzer did not return a result." and the game ended `crashed`; an empty turn lets its loop see
                # the stop and end the game `cancelled`
                return ta.AnalyzerTurnResult(step_executed=False)
            self._v62_force_restart = False
            self._v62_notes_asked = (level_now if self._v62_summary_state == 'done' and (not forced or position_restart)
                                     else None)                 # round 5m: also after the position restart's notes turn
            self._v62_summary_state = None
            self._v62_restart_reason = None
            if reason == 'noact':                              # v6.3 item B: counted and logged as its own population
                self._v6_stats['noact_restarts'] = self._v6_stats.get('noact_restarts', 0) + 1
                self._v6_log('v62-noact-restart', dict(level=level_now, tokens=self._v62_noact_tokens,
                                                       tokens_on_level=self._v61_level_tokens))
            current_frame, history_entries = self._v61_restart(state_path, current_frame, history_entries, reason=reason)
        if summary_turn:
            self._v62_attempt_request, self._v62_attempt_saved = level_now, False
            self._v64_notes_retry = None                       # v6.4 A1
            self._step_env_callback = _NoActions(step_env, self._v67_key('summary_turn_refusal'))   # v6.7 SOFT
            self._v6_stats['summary_turns'] = self._v6_stats.get('summary_turns', 0) + 1
            if position_notes:
                self._v6_stats['position_notes_turns'] = self._v6_stats.get('position_notes_turns', 0) + 1
            self._v6_log('v62-summary-turn', dict(level=level_now, tokens_on_level=self._v61_level_tokens,
                                                  noact_tokens=self._v62_noact_tokens, reason=self._v62_restart_reason,
                                                  **({'attempt_tokens': self._v69_att_tokens()} if RESTART_ACROSS69 else {})))
        step_executed = False
        stop_reason = None
        try:
            noact_request = summary_turn and self._v62_restart_reason == 'noact'      # v6.3 item B
            req_key = ('attempt_summary_request_position' if position_notes and 'attempt_summary_request_position' in TEXTS
                       else 'attempt_summary_request_noact' if noact_request and 'attempt_summary_request_noact' in TEXTS
                       else 'attempt_summary_request')    # round 5m: the position restart names its own reason
            across = self._v69_rt_across(req_key, level_now) if (RESTART_ACROSS69 and summary_turn) else None   # v6.9 RAB
            request = (_fill(TEXTS[self._v67_key(req_key)], level=level_now,      # v6.7 SOFT: the soft wording
                             tokens=format(self._v62_noact_tokens if noact_request else self._v61_level_tokens, ','))
                       if summary_turn and across is None else across or '')
            if V77 and request:                       # v7.7 (the user 10-24): the last level-up summary, for the notes to update
                request += self._v77_notes_prev(level_now)
            notes_texts = []                          # round 5l review (Fable): every completion of the notes turn
            if self._v6_resume:
                text = self._v6_resumption_text(current_frame, valid_actions, history_entries)
                if V73:                                   # v7.3: the conversation continued after a notes turn
                    text = self._v73_kept_text(text)
                if request:                           # round 5l review (Fable T4): no "call action(...)" reminder above it
                    text = text.replace(TEXTS['reminder'], '').rstrip() + '\n' + request
                if INLEVEL69 and not summary_turn:         # v6.9 INLEVEL: the in-level notes turn (at a resumption)
                    text, summary_turn = self._v69_ir_notes(text, level_now, step_env, 'resume', current_frame)
                if V70H_RESIDENT and self._v70h_board_resident(current_frame):   # v7.0 item 19b: the image is still there
                    content = [dict(type='text', text=text), dict(type='text', text=TEXTS['v70h_resident_pointer'])]
                    self._v6_stats['v70h_resident_pointers'] = self._v6_stats.get('v70h_resident_pointers', 0) + 1
                    self._v70h_ptr = dict(obs=self._v70h_img[1])          # IE: the residency after the next generation
                    self._v6_log('v70h-resident-pointer', dict(obs=self._v70h_img[1], step=getattr(current_frame, 'step', None)))
                else:
                    content = [dict(type='text', text=text), dict(type='text', text='Current board (unchanged).'),
                               current_grid_image_part(current_frame)]
                    if V70H_RESIDENT:
                        self._v6_stats['v70h_resident_images'] = self._v6_stats.get('v70h_resident_images', 0) + 1
                opener = {"role": "user", "content": content, "_arc3_observation": True}
                kind = 'resume'
                self._v6_resume = False
                self._v62_resume_error = False
                self._v6_pending_list = None
            else:
                text, kind = self._build_user_prompt(action_num, valid_actions=valid_actions, current_frame=current_frame,
                                                     history_entries=history_entries,
                                                     previous_step_summary=self._last_step_summary)
                if V73:                                   # v7.3: the conversation continued after a notes turn
                    text = self._v73_kept_text(text)
                if request:                           # round 5l review (Fable T4)
                    text = text.replace(TEXTS['reminder'], '').rstrip() + '\n' + request
                full_text = text                      # v6.5 C1: the signature reads the whole report
                key = _opener_key(text) if kind == 'same' else None
                if key and (TEXTS['game_over_no_restart'] in text or TEXTS['game_over_auto_reset'] in text
                            or 'Continuations that ended in GAME OVER' in text):
                    key = None                        # round 5l review (Fable T3): these stay in every report
                run = getattr(self, '_v62_collapse_run', 0)
                if key and key == getattr(self, '_v62_prev_opener_key', None) and run < 3:
                    # round 5l (C5, 392b g501: 75 identical turns of ~2K tokens): the same report is not repeated; the
                    # board image stays (item 78: the latest board always remains)
                    text = _collapse_opener(text)
                    self._v6_stats['collapsed_openers'] = self._v6_stats.get('collapsed_openers', 0) + 1
                    self._v62_collapse_run = run + 1
                else:
                    self._v62_collapse_run = 0        # every 4th identical report is shown in full
                if LOOP:                              # v6.7 LOOP: the ladder (rung 0 is C1's line); rung 2 / 3 make the notes turn
                    text, summary_turn = self._v67_loop_line(full_text, text, kind, summary_turn, level_now, step_env,
                                                             current_frame, history_entries)
                else:
                    text = self._v65_loop_line(full_text, text, kind)    # v6.5 C1
                if ACTREV69:                               # v6.9 ACT-REV shadow: log only (no model-facing change)
                    self._v69_as_report(kind, level_now, current_frame, history_entries)
                c70 = self.__dict__.get('_v70h_actrev') if V70H_ACTREV else None
                if c70 and not summary_turn:               # v7.0 item 17: ACT-REV's fire takes the 16-report fallback's place
                    req = self._v67_loop_trigger(level_now, step_env, current_frame, history_entries,
                                                 dict(p=c70.get('p'), k=c70.get('k'), n=int(c70.get('p') or 0) * int(c70.get('k') or 0)),
                                                 fallback=True)
                    self._v6_stats['v70h_actrev_fires'] = self._v6_stats.get('v70h_actrev_fires', 0) + 1
                    self._v6_log('v70h-actrev-fire', dict(level=level_now, step=getattr(current_frame, 'step', None),
                                                          p=c70.get('p'), k=c70.get('k'), consecutive=c70.get('consecutive'),
                                                          notes_turn=bool(req)))
                    if req:
                        text, summary_turn = text.replace(TEXTS['reminder'], '').rstrip() + '\n' + req, True
                if REFRESH and not summary_turn and kind != 'resume':   # v6.7 REFRESH: the completed level's summary turn
                    text, summary_turn = self._v67_refresh_summary(text, level_now, step_env)
                if INLEVEL69 and not summary_turn:         # v6.9 INLEVEL: the in-level notes turn
                    text, summary_turn = self._v69_ir_notes(text, level_now, step_env, kind, current_frame)
                self._v62_prev_opener_key = key
                opener = self._v6_opener(text, kind, current_frame, history_entries)
                if kind == 'takeover':
                    self._v61_takeover = False
            external = list(self._v6_pending_tools) + [opener]
            if not self._v6_started:
                external = [{"role": "system", "content": self._system_prompt}] + external
            self._v6_pending_tools = []
            self._v6_obs_count += 1
            if V70H_RESIDENT and kind != 'takeover' and any(p.get('type') == 'image_url' for p in opener['content']):
                # v7.0 item 19b: this observation's first image is the current board (a takeover's may be a clearing-step board)
                self._v70h_img = (self._v69_ir_conv(), self._v6_obs_count, current_frame.grid)
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
            if NOTES:                                     # v6.7 NOTES: the opener-size receipts per kind
                self._v67_opener_size(kind, text, opener)
            self._v6_log('v6-opener', dict(kind=kind, obs_index=self._v6_obs_count, text=text,
                                           **({'chars': len(text)} if NOTES else {}),
                                           images=sum(1 for p in opener['content'] if p.get('type') == 'image_url'),
                                           text_tokens=self._v63_text_tokens(opener['content']),   # v6.3 item G
                                           pending_tool_tokens=self._v63_text_tokens(
                                               [p for m in external if m.get('role') == 'tool' for p in
                                                (m['content'] if isinstance(m.get('content'), list)
                                                 else [dict(type='text', text=str(m.get('content') or ''))])])))
            self._turn_generated_tokens = 0
            while True:
                if should_stop is not None and should_stop():
                    stop_reason = 'stop_requested'
                    if external:                       # never leave a tool result unsent in a live session
                        self._v6_pending_tools = [m for m in external if m.get('role') == 'tool']
                    break
                result = self._v6_generate(external)
                self._v6_started = True
                if V70H_RESIDENT and self.__dict__.get('_v70h_ptr'):   # v7.0 item 19b receipt: the pointed image after this request
                    self._v70h_resident_after()
                external = []
                self._v6_stats['requests'] += 1
                self._accumulate_usage_tokens(result.usage)
                self._v61_level_tokens += int((result.usage or {}).get('completion_tokens') or 0)
                if RESTART_ACROSS69 or PARK69:                 # v6.9 RAB: the level's count over its conversations
                    self._v69_att = (int(self.__dict__.get('_v69_att') or 0)
                                     + int((result.usage or {}).get('completion_tokens') or 0))
                self._v62_noact_tokens += int((result.usage or {}).get('completion_tokens') or 0)   # v6.3 item B
                if V70_ON:                                     # v7.0: the level's count over everything, since admission
                    self._v70_tokens(int((result.usage or {}).get('completion_tokens') or 0))
                self._v6_log('model-turn', dict(usage=result.usage, finish_reason=result.finish_reason,
                                                pending=result.pending, closed_by=result.closed_by,
                                                message=result.message, seconds=round(result.seconds, 3),
                                                turn_tokens=self._turn_generated_tokens))
                if result.pending:                    # length pause: continue in place
                    self._v6_stats['continuations'] += 1
                    continue
                self._v6_stats['completions'] += 1
                msg = result.message
                # v6.2 round 3: the reply text first, then strings the code prints (A2 lp85: two summaries written only
                # in print() calls), then the reasoning (A2 lp85: the first match was the reasoning's draft). Round 5h:
                # kept for every completion, so a level-up can look at the clearing turn
                self._v62_last_texts = [ta._normalize_message_content(msg.get('content', '')),
                                        _printed_strings(msg.get('tool_calls')), ta._extract_reasoning_text(msg)]
                if self._v6_rule_capture is not None:
                    self._v6_capture_rule(self._v62_last_texts)
                if summary_turn:
                    notes_texts.append(self._v62_last_texts)
                if result.closed_by:                  # harness hard close / repeat close: recovery in the next append
                    self._v6_stats['closes'] += 1
                    if not (msg.get('tool_calls') or []):
                        rec = TURN_REPEAT_RECOVERY if result.closed_by == 'repeat_close' else TURN_BUDGET_RECOVERY
                        external = [{"role": "user", "content": rec}]
                        continue
                    # round 5i (item 99; 391 wx30 +253 min): the closed message already held a complete tool call, and a
                    # user message before its result is refused by the server ("New user input would split assistant
                    # tool calls from their results"), which cost a session error and a level restart. Its calls run as
                    # in any reply; their results continue the turn
                    self._v6_stats['closed_with_calls'] = self._v6_stats.get('closed_with_calls', 0) + 1
                    self._v6_log('v62-closed-with-calls', dict(closed_by=result.closed_by, calls=len(msg['tool_calls'])))
                reasoning = ta._extract_reasoning_text(msg)
                content = ta._normalize_message_content(msg.get('content', ''))
                tool_calls = json.loads(json.dumps(msg.get('tool_calls') or []))
                markup = ta._contains_tool_call_markup(reasoning, content)
                if not tool_calls and markup:
                    tool_calls = ta._recover_tool_calls_from_markup(reasoning, content)
                    if tool_calls:
                        self._v6_log('v6-markup-recovered', dict(count=len(tool_calls)))
                        for call in tool_calls:               # round 5h review: the tool result says so
                            call['_v62_recovered'] = True
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
                    bad = _v64_compile_error(code) if code else None
                    if bad:                                   # v6.4 A5 (12-h audit, sf06f ls20 T499: a tag fragment ran)
                        call['_v64_compile_error'] = bad
                        call['_v62_defect'] = call_defect(raw_text)
                        self._v6_stats['reparse_uncompiled'] = self._v6_stats.get('reparse_uncompiled', 0) + 1
                        self._v6_log('v64-reparse-uncompiled', dict(error=bad, defect=call['_v62_defect']))
                        code = None
                    if code:
                        fn['arguments'] = json.dumps({'code': code})
                        call['function'] = fn
                        call['_v62_recovered'] = True
                        call['_v62_defect'] = call_defect(raw_text)   # round 5i (item 98): which tag was missing
                        self._v6_stats['reparsed'] = self._v6_stats.get('reparsed', 0) + 1
                        self._v6_log('v61-reparse', dict(recovered=True, chars=len(code), defect=call['_v62_defect'],
                                                         code=code[:6000]))   # v6.2: auditable (sk48 A2: 203 calls)
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
                if summary_turn and (malformed or not tool_calls):
                    retry = self._v64_retry_text(notes_texts, 'the reply had no python call' if not tool_calls
                                                 else 'the python call was malformed')
                    if retry:                                           # v6.4 A1 (12-h audit, sf10f g525 / tu93)
                        external = [{"role": "user", "content": retry}]
                        continue
                    self._v62_end_summary_turn(notes_texts)             # round 5l (item 111): one request, no nudge
                    stop_reason = 'summary_turn'
                    break
                if malformed or not tool_calls:
                    kind_na, warn = self._v62_noact_check(history_entries, valid_actions)   # v6.3 item B
                    if kind_na == 'limit':
                        self._v6_resume = True
                        stop_reason = 'noact_limit'
                        break
                    if warn:                                   # the warning takes the slot of the nudge (user 10-12)
                        self._v6_stats['nudges_replaced'] = self._v6_stats.get('nudges_replaced', 0) + 1
                        external = [{"role": "user", "content": warn}]
                        continue
                    if malformed:
                        key, nudge = 'N3', TEXTS['N3']
                    elif markup:
                        key, nudge = 'N4', TEXTS['N4']
                    elif self._v6_degenerate(content) or self._v6_degenerate(reasoning):
                        key, nudge = 'N5', TEXTS['N5']
                    else:
                        key, nudge = 'N1', TEXTS['N1']
                        last = [l.strip() for l in (reasoning or content or '').split('\n') if l.strip()]
                        if last and last[-1].endswith(_V64_INJECTED_TAILS):     # v6.4 A3 (sf10f bx35 / g024; IE)
                            self._v6_stats['n1_quote_skipped'] = self._v6_stats.get('n1_quote_skipped', 0) + 1
                        elif last and PLAN_WORDS.search(last[-1]) and len(last[-1]) <= 300:
                            nudge += ' ' + _fill(TEXTS['N1_plan'], last_line=last[-1])
                    self._v6_stats['nudges'][key] = self._v6_stats['nudges'].get(key, 0) + 1
                    self._v6_log('v6-nudge', dict(kind=key))
                    external = [{"role": "user", "content": nudge}]
                    continue
                tool_messages = []
                acted = False
                saved = False                          # round 5l (item 108): a call of this reply saved a summary
                for i, call in enumerate(tool_calls):
                    if acted:
                        tool_messages.append({"role": "tool", "tool_call_id": call.get('id', ''),
                                              "content": json.dumps({"tool": "python", "error": (
                                                  "Not executed: an earlier call in this reply executed actions, "
                                                  "which ended the turn.")})})
                        continue
                    if call.get('_v61_format_error'):
                        err = _FORMAT_ERROR
                        if call.get('_v64_compile_error'):    # v6.4 A5 (IE): the recovered text's own error and the cause
                            err += ' The text recovered from the call did not compile (%s)%s.' % (
                                call['_v64_compile_error'],
                                ('; its markup: ' + call['_v62_defect']) if call.get('_v62_defect') else '')
                        tool_messages.append({"role": "tool", "tool_call_id": call.get('id', ''),
                                              "content": json.dumps({"tool": "python", "error": err})})
                        self._v61_noact_calls += 1
                        continue
                    fn = call.get('function') or {}
                    raw = fn.get('arguments', '{}')
                    args = json.loads(raw) if isinstance(raw, str) else (raw if isinstance(raw, dict) else {})
                    refusal = self._v67_loop_refusal(args) if LOOP else None   # v6.7 LOOP rung 1: a repeat of the cycle
                    if refusal:
                        tool_messages.append({"role": "tool", "tool_call_id": call.get('id', ''),
                                              "content": json.dumps({"tool": "python", "error": refusal})})
                        self._v61_noact_calls += 1
                        continue
                    t0 = time.monotonic()
                    self._v6_tool_images = []
                    self._v62_shown, self._v62_last_shown = [], 0      # round 5m (item 124): this call's show_board images
                    game = getattr(self._step_env_callback, '__self__', None)
                    n0 = len(getattr(game, 'history_entries', None) or [])
                    dispatch = self._dispatch_tool(state_path, str(fn.get('name', '')).strip(), args)
                    if REFRESH_PROGRAMS68:                     # v6.8 REFRESH_PROGRAMS: this executed call's program names
                        self._v68_count_program_refs(args.get('code'))
                    imgs, self._v6_tool_images = self._v6_tool_images, []
                    n_shown = min(len(imgs), self.__dict__.pop('_v62_last_shown', 0) or 0)
                    self._v6_log('tool-output', dict(seconds=round(time.monotonic() - t0, 3), acted=dispatch.step_executed,
                                                     history_len=len(history_entries), content=dispatch.content,
                                                     converted_boards=len(imgs) - n_shown, shown_boards=n_shown))
                    content = dispatch.content
                    if imgs:
                        self._v6_stats['converted_boards'] = self._v6_stats.get('converted_boards', 0) + len(imgs) - n_shown
                        content = [dict(type='text', text=dispatch.content)]
                        for cap, part in imgs:
                            content += [dict(type='text', text=cap), part]
                    if ANIM_PUSH and dispatch.step_executed:
                        push = self._v6_anim_push(game, n0)
                        if push:
                            if isinstance(content, str):
                                content = [dict(type='text', text=content)]
                            content = content + push
                    if call.get('_v62_recovered') and 'markup_recovered' in TEXTS:
                        defect = call.get('_v62_defect')
                        lines = len(str(args.get('code') or '').strip('\n').split('\n'))   # round 5l (A7, 392b te01)
                        note = (_fill(TEXTS['markup_recovered_defect'], what=defect, lines=_lines_text(lines))
                                if defect and 'markup_recovered_defect' in TEXTS
                                else _fill(TEXTS['markup_recovered'], lines=_lines_text(lines)))
                        content = _v62_with_note(content, note)
                        self._v6_stats['recovered_notes'] = self._v6_stats.get('recovered_notes', 0) + 1
                    tool_messages.append({"role": "tool", "tool_call_id": call.get('id', ''), "content": content})
                    if dispatch.step_executed:
                        acted = True
                        if LOOP:
                            self._v67_last_code = args.get('code')      # v6.7 LOOP: the acting call's code (the cycle's calls)
                        if EXT200 and self.__dict__.get('_v67x_ext'):   # v6.7x EXT200: a clear ends the assigned attempt
                            self._v67x_cleared()
                        if REFRESH and self.__dict__.get('_v67_refresh_watch'):   # v6.7 REFRESH receipt
                            self._v67_refresh_first_action()
                        if INLEVEL69 and self.__dict__.get('_v69_ir_watch') is not None:   # v6.9 INLEVEL receipt
                            self._v69_ir_first_action()
                        if V74 and self.__dict__.get('_v74_watch') is not None:   # v7.4 receipt: tokens to the action
                            self._v74_first_action()
                        self._v61_noact_calls = 0
                        self._v62_noact_tokens, self._v62_noact_warned = 0, 0     # v6.3 item B: the budget is restored
                        if V70_ON:                             # v7.0: an action since the admission (the swap's hysteresis)
                            self._v70_acted()
                    elif self._v62_saved_in_call:           # round 5l (item 108, Fable 2.1): saving the requested summary
                        saved = True                        # is not an idle call (no-action line, yield)
                    else:
                        self._v61_noact_calls += 1
                if acted:
                    step_executed = True
                    self._v6_pending_tools = tool_messages     # sent with the next opener
                    break
                if LOOP and not summary_turn and (self.__dict__.get('_v67_loop') or {}).get('refused', 0) >= 2:
                    req = self._v67_loop_trigger(level_now, step_env, current_frame, history_entries)   # v6.7 LOOP rung 2 / 3
                    if req:                                    # the 2nd refused repeat: this turn becomes the notes turn
                        summary_turn = True
                        external = list(tool_messages) + [{"role": "user", "content": req}]
                        continue
                if summary_turn:                               # round 5l (item 111): the turn ends after this reply
                    retry = self._v64_retry_text(notes_texts, self._v64_notes_reason(tool_messages))
                    if retry:                                  # v6.4 A1: the tool results first, then the retry note
                        external = list(tool_messages) + [{"role": "user", "content": retry}]
                        continue
                    self._v62_end_summary_turn(notes_texts)
                    self._v6_pending_tools = tool_messages
                    stop_reason = 'summary_turn'
                    break
                kind_na, warn = self._v62_noact_check(history_entries, valid_actions)   # v6.3 item B
                if kind_na == 'limit':                         # the budget is spent: the turn ends; the next call's
                    self._v6_pending_tools = tool_messages     # between-turn check runs the notes turn and the restart
                    self._v6_resume = True
                    self._v62_resume_error = self._v62_last_call_errored(tool_messages)
                    stop_reason = 'noact_limit'
                    break
                breaker = self._v61_breaker(tool_messages)
                if warn:                                       # the warning takes precedence over the breaker (user 10-12)
                    if breaker:
                        self._v6_stats['breaker_replaced'] = self._v6_stats.get('breaker_replaced', 0) + 1
                    breaker = None
                    tool_messages = tool_messages + [{"role": "user", "content": warn}]
                if breaker == 'yield':
                    self._v6_stats['breaker_yields'] = self._v6_stats.get('breaker_yields', 0) + 1
                    self._v6_pending_tools = tool_messages
                    self._v6_resume = True
                    self._v62_resume_error = self._v62_last_call_errored(tool_messages)   # round 5g
                    stop_reason = 'yield'
                    break
                if breaker:
                    self._v6_stats['breaker_notes'] = self._v6_stats.get('breaker_notes', 0) + 1
                    tool_messages = tool_messages + [{"role": "user", "content": breaker}]
                # round 5i (user 10-07): one automatic message at a time. A cue inside this reply (the nudge, the closing
                # cue, a repeat cue) or a breaker note takes priority over the token yield, which waits for the next
                # non-acting call (386: 53 of 782 yields came right after a reply with the soft cue)
                held = (sorted(set(getattr(result, 'cues', ()) or ()) & YIELD_HELD_BY) + (['breaker_note'] if breaker else [])
                        + (['noact_warning'] if warn else []))
                if YIELD_TOKENS and self._turn_generated_tokens >= YIELD_TOKENS and held:
                    self._v6_stats['yields_held'] = self._v6_stats.get('yields_held', 0) + 1
                    self._v6_log('v62-yield-held', dict(turn_tokens=self._turn_generated_tokens, by=held))
                elif YIELD_TOKENS and self._turn_generated_tokens >= YIELD_TOKENS and not saved:
                    self._v6_stats['yields'] += 1
                    self._v6_log('v6-yield', dict(turn_tokens=self._turn_generated_tokens))
                    self._v6_pending_tools = tool_messages
                    self._v6_resume = True
                    self._v62_resume_error = self._v62_last_call_errored(tool_messages)   # round 5g
                    stop_reason = 'yield'
                    break
                external = tool_messages
        finally:
            self._step_env_callback = None
            self._current_valid_actions = []
        return ta.AnalyzerTurnResult(step_executed=step_executed, reasoning='',
                                     yielded_control=stop_reason == 'yield')

    # ----------------------------------------------------------------------------------------------- v6.3
    def _v62_noact_check(self, history_entries, valid_actions):
        """v6.3 item B: ('limit', None) when the no-action budget is spent; ('warn', text) the first time the count crosses
        a NOACT_WARN threshold in this stretch (the untried-actions list rides with the first warning); else (None, None)."""
        k = self._v62_noact_tokens
        if NOACT_TOKENS and k >= NOACT_TOKENS:
            return (None, None) if getattr(self, '_v62_noact_late', False) else ('limit', None)
        n = sum(1 for t in NOACT_WARN if k >= t)
        if n <= self._v62_noact_warned:
            return None, None
        self._v62_noact_warned = n
        untried = self._v61_untried(history_entries, valid_actions)
        limit = format(NOACT_TOKENS, ',') if NOACT_TOKENS else ''
        early = max(0, len(NOACT_WARN) - 2)                   # v6.6: thresholds before the last two get the facts-only cue
        if n <= early and 'noact_warn_0' in TEXTS:
            text = _fill(TEXTS['noact_warn_0'], k=format(k, ','), untried=', '.join(untried) or '-')
            if not untried:
                text = text.split(' Actions not yet executed')[0]
        elif n - early <= 1:
            clause = (_fill(TEXTS[self._v67_noact_key('noact_limit_clause', history_entries)], limit=limit)
                      if NOACT_TOKENS else '')                # v6.7 SOFT: the restart the limit would bring
            text = _fill(TEXTS['noact_warn_1'], k=format(k, ','), limit_clause=clause, untried=', '.join(untried) or '-')
            if not untried:                                   # no vacuous "every valid action has been tried"
                text = text.split(' Actions not yet executed')[0]
        else:
            text = (_fill(TEXTS[self._v67_noact_key('noact_warn_2', history_entries)], k=format(k, ','), limit=limit)
                    if NOACT_TOKENS else _fill(TEXTS['noact_warn_2_nolimit'], k=format(k, ',')))   # v6.7 SOFT
        key = 'noact_warnings_%d' % (n if early else min(n, 2))
        self._v6_stats[key] = self._v6_stats.get(key, 0) + 1
        self._v6_log('v62-noact-warning', dict(which=n, tokens=k, threshold=NOACT_WARN[n - 1], untried=untried,
                                               calls=self._v61_noact_calls))
        return 'warn', text

    def _v63_levels_complete_end(self, level_now):
        """v6.3 item E (guarded fallback): True when this game's level completions have reached number_of_levels while the
        engine still reports NOT_FINISHED (official games report WIN, so this never fires on them): the session is closed
        and the game finished (taaf scores the completed levels; its step() would assert on any further action)."""
        game = self._v61_session()
        tg = getattr(game, 'game', None)
        run = getattr(tg, 'game_run', None)
        if run is None:
            return False
        try:
            n = int(getattr(tg, 'number_of_levels', 0) or 0)
            done = int(getattr(run, 'levels_completed', 0) or 0)
            state = tg.current_state.raw.state
        except Exception:  # noqa: BLE001
            return False
        import arcengine
        if n <= 0 or done < n or state == arcengine.GameState.WIN or getattr(run, 'state', 'playing') != 'playing':
            return False
        self._v6_stats['levels_complete_ends'] = self._v6_stats.get('levels_complete_ends', 0) + 1
        self._v6_log('v63-levels-complete-end', dict(level=level_now, levels_completed=done, n_levels=n,
                                                     engine_state=str(getattr(state, 'name', state)),
                                                     tokens_on_level=self._v61_level_tokens))
        try:
            self._v6_session.close()
        except Exception:  # noqa: BLE001
            pass
        run.solver_note = 'v6.3 item E: all %d levels completed without a WIN state; the game was ended' % n
        tg.finish_game()
        return True

    # ----------------------------------------------------------------------------------------------- round 5g
    def _v63_text_tokens(self, parts):
        """v6.3 item G: served-tokenizer count of the text parts of a content list (images are counted by the server)."""
        try:
            return sum(self._v6_count(p.get('text') or '') for p in (parts or []) if isinstance(p, dict) and p.get('type') == 'text')
        except Exception:  # noqa: BLE001
            return None

    def _v62_last_call_errored(self, tool_messages):
        """round 5g: the yield resumption names an error when the last Python call of the reply returned one (a tool
        error, the missing-`code` format error, or the third identical error that made the breaker yield)."""
        tools = [m for m in tool_messages if m.get('role') == 'tool']
        return bool(tools) and bool(self._v61_tool_error(tools[-1]))

    def _v62_resumption_key(self):
        if getattr(self, '_v62_resume_error', False) and 'resumption_error' in TEXTS:
            return 'resumption_error'
        if getattr(self, '_v62_noact_warned', 0) and 'resumption_noact' in TEXTS:   # v6.4 B3 (12-h audit, sf06f g541)
            self._v6_stats['resumption_noact'] = self._v6_stats.get('resumption_noact', 0) + 1
            return 'resumption_noact'
        return 'resumption'

    def _v62_position_notes_first(self):
        """round 5m (user 2026-10-08: "if it does, I think we should leave some margin for the model to write a summary if
        the current run is past say 50k"): True when the session has reached POSITION_RESTART and the notes turn of the
        token restart should come first: notes on, no session error pending, no notes turn yet in this conversation on
        this level, and at least POSITION_NOTES_MIN tokens generated on the level in this conversation. The restart (RESET
        and a new conversation, with the notes) follows that turn; below the minimum it is immediate, as before."""
        if not (POSITION_RESTART and POSITION_NOTES_MIN and RESTART_NOTES and getattr(self, '_v6_started', False)):
            return False
        if (self.__dict__.get('_v67_refresh_state') or {}).get('pending'):   # v6.7 REFRESH pre-empts the position restart
            return False
        if getattr(self, '_v62_force_restart', False) or self._v62_summary_state is not None:
            return False
        if (self._v61_level_tokens or 0) < POSITION_NOTES_MIN:
            return False
        receipt = getattr(getattr(self, '_v6_session', None), 'last_receipt', None) or {}
        return int(receipt.get('logical_tokens') or 0) >= POSITION_RESTART

    def _v62_position_check(self):
        """round 5g (user 2026-10-06): at POSITION_RESTART logical tokens in the session, restart the level in place
        (RESET and a new conversation, as after a session error) instead of letting the server rebase near its RoPE
        rows (a full re-prefill). Never reached in the first 7 v6.1/v6.2 runs (max 1.28M)."""
        if not POSITION_RESTART or not getattr(self, '_v6_started', False):
            return False
        if (self.__dict__.get('_v67_refresh_state') or {}).get('pending'):   # v6.7 REFRESH pre-empts the position restart
            return False
        receipt = getattr(getattr(self, '_v6_session', None), 'last_receipt', None) or {}
        n = int(receipt.get('logical_tokens') or 0)
        if n < POSITION_RESTART:
            return False
        self._v62_force_restart = True
        self._v6_stats['position_restarts'] = self._v6_stats.get('position_restarts', 0) + 1
        self._v6_log('v62-position-restart', dict(logical_tokens=n, limit=POSITION_RESTART))
        return True

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
        if V75_POLISH:                               # v7.5 TXT-10: the quoted error ends with a period before "Do not repeat"
            error = _v75_period(error)
        if 'code' in error and ('non-empty' in error or 'no `code`' in error):
            return _fill(_BREAKER, k=self._v61_error_streak, error=error)
        return ('The last %d python calls returned the same error: %s Do not repeat that call: change the code, '
                'or inspect the state first.' % (self._v61_error_streak, error))

    def _v61_restart(self, state_path, current_frame, history_entries, reason='tokens', kind=None):
        """item 6, run B: RESET the level and start a new conversation (system prompt, the level's first board, the
        completion summaries). The level's programs are archived; ledger, attempts and no-action counts start again.
        v6.3: `reason` ('tokens' | 'noact' | 'position' | 'forced') is logged.
        v6.7 SOFT: that is the hard kind. `kind` None: the kind the notes request stated (no action ran since), else as
        _v67_restart_kind decides from ARC3_V67_RESTART_KIND; 'hard' from a caller (a later loop-ladder rung) is always
        hard. A soft restart restarts only the conversation (a new session and the per-conversation state below) from the
        current board: no RESET, so the game's attempt record and start board are untouched and the attempt goes on; the
        level's programs are not archived and its history is not hidden. The event carries the kind, the reason for a hard
        restart under soft (hard_why) and the actions counted on the level so far."""
        game = self._v61_session()
        if game is None:
            return current_frame, history_entries
        import arcengine
        level = getattr(current_frame, 'level', None)
        tokens = self._v61_level_tokens
        step = getattr(current_frame, 'step', None)
        pending = self.__dict__.pop('_v67_pending', None) or {}
        if kind is None and pending.get('kind') and pending.get('level') == level:
            kind, why = pending['kind'], pending.get('why')
        else:
            kind, why = self._v67_restart_kind(current_frame, history_entries, kind)
        on_level = self._v67_actions_on_level(level, history_entries)
        if EXT200 and self.__dict__.get('_v67x_ext'):       # v6.7x EXT200: a harness restart (any reason) ends the attempt
            self._v67x_end(False, tokens, 'restart', reason=reason, history_entries=history_entries)
        loop = pending.get('loop') if reason == 'loop' and pending.get('level') == level else None   # v6.7 LOOP
        if kind == 'hard':
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
                            ('_v61_error_streak', 0),   # round 5l (B4, user): the death ledger is kept across it
                            ('_v62_noact_tokens', 0), ('_v62_noact_warned', 0),   # v6.3 item B: a new stretch
                            ):                    # round 5h (G6): the ledger's game-over numbers stay game-wide
            setattr(self, name, value)
        attempts = (getattr(game, 'v6_level_attempts', None) or {}).get(level, [])
        # round 5h (user, D2): attempts count from the level's start (decision 5 counted from the restart); the takeover
        # says that a RESET began this conversation (D1)
        archived, restored = (self._v62_archive_level_programs(level) if kind == 'hard'
                              else ([], []))            # v6.7 SOFT: the level's programs stay loaded
        self._v62_restart_archived = (level, len(archived))     # round 5l (B4): told in the takeover
        self._v62_attempt_request, self._v62_attempt_saved = None, False   # round 5l review (Fable E2)
        att69 = self._v69_att_tokens() if (RESTART_ACROSS69 or PARK69) else None   # v6.9 RAB: the level's count, logged below
        self._v61_level_tokens = 0
        if RESTART_ACROSS69 or PARK69:                         # v6.9 RAB: a harness restart starts the count and the rung again
            self._v69_att = 0
            self.__dict__.pop('_v69_pk', None)
        self._v61_restarts += 1
        self._v61_takeover = True
        current_frame, history_entries = load_runtime_state(state_path)
        self._v61_noact_level_from = len(history_entries)
        hidden = (self._v61_hide_level_history(history_entries, level) if RESTART_HISTORY == 'level' and kind == 'hard'
                  else None)                            # v6.7 SOFT: the level's history stays visible
        self._v67_note_restart(game, level, kind, why, reason, tokens, step, on_level, loop)
        self._v6_stats['restarts'] = self._v6_stats.get('restarts', 0) + 1
        self._v6_log('v61-restart', dict(level=level, step=step, tokens_on_level=tokens, attempts_before=len(attempts),
                                         reason=reason, kind=kind, hard_why=why, actions_on_level=on_level,
                                         archived=archived, restored=restored, summaries=sorted(self._v6_rules), hidden=hidden,
                                         lazy_history=_lazy_enabled(),
                                         **({'attempt_tokens': att69} if att69 is not None else {})))
        return current_frame, history_entries

    # ----------------------------------------------------------------------------------------------- v6.7 SOFT
    def _v67_restart_kind(self, current_frame, history_entries, kind=None):
        """v6.7 SOFT: (kind, why) of a harness restart from this board. The hard switch (default): ('hard', None). Under
        ARC3_V67_RESTART_KIND=soft: ('hard', why) when the caller asks for hard ('explicit'), when the game reports GAME
        OVER ('game_over': only a RESET continues) or when a dead-board rule fires ('dead_gameovers' | 'dead_window'),
        else ('soft', None). Read-only: the notes request and the no-action cues use it to state the coming restart.
        v6.7 LOOP: kind='soft' (a loop rung 2) is decided as under the soft switch, also under the hard switch (the soft
        texts come with ARC3_V67_LOOP=1)."""
        if V70_NHR:                                     # v7.0 NO_HARD_RESTART: every harness restart is a new conversation
            return 'soft', None                         # at the current board (the requests and cues state the soft form)
        if kind == 'hard':
            return 'hard', ('explicit' if RESTART_KIND == 'soft' else None)
        if RESTART_KIND != 'soft' and not (kind == 'soft' and 'takeover_soft' in TEXTS):
            return 'hard', None
        if _v62_game_over_now(self._v61_session()):
            return 'hard', 'game_over'
        dead = self._v67_dead_board(current_frame, history_entries)
        return ('hard', dead) if dead else ('soft', None)

    def _v67_dead_board(self, current_frame, history_entries):
        """v6.7 SOFT, the dead-board rules (env knobs; 0 = off). 'dead_gameovers': at least DEAD_GAMEOVERS GAME OVERs in
        this conversation on this level (fatal actions after its first entry: the level's start or the latest harness
        restart) came from a board identical to the current one, i.e. the continuation that ended in the GAME OVER, from
        its attempt's first board (the latest RESET, or the level's first board) to the fatal action, passed through it
        (boards compared as the death ledger compares them; a RESET that ended in GAME OVER is not counted, as in the
        ledger). 'dead_window': none of the last DEAD_WINDOW executed actions on this level (automatic restarts excluded)
        produced an interior (rows and columns 4-59) not seen earlier on the level. None when neither fires."""
        grid = getattr(current_frame, 'grid', None)
        entries = list(history_entries or [])
        if not grid or not entries or (DEAD_GAMEOVERS <= 0 and DEAD_WINDOW <= 0):
            return None
        first = 0                                       # the entry whose frame is the level's first board
        for i, e in enumerate(entries):
            if (getattr(e, 'result', None) or {}).get('level_completed'):
                first = i
        if DEAD_GAMEOVERS > 0:
            here, n = _v67_state(grid), 0
            for d in range(max(first + 1, int(self._v61_noact_level_from or 0)), len(entries)):
                r = getattr(entries[d], 'result', None) or {}
                if (not r.get('game_over') or r.get('automatic')
                        or str(getattr(entries[d], 'action', '') or '').upper() == 'RESET'):
                    continue
                a = d - 1
                while a > first and str(getattr(entries[a], 'action', '') or '').upper() != 'RESET':
                    a -= 1
                if any(getattr(entries[i], 'frame', None) is not None and _v67_state(entries[i].frame.grid) == here
                       for i in range(a, d)):
                    n += 1
            if n >= DEAD_GAMEOVERS:
                return 'dead_gameovers'
        if DEAD_WINDOW > 0:
            from inference.framework import solver as _sv
            b = int(getattr(_sv, 'NOOP_GUARD_BORDER_WIDTH', 4) or 0)
            seen, new = set(), []
            for i in range(first, len(entries)):
                f = getattr(entries[i], 'frame', None)
                if f is None:
                    continue
                inner = tuple(tuple(row[b:len(row) - b]) for row in f.grid[b:len(f.grid) - b])
                if i > first and not (getattr(entries[i], 'result', None) or {}).get('automatic'):
                    new.append(inner not in seen)
                seen.add(inner)
            if len(new) >= DEAD_WINDOW and not any(new[-DEAD_WINDOW:]):
                return 'dead_window'
        return None

    def _v67_key(self, key):
        """v6.7 SOFT: the TEXTS key for the pending restart: '<key>_soft' when the notes request states a soft restart."""
        pending = getattr(self, '_v67_pending', None) or {}
        return key + '_soft' if pending.get('kind') == 'soft' and key + '_soft' in TEXTS else key

    def _v67_noact_key(self, key, history_entries):
        """v6.7 SOFT: the no-action cue's clause for the restart the limit would bring from this board ('<key>_soft' when
        it would be soft; no action runs before the limit, so the board and the counts stay as they are)."""
        if V70_NHR and key + '_soft' in TEXTS:                 # v7.0 NO_HARD_RESTART: the limit brings a new conversation
            return key + '_soft'                               # at the current board
        if RESTART_KIND != 'soft' or key + '_soft' not in TEXTS:
            return key
        entries = list(history_entries or [])
        frame = getattr(entries[-1], 'frame', None) if entries else None
        return key + '_soft' if self._v67_restart_kind(frame, entries)[0] == 'soft' else key

    def _v67_reasons(self):
        """v6.7 SOFT: the refusal reasons for _refusal_text: REFUSAL_REASONS (None), or with the soft notes turn's own reason
        while a soft restart is pending."""
        if V76 and (self.__dict__.get('_v76') or {}).get('phase') == 'summary' and 'v76_summary_reason' in TEXTS:
            return dict(REFUSAL_REASONS, summary_turn=TEXTS['v76_summary_reason'])   # v7.6 HYBRID: the rebase's summary turn
        if INLEVEL69 and (self.__dict__.get('_v69_ir') or {}).get('pending') == 'notes' and 'inlevel_turn_reason' in TEXTS:
            return dict(REFUSAL_REASONS, summary_turn=TEXTS['inlevel_turn_reason'])   # v6.9 INLEVEL: its notes turn
        if (self.__dict__.get('_v67_refresh_state') or {}).get('pending') == 'summary' and 'refresh_turn_reason' in TEXTS:
            return dict(REFUSAL_REASONS, summary_turn=TEXTS['refresh_turn_reason'])   # v6.7 REFRESH: its summary turn
        if self._v67_key('summary_turn_refusal') == 'summary_turn_refusal' or 'summary_turn_reason_soft' not in TEXTS:
            return None
        return dict(REFUSAL_REASONS, summary_turn=TEXTS['summary_turn_reason_soft'])

    def _v67_actions_on_level(self, level, history_entries):
        """v6.7 SOFT: the actions counted on this level so far: the game run's per-level counter (taaf actions_per_level,
        which the official score reads: every executed action, RESETs and automatic restarts included); without a game run
        (tests), the history entries after the level's first board."""
        run = getattr(getattr(self._v61_session(), 'game', None), 'game_run', None)
        try:
            return int(run.actions_per_level[int(level) - 1])
        except Exception:  # noqa: BLE001
            pass
        entries = list(history_entries or [])
        first = 0
        for i, e in enumerate(entries):
            if (getattr(e, 'result', None) or {}).get('level_completed'):
                first = i
        return sum(1 for e in entries[first + 1:] if str(getattr(e, 'action', '') or '').strip())

    def _v67_note_restart(self, game, level, kind, why, reason, tokens, step, on_level, loop=None):
        """v6.7 SOFT: the restart's receipts (soft_restarts | hard_restarts, and hard_under_soft_<why>), its entry in the
        game session's per-level restart record (v67_level_restarts: kind, why, reason, step, the conversation's tokens on
        the level, the actions counted on the level, the attempts ended on the level), and the facts of a soft restart that
        the takeover states."""
        for key in (kind + '_restarts',) + (('hard_under_soft_' + why,) if why else ()):
            self._v6_stats[key] = self._v6_stats.get(key, 0) + 1
        rec = dict(kind=kind, why=why, reason=reason, step=step, tokens=tokens, actions_on_level=on_level,
                   attempts_ended=len((getattr(game, 'v6_level_attempts', None) or {}).get(level, [])))
        game.__dict__.setdefault('v67_level_restarts', {}).setdefault(level, []).append(rec)
        self._v67_takeover = dict(rec, level=level)
        if loop:                                        # v6.7 LOOP: rung 2 / 3 (the record, receipts, the soft restart event)
            rec['loop'] = dict(loop)
            self._v67_takeover['loop'] = dict(loop)
            key = 'loop_%s_restarts' % kind
            self._v6_stats[key] = self._v6_stats.get(key, 0) + 1
            if kind == 'soft':
                notes = list(self._v62_attempt_notes.get(level) or [])
                count = getattr(self, '_v6_count', None) or (lambda t: (len(t) + 3) // 4)
                self._v6_log('v67-soft-restart', dict(level=level, step=step, reason='loop', rung=loop.get('rung'),
                                                      notes_tokens=count(notes[-1]) if notes and self._v62_notes_asked == level
                                                      else 0))

    def _v67_takeover_lines(self, session, level, step):
        """v6.7 SOFT: the takeover's facts after a soft restart ([] after a hard one, whose lines stay as they are): this
        conversation continues the level from the current board; the previous conversation's tokens on the level, at a
        harness restart (not a game event); no RESET, so the level's actions so far stay counted; how the level's earlier
        attempts ended and its silent returns to the start (as after a hard restart, with the current attempt going on);
        the programs written on the level, still loaded. The notes and the summaries come as after a hard restart."""
        info = getattr(self, '_v67_takeover', None) or {}
        if info.get('kind') == 'inlevel' and info.get('level') == level and 'takeover_inlevel' in TEXTS:
            if V70_NHR and info.get('trigger') == 'loop' and 'takeover_loop_hard' in TEXTS:   # v7.0 NO_HARD_RESTART: the
                return [_fill(TEXTS['takeover_inlevel'], level=level, step=step), TEXTS['takeover_loop_hard']]   # loop rule
            if info.get('v73_notes_step') is not None:          # v7.3: the notes came from a kept notes turn of this attempt
                return [_fill(TEXTS['takeover_inlevel'], level=level, step=step),
                        _fill(TEXTS['v73_notes_age'], n=info['v73_notes_step'], m=step)]
            return [_fill(TEXTS['takeover_inlevel'], level=level, step=step)]   # v6.9 INLEVEL: one harness line
        if info.get('kind') == 'refresh' and info.get('level') == level and 'takeover_refresh' in TEXTS:
            done = info.get('done') or []                # v6.7 REFRESH: the fact line in place of the restart lines
            done = ('level %s' % _levels_text(done)) if len(done) == 1 else ('levels %s' % _levels_text(done))
            return [_fill(TEXTS['takeover_refresh'], level=level, done=done, logical=format(int(info.get('logical') or 0), ','))]
        if info.get('kind') != 'soft' or info.get('level') != level or 'takeover_soft' not in TEXTS:
            return []
        loop = info.get('loop') or {}
        if loop and 'takeover_loop' in TEXTS:            # v6.7 LOOP: rung 2's opener line in place of the soft line
            out = [_fill(TEXTS['takeover_loop'], level=level, step=step, n=loop.get('n', '?'), p=loop.get('p', '?'),
                         count=info.get('actions_on_level', '?'))]
        else:
            out = [_fill(TEXTS['takeover_soft'], level=level, step=step, tokens=format(int(info.get('tokens') or 0), ','),
                         n=info.get('actions_on_level', '?'))]
        d = getattr(session, '__dict__', {}) or {}
        facts = []
        ends = list((d.get('v62_level_attempt_ends') or {}).get(level, []))
        how = [('%d ended in GAME OVER' % ends.count('game_over')) if ends.count('game_over') else '',
               ('%d ended with your RESET' % ends.count('reset')) if ends.count('reset') else '',
               ('%d ended at a harness restart' % ends.count('restart')) if ends.count('restart') else '']
        how = [h for h in how if h]
        if ends and how:
            facts.append('Before the current attempt, this level had %d attempt%s: %s.' % (
                len(ends), '' if len(ends) == 1 else 's', ', '.join(how[:-1]) + (' and ' if len(how) > 1 else '') + how[-1]))
        returns = (d.get('v62_level_start_returns') or {}).get(level, 0)
        if returns:
            facts.append("The whole board returned to the level's starting board without a RESET %d time%s." % (
                returns, '' if returns == 1 else 's'))
        k = sum(1 for n in self._v6_library_order if n in self._kept_functions and self._v61_library_levels.get(n) == level)
        if k:
            facts.append('%d program%s written on this level %s still loaded.' % (k, '' if k == 1 else 's',
                                                                                   'is' if k == 1 else 'are'))
        if facts:
            out.append(' '.join(facts))
        return out

    def _v67_history_line(self, level, history_entries):
        """v6.7 SOFT: where this level's entries begin in the sandbox's `history` after a soft restart (which hides nothing):
        all kept, or from the RESET of an earlier hard restart on this level, whose hidden range stays hidden."""
        if (getattr(self, '_v67_takeover', None) or {}).get('kind') == 'inlevel':
            return ''                                    # v6.9 INLEVEL: its one harness line, no history line
        if (getattr(self, '_v67_takeover', None) or {}).get('kind') == 'refresh':
            return ''                                    # v6.7 REFRESH: a level start has no entries of its own yet
        if self._v61_hidden_level == level and _lazy_enabled() and self._v61_hidden:
            hi = self._v61_hidden[-1][1]
            entries = list(history_entries or [])
            at = getattr(getattr(entries[hi], 'frame', None), 'step', None) if hi < len(entries) else None
            return _fill(TEXTS['takeover_soft_history_cut'], what='this level' if int(level) > 1 else 'the game',
                         reset_step='?' if at is None else at)
        return TEXTS['takeover_soft_history']

    # ----------------------------------------------------------------------------------------------- v6.7 LOOP
    def _v67_loop_line(self, full, text, kind, summary_turn, level, step_env, current_frame, history_entries):
        """v6.7 LOOP: the ladder on C1's signatures, in place of _v65_loop_line (whose line is rung 0). One detector per
        conversation (a first or takeover opener resets it). Each acting turn's report adds its signature (executed actions,
        normalized report) and its acting call's code; a cycle report is a smallest period p <= LOOP_PMAX whose last k*p
        signatures are k >= 3 identical blocks. Consecutive cycle reports count up; a non-cycle report ends the episode
        (count 0); resumptions do neither. C1's line at every cycle report but the block's; at the LOOP_RUNG1-th (once per
        episode) the loop-rule block opens the report and the cycle's calls are refused from then on (_v67_loop_refusal);
        at the LOOP_RUNG2-th, or after the 2nd refused repeat, this turn becomes the notes turn of a loop restart
        (_v67_loop_trigger). Returns (text, summary_turn)."""
        mode = variants.loop_sig()
        st = self.__dict__.get('_v67_loop')
        if st is None or kind in ('first', 'takeover'):
            st = self._v67_loop = dict(sigs=[], codes=[], mark=self._v61_level_tokens, consecutive=0, block=None, cycle=None,
                                       refused=0)
            self.__dict__.pop('_v67_last_code', None)
        if LOOP_FRAME68 and kind in ('first', 'takeover'):   # v6.8 LOOP_FRAME: the board the next acting turn starts from
            st['frame'] = _v68_interior(current_frame)
        if kind in ('first', 'takeover', 'resume'):
            return text, summary_turn
        acts = tuple(str(a) for a in ((self._last_step_summary or {}).get('executed_actions') or ()))
        st['sigs'].append((acts, _v65_report_sig(full, mode if mode != 'off' else 'norm')))
        st['codes'].append(_v67_norm_code(self.__dict__.pop('_v67_last_code', None)))
        if LOOP_FRAME68:                                # v6.8 LOOP_FRAME: each acting turn's starting interior, beside its signature
            st.setdefault('starts', []).append(st.get('frame'))
            st['frame'] = _v68_interior(current_frame)
            st.setdefault('gos', []).append(1 if (self._last_step_summary or {}).get('game_over') else 0)   # v6.8b: GAME OVERs
        level_tokens = int(self._v61_level_tokens or 0)
        toks = level_tokens - st['mark'] if level_tokens >= st['mark'] else level_tokens
        st['mark'] = level_tokens
        del st['sigs'][:-512], st['codes'][:-512]
        if LOOP_FRAME68:
            del st['starts'][:-512], st['gos'][:-512]
        p, k = _v65_loop(st['sigs'], LOOP_PMAX)
        if k < 3 and not (st['refused'] >= 2 and st['cycle']):
            st.update(consecutive=0, block=None, cycle=None, refused=0)     # a non-cycle acting turn ends the episode
            return text, summary_turn
        stats = self._v6_stats
        if k >= 3:
            st['consecutive'] += 1
            for key, add in (('loop_reports', 1), ('loop_cycle_actions', len(acts)), ('loop_cycle_tokens', toks),
                             ('loop_episodes', 1 if st['consecutive'] == 1 else 0)):
                stats[key] = stats.get(key, 0) + add
            stats['loop_max_consecutive'] = max(stats.get('loop_max_consecutive', 0), st['consecutive'])
            a = sum(len(sig[0]) for sig in st['sigs'][-p:])
            cyc = dict(p=p, k=k, n=k * p, a=a, A=a * k, codes=list(st['codes'][-p:]))
        else:
            cyc = dict(st['cycle'])
        if k >= 3 and st['consecutive'] == LOOP_RUNG1 and st['block'] is None and st['refused'] < 2:
            block = TEXTS['loop_block']
            if not RESTART_TOKENS:
                block = block.replace(' This conversation has used {t} of the {T} tokens it can use on this level.', '')
            st['block'] = _fill(block, n=cyc['n'], p=cyc['p'], k=cyc['k'], a=cyc['a'], A=cyc['A'], level=level,   # v6.8: {level}
                                t=format(level_tokens, ','), T=format(self._v67x_budget(level) if EXT200 else RESTART_TOKENS, ','))
            if EXT200:                                   # v6.7x EXT200: the block's fields, to restate {T} if this attempt is extended
                st['block_fill'] = (block, dict(n=cyc['n'], p=cyc['p'], k=cyc['k'], a=cyc['a'], A=cyc['A'], level=level,
                                                t=format(level_tokens, ',')))
            st['cycle'] = cyc
            stats['loop_rung1'] = stats.get('loop_rung1', 0) + 1
            self._v67_loop_log(1, cyc, st, level)
            return st['block'] + '\n' + text, summary_turn
        if k >= 3:                                       # C1's line (rung 0, and after the block)
            line = (_fill(TEXTS['v65_loop_1'], k=k) if p == 1 else _fill(TEXTS['v65_loop_p'], n=k * p, p=p, k=k))
            stats['loop_lines'] = stats.get('loop_lines', 0) + 1
            stats['loop_max_k'] = max(stats.get('loop_max_k', 0), k)
            stats['loop_games'] = 1
            if mode != 'off':
                stats['loop_sig'] = mode
            self._v6_log('v65-loop-line', dict(p=p, k=k, mode=mode))
            lines = text.split('\n')
            at = next((i for i, l in enumerate(lines) if l.startswith('Step ')), len(lines))
            text = '\n'.join(lines[:at] + [line] + lines[at:])
        if st['refused'] >= 2 or st['consecutive'] >= LOOP_RUNG2:
            if not summary_turn:
                req = self._v67_loop_trigger(level, step_env, current_frame, history_entries, cyc,   # v6.8b: the fallback
                                             fallback=bool(LOOP_FALLBACK68 and k >= 3 and st['consecutive'] >= LOOP_FALLBACK68
                                                           and not V70H_ACTREV))   # v7.0 item 17: ACT-REV replaces it
                if req:                                  # rung 2 / 3: this turn is the notes turn
                    return text.replace(TEXTS['reminder'], '').rstrip() + '\n' + req, True
            return text, summary_turn
        self._v67_loop_log(1 if st['block'] else 0, cyc, st, level)
        return text, summary_turn

    def _v67_loop_log(self, rung, cyc, st, level, **extra):
        """v6.7 LOOP: the v67-loop event (rung, p, k, consecutive, calls, cycle actions, tokens on the level, logical
        tokens, level, attempt)."""
        game = self._v61_session()
        receipt = getattr(getattr(self, '_v6_session', None), 'last_receipt', None) or {}
        self._v6_log('v67-loop', dict(rung=rung, p=cyc.get('p'), k=cyc.get('k'), consecutive=st.get('consecutive'),
                                      calls=cyc.get('n'), cycle_actions=cyc.get('a'), tokens_on_level=self._v61_level_tokens,
                                      logical_tokens=receipt.get('logical_tokens'), level=level,
                                      attempt=len((getattr(game, 'v6_level_attempts', None) or {}).get(level, [])) + 1,
                                      refused=st.get('refused'), **extra))

    def _v67_loop_refusal(self, args):
        """v6.7 LOOP rung 1: the refusal text when this call's code repeats one of the detected cycle's calls (trailing
        whitespace aside) while the episode's block stands, else None. Counted (loop_refused, v67-loop-refused); the 2nd
        refusal brings rung 2."""
        st = self.__dict__.get('_v67_loop') or {}
        cyc = st.get('cycle')
        if not cyc or not st.get('block') or self._v62_summary_state is not None:
            return None
        code = _v67_norm_code((args or {}).get('code'))
        idx = next((i for i, c in enumerate(cyc.get('codes') or (), 1) if c and c == code), None)
        if idx is None:
            return None
        if LOOP_FRAME68 and not self._v68_frame_holds(st, cyc)[0]:   # v6.8b: refused only while the frame condition holds
            self._v6_stats['loop_refusal_frame_blocked'] = self._v6_stats.get('loop_refusal_frame_blocked', 0) + 1
            return None                                  # the identical call executes; the episode goes on
        import hashlib
        st['refused'] += 1
        self._v6_stats['loop_refused'] = self._v6_stats.get('loop_refused', 0) + 1
        self._v6_log('v67-loop-refused', dict(code_hash=hashlib.sha1(code.encode()).hexdigest()[:12], p=cyc.get('p'),
                                              call=idx, refused=st['refused']))
        return _fill(TEXTS['loop_refused'], k=idx) + '\n' + st['block']

    def _v67_loop_trigger(self, level, step_env, current_frame, history_entries, cyc=None, fallback=False):
        """v6.7 LOOP rung 2 / 3: decide the loop restart and start its notes turn; returns the notes request (None without
        notes, or when a restart is already pending). Rung 3 (hard) when this level had a loop soft restart within
        LOOP_HARD_TOKENS generated tokens, or LOOP_SOFT_CAP of them; else rung 2, a soft restart also under the hard switch
        (still hard in GAME OVER or on a dead board, as _v67_restart_kind decides). The kind is fixed here and the request
        states it; the restart follows the turn (_v62_analyze, reason 'loop')."""
        st = self.__dict__.get('_v67_loop') or {}
        cyc = dict(cyc or st.get('cycle') or {})
        if self._v62_summary_state is not None or self._v62_restart_reason == 'loop':
            return None
        fb = None
        if LOOP_FRAME68 and not self._v68_frames_repeat(st, cyc, level, record=not fallback):   # v6.8 / v6.8b LOOP_FRAME
            if not fallback:
                return None                              # no escalation past rung 1; the episode goes on
            fb = self._v68_fallback(st, cyc, level)      # v6.8b: the signature-only fallback, a hard rung 2
        game = self._v61_session()
        recs = (getattr(game, '__dict__', {}).get('v67_level_restarts') or {}).get(level, [])
        softs = [i for i, r in enumerate(recs) if r.get('reason') == 'loop'
                 and r.get('kind') == LOOP_RUNG2_KIND]   # v6.7x LOOPHARD: the earlier rung-2 kind's loop restarts
        since = (sum(int(r.get('tokens') or 0) for r in recs[softs[-1] + 1:]) + int(self._v61_level_tokens or 0)
                 if softs else None)
        if INLEVEL69 and since is not None:              # v6.9 INLEVEL: and the level's conversations that
            since += self._v69_ir_tokens_since(level, softs[-1])   # ended at an in-level refresh
        rung = 3 if (len(softs) >= LOOP_SOFT_CAP or (since is not None and since < LOOP_HARD_TOKENS)) else 2
        kind, why = self._v67_restart_kind(current_frame, history_entries,
                                           'hard' if (rung == 3 or fb is not None) else LOOP_RUNG2_KIND)   # v6.7x LOOPHARD; v6.8b
        key = ('attempt_summary_request_loop_again' if rung == 3 else
               'attempt_summary_request_loop' if kind == 'soft' else 'attempt_summary_request_loop_hard')
        self._v62_restart_reason = 'loop'
        self._v67_pending = dict(level=level, kind=kind, why=why,
                                 loop=dict(rung=rung, n=cyc.get('n'), p=cyc.get('p'), k=cyc.get('k'), since=since))
        self._v67_loop_log(rung, cyc, st, level, kind=kind, why=why, since=since,
                           **({'fallback': True} if fb is not None else {}))   # v6.8b: a fallback rung 2
        if not RESTART_NOTES:
            return None
        self._v62_summary_state = 'requested'
        self._v62_attempt_request, self._v62_attempt_saved = level, False
        self._v64_notes_retry = None
        self._step_env_callback = _NoActions(step_env, self._v67_key('summary_turn_refusal'))
        self._v6_stats['summary_turns'] = self._v6_stats.get('summary_turns', 0) + 1
        self._v6_log('v62-summary-turn', dict(level=level, tokens_on_level=self._v61_level_tokens,
                                              noact_tokens=self._v62_noact_tokens, reason='loop'))
        return _fill(TEXTS[key], n=cyc.get('n', '?'), p=cyc.get('p', '?'), level=level) + (
            self._v77_notes_prev(level) if V77 else '')         # v7.7: the last level-up summary, for the notes to update

    def _v67_park_due(self, level):
        """v6.7 LOOP cap: the LOOP_PARK_AT-th loop hard restart on a level (and any later one) parks the game: its swap-out
        queues it below every other game (_V62Snapshot.parked)."""
        pending = self.__dict__.get('_v67_pending') or {}
        if pending.get('kind') != 'hard' or not pending.get('loop'):
            return False
        game = self._v61_session()
        recs = (getattr(game, '__dict__', {}).get('v67_level_restarts') or {}).get(level, [])
        hard = sum(1 for r in recs if r.get('reason') == 'loop' and r.get('kind') == 'hard')
        if hard + 1 < LOOP_PARK_AT:
            return False
        self._v6_stats['loop_parks'] = self._v6_stats.get('loop_parks', 0) + 1
        self._v6_log('v67-loop-park', dict(level=level, loop_hard_restarts=hard + 1))
        return True

    # ----------------------------------------------------------------------------------------------- v6.7 REFRESH
    def _v67_refresh_gate(self, level):
        """v6.7 REFRESH: at the start of a turn, 'now' (refresh now: the completed level's summary is saved, or its summary
        turn has run), 'summary' (this level-up turn becomes that summary turn) or None. Checked once per level-up: the
        turn after an acting turn that completed a level in this conversation, with the session's logical length (the
        server-reported logical position count) at least REFRESH_LOGICAL."""
        if V73_NO_LEVELUP:                               # v7.3 (the user 10-22): no level-up refreshes; the strict cap alone
            return None
        st = self.__dict__.setdefault('_v67_refresh_state', {})
        if st.get('pending') == 'summary':
            if self._v62_summary_state == 'done':
                return 'now'                             # the summary turn has run
            if self._v62_summary_state is not None and self._v6_started:
                return None
            st.pop('pending', None)                      # a restart came first: no refresh (and the position checks resume)
        if (not self._v6_started or st.get('checked') == level
                or not (self._last_step_summary or {}).get('level_transition')):
            return None
        st['checked'] = level
        receipt = getattr(getattr(self, '_v6_session', None), 'last_receipt', None) or {}
        logical = int(receipt.get('logical_tokens') or 0)
        if logical < REFRESH_LOGICAL:
            return None
        if V72_RML and self._v72_late('levelup', level, logical=logical):   # v7.2b: no level-up refresh late in the run
            return None
        st.update(logical=logical, level=level, summary_turn=(int(level) - 1) not in self._v6_rules)
        if not st['summary_turn']:
            return 'now'
        st['pending'] = 'summary'
        return 'summary'

    def _v67_refresh_summary(self, text, level, step_env):
        """v6.7 REFRESH: make this level-up turn the completed level's summary turn (the SOFT notes-turn pattern: actions are
        refused; the level-up text's own request stands; the request says the conversation ends after the turn). Returns
        (text, summary_turn)."""
        st = self.__dict__.get('_v67_refresh_state') or {}
        if st.get('pending') != 'summary' or self._v62_summary_state is not None:
            return text, False
        self._v62_summary_state = 'requested'
        self._step_env_callback = _NoActions(step_env, 'refresh_turn_refusal')
        self._v6_stats['refresh_summary_turns'] = self._v6_stats.get('refresh_summary_turns', 0) + 1
        self._v6_log('v62-summary-turn', dict(level=level, tokens_on_level=self._v61_level_tokens,
                                              noact_tokens=self._v62_noact_tokens, reason='refresh'))
        req = _fill(TEXTS['refresh_summary_request'], logical=format(int(st.get('logical') or 0), ','), level=level,
                    prev=int(level) - 1)
        return text.replace(TEXTS['reminder'], '').rstrip() + '\n' + req, True

    def _v67_refresh(self, state_path, current_frame, history_entries):
        """v6.7 REFRESH: end this conversation and start a new one at the new level's first board (the current board): a new
        session and the per-conversation state as at a restart, but no RESET, no attempt, no swap-out or park, no archive,
        nothing hidden; the level's token budget starts fresh; the takeover opener states the refresh (takeover_refresh).
        Receipts: refreshes, refresh_logical, refresh_levels, refresh_summary_turns (counted at the turn); the v67-refresh
        event; the tokens to the first executed action follow (_v67_refresh_first_action)."""
        st = self.__dict__.setdefault('_v67_refresh_state', {})
        level = getattr(current_frame, 'level', None)
        start = getattr(self, '_v67_conv_level', None)
        done = (list(range(int(start), int(level))) if isinstance(start, int) and isinstance(level, int) and start < level
                else [int(level) - 1] if isinstance(level, int) and level > 1 else [])
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
                            ('_v61_error_streak', 0), ('_v62_noact_tokens', 0), ('_v62_noact_warned', 0)):
            setattr(self, name, value)                   # the per-conversation state, as at a restart
        self._v62_attempt_request, self._v62_attempt_saved = None, False
        self._v62_summary_state, self._v62_restart_reason = None, None
        self.__dict__.pop('_v67_pending', None)
        self._v61_level_tokens = 0
        if RESTART_ACROSS69 or PARK69:                         # v6.9 RAB: the level-up refresh starts the level's count afresh
            self._v69_att = 0
            self.__dict__.pop('_v69_pk', None)
        self._v61_takeover = True
        self._v67_takeover = dict(kind='refresh', level=level, logical=st.get('logical'), done=done)
        current_frame, history_entries = load_runtime_state(state_path)
        self._v61_noact_level_from = len(history_entries)
        stats = self._v6_stats
        conversation = (int(self._v61_restarts or 0) + int(stats.get('refreshes', 0))
                        + int(stats.get('inlevel_refreshes', 0)) + 1)   # v6.9 INLEVEL: its conversations counted
        stats['refreshes'] = stats.get('refreshes', 0) + 1
        stats.setdefault('refresh_logical', []).append(st.get('logical'))
        stats.setdefault('refresh_levels', []).append(level)
        self._v6_log('v67-refresh', dict(level=level, logical=st.get('logical'), conversation=conversation,
                                         tokens_prev_level=getattr(self, '_v67_prev_level_tokens', None),
                                         summary_turn=bool(st.get('summary_turn')), summary_retry=st.get('retried'),
                                         summary_saved=(int(level) - 1) in self._v6_rules if isinstance(level, int) else None,
                                         completed=done))
        if st.get('retried'):                            # the summary turn's retry and its outcome (v6.4 A1's event)
            k = int(level) - 1
            self._v6_log('v64-attempt-notes-retry', dict(level=k, reason=st['retried'], kind='refresh', outcome=(
                'saved' if k in self._v62_rule_fn else 'captured' if k in self._v6_rules else 'none')))
        self._v67_refresh_state = dict(checked=level)
        self._v67_refresh_watch = level
        return current_frame, history_entries

    def _v67_refresh_retry(self, reason):
        """v6.7 REFRESH (IE 08:12): the summary turn's one retry, v6.4 A1's pattern with its own state: the request when
        neither save_summary nor the "Rule Lk:" capture stored the completed level's summary; None after the one retry.
        Receipt refresh_summary_retries; the outcome is logged at the refresh (v67-refresh, v64-attempt-notes-retry)."""
        st = self.__dict__.get('_v67_refresh_state') or {}
        k = int(st.get('level') or 0) - 1
        if st.get('retried') or k in self._v6_rules:
            return None
        st['retried'] = reason
        self._v6_stats['refresh_summary_retries'] = self._v6_stats.get('refresh_summary_retries', 0) + 1
        return _fill(TEXTS['refresh_summary_retry'], reason=reason, level=k)

    def _v67_refresh_first_action(self):
        """v6.7 REFRESH receipt: the tokens generated in the new conversation before its first executed action
        (refresh_tokens_to_action, the v67-refresh-first-action event)."""
        level = self.__dict__.pop('_v67_refresh_watch', None)
        tokens = int(self._v62_noact_tokens or 0)
        self._v6_stats.setdefault('refresh_tokens_to_action', []).append(tokens)
        self._v6_log('v67-refresh-first-action', dict(level=level, tokens=tokens))

    # ----------------------------------------------------------------------------------------------- v6.7 NOTES
    def _v67_ledger_lines(self, history_entries):
        """v6.7 NOTES: the level-clear ledger, one line per completed level of the game (the last 10): the clearing action and
        up to 8 actions before it in its attempt (names with arguments; the attempt starts after the latest RESET, automatic
        restart or level completion). Receipt ledger_lines."""
        entries = list(history_entries or [])
        out, done = [], 0
        for i, e in enumerate(entries):
            if not (getattr(e, 'result', None) or {}).get('level_completed'):
                continue
            done += 1
            prior = []
            for j in range(i - 1, -1, -1):
                rj = getattr(entries[j], 'result', None) or {}
                a = str(getattr(entries[j], 'action', '') or '').strip()
                if not a or rj.get('level_completed') or rj.get('automatic') or a.upper() == 'RESET':
                    break
                prior.append(a)
                if len(prior) == 8:
                    break
            act = str(getattr(e, 'action', '') or '?')
            out.append(_fill(TEXTS['notes_ledger'], k=done, action=act, actions=', '.join(reversed(prior))) if prior
                       else _fill(TEXTS['notes_ledger_first'], k=done, action=act))
        out = out[-10:]
        if out:
            self._v6_stats['ledger_lines'] = self._v6_stats.get('ledger_lines', 0) + len(out)
        return out

    def _v67_coverage_lines(self, level, valid, history_entries, never_only=False):
        """v6.7 NOTES: capped coverage facts for this level (all its entries since its first board): per valid action type,
        how many distinct boards (whole board compared) it was executed from, and the valid action types never executed on
        it (at most 10). No per-click detail. Receipt coverage_lines."""
        entries = list(history_entries or [])
        first = max([i for i, e in enumerate(entries) if (getattr(e, 'result', None) or {}).get('level_completed')] or [0])
        boards = {}
        for i in range(first + 1, len(entries)):
            e = entries[i]
            if (getattr(e, 'result', None) or {}).get('automatic'):
                continue
            m = re.match(r'[A-Z0-9]+', str(getattr(e, 'action', '') or ''))
            prev = getattr(entries[i - 1], 'frame', None)
            if m and prev is not None:
                boards.setdefault(m.group(0), set()).add(tuple(tuple(row) for row in prev.grid))
        types = [a.strip() for a in str(valid or '').split(',') if a.strip() and a.strip() not in ('none', 'RESET')][:10]
        out = []
        seen = ['%s %d' % (t, len(boards[t])) for t in types if boards.get(t)]
        never = [t for t in types if not boards.get(t)]
        if seen and not never_only:                     # v6.8 NEVER_TYPES (never_only): no distinct-boards line
            out.append(_fill(TEXTS['notes_coverage'], pairs=', '.join(seen)))
        if never:
            out.append(_fill(TEXTS['notes_never'], names=', '.join(never)))
        if out:
            key = 'never_lines' if never_only else 'coverage_lines'
            self._v6_stats[key] = self._v6_stats.get(key, 0) + len(out)
        return out

    def _v67_notes_restart_lines(self, level, valid, history_entries):
        """v6.7 NOTES: in a restart opener, the level's action count (taaf's per-level counter, the official count), the
        level-clear ledger and the coverage facts; after a REFRESH (a level start) the ledger only."""
        if (getattr(self, '_v67_takeover', None) or {}).get('kind') == 'refresh':
            return self._v67_ledger_lines(history_entries)
        n = self._v67_actions_on_level(level, history_entries)
        return ([_fill(TEXTS['notes_action_count'], n=n)] + self._v67_ledger_lines(history_entries)
                + self._v67_coverage_lines(level, valid, history_entries))

    def _v67_opener_size(self, kind, text, opener):
        """v6.7 NOTES receipts: characters and text tokens of the openers, per opener kind (openers_<kind>,
        opener_chars_<kind>, opener_tokens_<kind>)."""
        stats = self._v6_stats
        tokens = self._v63_text_tokens(opener.get('content')) or 0
        for key, add in (('openers_' + kind, 1), ('opener_chars_' + kind, len(text or '')), ('opener_tokens_' + kind, tokens)):
            stats[key] = stats.get(key, 0) + add

    # ----------------------------------------------------------------------------------------------- v6.7x EXT200
    def _v67x_budget(self, level=None):
        """v6.7x EXT200: this attempt's token budget on the level: RESTART_TOKENS, or RESTART_TOKENS + EXT200_TOKENS while
        the attempt assigned to the extend arm lasts."""
        st = self.__dict__.get('_v67x_ext') or {}
        if st.get('arm') == 'extend' and (level is None or st.get('level') == level):
            return st['budget']
        return RESTART_TOKENS

    @staticmethod
    def _v67x_entries(game, history_entries=None):
        """v6.7x EXT200: the number of history entries (the game session's live list, else the given entries): one per
        executed action, automatic restarts and RESETs included (the unit of the official per-level action count)."""
        live = getattr(game, 'history_entries', None)
        entries = live if live is not None else history_entries
        return len(entries) if entries is not None else None

    def _v67x_assign(self, level, current_frame, history_entries):
        """v6.7x EXT200: the assignment point (_v62_analyze, before budget_hit, where the notes turn of a token-budget restart
        is decided; the caller excludes a session error, a position restart and the position notes turn). An attempt (this
        conversation on the level) without an assignment that has generated RESTART_TOKENS on the level, with no notes turn
        or restart pending (loop, no-action or token), gets its one assignment: 'late' when the late gate skips the restart
        (nothing changes), else 'extend' when u < EXT200 (the attempt's budget becomes RESTART_TOKENS + EXT200_TOKENS, and a
        loop-rule block standing in this conversation restates it), else 'restart' (unchanged). The attempt's index is the
        conversation's index on the level: 1 + the harness restarts recorded on the level (v67_level_restarts, any reason or
        kind). Receipts: ext200_<arm> (all five ext200 stats exist from a game's first assignment); the v67-ext200 event with
        the counters that locate this checkpoint in the log: requests (the game's model requests so far = its model-turn
        events), turns, the conversation's observation index (v6-opener obs_index), the persistent session's id and request
        count, the conversation index (as REFRESH counts it), the logical length and the step."""
        if (not RESTART_TOKENS or not self._v6_started or self.__dict__.get('_v67x_ext')
                or int(self._v61_level_tokens or 0) < RESTART_TOKENS
                or self._v62_summary_state is not None or self._v62_restart_reason is not None):
            return
        game = self._v61_session()
        gid = _v67x_game_id(game)
        attempt = len(((getattr(game, '__dict__', None) or {}).get('v67_level_restarts') or {}).get(level, [])) + 1
        u = _v67x_u(EXT200_SEED, gid, level, attempt)
        arm = 'late' if not self._v62_restart_time_ok(level) else 'extend' if u < EXT200 else 'restart'
        budget = RESTART_TOKENS + EXT200_TOKENS if arm == 'extend' else RESTART_TOKENS
        tokens = int(self._v61_level_tokens or 0)
        self._v67x_game = game
        self._v67x_ext = dict(game=gid, level=level, attempt=attempt, arm=arm, tokens=tokens, budget=budget,
                              entries=self._v67x_entries(game, history_entries))
        stats = self._v6_stats
        for key in ('ext200_extend', 'ext200_restart', 'ext200_late', 'ext200_extend_cleared'):
            stats.setdefault(key, 0)
        stats.setdefault('ext200_extend_tokens', [])
        stats['ext200_' + arm] += 1
        loop = self.__dict__.get('_v67_loop') or {}
        if arm == 'extend' and loop.get('block') and loop.get('block_fill'):   # a standing loop-rule block (repeated in
            template, kw = loop['block_fill']                                  # refusals) states the new budget
            loop['block'] = _fill(template, T=format(budget, ','), **kw)
        session = getattr(self, '_v6_session', None)
        receipt = getattr(session, 'last_receipt', None) or {}
        self._v6_log('v67-ext200', dict(
            game=gid, level=level, attempt=attempt, u=u, arm=arm, tokens_on_level=tokens, budget=budget,
            logical=receipt.get('logical_tokens'), step=getattr(current_frame, 'step', None),
            requests=stats.get('requests'), turns=stats.get('turns'), obs_index=self._v6_obs_count,
            session_id=getattr(session, 'session_id', None), session_requests=getattr(session, 'request_count', None),
            conversation=int(self._v61_restarts or 0) + int(stats.get('refreshes', 0) or 0) + 1,
            game_attempt=len(self._v61_attempts(game, level)) + 1,
            actions_on_level=self._v67_actions_on_level(level, history_entries),
            noact_tokens=int(self._v62_noact_tokens or 0), seed=EXT200_SEED, fraction=EXT200))

    def _v67x_cleared(self):
        """v6.7x EXT200 (after an acting call): a level completion in the call (the step summary's level_transition, or
        run_complete) ends the assigned attempt as cleared, with the tokens on the level at the clear (the acting call ends
        the turn, so no more tokens are generated on the level)."""
        s = self._last_step_summary or {}
        if s.get('level_transition') or s.get('run_complete'):
            self._v67x_end(True, self._v61_level_tokens, 'clear')

    def _v67x_end(self, cleared, tokens, ended, reason=None, history_entries=None):
        """v6.7x EXT200: the assigned attempt ends: 'clear' (after the acting call that completed its level), 'restart' (a
        harness restart of any reason, before its RESET; the tokens include the notes turn), 'level_change' (a clear the
        acting turn did not report) or 'run_end'. The v67-ext200-outcome event: arm, cleared, how it ended (with the
        restart's reason), the tokens on the level at the end and since the assignment, the actions executed since the
        assignment (history entries: every executed action, automatic restarts and RESETs included). In the extend arm these
        are the extension's; receipts ext200_extend_cleared and ext200_extend_tokens (the tokens since the assignment). The
        attempt's state is dropped, so the next attempt's budget is RESTART_TOKENS."""
        st = self.__dict__.pop('_v67x_ext', None)
        if not st:
            return
        n = self._v67x_entries(self._v61_session() or self.__dict__.get('_v67x_game'), history_entries)
        since = max(0, int(tokens or 0) - int(st['tokens']))
        actions = max(0, n - st['entries']) if n is not None and st.get('entries') is not None else None
        stats = self._v6_stats
        if st['arm'] == 'extend':
            stats['ext200_extend_cleared'] = stats.get('ext200_extend_cleared', 0) + (1 if cleared else 0)
            stats.setdefault('ext200_extend_tokens', []).append(since)
        self._v6_log('v67-ext200-outcome', dict(game=st['game'], level=st['level'], attempt=st['attempt'], arm=st['arm'],
                                                cleared=bool(cleared), ended=ended, reason=reason,
                                                tokens_on_level=int(tokens or 0), tokens_since=since, actions_since=actions,
                                                budget=st['budget']))

    def _v67x_run_end(self):
        """v6.7x EXT200 (run_v6, after the run): an assigned attempt still open ends 'run_end'; cleared when the game run has
        completed its level (a clear the acting turn did not report)."""
        st = self.__dict__.get('_v67x_ext')
        if not st:
            return
        run = getattr(getattr(self.__dict__.get('_v67x_game'), 'game', None), 'game_run', None)
        try:
            done = int(getattr(run, 'levels_completed', 0) or 0)
        except Exception:  # noqa: BLE001
            done = 0
        self._v67x_end(isinstance(st.get('level'), int) and done >= st['level'], self._v61_level_tokens, 'run_end')

    # ----------------------------------------------------------------------------------------------- v6.7x SUMLOOKUP
    def _v67x_summaries_payload(self):
        """v6.7x SUMLOOKUP: the saved level summaries for this python call's summaries() (sandbox_ext puts them in the
        call's initial state, as it does the library): [[level, text]] by level, the texts as stored (_v6_rules, truncation
        included); None with the switch off, so the call's state is unchanged."""
        if not SUMLOOKUP:
            return None
        return [[int(level), str(text)] for level, text in sorted(self._v6_rules.items())]

    def _v67x_summaries_call(self, request):
        """v6.7x SUMLOOKUP receipt, the host side of a summaries() call in the sandbox: summaries_calls counts every call;
        the v67-sumlookup event (level: the turn's level; n_summaries: the summaries the call returned) is logged for the
        first call in each conversation (the later ones are counted only)."""
        if not SUMLOOKUP:
            return {}
        stats = self._v6_stats
        stats['summaries_calls'] = stats.get('summaries_calls', 0) + 1
        if V77 and (request or {}).get('query'):             # v7.7 (C): a summaries(query) call
            stats['v77_summary_queries'] = stats.get('v77_summary_queries', 0) + 1
        conversation = int(self._v61_restarts or 0) + int(stats.get('refreshes', 0) or 0) + 1
        if self.__dict__.get('_v67x_sumlookup_logged') != conversation:
            self._v67x_sumlookup_logged = conversation
            try:
                n = int((request or {}).get('n'))
            except (TypeError, ValueError):
                n = None
            self._v6_log('v67-sumlookup', dict(level=self._v61_tokens_level, n_summaries=n, conversation=conversation,
                                               calls=stats['summaries_calls']))
        return {}

    # ----------------------------------------------------------------------------------------------- v6.8
    def _v68_restart_lines(self, level, valid, history_entries):
        """v6.8 LEDGER / NEVER_TYPES (NOTES off): in a restart opener the level-clear ledger (LEDGER) and the valid action
        types never executed on the level (NEVER_TYPES; v6.7's coverage computation, that line only, when non-empty); after a
        REFRESH (a level start) the ledger only."""
        refresh = (getattr(self, '_v67_takeover', None) or {}).get('kind') == 'refresh'
        out = self._v67_ledger_lines(history_entries) if LEDGER68 else []
        if NEVER68 and not refresh:
            out += self._v67_coverage_lines(level, valid, history_entries, never_only=True)
        return out

    def _v68_loop_takeover(self, level):
        """v6.8 LOOP_TEXTS: True when this conversation began at a loop-rule restart of this level."""
        info = getattr(self, '_v67_takeover', None) or {}
        return info.get('reason') == 'loop' and info.get('level') == level

    def _v68_frame_holds(self, st, cyc):
        """v6.8b LOOP_FRAME: (holds, p, k): the corrected frame condition now (_v68_ends_ok on the starting interiors of the
        conversation's acting turns and the current interior) for the detector's current cycle, else the episode's stored one."""
        p, k = _v65_loop(st.get('sigs') or [], LOOP_PMAX)
        if k < 3:
            p, k = int((cyc or {}).get('p') or 0), int((cyc or {}).get('k') or 0)
        return _v68_ends_ok(st.get('starts') or [], st.get('frame'), p, k), p, k

    def _v68_frames_repeat(self, st, cyc, level, record=True):
        """v6.8 LOOP_FRAME, v6.8b's corrected condition: True when the interiors at the end of each of the last k repetitions of
        the p-call cycle are identical, or repeat with a period of 2 repetitions (k >= 4). Otherwise the loop restart is
        blocked: with record, the receipt loop_frame_blocked and the v68-loop-frame-blocked event (p, k, level, attempt,
        consecutive, refused), once per new acting turn; the episode goes on (rungs 0 and 1, the fallback count)."""
        ok, p, k = self._v68_frame_holds(st, cyc)
        if ok or not record:
            return ok
        sigs = st.get('sigs') or []
        if st.get('frame_blocked_at') != len(sigs):
            st['frame_blocked_at'] = len(sigs)
            self._v6_stats['loop_frame_blocked'] = self._v6_stats.get('loop_frame_blocked', 0) + 1
            game = self._v61_session()
            self._v6_log('v68-loop-frame-blocked', dict(
                p=p, k=k, level=level, attempt=len((getattr(game, 'v6_level_attempts', None) or {}).get(level, [])) + 1,
                consecutive=st.get('consecutive'), refused=st.get('refused')))
        return False

    def _v68_fallback(self, st, cyc, level):
        """v6.8b LOOP_FALLBACK: LOOP_FALLBACK68 consecutive cycle reports bring a hard rung 2 without the frame check: the
        receipt loop_fallback and the v68-loop-fallback event (p, k, consecutive, level, attempt, the GAME OVERs in the last
        LOOP_FALLBACK68 acting reports)."""
        p, k = _v65_loop(st.get('sigs') or [], LOOP_PMAX)
        if k < 3:
            p, k = int((cyc or {}).get('p') or 0), int((cyc or {}).get('k') or 0)
        game = self._v61_session()
        gos = sum((st.get('gos') or [])[-LOOP_FALLBACK68:])
        self._v6_stats['loop_fallback'] = self._v6_stats.get('loop_fallback', 0) + 1
        self._v6_log('v68-loop-fallback', dict(
            p=p, k=k, consecutive=st.get('consecutive'), level=level,
            attempt=len((getattr(game, 'v6_level_attempts', None) or {}).get(level, [])) + 1, gameovers=gos))
        return dict(p=p, k=k, gameovers=gos)

    # ----------------------------------------------------------------------------------------------- v6.8 REFRESH_PROGRAMS
    def _v68_count_program_refs(self, code):
        """v6.8 REFRESH_PROGRAMS: one executed python call. A program its code names (a word-boundary match) more often than
        its own def lines there is called: its count grows and its last call becomes (this turn, this call's sequence number);
        a program with a def line here gets its last writing so. Over the whole game."""
        st = self.__dict__.setdefault('_v68_prog', dict(counts={}, last_call={}, last_def={}, seq=0))
        st['seq'] += 1
        code = str(code or '')
        mark = (self._v6_stats.get('turns', 0), st['seq'])
        names = set(self._kept_functions) | set(re.findall(r'(?m)^[ \t]*def[ \t]+([A-Za-z_]\w*)[ \t]*\(', code))
        for n in names:
            uses = len(re.findall(r'\b%s\b' % re.escape(n), code))
            if not uses:
                continue
            defs = len(re.findall(r'(?m)^[ \t]*def[ \t]+%s[ \t]*\(' % re.escape(n), code))
            if uses > defs:
                st['counts'][n] = st['counts'].get(n, 0) + 1
                st['last_call'][n] = mark
            if defs:
                st['last_def'][n] = mark

    def _v68_refresh_program_lines(self, level):
        """v6.8 REFRESH_PROGRAMS: the refresh opener's selection (_v68_select_programs on the library): a header, one line per
        program in program_list's format (signature, first docstring line), the pointer; nothing for an empty library or
        selection. Receipt refresh_program_lists; the v68-refresh-programs event (library size, the numbers shown, their call
        counts)."""
        names = [n for n in self._v6_library_order if n in self._kept_functions]
        st = self.__dict__.get('_v68_prog') or dict(counts={}, last_call={}, last_def={})
        most, rest = _v68_select_programs(names, st['counts'], st['last_call'], st['last_def'], PROG_FREQ68, PROG_RECENT68)
        if not most + rest:
            return []
        lines = [_fill(TEXTS['refresh_program_header_most' if most else 'refresh_program_header'], k=len(most + rest),
                       n=len(names))]
        for n in most + rest:
            src = self._kept_functions[n]
            try:
                sig = function_signature(src)
            except Exception:  # noqa: BLE001
                sig = n + '(...)'
            doc = _docline(src)
            lines.append('- %s: %s' % (sig, doc) if doc else '- %s' % sig)
        lines.append(_fill(TEXTS['refresh_program_pointer'], n=len(names)))
        self._v6_stats['refresh_program_lists'] = self._v6_stats.get('refresh_program_lists', 0) + 1
        self._v6_log('v68-refresh-programs', dict(level=int(level), library=len(names), most=len(most), recent=len(rest),
                                                  counts=[st['counts'].get(n, 0) for n in most + rest]))
        return ['\n'.join(lines)]

    # ----------------------------------------------------------------------------------------------- v6.9 CLEAR_STEP
    def _v69_clear_step_parts(self, current_frame):
        """v6.9 CLEAR_STEP (decisions 82 and 87): the clearing-step block of a new conversation's opener at level L >= 2, from
        the game session's record of level L - 1's clear (v62.solver, stored at the clear): the line (its own text part,
        on its own line: the template joins text parts), the caption and image of the board the clearing action was executed
        from, the caption and image of the first board it produced (worded "the first changed board" when that is not the
        action's frame 0); without such a board, the fallback line and the board before only. Receipts clear_step_shown,
        clear_step_fallback, clear_step_missing (a takeover at L >= 2 without a record) and clear_step_index ({frame index:
        openers} of the after-boards shown); the v69-clear-step event."""
        level = int(getattr(current_frame, 'level', 0) or 0)
        if level < 2:
            return []
        from inference.agent.runtime_state import Frame
        game = self._v61_session()
        rec = ((getattr(game, '__dict__', None) or {}).get('v69_clears') or {}).get(level - 1)
        info = getattr(self, '_v67_takeover', None) or {}
        how = 'refresh' if info.get('kind') == 'refresh' else str(info.get('reason') or 'restart')
        if not rec:
            self._v6_stats['clear_step_missing'] = self._v6_stats.get('clear_step_missing', 0) + 1
            self._v6_log('v69-clear-step', dict(level=level, takeover=how, cleared=level - 1, missing=True))
            return []
        k, s, act = rec['level'], int(rec['step']), str(rec['action'])
        fill = dict(k=k, s=s, s_before=rec.get('before_step', s - 1), action=act)   # the stored step of the before board (IE)
        before = current_grid_image_part(Frame(grid=rec['before'], step=s - 1, level=k))
        if rec.get('after') is not None:
            changed = rec.get('index') != 0
            parts = [dict(type='text', text='\n' + _fill(TEXTS['clearstep_line_changed' if changed else 'clearstep_line'],
                                                           **fill) + '\n'),
                     dict(type='text', text=_fill(TEXTS['clearstep_before'], **fill)), before,
                     dict(type='text', text=_fill(TEXTS['clearstep_after_changed' if changed else 'clearstep_after'], **fill)),
                     current_grid_image_part(Frame(grid=rec['after'], step=s, level=k))]
            self._v6_stats['clear_step_shown'] = self._v6_stats.get('clear_step_shown', 0) + 1
            index = self._v6_stats.setdefault('clear_step_index', {})
            index[str(rec.get('index'))] = index.get(str(rec.get('index')), 0) + 1
        else:
            parts = [dict(type='text', text='\n' + _fill(TEXTS['clearstep_fallback'], **fill) + '\n'),
                     dict(type='text', text=_fill(TEXTS['clearstep_before'], **fill)), before]
            self._v6_stats['clear_step_shown'] = self._v6_stats.get('clear_step_shown', 0) + 1
            self._v6_stats['clear_step_fallback'] = self._v6_stats.get('clear_step_fallback', 0) + 1
        self._v6_log('v69-clear-step', dict(level=level, takeover=how, cleared=k, step=s, action=act, index=rec.get('index'),
                                            frames=rec.get('frames'), fallback=rec.get('after') is None,
                                            step_now=getattr(current_frame, 'step', None)))
        return parts

    # ----------------------------------------------------------------------------------------------- v6.9 INLEVEL
    def _v69_ir_conv(self):
        """v6.9 INLEVEL: the game's conversations begun after its first (harness restarts, level-up refreshes, in-level
        refreshes); it identifies the conversation a pending in-level refresh belongs to."""
        return (int(self._v61_restarts or 0) + int(self._v6_stats.get('refreshes', 0))
                + int(self._v6_stats.get('inlevel_refreshes', 0)))

    def _v69_ir_notes(self, text, level, step_env, kind, current_frame):
        """v6.9 INLEVEL, the trigger, at an opener of the conversation after the loop ladder and the level-up refresh made this
        turn's decisions: when the conversation's logical position (the server receipt) has reached INLEVEL69 at a mid-level
        opener (kind 'same', 'restart' after an automatic RESET, or 'resume'; never the level-up turn or a conversation's
        first turn) and no harness restart, notes turn or level-up refresh is pending, this turn becomes the in-level notes
        turn: actions are refused (inlevel_turn_refusal), the request is appended as a restart's notes request is
        (inlevel_notes_request), save_summary stores the level's attempt notes; the refresh follows at the next turn start
        (_v69_ir_gate). The game reporting GAME OVER (only RESET executes) defers it (inlevel_deferred_game_over, per turn).
        With ARC3_V62_RESTART_NOTES=0 the refresh follows at the next turn start without a notes turn. Returns (text,
        summary_turn)."""
        st = self.__dict__.setdefault('_v69_ir', {})
        if st.get('pending') or kind not in ('same', 'restart', 'resume') or not self._v6_started:
            return text, False
        if (self._v62_summary_state is not None or self._v62_restart_reason is not None
                or self.__dict__.get('_v67_pending') or getattr(self, '_v62_force_restart', False)
                or (self.__dict__.get('_v67_refresh_state') or {}).get('pending')):
            return text, False
        receipt = getattr(getattr(self, '_v6_session', None), 'last_receipt', None) or {}
        logical = int(receipt.get('logical_tokens') or 0)
        park = self._v69_pk_due(level) if PARK69 else 0       # v6.9 RAB PARK: the rung (never-started games waiting)
        v70 = (self._v70_park_due(level) if (V70_PARKS and not park and not _v62_game_over_now(self._v61_session()))
               else None)                               # v7.0: the cap bind / the index swap (not decided in GAME OVER)
        v71 = self._v71_due() if V71 else None             # v7.1 ALLOCATION: the level's tokens in this conversation
        strict = r74 = r76 = None
        if V73:                                            # v7.3: the tokens trigger parks only; the strict cap
            if v71 is not None and not self._v73_fire(v71, level):
                v71 = None                                 # nobody else waits, or the run's end: the conversation continues
            if V74:                                        # v7.4: the rebase point replaces the strict cap (not when this
                if v71 is None and logical >= V74_AT and not self.__dict__.get('_v74_rebase_next'):   # opener rebases anyway)
                    if V76:                                # v7.6 HYBRID: the summary turn, then the rebase; or the in-level refresh
                        d76 = self._v76_point(level, logical, kind, current_frame)
                        if d76 == 'summary':
                            return self._v76_summary(text, level, logical, step_env, current_frame)
                        if d76 == 'refresh':
                            r76 = self._v76_refresh_numbers(level, logical)
                    else:
                        r74 = self._v74_point(level, logical)  # 'inplace': this opener's append rebases; 'park': below
                        if r74 == 'inplace':
                            return text, False
            elif v71 is None and logical >= V73_STRICT:
                strict = self._v73_strict(level, logical)  # None in the run's last ARC3_V71_GUARD_S seconds
        if (V71 or logical < INLEVEL69) and not park and not v70 and not v71 and not strict and not r74 and not r76:   # v7.1: no logical trigger
            return text, False
        if V72_RML and self._v72_late('inlevel', level, trigger='tokens' if v71 else 'strict' if strict else 'v76_refresh' if r76
                                      else 'park' if park else (
                v70.get('trigger') if v70 else 'logical'), tokens_on_level=int(self._v61_level_tokens or 0)):
            return text, False                         # v7.2b: no in-level refresh late in the run
        stats = self._v6_stats
        if _v62_game_over_now(self._v61_session()):
            stats['inlevel_deferred_game_over'] = stats.get('inlevel_deferred_game_over', 0) + 1
            return text, False
        step = getattr(current_frame, 'step', None)
        tokens = int(self._v61_level_tokens or 0)
        noted73 = self.__dict__.get('_v73_noted') if (V73 and v71) else None   # v7.3: notes from a kept notes turn
        notes69 = ((not V74 or V76) and RESTART_NOTES          # v7.4: no notes turn before a park (the user 10-22); v7.6: as v7.1
                   and not (noted73 is not None and tuple(noted73[:2]) == (self._v69_ir_conv(), level)))
        st.update(pending='notes' if notes69 else 'now', level=level, step=step, logical=logical, tokens=tokens,
                  conv=self._v69_ir_conv(), notes_before=len(self._v62_attempt_notes.get(level) or []), kind=kind,
                  **(dict(park=True, waiting=park) if park else {}))
        if v70:                                                 # v7.0: a park by the cap bind or the index swap
            st.update(park=True, waiting=v70['waiting'], trigger=v70['trigger'], v70=v70)
        if v71:                                                 # v7.1 ALLOCATION: the notes turn, then a park while a game waits
            st.update(park=True, waiting=v71['waiting'], trigger='tokens', v71=v71)
            stats['v71_fires'] = stats.get('v71_fires', 0) + 1
        if RESTART_NOTES and not notes69 and noted73 is not None:   # v7.3: this park reuses the kept notes: their step
            st['v73_notes_step'] = noted73[2]
        if strict:                                              # v7.3: the strict cap (park while another game waits, else in place)
            st.update(park=True, waiting=strict['waiting'], trigger='strict', v73=strict)
            stats['v73_strict_fires'] = stats.get('v73_strict_fires', 0) + 1
        if r74:                                                 # v7.4: the rebase point above the reset bound: the park
            st.update(park=True, waiting=self._v71_waiting(), trigger='rebase_park', v74=dict(logical=logical, tokens=tokens))
            stats['v74_park_fires'] = stats.get('v74_park_fires', 0) + 1
        if r76:                                                 # v7.6 HYBRID: the rebase point at or above ARC3_V76_REFRESH_AT:
            st.update(park=True, waiting=r76['waiting'], trigger='v76_refresh', v76=r76)   # the in-level refresh (park or in place)
            stats['v76_refresh_fires'] = stats.get('v76_refresh_fires', 0) + 1
        if park:                                                # v6.9 RAB PARK: the rung's notes turn
            stats['park_rung_fires'] = stats.get('park_rung_fires', 0) + 1
        self._v6_log('v69-inlevel-notes-turn', dict(level=level, step=step, logical=logical, tokens_on_level=tokens, kind=kind,
                                                    notes=bool(notes69),
                                                    **(dict(trigger='park' if park else 'logical', waiting=park,
                                                            attempt_tokens=self._v69_att_tokens()) if PARK69 else {}),
                                                    **({'v70': v70} if v70 else {}), **({'v71': v71} if v71 else {}),
                                                    **({'v73': strict} if strict else {}),
                                                    **({'v74': st.get('v74')} if r74 else {}), **({'v76': r76} if r76 else {})))
        if not notes69:
            return text, False
        self._v62_summary_state = 'requested'
        self._v62_attempt_request, self._v62_attempt_saved = level, False
        self._v64_notes_retry = None
        self._step_env_callback = _NoActions(step_env, 'inlevel_turn_refusal')
        stats['inlevel_notes_turns'] = stats.get('inlevel_notes_turns', 0) + 1
        req = _fill(TEXTS['inlevel_notes_request'], level=level)
        if V77:                                               # v7.7 (the user 10-24): the last level-up summary, to update
            req += self._v77_notes_prev(level)
        return text.replace(TEXTS['reminder'], '').rstrip() + '\n' + req, True

    def _v69_ir_gate(self, level):
        """v6.9 INLEVEL, at a turn start (after the level-up refresh's gate, before the position checks and the restart
        decisions, which it pre-empts): 'now' when an in-level refresh is pending for this conversation and level: its notes
        turn has ended (as planned, or cut short by a session error or a stop: the refresh keeps the board), or, without
        notes, the trigger came at the previous opener; None otherwise. A pending refresh whose conversation or level changed
        since its trigger (without notes, a level-up in the trigger turn) is dropped (inlevel_dropped, v69-inlevel-dropped)."""
        st = self.__dict__.get('_v69_ir') or {}
        if not st.get('pending'):
            return None
        if self._v6_started and st.get('level') == level and st.get('conv') == self._v69_ir_conv():
            return 'now'
        self._v6_stats['inlevel_dropped'] = self._v6_stats.get('inlevel_dropped', 0) + 1
        self._v6_log('v69-inlevel-dropped', dict(level=st.get('level'), level_now=level, step=st.get('step'),
                                                 pending=st.get('pending'), conversation=st.get('conv') == self._v69_ir_conv()))
        self._v69_ir = {}
        return None

    def _v69_ir_refresh(self, state_path, current_frame, history_entries):
        """v6.9 INLEVEL: end this conversation and start a new one at the current board, on the level-up refresh's path
        (_v67_refresh: a new session and the per-conversation state as at a restart; no RESET, no attempt ended, no swap-out
        or park, no archive, nothing hidden; the conversation's token count on the level starts again, as at every new
        conversation). The takeover opener states it (takeover_inlevel). Receipts: inlevel_refreshes and its lists, the
        v69-inlevel-refresh event, the game session's record v69_inlevel_refreshes[level] (step, logical at the trigger and
        now, the conversation's tokens on the level, the restart records before it, the notes outcome); the tokens to the
        first executed action follow (_v69_ir_first_action)."""
        st = self.__dict__.get('_v69_ir') or {}
        if V76:                                          # v7.6 HYBRID: the conversation ends: no pending summary
            self.__dict__.pop('_v76', None)
        level = getattr(current_frame, 'level', None)
        step = getattr(current_frame, 'step', None)
        game = self._v61_session()
        stats = self._v6_stats
        receipt = getattr(getattr(self, '_v6_session', None), 'last_receipt', None) or {}
        tokens = int(self._v61_level_tokens or 0)
        after_error = bool(getattr(self, '_v62_force_restart', False))     # a session error cut the notes turn short
        n_notes = len(self._v62_attempt_notes.get(level) or [])
        outcome = ('off' if st.get('pending') == 'now' else 'saved' if getattr(self, '_v62_attempt_saved', False)
                   else 'captured' if n_notes > int(st.get('notes_before') or 0) else 'none')
        d = getattr(game, '__dict__', None)
        d = d if isinstance(d, dict) else {}
        recs = (d.get('v67_level_restarts') or {}).get(level, [])
        mine = d.setdefault('v69_inlevel_refreshes', {}).setdefault(level, [])
        mine.append(dict(step=step, logical=st.get('logical'), logical_refresh=int(receipt.get('logical_tokens') or 0),
                         tokens=tokens, restarts_before=len(recs), notes=outcome))
        level_tokens = sum(int(r.get('tokens') or 0) for r in recs) + sum(int(r.get('tokens') or 0) for r in mine)
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
                            ('_v61_error_streak', 0), ('_v62_noact_tokens', 0), ('_v62_noact_warned', 0)):
            setattr(self, name, value)                   # the per-conversation state, as at a restart and a refresh
        self._v62_attempt_request, self._v62_attempt_saved = None, False
        self._v62_summary_state, self._v62_restart_reason = None, None
        self.__dict__.pop('_v67_pending', None)
        self._v62_force_restart = False                  # a session error in the notes turn: this new session replaces it
        self._v62_notes_asked = level if outcome != 'off' else None   # the "no notes were saved" line when none exist
        self._v61_level_tokens = 0
        self._v61_takeover = True
        self._v67_takeover = dict(kind='inlevel', reason='inlevel', level=level, step=step, n=len(mine),
                                  **({'v73_notes_step': st['v73_notes_step']} if st.get('v73_notes_step') is not None else {}))
        current_frame, history_entries = load_runtime_state(state_path)
        self._v61_noact_level_from = len(history_entries)
        conversation = self._v69_ir_conv() + 1           # the 1-based index of the conversation that ended (as v67-refresh's)
        stats['inlevel_refreshes'] = stats.get('inlevel_refreshes', 0) + 1
        for key, value in (('inlevel_refresh_logical', st.get('logical')), ('inlevel_refresh_levels', level),
                           ('inlevel_refresh_level_tokens', level_tokens)):
            stats.setdefault(key, []).append(value)
        attempt = len(self._v61_attempts(game, level)) + 1 if isinstance(level, int) else None
        self._v6_log('v69-inlevel-refresh', dict(
            game=_v67x_game_id(game), level=level, step=step, attempt=attempt, logical=st.get('logical'),
            logical_refresh=int(receipt.get('logical_tokens') or 0), threshold=INLEVEL69, tokens_on_level=st.get('tokens'),
            notes_tokens=tokens - int(st.get('tokens') or 0), level_tokens=level_tokens, refreshes_on_level=len(mine),
            refreshes=stats['inlevel_refreshes'], conversation=conversation,
            actions_on_level=self._v67_actions_on_level(level, history_entries), notes=outcome, after_error=after_error,
            **(self._v70_ir_fields(level, st) if (PARK69 or V70_ON) else {})))
        self._v69_ir = {}
        self._v69_ir_watch = dict(level=level, conv=self._v69_ir_conv())
        return current_frame, history_entries

    def _v69_ir_first_action(self):
        """v6.9 INLEVEL receipt: the tokens generated in the new conversation before its first executed action
        (inlevel_refresh_tokens_to_action, the v69-inlevel-refresh-first-action event); None (superseded) when another
        conversation began before that action."""
        w = self.__dict__.pop('_v69_ir_watch', None) or {}
        same = w.get('conv') == self._v69_ir_conv()
        tokens = int(self._v62_noact_tokens or 0) if same else None
        if same:
            self._v6_stats.setdefault('inlevel_refresh_tokens_to_action', []).append(tokens)
        self._v6_log('v69-inlevel-refresh-first-action', dict(level=w.get('level'), tokens=tokens, superseded=not same))

    def _v69_ir_tokens_since(self, level, idx):
        """v6.9 INLEVEL: the generated tokens of this level's conversations that ended at an in-level refresh after the level's
        restart record idx, so the loop ladder's tokens since its last loop restart (rung 3) cover every conversation."""
        d = getattr(self._v61_session(), '__dict__', None)
        recs = ((d if isinstance(d, dict) else {}).get('v69_inlevel_refreshes') or {}).get(level, [])
        return sum(int(r.get('tokens') or 0) for r in recs if int(r.get('restarts_before') or 0) > idx)

    # ----------------------------------------------------------------------------------------------- v6.9 ACT-REV shadow
    def _v69_as_report(self, kind, level, current_frame, history_entries):
        """v6.9 ACT-REV shadow (log only): the game's new history entries go to the level novelty record (_V69Novelty), then
        this opener to the conversation's detector (_V69ActRev; a conversation's first opener resets it; resumptions are not
        acting reports and do not reach here). Logs each check from ACTREV_LOG_FROM69 consecutive reports to the would-fire
        and an episode's end; never changes the opener, the turn or any state the harness acts on. A failure is counted
        (actrev_shadow_errors) and its first one logged (v69-actrev-shadow-error)."""
        stats = self._v6_stats
        self._v70h_actrev = None                      # v7.0 item 17: this report's would-fire check (None: none)
        try:
            nov = self.__dict__.get('_v69_nov')
            if nov is None:
                nov = self._v69_nov = _V69Novelty()
            entries = list(history_entries or [])
            for e in entries[len(nov):]:
                fr = getattr(e, 'frame', None)
                nov.feed(getattr(fr, 'level', None), _v69_interior_key(getattr(fr, 'grid', None) or ()),
                         bool((getattr(e, 'result', None) or {}).get('automatic')))
            det = self.__dict__.get('_v69_as')
            if det is None:
                det = self._v69_as = _V69ActRev(ACTREV_K69, ACTREV_PMAX69, ACTREV_REPS69, ACTREV_LOG_FROM69)
            if kind in ('first', 'takeover'):
                out = det.reset(len(entries))
            else:
                acts = tuple(str(a) for a in ((self._last_step_summary or {}).get('executed_actions') or ()))
                out = det.report(acts, len(entries), nov)
            stats['actrev_shadow_max_consecutive'] = max(stats.get('actrev_shadow_max_consecutive', 0), det.peak)
            if not out:
                return
            game = self._v61_session()
            receipt = getattr(getattr(self, '_v6_session', None), 'last_receipt', None) or {}
            base = dict(game=_v67x_game_id(game), level=level, step=getattr(current_frame, 'step', None),
                        logical=receipt.get('logical_tokens'))
            if out.get('ended'):
                self._v6_log('v69-actrev-shadow-end', dict(base, peak=out['ended']['peak'], fired_at=out['ended']['fired'],
                                                           by='new_conversation' if kind in ('first', 'takeover') else 'cycle_end'))
            c = out.get('check')
            if not c:
                return
            stats['actrev_shadow_checks'] = stats.get('actrev_shadow_checks', 0) + 1
            if c['first']:
                stats['actrev_shadow_episodes8'] = stats.get('actrev_shadow_episodes8', 0) + 1
            if c['would_fire']:
                stats['actrev_shadow_fires'] = stats.get('actrev_shadow_fires', 0) + 1
                self._v70h_actrev = dict(c)               # v7.0 item 17 (ARC3_V70H_ACTREV): acted on by the caller
            attempt = len(self._v61_attempts(game, level)) + 1 if isinstance(level, int) else None
            self._v6_log('v69-actrev-shadow', dict(base, attempt=attempt, consecutive=c['consecutive'], p=c['p'], k=c['k'],
                                                   window=c['hi'] - c['lo'], new_boards=c['new'],
                                                   check='pass' if c['new'] == 0 else 'fail', would_fire=c['would_fire'],
                                                   k_fire=ACTREV_K69, **self._v69_as_guard()))
        except Exception as exc:  # noqa: BLE001
            stats['actrev_shadow_errors'] = stats.get('actrev_shadow_errors', 0) + 1
            if stats['actrev_shadow_errors'] == 1:
                self._v6_log('v69-actrev-shadow-error', dict(error='%s: %s' % (type(exc).__name__, str(exc)[:200])))

    # ----------------------------------------------------------------------------------------------- v7.0 harness revision
    def _v70h_attempt_rows(self, session, level):
        """v7.0 item 10: [(actions, no-ops, how it ended)] of this level's ended attempts, counted as _v61_attempts counts them."""
        d = getattr(session, '__dict__', {}) or {}
        lengths = list((d.get('v6_level_attempts') or {}).get(level, []))
        noops = list((d.get('v61_level_attempt_noops') or {}).get(level, []))
        ends = list((d.get('v62_level_attempt_ends') or {}).get(level, []))
        noops += [0] * (len(lengths) - len(noops))
        ends += [None] * (len(lengths) - len(ends))
        offset = (d.get('v61_attempt_offset') or {}).get(level, 0)
        return list(zip(lengths, noops, ends))[offset:]

    @staticmethod
    def _v70h_row_text(n, m, end):
        how = {'game_over': TEXTS['v70h_end_game_over'], 'reset': TEXTS['v70h_end_reset'],
               'restart': TEXTS['v70h_end_restart']}.get(end, '')
        return _fill(TEXTS['v70h_attempt_row'], n=n, s='' if n == 1 else 's', how=how, m=m)

    def _v70h_lengths_text(self, session, level):
        """v7.0 item 10: the last 5 ended attempts with how each ended, and the level's total over all its ended attempts."""
        rows = self._v70h_attempt_rows(session, level)
        if not rows:
            return '?'
        k = len(rows)
        if V75_POLISH:                               # v7.5 TXT-4 / TXT-8: alike endings grouped; "the one ended attempt"
            return self._v75_lengths_text(rows)
        return '; '.join(self._v70h_row_text(*r) for r in rows[-5:]) + _fill(
            TEXTS['v70h_attempt_total'], total=sum(int(r[0] or 0) for r in rows), k=k, ks='' if k == 1 else 's')

    def _v75_lengths_text(self, rows):
        """v7.5 TXT-4 / TXT-8 (the user 10-23, review Part 3): _v70h_lengths_text's facts with consecutive attempts that ended the
        same way grouped ("12, 7 and 30 actions, each ended with your RESET (3, 0 and 5 changed nothing in the interior)"), and a
        single ended attempt's total said as "in the one ended attempt"."""
        groups = []
        for n, m, end in rows[-5:]:
            if groups and groups[-1][0] == end:
                groups[-1][1].append((n, m))
            else:
                groups.append((end, [(n, m)]))
        parts = []
        for end, items in groups:
            if len(items) == 1:
                parts.append(self._v70h_row_text(items[0][0], items[0][1], end))
                continue
            how = {'game_over': TEXTS['v75_end_each_game_over'], 'reset': TEXTS['v75_end_each_reset'],
                   'restart': TEXTS['v75_end_each_restart']}.get(end, '')
            parts.append(_fill(TEXTS['v75_attempt_group'], ns=_v75_and([n for n, _m in items]), how=how,
                               ms=_v75_and([m for _n, m in items])))
        k, total = len(rows), sum(int(r[0] or 0) for r in rows)
        if k == 1:
            tail = _fill(TEXTS['v75_attempt_total_one'], total=total, s='' if total == 1 else 's')
        else:
            tail = _fill(TEXTS['v70h_attempt_total'], total=total, k=k, ks='s')
        return '; '.join(parts) + tail

    def _v70h_reset_ledger(self, session, level, s):
        """v7.0 item 10: the line after a model RESET that ended an attempt ('' otherwise). Receipt v70h_reset_ledgers."""
        recs = list(s.get('v61_actions') or [])
        if not any(str(r.get('action_display') or '') == 'RESET' and r.get('v6_attempt') for r in recs):
            return ''
        rows = self._v70h_attempt_rows(session, level)
        if not rows:
            return ''
        self._v6_stats['v70h_reset_ledgers'] = self._v6_stats.get('v70h_reset_ledgers', 0) + 1
        key = 'v75_reset_ledger_last5' if V75_POLISH and len(rows) > 5 else 'v70h_reset_ledger'   # v7.5 TXT-4: 5 of k listed
        line = _fill(TEXTS[key], k=len(rows), level=level, lengths=self._v70h_lengths_text(session, level))
        if V77:                                              # v7.7 (lever 5): the ended attempt's step range, before the reminder
            st77 = self._v77_steps_line(len(rows), rows[-1][0], self._v77_last_step(s, 'reset'))
            if st77:
                line += ' ' + st77
        if V76:                                              # v7.6 (the user 10-24): the RESET reminder after a model RESET
            self._v6_stats['v76_reset_lines'] = self._v6_stats.get('v76_reset_lines', 0) + 1
            line += ' ' + TEXTS['v76_reset_line']
        return line

    @staticmethod
    def _v70h_short(verdict):
        return str(verdict).split(' |')[0].split(' (')[0]

    def _v70h_board_ledger_lines(self, current_frame, history_entries, valid_actions, death=False):
        """v7.0 item 15 (ARC3_V70H_BOARD_LEDGER): (1) this level's actions executed from a board identical to the current one
        (the whole board compared, as the death ledger does): counts per action type with their verdicts, and the valid types
        not executed from it; (2) when the current board equals the board after an earlier step of this attempt, that step and
        the actions since the latest board new to this level. A per-level record, fed in history order (game-scoped; a level
        change starts it again). Receipts v70h_board_ledger_lines, v70h_revisit_lines; the v70h-board-ledger event."""
        from inference.framework import solver as _sv
        entries = list(history_entries or [])
        level = getattr(current_frame, 'level', None)
        st = self.__dict__.get('_v70h_bl')
        if st is None or st['level'] != level or st['upto'] > len(entries):
            start = 0
            for i in range(len(entries) - 1, 0, -1):
                if (getattr(entries[i], 'result', None) or {}).get('level_completed'):
                    start = i
                    break
            st = self._v70h_bl = dict(level=level, upto=start + 1, frm={}, after={}, last_new=None, attempt_from=start)
            fr = getattr(entries[start], 'frame', None) if entries else None
            if fr is not None:
                st['after'][hash(getattr(fr, 'grid', None))] = [(start, getattr(fr, 'step', None))]
                st['last_new'] = getattr(fr, 'step', None)
        for i in range(max(int(st['upto']), 1), len(entries)):
            e, prev = entries[i], entries[i - 1]
            r = getattr(e, 'result', None) or {}
            fr, pf = getattr(e, 'frame', None), getattr(prev, 'frame', None)
            step = getattr(fr, 'step', None)
            if r.get('automatic') or (str(getattr(e, 'action', '') or '').upper().startswith('RESET') and r.get('board_changed')):
                st['attempt_from'] = i                        # an attempt starts at its first board
            if not r.get('automatic') and pf is not None:
                m = re.match(r'[A-Z0-9]+', str(getattr(e, 'action', '') or ''))
                if m:
                    c = st['frm'].setdefault(hash(getattr(pf, 'grid', None)), {})
                    kv = (m.group(0), self._v70h_short(_sv._action_verdict(r)))
                    c[kv] = c.get(kv, 0) + 1
            if fr is not None and not r.get('level_completed'):
                key = hash(getattr(fr, 'grid', None))
                if key not in st['after']:
                    st['last_new'] = step
                st['after'].setdefault(key, []).append((i, step))
        st['upto'] = len(entries)
        out = []
        cur = hash(getattr(current_frame, 'grid', None))
        counts = st['frm'].get(cur)
        stats = self._v6_stats
        untried = []
        if counts:
            types = {}
            for (t, v), n in counts.items():
                types.setdefault(t, {})
                types[t][v] = types[t].get(v, 0) + n
            acts = '; '.join('%s x%d (%s)' % (t, sum(vs.values()), ', '.join('%s x%d' % kv for kv in
                                                                             sorted(vs.items(), key=lambda kv: -kv[1])))
                             for t, vs in sorted(types.items(), key=lambda tv: -sum(tv[1].values())))
            valid = [a.strip() for a in _v6_valid_display(valid_actions).split(',')
                     if a.strip() and a.strip() not in ('none', 'RESET')]
            untried = [a for a in valid if a not in types]
            line = _fill(TEXTS['v70h_bl_line'], acts=acts)
            if untried:
                line += ' ' + _fill(TEXTS['v70h_bl_untried'], types=', '.join(untried))
            if not death:
                line += ' ' + TEXTS['v70h_bl_caveat']
            out.append(line)
            stats['v70h_board_ledger_lines'] = stats.get('v70h_board_ledger_lines', 0) + 1
        here = [s for (i, s) in st['after'].get(cur, []) if st['attempt_from'] <= i < len(entries) - 1]
        step_now = getattr(current_frame, 'step', None)
        if here and isinstance(step_now, int) and isinstance(st['last_new'], int):
            n = max(0, step_now - st['last_new'])
            out.append(_fill(TEXTS['v70h_revisit'], s=here[-1], n=n, ns='' if n == 1 else 's'))
            stats['v70h_revisit_lines'] = stats.get('v70h_revisit_lines', 0) + 1
        if out:
            self._v6_log('v70h-board-ledger', dict(level=level, step=step_now, ledger=bool(counts),
                                                   types=sorted({t for (t, _v) in (counts or {})}), untried=untried,
                                                   revisit=here[-1] if here else None))
        return out

    def _v70h_il_parts(self, current_frame, history_entries):
        """v7.0 item 11 (ARC3_V70H_IL_OPENER): an in-level refresh opener's block, after the clearing-step block and before the
        current board: the level's last executed actions as the reports stated them (newest first within
        ARC3_V70H_IL_OPENER_TOKENS tokens and ARC3_V70H_IL_OPENER_KMAX actions; shown oldest first), the level's first board,
        and the change since it. Receipts v70h_il_openers, v70h_il_k, v70h_il_tokens; the v70h-il-opener event."""
        if V72_ILB:                                         # v7.2: the current attempt's boards instead (apply_v72.py)
            return self._v72_ilb_parts(current_frame, history_entries)
        from inference.framework import solver as _sv
        entries = list(history_entries or [])
        level = getattr(current_frame, 'level', None)
        start = 0
        for i in range(len(entries) - 1, 0, -1):
            if (getattr(entries[i], 'result', None) or {}).get('level_completed'):
                start = i
                break
        first = getattr(entries[start], 'frame', None) if entries else None
        if first is not None and getattr(first, 'level', None) != level:
            first = None
        lines, total, k = [], 0, 0
        n_acts = max(0, len(entries) - start - 1)
        for i in range(len(entries) - 1, start, -1):
            e = entries[i]
            r = getattr(e, 'result', None) or {}
            step = getattr(getattr(e, 'frame', None), 'step', '?')
            if r.get('automatic'):
                line = _fill(TEXTS['v70h_il_auto'], step=step)
            else:
                verdict = _sv._action_verdict(r).replace('(attempt ? ended here)', '(the attempt ended here)')
                line = _fill(TEXTS['v70h_il_line'], step=step, action=str(getattr(e, 'action', '') or '?'), verdict=verdict)
            n = self._v6_count(line)
            if k >= V70H_ILOPEN_KMAX or (lines and total + n > V70H_ILOPEN_TOKENS):
                break
            lines.append(line)
            total += n
            k += 1
        lines.reverse()
        parts = []
        if lines:
            head = _fill(TEXTS['v70h_il_head'], k=k, ks='' if k == 1 else 's', level=level)
            if n_acts > k:
                head += ' ' + _fill(TEXTS['v70h_il_more'], m=n_acts - k)
            parts.append(dict(type='text', text='\n' + head + '\n' + '\n'.join(lines) + '\n'))
        if first is not None:
            parts += [dict(type='text', text=_fill(TEXTS['v70h_il_first'], level=level, step=getattr(first, 'step', '?'))),
                      current_grid_image_part(first)]
            if getattr(first, 'grid', None) == getattr(current_frame, 'grid', None):
                parts.append(dict(type='text', text=TEXTS['v70h_il_same']))
            else:
                d = diff_image_part(first, current_frame)
                if d is not None:
                    parts += [dict(type='text', text=TEXTS['v70h_il_change']), d]
        stats = self._v6_stats
        stats['v70h_il_openers'] = stats.get('v70h_il_openers', 0) + 1
        stats.setdefault('v70h_il_k', []).append(k)
        stats.setdefault('v70h_il_tokens', []).append(total)
        self._v6_log('v70h-il-opener', dict(level=level, step=getattr(current_frame, 'step', None), k=k, tokens=total,
                                            on_level=n_acts, first_step=getattr(first, 'step', None) if first is not None else None,
                                            images=sum(1 for p in parts if p.get('type') == 'image_url')))
        return parts

    def _v72_ilb_parts(self, current_frame, history_entries):
        """v7.2 item 11 redesign (ARC3_V72_IL_BOARDS; the user 10-20, V67_DESIGN 201-202; IE's refinements): the block of an
        in-level refresh opener (also the NO_HARD_RESTART and park returns) shows boards of the CURRENT ATTEMPT only (from the
        level's first board, or the board after the latest RESET or automatic restart): its distinct boards in step order (the
        interior key, as the sandbox's boards_seen), the current board excluded (the opener's own "Current board." image follows),
        at most V72_ILB_BOARDS: the attempt's starting board, the last before the current, evenly spaced ones between; the starting
        board as a text pointer when the clearing-step block above shows the identical board; each board captioned with its step
        and the action that produced it; then the attempt's last V72_ILB_ACTIONS executed actions with the verdicts the reports
        stated; then the line that the current board follows. Facts only. Receipts as item 11 (v70h_il_openers, v70h_il_k,
        v70h_il_tokens; the v70h-il-opener event) plus v72_ilb_openers / v72_ilb_boards and the v72-ilb event."""
        if V77:                                             # v7.7 (the user 10-24): the text first, then the snapshot boards
            return self._v77_ilb_parts(current_frame, history_entries)
        from inference.framework import solver as _sv
        from inference.agent.runtime_state import Frame
        entries = list(history_entries or [])
        level = getattr(current_frame, 'level', None)
        start = 0
        for i in range(len(entries) - 1, 0, -1):
            if (getattr(entries[i], 'result', None) or {}).get('level_completed'):
                start = i
                break
        a0, how = start, 'level'
        for i in range(len(entries) - 1, start, -1):
            r = getattr(entries[i], 'result', None) or {}
            if r.get('automatic'):
                a0, how = i, 'auto'
                break
            if str(r.get('action_name') or '') == 'RESET' or str(getattr(entries[i], 'action', '') or '').startswith('RESET'):
                a0, how = i, 'reset'
                break
        if not entries or getattr(getattr(entries[a0], 'frame', None), 'level', None) != level:
            a0, how = None, None
        cur_key = _v69_interior_key(getattr(current_frame, 'grid', None) or ())
        distinct, seen = [], {cur_key}
        for i in (range(a0, len(entries)) if a0 is not None else ()):
            fr = getattr(entries[i], 'frame', None)
            key = _v69_interior_key(getattr(fr, 'grid', None) or ())
            if fr is not None and key not in seen:
                seen.add(key)
                distinct.append(i)
        n = len(distinct)
        if n <= V72_ILB_BOARDS:
            pick = list(distinct)
        else:
            pick = [distinct[round(j * (n - 1) / (V72_ILB_BOARDS - 1))] for j in range(V72_ILB_BOARDS)]
        hows = {'level': TEXTS['v72_ilb_how_level'], 'reset': TEXTS['v72_ilb_how_reset'], 'auto': TEXTS['v72_ilb_how_auto']}
        start_step = getattr(getattr(entries[a0], 'frame', None), 'step', '?') if a0 is not None else None
        after = None                                        # the clearing-step block's after board (level >= 2): a pointer, no copy
        if CLEAR_STEP69 and int(level or 0) >= 2:
            rec = ((getattr(self._v61_session(), '__dict__', None) or {}).get('v69_clears') or {}).get(int(level) - 1)
            if rec and rec.get('after') is not None:
                after = (_v69_interior_key(rec['after']), rec.get('level'))
        parts, shown, pointer = [], [], False
        if a0 is not None:
            if not pick:
                head = _fill(TEXTS['v72_ilb_head_none'], level=level)
            elif len(pick) == n:                            # IE: every distinct board shown: no 'n of n'
                head = _fill(TEXTS['v72_ilb_head_all'], level=level, d=n, ds='' if n == 1 else 's')
            else:
                head = _fill(TEXTS['v72_ilb_head'], level=level, n=len(pick), d=n)
            parts.append(dict(type='text', text='\n' + head))
            if a0 not in pick:                              # the starting board equals the current board (excluded)
                parts.append(dict(type='text', text=_fill(TEXTS['v72_ilb_start_is_current'], step=start_step, how=hows[how])))
            for i in pick:
                fr = entries[i].frame
                if i == a0:
                    if after is not None and _v69_interior_key(getattr(fr, 'grid', None) or ()) == after[0]:
                        parts.append(dict(type='text', text=_fill(TEXTS['v72_ilb_start_shown'], step=start_step, how=hows[how],
                                                                  k=after[1])))
                        pointer = True
                        shown.append(getattr(fr, 'step', None))
                        continue
                    cap = _fill(TEXTS['v72_ilb_start'], step=start_step, how=hows[how])
                else:
                    cap = _fill(TEXTS['v72_ilb_board'], step=getattr(fr, 'step', '?'), action=str(getattr(entries[i], 'action', '') or '?'))
                img = current_grid_image_part(fr)
                if img is None:
                    continue
                parts += [dict(type='text', text=cap), img]
                shown.append(getattr(fr, 'step', None))
        lines, total = [], 0
        for i in range(len(entries) - 1, (a0 if a0 is not None else start), -1):
            if len(lines) >= V72_ILB_ACTIONS:
                break
            e = entries[i]
            r = getattr(e, 'result', None) or {}
            step = getattr(getattr(e, 'frame', None), 'step', '?')
            if r.get('automatic'):
                line = _fill(TEXTS['v70h_il_auto'], step=step)
            else:
                verdict = _sv._action_verdict(r).replace('(attempt ? ended here)', '(the attempt ended here)')
                line = _fill(TEXTS['v70h_il_line'], step=step, action=str(getattr(e, 'action', '') or '?'), verdict=verdict)
            lines.append(line)
            total += self._v6_count(line)
        lines.reverse()
        if lines:
            parts.append(dict(type='text', text='\n' + _fill(TEXTS['v72_ilb_actions_head'], k=len(lines), ks='' if len(lines) == 1 else 's')
                              + '\n' + '\n'.join(lines) + '\n'))
        elif a0 is not None:                                # IE: an attempt without an executed action (a refresh right after a RESET)
            parts.append(dict(type='text', text=TEXTS['v72_ilb_no_action']))
        if parts:
            parts.append(dict(type='text', text=TEXTS['v72_ilb_tail']))
        stats = self._v6_stats
        stats['v70h_il_openers'] = stats.get('v70h_il_openers', 0) + 1
        stats.setdefault('v70h_il_k', []).append(len(lines))
        stats.setdefault('v70h_il_tokens', []).append(total)
        stats['v72_ilb_openers'] = stats.get('v72_ilb_openers', 0) + 1
        stats.setdefault('v72_ilb_boards', []).append(len(shown))
        images = sum(1 for p_ in parts if p_.get('type') == 'image_url')
        self._v6_log('v70h-il-opener', dict(level=level, step=getattr(current_frame, 'step', None), k=len(lines), tokens=total,
                                            on_level=max(0, len(entries) - start - 1), first_step=start_step, images=images))
        self._v6_log('v72-ilb', dict(level=level, step=getattr(current_frame, 'step', None), attempt_start=start_step, how=how,
                                     distinct=n, shown=shown, pointer=pointer, images=images, actions=len(lines)))
        return parts

    def _v70h_resident_after(self):
        """v7.0 item 19b receipt (IE): after the generation that followed a pointer, the server's board_resident for the pointed
        observation (not True: the image left the context while that request was processed; the model still has current_frame).
        Event v70h-resident-after; stat v70h_resident_lost."""
        ptr = self.__dict__.pop('_v70h_ptr', None) or {}
        sess = getattr(self, '_v6_session', None)
        oid = 'obs-%08d' % int(ptr.get('obs') or 0)
        b = next((b for b in (getattr(sess, 'blocks', None) or []) if isinstance(b, dict) and b.get('observation_id') == oid), None)
        resident = None if b is None else b.get('board_resident')
        if resident is not True:
            self._v6_stats['v70h_resident_lost'] = self._v6_stats.get('v70h_resident_lost', 0) + 1
        self._v6_log('v70h-resident-after', dict(obs=ptr.get('obs'), resident=resident,
                                                 rebased=oid in (getattr(sess, 'rebased_away', None) or ())))

    def _v70h_board_resident(self, current_frame):
        """v7.0 item 19b: True when this conversation's latest observation that began with a board image showed the current
        board and the server reports that image resident (history_state.blocks[obs-t].board_resident True, not rebased away)."""
        rec = self.__dict__.get('_v70h_img')
        if not rec or rec[0] != self._v69_ir_conv() or rec[2] != getattr(current_frame, 'grid', None):
            return False
        sess = getattr(self, '_v6_session', None)
        oid = 'obs-%08d' % int(rec[1])
        if sess is None or oid in (getattr(sess, 'rebased_away', None) or ()):
            return False
        b = next((b for b in (getattr(sess, 'blocks', None) or []) if isinstance(b, dict) and b.get('observation_id') == oid), None)
        if not (b is not None and b.get('board_resident') is True):
            return False
        if V75_POLISH:                               # v7.5 RUN-2: only an image inside the protected recent tail (not evictable)
            receipt = getattr(sess, 'last_receipt', None) or {}
            tail = int((getattr(sess, 'history_eviction', None) or {}).get('protect_recent_tokens') or 0)
            start = b.get('start')
            if not isinstance(start, int) or start < int(receipt.get('logical_tokens') or 0) - tail:
                self._v6_stats['v75_pointer_outside_tail'] = self._v6_stats.get('v75_pointer_outside_tail', 0) + 1
                return False
        return True

    def _v69_as_guard(self):
        """v6.9 ACT-REV shadow: the loop ladder's state at this report (guard_rung None without the ladder or an episode): 0 its
        C1 lines, 1 the loop-rule block shown, 2 / 3 a loop restart pending; its consecutive report count and refused calls."""
        if not LOOP:
            return dict(guard_rung=None)
        st = self.__dict__.get('_v67_loop') or {}
        pend = self.__dict__.get('_v67_pending') or {}
        if self._v62_restart_reason == 'loop' and pend.get('loop'):
            rung = int(pend['loop'].get('rung') or 2)
        elif st.get('block'):
            rung = 1
        elif st.get('consecutive'):
            rung = 0
        else:
            rung = None
        return dict(guard_rung=rung, guard_consecutive=st.get('consecutive'), guard_refused=st.get('refused'))

    def _action_guard_hook(self, grid, action_name, action_data, level=None, remaining=None):
        """v6.2 fix (2a, decision 3: refuse an action already run from the same INTERIOR board in this snippet). His hash
        covers the whole frame (ARC3_STATE_IDENTITY_BORDER defaults to 0), so a step bar in the edge band made every
        board new and the guard could never fire in a game with a bar. Here the guard sees the board inside the 4-cell
        edge band. Only the repeat-state check reads this hash (the known-no-op and known-death refusals are off; the
        death ledger hashes its own frames, whole, as before)."""
        from inference.framework import solver as _sv
        b = int(getattr(_sv, 'NOOP_GUARD_BORDER_WIDTH', 4) or 0)
        if grid and b > 0 and len(grid) > 2 * b and len(grid[0]) > 2 * b:
            grid = tuple(tuple(row[b:len(row) - b]) for row in grid[b:len(grid) - b])
        verdict = super()._action_guard_hook(grid, action_name, action_data, level, remaining)
        # round 4 (A3 bp35 ~8K tokens, ls20 ~3K, g50t A2): say which board is compared and which earlier action matched
        try:
            key = (tuple(tuple(r) for r in grid), str(action_name), json.dumps(action_data, sort_keys=True, default=str))
        except Exception:  # noqa: BLE001
            return verdict
        first = self.__dict__.setdefault('_v62_guard_first', {})
        if verdict and verdict[0] == 'repeated_action_in_state' and isinstance(verdict[1], str):
            k = first.get(key)
            text = verdict[1].replace(
                'was already run from this exact board state earlier in this snippet',
                'was already run from this same board earlier in this snippet%s (the board inside the 4-cell edge '
                'band; the band is not compared)' % (' (as executed action %d of the snippet)' % k if k else ''))
            text = text.replace('Nothing after the call ran.', 'Nothing after the call ran; output printed before it is kept, '
                                'so print inside a loop to keep partial results.')   # tr87 A3 T46
            return (verdict[0], text) + tuple(verdict[2:])
        if not verdict:
            game = self._v61_session()
            first.setdefault(key, int((getattr(game, '__dict__', {}) or {}).get('v61_call_executed', 0)) + 1)
        return verdict

    # ----------------------------------------------------------------------------------------------- v6.9 RAB
    def _v69_att_tokens(self):
        """v6.9 RAB: the tokens generated on the level since it began or the harness last restarted it, over all its
        conversations (in-level refreshes and parks do not end this count; a level-up, a level-up refresh and every harness
        restart start it again). Kept only with ARC3_V69_RESTART_ACROSS or ARC3_V69_PARK_TOKENS."""
        return int(self.__dict__.get('_v69_att') or 0)

    def _v69_rt_across(self, req_key, level):
        """v6.9 RAB RESTART_ACROSS: the token restart's notes request when the level's count spans more than one conversation
        (an in-level refresh or a park since the level began or the harness last restarted it): attempt_summary_request_across
        with that count and since = 'it began' | 'the harness last restarted it' (the game session's restart record on the
        level); None otherwise (the request as before: the conversation's count equals the level's)."""
        att = self._v69_att_tokens()
        if (req_key != 'attempt_summary_request' or self._v62_restart_reason != 'tokens'
                or att <= int(self._v61_level_tokens or 0) or 'attempt_summary_request_across' not in TEXTS
                or (getattr(self, '_v67_pending', None) or {}).get('kind') == 'soft'):
            return None
        d = getattr(self._v61_session(), '__dict__', None)
        recs = ((d if isinstance(d, dict) else {}).get('v67_level_restarts') or {}).get(level) or []
        self._v6_stats['restart_notes_across'] = self._v6_stats.get('restart_notes_across', 0) + 1
        return _fill(TEXTS['attempt_summary_request_across'], level=level, tokens=format(att, ','),
                     since='the harness last restarted it' if recs else 'it began')

    def _v69_rt_supersede(self, level, current_frame, history_entries):
        """v6.9 RAB RESTART_ACROSS, at a turn start with an in-level refresh (or park) pending: when the level's count has reached
        the budget, no session error is pending and the late gate allows a restart, the restart wins at this turn: the pending
        refresh is dropped (inlevel_superseded, the v69-inlevel-superseded event) and the restart follows in this turn start
        with its notes turn done (the in-level one: summary state 'done'; reason 'tokens'; the kind decided here, as at a notes
        request). True then; False: the refresh goes ahead."""
        if not RESTART_TOKENS or getattr(self, '_v62_force_restart', False):
            return False
        att = self._v69_att_tokens()
        if att < RESTART_TOKENS or not self._v62_restart_time_ok(level):
            return False
        st = self.__dict__.get('_v69_ir') or {}
        kind67, why67 = self._v67_restart_kind(current_frame, history_entries)
        self._v67_pending = dict(level=level, kind=kind67, why=why67)
        self._v62_restart_reason = 'tokens'
        if st.get('pending') == 'notes':
            self._v62_summary_state = 'done'
        stats = self._v6_stats
        stats['inlevel_superseded'] = stats.get('inlevel_superseded', 0) + 1
        self._v6_log('v69-inlevel-superseded', dict(level=level, step=getattr(current_frame, 'step', None), attempt_tokens=att,
                                                    tokens_on_level=int(self._v61_level_tokens or 0), budget=RESTART_TOKENS,
                                                    pending=st.get('pending'), park=bool(st.get('park'))))
        self._v69_ir = {}
        return True

    def _v69_waiting(self):
        """v6.9 RAB PARK (simalloc69, decision 127: parking gains only when a never-started game takes the lane, and loses when any
        waiting game does): the NEVER-STARTED games waiting for a lane, read under the gate's lock. A never-started game queues with
        dfranzen's untrimmed priority (_PRIORITY_UNTRIMMED_BASE - its queue position, above _PRIORITY_BAND) and no snapshot; a game
        swapped out at a restart or parked queues with a _V62Snapshot in a lower band, and does not count. 0 without the gate."""
        gate = ta._priority_gate()
        if gate is None:
            return 0
        import contextlib
        lock = getattr(gate, '_cond', None)
        try:
            with (lock if lock is not None else contextlib.nullcontext()):
                waiting = list(getattr(gate, '_waiting', None) or [])
                snaps = dict(getattr(gate, '_snapshots', None) or {})
        except Exception:  # noqa: BLE001
            return 0
        band = int(getattr(ta, '_PRIORITY_BAND', 1_000_000))
        return sum(1 for neg, token in waiting if (token not in snaps and -int(neg) > band)
                   or (isinstance(snaps.get(token), _V70Snapshot) and snaps[token].state.get('never_started')))   # v7.0

    def _v69_pk_due(self, level):
        """v6.9 RAB PARK: at a mid-level opener, the number of never-started games waiting for a lane when the rung is due: the
        level's count lies PARK69 or more tokens past the rung's anchor on this level (its first in-level refresh in that count,
        or the game's last rung firing on it); 0 otherwise."""
        a = self.__dict__.get('_v69_pk') or {}
        if a.get('level') != level or self._v69_att_tokens() - int(a.get('base') or 0) < PARK69:
            return 0
        return self._v69_waiting()

    def _v69_pk_fingerprint(self):
        """v6.9 RAB PARK: what the new conversation after a park reads, one digest per record: the game session (the object's
        id; its history length; v69_clears, v69_inlevel_refreshes, v67_level_restarts, v6_level_attempts,
        v62_level_attempt_ends) and the agent (the attempt notes, the level summaries, the library's names and sources, the
        programs' levels)."""
        import hashlib

        def h(x):
            return hashlib.sha1(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()[:12]
        game = self._v61_session()
        d = getattr(game, '__dict__', None)
        d = d if isinstance(d, dict) else {}
        out = dict(session=id(game), history=len(getattr(game, 'history_entries', None) or []),
                   clears=sorted((d.get('v69_clears') or {}).keys()))
        for k in ('v69_clears', 'v69_inlevel_refreshes', 'v67_level_restarts', 'v6_level_attempts', 'v62_level_attempt_ends'):
            out[k] = h({str(kk): v for kk, v in (d.get(k) or {}).items()})
        out['notes'] = h({str(k): v for k, v in (self._v62_attempt_notes or {}).items()})
        out['summaries'] = h({str(k): v for k, v in (self._v6_rules or {}).items()})
        out['library'] = h({n: str(self._kept_functions.get(n)) for n in sorted(self._kept_functions)})
        out['library_levels'] = h(dict(self._v61_library_levels or {}))
        return out

    def _v69_pk_park(self, current_frame):
        """v6.9 RAB PARK, at the turn start after the rung's notes turn: when a never-started game still waits for a lane, give
        up this game's lane (the session closed, its KV freed) and queue in the parked band (_V62Snapshot.parked: below
        every other game, also when the gate re-scores the queue); True once the game holds a lane again (the in-level refresh
        then opens the new conversation). Without such a game, or without the gate, the lane is kept: False (park_skipped).
        The v69-park event carries the records the new conversation reads, fingerprinted before the wait, and whether they
        were the same after it (survived; park_records_changed counts the games where they were not)."""
        st = self.__dict__.get('_v69_ir') or {}
        if V71:                                                 # v7.1 ALLOCATION: its park (any waiting game, the time guard)
            return self._v71_park(current_frame)
        if st.get('trigger') in ('index', 'cap'):            # v7.0: the cap bind's and the index swap's park
            return self._v70_park(current_frame)
        level = getattr(current_frame, 'level', None) or 1
        gate = ta._priority_gate()
        waiting = self._v69_waiting() if gate is not None else 0
        a = self.__dict__.get('_v69_pk') or {}
        att = self._v69_att_tokens()
        stats = self._v6_stats
        base = dict(level=level, step=getattr(current_frame, 'step', None), attempt_tokens=att, anchor=a.get('base'),
                    since_anchor=att - int(a.get('base') or 0), waiting_at_trigger=st.get('waiting'), waiting=waiting)
        if not waiting:
            st['parked'] = False
            stats['park_skipped'] = stats.get('park_skipped', 0) + 1
            self._v6_log('v69-park', dict(base, parked=False, why='no_gate' if gate is None else 'no_game_waits'))
            return False
        before = self._v69_pk_fingerprint()
        game = self._v61_session()
        n_levels = getattr(getattr(getattr(game, 'game', None), 'game_run', None), 'number_of_levels', None)
        try:
            self._v6_session.close()                          # free the KV while waiting; the in-level refresh opens a new one
        except Exception:  # noqa: BLE001
            pass
        snapshot = _V62Snapshot(level, n_levels, self._v62_swaps.get(level, 0))
        snapshot.parked = True
        if V70_ADMIT:                                          # v7.0 INDEX_ADMIT: queued by its index (parked flag)
            snapshot = self._v70_snapshot(level, 'index')
        priority = snapshot.priority(time.monotonic(), getattr(gate, '_clock_deadline', None))
        t0 = time.time()
        gate.handover(priority, snapshot)
        waited = time.time() - t0
        after = self._v69_pk_fingerprint()
        st['parked'] = True
        stats['parks'] = stats.get('parks', 0) + 1
        stats['park_wait_s'] = round(stats.get('park_wait_s', 0) + waited, 1)
        stats.setdefault('park_levels', []).append(level)
        if after != before:
            stats['park_records_changed'] = stats.get('park_records_changed', 0) + 1
        self._v6_log('v69-park', dict(base, parked=True, priority=priority, waited_s=round(waited, 1), records=before,
                                      survived=after == before, changed=sorted(k for k in before if before[k] != after.get(k))))
        return True

    def _v69_pk_refreshed(self, level, st):
        """v6.9 RAB PARK, at an in-level refresh: the rung's anchor on the level (set at its first in-level refresh in the
        level's count; moved to the count now at every rung firing, parked or not) and the v69-inlevel-refresh event's fields
        (trigger park | logical, parked, attempt_tokens, park_anchor)."""
        a = self.__dict__.get('_v69_pk') or {}
        att = self._v69_att_tokens()
        if a.get('level') != level:
            a = dict(level=level, base=att, fires=0, parks=0)
        if st.get('park'):
            a = dict(a, base=att, fires=int(a.get('fires') or 0) + 1, parks=int(a.get('parks') or 0) + (1 if st.get('parked') else 0))
        self._v69_pk = a
        return dict(trigger='park' if st.get('park') else 'logical', parked=bool(st.get('parked')), attempt_tokens=att,
                    park_anchor=a['base'])

    def _v62_swap_out(self, current_frame, reason='tokens'):
        """v6.2 allocator: give up the lane (session closed) and re-queue; returns once a lane is held again (at once when
        nothing better waits). False when the gate is off (run B: the restart happens in place). v6.3: the queue band
        (item A) and the reason ('tokens' | 'noact' | 'position') are logged; zero-band swaps are counted."""
        gate = ta._priority_gate()
        if gate is None:
            return False
        level = getattr(current_frame, 'level', None) or 1
        swaps = self._v62_swaps.get(level, 0) + 1
        self._v62_swaps[level] = swaps
        game = self._v61_session()
        n_levels = getattr(getattr(getattr(game, 'game', None), 'game_run', None), 'number_of_levels', None)
        try:
            self._v6_session.close()                          # free the KV while waiting; the restart opens a new one
        except Exception:  # noqa: BLE001
            pass
        snapshot = _V62Snapshot(level, n_levels, swaps)
        snapshot.parked = bool(LOOP and reason == 'loop' and self._v67_park_due(level))   # v6.7 LOOP: parked
        if V70_ADMIT:                                          # v7.0 INDEX_ADMIT: queued by its index (a loop park lowest)
            snapshot = self._v70_snapshot(level, 'lowest' if snapshot.parked else 'index')
        waiting = len(getattr(gate, '_waiting', []) or [])
        priority = snapshot.priority(time.monotonic(), getattr(gate, '_clock_deadline', None))
        band = 'parked' if snapshot.parked else 'zero' if snapshot.zero_band else 'middle'
        if snapshot.zero_band:
            self._v6_stats['zero_band_swaps'] = self._v6_stats.get('zero_band_swaps', 0) + 1
        t0 = time.time()
        gate.handover(priority, snapshot)
        waited = time.time() - t0
        self._v6_stats['swaps'] = self._v6_stats.get('swaps', 0) + 1
        self._v6_stats['swap_wait_s'] = round(self._v6_stats.get('swap_wait_s', 0) + waited, 1)
        self._v6_log('v62-swap', dict(level=level, swaps_on_level=swaps, tokens_on_level=self._v61_level_tokens,
                                      n_levels=n_levels, priority=priority, waiting=waiting, waited_s=round(waited, 1),
                                      band=band, reason=reason, noact_tokens=getattr(self, '_v62_noact_tokens', 0)))
        return True

    # ----------------------------------------------------------------------------------------------- v7.0 SWAP70
    def _v70_gid(self):
        run = getattr(getattr(self._v61_session(), 'game', None), 'game_run', None)
        gid = getattr(run, 'game_id', None)
        if gid is not None:
            self._v70_gid_seen = gid
        return gid if gid is not None else self.__dict__.get('_v70_gid_seen')

    @staticmethod
    def _v70_clock():
        """(time left s or None, time run s) on the gate's shared monotonic clock"""
        gate = ta._priority_gate()
        now = time.monotonic()
        dl = getattr(gate, '_clock_deadline', None) if gate is not None else None
        t0 = getattr(gate, '_clock_start', None) if gate is not None else None
        return (None if dl is None else max(0.0, dl - now)), (0.0 if t0 is None else max(0.0, now - t0))

    def _v70_budget(self):
        """(the run's pool, game id, the game's count, its cap) or (None, gid, None, None)"""
        from v62 import solver as _v62s
        budget, game, gid = _v62s.GAMEBUDGET, self._v61_session(), self._v70_gid()
        if budget is None or gid is None or game is None:
            return None, gid, None, None
        return budget, gid, int(_v62s._gamebudget_spent(game)), budget.cap.get(gid)

    def _v70_state(self, level, parked=False):
        """the index's inputs now (v62/swap70.py index())"""
        tl, el = self._v70_clock()
        budget, gid, spent, cap = self._v70_budget()
        run = getattr(getattr(self._v61_session(), 'game', None), 'game_run', None)
        return dict(level=level, n_levels=getattr(run, 'number_of_levels', None),
                    tokens_on_level=int(self.__dict__.get('_v70_lt') or 0),
                    actions_on_level=int(self.__dict__.get('_v70_acts') or 0),
                    cleared=[list(c) for c in (self.__dict__.get('_v70_cleared') or [])], time_left_s=tl, elapsed_s=el,
                    cap_left=(cap - spent) if (budget is not None and cap is not None and not V70_SOFTCAP) else None,
                    never_started=False, parked=bool(parked))

    def _v70_snapshot(self, level, band):
        s = self._v70_state(level, parked=True)
        s.pop('time_left_s'), s.pop('elapsed_s')               # the gate fills them at every pump
        return _V70Snapshot(s, band, self._v70_gid())

    def _v70_level_change(self, level_now, history_entries):
        """v7.0: at a level change, the cleared level's tokens on level (over all its conversations) and actions (taaf's
        per-level count) join the game's cleared list (pace, b_hat); the level's count starts again"""
        prev = self.__dict__.get('_v70_level')
        if isinstance(prev, int) and isinstance(level_now, int) and level_now > prev:
            self.__dict__.setdefault('_v70_cleared', []).append(
                [int(self.__dict__.get('_v70_lt') or 0), int(self._v67_actions_on_level(prev, history_entries))])
            ret = self.__dict__.pop('_v72_ret', None)
            if ret is not None:                                # v7.2b PARK_RANK: the return's level cleared
                self._v6_stats['v72_return_clears'] = self._v6_stats.get('v72_return_clears', 0) + 1
                self._v6_log('v72-return-clear', dict(ret, level_now=level_now, seconds=round(time.time() - ret['epoch'], 1)))
        self._v70_level, self._v70_lt = level_now, 0

    def _v70_tokens(self, n):
        self._v70_lt = int(self.__dict__.get('_v70_lt') or 0) + n
        if self.__dict__.get('_v72_ret') is not None:          # v7.2b PARK_RANK: a return's tokens until its level clears
            self._v72_ret['tokens'] = int(self._v72_ret.get('tokens') or 0) + n
        self._v70_lt_total = int(self.__dict__.get('_v70_lt_total') or 0) + n
        self._v70_since = int(self.__dict__.get('_v70_since') or 0) + n
        if self.__dict__.get('_v70_return') is not None:       # a return's tokens to its first action
            self._v70_return['tokens'] = int(self._v70_return.get('tokens') or 0) + n

    def _v70_acted(self):
        self._v70_reoriented = True
        ret = self.__dict__.pop('_v70_return', None)
        if ret is not None:                                    # v7.0 receipt: a return's tokens to the first action
            self._v6_stats.setdefault('v70_return_tokens_to_action', []).append(int(ret.get('tokens') or 0))
            self._v6_log('v70-return-first-action', dict(ret))

    def _v70_turn(self, current_frame, history_entries, level):
        """v7.0, at every turn boundary: the claim sync (a claim without its pending park is dropped), the notes mark
        (NO_HARD_RESTART), the loop-recurrence receipt, and with INDEX the index of this game (running) and with the parked
        flag, published to the registry (the worst-running gate, the read); INDEX_SHADOW logs v70-index"""
        gid = self._v70_gid()
        st = self.__dict__.get('_v69_ir') or {}
        if V70_PARKS and gid is not None and not (st.get('pending') and st.get('trigger') in ('index', 'cap')):
            _s70.CLAIMS.drop(gid)
        if V70_NHR:
            if self._v62_summary_state is None:
                self._v70_notes_mark = (level, len(self._v62_attempt_notes.get(level) or []), int(self._v61_level_tokens or 0))
            mark = self.__dict__.get('_v70_after_loop')
            if mark and mark.get('conv') == self._v69_ir_conv() and mark.get('level') == level:
                stats = self._v6_stats
                rec = (stats.get('nhr_after_loop') or [{}])[-1]
                for k in ('loop_episodes', 'loop_rung1', 'loop_refused'):
                    rec[k] = int(stats.get(k) or 0) - int(mark['base'].get(k) or 0)
        if not V70_INDEX or not isinstance(level, int):
            return
        self._v70_acts = int(self._v67_actions_on_level(level, history_entries))
        s = self._v70_state(level)
        idx, comp = _s70.index(V70_C, s, detail=True)
        idx_p = _s70.index(V70_C, dict(s, parked=True))
        since = int(self.__dict__.get('_v70_since') or 0)
        reor = bool(self.__dict__.get('_v70_reoriented'))
        self._v70_last = dict(index=idx, parked=idx_p, level=level)
        _s70.REGISTRY.publish(gid, running=True, index=idx, parked_index=idx_p, level=level, since=since,
                              hyst_ok=since >= V70_C['hysteresis_tokens'] and reor)
        if V70_SHADOW:
            self._v6_stats['v70_index_logs'] = self._v6_stats.get('v70_index_logs', 0) + 1
            self._v6_log('v70-index', dict(comp, index_parked=idx_p, step=getattr(current_frame, 'step', None),
                                           tokens_on_level=s['tokens_on_level'], actions_on_level=s['actions_on_level'],
                                           cleared=s['cleared'], time_left_s=s['time_left_s'], elapsed_s=round(s['elapsed_s'], 1),
                                           cap_left=s['cap_left'], since_admission=since, reoriented=reor))

    def _v70_waiting(self, gate):
        """(eligible, candidates, best): the waiting games not over their own cap; those the swap compares with (all of them,
        or the never-started ones with park_new_only); the best candidate (index, kind, game) by the index"""
        import contextlib
        lock = getattr(gate, '_cond', None)
        with (lock if lock is not None else contextlib.nullcontext()):
            waiting = list(getattr(gate, '_waiting', None) or [])
            snaps = dict(getattr(gate, '_snapshots', None) or {})
        band = int(getattr(ta, '_PRIORITY_BAND', 1_000_000))
        now = time.monotonic()
        elig, cands, best = 0, 0, None
        kinds = self.__dict__.setdefault('_v70_kinds', {})
        kinds.clear()
        for neg, token in waiting:
            sn = snaps.get(token)
            k = (('never' if sn.state.get('never_started') else sn.band) if isinstance(sn, _V70Snapshot)
                 else 'parked' if getattr(sn, 'parked', False) else 'swapped' if sn is not None
                 else 'never' if -int(neg) > band else 'other')
            kinds[k] = kinds.get(k, 0) + 1
            if isinstance(sn, _V70Snapshot) and sn.band == 'overcap':
                continue
            elig += 1
            if V70_C is None:
                continue
            if isinstance(sn, _V70Snapshot):
                never, gid, v = bool(sn.state.get('never_started')), sn.gid, None
            elif sn is None and -int(neg) > band:              # a never-started game queued without a V70 snapshot
                never, gid, v = True, None, None
            else:
                continue
            if V70_C['park_new_only'] and not never:
                continue
            cands += 1
            if v is None:
                v = (sn.index_gate(gate, now) if sn is not None else
                     _V70Snapshot(dict(level=1, never_started=True)).index_gate(gate, now))
            if best is None or v > best[0]:
                best = (v, 'never' if never else 'parked', gid)
        return elig, cands, best

    def _v70_park_due(self, level):
        """v7.0, at a mid-level opener (_v69_ir_notes, the rung's place): SOFT_CAP's bind, else INDEX_SWAP's rule; the
        decision (trigger 'cap' | 'index', waiting, the numbers) when this game parks, else None"""
        gid = self._v70_gid()
        gate = ta._priority_gate()
        if gid is None or gate is None:
            return None
        elig, cands, best = self._v70_waiting(gate)
        if V70_SOFTCAP:
            d = self._v70_cap_due(gid, level, elig)
            if d:
                return d
        if V70_SWAP:
            return self._v70_swap_due(gid, level, cands, best)
        return None

    def _v70_cap_due(self, gid, level, elig):
        """v7.0 SOFT_CAP: over the cap with an eligible game waiting: the reserve first (once, the first-cap-hit rule), then
        the bind (a claim per waiting game); over the cap with nobody eligible waiting: play on (counted, logged once per
        level)"""
        budget, gid, spent, cap = self._v70_budget()
        if budget is None or cap is None or spent < cap:
            return None
        stats = self._v6_stats
        if not elig:
            stats['soft_cap_play_on_checks'] = stats.get('soft_cap_play_on_checks', 0) + 1
            if self.__dict__.get('_v70_cap_logged') != level:
                self._v70_cap_logged = level
                self._v6_log('v70-softcap', dict(kind='play_on', level=level, tokens=spent, cap=cap, over=spent - cap, waiting=0,
                                                 kinds=dict(self.__dict__.get('_v70_kinds') or {})))
            return None
        still = budget.soft_bind(gid, spent)
        cap2 = budget.cap.get(gid)
        if not still:
            self._v6_log('v70-softcap', dict(kind='reserve_lifted', level=level, tokens=spent, cap_before=cap, cap=cap2,
                                             waiting=elig))
            return None
        if not _s70.CLAIMS.take(gid, elig):
            stats['soft_cap_claim_waits'] = stats.get('soft_cap_claim_waits', 0) + 1
            return None
        stats['soft_cap_binds'] = stats.get('soft_cap_binds', 0) + 1
        d = dict(trigger='cap', waiting=elig, kinds=dict(self.__dict__.get('_v70_kinds') or {}), tokens=spent, cap=cap2,
                 over=spent - cap2)
        self._v6_log('v70-softcap', dict(d, kind='bind', level=level))
        return d

    def _v70_swap_due(self, gid, level, cands, best):
        """v7.0 INDEX_SWAP: the rule (swap70.swap_rule) and its gates; every evaluation with a candidate waiting is logged
        (v70-swap); a park takes a claim"""
        last = self.__dict__.get('_v70_last') or {}
        stats = self._v6_stats
        if not best or last.get('level') != level:
            return None
        since = int(self.__dict__.get('_v70_since') or 0)
        mark = self.__dict__.get('_v70_swap_mark')
        if V70_C['check_every_tokens'] and mark is not None and int(self.__dict__.get('_v70_lt_total') or 0) - mark < \
                V70_C['check_every_tokens']:
            return None
        self._v70_swap_mark = int(self.__dict__.get('_v70_lt_total') or 0)
        I_R, I_Rp = float(last['index']), float(last['parked'])
        I_W, kind_w, gid_w = best
        fire, thr = _s70.swap_rule(V70_C, I_R, I_Rp, I_W)
        tl, el = self._v70_clock()
        reor = bool(self.__dict__.get('_v70_reoriented'))
        hyst = since >= V70_C['hysteresis_tokens'] and reor
        need = _s70.min_left_s(V70_C, el)
        time_ok = tl is None or tl >= need
        others = {g: e for g, e in _s70.REGISTRY.running().items() if g != gid and e.get('hyst_ok')}
        worst = all(I_R <= float(e.get('index', 0.0)) + 1e-18 for e in others.values())
        outcome = ('no_margin' if not fire else 'hysteresis' if not hyst else 'time' if not time_ok
                   else 'not_worst' if not worst else None)
        if outcome is None:
            outcome = 'park' if _s70.CLAIMS.take(gid, cands) else 'claimed'
        stats['v70_swap_' + outcome] = stats.get('v70_swap_' + outcome, 0) + 1
        ev = dict(level=level, I_R=I_R, I_R_parked=I_Rp, I_W=I_W, W_kind=kind_w, W_game=gid_w, threshold=thr,
                  margin_rel=V70_C['margin_rel'], margin_abs=V70_C['margin_abs'], cost_weight=V70_C['cost_weight'],
                  since_admission=since, reoriented=reor, hysteresis_tokens=V70_C['hysteresis_tokens'], time_left_s=tl,
                  min_left_s=round(need, 1), worst=worst, running_candidates=len(others), candidates=cands,
                  claims=_s70.CLAIMS.count(), outcome=outcome)
        self._v6_log('v70-swap', ev)
        return dict(ev, trigger='index', waiting=cands) if outcome == 'park' else None

    def _v70_park(self, current_frame):
        """v7.0, at the turn start after the park's notes turn (the rung's path): with an eligible game still waiting, give up
        the lane (session closed) and queue with the V70 snapshot ('overcap' after a cap bind, else 'index'); True once a lane
        is held again (the in-level refresh then opens the new conversation). Else the lane is kept (park_skipped). The v69-park
        event as the rung's (records fingerprinted before the wait, survived after it) with the trigger."""
        st = self.__dict__.get('_v69_ir') or {}
        level = getattr(current_frame, 'level', None) or 1
        gate = ta._priority_gate()
        gid = self._v70_gid()
        trig = st.get('trigger')
        elig = self._v70_waiting(gate)[0] if gate is not None else 0
        stats = self._v6_stats
        base = dict(level=level, step=getattr(current_frame, 'step', None), trigger=trig, waiting_at_trigger=st.get('waiting'),
                    waiting=elig, v70=st.get('v70'), tokens_on_level=int(self.__dict__.get('_v70_lt') or 0))
        if not elig:
            _s70.CLAIMS.drop(gid)
            st['parked'] = False
            stats['park_skipped'] = stats.get('park_skipped', 0) + 1
            stats['v70_park_skipped_' + str(trig)] = stats.get('v70_park_skipped_' + str(trig), 0) + 1
            self._v6_log('v69-park', dict(base, parked=False, why='no_gate' if gate is None else 'no_game_waits'))
            return False
        before = self._v69_pk_fingerprint()
        try:
            self._v6_session.close()                          # free the KV while waiting; the in-level refresh opens a new one
        except Exception:  # noqa: BLE001
            pass
        snapshot = self._v70_snapshot(level, 'overcap' if trig == 'cap' else 'index')
        priority = snapshot.priority_gate(gate, time.monotonic())
        _s70.REGISTRY.publish(gid, running=False, parked=True)
        _s70.CLAIMS.drop(gid)
        t0 = time.time()
        gate.handover(priority, snapshot)
        waited = time.time() - t0
        self._v70_since, self._v70_reoriented = 0, False
        self._v70_return = dict(level=level, trigger=trig, waited_s=round(waited, 1), tokens=0)
        _s70.REGISTRY.publish(gid, running=True, parked=False, since=0, hyst_ok=False)
        after = self._v69_pk_fingerprint()
        st['parked'] = True
        stats['parks'] = stats.get('parks', 0) + 1
        stats['v70_parks_' + str(trig)] = stats.get('v70_parks_' + str(trig), 0) + 1
        stats['park_wait_s'] = round(stats.get('park_wait_s', 0) + waited, 1)
        stats.setdefault('park_levels', []).append(level)
        if after != before:
            stats['park_records_changed'] = stats.get('park_records_changed', 0) + 1
        self._v6_log('v69-park', dict(base, parked=True, priority=priority, waited_s=round(waited, 1), records=before,
                                      survived=after == before, changed=sorted(k for k in before if before[k] != after.get(k))))
        return True

    # ------------------------------------------------------------------------------------------- v7.1 ALLOCATION (apply_v71.py)
    def _v71_due(self):
        """v7.1 ALLOCATION, at a mid-level opener: the trigger's numbers when the level's generated tokens in this conversation
        (_v61_level_tokens) have reached ARC3_V71_LEVEL_TOKENS, else None (pure: the caller logs)."""
        n = int(self._v61_level_tokens or 0)
        if n < V71_TOKENS:
            return None
        tl, el = self._v70_clock()
        return dict(count=n, threshold=V71_TOKENS, waiting=self._v71_waiting(), time_left_s=None if tl is None else round(tl, 1))

    def _v71_waiting(self):
        """v7.1 ALLOCATION: the games waiting for a lane (every queue entry: never-started and parked), read under the gate's
        lock; 0 without the gate."""
        gate = ta._priority_gate()
        if gate is None:
            return 0
        import contextlib
        lock = getattr(gate, '_cond', None)
        try:
            with (lock if lock is not None else contextlib.nullcontext()):
                return len(getattr(gate, '_waiting', None) or [])
        except Exception:  # noqa: BLE001
            return 0

    def _v72_time_left(self):
        """v7.2b: the run's time left (s): the gate's shared clock (as the park guard), else the session's timing; None unknown"""
        tl, _el = self._v70_clock()
        if tl is not None:
            return tl
        try:
            r = self._v61_session().timing_payload().get('time_remaining_seconds')
        except Exception:  # noqa: BLE001
            return None
        return None if r is None else float(r)

    def _v72_late(self, kind, level, **info):
        """v7.2b REFRESH_MIN_LEFT: True when less than ARC3_V72_REFRESH_MIN_LEFT seconds of the run are left: this refresh
        trigger ('levelup', 'inlevel', 'budget') is skipped. Every skipped check counts in v72_late_<kind>; the first per kind
        and level is logged (v72-refresh-late) and counts in v72_late_levels_<kind>."""
        tl = self._v72_time_left()
        if tl is None or tl >= V72_RML:
            return False
        stats = self._v6_stats
        stats['v72_late_' + kind] = stats.get('v72_late_' + kind, 0) + 1
        seen = self.__dict__.setdefault('_v72_late_seen', set())
        if (kind, level) not in seen:
            seen.add((kind, level))
            stats['v72_late_levels_' + kind] = stats.get('v72_late_levels_' + kind, 0) + 1
            self._v6_log('v72-refresh-late', dict(kind=kind, level=level, time_left_s=round(tl, 1), min_left_s=V72_RML, **info))
        return True

    def _v72_park_snapshot(self, level, n_levels):
        """v7.2b PARK_RANK: the parked entry with the game's state at the park (the index's inputs: the level's tokens over its
        conversations, its actions as taaf counts them for the official score, the cleared levels; no cap under v7.1)"""
        s = self._v70_state(level, parked=True)
        s.pop('time_left_s'), s.pop('elapsed_s')              # the gate's pump fills them at every admission
        s['actions_on_level'] = int(self._v67_actions_on_level(level, None) or 0)
        pk = self.__dict__.setdefault('_v72_parks_level', {})       # v7.2c: the park count per level (a new level starts at 0)
        pk[level] = int(pk.get(level) or 0) + 1
        return _V72ParkSnapshot(s, self._v70_gid(), level, n_levels, self._v62_swaps.get(level, 0), band_k=pk[level])

    def _v73_once(self, key, level):
        """v7.3: True the first time this (key, conversation, level) is seen (its event is logged once; its checks counted)."""
        seen = self.__dict__.setdefault('_v73_seen', set())
        k = (key, self._v69_ir_conv(), level)
        if k in seen:
            return False
        seen.add(k)
        return True

    def _v73_fire(self, v71, level):
        """v7.3, at a mid-level opener with the tokens trigger due (_v71_due's numbers): True (the notes turn, then the park) while
        ANOTHER game waits (the gate's queue, read under its lock; this game holds its lane, so it is not in it) and at least
        ARC3_V71_GUARD_S seconds of the run are left. Otherwise False: no notes turn and no refresh; the conversation continues and
        the trigger stays armed (checked again at every opener). Receipts: v73_armed / v73_end_cuts once per conversation and level
        (the v73-armed / v73-end-cut events), v73_armed_checks / v73_end_checks per check, v73_armed_fires (an armed conversation
        whose trigger fired later, when a game waited)."""
        stats = self._v6_stats
        tl = v71.get('time_left_s')
        kind = ('end' if (tl is not None and tl < V71_GUARD_S)
                else 'armed' if (not v71.get('waiting') and not V76) else None)   # v7.6 HYBRID: nobody waits: in place
        if kind is None:
            if ('armed', self._v69_ir_conv(), level) in self.__dict__.get('_v73_seen', ()):
                stats['v73_armed_fires'] = stats.get('v73_armed_fires', 0) + 1
            return True
        name = 'v73_end' if kind == 'end' else 'v73_armed'
        stats[name + '_checks'] = stats.get(name + '_checks', 0) + 1
        if self._v73_once(kind, level):
            key = 'v73_end_cuts' if kind == 'end' else 'v73_armed'
            stats[key] = stats.get(key, 0) + 1
            self._v6_log('v73-end-cut' if kind == 'end' else 'v73-armed', dict(level=level, count=v71.get('count'),
                                                                                  waiting=v71.get('waiting'), time_left_s=tl))
        return False

    def _v73_strict(self, level, logical):
        """v7.3 STRICT: the strict in-level refresh at ARC3_V73_STRICT_LOGICAL logical positions: its numbers (the notes turn, then
        the park while another game waits, else the refresh in place), or None in the run's last ARC3_V71_GUARD_S seconds
        (v73_strict_late once per conversation and level with the v73-strict-late event; v73_strict_late_checks per check)."""
        tl, _el = self._v70_clock()
        if tl is not None and tl < V71_GUARD_S:
            stats = self._v6_stats
            stats['v73_strict_late_checks'] = stats.get('v73_strict_late_checks', 0) + 1
            if self._v73_once('strict_late', level):
                stats['v73_strict_late'] = stats.get('v73_strict_late', 0) + 1
                self._v6_log('v73-strict-late', dict(level=level, logical=logical, time_left_s=round(tl, 1)))
            return None
        return dict(logical=logical, threshold=V73_STRICT, waiting=self._v71_waiting(),
                    time_left_s=None if tl is None else round(tl, 1), tokens_on_level=int(self._v61_level_tokens or 0))

    def _v73_keep(self, level, current_frame):
        """v7.3, at the turn start after the tokens trigger's notes turn when nobody took the lane (the waiter left the queue
        meanwhile): no refresh. The conversation continues; its notes stay saved; the next opener says so (_v73_kept_text); a later
        park on this level in this conversation needs no second notes turn (_v73_noted). Receipt v73_kept, the v73-kept event."""
        st = self.__dict__.get('_v69_ir') or {}
        stats = self._v6_stats
        stats['v73_kept'] = stats.get('v73_kept', 0) + 1
        self._v6_log('v73-kept', dict(level=level, step=getattr(current_frame, 'step', None),
                                      tokens_on_level=int(self._v61_level_tokens or 0), waiting_at_trigger=st.get('waiting'),
                                      notes_saved=bool(getattr(self, '_v62_attempt_saved', False))))
        self._v62_attempt_request, self._v62_attempt_saved = None, False
        self._v62_summary_state, self._v62_restart_reason = None, None
        self._v73_noted = (self._v69_ir_conv(), level, st.get('step'))   # the step at which those notes were written
        self._v73_kept_line = level
        self._v69_ir = {}

    def _v73_kept_text(self, text):
        """v7.3: the line after a kept conversation's notes turn, once (the next opener)."""
        lv = self.__dict__.pop('_v73_kept_line', None)
        if lv is None:
            return text
        return text.rstrip() + '\n' + _fill(TEXTS['v73_kept'], level=lv)

    def _v73_opener_receipt(self, kind, content):
        """v7.3: images per opener: v73_openers_<k>, v73_opener_images_<k>, v73_opener_images_max_<k> (k = first, refresh,
        inlevel or restart)."""
        k = kind if kind == 'first' else ((self.__dict__.get('_v67_takeover') or {}).get('kind') or 'restart')
        k = k if k in ('first', 'refresh', 'inlevel') else 'restart'
        n = sum(1 for x in content if isinstance(x, dict) and x.get('type') == 'image_url')
        stats = self._v6_stats
        stats['v73_openers_' + k] = stats.get('v73_openers_' + k, 0) + 1
        stats['v73_opener_images_' + k] = stats.get('v73_opener_images_' + k, 0) + n
        stats['v73_opener_images_max_' + k] = max(stats.get('v73_opener_images_max_' + k, 0), n)

    def _v73_clear_parts(self, current_frame, history_entries):
        """v7.3 CLEAR_BOARDS (ARC3_V73_CLEAR_BOARDS = N >= 3; the user 10-22): v6.9's clearing-step block (its receipts, its event,
        its fallback and missing forms unchanged) with, when it shows the pair (the board the clearing action ran from, the first
        board it produced), boards of the cleared level's final attempt between its line and the pair: the attempt from the level's
        first board, or the board after its latest RESET or automatic restart, to the clearing action; its distinct boards in step
        order (the interior key) other than the pair, at most N - 2 (its starting board always, evenly spaced ones), each captioned
        with its step and the action that produced it; the line then names them. Receipts v73_clear_shown, v73_clear_boards; the
        v73-clear event."""
        parts = self._v69_clear_step_parts(current_frame)
        level = int(getattr(current_frame, 'level', 0) or 0)
        game = self._v61_session()
        rec = ((getattr(game, '__dict__', None) or {}).get('v69_clears') or {}).get(level - 1)
        if len(parts) != 5 or not rec or rec.get('after') is None:
            return parts                                       # v6.9's fallback (no after board) or nothing
        entries = list(history_entries or [])
        ci = None
        for i in range(len(entries) - 1, 0, -1):
            r = getattr(entries[i], 'result', None) or {}
            if r.get('level_completed') and getattr(getattr(entries[i], 'frame', None), 'level', None) == level:
                ci = i
                break
        if ci is None:
            return parts
        lo = 0
        for i in range(ci - 1, 0, -1):
            if (getattr(entries[i], 'result', None) or {}).get('level_completed'):
                lo = i
                break
        a0, how = lo, 'level'
        for i in range(ci - 1, lo, -1):
            r = getattr(entries[i], 'result', None) or {}
            if r.get('automatic'):
                a0, how = i, 'auto'
                break
            if str(r.get('action_name') or '') == 'RESET' or str(getattr(entries[i], 'action', '') or '').startswith('RESET'):
                a0, how = i, 'reset'
                break
        k = level - 1
        seen = {_v69_interior_key(rec['before']), _v69_interior_key(rec['after'])}
        distinct = []
        for i in range(a0, ci):
            fr = getattr(entries[i], 'frame', None)
            if fr is None or getattr(fr, 'level', None) != k:
                continue
            key = _v69_interior_key(getattr(fr, 'grid', None) or ())
            if key not in seen:
                seen.add(key)
                distinct.append(i)
        m, n = V73_CLEAR_BOARDS - 2, len(distinct)
        if n <= m:
            pick = list(distinct)
        elif m == 1:
            pick = distinct[:1]
        else:
            pick = [distinct[round(j * (n - 1) / (m - 1))] for j in range(m)]
        if not pick:
            return parts
        hows = {'level': TEXTS['v73_how_level'], 'reset': TEXTS['v73_how_reset'], 'auto': TEXTS['v73_how_auto']}
        boards, steps = [], []
        for i in pick:
            fr = entries[i].frame
            img = current_grid_image_part(fr)
            if img is None:
                continue
            cap = (_fill(TEXTS['v73_clear_start'], k=k, step=getattr(fr, 'step', '?'), how=hows[how]) if i == a0 else
                   _fill(TEXTS['v73_clear_board'], k=k, step=getattr(fr, 'step', '?'), action=str(getattr(entries[i], 'action', '') or '?')))
            boards += [dict(type='text', text=cap), img]
            steps.append(getattr(fr, 'step', None))
        if not steps:
            return parts
        act = str(rec['action'])
        sel = (_fill(TEXTS['v73_clear_sel_all'], d=n, ds='' if n == 1 else 's', action=act) if len(steps) == n else
               _fill(TEXTS['v73_clear_sel'], n=len(steps), d=n, action=act))
        line = _fill(TEXTS['v73_clear_line'], k=k, s=int(rec['step']), action=act, sel=sel,
                     ch='changed ' if rec.get('index') != 0 else '')
        stats = self._v6_stats
        stats['v73_clear_shown'] = stats.get('v73_clear_shown', 0) + 1
        stats['v73_clear_boards'] = stats.get('v73_clear_boards', 0) + len(steps)
        self._v6_log('v73-clear', dict(level=level, cleared=k, start_step=getattr(getattr(entries[a0], 'frame', None), 'step', None),
                                       how=how, distinct=n, shown=len(steps), steps=steps))
        return [dict(type='text', text='\n' + line + '\n')] + boards + parts[1:]

    # ----------------------------------------------------------------------------------------------- v7.7
    def _v77_prev_lines(self, k, lead='v77_prev_summary', tag='v77_prev'):
        """v7.7 (B; the user 10-24: "Present previous level's summary if it exists when asking for new summary so it can be
        updated"; "Make sure we handle the cases where summary was not given"): the level-up opener's lines after the request for
        level k+1's summary: level k's saved summary under v77_prev_summary; when level k has none, v77_prev_missing and the latest
        earlier level's summary, if any. [] at the first level-up (k < 1). Receipts <tag>_summaries, <tag>_missing, <tag>_fallback
        (v77_prev_*; the notes turns' v77_notes_prev_* with the lead v77_notes_prev_summary)."""
        if k < 1:
            return []
        stats, out = self._v6_stats, []
        j = k
        if k not in self._v6_rules:
            stats[tag + '_missing'] = stats.get(tag + '_missing', 0) + 1
            out.append(_fill(TEXTS['v77_prev_missing'], level=k))
            older = [x for x in self._v6_rules if x < k]
            if not older:
                return out
            j = max(older)
            stats[tag + '_fallback'] = stats.get(tag + '_fallback', 0) + 1
        stats[tag + '_summaries'] = stats.get(tag + '_summaries', 0) + 1
        out.append(_fill(TEXTS[lead], level=j) + '\n' + _rule_display(j, self._v6_rules[j]))
        return out

    def _v77_notes_prev(self, level):
        """v7.7 (the user 10-24: "maybe we should also do this during a refresh in the middle of a level using last level up
        summary"): the text a notes turn on `level` (the in-level refresh's, a restart's, a loop rung's) appends to its request:
        the last level-up summary (level - 1's) to update, with _v77_prev_lines' missing / fallback handling; '' on level 1."""
        lines = self._v77_prev_lines(int(level) - 1, lead='v77_notes_prev_summary', tag='v77_notes_prev')
        return ('\n' + '\n'.join(lines)) if lines else ''

    @staticmethod
    def _v77_last_step(s, kind, history_entries=None, index=None):
        """v7.7: the step of the action that ended the attempt: its record in the previous call (`kind` 'game_over': the GAME
        OVER action; 'reset': the RESET that ended an attempt), else the step of history_entries[index]; None if unknown."""
        for r in reversed(list((s or {}).get('v61_actions') or [])):
            if ((kind == 'game_over' and r.get('game_over'))
                    or (kind == 'reset' and str(r.get('action_display') or '') == 'RESET' and r.get('v6_attempt'))):
                try:
                    return int(r.get('action_num'))
                except (TypeError, ValueError):
                    break
        if history_entries is not None and index is not None:
            try:
                return int(getattr(getattr(history_entries[index], 'frame', None), 'step', None))
            except (TypeError, ValueError, IndexError):
                return None
        return None

    def _v77_steps_line(self, k, n, b):
        """v7.7 (the user 10-24, lever 5: "for each new attempt, only showing the last attempt's step range"): v77_attempt_steps for
        the attempt k that just ended (n actions, the last at step b): "Attempt k was steps a-b."; '' when unknown. Receipt
        v77_attempt_steps."""
        try:
            k, n, b = int(k), int(n), int(b)
        except (TypeError, ValueError):
            return ''
        if k < 1 or n < 1 or b < n:
            return ''
        a = b - n + 1
        self._v6_stats['v77_attempt_steps'] = self._v6_stats.get('v77_attempt_steps', 0) + 1
        return _fill(TEXTS['v77_attempt_steps'], k=k, steps=('step %d' % a) if a == b else ('steps %d-%d' % (a, b)))

    @staticmethod
    def _v77_snap_pick(entries, a0, current_frame):
        """v7.7 (the user 10-24: "the level's first board and current board, and 8 boards sampled evenly in between"; "I said
        evenly spaced boards"): the entries whose boards the snapshot shows before the current board: the attempt's starting
        board (entry a0, step a), then the boards at the steps a + round(j (b - a) / 9) (half up), j = 1..8, strictly between a and
        the current step b, each step once (fewer when b - a < 9); a step's board is that of the last entry at or before it. A
        board equal to another (or to the current board) is shown all the same: it is the state at that step. [] when the attempt
        has no action yet (b <= a); [a0] when the steps are unknown."""
        a = getattr(getattr(entries[a0], 'frame', None), 'step', None)
        b = getattr(current_frame, 'step', None)
        if not (isinstance(a, int) and isinstance(b, int)):
            return [a0]
        if b <= a:
            return []
        steps = []
        for i in range(a0 + 1, len(entries)):
            s = getattr(getattr(entries[i], 'frame', None), 'step', None)
            if isinstance(s, int):
                steps.append((s, i))
        pick = [a0]
        for t in sorted({a + (2 * j * (b - a) + V77_SNAP) // (2 * V77_SNAP) for j in range(1, V77_SNAP)} - {a, b}):
            best = None
            for s, i in steps:                                  # the last entry at or before step t
                if s > t:
                    break
                best = i
            if best is not None and best not in pick:
                pick.append(best)
        return pick

    def _v77_ilb_parts(self, current_frame, history_entries):
        """v7.7 (the user 10-24, V67_DESIGN 316-318): the block of an in-level refresh opener (also the NO_HARD_RESTART refreshes and
        park returns), all text first, so that the server's text pin and board pin are two runs of parts:
          text (one part): the head (what the boards below show), the attempt's range (its starting board's step and the current
            step), the attempt's last V72_ILB_ACTIONS executed actions with the verdicts the reports stated;
          boards: a snapshot of the CURRENT ATTEMPT (from the level's first board, or the board after the latest RESET or automatic
            restart): its starting board and the boards at up to 8 evenly spaced steps before the current one (_v77_snap_pick),
            each captioned with its step (and the action that produced it); the opener's "Current board." and image follow.
        self._v77_ilb_text = the number of leading text parts (1, or 0 for an empty text), for _v6_opener's split. Receipts as
        v7.2's (v70h_il_openers, v70h_il_k, v70h_il_tokens, v72_ilb_openers, v72_ilb_boards; the v70h-il-opener and v72-ilb
        events, the latter with snapshot True) plus v77_ilb_snapshots."""
        from inference.framework import solver as _sv
        entries = list(history_entries or [])
        level = getattr(current_frame, 'level', None)
        start = 0
        for i in range(len(entries) - 1, 0, -1):
            if (getattr(entries[i], 'result', None) or {}).get('level_completed'):
                start = i
                break
        a0, how = start, 'level'
        for i in range(len(entries) - 1, start, -1):
            r = getattr(entries[i], 'result', None) or {}
            if r.get('automatic'):
                a0, how = i, 'auto'
                break
            if str(r.get('action_name') or '') == 'RESET' or str(getattr(entries[i], 'action', '') or '').startswith('RESET'):
                a0, how = i, 'reset'
                break
        if not entries or getattr(getattr(entries[a0], 'frame', None), 'level', None) != level:
            a0, how = None, None
        hows = {'level': TEXTS['v72_ilb_how_level'], 'reset': TEXTS['v72_ilb_how_reset'], 'auto': TEXTS['v72_ilb_how_auto']}
        start_step = getattr(getattr(entries[a0], 'frame', None), 'step', '?') if a0 is not None else None
        cur = getattr(current_frame, 'step', None)
        seen = {_v69_interior_key(getattr(current_frame, 'grid', None) or ())}
        for i in (range(a0, len(entries)) if a0 is not None else ()):  # the receipt's distinct boards (v7.2's count)
            seen.add(_v69_interior_key(getattr(getattr(entries[i], 'frame', None), 'grid', None) or ()))
        pick = self._v77_snap_pick(entries, a0, current_frame) if a0 is not None else []
        lines, boards, shown = [], [], []
        for i in pick:
            fr = entries[i].frame
            cap = (_fill(TEXTS['v72_ilb_start'], step=start_step, how=hows[how]) if i == a0 else
                   _fill(TEXTS['v72_ilb_board'], step=getattr(fr, 'step', '?'), action=str(getattr(entries[i], 'action', '') or '?')))
            img = current_grid_image_part(fr)
            if img is None:
                continue
            boards += [dict(type='text', text=cap), img]
            shown.append(getattr(fr, 'step', None))
        if a0 is not None:
            if not shown:
                lines += [_fill(TEXTS['v72_ilb_head_none'], level=level),
                          _fill(TEXTS['v72_ilb_start_is_current'], step=start_step, how=hows[how])]
            elif len(shown) == 1:
                lines.append(_fill(TEXTS['v77_ilb_head_start'], level=level))
            else:
                n = len(shown) - 1
                lines.append(_fill(TEXTS['v77_ilb_head'], level=level, n=n, ns='' if n == 1 else 's'))
            if isinstance(start_step, int) and isinstance(cur, int) and cur > start_step:
                k77 = len(self._v61_attempts(self._v61_session(), level)) + 1
                lines.append(_fill(TEXTS['v77_ilb_range'], k=k77, level=level, a=start_step, b=cur))
        acts, total = [], 0
        for i in range(len(entries) - 1, (a0 if a0 is not None else start), -1):
            if len(acts) >= V72_ILB_ACTIONS:
                break
            e = entries[i]
            r = getattr(e, 'result', None) or {}
            step = getattr(getattr(e, 'frame', None), 'step', '?')
            if r.get('automatic'):
                line = _fill(TEXTS['v70h_il_auto'], step=step)
            else:
                verdict = _sv._action_verdict(r).replace('(attempt ? ended here)', '(the attempt ended here)')
                line = _fill(TEXTS['v70h_il_line'], step=step, action=str(getattr(e, 'action', '') or '?'), verdict=verdict)
            acts.append(line)
            total += self._v6_count(line)
        acts.reverse()
        if acts:
            lines.append('\n' + _fill(TEXTS['v72_ilb_actions_head'], k=len(acts), ks='' if len(acts) == 1 else 's') + '\n'
                         + '\n'.join(acts))
        elif a0 is not None:
            lines.append(TEXTS['v72_ilb_no_action'])
        parts = ([dict(type='text', text='\n' + '\n'.join(lines) + '\n')] if lines else []) + boards
        self._v77_ilb_text = 1 if lines else 0
        stats = self._v6_stats
        stats['v70h_il_openers'] = stats.get('v70h_il_openers', 0) + 1
        stats.setdefault('v70h_il_k', []).append(len(acts))
        stats.setdefault('v70h_il_tokens', []).append(total)
        stats['v72_ilb_openers'] = stats.get('v72_ilb_openers', 0) + 1
        stats.setdefault('v72_ilb_boards', []).append(len(shown))
        stats['v77_ilb_snapshots'] = stats.get('v77_ilb_snapshots', 0) + 1
        images = sum(1 for p_ in parts if p_.get('type') == 'image_url')
        self._v6_log('v70h-il-opener', dict(level=level, step=cur, k=len(acts), tokens=total,
                                            on_level=max(0, len(entries) - start - 1), first_step=start_step, images=images))
        self._v6_log('v72-ilb', dict(level=level, step=cur, attempt_start=start_step, how=how, distinct=len(seen) - 1, shown=shown,
                                     pointer=False, images=images, actions=len(acts), snapshot=True))
        return parts

    @staticmethod
    def _v77_split_boundary(content, k):
        """v7.7: the token boundary between the text pin [0, k) and the board pin [k, end): the first board part's leading newlines
        move to the end of the last text part (the rendered text is the same) and the text section ends with a newline (added when
        missing), so the text section tokenizes the same alone (the server's probe) and inside the whole opener."""
        first, last = content[k], content[k - 1]
        if first.get('type') == 'text':
            t = first.get('text') or ''
            s = t.lstrip('\n')
            if len(s) < len(t):
                content[k] = dict(first, text=s)
                content[k - 1] = last = dict(last, text=(last.get('text') or '') + t[:len(t) - len(s)])
        if not (last.get('text') or '').endswith('\n'):
            content[k - 1] = dict(last, text=(last.get('text') or '') + '\n')

    # ----------------------------------------------------------------------------------------------- v7.6 HYBRID
    def _v76_point(self, level, logical, kind, current_frame):
        """v7.6 HYBRID (the user 10-24, V67_DESIGN 305 and 308), at a mid-level opener with the conversation's logical position >=
        ARC3_V74_REBASE_AT, the tokens trigger not due and no rebase pending: the decision. Only at an action boundary (kind 'same'
        or 'restart'): a resumption waits for the next one (v76_point_waits); the game reporting GAME OVER (only RESET executes)
        defers it (v76_point_deferred_game_over). The level's tokens in this conversation (the v7.1 trigger's count) below
        ARC3_V76_REFRESH_AT: 'summary' (this opener becomes the rebase's summary turn; the count is kept); at or above it 'refresh'
        (this opener becomes the in-level refresh's notes turn: park or in place), except in the run's last V76_LATE_S seconds
        (the later of ARC3_V71_GUARD_S and ARC3_V72_REFRESH_MIN_LEFT), where no refresh can come: 'summary' (why late); fewer than
        ARC3_V76_REBASE_MIN_LEFT seconds left: 'too_late' (no rebase). Receipts v76_points_<decision> (summary and refresh every
        time; too_late once per conversation, level and summary count) with the v76-point event."""
        stats = self._v6_stats
        if kind not in ('same', 'restart'):
            stats['v76_point_waits'] = stats.get('v76_point_waits', 0) + 1
            return None
        if _v62_game_over_now(self._v61_session()):     # only RESET executes: deferred, as the in-level notes turn (IE C4)
            stats['v76_point_deferred_game_over'] = stats.get('v76_point_deferred_game_over', 0) + 1
            return None
        tl = self._v72_time_left()
        tokens = int(self._v61_level_tokens or 0)
        if tl is not None and tl < V76_MIN_LEFT:
            decision, why = 'too_late', 'min_left'
        elif tokens < V76_REFRESH_AT:
            decision, why = 'summary', 'below'
        elif tl is not None and tl < V76_LATE_S:
            decision, why = 'summary', 'late'
        else:
            decision, why = 'refresh', 'above'
        seen = self.__dict__.setdefault('_v76_seen', set())
        key = (decision, self._v69_ir_conv(), level, int(stats.get('v76_summary_turns', 0)))
        if decision != 'too_late' or key not in seen:
            seen.add(key)
            stats['v76_points_' + decision] = stats.get('v76_points_' + decision, 0) + 1
            self._v6_log('v76-point', dict(level=level, step=getattr(current_frame, 'step', None), logical=logical, threshold=V74_AT,
                                           tokens_on_level=tokens, refresh_at=V76_REFRESH_AT, decision=decision, why=why,
                                           time_left_s=None if tl is None else round(tl, 1)))
        return decision

    def _v76_refresh_numbers(self, level, logical):
        """v7.6 HYBRID: the in-level refresh trigger's numbers (the v69_ir state's v76 field and the notes-turn event's)."""
        tl = self._v72_time_left()
        return dict(logical=logical, threshold=V74_AT, refresh_at=V76_REFRESH_AT, tokens_on_level=int(self._v61_level_tokens or 0),
                    waiting=self._v71_waiting(), time_left_s=None if tl is None else round(tl, 1))

    def _v76_summary(self, text, level, logical, step_env, current_frame):
        """v7.6 HYBRID: this opener becomes the rebase's summary turn: the request appended (v76_summary_request with the server's
        frame size), actions refused for the turn (v76_summary_refusal; the refusal reason v76_summary_reason), no retry and nothing
        stored (the summary stays in the conversation's newest part, which the rebased frame keeps); the next turn start flags that
        turn's first append for the rebase (_v76_turn). Returns (text, True): the turn ends after the reply, as a notes turn."""
        stats = self._v6_stats
        stats['v76_summary_turns'] = stats.get('v76_summary_turns', 0) + 1
        self._v76 = dict(phase='summary', level=level, conv=self._v69_ir_conv(), logical=logical,
                         step=getattr(current_frame, 'step', None), tokens=int(self._v61_level_tokens or 0),
                         noact=int(self._v62_noact_tokens or 0), noact_warned=self._v62_noact_warned,
                         gen=int(self.__dict__.get('_v74_gen') or 0))
        self._v62_summary_state = 'requested'
        self._v64_notes_retry = None                     # as every notes turn's start (IE C1)
        self._step_env_callback = _NoActions(step_env, 'v76_summary_refusal')
        req = _fill(TEXTS['v76_summary_request'], frame=V76_FRAME_TEXT)
        return text.replace(TEXTS['reminder'], '').rstrip() + '\n' + req, True

    def _v76_turn(self, level_now):
        """v7.6 HYBRID, at a turn start (after the level-change bookkeeping, before the gates): after the rebase's summary turn (the
        same conversation and level, no session error) this turn's first append rebases (_v74_inplace without the reset: the level's
        count is kept) and its opener is the normal next turn (the pending tool results, then the board); the notes-turn state the
        summary turn left ('done') is cleared and the no-action count restored (the summary is not charged to the no-action budget).
        Otherwise the summary is dropped (v76_summary_dropped). Receipts v76_summaries, v76_summary_tokens (the summary turn's
        generated tokens), the v76-summary / v76-summary-dropped event."""
        stats = self._v6_stats
        conv = self._v69_ir_conv()
        s = self.__dict__.pop('_v76', None)
        if s is not None and s.get('phase') == 'summary':
            error = bool(getattr(self, '_v62_force_restart', False))
            if s.get('conv') == conv and s.get('level') == level_now and not error:
                gen = int(self.__dict__.get('_v74_gen') or 0) - int(s.get('gen') or 0)
                self._v62_summary_state = None
                self._v62_noact_tokens, self._v62_noact_warned = s.get('noact', 0), s.get('noact_warned', 0)
                self._v74_inplace(level_now, reset=False, reason='point')
                stats['v76_summaries'] = stats.get('v76_summaries', 0) + 1
                stats.setdefault('v76_summary_tokens', []).append(gen)
                self._v6_log('v76-summary', dict(level=level_now, step=s.get('step'), logical=s.get('logical'),
                                                 tokens_on_level=s.get('tokens'), summary_tokens=gen))
            else:
                stats['v76_summary_dropped'] = stats.get('v76_summary_dropped', 0) + 1
                self._v6_log('v76-summary-dropped', dict(level=s.get('level'), level_now=level_now, error=error,
                                                         same_conversation=s.get('conv') == conv))

    # ------------------------------------------------------------------------------------------- v7.4 REBASE (apply_v74.py)
    def _v74_point(self, level, logical):
        """v7.4, at a mid-level opener with the conversation's logical position >= ARC3_V74_REBASE_AT and the tokens trigger not due,
        the user's rule (10-22): the level's tokens in this conversation <= ARC3_V74_RESET_MAX: 'inplace' (this opener's append
        rebases, the count resets); above it 'park' (the notes-free park at the next turn start; its return rebases), except in
        the run's last ARC3_V71_GUARD_S seconds (no parks): 'inplace' without the reset. Receipts v74_points_inplace_reset /
        _inplace_late / _park and the v74-point event (at every opener that decides; a deferred park decides again)."""
        tokens = int(self._v61_level_tokens or 0)
        tl, _el = self._v70_clock()
        late = tl is not None and tl < V71_GUARD_S
        decision = 'inplace_reset' if tokens <= V74_RESET_MAX else 'inplace_late' if late else 'park'
        stats = self._v6_stats
        stats['v74_points_' + decision] = stats.get('v74_points_' + decision, 0) + 1
        self._v6_log('v74-point', dict(level=level, logical=logical, threshold=V74_AT, tokens_on_level=tokens, reset_max=V74_RESET_MAX,
                                       decision=decision, time_left_s=None if tl is None else round(tl, 1),
                                       waiting=self._v71_waiting()))
        if decision == 'park':
            return 'park'
        self._v74_inplace(level, reset=decision == 'inplace_reset', reason='point')
        return 'inplace'

    def _v74_inplace(self, level, reset, reason):
        """v7.4: this turn's first append carries the rebase flag; reset: the stint's count starts again (the v7.1 trigger's)."""
        self._v74_rebase_next = dict(reason=reason, reset=reset, level=level)
        if reset:
            self._v74_reset(level, noact=False)
        self._v74_watch = dict(kind=reason, level=level, gen=int(self.__dict__.get('_v74_gen') or 0))

    def _v74_reset(self, level, noact):
        """v7.4: the stint's counts start again: the level's tokens in this conversation (the v7.1 trigger's count) and the v7.3
        kept-notes states; at a return also the no-action budget (a new stint, as the refresh arm's new conversation)."""
        self._v61_level_tokens = 0
        self.__dict__.pop('_v73_noted', None)
        self.__dict__.pop('_v73_kept_line', None)
        if noact:
            self._v62_noact_tokens, self._v62_noact_warned = 0, 0
        self._v6_stats['v74_resets'] = self._v6_stats.get('v74_resets', 0) + 1

    def _v74_suspend(self):
        """v7.4, at a park: the server suspends the session (its engine request and KV freed; the token history kept). Any failure
        closes the session; the return then falls back to v7.3's in-level refresh (v74_suspend_failures, v74-suspend-failed)."""
        stats = self._v6_stats
        try:
            rec = self._v6_session.suspend()
            stats['v74_suspends'] = stats.get('v74_suspends', 0) + 1
            self._v74_parked_at = time.time()
            self._v6_log('v74-suspend', dict(rec or {}))
        except Exception as exc:  # noqa: BLE001
            stats['v74_suspend_failures'] = stats.get('v74_suspend_failures', 0) + 1
            self._v6_log('v74-suspend-failed', dict(error='%s: %s' % (type(exc).__name__, str(exc)[:300])))
            self._v74_suspend_failed = True
            try:
                self._v6_session.close()
            except Exception:  # noqa: BLE001
                pass

    def _v74_return(self, state_path, current_frame, history_entries, st, parked):
        """v7.4, at the turn start where v7.3 would refresh (the in-level gate 'now'); no new conversation:
          parked: the return; the suspended session continues: the stint's counts reset (_v74_reset with the no-action budget) and this
            turn's first append rebases; the opener is the normal next turn (the pending tool results, then the opener). After a
            failed suspend (the session closed): v7.3's in-level refresh opens a new conversation (v74_refresh_fallbacks).
          not parked, the rebase point's park (no game waits, no gate): the in-place rebase with the reset; the time guard: without.
          not parked, the tokens trigger: the conversation continues, the trigger armed (no kept line: no notes turn ran).
        Receipts v74_returns, v74_unparked_<trigger>, the v74-return event; the tokens to the first executed action follow."""
        level = getattr(current_frame, 'level', None)
        trig = st.get('trigger') or 'tokens'
        stats = self._v6_stats
        if parked and self.__dict__.pop('_v74_suspend_failed', False):
            stats['v74_refresh_fallbacks'] = stats.get('v74_refresh_fallbacks', 0) + 1
            return self._v69_ir_refresh(state_path, current_frame, history_entries)
        self._v69_ir = {}
        if parked:
            stats['v74_returns'] = stats.get('v74_returns', 0) + 1
            at = self.__dict__.pop('_v74_parked_at', None)
            self._v74_reset(level, noact=True)
            self._v74_rebase_next = dict(reason='return', reset=True, level=level, trigger=trig)
            self._v74_watch = dict(kind='return', level=level, gen=int(self.__dict__.get('_v74_gen') or 0))
            self._v6_log('v74-return', dict(level=level, trigger=trig, parked=True,
                                            parked_s=None if at is None else round(time.time() - at, 1)))
            return load_runtime_state(state_path)
        key = 'v74_unparked_' + str(trig)
        stats[key] = stats.get(key, 0) + 1
        late = None
        if trig == 'rebase_park':
            tl, _el = self._v70_clock()
            late = tl is not None and tl < V71_GUARD_S
            self._v74_inplace(level, reset=not late, reason='unparked')
        self._v6_log('v74-return', dict(level=level, trigger=trig, parked=False, inplace=trig == 'rebase_park', late=late))
        return current_frame, history_entries

    def _v74_after(self, result, flag):
        """v7.4: the generated-token count (the first-action receipt) and the server's rebase receipt of this request (v74_rebases,
        v74_rebases_<reason | server>, v74_rebase_to_tokens, v74-rebase); a flagged request without one: v74_rebase_missing."""
        stats = self._v6_stats
        self._v74_gen = int(self.__dict__.get('_v74_gen') or 0) + int((result.usage or {}).get('completion_tokens') or 0)
        rb = (getattr(result, 'persistent', None) or {}).get('rebase')
        if isinstance(rb, dict):
            reason = flag['reason'] if flag else 'server'
            stats['v74_rebases'] = stats.get('v74_rebases', 0) + 1
            stats['v74_rebases_' + reason] = stats.get('v74_rebases_' + reason, 0) + 1
            stats.setdefault('v74_rebase_to_tokens', []).append(rb.get('to_logical_tokens'))
            bnd = str(rb.get('boundary'))
            stats['v74_rebase_boundary_' + bnd] = stats.get('v74_rebase_boundary_' + bnd, 0) + 1
            w = self.__dict__.get('_v74_watch')
            if w is not None and bnd == 'pin_keep_first':        # IE 10-23: first-action tokens after keep-first frames apart
                w['keep_first'] = True
            self._v6_log('v74-rebase', dict(reason=reason, reset=flag.get('reset') if flag else None,
                                            dropped_observations=len(rb.get('dropped_observation_ids') or []),
                                            **{k: rb.get(k) for k in ('from_logical_tokens', 'to_logical_tokens', 'prefix_tokens',
                                                                      'window_start', 'boundary', 'images', 'restart_seconds')}))
        elif flag:
            stats['v74_rebase_missing'] = stats.get('v74_rebase_missing', 0) + 1
            self._v6_log('v74-rebase-missing', dict(flag))
        n = int(getattr(self._v6_session, 'resume_retries', 0) or 0)   # the transport's 429 retries of a resume (IE 10-23)
        if n:
            stats['v74_resume_retries'] = stats.get('v74_resume_retries', 0) + n
            self._v6_session.resume_retries = 0

    def _v74_first_action(self):
        """v7.4 receipt: the tokens generated from a return or an in-place rebase to the next executed action
        (v74_tokens_to_action_<point | return | unparked>, and _keep_first after a pin keep-first frame; the v74-first-action
        event)."""
        w = self.__dict__.pop('_v74_watch', None) or {}
        tokens = int(self.__dict__.get('_v74_gen') or 0) - int(w.get('gen') or 0)
        self._v6_stats.setdefault('v74_tokens_to_action_' + str(w.get('kind')), []).append(tokens)
        if w.get('keep_first'):
            self._v6_stats.setdefault('v74_tokens_to_action_keep_first', []).append(tokens)
        self._v6_log('v74-first-action', dict(kind=w.get('kind'), level=w.get('level'), tokens=tokens,
                                              keep_first=bool(w.get('keep_first'))))

    def _v71_park(self, current_frame):
        """v7.1 ALLOCATION, at the turn start after an in-level notes turn (the trigger 'tokens' or a NO_HARD_RESTART trigger): when
        a game waits for a lane and at least ARC3_V71_GUARD_S seconds of the run are left, give up the lane (the session closed,
        its KV freed) and queue in the parked band (_V62Snapshot.parked: after every never-started game; FIFO among parked games);
        True once the game holds a lane again (the in-level refresh then opens the new conversation). Otherwise the lane is kept:
        False (park_skipped; why no_gate | no_game_waits | time_guard). The v69-park event as the rung's (records fingerprinted
        before the wait, survived after it) with the trigger, the waiting count, the time left and the count."""
        st = self.__dict__.get('_v69_ir') or {}
        level = getattr(current_frame, 'level', None) or 1
        gate = ta._priority_gate()
        waiting = self._v71_waiting()
        tl, el = self._v70_clock()
        stats = self._v6_stats
        trig = st.get('trigger') or 'tokens'
        base = dict(level=level, step=getattr(current_frame, 'step', None), trigger=trig,
                    count=(st.get('v71') or {}).get('count', int(st.get('tokens') or 0)), waiting_at_trigger=st.get('waiting'),
                    waiting=waiting, time_left_s=None if tl is None else round(tl, 1))
        why = ('no_gate' if gate is None else 'no_game_waits' if not waiting
               else 'time_guard' if (tl is not None and tl < V71_GUARD_S) else None)
        if why:
            st['parked'] = False
            stats['park_skipped'] = stats.get('park_skipped', 0) + 1
            stats['v71_park_skipped_' + why] = stats.get('v71_park_skipped_' + why, 0) + 1
            self._v6_log('v69-park', dict(base, parked=False, why=why))
            return False
        before = self._v69_pk_fingerprint()
        game = self._v61_session()
        n_levels = getattr(getattr(getattr(game, 'game', None), 'game_run', None), 'number_of_levels', None)
        if V74 and not V76 and V75_POLISH and getattr(self, '_v62_force_restart', False):   # v7.5 RUN-1: a session that has just
            stats['v75_suspend_skipped_after_error'] = stats.get('v75_suspend_skipped_after_error', 0) + 1   # closed, not suspended
            self._v6_log('v75-suspend-skipped', dict(level=level))                   # (f74: every failed suspend was such a 409)
            self._v74_suspend_failed = True          # the return falls back to v7.3's in-level refresh, as after a failed suspend
            try:
                self._v6_session.close()
            except Exception:  # noqa: BLE001
                pass
        elif V74 and not V76:                        # v7.4: suspended (the KV freed; the return rebases the same session)
            self._v74_suspend()
        else:
            try:
                self._v6_session.close()                      # free the KV while waiting; the in-level refresh opens a new one
            except Exception:  # noqa: BLE001
                pass
        if V72_PARKX:                                # v7.2b PARK_RANK: the entry carries the game's state
            snapshot = self._v72_park_snapshot(level, n_levels)
            priority = snapshot.v72_prio
            v72_at_park = snapshot.v72_score(gate, time.monotonic())
            ret0 = self.__dict__.pop('_v72_ret', None)       # a return without a clear: its exposure ends at this park
            if ret0 is not None:
                stats['v72_return_ends'] = stats.get('v72_return_ends', 0) + 1
                self._v6_log('v72-return-end', dict(ret0, why='parked_again', seconds=round(time.time() - ret0['epoch'], 1)))
        else:
            snapshot = _V62Snapshot(level, n_levels, self._v62_swaps.get(level, 0))
            snapshot.parked = True
            priority = snapshot.priority(time.monotonic(), getattr(gate, '_clock_deadline', None))
        t0 = time.time()
        gate.handover(priority, snapshot)
        waited = time.time() - t0
        after = self._v69_pk_fingerprint()
        st['parked'] = True
        stats['parks'] = stats.get('parks', 0) + 1
        stats['v71_parks'] = stats.get('v71_parks', 0) + 1
        stats['park_wait_s'] = round(stats.get('park_wait_s', 0) + waited, 1)
        stats['v71_park_wait_s'] = round(stats.get('v71_park_wait_s', 0) + waited, 1)
        stats.setdefault('park_levels', []).append(level)
        if after != before:
            stats['park_records_changed'] = stats.get('park_records_changed', 0) + 1
        v72 = {}
        if V72_PARKX:                                # v7.2b PARK_RANK: the scores at the park and the admission
            adm = snapshot.v72_admit or {}
            pk = adm.get('picker') or 'none'
            stats['v72_rank_admits_' + pk] = stats.get('v72_rank_admits_' + pk, 0) + 1
            if adm.get('newest') and float(adm.get('waited_s') or 0) < 1.0:
                stats['v72_rank_immediate'] = stats.get('v72_rank_immediate', 0) + 1
            v72 = dict(v72=dict(rv_park=v72_at_park[0], ix_park=v72_at_park[1], state=dict(snapshot.state), admit=adm or None,
                                **({'band_k': snapshot.band_k} if V72_BANDS else {})))
            self._v72_ret = dict(n=adm.get('n'), picker=adm.get('picker'), level=level, rv=adm.get('rv'), ix=adm.get('ix'),
                                 tokens=0, epoch=round(time.time(), 1),   # the return's outcome (v72-return-clear)
                                 **({'band': snapshot.band_k} if V72_BANDS else {}))
        self._v6_log('v69-park', dict(base, parked=True, priority=priority, waited_s=round(waited, 1), records=before,
                                      survived=after == before, changed=sorted(k for k in before if before[k] != after.get(k)),
                                      **v72))
        return True

    def _v70_ir_fields(self, level, st):
        """the v69-inlevel-refresh event's park fields: the rung's (PARK69), with v7.0 the trigger (logical, park, index, cap
        or a NO_HARD_RESTART trigger) and the v7.0 decision"""
        f = self._v69_pk_refreshed(level, st) if PARK69 else {}
        if V70_ON:
            f = dict(f, trigger=st.get('trigger') or f.get('trigger') or 'logical', parked=bool(st.get('parked')),
                     **({'v70': st.get('v70')} if st.get('v70') else {}))
        return f

    def _v70_nhr_refresh(self, state_path, current_frame, history_entries, forced, position_restart, budget_hit, noact_hit,
                         should_stop=None):
        """v7.0 NO_HARD_RESTART: the harness restart due at this turn start (after its notes turn, or a session error without
        one) becomes an in-level refresh at the current board (_v69_ir_refresh: a new conversation; no RESET, no swap-out, no
        archive, nothing hidden) with the trigger's name; as after today's restart, the level's cross-conversation count and
        the rung start again (RESTART_ACROSS / PARK69). Receipts: nhr_refreshes, nhr_refreshes_<trigger>, noact_refreshes;
        the loop's recurrence (loop refreshes per level, the next conversation's loop episodes / rung-1 blocks / refusals,
        nhr_loop_again); the v70-nhr event."""
        reason = self._v62_restart_reason if (not forced and self._v62_restart_reason) else (
            'position' if position_restart else 'tokens' if budget_hit else 'noact' if noact_hit else 'forced')
        level = getattr(current_frame, 'level', None)
        pending = self.__dict__.get('_v67_pending') or {}
        loop = dict(pending.get('loop') or {}) if reason == 'loop' else None
        receipt = getattr(getattr(self, '_v6_session', None), 'last_receipt', None) or {}
        mark = self.__dict__.get('_v70_notes_mark') or (None, 0, 0)
        notes = self._v62_summary_state == 'done'
        noact_tokens = int(self._v62_noact_tokens or 0)
        self._v69_ir = dict(pending='notes' if notes else 'now', level=level, step=getattr(current_frame, 'step', None),
                            logical=int(receipt.get('logical_tokens') or 0),
                            tokens=int(mark[2]) if mark[0] == level else int(self._v61_level_tokens or 0),
                            conv=self._v69_ir_conv(), notes_before=int(mark[1]) if mark[0] == level else 0, kind='nhr',
                            trigger=reason)
        if V71:                                                 # v7.1 ALLOCATION: every in-level refresh trigger parks while a
            self._v69_ir['park'] = True                         # game waits; the run's end while parked: no new conversation
            if self._v71_park(current_frame):
                self._v6_stats['v71_nhr_parks'] = self._v6_stats.get('v71_nhr_parks', 0) + 1
                if should_stop is not None and should_stop():
                    self._v71_stopped = True
                    return current_frame, history_entries
        current_frame, history_entries = self._v69_ir_refresh(state_path, current_frame, history_entries)
        if RESTART_ACROSS69 or PARK69:                         # as after a harness restart: the level's count and the rung
            self._v69_att = 0
            self.__dict__.pop('_v69_pk', None)
        info = self.__dict__.get('_v67_takeover') or {}
        info['trigger'] = reason
        if loop:
            info['loop'] = dict(loop)
        stats = self._v6_stats
        stats['nhr_refreshes'] = stats.get('nhr_refreshes', 0) + 1
        stats['nhr_refreshes_' + reason] = stats.get('nhr_refreshes_' + reason, 0) + 1
        if reason == 'noact':
            stats['noact_refreshes'] = stats.get('noact_refreshes', 0) + 1
        game = self._v61_session()
        on_level = None
        if reason == 'loop' and game is not None:
            per = game.__dict__.setdefault('v70_loop_refreshes', {})
            per[level] = on_level = int(per.get(level) or 0) + 1
            if on_level >= 2:
                stats['nhr_loop_again'] = stats.get('nhr_loop_again', 0) + 1
            stats.setdefault('nhr_after_loop', []).append(dict(level=level, n_on_level=on_level, loop_episodes=0,
                                                               loop_rung1=0, loop_refused=0))
            self._v70_after_loop = dict(level=level, conv=self._v69_ir_conv(),
                                        base={k: int(stats.get(k) or 0) for k in ('loop_episodes', 'loop_rung1', 'loop_refused')})
        self._v6_log('v70-nhr', dict(level=level, trigger=reason, notes=notes, noact_tokens=noact_tokens,
                                     logical=int(receipt.get('logical_tokens') or 0), loop=loop, loop_refreshes_on_level=on_level,
                                     refreshes=stats['nhr_refreshes']))
        return current_frame, history_entries

    def _v62_archive_level_programs(self, level):
        """The restart's library step: programs written on the restarted level are archived; round 5d (C50 g016, g020,
        g022, g042): one that redefined an earlier level's program returns to that earlier version instead (its
        earlier-level version was lost, and programs depending on it could no longer be saved). Returns (archived,
        restored) names."""
        archived, restored = [], []
        pre_level = self.__dict__.get('_v62_pre_level', {})
        log = self.__dict__.get('_v62_library_log') or []
        for name in list(self._v6_library_order):
            if self._v61_library_levels.get(name) == level and name in self._kept_functions:
                # round 5l (item 110, S4; 392b g515: a redefined `show` archived instead of restored): the latest version
                # saved on an earlier level, from the append-only log; the round-5d record is the fallback
                earlier = next((e for e in reversed(log) if e.get('name') == name and e.get('status') == 'saved'
                                and isinstance(e.get('level'), int) and e['level'] < level), None)
                old = (earlier['level'], earlier['source'], earlier.get('step')) if earlier else pre_level.pop(name, None)
                pre_level.pop(name, None)
                if old is not None and old[0] is not None and old[0] < level:
                    # round 5d: an earlier level's program redefined on the restarted level returns to its earlier version
                    self._kept_functions[name] = old[1]
                    self._v61_library_levels[name] = old[0]
                    self._v6_library_steps[name] = old[2]
                    restored.append(name)
                    log.append(dict(name=name, level=old[0], step=old[2], source=old[1], status='restored'))
                    continue
                src = self._kept_functions.pop(name)
                try:
                    sig = function_signature(src)
                except Exception:  # noqa: BLE001
                    sig = name + '(...)'
                self._v61_archive[name] = dict(signature=sig, doc=_docline(src), level=level,
                                               step=self._v6_library_steps.get(name), source=src)
                self._v6_library_order.remove(name)
                archived.append(name)
                log.append(dict(name=name, level=level, step=self._v6_library_steps.get(name), status='archived'))
        return archived, restored

    def _v61_hide_level_history(self, history_entries, level):
        """Hide this level's entries before the restart's RESET (from the entry after the last level completion).
        Ranges from earlier levels stay hidden, so the history a conversation sees never changes under it."""
        entries = list(history_entries or [])
        reset_i = next((i for i in range(len(entries) - 1, -1, -1)
                        if (getattr(entries[i], 'result', None) or {}).get('action_name') == 'RESET'), None)
        if reset_i is None:
            return None
        lo = 0
        for i in range(reset_i):
            if (getattr(entries[i], 'result', None) or {}).get('level_completed'):
                lo = i + 1
        if reset_i <= lo:
            return None
        self._v61_hidden = [r for r in self._v61_hidden if r[1] <= lo] + [(lo, reset_i)]
        self._v61_hidden_level = level
        return [lo, reset_i]

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
        # v6.9 CLEAR_STEP: board_resident is the server's state of the observation's FIRST observation image; with the clearing-step block
        # that is the clear's before board in restart / refresh openers, so CLEAR_STEP refuses to start with the recap on (agent knobs)
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

    def _v6_capture_rule(self, texts, provisional=False):
        """A4: the lines that state the solved level's rule, and up to RULE_LINES following lines, to a blank. v6.2 fix (run
        A: the exact tag 'Rule L{n}:' in the first message after a level-up matched 21-22 of 108 level-ups): any header line
        with 'Rule'/'Rules' and 'L{n}' before its colon ('Rule L3:', 'Rule L3 summary:', 'Rule L1 (tested):', 'Rule
        summary (L2):', 'Rules established in L1:'), looked for in up to RULE_TURNS completions. Round 3 (A2): up to 40
        characters before 'Rule' (sb26 'Mechanics summary (Rule L7):'), the first of the texts that matches (reply,
        printed strings, reasoning), and 8 following lines. Round 5d (C50 g014, g023): a match whose body is empty or a
        placeholder no longer ends the capture (g014: `print("Rule L1:")` with computed prints after it stored nothing and
        closed the capture before the reasoning's summary was read; g023: a 'Rule L1: ...' stub); the last substantive
        match of the first text that has one is kept."""
        n = self._v6_rule_capture
        if n in getattr(self, '_v62_rule_fn', ()):          # round 5l (item 108): save_summary stored it
            self._v6_rule_capture = None
            return
        # round 4 (A3 sb26, sc25: 45 and 53 characters between "L5" and the colon; cn04 "Rule L1-3:"; sb26 "Level 1 rule:")
        pattern = (r'(?im)^[^\n]{0,40}(?:\brules?\b[^\n:]{0,50}\bL\s*(?:\d+\s*[-\u2013]\s*L?\s*)?%d\b[^\n:]{0,80}'
                   r'|\blevel\s*%d\b[^\n:]{0,30}\brules?\b[^\n:]{0,80}):' % (int(n), int(n)))
        # round 5c (B3 ls20: "Rule summary (5 lines):" twice, so L5 and L6 were never shown): an untagged summary header,
        # used only when no text of the completion has the tagged one
        fallback = r'(?im)^[^\n]{0,40}\brules?\b[^\n:]{0,30}\bsummary\b[^\n:]{0,40}:'
        texts = [texts] if isinstance(texts, str) else list(texts)
        # round 5c (B3: the summaries of 7 games ran on into the next level, e.g. "... fill all boxes. Now level 2. The
        # board: ..."): the summary is written when level n+1's board is first shown, so text about level n+1 is planning;
        # the summary ends at the sentence that names it
        nxt = re.compile(r'(?i)\blevel\s*%d\b' % (int(n) + 1))
        # round 5g: in a list, only the next level named at the start of a sentence ends an item (ka59 run-on)
        nxt_item = re.compile(r'(?i)(?:^|(?<=[.!?)])\s+)(?:now\s+)?(?:on\s+)?level\s*%d\b' % (int(n) + 1))
        rule, source = '', None
        for pat in (pattern, fallback):
            for i, text in enumerate(texts):
                found = [r for r in (_rule_body(text or '', m, nxt, nxt_item) for m in re.finditer(pat, text or '')
                                     if _header_ok(m.group(0), text or '', m.start())) if r]
                found = [r for r in (_strip_level_tag(r, n) for r in found) if r]
                if found:
                    rule, source = _merge_rule_lines(found), i
                    break
            if rule:
                break
        # round 5l (A2, 392b g524, g511, g500: a draft in the reasoning was kept and the reply's summary a few turns later
        # ignored): a summary found only in the reasoning is stored provisionally; the window stays open for one in a reply
        reasoning_only = bool(rule) and source == 2
        if not rule or reasoning_only:
            if not provisional:
                self._v62_rule_turns -= 1
                if self._v62_rule_turns <= 0:
                    self._v6_rule_capture = None
            if not rule:
                return
        if not provisional and not reasoning_only:   # round 5h (G1 iv): a provisional capture (the clearing turn) keeps
            self._v6_rule_capture = None             # the capture open, so a summary written after the level-up overrides it
        rule = _cap_rule(rule, getattr(self, '_v6_count', None))   # round 5c: at a word, marked; 5g: tokens, lists by item
        if n not in self._v6_rules:
            self._v6_stats['rules_recorded'] = self._v6_stats.get('rules_recorded', 0) + 1
            self._v6_stats['rules_via_capture'] = self._v6_stats.get('rules_via_capture', 0) + 1   # round 5l (item 108)
        if provisional:
            self._v6_stats['rules_from_clearing_turn'] = self._v6_stats.get('rules_from_clearing_turn', 0) + 1
        self._v6_rules[n] = rule
        self._v6_rules_dirty = True

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
