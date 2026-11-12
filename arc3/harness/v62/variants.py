"""Harness v6 text variants (spec §7, rewritten for coherence with dfranzen's core; user 2026-10-03: adopt prompts from
several sources only as one coherent, non-redundant text). Selected by environment, so two-arm screens differ only by
configuration:
  ARC3_V6_A13=1   A1 + A2 (default on since 2026-10-03; ARC3_V6_A13=0 restores the v6.0 texts)
  ARC3_V6_A3=1    A3 on top of A1 + A2 (default off since 2026-10-03 ~16:20, user: "A3 is not needed")
  ARC3_V6_A45=1   A4 + A5 (on top of whatever else is set)
A1 (priors and probing, sirikilohit's Wang/Ludvig section) is NOT added as its own section: its action priors extend
dfranzen's Actions bullets, its object priors and "priors are hypotheses" stance join Reading the board, and its
mechanics-to-goal workflow joins the Objective next to the existing "probe or plan that the evidence favours" line.
A2 renames the death verdict LEVEL RESTARTED everywhere the model reads it (verdict list, per-action verdict, restart
header); the Python flag stays game_over. A3 adds the life-length line to the restart block. A4 asks for the solved
level's rule at level-up and keeps it in a reminder block; A5 joins the level-up block's probing sentence instead of
repeating the objective as a separate line.
Every edit is an exact replacement with an asserted anchor, so a changed base text fails loudly."""
import os


def _sub(text, old, new):
    if text.count(old) != 1:
        raise RuntimeError('variant anchor not found exactly once: %r' % old[:70])
    return text.replace(old, new)


def show_board():
    """Round 5m (item 124, user 10-08): `show_board(cells, label)` in the sandbox and its two lines in the system prompt
    (sandbox_ext.SHOW_BOARD reads the same switch)."""
    return os.environ.get('ARC3_V62_SHOW_BOARD', '1') == '1'


def _r5m_show_board(t):
    """Round 5m (item 124; user 10-08: "We instruct the model not to print the full board. But what if it wants to see the
    simulated effects on the board?"): the no-print rule points to show_board, and the helper list describes it. Facts
    only (what it takes, where the image appears, the limit); no advice on when to use it."""
    s = t['system']
    s = _sub(s, "Print compact results: object lists, diffs, coordinates, counts and small crops.",
             "Print compact results: object lists, diffs, coordinates, counts and small crops. To look at a whole board "
             "your code built, call `show_board(cells)`: it is added to this call's result as an image, like the board "
             "image (at most 4 per call).")
    s = _sub(s, "`connected_components(cells, connectivity=4)`.\n",
             "`connected_components(cells, connectivity=4)`.\n"
             "- `show_board(cells, label=None)` adds a board (a frame, rows of symbols or ASCII text, as `grid_cells` "
             "accepts) to this call's result as an image captioned with `label`; at most 4 per call, and a board already "
             "shown in the call is not repeated.\n")
    t['system'] = s
    t['show_board_caption'] = "[Board image {j}: show_board{label}, {rows}x{cols}.]"
    t['show_board_full'] = "[Board not shown: this call already has {m} board images.]"
    t['show_board_invalid'] = ("[Board not shown: show_board takes rows of equal length, at most 64 x 64, of the board's "
                               "colour symbols.]")
    return t


def a13():
    return os.environ.get('ARC3_V6_A13', '1') == '1'           # default on since 2026-10-03 (user: A1-A3 adopted)


def a3():
    """A3, the restart block's life-length line: off by default (user 2026-10-03: keep A1 and A2, drop A3)."""
    return a13() and os.environ.get('ARC3_V6_A3', '0') == '1'


def a45():
    return os.environ.get('ARC3_V6_A45', '0') == '1'


def reset():
    """Voluntary level RESET (user 14:45): dfranzen's EXPOSE_RESET mechanics (RESET listed in valid actions, first game
    action of a snippet only, not learned as a no-op) plus one Actions bullet adapted from his RESET note."""
    return os.environ.get('EXPOSE_RESET', 'off').strip().lower() in ('on', '1', 'true', 'yes')


def recap():
    """Recap (the old harness's v5.29 feature; user 14:47): re-show the most recent boards that have left the context."""
    return os.environ.get('ARC3_V6_RECAP', '0') == '1'


def lean():
    """B3 (spec §8; user 14:45): openers without the standing reminder. Every sentence of the reminder is already in the
    system prompt (Working in Python, Python runtime, Tool rules), so nothing moves there."""
    return os.environ.get('ARC3_V6_LEAN', '0') == '1'


def r61():
    """v6.1 revision texts (HARNESS_REVISION_ITEMS_2026-10-05.md; user go-ahead 2026-10-05): on in v61, needs A1 + A2."""
    return a13() and os.environ.get('ARC3_V61_TEXTS', '1') == '1'


def restart_on():
    """v6.2 round 3: the level restart (run B) is on; its opener shows the captured summaries."""
    return int(os.environ.get('ARC3_V61_RESTART_TOKENS', '0') or 0) > 0


def summary():
    """v6.1 item 6: the level-completion mechanics summary ("Rule L{k}:") asked for and captured in both arms; shown
    only in a restarted conversation (run B). Replaces A4's request text; A5 and A4's rules reminder are not used."""
    return r61() and os.environ.get('ARC3_V61_SUMMARY', '1') == '1'


CALL_CAP = int(os.environ.get('ARC3_V61_CALL_CAP', '40'))   # v6.1 item 2d: executed actions per python call (0 = off)


def _r61(t):
    """v6.1 text edits. Every edit is an exact replacement with an asserted anchor (base = v6 + A1 + A2 [+ A3])."""
    s = t['system']
    # item 13 (method priors) - objective: a new element brings a new mechanic
    s = _sub(s, "New levels often add mechanics, sometimes through unfamiliar board elements, and those are often what the "
                "level needs.",
             "New levels often add mechanics, sometimes through unfamiliar board elements, and those are often what the "
             "level needs: probe a new kind of object or colour directly, with each available action, before planning "
             "around it.")
    # items 3 and 10 - what an opener reports
    s = _sub(s, "- A turn begins with a message that reports what your previous actions did, the current step and level, "
                "and the actions that are valid now.",
             "- A turn begins with a message that reports what your previous actions did, the current step, level and "
             "attempt, the actions that are valid now, and the time left in the whole run.")
    # item 13 - shape, size and orientation; self-moving things
    s = _sub(s, "- Some games are logic or layout puzzles",
             "- Compare an object's size, shape and orientation, not only its position: actions can grow, shrink, rotate "
             "or recolour things, and repeating an action can undo or cycle its effect. If something moves by itself, "
             "record its positions over several actions before planning around it.\n- Some games are logic or layout puzzles")
    # item 3 - the edge bar: facts only (no cap is known to the harness)
    s = _sub(s, "A long line or a segmented strip along an edge is usually a step or time budget that changes as you act.",
             "A long line or a segmented strip along an edge is usually a step budget that changes as you act. It may "
             "fill or shrink; its length and rate can differ between levels; whether no-ops count depends on the game; "
             "and reaching its end usually, but not always, ends the attempt.")
    # item 5 - MOUSE coordinates are frame pixels; results name the hit
    s = _sub(s, "Pass integer `row` and `col` from 0 to 63, zero-based from the top-left: row increases downward and col "
                "increases to the right.",
             "Pass integer `row` and `col` from 0 to 63, zero-based from the top-left: row increases downward and col "
             "increases to the right. These are frame pixels, not indices on a coarser grid of blocks or tiles you may "
             "have derived. Each MOUSE result names what the click hit on the board before it.")
    # item 4 - UNDO and the budget
    s = _sub(s, "- UNDO reverses a previous action, usually the last one. Check what it restores. It cannot undo a failed "
                "attempt.\n",
             "- UNDO reverses a previous action, usually the last one. Check what it restores. It cannot undo a failed "
             "attempt, and whether it gives back step budget depends on the game.\n")
    # item 13 - a no-op is evidence about one state
    s = _sub(s, "\n## Outcomes\n",
             "- An action that changed nothing in one state may work in another: retry it after the board has changed "
             "before ruling it out.\n\n## Outcomes\n")
    # item 3 - outcomes wording (bar, step budget)
    s = _sub(s, "An action that only shortens an edge bar changes the board but not the gameplay.",
             "An action that only changes an edge bar changes the board but not the gameplay.")
    s = _sub(s, "or because the step or time budget ran out.", "or because the step budget ran out.")
    # item 2b - REVERTED verdict
    s = _sub(s, "LEVEL RESTARTED (the attempt ended here; the flag is `game_over`), LEVEL COMPLETED or RUN COMPLETE.",
             "REVERTED (it animated, then the board returned to its previous state: a blocked move, or an effect that "
             "shows only during the animation), LEVEL RESTARTED (the attempt ended here; the flag is `game_over`), LEVEL COMPLETED or RUN COMPLETE.")
    # items 2a-2f - when batches and calls stop
    s = _sub(s, "Batching remains the right approach: a stop concerns one action on one board.\n",
             "Batching remains the right approach: a stop concerns one action on one board.\n- A batch also stops at "
             "the second REVERTED action in a row and when the board returns to this level's starting state. One call "
             "executes at most %d actions; later ones are not executed, and the next turn continues from the current "
             "board.\n" % CALL_CAP)
    s = _sub(s, "- From level 2 on, after an action that changed nothing, the next action(...) call in the same snippet is "
                "refused: re-read the state first.",
             "- From level 2 on, after an action that changed nothing, the next action(...) call in the same snippet is "
             "refused once with StaleStateActionError: re-read the state first; a later call proceeds. In one snippet, "
             "an action already executed from the same board is refused (RepeatedActionInStateError): write a new "
             "snippet if you are sure. An action that is not valid now raises InvalidActionError and executes nothing.")
    # item 2g - start frames
    s = _sub(s, "- History spans all levels and restarts.",
             "- `level_start_frame` is the first board of the current level and `attempt_start_frame` the first board of "
             "the current attempt.\n- History spans all levels and restarts.")
    # item 8 - pre-imports and retained constants
    s = _sub(s, "- Allowed imports: bisect, collections, copy, fractions, functools, heapq, itertools, json, math, "
                "operator, random, re, statistics, string.",
             "- Allowed imports: bisect, collections, copy, fractions, functools, heapq, itertools, json, math, "
             "operator, random, re, statistics, string. `collections`, `itertools`, `math`, `re` and `heapq` are "
             "already imported in every call.")
    s = _sub(s, "- Python variables reset between calls, but your functions persist.",
             "- Python variables reset between calls, but your functions persist, and so do top-level constants named in "
             "UPPER_CASE whose values are literals (numbers, strings, or lists, tuples, sets and dicts of them).")
    # item 12 - library listing: no recent list; level labels
    s = _sub(s, "When the library changes, the next turn lists the ten most recently added programs with their signatures "
                "and docstrings. `programs()` lists all of them, and `read_code(name)` shows a function's source.",
             "`programs()` lists the library with the level each program was last written on, and `read_code(name)` "
             "shows its source; a program written for another level may not fit this one. A new level's first turn "
             "says how many programs you have.")
    # item 11 - the self-made oracle
    s = _sub(s, "- When confidence is low, run the cheapest probe that tells your explanations apart.",
             "- When confidence is low, run the cheapest probe that tells your explanations apart.\n- An 'unreachable' or "
             "'no solution' result from your own simulator or search is a claim about your model of the game, not about "
             "the game: test it with the smallest real action before acting on it.")
    s = _sub(s, "Each call has a 10-second time limit,", "Each call has a limit of 10 seconds of computation,")  # v6.2: CPU time
    t['system'] = s

    # items 6, 9 and 13 - level-up: scoped carry-over, the completion summary
    lv = t['levelup']
    lv = _sub(lv, "Start from the mechanics you have established and do not rediscover them without reason.",
              "Start from the mechanics you have established, remembering which levels each held on: a rule seen on "
              "one level may not hold on this one. Check a carried mechanic with one cheap probe before a plan depends "
              "on it, and if one observation contradicts a carried rule or constant, probe it now instead of "
              "explaining it away.")
    if summary():
        lv = lv + ('\nBefore planning, summarise in at most five lines the game mechanics established so far and what '
                   'cleared level {level-1}, starting with "Rule L{level-1}:". For each mechanic, note the levels it held '
                   'on and whether you tested it or inferred it. If the level completed in a way your rules did not '
                   'predict, say so.' + (' This summary is kept and shown to you again later.'
                                         if restart_on() or game_pool_on() else ''))   # v6.7x GAMEBUDGET
        # v6.2 round 3 (b) (A2 bp35, sb26): the sentence only where a restart re-shows the summaries
    t['levelup'] = lv

    # items 3 and 4 - restart block: facts, not causes
    r = t['restart']
    r = _sub(r, "The fatal action was {fatal_action} (see the trace above).",
             "The fatal action was {fatal_action} (see the trace above); it {fatal_interior} the board interior.")
    if a3():
        r = _sub(r, "the attempts lasted {lengths} actions.", "the attempts lasted {lengths}.")
    r = _sub(r, "or from the step or time budget running out, usually shown by an edge bar that shrinks with each action.",
             "or from the step budget running out, usually shown by an edge bar that fills or shrinks with each action.")
    r = _sub(r, "`history[-1]` is the automatic restart.",
             "`history[-1]` is the automatic restart, and `attempt_start_frame` is the board you act on now.")
    r = _sub(r, "Budget rule: a bar that shrinks with every action is normal, so a slightly shorter bar is not the cause. "
                "The budget caused the death only if the bar is empty or nearly empty in the death frame.",
             "Budget rule: a bar that changes with every action is normal, so a slightly changed bar is not the cause. "
             "The budget caused the death only if the bar is at or near its end in the death frame (empty, or full if "
             "it fills).")
    t['restart'] = r

    # item 2b - animation note for a reverted action
    a = t['animation'].split('\n')
    a[1] = ("The last executed action ({action}) animated over {frames} frames, then the board returned to its previous "
            "state (REVERTED): {k} cells changed and changed back{where}. It may be a blocked move, or the animation "
            "itself may carry the information; `last_animation_timeline` shows what happened.")
    t['animation'] = '\n'.join(a)

    # new opener lines (v6.1)
    t['revert_stop'] = ("Your batch was stopped after two REVERTED actions in a row ({a1}, then {a2}): each animated and "
                        "left the board as it was (a blocked move, or an effect that shows only during the animation)."
                        "{skipped_part}")
    t['hazard_restart'] = ("The attempt ended at {action} without a game-over flag: the board returned to this level's "
                           "starting state and the edge band reset with it, as after a restart. That attempt lasted {n} actions."
                           "{skipped_part}")
    t['level_start_soft'] = ("After {action}, the board area matches this level's starting board: a return to the start, "
                             "or a restart the edge does not show.{skipped_part}")
    t['call_cap'] = ("Your call reached the limit of %d executed actions; {m} requested action(s) were not executed. "
                     "Continue from the current board." % CALL_CAP)
    t['run_clock'] = "Run time left: {minutes} min (the whole run, not an attempt timer)."
    t['no_action'] = ("You have made {k} Python calls in a row without executing an action on this board. Actions not "
                      "yet executed on this level: {untried}.")
    t['library_count'] = ("Your program library has {n} programs, written on {levels}; `programs()` lists them "
                          "with their levels, and `read_code(name)` shows any source.")
    t['edge_line'] = ("Edge band (the 4 cells along the border) in your previous call: {text}. Budget bars usually sit "
                      "there, but game objects can also move through this band.")
    t['takeover'] = "The game is at level {level} (levels 1-{prev} completed)."
    t['takeover_l1'] = "The game is at level 1."
    t['timeout_wall'] = ("The call was stopped after 120 seconds of wall-clock time. It may already have executed actions: "
                         "inspect the current state before retrying.")
    # round 5m (item 123): a sandbox process killed at its hard CPU limit (it previously read "Sandbox process exited
    # unexpectedly."); the stated limit's own error keeps the output, this one cannot
    t['compute_limit_lost'] = ("This call used its 10 seconds of computation and was stopped; its printed output was lost. "
                               "It may already have executed actions: inspect the current state before retrying.")
    # v6.2 (b): a no-op that ended its batch skipped nothing; say so instead of "Your batch was stopped ..."
    t['noop_last'] = ("{action}, the last action of its batch, executed but changed nothing inside the board area. "
                      "Re-locate from `current_frame` instead of counting moves.")
    t['takeover_history'] = ("In `history` and `transitions`, this level begins at the current board; earlier levels "
                             "are kept.")
    t['takeover_history_l1'] = "In `history` and `transitions`, the game begins at the current board."
    t['restart_summaries'] = ("Mechanics you summarised when completing earlier levels; later levels can change "
                              "them:")

    # v6.2 round 3 (A2 review; texts that misled the model or no longer matched the harness)
    s = t['system']
    # (b) tn36: the record is the latest action's; an action that did not animate leaves it empty
    s = _sub(s, "`last_animation_frames` holds the frames of the last action that animated, as views like "
                "`current_frame` (index 0 first; the last entry is the settled board; after a death, the frames of the "
                "fatal action).",
             "`last_animation_frames` holds the frames of your latest action, as views like `current_frame` (index 0 "
             "first; the last entry is the settled board; after a death, the frames of the fatal action), and is empty "
             "when that action did not animate.")
    # (b) tn36: a click or RESET that changed nothing no longer arms the refusal (v6.2); dc22, ka59: what a repeat
    # refusal does to the snippet
    s = _sub(s, "refused once with StaleStateActionError: re-read the state first; a later call proceeds. In one snippet, "
                "an action already executed from the same board is refused (RepeatedActionInStateError): write a new "
                "snippet if you are sure.",
             "refused once with StaleStateActionError: re-read the state first; a later call proceeds. A click or RESET "
             "that changed nothing does not trigger this. In one snippet, an action already executed from the same board "
             "is refused with RepeatedActionInStateError, which ends the snippet (`except Exception` does not catch it; "
             "output printed before it is kept): write a new snippet if you are sure.")
    # (b) lf52: 47 of 64 refusing snippets tested `stop_reason` after single actions, where it is None
    s = _sub(s, "Re-locate from `current_frame` rather than counting the moves you submitted.",
             "Re-locate from `current_frame` rather than counting the moves you submitted. This stop applies to batches "
             "of two or more actions: after a single action that changed nothing, `stop_reason` is None, so test its "
             "`gameplay_changed`.")
    # (a) ls20: a restart without a game-over flag now ends the snippet like a game over
    s = _sub(s, "- After game_over, level_completed or run_complete, any further action(...) call in the same snippet "
                "raises an error.",
             "- After game_over, level_completed, run_complete or a LEVEL RESTARTED verdict, any further action(...) call "
             "in the same snippet raises an error.")
    # round 4 (b) (A2/A3 cd82: inclusive frame_diff boxes read as exclusive): the conventions differ by source
    s = _sub(s, "carry a `bbox` as [r0, c0, r1, c1]", "carry a `bbox` as inclusive [r0, c0, r1, c1] (last row and column "
                "included)")
    s = _sub(s, "clockwise outer-contour corners as [row, col]", "clockwise outer-contour corner cells as [row, col]")
    # round 4 (b) (A2/A3 ft09: programs are kept from a call that later failed, which the prompt denied)
    s = _sub(s, "when the call finishes without error and they pass validation",
             "when their definition has run (even if the call fails later) and they pass validation")
    s = _sub(s, "bounding boxes use exclusive stops", "the starter helpers' bounding boxes use exclusive stops (unlike "
                "`frame_diff` and `segmentation`)")
    t['system'] = s
    t['nameerror_note'] = ("Python variables do not persist between calls; only programs in your library and UPPER_CASE "
                           "constants bound to literals do.")
    t['restart_lengths'] = "The attempts on this level so far lasted {lengths}."

    # v6.2 round 5 (user 2026-10-05 ~13:50): "the harness should avoid injecting any priors because we really do not know
    # what the game looks like" (the harness does not see the game). Rules are stated as measurements, interpretations
    # are dropped, and the inferred restarts and the level-start stop are gone (session_ext 2c). General board-reading
    # advice stays (user: "probably ok").
    s = t['system']
    s = _sub(s, "`board_changed`: any cell changed, including the 4-cell band along the edges where budget bars usually sit.",
             "`board_changed`: any cell changed, including the outer 4-cell band along the edges.")
    s = _sub(s, "`gameplay_changed`: a cell further inside changed. An action that only changes an edge bar changes the board "
                "but not the gameplay.",
             "`gameplay_changed`: a cell more than 4 cells from the edge changed (changes in the outer 4-cell band are "
             "not counted).")
    s = _sub(s, "`game_over`: this attempt failed, either because the action itself was fatal (for example a hazard) or because "
                "the step budget ran out.",
             "`game_over`: the game reported that this attempt ended.")
    s = _sub(s, "NO-OP (executed, but nothing changed inside the board area), REVERTED (it animated, then the board returned "
                "to its previous state: a blocked move, or an effect that shows only during the animation), LEVEL RESTARTED "
                "(the attempt ended here; the flag is `game_over`), LEVEL COMPLETED or RUN COMPLETE. A change limited to the "
                "edge band is not evidence that a move worked.",
             "NO-OP (executed, but nothing changed more than 4 cells from the edge), REVERTED (it animated, then the board "
             "returned to its previous state), LEVEL RESTARTED (the attempt ended here; the flag is `game_over`), LEVEL "
             "COMPLETED or RUN COMPLETE. When the board returns to the level's starting board, the line says so.")
    s = _sub(s, "- A batch stops at the first action that changes nothing inside the board area;",
             "- A batch stops at the first action that changes nothing more than 4 cells from the edge;")
    s = _sub(s, "- A batch also stops at the second REVERTED action in a row and when the board returns to this level's "
                "starting state. One call executes at most %d actions; later ones are not executed, and the next turn "
                "continues from the current board." % CALL_CAP,
             "- A batch also stops at the second REVERTED action in a row. One Python call (the whole snippet, all its "
             "action(...) calls together) executes at most %d actions; later ones are not executed, and the next turn "
             "continues from the current board." % CALL_CAP)
    s = _sub(s, "- After game_over, level_completed, run_complete or a LEVEL RESTARTED verdict, any further action(...) call",
             "- After game_over, level_completed or run_complete, any further action(...) call")
    s = _sub(s, "after a death, the frames of the fatal action), and is empty",
             "after a game over, the frames of the action that ended the attempt), and is empty")
    # round 5b (audit item 4): the two edge-bar checks, as measurements
    s = _sub(s, "- After every action, check whether gameplay objects changed or only an edge bar did.",
             "- After every action, check which cells changed and where they lie, inside the board or in the outer "
             "4-cell band.")
    # round 5b (audit item 1, user: off): no repeat-state refusal sentence when the guard is off
    if os.environ.get('ARC3_REPEAT_STATE_GUARD', '0') != '1':
        s = _sub(s, " In one snippet, an action already executed from the same board is refused with RepeatedActionInStateError, "
                    "which ends the snippet (`except Exception` does not catch it; output printed before it is kept): write a "
                    "new snippet if you are sure.", "")
    s = _sub(s, "- A long line or a segmented strip along an edge is usually a step budget that changes as you act. It may "
                "fill or shrink; its length and rate can differ between levels; whether no-ops count depends on the game; "
                "and reaching its end usually, but not always, ends the attempt. Plan within it rather than treating it as "
                "a puzzle piece, unless evidence shows that it interacts with the puzzle. A common failure is clicking "
                "through such a strip segment by segment: if your actions change only an edge strip while the interior "
                "stays the same, the strip is a status display.",
             "- If your actions change only cells in the outer 4-cell band, note that the NO-OP test ignores those cells.")
    # round 5b: the facts a loop can test (level start, repeats) replace the refusals
    s = _sub(s, "When the board returns to the level's starting board, the line says so.",
             "When the board returns to the level's starting board, the line says so, and that action's result has "
             "`level_start_match` ('full' when the whole board matches, 'interior' when the board apart from its outer "
             "4-cell band does) and `edge_cells_restored` (cells of the outer band that returned to their starting "
             "values). When this Python call already executed the same action from the same board (boards compared apart "
             "from their outer 4-cell band), the result has `repeat_in_call` (how many times). A loop can test these "
             "fields.")
    # round 5c (B3 cn04, sk48, sc25: `current_frame.segmentation()` raised TypeError, the docs saying it "returns")
    s = _sub(s, "- `.segmentation` returns `{'nodes': [...], 'adjacency_list': [...]}`.",
             "- `.segmentation` is an attribute, not a method (no parentheses): `{'nodes': [...], 'adjacency_list': [...]}`.")
    s = _sub(s, "Use `segmentation` as your primary view;", "Use `.segmentation` as your primary view;")
    t['system'] = s
    t['reminder'] = _sub(t['reminder'], "use `segmentation` first", "use `current_frame.segmentation` (an attribute) first")
    # round 5c (B3 vc33 L4 T38): the no-op stop's last sentence as a measurement too (see agent._NOOP_LAST)
    t['noop_stop'] = _sub(t['noop_stop'], "only the actions before it moved anything.",
                          "only the actions before it changed anything more than 4 cells from the edge.")
    t['reminder'] = _sub(t['reminder'], " Tell gameplay changes from edge-bar changes.", "")
    # the game-over block: facts only (no cause dichotomy, budget rule or "do not repeat that path")
    t['restart'] = (
        "LEVEL RESTARTED (attempt {next_attempt} of level {level}): the game reported game over during your previous "
        "call, and the level has restarted from its initial state; completed levels are kept. The last action of that "
        "attempt was {fatal_action} (see the trace above); it {fatal_interior} the board interior.{fatal_position}\n"
        + ("This level has now restarted {k} time(s); the attempts lasted {lengths}.\n" if a3() else "") +
        "`history[-2]` is that action and its `.frame` is the attempt's final board; `history[-3].frame` is the board just "
        "before it; `history[-1]` is the automatic restart, and `attempt_start_frame` is the board you act on now.\n"
        "[image] Current board, after the restart: this is the board you act on.\n"
        "[image] Final board of the attempt.\n"
        "[image] Last step: cells changed by the last action in their new colour, everything else dark navy.")
    a = t['animation'].split('\n')
    a[1] = ("The last executed action ({action}) animated over {frames} frames, then the board returned to its previous "
            "state (REVERTED): {k} cells changed and changed back{where}; `last_animation_timeline` shows what happened.")
    a[2] = a[2].replace("That fatal action animated", "That last action animated")
    t['animation'] = '\n'.join(a)
    # round 5c (B3 sc25 T28): the count runs across the action(...) calls of one Python call, so say so
    # round 5d (C50 g007 T21, g023 L5 T244, L7 T269: "the batch" was read as the whole Python call, whose later action()
    # calls still ran): the stop ends the action() list it fired in
    t['revert_stop'] = ("Two REVERTED actions in a row in this Python call ({a1}, then {a2}): each animated and then left "
                        "the board as it was, and that action() list stopped there.{skipped_part}")
    t['noop_stop'] = _sub(t['noop_stop'], "executed but changed nothing inside the board area",
                          "executed but changed nothing more than 4 cells from the edge")
    t['noop_last'] = _sub(t['noop_last'], "executed but changed nothing inside the board area",
                          "executed but changed nothing more than 4 cells from the edge")
    t['edge_line'] = "Edge band (the outer 4 cells, which the NO-OP test does not count) in your previous call: {text}."
    # round 5d (C50 g027 ~14K tokens, g019, g009, g030, g032: "more than 4 cells from the edge" against "outer 4-cell
    # band" read as off by one; reviewers asked for exact indices): the interior and the outer band by rows and columns
    s = t['system']
    s = _sub(s, "`board_changed`: any cell changed, including the outer 4-cell band along the edges.",
             "`board_changed`: any cell changed, including the outer band.")
    s = _sub(s, "`gameplay_changed`: a cell more than 4 cells from the edge changed (changes in the outer 4-cell band are "
                "not counted).",
             "`gameplay_changed`: a cell of the interior changed. The interior is rows 4-59 and columns 4-59; the outer "
             "band (rows 0-3 and 60-63, columns 0-3 and 60-63) is not counted.")
    s = _sub(s, "NO-OP (executed, but nothing changed more than 4 cells from the edge)",
             "NO-OP (executed, but nothing in the interior changed)")
    s = _sub(s, "A batch stops at the first action that changes nothing more than 4 cells from the edge;",
             "A batch stops at the first action that changes nothing in the interior;")
    s = _sub(s, "If your actions change only cells in the outer 4-cell band,", "If your actions change only cells of the outer band,")
    s = _sub(s, "'interior' when the board apart from its outer 4-cell band does)", "'interior' when the interior does)")
    s = _sub(s, "(boards compared apart from their outer 4-cell band)", "(boards compared on the interior)")
    s = _sub(s, "inside the board or in the outer 4-cell band", "in the interior or in the outer band")
    # round 5d (C50 g019 T472, g014 ~5K tokens: "an automatic restart after a death" was read as covering every return to
    # the level's start, so silent returns were dismissed as deaths): which restarts have an entry, as a fact
    s = _sub(s, "An automatic restart after a death is its own entry with `.result['automatic'] == True`, and the fatal "
                "action's entry comes just before it.",
             "After a GAME OVER the level restarts automatically: that restart is its own entry with "
             "`.result['automatic'] == True`, and the fatal action's entry comes just before it. A return to the level's "
             "starting board without a GAME OVER has no entry of its own; that action's result carries "
             "`level_start_match`.")
    # round 5d (user): the new per-action facts, documented with the others
    s = _sub(s, "the result has `repeat_in_call` (how many times). A loop can test these fields.",
             "the result has `repeat_in_call` (how many times). An action that changed the interior has `interior_changed` "
             "({'n', 'rows', 'cols'}: how many interior cells changed and their row and column span). When the interior is "
             "back at the level's starting board, or within 16 cells of it after being further away, the result has "
             "`actions_since_start` (actions since the board was last at the level's start), and for the near case "
             "`near_level_start` (how many interior cells still differ). A loop can test these fields.")
    t['system'] = s
    t['noop_stop'] = _sub(t['noop_stop'], "only the actions before it changed anything more than 4 cells from the edge.",
                          "only the actions before it changed anything in the interior.")
    for k in ('noop_stop', 'noop_last'):
        t[k] = _sub(t[k], "executed but changed nothing more than 4 cells from the edge", "executed but changed nothing in the interior")
    t['edge_line'] = ("Outer band (rows 0-3 and 60-63, columns 0-3 and 60-63; the NO-OP test does not count it) in your "
                      "previous call: {text}.")
    t['call_cap'] = ("Your Python call (the whole snippet) reached the limit of %d executed actions for one Python call (all "
                     "its action(...) calls together); {m} requested action(s) were not executed. The next Python call may "
                     "execute up to %d again; continue from the current board." % (CALL_CAP, CALL_CAP))
    # round 5f (Fable's final review, user 2026-10-06 ~04:30): E5 - REVERTED compares the interior (band cells may have
    # changed), named as in the round-5d verdicts; S1 - the near-start line fires again only after a larger excursion
    # (V62_BUILD item 74); S2 - the edge line no longer repeats the band's definition (the system prompt has it)
    t['system'] = _sub(t['system'], "REVERTED (it animated, then the board returned to its previous state)",
                       "REVERTED (it animated, then the interior returned to its previous state)")
    t['system'] = _sub(t['system'], "or within 16 cells of it after being further away,",
                       "or within 16 cells of it after having differed from it in more than 64 cells since the board was "
                       "last that close,")
    t['animation'] = _sub(t['animation'], "then the board returned to its previous state (REVERTED)",
                          "then the interior returned to its previous state (REVERTED)")
    t['revert_stop'] = _sub(t['revert_stop'], "each animated and then left the board as it was",
                            "each animated and then left the interior as it was")
    t['edge_line'] = "Outer band in your previous call: {text}."
    # round 5g (user 2026-10-06 ~21:40, after reading the drift cases; V62_BUILD items 76-78): the yield resumption after
    # a call that ended in an error no longer says it "computed or inspected state"; the context line says that parts of
    # later turns can be cut too and what always remains (protected: the system message and the latest board)
    t['resumption_error'] = _sub(t['resumption'], "Your last Python call executed no actions. That is fine: it computed or "
                                                  "inspected state.",
                                 "Your last Python call ended with an error and executed no actions.")
    t['resumption_noact'] = _sub(t['resumption'], "Your last Python call executed no actions. That is fine: it computed or "
                                                  "inspected state.",
                                 "Your last Python call executed no actions.")   # v6.4 B3: a no-action warning stands
    t['system'] = _sub(t['system'], "- Older turns may be removed from this conversation as it grows; `history` keeps every "
                                    "action and board of the game, and your library functions persist.",
                       "- Older turns, and parts of later turns, may be removed from this conversation as it grows, so "
                       "earlier messages can be missing or cut short; this system message and the latest board always "
                       "remain. `history` keeps every action and board of the game, and your library functions persist.")
    # round 5g (user 2026-10-06 ~22:05, V62_BUILD item 79): the level-completion summary as a bullet list of up to 10 items
    if summary():
        t['levelup'] = _sub(t['levelup'], 'summarise in at most five lines the game mechanics established so far and what '
                                          'cleared level {level-1}, starting with "Rule L{level-1}:".',
                            'summarise as a bullet list of at most 10 items the game mechanics established so far and what '
                            'cleared level {level-1}: a line "Rule L{level-1}:" followed by one item per line, each '
                            'starting with "- ". Keep the summary within 4,000 tokens; a longer one is truncated.')
    # round 5h (copycat-30 review + Fable, user 2026-10-07; V62_BUILD items 80-): facts and wording only
    t['takeover_restart'] = ("This conversation begins after a RESET of level {level} at step {step}; earlier attempts on "
                             "this level are not shown.")                                  # D1 (user)
    if 'This summary is kept and shown to you again later.' in t['levelup']:              # G8
        t['levelup'] = _sub(t['levelup'], 'This summary is kept and shown to you again later.',
                            'This summary is kept and shown to you again if this game restarts a level.')
    t['system'] = _sub(t['system'], "- Never give up: only the harness ends a game. When you are stuck, revise your hypotheses "
                                    "and try a different approach.",
                       "- Never give up: only the harness ends a game. When you are stuck, revise your hypotheses and try a "
                       "different approach.\n- No closing report is expected: keep playing until the harness ends the game, "
                       "however little run time is left.")                                # Fable Q4.1 (ly52, by35, gx0t)
    t['system'] = _sub(t['system'], "When the board returns to the level's starting board, the line says so, and that "
                                    "action's result has",
                       "These facts are in each executed action's result: its `history` entry's `.result`, and for the "
                       "last executed action of a call also the top level of `last_action_call_result`, which has no "
                       "per-action result records (its per-action lists are `executed_actions`, `action_trace` and "
                       "`gameplay_changed_per_action`). When the board returns to the level's starting board, the line "
                       "says so, and that action's result has")                            # G4 (cy82, dx22, sy25)
    t['system'] = _sub(t['system'], "An action that changed the interior has `interior_changed` ({'n', 'rows', 'cols'}: how "
                                    "many interior cells changed and their row and column span).",
                       "Every executed action has `interior_changed` ({'n', 'rows', 'cols'}: how many interior cells "
                       "changed and their row and column span; n is 0 when none did).")      # G4 (rx1l)
    t['system'] = _sub(t['system'], "`.animation_frames` (the frames the action passed through; empty if it did not animate).",
                       "`.animation_frames` (the frames the action passed through; empty if it did not animate). The entry "
                       "of an action that completed a level holds the next level's first board.")   # Fable Q4.3 (fy09)
    t['retention_kept'] = ("`{name}` was bound to another value or deleted in this call without a new definition; the saved "
                           "program `{name}` is kept for later calls.")                        # G2
    t['system'] = _sub(t['system'], "n is 0 when none did).",
                       "n is 0 when none did). When changed cells of the outer band connect, through changed cells, to "
                       "changed interior cells (one changed region crossing into the band), the result also has "
                       "`band_connected` (how many such band cells).")                       # G3 (sx25, sy25)
    t['system'] = _sub(t['system'], "A click or RESET that changed nothing does not trigger this.",
                       "A RESET that changed nothing does not trigger this, and after a click that changed nothing only "
                       "the same click (same cell) is refused.")                             # Fable Q3 (by35)
    t['system'] = _sub(t['system'], "`last_animation_timeline` lists, step by step, the cells that changed.",
                       "`last_animation_timeline` is a dict with `action`, `frames` and `steps` (one entry per frame that "
                       "changed cells: `step`, `changed`, `bbox`, and `changes` or `transitions`), or only `error` when "
                       "that action did not animate.")                                       # Fable Q4.4 (kx59, sy25, fx09)
    t['no_change'] = "No cell differs from the board at your previous turn."                 # G10 (gx0t)
    t['blocked_builtin_note'] = "`{name}` is a Python builtin that this sandbox does not provide."   # Fable Q4.5 (lx85)
    # round 5h review (387b severe-error check: 87 malformed calls, 78 with no arguments, in unbroken series per game)
    t['markup_recovered'] = ("Note: this tool call was malformed (the python tool takes one argument, `code`, and the call "
                             "did not carry it); the harness recovered your code from the message and ran it.")
    # round 5i (item 98; 391: series of recovered calls that each dropped the same tag, s5i5 26 of 26 after the note above):
    # the note names the missing tag and gives the format (the breaker's format sentence)
    t['markup_recovered'] += (" Write each python call in the required format: <tool_call>, <function=python>, "
                              "<parameter=code>, your code, </parameter>, </function>, </tool_call>.")
    t['markup_recovered_defect'] = ("Note: this tool call was malformed: {what}, so it carried no `code` argument. The "
                                    "harness recovered your code from the message and ran it. Write each python call in the "
                                    "required format: <tool_call>, <function=python>, <parameter=code>, your code, "
                                    "</parameter>, </function>, </tool_call>.")
    # round 5h review (387b: summaries captured at 60% of level-ups vs 81-91% in five earlier runs; in the misses the model
    # wrote the list but not the "Rule Lk:" line, which the 5g wording had put mid-sentence after "bullet list"): the header
    # line first, then the items enumerated in any form (user: "enumerate" so bullets or numbers both work)
    if summary():
        t['levelup'] = _sub(t['levelup'], 'summarise as a bullet list of at most 10 items the game mechanics established so far '
                                          'and what cleared level {level-1}: a line "Rule L{level-1}:" followed by one item per '
                                          'line, each starting with "- ". Keep the summary within 4,000 tokens; a longer one is '
                                          'truncated.',
                            'write a summary that begins with the line "Rule L{level-1}:" and then enumerates, in at most 10 '
                            'items, the game mechanics established so far and what cleared level {level-1}. Keep the summary '
                            'within 4,000 tokens; a longer one is truncated.')
    return t



def _v63_texts(t, with_show_board=True):
    """v6.3 item G (user 2026-10-14; TEXTS_REVIEW_2026-10-14.md): clarity and redundancy edits to the system prompt, applied
    after every earlier round. 1: the per-action result fields of Outcomes as sub-bullets instead of one paragraph. 2: withdrawn
    (the BAND-ONLY line stays, item 102). 3: show_board described once, in Working in Python;
    the Programs copy, which split the starter-helper list, removed. 4: "After every action, check which cells changed" merged
    into the default loop. 5: the change-image rule names the other two no-change-image cases."""
    s = t['system']
    # 1. Outcomes: the result fields, one per sub-bullet (same statements)
    a = s.find("These facts are in each executed action's result:")
    b_mark = "A loop can test these fields."
    b = s.find(b_mark)
    assert a > 0 and b > a and s.count("These facts are in each executed action's result:") == 1 and s.count(b_mark) == 1
    fields = (
        "\n  - These facts are in each executed action's result: its `history` entry's `.result`, and for the last executed action "
        "of a call also the top level of `last_action_call_result`, which has no per-action result records (its per-action lists "
        "are `executed_actions`, `action_trace` and `gameplay_changed_per_action`). A loop can test the fields below."
        "\n  - Every executed action has `interior_changed` ({'n', 'rows', 'cols'}: how many interior cells changed and their row "
        "and column span; n is 0 when none did)."
        "\n  - `band_connected` (how many band cells): when changed cells of the outer band connect, through changed cells, to "
        "changed interior cells (one changed region crossing into the band)."
        "\n  - `repeat_in_call` (how many times): when this Python call already executed the same action from the same board "
        "(boards compared on the interior)."
        "\n  - `level_start_match` ('full' when the whole board matches, 'interior' when the interior does) and "
        "`edge_cells_restored` (cells of the outer band that returned to their starting values): when the board returns to the "
        "level's starting board; the verdict line says so too."
        "\n  - `actions_since_start` (actions since the board was last at the level's start): when the interior is back at the "
        "level's starting board, or within 16 cells of it after having differed from it in more than 64 cells since the board "
        "was last that close; in the near case also `near_level_start` (how many interior cells still differ)."
        "\n  - `start_return` (True): when the whole board returns to the level's starting board without a RESET after its "
        "interior had differed from it in more than 64 cells."
        "\n  - `board_shift` ({'dy', 'dx', 'explained', 'changed'}: rows down and columns right): when one shift of the board "
        "content explains at least 90% of 64 or more changed cells."
        "\n  - `undo_restored` ({'step', 'action'}: the action that board came before): after an UNDO whose board equals an "
        "earlier board of the attempt.")
    s = s[:a].rstrip() + fields + s[b + len(b_mark):]
    # 2. (withdrawn) the BAND-ONLY line of Reading the board stays: round 5l item 102 added it on purpose (band changes read as
    #    NO-OP in 392b); the Outcomes list defines the verdict, this line states the distinction where the confusion arose
    # 3. show_board once, in Working in Python; the Programs copy removed (it split the starter-helper list)
    if with_show_board:
        s = _sub(s, "To look at a whole board your code built, call `show_board(cells)`: it is added to this call's result as an "
                    "image, like the board image (at most 4 per call).",
                 "To look at a whole board your code built, call `show_board(cells, label=None)` (a frame, rows of symbols or ASCII "
                 "text, as `grid_cells` accepts): it is added to this call's result as an image captioned with `label`, like the "
                 "board image; at most 4 per call, and a board already shown in the call is not repeated.")
        s = _sub(s, "- `show_board(cells, label=None)` adds a board (a frame, rows of symbols or ASCII text, as `grid_cells` accepts) "
                    "to this call's result as an image captioned with `label`; at most 4 per call, and a board already shown in the "
                    "call is not repeated.\n", "")
    # 4. "After every action, check which cells changed" merged into the default loop
    s = _sub(s, "execute the best probe or plan with action(...), then inspect exactly what changed.",
             "execute the best probe or plan with action(...), then inspect exactly what changed and where it lies: in the "
             "interior or in the outer band.")
    s = _sub(s, "- After every action, check which cells changed and where they lie, in the interior or in the outer band.\n", "")
    # 5. the change image: the other cases without one
    s = _sub(s, "A level's first turn and the turn after a restart have no change image.",
             "A level's first turn and the turn after a restart have no change image, and neither has a turn in which nothing "
             "changed or a resumption after a non-acting call.")
    t['system'] = s
    return t

def apply(texts):
    """Return a copy of the texts dict with the selected variants applied."""
    t = dict(texts)
    if a13():
        s = t['system']
        # A1: objective (workflow), reading the board (stance + object priors), actions (action priors)
        s = _sub(s, "- Investigate until your code has a probe or plan that the evidence favours; full certainty is not "
                    "required before acting.\n",
                 "- Investigate until your code has a probe or plan that the evidence favours; full certainty is not "
                 "required before acting. Work from mechanics toward the goal: once most objects and action effects are "
                 "pinned down, the goal usually shows in what the board rewards or changes, so test a candidate goal "
                 "cheaply and revise it. When the goal is clear before the mechanics, let it decide which mechanics are worth probing, "
                 "and stop probing once those are settled.\n")
        s = _sub(s, "- Treat each board as a scene: objects, blockers, targets, adjacency, containment, motion and symmetry.\n",
                 "- Treat each board as a scene: objects, blockers, targets, adjacency, containment, motion and symmetry.\n"
                 "- Start from the likely meaning of what you see, as an experienced player would, and probe to confirm "
                 "or discard it: priors are the first hypotheses to test, not conclusions. Shape and colour often signal "
                 "function: repeated small tokens tend to be collectibles or slots, enclosures tend to be containers or "
                 "walls, a single distinct shape tends to be the piece you control or the target, and shared colours "
                 "tend to mark things that belong together.\n")
        s = _sub(s, "- UP, DOWN, LEFT and RIGHT are directional controls; what they move depends on the game.\n",
                 "- UP, DOWN, LEFT and RIGHT are directional controls; what they move depends on the game, usually a "
                 "piece, a cursor or a selection.\n")
        s = _sub(s, "- SPACE performs a game-specific action such as interacting, selecting, rotating, attaching or "
                    "detaching, or executing. Test what it does rather than assuming.\n",
                 "- SPACE performs a game-specific action such as interacting, selecting, rotating, switching or cycling "
                 "the active object, attaching or detaching, or committing an attempt. Test what it does rather than "
                 "assuming.\n")
        s = _sub(s, "- MOUSE clicks one board cell. ",
                 "- MOUSE clicks one board cell, which often selects, toggles or triggers what lies under it. ")
        # A2: verdict list
        s = _sub(s, "effect, NO-OP (executed, but nothing changed inside the board area), GAME OVER, LEVEL COMPLETED or "
                    "RUN COMPLETE.",
                 "effect, NO-OP (executed, but nothing changed inside the board area), LEVEL RESTARTED (the attempt ended "
                 "here; the flag is `game_over`), LEVEL COMPLETED or RUN COMPLETE.")
        t['system'] = s
        # A2: restart header; A3 (ARC3_V6_A3=1 only): life-length line after the first paragraph
        r = t['restart']
        r = _sub(r, "GAME OVER: the attempt failed during your previous call. ",
                 "LEVEL RESTARTED (attempt {next_attempt} of level {level}): the previous attempt failed during your "
                 "previous call. ")
        if a3():
            first_end = r.index('\n')
            r = r[:first_end] + "\nThis level has now restarted {k} time(s); the attempts lasted {lengths} actions." + r[first_end:]
        r = _sub(r, "A game over comes either from", "A restart comes either from")
        t['restart'] = r
        t['restart_verdict'] = 'LEVEL RESTARTED (attempt {k} ended here)'
    if reset():
        s = t['system']
        s = _sub(s, "- UNDO reverses a previous action, usually the last one. Check what it restores. It cannot undo a failed "
                    "attempt.\n",
                 "- UNDO reverses a previous action, usually the last one. Check what it restores. It cannot undo a failed "
                 "attempt.\n- RESET restores the current level to its starting state, including the step or time bar, and keeps "
                 "completed levels. It is always available, although the valid actions listed in observations omit it. "
                 "Use it only as a last resort: when the evidence shows the level can no longer be won "
                 "from the current state (for example a piece is irreversibly stuck or a needed resource is used up), not "
                 "to try an idea the current state still allows. If UNDO can restore a winnable state, prefer it. RESET "
                 "counts as one action, and actions already spent still count. RESET must be the first game action in a "
                 "Python snippet; inspection or computation may precede it.\n")   # wording: user 15:15, last resort only
        t['system'] = s
    if a45():
        lv = t['levelup']
        lv = _sub(lv, "so test them with small, informative probes.",
                  "so test them with small, informative probes, and act on the first probe or plan the evidence "
                  "supports.")
        lv = lv + ('\nBefore planning, state in at most five lines the rule that cleared level {level-1}, starting with '
                   '"Rule L{level-1}:". It will be kept for you.')
        t['levelup'] = lv
        t['rules_block'] = 'Rules you recorded for solved levels:'
    if r61():
        t = _r61(t)
        t = _r5l(t)
        if show_board():
            t = _r5m_show_board(t)
        if os.environ.get('ARC3_V63_TEXTS', '1') == '1':      # v6.3 item G: after every round; its anchors are round-5l/5m texts
            t = _v63_texts(t, show_board())
        if os.environ.get('ARC3_V65_TEXTS', '1') == '1':      # v6.5 (2026-10-17): after v6.3 / v6.4
            t = _v65_texts(t)
        if os.environ.get('ARC3_V66_TEXTS', '1') == '1':      # v6.6 (2026-10-17): after v6.5
            t = _v66_texts(t)
        if not v65_batch():                                     # C6R (10-17): v6.4's stop, stated as a harness rule
            t = _c6r_texts(t)
        if os.environ.get('ARC3_V66_SEEDS', '1') == '1':      # v6.6 item C + SEED1 (user 10-17): on by default
            t = _v66_seed_texts(t)
        if texts67():                                           # v6.7 TEXTS67 (IE 10-18): before the soft and loop texts
            t = _v67_texts(t)
        if score_text_on():                                     # v6.7 SCORE (IE 10-18): the complete scoring rule (with SCO)
            t = _v67_score_texts(t)
        if v68_no_score():                                      # v6.8 NO_SCORE: the v6.6 scoring bullet removed
            t = _v68_no_score_texts(t)
        if notes_on():                                          # v6.7 NOTES (IE 10-18): before the refresh, soft and loop texts
            t = _v67_notes_texts(t)
        if (v68_ledger() or v68_never_types()) and not notes_on():   # v6.8 LEDGER / NEVER_TYPES: NOTES's opener texts
            t = _v68_ledger_texts(t)
        if v68_notes_neg():                                     # v6.8 NOTES_NEG: before the soft and loop requests derive
            t = _v68_notes_neg_texts(t)
        if refresh_on():                                        # v6.7 REFRESH (IE 10-18): the level-up refresh texts
            t = _v67_refresh_texts(t)
            if v68_refresh_texts():                             # v6.8 REFRESH_TEXTS
                t = _v68_refresh_texts(t)
            if v68_refresh_programs():                          # v6.8 REFRESH_PROGRAMS
                t = _v68_refresh_programs_texts(t)
        if restart_kind() == 'soft' or loop_on():               # v6.7 SOFT (IE 10-18); LOOP: rung 2 is a soft restart
            t = _v67_soft_texts(t)
        if loop_on():                                           # v6.7 LOOP (IE 10-18): the ladder's texts
            t = _v67_loop_texts(t)
        if loop_on() and loop_rung2_kind() == 'hard':           # v6.7x LOOPHARD (user decision 50): the block's rung 2
            t = _v67x_loophard_texts(t)
        if v68_loop_texts():                                    # v6.8 LOOP_TEXTS (needs LOOP and a hard rung 2)
            t = _v68_loop_texts(t)
        if sumlookup_on():                                      # v6.7x SUMLOOKUP (user decision 53): its two texts
            t = _v67x_sumlookup_texts(t)
        if v69_clear_step():                                    # v6.9 CLEAR_STEP (decisions 82, 87): the block's texts
            t = _v69_clear_step_texts(t)
        if v69_inlevel_refresh():                               # v6.9 INLEVEL (decisions 94-99): its texts
            t = _v69_inlevel_texts(t)
        if v69_restart_across():                                # v6.9 RAB RESTART_ACROSS: the request over conversations
            t = _v69_across_texts(t)
    if v69_hide_reset():                                        # v6.9 REX HIDE_RESET (decision 49): arm B's texts, last
        t = _v69_hide_reset_texts(t)
    if v70_no_hard_restart():                                   # v7.0 NO_HARD_RESTART: the refresh form, after every other
        t = _v70_nhr_texts(t)
    if v70h_any():                                              # v7.0 harness revision (apply_v70h.py): its texts, last
        t = _v70h_texts(t)
    if os.environ.get('ARC3_V71_TEXTS', '0') == '1':           # v7.1 prompt cleanup (apply_v71.py): after every other variant
        t = _v71_texts(t)
    if v75_polish():                                            # v7.5 POLISH (the user 10-23, review Parts 3-4): after every other
        t = _v75_texts(t)                                       # variant
    if v76_hybrid():                                            # v7.6 HYBRID (the user 10-24): the summary turn, the RESET line
        t = _v76_texts(t)
    if v77():                                                   # v7.7 (the user 10-24): the summaries, the board_shift fact
        t = _v77_texts(t)
    return t


def _r5l(t):
    """Round 5l (392b audit, Fable's review, user decisions 2026-10-07 23:51 / 10-08 00:31; V62_BUILD items 101-).
    Every edit is an exact replacement with an asserted anchor, as above."""
    # item 101 (A1, 392b rw01): a RESET that ended in GAME OVER gets no automatic restart; the opener says so and names the
    # one action that executes
    t['game_over_no_restart'] = ("The game reports GAME OVER, and no automatic restart followed: the last executed action "
                                 "was a RESET, and a RESET that ends in GAME OVER is not repeated automatically. RESET is "
                                 "the only action that executes now; put it first in an action(...) call.")
    # item 102 (A3 option b, user 10-07 23:51 / 10-08 00:31): NO-OP means nothing on the board changed; a change confined
    # to the outer band is BAND-ONLY CHANGE. The batch stop keeps the interior (both labels stop it); the stale refusal
    # follows a NO-OP only
    s = t['system']
    s = _sub(s, "- If your actions change only cells of the outer band, note that the NO-OP test ignores those cells.",
             "- If your actions change only cells of the outer band, their verdict is BAND-ONLY CHANGE, not NO-OP.")
    s = _sub(s, "effect, NO-OP (executed, but nothing in the interior changed), REVERTED",
             "effect, NO-OP (executed, but nothing on the board changed), BAND-ONLY CHANGE (executed; only cells of the "
             "outer band changed), REVERTED")
    s = _sub(s, "- A batch stops at the first action that changes nothing in the interior; the remaining actions are "
                "skipped and reported.",
             "- A batch stops at the first action that changes nothing in the interior (a NO-OP or a BAND-ONLY CHANGE); "
             "the remaining actions are skipped and reported.")
    s = _sub(s, "after a single action that changed nothing, `stop_reason` is None, so test its `gameplay_changed`.",
             "after a single action that changed nothing in the interior, `stop_reason` is None, so test its "
             "`gameplay_changed`.")
    s = _sub(s, "- From level 2 on, after an action that changed nothing, the next action(...) call in the same snippet is "
                "refused once with StaleStateActionError",
             "- From level 2 on, after a NO-OP (nothing on the board changed), the next action(...) call in the same "
             "snippet is refused once with StaleStateActionError")
    t['system'] = s
    t['noop_stop'] = _sub(t['noop_stop'], "Your batch was stopped by the no-op guard: ", "Your batch was stopped: ")
    # item 103 (B1, 392b: "Run time left: N min" read as seconds, the most frequent misreading): hours and minutes
    t['run_clock'] = "Run time left: {left} (the whole run, not an attempt timer)."
    # item 104 (B2, 392b: UNDO assumed available in most games without it; a false UNDO claim carried into a summary): UNDO
    # is described for the games that offer it, and the RESET bullet no longer prefers it unconditionally
    s = t['system']
    s = _sub(s, "- UNDO reverses a previous action, usually the last one. Check what it restores.",
             "- UNDO, when it is among the valid actions, reverses a previous action, usually the last one. Check what it "
             "restores.")
    if reset():                                       # the RESET bullet exists only with EXPOSE_RESET on
        s = _sub(s, "not to try an idea the current state still allows. If UNDO can restore a winnable state, prefer it. "
                    "RESET counts as one action",
                 "not to try an idea the current state still allows. RESET counts as one action")
    t['system'] = s
    # item 107 (A7, 392b): "levels 1-1 completed" (gf01); the two-REVERTED stop when the second ended its list (many games);
    # the malformed-call note says how much it recovered (te01)
    t['takeover'] = _sub(t['takeover'], "(levels 1-{prev} completed)", "({done} completed)")
    t['revert_stop_last'] = ("Two REVERTED actions in a row in this Python call ({a1}, then {a2}): each animated and then left "
                             "the interior as it was; the second was the last action of its action() list, so nothing was "
                             "skipped.{skipped_part}")
    t['markup_recovered'] = _sub(t['markup_recovered'], "the harness recovered your code from the message and ran it.",
                                 "the harness recovered {lines} of code from the message and ran it.")
    t['markup_recovered_defect'] = _sub(t['markup_recovered_defect'], "The harness recovered your code from the message and "
                                                                      "ran it.",
                                        "The harness recovered {lines} of code from the message and ran it.")
    # item 108 (S1, A2, B5; Fable 2.1): the level-completion summary goes through save_summary(text), in the next python call
    # with that call's other code; the "Rule Lk:" capture of the reply text stays as the fallback
    if summary():
        t['levelup'] = _sub(t['levelup'], 'Before planning, write a summary that begins with the line "Rule L{level-1}:" and '
                                          'then enumerates,',
                            "In your next python call, together with that call's other code, save a summary with "
                            'save_summary(text): a text that begins with the line "Rule L{level-1}:" and then enumerates,')
        t['levelup'] = _sub(t['levelup'], 'This summary is kept and shown to you again if this game restarts a level.',
                            'The saved summary is shown to you again if this game restarts a level.')
    t['system'] = _sub(t['system'], "- Give each reusable function a one-line docstring.",
                       "- `save_summary(text)` saves a summary when the harness asks for one (when a level is completed, and "
                       "before a level restarts) and prints what was saved; at other times it saves nothing.\n"
                       "- Give each reusable function a one-line docstring.")
    t['summary_saved'] = ("Saved your summary of level {level} ({tokens} tokens{cut}). It is shown to you again if this game "
                          "restarts a level.")
    t['summary_replaced'] = "It replaces the summary you saved earlier for that level."
    t['summary_saved_attempt'] = ("Saved your notes on level {level} ({tokens} tokens{cut}). They are shown to you after the "
                                  "restart.")
    t['restart_summary_missing'] = "- L{level}: no summary was saved for this level."   # v6.4 A7
    t['summary_not_requested'] = ("No summary is requested now, so nothing was saved. The harness asks for one when a level "
                                  "is completed and before a level restarts.")
    t['summary_not_replaced'] = ("A summary for level {level} is already saved and is not replaced after an action on "
                                 "level {next}, so this text was not saved.")   # v6.4 A2 (IE wording)
    t['summary_tag_closed'] = ("The text is tagged for level {level}, which already has a saved summary, so nothing was "
                               "saved.")   # v6.4 A2 (IE review); v6.5 A2r: only a level with a summary is refused
    t['summary_tag_open_level'] = ("Level {level} is not completed, so it has no summary yet and nothing was saved; the "
                                   "harness asks for notes when they are needed.")   # v6.4 A2 (IE review)
    t['summary_too_short'] = "Nothing was saved: the summary is empty or too short."
    # item 109 (S3: A4, A5, B4; Fable 2.3, 3.5, 3.10): the game's own return to the level's start (a fact and a batch stop),
    # and what a restarted conversation is told about the level's earlier attempts
    t['level_start_return'] = ("After {action}, the whole board, outer band included, matches this level's starting board "
                               "again, with no GAME OVER reported and no RESET; the interior had differed from it in more "
                               "than 64 cells. The batch stopped there.{skipped_part}")
    t['system'] = _sub(t['system'], "- A batch also stops at the second REVERTED action in a row.",
                       "- A batch also stops at the second REVERTED action in a row, and when the whole board returns to the "
                       "level's starting board without a RESET after its interior had differed from it in more than 64 "
                       "cells.")
    t['takeover_restart'] = _sub(t['takeover_restart'], "; earlier attempts on this level are not shown.",
                                 "; your earlier attempts on this level are not in this conversation or in `history`.")
    t['takeover_history'] = _sub(t['takeover_history'], "earlier levels are kept.",
                                 "earlier levels are kept. The step counter counts actions in the whole run, so it "
                                 "continues from before the restart.")
    # item 110 (S4, A6): a new definition that was not saved leaves the earlier version in the library
    t['retention_kept_previous'] = ("The new definition of `{name}` was not saved ({reason}); the earlier version of `{name}` "
                                    "is kept in the program library.")
    # item 111 (D1, user 10-07 23:51; Fable 5.2-5.4): the notes turn before a token restart, and the notes after it
    t['attempt_summary_request'] = ("This conversation has generated {tokens} tokens on level {level}, so the level restarts "
                                    "after this turn: a RESET, then a new conversation that starts from the level's first "
                                    "board. Actions are not executed in this turn. In one python call, save notes for the "
                                    "next conversation with save_summary(text): a text that begins with the line "
                                    "\"Notes L{level}:\" and then gives, in at most 10 items, what you established on this "
                                    "level, what you tried and what happened, and the hypotheses still open. Keep the notes "
                                    "within 4,000 tokens. At the restart, programs you saved on this level are set aside, and "
                                    "a program from an earlier level that you redefined on this level returns to its earlier "
                                    "version; programs from earlier levels stay loaded. Write down what your programs on "
                                    "this level computed rather than relying on them.")   # v6.4 B1 (12-h audit; IE)
    # round 5m (user 10-08): the same request before a position restart (the conversation's length limit)
    t['attempt_summary_request_position'] = _sub(
        t['attempt_summary_request'], "This conversation has generated {tokens} tokens on level {level}, so the level restarts",
        "This conversation has reached its length limit, so level {level} restarts")
    # v6.4 A1 (12-h audit; IE): the one retry of a notes turn that saved nothing
    t['summary_retry'] = ("No notes were saved ({reason}). Save them now with save_summary(text) in one python call, a text "
                          "that begins with the line \"Notes L{level}:\"; actions are not executed in this turn.")
    t['summary_turn_refusal'] = ("Not executed: actions are not executed in this turn, because the level restarts after it. "
                                 "Save your notes with save_summary(text).")
    # v6.3 item B (user 2026-10-12, HARNESS_V63_PLAN B): the no-action token budget's two warnings (explicit about the
    # budget and that acting restores it; the first carries the untried-actions list that the 8-call line carried) and the
    # notes request when that budget ends the conversation
    t['noact_limit_clause'] = (" If you reach {limit} tokens without executing an action, this conversation ends: the level "
                               "resets and a new conversation starts from its first board.")
    t['noact_warn_1'] = ("You have generated {k} tokens in this conversation since your last executed action.{limit_clause} The "
                         "budget is restored as soon as an action executes. Actions not yet executed on this level in this "
                         "conversation: {untried}.")
    # user 2026-10-14: the 48K cue must be strong enough to make the model act on its own (no harness enforcement)
    t['noact_warn_2'] = ("{k} of {limit} tokens generated without an executed action. Your next python call must execute an "
                         "action: take the most informative probe you can justify now, even a single action, and run it with "
                         "action(...) in that call. Further analysis without an action does not help: if no action executes "
                         "before {limit} tokens, this conversation ends and the level restarts from its first board. The budget "
                         "is restored as soon as one action executes.")
    t['noact_warn_2_nolimit'] = ("{k} tokens generated without an executed action. Your next python call must execute an "
                                 "action: take the most informative probe you can justify now, even a single action, and run it "
                                 "with action(...) in that call. Further analysis without an action does not help.")
    t['attempt_summary_request_noact'] = _sub(
        t['attempt_summary_request'], "This conversation has generated {tokens} tokens on level {level}, so the level restarts",
        "This conversation has generated {tokens} tokens on level {level} without executing an action, so the level restarts")
    t['attempt_notes_header'] = ("Notes you wrote at the end of your previous conversation on this level (your own notes, "
                                 "not verified by the harness):")
    t['attempt_notes_header_many'] = ("Notes you wrote at the end of your earlier conversations on this level, oldest first "
                                      "(your own notes, not verified by the harness):")
    t['attempt_notes_none'] = "No notes were saved at the end of your previous conversation on this level."
    # item 112 (C5, B6): an identical report is not repeated; the animation timeline's field types; "did you mean"
    t['same_report'] = ("Your previous call's report was identical to the one before it, so it is not repeated here.")   # v6.4 A4
    t['nameerror_close'] = "Did you mean `{name}`{where}?"
    t['system'] = _sub(t['system'], "`last_animation_timeline` is a dict with `action`, `frames` and `steps` (one entry per "
                                    "frame that changed cells:",
                       "`last_animation_timeline` is a dict: `action` (a string), `frames` (the number of frames, an int) "
                       "and `steps` (a list, one entry per frame that changed cells:")
    # C1 / C3: the new facts, documented where the others are
    t['system'] = _sub(t['system'], "A loop can test these fields.",
                       "When one shift of the board content explains at least 90% of 64 or more changed cells, the result "
                       "has `board_shift` ({'dy', 'dx', 'explained', 'changed'}: rows down and columns right). After an "
                       "UNDO whose board equals an earlier board of the attempt, the result has `undo_restored` ({'step', "
                       "'action'}: the action that board came before). A loop can test these fields.")
    # item 113 (S5): "1 programs" (template boundary test); the silent-return fact documented with the others (class 10)
    t['library_count'] = _sub(t['library_count'], "has {n} programs, written on {levels}; `programs()` lists them",
                              "has {n} program{s}, written on {levels}; `programs()` lists {them}")
    t['system'] = _sub(t['system'], "When one shift of the board content explains",
                       "When the whole board returns to the level's starting board without a RESET after its interior had "
                       "differed from it in more than 64 cells, the result has `start_return` (True). When one shift of the "
                       "board content explains")
    # item 114 (user 10-08 ~01:55: rows 12-14 of NO_PRIORS_TABLE_ROUND5L.md): no step bar assumed; the game-over exception
    if reset():
        t['system'] = _sub(t['system'], "RESET restores the current level to its starting state, including the step or time bar, "
                                        "and keeps completed levels.",
                           "RESET restores the current level to its starting state, including any step or time counter, and "
                           "keeps completed levels.")
    t['system'] = _sub(t['system'], "It cannot undo a failed attempt, and whether it gives back step budget depends on the game.",
                       "It cannot undo a failed attempt, and whether it gives back anything spent depends on the game.")
    t['system'] = _sub(t['system'], "The level restarts automatically from its initial state, and completed levels are kept.",
                       "The level restarts automatically from its initial state (except after a RESET that ended in GAME "
                       "OVER; the turn's message then says so), and completed levels are kept.")
    # round 5l review (Fable T1, T2, T5): the automatic restart's RESET that died; the takeover without hidden history; a cut line
    t['game_over_auto_reset'] = ("The game reports GAME OVER: the automatic restart's RESET itself ended in GAME OVER, and it "
                                 "is not repeated automatically. RESET is the only action that executes now; put it first in "
                                 "an action(...) call.")
    t['game_over_harness_reset'] = ("The game reports GAME OVER: the RESET that began this conversation ended in GAME OVER. "
                                    "RESET is the only action that executes now; put it first in an action(...) call.")
    t['takeover_restart_shown'] = ("This conversation begins after a RESET of level {level} at step {step}; your earlier "
                                   "attempts on this level are not in this conversation.")
    t['truncation_line_cut'] = ("[… a line over the limit was cut, about {t} tokens omitted. Tool output is limited to about "
                                "2,048 tokens; print a narrower selection …]")
    return t


def _v65_texts(t):
    """Harness v6.5 (2026-10-17; user via IE, the 18-h A/B; apply_v65.py): CLOCK, C3, C6 and the C1 / OPENER texts.
    After every earlier round: its anchors are the final v6.4 texts."""
    s = t['system']
    if os.environ.get('ARC3_V65_CLOCK', '0') != '1':          # CLOCK
        s = _sub(s, "the current step, level and attempt, the actions that are valid now, and the time left in the whole run.",
                 "the current step, level and attempt, and the actions that are valid now.")
        s = _sub(s, "keep playing until the harness ends the game, however little run time is left.",
                 "keep playing until the harness ends the game.")
    # C3 (IE final, user-delegated wording)
    s = _sub(s, "A turn ends as soon as a call executes at least one action; the next turn shows the outcome.\n",
             "A turn ends as soon as a call executes at least one action; the next turn shows the outcome.\n"
             "- The game is turn-based: it advances only when one of your actions executes, and nothing in the game changes "
             "while you think or between your calls.\n")
    s = _sub(s, "- Older turns, and parts of later turns, may be removed from this conversation as it grows, so earlier messages "
                "can be missing or cut short; this system message and the latest board always remain. `history` keeps every "
                "action and board of the game, and your library functions persist.",
             "- Older turns, and parts of later turns, are removed from this conversation as it grows, so earlier messages can "
             "be missing or cut short; this never ends the conversation, and this system message and the latest board always "
             "remain. `history` keeps every action and board of the game, and your library functions persist. A harness "
             "restart of the level is normally announced one turn ahead, with a request for notes.")
    if v65_batch():                                            # C6: the batch stop
        s = _sub(s, "- A batch stops at the first action that changes nothing in the interior (a NO-OP or a BAND-ONLY CHANGE); "
                    "the remaining actions are skipped and reported.",
                 "- A batch stops at the second consecutive action that changes nothing on the board (NO-OP); a BAND-ONLY "
                 "CHANGE does not stop it; the remaining actions are not executed.")
        s = _sub(s, "Batching remains the right approach: a stop concerns one action on one board.",
                 "Batching remains the right approach: a stop concerns two actions in a row on one board.")   # IE 10-17
        t['noop_stop'] = ("Your batch was stopped: {action} (position {p} of {n}) was the second action in a row that executed "
                          "but changed nothing on the board (NO-OP), so the remaining {m} action(s) were skipped: {skipped}. "
                          "It is the last action in the executed list above. Re-locate from `current_frame` instead of "
                          "counting moves.")
        t['noop_last'] = ("{action}, the last action of its batch, was the second action in a row that executed but changed "
                          "nothing on the board (NO-OP). Re-locate from `current_frame` instead of counting moves.")
    if os.environ.get('ARC3_STALE_STATE_BLOCK', '0') != '1':  # C6: no stale-state refusal (the arm is counted only)
        s = _sub(s, "- From level 2 on, after a NO-OP (nothing on the board changed), the next action(...) call in the same "
                    "snippet is refused once with StaleStateActionError: re-read the state first; a later call proceeds. A "
                    "RESET that changed nothing does not trigger this, and after a click that changed nothing only the same "
                    "click (same cell) is refused. An action that is not valid now raises InvalidActionError and executes "
                    "nothing.",
                 "- An action that is not valid now raises InvalidActionError and executes nothing.")
    t['system'] = s
    # C1 (IE 10-17; wording per signature mode)
    mode = loop_sig()
    if mode == 'exact':
        t['v65_loop_1'] = "The last {k} Python calls executed the same actions and produced identical reports."
        t['v65_loop_p'] = ("The last {n} Python calls repeated the same cycle of {p} calls {k} times: the same executed "
                           "actions and identical reports each time.")
    elif mode == 'acts':
        t['v65_loop_1'] = "The last {k} Python calls executed the same actions."
        t['v65_loop_p'] = "The last {n} Python calls repeated the same cycle of {p} calls {k} times: each cycle executed the same actions."
    else:
        t['v65_loop_1'] = ("The last {k} Python calls executed the same actions, and their reports were identical apart from "
                           "numbers and the listed continuations that ended in GAME OVER.")
        t['v65_loop_p'] = ("The last {n} Python calls repeated the same cycle of {p} calls {k} times: each cycle executed the "
                           "same actions, and the reports were identical apart from numbers and the listed continuations that ended in GAME OVER.")
    # OPENER (user via IE): the previous level's summary leads the restart opener; the others are named, not shown
    t['v65_prev_summary'] = "Your summary of level {level}, saved when you completed it; later levels can change its mechanics:"
    t['v65_summaries_not_shown'] = "Summaries of levels {levels} are not shown in this conversation."
    t['v65_summary_not_shown'] = "The summary of level {levels} is not shown in this conversation."
    t['v65_summary_missing'] = "No summary was saved for level {level}."
    return t


def v65_batch():
    """v6.5 C6: the batch stops at the second consecutive NO-OP, not at a BAND-ONLY CHANGE (ARC3_V65_BATCH=0: v6.4's rule)."""
    return os.environ.get('ARC3_V65_BATCH', '1') == '1'


def loop_sig():
    """v6.5 C1: the loop line's signature (norm: digits and the death ledger ignored; exact: _opener_key; acts: actions
    only; off)."""
    mode = os.environ.get('ARC3_V65_LOOP_SIG', 'norm')
    if mode not in ('norm', 'exact', 'acts', 'off'):
        raise ValueError('ARC3_V65_LOOP_SIG must be norm, exact, acts or off')
    return mode


def _v66_texts(t):
    """Harness v6.6 (2026-10-17; user with IE; apply_v66.py): the scoring rule as a fact in place of the action-minimising
    objective line, and the facts-only early no-action cue (noact_warn_0). After v6.5's texts; its anchor is the v6.0 line."""
    if os.environ.get('ARC3_V66_SCO', '1') == '1':           # v6.6g (user 10-18): the scoring fact alone can be switched off
        t['system'] = _sub(t['system'], "- Use as few in-game actions as possible while staying reliable.\n",
                           "- A cleared level scores in full when it takes no more actions than a hidden per-level baseline, and "
                           "less the further it goes over it; later levels count more; an uncleared level scores 0.\n")
    t['noact_warn_0'] = ("You have generated {k} tokens in this conversation since your last executed action. Actions not yet "
                         "executed on this level in this conversation: {untried}.")
    return t


def _v66_seed_texts(t):
    """Harness v6.6 item C (ARC3_V66_SEEDS=1; IE-agreed): the tools.py starter helpers in the system prompt."""
    s = t['system']
    s = _sub(s, "Coordinates are (row, col); the starter helpers' bounding boxes use exclusive stops (unlike `frame_diff` and "
                "`segmentation`).",
             "Coordinates are (row, col); grid.py's `bounding_box` uses exclusive stops, while tools.py, `frame_diff` and "
             "`segmentation` use inclusive bounds. A library function of the same name replaces a starter helper.")
    s = _sub(s, "`object_move(before, after, color, size=None, index=0)` gives (d_row, d_col).",
             "`object_move(before, after, color, size=None, index=0)` gives (d_row, d_col).\n"
             "  - tools.py: `crop(frame, r0, c0, r1, c1)` returns rows r0..r1 and columns c0..c1 (inclusive) with row and column "
             "numbers, and refuses windows over 32 x 32; `objects(frame, colors=None, min_size=1, connectivity=4)` lists "
             "connected same-symbol patches as dicts (`color`, `size`, inclusive `bbox`, `center`, `cells`), largest first "
             "(with connectivity 4, exactly the segmentation nodes), and `describe(frame, top=20, skip_largest=True)` gives "
             "one line per patch, omitting the largest region by area; `bfs(start, neighbors, is_goal, max_nodes=100000, "
             "max_seconds=6)` searches hashable states, where `neighbors(state)` yields `(move, next_state)` from your own "
             "model of the game, and returns `(moves or None, nodes, stop)` with stop `goal`, `exhausted`, `max_nodes` or "
             "`time`; `board_key(frame, region=None)` is a hashable key of the interior (rows and columns 4-59) or of an "
             "inclusive region, for visited sets; `band(frame)` gives the outer band's symbol runs per side and "
             "`band_diff(before, after)` its changed cells.")
    s = _sub(s, "instead of stepping by hand.", "instead of stepping by hand; the starter `bfs` runs one over your own step function.")
    t['system'] = s
    return t


def _c6r_texts(t):
    """C6R (user + IE, 2026-10-17; apply_v66.py): v6.4's batch stop and stale refusal stated as harness rules, facts only,
    the reason being wasted actions. Applies only with ARC3_V65_BATCH=0 (v6.4's stop); its anchors are v6.4's final texts."""
    s = t['system']
    s = _sub(s, "- A batch stops at the first action that changes nothing in the interior (a NO-OP or a BAND-ONLY CHANGE); the "
                "remaining actions are skipped and reported.",
             "- The harness (not the game) stops a batch at the first action that changes nothing in the interior (a NO-OP or a "
             "BAND-ONLY CHANGE), because the actions after it were planned for an interior change that did not happen, and "
             "executing them would waste actions; that action executes, the remaining actions are not executed, and the stop "
             "is reported.")
    if os.environ.get('ARC3_STALE_STATE_BLOCK', '0') == '1':
        s = _sub(s, "- From level 2 on, after a NO-OP (nothing on the board changed), the next action(...) call in the same "
                    "snippet is refused once with StaleStateActionError: re-read the state first; a later call proceeds.",
                 "- From level 2 on, after a NO-OP (nothing on the board changed), the harness (not the game) refuses the next "
                 "action(...) call in the same snippet once with StaleStateActionError, because positions and paths computed on "
                 "the assumption that the action had changed the board no longer describe it, and acting on them would waste "
                 "actions: re-read the state first; a later call proceeds.")
    t['system'] = s
    t['noop_stop'] = _sub(t['noop_stop'],
                          "Your batch was stopped: {action} (position {p} of {n}) executed but changed nothing in the interior, so "
                          "the remaining {m} action(s) were skipped: {skipped}. It is the last action in the executed list above; "
                          "only the actions before it changed anything in the interior. Re-locate from `current_frame` instead of "
                          "counting moves.",
                          "Your batch was stopped by the harness, not by the game: {action} (position {p} of {n}) executed but "
                          "changed nothing in the interior, and a batch stops at such an action because the actions after it were "
                          "planned for an interior change that did not happen, and executing them would waste actions. The "
                          "remaining {m} action(s) were not executed: {skipped}. It is the last action in the executed list above; "
                          "only the actions before it changed anything in the interior.")
    return t


def restart_kind():
    """v6.7 SOFT: ARC3_V67_RESTART_KIND, the kind of a harness restart: hard (default; today's: a RESET and the level's first
    board) or soft (a new conversation from the current board). The agent reads the same switch."""
    kind = os.environ.get('ARC3_V67_RESTART_KIND', 'hard').strip().lower()
    if kind not in ('hard', 'soft'):
        raise ValueError('ARC3_V67_RESTART_KIND must be hard or soft')
    return kind


def _v67_soft_texts(t):
    """v6.7 SOFT (ARC3_V67_RESTART_KIND=soft; agreed with IE 2026-10-18; apply_v67.py): the texts of a soft harness restart
    (a new conversation from the current board, without a RESET; the programs stay loaded), as '<key>_soft' beside the hard
    texts, which stay as they are (under the soft switch a restart is still hard in GAME OVER, on a dead board or at a
    caller's request). Facts only, in the hard texts' wording. After every earlier round: its anchors are the v6.6 texts."""
    r = _sub(t['attempt_summary_request'],
             "so the level restarts after this turn: a RESET, then a new conversation that starts from the level's first "
             "board.",
             "so it ends after this turn: a new conversation then continues level {level} from the current board, without a "
             "RESET (a harness restart, not a game event).")
    r = _sub(r, " At the restart, programs you saved on this level are set aside, and a program from an earlier level that you "
                "redefined on this level returns to its earlier version; programs from earlier levels stay loaded. Write down "
                "what your programs on this level computed rather than relying on them.",
             " Your programs stay loaded, including those saved on this level.")
    t['attempt_summary_request_soft'] = r
    t['attempt_summary_request_noact_soft'] = _sub(r, "tokens on level {level}, so it ends",
                                                   "tokens on level {level} without executing an action, so it ends")
    t['attempt_summary_request_position_soft'] = _sub(r, "This conversation has generated {tokens} tokens on level {level}, "
                                                         "so it ends", "This conversation has reached its length limit, so it ends")
    t['summary_turn_refusal_soft'] = _sub(t['summary_turn_refusal'], "because the level restarts after it.",
                                          "because this conversation ends after it.")
    t['summary_turn_reason_soft'] = "actions are not executed in the turn before this conversation ends"
    soft_end = "this conversation ends and a new conversation continues this level from the current board, without a RESET."
    t['noact_limit_clause_soft'] = _sub(t['noact_limit_clause'], "this conversation ends: the level resets and a new "
                                                                 "conversation starts from its first board.", soft_end)
    t['noact_warn_2_soft'] = _sub(t['noact_warn_2'], "this conversation ends and the level restarts from its first board.",
                                  soft_end)
    t['takeover_soft'] = ("This conversation continues level {level} from the current board, at step {step}. The previous "
                          "conversation ended after {tokens} tokens on this level, at a harness restart (not a game event). No "
                          "RESET was done, so the actions already taken on this level remain counted ({n} so far).")
    t['takeover_soft_history'] = ("In `history` and `transitions`, this level's entries are kept, including those of the "
                                  "previous conversation.")
    t['takeover_soft_history_cut'] = ("In `history` and `transitions`, {what} begins at the RESET of an earlier harness "
                                      "restart, at step {reset_step}; the entries after it are kept, including those of the "
                                      "previous conversation.")
    return t


def loop_on():
    """v6.7 LOOP: ARC3_V67_LOOP=1 turns the loop ladder on (default off). The agent reads the same switch."""
    return os.environ.get('ARC3_V67_LOOP', '0') == '1'


def _v67_loop_texts(t):
    """v6.7 LOOP (ARC3_V67_LOOP=1; agreed with IE 2026-10-18; apply_v67.py): the ladder's texts (the spec's wording): the
    loop-rule block (rung 1), the refusal of an identical call, the notes requests of a loop restart (rung 2 soft, rung 2
    hard in GAME OVER or on a dead board, rung 3) on the current notes request's instructions, and the opener line after a
    loop soft restart. After the soft texts, which ARC3_V67_LOOP=1 also brings."""
    t['loop_block'] = ("Loop rule (harness): your last {n} Python calls repeated one cycle of {p} call(s) {k} times, executing "
                       "{a} action(s) per cycle ({A} so far). Executed actions count toward this level's action total, and a "
                       "cleared level's score falls with the square of that total. The game changes only when an action "
                       "executes; it does not need actions to keep running. This conversation has used {t} of the {T} tokens "
                       "it can use on this level. A call that repeats this cycle is not executed; if the cycle continues, the "
                       "harness ends this conversation and starts a new one from the current board with your notes, without a "
                       "RESET.")
    t['loop_refused'] = "This call is identical to call {k} of the detected cycle and was not executed."
    mark = ' Actions are not executed in this turn.'

    def rest(key):
        assert t[key].count(mark) == 1, 'loop texts: the notes request anchor in ' + key
        return t[key][t[key].index(mark):]
    t['attempt_summary_request_loop'] = ("The last {n} Python calls repeated one cycle of {p} call(s), so this conversation "
                                         "ends after this turn (loop rule) and a new conversation starts from the current "
                                         "board of level {level}, without a RESET; actions already taken on this level still "
                                         "count." + rest('attempt_summary_request_soft'))
    t['attempt_summary_request_loop_hard'] = ("The last {n} Python calls repeated one cycle of {p} call(s), so level {level} "
                                              "restarts after this turn (loop rule): a RESET, then a new conversation from the "
                                              "level's first board." + rest('attempt_summary_request'))
    t['attempt_summary_request_loop_again'] = ("The last {n} Python calls repeated one cycle again after a loop-rule restart, "
                                               "so level {level} restarts after this turn: a RESET, then a new conversation "
                                               "from the level's first board." + rest('attempt_summary_request'))
    t['takeover_loop'] = ("This conversation starts from the board where the previous one stopped (level {level}, step "
                          "{step}); there was no RESET. The previous conversation ended under the loop rule: its last {n} "
                          "Python calls repeated one cycle of {p} call(s). The actions already taken on this level remain "
                          "counted ({count} so far).")
    return t


def texts67():
    """v6.7 TEXTS67: ARC3_V67_TEXTS=1 turns on the four factual text items (default off)."""
    return os.environ.get('ARC3_V67_TEXTS', '0') == '1'


def _v67_texts(t):
    """v6.7 TEXTS67 (ARC3_V67_TEXTS=1; agreed with IE 2026-10-18; apply_v67.py), every occurrence (recorded in
    CHANGES_v67.md):
      1. the C3 turn-based bullet (system) states that a game never ends or times out for lack of actions and that only the
         harness ends a conversation (at its token budget) or a game, merged into its one sentence;
      2. "keep playing until the harness ends the game" (system; one bullet, with and without v6.5's CLOCK clause) becomes
         "continue working on the level until the harness ends the game; no closing report is expected", the bullet led
         by that sentence so "no closing report is expected" is not said twice; "only the harness ends a game" leaves the
         Never-give-up bullet (item 1 states it);
      3. the sentence "The budget is restored as soon as an/one action executes." leaves noact_warn_1 and noact_warn_2
         (noact_warn_2_nolimit has none; the soft variant is derived after this);
      4. the yield resumption's "That is fine: it computed or inspected state." becomes the count "{t} tokens generated
         since the last executed action." (the agent fills {t}).
    Before the soft and loop texts, which derive from these."""
    s = t['system']
    s = _sub(s, "- The game is turn-based: it advances only when one of your actions executes, and nothing in the game changes "
                "while you think or between your calls.\n",
             "- The game is turn-based: it advances only when one of your actions executes, nothing in the game changes while "
             "you think or between your calls, and a game never ends or times out for lack of actions; only the harness ends a "
             "conversation at its token budget, or a game.\n")
    if "- No closing report is expected: keep playing until the harness ends the game, however little run time is left." in s:
        s = _sub(s, "- No closing report is expected: keep playing until the harness ends the game, however little run time is "
                    "left.",
                 "- Continue working on the level until the harness ends the game, however little run time is left; no closing "
                 "report is expected.")
    else:
        s = _sub(s, "- No closing report is expected: keep playing until the harness ends the game.",
                 "- Continue working on the level until the harness ends the game; no closing report is expected.")
    s = _sub(s, "- Never give up: only the harness ends a game. When you are stuck,", "- Never give up. When you are stuck,")
    t['system'] = s
    t['noact_warn_1'] = _sub(t['noact_warn_1'], " The budget is restored as soon as an action executes.", "")
    t['noact_warn_2'] = _sub(t['noact_warn_2'], " The budget is restored as soon as one action executes.", "")
    t['resumption'] = _sub(t['resumption'], "That is fine: it computed or inspected state.",
                           "{t} tokens generated since the last executed action.")
    return t


def refresh_on():
    """v6.7 REFRESH: ARC3_V67_REFRESH_LOGICAL=<positions> (> 0) turns the level-up refresh on (default 0 = off)."""
    return int(os.environ.get('ARC3_V67_REFRESH_LOGICAL', '0') or 0) > 0


def _v67_refresh_texts(t):
    """v6.7 REFRESH (IE SPEC UPDATE 3; apply_v67.py): the summary turn's request, refusal and refusal reason, and the new
    conversation's fact line (the spec's wording; its last clause shortened, since the opener shows the completed level's
    summary first and names the others)."""
    t['refresh_summary_request'] = ("This conversation has reached {logical} positions, so it ends after this turn: a new "
                                    "conversation then starts at level {level}'s first board, the current board, without a "
                                    "RESET (a harness mechanism, not a game event). Actions are not executed in this turn. In "
                                    "one python call, save your summary of level {prev} with save_summary(text) as asked "
                                    "above; it is shown in the new conversation, and your programs stay loaded.")
    t['refresh_turn_refusal'] = ("Not executed: actions are not executed in this turn, because this conversation ends after "
                                 "it. Save the summary with save_summary(text).")
    t['refresh_turn_reason'] = "actions are not executed in the turn before this conversation ends"
    t['refresh_summary_retry'] = ("No summary was saved ({reason}). Save it now with save_summary(text) in one python call, a "
                                  "text that begins with the line \"Rule L{level}:\"; actions are not executed in this turn.")
    t['takeover_refresh'] = ("This conversation starts at level {level}'s first board. The previous conversation completed "
                             "{done}; the harness started a new conversation at this level-up because the previous one had "
                             "reached {logical} positions. No RESET was done.")
    return t


def notes_on():
    """v6.7 NOTES: ARC3_V67_NOTES=1 turns the notes items on (default off)."""
    return os.environ.get('ARC3_V67_NOTES', '0') == '1'


def _v67_notes_texts(t):
    """v6.7 NOTES (ARC3_V67_NOTES=1; IE spec; apply_v67.py): the restart notes request asks for three parts under fixed
    headers (in the token, no-action and position requests; the soft and loop requests derive from these); the opener lines
    (the action count, the level-clear ledger, the coverage facts). Facts only; no directive to re-test."""
    for key in ('attempt_summary_request', 'attempt_summary_request_noact', 'attempt_summary_request_position'):
        t[key] = _sub(t[key], 'and then gives, in at most 10 items, what you established on this level, what you tried and what '
                              'happened, and the hypotheses still open.',
                      'and then has three parts under these headers: "Confirmed effects:" (what actions did, with the step '
                      'numbers where you saw it), "Untested ideas:" and "Negative claims:" (each claim that something does not '
                      'work or cannot be reached, with the step number of the board state it was tested in); at most 10 items '
                      'in all.')
    t['notes_action_count'] = "Actions so far on this level: {n}; all of them count for this level's score."
    t['notes_ledger'] = "Level {k} was completed by {action} after: {actions}."
    t['notes_ledger_first'] = "Level {k} was completed by {action}, the first action of its attempt."
    t['notes_coverage'] = ("Distinct boards (whole board compared) each action type was executed from on this level: "
                           "{pairs}.")
    t['notes_never'] = "Valid action types never executed on this level: {names}."
    return t


def score_text_on():
    """v6.7 SCORE: ARC3_V67_SCORE_TEXT=1 turns the complete scoring rule on (default off); it applies only with ARC3_V66_SCO=1
    (and the v6.6 texts), where it replaces v6.6's scoring sentence."""
    return (os.environ.get('ARC3_V67_SCORE_TEXT', '0') == '1' and os.environ.get('ARC3_V66_SCO', '1') == '1'
            and os.environ.get('ARC3_V66_TEXTS', '1') == '1')


def _v67_score_texts(t):
    """v6.7 SCORE (ARC3_V67_SCORE_TEXT=1 with ARC3_V66_SCO=1; IE spec; apply_v67.py): the complete scoring rule as a fact, in
    place of v6.6's qualitative sentence (the IE's exact text)."""
    t['system'] = _sub(t['system'], "- A cleared level scores in full when it takes no more actions than a hidden per-level "
                                    "baseline, and less the further it goes over it; later levels count more; an uncleared "
                                    "level scores 0.\n",
                       "- A cleared level scores 100 x (baseline / actions)^2, capped at 115, where the baseline is a hidden "
                       "per-level number of actions; every action on the level counts, including actions before a RESET or a "
                       "GAME OVER. A game's score is the average of its level scores weighted by level number (later levels "
                       "count more), capped at 100 times the weighted share of its levels that are cleared. An uncleared "
                       "level scores 0.\n")
    return t


def sumlookup_on():
    """v6.7x SUMLOOKUP (user decision 53): ARC3_V67_SUMLOOKUP=1 gives the sandbox summaries() (the saved level summaries)
    and its two texts (default off). The agent reads the same switch; sandbox_ext asks the agent."""
    return os.environ.get('ARC3_V67_SUMLOOKUP', '0') == '1'


def _v67x_sumlookup_texts(t):
    """v6.7x SUMLOOKUP: the system-prompt bullet after save_summary's, and the pointer in the opener line that names the
    summaries not shown (v6.5 OPENER's two forms, when present)."""
    t['system'] = _sub(t['system'], "and prints what was saved; at other times it saves nothing.\n",
                       "and prints what was saved; at other times it saves nothing.\n"
                       "- `summaries()` returns your saved level summaries as {level: text}.\n")
    if 'v65_summaries_not_shown' in t:
        t['v65_summaries_not_shown'] = _sub(t['v65_summaries_not_shown'], 'are not shown in this conversation.',
                                            'are not shown in this conversation; `summaries()` returns them.')
    if 'v65_summary_not_shown' in t:
        t['v65_summary_not_shown'] = _sub(t['v65_summary_not_shown'], 'is not shown in this conversation.',
                                          'is not shown in this conversation; `summaries()` returns it.')
    return t


def game_pool_on():
    """v6.7x GAMEBUDGET (user decision 55): ARC3_V67_GAME_POOL > 0, the run's token pool (v62/solver.py reads the same switch).
    In a pool run the level-up request keeps the sentence that the saved summary is shown again at a restart, also with the
    per-level budget off (ARC3_V61_RESTART_TOKENS=0): the no-action, loop and position restarts still show the summaries, and
    the texts stay those of the restart runs. Without it a run with the per-level budget off stops at a round-5l text anchor
    (as in the deployed tree)."""
    v71 = os.environ.get('ARC3_V71_ALLOC', '0')            # v7.1 ALLOCATION: no pool, the pool runs' texts (both arms alike)
    if v71 not in ('0', '1'):
        raise RuntimeError('ARC3_V71_ALLOC must be 0 or 1, not %r' % v71)
    return int(os.environ.get('ARC3_V67_GAME_POOL', '0') or 0) > 0 or v71 == '1'


def loop_rung2_kind():
    """v6.7x LOOPHARD (user decision 50): ARC3_V67_LOOP_RUNG2_KIND = soft (default, as built) | hard, the kind of the loop
    ladder's rung-2 restart. The agent reads the same switch."""
    kind = (os.environ.get('ARC3_V67_LOOP_RUNG2_KIND', 'soft') or 'soft').strip().lower()
    if kind not in ('soft', 'hard'):
        raise RuntimeError('ARC3_V67_LOOP_RUNG2_KIND must be soft or hard, not %r' % kind)
    return kind


def _v67x_loophard_texts(t):
    """v6.7x LOOPHARD: under a hard rung 2 the loop-rule block states the hard restart (its soft sentence would be false). The
    rung-2 request is the existing attempt_summary_request_loop_hard; the takeover is the hard restart's."""
    t['loop_block'] = _sub(t['loop_block'], "the harness ends this conversation and starts a new one from the current board "
                                            "with your notes, without a RESET.",
                           "the harness restarts this level: a RESET, then a new conversation from the level's first board "
                           "with your notes.")
    return t


def _v68_on(name):
    return os.environ.get('ARC3_V68_' + name, '0') == '1'


def v68_no_score():
    """v6.8 NO_SCORE: ARC3_V68_NO_SCORE=1 removes the v6.6 scoring bullet from the system prompt (default off)."""
    return _v68_on('NO_SCORE')


def v68_ledger():
    """v6.8 LEDGER: ARC3_V68_LEDGER=1, the v6.7 level-clear ledger without the other NOTES parts (the agent's part)."""
    return _v68_on('LEDGER')


def v68_never_types():
    """v6.8 NEVER_TYPES: ARC3_V68_NEVER_TYPES=1, the never-executed valid action types in restart openers (the agent's part)."""
    return _v68_on('NEVER_TYPES')


def v68_notes_neg():
    """v6.8 NOTES_NEG: ARC3_V68_NOTES_NEG=1, the notes request asks for the step of each concluded negative."""
    return _v68_on('NOTES_NEG')


def v68_loop_texts():
    """v6.8 LOOP_TEXTS: ARC3_V68_LOOP_TEXTS=1, the loop texts of the agreement (needs ARC3_V67_LOOP=1 and a hard rung 2)."""
    return _v68_on('LOOP_TEXTS')


def v68_loop_frame():
    """v6.8 LOOP_FRAME: ARC3_V68_LOOP_FRAME=1, a loop restart also needs the same interior board at each repetition's start
    (the agent's part; needs ARC3_V67_LOOP=1)."""
    return _v68_on('LOOP_FRAME')


def v68_refresh_texts():
    """v6.8 REFRESH_TEXTS: ARC3_V68_REFRESH_TEXTS=1, the refresh request and opener line of the agreement (with REFRESH)."""
    return _v68_on('REFRESH_TEXTS')


def _v68_no_score_texts(t):
    """v6.8 NO_SCORE: the v6.6 scoring bullet removed from the system prompt; only with ARC3_V66_SCO=1 (the bullet) and
    ARC3_V67_SCORE_TEXT off."""
    if (os.environ.get('ARC3_V66_SCO', '1') != '1' or os.environ.get('ARC3_V66_TEXTS', '1') != '1'
            or os.environ.get('ARC3_V67_SCORE_TEXT', '0') == '1'):
        raise RuntimeError('ARC3_V68_NO_SCORE=1 needs ARC3_V66_SCO=1 (with the v6.6 texts) and ARC3_V67_SCORE_TEXT off')
    t['system'] = _sub(t['system'], "- A cleared level scores in full when it takes no more actions than a hidden per-level "
                                    "baseline, and less the further it goes over it; later levels count more; an uncleared "
                                    "level scores 0.\n", "")
    return t


def _v68_ledger_texts(t):
    """v6.8 LEDGER / NEVER_TYPES (NOTES off): the ledger and never-executed texts exactly as v6.7 NOTES defines them."""
    notes = _v67_notes_texts(dict(t))
    for key in ('notes_ledger', 'notes_ledger_first', 'notes_never'):
        t[key] = notes[key]
    return t


def _v68_notes_neg_texts(t):
    """v6.8 NOTES_NEG: in the v6.6 notes request (the token, no-action and position requests; the soft and loop requests
    derive from them), the open hypotheses gain the step where each concluded negative was tested."""
    if notes_on():
        raise RuntimeError('ARC3_V68_NOTES_NEG=1 edits the v6.6 notes sentence, which ARC3_V67_NOTES=1 replaces; use one')
    for key in ('attempt_summary_request', 'attempt_summary_request_noact', 'attempt_summary_request_position'):
        t[key] = _sub(t[key], 'and the hypotheses still open.',
                      'and the hypotheses still open, noting, for anything you concluded does not work, the step where it '
                      'was tested.')
    return t


def _v68_refresh_texts(t):
    """v6.8 REFRESH_TEXTS: the refresh summary turn's request and the new conversation's harness line (the agreement's
    wording); the refusal, the pinned summary and the ledger stay."""
    t['refresh_summary_request'] = ("This conversation ends after this turn, and a new conversation starts at level {level}'s "
                                    "first board, the current board (no RESET; a harness step, not a game event). Actions are "
                                    "not executed in this turn. In one python call, save your summary of level {prev} with "
                                    "save_summary(text) as asked above; it is shown in the new conversation, and your programs "
                                    "stay loaded.")
    t['takeover_refresh'] = ("This conversation starts at level {level}'s first board. The harness started a new conversation "
                             "at this level-up; no RESET was done.")
    return t


def _v68_loop_texts(t):
    """v6.8 LOOP_TEXTS (needs ARC3_V67_LOOP=1 and ARC3_V67_LOOP_RUNG2_KIND=hard): the loop-rule block (facts only: no scoring
    clause, no token sentence; {level} filled by the agent), the rung-2 hard request and the rung-3 request with the
    agreement's leads on the current hard notes request, and the hard takeover's loop line. loop_refused is unchanged."""
    if not (loop_on() and loop_rung2_kind() == 'hard'):
        raise RuntimeError('ARC3_V68_LOOP_TEXTS=1 needs ARC3_V67_LOOP=1 and ARC3_V67_LOOP_RUNG2_KIND=hard')
    t['loop_block'] = ("Loop rule (harness): your last {n} Python calls repeated one cycle of {p} call(s) {k} times: each cycle "
                       "executed the same actions, and the reports were identical apart from numbers and the listed "
                       "continuations that ended in GAME OVER. The game changes only when an action executes; it does not need "
                       "actions to keep running. If the cycle continues, level "
                       "{level} restarts from its first board (a RESET) in a new conversation with your notes.")
    mark = ' Actions are not executed in this turn.'
    rest = t['attempt_summary_request'][t['attempt_summary_request'].index(mark):]
    t['attempt_summary_request_loop_hard'] = ("The last {n} Python calls repeated one cycle of {p} call(s), so level {level} "
                                              "restarts after this turn (loop rule): a RESET, then a new conversation from the "
                                              "level's first board." + rest)
    t['attempt_summary_request_loop_again'] = ("The last {n} Python calls repeated one cycle again after a loop-rule restart, "
                                               "so level {level} restarts after this turn: a RESET, then a new conversation "
                                               "from the level's first board." + rest)
    t['takeover_loop_hard'] = "The previous conversation on this level ended under the loop rule."
    return t


def v68_refresh_programs():
    """v6.8 REFRESH_PROGRAMS: ARC3_V68_REFRESH_PROGRAMS=1, the program selection after the library count line in the refresh
    opener (the agent's part; needs ARC3_V67_REFRESH_LOGICAL > 0)."""
    return _v68_on('REFRESH_PROGRAMS')


def _v68_refresh_programs_texts(t):
    """v6.8 REFRESH_PROGRAMS (the user, 18:02 UTC; the final rule from the refresh reviewer's data): the selection's headers
    (recent only; called-most first when ARC3_V68_REFRESH_PROG_FREQ > 0 adds any) and pointer; its program lines are
    program_list's format (- name(params): first docstring line)."""
    t['refresh_program_header'] = ('Your most recent programs, by your last call to each (or its writing, if not yet called) '
                                   '({k} of {n}):')
    t['refresh_program_header_most'] = ('Your most called programs, then your most recent ones by your last call to each (or its '
                                        'writing, if not yet called) ({k} of {n}):')
    t['refresh_program_pointer'] = '`programs()` lists all {n}, and `read_code(name)` shows any source.'
    return t


def v69_clear_step():
    """v6.9 CLEAR_STEP: ARC3_V69_CLEAR_STEP=1, the previous level's clearing step (two boards) in new-conversation openers at
    level >= 2 (restart takeovers and refreshes; decisions 82 and 87; the agent's and v62.solver's parts)."""
    return os.environ.get('ARC3_V69_CLEAR_STEP', '0') == '1'


def _v69_clear_step_texts(t):
    """v6.9 CLEAR_STEP (decision 82's agreed texts; the agent fills {k} the cleared level, {s} the clearing step, {s_before}
    = {s} - 1 and {action} the clearing action as the history shows it): the line, its "changed" form (the after-board is not
    the action's frame 0), the fallback line (no such board: the before board only) and the two captions."""
    t['clearstep_line'] = ("Level {k} was completed at step {s} by {action}. Two boards follow: the board {action} was executed "
                           "from, and the first board it produced.")
    t['clearstep_line_changed'] = ("Level {k} was completed at step {s} by {action}. Two boards follow: the board {action} was "
                                   "executed from, and the first changed board it produced.")
    t['clearstep_fallback'] = ("Level {k} was completed at step {s} by {action}; the board below is the one it was executed "
                               "from.")
    t['clearstep_before'] = "step {s_before}, level {k}: before {action}"
    t['clearstep_after'] = "step {s}, level {k}: first board after {action}; level {k} completed"
    t['clearstep_after_changed'] = "step {s}, level {k}: first changed board after {action}; level {k} completed"
    return t


def v69_inlevel_refresh():
    """v6.9 INLEVEL: ARC3_V69_INLEVEL_REFRESH, the logical position threshold of the in-level refresh (0 = off, the default; the
    agent reads the same switch)."""
    v = int(os.environ.get('ARC3_V69_INLEVEL_REFRESH', '0') or 0)
    if v < 0:
        raise ValueError('ARC3_V69_INLEVEL_REFRESH must be 0 (off) or a positive logical position')
    return v


def _v69_inlevel_texts(t):
    """v6.9 INLEVEL (decisions 94-99, IE's design points): the in-level refresh's texts. The notes request is the current
    attempt-notes request (v6.6's, or its NOTES / NOTES_NEG form) with its lead and its program sentence replaced, so it states
    this case's facts (the conversation ends; a new one continues the level from the current board; no RESET; the programs stay
    loaded) in the v6.8 refresh request's style; the refusal and its reason as the soft notes turn's; the save note; the new
    conversation's harness line (one line, facts only, in the v6.8 refresh line's style)."""
    r = _sub(t['attempt_summary_request'],
             "This conversation has generated {tokens} tokens on level {level}, so the level restarts after this turn: a RESET, "
             "then a new conversation that starts from the level's first board.",
             "This conversation ends after this turn, and a new conversation continues level {level} from the current board "
             "(no RESET; a harness step, not a game event).")
    r = _sub(r, " At the restart, programs you saved on this level are set aside, and a program from an earlier level that you "
                "redefined on this level returns to its earlier version; programs from earlier levels stay loaded. Write down "
                "what your programs on this level computed rather than relying on them.",
             " Your programs stay loaded, including those saved on this level.")
    t['inlevel_notes_request'] = r
    t['inlevel_turn_refusal'] = _sub(t['summary_turn_refusal'], "because the level restarts after it.",
                                     "because this conversation ends after it.")
    t['inlevel_turn_reason'] = "actions are not executed in the turn before this conversation ends"
    t['summary_saved_attempt_inlevel'] = _sub(t['summary_saved_attempt'], "They are shown to you after the restart.",
                                              "They are shown to you in the new conversation.")
    t['takeover_inlevel'] = ("This conversation continues level {level} from the current board, at step {step}. The harness "
                             "started this new conversation without a RESET; your earlier turns on this level are not in it.")
    return t


def v69_actrev_shadow():
    """v6.9 ACT-REV shadow: ARC3_V69_ACTREV_SHADOW=1, the executed-action cycle detector logs its would-fire decisions and never
    acts (the agent's part; no texts)."""
    return os.environ.get('ARC3_V69_ACTREV_SHADOW', '0') == '1'


def v69_restart_across():
    """v6.9 RAB RESTART_ACROSS (apply_rab69.py item R69RT): ARC3_V69_RESTART_ACROSS=1, the per-level budget restart counts the
    level's tokens over its conversations (the agent reads the same switch)."""
    return os.environ.get('ARC3_V69_RESTART_ACROSS', '0') == '1'


def _v69_across_texts(t):
    """v6.9 RAB RESTART_ACROSS: attempt_summary_request_across = the token restart's notes request in its final form (after
    every other variant) with its lead stating the count over the conversations; {since} is 'it began' or 'the harness last
    restarted it'."""
    t['attempt_summary_request_across'] = _sub(
        t['attempt_summary_request'], "This conversation has generated {tokens} tokens on level {level}, so the level restarts",
        "Your conversations on level {level} since {since} have generated {tokens} tokens, so the level restarts")
    return t


def v69_hide_reset():
    """v6.9 REX HIDE_RESET (decision 49, the user's RESET-exposure A/B; apply_rex69.py item R69HR): ARC3_V69_HIDE_RESET=1 (arm B)
    does not offer RESET to the model and refuses a model RESET (session_ext.step_env); run_v6 sets EXPOSE_RESET off with it,
    and the agent refuses to load with EXPOSE_RESET on. Off (default, arm A): today's harness."""
    return os.environ.get('ARC3_V69_HIDE_RESET', '0') == '1'


# v6.9 REX HIDE_RESET: the texts of arm B (facts only; for IE's review). The harness's own RESETs (the automatic restart after GAME
# OVER, harness restarts, the "without a RESET" facts) keep their texts in both arms.
HR69_RESET_REFUSAL = "RESET is not among the valid actions; this action() call executed nothing."
HR69_RESET_REFUSAL_BATCH = "RESET is not among the valid actions, so this action() call executed none of its {n} actions."
HR69_VALID_NOW = " Valid actions now: {valid}."
HR69_GAME_OVER_REFUSAL = ("Not executed: the game reported GAME OVER, so the harness repeated the level's restart (a RESET) "
                          "instead; this action() call executed nothing. The next turn shows the board.")
HR69_REPEAT = "Your next action(...) call makes the harness repeat the restart and executes nothing else."
HR69_SYSTEM = (
    ("(except after a RESET that ended in GAME OVER; the turn's message then says so)",
     "(if that restart itself ends in GAME OVER, the turn's message says so)"),
    ("A RESET that changed nothing does not trigger this, and after a click that changed nothing only the same click (same "
     "cell) is refused.",
     "After a click that changed nothing, only the same click (same cell) is refused."),
)


def _v69_hide_reset_texts(t):
    """v6.9 REX HIDE_RESET (arm B), after every other variant: the system prompt's two sentences that speak of the model's RESET
    (EXPOSE_RESET off has already removed its RESET bullet), the three GAME OVER-after-RESET texts (in arm B only a harness RESET
    can end in GAME OVER, and the harness repeats the restart at the model's next action call), and arm B's own texts (the step
    line's valid actions in GAME OVER, the refusal reasons and the two refusals, kept here for the render)."""
    s = t['system']
    for old, new in HR69_SYSTEM:
        s = _sub(s, old, new)
    t['system'] = s
    t['game_over_auto_reset'] = ("The game reports GAME OVER: the automatic restart's RESET itself ended in GAME OVER, and it was "
                                 "not repeated before this turn. " + HR69_REPEAT)
    t['game_over_harness_reset'] = ("The game reports GAME OVER: the RESET that began this conversation ended in GAME OVER. "
                                    + HR69_REPEAT)
    t['game_over_no_restart'] = ("The game reports GAME OVER, and no automatic restart followed: the last executed action was a "
                                 "RESET that ended in GAME OVER. " + HR69_REPEAT)
    t['hr69_valid_game_over'] = 'none while the game reports GAME OVER'
    t['hr69_reason_reset'] = 'RESET, which is not among the valid actions'
    t['hr69_reason_game_over'] = 'the game reported GAME OVER, so the harness repeated the restart instead'
    t['hr69_reason_game_over_stop'] = 'the game reported GAME OVER'
    t['hr69_reset_refusal'] = HR69_RESET_REFUSAL + HR69_VALID_NOW
    t['hr69_reset_refusal_batch'] = HR69_RESET_REFUSAL_BATCH + HR69_VALID_NOW
    t['hr69_game_over_refusal'] = HR69_GAME_OVER_REFUSAL
    return t



def v70_no_hard_restart():
    """v7.0 NO_HARD_RESTART (apply_swap70.py item S70NH): ARC3_V70_NO_HARD_RESTART=1 makes every harness hard restart an in-level
    refresh at the current board (v62/agent.py); the texts that state a harness restart of the level take the refresh form."""
    return os.environ.get('ARC3_V70_NO_HARD_RESTART', '0') == '1'


# v7.0 NO_HARD_RESTART: (old, new) pairs, each applied where present (for IE). The no-action cues and the notes requests take
# their existing _soft forms in the agent (the restart they announce is now a new conversation at the current board).
V70_NHR_LOOP = (
    ("If the cycle continues, level {level} restarts from its first board (a RESET) in a new conversation with your notes.",
     "If the cycle continues, this conversation ends and a new conversation continues level {level} from the current board "
     "with your notes, without a RESET."),
    ("the harness restarts this level: a RESET, then a new conversation from the level's first board with your notes.",
     "the harness ends this conversation and starts a new one from the current board with your notes, without a RESET."),
)
V70_NHR_SYSTEM = (
    ("A harness restart of the level is normally announced one turn ahead, with a request for notes.",
     "A new conversation started by the harness is normally announced one turn ahead, with a request for notes."),
    ("(when a level is completed, and before a level restarts)",
     "(when a level is completed, and before the harness starts a new conversation)"),
)


def _v70_nhr_texts(t):
    """v7.0 NO_HARD_RESTART, after every other variant: the loop-rule block's last sentence (the hard rung 2's restart becomes a
    new conversation at the current board) and the two system-prompt sentences on a harness restart of the level."""
    if 'loop_block' in t:
        for old, new in V70_NHR_LOOP:
            if old in t['loop_block']:
                t['loop_block'] = _sub(t['loop_block'], old, new)
    s = t['system']
    for old, new in V70_NHR_SYSTEM:
        if old in s:
            s = _sub(s, old, new)
    t['system'] = s
    return t


# v7.0 HARNESS REVISION (apply_v70h.py; V67_DESIGN 163-182): the texts of its items, each under its switch.
V70H_SWITCHES = ('ARC3_V70H_ATTEMPT_LEDGER', 'ARC3_V70H_IL_OPENER', 'ARC3_V70H_NO_CLOCK', 'ARC3_V70H_ALIASES',
                 'ARC3_V70H_ACTION_FACTS', 'ARC3_V70H_ACCESSORS', 'ARC3_V70H_BOARD_LEDGER', 'ARC3_V70H_REFUSAL_ONCE',
                 'ARC3_V70H_ACTREV', 'ARC3_V70H_RESIDENT_POINTER', 'ARC3_V70H_RESET0')


def _v70h_on(name):
    return os.environ.get(name, '0') == '1'


def v70h_any():
    return any(_v70h_on(n) for n in V70H_SWITCHES)


V70H_RESET0_REFUSAL = ("Not executed: this attempt has executed no action yet, so the level is already at its starting state; a "
                       "RESET would count as one action and change nothing. This action() call executed nothing.")
V70H_RESET0_REFUSAL_BATCH = ("Not executed: this attempt has executed no action yet, so the level is already at its starting "
                             "state; a RESET would count as one action and change nothing. This action() call executed none of "
                             "its {n} actions.")
V70H_TEXTS = {
    # 10 (ATTEMPT_LEDGER)
    'v70h_attempt_row': "{n} action{s}{how} ({m} changed nothing in the interior)",
    'v70h_end_game_over': ", ended in GAME OVER",
    'v70h_end_reset': ", ended with your RESET",
    'v70h_end_restart': ", ended at a harness restart",
    'v70h_attempt_total': "; {total} actions in all {k} ended attempt{ks} on this level",
    'v70h_reset_ledger': "Your RESET ended attempt {k} of level {level}. The attempts on this level so far lasted {lengths}.",
    # 11 (IL_OPENER)
    'v70h_il_head': "Your last {k} executed action{ks} on level {level} before this conversation, oldest first (step: action -> "
                    "the verdict your reports stated):",
    'v70h_il_more': "({m} earlier actions on this level are not listed; `history` has them.)",
    'v70h_il_line': "step {step}: {action} -> {verdict}",
    'v70h_il_auto': "step {step}: the automatic restart after GAME OVER (the level restarted from its initial state)",
    'v70h_il_first': "Level {level}'s first board (step {step}):",
    'v70h_il_change': "Change since the level's first board: cells that differ from it, in their current colour; everything "
                      "else dark navy.",
    'v70h_il_same': "The current board equals the level's first board.",
    # 13 (ALIASES)
    'v70h_import_helpers': "`{name}` is not a module here: the starter helpers are preloaded in every call; call them directly.",
    'v70h_import_library': "`{name}` is not a module here: your library programs are preloaded in every call (`programs()` lists "
                           "them).",
    'v70h_import_inspect': "To read a library program's source, call `read_code(name)`.",
    'v70h_import_time': "Each python call has 10 seconds of computation; bound loops and searches by counts (the starter `bfs` "
                        "takes `max_nodes` and `max_seconds`).",
    'v70h_import_global': "`{name}` is a runtime global, not a module: use it directly.",
    # 15 (BOARD_LEDGER)
    'v70h_bl_line': "Earlier actions on this level from a board identical to the current one (whole board compared): {acts}.",
    'v70h_bl_untried': "Not yet executed from it: {types}.",
    'v70h_bl_caveat': "State not shown on the board can differ between visits.",
    'v70h_revisit': "The whole board equals the board after step {s} of this attempt; the latest board new to this level came "
                    "{n} action{ns} ago.",
    # 19b (RESIDENT_POINTER)
    'v70h_resident_pointer': "Current board (unchanged): the latest board image above still shows it.",
    # 19c (RESET0)
    'v70h_reason_reset0': "a RESET in an attempt that had executed no action yet",
}
V70H_SYSTEM_ACCESSORS = (
    "`band(frame)` gives the outer band's symbol runs per side and `band_diff(before, after)` its changed cells.",
    "`band(frame)` gives the outer band's symbol runs per side and `band_diff(before, after)` its changed cells.\n"
    "  - history tools (they read `history`; nothing is interpreted): `level_trace(level=None, last=None)` lists this level's "
    "executed actions, oldest first, as dicts (`index` into `history`, `step`, `action`, `verdict`, `interior_n`, `rows`, `cols`, "
    "`automatic`); `attempt_trace(last=None)` does the same for the current attempt; `boards_seen(level=None, last=None)` gives "
    "{'boards': {board_key: [steps]}, 'distinct', 'last_new_step'} over the boards after this level's actions (it reads every "
    "board of the level; pass `last` to read fewer); `level_clears()` gives {level: {'step', 'action', 'before', 'after'}}: the "
    "board each completed level was cleared from and the next level's first board; `changes_between(i, j, limit=64)` gives the "
    "interior cells that differ between `history[i].frame` and `history[j].frame` ({'changed', 'changes' (row, col, old, new), "
    "'bbox' (inclusive), 'truncated'}); `view_offset()` gives (dy, dx), the summed `board_shift` of the current attempt's actions.")
V70H_SYSTEM_RESET0 = (
    "RESET must be the first game action in a Python snippet; inspection or computation may precede it.",
    "RESET must be the first game action in a Python snippet; inspection or computation may precede it. A RESET is not executed "
    "in an attempt that has executed no action yet, because the level is then already at its starting state.")
V70H_SYSTEM_ALIASES = (
    "`collections`, `itertools`, `math`, `re` and `heapq` are already imported in every call.",
    "`collections`, `itertools`, `math`, `re`, `heapq`, `json` and `random` are already imported in every call, and `Counter`, "
    "`deque` and `defaultdict` from collections are defined.")


# v7.2 item 11 REDESIGN (apply_v72.py; V67_DESIGN 201-202): the current attempt's boards in the in-level refresh opener.
V72_TEXTS = {
    'v72_ilb_head': "Boards of your current attempt on level {level} before this conversation, in step order ({n} of its {d} distinct "
                    "boards other than the current board):",
    'v72_ilb_head_all': "Boards of your current attempt on level {level} before this conversation, in step order (its {d} distinct "
                        "board{ds} other than the current board):",
    'v72_ilb_head_none': "Your current attempt on level {level} has no board other than the current board.",
    'v72_ilb_no_action': "No action has been executed in this attempt.",
    'v72_ilb_how_level': ", the level's first board",
    'v72_ilb_how_reset': ", after your RESET",
    'v72_ilb_how_auto': ", after the automatic restart",
    'v72_ilb_start': "Starting board of this attempt (step {step}{how}):",
    'v72_ilb_start_shown': "Starting board of this attempt (step {step}{how}): identical to the board after level {k}'s clearing "
                           "action, shown above.",
    'v72_ilb_start_is_current': "The starting board of this attempt (step {step}{how}) equals the current board.",
    'v72_ilb_board': "Board after step {step} ({action}):",
    'v72_ilb_actions_head': "Your last {k} executed action{ks} in this attempt, oldest first (step: action -> the verdict your "
                            "reports stated):",
    'v72_ilb_tail': "The current board is shown below.",
}


# v7.3 (apply_v73.py; the user 10-22): the clearing-step block's boards of the cleared level's final attempt; the line after a kept
# conversation's notes turn.
V73_TEXTS = {
    'v73_clear_line': "Level {k} was completed at step {s} by {action}. Boards of level {k}'s final attempt follow in step order ({sel}), "
                      "then the board {action} was executed from and the first {ch}board it produced.",
    'v73_clear_sel': "{n} of its {d} distinct boards before {action}",
    'v73_clear_sel_all': "its {d} distinct board{ds} before {action}",
    'v73_clear_start': "Starting board of level {k}'s final attempt (step {step}{how}):",
    'v73_clear_board': "Board after step {step} ({action}), level {k}:",
    'v73_how_level': ", the level's first board",
    'v73_how_reset': ", after your RESET",
    'v73_how_auto': ", after the automatic restart",
    'v73_kept': "This conversation continues on level {level}: the harness did not start a new conversation after your notes.",
    'v73_notes_age': "Your notes for this level were saved at step {n}; the attempt continued to step {m} before this new conversation.",
}


def _v70h_texts(t):
    """v7.0 harness revision, after every other variant: its new texts, and the system-prompt sentences of ACCESSORS (the history
    tools beside the starter helpers), RESET0 (the refused RESET) and ALIASES (the preloaded names)."""
    if _v70h_on('ARC3_V70H_RESET0') and os.environ.get('ARC3_V69_HIDE_RESET', '0') == '1':
        raise RuntimeError('ARC3_V70H_RESET0 needs RESET exposed (EXPOSE_RESET on, ARC3_V69_HIDE_RESET=0)')
    t.update(V70H_TEXTS)
    v72 = os.environ.get('ARC3_V72_IL_BOARDS', '0')
    if v72 not in ('0', '1'):
        raise RuntimeError('ARC3_V72_IL_BOARDS must be 0 or 1, not %r' % v72)
    if v72 == '1':                                     # v7.2 item 11 redesign (apply_v72.py)
        t.update(V72_TEXTS)
    if os.environ.get('ARC3_V73', '0') == '1' or int(os.environ.get('ARC3_V73_CLEAR_BOARDS', '0') or 0) > 0:
        t.update(V73_TEXTS)                            # v7.3 (apply_v73.py)
    s = t['system']
    if _v70h_on('ARC3_V70H_ACCESSORS'):
        s = _sub(s, *V70H_SYSTEM_ACCESSORS)
    if _v70h_on('ARC3_V70H_RESET0'):
        s = _sub(s, *V70H_SYSTEM_RESET0)
    if _v70h_on('ARC3_V70H_ALIASES'):
        s = _sub(s, *V70H_SYSTEM_ALIASES)
    t['system'] = s
    return t


# v7.1 PROMPT CLEANUP (apply_v71.py; the user 10-20, no A/B; V67_DESIGN 186, 189): coherence and redundancy edits of the system
# prompt, each anchored exactly once in the rendered system text (_sub).
V71_SYSTEM = (
    ('T2', "- Solve the whole game by clearing every level, not just the current screen. WIN means the whole game is solved.",
     "- Solve the whole game by clearing every level, not just the current screen."),
    ('T8', "When the goal is clear before the mechanics, let it decide which mechanics are worth probing, and stop probing once "
           "those are settled.",
     "When the goal is clear before the mechanics, let it decide which mechanics are worth probing."),
    ('T3', "- Never give up: only the harness ends a game. When you are stuck, revise your hypotheses and try a different "
           "approach.\n- No closing report is expected: keep playing until the harness ends the game.",
     "- Never give up, and write no closing report: only the harness ends a game. When you are stuck, revise your hypotheses "
     "and try a different approach."),
    ('T6', "- A batch also stops at the second REVERTED action in a row, and when the whole board returns to the level's starting "
           "board without a RESET after its interior had differed from it in more than 64 cells.",
     "- A batch also stops at the second REVERTED action in a row, and at a `start_return` (above)."),
    ('T5', "- From level 2 on, after a NO-OP (nothing on the board changed), the harness (not the game) refuses",
     "- From level 2 on, after a NO-OP, the harness (not the game) refuses"),
    ('T7', "You may call action(...) several times in one snippet, including inside loops; the state refreshes after each call.",
     "You may call action(...) several times in one snippet, including inside loops."),
)
V71_SYSTEM_RESET = ('T1', "Use only the actions listed as valid.\n- UP, DOWN,",
                    "Use only the actions listed as valid, and RESET (below).\n- UP, DOWN,")


def _v71_texts(t):
    """v7.1 prompt cleanup, after every other variant (ARC3_V71_TEXTS=1): T2-T8 always; T1 only with RESET exposed (with RESET
    hidden the actions line is already exact)."""
    s = t['system']
    for _name, old, new in V71_SYSTEM:
        s = _sub(s, old, new)
    if os.environ.get('ARC3_V69_HIDE_RESET', '0') != '1' and os.environ.get('EXPOSE_RESET', 'on') != 'off':
        s = _sub(s, V71_SYSTEM_RESET[1], V71_SYSTEM_RESET[2])
    t['system'] = s
    return t


# v7.5 POLISH (the user 10-23: the polish review's text batch and tool texts, REVIEW_2026-10-23.md Parts 3-4; BUILD_PLAN_polish75.md),
# after every other variant: each edit an exact replacement anchored once in the final texts (_sub), and the new texts the agent fills.
def v75_polish():
    v = os.environ.get('ARC3_V75_POLISH', '0')
    if v not in ('0', '1'):
        raise RuntimeError('ARC3_V75_POLISH must be 0 or 1, not %r' % v)
    return v == '1'


def v76_hybrid():
    """v7.6 HYBRID (the user 10-24): ARC3_V76_HYBRID, 0 (off, the default) or 1; the agent reads the same switch."""
    v = os.environ.get('ARC3_V76_HYBRID', '0')
    if v not in ('0', '1'):
        raise RuntimeError('ARC3_V76_HYBRID must be 0 or 1, not %r' % v)
    return v == '1'


def v77():
    """v7.7 (the user 10-24): ARC3_V77, 0 (off, the default) or 1; the agent, sandbox_ext, session_ext and patches read the same
    switch."""
    v = os.environ.get('ARC3_V77', '0')
    if v not in ('0', '1'):
        raise RuntimeError('ARC3_V77 must be 0 or 1, not %r' % v)
    return v == '1'


V75_TEXTS = {
    # TXT-4 / TXT-8 (agent._v75_lengths_text): consecutive attempts that ended the same way, grouped; a single ended attempt
    'v75_attempt_group': "{ns} actions{how} ({ms} changed nothing in the interior)",
    'v75_end_each_game_over': ", each ended in GAME OVER",
    'v75_end_each_reset': ", each ended with your RESET",
    'v75_end_each_restart': ", each ended at a harness restart",
    'v75_attempt_total_one': "; {total} action{s} in the one ended attempt on this level",
    'v75_reset_ledger_last5': "Your RESET ended attempt {k} of level {level}. The last 5 attempts on this level lasted {lengths}.",
    # TOOL-2: a runtime function called as a tool
    'v75_unknown_tool': "Unknown tool: {name}. The only tool is python: call {name}(...) inside its code.",
    # TOOL-3: subscripting a frame view or a history entry
    'v75_frame_subscript': "A frame is not a list: grid_cells(frame)[row][col] gives one cell, and frame.ascii gives its rows of symbols.",
    'v75_entry_subscript': ("A history entry is an object, not a list or a dictionary: use its .frame, .action, .result and .step "
                            "(the step of its board)."),
}
V75_SYSTEM_CROP = ("and refuses windows over 32 x 32;", "and refuses windows over 1,600 cells (40 x 40, or strips such as 25 x 64);")


def _v75_texts(t):
    """v7.5 POLISH texts: TXT-1 (a saved summary returns at the start of every later conversation; level restarts are off), TXT-6
    (summaries are asked for before the harness starts a new conversation), TXT-9 (the step counter counts the whole game), TXT-3
    (agreement in the library count), TOOL-4 (crop's 1,600-cell limit in the system prompt) and the new texts above. TXT-2 (the step
    line) is in agent.py; TOOL-4's helper and TXT-6's docstring in sandbox_ext.py."""
    if not r61():
        raise RuntimeError('ARC3_V75_POLISH needs the v6.1 texts (A1 + A2 and ARC3_V61_TEXTS=1)')
    t.update(V75_TEXTS)
    if summary():
        t['levelup'] = _sub(t['levelup'], 'The saved summary is shown to you again if this game restarts a level.',
                            'The saved summary is shown to you again at the start of every later conversation on this game.')
    t['summary_saved'] = _sub(t['summary_saved'], 'It is shown to you again if this game restarts a level.',
                              'It is shown to you again at the start of every later conversation on this game.')
    t['summary_not_requested'] = _sub(t['summary_not_requested'], 'when a level is completed and before a level restarts.',
                                      'when a level is completed and before the harness starts a new conversation.')
    t['takeover_history'] = _sub(t['takeover_history'], 'The step counter counts actions in the whole run,',
                                 'The step counter counts actions in the whole game,')
    t['library_count'] = _sub(t['library_count'], '`programs()` lists {them} with their levels,',
                              '`programs()` lists {them} with {their} level{ls},')
    if os.environ.get('ARC3_V66_SEEDS', '1') == '1':          # the tools.py line of the system prompt (v6.6 SEED1)
        t['system'] = _sub(t['system'], *V75_SYSTEM_CROP)
    return t


# v7.6 HYBRID (the user 10-24, V67_DESIGN 305 and 308): the rebase's summary turn and the RESET reminder. The request is the user's
# wording with "as text in your reply" (so the summary is not routed through save_summary, which is not accepted mid-level) and IE's
# precision the user adopted ("the instructions and its most recent part ... stay"); {frame} is the server's frame cap.
V76_TEXTS = {
    'v76_summary_request': ("After this turn the harness shortens this conversation: the instructions and its most recent part (about "
                            "{frame} tokens) stay. Write, as text in your reply, a summary of everything you know so far that you want to "
                            "keep. Actions are not executed in this turn."),
    'v76_summary_refusal': ("Not executed: actions are not executed in this turn, because the harness shortens this conversation "
                            "after it."),
    'v76_summary_reason': "actions are not executed in the turn before the harness shortens this conversation",
    'v76_reset_line': "Use RESET only when you are stuck, and use it sparingly.",
}


def _v76_texts(t):
    """v7.6 HYBRID texts: the rebase's summary request, its action refusal and the refusal reason; the RESET reminder."""
    t.update(V76_TEXTS)
    return t


# v7.7 (the user 10-24, V67_DESIGN 312): the level-up summary without the 10-item limit (A); the previous level's summary shown to
# update (B); summaries(query) in the system prompt (C); the board_shift fact as the whole interior's shift (the session_ext fix).
V77_LIMITS = (', in at most 10 items, the game mechanics', ' the game mechanics')   # the production form (v7.5)
V77_LIMITS_OLD = ('summarise as a bullet list of at most 10 items the game mechanics', 'summarise as a bullet list the game mechanics')
V77_NOTES_LIMITS = ((', in at most 10 items, what you established on this level', ' what you established on this level'),
                    ('the step number of the board state it was tested in); at most 10 items in all.',
                     'the step number of the board state it was tested in).'))           # v6.7 NOTES's form (off in production)
V77_SUMMARIES = ("- `summaries()` returns your saved level summaries as {level: text}.",
                 "- `summaries()` returns your saved level summaries as {level: text}; `summaries(query)` returns only their lines "
                 "that contain the query text, ignoring case, as {level: [lines]}.")
V77_BUDGET = ("- If your actions change only cells of the outer band, their verdict is BAND-ONLY CHANGE, not NO-OP.",
              "- If your actions change only cells of the outer band, their verdict is BAND-ONLY CHANGE, not NO-OP.\n- In some games "
              "the outer band shows a budget for the attempt: a bar or counter that changes as actions are spent; the attempt can end "
              "when it runs out.")
V77_REGIONS = ("`interior_changed` ({'n', 'rows', 'cols'}: how many interior cells changed and their row and column span; n is 0 when "
               "none did).",
               "`interior_changed` ({'n', 'rows', 'cols', 'regions'}: how many interior cells changed and their row and column span, "
               "n 0 when none did; `regions` lists the separate changed regions, largest first, at most 3, each {'n', 'rows', "
               "'cols', 'colors'} with its most common colour changes as [from, to, count]; `more` and `more_cells` count the "
               "others).")
V77_SHIFT = ("shift of the board content explains at least 90% of 64 or more changed cells",
             "shift of the whole interior explains at least 90% of 64 or more changed cells and the rest of the interior agrees "
             "with it (a part of the board that moves is not reported)")


def _v77_texts(t):
    """v7.7 texts: (A) the level-up request without "in at most 10 items" (either form; none may remain); (B) v77_prev_summary
    and v77_prev_missing; (C) the summaries(query) sentence in the system prompt (when summaries() is documented there); the
    board_shift fact's description (every occurrence); the outer-band budget sentence and the regions of `interior_changed`
    (when their anchors are there); v77_attempt_steps (lever 5)."""
    lu = t['levelup']
    for old, new in (V77_LIMITS, V77_LIMITS_OLD):
        if lu.count(old) == 1:
            lu = lu.replace(old, new)
            break
    if 'at most 10 items' in lu:
        raise RuntimeError('v7.7: the level-up request still has a 10-item limit')
    t['levelup'] = lu
    t['v77_prev_summary'] = "The new summary updates your summary of level {level}, saved when you completed it:"
    t['v77_prev_missing'] = "No summary was saved for level {level}."
    t['v77_attempt_steps'] = "Attempt {k} was {steps}."
    for k78 in ('summary_tag_closed', 'summary_tag_open_level'):   # v7.8 F2 (IE: both refusals): which level's summary is requested now
        if '{window}' not in t.get(k78, '{window}'):
            t[k78] = t[k78] + ' The summary requested now is for level {window}.'
    t['v77_notes_prev_summary'] = "The notes can update your summary of level {level}, saved when you completed it:"
    t['v77_ilb_range'] = ("Your current attempt (attempt {k} of level {level}) runs from its starting board at step {a} to the current "
                          "board at step {b}.")
    t['v77_ilb_head'] = ("Below, after your last actions, are boards of your current attempt on level {level} before this conversation, "
                         "in step order: its starting board and {n} board{ns} at evenly spaced steps, each captioned with its step, "
                         "then the current board.")
    t['v77_ilb_head_start'] = ("Below, after your last actions, are the starting board of your current attempt on level {level} and then "
                               "the current board.")
    for key in list(t):                                         # the notes requests (the user 10-24): no 10-item limit either
        if isinstance(t[key], str):
            for old, new in V77_NOTES_LIMITS:
                t[key] = t[key].replace(old, new)
    left = sorted(k for k, v in t.items() if isinstance(v, str) and 'at most 10 items' in v)
    if left:
        raise RuntimeError('v7.7: a 10-item limit is left in %s' % ', '.join(left))
    for old, new in (V77_BUDGET, V77_REGIONS):                  # the band budget sentence (the user 10-24); the regions field
        if t['system'].count(old) == 1:
            t['system'] = t['system'].replace(old, new)
    if '`summaries()`' in t['system']:
        t['system'] = _sub(t['system'], *V77_SUMMARIES)
    if V77_SHIFT[0] in t['system']:
        t['system'] = t['system'].replace(*V77_SHIFT)
    return t
