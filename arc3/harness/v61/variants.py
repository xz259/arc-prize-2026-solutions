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
             "REVERTED (it animated, then the board returned to its previous state: usually a blocked or rejected "
             "move), LEVEL RESTARTED (the attempt ended here; the flag is `game_over`), LEVEL COMPLETED or RUN COMPLETE.")
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
                   'predict, say so.')
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
            "state (REVERTED): {k} cells changed and changed back{where}. That usually means a blocked or rejected move; "
            "`last_animation_timeline` shows what happened.")
    t['animation'] = '\n'.join(a)

    # new opener lines (v6.1)
    t['revert_stop'] = ("Your batch was stopped after two REVERTED actions in a row ({a1}, then {a2}): each animated and "
                        "left the board as it was, which usually means those moves are blocked.{skipped_part}")
    t['hazard_restart'] = ("The attempt ended at {action} without a game-over flag: the whole board, edge included, "
                           "returned to this level's starting state, as after a restart. That attempt lasted {n} actions."
                           "{skipped_part}")
    t['level_start_soft'] = ("After {action}, the board area matches this level's starting board: a return to the start, "
                             "or a restart the edge does not show.{skipped_part}")
    t['call_cap'] = ("Your call reached the limit of %d executed actions; {m} requested action(s) were not executed. "
                     "Continue from the current board." % CALL_CAP)
    t['run_clock'] = "Run time left: {minutes} min (the whole run, not an attempt timer)."
    t['no_action'] = ("You have made {k} Python calls in a row without executing an action on this board. Actions not "
                      "yet executed on this level: {untried}.")
    t['library_count'] = ("Your program library has {n} programs, written on levels {levels}; `programs()` lists them "
                          "with their levels, and `read_code(name)` shows any source.")
    t['edge_line'] = "Edge band in your previous call: {text}."
    t['takeover'] = "The game is at level {level} (levels 1-{prev} completed)."
    t['takeover_l1'] = "The game is at level 1."
    t['restart_summaries'] = ("Mechanics you summarised when completing earlier levels; later levels can change "
                              "them:")
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
    return t
