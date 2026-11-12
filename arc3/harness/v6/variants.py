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
    return t
