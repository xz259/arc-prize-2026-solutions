# history70.py: read-only helpers over `history` (v7.0 item 14, ARC3_V70H_ACCESSORS=1). They read the same records the turn
# reports print; nothing here is interpreted. Coordinates are (row, col); the interior is rows and columns 4-59.

def _h70_result(e):
    r = getattr(e, 'result', None)
    return r if isinstance(r, dict) else {}


def _h70_verdict(r):
    """The first words of the verdict the reports print (the host's _action_verdict, without its detail)."""
    if r.get('game_over') and not r.get('automatic') and str(r.get('action_display') or r.get('action_name') or '') == 'RESET':
        return 'GAME OVER'
    if r.get('game_over'):
        return 'LEVEL RESTARTED'
    if r.get('run_complete') or r.get('done'):
        return 'RUN COMPLETE'
    if r.get('level_completed'):
        return 'LEVEL COMPLETED'
    if r.get('reverted'):
        return 'REVERTED'
    if r.get('gameplay_changed') is False:
        edge = r.get('v61_edge') or {}
        return 'BAND-ONLY CHANGE' if ((edge.get('n') if isinstance(edge, dict) else 0) or r.get('board_changed')) else 'NO-OP'
    return 'effect' if r.get('gameplay_changed') is True else 'executed'


def _h70_level_of(r):
    """The level the action was executed on (a completing action reports the next level)."""
    lv = r.get('level')
    if isinstance(lv, int) and r.get('level_completed'):
        return lv - 1
    return lv


def _h70_items(level=None, start=None):
    """The executed actions of `level` (default: the current level) as dicts; `start`: the first history index to read (the host's
    level / attempt start when known)."""
    current = level is None
    if current:
        level = current_frame.level
        ls = getattr(history, '_v6_level_start', None)        # the host's index of this level's first board (paged history)
        if start is None and isinstance(ls, int):
            start = ls + 1
    out = []
    for i in range(max(1, int(start or 1)), len(history)):
        e = history[i]
        r = _h70_result(e)
        if _h70_level_of(r) != level:
            continue
        ic = r.get('interior_changed') or {}
        out.append({'index': i, 'step': r.get('action_num'), 'action': str(getattr(e, 'action', '')),
                    'verdict': _h70_verdict(r), 'interior_n': ic.get('n') if isinstance(ic, dict) else None,
                    'rows': ic.get('rows') if isinstance(ic, dict) else None,
                    'cols': ic.get('cols') if isinstance(ic, dict) else None, 'automatic': bool(r.get('automatic'))})
    return out


def level_trace(level=None, last=None):
    """This level's (or `level`'s) executed actions, oldest first: dicts with index (into history), step, action, verdict,
    interior_n, rows, cols, automatic (True for the automatic restart after a GAME OVER)."""
    items = _h70_items(level)
    return items[-int(last):] if last else items


def _h70_attempt_start(items):
    start = 0
    for k, it in enumerate(items):
        r = _h70_result(history[it['index']])
        if it['automatic'] or (it['action'].upper() == 'RESET' and r.get('board_changed')):
            start = k + 1 if it['automatic'] else k
    return start


def attempt_trace(last=None):
    """The current attempt's executed actions (from the level start, the latest automatic restart or the latest RESET that
    changed the board), as level_trace gives them."""
    ats = getattr(history, '_v6_attempt_start', None)         # the host's index of this attempt's first board (paged history)
    if isinstance(ats, int):
        items = _h70_items(None, start=ats + 1)
    else:
        items = _h70_items(None)
        items = items[_h70_attempt_start(items):]
    return items[-int(last):] if last else items


def boards_seen(level=None, last=None):
    """{'boards': {board_key: [steps]}, 'distinct': n, 'last_new_step': s} over the boards after this level's actions (the
    interior key of tools.py's board_key). It reads every board of the level; pass last=N to read only the latest N."""
    items = _h70_items(level)
    if last:
        items = items[-int(last):]
    boards, last_new = {}, None
    for it in items:
        k = board_key(history[it['index']].frame)
        if k not in boards:
            boards[k] = []
            last_new = it['step']
        boards[k].append(it['step'])
    return {'boards': boards, 'distinct': len(boards), 'last_new_step': last_new}


def level_clears():
    """{level: {'step', 'action', 'before', 'after'}} for each completed level: the board the clearing action was executed from
    and the board after it (the next level's first board)."""
    out = {}
    for i in range(1, len(history)):
        r = _h70_result(history[i])
        if r.get('level_completed'):
            lv = _h70_level_of(r)
            out[lv] = {'step': r.get('action_num'), 'action': str(getattr(history[i], 'action', '')),
                       'before': history[i - 1].frame, 'after': history[i].frame}
    return out


def changes_between(i, j, limit=64):
    """The interior cells (rows and columns 4-59) that differ between history[i].frame and history[j].frame:
    {'changed', 'changes' [(row, col, old, new)] (at most `limit`), 'bbox' [r0, c0, r1, c1] inclusive or None, 'truncated'}."""
    a = history[i].frame.ascii.split('\n')
    b = history[j].frame.ascii.split('\n')
    changes = []
    for r in range(4, min(len(a), len(b)) - 4):
        ra, rb = a[r], b[r]
        for c in range(4, min(len(ra), len(rb)) - 4):
            if ra[c] != rb[c]:
                changes.append((r, c, ra[c], rb[c]))
    bbox = ([min(x[0] for x in changes), min(x[1] for x in changes), max(x[0] for x in changes), max(x[1] for x in changes)]
            if changes else None)
    return {'changed': len(changes), 'changes': changes[:int(limit)], 'bbox': bbox, 'truncated': len(changes) > int(limit)}


def view_offset():
    """(dy, dx): the sum of the `board_shift` measurements (rows down, columns right) of the current attempt's actions."""
    dy = dx = 0
    for it in attempt_trace():
        sh = _h70_result(history[it['index']]).get('board_shift')
        if isinstance(sh, dict):
            dy += int(sh.get('dy') or 0)
            dx += int(sh.get('dx') or 0)
    return (dy, dx)
