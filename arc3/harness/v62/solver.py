"""Harness v6 solver: dfranzen's HarnessSolver (10882e3) with the evaluation logs our token-matched scripts read and the
per-game session lifecycle (BUILD_PLAN component 9).
  - every executed action (including the automatic RESET after a death) is appended to
    <action_log_root>/game-NNN/log.jsonl as {kind: 'action', game_id, epoch, payload: {result}} (the control's
    prolong/ format: g7_common.py counts level completions against the model-turn token clock);
  - when a game's play() returns, its server session is closed (frees the KV of finished games).
His local-server management stays off (our server is external); everything else is his.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inference.framework import solver as sv

_LOCK = threading.Lock()


@dataclass
class V6Solver(sv.HarnessSolver):
    label: str = "V6Solver"
    action_log_root: Path | None = None
    game_index: dict = field(default_factory=dict)


_original_execute = sv._HarnessGameSession._execute_action
_original_play = sv._HarnessGameSession.play
RESULT_KEYS = ("executed", "action_num", "level", "score", "state", "board_changed", "gameplay_changed", "no_op",
               "done", "level_completed", "game_over", "run_complete", "action_name", "action_data", "action_display",
               "animation", "automatic", "interior_transient")


# v6.9 CLEAR_STEP (V67_DESIGN decisions 82 and 87, 2026-10-18; apply_v67x.py item V69CS): ARC3_V69_CLEAR_STEP=1 keeps, at every
# level clear that leads to another level (level_completed without run_complete), the game session's record of the clearing
# step, which the agent's new-conversation openers (restart takeovers and refreshes) show: v69_clears[k], k the level of the
# board the clearing action was executed from = the step, the action, that board (the history entry before the clear), the first
# of the action's raw frames (the engine renders one per internal step; the last is the next level's first board) that differs
# from that board and from the last frame, its index and the frame count. taaf's safety frame (an all-14 board it puts before a
# single-frame level transition) is not a game board and is skipped (v69_is_filler, the rule's one place). Stored at the clear
# on the game session, which outlives the conversations, so restarts and refreshes read it back. Off (default): nothing runs.
CLEAR_STEP69 = __import__('os').environ.get('ARC3_V69_CLEAR_STEP', '0') == '1'


def _v69_grid(frame):
    rows = frame.tolist() if hasattr(frame, 'tolist') else frame
    return tuple(tuple(int(c) for c in row) for row in rows)


V69_FILLER = 14     # the colour of taaf's safety frame (taaf.game.Game.execute_action: np.full((h, w), 14))


def v69_is_filler(frames, i, grid=None):
    """v6.9 CLEAR_STEP, the filler rule (its one place: v69_clear_frame and the frame-check replay call it): True when raw
    frame i is taaf's safety frame, which taaf.game.Game.execute_action puts first when a level transition returned a single
    frame: index 0 of a two-frame list, every cell V69_FILLER. `grid` is frame i already as a tuple grid (optional)."""
    if i != 0 or len(frames) != 2:
        return False
    g = _v69_grid(frames[0]) if grid is None else grid
    return bool(g) and all(c == V69_FILLER for row in g for c in row)


def v69_clear_frame(before, frames):
    """v6.9 CLEAR_STEP, the chooser: (index, grid) of the first raw frame of a level-clearing action that is not taaf's filler
    (v69_is_filler) and differs from `before` (the board the action was executed from) and from the last frame (the next
    level's first board); (None, None) when no frame does."""
    frames = list(frames or [])
    if not frames:
        return None, None
    last = _v69_grid(frames[-1])
    for i, f in enumerate(frames[:-1]):
        g = _v69_grid(f)
        if v69_is_filler(frames, i, g):
            continue
        if g != before and g != last:
            return i, g
    return None, None


def _v69_note_clear(session, payload):
    """v6.9 CLEAR_STEP: the record of the level clear just executed (see CLEAR_STEP69) in session.__dict__['v69_clears'];
    receipts in the agent's stats (the session's v65_stats link): clear_step_records, clear_step_record_errors. A failure is
    also kept on the session (v69_clear_errors, v69_clear_last_error) and never stops the action."""
    stats = session.__dict__.get('v65_stats')
    stats = stats if isinstance(stats, dict) else {}
    try:
        hist = getattr(session, 'history_entries', None) or []
        if len(hist) < 2:
            return
        prev, entry = hist[-2], hist[-1]
        before = tuple(tuple(int(c) for c in row) for row in prev.frame.grid)
        raw = getattr(getattr(getattr(session.game, 'current_state', None), 'raw', None), 'frame', None)
        raw = list(raw) if raw is not None else []
        index, after = v69_clear_frame(before, raw)
        k = int(prev.frame.level)
        session.__dict__.setdefault('v69_clears', {})[k] = dict(
            level=k, step=payload.get('action_num'), action=payload.get('action_display'), before=before,
            before_step=prev.frame.step, after=after, index=index, frames=len(raw), next_level=entry.frame.level)
        stats['clear_step_records'] = stats.get('clear_step_records', 0) + 1
    except Exception as exc:  # noqa: BLE001
        session.__dict__['v69_clear_errors'] = session.__dict__.get('v69_clear_errors', 0) + 1
        session.__dict__['v69_clear_last_error'] = '%s: %s' % (type(exc).__name__, str(exc)[:200])
        stats['clear_step_record_errors'] = stats.get('clear_step_record_errors', 0) + 1


def _execute_action(self, action, **kwargs):
    payload = _original_execute(self, action, **kwargs)
    if CLEAR_STEP69 and payload.get('level_completed') and not payload.get('run_complete'):   # v6.9 CLEAR_STEP: the record
        _v69_note_clear(self, payload)
    root = getattr(self.solver, 'action_log_root', None)
    run = self.game.game_run
    if root is not None and run is not None:
        gid = run.game_id
        i = self.solver.game_index.get(gid, self.game_index)
        d = Path(root) / f'game-{i:03d}'
        result = {k: payload[k] for k in RESULT_KEYS if k in payload}
        line = json.dumps(dict(kind='action', game_id=gid, epoch=time.time(), payload=dict(result=result)),
                          ensure_ascii=False)
        with _LOCK:
            d.mkdir(parents=True, exist_ok=True)
            with open(d / 'log.jsonl', 'a') as f:
                f.write(line + '\n')
    return payload


STAGGER = float(__import__('os').environ.get('ARC3_V62_START_STAGGER', '3'))   # v6.2: seconds between game starts

# v6.7x GAMEBUDGET (user decision 55, 2026-10-18; apply_v67x.py): ARC3_V67_GAME_POOL > 0 is the run's generated-token pool for all
# games together (0 = off). run_v6 calls gamebudget_install(games, writer): each game of the run's list gets the cap POOL // the
# number of games. The count is the analyzer's generated tokens (each response's completion tokens, as the per-level restart
# budget counts them, summed over the game's conversations), checked through the solver's per-game token limit (should_stop:
# at every loop boundary and before every model request), so a game stops at the first check after its count reaches its cap
# and finishes as games finish at the run's end (its lane frees); its solver note records budget_exhausted (the taaf run state
# stays gave_up, the state the scorer finalizes). A game that finishes otherwise (won, all levels done, crashed, given up) with
# tokens left under its cap releases them in equal shares to the games not yet finished, unstarted ones included; caps only
# grow. Every change holds one lock (games finish in parallel threads).
# v6.9 RAB RESERVE (the RESET A/B, decisions 111 and 117; apply_rab69.py item R69RV): ARC3_V69_POOL_RESERVE > 0 tokens held back
# from the caps (the caps stay pool // games); the first time any game's count reaches its cap, the reserve is released once by
# the redistribution's rule (equal shares to the games not finished and not exhausted, that game and unstarted games included;
# the remainder's tokens one each to the first of them in the run's list); a game still under its new cap goes on. Receipts:
# 'redistribution' events with source 'reserve', one 'reserve' event, the summary's reserve / reserve_released / reserve_trigger.
GAMEBUDGET = None


class GameBudget:
    """v6.7x GAMEBUDGET: the pool, the per-game caps, the redistribution and the receipts (one lock for all of them)."""

    def __init__(self, pool, games, writer=None, reserve=0):
        self.games = list(dict.fromkeys(str(g) for g in games))
        self.pool = int(pool)
        self.base = self.pool // max(1, len(self.games))
        self.unallocated = self.pool - self.base * len(self.games)     # the rounding remainder (fewer tokens than games)
        self.cap = {g: self.base for g in self.games}
        self.received = {g: 0 for g in self.games}
        self.released = {g: 0 for g in self.games}
        self.used, self.ended, self.exhausted = {}, {}, set()
        self.lock = threading.Lock()
        self.writer = writer
        self.reserve = max(0, int(reserve or 0))         # v6.9 RAB RESERVE: held back; released once (see _release_reserve)
        self.reserve_left, self.reserve_released, self.reserve_trigger = self.reserve, 0, None

    def _log(self, gid, **payload):
        if self.writer is not None:
            self.writer(gid)('v67-gamebudget', payload)

    def start(self, gid):
        """A game's play begins (it may still wait for a lane): the 'start' event with its cap, shares received included."""
        with self.lock:
            cap = self.cap.setdefault(gid, self.base)
            self.received.setdefault(gid, 0)
            self.released.setdefault(gid, 0)
            self._log(gid, reason='start', cap=cap, received=self.received[gid], pool=self.pool, games=len(self.games),
                      **({'reserve': self.reserve} if self.reserve else {}))

    def reached(self, gid, spent):
        """True when the game's count has reached its cap. The first time marks the game exhausted, with the 'exhausted'
        event (tokens, cap, overshoot)."""
        with self.lock:
            if gid in self.exhausted:
                return True
            cap = self.cap.get(gid)
            if cap is None or gid in self.ended or int(spent) < cap:
                return False
            if self.reserve_left > 0:                    # v6.9 RAB RESERVE: the first cap reached releases the reserve
                self._release_reserve(gid, int(spent), cap)
                cap = self.cap.get(gid)
                if int(spent) < cap:
                    return False
            self.exhausted.add(gid)
            self._log(gid, reason='exhausted', tokens=int(spent), cap=cap, overshoot=int(spent) - cap)
            return True

    def _release_reserve(self, gid, spent, cap):
        """v6.9 RAB RESERVE (the lock held): the reserve, once, in equal shares to the games not finished and not exhausted (the
        game whose cap was reached included; unstarted ones too), the remainder's tokens one each to the first of them in the
        run's list, as the redistribution hands out a finished game's leftover; each receiver's cap grows ('redistribution'
        events with source 'reserve'; one 'reserve' event on the game whose cap was reached)."""
        receivers = [g for g in self.games if g not in self.ended and g not in self.exhausted]
        if not receivers:
            return
        amount, self.reserve_left = self.reserve_left, 0
        share, extra = divmod(amount, len(receivers))
        for i, g in enumerate(receivers):
            add = share + (1 if i < extra else 0)
            if add:
                self.cap[g] += add
                self.received[g] = self.received.get(g, 0) + add
                self._log(g, reason='redistribution', source='reserve', amount=add, cap=self.cap[g])
        self.reserve_released, self.reserve_trigger = amount, gid
        self._log(gid, reason='reserve', amount=amount, receivers=len(receivers), tokens=spent, cap_before=cap,
                  cap=self.cap.get(gid))

    def finish(self, gid, spent, how):
        """A game's end (once): its count and how it ended ('exhausted', 'run_end', or the run state: won, gave_up, crashed).
        An end other than exhaustion or the run's end releases the tokens left under the cap to the games not yet finished
        (unstarted included; an exhausted game has stopped) in equal shares, the remainder's tokens one each to the first of
        them in the run's list; each receiver's cap grows ('redistribution' event). Returns the game's receipts."""
        with self.lock:
            if gid in self.ended:
                return None
            spent = max(0, int(spent))
            cap = self.cap.setdefault(gid, self.base)
            self.used[gid], self.ended[gid] = spent, how
            left = max(0, cap - spent)
            receivers = ([g for g in self.games if g != gid and g not in self.ended and g not in self.exhausted]
                         if left and how not in ('exhausted', 'run_end') else [])
            if receivers:
                share, extra = divmod(left, len(receivers))
                for i, g in enumerate(receivers):
                    add = share + (1 if i < extra else 0)
                    if add:
                        self.cap[g] += add
                        self.received[g] = self.received.get(g, 0) + add
                        self._log(g, reason='redistribution', source=gid, amount=add, cap=self.cap[g])
                self.released[gid] = left
            self._log(gid, reason='finish', how=how, tokens=spent, cap=cap, released=self.released.get(gid, 0),
                      receivers=len(receivers))
            return dict(game_cap_final=cap, game_tokens_used=spent, game_budget_exhausted=gid in self.exhausted,
                        redistributed_in=self.received.get(gid, 0), redistributed_out=self.released.get(gid, 0),
                        game_budget_end=how)

    def summary(self):
        """The run-level line for v6-stats.json: the pool, the sum of the caps net of the released tokens (the pool less the
        rounding remainder), the tokens used, the leftover never used (cap less released less used, summed over the games),
        the overshoot past the caps, the tokens redistributed, the exhausted games, the per-game numbers."""
        with self.lock:
            per = {g: dict(cap=self.cap[g], used=self.used.get(g, 0), received=self.received.get(g, 0),
                           released=self.released.get(g, 0), end=self.ended.get(g, 'not_finished'),
                           exhausted=g in self.exhausted) for g in self.cap}
            return dict(pool=self.pool, games=len(self.games), base_cap=self.base, unallocated=self.unallocated,
                        caps_sum=sum(p['cap'] - p['released'] for p in per.values()),
                        tokens_used=sum(p['used'] for p in per.values()),
                        leftover_unused=sum(max(0, p['cap'] - p['released'] - p['used']) for p in per.values()),
                        overshoot=sum(max(0, p['used'] - p['cap']) for p in per.values()),
                        redistributed=sum(p['released'] for p in per.values()), exhausted=len(self.exhausted),
                        per_game=per, **(dict(reserve=self.reserve, reserve_released=self.reserve_released,
                                              reserve_trigger=self.reserve_trigger) if self.reserve else {}))


def gamebudget_install(games, writer=None):
    """v6.7x GAMEBUDGET (run_v6): the run's budget when ARC3_V67_GAME_POOL > 0, else None (off)."""
    global GAMEBUDGET
    pool = int(__import__('os').environ.get('ARC3_V67_GAME_POOL', '0') or 0)
    reserve = int(__import__('os').environ.get('ARC3_V69_POOL_RESERVE', '0') or 0)   # v6.9 RAB RESERVE (0 = off)
    if reserve < 0 or (reserve and pool <= 0):
        raise RuntimeError('ARC3_V69_POOL_RESERVE must be 0 or a positive token count, with ARC3_V67_GAME_POOL > 0')
    GAMEBUDGET = GameBudget(pool, games, writer, reserve) if pool > 0 else None
    return GAMEBUDGET


def _gamebudget_spent(session):
    return sv._analyzer_reported_tokens(session.analyzer) - int(getattr(session, 'game_token_baseline', 0) or 0)


_original_token_limit = sv._HarnessGameSession.token_limit_reached


def _token_limit_reached(self):
    """v6.7x GAMEBUDGET: dfranzen's per-game token limit, and the game's cap in the run's pool (the first time it is reached,
    the solver note records budget_exhausted)."""
    budget = GAMEBUDGET
    run = self.game.game_run
    if budget is not None and run is not None:
        spent = _gamebudget_spent(self)
        if budget.reached(run.game_id, spent):
            if not str(run.solver_note or '').startswith('budget_exhausted'):
                run.solver_note = 'budget_exhausted: tokens=%d cap=%d' % (spent, budget.cap.get(run.game_id, 0))
            return True
    return _original_token_limit(self)


def _gamebudget_finish(session, budget, gid):
    """v6.7x GAMEBUDGET: at the end of a game's play, how it ended, the release of its leftover, its receipts in the agent's
    stats."""
    run = session.game.game_run
    soft = session.solver.soft_time_remaining_seconds()
    if gid in budget.exhausted:
        how = 'exhausted'
    elif (session.stop_event.is_set() or session.runtime_limit_reached() or (soft is not None and soft <= 0)
          or getattr(run, 'state', None) == 'cancelled'):
        how = 'run_end'
    else:
        how = str(getattr(run, 'state', None) or 'ended')
    receipts = budget.finish(gid, _gamebudget_spent(session), how)
    stats = getattr(session.analyzer, '_v6_stats', None)
    if receipts and isinstance(stats, dict):
        stats.update(receipts)


def _play(self):
    # v6.2 (run A: every game started together, so turns ended together and two host-wide stalls of 50-80 s at 4 and
    # 10 min timed out 40 tool calls at once; Fable: stagger the starts): the first 25 games start STAGGER s apart.
    if STAGGER > 0:
        time.sleep(STAGGER * min(int(getattr(self, 'game_index', 0) or 0), 24))
    budget = GAMEBUDGET                                     # v6.7x GAMEBUDGET: the game's cap in the run's pool
    gid = getattr(getattr(self.game, 'game_run', None), 'game_id', None)
    if budget is not None and gid is not None:
        budget.start(gid)
    try:
        return _original_play(self)
    finally:
        session = getattr(self.analyzer, '_v6_session', None)
        if session is not None:
            try:
                session.close()
            except Exception:  # noqa: BLE001
                pass
        if budget is not None and gid is not None:          # v6.7x GAMEBUDGET: the end, the leftover, the receipts
            try:
                _gamebudget_finish(self, budget, gid)
            except Exception as exc:  # noqa: BLE001
                print('GAMEBUDGET finish failed for %s: %s' % (gid, exc), flush=True)


sv._HarnessGameSession._execute_action = _execute_action
sv._HarnessGameSession.play = _play
sv._HarnessGameSession.token_limit_reached = _token_limit_reached   # v6.7x GAMEBUDGET
