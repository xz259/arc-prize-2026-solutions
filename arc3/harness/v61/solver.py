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


def _execute_action(self, action, **kwargs):
    payload = _original_execute(self, action, **kwargs)
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


def _play(self):
    try:
        return _original_play(self)
    finally:
        session = getattr(self.analyzer, '_v6_session', None)
        if session is not None:
            try:
                session.close()
            except Exception:  # noqa: BLE001
                pass


sv._HarnessGameSession._execute_action = _execute_action
sv._HarnessGameSession.play = _play
