"""ARC-AGI-2 tasks: loading, grid text, the Kaggle submission format and the local scorer.

Kaggle rule (sample_submission.json; the 2025 reference's validate_submission): one entry per test input, each
{"attempt_1": grid, "attempt_2": grid}; a task scores the fraction of its test inputs matched exactly by either attempt.
Solutions never leave this module's caller: the sandbox only ever receives the train pairs and the test inputs.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE.parent.parent / "official" / "data"          # arc2_stack/official/data
EVAL_CHALLENGES = DATA_DIR / "arc-agi_evaluation_challenges.json"
EVAL_SOLUTIONS = DATA_DIR / "arc-agi_evaluation_solutions.json"
PLACEHOLDER = [[0]]


@dataclass
class Task:
    task_id: str
    train: list                                # [{"input": grid, "output": grid}]
    test: list                                 # [{"input": grid}]
    solutions: list | None = field(default=None, repr=False)   # one output grid per test input

    def sandbox_payload(self, fmt="letters"):
        """What the attempt's Python kernel may see: the demonstrations and the test inputs, never the solutions."""
        return {"task_id": self.task_id, "format": fmt, "letters": LETTERS,
                "train": [{"input": p["input"], "output": p["output"]} for p in self.train],
                "test": [{"input": p["input"]} for p in self.test]}


def load_tasks(challenges=EVAL_CHALLENGES, solutions=None, ids=None):
    with open(challenges) as f:
        data = json.load(f)
    sols = {}
    if solutions:
        with open(solutions) as f:
            sols = json.load(f)
    keys = list(ids) if ids else sorted(data)
    missing = [k for k in keys if k not in data]
    if missing:
        raise KeyError(f"unknown task ids: {missing[:5]}")
    return [Task(k, data[k]["train"], data[k]["test"], sols.get(k)) for k in keys]


def to_list(grid):
    """A grid as a list of lists of Python ints (accepts numpy arrays and nested sequences)."""
    if hasattr(grid, "tolist"):
        grid = grid.tolist()
    return [[int(v) for v in row] for row in grid]


def grid_shape(grid):
    return len(grid), (len(grid[0]) if grid else 0)


def valid_grid(grid):
    """2-D, rectangular, 1-30 on each side, integers 0-9."""
    try:
        rows = to_list(grid)
    except (TypeError, ValueError):
        return False
    if not 1 <= len(rows) <= 30:
        return False
    width = len(rows[0])
    if not 1 <= width <= 30 or any(len(r) != width for r in rows):
        return False
    return all(0 <= v <= 9 for r in rows for v in r)


# ARC-3's board symbols (inference/utils/grid_utils.py ARC_COLOR_CHARS = "WwgGcBMPRbSYOrNp", one character per cell,
# rows joined by newlines, no separators; the drafter fr1cfinal was trained on text in this form). ARC-2's ten colours are
# written with the ARC-3 letter of the same colour: 0 black B, 1 blue b, 2 red R, 3 green N, 4 yellow Y, 5 gray g,
# 6 magenta M, 7 orange O, 8 sky blue S, 9 dark red (maroon) r.
LETTERS = "BbRNYgMOSr"
LETTER_NAMES = ("B=black, b=blue, R=red, N=green, Y=yellow, g=gray, M=magenta, O=orange, S=sky blue, r=dark red")
LETTER_CODE = {ch: i for i, ch in enumerate(LETTERS)}
GRID_FORMATS = ("letters", "digits")


def grid_text(grid, fmt="letters"):
    """One symbol per cell, rows on separate lines, no separators: ARC-3's letters (default) or ARC-2's digits."""
    alphabet = LETTERS if fmt == "letters" else "0123456789"
    return "\n".join("".join(alphabet[int(v)] for v in row) for row in to_list(grid))


def submission_entry(task, attempts):
    """attempts: {1: [grid per test input] | None, 2: [...] | None} -> the Kaggle list for this task."""
    entry = []
    for i in range(len(task.test)):
        slot = {}
        for a in (1, 2):
            outs = attempts.get(a) or []
            g = outs[i] if i < len(outs) else None
            slot[f"attempt_{a}"] = to_list(g) if (g is not None and valid_grid(g)) else PLACEHOLDER
        entry.append(slot)
    return entry


def placeholder_entry(task):
    return [{"attempt_1": PLACEHOLDER, "attempt_2": PLACEHOLDER} for _ in task.test]


def score_task(task, entry):
    """Fraction of the task's test inputs matched exactly by attempt_1 or attempt_2 (None without solutions)."""
    if task.solutions is None:
        return None
    hits = 0
    for i, sol in enumerate(task.solutions):
        slot = entry[i] if i < len(entry) else {}
        if any(slot.get(k) == sol for k in ("attempt_1", "attempt_2")):
            hits += 1
    return hits / len(task.solutions)


def score_outputs(tasks, submission):
    """The competition's metric: (test outputs matched by attempt_1 or attempt_2, all test outputs), every test output
    weighted equally (a task with two test inputs counts twice). Tasks without solutions are skipped."""
    hits = n = 0
    for t in tasks:
        if t.solutions is None:
            continue
        entry = submission.get(t.task_id, placeholder_entry(t))
        for i, sol in enumerate(t.solutions):
            slot = entry[i] if i < len(entry) else {}
            hits += int(any(slot.get(k) == sol for k in ("attempt_1", "attempt_2")))
            n += 1
    return hits, n


def score_submission(tasks, submission):
    per_task = {t.task_id: score_task(t, submission.get(t.task_id, placeholder_entry(t))) for t in tasks}
    scored = [v for v in per_task.values() if v is not None]
    return sum(scored), per_task


def write_json(path, obj):
    """Write via a temporary file and an atomic rename, so a reader never sees a partial file."""
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj))
    tmp.replace(path)
