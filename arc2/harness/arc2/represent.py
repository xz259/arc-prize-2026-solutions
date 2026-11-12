"""Re-presented task copies: the memorization control (user decision 2026-10-29: in every Phase B decision run).

Each task gets one non-identity symmetry of the square (rotation, reflection or transpose) and one permutation of the
colours 1-9 (0 is kept by default), applied consistently to every grid of the task, solutions included. The copy is a
new task with the same rule in a new frame; it is scored in that frame. Task ids get the suffix ".r<seed>", and the
transform of each task is written beside the files so per-task results can be paired with the originals.

    python -m arc2.represent --seed 1 --out DIR      # writes DIR/represented_s1_{challenges,solutions,keys}.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from . import tasks as T

SYMMETRIES = {
    "rot90": lambda a: np.rot90(a, 1),
    "rot180": lambda a: np.rot90(a, 2),
    "rot270": lambda a: np.rot90(a, 3),
    "flipud": np.flipud,
    "fliplr": np.fliplr,
    "transpose": lambda a: a.T,
    "antitranspose": lambda a: np.rot90(a, 2).T,
}


def task_key(task_id, seed, keep_zero=True):
    """Deterministic (symmetry, colour map) for one task and seed."""
    rng = random.Random(f"{task_id}:{seed}")
    sym = rng.choice(sorted(SYMMETRIES))
    colours = list(range(1, 10)) if keep_zero else list(range(10))
    shuffled = colours[:]
    rng.shuffle(shuffled)
    cmap = list(range(10))
    for a, b in zip(colours, shuffled):
        cmap[a] = b
    return {"symmetry": sym, "colour_map": cmap}


def apply_key(grid, key):
    a = np.asarray(grid, dtype=int)
    a = SYMMETRIES[key["symmetry"]](a)
    return np.asarray(key["colour_map"])[a].tolist()


INVERSE = {"rot90": "rot270", "rot270": "rot90", "rot180": "rot180", "flipud": "flipud", "fliplr": "fliplr",
           "transpose": "transpose", "antitranspose": "antitranspose", "identity": "identity"}


def pass_key(task_id, seed, symmetry=False):
    """v2o (the reviewer session's input 23, item 2; the user, 2026-11-01: "add it as a flag"): the frame of a re-presented
    pass: a permutation of the colours 1-9 (0 kept) and, with symmetry, one non-identity symmetry of the square, both
    deterministic per task and seed (task_key's draw)."""
    key = task_key(task_id, seed, keep_zero=True)
    return key if symmetry else {"symmetry": "identity", "colour_map": key["colour_map"]}


def to_frame(grid, key):
    """grid in key's frame (task_key's apply_key, with the identity symmetry allowed)."""
    a = np.asarray(grid, dtype=int)
    if key["symmetry"] != "identity":
        a = SYMMETRIES[key["symmetry"]](a)
    return np.asarray(key["colour_map"])[a].tolist()


def from_frame(grid, key):
    """The inverse of to_frame: a grid of key's frame back in the original frame (None stays None; a grid that is not a
    rectangle of colours 0-9 comes back unchanged, since it cannot match a solution in either frame)."""
    if grid is None:
        return None
    inv = [0] * 10
    for a, b in enumerate(key["colour_map"]):
        inv[b] = a
    try:
        g = np.asarray(grid, dtype=int)
        if g.ndim != 2 or g.size == 0 or g.min() < 0 or g.max() > 9:
            return grid
    except (TypeError, ValueError):
        return grid
    a = np.asarray(inv)[g]
    if key["symmetry"] != "identity":
        a = SYMMETRIES[INVERSE[key["symmetry"]]](a)
    return a.tolist()


def framed_task(task, key):
    """The task as a new puzzle in key's frame: the same id, every grid transformed, no solutions (scored after the outputs
    come back through from_frame)."""
    return T.Task(task.task_id, [{"input": to_frame(p["input"], key), "output": to_frame(p["output"], key)} for p in task.train],
                  [{"input": to_frame(p["input"], key)} for p in task.test], None)


def represent(challenges, solutions, seed=1, keep_zero=True):
    ch, so, keys = {}, {}, {}
    for tid in sorted(challenges):
        key = task_key(tid, seed, keep_zero)
        new = f"{tid}.r{seed}"
        keys[new] = dict(key, original=tid)
        ch[new] = {"train": [{"input": apply_key(p["input"], key), "output": apply_key(p["output"], key)}
                             for p in challenges[tid]["train"]],
                   "test": [{"input": apply_key(p["input"], key)} for p in challenges[tid]["test"]]}
        if solutions is not None and tid in solutions:
            so[new] = [apply_key(g, key) for g in solutions[tid]]
    return ch, so, keys


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--challenges", default=str(T.EVAL_CHALLENGES))
    p.add_argument("--solutions", default=str(T.EVAL_SOLUTIONS))
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--permute-zero", action="store_true", help="permute all ten colours (default keeps 0)")
    p.add_argument("--out", required=True)
    args = p.parse_args(argv)
    challenges = json.load(open(args.challenges))
    solutions = json.load(open(args.solutions)) if args.solutions else None
    ch, so, keys = represent(challenges, solutions, args.seed, keep_zero=not args.permute_zero)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    stem = f"represented_s{args.seed}"
    T.write_json(out / f"{stem}_challenges.json", ch)
    if so:
        T.write_json(out / f"{stem}_solutions.json", so)
    T.write_json(out / f"{stem}_keys.json", keys)
    print(f"{len(ch)} tasks -> {out}/{stem}_*.json")


if __name__ == "__main__":
    main()
