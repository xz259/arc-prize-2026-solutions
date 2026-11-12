"""Builds the model-facing messages of an ARC-2 attempt from texts.json.

The opener is the session's single observation. Its first text part (the puzzle: every demonstration and test grid)
is pinned on the server through the transport's summary pin (stage-2 part `sumpin`): protected from eviction like the
system prompt until the model stops attending to it (the part's GRACE / STREAK release rule). The short second part
(where the grids live in python) is not pinned.
"""
from __future__ import annotations

import json
from pathlib import Path

from .tasks import LETTER_NAMES, LETTERS, grid_shape, grid_text

TEXTS_PATH = Path(__file__).resolve().parent / "texts.json"


def load_texts(path=TEXTS_PATH):
    with open(path) as f:
        return json.load(f)


def system_message(texts, *, budget, notice, fmt="letters", key="system", reply_cap=None, image_cell=None):
    grid_format = texts[f"grid_format_{fmt}"].format(legend=LETTER_NAMES)
    grid_python = texts[f"grid_python_{fmt}"].format(order=" ".join(LETTERS))
    images_note = texts["images_note"].format(cell=image_cell) if image_cell else ""
    return {"role": "system", "content": texts[key].format(budget=budget, notice=notice, grid_format=grid_format,
                                                           grid_python=grid_python, reply_cap=reply_cap or 0,
                                                           images_note=images_note)}


def opener_text(task, texts, fmt="letters"):
    n_test = len(task.test)
    head = texts["opener_head"].format(task_id=task.task_id, n_train=len(task.train), n_test=n_test,
                                       inputs="input" if n_test == 1 else "inputs")
    return "\n\n".join([head] + grid_blocks(task, texts, fmt))


def grid_blocks(task, texts, fmt="letters"):
    """The puzzle's grids as text blocks: every demonstration pair, then every test input."""
    blocks = []
    for i, pair in enumerate(task.train):
        (h, w), (h2, w2) = grid_shape(pair["input"]), grid_shape(pair["output"])
        blocks.append(texts["opener_pair"].format(i=i, h=h, w=w, h2=h2, w2=w2, input=grid_text(pair["input"], fmt),
                                                  output=grid_text(pair["output"], fmt)))
    for j, t in enumerate(task.test):
        h, w = grid_shape(t["input"])
        blocks.append(texts["opener_test"].format(j=j, h=h, w=w, grid=grid_text(t["input"], fmt)))
    return blocks


def opener_message(task, texts, *, pin=True, fmt="letters", image_cell=None):
    """The opener as content parts; `_arc3_observation` registers it with the server's history registry, and
    `_arc3_pin_parts` asks the transport to pin its leading parts (the puzzle). With image_cell (the user, 2026-10-29:
    images only as an aid beside the text), every text part comes first and the captioned images after them, pinned as
    v7.7's text pin and board pin (`_arc3_pin_split`)."""
    text = [{"type": "text", "text": opener_text(task, texts, fmt)},
            {"type": "text", "text": "\n\n" + texts["opener_tail"]}]
    if not image_cell:
        msg = {"role": "user", "content": text, "_arc3_observation": True}
        if pin:
            msg["_arc3_pin_parts"] = 1
        return msg
    # v7.7's shape (the server's sumpin board pin, 2026-10-29: interleaved text and images failed the pin's boundary check):
    # every text part first, then the images, each after a short caption; the text pin covers [0, 2), the board pin the rest
    from .images import image_part
    boards = []
    for i, pair in enumerate(task.train):
        boards += [{"type": "text", "text": f"\n\nDemonstration {i}, input:"}, image_part(pair["input"], image_cell),
                   {"type": "text", "text": f"\n\nDemonstration {i}, output:"}, image_part(pair["output"], image_cell)]
    for j, t in enumerate(task.test):
        boards += [{"type": "text", "text": f"\n\nTest input {j}:"}, image_part(t["input"], image_cell)]
    msg = {"role": "user", "content": text + boards, "_arc3_observation": True}
    if pin:
        msg["_arc3_pin_parts"] = len(text) + len(boards)
        msg["_arc3_pin_split"] = len(text)
    return msg


def apply_transport_texts(transport_module, texts):
    """Replace the ARC-3 transport's in-reply cue texts (which speak of a game) with the ARC-2 wording. The transport
    reads them as module globals at call time, so rebinding the module attributes is enough."""
    replaced = []
    for name, value in texts.get("transport", {}).items():
        if hasattr(transport_module, name):
            setattr(transport_module, name, value)
            replaced.append(name)
    return replaced
