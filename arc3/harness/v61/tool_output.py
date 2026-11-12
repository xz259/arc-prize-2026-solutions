"""Harness v6 tool-output board conversion (spec §5.6 step 2; BUILD_PLAN 8). A fallback the prompt does not advertise
(it says never to print whole boards): printed grids whose own text reaches about a full board are replaced by a
one-line caption and attached as 4x images (66 tokens each), deduplicated, at most 4 per tool result.

A board run is a maximal run of consecutive lines that are all grid rows of the same width: ARC symbols
(WwgGcBMPRbSYOrNp) or digits/hex, optionally separated by single spaces or commas, at least MIN_ROWS rows of at least
MIN_COLS cells, and whose text reaches MIN_TOKENS tokens. Anything smaller (crops) stays text."""
import hashlib
import re

from inference.agent.runtime_state import Frame
from inference.agent.vision_context import frame_to_png_data_url
from inference.utils.grid_utils import ARC_COLOR_CHARS

MIN_ROWS, MIN_COLS, MIN_TOKENS, MAX_IMAGES, SCALE = 16, 16, 1024, 4, 4
_SYMBOL = {c: i for i, c in enumerate(ARC_COLOR_CHARS)}
_HEX = {c: int(c, 16) for c in '0123456789abcdefABCDEF'}
_ROW = re.compile(r'^\s*[\[\(]?\s*((?:[WwgGcBMPRbSYOrNp0-9a-fA-F][ ,]?){%d,})\s*[\]\)]?,?\s*$' % MIN_COLS)


def _cells(line):
    m = _ROW.match(line)
    if not m:
        return None
    body = m.group(1).replace(',', ' ').strip()
    toks = body.split() if ' ' in body else list(body)
    if any(len(t) != 1 for t in toks):
        return None
    if all(t in _SYMBOL for t in toks):
        return [_SYMBOL[t] for t in toks]
    if all(t in _HEX for t in toks):
        return [_HEX[t] for t in toks]
    return None


def convert(text, count_tokens, captions, state):
    """Replace full-board runs in `text`. Returns (new text, [(caption, image part)]). `state` is shared by the tool
    result's fields: dict(seen={grid hash: image number}, budget=images left, omitted=count)."""
    lines = text.split('\n')
    out, images, i = [], [], 0
    while i < len(lines):
        row = _cells(lines[i])
        if row is None:
            out.append(lines[i])
            i += 1
            continue
        j, grid = i, []
        while j < len(lines):
            r = _cells(lines[j])
            if r is None or len(r) != len(row):
                break
            grid.append(r)
            j += 1
        run_text = '\n'.join(lines[i:j])
        if len(grid) >= MIN_ROWS and count_tokens(run_text) >= MIN_TOKENS:
            h = hashlib.sha256(repr(grid).encode()).hexdigest()
            if h in state['seen']:
                out.append(captions['same'].replace('{j}', str(state['seen'][h])))
            elif state['budget'] > 0:
                state['budget'] -= 1
                n = len(state['seen']) + 1
                state['seen'][h] = n
                cap = captions['board'].replace('{rows}', str(len(grid))).replace('{cols}', str(len(row)))
                out.append(cap)
                part = {"type": "image_url",
                        "image_url": {"url": frame_to_png_data_url(Frame(grid=grid, step=0, level=0), upscale=SCALE)}}
                images.append((cap, part))
            else:
                state['omitted'] += 1
            i = j
        else:
            out.extend(lines[i:j])
            i = j
    return '\n'.join(out), images
