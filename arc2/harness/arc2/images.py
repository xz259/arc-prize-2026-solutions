"""Grid images as an aid beside the text grids (the user, 2026-10-29: images only ever in addition to the ASCII grids;
answers stay text).

The served model's preprocessor (preprocessor_config.json): 16-px patches merged 2x2, so one LLM token covers 32x32 px,
and images under 65,536 px^2 are resized by interpolation. So each cell is drawn as a CELL x CELL block (nearest
neighbour, default 16: one vision patch per cell), with the same cell size for every grid of a task, and the canvas is
padded with white (not an ARC-2 colour) to a multiple of 32 px and to at least 256 x 256, so the preprocessor never
resamples a grid. Cost: (width / 32) x (height / 32) tokens, 64 for a small grid, 225 for 30 x 30.
"""
from __future__ import annotations

import base64
import io

from PIL import Image

# The ARC colours 0-9 (black, blue, red, green, yellow, gray, magenta, orange, sky blue, dark red): the same colours
# as the letters B b R N Y g M O S r name in the text grids.
PALETTE = [(0, 0, 0), (0, 116, 217), (255, 65, 54), (46, 204, 64), (255, 220, 0), (170, 170, 170), (240, 18, 190),
           (255, 133, 27), (127, 219, 255), (135, 12, 37)]
PAD = (255, 255, 255)
TOKEN_PX = 32
MIN_SIDE = 256


def canvas_size(h, w, cell=16):
    def side(n):
        px = max(n * cell, MIN_SIDE)
        return -(-px // TOKEN_PX) * TOKEN_PX
    return side(w), side(h)


def grid_png(grid, cell=16):
    """PNG bytes of one grid: cells as cell x cell blocks at the top left, white padding to the canvas size."""
    h, w = len(grid), len(grid[0])
    cw, ch = canvas_size(h, w, cell)
    img = Image.new("RGB", (cw, ch), PAD)
    small = Image.new("RGB", (w, h))
    small.putdata([PALETTE[int(v)] for row in grid for v in row])
    img.paste(small.resize((w * cell, h * cell), Image.Resampling.NEAREST), (0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def image_part(grid, cell=16):
    data = base64.b64encode(grid_png(grid, cell)).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{data}"}}


def image_tokens(grid, cell=16):
    cw, ch = canvas_size(len(grid), len(grid[0]), cell)
    return (cw // TOKEN_PX) * (ch // TOKEN_PX)
