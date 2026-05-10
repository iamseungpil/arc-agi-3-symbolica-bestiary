"""v605 arm7: render the latest game grid as a PNG for multimodal LLM.

ARC-AGI-3 colors are integers 0-15 (palette). We render at 8x scale with
coordinate axis labels every 8 px so the LLM can name click coordinates
directly. No game-specific identifiers are introduced; the rendering is
fully general.
"""

from __future__ import annotations

import base64
import io
from typing import Any

from PIL import Image, ImageDraw, ImageFont

# ARC-AGI-3 standard 16-color palette (matches cycle237 era rendering).
_PALETTE = [
    (0, 0, 0),         # 0 black (background)
    (0, 116, 217),     # 1 blue
    (255, 65, 54),     # 2 red
    (46, 204, 64),     # 3 green
    (255, 220, 0),     # 4 yellow
    (170, 170, 170),   # 5 gray
    (240, 18, 190),    # 6 magenta
    (255, 133, 27),    # 7 orange
    (127, 219, 255),   # 8 light blue
    (135, 12, 37),     # 9 dark red
    (45, 27, 78),      # 10 dark blue
    (177, 156, 217),   # 11 lavender
    (24, 70, 86),      # 12 teal
    (107, 142, 35),    # 13 olive
    (255, 140, 0),     # 14 dark orange
    (200, 100, 50),    # 15 brown
]


def render_grid_png(grid: list[list[int]], scale: int = 8) -> bytes:
    """Render grid as a PNG image with scale-x upscale + coord ticks every 8 cells."""
    if not grid or not grid[0]:
        # blank 1x1 PNG
        img = Image.new("RGB", (1, 1), (0, 0, 0))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        return buf.getvalue()
    H = len(grid)
    W = len(grid[0])
    # base raster
    img = Image.new("RGB", (W, H), (0, 0, 0))
    pixels = img.load()
    for r in range(H):
        for c in range(W):
            v = grid[r][c]
            if 0 <= v < len(_PALETTE):
                pixels[c, r] = _PALETTE[v]
    # upscale
    big = img.resize((W * scale, H * scale), Image.NEAREST)
    # coord tick overlay
    draw = ImageDraw.Draw(big)
    try:
        font = ImageFont.load_default()
    except Exception:  # noqa: BLE001
        font = None
    for x in range(0, W, 8):
        draw.text((x * scale + 1, 0), str(x), fill=(255, 255, 255), font=font)
    for y in range(0, H, 8):
        draw.text((0, y * scale + 1), str(y), fill=(255, 255, 255), font=font)
    buf = io.BytesIO()
    big.save(buf, format="PNG")
    return buf.getvalue()


def render_grid_data_url(grid: list[list[int]]) -> str:
    """Return a data: URL suitable for chat completion image_url content."""
    png = render_grid_png(grid)
    b64 = base64.b64encode(png).decode("ascii")
    return f"data:image/png;base64,{b64}"


def get_latest_grid(state_dict: dict[str, Any]) -> list[list[int]] | None:
    """Pull the latest grid from a state dict that may carry it in different keys."""
    if not isinstance(state_dict, dict):
        return None
    g = state_dict.get("_latest_grid")
    if g:
        return g
    return None
