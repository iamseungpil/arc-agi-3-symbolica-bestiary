"""Replay cycle237 7-step path with PNG dump + pixel diff per step.

Reveals ft09's click mechanic visually: where does each click change the
frame, what color before→after, and is the change local or global?
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("ARC_USE_LOCAL_ENV_ONLY", "1")
os.environ.setdefault("OPERATION_MODE", "offline")
os.environ.setdefault("ENVIRONMENTS_DIR", "environment_files")


PALETTE = {
    0: (0, 0, 0), 1: (0, 116, 217), 2: (255, 65, 54), 3: (46, 204, 64),
    4: (255, 220, 0), 5: (170, 170, 170), 6: (240, 18, 190),
    7: (255, 133, 27), 8: (127, 219, 255), 9: (135, 12, 37),
    12: (10, 10, 80),  # appeared in starting frame row 64
}

CYCLE237 = [
    (6, 38), (34, 34), (6, 38), (38, 38),
    (38, 46), (54, 46), (38, 54),
]


def _grid(raw):
    import numpy as np
    frame = getattr(raw, "frame", None)
    layers = list(frame) if frame is not None else []
    for g in reversed(layers):
        g_list = g.tolist() if hasattr(g, "tolist") else list(g)
        if g_list and isinstance(g_list[0], (list, tuple)) and len(g_list[0]) > 0:
            return g_list
    return None


def _render(grid, out_path, mark=None):
    from PIL import Image, ImageDraw
    h, w = len(grid), len(grid[0])
    scale = 8
    img = Image.new("RGB", (w * scale, h * scale), (0, 0, 0))
    pixels = img.load()
    for y in range(h):
        for x in range(w):
            c = int(grid[y][x])
            color = PALETTE.get(c, (50, 50, 50))
            for dy in range(scale):
                for dx in range(scale):
                    pixels[x * scale + dx, y * scale + dy] = color
    # 4-pixel snap-grid overlay
    for y in range(0, h * scale, scale * 4):
        for x in range(w * scale):
            pixels[x, y] = (60, 60, 60)
    for x in range(0, w * scale, scale * 4):
        for y in range(h * scale):
            pixels[x, y] = (60, 60, 60)
    # Mark click coord
    if mark is not None:
        cx, cy = mark
        draw = ImageDraw.Draw(img)
        px, py = cx * scale, cy * scale
        draw.rectangle([px - 4, py - 4, px + scale + 4, py + scale + 4],
                       outline=(255, 0, 255), width=3)
    img.save(out_path)


def _diff(g_pre, g_post):
    diffs = []
    for y, (row_pre, row_post) in enumerate(zip(g_pre, g_post)):
        for x, (c_pre, c_post) in enumerate(zip(row_pre, row_post)):
            if c_pre != c_post:
                diffs.append({"xy": (x, y), "before": int(c_pre),
                              "after": int(c_post)})
    return diffs


def main():
    from arcengine import GameAction
    from arc_agi import Arcade, OperationMode
    arc = Arcade(operation_mode=OperationMode.OFFLINE)
    env = arc.make("ft09-9ab2447a")
    raw = env.reset()
    grid = _grid(raw)
    out_dir = REPO_ROOT / "reports" / "cycle237_replay"
    out_dir.mkdir(parents=True, exist_ok=True)
    _render(grid, out_dir / "step_00_reset.png")
    print(f"[step 0] reset levels={raw.levels_completed} -> step_00_reset.png")

    pre_grid = grid
    for i, (x, y) in enumerate(CYCLE237, start=1):
        raw = env.step(GameAction.ACTION6, data={"x": int(x), "y": int(y)})
        post_grid = _grid(raw)
        if post_grid is None:
            print(f"[step {i}] no grid"); continue
        diffs = _diff(pre_grid, post_grid)
        # Summary: bbox of changes, color transitions
        if diffs:
            xs = [d["xy"][0] for d in diffs]
            ys = [d["xy"][1] for d in diffs]
            bbox = (min(xs), min(ys), max(xs), max(ys))
            color_pairs = sorted({(d["before"], d["after"]) for d in diffs})
        else:
            bbox = None
            color_pairs = []
        print(f"[step {i}] click=({x},{y}) levels={raw.levels_completed} "
              f"diffs={len(diffs)} bbox={bbox} color_pairs={color_pairs}")
        path = out_dir / f"step_{i:02d}_click_{x:02d}_{y:02d}.png"
        _render(post_grid, path, mark=(x, y))
        pre_grid = post_grid

    print(f"\n[wrote] {out_dir}/")


if __name__ == "__main__":
    main()
