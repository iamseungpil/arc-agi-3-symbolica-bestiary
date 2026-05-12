"""Render ft09's starting frame as PNG so a human can SEE it.

Per codex 2026-05-12 baseline #1 (human-in-loop): paste the frame to a
human and ask 'what would you click?' to sanity-check what a smart agent
should propose.
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


# Minimal ARC palette (colors 0..9 + extras)
PALETTE = {
    0: (0, 0, 0),         # black
    1: (0, 116, 217),     # blue
    2: (255, 65, 54),     # red
    3: (46, 204, 64),     # green
    4: (255, 220, 0),     # yellow
    5: (170, 170, 170),   # grey
    6: (240, 18, 190),    # fuchsia
    7: (255, 133, 27),    # orange
    8: (127, 219, 255),   # light blue
    9: (135, 12, 37),     # maroon
}


def main():
    from arc_agi import Arcade, OperationMode
    arc = Arcade(operation_mode=OperationMode.OFFLINE)
    env = arc.make("ft09-9ab2447a")
    raw = env.reset()
    frame = getattr(raw, "frame", None)

    # frame is nested list/numpy 2D grids stack. Take last non-empty layer.
    import numpy as np  # noqa: PLC0415
    def _as_list(x):
        return x.tolist() if hasattr(x, "tolist") else list(x)
    layers = list(frame) if frame is not None else []
    grid = None
    for g in reversed(layers):
        try:
            g_list = _as_list(g)
        except Exception:
            continue
        if not g_list:
            continue
        row0 = g_list[0]
        if isinstance(row0, (list, tuple)) and len(row0) > 0:
            grid = g_list
            break
    if grid is None:
        print("[error] no usable grid"); sys.exit(2)

    h = len(grid)
    w = len(grid[0])
    print(f"ft09 starting frame: {h}x{w}, levels_completed={raw.levels_completed}")

    # Print as text (cell values)
    print("\n--- cell values (h x w) ---")
    for row in grid:
        print(" ".join(f"{int(c):x}" for c in row))

    # Render PNG via PIL
    try:
        from PIL import Image
        scale = 8  # 8 pixels per cell
        img = Image.new("RGB", (w * scale, h * scale), (0, 0, 0))
        pixels = img.load()
        for y in range(h):
            for x in range(w):
                c = int(grid[y][x])
                color = PALETTE.get(c, (50, 50, 50))
                for dy in range(scale):
                    for dx in range(scale):
                        pixels[x * scale + dx, y * scale + dy] = color
        # Grid overlay every 4 cells (snap-grid bins)
        for y in range(0, h * scale, scale * 4):
            for x in range(w * scale):
                pixels[x, y] = (80, 80, 80)
        for x in range(0, w * scale, scale * 4):
            for y in range(h * scale):
                pixels[x, y] = (80, 80, 80)
        out_path = REPO_ROOT / "reports" / "ft09_starting_frame.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path)
        print(f"\n[wrote PNG] {out_path}")
    except ImportError:
        print("[skip PNG] PIL not available")

    # Print color distribution
    from collections import Counter
    cnt = Counter(c for row in grid for c in row)
    print("\n--- color counts ---")
    for color, n in cnt.most_common():
        print(f"  color {color}: {n} cells")


if __name__ == "__main__":
    main()
