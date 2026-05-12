"""Minimal vision-first multimodal LLM solver for ft09.

No framework wrapping. Just:
  raw 64x64 grid + PNG → multimodal LLM → 4 click coords → env.step

Goal: verify codex Q5 (c/d) — that a vision-capable LLM CAN solve ft09
when given honest perception, without M1-M4 reflexion, anti-leak, or
predicate library.

Usage:
  python scripts/baselines/vision_llm_ft09_solver.py [--model NAME] [--max-attempts 3]
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import re
import sys
import time
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
    12: (10, 10, 80),
}


def render_grid_png_b64(grid, scale=8, with_grid_overlay=True):
    """Render 64x64 grid to PNG with 4-pixel snap-grid overlay. Return base64 str."""
    from PIL import Image
    h, w = len(grid), len(grid[0])
    img = Image.new("RGB", (w * scale, h * scale), (0, 0, 0))
    pixels = img.load()
    for y in range(h):
        for x in range(w):
            c = int(grid[y][x])
            color = PALETTE.get(c, (50, 50, 50))
            for dy in range(scale):
                for dx in range(scale):
                    pixels[x * scale + dx, y * scale + dy] = color
    if with_grid_overlay:
        for y in range(0, h * scale, scale * 4):
            for x in range(w * scale):
                pixels[x, y] = (60, 60, 60)
        for x in range(0, w * scale, scale * 4):
            for y in range(h * scale):
                pixels[x, y] = (60, 60, 60)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _get_grid(raw):
    frame = getattr(raw, "frame", None)
    layers = list(frame) if frame is not None else []
    for g in reversed(layers):
        g_list = g.tolist() if hasattr(g, "tolist") else list(g)
        if g_list and isinstance(g_list[0], (list, tuple)) and len(g_list[0]) > 0:
            return g_list
    return None


PROMPT = """You are looking at an ARC-AGI-3 puzzle called ft09.

The 64x64 grid has THREE reference quadrants and ONE target zone:
- Top-left (rows 2-23, cols 4-22): 3x3 grid of REFERENCE cards
- Top-right (rows 2-23, cols 38-58): 3x3 grid of REFERENCE cards
- Bottom-left (rows 36-59, cols 4-22): 3x3 grid of REFERENCE cards
- Bottom-right (rows 32-59, cols 32-59): TARGET zone, surrounded by yellow border with red corners. Contains 3x3 grid of cards (initially all maroon = color 9).

Each card is 6x6 pixels. Each card is either MAROON (color 9) or LIGHT BLUE (color 8). The center cards may contain a small interior pattern.

GAME MECHANIC: Clicking a maroon card inside the TARGET zone flips it to light blue. The game advances to the next level (L+1) when the target zone's 3x3 card pattern EXACTLY MATCHES one of the three reference quadrant patterns.

YOUR TASK: Look at the image. Determine which reference quadrant the target should match. Output exactly 4 (x, y) click coordinates that will flip the right cards to achieve the match.

CONSTRAINTS:
- Click x must satisfy x % 4 == 2 (i.e. x in {2, 6, 10, ..., 62})
- Click y must satisfy y % 2 == 0 (i.e. y in {0, 2, 4, ..., 62})
- A click hits the card centered at (x, y); typical card centers inside the target zone are x in {38, 46, 54} and y in {38, 46, 54}

The center of each card in the target zone is approximately:
  (38, 38) | (46, 38) | (54, 38)
  (38, 46) | (46, 46) | (54, 46)
  (38, 54) | (46, 54) | (54, 54)

You must respond with EXACTLY 4 lines, each "x,y", and NOTHING else. Example:
38,38
38,46
54,46
38,54

If you are unsure which reference matches best, pick the one with the most LIGHT BLUE cards and click TARGET zone cards corresponding to that reference's BLUE positions.
"""


async def call_multimodal(grid, model, png_data_url):
    from azure.identity import (
        AzureCliCredential, ChainedTokenCredential, DefaultAzureCredential,
    )
    from openai import AsyncAzureOpenAI

    cred = ChainedTokenCredential(AzureCliCredential(), DefaultAzureCredential())

    def _token_provider():
        return cred.get_token("api://trapi/.default").token

    client = AsyncAzureOpenAI(
        azure_endpoint="https://trapi.research.microsoft.com/gcr/shared",
        api_version="2025-04-01-preview",
        azure_ad_token_provider=_token_provider,
    )

    # Compact text rendering of the grid for redundancy.
    text_grid = "\n".join(" ".join(f"{int(c):x}" for c in row) for row in grid)

    messages = [
        {"role": "system", "content": "You are a careful visual puzzle solver."},
        {"role": "user", "content": [
            {"type": "text", "text": PROMPT + "\n\nRaw grid (hex, 64x64):\n" + text_grid},
            {"type": "image_url", "image_url": {"url": png_data_url}},
        ]},
    ]
    kwargs = {"model": model, "messages": messages, "max_completion_tokens": 200}
    if "5.5" not in model.lower():
        kwargs["temperature"] = 0.0
    resp = await client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content or ""


def parse_coords(text, max_n=4):
    out = []
    for line in text.strip().splitlines():
        line = line.strip().strip("()")
        m = re.match(r"^\s*(\d+)\s*[,\s]\s*(\d+)\s*$", line)
        if m:
            out.append((int(m.group(1)), int(m.group(2))))
            if len(out) >= max_n:
                break
    return out


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.4-pro_2026-03-05",
                        help="multimodal-capable TRAPI deployment")
    parser.add_argument("--max-attempts", type=int, default=3)
    parser.add_argument("--game", default="ft09-9ab2447a")
    parser.add_argument("--max-levels", type=int, default=1,
                        help="stop after reaching this many L+ events")
    parser.add_argument("--out", default=None)
    args = parser.parse_args(argv)

    from arcengine import GameAction
    from arc_agi import Arcade, OperationMode
    arc = Arcade(operation_mode=OperationMode.OFFLINE)
    env = arc.make(args.game)

    history = []
    t0 = time.time()
    raw = env.reset()
    grid = _get_grid(raw)
    levels = int(getattr(raw, "levels_completed", 0) or 0)
    initial_levels = levels
    l_plus_events = []

    for attempt in range(args.max_attempts):
        print(f"\n[attempt {attempt+1}/{args.max_attempts}] "
              f"levels={levels} grid_hash={hash(str(grid)) & 0xffff:04x}", flush=True)

        png_b64 = render_grid_png_b64(grid)
        data_url = f"data:image/png;base64,{png_b64}"

        try:
            response = asyncio.run(call_multimodal(grid, args.model, data_url))
        except Exception as e:
            print(f"  [LLM error] {e}", flush=True)
            history.append({"attempt": attempt, "error": str(e)})
            continue

        print(f"  [response]\n{response}", flush=True)
        coords = parse_coords(response)
        if not coords:
            print("  [parse failed] no coords", flush=True)
            history.append({"attempt": attempt, "response": response,
                            "parsed_coords": [], "error": "parse_failed"})
            continue

        print(f"  [executing {len(coords)} clicks] {coords}", flush=True)
        attempt_l_plus = []
        for i, (x, y) in enumerate(coords):
            raw = env.step(GameAction.ACTION6, data={"x": int(x), "y": int(y)})
            new_lvls = int(getattr(raw, "levels_completed", 0) or 0)
            if new_lvls > levels:
                attempt_l_plus.append({"click_idx": i, "coord": [x, y],
                                       "levels_before": levels,
                                       "levels_after": new_lvls})
                l_plus_events.append({"attempt": attempt, "click_idx": i,
                                      "coord": [x, y],
                                      "levels_before": levels,
                                      "levels_after": new_lvls})
                levels = new_lvls

        grid = _get_grid(raw)
        history.append({"attempt": attempt, "response": response,
                        "parsed_coords": coords, "levels_after": levels,
                        "attempt_l_plus": attempt_l_plus})
        print(f"  [verdict] levels: {initial_levels} -> {levels}, "
              f"this attempt L+ events: {len(attempt_l_plus)}", flush=True)
        if levels - initial_levels >= args.max_levels:
            print(f"  [DONE] reached target level delta", flush=True)
            break

    wall = time.time() - t0
    summary = {
        "game": args.game,
        "model": args.model,
        "max_attempts": args.max_attempts,
        "levels_initial": initial_levels,
        "levels_final": levels,
        "total_l_plus_events": len(l_plus_events),
        "wall_seconds": wall,
        "l_plus_events": l_plus_events,
        "history": history,
    }
    out = args.out or f"reports/vision_llm_ft09_{int(t0)}.json"
    out_full = (REPO_ROOT / out).resolve()
    out_full.parent.mkdir(parents=True, exist_ok=True)
    out_full.write_text(json.dumps(summary, indent=2))

    print(f"\n[FINAL] L+ events: {len(l_plus_events)} | "
          f"levels: {initial_levels} -> {levels} | wall: {wall:.1f}s")
    print(f"  wrote: {out_full}")
    return 0 if levels > initial_levels else 2


if __name__ == "__main__":
    raise SystemExit(main())
