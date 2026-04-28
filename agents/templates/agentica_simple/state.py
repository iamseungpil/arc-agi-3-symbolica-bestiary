"""Trimmed state helpers for ArcgenticaSimple v14.

Plan v14 §8 deletion manifest:
  KEEP: _grid_to_list, _grid_signature, _frame_payload, _normalize_action_label,
        _change_bbox, _bbox_edge_flags, _crop_grid, _nonzero_color_counts,
        _changed_cells, _change_pattern_summary, _dominant_color_label,
        _size_class, _shape_class, _visible_regions, SemanticPackets,
        _delta_cells_bin, TraceStore (legacy compat shell only).
  DROP: HypothesisLedger, ActionSequenceBook, SkillBook, TransformationClassIndex,
        interpret_change_summary, _summarize_sprite_families,
        _best_repeated_transition_hint.

SemanticPackets receives _NullTrace() at runtime, so trace-coupled state can
stay defined without dragging hypothesis-bookkeeping into the v14 main loop.
"""
from __future__ import annotations

import ast
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable


# ---------- Helpers ----------


def _normalize_action_step(step: Any) -> str:
    if isinstance(step, dict):
        action_name = str(step.get("action_name", "") or step.get("action", "")).strip().upper()
        if not action_name:
            return ""
        x = step.get("x")
        y = step.get("y")
        if x is not None and y is not None:
            return f"{action_name}({int(x)},{int(y)})"
        return action_name
    if isinstance(step, str):
        raw = step.strip()
        if not raw:
            return ""
        if raw.startswith("{") and raw.endswith("}"):
            try:
                parsed = ast.literal_eval(raw)
            except Exception:  # noqa: BLE001
                parsed = None
            if isinstance(parsed, dict):
                return _normalize_action_step(parsed)
        return raw
    return str(step).strip()


def _grid_to_list(grid: Any) -> list[list[int]]:
    return [[int(cell) for cell in row] for row in grid]


def _grid_signature(grid: list[list[int]]) -> str:
    if not grid:
        return "empty"
    rows = len(grid)
    cols = len(grid[0]) if grid[0] else 0
    total = sum(sum(int(cell) for cell in row) for row in grid)
    return f"{rows}x{cols}:{total}"


def _frame_payload(frame: Any) -> dict[str, Any]:
    grid = _grid_to_list(getattr(frame, "grid", []))
    return {
        "grid": grid,
        "signature": _grid_signature(grid),
        "level": int(getattr(frame, "levels_completed", 0)),
        "win_levels": int(getattr(frame, "win_levels", 0)),
        "state": str(getattr(getattr(frame, "state", None), "name", getattr(frame, "state", ""))),
        "available_actions": list(getattr(frame, "available_actions", [])),
    }


def _normalize_action_label(action: str) -> str:
    label = str(action).strip()
    if not label:
        return ""
    return label.split("(", 1)[0].strip().upper()


def _change_bbox(before: list[list[int]], after: list[list[int]]) -> dict[str, int] | None:
    min_x = min_y = None
    max_x = max_y = None
    max_rows = min(len(before), len(after))
    max_cols = min(len(before[0]) if before else 0, len(after[0]) if after else 0)
    for y in range(max_rows):
        for x in range(max_cols):
            if int(before[y][x]) == int(after[y][x]):
                continue
            min_x = x if min_x is None else min(min_x, x)
            min_y = y if min_y is None else min(min_y, y)
            max_x = x if max_x is None else max(max_x, x)
            max_y = y if max_y is None else max(max_y, y)
    if min_x is None:
        return None
    return {"min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y}


def _bbox_edge_flags(bbox: dict[str, int], *, rows: int, cols: int) -> tuple[bool, bool]:
    min_x = int(bbox.get("min_x", 0))
    min_y = int(bbox.get("min_y", 0))
    max_x = int(bbox.get("max_x", 0))
    max_y = int(bbox.get("max_y", 0))
    width = max_x - min_x + 1
    height = max_y - min_y + 1
    edge_touch = min_x == 0 or min_y == 0 or max_x == (cols - 1) or max_y == (rows - 1)
    thin_edge_like = edge_touch and (height <= 2 or width <= 2)
    return edge_touch, thin_edge_like


def _crop_grid(
    grid: list[list[int]],
    bbox: dict[str, int],
    *,
    pad: int = 1,
    max_side: int = 18,
) -> list[list[int]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    x0 = max(0, bbox["min_x"] - pad)
    y0 = max(0, bbox["min_y"] - pad)
    x1 = min(cols, bbox["max_x"] + pad + 1)
    y1 = min(rows, bbox["max_y"] + pad + 1)
    if (x1 - x0) > max_side:
        x1 = min(cols, x0 + max_side)
    if (y1 - y0) > max_side:
        y1 = min(rows, y0 + max_side)
    return [row[x0:x1] for row in grid[y0:y1]]


def _nonzero_color_counts(grid: list[list[int]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in grid:
        for cell in row:
            value = int(cell)
            if value == 0:
                continue
            key = str(value)
            counts[key] = counts.get(key, 0) + 1
    return counts


def _changed_cells(before: list[list[int]], after: list[list[int]]) -> list[tuple[int, int, int, int]]:
    changes: list[tuple[int, int, int, int]] = []
    if not before or not after:
        return changes
    for y, (before_row, after_row) in enumerate(zip(before, after)):
        for x, (before_value, after_value) in enumerate(zip(before_row, after_row)):
            if int(before_value) != int(after_value):
                changes.append((x, y, int(before_value), int(after_value)))
    return changes


def _change_pattern_summary(before: list[list[int]], after: list[list[int]]) -> dict[str, Any]:
    changes = _changed_cells(before, after)
    if not changes:
        return {
            "pattern_key": "none",
            "changed_cells": 0,
            "dominant_transition": None,
            "transitions": [],
            "relative_bbox": None,
        }
    xs = [item[0] for item in changes]
    ys = [item[1] for item in changes]
    min_x = min(xs)
    min_y = min(ys)
    max_x = max(xs)
    max_y = max(ys)
    transition_counts: dict[tuple[int, int], int] = {}
    relative_changes: list[tuple[int, int, int, int]] = []
    for x, y, before_value, after_value in changes:
        key = (before_value, after_value)
        transition_counts[key] = transition_counts.get(key, 0) + 1
        relative_changes.append((x - min_x, y - min_y, before_value, after_value))
    transition_rows = [
        {"from": int(before_value), "to": int(after_value), "count": int(count)}
        for (before_value, after_value), count in sorted(
            transition_counts.items(),
            key=lambda item: (-item[1], item[0][0], item[0][1]),
        )
    ]
    dominant_transition = transition_rows[0] if transition_rows else None
    pattern_payload = {
        "relative_changes": relative_changes,
        "transitions": transition_rows,
    }
    class_payload = {
        "transitions": transition_rows,
        "changed_cells": len(changes),
    }
    pattern_key = hashlib.sha1(
        json.dumps(pattern_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    transition_class_key = hashlib.sha1(
        json.dumps(class_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()[:12]
    return {
        "pattern_key": pattern_key,
        "transition_class_key": transition_class_key,
        "changed_cells": len(changes),
        "dominant_transition": dominant_transition,
        "transitions": transition_rows[:4],
        "relative_bbox": {
            "width": int(max_x - min_x + 1),
            "height": int(max_y - min_y + 1),
        },
    }


def _dominant_color_label(colors: dict[str, int]) -> int:
    if not colors:
        return 0
    ranked = sorted(
        ((int(color), int(count)) for color, count in colors.items()),
        key=lambda item: (-item[1], item[0]),
    )
    return ranked[0][0]


def _size_class(width: int, height: int, size: int) -> str:
    if width <= 8 and height <= 8 and size <= 100:
        return "compact"
    if width >= 20 or height >= 20 or size >= 250:
        return "broad"
    return "medium"


def _shape_class(width: int, height: int) -> str:
    if width >= height * 3:
        return "horizontal_strip"
    if height >= width * 3:
        return "vertical_strip"
    if abs(width - height) <= 2:
        return "square_like"
    return "rectangle"


def _visible_regions(
    grid: list[list[int]],
    *,
    background: int = 5,
    extra_backgrounds: tuple[int, ...] = (4,),  # v26: treat frame color 4 as also bg
    max_regions: int = 30,  # v22: was 8 — clipped multicolor bsT markers
    max_crop_size: int = 200,
    gqb_pair: tuple[int, int] | None = None,  # v29: dynamic toggle color cycle
) -> list[dict[str, Any]]:
    if not grid or not grid[0]:
        return []
    rows = len(grid)
    cols = len(grid[0])
    seen: set[tuple[int, int]] = set()
    regions: list[dict[str, Any]] = []
    bg_set = {int(background)} | {int(c) for c in extra_backgrounds}

    region_index = 1
    for y in range(rows):
        for x in range(cols):
            if int(grid[y][x]) in bg_set or (x, y) in seen:
                continue
            stack = [(x, y)]
            seen.add((x, y))
            points: list[tuple[int, int]] = []
            colors: dict[str, int] = {}
            while stack:
                cx, cy = stack.pop()
                points.append((cx, cy))
                value = str(int(grid[cy][cx]))
                colors[value] = colors.get(value, 0) + 1
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < cols and 0 <= ny < rows and int(grid[ny][nx]) not in bg_set and (nx, ny) not in seen:
                        seen.add((nx, ny))
                        stack.append((nx, ny))
            xs = [pt[0] for pt in points]
            ys = [pt[1] for pt in points]
            min_x = min(xs)
            max_x = max(xs)
            min_y = min(ys)
            max_y = max(ys)
            width = max_x - min_x + 1
            height = max_y - min_y + 1
            centroid_x = round(sum(xs) / len(xs), 1)
            centroid_y = round(sum(ys) / len(ys), 1)
            edge_touch = min_x == 0 or min_y == 0 or max_x == (cols - 1) or max_y == (rows - 1)
            thin_edge_like = edge_touch and (height <= 2 or width <= 2)
            dominant_color = _dominant_color_label(colors)
            size_class = _size_class(width, height, len(points))
            shape_class = _shape_class(width, height)
            # v22: include the pixel crop for SMALL regions (size <= max_crop_size)
            # so M1 can see the actual multicolor pattern. bsT constraint markers
            # are revealed by their non-uniform 3x3-style mask. Big regions like
            # the play area are skipped to keep token budget bounded.
            crop = None
            if len(points) <= max_crop_size:
                crop_bbox = {"min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y}
                crop = _crop_grid(grid, crop_bbox)
            regions.append(
                {
                    "id": f"R{region_index}",
                    "label": f"{size_class}-{shape_class}-c{dominant_color}",
                    "bbox": {"min_x": min_x, "min_y": min_y, "max_x": max_x, "max_y": max_y},
                    "size": len(points),
                    "centroid": {"x": centroid_x, "y": centroid_y},
                    "width": width,
                    "height": height,
                    "colors": colors,
                    "dominant_color": dominant_color,
                    "color_count": len(colors),
                    "size_class": size_class,
                    "shape_class": shape_class,
                    "edge_touch": edge_touch,
                    "thin_edge_like": thin_edge_like,
                    "crop": crop,  # v22: 2D pixel array or None
                    "is_multicolor": (len(colors) >= 3),  # v22: quick flag for constraint-marker candidates
                }
            )
            region_index += 1
    # v22: sort priority — multicolor markers (likely bsT constraint markers)
    # FIRST, then by edge/size as before. This ensures that even with a region
    # cap, the constraint markers are not pruned away.
    regions.sort(
        key=lambda item: (
            not bool(item.get("is_multicolor")),  # multicolor first
            bool(item.get("thin_edge_like")),
            bool(item.get("edge_touch")),
            -int(item.get("size", 0)),
        )
    )
    regions = regions[:max_regions]

    # v25: compute macro 3x3 neighbor topology for each MULTICOLOR region.
    # For each marker, find the 8 closest non-multicolor regions that fall
    # roughly into N/NE/E/SE/S/SW/W/NW directions relative to the marker
    # centroid. This gives M1 the spatial mapping that bsT.mask cells need
    # to derive specific Hkx target colors. Without this, agent can articulate
    # the mask rule but cannot translate it to "which neighbor must be color X".
    # v27: helper to decode bsT mask from crop and precompute per-direction
    # target colors per the source-verified rule:
    #   sprite.pixels[j][i] (3x3 mask), bsT.center = pixels[1][1].
    #   For each of 8 neighbor directions:
    #     mask cell == 0  → that neighbor must EQUAL bsT.center
    #     mask cell != 0  → that neighbor must DIFFER from bsT.center
    # ft09's gqb (toggle cycle) defaults to (9, 8); we infer "other" as the
    # alternate of bsT.center within {8, 9}.
    _DIR_BY_MASK_POS = {
        (0, 0): "NW", (0, 1): "N", (0, 2): "NE",
        (1, 0): "W",                (1, 2): "E",
        (2, 0): "SW", (2, 1): "S",  (2, 2): "SE",
    }
    def _decode_mask_targets(crop: list[list[int]] | None,
                             bsT_center: int | None,
                             gqb_pair: tuple[int, int] | None = None) -> dict[str, int] | None:
        if not crop or len(crop) < 7 or bsT_center is None:
            return None
        # crop is 8x8 with 1-cell pad; sprite cells at crop[1+2*j][1+2*i].
        # Build 3x3 mask.
        mask = []
        for j in range(3):
            row = []
            for i in range(3):
                cy = 1 + 2 * j
                cx = 1 + 2 * i
                if cy < len(crop) and 0 <= cx < len(crop[cy]):
                    row.append(int(crop[cy][cx]))
                else:
                    row.append(None)
            mask.append(row)
        # v29: prefer dynamic gqb (inferred from observed transitions). Fallback
        # to {9, 8} only when no gqb_pair is supplied.
        if gqb_pair and len(gqb_pair) == 2:
            a, b = int(gqb_pair[0]), int(gqb_pair[1])
            if int(bsT_center) == a:
                other = b
            elif int(bsT_center) == b:
                other = a
            else:
                # bsT center isn't in observed pair — keep heuristic default
                other = 9 if int(bsT_center) == 8 else 8
        else:
            other = 9 if int(bsT_center) == 8 else 8
        targets: dict[str, int] = {}
        for (j, i), d in _DIR_BY_MASK_POS.items():
            m = mask[j][i]
            if m is None:
                continue
            targets[d] = int(bsT_center) if m == 0 else other
        return targets

    by_id = {r["id"]: r for r in regions}
    for r in regions:
        if not r.get("is_multicolor"):
            continue
        cx = r["centroid"]["x"]
        cy = r["centroid"]["y"]
        # Distance + direction sectors. Sector by atan2 angle in 8 buckets.
        import math as _math
        candidates = []
        for other in regions:
            if other["id"] == r["id"] or other.get("is_multicolor"):
                continue
            ox = other["centroid"]["x"]
            oy = other["centroid"]["y"]
            dx = ox - cx
            dy = oy - cy
            dist = _math.hypot(dx, dy)
            if dist <= 0:
                continue
            ang = _math.degrees(_math.atan2(-dy, dx))  # screen y inverted
            if ang < 0:
                ang += 360
            # 8 sectors of 45° each, centered on E=0°, NE=45°, N=90°, NW=135°,
            # W=180°, SW=225°, S=270°, SE=315°
            sector_idx = int((ang + 22.5) // 45) % 8
            sector_name = ["E", "NE", "N", "NW", "W", "SW", "S", "SE"][sector_idx]
            candidates.append((dist, sector_name, other["id"]))
        # Pick nearest in each sector (within ~30 cells = 2 macro slots)
        neighbors_3x3: dict[str, str | None] = {
            d: None for d in ("N", "NE", "E", "SE", "S", "SW", "W", "NW")
        }
        candidates.sort(key=lambda t: t[0])
        for dist, sec, rid in candidates:
            if neighbors_3x3.get(sec) is None and dist <= 30:
                neighbors_3x3[sec] = rid
            if all(v is not None for v in neighbors_3x3.values()):
                break
        r["neighbors_3x3"] = neighbors_3x3
        # Bonus telemetry: bsT center-pixel value if crop is available.
        crop = r.get("crop")
        if isinstance(crop, list) and crop:
            mh = len(crop) // 2
            mw = len(crop[mh]) // 2 if crop[mh] else 0
            r["bsT_center_color"] = int(crop[mh][mw]) if mw < len(crop[mh]) else None
        else:
            r["bsT_center_color"] = None
        # v27: precompute per-direction target colors (the win-state per
        # neighbor) so M1/M2 don't have to derive from raw pixels in their head.
        # v28: when neighbor region is merged/missing, sample the grid directly
        # at the expected offset position (±8 cells from R31 center for ft09).
        target_colors = _decode_mask_targets(r.get("crop"), r.get("bsT_center_color"), gqb_pair)
        # ft09 sprite scaling: 3x3 sprite × 2x2 cells/pixel = 6x6 sprite block
        # plus 2-cell gap between sprites. So neighbor centers are ±8 from
        # this marker's center in cardinal directions, ±8 in diagonals (each axis).
        offset_by_dir = {
            "NW": (-8, -8), "N": (0, -8), "NE": (+8, -8),
            "W":  (-8,  0),                 "E": (+8,  0),
            "SW": (-8, +8), "S": (0, +8), "SE": (+8, +8),
        }
        per_neighbor_target: dict[str, dict] = {}
        if target_colors:
            for d, target in target_colors.items():
                nb_id = r["neighbors_3x3"].get(d)
                # First try region-based lookup
                if nb_id and nb_id in by_id:
                    cur = int(by_id[nb_id].get("dominant_color") or -1)
                    per_neighbor_target[d] = {
                        "neighbor_id": nb_id,
                        "current_color": cur,
                        "target_color": int(target),
                        "needs_toggle": (cur != int(target)),
                        "source": "region",
                    }
                else:
                    # v28: fallback — sample grid directly at expected offset
                    dx, dy = offset_by_dir.get(d, (0, 0))
                    sx = int(round(cx + dx))
                    sy = int(round(cy + dy))
                    cur = -1
                    if 0 <= sy < rows and 0 <= sx < cols:
                        cur = int(grid[sy][sx])
                    per_neighbor_target[d] = {
                        "neighbor_id": None,
                        "current_color": cur,
                        "target_color": int(target),
                        "needs_toggle": (cur != int(target)) if cur >= 0 else None,
                        "neighbor_xy": (sx, sy),
                        "source": "grid_sample",
                    }
        r["expected_neighbor_colors"] = target_colors
        r["per_neighbor_target"] = per_neighbor_target
    return regions


# ---------- SemanticPackets ----------


class SemanticPackets:
    """Minimal helper that keeps the latest grid and recent trajectory together.

    v14: trace argument is a _NullTrace shim from goal_board, not a real
    TraceStore. The single trace callsite is ``self._trace.list(kind="action")``
    which returns []; downstream pattern aggregation collapses to empty
    trajectory. Visible regions, render rows, and frame signature are still
    computed from the latest frame and passed to M1.
    """

    def __init__(
        self,
        history_fn,
        trace_store,
        game_id: str = "",
        live_object_supplier: Callable[[], list[dict[str, Any]]] | None = None,
    ) -> None:
        self._history = history_fn
        self._trace = trace_store
        self._game_id = game_id
        self._live_object_supplier = live_object_supplier

    def current(self, trajectory_limit: int = 4, max_regions: int = 3, gqb_pair: tuple[int, int] | None = None) -> dict[str, Any]:
        history_rows = self._history(max(trajectory_limit + 1, 1))
        if not history_rows:
            return {
                "game_id": self._game_id,
                "frame_signature": {},
                "recent_trajectory": [],
                "stateful_regions": [],
                "render_rows": {},
                "visible_regions": [],
                "visible_sprites": [],
            }

        latest_frame = history_rows[-1][1]
        current_payload = _frame_payload(latest_frame)
        current_grid = current_payload["grid"]
        recent_actions = self._trace.list(kind="action")[-trajectory_limit:]
        trajectory: list[dict[str, Any]] = []
        regions: list[dict[str, Any]] = []
        render_row_ids: set[int] = set()

        for event in recent_actions:
            before = event.get("before", {}).get("grid", [])
            after = event.get("after", {}).get("grid", [])
            bbox = _change_bbox(before, after)
            pattern = _change_pattern_summary(before, after)
            trajectory.append(
                {
                    "action": str(event.get("action", "")).strip(),
                    "changed_cells": int(event.get("change_summary", {}).get("changed_cells", 0) or 0),
                    "bbox": bbox,
                    "dominant_transition": pattern.get("dominant_transition"),
                    "transitions": pattern.get("transitions", []),
                }
            )
            if bbox is None:
                continue
            edge_touch, thin_edge_like = _bbox_edge_flags(
                bbox,
                rows=len(current_grid),
                cols=len(current_grid[0]) if current_grid else 0,
            )
            regions.append(
                {
                    "action": str(event.get("action", "")).strip(),
                    "bbox": bbox,
                    "changed_cells": int(event.get("change_summary", {}).get("changed_cells", 0) or 0),
                    "edge_touch": edge_touch,
                    "thin_edge_like": thin_edge_like,
                    "crop": _crop_grid(current_grid, bbox),
                }
            )
            render_row_ids.update(range(max(0, bbox["min_y"] - 1), bbox["max_y"] + 2))

        # v22: include ALL rows (not just 10). The earlier 10-row cap was
        # masking constraint-marker bands for grids where stateful regions
        # span multiple non-contiguous y-bands (e.g. ft09's macro 3x3 layout).
        # Token cost: 64 rows × ~150 chars = ~10KB; acceptable.
        rendered_rows: dict[str, str] = {}
        try:
            rendered = latest_frame.render(y_ticks=True, x_ticks=False)
            lines = rendered.splitlines()
            for row_id, line in enumerate(lines):
                rendered_rows[str(row_id)] = line
        except Exception:  # noqa: BLE001
            rendered_rows = {}

        visible_sprites: list[dict[str, Any]] = []
        if callable(self._live_object_supplier):
            try:
                supplied = self._live_object_supplier()
                if isinstance(supplied, list):
                    visible_sprites = [dict(item) for item in supplied if isinstance(item, dict)]
            except Exception:  # noqa: BLE001
                visible_sprites = []

        return {
            "game_id": self._game_id,
            "frame_signature": {
                "level": current_payload["level"],
                "win_levels": current_payload["win_levels"],
                "state": current_payload["state"],
                "available_actions": current_payload["available_actions"],
                "grid_signature": current_payload["signature"],
                "color_counts": _nonzero_color_counts(current_grid),
            },
            "visible_regions": _visible_regions(current_grid, gqb_pair=gqb_pair),
            "visible_sprites": visible_sprites[:12],
            "recent_trajectory": trajectory,
            "stateful_regions": sorted(
                regions,
                key=lambda item: (
                    bool(item.get("thin_edge_like")),
                    bool(item.get("edge_touch")),
                    -int(item.get("changed_cells", 0)),
                ),
            )[:max_regions],
            "render_rows": rendered_rows,
        }

    def summary(self, trajectory_limit: int = 4) -> str:
        packet = self.current(trajectory_limit=trajectory_limit)
        return (
            f"level={packet.get('frame_signature', {}).get('level', 0)} "
            f"actions={len(packet.get('recent_trajectory', []))} "
            f"regions={len(packet.get('stateful_regions', []))}"
        )


def _delta_cells_bin(changed_cells: int) -> str:
    """Coarse bucketing so two near-identical magnitudes hash to the same class."""
    cells = int(changed_cells or 0)
    if cells <= 0:
        return "0"
    if cells <= 2:
        return "1-2"
    if cells <= 8:
        return "3-8"
    if cells <= 24:
        return "9-24"
    if cells <= 64:
        return "25-64"
    return "65+"


# ---------- Legacy TraceStore shell (parent compat only) ----------


class TraceStore:
    """Legacy compat shell. v14 uses _NullTrace; this stays only so external
    callers that still ``from .state import TraceStore`` do not crash."""

    def __init__(self, path: Path | None = None) -> None:
        self.path = Path(path) if path is not None else None
        self._events: list[dict[str, Any]] = []
        self._active_plan: dict[str, Any] | None = None

    def list(self, kind: str | None = None) -> list[dict[str, Any]]:
        if kind is None:
            return list(self._events)
        return [e for e in self._events if e.get("kind") == kind]

    def event(self, *args, **kwargs) -> None:
        return None

    def plan(self, *args, **kwargs) -> None:
        return None

    def record_action(self, *args, **kwargs) -> None:
        return None

    def current_plan(self) -> dict[str, Any] | None:
        return None

    def summary(self) -> str:
        return "TraceStore (legacy compat shell)"
