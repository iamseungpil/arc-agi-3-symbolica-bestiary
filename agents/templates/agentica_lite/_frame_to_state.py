"""Generic FrameData -> ArcgenticaLite state translator (v603 adapter helper).

Whitelist policy (Codex Contract C1):
  Inputs allowed:
    * frame.frame                (raw 64x64 grid stack from arcengine)
    * frame.levels_completed     (int)
    * frame.state                (GameState enum)
    * frame.available_actions    (list[GameAction])
    * action_history             (list of (coord_xy, prev_grid, new_grid, level_delta_after))
                                  accumulated by the adapter from its OWN emitted ACTION6 calls.

  Forbidden inputs (Contract C2):
    * No game-specific hardcoded region IDs.
    * No game-specific saturation thresholds.
    * No engine-internal state beyond FrameData fields.
    * No prior-cycle persisted memory files are read here.

The translator uses ONLY:
    1. 4-connected flood-fill on the latest grid -> components.
    2. Generic features per component: bbox, color (or "multicolor"),
       size in pixels, neighbors_3x3 from bbox topology.
    3. Generic stable region IDs derived from spatial position
       (id = "C{rank}" where rank is by (top_row, left_col)).
       The naming is deterministic across calls within an episode IFF the
       segmentation is stable, which is the same assumption the v601
       framework makes about R-ids.
    4. dominant_transition: count of cells whose color changed between the
       previous grid (from action_history's last entry) and the new grid.
    5. level_delta: levels_completed[t] - levels_completed[t-1].
    6. marker_neighbor_states: derived heuristically as components whose
       neighbors_3x3 has >=4 populated entries; per-direction click counts
       come ONLY from the action_history (Contract C3).

Each function carries a `# WHITELIST: clauseN` comment indicating which
clause of Contract C1 it satisfies.
"""

from __future__ import annotations

from collections import deque
from typing import Any


# WHITELIST: clause 1 (4-connected flood-fill on raw grid).
def _flood_fill_components(grid: list[list[int]]) -> list[dict[str, Any]]:
    """Return list of components with id, bbox, color, is_multicolor, size, cells_seed.

    Components are ranked by (top_row, left_col) so IDs are deterministic.
    The "background" (color 0) is included as a normal component; downstream
    code can filter if needed.
    """
    if not grid or not grid[0]:
        return []
    H = len(grid)
    W = len(grid[0])
    visited = [[False] * W for _ in range(H)]
    raw_comps: list[dict[str, Any]] = []
    for r0 in range(H):
        for c0 in range(W):
            if visited[r0][c0]:
                continue
            seed_color = grid[r0][c0]
            # 4-conn BFS
            queue: deque = deque()
            queue.append((r0, c0))
            visited[r0][c0] = True
            cells: list[tuple[int, int]] = []
            colors: set[int] = set()
            while queue:
                r, c = queue.popleft()
                cells.append((r, c))
                colors.add(grid[r][c])
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and not visited[nr][nc]:
                        # connect on equality OR through any contiguous non-bg
                        # block of similar palette. We use simple equality
                        # connectivity here to keep the function generic and
                        # auditable.
                        if grid[nr][nc] == seed_color:
                            visited[nr][nc] = True
                            queue.append((nr, nc))
            min_r = min(c[0] for c in cells)
            max_r = max(c[0] for c in cells)
            min_c = min(c[1] for c in cells)
            max_c = max(c[1] for c in cells)
            raw_comps.append(
                {
                    "_seed_color": seed_color,
                    "_cells": cells,
                    "bbox": [min_c, min_r, max_c, max_r],  # x0, y0, x1, y1
                    "size": len(cells),
                    "color": seed_color,
                    "is_multicolor": len(colors) > 1,
                }
            )
    # v605 arm6: multicolor super-component merge pass.
    # Equality flood-fill splits a multicolor 3x3 marker into 9 single-cell
    # components. Detect tight clusters of 4+ small components in a 3x3 bbox
    # and merge them into one multicolor super-component.
    raw_comps = _merge_multicolor_clusters(raw_comps)

    # v606.2 codex r14-r15: Sig-B greedy cross-turn matching.
    # Each component is matched to the prev-turn region with the highest IoU
    # among compatible candidates (same color OR both multicolor). Reuse the
    # prior region_id on match; allocate a fresh C{idx} from a monotonic
    # per-episode counter on miss. This preserves region identity across
    # split/merge/move events so the LLM can accumulate per-region evidence.
    raw_comps.sort(key=lambda c: (c["bbox"][1], c["bbox"][0]))
    out: list[dict[str, Any]] = []
    state = _STABILITY_STATE
    used_prev: set[str] = set()
    for c in raw_comps:
        rid = _match_or_assign(c, state, used_prev)
        out.append(
            {
                "id": rid,
                "region_id": rid,
                "bbox": c["bbox"],
                "size": c["size"],
                "color": c["color"],
                "is_multicolor": c["is_multicolor"],
                "_cells_first": c["_cells"][0] if c["_cells"] else (0, 0),
                "_cells_centroid": _centroid(c["_cells"]),
                "y_band": _y_band(c["bbox"]),
            }
        )
    # Update memory of prev_signatures for next turn
    state["prev"] = [
        {
            "id": r["region_id"],
            "bbox": r["bbox"],
            "color": r["color"],
            "is_multicolor": r["is_multicolor"],
            "size": r["size"],
        }
        for r in out
    ]
    return out


# Module-level state: stays alive for the duration of one episode (process).
# adapter.reset_episode() must call _reset_stability_state() to clear.
_STABILITY_STATE: dict[str, Any] = {"prev": [], "next_idx": 1}


def _reset_stability_state() -> None:
    """Called at start of each new episode (per adapter contract)."""
    _STABILITY_STATE["prev"] = []
    _STABILITY_STATE["next_idx"] = 1


def _iou_xyxy(a: list[int], b: list[int]) -> float:
    ax0, ay0, ax1, ay1 = a[0], a[1], a[2], a[3]
    bx0, by0, bx1, by1 = b[0], b[1], b[2], b[3]
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    if ix1 < ix0 or iy1 < iy0:
        return 0.0
    inter = (ix1 - ix0 + 1) * (iy1 - iy0 + 1)
    area_a = (ax1 - ax0 + 1) * (ay1 - ay0 + 1)
    area_b = (bx1 - bx0 + 1) * (by1 - by0 + 1)
    union = area_a + area_b - inter
    return inter / max(union, 1)


def _compatible(c: dict, p: dict) -> bool:
    # Same color OR both multicolor — informative match constraint.
    if c.get("is_multicolor") and p.get("is_multicolor"):
        return True
    return c.get("color") == p.get("color")


def _match_or_assign(c: dict, state: dict, used: set[str]) -> str:
    best_id, best_iou = None, 0.0
    for p in state["prev"]:
        if p["id"] in used or not _compatible(c, p):
            continue
        s = _iou_xyxy(c["bbox"], p["bbox"])
        if s > best_iou:
            best_iou, best_id = s, p["id"]
    if best_id is not None and best_iou >= 0.3:
        used.add(best_id)
        return best_id
    rid = f"C{state['next_idx']}"
    state["next_idx"] += 1
    return rid


def _merge_multicolor_clusters(raw_comps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """v605 arm6 helper: detect 3x3 clusters of small components and merge.

    Strategy: among components with size <= 2, find groups of 4+ whose
    bounding box (after merge) fits in a 4x4 area. Merge them into one
    multicolor super-component (same total cells, multicolor=True).
    Background-color (0) components are excluded from merging.
    """
    if not raw_comps:
        return raw_comps
    small = [c for c in raw_comps if c["size"] <= 2 and c["_seed_color"] != 0]
    other = [c for c in raw_comps if not (c["size"] <= 2 and c["_seed_color"] != 0)]
    used = [False] * len(small)
    merged: list[dict[str, Any]] = []
    for i, c in enumerate(small):
        if used[i]:
            continue
        cluster = [c]
        used[i] = True
        cx0, cy0 = c["_cells"][0][1], c["_cells"][0][0]
        for j, d in enumerate(small):
            if used[j] or i == j:
                continue
            dx0, dy0 = d["_cells"][0][1], d["_cells"][0][0]
            if abs(cx0 - dx0) <= 3 and abs(cy0 - dy0) <= 3:
                cluster.append(d)
                used[j] = True
        if len(cluster) >= 4:
            all_cells: list[tuple[int, int]] = []
            all_colors: set[int] = set()
            for cc in cluster:
                all_cells.extend(cc["_cells"])
                all_colors.add(cc["_seed_color"])
            min_r = min(c[0] for c in all_cells)
            max_r = max(c[0] for c in all_cells)
            min_c = min(c[1] for c in all_cells)
            max_c = max(c[1] for c in all_cells)
            merged.append({
                "_seed_color": cluster[0]["_seed_color"],
                "_cells": all_cells,
                "bbox": [min_c, min_r, max_c, max_r],
                "size": len(all_cells),
                "color": cluster[0]["_seed_color"],
                "is_multicolor": len(all_colors) > 1,
            })
        else:
            # not a multicolor cluster — keep components individually
            merged.extend(cluster)
    return other + merged


def _centroid(cells: list[tuple[int, int]]) -> tuple[int, int]:
    """Cell-list centroid as (x, y). WHITELIST: clause 2."""
    if not cells:
        return (0, 0)
    sr = sum(c[0] for c in cells)
    sc = sum(c[1] for c in cells)
    n = len(cells)
    return (sc // n, sr // n)


def _y_band(bbox: list[int]) -> str:
    """Generic vertical band classification. WHITELIST: clause 2.

    Splits a 64-row grid into top/play/bottom thirds. No game-specific constants.
    """
    y0 = bbox[1]
    if y0 < 16:
        return "top_strip"
    if y0 >= 48:
        return "bottom_strip"
    return "play_zone"


# WHITELIST: clause 2 (bbox topology only).
def _compute_neighbors_3x3(comps: list[dict[str, Any]]) -> None:
    """Mutate comps to add neighbors_3x3 dict using bbox-center directions.

    For each component, we sweep all others and pick the closest one in each
    of 8 compass directions whose centroid sits in the relevant quadrant.
    """
    for me in comps:
        mx0, my0, mx1, my1 = me["bbox"]
        cx = (mx0 + mx1) / 2.0
        cy = (my0 + my1) / 2.0
        nbrs: dict[str, str | None] = {
            "N": None, "NE": None, "E": None, "SE": None,
            "S": None, "SW": None, "W": None, "NW": None,
        }
        nbr_dist: dict[str, float] = {k: float("inf") for k in nbrs}
        for other in comps:
            if other["id"] == me["id"]:
                continue
            ox0, oy0, ox1, oy1 = other["bbox"]
            ocx = (ox0 + ox1) / 2.0
            ocy = (oy0 + oy1) / 2.0
            dx = ocx - cx
            dy = ocy - cy
            dist = (dx * dx + dy * dy) ** 0.5
            d = _direction_label(dx, dy)
            if d is not None and dist < nbr_dist[d]:
                nbr_dist[d] = dist
                nbrs[d] = other["id"]
        me["neighbors_3x3"] = nbrs


def _direction_label(dx: float, dy: float) -> str | None:
    """Classify a vector into one of 8 compass directions, or None when
    the magnitude is too small. WHITELIST: clause 2."""
    if abs(dx) < 1.0 and abs(dy) < 1.0:
        return None
    angle_h = abs(dy) < abs(dx) * 0.4
    angle_v = abs(dx) < abs(dy) * 0.4
    if angle_h:
        return "E" if dx > 0 else "W"
    if angle_v:
        return "S" if dy > 0 else "N"
    if dx > 0 and dy < 0:
        return "NE"
    if dx > 0 and dy > 0:
        return "SE"
    if dx < 0 and dy < 0:
        return "NW"
    if dx < 0 and dy > 0:
        return "SW"
    return None


# WHITELIST: clauses 2 + 3 (markers heuristically defined as well-connected
# multi-neighbor components; click counts come ONLY from action_history).
def _markers_from_components(
    comps: list[dict[str, Any]],
    action_history: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Identify likely markers and build their compass dicts.

    v605 arm5: tighten heuristic to reduce false positives. ft09 markers
    are 3x3 multicolor regions; our flood-fill splits them into many
    single-cell components. To approximate "real markers" we require:
      - size >= 4 cells (not a single isolated cell), AND
      - >= 4 populated 8-direction neighbors,
      - OR is_multicolor=True (real multicolor blocks).
    """
    markers: list[dict[str, Any]] = []
    for c in comps:
        nbrs = c.get("neighbors_3x3") or {}
        populated = [(d, n) for d, n in nbrs.items() if n is not None]
        is_multi = bool(c.get("is_multicolor", False))
        size = int(c.get("size", 0))
        # v605 arm5 filter: must be size>=4 with >=4 neighbors, OR multicolor.
        if not (is_multi or (size >= 4 and len(populated) >= 4)):
            continue
        compass: dict[str, dict[str, Any]] = {}
        for direction, nbr_id in populated:
            nbr_comp = next((cc for cc in comps if cc["id"] == nbr_id), None)
            clicks = _click_count_in_region(nbr_comp, action_history) if nbr_comp else 0
            compass[direction] = {
                "region_id": nbr_id,
                "current_color": (nbr_comp.get("color") if nbr_comp else None),
                "clicks": clicks,
            }
        markers.append(
            {
                "marker_id": c["id"],
                "compass": compass,
                "is_primary_marker": None,
            }
        )
    return markers


def _click_count_in_region(
    region: dict[str, Any] | None,
    action_history: list[dict[str, Any]],
) -> int:
    """Count action_history entries whose coord_xy lies inside region.bbox.
    WHITELIST: clause 3 (history is the agent's own emitted actions only)."""
    if region is None:
        return 0
    x0, y0, x1, y1 = region["bbox"]
    n = 0
    for entry in action_history:
        cx, cy = entry["coord_xy"]
        if x0 <= cx <= x1 and y0 <= cy <= y1:
            n += 1
    return n


def _dominant_transition(
    prev_grid: list[list[int]] | None,
    curr_grid: list[list[int]],
) -> dict[str, Any]:
    """Compute (from_color -> to_color, count) for the modal transition between
    two grids. WHITELIST: clause 1 (raw-grid comparison only)."""
    if prev_grid is None:
        return {"from": None, "to": None, "count": 0}
    H = min(len(prev_grid), len(curr_grid))
    if H == 0:
        return {"from": None, "to": None, "count": 0}
    W = min(len(prev_grid[0]) if prev_grid[0] else 0, len(curr_grid[0]) if curr_grid[0] else 0)
    counts: dict[tuple[int, int], int] = {}
    for r in range(H):
        for c in range(W):
            a = prev_grid[r][c]
            b = curr_grid[r][c]
            if a != b:
                k = (a, b)
                counts[k] = counts.get(k, 0) + 1
    if not counts:
        return {"from": None, "to": None, "count": 0}
    (a, b), n = max(counts.items(), key=lambda kv: kv[1])
    return {"from": a, "to": b, "count": n}


def _primary_region_id_from_last_click(
    comps: list[dict[str, Any]],
    last_coord: tuple[int, int] | None,
) -> str | None:
    """Pick the component whose bbox contains the last clicked coord.
    WHITELIST: clauses 2 + 3."""
    if last_coord is None:
        return None
    x, y = last_coord
    for c in comps:
        x0, y0, x1, y1 = c["bbox"]
        if x0 <= x <= x1 and y0 <= y <= y1:
            return c["id"]
    return None


def _latest_grid_from_frame(frame_field: Any) -> list[list[int]] | None:
    """frame.frame may be a list of grids; pick the latest non-empty one."""
    if not frame_field:
        return None
    # frame_field is list[list[list[int]]]. Pick last non-empty grid.
    for grid in reversed(frame_field):
        if grid and grid[0]:
            return grid
    return None


def frame_to_state(
    frame: Any,
    action_history: list[dict[str, Any]],
    prev_levels_completed: int,
) -> dict[str, Any]:
    """Translate FrameData -> ArcgenticaLite state dict.

    The output schema matches what `ArcgenticaLite.run_turn` reads:
      visible_regions, marker_neighbor_states, last_observation
      (with primary_region_id, dominant_transition, level_delta).

    WHITELIST INPUTS (Contract C1):
      - frame.frame, frame.levels_completed
      - action_history: list of dicts with keys
            'coord_xy' (tuple[int, int]), 'prev_grid' (last grid before action),
            'curr_grid' (grid after action), 'level_delta' (int).
        Only the agent's OWN emitted actions populate this list.

    NO game-specific constants are introduced here (Contract C2).
    """
    grid = _latest_grid_from_frame(getattr(frame, "frame", None))
    if grid is None:
        # Defensive: empty frame -> empty state, no crash.
        return {
            "visible_regions": [],
            "marker_neighbor_states": [],
            "last_observation": {
                "primary_region_id": None,
                "level_delta": 0,
                "dominant_transition": {"from": None, "to": None, "count": 0},
            },
            "primary_region_id": None,
            "dominant_transition": {"from": None, "to": None, "count": 0},
            "level_delta": 0,
        }
    comps = _flood_fill_components(grid)
    _compute_neighbors_3x3(comps)
    markers = _markers_from_components(comps, action_history)

    last_entry = action_history[-1] if action_history else None
    last_coord = last_entry["coord_xy"] if last_entry else None
    prev_grid = last_entry["prev_grid"] if last_entry else None
    dt = _dominant_transition(prev_grid, grid)

    levels_completed = int(getattr(frame, "levels_completed", 0) or 0)
    level_delta = max(0, levels_completed - int(prev_levels_completed or 0))

    primary_rid = _primary_region_id_from_last_click(comps, last_coord)

    # Strip private fields before exposing visible_regions to the framework.
    visible_regions = [
        {
            "id": c["id"],
            "region_id": c["id"],
            "bbox": c["bbox"],
            "size": c["size"],
            "color": c["color"],
            "is_multicolor": c["is_multicolor"],
            "y_band": c["y_band"],
            "neighbors_3x3": c.get("neighbors_3x3", {}),
            "is_marker_neighbor": any(
                c["id"] in (m["compass"][d]["region_id"] for d in m["compass"])
                for m in markers
            ),
            "is_primary_marker": any(m["marker_id"] == c["id"] for m in markers),
            "click_response": {
                "clicks": _click_count_in_region(c, action_history),
                "responses": 0,
            },
        }
        for c in comps
    ]

    return {
        "visible_regions": visible_regions,
        "marker_neighbor_states": markers,
        "last_observation": {
            "primary_region_id": primary_rid,
            "level_delta": level_delta,
            "dominant_transition": dt,
        },
        # ArcgenticaLite._last_observation also accepts top-level shortcuts:
        "primary_region_id": primary_rid,
        "dominant_transition": dt,
        "level_delta": level_delta,
        "coord": list(last_coord) if last_coord else None,
        # v605 arm7: expose raw grid for multimodal proposer rendering.
        "_latest_grid": grid,
    }
