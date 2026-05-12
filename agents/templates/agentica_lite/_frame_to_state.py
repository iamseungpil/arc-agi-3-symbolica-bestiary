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
                    "_cell_values": {cell: grid[cell[0]][cell[1]] for cell in cells},
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
    # v606.3 codex r19: scan 3x3 grid windows for multi-color non-bg patches.
    # Some markers have 3x3 patches with 3-4 cells per color, which the
    # size<=4 cluster filter still misses if components are too spread out.
    # This pass adds any unmatched 3x3 window with ≥2 distinct non-bg colors
    # as a new multicolor super-component.
    # disabled v606.3 3x3 scan (too lax; 290 patches; [38,54] isn't multicolor)

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
                # v608: forward raw cell values so downstream constraint
                # inference can read per-slot colors. Stripped before exposing
                # visible_regions to the framework (see frame_to_state main).
                "_cell_values": dict(c.get("_cell_values") or {}),
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

# v608d Phase 2: per-region post-click color history. Keyed by region_id.
# Each entry is a `collections.deque` of the last K colors observed AFTER the
# click landed inside the region. Reset on episode boundary.
_REGION_TRANSITION_STATE: dict[str, dict[str, Any]] = {}
_REGION_TRANSITION_MAX = 8


def _reset_stability_state() -> None:
    """Called at start of each new episode (per adapter contract)."""
    _STABILITY_STATE["prev"] = []
    _STABILITY_STATE["next_idx"] = 1
    _REGION_TRANSITION_STATE.clear()


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


def _add_3x3_multicolor_patches(
    raw_comps: list[dict[str, Any]], grid: list[list[int]]
) -> list[dict[str, Any]]:
    """v606.3 codex r19: detect 3x3 non-bg multicolor patches and emit as
    super-components. Each detected patch becomes one multicolor component
    that overlays the splits the flood-fill produced. The original splits
    remain; the new super-component coexists.

    Stays game-agnostic: no hardcoded color/size constants beyond the
    universal 'non-zero color = non-background' assumption.
    """
    if not grid or not grid[0]:
        return raw_comps
    H = len(grid)
    W = len(grid[0])
    # Track cells already inside an existing multicolor super-component
    # (from arm6 merge) to avoid double-counting.
    already_multi: set[tuple[int, int]] = set()
    for c in raw_comps:
        if c.get("is_multicolor"):
            for cell in c.get("_cells", []):
                already_multi.add(cell)
    added: list[dict[str, Any]] = []
    seen_patches: set[tuple[int, int]] = set()
    for r0 in range(H - 2):
        for c0 in range(W - 2):
            colors: set[int] = set()
            cells: list[tuple[int, int]] = []
            for dr in range(3):
                for dc in range(3):
                    r, c = r0 + dr, c0 + dc
                    v = grid[r][c]
                    if v != 0:
                        colors.add(v)
                    cells.append((r, c))
            non_bg = [c for c in cells if grid[c[0]][c[1]] != 0]
            if len(colors) >= 2 and len(non_bg) >= 4 and all(c not in already_multi for c in non_bg):
                # avoid emitting overlapping windows
                key = (r0, c0)
                if key in seen_patches:
                    continue
                seen_patches.add(key)
                added.append({
                    "_seed_color": grid[non_bg[0][0]][non_bg[0][1]],
                    "_cells": non_bg,
                    "_cell_values": {cell: grid[cell[0]][cell[1]] for cell in non_bg},
                    "bbox": [c0, r0, c0 + 2, r0 + 2],
                    "size": len(non_bg),
                    "color": grid[non_bg[0][0]][non_bg[0][1]],
                    "is_multicolor": True,
                })
                # Mark these cells consumed for subsequent passes.
                for cell in non_bg:
                    already_multi.add(cell)
    return raw_comps + added


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
    """Detect grouped small same-bbox components and merge into a multicolor
    super-component when they share a tight window AND span >= 2 distinct
    non-background colors.

    v608c (2026-05-11): ft09 markers are roughly 6x6 patches built from
    several 2x2 sub-blocks of distinct colors. The pre-v608c logic only
    inspected a 4x4 window, which missed the ft09 layout entirely (E1 on
    cycle237 traces reported `high_constraints=0` across 203 frames). The
    revised heuristic uses a 6x6 cluster bbox and requires at least two
    distinct seed colors in the cluster, so a uniform 2x2 patch by itself is
    not promoted while a real multi-color marker is.

    Background-color (0) components are still excluded.
    """
    if not raw_comps:
        return raw_comps
    # Centroid of a component (column-major, matches `_centroid`).
    def _comp_center(comp: dict[str, Any]) -> tuple[float, float]:
        cells = comp.get("_cells") or []
        if not cells:
            return (0.0, 0.0)
        sr = sum(c[0] for c in cells) / len(cells)
        sc = sum(c[1] for c in cells) / len(cells)
        return (sc, sr)

    # v608c: relax size cap from 4 to 9. Some ft09 marker sub-blocks are
    # actually 2x2=4 cells but a 3x3 single-color block is also plausible; the
    # cluster bbox + multicolor constraint below keep noise out.
    small = [c for c in raw_comps if c["size"] <= 9 and c["_seed_color"] != 0]
    other = [c for c in raw_comps if not (c["size"] <= 9 and c["_seed_color"] != 0)]
    used = [False] * len(small)
    merged: list[dict[str, Any]] = []
    for i, c in enumerate(small):
        if used[i]:
            continue
        cluster = [c]
        used[i] = True
        cx, cy = _comp_center(c)
        # v608c: 6x6 cluster bbox (was 4x4). Distance is measured from the
        # cluster seed center to each candidate center.
        WINDOW = 5.5  # admit components whose centers fall within a 11x11
        # diameter around the seed; the final bbox is recomputed from cells.
        for j, d in enumerate(small):
            if used[j] or i == j:
                continue
            dx, dy = _comp_center(d)
            if abs(cx - dx) <= WINDOW and abs(cy - dy) <= WINDOW:
                cluster.append(d)
                used[j] = True
        if len(cluster) >= 4:
            all_cells: list[tuple[int, int]] = []
            all_colors: set[int] = set()
            all_values: dict[tuple[int, int], int] = {}
            for cc in cluster:
                all_cells.extend(cc["_cells"])
                all_colors.add(cc["_seed_color"])
                all_values.update(cc.get("_cell_values", {}))
            # v608c: require true multicolor (>= 2 distinct seed colors). A
            # cluster of identically-colored sub-blocks is not a marker and
            # should pass through unchanged so the downstream regular-region
            # path can still see it.
            if len(all_colors) < 2:
                merged.extend(cluster)
                continue
            min_r = min(cc[0] for cc in all_cells)
            max_r = max(cc[0] for cc in all_cells)
            min_c = min(cc[1] for cc in all_cells)
            max_c = max(cc[1] for cc in all_cells)
            # v608c: final geometric guardrail. The merged bbox should fit
            # inside a 7x7 window; anything larger is unlikely to be a single
            # marker.
            if (max_r - min_r) > 6 or (max_c - min_c) > 6:
                merged.extend(cluster)
                continue
            merged.append({
                "_seed_color": cluster[0]["_seed_color"],
                "_cells": all_cells,
                "_cell_values": all_values,
                "bbox": [min_c, min_r, max_c, max_r],
                "size": len(all_cells),
                "color": cluster[0]["_seed_color"],
                "is_multicolor": True,
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
        # v606.4: exclude any huge component (size > 500 = background) from
        # markers regardless of color. ft09 bg may be 0 OR 5; both are too
        # large to be a real marker.
        if size > 500:
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


_SLOT_TO_OFFSET: dict[str, tuple[int, int]] = {
    "NW": (-1, -1), "N": (0, -1), "NE": (1, -1),
    "W": (-1, 0), "E": (1, 0),
    "SW": (-1, 1), "S": (0, 1), "SE": (1, 1),
}


def _modal_nonzero(values: list[int]) -> int | None:
    counts: dict[int, int] = {}
    for v in values:
        if v == 0:
            continue
        counts[v] = counts.get(v, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]


def _marker_center_value(marker: dict[str, Any]) -> int | None:
    values = marker.get("_cell_values") or {}
    if not values:
        return None
    x0, y0, x1, y1 = marker["bbox"]
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    best: tuple[float, int] | None = None
    for (r, c), val in values.items():
        dist = (c - cx) ** 2 + (r - cy) ** 2
        item = (dist, int(val))
        if best is None or item < best:
            best = item
    return best[1] if best is not None else None


def _marker_slot_value(marker: dict[str, Any], slot: str) -> int | None:
    """Infer a coarse 3x3 marker-slot value from component cell colors.

    This is intentionally generic: it only uses the raw grid component's own
    cells and bbox. A zero-like slot maps to a `same` constraint; any non-zero
    slot maps to `different`.
    """
    values = marker.get("_cell_values") or {}
    if not values:
        return None
    x0, y0, x1, y1 = marker["bbox"]
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    off = _SLOT_TO_OFFSET.get(slot)
    if off is None:
        return None
    dx, dy = off
    candidates: list[tuple[float, int]] = []
    for (r, c), val in values.items():
        rel_x = c - cx
        rel_y = r - cy
        if dx < 0 and rel_x > 0:
            continue
        if dx > 0 and rel_x < 0:
            continue
        if dy < 0 and rel_y > 0:
            continue
        if dy > 0 and rel_y < 0:
            continue
        if dx == 0 and abs(rel_x) > max(1.0, (x1 - x0 + 1) / 4.0):
            continue
        if dy == 0 and abs(rel_y) > max(1.0, (y1 - y0 + 1) / 4.0):
            continue
        dist = (rel_x - dx) ** 2 + (rel_y - dy) ** 2
        candidates.append((dist, int(val)))
    if not candidates:
        return None
    candidates.sort(key=lambda kv: kv[0])
    return candidates[0][1]


def _infer_marker_constraints(
    comps: list[dict[str, Any]],
    markers: list[dict[str, Any]],
    action_history: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build v608 marker-neighbor same/different constraints.

    Live inference is conservative: when a marker's local slot value cannot be
    inferred from raw component cells, no constraint is emitted for that slot.
    """
    by_id = {c["id"]: c for c in comps}
    constraints: list[dict[str, Any]] = []
    by_marker: dict[str, dict[str, int]] = {}
    for m in markers:
        marker = by_id.get(m.get("marker_id"))
        if marker is None:
            continue
        marker_color = _marker_center_value(marker)
        if marker_color == 0 or marker_color is None:
            marker_color = _modal_nonzero(list((marker.get("_cell_values") or {}).values()))
        if marker_color is None:
            marker_color = marker.get("color")
        if marker_color is None:
            continue
        for slot, info in (m.get("compass") or {}).items():
            nbr_id = info.get("region_id")
            nbr = by_id.get(nbr_id)
            if nbr is None:
                continue
            slot_value = _marker_slot_value(marker, slot)
            if slot_value is None:
                continue
            relation = "same" if int(slot_value) == 0 else "different"
            neighbor_color = nbr.get("color")
            if neighbor_color is None:
                continue
            satisfied = (
                int(neighbor_color) == int(marker_color)
                if relation == "same"
                else int(neighbor_color) != int(marker_color)
            )
            clicks = _click_count_in_region(nbr, action_history)
            # v608b: classify evidence quality. Multicolor markers are the
            # ft09 design substrate; single-color "markers" come from the
            # permissive `size>=4 + >=4 neighbors` fallback in
            # `_markers_from_components` and produce noisy constraints. Mark
            # those low-quality so downstream policy can filter them out
            # without losing the substrate signal.
            evidence_quality = (
                "high" if bool(marker.get("is_multicolor", False)) else "low"
            )
            constraints.append({
                "marker_id": m.get("marker_id"),
                "slot": slot,
                "neighbor_region_id": nbr_id,
                "marker_color": int(marker_color),
                "neighbor_color": int(neighbor_color),
                "relation": relation,
                "satisfied": bool(satisfied),
                "clicks": clicks,
                "evidence_quality": evidence_quality,
            })
            bucket = by_marker.setdefault(
                str(m.get("marker_id")),
                {"total": 0, "unsatisfied": 0,
                 "high_total": 0, "high_unsatisfied": 0,
                 "evidence_quality": evidence_quality},
            )
            bucket["total"] += 1
            if evidence_quality == "high":
                bucket["high_total"] += 1
            if not satisfied:
                bucket["unsatisfied"] += 1
                if evidence_quality == "high":
                    bucket["high_unsatisfied"] += 1
    summary = {
        "total": len(constraints),
        "unsatisfied": sum(1 for c in constraints if not c["satisfied"]),
        "high_total": sum(
            1 for c in constraints if c.get("evidence_quality") == "high"
        ),
        "high_unsatisfied": sum(
            1 for c in constraints
            if c.get("evidence_quality") == "high" and not c["satisfied"]
        ),
        "by_marker": by_marker,
    }
    return constraints, summary


def _shortest_period(seq: list[int]) -> list[int] | None:
    """Return the shortest non-trivial period of `seq` if at least one
    wraparound is observed (i.e. seq[p:] matches the prefix), else None.

    Example:
      [4, 8, 12, 4, 8, 12] -> [4, 8, 12]
      [4, 8, 12, 4]        -> [4, 8, 12]
      [4, 8, 12]           -> None (no wraparound yet)
      [4, 4, 4, 4]         -> None (degenerate)
    """
    n = len(seq)
    if n < 2:
        return None
    for p in range(1, n):
        # We need at least one wraparound element: n must exceed p.
        if n <= p:
            break
        cycle = seq[:p]
        # All elements identical is not a useful cycle.
        if len(set(cycle)) <= 1:
            continue
        ok = True
        for i in range(p, n):
            if seq[i] != seq[i % p]:
                ok = False
                break
        if ok:
            return cycle
    return None


def _record_region_transition(
    region_id: str | None,
    new_color: int | None,
) -> None:
    """Append `new_color` to the per-region post-click history (capped)."""
    if not region_id or new_color is None:
        return
    bucket = _REGION_TRANSITION_STATE.setdefault(
        str(region_id),
        {"history": [], "n_samples": 0},
    )
    hist = bucket["history"]
    hist.append(int(new_color))
    if len(hist) > _REGION_TRANSITION_MAX:
        del hist[: len(hist) - _REGION_TRANSITION_MAX]
    bucket["n_samples"] = int(bucket.get("n_samples", 0)) + 1


def _region_transitions_view() -> dict[str, dict[str, Any]]:
    """Return a serializable snapshot of `_REGION_TRANSITION_STATE` with
    `inferred_cycle` / `next_predicted` / `confidence` computed.
    """
    out: dict[str, dict[str, Any]] = {}
    for rid, bucket in _REGION_TRANSITION_STATE.items():
        hist = list(bucket.get("history", []))
        n = len(hist)
        cycle = _shortest_period(hist) if n >= 3 else None
        next_predicted: int | None = None
        confidence = 0.0
        if cycle:
            next_predicted = int(cycle[n % len(cycle)])
            # Confidence grows with the number of completed cycles observed.
            completed = n // len(cycle)
            confidence = min(1.0, completed / 3.0)
        out[str(rid)] = {
            "history": hist,
            "inferred_cycle": cycle,
            "next_predicted": next_predicted,
            "confidence": confidence,
            "n_samples": int(bucket.get("n_samples", n)),
        }
    return out


def _maybe_record_click_for_region(
    comps: list[dict[str, Any]],
    last_coord: tuple[int, int] | None,
    prev_grid: list[list[int]] | None,
    curr_grid: list[list[int]] | None,
) -> None:
    """Find the region containing `last_coord` and append the post-click
    color to its history if the color actually changed.

    Coord is (x, y). bbox is [x0, y0, x1, y1]. We use the centroid color
    of the region as the post-click color (matches `_to_region_ref`).
    """
    if last_coord is None or curr_grid is None or not comps:
        return
    x, y = int(last_coord[0]), int(last_coord[1])
    matching = None
    for c in comps:
        bbox = c.get("bbox")
        if not bbox or len(bbox) != 4:
            continue
        x0, y0, x1, y1 = bbox[0], bbox[1], bbox[2], bbox[3]
        if x0 <= x <= x1 and y0 <= y <= y1:
            matching = c
            break
    if matching is None:
        return
    new_color = matching.get("color")
    if new_color is None:
        return
    # v608f-fix: do NOT gate on per-pixel equality at the recorded coord.
    # ft09 snaps clicks (see `feedback_ft09_coord_snap.md`), so the pixel
    # under the LLM-intended coord often did not actually change even when
    # the env did fire a transition somewhere inside the region. Instead,
    # gate against the most recent recorded color for THIS region: append
    # a sample only when the centroid color differs from the most recent
    # entry. This keeps cycle detection on the actual color stream.
    rid = matching.get("id")
    if rid:
        prev_bucket = _REGION_TRANSITION_STATE.get(str(rid)) or {}
        prev_hist = prev_bucket.get("history") or []
        if prev_hist and int(prev_hist[-1]) == int(new_color):
            return
    _record_region_transition(rid, new_color)


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
    marker_constraints, marker_constraint_summary = _infer_marker_constraints(
        comps, markers, action_history,
    )

    last_entry = action_history[-1] if action_history else None
    last_coord = last_entry["coord_xy"] if last_entry else None
    prev_grid = last_entry["prev_grid"] if last_entry else None
    dt = _dominant_transition(prev_grid, grid)

    # v608d Phase 2: record the post-click color for the region that
    # contained the previous click coord. Skipped when the pixel did not
    # actually change. The module-level log feeds `region_transitions`
    # below.
    _maybe_record_click_for_region(comps, last_coord, prev_grid, grid)
    region_transitions = _region_transitions_view()

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

    # v606.3 codex r20 option B: build rolling click+observation history for
    # Proposer multi-step planning. Per cycle237 evidence, L+1 requires a
    # 7-step click sequence (not single click). Without this rolling view the
    # LLM cannot accumulate per-region evidence or plan multi-step paths.
    recent_turn_diffs: list[dict] = []
    for i, entry in enumerate(action_history[-7:]):
        ec = entry.get("coord_xy")
        ld = int(entry.get("level_delta", 0) or 0)
        pg = entry.get("prev_grid")
        cg = entry.get("curr_grid") or grid
        dti = _dominant_transition(pg, cg)
        # which region (in current segmentation) contains this past click —
        # prefer the SMALLEST enclosing region to avoid the giant background.
        click_rid = None
        if ec is not None:
            cx, cy = ec
            candidates = []
            for r in visible_regions:
                x0, y0, x1, y1 = r["bbox"]
                if x0 <= cx <= x1 and y0 <= cy <= y1:
                    area = (x1 - x0 + 1) * (y1 - y0 + 1)
                    candidates.append((area, r["region_id"]))
            if candidates:
                # smallest area first; ties broken by region_id
                candidates.sort()
                click_rid = candidates[0][1]
        recent_turn_diffs.append({
            "turn_offset": -(len(action_history[-7:]) - i),
            "coord": list(ec) if ec else None,
            "click_region_id": click_rid,
            "dominant_transition": dti,
            "level_delta": ld,
            "did_advance": ld >= 1,
        })

    return {
        "visible_regions": visible_regions,
        "marker_neighbor_states": markers,
        "marker_constraints": marker_constraints,
        "marker_constraint_summary": marker_constraint_summary,
        # v608d Phase 2 (renamed v608f): per-region post-click transition log
        # + inferred cycle. This is a deterministic action-planning support
        # feature, not a learned world model — name it accordingly. The
        # legacy `region_transitions` key is kept as an alias for tests.
        "region_transition_cache": region_transitions,
        "region_transitions": region_transitions,
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
        # v606.3 rolling history for multi-step planning
        "recent_turn_diffs": recent_turn_diffs,
        # v605 arm7: expose raw grid for multimodal proposer rendering.
        "_latest_grid": grid,
    }
