"""Codex objective-reviewer Q4: marker-neighbor perturbation audit.

The translator must be EQUIVARIANT under color permutation (color labels are
arbitrary in ARC-AGI-3; same geometry under different palette must produce
same marker IDs and same compass relations).

If marker identity or compass topology changes under color remap, that
indicates color-specific leakage.
"""

from __future__ import annotations

from agents.templates.agentica_lite._frame_to_state import (
    _flood_fill_components,
    _compute_neighbors_3x3,
    _markers_from_components,
    frame_to_state,
)


class _StubFrame:
    def __init__(self, grid: list[list[int]], levels_completed: int = 0):
        self.frame = [grid]
        self.levels_completed = levels_completed
        self.state = "NOT_STARTED"
        self.available_actions = []


def _build_grid_with_two_markers() -> list[list[int]]:
    """64x64 grid with two distinct multicolor markers + several
    single-color neighbors. Geometry: markers at (10,10) and (10,40)
    with 3x3 multicolor pattern; neighbors at the 8 compass slots
    around each marker as 3x3 single-color blocks.
    """
    H, W = 64, 64
    g = [[0 for _ in range(W)] for _ in range(H)]

    # marker A at (10,10) — 3x3 multicolor (alternating colors 1/2)
    for r in range(9, 12):
        for c in range(9, 12):
            g[r][c] = 1 if (r + c) % 2 == 0 else 2

    # marker B at (10,40) — 3x3 multicolor (alternating 3/4)
    for r in range(9, 12):
        for c in range(39, 42):
            g[r][c] = 3 if (r + c) % 2 == 0 else 4

    # 8 single-color 3x3 neighbors around marker A
    # Disjoint palettes so palette permutation doesn't create palette-A
    # / palette-B color collisions at boundaries (which would change
    # populated counts under permutation due to local merge effects).
    palette_A = [5, 6, 7, 8, 9, 10, 11, 12]
    palette_B = [15, 16, 17, 18, 19, 20, 21, 22]
    offsets = [(-3, -3), (-3, 0), (-3, 3), (0, 3), (3, 3), (3, 0), (3, -3), (0, -3)]
    for color, (dr, dc) in zip(palette_A, offsets):
        for r in range(10 + dr - 1, 10 + dr + 2):
            for c in range(10 + dc - 1, 10 + dc + 2):
                if 0 <= r < H and 0 <= c < W:
                    g[r][c] = color
    for color, (dr, dc) in zip(palette_B, offsets):
        for r in range(10 + dr - 1, 10 + dr + 2):
            for c in range(40 + dc - 1, 40 + dc + 2):
                if 0 <= r < H and 0 <= c < W:
                    g[r][c] = color
    return g


def _permute_colors(grid: list[list[int]], mapping: dict[int, int]) -> list[list[int]]:
    return [[mapping.get(v, v) for v in row] for row in grid]


def _markers_signature(markers: list[dict]) -> list[tuple]:
    """Reduce markers to (compass_direction_set, populated_count) tuples
    sorted by something positional. Color-invariant signature."""
    sig = []
    for m in markers:
        compass = m["compass"]
        dirs = sorted(compass.keys())
        populated = sum(1 for d in dirs if compass[d]["region_id"] is not None)
        sig.append((tuple(dirs), populated))
    return sorted(sig)


def test_perturbation_color_permutation_equivariance():
    """Permute colors -> identical marker count, IDs (spatial), and per-marker
    populated counts. This is the geometric-invariance check codex Q4 requires."""
    grid = _build_grid_with_two_markers()
    frame_a = _StubFrame(grid)
    state_a = frame_to_state(frame_a, action_history=[], prev_levels_completed=0)

    perm = {1: 4, 2: 3, 3: 2, 4: 1,
            5: 12, 6: 11, 7: 10, 8: 9, 9: 8, 10: 7, 11: 6, 12: 5,
            15: 22, 16: 21, 17: 20, 18: 19, 19: 18, 20: 17, 21: 16, 22: 15}
    grid_perm = _permute_colors(grid, perm)
    frame_b = _StubFrame(grid_perm)
    state_b = frame_to_state(frame_b, action_history=[], prev_levels_completed=0)

    # Same component + marker counts
    assert len(state_a["visible_regions"]) == len(state_b["visible_regions"])
    assert len(state_a["marker_neighbor_states"]) == len(state_b["marker_neighbor_states"])

    # Same marker IDs (deterministic spatial ordering)
    ids_a = sorted(m["marker_id"] for m in state_a["marker_neighbor_states"])
    ids_b = sorted(m["marker_id"] for m in state_b["marker_neighbor_states"])
    assert ids_a == ids_b, f"marker ID set changed under color perm"

    # Per-marker populated count must match (compass topology invariant)
    pop_a = {
        m["marker_id"]: sum(1 for v in m["compass"].values() if v.get("region_id"))
        for m in state_a["marker_neighbor_states"]
    }
    pop_b = {
        m["marker_id"]: sum(1 for v in m["compass"].values() if v.get("region_id"))
        for m in state_b["marker_neighbor_states"]
    }
    diffs = [k for k in pop_a if pop_a.get(k) != pop_b.get(k)]
    assert not diffs, f"per-marker populated count differs at: {diffs[:3]}"


def test_perturbation_horizontal_mirror():
    """Mirror horizontally -> same marker COUNT and compass populated count;
    direction labels swap E<->W, NE<->NW, SE<->SW (geometric equivariance)."""
    grid = _build_grid_with_two_markers()
    frame_a = _StubFrame(grid)
    state_a = frame_to_state(frame_a, action_history=[], prev_levels_completed=0)

    grid_mirror = [list(reversed(row)) for row in grid]
    frame_b = _StubFrame(grid_mirror)
    state_b = frame_to_state(frame_b, action_history=[], prev_levels_completed=0)

    # Same component count + same marker count (geometry is a bijection)
    assert len(state_a["visible_regions"]) == len(state_b["visible_regions"])
    assert len(state_a["marker_neighbor_states"]) == len(state_b["marker_neighbor_states"])

    # Compass populated counts must match (mirror only swaps direction labels)
    pop_a = sorted(
        sum(1 for v in m["compass"].values() if v["region_id"] is not None)
        for m in state_a["marker_neighbor_states"]
    )
    pop_b = sorted(
        sum(1 for v in m["compass"].values() if v["region_id"] is not None)
        for m in state_b["marker_neighbor_states"]
    )
    assert pop_a == pop_b, f"populated counts differ: {pop_a} vs {pop_b}"


def test_perturbation_color_permutation_preserves_marker_id_set():
    """Codex Q4: color permutation must NOT change which spatial positions
    are detected as markers. The set of marker IDs (deterministic spatial
    ranks) is the cleanest single-property invariant."""
    grid = _build_grid_with_two_markers()
    state_a = frame_to_state(_StubFrame(grid), action_history=[], prev_levels_completed=0)

    # Strong permutation: shuffle every used color
    perm = {1: 4, 2: 3, 3: 2, 4: 1,
            5: 12, 6: 11, 7: 10, 8: 9, 9: 8, 10: 7, 11: 6, 12: 5,
            15: 22, 16: 21, 17: 20, 18: 19, 19: 18, 20: 17, 21: 16, 22: 15}
    grid_perm = _permute_colors(grid, perm)
    state_b = frame_to_state(_StubFrame(grid_perm), action_history=[], prev_levels_completed=0)

    ids_a = sorted(m["marker_id"] for m in state_a["marker_neighbor_states"])
    ids_b = sorted(m["marker_id"] for m in state_b["marker_neighbor_states"])
    assert ids_a == ids_b, (
        f"marker ID set diverged under color perm; codex Q4 leakage signal. "
        f"A={len(ids_a)} markers vs B={len(ids_b)} markers"
    )
