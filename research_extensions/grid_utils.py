from __future__ import annotations

import hashlib
from typing import Any


def _normalize_grid(grid: Any) -> list[list[int]]:
    if grid is None:
        return []
    if hasattr(grid, "tolist"):
        grid = grid.tolist()
    if not isinstance(grid, (list, tuple)):
        return []
    rows: list[list[int]] = []
    for row in grid:
        if hasattr(row, "tolist"):
            row = row.tolist()
        if isinstance(row, (list, tuple)):
            rows.append([int(cell) for cell in row])
    return rows


def current_grid(frame: Any) -> list[list[int]]:
    """Return the current visible grid for a FrameData-like object.

    Upstream Arcgentica treats ``data.frame[-1]`` as the current level grid and
    reserves earlier layers for winning-frame snapshots on level transitions.
    """
    grid = _normalize_grid(getattr(frame, "grid", None))
    if grid:
        return grid
    grid_np = _normalize_grid(getattr(frame, "grid_np", None))
    if grid_np:
        return grid_np
    layers = getattr(frame, "frame", None) or []
    if not layers:
        return []
    return _normalize_grid(layers[-1])


def grid_diff_magnitude(before: Any, after: Any) -> float:
    before_grid = current_grid(before)
    after_grid = current_grid(after)
    if not before_grid or not after_grid:
        return 0.0

    diff = 0
    for row_a, row_b in zip(before_grid, after_grid):
        if row_a == row_b:
            continue
        for a, b in zip(row_a, row_b):
            if a != b:
                diff += 1
    return float(diff)


def grid_signature(frame: Any) -> str:
    grid = current_grid(frame)
    if not grid:
        return "empty"

    digest = hashlib.sha1()
    digest.update(f"{len(grid)}x{len(grid[0]) if grid else 0}|".encode("ascii"))
    for row in grid:
        digest.update(bytes(int(cell) & 0xFF for cell in row))
    return digest.hexdigest()[:20]


def grid_feature_vector(frame: Any, bands: int = 8) -> list[float]:
    """Generic, observable-only feature vector for approximate state matching.

    The vector intentionally uses coarse visible-grid statistics rather than
    hidden progress markers:
    - normalized value histogram over 0..15
    - normalized non-zero mass by row band
    - normalized non-zero mass by column band
    """
    grid = current_grid(frame)
    if not grid:
        return [0.0] * (16 + bands + bands)

    rows = len(grid)
    cols = len(grid[0]) if grid else 0
    total = max(rows * cols, 1)

    value_hist = [0.0] * 16
    row_band_mass = [0.0] * bands
    col_band_mass = [0.0] * bands

    for r, row in enumerate(grid):
        row_band = min(bands - 1, int(r * bands / max(rows, 1)))
        for c, cell in enumerate(row):
            value = int(cell)
            if 0 <= value < 16:
                value_hist[value] += 1.0
            if value != 0:
                col_band = min(bands - 1, int(c * bands / max(cols, 1)))
                row_band_mass[row_band] += 1.0
                col_band_mass[col_band] += 1.0

    nonzero_total = max(sum(row_band_mass), 1.0)
    return (
        [count / total for count in value_hist]
        + [mass / nonzero_total for mass in row_band_mass]
        + [mass / nonzero_total for mass in col_band_mass]
    )


def encode_row(row: list[int], max_cols: int | None = None) -> str:
    cols = row if max_cols is None else row[:max_cols]
    chars: list[str] = []
    for cell in cols:
        if 0 <= cell <= 9:
            chars.append(chr(48 + cell))
        else:
            chars.append(chr(97 + min(int(cell) - 10, 25)))
    return "".join(chars)
