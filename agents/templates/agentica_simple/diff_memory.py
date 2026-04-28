"""Diff memory: global FIFO of observation entries plus coarse/fine cluster views.

Plan v10/v11 §4.5b. Used by M1 to seed hypothesis generation. The memory is a
ring buffer of the most recent ``_MAX_ENTRIES`` observations; cluster views
group observations by (count_bin, transition[, region, full_reset_id]) so M1
sees recurrence without seeing every individual diff.
"""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from typing import Any

_MAX_ENTRIES = 200


def bin_count(n: int | None) -> str:
    """Coarse cell-count bin used as an equivalence axis."""
    if n is None:
        return "0"
    n = int(n)
    if n <= 0:
        return "0"
    if n <= 3:
        return "1-3"
    if n <= 10:
        return "4-10"
    if n <= 30:
        return "11-30"
    return "31+"


@dataclass
class DiffEntry:
    action: str
    changed_cells: int
    dominant_transition: dict | None = None
    primary_region_id: str = "_outside_"
    full_reset_id: int = 0
    level_delta: int = 0
    extras: dict = field(default_factory=dict)

    @classmethod
    def from_observation(cls, observed: dict) -> "DiffEntry":
        return cls(
            action=str(observed.get("action", "")).strip(),
            changed_cells=int(observed.get("changed_cells", 0) or 0),
            dominant_transition=observed.get("dominant_transition"),
            primary_region_id=str(observed.get("primary_region_id", "_outside_")),
            full_reset_id=int(
                observed.get("full_reset_id", observed.get("episode_id", 0)) or 0
            ),
            level_delta=int(observed.get("level_delta", 0) or 0),
            extras={k: v for k, v in observed.items() if k not in {
                "action",
                "changed_cells",
                "dominant_transition",
                "primary_region_id",
                "full_reset_id",
                "episode_id",
                "level_delta",
            }},
        )

    def to_dict(self) -> dict:
        d = asdict(self)
        d["count_bin"] = bin_count(self.changed_cells)
        return d


def _transition_key(t: Any) -> str:
    if isinstance(t, dict):
        f = t.get("from")
        to = t.get("to")
        if f is not None and to is not None:
            return f"{int(f)}->{int(to)}"
    if isinstance(t, str) and t.strip():
        return t.strip()
    return "none"


class DiffMemory:
    """Global FIFO observation memory + cluster views for M1."""

    MAX_ENTRIES: int = _MAX_ENTRIES

    def __init__(self) -> None:
        self._entries: list[DiffEntry] = []
        # v30 Fix A: indices (into self._entries) right AFTER each level_delta>0
        # observation. _infer_gqb_pair uses entries from last_boundary onward
        # so a fresh level resets the inferred toggle pair.
        self._level_boundaries: list[int] = []

    def append(self, observed: dict | DiffEntry) -> None:
        if isinstance(observed, DiffEntry):
            entry = observed
        else:
            entry = DiffEntry.from_observation(observed)
        self._entries.append(entry)
        if entry.level_delta > 0:
            # Boundary marks the index of the FIRST entry on the NEW level.
            # The level-up entry itself is the last one of the OLD level, so
            # the new level starts at len(self._entries) (next append onward).
            self._level_boundaries.append(len(self._entries))
        if len(self._entries) > self.MAX_ENTRIES:
            drop = len(self._entries) - self.MAX_ENTRIES
            self._entries = self._entries[-self.MAX_ENTRIES :]
            self._level_boundaries = [
                b - drop for b in self._level_boundaries if b - drop > 0
            ]

    def last_level_boundary(self) -> int:
        """v30 Fix A: index of first entry on current level (0 if none)."""
        return self._level_boundaries[-1] if self._level_boundaries else 0

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> list[DiffEntry]:
        return list(self._entries)

    def cluster_coarse(self) -> dict[tuple[str, str], list[str]]:
        """Group by (count_bin, dominant_transition_key) -> list of action labels."""
        out: "OrderedDict[tuple[str, str], list[str]]" = OrderedDict()
        for e in self._entries:
            key = (bin_count(e.changed_cells), _transition_key(e.dominant_transition))
            out.setdefault(key, []).append(e.action)
        return dict(out)

    def cluster_fine(self) -> dict[tuple[str, str, str, int], list[str]]:
        """Group by (count_bin, transition, region_id, full_reset_id)."""
        out: "OrderedDict[tuple[str, str, str, int], list[str]]" = OrderedDict()
        for e in self._entries:
            key = (
                bin_count(e.changed_cells),
                _transition_key(e.dominant_transition),
                e.primary_region_id,
                e.full_reset_id,
            )
            out.setdefault(key, []).append(e.action)
        return dict(out)

    def click_history_for_m2(self, last_n: int = 30) -> list[dict]:
        """v24: per-click telemetry for M2 — coord, change_bbox, transition,
        verdict-relevant flags. Lets M2 learn click→Hkx mapping and avoid
        gap zones (coords that produced 0 changed cells).

        v33 effect_type: classify click cardinality so skills/cards can
        distinguish Hkx (single-cell toggle, ~36 cells) vs NTi (multi-cell
        toggle, ~80+ cells) vs no-op (bsT body / empty space, 0 cells) vs
        level_rise (full repaint, 1000+ cells). Without this M3 cannot
        articulate the difference and agent cycles same NTi forever."""
        import re as _re
        out = []
        for e in self._entries[-last_n:]:
            m = _re.search(r"\((\d+),\s*(\d+)\)", e.action)
            x = int(m.group(1)) if m else None
            y = int(m.group(2)) if m else None
            bbox = e.extras.get("change_bbox") if isinstance(e.extras, dict) else None
            cc = int(e.changed_cells or 0)
            ld = int(e.level_delta or 0)
            if ld > 0:
                eff = "level_rise"
            elif cc == 0:
                eff = "no_effect"
            elif cc <= 50:
                eff = "single_toggle"
            else:
                eff = "multi_toggle"
            out.append({
                "click_x": x,
                "click_y": y,
                "changed_cells": cc,
                "change_bbox": bbox,
                "dominant_transition": e.dominant_transition,
                "primary_region_id": e.primary_region_id,
                "is_gap_miss": (cc == 0 and x is not None),
                "effect_type": eff,
            })
        return out

    def known_hkx_states(self) -> dict:
        """v24: track per-change_bbox last-known dominant color. After two
        toggles at the same bbox, the color flips back. Encodes Hkx state
        as 'last write wins' so M2 knows whether next click flips to 9 or 8.

        v31 Fix B: only count entries since the last level transition.
        toggle_count then = clicks on this bbox in the CURRENT level. M2 uses
        this to mark a bbox EXHAUSTED after >=2 clicks with no level rise,
        which generalises beyond 2-color parity (handles 3+ color cycles)."""
        states: "OrderedDict[str, dict]" = OrderedDict()
        boundary = self.last_level_boundary()
        for e in self._entries[boundary:]:
            bbox = e.extras.get("change_bbox") if isinstance(e.extras, dict) else None
            if not isinstance(bbox, dict):
                continue
            key = f"{bbox.get('min_x')},{bbox.get('min_y')}-{bbox.get('max_x')},{bbox.get('max_y')}"
            dt = e.dominant_transition or {}
            to_color = dt.get("to") if isinstance(dt, dict) else None
            from_color = dt.get("from") if isinstance(dt, dict) else None
            states[key] = {
                "bbox": bbox,
                "current_color": to_color,
                "previous_color": from_color,
                "toggle_count": (states.get(key, {}).get("toggle_count", 0) + 1),
            }
        return dict(states)

    def snapshot_for_m1(self) -> dict:
        """JSON-serialisable snapshot for M1 input."""
        recent = [e.to_dict() for e in self._entries[-40:]]
        coarse = {
            f"{cb}|{tr}": acts
            for (cb, tr), acts in self.cluster_coarse().items()
        }
        fine = {
            f"{cb}|{tr}|{rid}|fr{fr}": acts
            for (cb, tr, rid, fr), acts in self.cluster_fine().items()
        }
        return {
            "full_log_recent": recent,
            "cluster_coarse": coarse,
            "cluster_fine": fine,
            "total_entries": len(self._entries),
        }
