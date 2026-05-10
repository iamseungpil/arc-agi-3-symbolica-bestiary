"""Fixture loader + runner for plan v600 §5 fixture suite.

Three fixture pools:
  - Synthetic (≥30): hand-labeled F1-F6 failure modes (plan §5.1).
  - Replay (≥20): real turn snapshots from failed cycles.
  - Blind held-out (≥20): sealed seeds, run only at end of Phase D.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Fixture:
    id: str
    failure_mode: str  # F1-F6
    pool: str  # "synthetic" | "replay" | "held_out"
    source_trace: str | None
    source_turn: int | None
    input: dict
    expected_module: str  # "predicate_posterior" | "stalemate_trigger" | "llm_extender" | "library_install" | "memory_journal"
    expected_behavior: str
    expected_assertion: str  # Python expression eval'd against module output
    split: str  # "train" | "val" | "blind"


def load_fixtures(fixtures_dir: str | Path) -> list[Fixture]:
    """Load all .json fixtures from a directory tree."""
    base = Path(fixtures_dir)
    out: list[Fixture] = []
    for p in base.rglob("*.json"):
        if p.name.startswith("_"):
            continue
        with open(p, encoding="utf-8") as f:
            d = json.load(f)
        out.append(Fixture(**d))
    return out


def split_pass_rates(results: list[tuple[Fixture, bool]]) -> dict[str, float]:
    """Aggregate pass rates by split."""
    counts: dict[str, list[bool]] = {"train": [], "val": [], "blind": []}
    for fx, ok in results:
        counts.setdefault(fx.split, []).append(ok)
    return {
        k: (sum(v) / len(v) if v else 0.0)
        for k, v in counts.items()
    }
