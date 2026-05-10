"""Append-only single-writer journal at episode end (plan §6.3).

NOT cross_run_memory.json (which v52-v593 had race issues with).
Schema: one row per episode in `simple_logs/<game_id>/agentica_lite_journal.jsonl`.

RASI prior derivation (plan §1.6) reads this journal at next episode start
and applies retention scoring (plan §1.11) to filter survivors.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class EpisodeRecord:
    episode_id: str
    seed: int
    framework_version: str  # "v600"
    git_sha: str  # plan §1.17 hash-and-freeze
    ts_start: float
    ts_end: float
    turns: int
    max_level: int
    # plan §1.18 RASI ablation arm label
    rasi_arm: str = "null"  # "null" | "targeted" | "shuffle_<seed>" | "phase_shift" | "variance_match" | "sign_invert" | "uniform_cal"
    predicates_installed: list[dict] = field(default_factory=list)
    stalemate_events: int = 0
    posterior_top10_at_end: list[tuple[str, str, float]] = field(default_factory=list)
    # latency budget telemetry (plan §1.14)
    latency_p50: float = 0.0
    latency_p95: float = 0.0
    latency_worst: float = 0.0
    extra: dict[str, Any] = field(default_factory=dict)


class MemoryJournal:
    """Append-only writer. One file per game_id."""

    def __init__(self, game_id: str, base_dir: str | Path = "simple_logs") -> None:
        self.path = Path(base_dir) / game_id / "agentica_lite_journal.jsonl"
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, record: EpisodeRecord) -> None:
        """Atomic append. fsync on close to survive crash."""
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(record), separators=(",", ":")) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def load_all(self) -> list[EpisodeRecord]:
        """Read all records (small files; per-game journal grows slowly)."""
        if not self.path.exists():
            return []
        records: list[EpisodeRecord] = []
        with open(self.path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                d = json.loads(line)
                records.append(EpisodeRecord(**d))
        return records
