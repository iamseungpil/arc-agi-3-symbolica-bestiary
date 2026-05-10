"""Smoke test: 5-turn run on a stub state should complete without exception
and the journal should record one episode.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.templates.agentica_lite import ArcgenticaLite  # noqa: E402


def _stub_state(turn: int):
    return {
        "turn": turn,
        "visible_regions": [
            {"id": f"R{i}", "region_id": f"R{i}",
             "bbox": {"min_x": i * 5, "min_y": 0, "max_x": i * 5 + 4, "max_y": 4},
             "color": i % 4, "is_multicolor": False, "y_band": "play_zone",
             "is_marker_neighbor": False, "is_primary_marker": False}
            for i in range(3)
        ],
        "last_observation": {
            "primary_region_id": "R0",
            "dominant_transition": {"from": 0, "to": 1, "count": 12},
            "level_delta": 1 if turn == 0 else 0,
        },
    }


def test_smoke_5_turns(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = ArcgenticaLite(game_id="ft09-test", seed=42)

    async def _run():
        for t in range(5):
            await agent.run_turn(_stub_state(t))
        rec = agent.episode_end()
        return rec

    rec = asyncio.run(_run())
    assert rec.turns == 5
    journal = agent.journal.load_all()
    assert len(journal) >= 1
    assert journal[-1].episode_id == rec.episode_id
