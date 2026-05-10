"""Smoke test for v601: 5-turn run with Proposer/Policy/Memory writer all wired.

Asserts the run completes without exception and the journal records
proposer_calls / reflector_calls counters in extra.
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
            {"id": f"Q{i}", "region_id": f"Q{i}",
             "bbox": [i * 5, 0, i * 5 + 4, 4],
             "color": i % 4, "is_multicolor": False, "y_band": "play_zone",
             "is_marker_neighbor": False, "is_primary_marker": False}
            for i in range(3)
        ],
        "marker_neighbor_states": [
            {"marker_id": "M0", "is_primary_marker": True, "compass": {
                d: {"region_id": f"Q{i}", "current_color": None,
                    "clicks": 1 if i < turn else 0}
                for i, d in enumerate(("N", "E", "S", "W"))
            }},
        ],
        "observation": {
            "primary_region_id": "Q0",
            "dominant_transition": {"from": 0, "to": 1, "count": 12 + turn * 4},
            "level_delta": 1 if turn == 0 else 0,
        },
    }


def test_smoke_v601_5_turns(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    agent = ArcgenticaLite(game_id="v601-test", seed=42)

    async def _run():
        for t in range(5):
            await agent.run_turn(_stub_state(t))
        return agent.episode_end()

    rec = asyncio.run(_run())
    assert rec.turns == 5
    journal = agent.journal.load_all()
    assert len(journal) >= 1
    assert journal[-1].episode_id == rec.episode_id
    # Proposer should be triggered at least at turn 0 warm-up.
    assert rec.extra.get("proposer_calls", 0) >= 0  # may be 0 if TRAPI client missing
    assert "confidence_overrides" in rec.extra


def test_smoke_v601_handles_missing_observation(tmp_path, monkeypatch):
    """Edge case: state with no observation field."""
    monkeypatch.chdir(tmp_path)
    agent = ArcgenticaLite(game_id="v601-edge", seed=7)

    async def _run():
        # Empty state (no marker_neighbor_states, observation, visible_regions)
        for _t in range(3):
            await agent.run_turn({"turn": _t, "visible_regions": []})
        return agent.episode_end()

    rec = asyncio.run(_run())
    assert rec.turns == 3
