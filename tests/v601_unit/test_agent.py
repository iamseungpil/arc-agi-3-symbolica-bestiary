"""Per-module unit tests for agent.py (ArcgenticaLite orchestrator).

Plan v602 §11 addendum: 5 critical branch tests.

Branches under test:
  1. proposer-fired (warm-up at turn 0 invokes Proposer; counter increments)
  2. proposer-skipped (when warm-up done + no stalemate, Proposer not called)
  3. episode-end (episode_end() returns EpisodeRecord with v601 framework_version
     and extra counters)
  4. budget-breach (turn that exceeds global_turn_budget_s logs warning but
     completes without exception)
  5. missing-obs (state with no observation field handled gracefully)
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.templates.agentica_lite import ArcgenticaLite  # noqa: E402
from agents.templates.agentica_lite.proposer import (  # noqa: E402
    ProposerOutput, ProposerResult,
)


def _state(turn: int, with_obs: bool = True) -> dict:
    state: dict = {
        "turn": turn,
        "visible_regions": [
            {"id": f"R{i}", "region_id": f"R{i}",
             "bbox": [i * 5, 0, i * 5 + 4, 4], "color": i,
             "is_multicolor": False, "y_band": "play_zone",
             "is_marker_neighbor": False, "is_primary_marker": False}
            for i in range(3)
        ],
        "marker_neighbor_states": [
            {"marker_id": "M0", "is_primary_marker": True, "compass": {
                d: {"region_id": f"R{i}", "current_color": None,
                    "clicks": 1 if i < turn else 0}
                for i, d in enumerate(("N", "E", "S", "W"))
            }},
        ],
    }
    if with_obs:
        state["observation"] = {
            "primary_region_id": "R0",
            "dominant_transition": {"from": 0, "to": 1, "count": 12 + turn * 4},
            "level_delta": 1 if turn == 0 else 0,
        }
    return state


# ---------- 1. proposer-fired -------------------------------------------------

def test_proposer_fires_on_warm_up_turn(tmp_path, monkeypatch):
    """Warm-up condition (turn 0) fires Proposer; counter increments by 1."""
    monkeypatch.chdir(tmp_path)
    agent = ArcgenticaLite(game_id="test-fired", seed=1)

    # Replace proposer.propose with a mock that returns a successful result so
    # the call is observable without hitting TRAPI.
    async def _mock_propose(state, ids):
        return ProposerResult(output=ProposerOutput(
            candidate_predicate_id="P_saturation_progress",
            region_hint=ids[0] if ids else "R0",
            expected_signature={"level_delta": 1},
            required_pre_state={"marker_id": "M0", "saturation_threshold": 0,
                                "saturation_denominator": 4},
            confidence=0.8,
            thought="",
        ), failure_reason=None)

    agent.proposer.propose = _mock_propose  # type: ignore[assignment]
    asyncio.run(agent.run_turn(_state(0)))
    assert agent._proposer_calls == 1
    assert agent.stalemate.warm_up_done is True


# ---------- 2. proposer-skipped ----------------------------------------------

def test_proposer_skipped_when_warm_up_done_and_no_stalemate(tmp_path, monkeypatch):
    """After warm-up + no stalemate, Proposer is not invoked on subsequent turns."""
    monkeypatch.chdir(tmp_path)
    agent = ArcgenticaLite(game_id="test-skipped", seed=2)

    call_count = {"n": 0}

    async def _mock_propose(state, ids):
        call_count["n"] += 1
        return ProposerResult(output=None, failure_reason="llm_no_client")

    agent.proposer.propose = _mock_propose  # type: ignore[assignment]
    # Turn 0 should trigger warm-up
    asyncio.run(agent.run_turn(_state(0)))
    after_warmup = call_count["n"]
    # Turn 1 — no warm-up needed, posterior still cold so stalemate threshold not yet crossed
    asyncio.run(agent.run_turn(_state(1)))
    assert call_count["n"] == after_warmup, "Proposer should not be re-invoked at turn 1"


# ---------- 3. episode-end ----------------------------------------------------

def test_episode_end_returns_record_with_extra_counters(tmp_path, monkeypatch):
    """episode_end() builds EpisodeRecord with v601 version + extra counters set."""
    monkeypatch.chdir(tmp_path)
    agent = ArcgenticaLite(game_id="test-end", seed=3)

    async def _block_proposer(state, ids):
        return ProposerResult(output=None, failure_reason="llm_no_client")

    agent.proposer.propose = _block_proposer  # type: ignore[assignment]
    asyncio.run(agent.run_turn(_state(0)))
    asyncio.run(agent.run_turn(_state(1)))
    rec = agent.episode_end()
    assert rec.framework_version == "v601"
    assert rec.turns == 2
    assert "proposer_calls" in rec.extra
    assert "reflector_calls" in rec.extra
    assert "confidence_overrides" in rec.extra
    # Latency counters
    assert rec.latency_p50 >= 0.0


# ---------- 4. budget-breach --------------------------------------------------

def test_budget_breach_does_not_raise(tmp_path, monkeypatch, caplog):
    """A turn that takes longer than global_turn_budget_s logs a warning but
    does not raise."""
    monkeypatch.chdir(tmp_path)
    # Set absurdly tight budget to provoke the breach branch
    agent = ArcgenticaLite(game_id="test-budget", seed=4, global_turn_budget_s=0.0)

    async def _block_proposer(state, ids):
        return ProposerResult(output=None, failure_reason="llm_no_client")

    agent.proposer.propose = _block_proposer  # type: ignore[assignment]
    # Should complete without raising even though budget is 0.
    res = asyncio.run(agent.run_turn(_state(0)))
    # res may be Action or None depending on policy decision; either is fine.
    assert res is None or hasattr(res, "predicate_id")
    # Latency record present
    assert len(agent._turn_latencies) == 1


# ---------- 5. missing-obs ----------------------------------------------------

def test_missing_observation_handled_gracefully(tmp_path, monkeypatch):
    """State without an 'observation' field is processed without crashing."""
    monkeypatch.chdir(tmp_path)
    agent = ArcgenticaLite(game_id="test-missing-obs", seed=5)

    async def _block_proposer(state, ids):
        return ProposerResult(output=None, failure_reason="llm_no_client")

    agent.proposer.propose = _block_proposer  # type: ignore[assignment]

    # Run with state missing "observation" key entirely
    result = asyncio.run(agent.run_turn(_state(0, with_obs=False)))
    # Should not raise; result may be None (no obs -> verdict path skipped)
    rec = agent.episode_end()
    assert rec.turns == 1
