"""v587 B14 — hierarchical memory tests (ESC + ASMW + CPSR).

ESC = Episodic Signature Clustering
ASMW = Adaptive Stuck-Mode Window
CPSR = Counterfactual Past-Segment Retrieval

Tests cover the deterministic data layer only (no LLM).
"""
import asyncio
import json
import pathlib
import re
import tempfile
from unittest.mock import patch

import pytest

from agents.templates.agentica_v57.agent import (
    V57Board,
    _approx_current_metrics,
    _retrieve_analogous_segments,
    call_action,
)


# -------------------------- ESC -------------------------------------


def test_esc_1_signature_dedup_increments_confirmed_runs():
    """Two L+ events with the same structural signature collapse into
    one cluster with confirmed_runs=2 and cluster_observations of size 2."""
    with tempfile.TemporaryDirectory() as tmp:
        wd = pathlib.Path(tmp) / "ns1"
        b = V57Board(namespace="ns1", game_id="ft09", workdir=wd)
        sig = "cc_5to6|rc_1|tr_flat|kd_all_non_marker"
        b.add_to_1A("First observation about clicks", turn=10, signature=sig)
        b.add_to_1A("Second observation about clicks", turn=22, signature=sig)
        am = b.cross_run_memory["abstract_mechanics"]
        assert len(am) == 1
        assert am[0]["confirmed_runs"] == 2
        assert am[0]["signature"] == sig
        obs = am[0]["cluster_observations"]
        assert len(obs) == 2
        assert "First" in obs[0]
        assert "Second" in obs[1]


def test_esc_2_different_signatures_stay_separate():
    with tempfile.TemporaryDirectory() as tmp:
        wd = pathlib.Path(tmp) / "ns1"
        b = V57Board(namespace="ns1", game_id="ft09", workdir=wd)
        b.add_to_1A("A", turn=1, signature="sig_A")
        b.add_to_1A("B", turn=2, signature="sig_B")
        am = b.cross_run_memory["abstract_mechanics"]
        assert len(am) == 2
        assert {e["signature"] for e in am} == {"sig_A", "sig_B"}


def test_esc_3_legacy_text_dedup_still_works_when_signature_is_None():
    with tempfile.TemporaryDirectory() as tmp:
        wd = pathlib.Path(tmp) / "ns1"
        b = V57Board(namespace="ns1", game_id="ft09", workdir=wd)
        b.add_to_1A("legacy text", turn=1)  # signature=None
        b.add_to_1A("legacy text", turn=5)  # repeat
        am = b.cross_run_memory["abstract_mechanics"]
        assert len(am) == 1
        assert am[0]["confirmed_runs"] == 2
        assert am[0].get("signature") is None


def test_esc_4_legacy_entry_migrates_to_cluster_when_signature_arrives():
    """A legacy text-only entry, when re-observed via signature path,
    is migrated INTO a new cluster (carrying its history)."""
    with tempfile.TemporaryDirectory() as tmp:
        wd = pathlib.Path(tmp) / "ns1"
        b = V57Board(namespace="ns1", game_id="ft09", workdir=wd)
        # Step 1: legacy text-only deposit.
        b.add_to_1A("text X", turn=5)
        am = b.cross_run_memory["abstract_mechanics"]
        assert len(am) == 1
        assert am[0].get("signature") is None
        # Step 2: same text now has signature → migrates.
        b.add_to_1A("text X", turn=10, signature="sig_X")
        am2 = b.cross_run_memory["abstract_mechanics"]
        assert len(am2) == 1  # still one entry, MIGRATED not duplicated
        assert am2[0]["signature"] == "sig_X"
        assert am2[0]["confirmed_runs"] == 2  # 1 legacy + 1 new
        assert "text X" in am2[0]["cluster_observations"]


def test_esc_5_cluster_obs_capped_but_cfr_unbounded():
    """observation list capped at _CLUSTER_OBS_CAP; counter is not."""
    with tempfile.TemporaryDirectory() as tmp:
        wd = pathlib.Path(tmp) / "ns1"
        b = V57Board(namespace="ns1", game_id="ft09", workdir=wd)
        sig = "sig_constant"
        for i in range(10):
            b.add_to_1A(f"observation_{i}", turn=i, signature=sig)
        am = b.cross_run_memory["abstract_mechanics"]
        assert len(am) == 1
        assert am[0]["confirmed_runs"] == 10
        assert len(am[0]["cluster_observations"]) <= 5


# -------------------------- ASMW ------------------------------------


def test_asmw_1_stuck_severity_zero_when_lp_recent():
    with tempfile.TemporaryDirectory() as tmp:
        wd = pathlib.Path(tmp) / "ns1"
        b = V57Board(namespace="ns1", game_id="ft09", workdir=wd)
        b.last_lp_event_turn = 20
        b.turn_index = 30  # gap = 10 < K_STUCK=15
        assert b.stuck_severity() == 0


def test_asmw_2_stuck_severity_grows_with_gap():
    with tempfile.TemporaryDirectory() as tmp:
        wd = pathlib.Path(tmp) / "ns1"
        b = V57Board(namespace="ns1", game_id="ft09", workdir=wd)
        b.last_lp_event_turn = 0
        b.turn_index = 25  # gap=25 - K_STUCK=15 → severity=10
        assert b.stuck_severity() == 10


def test_asmw_3_hysteresis_stuck_on_off():
    """ON when gap >= K_STUCK; OFF only after L+ event AND gap drops
    to ≤ STUCK_HYSTERESIS_OFF=7."""
    with tempfile.TemporaryDirectory() as tmp:
        wd = pathlib.Path(tmp) / "ns1"
        b = V57Board(namespace="ns1", game_id="ft09", workdir=wd)
        # gap=15, no L+ → ON.
        b.last_lp_event_turn = 0
        b.turn_index = 15
        b.update_stuck_mode(level_delta=0)
        assert b.stuck_mode is True
        # gap=20, no L+ → still ON.
        b.turn_index = 20
        b.update_stuck_mode(level_delta=0)
        assert b.stuck_mode is True
        # gap=20 (after gap=20), L+ event fires → gap > 7 still, OFF gate
        # only triggers when gap ≤ 7 after L+. We measure gap *before*
        # last_lp_event_turn update, so gap=20 here is too large to flip
        # off yet → stays ON.
        b.turn_index = 21
        b.update_stuck_mode(level_delta=1)  # but gap was 21, so OFF gate fails
        assert b.stuck_mode is True
        # Now last_lp_event_turn updated to 21. gap from new turn 25 is 4.
        b.turn_index = 25
        b.update_stuck_mode(level_delta=1)  # gap=4, L+ → OFF
        assert b.stuck_mode is False


def test_asmw_4_serialised_expand_widens_visible_window():
    with tempfile.TemporaryDirectory() as tmp:
        wd = pathlib.Path(tmp) / "ns1"
        b = V57Board(namespace="ns1", game_id="ft09", workdir=wd)
        for t in range(20):
            b.append_turn_diff({"turn": t})
        assert len(b.serialised_recent_turn_diffs(expand=0)) == 6
        assert len(b.serialised_recent_turn_diffs(expand=1)) == 8
        assert len(b.serialised_recent_turn_diffs(expand=3)) == 12


# -------------------------- CPSR ------------------------------------


def test_cpsr_1_empty_index_returns_empty():
    out = _retrieve_analogous_segments(
        current_signature="any|sig", segment_index=[], k_per_class=1
    )
    assert out == []


def test_cpsr_2_returns_one_success_one_failure():
    idx = [
        {"kind": "post_L+_recovery", "did_progress": True,
         "abstract_signature": "cc_5to6|rc_0|tr_flat|kd_all_non_marker"},
        {"kind": "post_L+_stuck", "did_progress": False,
         "abstract_signature": "cc_5to6|rc_0|tr_flat|kd_all_non_marker"},
        {"kind": "post_L+_stuck", "did_progress": False,
         "abstract_signature": "cc_le4|rc_2plus|tr_active|kd_mixed"},
    ]
    out = _retrieve_analogous_segments(
        current_signature="cc_5to6|rc_0|tr_flat|kd_all_non_marker",
        segment_index=idx, k_per_class=1,
    )
    assert len(out) == 2
    assert any(s["did_progress"] is True for s in out)
    assert any(s["did_progress"] is False for s in out)
    # The match_score for the perfect-match success should be 4 (all
    # 4 tokens equal).
    success = next(s for s in out if s["did_progress"])
    assert success["match_score"] == 4


def test_cpsr_3_anti_leak_no_R_id_in_payload():
    """Segment payload must contain no R-id pattern (R\\d+) and no
    coord-pair pattern. Static check on the retrieve function output."""
    idx = [
        {"kind": "pre_L+_5turns", "did_progress": True,
         "abstract_signature": "cc_5to6|rc_0|tr_flat|kd_all_non_marker",
         "what_changed_in_next_5turns": "marker_progress +2; kind shifted"},
    ]
    out = _retrieve_analogous_segments(
        current_signature="cc_5to6|rc_0|tr_flat|kd_all_non_marker",
        segment_index=idx, k_per_class=1,
    )
    serialised = json.dumps(out)
    assert not re.search(r"\bR\d+\b", serialised)
    assert not re.search(r"\b\d{1,2}\s*,\s*\d{1,2}\b", serialised)


def test_cpsr_4_approx_current_metrics_from_diffs():
    diffs = [
        {"click_region_id": "R5", "region_kind_pre": "non_marker",
         "compass_changes": [{"x": 1}]},
        {"click_region_id": "R5", "region_kind_pre": "non_marker",
         "compass_changes": []},
        {"click_region_id": "R6", "region_kind_pre": "non_marker",
         "compass_changes": []},
    ]
    out = _approx_current_metrics(diffs)
    assert out["click_count"] == 3
    assert out["repeat_clicks"] == 1     # R5 repeated
    assert out["kind_distribution"] == {"non_marker": 3}
    assert out["compass_change_traj"] == [1, 0, 0]


# -------------------------- Integration ------------------------------


def test_int_call_action_payload_includes_b14_keys():
    captured = {}

    class _StubAgent:
        async def call(self, _t, task, **kwargs):
            captured["task"] = task
            return '{"thought":"x","chosen_hypothesis_id":null,"action":{"type":"ACTION6","coord":[32,32]}}'

    async def _fake_spawn(**kwargs):
        return _StubAgent()

    async def _run():
        with patch("agents.templates.agentica_v57.agent.spawn", new=_fake_spawn):
            await call_action(
                summary="",
                visible_regions=[],
                active_hypotheses=[],
                recent_turns=[],
                gqb_pair=None,
                image_b64=None,
                stuck_mode=True,
                analogous_past_segments=[
                    {"kind": "post_L+_recovery", "did_progress": True,
                     "abstract_signature": "cc_5to6|rc_0|tr_flat|kd_all_non_marker"},
                    {"kind": "post_L+_stuck", "did_progress": False,
                     "abstract_signature": "cc_5to6|rc_0|tr_flat|kd_all_non_marker"},
                ],
            )

    asyncio.run(_run())
    assert "stuck_mode" in captured["task"]
    assert "analogous_past_segments" in captured["task"]
    assert "post_L+_recovery" in captured["task"]
