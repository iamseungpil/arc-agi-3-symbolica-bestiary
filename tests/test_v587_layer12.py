"""v587 Layer 1 (B12 turn-diff buffer) + Layer 2 (B13 level_bridges) tests.

Layer 1 — recent_turn_diffs (Symbolica per-turn structured diff):
  T-L1-1: compass diff between two snapshots — only changed cells.
  T-L1-2: append_turn_diff trims to max 6 entries.
  T-L1-3: fill_pending_compass_changes fills last entry only when None.
  T-L1-4: serialised_recent_turn_diffs strips _pre_compass_snapshot.
  T-L1-5: _compass_snapshot_from_marker_states returns
          {marker_id: {direction: color}} structure.

Layer 2 — level_bridges (Hermes episodic):
  T-L2-1: load/save roundtrip on level_bridges.json.
  T-L2-2: append_level_bridge race-safe (reload-merge-save).
  T-L2-3: _compose_level_bridge_entry distils metrics correctly.
  T-L2-4: _build_level_bridge_priors strips run-specific fields and
          dedups by abstract_signature.
  T-L2-5: M1 prompt input payload contains both RECENT_TURN_DIFFS
          and LEVEL_BRIDGE_PRIORS keys.
"""
import asyncio
import json
import pathlib
import tempfile
from unittest.mock import AsyncMock, patch

import pytest

from agents.templates.agentica_v57.agent import (
    V57Board,
    _build_level_bridge_priors,
    _compose_level_bridge_entry,
    call_action,
)


# -------------------------- Layer 1 (B12) ----------------------------


def test_l1_1_compass_diff_returns_only_changed():
    prev = {
        "R12": {"N": 9, "E": 9, "S": 12, "W": 9},
        "R20": {"N": 2, "E": 2},
    }
    cur = {
        "R12": {"N": 9, "E": 2, "S": 12, "W": 9},   # E changed
        "R20": {"N": 2, "E": 0},                     # E changed
    }
    out = V57Board._diff_compass_snapshots(prev, cur)
    sigs = {(c["marker_id"], c["direction"]) for c in out}
    assert ("R12", "E") in sigs
    assert ("R20", "E") in sigs
    assert len(out) == 2  # nothing else


def test_l1_2_append_turn_diff_trims_to_max_raw():
    """v587 B14: raw cap raised from 6 to 24 so ASMW can expand window
    when stuck. Default serialised exposure (expand=0) still returns 6."""
    with tempfile.TemporaryDirectory() as tmp:
        wd = pathlib.Path(tmp) / "ns1"
        b = V57Board(namespace="ns1", game_id="ft09", workdir=wd)
        for t in range(30):
            b.append_turn_diff({"turn": t, "click": [0, 0]})
        # Raw buffer holds up to 24.
        assert len(b.recent_turn_diffs) == 24
        # Oldest 6 dropped (0..5), latest kept (29).
        assert b.recent_turn_diffs[0]["turn"] == 6
        assert b.recent_turn_diffs[-1]["turn"] == 29
        # Default serialised (expand=0) returns last 6 entries.
        ser_default = b.serialised_recent_turn_diffs(expand=0)
        assert len(ser_default) == 6
        assert ser_default[-1]["turn"] == 29
        # Expanded by 2 → 6 + 2*2 = 10 entries.
        ser_expand2 = b.serialised_recent_turn_diffs(expand=2)
        assert len(ser_expand2) == 10
        # Expand beyond cap is clamped to 24.
        ser_expand_huge = b.serialised_recent_turn_diffs(expand=100)
        assert len(ser_expand_huge) == 24


def test_l1_3_fill_pending_compass_changes():
    with tempfile.TemporaryDirectory() as tmp:
        wd = pathlib.Path(tmp) / "ns1"
        b = V57Board(namespace="ns1", game_id="ft09", workdir=wd)
        # Append a diff still awaiting post-click compass.
        b.append_turn_diff({
            "turn": 1,
            "click": [10, 10],
            "compass_changes": None,
            "_pre_compass_snapshot": {"R12": {"N": 9}},
        })
        # Simulate next turn's compass.
        b.fill_pending_compass_changes({"R12": {"N": 2}})
        assert b.recent_turn_diffs[0]["compass_changes"] == [
            {"marker_id": "R12", "direction": "N", "from": 9, "to": 2}
        ]
        # _pre_compass_snapshot should have been popped.
        assert "_pre_compass_snapshot" not in b.recent_turn_diffs[0]
        # Idempotent — calling again does not double-fill.
        b.fill_pending_compass_changes({"R12": {"N": 12}})
        assert len(b.recent_turn_diffs[0]["compass_changes"]) == 1


def test_l1_4_serialised_strips_internal_keys():
    with tempfile.TemporaryDirectory() as tmp:
        wd = pathlib.Path(tmp) / "ns1"
        b = V57Board(namespace="ns1", game_id="ft09", workdir=wd)
        b.append_turn_diff({
            "turn": 5,
            "click": [3, 4],
            "compass_changes": [],
            "_pre_compass_snapshot": {"R1": {"N": 9}},
        })
        out = b.serialised_recent_turn_diffs()
        assert all("_pre_compass_snapshot" not in e for e in out)
        # Public keys preserved.
        assert out[0]["turn"] == 5
        assert out[0]["click"] == [3, 4]


def test_l1_5_compass_snapshot_from_marker_states():
    states = [
        {
            "marker_id": "R12",
            "compass": {
                "N": {"region_id": "R30", "current_color": 9, "clicks": 0},
                "E": {"region_id": "R31", "current_color": 12, "clicks": 1},
            },
        }
    ]
    snap = V57Board._compass_snapshot_from_marker_states(states)
    assert snap == {"R12": {"N": 9, "E": 12}}


# -------------------------- Layer 2 (B13) ----------------------------


def test_l2_1_level_bridges_load_save_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        wd = pathlib.Path(tmp) / "ns1"
        b = V57Board(namespace="ns1", game_id="ft09", workdir=wd)
        assert b.level_bridges["schema_version"] == 1
        assert b.level_bridges["bridges"] == []
        b.append_level_bridge({
            "from_level_delta": 1,
            "click_count": 11,
            "abstract_signature": "11 clicks summary",
        })
        # Reload sees it.
        b2 = V57Board(namespace="ns2", game_id="ft09", workdir=wd)
        assert len(b2.level_bridges["bridges"]) == 1
        assert b2.level_bridges["bridges"][0]["id"] == "B1"


def test_l2_2_append_level_bridge_race_safe():
    with tempfile.TemporaryDirectory() as tmp:
        wd = pathlib.Path(tmp) / "ns1"
        b1 = V57Board(namespace="r1", game_id="ft09", workdir=wd)
        b2 = V57Board(namespace="r2", game_id="ft09", workdir=wd)
        # Both boards started with empty level_bridges. b1 writes one.
        b1.append_level_bridge({"abstract_signature": "shape A"})
        # b2 in-memory has empty bridges, but its append should reload
        # b1's contribution before saving.
        b2.append_level_bridge({"abstract_signature": "shape B"})
        # Final on-disk should have both, in append order.
        path = wd.parent / "level_bridges.json"
        data = json.loads(path.read_text())
        sigs = [b["abstract_signature"] for b in data["bridges"]]
        assert sigs == ["shape A", "shape B"]


def test_l2_3_compose_level_bridge_entry_metrics():
    with tempfile.TemporaryDirectory() as tmp:
        wd = pathlib.Path(tmp) / "ns1"
        b = V57Board(namespace="ns1", game_id="ft09", workdir=wd)
        b.turn_index = 11
        diffs = [
            {"click_region_id": "R5", "region_kind_pre": "non_marker",
             "compass_changes": [{"marker_id": "R12", "direction": "N", "from": 9, "to": 2}]},
            {"click_region_id": "R6", "region_kind_pre": "non_marker",
             "compass_changes": []},
            {"click_region_id": "R5", "region_kind_pre": "non_marker",
             "compass_changes": [{"marker_id": "R12", "direction": "E", "from": 9, "to": 2}]},
            {"click_region_id": "R12", "region_kind_pre": "marker_multicolor",
             "compass_changes": []},
        ]
        entry = _compose_level_bridge_entry(
            board=b, from_level_delta=1, recent_diffs_serialised=diffs
        )
        assert entry["click_count"] == 4
        assert entry["unique_regions_clicked"] == 3   # R5,R6,R12 (R5 repeated)
        assert entry["repeat_clicks"] == 1
        assert entry["kind_distribution"]["non_marker"] == 3
        assert entry["kind_distribution"]["marker_multicolor"] == 1
        assert entry["compass_change_traj"] == [1, 0, 1, 0]
        assert "abstract_signature" in entry
        assert entry["run_namespace"] == "ns1"


def test_l2_4_build_priors_strips_runspec_and_dedups():
    bridges = [
        {"id": "B1", "run_namespace": "r1", "abstract_signature": "shape A",
         "click_count": 11, "from_level_delta": 1},
        {"id": "B2", "run_namespace": "r2", "abstract_signature": "shape A",
         "click_count": 12, "from_level_delta": 1},  # dup signature
        {"id": "B3", "run_namespace": "r3", "abstract_signature": "shape B",
         "click_count": 9, "from_level_delta": 1},
    ]
    priors = _build_level_bridge_priors(bridges)
    sigs = [p["abstract_signature"] for p in priors]
    # Dedup keeps one per signature; reversed order = most recent first.
    assert sigs == ["shape B", "shape A"]
    assert all("run_namespace" not in p for p in priors)
    assert all("id" not in p for p in priors)


def test_l2_5_call_action_payload_contains_layer12_keys():
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
                recent_turn_diffs=[{"turn": 5, "click": [10, 10]}],
                level_bridge_priors=[
                    {"abstract_signature": "shape demo",
                     "click_count": 11,
                     "compass_change_traj": [1, 1, 0, 1]}
                ],
            )

    asyncio.run(_run())
    assert "recent_turn_diffs" in captured["task"]
    assert "level_bridge_priors" in captured["task"]
    assert "shape demo" in captured["task"]
