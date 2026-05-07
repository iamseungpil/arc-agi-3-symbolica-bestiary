"""B9 cross-run memory wiring tests (no LLM calls).

Verifies that:
  T-B9-1: V57Board loads/saves cross_run_memory.json correctly.
  T-B9-2: add_to_1A appends new mechanic; repeated text increments
          confirmed_runs.
  T-B9-3: run_turn passes cross_run_priors to spawn_hypothesize.
  T-B9-4: run_turn calls board.add_to_1A when M4 emits promote_to_1A.
"""
import asyncio
import json
import pathlib
import tempfile

import pytest

from agents.templates.agentica_v57.agent import V57Board, run_turn


def test_b9_1_load_save_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        wd = pathlib.Path(tmp) / "ns1"
        b = V57Board(namespace="ns1", game_id="ft09", workdir=wd)
        assert b.cross_run_memory["schema_version"] == 1
        assert b.cross_run_memory["abstract_mechanics"] == []

        b.add_to_1A("Sample mechanism observation.", turn=3)
        assert len(b.cross_run_memory["abstract_mechanics"]) == 1
        assert b.cross_run_memory["abstract_mechanics"][0]["confirmed_runs"] == 1

        # Reload — file persists.
        b2 = V57Board(namespace="ns2", game_id="ft09", workdir=wd)
        assert len(b2.cross_run_memory["abstract_mechanics"]) == 1


def test_b9_2_add_to_1A_dedup_increments():
    with tempfile.TemporaryDirectory() as tmp:
        wd = pathlib.Path(tmp) / "ns1"
        b = V57Board(namespace="ns1", game_id="ft09", workdir=wd)
        b.add_to_1A("Same text.", turn=1)
        b.add_to_1A("  same TEXT.  ", turn=2)  # case + whitespace
        assert len(b.cross_run_memory["abstract_mechanics"]) == 1
        assert b.cross_run_memory["abstract_mechanics"][0]["confirmed_runs"] == 2


def _make_run_turn_inputs(tmp, prior_text=None):
    wd = pathlib.Path(tmp) / "ns1"
    b = V57Board(namespace="ns1", game_id="ft09", workdir=wd)
    if prior_text:
        b.add_to_1A(prior_text, turn=0)
        # New board to load it
        b = V57Board(namespace="ns2", game_id="ft09", workdir=wd)
    return b


def test_b9_3_priors_passed_to_hypothesize():
    received = {}

    async def fake_hypo(*, summary, visible_regions, falsified_recent,
                       gqb_pair, image_b64, next_card_id_seed,
                       cross_run_priors=None, **kw):
        received["priors"] = cross_run_priors
        return {"thought": "", "cards": []}

    async def fake_action(*a, **k):
        return {"thought": "", "chosen_card_id": None, "click": {"x": 30, "y": 30}}

    async def fake_refl(*a, **k):
        return {"summary": "ok"}

    async def fake_exec(cx, cy):
        return {"level_delta": 0, "primary_region_id": "R3",
                "dominant_transition": {"from": 9, "to": 12, "count": 36}}

    async def main():
        with tempfile.TemporaryDirectory() as tmp:
            board = _make_run_turn_inputs(tmp, prior_text="Prior atomic obs.")
            await run_turn(
                board=board, visible_regions=[
                    {"id": "R3", "bbox": {"min_x": 10, "min_y": 10, "max_x": 15, "max_y": 15}, "color": 9,
                     "is_multicolor": False, "size": 36, "y_band": "play_zone",
                     "click_response": {"clicks": 0, "responses": 0},
                     "is_marker_neighbor": False},
                ], image_b64=None,
                spawn_action=fake_action, spawn_hypothesize=fake_hypo,
                spawn_reflexion=fake_refl, execute_action=fake_exec,
            )
        assert "priors" in received
        assert received["priors"] is not None
        assert len(received["priors"]) == 1
        assert "Prior atomic obs." == received["priors"][0]["text"]

    asyncio.run(main())


def test_b9_4_promote_to_1A_persisted():
    captured_promotions = []

    async def fake_hypo(*a, **k): return {"thought": "", "cards": []}
    async def fake_action(*a, **k):
        return {"thought": "", "chosen_card_id": None, "click": {"x": 30, "y": 30}}

    async def fake_refl(*a, **k):
        return {
            "summary": "5-turn window summary",
            "promote_to_1A": ["A click on a region returning prior color is the reverse of the prior click."],
        }

    async def fake_exec(cx, cy):
        # Force level_delta=1 to trigger reflexion
        return {"level_delta": 1, "primary_region_id": "R3",
                "dominant_transition": {"from": 5, "to": 4, "count": 1824}}

    async def main():
        with tempfile.TemporaryDirectory() as tmp:
            wd = pathlib.Path(tmp) / "ns1"
            board = V57Board(namespace="ns1", game_id="ft09", workdir=wd)
            await run_turn(
                board=board, visible_regions=[
                    {"id": "R3", "bbox": {"min_x": 10, "min_y": 10, "max_x": 15, "max_y": 15}, "color": 9,
                     "is_multicolor": False, "size": 36, "y_band": "play_zone",
                     "click_response": {"clicks": 0, "responses": 0},
                     "is_marker_neighbor": False},
                ], image_b64=None,
                spawn_action=fake_action, spawn_hypothesize=fake_hypo,
                spawn_reflexion=fake_refl, execute_action=fake_exec,
            )
            assert len(board.cross_run_memory["abstract_mechanics"]) == 1
            assert "reverse of the prior click" in board.cross_run_memory["abstract_mechanics"][0]["text"]

    asyncio.run(main())
