"""v586 splice fixture tests.

Two tiers:
  - **Offline** (always run, fast): verify fixture files exist and have
    well-formed shape; verify scoring helpers behave correctly on
    synthetic inputs.
  - **Live LLM** (opt-in via `RUN_V586_LIVE=1`): N=10 trials per splice
    against the real spawn_action/spawn_hypothesize/spawn_reflexion via
    TRAPI. Skipped by default — ~80 LLM calls per full V2 pass.

Scoring functions are exposed for `scripts/run_splice_v586.py` to import.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
FIXTURE_DIR = REPO / "tests" / "fixtures" / "v586_splice"


# ---------------------------------------------------------------------------
# Scoring helpers (also importable by the live-runner script).
# ---------------------------------------------------------------------------


def score_m1_coord_in_targets(coord: list[int],
                              targets: list[dict]) -> bool:
    """H_M1.A primary criterion: coord lies inside one of `targets[].bbox`.

    bbox format: [min_x, min_y, max_x, max_y] inclusive.
    """
    if not coord or len(coord) != 2:
        return False
    cx, cy = coord
    for t in targets:
        bb = t.get("bbox") or []
        if len(bb) != 4:
            continue
        if bb[0] <= cx <= bb[2] and bb[1] <= cy <= bb[3]:
            return True
    return False


def score_m1_manhattan_from_anchor(coord: list[int], anchor: list[int],
                                   min_dist: int) -> bool:
    """H_M1.A counterfactual: coord ≥ min_dist Manhattan from anchor."""
    if not coord or len(coord) != 2 or not anchor or len(anchor) != 2:
        return False
    return abs(coord[0] - anchor[0]) + abs(coord[1] - anchor[1]) >= min_dist


def _bbox_xyxy(bb) -> tuple[int, int, int, int] | None:
    """Normalize bbox in list[4] OR dict{min_x,...} form to tuple."""
    if isinstance(bb, dict):
        try:
            return (int(bb["min_x"]), int(bb["min_y"]),
                    int(bb["max_x"]), int(bb["max_y"]))
        except (KeyError, TypeError):
            return None
    if isinstance(bb, (list, tuple)) and len(bb) == 4:
        return tuple(int(x) for x in bb)
    return None


def score_m1_in_any_visible_bbox_no_repeat(coord: list[int],
                                           visible_regions: list[dict],
                                           recent_coords: list[list[int]]
                                           ) -> bool:
    """Generic well-formedness gate."""
    if not coord or len(coord) != 2:
        return False
    cx, cy = coord
    in_any = False
    for vr in visible_regions:
        bb = _bbox_xyxy(vr.get("bbox"))
        if bb and bb[0] <= cx <= bb[2] and bb[1] <= cy <= bb[3]:
            in_any = True
            break
    if not in_any:
        return False
    for rc in recent_coords[-3:]:
        if rc and len(rc) == 2 and rc[0] == cx and rc[1] == cy:
            return False
    return True


def score_m3_keyword_in_card_text(cards: list[dict],
                                  keyword_set: list[str],
                                  min_matches: int = 1) -> bool:
    """H_M3.A: at least `min_matches` cards contain ≥1 keyword (in
    rule_hypothesis OR predicate, case-insensitive)."""
    if not cards:
        return False
    matched = 0
    for c in cards:
        text = " ".join([
            str(c.get("rule_hypothesis") or ""),
            str(c.get("predicate") or ""),
            str(c.get("abstract_claim") or ""),
        ]).lower()
        if any(kw.lower() in text for kw in keyword_set):
            matched += 1
            if matched >= min_matches:
                return True
    return False


def score_reflexion_summary(summary: str, expected: dict) -> bool:
    """H_R.A: contains magnitude token, no [CAUSE prefix, ≤max_length."""
    if not summary or not isinstance(summary, str):
        return False
    if expected.get("must_not_start_with"):
        prefix = expected["must_not_start_with"]
        if summary.lstrip().startswith(prefix):
            return False
    max_len = expected.get("max_length")
    if max_len is not None and len(summary) > max_len:
        return False
    must_any = expected.get("must_contain_any") or []
    if must_any:
        low = summary.lower()
        if not any(tok.lower() in low for tok in must_any):
            return False
    return True


def score_fixture(fx: dict, module_output: dict) -> bool:
    """Generic dispatcher: fixture+output → pass/fail."""
    expected = fx["expected"]
    crit = expected["criterion"]
    mod = fx["module"]
    if mod == "M1":
        coord = (module_output.get("action") or {}).get("coord")
        if crit == "coord_in_one_of_target_bboxes":
            return score_m1_coord_in_targets(coord, expected["targets"])
        if crit == "coord_min_manhattan_from_prior_trigger":
            return score_m1_manhattan_from_anchor(
                coord, expected["anchor"], expected["min_manhattan"])
        if crit == "coord_in_any_visible_bbox_and_not_recent_repeat":
            recent_coords = [t.get("coord") for t
                             in fx["input"].get("recent_turns") or []]
            return score_m1_in_any_visible_bbox_no_repeat(
                coord, fx["input"].get("visible_regions") or [],
                recent_coords)
        return False
    if mod == "M3":
        cards = module_output.get("cards") or []
        return score_m3_keyword_in_card_text(
            cards, expected["keyword_set"],
            expected.get("min_matches", 1))
    if mod == "R":
        summary = module_output.get("summary") or ""
        return score_reflexion_summary(summary, expected)
    return False


# ---------------------------------------------------------------------------
# Offline tests — always run.
# ---------------------------------------------------------------------------


def _load_fixtures(module: str) -> list[dict]:
    p = FIXTURE_DIR / f"{module}.jsonl"
    if not p.exists():
        return []
    with p.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def test_fixture_files_exist():
    assert (FIXTURE_DIR / "M1.jsonl").exists(), (
        "Run scripts/extract_splice_v586.py first")
    assert (FIXTURE_DIR / "M3.jsonl").exists()
    assert (FIXTURE_DIR / "R.jsonl").exists()


def test_m1_fixtures_well_formed():
    fixtures = _load_fixtures("M1")
    assert len(fixtures) >= 6, f"got {len(fixtures)} M1 fixtures"
    train = [f for f in fixtures if f.get("split") == "train"]
    val = [f for f in fixtures if f.get("split") == "val"]
    assert len(train) >= 5
    assert len(val) >= 4
    # Required fields per fixture.
    for fx in fixtures:
        assert fx["module"] == "M1"
        assert "input" in fx and "expected" in fx
        assert fx["input"].get("visible_regions"), (
            f"{fx['id']}: no visible_regions reconstructed")
        assert fx["expected"].get("criterion")


def test_m1_train_targets_sound_for_cycle126_t49():
    """The H_M1.A pivot fixture: targets must contain R3 and R4 with
    bboxes that actually cover the click coord cycle126 made."""
    fixtures = _load_fixtures("M1")
    fx = next((f for f in fixtures
               if f["cycle"] == "cycle126" and f["turn"] == 49), None)
    assert fx is not None, "cycle126 t49 fixture missing"
    targets = fx["expected"]["targets"]
    rids = {t["region_id"] for t in targets}
    assert "R3" in rids and "R4" in rids
    # Actual click was at [30, 16], primary=R3 → must be in R3 bbox.
    actual = fx["actual_outcome"]["coord"]
    assert score_m1_coord_in_targets(actual, targets), (
        f"actual click {actual} should be in one of {targets}")


def test_m3_fixture_at_cycle126_t10_present():
    fixtures = _load_fixtures("M3")
    fx = next((f for f in fixtures
               if f["cycle"] == "cycle126" and f["turn"] == 10), None)
    assert fx is not None, "cycle126 t10 M3 splice missing"
    assert fx["expected"]["criterion"] == (
        "at_least_one_card_text_matches_keyword_set")
    assert "marker crop" in fx["expected"]["keyword_set"]


def test_reflexion_fixture_cycle126_t27_present():
    fixtures = _load_fixtures("R")
    fx = next((f for f in fixtures
               if f["cycle"] == "cycle126" and f["turn"] == 27), None)
    assert fx is not None
    # Note: t27 itself is not L+1; it is the post-L+1 reflexion turn
    # consuming recent_turns ending at t26 (the L+ event).
    assert fx["expected"]["max_length"] == 380


# ---- Scoring-function unit tests (no LLM) ----


def test_score_m1_coord_in_targets_basic():
    targets = [{"region_id": "R3", "bbox": [28, 14, 33, 19]},
               {"region_id": "R4", "bbox": [36, 14, 41, 19]}]
    assert score_m1_coord_in_targets([30, 16], targets) is True
    assert score_m1_coord_in_targets([38, 16], targets) is True
    assert score_m1_coord_in_targets([5, 5], targets) is False
    assert score_m1_coord_in_targets([], targets) is False


def test_score_m1_manhattan():
    # |0-38|+|0-54|=92 ≥ 35
    assert score_m1_manhattan_from_anchor([0, 0], [38, 54], 35) is True
    # 0 distance < 35
    assert score_m1_manhattan_from_anchor([38, 54], [38, 54], 35) is False
    # |10-38|+|10-54|=72 ≥ 35
    assert score_m1_manhattan_from_anchor([10, 10], [38, 54], 35) is True
    # |35-38|+|50-54|=7 < 35
    assert score_m1_manhattan_from_anchor([35, 50], [38, 54], 35) is False
    # Boundary: exactly 35
    assert score_m1_manhattan_from_anchor([3, 54], [38, 54], 35) is True


def test_score_m3_keyword():
    cards = [
        {"rule_hypothesis": "Click R3 then R4 to test marker crop alignment",
         "predicate": ""},
        {"rule_hypothesis": "Single toggle on R8", "predicate": "color flip"},
    ]
    assert score_m3_keyword_in_card_text(
        cards, ["marker crop", "compass sweep"]) is True
    assert score_m3_keyword_in_card_text(
        cards, ["does not appear"]) is False


def test_score_reflexion_basic():
    expected = {
        "must_not_start_with": "[CAUSE",
        "max_length": 380,
        "must_contain_any": ["1824", "flood"],
    }
    good = "L+1 event on R32 with count=1824 transition 5→4."
    assert score_reflexion_summary(good, expected) is True
    bad_prefix = "[CAUSE turn=26 region=R32 count=1824] " + good
    assert score_reflexion_summary(bad_prefix, expected) is False
    bad_no_token = "L+1 event on R32 with transition 5→4."
    assert score_reflexion_summary(bad_no_token, expected) is False
    bad_long = "x" * 400 + " 1824"
    assert score_reflexion_summary(bad_long, expected) is False


# ---------------------------------------------------------------------------
# Live-LLM tests — opt-in via RUN_V586_LIVE=1. Skip otherwise.
# ---------------------------------------------------------------------------

LIVE = os.environ.get("RUN_V586_LIVE", "0") == "1"


@pytest.mark.skipif(not LIVE, reason="set RUN_V586_LIVE=1 to run")
def test_v2_baseline_smoke_m1_cycle126_t49():
    """Single-trial smoke of H_M1.A pivot. Full V2 measurement uses
    scripts/run_splice_v586.py with N=10 per fixture."""
    import asyncio
    os.environ.setdefault("ARC_NO_GOAL_LEAK", "1")
    from agents.templates.agentica_v57.agent import call_action  # noqa

    fixtures = _load_fixtures("M1")
    fx = next(f for f in fixtures
              if f["cycle"] == "cycle126" and f["turn"] == 49)
    inp = fx["input"]

    async def _run():
        return await call_action(
            summary=inp["summary"],
            visible_regions=inp["visible_regions"],
            active_hypotheses=[{"id": cid, "predicate": "(reconstructed)",
                                "type": "single"}
                               for cid in inp["active_hypotheses_ids"]][:5],
            recent_turns=inp["recent_turns"],
            gqb_pair=None,
            image_b64=None,
        )

    out = asyncio.run(_run())
    coord = (out.get("action") or {}).get("coord")
    print(f"\nLIVE smoke t49 coord={coord} thought_head="
          f"{(out.get('thought') or '')[:120]}")
    # No assertion — V2 is a measurement, not a gate.
