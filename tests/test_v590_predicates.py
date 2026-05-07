"""B18 plan v590 — predicate_generator tests.

15 tests per plan v590 §4.5 (round-3 frozen).
"""
import json
import re
import pytest

from tools.predicate_generator import (
    PREDICATE_LIBRARY,
    _FORBIDDEN_VOCAB,
    _audit_template_id,
    _beta_bernoulli_score,
    _coord_for,
    ChainState,
    RegionRef,
    build_chain_state_from_inputs,
    generate_predicates,
    update_predicate_log_with_observation,
)


def _marker(rid="R12", color=12, neighbors=None, click_response=None):
    return {
        "id": rid, "color": color, "is_multicolor": True, "size": 36,
        "bbox": {"min_x": 10, "min_y": 10, "max_x": 18, "max_y": 18},
        "neighbors_3x3": neighbors or {"N": "R20", "E": "R21", "S": "R22", "W": "R23"},
        "click_response": click_response or {"clicks": 1, "responses": 1},
    }


def _non_marker(rid="R20", color=9, bbox=None):
    return {
        "id": rid, "color": color, "is_multicolor": False, "size": 9,
        "bbox": bbox or {"min_x": 30, "min_y": 30, "max_x": 38, "max_y": 38},
    }


def _diff(turn, rid="R20", level_delta=0, compass=None):
    return {
        "turn": turn, "click_region_id": rid,
        "level_delta": level_delta,
        "compass_changes": compass or [],
        "color_transitions": [],
    }


def _mns(marker_id="R12", n_color_uniform=False):
    if n_color_uniform:
        compass = {d: {"region_id": f"R2{i}", "current_color": 9, "clicks": 0}
                   for i, d in enumerate(["N","NE","E","SE","S","SW","W","NW"])}
    else:
        compass = {
            "N": {"region_id": "R20", "current_color": 9, "clicks": 0},
            "E": {"region_id": "R21", "current_color": 12, "clicks": 1},
            "S": {"region_id": "R22", "current_color": 9, "clicks": 0},
            "W": {"region_id": "R23", "current_color": 12, "clicks": 0},
        }
    return [{"marker_id": marker_id, "compass": compass}]


# ---------- T-PRED-1..3: basic instantiation ----------


def test_pred_1_p00_fallback_emits_when_marker_visible():
    cs = build_chain_state_from_inputs(
        visible_regions=[_marker()],
        recent_turn_diffs=[], marker_neighbor_states=[], recent_clicks=[],
    )
    out = generate_predicates(chain_state=cs, turn_index=0, k_global=12)
    pids = [c["template_id"] for c in out]
    assert "P00_explore_uncovered_marker" in pids


def test_pred_2_p01_active_marker_neighbor():
    cs = build_chain_state_from_inputs(
        visible_regions=[_marker(), _non_marker(rid="R20"), _non_marker(rid="R21")],
        recent_turn_diffs=[], marker_neighbor_states=[], recent_clicks=[],
    )
    out = generate_predicates(chain_state=cs, turn_index=5, k_global=12)
    p01 = [c for c in out if c["template_id"] == "P01_unclicked_neighbor_of_active_marker"]
    assert p01


def test_pred_3_empty_input():
    cs = build_chain_state_from_inputs(
        visible_regions=[], recent_turn_diffs=[],
        marker_neighbor_states=[], recent_clicks=[],
    )
    out = generate_predicates(chain_state=cs, turn_index=0)
    assert out == []


# ---------- T-PRED-4..5: leak audit ----------


def test_pred_4_v_leak_on_library_source():
    """Audit each template's body source (NOT the entire module —
    that would match the FORBIDDEN_VOCAB regex itself)."""
    import inspect
    for tid, meta in PREDICATE_LIBRARY.items():
        body = inspect.getsource(meta["fn"])
        # Filter to non-empty match segments.
        hits = [h for h in _FORBIDDEN_VOCAB.findall(body) if h]
        # Also do a positive search for full pattern (not group capture).
        m = _FORBIDDEN_VOCAB.search(body)
        assert m is None and not hits, (
            f"forbidden vocab in {tid}: {m.group(0) if m else hits}"
        )


def test_pred_5_emitted_template_ids_in_library():
    cs = build_chain_state_from_inputs(
        visible_regions=[_marker(), _non_marker()],
        recent_turn_diffs=[
            _diff(1, rid="R20", level_delta=1),
            _diff(2, rid="R20", level_delta=0,
                  compass=[{"marker_id": "R12", "direction": "N",
                            "from": 9, "to": 12}]),
        ],
        marker_neighbor_states=_mns(), recent_clicks=["R20"],
    )
    out = generate_predicates(chain_state=cs, turn_index=10, k_global=12)
    for c in out:
        assert c["template_id"] in PREDICATE_LIBRARY


# ---------- T-PRED-6..7: anchor + coord ----------


def test_pred_6_anchor_region_observable():
    cs = build_chain_state_from_inputs(
        visible_regions=[_marker(), _non_marker()],
        recent_turn_diffs=[], marker_neighbor_states=[], recent_clicks=[],
    )
    out = generate_predicates(chain_state=cs, turn_index=0)
    visible_ids = {r["id"] for r in [_marker(), _non_marker()]}
    for c in out:
        assert c["anchor_region"]["region_id"] in visible_ids


def test_pred_7_suggested_coord_in_bbox():
    cs = build_chain_state_from_inputs(
        visible_regions=[_marker(), _non_marker()],
        recent_turn_diffs=[], marker_neighbor_states=[], recent_clicks=[],
    )
    out = generate_predicates(chain_state=cs, turn_index=0)
    for c in out:
        bbox = c["anchor_region"]["bbox"]
        cx, cy = c["suggested_coord"]
        assert bbox["min_x"] <= cx <= bbox["max_x"]
        assert bbox["min_y"] <= cy <= bbox["max_y"]


# ---------- T-PRED-8: score formula ----------


def test_pred_8_beta_bernoulli_score():
    # Cold-start: P00 capped at 0.30
    s = _beta_bernoulli_score("P00_explore_uncovered_marker", {})
    assert s <= 0.30
    # Simple template cold-start: alpha=2, beta=1 → score = 2/3 ≈ 0.667
    s_simple = _beta_bernoulli_score("P01_unclicked_neighbor_of_active_marker", {})
    assert 0.6 <= s_simple <= 0.7
    # Compound template cold-start: alpha=1, beta=1 → 0.5
    s_compound = _beta_bernoulli_score("P03_marker_compass_uniform_breaker", {})
    assert s_compound == 0.5
    # With history: 5 supported, 0 refuted on simple → 7/8 = 0.875
    s_hist = _beta_bernoulli_score(
        "P01_unclicked_neighbor_of_active_marker",
        {"P01_unclicked_neighbor_of_active_marker":
         {"supported": 5, "refuted": 0}},
    )
    assert s_hist > 0.85


# ---------- T-PRED-9..10: caps ----------


def test_pred_9_global_cap():
    # Many markers + non-markers
    visible = [_marker(rid=f"R{i}",
                        neighbors={"N": f"R{i*100}", "E": f"R{i*100+1}",
                                   "S": f"R{i*100+2}", "W": f"R{i*100+3}"})
               for i in range(1, 6)]
    visible += [_non_marker(rid=f"R{i*100+j}", color=9 if j == 0 else 12)
                for i in range(1, 6) for j in range(4)]
    cs = build_chain_state_from_inputs(
        visible_regions=visible,
        recent_turn_diffs=[], marker_neighbor_states=[], recent_clicks=[],
    )
    out = generate_predicates(chain_state=cs, turn_index=10, k_global=12)
    assert len(out) <= 12


def test_pred_10_unique_predicate_ids():
    cs = build_chain_state_from_inputs(
        visible_regions=[_marker(), _non_marker(rid="R20"), _non_marker(rid="R21")],
        recent_turn_diffs=[], marker_neighbor_states=[], recent_clicks=[],
    )
    out = generate_predicates(chain_state=cs, turn_index=5, k_global=12)
    ids = [c["predicate_id"] for c in out]
    assert len(ids) == len(set(ids))


# ---------- T-PRED-11: action-conditioned verdict ----------


def test_pred_11_verdict_supported_when_clicked_anchor():
    log = [{
        "predicate_id": "P01:T5:R20",
        "template_id": "P01_unclicked_neighbor_of_active_marker",
        "anchor_region": {"region_id": "R20",
                          "bbox": {"min_x": 30, "min_y": 30,
                                   "max_x": 38, "max_y": 38}},
        "emitted_at_turn": 5,
    }]
    obs = {"primary_region_id": "R20", "level_delta": 1}
    out = update_predicate_log_with_observation(
        predicate_log=log, observation=obs,
        click_region_id="R20", click_coord=[34, 34], turn_now=6,
    )
    assert len(out) == 1
    assert out[0]["verdict"] == "supported"


def test_pred_12_verdict_refuted_when_clicked_anchor_no_lp():
    log = [{
        "predicate_id": "P01:T5:R20",
        "template_id": "P01_unclicked_neighbor_of_active_marker",
        "anchor_region": {"region_id": "R20",
                          "bbox": {"min_x": 30, "min_y": 30,
                                   "max_x": 38, "max_y": 38}},
        "emitted_at_turn": 5,
    }]
    obs = {"primary_region_id": "R20", "level_delta": 0}
    out = update_predicate_log_with_observation(
        predicate_log=log, observation=obs,
        click_region_id="R20", click_coord=[34, 34], turn_now=7,
    )
    assert len(out) == 1
    assert out[0]["verdict"] == "refuted"


def test_pred_13_verdict_skipped_when_not_clicked_anchor():
    log = [{
        "predicate_id": "P01:T5:R20",
        "template_id": "P01_unclicked_neighbor_of_active_marker",
        "anchor_region": {"region_id": "R20",
                          "bbox": {"min_x": 30, "min_y": 30,
                                   "max_x": 38, "max_y": 38}},
        "emitted_at_turn": 5,
    }]
    obs = {"primary_region_id": "R99", "level_delta": 1}
    out = update_predicate_log_with_observation(
        predicate_log=log, observation=obs,
        click_region_id="R99", click_coord=[5, 5], turn_now=6,
    )
    assert out == []


# ---------- T-PRED-14: bbox-tolerant verdict ----------


def test_pred_14_verdict_bbox_match_even_if_rid_mismatch():
    """Round-2 C-5: accept coord INSIDE anchor bbox even when
    obs.primary_region_id differs (game segmentation can shift)."""
    log = [{
        "predicate_id": "P01:T5:R20",
        "template_id": "P01_unclicked_neighbor_of_active_marker",
        "anchor_region": {"region_id": "R20",
                          "bbox": {"min_x": 30, "min_y": 30,
                                   "max_x": 38, "max_y": 38}},
        "emitted_at_turn": 5,
    }]
    obs = {"primary_region_id": "R_segmented_diff", "level_delta": 1}
    out = update_predicate_log_with_observation(
        predicate_log=log, observation=obs,
        click_region_id=None, click_coord=[34, 34], turn_now=6,
    )
    assert len(out) == 1
    assert out[0]["verdict"] == "supported"


# ---------- T-PRED-15: registry decorator audit ----------


def test_pred_15_all_templates_decorated_with_required_metadata():
    """Round-3 C-12: every entry in PREDICATE_LIBRARY has required fields."""
    for tid, meta in PREDICATE_LIBRARY.items():
        assert "fn" in meta
        assert "complexity" in meta
        assert meta["complexity"] in ("simple", "compound")
        assert "coord_policy" in meta
        assert tid.startswith("P")
        # No leak vocab in template id.
        assert not _FORBIDDEN_VOCAB.search(tid)
