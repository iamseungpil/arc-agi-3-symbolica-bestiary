"""B19 v591 — chid grammar acceptance tests.

Cycle237's invented chids define the empirical reference set. Every chid
in CYCLE237_INVENTED must validate when its referenced region is in the
visible set; rejection cases come from synthetic edge inputs.
"""

from __future__ import annotations

import pytest

from tools.chid_grammar import (
    is_non_trivial_invention,
    is_tier_a_predicate_id,
    parse_invented_chid,
    template_id,
    validate_invented_chid,
)

# Real cycle237 invented chids extracted from
# simple_logs/ft09-9ab2447a/v57_1778180868_3399613/trace.jsonl. Chids
# without R-id but containing a history-keyword (replay/trigger/prior/
# overlap/relation/plan/complete) are accepted via _HISTORY_KEYWORDS.
CYCLE237_INVENTED_REGION_ANCHORED = [
    # (chid, expected_visible_region)
    ("H_static_nonmarker_R19", "R19"),
    ("H_crop_align_R31_NW", "R31"),
    ("H_crop_align_R31_W", "R31"),
    ("P_crop_compass_sweep_R31", "R31"),
    ("H_fresh_neighbor_toggle_R16", "R16"),
    ("H_fresh_neighbor_toggle_R9", "R9"),
    ("H_crop2_complete_R12_W", "R12"),
    ("P_shared_blank_sweep_R12", "R12"),
    ("H_upper_crop2_R3", "R3"),
    ("H_shared_blank_R8", "R8"),
    ("H_R15_lower_S", "R15"),
    ("H_R5_upper_W", "R5"),
    ("H_R14_lower_SW", "R14"),
    ("P_R6_crop_sector_alignment", "R6"),
    ("P_R12_crop_sector_alignment", "R12"),
]

# History-anchored chids actually emitted in cycle237 — no R-id, no
# T-token, only a temporal keyword in the body.
CYCLE237_INVENTED_HISTORY_ANCHORED = [
    "H_replay_prior_trigger",
    "H_crop_overlap_probe",
    "H_crop_relation_plan",
    "P_complete_upper_crop2",
]

# ---------------- Acceptance ----------------


@pytest.mark.parametrize("chid,visible_rid", CYCLE237_INVENTED_REGION_ANCHORED)
def test_cycle237_region_anchored_chids_accepted(chid, visible_rid):
    ok, reason = validate_invented_chid(chid, [visible_rid])
    assert ok, f"{chid} should be accepted but got reason={reason}"


@pytest.mark.parametrize("chid", CYCLE237_INVENTED_HISTORY_ANCHORED)
def test_cycle237_history_anchored_chids_accepted(chid):
    ok, reason = validate_invented_chid(chid, ["R5", "R10"])
    assert ok, f"{chid} should be accepted (history keyword) but got {reason}"


def test_explicit_t_token_accepted_without_region():
    chid = "H_replay_T34_decision"
    ok, reason = validate_invented_chid(chid, ["R5", "R10"])
    assert ok, reason


# ---------------- Rejection ----------------


def test_reject_ungrammatical_no_prefix():
    ok, reason = validate_invented_chid("crop_align_R31_NW", ["R31"])
    assert not ok and reason == "ungrammatical"


def test_reject_no_anchor_no_history():
    # Body lacks R-id, T-token AND any history keyword.
    ok, reason = validate_invented_chid("H_static_uniform", ["R31"])
    assert not ok and reason == "no_anchor"


def test_reject_region_not_visible():
    ok, reason = validate_invented_chid("H_crop_align_R31_NW", ["R5"])
    assert not ok and reason == "region_not_visible"


def test_reject_leak_vocab_target_color():
    ok, reason = validate_invented_chid("H_target_color_R5", ["R5"])
    assert not ok and reason.startswith("leak_vocab:")


def test_reject_leak_vocab_win_state():
    ok, reason = validate_invented_chid("P_win_state_R5", ["R5"])
    assert not ok and reason.startswith("leak_vocab:")


def test_reject_too_short_body():
    # body must be at least 3 chars after the H_/P_ prefix.
    ok, reason = validate_invented_chid("H_ab", ["R5"])
    assert not ok and reason == "ungrammatical"


def test_reject_too_long_body():
    long_body = "a" * 100
    ok, reason = validate_invented_chid(f"H_{long_body}_R5", ["R5"])
    assert not ok and reason == "ungrammatical"


def test_reject_special_characters():
    ok, reason = validate_invented_chid("H_crop-align_R5", ["R5"])
    assert not ok and reason == "ungrammatical"


# ---------------- TIER-A discriminator ----------------


def test_tier_a_predicate_id_accepted_by_discriminator():
    assert is_tier_a_predicate_id("P01_unclicked_neighbor_of_active_marker:T0:R21")


def test_invented_chid_not_a_tier_a():
    assert not is_tier_a_predicate_id("H_static_nonmarker_R19")


# ---------------- Non-trivial invention (V-H4) ----------------


def test_non_trivial_with_direction():
    assert is_non_trivial_invention("H_crop_align_R31_NW")


def test_non_trivial_with_relation_keyword():
    assert is_non_trivial_invention("P_R12_crop_sector_alignment")


def test_non_trivial_fresh_neighbor_toggle():
    # Review issue C2 — the keyword extension covers cycle237's
    # H_fresh_neighbor_toggle_R* family.
    assert is_non_trivial_invention("H_fresh_neighbor_toggle_R16")


def test_non_trivial_replay_prior_trigger():
    assert is_non_trivial_invention("H_replay_prior_trigger")


def test_trivial_chid_with_region_only():
    # Just primitive click — no direction, no rich rationale.
    assert not is_non_trivial_invention("H_click_R5")


# ---------------- template_id stability (cross-run) ----------------


def test_template_id_strips_region_id():
    assert template_id("H_crop_align_R31_NW") == template_id("H_crop_align_R12_NW")


def test_template_id_keeps_direction():
    a = template_id("H_crop_align_R31_NW")
    b = template_id("H_crop_align_R31_SW")
    assert a is not None and b is not None and a != b


def test_template_id_keeps_rationale_distinct():
    a = template_id("P_R12_crop_sector_alignment")
    b = template_id("P_R12_crop_compass_sweep")
    assert a is not None and b is not None and a != b


# ---------------- parse helper ----------------


def test_parse_extracts_region_ids():
    p = parse_invented_chid("H_R15_lower_S")
    assert p is not None
    assert p["region_ids"] == ["R15"]
    assert p["direction"] == "S"


def test_parse_returns_none_for_invalid():
    assert parse_invented_chid("not-a-chid") is None


# ---------------- Cited region extraction (W7 cross-check) ----------------


def test_cited_region_returns_first_r_id():
    from tools.chid_grammar import cited_region_id

    assert cited_region_id("H_crop_align_R31_NW") == "R31"
    assert cited_region_id("P_R12_crop_sector_alignment") == "R12"


def test_cited_region_none_for_history_anchored():
    from tools.chid_grammar import cited_region_id

    assert cited_region_id("H_replay_prior_trigger") is None


# ---------------- Future-proof discriminator (S16) ----------------


def test_tier_a_three_digit_template_accepted():
    assert is_tier_a_predicate_id("P100_some_template:T5:R12")
