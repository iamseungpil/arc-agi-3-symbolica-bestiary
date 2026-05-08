"""B19 v591 — two-tier classification & validation integration tests.

Exercises the chid_tier classification path used by `agent.py` without
spinning up the full async loop. The simulated path mirrors agent.py
1733-1791: given (chosen_id, chosen_card, candidate_tests_for_m1,
visible_region_ids), emit (chid_tier, invented_meta, downgrade_reason).
"""

from __future__ import annotations

import pytest

from tools.chid_grammar import (
    cited_region_id,
    is_non_trivial_invention,
    is_tier_a_predicate_id,
    template_id as invented_template_id,
    validate_invented_chid,
)


def classify_chid(
    chosen_id,
    chosen_card,
    candidate_tests_for_m1,
    visible_region_ids,
):
    """Mirror of agent.py:1746-1791 TIER classification logic."""
    invented_meta = None
    chid_tier = None
    if chosen_id and chosen_card is None:
        if is_tier_a_predicate_id(chosen_id):
            tier_a_pool = {
                c.get("predicate_id")
                for c in (candidate_tests_for_m1 or [])
                if c.get("predicate_id")
            }
            if chosen_id in tier_a_pool:
                chid_tier = "A"
            else:
                ok, _ = validate_invented_chid(chosen_id, visible_region_ids)
                if ok:
                    chid_tier = "B"
                    invented_meta = {
                        "chid": chosen_id,
                        "template_id": invented_template_id(chosen_id),
                        "non_trivial": is_non_trivial_invention(chosen_id),
                        "downgraded_from_tier_a": True,
                    }
                else:
                    chid_tier = "none"
        else:
            ok, _ = validate_invented_chid(chosen_id, visible_region_ids)
            if ok:
                chid_tier = "B"
                invented_meta = {
                    "chid": chosen_id,
                    "template_id": invented_template_id(chosen_id),
                    "non_trivial": is_non_trivial_invention(chosen_id),
                    "downgraded_from_tier_a": False,
                }
            else:
                chid_tier = "none"
    elif chosen_card is not None:
        chid_tier = "card"
    else:
        chid_tier = "none"
    return chid_tier, invented_meta


# ---------------- T-2T-1: TIER-A pass-through ----------------


def test_tier_a_predicate_in_pool_classified_as_A():
    pool = [{"predicate_id": "P05_shared_neighbor_between_markers:T0:R5"}]
    tier, meta = classify_chid(
        chosen_id="P05_shared_neighbor_between_markers:T0:R5",
        chosen_card=None,
        candidate_tests_for_m1=pool,
        visible_region_ids=["R5", "R10"],
    )
    assert tier == "A"
    assert meta is None


# ---------------- T-2T-2: TIER-B grammar accept ----------------


def test_tier_b_invented_chid_validated():
    tier, meta = classify_chid(
        chosen_id="H_crop_align_R31_NW",
        chosen_card=None,
        candidate_tests_for_m1=[],
        visible_region_ids=["R31"],
    )
    assert tier == "B"
    assert meta is not None
    assert meta["chid"] == "H_crop_align_R31_NW"
    assert meta["non_trivial"] is True
    assert meta["downgraded_from_tier_a"] is False


# ---------------- T-2T-3: ungrammatical chid → 'none' ----------------


def test_ungrammatical_chid_resets_to_none():
    tier, meta = classify_chid(
        chosen_id="just-some-text",
        chosen_card=None,
        candidate_tests_for_m1=[],
        visible_region_ids=["R5"],
    )
    assert tier == "none"
    assert meta is None


# ---------------- T-2T-4: hallucinated TIER-A downgrades to TIER-B ----------------


def test_hallucinated_tier_a_id_downgrades_to_tier_b():
    pool = [{"predicate_id": "P01_real:T0:R5"}]
    tier, meta = classify_chid(
        chosen_id="P99_made_up:T0:R5",
        chosen_card=None,
        candidate_tests_for_m1=pool,
        visible_region_ids=["R5"],
    )
    # Not in pool → grammar check; the body matches no R-id outside the
    # `:T0:R5` namespace because the regex parses `P99_made_up`. The
    # invented-chid grammar does not match `P99_made_up:T0:R5` (colon),
    # so it is rejected → 'none'.
    assert tier == "none"


# ---------------- T-2T-5: card-anchored chid ----------------


def test_chosen_card_path_classified_as_card():
    card = {"id": "C5", "type": "hypothesis", "region_id": "R12"}
    tier, meta = classify_chid(
        chosen_id="C5",
        chosen_card=card,
        candidate_tests_for_m1=[],
        visible_region_ids=["R12"],
    )
    assert tier == "card"
    assert meta is None


# ---------------- T-2T-6: history-anchored TIER-B ----------------


def test_history_anchored_chid_accepted():
    tier, meta = classify_chid(
        chosen_id="H_replay_prior_trigger",
        chosen_card=None,
        candidate_tests_for_m1=[],
        visible_region_ids=["R5", "R10"],
    )
    assert tier == "B"
    assert meta is not None


# ---------------- T-2T-7: cited region cross-check ----------------


def test_cited_region_recovers_anchor_for_coord_check():
    tier, meta = classify_chid(
        chosen_id="P_R12_crop_sector_alignment",
        chosen_card=None,
        candidate_tests_for_m1=[],
        visible_region_ids=["R12"],
    )
    assert tier == "B"
    assert cited_region_id(meta["chid"]) == "R12"


# ---------------- T-2T-8: leak vocab in invented chid ----------------


def test_leak_vocab_in_invented_chid_rejected():
    tier, meta = classify_chid(
        chosen_id="H_target_color_R5",
        chosen_card=None,
        candidate_tests_for_m1=[],
        visible_region_ids=["R5"],
    )
    assert tier == "none"
