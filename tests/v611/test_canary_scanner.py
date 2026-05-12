"""v611 Step 1 — canary scanner fixture suite.

Plan v611 rev B §canary leak test, FROZEN spec. Codex 7-round
adversarial review converged.

These fixtures encode the SCANNER CONTRACT:
- canon() pipeline is deterministic & shared
- per-source independent scan (no cross-source n-grams)
- 0 false negatives on positive canary phrases
- low false positives on neutral text (baseline B)
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.templates.agentica_lite.v611_leak_scanner import (  # noqa: E402
    CANARY_BIGRAMS,
    CANARY_TRIGRAMS,
    CANARY_UNIGRAMS,
    HygieneResult,
    canon,
    filter_inheritable_skills,
    ngrams,
    scan_all_sources,
    scan_artifact,
    validate_confirmed_skill,
)


# ─────────────────────────────────────────────────────────────────
# canon() pipeline
# ─────────────────────────────────────────────────────────────────


def test_canon_lowercase_strip_punct():
    """canon('2-step Rotation!') -> ['2', 'step', 'rotation']."""
    assert canon("2-step Rotation!") == ["2", "step", "rotation"]


def test_canon_keeps_digits():
    """digits retained (we don't want 'L1' lost — handled separately)."""
    assert canon("Click 38 44") == ["click", "38", "44"]


def test_canon_collapses_whitespace():
    assert canon("a    b  \n c") == ["a", "b", "c"]


def test_canon_empty_input():
    assert canon("") == []
    assert canon(None) == []
    assert canon("   ") == []


# ─────────────────────────────────────────────────────────────────
# Phrase library compilation
# ─────────────────────────────────────────────────────────────────


def test_phrase_library_includes_quadrant_unigram():
    assert "quadrant" in CANARY_UNIGRAMS


def test_phrase_library_includes_two_step_bigram():
    """Both '2-step' and 'two step' must canonicalize to bigrams in the
    library. '2-step' -> ['2','step'], 'two step' -> ['two','step']."""
    assert ("2", "step") in CANARY_BIGRAMS
    assert ("two", "step") in CANARY_BIGRAMS


def test_phrase_library_includes_trigram():
    assert ("q4", "wraps", "to") in CANARY_TRIGRAMS
    assert ("two", "step", "rotation") in CANARY_TRIGRAMS


# ─────────────────────────────────────────────────────────────────
# Positive canary detection (TRUE POSITIVES)
# ─────────────────────────────────────────────────────────────────


def test_scan_detects_quadrant_unigram():
    """Episode-2 thought literally references 'quadrant' -> hit."""
    hit = scan_artifact(
        "m1_thought",
        "Looking at the grid, I see the quadrant structure clearly.",
    )
    assert "quadrant" in hit.unigram_hits
    assert hit.total() >= 1


def test_scan_detects_two_step_paraphrase():
    """Paraphrase of canary ('two step') is detected as bigram."""
    hit = scan_artifact(
        "m2_thought",
        "The strategy follows a two step pattern across rows.",
    )
    assert ("two", "step") in hit.bigram_hits


def test_scan_detects_q4_wraps_trigram():
    """Trigram match (q4 wraps to)."""
    hit = scan_artifact(
        "m4_paragraph",
        "Observation: q4 wraps to the start after a few clicks.",
    )
    assert ("q4", "wraps", "to") in hit.trigram_hits


# ─────────────────────────────────────────────────────────────────
# Negative cases (NO false positives on ARC-AGI vocab)
# ─────────────────────────────────────────────────────────────────


def test_scan_neutral_text_no_hits():
    """A neutral M1 thought about ft09 (no canary tokens) -> 0 hits."""
    hit = scan_artifact(
        "m1_thought",
        ("I will click near the bottom-right tile because the marker "
         "appears to require neighbor satisfaction along the east "
         "edge. Expected effect: frame changes."),
    )
    assert hit.total() == 0, f"unexpected hits: {hit}"


def test_scan_marker_constraint_vocab_safe():
    """v608 marker_constraint terms should NOT trigger canary."""
    hit = scan_artifact(
        "m1_thought",
        ("marker C4 has neighbor at slot E with relation different. "
         "satisfaction count is 3 of 8. clicking color 5 region."),
    )
    assert hit.total() == 0


def test_scan_does_not_collide_cross_source():
    """Per-source independent scan: 'two' in source A + 'step' in
    source B should NOT form bigram. (Verified by separate calls.)
    """
    hit_a = scan_artifact("m1_thought", "I see two distinct markers.")
    hit_b = scan_artifact("m2_thought", "The next step is to click.")
    # Neither source has bigram 'two step' internally.
    assert ("two", "step") not in hit_a.bigram_hits
    assert ("two", "step") not in hit_b.bigram_hits
    # Even when scanned in same call, sources are independent.
    total, per_src = scan_all_sources({
        "m1_thought": "I see two distinct markers.",
        "m2_thought": "The next step is to click.",
    })
    assert total == 0


# ─────────────────────────────────────────────────────────────────
# scan_all_sources aggregation
# ─────────────────────────────────────────────────────────────────


def test_scan_all_sources_aggregates_per_source():
    sources = {
        "m1_thought": "Looking at the grid quadrant.",
        "m2_thought": "The next step is to click safely.",
        "m4_paragraph": "q4 wraps to nothing meaningful here.",
    }
    total, per_src = scan_all_sources(sources)
    src_names = {h.source_name for h in per_src}
    # m1 has 'quadrant', m4 has 'q4 wraps to'. m2 is clean.
    assert "m1_thought" in src_names
    assert "m4_paragraph" in src_names
    assert "m2_thought" not in src_names
    assert total >= 2


def test_scan_all_sources_clean_returns_zero():
    sources = {
        "m1_thought": "Standard marker repair analysis.",
        "m4_paragraph": "Predicted effect matches observed delta.",
    }
    total, per_src = scan_all_sources(sources)
    assert total == 0
    assert per_src == []


# ─────────────────────────────────────────────────────────────────
# Hygiene validator for confirmed_skill cards
# ─────────────────────────────────────────────────────────────────


def test_hygiene_clean_nl_only_skill_passes():
    skill = {
        "skill_id": "S-NL-001",
        "nl_description": ("Clicking near a corner tile of the same color "
                            "as a marker's neighbor triggers a satisfaction "
                            "change at that marker."),
        "abstract_precondition": "marker has at least one unsatisfied neighbor",
        "expected_observed_effect": "neighbor satisfaction count decreases",
    }
    res = validate_confirmed_skill(skill)
    assert res.ok, f"clean skill rejected: {res.violations}"


def test_hygiene_rejects_raw_coordinate():
    skill = {
        "skill_id": "S-LEAK-001",
        "nl_description": "Click at (38, 38) to advance.",
    }
    res = validate_confirmed_skill(skill)
    assert not res.ok
    assert any("38" in v for v in res.violations)


def test_hygiene_rejects_region_id():
    skill = {
        "skill_id": "S-LEAK-002",
        "nl_description": "Target region C5 for the next click.",
    }
    res = validate_confirmed_skill(skill)
    assert not res.ok
    assert any("C5" in v for v in res.violations)


def test_hygiene_rejects_level_id():
    skill = {
        "skill_id": "S-LEAK-003",
        "nl_description": "This skill works on L2 onwards.",
    }
    res = validate_confirmed_skill(skill)
    assert not res.ok
    assert any("L2" in v for v in res.violations)


def test_hygiene_rejects_ft09_vocab():
    skill = {
        "skill_id": "S-LEAK-004",
        "nl_description": "The bsT marker responds to gqb-type clicks.",
    }
    res = validate_confirmed_skill(skill)
    assert not res.ok
    vocab_hits = [v for v in res.violations if ("bsT" in v or "gqb" in v)]
    assert len(vocab_hits) >= 2


def test_filter_inheritable_skills_separates_clean_from_dirty():
    skills = [
        {"skill_id": "S-A", "nl_description": "Generic marker repair pattern."},
        {"skill_id": "S-B", "nl_description": "Click at (44, 44) directly."},
        {"skill_id": "S-C", "nl_description": "Region C3 trick."},
        {"skill_id": "S-D", "nl_description": "Standard tile toggle behavior."},
    ]
    clean, rejected = filter_inheritable_skills(skills)
    clean_ids = {s["skill_id"] for s in clean}
    rejected_ids = {r.skill_id for r in rejected}
    assert clean_ids == {"S-A", "S-D"}
    assert rejected_ids == {"S-B", "S-C"}


# ─────────────────────────────────────────────────────────────────
# Smoke: end-to-end canary detection on a sample episode artifact
# ─────────────────────────────────────────────────────────────────


def test_canary_episode2_smoke_detected():
    """Simulate episode-2 M1 thought that references the inherited
    canary card. Scanner must detect the contamination."""
    canary_card_text = (
        "Quadrant 4 always wraps to quadrant 0 after 2-step rotation."
    )
    # Episode-2 M1 references the inherited canary in its reasoning.
    episode2_m1 = (
        "Based on the prior skill about quadrant wrapping, I will "
        "attempt a 2-step click sequence next."
    )
    sources = {
        "canary_card_in_skill_md": canary_card_text,
        "episode2_m1_thought": episode2_m1,
    }
    total, per_src = scan_all_sources(sources)
    src_names = {h.source_name for h in per_src}
    assert total > 0
    assert "canary_card_in_skill_md" in src_names
    assert "episode2_m1_thought" in src_names


def test_canary_episode2_clean_zero_hits():
    """Episode-2 with NO canary inheritance and NO contamination -> 0."""
    sources = {
        "episode2_m1_thought": ("I observe a marker with unsatisfied neighbor "
                                 "on the east side. I will click near it."),
        "episode2_m4_paragraph": ("Predicted: frame changes. Observed: "
                                    "frame changed. Hypothesis supported."),
    }
    total, per_src = scan_all_sources(sources)
    assert total == 0
