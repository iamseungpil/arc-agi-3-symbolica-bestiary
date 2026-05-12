"""v609 P5: M-004 + P17 seed-card leak guard.

Ensures the new card text does NOT introduce forbidden ft09 tokens via
the rendered SKILL.md path, and that standard scoring skips P17.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.templates.agentica_lite.policy import (
    V608F_REPEAT_PREDICATE_ID,
    V609_RESERVED_PREDICATE_IDS,
    V609_SEARCH_PREDICATE_ID,
)
from agents.templates.agentica_lite.seed_cards import ensure_seeded
from agents.templates.agentica_lite.skill_md_renderer import render
from agents.templates.agentica_lite.skill_state import SkillState, SkillStateMetadata


# V_LEAK ft09 tokens are mixed-case identifiers; production leak guard
# checks them as substrings without case folding. Alpha-only cycle237
# vocab (compass / sweep / ...) is already grandfathered into H-001's
# falsifier text, so we restrict the alpha check to the NEW M-004 fields.
_V_LEAK_TOKENS = (
    "R28", "R31", "R12",
    "gqb", "bsT", "Hkx", "NTi", "kCv", "cwU", "elp", "Ycb",
)
_FORBIDDEN_ALPHA = (
    "compass", "sweep", "sector", "alignment", "crop",
)


def test_m004_present_after_seeding():
    s = SkillState(metadata=SkillStateMetadata(game_id="ft09-test"))
    n = ensure_seeded(s)
    ids = {c.id for c in s.cards}
    assert "M-004" in ids
    m004 = next(c for c in s.cards if c.id == "M-004")
    assert m004.policy_hooks == ["P17_search_step"]
    assert m004.status == "active"


def test_m004_text_has_no_forbidden_tokens():
    """The NEW M-004 fields must be leak-free. We assemble its claim +
    predicts + falsifiers + state_features + policy_hooks and run both
    the case-sensitive V_LEAK check and the lowercase alpha-vocab check
    against that bundle only — existing seed cards remain grandfathered
    where v608d already documented them."""
    s = SkillState(metadata=SkillStateMetadata(game_id="ft09-test"))
    ensure_seeded(s)
    m004 = next(c for c in s.cards if c.id == "M-004")
    bundle = " ".join([
        m004.claim,
        " ".join(m004.predicts),
        " ".join(m004.falsifiers),
        " ".join(m004.state_features),
        " ".join(m004.policy_hooks),
    ])
    for tok in _V_LEAK_TOKENS:
        assert tok not in bundle, (
            f"V_LEAK token '{tok}' appears in M-004 fields: {bundle!r}"
        )
    bundle_lower = bundle.lower()
    for tok in _FORBIDDEN_ALPHA:
        assert tok not in bundle_lower, (
            f"alpha vocab '{tok}' appears in M-004 fields: {bundle!r}"
        )


def test_m004_skill_md_render_has_no_v_leak_tokens():
    """SKILL.md rendering must not introduce any V_LEAK ft09 token. The
    alpha cycle237 vocab is grandfathered in older seed cards and is
    checked separately on the M-004 bundle above."""
    s = SkillState(metadata=SkillStateMetadata(game_id="ft09-test"))
    ensure_seeded(s)
    md = render(s)
    for tok in _V_LEAK_TOKENS:
        assert tok not in md, (
            f"V_LEAK token '{tok}' surfaced in rendered SKILL.md"
        )


def test_v609_reserved_ids_contains_p16_and_p17():
    assert V608F_REPEAT_PREDICATE_ID in V609_RESERVED_PREDICATE_IDS
    assert V609_SEARCH_PREDICATE_ID in V609_RESERVED_PREDICATE_IDS


def test_p17_predicate_registered():
    """The P17 predicate must be registered so library.all_predicates()
    can resolve_coord on it when the solver stamps an action."""
    from agents.templates.agentica_lite.predicate_library import (
        PredicateLibrary,
    )
    lib = PredicateLibrary()
    preds = lib.all_predicates()
    assert "P17_search_step" in preds
    assert preds["P17_search_step"].family == "graph_search"


def test_p17_not_picked_by_standard_scoring():
    """End-to-end check: select_arm with no override conditions must NOT
    return a P17-stamped arm. The reserved-ids guard ensures this."""
    from agents.templates.agentica_lite.policy import EpisodeState, select_arm
    from agents.templates.agentica_lite.predicate_library import (
        PredicateLibrary,
    )
    from agents.templates.agentica_lite.predicate_posterior import (
        PredicatePosterior,
    )
    state = {
        "visible_regions": [
            {"id": "R0", "region_id": "R0", "bbox": [10, 10, 12, 12],
             "size": 9, "color": 8, "is_multicolor": False, "y_band": "mid"},
            {"id": "R1", "region_id": "R1", "bbox": [40, 40, 42, 42],
             "size": 9, "color": 4, "is_multicolor": False, "y_band": "low"},
        ],
        "marker_neighbor_states": [],
        "marker_constraints": [],
        "marker_constraint_summary": {"total": 0, "unsatisfied": 0,
                                       "by_marker": {}},
        "last_observation": {"primary_region_id": None, "level_delta": 0,
                              "dominant_transition": None},
    }
    ep = EpisodeState()
    decision = select_arm(state, PredicatePosterior(), PredicateLibrary(),
                          None, ep)
    assert decision is not None
    assert decision.arm_key.predicate_id != V609_SEARCH_PREDICATE_ID
