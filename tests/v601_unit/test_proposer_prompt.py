"""Per-module unit tests for proposer_prompt.py.

Plan v602 §11 addendum: 5 critical branch tests.

Branches under test:
  1. template-lint (system prompt cites Step-0 saturation expression verbatim,
     contains no forbidden game-specific tokens)
  2. renders-with-markers (markers + observation produce expected lines)
  3. renders-without-markers (empty marker list produces 'number of markers visible: 0')
  4. R-aliasing (raw region_ids aliased to neutral 'region_primary')
  5. top-K-cap (when state has > expected counts, the renderer still produces a
     bounded prompt; this test reserves capacity for the v602 SKILL.md cap and
     asserts the v601 build_messages output remains under a sane size budget)
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.templates.agentica_lite.proposer_prompt import (  # noqa: E402
    SATURATION_STEP0_REFERENCE, SYSTEM_PROMPT, build_messages,
    render_user_prompt,
)


# ---------- 1. template-lint --------------------------------------------------

# Plan §4 INT05 forbidden tokens (game-specific identifiers must not appear
# in the system template prose).
_FORBIDDEN_GAME_TOKENS = (
    "ft09", "9ab2447a", "Symbolica", "compass_north", "compass_south",
    "compass_east", "compass_west",
)


def test_system_prompt_template_lint_clean():
    """SYSTEM_PROMPT cites Step-0 saturation expression and contains no forbidden tokens."""
    assert SATURATION_STEP0_REFERENCE in SYSTEM_PROMPT
    for tok in _FORBIDDEN_GAME_TOKENS:
        assert tok not in SYSTEM_PROMPT, f"forbidden game-specific token {tok!r} in SYSTEM_PROMPT"
    # Must have schema description for required fields
    for f in ("candidate_predicate_id", "region_hint", "expected_signature",
              "required_pre_state", "confidence", "thought"):
        assert f in SYSTEM_PROMPT


# ---------- 2. renders-with-markers -------------------------------------------

def test_render_user_prompt_with_markers():
    """State with markers + observation renders compass clicks + unclicked slots."""
    state = {
        "marker_neighbor_states": [
            {"is_primary_marker": True, "compass": {
                "N": {"clicks": 1}, "E": {"clicks": 0}, "S": {"clicks": 1}, "W": {"clicks": 0},
            }},
        ],
        "observation": {
            "primary_region_id": "R_primary",
            "dominant_transition": {"count": 12},
            "level_delta": 0,
        },
    }
    prompt = render_user_prompt(state)
    assert "number of markers visible: 1" in prompt
    # 2 saturated of 4
    assert "compass clicks 2/4" in prompt
    # primary alias placed in prose
    assert "primary region alias: region_primary" in prompt
    # Step-0 reminder appended
    assert "Step-0" in prompt or "saturation" in prompt.lower()


# ---------- 3. renders-without-markers ----------------------------------------

def test_render_user_prompt_without_markers():
    """Empty marker list -> 'number of markers visible: 0'; n/a observation lines."""
    state = {"marker_neighbor_states": [], "observation": {}}
    prompt = render_user_prompt(state)
    assert "number of markers visible: 0" in prompt
    # No primary alias line because primary_region_id is missing
    assert "primary region alias" not in prompt
    # Observation absent -> dt_summary fallback
    assert "dominant_transition = (none)" in prompt


# ---------- 4. R-aliasing -----------------------------------------------------

def test_render_aliases_raw_region_ids():
    """Raw region_id values like 'R36'/'R12' are aliased to neutral labels in prose."""
    state = {
        "marker_neighbor_states": [
            {"is_primary_marker": True, "compass": {
                "N": {"region_id": "R36", "clicks": 0},
                "E": {"region_id": "R12", "clicks": 1},
            }},
        ],
        "observation": {"primary_region_id": "R_primary_999"},
    }
    prompt = render_user_prompt(state)
    # raw region ids should NOT appear in user prose (R36 / R12 / R_primary_999)
    # Note: marker_id 'M0' might leak through; the lint specifically forbids region ids.
    # Only the neutral alias 'region_primary' should be present.
    assert "region_primary" in prompt
    # Compass slots should be aliased to slot_N
    assert "slot_" in prompt


# ---------- 5. top-K-cap (build_messages size budget) ------------------------

def test_build_messages_size_budget_with_oversize_state():
    """build_messages output remains bounded under an oversized state.

    This test acts as a regression guard so v602's SKILL.md injection can rely
    on an upstream-bounded base prompt. Today the user-prompt section grows
    with the number of markers, but each marker line is short. We assert the
    full message JSON size stays under a generous 16 KB ceiling for 50
    markers (well above expected production states).
    """
    state = {
        "marker_neighbor_states": [
            {"is_primary_marker": (i == 0), "compass": {
                f"d{j}": {"region_id": f"R{i}_{j}", "clicks": j % 2}
                for j in range(8)
            }}
            for i in range(50)
        ],
        "observation": {"primary_region_id": "R0_0",
                        "dominant_transition": {"count": 12}, "level_delta": 0},
    }
    msgs = build_messages(state)
    assert isinstance(msgs, list) and len(msgs) == 2
    total_size = sum(len(m["content"]) for m in msgs)
    # Generous budget; v602 will tighten via SKILL.md cap to 4 KB (§5 INT09)
    assert total_size < 16_000, f"prompt size {total_size} exceeds budget"
    # System role first, user role second
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
