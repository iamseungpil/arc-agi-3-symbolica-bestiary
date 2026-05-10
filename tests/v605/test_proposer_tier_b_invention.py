"""G-ii fixture: Proposer TIER-B region-anchored predicate invention.

Per codex round 5-7 convergence: before live autoresearch, verify the v605 lite
Proposer can produce TIER-B chids like `P_R{N}_*` when given cycle237 T5 state
inputs. Pre-patch (354/354 emissions = P_saturation_progress) this should fail;
post-patch (Step 1 split into branches A+B) this should pass with ≥1 TIER-B
emission across multiple samples.

This fixture is MOCK — it does NOT call TRAPI. Instead it inspects:
  (1) the system prompt CONTAINS the TIER-B branch language (static check)
  (2) the user prompt with cycle237 T5 state INCLUDES region ids (R1..R30) so
      the LLM has anchors to invent on (static check)
  (3) the schema validator ACCEPTS a TIER-B candidate (e.g. P_R12_crop_sector_alignment)
      with dummy saturation fields (functional check)

The "live" multi-sample diversity test is reserved for the autoresearch smoke,
not for this gate fixture. Gate criterion: all 3 static/functional checks pass.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest


CYCLE237_TRACE = Path("simple_logs/ft09-9ab2447a/v57_1778180868_3399613/trace.jsonl")


def _load_cycle237_t5_state() -> dict:
    """Read cycle237 T5 row from real trace, build lite-compatible state dict."""
    assert CYCLE237_TRACE.exists(), f"missing cycle237 trace: {CYCLE237_TRACE}"
    with CYCLE237_TRACE.open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("turn") == 5:
                vr = row.get("visible_regions") or []
                mns = row.get("marker_neighbor_states") or []
                return {
                    "visible_regions": vr,
                    "marker_neighbor_states": mns,
                    "observation": {"dominant_transition": None, "level_delta": 0},
                }
    raise AssertionError("cycle237 trace lacks T5")


# ---------------------------------------------------------------------------
# Check 1: SYSTEM_PROMPT contains TIER-B branch language
# ---------------------------------------------------------------------------


def test_system_prompt_announces_tier_b_branch():
    from agents.templates.agentica_lite.proposer_prompt import SYSTEM_PROMPT
    # Branch (B) must be advertised
    assert "TIER-B" in SYSTEM_PROMPT, "SYSTEM_PROMPT missing TIER-B mention"
    assert "region-anchored" in SYSTEM_PROMPT, "SYSTEM_PROMPT missing region-anchored language"
    # Example chid for TIER-B must appear
    assert "P_R12" in SYSTEM_PROMPT or "P_R{" in SYSTEM_PROMPT or "P_R" in SYSTEM_PROMPT, \
        "SYSTEM_PROMPT missing TIER-B chid example"
    # Branch (A) must remain
    assert "saturation" in SYSTEM_PROMPT.lower(), "branch (A) saturation path removed"


# ---------------------------------------------------------------------------
# Check 2: cycle237 T5 user prompt exposes visible region ids
# ---------------------------------------------------------------------------


def test_user_prompt_exposes_region_ids():
    from agents.templates.agentica_lite.proposer_prompt import render_user_prompt
    state = _load_cycle237_t5_state()
    body = render_user_prompt(state)
    # at least one R-prefix visible region id should appear (so LLM has anchors)
    import re
    rids = re.findall(r"\bR\d+\b", body)
    assert rids, f"user prompt has no visible R-prefix region ids; body:\n{body[:500]}"


# ---------------------------------------------------------------------------
# Check 3: schema validator accepts a TIER-B candidate with dummy fields
# ---------------------------------------------------------------------------


def test_validator_accepts_tier_b_with_dummy_saturation_fields():
    from agents.templates.agentica_lite.proposer import parse_and_validate
    # Visible region list mimicking what cycle237 T5 had
    state = _load_cycle237_t5_state()
    vis_ids = [
        (r.get("region_id") or r.get("id"))
        for r in (state.get("visible_regions") or [])
        if r
    ]
    # Pick a marker_id and a DIFFERENT region_hint (cycle237 T5 pattern)
    # cycle237 used P_R12_crop_sector_alignment + region_hint = compass neighbor
    mns = state.get("marker_neighbor_states") or []
    assert mns, "no markers in cycle237 T5"
    marker = mns[0]
    marker_id = marker.get("marker_id")
    compass = marker.get("compass") or {}
    # pick first compass neighbor region_id different from marker_id
    region_hint = None
    for slot in compass.values():
        rid = (slot or {}).get("region_id")
        if rid and rid != marker_id:
            region_hint = rid
            break
    assert region_hint and marker_id and region_hint != marker_id, \
        f"could not derive distinct marker_id/region_hint from T5; marker={marker_id} compass={list(compass)}"
    # cycle237's actual winning chid form: P_R{N}_crop_sector_alignment
    candidate = {
        "candidate_predicate_id": f"P_{marker_id}_crop_sector_alignment",
        "region_hint": region_hint,
        "expected_signature": {"level_delta": 1},
        "required_pre_state": {
            "marker_id": marker_id,
            "saturation_threshold": 0,
            "saturation_denominator": 1,
        },
        "confidence": 0.5,
        "thought": "Step-0 saturation = mean(c.clicks >= 1 for c in M.compass); "
                   "TIER-B region-anchored rationale: sector alignment hypothesis.",
    }
    result = parse_and_validate(candidate, vis_ids)
    assert result.failure_reason is None, \
        f"validator rejected TIER-B candidate: {result.failure_reason} / {result.schema_error_code}"
    assert result.output is not None
    assert marker_id in result.output.candidate_predicate_id, \
        "expected region-anchored predicate ID preserved"


if __name__ == "__main__":
    test_system_prompt_announces_tier_b_branch()
    test_user_prompt_exposes_region_ids()
    test_validator_accepts_tier_b_with_dummy_saturation_fields()
    print("All G-ii checks passed.")
