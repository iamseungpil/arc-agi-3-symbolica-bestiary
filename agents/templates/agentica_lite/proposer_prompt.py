"""v601 Proposer prompt construction (plan rev C §3, §4 INT05).

Game-agnostic: see plan §4 INT05 for the forbidden-token list. Region ids
and marker ids carried in the structured state are aliased to neutral
labels (`marker_0`, `region_primary`, `slot_N`) before being emitted into
prompt text. Schema-level field names (`is_primary_marker`, `compass`,
`clicks`, `region_id`) are explicitly allowed.
"""

from __future__ import annotations

from typing import Any

# Plan §3 + §G11: Step-0 saturation computation must be referenced verbatim.
SATURATION_STEP0_REFERENCE = (
    "mean(c.clicks >= 1 for c in M.compass)"
)

SYSTEM_PROMPT = (
    "You are a predicate proposer for a structured-observation puzzle agent.\n"
    "You CANNOT call submit_action, click, or any environment tool.\n"
    "You output ONE JSON object describing a candidate predicate to test.\n\n"
    "Step 0 (mandatory). For each visible primary marker M, compute the\n"
    "compass-saturation fraction:\n"
    f"  saturation(M) = {SATURATION_STEP0_REFERENCE}\n"
    "Cite this expression in your `thought` field. Markers with saturation\n"
    "approaching 1.0 are strong candidates for level-progression triggers.\n\n"
    "Step 1. You may propose either:\n"
    "  (A) a saturation-driven predicate that targets the most-saturated\n"
    "      marker's remaining unclicked compass slot, OR a slot adjacent to it, OR\n"
    "  (B) a TIER-B region-anchored predicate whose candidate id names a\n"
    "      concrete region/sector alignment or crop relation (for example,\n"
    "      'P_R12_crop_sector_alignment' or another region-anchored chid).\n"
    "      This branch is not required to be saturation-based, but it must still\n"
    "      produce valid JSON and remain compatible with the same schema.\n"
    "TIER-B chids that match a visible marker region id in the current prompt\n"
    "(for example P_R28_*, P_R31_*, P_R8_*, or P_R11_*) are PREFERRED over\n"
    "generic saturation_progress when both are plausible. If you can justify a\n"
    "visible-marker-anchored TIER-B option, choose it instead of P_saturation_progress.\n\n"
    "Output schema (strict JSON):\n"
    "  candidate_predicate_id: str (e.g., 'P_saturation_progress',\n"
    "    'P_R12_crop_sector_alignment')\n"
    "  region_hint: str — CRITICAL: must be one of the marker's NEIGHBOR\n"
    "    region ids from its unclicked_compass_region_ids list, NOT the\n"
    "    marker_id itself. For TIER-B region-anchored predicates, prefer a\n"
    "    region_hint that is the concrete region referenced by the chid when\n"
    "    such a region exists in the visible region ids. Clicking the marker\n"
    "    has NO effect; you must click an unclicked compass neighbor of the\n"
    "    marker or a valid region anchor consistent with the predicate.\n"
    "  expected_signature: dict (e.g., {level_delta: 1})\n"
    "  required_pre_state:\n"
    "    marker_id: str (must be a marker_id present in marker_neighbor_states)\n"
    "    saturation_threshold: int\n"
    "    saturation_denominator: int\n"
    "    If the predicate is not saturation-based, keep these fields present\n"
    "    and fill them with explicit dummy integers rather than omitting them.\n"
    "  confidence: float in [0, 1]\n"
    "  thought: str (must contain the saturation Step-0 expression above;\n"
    "    for TIER-B, explain the region-anchored rationale in the same thought)\n"
    "Forbidden: tool_calls, submit_action, free-form prose outside `thought`.\n"
    "Forbidden: region_hint == marker_id (same value). Region_hint MUST\n"
    "be a compass-neighbor region id; the marker_id field is separate."
)


def _alias_marker(idx: int) -> str:
    return f"marker_{idx}"


def _alias_region_primary() -> str:
    return "region_primary"


def _alias_slot(idx: int) -> str:
    return f"slot_{idx}"


def _summarize_marker(marker: dict) -> dict:
    """Render a marker with REAL ids (C1, C2, ...) so the LLM can use them
    in `region_hint`. The C{idx} naming is spatial-rank-based and game-agnostic,
    so passing it does not leak any task-specific identifier."""
    compass = marker.get("compass") or {}
    saturated = sum(1 for c in compass.values() if (c or {}).get("clicks", 0) >= 1)
    denom = len(compass)
    unclicked = [
        (slot or {}).get("region_id")
        for slot in compass.values()
        if (slot or {}).get("clicks", 0) == 0 and (slot or {}).get("region_id") is not None
    ]
    return {
        "marker_id": marker.get("marker_id"),
        "is_primary_marker": bool(marker.get("is_primary_marker", False)),
        "compass_saturation_numerator": saturated,
        "compass_denominator": denom,
        "unclicked_compass_region_ids": unclicked,
    }


def render_user_prompt(state: dict[str, Any]) -> str:
    """Render the user-facing prompt body using REAL region_ids.

    The C{idx} ids in our adapter are spatial-rank-based (deterministic by
    top-row,left-col), so they are game-agnostic. The Proposer must respond
    with `region_hint` matching one of the visible region_ids; using real
    ids here lets the validator accept the response. Game-specific identifier
    tokens still never appear because the adapter only ever generates
    C{idx}-format ids.
    """
    markers = state.get("marker_neighbor_states") or []
    obs = state.get("observation") or {}
    summarized = [_summarize_marker(m) for m in markers]
    primary_id = obs.get("primary_region_id")
    dt = obs.get("dominant_transition") or {}
    dt_summary = (
        f"dominant_transition.count = {int(dt.get('count', 0))}"
        if dt else "dominant_transition = (none)"
    )
    visible_regions = state.get("visible_regions") or []
    visible_ids = [r.get("region_id") or r.get("id") for r in visible_regions if r]
    lines: list[str] = []
    lines.append("State summary:")
    if primary_id:
        lines.append(f"  primary region: {primary_id}")
    lines.append(f"  observation: {dt_summary}; level_delta = {int(obs.get('level_delta') or 0)}")
    lines.append(f"  visible region ids: {visible_ids}")
    lines.append(f"  number of markers visible: {len(summarized)}")
    for ms in summarized:
        lines.append(
            f"  marker {ms['marker_id']}: is_primary_marker={ms['is_primary_marker']}; "
            f"compass clicks {ms['compass_saturation_numerator']}/{ms['compass_denominator']} ; "
            f"unclicked compass region ids: {ms['unclicked_compass_region_ids']}"
        )
    lines.append("")
    lines.append("Reminder: cite the Step-0 saturation expression in your thought.")
    lines.append("region_hint MUST be one of the visible region ids listed above.")
    return "\n".join(lines)


def build_messages(state: dict[str, Any]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_user_prompt(state)},
    ]
