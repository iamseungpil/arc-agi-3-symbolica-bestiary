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
    "Step 1. Propose a predicate that targets the most-saturated marker's\n"
    "remaining unclicked compass slot, OR a slot adjacent to it.\n\n"
    "Output schema (strict JSON):\n"
    "  candidate_predicate_id: str (e.g., 'P_saturation_progress')\n"
    "  region_hint: str (must be a region_id present in `visible_regions`)\n"
    "  expected_signature: dict (e.g., {level_delta: 1})\n"
    "  required_pre_state:\n"
    "    marker_id: str (must be a marker_id present in marker_neighbor_states)\n"
    "    saturation_threshold: int\n"
    "    saturation_denominator: int\n"
    "  confidence: float in [0, 1]\n"
    "  thought: str (must contain the saturation Step-0 expression above)\n"
    "Forbidden: tool_calls, submit_action, free-form prose outside `thought`."
)


def _alias_marker(idx: int) -> str:
    return f"marker_{idx}"


def _alias_region_primary() -> str:
    return "region_primary"


def _alias_slot(idx: int) -> str:
    return f"slot_{idx}"


def _summarize_marker(marker: dict, alias: str) -> dict:
    """Render a marker into prompt-safe form using neutral aliases."""
    compass = marker.get("compass") or {}
    saturated = sum(1 for c in compass.values() if (c or {}).get("clicks", 0) >= 1)
    denom = len(compass)
    return {
        "alias": alias,
        "is_primary_marker": bool(marker.get("is_primary_marker", False)),
        "compass_saturation_numerator": saturated,
        "compass_denominator": denom,
        "compass_unclicked_slot_aliases": [
            _alias_slot(i)
            for i, slot in enumerate(compass.values())
            if (slot or {}).get("clicks", 0) == 0
        ],
    }


def render_user_prompt(state: dict[str, Any]) -> str:
    """Render the user-facing prompt body from structured state.

    The output is game-agnostic prose: no raw region_id values appear in the
    prompt prose; only neutral aliases (marker_0, region_primary, slot_N).
    The agent's structured input is passed via the structured fields below
    (which the LLM client serializes separately into JSON content).
    """
    markers = state.get("marker_neighbor_states") or []
    obs = state.get("observation") or {}
    summarized: list[dict] = []
    for i, m in enumerate(markers):
        summarized.append(_summarize_marker(m, _alias_marker(i)))
    primary_alias = _alias_region_primary() if obs.get("primary_region_id") else None
    dt = obs.get("dominant_transition") or {}
    dt_summary = (
        f"dominant_transition.count = {int(dt.get('count', 0))}"
        if dt else "dominant_transition = (none)"
    )
    lines: list[str] = []
    lines.append("State summary:")
    if primary_alias is not None:
        lines.append(f"  primary region alias: {primary_alias}")
    lines.append(f"  observation: {dt_summary}; level_delta = {int(obs.get('level_delta') or 0)}")
    lines.append(f"  number of markers visible: {len(summarized)}")
    for ms in summarized:
        lines.append(
            f"  {ms['alias']}: is_primary_marker={ms['is_primary_marker']}; "
            f"compass clicks {ms['compass_saturation_numerator']}/{ms['compass_denominator']} ; "
            f"unclicked slot aliases: {ms['compass_unclicked_slot_aliases']}"
        )
    lines.append("")
    lines.append("Reminder: cite the Step-0 saturation expression in your thought.")
    return "\n".join(lines)


def build_messages(state: dict[str, Any]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_user_prompt(state)},
    ]
