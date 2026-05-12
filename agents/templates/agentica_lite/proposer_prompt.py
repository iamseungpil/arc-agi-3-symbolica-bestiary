"""v601 Proposer prompt construction (plan rev C §3, §4 INT05).

Game-agnostic: see plan §4 INT05 for the forbidden-token list. Region ids
and marker ids carried in the structured state are aliased to neutral
labels (`marker_0`, `region_primary`, `slot_N`) before being emitted into
prompt text. Schema-level field names (`is_primary_marker`, `compass`,
`clicks`, `region_id`) are explicitly allowed.
"""

from __future__ import annotations

from typing import Any

# Legacy v601 saturation expression retained for migration tests. v608 prompts
# use the constraint expression below as the primary Step 0.
SATURATION_STEP0_REFERENCE = (
    "mean(c.clicks >= 1 for c in M.compass)"
)
CONSTRAINT_STEP0_REFERENCE = (
    "for each marker slot, relation = same if marker slot value is zero-like "
    "else different; satisfied = neighbor_color relation marker_color"
)

SYSTEM_PROMPT = (
    "You are a predicate proposer for a structured-observation puzzle agent.\n"
    "You CANNOT call submit_action, click, or any environment tool.\n"
    "You output ONE JSON object describing a candidate predicate to test.\n\n"
    "Step 0 (mandatory). Build local marker-constraint cards:\n"
    f"  {CONSTRAINT_STEP0_REFERENCE}\n"
    "Cite this constraint expression in your `thought` field. Prefer actions\n"
    "that reduce the count of unsatisfied marker constraints. Treat raw\n"
    f"saturation ({SATURATION_STEP0_REFERENCE}) only as auxiliary history.\n\n"
    "Step 1. You may propose either:\n"
    "  (A) a constraint-repair predicate that targets an unsatisfied marker\n"
    "      neighbor slot, OR\n"
    "  (B) a region-anchored predicate whose candidate id names a NEW\n"
    "      <verb>_<noun> rationale invented for this turn's evidence.\n"
    "      INVENT the verb_noun pair distinct from prior turns; do NOT reuse\n"
    "      generic terms like 'check'/'select'/'click'/'progress'.\n"
    "Region-anchored predicates anchored to one of the visible marker ids in\n"
    "marker_neighbor_states (form: P_<visible_marker_id>_<verb>_<noun>, e.g.\n"
    "P_<marker_id>_<your_invented_rationale>) are PREFERRED over generic\n"
    "constraint_repair when both are plausible. Substitute <visible_marker_id>\n"
    "with an actual marker_id from the user prompt's marker list.\n"
    "If skill template hints are present in the user prompt (DISCOVERED skill\n"
    "templates with EV scores), prefer instantiating one of those over\n"
    "inventing a new family.\n\n"
    "Output schema (strict JSON):\n"
    "  candidate_predicate_id: str — your invented or top-EV predicate id\n"
    "  region_hint: str — CRITICAL: must be one of the marker's NEIGHBOR\n"
    "    region ids from its unclicked_neighbor_region_ids list, NOT the\n"
    "    marker_id itself. For region-anchored predicates, prefer a\n"
    "    region_hint that is the concrete region referenced by the chid when\n"
    "    such a region exists in the visible region ids. Clicking the marker\n"
    "    has NO effect; you must click an unclicked neighbor of the\n"
    "    marker or a valid region anchor consistent with the predicate.\n"
    "  expected_signature: dict (e.g., {level_delta: 1})\n"
    "  required_pre_state:\n"
    "    marker_id: str (must be a marker_id present in marker_neighbor_states)\n"
    "    saturation_threshold: int (legacy; fill with 0 if unused)\n"
    "    saturation_denominator: int (legacy; fill with 0 if unused)\n"
    "    If the predicate is not saturation-based, keep these fields present\n"
    "    and fill them with explicit dummy integers rather than omitting them.\n"
    "  confidence: float in [0, 1]\n"
    "  thought: str (must contain the constraint Step-0 expression above;\n"
    "    explain the region-anchored rationale in the same thought)\n"
    "Forbidden: tool_calls, submit_action, free-form prose outside `thought`.\n"
    "Forbidden: region_hint == marker_id (same value). Region_hint MUST\n"
    "be a NEIGHBOR region id; the marker_id field is separate."
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
    constraints = state.get("marker_constraints") or []
    constraint_summary = state.get("marker_constraint_summary") or {}
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
    if constraints:
        lines.append(
            "  marker constraints: "
            f"unsatisfied={int(constraint_summary.get('unsatisfied', 0) or 0)} "
            f"/ total={int(constraint_summary.get('total', len(constraints)) or 0)}"
        )
        for c in constraints[:12]:
            lines.append(
                "    "
                f"marker={c.get('marker_id')} slot={c.get('slot')} "
                f"neighbor={c.get('neighbor_region_id')} "
                f"relation={c.get('relation')} satisfied={bool(c.get('satisfied'))}"
            )
    lines.append("")
    # codex r20 option B: rolling click+observation history for multi-step
    # planning (cycle237 evidence: L+1 requires 7-step sequence, not 1 click)
    rtd = state.get("recent_turn_diffs") or []
    if rtd:
        lines.append("Recent click history (most recent last; use this to plan multi-step):")
        for d in rtd:
            t = d.get("turn_offset")
            coord = d.get("coord")
            r = d.get("click_region_id")
            ld = d.get("level_delta", 0)
            adv = d.get("did_advance", False)
            dt_d = d.get("dominant_transition") or {}
            from_c, to_c, cnt = dt_d.get("from"), dt_d.get("to"), dt_d.get("count", 0)
            trans_s = f"trans={from_c}->{to_c}(n={cnt})" if from_c is not None else "trans=none"
            advance_s = " [LEVEL_ADVANCED!]" if adv else ""
            lines.append(
                f"  T{t}: click={coord} region={r} {trans_s} level_delta={ld}{advance_s}"
            )
        lines.append("")
    # v608d Phase 2 / v608e: per-region color-cycle world model. The cycle
    # detector can infer a period after 3 samples, and cycle401 showed the
    # stricter n>=4 gate injected zero world-model slices in live play.
    rt = (state.get("region_transition_cache")
          or state.get("region_transitions")
          or {})
    confirmed = [
        (rid, view) for rid, view in rt.items()
        if isinstance(view, dict)
        and view.get("inferred_cycle")
        and int(view.get("n_samples", 0) or 0) >= 3
    ]
    if confirmed:
        confirmed.sort(key=lambda kv: -float(kv[1].get("confidence", 0.0)))
        lines.append(
            "Region click-cycle local transition cache (cycle inferred from "
            "your own click history; use this to predict the next observation):"
        )
        for rid, view in confirmed[:8]:
            cycle = view.get("inferred_cycle")
            nxt = view.get("next_predicted")
            conf = float(view.get("confidence", 0.0) or 0.0)
            ns = int(view.get("n_samples", 0) or 0)
            lines.append(
                f"  {rid}: cycle={cycle} next_predicted={nxt} "
                f"confidence={conf:.2f} samples={ns}"
            )
        lines.append("")
    # v607 Phase 8: dynamic chid_template injection from skill_state top-k posterior.
    # Replaces v606.x hardcoded `P_<m>_crop_sector_alignment` (which caused 60/60
    # monomania AND leaked cycle237 vocab compass/sector/sweep/crop/alignment).
    marker_ids_visible = [ms["marker_id"] for ms in summarized if ms.get("marker_id")]
    chid_hints = state.get("v607_chid_template_hints") or []
    if chid_hints and marker_ids_visible:
        formatted: list[str] = []
        for hint in chid_hints[:3]:
            if isinstance(hint, tuple):
                tmpl = hint[0]
                ev = hint[1] if len(hint) > 1 else None
            else:
                tmpl, ev = hint, None
            inst = tmpl
            if "{m}" in inst:
                inst = inst.replace("{m}", str(marker_ids_visible[0]))
            elif "{marker_id}" in inst:
                inst = inst.replace("{marker_id}", str(marker_ids_visible[0]))
            tag = f" (EV={ev:.2f})" if isinstance(ev, (int, float)) else ""
            formatted.append(f"'{inst}'{tag}")
        lines.append(
            "DISCOVERED skill templates (top-k by Bayesian EV, pick one if it fits "
            "this turn's evidence, else INVENT a new <verb>_<noun> family): "
            + ", ".join(formatted)
        )
    elif marker_ids_visible:
        # Cold-start (no Reflector emissions yet): anonymized invention prompt
        # (no leak vocab; LLM must invent a verb_noun pair for this turn).
        lines.append(
            "Predicate template format: P_<verb>_<noun>_C{marker_id}. "
            "Invent a NEW verb_noun pair distinct from prior turns; do NOT "
            "use generic terms like 'check'/'select'/'click'."
        )
    lines.append("Reminder: cite the Step-0 marker-constraint expression in your thought.")
    lines.append("region_hint MUST be one of the visible region ids listed above.")
    lines.append(
        "If a region click-cycle world model entry is shown above, your "
        "thought MUST compare the previous turn's predicted next color to "
        "this turn's observed color when you click that region."
    )
    lines.append(
        "If recent click history shows no level_delta advance after several "
        "tries on the same region, try a DIFFERENT region this turn."
    )
    return "\n".join(lines)


def build_messages(state: dict[str, Any]) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_user_prompt(state)},
    ]
