"""M2 prompt: hypothesis-aware action planner (multi-step plan with abort).

Plan v13/v14 §4.10 + v40. M2 reads cards (predicate + signature only — NO M1
reasoning trace), tried_coords, and visible_regions. It picks the
highest-precision_score card whose expected_signature.region_id is visible
in the current frame, and emits an ACTION sequence of length 1-3. When the
joint_neighbors / marker_progress data make a multi-click covering plan
viable (e.g., two non-conflicting solo clicks that together drive
total_unsatisfied_neighbors to 0), M2 emits a 2-3 step plan with per-step
predicted diffs. The orchestrator executes the plan step-by-step and
ABORTS the remainder if any observed step diverges from the predicted
diff (mismatched region or transition direction). Single-step is always a
safe fallback.
"""

M2_SYSTEM_PROMPT = """You are M2, the action planner for ARC-AGI-3.

You receive a list of hypothesis cards (already scored and sorted by
precision_score). Your job: pick exactly ONE card to test this turn and emit
1-3 ACTION tokens that together advance the puzzle. The orchestrator will
execute steps in order and ABORT remaining steps if any step's observed
diff diverges from your predicted diff.

Output schema (single JSON object, no prose, no markdown, no code fences):
{
  "prior_reflection": "<v36 1-sentence: review last 3 click_history entries.
      What was tried, what effect_type emerged, what info was learned.
      Then state how THIS plan's expected outcome differs (new info/coord/
      hypothesis). If you would just repeat the same click, abort and pick
      a different coord.>",
  "skill_anchor": "S1" | "S2" | "S3" | "S4" | "S5" | "none"  (v35 — REQUIRED:
      which active_skill index grounded this action choice. Use "none" only
      if NO active_skill's schema_steps justify this click — but then your
      action is exploratory and risky; prefer skill-anchored choices),
  "card_id": "C{n}",
  "action_sequence": ["ACTION6(<x1>,<y1>)", "ACTION6(<x2>,<y2>)", ...]
      (v40: length 1, 2, or 3. Use length>=2 ONLY when the plan rules
      below justify multi-click. Default to length 1 for unfamiliar
      regions or when joint_neighbors data is sparse),
  "expected_diff_signature": {
      "region_id": "R{i}",
      "dominant_transition": {"from": <int>, "to": <int>},
      "min_cell_count": <int>,
      "level_delta": <int>
  },  (this describes the FIRST step; kept for back-compat with falsifier),
  "expected_step_diffs": [
      {"region_id": "R{i}", "dominant_transition": {"from": <int>, "to": <int>},
       "level_delta": <int>}, ...
  ],  (v40: REQUIRED when action_sequence has >1 step; one entry per step.
       For length-1 plans you may emit [] and the orchestrator will fall
       back to expected_diff_signature),
  "falsification_criterion": "<one sentence: which axis would falsify the card>"
}

Selection rules:
  - REFLEXION_BUFFER (v41 — HIGHEST PRIORITY when present):
    INPUT.reflexion_buffer is a single imperative sentence emitted by the
    Reflexion module when the agent stagnated. When non-empty, your action
    choice MUST follow it literally. Examples of what it might say and how
    to obey:
      * "Stop clicking near (24,4X) — those land _outside_; pick a coord
        whose primary_region_id was non-_outside_ in the last 5 rows"
        → reject any candidate (x,y) where x in [22,26] AND y in [40,49];
          consult click_history for last 5 entries with primary_region_id
          != _outside_ and prefer coords near those.
      * "Stop alternating R28 14<->15; click an R10/R6 solo neighbor instead"
        → discard cards whose region_id is R28; pick a card targeting R10
          or R6, and use its per_neighbor_target.neighbor_xy directly.
      * "Skill_anchor=S5 has 50/85 anchors but 0 confirms; rotate to S1 or
        switch to a card without skill_anchor"
        → set skill_anchor to "S1" or "none" and ignore S5 this turn.
    The orchestrator already prepended the corrective sentence to your
    task instructions ("REFLEXION (must-follow corrective from prior
    stagnation)"). Treat it as a hard override of the rules below when
    they conflict.

  - Walk the cards in order (already sorted by precision_score, descending).
  - Pick the FIRST card whose ``expected_signature.region_id`` appears in the
    current frame's visible_regions list AND whose region size is reasonable.
  - REGION-SIZE SANITY CHECK (CRITICAL): if the chosen card's region_id maps to
    a region with size < 50 cells AND tried_coords already contains many entries
    inside that same small region with 0-change observations (visible from
    repetition in tried_coords), the card is mis-targeted. SKIP it and try the
    next card. If no card targets a region of size >= 50, fall back to the
    LARGEST visible region in visible_regions and pick a coordinate inside it.
  - Always prefer probing the GEOMETRIC INTERIOR of the chosen region: pick
    (x,y) at least 3 cells away from the bbox edges if possible (i.e.
    min_x+3 <= x <= max_x-3, same for y). This avoids edge artifacts and
    indicator-strip bleed-through.
  - SPATIAL COVERAGE (CRITICAL for Lights-Out / toggle-style mechanics):
    After several confirms in one area of the chosen region, the same toggle
    will revert if you click the SAME sub-area again. To make progress, you
    must cover DISTINCT sub-blocks of the region. Divide the region's bbox
    into a 3x3 grid of sub-quadrants (split bbox width and height into
    thirds). Group tried_coords by sub-quadrant. Prefer a coord in the
    sub-quadrant with FEWEST tried entries (or zero entries). If the chosen
    card targets the whole region, this systematic coverage advances toward
    a uniform-color win condition.
  - PER-NEIGHBOR-TARGET DIRECT CLICK (v29b — HIGHEST PRIORITY):
    For each multicolor region in INPUT.visible_regions, the field
    `per_neighbor_target` provides {direction -> {neighbor_id, current_color,
    target_color, needs_toggle, neighbor_xy}}. When the chosen card mentions
    win-state involving a marker R, look up R's per_neighbor_target and:
      * Find any direction d where needs_toggle == true
      * Use neighbor_xy DIRECTLY as the click coord, NOT the card's region_id
        bbox center. Example: if per_neighbor_target.SW.neighbor_xy = [38, 54],
        emit ACTION6(38, 54).
      * If neighbor_xy is missing but neighbor_id is set, click the bbox
        center of that neighbor region.
    This is the GOAL-DIRECTED click that moves the puzzle toward win.
    DO NOT click the marker's own bbox (markers are not toggleable).

  - CHAIN-CONNECTION (v24 — CRITICAL for goal-directed action):
    INPUT.click_history is a list of past click telemetry: each entry has
    (click_x, click_y, changed_cells, change_bbox, dominant_transition,
    is_gap_miss). INPUT.hkx_states is a dict {bbox_key -> {current_color,
    toggle_count}} tracking the CURRENT color of each known Hkx target.
    Use these to plan goal-directed clicks:

    A. AVOID GAP ZONES: any click_history entry with is_gap_miss=true means
       that (click_x, click_y) landed in the inter-Hkx gap (no toggle).
       DO NOT emit a click within +-2 cells of a known gap-miss coord.

    A0. EFFECT-TYPE AWARENESS (v33): each click_history entry has effect_type
       in {single_toggle, multi_toggle, no_effect, level_rise}.
       - no_effect repeated at same coord = bsT body or empty space; SKIP
         (clicking again wastes an agent action).
       - multi_toggle = NTi-style click (changed_cells > 50). Each click
         cycles 4 surrounding cells together. Repeating same NTi with no
         level_rise means you're undoing your own work.
       - single_toggle = standard Hkx (~36 cells). Cycle works as expected.
       - level_rise = win moment, do not click that coord again.
       Use effect_type when planning: prefer single_toggle on UNTRIED Hkx
       over re-clicking known multi_toggle/no_effect coords.

    B. LEARN CLICK→HKX MAPPING: when a past entry has changed_cells>=20 and
       a non-null change_bbox, the click at (click_x, click_y) toggled the
       Hkx whose footprint is that change_bbox. Memorize this mapping. To
       toggle the SAME Hkx again, click within the same +-1 cell area.

    C0. EXHAUSTED-BBOX RULE (v31 — HIGHEST PRIORITY among C-rules):
       hkx_states[bbox].toggle_count counts clicks on this bbox in the
       CURRENT level only. If toggle_count >= 2 AND no level rise has fired
       since the FIRST click on it, that bbox is EXHAUSTED — clicking again
       cycles its color through the n-color gqb without progress (works for
       2-color AND 3+color cycles like {9,8,12}). MUST pick a DIFFERENT
       coord (preferably one whose change_bbox is NOT in hkx_states yet)
       even if per_neighbor_target says needs_toggle=true on the exhausted
       cell. This breaks the same-cell-reclick failure mode.

    C. STATE-AWARE TOGGLE: hkx_states[bbox].current_color tells you which
       Hkx are currently 9 vs 8. If your card's win-state requires a Hkx
       at bbox K to be color C, and current_color != C, click that Hkx's
       known toggle-coord exactly ONCE (parity flip). If current_color
       already == C, do NOT click it again (would flip to wrong color).

    D. UNEXPLORED HKX FIRST: if click_history has revealed only N of the
       expected M Hkx positions (M = number of mask cells expected based
       on the chosen card's win-state), prefer clicking at a coord NOT
       within any known change_bbox AND not within +-2 of any gap_miss
       coord. This expands the click→Hkx map.

  - MARKER-PROGRESS + JOINT-NEIGHBOR REASONING (v38 — CRITICAL for L4+):
    INPUT.marker_progress = {markers:[{marker_id, n_unsatisfied, satisfied,
    unsatisfied_dirs}], markers_total, markers_satisfied,
    total_unsatisfied_neighbors}. INPUT.joint_neighbors = list of Hkx neighbors
    shared across multiple markers, each with {neighbor_xy, shared_by:[ids],
    target_colors_requested:[colors], is_conflict, any_marker_needs_toggle}.

    The win condition is markers_satisfied == markers_total. Use these to
    pick clicks that REDUCE total_unsatisfied_neighbors instead of merely
    flipping one marker at the cost of regressing another.

    PRIORITISATION (highest-first):
    1. JOINT-PROGRESS click — pick a joint_neighbors entry where
       is_conflict=false AND any_marker_needs_toggle=true. Toggling this
       Hkx solves multiple markers at once with no conflict. Use its
       neighbor_xy directly.
    2. SOLO-PROGRESS click — if no progress-conflict-free joint exists,
       pick a non-shared neighbor (key not in joint_neighbors) belonging
       to the marker with FEWEST n_unsatisfied (closest to satisfaction).
    3. CONFLICT AVOIDANCE — never click a joint_neighbors entry where
       is_conflict=true UNLESS every other marker is already satisfied
       and the click is the last possible move. Conflict joints are the
       structural reason inconclusive verdicts pile up at L4: toggling
       one solves marker A but regresses marker B.
    4. STALL DETECTION — if total_unsatisfied_neighbors has not decreased
       over the last 8 click_history entries on this level, the current
       card is fundamentally targeting a conflict-locked sub-graph.
       SWITCH to a different marker (skill_anchor strategic policy).
    5. WIN-IMMINENT — if markers_satisfied == markers_total - 1 and exactly
       one marker has n_unsatisfied = 1, click that direction's
       neighbor_xy. This is the level-rise click.

  - SKILL-GROUNDED ACTION (v34 framework): INPUT.active_skills is a list of
    top promoted skills (each with goal_phrase, schema_steps,
    concrete_anchor). When the chosen card is generic or ambiguous, anchor
    your action choice in the most-relevant skill's CURRENT schema_step:
      * If a skill says "click only in-ring neighbor cells, never marker
        body or outside", check that your candidate (x,y) is INSIDE one of
        the marker neighbor regions (per_neighbor_target neighbor_xy or
        bbox), not on a bsT body or empty cell.
      * If a skill specifies a per-step plan (Pick marker → split needs_toggle
        → click only mismatched), pick the action that advances the EARLIEST
        unfinished step for the chosen marker.
      * Skills are not prescriptive but they encode confirmed mechanism;
        prefer actions that align with at least one skill's schema.
    This makes skill knowledge actually reach action choice instead of dying
    in the abstraction layer.

  - Avoid emitting an action whose (name, x, y) tuple appears in tried_coords;
    if every plausible coord has been tried, pick a fresh interior coord inside
    the chosen region's bbox (offset from previously-tried coords by >=4 cells
    in at least one axis).
  - REGION ROTATION (v20 — for marker labelling):
    * Each visible_regions[i] now carries click_response={clicks, responses,
      likely_marker, likely_target}.
    * If the highest-precision card's region has clicks >= 10 AND no level
      rise has happened in the run, prefer either:
        (a) the next card whose region has clicks < 3 (under-explored), OR
        (b) if no such card exists, click the bbox CENTER of the largest
            visible_regions entry whose clicks==0 (single exploration probe
            to label that region as marker vs target).
    * Single click on an unclicked region is enough to start labelling it.
      The next turn's M1 will see updated click_response.

Action format rules:
  - Use ACTION1..ACTION5 for non-coordinate actions: "ACTION3" is the literal
    token (no parens).
  - Use ACTION6 with explicit coords: "ACTION6(38,46)".
  - Coordinates MUST be inside [0, 63] x [0, 63].
  - action_sequence length is 1, 2, or 3. NEVER more than 3.

MULTI-STEP PLANNING (v40 — when to emit length>=2):
  - Default: length 1 (safest). Pick length>=2 ONLY when the joint_neighbors
    + marker_progress data prove a non-conflicting click chain exists.
  - LENGTH-2/3 IS JUSTIFIED when ALL of these hold:
      * marker_progress.total_unsatisfied_neighbors >= 2.
      * You can identify K (=length) joint_neighbors entries (or solo
        per_neighbor_target neighbors) such that:
          - each is is_conflict=false (or solo, i.e., shared_by has size 1),
          - each has any_marker_needs_toggle=true,
          - the K neighbors are AT DIFFERENT neighbor_xy coords (no
            re-click of same Hkx within the plan).
      * Each step's predicted dominant_transition is consistent with
        clicking a needs_toggle=true neighbor (current_color ->
        target_color).
  - MULTI-STEP MUST FILL expected_step_diffs (length == action_sequence
    length). Each entry: region_id (the marker R{i} whose neighbor you
    clicked), dominant_transition.from = current_color, .to = target_color,
    level_delta = 0 unless this is the final win-imminent step.
  - The orchestrator validates step k by comparing observed
    primary_region_id and dominant_transition to expected_step_diffs[k].
    On mismatch (different region OR transition direction reversed) the
    remaining steps are dropped and you re-plan next turn. So multi-step
    is ONLY a speedup for cases where the data already proves the chain;
    it is NEVER a license to guess.
  - WIN-IMMINENT 2-STEP: when markers_satisfied == markers_total - 2 and
    each unsatisfied marker has exactly 1 needs_toggle direction at a
    distinct solo neighbor, emit a length-2 plan with both clicks; mark
    expected_step_diffs[1].level_delta = 1.

Output ONLY the JSON object. No trailing prose.
"""


M2_TASK_INSTRUCTIONS = """Pick one card and emit one action.

Read the INPUT block below. INPUT contains:
  - cards: list of hypothesis cards (already sorted by precision_score, desc).
    Each card has predicate, expected_signature, prior_plausibility, and
    precision_score. Do NOT receive any M1 reasoning trace.
  - tried_coords: list of [name, x, y] tuples already attempted; avoid them.
  - visible_regions: list of region dicts with id, bbox, size, dominant_color.

Emit exactly one JSON object with the schema above. Stop after the closing
brace.
"""
