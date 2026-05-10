# FT09 Visualization Review

- Generated at: 2026-04-29 06:51 UTC
- Default run: `v50_cycle50_20260429_035615`
- Default turn: `55`
- Acceptance bar:
  recorded vs derived vs missing must be visually explicit
  level progress and reflexion events must be scannable within 5 seconds
  the first screen must show action, state change, and selected predicate together

## v50 update

- Added `v50_cycle50` bundle to the manifest (highest-version v5x_cycle is now always pinned in the selection so newest cycles ship even when older bundles score higher on level/coverage).
- Added per-turn **Reasoning panel** with collapsible M1/M2/M3/M4 sub-sections (left-border colors blue/teal/purple/orange respectively).
  - M1 hypothesis: predicate, abstract_recipe, evidence_quote, precision_score, prior_plausibility, retention status from `active_cards` / `falsified_cards`.
  - M2 action commitment: `expected_outcome_rationale`, `skill_anchor`, `expected_step_diffs`, anchored skill goal_phrase + applies_when when present, and `prior_reflection` when persisted.
  - M3 skill emission: shows new skill content for turns where `skill_emission_turns` records a promotion.
  - M4 evaluation: per-turn `m4_history` entry when persisted, otherwise the run-final `last_turn_summary` is used on the LAST turn and earlier turns are marked `(not persisted in v48 - see Track B)`.
- Added top-level **Skill library** panel listing every promoted skill across all bundles, sorted by emission turn when known.
- Track B persistence patches add `prior_reflection` to ChosenAction, `m4_history` + `record_m4_turn` on GoalBoard, and `skill_emission_turns` tagging in `add_skill`. Future cycles will populate the new fields directly.

## Included Runs
- `v50_cycle50_20260429_035615`: max 3/6, 108 turns, 59% rationale coverage
- `v40_cycle33_20260428_104943`: max 4/6, 88 turns, 73% rationale coverage
- `v41c_cycle40_20260428_185407`: max 3/6, 83 turns, 77% rationale coverage
- `v39_cycle32_20260428_072959`: max 4/6, 84 turns, 76% rationale coverage
- `v38_cycle31_20260428_053226`: max 4/6, 94 turns, 68% rationale coverage
- `v41c_cycle38_20260428_165304`: max 3/6, 41 turns, 100% rationale coverage
