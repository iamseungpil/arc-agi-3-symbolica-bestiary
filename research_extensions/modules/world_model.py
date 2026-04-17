"""World-model module.

Philosophy (CWM, arxiv 2510.02387 adapted):
- The agent builds its own mental model of the environment directly.
- We do not score against hidden progress markers. We only check whether
  predicted-vs-observed grid changes match.
- Any mismatch is a surprise and goes onto the shared bridge. Other modules
  (DreamCoder, Meta-Harness) can react to those surprises without needing to
  know the details of how we predicted.

This module can run alone or combined with the other two.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Any

from ..bridge import Observation, Prediction, SharedBridge, SurpriseEvent
from ..grid_utils import current_grid, encode_row, grid_diff_magnitude, grid_feature_vector, grid_signature


@dataclass(slots=True)
class WorldDraft:
    body: str
    focus: str = ""
    self_score: float = 0.0
    empirical_matches: int = 0
    empirical_mismatches: int = 0
    simulator_matches: int = 0
    simulator_mismatches: int = 0
    revisions: int = 1
    # CWM unit-test refinement bookkeeping. Each replay against the
    # accumulated transitions produces a transition_accuracy in [0, 1];
    # we cache the latest value so the agent can decide whether to
    # refine further (target = 1.0, per Lehrach et al. 2025).
    unit_tests_seen: int = 0
    unit_tests_passed: int = 0
    transition_accuracy: float = 0.0
    last_unit_test_eval_at: float = 0.0
    created_at: float = field(default_factory=time.time)
    last_updated_at: float = field(default_factory=time.time)

    def score(self) -> float:
        # v3.11 scoring (after user feedback: "failures must penalize"):
        # - empirical mismatch penalty doubled (1.0 → 2.0)
        # - simulator mismatch doubled (1.0 → 2.0)
        # - no-prediction turns get a small drift penalty so silence ≠ free
        #   (tracked by revisions count vs total_unit_tests)
        silence_penalty = 0.0
        if self.unit_tests_seen > 0 and (
            self.empirical_matches + self.empirical_mismatches
        ) < self.unit_tests_seen * 0.5:
            # If the draft wasn't scored on ≥50% of tests, treat as
            # avoidance.
            silence_penalty = 0.5 * (
                self.unit_tests_seen
                - (self.empirical_matches + self.empirical_mismatches)
            )
        return (
            self.self_score
            + 1.5 * self.empirical_matches
            - 2.0 * self.empirical_mismatches
            + 1.75 * self.simulator_matches
            - 2.0 * self.simulator_mismatches
            + 0.25 * self.revisions
            + 2.0 * self.transition_accuracy
            - silence_penalty
        )


class WorldModelModule:
    name = "world_model"

    def __init__(self, params: dict[str, Any], context, bridge: SharedBridge) -> None:
        self.params = params
        self.context = context
        self.bridge = bridge
        self.transitions: dict[str, dict[str, dict[str, float]]] = {}
        self.signature_features: dict[str, list[float]] = {}
        self.agent_env_notes: list[str] = []
        self.world_drafts: list[WorldDraft] = []
        self.prior_suggestions: int = 0
        self.prior_overrides: int = 0
        self.structured_predictions: int = 0
        self.scored_prediction_matches: int = 0
        self.scored_prediction_mismatches: int = 0
        self.recent_prior_decisions: list[dict[str, Any]] = []
        self.recent_outcomes: list[tuple[str, float]] = []
        # CWM unit-test buffer: each entry is one observed transition
        # (before_signature, action, after_signature, diff_band,
        # new_family, branch_escape). The simulator draft is "compiled"
        # against this buffer to compute transition_accuracy.
        self.unit_tests: list[dict[str, Any]] = []
        self.unit_test_capacity = int(params.get("unit_test_capacity", 200))
        self.unit_test_eval_window = int(params.get("unit_test_eval_window", 64))
        self._pending_prediction_was_structured: bool = False
        self._pending_posthoc_score: bool | None = None
        self._latest_action_hint: str | None = None
        self._memories: Any | None = None
        self._state_path = context.shared_dir / "world_model.json"
        if self._state_path.exists():
            try:
                raw = json.loads(self._state_path.read_text(encoding="utf-8"))
                self.transitions = raw.get("transitions", {})
                self.signature_features = raw.get("signature_features", {})
                self.agent_env_notes = raw.get("agent_env_notes", [])
                self.world_drafts = [
                    WorldDraft(**item) for item in raw.get("world_drafts", [])
                ]
                self.prior_suggestions = int(raw.get("prior_suggestions", 0))
                self.prior_overrides = int(raw.get("prior_overrides", 0))
                self.structured_predictions = int(raw.get("structured_predictions", 0))
                self.scored_prediction_matches = int(
                    raw.get("scored_prediction_matches", 0)
                )
                self.scored_prediction_mismatches = int(
                    raw.get("scored_prediction_mismatches", 0)
                )
                self.recent_prior_decisions = list(raw.get("recent_prior_decisions", []))
                self.recent_outcomes = [
                    (str(item[0]), float(item[1]))
                    for item in raw.get("recent_outcomes", [])
                    if isinstance(item, (list, tuple)) and len(item) == 2
                ]
                self.unit_tests = [
                    item
                    for item in raw.get("unit_tests", [])
                    if isinstance(item, dict)
                ][-self.unit_test_capacity :]
            except Exception:
                pass
        self._publish_world_context()

    # -- Hook surface -----------------------------------------------------
    def prompt_overlay(self) -> str:
        """Ask the agent to think in terms of 'what do I believe about this
        environment' rather than any hidden marker."""
        parts: list[str] = [
            "[World Model]",
            "Before acting, state a scorable prediction in the `predict` field.",
            "Prefix it with either `EXPECT_CHANGE:` or `EXPECT_NO_CHANGE:` and "
            "then add a short note. After each action, compare the actual grid "
            "change to that prediction.",
            "For stronger predictions, prefer a dict with `observation_prediction`, "
            "`progress_prediction`, `action_recommendation`, and optional `rival_predictions`.",
            "If you are surprised, that is valuable — it means your model of the game is incomplete.",
            "Keep a compact local rule draft in `world_update`. Prefer markdown or a short fenced code/pseudocode block.",
            "After the first informative action, always emit or revise `world_update` on each turn. A tiny stub is better than no draft.",
            "Do not rewrite the full model every turn. Revise the current draft, score its confidence, then test it with the next action.",
            "Markdown is preferred. A good pattern is: `score: 2.5`, `focus: ...`, then a short rule sketch.",
            "Prefer a tiny transition function or executable-looking pseudocode over prose-only summaries.",
            "REAL WORLD MODEL contract (CWM state transition, v3.12): "
            "`def predict_effect(action, observation) -> dict` receives `observation['grid']` "
            "(FULL 2D list, typically 64x64). Compute the NEXT grid from scratch and return it. "
            "**MANDATORY fields**: `expect_change: True|False`, `expected_diff_band: 'zero'|'small'|'large'`, "
            "`expected_diff_cells: int`, `expected_next_grid: List[List[int]]` (full next grid ≥95% accurate). "
            "Delta-only predictions now FAIL — the scorer requires the whole board. "
            "Strict scoring: `'unknown'` or `None` = WRONG. "
            "Template (DATA-DRIVEN, adapt to your hypothesis):\n"
            "```python\n"
            "def predict_effect(action, observation):\n"
            "    g = observation['grid']\n"
            "    H, W = len(g), len(g[0])\n"
            "    new = [row[:] for row in g]\n"
            "    # Use empirical delta hints — avoids guessing 1-cell predictions.\n"
            "    hints = observation.get('per_action_delta_hints', {}).get(action, {})\n"
            "    samples = hints.get('sample_changes', [])  # [[x,y,v], ...]\n"
            "    # HYPOTHESIS: replay the most common past change locations.\n"
            "    # Replace this with your inferred rule once you see a pattern.\n"
            "    if samples:\n"
            "        for change_list in samples[:1]:  # use first sample\n"
            "            for x, y, new_val in change_list:\n"
            "                if 0 <= y < H and 0 <= x < W:\n"
            "                    new[y][x] = new_val\n"
            "    n_changed = sum(1 for y in range(H) for x in range(W) if g[y][x] != new[y][x])\n"
            "    return {\n"
            "        'expect_change': n_changed > 0,\n"
            "        'expected_diff_band': 'zero' if n_changed==0 else ('small' if n_changed<=4 else 'large'),\n"
            "        'expected_diff_cells': n_changed,\n"
            "        'expected_next_grid': new,\n"
            "    }\n"
            "```\n"
            "`per_action_delta_hints[action]` now contains `{n, mean_diff_cells, sample_changes}` "
            "from the last 32 real transitions. Use them instead of guessing. "
            "ANTI-CHEAT: returning `expected_next_grid == observation['grid']` (identity) "
            "while diff_band != 'zero' now scores as WRONG. The scorer compares against "
            "the real next grid — a do-nothing prediction is no longer safe.",
            "The `observation` object reflects the real interface: signature, visible grid rows, feature vector, available actions, recent actions, and recent outcomes.",
            "When uncertain, keep a best hypothesis and at least one rival. Prefer actions that seek an unseen observation or falsify the current best hypothesis.",
        ]
        if self.agent_env_notes:
            parts.append("Your running environment notes:")
            for note in self.agent_env_notes[-4:]:
                parts.append(f"- {note}")
        skill_hint = self.bridge.read_hint("dreamcoder")
        if isinstance(skill_hint, dict):
            top_skills = skill_hint.get("top_skills") or []
            if top_skills:
                parts.append("Top reusable skills currently in library:")
                for skill in top_skills[:3]:
                    name = str(skill.get("name", "")).strip()
                    preview = str(skill.get("preview", "")).strip()
                    if name:
                        parts.append(f"- {name}: {preview[:180]}")
                parts.append(
                    "When revising the world draft, explain why these reusable skill fragments work in terms of observation -> action -> observed effect."
                )
        if not self.world_drafts:
            parts.append(
                "No world draft exists yet. After the next informative change, write a tiny "
                "local transition rule in `world_update` as code or pseudocode."
            )
        else:
            parts.append(
                "A draft already exists, so revise it rather than only narrating observations. "
                "Keep the update compact and executable-looking."
            )
        if self.structured_predictions == 0:
            parts.append(
                "You have not made a scored prediction yet. Do that before your next action so the model can be tested."
            )
        best = self._best_draft()
        if best is not None:
            parts.append(
                f"Current best draft (score={best.score():.1f}, self={best.self_score:.1f}, matches={best.empirical_matches}, mismatches={best.empirical_mismatches}):"
            )
            parts.append(best.body[:500])
            if "```" not in best.body and not any(
                token in best.body for token in ("if ", "return ", "def ", "for ", "while ")
            ):
                parts.append(
                    "Rewrite the current best draft into a tiny function or fenced pseudocode block before extending it."
                )
            if best.unit_tests_seen >= 4:
                parts.append(
                    f"[CWM transition_accuracy] {best.unit_tests_passed}/{best.unit_tests_seen} "
                    f"= {best.transition_accuracy:.2f} on the last {best.unit_tests_seen} observed transitions. "
                    "Per CWM (Lehrach et al. 2025) the target is 1.0 — refine `predict_effect` "
                    "until every recent transition replays correctly, then it is safe to plan with."
                )
                if best.transition_accuracy < 1.0:
                    parts.append(
                        "Look at the most recent surprises and edit the simulator branches "
                        "responsible for them. One tightened conditional often fixes several "
                        "failing tests at once."
                    )
        if self._latest_action_hint:
            parts.append(f"Current action prior: {self._latest_action_hint}")
        recent_surprises = self.bridge.recent_surprises(3)
        if recent_surprises:
            parts.append("Recent surprises (prediction vs reality mismatches):")
            for event in recent_surprises:
                expected = (
                    event.predicted.expected_change_summary if event.predicted else ""
                )
                parts.append(
                    f"- {event.action}: expected '{expected[:60]}' but observed diff_magnitude={event.observation.diff_magnitude:.0f}"
                )
        return "\n".join(parts)

    def on_memories_ready(self, memories: Any) -> None:
        self._memories = memories if hasattr(memories, "add") else None

    def before_action(
        self, frame: Any | None, action_name: str, available_actions: list[str]
    ) -> str | None:
        if frame is None:
            self._latest_action_hint = None
            self._publish_world_context()
            return action_name
        signature = self._signature(frame)
        bucket = self.transitions.get(signature, {})
        bucket_origin = "exact"
        if not bucket:
            bucket = self._nearest_bucket(frame)
            bucket_origin = "approximate"
        if action_name == "RESET":
            novelty_action = self._novelty_probe_action(bucket, available_actions)
            if novelty_action is not None:
                self._latest_action_hint = (
                    f"Avoid RESET and probe {novelty_action}: there is still an untested or least-used "
                    f"{bucket_origin} action available from this observable state."
                )
                self._record_prior_decision(
                    suggested=novelty_action,
                    chosen=novelty_action,
                    overridden=True,
                    count=float(bucket.get(novelty_action, {}).get("count", 0.0)),
                    avg_diff=float(bucket.get(novelty_action, {}).get("avg_diff", 0.0)),
                    zero_rate=self._zero_rate(bucket, novelty_action),
                )
                self.prior_suggestions += 1
                self.prior_overrides += 1
                self._publish_world_context(
                    suggested_action=novelty_action,
                    recent_surprise=False,
                )
                return novelty_action
        if not bucket:
            self._latest_action_hint = "No local transition model yet for this state or a nearby observable state."
            self._publish_world_context()
            return action_name

        candidates = []
        for candidate in available_actions:
            stats = bucket.get(candidate)
            if not stats:
                continue
            count = float(stats.get("count", 0.0))
            avg_diff = float(stats.get("avg_diff", 0.0))
            zero_rate = float(stats.get("zero_count", 0.0)) / max(count, 1.0)
            new_signature_rate = float(stats.get("new_signature_count", 0.0)) / max(count, 1.0)
            new_family_rate = float(stats.get("new_family_count", 0.0)) / max(count, 1.0)
            branch_escape_rate = float(stats.get("branch_escape_count", 0.0)) / max(count, 1.0)
            falsify_rate = float(stats.get("falsify_count", 0.0)) / max(count, 1.0)
            repeat_penalty = float(stats.get("repeat_depth_sum", 0.0)) / max(count, 1.0)
            value = (
                8.0 * new_family_rate
                + 6.0 * branch_escape_rate
                + 4.0 * new_signature_rate
                + 4.0 * falsify_rate
                + 0.1 * avg_diff
                - 4.0 * zero_rate
                - 1.5 * repeat_penalty
            )
            candidates.append(
                (
                    value,
                    count,
                    candidate,
                    avg_diff,
                    zero_rate,
                    new_signature_rate,
                    new_family_rate,
                    branch_escape_rate,
                    falsify_rate,
                    repeat_penalty,
                )
            )
        if not candidates:
            self._latest_action_hint = "This state has no usable local action statistics yet; explore available actions."
            self._publish_world_context()
            return action_name

        candidates.sort(reverse=True)
        self.prior_suggestions += 1
        (
            best_value,
            count,
            best_action,
            avg_diff,
            zero_rate,
            new_signature_rate,
            new_family_rate,
            branch_escape_rate,
            falsify_rate,
            repeat_penalty,
        ) = candidates[0]
        self._latest_action_hint = (
            f"{best_action} is currently the best-known {bucket_origin} local action "
            f"(count={int(count)}, new_family={new_family_rate:.2f}, branch_escape={branch_escape_rate:.2f}, "
            f"falsify={falsify_rate:.2f}, avg_diff={avg_diff:.1f}, repeat_penalty={repeat_penalty:.2f})."
        )
        if not bool(self.params.get("action_prior_enabled", True)):
            self._record_prior_decision(
                suggested=best_action,
                chosen=action_name,
                overridden=False,
                count=count,
                avg_diff=avg_diff,
                zero_rate=zero_rate,
            )
            self._publish_world_context(
                suggested_action=best_action,
                recent_surprise=False,
            )
            return action_name
        min_count = float(self.params.get("override_min_count", 2))
        min_value = float(self.params.get("override_min_value", 1.0))
        if self._should_force_novelty_probe(bucket, available_actions):
            novelty_action = self._novelty_probe_action(bucket, available_actions)
            if novelty_action is not None and novelty_action != action_name:
                self._latest_action_hint = (
                    f"Recent outcomes are stalled, so force a novelty probe with {novelty_action} "
                    f"instead of repeating low-information local actions."
                )
                self._record_prior_decision(
                    suggested=novelty_action,
                    chosen=novelty_action,
                    overridden=True,
                    count=float(bucket.get(novelty_action, {}).get("count", 0.0)),
                    avg_diff=float(bucket.get(novelty_action, {}).get("avg_diff", 0.0)),
                    zero_rate=self._zero_rate(bucket, novelty_action),
                )
                self.prior_overrides += 1
                self._publish_world_context(
                    suggested_action=novelty_action,
                    recent_surprise=False,
                )
                return novelty_action
        if self._should_break_override_loop(best_action):
            self._latest_action_hint = (
                f"{best_action} is locally strong, but it has repeated with the same recent outcome. "
                "Yielding control so the agent can probe a different direction."
            )
            self._record_prior_decision(
                suggested=best_action,
                chosen=action_name,
                overridden=False,
                count=count,
                avg_diff=avg_diff,
                zero_rate=zero_rate,
            )
            self._publish_world_context(
                suggested_action=best_action,
                recent_surprise=False,
            )
            return action_name
        if count >= min_count and best_value >= min_value:
            self.prior_overrides += 1
            self._record_prior_decision(
                suggested=best_action,
                chosen=best_action,
                overridden=True,
                count=count,
                avg_diff=avg_diff,
                zero_rate=zero_rate,
            )
            self._publish_world_context(
                suggested_action=best_action,
                recent_surprise=False,
            )
            return best_action
        self._record_prior_decision(
            suggested=best_action,
            chosen=action_name,
            overridden=False,
            count=count,
            avg_diff=avg_diff,
            zero_rate=zero_rate,
        )
        self._publish_world_context(
            suggested_action=best_action,
            recent_surprise=False,
        )
        return action_name

    def after_action(self, before: Any | None, action_name: str, after: Any) -> None:
        if before is None:
            return

        before_sig = self._signature(before)
        after_sig = self._signature(after)
        self.signature_features.setdefault(before_sig, self._features(before))
        self.signature_features.setdefault(after_sig, self._features(after))
        diff_mag = grid_diff_magnitude(before, after)
        self.recent_outcomes.append((action_name, float(diff_mag)))
        self.recent_outcomes = self.recent_outcomes[-16:]

        observation = Observation(
            action=action_name,
            before_signature=before_sig,
            after_signature=after_sig,
            diff_magnitude=diff_mag,
            before_family=self._family_id(before),
            after_family=self._family_id(after),
        )

        bucket = self.transitions.setdefault(before_sig, {})
        action_stats = bucket.setdefault(
            action_name,
            {
                "count": 0.0,
                "avg_diff": 0.0,
                "zero_count": 0.0,
                "new_signature_count": 0.0,
                "new_family_count": 0.0,
                "branch_escape_count": 0.0,
                "falsify_count": 0.0,
                "support_count": 0.0,
                "repeat_depth_sum": 0.0,
            },
        )
        action_stats["count"] += 1.0
        action_stats["avg_diff"] = (
            action_stats["avg_diff"] * (action_stats["count"] - 1) + diff_mag
        ) / action_stats["count"]
        if diff_mag == 0:
            action_stats["zero_count"] += 1.0

        current_prediction = self.bridge.current_prediction
        self._pending_prediction_was_structured = bool(
            current_prediction
            and current_prediction.action == action_name
            and self._is_scored_prediction(current_prediction.expected_change_summary)
        )
        if self._pending_prediction_was_structured:
            self.structured_predictions += 1

        # Ask the bridge whether this matches the current prediction, if any.
        surprise = self.bridge.record_observation(observation)
        if current_prediction is not None and current_prediction.action == action_name:
            observation.hypothesis_falsified = surprise is not None
            observation.hypothesis_supported = surprise is None
        action_stats["new_signature_count"] += float(observation.new_signature)
        action_stats["new_family_count"] += float(observation.new_family)
        action_stats["branch_escape_count"] += float(observation.branch_escape)
        action_stats["falsify_count"] += float(bool(observation.hypothesis_falsified))
        action_stats["support_count"] += float(bool(observation.hypothesis_supported))
        action_stats["repeat_depth_sum"] += float(observation.repeated_family_depth)
        if surprise is not None:
            self._note_surprise(surprise)
        self._maybe_seed_world_draft(before_sig)
        self._score_latest_draft(
            structured_prediction=self._pending_prediction_was_structured,
            surprise=surprise,
        )
        self._score_simulator_draft(before, action_name, after)
        # Append this real transition to the CWM unit-test buffer and
        # immediately re-evaluate the best draft against the most recent
        # window of tests. The agent's job is to revise the draft until
        # transition_accuracy = 1.0 on the window.
        self._append_unit_test(
            before_sig=before_sig,
            action=action_name,
            after_sig=after_sig,
            diff_band=self._diff_band(diff_mag),
            diff_magnitude=float(diff_mag),
            new_family=bool(observation.new_family),
            branch_escape=bool(observation.branch_escape),
            before=before,
            after=after,
        )
        self._refresh_transition_accuracy()
        if self._pending_prediction_was_structured:
            self._pending_posthoc_score = surprise is None
        self._pending_prediction_was_structured = False
        self._publish_world_context(recent_surprise=surprise is not None)

    def on_run_end(
        self,
        *,
        history: list[tuple[str, Any]],
        finish_status: Any,
        workdir: Path,
    ) -> None:
        self._publish_world_context()
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(
            json.dumps(
                {
                    "transitions": self.transitions,
                    "signature_features": self.signature_features,
                    "agent_env_notes": self.agent_env_notes[-16:],
                    "world_drafts": [asdict(draft) for draft in self.world_drafts[-8:]],
                    "prior_suggestions": self.prior_suggestions,
                    "prior_overrides": self.prior_overrides,
                    "structured_predictions": self.structured_predictions,
                    "scored_prediction_matches": self.scored_prediction_matches,
                    "scored_prediction_mismatches": self.scored_prediction_mismatches,
                    "recent_prior_decisions": self.recent_prior_decisions[-16:],
                    "recent_outcomes": self.recent_outcomes[-16:],
                    "unit_tests": self.unit_tests[-self.unit_test_capacity :],
                    # v3 monitor §7: persist transition_accuracy explicitly
                    # so the 20-min monitor can read it without re-running
                    # the simulator.
                    "transition_accuracy": (
                        self._best_draft().transition_accuracy
                        if self._best_draft() is not None else 0.0
                    ),
                    "unit_tests_passed": (
                        self._best_draft().unit_tests_passed
                        if self._best_draft() is not None else 0
                    ),
                    "unit_tests_seen": (
                        self._best_draft().unit_tests_seen
                        if self._best_draft() is not None else 0
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # -- Agent tools ------------------------------------------------------
    def record_agent_prediction(self, action: str, expected_change: Any) -> None:
        """Called when the agent's LLM response contains a prediction."""
        prediction = self._normalize_prediction_payload(action, expected_change)
        self.bridge.record_prediction(prediction)

    def record_agent_env_note(self, text: str) -> None:
        """Called when the agent wants to add a note to its own environment model."""
        text = text.strip()
        if text:
            self.agent_env_notes.append(text)
            self.agent_env_notes = self.agent_env_notes[-32:]

    def record_agent_world_update(self, payload: Any) -> None:
        body, focus, self_score = self._normalize_world_update(payload)
        if not body:
            return
        current = self._best_draft()
        if current is not None and current.body == body:
            current.self_score = self_score
            current.focus = focus or current.focus
            current.last_updated_at = time.time()
            self._remember_world_draft(current, event="rescored")
            self._publish_world_context()
            return
        if current is not None:
            current.body = body
            current.focus = focus or current.focus
            current.self_score = self_score
            current.revisions += 1
            current.last_updated_at = time.time()
            self._apply_pending_posthoc_score(current)
            self._remember_world_draft(current, event="revised")
            self._publish_world_context()
            return
        self.world_drafts.append(WorldDraft(body=body, focus=focus, self_score=self_score))
        self.world_drafts = self.world_drafts[-8:]
        self._apply_pending_posthoc_score(self.world_drafts[-1])
        self._remember_world_draft(self.world_drafts[-1], event="added")
        self._publish_world_context()

    # -- Helpers ----------------------------------------------------------
    def _note_surprise(self, event: SurpriseEvent) -> None:
        summary = (
            event.predicted.expected_change_summary
            if event.predicted and event.predicted.expected_change_summary
            else "(no prediction given)"
        )
        self.agent_env_notes.append(
            f"Surprise on {event.action}: expected '{summary[:60]}' / observed diff={event.observation.diff_magnitude:.0f}"
        )
        self.agent_env_notes = self.agent_env_notes[-32:]
        if self._memories is not None:
            self._memories.add(
                f"[World Surprise] {event.action}",
                (
                    f"expected: {summary}\n"
                    f"observed_diff: {event.observation.diff_magnitude:.0f}\n"
                    f"trajectory: {event.recent_trajectory[-8:]}"
                ),
            )

    def _score_latest_draft(
        self, *, structured_prediction: bool, surprise: SurpriseEvent | None
    ) -> None:
        draft = self._best_draft()
        if draft is None or not structured_prediction:
            return
        if surprise is None:
            draft.empirical_matches += 1
            self.scored_prediction_matches += 1
            self.agent_env_notes.append(
                f"✅ PREDICTION HIT last action. Draft score now {draft.score():.1f}."
            )
        else:
            draft.empirical_mismatches += 1
            self.scored_prediction_mismatches += 1
            # Concrete mismatch feedback for immediate learning.
            pred = surprise.predicted
            obs = surprise.observation
            expected_summary = (
                pred.expected_change_summary if pred else "(no prediction)"
            )
            actual = (
                f"actual: diff={obs.diff_magnitude:.0f} cells "
                f"new_family={obs.new_family} branch_escape={obs.branch_escape}"
            )
            self.agent_env_notes.append(
                f"❌ PREDICTION MISS on {surprise.action}: "
                f"expected '{expected_summary[:80]}' | {actual}. "
                f"Draft score now {draft.score():.1f} (penalty -2.0 applied). "
                f"Edit predict_effect NOW to absorb this transition."
            )
        draft.last_updated_at = time.time()
        self.agent_env_notes = self.agent_env_notes[-32:]
        if self._memories is not None:
            self._memories.add(
                f"[World Score] {draft.focus or 'draft'}",
                (
                    f"matches: {draft.empirical_matches}\n"
                    f"mismatches: {draft.empirical_mismatches}\n"
                    f"score: {draft.score():.1f}\n"
                    f"body:\n{draft.body}"
                ),
            )

    def _apply_pending_posthoc_score(self, draft: WorldDraft) -> None:
        if self._pending_posthoc_score is None:
            return
        if draft.empirical_matches > 0 or draft.empirical_mismatches > 0:
            self._pending_posthoc_score = None
            return
        if self._pending_posthoc_score:
            draft.empirical_matches += 1
        else:
            draft.empirical_mismatches += 1
        draft.last_updated_at = time.time()
        self._pending_posthoc_score = None

    def _record_prior_decision(
        self,
        *,
        suggested: str,
        chosen: str,
        overridden: bool,
        count: float,
        avg_diff: float,
        zero_rate: float,
    ) -> None:
        self.recent_prior_decisions.append(
            {
                "suggested": suggested,
                "chosen": chosen,
                "overridden": overridden,
                "count": float(count),
                "avg_diff": float(avg_diff),
                "zero_rate": float(zero_rate),
            }
        )
        self.recent_prior_decisions = self.recent_prior_decisions[-32:]

    def _best_draft(self) -> WorldDraft | None:
        if not self.world_drafts:
            return None
        return max(self.world_drafts, key=lambda d: (d.score(), d.last_updated_at))

    def _publish_world_context(
        self,
        *,
        suggested_action: str | None = None,
        recent_surprise: bool | None = None,
    ) -> None:
        best = self._best_draft()
        self.bridge.update_hint(
            "world_model",
            {
                "suggested_action": suggested_action or "",
                "hint": self._latest_action_hint or "",
                "best_focus": best.focus if best is not None else "",
                "best_score": round(best.score(), 2) if best is not None else 0.0,
                "local_attractor_pressure": self._local_attractor_pressure(),
                "recent_surprise": bool(recent_surprise),
            },
        )

    @staticmethod
    def _zero_rate(bucket: dict[str, dict[str, float]], action: str) -> float:
        stats = bucket.get(action, {})
        count = float(stats.get("count", 0.0))
        if count <= 0:
            return 0.0
        return float(stats.get("zero_count", 0.0)) / count

    def _should_break_override_loop(self, suggested_action: str) -> bool:
        window = int(self.params.get("override_loop_guard_window", 3))
        max_spread = float(self.params.get("override_loop_guard_diff_spread", 0.5))
        if window <= 1 or len(self.recent_outcomes) < window:
            return False
        recent = self.recent_outcomes[-window:]
        if any(action != suggested_action for action, _ in recent):
            return False
        diffs = [diff for _, diff in recent]
        return max(diffs) - min(diffs) <= max_spread

    def _should_force_novelty_probe(
        self,
        bucket: dict[str, dict[str, float]],
        available_actions: list[str],
    ) -> bool:
        if not bool(self.params.get("exploration_override_enabled", True)):
            return False
        stall_window = int(self.params.get("exploration_stall_window", 4))
        if stall_window <= 1 or len(self.recent_outcomes) < stall_window:
            return False
        recent = self.recent_outcomes[-stall_window:]
        family_window = self.bridge.recent_families[-stall_window:]
        same_family_trap = (
            len(family_window) >= stall_window
            and len(set(family_window)) == 1
        )
        if any(diff > 0 for _, diff in recent) and not same_family_trap:
            return False
        if len(set(action for action, _ in recent)) >= len(set(available_actions)):
            return True
        return any(action not in bucket for action in available_actions)

    def _novelty_probe_action(
        self,
        bucket: dict[str, dict[str, float]],
        available_actions: list[str],
    ) -> str | None:
        unseen = [action for action in available_actions if action not in bucket]
        if unseen:
            return sorted(unseen)[0]
        ranked = []
        for action in available_actions:
            stats = bucket.get(action, {})
            count = float(stats.get("count", 0.0))
            avg_diff = float(stats.get("avg_diff", 0.0))
            zero_rate = self._zero_rate(bucket, action)
            ranked.append((count, zero_rate, avg_diff, action))
        if not ranked:
            return None
        ranked.sort()
        return ranked[0][3]

    def _remember_world_draft(self, draft: WorldDraft, *, event: str) -> None:
        if self._memories is None:
            return
        self._memories.add(
            f"[World Draft] {draft.focus or 'local-transition'}",
            (
                f"event: {event}\n"
                f"self_score: {draft.self_score:.1f}\n"
                f"matches: {draft.empirical_matches}\n"
                f"mismatches: {draft.empirical_mismatches}\n"
                f"sim_matches: {draft.simulator_matches}\n"
                f"sim_mismatches: {draft.simulator_mismatches}\n"
                f"revisions: {draft.revisions}\n"
                f"body:\n{draft.body}"
            ),
        )

    def _score_simulator_draft(self, before: Any, action_name: str, after: Any) -> None:
        draft = self._best_draft()
        if draft is None:
            return
        prediction = self._run_simulator_draft(draft, before, action_name)
        if prediction is None:
            return
        actual_diff = grid_diff_magnitude(before, after)
        actual_band = self._diff_band(actual_diff)
        observation = self.bridge.last_observation
        expect_change = prediction.get("expect_change")
        expected_band = str(prediction.get("expected_diff_band", "unknown")).strip().lower()
        progress_prediction = str(prediction.get("progress_prediction", "")).strip().lower()
        observation_prediction = str(prediction.get("observation_prediction", "")).strip().lower()

        band_ok = expected_band in {"", "unknown", actual_band}
        change_ok = (
            expect_change is None
            or bool(expect_change) == (actual_diff > 0)
        )
        progress_ok = True
        if observation is not None and progress_prediction:
            if "new_family" in progress_prediction and not observation.new_family:
                progress_ok = False
            if "branch_escape" in progress_prediction and not observation.branch_escape:
                progress_ok = False
            if "falsify" in progress_prediction and not bool(observation.hypothesis_falsified):
                progress_ok = False
        observation_ok = True
        if observation is not None and observation_prediction:
            if "same_family" in observation_prediction and observation.branch_escape:
                observation_ok = False
            if "new_family" in observation_prediction and not observation.new_family:
                observation_ok = False
        matched = band_ok and change_ok and progress_ok and observation_ok
        if matched:
            draft.simulator_matches += 1
        else:
            draft.simulator_mismatches += 1
            self.agent_env_notes.append(
                "Simulator mismatch: "
                f"pred={prediction} actual_diff={actual_diff:.0f} actual_band={actual_band}"
            )
            self.agent_env_notes = self.agent_env_notes[-32:]
        draft.last_updated_at = time.time()

    # -- v3 simulator API ---------------------------------------------------
    def simulate_step(
        self,
        state_signature: str,
        action: str,
        observation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Run the best draft's `predict_effect` on a hypothetical state.

        This is the planner-facing API. It must return a dict with at
        least:
          - next_signature_hint: str  (or 'unknown')
          - reward_signals: dict with keys
              new_family, branch_escape, expect_change, expected_diff_band,
              progress_prediction
          - exception_flag: bool  (True if predict_effect raised)

        Per plan H5b, callers should gate on
        `self._best_draft().transition_accuracy >= t*` before treating
        results as reliable. We do not gate here — the planner does.

        `observation` is optional; if missing we synthesise a minimal one
        matching the contract used by `_score_simulator_draft`."""
        draft = self._best_draft()
        if draft is None:
            return {
                "next_signature_hint": "unknown",
                "reward_signals": {
                    "new_family": False,
                    "branch_escape": False,
                    "expect_change": None,
                    "expected_diff_band": "unknown",
                    "progress_prediction": "",
                },
                "exception_flag": False,
                "draft_present": False,
            }
        if observation is None:
            observation = {
                "signature": state_signature,
                "family": self.signature_features.get(state_signature, []),
                "available_actions": [],
                "recent_actions": [a for a, _d in self.recent_outcomes[-8:]],
                "grid": [],
                "grid_text": "",
                "shape": (0, 0),
                "features": [],
                "recent_outcomes": [],
            }
        else:
            observation = dict(observation)
            observation.setdefault("signature", state_signature)
        code = self._extract_code_block(draft.body)
        if not code:
            return {
                "next_signature_hint": "unknown",
                "reward_signals": {
                    "new_family": False,
                    "branch_escape": False,
                    "expect_change": None,
                    "expected_diff_band": "unknown",
                    "progress_prediction": "",
                },
                "exception_flag": False,
                "draft_present": True,
                "code_present": False,
            }
        globals_dict = {
            "__builtins__": {
                "len": len, "min": min, "max": max, "sum": sum,
                "abs": abs, "sorted": sorted, "range": range,
                "tuple": tuple, "list": list, "dict": dict, "set": set,
                "str": str, "int": int, "float": float, "bool": bool,
                "enumerate": enumerate, "zip": zip, "map": map, "filter": filter,
                "any": any, "all": all, "round": round, "isinstance": isinstance,
                "True": True, "False": False, "None": None,
            }
        }
        locals_dict: dict[str, Any] = {}
        try:
            exec(code, globals_dict, locals_dict)
        except Exception:
            return {
                "next_signature_hint": "unknown",
                "reward_signals": {
                    "new_family": False, "branch_escape": False,
                    "expect_change": None, "expected_diff_band": "unknown",
                    "progress_prediction": "",
                },
                "exception_flag": True,
                "draft_present": True,
                "code_present": True,
            }
        fn = locals_dict.get("predict_effect") or globals_dict.get("predict_effect")
        if not callable(fn):
            return {
                "next_signature_hint": "unknown",
                "reward_signals": {
                    "new_family": False, "branch_escape": False,
                    "expect_change": None, "expected_diff_band": "unknown",
                    "progress_prediction": "",
                },
                "exception_flag": False,
                "draft_present": True,
                "code_present": True,
                "fn_present": False,
            }
        try:
            pred = fn(action, observation)
        except Exception:
            return {
                "next_signature_hint": "unknown",
                "reward_signals": {
                    "new_family": False, "branch_escape": False,
                    "expect_change": None, "expected_diff_band": "unknown",
                    "progress_prediction": "",
                },
                "exception_flag": True,
            }
        if not isinstance(pred, dict):
            pred = {}
        next_sig_hint = str(pred.get("next_signature_hint", "unknown")) or "unknown"
        progress = str(pred.get("progress_prediction", "")).lower()
        observation_pred = str(pred.get("observation_prediction", "")).lower()
        return {
            "next_signature_hint": next_sig_hint,
            "reward_signals": {
                "new_family": "new_family" in progress
                    or "new_family" in observation_pred,
                "branch_escape": "branch_escape" in progress,
                "expect_change": pred.get("expect_change"),
                "expected_diff_band": str(pred.get("expected_diff_band", "unknown")).lower(),
                "progress_prediction": progress,
            },
            "exception_flag": False,
            "transition_accuracy": draft.transition_accuracy,
            "draft_score": draft.score(),
        }

    def wake_prompt_section(self) -> str:
        """Plan §4: Wake-specific overlay block.

        Drained when registry's wake_phase fires. Asks the LLM to rewrite
        the failing branch of `predict_effect`. Empty string when no wake
        is queued (avoids overlay clutter)."""
        events = self.bridge.drain_wake_triggers() if hasattr(self.bridge, "drain_wake_triggers") else []
        if not events:
            return ""
        # v3.15: persist the wake event to shared memory so the LLM sees
        # it even across turn boundaries where the pending queue drains
        # before it's rendered.
        if self._memories is not None:
            for ev in events[-3:]:
                obs = ev.observation
                pred = ev.predicted
                self._memories.add(
                    f"[Wake] {ev.action}",
                    (
                        f"expected: {(pred.expected_change_summary if pred else '?')[:100]}\n"
                        f"observed diff: {obs.diff_magnitude:.0f} cells, "
                        f"new_family={obs.new_family}, branch_escape={obs.branch_escape}\n"
                        f"Edit predict_effect to absorb this transition."
                    ),
                )
        draft = self._best_draft()
        body_excerpt = (draft.body[:600] if draft else "(no draft yet)").strip()
        lines = [
            "[Wake] Real surprise occurred. Rewrite `predict_effect` so the",
            "       failing transition replays correctly. Do not invent",
            "       branches that aren't supported by recent observation.",
        ]
        for ev in events[-3:]:
            obs = ev.observation
            pred = ev.predicted
            lines.append(
                f"- action={ev.action} expected="
                f"'{(pred.expected_change_summary if pred else '?')[:80]}' "
                f"observed_diff={obs.diff_magnitude:.0f} "
                f"new_family={obs.new_family} branch_escape={obs.branch_escape}"
            )
        lines.append("Current best draft body (excerpt):")
        lines.append(body_excerpt)
        # Per-action aggregate stats so the LLM stops oscillating between
        # incompatible theories on every surprise. Without this, 79 surprises
        # produced 4 mutually inconsistent drafts; all shared the same
        # cross-action invariant (diff ≈ 52 regardless of direction).
        stats_summary = self._per_action_stats_summary()
        if stats_summary:
            lines.append("Per-action aggregate statistics across this run:")
            lines.extend(f"  {line}" for line in stats_summary)
            lines.append(
                "Notice which invariants hold across all actions vs which "
                "actually distinguish them. A draft whose branches differ "
                "only on the shared invariants will keep missing."
            )
        lines.append(
            "Goal: minimal edit. Add or tighten one conditional in "
            "`predict_effect` so the surprise above would not fire again "
            "AND the draft remains consistent with the aggregate stats."
        )
        return "\n".join(lines)

    def _per_action_stats_summary(self) -> list[str]:
        """One line per action with non-trivial count: mean_diff, new_family
        rate, branch_escape rate, zero rate."""
        from collections import defaultdict
        agg: dict[str, dict[str, float]] = defaultdict(lambda: {
            "count": 0.0, "avg_diff_sum": 0.0, "new_family_count": 0.0,
            "branch_escape_count": 0.0, "zero_count": 0.0,
        })
        for state_sig, bucket in self.transitions.items():
            for action, stats in bucket.items():
                count = float(stats.get("count", 0.0))
                if count <= 0:
                    continue
                agg[action]["count"] += count
                agg[action]["avg_diff_sum"] += float(stats.get("avg_diff", 0.0)) * count
                agg[action]["new_family_count"] += float(stats.get("new_family_count", 0.0))
                agg[action]["branch_escape_count"] += float(stats.get("branch_escape_count", 0.0))
                agg[action]["zero_count"] += float(stats.get("zero_count", 0.0))
        out: list[str] = []
        for action in sorted(agg.keys()):
            st = agg[action]
            n = st["count"]
            if n < 2:
                continue
            mean_diff = st["avg_diff_sum"] / n
            nf = st["new_family_count"] / n
            be = st["branch_escape_count"] / n
            zr = st["zero_count"] / n
            out.append(
                f"{action}: n={int(n)} mean_diff={mean_diff:.1f} "
                f"new_family={nf:.0%} branch_escape={be:.0%} zero={zr:.0%}"
            )
        return out

    def _append_unit_test(
        self,
        *,
        before_sig: str,
        action: str,
        after_sig: str,
        diff_band: str,
        diff_magnitude: float = 0.0,
        new_family: bool,
        branch_escape: bool,
        before: Any,
        after: Any = None,
    ) -> None:
        """Record one observed transition as a CWM unit test.

        Per Lehrach et al. 2025: unit tests are auto-generated from
        offline trajectories and the world-model program is refined until
        transition_accuracy = matched/total reaches 1.0. We mirror that
        here by storing the minimum information needed to re-evaluate
        any candidate ``predict_effect`` simulator on the same input."""
        # We embed a compact simulator-observation snapshot so the test
        # is reproducible without the full FrameData object. Only the
        # signature, family, available_actions, and recent_actions are
        # needed by predict_effect; the grid hash already collapses the
        # full visible state into the signature key.
        sim_obs = self._make_simulator_observation(before)
        # Pixel-level ground truth for CWM-faithful state-transition
        # scoring. Storing the full 64x64 before/after grids is costly
        # but necessary for `expected_grid_delta` / `expected_next_grid`
        # matching. We keep only the last `unit_test_capacity` tests so
        # the persisted JSON stays bounded.
        before_grid = current_grid(before) if before is not None else []
        after_grid = current_grid(after) if after is not None else []
        self.unit_tests.append(
            {
                "before_signature": before_sig,
                "action": action,
                "after_signature": after_sig,
                "diff_band": diff_band,
                "diff_magnitude": float(diff_magnitude),
                "new_family": bool(new_family),
                "branch_escape": bool(branch_escape),
                "before_grid": before_grid,
                "after_grid": after_grid,
                "observation": {
                    "signature": sim_obs["signature"],
                    "family": sim_obs["family"],
                    "grid": before_grid,
                    "grid_text": sim_obs.get("grid_text", ""),
                    "available_actions": list(sim_obs["available_actions"]),
                    "recent_actions": list(sim_obs["recent_actions"]),
                },
            }
        )
        if len(self.unit_tests) > self.unit_test_capacity:
            self.unit_tests = self.unit_tests[-self.unit_test_capacity :]

    def _refresh_transition_accuracy(self) -> None:
        """Re-evaluate the best draft against the recent unit-test window.

        We deliberately use a sliding window (default 64) rather than the
        full buffer so the score reflects the agent's *current* model of
        the *current* level, not a stale aggregate from earlier levels."""
        draft = self._best_draft()
        if draft is None or not self.unit_tests:
            return
        window = self.unit_tests[-self.unit_test_eval_window :]
        passed, total = self._evaluate_draft_on_unit_tests(draft, window)
        draft.unit_tests_seen = total
        draft.unit_tests_passed = passed
        draft.transition_accuracy = (passed / total) if total > 0 else 0.0
        draft.last_unit_test_eval_at = time.time()
        if total >= 4 and draft.transition_accuracy < 1.0:
            self.agent_env_notes.append(
                f"World draft transition_accuracy={draft.transition_accuracy:.2f} "
                f"({passed}/{total}). Refine the simulator until accuracy = 1.0."
            )
            self.agent_env_notes = self.agent_env_notes[-32:]

    def _evaluate_draft_on_unit_tests(
        self,
        draft: WorldDraft,
        tests: list[dict[str, Any]],
    ) -> tuple[int, int]:
        code = self._extract_code_block(draft.body)
        if not code:
            # Without a code-shaped draft we cannot run the simulator;
            # report 0/0 so the agent is prompted to add one.
            return (0, 0)
        globals_dict = {
            "__builtins__": {
                "len": len,
                "min": min,
                "max": max,
                "sum": sum,
                "abs": abs,
                "sorted": sorted,
                "range": range,
                "tuple": tuple,
                "list": list,
                "dict": dict,
                "set": set,
            }
        }
        locals_dict: dict[str, Any] = {}
        try:
            exec(code, globals_dict, locals_dict)
        except Exception:
            return (0, len(tests))
        fn = locals_dict.get("predict_effect") or globals_dict.get("predict_effect")
        if not callable(fn):
            return (0, len(tests))
        passed = 0
        for test in tests:
            try:
                pred = fn(test["action"], test["observation"])
            except Exception:
                continue
            if not isinstance(pred, dict):
                continue
            if self._unit_test_matches(pred, test):
                passed += 1
        return (passed, len(tests))

    @staticmethod
    def _unit_test_matches(pred: dict[str, Any], test: dict[str, Any]) -> bool:
        """STRICT scorer (v3.9+). An earlier version let 'unknown'/None fields
        pass through, which lets the LLM game the metric by emitting
        wildcards on every turn. Now every field must be concrete AND
        correct, OR the test fails.

        Minimum required fields on every prediction:
          - expect_change ∈ {True, False} (None → FAIL)
          - expected_diff_band ∈ {"zero", "small", "large"} ("unknown"/"" → FAIL)
        Optional but strict when present:
          - observation_prediction → "new_family" | "same_family" literal check
          - progress_prediction    → same style
          - next_signature_hint    → semantic keyword must agree with outcome
          - expected_diff_cells (int) → must be within ±30% of actual
        """
        # 1. expect_change — mandatory concrete bool.
        expect_change = pred.get("expect_change")
        actual_band = test["diff_band"]
        actual_change = actual_band != "zero"
        if expect_change not in (True, False):
            return False
        if bool(expect_change) != actual_change:
            return False

        # 2. expected_diff_band — mandatory concrete band.
        expected_band = str(pred.get("expected_diff_band", "")).strip().lower()
        if expected_band not in {"zero", "small", "large"}:
            return False
        if expected_band != actual_band:
            return False

        # 3. optional observation_prediction (strict literal if provided).
        observation_pred = str(pred.get("observation_prediction", "")).strip().lower()
        if observation_pred:
            if "new_family" in observation_pred and not test["new_family"]:
                return False
            if "same_family" in observation_pred and test["branch_escape"]:
                return False

        # 4. optional progress_prediction (strict literal if provided).
        progress_pred = str(pred.get("progress_prediction", "")).strip().lower()
        if progress_pred:
            if "new_family" in progress_pred and not test["new_family"]:
                return False
            if "branch_escape" in progress_pred and not test["branch_escape"]:
                return False
            if "same_family" in progress_pred and test["branch_escape"]:
                return False

        # 5. optional next_signature_hint — must agree semantically.
        next_sig_hint = pred.get("next_signature_hint")
        if isinstance(next_sig_hint, str) and next_sig_hint:
            hint_lower = next_sig_hint.strip().lower()
            if hint_lower == "unknown":
                # Explicit 'unknown' on next_signature_hint is a cop-out;
                # do not fail it here but it no longer provides a free
                # pass either (expect_change/band already gate the test).
                pass
            else:
                looks_like_hash = (
                    len(hint_lower) >= 16
                    and all(c in "0123456789abcdef" for c in hint_lower)
                )
                if looks_like_hash:
                    if next_sig_hint != test["after_signature"]:
                        return False
                else:
                    if any(k in hint_lower for k in ("new_", "newfam", "new family", "different")):
                        if not test["new_family"]:
                            return False
                    if any(k in hint_lower for k in ("same", "unchanged", "stay")):
                        if test["new_family"] or test["branch_escape"]:
                            return False

        # 6. optional expected_diff_cells — must be within ±30% of actual.
        predicted_cells = pred.get("expected_diff_cells")
        if isinstance(predicted_cells, (int, float)):
            actual_cells = float(test.get("diff_magnitude", 0.0))
            if actual_cells > 0:
                rel = abs(float(predicted_cells) - actual_cells) / actual_cells
                if rel > 0.3:
                    return False
            else:
                # actual zero; predicted must also be ≈ 0 (≤1 cell tolerance)
                if float(predicted_cells) > 1:
                    return False

        # 7. PIXEL-LEVEL prediction (CWM-faithful state transition).
        # Two acceptable forms:
        #   (a) `expected_grid_delta`: list[(x,y,new_val)] — predicted cell
        #       changes; at least 50% must match actual changes in test.
        #   (b) `expected_next_grid`: full 2D list 64x64 — each cell must
        #       match actual after_grid exactly (tolerance 0 by default).
        # When neither provided, scoring falls through on (1)-(6) above.
        # The test payload must carry `after_grid` + `before_grid` for
        # this check; when absent we skip pixel scoring.
        after_grid = test.get("after_grid")
        before_grid = test.get("before_grid")
        predicted_delta = pred.get("expected_grid_delta")
        predicted_grid = pred.get("expected_next_grid")

        # v3.12: full-grid prediction is MANDATORY when we have ground truth.
        # delta-only is no longer enough — agent would game it by predicting
        # 1 cell and scoring 100% precision with near-zero recall. To pass
        # pixel scoring the agent must produce an `expected_next_grid` that
        # is ≥95% cell-accurate; delta remains a supplementary signal.
        if after_grid is not None:
            has_full_grid = isinstance(predicted_grid, list) and predicted_grid
            if not has_full_grid:
                # Missing expected_next_grid — FAIL (was previously
                # allowed to fall through on delta alone).
                return False

            # v3.15 IDENTITY GUARD: if predicted_grid == before_grid and
            # the test says the grid actually changed (diff_band != zero),
            # the agent is gaming via "do-nothing" prediction. FAIL.
            if before_grid is not None and test.get("diff_band") != "zero":
                grid_h = min(len(predicted_grid), len(before_grid))
                grid_w = min(
                    len(predicted_grid[0]) if predicted_grid and predicted_grid[0] else 0,
                    len(before_grid[0]) if before_grid and before_grid[0] else 0,
                )
                is_identity = True
                for y in range(grid_h):
                    for x in range(grid_w):
                        if int(predicted_grid[y][x]) != int(before_grid[y][x]):
                            is_identity = False
                            break
                    if not is_identity:
                        break
                if is_identity:
                    return False

            # Full grid prediction: count cell mismatches, require ≥95%.
            h = min(len(predicted_grid), len(after_grid))
            w = min(
                len(predicted_grid[0]) if predicted_grid and predicted_grid[0] else 0,
                len(after_grid[0]) if after_grid and after_grid[0] else 0,
            )
            mismatches = 0
            for y in range(h):
                for x in range(w):
                    if int(predicted_grid[y][x]) != int(after_grid[y][x]):
                        mismatches += 1
            # Allow up to 5% cell error to accommodate rendering noise.
            total_cells = max(h * w, 1)
            if mismatches / total_cells > 0.05:
                return False

            # Delta (if provided) must also be consistent — precision
            # and recall guarantees on top of the full grid.
            if isinstance(predicted_delta, list) and predicted_delta:
                # Delta prediction: ≥80% of predicted cells must agree
                # with actual AND ≥50% of actual changes must be covered
                # by the predicted delta (precision + recall).
                actual_changes: set[tuple[int, int, int]] = set()
                if before_grid is not None:
                    for y in range(min(len(before_grid), len(after_grid))):
                        for x in range(min(len(before_grid[y]), len(after_grid[y]))):
                            b, a = int(before_grid[y][x]), int(after_grid[y][x])
                            if b != a:
                                actual_changes.add((x, y, a))
                hits = 0
                predicted_set: set[tuple[int, int, int]] = set()
                for entry in predicted_delta:
                    if not (isinstance(entry, (list, tuple)) and len(entry) >= 3):
                        continue
                    try:
                        triple = (int(entry[0]), int(entry[1]), int(entry[2]))
                    except Exception:
                        continue
                    predicted_set.add(triple)
                    if (
                        0 <= triple[1] < len(after_grid)
                        and 0 <= triple[0] < len(after_grid[triple[1]])
                        and int(after_grid[triple[1]][triple[0]]) == triple[2]
                    ):
                        hits += 1
                if predicted_set:
                    precision = hits / len(predicted_set)
                    if precision < 0.8:
                        return False
                    if actual_changes:
                        recall = len(predicted_set & actual_changes) / max(
                            1, len(actual_changes)
                        )
                        if recall < 0.5:
                            return False

        return True

    def _run_simulator_draft(
        self,
        draft: WorldDraft,
        before: Any,
        action_name: str,
    ) -> dict[str, Any] | None:
        code = self._extract_code_block(draft.body)
        if not code:
            return None
        globals_dict = {
            "__builtins__": {
                "len": len,
                "min": min,
                "max": max,
                "sum": sum,
                "abs": abs,
                "sorted": sorted,
                "range": range,
                "tuple": tuple,
                "list": list,
                "dict": dict,
                "set": set,
            }
        }
        locals_dict: dict[str, Any] = {}
        try:
            exec(code, globals_dict, locals_dict)
        except Exception:
            return self._heuristic_predict_effect_from_text(draft.body, action_name, before)
        fn = locals_dict.get("predict_effect") or globals_dict.get("predict_effect")
        if not callable(fn):
            return self._heuristic_predict_effect_from_text(draft.body, action_name, before)
        try:
            result = fn(action_name, self._make_simulator_observation(before))
        except Exception:
            return self._heuristic_predict_effect_from_text(draft.body, action_name, before)
        if not isinstance(result, dict):
            return self._heuristic_predict_effect_from_text(draft.body, action_name, before)
        return result

    def _maybe_seed_world_draft(self, signature: str) -> None:
        if self.world_drafts:
            return
        min_obs = int(self.params.get("auto_seed_min_observations", 3))
        if len(self.recent_outcomes) < min_obs:
            return
        bucket = self.transitions.get(signature, {})
        if not bucket:
            return
        zero_actions = sorted(
            action
            for action, stats in bucket.items()
            if float(stats.get("count", 0.0)) >= 1.0
            and float(stats.get("zero_count", 0.0)) >= float(stats.get("count", 0.0))
        )
        observed = sorted(bucket.keys())
        focus = "observed local transition baseline"
        body = (
            "score: 0.5\n"
            f"focus: {focus}\n"
            "```python\n"
            "def predict_effect(action, observation):\n"
            f"    if observation['signature'] == {signature!r} and action in {tuple(zero_actions)!r}:\n"
            "        return {'expect_change': False, 'expected_diff_band': 'zero', 'expected_diff_cells': 0, 'next_signature_hint': observation['signature'], 'observation_prediction': 'same_family', 'progress_prediction': 'same_family'}\n"
            "    # Replace this default with a concrete guess BEFORE strict scoring starts.\n"
            "    return {'expect_change': True, 'expected_diff_band': 'large', 'expected_diff_cells': 52, 'next_signature_hint': 'remap-large', 'observation_prediction': 'same_family', 'progress_prediction': 'same_family', 'action_recommendation': ''}\n"
            f"# observed_actions={observed!r}\n"
            "```"
        )
        self.record_agent_world_update(body)

    @staticmethod
    def _is_scored_prediction(text: str) -> bool:
        upper = text.strip().upper()
        return upper.startswith("EXPECT_CHANGE") or upper.startswith("EXPECT_NO_CHANGE")

    @staticmethod
    def _normalize_world_update(payload: Any) -> tuple[str, str, float]:
        if isinstance(payload, str):
            body, focus, self_score = WorldModelModule._parse_markdown_world_update(payload)
            return WorldModelModule._ensure_code_like_body(body, focus), focus, self_score
        if not isinstance(payload, dict):
            body = str(payload).strip()
            return WorldModelModule._ensure_code_like_body(body, ""), "", 0.0
        body = str(payload.get("draft", payload.get("body", ""))).strip()
        focus = str(payload.get("focus", "")).strip()
        score_raw = payload.get("self_score", payload.get("score", 0.0))
        try:
            self_score = float(score_raw)
        except Exception:
            self_score = 0.0
        return WorldModelModule._ensure_code_like_body(body, focus), focus, self_score

    @staticmethod
    def _parse_markdown_world_update(text: str) -> tuple[str, str, float]:
        lines = [line.rstrip() for line in text.strip().splitlines()]
        focus = ""
        self_score = 0.0
        body_lines: list[str] = []
        for line in lines:
            lower = line.lower().strip()
            if lower.startswith("score:"):
                raw = line.split(":", 1)[1].strip().split("/", 1)[0].strip()
                try:
                    self_score = float(raw)
                except Exception:
                    self_score = 0.0
                continue
            if lower.startswith("focus:"):
                focus = line.split(":", 1)[1].strip()
                continue
            body_lines.append(line)
        return "\n".join(body_lines).strip(), focus, self_score

    @staticmethod
    def _ensure_code_like_body(body: str, focus: str) -> str:
        body = body.strip()
        if not body:
            return body
        legacy = WorldModelModule._legacy_rule_to_predict_effect(body, focus)
        if legacy:
            return legacy
        if "```" in body or re.search(r"\b(if|elif|else|return|def|for|while)\b", body):
            return body
        comments = [
            line.strip().lstrip("-*").strip()
            for line in body.splitlines()
            if line.strip()
        ][:8]
        comment_block = "\n".join(f"    # {line}" for line in comments) or "    pass"
        focus_comment = f"    # focus: {focus}\n" if focus else ""
        scaffold = (
            "```python\n"
            "def predict_effect(action, observation):\n"
            f"{focus_comment}"
            f"{comment_block}\n"
            "    return {'expect_change': None, 'expected_diff_band': 'unknown', 'next_signature_hint': 'unknown'}\n"
            "```"
        )
        return f"{body}\n\n{scaffold}"

    @staticmethod
    def _extract_code_block(body: str) -> str:
        matches = re.findall(r"```python\s*(.*?)```", body, flags=re.DOTALL | re.IGNORECASE)
        for block in matches:
            if "def predict_effect" in block:
                return block.strip()
        if matches:
            return matches[0].strip()
        return body if "def predict_effect" in body else ""

    @staticmethod
    def _legacy_rule_to_predict_effect(body: str, focus: str) -> str:
        body = body.strip()
        if not body or "def predict_effect" in body:
            return ""
        code = WorldModelModule._extract_code_block(body)
        source = code or body
        normalized = " ".join(source.split())
        action_match = re.search(r"action\s*==\s*['\"](ACTION[1-7]|RESET)['\"]", normalized, flags=re.IGNORECASE)
        if not action_match:
            return ""
        target_action = action_match.group(1).upper()
        lower = normalized.lower()
        expect_change: bool | None
        expected_band = "unknown"
        if "expect no change" in lower or "no change" in lower:
            expect_change = False
            expected_band = "zero"
        elif "expect change" in lower or "change" in lower:
            expect_change = True
            expected_band = "small"
        else:
            expect_change = None
        focus_comment = f"    # focus: {focus}\n" if focus else ""
        return (
            "```python\n"
            "def predict_effect(action, observation):\n"
            f"{focus_comment}"
            f"    if action == {target_action!r}:\n"
            f"        return {{'expect_change': {expect_change!r}, 'expected_diff_band': {expected_band!r}, 'next_signature_hint': 'unknown'}}\n"
            "    return {'expect_change': None, 'expected_diff_band': 'unknown', 'next_signature_hint': 'unknown'}\n"
            "```"
        )

    def _heuristic_predict_effect_from_text(
        self,
        body: str,
        action_name: str,
        before: Any,
    ) -> dict[str, Any] | None:
        upgraded = self._legacy_rule_to_predict_effect(body, "")
        if not upgraded:
            return None
        code = self._extract_code_block(upgraded)
        if not code:
            return None
        globals_dict = {"__builtins__": {"dict": dict}}
        locals_dict: dict[str, Any] = {}
        try:
            exec(code, globals_dict, locals_dict)
        except Exception:
            return None
        fn = locals_dict.get("predict_effect") or globals_dict.get("predict_effect")
        if not callable(fn):
            return None
        try:
            result = fn(action_name, self._make_simulator_observation(before))
        except Exception:
            return None
        return result if isinstance(result, dict) else None

    def _compute_recent_delta_hints(self) -> dict[str, Any]:
        """v3.13: compute per-action aggregate cell-change hints from
        the last N unit tests, so predict_effect can stop guessing
        `expected_diff_cells=1` and instead cite empirical data."""
        from collections import defaultdict
        agg: dict[str, dict[str, Any]] = defaultdict(lambda: {
            "n": 0, "total_cells": 0.0, "sample_changes": [],
        })
        for t in self.unit_tests[-32:]:
            a = t.get("action", "?")
            agg[a]["n"] += 1
            agg[a]["total_cells"] += float(t.get("diff_magnitude", 0))
            # Capture up to 3 sample (x,y,new_val) changes for this action.
            before = t.get("before_grid", [])
            after = t.get("after_grid", [])
            if len(agg[a]["sample_changes"]) < 3 and before and after:
                changes = []
                for y in range(min(len(before), len(after))):
                    for x in range(min(len(before[y]), len(after[y]))):
                        if int(before[y][x]) != int(after[y][x]):
                            changes.append([x, y, int(after[y][x])])
                            if len(changes) >= 6:
                                break
                    if len(changes) >= 6:
                        break
                if changes:
                    agg[a]["sample_changes"].append(changes)
        # Finalize stats
        out: dict[str, Any] = {}
        for a, st in agg.items():
            if st["n"] == 0:
                continue
            out[a] = {
                "n": st["n"],
                "mean_diff_cells": round(st["total_cells"] / st["n"], 1),
                "sample_changes": st["sample_changes"][:3],
            }
        return out

    def _make_simulator_observation(self, frame: Any) -> dict[str, Any]:
        """Symbolica-unified simulator observation.

        The LLM reads both (a) a compact feature vector for fast rules and
        (b) a Symbolica-style text render of the full grid so that any
        `predict_effect` code can condition on actual shapes (e.g. player
        color, wall pattern), not just abstract signatures.
        """
        grid = current_grid(frame)
        available = list(getattr(frame, "available_actions", []) or [])
        # Full-grid text render (matches agent.py's frame_render tool).
        try:
            from agents.templates.agentica.scope.frame import Frame
            rich = Frame(frame, prev_levels_completed=None)
            grid_text = rich.render(y_ticks=False, x_ticks=False)
        except Exception:
            # Defensive: if Frame() fails (e.g. empty frame.frame), still
            # give the FULL grid (no truncation) with the same keys/gap as
            # Symbolica Frame.render so predict_effect sees the exact same
            # picture the LLM sees.
            KEYS = "0123456789abcdef"
            grid_text = "\n".join(
                " ".join(KEYS[int(c) % 16] for c in row) for row in grid
            )
        return {
            "signature": self._signature(frame),
            "family": self._family_id(frame),
            "grid": grid,
            "grid_text": grid_text,
            "shape": (len(grid), len(grid[0]) if grid else 0),
            "features": self._features(frame),
            "available_actions": available,
            "recent_actions": [action for action, _diff in self.recent_outcomes[-8:]],
            "recent_outcomes": [
                {"action": action, "diff_band": self._diff_band(diff), "diff_magnitude": float(diff)}
                for action, diff in self.recent_outcomes[-8:]
            ],
            # v3.13: empirical per-action delta hints (mean_diff_cells +
            # sample (x,y,new_val) lists). predict_effect can read these
            # to stop emitting 1-cell guesses when actions actually change
            # ~42 cells.
            "per_action_delta_hints": self._compute_recent_delta_hints(),
        }

    @staticmethod
    def _diff_band(diff: float) -> str:
        if diff <= 0:
            return "zero"
        if diff <= 4:
            return "small"
        return "large"

    def _signature(self, frame: Any) -> str:
        mode = str(self.params.get("signature_mode", "full_grid_hash"))
        if mode not in {"full_grid_hash", "observable_generalized"}:
            return grid_signature(frame)
        return grid_signature(frame)

    def _family_id(self, frame: Any) -> str:
        features = self._features(frame)
        if not features:
            return "empty"
        bins = []
        for value in features[: min(len(features), 12)]:
            bins.append(str(min(4, max(0, int(value * 5)))))
        return "f:" + "".join(bins)

    def _features(self, frame: Any) -> list[float]:
        return grid_feature_vector(frame, bands=int(self.params.get("feature_bands", 8)))

    def _nearest_bucket(self, frame: Any) -> dict[str, dict[str, float]]:
        if not bool(self.params.get("action_prior_generalization_enabled", True)):
            return {}
        if not self.signature_features:
            return {}

        current = self._features(frame)
        max_distance = float(self.params.get("generalize_max_distance", 0.18))
        max_neighbors = int(self.params.get("generalize_max_neighbors", 6))

        ranked: list[tuple[float, str]] = []
        for signature, features in self.signature_features.items():
            if signature not in self.transitions:
                continue
            dist = self._feature_distance(current, features)
            if dist <= max_distance:
                ranked.append((dist, signature))
        if not ranked:
            return {}

        ranked.sort(key=lambda item: item[0])
        aggregate: dict[str, dict[str, float]] = {}
        for dist, signature in ranked[:max_neighbors]:
            weight = 1.0 / max(0.05, dist + 0.05)
            for action, stats in self.transitions.get(signature, {}).items():
                bucket = aggregate.setdefault(
                    action,
                    {
                        "count": 0.0,
                        "avg_diff_num": 0.0,
                        "zero_count": 0.0,
                        "new_signature_count": 0.0,
                        "new_family_count": 0.0,
                        "branch_escape_count": 0.0,
                        "falsify_count": 0.0,
                        "support_count": 0.0,
                        "repeat_depth_sum": 0.0,
                    },
                )
                count = float(stats.get("count", 0.0))
                avg_diff = float(stats.get("avg_diff", 0.0))
                zero_count = float(stats.get("zero_count", 0.0))
                bucket["count"] += weight * count
                bucket["avg_diff_num"] += weight * count * avg_diff
                bucket["zero_count"] += weight * zero_count
                for key in (
                    "new_signature_count",
                    "new_family_count",
                    "branch_escape_count",
                    "falsify_count",
                    "support_count",
                    "repeat_depth_sum",
                ):
                    bucket[key] += weight * float(stats.get(key, 0.0))

        finalized: dict[str, dict[str, float]] = {}
        for action, stats in aggregate.items():
            count = float(stats.get("count", 0.0))
            if count <= 0:
                continue
            finalized[action] = {
                "count": count,
                "avg_diff": float(stats.get("avg_diff_num", 0.0)) / count,
                "zero_count": float(stats.get("zero_count", 0.0)),
                "new_signature_count": float(stats.get("new_signature_count", 0.0)),
                "new_family_count": float(stats.get("new_family_count", 0.0)),
                "branch_escape_count": float(stats.get("branch_escape_count", 0.0)),
                "falsify_count": float(stats.get("falsify_count", 0.0)),
                "support_count": float(stats.get("support_count", 0.0)),
                "repeat_depth_sum": float(stats.get("repeat_depth_sum", 0.0)),
            }
        return finalized

    @staticmethod
    def _feature_distance(left: list[float], right: list[float]) -> float:
        if not left or not right or len(left) != len(right):
            return 1.0
        return sum(abs(a - b) for a, b in zip(left, right)) / len(left)

    @staticmethod
    def _normalize_prediction_payload(action: str, raw: Any) -> Prediction:
        if isinstance(raw, Prediction):
            return raw
        if isinstance(raw, str):
            return Prediction(action=action, expected_change_summary=raw)
        if not isinstance(raw, dict):
            return Prediction(action=action, expected_change_summary=str(raw))

        expect_change = raw.get("expect_change")
        focus = str(raw.get("focus", "")).strip()
        note = str(raw.get("note", "")).strip()
        details = " ".join(part for part in [focus, note] if part).strip()

        if expect_change is True:
            summary = f"EXPECT_CHANGE: {details}".strip()
        elif expect_change is False:
            summary = f"EXPECT_NO_CHANGE: {details}".strip()
        else:
            summary = details

        return Prediction(
            action=action,
            expected_change_summary=summary,
            observation_prediction=str(raw.get("observation_prediction", "")).strip(),
            progress_prediction=str(raw.get("progress_prediction", "")).strip(),
            action_recommendation=str(raw.get("action_recommendation", "")).strip(),
            recommendation_rationale=str(raw.get("recommendation_rationale", "")).strip(),
            rival_predictions=[
                str(item).strip()
                for item in (raw.get("rival_predictions") or [])
                if str(item).strip()
            ],
            expected_signature=(
                str(raw.get("expected_signature")).strip()
                if raw.get("expected_signature") is not None
                else None
            ),
        )

    def _local_attractor_pressure(self) -> int:
        observation = self.bridge.last_observation
        if observation is None:
            return 0
        return max(0, observation.repeated_family_depth - 1)
