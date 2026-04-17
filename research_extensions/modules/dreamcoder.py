"""DreamCoder-style module.

Philosophy (Ellis et al. 2021 adapted):
- The library is grown by the agent itself. The agent chooses skill names,
  descriptions, trigger conditions, and bodies. We never force a template.
- A skill body may contain primitive actions or references to other skills
  the agent has already proposed, which gives a natural hierarchy.
- Sleep phase is triggered by surprise events on the shared bridge.
  When reality diverges from prediction, DreamCoder wakes up and:
    1. Consolidates recent trajectories around the surprise.
    2. Surfaces abstraction candidates to the next agent turn.
    3. Prunes skills that have repeatedly failed.

This module is fully independent: it can run alone, or alongside the
world-model module. The two communicate only via the shared bridge.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from ..bridge import (
    Observation,
    Prediction,
    ProposedSkill,
    SharedBridge,
    SkillExecutionState,
    SurpriseEvent,
)
from ..grid_utils import grid_diff_magnitude
from .seed_library import abstract_primitive_seeds


# DreamCoder-style skill kinds. Free-form payloads default to "" so old
# library snapshots remain forward-compatible — the field is only read,
# never required.
SKILL_KIND_ABSTRACT = "abstract_primitive"
SKILL_KIND_WRAPPER = "wrapper"
SKILL_KIND_EXACT = "exact_spine"


@dataclass(slots=True)
class SkillRecord:
    """A skill is whatever the agent said it is. We only stamp metadata."""
    payload: dict[str, Any]
    times_referenced: int = 0
    times_linked_to_surprise: int = 0
    observed_support: int = 0
    informative_hits: int = 0
    novel_signature_hits: int = 0
    novel_family_hits: int = 0
    branch_escape_hits: int = 0
    surprise_recovery_hits: int = 0
    falsification_utility_hits: int = 0
    times_selected: int = 0
    reset_intercepts: int = 0
    last_matched_trajectory: list[str] = field(default_factory=list)
    revision_count: int = 0
    depth: int = 0  # LLM can set this; otherwise we infer from body references
    created_at: float = field(default_factory=time.time)
    last_updated_at: float = field(default_factory=time.time)

    @property
    def kind(self) -> str:
        return str(self.payload.get("kind", "")).strip()

    def is_abstract_primitive(self) -> bool:
        return self.kind == SKILL_KIND_ABSTRACT

    def score(self) -> float:
        # Abstract primitives carry a small Bayesian-prior bonus so the
        # seeded library entries (BFS, button-semantics, rival probe) are
        # not pruned away before they get reused. This mirrors DreamCoder
        # treating library primitives as having prior support irrespective
        # of in-run usage.
        prior = 1.5 if self.is_abstract_primitive() else 0.0
        return (
            2.0 * self.times_referenced
            + 0.75 * self.observed_support
            + 1.0 * self.informative_hits
            + 2.0 * self.novel_signature_hits
            + 3.0 * self.novel_family_hits
            + 3.0 * self.branch_escape_hits
            + 2.0 * self.surprise_recovery_hits
            + 2.5 * self.falsification_utility_hits
            + 0.5 * self.times_linked_to_surprise
            + 0.75 * self.times_selected
            + 0.5 * self.reset_intercepts
            + 0.25 * self.depth
            + 0.25 * self.revision_count
            + prior
        )


class DreamCoderModule:
    name = "dreamcoder"

    def __init__(self, params: dict[str, Any], context, bridge: SharedBridge) -> None:
        self.params = params
        self.context = context
        self.bridge = bridge
        self.max_skills = int(params.get("max_skills", 64))
        self.body_preview_items = int(params.get("body_preview_items", 32))
        self.sleep_trajectory_window = int(params.get("sleep_trajectory_window", 16))
        self.surprise_match_suffix = int(params.get("surprise_match_suffix", 4))
        self.max_skills_in_prompt = int(params.get("max_skills_in_prompt", 8))
        self.proposal_cadence = int(params.get("proposal_cadence", 4))
        self.routing_min_score = float(params.get("routing_min_score", 0.75))
        self.routing_max_skills = int(params.get("routing_max_skills", 6))
        self.reset_intercept_enabled = bool(params.get("reset_intercept_enabled", True))
        self.route_on_stall_enabled = bool(params.get("route_on_stall_enabled", False))
        self.stall_window = int(params.get("stall_window", 4))
        self.revision_overlap_threshold = float(
            params.get("revision_overlap_threshold", 0.6)
        )
        self.revision_text_overlap_threshold = float(
            params.get("revision_text_overlap_threshold", 0.34)
        )
        self.skills: list[SkillRecord] = []
        self._proposal_cursor = 0
        self._surprise_cursor = 0
        self._pending_sleep_cues: list[str] = []
        self._last_sleep_at: float = 0.0
        self._informative_actions_since_proposal: int = 0
        self._recent_actions: list[str] = []
        self._latest_route_hint: str | None = None
        self._memories: Any | None = None
        # Seeding is opt-in: production configs (all_on_generalized.yaml)
        # turn it on, while tests that assert exact library shapes keep
        # the default off and start with an empty library.
        self.seed_abstract_primitives = bool(
            params.get("seed_abstract_primitives", False)
        )
        self._state_path = context.shared_dir / "dreamcoder_library.json"
        if self._state_path.exists():
            try:
                raw = json.loads(self._state_path.read_text(encoding="utf-8"))
                self.skills = [SkillRecord(**item) for item in raw]
                self._synchronize_wrappers_for_spines()
            except Exception:
                self.skills = []
        if self.seed_abstract_primitives:
            self._install_abstract_primitive_seeds()
        self._publish_skill_context()

    # -- Hook surface -----------------------------------------------------
    def prompt_overlay(self) -> str:
        lines: list[str] = []
        # [Observed Laws] — most actionable section, surfaced FIRST so the
        # LLM sees consolidated empirical knowledge before proposing new
        # hypotheses. Each law compresses n-observation statistics into a
        # single implication string.
        laws = [r for r in self.skills if str(r.payload.get("kind", "")) == "observed_law"]
        if laws:
            lines.append("[Observed Laws] (consolidated from real transitions — trust these over guesses)")
            for record in sorted(
                laws,
                key=lambda r: -int(r.payload.get("observation_summary", {}).get("n", 0)),
            )[:6]:
                p = record.payload
                obs = p.get("observation_summary", {})
                lines.append(
                    f"- {p.get('name', '')}: n={obs.get('n',0)} mean_diff={obs.get('mean_diff',0):.1f} "
                    f"new_family={obs.get('new_family_rate',0)*100:.0f}% "
                    f"branch_escape={obs.get('branch_escape_rate',0)*100:.0f}%"
                )
                impl = str(p.get("implication", "")).strip()
                if impl:
                    lines.append(f"  → {impl}")
            lines.append(
                "Use the laws above to constrain your next hypothesis: if all directional "
                "actions show ≤5% new_family rate, directional keys alone are not the "
                "escape — look for state-dependent triggers, order-dependent sequences, "
                "or states where the same action behaves differently."
            )
            lines.append("")
        lines.append("[Skill Library]")
        primitives = [r for r in self.skills if r.is_abstract_primitive()]
        if primitives:
            lines.append(
                "[Abstract Primitives] (always available — compose with these by "
                "name in `subskills` or by referencing `skill:<name>` in a body)"
            )
            for record in sorted(
                primitives,
                key=lambda r: str(r.payload.get("name", "")),
            )[:8]:
                payload = record.payload
                name = str(payload.get("name", "")).strip()
                description = str(payload.get("description", "")).strip()
                lines.append(
                    f"- {name} (kind=abstract_primitive, refs={record.times_referenced}, "
                    f"selected={record.times_selected}): {description[:160]}"
                )
            lines.append(
                "Prefer composing higher-level skills from these primitives — "
                "for example, build a `solve-opening` skill whose `subskills` are "
                "[`skill:bfs-explore-grid`, `skill:discover-ACTION3-semantics`] "
                "instead of restating raw ACTIONk loops every run."
            )
            lines.append("[Free-form skills]")
        if not self.skills:
            lines.append(
                "No skills yet. If you notice a useful pattern, you may propose one."
            )
        else:
            free_form = [r for r in self.skills if not r.is_abstract_primitive()]
            for record in sorted(
                free_form,
                key=lambda r: (-r.score(), -r.last_updated_at),
            )[: self.max_skills_in_prompt]:
                payload = record.payload
                name = str(payload.get("name", "unnamed")).strip()
                description = str(payload.get("description", "")).strip()
                body_str = self._payload_preview(payload)
                lines.append(
                    f"- {name} (score={record.score():.1f}, depth={record.depth}, refs={record.times_referenced}, support={record.observed_support}, selected={record.times_selected}, reset_saves={record.reset_intercepts}, revs={record.revision_count}): {description}"
                )
                if body_str:
                    lines.append(f"  body: {body_str}")
                if record.last_matched_trajectory:
                    lines.append(
                        f"  recent_support: {record.last_matched_trajectory[-self.surprise_match_suffix:]}"
                    )
        lines.append(
            "You may propose a new skill or reference an existing one in the body of a new skill. "
            "You choose the name, description, trigger, and body — no fixed schema."
        )
        lines.append(
            "Prefer names and descriptions grounded in observable behavior, object relations, "
            "or state-conditioned tactics. Avoid naming a skill after a level number or "
            "a one-off solve trace if the same idea could be described more generally."
        )
        lines.append(
            "If a recent surprise shows that an earlier skill was incomplete, revise or "
            "supersede that abstraction instead of only adding another near-duplicate label."
        )
        lines.append(
            "Do not wait for a perfect theory. If a short action pattern, recovery tactic, or solved subgoal "
            "already looks reusable, propose or revise one skill now and let later surprises refine it."
        )
        lines.append(
            "Prefer several small observable tactics over one omnibus sweep. If two action families, trigger conditions, "
            "or outcomes look meaningfully different, keep them as separate skills instead of compressing them into one generic loop."
        )
        lines.append(
            "If a short concrete action spine actually worked or got farther than a generic probe, preserve it as evidence or a leaf fallback primitive. "
            "Do not let the exact replay become the main reusable abstraction when you can describe a more general precondition, controller, and expected effect one layer up."
        )
        lines.append(
            "When an exact spine looks valuable, extract one layer up: write a wrapper skill with an observable precondition, "
            "a compact controller, an expected effect, and optionally `subskills` that point at the exact reusable spine as an example-backed leaf."
        )
        lines.append(
            "If you are explicitly revising an earlier skill, you may point to it with "
            "`supersedes` or `revises` using that skill's name."
        )
        lines.append(
            "If you want a skill to be reusable and scoreable later, include explicit action names "
            "or `skill:...` references somewhere in its body, rule, policy, or steps."
        )
        lines.append(
            "If you cannot yet cite an explicit action spine, do not promote the skill yet; first test until you can write one."
        )
        lines.append(
            "Prefer skills that help distinguish rival hypotheses, reach unseen observations, escape a repeated family, or recover after surprise."
        )
        lines.append(
            "Before proposing or revising a skill, check `memories.summaries()` and, if needed, "
            "`await memories.query(...)` so you extend or refine what is already known instead of "
            "creating a duplicate label for the same tactic."
        )
        lines.append(
            "Before resetting or launching a brand-new probe sequence, inspect `research.skills()` "
            "and `research.read_skill(name)` for reusable tactics already in the library."
        )
        lines.append(
            "Bare labels or description-only sketches are weak. Prefer a compact controller, "
            "decision rule, or action-grounded procedure that another run could actually reuse."
        )
        lines.append(
            "Good reusable skills usually specify a precondition, a short controller/body, an expected effect, "
            "and optionally a recovery branch if the expected effect fails."
        )
        world_hint = self.bridge.read_hint("world_model")
        if isinstance(world_hint, dict):
            focus = str(world_hint.get("best_focus", "")).strip()
            suggested = str(world_hint.get("suggested_action", "")).strip()
            rationale = str(world_hint.get("hint", "")).strip()
            surprise = bool(world_hint.get("recent_surprise"))
            lines.append("[World-model cues]")
            if focus:
                lines.append(
                    f"- Current model focus: {focus}. Prefer skill names and preconditions that explain this observable regularity or failure mode."
                )
            if suggested:
                lines.append(
                    f"- Current world prior favors {suggested}. If you promote a skill around this region, state when following {suggested} is justified and when to branch away."
                )
            if rationale:
                lines.append(f"- Prior rationale: {rationale}")
            if surprise:
                lines.append(
                    "- A recent prediction mismatch occurred. Prefer revising the closest existing skill or adding a recovery branch for that mismatch."
                )
        lines.append("[Example skill shapes]")
        lines.append(
            "- Example: name='Edge-confirm then commit'; precondition='two recent ACTION3 moves both changed the grid'; "
            "controller=['ACTION4','ACTION1','ACTION1','ACTION2']; expected_effect='reaches a checkpoint without reset'."
        )
        lines.append(
            "- Example: name='Surprise-driven detour'; trigger='predicted change failed after ACTION2'; "
            "controller=['ACTION3','ACTION1']; expected_effect='switches to a different corridor hypothesis'."
        )
        lines.append(
            "- Example: name='Rival discriminator'; precondition='two explanations remain plausible'; "
            "controller='take the action that would create an unseen observation or falsify the stronger explanation'; "
            "expected_effect='separates the rival hypotheses instead of merely confirming the familiar branch'."
        )
        lines.append(
            "- Example: name='Checkpoint macro'; precondition='frontier checkpoint already reached'; "
            "subskills=['skill:local-opening','skill:loop-breaker']; expected_effect='composes two local tactics into a longer tactic'."
        )
        lines.append(
            "- Example: name='Stable opening wrapper'; precondition='fresh board and no loop evidence yet'; "
            "subskills=['skill:exact-opening-spine']; controller='run the stored opening, then branch only if the observed effect is weaker than expected'; "
            "expected_effect='lands in the same checkpoint region more reliably than ad-hoc probing'."
        )
        if self.skills:
            lines.append(
                "If an earlier skill lacks an action-grounded body, revise that skill now instead "
                "of adding another abstract paraphrase."
            )
        if self._latest_route_hint:
            lines.append(f"Current skill-route hint: {self._latest_route_hint}")
        if self._informative_actions_since_proposal >= self.proposal_cadence:
            lines.append(
                "You have seen several informative transitions since the last skill update. "
                "Capture the most reusable pattern now, or revise the closest existing skill before taking many more actions."
            )
            lines.append(
                "If you have already captured one pattern from this area, look for a second distinct local rule with a different trigger or outcome before continuing the same probe loop."
            )
        if self._pending_sleep_cues:
            lines.append("[Sleep cues from recent surprises]")
            for cue in self._pending_sleep_cues[-4:]:
                lines.append(f"- {cue}")
            self._pending_sleep_cues = []
        return "\n".join(lines)

    def on_memories_ready(self, memories: Any) -> None:
        self._memories = memories if hasattr(memories, "add") else None

    def before_action(
        self, frame: Any | None, action_name: str, available_actions: list[str]
    ) -> str | None:
        suggestion = self._route_action(action_name, available_actions)
        if suggestion is None:
            self._latest_route_hint = None
            return action_name
        self._latest_route_hint = suggestion[1]
        return suggestion[0]

    def after_action(self, before: Any | None, action_name: str, after: Any) -> None:
        # Pick up any new skills the agent proposed since last turn.
        self._drain_new_proposals()
        self._score_skills_from_observation(before, action_name, after)
        self._publish_skill_context()

        # Check the bridge for new surprise events and enter sleep on them.
        new_surprises = self.bridge.surprises[self._surprise_cursor :]
        if new_surprises:
            self._surprise_cursor = len(self.bridge.surprises)
            for event in new_surprises:
                self._sleep_on_surprise(event)
        self._publish_skill_context()

    def on_run_end(
        self,
        *,
        history: list[tuple[str, Any]],
        finish_status: Any,
        workdir: Path,
    ) -> None:
        self._drain_new_proposals()
        self._promote_validated_exact_spine(history, finish_status)
        self._consolidate_and_prune()
        self._publish_skill_context()
        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(
            json.dumps([asdict(r) for r in self.skills], indent=2),
            encoding="utf-8",
        )

    # -- Agent tool ---------------------------------------------------------
    def record_agent_proposal(self, payload: dict[str, Any]) -> ProposedSkill:
        """The agent's LLM response contained a propose_skill block."""
        return self.bridge.record_proposed_skill(payload)

    def list_skills(self) -> list[dict[str, Any]]:
        return [
            {
                "name": str(record.payload.get("name", "")).strip(),
                "description": str(record.payload.get("description", "")).strip(),
                "trigger": str(record.payload.get("trigger", "")).strip(),
                "score": round(record.score(), 2),
                "selected": record.times_selected,
                "reset_saves": record.reset_intercepts,
                "body_preview": self._payload_preview(record.payload),
            }
            for record in sorted(
                self.skills,
                key=lambda r: (-r.score(), -r.last_updated_at),
            )
        ]

    def read_skill(self, name: str) -> dict[str, Any] | None:
        record = self._find_by_name(name.strip())
        if record is None:
            return None
        return {
            "payload": record.payload,
            "score": round(record.score(), 2),
            "times_referenced": record.times_referenced,
            "observed_support": record.observed_support,
            "informative_hits": record.informative_hits,
            "times_selected": record.times_selected,
            "reset_intercepts": record.reset_intercepts,
            "revision_count": record.revision_count,
            "last_matched_trajectory": list(record.last_matched_trajectory),
        }

    # -- v3 option execution (plan §4 P0) -----------------------------------
    def commit_skill(self, skill_record: "SkillRecord", source: str = "agent") -> SkillExecutionState | None:
        """Start executing `skill_record` as an option in the real env.

        Returns the SkillExecutionState already pushed onto the bridge, or
        None if the skill has no executable body."""
        body = self._extract_executable_body(skill_record.payload)
        if not body:
            return None
        state = SkillExecutionState(
            skill_name=str(skill_record.payload.get("name", "(unnamed)")).strip() or "(unnamed)",
            body=body,
            source=source,
        )
        self.bridge.commit_skill(state)
        skill_record.times_selected += 1
        skill_record.last_updated_at = time.time()
        return state

    def next_skill_action(self) -> str | None:
        """Return the next ACTION token if a skill is in execution, else None.

        Plan §4: registry uses this to bypass the LLM during option
        execution. Aborted options return None and clear the bridge slot."""
        state = self.bridge.current_skill_in_execution
        if state is None or state.is_done:
            if state is not None and state.is_done:
                self.bridge.clear_committed_skill()
            return None
        return state.next_action()

    def on_skill_step(
        self,
        *,
        predicted: dict[str, Any] | None,
        observation: Observation,
    ) -> tuple[bool, str | None]:
        """After each real action while a skill is executing, check for
        multi-head surprise (plan H6). On abort, queue a wake trigger and
        clear the bridge.

        Returns (aborted, reason)."""
        state = self.bridge.current_skill_in_execution
        if state is None:
            return False, None
        # Advance cursor for the action that was just taken (the registry
        # called this AFTER submit_action returned).
        state.advance()
        head_disagreements = self._count_head_disagreements(predicted, observation)
        if predicted is not None and head_disagreements >= 2:
            state.surprises_during += 1
            reason = f"multi-head surprise ({head_disagreements} heads disagreed)"
            self.bridge.clear_committed_skill(reason=reason)
            # Queue a wake trigger so the LLM rewrites predict_effect.
            ev = SurpriseEvent(
                action=observation.action,
                predicted=Prediction(
                    action=observation.action,
                    expected_change_summary=str(predicted.get("expect_change", "")),
                    observation_prediction=str(predicted.get("observation_prediction", "")),
                    progress_prediction=str(predicted.get("progress_prediction", "")),
                    expected_signature=predicted.get("next_signature_hint"),
                ) if predicted else None,
                observation=observation,
                recent_trajectory=list(self.bridge.recent_actions),
            )
            self.bridge.queue_wake_trigger(ev)
            return True, reason
        if state.is_done:
            self.bridge.clear_committed_skill()
            return False, "skill_completed"
        return False, None

    def propose_from_mcts(
        self,
        plan_result: dict[str, Any],
        *,
        root_state_signature: str,
        bfs_baseline: float,
    ) -> list["SkillRecord"]:
        """Convert MCTS top paths into skill records (plan §4 P0).

        Plan H5 / Q4: only propose paths whose simulator reward exceeds
        BFS baseline + 0.5 std. Plan §4 conflict resolution: same-name
        agent-proposed skills get arbitrated by simulator reward — the
        loser is tagged `superseded_by`."""
        added: list[SkillRecord] = []
        if not plan_result.get("gate_ok"):
            return added
        bfs_paths = plan_result.get("bfs_top_paths", [])
        if bfs_paths:
            rewards = [r for _seq, r in bfs_paths]
            mean = sum(rewards) / len(rewards)
            var = sum((r - mean) ** 2 for r in rewards) / len(rewards)
            std = var ** 0.5
            threshold = bfs_baseline + 0.5 * std
        else:
            threshold = bfs_baseline
        for seq, reward, visits in plan_result.get("mcts_top_paths", []):
            if reward <= threshold:
                continue
            if not seq:
                continue
            name = self._mcts_skill_name(seq, root_state_signature)
            payload = {
                "name": name,
                "kind": "mcts_proposal",
                "description": (
                    f"MCTS-discovered path with simulator reward {reward:.2f} "
                    f"(visits={visits}). Auto-proposed; verify on real env."
                ),
                "precondition": (
                    f"State signature == {root_state_signature[:8]}.. and the "
                    "first action of body is available."
                ),
                "controller": list(seq),
                "body": list(seq),
                "action_spine": list(seq),
                "expected_effect": (
                    f"In simulation: simulator reward {reward:.2f} over "
                    f"{len(seq)} steps. Real env may diverge — abort on "
                    "multi-head surprise."
                ),
                "mcts_simulator_reward": reward,
                "mcts_visits": visits,
            }
            existing = self._find_by_name(name)
            if existing is not None:
                self._resolve_skill_conflict(existing, payload, reward)
                continue
            record = SkillRecord(payload=payload, depth=0)
            record.observed_support = 1  # synthesised "1 simulator support"
            self.skills.append(record)
            self._remember_skill(record, event="mcts_proposed")
            added.append(record)
        return added

    @staticmethod
    def _mcts_skill_name(seq: list[str], root_signature: str) -> str:
        head = "-".join(seq[:4])
        return f"mcts:{root_signature[:6]}:{head}"

    def _resolve_skill_conflict(
        self,
        existing: "SkillRecord",
        new_payload: dict[str, Any],
        new_reward: float,
    ) -> None:
        """Plan §4 conflict resolution: keep the higher-reward winner;
        tag the loser with `superseded_by`."""
        existing_reward = float(existing.payload.get("mcts_simulator_reward", -float("inf")))
        if new_reward > existing_reward:
            existing.payload["superseded_by"] = new_payload["name"]
            existing.payload = new_payload
            existing.last_updated_at = time.time()
        else:
            new_payload["superseded_by"] = existing.payload.get("name", "")
            # Do not insert the loser; we keep only one entry per name.

    def _count_head_disagreements(
        self,
        predicted: dict[str, Any] | None,
        observation: Observation,
    ) -> int:
        """Count how many of {expect_change, expected_diff_band,
        next_signature_hint, progress_prediction} disagree with the
        observation. Used for multi-head (≥2) option-termination per H6.

        We deliberately do NOT use this for transition_accuracy scoring —
        that stays single-head per the world-model unit-test logic."""
        if predicted is None:
            return 0
        n = 0
        # 1. expect_change
        ec = predicted.get("expect_change")
        if ec is not None:
            actual_change = observation.diff_magnitude > 0
            if bool(ec) != actual_change:
                n += 1
        # 2. expected_diff_band
        band = str(predicted.get("expected_diff_band", "")).lower()
        if band and band != "unknown":
            actual_band = (
                "zero" if observation.diff_magnitude <= 0
                else "small" if observation.diff_magnitude <= 4
                else "large"
            )
            if band != actual_band:
                n += 1
        # 3. next_signature_hint
        sig_hint = predicted.get("next_signature_hint")
        if isinstance(sig_hint, str) and sig_hint and sig_hint != "unknown":
            if sig_hint != observation.after_signature:
                n += 1
        # 4. progress_prediction (only counts if explicit)
        progress = str(predicted.get("progress_prediction", "")).lower()
        if "branch_escape" in progress and not observation.branch_escape:
            n += 1
        elif "new_family" in progress and not observation.new_family:
            n += 1
        return n

    def consolidate_experience(self, world_model=None) -> dict[str, Any]:
        """Compress accumulated surprises + transition statistics into
        one or more `observed-law:*` skills (plan §3 sleep + plan §5
        scientist control state).

        Problem this fixes: without consolidation, the agent re-discovers
        the same lesson every 10-20 turns. 79 surprises in v3-full all
        had `observed_diff ≈ 52` yet world drafts cycled through 4
        different incorrect theories.

        Output: a skill payload keyed by action-family invariant that
        summarises (n, mean_diff, new_family_rate, family_stability).
        The summary is then surfaced to the agent via
        `memories.add([Observed Law] ...)` so the next turn sees it.
        """
        summary: dict[str, Any] = {"laws_added": 0, "laws_updated": 0}
        if world_model is None:
            return summary
        transitions = getattr(world_model, "transitions", {})
        if not transitions:
            return summary
        # Flatten: per-action stats aggregated across all state
        # signatures.
        from collections import defaultdict
        agg: dict[str, dict[str, float]] = defaultdict(lambda: {
            "count": 0.0, "avg_diff_sum": 0.0, "new_family_count": 0.0,
            "branch_escape_count": 0.0, "zero_count": 0.0,
        })
        for state_sig, bucket in transitions.items():
            for action, stats in bucket.items():
                count = float(stats.get("count", 0.0))
                if count <= 0:
                    continue
                agg[action]["count"] += count
                agg[action]["avg_diff_sum"] += float(stats.get("avg_diff", 0.0)) * count
                agg[action]["new_family_count"] += float(stats.get("new_family_count", 0.0))
                agg[action]["branch_escape_count"] += float(stats.get("branch_escape_count", 0.0))
                agg[action]["zero_count"] += float(stats.get("zero_count", 0.0))
        if not agg:
            return summary
        # Build a single consolidated payload per action with count >= 3
        for action, st in agg.items():
            n = st["count"]
            if n < 3:
                continue
            mean_diff = st["avg_diff_sum"] / n
            new_family_rate = st["new_family_count"] / n
            branch_escape_rate = st["branch_escape_count"] / n
            zero_rate = st["zero_count"] / n
            name = f"observed-law:{action}"
            implication = self._law_implication(
                action=action,
                mean_diff=mean_diff,
                new_family_rate=new_family_rate,
                branch_escape_rate=branch_escape_rate,
                zero_rate=zero_rate,
                n=int(n),
            )
            payload = {
                "name": name,
                "kind": "observed_law",
                "description": (
                    f"Empirical law for {action} over {int(n)} observations: "
                    f"mean_diff={mean_diff:.1f}, new_family_rate={new_family_rate:.2%}, "
                    f"branch_escape_rate={branch_escape_rate:.2%}, "
                    f"zero_rate={zero_rate:.2%}."
                ),
                "precondition": f"about to take {action} from an already-observed state signature",
                "action_spine": [action],
                "body": [action],
                "observation_summary": {
                    "n": int(n),
                    "mean_diff": round(mean_diff, 2),
                    "new_family_rate": round(new_family_rate, 3),
                    "branch_escape_rate": round(branch_escape_rate, 3),
                    "zero_rate": round(zero_rate, 3),
                },
                "implication": implication,
                "confidence": min(1.0, n / 10.0),
            }
            existing = self._find_by_name(name)
            if existing is not None:
                existing.payload = payload
                existing.last_updated_at = time.time()
                existing.revision_count += 1
                summary["laws_updated"] += 1
            else:
                record = SkillRecord(payload=payload, depth=1)
                record.observed_support = int(n)
                self.skills.append(record)
                summary["laws_added"] += 1
            if self._memories is not None:
                self._memories.add(
                    f"[Observed Law] {action}",
                    (
                        f"n={int(n)} mean_diff={mean_diff:.1f} "
                        f"new_family_rate={new_family_rate:.2%} "
                        f"branch_escape_rate={branch_escape_rate:.2%}\n"
                        f"implication: {implication}"
                    ),
                )
        return summary

    @staticmethod
    def _law_implication(
        *, action: str, mean_diff: float, new_family_rate: float,
        branch_escape_rate: float, zero_rate: float, n: int,
    ) -> str:
        """Short natural-language implication the LLM can act on."""
        if zero_rate > 0.5:
            return f"{action} rarely changes the grid; prefer other actions or verify precondition."
        if new_family_rate < 0.05 and branch_escape_rate < 0.05 and n >= 5:
            return (
                f"{action} consistently lands in the same observation family (new_family "
                f"only {new_family_rate:.0%}, branch_escape {branch_escape_rate:.0%}). "
                "Continuing this action is unlikely to open new territory — try an unused "
                "action family or RESET-probe instead."
            )
        if new_family_rate > 0.3:
            return f"{action} often enters a new family ({new_family_rate:.0%}); preserve this as a branch-opening move."
        return f"{action} has mixed effects ({new_family_rate:.0%} new_family); further probing needed."

    def sleep_refactor(self) -> dict[str, Any]:
        """Plan H8 sleep refactor.

        (a) N-gram (prefix length ≥ 2) extraction across MCTS-proposed
            skills; create wrapper skills when ≥ 2 children share a prefix.
        (b) Demote skills with simulator_reward < 0 to the bottom ordering
            (they still exist as evidence but routing deprioritises them).
        (c) Preserve seeded abstract primitives (kind=abstract_primitive).
        (d) Cap library at `max_skills`, never evicting primitives.

        Returns counts for smoke-test gates."""
        counts = {"wrappers_created": 0, "demoted": 0, "pruned": 0, "library_size_before": len(self.skills)}
        # (c) partition — preserve primitives AND observed-law records
        def _is_preserved_sleep(r: SkillRecord) -> bool:
            return (
                r.is_abstract_primitive()
                or str(r.payload.get("kind", "")).strip() == "observed_law"
            )
        primitives = [r for r in self.skills if _is_preserved_sleep(r)]
        rest = [r for r in self.skills if not _is_preserved_sleep(r)]

        # (a) N-gram wrapper: group both MCTS-proposed AND agent-proposed
        # skills by length-2 prefix. v3.14 fix: previously only
        # mcts_proposal skills were wrappable so the 10 agent-proposed
        # free-form skills in v3.11 got no compression.
        from collections import defaultdict
        by_prefix: dict[tuple[str, str], list[SkillRecord]] = defaultdict(list)
        for r in rest:
            # Accept mcts_proposal, free-form, and wrapper candidates;
            # skip observed_law / abstract_primitive which already got
            # filtered into `primitives`.
            kind = str(r.payload.get("kind", "") or "")
            if kind in ("observed_law", "abstract_primitive"):
                continue
            body = self._extract_executable_body(r.payload)
            if len(body) >= 2:
                by_prefix[(body[0], body[1])].append(r)
        for prefix, members in by_prefix.items():
            if len(members) < 2:
                continue
            wrapper_name = f"wrapper:{prefix[0]}-{prefix[1]}"
            if self._find_by_name(wrapper_name) is not None:
                continue
            best_reward = max(
                float(m.payload.get("mcts_simulator_reward", 0.0)) for m in members
            )
            wrapper_payload = {
                "name": wrapper_name,
                "kind": "wrapper",
                "description": (
                    f"Auto-wrapper over {len(members)} MCTS-proposed skills that "
                    f"share prefix [{prefix[0]}, {prefix[1]}]."
                ),
                "precondition": (
                    f"First two available actions include {prefix[0]} and "
                    f"{prefix[1]}, and world-model recommends this opening."
                ),
                "action_spine": list(prefix),
                "body": list(prefix),
                "subskills": [f"skill:{m.payload.get('name')}" for m in members],
                "expected_effect": (
                    f"After running the shared prefix, branches into one of "
                    f"{len(members)} MCTS-validated continuations."
                ),
                "mcts_simulator_reward": best_reward,
            }
            wrapper = SkillRecord(payload=wrapper_payload, depth=1)
            wrapper.observed_support = 1
            rest.append(wrapper)
            counts["wrappers_created"] += 1

        # (b) demote low-reward
        for r in rest:
            reward = float(r.payload.get("mcts_simulator_reward", 0.0))
            if reward < 0:
                counts["demoted"] += 1
                # synthetic negative score: set times_selected to 0 and add
                # a tag so routing filters it out.
                r.payload["demoted"] = True

        # (d) cap
        rest.sort(
            key=lambda r: (
                # wrappers first (structural), then by reward
                -int(r.payload.get("kind") == "wrapper"),
                -float(r.payload.get("mcts_simulator_reward", 0.0)),
                -r.score(),
            )
        )
        budget = max(0, self.max_skills - len(primitives))
        kept_rest = rest[:budget]
        counts["pruned"] = len(rest) - len(kept_rest)
        self.skills = primitives + kept_rest
        counts["library_size_after"] = len(self.skills)
        return counts

    @staticmethod
    def _extract_executable_body(payload: dict[str, Any]) -> list[str]:
        """Return a flat list of ACTION tokens for option execution.
        Prefers `action_spine`, then `body`, then `controller`."""
        for key in ("action_spine", "body", "controller", "steps"):
            value = payload.get(key)
            if not value:
                continue
            tokens = DreamCoderModule._body_as_items(value)
            flat: list[str] = []
            for tok in tokens:
                refs = DreamCoderModule._inline_action_refs(tok)
                if refs:
                    for r in refs:
                        if r.startswith("ACTION") or r == "RESET":
                            flat.append(r)
                elif isinstance(tok, str) and (tok.startswith("ACTION") or tok == "RESET"):
                    flat.append(tok)
            if flat:
                return flat
        return []

    # -- Internals ----------------------------------------------------------
    def _drain_new_proposals(self) -> None:
        total = len(self.bridge.proposed_skills)
        if total == self._proposal_cursor:
            return
        new_items = self.bridge.proposed_skills[self._proposal_cursor :]
        self._proposal_cursor = total
        for proposal in new_items:
            payload = proposal.payload
            if not isinstance(payload, dict):
                continue
            payload = self._attach_recent_action_spine(payload)
            if not self._is_action_grounded_payload(payload):
                name = str(payload.get("name", "(unnamed)")).strip() or "(unnamed)"
                self._pending_sleep_cues.append(
                    f"Skipped ungrounded proposal '{name}' because it lacked an explicit action spine."
                )
                if self._memories is not None:
                    self._memories.add(
                        f"[Skill Draft Rejected] {name}",
                        json.dumps(payload, ensure_ascii=True),
                    )
                continue
            self._mark_referenced_skills(payload)
            depth = self._infer_depth(payload)
            name = str(payload.get("name", "")).strip()
            existing = self._find_by_name(name) if name else None
            explicit_revision = self._find_explicit_revision_target(payload)
            revision_target = existing or explicit_revision or self._find_revision_target(payload)
            if existing is not None:
                existing.payload = payload
                existing.depth = depth
                existing.last_updated_at = time.time()
                record = existing
                self._remember_skill(record, event="updated")
            elif revision_target is not None:
                revision_target.payload = payload
                revision_target.depth = depth
                revision_target.revision_count += 1
                revision_target.last_updated_at = time.time()
                record = revision_target
                self._pending_sleep_cues.append(
                    f"Revised existing skill '{payload.get('name', '(unnamed)')}' instead of increasing library size."
                )
                self._remember_skill(record, event="revised")
            else:
                record = SkillRecord(payload=payload, depth=depth)
                self.skills.append(record)
                self._remember_skill(record, event="added")
            self._pending_sleep_cues.append(
                f"Registered proposal '{payload.get('name', '(unnamed)')}' at depth {depth}."
            )
            self._informative_actions_since_proposal = 0

    def _sleep_on_surprise(self, event: SurpriseEvent) -> None:
        """Sleep phase: look at recent trajectory, bump related skills,
        queue a hint for the next prompt turn. The actual consolidation /
        abstraction is left to the next agent turn — the agent decides."""
        self._last_sleep_at = time.time()
        trajectory = event.recent_trajectory[-self.sleep_trajectory_window :]
        for record in self.skills:
            if self._matches_recent_suffix(self._execution_items(record.payload), trajectory):
                record.times_linked_to_surprise += 1
                record.last_updated_at = time.time()
        cue = (
            f"Surprise after {event.action} with recent trajectory {trajectory}. "
            "Consider proposing or revising a skill that captures this."
        )
        self._pending_sleep_cues.append(cue)

    def _consolidate_and_prune(self) -> None:
        """At the end of a run, drop skills that have not earned any use or
        surprise linkage, and keep the library bounded."""
        if not self.skills:
            return
        self._synchronize_wrappers_for_spines()
        # Always keep abstract-primitive seeds AND observed-law records —
        # they form the agent's composable vocabulary + consolidated
        # empirical knowledge. Pruning would force the next run to
        # re-discover them from scratch.
        def _is_preserved(r: SkillRecord) -> bool:
            return (
                r.is_abstract_primitive()
                or str(r.payload.get("kind", "")).strip() == "observed_law"
            )
        primitives = [r for r in self.skills if _is_preserved(r)]
        non_primitives = [r for r in self.skills if not _is_preserved(r)]
        non_primitives.sort(
            key=lambda r: (
                -int(bool(r.payload.get("action_spine"))),
                -self._structural_field_count(r.payload),
                -r.score(),
                len(self._execution_signature(r.payload)) or 999,
                r.created_at,
            )
        )
        keep: list[SkillRecord] = []
        seen_names: set[str] = set()
        seen_signatures: set[tuple[tuple[str, ...], str]] = set()
        for record in primitives + non_primitives:
            name = str(record.payload.get("name", "")).strip() or f"unnamed-{id(record)}"
            if name in seen_names:
                continue
            signature = self._execution_signature(record.payload)
            signature_kind = self._signature_kind(record.payload)
            signature_key = (signature, signature_kind)
            if (
                signature
                and signature_key in seen_signatures
                and not _is_preserved(record)
            ):
                continue
            seen_names.add(name)
            if signature:
                seen_signatures.add(signature_key)
            keep.append(record)
            if len(keep) >= self.max_skills and not any(
                r.is_abstract_primitive() and r not in keep for r in self.skills
            ):
                break
        self.skills = keep
        if self.seed_abstract_primitives:
            self._install_abstract_primitive_seeds()

    def _install_abstract_primitive_seeds(self) -> None:
        """Idempotently ensure the seeded abstract primitives are present.

        Seeds are kept *if and only if* they are missing from the current
        library — agent-revised versions are never overwritten. This keeps
        the wake-sleep loop free to refine BFS, button-semantics, etc.,
        while still guaranteeing the seeds exist on a fresh run."""
        existing_names = {
            str(r.payload.get("name", "")).strip() for r in self.skills
        }
        for payload in abstract_primitive_seeds():
            name = str(payload.get("name", "")).strip()
            if not name or name in existing_names:
                continue
            record = SkillRecord(
                payload=dict(payload),
                depth=int(payload.get("depth", 1)),
            )
            # Give seeds a tiny bit of synthetic support so they appear
            # in the prompt overlay on turn 1 instead of waiting for the
            # agent to reference them. Without this, the agent never sees
            # the seeded vocabulary on the very first turn and falls
            # back to ad-hoc probing.
            record.observed_support = 1
            self.skills.append(record)
            existing_names.add(name)

    def _remember_skill(self, record: SkillRecord, *, event: str) -> None:
        if self._memories is None:
            return
        payload = record.payload
        name = str(payload.get("name", "(unnamed)")).strip() or "(unnamed)"
        description = str(payload.get("description", "")).strip()
        body_preview = self._body_preview(payload.get("body", payload.get("policy", payload.get("steps", ""))))
        details = "\n".join(
            part
            for part in [
                f"event: {event}",
                f"description: {description}" if description else "",
                f"depth: {record.depth}",
                f"score: {record.score():.1f}",
                f"body: {body_preview}" if body_preview else "",
                f"payload: {json.dumps(payload, ensure_ascii=True)}",
            ]
            if part
        )
        self._memories.add(f"[Skill] {name}", details)

    def _publish_skill_context(self) -> None:
        top = sorted(
            self.skills,
            key=self._display_priority_key,
        )[:3]
        self.bridge.update_hint(
            "dreamcoder",
            {
                "top_skills": [
                    {
                        "name": str(record.payload.get("name", "")).strip(),
                        "score": round(record.score(), 2),
                        "preview": self._payload_preview(record.payload),
                        "structured": self._structural_field_count(record.payload) > 0,
                        "depth": record.depth,
                    }
                    for record in top
                ]
            },
        )

    def _promote_validated_exact_spine(
        self,
        history: list[tuple[str, Any]],
        finish_status: Any,
    ) -> None:
        status = str(getattr(finish_status, "status", "") or "").strip().lower()
        if status != "win":
            return
        actions = [action for action, _frame in history if action != "RESET"]
        if len(actions) < 2:
            return
        signature = tuple(actions)
        for record in self.skills:
            if self._execution_signature(record.payload) == signature:
                payload = dict(record.payload)
                payload.setdefault(
                    "description",
                    "Exact action spine preserved from a validated successful run.",
                )
                payload["action_spine"] = list(signature)
                record.payload = payload
                record.observed_support += 3
                record.informative_hits += 2
                record.times_selected += 1
                record.last_updated_at = time.time()
                self._remember_skill(record, event="validated_spine")
                self._ensure_wrapper_for_spine(record)
                return

        payload = {
            "name": "Validated exact spine " + "-".join(actions[:6]),
            "description": "Exact action spine preserved from a validated successful run.",
            "trigger": "Fresh run after reset when broader probes have not yet produced a better exact route.",
            "body": list(signature),
            "action_spine": list(signature),
        }
        record = SkillRecord(payload=payload, depth=0)
        record.observed_support = 3
        record.informative_hits = 2
        record.times_selected = 1
        self.skills.append(record)
        self._remember_skill(record, event="validated_spine")
        self._ensure_wrapper_for_spine(record)

    def _ensure_wrapper_for_spine(self, spine_record: SkillRecord) -> None:
        payload = spine_record.payload
        spine_name = str(payload.get("name", "")).strip()
        if not spine_name:
            return
        if not self._is_wrapper_worthy_spine(payload):
            return
        for record in self.skills:
            if record is spine_record:
                continue
            refs = self._extract_skill_refs(self._payload_as_items(record.payload))
            if spine_name in refs:
                return
        wrapper_name = self._wrapper_name_for_spine(spine_name)
        if self._find_by_name(wrapper_name) is not None:
            return
        wrapper_payload = {
            "name": wrapper_name,
            "description": "Higher-level wrapper that preserves a validated opening spine while exposing when to reuse it and how to recover if it underperforms.",
            "precondition": "Fresh run, no stronger opening evidence yet, and available actions match the stored opening.",
            "subskills": [f"skill:{spine_name}"],
            "controller": "Run the referenced opening spine as the default entry tactic, then compare the observed effect against recent expectations before branching.",
            "expected_effect": "Reaches the same opening checkpoint or a similarly informative early state with less arbitrary probing.",
            "failure_recovery": "If the observed effect is clearly weaker than expected, stop repeating the opening blindly and switch to a different local probe family.",
        }
        wrapper = SkillRecord(
            payload=wrapper_payload,
            depth=max(1, spine_record.depth + 1),
        )
        wrapper.times_referenced = 1
        wrapper.observed_support = max(1, spine_record.observed_support // 2)
        wrapper.informative_hits = max(1, spine_record.informative_hits // 2)
        self.skills.append(wrapper)
        self._remember_skill(wrapper, event="auto_wrapper")

    @staticmethod
    def _wrapper_name_for_spine(spine_name: str) -> str:
        if spine_name.lower().startswith("validated exact spine "):
            suffix = spine_name[len("Validated exact spine ") :].strip()
            if suffix:
                return f"Opening wrapper for {suffix}"
        return f"Wrapper for {spine_name}"

    def _synchronize_wrappers_for_spines(self) -> None:
        for record in list(self.skills):
            if record.payload.get("action_spine"):
                self._ensure_wrapper_for_spine(record)

    @staticmethod
    def _is_wrapper_worthy_spine(payload: dict[str, Any]) -> bool:
        name = str(payload.get("name", "")).strip().lower()
        description = str(payload.get("description", "")).strip().lower()
        return name.startswith("validated exact spine") or "validated successful run" in description

    def _attach_recent_action_spine(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._is_action_grounded_payload(payload):
            return payload
        spine = self._recent_action_spine()
        if len(spine) < 2:
            return payload
        updated = dict(payload)
        updated["action_spine"] = spine
        return updated

    def _recent_action_spine(self) -> list[str]:
        spine: list[str] = []
        for action in self.bridge.recent_actions[-self.sleep_trajectory_window :]:
            if action == "RESET":
                continue
            if spine and spine[-1] == action:
                continue
            spine.append(action)
        return spine[-6:]

    def _route_action(
        self, action_name: str, available_actions: list[str]
    ) -> tuple[str, str] | None:
        if not self.skills or not available_actions:
            return None

        recent = list(self.bridge.recent_actions[-self.sleep_trajectory_window :])
        stalled = self._is_stalled()
        world_hint = self.bridge.read_hint("world_model")
        world_suggested = ""
        world_recent_surprise = False
        if isinstance(world_hint, dict):
            world_suggested = str(world_hint.get("suggested_action", "")).strip()
            world_recent_surprise = bool(world_hint.get("recent_surprise"))
        best: tuple[float, str, str, int, SkillRecord] | None = None

        for record in self._ordered_skills_for_routing():
            plan = self._resolve_action_plan(record)
            if not plan:
                continue
            next_action, progress = self._next_action_from_plan(
                plan, recent, available_actions
            )
            if next_action is None:
                continue
            has_spine = bool(record.payload.get("action_spine"))
            restarting_recent_completed_spine = False
            if (
                progress == 0
                and len(recent) >= len(plan)
                and recent[-len(plan) :] == plan
            ):
                restarting_recent_completed_spine = True
            if (
                record.score() < self.routing_min_score
                and progress == 0
                and action_name != "RESET"
            ):
                continue
            route_score = (
                max(record.score(), 0.5)
                + 12.0 * progress
                + 6.0 * record.novel_family_hits
                + 5.0 * record.branch_escape_hits
                + 4.0 * record.falsification_utility_hits
                + 3.0 * record.surprise_recovery_hits
                + 2.0 * int(has_spine)
                + 0.75 * self._structural_field_count(record.payload)
                + 0.5 * min(record.times_linked_to_surprise, 2)
                + 0.25 * min(record.informative_hits, 3)
                + (1.0 if world_suggested and next_action == world_suggested else 0.0)
                + (
                    0.75
                    if world_recent_surprise and record.times_linked_to_surprise > 0
                    else 0.0
                )
                - (20.0 if restarting_recent_completed_spine else 0.0)
                - 0.05 * max(0, len(plan) - 6)
                - (8.0 if progress == 0 and not has_spine else 0.0)
                - (
                    40.0
                    if self._is_exact_evidence_spine(record.payload)
                    and progress == 0
                    and record.novel_family_hits == 0
                    and record.branch_escape_hits == 0
                    and record.falsification_utility_hits == 0
                    else 0.0
                )
            )
            reason = (
                f"skill '{record.payload.get('name', '(unnamed)')}' suggests {next_action} "
                f"(progress={progress}, score={record.score():.1f})."
            )
            if world_suggested and next_action == world_suggested:
                reason += f" It agrees with the current world-model prior for {world_suggested}."
            if best is None or route_score > best[0]:
                best = (route_score, next_action, reason, progress, record)

        if best is None:
            return None

        _, suggested_action, reason, progress, record = best
        if action_name == suggested_action:
            if progress > 0 or bool(record.payload.get("action_spine")):
                record.times_selected += 1
                record.last_updated_at = time.time()
            return suggested_action, reason
        if action_name == "RESET" and self.reset_intercept_enabled:
            record.times_selected += 1
            record.reset_intercepts += 1
            record.last_updated_at = time.time()
            return suggested_action, f"Avoid RESET: {reason}"
        if progress > 0:
            record.times_selected += 1
            record.last_updated_at = time.time()
            return suggested_action, f"Continue active skill: {reason}"
        if (
            suggested_action != action_name
            and (
                record.novel_family_hits > 0
                or record.branch_escape_hits > 0
                or record.falsification_utility_hits > 0
                or self._structural_field_count(record.payload) > 0
            )
        ):
            record.times_selected += 1
            record.last_updated_at = time.time()
            return suggested_action, f"Frontier-oriented reroute: {reason}"
        if stalled and self.route_on_stall_enabled:
            record.times_selected += 1
            record.last_updated_at = time.time()
            return suggested_action, f"Stall detected, so {reason}"
        return None

    def _ordered_skills_for_routing(self) -> list[SkillRecord]:
        return sorted(
            self.skills,
            key=lambda r: (
                *self._display_priority_key(r),
                -r.novel_family_hits,
                -r.branch_escape_hits,
                -r.falsification_utility_hits,
                -r.times_linked_to_surprise,
                -r.observed_support,
            ),
        )[: self.routing_max_skills]

    @classmethod
    def _is_exact_evidence_spine(cls, payload: Any) -> bool:
        return (
            isinstance(payload, dict)
            and bool(payload.get("action_spine"))
            and cls._structural_field_count(payload) == 0
        )

    @classmethod
    def _display_priority_key(cls, record: SkillRecord) -> tuple[float, ...]:
        payload = record.payload
        structural = cls._structural_field_count(payload)
        exact_evidence = cls._is_exact_evidence_spine(payload)
        return (
            -float(structural > 0),
            -float(record.depth),
            -float(record.novel_family_hits > 0 or record.branch_escape_hits > 0),
            -float(record.falsification_utility_hits > 0),
            float(exact_evidence),
            -float(record.score()),
            -float(record.last_updated_at),
        )

    def _resolve_action_plan(
        self, record: SkillRecord, seen: set[str] | None = None
    ) -> list[str]:
        seen = set() if seen is None else set(seen)
        name = str(record.payload.get("name", "")).strip()
        if name:
            if name in seen:
                return []
            seen.add(name)

        plan: list[str] = []
        payload_items = self._execution_items(record.payload)
        if isinstance(record.payload, dict) and record.payload.get("action_spine"):
            spine_items = self._body_as_items(record.payload.get("action_spine"))
            body_items = self._body_as_items(record.payload.get("body"))
            if spine_items and body_items == spine_items:
                filtered_items: list[Any] = []
                skipped_body = False
                for key in (
                    "action_spine",
                    "body",
                    "rule",
                    "policy",
                    "usage",
                    "observed_mapping",
                    "controller",
                    "failure_recovery",
                    "subskills",
                    "plan",
                    "steps",
                ):
                    if key not in record.payload:
                        continue
                    if key == "body" and not skipped_body:
                        skipped_body = True
                        continue
                    filtered_items.extend(self._body_as_items(record.payload.get(key)))
                payload_items = filtered_items
        for item in payload_items:
            refs = self._inline_action_refs(item)
            if not refs:
                token = str(item).strip()
                if token.startswith("ACTION") or token == "RESET":
                    refs = [token]
                elif token.startswith("skill:"):
                    refs = [token]
            for token in refs:
                if token.startswith("ACTION") or token == "RESET":
                    plan.append(token)
                    continue
                if not token.startswith("skill:"):
                    continue
                ref = token.split("skill:", 1)[1].strip()
                target = self._find_by_name(ref)
                if target is None:
                    continue
                plan.extend(self._resolve_action_plan(target, seen))
        return plan

    @staticmethod
    def _next_action_from_plan(
        plan: list[str], recent: list[str], available_actions: list[str]
    ) -> tuple[str | None, int]:
        filtered = [token for token in plan if token in available_actions]
        if not filtered:
            return None, 0
        # If the recent trajectory already completed the whole local plan,
        # do not immediately restart the same exact spine from the top.
        if len(recent) >= len(filtered) and recent[-len(filtered) :] == filtered:
            return None, len(filtered)
        max_progress = min(len(filtered) - 1, len(recent))
        for progress in range(max_progress, 0, -1):
            if recent[-progress:] == filtered[:progress]:
                return filtered[progress], progress
        return filtered[0], 0

    def _is_stalled(self) -> bool:
        if self.stall_window <= 1 or len(self._recent_actions) < self.stall_window:
            return False
        recent = self._recent_actions[-self.stall_window :]
        return len(set(recent)) <= max(1, self.stall_window // 2)

    @staticmethod
    def _infer_depth(payload: dict[str, Any]) -> int:
        explicit = payload.get("depth")
        if isinstance(explicit, int):
            return max(0, explicit)
        body = DreamCoderModule._payload_as_items(payload)
        max_inner = 0
        for entry in body:
            if isinstance(entry, str) and entry.startswith("skill:"):
                max_inner = max(max_inner, 1)
            elif isinstance(entry, dict) and entry.get("kind") == "skill":
                max_inner = max(max_inner, int(entry.get("depth", 0)) + 1)
        return max_inner

    def _body_preview(self, body: Any) -> str:
        items = self._body_as_items(body)
        if not items:
            return ""
        if len(items) <= self.body_preview_items:
            return ", ".join(str(x) for x in items)

        head = max(1, self.body_preview_items - 4)
        preview = [str(x) for x in items[:head]]
        tail = [str(x) for x in items[-4:]]
        return ", ".join(preview + [f"... ({len(items)} steps total) ..."] + tail)

    def _payload_preview(self, payload: dict[str, Any]) -> str:
        primary = payload.get(
            "body",
            payload.get("policy", payload.get("steps", "")),
        )
        preview = self._body_preview(primary)
        if preview:
            if payload.get("action_spine"):
                spine_preview = self._body_preview(payload.get("action_spine"))
                if spine_preview and spine_preview not in preview:
                    return f"{preview} | action_spine: {spine_preview}"
            return preview
        if payload.get("action_spine"):
            return self._body_preview(payload.get("action_spine"))
        return self._body_preview(self._payload_action_refs(payload))

    @classmethod
    def _execution_signature(cls, payload: Any) -> tuple[str, ...]:
        if isinstance(payload, dict) and payload.get("action_spine"):
            items = cls._body_as_items(payload.get("action_spine"))
        else:
            items = cls._execution_items(payload)
        signature: list[str] = []
        for item in items:
            refs = cls._inline_action_refs(item)
            if refs:
                signature.extend(
                    token for token in refs if token.startswith("ACTION") or token == "RESET"
                )
        return tuple(signature)

    def _mark_referenced_skills(self, payload: dict[str, Any]) -> None:
        refs = self._extract_skill_refs(self._payload_as_items(payload))
        if not refs:
            return
        for record in self.skills:
            name = str(record.payload.get("name", "")).strip()
            if name and name in refs:
                record.times_referenced += 1
                record.last_updated_at = time.time()

    def _find_by_name(self, name: str) -> SkillRecord | None:
        if not name:
            return None
        for record in self.skills:
            if str(record.payload.get("name", "")).strip() == name:
                return record
        return None

    def _find_revision_target(self, payload: dict[str, Any]) -> SkillRecord | None:
        candidate_tokens = self._signature_tokens(payload)
        candidate_text = self._content_tokens(payload)
        if not candidate_tokens:
            return None
        best: SkillRecord | None = None
        best_overlap = 0.0
        for record in self.skills:
            tokens = self._signature_tokens(record.payload)
            if not tokens:
                continue
            action_overlap = len(candidate_tokens & tokens) / max(
                1, len(candidate_tokens | tokens)
            )
            text_tokens = self._content_tokens(record.payload)
            text_overlap = len(candidate_text & text_tokens) / max(
                1, len(candidate_text | text_tokens)
            )
            allow_revision = (
                action_overlap >= self.revision_overlap_threshold
                and text_overlap >= self.revision_text_overlap_threshold
            )
            if allow_revision and action_overlap > best_overlap:
                best = record
                best_overlap = action_overlap
        return best

    def _find_explicit_revision_target(self, payload: dict[str, Any]) -> SkillRecord | None:
        for key in ("supersedes", "revises", "updates"):
            target = str(payload.get(key, "")).strip()
            if not target:
                continue
            match = self._find_by_name(target)
            if match is not None:
                return match
        return None

    def _score_skills_from_observation(
        self, before: Any | None, action_name: str, after: Any | None
    ) -> None:
        if before is None or after is None:
            return
        diff_mag = grid_diff_magnitude(before, after)
        if diff_mag > 0:
            self._informative_actions_since_proposal += 1
        if action_name:
            self._recent_actions.append(action_name)
            self._recent_actions = self._recent_actions[-self.sleep_trajectory_window :]
        trajectory = list(self._recent_actions)
        for record in self.skills:
            if not self._matches_recent_suffix(self._execution_items(record.payload), trajectory):
                continue
            record.observed_support += 1
            if diff_mag > 0:
                record.informative_hits += 1
            observation = self.bridge.last_observation
            if observation is not None:
                if observation.new_signature:
                    record.novel_signature_hits += 1
                if observation.new_family:
                    record.novel_family_hits += 1
                if observation.branch_escape:
                    record.branch_escape_hits += 1
                if observation.hypothesis_falsified:
                    record.falsification_utility_hits += 1
                if observation.hypothesis_supported is False and observation.branch_escape:
                    record.surprise_recovery_hits += 1
            record.last_matched_trajectory = trajectory[-self.surprise_match_suffix :]
            record.last_updated_at = time.time()

    def _extract_skill_refs(self, body: Any) -> set[str]:
        body = self._body_as_items(body)
        refs: set[str] = set()
        for entry in body:
            if isinstance(entry, str) and entry.startswith("skill:"):
                refs.add(entry.split("skill:", 1)[1].strip())
            elif isinstance(entry, dict) and entry.get("kind") == "skill":
                name = str(entry.get("name", "")).strip()
                if name:
                    refs.add(name)
        return refs

    def _matches_recent_suffix(self, body: Any, trajectory: list[str]) -> bool:
        body = self._body_as_items(body)
        if not body or not trajectory:
            return False
        flat_body: list[str] = []
        for item in body:
            refs = self._inline_action_refs(item)
            if refs:
                flat_body.extend(
                    token for token in refs if token.startswith("ACTION") or token == "RESET"
                )
            else:
                flat_body.append(str(item))
        min_len = min(self.surprise_match_suffix, len(trajectory), len(flat_body))
        for size in range(min_len, 1, -1):
            suffix = trajectory[-size:]
            for idx in range(0, len(flat_body) - size + 1):
                if flat_body[idx : idx + size] == suffix:
                    return True
        return False

    @staticmethod
    def _body_as_items(body: Any) -> list[Any]:
        if isinstance(body, list):
            return body
        if isinstance(body, dict):
            return [body]
        if not isinstance(body, str):
            return []
        action_tokens = re.findall(r"ACTION[1-7]|RESET|skill:[A-Za-z0-9_./\\-]+", body)
        if action_tokens:
            return action_tokens
        lines = [line.strip("-* \t") for line in body.splitlines() if line.strip()]
        return lines

    @classmethod
    def _execution_items(cls, payload: Any) -> list[Any]:
        if isinstance(payload, list):
            return cls._body_as_items(payload)
        if not isinstance(payload, dict):
            return cls._body_as_items(payload)
        items: list[Any] = []
        if payload.get("action_spine"):
            items.extend(cls._body_as_items(payload.get("action_spine")))
        for key in (
            "body",
            "rule",
            "policy",
            "usage",
            "observed_mapping",
            "controller",
            "failure_recovery",
            "subskills",
            "plan",
            "steps",
        ):
            if key in payload:
                items.extend(cls._body_as_items(payload.get(key)))
        return items

    @classmethod
    def _payload_as_items(cls, payload: Any) -> list[Any]:
        if isinstance(payload, list):
            return cls._body_as_items(payload)
        if not isinstance(payload, dict):
            return cls._body_as_items(payload)
        items: list[Any] = []
        for key in (
            "body",
            "rule",
            "policy",
            "usage",
            "action_spine",
            "observed_mapping",
            "controller",
            "failure_recovery",
            "subskills",
            "plan",
            "steps",
            "precondition",
            "postcondition",
            "expected_effect",
            "trigger",
            "description",
        ):
            if key in payload:
                items.extend(cls._body_as_items(payload.get(key)))
        return items

    @staticmethod
    def _structural_field_count(payload: Any) -> int:
        if not isinstance(payload, dict):
            return 0
        count = 0
        for key in (
            "precondition",
            "controller",
            "expected_effect",
            "failure_recovery",
            "subskills",
            "plan",
            "postcondition",
        ):
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                count += 1
            elif isinstance(value, list) and value:
                count += 1
            elif isinstance(value, dict) and value:
                count += 1
        return count

    @classmethod
    def _signature_kind(cls, payload: Any) -> str:
        if not isinstance(payload, dict):
            return "flat"
        if payload.get("action_spine") and cls._is_wrapper_worthy_spine(payload):
            return "spine"
        if cls._structural_field_count(payload) > 0:
            return "structured"
        return "flat"

    @classmethod
    def _signature_tokens(cls, payload: Any) -> set[str]:
        return {token.upper() for token in cls._payload_action_refs(payload)}

    @classmethod
    def _content_tokens(cls, payload: Any) -> set[str]:
        stop = {
            "action",
            "actions",
            "then",
            "with",
            "from",
            "into",
            "that",
            "this",
            "when",
            "while",
            "after",
            "before",
            "once",
            "again",
            "using",
            "use",
            "step",
            "steps",
            "probe",
            "repeat",
            "repeated",
            "continue",
            "available",
            "skill",
            "body",
            "rule",
            "policy",
            "trigger",
            "description",
        }
        tokens: set[str] = set()
        for item in cls._payload_as_items(payload):
            if isinstance(item, dict):
                item = json.dumps(item, ensure_ascii=True)
            for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", str(item).lower()):
                if token in stop:
                    continue
                if token.startswith("action") or token == "reset":
                    continue
                tokens.add(token)
        return tokens

    @classmethod
    def _is_action_grounded_payload(cls, payload: dict[str, Any]) -> bool:
        return bool(cls._signature_tokens(payload))

    @classmethod
    def _payload_action_refs(cls, payload: Any) -> list[str]:
        refs: list[str] = []
        for item in cls._payload_as_items(payload):
            inline = cls._inline_action_refs(item)
            if inline:
                refs.extend(inline)
                continue
            text = str(item).strip()
            if text.startswith("ACTION") or text == "RESET" or text.startswith("skill:"):
                refs.append(text)
        return refs

    @staticmethod
    def _inline_action_refs(item: Any) -> list[str]:
        if isinstance(item, dict):
            if item.get("kind") == "skill":
                name = str(item.get("name", "")).strip()
                return [f"skill:{name}"] if name else []
            item = json.dumps(item, ensure_ascii=True)
        return re.findall(r"ACTION[1-7]|RESET|skill:[A-Za-z0-9_./\\-]+", str(item))
