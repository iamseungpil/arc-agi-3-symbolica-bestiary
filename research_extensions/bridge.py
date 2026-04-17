"""Shared state bridge between research modules.

When modules communicate through a bridge instead of direct coupling, each
module can be toggled independently while still sharing surprise signals,
trajectories, and proposed skills.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Prediction:
    """What one module thinks will happen after an action."""
    action: str
    expected_change_summary: str = ""
    observation_prediction: str = ""
    progress_prediction: str = ""
    action_recommendation: str = ""
    recommendation_rationale: str = ""
    rival_predictions: list[str] = field(default_factory=list)
    expected_signature: str | None = None


@dataclass(slots=True)
class Observation:
    """What actually happened after an action."""
    action: str
    before_signature: str
    after_signature: str
    diff_magnitude: float
    before_family: str = ""
    after_family: str = ""
    new_signature: bool = False
    new_family: bool = False
    branch_escape: bool = False
    repeated_family_depth: int = 0
    hypothesis_supported: bool | None = None
    hypothesis_falsified: bool | None = None
    free_form_note: str = ""


@dataclass(slots=True)
class SurpriseEvent:
    """Emitted when a prediction does not match the observation."""
    action: str
    predicted: Prediction | None
    observation: Observation
    recent_trajectory: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class ProposedSkill:
    """A skill the agent wrote in natural language. Names, triggers, body
    are all the agent's choice. We never force a format on them."""
    payload: dict[str, Any]
    created_at: float = field(default_factory=time.time)


@dataclass(slots=True)
class SkillExecutionState:
    """Bookkeeping for an option being executed in the real environment.

    Carries enough state for the registry to bypass the LLM on each
    intermediate step, and for the world model to compare predicted vs
    observed transitions. Aborted on the first surprise (per H6)."""
    skill_name: str
    body: list[str]
    cursor: int = 0
    aborted_reason: str | None = None
    started_at: float = field(default_factory=time.time)
    last_predicted: Prediction | None = None
    surprises_during: int = 0
    source: str = "agent"  # "agent" | "mcts" | "wrapper"

    @property
    def is_done(self) -> bool:
        return self.aborted_reason is not None or self.cursor >= len(self.body)

    def next_action(self) -> str | None:
        if self.is_done:
            return None
        return self.body[self.cursor]

    def advance(self) -> None:
        self.cursor += 1


@dataclass(slots=True)
class ScientistControlState:
    """Plan §5: agent-side state that every module reads/writes.

    Modules MUST publish to this object — local quality without state
    publication is a plan violation."""
    best_hypothesis: str = ""
    rival_hypotheses: list[str] = field(default_factory=list)
    target_unseen_observation: str = ""
    target_falsifier: str = ""
    progress_variables: dict[str, Any] = field(default_factory=dict)
    local_attractor_pressure: int = 0
    last_updated_at: float = field(default_factory=time.time)


class SharedBridge:
    """Lightweight shared bus used by DC / WM / MH modules."""

    def __init__(self) -> None:
        self.surprises: list[SurpriseEvent] = []
        self.proposed_skills: list[ProposedSkill] = []
        self.recent_actions: list[str] = []
        self.recent_families: list[str] = []
        self.trajectory_window: int = 32
        self.current_prediction: Prediction | None = None
        self.last_observation: Observation | None = None
        self.seen_signatures: set[str] = set()
        self.seen_families: set[str] = set()
        self.shared_hints: dict[str, Any] = {}
        # v3 additions for option execution + plan §5 control state.
        self.current_skill_in_execution: SkillExecutionState | None = None
        self.scientist_state: ScientistControlState = ScientistControlState()
        # Wake-trigger queue: entries are surprise events that the LLM
        # must rewrite predict_effect for on the next turn. Drained by
        # registry.wake_phase().
        self.pending_wake_triggers: list[SurpriseEvent] = []
        # MCTS imagination output queue: each entry is a candidate skill
        # payload that DC.propose_from_mcts will turn into a SkillRecord.
        # Kept on the bridge so planner can run before DC sees it.
        self.pending_mcts_proposals: list[dict[str, Any]] = []

    # -- Writing ----------------------------------------------------------
    def record_prediction(self, prediction: Prediction) -> None:
        self.current_prediction = prediction

    def record_observation(self, observation: Observation) -> SurpriseEvent | None:
        after_family = observation.after_family or observation.after_signature
        before_family = observation.before_family or observation.before_signature
        observation.before_family = before_family
        observation.after_family = after_family
        observation.new_signature = observation.after_signature not in self.seen_signatures
        observation.new_family = after_family not in self.seen_families
        observation.branch_escape = before_family != after_family
        self.recent_families.append(after_family)
        if len(self.recent_families) > self.trajectory_window:
            self.recent_families = self.recent_families[-self.trajectory_window :]
        repeated_depth = 0
        for family in reversed(self.recent_families):
            if family != after_family:
                break
            repeated_depth += 1
        observation.repeated_family_depth = repeated_depth
        self.seen_signatures.add(observation.after_signature)
        self.seen_families.add(after_family)
        self.last_observation = observation

        self.recent_actions.append(observation.action)
        if len(self.recent_actions) > self.trajectory_window:
            self.recent_actions = self.recent_actions[-self.trajectory_window :]

        prediction = self.current_prediction
        self.current_prediction = None

        if prediction is None:
            return None
        if prediction.action != observation.action:
            matched = False
        else:
            matched = self._prediction_matches_observation(prediction, observation)

        if matched:
            return None

        event = SurpriseEvent(
            action=observation.action,
            predicted=prediction,
            observation=observation,
            recent_trajectory=list(self.recent_actions),
        )
        self.surprises.append(event)
        if len(self.surprises) > 64:
            self.surprises = self.surprises[-64:]
        # v3.14: every surprise queues a wake trigger, not just those
        # during skill execution. Previously wake only fired during
        # DreamCoder's option-execution path (on_skill_step), leaving
        # 97/97 v3.11 surprises with zero wake prompts to the LLM.
        self.pending_wake_triggers.append(event)
        if len(self.pending_wake_triggers) > 32:
            self.pending_wake_triggers = self.pending_wake_triggers[-32:]
        # v3.15: surface the wake event as a dedicated bridge hint too,
        # so agent.py's _prompt_overlay can render it prominently even
        # when wake_prompt_section was drained earlier in the turn.
        self.shared_hints.setdefault("wake_events_log", [])
        self.shared_hints["wake_events_log"].append({
            "action": event.action,
            "diff_magnitude": event.observation.diff_magnitude,
            "new_family": event.observation.new_family,
            "branch_escape": event.observation.branch_escape,
        })
        self.shared_hints["wake_events_log"] = self.shared_hints["wake_events_log"][-16:]
        return event

    def record_proposed_skill(self, payload: dict[str, Any]) -> ProposedSkill:
        skill = ProposedSkill(payload=dict(payload))
        self.proposed_skills.append(skill)
        return skill

    # -- Reading ----------------------------------------------------------
    def recent_surprises(self, limit: int = 5) -> list[SurpriseEvent]:
        return self.surprises[-limit:]

    def unseen_proposals(self, since: int = 0) -> list[ProposedSkill]:
        return self.proposed_skills[since:]

    def reset_prediction(self) -> None:
        self.current_prediction = None

    def update_hint(self, key: str, value: Any) -> None:
        self.shared_hints[key] = value

    def read_hint(self, key: str, default: Any = None) -> Any:
        return self.shared_hints.get(key, default)

    # -- v3 option execution surface --------------------------------------
    def commit_skill(self, state: SkillExecutionState) -> None:
        self.current_skill_in_execution = state

    def clear_committed_skill(self, reason: str | None = None) -> None:
        if self.current_skill_in_execution is not None and reason is not None:
            self.current_skill_in_execution.aborted_reason = reason
        self.current_skill_in_execution = None

    def queue_wake_trigger(self, event: SurpriseEvent) -> None:
        self.pending_wake_triggers.append(event)
        if len(self.pending_wake_triggers) > 32:
            self.pending_wake_triggers = self.pending_wake_triggers[-32:]

    def drain_wake_triggers(self) -> list[SurpriseEvent]:
        items = list(self.pending_wake_triggers)
        self.pending_wake_triggers.clear()
        return items

    def queue_mcts_proposal(self, payload: dict[str, Any]) -> None:
        self.pending_mcts_proposals.append(payload)
        if len(self.pending_mcts_proposals) > 16:
            self.pending_mcts_proposals = self.pending_mcts_proposals[-16:]

    def drain_mcts_proposals(self) -> list[dict[str, Any]]:
        items = list(self.pending_mcts_proposals)
        self.pending_mcts_proposals.clear()
        return items

    @staticmethod
    def _prediction_matches_observation(
        prediction: Prediction, observation: Observation
    ) -> bool:
        # Signature-based surprise if predictor gave a signature.
        if prediction.expected_signature is not None:
            return prediction.expected_signature == observation.after_signature

        progress = prediction.progress_prediction.strip().lower()
        if progress:
            if "new_family" in progress and not observation.new_family:
                return False
            if "new_signature" in progress and not observation.new_signature:
                return False
            if "branch_escape" in progress and not observation.branch_escape:
                return False
            if "falsify" in progress and not bool(observation.hypothesis_falsified):
                return False
            if "support" in progress and not bool(observation.hypothesis_supported):
                return False

        observation_prediction = prediction.observation_prediction.strip().lower()
        if observation_prediction:
            if "new_family" in observation_prediction and not observation.new_family:
                return False
            if "same_family" in observation_prediction and observation.branch_escape:
                return False

        summary = prediction.expected_change_summary.strip().upper()
        if summary.startswith("EXPECT_NO_CHANGE"):
            return observation.diff_magnitude == 0
        if summary.startswith("EXPECT_CHANGE"):
            return observation.diff_magnitude > 0

        # Unstructured predictions are not scored; treating them as matched would
        # create false confidence, and treating them as mismatched would create
        # false surprises.
        return True
