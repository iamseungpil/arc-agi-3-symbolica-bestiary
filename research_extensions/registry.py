from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bridge import SharedBridge
from .config import ResearchConfig

logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ResearchRuntimeContext:
    game_id: str
    workdir: Path
    shared_dir: Path


class ModuleRegistry:
    """Loads toggleable research modules and dispatches hooks to them.

    Modules never hold references to each other directly. Cross-module
    information flows only through the SharedBridge so any subset of the
    three modules can be enabled or disabled independently.
    """

    def __init__(self, config: ResearchConfig, context: ResearchRuntimeContext):
        self.config = config
        self.context = context
        self.bridge = SharedBridge()
        self._modules: dict[str, Any] = {}

    def load(self) -> None:
        if self.config.dreamcoder.enabled:
            from .modules.dreamcoder import DreamCoderModule

            self._modules["dreamcoder"] = DreamCoderModule(
                self.config.dreamcoder.params, self.context, self.bridge
            )
        if self.config.world_model.enabled:
            from .modules.world_model import WorldModelModule

            self._modules["world_model"] = WorldModelModule(
                self.config.world_model.params, self.context, self.bridge
            )
        if self.config.meta_harness.enabled:
            from .modules.meta_harness import MetaHarnessModule

            self._modules["meta_harness"] = MetaHarnessModule(
                self.config.meta_harness.params, self.context, self.bridge
            )
        if getattr(self.config, "planner", None) is not None and self.config.planner.enabled:
            from .modules.planner import MCTSPlanner

            self._modules["planner"] = MCTSPlanner(
                self.config.planner.params, self.context, self.bridge
            )
        if self._modules:
            logger.info("Active research modules: %s", ", ".join(self._modules))

    def active(self) -> dict[str, Any]:
        return dict(self._modules)

    def get(self, name: str) -> Any | None:
        return self._modules.get(name)

    def prompt_overlay(self) -> str:
        parts = []
        log = self._bypass_log_overlay()
        if log:
            parts.append(log)
        for module in self._modules.values():
            text = module.prompt_overlay()
            if text:
                parts.append(text)
        return "\n\n".join(parts)

    def _bypass_log_overlay(self) -> str:
        """Summarise any skill-bypass steps that happened since control
        last returned to the LLM. Keeps the agent informed of autonomous
        actions without flooding the prompt."""
        log = self.bridge.read_hint("bypass_log", [])
        if not log:
            return ""
        lines = ["[Skill Execution Log] (autonomous actions while LLM was idle)"]
        for entry in log[-6:]:
            lines.append(
                f"- {entry.get('turn', '?')} skill={entry.get('skill', '?')[:40]} "
                f"action={entry.get('action', '?')} outcome={entry.get('outcome', '?')}"
            )
        if len(log) > 6:
            lines.append(f"... ({len(log) - 6} earlier entries omitted)")
        # Clear the log once the LLM has seen it.
        self.bridge.update_hint("bypass_log", [])
        return "\n".join(lines)

    def record_bypass_step(self, *, skill: str, action: str, outcome: str, turn: int) -> None:
        log = list(self.bridge.read_hint("bypass_log", []))
        log.append({"skill": skill, "action": action, "outcome": outcome, "turn": turn})
        self.bridge.update_hint("bypass_log", log[-32:])

    def on_memories_ready(self, memories: Any) -> None:
        for module in self._modules.values():
            module.on_memories_ready(memories)

    def before_action(
        self, frame: Any | None, action_name: str, available_actions: list[str]
    ) -> str | None:
        chosen = action_name
        ordered_names = [name for name in self._modules if name != "dreamcoder"]
        if "dreamcoder" in self._modules:
            ordered_names.append("dreamcoder")
        for name in ordered_names:
            module = self._modules[name]
            override = module.before_action(frame, chosen, available_actions)
            if override and override in available_actions:
                chosen = override
        # Exploration-phase override: after modules have run, if we are
        # in the exploration window, force the least-tried action. This
        # lets the LLM still emit side channels (world_update etc.) while
        # the registry controls action selection.
        in_explore, _ = self._in_exploration_phase()
        if in_explore and frame is not None:
            exp_action = self._exploration_action(frame, available_actions)
            if exp_action and exp_action in available_actions:
                chosen = exp_action
        return chosen

    def after_action(self, before: Any | None, action_name: str, after: Any) -> None:
        # Snapshot: was a skill in execution when this action was committed?
        skill_was_running = self.bridge.current_skill_in_execution is not None
        for module in self._modules.values():
            module.after_action(before, action_name, after)
        # Plan §4 P0: if a skill was running, compare the prediction stored
        # on the bridge (either agent-emitted or synthesised on bypass)
        # against the real observation and possibly terminate the option.
        if skill_was_running:
            dc = self._modules.get("dreamcoder")
            if dc is not None and hasattr(dc, "on_skill_step"):
                observation = self.bridge.last_observation
                predicted = self.bridge.read_hint("last_synthetic_prediction")
                if observation is not None:
                    dc.on_skill_step(predicted=predicted, observation=observation)
        # Periodic experience consolidation (every 10 non-reset actions).
        # This runs even when MCTS gate is closed so observed-law skills
        # accumulate from day 1.
        self._steps_since_consolidation = getattr(self, "_steps_since_consolidation", 0) + 1
        if action_name != "RESET" and self._steps_since_consolidation >= 10:
            self._steps_since_consolidation = 0
            dc = self._modules.get("dreamcoder")
            wm = self._modules.get("world_model")
            if dc is not None and wm is not None and hasattr(dc, "consolidate_experience"):
                try:
                    dc.consolidate_experience(world_model=wm)
                except Exception:
                    logger.exception("periodic consolidate_experience failed")

    # -- v3 orchestration hooks (plan §4 P0) -------------------------------
    def _in_exploration_phase(self) -> tuple[bool, str]:
        """Pre-MCTS exploration gate.

        World coder is not trained yet on turn 1. Forcing MCTS (which
        uses the LLM-written predict_effect) too early just amplifies
        noise. We stay in exploration mode until we have N real
        transitions AND at least M distinct observation families.

        Returns (in_exploration, reason)."""
        wm = self._modules.get("world_model")
        planner = self._modules.get("planner")
        if wm is None or planner is None:
            # Exploration gate is only meaningful when the planner is
            # active (it feeds real transitions into the simulator). For
            # legacy DC-only or WM-only configs, let the original
            # before_action chain run unchanged.
            return False, ""
        min_transitions = int(
            (self.config.planner.params.get("exploration_min_transitions", 15)
             if getattr(self.config, "planner", None) else 15)
        )
        min_families = int(
            (self.config.planner.params.get("exploration_min_families", 3)
             if getattr(self.config, "planner", None) else 3)
        )
        n_trans = len(getattr(wm, "unit_tests", []))
        n_families = len(getattr(self.bridge, "seen_families", set()))
        if n_trans < min_transitions:
            return True, f"exploration: {n_trans}/{min_transitions} transitions"
        if n_families < min_families:
            return True, f"exploration: {n_families}/{min_families} families"
        return False, ""

    def _exploration_action(
        self, frame: Any, available_actions: list[str]
    ) -> str | None:
        """Pick the globally least-tried action (not per-state).

        Previously this picked 'first untried in local bucket', but since
        ls20 signatures change every step, every state looked untried and
        the sort tiebreak always picked ACTION1 — all 80 turns used
        ACTION1. Global counts + random tiebreak prevents this.
        """
        wm = self._modules.get("world_model")
        if wm is None or not available_actions:
            return None
        import random as _r
        counts: dict[str, float] = {a: 0.0 for a in available_actions if a != "RESET"}
        for state_bucket in wm.transitions.values():
            for a, stats in state_bucket.items():
                if a in counts:
                    counts[a] += float(stats.get("count", 0.0))
        if not counts:
            return None
        # Rank by count ASC, tiebreak random so successive exploration
        # turns don't collapse to ACTION1.
        min_count = min(counts.values())
        candidates = [a for a, c in counts.items() if c == min_count]
        return _r.choice(sorted(candidates))

    def imagine_and_maybe_commit(
        self, frame: Any, available_actions: list[str]
    ) -> str | None:
        """Returns the next real-env action if a skill is (or just got)
        committed, else None. Used by the agent to bypass the LLM.

        Order:
        1. Already-executing skill → return its next body action.
        2. Otherwise, run imagination MCTS. If the gate is open and at
           least one proposal beats the BFS baseline, commit the best
           one, run sleep_refactor, and return its first body action.
        3. If nothing useful came out of imagination, return None."""
        dc = self._modules.get("dreamcoder")
        wm = self._modules.get("world_model")
        planner = self._modules.get("planner")

        if dc is not None and hasattr(dc, "next_skill_action"):
            committed = dc.next_skill_action()
            if committed and committed in available_actions:
                return committed

        if dc is None or wm is None or planner is None:
            return None

        # Exploration-first gate. Unlike MCTS bypass, exploration phase
        # does NOT skip the LLM — we still want the agent to emit
        # `world_update`, `propose_skill`, and `predict` so the world
        # coder starts training. `before_action` handles the action
        # override separately.
        in_explore, reason = self._in_exploration_phase()
        self.bridge.update_hint("exploration_reason", reason if in_explore else "")
        if in_explore:
            # Return None → agent.py will keep the LLM path; registry's
            # `before_action` pass below will override the chosen action.
            return None

        try:
            sig = wm._signature(frame)
        except Exception:
            return None

        def obs_factory(state_signature: str) -> dict[str, Any]:
            try:
                obs = wm._make_simulator_observation(frame)
                obs["signature"] = state_signature
                return obs
            except Exception:
                return {"signature": state_signature, "available_actions": list(available_actions)}

        result = planner.plan(
            world_model=wm,
            root_state_signature=sig,
            available_actions=available_actions,
            observation_factory=obs_factory,
        )
        if not result.get("gate_ok"):
            return None
        proposed = dc.propose_from_mcts(
            result,
            root_state_signature=sig,
            bfs_baseline=result.get("bfs_mean_reward", 0.0),
        )
        if proposed:
            best = max(
                proposed,
                key=lambda r: float(r.payload.get("mcts_simulator_reward", 0.0)),
            )
            state = dc.commit_skill(best, source="mcts")
            if state is not None:
                try:
                    dc.sleep_refactor()
                except Exception:
                    logger.exception("sleep_refactor failed; continuing without refactor")
                # Consolidate experience every imagination call so the
                # observed-law skills stay current for the next LLM turn.
                try:
                    if hasattr(dc, "consolidate_experience"):
                        dc.consolidate_experience(world_model=wm)
                except Exception:
                    logger.exception("consolidate_experience failed; continuing")
                if state.next_action() in available_actions:
                    return state.next_action()
        return None

    def synthesize_prediction(self, frame: Any, action: str) -> None:
        """Called right before a bypass action commits. Uses the world
        model simulator to build a predicted dict and stash it on the
        bridge for `after_action` to compare against reality."""
        wm = self._modules.get("world_model")
        if wm is None or not hasattr(wm, "simulate_step"):
            return
        try:
            sig = wm._signature(frame)
        except Exception:
            return
        try:
            obs = wm._make_simulator_observation(frame)
        except Exception:
            obs = None
        sim = wm.simulate_step(sig, action, obs)
        predicted = {
            "expect_change": sim["reward_signals"].get("expect_change"),
            "expected_diff_band": sim["reward_signals"].get("expected_diff_band"),
            "next_signature_hint": sim.get("next_signature_hint"),
            "observation_prediction": "",  # bypass skips LLM prose
            "progress_prediction": sim["reward_signals"].get("progress_prediction", ""),
        }
        self.bridge.update_hint("last_synthetic_prediction", predicted)
        # Also set bridge.current_prediction so world_model.record_observation
        # can score this transition as a unit test, same as an agent-emitted
        # prediction.
        from .bridge import Prediction as _P
        self.bridge.record_prediction(_P(
            action=action,
            expected_change_summary=(
                ("EXPECT_CHANGE: " if predicted["expect_change"] else
                 "EXPECT_NO_CHANGE: " if predicted["expect_change"] is False else "")
                + "(synthesised from simulator)"
            ),
            observation_prediction=predicted["observation_prediction"],
            progress_prediction=predicted["progress_prediction"],
            expected_signature=predicted["next_signature_hint"],
        ))

    def wake_overlay(self) -> str:
        wm = self._modules.get("world_model")
        if wm is None or not hasattr(wm, "wake_prompt_section"):
            return ""
        return wm.wake_prompt_section()

    def on_run_end(self, *, history: list[tuple[str, Any]], finish_status: Any) -> None:
        for module in self._modules.values():
            module.on_run_end(
                history=history,
                finish_status=finish_status,
                workdir=self.context.workdir,
            )

    # Agent-facing bridge API ---------------------------------------------
    def record_agent_prediction(self, action: str, expected_change: Any) -> None:
        wm = self._modules.get("world_model")
        if wm is not None and hasattr(wm, "record_agent_prediction"):
            wm.record_agent_prediction(action, expected_change)

    def record_agent_skill_proposal(self, payload: dict[str, Any]) -> None:
        dc = self._modules.get("dreamcoder")
        if dc is not None and hasattr(dc, "record_agent_proposal"):
            dc.record_agent_proposal(payload)

    def record_agent_env_note(self, text: str) -> None:
        wm = self._modules.get("world_model")
        if wm is not None and hasattr(wm, "record_agent_env_note"):
            wm.record_agent_env_note(text)

    def record_agent_world_update(self, payload: Any) -> None:
        wm = self._modules.get("world_model")
        if wm is not None and hasattr(wm, "record_agent_world_update"):
            wm.record_agent_world_update(payload)
