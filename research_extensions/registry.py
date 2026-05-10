from __future__ import annotations

from collections import Counter
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .bridge import SharedBridge
from .config import ResearchConfig

logger = logging.getLogger(__name__)


def _was_prediction_correct(pre_prediction: Any, observation: Any) -> bool | None:
    """Coarse correctness check for world-model predictions (Issue #6).

    Real ``observation.after_signature`` values are content hashes, whereas
    ``Prediction.expected_signature`` is frequently a semantic hint like
    ``"new_family_frontier"`` emitted by the simulator stub or the agent.
    A strict string equality check therefore collapses ``p_wm_correct`` to
    0 in production even when the prediction was semantically correct.

    We introduce a coarse check that (a) accepts literal hash-equality, and
    (b) maps the common semantic hints onto observation flags so they count
    as correct whenever the flag fires. Returns ``None`` when there is no
    prediction to score — callers must treat ``None`` as "skip recording".
    """
    if pre_prediction is None or getattr(pre_prediction, "expected_signature", None) is None:
        return None
    expected = pre_prediction.expected_signature
    actual = getattr(observation, "after_signature", None)
    if expected == actual:
        return True
    # Semantic hint fallbacks: map predicted labels onto observation flags.
    if expected == "new_family_frontier":
        return bool(getattr(observation, "new_family", False))
    if expected == "branch_escape":
        return bool(getattr(observation, "branch_escape", False))
    if expected == "new_signature":
        return bool(getattr(observation, "new_signature", False))
    return False


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

    B1: ``hypothesis_store`` is attached to the bridge during :meth:`load`
    unless ``config.hypothesis_store.enabled=False``. The attachment happens
    *before* the M5 / M6 / M7 wiring blocks so the abstraction engine and
    hierarchical router see a live store at construction time.
    :meth:`after_action` calls ``bridge.hypothesis_store.advance_turn()``
    once per env step, gated on the same live-store check.
    """

    def __init__(self, config: ResearchConfig, context: ResearchRuntimeContext):
        self.config = config
        self.context = context
        self.bridge = SharedBridge()
        self._modules: dict[str, Any] = {}
        # P3: phase event logger writes per-turn phase entries to
        # <shared_dir>/phase_events.jsonl. Imported lazily to avoid
        # circular imports during registry construction.
        from .phase_events import PhaseEventLogger
        self._phase_logger = PhaseEventLogger(context.shared_dir)
        self._turn_counter: int = 0

    def load(self) -> None:
        """Instantiate all enabled research modules and wire cross-module state.

        The HypothesisStore is attached on the bridge at the start of load()
        if not present. Attaching it after load() has no effect on the
        hierarchical router or abstraction engine which are built during
        load(): both read ``bridge.hypothesis_store`` once, at construction
        time, and keep that reference for their lifetime.

        If ``seed_library.initial_goals("L0")`` is available, each returned
        hypothesis payload is injected via ``store.propose_hypothesis`` so the
        goal tier has at least one active goal to route against from turn 1
        (plan §1.6).
        """
        # B1: ensure the HypothesisStore is attached to the bridge BEFORE any
        # module construction so downstream builders (abstraction engine,
        # hierarchical router) see a live store. The store is cheap and
        # universal — per-module opt-in is not supported.
        from .hypothesis_store import HypothesisStore
        hypo_cfg = getattr(self.config, "hypothesis_store", None)
        hypo_enabled = True if hypo_cfg is None else bool(
            getattr(hypo_cfg, "enabled", True)
        )
        if hypo_enabled and self.bridge.hypothesis_store is None:
            self.bridge.hypothesis_store = HypothesisStore()
            # Seed initial goals from the seed library if supported. Any
            # failure here is non-fatal — seeding is a convenience, not a
            # correctness requirement.
            try:
                from .modules import seed_library  # local import, optional
                initial_goals_fn = getattr(seed_library, "initial_goals", None)
                if callable(initial_goals_fn):
                    try:
                        goals = initial_goals_fn("L0") or []
                    except Exception:
                        logger.exception(
                            "seed_library.initial_goals raised; skipping seed"
                        )
                        goals = []
                    for payload in goals:
                        try:
                            self.bridge.hypothesis_store.propose_hypothesis(payload)
                        except Exception:
                            logger.exception(
                                "failed to inject seed goal hypothesis; continuing"
                            )
                    # Bootstrap seeding happens before the first real env step.
                    # Keep the seed hypotheses, but do not let them consume the
                    # first live Wake turn's proposal quota.
                    self.bridge.hypothesis_store.clear_turn_proposal_quota()
            except Exception:
                logger.exception("seed_library import failed; skipping seed goals")
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
        # M5 (plan v4.final §1.3 + §3.3.5): once both WM and DC are loaded,
        # wire the DreamCoder handle into the WM promotion engine so that
        # hypothesis promotion/demotion can fire the skill life-cycle
        # callbacks (provisional → committed → watchdog → retired).
        wm_module = self._modules.get("world_model")
        dc_module = self._modules.get("dreamcoder")
        if wm_module is not None and dc_module is not None:
            setter = getattr(wm_module, "set_dc_ref", None)
            if callable(setter):
                setter(dc_module)
        # M6 (plan v4.final §3.3 + §6): once DC / WM / planner / store are
        # all live, stand up the abstraction engine. Absent deps make the
        # engine a no-op — intentional, so partial-module configs (DC-only
        # tests) keep working without boot-time errors.
        self._abstraction_engine = self._maybe_build_abstraction_engine()
        self._turn_counter_for_abstraction = 0
        # M7 (plan v4.final §4): hierarchical router — read-only skill
        # selector over (goal tier → owner tier → MCTS → none). Built only
        # when the hypothesis store has been attached to the bridge; absent
        # dependencies leave ``_hierarchical_router = None`` so callers can
        # transparently fall back to the existing MCTS path.
        self._hierarchical_router = self._maybe_build_hierarchical_router()
        if self._modules:
            logger.info("Active research modules: %s", ", ".join(self._modules))

    def active(self) -> dict[str, Any]:
        return dict(self._modules)

    def get(self, name: str) -> Any | None:
        return self._modules.get(name)

    @staticmethod
    def _mcts_bypass_enabled() -> bool:
        """Return whether imagination/UCB bypass may commit real actions.

        Default is OFF so the online loop stays on the Wake LLM path.
        Set ``ARC_ENABLE_MCTS_BYPASS=1`` to re-enable the old behavior for
        ablations where committed planner skills should directly drive play.
        """
        raw = os.environ.get("ARC_ENABLE_MCTS_BYPASS", "0").strip()
        return raw not in ("", "0", "false", "False", "no", "off")

    # -- M6: abstraction engine wiring ------------------------------------
    def _maybe_build_abstraction_engine(self) -> Any | None:
        """Construct the M6 AbstractionEngine when deps are available.

        Returns ``None`` when the hypothesis store hasn't been attached to
        the bridge yet (common during early-boot tests) or when no
        DreamCoder / WorldModel is loaded — the engine needs at least one
        of those to do meaningful work.
        """
        store = getattr(self.bridge, "hypothesis_store", None)
        dc = self._modules.get("dreamcoder")
        wm = self._modules.get("world_model")
        if store is None and dc is None and wm is None:
            return None
        try:
            from .abstraction import AbstractionEngine
        except Exception:
            logger.exception("failed to import AbstractionEngine; M6 disabled")
            return None
        snapshot_path = self.context.shared_dir / "coevolution_metrics.csv"
        return AbstractionEngine(
            bridge=self.bridge,
            dreamcoder=dc,
            world_model=wm,
            hypothesis_store=store,
            llm_client=None,  # wired by the agent layer when TRAPI is live
            snapshot_path=snapshot_path,
        )

    def abstraction_engine(self) -> Any | None:
        """Expose the engine so agent-layer code can inject a TRAPI client."""
        return getattr(self, "_abstraction_engine", None)

    # -- M7: hierarchical router wiring ----------------------------------
    def _hierarchical_routing_enabled(self) -> bool:
        """Read the ``hierarchical_routing_enabled`` config flag.

        Looks for the flag on ``config.dreamcoder.params`` first (that is
        where agent-facing behaviour flags already live) and falls back to
        ``True`` when unset. A disabled flag causes :meth:`select_routed_skill`
        to short-circuit without consulting the router, mirroring the
        M6 / M5 opt-out pattern.
        """
        dc_cfg = getattr(self.config, "dreamcoder", None)
        params = getattr(dc_cfg, "params", None) if dc_cfg is not None else None
        if isinstance(params, dict) and "hierarchical_routing_enabled" in params:
            return bool(params["hierarchical_routing_enabled"])
        return True

    def _maybe_build_hierarchical_router(self) -> Any | None:
        """Construct the M7 :class:`HierarchicalRouter` when deps are live.

        Requires a hypothesis store on the bridge plus a DreamCoder module
        (the library source). Planner is optional — without it the router
        reports ``tier="none"`` instead of ``"mcts"``. Absent deps leave
        the router as ``None`` so callers transparently fall through to
        the legacy skill-selection path.
        """
        store = getattr(self.bridge, "hypothesis_store", None)
        dc = self._modules.get("dreamcoder")
        if store is None or dc is None:
            return None
        if not self._hierarchical_routing_enabled():
            return None
        try:
            from .routing import HierarchicalRouter
        except Exception:
            logger.exception("failed to import HierarchicalRouter; M7 disabled")
            return None
        return HierarchicalRouter(
            store=store,
            dreamcoder=dc,
            planner=self._modules.get("planner"),
            trigger_dsl_module=None,
        )

    def hierarchical_router(self) -> Any | None:
        """Expose the router so tests / agent code can drive it directly."""
        return getattr(self, "_hierarchical_router", None)

    def select_routed_skill(
        self,
        current_action_context: Any | None = None,
        recent_actions: Any | None = None,
    ) -> Any | None:
        """Return the router's skill pick when it is not MCTS / none.

        Updates ``shared_hints["routing_decision_last_turn"]`` with a
        serialised decision for observability. Returns ``None`` when:

          * the router is not available (disabled flag, missing deps),
          * the decision tier is ``mcts`` (caller should run planner), or
          * the decision tier is ``none``.
        """
        router = getattr(self, "_hierarchical_router", None)
        if router is None:
            return None
        try:
            decision = router.select_next_skill(
                current_action_context=current_action_context,
                recent_actions=recent_actions,
            )
        except Exception:
            logger.exception("hierarchical_router.select_next_skill failed")
            return None
        payload = {
            "tier": decision.tier,
            "hypothesis_id": decision.hypothesis_id,
            "score": decision.score,
            # W5: surface the tier-neutral unit tag so telemetry consumers
            # can reason about the score without re-deriving it from tier.
            "score_units": getattr(decision, "score_units", ""),
            "reason": decision.reason,
            "explanation": router.route_explanation(decision),
        }
        if decision.skill is not None:
            name = ""
            skill_payload = getattr(decision.skill, "payload", None)
            if isinstance(skill_payload, dict):
                name = str(skill_payload.get("name", "") or "")
            payload["skill_name"] = name
        self.bridge.update_hint("routing_decision_last_turn", payload)
        if decision.tier in ("mcts", "none"):
            return None
        return decision.skill

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
        # P3: open a wake phase at the start of every real-env action.
        self._turn_counter += 1
        self._phase_logger.enter("wake", turn=self._turn_counter)
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
        # M4: capture the predicted next-signature BEFORE modules run, because
        # world_model.after_action → bridge.record_observation() clears
        # ``current_prediction``. We read both the structured ``Prediction``
        # object and the synthesised hint stored in shared_hints so either
        # source can satisfy the was_correct comparison below.
        pre_prediction = getattr(self.bridge, "current_prediction", None)
        predicted_expected_signature: str | None = None
        if pre_prediction is not None and pre_prediction.action == action_name:
            predicted_expected_signature = pre_prediction.expected_signature
        if predicted_expected_signature is None:
            hint = self.bridge.read_hint("last_wm_prediction")
            if isinstance(hint, dict):
                raw = hint.get("next_signature_hint")
                if isinstance(raw, str):
                    predicted_expected_signature = raw
            elif isinstance(hint, str):
                predicted_expected_signature = hint
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
        # M4: feed the real-env outcome into the planner's per-class
        # world-model correctness estimator. ``was_correct`` is evaluated
        # via ``_was_prediction_correct`` (Issue #6) so semantic hints like
        # ``"new_family_frontier"`` don't collapse to 0 correctness just
        # because they fail to hash-match the post-step signature. We skip
        # silently when no prediction was captured (legacy configs or LLM
        # turns without a structured prediction) so this hook is a no-op
        # for runs that don't exercise the planner.
        planner_mod = self._modules.get("planner")
        if (
            planner_mod is not None
            and hasattr(planner_mod, "record_wm_outcome")
            and predicted_expected_signature is not None
        ):
            observation = self.bridge.last_observation
            if observation is not None and observation.action == action_name:
                # Build a synthetic Prediction-shaped object carrying the
                # expected_signature we captured above so _was_prediction_correct
                # can use its semantic-hint fallbacks regardless of whether
                # the original pre_prediction was an agent-emitted structured
                # Prediction or a shared-hint string.
                effective_pred = pre_prediction
                if (
                    effective_pred is None
                    or getattr(effective_pred, "expected_signature", None)
                    != predicted_expected_signature
                ):
                    from types import SimpleNamespace as _SN
                    effective_pred = _SN(
                        expected_signature=predicted_expected_signature,
                        action=action_name,
                    )
                was_correct = _was_prediction_correct(effective_pred, observation)
                if was_correct is not None:
                    before_family_id = (
                        observation.before_family
                        or observation.before_signature
                        or ""
                    )
                    try:
                        planner_mod.record_wm_outcome(
                            action_name, before_family_id, was_correct
                        )
                    except Exception:
                        logger.exception("planner.record_wm_outcome failed; continuing")
        # Plan v4 §2 + §3.1 M2: run hypothesis evaluation on every env step.
        # The HypothesisStore is only consulted when it has been wired onto
        # the bridge (M1 leaves it as None until the agent-side integration
        # lands). Disconfirms trigger the active skill abort policy.
        if getattr(self.bridge, "hypothesis_store", None) is not None:
            # B1: advance turn once per env step so hysteresis + no-trigger
            # retirement fire in production. Guarded on a live store.
            try:
                self.bridge.hypothesis_store.advance_turn()
            except Exception:
                logger.exception("hypothesis_store.advance_turn failed; continuing")
            helpers = self._get_symbolica_helpers()
            env_state = self._get_env_state_dict(after)
            try:
                hypo_events = self.bridge.evaluate_hypotheses(
                    action=action_name,
                    before=before,
                    after=after,
                    env_state=env_state,
                    helpers=helpers,
                )
            except Exception:
                logger.exception("evaluate_hypotheses failed; continuing")
                hypo_events = []
            if hypo_events:
                abort_reason = self.bridge.apply_abort_policy(hypo_events)
                if (
                    abort_reason == "disconfirm"
                    and self.bridge.current_skill_in_execution is not None
                ):
                    self.bridge.clear_committed_skill(reason="hypothesis_disconfirm")
                # P3-rev-B (§R7.3): wake three-object revisit. When a
                # hypothesis disconfirms, review the skill that was
                # executing + hypothesis statuses + queue WM draft rewrite.
                had_disconfirm = any(
                    e.kind == "disconfirm" for e in hypo_events
                )
                if had_disconfirm:
                    dc = self._modules.get("dreamcoder")
                    if dc is not None and hasattr(dc, "flag_skills_for_review"):
                        recent = list(self.bridge.recent_actions)[-4:]
                        try:
                            dc.flag_skills_for_review(recent)
                        except Exception:
                            logger.exception(
                                "flag_skills_for_review failed; continuing"
                            )
                    try:
                        self.bridge.hypothesis_store.revisit_on_wake(
                            action=action_name,
                            before=before,
                            after=after,
                            helpers=helpers,
                            kind="disconfirm",
                        )
                    except Exception:
                        logger.exception(
                            "hypothesis_store.revisit_on_wake failed; continuing"
                        )
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
        # M6 (plan v4.final §3.3 + §6): run the abstraction pass every
        # ``period`` turns (default 30) or when the hypothesis pool hits
        # its cap. The engine may be absent (partial-module test configs);
        # in that case this block is a no-op.
        self._turn_counter_for_abstraction = (
            getattr(self, "_turn_counter_for_abstraction", 0) + 1
        )
        engine = getattr(self, "_abstraction_engine", None)
        if engine is not None:
            try:
                engine.run_if_due(self._turn_counter_for_abstraction)
            except Exception:
                logger.exception("abstraction.run_if_due failed; continuing")
        # P3: close the wake phase. ``llm_bypassed`` is True when a
        # committed skill supplied the action and the LLM was not
        # invoked this turn (§R2.3). Default false; agent.py sets the
        # hint when it bypasses.
        bypassed = bool(self.bridge.shared_hints.pop("wake_llm_bypassed", False))
        try:
            self._phase_logger.exit(
                "wake",
                action=action_name,
                llm_bypassed=bypassed,
            )
        except Exception:
            logger.exception("phase_logger.exit(wake) failed; continuing")

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
        dc = self._modules.get("dreamcoder")
        if dc is not None and hasattr(dc, "suggest_fresh_opening_action"):
            try:
                opening_action = dc.suggest_fresh_opening_action(available_actions)
            except Exception:
                logger.exception("suggest_fresh_opening_action failed; falling back")
                opening_action = None
            if opening_action in available_actions:
                return opening_action
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
        if not self._mcts_bypass_enabled():
            self.bridge.update_hint("mcts_bypass_disabled", True)
            return None

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
        if self._should_defer_weak_mcts_bypass(result):
            self.bridge.update_hint("mcts_bypass_deferred", True)
            return None
        proposed = dc.propose_from_mcts(
            result,
            root_state_signature=sig,
            bfs_baseline=result.get("bfs_mean_reward", 0.0),
        )
        if proposed:
            best = self._select_mcts_proposal(proposed)
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

    def _select_mcts_proposal(self, proposed: list[Any]) -> Any:
        """Break equal-reward MCTS ties by preferring under-used first actions."""
        recent_counts = Counter(
            str(action).strip()
            for action in list(getattr(self.bridge, "recent_actions", []) or [])[-16:]
            if action
        )

        def _key(record: Any) -> tuple[float, float, float, float]:
            payload = getattr(record, "payload", {}) or {}
            reward = float(payload.get("mcts_simulator_reward", 0.0))
            visits = float(payload.get("mcts_visits", 0.0))
            body = list(
                payload.get("action_spine")
                or payload.get("controller")
                or payload.get("body")
                or []
            )
            first_action = str(body[0]).strip() if body else ""
            novelty = -float(recent_counts.get(first_action, 0))
            length = float(len(body))
            return (reward, novelty, visits, length)

        return max(proposed, key=_key)

    def _should_defer_weak_mcts_bypass(self, plan_result: dict[str, Any]) -> bool:
        """Let the LLM act when MCTS only has weak 1-step tie candidates.

        On ls20, once the planner opens it often emits several equal-reward
        single-step paths with tiny positive reward. Auto-committing those
        consumes turns that could instead be used for explicit hypothesis
        proposals and falsification. Defer only under clear local-attractor
        pressure so stronger multi-step options still bypass normally.
        """
        world_hint = self.bridge.read_hint("world_model")
        if not isinstance(world_hint, dict):
            return False
        try:
            pressure = int(world_hint.get("local_attractor_pressure", 0))
        except (TypeError, ValueError):
            pressure = 0
        if pressure < 3:
            return False
        paths = list(plan_result.get("mcts_top_paths") or [])
        if not paths:
            return False
        top_reward = float(paths[0][1])
        if top_reward > 0.5:
            return False
        epsilon = 1e-9
        tied = [seq for seq, reward, _visits in paths if abs(float(reward) - top_reward) <= epsilon]
        if len(tied) < 2:
            return False
        return all(len(seq) <= 1 for seq in tied)

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
        # M4 alias: planner.record_wm_outcome falls back to this hint when a
        # structured ``Prediction.expected_signature`` isn't available.
        self.bridge.update_hint("last_wm_prediction", predicted)
        # Also set bridge.current_prediction so world_model.record_observation
        # can score this transition as a unit test, same as an agent-emitted
        # prediction.
        from .bridge import Prediction as _P
        raw_expected_signature = predicted.get("next_signature_hint")
        expected_signature = None
        if isinstance(raw_expected_signature, str):
            candidate = raw_expected_signature.strip()
            semantic_hints = {
                "",
                "unknown",
                "changed",
                "same",
                "same_family",
                "new_family",
                "branch_escape",
            }
            if candidate and candidate.lower() not in semantic_hints:
                expected_signature = candidate
        self.bridge.record_prediction(_P(
            action=action,
            expected_change_summary=(
                ("EXPECT_CHANGE: " if predicted["expect_change"] else
                 "EXPECT_NO_CHANGE: " if predicted["expect_change"] is False else "")
                + "(synthesised from simulator)"
            ),
            observation_prediction=predicted["observation_prediction"],
            progress_prediction=predicted["progress_prediction"],
            expected_signature=expected_signature,
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
        # Persist the hypothesis store under shared_dir so subsequent runs
        # can load the accumulated catalogue. Best-effort; failures are
        # logged but do not block module on_run_end cleanup.
        store = getattr(self.bridge, "hypothesis_store", None)
        if store is not None:
            try:
                store.save(self.context.shared_dir / "hypothesis_store.json")
            except Exception:
                logger.exception(
                    "hypothesis_store.save failed; skipping persistence for this run"
                )
        # P3: flush any dangling phase entries.
        try:
            self._phase_logger.close_open_phases(reason="run_end")
        except Exception:
            logger.exception("phase_logger flush failed; continuing")

    # Agent-facing bridge API ---------------------------------------------
    def record_agent_prediction(self, action: str, expected_change: Any) -> None:
        wm = self._modules.get("world_model")
        if wm is not None and hasattr(wm, "record_agent_prediction"):
            wm.record_agent_prediction(action, expected_change)

    def record_agent_skill_proposal(
        self, payload: dict[str, Any]
    ) -> tuple[Any, dict[str, Any] | None]:
        """Route a skill proposal to DC with §P2 category enforcement.

        Returns ``(record_or_none, error_or_none)``. If no DC module is
        active, the bridge still records the proposal verbatim so the
        default-config smoke still exercises the proposal path, and
        ``error`` is None.
        """
        dc = self._modules.get("dreamcoder")
        if dc is not None and hasattr(dc, "record_agent_proposal"):
            return dc.record_agent_proposal(payload)
        # Fallback: no DC → record on bridge without category gating so
        # baseline tests still see a proposal stream.
        rec = self.bridge.record_proposed_skill(payload)
        return rec, None

    def record_agent_env_note(self, text: str) -> None:
        wm = self._modules.get("world_model")
        if wm is not None and hasattr(wm, "record_agent_env_note"):
            wm.record_agent_env_note(text)

    def record_agent_world_update(self, payload: Any) -> None:
        wm = self._modules.get("world_model")
        if wm is not None and hasattr(wm, "record_agent_world_update"):
            wm.record_agent_world_update(payload)

    def record_agent_hypothesis(
        self, payload: dict[str, Any]
    ) -> tuple[str | None, dict[str, Any] | None]:
        """Forward an agent-authored hypothesis payload to the store.

        Returns ``(h_id, error)`` per :meth:`HypothesisStore.propose_hypothesis_strict`.
        If the store is not attached (e.g. store disabled in config),
        returns ``(None, {"reason": "store_disabled"})`` so the agent tool
        layer can surface a uniform error shape.
        """
        store = getattr(self.bridge, "hypothesis_store", None)
        if store is None:
            return None, {"reason": "store_disabled"}
        payload = dict(payload)
        payload.setdefault("source", "agent")
        return store.propose_hypothesis_strict(payload)

    # -- M2: symbolica helper injection for hypothesis evaluation ---------
    def _get_symbolica_helpers(self) -> dict[str, Any]:
        """Return the 4 single-arg frame helpers used by the trigger DSL.

        Plan v4 §2.1 whitelists ``frame_color_counts``, ``frame_bounding_box``,
        ``frame_render``, and ``frame_find``. The agent-side templates expose
        these as Frame methods (``Frame.render``, ``Frame.color_counts``, …);
        we wrap each as a single-argument callable so a hypothesis
        ``check_code`` snippet like ``result = frame_color_counts(after) ==
        {...}`` works unchanged regardless of how the frame object was
        constructed (real ``Frame`` or a ``SimpleNamespace`` mock).

        Helpers always degrade safely: if the underlying frame does not
        expose the method, the wrapper returns ``None`` rather than raising,
        so a hypothesis using that helper simply falls through to ambiguous.
        """
        def _safe_call(frame: Any, attr: str, *args: Any) -> Any:
            if frame is None:
                return None
            fn = getattr(frame, attr, None)
            if fn is None:
                return None
            try:
                return fn(*args)
            except Exception:
                return None

        def _raw_grid(frame: Any) -> list | None:
            """Extract the current-level 2D grid from either a rich Frame
            (``frame.grid`` tuple) or a raw FrameData (``frame.frame[-1]``).
            Returns ``None`` when the shape is not recognisable."""
            grid = getattr(frame, "grid", None)
            if grid is not None:
                return [list(row) for row in grid]
            layers = getattr(frame, "frame", None)
            if layers:
                try:
                    return [list(row) for row in layers[-1]]
                except Exception:
                    return None
            return None

        def frame_color_counts(frame: Any) -> Any:
            # Prefer the Frame method when available.
            v = _safe_call(frame, "color_counts")
            if v is not None:
                return v
            # Fallback for raw FrameData: compute from grid.
            grid = _raw_grid(frame)
            if grid is None:
                return None
            counts: dict[int, int] = {}
            for row in grid:
                for c in row:
                    ic = int(c)
                    counts[ic] = counts.get(ic, 0) + 1
            return counts

        def frame_diff(before_frame: Any, after_frame: Any) -> Any:
            before_grid = _raw_grid(before_frame)
            after_grid = _raw_grid(after_frame)
            if before_grid is None or after_grid is None:
                return None
            out: list[tuple[int, int, int, int]] = []
            height = min(len(before_grid), len(after_grid))
            width = min(
                len(before_grid[0]) if before_grid else 0,
                len(after_grid[0]) if after_grid else 0,
            )
            for y in range(height):
                for x in range(width):
                    old = int(before_grid[y][x])
                    new = int(after_grid[y][x])
                    if old != new:
                        out.append((y, x, old, new))
            return out

        def frame_bounding_box(frame: Any, *values: Any) -> Any:
            v = _safe_call(frame, "bounding_box", *values)
            if v is not None:
                return v
            grid = _raw_grid(frame)
            if grid is None:
                return None
            wanted = {int(v) for v in values} if values else None
            ys = []; xs = []
            for y, row in enumerate(grid):
                for x, c in enumerate(row):
                    ic = int(c)
                    if wanted is not None:
                        if ic in wanted:
                            ys.append(y); xs.append(x)
                    elif ic != 0:
                        ys.append(y); xs.append(x)
            if not ys:
                return None
            return (min(xs), min(ys), max(xs) + 1, max(ys) + 1)

        def frame_render(frame: Any) -> Any:
            v = _safe_call(frame, "render")
            if v is not None:
                return v
            grid = _raw_grid(frame)
            if grid is None:
                return None
            # Minimal fallback text render for raw FrameData.
            return "\n".join(
                " ".join(str(int(c) & 0xF) for c in row) for row in grid
            )

        def frame_find(frame: Any, *values: Any) -> Any:
            v = _safe_call(frame, "find", *values)
            if v is not None:
                return v
            grid = _raw_grid(frame)
            if grid is None:
                return None
            wanted = {int(v) for v in values} if values else None
            return [
                (x, y, int(c))
                for y, row in enumerate(grid)
                for x, c in enumerate(row)
                if (wanted is None and int(c) != 0)
                or (wanted is not None and int(c) in wanted)
            ]

        return {
            "frame_diff": frame_diff,
            "frame_color_counts": frame_color_counts,
            "frame_bounding_box": frame_bounding_box,
            "frame_render": frame_render,
            "frame_find": frame_find,
        }

    def _get_env_state_dict(self, frame: Any) -> dict[str, Any]:
        """Return env_state used by hypothesis check_code snippets.

        Carries ``status`` + ``score`` (legacy) plus P14/P17 additions:
        ``turn`` (monotonic HypothesisStore turn counter) and
        ``goal_budget`` (ceil(MAX_ACTIONS × 0.3)). These let seed goal /
        budget hypotheses branch on turn-index and budget without
        needing a separate helper.
        """
        base: dict[str, Any] = {
            "status": None if frame is None else getattr(frame, "state", None),
            "score": None if frame is None else getattr(frame, "score", None),
        }
        store = getattr(self.bridge, "hypothesis_store", None)
        base["turn"] = int(store.current_turn) if store is not None else 0
        try:
            import os as _os
            max_actions = int(_os.environ.get("ARC_SMOKE_MAX_ACTIONS", "200"))
        except Exception:
            max_actions = 200
        base["goal_budget"] = max(1, int(max_actions * 0.3))
        base["levels_completed"] = (
            int(getattr(frame, "levels_completed", 0) or 0) if frame is not None else 0
        )
        return base
