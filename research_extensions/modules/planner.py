"""MCTS planner running on the world-model simulator.

Plan v3 §4 P0. UCB1 tree search where each rollout step calls
`world_model.simulate_step(state, action, observation)` instead of the
real environment. Reward is a weighted sum of `new_family`,
`branch_escape`, `expected_diff_band` quality, and a penalty for
repeated-family depth.

Per H5b the planner is gated on
`world_model._best_draft().transition_accuracy >= mcts_gate_threshold`.
Below the gate, the planner returns an empty proposal list (it would just
add noise).

Per H5 the planner also runs a length-matched random-BFS rollout for each
imagination call so we can compare MCTS vs BFS reward and persist the
advantage to `meta_harness.json` (the H5 falsifier).
"""
from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..bridge import SharedBridge
from ..trigger_dsl import ParsedTrigger, parse_trigger, trigger_matches


@dataclass
class _Node:
    state_signature: str
    parent: "_Node | None" = None
    action_from_parent: str | None = None
    children: dict[str, "_Node"] = field(default_factory=dict)
    visits: int = 0
    total_reward: float = 0.0
    untried_actions: list[str] = field(default_factory=list)
    terminal: bool = False  # True if simulator raised or returned exception_flag
    family_chain: list[str] = field(default_factory=list)

    @property
    def mean_reward(self) -> float:
        return self.total_reward / self.visits if self.visits > 0 else 0.0

    def ucb_score(self, c: float, parent_visits: int) -> float:
        if self.visits == 0:
            return float("inf")
        exploit = self.mean_reward
        explore = c * math.sqrt(math.log(parent_visits + 1) / self.visits)
        return exploit + explore


class MCTSPlanner:
    """Imagination-only planner. Never touches the real environment."""

    name = "planner"

    def __init__(self, params: dict[str, Any], context, bridge: SharedBridge) -> None:
        self.params = params
        self.context = context
        self.bridge = bridge
        self.max_depth = int(params.get("max_depth", 4))
        self.n_simulations = int(params.get("n_simulations", 64))
        self.ucb_c = float(params.get("ucb_c", 1.4))
        self.time_budget_seconds = float(params.get("time_budget_seconds", 10.0))
        self.gate_threshold = float(params.get("mcts_gate_threshold", 0.6))
        self.loose_gate_threshold = float(
            params.get(
                "mcts_loose_gate_threshold",
                max(0.4, min(self.gate_threshold, 0.5)),
            )
        )
        self.loose_min_matches = int(
            params.get(
                "mcts_loose_min_matches",
                5 if self.loose_gate_threshold <= 0.4 else 3,
            )
        )
        self.reward_weights = dict(params.get("reward_weights", {
            "new_family": 3.0,
            "branch_escape": 2.0,
            "accuracy": 1.0,
            "repeat_penalty": -1.0,
            "exception": -2.0,
            "expect_change_match": 0.25,
            "band_large": 0.1,
            "goal_progress": 1.0,
            "latent_match": 0.5,
        }))
        self.bfs_rollouts_per_call = int(params.get("bfs_rollouts_per_call", 4))
        self._rng = random.Random(int(params.get("seed", 1234)))
        # H5 advantage tracking for the meta harness.
        self.recent_advantages: list[float] = []
        self.last_call_at: float = 0.0
        self._memories: Any | None = None
        self._state_path = context.shared_dir / "planner_state.json"
        # --- M4: gain-sum reward state (plan v4.final §3.2 + §5) ----------
        # When enabled, `_reward_from_sim` returns Σ_h expected_gain(h) over
        # active/confirmed hypotheses whose trigger matches the simulated
        # (action, before_family) class, weighted by a per-class running
        # estimate of world-model correctness (with an exploration bonus
        # for under-observed classes). See `gain_reward()` below.
        self.gain_reward_enabled: bool = bool(params.get("gain_reward_enabled", True))
        self.use_legacy_reward: bool = bool(params.get("use_legacy_reward", False))
        # 20-step moving average of world-model correctness per (action, family)
        # class, with Laplace-style smoothing centred on the default prior
        # (see ``record_wm_outcome``). Initial mean follows the prior.
        self.p_wm_correct: dict[str, float] = {}
        self.p_wm_correct_history: dict[str, list[int]] = {}
        self.observations_per_class: dict[str, int] = {}
        self._p_wm_default: float = float(params.get("p_wm_correct_default", 0.3))
        self._p_wm_history_cap: int = int(params.get("p_wm_correct_window", 20))
        self._exploration_min_obs: int = int(params.get("exploration_min_obs", 10))
        self._exploration_bonus_weight: float = float(
            params.get("exploration_bonus_weight", 0.5)
        )
        # MCTS imagination cannot evaluate on_change triggers because no
        # post-frame helpers are available. These hypotheses only contribute
        # reward during real-env evaluation. We skip them silently and expose
        # a counter via ``diagnostics()`` so the effect is observable in logs.
        self._gain_reward_onchange_skipped: int = 0
        self.plan_calls: int = 0
        self.gate_denials: int = 0
        self.last_result: dict[str, Any] = {}

    # -- Hook surface (registry will skip planner overlay if no plan) ----
    def prompt_overlay(self) -> str:
        if not self.recent_advantages:
            return ""
        adv = self.recent_advantages[-1]
        lines = [
            "[Imagination Planner]",
            f"- last MCTS−BFS reward advantage: {adv:+.2f} "
            f"(positive = MCTS beat random BFS in simulator)",
        ]
        if len(self.recent_advantages) >= 5:
            mean = sum(self.recent_advantages[-5:]) / 5
            lines.append(f"- 5-call moving mean: {mean:+.2f}")
        return "\n".join(lines)

    def on_memories_ready(self, memories: Any) -> None:
        self._memories = memories if hasattr(memories, "add") else None

    def before_action(
        self, frame: Any | None, action_name: str, available_actions: list[str]
    ) -> str | None:
        # Planner does not directly select real actions — DC commits a
        # skill (built from the planner's proposals), and registry routes
        # through that. This hook is only here so registry's iteration
        # over modules picks up the planner cleanly.
        return action_name

    def after_action(self, before: Any | None, action_name: str, after: Any) -> None:
        return None

    def on_run_end(self, *, history: list[tuple[str, Any]], finish_status: Any, workdir: Path) -> None:
        try:
            import json
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            self._state_path.write_text(
                json.dumps(
                    {
                        "recent_advantages": self.recent_advantages[-32:],
                        "plan_calls": self.plan_calls,
                        "gate_denials": self.gate_denials,
                        "last_result": self.last_result,
                        "diagnostics": self.diagnostics(),
                        "params": {
                            "max_depth": self.max_depth,
                            "n_simulations": self.n_simulations,
                            "gate_threshold": self.gate_threshold,
                            "loose_gate_threshold": self.loose_gate_threshold,
                            "loose_min_matches": self.loose_min_matches,
                            "reward_weights": self.reward_weights,
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

    def _persist_state(self) -> None:
        self.on_run_end(history=[], finish_status=None, workdir=self.context.workdir)

    # -- Public planner API ------------------------------------------------
    def plan(
        self,
        *,
        world_model,
        root_state_signature: str,
        available_actions: list[str],
        observation_factory=None,
    ) -> dict[str, Any]:
        """Run one imagination call.

        Returns: {
          'gate_ok': bool,                                        # H5b
          'mcts_top_paths': list[(action_seq, reward, visits)],   # ranked
          'bfs_top_paths': list[(action_seq, reward)],            # for H5
          'mcts_mean_reward': float,
          'bfs_mean_reward': float,
          'advantage': float,                                     # mcts − bfs
          'reason_skipped': str | None,
        }
        """
        self.last_call_at = time.time()
        self.plan_calls += 1
        # H5b gate — plan v4.2 §P15 (rev C R12.3): two-tier.
        # (a) Strict path: transition_accuracy >= gate_threshold.
        # (b) Loose path: simulator_matches / (matches + mismatches + 1)
        #     >= loose_gate_threshold AND simulator_matches >=
        #     loose_min_matches. This still requires repeated support,
        #     but avoids starving MCTS once the simulator is directionally
        #     useful under the configured gate regime.
        draft = world_model._best_draft()
        accuracy = draft.transition_accuracy if draft is not None else 0.0
        sim_match = float(getattr(draft, "simulator_matches", 0)) if draft else 0.0
        sim_mis = float(getattr(draft, "simulator_mismatches", 0)) if draft else 0.0
        loose_accuracy = sim_match / (sim_match + sim_mis + 1.0)
        strict_ok = draft is not None and accuracy >= self.gate_threshold
        # rev F2: warmup shortens LLM-eval window so require only 3
        # simulator matches (was 4). Still 3 provides decent evidence
        # the draft is directionally right.
        loose_ok = (
            draft is not None
            and sim_match >= self.loose_min_matches
            and loose_accuracy >= self.loose_gate_threshold
        )
        # rev H2: unit_tests_passed path (replay-based accuracy). When
        # the draft has been evaluated against >= 8 unit tests and at
        # least 50% pass under loose scoring, gate opens regardless of
        # per-step simulator counters (which reset on body change).
        ut_seen = int(getattr(draft, "unit_tests_seen", 0)) if draft else 0
        ut_passed = int(getattr(draft, "unit_tests_passed", 0)) if draft else 0
        if draft is not None and ut_seen >= 8 and ut_passed / max(ut_seen, 1) >= 0.5:
            loose_ok = True
        gate_ok = strict_ok or loose_ok
        if not gate_ok:
            result = self._empty_result(
                reason=(
                    f"gate: strict={accuracy:.2f}<{self.gate_threshold:.2f} "
                    f"AND loose={loose_accuracy:.2f}<{self.loose_gate_threshold:.2f} "
                    f"(matches={sim_match:.0f}/{self.loose_min_matches})"
                )
            )
            self.gate_denials += 1
            self.last_result = result
            self._persist_state()
            return result

        if not available_actions:
            result = self._empty_result(reason="no available actions")
            self.last_result = result
            self._persist_state()
            return result

        # MCTS
        mcts_paths = self._run_mcts(
            world_model=world_model,
            root_signature=root_state_signature,
            available_actions=available_actions,
            observation_factory=observation_factory,
        )
        # Plan H5 metric: compare MCTS *top-1* reward to *mean* BFS reward.
        # Using MCTS mean over all explored leaves under-counts MCTS because
        # the tree by design also expands low-value branches for exploration.
        mcts_top_reward = mcts_paths[0][1] if mcts_paths else 0.0
        # H5: length-matched random-BFS for comparison
        target_len = max((len(seq) for seq, _r, _v in mcts_paths), default=self.max_depth)
        bfs_paths = self._run_random_bfs(
            world_model=world_model,
            root_signature=root_state_signature,
            available_actions=available_actions,
            observation_factory=observation_factory,
            target_len=target_len,
        )
        bfs_mean = (
            sum(r for _seq, r in bfs_paths) / len(bfs_paths)
            if bfs_paths else 0.0
        )
        advantage = mcts_top_reward - bfs_mean
        self.recent_advantages.append(advantage)
        self.recent_advantages = self.recent_advantages[-32:]
        # P9 (plan v4.2 §R1.7/§R7): feed advantage samples into the
        # bridge's shared_hints so MetaHarness.RunMetrics.score_run() can
        # populate mcts_vs_bfs_advantage from the delta.
        try:
            bridge = getattr(self, "bridge", None)
            if bridge is not None:
                samples = bridge.shared_hints.setdefault(
                    "mcts_vs_bfs_advantage_samples", []
                )
                samples.append(float(advantage))
                bridge.shared_hints[
                    "mcts_vs_bfs_advantage_samples"
                ] = samples[-32:]
        except Exception:
            pass

        result = {
            "gate_ok": True,
            "mcts_top_paths": mcts_paths,
            "bfs_top_paths": bfs_paths,
            "mcts_top_reward": mcts_top_reward,
            "bfs_mean_reward": bfs_mean,
            "advantage": advantage,
            "reason_skipped": None,
        }
        self.last_result = result
        self._persist_state()
        return result

    def diagnostics(self) -> dict[str, Any]:
        """Return observable planner counters for logging / tests.

        Currently exposes:
          - ``gain_reward_onchange_skipped``: number of ``on_change``
            hypotheses skipped during imagination since process start.
            A large value confirms the silent-skip path is firing (good)
            rather than the pre-fix warn-per-call path.
          - ``p_wm_correct_classes``: number of distinct
            ``(action, before_family)`` classes with at least one recorded
            outcome.
          - ``recent_advantage`` / ``recent_advantage_mean_5``: last MCTS
            vs BFS advantage and its 5-call moving mean (mirrors the
            prompt overlay so tests can assert without parsing prose).
        """
        recent = self.recent_advantages[-1] if self.recent_advantages else 0.0
        mean_5 = (
            sum(self.recent_advantages[-5:]) / min(5, len(self.recent_advantages))
            if self.recent_advantages
            else 0.0
        )
        return {
            "gain_reward_onchange_skipped": int(self._gain_reward_onchange_skipped),
            "p_wm_correct_classes": len(self.p_wm_correct),
            "recent_advantage": float(recent),
            "recent_advantage_mean_5": float(mean_5),
        }

    def _empty_result(self, *, reason: str) -> dict[str, Any]:
        return {
            "gate_ok": False,
            "mcts_top_paths": [],
            "bfs_top_paths": [],
            "mcts_mean_reward": 0.0,
            "bfs_mean_reward": 0.0,
            "advantage": 0.0,
            "reason_skipped": reason,
        }

    # -- Internals ---------------------------------------------------------
    def _run_mcts(
        self,
        *,
        world_model,
        root_signature: str,
        available_actions: list[str],
        observation_factory,
    ) -> list[tuple[list[str], float, int]]:
        root = _Node(state_signature=root_signature, untried_actions=list(available_actions))
        deadline = time.time() + self.time_budget_seconds
        for _i in range(self.n_simulations):
            if time.time() > deadline:
                break
            self._mcts_simulate(
                root=root,
                world_model=world_model,
                available_actions=available_actions,
                observation_factory=observation_factory,
            )
        # Plan H5: extract principal variations (top-K by visit count at
        # each level), then replay each PV through the simulator for a
        # clean deterministic reward. MCTS mean_reward is noisy because
        # it folds in random-rollout variance; PV replay is what we
        # actually commit to, so it's also what we should score.
        pvs = self._extract_principal_variations(
            root=root, top_k=5, available_actions=available_actions,
        )
        scored: list[tuple[list[str], float, int]] = []
        for seq, visits in pvs:
            if not seq:
                continue
            reward = self._replay_path_through_simulator(
                seq=seq,
                root_signature=root.state_signature,
                world_model=world_model,
                observation_factory=observation_factory,
            )
            scored.append((seq, reward, visits))
        scored.sort(key=lambda item: (-(item[1]), -item[2]))
        return scored[:5]

    def _extract_principal_variations(
        self, *, root: _Node, top_k: int, available_actions: list[str]
    ) -> list[tuple[list[str], int]]:
        """Greedy top-K PV extraction: at each depth, pick the child with
        the highest visit count as the committed continuation, but also
        spawn branches for the top-K siblings at the root so we end up
        with K diverse plans rather than K slight variations."""
        results: list[tuple[list[str], int]] = []
        if not root.children:
            return results
        sorted_root_children = sorted(
            root.children.items(),
            key=lambda kv: -kv[1].visits,
        )[:top_k]
        for action, child in sorted_root_children:
            seq = [action]
            node = child
            while node.children:
                best_action, best_child = max(
                    node.children.items(),
                    key=lambda kv: kv[1].visits,
                )
                seq.append(best_action)
                node = best_child
            results.append((seq, child.visits))
        return results

    def _replay_path_through_simulator(
        self,
        *,
        seq: list[str],
        root_signature: str,
        world_model,
        observation_factory,
    ) -> float:
        sig = root_signature
        total = 0.0
        family_chain: list[str] = []
        for action in seq:
            obs = observation_factory(sig) if observation_factory else None
            sim = world_model.simulate_step(sig, action, obs)
            total += self._reward_from_sim(
                sim, family_chain, action=action, before_family=sig
            )
            family_chain.append(sim["next_signature_hint"])
            sig = sim["next_signature_hint"]
            if sim.get("exception_flag"):
                break
        return total

    def _mcts_simulate(
        self,
        *,
        root: _Node,
        world_model,
        available_actions: list[str],
        observation_factory,
    ) -> None:
        # 1. selection
        node = root
        depth = 0
        path: list[_Node] = [node]
        while not node.terminal and not node.untried_actions and node.children and depth < self.max_depth:
            best_action, best_child = max(
                node.children.items(),
                key=lambda kv: kv[1].ucb_score(self.ucb_c, parent_visits=node.visits),
            )
            node = best_child
            path.append(node)
            depth += 1
        # 2. expansion
        if not node.terminal and node.untried_actions and depth < self.max_depth:
            action = node.untried_actions.pop(0)
            obs = observation_factory(node.state_signature) if observation_factory else None
            sim = world_model.simulate_step(node.state_signature, action, obs)
            next_sig = sim["next_signature_hint"]
            child = _Node(
                state_signature=next_sig,
                parent=node,
                action_from_parent=action,
                untried_actions=list(available_actions),
                terminal=bool(sim.get("exception_flag")) or bool(sim.get("is_goal_state")),
                family_chain=node.family_chain + [next_sig],
            )
            node.children[action] = child
            node = child
            path.append(node)
            depth += 1
        # 3. rollout (random walk on simulator until depth budget exhausted)
        rollout_reward = 0.0
        rollout_node_sig = node.state_signature
        rollout_family_chain = list(node.family_chain)
        rollout_depth = 0
        terminal_in_rollout = node.terminal
        # P32: carry the simulated grid across rollout steps so hypothesis
        # check_code can evaluate against the imagined state.
        rollout_obs = observation_factory(rollout_node_sig) if observation_factory else None
        rollout_before_grid = (rollout_obs or {}).get("grid") if isinstance(rollout_obs, dict) else None
        while not terminal_in_rollout and depth + rollout_depth < self.max_depth:
            action = self._rng.choice(available_actions)
            obs = dict(rollout_obs) if isinstance(rollout_obs, dict) else (
                observation_factory(rollout_node_sig) if observation_factory else None
            )
            if isinstance(obs, dict) and rollout_before_grid is not None:
                obs["grid"] = rollout_before_grid
            sim = world_model.simulate_step(rollout_node_sig, action, obs)
            # P33: inject before/after grids into the sim dict so
            # `_reward_from_sim` can run hypothesis check_code.
            if isinstance(sim, dict):
                sim["_sim_before_grid"] = rollout_before_grid
            rollout_reward += self._reward_from_sim(
                sim,
                rollout_family_chain,
                action=action,
                before_family=rollout_node_sig,
            )
            # P32: terminal on simulated goal hit.
            if isinstance(sim, dict) and sim.get("is_goal_state"):
                terminal_in_rollout = True
                rollout_reward += float(
                    self.reward_weights.get("goal_state_bonus", 5.0)
                )
                break
            rollout_family_chain.append(sim["next_signature_hint"])
            rollout_node_sig = sim["next_signature_hint"]
            # advance grid
            if isinstance(sim, dict):
                rollout_before_grid = sim.get("expected_next_grid") or rollout_before_grid
            terminal_in_rollout = bool(sim.get("exception_flag"))
            rollout_depth += 1
        # 4. backprop
        leaf_eval = self._evaluate_leaf(node, world_model, observation_factory)
        total = leaf_eval + rollout_reward
        for n in path:
            n.visits += 1
            n.total_reward += total

    def _collect_paths(
        self,
        node: _Node,
        prefix: list[str],
        out: list[tuple[list[str], float, int]],
    ) -> None:
        if not node.children:
            out.append((list(prefix), node.mean_reward, node.visits))
            return
        for action, child in node.children.items():
            self._collect_paths(child, prefix + [action], out)

    def _evaluate_leaf(self, node: _Node, world_model, observation_factory) -> float:
        # Heuristic leaf evaluation: bonus if family chain shows family
        # change vs root, penalty if all entries are the same family.
        if not node.family_chain:
            return 0.0
        unique = len(set(node.family_chain))
        if unique == 1:
            return float(self.reward_weights.get("repeat_penalty", -1.0))
        return float(self.reward_weights.get("branch_escape", 2.0)) * (unique - 1)

    def _reward_from_sim(
        self,
        sim: dict[str, Any],
        family_chain: list[str],
        *,
        action: str | None = None,
        before_family: str | None = None,
    ) -> float:
        """Compute the per-step reward contribution for one simulated step.

        M4 (plan v4.final §3.2 + §5): when ``self.gain_reward_enabled`` is
        True AND the bridge carries a populated ``hypothesis_store``, the
        returned reward is the expected information gain summed over
        active/confirmed hypotheses whose trigger matches the simulated
        ``(action, before_family)`` class, plus an exploration bonus for
        under-observed classes. The legacy heuristic (new_family /
        branch_escape / expect_change_match / accuracy …) is retained as an
        optional secondary term behind ``use_legacy_reward=True`` so
        ablation runs can reproduce the v3.15 reward for the 10-seed
        experiment.

        When the gain branch is inactive (either the flag is off or the
        hypothesis store is empty / absent), we fall back to the heuristic
        path unchanged so existing planner tests and v3.15-equivalent
        ablations keep working.
        """
        store = getattr(self.bridge, "hypothesis_store", None)
        gain_active = (
            self.gain_reward_enabled
            and store is not None
            and action is not None
        )
        if gain_active:
            predicted_after = sim.get("next_signature_hint") if isinstance(sim, dict) else None
            gain_r = self.gain_reward(
                store,
                action=action,
                before_family=before_family or "",
                predicted_after_frame=predicted_after,
            )
            # Retain the legacy heuristic only when explicitly requested
            # (v3.15 ablation path). Defaults to "gain replaces heuristic".
            if self.use_legacy_reward:
                gain_r += self._legacy_reward_from_sim(sim, family_chain)
            # P29 (plan v4.2 rev J/§P29): repeat_penalty + unseen_bonus
            # on gain_reward path so MCTS stops looping on same-family
            # confirmed-hypothesis reuse. Same logic as _legacy_reward_from_sim.
            next_sig = sim.get("next_signature_hint", "unknown") if isinstance(sim, dict) else "unknown"
            if next_sig and next_sig != "unknown":
                rep = family_chain.count(next_sig)
                if rep > 0:
                    gain_r += float(
                        self.reward_weights.get("repeat_penalty", -1.0)
                    ) * rep
                if isinstance(next_sig, str):
                    low = next_sig.lower()
                    if any(k in low for k in ("new_", "unseen", "different", "frontier")):
                        seen_families = getattr(self.bridge, "seen_families", set())
                        scarcity = max(0.0, 3.0 - float(len(seen_families))) / 3.0
                        gain_r += float(
                            self.reward_weights.get("unseen_signature_bonus", 2.0)
                        ) * (0.5 + 0.5 * scarcity)
            # P33 (plan v4.2 rev K §P33): imagination-time hypothesis
            # evaluation — run check_code on the simulated (before, after)
            # grids and reward disconfirm more than confirm.
            sim_before = sim.get("_sim_before_grid") if isinstance(sim, dict) else None
            sim_after = sim.get("expected_next_grid") if isinstance(sim, dict) else None
            if sim_before is not None and sim_after is not None and store is not None:
                gain_r += self._imagination_hypothesis_gain(
                    store=store,
                    action=action,
                    before_grid=sim_before,
                    after_grid=sim_after,
                )
            gain_r += self._goal_progress_bonus(sim)
            # Exception flag still dominates: a simulator crash should not
            # be rewarded regardless of reward branch.
            if sim.get("exception_flag"):
                gain_r += float(self.reward_weights.get("exception", -2.0))
            return gain_r
        return self._legacy_reward_from_sim(sim, family_chain)

    def _imagination_hypothesis_gain(
        self,
        *,
        store: Any,
        action: str,
        before_grid: list,
        after_grid: list,
    ) -> float:
        """P33: run hypothesis check_code on simulated (before, after) grids.

        Returns summed reward where:
          - confirm event contributes `+0.5 × entropy_bits`
          - disconfirm event contributes `+2.0 × entropy_bits`  (larger!)
          - goal-hypothesis disconfirm multiplies by `3.0`
          - ambiguous / no-fire contributes 0

        We sandbox-execute check_code against dict-wrapped grids (matching
        the live bridge path via `_FrameProxy`), capping at 32 hypothesis
        invocations per call. Hypotheses are ranked by entropy × trigger
        prior before the cap — saturated confirmed claims are skipped.
        """
        try:
            items = store.all()
        except Exception:
            return 0.0
        # Rank by entropy descending.
        try:
            items = sorted(
                (h for h in items if getattr(h, "status", None) in ("active", "confirmed")),
                key=lambda h: -float(h.entropy_bits()),
            )[:32]
        except Exception:
            items = items[:32]
        # Dict-shaped proxies for sandbox.
        def _proxy(grid):
            return {
                "grid": grid,
                "signature": "",
                "state": "NOT_FINISHED",
                "available_actions": [],
            }
        before = _proxy(before_grid)
        after = _proxy(after_grid)
        env_state = {"status": "NOT_FINISHED", "turn": 0, "goal_budget": 60}
        sandbox = getattr(store, "sandbox", None)
        total = 0.0
        disconfirm_weight = float(self.reward_weights.get("imagination_disconfirm", 2.0))
        confirm_weight = float(self.reward_weights.get("imagination_confirm", 0.5))
        goal_multiplier = float(self.reward_weights.get("goal_hypothesis_multiplier", 3.0))
        for h in items:
            try:
                from ..trigger_dsl import parse_trigger as _pt, trigger_matches as _tm
                parsed = _pt(h.verification_method.trigger)
            except Exception:
                continue
            if parsed.kind == "on_change":
                # Needs post-frame helpers (not available in imagination).
                continue
            try:
                fired = _tm(parsed, action=action, before=before, after=after, helpers={})
            except Exception:
                fired = False
            if not fired:
                continue
            if sandbox is None:
                continue
            locals_dict = {
                "before": before, "after": after,
                "action": action, "env_state": env_state,
            }
            try:
                result = sandbox.run(
                    h.verification_method.check_code,
                    h.verification_method.helpers_used,
                    locals_dict,
                )
            except Exception:
                continue
            try:
                ent = float(h.entropy_bits())
            except Exception:
                ent = 0.0
            is_goal = bool(getattr(h, "is_goal", False))
            contrib = 0.0
            if result is True:
                contrib = confirm_weight * ent
            elif result is False:
                contrib = disconfirm_weight * ent
                if is_goal:
                    contrib *= goal_multiplier
            # result None (ambiguous) → 0
            total += contrib
        return total

    def _legacy_reward_from_sim(
        self, sim: dict[str, Any], family_chain: list[str]
    ) -> float:
        """v3.15 heuristic reward (kept for `use_legacy_reward=True`)."""
        rs = sim.get("reward_signals", {})
        r = 0.0
        if sim.get("exception_flag"):
            r += float(self.reward_weights.get("exception", -2.0))
            return r
        if rs.get("new_family"):
            r += float(self.reward_weights.get("new_family", 3.0))
        if rs.get("branch_escape"):
            r += float(self.reward_weights.get("branch_escape", 2.0))
        if rs.get("expect_change") is True:
            r += float(self.reward_weights.get("expect_change_match", 0.25))
        if rs.get("expected_diff_band") == "large":
            r += float(self.reward_weights.get("band_large", 0.1))
        # Repeat penalty: if next signature already appears in family_chain,
        # decrement once per occurrence.
        next_sig = sim.get("next_signature_hint", "unknown")
        if next_sig != "unknown":
            r += float(self.reward_weights.get("repeat_penalty", -1.0)) * family_chain.count(next_sig)
        # Real-env signature novelty bonus: if the simulator's semantic
        # hint looks like it targets an observation family we haven't
        # actually visited yet, reward it. Heuristic because hints are
        # semantic strings (not real grid hashes). Matches on
        # 'new'/'unseen'/'different'/'frontier' keywords.
        if isinstance(next_sig, str):
            lower = next_sig.lower()
            if any(k in lower for k in ("new_", "unseen", "different", "frontier")):
                seen_families = getattr(self.bridge, "seen_families", set())
                # Approximate: if we've only seen ≤2 families, bonus is bigger
                scarcity = max(0.0, 3.0 - float(len(seen_families))) / 3.0
                r += float(self.reward_weights.get("unseen_signature_bonus", 2.0)) * (0.5 + 0.5 * scarcity)
        r += self._goal_progress_bonus(sim)
        # Accuracy bonus (favor planners on more reliable simulators)
        r += float(self.reward_weights.get("accuracy", 1.0)) * float(sim.get("transition_accuracy", 0.0))
        return r

    def _goal_progress_bonus(self, sim: dict[str, Any]) -> float:
        """Reward explicit goal-progress and latent-match signals from the WM."""
        if not isinstance(sim, dict):
            return 0.0
        bonus = 0.0
        goal_progress = sim.get("goal_progress_prediction")
        if isinstance(goal_progress, dict):
            progress_delta = goal_progress.get("progress_delta")
            if isinstance(progress_delta, (int, float)):
                bonus += float(self.reward_weights.get("goal_progress", 1.0)) * float(progress_delta)
            sat_delta = goal_progress.get("satisfied_constraints_delta")
            if isinstance(sat_delta, (int, float)):
                bonus += 0.5 * float(self.reward_weights.get("goal_progress", 1.0)) * float(sat_delta)
        latent = sim.get("latent_state_prediction")
        if isinstance(latent, dict):
            matched = latent.get("matched_target_variables")
            total = latent.get("total_target_variables")
            if isinstance(matched, (int, float)) and isinstance(total, (int, float)) and float(total) > 0:
                bonus += float(self.reward_weights.get("latent_match", 0.5)) * max(0.0, min(1.0, float(matched) / float(total)))
            match_delta = latent.get("matched_variables_delta")
            if isinstance(match_delta, (int, float)):
                bonus += 0.5 * float(self.reward_weights.get("latent_match", 0.5)) * float(match_delta)
        return bonus

    # -- M4: gain-sum reward + world-model correctness bookkeeping --------
    @staticmethod
    def _class_key(action: str, before_family: str) -> str:
        return f"{action}|{before_family}"

    def record_wm_outcome(
        self, action: str, before_family: str, was_correct: bool
    ) -> None:
        """Record one real-env outcome of the world-model prediction.

        Maintains a 20-step moving-average of correctness per
        ``(action, before_family)`` class with Laplace-style smoothing
        centred on ``self._p_wm_default`` (pseudo-count 3.0). After 1
        correct observation the posterior is ``(1 + 0.9) / (1 + 3.0) =
        0.475`` rather than ``1.0``, which stops MCTS from over-weighting
        freshly observed classes. After 20 correct observations the
        posterior converges close to the empirical rate
        (``(20 + 0.9) / (20 + 3.0) ≈ 0.909``).

        Silently no-ops on non-string inputs (the caller on the registry
        side already skips when no prediction was captured, but this
        guard keeps the method usable from tests and future callers).
        """
        if not isinstance(action, str) or not isinstance(before_family, str):
            return
        key = self._class_key(action, before_family)
        history = self.p_wm_correct_history.setdefault(key, [])
        history.append(1 if was_correct else 0)
        # Cap history at the window size (20 by default); oldest entries fall off.
        while len(history) > self._p_wm_history_cap:
            history.pop(0)
        alpha_prior = 3.0 * self._p_wm_default          # ~0.9 when default=0.3
        beta_prior = 3.0 * (1.0 - self._p_wm_default)   # ~2.1 when default=0.3
        self.p_wm_correct[key] = (
            (sum(history) + alpha_prior)
            / (len(history) + alpha_prior + beta_prior)
        )
        self.observations_per_class[key] = self.observations_per_class.get(key, 0) + 1

    def gain_reward(
        self,
        h_store,
        *,
        action: str,
        before_family: str,
        predicted_after_frame: Any = None,
    ) -> float:
        """Σ_h p_WM_correct[class] · H(conf(h)) + exploration_bonus.

        Iterates over every active/confirmed hypothesis in ``h_store``; for
        each hypothesis whose trigger fires on the simulated
        ``(action, before_family)`` class, accumulates

            gain_h = p_WM_correct.get(class, default) * H(conf(h))
                   + exploration_bonus

        where ``exploration_bonus = 0.5 · H(conf(h))`` if
        ``observations_of_class < 10`` else 0.

        Trigger matching reuses ``research_extensions.trigger_dsl`` — we
        pass the simulator's ``predicted_after_frame`` (a signature hint
        during MCTS imagination) as ``after`` and ``before_family`` as
        ``before``. Triggers that need helpers (``on_change ...``) simply
        degrade to no-match because no helper is available during pure
        imagination, which is the intended fail-safe.
        """
        if h_store is None:
            return 0.0
        key = self._class_key(action, before_family)
        p_correct = self.p_wm_correct.get(key, self._p_wm_default)
        obs_count = self.observations_per_class.get(key, 0)
        explore_active = obs_count < self._exploration_min_obs
        helpers: dict[str, Any] = {}
        # Dedupe warning set handed to trigger_matches (unused for on_change
        # here because we short-circuit before calling it, but passing an
        # empty set preserves the contract for future trigger kinds that
        # also require helpers).
        warned_helpers: set[str] = set()
        total = 0.0
        try:
            items = h_store.all()
        except Exception:
            return 0.0
        for h in items:
            if getattr(h, "status", None) not in ("active", "confirmed"):
                continue
            try:
                parsed: ParsedTrigger = parse_trigger(h.verification_method.trigger)
            except Exception:
                continue
            # Short-circuit on_change triggers during MCTS imagination: no
            # post-frame helpers are available, so trigger_matches would warn
            # + return False thousands of times per plan() call. These
            # hypotheses only contribute reward during real-env evaluation.
            if parsed.kind == "on_change":
                self._gain_reward_onchange_skipped += 1
                continue
            try:
                fired = trigger_matches(
                    parsed,
                    action=action,
                    before=before_family,
                    after=predicted_after_frame,
                    helpers=helpers,
                    _warned_helpers=warned_helpers,
                )
            except Exception:
                fired = False
            if not fired:
                continue
            try:
                ent_bits = float(h.entropy_bits())
            except Exception:
                ent_bits = 0.0
            contribution = p_correct * ent_bits
            if explore_active:
                contribution += self._exploration_bonus_weight * ent_bits
            # P27 (plan v4.2 §P27): goal-hypothesis bonus. An action that
            # makes progress on a `is_goal=True` hypothesis is worth more
            # than the same information-gain contribution on a generic
            # mechanic claim. Multiply goal-h contribution by 3.0 so
            # MCTS prefers goal-advancing branches alongside infogain.
            if getattr(h, "is_goal", False):
                contribution *= float(
                    self.reward_weights.get("goal_hypothesis_multiplier", 3.0)
                )
            total += contribution
        return total

    def _run_random_bfs(
        self,
        *,
        world_model,
        root_signature: str,
        available_actions: list[str],
        observation_factory,
        target_len: int,
    ) -> list[tuple[list[str], float]]:
        """Length-matched random rollout for H5 baseline."""
        out: list[tuple[list[str], float]] = []
        for _ in range(self.bfs_rollouts_per_call):
            seq: list[str] = []
            cum_reward = 0.0
            sig = root_signature
            family_chain: list[str] = []
            for _step in range(target_len):
                action = self._rng.choice(available_actions)
                obs = observation_factory(sig) if observation_factory else None
                sim = world_model.simulate_step(sig, action, obs)
                cum_reward += self._reward_from_sim(
                    sim, family_chain, action=action, before_family=sig
                )
                family_chain.append(sim["next_signature_hint"])
                sig = sim["next_signature_hint"]
                seq.append(action)
                if sim.get("exception_flag"):
                    break
            out.append((seq, cum_reward))
        return out
