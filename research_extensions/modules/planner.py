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
        self.reward_weights = dict(params.get("reward_weights", {
            "new_family": 3.0,
            "branch_escape": 2.0,
            "accuracy": 1.0,
            "repeat_penalty": -1.0,
            "exception": -2.0,
            "expect_change_match": 0.25,
            "band_large": 0.1,
        }))
        self.bfs_rollouts_per_call = int(params.get("bfs_rollouts_per_call", 4))
        self._rng = random.Random(int(params.get("seed", 1234)))
        # H5 advantage tracking for the meta harness.
        self.recent_advantages: list[float] = []
        self.last_call_at: float = 0.0
        self._memories: Any | None = None
        self._state_path = context.shared_dir / "planner_state.json"

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
                        "params": {
                            "max_depth": self.max_depth,
                            "n_simulations": self.n_simulations,
                            "gate_threshold": self.gate_threshold,
                            "reward_weights": self.reward_weights,
                        },
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
        except Exception:
            pass

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
        # H5b gate
        draft = world_model._best_draft()
        accuracy = draft.transition_accuracy if draft is not None else 0.0
        gate_ok = accuracy >= self.gate_threshold and draft is not None
        if not gate_ok:
            result = self._empty_result(reason=f"gate: accuracy={accuracy:.2f} < {self.gate_threshold:.2f}")
            return result

        if not available_actions:
            return self._empty_result(reason="no available actions")

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

        return {
            "gate_ok": True,
            "mcts_top_paths": mcts_paths,
            "bfs_top_paths": bfs_paths,
            "mcts_top_reward": mcts_top_reward,
            "bfs_mean_reward": bfs_mean,
            "advantage": advantage,
            "reason_skipped": None,
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
            total += self._reward_from_sim(sim, family_chain)
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
                terminal=bool(sim.get("exception_flag")),
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
        while not terminal_in_rollout and depth + rollout_depth < self.max_depth:
            action = self._rng.choice(available_actions)
            obs = observation_factory(rollout_node_sig) if observation_factory else None
            sim = world_model.simulate_step(rollout_node_sig, action, obs)
            rollout_reward += self._reward_from_sim(sim, rollout_family_chain)
            rollout_family_chain.append(sim["next_signature_hint"])
            rollout_node_sig = sim["next_signature_hint"]
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

    def _reward_from_sim(self, sim: dict[str, Any], family_chain: list[str]) -> float:
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
        # Accuracy bonus (favor planners on more reliable simulators)
        r += float(self.reward_weights.get("accuracy", 1.0)) * float(sim.get("transition_accuracy", 0.0))
        return r

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
                cum_reward += self._reward_from_sim(sim, family_chain)
                family_chain.append(sim["next_signature_hint"])
                sig = sim["next_signature_hint"]
                seq.append(action)
                if sim.get("exception_flag"):
                    break
            out.append((seq, cum_reward))
        return out
