"""Meta-harness module.

Inspiration: "Meta-Harness" (arxiv 2603.28052) and
"Natural-Language Agent Harnesses" (arxiv 2603.25723).

Philosophy:
- The harness itself is an optimizable artifact. We keep a pool of natural
  language overlays that can be appended to the agent premise. Each run lets
  us score how well the active overlay is doing and the module decides which
  overlays to keep, drop, or add on the next run.
- We never reference hidden progress markers. The score is built from
  measurable, agent-side signals only:
    * diversity of non-RESET actions taken
    * fraction of actions that produced a non-zero grid diff
    * number of surprises on the shared bridge
    * number of new skills proposed on the shared bridge
- The overlay texts are generic agent-hygiene hints. None of them names any
  specific action, skill, or progress marker. The agent is free to choose
  its own vocabulary on top of them.

This module is fully independent: it can run alone and it does not touch
the DreamCoder or World-Model state except by reading counts off the shared
bridge.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

from ..bridge import SharedBridge
from ..grid_utils import grid_diff_magnitude


# Small pool of *generic* overlay candidates. The module is free to introduce
# new ones at runtime — this is just the starting seed.
_SEED_OVERLAYS: dict[str, str] = {
    "stay_curious": (
        "After each action, briefly note whether the grid changed in a way you "
        "did or did not expect. The unexpected ones are the most informative."
    ),
    "avoid_tight_loops": (
        "If the same short action loop is producing the same result for the "
        "third time, break the loop — pick the least-used available action."
    ),
    "cover_actions": (
        "Before concluding that the environment is unresponsive, make sure you "
        "have tried each available action at least once from this state."
    ),
    "reset_as_last_resort": (
        "Do not reset until you can state, in your own words, why the current "
        "branch is no longer informative."
    ),
    "seek_unseen_observation": (
        "If the current line only confirms what you already know, choose the action most likely to reach an unseen observation family."
    ),
    "falsify_before_confirm": (
        "Keep a best hypothesis and a rival. Prefer an action that could falsify the stronger one before spending more actions confirming it."
    ),
    "escape_local_attractor": (
        "If you keep re-entering the same family, stop replaying the familiar corridor and take a branch-escape action."
    ),
}


@dataclass(slots=True)
class OverlayRecord:
    key: str
    text: str
    uses: int = 0
    last_score: float = 0.0
    cumulative_score: float = 0.0


@dataclass(slots=True)
class RunMetrics:
    """All agent-side metrics we score an overlay pool on. No hidden markers."""
    total_actions: int = 0
    non_reset_actions: int = 0
    resets: int = 0
    unique_actions: int = 0
    nonzero_diff_actions: int = 0
    surprises_added: int = 0
    skills_proposed: int = 0
    new_signature_count: int = 0
    new_family_count: int = 0
    falsification_probe_count: int = 0
    repeated_family_penalty: int = 0
    branch_escape_count: int = 0
    unresolved_rival_count: int = 0
    active_overlay_keys: list[str] = field(default_factory=list)

    def score(self) -> float:
        coverage = self.unique_actions
        diff_rate = (
            self.nonzero_diff_actions / self.non_reset_actions
            if self.non_reset_actions > 0
            else 0.0
        )
        return (
            0.5 * coverage
            + 1.0 * diff_rate
            + 0.5 * self.surprises_added
            + 0.75 * self.skills_proposed
            + 2.0 * self.new_signature_count
            + 3.0 * self.new_family_count
            + 3.0 * self.falsification_probe_count
            + 3.0 * self.branch_escape_count
            - 1.5 * self.repeated_family_penalty
            - 1.0 * self.unresolved_rival_count
            - 0.25 * self.resets
        )


class MetaHarnessModule:
    name = "meta_harness"

    def __init__(self, params: dict[str, Any], context, bridge: SharedBridge) -> None:
        self.params = params
        self.context = context
        self.bridge = bridge
        self.max_overlays = int(params.get("max_overlays", 3))
        self.keep_top_k = int(params.get("keep_top_k", 4))
        self.pool: dict[str, OverlayRecord] = {}
        self.active_keys: list[str] = []
        self.run_history: list[dict[str, Any]] = []
        self._surprise_cursor_at_start: int = 0
        self._proposal_cursor_at_start: int = 0
        self._run_actions: list[str] = []
        self._nonzero_diff_actions: int = 0
        self._seen_diff_actions: set[str] = set()
        self._new_signature_count: int = 0
        self._new_family_count: int = 0
        self._falsification_probe_count: int = 0
        self._repeated_family_penalty: int = 0
        self._branch_escape_count: int = 0
        self._state_path = context.shared_dir / "meta_harness.json"
        self._load_state()
        self._ensure_starting_pool()
        self._seed_active_overlays()
        self._snapshot_bridge_cursors()

    # -- Hook surface -----------------------------------------------------
    def prompt_overlay(self) -> str:
        if not self.active_keys:
            return ""
        lines = ["[Harness Guidance]"]
        for key in self.active_keys:
            record = self.pool.get(key)
            if record is None:
                continue
            lines.append(f"- {record.text}")
        return "\n".join(lines)

    def on_memories_ready(self, memories: Any) -> None:
        overlay = self.prompt_overlay()
        if overlay and hasattr(memories, "add"):
            memories.add("[MetaHarness] active harness overlays", overlay)

    def before_action(
        self, frame: Any | None, action_name: str, available_actions: list[str]
    ) -> str | None:
        pressure = int(self.bridge.read_hint("world_model", {}).get("local_attractor_pressure", 0)) if isinstance(self.bridge.read_hint("world_model", {}), dict) else 0
        if pressure >= 3 and "escape_local_attractor" in self.pool:
            self.active_keys = self._dedupe_preserve_order(
                ["escape_local_attractor", "falsify_before_confirm"] + list(self.active_keys)
            )[: self.max_overlays]
        return action_name

    def after_action(self, before: Any | None, action_name: str, after: Any) -> None:
        self._run_actions.append(action_name)
        if before is None or after is None:
            return
        diff_mag = grid_diff_magnitude(before, after)
        if diff_mag > 0:
            self._nonzero_diff_actions += 1
            self._seen_diff_actions.add(action_name)
        observation = self.bridge.last_observation
        if observation is not None:
            self._new_signature_count += int(observation.new_signature)
            self._new_family_count += int(observation.new_family)
            self._branch_escape_count += int(observation.branch_escape)
            self._falsification_probe_count += int(bool(observation.hypothesis_falsified))
            self._repeated_family_penalty += max(0, observation.repeated_family_depth - 1)

    def on_run_end(
        self,
        *,
        history: list[tuple[str, Any]],
        finish_status: Any,
        workdir: Path,
    ) -> None:
        metrics = self._score_run()

        # Propose: bump scores of active overlays by the run's score and
        # introduce fresh candidates if the run was too low on diversity.
        for key in self.active_keys:
            record = self.pool.get(key)
            if record is None:
                continue
            record.uses += 1
            record.last_score = metrics.score()
            record.cumulative_score += metrics.score()

        if metrics.unique_actions <= 1 and "cover_actions" not in self.pool:
            self.pool["cover_actions"] = OverlayRecord(
                key="cover_actions",
                text=_SEED_OVERLAYS["cover_actions"],
            )
        if metrics.resets >= 3 and "reset_as_last_resort" not in self.pool:
            self.pool["reset_as_last_resort"] = OverlayRecord(
                key="reset_as_last_resort",
                text=_SEED_OVERLAYS["reset_as_last_resort"],
            )
        if metrics.new_family_count == 0 and "seek_unseen_observation" not in self.pool:
            self.pool["seek_unseen_observation"] = OverlayRecord(
                key="seek_unseen_observation",
                text=_SEED_OVERLAYS["seek_unseen_observation"],
            )
        if metrics.falsification_probe_count == 0 and "falsify_before_confirm" not in self.pool:
            self.pool["falsify_before_confirm"] = OverlayRecord(
                key="falsify_before_confirm",
                text=_SEED_OVERLAYS["falsify_before_confirm"],
            )
        if metrics.repeated_family_penalty >= 3 and "escape_local_attractor" not in self.pool:
            self.pool["escape_local_attractor"] = OverlayRecord(
                key="escape_local_attractor",
                text=_SEED_OVERLAYS["escape_local_attractor"],
            )

        # Evaluate + keep: sort the pool by avg score and select top-k.
        ranked = sorted(
            self.pool.values(),
            key=lambda r: (r.cumulative_score / max(r.uses, 1), r.uses),
            reverse=True,
        )
        kept = ranked[: self.keep_top_k]
        self.pool = {r.key: r for r in kept}

        # Select active overlays for next run: up to max_overlays, preferring
        # records with highest recent score.
        sorted_for_active = sorted(
            self.pool.values(),
            key=lambda r: (r.last_score, r.uses),
            reverse=True,
        )
        self.active_keys = [r.key for r in sorted_for_active[: self.max_overlays]]

        self.run_history.append(
            {
                "metrics": asdict(metrics),
                "selected_next_run": list(self.active_keys),
            }
        )
        self.run_history = self.run_history[-20:]

        self._state_path.parent.mkdir(parents=True, exist_ok=True)
        self._state_path.write_text(
            json.dumps(
                {
                    "pool": [asdict(r) for r in self.pool.values()],
                    "active_keys": self.active_keys,
                    "run_history": self.run_history,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    # -- Internals --------------------------------------------------------
    def _load_state(self) -> None:
        if not self._state_path.exists():
            return
        try:
            data = json.loads(self._state_path.read_text(encoding="utf-8"))
        except Exception:
            return
        for item in data.get("pool", []):
            if not isinstance(item, dict) or "key" not in item:
                continue
            self.pool[item["key"]] = OverlayRecord(
                key=item["key"],
                text=item.get("text", ""),
                uses=int(item.get("uses", 0)),
                last_score=float(item.get("last_score", 0.0)),
                cumulative_score=float(item.get("cumulative_score", 0.0)),
            )
        self.active_keys = list(data.get("active_keys", []))
        self.run_history = list(data.get("run_history", []))

    def _ensure_starting_pool(self) -> None:
        for key, text in _SEED_OVERLAYS.items():
            self.pool.setdefault(key, OverlayRecord(key=key, text=text))

    def _seed_active_overlays(self) -> None:
        if self.active_keys:
            return
        # First run: pick generic curiosity + anti-loop overlays.
        defaults = ["stay_curious", "avoid_tight_loops"]
        self.active_keys = [key for key in defaults if key in self.pool][
            : self.max_overlays
        ]

    def _snapshot_bridge_cursors(self) -> None:
        self._surprise_cursor_at_start = len(self.bridge.surprises)
        self._proposal_cursor_at_start = len(self.bridge.proposed_skills)

    def _score_run(self) -> RunMetrics:
        non_reset = [a for a in self._run_actions if a != "RESET"]
        resets = len(self._run_actions) - len(non_reset)
        unique = len(set(non_reset))
        surprises_added = max(
            0, len(self.bridge.surprises) - self._surprise_cursor_at_start
        )
        proposals_added = max(
            0, len(self.bridge.proposed_skills) - self._proposal_cursor_at_start
        )
        return RunMetrics(
            total_actions=len(self._run_actions),
            non_reset_actions=len(non_reset),
            resets=resets,
            unique_actions=unique,
            nonzero_diff_actions=self._nonzero_diff_actions,
            surprises_added=surprises_added,
            skills_proposed=proposals_added,
            new_signature_count=self._new_signature_count,
            new_family_count=self._new_family_count,
            falsification_probe_count=self._falsification_probe_count,
            repeated_family_penalty=self._repeated_family_penalty,
            branch_escape_count=self._branch_escape_count,
            unresolved_rival_count=0,
            active_overlay_keys=list(self.active_keys),
        )

    @staticmethod
    def _dedupe_preserve_order(items: list[str]) -> list[str]:
        seen: set[str] = set()
        out: list[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            out.append(item)
        return out
