"""Stalemate trigger — deterministic.

Plan §1.12: H-3 strengthened to outcome-correlated threshold.
Plan §1.19: stratified permutation validation in Phase D.

Hot-path (used in agent.py) is a simple deterministic check:
  fires := (turns_since_L_plus > K_threshold) AND (max_posterior < theta_threshold)
       AND NOT (once_per_episode AND fired_this_episode)

Validation (Phase D) wraps this with stratified permutation test.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StalemateConfig:
    K_threshold: int = 8       # turns since last L+ (plan §6.6 default; tune in Phase B)
    theta_threshold: float = 0.45  # max posterior threshold
    once_per_episode: bool = True  # avoid spamming LLM extender


class StalemateTrigger:
    def __init__(self, cfg: StalemateConfig | None = None) -> None:
        self.cfg = cfg or StalemateConfig()
        self.fired_this_episode: bool = False
        # v601 G8: warm-up tracking — turn 0 Proposer is counted separately.
        self.warm_up_done: bool = False

    def fires(self, turns_since_L_plus: int, max_posterior: float) -> bool:
        """Deterministic check (plan §1.12 hot-path)."""
        if self.cfg.once_per_episode and self.fired_this_episode:
            return False
        return (
            turns_since_L_plus > self.cfg.K_threshold
            and max_posterior < self.cfg.theta_threshold
        )

    def warm_up_fires(self, turn_count: int) -> bool:
        """v601 §4.3 G8: warm-up Proposer call at turn 0 (counted separately)."""
        if self.warm_up_done:
            return False
        return turn_count == 0

    def mark_warm_up_done(self) -> None:
        self.warm_up_done = True

    def mark_fired(self) -> None:
        self.fired_this_episode = True

    def reset_episode(self) -> None:
        self.fired_this_episode = False
        self.warm_up_done = False
