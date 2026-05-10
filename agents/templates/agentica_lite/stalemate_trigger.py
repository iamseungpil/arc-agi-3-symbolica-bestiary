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

    def fires(self, turns_since_L_plus: int, max_posterior: float) -> bool:
        """Deterministic check (plan §1.12 hot-path)."""
        if self.cfg.once_per_episode and self.fired_this_episode:
            return False
        return (
            turns_since_L_plus > self.cfg.K_threshold
            and max_posterior < self.cfg.theta_threshold
        )

    def mark_fired(self) -> None:
        self.fired_this_episode = True

    def reset_episode(self) -> None:
        self.fired_this_episode = False
