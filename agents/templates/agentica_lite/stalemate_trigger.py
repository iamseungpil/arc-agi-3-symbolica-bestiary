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


def _env_int(name: str, default: int) -> int:
    import os
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass
class StalemateConfig:
    # v605 codex round 3 C1 fix: env-overridable for arm-tuning experiments.
    K_threshold: int = _env_int("ARC_LITE_K_THRESHOLD", 3)
    theta_threshold: float = 0.45
    once_per_episode: bool = False
    periodic_every_n: int = _env_int("ARC_LITE_PERIODIC_EVERY_N", 5)
    max_calls_per_episode: int = _env_int("ARC_LITE_MAX_CALLS", 50)


class StalemateTrigger:
    def __init__(self, cfg: StalemateConfig | None = None) -> None:
        self.cfg = cfg or StalemateConfig()
        self.fired_this_episode: bool = False
        # v601 G8: warm-up tracking — turn 0 Proposer is counted separately.
        self.warm_up_done: bool = False
        # v605: per-episode fire counter for cap + last-fire-turn for periodic.
        self.fires_this_episode: int = 0
        self.last_fire_turn: int = -1

    def fires(self, turns_since_L_plus: int, max_posterior: float, turn_count: int = 0) -> bool:
        """v605: stalemate OR periodic, capped by max_calls_per_episode."""
        if self.fires_this_episode >= self.cfg.max_calls_per_episode:
            return False
        # original stalemate condition
        stalemate = (
            turns_since_L_plus > self.cfg.K_threshold
            and max_posterior < self.cfg.theta_threshold
        )
        if self.cfg.once_per_episode and self.fired_this_episode and not stalemate:
            return False
        # v605: periodic fire — every periodic_every_n turns since last fire
        periodic = (
            self.cfg.periodic_every_n > 0
            and turn_count - self.last_fire_turn >= self.cfg.periodic_every_n
        )
        return stalemate or periodic

    def warm_up_fires(self, turn_count: int) -> bool:
        """v601 §4.3 G8: warm-up Proposer call at turn 0 (counted separately)."""
        if self.warm_up_done:
            return False
        return turn_count == 0

    def mark_warm_up_done(self) -> None:
        self.warm_up_done = True

    def mark_fired(self, turn_count: int = 0) -> None:
        self.fired_this_episode = True
        self.fires_this_episode += 1
        self.last_fire_turn = turn_count

    def reset_episode(self) -> None:
        self.fired_this_episode = False
        self.warm_up_done = False
        self.fires_this_episode = 0
        self.last_fire_turn = -1
