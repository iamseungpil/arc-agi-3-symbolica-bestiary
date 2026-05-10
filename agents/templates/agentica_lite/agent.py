"""ArcgenticaLite — plan v600 minimal-fast dispatcher.

Hot path per turn (plan §6.2):
  1. Update posterior from last observation.
  2. If stalemate fires AND not yet fired this episode: one LLM call (≤45 s),
     sandbox-install, bootstrap posterior arms.
  3. UCB1 selection over visible (predicate × region) pairs.
  4. Resolve coord per the predicate's coord_policy.
60 s wall-clock per turn (plan §1.14).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from .llm_extender import ExtenderInput, LLMExtender
from .memory_journal import EpisodeRecord, MemoryJournal
from .predicate_library import PredicateLibrary
from .predicate_posterior import ArmKey, PredicatePosterior
from .stalemate_trigger import StalemateConfig, StalemateTrigger

logger = logging.getLogger(__name__)


@dataclass
class Action:
    predicate_id: str
    region_id: str
    coord_xy: tuple[int, int]


class ArcgenticaLite:
    FRAMEWORK_VERSION = "v600"

    def __init__(
        self,
        game_id: str,
        seed: int,
        rasi_arm: str = "null",
        stalemate_cfg: StalemateConfig | None = None,
        global_turn_budget_s: float = 60.0,
    ) -> None:
        self.game_id = game_id
        self.seed = seed
        self.rasi_arm = rasi_arm
        self.global_turn_budget_s = global_turn_budget_s
        self.posterior = PredicatePosterior()
        self.library = PredicateLibrary()
        self.stalemate = StalemateTrigger(stalemate_cfg)
        self.extender = LLMExtender()
        self.journal = MemoryJournal(game_id)
        self.turns_since_L_plus = 0
        self.last_max_level = 0
        self._turn_latencies: list[float] = []
        self._turn_count = 0
        self._t_episode_start = time.time()
        self._stalemate_events = 0
        self._predicates_installed: list[dict] = []
        self._recent_failures: list[dict] = []
        self._last_visible_regions: list[Any] = []

    @staticmethod
    def _visible_regions(state: Any) -> list[Any]:
        if state is None:
            return []
        if isinstance(state, dict):
            return state.get("visible_regions") or []
        return getattr(state, "visible_regions", []) or []

    @staticmethod
    def _last_observation(state: Any) -> dict[str, Any]:
        if state is None:
            return {}
        if isinstance(state, dict):
            return state.get("last_observation") or {}
        return getattr(state, "last_observation", {}) or {}

    def _max_posterior(self, visible_regions: list[Any]) -> float:
        if not visible_regions or not self.posterior.arms:
            return 0.0
        rids = {self.posterior._region_id(r) for r in visible_regions}
        rids.discard(None)
        if not rids:
            return 0.0
        best = 0.0
        for key, arm in self.posterior.arms.items():
            if key.region_id in rids:
                p = arm.alpha / (arm.alpha + arm.beta)
                if p > best:
                    best = p
        return best

    async def run_turn(self, state: Any) -> Action | None:
        t0 = time.monotonic()
        self._turn_count += 1
        deadline = t0 + self.global_turn_budget_s
        visible_regions = self._visible_regions(state)
        self._last_visible_regions = visible_regions

        try:
            obs = self._last_observation(state)
            self.posterior.update(obs)
            if int(obs.get("level_delta", 0) or 0) > 0:
                self.turns_since_L_plus = 0
                self.last_max_level += 1
            else:
                self.turns_since_L_plus += 1
        except Exception as e:  # noqa: BLE001
            logger.warning("posterior.update failed: %s", e)

        try:
            max_post = self._max_posterior(visible_regions)
            if self.stalemate.fires(self.turns_since_L_plus, max_post):
                self._stalemate_events += 1
                if (deadline - time.monotonic()) >= 50.0:
                    await self._fire_extender(visible_regions, deadline)
                    self.stalemate.mark_fired()
        except Exception as e:  # noqa: BLE001
            logger.warning("stalemate/extender path failed: %s", e)

        try:
            pick = self.posterior.select(visible_regions, self.library)
        except Exception as e:  # noqa: BLE001
            logger.warning("posterior.select failed: %s", e)
            pick = None
        if pick is None:
            self._turn_latencies.append(time.monotonic() - t0)
            return None
        pred_id, region_id = pick
        region_obj = next(
            (r for r in visible_regions if self.posterior._region_id(r) == region_id),
            None,
        )
        try:
            xy = self.library.resolve_coord(pred_id, region_obj or {}, state)
        except Exception as e:  # noqa: BLE001
            logger.warning("resolve_coord failed: %s", e)
            xy = (0, 0)

        self.posterior.record_emission([ArmKey(pred_id, region_id)])
        latency = time.monotonic() - t0
        if latency > self.global_turn_budget_s:
            logger.warning("wall_clock_breach: %.2fs", latency)
        self._turn_latencies.append(latency)
        return Action(pred_id, region_id, xy)

    async def _fire_extender(self, visible_regions: list[Any], deadline: float) -> None:
        time_left = max(1.0, deadline - time.monotonic() - 5.0)
        self.extender.llm_timeout_s = min(self.extender.llm_timeout_s, time_left)
        top10 = self.posterior.rank_top(visible_regions, self.library, k=10)
        try:
            out = await self.extender.propose(ExtenderInput(
                visible_regions=visible_regions,
                posterior_top10=top10,
                recent_failures=self._recent_failures,
            ))
        except Exception as e:  # noqa: BLE001
            logger.warning("extender.propose raised: %s", e)
            return
        if not out.accepted or not out.lambda_body:
            return
        try:
            res = self.library.install(
                out.lambda_body,
                predicate_id=out.name,
                family=out.family or "extended",
                coord_policy=out.coord_policy or "centroid",
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("library.install raised: %s", e)
            return
        if not res.accepted:
            logger.info("install rejected: %s", res.reason)
            return
        installed_id = out.name or f"L_ext_{res.fingerprint}"
        rids = [r for r in (self.posterior._region_id(r) for r in visible_regions) if r is not None]
        self.posterior.bootstrap(installed_id, rids, anchor_alpha=1.0)
        self._predicates_installed.append(
            {"id": installed_id, "fingerprint": res.fingerprint, "reason": res.reason}
        )

    def episode_end(self) -> EpisodeRecord:
        lat = sorted(self._turn_latencies)
        n = len(lat)
        p50 = lat[n // 2] if n else 0.0
        p95 = lat[max(0, int(0.95 * n) - 1)] if n else 0.0
        worst = lat[-1] if n else 0.0
        top10 = (self.posterior.rank_top(self._last_visible_regions, self.library, k=10)
                 if self._last_visible_regions else [])
        rec = EpisodeRecord(
            episode_id=f"{self.game_id}_seed{self.seed}_t{int(self._t_episode_start)}",
            seed=self.seed,
            framework_version=self.FRAMEWORK_VERSION,
            git_sha="unknown",
            ts_start=self._t_episode_start,
            ts_end=time.time(),
            turns=self._turn_count,
            max_level=self.last_max_level,
            rasi_arm=self.rasi_arm,
            predicates_installed=list(self._predicates_installed),
            stalemate_events=self._stalemate_events,
            posterior_top10_at_end=top10,
            latency_p50=p50,
            latency_p95=p95,
            latency_worst=worst,
        )
        self.journal.append(rec)
        return rec
