"""Beta-Bernoulli posterior with batched eval + co-occurrence decay (plan §1.8).

Each (predicate × region) is one arm. Wilson LCB for retention (plan §1.11).
RASI prior support (plan §1.13): cycle237 trace can boost alpha; 7-arm ablation.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ArmKey:
    predicate_id: str
    region_id: str

    def __repr__(self) -> str:
        return f"ArmKey({self.predicate_id}, {self.region_id})"


@dataclass
class ArmStats:
    alpha: float = 1.0
    beta: float = 1.0
    n_emit: int = 0
    last_level_delta: int = 0
    last_emit_turn: int = -1


# Cold-start tiebreaker: specific families up-rank above the fallback.
_FAMILY_PRIORITY: dict[str, float] = {
    "sector_alignment": 4.0, "marker_align": 3.0, "dominant_transition": 2.0,
    "neighbor_pair": 1.5, "unclicked_neighbor": 1.5, "compass_change": 1.5,
    "size_anchor": 1.0, "color_invariance": 1.0, "corner_align": 1.0,
    "marker_no_recent": 1.0, "recent_response": 1.0,
    "extended": 0.5, "fallback": 0.0,
}


def _wilson_lower_cb(alpha: float, beta_: float) -> float:
    """One-sided Wilson 95% lower bound on p = (alpha-1) / ((alpha-1)+(beta-1))."""
    n = (alpha - 1.0) + (beta_ - 1.0)
    if n <= 0.0:
        return 0.0
    p_hat = (alpha - 1.0) / n
    z = 1.6449
    denom = 1.0 + (z * z) / n
    centre = p_hat + (z * z) / (2.0 * n)
    margin = z * math.sqrt(max(p_hat * (1.0 - p_hat) / n + (z * z) / (4.0 * n * n), 0.0))
    lcb = (centre - margin) / denom
    return max(0.0, min(1.0, lcb))


class PredicatePosterior:
    def __init__(self) -> None:
        self.arms: dict[ArmKey, ArmStats] = {}
        self.recent_emissions: list[list[ArmKey]] = []
        self.co_decay_window: int = 5
        self.turn_count: int = 0

    def _ensure_arm(self, key: ArmKey) -> ArmStats:
        arm = self.arms.get(key)
        if arm is None:
            arm = ArmStats()
            self.arms[key] = arm
        return arm

    def _total_selections(self) -> int:
        return sum(a.n_emit for a in self.arms.values())

    @staticmethod
    def _region_id(region: Any) -> str | None:
        if region is None:
            return None
        if isinstance(region, dict):
            return region.get("id") or region.get("region_id")
        return getattr(region, "region_id", None) or getattr(region, "id", None)

    def update(self, observation: dict[str, Any]) -> None:
        if not isinstance(observation, dict):
            return
        try:
            delta_clipped = 1 if int(observation.get("level_delta", 0)) > 0 else 0
        except (TypeError, ValueError):
            delta_clipped = 0
        if not self.recent_emissions:
            return
        window = self.recent_emissions[-self.co_decay_window:]
        for idx, batch in enumerate(window):
            credit = 1.0 if idx == len(window) - 1 else 0.5
            for key in batch:
                arm = self._ensure_arm(key)
                if delta_clipped == 1:
                    arm.alpha += credit
                else:
                    arm.beta += credit
                arm.last_level_delta = delta_clipped

    def record_emission(self, keys: list[ArmKey]) -> None:
        self.turn_count += 1
        self.recent_emissions.append(list(keys))
        for k in keys:
            arm = self._ensure_arm(k)
            arm.n_emit += 1
            arm.last_emit_turn = self.turn_count

    def _score(self, pid: str, pred: Any, region_id: str, log_N: float) -> float:
        arm = self.arms.get(ArmKey(pid, region_id))
        if arm is None or arm.n_emit == 0:
            family = getattr(pred, "family", "fallback") if pred is not None else "fallback"
            return 1e6 + _FAMILY_PRIORITY.get(family, 0.0)
        exploit = arm.alpha / (arm.alpha + arm.beta)
        explore = math.sqrt(2.0 * log_N / arm.n_emit)
        return exploit + explore

    def select(self, visible_regions: list[Any], library: Any) -> tuple[str, str] | None:
        if not visible_regions:
            return None
        try:
            preds = library.all_predicates() if library is not None else {}
        except Exception:
            preds = {}
        if not preds:
            return None
        N = max(1, self._total_selections())
        log_N = math.log(N + 1)
        best_key: ArmKey | None = None
        best_score = -math.inf
        for region in visible_regions:
            rid = self._region_id(region)
            if rid is None:
                continue
            for pid, pred in preds.items():
                s = self._score(pid, pred, rid, log_N)
                if s > best_score:
                    best_score = s
                    best_key = ArmKey(pid, rid)
        return (best_key.predicate_id, best_key.region_id) if best_key is not None else None

    def rank_top(self, visible_regions: list[Any], library: Any, k: int = 10) -> list[tuple[str, str, float]]:
        if not visible_regions:
            return []
        try:
            preds = library.all_predicates() if library is not None else {}
        except Exception:
            preds = {}
        if not preds:
            return []
        N = max(1, self._total_selections())
        log_N = math.log(N + 1)
        rows: list[tuple[str, str, float]] = []
        for region in visible_regions:
            rid = self._region_id(region)
            if rid is None:
                continue
            for pid, pred in preds.items():
                rows.append((pid, rid, self._score(pid, pred, rid, log_N)))
        rows.sort(key=lambda r: r[2], reverse=True)
        return rows[:k]

    def bootstrap(self, predicate_id: str, region_ids: list[str], anchor_alpha: float = 1.0) -> None:
        for rid in region_ids:
            self.arms[ArmKey(predicate_id, rid)] = ArmStats(alpha=1.0 + anchor_alpha, beta=1.0)

    def load_rasi_prior(self, trace_path: str, weight: float = 1.0, mode: str = "targeted") -> None:
        if mode == "none":
            return
        path = Path(trace_path)
        if not path.exists():
            return
        events: list[tuple[str, str]] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("level_delta", 0) <= 0:
                    continue
                pid = rec.get("predicate_id") or rec.get("template_id")
                rid = rec.get("region_id")
                if pid and rid:
                    events.append((pid, rid))
        if not events:
            return
        if mode == "shuffle":
            rng = random.Random(0)
            pids = [p for p, _ in events]
            rng.shuffle(pids)
            events = list(zip(pids, [r for _, r in events]))
        elif mode == "phase_shift":
            rids = [r for _, r in events]
            rids = rids[5:] + rids[:5]
            events = list(zip([p for p, _ in events], rids))
        elif mode == "sign_invert":
            for pid, rid in events:
                self._ensure_arm(ArmKey(pid, rid)).beta += weight
            return
        elif mode == "uniform_cal":
            keys = [ArmKey(pid, rid) for pid, rid in events]
            share = weight / max(1, len(keys))
            for k in keys:
                self._ensure_arm(k).alpha += share
            return
        elif mode == "variance_match":
            rng = random.Random(0)
            for pid, rid in events:
                self._ensure_arm(ArmKey(pid, rid)).alpha += weight * rng.uniform(0.5, 1.5)
            return
        for pid, rid in events:
            self._ensure_arm(ArmKey(pid, rid)).alpha += weight

    def retention_score(self, key: ArmKey) -> float:
        arm = self.arms.get(key)
        return _wilson_lower_cb(arm.alpha, arm.beta) if arm is not None else 0.0

    def retention_pass(self, key: ArmKey, n_min: int = 10, lcb_min: float = 0.5) -> bool:
        arm = self.arms.get(key)
        if arm is None or arm.n_emit < n_min:
            return False
        return self.retention_score(key) >= lcb_min
