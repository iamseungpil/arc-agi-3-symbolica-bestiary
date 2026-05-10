"""v601 Role 3: Memory writer (deterministic). Plan rev C §3.7-3.11.

Detects same-coord/different-outcome pairings, extracts pre-state diff via
plan §3.10 whitelist, computes severity, and decides whether to spawn the
Reflector subagent under §3.11 cooldown + rev C C2 support_count >= 2.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any

logger = logging.getLogger(__name__)

# Plan §3.10 ranked feature whitelist (highest priority first).
_FEATURE_PRIORITY = [
    "marker_*_compass_saturation_numerator",
    "marker_*_compass_recent_click_direction",
    "recent_dominant_transition_direction",
    "region_*_click_count",
    "level_delta_since_last_paired_cf",
]


@dataclass
class Outcome:
    coord: tuple[int, int]
    primary_region_id: str | None
    level_delta: int
    dt_count: int
    pre_state_features: dict[str, Any] = field(default_factory=dict)
    turn: int = -1


@dataclass
class PairedCFEntry:
    coord: tuple[int, int]
    primary_region_id: str | None
    outcomes: list[Outcome] = field(default_factory=list)
    discriminating_features: list[str] = field(default_factory=list)
    best_pair_severity: float = 0.0
    support_count: int = 0


def _coord_tuple(c: Any) -> tuple[int, int]:
    if isinstance(c, (list, tuple)) and len(c) >= 2:
        return (int(c[0]), int(c[1]))
    return (-1, -1)


def severity(outcome_a: Outcome, outcome_b: Outcome) -> float:
    """Plan §3.11."""
    s_count = abs(outcome_a.dt_count - outcome_b.dt_count) / max(
        outcome_a.dt_count, outcome_b.dt_count, 1
    )
    s_level = 0.5 if outcome_a.level_delta != outcome_b.level_delta else 0.0
    return max(s_count, s_level)


def _matches_priority(feature_name: str) -> int:
    """Return priority index (lower=higher priority) or len(_FEATURE_PRIORITY) if no match."""
    for i, pat in enumerate(_FEATURE_PRIORITY):
        if "*" in pat:
            prefix, suffix = pat.split("*")
            if feature_name.startswith(prefix) and feature_name.endswith(suffix):
                return i
        elif feature_name == pat:
            return i
    return len(_FEATURE_PRIORITY)


def _diff_features(outcome_a: Outcome, outcome_b: Outcome) -> list[str]:
    """Return discriminating features (top priority first), restricted to plan §3.10 whitelist."""
    diffs: list[tuple[int, str]] = []
    keys = set(outcome_a.pre_state_features) | set(outcome_b.pre_state_features)
    for k in keys:
        v_a = outcome_a.pre_state_features.get(k)
        v_b = outcome_b.pre_state_features.get(k)
        if v_a is None and v_b is None:
            continue
        # numeric magnitude check
        if isinstance(v_a, (int, float)) and isinstance(v_b, (int, float)):
            if abs(v_a - v_b) < 1:
                continue
        elif v_a == v_b:
            continue
        prio = _matches_priority(k)
        if prio >= len(_FEATURE_PRIORITY):
            continue  # skip non-whitelisted features
        diffs.append((prio, k))
    diffs.sort(key=lambda x: x[0])
    return [k for _, k in diffs]


def discriminator(outcome_a: Outcome, outcome_b: Outcome) -> tuple[str | None, list[str]]:
    """Return (top_feature, full_list)."""
    feats = _diff_features(outcome_a, outcome_b)
    return (feats[0] if feats else None), feats


@dataclass
class PairedCFAnalysis:
    triggered: bool
    entry: PairedCFEntry | None
    spawn_reflector: bool
    spawn_reason: str  # "ok", "support_count_below_2", "cooldown_active", "severity_below_threshold"


def analyze(
    outcomes: list[Outcome],
    current_turn: int,
    last_reflector_turn: int,
    cooldown_turns: int = 5,
    severity_threshold: float = 0.5,
    require_support_for_n_ge_3: bool = True,
    severity_threshold_for_support: float = 0.5,
) -> PairedCFAnalysis:
    """Plan §3.7 G23 + §3.11 + rev C C2: lifecycle for n>=2 outcomes."""
    if len(outcomes) < 2:
        return PairedCFAnalysis(False, None, False, "insufficient_outcomes")

    coord = outcomes[0].coord
    primary_region_id = outcomes[0].primary_region_id

    pairs = list(combinations(outcomes, 2))
    severities = [severity(a, b) for a, b in pairs]
    best_pair_severity = max(severities)

    # Check trigger condition (G4)
    deltas = [o.level_delta for o in outcomes]
    counts = [o.dt_count for o in outcomes]
    triggered = (
        (max(deltas) - min(deltas) >= 1)
        or (min(counts) > 0 and max(counts) > 3 * min(counts))
    )
    if not triggered:
        return PairedCFAnalysis(False, None, False, "trigger_predicate_false")

    # Plan §3.7 G23: top_disc = argmax_feature(variance across outcomes); support_count
    # = number of pairs whose own pair-level top discriminator equals the global top.
    # Variance is restricted to whitelist features. For numeric features we use
    # population variance; for categoricals, the count of distinct non-null values - 1.
    whitelist_features: set[str] = set()
    for o in outcomes:
        for k in o.pre_state_features:
            if _matches_priority(k) < len(_FEATURE_PRIORITY):
                whitelist_features.add(k)

    def _variance(feature: str) -> float:
        vals = [o.pre_state_features.get(feature) for o in outcomes]
        nums = [v for v in vals if isinstance(v, (int, float))]
        if len(nums) == len(vals) and len(nums) >= 2:
            mean = sum(nums) / len(nums)
            return sum((x - mean) ** 2 for x in nums) / len(nums)
        # categorical: number of distinct non-null values minus 1
        distinct = {v for v in vals if v is not None}
        return float(max(0, len(distinct) - 1))

    # Tie-break: priority (lower index = higher priority), then alphabetical name.
    ranked_features = sorted(
        whitelist_features,
        key=lambda f: (-_variance(f), _matches_priority(f), f),
    )
    top_disc: str | None = None
    if ranked_features and _variance(ranked_features[0]) > 0:
        top_disc = ranked_features[0]

    # Pair-level discriminator (top feature for that pair) — must match the global top_disc.
    # Per plan §3.7 example "1 pairwise severity >0.5, others <0.5 → support_count=1",
    # support_count counts only pairs whose severity exceeds threshold AND whose pair-level
    # top discriminator equals the global top. This makes a noisy outlier (1 high-sev pair)
    # produce support_count=1 even when nominally 2 pairs share the same dominant feature.
    pair_top_features: list[str | None] = []
    pair_severities: list[float] = []
    for a, b in pairs:
        feats = _diff_features(a, b)
        pair_top_features.append(feats[0] if feats else None)
        pair_severities.append(severity(a, b))
    support_count = sum(
        1 for f, sev in zip(pair_top_features, pair_severities)
        if top_disc is not None and f == top_disc and sev > severity_threshold_for_support
    ) if top_disc else 0

    # Aggregate display: include all whitelist features whose top is the dominant.
    discriminating_features = [top_disc] if top_disc else []

    entry = PairedCFEntry(
        coord=_coord_tuple(coord),
        primary_region_id=primary_region_id,
        outcomes=list(outcomes),
        discriminating_features=discriminating_features,
        best_pair_severity=round(best_pair_severity, 4),
        support_count=support_count,
    )

    # Spawn decision (§3.11 cooldown + rev C C2 support_count)
    if best_pair_severity <= severity_threshold:
        return PairedCFAnalysis(True, entry, False, "severity_below_threshold")

    if last_reflector_turn >= 0 and (current_turn - last_reflector_turn) < cooldown_turns:
        return PairedCFAnalysis(True, entry, False, "cooldown_active")

    if require_support_for_n_ge_3 and len(outcomes) >= 3 and support_count < 2:
        return PairedCFAnalysis(True, entry, False, "support_count_below_2")

    return PairedCFAnalysis(True, entry, True, "ok")


class PairedCFStore:
    """Per-episode store keyed by (coord, primary_region_id). Plan §3.7 G23."""

    def __init__(self) -> None:
        self._store: dict[tuple[tuple[int, int], str | None], list[Outcome]] = {}

    def append(self, outcome: Outcome) -> list[Outcome]:
        """Append outcome to its key, return the cumulative outcome list."""
        key = (_coord_tuple(outcome.coord), outcome.primary_region_id)
        bucket = self._store.setdefault(key, [])
        bucket.append(outcome)
        return list(bucket)

    def all(self) -> dict[tuple[tuple[int, int], str | None], list[Outcome]]:
        return dict(self._store)


@dataclass
class MemoryWriterDecision:
    new_outcomes_for_arm: bool  # whether posterior should be updated
    paired_cf_entry: PairedCFEntry | None
    spawn_reflector: bool
    spawn_reason: str


class MemoryWriter:
    def __init__(
        self,
        cooldown_turns: int = 5,
        severity_threshold: float = 0.5,
    ) -> None:
        self.store = PairedCFStore()
        self.last_reflector_turn = -1
        self.cooldown_turns = cooldown_turns
        self.severity_threshold = severity_threshold

    def record(self, outcome: Outcome, current_turn: int) -> MemoryWriterDecision:
        outcomes = self.store.append(outcome)
        analysis = analyze(
            outcomes,
            current_turn=current_turn,
            last_reflector_turn=self.last_reflector_turn,
            cooldown_turns=self.cooldown_turns,
            severity_threshold=self.severity_threshold,
        )
        if analysis.spawn_reflector:
            self.last_reflector_turn = current_turn
        return MemoryWriterDecision(
            new_outcomes_for_arm=True,
            paired_cf_entry=analysis.entry,
            spawn_reflector=analysis.spawn_reflector,
            spawn_reason=analysis.spawn_reason,
        )
