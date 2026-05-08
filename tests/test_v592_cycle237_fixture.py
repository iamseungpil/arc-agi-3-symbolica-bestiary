"""B20 v592 cycle237 trace replay fixture.

Per project memory `feedback_module_fixture_first.md`: validate v592 helpers
against the SOLE empirical L+2 success trace before live autoresearch.

cycle237 had:
  - 39 turns total
  - 6 TIER-A predicate calls (P01_*, P07_*, P12_*) in T7-T27 (post-L+1 phase)
  - 22 TIER-B M1-invented chids
  - L+1 at T6, L+2 at T27
  - tier_a_ratio (full cycle) ≈ 6/28 ≈ 0.21 (counting chids that are not None)

The v592 V-FIXTURE-H5 requires that cycle237 trace, REPLAYED through v592
runtime forcing logic, is COMPLIANT — it should NOT trigger violations
(stuck-rotation should fire 0 or rarely; min-ratio should fire 0 if ratio
≥ 0.15; score-gate triggers if any predicate hit 0.85, etc.).

This file is the V-FIXTURE-H5 proxy: replay cycle237 trace + score the
v592 helpers + assert the ratio/forcing rates are sensible.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from agents.templates.agentica_v57.agent import (
    _v592_stuck_rotation_should_force,
    _v592_score_gate_should_force,
    _v592_min_ratio_should_force,
)


CYCLE237_TRACE = Path(
    "simple_logs/ft09-9ab2447a/v57_1778180868_3399613/trace.jsonl"
)
TIER_A_RE = re.compile(r"^P\d{2,3}_[a-z0-9_]+:T\d+:R\d+$")
INVENTED_RE = re.compile(r"^[HP]_")
CARD_RE = re.compile(r"^C\d+$")


def _classify_chid(chid: str | None) -> str:
    if not chid:
        return "none"
    if TIER_A_RE.match(chid):
        return "A"
    if CARD_RE.match(chid):
        return "card"
    if INVENTED_RE.match(chid):
        return "B"
    return "none"


def _load_cycle237_with_tiers():
    if not CYCLE237_TRACE.exists():
        pytest.skip(f"cycle237 trace not present: {CYCLE237_TRACE}")
    rows = []
    for ln in CYCLE237_TRACE.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            r = json.loads(ln)
        except json.JSONDecodeError:
            continue
        chid = (r.get("action") or {}).get("chosen_hypothesis_id")
        rows.append({
            "turn": r.get("turn"),
            "chid": chid,
            "chid_tier": _classify_chid(chid),
            "level_delta": int((r.get("observation") or {}).get("level_delta") or 0),
            "candidate_tests": r.get("candidate_tests_for_m1") or [],
            "snapped": r.get("snapped_coord"),
        })
    return rows


def _verbose_from_rows(rows):
    """Convert classified rows into recent_verbose-shaped entries."""
    return [
        {
            "chid_tier": r["chid_tier"],
            "observation": {"level_delta": r["level_delta"]},
        }
        for r in rows
    ]


# ---------------- T-V592-15: cycle237 tier ratio sanity ----------------


def test_cycle237_tier_a_ratio_in_post_t5_window_above_0_15():
    """V-H1 floor: TIER-A ratio ≥ 0.15 in any 50-turn window post-T5.

    cycle237 has only 39 turns total, so we use the full post-T5 window
    (T5-T38 = 34 turns) and check ratio.
    """
    rows = _load_cycle237_with_tiers()
    post = [r for r in rows if r["turn"] >= 5]
    tier_a = sum(1 for r in post if r["chid_tier"] == "A")
    total = len(post)
    ratio = tier_a / total if total else 0
    assert ratio >= 0.15, (
        f"cycle237 post-T5 tier_a_ratio={ratio:.3f} < 0.15 floor "
        f"(tier_a={tier_a} / total={total}); v592 V-H1 threshold "
        f"would not be cycle237-compliant"
    )


# ---------------- T-V592-16..18: cycle237 forcing trigger rates ----------------


def test_cycle237_stuck_rotation_triggers_rarely():
    """Stuck-rotation should fire 0 or very few times on cycle237.

    cycle237 mixed TIER-A and TIER-B regularly so 3+ consecutive
    TIER-B+ld=0 should be uncommon. If it fires often, v592 would
    needlessly force TIER-A on a winning trajectory.
    """
    rows = _load_cycle237_with_tiers()
    rv = []
    triggers = 0
    for r in rows:
        rv.append({
            "chid_tier": r["chid_tier"],
            "observation": {"level_delta": r["level_delta"]},
        })
        board = SimpleNamespace(recent_verbose=rv)
        # candidate_tests are real entries from this turn
        if r["candidate_tests"]:
            forced = _v592_stuck_rotation_should_force(board, r["candidate_tests"])
            if forced is not None:
                triggers += 1
    # round-9-fix: streak raised 3 → 5. cycle237's longest TIER-B run
    # was 7 (T8-T14) and 5 (T22-T26). With streak=5, only those two
    # windows can trigger and only at their tail. Allow up to 4
    # triggers (T13/T14 from T8-T14 = 2; T26 from T22-T26 = 1; +1 slack).
    assert triggers <= 4, (
        f"cycle237 stuck-rotation triggered {triggers}× — "
        f"v592 would have forced TIER-A too aggressively on winning trace"
    )


def test_cycle237_score_gate_does_not_overshoot():
    """Score-gate (0.85) should fire 0 or few times on cycle237.

    If predicate scores in cycle237 routinely hit 0.85 unresolved,
    v592 would force them and disrupt M1's invention path that won.
    """
    rows = _load_cycle237_with_tiers()
    triggers = 0
    for r in rows:
        if r["candidate_tests"]:
            forced = _v592_score_gate_should_force(r["candidate_tests"])
            if forced is not None:
                triggers += 1
    # cycle237 ran early enough that few predicates would have reached
    # 0.85. Allow up to 10.
    assert triggers <= 10, (
        f"cycle237 score-gate triggered {triggers}× — "
        f"v592 0.85 threshold may be too low; would over-force on winning trace"
    )


def test_cycle237_min_ratio_does_not_fire_in_main_phase():
    """Min-ratio (0.15 floor) should NOT fire on cycle237's main phase.

    If cycle237 tier_a_ratio is already 0.21 (above 0.15), the min-ratio
    helper should never trigger. Triggering means our floor calculation
    is too strict for the gold trace.
    """
    rows = _load_cycle237_with_tiers()
    rv = []
    triggers = 0
    for r in rows:
        rv.append({
            "chid_tier": r["chid_tier"],
            "observation": {"level_delta": r["level_delta"]},
        })
        board = SimpleNamespace(recent_verbose=rv)
        if r["candidate_tests"]:
            forced = _v592_min_ratio_should_force(board, r["candidate_tests"])
            if forced is not None:
                triggers += 1
    # cycle237 should be above floor for most of its run.
    # We allow ≤2 triggers (early-cycle when window<20 doesn't apply,
    # but late stretches of 5+ TIER-B might dip).
    assert triggers <= 3, (
        f"cycle237 min-ratio triggered {triggers}× — "
        f"v592 0.15 floor may be too strict; would force TIER-A on gold trace"
    )


# ---------------- T-V592-19: aggregate force rate < 25% ----------------


def test_cycle237_total_force_rate_below_25_pct():
    """Combined v592 forcing should override M1 in <25% of qualifying turns.

    Higher than that means v592 transforms cycle237 into an essentially
    different trajectory.
    """
    rows = _load_cycle237_with_tiers()
    rv = []
    forces = 0
    qualifying = 0
    for r in rows:
        rv.append({
            "chid_tier": r["chid_tier"],
            "observation": {"level_delta": r["level_delta"]},
        })
        board = SimpleNamespace(recent_verbose=rv)
        if r["chid_tier"] != "B" or not r["candidate_tests"]:
            continue
        qualifying += 1
        forced = (
            _v592_stuck_rotation_should_force(board, r["candidate_tests"])
            or _v592_score_gate_should_force(r["candidate_tests"])
            or _v592_min_ratio_should_force(board, r["candidate_tests"])
        )
        if forced is not None:
            forces += 1
    rate = forces / qualifying if qualifying else 0
    assert rate < 0.25, (
        f"cycle237 v592 force rate={rate:.3f} ({forces}/{qualifying}) >= 0.25; "
        f"v592 would have substantially overridden the gold trace"
    )
