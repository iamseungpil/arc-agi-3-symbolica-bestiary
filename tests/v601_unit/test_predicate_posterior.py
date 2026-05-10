"""Per-module unit tests for predicate_posterior.py.

Plan v602 §11 addendum: 11 critical branch tests for posterior + ArmKey + RASI.

Branches under test:
  1. arm-key default (saturation_status defaults to "n/a")
  2. arm-key with saturation (named status)
  3. repr (saturation-aware __repr__)
  4. RASI floor (split_rasi_with_saturation respects alpha_floor/beta_floor)
  5. no-split-n/a (sentinel arms preserved as well as new sub-arms)
  6. co-decay (recent emissions weighted differently across the window)
  7. record-emission (alpha/beta unchanged, n_emit + last_emit_turn updated)
  8. select skip-no-target (P12 saturation_progress skipped when target_marker_id absent)
  9. select with-target (P12 selectable when state has target_marker_id)
  10. rank-top truncation
  11. load-rasi 7 modes (none / targeted / shuffle / phase_shift / sign_invert /
                         uniform_cal / variance_match)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.templates.agentica_lite.predicate_library import PredicateLibrary  # noqa: E402
from agents.templates.agentica_lite.predicate_posterior import (  # noqa: E402
    ArmKey, ArmStats, PredicatePosterior,
)


# ---------- 1. arm-key default ------------------------------------------------

def test_arm_key_default_saturation_status_is_na():
    """ArmKey(predicate, region) defaults saturation_status to 'n/a' (v600 backward-compat)."""
    k = ArmKey("P03", "R01")
    assert k.saturation_status == "n/a"
    # frozen + hashable
    s: set[ArmKey] = set()
    s.add(k)
    s.add(ArmKey("P03", "R01"))
    assert len(s) == 1


# ---------- 2. arm-key with saturation ---------------------------------------

def test_arm_key_with_named_saturation_distinct():
    """Different saturation_status values produce distinct keys/hashes."""
    k_a = ArmKey("P03", "R01", "near_complete")
    k_b = ArmKey("P03", "R01", "complete")
    k_def = ArmKey("P03", "R01")
    assert k_a != k_b
    assert k_a != k_def
    assert hash(k_a) != hash(k_def)


# ---------- 3. repr -----------------------------------------------------------

def test_arm_key_repr_omits_na_sentinel():
    """__repr__ omits 'n/a' (v600 form), includes named statuses (v601 form)."""
    assert repr(ArmKey("P03", "R01")) == "ArmKey(P03, R01)"
    assert repr(ArmKey("P03", "R01", "near_complete")) == "ArmKey(P03, R01, near_complete)"


# ---------- 4. RASI floor -----------------------------------------------------

def test_split_rasi_floor_prevents_zero():
    """split_rasi_with_saturation enforces alpha_floor/beta_floor on tiny base values."""
    p = PredicatePosterior()
    # tiny base alpha/beta below floor
    p.arms[ArmKey("P01", "R01")] = ArmStats(alpha=0.6, beta=0.3, n_emit=2)
    p.split_rasi_with_saturation(alpha_floor=1.0, beta_floor=1.0)
    # 3 sub-arms + sentinel preserved
    sub_keys = [k for k in p.arms if k.predicate_id == "P01" and k.region_id == "R01"]
    sat_statuses = {k.saturation_status for k in sub_keys}
    assert sat_statuses == {"n/a", "none", "near_complete", "complete"}
    # all sub-arms have alpha>=floor and beta>=floor
    for k in sub_keys:
        if k.saturation_status == "n/a":
            continue
        a = p.arms[k]
        assert a.alpha >= 1.0 and a.beta >= 1.0


# ---------- 5. no-split-n/a sentinel preserved -------------------------------

def test_split_rasi_preserves_sentinel_and_emit_counters():
    """Sentinel arm preserved unchanged; sub-arms inherit n_emit + last counters."""
    p = PredicatePosterior()
    p.arms[ArmKey("P02", "R02")] = ArmStats(
        alpha=6.0, beta=3.0, n_emit=4, last_level_delta=1, last_emit_turn=12,
    )
    p.split_rasi_with_saturation()
    # sentinel preserved
    sentinel = p.arms[ArmKey("P02", "R02")]
    assert sentinel.alpha == 6.0 and sentinel.beta == 3.0
    # each sub-arm carries the original counters
    for status in ("none", "near_complete", "complete"):
        sub = p.arms[ArmKey("P02", "R02", status)]
        assert sub.n_emit == 4
        assert sub.last_emit_turn == 12
        assert sub.last_level_delta == 1


# ---------- 6. co-decay -------------------------------------------------------

def test_update_applies_co_decay_credit():
    """Most-recent emission gets full credit (1.0); older gets 0.5."""
    p = PredicatePosterior()
    older_key = ArmKey("P01", "R01")
    recent_key = ArmKey("P02", "R02")
    p.record_emission([older_key])
    p.record_emission([recent_key])
    # Now update with positive level_delta
    p.update({"level_delta": 1})
    # recent_key alpha should be 1.0+1.0 = 2.0
    assert p.arms[recent_key].alpha == 2.0
    # older_key alpha should be 1.0+0.5 = 1.5
    assert p.arms[older_key].alpha == 1.5


# ---------- 7. record-emission ------------------------------------------------

def test_record_emission_updates_counters_only():
    """record_emission increments n_emit + last_emit_turn but does NOT change alpha/beta."""
    p = PredicatePosterior()
    k = ArmKey("P01", "R01")
    p.record_emission([k])
    arm = p.arms[k]
    assert arm.n_emit == 1
    assert arm.last_emit_turn == 1
    assert arm.alpha == 1.0 and arm.beta == 1.0  # unchanged
    # Second emission
    p.record_emission([k])
    assert p.arms[k].n_emit == 2
    assert p.arms[k].last_emit_turn == 2


# ---------- 8. select skip-no-target -----------------------------------------

def test_select_skips_saturation_progress_when_no_target():
    """Without state.target_marker_id, P12_saturation_progress is excluded from select()."""
    p = PredicatePosterior()
    library = PredicateLibrary()
    regions = [{"region_id": "R01", "id": "R01"}]
    chosen = p.select(regions, library, state=None)
    assert chosen is not None
    pid, rid = chosen
    assert pid not in ("P12_saturation_progress", "P_saturation_progress")
    assert rid == "R01"
    # also state-with-no-target should also skip
    chosen2 = p.select(regions, library, state={"target_marker_id": None})
    assert chosen2 is not None
    assert chosen2[0] not in ("P12_saturation_progress", "P_saturation_progress")


# ---------- 9. select with-target --------------------------------------------

def test_select_allows_saturation_progress_with_target():
    """With state.target_marker_id set, P12 is in the candidate set; rank_top reflects it."""
    p = PredicatePosterior()
    library = PredicateLibrary()
    regions = [{"region_id": "R01", "id": "R01"}]
    rows = p.rank_top(regions, library, k=20, state={"target_marker_id": "M0"})
    pids = {r[0] for r in rows}
    # P12 family (priority 5.0) should rank top in cold-start
    assert "P12_saturation_progress" in pids
    # rank_top sorts highest first
    assert rows[0][0] == "P12_saturation_progress"


# ---------- 10. rank-top -----------------------------------------------------

def test_rank_top_caps_at_k():
    """rank_top returns at most k rows."""
    p = PredicatePosterior()
    library = PredicateLibrary()
    regions = [{"region_id": f"R{i}", "id": f"R{i}"} for i in range(5)]
    rows = p.rank_top(regions, library, k=3)  # state=None -> P12 skipped
    assert len(rows) == 3
    # Cold start: all scores are 1e6 + family_priority -> should be sorted desc
    assert rows[0][2] >= rows[1][2] >= rows[2][2]


# ---------- 11. load-rasi 7 modes --------------------------------------------

def test_load_rasi_seven_modes(tmp_path):
    """All 7 RASI modes execute without error and produce expected arm-side effects."""
    trace = tmp_path / "trace.jsonl"
    rows = [
        {"predicate_id": "P01", "region_id": "R01", "level_delta": 1},
        {"predicate_id": "P02", "region_id": "R02", "level_delta": 1},
        {"predicate_id": "P03", "region_id": "R03", "level_delta": 0},  # filtered out
    ]
    trace.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")

    # Mode "none": no-op
    p_none = PredicatePosterior()
    p_none.load_rasi_prior(str(trace), weight=1.0, mode="none")
    assert p_none.arms == {}

    # Mode "targeted" (default): each L+ event adds weight to alpha
    p_t = PredicatePosterior()
    p_t.load_rasi_prior(str(trace), weight=1.0, mode="targeted")
    assert p_t.arms[ArmKey("P01", "R01")].alpha == 2.0  # 1 + 1
    assert p_t.arms[ArmKey("P02", "R02")].alpha == 2.0

    # Mode "shuffle": pids permuted; same set of arms but possibly different mapping
    p_s = PredicatePosterior()
    p_s.load_rasi_prior(str(trace), weight=1.0, mode="shuffle")
    # 2 arms still installed (deterministic seed=0)
    assert len(p_s.arms) == 2

    # Mode "phase_shift": rids rotated by 5 (only 2 events -> rids unchanged after slice)
    p_ph = PredicatePosterior()
    p_ph.load_rasi_prior(str(trace), weight=1.0, mode="phase_shift")
    # rids[5:] + rids[:5] over 2 events == same 2 events; arms should still match pids
    assert len(p_ph.arms) == 2

    # Mode "sign_invert": adds to beta (negative evidence)
    p_si = PredicatePosterior()
    p_si.load_rasi_prior(str(trace), weight=1.0, mode="sign_invert")
    assert p_si.arms[ArmKey("P01", "R01")].beta == 2.0
    assert p_si.arms[ArmKey("P02", "R02")].beta == 2.0

    # Mode "uniform_cal": share weight across all arms
    p_uc = PredicatePosterior()
    p_uc.load_rasi_prior(str(trace), weight=2.0, mode="uniform_cal")
    # 2 arms, weight 2.0 -> share = 1.0 each
    assert p_uc.arms[ArmKey("P01", "R01")].alpha == 2.0  # 1.0 + 1.0
    assert p_uc.arms[ArmKey("P02", "R02")].alpha == 2.0

    # Mode "variance_match": deterministic seed=0; some random spread
    p_vm = PredicatePosterior()
    p_vm.load_rasi_prior(str(trace), weight=1.0, mode="variance_match")
    assert len(p_vm.arms) == 2
    # alpha should be > 1.0 (base) since 0.5 <= rng.uniform(0.5, 1.5) <= 1.5
    for k in [ArmKey("P01", "R01"), ArmKey("P02", "R02")]:
        assert p_vm.arms[k].alpha > 1.0
