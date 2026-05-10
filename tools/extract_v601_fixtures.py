"""Extract v601 INT fixtures from cycle traces (plan rev C §5.6).

INT fixture types (plan §4):
  INT01 — One-Turn Policy Pipeline (mocked Proposer, pre-T6 R31 state)
  INT02 — Paired-CF Discriminator (T8 vs T27 injection)
  INT03 — Stalemate Cadence + Warm-up (3 synthetic 12-turn variants)
  INT04 — Schema Violation Rejection (5 synthetic JSONs)
  INT05 — Game-Agnostic Prompt Lint (template + rendered)
  INT06 — Severity + Cooldown + n>=3 Lifecycle (4 synthetic sub-cases)

Usage:
  extract_v601_fixtures.py --fixture-type INT01 --trace TRACE.jsonl --turn-a N \\
      --out tests/v601/fixtures/INT01_one_turn_R31_pre_T6.json
  extract_v601_fixtures.py --fixture-type INT02 --trace TRACE.jsonl \\
      --turn-a 8 --turn-b 27 --out tests/v601/fixtures/INT02_paired_cf_R16_T8_T27.json
  extract_v601_fixtures.py --fixture-type INT03 --out-dir tests/v601/fixtures/  # writes 3 variants
  extract_v601_fixtures.py --fixture-type INT04 --out-dir tests/v601/fixtures/  # writes 5 variants
  extract_v601_fixtures.py --fixture-type INT05 --out-dir tests/v601/fixtures/  # writes 2 (a/b)
  extract_v601_fixtures.py --fixture-type INT06 --out-dir tests/v601/fixtures/  # writes 4 sub-cases

Pre-state feature whitelist (plan §3.10):
  1. marker_X_compass_saturation_numerator
  2. marker_X_compass_recent_click_direction
  3. recent_dominant_transition_direction
  4. region_X_click_count
  5. level_delta_since_last_paired_cf
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


# ----------------------------------------------------------------- core helpers

def _load_turn(trace_path: Path, idx: int) -> dict[str, Any]:
    with open(trace_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == idx:
                return json.loads(line)
    raise IndexError(f"turn {idx} not in {trace_path}")


def _compute_marker_saturation(marker: dict) -> tuple[int, int, str]:
    """Return (clicked_count, denominator, status) per plan §3.6."""
    compass = marker.get("compass") or {}
    n = len(compass)
    clicked = sum(1 for c in compass.values() if (c or {}).get("clicks", 0) >= 1)
    if n == 0:
        return 0, 0, "n/a"
    if clicked == n:
        return clicked, n, "complete"
    if clicked == n - 1:
        return clicked, n, "near_complete"
    return clicked, n, "none"


def _last_clicked_direction(marker: dict) -> str | None:
    """Return compass direction with most recent (highest) clicks, or None."""
    best_dir, best_clicks = None, 0
    for d, slot in (marker.get("compass") or {}).items():
        c = (slot or {}).get("clicks", 0)
        if c > best_clicks:
            best_clicks = c
            best_dir = d
    return best_dir


def _extract_pre_state_features(turn_record: dict) -> dict[str, Any]:
    """Plan §3.10 whitelist (5 features)."""
    feats: dict[str, Any] = {}
    for m in turn_record.get("marker_neighbor_states") or []:
        mid = m.get("marker_id")
        if mid is None:
            continue
        clicked, _denom, _status = _compute_marker_saturation(m)
        feats[f"marker_{mid}_compass_saturation_numerator"] = clicked
        feats[f"marker_{mid}_compass_recent_click_direction"] = _last_clicked_direction(m)
    obs = turn_record.get("observation") or {}
    dt = obs.get("dominant_transition") or {}
    if dt:
        feats["recent_dominant_transition_direction"] = f"{dt.get('from')}->{dt.get('to')}"
    rcl = turn_record.get("region_clicks_this_level") or {}
    if isinstance(rcl, dict):
        for rid, cnt in rcl.items():
            feats[f"region_{rid}_click_count"] = int(cnt)
    feats["level_delta"] = int(obs.get("level_delta") or 0)
    feats["primary_region_id"] = obs.get("primary_region_id")
    feats["dt_count"] = int(dt.get("count", 0))
    return feats


def _snapshot_turn(turn_record: dict, idx: int) -> dict[str, Any]:
    obs = turn_record.get("observation") or {}
    return {
        "turn": turn_record.get("turn", idx),
        "visible_regions": turn_record.get("visible_regions", []),
        "marker_neighbor_states": turn_record.get("marker_neighbor_states") or [],
        "observation": {
            "primary_region_id": obs.get("primary_region_id"),
            "level_delta": int(obs.get("level_delta") or 0),
            "dominant_transition": obs.get("dominant_transition"),
        },
        "coord": turn_record.get("coord"),
        "pre_state_features": _extract_pre_state_features(turn_record),
    }


# ----------------------------------------------------------------- INT01

def build_int01(trace: Path, turn_a: int) -> dict:
    rec = _load_turn(trace, turn_a)
    snap = _snapshot_turn(rec, turn_a)
    return {
        "id": "INT01_one_turn_R31_pre_T6",
        "fixture_type": "INT01",
        "source_trace": str(trace),
        "source_turn": turn_a,
        "input": {
            "state": snap,
            "mock_proposer_output": {
                "candidate_predicate_id": "P_saturation_progress",
                "region_hint": "R36",
                "expected_signature": {"level_delta": 1},
                "required_pre_state": {
                    "marker_id": "R31",
                    "saturation_threshold": 7,
                    "saturation_denominator": 8,
                },
                "confidence": 0.7,
            },
        },
        "expected": {
            "arm_predicate_id": "P_saturation_progress",
            "arm_region_id": "R36",
            "arm_saturation_status": "none",  # R31 has 2/8 clicked at T5
            "coord_xy": [(36 + 41) // 2, (52 + 57) // 2],  # centroid of R36 bbox
            "paired_cf_triggered": False,
        },
    }


# ----------------------------------------------------------------- INT02

def build_int02(trace: Path, turn_a: int, turn_b: int) -> dict:
    rec_a = _load_turn(trace, turn_a)
    rec_b = _load_turn(trace, turn_b)
    snap_a = _snapshot_turn(rec_a, turn_a)
    snap_b = _snapshot_turn(rec_b, turn_b)
    feats_a = snap_a["pre_state_features"]
    feats_b = snap_b["pre_state_features"]
    dt_a = (snap_a["observation"].get("dominant_transition") or {}).get("count", 0)
    dt_b = (snap_b["observation"].get("dominant_transition") or {}).get("count", 0)
    s_count = abs(dt_a - dt_b) / max(dt_a, dt_b, 1)
    ld_a = snap_a["observation"]["level_delta"]
    ld_b = snap_b["observation"]["level_delta"]
    s_level = 0.5 if ld_a != ld_b else 0.0
    severity = max(s_count, s_level)
    discriminator_features: list[str] = []
    for k, v_a in feats_a.items():
        if k not in feats_b:
            continue
        v_b = feats_b[k]
        if isinstance(v_a, (int, float)) and isinstance(v_b, (int, float)):
            if abs(v_a - v_b) >= 1:
                discriminator_features.append(k)
        elif v_a != v_b:
            discriminator_features.append(k)
    return {
        "id": "INT02_paired_cf_R16_T8_T27",
        "fixture_type": "INT02",
        "source_trace": str(trace),
        "source_turn_a": turn_a,
        "source_turn_b": turn_b,
        "input": {
            "outcome_a": {
                "coord": snap_a["coord"],
                "primary_region_id": snap_a["observation"]["primary_region_id"],
                "level_delta": ld_a,
                "dt_count": dt_a,
                "pre_state_features": feats_a,
            },
            "outcome_b": {
                "coord": snap_b["coord"],
                "primary_region_id": snap_b["observation"]["primary_region_id"],
                "level_delta": ld_b,
                "dt_count": dt_b,
                "pre_state_features": feats_b,
            },
        },
        "expected": {
            "paired_cf_triggered": True,
            "severity": round(severity, 4),
            "spawn_reflector": severity > 0.5,
            "discriminator_top_feature_candidates": discriminator_features,
            "discriminator_must_include": "marker_R12_compass_saturation_numerator",
        },
    }


# ----------------------------------------------------------------- INT03 (synthetic)

def _stub_visible_region(rid: str, x: int, y: int) -> dict:
    return {
        "id": rid, "region_id": rid,
        "bbox": [x, y, x + 4, y + 4],
        "color": 0, "is_multicolor": False,
        "is_marker_neighbor": False, "is_primary_marker": False,
        "y_band": "play_zone",
    }


def _stub_turn(turn: int, rid_primary: str = "R0", level_delta: int = 0,
               dt_count: int = 12, max_post_hint: float = 0.20) -> dict:
    return {
        "turn": turn,
        "visible_regions": [_stub_visible_region(f"R{i}", i * 5, 0) for i in range(3)],
        "marker_neighbor_states": [],
        "observation": {
            "primary_region_id": rid_primary,
            "level_delta": level_delta,
            "dominant_transition": {"from": 0, "to": 1, "count": dt_count},
        },
        "coord": [2, 2],
        "max_posterior_hint": max_post_hint,
    }


def build_int03_variants() -> list[dict]:
    variants: list[dict] = []
    # Variant 1: uniform stall — 12 turns, max_posterior stays low, no L+
    turns = [_stub_turn(t) for t in range(12)]
    variants.append({
        "id": "INT03_stalemate_uniform",
        "fixture_type": "INT03",
        "variant": "uniform",
        "input": {
            "turns": turns,
            "stalemate_K_threshold": 8,
            "stalemate_theta_threshold": 0.45,
        },
        "expected": {
            "warm_up_proposer_calls": 1,  # turn 0
            "stalemate_proposer_calls": 1,  # exactly 1 across the 12-turn window
            "total_proposer_calls": 2,
        },
    })
    # Variant 2: declining max_posterior (starts at 0.6, drops linearly to 0.10)
    turns2 = []
    for t in range(12):
        mp = max(0.10, 0.60 - 0.05 * t)
        turns2.append(_stub_turn(t, max_post_hint=mp))
    variants.append({
        "id": "INT03_stalemate_declining",
        "fixture_type": "INT03",
        "variant": "declining",
        "input": {
            "turns": turns2,
            "stalemate_K_threshold": 8,
            "stalemate_theta_threshold": 0.45,
        },
        "expected": {
            "warm_up_proposer_calls": 1,
            "stalemate_proposer_calls": 1,
            "total_proposer_calls": 2,
        },
    })
    # Variant 3: oscillating max_posterior — alternates 0.20/0.50, but K threshold still met past turn 8
    turns3 = []
    for t in range(12):
        mp = 0.20 if t % 2 == 0 else 0.50
        turns3.append(_stub_turn(t, max_post_hint=mp))
    variants.append({
        "id": "INT03_stalemate_oscillating",
        "fixture_type": "INT03",
        "variant": "oscillating",
        "input": {
            "turns": turns3,
            "stalemate_K_threshold": 8,
            "stalemate_theta_threshold": 0.45,
        },
        "expected": {
            "warm_up_proposer_calls": 1,
            "stalemate_proposer_calls": 1,  # once_per_episode caps at 1
            "total_proposer_calls": 2,
        },
    })
    return variants


# ----------------------------------------------------------------- INT04 (synthetic)

def build_int04_variants() -> list[dict]:
    """5 synthetic malformed JSONs per plan §4 INT04."""
    return [
        {
            "id": "INT04_schema_missing_field",
            "fixture_type": "INT04",
            "variant": "missing_field",
            "input": {
                "raw_proposer_json": {
                    # missing `region_hint`
                    "candidate_predicate_id": "P_saturation_progress",
                    "expected_signature": {"level_delta": 1},
                    "required_pre_state": {
                        "marker_id": "M0", "saturation_threshold": 7,
                        "saturation_denominator": 8,
                    },
                    "confidence": 0.5,
                },
                "visible_region_ids": ["R0", "R1"],
            },
            "expected": {
                "schema_valid": False,
                "error_code": "schema_missing_field",
                "policy_proposer_output": None,
            },
        },
        {
            "id": "INT04_schema_predicate_blacklisted",
            "fixture_type": "INT04",
            "variant": "predicate_blacklisted",
            "input": {
                "raw_proposer_json": {
                    "candidate_predicate_id": "submit_action",  # blacklisted
                    "region_hint": "R0",
                    "expected_signature": {"level_delta": 1},
                    "required_pre_state": {
                        "marker_id": "M0", "saturation_threshold": 7,
                        "saturation_denominator": 8,
                    },
                    "confidence": 0.5,
                },
                "visible_region_ids": ["R0", "R1"],
            },
            "expected": {
                "schema_valid": False,
                "error_code": "predicate_blacklisted",
                "policy_proposer_output": None,
            },
        },
        {
            "id": "INT04_schema_tool_call_blocked",
            "fixture_type": "INT04",
            "variant": "tool_call_blocked",
            "input": {
                "raw_proposer_json": {
                    "candidate_predicate_id": "P_saturation_progress",
                    "region_hint": "R0",
                    "expected_signature": {"level_delta": 1},
                    "required_pre_state": {
                        "marker_id": "M0", "saturation_threshold": 7,
                        "saturation_denominator": 8,
                    },
                    "confidence": 0.5,
                    "tool_calls": [{"name": "submit_action", "args": {}}],  # embedded tool
                },
                "visible_region_ids": ["R0", "R1"],
            },
            "expected": {
                "schema_valid": False,
                "error_code": "tool_call_blocked",
                "policy_proposer_output": None,
            },
        },
        {
            "id": "INT04_schema_confidence_out_of_range",
            "fixture_type": "INT04",
            "variant": "confidence_out_of_range",
            "input": {
                "raw_proposer_json": {
                    "candidate_predicate_id": "P_saturation_progress",
                    "region_hint": "R0",
                    "expected_signature": {"level_delta": 1},
                    "required_pre_state": {
                        "marker_id": "M0", "saturation_threshold": 7,
                        "saturation_denominator": 8,
                    },
                    "confidence": 1.5,  # > 1.0
                },
                "visible_region_ids": ["R0", "R1"],
            },
            "expected": {
                "schema_valid": False,
                "error_code": "confidence_out_of_range",
                "policy_proposer_output": None,
            },
        },
        {
            "id": "INT04_schema_region_unknown",
            "fixture_type": "INT04",
            "variant": "region_unknown",
            "input": {
                "raw_proposer_json": {
                    "candidate_predicate_id": "P_saturation_progress",
                    "region_hint": "R_NOT_VISIBLE",  # not in visible_region_ids
                    "expected_signature": {"level_delta": 1},
                    "required_pre_state": {
                        "marker_id": "M0", "saturation_threshold": 7,
                        "saturation_denominator": 8,
                    },
                    "confidence": 0.5,
                },
                "visible_region_ids": ["R0", "R1"],
            },
            "expected": {
                "schema_valid": False,
                "error_code": "region_unknown",
                "policy_proposer_output": None,
            },
        },
    ]


# ----------------------------------------------------------------- INT05 (synthetic)

def build_int05_variants() -> list[dict]:
    """Two sub-checks: template lint (a) and rendered prompt lint (b)."""
    forbidden_tokens = [
        "R6", "R12", "R16", "R31", "ft09", "XOR", "parity",
        "saturation_R", "marker_R12_compass",
    ]
    return [
        {
            "id": "INT05a_template_lint",
            "fixture_type": "INT05",
            "variant": "template_lint",
            "input": {
                "files_to_lint": [
                    "agents/templates/agentica_lite/proposer_prompt.py",
                    "agents/templates/agentica_lite/agent.py",
                ],
                "forbidden_tokens": forbidden_tokens,
                "allowed_tokens": [
                    "is_primary_marker", "compass", "clicks", "region_id",
                ],
            },
            "expected": {"forbidden_match_count": 0},
        },
        {
            "id": "INT05b_rendered_prompt_lint",
            "fixture_type": "INT05",
            "variant": "rendered_prompt_lint",
            "input": {
                "synthetic_state": {
                    "visible_regions": [
                        {"id": "R12", "region_id": "R12",
                         "bbox": [10, 10, 14, 14], "color": 9, "is_multicolor": False,
                         "y_band": "play_zone",
                         "is_marker_neighbor": False, "is_primary_marker": True},
                        {"id": "R16", "region_id": "R16",
                         "bbox": [36, 46, 41, 51], "color": 4, "is_multicolor": False,
                         "y_band": "play_zone",
                         "is_marker_neighbor": True, "is_primary_marker": False},
                    ],
                    "marker_neighbor_states": [
                        {"marker_id": "R12",
                         "compass": {
                            "N": {"region_id": "R9", "current_color": None, "clicks": 0},
                            "E": {"region_id": "R13", "current_color": None, "clicks": 1},
                         }},
                    ],
                    "observation": {
                        "primary_region_id": "R16",
                        "level_delta": 0,
                        "dominant_transition": {"from": 4, "to": 8, "count": 564},
                    },
                },
                "forbidden_tokens": forbidden_tokens,
            },
            "expected": {
                "forbidden_match_count_in_prompt_prose": 0,
            },
        },
    ]


# ----------------------------------------------------------------- INT06 (synthetic)

def build_int06_variants() -> list[dict]:
    """Severity + cooldown + n>=3 lifecycle (rev C C1)."""
    def _outcome(level_delta: int, dt_count: int, disc_value: int,
                 dom_dir: str | None = None) -> dict:
        feats: dict[str, Any] = {
            "marker_M0_compass_saturation_numerator": disc_value,
            "level_delta": level_delta,
            "dt_count": dt_count,
        }
        if dom_dir is not None:
            feats["recent_dominant_transition_direction"] = dom_dir
        return {
            "coord": [10, 10],
            "primary_region_id": "R0",
            "level_delta": level_delta,
            "dt_count": dt_count,
            "pre_state_features": feats,
        }
    return [
        {
            "id": "INT06_n3_high_support",
            "fixture_type": "INT06",
            "variant": "n3_high_support",
            "input": {
                # 3 outcomes; all 3 pairs differ on saturation_numerator AND have severity>0.5
                "outcomes": [
                    _outcome(0, 50, 1),
                    _outcome(0, 200, 4),
                    _outcome(1, 600, 7),
                ],
                "current_turn": 10,
                "last_reflector_turn": -1,
            },
            "expected": {
                "best_pair_severity_min": 0.6,
                "support_count_min": 2,
                "spawn_reflector": True,
            },
        },
        {
            "id": "INT06_n3_outlier_low_support",
            "fixture_type": "INT06",
            "variant": "n3_outlier_low_support",
            "input": {
                # Construct one pair with sat as discriminator and one pair with
                # dominant-transition-direction as discriminator, leaving the third
                # pair below the severity threshold. Global top_disc is sat (priority 0,
                # higher variance), but only 1 high-severity pair has sat as its
                # pair-level top.
                "outcomes": [
                    _outcome(0, 50, 3, dom_dir="N->S"),
                    _outcome(0, 300, 3, dom_dir="E->W"),
                    _outcome(0, 320, 10, dom_dir="E->W"),
                ],
                "current_turn": 10,
                "last_reflector_turn": -1,
            },
            "expected": {
                "spawn_reflector": False,
                "reason": "support_count_below_2",
            },
        },
        {
            "id": "INT06_cooldown_suppress",
            "fixture_type": "INT06",
            "variant": "cooldown_suppress",
            "input": {
                # Two consecutive severity-high contrasts within 3 turns of each other.
                # First call at turn 7 spawns Reflector; second at turn 10 (only 3 turns later) is suppressed by cooldown.
                "outcomes_first": [
                    _outcome(0, 50, 1),
                    _outcome(1, 600, 7),
                ],
                "outcomes_second": [
                    _outcome(0, 80, 2),
                    _outcome(1, 700, 8),
                ],
                "first_turn": 7,
                "second_turn": 10,
                "last_reflector_turn_before_first": -1,
            },
            "expected": {
                "first_spawn": True,
                "second_spawn": False,
                "second_suppression_reason": "cooldown_active",
            },
        },
        {
            "id": "INT06_n4_accumulate",
            "fixture_type": "INT06",
            "variant": "n4_accumulate",
            "input": {
                "outcomes": [
                    _outcome(0, 50, 1),
                    _outcome(0, 80, 2),
                    _outcome(0, 110, 3),
                    _outcome(1, 600, 7),
                ],
                "current_turn": 10,
                "last_reflector_turn": -1,
            },
            "expected": {
                "stored_outcome_count": 4,
                "no_overwrite": True,
            },
        },
    ]


# ----------------------------------------------------------------- main


def _write(out: Path, obj: dict) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--fixture-type", required=True,
                    choices=["INT01", "INT02", "INT03", "INT04", "INT05", "INT06"])
    ap.add_argument("--trace", type=Path, default=None)
    ap.add_argument("--turn-a", type=int, default=None)
    ap.add_argument("--turn-b", type=int, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    t = args.fixture_type
    if t == "INT01":
        if not (args.trace and args.turn_a is not None and args.out):
            ap.error("INT01 requires --trace --turn-a --out")
        _write(args.out, build_int01(args.trace, args.turn_a))
    elif t == "INT02":
        if not (args.trace and args.turn_a is not None and args.turn_b is not None and args.out):
            ap.error("INT02 requires --trace --turn-a --turn-b --out")
        _write(args.out, build_int02(args.trace, args.turn_a, args.turn_b))
    elif t == "INT03":
        if not args.out_dir:
            ap.error("INT03 requires --out-dir")
        for v in build_int03_variants():
            _write(args.out_dir / f"{v['id']}.json", v)
    elif t == "INT04":
        if not args.out_dir:
            ap.error("INT04 requires --out-dir")
        for v in build_int04_variants():
            _write(args.out_dir / f"{v['id']}.json", v)
    elif t == "INT05":
        if not args.out_dir:
            ap.error("INT05 requires --out-dir")
        for v in build_int05_variants():
            _write(args.out_dir / f"{v['id']}.json", v)
    elif t == "INT06":
        if not args.out_dir:
            ap.error("INT06 requires --out-dir")
        for v in build_int06_variants():
            _write(args.out_dir / f"{v['id']}.json", v)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
