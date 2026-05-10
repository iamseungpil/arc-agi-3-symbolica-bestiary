"""v601 integration fixtures (plan rev C §4 INT01-INT06).

Each fixture loads JSON, exercises the relevant role(s), and asserts the
fixture's `expected` block. 100% pass required.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from agents.templates.agentica_lite.memory_writer import (  # noqa: E402
    MemoryWriter, Outcome, analyze, severity,
)
from agents.templates.agentica_lite.policy import (  # noqa: E402
    EpisodeState, compute_saturation_status, select_arm, select_target_marker,
)
from agents.templates.agentica_lite.predicate_library import PredicateLibrary  # noqa: E402
from agents.templates.agentica_lite.predicate_posterior import (  # noqa: E402
    ArmKey, PredicatePosterior,
)
from agents.templates.agentica_lite.proposer import (  # noqa: E402
    ProposerOutput, parse_and_validate,
)
from agents.templates.agentica_lite.proposer_prompt import (  # noqa: E402
    SYSTEM_PROMPT, render_user_prompt,
)
from agents.templates.agentica_lite.stalemate_trigger import (  # noqa: E402
    StalemateConfig, StalemateTrigger,
)

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def _load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------- INT01

def test_int01_one_turn_pipeline():
    fx = _load(FIXTURES_DIR / "INT01_one_turn_R31_pre_T6.json")
    state = fx["input"]["state"]
    mock = fx["input"]["mock_proposer_output"]
    expected = fx["expected"]

    visible_rids = [
        r.get("region_id") or r.get("id") for r in state.get("visible_regions", [])
    ]
    pres = parse_and_validate(mock, visible_rids)
    assert pres.failure_reason is None, f"mock should validate: {pres.schema_error_code}"
    proposer_output = pres.output
    assert proposer_output is not None

    posterior = PredicatePosterior()
    library = PredicateLibrary()
    episode = EpisodeState()
    decision = select_arm(state, posterior, library, proposer_output, episode)
    assert decision is not None
    assert decision.arm_key.predicate_id == expected["arm_predicate_id"]
    assert decision.arm_key.region_id == expected["arm_region_id"]
    assert decision.arm_key.saturation_status == expected["arm_saturation_status"]
    # coord centroid of R36 bbox [36,52,41,57] -> (38, 54)
    assert list(decision.coord_xy) == expected["coord_xy"]
    # Memory writer with no prior outcome at this coord -> no paired-cf trigger
    mw = MemoryWriter()
    out = Outcome(
        coord=decision.coord_xy,
        primary_region_id=state["observation"]["primary_region_id"],
        level_delta=int(state["observation"].get("level_delta") or 0),
        dt_count=int((state["observation"].get("dominant_transition") or {}).get("count", 0)),
        pre_state_features=state.get("pre_state_features") or {},
        turn=state["turn"],
    )
    mw_dec = mw.record(out, current_turn=state["turn"])
    # First record at this coord — no paired-cf possible
    assert mw_dec.paired_cf_entry is None or len(mw_dec.paired_cf_entry.outcomes) == 1
    assert not mw_dec.spawn_reflector


# ----------------------------------------------------------------- INT02

def test_int02_paired_cf_T8_T27():
    fx = _load(FIXTURES_DIR / "INT02_paired_cf_R16_T8_T27.json")
    inp = fx["input"]
    expected = fx["expected"]

    a = inp["outcome_a"]
    b = inp["outcome_b"]
    out_a = Outcome(
        coord=tuple(a["coord"]),
        primary_region_id=a["primary_region_id"],
        level_delta=a["level_delta"],
        dt_count=a["dt_count"],
        pre_state_features=a["pre_state_features"],
        turn=8,
    )
    out_b = Outcome(
        coord=tuple(b["coord"]),
        primary_region_id=b["primary_region_id"],
        level_delta=b["level_delta"],
        dt_count=b["dt_count"],
        pre_state_features=b["pre_state_features"],
        turn=27,
    )
    sev = severity(out_a, out_b)
    assert abs(sev - expected["severity"]) < 1e-3, f"severity {sev} != {expected['severity']}"

    mw = MemoryWriter()
    mw.record(out_a, current_turn=8)
    dec = mw.record(out_b, current_turn=27)
    assert dec.paired_cf_entry is not None
    entry = dec.paired_cf_entry
    assert len(entry.outcomes) == 2
    assert entry.best_pair_severity > 0.5
    assert dec.spawn_reflector is True
    # Discriminator MUST be a saturation_numerator feature (R6 or R12 — both are valid
    # since both markers transitioned to saturation between T8 and T27. Plan §3.10
    # whitelist is feature-FAMILY-level; R6 and R12 share the same family).
    saturation_features = [
        f for f in entry.discriminating_features
        if "compass_saturation_numerator" in f
    ]
    assert saturation_features, (
        f"discriminator should include a marker_*_compass_saturation_numerator "
        f"feature, got {entry.discriminating_features}"
    )


# ----------------------------------------------------------------- INT03

@pytest.mark.parametrize(
    "fixture_name",
    [
        "INT03_stalemate_uniform.json",
        "INT03_stalemate_declining.json",
        "INT03_stalemate_oscillating.json",
    ],
)
def test_int03_stalemate_cadence(fixture_name):
    fx = _load(FIXTURES_DIR / fixture_name)
    inp = fx["input"]
    expected = fx["expected"]

    cfg = StalemateConfig(
        K_threshold=inp["stalemate_K_threshold"],
        theta_threshold=inp["stalemate_theta_threshold"],
        once_per_episode=True,
    )
    trigger = StalemateTrigger(cfg)

    warm_up_calls = 0
    stalemate_calls = 0
    turns_since_L_plus = 0  # synthetic: no L+ throughout
    for turn in inp["turns"]:
        t_idx = turn["turn"]
        # Warm-up at turn 0
        if trigger.warm_up_fires(t_idx):
            warm_up_calls += 1
            trigger.mark_warm_up_done()
        # Stalemate (uses synthetic max_post hint)
        max_post = float(turn.get("max_posterior_hint", 0.0))
        if trigger.fires(turns_since_L_plus, max_post):
            stalemate_calls += 1
            trigger.mark_fired()
        turns_since_L_plus += 1
    assert warm_up_calls == expected["warm_up_proposer_calls"]
    assert stalemate_calls == expected["stalemate_proposer_calls"]
    assert (warm_up_calls + stalemate_calls) == expected["total_proposer_calls"]


# ----------------------------------------------------------------- INT04

@pytest.mark.parametrize(
    "fixture_name",
    [
        "INT04_schema_missing_field.json",
        "INT04_schema_predicate_blacklisted.json",
        "INT04_schema_tool_call_blocked.json",
        "INT04_schema_confidence_out_of_range.json",
        "INT04_schema_region_unknown.json",
    ],
)
def test_int04_schema_violation(fixture_name):
    fx = _load(FIXTURES_DIR / fixture_name)
    raw = fx["input"]["raw_proposer_json"]
    visible_rids = fx["input"]["visible_region_ids"]
    expected = fx["expected"]

    pres = parse_and_validate(raw, visible_rids)
    assert pres.failure_reason == "schema_invalid"
    assert pres.schema_error_code == expected["error_code"], (
        f"expected {expected['error_code']} got {pres.schema_error_code}"
    )
    assert pres.output is None


# ----------------------------------------------------------------- INT05

def _scan_forbidden(text: str, forbidden: list[str]) -> list[str]:
    """Return list of (token, line) hits in text, using \b-bounded regex."""
    hits: list[str] = []
    for tok in forbidden:
        # Use simple substring match for tokens like "saturation_R" (prefix);
        # use word-boundary for short alphanumeric tokens.
        if tok.endswith("_R"):
            pattern = re.compile(re.escape(tok))
        else:
            pattern = re.compile(r"\b" + re.escape(tok) + r"\b")
        if pattern.search(text):
            hits.append(tok)
    return hits


def test_int05a_template_lint():
    fx = _load(FIXTURES_DIR / "INT05a_template_lint.json")
    inp = fx["input"]
    files = [REPO_ROOT / p for p in inp["files_to_lint"]]
    forbidden = inp["forbidden_tokens"]
    all_hits: dict[str, list[str]] = {}
    for f in files:
        if not f.exists():
            continue
        text = f.read_text(encoding="utf-8")
        # Strip docstrings/comments? Plan: scan template source only. Templates
        # don't have ID values; the policy/agent code may reference R-style names
        # internally for predicate IDs (e.g., P00) but those are not forbidden.
        # We restrict to the proposer_prompt.py file (template) and prompt-construction
        # functions in agent.py. agent.py has no prompt-construction text now (those
        # moved to proposer_prompt.py), so we lint both files for the forbidden set.
        hits = _scan_forbidden(text, forbidden)
        if hits:
            all_hits[str(f)] = hits
    assert not all_hits, f"forbidden tokens in templates: {all_hits}"


def test_int05b_rendered_prompt_lint():
    fx = _load(FIXTURES_DIR / "INT05b_rendered_prompt_lint.json")
    inp = fx["input"]
    state = inp["synthetic_state"]
    forbidden = inp["forbidden_tokens"]
    rendered = render_user_prompt(state)
    hits = _scan_forbidden(rendered, forbidden)
    assert not hits, f"forbidden tokens in rendered prompt: {hits}\n--- prompt ---\n{rendered}"
    # Also check the system prompt (template).
    sys_hits = _scan_forbidden(SYSTEM_PROMPT, forbidden)
    assert not sys_hits, f"forbidden tokens in system prompt: {sys_hits}"


# ----------------------------------------------------------------- INT06

def test_int06_n3_high_support():
    fx = _load(FIXTURES_DIR / "INT06_n3_high_support.json")
    inp = fx["input"]
    expected = fx["expected"]
    outs = [
        Outcome(coord=tuple(o["coord"]), primary_region_id=o["primary_region_id"],
                level_delta=o["level_delta"], dt_count=o["dt_count"],
                pre_state_features=o["pre_state_features"], turn=i)
        for i, o in enumerate(inp["outcomes"])
    ]
    a = analyze(outs, current_turn=inp["current_turn"],
                last_reflector_turn=inp["last_reflector_turn"])
    assert a.entry is not None
    assert a.entry.best_pair_severity >= expected["best_pair_severity_min"]
    assert a.entry.support_count >= expected["support_count_min"]
    assert a.spawn_reflector is True


def test_int06_n3_outlier_low_support():
    fx = _load(FIXTURES_DIR / "INT06_n3_outlier_low_support.json")
    inp = fx["input"]
    expected = fx["expected"]
    outs = [
        Outcome(coord=tuple(o["coord"]), primary_region_id=o["primary_region_id"],
                level_delta=o["level_delta"], dt_count=o["dt_count"],
                pre_state_features=o["pre_state_features"], turn=i)
        for i, o in enumerate(inp["outcomes"])
    ]
    a = analyze(outs, current_turn=inp["current_turn"],
                last_reflector_turn=inp["last_reflector_turn"])
    assert a.entry is not None
    assert a.spawn_reflector is False
    assert a.spawn_reason == expected["reason"]


def test_int06_cooldown_suppress():
    fx = _load(FIXTURES_DIR / "INT06_cooldown_suppress.json")
    inp = fx["input"]
    expected = fx["expected"]
    outs1 = [
        Outcome(coord=tuple(o["coord"]), primary_region_id=o["primary_region_id"],
                level_delta=o["level_delta"], dt_count=o["dt_count"],
                pre_state_features=o["pre_state_features"], turn=i)
        for i, o in enumerate(inp["outcomes_first"])
    ]
    a1 = analyze(outs1, current_turn=inp["first_turn"],
                 last_reflector_turn=inp["last_reflector_turn_before_first"])
    assert a1.spawn_reflector is expected["first_spawn"]
    # Now the second contrast is at second_turn, with last_reflector_turn = first_turn
    outs2 = [
        Outcome(coord=tuple(o["coord"]), primary_region_id=o["primary_region_id"],
                level_delta=o["level_delta"], dt_count=o["dt_count"],
                pre_state_features=o["pre_state_features"], turn=i)
        for i, o in enumerate(inp["outcomes_second"])
    ]
    a2 = analyze(outs2, current_turn=inp["second_turn"],
                 last_reflector_turn=inp["first_turn"])
    assert a2.spawn_reflector is expected["second_spawn"]
    assert a2.spawn_reason == expected["second_suppression_reason"]


def test_int06_n4_accumulate():
    fx = _load(FIXTURES_DIR / "INT06_n4_accumulate.json")
    inp = fx["input"]
    expected = fx["expected"]
    mw = MemoryWriter()
    final_outcomes = []
    for i, o in enumerate(inp["outcomes"]):
        out = Outcome(
            coord=tuple(o["coord"]), primary_region_id=o["primary_region_id"],
            level_delta=o["level_delta"], dt_count=o["dt_count"],
            pre_state_features=o["pre_state_features"], turn=i,
        )
        dec = mw.record(out, current_turn=inp["current_turn"])
        if dec.paired_cf_entry is not None:
            final_outcomes = dec.paired_cf_entry.outcomes
    assert len(final_outcomes) == expected["stored_outcome_count"], (
        f"expected {expected['stored_outcome_count']}, got {len(final_outcomes)}"
    )


# ----------------------------------------------------------------- deterministic unit/property

def test_property_arm_key_default_saturation():
    """ArmKey backward-compat: 2-arg form sets saturation_status='n/a'."""
    k = ArmKey("P00", "R0")
    assert k.saturation_status == "n/a"
    k2 = ArmKey("P00", "R0", "complete")
    assert k2.saturation_status == "complete"
    assert k != k2


def test_property_compute_saturation_status_thresholds():
    """Plan §3.6: complete iff all clicked; near_complete iff exactly one unclicked; else none."""
    def m(clicked_dirs):
        compass = {d: {"clicks": 1 if d in clicked_dirs else 0, "region_id": f"R{d}"}
                   for d in ("N", "NE", "E", "SE", "S", "SW", "W", "NW")}
        return {"marker_id": "M", "compass": compass}
    _, _, st = compute_saturation_status(m([]))
    assert st == "none"
    _, _, st = compute_saturation_status(m(["N"]))
    assert st == "none"
    _, _, st = compute_saturation_status(m(["N", "NE", "E", "SE", "S", "SW", "W"]))
    assert st == "near_complete"
    _, _, st = compute_saturation_status(m(["N", "NE", "E", "SE", "S", "SW", "W", "NW"]))
    assert st == "complete"


def test_property_select_target_marker_priority():
    """Plan §3.6 G19' priority list."""
    state = {
        "marker_neighbor_states": [
            {"marker_id": "M_a", "compass": {
                d: {"clicks": 0, "region_id": f"Ra_{d}"}
                for d in ("N", "E", "S", "W")
            }},
            {"marker_id": "M_b", "compass": {
                d: {"clicks": 1 if d != "S" else 0, "region_id": f"Rb_{d}"}
                for d in ("N", "E", "S", "W")
            }},
        ]
    }
    # No proposer hint: should pick M_b (near_complete)
    target, reason = select_target_marker(state, None)
    assert target is not None and target["marker_id"] == "M_b"
    assert reason == "near_complete"

    # With invalid proposer hint (denominator mismatch): demote to near_complete
    class Hint:
        required_pre_state = {"marker_id": "M_a", "saturation_denominator": 99}

    target, reason = select_target_marker(state, Hint())
    assert target["marker_id"] == "M_b"
    assert reason == "near_complete"

    # With valid proposer hint
    class Hint2:
        required_pre_state = {"marker_id": "M_a", "saturation_denominator": 4}

    target, reason = select_target_marker(state, Hint2())
    assert target["marker_id"] == "M_a"
    assert reason == "proposer_hint"


def test_property_rasi_split_floor():
    """Plan §3.8 G21: floor at 1.0 prevents zero counts."""
    posterior = PredicatePosterior()
    posterior.arms[ArmKey("P00", "R0")] = type(posterior.arms[ArmKey("P00", "R0")] if ArmKey("P00", "R0") in posterior.arms else None)  # placeholder
    from agents.templates.agentica_lite.predicate_posterior import ArmStats
    posterior.arms[ArmKey("P00", "R0")] = ArmStats(alpha=0.6, beta=0.6, n_emit=0)
    posterior.split_rasi_with_saturation()
    for status in ("none", "near_complete", "complete"):
        k = ArmKey("P00", "R0", status)
        assert k in posterior.arms
        assert posterior.arms[k].alpha >= 1.0
        assert posterior.arms[k].beta >= 1.0


def test_property_paired_cf_severity_calc():
    """Plan §3.11 severity formula."""
    a = Outcome(coord=(0, 0), primary_region_id="R0", level_delta=0, dt_count=36,
                pre_state_features={})
    b = Outcome(coord=(0, 0), primary_region_id="R0", level_delta=1, dt_count=564,
                pre_state_features={})
    sev = severity(a, b)
    assert abs(sev - 0.9362) < 0.01


def test_property_proposer_schema_valid_minimal():
    raw = {
        "candidate_predicate_id": "P_saturation_progress",
        "region_hint": "R0",
        "expected_signature": {"level_delta": 1},
        "required_pre_state": {
            "marker_id": "M0", "saturation_threshold": 7, "saturation_denominator": 8,
        },
        "confidence": 0.7,
        "thought": f"Step-0 mean(c.clicks >= 1 for c in M.compass) computed.",
    }
    res = parse_and_validate(raw, ["R0", "R1"])
    assert res.failure_reason is None
    assert res.output.confidence == 0.7
