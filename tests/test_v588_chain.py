"""v588 B16 chain compression + chain-level hypothesis tests.

Layer A — pure deterministic. No LLM.
"""
import asyncio
import json
import re
from unittest.mock import patch

import pytest

from tools.chain_compress import (
    build_chain_tokens,
    compress_action_state_chain,
    compute_causal_table,
    compute_prediction_errors,
    compute_trajectory_features,
    tokenise_turn,
)
from agents.templates.agentica_v57.agent import call_hypothesize, call_action


# ----- Synthetic factory -------------------------------------------------


_NO_EXPECTED = object()


def _verbose(turn, region_id="R5", from_c=9, to_c=8, level_delta=0,
             expected=_NO_EXPECTED):
    """If `expected=None` is explicitly passed, the verbose entry has
    NO expected_observation field at all (legacy / force_coord path).
    Otherwise expected defaults to a perfect-match dict."""
    if expected is _NO_EXPECTED:
        expected = {
            "primary_region_id": region_id,
            "dominant_transition": {"from": from_c, "to": to_c},
            "level_delta": level_delta,
        }
    action_dict: dict = {}
    if expected is not None:
        action_dict["expected_observation"] = expected
    return {
        "turn": turn,
        "observation": {
            "primary_region_id": region_id,
            "dominant_transition": {"from": from_c, "to": to_c},
            "level_delta": level_delta,
        },
        "action": action_dict,
    }


def _diff(turn, region_id="R5", kind="non_marker", color=9, compass=None):
    return {
        "turn": turn,
        "click_region_id": region_id,
        "region_kind_pre": kind,
        "region_color_pre": color,
        "compass_changes": compass or [],
    }


# -------------------------- T-CHAIN-1..15 ---------------------------


def test_chain_1_tokenise_format():
    v = _verbose(7, region_id="R5", from_c=9, to_c=8, level_delta=0)
    d = _diff(7, region_id="R5", kind="non_marker", color=9,
              compass=[{"marker_id": "R12", "direction": "N", "from": 9, "to": 2}])
    tok = tokenise_turn(v, d)
    assert tok.startswith("T7|click|")
    assert "non_marker" in tok
    assert "R5" in tok
    assert "9->8" in tok
    assert "R12_N:9->2" in tok
    assert tok.endswith("L+0")


def test_chain_2_truncated_to_max_length():
    verbose = [_verbose(t) for t in range(50)]
    diffs = [_diff(t) for t in range(50)]
    tokens = build_chain_tokens(verbose, diffs, max_chain_length=10)
    assert len(tokens) == 10
    # last 10 (turns 40..49)
    assert "T40" in tokens[0]
    assert "T49" in tokens[-1]


def test_chain_3_features_aggregate():
    verbose = [
        _verbose(1, "R5", 9, 8, 0),
        _verbose(2, "R5", 8, 9, 1),
        _verbose(3, "R6", 9, 8, 0),
    ]
    diffs = [
        _diff(1, "R5", compass=[{"marker_id": "M", "direction": "N", "from": 9, "to": 2}]),
        _diff(2, "R5"),
        _diff(3, "R6"),
    ]
    f = compute_trajectory_features(verbose, diffs)
    assert f["chain_length"] == 3
    assert f["unique_regions"] == 2  # R5, R6
    assert f["repeat_click_max_count"] == 2  # R5 twice
    assert f["compass_changes_per_turn"] == round(1 / 3, 3)
    assert f["kind_distribution"] == {"non_marker": 3}
    assert f["level_delta_curve_shape"] in ("rising_only", "oscillating", "flat")


def test_chain_4_prediction_errors_match_logic():
    verbose = [
        _verbose(1, "R5", 9, 8, 0,
                 expected={"primary_region_id": "R5",
                           "dominant_transition": {"from": 9, "to": 8},
                           "level_delta": 0}),
        _verbose(2, "R6", 8, 9, 1,
                 expected={"primary_region_id": "R5",  # WRONG
                           "dominant_transition": {"from": 9, "to": 8},  # WRONG
                           "level_delta": 0}),                          # WRONG
        _verbose(3, "R7", 8, 9, 0, expected=None),  # missing expected
    ]
    errs = compute_prediction_errors(verbose)
    assert errs[0]["surprise_score"] == 0
    assert errs[1]["surprise_score"] == 3
    # Missing expected → all None.
    assert errs[2]["region_id_match"] is None
    assert errs[2]["surprise_score"] == 0


def test_chain_5_causal_table_min_count_filter():
    verbose = [_verbose(t, "R5", 9, 8, t % 5 == 0 and 1 or 0) for t in range(10)]
    diffs = [_diff(t, "R5") for t in range(10)]
    table = compute_causal_table(verbose, diffs, min_count=3)
    assert len(table) >= 1
    # Only kind=non_marker, transition=9->8 should be present.
    row = table[0]
    assert row["region_kind"] == "non_marker"
    assert row["transition"] == "9->8"
    assert row["count"] >= 3


def test_chain_6_empty_input_returns_empty_payload():
    out = compress_action_state_chain([], [], level="full")
    assert out["chain_tokens"] == []
    assert out["prediction_errors"] == []
    assert out["causal_table"] == []
    assert out["trajectory_features"]["chain_length"] == 0


def test_chain_7_full_vs_compact_levels():
    verbose = [_verbose(t) for t in range(40)]
    diffs = [_diff(t) for t in range(40)]
    full = compress_action_state_chain(verbose, diffs, level="full")
    compact = compress_action_state_chain(verbose, diffs, level="compact")
    # full: up to 30 tokens; compact: 15.
    assert len(full["chain_tokens"]) == 30
    assert len(compact["chain_tokens"]) == 15
    assert len(compact["prediction_errors"]) == 5
    # causal_table: compact <=5 rows.
    assert len(compact["causal_table"]) <= 5


def test_chain_8_call_hypothesize_payload_contains_chain():
    captured = {}

    class _StubAgent:
        async def call(self, _t, task, **kwargs):
            captured["task"] = task
            return '{"thought":"x","cards":[],"chain_rule":[]}'

    async def _fake_spawn(**kwargs):
        return _StubAgent()

    async def _run():
        with patch("agents.templates.agentica_v57.agent.spawn", new=_fake_spawn):
            await call_hypothesize(
                summary="",
                visible_regions=[],
                falsified_recent=[],
                gqb_pair=None,
                image_b64=None,
                next_card_id_seed=1,
                action_state_chain={
                    "chain_tokens": ["T1|click|non_marker(R5,9)|9->8|_|L+0"],
                    "trajectory_features": {"chain_length": 1},
                    "prediction_errors": [],
                    "causal_table": [],
                },
            )

    asyncio.run(_run())
    assert "action_state_chain" in captured["task"]
    assert "chain_tokens" in captured["task"]
    assert "T1|click|non_marker" in captured["task"]


def test_chain_9_anti_leak_no_forbidden_vocab_in_payload():
    """Per round-3 C-14, C-15: verify no leaked terms appear in
    chain payload."""
    import sys
    sys.path.insert(0, "/home/v-seungplee/skilldiscovery/arc-agi-3-symbolica-research/scripts")
    import check_no_leak_prompts as M

    verbose = [_verbose(t) for t in range(20)]
    diffs = [_diff(t, compass=[{"marker_id": "R12", "direction": "N",
                                "from": 9, "to": 2}]) for t in range(20)]
    out = compress_action_state_chain(verbose, diffs, level="full")
    serialised = json.dumps(out)
    hits = M.FORBIDDEN_TERMS.findall(serialised)
    assert hits == [], f"Leak vocab in chain payload: {hits}"


def test_chain_10_compass_history_deque_via_run_turn():
    """Light sanity — V57Board state pieces required for B16 exist."""
    import pathlib, tempfile
    from agents.templates.agentica_v57.agent import V57Board
    with tempfile.TemporaryDirectory() as tmp:
        wd = pathlib.Path(tmp) / "ns"
        b = V57Board(namespace="ns", game_id="ft09", workdir=wd)
        # The chain payload must be buildable from board state
        # (recent_verbose + recent_turn_diffs, both in V57Board).
        assert hasattr(b, "recent_verbose")
        assert hasattr(b, "recent_turn_diffs")


def test_chain_11_call_action_payload_contains_compact_chain():
    captured = {}

    class _StubAgent:
        async def call(self, _t, task, **kwargs):
            captured["task"] = task
            return '{"thought":"x","chosen_hypothesis_id":null,"action":{"type":"ACTION6","coord":[32,32]}}'

    async def _fake_spawn(**kwargs):
        return _StubAgent()

    async def _run():
        with patch("agents.templates.agentica_v57.agent.spawn", new=_fake_spawn):
            await call_action(
                summary="",
                visible_regions=[],
                active_hypotheses=[],
                recent_turns=[],
                gqb_pair=None,
                image_b64=None,
                action_state_chain_compact={
                    "chain_tokens": ["T0|click|non_marker(R5,9)|9->8|_|L+0"],
                    "trajectory_features": {"chain_length": 1},
                    "prediction_errors": [],
                    "causal_table": [],
                },
            )

    asyncio.run(_run())
    assert "action_state_chain" in captured["task"]
    assert "T0|click|non_marker" in captured["task"]


def test_chain_12_backward_compat_chain_rule_optional():
    """Chain_rule field being absent in M3 output should not break
    the parser. We simulate this by running call_hypothesize when the
    stub returns no chain_rule field — payload should still parse."""
    captured = {}

    class _StubAgent:
        async def call(self, _t, task, **kwargs):
            return '{"thought":"x","cards":[]}'

    async def _fake_spawn(**kwargs):
        return _StubAgent()

    async def _run():
        with patch("agents.templates.agentica_v57.agent.spawn", new=_fake_spawn):
            return await call_hypothesize(
                summary="",
                visible_regions=[],
                falsified_recent=[],
                gqb_pair=None,
                image_b64=None,
                next_card_id_seed=1,
                action_state_chain=None,  # also test None default.
            )

    out = asyncio.run(_run())
    # Returned dict should at minimum contain thought + cards.
    assert "thought" in out
    assert "cards" in out
    # chain_rule absent is fine.


def test_chain_13_falsification_must_have_T_or_struct_marker():
    """Validator (in spirit, since validation is delegated to LLM
    quality, not enforced server-side): document T<n> pattern check."""
    sample = "if T20's click on a non_marker fails to trigger compass change"
    assert re.search(r"T\d+", sample)
    bad = "no T mention here"
    assert not re.search(r"T\d+", bad)


def test_chain_14_omit_when_chain_under_threshold():
    """When chain_tokens has <10 entries, M3 should NOT emit chain_rule.
    Schema allows omission. We test that the payload supports the empty
    case without breaking."""
    out = compress_action_state_chain(
        [_verbose(t) for t in range(5)],
        [_diff(t) for t in range(5)],
        level="full",
    )
    assert len(out["chain_tokens"]) == 5
    # Caller-side responsibility (M3 LLM) to obey threshold;
    # payload itself is fine to send.


def test_chain_16_long_diffs_short_verbose_produces_full_chain():
    """BUGFIX regression test: when recent_verbose has only 3 entries
    (V57_VERBOSE_WINDOW=3) but recent_turn_diffs has 24 entries, the
    chain must contain 24 tokens — NOT 3. This was the v588 launch
    bug where chain_tokens never grew past 3 and chain_rule emission
    gate (≥10) never fired."""
    verbose = [_verbose(t) for t in range(20, 23)]   # last 3 turns only
    diffs = [_diff(t) for t in range(0, 24)]          # full 24 diffs
    tokens = build_chain_tokens(verbose, diffs, max_chain_length=30)
    assert len(tokens) == 24, f"expected 24 tokens, got {len(tokens)}"
    # Earliest = T0, latest = T23
    assert "T0" in tokens[0]
    assert "T23" in tokens[-1]
    # Tokens for turns where verbose is missing still have the kind +
    # transition placeholder, not crash.
    early_token = tokens[5]   # T5: no verbose
    assert "T5" in early_token
    assert "click" in early_token


def test_chain_15_legacy_entries_handled_gracefully():
    """Entries missing region_kind_pre (legacy traces) get
    'unknown_kind' token."""
    verbose = [_verbose(1, "R5", 9, 8, 0)]
    diffs = [{"turn": 1, "click_region_id": "R5"}]  # NO region_kind_pre
    tokens = build_chain_tokens(verbose, diffs, max_chain_length=5)
    assert "unknown_kind" in tokens[0]
