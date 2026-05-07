"""B17 plan v589 — fixture-coverage tests per project memory.

feedback_module_fixture_first.md mandates:
  ≥30 fixtures, train/val 60/40, train≥80%, val≥75%, gap≤15pt
  before live autoresearch.

Each fixture is a real-state snapshot from prior trace files.
Score: 1 if candidate_generator emits ≥1 leak-safe candidate AND at
least one role with anchor_marker_id is present (testing actual
applicability, not just non-empty list).
"""
import json
import re
import pytest
from pathlib import Path

from tools.candidate_generator import (
    ALLOWED_ROLES,
    _FORBIDDEN_VOCAB,
    generate_candidates,
)


FIX = Path(__file__).parent / "fixtures" / "v589_b17_fixtures.json"


def _score(fixture: dict) -> bool:
    """1 if generator emits ≥1 leak-safe candidate that targets an
    actual marker (anchor_marker_id non-null) and uses an allowed role.
    0 otherwise (empty emission OR only non-marker proposals OR leak)."""
    out = generate_candidates(
        visible_regions=fixture.get("visible_regions") or [],
        recent_turn_diffs=fixture.get("recent_turn_diffs") or [],
        marker_neighbor_states=fixture.get("marker_neighbor_states") or [],
        level_bridges=fixture.get("level_bridges") or [],
        chain_rule_log=fixture.get("chain_rule_log") or [],
        role_history=fixture.get("role_history") or {},
        recent_emissions=fixture.get("recent_emissions") or [],
        recent_clicks=fixture.get("recent_clicks") or [],
        turn_index=int(fixture.get("turn_index") or 0),
        chain_tokens_len=int(fixture.get("chain_tokens_len") or 0),
    )
    if not out:
        return False
    # Anti-leak.
    if _FORBIDDEN_VOCAB.search(json.dumps(out)):
        return False
    # Allowed-role-only.
    for c in out:
        role = (c.get("suggested_test") or {}).get("role")
        if role not in ALLOWED_ROLES:
            return False
    # Anchor non-null on at least one entry (otherwise candidates
    # are not actually applicable to the current state).
    has_anchor = any(
        (c.get("suggested_test") or {}).get("anchor_marker_id")
        for c in out
    )
    return bool(has_anchor)


def _load_split():
    data = json.loads(FIX.read_text())
    return data["train"], data["val"]


def test_v589_fixture_train_coverage():
    train, _ = _load_split()
    rate = sum(1 for f in train if _score(f)) / max(1, len(train))
    assert rate >= 0.80, (
        f"v589 fixture train coverage {rate:.2%} < 80% "
        f"(n_train={len(train)})"
    )


def test_v589_fixture_val_coverage():
    _, val = _load_split()
    rate = sum(1 for f in val if _score(f)) / max(1, len(val))
    assert rate >= 0.75, (
        f"v589 fixture val coverage {rate:.2%} < 75% "
        f"(n_val={len(val)})"
    )


def test_v589_fixture_gap_bounded():
    train, val = _load_split()
    tr = sum(1 for f in train if _score(f)) / max(1, len(train))
    vr = sum(1 for f in val if _score(f)) / max(1, len(val))
    assert (tr - vr) <= 0.15, (
        f"v589 fixture train-val gap {(tr - vr):.2%} > 15pt "
        f"(train={tr:.2%} val={vr:.2%})"
    )


def test_v589_fixture_count_minimum():
    """Project memory: ≥30 fixtures total."""
    data = json.loads(FIX.read_text())
    assert data["n_total"] >= 30, f"only {data['n_total']} fixtures"
