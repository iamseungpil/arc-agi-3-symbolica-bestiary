"""B18 v590 fixture-coverage tests per project memory."""
import json
import re
from pathlib import Path

from tools.predicate_generator import (
    PREDICATE_LIBRARY,
    _FORBIDDEN_VOCAB,
    build_chain_state_from_inputs,
    generate_predicates,
)

FIX = Path(__file__).parent / "fixtures" / "v590_b18_fixtures.json"


def _score(fixture: dict) -> bool:
    """1 if generator emits ≥1 leak-safe predicate that targets a
    real visible region (anchor.region_id matches a visible region)."""
    cs = build_chain_state_from_inputs(
        visible_regions=fixture.get("visible_regions") or [],
        recent_turn_diffs=fixture.get("recent_turn_diffs") or [],
        marker_neighbor_states=fixture.get("marker_neighbor_states") or [],
        recent_clicks=fixture.get("recent_clicks") or [],
    )
    out = generate_predicates(
        chain_state=cs,
        turn_index=int(fixture.get("turn_index") or 0),
        template_history={},
    )
    if not out:
        return False
    if _FORBIDDEN_VOCAB.search(json.dumps(out)):
        return False
    visible_ids = {r.get("id") for r in (fixture.get("visible_regions") or [])
                   if isinstance(r, dict)}
    for p in out:
        rid = (p.get("anchor_region") or {}).get("region_id")
        if rid not in visible_ids:
            return False
        if p.get("template_id") not in PREDICATE_LIBRARY:
            return False
    return True


def _load_split():
    data = json.loads(FIX.read_text())
    return data["train"], data["val"]


def test_v590_fixture_count_min():
    data = json.loads(FIX.read_text())
    assert data["n_total"] >= 30, f"only {data['n_total']} fixtures"


def test_v590_fixture_train_coverage():
    train, _ = _load_split()
    rate = sum(1 for f in train if _score(f)) / max(1, len(train))
    assert rate >= 0.80, f"v590 train coverage {rate:.2%} < 80%"


def test_v590_fixture_val_coverage():
    _, val = _load_split()
    rate = sum(1 for f in val if _score(f)) / max(1, len(val))
    assert rate >= 0.75, f"v590 val coverage {rate:.2%} < 75%"


def test_v590_fixture_gap():
    train, val = _load_split()
    tr = sum(1 for f in train if _score(f)) / max(1, len(train))
    vr = sum(1 for f in val if _score(f)) / max(1, len(val))
    assert (tr - vr) <= 0.15, f"gap {(tr - vr):.2%} > 15pt"
