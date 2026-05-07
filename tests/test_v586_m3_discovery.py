"""
M3 hypothesis discovery fixture test.

Verifies the HYPOTHESIZE module's capability to propose joint-multi-region
configuration hypotheses (escalation from single-toggle plateau) — the
upstream signal that allows agent to discover the XOR-parity-style L+
trigger through play, without leaking the rule.

Pass criteria (project memory `feedback_module_fixture_first.md`):
  - train ≥ 80%
  - val   ≥ 75%
  - gap   ≤ 15 pt

Fixtures: tests/fixtures/m3_discovery_fixtures.json (30 entries built
from cycle149..164 post-L+1 traces; 18 train + 12 val by seed-42 shuffle).
"""

import json
import pathlib
import random

FIX = pathlib.Path(__file__).parent / "fixtures" / "m3_discovery_fixtures.json"

POS_KEYWORDS = (
    "joint", "configuration", "multi-region", "pattern", "structural",
    "compass sweep", "directional sector", "marker crop", "subset",
    "multi-step", "sequence of clicks", "across markers",
)


def _score(fixture: dict) -> bool:
    cards = fixture.get("m3_emitted", []) or []
    if not cards:
        return False
    for c in cards:
        text = ((c.get("predicate") or "") + " "
                + (c.get("rule_hypothesis") or "")).lower()
        if c.get("type") == "plan" and len(c.get("click_sequence") or []) >= 2:
            return True
        if any(kw in text for kw in POS_KEYWORDS):
            return True
    return False


def _split():
    fxs = json.loads(FIX.read_text())["fixtures"]
    rng = random.Random(42)
    rng.shuffle(fxs)
    return fxs[:18], fxs[18:]


def test_train_threshold():
    train, _ = _split()
    passes = sum(1 for f in train if _score(f))
    rate = passes / len(train)
    assert rate >= 0.80, f"M3 discovery train rate {rate:.2%} < 80%"


def test_val_threshold():
    _, val = _split()
    passes = sum(1 for f in val if _score(f))
    rate = passes / len(val)
    assert rate >= 0.75, f"M3 discovery val rate {rate:.2%} < 75%"


def test_gap_bounded():
    train, val = _split()
    tr = sum(1 for f in train if _score(f)) / len(train)
    vr = sum(1 for f in val if _score(f)) / len(val)
    assert (tr - vr) <= 0.15, f"M3 discovery train-val gap {tr-vr:.2%} > 15pt"
