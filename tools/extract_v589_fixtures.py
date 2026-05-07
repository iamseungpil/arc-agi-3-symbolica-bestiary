"""Extract v589 B17 candidate-generator fixtures from prior trace files.

Per project memory feedback_module_fixture_first.md: ≥30 fixtures,
train/val 60/40, train≥80% / val≥75% / gap≤15pt before live autoresearch.

Each fixture is a self-contained snapshot of a turn from existing
trace.jsonl files containing all inputs the candidate_generator needs.
Output: tests/fixtures/v589_b17_fixtures.json
"""

from __future__ import annotations

import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SL = REPO / "simple_logs" / "ft09-9ab2447a"
OUT = REPO / "tests" / "fixtures" / "v589_b17_fixtures.json"


def parse_trace(p: Path) -> list[dict]:
    out = []
    if not p.exists():
        return out
    with p.open("r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
    return out


def good_fixture(entry: dict) -> bool:
    """A fixture is "rich enough" when:
       - visible_regions has ≥1 multicolor marker
       - turn_index ≥ 5 (so chain_tokens can be non-trivial)
       - has at least some recent_turn_diffs context.
    """
    vrs = entry.get("visible_regions") or []
    if not isinstance(vrs, list) or len(vrs) < 2:
        return False
    has_marker = any(
        isinstance(r, dict) and r.get("is_multicolor") for r in vrs
    )
    if not has_marker:
        return False
    if int(entry.get("turn") or 0) < 5:
        return False
    return True


def build_fixture(entry: dict, prev_diffs: list[dict],
                  recent_clicks: list[str]) -> dict:
    """Each fixture carries the inputs candidate_generator expects."""
    return {
        "namespace": entry.get("_namespace", ""),
        "turn": int(entry.get("turn") or 0),
        "visible_regions": entry.get("visible_regions") or [],
        "recent_turn_diffs": prev_diffs[-15:],
        "marker_neighbor_states": entry.get("marker_neighbor_states") or [],
        "level_bridges": [],
        "chain_rule_log": [],
        "role_history": {},
        "recent_emissions": [],
        "recent_clicks": recent_clicks[-5:],
        "turn_index": int(entry.get("turn") or 0),
        "chain_tokens_len": len(
            (entry.get("action_state_chain_compact") or {})
            .get("chain_tokens", []) or []
        ),
    }


def main() -> int:
    random.seed(42)
    cycles = sorted(SL.glob("cycle*"), key=lambda p: p.name, reverse=True)
    fixtures: list[dict] = []
    for cdir in cycles:
        tp = cdir / "trace.jsonl"
        entries = parse_trace(tp)
        if not entries:
            continue
        # Track running prev_diffs and recent_clicks per cycle.
        # We don't have direct recent_turn_diffs per entry pre-B12
        # cycles; reconstruct simple from observation.
        prev_diffs: list[dict] = []
        recent_clicks: list[str] = []
        for e in entries:
            obs = e.get("observation") or {}
            # Synthesize a "diff" from observation for old cycles.
            diff = {
                "turn": int(e.get("turn") or 0),
                "click_region_id": obs.get("primary_region_id"),
                "level_delta": int(obs.get("level_delta") or 0),
                "compass_changes": [],
                "color_transitions": (
                    [{"region_id": obs.get("primary_region_id"),
                      "from": (obs.get("dominant_transition") or {}).get("from"),
                      "to": (obs.get("dominant_transition") or {}).get("to")}]
                    if (obs.get("dominant_transition") or {}).get("from") is not None
                    else []
                ),
                "region_kind_pre": "marker_multicolor"
                    if any(isinstance(r, dict) and r.get("is_multicolor")
                           and r.get("id") == obs.get("primary_region_id")
                           for r in (e.get("visible_regions") or []))
                    else ("non_marker" if obs.get("primary_region_id")
                          and obs.get("primary_region_id") != "_outside_"
                          else "outside"),
                "region_color_pre": next(
                    (r.get("color") for r in (e.get("visible_regions") or [])
                     if isinstance(r, dict)
                     and r.get("id") == obs.get("primary_region_id")),
                    None,
                ),
            }
            # Prefer the in-trace turn_diff entries (B12+) when present.
            tracted_diffs = e.get("recent_turn_diffs") or []
            if tracted_diffs:
                prev_diffs = list(tracted_diffs)
            else:
                prev_diffs.append(diff)
                if len(prev_diffs) > 20:
                    prev_diffs = prev_diffs[-20:]

            rid = obs.get("primary_region_id")
            if rid and rid != "_outside_":
                recent_clicks.append(rid)
                if len(recent_clicks) > 10:
                    recent_clicks = recent_clicks[-10:]

            if not good_fixture(e):
                continue
            e2 = dict(e)
            e2["_namespace"] = cdir.name
            fixtures.append(build_fixture(e2, prev_diffs, recent_clicks))
            if len(fixtures) >= 60:
                break
        if len(fixtures) >= 60:
            break

    # Shuffle and split.
    random.shuffle(fixtures)
    fixtures = fixtures[:50]
    train_n = int(len(fixtures) * 0.6)
    train = fixtures[:train_n]
    val = fixtures[train_n:]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "n_total": len(fixtures),
        "n_train": len(train),
        "n_val": len(val),
        "train": train,
        "val": val,
    }, indent=2, default=str), encoding="utf-8")
    print(f"Saved {len(fixtures)} fixtures (train={len(train)}, val={len(val)}) to {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
