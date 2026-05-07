"""Extract v590 B18 predicate-generator fixtures from prior trace files.

Per project memory feedback_module_fixture_first.md.
Output: tests/fixtures/v590_b18_fixtures.json
"""
from __future__ import annotations
import json
import random
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SL = REPO / "simple_logs" / "ft09-9ab2447a"
OUT = REPO / "tests" / "fixtures" / "v590_b18_fixtures.json"


def parse_trace(p):
    out = []
    if not p.exists(): return out
    with p.open("r") as f:
        for ln in f:
            ln = ln.strip()
            if not ln: continue
            try: out.append(json.loads(ln))
            except: continue
    return out


def good(e):
    vrs = e.get("visible_regions") or []
    if len(vrs) < 2: return False
    if not any(isinstance(r, dict) and r.get("is_multicolor") for r in vrs):
        return False
    if int(e.get("turn") or 0) < 5: return False
    return True


def build_fixture(entry, prev_diffs, recent_clicks):
    return {
        "namespace": entry.get("_namespace", ""),
        "turn": int(entry.get("turn") or 0),
        "visible_regions": entry.get("visible_regions") or [],
        "recent_turn_diffs": prev_diffs[-15:],
        "marker_neighbor_states": entry.get("marker_neighbor_states") or [],
        "recent_clicks": recent_clicks[-10:],
        "turn_index": int(entry.get("turn") or 0),
    }


def main():
    random.seed(42)
    fixtures = []
    for cdir in sorted(SL.glob("cycle*"), key=lambda p: p.name, reverse=True):
        tp = cdir / "trace.jsonl"
        entries = parse_trace(tp)
        if not entries: continue
        prev_diffs, recent_clicks = [], []
        for e in entries:
            obs = e.get("observation") or {}
            tracted = e.get("recent_turn_diffs") or []
            if tracted:
                prev_diffs = list(tracted)
            else:
                diff = {
                    "turn": int(e.get("turn") or 0),
                    "click_region_id": obs.get("primary_region_id"),
                    "level_delta": int(obs.get("level_delta") or 0),
                    "compass_changes": [],
                }
                prev_diffs.append(diff)
                if len(prev_diffs) > 20: prev_diffs = prev_diffs[-20:]
            rid = obs.get("primary_region_id")
            if rid and rid != "_outside_":
                recent_clicks.append(rid)
                if len(recent_clicks) > 10: recent_clicks = recent_clicks[-10:]
            if not good(e): continue
            e2 = dict(e); e2["_namespace"] = cdir.name
            fixtures.append(build_fixture(e2, prev_diffs, recent_clicks))
            if len(fixtures) >= 60: break
        if len(fixtures) >= 60: break

    random.shuffle(fixtures)
    fixtures = fixtures[:50]
    n_train = int(len(fixtures) * 0.6)
    train, val = fixtures[:n_train], fixtures[n_train:]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "n_total": len(fixtures), "n_train": len(train), "n_val": len(val),
        "train": train, "val": val,
    }, indent=2, default=str))
    print(f"Saved {len(fixtures)} fixtures (train={len(train)}, val={len(val)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
