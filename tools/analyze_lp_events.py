"""Plan v587 step 1 — empirical L+ event bucket / K_STUCK analysis.

Walks all simple_logs/<game>/cycle*/trace.jsonl, extracts L+ events with
their preceding 6-turn window, computes click_count + repeat_clicks +
inter-L+ gap distributions. Output: JSON to reports/plan_v587_buckets.json
that the agent code reads at startup."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from collections import Counter
from statistics import median


def parse_trace(trace_path: Path) -> list[dict]:
    entries = []
    try:
        with trace_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entries.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        pass
    return entries


def extract_lp_event_metrics(trace: list[dict]) -> list[dict]:
    """Find each L+ event in the trace and compute its preceding-6-turn
    metrics. Returns list of {turn, click_count_in_window, repeat_clicks,
    unique_regions, gap_since_prev_lp}."""
    out = []
    last_lp_turn = None
    # Build sequence of (turn, region_id, level_delta).
    seq = []
    for e in trace:
        obs = e.get("observation") or {}
        if not isinstance(obs, dict):
            continue
        seq.append({
            "turn": e.get("turn"),
            "region_id": obs.get("primary_region_id"),
            "level_delta": int(obs.get("level_delta") or 0),
        })
    for i, ev in enumerate(seq):
        if ev["level_delta"] >= 1:
            window = seq[max(0, i - 5): i + 1]  # last 6 entries incl. trigger
            valid = [w for w in window if w["region_id"] and w["region_id"] != "_outside_"]
            unique = len({w["region_id"] for w in valid})
            click_count = len(window)
            repeat_clicks = len(valid) - unique
            gap = (ev["turn"] - last_lp_turn) if last_lp_turn is not None else None
            out.append({
                "turn": ev["turn"],
                "click_count_in_window": click_count,
                "repeat_clicks": repeat_clicks,
                "unique_regions": unique,
                "gap_since_prev_lp": gap,
            })
            last_lp_turn = ev["turn"]
    return out


def quartile_buckets(values: list[int], n_buckets: int = 3) -> list[int]:
    """Return n_buckets-1 cut points (so n_buckets disjoint ranges)."""
    if not values:
        return []
    s = sorted(values)
    cuts = []
    for k in range(1, n_buckets):
        idx = int(len(s) * k / n_buckets)
        cuts.append(s[min(idx, len(s) - 1)])
    return cuts


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    sl = root / "simple_logs"
    games = [p for p in sl.glob("*-*") if p.is_dir()]
    all_events: list[dict] = []
    cycles_seen = 0
    for g in games:
        for cycle_dir in g.glob("cycle*"):
            tp = cycle_dir / "trace.jsonl"
            if not tp.exists():
                continue
            cycles_seen += 1
            trace = parse_trace(tp)
            all_events.extend(extract_lp_event_metrics(trace))
    if not all_events:
        print("No L+ events found across all traces.", file=sys.stderr)
        return 1
    cc_vals = [e["click_count_in_window"] for e in all_events]
    rc_vals = [e["repeat_clicks"] for e in all_events]
    gap_vals = [e["gap_since_prev_lp"] for e in all_events if e["gap_since_prev_lp"] is not None]
    cc_cuts = quartile_buckets(cc_vals, n_buckets=3)
    rc_cuts = quartile_buckets(rc_vals, n_buckets=3)
    gap_med = median(gap_vals) if gap_vals else None
    out = {
        "n_lp_events": len(all_events),
        "n_cycles_scanned": cycles_seen,
        "click_count_distribution": dict(sorted(Counter(cc_vals).items())),
        "repeat_clicks_distribution": dict(sorted(Counter(rc_vals).items())),
        "inter_lp_gap_median": gap_med,
        "click_count_cut_points": cc_cuts,        # bucketing thresholds
        "repeat_clicks_cut_points": rc_cuts,
        "K_STUCK": int(2 * gap_med) if gap_med else 10,
        "click_count_buckets": (
            [f"<={cc_cuts[0]}", f"{cc_cuts[0]+1}-{cc_cuts[1]}", f">{cc_cuts[1]}"]
            if len(cc_cuts) == 2 else None
        ),
        "repeat_clicks_buckets": (
            [f"<={rc_cuts[0]}", f"{rc_cuts[0]+1}-{rc_cuts[1]}", f">{rc_cuts[1]}"]
            if len(rc_cuts) == 2 else None
        ),
    }
    out_path = root / "reports" / "plan_v587_buckets.json"
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=2))
    print(f"\nWritten: {out_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
