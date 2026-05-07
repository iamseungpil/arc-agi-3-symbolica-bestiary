"""Plan v587 B14 step 9 — segment-index builder for CPSR.

Walks all simple_logs/<game_id>/cycle*/trace.jsonl files and produces
simple_logs/<game_id>/segment_index.json with anonymised
(no R-ids, no coords) segment entries. Idempotent — re-runs append new
segments.

Segment kinds (5):
  - cold_start         — first 15 turns, no L+ in window
  - pre_L+_5turns      — 5 turns immediately before a level_delta>=1
  - post_L+_recovery   — 10 turns after L+ where another L+ also fires
  - post_L+_stuck      — 10 turns after L+ with no L+ in next 30 turns
  - oscillation        — 10 turns where unique_regions/click_count < 0.5

Disambiguation: oscillation takes priority — if a window meets the
oscillation condition, it is labelled oscillation regardless of L+
proximity.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable


# ---------------------------------------------------------------------
# Bucket helpers — must match V57Board static methods byte-for-byte.
# ---------------------------------------------------------------------


def _bucket_click_count(cc: int) -> str:
    if cc <= 4:
        return "cc_le4"
    if cc <= 6:
        return "cc_5to6"
    return "cc_gt6"


def _bucket_repeat_clicks(rc: int) -> str:
    if rc <= 0:
        return "rc_0"
    if rc == 1:
        return "rc_1"
    return "rc_2plus"


def _bucket_compass_traj(traj: list[int]) -> str:
    if not traj:
        return "tr_flat"
    nz = [t for t in traj if isinstance(t, int) and t > 0]
    if not nz:
        return "tr_flat"
    mean = sum(traj) / max(1, len(traj))
    if mean < 1.0:
        return "tr_sparse"
    return "tr_active"


def _bucket_kind_distribution(kd: dict) -> str:
    if not isinstance(kd, dict):
        return "kd_unknown"
    keys = set(kd.keys())
    if keys == {"non_marker"}:
        return "kd_all_non_marker"
    if "marker_multicolor" in keys and kd.get("marker_multicolor", 0) >= 1:
        return "kd_has_marker"
    if "outside" in keys and kd.get("outside", 0) >= 1:
        return "kd_has_outside"
    return "kd_mixed"


def compute_signature(metrics: dict) -> str:
    return "|".join([
        _bucket_click_count(int(metrics.get("click_count", 0))),
        _bucket_repeat_clicks(int(metrics.get("repeat_clicks", 0))),
        _bucket_compass_traj(metrics.get("compass_change_traj") or []),
        _bucket_kind_distribution(metrics.get("kind_distribution") or {}),
    ])


# ---------------------------------------------------------------------
# Trace parsing.
# ---------------------------------------------------------------------


def parse_trace(trace_path: Path) -> list[dict]:
    out = []
    try:
        with trace_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        pass
    return out


def extract_turn_obs(trace: list[dict]) -> list[dict]:
    """Compress trace entries into per-turn structural observations
    used to build segments."""
    out = []
    for e in trace:
        if not isinstance(e, dict):
            continue
        obs = e.get("observation") or {}
        action = e.get("action") or {}
        coord = e.get("coord") or [0, 0]
        # Best-effort kind classification using available fields.
        rid = obs.get("primary_region_id") or "_outside_"
        # We don't have full visible_regions here; infer kind only when
        # explicit. Default 'non_marker' for non-outside, 'outside'
        # otherwise. (Errs on conservative side; signatures still useful
        # because the kind-distribution bucket is coarse.)
        if rid == "_outside_":
            kind = "outside"
        else:
            kind = "non_marker"
        # Compass-change count — we don't have it post-hoc; use 0
        # everywhere. The compass_change_traj bucket then collapses to
        # 'tr_flat' for older traces, which is fine — it just means
        # those segments cluster together by compass-flatness.
        out.append({
            "turn": e.get("turn"),
            "click_region_id": rid,
            "region_kind_pre": kind,
            "compass_changes": [],
            "level_delta": int(obs.get("level_delta") or 0),
        })
    return out


def metrics_for_window(window: list[dict]) -> dict:
    cc = len(window)
    region_ids = [w.get("click_region_id") for w in window if w.get("click_region_id")]
    valid = [r for r in region_ids if r and r != "_outside_"]
    unique = len(dict.fromkeys(valid))
    rc = len(valid) - unique
    kd: dict = {}
    for w in window:
        k = w.get("region_kind_pre") or "outside"
        kd[k] = kd.get(k, 0) + 1
    traj = [len(w.get("compass_changes") or []) for w in window]
    return {
        "click_count": cc,
        "repeat_clicks": rc,
        "unique_regions": unique,
        "kind_distribution": kd,
        "compass_change_traj": traj,
    }


def is_oscillation(window: list[dict]) -> bool:
    cc = len(window)
    if cc == 0:
        return False
    valid = [
        w["click_region_id"] for w in window
        if w.get("click_region_id") and w["click_region_id"] != "_outside_"
    ]
    if len(valid) < 4:
        return False
    unique = len(dict.fromkeys(valid))
    return (unique / len(valid)) < 0.5


# ---------------------------------------------------------------------
# Segment extraction per trace.
# ---------------------------------------------------------------------


def extract_segments(turns: list[dict], run_namespace: str) -> list[dict]:
    """Walk the turns list and emit segments of the 5 kinds."""
    segments: list[dict] = []
    if not turns:
        return segments
    n = len(turns)
    lp_indices = [i for i, t in enumerate(turns) if t["level_delta"] >= 1]

    # Pre-L+ 5-turn windows.
    for lp_i in lp_indices:
        start = max(0, lp_i - 4)
        window = turns[start: lp_i + 1]
        if is_oscillation(window):
            kind = "oscillation"
        else:
            kind = "pre_L+_5turns"
        m = metrics_for_window(window)
        sig = compute_signature(m)
        next5 = turns[lp_i + 1: lp_i + 6]
        next5_progress = any(t["level_delta"] >= 1 for t in next5)
        what_changed = (
            f"in next 5 turns: level_progress={next5_progress}, "
            f"clicks={len(next5)}, "
            f"new_kind_dist={metrics_for_window(next5)['kind_distribution']}"
        )
        segments.append({
            "run_namespace": run_namespace,
            "turn_range": [turns[start]["turn"], turns[lp_i]["turn"]],
            "kind": kind,
            "abstract_signature": sig,
            "did_progress": True,  # this segment ENDED with L+
            "what_changed_in_next_5turns": what_changed,
        })

    # Post-L+ recovery vs stuck windows.
    for lp_i in lp_indices:
        post_start = lp_i + 1
        post_end = min(n, post_start + 10)
        if post_start >= n:
            continue
        window = turns[post_start: post_end]
        if is_oscillation(window):
            kind = "oscillation"
            did_progress = False
        else:
            # Look ahead 30 turns from L+ for next L+.
            next_lp = next(
                (j for j in lp_indices if j > lp_i and j <= lp_i + 30), None
            )
            if next_lp is not None:
                kind = "post_L+_recovery"
                did_progress = True
            else:
                kind = "post_L+_stuck"
                did_progress = False
        m = metrics_for_window(window)
        sig = compute_signature(m)
        next5 = turns[post_end: post_end + 5]
        next5_progress = any(t["level_delta"] >= 1 for t in next5)
        what_changed = (
            f"in next 5 turns: level_progress={next5_progress}, "
            f"clicks={len(next5)}, "
            f"new_kind_dist={metrics_for_window(next5)['kind_distribution']}"
        )
        segments.append({
            "run_namespace": run_namespace,
            "turn_range": [turns[post_start]["turn"], turns[post_end - 1]["turn"]],
            "kind": kind,
            "abstract_signature": sig,
            "did_progress": did_progress,
            "what_changed_in_next_5turns": what_changed,
        })

    # Cold-start segment (first 15 turns if no L+ in that window).
    cold_end = min(15, n)
    if cold_end >= 5:
        cold_window = turns[:cold_end]
        if not any(t["level_delta"] >= 1 for t in cold_window):
            m = metrics_for_window(cold_window)
            sig = compute_signature(m)
            after = turns[cold_end: cold_end + 5]
            after_progress = any(t["level_delta"] >= 1 for t in after)
            what_changed = (
                f"in next 5 turns: level_progress={after_progress}, "
                f"clicks={len(after)}, "
                f"new_kind_dist={metrics_for_window(after)['kind_distribution']}"
            )
            kind = "oscillation" if is_oscillation(cold_window) else "cold_start"
            segments.append({
                "run_namespace": run_namespace,
                "turn_range": [
                    cold_window[0]["turn"], cold_window[-1]["turn"]
                ],
                "kind": kind,
                "abstract_signature": sig,
                "did_progress": after_progress,
                "what_changed_in_next_5turns": what_changed,
            })

    return segments


# ---------------------------------------------------------------------
# Main — walk simple_logs, write segment_index.json per game.
# ---------------------------------------------------------------------


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    sl = root / "simple_logs"
    games = [p for p in sl.glob("*-*") if p.is_dir()]
    if not games:
        print("No game dirs under simple_logs.", file=sys.stderr)
        return 1
    total_segments = 0
    for g in games:
        out_path = g / "segment_index.json"
        # Idempotent: load existing index, skip segments already
        # indexed (keyed by run_namespace + turn_range).
        existing: list[dict] = []
        seen_keys: set[tuple] = set()
        if out_path.exists():
            try:
                data = json.loads(out_path.read_text())
                if isinstance(data, dict):
                    existing = data.get("segments", []) or []
                elif isinstance(data, list):
                    existing = data
            except Exception:
                existing = []
            for s in existing:
                seen_keys.add((s.get("run_namespace"), tuple(s.get("turn_range") or [])))
        new_segments: list[dict] = []
        for cycle_dir in g.glob("cycle*"):
            tp = cycle_dir / "trace.jsonl"
            if not tp.exists():
                continue
            ns = cycle_dir.name
            trace = parse_trace(tp)
            turns = extract_turn_obs(trace)
            segs = extract_segments(turns, run_namespace=ns)
            for s in segs:
                key = (s.get("run_namespace"), tuple(s.get("turn_range") or []))
                if key not in seen_keys:
                    seen_keys.add(key)
                    new_segments.append(s)
        all_segments = existing + new_segments
        out = {"schema_version": 1, "segments": all_segments}
        out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
        kind_counts: dict = {}
        progress_counts = {True: 0, False: 0}
        for s in all_segments:
            kind_counts[s.get("kind")] = kind_counts.get(s.get("kind"), 0) + 1
            progress_counts[bool(s.get("did_progress"))] = (
                progress_counts[bool(s.get("did_progress"))] + 1
            )
        print(
            f"[{g.name}] +{len(new_segments)} new segments, "
            f"total={len(all_segments)} kinds={kind_counts} "
            f"progress={dict(progress_counts)}"
        )
        total_segments += len(new_segments)
    print(f"\nTotal new segments across games: {total_segments}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
