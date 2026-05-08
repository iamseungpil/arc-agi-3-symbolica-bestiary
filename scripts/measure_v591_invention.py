"""B19 v591 — V-H4 / V-H5 post-cycle metric extraction.

Reads a v57 trace.jsonl and reports:
  - total_turns
  - tier_distribution: {A, B, card, none}
  - non_trivial_invention_rate (V-H4 ≥ 0.30)
  - tier_b_template_supported (V-H5 ≥ 1 template with supported>=3)
  - in_region_rate (V-H3 ≥ 0.85)
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_turns(trace_path: Path):
    rows = []
    for ln in trace_path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    return rows


def measure(trace_path: Path) -> dict:
    rows = load_turns(trace_path)
    n = len(rows)
    tiers = Counter()
    non_trivial = 0
    tier_b_total = 0
    template_supported = defaultdict(lambda: {"supported": 0, "refuted": 0})
    in_region = 0
    in_region_denominator = 0
    for r in rows:
        tier = r.get("chid_tier") or "unknown"
        tiers[tier] += 1
        meta = r.get("invented_meta")
        if tier == "B" and isinstance(meta, dict):
            tier_b_total += 1
            if meta.get("non_trivial"):
                non_trivial += 1
            tpl = meta.get("template_id")
            obs = r.get("observation") or {}
            ld = int(obs.get("level_delta") or 0)
            if tpl:
                if ld >= 1:
                    template_supported[tpl]["supported"] += 1
                else:
                    template_supported[tpl]["refuted"] += 1
        # V-H3 in-region rate: coord ∈ ⋃ visible_regions.bbox.
        cx, cy = (r.get("coord") or [None, None])[:2]
        if cx is None:
            continue
        in_region_denominator += 1
        for vr in (r.get("visible_regions") or []):
            bb = vr.get("bbox") or []
            if isinstance(bb, dict):
                bb = [bb.get("min_x"), bb.get("min_y"), bb.get("max_x"), bb.get("max_y")]
            if not (isinstance(bb, list) and len(bb) >= 4):
                continue
            try:
                if bb[0] <= cx <= bb[2] and bb[1] <= cy <= bb[3]:
                    in_region += 1
                    break
            except TypeError:
                continue
    rate_non_trivial = non_trivial / max(tier_b_total, 1)
    rate_in_region = in_region / max(in_region_denominator, 1)
    promoted = [
        tpl for tpl, stats in template_supported.items()
        if stats["supported"] >= 3 and (stats["supported"] - stats["refuted"]) >= 2
    ]
    return {
        "trace_path": str(trace_path),
        "total_turns": n,
        "tier_distribution": dict(tiers),
        "tier_b_total": tier_b_total,
        "non_trivial_invention_rate": round(rate_non_trivial, 3),
        "v_h4_pass": rate_non_trivial >= 0.30,
        "in_region_rate": round(rate_in_region, 3),
        "v_h3_pass": rate_in_region >= 0.85,
        "tier_b_promotable_templates": promoted,
        "v_h5_pass": len(promoted) >= 1,
        "template_supported_counts": dict(template_supported),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("trace_path", type=Path)
    p.add_argument("--json", action="store_true", help="emit JSON")
    args = p.parse_args()
    if not args.trace_path.exists():
        print(f"trace not found: {args.trace_path}", file=sys.stderr)
        sys.exit(2)
    report = measure(args.trace_path)
    if args.json:
        print(json.dumps(report, indent=2))
    else:
        for k, v in report.items():
            print(f"{k}: {v}")
    sys.exit(0 if report["v_h4_pass"] else 1)


if __name__ == "__main__":
    main()
