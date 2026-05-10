"""Extract v600 fixtures from cycle traces (plan §5).

Three pools:
  - synthetic: hand-crafted F1-F6 failure modes (≥30)
  - replay: real turn snapshots from failed cycles (≥20)
  - blind held-out: sealed seeds, never used in tuning (≥20, populated separately)

Usage:
  python tools/extract_v600_fixtures.py \\
    --trace simple_logs/ft09-9ab2447a/v57_1778180868_3399613/trace.jsonl \\
    --turn 27 \\
    --failure-mode F0_gold_L_plus_2 \\
    --module predicate_posterior \\
    --behavior 'select cycle237-style sector_alignment predicate' \\
    --assertion 'output[0][0].startswith("P") or "sector_alignment" in str(output)' \\
    --pool replay \\
    --split val \\
    --out tests/v600/fixtures/replay/cycle237_T27_gold.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def extract_turn_snapshot(trace_path: Path, turn_idx: int) -> dict:
    """Read trace.jsonl, return the turn record at index turn_idx.

    We snapshot input fields agentica_lite needs:
      - visible_regions
      - visible_region_ids
      - active_hypotheses_after (used as "active before next turn")
      - candidate_tests_for_m1 (predicate library candidates exposed at turn entry)
      - matched_prior_triggers
      - last observation diff and verdict
    """
    with open(trace_path) as f:
        for i, line in enumerate(f):
            if i == turn_idx:
                d = json.loads(line)
                return {
                    "turn": d.get("turn", i),
                    "visible_regions": d.get("visible_regions", []),
                    "visible_region_ids": d.get("visible_region_ids", []),
                    "active_hypotheses": d.get("active_hypotheses_after", []),
                    "candidate_tests": d.get("candidate_tests_for_m1", []),
                    "matched_prior_triggers": d.get("matched_prior_triggers", []),
                    "stuck_mode": d.get("stuck_mode"),
                    "stuck_severity": d.get("stuck_severity"),
                    "marker_neighbor_states": d.get("marker_neighbor_states"),
                    "summary": d.get("summary_after"),
                    "last_observation": (
                        d.get("observation", {}) if i > 0 else {}
                    ),
                    "thought": (d.get("action") or {}).get("thought", "")[:1000],
                    "coord_taken": d.get("coord"),
                    "snapped_coord": d.get("snapped_coord"),
                    "verdict": d.get("verdict"),
                }
    raise IndexError(f"turn {turn_idx} not found in {trace_path}")


def build_fixture(
    fixture_id: str,
    failure_mode: str,
    pool: str,
    source_trace: str | None,
    source_turn: int | None,
    input_dict: dict,
    expected_module: str,
    expected_behavior: str,
    expected_assertion: str,
    split: str,
) -> dict:
    return {
        "id": fixture_id,
        "failure_mode": failure_mode,
        "pool": pool,
        "source_trace": source_trace,
        "source_turn": source_turn,
        "input": input_dict,
        "expected_module": expected_module,
        "expected_behavior": expected_behavior,
        "expected_assertion": expected_assertion,
        "split": split,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--trace", type=Path, default=None,
                    help="trace.jsonl (omit for synthetic fixtures)")
    ap.add_argument("--turn", type=int, default=None)
    ap.add_argument("--failure-mode", required=True,
                    help="F0_gold_L_plus_2 | F1 | F2 | F3 | F4 | F5 | F6")
    ap.add_argument("--module", required=True,
                    help="predicate_posterior | stalemate_trigger | llm_extender | "
                         "library_install | memory_journal | region_coord_resolver")
    ap.add_argument("--behavior", required=True)
    ap.add_argument("--assertion", required=True,
                    help="Python expression eval'd against module output, e.g. "
                         "'output[0][0] == \"P03\"'")
    ap.add_argument("--pool", choices=["synthetic", "replay", "held_out"],
                    required=True)
    ap.add_argument("--split", choices=["train", "val", "blind"], required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--id", default=None,
                    help="Fixture id override; default auto-generated")
    args = ap.parse_args()

    if args.trace and args.turn is not None:
        input_dict = extract_turn_snapshot(args.trace, args.turn)
        source_trace = str(args.trace)
        source_turn = args.turn
        if args.id is None:
            cycle_tag = args.trace.parent.name
            args.id = f"{args.failure_mode}-{cycle_tag}-T{args.turn}"
    else:
        # Synthetic — read input dict from stdin or use empty
        input_dict = json.loads(sys.stdin.read()) if not sys.stdin.isatty() else {}
        source_trace = None
        source_turn = None
        if args.id is None:
            args.id = f"{args.failure_mode}-synthetic-{args.module}"

    fixture = build_fixture(
        fixture_id=args.id,
        failure_mode=args.failure_mode,
        pool=args.pool,
        source_trace=source_trace,
        source_turn=source_turn,
        input_dict=input_dict,
        expected_module=args.module,
        expected_behavior=args.behavior,
        expected_assertion=args.assertion,
        split=args.split,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(fixture, f, indent=2, default=str)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
