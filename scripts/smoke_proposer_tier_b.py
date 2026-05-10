"""Real-LLM smoke: 1 Proposer call against cycle237 T5 state, check TIER-B emission.

Per codex round-8 sign-off: G-ii static checks are necessary but not sufficient
as the live-autoresearch gate. This adds a single LLM call to verify the
patched prompt actually changes model behavior.

Success criterion (codex Q2):
  ≥1 emission with candidate_predicate_id matching `P_R\d+_.*`
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
from pathlib import Path

# Ensure repo on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.templates.agentica_lite.proposer import Proposer  # noqa: E402


CYCLE237_TRACE = Path("simple_logs/ft09-9ab2447a/v57_1778180868_3399613/trace.jsonl")
TIER_B_RE = re.compile(r"^P_R\d+_.*")


def load_cycle237_state(turn: int) -> dict:
    with CYCLE237_TRACE.open() as f:
        for line in f:
            row = json.loads(line)
            if row.get("turn") == turn:
                return {
                    "visible_regions": row.get("visible_regions") or [],
                    "marker_neighbor_states": row.get("marker_neighbor_states") or [],
                    "observation": {
                        "dominant_transition": None, "level_delta": 0,
                        "primary_region_id": (row.get("visible_regions") or [{}])[0].get("region_id"),
                    },
                }
    raise AssertionError(f"cycle237 trace missing turn={turn}")


async def main(n_calls: int = 3) -> int:
    state = load_cycle237_state(5)
    vis_ids = [
        (r.get("region_id") or r.get("id")) for r in state["visible_regions"] if r
    ]
    print(f"cycle237 T5 visible ids: {vis_ids[:15]}{'...' if len(vis_ids)>15 else ''}")
    print(f"markers: {[m.get('marker_id') for m in state['marker_neighbor_states']]}")
    print(f"running {n_calls} Proposer calls (gpt-5.4-mini)...")

    proposer = Proposer(llm_timeout_s=45.0)
    emissions: list[str] = []
    raw_outputs: list[str] = []
    for i in range(n_calls):
        try:
            result = await proposer.propose(state, vis_ids)
        except Exception as e:  # noqa: BLE001
            print(f"call[{i}] error: {e}")
            continue
        if result.output is None:
            print(f"call[{i}] failure_reason={result.failure_reason} schema_err={result.schema_error_code}")
        else:
            chid = result.output.candidate_predicate_id
            emissions.append(chid)
            print(f"call[{i}] chid={chid!r} region_hint={result.output.region_hint!r}")

    tier_b = [c for c in emissions if TIER_B_RE.match(c)]
    distinct = set(emissions)
    print()
    print(f"==> emissions {len(emissions)}/{n_calls}, distinct={len(distinct)}, TIER-B={len(tier_b)}")
    print(f"==> all chids: {emissions}")
    print(f"==> TIER-B: {tier_b}")

    if tier_b:
        print("\nSMOKE PASS: real LLM emitted ≥1 TIER-B chid")
        return 0
    print("\nSMOKE FAIL: 0 TIER-B chids in real LLM output")
    return 1


if __name__ == "__main__":
    n = int(os.environ.get("SMOKE_N_CALLS", "3"))
    rc = asyncio.run(main(n))
    sys.exit(rc)
