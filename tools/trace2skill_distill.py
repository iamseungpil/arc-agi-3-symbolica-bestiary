"""B19 v591 round-8 — Trace2Skill-style distillation of cycle237 success.

Inspired by Trace2Skill (arxiv 2603.25158, March 2026): distill
trajectory-local lessons into transferable agent skills via parallel
sub-agent analysis + hierarchical inductive consolidation.

For ARC-AGI-3 ft09: cycle237 trace (39 turns, L+1@T6, L+2@T27) is the
sole successful trajectory we have. We extract:
  - The L+ event chids + coords + observation deltas
  - The 3-turn context window before each L+ event
And ask an LLM (gpt-5.5) to distill these into 2-5 reusable
"abstract mechanic" guidelines that future cycles' M1 can read from
cross_run_memory.json Section 1A.

The output is leak-safe: we strip R-ids and replace with "marker" /
"region" tokens before feeding to the LLM. Output skills must NOT
contain ft09-specific R-ids or coords; they describe abstract patterns.

Usage:
  python tools/trace2skill_distill.py \
    --trace simple_logs/ft09-9ab2447a/v57_1778180868_3399613/trace.jsonl \
    --game-dir simple_logs/ft09-9ab2447a \
    --max-skills 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path


def load_lplus_episodes(trace_path: Path, context_turns: int = 3) -> list[dict]:
    """Return list of {l_plus_turn, chid, coord, primary, prev_turns}."""
    rows = []
    for ln in trace_path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    episodes = []
    for i, r in enumerate(rows):
        obs = r.get("observation") or {}
        ld = int(obs.get("level_delta") or 0)
        if ld >= 1:
            prev = rows[max(0, i - context_turns) : i]
            episodes.append({
                "l_plus_turn": r.get("turn"),
                "chid": (r.get("action") or {}).get("chosen_hypothesis_id"),
                "coord": r.get("coord"),
                "primary": obs.get("primary_region_id"),
                "level_delta": ld,
                "prev_turns": [
                    {
                        "turn": p.get("turn"),
                        "chid": (p.get("action") or {}).get("chosen_hypothesis_id"),
                        "coord": p.get("coord"),
                        "primary": (p.get("observation") or {}).get("primary_region_id"),
                    }
                    for p in prev
                ],
            })
    return episodes


def anonymise_episode(ep: dict) -> dict:
    """Strip specific R-ids, leaving structural skeleton."""
    import re

    R = re.compile(r"R\d+")

    def anon(s):
        if not isinstance(s, str):
            return s
        return R.sub("R<id>", s)

    out = dict(ep)
    out["chid"] = anon(out.get("chid"))
    out["primary"] = anon(out.get("primary"))
    out["prev_turns"] = [
        {**p, "chid": anon(p.get("chid")), "primary": anon(p.get("primary"))}
        for p in out.get("prev_turns", [])
    ]
    return out


def distill_via_llm(episodes: list[dict], max_skills: int = 5) -> list[str]:
    """Single TRAPI gpt-5.5 call. Return list of skill text strings."""
    from agents.templates.agentica.compat import _request_text

    payload = {
        "episodes": episodes,
        "task_description": (
            "These are L+ (level rise) events from a single successful "
            "trace on an ARC-AGI-3 puzzle. Each episode shows the "
            "winning click + the 3 turns leading to it. Distill at "
            "most {n} REUSABLE abstract guidelines that an agent could "
            "follow on a fresh run. Each guideline must be 1-2 "
            "sentences, name no specific region id (they appear as "
            "R<id>), and describe a STRUCTURAL pattern (e.g. 'when "
            "stuck on a multicolor marker, click its NW corner before "
            "trying the centroid') not a specific coord."
        ).format(n=max_skills),
    }
    instructions = (
        "You are a Trace2Skill distiller. Output JSON: "
        '{"skills": ["text1", "text2", ...]}. Skills must be transferable '
        "(no R<id> or coord literals), structural, and ordered most-likely-"
        "useful first. Forbidden vocab: per_neighbor_target, target_color, "
        "needs_toggle, marker_progress, joint_neighbors, "
        "expected_neighbor_colors, win_state, goal_state, is_target_state, "
        "precision_score."
    )
    raw = _request_text(
        model="gpt-5.5",
        instructions=instructions,
        messages=[{"role": "user", "content": json.dumps(payload)[:14000]}],
        max_output_tokens=2048,
    )
    # Try to parse JSON from raw text.
    try:
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        parsed = json.loads(raw.strip())
        skills = parsed.get("skills", [])
        if not isinstance(skills, list):
            return []
        return [str(s).strip() for s in skills if s][:max_skills]
    except (json.JSONDecodeError, IndexError):
        return [raw.strip()[:200]]


def merge_into_cross_run(game_dir: Path, skills: list[str]) -> None:
    """Append skills to <game_dir>/cross_run_memory.json Section 1A."""
    memory_path = game_dir / "cross_run_memory.json"
    if memory_path.exists():
        memory = json.loads(memory_path.read_text())
    else:
        memory = {"schema_version": 1, "abstract_mechanics": []}
    existing_texts = {m.get("text", "").strip() for m in memory.get("abstract_mechanics", [])}
    for skill_text in skills:
        if skill_text.strip() in existing_texts:
            continue
        memory.setdefault("abstract_mechanics", []).append({
            "id": f"A{len(memory['abstract_mechanics']) + 1}",
            "text": skill_text,
            "confirmed_runs": 1,
            "last_seen_run": "trace2skill_cycle237",
            "refuted_runs": 0,
            "source": "Trace2Skill arxiv 2603.25158 distillation",
        })
    memory_path.write_text(json.dumps(memory, indent=2))
    print(f"merged {len(skills)} skills into {memory_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--trace", type=Path, required=True)
    p.add_argument("--game-dir", type=Path, required=True)
    p.add_argument("--max-skills", type=int, default=5)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    if not args.trace.exists():
        print(f"trace not found: {args.trace}", file=sys.stderr)
        sys.exit(2)
    episodes = load_lplus_episodes(args.trace)
    if not episodes:
        print("no L+ events in trace", file=sys.stderr)
        sys.exit(1)
    print(f"found {len(episodes)} L+ episodes")
    anonymised = [anonymise_episode(e) for e in episodes]
    skills = distill_via_llm(anonymised, max_skills=args.max_skills)
    print(f"distilled {len(skills)} skills:")
    for i, s in enumerate(skills, 1):
        print(f"  [{i}] {s}")
    if args.dry_run:
        return
    args.game_dir.mkdir(parents=True, exist_ok=True)
    merge_into_cross_run(args.game_dir, skills)


if __name__ == "__main__":
    main()
