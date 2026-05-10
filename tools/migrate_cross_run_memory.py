"""v602 one-time importer: cross_run_memory.json -> skill_state.json.

Plan v602 §7 4-step migration:
  1. introduce skill_state.json (NEW)
  2. one-time importer: if skill_state.json missing AND cross_run_memory.json
     exists, import 1A entries -> skill_state.confirmed_mechanics
  3. render SKILL.md from skill_state.json after import
  4. mark cross_run_memory.json deprecated

Usage:
  python -m tools.migrate_cross_run_memory --src path/to/cross_run_memory.json \
      --dest path/to/skill_state.json [--render-skill-md path/to/SKILL.md]

Idempotency: persists `metadata.cross_run_imported = True` so re-running on a
state that already imported is a no-op (returns early).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from agents.templates.agentica_lite.skill_md_renderer import render
from agents.templates.agentica_lite.skill_state import (
    ConfirmedMechanic, SkillState, SkillStateMetadata, atomic_write_text,
    load_state, save_state,
)


def import_cross_run_memory(src: dict[str, Any], game_id: str = "") -> list[ConfirmedMechanic]:
    """Convert cross_run_memory.json shape into ConfirmedMechanic records.

    Accepted source shapes:
      {"confirmed_mechanics": [{...}, ...]}  (canonical)
      {"1A": [{...}, ...]}                    (legacy alias)
    """
    raw_items = src.get("confirmed_mechanics") or src.get("1A") or []
    out: list[ConfirmedMechanic] = []
    for i, raw in enumerate(raw_items):
        cm = ConfirmedMechanic(
            id=raw.get("id") or f"A{i+1:03d}",
            natural_language=raw.get("natural_language", "") or raw.get("text", ""),
            evidence_runs=list(raw.get("evidence_runs") or []),
            first_seen_run=raw.get("first_seen_run") or raw.get("first_seen", ""),
            first_seen_turn=int(raw.get("first_seen_turn") or -1),
            confirmation_count=int(raw.get("confirmation_count") or 1),
            paired_cf_offsets=list(raw.get("paired_cf_offsets") or []),
        )
        out.append(cm)
    return out


def migrate(src_path: Path, dest_path: Path, game_id: str = "",
            render_skill_md_to: Path | None = None) -> dict[str, Any]:
    """Run the 4-step migration.

    Returns a small summary dict for logging.
    """
    summary: dict[str, Any] = {"status": "noop", "imported": 0,
                                "src_exists": src_path.exists(),
                                "dest_exists": dest_path.exists()}

    # Idempotent path: if dest exists AND already flagged imported, skip.
    existing = load_state(dest_path)
    if existing is not None and existing.metadata.cross_run_imported:
        summary["status"] = "already_imported"
        return summary

    if not src_path.exists():
        # No source to import from; if dest is also missing, create empty state
        if existing is None:
            empty = SkillState(metadata=SkillStateMetadata(
                game_id=game_id, cross_run_imported=False,
            ))
            save_state(empty, dest_path)
            summary["status"] = "empty_state_initialized"
        return summary

    src_data = json.loads(src_path.read_text(encoding="utf-8"))
    imported = import_cross_run_memory(src_data, game_id)

    # build state
    state = existing or SkillState(metadata=SkillStateMetadata(game_id=game_id))
    # avoid double-import: collisions by id
    existing_ids = {c.id for c in state.confirmed_mechanics}
    for cm in imported:
        if cm.id not in existing_ids:
            state.confirmed_mechanics.append(cm)

    state.metadata.cross_run_imported = True
    save_state(state, dest_path)

    if render_skill_md_to is not None:
        atomic_write_text(render_skill_md_to, render(state))

    summary["status"] = "imported"
    summary["imported"] = len(imported)
    return summary


def _main() -> None:
    ap = argparse.ArgumentParser(description="v602 cross_run_memory.json -> skill_state.json")
    ap.add_argument("--src", required=True, type=Path)
    ap.add_argument("--dest", required=True, type=Path)
    ap.add_argument("--game-id", default="")
    ap.add_argument("--render-skill-md", default=None, type=Path)
    args = ap.parse_args()
    summary = migrate(args.src, args.dest, args.game_id, args.render_skill_md)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _main()
