#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import os
import re
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PRESENTATION_DIR = REPO_ROOT / "presentation"
REPORTS_DIR = REPO_ROOT / "reports"
GAME_ID = "ft09-9ab2447a"
GAME_PREFIX = GAME_ID.split("-")[0]
MAX_RUNS_DEFAULT = 5

PALETTE = [
    "#000000",
    "#0074D9",
    "#FF4136",
    "#2ECC40",
    "#FFDC00",
    "#AAAAAA",
    "#F012BE",
    "#FF851B",
    "#7FDBFF",
    "#870C25",
    "#555555",
    "#B10DC9",
    "#001F3F",
    "#7D4E4E",
    "#4E7D4E",
    "#4E4E7D",
]

VIEWER_HTML = PRESENTATION_DIR / "ft09_trace_explorer.html"
VIEWER_DATA_JSON = PRESENTATION_DIR / "ft09_trace_explorer.data.json"
FLOW_HTML = PRESENTATION_DIR / "ft09_runtime_flow_v5.html"
FLOW_SVG = PRESENTATION_DIR / "ft09_runtime_flow_v5.svg"
FLOW_PDF = REPORTS_DIR / "v5_runtime_flow_report.pdf"
REVIEW_NOTE = REPORTS_DIR / "ft09_visualization_review.md"

ACTION_LINE_RE = re.compile(
    rf"{re.escape(GAME_ID)} - (?P<action>[A-Z0-9]+): count (?P<count>\d+), level (?P<level>\d+)/(?P<win>\d+)"
)
REFLEXION_RE = re.compile(
    r"v41 reflexion set \(turn=(?P<turn>\d+), stag=(?P<stag>\d+)\): (?P<text>.*)"
)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def repo_rel(path: Path | None) -> str | None:
    if path is None:
        return None
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def ts_to_text(ts: float | None) -> str:
    if ts is None:
        return "unknown"
    return datetime.fromtimestamp(ts, tz=UTC).strftime("%Y-%m-%d %H:%M UTC")


def html_text(value: Any) -> str:
    return html.escape("" if value is None else str(value))


def shorten(text: str | None, limit: int = 180) -> str:
    if not text:
        return ""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= limit:
        return text
    return text[: limit - 1].rstrip() + "..."


def unwrap_grid(value: Any) -> list[list[int]]:
    if not isinstance(value, list):
        return []
    if value and isinstance(value[0], list) and value[0] and isinstance(value[0][0], list):
        return [
            [int(cell) for cell in row]
            for row in value[0]
            if isinstance(row, list)
        ]
    return [
        [int(cell) for cell in row]
        for row in value
        if isinstance(row, list)
    ]


def encode_grid(grid: list[list[int]]) -> list[str]:
    return ["".join(format(int(cell) & 0xF, "x") for cell in row) for row in grid]


def decode_grid(encoded: list[str]) -> list[list[int]]:
    return [[int(ch, 16) for ch in row] for row in encoded]


def parse_action_input(action_input: dict[str, Any] | None) -> dict[str, Any]:
    action_input = action_input or {}
    action_id = action_input.get("id")
    data = action_input.get("data") or {}
    if action_id == 0:
        return {"label": "RESET", "name": "RESET", "x": None, "y": None}
    if action_id is None:
        return {"label": "UNKNOWN", "name": "UNKNOWN", "x": None, "y": None}
    name = f"ACTION{int(action_id)}"
    x = data.get("x")
    y = data.get("y")
    if int(action_id) == 6 and x is not None and y is not None:
        label = f"{name}({x},{y})"
    else:
        label = name
    return {"label": label, "name": name, "x": x, "y": y}


def diff_signature(before: list[list[int]], after: list[list[int]]) -> dict[str, Any]:
    if not before or not after:
        return {
            "changed_cells": 0,
            "dominant_transition": None,
            "bbox": None,
        }
    rows = min(len(before), len(after))
    cols = min(len(before[0]), len(after[0]))
    transitions: Counter[tuple[int, int]] = Counter()
    changed = 0
    min_x = cols
    min_y = rows
    max_x = -1
    max_y = -1
    for y in range(rows):
        for x in range(cols):
            prev = int(before[y][x])
            nxt = int(after[y][x])
            if prev == nxt:
                continue
            changed += 1
            transitions[(prev, nxt)] += 1
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
    dominant = None
    if transitions:
        ((from_color, to_color), count) = max(
            transitions.items(),
            key=lambda item: (item[1], -item[0][0], -item[0][1]),
        )
        dominant = {"from": from_color, "to": to_color, "count": count}
    bbox = None
    if changed:
        bbox = [min_x, min_y, max_x, max_y]
    return {
        "changed_cells": changed,
        "dominant_transition": dominant,
        "bbox": bbox,
    }


def grid_svg(
    grid: list[list[int]],
    *,
    max_side: int = 340,
    highlight: dict[str, Any] | None = None,
) -> str:
    if not grid or not grid[0]:
        return '<div class="empty">no grid</div>'
    rows = len(grid)
    cols = len(grid[0])
    cell = max(3, max_side // max(rows, cols))
    width = cols * cell
    height = rows * cell
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" class="grid-svg" aria-hidden="true">',
        f'<rect width="{width}" height="{height}" rx="18" fill="#13252B"/>',
    ]
    for y, row in enumerate(grid):
        run_start = 0
        current = int(row[0])
        for x in range(1, cols + 1):
            value = int(row[x]) if x < cols else None
            if x == cols or value != current:
                parts.append(
                    f'<rect x="{run_start * cell}" y="{y * cell}" width="{(x - run_start) * cell}" height="{cell}" fill="{PALETTE[current % 16]}"/>'
                )
                if x < cols:
                    run_start = x
                    current = value
    parts.append(
        f'<rect width="{width}" height="{height}" rx="18" fill="none" stroke="#D8E3E1" stroke-width="2"/>'
    )
    if highlight and highlight.get("x") is not None and highlight.get("y") is not None:
        cx = (int(highlight["x"]) + 0.5) * cell
        cy = (int(highlight["y"]) + 0.5) * cell
        radius = max(4, cell * 1.5)
        parts.append(
            f'<circle cx="{cx}" cy="{cy}" r="{radius}" class="click-ring" fill="none" stroke="#F4A341" stroke-width="{max(2, cell / 2.6)}"/>'
        )
        parts.append(
            f'<circle cx="{cx}" cy="{cy}" r="{max(2, cell / 2.4)}" fill="#F4A341" opacity="0.86"/>'
        )
    parts.append("</svg>")
    return "".join(parts)


def diff_svg(
    before: list[list[int]],
    after: list[list[int]],
    *,
    max_side: int = 340,
    highlight: dict[str, Any] | None = None,
) -> str:
    if not before or not after:
        return '<div class="empty">no diff</div>'
    rows = min(len(before), len(after))
    cols = min(len(before[0]), len(after[0]))
    cell = max(3, max_side // max(rows, cols))
    width = cols * cell
    height = rows * cell
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" class="grid-svg" aria-hidden="true">',
        f'<rect width="{width}" height="{height}" rx="18" fill="#122028"/>',
    ]
    for y in range(rows):
        for x in range(cols):
            prev = int(before[y][x])
            nxt = int(after[y][x])
            if prev == nxt:
                continue
            parts.append(
                f'<rect x="{x * cell}" y="{y * cell}" width="{cell}" height="{cell}" fill="{PALETTE[nxt % 16]}"/>'
            )
            parts.append(
                f'<rect x="{x * cell}" y="{y * cell}" width="{cell}" height="{cell}" fill="none" stroke="#F4A341" stroke-width="{max(1, cell / 4)}"/>'
            )
    parts.append(
        f'<rect width="{width}" height="{height}" rx="18" fill="none" stroke="#D8E3E1" stroke-width="2"/>'
    )
    if highlight and highlight.get("x") is not None and highlight.get("y") is not None:
        cx = (int(highlight["x"]) + 0.5) * cell
        cy = (int(highlight["y"]) + 0.5) * cell
        radius = max(4, cell * 1.5)
        parts.append(
            f'<circle cx="{cx}" cy="{cy}" r="{radius}" class="click-ring" fill="none" stroke="#9BD1D4" stroke-width="{max(2, cell / 2.6)}"/>'
        )
    parts.append("</svg>")
    return "".join(parts)


def find_offset(haystack: list[str], needle: list[str], *, prefix: int = 8) -> int | None:
    if not needle:
        return 0
    probe = needle[: min(prefix, len(needle))]
    upper = len(haystack) - len(needle)
    for start in range(max(0, upper + 1)):
        if haystack[start : start + len(probe)] != probe:
            continue
        if haystack[start : start + len(needle)] == needle:
            return start
    return None


def parse_recording(recording_path: Path) -> list[dict[str, Any]]:
    rows = read_jsonl(recording_path)
    steps: list[dict[str, Any]] = []
    prev_grid: list[list[int]] | None = None
    prev_levels = 0
    for index, row in enumerate(rows):
        data = row.get("data") or {}
        action = parse_action_input(data.get("action_input"))
        after_grid = unwrap_grid(data.get("frame"))
        if prev_grid is None:
            before_grid = after_grid
        else:
            before_grid = prev_grid
        after_levels = int(data.get("levels_completed", 0) or 0)
        steps.append(
            {
                "recording_index": index,
                "timestamp": row.get("timestamp"),
                "action": action,
                "before_grid": before_grid,
                "after_grid": after_grid,
                "levels_before": prev_levels,
                "levels_after": after_levels,
                "state_after": data.get("state"),
                "full_reset_after": bool(data.get("full_reset", False)),
            }
        )
        prev_grid = after_grid
        prev_levels = after_levels
    return steps


def parse_goal_board(goal_board_path: Path) -> dict[str, Any]:
    raw = read_json(goal_board_path)
    choices = list(raw.get("choice_history", []) or [])
    observations = list(raw.get("observation_log", []) or [])
    skills = list(raw.get("skills", []) or [])
    cross = list(raw.get("cross_level_confirmed", []) or [])
    return {
        "raw": raw,
        "choices": choices,
        "observations": observations,
        "skills": skills,
        "cross_level_confirmed": cross,
    }


def parse_run_log(log_path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {"reflexion_events": [], "action_counts": []}
    if not log_path.exists():
        return result
    with log_path.open(encoding="utf-8", errors="replace") as handle:
        for line in handle:
            line = line.rstrip()
            reflex_match = REFLEXION_RE.search(line)
            if reflex_match:
                result["reflexion_events"].append(
                    {
                        "turn_after": int(reflex_match.group("turn")),
                        "stagnation_window": int(reflex_match.group("stag")),
                        "text": reflex_match.group("text").strip(),
                        "line": line,
                    }
                )
                continue
            action_match = ACTION_LINE_RE.search(line)
            if action_match:
                result["action_counts"].append(
                    {
                        "action_name": action_match.group("action"),
                        "count": int(action_match.group("count")),
                        "level": int(action_match.group("level")),
                        "win_levels": int(action_match.group("win")),
                    }
                )
    return result


def active_namespace() -> str | None:
    try:
        proc_root = Path("/proc")
        for proc_dir in proc_root.iterdir():
            if not proc_dir.name.isdigit():
                continue
            try:
                cmdline = (proc_dir / "cmdline").read_text(encoding="utf-8", errors="ignore")
            except OSError:
                continue
            if "python3" not in cmdline or "main.py" not in cmdline or GAME_ID not in cmdline:
                continue
            try:
                env = (proc_dir / "environ").read_bytes().decode("utf-8", errors="ignore").split("\x00")
            except OSError:
                continue
            for item in env:
                if item.startswith("ARC_SIMPLE_NAMESPACE="):
                    return item.split("=", 1)[1]
    except OSError:
        return None
    return None


def choose_recording(namespace: str, run_dir: Path, log_path: Path) -> Path | None:
    recording_dir = REPO_ROOT / "recordings"
    if not recording_dir.exists():
        return None
    candidates = list(recording_dir.glob(f"{GAME_ID}.arcgenticasimple.*.recording.jsonl"))
    if not candidates:
        return None
    target_mtime = log_path.stat().st_mtime if log_path.exists() else run_dir.stat().st_mtime
    return min(
        candidates,
        key=lambda path: (abs(path.stat().st_mtime - target_mtime), path.name),
    )


def version_rank(namespace: str) -> int:
    match = re.match(r"^v(?P<num>\d+)", namespace)
    if not match:
        return -1
    return int(match.group("num"))


def score_bundle(bundle: dict[str, Any]) -> tuple[float, ...]:
    stats = bundle["stats"]
    active_bonus = 1.0 if bundle["id"] == bundle.get("active_namespace") else 0.0
    return (
        float(stats["max_level"]),
        active_bonus,
        float(version_rank(bundle["id"])),
        float(stats["rationale_coverage"]),
        float(stats["turns_total"]),
        float(bundle["mtime"] or 0.0),
    )


def build_bundle(run_dir: Path, *, active_ns: str | None) -> dict[str, Any] | None:
    namespace = run_dir.name
    goal_board_path = run_dir / "goal_board.json"
    if not goal_board_path.exists():
        return None
    log_path = REPO_ROOT / "experiment_logs" / "run_logs" / f"{GAME_PREFIX}_{namespace}.log"
    recording_path = choose_recording(namespace, run_dir, log_path)
    if recording_path is None:
        return None
    record_steps = parse_recording(recording_path)
    goal = parse_goal_board(goal_board_path)
    choices = goal["choices"]
    observations = goal["observations"]
    if not choices or not record_steps:
        return None

    record_labels = [step["action"]["label"] for step in record_steps]
    choice_labels = [str((choice.get("action_sequence") or [""])[0]) for choice in choices]
    record_offset = find_offset(record_labels, choice_labels, prefix=10)
    if record_offset is None:
        return None

    obs_actions = [str(obs.get("action") or "") for obs in observations]
    obs_offset = find_offset(choice_labels, obs_actions, prefix=6)
    observation_map: dict[int, dict[str, Any]] = {}
    if obs_offset is not None:
        for idx, observation in enumerate(observations):
            observation_map[obs_offset + idx] = observation

    log_meta = parse_run_log(log_path)
    reflexion_events = sorted(
        log_meta["reflexion_events"],
        key=lambda event: (event["turn_after"], event["text"]),
    )

    skill_anchor_hist = Counter(
        str(choice.get("skill_anchor") or "none")
        for choice in choices
    )
    final_skills = [
        {
            "skill_type": str(item.get("skill_type") or "unknown"),
            "goal_phrase": shorten(str(item.get("goal_phrase") or ""), 170),
            "causal_mapping": shorten(str(item.get("causal_mapping") or ""), 180),
        }
        for item in goal["skills"][-3:]
    ]
    final_cross = [
        {
            "id": str(item.get("id") or ""),
            "predicate": shorten(str(item.get("predicate") or ""), 160),
            "level_total": item.get("confirmed_at_level_delta_total"),
        }
        for item in goal["cross_level_confirmed"][-3:]
    ]

    turns: list[dict[str, Any]] = []
    max_level = 0
    level_up_turns: list[int] = []
    active_reflexion: dict[str, Any] | None = None
    reflexion_pointer = 0
    rationale_available = 0
    for choice_index, choice in enumerate(choices):
        step_index = record_offset + choice_index
        if step_index >= len(record_steps):
            break
        step = record_steps[step_index]
        observation = observation_map.get(choice_index)
        if step["levels_before"] < step["levels_after"]:
            active_reflexion = None
        if reflexion_pointer < len(reflexion_events):
            next_event = reflexion_events[reflexion_pointer]
            if next_event["turn_after"] + 1 == choice_index + 1:
                active_reflexion = next_event
                reflexion_pointer += 1
        derived = diff_signature(step["before_grid"], step["after_grid"])
        predicate = None
        if observation is not None:
            predicate = str(observation.get("predicate") or "").strip() or None
        if predicate:
            rationale_available += 1
        reported_changed = observation.get("changed_cells") if observation is not None else None
        levels_before = int(step["levels_before"])
        levels_after = int(step["levels_after"])
        level_delta = levels_after - levels_before
        if level_delta > 0:
            level_up_turns.append(choice_index + 1)
        max_level = max(max_level, levels_after)
        turn = {
            "turn_index": choice_index + 1,
            "recording_index": step["recording_index"],
            "timestamp": step["timestamp"],
            "action_label": step["action"]["label"],
            "action_name": step["action"]["name"],
            "x": step["action"]["x"],
            "y": step["action"]["y"],
            "card_id": str(choice.get("card_id") or ""),
            "skill_anchor": str(choice.get("skill_anchor") or "none"),
            "choice_verdict": str(choice.get("verdict") or ""),
            "observation_verdict": str(
                (observation or {}).get("verdict")
                or choice.get("verdict")
                or ""
            ),
            "predicate": predicate,
            "predicate_source": "recorded" if predicate else "missing",
            "primary_region_id": (observation or {}).get("primary_region_id"),
            "reported_changed_cells": reported_changed,
            "derived_changed_cells": int(derived["changed_cells"]),
            "dominant_transition": (observation or {}).get("dominant_transition") or derived["dominant_transition"],
            "derived_transition": derived["dominant_transition"],
            "bbox": derived["bbox"],
            "levels_before": levels_before,
            "levels_after": levels_after,
            "level_delta": int((observation or {}).get("level_delta", level_delta) or level_delta),
            "state_after": step["state_after"],
            "before_grid": encode_grid(step["before_grid"]),
            "after_grid": encode_grid(step["after_grid"]),
            "reflexion_active": active_reflexion,
            "reflexion_started_here": bool(
                active_reflexion
                and active_reflexion["turn_after"] + 1 == choice_index + 1
            ),
            "provenance": {
                "state_frames": "recorded",
                "predicate": "recorded" if predicate else "missing",
                "grid_diff": "derived",
                "reflexion": "recorded" if active_reflexion else "missing",
                "m2_expected_step_diffs": "missing",
            },
        }
        turns.append(turn)

    if not turns:
        return None

    default_turn_index = max(
        (
            turn["turn_index"]
            for turn in turns
            if turn["levels_after"] == max_level and turn["level_delta"] > 0
        ),
        default=max(
            turns,
            key=lambda item: (
                item["levels_after"],
                item["derived_changed_cells"],
                item["turn_index"],
            ),
        )["turn_index"],
    )

    stats = {
        "turns_total": len(turns),
        "max_level": max_level,
        "level_up_turns": level_up_turns,
        "rationale_turns": rationale_available,
        "rationale_coverage": round(rationale_available / max(1, len(turns)), 3),
        "skills_final": len(goal["skills"]),
        "cross_level_final": len(goal["cross_level_confirmed"]),
        "reflexion_events": len(reflexion_events),
        "record_offset": record_offset,
        "observation_offset": obs_offset,
        "skill_anchor_histogram": dict(skill_anchor_hist),
    }

    return {
        "id": namespace,
        "title": namespace.replace("_", " "),
        "namespace": namespace,
        "game_id": GAME_ID,
        "default_turn_index": default_turn_index,
        "turns": turns,
        "timeline": {
            "reflexion_events": reflexion_events,
            "level_up_turns": level_up_turns,
        },
        "final_skill_bank": final_skills,
        "final_cross_level": final_cross,
        "stats": stats,
        "sources": {
            "goal_board": repo_rel(goal_board_path),
            "recording": repo_rel(recording_path),
            "run_log": repo_rel(log_path),
        },
        "mtime": max(
            goal_board_path.stat().st_mtime,
            recording_path.stat().st_mtime,
            log_path.stat().st_mtime if log_path.exists() else 0.0,
        ),
        "active_namespace": active_ns,
    }


def choose_bundles(*, active_ns: str | None, max_runs: int) -> list[dict[str, Any]]:
    base_dir = REPO_ROOT / "simple_logs" / GAME_ID
    candidates: list[dict[str, Any]] = []
    if not base_dir.exists():
        return candidates
    for run_dir in sorted(base_dir.iterdir()):
        if not run_dir.is_dir():
            continue
        if not re.match(r"^v\d+.*cycle", run_dir.name):
            continue
        bundle = build_bundle(run_dir, active_ns=active_ns)
        if bundle is None:
            continue
        if bundle["stats"]["max_level"] < 1:
            continue
        candidates.append(bundle)
    candidates.sort(key=score_bundle, reverse=True)

    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    if candidates:
        selected.append(candidates[0])
        seen.add(candidates[0]["id"])
    if active_ns:
        active_bundle = next((item for item in candidates if item["id"] == active_ns), None)
        if active_bundle and active_bundle["id"] not in seen:
            insert_at = 1 if selected else 0
            selected.insert(insert_at, active_bundle)
            seen.add(active_bundle["id"])
    recent_candidates = [
        item for item in candidates
        if version_rank(item["id"]) >= 38 and item["id"] not in seen
    ]
    for bundle in recent_candidates:
        if len(selected) >= max_runs:
            break
        selected.append(bundle)
        seen.add(bundle["id"])
    if len(selected) < max_runs:
        for bundle in candidates:
            if bundle["id"] in seen:
                continue
            selected.append(bundle)
            seen.add(bundle["id"])
            if len(selected) >= max_runs:
                break
    selected = selected[:max_runs]
    return selected


def badge(text: str, tone: str = "neutral") -> str:
    return f'<span class="badge badge-{tone}">{html_text(text)}</span>'


def provenance_badge(kind: str) -> str:
    cls = {
        "recorded": "prov-recorded",
        "derived": "prov-derived",
        "missing": "prov-missing",
    }.get(kind, "prov-neutral")
    return f'<span class="prov {cls}">{html_text(kind)}</span>'


def transition_text(value: dict[str, Any] | None) -> str:
    if not value:
        return "none"
    return f'{value.get("from")} -> {value.get("to")} x{value.get("count")}'


def build_timeline_html(bundle: dict[str, Any], selected_turn: int) -> str:
    parts = ['<div class="timeline-strip">']
    reflex_starts = {
        event["turn_after"] + 1
        for event in bundle["timeline"]["reflexion_events"]
    }
    level_ups = set(bundle["timeline"]["level_up_turns"])
    for turn in bundle["turns"]:
        classes = ["timeline-step", f'verdict-{turn["observation_verdict"] or "unknown"}']
        if turn["turn_index"] == selected_turn:
            classes.append("is-active")
        if turn["turn_index"] in level_ups:
            classes.append("has-level-up")
        if turn["turn_index"] in reflex_starts:
            classes.append("has-reflexion")
        title = (
            f'Turn {turn["turn_index"]} | {turn["action_label"]} | '
            f'L{turn["levels_after"]} | {turn["observation_verdict"]}'
        )
        parts.append(
            f'<button class="{" ".join(classes)}" data-turn="{turn["turn_index"]}" title="{html_text(title)}">'
            f'<span class="timeline-index">{turn["turn_index"]}</span>'
            '</button>'
        )
    parts.append("</div>")
    return "".join(parts)


def build_grid_triptych(turn: dict[str, Any]) -> str:
    before = decode_grid(turn["before_grid"])
    after = decode_grid(turn["after_grid"])
    highlight = {"x": turn["x"], "y": turn["y"]}
    return (
        '<div class="grid-triptych">'
        '<section class="grid-card">'
        '<div class="panel-kicker">Before</div>'
        f"{grid_svg(before, highlight=highlight)}"
        '</section>'
        '<section class="grid-card">'
        '<div class="panel-kicker">Delta</div>'
        f"{diff_svg(before, after, highlight=highlight)}"
        '</section>'
        '<section class="grid-card">'
        '<div class="panel-kicker">After</div>'
        f"{grid_svg(after, highlight=highlight)}"
        '</section>'
        "</div>"
    )


def build_turn_facts(turn: dict[str, Any]) -> str:
    changed = turn["reported_changed_cells"]
    if changed is None:
        changed_text = f'derived {turn["derived_changed_cells"]}'
    else:
        changed_text = str(changed)
    level_chip = (
        badge(f'+{turn["level_delta"]} level', "accent")
        if int(turn["level_delta"]) > 0
        else badge("no level change", "neutral")
    )
    return (
        '<div class="fact-grid">'
        f'<div class="fact-card"><div class="fact-label">Action</div><div class="fact-value">{html_text(turn["action_label"])}</div></div>'
        f'<div class="fact-card"><div class="fact-label">Card</div><div class="fact-value">{html_text(turn["card_id"])}</div></div>'
        f'<div class="fact-card"><div class="fact-label">Verdict</div><div class="fact-value">{html_text(turn["observation_verdict"] or turn["choice_verdict"] or "unknown")}</div></div>'
        f'<div class="fact-card"><div class="fact-label">Region</div><div class="fact-value">{html_text(turn["primary_region_id"] or "unknown")}</div></div>'
        f'<div class="fact-card"><div class="fact-label">Changed cells</div><div class="fact-value">{html_text(changed_text)}</div></div>'
        f'<div class="fact-card"><div class="fact-label">Dominant transition</div><div class="fact-value">{html_text(transition_text(turn["dominant_transition"]))}</div></div>'
        f'<div class="fact-card"><div class="fact-label">State</div><div class="fact-value">L{turn["levels_before"]} to L{turn["levels_after"]}</div></div>'
        f'<div class="fact-card"><div class="fact-label">Level delta</div><div class="fact-value">{level_chip}</div></div>'
        "</div>"
    )


def build_module_card(title: str, body: str, *, prov: str | None = None) -> str:
    prov_html = provenance_badge(prov) if prov else ""
    return (
        '<section class="module-card">'
        f'<div class="module-head"><h3>{html_text(title)}</h3>{prov_html}</div>'
        f'<div class="module-body">{body}</div>'
        "</section>"
    )


def build_module_stack(bundle: dict[str, Any], turn: dict[str, Any]) -> str:
    predicate_body = (
        f'<p class="module-text">{html_text(turn["predicate"])}</p>'
        if turn["predicate"]
        else (
            '<p class="module-text muted">Selected predicate is unavailable for this turn. '
            "This run kept only the tail of observation_log, so early turns retain action history but not predicate text.</p>"
        )
    )
    m2_body = (
        "<dl class=\"mini-defs\">"
        f'<div><dt>chosen action</dt><dd>{html_text(turn["action_label"])}</dd></div>'
        f'<div><dt>skill anchor</dt><dd>{html_text(turn["skill_anchor"])}</dd></div>'
        f'<div><dt>raw step diff JSON</dt><dd class="muted">not persisted in this runtime</dd></div>'
        "</dl>"
    )
    observer_body = (
        "<dl class=\"mini-defs\">"
        f'<div><dt>primary region</dt><dd>{html_text(turn["primary_region_id"] or "unknown")}</dd></div>'
        f'<div><dt>dominant transition</dt><dd>{html_text(transition_text(turn["dominant_transition"]))}</dd></div>'
        f'<div><dt>changed cells</dt><dd>{html_text(turn["reported_changed_cells"] if turn["reported_changed_cells"] is not None else turn["derived_changed_cells"])}</dd></div>'
        f'<div><dt>verdict</dt><dd>{html_text(turn["observation_verdict"] or turn["choice_verdict"] or "unknown")}</dd></div>'
        "</dl>"
    )
    reflexion = turn["reflexion_active"]
    reflexion_body = (
        (
            f'<p class="module-text">{html_text(reflexion["text"])}</p>'
            f'<div class="reflex-note">injected after turn {reflexion["turn_after"]} '
            f'(stagnation {reflexion["stagnation_window"]})</div>'
        )
        if reflexion
        else '<p class="module-text muted">No active reflexion buffer on this turn.</p>'
    )
    skills_body = (
        "<div class=\"skill-list\">"
        + (
            "".join(
                '<article class="skill-chip-card">'
                f'<div class="skill-type">{html_text(item["skill_type"])}</div>'
                f'<div class="skill-goal">{html_text(item["goal_phrase"])}</div>'
                "</article>"
                for item in bundle["final_skill_bank"]
            )
            if bundle["final_skill_bank"]
            else '<p class="module-text muted">Final skill bank is empty for this run.</p>'
        )
        + "</div>"
    )
    cross_body = (
        "<div class=\"cross-list\">"
        + (
            "".join(
                '<article class="cross-chip-card">'
                f'<div class="cross-id">{html_text(item["id"])}</div>'
                f'<div class="cross-text">{html_text(item["predicate"])}</div>'
                f'<div class="cross-level">level total {html_text(item["level_total"])}</div>'
                "</article>"
                for item in bundle["final_cross_level"]
            )
            if bundle["final_cross_level"]
            else '<p class="module-text muted">No cross-level confirmed predicates persisted.</p>'
        )
        + "</div>"
    )
    return (
        build_module_card("M1 selected hypothesis", predicate_body, prov=turn["provenance"]["predicate"])
        + build_module_card("M2 emitted action", m2_body, prov="recorded")
        + build_module_card("Observer and falsifier", observer_body, prov="recorded")
        + build_module_card("Reflexion memory", reflexion_body, prov=turn["provenance"]["reflexion"])
        + build_module_card("Final skill bank snapshot", skills_body, prov="recorded")
        + build_module_card("Cross-level memory snapshot", cross_body, prov="recorded")
    )


def build_meta_panel(bundle: dict[str, Any]) -> str:
    stats = bundle["stats"]
    coverage_pct = int(round(stats["rationale_coverage"] * 100))
    active_label = "active run" if bundle["id"] == bundle.get("active_namespace") else "completed run"
    selection_reason = (
        "Chosen as the default because it reaches the highest solved level while retaining full persisted rationale coverage."
        if stats["max_level"] >= 4 and coverage_pct == 100
        else "Chosen to balance progress depth with how much persisted rationale survived in the artifacts."
    )
    items = [
        badge(f'max {stats["max_level"]}/6', "accent"),
        badge(f'{stats["turns_total"]} turns', "neutral"),
        badge(f'{coverage_pct}% rationale coverage', "info"),
        badge(f'{stats["reflexion_events"]} reflexion events', "info"),
        badge(active_label, "neutral"),
    ]
    source_lines = "".join(
        f'<div class="source-line"><span>{html_text(key)}</span><code>{html_text(value or "missing")}</code></div>'
        for key, value in bundle["sources"].items()
    )
    return (
        '<section class="hero-card">'
        f'<div class="hero-title-row"><div><div class="eyebrow">Best-progress FT09 trace</div><h1>{html_text(bundle["title"])}</h1></div>'
        f'<div class="hero-updated">updated {html_text(ts_to_text(bundle["mtime"]))}</div></div>'
        f'<div class="hero-badges">{"".join(items)}</div>'
        f'<p class="hero-copy">{html_text(selection_reason)} The viewer synchronizes actual ARC state changes, per-turn actions, and the persisted module outputs that survived this runtime. Recorded fields stay distinct from derived frame diffs and missing raw module IO.</p>'
        f'<div class="source-block">{source_lines}</div>'
        "</section>"
    )


def render_trace_html(payload: dict[str, Any]) -> str:
    default_run_id = payload["default_run_id"]
    bundle = payload["bundles"][default_run_id]
    turn = bundle["turns"][bundle["default_turn_index"] - 1]
    select_options = "".join(
        f'<option value="{html_text(item["id"])}"{" selected" if item["id"] == default_run_id else ""}>'
        f'{html_text(item["label"])}'
        "</option>"
        for item in payload["manifest"]
    )
    manifest_cards = "".join(
        '<article class="run-card">'
        f'<div class="run-name">{html_text(item["label"])}</div>'
        f'<div class="run-stats">{badge(f"max {item["max_level"]}/6", "accent")}{badge(f"{item["turns_total"]} turns", "neutral")}{badge(f"{item["coverage_pct"]}% rationale", "info")}</div>'
        "</article>"
        for item in payload["manifest"]
    )
    trace_data = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>FT09 Trace Explorer</title>
  <style>
    :root {{
      --bg: #f4f2ea;
      --panel: rgba(255,255,255,0.92);
      --panel-strong: rgba(255,255,255,0.98);
      --ink: #163f48;
      --ink-soft: #4a676c;
      --accent: #eb9b3f;
      --accent-soft: #f5d8a7;
      --teal: #1d6b72;
      --teal-soft: #cfe8e7;
      --shadow: 0 18px 42px rgba(28,57,63,0.12);
      --border: rgba(24,69,76,0.12);
      --danger: #ad503e;
      --success: #2d7953;
      --muted: #71888d;
      --viewer-max: 1540px;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      background:
        radial-gradient(circle at 0% 0%, rgba(253, 211, 144, 0.34), transparent 34%),
        radial-gradient(circle at 100% 0%, rgba(157, 205, 208, 0.22), transparent 28%),
        linear-gradient(180deg, rgba(255,255,255,0.6), rgba(244,242,234,1) 28%),
        var(--bg);
      font-family: "Avenir Next", "Segoe UI", "Trebuchet MS", sans-serif;
    }}
    body::before {{
      content: "";
      position: fixed;
      inset: 0;
      background-image:
        linear-gradient(rgba(22,63,72,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(22,63,72,0.03) 1px, transparent 1px);
      background-size: 32px 32px;
      pointer-events: none;
      z-index: -1;
    }}
    .shell {{
      max-width: var(--viewer-max);
      margin: 0 auto;
      padding: 28px 28px 44px;
    }}
    .hero-card, .control-card, .panel-card, .legend-card {{
      background: var(--panel);
      border: 1px solid var(--border);
      box-shadow: var(--shadow);
      border-radius: 24px;
    }}
    .hero-card {{
      padding: 26px 28px 24px;
      position: relative;
      overflow: hidden;
    }}
    .hero-card::after {{
      content: "";
      position: absolute;
      left: 0;
      right: 0;
      top: 0;
      height: 6px;
      background: linear-gradient(90deg, var(--accent), #e1bf6e 42%, #8bc6cb 100%);
    }}
    .hero-title-row {{
      display: flex;
      align-items: end;
      justify-content: space-between;
      gap: 20px;
    }}
    .eyebrow {{
      text-transform: uppercase;
      letter-spacing: 0.22em;
      font-size: 11px;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    h1, h2, h3 {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
      font-weight: 600;
    }}
    h1 {{
      font-size: clamp(30px, 4vw, 48px);
      line-height: 1.02;
    }}
    .hero-updated {{
      color: var(--ink-soft);
      font-size: 14px;
      white-space: nowrap;
    }}
    .hero-badges, .run-stats {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 18px;
    }}
    .hero-copy {{
      margin: 18px 0 0;
      max-width: 980px;
      line-height: 1.6;
      color: var(--ink-soft);
      font-size: 15px;
    }}
    .badge {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 7px 12px;
      font-size: 13px;
      font-weight: 600;
      letter-spacing: 0.01em;
    }}
    .badge-neutral {{ background: #edf2f1; color: var(--ink); }}
    .badge-accent {{ background: var(--accent-soft); color: #7a4c12; }}
    .badge-info {{ background: var(--teal-soft); color: #0f5457; }}
    .prov {{
      display: inline-flex;
      align-items: center;
      border-radius: 999px;
      padding: 4px 9px;
      font-size: 11px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.08em;
    }}
    .prov-recorded {{ background: #daf0e0; color: #1d6b47; }}
    .prov-derived {{ background: #dbe8f3; color: #2d5e87; }}
    .prov-missing {{ background: #f2dfd8; color: #9a4836; }}
    .prov-neutral {{ background: #e9ecef; color: #53616a; }}
    .source-block {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      margin-top: 20px;
    }}
    .source-line {{
      display: grid;
      grid-template-columns: 120px 1fr;
      gap: 14px;
      align-items: center;
      font-size: 14px;
      color: var(--ink-soft);
    }}
    .source-line code {{
      display: block;
      overflow-wrap: anywhere;
      font-size: 13px;
      padding: 7px 10px;
      background: rgba(22,63,72,0.04);
      border-radius: 12px;
      color: var(--ink);
    }}
    .run-overview {{
      display: flex;
      flex-wrap: wrap;
      gap: 14px;
      margin: 20px 0 22px;
    }}
    .run-card {{
      background: rgba(255,255,255,0.7);
      border: 1px solid var(--border);
      border-radius: 18px;
      padding: 16px 16px 14px;
      flex: 1 1 240px;
      min-width: 240px;
    }}
    .run-name {{
      font-size: 16px;
      font-weight: 700;
      color: var(--ink);
    }}
    .control-card {{
      margin-top: 18px;
      padding: 20px 22px;
    }}
    .control-row {{
      display: grid;
      grid-template-columns: minmax(220px, 360px) 1fr auto;
      gap: 16px;
      align-items: center;
    }}
    .control-stack {{
      display: grid;
      gap: 8px;
    }}
    .label {{
      font-size: 11px;
      letter-spacing: 0.18em;
      text-transform: uppercase;
      color: var(--muted);
      font-weight: 700;
    }}
    select, button, input[type="range"] {{
      font: inherit;
    }}
    select {{
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 12px 14px;
      background: var(--panel-strong);
      color: var(--ink);
    }}
    .turn-bar {{
      display: grid;
      gap: 10px;
    }}
    .turn-actions {{
      display: flex;
      align-items: center;
      gap: 10px;
      justify-content: end;
    }}
    button {{
      border: 0;
      border-radius: 999px;
      padding: 11px 16px;
      cursor: pointer;
      color: white;
      background: var(--teal);
      box-shadow: 0 10px 24px rgba(29,107,114,0.18);
    }}
    button.ghost {{
      background: white;
      color: var(--ink);
      border: 1px solid var(--border);
      box-shadow: none;
    }}
    input[type="range"] {{
      width: 100%;
      accent-color: var(--accent);
    }}
    .timeline-panel {{
      margin-top: 16px;
      overflow-x: auto;
      padding-bottom: 2px;
    }}
    .timeline-strip {{
      display: inline-flex;
      gap: 7px;
      min-width: 100%;
    }}
    .timeline-step {{
      position: relative;
      width: 40px;
      min-width: 40px;
      height: 52px;
      border-radius: 14px;
      border: 1px solid rgba(20,55,61,0.08);
      background: #f1f4f3;
      color: var(--ink);
      box-shadow: none;
      padding: 0;
    }}
    .timeline-step.verdict-confirm {{ background: #d8efe0; }}
    .timeline-step.verdict-falsify {{ background: #f2dfd8; }}
    .timeline-step.verdict-inconclusive {{ background: #ebe5d8; }}
    .timeline-step.is-active {{
      outline: 3px solid var(--accent);
      outline-offset: 2px;
      transform: translateY(-2px);
    }}
    .timeline-step.has-level-up::before {{
      content: "";
      position: absolute;
      left: 6px;
      right: 6px;
      top: 5px;
      height: 4px;
      border-radius: 999px;
      background: var(--teal);
    }}
    .timeline-step.has-reflexion::after {{
      content: "";
      position: absolute;
      right: 4px;
      bottom: 4px;
      width: 9px;
      height: 9px;
      border-radius: 50%;
      background: var(--accent);
      box-shadow: 0 0 0 3px rgba(244,163,65,0.18);
    }}
    .timeline-index {{
      display: flex;
      align-items: center;
      justify-content: center;
      height: 100%;
      font-weight: 800;
      font-size: 13px;
    }}
    .content-grid {{
      display: grid;
      grid-template-columns: minmax(0, 1.24fr) minmax(320px, 0.88fr) minmax(360px, 1fr);
      gap: 18px;
      margin-top: 18px;
      align-items: start;
    }}
    .panel-card {{
      padding: 18px 18px 20px;
    }}
    .panel-head {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: start;
      margin-bottom: 16px;
    }}
    .panel-kicker {{
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.18em;
      color: var(--muted);
      margin-bottom: 10px;
      font-weight: 700;
    }}
    .turn-title {{
      font-size: clamp(24px, 2vw, 34px);
    }}
    .turn-subtitle {{
      margin-top: 6px;
      color: var(--ink-soft);
      font-size: 15px;
    }}
    .grid-triptych {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 12px;
    }}
    .grid-card {{
      background: linear-gradient(180deg, rgba(19,37,43,0.98), rgba(13,27,31,0.98));
      border-radius: 20px;
      padding: 14px;
      color: white;
      min-height: 100%;
    }}
    .grid-card .panel-kicker {{
      color: rgba(255,255,255,0.72);
      margin-bottom: 12px;
    }}
    .grid-svg {{
      display: block;
      width: 100%;
      height: auto;
    }}
    .click-ring {{
      transform-origin: center;
      animation: clickPulse 1.8s ease-out infinite;
    }}
    @keyframes clickPulse {{
      0% {{ opacity: 0.95; transform: scale(0.92); }}
      70% {{ opacity: 0.18; transform: scale(1.18); }}
      100% {{ opacity: 0.95; transform: scale(0.92); }}
    }}
    .fact-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 12px;
    }}
    .fact-card {{
      padding: 14px 14px 12px;
      border-radius: 18px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.74);
    }}
    .fact-label {{
      font-size: 11px;
      font-weight: 700;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.16em;
      margin-bottom: 8px;
    }}
    .fact-value {{
      line-height: 1.45;
      font-weight: 700;
    }}
    .module-stack {{
      display: grid;
      gap: 12px;
    }}
    .module-card {{
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.78);
      border-radius: 18px;
      padding: 14px 15px 15px;
    }}
    .module-head {{
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      margin-bottom: 10px;
    }}
    .module-head h3 {{
      font-size: 18px;
    }}
    .module-text {{
      margin: 0;
      line-height: 1.6;
      color: var(--ink);
      font-size: 14px;
    }}
    .muted {{
      color: var(--muted);
    }}
    .mini-defs {{
      display: grid;
      gap: 8px;
      margin: 0;
    }}
    .mini-defs div {{
      display: grid;
      grid-template-columns: 110px 1fr;
      gap: 10px;
      align-items: start;
    }}
    .mini-defs dt {{
      margin: 0;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.14em;
      color: var(--muted);
      font-weight: 700;
    }}
    .mini-defs dd {{
      margin: 0;
      line-height: 1.45;
      color: var(--ink);
      font-size: 14px;
    }}
    .reflex-note {{
      margin-top: 10px;
      font-size: 12px;
      color: var(--ink-soft);
    }}
    .skill-list, .cross-list {{
      display: grid;
      gap: 10px;
    }}
    .skill-chip-card, .cross-chip-card {{
      border-radius: 14px;
      border: 1px solid var(--border);
      background: rgba(243,247,246,0.86);
      padding: 11px 12px;
    }}
    .skill-type, .cross-id {{
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.16em;
      color: var(--muted);
      font-weight: 700;
      margin-bottom: 7px;
    }}
    .skill-goal, .cross-text {{
      line-height: 1.5;
      font-size: 13px;
    }}
    .cross-level {{
      margin-top: 8px;
      color: var(--ink-soft);
      font-size: 12px;
    }}
    .legend-card {{
      margin-top: 18px;
      padding: 16px 18px;
      display: flex;
      flex-wrap: wrap;
      gap: 12px 16px;
      align-items: center;
    }}
    .legend-copy {{
      font-size: 14px;
      color: var(--ink-soft);
    }}
    .empty {{
      display: grid;
      place-items: center;
      min-height: 240px;
      color: rgba(255,255,255,0.7);
      font-size: 14px;
    }}
    @media (max-width: 1220px) {{
      .content-grid {{
        grid-template-columns: 1fr;
      }}
      .source-block {{
        grid-template-columns: 1fr;
      }}
    }}
    @media (max-width: 880px) {{
      .control-row {{
        grid-template-columns: 1fr;
      }}
      .turn-actions {{
        justify-content: start;
        flex-wrap: wrap;
      }}
      .grid-triptych {{
        grid-template-columns: 1fr;
      }}
      .fact-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section id="metaPanel">{build_meta_panel(bundle)}</section>
    <section class="run-overview" id="runOverview">{manifest_cards}</section>
    <section class="control-card">
      <div class="control-row">
        <div class="control-stack">
          <label class="label" for="runSelector">Run selector</label>
          <select id="runSelector">{select_options}</select>
        </div>
        <div class="control-stack turn-bar">
          <label class="label" for="turnRange">Turn</label>
          <input id="turnRange" type="range" min="1" max="{bundle["stats"]["turns_total"]}" value="{bundle["default_turn_index"]}">
        </div>
        <div class="turn-actions">
          <button class="ghost" id="prevTurn" type="button">Prev</button>
          <button id="playToggle" type="button">Play</button>
          <button class="ghost" id="nextTurn" type="button">Next</button>
        </div>
      </div>
      <div class="timeline-panel" id="timelinePanel">{build_timeline_html(bundle, bundle["default_turn_index"])}</div>
    </section>

    <section class="content-grid">
      <section class="panel-card" id="gridPanel">
        <div class="panel-head">
          <div>
            <div class="panel-kicker">State change</div>
            <h2 class="turn-title" id="turnTitle">Turn {turn["turn_index"]}: {html_text(turn["action_label"])}</h2>
            <div class="turn-subtitle" id="turnSubtitle">Card {html_text(turn["card_id"])} | L{turn["levels_before"]} to L{turn["levels_after"]} | {html_text(turn["observation_verdict"] or turn["choice_verdict"] or "unknown")}</div>
          </div>
          <div id="turnProvenance">{provenance_badge("recorded")}{provenance_badge("derived")}</div>
        </div>
        <div id="gridTriptych">{build_grid_triptych(turn)}</div>
      </section>

      <section class="panel-card" id="factsPanel">
        <div class="panel-kicker">Action and outcome</div>
        <div id="factGrid">{build_turn_facts(turn)}</div>
      </section>

      <section class="panel-card" id="modulePanel">
        <div class="panel-kicker">Persisted module responses</div>
        <div class="module-stack" id="moduleStack">{build_module_stack(bundle, turn)}</div>
      </section>
    </section>

    <section class="legend-card">
      <span class="legend-copy">Legend:</span>
      {provenance_badge("recorded")}
      <span class="legend-copy">persisted in run artifacts</span>
      {provenance_badge("derived")}
      <span class="legend-copy">computed from frame deltas</span>
      {provenance_badge("missing")}
      <span class="legend-copy">not stored by this runtime, so the viewer marks the gap explicitly</span>
    </section>
  </main>

  <script id="trace-data" type="application/json">{trace_data}</script>
  <script>
    const DATA = JSON.parse(document.getElementById("trace-data").textContent);
    const PALETTE = {json.dumps(PALETTE)};
    const state = {{
      runId: DATA.default_run_id,
      turnIndex: DATA.bundles[DATA.default_run_id].default_turn_index,
      playing: false,
      timer: null,
    }};

    const $ = (id) => document.getElementById(id);

    function escapeHtml(text) {{
      return String(text ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;")
        .replace(/"/g, "&quot;")
        .replace(/'/g, "&#39;");
    }}

    function shorten(text, limit = 180) {{
      const value = String(text || "").replace(/\\s+/g, " ").trim();
      if (value.length <= limit) return value;
      return value.slice(0, limit - 1).trimEnd() + "...";
    }}

    function decodeGrid(encoded) {{
      return (encoded || []).map((row) => Array.from(row, (ch) => parseInt(ch, 16)));
    }}

    function badge(text, tone) {{
      return `<span class="badge badge-${{tone}}">${{escapeHtml(text)}}</span>`;
    }}

    function prov(kind) {{
      const cls = {{
        recorded: "prov-recorded",
        derived: "prov-derived",
        missing: "prov-missing",
      }}[kind] || "prov-neutral";
      return `<span class="prov ${{cls}}">${{escapeHtml(kind)}}</span>`;
    }}

    function transitionText(value) {{
      if (!value) return "none";
      return `${{value.from}} -> ${{value.to}} x${{value.count}}`;
    }}

    function gridSvg(grid, {{ maxSide = 340, highlight = null }} = {{}}) {{
      if (!grid.length || !grid[0].length) return '<div class="empty">no grid</div>';
      const rows = grid.length;
      const cols = grid[0].length;
      const cell = Math.max(3, Math.floor(maxSide / Math.max(rows, cols)));
      const width = cols * cell;
      const height = rows * cell;
      const parts = [
        `<svg viewBox="0 0 ${{width}} ${{height}}" width="${{width}}" height="${{height}}" class="grid-svg" aria-hidden="true">`,
        `<rect width="${{width}}" height="${{height}}" rx="18" fill="#13252B"></rect>`
      ];
      for (let y = 0; y < rows; y += 1) {{
        let runStart = 0;
        let current = grid[y][0];
        for (let x = 1; x <= cols; x += 1) {{
          const value = x < cols ? grid[y][x] : null;
          if (x === cols || value !== current) {{
            parts.push(`<rect x="${{runStart * cell}}" y="${{y * cell}}" width="${{(x - runStart) * cell}}" height="${{cell}}" fill="${{PALETTE[current % 16]}}"></rect>`);
            if (x < cols) {{
              runStart = x;
              current = value;
            }}
          }}
        }}
      }}
      parts.push(`<rect width="${{width}}" height="${{height}}" rx="18" fill="none" stroke="#D8E3E1" stroke-width="2"></rect>`);
      if (highlight && highlight.x != null && highlight.y != null) {{
        const cx = (Number(highlight.x) + 0.5) * cell;
        const cy = (Number(highlight.y) + 0.5) * cell;
        const radius = Math.max(4, cell * 1.5);
        parts.push(`<circle cx="${{cx}}" cy="${{cy}}" r="${{radius}}" class="click-ring" fill="none" stroke="#F4A341" stroke-width="${{Math.max(2, cell / 2.6)}}"></circle>`);
        parts.push(`<circle cx="${{cx}}" cy="${{cy}}" r="${{Math.max(2, cell / 2.4)}}" fill="#F4A341" opacity="0.86"></circle>`);
      }}
      parts.push("</svg>");
      return parts.join("");
    }}

    function diffSvg(before, after, {{ maxSide = 340, highlight = null }} = {{}}) {{
      if (!before.length || !after.length) return '<div class="empty">no diff</div>';
      const rows = Math.min(before.length, after.length);
      const cols = Math.min(before[0].length, after[0].length);
      const cell = Math.max(3, Math.floor(maxSide / Math.max(rows, cols)));
      const width = cols * cell;
      const height = rows * cell;
      const parts = [
        `<svg viewBox="0 0 ${{width}} ${{height}}" width="${{width}}" height="${{height}}" class="grid-svg" aria-hidden="true">`,
        `<rect width="${{width}}" height="${{height}}" rx="18" fill="#122028"></rect>`
      ];
      for (let y = 0; y < rows; y += 1) {{
        for (let x = 0; x < cols; x += 1) {{
          if (before[y][x] === after[y][x]) continue;
          parts.push(`<rect x="${{x * cell}}" y="${{y * cell}}" width="${{cell}}" height="${{cell}}" fill="${{PALETTE[after[y][x] % 16]}}"></rect>`);
          parts.push(`<rect x="${{x * cell}}" y="${{y * cell}}" width="${{cell}}" height="${{cell}}" fill="none" stroke="#F4A341" stroke-width="${{Math.max(1, cell / 4)}}"></rect>`);
        }}
      }}
      parts.push(`<rect width="${{width}}" height="${{height}}" rx="18" fill="none" stroke="#D8E3E1" stroke-width="2"></rect>`);
      if (highlight && highlight.x != null && highlight.y != null) {{
        const cx = (Number(highlight.x) + 0.5) * cell;
        const cy = (Number(highlight.y) + 0.5) * cell;
        const radius = Math.max(4, cell * 1.5);
        parts.push(`<circle cx="${{cx}}" cy="${{cy}}" r="${{radius}}" class="click-ring" fill="none" stroke="#9BD1D4" stroke-width="${{Math.max(2, cell / 2.6)}}"></circle>`);
      }}
      parts.push("</svg>");
      return parts.join("");
    }}

    function buildTimeline(bundle, selectedTurn) {{
      const reflexStarts = new Set((bundle.timeline.reflexion_events || []).map((event) => event.turn_after + 1));
      const levelUps = new Set(bundle.timeline.level_up_turns || []);
      return `<div class="timeline-strip">${{bundle.turns.map((turn) => {{
        const classes = ["timeline-step", `verdict-${{turn.observation_verdict || "unknown"}}`];
        if (turn.turn_index === selectedTurn) classes.push("is-active");
        if (levelUps.has(turn.turn_index)) classes.push("has-level-up");
        if (reflexStarts.has(turn.turn_index)) classes.push("has-reflexion");
        const title = `Turn ${{turn.turn_index}} | ${{turn.action_label}} | L${{turn.levels_after}} | ${{turn.observation_verdict || turn.choice_verdict || "unknown"}}`;
        return `<button class="${{classes.join(" ")}}" data-turn="${{turn.turn_index}}" title="${{escapeHtml(title)}}"><span class="timeline-index">${{turn.turn_index}}</span></button>`;
      }}).join("")}}</div>`;
    }}

    function buildMetaPanel(bundle) {{
      const stats = bundle.stats;
      const coveragePct = Math.round((stats.rationale_coverage || 0) * 100);
      const activeLabel = bundle.id === DATA.active_namespace ? "active run" : "completed run";
      const selectionReason = (stats.max_level >= 4 && coveragePct === 100)
        ? "Chosen as the default because it reaches the highest solved level while retaining full persisted rationale coverage."
        : "Chosen to balance progress depth with how much persisted rationale survived in the artifacts.";
      const badges = [
        badge(`max ${{stats.max_level}}/6`, "accent"),
        badge(`${{stats.turns_total}} turns`, "neutral"),
        badge(`${{coveragePct}}% rationale coverage`, "info"),
        badge(`${{stats.reflexion_events}} reflexion events`, "info"),
        badge(activeLabel, "neutral"),
      ].join("");
      const sourceLines = Object.entries(bundle.sources || {{}}).map(([key, value]) =>
        `<div class="source-line"><span>${{escapeHtml(key)}}</span><code>${{escapeHtml(value || "missing")}}</code></div>`
      ).join("");
      return `<section class="hero-card">
        <div class="hero-title-row">
          <div>
            <div class="eyebrow">Best-progress FT09 trace</div>
            <h1>${{escapeHtml(bundle.title)}}</h1>
          </div>
          <div class="hero-updated">updated ${{escapeHtml(bundle.updated_at || "")}}</div>
        </div>
        <div class="hero-badges">${{badges}}</div>
        <p class="hero-copy">${{escapeHtml(selectionReason)}} The viewer synchronizes actual ARC state changes, per-turn actions, and the persisted module outputs that survived this runtime. Recorded fields stay distinct from derived frame diffs and missing raw module IO.</p>
        <div class="source-block">${{sourceLines}}</div>
      </section>`;
    }}

    function buildRunOverview() {{
      return DATA.manifest.map((item) => `
        <article class="run-card">
          <div class="run-name">${{escapeHtml(item.label)}}</div>
          <div class="run-stats">
            ${{badge(`max ${{item.max_level}}/6`, "accent")}}
            ${{badge(`${{item.turns_total}} turns`, "neutral")}}
            ${{badge(`${{item.coverage_pct}}% rationale`, "info")}}
          </div>
        </article>
      `).join("");
    }}

    function buildGridTriptych(turn) {{
      const before = decodeGrid(turn.before_grid);
      const after = decodeGrid(turn.after_grid);
      const highlight = {{ x: turn.x, y: turn.y }};
      return `
        <div class="grid-triptych">
          <section class="grid-card">
            <div class="panel-kicker">Before</div>
            ${{gridSvg(before, {{ highlight }})}}
          </section>
          <section class="grid-card">
            <div class="panel-kicker">Delta</div>
            ${{diffSvg(before, after, {{ highlight }})}}
          </section>
          <section class="grid-card">
            <div class="panel-kicker">After</div>
            ${{gridSvg(after, {{ highlight }})}}
          </section>
        </div>
      `;
    }}

    function buildFactGrid(turn) {{
      const changedText = turn.reported_changed_cells == null ? `derived ${{turn.derived_changed_cells}}` : String(turn.reported_changed_cells);
      const levelChip = Number(turn.level_delta) > 0 ? badge(`+${{turn.level_delta}} level`, "accent") : badge("no level change", "neutral");
      return `
        <div class="fact-grid">
          <div class="fact-card"><div class="fact-label">Action</div><div class="fact-value">${{escapeHtml(turn.action_label)}}</div></div>
          <div class="fact-card"><div class="fact-label">Card</div><div class="fact-value">${{escapeHtml(turn.card_id)}}</div></div>
          <div class="fact-card"><div class="fact-label">Verdict</div><div class="fact-value">${{escapeHtml(turn.observation_verdict || turn.choice_verdict || "unknown")}}</div></div>
          <div class="fact-card"><div class="fact-label">Region</div><div class="fact-value">${{escapeHtml(turn.primary_region_id || "unknown")}}</div></div>
          <div class="fact-card"><div class="fact-label">Changed cells</div><div class="fact-value">${{escapeHtml(changedText)}}</div></div>
          <div class="fact-card"><div class="fact-label">Dominant transition</div><div class="fact-value">${{escapeHtml(transitionText(turn.dominant_transition))}}</div></div>
          <div class="fact-card"><div class="fact-label">State</div><div class="fact-value">L${{turn.levels_before}} to L${{turn.levels_after}}</div></div>
          <div class="fact-card"><div class="fact-label">Level delta</div><div class="fact-value">${{levelChip}}</div></div>
        </div>
      `;
    }}

    function moduleCard(title, body, provKind) {{
      return `
        <section class="module-card">
          <div class="module-head"><h3>${{escapeHtml(title)}}</h3>${{provKind ? prov(provKind) : ""}}</div>
          <div class="module-body">${{body}}</div>
        </section>
      `;
    }}

    function buildModuleStack(bundle, turn) {{
      const predicateBody = turn.predicate
        ? `<p class="module-text">${{escapeHtml(turn.predicate)}}</p>`
        : `<p class="module-text muted">Selected predicate is unavailable for this turn. This run kept only the tail of observation_log, so early turns retain action history but not predicate text.</p>`;
      const m2Body = `
        <dl class="mini-defs">
          <div><dt>chosen action</dt><dd>${{escapeHtml(turn.action_label)}}</dd></div>
          <div><dt>skill anchor</dt><dd>${{escapeHtml(turn.skill_anchor)}}</dd></div>
          <div><dt>raw step diff JSON</dt><dd class="muted">not persisted in this runtime</dd></div>
        </dl>`;
      const observerBody = `
        <dl class="mini-defs">
          <div><dt>primary region</dt><dd>${{escapeHtml(turn.primary_region_id || "unknown")}}</dd></div>
          <div><dt>dominant transition</dt><dd>${{escapeHtml(transitionText(turn.dominant_transition))}}</dd></div>
          <div><dt>changed cells</dt><dd>${{escapeHtml(turn.reported_changed_cells == null ? turn.derived_changed_cells : turn.reported_changed_cells)}}</dd></div>
          <div><dt>verdict</dt><dd>${{escapeHtml(turn.observation_verdict || turn.choice_verdict || "unknown")}}</dd></div>
        </dl>`;
      const reflexion = turn.reflexion_active;
      const reflexionBody = reflexion
        ? `<p class="module-text">${{escapeHtml(reflexion.text)}}</p><div class="reflex-note">injected after turn ${{reflexion.turn_after}} (stagnation ${{reflexion.stagnation_window}})</div>`
        : `<p class="module-text muted">No active reflexion buffer on this turn.</p>`;
      const skillsBody = (bundle.final_skill_bank || []).length
        ? `<div class="skill-list">${{bundle.final_skill_bank.map((item) => `
            <article class="skill-chip-card">
              <div class="skill-type">${{escapeHtml(item.skill_type)}}</div>
              <div class="skill-goal">${{escapeHtml(item.goal_phrase)}}</div>
            </article>`).join("")}}</div>`
        : `<p class="module-text muted">Final skill bank is empty for this run.</p>`;
      const crossBody = (bundle.final_cross_level || []).length
        ? `<div class="cross-list">${{bundle.final_cross_level.map((item) => `
            <article class="cross-chip-card">
              <div class="cross-id">${{escapeHtml(item.id)}}</div>
              <div class="cross-text">${{escapeHtml(item.predicate)}}</div>
              <div class="cross-level">level total ${{escapeHtml(item.level_total)}}</div>
            </article>`).join("")}}</div>`
        : `<p class="module-text muted">No cross-level confirmed predicates persisted.</p>`;
      return [
        moduleCard("M1 selected hypothesis", predicateBody, turn.provenance.predicate),
        moduleCard("M2 emitted action", m2Body, "recorded"),
        moduleCard("Observer and falsifier", observerBody, "recorded"),
        moduleCard("Reflexion memory", reflexionBody, turn.provenance.reflexion),
        moduleCard("Final skill bank snapshot", skillsBody, "recorded"),
        moduleCard("Cross-level memory snapshot", crossBody, "recorded"),
      ].join("");
    }}

    function stopPlayback() {{
      state.playing = false;
      if (state.timer) {{
        window.clearInterval(state.timer);
        state.timer = null;
      }}
      $("playToggle").textContent = "Play";
    }}

    function currentBundle() {{
      return DATA.bundles[state.runId];
    }}

    function currentTurn() {{
      const bundle = currentBundle();
      return bundle.turns[state.turnIndex - 1];
    }}

    function render() {{
      const bundle = currentBundle();
      const turn = currentTurn();
      $("metaPanel").innerHTML = buildMetaPanel(bundle);
      $("runOverview").innerHTML = buildRunOverview();
      $("runSelector").value = bundle.id;
      $("turnRange").max = String(bundle.stats.turns_total);
      $("turnRange").value = String(state.turnIndex);
      $("timelinePanel").innerHTML = buildTimeline(bundle, state.turnIndex);
      $("turnTitle").textContent = `Turn ${{turn.turn_index}}: ${{turn.action_label}}`;
      $("turnSubtitle").textContent = `Card ${{turn.card_id}} | L${{turn.levels_before}} to L${{turn.levels_after}} | ${{turn.observation_verdict || turn.choice_verdict || "unknown"}}`;
      $("turnProvenance").innerHTML = prov("recorded") + prov("derived");
      $("gridTriptych").innerHTML = buildGridTriptych(turn);
      $("factGrid").innerHTML = buildFactGrid(turn);
      $("moduleStack").innerHTML = buildModuleStack(bundle, turn);
      for (const button of $("timelinePanel").querySelectorAll("button[data-turn]")) {{
        button.addEventListener("click", () => {{
          stopPlayback();
          state.turnIndex = Number(button.dataset.turn);
          render();
        }});
      }}
    }}

    $("runSelector").addEventListener("change", (event) => {{
      stopPlayback();
      state.runId = event.target.value;
      state.turnIndex = DATA.bundles[state.runId].default_turn_index;
      render();
    }});
    $("turnRange").addEventListener("input", (event) => {{
      stopPlayback();
      state.turnIndex = Number(event.target.value);
      render();
    }});
    $("prevTurn").addEventListener("click", () => {{
      stopPlayback();
      state.turnIndex = Math.max(1, state.turnIndex - 1);
      render();
    }});
    $("nextTurn").addEventListener("click", () => {{
      stopPlayback();
      const bundle = currentBundle();
      state.turnIndex = Math.min(bundle.stats.turns_total, state.turnIndex + 1);
      render();
    }});
    $("playToggle").addEventListener("click", () => {{
      if (state.playing) {{
        stopPlayback();
        return;
      }}
      state.playing = true;
      $("playToggle").textContent = "Pause";
      state.timer = window.setInterval(() => {{
        const bundle = currentBundle();
        if (state.turnIndex >= bundle.stats.turns_total) {{
          stopPlayback();
          return;
        }}
        state.turnIndex += 1;
        render();
      }}, 1300);
    }});
    render();
  </script>
</body>
</html>
"""


def render_flow_svg() -> str:
    return """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1600 900" width="1600" height="900" role="img" aria-labelledby="title desc">
  <title id="title">FT09 runtime flow v5</title>
  <desc id="desc">Runtime flow from main.py through Swarm, ArcgenticaSimple, SemanticPackets, Reflexion, M1, M2, environment stepping, observation, GoalBoard, DiffMemory, and persisted artifacts.</desc>
  <defs>
    <linearGradient id="bgWash" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#F7F3EA"/>
      <stop offset="55%" stop-color="#F4F1E8"/>
      <stop offset="100%" stop-color="#EEF4F3"/>
    </linearGradient>
    <linearGradient id="accentLine" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#E99B3F"/>
      <stop offset="45%" stop-color="#F2C768"/>
      <stop offset="100%" stop-color="#6FB0B4"/>
    </linearGradient>
    <filter id="shadow" x="-20%" y="-20%" width="140%" height="160%">
      <feDropShadow dx="0" dy="12" stdDeviation="18" flood-color="#1B3940" flood-opacity="0.12"/>
    </filter>
    <style>
      .ink {{ fill: #184751; }}
      .soft {{ fill: #5D767B; }}
      .title {{ font: 600 44px 'Iowan Old Style', 'Palatino Linotype', serif; letter-spacing: 0.01em; fill: #163F48; }}
      .subtitle {{ font: 500 18px 'Avenir Next', 'Segoe UI', sans-serif; fill: #5D767B; }}
      .footer {{ font: 500 14px 'Avenir Next', 'Segoe UI', sans-serif; fill: #60797E; }}
      .phase-kicker {{ font: 700 11px 'Avenir Next', 'Segoe UI', sans-serif; letter-spacing: 0.22em; text-transform: uppercase; fill: #7A8F92; }}
      .node-title {{ font: 700 22px 'Avenir Next', 'Segoe UI', sans-serif; fill: #163F48; }}
      .node-copy {{ font: 500 15px 'Avenir Next', 'Segoe UI', sans-serif; fill: #36565C; }}
      .small-copy {{ font: 500 13px 'Avenir Next', 'Segoe UI', sans-serif; fill: #50676D; }}
      .artifact-title {{ font: 700 18px 'Avenir Next', 'Segoe UI', sans-serif; fill: #163F48; }}
      .artifact-copy {{ font: 500 13px 'Avenir Next', 'Segoe UI', sans-serif; fill: #4F666B; }}
      .node-box {{ filter: url(#shadow); }}
      .pipe {{ stroke: #1A6C72; stroke-width: 4; fill: none; stroke-linecap: round; stroke-linejoin: round; }}
      .pipe-accent {{ stroke: #E99B3F; stroke-width: 4; fill: none; stroke-linecap: round; stroke-linejoin: round; }}
      .pipe-soft {{ stroke: #7B9396; stroke-width: 3; fill: none; stroke-dasharray: 10 10; stroke-linecap: round; stroke-linejoin: round; }}
      .loop-label {{ font: 700 12px 'Avenir Next', 'Segoe UI', sans-serif; letter-spacing: 0.12em; text-transform: uppercase; }}
    </style>
    <marker id="arrowTeal" markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" refX="6.2" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 z" fill="#1A6C72"/>
    </marker>
    <marker id="arrowOrange" markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" refX="6.2" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 z" fill="#E99B3F"/>
    </marker>
    <marker id="arrowSoft" markerWidth="8" markerHeight="8" markerUnits="userSpaceOnUse" refX="6.2" refY="4" orient="auto">
      <path d="M0,0 L8,4 L0,8 z" fill="#7B9396"/>
    </marker>
  </defs>

  <rect width="1600" height="900" fill="url(#bgWash)"/>
  <rect x="0" y="0" width="1600" height="10" fill="url(#accentLine)"/>

  <text x="80" y="84" class="phase-kicker">FT09 runtime story</text>
  <text x="80" y="136" class="title">main.py to persisted trace, with the real feedback loops highlighted</text>
  <text x="80" y="172" class="subtitle">The viewer is built from what this runtime actually stores: recording frames, GoalBoard summaries, and run-log reflexion lines.</text>

  <g class="node-box">
    <rect x="88" y="234" rx="28" ry="28" width="228" height="118" fill="#FFFFFF" stroke="#DCE6E5"/>
    <text x="116" y="270" class="phase-kicker">Entrypoint</text>
    <text x="116" y="304" class="node-title">main.py</text>
    <text x="116" y="332" class="node-copy">load env, choose agent,</text>
    <text x="116" y="354" class="node-copy">pick local/API game list</text>
  </g>

  <g class="node-box">
    <rect x="354" y="234" rx="28" ry="28" width="232" height="118" fill="#F8FCFB" stroke="#D1E5E3"/>
    <text x="382" y="270" class="phase-kicker">Orchestration</text>
    <text x="382" y="304" class="node-title">Swarm.main()</text>
    <text x="382" y="332" class="node-copy">build ArcgenticaSimple</text>
    <text x="382" y="354" class="node-copy">and attach ARC environment</text>
  </g>

  <g class="node-box">
    <rect x="626" y="214" rx="32" ry="32" width="332" height="164" fill="#FFFFFF" stroke="#DCE6E5"/>
    <text x="656" y="252" class="phase-kicker">Per-turn loop</text>
    <text x="656" y="292" class="node-title">ArcgenticaSimple._run()</text>
    <text x="656" y="322" class="node-copy">RESET, SemanticPackets.current(),</text>
    <text x="656" y="346" class="node-copy">spawn Reflexion / M1 / M2,</text>
    <text x="656" y="370" class="node-copy">submit_action_raw(), update memory</text>
  </g>

  <g class="node-box">
    <rect x="1018" y="234" rx="28" ry="28" width="218" height="118" fill="#F7FCFC" stroke="#D1E6E7"/>
    <text x="1046" y="270" class="phase-kicker">Sense</text>
    <text x="1046" y="304" class="node-title">SemanticPackets</text>
    <text x="1046" y="332" class="node-copy">visible regions, render rows,</text>
    <text x="1046" y="354" class="node-copy">joint-neighbor state</text>
  </g>

  <g class="node-box">
    <rect x="1268" y="214" rx="28" ry="28" width="252" height="164" fill="#FFF8EF" stroke="#F0D4AD"/>
    <text x="1298" y="252" class="phase-kicker">Decide</text>
    <text x="1298" y="290" class="node-title">Reflexion -> M1 -> M2</text>
    <text x="1298" y="320" class="node-copy">corrective sentence,</text>
    <text x="1298" y="344" class="node-copy">hypothesis cards,</text>
    <text x="1298" y="368" class="node-copy">chosen action sequence</text>
  </g>

  <g class="node-box">
    <rect x="1198" y="478" rx="28" ry="28" width="322" height="128" fill="#F7FCFB" stroke="#D5E6E1"/>
    <text x="1228" y="514" class="phase-kicker">Execute and score</text>
    <text x="1228" y="548" class="node-title">environment step + observer</text>
    <text x="1228" y="576" class="node-copy">extract_observation(), evaluate_falsifier(),</text>
    <text x="1228" y="600" class="node-copy">level rise, region anchor, dominant transition</text>
  </g>

  <g class="node-box">
    <rect x="812" y="466" rx="32" ry="32" width="330" height="152" fill="#FFFFFF" stroke="#DCE6E5"/>
    <text x="842" y="504" class="phase-kicker">Working memory</text>
    <text x="842" y="542" class="node-title">GoalBoard + DiffMemory</text>
    <text x="842" y="572" class="node-copy">choice_history, observation_log,</text>
    <text x="842" y="596" class="node-copy">skill bank, cross-level confirms,</text>
    <text x="842" y="620" class="node-copy">region click stats, stagnation window</text>
  </g>

  <path d="M316 293 H354" fill="none" stroke="#1A6C72" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" marker-end="url(#arrowTeal)"/>
  <path d="M586 293 H626" fill="none" stroke="#1A6C72" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" marker-end="url(#arrowTeal)"/>
  <path d="M958 293 H1018" fill="none" stroke="#1A6C72" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" marker-end="url(#arrowTeal)"/>
  <path d="M1236 293 H1268" fill="none" stroke="#E99B3F" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" marker-end="url(#arrowOrange)"/>
  <path d="M1424 378 V448 H1360" fill="none" stroke="#E99B3F" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" marker-end="url(#arrowOrange)"/>
  <path d="M1198 542 H1142" fill="none" stroke="#1A6C72" stroke-width="4" stroke-linecap="round" stroke-linejoin="round" marker-end="url(#arrowTeal)"/>

  <path d="M980 620 L980 704 L1292 704 L1292 646" fill="none" stroke="#7B9396" stroke-width="3" stroke-dasharray="10 10" stroke-linecap="round" stroke-linejoin="round"/>
  <text x="1028" y="690" class="loop-label soft">persistence loop</text>

  <path d="M812 542 L724 542 L676 424" fill="none" stroke="#1A6C72" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
  <text x="592" y="430" class="loop-label" fill="#1A6C72">next-turn memory</text>

  <path d="M900 466 L900 406 L1268 406 L1268 390" fill="none" stroke="#E99B3F" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
  <text x="1012" y="424" class="loop-label" fill="#C97C28">reflexion and skill feedback</text>

  <g class="node-box">
    <rect x="78" y="510" rx="24" ry="24" width="250" height="120" fill="#FFFFFF" stroke="#D9E3E2"/>
    <text x="108" y="548" class="artifact-title">recording.jsonl</text>
    <text x="108" y="576" class="artifact-copy">raw frame after each submitted action</text>
    <text x="108" y="598" class="artifact-copy">plus action_input and level count</text>
  </g>

  <g class="node-box">
    <rect x="362" y="510" rx="24" ry="24" width="252" height="120" fill="#FFFFFF" stroke="#D9E3E2"/>
    <text x="392" y="548" class="artifact-title">goal_board.json</text>
    <text x="392" y="576" class="artifact-copy">selected predicate, verdict,</text>
    <text x="392" y="598" class="artifact-copy">skill bank, cross-level memory</text>
  </g>

  <g class="node-box">
    <rect x="648" y="510" rx="24" ry="24" width="130" height="120" fill="#FFFFFF" stroke="#D9E3E2"/>
    <text x="676" y="548" class="artifact-title">run log</text>
    <text x="676" y="576" class="artifact-copy">reflexion text</text>
    <text x="676" y="598" class="artifact-copy">action counts</text>
  </g>

  <path d="M812 588 C760 588 760 598 778 598" fill="none" stroke="#7B9396" stroke-width="3" stroke-dasharray="10 10" stroke-linecap="round" stroke-linejoin="round" marker-end="url(#arrowSoft)"/>
  <path d="M812 574 C712 574 638 570 614 570" fill="none" stroke="#7B9396" stroke-width="3" stroke-dasharray="10 10" stroke-linecap="round" stroke-linejoin="round" marker-end="url(#arrowSoft)"/>
  <path d="M812 558 C592 558 416 558 328 558" fill="none" stroke="#7B9396" stroke-width="3" stroke-dasharray="10 10" stroke-linecap="round" stroke-linejoin="round" marker-end="url(#arrowSoft)"/>

  <text x="80" y="742" class="phase-kicker">What the visualizer can show faithfully</text>
  <text x="80" y="786" class="small-copy">1. action and state change from recording frames</text>
  <text x="80" y="812" class="small-copy">2. chosen hypothesis, verdict, and final skill memory from GoalBoard</text>
  <text x="80" y="838" class="small-copy">3. reflexion directives from run logs when they were persisted</text>
  <text x="856" y="742" class="phase-kicker">What is still missing in current persistence</text>
  <text x="856" y="786" class="small-copy">raw M1 candidate set for each turn</text>
  <text x="856" y="812" class="small-copy">full raw M2 JSON including expected_step_diffs</text>
  <text x="856" y="838" class="small-copy">complete semantic-packet snapshot per turn</text>

  <text x="80" y="878" class="footer">Code anchors: main.py, agents/swarm.py, agents/templates/agentica_simple/agent.py, goal_board.py, reflexion_prompt.py</text>
</svg>
"""


def render_flow_html(svg_markup: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>FT09 Runtime Flow v5</title>
  <style>
    @page {{
      size: 16in 9in;
      margin: 0;
    }}
    html, body {{
      margin: 0;
      padding: 0;
      width: 100%;
      height: 100%;
      background: #f4f2ea;
    }}
    body {{
      display: flex;
      align-items: center;
      justify-content: center;
    }}
    svg {{
      width: 100%;
      height: 100%;
      display: block;
    }}
  </style>
</head>
<body>
{svg_markup}
</body>
</html>
"""


def write_review_note(payload: dict[str, Any]) -> None:
    manifest = payload["manifest"]
    default = payload["bundles"][payload["default_run_id"]]
    lines = [
        "# FT09 Visualization Review",
        "",
        f"- Generated at: {payload['generated_at']}",
        f"- Default run: `{default['id']}`",
        f"- Default turn: `{default['default_turn_index']}`",
        "- Acceptance bar:",
        "  recorded vs derived vs missing must be visually explicit",
        "  level progress and reflexion events must be scannable within 5 seconds",
        "  the first screen must show action, state change, and selected predicate together",
        "",
        "## Included Runs",
    ]
    for item in manifest:
        lines.append(
            f"- `{item['id']}`: max {item['max_level']}/6, {item['turns_total']} turns, {item['coverage_pct']}% rationale coverage"
        )
    REVIEW_NOTE.write_text("\n".join(lines) + "\n", encoding="utf-8")


def build_payload(*, max_runs: int, default_run: str | None) -> dict[str, Any]:
    active_ns = active_namespace()
    bundles = choose_bundles(active_ns=active_ns, max_runs=max_runs)
    if not bundles:
        raise RuntimeError(f"No FT09 bundles could be built under simple_logs/{GAME_ID}/")
    bundle_map = {}
    manifest = []
    for bundle in bundles:
        bundle["updated_at"] = ts_to_text(bundle["mtime"])
        bundle_map[bundle["id"]] = bundle
        manifest.append(
            {
                "id": bundle["id"],
                "label": bundle["title"],
                "max_level": bundle["stats"]["max_level"],
                "turns_total": bundle["stats"]["turns_total"],
                "coverage_pct": int(round(bundle["stats"]["rationale_coverage"] * 100)),
            }
        )
    default_run_id = default_run if default_run in bundle_map else bundles[0]["id"]
    return {
        "generated_at": datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC"),
        "active_namespace": active_ns,
        "default_run_id": default_run_id,
        "manifest": manifest,
        "bundles": bundle_map,
    }


def export_flow_pdf(*, skip_pdf: bool) -> None:
    if skip_pdf:
        return
    subprocess.run(
        ["weasyprint", str(FLOW_HTML), str(FLOW_PDF)],
        cwd=REPO_ROOT,
        check=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FT09 runtime flow and trace explorer artifacts.")
    parser.add_argument("--max-runs", type=int, default=MAX_RUNS_DEFAULT)
    parser.add_argument("--default-run", type=str, default=None)
    parser.add_argument("--skip-pdf", action="store_true")
    args = parser.parse_args()

    PRESENTATION_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    payload = build_payload(max_runs=args.max_runs, default_run=args.default_run)
    VIEWER_DATA_JSON.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    VIEWER_HTML.write_text(render_trace_html(payload), encoding="utf-8")

    flow_svg = render_flow_svg()
    FLOW_SVG.write_text(flow_svg, encoding="utf-8")
    FLOW_HTML.write_text(render_flow_html(flow_svg), encoding="utf-8")
    export_flow_pdf(skip_pdf=args.skip_pdf)
    write_review_note(payload)

    print(f"Wrote {repo_rel(VIEWER_HTML)}")
    print(f"Wrote {repo_rel(VIEWER_DATA_JSON)}")
    print(f"Wrote {repo_rel(FLOW_SVG)}")
    print(f"Wrote {repo_rel(FLOW_HTML)}")
    if not args.skip_pdf:
        print(f"Wrote {repo_rel(FLOW_PDF)}")
    print(f"Wrote {repo_rel(REVIEW_NOTE)}")


if __name__ == "__main__":
    main()
