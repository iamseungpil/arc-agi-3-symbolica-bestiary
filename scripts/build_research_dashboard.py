#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_GAME = "ls20-cb3b57cc"
DEFAULT_NAMESPACE = "arcgentica_skillread_all_20260416k"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def detect_type(payload: dict[str, Any]) -> str:
    body = payload.get("body")
    subskills = payload.get("subskills")
    if isinstance(subskills, list) and subskills:
        return "hierarchical"
    if isinstance(body, list) and body and all(isinstance(item, str) and item.startswith("ACTION") for item in body):
        return "exact-spine"
    return "structured"


def score_skill(item: dict[str, Any]) -> float:
    if "score" in item and isinstance(item["score"], (int, float)):
        return float(item["score"])
    return (
        2.0 * float(item.get("times_referenced", 0))
        + 1.5 * float(item.get("observed_support", 0))
        + 1.0 * float(item.get("informative_hits", 0))
        + 0.5 * float(item.get("times_linked_to_surprise", 0))
        + 0.75 * float(item.get("times_selected", 0))
        + 0.5 * float(item.get("reset_intercepts", 0))
        + 0.25 * float(item.get("depth", 0))
        + 0.25 * float(item.get("revision_count", 0))
    )


def preview_payload(payload: dict[str, Any]) -> str:
    for key in ("controller", "body", "steps", "rule", "policy", "trigger", "precondition", "expected_effect"):
        value = payload.get(key)
        if not value:
            continue
        if isinstance(value, list):
            text = " -> ".join(str(item) for item in value[:10])
        else:
            text = str(value)
        if len(text) > 220:
            return text[:217] + "..."
        return text
    return ""


def extract_probe_fields(reason: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for key in ("current_hypothesis", "next_step"):
        patterns = [
            rf"'{key}': '([^']+)'",
            rf'"{key}": "([^"]+)"',
        ]
        for pattern in patterns:
            match = re.search(pattern, reason)
            if match:
                fields[key] = match.group(1)
                break
    return fields


def load_latest_episode(actions_path: Path) -> list[dict[str, Any]]:
    if not actions_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with actions_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except Exception:
                continue
    if not rows:
        return []
    start = 0
    for idx, row in enumerate(rows):
        if row.get("action") == "RESET":
            start = idx
    return rows[start:]


def load_recent_autoresearch(tsv_path: Path, limit: int = 8) -> list[dict[str, str]]:
    if not tsv_path.exists():
        return []
    with tsv_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return rows[-limit:]


def relpath(path: Path, start: Path) -> str:
    try:
        return str(path.relative_to(start))
    except Exception:
        return str(path)


def build_html(
    *,
    namespace: str,
    game_id: str,
    summary: dict[str, Any],
    report: dict[str, Any],
    skills: list[dict[str, Any]],
    world_model: dict[str, Any],
    latest_episode: list[dict[str, Any]],
    autoresearch_rows: list[dict[str, str]],
    artifact_paths: dict[str, Path],
) -> str:
    finish_status = summary.get("finish_status", {}) if isinstance(summary.get("finish_status"), dict) else {}
    stop_reason = str(finish_status.get("reason", "") or "")
    probe_fields = extract_probe_fields(stop_reason)
    active_modules = summary.get("active_modules") or []
    best_level = summary.get("best_level", "?")
    actions = summary.get("actions", "?")
    resets = summary.get("reset_count", "?")
    latest_draft = (world_model.get("world_drafts") or [{}])[-1] if world_model.get("world_drafts") else {}

    rendered_skills = []
    for item in sorted(skills, key=score_skill, reverse=True)[:10]:
        payload = item.get("payload", {})
        skill_type = detect_type(payload)
        rendered_skills.append(
            f"""
            <tr>
              <td>{html.escape(str(payload.get("name", "unnamed")))}</td>
              <td><span class="pill {skill_type}">{skill_type}</span></td>
              <td>{score_skill(item):.1f}</td>
              <td>{int(item.get("observed_support", 0))}</td>
              <td>{int(item.get("times_selected", 0))}</td>
              <td>{int(item.get("revision_count", 0))}</td>
              <td>{html.escape(preview_payload(payload) or "-")}</td>
            </tr>
            """
        )

    timeline = []
    for row in latest_episode[:24]:
        action = str(row.get("action", "?"))
        count = row.get("count", "?")
        timeline.append(
            f'<div class="action-chip"><span class="action-name">{html.escape(action)}</span><span class="action-count">#{html.escape(str(count))}</span></div>'
        )

    recent_rows_html = []
    for row in autoresearch_rows:
        recent_rows_html.append(
            f"""
            <tr>
              <td>{html.escape(row.get("iteration", ""))}</td>
              <td>{html.escape(row.get("hypothesis", ""))}</td>
              <td><span class="pill {'keep' if row.get('decision') == 'keep' else 'discard'}">{html.escape(row.get("decision", ""))}</span></td>
              <td>{html.escape((row.get("metric_summary", "")[:160] + "...") if len(row.get("metric_summary", "")) > 160 else row.get("metric_summary", ""))}</td>
            </tr>
            """
        )

    artifact_links = []
    presentation_dir = artifact_paths["output"].parent
    for label, path in artifact_paths.items():
        if label == "output":
            continue
        if not path.exists():
            continue
        artifact_links.append(
            f'<li><a href="{html.escape(relpath(path, presentation_dir))}">{html.escape(label)}</a></li>'
        )

    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Research Dashboard</title>
  <style>
    :root {{
      --bg: #0f172a;
      --panel: #111827;
      --panel-2: #172033;
      --border: #263247;
      --text: #ecf2ff;
      --muted: #9fb0cf;
      --accent: #4ade80;
      --warn: #f59e0b;
      --bad: #fb7185;
      --blue: #60a5fa;
      --mono: "SF Mono", "Cascadia Code", "Fira Code", monospace;
      --sans: "Inter", "Segoe UI", sans-serif;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      background:
        radial-gradient(circle at top left, rgba(96,165,250,0.18), transparent 28%),
        radial-gradient(circle at top right, rgba(74,222,128,0.16), transparent 24%),
        linear-gradient(180deg, #08101f 0%, var(--bg) 100%);
      color: var(--text);
      font-family: var(--sans);
    }}
    .page {{ max-width: 1400px; margin: 0 auto; padding: 24px; }}
    .hero {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: end;
      margin-bottom: 20px;
    }}
    h1 {{ margin: 0; font-size: 30px; }}
    .sub {{ color: var(--muted); margin-top: 6px; }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(12, 1fr);
      gap: 16px;
    }}
    .card {{
      background: linear-gradient(180deg, rgba(23,32,51,0.96), rgba(15,23,42,0.96));
      border: 1px solid var(--border);
      border-radius: 16px;
      padding: 18px;
      box-shadow: 0 20px 40px rgba(0, 0, 0, 0.22);
    }}
    .span-3 {{ grid-column: span 3; }}
    .span-4 {{ grid-column: span 4; }}
    .span-5 {{ grid-column: span 5; }}
    .span-6 {{ grid-column: span 6; }}
    .span-7 {{ grid-column: span 7; }}
    .span-8 {{ grid-column: span 8; }}
    .span-12 {{ grid-column: span 12; }}
    .metric-label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; }}
    .metric-value {{ font-size: 36px; font-weight: 700; margin-top: 8px; }}
    .metric-note {{ color: var(--muted); margin-top: 8px; line-height: 1.5; }}
    .pill {{
      display: inline-flex;
      align-items: center;
      gap: 6px;
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 12px;
      border: 1px solid var(--border);
      color: var(--text);
      background: rgba(255,255,255,0.04);
    }}
    .pill.keep {{ color: var(--accent); }}
    .pill.discard {{ color: var(--bad); }}
    .pill.structured {{ color: var(--blue); }}
    .pill.hierarchical {{ color: var(--accent); }}
    .pill.exact-spine {{ color: var(--warn); }}
    h2 {{ margin: 0 0 12px; font-size: 18px; }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      font-family: var(--mono);
      font-size: 12px;
      line-height: 1.55;
      background: rgba(255,255,255,0.03);
      border: 1px solid var(--border);
      border-radius: 12px;
      padding: 14px;
      margin: 0;
      overflow: auto;
      max-height: 320px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      text-align: left;
      padding: 10px 8px;
      border-top: 1px solid rgba(255,255,255,0.06);
      vertical-align: top;
    }}
    th {{ color: var(--muted); font-weight: 600; border-top: none; }}
    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .action-chip {{
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.04);
      border-radius: 12px;
      padding: 8px 10px;
      min-width: 90px;
    }}
    .action-name {{
      display: block;
      font-weight: 700;
      font-family: var(--mono);
      color: var(--text);
    }}
    .action-count {{
      display: block;
      color: var(--muted);
      margin-top: 3px;
      font-size: 12px;
    }}
    a {{ color: #93c5fd; text-decoration: none; }}
    ul {{ margin: 0; padding-left: 18px; }}
    .callout {{
      border-left: 3px solid var(--warn);
      padding-left: 12px;
      color: var(--text);
      line-height: 1.6;
    }}
    @media (max-width: 960px) {{
      .span-3, .span-4, .span-5, .span-6, .span-7, .span-8, .span-12 {{ grid-column: span 12; }}
      .hero {{ flex-direction: column; align-items: start; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <div class="hero">
      <div>
        <h1>ARC Research Dashboard</h1>
        <div class="sub">{html.escape(game_id)} / {html.escape(namespace)}</div>
        <div class="sub">Generated {generated_at}</div>
      </div>
      <div class="pill">Modules: {html.escape(", ".join(active_modules) if active_modules else "none")}</div>
    </div>

    <div class="grid">
      <section class="card span-3">
        <div class="metric-label">Best Level</div>
        <div class="metric-value">{html.escape(str(best_level))}</div>
        <div class="metric-note">Latest bounded summary result.</div>
      </section>
      <section class="card span-3">
        <div class="metric-label">Non-reset Actions</div>
        <div class="metric-value">{html.escape(str(actions))}</div>
        <div class="metric-note">Current failure mode is shallow stop after a short probe.</div>
      </section>
      <section class="card span-3">
        <div class="metric-label">Reset Count</div>
        <div class="metric-value">{html.escape(str(resets))}</div>
        <div class="metric-note">Useful to distinguish startup loop issues from deeper search.</div>
      </section>
      <section class="card span-3">
        <div class="metric-label">Skill Library</div>
        <div class="metric-value">{len(skills)}</div>
        <div class="metric-note">Watch structured vs exact-spine balance, not just raw count.</div>
      </section>

      <section class="card span-7">
        <h2>Why It Stopped</h2>
        <div class="callout">
          <strong>Status:</strong> {html.escape(str(finish_status.get("status", "?")))}<br />
          <strong>Interpretation:</strong> the harness accepted a probe report as terminal even though it contained a concrete next step.
        </div>
        <div style="height: 12px"></div>
        <pre>{html.escape(stop_reason or "No finish reason found.")}</pre>
      </section>

      <section class="card span-5">
        <h2>Probe Extraction</h2>
        <table>
          <tr><th>Field</th><th>Value</th></tr>
          <tr><td>current_hypothesis</td><td>{html.escape(probe_fields.get("current_hypothesis", "-"))}</td></tr>
          <tr><td>next_step</td><td>{html.escape(probe_fields.get("next_step", "-"))}</td></tr>
          <tr><td>report_ok</td><td>{html.escape(str(report.get("ok", False)))}</td></tr>
          <tr><td>timed_out</td><td>{html.escape(str(report.get("timed_out", False)))}</td></tr>
        </table>
      </section>

      <section class="card span-12">
        <h2>Latest Action Episode</h2>
        <div class="chips">
          {''.join(timeline) or '<span class="sub">No action log found.</span>'}
        </div>
      </section>

      <section class="card span-8">
        <h2>Top Skills</h2>
        <table>
          <tr>
            <th>Name</th>
            <th>Type</th>
            <th>Score</th>
            <th>Support</th>
            <th>Selected</th>
            <th>Revs</th>
            <th>Preview</th>
          </tr>
          {''.join(rendered_skills) or '<tr><td colspan="7">No skills found.</td></tr>'}
        </table>
      </section>

      <section class="card span-4">
        <h2>World Model</h2>
        <table>
          <tr><th>Metric</th><th>Value</th></tr>
          <tr><td>drafts</td><td>{len(world_model.get("world_drafts", []))}</td></tr>
          <tr><td>structured_predictions</td><td>{html.escape(str(world_model.get("structured_predictions", 0)))}</td></tr>
          <tr><td>matches</td><td>{html.escape(str(world_model.get("scored_prediction_matches", 0)))}</td></tr>
          <tr><td>mismatches</td><td>{html.escape(str(world_model.get("scored_prediction_mismatches", 0)))}</td></tr>
          <tr><td>prior_suggestions</td><td>{html.escape(str(world_model.get("prior_suggestions", 0)))}</td></tr>
          <tr><td>prior_overrides</td><td>{html.escape(str(world_model.get("prior_overrides", 0)))}</td></tr>
        </table>
        <div style="height: 12px"></div>
        <pre>{html.escape(str(latest_draft.get("body", "No retained world-model draft.")))}</pre>
      </section>

      <section class="card span-7">
        <h2>Recent Autoresearch Rows</h2>
        <table>
          <tr>
            <th>Iteration</th>
            <th>Hypothesis</th>
            <th>Decision</th>
            <th>Metric Summary</th>
          </tr>
          {''.join(recent_rows_html) or '<tr><td colspan="4">No autoresearch log rows found.</td></tr>'}
        </table>
      </section>

      <section class="card span-5">
        <h2>Artifacts</h2>
        <ul>
          {''.join(artifact_links) or '<li>No artifact links found.</li>'}
        </ul>
        <div style="height: 16px"></div>
        <div class="metric-note">
          Visualizer design principle: the first screen should explain <em>why the run stopped</em>,
          <em>which skill or world prior shaped the last branch</em>, and <em>whether the library is growing in a reusable direction</em>.
          Raw frames are still useful, but they should be secondary to stop diagnostics, action chronology, and abstraction quality.
        </div>
      </section>
    </div>
  </div>
</body>
</html>
"""


###############################################################################
# v3 additions (transition accuracy, MCTS vs BFS, library breakdown, per-turn,
# wake events). These are pure-additions; they do not modify build_html.
###############################################################################


V3_PLACEHOLDER_NOTE = (
    '<div class="metric-note" style="opacity:0.6">'
    "No v3 artifact detected; panel skipped."
    "</div>"
)


def _svg_placeholder(message: str, width: int = 560, height: int = 180) -> str:
    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        'xmlns="http://www.w3.org/2000/svg" role="img">'
        f'<rect x="0" y="0" width="{width}" height="{height}" fill="none" '
        'stroke="#263247" stroke-dasharray="4 4" rx="10" />'
        f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" '
        'fill="#9fb0cf" font-family="Inter, sans-serif" font-size="13">'
        f"{html.escape(message)}</text></svg>"
    )


def render_transition_accuracy_svg(world_model: dict[str, Any]) -> str:
    """Line chart of cumulative unit_tests_passed / unit_tests_seen.

    Individual entries in ``world_model["unit_tests"]`` do not carry a per-step
    pass flag, so we render the count of unit tests observed over time along
    with a horizontal reference line at the final ``transition_accuracy``.
    """

    unit_tests = world_model.get("unit_tests") or []
    final_passed = world_model.get("unit_tests_passed")
    final_seen = world_model.get("unit_tests_seen")
    final_acc = world_model.get("transition_accuracy")

    if not unit_tests or final_seen in (None, 0):
        return _svg_placeholder("No unit_tests entries recorded.")

    width, height = 640, 220
    pad_l, pad_r, pad_t, pad_b = 44, 18, 18, 30
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    n = len(unit_tests)
    # Points: x = i+1, y = final_acc (constant — we only know end-state).
    # To make the curve informative, we render the cumulative "new_family" and
    # "branch_escape" ratios over time, using transition_accuracy as a reference.
    new_family_running = 0
    branch_running = 0
    acc_points: list[tuple[float, float]] = []
    nf_points: list[tuple[float, float]] = []
    be_points: list[tuple[float, float]] = []
    for i, entry in enumerate(unit_tests):
        if entry.get("new_family"):
            new_family_running += 1
        if entry.get("branch_escape"):
            branch_running += 1
        x_frac = (i + 1) / n
        x_px = pad_l + x_frac * plot_w
        nf_points.append((x_px, pad_t + plot_h - (new_family_running / n) * plot_h))
        be_points.append((x_px, pad_t + plot_h - (branch_running / n) * plot_h))
        acc_points.append((x_px, pad_t + plot_h - float(final_acc or 0.0) * plot_h))

    def _polyline(points: list[tuple[float, float]], color: str) -> str:
        pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in points)
        return (
            f'<polyline points="{pts}" fill="none" stroke="{color}" '
            'stroke-width="2" stroke-linejoin="round" />'
        )

    # Axes + grid
    grid = []
    for frac in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = pad_t + plot_h - frac * plot_h
        grid.append(
            f'<line x1="{pad_l}" y1="{y:.1f}" x2="{pad_l + plot_w}" y2="{y:.1f}" '
            'stroke="#1f2b40" stroke-width="1" />'
        )
        grid.append(
            f'<text x="{pad_l - 6}" y="{y + 3:.1f}" text-anchor="end" '
            f'fill="#9fb0cf" font-size="10" font-family="Inter, sans-serif">{frac:.2f}</text>'
        )

    legend = (
        f'<g font-family="Inter, sans-serif" font-size="11" fill="#ecf2ff">'
        f'<rect x="{pad_l + 6}" y="{pad_t + 6}" width="10" height="10" fill="#4ade80"/>'
        f'<text x="{pad_l + 22}" y="{pad_t + 15}">transition_accuracy = {float(final_acc or 0):.3f}</text>'
        f'<rect x="{pad_l + 210}" y="{pad_t + 6}" width="10" height="10" fill="#60a5fa"/>'
        f'<text x="{pad_l + 226}" y="{pad_t + 15}">cum. new_family / n</text>'
        f'<rect x="{pad_l + 360}" y="{pad_t + 6}" width="10" height="10" fill="#f59e0b"/>'
        f'<text x="{pad_l + 376}" y="{pad_t + 15}">cum. branch_escape / n</text>'
        f'</g>'
    )

    xlabel = (
        f'<text x="{pad_l + plot_w / 2:.1f}" y="{height - 8}" text-anchor="middle" '
        f'fill="#9fb0cf" font-size="11" font-family="Inter, sans-serif">'
        f'unit_test index (n={n}, passed={final_passed}/{final_seen})</text>'
    )

    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        'xmlns="http://www.w3.org/2000/svg" role="img">'
        f'<rect x="{pad_l}" y="{pad_t}" width="{plot_w}" height="{plot_h}" '
        'fill="rgba(255,255,255,0.02)" stroke="#263247" />'
        + "".join(grid)
        + _polyline(acc_points, "#4ade80")
        + _polyline(nf_points, "#60a5fa")
        + _polyline(be_points, "#f59e0b")
        + legend
        + xlabel
        + "</svg>"
    )


def render_advantage_bar_svg(planner_state: dict[str, Any]) -> str:
    """Bar chart of MCTS minus BFS advantage; positive green, negative red."""

    advantages = planner_state.get("recent_advantages") or []
    if not advantages:
        return _svg_placeholder("planner_state.json has no recent_advantages.")

    width, height = 640, 220
    pad_l, pad_r, pad_t, pad_b = 44, 18, 18, 30
    plot_w = width - pad_l - pad_r
    plot_h = height - pad_t - pad_b

    values = [float(v) for v in advantages]
    max_abs = max((abs(v) for v in values), default=1.0) or 1.0
    n = len(values)
    bar_w = plot_w / n
    zero_y = pad_t + plot_h / 2

    bars: list[str] = []
    for i, v in enumerate(values):
        x = pad_l + i * bar_w + bar_w * 0.12
        w = bar_w * 0.76
        h = (abs(v) / max_abs) * (plot_h / 2)
        y = zero_y - h if v >= 0 else zero_y
        color = "#4ade80" if v >= 0 else "#fb7185"
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{w:.1f}" height="{h:.1f}" '
            f'fill="{color}" opacity="0.9"><title>call #{i + 1}: {v:+.3f}</title></rect>'
        )

    mean_v = sum(values) / n
    axis = (
        f'<line x1="{pad_l}" y1="{zero_y:.1f}" x2="{pad_l + plot_w}" y2="{zero_y:.1f}" '
        'stroke="#9fb0cf" stroke-width="1" />'
    )
    ticks = []
    for frac, label in ((-1.0, f"-{max_abs:.2f}"), (0.0, "0"), (1.0, f"+{max_abs:.2f}")):
        y = zero_y - frac * (plot_h / 2)
        ticks.append(
            f'<text x="{pad_l - 6}" y="{y + 3:.1f}" text-anchor="end" '
            f'fill="#9fb0cf" font-size="10" font-family="Inter, sans-serif">{label}</text>'
        )

    legend = (
        f'<text x="{pad_l + 6}" y="{pad_t + 14}" font-family="Inter, sans-serif" '
        f'font-size="11" fill="#ecf2ff">MCTS - BFS reward ('
        f'n={n}, mean={mean_v:+.3f})</text>'
    )
    xlabel = (
        f'<text x="{pad_l + plot_w / 2:.1f}" y="{height - 8}" text-anchor="middle" '
        f'fill="#9fb0cf" font-size="11" font-family="Inter, sans-serif">'
        f'imagination call index</text>'
    )

    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        'xmlns="http://www.w3.org/2000/svg" role="img">'
        + "".join(bars)
        + axis
        + "".join(ticks)
        + legend
        + xlabel
        + "</svg>"
    )


def render_library_breakdown_svg(skills: list[dict[str, Any]]) -> str:
    """Stacked horizontal bar of skill `kind` counts."""

    if not skills:
        return _svg_placeholder("dreamcoder_library.json empty or missing.")

    counts: dict[str, int] = {}
    for item in skills:
        payload = item.get("payload", {}) if isinstance(item, dict) else {}
        kind = str(payload.get("kind") or "free")
        counts[kind] = counts.get(kind, 0) + 1

    total = sum(counts.values()) or 1
    palette = {
        "abstract_primitive": "#60a5fa",
        "mcts_proposal": "#4ade80",
        "wrapper": "#f59e0b",
        "free": "#a78bfa",
    }
    ordered = sorted(counts.items(), key=lambda kv: -kv[1])

    width, height = 560, 170
    bar_x, bar_y, bar_w, bar_h = 30, 40, width - 60, 38
    segments = []
    legend_items = []
    cursor = bar_x
    for i, (kind, cnt) in enumerate(ordered):
        frac = cnt / total
        seg_w = frac * bar_w
        color = palette.get(kind, "#9fb0cf")
        segments.append(
            f'<rect x="{cursor:.1f}" y="{bar_y}" width="{seg_w:.1f}" height="{bar_h}" '
            f'fill="{color}"><title>{html.escape(kind)}: {cnt}</title></rect>'
        )
        if seg_w > 36:
            segments.append(
                f'<text x="{cursor + seg_w / 2:.1f}" y="{bar_y + bar_h / 2 + 4:.1f}" '
                f'text-anchor="middle" fill="#0f172a" font-size="12" '
                f'font-family="Inter, sans-serif" font-weight="700">{cnt}</text>'
            )
        cursor += seg_w
        ly = bar_y + bar_h + 22 + (i // 4) * 20
        lx = bar_x + (i % 4) * 140
        legend_items.append(
            f'<g><rect x="{lx}" y="{ly - 10}" width="12" height="12" fill="{color}"/>'
            f'<text x="{lx + 18}" y="{ly}" fill="#ecf2ff" font-size="12" '
            f'font-family="Inter, sans-serif">{html.escape(kind)} ({cnt})</text></g>'
        )

    title = (
        f'<text x="{bar_x}" y="{bar_y - 12}" fill="#9fb0cf" font-size="11" '
        f'font-family="Inter, sans-serif">total skills: {total}</text>'
    )

    return (
        f'<svg viewBox="0 0 {width} {height}" width="100%" height="{height}" '
        'xmlns="http://www.w3.org/2000/svg" role="img">'
        + title
        + "".join(segments)
        + "".join(legend_items)
        + "</svg>"
    )


def build_per_turn_timeline(
    latest_episode: list[dict[str, Any]],
    skills: list[dict[str, Any]],
) -> str:
    if not latest_episode:
        return '<div class="sub">No action log found.</div>'

    # Build index of first-action -> True for mcts_proposal skills with
    # times_selected > 0. Matches when a library mcts_proposal was committed.
    mcts_first_actions: set[str] = set()
    for item in skills:
        if not isinstance(item, dict):
            continue
        payload = item.get("payload", {}) or {}
        if str(payload.get("kind")) != "mcts_proposal":
            continue
        spine = payload.get("action_spine") or payload.get("body") or []
        if not isinstance(spine, list) or not spine:
            continue
        if int(item.get("times_selected", 0) or 0) <= 0:
            continue
        first = str(spine[0])
        if first:
            mcts_first_actions.add(first)

    rows = latest_episode[-20:]
    chips: list[str] = []
    for row in rows:
        action = str(row.get("action", "?"))
        count = row.get("count", "?")
        is_mcts = action in mcts_first_actions and action != "RESET"
        badge_color = "#4ade80" if is_mcts else "#9fb0cf"
        badge_text = "MCTS" if is_mcts else "LLM"
        chips.append(
            f'<div class="action-chip" style="border-color:{badge_color}55">'
            f'<span class="action-name">{html.escape(action)}</span>'
            f'<span class="action-count">#{html.escape(str(count))}</span>'
            f'<span class="action-count" style="color:{badge_color};margin-top:2px">'
            f'{badge_text}</span>'
            f"</div>"
        )

    return f'<div class="chips">{"".join(chips)}</div>'


def build_wake_events_panel(world_model: dict[str, Any]) -> str:
    notes = world_model.get("agent_env_notes") or []
    if not notes:
        return V3_PLACEHOLDER_NOTE

    collected: list[tuple[str, str]] = []
    for idx, note in enumerate(notes):
        if isinstance(note, dict):
            text = str(note.get("message") or note.get("text") or json.dumps(note))
            ts = str(note.get("timestamp") or note.get("ts") or "")
        else:
            text = str(note)
            ts = ""
        if ("Surprise" in text) or ("Wake" in text) or ("surprise" in text) or ("wake" in text):
            collected.append((ts or f"#{idx + 1}", text))

    if not collected:
        return '<div class="metric-note" style="opacity:0.7">No surprise/wake entries in agent_env_notes.</div>'

    collected = collected[-10:]
    rows = []
    for ts, text in collected:
        rows.append(
            f"<tr><td style='white-space:nowrap;color:#9fb0cf'>{html.escape(ts)}</td>"
            f"<td>{html.escape(text[:240] + ('...' if len(text) > 240 else ''))}</td></tr>"
        )
    return (
        "<table><tr><th>When</th><th>Note</th></tr>"
        + "".join(rows)
        + "</table>"
    )


def build_v3_section(
    *,
    world_model: dict[str, Any],
    planner_state: dict[str, Any],
    skills: list[dict[str, Any]],
    meta_harness: dict[str, Any],
    latest_episode: list[dict[str, Any]],
) -> str:
    """Assemble v3 panels as a self-contained grid section.

    Returns an HTML fragment that is appended after the existing grid inside
    ``.page``. No existing markup is modified.
    """

    transition_svg = render_transition_accuracy_svg(world_model) if world_model else _svg_placeholder("world_model.json missing.")
    advantage_svg = render_advantage_bar_svg(planner_state) if planner_state else _svg_placeholder("planner_state.json missing.")
    library_svg = render_library_breakdown_svg(skills)
    timeline_html = build_per_turn_timeline(latest_episode, skills)
    wake_html = build_wake_events_panel(world_model) if world_model else V3_PLACEHOLDER_NOTE

    run_history = meta_harness.get("run_history") if isinstance(meta_harness, dict) else None
    last_metrics: dict[str, Any] = {}
    if isinstance(run_history, list) and run_history:
        last_metrics = run_history[-1].get("metrics", {}) or {}
    fpc = last_metrics.get("falsification_probe_count", "-")
    bec = last_metrics.get("branch_escape_count", "-")
    rfp = last_metrics.get("repeated_family_penalty", "-")

    transition_accuracy = world_model.get("transition_accuracy") if world_model else None
    passed = world_model.get("unit_tests_passed") if world_model else None
    seen = world_model.get("unit_tests_seen") if world_model else None
    acc_text = f"{float(transition_accuracy):.3f}" if isinstance(transition_accuracy, (int, float)) else "-"

    return f"""
    <div class="grid" style="margin-top:20px">
      <section class="card span-12">
        <h2>v3 Panels</h2>
        <div class="sub">Plug-in panels sourced from v3 artifacts (world_model / planner_state / dreamcoder_library / meta_harness). Existing layout above is unchanged.</div>
      </section>

      <section class="card span-3">
        <div class="metric-label">Transition Accuracy</div>
        <div class="metric-value">{html.escape(acc_text)}</div>
        <div class="metric-note">unit_tests_passed / seen = {html.escape(str(passed))} / {html.escape(str(seen))}</div>
      </section>
      <section class="card span-3">
        <div class="metric-label">Falsification Probe</div>
        <div class="metric-value">{html.escape(str(fpc))}</div>
        <div class="metric-note">Last run_history entry (meta_harness).</div>
      </section>
      <section class="card span-3">
        <div class="metric-label">Branch Escape</div>
        <div class="metric-value">{html.escape(str(bec))}</div>
        <div class="metric-note">Goal: &gt; 0 after imagination warm-up.</div>
      </section>
      <section class="card span-3">
        <div class="metric-label">Repeated-family Penalty</div>
        <div class="metric-value">{html.escape(str(rfp))}</div>
        <div class="metric-note">Watch per-action ratio, not raw value.</div>
      </section>

      <section class="card span-7">
        <h2>Transition Accuracy Timeline</h2>
        {transition_svg}
        <div class="metric-note">Green reference line = final transition_accuracy. Blue/orange lines show cumulative new_family / branch_escape share over the logged unit_tests stream.</div>
      </section>
      <section class="card span-5">
        <h2>Skill Library Breakdown</h2>
        {library_svg}
        <div class="metric-note">Counts by payload.kind. abstract_primitive are seeded; mcts_proposal are auto-distilled from MCTS top paths; wrapper are sleep-refactor merges.</div>
      </section>

      <section class="card span-7">
        <h2>MCTS vs BFS Advantage</h2>
        {advantage_svg}
        <div class="metric-note">planner_state.recent_advantages — one bar per imagination call. H5 passes when mean advantage exceeds +1 stdev over the first 20 calls.</div>
      </section>
      <section class="card span-5">
        <h2>Wake / Surprise Events</h2>
        {wake_html}
      </section>

      <section class="card span-12">
        <h2>Per-turn Timeline (last 20 actions)</h2>
        {timeline_html}
        <div class="metric-note">Green &ldquo;MCTS&rdquo; badge = first action matches a committed mcts_proposal skill (times_selected &gt; 0). Grey &ldquo;LLM&rdquo; badge = chosen via the LLM agent path.</div>
      </section>
    </div>
    """


def _inject_v3_section(html_text: str, v3_fragment: str) -> str:
    """Insert v3 fragment before the outer `.page` closing without touching the
    rest of the document.
    """

    marker = "  </div>\n</body>"
    if marker in html_text:
        return html_text.replace(marker, v3_fragment + "\n" + marker, 1)
    # Fallback: just append before </body> if formatting changed.
    return html_text.replace("</body>", v3_fragment + "\n</body>", 1)


def _v3_files_exist(paths: dict[str, Path]) -> bool:
    return any(p.exists() for key, p in paths.items() if key in {"world_model", "planner_state"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--game-id", default=DEFAULT_GAME)
    parser.add_argument("--namespace", default=DEFAULT_NAMESPACE)
    parser.add_argument(
        "--output",
        default=str(ROOT / "presentation" / "research_dashboard_latest.html"),
    )
    parser.add_argument(
        "--include-v3-panels",
        default="auto",
        choices=["auto", "true", "false"],
        help="Render v3 panels. 'auto' (default) enables them iff v3 JSONs exist.",
    )
    args = parser.parse_args()

    summary_path = ROOT / "research_logs" / args.game_id / args.namespace / "summary.json"
    report_path = ROOT / "experiment_logs" / "reports" / f"{args.namespace}.json"
    library_path = ROOT / "research_logs" / "shared" / args.game_id / args.namespace / "dreamcoder_library.json"
    world_model_path = ROOT / "research_logs" / "shared" / args.game_id / args.namespace / "world_model.json"
    planner_state_path = ROOT / "research_logs" / "shared" / args.game_id / args.namespace / "planner_state.json"
    meta_harness_path = ROOT / "research_logs" / "shared" / args.game_id / args.namespace / "meta_harness.json"
    actions_path = ROOT / "actions_log" / args.game_id / "level_0.jsonl"
    autoresearch_path = ROOT / "experiment_logs" / "autoresearch_results.tsv"
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    recording_candidates = sorted(
        (ROOT / "recordings").glob(f"{args.game_id}.arcgentica*.recording.jsonl"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )[:3]

    latest_episode = load_latest_episode(actions_path)
    skills = load_json(library_path) if library_path.exists() else []
    world_model = load_json(world_model_path)
    planner_state = load_json(planner_state_path)
    meta_harness = load_json(meta_harness_path)

    html_text = build_html(
        namespace=args.namespace,
        game_id=args.game_id,
        summary=load_json(summary_path),
        report=load_json(report_path),
        skills=skills,
        world_model=world_model,
        latest_episode=latest_episode,
        autoresearch_rows=load_recent_autoresearch(autoresearch_path),
        artifact_paths={
            "output": output_path,
            "summary.json": summary_path,
            "bounded_report.json": report_path,
            "dreamcoder_library.json": library_path,
            "world_model.json": world_model_path,
            "planner_state.json": planner_state_path,
            "meta_harness.json": meta_harness_path,
            "weekly_report.md": ROOT / "WEEKLY_REPORT_20260416.md",
            **{f"recording_{idx+1}.jsonl": path for idx, path in enumerate(recording_candidates)},
        },
    )

    if args.include_v3_panels == "true":
        include_v3 = True
    elif args.include_v3_panels == "false":
        include_v3 = False
    else:
        include_v3 = _v3_files_exist({
            "world_model": world_model_path,
            "planner_state": planner_state_path,
        })

    if include_v3:
        v3_fragment = build_v3_section(
            world_model=world_model,
            planner_state=planner_state,
            skills=skills if isinstance(skills, list) else [],
            meta_harness=meta_harness,
            latest_episode=latest_episode,
        )
        html_text = _inject_v3_section(html_text, v3_fragment)

    output_path.write_text(html_text, encoding="utf-8")
    print(output_path)


if __name__ == "__main__":
    main()
