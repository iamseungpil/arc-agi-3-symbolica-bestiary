#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
PRESENTATION_ROOT = REPO_ROOT / "presentation"
COLOR_PALETTE = [
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
    "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25",
    "#555555", "#B10DC9", "#001F3F", "#7d4e4e", "#4e7d4e", "#4e4e7d",
]


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _grid_svg(grid: list[list[int]], *, max_side: int = 168) -> str:
    if not grid or not grid[0]:
        return '<div class="empty">empty</div>'
    rows = len(grid)
    cols = len(grid[0])
    cell = max(1, max_side // max(rows, cols))
    width = cols * cell
    height = rows * cell
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        'style="image-rendering:pixelated;border:1px solid #3f3f46;border-radius:6px;background:#09090b">'
    ]
    for y, row in enumerate(grid):
        for x, value in enumerate(row):
            color = COLOR_PALETTE[int(value) % 16]
            parts.append(
                f'<rect x="{x*cell}" y="{y*cell}" width="{cell}" height="{cell}" fill="{color}"/>'
            )
    parts.append("</svg>")
    return "".join(parts)


def _diff_svg(before: list[list[int]], after: list[list[int]], *, max_side: int = 168) -> str:
    if not before or not after:
        return '<div class="empty">no diff</div>'
    rows = min(len(before), len(after))
    cols = min(len(before[0]), len(after[0]))
    cell = max(1, max_side // max(rows, cols))
    width = cols * cell
    height = rows * cell
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        'style="image-rendering:pixelated;border:1px solid #3f3f46;border-radius:6px;background:#09090b">'
    ]
    for y in range(rows):
        for x in range(cols):
            if int(before[y][x]) == int(after[y][x]):
                continue
            parts.append(
                f'<rect x="{x*cell}" y="{y*cell}" width="{cell}" height="{cell}" fill="#ef4444"/>'
            )
    parts.append("</svg>")
    return "".join(parts)


def _module_card(title: str, body: str) -> str:
    return (
        '<div class="module-card">'
        f'<div class="module-title">{html.escape(title)}</div>'
        f'<div class="module-body">{body}</div>'
        '</div>'
    )


def _summarize_module_events(trace: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    buckets: dict[str, list[dict[str, Any]]] = {
        "hypothesis": [],
        "action": [],
        "skill": [],
    }
    for event in trace:
        module = str(event.get("module", "")).lower()
        kind = str(event.get("kind", "")).lower()
        if "skill" in module or event.get("skill_name"):
            buckets["skill"].append(event)
        elif kind == "action":
            buckets["action"].append(event)
        else:
            buckets["hypothesis"].append(event)
    return buckets


def _example_block(event: dict[str, Any]) -> str:
    kind = event.get("kind")
    if kind == "action":
        before = event.get("before", {})
        after = event.get("after", {})
        return (
            '<div class="module-example-grid">'
            f'<div><div class="caption">Before</div>{_grid_svg(before.get("grid", []), max_side=128)}</div>'
            f'<div><div class="caption">Diff</div>{_diff_svg(before.get("grid", []), after.get("grid", []), max_side=128)}</div>'
            f'<div><div class="caption">After</div>{_grid_svg(after.get("grid", []), max_side=128)}</div>'
            '</div>'
            '<pre>'
            + html.escape(
                json.dumps(
                    {
                        "action": event.get("action", ""),
                        "hypothesis_id": event.get("hypothesis_id", ""),
                        "planned_actions": event.get("planned_actions", []),
                        "expected": event.get("expected", ""),
                        "rationale": event.get("rationale", ""),
                        "change_summary": event.get("change_summary", {}),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            + '</pre>'
        )
    if kind == "plan":
        return (
            '<pre>'
            + html.escape(
                json.dumps(
                    {
                        "hypothesis_id": event.get("hypothesis_id", ""),
                        "state_summary": event.get("state_summary", ""),
                        "planned_actions": event.get("planned_actions", []),
                        "expected": event.get("expected", ""),
                        "rationale": event.get("rationale", ""),
                    },
                    indent=2,
                    ensure_ascii=False,
                )
            )
            + '</pre>'
        )
    return '<pre>' + html.escape(json.dumps(event, indent=2, ensure_ascii=False)) + '</pre>'


def _module_overview(name: str, events: list[dict[str, Any]]) -> str:
    if not events:
        return _module_card(name, '<span class="muted">no events</span>')
    first = events[0]
    last = events[-1]
    body = (
        f'<div><b>count</b>: {len(events)}</div>'
        f'<div><b>first module</b>: {html.escape(str(first.get("module", "")))}</div>'
        f'<div><b>latest rationale</b></div>'
        f'<pre>{html.escape(str(last.get("rationale", ""))[:500])}</pre>'
        f'<div><b>representative example</b></div>'
        f'{_example_block(last)}'
    )
    return _module_card(name, body)


def build_html(namespace: str, root: Path) -> str:
    summary = _load_json(root / "summary.json")
    hypotheses = _load_json(root / "hypotheses.json").get("hypotheses", [])
    sequences = _load_json(root / "action_sequences.json").get("action_sequences", [])
    skills = _load_json(root / "skills.json").get("skills", [])
    trace = _load_json(root / "trace_steps.json").get("events", [])
    cycle_packets = _load_json(root / "cycle_packets.json").get("cycle_packets", [])
    diagnosis = _load_json(root / "diagnosis.json")
    skill_effect = _load_json(root / "skill_effect.json")
    plan_bridge = _load_json(root / "plan_bridge.json")
    module_events = _summarize_module_events(trace)

    module_cards = [
        _module_card(
            "Hypothesis",
            "<br/>".join(
                html.escape(f"{item['id']} [{item.get('confidence', 'low')}] {item['title']}: {item['statement']}")
                for item in hypotheses[-5:]
            ) or '<span class="muted">none</span>',
        ),
        _module_card(
            "Action Sequence",
            "<br/>".join(
                html.escape(f"{item['id']} [{item.get('status', 'planned')}] {' -> '.join(item.get('sequence', []))}")
                for item in sequences[-5:]
            ) or '<span class="muted">none</span>',
        ),
        _module_card(
            "Skill",
            "<br/>".join(
                html.escape(
                    f"{item['name']} [{item.get('kind', 'thinking')}]: {item['description']} | strategy={item.get('strategy', '')} | cpitr={json.dumps(item.get('cpitr', {}), ensure_ascii=False)}"
                )
                for item in skills[-5:]
            ) or '<span class="muted">none</span>',
        ),
        _module_card(
            "Summary",
            "<br/>".join(
                html.escape(f"{key}: {value}")
                for key, value in summary.items()
            ) or '<span class="muted">none</span>',
        ),
        _module_overview("Hypothesis Module", module_events["hypothesis"]),
        _module_overview("Action Module", module_events["action"]),
        _module_overview("Skill Module", module_events["skill"]),
    ]

    cycle_rows: list[str] = []
    for packet in cycle_packets:
        reward_snapshot = packet.get("reward_snapshot", {})
        cycle_rows.append(
            f"""
            <details class="step-card">
              <summary>cycle #{int(packet.get('cycle_index', 0)):02d} [{html.escape(str(packet.get('phase', '')))}] {html.escape(str(packet.get('action') or 'no action'))}</summary>
              <div class="step-detail">
                <div><b>selected hypothesis</b><pre>{html.escape(json.dumps(packet.get('selected_hypothesis', {}), indent=2, ensure_ascii=False))}</pre></div>
                <div><b>input packet</b><pre>{html.escape(json.dumps(packet.get('input_packet', {}), indent=2, ensure_ascii=False)[:4000])}</pre></div>
                <div><b>expected vs observed</b><pre>{html.escape(json.dumps({'expected': packet.get('expected', ''), 'observed': packet.get('observed', {})}, indent=2, ensure_ascii=False))}</pre></div>
                <div><b>reward snapshot</b><pre>{html.escape(json.dumps(reward_snapshot, indent=2, ensure_ascii=False))}</pre></div>
                <div><b>diagnosis</b><pre>{html.escape(str(packet.get('diagnosis', '')))}</pre></div>
              </div>
            </details>
            """
        )

    diagnosis_block = ""
    if diagnosis:
        diagnosis_block = (
            '<h2>Diagnosis</h2>'
            '<div class="module-grid">'
            + _module_card("What Worked", "<br/>".join(html.escape(str(item)) for item in diagnosis.get("what_worked", [])))
            + _module_card("Next Problem", html.escape(str(diagnosis.get("next_problem", ""))))
            + _module_card("Next Cycle", "<br/>".join(html.escape(str(item)) for item in diagnosis.get("recommended_next_cycle", [])))
            + _module_card("Not Yet Known", "<br/>".join(html.escape(str(item)) for item in diagnosis.get("not_yet_known", [])))
            + "</div>"
        )
    if skill_effect:
        diagnosis_block += (
            '<h2>Thinking Skill Effect</h2>'
            '<div class="module-grid">'
            + _module_card(
                "Seeded Skill",
                '<pre>' + html.escape(json.dumps(skill_effect.get("seeded_skill", {}), indent=2, ensure_ascii=False)) + '</pre>',
            )
            + _module_card(
                "Without Skill",
                '<pre>' + html.escape(json.dumps(skill_effect.get("without_skill", []), indent=2, ensure_ascii=False)) + '</pre>',
            )
            + _module_card(
                "With Skill",
                '<pre>' + html.escape(json.dumps(skill_effect.get("with_skill", []), indent=2, ensure_ascii=False)) + '</pre>',
            )
            + _module_card(
                "Effect Diagnosis",
                '<pre>' + html.escape(json.dumps(skill_effect.get("diagnosis", {}), indent=2, ensure_ascii=False)) + '</pre>',
            )
            + '</div>'
        )
    if plan_bridge:
        diagnosis_block += (
            '<h2>Plan Bridge</h2>'
            '<div class="module-grid">'
            + _module_card(
                "Seeded Control Skill",
                '<pre>' + html.escape(json.dumps(plan_bridge.get("seeded_control_skill", {}), indent=2, ensure_ascii=False)) + '</pre>',
            )
            + _module_card(
                "Generic Plan",
                '<pre>' + html.escape(json.dumps(plan_bridge.get("generic_plan", {}), indent=2, ensure_ascii=False)) + '</pre>',
            )
            + _module_card(
                "Evidence-Conditioned Plan",
                '<pre>' + html.escape(json.dumps(plan_bridge.get("evidence_conditioned_plan", {}), indent=2, ensure_ascii=False)) + '</pre>',
            )
            + _module_card(
                "Plan Diagnosis",
                '<pre>' + html.escape(json.dumps(plan_bridge.get("diagnosis", {}), indent=2, ensure_ascii=False)) + '</pre>',
            )
            + '</div>'
        )

    rows: list[str] = []
    for index, event in enumerate(trace):
        if event.get("kind") == "plan":
            rows.append(
                f"""
                <details class="step-card">
                  <summary>plan #{index:03d} [{html.escape(event.get('module', ''))}] {html.escape(event.get('hypothesis_id', ''))}</summary>
                  <div class="step-detail">
                    <div><b>state_summary</b><pre>{html.escape(event.get('state_summary', ''))}</pre></div>
                    <div><b>rationale</b><pre>{html.escape(event.get('rationale', ''))}</pre></div>
                    <div><b>planned_actions</b><pre>{html.escape(' -> '.join(event.get('planned_actions', [])))}</pre></div>
                    <div><b>expected</b><pre>{html.escape(event.get('expected', ''))}</pre></div>
                  </div>
                </details>
                """
            )
            continue
        if event.get("kind") == "event":
            rows.append(
                f"""
                <details class="step-card">
                  <summary>event #{index:03d} [{html.escape(event.get('module', ''))}] {html.escape(event.get('title', ''))}</summary>
                  <div class="step-detail">
                    <div><b>rationale</b><pre>{html.escape(event.get('rationale', ''))}</pre></div>
                    <div><b>payload</b><pre>{html.escape(json.dumps(event.get('payload', {}), indent=2, ensure_ascii=False))}</pre></div>
                  </div>
                </details>
                """
            )
            continue
        before = event.get("before", {})
        after = event.get("after", {})
        rows.append(
            f"""
            <details class="step-card" open>
              <summary>action #{index:03d} [{html.escape(event.get('module', ''))}] {html.escape(event.get('action', ''))}</summary>
              <div class="step-grid">
                <div>
                  <div class="caption">Before state</div>
                  {_grid_svg(before.get('grid', []))}
                  <pre>{html.escape(json.dumps({k: before.get(k) for k in ('signature', 'level', 'state', 'available_actions')}, indent=2, ensure_ascii=False))}</pre>
                </div>
                <div>
                  <div class="caption">State-action pair</div>
                  <pre>{html.escape(json.dumps({
                      'hypothesis_id': event.get('hypothesis_id', ''),
                      'sequence_id': event.get('sequence_id', ''),
                      'planned_actions': event.get('planned_actions', []),
                      'action': event.get('action', ''),
                      'expected': event.get('expected', ''),
                      'rationale': event.get('rationale', ''),
                      'matched_plan': event.get('matched_plan', False),
                  }, indent=2, ensure_ascii=False))}</pre>
                  <div class="caption">Change mask</div>
                  {_diff_svg(before.get('grid', []), after.get('grid', []))}
                </div>
                <div>
                  <div class="caption">After state</div>
                  {_grid_svg(after.get('grid', []))}
                  <pre>{html.escape(json.dumps({
                      'signature': after.get('signature', ''),
                      'level': after.get('level', 0),
                      'state': after.get('state', ''),
                      'available_actions': after.get('available_actions', []),
                      'change_summary': event.get('change_summary', {}),
                  }, indent=2, ensure_ascii=False))}</pre>
                </div>
              </div>
            </details>
            """
        )

    return f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8"/>
      <title>Simple Trace Viewer {html.escape(namespace)}</title>
      <style>
        body {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; background:#09090b; color:#fafafa; margin:0; padding:24px; }}
        h1, h2 {{ margin:0 0 14px; }}
        .muted {{ color:#a1a1aa; }}
        .module-grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:14px; margin:18px 0 26px; }}
        .module-card {{ border:1px solid #27272a; border-radius:10px; background:#111214; padding:14px; }}
        .module-title {{ font-size:13px; letter-spacing:.08em; text-transform:uppercase; color:#93c5fd; margin-bottom:8px; }}
        .module-body {{ color:#e4e4e7; font-size:12px; line-height:1.45; white-space:pre-wrap; }}
        .module-example-grid {{ display:grid; grid-template-columns:repeat(3,minmax(120px,1fr)); gap:10px; margin:10px 0; }}
        .step-card {{ margin:12px 0; border:1px solid #27272a; border-radius:10px; background:#111214; overflow:hidden; }}
        .step-card summary {{ cursor:pointer; padding:12px 14px; background:#18181b; }}
        .step-detail {{ padding:14px; display:grid; gap:10px; }}
        .step-grid {{ padding:14px; display:grid; grid-template-columns:repeat(3,minmax(250px,1fr)); gap:16px; align-items:start; }}
        .caption {{ color:#a1a1aa; font-size:12px; margin:0 0 8px; }}
        pre {{ white-space:pre-wrap; word-break:break-word; font-size:12px; line-height:1.4; background:#0a0a0c; border:1px solid #27272a; border-radius:8px; padding:10px; }}
        .empty {{ color:#71717a; border:1px dashed #3f3f46; border-radius:6px; padding:18px; text-align:center; }}
      </style>
    </head>
    <body>
      <h1>Simple Trace Viewer</h1>
      <div class="muted">namespace={html.escape(namespace)} path={html.escape(str(root))}</div>
      <h2>Modules</h2>
      <div class="module-grid">{''.join(module_cards)}</div>
      {'<h2>Cycle Packets</h2>' + ''.join(cycle_rows) if cycle_rows else ''}
      {diagnosis_block}
      <h2>Trace</h2>
      {''.join(rows) or '<div class="muted">No trace events found.</div>'}
    </body>
    </html>
    """


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a simple trace viewer for ArcgenticaSimple runs.")
    parser.add_argument("namespace")
    parser.add_argument("--game-id", default="ls20-cb3b57cc")
    parser.add_argument("--root", help="Override log root directory.")
    parser.add_argument("--output", help="Output HTML path.")
    args = parser.parse_args()

    root = Path(args.root) if args.root else (REPO_ROOT / "simple_logs" / args.game_id / args.namespace)
    output = Path(args.output) if args.output else (PRESENTATION_ROOT / f"simple_trace_{args.namespace}.html")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_html(args.namespace, root), encoding="utf-8")
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
