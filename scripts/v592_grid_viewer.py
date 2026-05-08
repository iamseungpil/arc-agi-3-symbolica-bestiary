#!/usr/bin/env python3
"""v592 grid viewer — step-by-step visual + reasoning + skill.md per turn.

Renders trace.jsonl as a single-page interactive HTML:
  - top: sticky timeline bar (click to jump per turn)
  - top: skill.md full text panel (collapsible) showing
    cross_run_memory.json abstract_mechanics
  - per turn: 64x64 grid SVG with visible regions filled by color,
    click coord marked, multicolor markers highlighted, primary
    region highlighted
  - per turn side panel: M1 thought, chosen chid, candidate_tests
    summary, observation, verdict, M3 hypothesize call, M4 reflexion

Usage:
    python scripts/v592_grid_viewer.py <trace.jsonl> -o out.html [--cross-run path]
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

ARC_PALETTE = [
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
    "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25",
    "#555555", "#B10DC9", "#001F3F", "#7d4e4e", "#4e7d4e", "#4e4e7d",
]

CSS = """
* { box-sizing: border-box; }
body {
    margin: 0; padding: 0;
    background: #0d0d12; color: #e4e4e7;
    font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "Segoe UI", Roboto, sans-serif;
    font-size: 13px; line-height: 1.5;
}
.layout { display: grid; grid-template-columns: 280px 1fr; min-height: 100vh; }
.sidebar {
    background: #15151c; border-right: 1px solid #27272a;
    padding: 16px; position: sticky; top: 0; max-height: 100vh; overflow-y: auto;
}
.main { padding: 16px 24px; }
h1 { margin: 0 0 4px; font-size: 18px; color: #fafafa; }
h2 { margin: 16px 0 6px; font-size: 14px; color: #a1a1aa; text-transform: uppercase; letter-spacing: .04em; }
h3 { margin: 12px 0 4px; font-size: 13px; color: #93c5fd; }
.meta { color: #a1a1aa; font-family: ui-monospace, SFMono-Regular, monospace; font-size: 11px; }
.timeline { display: grid; grid-template-columns: repeat(auto-fill, 18px); gap: 2px; margin: 8px 0; }
.timeline a {
    display: block; height: 22px; border-radius: 2px;
    background: #3f3f46; text-decoration: none; transition: transform .1s;
}
.timeline a:hover { transform: scale(1.4); z-index: 5; }
.timeline a.lp1 { background: #f59e0b; }
.timeline a.lp2 { background: #22c55e; }
.timeline a.tier-a { background: #60a5fa; }
.timeline a.outside { opacity: .4; }
.skill-panel {
    background: #18181b; border: 1px solid #3f3f46; border-radius: 6px;
    padding: 10px 12px; margin-top: 8px; font-size: 12px;
}
.skill-panel summary { cursor: pointer; color: #93c5fd; font-weight: 600; }
.skill-entry {
    margin: 6px 0; padding: 6px 8px; background: #0d0d12; border-radius: 4px;
    border-left: 2px solid #3f3f46;
}
.skill-entry.confirmed { border-left-color: #22c55e; }
.skill-entry .sid { color: #a1a1aa; font-family: monospace; font-size: 11px; }

.turn-card {
    background: #15151c; border: 1px solid #27272a; border-radius: 8px;
    padding: 12px; margin: 10px 0; scroll-margin-top: 12px;
}
.turn-card.lp1 { border-color: #f59e0b; }
.turn-card.lp2 { border-color: #22c55e; box-shadow: 0 0 16px rgba(34,197,94,.2); }
.turn-header { display: flex; gap: 12px; align-items: center; margin-bottom: 8px; }
.turn-id { font-size: 16px; font-weight: 700; color: #fafafa; }
.badge { padding: 2px 8px; border-radius: 10px; font-size: 11px; font-family: monospace; }
.badge.tier-a { background: #1e3a8a; color: #93c5fd; }
.badge.tier-b { background: #4a044e; color: #f0abfc; }
.badge.tier-card { background: #14532d; color: #86efac; }
.badge.tier-none { background: #3f3f46; color: #a1a1aa; }
.badge.lp1 { background: #78350f; color: #fbbf24; }
.badge.lp2 { background: #14532d; color: #4ade80; }
.badge.snapped { background: #422006; color: #fcd34d; }
.badge.forced { background: #831843; color: #f9a8d4; }
.body-grid { display: grid; grid-template-columns: 280px 1fr; gap: 16px; align-items: start; }
.svg-wrap { background: #0a0a0c; padding: 8px; border-radius: 6px; }
.svg-wrap svg { display: block; width: 264px; height: 264px; }
.detail-grid { display: grid; gap: 10px; }
.field { background: #0a0a0c; padding: 8px 10px; border-radius: 4px; font-family: ui-monospace, monospace; font-size: 11.5px; }
.field-label { color: #a1a1aa; font-size: 10px; text-transform: uppercase; letter-spacing: .04em; margin-bottom: 4px; }
.thought { white-space: pre-wrap; word-break: break-word; max-height: 240px; overflow-y: auto; line-height: 1.6; }
details summary { cursor: pointer; color: #93c5fd; }
details summary:hover { color: #bfdbfe; }
pre { white-space: pre-wrap; word-break: break-word; margin: 4px 0; font-size: 11px; }
.click-marker { stroke: #fbbf24; stroke-width: 2; fill: none; }
.click-x { stroke: #fbbf24; stroke-width: 2; }
.region-bbox { stroke-width: 0.4; opacity: .85; }
.region-multicolor { stroke: #f5f5f4; stroke-width: 0.6; }
.region-primary { stroke: #fbbf24; stroke-width: 1.0; }
"""

_SVG_GRID = 64


def _palette(idx: int) -> str:
    return ARC_PALETTE[idx % len(ARC_PALETTE)]


def _bbox_xyxy(bb: Any) -> tuple[int, int, int, int] | None:
    if not bb:
        return None
    if isinstance(bb, dict):
        return (
            int(bb.get("min_x", 0)), int(bb.get("min_y", 0)),
            int(bb.get("max_x", 0)), int(bb.get("max_y", 0)),
        )
    if isinstance(bb, (list, tuple)) and len(bb) >= 4:
        return (int(bb[0]), int(bb[1]), int(bb[2]), int(bb[3]))
    return None


def _render_grid_svg(visible_regions: list[dict], coord: list[int],
                     primary_id: str | None, snapped: bool) -> str:
    """Render visible_regions as filled rectangles + click marker."""
    cell = 4  # 64*4 = 256 px
    parts = [
        f'<svg viewBox="0 0 {_SVG_GRID*cell} {_SVG_GRID*cell}" '
        f'xmlns="http://www.w3.org/2000/svg">'
    ]
    # background
    parts.append(f'<rect x="0" y="0" width="{_SVG_GRID*cell}" '
                 f'height="{_SVG_GRID*cell}" fill="#0a0a0c"/>')
    for r in visible_regions or []:
        bb = _bbox_xyxy(r.get("bbox"))
        if not bb:
            continue
        x0, y0, x1, y1 = bb
        w = (x1 - x0 + 1) * cell
        h = (y1 - y0 + 1) * cell
        color_idx = r.get("color")
        fill = _palette(color_idx) if isinstance(color_idx, int) else "#3f3f46"
        is_mc = bool(r.get("is_multicolor"))
        rid = r.get("id")
        cls = "region-bbox"
        if is_mc:
            cls += " region-multicolor"
        if rid == primary_id:
            cls += " region-primary"
        parts.append(
            f'<rect class="{cls}" x="{x0*cell}" y="{y0*cell}" '
            f'width="{w}" height="{h}" fill="{fill}"/>'
        )
        # If multicolor + has crop, render 3x3 mini-pattern overlay
        crop = r.get("crop")
        if is_mc and isinstance(crop, list) and len(crop) == 3:
            mini = (x1 - x0 + 1) / 3 * cell
            for ry, row in enumerate(crop[:3]):
                if not isinstance(row, list):
                    continue
                for rx, c in enumerate(row[:3]):
                    if not isinstance(c, int):
                        continue
                    parts.append(
                        f'<rect x="{x0*cell + rx*mini}" '
                        f'y="{y0*cell + ry*mini}" '
                        f'width="{mini-0.5}" height="{mini-0.5}" '
                        f'fill="{_palette(c)}" opacity="0.95"/>'
                    )
        # region id label
        parts.append(
            f'<text x="{x0*cell + 1}" y="{y0*cell + 8}" '
            f'fill="#fafafa" font-size="6" font-family="monospace" '
            f'opacity="0.7">{rid}</text>'
        )
    # Click marker (X + circle)
    if isinstance(coord, list) and len(coord) == 2:
        cx, cy = coord
        cx_px = cx * cell + cell / 2
        cy_px = cy * cell + cell / 2
        size = 8
        cstroke = "#22c55e" if snapped else "#fbbf24"
        parts.append(
            f'<circle class="click-marker" cx="{cx_px}" cy="{cy_px}" r="6" '
            f'stroke="{cstroke}" fill="none" stroke-width="2"/>'
        )
        parts.append(
            f'<line class="click-x" x1="{cx_px - size}" y1="{cy_px - size}" '
            f'x2="{cx_px + size}" y2="{cy_px + size}" '
            f'stroke="{cstroke}" stroke-width="2"/>'
        )
        parts.append(
            f'<line class="click-x" x1="{cx_px + size}" y1="{cy_px - size}" '
            f'x2="{cx_px - size}" y2="{cy_px + size}" '
            f'stroke="{cstroke}" stroke-width="2"/>'
        )
    parts.append('</svg>')
    return "".join(parts)


def _render_skill_panel(cross_run: dict | None) -> str:
    if not cross_run:
        return '<div class="skill-panel">no cross_run_memory.json found</div>'
    mechs = cross_run.get("abstract_mechanics", [])
    schema = cross_run.get("schema_version", "?")
    parts = [
        f'<details class="skill-panel" open>',
        f'<summary>📚 cross_run_memory.json '
        f'(schema={schema}, {len(mechs)} mechanics)</summary>',
    ]
    for m in mechs:
        confirmed = int(m.get("confirmed_runs", 0))
        cls = "skill-entry confirmed" if confirmed >= 1 else "skill-entry"
        sid = html.escape(str(m.get("id", "?")))
        text = html.escape(str(m.get("text", "")))
        src = html.escape(str(m.get("source", m.get("last_seen_run", "?"))))
        parts.append(
            f'<div class="{cls}"><span class="sid">{sid} '
            f'(confirmed={confirmed}, src={src[:48]})</span><br/>{text}</div>'
        )
    parts.append('</details>')
    return "".join(parts)


def _render_field(label: str, content: str, *, mono: bool = True) -> str:
    cls = "field thought" if "thought" in label.lower() else "field"
    return (
        f'<div class="{cls}">'
        f'<div class="field-label">{html.escape(label)}</div>'
        f'<div>{content if mono else html.escape(content)}</div>'
        f'</div>'
    )


def _short(x: Any, n: int = 200) -> str:
    s = json.dumps(x, ensure_ascii=False, default=str)
    if len(s) > n:
        s = s[:n - 3] + "..."
    return html.escape(s)


def _render_turn(row: dict) -> str:
    t = row.get("turn", "?")
    obs = row.get("observation") or {}
    a = row.get("action") or {}
    chid = a.get("chosen_hypothesis_id")
    ld = int(obs.get("level_delta") or 0)
    coord = row.get("coord") or [32, 32]
    tier = row.get("chid_tier") or "none"
    snapped = bool(row.get("snapped_coord"))
    primary = obs.get("primary_region_id")
    force_reason = row.get("v592_force_reason")
    visible = row.get("visible_regions") or []

    badges = [
        f'<span class="turn-id">T{t}</span>',
        f'<span class="badge tier-{tier.lower()}">{tier}</span>',
    ]
    if ld == 1:
        badges.append('<span class="badge lp1">L+1 ⬆</span>')
    elif ld >= 2:
        badges.append('<span class="badge lp2">L+2 🎯</span>')
    if snapped:
        badges.append('<span class="badge snapped">snap</span>')
    if force_reason:
        badges.append(f'<span class="badge forced">v592:{html.escape(force_reason)}</span>')

    card_cls = "turn-card"
    if ld == 1:
        card_cls += " lp1"
    elif ld >= 2:
        card_cls += " lp2"

    svg = _render_grid_svg(visible, coord, primary, snapped)
    thought = html.escape((a.get("thought") or row.get("thought") or "").strip())
    chid_str = html.escape(str(chid))
    coord_str = html.escape(str(coord))
    prim_str = html.escape(str(primary))
    verdict = html.escape(str(row.get("verdict") or ""))
    obs_str = _short(obs, 300)
    cands = row.get("candidate_tests_for_m1") or []
    cands_summary = ", ".join(
        f'{c.get("predicate_id","?")[:40]} (s={c.get("score","?")})'
        for c in cands[:5]
    )
    invented = row.get("invented_meta")
    invented_str = _short(invented, 200) if invented else "—"
    h_call = row.get("hypothesize_call")
    r_call = row.get("reflexion_call")
    summary_after = html.escape((row.get("summary_after") or "")[:600])

    detail_parts = [
        _render_field("M1 thought", f"<pre>{thought}</pre>"),
        _render_field("chid / coord / primary",
                      f"<pre>chid={chid_str}\ncoord={coord_str}\nprimary={prim_str}\nverdict={verdict}</pre>"),
        _render_field("invented_meta", f"<pre>{invented_str}</pre>"),
        _render_field("candidate_tests (top5)",
                      f"<pre>{html.escape(cands_summary or 'none')}</pre>"),
        _render_field("observation", f"<pre>{obs_str}</pre>"),
    ]
    if h_call:
        detail_parts.append(
            f'<details><summary>M3 hypothesize call</summary>'
            f'<pre>{_short(h_call, 1500)}</pre></details>'
        )
    if r_call:
        detail_parts.append(
            f'<details><summary>M4 reflexion call</summary>'
            f'<pre>{_short(r_call, 1500)}</pre></details>'
        )
    if summary_after:
        detail_parts.append(
            f'<details><summary>summary_after</summary>'
            f'<pre>{summary_after}</pre></details>'
        )

    return (
        f'<div id="t{t}" class="{card_cls}">'
        f'<div class="turn-header">{"".join(badges)}</div>'
        f'<div class="body-grid">'
        f'<div class="svg-wrap">{svg}</div>'
        f'<div class="detail-grid">{"".join(detail_parts)}</div>'
        f'</div></div>'
    )


def render(trace_path: Path, output: Path,
           cross_run_path: Path | None = None) -> None:
    rows = []
    for ln in trace_path.read_text().splitlines():
        ln = ln.strip()
        if not ln:
            continue
        try:
            rows.append(json.loads(ln))
        except json.JSONDecodeError:
            continue
    cross_run = None
    if cross_run_path and cross_run_path.exists():
        cross_run = json.loads(cross_run_path.read_text())

    timeline = []
    for r in rows:
        t = r.get("turn", 0)
        obs = r.get("observation") or {}
        ld = int(obs.get("level_delta") or 0)
        tier = r.get("chid_tier") or "none"
        primary = obs.get("primary_region_id")
        cls_parts = []
        if ld == 1:
            cls_parts.append("lp1")
        elif ld >= 2:
            cls_parts.append("lp2")
        if tier == "A":
            cls_parts.append("tier-a")
        if primary == "_outside_":
            cls_parts.append("outside")
        cls = " ".join(cls_parts)
        title = f"t={t} tier={tier} ld={ld} primary={primary}"
        timeline.append(
            f'<a href="#t{t}" class="{cls}" title="{html.escape(title)}"></a>'
        )

    cycle_label = trace_path.parent.name
    total = len(rows)
    lp_total = sum(int((r.get("observation") or {}).get("level_delta") or 0)
                   for r in rows)
    tier_counts = {}
    for r in rows:
        tier = r.get("chid_tier") or "none"
        tier_counts[tier] = tier_counts.get(tier, 0) + 1

    sidebar = (
        f'<h1>v592 grid viewer</h1>'
        f'<div class="meta">{html.escape(cycle_label)}</div>'
        f'<h2>Summary</h2>'
        f'<div class="meta">turns={total}<br/>'
        f'max_level=L+{lp_total}<br/>'
        f'tier_dist={tier_counts}</div>'
        f'<h2>Timeline</h2>'
        f'<div class="timeline">{"".join(timeline)}</div>'
        + _render_skill_panel(cross_run)
        + '<h2>Legend</h2>'
        + '<div class="meta">timeline:<br/>'
          '<span style="color:#22c55e">■</span> L+2  '
          '<span style="color:#f59e0b">■</span> L+1<br/>'
          '<span style="color:#60a5fa">■</span> TIER-A  '
          '<span style="color:#3f3f46">■</span> other<br/><br/>'
          'grid:<br/>'
          '<span style="color:#fbbf24">⊗</span> M1 click  '
          '<span style="color:#22c55e">⊗</span> snapped<br/>'
          'gold border = primary region<br/>'
          'white border = multicolor marker</div>'
    )

    main_parts = [_render_turn(r) for r in rows]

    page = (
        '<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/>'
        f'<title>v592 grid viewer — {html.escape(cycle_label)}</title>'
        f'<style>{CSS}</style></head><body>'
        f'<div class="layout">'
        f'<aside class="sidebar">{sidebar}</aside>'
        f'<main class="main">{"".join(main_parts)}</main>'
        f'</div></body></html>'
    )
    output.write_text(page, encoding="utf-8")
    print(f"wrote {output} ({total} turns, L+{lp_total})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("trace", type=Path)
    p.add_argument("-o", "--output", type=Path, default=Path("v592_grid_viewer.html"))
    p.add_argument("--cross-run", type=Path, default=None,
                   help="path to cross_run_memory.json")
    args = p.parse_args()
    if not args.cross_run:
        # default: <game_dir>/cross_run_memory.json
        guess = args.trace.parent.parent / "cross_run_memory.json"
        if guess.exists():
            args.cross_run = guess
    render(args.trace, args.output, cross_run_path=args.cross_run)


if __name__ == "__main__":
    main()
