"""trace_visualizer — single-file HTML browser/debugger for v57 trace.jsonl.

Usage:
  python3 tools/trace_visualizer.py <namespace>
  python3 tools/trace_visualizer.py cycle189_v587_B14_1778155086
  python3 tools/trace_visualizer.py --all              # build for all cycles + index
  python3 tools/trace_visualizer.py --latest 5         # 5 most recent

Output: reports/trace_viz/<namespace>.html (and reports/trace_viz/index.html for --all).

Design: single self-contained HTML per cycle. Top: metadata + L+ timeline +
stuck/oscillation map. Body: per-turn cards (collapsible) showing thought,
coord, observation, verdict, reflexion. Color codes: L+ event (green),
stuck_mode (orange), oscillation (red), normal (gray).

The HTML uses minimal CSS, no JS frameworks — view source-friendly.
"""

from __future__ import annotations

import argparse
import json
import sys
from html import escape
from pathlib import Path
from typing import Iterable


REPO = Path(__file__).resolve().parent.parent
SL = REPO / "simple_logs"
OUT = REPO / "reports" / "trace_viz"


# ---------------------------------------------------------------------
# Trace parsing.
# ---------------------------------------------------------------------


def load_trace(trace_path: Path) -> list[dict]:
    out = []
    if not trace_path.exists():
        return out
    with trace_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def find_cycle_dir(namespace: str) -> Path | None:
    for game_dir in SL.glob("*-*"):
        cand = game_dir / namespace
        if cand.exists() and (cand / "trace.jsonl").exists():
            return cand
    return None


def list_all_cycles() -> list[Path]:
    out = []
    for game_dir in SL.glob("*-*"):
        for cycle_dir in game_dir.glob("cycle*"):
            if (cycle_dir / "trace.jsonl").exists():
                out.append(cycle_dir)
    return sorted(out, key=lambda p: p.stat().st_mtime, reverse=True)


# ---------------------------------------------------------------------
# Per-turn analysis helpers.
# ---------------------------------------------------------------------


def turn_summary(entry: dict) -> dict:
    obs = entry.get("observation") or {}
    action = entry.get("action") or {}
    aresp = action if isinstance(action, dict) else {}
    return {
        "turn": entry.get("turn"),
        "ts": entry.get("ts"),
        "coord": entry.get("coord") or [0, 0],
        "snapped": entry.get("snapped_coord", False),
        "primary_region_id": obs.get("primary_region_id") or "_outside_",
        "dominant_transition": obs.get("dominant_transition"),
        "level_delta": int(obs.get("level_delta") or 0),
        "verdict": entry.get("verdict") or obs.get("verdict") or aresp.get("verdict"),
        "thought": (aresp.get("thought") or "")[:2000],
        "chosen_hypothesis_id": aresp.get("chosen_hypothesis_id"),
        "expected_observation": aresp.get("expected_observation") or {},
        "reflexion_call": entry.get("reflexion_call") or {},
        "hypothesize_call": entry.get("hypothesize_call") or {},
        # Round-1 critic additions:
        "visible_regions": entry.get("visible_regions") or [],
        "visible_region_ids": entry.get("visible_region_ids") or [],
        "active_hypotheses_after": entry.get("active_hypotheses_after") or [],
        "matched_prior_triggers": entry.get("matched_prior_triggers") or [],
        "summary_after": entry.get("summary_after") or "",
        # Round-3 (will be empty until agent.py traces it).
        # NOTE: agent.py writes "action_state_chain_compact" — accept
        # both keys for forward compat.
        "stuck_mode": entry.get("stuck_mode"),
        "action_state_chain": (
            entry.get("action_state_chain")
            or entry.get("action_state_chain_compact")
            or {}
        ),
        "analogous_past_segments": entry.get("analogous_past_segments") or [],
        "region_clicks_this_level": entry.get("region_clicks_this_level") or {},
        "marker_neighbor_states": entry.get("marker_neighbor_states") or [],
    }


def render_coord_grid(coord: list[int], size: int = 16) -> str:
    """ASCII rendering of click position in 64x64 grid scaled to size×size.
    Shows X at click cell, dots elsewhere. Compact (≤size lines)."""
    cx, cy = int(coord[0]), int(coord[1])
    # 64-cell grid scaled to size cells; each cell = 64/size pixels
    scale = 64 / size
    cell_x = min(int(cx / scale), size - 1)
    cell_y = min(int(cy / scale), size - 1)
    rows = []
    for y in range(size):
        row = []
        for x in range(size):
            if x == cell_x and y == cell_y:
                row.append("█")
            elif x == 0 or y == 0 or x == size - 1 or y == size - 1:
                row.append("·")
            else:
                row.append(" ")
        rows.append("".join(row))
    return "\n".join(rows)


def render_visible_regions_compact(regions: list[dict]) -> str:
    """Compact one-line-per-region listing. Highlights multicolor."""
    if not regions:
        return "<span class='label'>—</span>"
    rows = []
    for r in regions[:12]:
        if not isinstance(r, dict):
            continue
        rid = r.get("id", "?")
        color = r.get("color", "?")
        size = r.get("size", "?")
        is_mc = r.get("is_multicolor")
        bbox = r.get("bbox")
        marker = " <b style='color:#f39c12'>marker</b>" if is_mc else ""
        bbox_str = ""
        if isinstance(bbox, dict):
            bbox_str = f" bbox=[{bbox.get('min_x','?')},{bbox.get('min_y','?')},{bbox.get('max_x','?')},{bbox.get('max_y','?')}]"
        elif isinstance(bbox, list) and len(bbox) >= 4:
            bbox_str = f" bbox={bbox[:4]}"
        rows.append(
            f"<span class='region-id'>{rid}</span> color={color} size={size}"
            f"{bbox_str}{marker}"
        )
    extra = f" ... +{len(regions) - 12} more" if len(regions) > 12 else ""
    return "<br>".join(rows) + extra


def render_active_hypotheses(cards: list[dict]) -> str:
    if not cards:
        return "<span class='label'>—</span>"
    rows = []
    for c in cards[:5]:
        if not isinstance(c, dict):
            continue
        cid = c.get("id", "?")
        ctype = c.get("type", "?")
        rid = c.get("region_id", "?")
        pred = (c.get("predicate") or c.get("rule_hypothesis") or "")[:120]
        tested = c.get("tested_count", 0)
        rows.append(
            f"<span class='turn-id'>{cid}</span> [{ctype}] R={rid} tested={tested}<br>"
            f"<span style='color:#aaa;font-size:10px'>&nbsp;&nbsp;{pred}</span>"
        )
    extra = f" ... +{len(cards) - 5} more" if len(cards) > 5 else ""
    return "<br>".join(rows) + extra


def detect_oscillation(entries: list[dict], window: int = 8) -> list[bool]:
    """Per-turn: True if last `window` turns show region-repeat pattern with
    no level progress. Used as a debugging heatmap in the timeline."""
    out: list[bool] = []
    for i in range(len(entries)):
        lo = max(0, i - window + 1)
        win = entries[lo: i + 1]
        regions = [e.get("primary_region_id") for e in win if e.get("primary_region_id")]
        valid = [r for r in regions if r and r != "_outside_"]
        if len(valid) < 4:
            out.append(False)
            continue
        unique = len(set(valid))
        any_lp = any(int(e.get("level_delta") or 0) >= 1 for e in win)
        out.append((unique / len(valid)) < 0.5 and not any_lp)
    return out


def detect_lp_oscillation(entries: list[dict], window: int = 10) -> list[bool]:
    """Per-turn: True if last `window` turns saw multiple L+ events but
    no monotonic level progression (i.e. up-down pattern)."""
    out: list[bool] = []
    for i in range(len(entries)):
        lo = max(0, i - window + 1)
        win = entries[lo: i + 1]
        n_lp = sum(1 for e in win if int(e.get("level_delta") or 0) >= 1)
        # Heuristic: ≥3 L+ in 10 turns + same-window repeat pattern
        out.append(n_lp >= 3)
    return out


# ---------------------------------------------------------------------
# HTML rendering.
# ---------------------------------------------------------------------


CSS = """
:root {
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
  --danger-soft: #f2dfd8;
  --success: #2d7953;
  --success-soft: #d8efe0;
  --muted: #71888d;
  --warm: #d68910;
  --warm-soft: #f6e5c5;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  padding: 28px 28px 44px;
  max-width: 1540px;
  background:
    radial-gradient(circle at 0% 0%, rgba(253,211,144,0.34), transparent 34%),
    radial-gradient(circle at 100% 0%, rgba(157,205,208,0.22), transparent 28%),
    linear-gradient(180deg, rgba(255,255,255,0.6), rgba(244,242,234,1) 28%),
    var(--bg);
  color: var(--ink);
  font-family: "Avenir Next", "Segoe UI", "Trebuchet MS", sans-serif;
}
body::before {
  content: "";
  position: fixed; inset: 0; pointer-events: none; z-index: -1;
  background-image:
    linear-gradient(rgba(22,63,72,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(22,63,72,0.03) 1px, transparent 1px);
  background-size: 32px 32px;
}
h1, h2, h3 {
  margin: 0;
  font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
  font-weight: 600;
}
h1 { font-size: clamp(28px, 3vw, 40px); line-height: 1.05; margin-bottom: 6px; }
h2 { font-size: 20px; color: var(--ink); margin: 22px 0 10px; }
.metadata {
  background: var(--panel); border: 1px solid var(--border);
  box-shadow: var(--shadow); border-radius: 24px;
  padding: 18px 22px; margin-bottom: 16px;
  font-size: 14px; color: var(--ink-soft); line-height: 1.55;
}
.metadata b { color: var(--ink); }
.timeline {
  background: var(--panel); border: 1px solid var(--border);
  box-shadow: var(--shadow); border-radius: 24px;
  padding: 14px 16px; margin: 10px 0 22px;
  display: flex; flex-wrap: wrap; gap: 7px; font-size: 12px;
}
.tick {
  width: 36px; height: 44px; display: inline-flex; align-items: center;
  justify-content: center; border-radius: 12px;
  border: 1px solid rgba(20,55,61,0.08);
  background: #f1f4f3; color: var(--ink);
  font-weight: 700; font-size: 11px;
  cursor: pointer; text-decoration: none;
  transition: transform 0.15s ease;
}
.tick.lp1 { background: var(--success-soft); color: var(--success); }
.tick.lp2 {
  background: var(--success); color: white;
  box-shadow: 0 8px 20px rgba(45,121,83,0.25);
}
.tick.osc { background: var(--danger-soft); color: var(--danger); }
.tick.lposc { background: var(--warm-soft); color: var(--warm); }
.tick.stuck { background: var(--accent-soft); color: #7a4c12; }
.tick:hover { transform: translateY(-2px); outline: 2px solid var(--accent); }
.turn {
  background: var(--panel); border: 1px solid var(--border);
  box-shadow: var(--shadow); border-radius: 24px;
  padding: 18px 22px; margin: 14px 0; font-size: 13px;
  line-height: 1.55;
}
.turn.lp1 { border-color: rgba(45,121,83,0.35); background: linear-gradient(180deg, rgba(216,239,224,0.55), var(--panel)); }
.turn.lp2 { border-color: var(--success); background: linear-gradient(180deg, rgba(216,239,224,0.85), var(--panel)); }
.turn.osc { border-color: rgba(173,80,62,0.35); background: linear-gradient(180deg, rgba(242,223,216,0.55), var(--panel)); }
.turn.lposc { border-color: rgba(214,137,16,0.35); background: linear-gradient(180deg, rgba(246,229,197,0.5), var(--panel)); }
.turn.stuck { border-color: var(--accent); background: linear-gradient(180deg, rgba(245,216,167,0.42), var(--panel)); }
.turn-header {
  display: flex; gap: 14px; align-items: baseline;
  margin-bottom: 12px; font-weight: 700;
  border-bottom: 1px solid var(--border); padding-bottom: 10px;
}
.turn-id { color: var(--teal); font-size: 16px; font-family: "Iowan Old Style", serif; }
.turn-coord {
  color: var(--ink-soft);
  font-family: "JetBrains Mono", "SF Mono", Menlo, monospace;
  font-size: 12px;
  background: rgba(22,63,72,0.05);
  padding: 3px 9px; border-radius: 999px;
}
.lp-badge, .lp2-badge, .osc-badge, .stuck-badge {
  display: inline-flex; align-items: center;
  border-radius: 999px; padding: 4px 11px;
  font-size: 11px; font-weight: 700;
  letter-spacing: 0.04em;
}
.lp-badge { background: var(--success-soft); color: var(--success); }
.lp2-badge { background: var(--success); color: white; box-shadow: 0 6px 16px rgba(45,121,83,0.22); }
.osc-badge { background: var(--danger-soft); color: var(--danger); }
.stuck-badge { background: var(--accent-soft); color: #7a4c12; }
.section { margin-top: 10px; }
.label {
  color: var(--muted); font-size: 11px; margin-right: 8px;
  font-weight: 700; text-transform: uppercase; letter-spacing: 0.14em;
}
.thought, .reflexion {
  padding: 12px 14px; border-radius: 16px;
  font-size: 12px; line-height: 1.6;
  color: var(--ink); white-space: pre-wrap;
  background: rgba(207,232,231,0.32);
  border: 1px solid var(--border);
  margin-top: 6px;
  font-family: "Avenir Next", "Segoe UI", sans-serif;
}
.thought { border-left: 4px solid var(--teal); }
.reflexion { border-left: 4px solid var(--accent); background: rgba(245,216,167,0.28); margin-top: 10px; }
.transition {
  font-family: "JetBrains Mono", "SF Mono", Menlo, monospace;
  color: #7a4c12;
  background: var(--accent-soft);
  padding: 1px 7px; border-radius: 6px;
  font-size: 11px;
}
.region-id {
  font-family: "JetBrains Mono", "SF Mono", Menlo, monospace;
  color: var(--teal);
  background: var(--teal-soft);
  padding: 1px 7px; border-radius: 6px;
  font-size: 11px;
}
details {
  margin-top: 8px;
  background: rgba(255,255,255,0.6);
  border: 1px solid var(--border);
  border-radius: 14px;
  padding: 6px 12px;
}
details summary {
  cursor: pointer; color: var(--ink-soft); font-size: 12px;
  font-weight: 600; padding: 4px 0;
}
details summary:hover { color: var(--teal); }
.legend {
  background: var(--panel); border: 1px solid var(--border);
  box-shadow: var(--shadow); border-radius: 24px;
  padding: 14px 18px; margin-bottom: 16px;
  font-size: 12px; color: var(--ink-soft);
  display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
}
.legend span {
  display: inline-flex; align-items: center;
  padding: 4px 11px; border-radius: 999px;
  font-size: 11px; font-weight: 700;
}
table {
  border-collapse: separate; border-spacing: 0;
  font-size: 12px; width: 100%;
  background: var(--panel-strong); border-radius: 16px;
  overflow: hidden; box-shadow: var(--shadow);
  margin: 10px 0;
}
table td, table th {
  padding: 9px 14px;
  border-bottom: 1px solid var(--border);
  text-align: left;
}
table th {
  background: var(--teal-soft); color: var(--teal);
  font-weight: 700; font-size: 11px;
  text-transform: uppercase; letter-spacing: 0.1em;
}
table tr:last-child td { border-bottom: none; }
table tr:nth-child(even) td { background: rgba(244,242,234,0.4); }
a { color: var(--teal); text-decoration: none; }
a:hover { color: var(--accent); text-decoration: underline; }
"""


def render_legend() -> str:
    return (
        '<div class="legend">'
        '<span class="lp-badge">L+1</span>'
        '<span class="lp2-badge">L+2</span>'
        '<span class="osc-badge">OSCILLATION</span>'
        '<span class="stuck-badge">STUCK</span>'
        '</div>'
    )


def render_timeline(summaries: list[dict], oscillations: list[bool],
                    lp_oscillations: list[bool]) -> str:
    parts = ['<div class="timeline">']
    for i, s in enumerate(summaries):
        cls = "tick"
        ld = s["level_delta"]
        if ld >= 2:
            cls += " lp2"; label = f"L{ld}"
        elif ld >= 1:
            cls += " lp1"; label = "L+"
        elif oscillations[i]:
            cls += " osc"; label = "O"
        elif lp_oscillations[i]:
            cls += " lposc"; label = "~"
        else:
            label = ""
        parts.append(
            f'<a href="#t{s["turn"]}" class="{cls}" '
            f'title="turn={s["turn"]} L+={ld} R={s["primary_region_id"]}">'
            f'{label}</a>'
        )
    parts.append('</div>')
    return "".join(parts)


def render_turn(s: dict, oscillation: bool, lp_oscillation: bool) -> str:
    classes = ["turn"]
    badges = []
    if s["level_delta"] >= 2:
        classes.append("lp2"); badges.append('<span class="lp2-badge">L+2</span>')
    elif s["level_delta"] >= 1:
        classes.append("lp1"); badges.append('<span class="lp-badge">L+1</span>')
    if oscillation:
        classes.append("osc"); badges.append('<span class="osc-badge">OSCILLATION</span>')
    if lp_oscillation and not oscillation and s["level_delta"] == 0:
        classes.append("lposc"); badges.append('<span class="osc-badge" style="background:#8b4513">L+OSC</span>')

    dt = s["dominant_transition"]
    dt_str = (
        f'<span class="transition">{dt.get("from")}→{dt.get("to")} '
        f'cnt={dt.get("count","?")}</span>'
        if isinstance(dt, dict) and dt.get("from") is not None else "<span class='label'>—</span>"
    )

    rfl = s["reflexion_call"]
    rfl_section = ""
    if rfl:
        summary = (rfl.get("summary") or "")[:1000]
        promotions = rfl.get("promote_to_1A") or []
        prom_html = ""
        if promotions and isinstance(promotions, list):
            items = []
            for p in promotions[:5]:
                t = p if isinstance(p, str) else (p.get("text") if isinstance(p, dict) else "")
                if t:
                    items.append(f"<li>{escape(t[:200])}</li>")
            if items:
                prom_html = f'<ul style="margin:4px 0;font-size:11px">{"".join(items)}</ul>'
        rfl_section = (
            f'<details><summary>Reflexion (M4) summary + promotions</summary>'
            f'<div class="reflexion">{escape(summary)}</div>'
            f'{prom_html}</details>'
        )

    hyp = s["hypothesize_call"]
    hyp_section = ""
    if hyp:
        thought = (hyp.get("thought") or "")[:1500]
        cards = hyp.get("cards") or []
        cards_html_parts = []
        for c in cards[:5]:
            if not isinstance(c, dict):
                continue
            cid = c.get("id", "?")
            ctype = c.get("type", "?")
            abst = c.get("abstraction_level", "?")
            rule = (c.get("rule_hypothesis") or c.get("predicate") or "")[:600]
            rid = c.get("region_id", "")
            extras = []
            if rid:
                extras.append(f"R={escape(str(rid))}")
            seq = c.get("click_sequence") or []
            if seq:
                extras.append(f"seq_steps={len(seq)}")
            extras_str = " · ".join(extras)
            cards_html_parts.append(
                f"<div style='margin:6px 0;border-left:2px solid #9b59b6;padding-left:8px'>"
                f"<span class='turn-id'>{escape(cid)}</span> [{escape(ctype)}/{escape(abst)}] "
                f"{extras_str}<br>"
                f"<span style='color:#bbb;font-size:11px'>{escape(rule)}</span>"
                f"</div>"
            )
        # Promote_to_1A appears in M4 (Reflexion), but M3 may also have
        # card_rewrites / card_discards / chain_rule (B16) — show all three.
        m3_extras = []
        for key in ("card_rewrites", "card_discards", "chain_rule",
                    "promote_to_1A"):
            v = hyp.get(key)
            if v:
                if isinstance(v, list):
                    m3_extras.append(f"<b>{key}:</b> {len(v)} entries")
                    if isinstance(v[0], dict):
                        m3_extras.append(
                            f"<pre style='font-size:10px;color:#aaa'>"
                            f"{escape(json.dumps(v, indent=2)[:1000])}</pre>"
                        )
                    else:
                        m3_extras.append(
                            f"<pre style='font-size:10px;color:#aaa'>"
                            f"{escape(str(v)[:500])}</pre>"
                        )
        m3_extras_html = "<br>".join(m3_extras) if m3_extras else ""
        if thought or cards_html_parts or m3_extras_html:
            hyp_section = (
                f'<details><summary>Hypothesize (M3) — thought + cards + emissions</summary>'
                f'<div class="reflexion" style="border-left-color:#9b59b6">'
                f"<b>thought:</b><br>{escape(thought)}"
                f"{''.join(cards_html_parts)}"
                f"{m3_extras_html}"
                f'</div></details>'
            )

    expected = s["expected_observation"]
    expected_str = ""
    if isinstance(expected, dict) and expected:
        expected_str = (
            f'<div class="section"><span class="label">expected:</span> '
            f'R={escape(str(expected.get("primary_region_id","?")))} '
            f'Δ={escape(str(expected.get("dominant_transition","?")))} '
            f'L+={expected.get("level_delta","?")}</div>'
        )

    # Coord grid mini-plot (16x16 ASCII).
    coord_grid_html = (
        f'<details><summary>coord plot (64×64 → 16×16 cells)</summary>'
        f'<pre style="font-size:10px;line-height:1.0;color:#5dade2;'
        f'background:#0d0d0d;padding:6px;border-radius:3px">'
        f'{escape(render_coord_grid(s["coord"], size=16))}</pre>'
        f'</details>'
    )

    # Visible regions compact list.
    vr_html = ""
    if s["visible_regions"]:
        n = len(s["visible_regions"])
        n_mc = sum(1 for r in s["visible_regions"]
                   if isinstance(r, dict) and r.get("is_multicolor"))
        vr_html = (
            f'<details><summary>visible_regions ({n} total, {n_mc} multicolor)'
            f'</summary><div class="thought" style="border-left-color:#27ae60">'
            f'{render_visible_regions_compact(s["visible_regions"])}</div></details>'
        )

    # Active hypotheses pool — IDs only at this trace level.
    ah_html = ""
    ah = s["active_hypotheses_after"]
    if ah:
        ids_str = ", ".join(str(c) for c in ah)
        ah_html = (
            f'<div class="section"><span class="label">active pool ({len(ah)}):</span>'
            f' <span class="region-id">{escape(ids_str)}</span></div>'
        )

    # Matched prior triggers (B9).
    mpt_html = ""
    if s["matched_prior_triggers"]:
        items = []
        for t in s["matched_prior_triggers"][:5]:
            if isinstance(t, dict):
                items.append(
                    f"R={escape(str(t.get('region_id','?')))} "
                    f"@t{t.get('turn','?')} coord={t.get('coord','?')} "
                    f"L+{t.get('level_at_event','?')}"
                )
        if items:
            mpt_html = (
                f'<details><summary>matched_prior_triggers ({len(s["matched_prior_triggers"])})'
                f'</summary><div class="thought" style="border-left-color:#16a085">'
                + "<br>".join(items) + '</div></details>'
            )

    # Summary after (M4 paragraph).
    sa_html = ""
    if s["summary_after"]:
        sa_html = (
            f'<details><summary>summary_after (M4)</summary>'
            f'<div class="reflexion">{escape(s["summary_after"][:1500])}</div>'
            f'</details>'
        )

    # Stuck mode + region clicks (B14).
    state_html = ""
    state_parts = []
    if s.get("stuck_mode") is not None:
        sm_label = "TRUE" if s["stuck_mode"] else "false"
        state_parts.append(f"stuck_mode=<b>{sm_label}</b>")
    if s.get("region_clicks_this_level"):
        clicks = s["region_clicks_this_level"]
        if isinstance(clicks, dict) and clicks:
            top_clicks = sorted(clicks.items(), key=lambda x: -x[1])[:8]
            click_str = ", ".join(f"{k}={v}" for k, v in top_clicks)
            state_parts.append(f"region_clicks: {click_str}")
    if state_parts:
        state_html = (
            f'<div class="section" style="font-size:11px;color:#aaa">'
            f'{" | ".join(state_parts)}</div>'
        )

    # B16 chain tokens (compact).
    chain_html = ""
    chain = s.get("action_state_chain") or {}
    if isinstance(chain, dict) and chain.get("chain_tokens"):
        tokens = chain["chain_tokens"][-15:]
        feats = chain.get("trajectory_features", {})
        errs = chain.get("prediction_errors", [])
        causal = chain.get("causal_table", [])
        tokens_str = "\n".join(tokens)
        feats_str = json.dumps(feats, indent=2)[:600] if feats else ""
        errs_str = json.dumps(errs[-5:], indent=2)[:500] if errs else ""
        causal_str = json.dumps(causal[:5], indent=2)[:500] if causal else ""
        chain_html = (
            f'<details><summary>action_state_chain (B16: {len(chain["chain_tokens"])} tokens)'
            f'</summary>'
            f'<pre style="font-size:10px;color:#5dade2;background:#0d0d0d;'
            f'padding:6px;border-radius:3px">{escape(tokens_str)}</pre>'
            f'<details><summary>trajectory_features</summary>'
            f'<pre style="font-size:10px;color:#bbb">{escape(feats_str)}</pre></details>'
            f'<details><summary>prediction_errors (last 5)</summary>'
            f'<pre style="font-size:10px;color:#bbb">{escape(errs_str)}</pre></details>'
            f'<details><summary>causal_table (top 5)</summary>'
            f'<pre style="font-size:10px;color:#bbb">{escape(causal_str)}</pre></details>'
            f'</details>'
        )

    # Marker neighbor states (B11).
    mns_html = ""
    mns = s.get("marker_neighbor_states") or []
    if isinstance(mns, list) and mns:
        rows = []
        for m in mns[:6]:
            if not isinstance(m, dict):
                continue
            mid = m.get("marker_id", "?")
            compass = m.get("compass") or {}
            compass_dirs = []
            for dir_key in ("N", "NE", "E", "SE", "S", "SW", "W", "NW"):
                cell = compass.get(dir_key)
                if isinstance(cell, dict):
                    rid = cell.get("region_id", "?")
                    color = cell.get("current_color", "?")
                    clicks = cell.get("clicks", 0)
                    compass_dirs.append(
                        f"{dir_key}={rid}@c{color}({clicks})"
                    )
            rows.append(
                f"<span class='turn-id'>{escape(str(mid))}</span> "
                + " ".join(compass_dirs)
            )
        if rows:
            mns_html = (
                f'<details><summary>marker_neighbor_states (B11, {len(mns)} markers)'
                f'</summary><div class="thought" style="border-left-color:#f39c12;font-size:11px">'
                + "<br>".join(rows) + '</div></details>'
            )

    # Analogous past segments (B14 CPSR).
    aps_html = ""
    aps = s.get("analogous_past_segments") or []
    if isinstance(aps, list) and aps:
        items = []
        for a in aps[:4]:
            if not isinstance(a, dict):
                continue
            kind = a.get("kind", "?")
            did = "✓ progress" if a.get("did_progress") else "✗ stuck"
            sig = a.get("abstract_signature", "")[:80]
            wc = (a.get("what_changed_in_next_5turns") or "")[:120]
            items.append(
                f"<b>{escape(kind)}</b> [{did}] sig={escape(sig)}<br>"
                f"<span style='color:#aaa;font-size:11px'>{escape(wc)}</span>"
            )
        aps_html = (
            f'<details><summary>analogous_past_segments (B14 CPSR, {len(aps)})'
            f'</summary><div class="thought" style="border-left-color:#e74c3c">'
            + "<br><br>".join(items) + '</div></details>'
        )

    return (
        f'<div class="{" ".join(classes)}" id="t{s["turn"]}">'
        f'<div class="turn-header">'
        f'<span class="turn-id">turn {s["turn"]}</span>'
        f'<span class="turn-coord">click=({s["coord"][0]},{s["coord"][1]})</span>'
        f'<span class="region-id">→ {escape(s["primary_region_id"])}</span>'
        f'{dt_str}'
        f'<span class="label">verdict={escape(str(s["verdict"]))}</span>'
        f'<span class="label">card={escape(str(s["chosen_hypothesis_id"] or "—"))}</span>'
        f'{" ".join(badges)}'
        f'</div>'
        f'<details open><summary>thought (M1)</summary>'
        f'<div class="thought">{escape(s["thought"])}</div></details>'
        f'{expected_str}'
        f'{coord_grid_html}'
        f'{state_html}'
        f'{vr_html}'
        f'{ah_html}'
        f'{mns_html}'
        f'{chain_html}'
        f'{aps_html}'
        f'{mpt_html}'
        f'{sa_html}'
        f'{rfl_section}'
        f'{hyp_section}'
        f'</div>'
    )


def build_html_for_cycle(cycle_dir: Path) -> tuple[str, dict]:
    """Return (html, stats) for the cycle. stats used in cross-cycle index."""
    trace_path = cycle_dir / "trace.jsonl"
    entries = load_trace(trace_path)
    if not entries:
        return f"<html><body>No trace at {trace_path}</body></html>", {}

    summaries = [turn_summary(e) for e in entries]
    oscs = detect_oscillation(summaries)
    lp_oscs = detect_lp_oscillation(summaries)

    n_turns = len(summaries)
    n_lp1 = sum(1 for s in summaries if s["level_delta"] >= 1)
    n_lp2 = sum(1 for s in summaries if s["level_delta"] >= 2)
    n_osc = sum(oscs)
    n_lposc = sum(lp_oscs)

    # L+ event quick-jump list.
    lp_jumps_html = ""
    lp_turn_list = [
        s for s in summaries if s["level_delta"] >= 1
    ]
    if lp_turn_list:
        items = []
        for s in lp_turn_list:
            ld = s["level_delta"]
            badge = "lp2" if ld >= 2 else "lp1"
            items.append(
                f'<a href="#t{s["turn"]}" class="tick {badge}" '
                f'style="width:auto;padding:2px 6px;text-decoration:none">'
                f'L+{ld}@T{s["turn"]} R={escape(s["primary_region_id"])}'
                f'</a>'
            )
        lp_jumps_html = (
            f'<div class="metadata">L+ events ({len(lp_turn_list)}): '
            + " ".join(items) + '</div>'
        )

    # Verdict / region / coord aggregate stats.
    verdict_counts: dict = {}
    region_clicks_total: dict = {}
    for s in summaries:
        v = str(s["verdict"] or "—")
        verdict_counts[v] = verdict_counts.get(v, 0) + 1
        rid = s["primary_region_id"] or "?"
        region_clicks_total[rid] = region_clicks_total.get(rid, 0) + 1
    top_regions = sorted(region_clicks_total.items(), key=lambda x: -x[1])[:8]
    aggregate_html = (
        f'<div class="metadata">'
        f'verdict_counts: {escape(str(verdict_counts))}<br>'
        f'top_clicked_regions: {escape(str(top_regions))}'
        f'</div>'
    )

    metadata = (
        f'<div class="metadata">'
        f'cycle: <b>{escape(cycle_dir.name)}</b><br>'
        f'game: {escape(cycle_dir.parent.name)}<br>'
        f'turns: <b>{n_turns}</b> · L+1 events: <b>{n_lp1}</b> · '
        f'L+2 events: <b>{n_lp2}</b> · oscillation turns: <b>{n_osc}</b> · '
        f'L+ oscillation: <b>{n_lposc}</b><br>'
        f'trace: {escape(str(trace_path))}'
        f'</div>'
        f'{lp_jumps_html}'
        f'{aggregate_html}'
    )

    # Cross-run memory + bridges block.
    crm_path = cycle_dir.parent / "cross_run_memory.json"
    bridges_path = cycle_dir.parent / "level_bridges.json"
    seg_path = cycle_dir.parent / "segment_index.json"
    memory_html = '<h2>Memory state (snapshot at viz-build time)</h2>'
    try:
        crm = json.loads(crm_path.read_text()) if crm_path.exists() else {}
        am = crm.get("abstract_mechanics", []) or []
        sig = [m for m in am if m.get("signature")]
        max_cfr = max((m.get("confirmed_runs", 0) for m in sig), default=0)
        memory_html += (
            f'<div class="metadata">cross_run_memory: total={len(am)} '
            f'signatured={len(sig)} max_cfr_signatured={max_cfr}</div>'
        )
    except Exception:
        pass
    try:
        b = json.loads(bridges_path.read_text()) if bridges_path.exists() else {}
        bridges = b.get("bridges", []) or []
        if bridges:
            rows = "".join(
                f'<tr><td>{escape(br.get("id",""))}</td>'
                f'<td>{escape(br.get("run_namespace","").split("_")[0])}</td>'
                f'<td>{br.get("click_count","")}</td>'
                f'<td>{br.get("unique_regions_clicked","")}</td>'
                f'<td>{br.get("repeat_clicks","")}</td>'
                f'<td>{escape(str(br.get("kind_distribution","")))}</td>'
                f'</tr>'
                for br in bridges[-8:]
            )
            memory_html += (
                f'<table><tr><th>id</th><th>cycle</th><th>clicks</th>'
                f'<th>unique</th><th>repeat</th><th>kind_dist</th></tr>'
                f'{rows}</table>'
            )
    except Exception:
        pass
    try:
        seg = json.loads(seg_path.read_text()) if seg_path.exists() else {}
        segs = seg.get("segments", [])
        if segs:
            from collections import Counter
            kinds = dict(Counter(s.get("kind") for s in segs))
            memory_html += (
                f'<div class="metadata">segment_index: total={len(segs)} '
                f'kinds={escape(str(kinds))}</div>'
            )
    except Exception:
        pass

    body_parts = [render_turn(s, oscs[i], lp_oscs[i]) for i, s in enumerate(summaries)]

    filter_ui = """
<div class="metadata" style="position:sticky;top:0;background:#1a1a1a;
     z-index:10;padding:8px;border-radius:4px">
  <input id="search" type="text" placeholder="filter turns: text in thought / coord / region (e.g. R5, L+1, click=)"
    style="width:60%;background:#0d0d0d;color:#e0e0e0;border:1px solid #444;
           padding:6px;font-family:monospace;font-size:12px"
    oninput="filterTurns()"/>
  <label style="margin-left:8px;font-size:11px">
    <input type="checkbox" id="onlyLp" onchange="filterTurns()"/> L+ only
  </label>
  <label style="margin-left:8px;font-size:11px">
    <input type="checkbox" id="onlyOsc" onchange="filterTurns()"/> osc only
  </label>
  <span id="filterCount" style="margin-left:12px;color:#aaa;font-size:11px"></span>
</div>
<script>
function filterTurns() {
  const q = (document.getElementById('search').value || '').toLowerCase();
  const onlyLp = document.getElementById('onlyLp').checked;
  const onlyOsc = document.getElementById('onlyOsc').checked;
  const turns = document.querySelectorAll('.turn');
  let shown = 0;
  turns.forEach(t => {
    const txt = t.innerText.toLowerCase();
    const isLp = t.classList.contains('lp1') || t.classList.contains('lp2');
    const isOsc = t.classList.contains('osc') || t.classList.contains('lposc');
    let show = true;
    if (q && !txt.includes(q)) show = false;
    if (onlyLp && !isLp) show = false;
    if (onlyOsc && !isOsc) show = false;
    t.style.display = show ? '' : 'none';
    if (show) shown += 1;
  });
  document.getElementById('filterCount').innerText = `showing ${shown}/${turns.length}`;
}
</script>
"""

    html = (
        '<!doctype html><html><head>'
        f'<meta charset="utf-8"><title>{escape(cycle_dir.name)}</title>'
        f'<style>{CSS}</style></head><body>'
        f'<h1>{escape(cycle_dir.name)} — trace viewer</h1>'
        f'{metadata}'
        f'{memory_html}'
        '<h2>Timeline (click a tick to jump)</h2>'
        f'{render_legend()}'
        f'{render_timeline(summaries, oscs, lp_oscs)}'
        f'{filter_ui}'
        '<h2>Per-turn detail</h2>'
        + "".join(body_parts) +
        '</body></html>'
    )

    stats = {
        "namespace": cycle_dir.name,
        "game": cycle_dir.parent.name,
        "turns": n_turns,
        "lp1": n_lp1,
        "lp2": n_lp2,
        "oscillation_turns": n_osc,
        "lp_oscillation_turns": n_lposc,
    }
    return html, stats


def build_index_html(stats_list: list[dict]) -> str:
    rows = []
    for s in sorted(stats_list, key=lambda x: x["namespace"], reverse=True):
        rows.append(
            f'<tr>'
            f'<td><a href="{escape(s["namespace"])}.html">{escape(s["namespace"])}</a></td>'
            f'<td>{escape(s["game"])}</td>'
            f'<td>{s["turns"]}</td>'
            f'<td>{s["lp1"]}</td>'
            f'<td>{s["lp2"]}</td>'
            f'<td>{s["oscillation_turns"]}</td>'
            f'<td>{s["lp_oscillation_turns"]}</td>'
            f'</tr>'
        )
    body = "".join(rows)
    return (
        f'<!doctype html><html><head><meta charset="utf-8">'
        f'<title>trace_viz index</title><style>{CSS}</style></head><body>'
        f'<h1>trace_viz index ({len(stats_list)} cycles)</h1>'
        f'<table>'
        f'<tr><th>namespace</th><th>game</th><th>turns</th>'
        f'<th>L+1</th><th>L+2</th><th>osc_turns</th><th>L+osc_turns</th></tr>'
        f'{body}</table></body></html>'
    )


# ---------------------------------------------------------------------
# CLI.
# ---------------------------------------------------------------------


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("namespace", nargs="?")
    p.add_argument("--all", action="store_true")
    p.add_argument("--latest", type=int, default=0)
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)

    targets: list[Path] = []
    if args.all:
        targets = list_all_cycles()
    elif args.latest > 0:
        targets = list_all_cycles()[: args.latest]
    elif args.namespace:
        d = find_cycle_dir(args.namespace)
        if not d:
            print(f"namespace not found: {args.namespace}", file=sys.stderr)
            return 1
        targets = [d]
    else:
        # Default: latest 5.
        targets = list_all_cycles()[:5]

    if not targets:
        print("no cycles found", file=sys.stderr)
        return 1

    stats_list = []
    for cd in targets:
        html, stats = build_html_for_cycle(cd)
        if not stats:
            continue
        out_path = OUT / f"{cd.name}.html"
        out_path.write_text(html, encoding="utf-8")
        stats_list.append(stats)
        print(f"  built: {out_path}")

    if len(stats_list) > 1 or args.all:
        idx_path = OUT / "index.html"
        idx_path.write_text(build_index_html(stats_list), encoding="utf-8")
        print(f"  index: {idx_path}")

    print(f"\nTotal cycles rendered: {len(stats_list)}")
    print(f"Open: file://{OUT.resolve()}/index.html")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
