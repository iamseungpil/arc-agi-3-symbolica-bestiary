#!/usr/bin/env python3
"""Build a step-by-step trace viewer (Symbolica-style, dark theme).

Renders:
- Header badges: game_id, level, total actions, state, finish status
- Grid panel with mini canvas (per-step before/after)
- D-pad with action meanings (ACTION1=Up, etc.) — highlights the taken action
- Per-step accordion:
    index | action | predicted | observed | skill-committed | surprise/wake
- Skill-library summary + observed-law highlights

Usage:
    python scripts/build_trace_viewer.py <namespace>
    # OR
    python scripts/build_trace_viewer.py --shared-dir research_logs/shared/ls20-cb3b57cc/<ns>

Input files (any missing ones are tolerated):
    research_logs/shared/<game_id>/<namespace>/world_model.json
    research_logs/shared/<game_id>/<namespace>/dreamcoder_library.json
    research_logs/shared/<game_id>/<namespace>/meta_harness.json
    research_logs/shared/<game_id>/<namespace>/planner_state.json
    research_logs/shared/<game_id>/<namespace>/memories.json
    recordings/<game_id>.<agent>.<guid>.recording.jsonl (most recent matching the run)

Output: `presentation/trace_viewer_<namespace>.html`
"""
from __future__ import annotations

import argparse
import html
import json
import re
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# Symbolica's GAME_REFERENCE action semantics
ACTION_SEMANTIC = {
    "ACTION1": ("Up", "↑"),
    "ACTION2": ("Down", "↓"),
    "ACTION3": ("Left", "←"),
    "ACTION4": ("Right", "→"),
    "ACTION5": ("Space", "␣"),
    "ACTION6": ("Click(x,y)", "◉"),
    "ACTION7": ("Aux", "*"),
    "RESET": ("Reset", "⟲"),
}

# 16-colour palette for grid render (matches Symbolica colour convention)
COLOR_PALETTE = [
    "#000000", "#0074D9", "#FF4136", "#2ECC40", "#FFDC00",
    "#AAAAAA", "#F012BE", "#FF851B", "#7FDBFF", "#870C25",
    "#555555", "#B10DC9", "#001F3F", "#7d4e4e", "#4e7d4e",
    "#4e4e7d",
]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def load_recording(recording_path: Path) -> list[dict[str, Any]]:
    if not recording_path.exists():
        return []
    out: list[dict[str, Any]] = []
    with recording_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out


def find_recording_for_namespace(namespace: str, game_prefix: str) -> Path | None:
    """Find the most recent .recording.jsonl matching this namespace by mtime.

    Recordings are named like
    `<game_id>.<agent>.<modules>.<guid>.recording.jsonl`; we pick the
    most recent one that also contains the game prefix."""
    rec_dir = REPO_ROOT / "recordings"
    if not rec_dir.exists():
        return None
    candidates = sorted(
        (p for p in rec_dir.glob("*.recording.jsonl") if game_prefix in p.name),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def grid_to_svg(grid: list[list[int]], *, max_side: int = 160) -> str:
    if not grid or not grid[0]:
        return ""
    rows = len(grid)
    cols = len(grid[0])
    cell = max(1, max_side // max(rows, cols))
    width = cell * cols
    height = cell * rows
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'style="image-rendering:pixelated;border:1px solid #52525b;border-radius:4px">'
    ]
    # Build a run-length encoded row to reduce SVG size.
    for r, row in enumerate(grid):
        c = 0
        while c < cols:
            val = int(row[c])
            run = 1
            while c + run < cols and int(row[c + run]) == val:
                run += 1
            colour = COLOR_PALETTE[val % 16]
            parts.append(
                f'<rect x="{c*cell}" y="{r*cell}" '
                f'width="{run*cell}" height="{cell}" fill="{colour}"/>'
            )
            c += run
    parts.append("</svg>")
    return "".join(parts)


# ---------------------------------------------------------------------------
# Failure-analysis helpers (pure appends — no existing code path touches these)
# ---------------------------------------------------------------------------

def grid_to_svg_diff(
    grid: list[list[int]],
    ref_grid: list[list[int]] | None = None,
    *,
    max_side: int = 140,
) -> str:
    """Render `grid` highlighting cells that differ from `ref_grid` in red."""
    if not grid or not grid[0]:
        return ""
    rows = len(grid)
    cols = len(grid[0])
    cell = max(1, max_side // max(rows, cols))
    width = cell * cols
    height = cell * rows
    parts = [
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'style="image-rendering:pixelated;border:1px solid #52525b;border-radius:4px">'
    ]
    ref_ok = (
        isinstance(ref_grid, list)
        and len(ref_grid) == rows
        and all(isinstance(r, list) and len(r) == cols for r in ref_grid)
    )
    for r, row in enumerate(grid):
        for c in range(cols):
            val = int(row[c])
            colour = COLOR_PALETTE[val % 16]
            parts.append(
                f'<rect x="{c*cell}" y="{r*cell}" '
                f'width="{cell}" height="{cell}" fill="{colour}"/>'
            )
            if ref_ok and int(ref_grid[r][c]) != val:
                parts.append(
                    f'<rect x="{c*cell}" y="{r*cell}" '
                    f'width="{cell}" height="{cell}" fill="none" '
                    f'stroke="#ef4444" stroke-width="1" opacity="0.9"/>'
                )
    parts.append("</svg>")
    return "".join(parts)


def _strip_code_fences(body: str) -> str:
    """Strip ```python ... ``` fences if present, else return as-is."""
    if not isinstance(body, str):
        return ""
    s = body.strip()
    if s.startswith("```"):
        # drop first line (```python) and trailing ```
        first_nl = s.find("\n")
        if first_nl != -1:
            s = s[first_nl + 1:]
        if s.rstrip().endswith("```"):
            s = s.rstrip()[:-3]
    return s


def _compile_predict_effect(body: str):
    """Compile the draft body and return its predict_effect callable, or None."""
    source = _strip_code_fences(body)
    if not source.strip():
        return None
    try:
        ns: dict[str, Any] = {}
        exec(compile(source, "<world_draft>", "exec"), ns)  # noqa: S102 — controlled input
        fn = ns.get("predict_effect")
        return fn if callable(fn) else None
    except Exception:
        return None


def _match_failure_tests(
    notes: list[str], unit_tests: list[dict], limit: int = 12,
) -> list[dict]:
    """Pair failure-ish notes with the best unit_test (by action name) for rendering.

    We keep it simple: walk the notes in order; each time we see a
    miss/mismatch/surprise note, we consume the next unit_test whose
    `diff_band` is nonzero OR whose `new_family` is True (best-guess).
    """
    failure_keywords = ("PREDICTION MISS", "mismatch", "surprise", "Surprise")
    # Pick the unit_tests that clearly represent prediction failures.
    failure_uts = [
        ut for ut in unit_tests
        if ut.get("new_family") or ut.get("diff_band") == "large"
    ]
    out: list[dict] = []
    ut_iter = iter(failure_uts)
    for note in notes:
        if not isinstance(note, str):
            continue
        if not any(k in note for k in failure_keywords):
            continue
        try:
            ut = next(ut_iter)
        except StopIteration:
            break
        out.append({"note": note, "unit_test": ut})
        if len(out) >= limit:
            break
    # Fallback: if no notes matched (e.g. only Simulator mismatch shorthand),
    # at least surface the top-N raw failing unit_tests.
    if not out:
        for ut in failure_uts[:limit]:
            out.append({"note": "Simulator mismatch (inferred)", "unit_test": ut})
    return out


def _parse_draft_scores(notes: list[str]) -> list[float]:
    """Extract the float score trail from 'Draft score now X.X' lines."""
    trail: list[float] = []
    pat = re.compile(r"Draft score now ([-+]?\d+(?:\.\d+)?)")
    for n in notes:
        if not isinstance(n, str):
            continue
        m = pat.search(n)
        if m:
            try:
                trail.append(float(m.group(1)))
            except ValueError:
                continue
    return trail


def _score_timeline_svg(scores: list[float]) -> str:
    """Render an inline sparkline SVG. Empty input returns empty string."""
    if not scores:
        return ""
    width = max(320, 24 * len(scores))
    height = 90
    pad = 10
    lo = min(scores)
    hi = max(scores)
    span = max(hi - lo, 1e-6)
    xs = [
        pad + (width - 2 * pad) * (i / max(len(scores) - 1, 1))
        for i in range(len(scores))
    ]
    ys = [
        height - pad - (height - 2 * pad) * ((s - lo) / span)
        for s in scores
    ]
    pts = " ".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs, ys))
    circles = "".join(
        f'<circle cx="{x:.1f}" cy="{y:.1f}" r="2.5" fill="#eab308"/>'
        for x, y in zip(xs, ys)
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" width="{width}" height="{height}" '
        f'style="background:#0e0e12;border:1px solid #3f3f46;border-radius:4px">'
        f'<polyline points="{pts}" fill="none" stroke="#eab308" stroke-width="2"/>'
        f'{circles}'
        f'<text x="{pad}" y="12" fill="#a1a1aa" font-size="10" font-family="monospace">min={lo:.2f}</text>'
        f'<text x="{width-pad}" y="12" fill="#a1a1aa" font-size="10" font-family="monospace" '
        f'text-anchor="end">max={hi:.2f}</text>'
        f'</svg>'
    )


def render_failure_gallery(world_model: dict) -> str:
    """Section 1 — Failure Gallery: before | predicted-text | after(+diff)."""
    notes = world_model.get("agent_env_notes", []) or []
    unit_tests = world_model.get("unit_tests", []) or []
    entries = _match_failure_tests(notes, unit_tests, limit=12)
    if not entries:
        return (
            '<h2>Failure Gallery</h2>'
            '<div style="color:#a1a1aa;font-size:13px">No prediction failures detected.</div>'
        )
    cards = []
    for e in entries:
        ut = e["unit_test"]
        note = e["note"]
        bg = ut.get("before_grid") or []
        ag = ut.get("after_grid") or []
        # Parse any "pred={...}" or "expected: ..." content from the note
        pred_text = ""
        m = re.search(r"pred=(\{[^}]+\})", note)
        if m:
            pred_text = m.group(1)
        else:
            m2 = re.search(r"expected ['\"]?([^'\"/]+)['\"]?", note)
            if m2:
                pred_text = m2.group(1).strip()
        action = str(ut.get("action", "?"))
        diff_mag = ut.get("diff_magnitude", "?")
        diff_band = ut.get("diff_band", "?")
        cards.append(f"""
        <div style="display:grid;grid-template-columns:160px 1fr 160px;gap:14px;
                    padding:12px;background:#27272a;border:1px solid #3f3f46;
                    border-radius:6px;margin-bottom:8px;align-items:start">
          <div>
            <div style="color:#a1a1aa;font-size:10px;margin-bottom:4px">before</div>
            {grid_to_svg(bg, max_side=140)}
          </div>
          <div style="font-family:monospace;font-size:11px;color:#fafafa">
            <div><span style="color:#a78bfa">action:</span> {html.escape(action)}</div>
            <div style="margin-top:4px"><span style="color:#a78bfa">diff:</span>
              band={html.escape(str(diff_band))} magnitude={html.escape(str(diff_mag))}</div>
            <div style="margin-top:6px;color:#a1a1aa">predicted:</div>
            <div style="padding:6px;background:#18181b;border-left:3px solid #a78bfa;
                        white-space:pre-wrap;max-height:120px;overflow:auto">
              {html.escape(pred_text or '(no predicted payload in note)')}
            </div>
            <div style="margin-top:6px;color:#a1a1aa">note:</div>
            <div style="padding:6px;background:#18181b;border-left:3px solid #ef4444;
                        white-space:pre-wrap">{html.escape(note[:400])}</div>
          </div>
          <div>
            <div style="color:#a1a1aa;font-size:10px;margin-bottom:4px">after (red = diff)</div>
            {grid_to_svg_diff(ag, bg, max_side=140)}
          </div>
        </div>""")
    return (
        f'<h2>Failure Gallery ({len(cards)} events)</h2>'
        + "".join(cards)
    )


def render_predict_inspector(world_model: dict) -> str:
    """Section 2 — Predict_effect Output Inspector.

    Execute the first draft's `predict_effect` against 3 sample tests.
    Safely fails (returns a diagnostic div) if draft cannot compile or run.
    """
    drafts = world_model.get("world_drafts", []) or []
    unit_tests = world_model.get("unit_tests", []) or []
    if not drafts or not unit_tests:
        return (
            '<h2>Predict_effect Output Inspector</h2>'
            '<div style="color:#a1a1aa;font-size:13px">No draft or unit tests to inspect.</div>'
        )
    body = drafts[0].get("body", "")
    fn = _compile_predict_effect(body)
    header = '<h2>Predict_effect Output Inspector</h2>'
    source_block = (
        '<details style="margin-bottom:10px"><summary style="color:#a1a1aa;font-size:12px;cursor:pointer">'
        'show compiled draft body</summary>'
        f'<pre style="background:#0e0e12;padding:10px;border-radius:4px;color:#e4e4e7;'
        f'font-size:11px;max-height:280px;overflow:auto">{html.escape(_strip_code_fences(body))}</pre>'
        '</details>'
    )
    if fn is None:
        return (
            header + source_block
            + '<div style="color:#ef4444;font-size:13px">Could not compile '
              '<code>predict_effect</code> from draft body.</div>'
        )
    n = len(unit_tests)
    idxs = [0, n // 2, n - 1] if n >= 3 else list(range(n))
    idxs = sorted(set(idxs))
    rows = []
    for i in idxs:
        ut = unit_tests[i]
        obs = ut.get("observation", {}) or {}
        action = str(ut.get("action", "?"))
        try:
            pred = fn(action, obs)
            err = ""
        except Exception as exc:  # noqa: BLE001 — surface any failure
            pred = {}
            err = f"exception: {type(exc).__name__}: {exc}"
        pred_items = []
        expected_next_grid = None
        if isinstance(pred, dict):
            for k, v in pred.items():
                if k == "expected_next_grid" and isinstance(v, list):
                    expected_next_grid = v
                    pred_items.append(f"{k}: <grid {len(v)}x{len(v[0]) if v and isinstance(v[0], list) else '?'}>")
                else:
                    rendered = repr(v)
                    if len(rendered) > 120:
                        rendered = rendered[:117] + "..."
                    pred_items.append(f"{k}: {rendered}")
        actual = (
            f"diff_band={ut.get('diff_band','?')} "
            f"diff_magnitude={ut.get('diff_magnitude','?')} "
            f"new_family={ut.get('new_family', False)}"
        )
        after_grid = ut.get("after_grid") or []
        cmp_svgs = ""
        if expected_next_grid is not None and after_grid:
            cmp_svgs = (
                '<div style="display:flex;gap:10px;margin-top:6px">'
                '<div><div style="color:#a1a1aa;font-size:10px">predicted grid</div>'
                f'{grid_to_svg(expected_next_grid, max_side=120)}</div>'
                '<div><div style="color:#a1a1aa;font-size:10px">actual (red=mismatch vs predicted)</div>'
                f'{grid_to_svg_diff(after_grid, expected_next_grid, max_side=120)}</div>'
                '</div>'
            )
        rows.append(f"""
        <div style="padding:10px;background:#27272a;border:1px solid #3f3f46;
                    border-radius:6px;margin-bottom:8px;font-family:monospace;font-size:11px">
          <div style="color:#a78bfa;margin-bottom:4px">unit_test #{i} · action={html.escape(action)}</div>
          <div style="color:#a1a1aa">predicted dict:</div>
          <pre style="background:#0e0e12;padding:6px;border-radius:3px;margin:2px 0;
                      white-space:pre-wrap;color:#e4e4e7">{html.escape(chr(10).join(pred_items) or '(empty)')}</pre>
          <div style="color:#a1a1aa">actual:</div>
          <div style="padding:4px 6px;background:#0e0e12;border-radius:3px;color:#22c55e">
            {html.escape(actual)}</div>
          {f'<div style="color:#ef4444;margin-top:4px">{html.escape(err)}</div>' if err else ''}
          {cmp_svgs}
        </div>""")
    return header + source_block + "".join(rows)


def render_score_timeline(world_model: dict) -> str:
    """Section 3 — Score timeline sparkline."""
    notes = world_model.get("agent_env_notes", []) or []
    scores = _parse_draft_scores(notes)
    if not scores:
        return (
            '<h2>Score Timeline</h2>'
            '<div style="color:#a1a1aa;font-size:13px">'
            'No "Draft score now X.X" events found in agent_env_notes.</div>'
        )
    svg = _score_timeline_svg(scores)
    return (
        '<h2>Score Timeline</h2>'
        f'<div style="color:#a1a1aa;font-size:12px;margin-bottom:6px">'
        f'{len(scores)} score samples · first={scores[0]:.2f} last={scores[-1]:.2f}</div>'
        f'{svg}'
    )


def render_mcts_call_log(planner_state: dict) -> str:
    """Section 4 — MCTS Call Log compact table from recent_advantages."""
    advs = planner_state.get("recent_advantages", []) if isinstance(planner_state, dict) else []
    if not isinstance(advs, list) or not advs:
        return (
            '<h2>MCTS Call Log</h2>'
            '<div style="color:#a1a1aa;font-size:13px">No recent_advantages recorded.</div>'
        )
    rows = []
    for i, a in enumerate(advs):
        try:
            a_f = float(a)
        except (TypeError, ValueError):
            a_f = 0.0
        selected = a_f != 0.0
        badge = (
            '<span style="padding:2px 6px;border-radius:3px;background:#166534;'
            'color:#22c55e;font-size:11px">selected</span>'
            if selected else
            '<span style="padding:2px 6px;border-radius:3px;background:#3f3f46;'
            'color:#a1a1aa;font-size:11px">skipped</span>'
        )
        rows.append(
            f'<tr><td style="padding:4px 10px;font-family:monospace;color:#a1a1aa">#{i}</td>'
            f'<td style="padding:4px 10px;font-family:monospace;color:#fafafa">{a_f:+.4f}</td>'
            f'<td style="padding:4px 10px">{badge}</td></tr>'
        )
    return (
        '<h2>MCTS Call Log</h2>'
        '<table style="border-collapse:collapse;background:#27272a;'
        'border:1px solid #3f3f46;border-radius:6px;overflow:hidden">'
        '<thead><tr style="background:#1f1f23;color:#a1a1aa;font-size:11px">'
        '<th style="padding:6px 10px;text-align:left">call#</th>'
        '<th style="padding:6px 10px;text-align:left">advantage</th>'
        '<th style="padding:6px 10px;text-align:left">was_selected</th>'
        '</tr></thead>'
        f'<tbody>{"".join(rows)}</tbody></table>'
    )


def dpad_html(available: list[str], taken: str | None) -> str:
    """Render a D-pad + extra buttons with the taken action highlighted."""
    btn_style = (
        "display:inline-flex;align-items:center;justify-content:center;"
        "width:40px;height:40px;border-radius:50%;background:#3f3f46;"
        "border:1px solid #52525b;color:#fafafa;font-size:18px;"
    )
    pill_style = (
        "display:inline-flex;align-items:center;justify-content:center;"
        "padding:8px 12px;border-radius:18px;background:#3f3f46;"
        "border:1px solid #52525b;color:#fafafa;font-size:12px;"
    )
    taken_glow = ";box-shadow:0 0 10px 2px #6366f1;background:#4f46e5"
    dim = ";opacity:.25"

    def style_for(action: str) -> str:
        s = btn_style
        if action in ACTION_SEMANTIC and len(ACTION_SEMANTIC[action][0]) <= 5:
            s = btn_style
        else:
            s = pill_style
        if taken == action:
            s = s + taken_glow
        if available and action not in available:
            s = s + dim
        return s

    def btn(action: str) -> str:
        label, icon = ACTION_SEMANTIC.get(action, (action, action))
        inner = icon if action in ("ACTION1", "ACTION2", "ACTION3", "ACTION4") else label
        return (
            f'<span style="{style_for(action)}" title="{html.escape(action)} ({html.escape(label)})">'
            f'{html.escape(inner)}</span>'
        )

    # D-pad layout + space + reset
    left, up, down, right = btn("ACTION3"), btn("ACTION1"), btn("ACTION2"), btn("ACTION4")
    return (
        '<div style="display:inline-flex;align-items:center;gap:20px">'
        f'<div style="display:inline-flex;align-items:center;gap:4px">{left}'
        f'<div style="display:flex;flex-direction:column;gap:2px">{up}{down}</div>'
        f'{right}</div>'
        f'<div style="display:inline-flex;gap:6px">{btn("ACTION5")}{btn("ACTION6")}{btn("RESET")}</div>'
        '</div>'
    )


def _extract_reasoning_for_step(
    idx: int, action: str, memory_events: list[dict],
) -> dict[str, str]:
    """Pull the LLM's thinking for this step from memory events.

    Memories are not perfectly ordered per step (some entries accumulate
    across multiple surprises), so we match on the action token and
    heuristically use the Nth-action occurrence.
    """
    out = {"predict": "", "world_update": "", "propose_skill": "", "surprise": "", "score": ""}
    # Filter surprise events for this step by action match, sequentially.
    surprises_for = [m for m in memory_events
                      if isinstance(m.get("summary"), str)
                      and "[World Surprise]" in m["summary"] and action in m["summary"]]
    if idx < len(surprises_for):
        det = str(surprises_for[idx].get("details", ""))
        exp_match = re.search(r"expected: ([^\n]+)", det)
        if exp_match:
            out["predict"] = exp_match.group(1).strip()[:400]
        out["surprise"] = det[:300]
    # World drafts (usually few total, attribute the most recent draft <= idx)
    drafts = [m for m in memory_events
              if isinstance(m.get("summary"), str) and "[World Draft]" in m["summary"]]
    if drafts:
        d = drafts[min(len(drafts) - 1, idx // 20)]
        det = str(d.get("details", ""))
        focus = re.search(r"event: (\w+)", det)
        body = re.search(r"body:\s*```python(.*?)```", det, re.DOTALL)
        if body:
            out["world_update"] = body.group(1).strip()[:600]
        elif focus:
            out["world_update"] = f"({focus.group(1)}) " + det[:300]
    # Skill proposals: match by ordering among [Skill] entries
    skills_all = [m for m in memory_events
                   if isinstance(m.get("summary"), str) and "[Skill]" in m["summary"]]
    if skills_all and idx < len(skills_all) * 3:  # rough match
        s = skills_all[min(len(skills_all) - 1, idx // 5)]
        summ = str(s.get("summary", "")).replace("[Skill] ", "")
        out["propose_skill"] = summ[:200]
    # Score events (empirical/simulator pass/fail)
    scores = [m for m in memory_events
              if isinstance(m.get("summary"), str) and "[World Score]" in m["summary"]]
    if idx < len(scores):
        det = str(scores[idx].get("details", ""))
        sm = re.search(r"matches: (\d+)\nmismatches: (\d+)\nscore: ([-\d.]+)", det)
        if sm:
            out["score"] = f"matches={sm.group(1)}/mismatches={sm.group(2)} draft_score={sm.group(3)}"
    return out


def summarise_step(
    *, idx: int, action: str, unit_test: dict | None, memory_events: list[dict],
    skill_committed: str | None, recording_frame: list | None,
    available: list[str],
) -> str:
    """Render one step as an expandable card."""
    label, icon = ACTION_SEMANTIC.get(action, (action, ""))
    # Collect predicted/observed from the unit_test if present
    if unit_test is not None:
        predicted_band = "?"
        actual_band = unit_test.get("diff_band", "?")
        new_family = unit_test.get("new_family", False)
        branch_escape = unit_test.get("branch_escape", False)
        before_sig = str(unit_test.get("before_signature", ""))[:12]
        after_sig = str(unit_test.get("after_signature", ""))[:12]
    else:
        actual_band = "?"
        new_family = False
        branch_escape = False
        before_sig = after_sig = ""
    outcome_badges = []
    if new_family:
        outcome_badges.append(
            '<span style="padding:2px 6px;border-radius:3px;background:#166534;color:#22c55e;font-size:11px">new_family</span>'
        )
    if branch_escape:
        outcome_badges.append(
            '<span style="padding:2px 6px;border-radius:3px;background:#4f46e5;color:#fff;font-size:11px">branch_escape</span>'
        )
    if actual_band == "large":
        outcome_badges.append(
            '<span style="padding:2px 6px;border-radius:3px;background:#7f1d1d;color:#ef4444;font-size:11px">large_diff</span>'
        )
    surprises_here = [
        m for m in memory_events
        if isinstance(m.get("summary"), str)
        and m["summary"].startswith("[World Surprise]")
        and action in m.get("summary", "")
    ]
    surprise_badge = ""
    if surprises_here:
        last = surprises_here[-1]
        det = str(last.get("details", ""))
        m = re.search(r"observed_diff: (\d+)", det)
        diff = m.group(1) if m else "?"
        surprise_badge = (
            f'<span style="padding:2px 6px;border-radius:3px;background:#7f1d1d;color:#ef4444;font-size:11px">SURPRISE diff={diff}</span>'
        )
    skill_html = ""
    if skill_committed:
        skill_html = (
            f'<span style="padding:2px 6px;border-radius:3px;background:#166534;color:#22c55e;font-size:11px">skill:{html.escape(skill_committed[:40])}</span>'
        )
    grid_svg = grid_to_svg(recording_frame[-1]) if recording_frame else ""
    dpad = dpad_html(available, action)
    reasoning = _extract_reasoning_for_step(idx, action, memory_events)

    reasoning_blocks = []
    if reasoning.get("predict"):
        reasoning_blocks.append(
            f'<div class="log-block chunk-predict" style="color:#a78bfa;opacity:.9;padding:6px 10px;background:#27272a;border-left:3px solid #a78bfa;margin-bottom:6px;font-family:monospace;font-size:12px"><b>predict:</b> {html.escape(reasoning["predict"])}</div>'
        )
    if reasoning.get("world_update"):
        reasoning_blocks.append(
            f'<div class="log-block chunk-world" style="color:#60a5fa;padding:6px 10px;background:#1e2a3f;border-left:3px solid #60a5fa;margin-bottom:6px;font-family:monospace;font-size:12px;white-space:pre-wrap;max-height:180px;overflow-y:auto"><b>world_update:</b>\n{html.escape(reasoning["world_update"])}</div>'
        )
    if reasoning.get("propose_skill"):
        reasoning_blocks.append(
            f'<div class="log-block chunk-skill" style="color:#22c55e;padding:6px 10px;background:#14261a;border-left:3px solid #22c55e;margin-bottom:6px;font-family:monospace;font-size:12px"><b>propose_skill:</b> {html.escape(reasoning["propose_skill"])}</div>'
        )
    if reasoning.get("score"):
        reasoning_blocks.append(
            f'<div class="log-block chunk-score" style="color:#eab308;padding:4px 10px;background:#26220a;font-family:monospace;font-size:11px">score: {html.escape(reasoning["score"])}</div>'
        )

    reasoning_html = (
        f'<div style="margin-top:12px;padding:10px;background:#0e0e12;border-radius:4px">'
        f'<div style="color:#a1a1aa;font-size:11px;margin-bottom:6px">agent reasoning for this step</div>'
        + "".join(reasoning_blocks) + '</div>'
    ) if reasoning_blocks else ""

    return f"""
    <details class="step-row" id="step-{idx}" open>
      <summary style="padding:10px 14px;background:#27272a;border-radius:6px;margin-bottom:4px">
        <span style="display:inline-flex;gap:10px;align-items:center">
          <span style="font-family:monospace;color:#a1a1aa;min-width:3em">#{idx:03d}</span>
          <span style="font-family:monospace;background:#4f46e5;color:#fff;padding:2px 8px;border-radius:3px">{html.escape(action)}</span>
          <span style="font-size:18px">{html.escape(icon)}</span>
          <span style="color:#a1a1aa">{html.escape(label)}</span>
          {' '.join(outcome_badges)}
          {surprise_badge}
          {skill_html}
        </span>
      </summary>
      <div style="padding:14px 14px 18px;background:#18181b;border:1px solid #3f3f46;border-top:none;border-radius:0 0 6px 6px;margin-bottom:8px;display:grid;grid-template-columns:180px 1fr;gap:18px;align-items:start">
        <div>
          {grid_svg}
          <div style="color:#a1a1aa;font-size:10px;margin-top:4px;font-family:monospace">
            {html.escape(before_sig)} → {html.escape(after_sig)}
          </div>
        </div>
        <div>
          <div style="color:#a1a1aa;font-size:11px;margin-bottom:6px">available actions</div>
          {dpad}
          <div style="color:#a1a1aa;font-size:11px;margin-top:12px">diff_band: <b style="color:#fafafa">{html.escape(actual_band)}</b></div>
          {reasoning_html}
        </div>
      </div>
    </details>"""


def build_trace_html(
    *, namespace: str, world_model: dict, skills: list, meta_harness: dict,
    planner_state: dict, memories: list, recording: list[dict],
) -> str:
    unit_tests = world_model.get("unit_tests", [])
    run_history = meta_harness.get("run_history", [])
    last_metrics = run_history[-1].get("metrics", {}) if run_history else {}

    game_id = ""
    if recording:
        game_id = recording[0].get("data", {}).get("game_id", "")

    # Extract mcts_proposal first-actions for skill badge matching.
    mcts_first_actions: dict[str, str] = {}
    for item in skills:
        payload = item.get("payload", {}) or {}
        if payload.get("kind") != "mcts_proposal":
            continue
        spine = payload.get("action_spine") or payload.get("body") or []
        if spine and isinstance(spine, list) and int(item.get("times_selected", 0) or 0) > 0:
            mcts_first_actions.setdefault(str(spine[0]), str(payload.get("name", "")))

    # Build per-step rows. Align by recording index; unit_tests is one-per-real-action.
    step_rows = []
    # Map unit_tests by order
    ut_by_idx = list(unit_tests)
    # Recording: index 0 is initial frame; 1..N are after action 1..N.
    recording_frames: list[list] = []
    actions_from_recording: list[str] = []
    for rec in recording:
        d = rec.get("data", {})
        frame_list = d.get("frame", [])
        recording_frames.append(frame_list if frame_list else [[]])
        ai = d.get("action_input", {}) or {}
        aid = ai.get("id", 0)
        from_recording_action = (
            f"ACTION{aid}" if isinstance(aid, int) and 1 <= aid <= 7
            else "RESET" if aid == 0 else f"ACTION{aid}"
        )
        actions_from_recording.append(from_recording_action)

    for i, ut in enumerate(ut_by_idx):
        action = str(ut.get("action", "?"))
        available = ut.get("observation", {}).get("available_actions", [])
        available_names = [f"ACTION{a}" if isinstance(a, int) else str(a) for a in (available or [])]
        # Pair with recording frame i+1 if available (frame 0 is init).
        rec_frame = recording_frames[i + 1] if i + 1 < len(recording_frames) else None
        step_rows.append(summarise_step(
            idx=i,
            action=action,
            unit_test=ut,
            memory_events=memories,
            skill_committed=mcts_first_actions.get(action),
            recording_frame=rec_frame,
            available=available_names,
        ))

    # Observed-law summary cards
    law_cards = []
    for item in skills:
        payload = item.get("payload", {}) or {}
        if payload.get("kind") != "observed_law":
            continue
        name = payload.get("name", "")
        obs = payload.get("observation_summary", {})
        law_cards.append(f"""
        <div style="padding:12px;background:#27272a;border:1px solid #3f3f46;border-radius:6px;margin-bottom:8px">
          <div style="font-family:monospace;color:#22c55e;font-size:13px">{html.escape(name)}</div>
          <div style="color:#a1a1aa;font-size:11px;margin-top:4px">n={obs.get('n',0)} mean_diff={obs.get('mean_diff',0):.1f} new_family={obs.get('new_family_rate',0)*100:.0f}% branch_escape={obs.get('branch_escape_rate',0)*100:.0f}%</div>
          <div style="color:#eab308;font-size:11px;margin-top:6px">{html.escape(str(payload.get('implication',''))[:200])}</div>
        </div>""")

    # KPI badges
    kpi_bar = f"""
    <div style="display:flex;gap:8px;flex-wrap:wrap;padding:12px 0">
      <span style="padding:6px 12px;background:#27272a;border-radius:4px;font-family:monospace">game={html.escape(game_id or '?')}</span>
      <span style="padding:6px 12px;background:#27272a;border-radius:4px">actions={last_metrics.get('total_actions', len(ut_by_idx))}</span>
      <span style="padding:6px 12px;background:#166534;color:#22c55e;border-radius:4px">accuracy={world_model.get('transition_accuracy', 0):.2f}</span>
      <span style="padding:6px 12px;background:#27272a;border-radius:4px">falsification={last_metrics.get('falsification_probe_count', 0)}</span>
      <span style="padding:6px 12px;background:#27272a;border-radius:4px">branch_escape={last_metrics.get('branch_escape_count', 0)}</span>
      <span style="padding:6px 12px;background:{('#7f1d1d' if last_metrics.get('new_family_count', 0) <= 1 else '#166534')};color:{'#ef4444' if last_metrics.get('new_family_count', 0) <= 1 else '#22c55e'};border-radius:4px">new_family={last_metrics.get('new_family_count', 0)}</span>
    </div>"""

    # Symbolica-style clickable timeline ticks (sticky top).
    tick_html = []
    for i, ut in enumerate(ut_by_idx):
        act = str(ut.get("action", "?"))
        # colour tick by outcome: green = new_family, red = surprise, grey = neither
        if ut.get("new_family"):
            colour = "#22c55e"
        elif ut.get("branch_escape"):
            colour = "#60a5fa"
        else:
            colour = "#52525b"
        tick_html.append(
            f'<a href="#step-{i}" title="#{i:03d} {act}" '
            f'style="display:inline-block;width:8px;height:20px;background:{colour};margin:0 1px;border-radius:1px;vertical-align:middle" '
            f'onclick="document.getElementById(\'step-{i}\').scrollIntoView({{behavior:\'smooth\',block:\'start\'}});return false;"></a>'
        )
    timeline = (
        '<div class="timeline-bar">'
        '<span style="color:#a1a1aa;font-size:12px;margin-right:12px">Timeline (click to jump)</span>'
        + "".join(tick_html) +
        '<span style="color:#a1a1aa;font-size:11px;margin-left:16px">'
        '<span style="color:#22c55e">■</span> new_family &nbsp;'
        '<span style="color:#60a5fa">■</span> branch_escape &nbsp;'
        '<span style="color:#52525b">■</span> same family'
        '</span>'
        '</div>'
    )

    # ---------------- Pure-append failure-analysis sections ----------------
    # These are computed here and rendered ABOVE the existing timeline so
    # failure cases are immediately visible without disturbing any existing
    # render function above this point.
    failure_gallery_html = render_failure_gallery(world_model)
    predict_inspector_html = render_predict_inspector(world_model)
    score_timeline_html = render_score_timeline(world_model)
    mcts_log_html = render_mcts_call_log(planner_state)

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Trace: {html.escape(namespace)}</title>
<style>
  body {{
    background:#18181b;color:#fafafa;
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
    margin:0;padding:24px 32px 96px;
  }}
  h1 {{ margin:0 0 4px;font-size:22px }}
  h2 {{ margin:28px 0 12px;font-size:16px;color:#a1a1aa }}
  details summary {{ cursor:pointer;list-style:none }}
  details summary::-webkit-details-marker {{ display:none }}
  .step-row {{ margin-bottom:2px;scroll-margin-top:80px }}
  .action-legend {{
    display:flex;gap:16px;flex-wrap:wrap;padding:12px;
    background:#27272a;border-radius:6px;margin-bottom:16px;font-size:12px;color:#a1a1aa
  }}
  .action-legend b {{ color:#fafafa;font-family:monospace;margin-right:4px }}
  .timeline-bar {{
    position:sticky;top:0;z-index:10;
    background:#0b0b0f;padding:10px 16px;border-bottom:1px solid #3f3f46;
    margin:0 -32px 16px;white-space:nowrap;overflow-x:auto
  }}
  .log-block {{ font-family:ui-monospace,'SF Mono',monospace }}
</style></head><body>
  <div class="timeline-bar">{timeline}</div>
  <h1>Step-by-step trace</h1>
  <div style="color:#a1a1aa;font-family:monospace;font-size:12px">namespace: {html.escape(namespace)}</div>
  {kpi_bar}
  <div class="action-legend">
    <span><b>ACTION1</b> Up ↑</span>
    <span><b>ACTION2</b> Down ↓</span>
    <span><b>ACTION3</b> Left ←</span>
    <span><b>ACTION4</b> Right →</span>
    <span><b>ACTION5</b> Space/Enter</span>
    <span><b>ACTION6</b> Click(x,y)</span>
    <span><b>RESET</b> Restart level ⟲</span>
  </div>
  <div style="color:#a1a1aa;font-size:12px;margin-bottom:14px">
    Each step card shows: action+direction icon, grid SVG, D-pad (glowing=last-taken),
    outcome badges, <b>agent reasoning</b> (purple=predict, blue=world_update, green=skill, yellow=score).
  </div>
  {failure_gallery_html}
  {predict_inspector_html}
  {score_timeline_html}
  {mcts_log_html}
  <h2>Observed laws (consolidated knowledge)</h2>
  {''.join(law_cards) if law_cards else '<div style="color:#a1a1aa;font-size:13px">No observed-law skills yet — consolidation fires every 10 actions.</div>'}
  <h2>Per-step timeline ({len(step_rows)} steps)</h2>
  {''.join(step_rows)}
</body></html>"""


# =====================================================================
# BESTIARY VIEW (append-only, v3.15+)
# ---------------------------------------------------------------------
# Renders a standalone multi-namespace index page telling the
# "LLM cheating-mode bestiary" story across v3.6 → v3.15 runs.
# All code below is PURE ADDITION — no existing function is touched.
# Output: presentation/bestiary.html
# CLI:    python scripts/build_trace_viewer.py --bestiary
# =====================================================================

SHARED_ROOT = REPO_ROOT / "research_logs" / "shared" / "ls20-cb3b57cc"
PRESENTATION_ROOT = REPO_ROOT / "presentation"
WORLD_MODEL_SRC = REPO_ROOT / "research_extensions" / "modules" / "world_model.py"

# Each card = one cheating mode. `namespace` is a prefix; we take the most
# recent matching dir so the bestiary keeps working as new runs land.
BESTIARY_MODES: list[dict[str, str]] = [
    {
        "key": "wildcard",
        "title": "Wildcard cheat",
        "namespace_prefix": "arcgentica_v3_6_sym_prompt_",
        "scored": "transition_accuracy = 1.000 (reported)",
        "actual": "Every prediction returned `'unknown'` for diff_band / "
                  "`None` for expect_change → matcher (pre-v3.9) accepted "
                  "these as wildcards and awarded a pass.",
        "fix": "v3.9 strict matcher: `unknown`/`None` now FAIL.",
        "verdict": "cheat",
    },
    {
        "key": "one_cell",
        "title": "1-cell delta cheat",
        "namespace_prefix": "arcgentica_v3_11_strict_penalty_",
        "scored": "passes (pre-v3.12) by emitting tiny `expected_grid_delta`",
        "actual": "0 `expected_next_grid` emits across 160 tests — agent "
                  "returned 1-cell deltas that maximised precision while "
                  "ignoring the rest of the 64×64 board.",
        "fix": "v3.12 full_grid mandatory: delta-only → FAIL.",
        "verdict": "cheat",
    },
    {
        "key": "identity",
        "title": "Identity cheat",
        "namespace_prefix": "arcgentica_v3_14_wake_fix_",
        "scored": "passes pixel check by returning `new = [row[:] for row in g]`",
        "actual": "Predicted grid == observation['grid']; the do-nothing "
                  "branch trivially matches when the env is quiescent and "
                  "dominates the evaluation window.",
        "fix": "v3.15 identity guard: predicted == before AND diff_band != "
               "'zero' → FAIL.",
        "verdict": "cheat",
    },
    {
        "key": "current",
        "title": "v3.15+ current run",
        "namespace_prefix": "arcgentica_v3_15_",
        "scored": "strict + identity guard active",
        "actual": "Most recent v3.15* run — status live, no free lunch.",
        "fix": "next: v3.16 calibrated-confidence scoring.",
        "verdict": "live",
    },
]


def _latest_namespace(prefix: str) -> str | None:
    """Return the most recent shared-dir namespace whose name starts with `prefix`.

    Dirs are sorted lexicographically; the suffix is a YYYYMMDD_HHMM
    timestamp so lexicographic == chronological."""
    if not SHARED_ROOT.exists():
        return None
    candidates = sorted(
        p.name for p in SHARED_ROOT.iterdir()
        if p.is_dir() and p.name.startswith(prefix)
    )
    return candidates[-1] if candidates else None


def _load_world_model_safe(namespace: str) -> dict[str, Any]:
    if not namespace:
        return {}
    return load_json(SHARED_ROOT / namespace / "world_model.json")


def _extract_code_block_simple(body: str) -> str:
    """Mirror `WorldModelModule._extract_code_block` without importing
    the whole module (keeps bestiary renderable even if the import
    chain is broken)."""
    if not body:
        return ""
    matches = re.findall(r"```python\s*(.*?)```", body, flags=re.DOTALL | re.IGNORECASE)
    for block in matches:
        if "def predict_effect" in block:
            return block.strip()
    if matches:
        return matches[0].strip()
    return body if "def predict_effect" in body else ""


def _honest_transition_accuracy(namespace: str) -> dict[str, Any]:
    """Re-score this namespace's `world_drafts[0].body` against its own
    `unit_tests` using the CURRENT strict + identity-guard matcher.

    Returns a dict with:
      - passed, total, accuracy
      - reported_accuracy (what the run stored on disk)
      - error: optional string if re-evaluation failed

    Importing `WorldModelModule` may fail in a stripped tree; we
    fall back gracefully to the reported value in that case."""
    wm = _load_world_model_safe(namespace)
    reported = wm.get("transition_accuracy")
    drafts = wm.get("world_drafts") or []
    tests = wm.get("unit_tests") or []
    if not drafts or not tests:
        return {
            "passed": 0, "total": 0, "accuracy": None,
            "reported_accuracy": reported,
            "error": "no drafts or no unit tests on disk",
        }
    body = drafts[0].get("body", "")
    code = _extract_code_block_simple(body)
    if not code:
        return {
            "passed": 0, "total": len(tests), "accuracy": 0.0,
            "reported_accuracy": reported,
            "error": "draft body contained no executable predict_effect",
        }
    try:
        sys.path.insert(0, str(REPO_ROOT))
        from research_extensions.modules.world_model import WorldModelModule  # type: ignore
        matcher = WorldModelModule._unit_test_matches
    except Exception as e:
        return {
            "passed": 0, "total": len(tests), "accuracy": None,
            "reported_accuracy": reported,
            "error": f"cannot import WorldModelModule: {e}",
        }

    globals_dict = {
        "__builtins__": {
            "len": len, "min": min, "max": max, "sum": sum, "abs": abs,
            "sorted": sorted, "range": range, "tuple": tuple,
            "list": list, "dict": dict, "set": set,
        }
    }
    locals_dict: dict[str, Any] = {}
    try:
        exec(code, globals_dict, locals_dict)  # noqa: S102 — trusted self-authored code
    except Exception as e:
        return {
            "passed": 0, "total": len(tests), "accuracy": 0.0,
            "reported_accuracy": reported,
            "error": f"draft exec error: {e}",
        }
    fn = locals_dict.get("predict_effect") or globals_dict.get("predict_effect")
    if not callable(fn):
        return {
            "passed": 0, "total": len(tests), "accuracy": 0.0,
            "reported_accuracy": reported,
            "error": "predict_effect not defined after exec",
        }
    passed = 0
    for test in tests:
        try:
            pred = fn(test["action"], test["observation"])
        except Exception:
            continue
        if not isinstance(pred, dict):
            continue
        try:
            if matcher(pred, test):
                passed += 1
        except Exception:
            continue
    total = len(tests)
    return {
        "passed": passed,
        "total": total,
        "accuracy": (passed / total) if total else 0.0,
        "reported_accuracy": reported,
        "error": None,
    }


def _extract_prompt_contract_text() -> str:
    """Pull the CWM contract snippet out of `world_model.py::prompt_overlay`
    so the bestiary always shows the live contract rather than a stale
    copy."""
    if not WORLD_MODEL_SRC.exists():
        return "(world_model.py not found)"
    try:
        src = WORLD_MODEL_SRC.read_text(encoding="utf-8").splitlines()
    except Exception as e:
        return f"(read error: {e})"
    # Find the prompt_overlay function and slice lines between
    # `REAL WORLD MODEL contract` start and the closing triple-backtick
    # docstring sentinel.
    start = None
    end = None
    for i, line in enumerate(src):
        if start is None and "REAL WORLD MODEL contract" in line:
            start = i
        elif start is not None and "ANTI-CHEAT" in line:
            # keep through the end of this string literal
            for j in range(i, min(i + 15, len(src))):
                if src[j].rstrip().endswith('",'):
                    end = j + 1
                    break
            if end is None:
                end = i + 10
            break
    if start is None or end is None:
        return "(contract block not found in prompt_overlay)"
    block = "\n".join(src[start:end])
    # Strip leading Python-string indentation artifacts for readability.
    block = re.sub(r'^\s*"', "", block, flags=re.MULTILINE)
    block = re.sub(r'"\s*$', "", block, flags=re.MULTILINE)
    block = block.replace("\\n", "\n")
    return block


def _svg_loop_diagram() -> str:
    """Inline SVG: IMAGINE → ACT → WAKE → SLEEP clockwise loop."""
    return """
<svg viewBox="0 0 620 360" xmlns="http://www.w3.org/2000/svg"
     style="width:100%;max-width:720px;height:auto;background:#0b0b0f;border-radius:8px">
  <defs>
    <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5"
            markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="#22c55e"/>
    </marker>
  </defs>
  <!-- nodes -->
  <g font-family="ui-monospace,monospace" font-size="13" fill="#fafafa">
    <rect x="40"  y="40"  width="180" height="70" rx="8" fill="#27272a" stroke="#22c55e"/>
    <text x="130" y="68" text-anchor="middle" font-weight="bold">IMAGINE</text>
    <text x="130" y="92" text-anchor="middle" fill="#a1a1aa">MCTS on predict_effect</text>

    <rect x="400" y="40"  width="180" height="70" rx="8" fill="#27272a" stroke="#22c55e"/>
    <text x="490" y="68" text-anchor="middle" font-weight="bold">ACT</text>
    <text x="490" y="92" text-anchor="middle" fill="#a1a1aa">skill body on real env</text>

    <rect x="400" y="250" width="180" height="70" rx="8" fill="#27272a" stroke="#ef4444"/>
    <text x="490" y="278" text-anchor="middle" font-weight="bold">WAKE</text>
    <text x="490" y="302" text-anchor="middle" fill="#a1a1aa">rewrite on surprise</text>

    <rect x="40"  y="250" width="180" height="70" rx="8" fill="#27272a" stroke="#60a5fa"/>
    <text x="130" y="278" text-anchor="middle" font-weight="bold">SLEEP</text>
    <text x="130" y="302" text-anchor="middle" fill="#a1a1aa">DC refactors library</text>
  </g>
  <!-- arrows (clockwise) -->
  <g stroke="#22c55e" stroke-width="2" fill="none" marker-end="url(#arrow)">
    <path d="M 220,75 L 400,75"/>
    <path d="M 490,110 L 490,250"/>
    <path d="M 400,285 L 220,285"/>
    <path d="M 130,250 L 130,110"/>
  </g>
  <!-- arrow labels -->
  <g font-family="ui-monospace,monospace" font-size="11" fill="#22c55e">
    <text x="310" y="68"  text-anchor="middle">imagine_and_maybe_commit</text>
    <text x="505" y="185" text-anchor="start">on_skill_step</text>
    <text x="310" y="278" text-anchor="middle">wake_prompt_section</text>
    <text x="118" y="185" text-anchor="end">sleep_refactor</text>
  </g>
</svg>
""".strip()


def _svg_progression_chart(points: list[tuple[str, float | None, float | None]]) -> str:
    """Inline SVG line chart. `points` is a list of
    (version-label, honest_acc, reported_acc). Honest in green, reported
    in grey. None values are plotted as open circles at 0 with a tag."""
    w, h = 640, 260
    pad_l, pad_r, pad_t, pad_b = 50, 20, 24, 52
    plot_w = w - pad_l - pad_r
    plot_h = h - pad_t - pad_b
    n = max(len(points), 1)
    step = plot_w / max(n - 1, 1) if n > 1 else 0

    def y_of(v: float) -> float:
        return pad_t + (1.0 - max(0.0, min(1.0, v))) * plot_h

    axes = [
        f'<rect x="0" y="0" width="{w}" height="{h}" fill="#0b0b0f" rx="8"/>',
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{h - pad_b}" stroke="#3f3f46"/>',
        f'<line x1="{pad_l}" y1="{h - pad_b}" x2="{w - pad_r}" y2="{h - pad_b}" stroke="#3f3f46"/>',
    ]
    # y-axis ticks at 0, 0.5, 1.0
    for val in (0.0, 0.5, 1.0):
        y = y_of(val)
        axes.append(
            f'<line x1="{pad_l - 4}" y1="{y}" x2="{w - pad_r}" y2="{y}" '
            f'stroke="#27272a" stroke-dasharray="2,4"/>'
        )
        axes.append(
            f'<text x="{pad_l - 8}" y="{y + 4}" text-anchor="end" '
            f'font-family="ui-monospace,monospace" font-size="11" '
            f'fill="#a1a1aa">{val:.1f}</text>'
        )

    honest_pts: list[str] = []
    reported_pts: list[str] = []
    x_labels: list[str] = []
    for i, (label, honest, reported) in enumerate(points):
        x = pad_l + i * step
        x_labels.append(
            f'<text x="{x}" y="{h - pad_b + 18}" text-anchor="middle" '
            f'font-family="ui-monospace,monospace" font-size="11" fill="#a1a1aa" '
            f'transform="rotate(-18 {x} {h - pad_b + 18})">{html.escape(label)}</text>'
        )
        if honest is not None:
            honest_pts.append(f"{x},{y_of(honest)}")
            axes.append(
                f'<circle cx="{x}" cy="{y_of(honest)}" r="5" fill="#22c55e"/>'
            )
            axes.append(
                f'<text x="{x}" y="{y_of(honest) - 10}" text-anchor="middle" '
                f'font-family="ui-monospace,monospace" font-size="10" '
                f'fill="#22c55e">{honest:.2f}</text>'
            )
        else:
            axes.append(
                f'<circle cx="{x}" cy="{y_of(0.0)}" r="5" fill="none" stroke="#71717a"/>'
            )
        if reported is not None:
            reported_pts.append(f"{x},{y_of(reported)}")
            axes.append(
                f'<circle cx="{x}" cy="{y_of(reported)}" r="3" fill="#a1a1aa"/>'
            )
    if len(honest_pts) >= 2:
        axes.append(
            f'<polyline points="{" ".join(honest_pts)}" '
            f'fill="none" stroke="#22c55e" stroke-width="2"/>'
        )
    if len(reported_pts) >= 2:
        axes.append(
            f'<polyline points="{" ".join(reported_pts)}" '
            f'fill="none" stroke="#71717a" stroke-width="1" stroke-dasharray="4,3"/>'
        )
    # legend
    axes.append(
        f'<g font-family="ui-monospace,monospace" font-size="11">'
        f'<circle cx="{w - pad_r - 180}" cy="16" r="4" fill="#22c55e"/>'
        f'<text x="{w - pad_r - 170}" y="20" fill="#fafafa">honest (strict+identity guard)</text>'
        f'<circle cx="{w - pad_r - 60}" cy="16" r="3" fill="#a1a1aa"/>'
        f'<text x="{w - pad_r - 52}" y="20" fill="#fafafa">reported</text>'
        f'</g>'
    )
    return (
        f'<svg viewBox="0 0 {w} {h}" xmlns="http://www.w3.org/2000/svg" '
        f'style="width:100%;max-width:720px;height:auto;background:#0b0b0f;border-radius:8px">'
        + "".join(axes)
        + "".join(x_labels)
        + "</svg>"
    )


def _bestiary_card(mode: dict[str, str]) -> tuple[str, str, float | None, float | None]:
    """Render one bestiary card + return the (namespace, label, honest, reported)
    row for the progression chart."""
    ns = _latest_namespace(mode["namespace_prefix"])
    honest = None
    reported = None
    status_html = ""
    if ns is None:
        status_html = (
            '<div style="color:#71717a;font-style:italic">pending — no matching '
            'shared dir yet.</div>'
        )
        label = mode["namespace_prefix"].rstrip("_")
        return (
            f'<div class="card pending">'
            f'<div class="card-title">{html.escape(mode["title"])} '
            f'<span class="badge badge-pending">pending</span></div>'
            f'{status_html}</div>',
            label, None, None,
        )
    scored_result = _honest_transition_accuracy(ns)
    honest = scored_result.get("accuracy")
    reported = scored_result.get("reported_accuracy")
    verdict = mode["verdict"]
    badge_cls = {
        "cheat": "badge-cheat",
        "live": "badge-live",
    }.get(verdict, "badge-pending")
    # link to per-ns trace viewer if it exists
    trace_path = PRESENTATION_ROOT / f"trace_viewer_{ns}.html"
    trace_link = (
        f'<a href="trace_viewer_{html.escape(ns)}.html" class="trace-link">'
        f'→ open per-step trace</a>'
        if trace_path.exists() else
        '<span style="color:#71717a;font-style:italic">'
        '(per-step trace not yet built for this namespace)</span>'
    )
    reported_s = (
        f"{reported:.3f}" if isinstance(reported, (int, float)) else "n/a"
    )
    honest_s = (
        f"{honest:.3f}" if isinstance(honest, (int, float)) else "n/a"
    )
    err_row = ""
    if scored_result.get("error"):
        err_row = (
            f'<div style="color:#f59e0b;font-size:11px;font-family:monospace">'
            f'eval note: {html.escape(str(scored_result["error"]))}</div>'
        )
    card = f"""
<div class="card">
  <div class="card-title">
    {html.escape(mode['title'])}
    <span class="badge {badge_cls}">{html.escape(verdict)}</span>
  </div>
  <div class="card-ns">namespace: <code>{html.escape(ns)}</code></div>
  <table class="card-table">
    <tr><td>Scored as (gamed)</td><td>{html.escape(mode['scored'])}</td></tr>
    <tr><td>Actual behaviour</td><td>{html.escape(mode['actual'])}</td></tr>
    <tr><td>Reported accuracy</td><td><code>{reported_s}</code></td></tr>
    <tr><td>Honest accuracy <span style="color:#a1a1aa">(strict+identity)</span></td>
        <td><code>{honest_s}</code> <span style="color:#a1a1aa">
        = {scored_result.get('passed', 0)}/{scored_result.get('total', 0)}
        </span></td></tr>
    <tr><td>Fix</td><td>{html.escape(mode['fix'])}</td></tr>
  </table>
  {err_row}
  <div style="margin-top:10px">{trace_link}</div>
</div>
""".strip()
    return card, ns, honest, reported


def build_bestiary_index() -> str:
    """Assemble the full HTML for presentation/bestiary.html."""
    # --- collect cards + progression data in one pass ---
    card_htmls: list[str] = []
    chart_points: list[tuple[str, float | None, float | None]] = []
    for mode in BESTIARY_MODES:
        card_html, ns, honest, reported = _bestiary_card(mode)
        card_htmls.append(card_html)
        # x-axis label: 'v3.6', 'v3.11', etc.
        m = re.search(r"v(\d+_\d+)", ns or mode["namespace_prefix"])
        version_label = (
            "v" + m.group(1).replace("_", ".")
            if m else mode["namespace_prefix"].rstrip("_")
        )
        chart_points.append((version_label, honest, reported))

    contract_text = _extract_prompt_contract_text()
    loop_svg = _svg_loop_diagram()
    chart_svg = _svg_progression_chart(chart_points)

    # footer — list every existing per-namespace trace viewer
    trace_files = sorted(PRESENTATION_ROOT.glob("trace_viewer_*.html"))
    footer_links = "".join(
        f'<li><a href="{p.name}">{html.escape(p.name)}</a></li>'
        for p in trace_files
    ) or '<li style="color:#71717a">(no per-namespace traces built yet)</li>'

    return f"""<!DOCTYPE html>
<html lang="en"><head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>Bestiary of LLM World-Model Cheating Modes</title>
<style>
  body {{
    background:#18181b;color:#fafafa;
    font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
    margin:0;padding:24px 32px 96px;max-width:1120px;
  }}
  h1 {{ margin:0 0 6px;font-size:26px }}
  h2 {{ margin:32px 0 12px;font-size:18px;color:#a1a1aa;
        border-bottom:1px solid #3f3f46;padding-bottom:6px }}
  .banner {{
    background:linear-gradient(90deg,#166534 0%,#7f1d1d 100%);
    padding:18px 20px;border-radius:10px;margin-bottom:18px;
  }}
  .banner small {{ color:#e4e4e7 }}
  details {{ background:#27272a;border-radius:8px;padding:10px 14px;margin:12px 0 }}
  details summary {{ cursor:pointer;color:#22c55e;font-family:monospace }}
  details pre {{
    background:#000;color:#22c55e;padding:12px;border-radius:6px;
    overflow-x:auto;font-size:12px;line-height:1.5;
    border:1px solid #166534;margin:10px 0 0
  }}
  .grid {{ display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:14px }}
  .card {{
    background:#27272a;border-radius:8px;padding:14px 16px;
    border-left:4px solid #ef4444;
  }}
  .card.pending {{ border-left-color:#71717a }}
  .card-title {{ font-size:15px;font-weight:bold;margin-bottom:4px }}
  .card-ns {{ font-family:monospace;font-size:11px;color:#a1a1aa;margin-bottom:10px }}
  .card-table {{ width:100%;border-collapse:collapse;font-size:12px }}
  .card-table td {{ padding:4px 6px;vertical-align:top;border-bottom:1px solid #3f3f46 }}
  .card-table td:first-child {{ color:#a1a1aa;width:38% }}
  .badge {{
    font-size:10px;padding:2px 8px;border-radius:10px;font-weight:bold;
    margin-left:6px;font-family:monospace;text-transform:uppercase
  }}
  .badge-cheat {{ background:#7f1d1d;color:#ef4444 }}
  .badge-live {{ background:#166534;color:#22c55e }}
  .badge-pending {{ background:#3f3f46;color:#a1a1aa }}
  .trace-link {{ color:#22c55e;text-decoration:none;font-family:monospace }}
  .trace-link:hover {{ text-decoration:underline }}
  ul {{ line-height:1.8 }}
  ul a {{ color:#22c55e;text-decoration:none;font-family:monospace;font-size:13px }}
  ul a:hover {{ text-decoration:underline }}
  code {{ font-family:ui-monospace,'SF Mono',monospace;font-size:12px }}
</style></head><body>
  <div class="banner">
    <h1>Bestiary of LLM World-Model Cheating Modes</h1>
    <small>ls20 autoresearch — agents that game the transition-accuracy metric
    instead of learning the world. One card per failure mode, with the fix
    that closed it.</small>
  </div>

  <h2>1. The world-coder contract</h2>
  <details>
    <summary>▸ Show the live CWM contract prompt (from <code>world_model.py::prompt_overlay</code>)</summary>
    <pre>{html.escape(contract_text)}</pre>
  </details>

  <h2>2. The loop</h2>
  <div style="color:#a1a1aa;font-size:13px;margin-bottom:8px">
    IMAGINE (MCTS over <code>predict_effect</code>) → ACT (real env) →
    WAKE (rewrite on surprise) → SLEEP (DreamCoder refactor). The arrows are
    labelled with the function that actually fires the transition.
  </div>
  {loop_svg}

  <h2>3. The bestiary</h2>
  <div class="grid">
    {"".join(card_htmls)}
  </div>

  <h2>4. Progression — honest transition_accuracy across versions</h2>
  <div style="color:#a1a1aa;font-size:13px;margin-bottom:8px">
    Each run's <code>world_drafts[0].body</code> re-scored against its own
    <code>unit_tests</code> using <code>WorldModelModule._unit_test_matches</code>
    (v3.15 strict + identity-guard matcher). Dashed grey = what the run itself
    reported; solid green = what the current matcher accepts.
  </div>
  {chart_svg}

  <h2>5. Per-namespace trace viewers</h2>
  <ul>
    {footer_links}
  </ul>

  <div style="color:#52525b;font-size:11px;margin-top:40px">
    Generated by <code>scripts/build_trace_viewer.py --bestiary</code>.
    Source: <code>research_logs/shared/ls20-cb3b57cc/&lt;ns&gt;/world_model.json</code>.
  </div>
</body></html>"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("namespace", nargs="?", help="Shared namespace (e.g. arcgentica_v3_full_...)")
    parser.add_argument("--shared-dir", help="Explicit shared dir (overrides namespace lookup)")
    parser.add_argument("--game-prefix", default="ls20-cb3b57cc")
    parser.add_argument("--output", help="Output HTML path (default: presentation/trace_viewer_<ns>.html)")
    parser.add_argument(
        "--bestiary",
        action="store_true",
        help="Build presentation/bestiary.html (multi-namespace index) and exit",
    )
    args = parser.parse_args()

    # --- bestiary index mode (additive; no effect on per-ns rendering) ---
    if args.bestiary:
        out_path = Path(args.output) if args.output else (PRESENTATION_ROOT / "bestiary.html")
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(build_bestiary_index(), encoding="utf-8")
        print(f"wrote {out_path}")
        return 0

    if args.shared_dir:
        shared_dir = Path(args.shared_dir)
        namespace = shared_dir.name
    elif args.namespace:
        shared_dir = REPO_ROOT / "research_logs" / "shared" / args.game_prefix / args.namespace
        namespace = args.namespace
    else:
        print("Provide a namespace or --shared-dir", file=sys.stderr)
        return 2

    if not shared_dir.exists():
        print(f"Shared dir not found: {shared_dir}", file=sys.stderr)
        return 2

    world_model = load_json(shared_dir / "world_model.json")
    dreamcoder = load_json(shared_dir / "dreamcoder_library.json") or []
    meta = load_json(shared_dir / "meta_harness.json")
    planner = load_json(shared_dir / "planner_state.json")
    memories = load_json(shared_dir / "memories.json") or []

    recording_path = find_recording_for_namespace(namespace, args.game_prefix)
    recording = load_recording(recording_path) if recording_path else []

    html_str = build_trace_html(
        namespace=namespace,
        world_model=world_model,
        skills=dreamcoder if isinstance(dreamcoder, list) else [],
        meta_harness=meta,
        planner_state=planner,
        memories=memories,
        recording=recording,
    )

    output = Path(args.output) if args.output else (REPO_ROOT / "presentation" / f"trace_viewer_{namespace}.html")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(html_str, encoding="utf-8")
    print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
