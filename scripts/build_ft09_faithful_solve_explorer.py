#!/usr/bin/env python3
"""Faithful-agent step-by-step SOLVE explorer (codex-recommended, NOT the
v50 module runtime-story builder).

For the BYTE-FAITHFUL original Symbolica agent there are NO M1-M4 modules.
The only honest sources are:
  * recorder jsonl  -> AUTHORITATIVE per-action spine: action_name,
    level_before/after, cleared_a_level (no coords, no reasoning).
  * proxy.jsonl     -> per-turn: plaintext assistant message text (the
    readable "why"; the raw chain-of-thought is PROVIDER-ENCRYPTED and
    unreadable), and the executed ``submit_action("ACTIONx", x=, y=)``
    calls (click coords).

Alignment (codex pitfall #4): the recorder is the spine. Executed
``submit_action`` literals are extracted IN ORDER from proxy tool-call
code and matched positionally to recorder actions; any count/name
mismatch is shown explicitly, never silently zipped. Encrypted reasoning
and unparseable (loop/variable) submit calls are labelled, never faked.

Usage:
  build_ft09_faithful_solve_explorer.py RECORDER.jsonl PROXY.jsonl OUT.html
"""
from __future__ import annotations

import html
import json
import re
import sys
from pathlib import Path

SUBMIT_RE = re.compile(
    r'submit_action\(\s*["\']?(ACTION\d|RESET)["\']?'
    r'(?:[^)]*?x\s*=\s*(\d+))?(?:[^)]*?y\s*=\s*(\d+))?',
    re.S,
)


def _jsonl(p: Path) -> list[dict]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def _epoch(ts):
    try:
        return float(ts)
    except Exception:
        return 0.0


def main() -> int:
    if len(sys.argv) != 4:
        print(__doc__)
        return 2
    rec_p, px_p, out_p = (Path(a) for a in sys.argv[1:])
    rec = _jsonl(rec_p)
    ev = _jsonl(px_p)

    # ---- authoritative spine ----
    spine = [
        {
            "i": r.get("action_index"),
            "act": str(r.get("action_name")),
            "lb": int(r.get("level_before_action") or 0),
            "la": int(r.get("level_after_action") or 0),
            "clr": bool(r.get("cleared_a_level")),
        }
        for r in rec
    ]

    # ---- proxy: ordered executed submit_action literals + readable msgs ----
    sc = sorted(
        (e for e in ev if e.get("event") == "stream_complete"),
        key=lambda e: _epoch(e.get("ts")),
    )
    executed: list[dict] = []  # ordered {act,x,y} from tool-call code
    msgs: list[str] = []  # ordered plaintext assistant message excerpts
    enc_reasoning = 0
    seen_code: set[str] = set()
    for e in sc:
        for it in e.get("output_items", []) or []:
            t = it.get("type")
            if t == "reasoning" and it.get("encrypted_content"):
                enc_reasoning += 1
            if t == "message":
                c = it.get("content")
                txt = ""
                if isinstance(c, list):
                    txt = " ".join(
                        seg.get("text", "")
                        for seg in c
                        if isinstance(seg, dict)
                    )
                elif isinstance(c, str):
                    txt = c
                txt = txt.strip()
                if len(txt) > 8:
                    msgs.append(txt)
            if t == "custom_tool_call":
                code = str(it.get("input", ""))
                key = code[:120]
                if key in seen_code:
                    continue
                seen_code.add(key)
                for m in SUBMIT_RE.finditer(code):
                    executed.append(
                        {
                            "act": m.group(1),
                            "x": m.group(2),
                            "y": m.group(3),
                        }
                    )

    # ---- positional alignment (recorder spine vs executed literals) ----
    n_rec = len(spine)
    n_exe = len(executed)
    aligned = []
    for k, s in enumerate(spine):
        ex = executed[k] if k < n_exe else None
        name_ok = bool(ex) and ex["act"] == s["act"]
        aligned.append({**s, "ex": ex, "name_ok": name_ok})
    n_name_ok = sum(1 for a in aligned if a["name_ok"])

    # ---- group by level, render ----
    levels: dict[int, list] = {}
    for a in aligned:
        levels.setdefault(a["lb"], []).append(a)
    maxlb = max(levels) if levels else 0

    blocks = []
    for L in range(0, maxlb + 1):
        seg = levels.get(L, [])
        if not seg:
            continue
        rows = []
        for a in seg:
            ex = a["ex"]
            if ex and ex["x"] and ex["y"]:
                coord = f'({ex["x"]},{ex["y"]})'
            elif ex:
                coord = "(no static coords — loop/var code)"
            else:
                coord = "(no matched submit literal)"
            badge = (
                '<span class="ok">name✓</span>'
                if a["name_ok"]
                else '<span class="warn">name≠ / unmatched</span>'
            )
            clr = (
                f' <b class="grn">→ L{a["la"]} CLEARED</b>'
                if a["clr"]
                else ""
            )
            rows.append(
                f'<tr><td>#{a["i"]}</td><td><b>{html.escape(a["act"])}</b></td>'
                f'<td>{html.escape(coord)}</td><td>{badge}</td>'
                f'<td>L{a["lb"]}{clr}</td></tr>'
            )
        clr_row = next((a for a in seg if a["clr"]), None)
        head = (
            f'L{L} &rarr; L{clr_row["la"]} '
            f'(cleared by action #{clr_row["i"]})'
            if clr_row
            else f'L{L} (no recorded clear in-loop)'
        )
        blocks.append(
            f'<h2>{head} &mdash; {len(seg)} actions</h2>'
            f'<table><thead><tr><th>idx</th><th>action</th>'
            f'<th>click coords (proxy)</th><th>spine-match</th>'
            f'<th>level</th></tr></thead><tbody>'
            f'{"".join(rows)}</tbody></table>'
        )

    # readable assistant-message excerpts (the recoverable "why")
    why = "".join(
        f'<div class="msg">{html.escape(m[:600])}</div>' for m in msgs[:40]
    )

    banner = (
        f'recorder spine = {n_rec} actions (authoritative). '
        f'executed submit_action literals parsed from proxy = {n_exe}; '
        f'positional name-match = {n_name_ok}/{n_rec}. '
        f'reasoning items: {enc_reasoning} PROVIDER-ENCRYPTED (raw CoT '
        f'unreadable) — only {len(msgs)} plaintext assistant messages are '
        f'recoverable. Coords shown only where a literal '
        f'submit_action("ACTIONx", x=, y=) was statically parseable; '
        f'loop/variable-driven submits are labelled, never fabricated.'
    )

    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8">
<title>FT09 faithful solve — s47 step-by-step (recorder spine + proxy)</title>
<style>
body{{background:#0d1117;color:#c9d1d9;font:13px/1.5 ui-monospace,Menlo,monospace;
margin:0}}.w{{max-width:1080px;margin:0 auto;padding:22px}}
h1{{font-size:18px;margin:0 0 6px}}h2{{font-size:13px;color:#388bfd;
border-bottom:1px solid #30363d;padding-bottom:4px;margin:20px 0 8px}}
.ban{{background:#161b22;border:1px solid #d29922;border-radius:6px;
padding:10px;color:#e3b341;font-size:12px;margin-bottom:14px}}
table{{border-collapse:collapse;width:100%;font-size:12px;margin-bottom:6px}}
th,td{{border:1px solid #30363d;padding:4px 8px;text-align:left}}
th{{background:#1c2330;color:#8b949e}}
.grn{{color:#3fb950}}.ok{{color:#3fb950}}.warn{{color:#f85149}}
.msg{{background:#0a0e14;border:1px solid #30363d;border-radius:5px;
padding:8px;margin:6px 0;white-space:pre-wrap;color:#9db4d0;font-size:11.5px}}
</style></head><body><div class="w">
<h1>FT09 &mdash; how s47 (byte-faithful Symbolica + trace2skill) actually
reached L6, step by step</h1>
<div class="ban">{banner}</div>
{''.join(blocks)}
<h2>Recoverable assistant-message excerpts (the readable "why" — full
chain-of-thought is provider-encrypted)</h2>
{why or '<div class="msg">(no plaintext assistant messages recovered)</div>'}
</div></body></html>"""
    out_p.write_text(doc)
    print(
        f"WROTE {out_p} ({out_p.stat().st_size}B)  "
        f"spine={n_rec} executed={n_exe} name_ok={n_name_ok} "
        f"enc_reasoning={enc_reasoning} plaintext_msgs={len(msgs)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
